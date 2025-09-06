"""
Constraint definitions for the optimization model.
"""

from ortools.sat.python import cp_model
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TrainSchedulingConstraints:
    """
    Defines constraints for train scheduling optimization problem.
    """

    def __init__(self, model: cp_model.CpModel):
        self.model = model
        self.variables = {}
        self.intervals = {}

    def add_track_conflict_constraints(self, trains: List[Dict], sections: List[Dict]) -> None:
        """
        Add improved constraints to prevent two trains from occupying the same track section simultaneously.
        Now properly handles section occupancy vs station occupancy.

        Args:
            trains: List of train data
            sections: List of section data with single_track information
        """
        logger.info("Adding improved track conflict constraints")

        # Group trains by section
        section_trains = {}
        for train in trains:
            section_id = train['section_id']
            if section_id not in section_trains:
                section_trains[section_id] = []
            section_trains[section_id].append(train)

        # Add constraints for each section
        for section_id, section_trains_list in section_trains.items():
            if len(section_trains_list) <= 1:
                continue

            # Find section data
            section_data = next((s for s in sections if s['section_id'] == section_id), None)
            if not section_data:
                # Default to single track behavior if no section data
                logger.warning(f"No section data found for {section_id}, assuming single track")
                section_data = {'single_track': True}

            is_single_track = section_data.get('single_track', True)

            if is_single_track:
                # For single track sections, trains cannot overlap while in the section
                intervals = []
                for train in section_trains_list:
                    train_id = train['train_id']
                    # Use section_intervals (departure -> departure + travel_time)
                    if hasattr(self, 'section_intervals') and train_id in getattr(self, 'section_intervals', {}):
                        intervals.append(self.section_intervals[train_id])
                    elif train_id in self.intervals:
                        # Fallback to station interval if section interval not available
                        intervals.append(self.intervals[train_id])

                if len(intervals) > 1:
                    self.model.AddNoOverlap(intervals)
                    logger.debug(f"Added no-overlap constraint for single-track section {section_id} with {len(intervals)} trains")
            else:
                # Multi-track section - trains can operate simultaneously
                # Only platform conflicts matter (handled separately)
                logger.debug(f"Multi-track section {section_id} - no track overlap constraints needed")

    def add_priority_constraints(self, trains: List[Dict]) -> None:
        """
        Add constraints to respect train priorities (Express > Passenger > Freight).
        Only applied when trains have different priorities in the same section.

        Args:
            trains: List of train data with priorities
        """
        logger.info("Adding priority constraints")

        # Group trains by section
        section_trains = {}
        for train in trains:
            section_id = train['section_id']
            if section_id not in section_trains:
                section_trains[section_id] = []
            section_trains[section_id].append(train)

        # Apply priority constraints within each section
        for section_id, section_trains_list in section_trains.items():
            # Sort trains by priority (higher number = higher priority)
            sorted_trains = sorted(section_trains_list, key=lambda x: x['priority'], reverse=True)

            for i, high_priority_train in enumerate(sorted_trains[:-1]):
                for low_priority_train in sorted_trains[i+1:]:
                    # Only enforce if priorities are actually different
                    if high_priority_train['priority'] > low_priority_train['priority']:
                        high_train_id = high_priority_train['train_id']
                        low_train_id = low_priority_train['train_id']

                        if high_train_id in self.variables and low_train_id in self.variables:
                            # High priority train should start before or at same time as low priority
                            self.model.Add(
                                self.variables[high_train_id]['start'] <=
                                self.variables[low_train_id]['start']
                            )
                            logger.debug(f"Added priority constraint: {high_train_id} (priority {high_priority_train['priority']}) before {low_train_id} (priority {low_priority_train['priority']})")

    def add_platform_availability_constraints(self, trains: List[Dict], platforms: List[Dict]) -> None:
        """
        Add constraints for platform availability and capacity.

        Args:
            trains: List of train data
            platforms: List of platform data
        """
        logger.info("Adding platform availability constraints")

        # Group trains by platform need
        platform_trains = {}
        for train in trains:
            platform_need = train['platform_need']
            if platform_need not in platform_trains:
                platform_trains[platform_need] = []
            platform_trains[platform_need].append(train)

        # Add capacity constraints for each platform
        for platform_id, platform_trains_list in platform_trains.items():
            # Find platform data
            platform_data = next((p for p in platforms if p['platform_id'] == platform_id), None)
            if not platform_data:
                continue

            platform_capacity = platform_data.get('capacity', 1)

            # If more trains than capacity, add scheduling constraints
            if len(platform_trains_list) > platform_capacity:
                intervals = []
                for train in platform_trains_list:
                    train_id = train['train_id']
                    if train_id in self.intervals:
                        intervals.append(self.intervals[train_id])

                # Add constraint that at most 'capacity' trains can use platform simultaneously
                if intervals and platform_capacity < len(intervals):
                    self.model.AddNoOverlap(intervals)
                    logger.debug(f"Added platform no-overlap constraint for {platform_id} with {len(intervals)} trains")

    def add_timing_constraints(self, trains: List[Dict], time_buffer_minutes: int = 5) -> None:
        """
        Add timing constraints like minimum separation between trains.

        Args:
            trains: List of train data
            time_buffer_minutes: Minimum time buffer between consecutive trains
        """
        logger.info(f"Adding timing constraints with {time_buffer_minutes} minute buffer")

        for train in trains:
            train_id = train['train_id']
            if train_id not in self.variables:
                continue

            # Ensure departure is after arrival
            arrival_var = self.variables[train_id]['start']
            departure_var = self.variables[train_id]['end']

            # Minimum dwell time constraint
            min_dwell_minutes = 2  # Minimum 2 minutes dwell time
            self.model.Add(departure_var >= arrival_var + min_dwell_minutes)

            # Maximum dwell time constraint (optional)
            max_dwell_minutes = 30
            self.model.Add(departure_var <= arrival_var + max_dwell_minutes)

            logger.debug(f"Added timing constraints for {train_id}: dwell time between {min_dwell_minutes}-{max_dwell_minutes} minutes")

    def add_maintenance_window_constraints(self, trains: List[Dict], sections: List[Dict]) -> None:
        """
        Add constraints to avoid scheduled maintenance windows.

        Args:
            trains: List of train data
            sections: List of section data with maintenance windows
        """
        logger.info("Adding maintenance window constraints")

        maintenance_constraints_added = 0

        for section in sections:
            section_id = section['section_id']
            maintenance_windows = section.get('maintenance_windows', [])

            if not maintenance_windows:
                continue

            for train in trains:
                if train['section_id'] != section_id:
                    continue

                train_id = train['train_id']
                if train_id not in self.intervals:
                    continue

                # Ensure train doesn't conflict with maintenance windows
                for start_maint, end_maint in maintenance_windows:
                    # Convert to minutes for constraint
                    start_maint_min = int(start_maint.timestamp() // 60)
                    end_maint_min = int(end_maint.timestamp() // 60)

                    # Train must finish before maintenance or start after
                    train_start = self.variables[train_id]['start']
                    train_end = self.variables[train_id]['end']

                    # Create boolean variables for before/after maintenance
                    before_maint = self.model.NewBoolVar(f'{train_id}_before_maint_{start_maint_min}')
                    after_maint = self.model.NewBoolVar(f'{train_id}_after_maint_{start_maint_min}')

                    # Exactly one must be true
                    self.model.Add(before_maint + after_maint == 1)

                    # If before maintenance, train must end before it starts
                    self.model.Add(train_end <= start_maint_min).OnlyEnforceIf(before_maint)

                    # If after maintenance, train must start after it ends
                    self.model.Add(train_start >= end_maint_min).OnlyEnforceIf(after_maint)

                    maintenance_constraints_added += 1

        logger.info(f"Added {maintenance_constraints_added} maintenance window constraints")

    def add_section_capacity_constraints(self, trains: List[Dict], sections: List[Dict]) -> None:
        """
        Add constraints for sections with limited capacity (multiple tracks).

        Args:
            trains: List of train data
            sections: List of section data with capacity information
        """
        logger.info("Adding section capacity constraints")

        for section in sections:
            section_id = section['section_id']
            section_capacity = section.get('capacity', 1)

            if section_capacity <= 1:
                # Single track already handled by track conflict constraints
                continue

            # Find trains using this section
            section_trains = [train for train in trains if train['section_id'] == section_id]

            if len(section_trains) <= section_capacity:
                # No capacity constraint needed
                continue

            # Group trains into time slots to respect capacity
            intervals = []
            for train in section_trains:
                train_id = train['train_id']
                if train_id in self.intervals:
                    intervals.append(self.intervals[train_id])

            if len(intervals) > section_capacity:
                # Use cumulative constraint for capacity limitation
                # This is more complex and may need custom implementation
                logger.warning(f"Section {section_id} capacity constraint not fully implemented for capacity > 1")

    def add_buffer_constraints(self, trains: List[Dict], minimum_separation_minutes: int = 3) -> None:
        """
        Add minimum separation time between consecutive trains in the same section.

        Args:
            trains: List of train data
            minimum_separation_minutes: Minimum time gap between trains
        """
        logger.info(f"Adding buffer constraints with {minimum_separation_minutes} minute minimum separation")

        # Group trains by section
        section_trains = {}
        for train in trains:
            section_id = train['section_id']
            if section_id not in section_trains:
                section_trains[section_id] = []
            section_trains[section_id].append(train)

        buffer_constraints_added = 0

        for section_id, section_trains_list in section_trains.items():
            if len(section_trains_list) <= 1:
                continue

            # Sort trains by original arrival time
            sorted_trains = sorted(section_trains_list, key=lambda x: x['arrival_time'])

            for i in range(len(sorted_trains) - 1):
                train1 = sorted_trains[i]
                train2 = sorted_trains[i + 1]

                train1_id = train1['train_id']
                train2_id = train2['train_id']

                if train1_id in self.variables and train2_id in self.variables:
                    # Ensure minimum separation between train1 end and train2 start
                    self.model.Add(
                        self.variables[train2_id]['start'] >= 
                        self.variables[train1_id]['end'] + minimum_separation_minutes
                    )
                    buffer_constraints_added += 1

        logger.info(f"Added {buffer_constraints_added} buffer constraints")