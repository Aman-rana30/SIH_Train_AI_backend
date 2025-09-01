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
        Add constraints to prevent two trains from occupying the same track section simultaneously.

        Args:
            trains: List of train data
            sections: List of section data
        """
        logger.info("Adding track conflict constraints")

        # Group trains by section
        section_trains = {}
        for train in trains:
            section_id = train['section_id']
            if section_id not in section_trains:
                section_trains[section_id] = []
            section_trains[section_id].append(train)

        # Add no-overlap constraints for trains in same section
        for section_id, section_trains_list in section_trains.items():
            if len(section_trains_list) > 1:
                intervals = []
                for train in section_trains_list:
                    train_id = train['train_id']
                    if train_id in self.intervals:
                        intervals.append(self.intervals[train_id])

                if intervals:
                    self.model.AddNoOverlap(intervals)
                    logger.debug(f"Added no-overlap constraint for section {section_id} with {len(intervals)} trains")

    def add_priority_constraints(self, trains: List[Dict]) -> None:
        """
        Add constraints to respect train priorities (Express > Passenger > Freight).

        Args:
            trains: List of train data with priorities
        """
        logger.info("Adding priority constraints")

        # Sort trains by priority (higher number = higher priority)
        sorted_trains = sorted(trains, key=lambda x: x['priority'], reverse=True)

        for i, high_priority_train in enumerate(sorted_trains[:-1]):
            for low_priority_train in sorted_trains[i+1:]:
                # If trains are in same section, high priority should go first
                if (high_priority_train['section_id'] == low_priority_train['section_id']):
                    high_train_id = high_priority_train['train_id']
                    low_train_id = low_priority_train['train_id']

                    if high_train_id in self.variables and low_train_id in self.variables:
                        # High priority train should start before or at same time as low priority
                        self.model.Add(
                            self.variables[high_train_id]['start'] <= 
                            self.variables[low_train_id]['start']
                        )

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

    def add_maintenance_window_constraints(self, trains: List[Dict], sections: List[Dict]) -> None:
        """
        Add constraints to avoid scheduled maintenance windows.

        Args:
            trains: List of train data
            sections: List of section data with maintenance windows
        """
        logger.info("Adding maintenance window constraints")

        for section in sections:
            section_id = section['section_id']
            maintenance_windows = section.get('maintenance_windows', [])

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
                    before_maint = self.model.NewBoolVar(f'{train_id}_before_maint')
                    after_maint = self.model.NewBoolVar(f'{train_id}_after_maint')

                    # Exactly one must be true
                    self.model.Add(before_maint + after_maint == 1)

                    # If before maintenance, train must end before it starts
                    self.model.Add(train_end <= start_maint_min).OnlyEnforceIf(before_maint)

                    # If after maintenance, train must start after it ends  
                    self.model.Add(train_start >= end_maint_min).OnlyEnforceIf(after_maint)
