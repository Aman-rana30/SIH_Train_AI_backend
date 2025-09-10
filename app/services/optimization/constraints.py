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
        Add SMART priority constraints that don't conflict with chronology.
        Only enforce priority when trains are close in time (within 2 hours) and in the same section.
        This prevents impossible constraints like forcing a 10 AM train to wait for 6 PM train.

        Args:
            trains: List of train data with priorities
        """
        logger.info("Adding smart priority constraints (time-aware)")

        # Group trains by section for section-specific priority rules
        section_trains = {}
        for train in trains:
            section_id = train['section_id']
            if section_id not in section_trains:
                section_trains[section_id] = []
            section_trains[section_id].append(train)

        constraints_added = 0
        for section_id, section_trains_list in section_trains.items():
            # Sort trains chronologically first
            sorted_trains = sorted(section_trains_list, key=lambda x: x['arrival_time'])

            for i, train1 in enumerate(sorted_trains):
                for j, train2 in enumerate(sorted_trains[i+1:], i+1):
                    # Only consider trains within 2 hours of each other
                    time_diff = abs((train2['arrival_time'] - train1['arrival_time']).total_seconds() / 3600)
                    
                    if time_diff <= 2.0:  # Within 2 hours
                        # Check if there's a significant priority difference (at least 2 levels)
                        if abs(train1['priority'] - train2['priority']) >= 2:
                            high_train = train1 if train1['priority'] > train2['priority'] else train2
                            low_train = train2 if train1['priority'] > train2['priority'] else train1

                            high_train_id = high_train['train_id']
                            low_train_id = low_train['train_id']

                            if high_train_id in self.variables and low_train_id in self.variables:
                                # SMART CONSTRAINT: Only enforce if high priority train comes first chronologically
                                # This prevents impossible time relationships
                                if high_train['arrival_time'] <= low_train['arrival_time']:
                                    # High priority train should start before or at same time as low priority
                                    self.model.Add(
                                        self.variables[high_train_id]['start'] <= 
                                        self.variables[low_train_id]['start']
                                    )
                                    constraints_added += 1
                                    logger.debug(f"Smart priority constraint: {high_train_id} (P{high_train['priority']}) before {low_train_id} (P{low_train['priority']}) - {time_diff:.1f}h apart")
                                else:
                                    # High priority comes later chronologically - use soft preference in objective instead
                                    logger.debug(f"Skipped priority constraint: {high_train_id} (P{high_train['priority']}) comes after {low_train_id} (P{low_train['priority']}) chronologically")
                    else:
                        logger.debug(f"Skipped priority constraint: trains {train1['train_id']} and {train2['train_id']} are {time_diff:.1f}h apart (>2h)")

        logger.info(f"Added {constraints_added} smart priority constraints (time-aware)")

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
        Made more flexible to reduce over-constraining.

        Args:
            trains: List of train data
            time_buffer_minutes: Minimum time buffer between consecutive trains
        """
        logger.info(f"Adding flexible timing constraints with {time_buffer_minutes} minute buffer")

        for train in trains:
            train_id = train['train_id']
            if train_id not in self.variables:
                continue

            # Ensure departure is after arrival
            arrival_var = self.variables[train_id]['start']
            departure_var = self.variables[train_id]['end']

            # Minimum dwell time constraint (more flexible)
            min_dwell_minutes = 1  # Reduced from 2 to 1 minute minimum
            self.model.Add(departure_var >= arrival_var + min_dwell_minutes)

            # Maximum dwell time constraint (more generous)
            max_dwell_minutes = 480  # Increased from 300 to 480 minutes (8 hours)
            self.model.Add(departure_var <= arrival_var + max_dwell_minutes)

            logger.debug(f"Added flexible timing constraints for {train_id}: dwell time between {min_dwell_minutes}-{max_dwell_minutes} minutes")

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
        Made more flexible to reduce solver failures.

        Args:
            trains: List of train data
            minimum_separation_minutes: Minimum time gap between trains
        """
        logger.info(f"Adding flexible buffer constraints with {minimum_separation_minutes} minute minimum separation")

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

                # Only add buffer constraint if trains are close in time (within 4 hours)
                time_diff = (train2['arrival_time'] - train1['arrival_time']).total_seconds() / 3600

                if time_diff <= 4.0:  # Only for trains within 4 hours
                    train1_id = train1['train_id']
                    train2_id = train2['train_id']

                    if train1_id in self.variables and train2_id in self.variables:
                        # Ensure minimum separation between train1 end and train2 start
                        self.model.Add(
                            self.variables[train2_id]['start'] >= 
                            self.variables[train1_id]['end'] + minimum_separation_minutes
                        )
                        buffer_constraints_added += 1
                        logger.debug(f"Added buffer constraint between {train1_id} and {train2_id} ({time_diff:.1f}h apart)")

        logger.info(f"Added {buffer_constraints_added} flexible buffer constraints")

    def add_soft_priority_preferences(self, trains: List[Dict]) -> List[Any]:
        """
        Add soft priority preferences that can be used in the objective function.
        These don't create hard constraints but influence the optimization goal.

        Args:
            trains: List of train data with priorities

        Returns:
            List of preference terms for the objective function
        """
        logger.info("Adding soft priority preferences for objective")

        preference_terms = []

        # Group trains by section
        section_trains = {}
        for train in trains:
            section_id = train['section_id']
            if section_id not in section_trains:
                section_trains[section_id] = []
            section_trains[section_id].append(train)

        for section_id, section_trains_list in section_trains.items():
            # Sort trains chronologically
            sorted_trains = sorted(section_trains_list, key=lambda x: x['arrival_time'])

            for i, train1 in enumerate(sorted_trains):
                for j, train2 in enumerate(sorted_trains[i+1:], i+1):
                    # Create soft preference for priority ordering
                    if train1['priority'] != train2['priority']:
                        high_train = train1 if train1['priority'] > train2['priority'] else train2
                        low_train = train2 if train1['priority'] > train2['priority'] else train1

                        high_train_id = high_train['train_id']
                        low_train_id = low_train['train_id']

                        if high_train_id in self.variables and low_train_id in self.variables:
                            # Create a preference variable for priority ordering
                            priority_bonus = self.model.NewBoolVar(f'priority_bonus_{high_train_id}_{low_train_id}')

                            # If high priority starts before low priority, give bonus
                            self.model.Add(
                                self.variables[high_train_id]['start'] <= self.variables[low_train_id]['start']
                            ).OnlyEnforceIf(priority_bonus)

                            # Add weighted bonus to objective (this doesn't constrain, just influences)
                            priority_weight = abs(high_train['priority'] - low_train['priority'])
                            preference_terms.append(priority_bonus * priority_weight * 5)

                            logger.debug(f"Added soft priority preference: {high_train_id} (P{high_train['priority']}) before {low_train_id} (P{low_train['priority']})")

        logger.info(f"Added {len(preference_terms)} soft priority preferences")
        return preference_terms

    def debug_constraint_conflicts(self, trains: List[Dict]) -> Dict[str, Any]:
        """
        Debug constraint conflicts and provide diagnostic information.

        Args:
            trains: List of train data

        Returns:
            Dictionary with diagnostic information
        """
        logger.info("Running constraint conflict diagnostics")

        diagnostics = {
            'total_trains': len(trains),
            'total_variables': len(self.variables),
            'total_intervals': len(self.intervals),
            'potential_conflicts': [],
            'time_spans': {},
            'priority_distribution': {}
        }

        # Analyze time spans
        if trains:
            min_time = min(train['arrival_time'] for train in trains)
            max_time = max(train['departure_time'] for train in trains)
            total_span_hours = (max_time - min_time).total_seconds() / 3600

            diagnostics['time_spans'] = {
                'min_time': min_time,
                'max_time': max_time,
                'total_span_hours': total_span_hours
            }

        # Analyze priority distribution
        priorities = [train['priority'] for train in trains]
        diagnostics['priority_distribution'] = {
            'min_priority': min(priorities) if priorities else 0,
            'max_priority': max(priorities) if priorities else 0,
            'unique_priorities': len(set(priorities)) if priorities else 0
        }

        # Find potential conflicts
        section_trains = {}
        for train in trains:
            section_id = train['section_id']
            if section_id not in section_trains:
                section_trains[section_id] = []
            section_trains[section_id].append(train)

        for section_id, section_trains_list in section_trains.items():
            if len(section_trains_list) > 1:
                # Sort by time
                sorted_trains = sorted(section_trains_list, key=lambda x: x['arrival_time'])

                for i in range(len(sorted_trains) - 1):
                    train1 = sorted_trains[i]
                    train2 = sorted_trains[i + 1]

                    gap_minutes = (train2['arrival_time'] - train1['departure_time']).total_seconds() / 60

                    if gap_minutes < 15:  # Less than 15 minutes gap
                        diagnostics['potential_conflicts'].append({
                            'section_id': section_id,
                            'train1': train1['train_id'],
                            'train2': train2['train_id'],
                            'gap_minutes': gap_minutes,
                            'priority_conflict': train1['priority'] < train2['priority']  # Lower priority before higher
                        })

        logger.info(f"Diagnostics: {diagnostics['total_trains']} trains, {len(diagnostics['potential_conflicts'])} potential conflicts")
        return diagnostics

    def validate_constraints(self, trains: List[Dict], sections: List[Dict]) -> bool:
        """
        Validate that constraints are reasonable and not over-constrained.

        Args:
            trains: List of train data
            sections: List of section data

        Returns:
            True if constraints seem reasonable, False if likely over-constrained
        """
        logger.info("Validating constraint reasonableness")

        # Check for basic issues
        if not trains:
            logger.warning("No trains provided")
            return False

        if not self.variables:
            logger.warning("No variables created")
            return False

        # Check time spans
        min_time = min(train['arrival_time'] for train in trains)
        max_time = max(train['departure_time'] for train in trains)
        time_span_hours = (max_time - min_time).total_seconds() / 3600

        if time_span_hours > 24:
            logger.warning(f"Large time span ({time_span_hours:.1f} hours) may cause solver issues")

        # Check for priority conflicts
        diagnostics = self.debug_constraint_conflicts(trains)
        if len(diagnostics['potential_conflicts']) > len(trains):
            logger.warning(f"High conflict ratio: {len(diagnostics['potential_conflicts'])} conflicts for {len(trains)} trains")
            return False

        logger.info("Constraint validation passed")
        return True