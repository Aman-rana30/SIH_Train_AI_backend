"""
Main optimization engine using Google OR-Tools CP-SAT solver.
"""
from ortools.sat.python import cp_model
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
import logging
import uuid

from app.services.optimization.models import (
    TrainData, OptimizationInput, OptimizationOutput, DisruptionEvent
)
from app.services.optimization.constraints import TrainSchedulingConstraints

logger = logging.getLogger(__name__)


class TrainSchedulingOptimizer:
    """
    Main optimizer class for train scheduling using constraint programming.

    Uses Google OR-Tools CP-SAT solver to optimize train schedules while respecting
    constraints like track conflicts, priorities, and platform availability.
    """

    def __init__(self, time_limit_seconds: int = 30):
        self.time_limit_seconds = time_limit_seconds
        self.model = None
        self.solver = None
        self.variables = {}
        self.intervals = {}

    def optimize_schedule(self, optimization_input: OptimizationInput) -> OptimizationOutput:
        """
        Main optimization method that schedules trains to minimize delays and conflicts.

        Args:
            optimization_input: Complete optimization input data

        Returns:
            OptimizationOutput with optimized schedules and metrics
        """
        logger.info(f"Starting optimization for {len(optimization_input.trains)} trains")
        start_time = time.time()

        # Initialize the CP-SAT model
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = self.time_limit_seconds

        try:
            # Create decision variables
            self._create_variables(optimization_input.trains, optimization_input.time_horizon)

            # Add constraints
            self._add_constraints(optimization_input)

            # Set objective function
            self._set_objective(optimization_input.trains)

            # Solve the model
            status = self.solver.Solve(self.model)

            # Process results
            computation_time = time.time() - start_time
            return self._process_results(
                optimization_input.trains, status, computation_time
            )

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return self._create_fallback_solution(optimization_input.trains, time.time() - start_time)

    def what_if_analysis(self, 
                        current_schedule: List[Dict],
                        disruption: DisruptionEvent) -> OptimizationOutput:
        """
        Perform what-if analysis for disruption scenarios.

        Args:
            current_schedule: Current optimized schedule
            disruption: Disruption event to analyze

        Returns:
            Updated optimization output accounting for disruption
        """
        logger.info(f"Performing what-if analysis for {disruption.event_type} disruption")

        # Modify trains based on disruption
        modified_trains = self._apply_disruption(current_schedule, disruption)

        # Create new optimization input
        time_horizon = (
            datetime.now(),
            datetime.now() + timedelta(hours=12)
        )

        optimization_input = OptimizationInput(
            trains=modified_trains,
            sections=[],  # Simplified for what-if
            platforms=[],
            time_horizon=time_horizon,
            constraints={}
        )

        # Run optimization with disruption
        return self.optimize_schedule(optimization_input)

    def _create_variables(self, trains: List[TrainData], time_horizon: Tuple[datetime, datetime]) -> None:
        """Create decision variables for each train."""
        logger.debug("Creating decision variables")

        start_time_min = int(time_horizon[0].timestamp() // 60)
        end_time_min = int(time_horizon[1].timestamp() // 60)

        for train in trains:
            train_id = train.train_id

            # Original scheduled times in minutes
            original_arrival_min = int(train.arrival_time.timestamp() // 60)
            original_departure_min = int(train.departure_time.timestamp() // 60)

            # Allow some flexibility around original times
            flex_window = 60  # 60 minutes flexibility

            arrival_start = max(start_time_min, original_arrival_min - flex_window)
            arrival_end = min(end_time_min, original_arrival_min + flex_window)

            departure_start = max(start_time_min, original_departure_min - flex_window) 
            departure_end = min(end_time_min, original_departure_min + flex_window)

            # Create start and end variables
            start_var = self.model.NewIntVar(
                arrival_start, arrival_end, f'{train_id}_start'
            )
            end_var = self.model.NewIntVar(
                departure_start, departure_end, f'{train_id}_end'
            )

            # Create interval variable for this train
            duration = original_departure_min - original_arrival_min
            interval_var = self.model.NewIntervalVar(
                start_var, duration, end_var, f'{train_id}_interval'
            )

            self.variables[train_id] = {
                'start': start_var,
                'end': end_var,
                'original_arrival': original_arrival_min,
                'original_departure': original_departure_min
            }
            self.intervals[train_id] = interval_var

    def _add_constraints(self, optimization_input: OptimizationInput) -> None:
        """Add all constraints to the model."""
        logger.debug("Adding constraints to model")

        constraints_handler = TrainSchedulingConstraints(self.model)
        constraints_handler.variables = self.variables
        constraints_handler.intervals = self.intervals

        # Convert train data to dict format for constraints
        trains_dict = []
        for train in optimization_input.trains:
            trains_dict.append({
                'train_id': train.train_id,
                'type': train.type,
                'section_id': train.section_id,
                'platform_need': train.platform_need,
                'priority': train.priority,
                'arrival_time': train.arrival_time,
                'departure_time': train.departure_time
            })

        sections_dict = []
        for section in optimization_input.sections:
            sections_dict.append({
                'section_id': section.section_id,
                'capacity': section.capacity,
                'maintenance_windows': section.maintenance_windows
            })

        platforms_dict = []
        for platform in optimization_input.platforms:
            platforms_dict.append({
                'platform_id': platform.platform_id,
                'capacity': platform.capacity
            })

        # Add constraint types
        constraints_handler.add_track_conflict_constraints(trains_dict, sections_dict)
        constraints_handler.add_priority_constraints(trains_dict)
        constraints_handler.add_platform_availability_constraints(trains_dict, platforms_dict)
        constraints_handler.add_timing_constraints(trains_dict)
        constraints_handler.add_maintenance_window_constraints(trains_dict, sections_dict)

    def _set_objective(self, trains: List[TrainData]) -> None:
        """Set the objective function to minimize total delay and maximize throughput."""
        logger.debug("Setting optimization objective")

        # Minimize total weighted delay
        delay_terms = []

        for train in trains:
            train_id = train.train_id
            if train_id not in self.variables:
                continue

            # Calculate delay from original schedule
            original_arrival = self.variables[train_id]['original_arrival']
            actual_start = self.variables[train_id]['start']

            # Weight delays by train priority (higher priority = higher weight)
            priority_weight = train.priority

            # Delay can be positive (late) or negative (early)
            delay_var = self.model.NewIntVar(-1000, 1000, f'{train_id}_delay')
            self.model.Add(delay_var == actual_start - original_arrival)

            # Penalize positive delays more than negative
            positive_delay = self.model.NewIntVar(0, 1000, f'{train_id}_pos_delay')
            self.model.AddMaxEquality(positive_delay, [delay_var, 0])

            # Add weighted delay term
            delay_terms.append(positive_delay * priority_weight)

        if delay_terms:
            total_delay = sum(delay_terms)
            self.model.Minimize(total_delay)

    def _process_results(self, 
                        trains: List[TrainData], 
                        status: int, 
                        computation_time: float) -> OptimizationOutput:
        """Process optimization results and create output."""
        logger.debug(f"Processing results with status: {status}")

        schedules = []
        total_delay = 0.0
        conflicts_resolved = 0

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            for train in trains:
                train_id = train.train_id
                if train_id not in self.variables:
                    continue

                # Get optimized times
                optimized_start = self.solver.Value(self.variables[train_id]['start'])
                optimized_end = self.solver.Value(self.variables[train_id]['end'])

                # Convert back to datetime
                optimized_arrival = datetime.fromtimestamp(optimized_start * 60)
                optimized_departure = datetime.fromtimestamp(optimized_end * 60)

                # Calculate delay
                original_arrival_min = self.variables[train_id]['original_arrival']
                delay_minutes = optimized_start - original_arrival_min
                total_delay += max(0, delay_minutes)  # Only count positive delays

                schedule = {
                    'train_id': train_id,
                    'original_arrival': train.arrival_time,
                    'original_departure': train.departure_time,
                    'optimized_arrival': optimized_arrival,
                    'optimized_departure': optimized_departure,
                    'delay_minutes': delay_minutes,
                    'section_id': train.section_id,
                    'platform_need': train.platform_need,
                    'priority': train.priority
                }
                schedules.append(schedule)

                if delay_minutes != 0:
                    conflicts_resolved += 1

            objective_value = self.solver.ObjectiveValue() if status == cp_model.OPTIMAL else -1
            status_str = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
        else:
            # Fallback to priority-based heuristic
            logger.warning("Optimization failed, using fallback heuristic")
            schedules = self._priority_fallback_heuristic(trains)
            objective_value = -1
            status_str = "FALLBACK_HEURISTIC"
            conflicts_resolved = len([s for s in schedules if s['delay_minutes'] > 0])

        # Calculate metrics
        metrics = {
            'average_delay': total_delay / len(schedules) if schedules else 0,
            'max_delay': max([s['delay_minutes'] for s in schedules]) if schedules else 0,
            'on_time_percentage': len([s for s in schedules if s['delay_minutes'] <= 0]) / len(schedules) * 100 if schedules else 0,
            'throughput': len(schedules),
            'solver_status': status_str
        }

        return OptimizationOutput(
            schedules=schedules,
            objective_value=objective_value,
            computation_time=computation_time,
            status=status_str,
            metrics=metrics,
            conflicts_resolved=conflicts_resolved,
            total_delay=total_delay
        )

    def _priority_fallback_heuristic(self, trains: List[TrainData]) -> List[Dict]:
        """Fallback heuristic using priority-based scheduling."""
        logger.info("Using priority-based fallback heuristic")

        # Sort trains by priority and arrival time
        sorted_trains = sorted(
            trains, 
            key=lambda x: (x.priority, x.arrival_time), 
            reverse=True
        )

        schedules = []
        section_last_departure = {}

        for train in sorted_trains:
            # Check for conflicts with previous trains in same section
            delay_minutes = 0
            optimized_arrival = train.arrival_time
            optimized_departure = train.departure_time

            if train.section_id in section_last_departure:
                last_departure = section_last_departure[train.section_id]
                if optimized_arrival <= last_departure:
                    # Delay this train
                    delay_minutes = int((last_departure - optimized_arrival).total_seconds() / 60) + 5
                    optimized_arrival = last_departure + timedelta(minutes=5)
                    optimized_departure = optimized_arrival + (train.departure_time - train.arrival_time)

            section_last_departure[train.section_id] = optimized_departure

            schedule = {
                'train_id': train.train_id,
                'original_arrival': train.arrival_time,
                'original_departure': train.departure_time,
                'optimized_arrival': optimized_arrival,
                'optimized_departure': optimized_departure,
                'delay_minutes': delay_minutes,
                'section_id': train.section_id,
                'platform_need': train.platform_need,
                'priority': train.priority
            }
            schedules.append(schedule)

        return schedules

    def _create_fallback_solution(self, 
                                 trains: List[TrainData], 
                                 computation_time: float) -> OptimizationOutput:
        """Create a fallback solution when optimization fails."""
        schedules = self._priority_fallback_heuristic(trains)

        total_delay = sum(max(0, s['delay_minutes']) for s in schedules)
        metrics = {
            'average_delay': total_delay / len(schedules) if schedules else 0,
            'max_delay': max([s['delay_minutes'] for s in schedules]) if schedules else 0,
            'on_time_percentage': len([s for s in schedules if s['delay_minutes'] <= 0]) / len(schedules) * 100 if schedules else 0,
            'throughput': len(schedules),
            'solver_status': "FALLBACK"
        }

        return OptimizationOutput(
            schedules=schedules,
            objective_value=-1,
            computation_time=computation_time,
            status="FALLBACK",
            metrics=metrics,
            conflicts_resolved=len([s for s in schedules if s['delay_minutes'] > 0]),
            total_delay=total_delay
        )

    def _apply_disruption(self, 
                         current_schedule: List[Dict], 
                         disruption: DisruptionEvent) -> List[TrainData]:
        """Apply disruption to current schedule and return modified trains."""
        modified_trains = []

        for schedule in current_schedule:
            train_data = TrainData(
                train_id=schedule['train_id'],
                type="Express",  # Simplified for what-if
                arrival_time=schedule['optimized_arrival'],
                departure_time=schedule['optimized_departure'],
                section_id=schedule['section_id'],
                platform_need=schedule['platform_need'],
                priority=schedule['priority']
            )

            # Apply disruption if this train is affected
            if train_data.train_id in disruption.affected_trains:
                if disruption.event_type == "delay":
                    train_data.arrival_time += timedelta(minutes=disruption.delay_minutes)
                    train_data.departure_time += timedelta(minutes=disruption.delay_minutes)
                elif disruption.event_type == "cancellation":
                    continue  # Skip cancelled trains

            modified_trains.append(train_data)

        return modified_trains
