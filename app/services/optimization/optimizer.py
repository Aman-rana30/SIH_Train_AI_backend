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
    TrainData, OptimizationInput, OptimizationOutput, DisruptionEvent, SectionData
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
        # Station/platform occupancy intervals (arrival -> departure)
        self.intervals = {}
        # Section occupancy intervals (departure -> departure + travel_time)
        self.section_intervals = {}
        # Keep sections for post-processing
        self.sections: List[SectionData] = []

    def _calculate_travel_time(self, train: TrainData, sections: List[SectionData]) -> int:
        """Calculates the estimated total occupancy time for a train in a section (travel + dwell)."""
        try:
            section = next(s for s in sections if s.section_id == train.section_id)
            
            # Calculate actual travel time through the section
            travel_time_minutes = section.calculate_travel_time()
            
            # Calculate dwell time at station
            dwell_time_minutes = (train.departure_time - train.arrival_time).total_seconds() / 60
            
            # Total time = travel time + dwell time
            total_minutes = int(travel_time_minutes + dwell_time_minutes)
            
            logger.debug(f"Train {train.train_id} in section {section.section_id}: "
                        f"travel={travel_time_minutes}min, dwell={dwell_time_minutes}min, total={total_minutes}min")
            
            return max(1, total_minutes)
        except (StopIteration, AttributeError) as e:
            logger.warning(f"Section data not found for {train.section_id}, using fallback calculation")
            # Fallback if section data is missing or malformed
            return max(1, int((train.departure_time - train.arrival_time).total_seconds() / 60))

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
            # Create decision variables with section data
            self.sections = optimization_input.sections
            self._create_variables(optimization_input.trains, optimization_input.time_horizon, optimization_input.sections)

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

    def _create_variables(self, trains: List[TrainData], time_horizon: Tuple[datetime, datetime], sections: List[SectionData]) -> None:
        """Create decision variables for each train using realistic travel and dwell times.
        - Station interval: arrival -> departure (dwell)
        - Section interval: departure -> departure + travel_time
        """
        logger.debug("Creating decision variables with realistic travel and dwell times")

        start_time_min = int(time_horizon[0].timestamp() // 60)
        end_time_min = int(time_horizon[1].timestamp() // 60)

        for train in trains:
            train_id = train.train_id

            # Original scheduled times in minutes
            original_arrival_min = int(train.arrival_time.timestamp() // 60)
            original_departure_min = int(train.departure_time.timestamp() // 60)

            # Reduced flexibility for more realistic scheduling
            flex_window = 15  # 15 minutes flexibility instead of 60

            arrival_start = max(start_time_min, original_arrival_min - flex_window)
            arrival_end = min(end_time_min, original_arrival_min + flex_window)

            # Create arrival (start) variable
            start_var = self.model.NewIntVar(
                arrival_start, arrival_end, f'{train_id}_start'
            )

            # Dwell duration (station occupancy): scheduled dwell +/- bounded in timing constraints
            dwell_duration = max(1, int((train.departure_time - train.arrival_time).total_seconds() // 60))

            departure_start = max(start_time_min, original_departure_min - flex_window)
            departure_end = min(end_time_min, original_departure_min + flex_window)

            # Create departure (end) variable for station interval, anchored by dwell
            end_var = self.model.NewIntVar(
                arrival_start + dwell_duration,
                arrival_end + dwell_duration,
                f'{train_id}_end'
            )

            # Ensure departure respects original departure flexibility
            self.model.Add(end_var >= departure_start)
            self.model.Add(end_var <= departure_end)

            # Station/platform occupancy interval (arrival -> departure)
            station_interval = self.model.NewIntervalVar(
                start_var, dwell_duration, end_var, f'{train_id}_station_interval'
            )
            self.intervals[train_id] = station_interval

            # Section occupancy (departure -> departure + travel_time)
            travel_duration = self._calculate_travel_time(train, sections)
            section_end_var = self.model.NewIntVar(
                departure_start + travel_duration,
                departure_end + travel_duration,
                f'{train_id}_section_end'
            )
            section_interval = self.model.NewIntervalVar(
                end_var, travel_duration, section_end_var, f'{train_id}_section_interval'
            )
            self.section_intervals[train_id] = section_interval

            self.variables[train_id] = {
                'start': start_var,
                'end': end_var,
                'section_end': section_end_var,
                'original_arrival': original_arrival_min,
                'original_departure': original_departure_min
            }

            logger.debug(
                f"Created vars for {train_id}: dwell={dwell_duration}min, travel={travel_duration}min, "
                f"arrival_window=[{arrival_start}-{arrival_end}], departure_window=[{departure_start}-{departure_end}]"
            )

    def _add_constraints(self, optimization_input: OptimizationInput) -> None:
        """Add all constraints to the model."""
        logger.debug("Adding constraints to model")

        constraints_handler = TrainSchedulingConstraints(self.model)
        constraints_handler.variables = self.variables
        constraints_handler.intervals = self.intervals
        # Provide section occupancy intervals to constraints for proper no-overlap in sections
        setattr(constraints_handler, 'section_intervals', self.section_intervals)

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
                'maintenance_windows': section.maintenance_windows,
                'single_track': getattr(section, 'single_track', True)
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

        # Minimize total weighted departure delay (more relevant for section conflicts)
        delay_terms = []
        for train in trains:
            train_id = train.train_id
            if train_id not in self.variables:
                continue

            original_departure = self.variables[train_id]['original_departure']
            actual_departure = self.variables[train_id]['end']

            # Weight delays by train priority (higher priority = higher weight)
            priority_weight = train.priority

            # Delay can be positive (late) or negative (early)
            delay_var = self.model.NewIntVar(-1000, 1000, f'{train_id}_dep_delay')
            self.model.Add(delay_var == actual_departure - original_departure)

            # Penalize positive delays more than negative
            positive_delay = self.model.NewIntVar(0, 1000, f'{train_id}_dep_pos_delay')
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
            # 1) Read solver values
            raw = []
            for train in trains:
                train_id = train.train_id
                if train_id not in self.variables:
                    continue
                start_min = self.solver.Value(self.variables[train_id]['start'])
                end_min = self.solver.Value(self.variables[train_id]['end'])
                raw.append((train, start_min, end_min))

            # 2) Section-occupancy post-processing: enforce [dep, dep+travel]
            #    Priority: higher priority first; tie-breaker earlier arrival
            def travel_minutes(t: TrainData) -> int:
                try:
                    section = next(s for s in self.sections if s.section_id == t.section_id)
                    return max(1, section.calculate_travel_time())
                except StopIteration:
                    # fallback 37 min if section unknown
                    return 37

            # Prepare per-section buckets
            by_section: Dict[str, List[tuple[TrainData, int, int]]] = {}
            for t, smin, emin in raw:
                by_section.setdefault(t.section_id, []).append((t, smin, emin))

            adjusted: Dict[str, tuple[int, int]] = {}
            for section_id, items in by_section.items():
                # Sort by priority desc, then by arrival time asc
                items.sort(key=lambda x: (-x[0].priority, x[0].arrival_time))
                current_blocked_until = -10**9
                for t, smin, emin in items:
                    dep = emin  # departure minute
                    trav = travel_minutes(t)
                    const_buffer = 3  # minutes buffer after section clears
                    sec_clear = dep + trav
                    # If overlaps previous block, push departure to current_blocked_until
                    if dep < current_blocked_until:
                        dep = current_blocked_until
                        emin = dep  # end var is the departure
                        smin = dep - max(1, int((t.departure_time - t.arrival_time).total_seconds() // 60))
                        sec_clear = dep + trav
                    # Update rolling block end with buffer
                    current_blocked_until = max(current_blocked_until, sec_clear + const_buffer)
                    adjusted[t.train_id] = (smin, emin)

            # 3) Build schedules and compute departure-based delay
            for train in trains:
                train_id = train.train_id
                if train_id not in self.variables:
                    continue

                if train_id in adjusted:
                    optimized_start, optimized_end = adjusted[train_id]
                else:
                    optimized_start = self.solver.Value(self.variables[train_id]['start'])
                    optimized_end = self.solver.Value(self.variables[train_id]['end'])

                optimized_arrival = datetime.fromtimestamp(optimized_start * 60)
                optimized_departure = datetime.fromtimestamp(optimized_end * 60)

                original_departure_min = self.variables[train_id]['original_departure']
                delay_minutes = optimized_end - original_departure_min

                total_delay += max(0, delay_minutes)

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