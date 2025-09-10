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

            # Allow trains to run on time with minimal flexibility
            flex_window = 12  # 5 minutes flexibility to allow on-time operation

            arrival_start = max(start_time_min, original_arrival_min - flex_window)
            arrival_end = min(end_time_min, original_arrival_min + flex_window)

            # Create arrival (start) variable
            start_var = self.model.NewIntVar(
                arrival_start, arrival_end, f'{train_id}_start'
            )

            # Dwell duration (station occupancy): scheduled dwell +/- bounded in timing constraints
            dwell_duration = max(0, int((train.departure_time - train.arrival_time).total_seconds() // 60))

            departure_start = max(start_time_min, original_departure_min - flex_window)
            departure_end = min(end_time_min, original_departure_min + flex_window)

            # Create departure (end) variable for station interval, anchored by dwell
            end_var = self.model.NewIntVar(
                departure_start,
                departure_end,
                f'{train_id}_end'
            )

            # Ensure minimum dwell time is respected
            self.model.Add(end_var >= start_var + dwell_duration)

            # Station/platform occupancy interval (arrival -> departure)
            # Use actual duration variable instead of fixed dwell_duration
            actual_dwell = self.model.NewIntVar(1, dwell_duration + 10, f'{train_id}_dwell')
            self.model.Add(actual_dwell == end_var - start_var)
            station_interval = self.model.NewIntervalVar(
                start_var, actual_dwell, end_var, f'{train_id}_station_interval'
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

            # Penalize positive delays more than negative, but also slightly penalize early departures
            positive_delay = self.model.NewIntVar(0, 1000, f'{train_id}_dep_pos_delay')
            negative_delay = self.model.NewIntVar(0, 1000, f'{train_id}_dep_neg_delay')
            
            self.model.AddMaxEquality(positive_delay, [delay_var, 0])
            self.model.AddMaxEquality(negative_delay, [-delay_var, 0])

            # Weight positive delays much more heavily than negative delays
            # This encourages on-time operation over early operation
            delay_terms.append(positive_delay * priority_weight * 10 + negative_delay * priority_weight)

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
                    
                    # Only adjust if there's an actual conflict
                    if dep < current_blocked_until:
                        # Calculate the minimum delay needed
                        original_dep = dep
                        dep = current_blocked_until
                        emin = dep  # end var is the departure
                        smin = dep - max(1, int((t.departure_time - t.arrival_time).total_seconds() // 60))
                        sec_clear = dep + trav
                        logger.debug(f"Train {t.train_id} delayed from {original_dep} to {dep} due to section conflict")
                    else:
                        # No conflict - train can run on time
                        logger.debug(f"Train {t.train_id} running on time at {dep}")
                    
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

                # Only count positive delays (late trains) for total delay
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

        """
    COMPLETE FIXED IMPLEMENTATION
    Replace the _priority_fallback_heuristic method in optimizer.py with this code
    """

    def _priority_fallback_heuristic(self, trains: List[TrainData]) -> List[Dict]:
        """
        Improved fallback heuristic using time-based scheduling with priority-based conflict resolution.
        
        Algorithm:
        1. Process trains in chronological order by departure time
        2. When conflicts occur (within 12-minute buffer), resolve using priority:
        - Higher priority trains keep their original schedule
        - Lower priority trains get delayed by the minimum needed time
        3. Same priority conflicts use FIFO (First In, First Out) rule
        
        This ensures higher priority trains are never delayed due to lower priority conflicts.
        """
        logger.info("Using improved time-based fallback heuristic with priority-based conflict resolution")

        # STEP 1: Sort trains by departure time (chronological processing)
        sorted_trains = sorted(trains, key=lambda x: x.departure_time)
        
        schedules = []
        DEPARTURE_BUFFER_MINUTES = 12
        
        logger.debug(f"Processing {len(sorted_trains)} trains in chronological order with {DEPARTURE_BUFFER_MINUTES}-minute buffer")

        for current_train in sorted_trains:
            # Start with original schedule
            optimized_arrival = current_train.arrival_time
            optimized_departure = current_train.departure_time
            current_delayed = False
            delay_reason = ""

            # STEP 2: Check for conflicts with all previously scheduled trains
            conflicts_resolved = 0
            for i, prev_schedule in enumerate(schedules):
                prev_departure = prev_schedule['optimized_departure']
                conflict_window_end = prev_departure + timedelta(minutes=DEPARTURE_BUFFER_MINUTES)
                
                # Check if current train conflicts with previous train's buffer window
                if optimized_departure < conflict_window_end:
                    conflict_duration = int((conflict_window_end - optimized_departure).total_seconds() / 60)
                    
                    logger.debug(f"CONFLICT: {current_train.train_id} (P{current_train.priority}) vs {prev_schedule['train_id']} (P{prev_schedule['priority']}) - {conflict_duration}min overlap")
                    
                    # STEP 3: Priority-based conflict resolution
                    if current_train.priority > prev_schedule['priority']:
                        # Current train has HIGHER priority - delay the previous train
                        original_prev_departure = prev_schedule['original_departure']
                        prev_delay_needed = conflict_window_end - original_prev_departure
                        prev_dwell = prev_schedule['optimized_arrival'] - prev_schedule['original_arrival']
                        
                        # Update previous train's schedule
                        schedules[i]['optimized_departure'] = conflict_window_end
                        schedules[i]['optimized_arrival'] = conflict_window_end - (original_prev_departure - prev_schedule['original_arrival'])
                        schedules[i]['delay_minutes'] = int(prev_delay_needed.total_seconds() / 60)
                        
                        logger.info(f"PRIORITY OVERRIDE: Delayed {prev_schedule['train_id']} (P{prev_schedule['priority']}) by {conflict_duration}min to accommodate higher priority {current_train.train_id} (P{current_train.priority})")
                        conflicts_resolved += 1
                        
                    elif current_train.priority < prev_schedule['priority']:
                        # Previous train has HIGHER priority - delay current train
                        delay_needed = conflict_window_end - optimized_departure
                        optimized_departure = conflict_window_end
                        optimized_arrival = optimized_arrival + delay_needed  # Maintain original dwell time
                        current_delayed = True
                        delay_reason = f"Higher priority {prev_schedule['train_id']} (P{prev_schedule['priority']})"
                        
                        logger.info(f"PRIORITY RESPECT: Delayed {current_train.train_id} (P{current_train.priority}) by {conflict_duration}min for higher priority {prev_schedule['train_id']} (P{prev_schedule['priority']})")
                        conflicts_resolved += 1
                        break  # Stop checking once current train is delayed
                        
                    else:
                        # Same priority - FIFO rule (delay current train)
                        delay_needed = conflict_window_end - optimized_departure
                        optimized_departure = conflict_window_end
                        optimized_arrival = optimized_arrival + delay_needed
                        current_delayed = True
                        delay_reason = f"FIFO rule vs {prev_schedule['train_id']} (same priority)"
                        
                        logger.info(f"SAME PRIORITY: Delayed {current_train.train_id} by {conflict_duration}min (FIFO rule)")
                        conflicts_resolved += 1
                        break  # Stop checking once current train is delayed

            # STEP 4: Log result for current train
            if not current_delayed:
                logger.debug(f"NO CONFLICT: {current_train.train_id} (P{current_train.priority}) runs on time at {optimized_departure.strftime('%H:%M')}")
            
            # STEP 5: Calculate final delay and add to schedule
            delay_minutes = int((optimized_departure - current_train.departure_time).total_seconds() / 60)
            
            schedule = {
                'train_id': current_train.train_id,
                'original_arrival': current_train.arrival_time,
                'original_departure': current_train.departure_time,
                'optimized_arrival': optimized_arrival,
                'optimized_departure': optimized_departure,
                'delay_minutes': delay_minutes,
                'section_id': current_train.section_id,
                'platform_need': current_train.platform_need,
                'priority': current_train.priority
            }

            schedules.append(schedule)

        # STEP 6: Final sort by optimized departure time for consistent output
        schedules.sort(key=lambda x: x['optimized_departure'])
        
        # STEP 7: Log summary statistics
        total_delay = sum(max(0, s['delay_minutes']) for s in schedules)
        on_time_count = len([s for s in schedules if s['delay_minutes'] <= 0])
        delayed_count = len(schedules) - on_time_count
        
        logger.info(f"Heuristic completed: {on_time_count}/{len(schedules)} trains on time, {delayed_count} delayed, total delay: {total_delay} minutes")
        
        return schedules

    # Additional method to add debugging capabilities to the optimizer
    def _debug_solver_failure(self, status: int) -> None:
        """Add debugging information when OR-Tools solver fails."""
        status_map = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE", 
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.UNKNOWN: "UNKNOWN",
            cp_model.MODEL_INVALID: "MODEL_INVALID"
        }
        
        status_name = status_map.get(status, f"UNKNOWN_STATUS_{status}")
        logger.warning(f"OR-Tools solver failed with status: {status_name}")
        
        if hasattr(self.solver, 'ResponseStats'):
            stats = self.solver.ResponseStats()
            logger.warning(f"Solver statistics: {stats}")
        
        if status == cp_model.INFEASIBLE:
            logger.error("Problem is INFEASIBLE - constraints are too restrictive")
            logger.error("Suggestions:")
            logger.error("  1. Increase flexibility window in _create_variables")
            logger.error("  2. Relax priority constraints") 
            logger.error("  3. Check for conflicting maintenance windows")
        elif status == cp_model.MODEL_INVALID:
            logger.error("Model is INVALID - check constraint definitions")
        elif status == cp_model.UNKNOWN:
            logger.error("Solver timed out or hit memory limits")
            logger.error(f"Current time limit: {self.time_limit_seconds} seconds")

    # Usage instructions:
    """
    1. Replace the existing _priority_fallback_heuristic method in optimizer.py with the code above
    2. Optionally add the _debug_solver_failure method for better error diagnosis
    3. Add a call to _debug_solver_failure in the optimize_schedule method after solver.Solve():

    status = self.solver.Solve(self.model)
    if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
        self._debug_solver_failure(status)  # Add this line

    This will give you proper priority-based conflict resolution where:
    - Higher priority trains never get delayed due to lower priority conflicts
    - Lower priority trains absorb necessary delays
    - Same priority conflicts use FIFO ordering
    - Total system delays are minimized
    """

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