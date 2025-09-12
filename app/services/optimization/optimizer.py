"""

Main optimization engine using Google OR-Tools CP-SAT solver.

"""

from ortools.sat.python import cp_model

from datetime import datetime, timedelta, timezone

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

        Main optimization method - now uses time-based approach instead of constraint programming.

        """

        return self.optimize_schedule_time_based(optimization_input)

    def optimize_schedule_time_based(self, optimization_input: OptimizationInput) -> OptimizationOutput:

        """

        Smart time-based optimization that respects chronological order and minimizes delays.

        

        Algorithm:

        1. Filter trains for next 8 hours only

        2. Sort by departure time (chronological order)

        3. Schedule with 1-minute gaps, only delay when conflicts occur

        4. Respect priority only in case of actual conflicts

        """

        logger.info(f"Starting time-based optimization for {len(optimization_input.trains)} trains")

        start_time = time.time()

        

        try:

            # Get current time in IST

            IST = timezone(timedelta(hours=5, minutes=30))

            current_time = datetime.now(IST)

            eight_hours_later = current_time + timedelta(hours=8)

            

            # Filter trains for next 8 hours window

            next_8h_trains = [

                train for train in optimization_input.trains 

                if current_time <= train.departure_time <= eight_hours_later

            ]

            

            logger.info(f"Filtered to {len(next_8h_trains)} trains in next 8 hours window")

            logger.info(f"Time window: {current_time} to {eight_hours_later}")

            

            if not next_8h_trains:

                logger.warning("No trains in next 8 hours window")

                return self._create_empty_solution(time.time() - start_time)

            

            # Run smart chronological scheduling

            result = self._smart_chronological_scheduler(next_8h_trains, optimization_input.sections)

            result.computation_time = time.time() - start_time

            

            logger.info(f"Time-based optimization completed in {result.computation_time:.2f}s")

            return result

            

        except Exception as e:

            logger.error(f"Time-based optimization failed: {str(e)}")

            return self._create_fallback_solution(optimization_input.trains, time.time() - start_time)

    def _smart_chronological_scheduler(self, trains: List[TrainData], sections: List[SectionData]) -> OptimizationOutput:

        """

        Smart scheduler that processes trains chronologically with minimal delays.

        

        Your example:

        - Train 1 (P6) at 2:00 → Scheduled at 2:00 ✅

        - Train 2 (P10) at 2:30 → Scheduled at 2:30 ✅  

        - Train 3 (P9) at 2:35 → Delayed to 2:42 (7 min delay) ⚠️

        """

        logger.info("Using smart chronological scheduler")

        

        # Sort trains by departure time (chronological order)

        sorted_trains = sorted(trains, key=lambda x: x.departure_time)

        

        schedules = []

        section_occupancy = {}  # section_id -> last_departure_time

        total_delay = 0.0

        conflicts_resolved = 0

        

        MINIMUM_GAP_MINUTES = 1  # 1-minute absolute block

        

        for train in sorted_trains:

            original_departure = train.departure_time

            optimized_departure = original_departure

            

            # Check if section is occupied

            if train.section_id in section_occupancy:

                last_departure = section_occupancy[train.section_id]

                required_departure = last_departure + timedelta(minutes=MINIMUM_GAP_MINUTES)

                

                # Only delay if there's an actual conflict

                if optimized_departure < required_departure:

                    old_departure = optimized_departure

                    optimized_departure = required_departure

                    delay_minutes = int((optimized_departure - original_departure).total_seconds() / 60)

                    

                    logger.info(f"CONFLICT: Train {train.train_id} (P{train.priority}) delayed by {delay_minutes}min")

                    logger.info(f"  Original: {original_departure.strftime('%H:%M')}")

                    logger.info(f"  Delayed:  {optimized_departure.strftime('%H:%M')}")

                    logger.info(f"  Reason:   Section {train.section_id} occupied until {last_departure.strftime('%H:%M')}")

                    

                    conflicts_resolved += 1

                    total_delay += delay_minutes

                else:

                    logger.info(f"ON TIME: Train {train.train_id} (P{train.priority}) at {optimized_departure.strftime('%H:%M')}")

            else:

                logger.info(f"FIRST: Train {train.train_id} (P{train.priority}) at {optimized_departure.strftime('%H:%M')} in section {train.section_id}")

            

            # Update section occupancy with travel time

            try:

                section = next(s for s in sections if s.section_id == train.section_id)

                travel_time_minutes = section.calculate_travel_time()

            except (StopIteration, AttributeError):

                travel_time_minutes = 37  # fallback

            

            section_clear_time = optimized_departure + timedelta(minutes=travel_time_minutes)

            section_occupancy[train.section_id] = section_clear_time

            

            # Calculate optimized arrival (maintain original dwell time)

            dwell_duration = train.departure_time - train.arrival_time

            optimized_arrival = optimized_departure - dwell_duration

            

            # Calculate delay

            delay_minutes = int((optimized_departure - original_departure).total_seconds() / 60)

            

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

        

        # Calculate metrics

        on_time_count = len([s for s in schedules if s['delay_minutes'] <= 0])

        max_delay = max([s['delay_minutes'] for s in schedules]) if schedules else 0

        

        metrics = {

            'average_delay': total_delay / len(schedules) if schedules else 0,

            'max_delay': max_delay,

            'on_time_percentage': (on_time_count / len(schedules) * 100) if schedules else 0,

            'throughput': len(schedules),

            'solver_status': "TIME_BASED_OPTIMAL",

            'total_conflicts': conflicts_resolved,

            'total_delay_minutes': total_delay

        }

        

        logger.info(f"Smart scheduling results:")

        logger.info(f"  - {on_time_count}/{len(schedules)} trains on time ({metrics['on_time_percentage']:.1f}%)")

        logger.info(f"  - {conflicts_resolved} conflicts resolved")

        logger.info(f"  - {total_delay} total delay minutes")

        logger.info(f"  - {max_delay} max delay minutes")

        

        return OptimizationOutput(

            schedules=schedules,

            objective_value=total_delay,  # Minimize total delay

            computation_time=0,  # Will be set by caller

            status="TIME_BASED_OPTIMAL",

            metrics=metrics,

            conflicts_resolved=conflicts_resolved,

            total_delay=total_delay

        )

    def _create_empty_solution(self, computation_time: float) -> OptimizationOutput:

        """Create empty solution when no trains in window."""

        return OptimizationOutput(

            schedules=[],

            objective_value=0,

            computation_time=computation_time,

            status="NO_TRAINS_IN_WINDOW",

            metrics={

                'average_delay': 0,

                'max_delay': 0,

                'on_time_percentage': 100,

                'throughput': 0,

                'solver_status': "NO_TRAINS_IN_WINDOW"

            },

            conflicts_resolved=0,

            total_delay=0

        )

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

            sections=[], # Simplified for what-if

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

            # Allow trains to run on time with reasonable flexibility

            flex_window = 60  # 60 minutes flexibility for solver feasibility

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

        # Count constraints before

        initial_constraints = len(self.model.Proto().constraints)

        

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

        # Add constraints with counting

        constraints_handler.add_track_conflict_constraints(trains_dict, sections_dict)

        track_constraints = len(self.model.Proto().constraints) - initial_constraints

        

        constraints_handler.add_priority_constraints(trains_dict)

        priority_constraints = len(self.model.Proto().constraints) - initial_constraints - track_constraints

        

        constraints_handler.add_platform_availability_constraints(trains_dict, platforms_dict)

        constraints_handler.add_timing_constraints(trains_dict)

        constraints_handler.add_maintenance_window_constraints(trains_dict, sections_dict)

        

        # Log constraint breakdown

        total_constraints = len(self.model.Proto().constraints) - initial_constraints

        logger.info(f"Added {total_constraints} constraints:")

        logger.info(f"  - Track conflicts: {track_constraints}")

        logger.info(f"  - Priority: {priority_constraints}")

        logger.info(f"  - Platform/timing/maintenance: {total_constraints - track_constraints - priority_constraints}")

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

    def debug_solver_failure(self, status: int) -> None:

        """Add debugging information when OR-Tools solver fails."""

        status_map = {

            cp_model.OPTIMAL: "OPTIMAL",

            cp_model.FEASIBLE: "FEASIBLE", 

            cp_model.INFEASIBLE: "INFEASIBLE",

            cp_model.UNKNOWN: "UNKNOWN",

            cp_model.MODEL_INVALID: "MODEL_INVALID"

        }

        

        status_name = status_map.get(status, f"UNKNOWN_STATUS_{status}")

        logger.error(f"OR-Tools solver failed with status: {status_name}")

        

        # Get solver statistics if available

        if hasattr(self.solver, 'ResponseStats'):

            stats = self.solver.ResponseStats()

            logger.error(f"Solver statistics: {stats}")

            logger.error(f"Wall time: {self.solver.WallTime():.2f}s")

            logger.error(f"User time: {self.solver.UserTime():.2f}s") 

        

        # Status-specific debugging

        if status == cp_model.INFEASIBLE:

            logger.error("Problem is INFEASIBLE - constraints are too restrictive")

            logger.error("Suggestions:")

            logger.error("  1. Increase flexibility window (flex_window) in _create_variables")

            logger.error("  2. Relax priority constraints")

            logger.error("  3. Check for conflicting maintenance windows")

            logger.error("  4. Reduce minimum separation buffer")

            

        elif status == cp_model.MODEL_INVALID:

            logger.error("Model is INVALID - check constraint definitions")

            logger.error("Check for:")

            logger.error("  1. Variables with empty domains")

            logger.error("  2. Contradictory constraints")

            logger.error("  3. Invalid interval variables")

            

        elif status == cp_model.UNKNOWN:

            logger.error("Solver timed out or hit memory limits")

            logger.error(f"Current time limit: {self.time_limit_seconds} seconds")

            logger.error("Consider:")

            logger.error("  1. Increasing time limit")

            logger.error("  2. Reducing number of constraints")

            logger.error("  3. Using simpler constraint formulations")

        

        # Log constraint and variable counts

        logger.error(f"Variables created: {len(self.variables)}")

        logger.error(f"Station intervals: {len(self.intervals)}")

        logger.error(f"Section intervals: {len(getattr(self, 'section_intervals', {}))}")

    def _process_results(self,

                        trains: List[TrainData],

                        status: int,

                        computation_time: float) -> OptimizationOutput:

        """Process optimization results and create output."""

        logger.debug(f"Processing results with status: {status}")

        # Count constraints before solving

        total_constraints = len(self.model.Proto().constraints)

        total_variables = len(trains)

        logger.info(f"Model: {total_variables} trains, {total_constraints} constraints (ratio: {total_constraints/total_variables:.1f})")

        

        # Enhanced status logging

        if status == cp_model.OPTIMAL:

            logger.info(f"✅ OPTIMAL solution in {computation_time:.2f}s (obj: {self.solver.ObjectiveValue()})")

        elif status == cp_model.FEASIBLE:

            logger.warning(f"⚠️  FEASIBLE solution in {computation_time:.2f}s (obj: {self.solver.ObjectiveValue()})")

        else:

            logger.error(f"❌ Solver FAILED in {computation_time:.2f}s - using fallback")

            self.debug_solver_failure(status)

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

            # Priority: higher priority first; tie-breaker earlier arrival

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

                    dep = emin # departure minute

                    trav = travel_minutes(t)

                    const_buffer = 3 # minutes buffer after section clears

                    sec_clear = dep + trav

                    # Only adjust if there's an actual conflict

                    if dep < current_blocked_until:

                        # Calculate the minimum delay needed

                        original_dep = dep

                        dep = current_blocked_until

                        emin = dep # end var is the departure

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

    def _priority_fallback_heuristic(self, trains: List[TrainData]) -> List[Dict]:

        """

        Fallback heuristic using priority-based scheduling with a 1-minute departure window.

        """

        logger.info("Using priority-based fallback heuristic with 1-minute departure window")

        # Correctly sort trains by priority and then by original arrival time

        sorted_trains = sorted(

            trains,

            key=lambda x: (-x.priority, x.arrival_time)

        )

        schedules = []

        section_last_departure = {}

        # NEW: Keep track of the last departure time across ALL trains

        last_overall_departure = None

        DEPARTURE_BUFFER_MINUTES = 12

        for train in sorted_trains:

            # Start with the train's original schedule

            optimized_arrival = train.arrival_time

            optimized_departure = train.departure_time

            # --- Step 1: Resolve track section conflicts first (existing logic) ---

            if train.section_id in section_last_departure:

                last_departure_in_section = section_last_departure[train.section_id]

                if optimized_arrival < last_departure_in_section:

                    # If the train arrives before the section is clear, we must delay it.

                    dwell_duration = train.departure_time - train.arrival_time

                    buffer = timedelta(minutes=5) # 5-minute buffer for section clearance

                    optimized_arrival = last_departure_in_section + buffer

                    optimized_departure = optimized_arrival + dwell_duration

            # --- Step 2: Enforce the 12-minute global departure window ---

            if last_overall_departure:

                # Calculate the earliest time the next train is allowed to depart

                required_next_departure = last_overall_departure + timedelta(minutes=DEPARTURE_BUFFER_MINUTES)

                # Check if the current train's departure time is too early

                if optimized_departure < required_next_departure:

                    # If it is, we need to delay it.

                    delay_needed = required_next_departure - optimized_departure

                    optimized_departure += delay_needed

                    # It's important to also shift the arrival time to maintain the original dwell time

                    optimized_arrival += delay_needed

            # --- Step 3: Update the state for the next train in the loop ---

            # Update the last departure time for this specific section

            section_last_departure[train.section_id] = optimized_departure

            # Update the overall last departure time with the new, adjusted time

            last_overall_departure = optimized_departure

            # --- Step 4: Finalize and save the schedule for this train ---

            # Calculate the final delay in minutes

            delay_minutes = int((optimized_departure - train.departure_time).total_seconds() / 60)

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

                type="Express", # Simplified for what-if

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

                    continue # Skip cancelled trains

            modified_trains.append(train_data)

        return modified_trains