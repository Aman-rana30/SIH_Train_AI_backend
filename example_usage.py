#!/usr/bin/env python3
"""
Example usage of the Train Traffic Control Optimization System.

This script demonstrates how to use the optimization engine directly
and shows a complete working example with sample data.
"""
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from app.services.optimization.optimizer import TrainSchedulingOptimizer
from app.services.optimization.models import TrainData, OptimizationInput


def load_sample_data():
    """Load sample train data from JSON file."""
    data_file = Path(__file__).parent / "data" / "sample_trains.json"

    with open(data_file, 'r') as f:
        data = json.load(f)

    return data


def create_train_data_from_sample(sample_trains):
    """Convert sample train data to TrainData objects."""
    trains = []

    for train_info in sample_trains:
        train_data = TrainData(
            train_id=train_info["train_id"],
            type=train_info["type"],
            arrival_time=datetime.fromisoformat(train_info["arrival_time"]),
            departure_time=datetime.fromisoformat(train_info["departure_time"]),
            section_id=train_info["section_id"],
            platform_need=train_info["platform_need"],
            priority=train_info["priority"],
            origin=train_info.get("origin"),
            destination=train_info.get("destination")
        )
        trains.append(train_data)

    return trains


def print_optimization_results(result):
    """Print optimization results in a formatted way."""
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)

    print(f"Status: {result.status}")
    print(f"Computation Time: {result.computation_time:.3f} seconds")
    print(f"Objective Value: {result.objective_value}")
    print(f"Total Delay: {result.total_delay:.1f} minutes")
    print(f"Conflicts Resolved: {result.conflicts_resolved}")

    print(f"\nMetrics:")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print(f"\nSchedules ({len(result.schedules)} trains):")
    print("-" * 60)

    for schedule in result.schedules:
        delay_str = f"+{schedule['delay_minutes']}m" if schedule['delay_minutes'] > 0 else "On time"

        print(f"Train {schedule['train_id']:6} | "
              f"Priority: {schedule['priority']:2} | "
              f"Section: {schedule['section_id']:5} | "
              f"Platform: {schedule['platform_need']:3} | "
              f"Delay: {delay_str:8}")

        print(f"  Original:  {schedule['original_arrival'].strftime('%H:%M:%S')} -> "
              f"{schedule['original_departure'].strftime('%H:%M:%S')}")

        print(f"  Optimized: {schedule['optimized_arrival'].strftime('%H:%M:%S')} -> "
              f"{schedule['optimized_departure'].strftime('%H:%M:%S')}")
        print()


def run_basic_optimization_example():
    """Run a basic optimization example with sample data."""
    print("Loading sample data...")
    sample_data = load_sample_data()

    # Use first 5 trains for this example
    sample_trains = sample_data["sample_trains"][:5]
    train_data = create_train_data_from_sample(sample_trains)

    print(f"\nRunning optimization for {len(train_data)} trains...")

    # Set up time horizon
    min_time = min(train.arrival_time for train in train_data)
    max_time = max(train.departure_time for train in train_data)
    time_horizon = (
        min_time - timedelta(minutes=30),
        max_time + timedelta(minutes=60)
    )

    # Create optimization input
    optimization_input = OptimizationInput(
        trains=train_data,
        sections=[],  # Simplified for example
        platforms=[],
        time_horizon=time_horizon,
        constraints={"max_delay": 30}
    )

    # Run optimization
    optimizer = TrainSchedulingOptimizer(time_limit_seconds=20)
    result = optimizer.optimize_schedule(optimization_input)

    # Print results
    print_optimization_results(result)


def run_conflict_resolution_example():
    """Run an example showing conflict resolution between trains."""
    print("\n" + "="*60)
    print("CONFLICT RESOLUTION EXAMPLE")
    print("="*60)

    # Create conflicting trains on the same section
    base_time = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)

    conflicting_trains = [
        TrainData(
            train_id="EXPRESS_A",
            type="Express",
            arrival_time=base_time,
            departure_time=base_time + timedelta(minutes=10),
            section_id="MAIN_LINE",
            platform_need="P1",
            priority=10  # Highest priority
        ),
        TrainData(
            train_id="LOCAL_B",
            type="Local", 
            arrival_time=base_time + timedelta(minutes=5),  # Conflicts with EXPRESS_A
            departure_time=base_time + timedelta(minutes=15),
            section_id="MAIN_LINE",  # Same section
            platform_need="P1",     # Same platform
            priority=4   # Lower priority
        ),
        TrainData(
            train_id="FREIGHT_C",
            type="Freight",
            arrival_time=base_time + timedelta(minutes=8),  # Also conflicts
            departure_time=base_time + timedelta(minutes=25),
            section_id="MAIN_LINE",  # Same section
            platform_need="P2",     # Different platform
            priority=2   # Lowest priority
        )
    ]

    print("\nOriginal Schedule (with conflicts):")
    for train in conflicting_trains:
        print(f"  {train.train_id:10} | Priority: {train.priority:2} | "
              f"{train.arrival_time.strftime('%H:%M')} - {train.departure_time.strftime('%H:%M')} | "
              f"Section: {train.section_id}")

    # Set up optimization
    time_horizon = (base_time - timedelta(minutes=30), base_time + timedelta(hours=2))
    optimization_input = OptimizationInput(
        trains=conflicting_trains,
        sections=[],
        platforms=[],
        time_horizon=time_horizon,
        constraints={}
    )

    # Run optimization
    optimizer = TrainSchedulingOptimizer(time_limit_seconds=15)
    result = optimizer.optimize_schedule(optimization_input)

    # Print results
    print_optimization_results(result)


def run_what_if_scenario():
    """Run a what-if analysis scenario."""
    print("\n" + "="*60) 
    print("WHAT-IF ANALYSIS EXAMPLE")
    print("="*60)

    sample_data = load_sample_data()
    sample_trains = sample_data["sample_trains"][:4]

    # Create current schedule
    current_schedule = []
    for train_info in sample_trains:
        current_schedule.append({
            "train_id": train_info["train_id"],
            "optimized_arrival": datetime.fromisoformat(train_info["arrival_time"]),
            "optimized_departure": datetime.fromisoformat(train_info["departure_time"]),
            "section_id": train_info["section_id"],
            "platform_need": train_info["platform_need"],
            "priority": train_info["priority"]
        })

    print("\nOriginal Schedule:")
    for schedule in current_schedule:
        print(f"  {schedule['train_id']:10} | "
              f"{schedule['optimized_arrival'].strftime('%H:%M')} - "
              f"{schedule['optimized_departure'].strftime('%H:%M')} | "
              f"Section: {schedule['section_id']}")

    # Create disruption scenario
    from app.services.optimization.models import DisruptionEvent

    disruption = DisruptionEvent(
        event_type="delay",
        affected_trains=["EXP001"],  # Delay the express train
        delay_minutes=45,
        start_time=datetime.now(),
        duration_minutes=60,
        description="Signal failure on main line"
    )

    print(f"\nDisruption Scenario:")
    print(f"  Type: {disruption.event_type}")
    print(f"  Affected Trains: {', '.join(disruption.affected_trains)}")
    print(f"  Delay: {disruption.delay_minutes} minutes")
    print(f"  Description: {disruption.description}")

    # Run what-if analysis
    optimizer = TrainSchedulingOptimizer(time_limit_seconds=15)
    result = optimizer.what_if_analysis(current_schedule, disruption)

    # Print results
    print_optimization_results(result)


async def main():
    """Main function to run all examples."""
    print("TRAIN TRAFFIC CONTROL OPTIMIZATION SYSTEM")
    print("Example Usage and Demonstration")
    print("=" * 60)

    try:
        # Run basic optimization example
        run_basic_optimization_example()

        # Run conflict resolution example
        run_conflict_resolution_example()

        # Run what-if analysis example
        run_what_if_scenario()

        print("\n" + "="*60)
        print("EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nNext Steps:")
        print("1. Set up PostgreSQL database and update DATABASE_URL in .env")
        print("2. Run: alembic upgrade head (to create database tables)")
        print("3. Start API server: uvicorn app.main:app --reload")
        print("4. Visit http://localhost:8000/docs for API documentation")
        print("5. Test WebSocket: Connect to ws://localhost:8000/ws/updates")

    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        print("  or")
        print("  poetry install")


if __name__ == "__main__":
    asyncio.run(main())
