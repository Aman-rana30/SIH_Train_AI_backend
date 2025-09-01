"""
Tests for optimization engine functionality.
"""
import pytest
from datetime import datetime, timedelta

from app.services.optimization.optimizer import TrainSchedulingOptimizer
from app.services.optimization.models import TrainData, OptimizationInput, DisruptionEvent


class TestTrainSchedulingOptimizer:
    """Test cases for train scheduling optimizer."""

    def test_basic_optimization(self, sample_trains):
        """Test basic optimization functionality."""
        optimizer = TrainSchedulingOptimizer(time_limit_seconds=10)

        # Convert sample data to TrainData objects
        train_data = []
        for train in sample_trains:
            train_data.append(TrainData(
                train_id=train["train_id"],
                type=train["type"],
                arrival_time=train["arrival_time"],
                departure_time=train["departure_time"],
                section_id=train["section_id"],
                platform_need=train["platform_need"],
                priority=train["priority"],
                origin=train.get("origin"),
                destination=train.get("destination")
            ))

        # Create optimization input
        time_horizon = (
            datetime.now(),
            datetime.now() + timedelta(hours=6)
        )

        optimization_input = OptimizationInput(
            trains=train_data,
            sections=[],
            platforms=[],
            time_horizon=time_horizon,
            constraints={}
        )

        # Run optimization
        result = optimizer.optimize_schedule(optimization_input)

        # Assertions
        assert result is not None
        assert len(result.schedules) == len(train_data)
        assert result.computation_time > 0
        assert result.status in ["OPTIMAL", "FEASIBLE", "FALLBACK", "FALLBACK_HEURISTIC"]
        assert "throughput" in result.metrics

    def test_priority_ordering(self, sample_trains):
        """Test that higher priority trains are scheduled first when conflicts occur."""
        optimizer = TrainSchedulingOptimizer(time_limit_seconds=5)

        # Create conflicting trains on same section
        base_time = datetime.now().replace(microsecond=0)
        conflicting_trains = [
            TrainData(
                train_id="HIGH_PRI",
                type="Express",
                arrival_time=base_time + timedelta(hours=1),
                departure_time=base_time + timedelta(hours=1, minutes=10),
                section_id="SEC01",
                platform_need="P1", 
                priority=10  # High priority
            ),
            TrainData(
                train_id="LOW_PRI",
                type="Freight",
                arrival_time=base_time + timedelta(hours=1, minutes=5),  # Slight overlap
                departure_time=base_time + timedelta(hours=1, minutes=15),
                section_id="SEC01",
                platform_need="P1",
                priority=2   # Low priority
            )
        ]

        optimization_input = OptimizationInput(
            trains=conflicting_trains,
            sections=[],
            platforms=[],
            time_horizon=(base_time, base_time + timedelta(hours=3)),
            constraints={}
        )

        result = optimizer.optimize_schedule(optimization_input)

        # Find schedules for each train
        high_pri_schedule = next(s for s in result.schedules if s["train_id"] == "HIGH_PRI")
        low_pri_schedule = next(s for s in result.schedules if s["train_id"] == "LOW_PRI")

        # High priority train should have minimal delay
        assert high_pri_schedule["delay_minutes"] <= low_pri_schedule["delay_minutes"]

    def test_what_if_analysis(self, sample_trains):
        """Test what-if analysis functionality."""
        optimizer = TrainSchedulingOptimizer(time_limit_seconds=5)

        # Create current schedule (simplified)
        current_schedule = []
        for train in sample_trains[:3]:  # Use first 3 trains
            current_schedule.append({
                "train_id": train["train_id"],
                "optimized_arrival": train["arrival_time"],
                "optimized_departure": train["departure_time"],
                "section_id": train["section_id"],
                "platform_need": train["platform_need"],
                "priority": train["priority"]
            })

        # Create disruption
        disruption = DisruptionEvent(
            event_type="delay",
            affected_trains=["EXP001"],
            delay_minutes=30,
            start_time=datetime.now(),
            duration_minutes=60,
            description="Signal failure"
        )

        # Run what-if analysis
        result = optimizer.what_if_analysis(current_schedule, disruption)

        # Assertions
        assert result is not None
        assert len(result.schedules) <= len(current_schedule)  # Might exclude cancelled trains
        assert result.status.startswith("WHATIF") or result.status in ["OPTIMAL", "FEASIBLE", "FALLBACK_HEURISTIC"]

    def test_empty_train_list(self):
        """Test optimization with empty train list."""
        optimizer = TrainSchedulingOptimizer(time_limit_seconds=1)

        optimization_input = OptimizationInput(
            trains=[],
            sections=[],
            platforms=[],
            time_horizon=(datetime.now(), datetime.now() + timedelta(hours=1)),
            constraints={}
        )

        result = optimizer.optimize_schedule(optimization_input)

        assert result is not None
        assert len(result.schedules) == 0
        assert result.total_delay == 0

    def test_fallback_heuristic(self, sample_trains):
        """Test fallback heuristic when optimization fails."""
        optimizer = TrainSchedulingOptimizer(time_limit_seconds=0.1)  # Very short time limit

        # Convert to TrainData
        train_data = []
        for train in sample_trains:
            train_data.append(TrainData(
                train_id=train["train_id"],
                type=train["type"],
                arrival_time=train["arrival_time"],
                departure_time=train["departure_time"],
                section_id=train["section_id"],
                platform_need=train["platform_need"],
                priority=train["priority"]
            ))

        # Test fallback heuristic directly
        schedules = optimizer._priority_fallback_heuristic(train_data)

        assert len(schedules) == len(train_data)

        # Check that all trains have schedules
        train_ids = {s["train_id"] for s in schedules}
        expected_ids = {t.train_id for t in train_data}
        assert train_ids == expected_ids
