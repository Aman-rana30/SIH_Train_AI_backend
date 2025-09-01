"""
Data models for optimization operations.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional


@dataclass
class TrainData:
    """Data structure for train information used in optimization."""
    train_id: str
    type: str
    arrival_time: datetime
    departure_time: datetime
    section_id: str
    platform_need: str
    priority: int
    origin: Optional[str] = None
    destination: Optional[str] = None


@dataclass
class SectionData:
    """Data structure for railway section information."""
    section_id: str
    capacity: int
    length_km: float
    max_speed: int
    maintenance_windows: List[tuple[datetime, datetime]]


@dataclass
class PlatformData:
    """Data structure for platform information."""
    platform_id: str
    type: str  # passenger, freight, mixed
    capacity: int
    available_times: List[tuple[datetime, datetime]]


@dataclass
class OptimizationInput:
    """Complete input data for optimization."""
    trains: List[TrainData]
    sections: List[SectionData] 
    platforms: List[PlatformData]
    time_horizon: tuple[datetime, datetime]
    constraints: Dict[str, any]


@dataclass
class OptimizationOutput:
    """Results from optimization."""
    schedules: List[Dict[str, any]]
    objective_value: float
    computation_time: float
    status: str
    metrics: Dict[str, float]
    conflicts_resolved: int
    total_delay: float


@dataclass
class DisruptionEvent:
    """Data structure for disruption events in what-if analysis."""
    event_type: str  # delay, cancellation, emergency
    affected_trains: List[str]
    delay_minutes: int
    start_time: datetime
    duration_minutes: int
    description: str
