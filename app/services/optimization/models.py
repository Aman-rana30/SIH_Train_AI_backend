"""
Data models for optimization operations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Union

@dataclass
class TrainData:
    """Data structure for train information used in optimization."""
    train_id: str
    type: str
    arrival_time: Union[datetime, str]
    departure_time: Union[datetime, str]
    section_id: str
    platform_need: str
    priority: int
    origin: Optional[str] = None
    destination: Optional[str] = None

    def __post_init__(self):
        """Convert string timestamps to datetime objects if needed."""
        if isinstance(self.arrival_time, str):
            self.arrival_time = datetime.fromisoformat(self.arrival_time.replace('Z', '+00:00'))
        if isinstance(self.departure_time, str):
            self.departure_time = datetime.fromisoformat(self.departure_time.replace('Z', '+00:00'))

@dataclass
class SectionData:
    """Data structure for railway section information."""
    section_id: str
    length_km: float
    max_speed_kmh: int  # Updated to match database model
    maintenance_windows: List[tuple[datetime, datetime]]
    capacity: int = 1  # Default capacity for single track
    single_track: bool = True  # Default assume single track

    def calculate_travel_time(self, buffer_factor: float = 1.15) -> int:
        """Calculate travel time for this section with buffer."""
        base_time_hours = self.length_km / self.max_speed_kmh
        actual_time_hours = base_time_hours * buffer_factor
        return int(actual_time_hours * 60)  # Convert to minutes

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