"""
Pydantic schemas for train-related API operations.
"""
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime
from typing import Optional, List, Any, Union
from enum import Enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .schedule import Schedule
    from .override import Override

class TrainType(str, Enum):
    """Core train operational categories."""
    EXPRESS = "Express"
    SUPERFAST = "Superfast"
    SHATABDI = "Shatabdi"
    RAJDHANI = "Rajdhani"
    DURONTO = "Duronto"
    GARIB_RATH = "GaribRath"
    PASSENGER = "Passenger"
    FREIGHT = "Freight"
    LOCAL = "Local"


class TrainBase(BaseModel):
    """Base train schema with common fields."""
    train_id: str = Field(..., description="Unique train identifier")
    type: TrainType = Field(..., description="Core train operational category")
    sub_type: Optional[str] = Field(None, description="Optional specific train name/variant (e.g., 'Hirakund Express', 'Swarna Jayanti')")
    arrival_time: Union[datetime, str] = Field(..., description="Scheduled arrival time")
    departure_time: Union[datetime, str] = Field(..., description="Scheduled departure time")
    section_id: str = Field(..., description="Railway section identifier")
    platform_need: str = Field(..., description="Required platform")
    priority: int = Field(default=1, ge=1, le=10, description="Train priority (1-10)")
    origin: Optional[str] = Field(None, description="Origin station")
    destination: Optional[str] = Field(None, description="Destination station")
    capacity: Optional[int] = Field(None, ge=0, description="Train capacity")

    @field_validator('arrival_time', 'departure_time', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError('Invalid datetime format. Use ISO format (YYYY-MM-DDTHH:MM:SS)')
        return v

    model_config = ConfigDict(from_attributes=True)


class TrainCreate(TrainBase):
    """Schema for creating a new train."""
    pass


class TrainUpdate(BaseModel):
    """Schema for updating train information."""
    type: Optional[TrainType] = None
    sub_type: Optional[str] = None
    arrival_time: Optional[Union[datetime, str]] = None
    departure_time: Optional[Union[datetime, str]] = None
    section_id: Optional[str] = None
    platform_need: Optional[str] = None
    priority: Optional[int] = Field(None, ge=1, le=10)
    origin: Optional[str] = None
    destination: Optional[str] = None
    capacity: Optional[int] = Field(None, ge=0)
    active: Optional[bool] = None

    @field_validator('arrival_time', 'departure_time', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError('Invalid datetime format. Use ISO format (YYYY-MM-DDTHH:MM:SS)')
        return v

    model_config = ConfigDict(from_attributes=True)


class Train(TrainBase):
    """Complete train schema for API responses."""
    id: int
    active: bool = True
    # Removed circular references to avoid recursion issues
    # schedules: List["Schedule"] = Field(default_factory=list)
    # overrides: List["Override"] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class OptimizationRequest(BaseModel):
    """Schema for optimization requests."""
    trains: list[TrainBase] = Field(..., description="List of trains to optimize")
    optimization_params: Optional[dict] = Field(
        default_factory=dict,
        description="Additional optimization parameters"
    )

    model_config = ConfigDict(from_attributes=True)


class SimulationType(str, Enum):
    """Simulation type enumeration."""
    TRAIN_DELAY = "TRAIN_DELAY"
    ENVIRONMENTAL_CONDITION = "ENVIRONMENTAL_CONDITION"


class TrackConditionEnum(str, Enum):
    """Track condition enumeration for API schemas."""
    GOOD = "GOOD"
    WORN = "WORN"
    MAINTENANCE = "MAINTENANCE"


class WeatherConditionEnum(str, Enum):
    """Weather condition enumeration for API schemas."""
    CLEAR = "CLEAR"
    RAIN = "RAIN"
    HEAVY_RAIN = "HEAVY_RAIN"
    FOG = "FOG"


class DisruptionEvent(BaseModel):
    """Enhanced disruption event model for environmental and delay simulations."""
    simulation_type: SimulationType = Field(..., description="Type of simulation to run")
    
    # For train delay simulations
    delay_minutes: Optional[int] = Field(None, description="Delay in minutes for train delay simulation")
    affected_trains: Optional[list[str]] = Field(None, description="List of affected train IDs for delay simulation")
    
    # For environmental condition simulations
    affected_sections: Optional[list[str]] = Field(None, description="List of affected section IDs for environmental simulation")
    weather_condition: Optional[WeatherConditionEnum] = Field(None, description="Weather condition for environmental simulation")
    track_condition: Optional[TrackConditionEnum] = Field(None, description="Track condition for environmental simulation")
    
    # Common fields
    description: Optional[str] = Field(None, description="Description of the disruption event")
    duration_minutes: Optional[int] = Field(60, description="Duration of the disruption in minutes")

    # Defensive: coerce numeric IDs to string for compatibility
    @field_validator('affected_trains', mode='before')
    @classmethod
    def coerce_affected_trains(cls, v):
        if v is None:
            return v
        try:
            return [str(x) for x in v]
        except Exception:
            return v

    model_config = ConfigDict(from_attributes=True)


class WhatIfRequest(BaseModel):
    """Schema for what-if analysis requests."""
    disruption_event: DisruptionEvent = Field(..., description="Detailed disruption event parameters")
    
    # Legacy support for backward compatibility
    disruption: Optional[dict] = Field(None, description="Legacy disruption parameters (deprecated)")
    affected_trains: Optional[list[str]] = Field(None, description="Legacy affected trains list (deprecated)")

    model_config = ConfigDict(from_attributes=True)


