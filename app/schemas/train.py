"""
Pydantic schemas for train-related API operations.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from enum import Enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .schedule import Schedule
    from .override import Override

class TrainType(str, Enum):
    """Train type enumeration."""
    EXPRESS = "Express"
    PASSENGER = "Passenger"
    FREIGHT = "Freight"
    LOCAL = "Local"


class TrainBase(BaseModel):
    """Base train schema with common fields."""
    train_id: str = Field(..., description="Unique train identifier")
    type: TrainType = Field(..., description="Train type")
    arrival_time: datetime = Field(..., description="Scheduled arrival time")
    departure_time: datetime = Field(..., description="Scheduled departure time")
    section_id: str = Field(..., description="Railway section identifier")
    platform_need: str = Field(..., description="Required platform")
    priority: int = Field(default=1, ge=1, le=10, description="Train priority (1-10)")
    origin: Optional[str] = Field(None, description="Origin station")
    destination: Optional[str] = Field(None, description="Destination station")
    capacity: Optional[int] = Field(None, ge=0, description="Train capacity")


class TrainCreate(TrainBase):
    """Schema for creating a new train."""
    pass


class TrainUpdate(BaseModel):
    """Schema for updating train information."""
    type: Optional[TrainType] = None
    arrival_time: Optional[datetime] = None
    departure_time: Optional[datetime] = None
    section_id: Optional[str] = None
    platform_need: Optional[str] = None
    priority: Optional[int] = Field(None, ge=1, le=10)
    origin: Optional[str] = None
    destination: Optional[str] = None
    capacity: Optional[int] = Field(None, ge=0)
    active: Optional[bool] = None


class Train(TrainBase):
    """Complete train schema for API responses."""
    id: int
    active: bool = True
    # Removed circular references to avoid recursion issues
    # schedules: List["Schedule"] = Field(default_factory=list)
    # overrides: List["Override"] = Field(default_factory=list)

    class Config:
        from_attributes = True


class OptimizationRequest(BaseModel):
    """Schema for optimization requests."""
    trains: list[TrainBase] = Field(..., description="List of trains to optimize")
    optimization_params: Optional[dict] = Field(
        default_factory=dict,
        description="Additional optimization parameters"
    )


class WhatIfRequest(BaseModel):
    """Schema for what-if analysis requests."""
    disruption: dict = Field(..., description="Disruption parameters")
    affected_trains: Optional[list[str]] = Field(
        None,
        description="List of affected train IDs"
    )


