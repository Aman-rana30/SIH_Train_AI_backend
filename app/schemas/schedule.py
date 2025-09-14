"""
Pydantic schemas for schedule-related API operations.
"""
from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import datetime
from typing import Optional
from enum import Enum


# Import Train for forward reference resolution
from app.schemas.train import Train

class ScheduleStatus(str, Enum):
    """Schedule status enumeration."""
    WAITING = "WAITING"
    MOVING = "MOVING"
    DELAYED = "DELAYED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class ScheduleBase(BaseModel):
    """Base schedule schema."""
    schedule_id: str = Field(..., description="Unique schedule identifier")
    train_id: str = Field(..., description="Associated train ID (human-readable, e.g., '14505')")
    planned_time: datetime = Field(..., description="Original planned time")
    optimized_time: datetime = Field(..., description="AI-optimized time")
    section_id: str = Field(..., description="Railway section")
    platform_id: Optional[str] = Field(None, description="Assigned platform")

    # Defensive: ensure train_id is always coerced to string
    @field_validator('train_id', mode='before')
    @classmethod
    def ensure_train_id_str(cls, v):
        return str(v) if v is not None else v


class ScheduleCreate(ScheduleBase):
    """Schema for creating schedules."""
    status: ScheduleStatus = ScheduleStatus.WAITING
    delay_minutes: int = Field(default=0, description="Delay in minutes")
    optimization_run_id: Optional[str] = None


class ScheduleUpdate(BaseModel):
    """Schema for updating schedules."""
    optimized_time: Optional[datetime] = None
    status: Optional[ScheduleStatus] = None
    platform_id: Optional[str] = None
    delay_minutes: Optional[int] = None


class Schedule(ScheduleBase):
    """Complete schedule schema for responses."""
    id: int
    status: ScheduleStatus
    delay_minutes: int
    optimization_run_id: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    train: Optional["Train"] = None

    model_config = ConfigDict(from_attributes=True)


class OptimizationResult(BaseModel):
    """Schema for optimization results."""
    optimization_run_id: str = Field(..., description="Unique optimization run ID")
    schedules: list["Schedule"] = Field(..., description="Optimized schedules")
    metrics: dict = Field(..., description="Optimization metrics")
    computation_time: float = Field(..., description="Time taken for optimization")
    status: str = Field(..., description="Optimization status")
  
    # Fix for Pydantic forward references
    Schedule.model_rebuild()


