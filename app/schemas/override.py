"""
Pydantic schemas for override-related API operations.
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class OverrideBase(BaseModel):
    """Base override schema."""
    override_id: str = Field(..., description="Unique override identifier")
    train_id: int = Field(..., description="Affected train ID")
    controller_decision: str = Field(..., description="Manual controller decision")
    ai_recommendation: Optional[str] = Field(None, description="AI recommendation")
    reason: Optional[str] = Field(None, description="Reason for override")
    controller_id: Optional[str] = Field(None, description="Controller ID")


class OverrideCreate(OverrideBase):
    """Schema for creating overrides."""
    impact_delay: int = Field(default=0, description="Estimated delay impact in minutes")


class Override(OverrideBase):
    """Complete override schema for responses."""
    id: int
    impact_delay: int
    timestamp: datetime

    class Config:
        from_attributes = True


class OverrideRequest(BaseModel):
    """Schema for override requests from controllers."""
    train_id: int = Field(..., description="Train to override")
    decision: str = Field(..., description="Controller decision")
    reason: Optional[str] = Field(None, description="Reason for override")
    new_schedule_time: Optional[datetime] = Field(None, description="New scheduled time")
    controller_id: str = Field(..., description="Controller making override")
