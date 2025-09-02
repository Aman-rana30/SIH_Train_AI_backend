"""
Schedule database model for train scheduling.
"""
from sqlalchemy import Column, Integer, String, DateTime, Enum, Text, ForeignKey
from sqlalchemy.orm import relationship
import enum
from datetime import datetime

from app.db.base import Base


class ScheduleStatus(enum.Enum):
    """Enumeration for schedule status."""
    WAITING = "WAITING"
    MOVING = "MOVING"
    DELAYED = "DELAYED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class Schedule(Base):
    """
    Schedule model representing optimized train schedules.

    Attributes:
        schedule_id: Unique schedule identifier
        train_id: Foreign key to train
        planned_time: Original planned time
        optimized_time: AI-optimized time
        status: Current schedule status
        section_id: Railway section
        platform_id: Assigned platform
        delay_minutes: Calculated delay in minutes
        optimization_run_id: ID of optimization run that created this
    """
    __tablename__ = "schedules"

    id = Column(Integer, primary_key=True, index=True)
    schedule_id = Column(String(50), unique=True, index=True, nullable=False)
    train_id = Column(Integer, nullable=False)  # Temporarily removed foreign key for testing
    planned_time = Column(DateTime, nullable=False)
    optimized_time = Column(DateTime, nullable=False)
    status = Column(Enum(ScheduleStatus), nullable=False, default=ScheduleStatus.WAITING)
    section_id = Column(String(20), nullable=False)
    platform_id = Column(String(10))
    delay_minutes = Column(Integer, default=0)
    optimization_run_id = Column(String(50))
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime)

    # Relationships - temporarily commented out to fix circular import issues
    # train = relationship("Train", back_populates="schedules")

    def __repr__(self) -> str:
        return f"<Schedule(schedule_id={self.schedule_id}, train_id={self.train_id}, status={self.status})>"
