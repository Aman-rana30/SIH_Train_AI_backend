"""
Schedule database model.
"""
from sqlalchemy import Column, Integer, String, DateTime, Enum, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class ScheduleStatus(enum.Enum):
    """Enumeration for schedule status."""
    WAITING = "waiting"
    MOVING = "moving"
    DELAYED = "delayed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


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
    train_id = Column(Integer, ForeignKey("trains.id"), nullable=False)
    planned_time = Column(DateTime, nullable=False)
    optimized_time = Column(DateTime, nullable=False)
    status = Column(Enum(ScheduleStatus), nullable=False, default=ScheduleStatus.WAITING)
    section_id = Column(String(20), nullable=False)
    platform_id = Column(String(10))
    delay_minutes = Column(Integer, default=0)
    optimization_run_id = Column(String(50))
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime)

    # Relationships
    train = relationship("Train", back_populates="schedules")

    def __repr__(self) -> str:
        return f"<Schedule(schedule_id={self.schedule_id}, train_id={self.train_id}, status={self.status})>"
