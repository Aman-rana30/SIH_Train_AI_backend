"""
Train database model.
"""
from sqlalchemy import Column, Integer, String, DateTime, Enum, Boolean
from sqlalchemy.orm import relationship
import enum

from app.db.base import Base


class TrainType(enum.Enum):
    """Enumeration for train types with priority order."""
    EXPRESS = "Express"
    PASSENGER = "Passenger" 
    FREIGHT = "Freight"
    LOCAL = "Local"


class Train(Base):
    """
    Train model representing individual trains in the system.

    Attributes:
        train_id: Unique identifier for the train
        type: Train type (Express, Passenger, Freight, Local)
        arrival_time: Scheduled arrival time
        departure_time: Scheduled departure time
        section_id: Railway section identifier
        platform_need: Required platform identifier
        priority: Numeric priority (higher = more important)
        active: Whether the train is currently active
        origin: Origin station
        destination: Destination station
        capacity: Maximum passenger/cargo capacity
    """
    __tablename__ = "trains"

    id = Column(Integer, primary_key=True, index=True)
    train_id = Column(String(50), unique=True, index=True, nullable=False)
    type = Column(Enum(TrainType), nullable=False)
    arrival_time = Column(DateTime, nullable=False)
    departure_time = Column(DateTime, nullable=False)
    section_id = Column(String(20), nullable=False)
    platform_need = Column(String(10), nullable=False)
    priority = Column(Integer, nullable=False, default=1)
    active = Column(Boolean, default=True)
    origin = Column(String(100))
    destination = Column(String(100))
    capacity = Column(Integer)

    # Relationships - temporarily commented out to fix circular import issues
    # schedules = relationship("Schedule", back_populates="train")
    # overrides = relationship("Override", back_populates="train")

    def __repr__(self) -> str:
        return f"<Train(train_id={self.train_id}, type={self.type}, priority={self.priority})>"
