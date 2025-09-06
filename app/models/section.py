"""
Section database model for track segment properties.
"""
from sqlalchemy import Column, Integer, String, Float, Enum
from enum import Enum as PyEnum
from app.db.base import Base

class TrackCondition(PyEnum):
    """Track condition enumeration."""
    GOOD = "GOOD"
    WORN = "WORN"
    MAINTENANCE = "MAINTENANCE"

class WeatherCondition(PyEnum):
    """Weather condition enumeration."""
    CLEAR = "CLEAR"
    RAIN = "RAIN"
    HEAVY_RAIN = "HEAVY_RAIN"
    FOG = "FOG"

class Section(Base):
    """
    Section model representing the physical properties of a track segment.

    Attributes:
        id: Unique database identifier for the section.
        section_id: Human-readable unique identifier (e.g., "JUC-LDH").
        length_km: The length of the section in kilometers.
        max_speed_kmh: The maximum permissible speed in the section in km/h.
        description: A brief description of the section.
        track_condition: Current track condition (GOOD, WORN, MAINTENANCE).
        current_weather: Current weather condition (CLEAR, RAIN, HEAVY_RAIN, FOG).
    """
    __tablename__ = "sections"

    id = Column(Integer, primary_key=True, index=True)
    section_id = Column(String(50), unique=True, index=True, nullable=False)
    length_km = Column(Float, nullable=False)
    max_speed_kmh = Column(Integer, nullable=False)
    description = Column(String(255), nullable=True)
    track_condition = Column(Enum(TrackCondition), default=TrackCondition.GOOD, nullable=False)
    current_weather = Column(Enum(WeatherCondition), default=WeatherCondition.CLEAR, nullable=False)

    def __repr__(self) -> str:
        return f"<Section(section_id={self.section_id}, length_km={self.length_km}, track_condition={self.track_condition}, weather={self.current_weather})>"