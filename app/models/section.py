"""
Section database model for track segment properties.
"""
from sqlalchemy import Column, Integer, String, Float
from app.db.base import Base

class Section(Base):
    """
    Section model representing the physical properties of a track segment.

    Attributes:
        id: Unique database identifier for the section.
        section_id: Human-readable unique identifier (e.g., "JUC-LDH").
        length_km: The length of the section in kilometers.
        max_speed_kmh: The maximum permissible speed in the section in km/h.
        description: A brief description of the section.
    """
    __tablename__ = "sections"

    id = Column(Integer, primary_key=True, index=True)
    section_id = Column(String(50), unique=True, index=True, nullable=False)
    length_km = Column(Float, nullable=False)
    max_speed_kmh = Column(Integer, nullable=False)
    description = Column(String(255), nullable=True)

    def __repr__(self) -> str:
        return f"<Section(section_id={self.section_id}, length_km={self.length_km})>"