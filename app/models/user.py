from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.orm import relationship
from app.db.base import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    role = Column(String, nullable=False, default="Controller")
    phone = Column(String, nullable=True)
    section_id = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)

    # Relationship
    settings = relationship("UserSettings", back_populates="user", uselist=False)
