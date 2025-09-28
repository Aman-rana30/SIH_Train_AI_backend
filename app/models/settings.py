from sqlalchemy import Column, Integer, String, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base import Base

class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    section = Column(String, nullable=False, default="")
    timezone = Column(String, nullable=False, default="Asia/Kolkata")
    language = Column(String, nullable=False, default="en")
    notifications = Column(JSON, nullable=False, default={
        "train_delay_alerts": True,
        "disruption_alerts": True,
        "maintenance_alerts": False,
        "system_updates": True,
        "in_app": True,
        "email": False,
        "sms": False,
    })

    # Relationship
    user = relationship("User", back_populates="settings")
