"""
Override database model for controller decisions.
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base import Base


class Override(Base):
    """
    Override model for manual controller decisions.

    Attributes:
        override_id: Unique override identifier
        train_id: Foreign key to train
        controller_decision: Manual decision made by controller
        ai_recommendation: Original AI recommendation
        reason: Reason for override
        controller_id: ID of controller making decision
        impact_delay: Estimated delay impact in minutes
        timestamp: When override was made
    """
    __tablename__ = "overrides"

    id = Column(Integer, primary_key=True, index=True)
    override_id = Column(String(50), unique=True, index=True, nullable=False)
    train_id = Column(Integer, nullable=False)  # Temporarily removed foreign key for testing
    controller_decision = Column(Text, nullable=False)
    ai_recommendation = Column(Text)
    reason = Column(Text)
    controller_id = Column(String(50))
    impact_delay = Column(Integer, default=0)
    timestamp = Column(DateTime, nullable=False)

    # Relationships - temporarily commented out to fix circular import issues
    # train = relationship("Train", back_populates="overrides")

    def __repr__(self) -> str:
        return f"<Override(override_id={self.override_id}, train_id={self.train_id}, decision={self.controller_decision})>"
