"""
Override database model for manual controller decisions.
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Override(Base):
    """
    Override model for tracking manual controller overrides of AI recommendations.

    Attributes:
        override_id: Unique override identifier
        train_id: Foreign key to affected train
        controller_decision: Description of manual decision made
        ai_recommendation: What the AI system recommended
        reason: Reason for override
        impact_delay: Estimated delay impact in minutes
        timestamp: When override was made
        controller_id: ID of controller who made override
    """
    __tablename__ = "overrides"

    id = Column(Integer, primary_key=True, index=True)
    override_id = Column(String(50), unique=True, index=True, nullable=False)
    train_id = Column(Integer, ForeignKey("trains.id"), nullable=False)
    controller_decision = Column(Text, nullable=False)
    ai_recommendation = Column(Text)
    reason = Column(Text)
    impact_delay = Column(Integer, default=0)
    timestamp = Column(DateTime, nullable=False)
    controller_id = Column(String(50))

    # Relationships
    train = relationship("Train", back_populates="overrides")

    def __repr__(self) -> str:
        return f"<Override(override_id={self.override_id}, train_id={self.train_id})>"
