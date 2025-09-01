"""
Metrics database model for system performance tracking.
"""
from sqlalchemy import Column, Integer, Float, Date, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Metrics(Base):
    """
    Metrics model for tracking system performance KPIs.

    Attributes:
        metric_id: Unique metric identifier  
        date: Date for this metric record
        average_delay: Average delay in minutes for the day
        throughput: Number of trains processed
        utilization: Platform/section utilization percentage
        ai_decisions: Number of AI recommendations followed
        override_decisions: Number of manual overrides
        total_trains: Total trains processed
        optimization_time: Average optimization computation time
        accuracy_score: Prediction accuracy score
    """
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_id = Column(Integer, unique=True, index=True, nullable=False)
    date = Column(Date, nullable=False, index=True)
    average_delay = Column(Float, default=0.0)
    throughput = Column(Integer, default=0)
    utilization = Column(Float, default=0.0)
    ai_decisions = Column(Integer, default=0)
    override_decisions = Column(Integer, default=0)
    total_trains = Column(Integer, default=0)
    optimization_time = Column(Float, default=0.0)
    accuracy_score = Column(Float, default=0.0)
    created_at = Column(DateTime, nullable=False)

    @property
    def ai_vs_override_ratio(self) -> float:
        """Calculate percentage of AI decisions vs overrides."""
        total_decisions = self.ai_decisions + self.override_decisions
        if total_decisions == 0:
            return 0.0
        return (self.ai_decisions / total_decisions) * 100.0

    def __repr__(self) -> str:
        return f"<Metrics(date={self.date}, throughput={self.throughput}, avg_delay={self.average_delay})>"
