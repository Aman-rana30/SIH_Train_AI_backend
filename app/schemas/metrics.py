"""
Pydantic schemas for metrics-related API operations.
"""
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional, Dict, List, Any


class MetricsBase(BaseModel):
    """Base metrics schema."""
    date: datetime = Field(..., description="Date for metrics")
    average_delay: float = Field(default=0.0, ge=0, description="Average delay in minutes")
    throughput: int = Field(default=0, ge=0, description="Number of trains processed")
    utilization: float = Field(default=0.0, ge=0, le=100, description="Utilization percentage")
    ai_decisions: int = Field(default=0, ge=0, description="Number of AI decisions")
    override_decisions: int = Field(default=0, ge=0, description="Number of overrides")
    total_trains: int = Field(default=0, ge=0, description="Total trains processed")
    optimization_time: float = Field(default=0.0, ge=0, description="Average optimization time")
    accuracy_score: float = Field(default=0.0, ge=0, le=1, description="Prediction accuracy")


class MetricsCreate(MetricsBase):
    """Schema for creating metrics."""
    pass


class Metrics(MetricsBase):
    """Complete metrics schema for responses."""
    id: int
    metric_id: int
    created_at: datetime
    ai_vs_override_ratio: float = Field(..., description="Percentage of AI vs override decisions")

    model_config = ConfigDict(from_attributes=True)


class KPIResponse(BaseModel):
    """Schema for KPI dashboard response."""
    current_metrics: Metrics
    trends: Dict[str, Any] = Field(..., description="Historical trends")
    alerts: List[str] = Field(default_factory=list, description="System alerts")
    recommendations: List[str] = Field(default_factory=list, description="Optimization recommendations")


class MetricsFilter(BaseModel):
    """Schema for filtering metrics queries."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_throughput: Optional[int] = None
    max_delay: Optional[float] = None
