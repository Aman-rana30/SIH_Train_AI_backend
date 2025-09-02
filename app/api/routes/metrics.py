"""
API routes for system performance metrics and KPIs.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from app.core.dependencies import get_db
from app.schemas.metrics import Metrics as MetricsSchema, KPIResponse, MetricsFilter
from app.models.metrics import Metrics as MetricsModel
from app.models.schedule import Schedule as ScheduleModel, ScheduleStatus
from app.models.override import Override as OverrideModel

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_model=KPIResponse)
async def get_metrics(
    target_date: Optional[str] = Query(None, description="Date for metrics in YYYY-MM-DD format (default: today)"),
    db: Session = Depends(get_db)
):
    """
    Get system performance metrics and KPIs.

    Args:
        target_date: Date for metrics in YYYY-MM-DD format (defaults to today)
        db: Database session

    Returns:
        KPI response with current metrics, trends, and alerts
    """
    if not target_date:
        target_date_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        try:
            # Parse the date string to datetime
            target_date_dt = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid date format. Use YYYY-MM-DD format."
            )

    logger.info(f"Fetching metrics for {target_date_dt}")

    try:
        # Get or create today's metrics
        current_metrics = db.query(MetricsModel).filter(
            MetricsModel.date == target_date_dt
        ).first()

        if not current_metrics:
            # Calculate metrics for the day
            current_metrics = await _calculate_daily_metrics(db, target_date_dt)

        # Get trends (last 7 days)
        trends = await _calculate_trends(db, target_date_dt)

        # Generate alerts
        alerts = _generate_alerts(current_metrics, trends)

        # Generate recommendations
        recommendations = _generate_recommendations(current_metrics, trends)

        return KPIResponse(
            current_metrics=MetricsSchema.from_orm(current_metrics),
            trends=trends,
            alerts=alerts,
            recommendations=recommendations
        )

    except Exception as e:
        logger.error(f"Failed to fetch metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch metrics: {str(e)}"
        )


@router.get("/history", response_model=List[MetricsSchema])
async def get_metrics_history(
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    limit: int = Query(30, ge=1, le=365, description="Number of days to return"),
    db: Session = Depends(get_db)
):
    """
    Get historical metrics data.

    Args:
        start_date: Start date for metrics range
        end_date: End date for metrics range  
        limit: Maximum number of records to return
        db: Database session

    Returns:
        List of historical metrics
    """
    try:
        query = db.query(MetricsModel)

        if start_date:
            query = query.filter(MetricsModel.date >= start_date)
        if end_date:
            query = query.filter(MetricsModel.date <= end_date)

        metrics = query.order_by(MetricsModel.date.desc()).limit(limit).all()

        return [MetricsSchema.from_orm(metric) for metric in metrics]

    except Exception as e:
        logger.error(f"Failed to fetch metrics history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch metrics history"
        )


@router.get("/summary", response_model=dict)
async def get_metrics_summary(
    days: int = Query(7, ge=1, le=30, description="Number of days for summary"),
    db: Session = Depends(get_db)
):
    """
    Get aggregated metrics summary.

    Args:
        days: Number of days to include in summary
        db: Database session

    Returns:
        Aggregated metrics summary
    """
    try:
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=days-1)

        # Get metrics for the period
        metrics = db.query(MetricsModel).filter(
            and_(
                MetricsModel.date >= start_date,
                MetricsModel.date <= end_date
            )
        ).all()

        if not metrics:
            return {
                "period": f"{start_date} to {end_date}",
                "total_days": days,
                "data_available": 0,
                "summary": {}
            }

        # Calculate aggregated metrics
        total_trains = sum(m.total_trains for m in metrics)
        total_delays = sum(m.average_delay * m.total_trains for m in metrics)
        total_ai_decisions = sum(m.ai_decisions for m in metrics)
        total_overrides = sum(m.override_decisions for m in metrics)

        avg_delay = total_delays / total_trains if total_trains > 0 else 0
        avg_throughput = total_trains / len(metrics)
        avg_utilization = sum(m.utilization for m in metrics) / len(metrics)
        ai_decision_rate = total_ai_decisions / (total_ai_decisions + total_overrides) * 100 if (total_ai_decisions + total_overrides) > 0 else 0

        summary = {
            "period": f"{start_date} to {end_date}",
            "total_days": days,
            "data_available": len(metrics),
            "summary": {
                "total_trains_processed": total_trains,
                "average_delay_minutes": round(avg_delay, 2),
                "average_daily_throughput": round(avg_throughput, 1),
                "average_utilization_percent": round(avg_utilization, 1),
                "ai_decision_rate_percent": round(ai_decision_rate, 1),
                "total_ai_decisions": total_ai_decisions,
                "total_overrides": total_overrides
            }
        }

        return summary

    except Exception as e:
        logger.error(f"Failed to get metrics summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics summary: {str(e)}"
        )


async def _calculate_daily_metrics(db: Session, target_date: datetime) -> MetricsModel:
    """Calculate metrics for a specific date."""
    start_datetime = datetime.combine(target_date.date(), datetime.min.time())
    end_datetime = start_datetime + timedelta(days=1)

    # Get schedules for the day
    schedules = db.query(ScheduleModel).filter(
        and_(
            ScheduleModel.created_at >= start_datetime,
            ScheduleModel.created_at < end_datetime
        )
    ).all()

    # Get overrides for the day  
    overrides = db.query(OverrideModel).filter(
        and_(
            OverrideModel.timestamp >= start_datetime,
            OverrideModel.timestamp < end_datetime
        )
    ).all()

    # Calculate metrics
    total_trains = len(schedules)
    total_delay = sum(max(0, s.delay_minutes) for s in schedules)
    avg_delay = total_delay / total_trains if total_trains > 0 else 0

    # Calculate utilization (simplified)
    utilization = min(100.0, total_trains * 10)  # Simplified calculation

    ai_decisions = total_trains - len(overrides)
    override_decisions = len(overrides)

    # Create metrics record
    metrics = MetricsModel(
        metric_id=int(target_date.strftime("%Y%m%d")),
        date=target_date,
        average_delay=avg_delay,
        throughput=total_trains,
        utilization=utilization,
        ai_decisions=max(0, ai_decisions),
        override_decisions=override_decisions,
        total_trains=total_trains,
        optimization_time=5.0,  # Default value
        accuracy_score=0.85,    # Default value
        created_at=datetime.utcnow()
    )

    db.add(metrics)
    db.commit()

    return metrics


async def _calculate_trends(db: Session, target_date: datetime) -> dict:
    """Calculate trends for the last 7 days."""
    end_date = target_date
    start_date = end_date - timedelta(days=6)

    metrics = db.query(MetricsModel).filter(
        and_(
            MetricsModel.date >= start_date,
            MetricsModel.date <= end_date
        )
    ).order_by(MetricsModel.date).all()

    if len(metrics) < 2:
        return {"trend_available": False}

    # Calculate trends
    delays = [m.average_delay for m in metrics]
    throughputs = [m.throughput for m in metrics]
    utilizations = [m.utilization for m in metrics]
    ai_rates = [m.ai_vs_override_ratio for m in metrics]

    return {
        "trend_available": True,
        "delay_trend": _calculate_trend(delays),
        "throughput_trend": _calculate_trend(throughputs),
        "utilization_trend": _calculate_trend(utilizations),
        "ai_decision_trend": _calculate_trend(ai_rates),
        "daily_data": {
            "dates": [m.date.isoformat() for m in metrics],
            "delays": delays,
            "throughputs": throughputs,
            "utilizations": utilizations,
            "ai_rates": ai_rates
        }
    }


def _calculate_trend(values: List[float]) -> str:
    """Calculate trend direction."""
    if len(values) < 2:
        return "stable"

    first_half = sum(values[:len(values)//2]) / (len(values)//2)
    second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)

    change_percent = ((second_half - first_half) / first_half * 100) if first_half != 0 else 0

    if change_percent > 5:
        return "increasing"
    elif change_percent < -5:
        return "decreasing"
    else:
        return "stable"


def _generate_alerts(metrics: MetricsModel, trends: dict) -> List[str]:
    """Generate system alerts based on metrics."""
    alerts = []

    # High delay alert
    if metrics.average_delay > 15:
        alerts.append(f"High average delay detected: {metrics.average_delay:.1f} minutes")

    # Low throughput alert
    if metrics.throughput < 10:
        alerts.append(f"Low throughput detected: {metrics.throughput} trains processed")

    # High utilization alert
    if metrics.utilization > 90:
        alerts.append(f"High utilization: {metrics.utilization:.1f}%")

    # Override rate alert
    if metrics.ai_vs_override_ratio < 70:
        alerts.append(f"High override rate: {100 - metrics.ai_vs_override_ratio:.1f}% of decisions overridden")

    # Trend alerts
    if trends.get("delay_trend") == "increasing":
        alerts.append("Delay trend increasing over last 7 days")

    if trends.get("throughput_trend") == "decreasing":
        alerts.append("Throughput trend decreasing over last 7 days")

    return alerts


def _generate_recommendations(metrics: MetricsModel, trends: dict) -> List[str]:
    """Generate optimization recommendations."""
    recommendations = []

    # Delay recommendations
    if metrics.average_delay > 10:
        recommendations.append("Consider increasing buffer times between trains")
        recommendations.append("Review priority assignments for critical trains")

    # Throughput recommendations
    if metrics.throughput < 20:
        recommendations.append("Evaluate schedule density and capacity utilization")
        recommendations.append("Consider parallel processing for non-conflicting sections")

    # Override recommendations
    if metrics.ai_vs_override_ratio < 80:
        recommendations.append("Review AI model parameters and constraints")
        recommendations.append("Provide additional training data for edge cases")

    # Trend-based recommendations
    if trends.get("utilization_trend") == "increasing":
        recommendations.append("Monitor capacity limits and plan for scaling")

    return recommendations
