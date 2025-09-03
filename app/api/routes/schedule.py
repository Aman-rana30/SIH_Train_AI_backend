"""
API routes for train schedule management and optimization.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from datetime import datetime, timedelta, timezone
import logging
from pydantic import BaseModel, ConfigDict

from app.core.dependencies import get_db
from app.schemas.train import OptimizationRequest, WhatIfRequest
# Import the broadcast function from the websocket routes
from app.api.routes.websocket import broadcast_optimization_complete

# Import proper schemas to avoid circular imports
from app.schemas.schedule import Schedule, ScheduleCreate, OptimizationResult
from app.schemas.override import Override, OverrideRequest

# Import the SQLAlchemy model and Enum
from app.models.train import Train as TrainModel, TrainType as TrainTypeModel
from app.models.schedule import Schedule as ScheduleModel, ScheduleStatus
from app.models.override import Override as OverrideModel
from app.schemas.train import Train as TrainSchema
from app.services.optimization.optimizer import TrainSchedulingOptimizer
from app.services.optimization.models import (
    TrainData, OptimizationInput, DisruptionEvent
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/optimize")
async def optimize_schedule(
    request: OptimizationRequest,
    db: Session = Depends(get_db)
):
    """
    Optimize train schedules using OR-Tools constraint programming.

    Args:
        request: List of trains and optimization parameters
        db: Database session

    Returns:
        Optimized schedules with metrics
    """
    logger.info(f"Received optimization request for {len(request.trains)} trains")

    # Validate request
    if not request.trains:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Trains list cannot be empty"
        )

    try:
        # Convert request to optimization input
        train_data = []
        for train_req in request.trains:
            # --- START FIX: Ensure train exists in DB or create it ---
            train_in_db = db.query(TrainModel).filter(TrainModel.train_id == train_req.train_id).first()
            if not train_in_db:
                # Convert pydantic model to dict for DB creation
                train_data_for_db = train_req.model_dump()
                # Correctly convert the Pydantic Enum to the SQLAlchemy Enum
                train_data_for_db['type'] = TrainTypeModel[train_req.type.name]
                
                train_in_db = TrainModel(**train_data_for_db)
                db.add(train_in_db)
                db.commit()
                db.refresh(train_in_db)
            # --- END FIX ---
            
            train_data.append(TrainData(
                train_id=train_req.train_id,
                type=train_req.type.value,
                arrival_time=train_req.arrival_time,
                departure_time=train_req.departure_time,
                section_id=train_req.section_id,
                platform_need=train_req.platform_need,
                priority=train_req.priority,
                origin=train_req.origin,
                destination=train_req.destination
            ))

        # Set time horizon
        min_time = min(train.arrival_time for train in train_data)
        max_time = max(train.departure_time for train in train_data)
        time_horizon = (
            min_time - timedelta(hours=1),
            max_time + timedelta(hours=2)
        )

        optimization_input = OptimizationInput(
            trains=train_data,
            sections=[],  # Simplified for demo
            platforms=[],
            time_horizon=time_horizon,
            constraints=request.optimization_params or {}
        )

        # Run optimization
        optimizer = TrainSchedulingOptimizer(time_limit_seconds=30)
        result = optimizer.optimize_schedule(optimization_input)

        # Generate unique optimization run ID
        optimization_run_id = str(uuid.uuid4())

        # Save schedules to database
        saved_schedules = []
        for schedule_data in result.schedules:
            # --- START FIX: Look up the train to get its database ID ---
            train = db.query(TrainModel).filter(TrainModel.train_id == schedule_data['train_id']).first()
            if not train:
                logger.warning(f"Could not find train {schedule_data['train_id']} in the database to save schedule.")
                continue
            
            schedule_create = ScheduleCreate(
                schedule_id=f"{optimization_run_id}_{schedule_data['train_id']}",
                train_id=train.id, # Use the actual database ID
                planned_time=schedule_data['original_arrival'],
                optimized_time=schedule_data['optimized_arrival'],
                section_id=schedule_data['section_id'],
                platform_id=schedule_data['platform_need'],
                delay_minutes=schedule_data['delay_minutes'],
                optimization_run_id=optimization_run_id,
                status=ScheduleStatus.WAITING
            )
            # --- END FIX ---

            # Create database record
            db_schedule = ScheduleModel(
                schedule_id=schedule_create.schedule_id,
                train_id=schedule_create.train_id,
                planned_time=schedule_create.planned_time,
                optimized_time=schedule_create.optimized_time,
                section_id=schedule_create.section_id,
                platform_id=schedule_create.platform_id,
                delay_minutes=schedule_create.delay_minutes,
                optimization_run_id=schedule_create.optimization_run_id,
                status=schedule_create.status,
                created_at=datetime.now(timezone.utc)
            )

            db.add(db_schedule)
            db.flush() # Use flush to get the ID before commit
            saved_schedules.append(Schedule.model_validate(db_schedule))

        db.commit()

        logger.info(f"Optimization completed in {result.computation_time:.2f}s with status {result.status}")

        final_result = OptimizationResult(
            optimization_run_id=optimization_run_id,
            schedules=saved_schedules,
            metrics=result.metrics,
            computation_time=result.computation_time,
            status=result.status
        )

        # Broadcast the result to all connected WebSocket clients
        await broadcast_optimization_complete(final_result.model_dump(mode='json'))

        return final_result

    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )


@router.post("/whatif")
async def whatif_analysis(
    request: WhatIfRequest,
    db: Session = Depends(get_db)
):
    """
    Perform what-if analysis for disruption scenarios.

    Args:
        request: Disruption parameters and affected trains
        db: Database session

    Returns:
        Updated optimization results
    """
    logger.info("Received what-if analysis request")

    try:
        # Get current schedule from database
        current_schedules = db.query(ScheduleModel).filter(
            ScheduleModel.status.in_([ScheduleStatus.WAITING, ScheduleStatus.MOVING])
        ).all()

        if not current_schedules:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active schedules found"
            )

        # Convert to dict format
        current_schedule_data = []
        for schedule in current_schedules:
            current_schedule_data.append({
                'train_id': str(schedule.train_id),
                'optimized_arrival': schedule.optimized_time,
                'optimized_departure': schedule.optimized_time + timedelta(minutes=10),  # Simplified
                'section_id': schedule.section_id,
                'platform_need': schedule.platform_id or 'P1',
                'priority': 5  # Default priority
            })

        # Create disruption event
        disruption = DisruptionEvent(
            event_type=request.disruption.get('type', 'delay'),
            affected_trains=request.affected_trains or [],
            delay_minutes=request.disruption.get('delay_minutes', 0),
            start_time=datetime.now(timezone.utc),
            duration_minutes=request.disruption.get('duration_minutes', 60),
            description=request.disruption.get('description', 'Unknown disruption')
        )

        # Run what-if analysis
        optimizer = TrainSchedulingOptimizer(time_limit_seconds=20)
        result = optimizer.what_if_analysis(current_schedule_data, disruption)

        # Generate what-if run ID
        whatif_run_id = f"whatif_{str(uuid.uuid4())[:8]}"

        # Save what-if results (marked as temporary)
        saved_schedules = []
        for schedule_data in result.schedules:
            schedule = Schedule(
                id=0,  # Temporary ID
                schedule_id=f"{whatif_run_id}_{schedule_data['train_id']}",
                train_id=1,
                planned_time=schedule_data['original_arrival'],
                optimized_time=schedule_data['optimized_arrival'],
                section_id=schedule_data['section_id'],
                platform_id=schedule_data['platform_need'],
                delay_minutes=schedule_data['delay_minutes'],
                optimization_run_id=whatif_run_id,
                status=ScheduleStatus.WAITING,
                created_at=datetime.now(timezone.utc)
            )
            saved_schedules.append(schedule)

        logger.info(f"What-if analysis completed: {len(saved_schedules)} schedules analyzed")

        # Return the OptimizationResult object for frontend rendering
        return OptimizationResult(
            optimization_run_id=whatif_run_id,
            schedules=saved_schedules,
            metrics=result.metrics,
            computation_time=result.computation_time,
            status=f"WHATIF_{result.status}"
        )

    except Exception as e:
        logger.error(f"What-if analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"What-if analysis failed: {str(e)}"
        )


@router.get("/current")
async def get_current_schedule(db: Session = Depends(get_db)):
    """
    Get the current optimized schedule from database.

    Args:
        db: Database session

    Returns:
        List of current active schedules with full train details
    """
    try:
        # Join with train table to get full train details
        schedules = db.query(ScheduleModel).join(
            TrainModel, ScheduleModel.train_id == TrainModel.id
        ).filter(
            ScheduleModel.status.in_([
                ScheduleStatus.WAITING, 
                ScheduleStatus.MOVING
            ])
        ).order_by(ScheduleModel.optimized_time).all()

        # Create schedule objects with train details
        result_schedules = []
        for schedule in schedules:
            # Get the associated train
            train = db.query(TrainModel).filter(TrainModel.id == schedule.train_id).first()
            # Create schedule dict with train details
            train_dict = None
            if train:
                train_dict = {
                    "id": train.id,
                    "train_id": train.train_id,
                    "type": train.type.value if hasattr(train.type, 'value') else str(train.type),
                    "origin": train.origin,
                    "destination": train.destination,
                    "priority": train.priority,
                    "capacity": train.capacity,
                    "active": train.active,
                    "arrival_time": train.arrival_time.isoformat() if hasattr(train, 'arrival_time') and train.arrival_time else None,
                    "departure_time": train.departure_time.isoformat() if hasattr(train, 'departure_time') and train.departure_time else None,
                    "platform_need": train.platform_need
                }
            schedule_dict = {
                "id": schedule.id,
                "schedule_id": schedule.schedule_id,
                "train_id": schedule.train_id,
                "planned_time": schedule.planned_time,
                "optimized_time": schedule.optimized_time,
                "section_id": schedule.section_id,
                "platform_id": schedule.platform_id,
                "status": schedule.status.value if hasattr(schedule.status, 'value') else str(schedule.status),
                "delay_minutes": schedule.delay_minutes,
                "optimization_run_id": schedule.optimization_run_id,
                "created_at": schedule.created_at,
                "updated_at": schedule.updated_at,
                "train": train_dict
            }
            result_schedules.append(schedule_dict)

        return result_schedules

    except Exception as e:
        logger.error(f"Failed to fetch current schedule: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch current schedule"
        )


@router.post("/override")
async def override_decision(
    request: OverrideRequest,
    db: Session = Depends(get_db)
):
    """
    Allow controller to override AI recommendation.

    Args:
        request: Override decision parameters
        db: Database session

    Returns:
        Created override record
    """
    logger.info(f"Received override request for train {request.train_id}")

    try:
        # Check if train exists
        train = db.query(TrainModel).filter(TrainModel.id == request.train_id).first()
        if not train:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Train not found"
            )

        # Get current AI recommendation for this train
        current_schedule = db.query(ScheduleModel).filter(
            ScheduleModel.train_id == request.train_id
        ).order_by(ScheduleModel.created_at.desc()).first()

        ai_recommendation = ""
        if current_schedule:
            ai_recommendation = f"AI recommended: {current_schedule.optimized_time}"

        # Create override record
        override_record = OverrideModel(
            override_id=str(uuid.uuid4()),
            train_id=request.train_id,
            controller_decision=request.decision,
            ai_recommendation=ai_recommendation,
            reason=request.reason,
            timestamp=datetime.now(timezone.utc),
            controller_id=request.controller_id,
            impact_delay=0  # Could calculate this based on new_schedule_time
        )

        db.add(override_record)

        # Update schedule if new time provided
        if request.new_schedule_time and current_schedule:
            current_schedule.optimized_time = request.new_schedule_time
            current_schedule.updated_at = datetime.now(timezone.utc)
            current_schedule.status = ScheduleStatus.MOVING

        db.commit()

        logger.info(f"Override created for train {request.train_id}")
        return Override.model_validate(override_record)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create override: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create override: {str(e)}"
        )