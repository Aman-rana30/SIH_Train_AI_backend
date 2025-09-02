"""
Model imports for database initialization.
This file imports all models to ensure they are registered with the Base metadata.
"""
from app.db.base import Base

# Import all models here so they are registered with Base
# Import order matters for foreign key constraints
from app.models.train import Train  # Base table
from app.models.schedule import Schedule, ScheduleStatus  # References Train
from app.models.override import Override  # References Train
from app.models.metrics import Metrics  # Independent table

# Ensure all models are registered
__all__ = ["Base", "Train", "Schedule", "ScheduleStatus", "Override", "Metrics"]
