"""
Database base configuration and model imports.
"""
from sqlalchemy.ext.declarative import declarative_base

# Import all the models, so that Base has them before being
# imported by Alembic
from app.models.train import Train  # noqa
from app.models.schedule import Schedule  # noqa  
from app.models.override import Override  # noqa
from app.models.metrics import Metrics  # noqa

Base = declarative_base()
