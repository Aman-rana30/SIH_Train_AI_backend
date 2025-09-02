"""
Database base configuration and imports.
"""
from sqlalchemy.orm import declarative_base

# Create declarative base instance
Base = declarative_base()

# Models will be imported separately to avoid circular dependencies
