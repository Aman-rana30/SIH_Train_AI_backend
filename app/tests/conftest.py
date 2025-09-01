"""
Test configuration and fixtures for train traffic control system.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.base import Base
from app.core.dependencies import get_db

# Create test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
def db_session():
    """Create database session for testing."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_trains():
    """Sample train data for testing."""
    from datetime import datetime, timedelta

    base_time = datetime.now().replace(microsecond=0)

    return [
        {
            "train_id": "EXP001",
            "type": "Express",
            "arrival_time": base_time + timedelta(hours=1),
            "departure_time": base_time + timedelta(hours=1, minutes=15),
            "section_id": "SEC01",
            "platform_need": "P1",
            "priority": 9,
            "origin": "Central Station",
            "destination": "North Terminal"
        },
        {
            "train_id": "PASS002", 
            "type": "Passenger",
            "arrival_time": base_time + timedelta(hours=1, minutes=20),
            "departure_time": base_time + timedelta(hours=1, minutes=35),
            "section_id": "SEC01",
            "platform_need": "P2",
            "priority": 6,
            "origin": "West Junction",
            "destination": "East Station"
        },
        {
            "train_id": "FRT003",
            "type": "Freight",
            "arrival_time": base_time + timedelta(hours=2),
            "departure_time": base_time + timedelta(hours=2, minutes=30),
            "section_id": "SEC02",
            "platform_need": "P3",
            "priority": 3,
            "origin": "Cargo Terminal",
            "destination": "Industrial Zone"
        },
        {
            "train_id": "LOC004",
            "type": "Local",
            "arrival_time": base_time + timedelta(hours=1, minutes=10),
            "departure_time": base_time + timedelta(hours=1, minutes=22),
            "section_id": "SEC01",
            "platform_need": "P1",
            "priority": 4,
            "origin": "Suburb Station",
            "destination": "City Center"
        },
        {
            "train_id": "EXP005",
            "type": "Express",
            "arrival_time": base_time + timedelta(hours=2, minutes=15),
            "departure_time": base_time + timedelta(hours=2, minutes=25),
            "section_id": "SEC03",
            "platform_need": "P1",
            "priority": 10,
            "origin": "Airport Terminal",
            "destination": "Downtown"
        }
    ]
