"""
Test configuration and fixtures for train traffic control system.
"""
import pytest
import os
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Set testing environment variable
os.environ["TESTING"] = "true"

# Create test database engine first
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Import app after setting up test database
from app.main import app
from app.db.models import Base  # Import from models file
from app.core.dependencies import get_db

# Override the database dependency
def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

# Create all tables in the test database
Base.metadata.create_all(bind=engine)


@pytest.fixture(scope="session")
def db_session():
    """Create database session for testing."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="session")
def client():
    """Create test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def cleanup_db():
    """Clean up database after each test."""
    yield
    # Clear all data but keep tables
    with engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
        conn.commit()


@pytest.fixture
def sample_trains():
    """Sample train data for testing."""
    from datetime import datetime, timedelta

    base_time = datetime.now().replace(microsecond=0)

    return [
        {
            "train_id": "EXP001",
            "type": "Express",
            "arrival_time": (base_time + timedelta(hours=1)).isoformat(),
            "departure_time": (base_time + timedelta(hours=1, minutes=15)).isoformat(),
            "section_id": "SEC01",
            "platform_need": "P1",
            "priority": 9,
            "origin": "Central Station",
            "destination": "North Terminal"
        },
        {
            "train_id": "PASS002", 
            "type": "Passenger",
            "arrival_time": (base_time + timedelta(hours=1, minutes=20)).isoformat(),
            "departure_time": (base_time + timedelta(hours=1, minutes=35)).isoformat(),
            "section_id": "SEC01",
            "platform_need": "P2",
            "priority": 6,
            "origin": "West Junction",
            "destination": "East Station"
        },
        {
            "train_id": "FRT003",
            "type": "Freight",
            "arrival_time": (base_time + timedelta(hours=2)).isoformat(),
            "departure_time": (base_time + timedelta(hours=2, minutes=30)).isoformat(),
            "section_id": "SEC02",
            "platform_need": "P3",
            "priority": 3,
            "origin": "Cargo Terminal",
            "destination": "Industrial Zone"
        },
        {
            "train_id": "LOC004",
            "type": "Local",
            "arrival_time": (base_time + timedelta(hours=1, minutes=10)).isoformat(),
            "departure_time": (base_time + timedelta(hours=1, minutes=22)).isoformat(),
            "section_id": "SEC01",
            "platform_need": "P1",
            "priority": 4,
            "origin": "Suburb Station",
            "destination": "City Center"
        },
        {
            "train_id": "EXP005",
            "type": "Express",
            "arrival_time": (base_time + timedelta(hours=2, minutes=15)).isoformat(),
            "departure_time": (base_time + timedelta(hours=2, minutes=25)).isoformat(),
            "section_id": "SEC03",
            "platform_need": "P1",
            "priority": 10,
            "origin": "Airport Terminal",
            "destination": "Downtown"
        }
    ]
