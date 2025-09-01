# Train Traffic Control Decision-Support System

An AI-powered backend system for optimizing train scheduling and traffic control using constraint programming and operations research techniques.

## Features

- **Real-time Optimization**: Uses Google OR-Tools CP-SAT solver for train scheduling
- **RESTful API**: FastAPI-based endpoints for schedule management
- **WebSocket Support**: Real-time updates for schedule changes
- **What-if Analysis**: Simulate disruptions and get optimized responses
- **Override Management**: Allow manual controller overrides with tracking
- **Performance Metrics**: Monitor system KPIs and decision effectiveness

## Tech Stack

- **Backend**: Python 3.10+ with FastAPI
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Optimization**: Google OR-Tools (CP-SAT), PuLP
- **Real-time**: WebSocket connections
- **Testing**: Pytest with async support

## Installation

### Using Poetry (Recommended)
```bash
pip install poetry
poetry install
poetry shell
```

### Using pip
```bash
pip install -r requirements.txt
```

## Database Setup

1. Install PostgreSQL and create a database
2. Set environment variables:
```bash
export DATABASE_URL="postgresql://user:password@localhost/train_control"
```

3. Run migrations:
```bash
alembic upgrade head
```

## Running the Application

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /api/schedule/optimize` - Optimize train schedules
- `POST /api/schedule/whatif` - What-if analysis for disruptions
- `GET /api/schedule/current` - Get current optimized schedule
- `POST /api/schedule/override` - Manual controller override
- `GET /api/metrics` - System performance metrics
- `WS /ws/updates` - Real-time schedule updates

## Testing

```bash
pytest
```

## Project Structure

```
train_traffic_control/
├── app/
│   ├── main.py              # FastAPI application
│   ├── api/routes/          # API endpoint routes
│   ├── core/               # Configuration and dependencies
│   ├── models/             # SQLAlchemy database models
│   ├── schemas/            # Pydantic schemas
│   ├── services/           # Business logic and optimization
│   ├── db/                 # Database configuration
│   └── tests/              # Test cases
├── alembic/                # Database migrations
└── data/                   # Sample data and examples
```
