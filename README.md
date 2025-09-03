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

### Schedule Management
- `POST /api/schedule/optimize`: Run the optimization engine for a given list of trains.
- `POST /api/schedule/whatif`: Perform what-if analysis for disruption scenarios.
- `GET /api/schedule/current`: Retrieve the current, active, and optimized train schedule.
- `POST /api/schedule/override`: Allow a controller to manually override an AI-generated schedule.

### Metrics & KPIs
- `GET /api/metrics/`: Get a snapshot of current system performance metrics, alerts, and recommendations.
- `GET /api/metrics/history`: Retrieve historical metrics data.
- `GET /api/metrics/summary`: Get an aggregated summary of metrics over a specified period.

### Real-time Updates
- `WS /ws/updates`: WebSocket endpoint for broadcasting real-time schedule changes and train movements.

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
