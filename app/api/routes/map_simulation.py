"""
Map simulation API routes integrated into Train AI Backend.
Provides real-time train positions, railway tracks, and conflict detection.
"""

from fastapi import APIRouter, HTTPException, status
from typing import List, Optional, Dict
import requests
import logging
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from pydantic import BaseModel

# Indian Standard Time timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

router = APIRouter()
logger = logging.getLogger(__name__)

# Pydantic models for map simulation
class TrackPoint(BaseModel):
    lat: float
    lon: float

class TrackResponse(BaseModel):
    track_polyline: List[TrackPoint]
    total_points: int
    section: str

class Station(BaseModel):
    name: str
    lat: float
    lon: float
    city: Optional[str] = None
    state: Optional[str] = None
    type: str

class StationsResponse(BaseModel):
    stations: List[Station]
    total_stations: int

class TrainPriority(str, Enum):
    EXPRESS = "express"
    PASSENGER = "passenger"
    FREIGHT = "freight"

class TrainPosition(BaseModel):
    train_id: str
    train_name: str
    priority: TrainPriority
    current_lat: float
    current_lon: float
    progress: float  # 0.0 to 1.0 along the route
    speed: float  # km/h
    status: str  # "moving", "delayed", "stopped"
    is_conflicted: bool
    next_junction: Optional[str] = None
    eta_next_junction: Optional[datetime] = None

class TrainPositionsResponse(BaseModel):
    trains: List[TrainPosition]
    total_trains: int
    simulation_time: datetime

# Configuration
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"

# Bounding box for Jalandhar to Ludhiana (south, west, north, east)
BBOX = "30.8,75.4,31.4,75.9"

# Mock data for fallback when APIs are unavailable
MOCK_TRACK_DATA = [
    {"lat": 31.3260, "lon": 75.5762},  # Jalandhar
    {"lat": 31.3200, "lon": 75.5900},
    {"lat": 31.3150, "lon": 75.6100},
    {"lat": 31.3100, "lon": 75.6300},
    {"lat": 31.3050, "lon": 75.6500},
    {"lat": 31.3000, "lon": 75.6700},
    {"lat": 31.2950, "lon": 75.6900},
    {"lat": 31.2900, "lon": 75.7100},
    {"lat": 31.2850, "lon": 75.7300},
    {"lat": 31.2800, "lon": 75.7500},
    {"lat": 31.2750, "lon": 75.7700},
    {"lat": 31.2700, "lon": 75.7900},
    {"lat": 31.2650, "lon": 75.8100},
    {"lat": 31.2600, "lon": 75.8300},
    {"lat": 31.2550, "lon": 75.8500},
    {"lat": 31.2500, "lon": 75.8700},
    {"lat": 31.1814, "lon": 75.8458}   # Ludhiana
]

MOCK_STATIONS_DATA = [
    {
        "name": "Jalandhar City Junction",
        "lat": 31.3260,
        "lon": 75.5762,
        "city": "Jalandhar",
        "state": "Punjab",
        "type": "junction"
    },
    {
        "name": "Nakodar Junction",
        "lat": 31.1250,
        "lon": 75.4750,
        "city": "Nakodar",
        "state": "Punjab",
        "type": "junction"
    },
    {
        "name": "Phillaur Junction",
        "lat": 31.0180,
        "lon": 75.7850,
        "city": "Phillaur",
        "state": "Punjab",
        "type": "junction"
    },
    {
        "name": "Apra Mandi",
        "lat": 31.2450,
        "lon": 75.7200,
        "city": "Apra",
        "state": "Punjab",
        "type": "station"
    },
    {
        "name": "Ludhiana Junction",
        "lat": 31.1814,
        "lon": 75.8458,
        "city": "Ludhiana",
        "state": "Punjab",
        "type": "junction"
    }
]

# Global simulation state
SIMULATION_START_TIME = datetime.now(IST)
SIMULATION_SPEED_MULTIPLIER = 1.0

# Overpass queries
TRACK_QUERY = f"""
[out:json][timeout:60];
(
  way["railway"="rail"]({BBOX});
  way["railway"="light_rail"]({BBOX});
  way["railway"="narrow_gauge"]({BBOX});
);
out geom;
"""

STATIONS_QUERY = f"""
[out:json][timeout:60];
(
  node["railway"="station"]({BBOX});
  node["railway"="junction"]({BBOX});
  node["railway"="halt"]({BBOX});
  node["public_transport"="station"]["railway"]({BBOX});
);
out meta;
"""

def make_overpass_request(query: str, retries: int = 3) -> dict:
    """Make request to Overpass API with retry logic and better error handling"""
    for attempt in range(retries):
        try:
            logger.info(f"Attempting Overpass API request (attempt {attempt + 1}/{retries})")
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=30,
                headers={"User-Agent": "Train AI Backend/1.0"}
            )
            response.raise_for_status()
            data = response.json()
            logger.info("Successfully fetched real OSM data from Overpass API")
            return data
        except requests.exceptions.Timeout:
            logger.warning(f"Overpass API timeout on attempt {attempt + 1}")
            if attempt < retries - 1:
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Overpass API error on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    
    logger.error("All Overpass API attempts failed - will use fallback data")
    return None

@router.get("/track", response_model=TrackResponse)
async def get_railway_track():
    """Get railway track polyline between Jalandhar and Ludhiana"""
    logger.info("Fetching railway track data")
    
    try:
        # Try to get track data from Overpass API
        data = make_overpass_request(TRACK_QUERY)
        
        track_points = []
        
        if data and data.get("elements"):
            # Process each way (track segment)
            for element in data.get("elements", []):
                if element.get("type") == "way" and "geometry" in element:
                    for point in element["geometry"]:
                        track_points.append(TrackPoint(
                            lat=point["lat"],
                            lon=point["lon"]
                        ))
        
        # Use mock data only if no real data available
        if not track_points:
            logger.warning("No real OSM track data found - using mock data as last resort")
            track_points = [TrackPoint(lat=point["lat"], lon=point["lon"]) for point in MOCK_TRACK_DATA]
        
        logger.info(f"Returning {len(track_points)} track points")
        
        return TrackResponse(
            track_polyline=track_points,
            total_points=len(track_points),
            section="Jalandhar to Ludhiana"
        )
        
    except Exception as e:
        logger.error(f"Error processing track data: {e}")
        # Fallback to mock data on any error
        logger.info("Using mock track data as fallback")
        track_points = [TrackPoint(lat=point["lat"], lon=point["lon"]) for point in MOCK_TRACK_DATA]
        return TrackResponse(
            track_polyline=track_points,
            total_points=len(track_points),
            section="Jalandhar to Ludhiana"
        )

@router.get("/stations", response_model=StationsResponse)
async def get_railway_stations():
    """Get railway stations and junctions with city/state information"""
    logger.info("Fetching railway stations data")
    
    try:
        # Try to get stations data from Overpass API
        data = make_overpass_request(STATIONS_QUERY)
        
        stations = []
        
        if data and data.get("elements"):
            # Process each station/junction
            for element in data.get("elements", []):
                if element.get("type") == "node":
                    tags = element.get("tags", {})
                    
                    # Get station name
                    name = (tags.get("name") or 
                           tags.get("railway:name") or
                           f"Station at {element['lat']:.4f}, {element['lon']:.4f}")
                    
                    # Determine type
                    station_type = tags.get("railway", "station")
                    
                    # Get coordinates
                    lat = element["lat"]
                    lon = element["lon"]
                    
                    # Get city and state (simplified for integration)
                    city = tags.get("addr:city", "Unknown")
                    state = tags.get("addr:state", "Punjab")
                    
                    stations.append(Station(
                        name=name,
                        lat=lat,
                        lon=lon,
                        city=city,
                        state=state,
                        type=station_type
                    ))
        
        # Use mock data if no real data available
        if not stations:
            logger.info("Using mock stations data")
            stations = [Station(**station_data) for station_data in MOCK_STATIONS_DATA]
        
        logger.info(f"Returning {len(stations)} stations")
        
        return StationsResponse(
            stations=stations,
            total_stations=len(stations)
        )
        
    except Exception as e:
        logger.error(f"Error processing stations data: {e}")
        # Fallback to mock data on any error
        logger.info("Using mock stations data as fallback")
        stations = [Station(**station_data) for station_data in MOCK_STATIONS_DATA]
        return StationsResponse(
            stations=stations,
            total_stations=len(stations)
        )

@router.get("/train-positions", response_model=TrainPositionsResponse)
async def get_train_positions():
    """
    Get real-time positions of all trains integrated with AI backend data.
    This endpoint bridges the map simulation with actual train schedule data.
    """
    logger.info("Fetching real-time train positions from AI backend integration")
    
    try:
        # Import here to avoid circular imports
        from app.models.train import Train as TrainModel
        from app.models.schedule import Schedule as ScheduleModel, ScheduleStatus
        from app.core.dependencies import get_db
        from sqlalchemy.orm import Session
        
        # Get database session (simplified for integration)
        # In production, you'd inject this properly
        from app.db.session import SessionLocal
        db = SessionLocal()
        
        try:
            # Get current time in IST
            current_time = datetime.now(IST)
            
            # Get active schedules from the next 8 hours
            eight_hours_later = current_time + timedelta(hours=8)
            
            # Query active trains with their schedules
            active_schedules = (
                db.query(ScheduleModel, TrainModel)
                .join(TrainModel, ScheduleModel.train_id == TrainModel.train_id)
                .filter(
                    ScheduleModel.status.in_([ScheduleStatus.WAITING, ScheduleStatus.MOVING]),
                    ScheduleModel.planned_time >= current_time,
                    ScheduleModel.planned_time <= eight_hours_later
                )
                .all()
            )
            
            # Get track data for position calculations
            track_data = await get_railway_track()
            track_points = [{"lat": point.lat, "lon": point.lon} for point in track_data.track_polyline]
            
            # Calculate positions for all active trains
            train_positions = []
            for schedule, train in active_schedules:
                # Calculate current position based on schedule and time
                position = calculate_train_position_from_schedule(
                    schedule, train, track_points, current_time
                )
                if position:
                    train_positions.append(position)
            
            logger.info(f"Calculated positions for {len(train_positions)} trains")
            
            return TrainPositionsResponse(
                trains=train_positions,
                total_trains=len(train_positions),
                simulation_time=current_time
            )
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error calculating train positions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating train positions: {str(e)}"
        )

def calculate_train_position_from_schedule(schedule, train, track_points: List[dict], current_time: datetime) -> Optional[TrainPosition]:
    """Calculate current position of a train based on its schedule and current time"""
    try:
        # Determine train priority based on type
        priority_map = {
            "Express": TrainPriority.EXPRESS,
            "Passenger": TrainPriority.PASSENGER,
            "Freight": TrainPriority.FREIGHT
        }
        
        train_type_str = train.type.value if hasattr(train.type, 'value') else str(train.type)
        priority = priority_map.get(train_type_str, TrainPriority.PASSENGER)
        
        # Calculate progress based on time
        planned_departure = schedule.planned_time
        optimized_departure = schedule.optimized_time
        
        # Use optimized time as reference
        departure_time = optimized_departure if optimized_departure else planned_departure
        
        # Calculate elapsed time since departure (or time until departure)
        time_diff = (current_time - departure_time).total_seconds() / 60  # minutes
        
        # Determine status and position
        if time_diff < -30:  # More than 30 minutes before departure
            status = "stopped"
            progress = 0.0
            speed = 0.0
        elif time_diff < 0:  # Before departure but close
            status = "stopped"
            progress = 0.0
            speed = 0.0
        else:  # After departure time
            # Calculate progress based on elapsed time and estimated journey time
            # Assume average speed of 60 km/h for journey time estimation
            estimated_journey_time = 90  # minutes for Jalandhar-Ludhiana
            progress = min(time_diff / estimated_journey_time, 1.0)
            
            # Determine status
            delay_minutes = schedule.delay_minutes or 0
            if delay_minutes > 0:
                status = "delayed"
                speed = 45.0  # Reduced speed for delayed trains
            else:
                status = "moving"
                speed = 60.0  # Normal speed
        
        # Get current position along track
        if progress >= 1.0:
            # Train has reached destination
            track_index = len(track_points) - 1
            status = "stopped"
            speed = 0.0
        else:
            track_index = int(progress * (len(track_points) - 1))
            track_index = min(track_index, len(track_points) - 1)
        
        current_point = track_points[track_index]
        
        # Check if train is conflicted (has delays)
        is_conflicted = (schedule.delay_minutes or 0) > 0
        
        return TrainPosition(
            train_id=train.train_id,
            train_name=f"{train.train_id} ({train.origin} - {train.destination})",
            priority=priority,
            current_lat=current_point["lat"],
            current_lon=current_point["lon"],
            progress=progress,
            speed=speed,
            status=status,
            is_conflicted=is_conflicted,
            next_junction=train.destination,
            eta_next_junction=departure_time + timedelta(minutes=90)  # Estimated arrival
        )
        
    except Exception as e:
        logger.error(f"Error calculating position for train {train.train_id}: {e}")
        return None

@router.post("/simulation/speed/{multiplier}")
async def set_simulation_speed(multiplier: float):
    """Set simulation speed multiplier"""
    global SIMULATION_SPEED_MULTIPLIER
    
    if multiplier < 0.1 or multiplier > 10.0:
        raise HTTPException(status_code=400, detail="Speed multiplier must be between 0.1 and 10.0")
    
    SIMULATION_SPEED_MULTIPLIER = multiplier
    logger.info(f"Simulation speed set to {multiplier}x")
    
    return {"message": f"Simulation speed set to {multiplier}x", "multiplier": multiplier}

@router.post("/simulation/reset")
async def reset_simulation():
    """Reset simulation to start time"""
    global SIMULATION_START_TIME
    
    SIMULATION_START_TIME = datetime.now(IST)
    logger.info("Simulation reset")
    
    return {"message": "Simulation reset", "start_time": SIMULATION_START_TIME}

@router.get("/route-info")
async def get_route_info():
    """Get combined route information with tracks and stations"""
    logger.info("Fetching complete route information")
    
    try:
        # Get both track and stations data
        track_data = await get_railway_track()
        stations_data = await get_railway_stations()
        
        return {
            "route": "Jalandhar to Ludhiana",
            "track_polyline": track_data.track_polyline,
            "stations": stations_data.stations,
            "total_track_points": track_data.total_points,
            "total_stations": stations_data.total_stations,
            "bounding_box": BBOX
        }
        
    except Exception as e:
        logger.error(f"Error getting route info: {e}")
        raise HTTPException(status_code=500, detail="Error getting route information")
