"""
Updated FastAPI Backend Integration with Accurate OSM Routes

Replace your existing map_simulation.py with this version to use accurate train positioning.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Optional, Dict
import requests
import logging
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.core.dependencies import get_db

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
    progress: float # 0.0 to 1.0 along the route
    speed: float # km/h
    status: str # "moving", "delayed", "stopped"
    is_conflicted: bool
    next_junction: Optional[str] = None
    eta_next_junction: Optional[datetime] = None

class TrainPositionsResponse(BaseModel):
    trains: List[TrainPosition]
    total_trains: int
    simulation_time: datetime

# Configuration
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
ACCURATE_PYTHON_API_URL = "http://localhost:5000"

# Bounding box for Jalandhar to Ludhiana region
BBOX = "30.8,75.4,31.4,75.9"

# Mock data for fallback when APIs are unavailable
MOCK_TRACK_DATA = [
    {"lat": 31.3260, "lon": 75.5762}, # Jalandhar Cantt
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
    {"lat": 31.1814, "lon": 75.8458} # Ludhiana
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
        "name": "Ludhiana Junction",
        "lat": 31.1814,
        "lon": 75.8458,
        "city": "Ludhiana", 
        "state": "Punjab",
        "type": "junction"
    }
]

def make_overpass_request(query: str, retries: int = 3) -> dict:
    """Make request to Overpass API with retry logic"""
    for attempt in range(retries):
        try:
            logger.info(f"Overpass API request attempt {attempt + 1}/{retries}")
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=30,
                headers={"User-Agent": "Train AI Backend/1.0"}
            )
            response.raise_for_status()
            data = response.json()
            logger.info("Successfully fetched OSM data")
            return data
        except requests.exceptions.Timeout:
            logger.warning(f"Overpass API timeout on attempt {attempt + 1}")
            if attempt < retries - 1:
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Overpass API error on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    
    logger.error("All Overpass API attempts failed")
    return None

@router.get("/track", response_model=TrackResponse)
async def get_railway_track():
    """Get railway track polyline with enhanced accuracy"""
    logger.info("Fetching railway track data")
    
    try:
        # Try to get accurate track data from the Python API first
        try:
            response = requests.get(f"{ACCURATE_PYTHON_API_URL}/api/routes", timeout=10)
            if response.status_code == 200:
                routes_data = response.json()
                
                # Combine all route waypoints to create comprehensive track
                all_waypoints = []
                for route_key, waypoints in routes_data.get("route_details", {}).items():
                    all_waypoints.extend(waypoints)
                
                if all_waypoints:
                    track_points = [TrackPoint(lat=wp["lat"], lon=wp["lon"]) for wp in all_waypoints]
                    logger.info(f"Using accurate track data with {len(track_points)} points from Python API")
                    return TrackResponse(
                        track_polyline=track_points,
                        total_points=len(track_points),
                        section="Accurate OSM Railway Tracks"
                    )
        except Exception as e:
            logger.warning(f"Could not fetch accurate tracks from Python API: {e}")
        
        # Fallback to OSM Overpass API
        query = f"""
        [out:json][timeout:60];
        (
          way["railway"="rail"]({BBOX});
          way["railway"="light_rail"]({BBOX});
        );
        out geom;
        """
        
        data = make_overpass_request(query)
        track_points = []
        
        if data and data.get("elements"):
            for element in data.get("elements", []):
                if element.get("type") == "way" and "geometry" in element:
                    for point in element["geometry"]:
                        track_points.append(TrackPoint(
                            lat=point["lat"],
                            lon=point["lon"]
                        ))
        
        if not track_points:
            logger.warning("Using mock track data as fallback")
            track_points = [TrackPoint(lat=point["lat"], lon=point["lon"]) for point in MOCK_TRACK_DATA]
        
        logger.info(f"Returning {len(track_points)} track points")
        return TrackResponse(
            track_polyline=track_points,
            total_points=len(track_points),
            section="Jalandhar to Ludhiana Railway Section"
        )
        
    except Exception as e:
        logger.error(f"Error processing track data: {e}")
        track_points = [TrackPoint(lat=point["lat"], lon=point["lon"]) for point in MOCK_TRACK_DATA]
        return TrackResponse(
            track_polyline=track_points,
            total_points=len(track_points),
            section="Fallback Railway Track"
        )

@router.get("/stations", response_model=StationsResponse)
async def get_railway_stations():
    """Get railway stations with enhanced data"""
    logger.info("Fetching railway stations data")
    
    try:
        # Try to get station data from accurate Python API
        try:
            response = requests.get(f"{ACCURATE_PYTHON_API_URL}/api/routes", timeout=10)
            if response.status_code == 200:
                routes_data = response.json()
                stations_data = routes_data.get("stations", {})
                
                if stations_data:
                    stations = []
                    for code, station_info in stations_data.items():
                        stations.append(Station(
                            name=station_info["name"],
                            lat=station_info["lat"],
                            lon=station_info["lon"],
                            city=station_info["name"].split()[0],  # Extract city from name
                            state="Punjab",
                            type="junction" if code == "JUC" else "station"
                        ))
                    
                    logger.info(f"Using accurate station data with {len(stations)} stations")
                    return StationsResponse(
                        stations=stations,
                        total_stations=len(stations)
                    )
        except Exception as e:
            logger.warning(f"Could not fetch accurate stations: {e}")
        
        # Fallback to OSM data
        query = f"""
        [out:json][timeout:60];
        (
          node["railway"="station"]({BBOX});
          node["railway"="junction"]({BBOX});
          node["railway"="halt"]({BBOX});
        );
        out meta;
        """
        
        data = make_overpass_request(query)
        stations = []
        
        if data and data.get("elements"):
            for element in data.get("elements", []):
                if element.get("type") == "node":
                    tags = element.get("tags", {})
                    name = (tags.get("name") or 
                           f"Station at {element['lat']:.4f}, {element['lon']:.4f}")
                    
                    stations.append(Station(
                        name=name,
                        lat=element["lat"],
                        lon=element["lon"],
                        city=tags.get("addr:city", "Unknown"),
                        state=tags.get("addr:state", "Punjab"),
                        type=tags.get("railway", "station")
                    ))
        
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
        stations = [Station(**station_data) for station_data in MOCK_STATIONS_DATA]
        return StationsResponse(
            stations=stations,
            total_stations=len(stations)
        )

@router.get("/train-positions-accurate", response_model=TrainPositionsResponse)
async def get_accurate_train_positions_from_python(db: Session = Depends(get_db)):
    """
    Get accurate train positions from Python API with OSM railway routes
    This ensures trains stick perfectly to railway tracks
    """
    logger.info("Fetching accurate train positions from Python API")
    
    try:
        # First, try to update Python API with latest database data
        try:
            await update_python_api_with_db_data(db)
            logger.info("✅ Updated Python API with latest database data")
        except Exception as update_error:
            logger.warning(f"Could not update Python API: {update_error}")
        
        # Fetch accurate positions from Python API
        try:
            response = requests.get(f"{ACCURATE_PYTHON_API_URL}/api/train-positions", timeout=10)
            response.raise_for_status()
            python_data = response.json()
            
            logger.info(f"Received {len(python_data.get('trains', []))} accurate train positions")
            
            # Convert Python API response to FastAPI format
            train_positions = []
            for train_data in python_data.get('trains', []):
                # Map priority strings
                priority_map = {
                    'EXPRESS': TrainPriority.EXPRESS,
                    'PASSENGER': TrainPriority.PASSENGER,
                    'FREIGHT': TrainPriority.FREIGHT
                }
                
                priority = priority_map.get(train_data.get('priority', 'PASSENGER'), TrainPriority.PASSENGER)
                
                # Parse ETA
                eta_next_junction = None
                if train_data.get('eta_next_junction'):
                    try:
                        eta_next_junction = datetime.fromisoformat(train_data['eta_next_junction'].replace('Z', '+00:00'))
                    except ValueError:
                        eta_next_junction = None
                
                train_position = TrainPosition(
                    train_id=train_data.get('train_id', ''),
                    train_name=train_data.get('train_name', ''),
                    priority=priority,
                    current_lat=train_data.get('current_lat', 31.3260),
                    current_lon=train_data.get('current_lon', 75.5762),
                    progress=train_data.get('progress', 0.0),
                    speed=train_data.get('speed', 0.0),
                    status=train_data.get('status', 'stopped'),
                    is_conflicted=train_data.get('is_conflicted', False),
                    next_junction=train_data.get('next_junction'),
                    eta_next_junction=eta_next_junction
                )
                train_positions.append(train_position)
                
                logger.debug(f"🚂 Train {train_position.train_id}: {train_position.status} at "
                           f"{train_position.current_lat:.4f}, {train_position.current_lon:.4f}")
            
            return TrainPositionsResponse(
                trains=train_positions,
                total_trains=len(train_positions),
                simulation_time=datetime.fromisoformat(python_data.get('simulation_time', datetime.now(IST).isoformat()).replace('Z', '+00:00'))
            )
            
        except requests.RequestException as e:
            logger.warning(f"Accurate Python API not available: {e}. Falling back to database integration.")
            # Fallback to existing database integration
            return await get_train_positions(db)
    
    except Exception as e:
        logger.error(f"Error fetching accurate positions: {e}")
        # Fallback to existing method
        return await get_train_positions(db)

@router.get("/train-positions", response_model=TrainPositionsResponse)
async def get_train_positions(db: Session = Depends(get_db)):
    """
    Original train positions endpoint (fallback for compatibility)
    """
    logger.info("Fetching train positions from database integration")
    
    try:
        # Import here to avoid circular imports
        from app.models.train import Train as TrainModel
        from app.models.schedule import Schedule as ScheduleModel, ScheduleStatus
        
        # Get current time in IST
        current_time = datetime.now(IST)
        eight_hours_later = current_time + timedelta(hours=8)
        
        # Query active trains with schedules
        active_schedules = (
            db.query(ScheduleModel, TrainModel)
            .join(TrainModel, ScheduleModel.train_id == TrainModel.train_id)
            .filter(
                TrainModel.active == True,
                ScheduleModel.status.in_([ScheduleStatus.WAITING, ScheduleStatus.MOVING, ScheduleStatus.DEPARTED]),
                ScheduleModel.planned_time >= current_time,
                ScheduleModel.planned_time <= eight_hours_later
            )
            .all()
        )
        
        # Get track data
        track_data = await get_railway_track()
        track_points = [{"lat": point.lat, "lon": point.lon} for point in track_data.track_polyline]
        
        # Calculate positions
        train_positions = []
        for schedule, train in active_schedules:
            position = calculate_train_position_from_schedule(
                schedule, train, track_points, current_time, 90.0
            )
            if position:
                train_positions.append(position)
        
        logger.info(f"Calculated positions for {len(train_positions)} trains from database")
        return TrainPositionsResponse(
            trains=train_positions,
            total_trains=len(train_positions),
            simulation_time=current_time
        )
        
    except Exception as e:
        logger.error(f"Database integration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating train positions: {str(e)}"
        )

@router.post("/update-python-data")
async def update_python_train_data(db: Session = Depends(get_db)):
    """Update Python API with latest database train data"""
    logger.info("Updating Python API with database train data")
    
    try:
        result = await update_python_api_with_db_data(db)
        return result
    except Exception as e:
        logger.error(f"Error updating Python API: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update Python API: {str(e)}"
        )

async def update_python_api_with_db_data(db: Session):
    """Helper function to update Python API with database data"""
    from app.models.train import Train as TrainModel
    
    current_time = datetime.now(IST)
    eight_hours_later = current_time + timedelta(hours=8)
    
    # Fetch active trains
    active_trains = (
        db.query(TrainModel)
        .filter(
            TrainModel.active == True,
            TrainModel.departure_time >= current_time,
            TrainModel.departure_time <= eight_hours_later
        )
        .all()
    )
    
    # Convert to Python API format
    trains_data = {"trains": []}
    
    for train in active_trains:
        train_dict = {
            "train_id": train.train_id,
            "type": train.type.value if hasattr(train.type, 'value') else str(train.type),
            "arrival_time": train.arrival_time.isoformat() if train.arrival_time else None,
            "departure_time": train.departure_time.isoformat() if train.departure_time else None,
            "section_id": train.section_id,
            "platform_need": train.platform_need,
            "priority": train.priority,
            "origin": train.origin,
            "destination": train.destination
        }
        trains_data["trains"].append(train_dict)
    
    # Send to Python API
    response = requests.post(f"{ACCURATE_PYTHON_API_URL}/api/train-data", json=trains_data, timeout=10)
    response.raise_for_status()
    result = response.json()
    
    logger.info(f"Updated Python API with {len(trains_data['trains'])} trains")
    
    return {
        "message": "Python API updated with accurate positioning",
        "trains_sent": len(trains_data['trains']),
        "python_response": result,
        "timestamp": current_time.isoformat(),
        "positioning": "accurate_osm_routes"
    }

@router.get("/accurate-status")
async def get_accurate_positioning_status():
    """Get status of accurate positioning system"""
    try:
        response = requests.get(f"{ACCURATE_PYTHON_API_URL}/api/status", timeout=5)
        if response.status_code == 200:
            status_data = response.json()
            return {
                "accurate_python_api": "running",
                "positioning": status_data.get("positioning", "unknown"),
                "routes_available": status_data.get("routes_available", 0),
                "routes_source": status_data.get("routes_source", "unknown"),
                "websocket_enabled": status_data.get("websocket_enabled", False),
                "websocket_clients": status_data.get("websocket_clients", 0),
                "current_trains": status_data.get("current_trains", 0),
                "last_check": datetime.now(IST).isoformat()
            }
        else:
            return {
                "accurate_python_api": "error",
                "message": f"HTTP {response.status_code}",
                "positioning": "fallback"
            }
    except requests.RequestException as e:
        return {
            "accurate_python_api": "offline", 
            "message": str(e),
            "positioning": "fallback"
        }

def calculate_train_position_from_schedule(schedule, train, track_points: List[dict], current_time: datetime, estimated_journey_time: Optional[float] = None) -> Optional[TrainPosition]:
    """Calculate train position from schedule (fallback method)"""
    try:
        # Priority mapping
        priority_map = {
            "Express": TrainPriority.EXPRESS,
            "Passenger": TrainPriority.PASSENGER,
            "Freight": TrainPriority.FREIGHT
        }
        
        train_type_str = train.type.value if hasattr(train.type, 'value') else str(train.type)
        priority = priority_map.get(train_type_str, TrainPriority.PASSENGER)
        
        # Calculate progress
        planned_departure = schedule.planned_time
        optimized_departure = schedule.optimized_time
        departure_time = optimized_departure if optimized_departure else planned_departure
        
        time_diff = (current_time - departure_time).total_seconds() / 60
        
        if time_diff < -30:
            status = "stopped"
            progress = 0.0
            speed = 0.0
        elif time_diff < 0:
            status = "stopped"
            progress = 0.0 
            speed = 0.0
        else:
            journey_minutes = estimated_journey_time if estimated_journey_time else 90.0
            progress = min(time_diff / journey_minutes, 1.0)
            
            delay_minutes = schedule.delay_minutes or 0
            if delay_minutes > 0:
                status = "delayed"
                speed = max(30.0, 45.0)
            else:
                status = "moving"
                speed = max(30.0, 60.0)
        
        # Get position
        if progress >= 1.0:
            track_index = len(track_points) - 1
            status = "stopped"
            speed = 0.0
        else:
            track_index = int(progress * (len(track_points) - 1))
            track_index = min(track_index, len(track_points) - 1)
        
        current_point = track_points[track_index]
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
            eta_next_junction=departure_time + timedelta(minutes=90)
        )
        
    except Exception as e:
        logger.error(f"Error calculating position for train {train.train_id}: {e}")
        return None

@router.get("/route-info")
async def get_route_info():
    """Get enhanced route information with accurate positioning"""
    logger.info("Fetching enhanced route information")
    
    try:
        # Try to get data from accurate Python API
        try:
            response = requests.get(f"{ACCURATE_PYTHON_API_URL}/api/routes", timeout=10)
            if response.status_code == 200:
                routes_data = response.json()
                return {
                    "route": "Jalandhar Railway Network",
                    "positioning": "accurate_osm_routes",
                    "stations": routes_data.get("stations", {}),
                    "available_routes": routes_data.get("routes", []),
                    "total_routes": routes_data.get("total_routes", 0),
                    "source": routes_data.get("source", "OSM"),
                    "generated_at": routes_data.get("generated_at"),
                    "bounding_box": BBOX
                }
        except Exception as e:
            logger.warning(f"Could not get accurate route info: {e}")
        
        # Fallback to basic route info
        track_data = await get_railway_track()
        stations_data = await get_railway_stations()
        
        return {
            "route": "Jalandhar to Ludhiana",
            "positioning": "fallback",
            "track_polyline": track_data.track_polyline,
            "stations": stations_data.stations,
            "total_track_points": track_data.total_points,
            "total_stations": stations_data.total_stations,
            "bounding_box": BBOX
        }
        
    except Exception as e:
        logger.error(f"Error getting route info: {e}")
        raise HTTPException(status_code=500, detail="Error getting route information")