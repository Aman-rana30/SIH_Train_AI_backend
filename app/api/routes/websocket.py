"""
WebSocket routes for real-time schedule updates.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import json
import asyncio
import logging
from datetime import datetime

from app.core.dependencies import get_db
from app.models.schedule import Schedule as ScheduleModel, ScheduleStatus

router = APIRouter()
logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        # Stores metadata per connection (e.g., section_id)
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_info: Dict[str, Any] = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_info[websocket] = client_info or {}
        logger.info(
            f"WebSocket connection established. "
            f"Section={client_info.get('section_id') if client_info else None}, "
            f"Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.connection_info.pop(websocket, None)
            logger.info(
                f"WebSocket connection closed. Total connections: {len(self.active_connections)}"
            )

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific client."""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"Failed to send personal message: {str(e)}")
            self.disconnect(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients (no filtering)."""
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.error(f"Failed to broadcast to connection: {str(e)}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_section_filtered(self, message: Dict[str, Any]):
        """Broadcast only relevant section-specific data to each client."""
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            section_id = self.connection_info.get(connection, {}).get("section_id")
            if section_id is None:
                # Skip clients without a section subscription for section-specific data
                continue

            # Filter the message data for this section
            filtered_data = self._filter_for_section(message.get("data", {}), section_id)
            if filtered_data is None:
                continue  # nothing relevant for this section

            personal_msg = {
                **message,
                "data": filtered_data,
            }

            try:
                await connection.send_text(json.dumps(personal_msg, default=str))
            except Exception as e:
                logger.error(f"Failed to broadcast filtered data: {str(e)}")
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)

    def _filter_for_section(self, data: Dict[str, Any], section_id: str) -> Optional[Dict[str, Any]]:
        """
        Extracts only the portion of data relevant to a given section.
        Handles optimization results of the form:
        {
            "schedules": [...],
            "metrics": {...}
        }
        """
        if not data:
            return None

        # Case 1: optimization result with schedules + metrics
        if isinstance(data, dict) and "schedules" in data:
            schedules = data.get("schedules", [])
            filtered_schedules = [s for s in schedules if s.get("section_id") == section_id]
            if not filtered_schedules:
                return None
            return {
                "schedules": filtered_schedules,
                "metrics": data.get("metrics", {})
            }

        # Case 2: single schedule dict
        if isinstance(data, dict) and "section_id" in data:
            return data if data["section_id"] == section_id else None

        # Case 3: list of schedules
        if isinstance(data, list):
            filtered = [item for item in data if item.get("section_id") == section_id]
            return filtered if filtered else None

        return None

    async def send_schedule_update(self, schedule_data: Dict[str, Any]):
        """Send section-filtered schedule update."""
        message = {
            "type": "schedule_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": schedule_data
        }
        await self.broadcast_section_filtered(message)

    async def send_optimization_complete(self, optimization_result: Dict[str, Any]):
        """Send section-filtered optimization completion notification."""
        message = {
            "type": "optimization_complete",
            "timestamp": datetime.utcnow().isoformat(),
            "data": optimization_result
        }
        await self.broadcast_section_filtered(message)

    async def send_alert(self, alert_data: Dict[str, Any]):
        """Send alert notification to ALL clients (not section-specific)."""
        message = {
            "type": "alert",
            "timestamp": datetime.utcnow().isoformat(),
            "data": alert_data
        }
        await self.broadcast(message)

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about current connections."""
        return {
            "total_connections": len(self.active_connections),
            "connection_details": [
                {
                    "id": id(conn),
                    "info": self.connection_info.get(conn, {})
                }
                for conn in self.active_connections
            ]
        }


# Global connection manager instance
manager = ConnectionManager()


@router.websocket("/updates")
async def websocket_endpoint(websocket: WebSocket, section_id: Optional[str] = Query(None)):
    """
    WebSocket endpoint for real-time schedule updates.

    Clients must connect with ?section_id=XYZ if they want section-filtered updates.

    Clients can receive:
    - Schedule updates (filtered by section)
    - Optimization completion notifications (filtered by section)
    - System alerts (to all clients)
    - Heartbeats / metrics
    """
    await manager.connect(websocket, client_info={"section_id": section_id})

    try:
        # Send initial connection confirmation
        await manager.send_personal_message({
            "type": "connection_established",
            "message": f"Connected to train traffic control system (section={section_id})",
            "timestamp": datetime.utcnow().isoformat()
        }, websocket)

        # Handle incoming messages
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                await handle_client_message(websocket, message)
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat()
                }, websocket)
            except Exception as e:
                logger.error(f"Error handling client message: {str(e)}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Error processing message",
                    "timestamp": datetime.utcnow().isoformat()
                }, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)


async def handle_client_message(websocket: WebSocket, message: Dict[str, Any]):
    """
    Handle incoming messages from WebSocket clients.
    """
    message_type = message.get("type", "unknown")
    logger.debug(f"Received WebSocket message: {message_type}")

    if message_type == "subscribe_schedules":
        await manager.send_personal_message({
            "type": "subscription_confirmed",
            "subscription": "schedules",
            "message": "Subscribed to schedule updates",
            "timestamp": datetime.utcnow().isoformat()
        }, websocket)

    elif message_type == "subscribe_metrics":
        await manager.send_personal_message({
            "type": "subscription_confirmed",
            "subscription": "metrics",
            "message": "Subscribed to metrics updates",
            "timestamp": datetime.utcnow().isoformat()
        }, websocket)

    elif message_type == "request_current_schedule":
        await manager.send_personal_message({
            "type": "current_schedule",
            "data": {"message": "Schedule data would be here"},
            "timestamp": datetime.utcnow().isoformat()
        }, websocket)

    elif message_type == "ping":
        await manager.send_personal_message({
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat()
        }, websocket)

    else:
        await manager.send_personal_message({
            "type": "error",
            "message": f"Unknown message type: {message_type}",
            "timestamp": datetime.utcnow().isoformat()
        }, websocket)


@router.get("/connections")
async def get_connection_stats():
    """Get statistics about current WebSocket connections."""
    return manager.get_connection_stats()


# Utility functions for sending updates (to be called from other parts of the application)
async def broadcast_schedule_update(schedule_data: Dict[str, Any]):
    await manager.send_schedule_update(schedule_data)


async def broadcast_optimization_complete(optimization_result: Dict[str, Any]):
    await manager.send_optimization_complete(optimization_result)


async def broadcast_alert(alert_data: Dict[str, Any]):
    await manager.send_alert(alert_data)


async def broadcast_metrics_update(metrics_data: Dict[str, Any]):
    message = {
        "type": "metrics_update",
        "timestamp": datetime.utcnow().isoformat(),
        "data": metrics_data
    }
    await manager.broadcast(message)


# Background task for periodic updates (example)
async def periodic_updates():
    while True:
        try:
            if manager.active_connections:
                await manager.broadcast({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                    "active_connections": len(manager.active_connections)
                })
            await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"Error in periodic updates: {str(e)}")
            await asyncio.sleep(30)
