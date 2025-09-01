"""
WebSocket routes for real-time schedule updates.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any
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
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_info: Dict[str, Any] = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_info[websocket] = client_info or {}
        logger.info(f"WebSocket connection established. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.connection_info.pop(websocket, None)
            logger.info(f"WebSocket connection closed. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific client."""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"Failed to send personal message: {str(e)}")
            self.disconnect(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
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

    async def send_schedule_update(self, schedule_data: Dict[str, Any]):
        """Send schedule update to all connected clients."""
        message = {
            "type": "schedule_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": schedule_data
        }
        await self.broadcast(message)

    async def send_optimization_complete(self, optimization_result: Dict[str, Any]):
        """Send optimization completion notification."""
        message = {
            "type": "optimization_complete",
            "timestamp": datetime.utcnow().isoformat(),
            "data": optimization_result
        }
        await self.broadcast(message)

    async def send_alert(self, alert_data: Dict[str, Any]):
        """Send alert notification to connected clients."""
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
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time schedule updates.

    Clients can connect to receive:
    - Schedule updates
    - Optimization completion notifications  
    - System alerts
    - Performance metrics
    """
    await manager.connect(websocket)

    try:
        # Send initial connection confirmation
        await manager.send_personal_message({
            "type": "connection_established",
            "message": "Connected to train traffic control system",
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

    Args:
        websocket: Client WebSocket connection
        message: Parsed JSON message from client
    """
    message_type = message.get("type", "unknown")
    logger.debug(f"Received WebSocket message: {message_type}")

    if message_type == "subscribe_schedules":
        # Client wants to subscribe to schedule updates
        await manager.send_personal_message({
            "type": "subscription_confirmed",
            "subscription": "schedules",
            "message": "Subscribed to schedule updates",
            "timestamp": datetime.utcnow().isoformat()
        }, websocket)

    elif message_type == "subscribe_metrics":
        # Client wants to subscribe to metrics updates
        await manager.send_personal_message({
            "type": "subscription_confirmed", 
            "subscription": "metrics",
            "message": "Subscribed to metrics updates",
            "timestamp": datetime.utcnow().isoformat()
        }, websocket)

    elif message_type == "request_current_schedule":
        # Client requesting current schedule
        # This would typically fetch from database
        await manager.send_personal_message({
            "type": "current_schedule",
            "data": {"message": "Schedule data would be here"},
            "timestamp": datetime.utcnow().isoformat()
        }, websocket)

    elif message_type == "ping":
        # Client ping for connection health check
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
    """
    Get statistics about current WebSocket connections.

    Returns:
        Connection statistics
    """
    return manager.get_connection_stats()


# Utility functions for sending updates (to be called from other parts of the application)

async def broadcast_schedule_update(schedule_data: Dict[str, Any]):
    """
    Broadcast schedule update to all connected clients.

    Args:
        schedule_data: Updated schedule information
    """
    await manager.send_schedule_update(schedule_data)


async def broadcast_optimization_complete(optimization_result: Dict[str, Any]):
    """
    Broadcast optimization completion to all connected clients.

    Args:
        optimization_result: Results of optimization run
    """
    await manager.send_optimization_complete(optimization_result)


async def broadcast_alert(alert_data: Dict[str, Any]):
    """
    Broadcast alert to all connected clients.

    Args:
        alert_data: Alert information
    """
    await manager.send_alert(alert_data)


async def broadcast_metrics_update(metrics_data: Dict[str, Any]):
    """
    Broadcast metrics update to all connected clients.

    Args:
        metrics_data: Updated metrics information
    """
    message = {
        "type": "metrics_update",
        "timestamp": datetime.utcnow().isoformat(),
        "data": metrics_data
    }
    await manager.broadcast(message)


# Background task for periodic updates (example)
async def periodic_updates():
    """
    Background task that sends periodic updates to connected clients.
    This would typically be started as a background task in the main application.
    """
    while True:
        try:
            # Send periodic heartbeat
            if manager.active_connections:
                await manager.broadcast({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                    "active_connections": len(manager.active_connections)
                })

            # Wait 30 seconds before next heartbeat
            await asyncio.sleep(30)

        except Exception as e:
            logger.error(f"Error in periodic updates: {str(e)}")
            await asyncio.sleep(30)
