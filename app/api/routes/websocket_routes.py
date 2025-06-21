"""WebSocket routes for real-time job progress tracking."""

import json
import asyncio
import logging
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from datetime import datetime

from app.models.job_models import ProgressUpdate
from app.services.database_service import get_database_service
from app.services.job_queue_service import get_job_queue_service
from app.services.util import get_env_var

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["websockets"])

WEBSOCKET_PING_INTERVAL = get_env_var("WEBSOCKET_PING_INTERVAL", 30, int)


class WebSocketManager:
    """Manages WebSocket connections for real-time job progress updates."""
    
    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {}
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket, job_id: str):
        """Accept WebSocket connection for a specific job."""
        await websocket.accept()
        
        if job_id not in self.connections:
            self.connections[job_id] = set()
        
        self.connections[job_id].add(websocket)
        self.active_connections.add(websocket)
        
        logger.info(f"WebSocket connected for job {job_id} (total connections: {len(self.active_connections)})")
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        """Remove WebSocket connection."""
        if job_id in self.connections:
            self.connections[job_id].discard(websocket)
            
            if not self.connections[job_id]:
                del self.connections[job_id]
        
        self.active_connections.discard(websocket)
        
        logger.info(f"WebSocket disconnected for job {job_id} (total connections: {len(self.active_connections)})")
    
    async def send_progress_update(self, job_id: str, progress_update: ProgressUpdate):
        """Send progress update to all connections for a specific job."""
        if job_id not in self.connections:
            return
        
        message = {
            "type": "progress_update",
            "data": {
                "job_id": progress_update.job_id,
                "status": progress_update.status,
                "stage": progress_update.stage,
                "percentage": progress_update.percentage,
                "message": progress_update.message,
                "timestamp": progress_update.timestamp.isoformat(),
                "estimated_completion": progress_update.estimated_completion.isoformat() if progress_update.estimated_completion else None,
                "error_details": progress_update.error_details
            }
        }
        
        message_text = json.dumps(message)
        
        connections_to_remove = []
        for websocket in self.connections[job_id].copy():
            try:
                await websocket.send_text(message_text)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket for job {job_id}: {e}")
                connections_to_remove.append(websocket)
        
        for websocket in connections_to_remove:
            self.disconnect(websocket, job_id)
    
    async def send_job_status_update(self, job_id: str, status: str, message: str = None):
        """Send job status update to all connections for a specific job."""
        progress_update = ProgressUpdate(
            job_id=job_id,
            status=status,
            stage=status,
            percentage=100.0 if status in ["completed", "failed", "cancelled"] else 0.0,
            message=message or f"Job status changed to {status}"
        )
        
        await self.send_progress_update(job_id, progress_update)
    
    def get_connection_count(self, job_id: str = None) -> int:
        """Get number of active connections."""
        if job_id:
            return len(self.connections.get(job_id, set()))
        return len(self.active_connections)


websocket_manager = WebSocketManager()


@router.websocket("/jobs/{job_id}/progress")
async def websocket_job_progress(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job progress updates."""
    
    try:
        db_service = await get_database_service()
        job = await db_service.get_job(job_id)
        
        if not job:
            await websocket.close(code=4004, reason=f"Job {job_id} not found")
            return
        
        await websocket_manager.connect(websocket, job_id)
        
        job_queue = get_job_queue_service()
        
        async def progress_callback(progress_update: ProgressUpdate):
            """Callback to send progress updates via WebSocket."""
            await websocket_manager.send_progress_update(job_id, progress_update)
        
        job_queue.add_progress_callback(job_id, progress_callback)
        
        await websocket_manager.send_job_status_update(
            job_id, 
            job.status, 
            f"Connected to job {job_id} progress updates"
        )
        
        try:
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=float(WEBSOCKET_PING_INTERVAL))
                    
                    try:
                        message = json.loads(data)
                        await handle_websocket_message(websocket, job_id, message)
                    except json.JSONDecodeError:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Invalid JSON format"
                        }))
                        
                except asyncio.TimeoutError:
                    await websocket.send_text(json.dumps({
                        "type": "ping",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket client disconnected from job {job_id}")
            
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        try:
            await websocket.close(code=1011, reason=f"Server error: {str(e)}")
        except:
            pass
            
    finally:
        websocket_manager.disconnect(websocket, job_id)


async def handle_websocket_message(websocket: WebSocket, job_id: str, message: dict):
    """Handle incoming WebSocket messages from clients."""
    
    message_type = message.get("type")
    
    if message_type == "ping":
        await websocket.send_text(json.dumps({
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat()
        }))
        
    elif message_type == "get_status":
        try:
            db_service = await get_database_service()
            job = await db_service.get_job(job_id)
            
            if job:
                response = {
                    "type": "status_response",
                    "data": {
                        "job_id": job.id,
                        "status": job.status,
                        "original_filename": job.original_filename,
                        "progress_stage": job.job_metadata.get("progress_stage") if job.job_metadata else None,
                        "progress_percentage": job.job_metadata.get("progress_percentage") if job.job_metadata else None,
                        "created_at": job.created_at.isoformat(),
                        "updated_at": job.updated_at.isoformat(),
                        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                        "error_message": job.error_message
                    }
                }
            else:
                response = {
                    "type": "error",
                    "message": f"Job {job_id} not found"
                }
                
            await websocket.send_text(json.dumps(response))
            
        except Exception as e:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Failed to get job status: {str(e)}"
            }))
            
    elif message_type == "cancel_job":
        try:
            job_queue = get_job_queue_service()
            cancelled = await job_queue.cancel_job(job_id)
            
            if cancelled:
                await websocket.send_text(json.dumps({
                    "type": "job_cancelled",
                    "message": f"Job {job_id} cancelled successfully"
                }))
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Failed to cancel job {job_id}"
                }))
                
        except Exception as e:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Error cancelling job: {str(e)}"
            }))
            
    else:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        }))


@router.get("/websocket/status")
async def get_websocket_status():
    """Get WebSocket connection statistics."""
    
    try:
        connection_stats = {}
        total_connections = 0
        
        for job_id, connections in websocket_manager.connections.items():
            connection_count = len(connections)
            connection_stats[job_id] = connection_count
            total_connections += connection_count
        
        return {
            "total_connections": total_connections,
            "active_job_connections": len(websocket_manager.connections),
            "connections_per_job": connection_stats,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get WebSocket status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get WebSocket status: {str(e)}"
        )


def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocket manager instance."""
    return websocket_manager 