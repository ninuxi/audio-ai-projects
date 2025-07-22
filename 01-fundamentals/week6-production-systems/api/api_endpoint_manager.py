"""
API Endpoints Manager
====================

RESTful API endpoints for cultural heritage digitization system.
Provides secure access for institutional integration.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import uuid

try:
    from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from .auth_manager import AuthenticationManager
from .batch_processor import BatchProcessor
from .monitoring import APIMonitor

# Pydantic models for request/response validation
class ItemMetadata(BaseModel):
    title: str = Field(..., description="Title of the heritage item")
    institution: str = Field(..., description="Institution name")
    collection: str = Field(..., description="Collection name")
    description: Optional[str] = Field(None, description="Item description")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    cultural_context: Dict[str, Any] = Field(default_factory=dict, description="Cultural context")

class ProcessingRequest(BaseModel):
    item_id: str = Field(..., description="Item identifier")
    processing_type: str = Field(..., description="Type of processing requested")
    priority: int = Field(default=5, ge=1, le=10, description="Processing priority (1-10)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Processing parameters")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Search filters")
    limit: int = Field(default=50, ge=1, le=1000, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Results offset")

class BatchJobRequest(BaseModel):
    job_name: str = Field(..., description="Batch job name")
    processing_type: str = Field(..., description="Processing type")
    item_ids: List[str] = Field(..., description="List of item IDs to process")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Job parameters")

class APIResponse(BaseModel):
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class APIManager:
    """
    Manages RESTful API endpoints for cultural heritage system.
    
    Features:
    - Secure authentication and authorization
    - Item management and metadata operations
    - Batch processing capabilities
    - Real-time monitoring and status
    - Institutional integration support
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize API manager"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for API functionality. Install with: pip install fastapi uvicorn")
        
        # Initialize components
        self.auth_manager = AuthenticationManager(self.config.get('auth', {}))
        self.batch_processor = BatchProcessor(self.config.get('batch', {}))
        self.monitor = APIMonitor(self.config.get('monitoring', {}))
        
        # API configuration
        self.host = self.config.get('host', '0.0.0.0')
        self.port = self.config.get('port', 8000)
        self.enable_docs = self.config.get('enable_docs', True)
        
        # Create FastAPI app
        self.app = self._create_app()
        
        # Database manager (injected)
        self.db_manager = None
        
        self.logger.info("API Manager initialized")
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application with all endpoints"""
        
        app = FastAPI(
            title="Cultural Heritage Digitization API",
            description="RESTful API for cultural heritage audio digitization and management",
            version="1.0.0",
            docs_url="/docs" if self.enable_docs else None,
            redoc_url="/redoc" if self.enable_docs else None
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get('allowed_origins', ["*"]),
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
        
        # Add custom middleware
        app.middleware("http")(self._logging_middleware)
        app.middleware("http")(self._monitoring_middleware)
        
        # Security
        security = HTTPBearer()
        
        # Dependency for authentication
        async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
            user = await self.auth_manager.validate_token(credentials.credentials)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return user
        
        # Dependency for database
        async def get_db():
            if not self.db_manager:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Database not available"
                )
            return self.db_manager
        
        # Health check endpoint
        @app.get("/health", response_model=Dict[str, Any])
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now(),
                "version": "1.0.0",
                "services": {
                    "api": "running",
                    "database": "connected" if self.db_manager else "disconnected",
                    "auth": "active",
                    "batch_processor": "ready"
                }
            }
        
        # Authentication endpoints
        @app.post("/auth/login", response_model=Dict[str, Any])
        async def login(credentials: Dict[str, str]):
            """Authenticate user and return access token"""
            username = credentials.get("username")
            password = credentials.get("password")
            
            if not username or not password:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username and password required"
                )
            
            token = await self.auth_manager.authenticate(username, password)
            if not token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
            
            return {
                "access_token": token,
                "token_type": "bearer",
                "expires_in": self.auth_manager.token_expiry_hours * 3600
            }
        
        @app.post("/auth/refresh", response_model=Dict[str, Any])
        async def refresh_token(current_user: dict = Depends(get_current_user)):
            """Refresh access token"""
            new_token = await self.auth_manager.refresh_token(current_user["username"])
            return {
                "access_token": new_token,
                "token_type": "bearer",
                "expires_in": self.auth_manager.token_expiry_hours * 3600
            }
        
        # Item management endpoints
        @app.post("/items", response_model=APIResponse)
        async def create_item(
            metadata: ItemMetadata,
            current_user: dict = Depends(get_current_user),
            db = Depends(get_db)
        ):
            """Create new cultural heritage item"""
            try:
                item_id = str(uuid.uuid4())
                
                # Prepare metadata dict
                metadata_dict = {
                    "title": metadata.title,
                    "institution": metadata.institution,
                    "collection": metadata.collection,
                    "description": metadata.description,
                    "keywords": metadata.keywords,
                    "cultural_context": metadata.cultural_context,
                    "created_by": current_user["username"],
                    "access_level": current_user.get("default_access_level", "restricted")
                }
                
                success = await db.store_item(item_id, metadata_dict)
                
                if success:
                    return APIResponse(
                        success=True,
                        message="Item created successfully",
                        data={"item_id": item_id}
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to create item"
                    )
                    
            except Exception as e:
                self.logger.error(f"Failed to create item: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
        
        @app.get("/items/{item_id}", response_model=APIResponse)
        async def get_item(
            item_id: str,
            current_user: dict = Depends(get_current_user),
            db = Depends(get_db)
        ):
            """Get cultural heritage item by ID"""
            try:
                item = await db.get_item(item_id)
                
                if not item:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Item not found"
                    )
                
                # Check access permissions
                if not await self._check_item_access(item, current_user):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied"
                    )
                
                return APIResponse(
                    success=True,
                    message="Item retrieved successfully",
                    data=item
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get item {item_id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
        
        @app.put("/items/{item_id}", response_model=APIResponse)
        async def update_item(
            item_id: str,
            metadata: ItemMetadata,
            current_user: dict = Depends(get_current_user),
            db = Depends(get_db)
        ):
            """Update cultural heritage item metadata"""
            try:
                # Check if item exists and user has permission
                existing_item = await db.get_item(item_id)
                if not existing_item:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Item not found"
                    )
                
                if not await self._check_item_write_access(existing_item, current_user):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Write access denied"
                    )
                
                # Update metadata
                updated_metadata = existing_item.copy()
                updated_metadata.update({
                    "title": metadata.title,
                    "institution": metadata.institution,
                    "collection": metadata.collection,
                    "description": metadata.description,
                    "keywords": metadata.keywords,
                    "cultural_context": metadata.cultural_context,
                    "modified_by": current_user["username"],
                    "modified_date": datetime.now()
                })
                
                success = await db.store_item(item_id, updated_metadata)
                
                if success:
                    return APIResponse(
                        success=True,
                        message="Item updated successfully",
                        data={"item_id": item_id}
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to update item"
                    )
                    
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to update item {item_id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
        
        @app.delete("/items/{item_id}", response_model=APIResponse)
        async def delete_item(
            item_id: str,
            current_user: dict = Depends(get_current_user),
            db = Depends(get_db)
        ):
            """Delete cultural heritage item"""
            try:
                # Check permissions (usually admin only)
                if not current_user.get("is_admin", False):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Admin access required for deletion"
                    )
                
                # Implementation would depend on database manager
                # For now, return not implemented
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="Delete functionality not implemented"
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to delete item {item_id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
        
        # Search endpoints
        @app.post("/search", response_model=APIResponse)
        async def search_items(
            search_request: SearchRequest,
            current_user: dict = Depends(get_current_user),
            db = Depends(get_db)
        ):
            """Search cultural heritage items"""
            try:
                # Build search query
                query = {
                    "text_search": search_request.query,
                    **search_request.filters
                }
                
                # Add institutional access filters if needed
                if not current_user.get("is_admin", False):
                    allowed_institutions = current_user.get("institutions", [])
                    if allowed_institutions:
                        query["institution"] = {"$in": allowed_institutions}
                
                results = await db.search_items(query, limit=search_request.limit)
                
                return APIResponse(
                    success=True,
                    message=f"Found {len(results)} items",
                    data={
                        "items": results,
                        "count": len(results),
                        "query": search_request.query,
                        "filters": search_request.filters
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Search failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Search failed"
                )
        
        # File upload endpoints
        @app.post("/items/{item_id}/upload", response_model=APIResponse)
        async def upload_file(
            item_id: str,
            file: UploadFile = File(...),
            current_user: dict = Depends(get_current_user),
            db = Depends(get_db)
        ):
            """Upload audio file for heritage item"""
            try:
                # Check item exists and user has write access
                item = await db.get_item(item_id)
                if not item:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Item not found"
                    )
                
                if not await self._check_item_write_access(item, current_user):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Write access denied"
                    )
                
                # Validate file type
                allowed_types = ['.wav', '.mp3', '.flac', '.aiff', '.mp4']
                file_extension = Path(file.filename).suffix.lower()
                
                if file_extension not in allowed_types:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"File type {file_extension} not supported. Allowed: {allowed_types}"
                    )
                
                # Save file
                upload_dir = Path(self.config.get('upload_directory', 'uploads'))
                upload_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = upload_dir / f"{item_id}_{file.filename}"
                
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                # Update item metadata with file information
                file_metadata = {
                    "original_filename": file.filename,
                    "file_path": str(file_path),
                    "file_size": len(content),
                    "content_type": file.content_type,
                    "uploaded_by": current_user["username"],
                    "uploaded_at": datetime.now()
                }
                
                # Update item in database
                updated_item = item.copy()
                updated_item["file_metadata"] = file_metadata
                
                await db.store_item(item_id, updated_item)
                
                return APIResponse(
                    success=True,
                    message="File uploaded successfully",
                    data={
                        "file_path": str(file_path),
                        "file_size": len(content),
                        "filename": file.filename
                    }
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"File upload failed for item {item_id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="File upload failed"
                )
        
        # Processing endpoints
        @app.post("/process", response_model=APIResponse)
        async def request_processing(
            processing_request: ProcessingRequest,
            current_user: dict = Depends(get_current_user),
            db = Depends(get_db)
        ):
            """Request processing for heritage item"""
            try:
                # Check item exists and user has access
                item = await db.get_item(processing_request.item_id)
                if not item:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Item not found"
                    )
                
                if not await self._check_item_access(item, current_user):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied"
                    )
                
                # Submit processing job
                job_id = await self._submit_processing_job(
                    processing_request.item_id,
                    processing_request.processing_type,
                    processing_request.priority,
                    processing_request.parameters,
                    current_user["username"]
                )
                
                return APIResponse(
                    success=True,
                    message="Processing job submitted",
                    data={
                        "job_id": job_id,
                        "item_id": processing_request.item_id,
                        "processing_type": processing_request.processing_type,
                        "priority": processing_request.priority
                    }
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Processing request failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Processing request failed"
                )
        
        @app.get("/jobs/{job_id}", response_model=APIResponse)
        async def get_job_status(
            job_id: str,
            current_user: dict = Depends(get_current_user)
        ):
            """Get processing job status"""
            try:
                job_status = await self.batch_processor.get_job_status(job_id)
                
                if not job_status:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Job not found"
                    )
                
                # Check if user can access this job
                if (job_status.get("submitted_by") != current_user["username"] and 
                    not current_user.get("is_admin", False)):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied"
                    )
                
                return APIResponse(
                    success=True,
                    message="Job status retrieved",
                    data=job_status
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Job status retrieval failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Job status retrieval failed"
                )
        
        # Batch processing endpoints
        @app.post("/batch/jobs", response_model=APIResponse)
        async def create_batch_job(
            batch_request: BatchJobRequest,
            current_user: dict = Depends(get_current_user),
            db = Depends(get_db)
        ):
            """Create batch processing job"""
            try:
                # Check user has batch processing permissions
                if not current_user.get("can_batch_process", False):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Batch processing permission required"
                    )
                
                # Validate items exist and user has access
                accessible_items = []
                for item_id in batch_request.item_ids:
                    item = await db.get_item(item_id)
                    if item and await self._check_item_access(item, current_user):
                        accessible_items.append(item_id)
                
                if not accessible_items:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="No accessible items found"
                    )
                
                # Create batch job
                batch_job_id = await self.batch_processor.create_batch_job(
                    batch_request.job_name,
                    batch_request.processing_type,
                    accessible_items,
                    batch_request.parameters,
                    current_user["username"]
                )
                
                return APIResponse(
                    success=True,
                    message="Batch job created",
                    data={
                        "batch_job_id": batch_job_id,
                        "job_name": batch_request.job_name,
                        "item_count": len(accessible_items),
                        "processing_type": batch_request.processing_type
                    }
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Batch job creation failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Batch job creation failed"
                )
        
        @app.get("/batch/jobs", response_model=APIResponse)
        async def list_batch_jobs(
            current_user: dict = Depends(get_current_user),
            limit: int = 50,
            offset: int = 0
        ):
            """List batch processing jobs"""
            try:
                jobs = await self.batch_processor.list_jobs(
                    current_user["username"] if not current_user.get("is_admin") else None,
                    limit,
                    offset
                )
                
                return APIResponse(
                    success=True,
                    message=f"Retrieved {len(jobs)} batch jobs",
                    data={
                        "jobs": jobs,
                        "count": len(jobs),
                        "limit": limit,
                        "offset": offset
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Batch job listing failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Batch job listing failed"
                )
        
        @app.get("/batch/jobs/{batch_job_id}", response_model=APIResponse)
        async def get_batch_job_status(
            batch_job_id: str,
            current_user: dict = Depends(get_current_user)
        ):
            """Get batch job status and progress"""
            try:
                job_status = await self.batch_processor.get_batch_job_status(batch_job_id)
                
                if not job_status:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Batch job not found"
                    )
                
                # Check access permissions
                if (job_status.get("created_by") != current_user["username"] and 
                    not current_user.get("is_admin", False)):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied"
                    )
                
                return APIResponse(
                    success=True,
                    message="Batch job status retrieved",
                    data=job_status
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Batch job status retrieval failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Batch job status retrieval failed"
                )
        
        # Statistics and monitoring endpoints
        @app.get("/statistics", response_model=APIResponse)
        async def get_statistics(
            current_user: dict = Depends(get_current_user),
            db = Depends(get_db)
        ):
            """Get collection and processing statistics"""
            try:
                # Get collection statistics
                institution = None if current_user.get("is_admin") else current_user.get("institution")
                
                collection_stats = await db.get_collection_statistics(institution)
                
                # Get processing statistics
                processing_stats = await self.batch_processor.get_processing_statistics(
                    current_user["username"] if not current_user.get("is_admin") else None
                )
                
                # Get system monitoring data
                system_stats = await self.monitor.get_system_statistics()
                
                return APIResponse(
                    success=True,
                    message="Statistics retrieved",
                    data={
                        "collection": collection_stats,
                        "processing": processing_stats,
                        "system": system_stats,
                        "generated_at": datetime.now()
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Statistics retrieval failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Statistics retrieval failed"
                )
        
        @app.get("/monitoring/health", response_model=APIResponse)
        async def detailed_health_check(
            current_user: dict = Depends(get_current_user)
        ):
            """Detailed health check (admin only)"""
            try:
                if not current_user.get("is_admin", False):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Admin access required"
                    )
                
                health_data = await self.monitor.get_detailed_health()
                
                return APIResponse(
                    success=True,
                    message="Health check completed",
                    data=health_data
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Health check failed"
                )
        
        # Export endpoints
        @app.post("/export", response_model=APIResponse)
        async def export_collection(
            export_request: Dict[str, Any],
            current_user: dict = Depends(get_current_user),
            db = Depends(get_db)
        ):
            """Export collection data"""
            try:
                export_format = export_request.get("format", "json")
                filters = export_request.get("filters", {})
                
                # Add institutional filters if needed
                if not current_user.get("is_admin", False):
                    allowed_institutions = current_user.get("institutions", [])
                    if allowed_institutions:
                        filters["institution"] = {"$in": allowed_institutions}
                
                # Get items matching filters
                items = await db.search_items(filters, limit=10000)  # Large limit for export
                
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "exported_by": current_user["username"],
                    "format": export_format,
                    "filters": filters,
                    "item_count": len(items),
                    "items": items
                }
                
                # Generate export file
                export_id = str(uuid.uuid4())
                export_file = await self._generate_export_file(
                    export_data, export_format, export_id
                )
                
                return APIResponse(
                    success=True,
                    message="Export generated",
                    data={
                        "export_id": export_id,
                        "export_file": export_file,
                        "item_count": len(items),
                        "format": export_format
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Export failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Export failed"
                )
        
        return app
    
    async def _logging_middleware(self, request, call_next):
        """Logging middleware for API requests"""
        start_time = datetime.now()
        
        # Log request
        self.logger.info(f"{request.method} {request.url.path} - Started")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code}, Time: {process_time:.3f}s"
        )
        
        return response
    
    async def _monitoring_middleware(self, request, call_next):
        """Monitoring middleware for API metrics"""
        start_time = datetime.now()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            await self.monitor.record_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                response_time=(datetime.now() - start_time).total_seconds()
            )
            
            return response
            
        except Exception as e:
            # Record error metrics
            await self.monitor.record_error(
                method=request.method,
                path=request.url.path,
                error=str(e),
                response_time=(datetime.now() - start_time).total_seconds()
            )
            raise
    
    async def _check_item_access(self, item: Dict[str, Any], user: Dict[str, Any]) -> bool:
        """Check if user has read access to item"""
        
        # Admin has access to everything
        if user.get("is_admin", False):
            return True
        
        # Check institutional access
        user_institutions = user.get("institutions", [])
        item_institution = item.get("institution")
        
        if item_institution and item_institution in user_institutions:
            return True
        
        # Check access level
        item_access_level = item.get("access_level", "restricted")
        user_access_level = user.get("access_level", "basic")
        
        access_hierarchy = {"public": 1, "restricted": 2, "private": 3, "admin": 4}
        
        user_level = access_hierarchy.get(user_access_level, 1)
        item_level = access_hierarchy.get(item_access_level, 3)
        
        return user_level >= item_level
    
    async def _check_item_write_access(self, item: Dict[str, Any], user: Dict[str, Any]) -> bool:
        """Check if user has write access to item"""
        
        # Admin has write access to everything
        if user.get("is_admin", False):
            return True
        
        # Check if user is the creator
        if item.get("created_by") == user["username"]:
            return True
        
        # Check institutional write permissions
        if user.get("can_edit_institutional_items", False):
            user_institutions = user.get("institutions", [])
            item_institution = item.get("institution")
            return item_institution in user_institutions
        
        return False
    
    async def _submit_processing_job(self, item_id: str, processing_type: str, 
                                   priority: int, parameters: Dict[str, Any], 
                                   username: str) -> str:
        """Submit processing job to batch processor"""
        
        job_data = {
            "item_id": item_id,
            "processing_type": processing_type,
            "priority": priority,
            "parameters": parameters,
            "submitted_by": username,
            "submitted_at": datetime.now()
        }
        
        return await self.batch_processor.submit_job(job_data)
    
    async def _generate_export_file(self, export_data: Dict[str, Any], 
                                  format: str, export_id: str) -> str:
        """Generate export file in specified format"""
        
        export_dir = Path(self.config.get('export_directory', 'exports'))
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            export_file = export_dir / f"export_{export_id}.json"
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format.lower() == 'csv':
            export_file = export_dir / f"export_{export_id}.csv"
            # Implement CSV export
            # Would flatten the data structure and write to CSV
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="CSV export not yet implemented"
            )
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Export format '{format}' not supported"
            )
        
        return str(export_file)
    
    def set_database_manager(self, db_manager):
        """Set database manager instance"""
        self.db_manager = db_manager
        self.logger.info("Database manager attached to API")
    
    async def start(self):
        """Start the API server"""
        
        try:
            # Initialize components
            await self.auth_manager.initialize()
            await self.batch_processor.initialize()
            await self.monitor.initialize()
            
            # Start server
            config = uvicorn.Config(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=True
            )
            
            server = uvicorn.Server(config)
            
            self.logger.info(f"Starting API server on {self.host}:{self.port}")
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"Failed to start API server: {e}")
            raise
    
    async def stop(self):
        """Stop the API server and cleanup"""
        
        try:
            # Cleanup components
            await self.auth_manager.cleanup()
            await self.batch_processor.cleanup()
            await self.monitor.cleanup()
            
            self.logger.info("API server stopped and cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during API server shutdown: {e}")