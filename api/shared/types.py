"""
Shared type definitions for the Garuda ML Pipeline API

These types are used across all API endpoints to ensure consistency
in request/response formats and data validation. Updated to match
the new database schema with comprehensive data persistence.
"""

from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from uuid import UUID

# ===================================
# ENUMS
# ===================================

class PipelineStage(str, Enum):
    """Pipeline stages in order"""
    UPLOAD = "upload"
    PREPROCESS = "preprocess" 
    AUGMENT = "augment"
    EDA = "eda"
    MODEL = "model"

class DataType(str, Enum):
    """Supported data types"""
    NUMERIC = "numeric"
    TEXT = "text"
    IMAGE = "image"
    MIXED = "mixed"

class ProcessingStatus(str, Enum):
    """Processing status for pipeline operations"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# ===================================
# DATABASE MODELS
# ===================================

class UserProfile(BaseModel):
    """User profile model"""
    id: UUID
    email: Optional[str] = None
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    subscription_tier: str = "free"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class PipelineArtifact(BaseModel):
    """Pipeline artifact model for database storage"""
    id: Optional[UUID] = None
    user_id: UUID
    dataset_id: UUID
    stage: PipelineStage
    operation: str
    bucket_key: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    input_artifact_id: Optional[UUID] = None
    parameters: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    execution_time: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class DataSample(BaseModel):
    """Data sample model for quick preview"""
    id: Optional[UUID] = None
    artifact_id: UUID
    sample_data: Dict[str, Any]
    total_rows: int
    total_columns: int
    column_info: Dict[str, Any] = {}
    created_at: Optional[datetime] = None

class ProcessingLog(BaseModel):
    """Processing log model"""
    id: Optional[UUID] = None
    artifact_id: UUID
    log_level: str = "info"
    message: str
    details: Dict[str, Any] = {}
    created_at: Optional[datetime] = None

class DataQualityMetrics(BaseModel):
    """Data quality metrics model"""
    id: Optional[UUID] = None
    artifact_id: UUID
    missing_values: Dict[str, Any] = {}
    outliers: Dict[str, Any] = {}
    duplicates: int = 0
    data_types: Dict[str, Any] = {}
    statistics: Dict[str, Any] = {}
    created_at: Optional[datetime] = None

class ModelResult(BaseModel):
    """Model training results"""
    id: Optional[UUID] = None
    user_id: UUID
    dataset_id: UUID
    model_type: str
    hyperparameters: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    model_path: Optional[str] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    training_time: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

# ===================================
# BASE REQUEST/RESPONSE MODELS
# ===================================

class PipelineRequest(BaseModel):
    """Base request model for all pipeline operations"""
    dataset_key: str = Field(..., description="Supabase storage key for input dataset")
    user_id: str = Field(..., description="User ID for authentication and storage isolation")
    params: Optional[Dict[str, Any]] = Field(default={}, description="Operation-specific parameters")

class PipelineResponse(BaseModel):
    """Base response model for all pipeline operations"""
    success: bool = Field(..., description="Whether the operation completed successfully")
    message: str = Field(..., description="Human-readable status message")
    output_key: Optional[str] = Field(None, description="Supabase storage key for output data")
    artifact_id: Optional[str] = Field(None, description="Database artifact ID")
    meta: Optional[Dict[str, Any]] = Field(default={}, description="Operation metadata and results")
    logs: Optional[List[str]] = Field(default=[], description="Processing logs")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    timestamp: Optional[str] = Field(None, description="Response timestamp in ISO format")

# ===================================
# UPLOAD MODELS
# ===================================

class UploadRequest(BaseModel):
    """Request model for dataset upload"""
    user_id: str = Field(..., description="User ID")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    data_type: DataType = Field(..., description="Type of data being uploaded")
    description: Optional[str] = Field(None, description="Optional dataset description")

class UploadResponse(PipelineResponse):
    """Response model for dataset upload"""
    dataset_id: Optional[str] = Field(None, description="Generated dataset ID")
    file_info: Optional[Dict[str, Any]] = Field(default={}, description="File metadata")

# ===================================
# PREPROCESSING MODELS
# ===================================

class PreprocessRequest(PipelineRequest):
    """Request model for preprocessing operations"""
    operation: str = Field(..., description="Specific preprocessing operation")
    
class PreprocessResponse(PipelineResponse):
    """Response model for preprocessing operations"""
    changes_summary: Optional[Dict[str, Any]] = Field(default={}, description="Summary of changes made")
    data_quality: Optional[Dict[str, Any]] = Field(default={}, description="Data quality metrics")

# ===================================
# AUGMENTATION MODELS  
# ===================================

class AugmentRequest(PipelineRequest):
    """Request model for data augmentation operations"""
    technique: str = Field(..., description="Augmentation technique to apply")
    target_size: Optional[int] = Field(None, description="Target dataset size after augmentation")

class AugmentResponse(PipelineResponse):
    """Response model for data augmentation operations"""
    original_size: Optional[int] = Field(None, description="Original dataset size")
    augmented_size: Optional[int] = Field(None, description="Augmented dataset size")
    augmentation_ratio: Optional[float] = Field(None, description="Augmentation ratio achieved")

# ===================================
# EDA MODELS
# ===================================

class EDARequest(PipelineRequest):
    """Request model for exploratory data analysis operations"""
    analysis_type: str = Field(..., description="Type of EDA analysis to perform")
    output_format: Optional[str] = Field("json", description="Output format: json, html, or images")

class EDAResponse(PipelineResponse):
    """Response model for EDA operations"""
    analysis_results: Optional[Dict[str, Any]] = Field(default={}, description="Analysis results")
    visualizations: Optional[List[str]] = Field(default=[], description="Generated visualization keys")
    insights: Optional[List[str]] = Field(default=[], description="Key insights discovered")

# ===================================
# MODEL TRAINING MODELS
# ===================================

class ModelRequest(PipelineRequest):
    """Request model for model training operations"""
    model_type: str = Field(..., description="Type of model to train")
    hyperparameters: Optional[Dict[str, Any]] = Field(default={}, description="Model hyperparameters")
    validation_split: Optional[float] = Field(0.2, description="Validation data split ratio")

class ModelResponse(PipelineResponse):
    """Response model for model training operations"""
    model_id: Optional[str] = Field(None, description="Trained model identifier")
    metrics: Optional[Dict[str, float]] = Field(default={}, description="Training and validation metrics")
    model_size: Optional[int] = Field(None, description="Model size in bytes")
    training_time: Optional[float] = Field(None, description="Training time in seconds")

# ===================================
# DATA RETRIEVAL MODELS
# ===================================

class DataRetrievalRequest(BaseModel):
    """Request model for data retrieval operations"""
    user_id: str = Field(..., description="User ID")
    dataset_id: Optional[str] = Field(None, description="Specific dataset ID")
    stage: Optional[PipelineStage] = Field(None, description="Pipeline stage filter")
    limit: Optional[int] = Field(10, description="Number of records to return")
    offset: Optional[int] = Field(0, description="Offset for pagination")

class DataRetrievalResponse(BaseModel):
    """Response model for data retrieval operations"""
    success: bool = Field(..., description="Whether the operation completed successfully")
    datasets: Optional[List[Dict[str, Any]]] = Field(default=[], description="Dataset records")
    artifacts: Optional[List[Dict[str, Any]]] = Field(default=[], description="Artifact records")
    total_count: Optional[int] = Field(None, description="Total number of records")
    has_more: Optional[bool] = Field(None, description="Whether more records are available")

# ===================================
# ERROR MODELS
# ===================================

class APIError(BaseModel):
    """Error response model"""
    success: bool = Field(False, description="Always false for errors")
    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default={}, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

# ===================================
# ARTIFACT MODELS
# ===================================

class Artifact(BaseModel):
    """Model for pipeline artifacts stored in database"""
    id: Optional[str] = Field(None, description="Artifact ID")
    user_id: str = Field(..., description="User ID")
    dataset_id: str = Field(..., description="Dataset ID")
    stage: PipelineStage = Field(..., description="Pipeline stage")
    bucket_key: str = Field(..., description="Supabase storage key")
    meta: Dict[str, Any] = Field(default={}, description="Artifact metadata")
    status: ProcessingStatus = Field(ProcessingStatus.PENDING, description="Processing status")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

class Dataset(BaseModel):
    """Model for dataset metadata"""
    id: Optional[str] = Field(None, description="Dataset ID")
    user_id: str = Field(..., description="User ID")
    bucket_key: str = Field(..., description="Supabase storage key")
    filename: str = Field(..., description="Original filename")
    filetype: str = Field(..., description="File type/extension")
    data_type: DataType = Field(..., description="Data type")
    file_size: int = Field(..., description="File size in bytes")
    status: ProcessingStatus = Field(ProcessingStatus.PENDING, description="Processing status")
    description: Optional[str] = Field(None, description="Dataset description")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

# ===================================
# UTILITY TYPES
# ===================================

class HealthCheck(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="API version")
    services: Dict[str, str] = Field(default={}, description="Service status details")

class PipelineStatus(BaseModel):
    """Pipeline status model for tracking progress"""
    user_id: str = Field(..., description="User ID")
    dataset_id: str = Field(..., description="Dataset ID")
    current_stage: PipelineStage = Field(..., description="Current pipeline stage")
    completed_stages: List[PipelineStage] = Field(default=[], description="Completed stages")
    artifacts: List[Artifact] = Field(default=[], description="Generated artifacts")
    status: ProcessingStatus = Field(..., description="Overall pipeline status")
    progress_percentage: float = Field(..., description="Progress percentage (0-100)")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
