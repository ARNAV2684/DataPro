"""
Enhanced Supabase utilities for the Garuda ML Pipeline

This module provides enhanced utilities for interacting with Supabase Storage and Database,
including file operations, artifact management, and pipeline state tracking.
"""

import os
import tempfile
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, BinaryIO, Union
from pathlib import Path
import logging

from supabase import create_client, Client
from postgrest.exceptions import APIError as PostgrestError

from .types import (
    Artifact, Dataset, PipelineStage, ProcessingStatus, DataType,
    UserProfile, PipelineArtifact, DataSample, ProcessingLog, 
    DataQualityMetrics, ModelResult
)

# Set up logging
logger = logging.getLogger(__name__)

class SupabaseManager:
    """Enhanced Supabase client with pipeline-specific utilities"""
    
    def __init__(self, url: str, service_role_key: str):
        """
        Initialize Supabase client with service role key for full access
        
        Args:
            url: Supabase project URL
            service_role_key: Service role key (not anon key) for backend operations
        """
        self.client: Client = create_client(url, service_role_key)
        self.storage = self.client.storage
        self.db = self.client.table
        
        # Define bucket names
        self.buckets = {
            "datasets": "datasets",
            "preprocessed": "preprocessed", 
            "augmented": "augmented",
            "eda": "eda",
            "models": "models"
        }
        
        logger.info("SupabaseManager initialized successfully")
    
    # ===================================
    # BUCKET MANAGEMENT
    # ===================================
    
    async def ensure_buckets_exist(self) -> Dict[str, bool]:
        """
        Ensure all required buckets exist, create them if they don't
        
        Returns:
            Dict mapping bucket names to creation status
        """
        results = {}
        
        for bucket_name in self.buckets.values():
            try:
                # Try to get bucket info
                bucket_info = self.storage.get_bucket(bucket_name)
                if bucket_info:
                    results[bucket_name] = False  # Already existed
                    logger.info(f"Bucket '{bucket_name}' already exists")
                else:
                    # Create bucket if it doesn't exist
                    self.storage.create_bucket(bucket_name, {"public": False})
                    results[bucket_name] = True  # Created new
                    logger.info(f"Created new bucket '{bucket_name}'")
                    
            except Exception as e:
                logger.error(f"Error managing bucket '{bucket_name}': {str(e)}")
                results[bucket_name] = False
                
        return results
    
    # ===================================
    # FILE OPERATIONS
    # ===================================
    
    def generate_storage_key(self, user_id: str, dataset_id: str, stage: str, 
                           filename: str, timestamp: Optional[datetime] = None) -> str:
        """
        Generate standardized storage key following the convention:
        user_{userId}/dataset_{datasetId}/stage_{stage}/run_{timestamp}.{ext}
        
        Args:
            user_id: User identifier
            dataset_id: Dataset identifier  
            stage: Pipeline stage name
            filename: Original filename (for extension)
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            Generated storage key
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        timestamp_str = str(int(timestamp.timestamp()))
        extension = Path(filename).suffix
        
        return f"user_{user_id}/dataset_{dataset_id}/stage_{stage}/run_{timestamp_str}{extension}"
    
    def upload_file(self, bucket_name: str, storage_key: str, 
                   file_data: Union[bytes, BinaryIO], content_type: str = "application/octet-stream") -> Dict[str, Any]:
        """
        Upload file to Supabase Storage
        
        Args:
            bucket_name: Target bucket name
            storage_key: Storage key/path
            file_data: File data as bytes or file-like object
            content_type: MIME content type
            
        Returns:
            Upload result metadata
        """
        try:
            result = self.storage.from_(bucket_name).upload(
                path=storage_key,
                file=file_data,
                file_options={"content-type": content_type}
            )
            
            logger.info(f"Successfully uploaded file to {bucket_name}/{storage_key}")
            return {
                "success": True,
                "bucket": bucket_name,
                "key": storage_key,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Failed to upload file to {bucket_name}/{storage_key}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "bucket": bucket_name,
                "key": storage_key
            }
    
    def download_file(self, bucket_name: str, storage_key: str, 
                     local_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Download file from Supabase Storage
        
        Args:
            bucket_name: Source bucket name
            storage_key: Storage key/path
            local_path: Local path to save file (optional, creates temp file if not provided)
            
        Returns:
            Download result with local file path
        """
        try:
            # Download file data
            file_data = self.storage.from_(bucket_name).download(storage_key)
            
            # Create local file path if not provided
            if local_path is None:
                # Create temp file with original extension
                extension = Path(storage_key).suffix
                temp_fd, local_path = tempfile.mkstemp(suffix=extension)
                os.close(temp_fd)  # Close file descriptor, we'll write with open()
            
            # Write file data to local path
            with open(local_path, 'wb') as f:
                f.write(file_data)
            
            logger.info(f"Successfully downloaded {bucket_name}/{storage_key} to {local_path}")
            return {
                "success": True,
                "local_path": local_path,
                "bucket": bucket_name,
                "key": storage_key,
                "size": len(file_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to download {bucket_name}/{storage_key}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "bucket": bucket_name,
                "key": storage_key
            }
    
    def delete_file(self, bucket_name: str, storage_key: str) -> Dict[str, Any]:
        """
        Delete file from Supabase Storage
        
        Args:
            bucket_name: Bucket name
            storage_key: Storage key/path
            
        Returns:
            Deletion result
        """
        try:
            result = self.storage.from_(bucket_name).remove([storage_key])
            logger.info(f"Successfully deleted {bucket_name}/{storage_key}")
            return {"success": True, "result": result}
            
        except Exception as e:
            logger.error(f"Failed to delete {bucket_name}/{storage_key}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    # ===================================
    # DATABASE OPERATIONS
    # ===================================
    
    def _convert_datetime_to_string(self, data: Any) -> Any:
        """
        Recursively convert datetime and UUID objects to strings for JSON serialization
        
        Args:
            data: Data that may contain datetime or UUID objects
            
        Returns:
            Data with datetime and UUID objects converted to strings
        """
        if hasattr(data, 'isoformat'):  # datetime object
            return data.isoformat()
        elif isinstance(data, uuid.UUID):  # UUID object
            return str(data)
        elif isinstance(data, dict):
            return {key: self._convert_datetime_to_string(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_datetime_to_string(item) for item in data]
        else:
            return data

    def insert_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Insert dataset metadata into database
        
        Args:
            dataset: Dataset metadata
            
        Returns:
            Insert result with dataset ID
        """
        try:
            dataset_data = dataset.dict(exclude_none=True)
            if 'id' not in dataset_data or dataset_data['id'] is None:
                dataset_data['id'] = str(uuid.uuid4())
            
            # Convert all datetime objects to ISO strings for JSON serialization
            dataset_data = self._convert_datetime_to_string(dataset_data)
            
            result = self.db("datasets").insert(dataset_data).execute()
            
            logger.info(f"Successfully inserted dataset {dataset_data['id']}")
            return {
                "success": True,
                "dataset_id": dataset_data['id'],
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Failed to insert dataset: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def insert_artifact(self, artifact: Artifact) -> Dict[str, Any]:
        """
        Insert artifact metadata into database
        
        Args:
            artifact: Artifact metadata
            
        Returns:
            Insert result with artifact ID
        """
        try:
            artifact_data = artifact.dict(exclude_none=True)
            if 'id' not in artifact_data or artifact_data['id'] is None:
                artifact_data['id'] = str(uuid.uuid4())
            
            # Convert all datetime objects to ISO strings for JSON serialization
            artifact_data = self._convert_datetime_to_string(artifact_data)
            
            result = self.db("artifacts").insert(artifact_data).execute()
            
            logger.info(f"Successfully inserted artifact {artifact_data['id']}")
            return {
                "success": True,
                "artifact_id": artifact_data['id'],
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Failed to insert artifact: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_latest_artifact(self, user_id: str, dataset_id: str, 
                           stage: PipelineStage) -> Optional[Dict[str, Any]]:
        """
        Get the latest artifact for a specific user, dataset, and stage
        
        Args:
            user_id: User ID
            dataset_id: Dataset ID  
            stage: Pipeline stage
            
        Returns:
            Latest artifact data or None
        """
        try:
            result = self.db("artifacts").select("*").eq("user_id", user_id).eq("dataset_id", dataset_id).eq("stage", stage.value).order("created_at", desc=True).limit(1).execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to get latest artifact: {str(e)}")
            return None
    
    def get_artifacts_by_dataset(self, dataset_id: str, stage: Optional[str] = None, 
                                limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get artifacts for a specific dataset, optionally filtered by stage
        
        Args:
            dataset_id: Dataset ID
            stage: Optional pipeline stage to filter by
            limit: Maximum number of artifacts to return
            
        Returns:
            List of artifact data
        """
        try:
            query = self.db("artifacts").select("*").eq("dataset_id", dataset_id)
            
            if stage:
                query = query.eq("stage", stage)
                
            result = query.order("created_at", desc=True).limit(limit).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Failed to get artifacts by dataset: {str(e)}")
            return []
    
    def update_artifact_status(self, artifact_id: str, status: ProcessingStatus, 
                             meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update artifact status and metadata
        
        Args:
            artifact_id: Artifact ID
            status: New processing status
            meta: Optional metadata to merge
            
        Returns:
            Update result
        """
        try:
            update_data = {
                "status": status.value,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if meta:
                update_data["meta"] = meta
            
            result = self.db("artifacts").update(update_data).eq("id", artifact_id).execute()
            
            logger.info(f"Successfully updated artifact {artifact_id} status to {status.value}")
            return {"success": True, "result": result}
            
        except Exception as e:
            logger.error(f"Failed to update artifact status: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_pipeline_status(self, user_id: str, dataset_id: str) -> Dict[str, Any]:
        """
        Get complete pipeline status for a dataset
        
        Args:
            user_id: User ID
            dataset_id: Dataset ID
            
        Returns:
            Pipeline status with all artifacts
        """
        try:
            # Get all artifacts for this dataset
            artifacts_result = self.db("artifacts").select("*").eq("user_id", user_id).eq("dataset_id", dataset_id).order("created_at", desc=False).execute()
            
            # Get dataset info
            dataset_result = self.db("datasets").select("*").eq("id", dataset_id).eq("user_id", user_id).execute()
            
            artifacts = artifacts_result.data if artifacts_result.data else []
            dataset = dataset_result.data[0] if dataset_result.data else None
            
            # Calculate progress
            completed_stages = [a["stage"] for a in artifacts if a["status"] == "completed"]
            total_stages = len(PipelineStage)
            progress = (len(completed_stages) / total_stages) * 100
            
            return {
                "success": True,
                "dataset": dataset,
                "artifacts": artifacts,
                "completed_stages": completed_stages,
                "progress_percentage": progress,
                "total_stages": total_stages
            }
            
        except Exception as e:
            logger.error(f"Failed to get pipeline status: {str(e)}")
            return {"success": False, "error": str(e)}

    # ===================================
    # NEW DATABASE OPERATIONS FOR COMPREHENSIVE SCHEMA
    # ===================================
    
    def insert_user_profile(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Insert user profile into database"""
        try:
            profile_data = user_profile.dict(exclude_none=True)
            if 'id' not in profile_data or profile_data['id'] is None:
                profile_data['id'] = str(uuid.uuid4())
            
            profile_data = self._convert_datetime_to_string(profile_data)
            result = self.db("user_profiles").insert(profile_data).execute()
            
            logger.info(f"Successfully inserted user profile {profile_data['id']}")
            return {"success": True, "user_profile_id": profile_data['id'], "result": result}
            
        except Exception as e:
            logger.error(f"Failed to insert user profile: {str(e)}")
            return {"success": False, "error": str(e)}

    def insert_pipeline_artifact(self, artifact: PipelineArtifact) -> Dict[str, Any]:
        """Insert pipeline artifact into database"""
        try:
            artifact_data = artifact.dict(exclude_none=True)
            if 'id' not in artifact_data or artifact_data['id'] is None:
                artifact_data['id'] = str(uuid.uuid4())
            
            artifact_data = self._convert_datetime_to_string(artifact_data)
            result = self.db("pipeline_artifacts").insert(artifact_data).execute()
            
            logger.info(f"Successfully inserted pipeline artifact {artifact_data['id']}")
            return {"success": True, "artifact_id": artifact_data['id'], "result": result}
            
        except Exception as e:
            logger.error(f"Failed to insert pipeline artifact: {str(e)}")
            return {"success": False, "error": str(e)}

    def insert_data_sample(self, data_sample: DataSample) -> Dict[str, Any]:
        """Insert data sample into database"""
        try:
            sample_data = data_sample.dict(exclude_none=True)
            if 'id' not in sample_data or sample_data['id'] is None:
                sample_data['id'] = str(uuid.uuid4())
            
            sample_data = self._convert_datetime_to_string(sample_data)
            result = self.db("data_samples").insert(sample_data).execute()
            
            logger.info(f"Successfully inserted data sample {sample_data['id']}")
            return {"success": True, "sample_id": sample_data['id'], "result": result}
            
        except Exception as e:
            logger.error(f"Failed to insert data sample: {str(e)}")
            return {"success": False, "error": str(e)}

    def insert_processing_log(self, log: ProcessingLog) -> Dict[str, Any]:
        """Insert processing log into database"""
        try:
            log_data = log.dict(exclude_none=True)
            if 'id' not in log_data or log_data['id'] is None:
                log_data['id'] = str(uuid.uuid4())
            
            log_data = self._convert_datetime_to_string(log_data)
            result = self.db("processing_logs").insert(log_data).execute()
            
            logger.info(f"Successfully inserted processing log {log_data['id']}")
            return {"success": True, "log_id": log_data['id'], "result": result}
            
        except Exception as e:
            logger.error(f"Failed to insert processing log: {str(e)}")
            return {"success": False, "error": str(e)}

    def insert_data_quality_metrics(self, metrics: DataQualityMetrics) -> Dict[str, Any]:
        """Insert data quality metrics into database"""
        try:
            metrics_data = metrics.dict(exclude_none=True)
            if 'id' not in metrics_data or metrics_data['id'] is None:
                metrics_data['id'] = str(uuid.uuid4())
            
            metrics_data = self._convert_datetime_to_string(metrics_data)
            result = self.db("data_quality_metrics").insert(metrics_data).execute()
            
            logger.info(f"Successfully inserted data quality metrics {metrics_data['id']}")
            return {"success": True, "metrics_id": metrics_data['id'], "result": result}
            
        except Exception as e:
            logger.error(f"Failed to insert data quality metrics: {str(e)}")
            return {"success": False, "error": str(e)}

    def insert_model_result(self, model_result: ModelResult) -> Dict[str, Any]:
        """Insert model result into database"""
        try:
            result_data = model_result.dict(exclude_none=True)
            if 'id' not in result_data or result_data['id'] is None:
                result_data['id'] = str(uuid.uuid4())
            
            result_data = self._convert_datetime_to_string(result_data)
            result = self.db("model_results").insert(result_data).execute()
            
            logger.info(f"Successfully inserted model result {result_data['id']}")
            return {"success": True, "result_id": result_data['id'], "result": result}
            
        except Exception as e:
            logger.error(f"Failed to insert model result: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_user_datasets(self, user_id: str, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        """Get datasets for a user with pagination"""
        try:
            result = self.db("datasets").select("*").eq("user_id", user_id).order("created_at", desc=True).range(offset, offset + limit - 1).execute()
            
            # Get total count
            count_result = self.db("datasets").select("id", count="exact").eq("user_id", user_id).execute()
            total_count = count_result.count if hasattr(count_result, 'count') else len(result.data)
            
            return {
                "success": True,
                "datasets": result.data,
                "total_count": total_count,
                "has_more": total_count > offset + limit
            }
            
        except Exception as e:
            logger.error(f"Failed to get user datasets: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_pipeline_artifacts_by_dataset(self, dataset_id: str, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        """Get pipeline artifacts for a dataset with pagination"""
        try:
            result = self.db("pipeline_artifacts").select("*").eq("dataset_id", dataset_id).order("created_at", desc=True).range(offset, offset + limit - 1).execute()
            
            # Get total count
            count_result = self.db("pipeline_artifacts").select("id", count="exact").eq("dataset_id", dataset_id).execute()
            total_count = count_result.count if hasattr(count_result, 'count') else len(result.data)
            
            return {
                "success": True,
                "artifacts": result.data,
                "total_count": total_count,
                "has_more": total_count > offset + limit
            }
            
        except Exception as e:
            logger.error(f"Failed to get pipeline artifacts: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_data_samples_by_dataset(self, dataset_id: str, limit: int = 5, offset: int = 0) -> Dict[str, Any]:
        """Get data samples for a dataset with pagination"""
        try:
            result = self.db("data_samples").select("*").eq("dataset_id", dataset_id).order("created_at", desc=True).range(offset, offset + limit - 1).execute()
            
            return {
                "success": True,
                "samples": result.data,
                "count": len(result.data)
            }
            
        except Exception as e:
            logger.error(f"Failed to get data samples: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_processing_logs_by_artifact(self, artifact_id: str, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Get processing logs for an artifact with pagination"""
        try:
            result = self.db("processing_logs").select("*").eq("artifact_id", artifact_id).order("created_at", desc=True).range(offset, offset + limit - 1).execute()
            
            return {
                "success": True,
                "logs": result.data,
                "count": len(result.data)
            }
            
        except Exception as e:
            logger.error(f"Failed to get processing logs: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_latest_data_quality_metrics(self, dataset_id: str) -> Dict[str, Any]:
        """Get the latest data quality metrics for a dataset"""
        try:
            result = self.db("data_quality_metrics").select("*").eq("dataset_id", dataset_id).order("created_at", desc=True).limit(1).execute()
            
            return {
                "success": True,
                "metrics": result.data[0] if result.data else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get data quality metrics: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_model_results_by_dataset(self, dataset_id: str, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        """Get model results for a dataset with pagination"""
        try:
            result = self.db("model_results").select("*").eq("dataset_id", dataset_id).order("created_at", desc=True).range(offset, offset + limit - 1).execute()
            
            return {
                "success": True,
                "results": result.data,
                "count": len(result.data)
            }
            
        except Exception as e:
            logger.error(f"Failed to get model results: {str(e)}")
            return {"success": False, "error": str(e)}

    def update_pipeline_artifact_status(self, artifact_id: str, status: ProcessingStatus, 
                                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update pipeline artifact status and metadata"""
        try:
            update_data = {
                "status": status.value,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if metadata:
                update_data["metadata"] = metadata
            
            result = self.db("pipeline_artifacts").update(update_data).eq("id", artifact_id).execute()
            
            logger.info(f"Successfully updated pipeline artifact {artifact_id} status to {status.value}")
            return {"success": True, "result": result}
            
        except Exception as e:
            logger.error(f"Failed to update pipeline artifact status: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_public_url(self, bucket_name: str, storage_key: str) -> str:
        """
        Generate a public URL for a file in Supabase Storage
        
        Args:
            bucket_name: Name of the storage bucket
            storage_key: Key/path of the file in storage
            
        Returns:
            Public URL string for the file
        """
        try:
            response = self.storage.from_(bucket_name).get_public_url(storage_key)
            return response
        except Exception as e:
            logger.error(f"Failed to get public URL for {bucket_name}/{storage_key}: {str(e)}")
            return ""

# ===================================
# GLOBAL INSTANCE
# ===================================

# Global SupabaseManager instance (to be initialized by main app)
supabase_manager: Optional[SupabaseManager] = None

def get_supabase_manager() -> SupabaseManager:
    """Get the global SupabaseManager instance"""
    if supabase_manager is None:
        raise RuntimeError("SupabaseManager not initialized. Call init_supabase_manager() first.")
    return supabase_manager

def init_supabase_manager(url: str, service_role_key: str) -> SupabaseManager:
    """Initialize the global SupabaseManager instance"""
    global supabase_manager
    supabase_manager = SupabaseManager(url, service_role_key)
    return supabase_manager
