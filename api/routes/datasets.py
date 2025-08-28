"""
Dataset management endpoints for the Garuda ML Pipeline

Handles dataset listing, retrieval, and management operations.
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, List
from pydantic import BaseModel

from shared.types import Dataset, DataType
from shared.supabase_utils import get_supabase_manager

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

class DatasetListRequest(BaseModel):
    user_id: str

class DatasetListResponse(BaseModel):
    success: bool
    message: str
    datasets: List[Dict[str, Any]]
    total_count: int

@router.post("/list", response_model=DatasetListResponse)
async def list_datasets(request: DatasetListRequest):
    """
    List all datasets for a specific user
    
    Args:
        request: DatasetListRequest with user_id
        
    Returns:
        DatasetListResponse with list of user's datasets
    """
    try:
        supabase_manager = get_supabase_manager()
        
        # Query datasets for the user
        result = supabase_manager.client.table("datasets").select("*").eq("user_id", request.user_id).execute()
        
        datasets = []
        for dataset_data in result.data:
            datasets.append({
                "dataset_id": dataset_data.get("id"),
                "dataset_key": dataset_data.get("bucket_key"),
                "filename": dataset_data.get("filename"),
                "data_type": dataset_data.get("data_type"),
                "file_size": dataset_data.get("file_size"),
                "status": dataset_data.get("status"),
                "description": dataset_data.get("description"),
                "created_at": dataset_data.get("created_at"),
                "updated_at": dataset_data.get("updated_at")
            })
        
        return DatasetListResponse(
            success=True,
            message=f"Found {len(datasets)} datasets for user {request.user_id}",
            datasets=datasets,
            total_count=len(datasets)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list datasets: {str(e)}"
        )

@router.get("/test-connection")
async def test_supabase_connection():
    """
    Test Supabase connection and bucket access
    """
    try:
        supabase_manager = get_supabase_manager()
        
        # Test database connection
        db_test = supabase_manager.client.table("datasets").select("count", count="exact").execute()
        
        # Test storage connection
        storage_test = supabase_manager.storage.list_buckets()
        
        return {
            "success": True,
            "message": "Supabase connection successful",
            "database_connection": True,
            "storage_connection": True,
            "total_datasets": db_test.count if hasattr(db_test, 'count') else 0,
            "available_buckets": [bucket.name for bucket in storage_test] if storage_test else []
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Supabase connection failed: {str(e)}",
            "database_connection": False,
            "storage_connection": False,
            "error": str(e)
        }
