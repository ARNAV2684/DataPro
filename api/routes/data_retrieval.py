"""
Data retrieval endpoints for the Garuda ML Pipeline

Handles browsing and retrieving pipeline data from database tables,
including datasets, artifacts, processing logs, and data samples.
"""

from fastapi import APIRouter, HTTPException, status, Query
from typing import Optional, List, Dict, Any

from shared.types import (
    DataRetrievalRequest, DataRetrievalResponse,
    PipelineStage, ProcessingStatus
)
from shared.supabase_utils import get_supabase_manager

router = APIRouter(prefix="/api/data", tags=["data-retrieval"])

@router.get("/datasets", response_model=DataRetrievalResponse)
async def get_user_datasets(
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(10, description="Number of datasets to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """
    Get datasets for a user with pagination
    
    Args:
        user_id: User ID
        limit: Number of records to return (default: 10)
        offset: Offset for pagination (default: 0)
        
    Returns:
        DataRetrievalResponse with datasets
    """
    try:
        supabase_manager = get_supabase_manager()
        
        result = supabase_manager.get_user_datasets(
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve datasets: {result['error']}"
            )
        
        return DataRetrievalResponse(
            success=True,
            datasets=result["datasets"],
            total_count=result["total_count"],
            has_more=result["has_more"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error retrieving datasets: {str(e)}"
        )

@router.get("/datasets/{dataset_id}/artifacts", response_model=DataRetrievalResponse)
async def get_dataset_artifacts(
    dataset_id: str,
    limit: int = Query(10, description="Number of artifacts to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """
    Get pipeline artifacts for a specific dataset
    
    Args:
        dataset_id: Dataset ID
        limit: Number of records to return (default: 10)
        offset: Offset for pagination (default: 0)
        
    Returns:
        DataRetrievalResponse with artifacts
    """
    try:
        supabase_manager = get_supabase_manager()
        
        result = supabase_manager.get_pipeline_artifacts_by_dataset(
            dataset_id=dataset_id,
            limit=limit,
            offset=offset
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve artifacts: {result['error']}"
            )
        
        return DataRetrievalResponse(
            success=True,
            artifacts=result["artifacts"],
            total_count=result["total_count"],
            has_more=result["has_more"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error retrieving artifacts: {str(e)}"
        )

@router.get("/datasets/{dataset_id}/samples")
async def get_dataset_samples(
    dataset_id: str,
    limit: int = Query(5, description="Number of samples to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """
    Get data samples for a specific dataset for quick preview
    
    Args:
        dataset_id: Dataset ID
        limit: Number of samples to return (default: 5)
        offset: Offset for pagination (default: 0)
        
    Returns:
        JSON response with data samples
    """
    try:
        supabase_manager = get_supabase_manager()
        
        result = supabase_manager.get_data_samples_by_dataset(
            dataset_id=dataset_id,
            limit=limit,
            offset=offset
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve data samples: {result['error']}"
            )
        
        return {
            "success": True,
            "samples": result["samples"],
            "count": result["count"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error retrieving data samples: {str(e)}"
        )

@router.get("/artifacts/{artifact_id}/logs")
async def get_artifact_logs(
    artifact_id: str,
    limit: int = Query(50, description="Number of logs to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """
    Get processing logs for a specific artifact
    
    Args:
        artifact_id: Artifact ID
        limit: Number of logs to return (default: 50)
        offset: Offset for pagination (default: 0)
        
    Returns:
        JSON response with processing logs
    """
    try:
        supabase_manager = get_supabase_manager()
        
        result = supabase_manager.get_processing_logs_by_artifact(
            artifact_id=artifact_id,
            limit=limit,
            offset=offset
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve processing logs: {result['error']}"
            )
        
        return {
            "success": True,
            "logs": result["logs"],
            "count": result["count"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error retrieving processing logs: {str(e)}"
        )

@router.get("/datasets/{dataset_id}/quality-metrics")
async def get_dataset_quality_metrics(dataset_id: str):
    """
    Get the latest data quality metrics for a dataset
    
    Args:
        dataset_id: Dataset ID
        
    Returns:
        JSON response with data quality metrics
    """
    try:
        supabase_manager = get_supabase_manager()
        
        result = supabase_manager.get_latest_data_quality_metrics(dataset_id=dataset_id)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve quality metrics: {result['error']}"
            )
        
        return {
            "success": True,
            "metrics": result["metrics"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error retrieving quality metrics: {str(e)}"
        )

@router.get("/datasets/{dataset_id}/models")
async def get_dataset_model_results(
    dataset_id: str,
    limit: int = Query(10, description="Number of model results to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """
    Get model training results for a specific dataset
    
    Args:
        dataset_id: Dataset ID
        limit: Number of results to return (default: 10)
        offset: Offset for pagination (default: 0)
        
    Returns:
        JSON response with model results
    """
    try:
        supabase_manager = get_supabase_manager()
        
        result = supabase_manager.get_model_results_by_dataset(
            dataset_id=dataset_id,
            limit=limit,
            offset=offset
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve model results: {result['error']}"
            )
        
        return {
            "success": True,
            "results": result["results"],
            "count": result["count"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error retrieving model results: {str(e)}"
        )

@router.get("/pipeline-status/{dataset_id}")
async def get_pipeline_status(dataset_id: str, user_id: str = Query(...)):
    """
    Get complete pipeline status for a dataset including all stages and artifacts
    
    Args:
        dataset_id: Dataset ID
        user_id: User ID for authorization
        
    Returns:
        JSON response with complete pipeline status
    """
    try:
        supabase_manager = get_supabase_manager()
        
        result = supabase_manager.get_pipeline_status(
            user_id=user_id,
            dataset_id=dataset_id
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve pipeline status: {result['error']}"
            )
        
        return {
            "success": True,
            "dataset": result["dataset"],
            "artifacts": result["artifacts"],
            "completed_stages": result["completed_stages"],
            "progress_percentage": result["progress_percentage"],
            "total_stages": result["total_stages"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error retrieving pipeline status: {str(e)}"
        )
