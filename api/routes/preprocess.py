"""
Preprocessing endpoints for the Garuda ML Pipeline

Handles data preprocessing operations like missing value handling,
outlier detection, feature scaling, and tokenization.
"""

import time
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any

from shared.types import (
    PreprocessRequest, PreprocessResponse, 
    PipelineArtifact, DataSample, ProcessingLog, DataQualityMetrics,
    PipelineStage, ProcessingStatus
)
from shared.supabase_utils import get_supabase_manager
from shared.module_runner import get_module_runner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/preprocess", tags=["preprocess"])

@router.post("/handle-missing-values", response_model=PreprocessResponse)
async def handle_missing_values(request: PreprocessRequest):
    """
    Handle missing values in numeric datasets
    
    Runs: 2and3week/PPnumeric/handle_missing_values.py
    """
    return await _run_preprocessing_module(
        request=request,
        module_path="2and3week/PPnumeric/handle_missing_values.py",
        operation_name="handle_missing_values"
    )

@router.post("/handle-outliers", response_model=PreprocessResponse)
async def handle_outliers(request: PreprocessRequest):
    """
    Handle outliers in numeric datasets
    
    Runs: 2and3week/PPnumeric/handle_outliers.py
    """
    return await _run_preprocessing_module(
        request=request,
        module_path="2and3week/PPnumeric/handle_outliers.py", 
        operation_name="handle_outliers"
    )

@router.post("/scale-numeric-features", response_model=PreprocessResponse)
async def scale_numeric_features(request: PreprocessRequest):
    """
    Scale numeric features (standardization, normalization, etc.)
    
    Runs: 2and3week/PPnumeric/scale_numeric_features.py
    """
    return await _run_preprocessing_module(
        request=request,
        module_path="2and3week/PPnumeric/scale_numeric_features.py",
        operation_name="scale_numeric_features"
    )

@router.post("/transform-features", response_model=PreprocessResponse)
async def transform_features(request: PreprocessRequest):
    """
    Transform features (log, sqrt, polynomial, etc.)
    
    Runs: 2and3week/PPnumeric/transform_features.py
    """
    return await _run_preprocessing_module(
        request=request,
        module_path="2and3week/PPnumeric/transform_features.py",
        operation_name="transform_features"
    )

@router.post("/comparison", response_model=PreprocessResponse)
async def data_comparison(request: PreprocessRequest):
    """
    Compare processed datasets with original data
    
    Runs: 2and3week/PPnumeric/comparison.py
    """
    return await _run_preprocessing_module(
        request=request,
        module_path="2and3week/PPnumeric/comparison.py",
        operation_name="comparison"
    )

@router.post("/tokenization", response_model=PreprocessResponse)  
async def tokenization(request: PreprocessRequest):
    """
    Tokenize text data using various tokenization strategies
    
    Runs: 2and3week/PPtext/T1_Tokenization.py
    """
    return await _run_preprocessing_module(
        request=request,
        module_path="2and3week/PPtext/T1_Tokenization.py",
        operation_name="tokenization"
    )

# ===================================
# HELPER FUNCTION
# ===================================

async def _run_preprocessing_module(request: PreprocessRequest, module_path: str, 
                                  operation_name: str) -> PreprocessResponse:
    """
    Generic function to run preprocessing modules
    
    Args:
        request: Preprocessing request
        module_path: Path to the Python module
        operation_name: Name of the operation for logging
        
    Returns:
        PreprocessResponse with results
    """
    start_time = time.time()
    
    try:
        # Get services
        supabase_manager = get_supabase_manager()
        module_runner = get_module_runner()
        
        # Download input dataset from storage
        logger.info(f"Attempting to download dataset: {request.dataset_key}")
        download_result = supabase_manager.download_file(
            bucket_name=supabase_manager.buckets["datasets"],
            storage_key=request.dataset_key
        )
        
        logger.info(f"Download result: {download_result}")
        if not download_result["success"]:
            logger.error(f"Download failed for {request.dataset_key}: {download_result.get('error', 'Unknown error')}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Failed to download dataset: {download_result['error']}"
            )
        
        input_file = download_result["local_path"]
        logger.info(f"Downloaded dataset to: {input_file}")
        
        # Run the preprocessing module
        logger.info(f"Running preprocessing module: {module_path}")
        processing_result = module_runner.run_module_script(
            module_path=module_path,
            input_file=input_file,
            params=request.params
        )
        
        logger.info(f"Processing result: {processing_result}")
        if not processing_result["success"]:
            logger.error(f"Processing failed: {processing_result.get('error', 'Unknown error')}")
            logger.error(f"STDOUT: {processing_result.get('stdout', 'No stdout')}")
            logger.error(f"STDERR: {processing_result.get('stderr', 'No stderr')}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Processing failed: {processing_result['error']}"
            )
        
        # Generate output storage key
        # Extract dataset_id from storage key (format: user_<user_id>/dataset_<dataset_id>/stage_<stage>/<filename>)
        try:
            dataset_id = request.dataset_key.split('/')[1].replace('dataset_', '')
        except (IndexError, ValueError):
            # Fallback to a generated ID if extraction fails
            dataset_id = f"processed_{int(time.time())}"
        
        output_key = supabase_manager.generate_storage_key(
            user_id=request.user_id,
            dataset_id=dataset_id,
            stage="preprocessed",
            filename="processed_data.csv"
        )
        
        # Upload processed data to storage
        with open(processing_result["output_path"], 'rb') as f:
            upload_result = supabase_manager.upload_file(
                bucket_name=supabase_manager.buckets["preprocessed"],
                storage_key=output_key,
                file_data=f.read(),
                content_type="text/csv"
            )
        
        if not upload_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload results: {upload_result['error']}"
            )
        
        # Create pipeline artifact record
        import uuid
        artifact_id = str(uuid.uuid4())
        
        artifact = PipelineArtifact(
            id=artifact_id,
            user_id=request.user_id,
            dataset_id=dataset_id,
            stage=PipelineStage.PREPROCESS,
            operation=operation_name,
            bucket_key=output_key,
            status=ProcessingStatus.COMPLETED,
            parameters=request.params,
            metadata={
                "module_path": module_path,
                "input_key": request.dataset_key,
                "processing_result": processing_result["meta"]
            },
            execution_time=time.time() - start_time,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Insert pipeline artifact into database
        db_result = supabase_manager.insert_pipeline_artifact(artifact)
        
        if not db_result["success"]:
            # Log error but don't fail the request
            print(f"Warning: Failed to save pipeline artifact metadata: {db_result['error']}")
        
        artifact_id = db_result.get("artifact_id", artifact_id)
        
        # Create processing logs
        if processing_result.get("stdout"):
            log_entry = ProcessingLog(
                artifact_id=artifact_id,
                level="info",
                message="Processing completed successfully",
                details={
                    "stdout": processing_result["stdout"],
                    "operation": operation_name
                },
                created_at=datetime.utcnow()
            )
            supabase_manager.insert_processing_log(log_entry)
        
        if processing_result.get("stderr"):
            log_entry = ProcessingLog(
                artifact_id=artifact_id,
                level="warning",
                message="Processing stderr output",
                details={
                    "stderr": processing_result["stderr"],
                    "operation": operation_name
                },
                created_at=datetime.utcnow()
            )
            supabase_manager.insert_processing_log(log_entry)
        
        # Create data sample for quick preview (if output file is readable)
        try:
            import pandas as pd
            df = pd.read_csv(processing_result["output_path"])
            sample_size = min(5, len(df))
            sample_records = df.head(sample_size).to_dict(orient='records')
            
            # Convert list of records to a proper dictionary format
            sample_data = {
                'records': sample_records,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict()
            }
            
            data_sample = DataSample(
                artifact_id=artifact_id,
                sample_data=sample_data,
                total_rows=len(df),
                total_columns=len(df.columns),
                column_info=df.dtypes.astype(str).to_dict(),
                created_at=datetime.utcnow()
            )
            supabase_manager.insert_data_sample(data_sample)
        except Exception as e:
            print(f"Warning: Failed to create data sample: {str(e)}")
        
        # Create basic data quality metrics
        try:
            import pandas as pd
            df = pd.read_csv(processing_result["output_path"])
            
            quality_metrics = DataQualityMetrics(
                artifact_id=artifact_id,
                missing_values=df.isnull().sum().to_dict(),
                outliers={},  # Could be enhanced with outlier detection
                duplicates=df.duplicated().sum(),
                data_types=df.dtypes.astype(str).to_dict(),
                statistics={
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "operation": operation_name
                },
                created_at=datetime.utcnow()
            )
            supabase_manager.insert_data_quality_metrics(quality_metrics)
        except Exception as e:
            print(f"Warning: Failed to create data quality metrics: {str(e)}")
        
        execution_time = time.time() - start_time
        
        return PreprocessResponse(
            success=True,
            message=f"Preprocessing operation '{operation_name}' completed successfully",
            output_key=output_key,
            artifact_id=artifact_id,
            meta={
                "operation": operation_name,
                "module_path": module_path,
                "input_key": request.dataset_key,
                "params": request.params,
                "dataset_id": dataset_id
            },
            changes_summary=processing_result.get("meta", {}),
            data_quality={},  # To be populated by analysis
            logs=[
                processing_result.get("stdout", ""),
                processing_result.get("stderr", "")
            ],
            execution_time=execution_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error in {operation_name}: {str(e)}"
        )
