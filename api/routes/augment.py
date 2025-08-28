"""
Augmentation endpoints for the Garuda ML Pipeline

Handles data augmentation operations including SMOTE, noise augmentation,
synonym replacement, and other data augmentation techniques.
"""

import time
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any

from shared.types import (
    AugmentRequest, AugmentResponse,
    PipelineArtifact, DataSample, ProcessingLog, DataQualityMetrics,
    PipelineStage, ProcessingStatus
)
from shared.supabase_utils import get_supabase_manager
from shared.module_runner import get_module_runner

router = APIRouter(prefix="/api/augment", tags=["augment"])

@router.post("/smote", response_model=AugmentResponse)
async def smote_augmentation(request: AugmentRequest):
    """
    Perform SMOTE (Synthetic Minority Oversampling Technique) augmentation
    
    Runs: 4and5week/Augmentation/numericaug/smote_aug_cli.py
    
    Parameters:
    - target: str - Name of the target column for classification (required)
    - k_neighbors: int - Number of nearest neighbors for SMOTE (default: 5)
    - columns: list - Specific feature columns to use (optional)
    """
    return await _run_augmentation_module(
        request=request,
        module_path="4and5week/Augmentation/numericaug/smote_aug_cli.py",
        technique_name="smote"
    )

@router.post("/noise", response_model=AugmentResponse)
async def noise_augmentation(request: AugmentRequest):
    """
    Perform noise-based data augmentation by adding Gaussian noise
    
    Runs: 4and5week/Augmentation/numericaug/noise_aug_cli.py
    
    Parameters:
    - noise_factor: float - Factor controlling noise intensity (default: 0.1)
    - augmentation_factor: int - Number of augmented samples per original (default: 2)
    - columns: list - Specific columns to augment (default: all numeric)
    """
    print(f"[DEBUG] Noise augmentation request received: {request.dataset_key}")
    print(f"[DEBUG] Parameters: {request.params}")
    
    return await _run_augmentation_module(
        request=request,
        module_path="4and5week/Augmentation/numericaug/noise_aug_cli.py",
        technique_name="noise"
    )

@router.post("/scale", response_model=AugmentResponse)
async def scale_augmentation(request: AugmentRequest):
    """
    Perform scale and jitter based data augmentation
    
    Runs: 4and5week/Augmentation/numericaug/scale_aug_cli.py
    
    Parameters:
    - scale_min: float - Minimum scaling factor (default: 0.8)
    - scale_max: float - Maximum scaling factor (default: 1.2)
    - jitter: float - Jitter factor for noise (default: 0.1)
    - columns: list - Specific columns to augment (optional)
    """
    return await _run_augmentation_module(
        request=request,
        module_path="4and5week/Augmentation/numericaug/scale_aug_cli.py",
        technique_name="scale"
    )

@router.post("/synonym", response_model=AugmentResponse)
async def synonym_augmentation(request: AugmentRequest):
    """
    Perform synonym replacement augmentation for text data
    
    Runs: 4and5week/Augmentation/textaug/synonymaug.py
    
    Parameters:
    - replacement_rate: float - Proportion of words to replace (default: 0.1)
    - augmentation_factor: int - Number of augmented samples per original
    - text_column: str - Name of the text column to augment
    """
    return await _run_augmentation_module(
        request=request,
        module_path="4and5week/Augmentation/textaug/synonymaug.py",
        technique_name="synonym"
    )

@router.post("/mlm", response_model=AugmentResponse)
async def mlm_augmentation(request: AugmentRequest):
    """
    Perform Masked Language Model (MLM) based text augmentation
    
    Runs: 4and5week/Augmentation/textaug/mlm_aug.py
    
    Parameters:
    - mask_rate: float - Proportion of tokens to mask (default: 0.15)
    - augmentation_factor: int - Number of augmented samples per original
    - text_column: str - Name of the text column to augment
    - model_name: str - Hugging Face model name (default: 'bert-base-uncased')
    """
    return await _run_augmentation_module(
        request=request,
        module_path="4and5week/Augmentation/textaug/mlm_aug.py",
        technique_name="mlm"
    )

@router.post("/random", response_model=AugmentResponse)
async def random_augmentation(request: AugmentRequest):
    """
    Perform random text augmentation (insertion, deletion, swapping)
    
    Runs: 4and5week/Augmentation/textaug/random_aug.py
    
    Parameters:
    - insertion_rate: float - Rate of random word insertion
    - deletion_rate: float - Rate of random word deletion
    - swap_rate: float - Rate of random word swapping
    - augmentation_factor: int - Number of augmented samples per original
    """
    return await _run_augmentation_module(
        request=request,
        module_path="4and5week/Augmentation/textaug/random_aug.py",
        technique_name="random"
    )

# ===================================
# HELPER FUNCTION
# ===================================

async def _run_augmentation_module(request: AugmentRequest, module_path: str, 
                                 technique_name: str) -> AugmentResponse:
    """
    Generic function to run augmentation modules
    
    Args:
        request: Augmentation request
        module_path: Path to the Python module
        technique_name: Name of the augmentation technique
        
    Returns:
        AugmentResponse with results
    """
    start_time = time.time()
    
    try:
        print(f"[DEBUG] Starting {technique_name} augmentation")
        
        # Get services
        supabase_manager = get_supabase_manager()
        module_runner = get_module_runner()
        
        print(f"[DEBUG] Services initialized")
        
        # Download input dataset - try preprocessed bucket first, then datasets bucket
        print(f"[DEBUG] Downloading dataset: {request.dataset_key}")
        download_result = supabase_manager.download_file(
            bucket_name=supabase_manager.buckets["preprocessed"],
            storage_key=request.dataset_key
        )
        
        # If not found in preprocessed bucket, try datasets bucket (fallback to original data)
        if not download_result["success"]:
            print(f"[DEBUG] Not found in preprocessed bucket, trying datasets bucket")
            download_result = supabase_manager.download_file(
                bucket_name=supabase_manager.buckets["datasets"],
                storage_key=request.dataset_key
            )
        
        if not download_result["success"]:
            print(f"[DEBUG] Dataset download failed: {download_result['error']}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Failed to download dataset from both preprocessed and datasets buckets: {download_result['error']}"
            )
        
        input_file = download_result["local_path"]
        print(f"[DEBUG] Dataset downloaded to: {input_file}")
        
        # Run the augmentation module using CLI script
        processing_result = module_runner.run_module_script(
            module_path=module_path,
            input_file=input_file,
            params=request.params
        )
        
        if not processing_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Augmentation failed: {processing_result['error']}"
            )
        
        # Extract dataset_id from input key (user_123/dataset_456/...)
        try:
            path_parts = request.dataset_key.split('/')
            if len(path_parts) >= 2:
                # Look for dataset_id in any part that starts with 'dataset_'
                dataset_part = None
                for part in path_parts:
                    if part.startswith('dataset_'):
                        dataset_part = part
                        break
                
                if dataset_part:
                    # Extract everything after 'dataset_'
                    dataset_id = dataset_part[8:]  # Remove 'dataset_' prefix
                else:
                    # Fallback: use timestamp
                    dataset_id = f"augmented_{int(time.time())}"
            else:
                # Fallback: use timestamp  
                dataset_id = f"augmented_{int(time.time())}"
        except Exception as e:
            # Fallback: use timestamp
            dataset_id = f"augmented_{int(time.time())}"
        
        # Generate output storage key
        output_key = supabase_manager.generate_storage_key(
            user_id=request.user_id,
            dataset_id=dataset_id,
            stage="augmented",
            filename=f"{technique_name}_augmented_data.csv"
        )
        
        # Upload augmented data to storage
        with open(processing_result["output_path"], 'rb') as f:
            upload_result = supabase_manager.upload_file(
                bucket_name=supabase_manager.buckets["augmented"],
                storage_key=output_key,
                file_data=f.read(),
                content_type="text/csv"
            )
        
        if not upload_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload augmented data: {upload_result['error']}"
            )
        
        # Calculate augmentation metrics
        try:
            import pandas as pd
            original_df = pd.read_csv(input_file)
            augmented_df = pd.read_csv(processing_result["output_path"])
            
            original_size = len(original_df)
            augmented_size = len(augmented_df)
            augmentation_ratio = augmented_size / original_size if original_size > 0 else 0
        except Exception as e:
            # If we can't read the files, use defaults
            original_size = None
            augmented_size = None
            augmentation_ratio = None
        
        # Create pipeline artifact record
        import uuid
        artifact_id = str(uuid.uuid4())
        
        artifact = PipelineArtifact(
            id=artifact_id,
            user_id=request.user_id,
            dataset_id=dataset_id,
            stage=PipelineStage.AUGMENT,
            operation=technique_name,
            bucket_key=output_key,
            status=ProcessingStatus.COMPLETED,
            parameters=request.params,
            metadata={
                "module_path": module_path,
                "input_key": request.dataset_key,
                "original_size": original_size,
                "augmented_size": augmented_size,
                "augmentation_ratio": augmentation_ratio,
                "processing_result": processing_result.get("meta", {})
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
                message="Augmentation completed successfully",
                details={
                    "stdout": processing_result["stdout"],
                    "technique": technique_name
                },
                created_at=datetime.utcnow()
            )
            supabase_manager.insert_processing_log(log_entry)
        
        if processing_result.get("stderr"):
            log_entry = ProcessingLog(
                artifact_id=artifact_id,
                level="warning",
                message="Augmentation stderr output",
                details={
                    "stderr": processing_result["stderr"],
                    "technique": technique_name
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
        
        # Create data quality metrics
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
                    "technique": technique_name,
                    "original_size": original_size,
                    "augmented_size": augmented_size,
                    "augmentation_ratio": augmentation_ratio
                },
                created_at=datetime.utcnow()
            )
            supabase_manager.insert_data_quality_metrics(quality_metrics)
        except Exception as e:
            print(f"Warning: Failed to create data quality metrics: {str(e)}")
        
        execution_time = time.time() - start_time
        
        try:
            response = AugmentResponse(
                success=True,
                message=f"Augmentation technique '{technique_name}' completed successfully",
                output_key=output_key,
                artifact_id=artifact_id,
                meta={
                    "technique": technique_name,
                    "module_path": module_path,
                    "input_key": request.dataset_key,
                    "params": request.params,
                    "dataset_id": dataset_id
                },
                original_size=original_size,
                augmented_size=augmented_size,
                augmentation_ratio=augmentation_ratio,
                logs=[
                    processing_result.get("stdout", ""),
                    processing_result.get("stderr", "")
                ],
                execution_time=execution_time,
                timestamp=datetime.utcnow().isoformat()
            )
            return response
        except Exception as response_error:
            print(f"Error creating AugmentResponse: {response_error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Response creation failed: {str(response_error)}"
            )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error in {technique_name} augmentation: {str(e)}"
        )
