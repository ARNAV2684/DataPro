"""
Upload endpoints for the Garuda ML Pipeline

Handles dataset uploading and storage to Supabase Storage buckets.
"""

import uuid
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from typing import Optional

from shared.types import (
    UploadRequest, UploadResponse, 
    Dataset, DataSample, ProcessingLog,
    DataType, ProcessingStatus
)
from shared.supabase_utils import get_supabase_manager
from shared.module_runner import get_module_runner

router = APIRouter(prefix="/api/upload", tags=["upload"])

@router.post("/dataset", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    data_type: DataType = Form(...),
    description: Optional[str] = Form(None)
):
    """
    Upload a dataset file to Supabase Storage
    
    Args:
        file: The dataset file to upload
        user_id: User ID for authentication and storage isolation
        data_type: Type of data (numeric, text, image, mixed)
        description: Optional description of the dataset
        
    Returns:
        UploadResponse with dataset_id and storage information
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )
        
        # Generate dataset ID and storage key
        dataset_id = str(uuid.uuid4())
        supabase_manager = get_supabase_manager()
        
        storage_key = supabase_manager.generate_storage_key(
            user_id=user_id,
            dataset_id=dataset_id, 
            stage="upload",
            filename=file.filename
        )
        
        # Read file data
        file_data = await file.read()
        file_size = len(file_data)
        
        # Upload to Supabase Storage
        upload_result = supabase_manager.upload_file(
            bucket_name=supabase_manager.buckets["datasets"],
            storage_key=storage_key,
            file_data=file_data,
            content_type=file.content_type or "application/octet-stream"
        )
        
        if not upload_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload file: {upload_result['error']}"
            )
        
        # Create dataset metadata for new database schema
        dataset = Dataset(
            id=dataset_id,
            user_id=user_id,
            bucket_key=storage_key,
            filename=file.filename,
            filetype=file.filename.split('.')[-1] if '.' in file.filename else '',
            data_type=data_type,
            file_size=file_size,
            status=ProcessingStatus.COMPLETED,
            description=description,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Insert dataset metadata into database
        db_result = supabase_manager.insert_dataset(dataset)
        
        if not db_result["success"]:
            # Try to clean up uploaded file on DB failure
            supabase_manager.delete_file(
                supabase_manager.buckets["datasets"], 
                storage_key
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save dataset metadata: {db_result['error']}"
            )
        
        # Create data sample for quick preview (if file is CSV)
        try:
            if file.filename.lower().endswith('.csv'):
                import pandas as pd
                import io
                
                # Read CSV for preview
                df = pd.read_csv(io.BytesIO(file_data))
                sample_size = min(5, len(df))
                sample_data = df.head(sample_size).to_dict(orient='records')
                
                # For raw uploads, we'll skip creating DataSample since it requires artifact_id
                # This can be created later when the dataset is processed
                print(f"Dataset preview: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"Warning: Failed to create data sample for uploaded file: {str(e)}")
        
        return UploadResponse(
            success=True,
            message=f"Dataset '{file.filename}' uploaded successfully",
            output_key=storage_key,
            dataset_id=dataset_id,
            meta={
                "original_filename": file.filename,
                "file_size": file_size,
                "data_type": data_type.value,
                "content_type": file.content_type
            },
            file_info={
                "filename": file.filename,
                "size": file_size,
                "type": file.content_type,
                "data_type": data_type.value
            },
            execution_time=0.0,  # Upload time not tracked yet
            timestamp=datetime.utcnow().isoformat()  # Convert to string
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during upload: {str(e)}"
        )
