"""
Download endpoints for file retrieval from Supabase storage
"""

from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import StreamingResponse
import io
import urllib.parse

from shared.supabase_utils import get_supabase_manager

router = APIRouter(tags=["downloads"])

@router.get("/download/{bucket_name}/{file_path:path}")
async def download_file(
    bucket_name: str,
    file_path: str,
    user_id: str = Query(..., description="User ID for authentication")
):
    """
    Download a file from Supabase storage
    
    Args:
        bucket_name: Storage bucket name (e.g., 'preprocessed', 'augmented')
        file_path: Full file path in storage (including subdirectories)
        user_id: User ID for authentication
        
    Returns:
        File download response
    """
    try:
        supabase_manager = get_supabase_manager()
        
        # URL decode the file_path and user_id to handle encoded characters
        decoded_file_path = urllib.parse.unquote(file_path)
        decoded_user_id = urllib.parse.unquote(user_id)
        
        # Security check: ensure the file path contains the user_id
        if decoded_user_id not in decoded_file_path:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: File does not belong to user"
            )
        
        # Get file data from storage using the decoded path
        file_data = None
        try:
            file_data = supabase_manager.storage.from_(bucket_name).download(decoded_file_path)
        except Exception as e:
            # Try to find the correct storage key from artifacts table
            path_parts = decoded_file_path.split('/')
            
            # Handle different URL formats
            dataset_id = None
            filename = None
            user_email = None
            
            print(f"DEBUG: Path parts: {path_parts}")
            
            if len(path_parts) >= 2:
                first_part = path_parts[0]
                if '_' in first_part:
                    if first_part.startswith('boosting_') or first_part.startswith('logistic_') or first_part.startswith('random_') or first_part.startswith('xgboost_'):
                        # Format: modeltype_user@domain.com
                        parts = first_part.split('_', 1)
                        user_email = parts[1] if len(parts) > 1 else None
                        dataset_id = path_parts[1] if len(path_parts) > 1 else None
                        filename = path_parts[2] if len(path_parts) > 2 else None
                    else:
                        # Format: user@domain.com_dataset-id
                        at_index = first_part.find('@')
                        if at_index > 0:
                            underscore_after_at = first_part.find('_', at_index)
                            if underscore_after_at > 0:
                                user_email = first_part[:underscore_after_at]
                                dataset_id = first_part[underscore_after_at+1:]
                                filename = path_parts[-1] if len(path_parts) > 1 else None
                
                # If we couldn't parse from first part, try direct approach
                if not dataset_id and len(path_parts) >= 3:
                    dataset_id = path_parts[1]
                    filename = path_parts[-1]
                    if '@' in first_part:
                        user_email = first_part.split('_')[-1] if '_' in first_part else first_part
                
                print(f"DEBUG: Extracted - user_email: {user_email}, dataset_id: {dataset_id}, filename: {filename}")
                
                if dataset_id and filename:
                    # Query artifacts for this dataset to find the correct storage key
                    # Query artifacts for this dataset to find the correct storage key
                    artifacts_list = supabase_manager.get_artifacts_by_dataset(
                        dataset_id=dataset_id,
                        stage=None  # Get all stages
                    )
                    
                    if artifacts_list:
                        print(f"DEBUG: Found {len(artifacts_list)} artifacts for dataset")
                        for i, artifact in enumerate(artifacts_list):
                            bucket_key = artifact.get("bucket_key", "")
                            stage = artifact.get("stage", "")
                            print(f"DEBUG: Artifact {i+1}: stage={stage}, bucket_key={bucket_key}")
                        
                        # First, try to find exact extension matches
                        exact_matches = []
                        base_matches = []
                        
                        filename_ext = filename.split('.')[-1] if '.' in filename else ""
                        
                        for artifact in artifacts_list:
                            bucket_key = artifact.get("bucket_key", "")
                            bucket_ext = bucket_key.split('.')[-1] if '.' in bucket_key else ""
                            
                            # Exact extension match gets highest priority
                            if filename_ext == bucket_ext and filename_ext != "":
                                exact_matches.append(artifact)
                            else:
                                base_matches.append(artifact)
                        
                        # Sort exact matches by recency, then try base matches
                        candidates = sorted(exact_matches, key=lambda x: x.get("created_at", ""), reverse=True) + base_matches
                        
                        for artifact in candidates:
                            bucket_key = artifact.get("bucket_key", "")
                            
                            # Check multiple ways the filename might match
                            filename_base = filename.split('.')[0] if '.' in filename else filename
                            bucket_filename = bucket_key.split('/')[-1] if '/' in bucket_key else bucket_key
                            bucket_ext = bucket_key.split('.')[-1] if '.' in bucket_key else ""
                            
                            # For exact extension matches, be more lenient with base name matching
                            # For different extensions, require stronger matches
                            if filename_ext == bucket_ext:
                                matches = (
                                    filename in bucket_key or
                                    bucket_filename == filename or
                                    any(part in bucket_key for part in filename_base.split('_') if len(part) > 3)
                                )
                            else:
                                matches = (
                                    filename in bucket_key or
                                    bucket_filename == filename
                                )
                            
                            print(f"DEBUG: Testing {bucket_key} against {filename}")
                            print(f"DEBUG: - filename_ext: {filename_ext}, bucket_ext: {bucket_ext}")
                            print(f"DEBUG: - exact_ext_match: {filename_ext == bucket_ext}")
                            print(f"DEBUG: - matches: {matches}")
                            
                            if matches:
                                print(f"DEBUG: Found matching artifact with bucket_key: {bucket_key}")
                                try:
                                    file_data = supabase_manager.storage.from_(bucket_name).download(bucket_key)
                                    print(f"DEBUG: Success with database bucket_key!")
                                    print(f"DEBUG: Downloaded file_data type: {type(file_data)}")
                                    print(f"DEBUG: Downloaded file_data length: {len(file_data) if file_data else 'None'}")
                                    print(f"DEBUG: file_data is None: {file_data is None}")
                                    print(f"DEBUG: file_data is empty: {len(file_data) == 0 if file_data else 'N/A'}")
                                    break
                                except Exception as e3:
                                    print(f"DEBUG: Database bucket_key failed: {e3}")
                                    continue
                    else:
                        print(f"DEBUG: No artifacts found for dataset {dataset_id}")
            
            if not file_data:
                raise e  # Re-raise original exception
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"DEBUG: Unexpected error in download route: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download file: {str(e)}"
        )
    
    # Final validation and response creation
    print(f"DEBUG: About to check if file_data exists...")
    print(f"DEBUG: file_data is None: {file_data is None}")
    print(f"DEBUG: file_data type: {type(file_data)}")
    
    if not file_data:
        print(f"DEBUG: file_data is falsy, raising 404")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    print(f"DEBUG: Final file_data type: {type(file_data)}")
    print(f"DEBUG: Final file_data length: {len(file_data) if file_data else 'None'}")
    
    # Extract filename from decoded file_path
    filename = decoded_file_path.split('/')[-1] if '/' in decoded_file_path else decoded_file_path
    
    print(f"DEBUG: Returning file with filename: {filename}")
    
    try:
        response = StreamingResponse(
            io.BytesIO(file_data),
            media_type='application/octet-stream',
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        print(f"DEBUG: StreamingResponse created successfully")
        return response
    except Exception as e:
        print(f"DEBUG: Error creating StreamingResponse: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create download response: {str(e)}"
        )
