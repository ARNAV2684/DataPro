"""
Model Training endpoints for the Garuda ML Pipeline

Handles training of various ML models including Logistic Regression, Random Forest,
XGBoost, and DistilBERT fine-tuning for text classification.
"""

import time
import os
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
import json
import tempfile

logger = logging.getLogger(__name__)

from shared.types import ModelRequest, ModelResponse, Artifact, PipelineStage, ProcessingStatus
from shared.supabase_utils import get_supabase_manager
from shared.module_runner import get_module_runner

router = APIRouter(prefix="/api/model", tags=["model"])

# ===================================
# MODEL TRAINING ENDPOINTS
# ===================================

@router.post("/logistic-regression", response_model=ModelResponse)
async def train_logistic_regression(request: ModelRequest):
    """
    Train a Logistic Regression model with mixed features (numeric + text via TF-IDF)
    
    Runs: 6week/models/logistic_regression.py
    
    Parameters:
    - target_column: str - Name of the target/label column  
    - max_features: int - Maximum TF-IDF features (default: 10000)
    - C: float - Regularization parameter (default: 1.0)
    - max_iter: int - Maximum iterations (default: 1000)
    """
    return await _run_training_module(
        request=request,
        module_path="6week/models/logistic_regression.py",
        model_name="logistic_regression",
        model_type="Logistic Regression"
    )

@router.post("/random-forest", response_model=ModelResponse)
async def train_random_forest(request: ModelRequest):
    """
    Train a Random Forest model with mixed features (numeric + text)
    
    Runs: 6week/models/random_forest.py
    
    Parameters:
    - target_column: str - Name of the target/label column
    - n_estimators: int - Number of trees (default: 100)
    - max_depth: int - Maximum tree depth (default: None)
    - min_samples_split: int - Minimum samples to split (default: 2)
    - max_features: int - Maximum TF-IDF features (default: 5000)
    """
    return await _run_training_module(
        request=request,
        module_path="6week/models/random_forest.py", 
        model_name="random_forest",
        model_type="Random Forest"
    )

@router.post("/xgboost", response_model=ModelResponse)
async def train_xgboost(request: ModelRequest):
    """
    Train an XGBoost model with early stopping and hyperparameter tuning
    
    Runs: 6week/models/xgboost_train.py
    
    Parameters:
    - target_column: str - Name of the target/label column
    - n_estimators: int - Number of boosting rounds (default: 100)
    - learning_rate: float - Learning rate (default: 0.1)
    - max_depth: int - Maximum tree depth (default: 6)
    - early_stopping_rounds: int - Early stopping patience (default: 10)
    """
    return await _run_training_module(
        request=request,
        module_path="6week/models/xgboost_train.py",
        model_name="xgboost",
        model_type="XGBoost"
    )

@router.post("/gradient-boosting", response_model=ModelResponse)
async def train_gradient_boosting(request: ModelRequest):
    """
    Train a Gradient Boosting model with mixed features (numeric + text)
    
    Runs: 6week/models/gradient_boosting.py
    
    Parameters:
    - target_column: str - Name of the target/label column
    - n_estimators: int - Number of boosting stages (default: 100)
    - learning_rate: float - Learning rate (default: 0.1)
    - max_depth: int - Maximum tree depth (default: 3)
    - subsample: float - Subsample ratio (default: 1.0)
    """
    return await _run_training_module(
        request=request,
        module_path="6week/models/gradient_boosting.py",
        model_name="gradient_boosting",
        model_type="Gradient Boosting"
    )

@router.post("/distilbert-finetune", response_model=ModelResponse)
async def train_distilbert(request: ModelRequest):
    """
    Fine-tune DistilBERT model for text classification
    
    Runs: 6week/models/distilbert_finetune.py
    
    Parameters:
    - target_column: str - Name of the target/label column
    - text_column: str - Name of the text column (auto-detected if not provided)
    - num_epochs: int - Number of training epochs (default: 3)
    - learning_rate: float - Learning rate (default: 2e-5)
    - batch_size: int - Training batch size (default: 16)
    - max_length: int - Maximum sequence length (default: 512)
    """
    return await _run_training_module(
        request=request,
        module_path="6week/models/distilbert_finetune.py",
        model_name="distilbert",
        model_type="DistilBERT Fine-tuning"
    )

# ===================================
# HELPER FUNCTION
# ===================================

async def _run_training_module(request: ModelRequest, module_path: str, 
                              model_name: str, model_type: str) -> ModelResponse:
    """
    Generic function to run model training modules
    
    Args:
        request: Model training request
        module_path: Path to the Python training module
        model_name: Name of the model for storage
        model_type: Human-readable model type description
        
    Returns:
        ModelResponse with training results and model artifacts
    """
    start_time = time.time()
    
    try:
        # Get services
        supabase_manager = get_supabase_manager()
        module_runner = get_module_runner()
        
        # Extract dataset_id and user_id from input key
        path_parts = request.dataset_key.split('/')
        if len(path_parts) >= 3 and path_parts[0].startswith('user_') and path_parts[1].startswith('dataset_'):
            user_id = path_parts[0].split('_', 1)[1]  # Extract ID from 'user_123'
            dataset_id = path_parts[1].split('_', 1)[1]  # Extract ID from 'dataset_456'
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid dataset key format: {request.dataset_key}"
            )
        
        # For model training, we need the actual dataset, not EDA results
        # Try to download the original dataset from multiple buckets
        input_bucket = None
        download_result = None
        
        # Try buckets in order: preprocessed -> augmented -> datasets
        for bucket in ["preprocessed", "augmented", "datasets"]:
            try:
                logger.info(f"Attempting to download training data from {bucket} bucket: {request.dataset_key}")
                download_result = supabase_manager.download_file(
                    bucket_name=supabase_manager.buckets[bucket],
                    storage_key=request.dataset_key
                )
                if download_result["success"]:
                    input_bucket = bucket
                    logger.info(f"Successfully downloaded training data from {bucket} bucket")
                    break
                else:
                    logger.warning(f"Download failed from {bucket}: {download_result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.warning(f"Exception when downloading from {bucket} bucket: {e}")
                continue
        
        if not download_result or not download_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Failed to download dataset from any bucket (preprocessed, augmented, datasets). Key: {request.dataset_key}"
            )
        
        input_file = download_result["local_path"]
        
        # Create output directory for model artifacts
        output_dir = module_runner.temp_dir / f"model_{model_name}_{dataset_id}"
        output_dir.mkdir(exist_ok=True)
        
        # Prepare CLI arguments for training
        cli_args = [
            "--data-path", input_file,
            "--target-col", request.params.get("target_column", "label"),
            "--output-dir", str(output_dir)
        ]
        
        # Add hyperparameters and other params as CLI arguments
        for key, value in {**request.hyperparameters, **request.params}.items():
            if key != "target_column":  # Already added above
                # Convert key to CLI format (snake_case to kebab-case)
                cli_key = key.replace("_", "-")
                cli_args.extend([f"--{cli_key}", str(value)])
        
        # Run the training module using CLI interface
        training_result = module_runner.run_cli_script(
            module_path=module_path,
            cli_args=cli_args
        )
        
        if not training_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model training failed: {training_result['error']}"
            )
        
        # Find and upload model artifacts
        model_artifacts = []
        model_size = 0
        metrics_data = {}
        
        # Look for common model file extensions
        model_extensions = ['.pkl', '.joblib', '.json', '.pt', '.pth', '.bin', '.h5']
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                model_size += file_size
                
                # Generate storage key for this artifact
                artifact_key = supabase_manager.generate_storage_key(
                    user_id=request.user_id,
                    dataset_id=dataset_id,
                    stage="models",
                    filename=f"{model_name}_{file_path.name}"
                )
                
                # Upload artifact
                with open(file_path, 'rb') as f:
                    # Determine content type
                    content_type = "application/octet-stream"  # Default binary
                    if file_path.suffix == '.json':
                        content_type = "application/json"
                    elif file_path.suffix == '.txt':
                        content_type = "text/plain"
                    
                    upload_result = supabase_manager.upload_file(
                        bucket_name=supabase_manager.buckets["models"],
                        storage_key=artifact_key,
                        file_data=f.read(),
                        content_type=content_type
                    )
                
                if upload_result["success"]:
                    model_artifacts.append({
                        "filename": file_path.name,
                        "storage_key": artifact_key,
                        "size": file_size,
                        "type": file_path.suffix
                    })
        
        # Try to extract metrics from training result and output files
        # First check for metrics.json file in output directory
        metrics_file = output_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics_json = json.load(f)
                    if "test_metrics" in metrics_json:
                        # Filter test_metrics to only include simple numeric values
                        test_metrics = metrics_json["test_metrics"]
                        for key, value in test_metrics.items():
                            # Only include simple numeric metrics, skip complex nested objects
                            if isinstance(value, (int, float)) and key != "classification_report":
                                metrics_data[key] = value
            except Exception as e:
                logger.warning(f"Failed to load metrics from {metrics_file}: {e}")
        
        # Also try to extract metrics from stdout if available
        if training_result.get("stdout"):
            stdout = training_result["stdout"]
            # Look for common metric patterns in output
            import re
            accuracy_match = re.search(r"Test Accuracy:\s*([\d.]+)", stdout)
            if accuracy_match:
                metrics_data["accuracy"] = float(accuracy_match.group(1))
            
            f1_match = re.search(r"Test F1-Score:\s*([\d.]+)", stdout)
            if f1_match:
                metrics_data["f1_score"] = float(f1_match.group(1))
        
        # Generate model ID
        model_id = f"{model_name}_{request.user_id}_{dataset_id}_{int(time.time())}"
        
        # Use the primary model file as the main output key
        primary_artifact = model_artifacts[0] if model_artifacts else None
        primary_output_key = primary_artifact["storage_key"] if primary_artifact else None
        
        # Create a model metadata file if no artifacts were found
        if not primary_output_key:
            metadata_key = supabase_manager.generate_storage_key(
                user_id=request.user_id,
                dataset_id=dataset_id,
                stage="models",
                filename=f"{model_name}_metadata.json"
            )
            
            metadata = {
                "model_id": model_id,
                "model_type": model_type,
                "model_name": model_name,
                "training_args": cli_args,
                "metrics": metrics_data,
                "timestamp": datetime.utcnow().isoformat(),
                "training_result": {
                    "stdout": training_result.get("stdout", ""),
                    "stderr": training_result.get("stderr", "")
                }
            }
            
            upload_result = supabase_manager.upload_file(
                bucket_name=supabase_manager.buckets["models"],
                storage_key=metadata_key,
                file_data=json.dumps(metadata, indent=2).encode(),
                content_type="application/json"
            )
            
            if upload_result["success"]:
                primary_output_key = metadata_key
        
        # Create artifact record
        artifact = Artifact(
            user_id=request.user_id,
            dataset_id=dataset_id,
            stage=PipelineStage.MODEL,
            bucket_key=primary_output_key or "",
            status=ProcessingStatus.COMPLETED,
            meta={
                "model_id": model_id,
                "model_type": model_type,
                "model_name": model_name,
                "module_path": module_path,
                "input_key": request.dataset_key,
                "training_args": cli_args,
                "artifacts": model_artifacts,
                "metrics": metrics_data,
                "model_size": model_size,
                "training_result": {
                    "stdout": training_result.get("stdout", ""),
                    "stderr": training_result.get("stderr", "")
                }
            },
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Insert artifact into database
        db_result = supabase_manager.insert_artifact(artifact)
        
        if not db_result["success"]:
            # Log error but don't fail the request
            print(f"Warning: Failed to save artifact metadata: {db_result['error']}")
        
        training_time = time.time() - start_time
        
        # Serialize metrics to ensure proper format for ModelResponse
        # ModelResponse.metrics expects Dict[str, float], but classification_report is a dict
        # So we'll put complex metrics in meta and keep simple metrics in metrics
        simple_metrics = {}
        complex_metrics = {}
        
        for key, value in metrics_data.items():
            if isinstance(value, (int, float)):
                simple_metrics[key] = float(value)
            else:
                complex_metrics[key] = value
        
        # Also load the full metrics JSON for metadata
        full_metrics = {}
        for file_path in output_dir.rglob("*.json"):
            if "metadata" in file_path.name:
                try:
                    with open(file_path, 'r') as f:
                        metadata_json = json.load(f)
                        if "metrics" in metadata_json:
                            full_metrics = metadata_json["metrics"]
                        break
                except Exception:
                    pass
        
        return ModelResponse(
            success=True,
            message=f"{model_type} model training completed successfully",
            output_key=primary_output_key,
            meta={
                "model_type": model_type,
                "model_name": model_name,
                "module_path": module_path,
                "input_key": request.dataset_key,
                "training_args": cli_args,
                "artifacts_count": len(model_artifacts),
                "detailed_metrics": full_metrics,  # Full metrics including classification_report
                "complex_metrics": complex_metrics  # Any complex metrics that couldn't fit in metrics field
            },
            model_id=model_id,
            metrics=simple_metrics,  # Only simple numeric metrics
            model_size=model_size,
            training_time=training_time,
            logs=[
                training_result.get("stdout", ""),
                training_result.get("stderr", "")
            ],
            execution_time=training_time,
            timestamp=datetime.utcnow().isoformat()  # Convert to ISO string
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        training_time = time.time() - start_time
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error in {model_type} training: {str(e)}"
        )
