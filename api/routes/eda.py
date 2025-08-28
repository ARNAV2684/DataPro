"""
EDA (Exploratory Data Analysis) endpoints for the Garuda ML Pipeline

Handles both numeric and text EDA operations including correlation analysis,
statistical analysis, visualizations, sentiment analysis, and more.
"""

import time
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, List
import os
import tempfile

from shared.types import EDARequest, EDAResponse, Artifact, PipelineStage, ProcessingStatus
from shared.supabase_utils import get_supabase_manager
from shared.module_runner import get_module_runner

router = APIRouter(prefix="/api/eda", tags=["eda"])
logger = logging.getLogger(__name__)

# ===================================
# NUMERIC EDA ENDPOINTS
# ===================================

@router.post("/statistical-analysis", response_model=EDAResponse)
async def statistical_analysis(request: EDARequest):
    """
    Perform comprehensive statistical analysis using CLI
    
    Runs: EDA/numeric_EDA/statistical_analysis_cli.py
    
    Parameters:
    - no_plots: bool - Skip generating visualization plots
    - plots_only: bool - Generate only plots, skip statistical analysis
    """
    return _run_eda_cli(
        request=request,
        module_path="EDA/numeric_EDA/statistical_analysis_cli.py",
        analysis_name="statistical_analysis"
    )

@router.post("/correlation-analysis", response_model=EDAResponse)
async def correlation_analysis(request: EDARequest):
    """
    Perform correlation analysis using CLI
    
    Runs: EDA/numeric_EDA/correlation_analysis_cli.py
    
    Parameters:
    - threshold: float - Correlation threshold for identifying high correlations (default: 0.7)
    - method: str - Correlation method ('pearson', 'spearman', 'both')
    - no_plots: bool - Skip generating visualization plots
    """
    return _run_eda_cli(
        request=request,
        module_path="EDA/numeric_EDA/correlation_analysis_cli.py",
        analysis_name="correlation_analysis"
    )@router.post("/advanced-visualization", response_model=EDAResponse)
async def advanced_visualization(request: EDARequest):
    """
    Generate advanced visualizations using CLI
    
    Runs: EDA/numeric_EDA/advanced_visualization_cli.py
    
    Parameters:
    - pca_components: int - Number of PCA components to compute
    - sample_size: int - Sample size for pair plots (default: 1000)
    - skip_pca: bool - Skip PCA analysis
    - skip_pairs: bool - Skip pair plot generation
    - plots_only: bool - Generate only plots, skip result CSV
    """
    return _run_eda_cli(
        request=request,
        module_path="EDA/numeric_EDA/advanced_visualization_cli.py",
        analysis_name="advanced_visualization"
    )

@router.post("/eda-manager", response_model=EDAResponse)
async def eda_manager(request: EDARequest):
    """
    Run comprehensive EDA using the manager CLI
    
    Runs: EDA/numeric_EDA/eda_manager_cli.py
    
    Parameters:
    - technique: str - Specific EDA technique to run
    - all: bool - Run all available EDA techniques
    - threshold: float - Correlation threshold for correlation_analysis
    - pca_components: int - Number of PCA components for advanced_visualization
    - sample_size: int - Sample size for pair plots
    """
    return _run_eda_cli(
        request=request,
        module_path="EDA/numeric_EDA/eda_manager_cli.py",
        analysis_name="eda_manager"
    )

# ===================================
# TEXT EDA ENDPOINTS
# ===================================

@router.post("/sentiment-analysis", response_model=EDAResponse)
async def sentiment_analysis(request: EDARequest):
    """
    Perform sentiment analysis on text data
    
    Runs: EDA/text_EDA/sentiment.py
    
    Parameters:
    - text_column: str - Name of the text column to analyze
    - sentiment_engine: str - Sentiment analysis engine ('textblob', 'vader')
    - include_visualization: bool - Generate sentiment plots
    """
    return await _run_eda_module(
        request=request,
        module_path="EDA/text_EDA/sentiment.py",
        analysis_name="sentiment_analysis",
        is_visualization=True
    )

@router.post("/word-frequency", response_model=EDAResponse)
async def word_frequency_analysis(request: EDARequest):
    """
    Analyze word frequency and generate word clouds
    
    Runs: EDA/text_EDA/word_freq.py
    
    Parameters:
    - text_column: str - Name of the text column to analyze
    - top_n: int - Number of top words to analyze (default: 50)
    - remove_stopwords: bool - Remove common stopwords
    - min_word_length: int - Minimum word length to include
    """
    return await _run_eda_module(
        request=request,
        module_path="EDA/text_EDA/word_freq.py",
        analysis_name="word_frequency",
        is_visualization=True
    )

@router.post("/text-length", response_model=EDAResponse)
async def text_length_analysis(request: EDARequest):
    """
    Analyze text length patterns and statistics
    
    Runs: EDA/text_EDA/text_length.py
    
    Parameters:
    - text_column: str - Name of the text column to analyze
    - unit: str - Unit of measurement ('characters', 'words', 'sentences')
    - include_distribution: bool - Generate length distribution plots
    """
    return await _run_eda_module(
        request=request,
        module_path="EDA/text_EDA/text_length.py",
        analysis_name="text_length",
        is_visualization=True
    )

@router.post("/topic-modeling", response_model=EDAResponse)
async def topic_modeling(request: EDARequest):
    """
    Perform topic modeling and distribution analysis
    
    Runs: EDA/text_EDA/topic_dist.py
    
    Parameters:
    - text_column: str - Name of the text column to analyze
    - num_topics: int - Number of topics to extract (default: 5)
    - algorithm: str - Topic modeling algorithm ('lda', 'nmf')
    - max_features: int - Maximum number of features for vectorization
    """
    return await _run_eda_module(
        request=request,
        module_path="EDA/text_EDA/topic_dist.py",
        analysis_name="topic_modeling",
        is_visualization=True
    )

@router.post("/ngram-analysis", response_model=EDAResponse)
async def ngram_analysis(request: EDARequest):
    """
    Perform N-gram analysis on text data
    
    Runs: EDA/text_EDA/ngram_analysis.py
    
    Parameters:
    - text_column: str - Name of the text column to analyze
    - ngram_range: tuple - N-gram range (e.g., (1, 2) for unigrams and bigrams)
    - top_n: int - Number of top N-grams to analyze
    - include_visualization: bool - Generate N-gram frequency plots
    """
    return await _run_eda_module(
        request=request,
        module_path="EDA/text_EDA/ngram_analysis.py",
        analysis_name="ngram_analysis",
        is_visualization=True
    )

# ===================================
# HELPER FUNCTION
# ===================================

def _run_eda_cli(request: EDARequest, module_path: str, analysis_name: str) -> EDAResponse:
    """Helper function to run EDA CLI scripts"""
    start_time = time.time()
    
    try:
        logger.info(f"Starting {analysis_name} EDA analysis")
        
        # Get services
        supabase_manager = get_supabase_manager()
        module_runner = get_module_runner()
        
        # Download input dataset - try multiple buckets
        input_bucket = None
        download_result = None
        
        # Try buckets in order: augmented -> preprocessed -> datasets
        for bucket in ["augmented", "preprocessed", "datasets"]:
            try:
                logger.info(f"Attempting to download from {bucket} bucket: {request.dataset_key}")
                download_result = supabase_manager.download_file(
                    bucket_name=supabase_manager.buckets[bucket],
                    storage_key=request.dataset_key
                )
                if download_result["success"]:
                    input_bucket = bucket
                    logger.info(f"Successfully downloaded from {bucket} bucket")
                    break
                else:
                    logger.warning(f"Download failed from {bucket}: {download_result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.warning(f"Exception when downloading from {bucket} bucket: {e}")
                continue
        
        if not download_result or not download_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Failed to download dataset from any bucket (augmented, preprocessed, datasets). Key: {request.dataset_key}"
            )
        
        input_file = download_result["local_path"]
        
        # Build CLI arguments from request parameters based on the specific script
        cli_args = [
            "--input", input_file,
            "--output", f"temp_{analysis_name}_output.csv"
        ]
        
        # Define valid parameters for each script to avoid argument errors
        valid_params = {
            "statistical_analysis": ["no_plots", "plots_only"],
            "correlation_analysis": ["threshold", "method", "no_plots"],
            "advanced_visualization": ["pca_components", "sample_size", "skip_pca", "skip_pairs", "plots_only"],
            "eda_manager": ["technique", "all", "threshold", "pca_components", "sample_size"]
        }
        
        # Get script name from module_path for parameter validation
        script_name = module_path.split('/')[-1].replace('_cli.py', '').replace('.py', '')
        allowed_params = valid_params.get(script_name, [])
        
        # Add optional parameters based on request, but only if they're valid for this script
        if hasattr(request, 'params') and request.params:
            for key, value in request.params.items():
                if key in allowed_params:
                    # Convert parameter names to CLI format (underscores to hyphens)
                    cli_key = f"--{key.replace('_', '-')}"
                    
                    # Handle boolean flags
                    if isinstance(value, bool):
                        if value:  # Only add flag if True
                            cli_args.append(cli_key)
                    else:
                        # Add parameter with value
                        cli_args.extend([cli_key, str(value)])
                else:
                    logger.warning(f"Ignoring invalid parameter '{key}' for {script_name}")
        
        # Run the CLI script using module runner
        processing_result = module_runner.run_cli_script(module_path, cli_args)
        
        if not processing_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"EDA analysis failed: {processing_result.get('error', 'Unknown error')}"
            )
        
        # Extract dataset_id from input key
        path_parts = request.dataset_key.split('/')
        logger.info(f"Dataset key path parts: {path_parts}")
        
        if len(path_parts) >= 2 and path_parts[0].startswith('user_') and path_parts[1].startswith('dataset_'):
            # Extract UUID from dataset_<uuid> format
            dataset_part = path_parts[1]  # e.g., 'dataset_6ff0d0cc-77bd-47ca-8143-e817149c2b8e'
            dataset_id = dataset_part[8:]  # Remove 'dataset_' prefix (8 characters)
            logger.info(f"Extracted dataset_id: {dataset_id}")
        else:
            logger.error(f"Invalid dataset key format - parts: {len(path_parts)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid dataset key format: {request.dataset_key}"
            )
        
        # Process generated files
        outputs_to_upload = []
        visualization_keys = []
        
        # Log current working directory and files
        current_dir = os.getcwd()
        logger.info(f"Current working directory: {current_dir}")
        logger.info(f"Files in current directory: {os.listdir(current_dir) if os.path.exists(current_dir) else 'Directory not found'}")
        
        # Check for output CSV file in the script's execution directory
        output_csv = f"temp_{analysis_name}_output.csv"
        
        # Determine the script execution directory based on module_path
        # For example: module_path = "EDA/numeric_EDA/statistical_analysis_cli.py"
        # We want the directory: "EDA/numeric_EDA"
        try:
            # Get workspace root (parent of api directory)
            workspace_root = os.path.dirname(current_dir)
            script_dir = os.path.join(workspace_root, os.path.dirname(module_path))
            script_output_csv = os.path.join(script_dir, output_csv)
        except Exception as e:
            logger.warning(f"Failed to calculate script directory: {e}")
            script_dir = current_dir
            script_output_csv = output_csv
        
        logger.info(f"Looking for output CSV: {output_csv}")
        logger.info(f"Script directory: {script_dir}")
        logger.info(f"Script output path: {script_output_csv}")
        logger.info(f"Output CSV exists in current dir: {os.path.exists(output_csv)}")
        logger.info(f"Output CSV exists in script dir: {os.path.exists(script_output_csv)}")
        
        # Use the correct path for the output file
        final_output_csv = script_output_csv if os.path.exists(script_output_csv) else output_csv
        
        if os.path.exists(final_output_csv):
            # Upload CSV results
            csv_key = supabase_manager.generate_storage_key(
                user_id=request.user_id,
                dataset_id=dataset_id,
                stage="eda",
                filename=f"{analysis_name}_results.csv"
            )
            
            with open(final_output_csv, 'rb') as f:
                upload_result = supabase_manager.upload_file(
                    bucket_name=supabase_manager.buckets["eda"],
                    storage_key=csv_key,
                    file_data=f.read(),
                    content_type="text/csv"
                )
            
            if upload_result["success"]:
                outputs_to_upload.append(csv_key)
                logger.info(f"Successfully uploaded CSV results: {csv_key}")
        
        # Look for generated plots in script directory and other locations
        plot_extensions = ['.png', '.jpg', '.jpeg', '.pdf', '.svg']
        search_dirs = [".", "EDA", "EDA/numeric_EDA"]
        
        # Add script directory to search paths if it exists and is different
        if os.path.exists(script_dir) and script_dir not in search_dirs:
            search_dirs.append(script_dir)
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for file in os.listdir(search_dir):
                    if any(file.lower().endswith(ext) for ext in plot_extensions):
                        plot_path = os.path.join(search_dir, file)
                        
                        # Upload plot to EDA bucket
                        plot_key = supabase_manager.generate_storage_key(
                            user_id=request.user_id,
                            dataset_id=dataset_id,
                            stage="eda",
                            filename=f"{analysis_name}_{file}"
                        )
                        
                        with open(plot_path, 'rb') as f:
                            content_type = "image/png" if file.lower().endswith('.png') else "image/jpeg"
                            upload_result = supabase_manager.upload_file(
                                bucket_name=supabase_manager.buckets["eda"],
                                storage_key=plot_key,
                                file_data=f.read(),
                                content_type=content_type
                            )
                        
                        if upload_result["success"]:
                            visualization_keys.append(plot_key)
                            logger.info(f"Successfully uploaded visualization: {plot_key}")
        
        # Use main output as primary output_key
        primary_output_key = outputs_to_upload[0] if outputs_to_upload else (
            visualization_keys[0] if visualization_keys else None
        )
        
        # Create summary if no outputs
        if not primary_output_key:
            summary_key = supabase_manager.generate_storage_key(
                user_id=request.user_id,
                dataset_id=dataset_id,
                stage="eda",
                filename=f"{analysis_name}_summary.json"
            )
            
            import json
            summary_data = {
                "analysis_type": analysis_name,
                "timestamp": datetime.utcnow().isoformat(),
                "params": getattr(request, 'params', {}),
                "results": processing_result.get("meta", {})
            }
            
            upload_result = supabase_manager.upload_file(
                bucket_name=supabase_manager.buckets["eda"],
                storage_key=summary_key,
                file_data=json.dumps(summary_data, indent=2).encode(),
                content_type="application/json"
            )
            
            if upload_result["success"]:
                primary_output_key = summary_key
        
        # Generate insights based on analysis type
        insights = []
        if analysis_name == "correlation_analysis":
            insights.append("Correlation analysis completed - check heatmap for feature relationships")
        elif analysis_name == "statistical_analysis":
            insights.append("Statistical analysis completed - check descriptive statistics and distributions")
        elif analysis_name == "advanced_visualization":
            insights.append("Advanced visualizations generated - check plots for data patterns")
        elif analysis_name == "eda_manager":
            insights.append("Comprehensive EDA completed - multiple analysis techniques applied")
        
        # Create artifact record
        artifact = Artifact(
            user_id=request.user_id,
            dataset_id=dataset_id,
            stage=PipelineStage.EDA,
            bucket_key=primary_output_key or "",
            status=ProcessingStatus.COMPLETED,
            meta={
                "analysis_type": analysis_name,
                "module_path": module_path,
                "input_key": request.dataset_key,
                "input_bucket": input_bucket,
                "params": getattr(request, 'params', {}),
                "visualizations": visualization_keys,
                "all_outputs": outputs_to_upload + visualization_keys,
                "cli_args": cli_args
            },
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Insert artifact into database
        db_result = supabase_manager.insert_artifact(artifact)
        
        if not db_result["success"]:
            logger.warning(f"Failed to save artifact metadata: {db_result.get('error', 'Unknown error')}")
        
        # Clean up temporary files
        try:
            if os.path.exists(input_file):
                os.remove(input_file)
            if os.path.exists(final_output_csv):
                os.remove(final_output_csv)
            
            # Clean up generated plots
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    for file in os.listdir(search_dir):
                        if any(file.lower().endswith(ext) for ext in plot_extensions):
                            try:
                                os.remove(os.path.join(search_dir, file))
                            except Exception:
                                pass
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")
        
        # Generate public URLs for visualizations
        visualization_urls = []
        for viz_key in visualization_keys:
            public_url = supabase_manager.get_public_url(
                bucket_name=supabase_manager.buckets["eda"],
                storage_key=viz_key
            )
            if public_url:
                visualization_urls.append({
                    "key": viz_key,
                    "url": public_url,
                    "filename": viz_key.split('/')[-1]
                })
        
        execution_time = time.time() - start_time
        
        logger.info(f"Successfully completed {analysis_name} EDA analysis")
        
        return EDAResponse(
            success=True,
            message=f"EDA analysis '{analysis_name}' completed successfully",
            output_key=primary_output_key,
            meta={
                "analysis_type": analysis_name,
                "module_path": module_path,
                "input_key": request.dataset_key,
                "input_bucket": input_bucket,
                "params": getattr(request, 'params', {}),
                "total_outputs": len(outputs_to_upload) + len(visualization_keys),
                "visualization_urls": visualization_urls
            },
            analysis_results=processing_result.get("meta", {}),
            visualizations=visualization_keys,
            insights=insights,
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
        logger.error(f"Error in {analysis_name} EDA analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error in {analysis_name} analysis: {str(e)}"
        )


async def _run_eda_module(request: EDARequest, module_path: str, analysis_name: str, is_visualization: bool = False) -> EDAResponse:
    """Legacy helper function for text EDA modules that don't have CLI versions yet"""
    start_time = time.time()
    
    try:
        logger.info(f"Starting {analysis_name} EDA analysis (legacy)")
        
        # Get services
        supabase_manager = get_supabase_manager()
        module_runner = get_module_runner()
        
        # Download input dataset from augmented bucket (or preprocessed as fallback)
        input_bucket = "augmented"
        try:
            download_result = supabase_manager.download_file(
                bucket_name=supabase_manager.buckets[input_bucket],
                storage_key=request.dataset_key
            )
        except Exception:
            # Fallback to preprocessed bucket
            input_bucket = "preprocessed"
            download_result = supabase_manager.download_file(
                bucket_name=supabase_manager.buckets[input_bucket],
                storage_key=request.dataset_key
            )
        
        if not download_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Failed to download dataset: {download_result['error']}"
            )
        
        input_file = download_result["local_path"]
        
        # Prepare EDA parameters
        eda_params = {
            "analysis_type": request.analysis_type,
            "output_format": request.output_format,
            **request.params
        }
        
        # Run the EDA module with visualization support
        if is_visualization:
            processing_result = module_runner.run_eda_module(
                module_path=module_path,
                input_file=input_file,
                analysis_type=request.analysis_type
            )
        else:
            processing_result = module_runner.adapt_legacy_module(
                module_path=module_path,
                input_file=input_file,
                params=eda_params
            )
        
        if not processing_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"EDA analysis failed: {processing_result.get('error', 'Unknown error')}"
            )
        
        # Extract dataset_id from input key
        path_parts = request.dataset_key.split('/')
        logger.info(f"[Text EDA] Dataset key path parts: {path_parts}")
        
        if len(path_parts) >= 2 and path_parts[0].startswith('user_') and path_parts[1].startswith('dataset_'):
            # Extract UUID from dataset_<uuid> format
            dataset_part = path_parts[1]  # e.g., 'dataset_6ff0d0cc-77bd-47ca-8143-e817149c2b8e'
            dataset_id = dataset_part[8:]  # Remove 'dataset_' prefix (8 characters)
            logger.info(f"[Text EDA] Extracted dataset_id: {dataset_id}")
        else:
            logger.error(f"[Text EDA] Invalid dataset key format - parts: {len(path_parts)}, user check: {path_parts[0].startswith('user_') if len(path_parts) > 0 else False}, dataset check: {path_parts[1].startswith('dataset_') if len(path_parts) > 1 else False}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid dataset key format: {request.dataset_key}"
            )
        
        # Handle basic response for now - this can be expanded when text CLI scripts are created
        return EDAResponse(
            success=True,
            message=f"EDA analysis '{analysis_name}' completed successfully",
            output_key="",
            meta={
                "analysis_type": analysis_name,
                "module_path": module_path,
                "input_key": request.dataset_key,
                "input_bucket": input_bucket,
                "params": eda_params
            },
            analysis_results=processing_result.get("meta", {}),
            visualizations=[],
            insights=[f"{analysis_name} analysis completed"],
            logs=[
                processing_result.get("stdout", ""),
                processing_result.get("stderr", "")
            ],
            execution_time=time.time() - start_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in {analysis_name} EDA analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error in {analysis_name} analysis: {str(e)}"
        )
