from fastapi import FastAPI, HTTPException, status
# Import route modules
from routes.upload import router as upload_router
from routes.preprocess import router as preprocess_router
from routes.augment import router as augment_router
from routes.eda import router as eda_router
from routes.model import router as model_router
from routes.datasets import router as datasets_router
from routes.data_retrieval import router as data_retrieval_router
from routes.download import router as download_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our models and services
from models.request_models import MixupRequest, MixupResponse
# from services.mixup_service import get_mixup_service  # Temporarily disabled due to NumPy compatibility

# Import new route modules
from routes.upload import router as upload_router
from routes.preprocess import router as preprocess_router
from routes.augment import router as augment_router
from routes.eda import router as eda_router
from routes.model import router as model_router

# Import shared utilities
from shared.supabase_utils import init_supabase_manager
from shared.module_runner import init_module_runner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Garuda ML Pipeline API",
    description="Complete ML Pipeline API with upload, preprocessing, augmentation, EDA, and model training",
    version="2.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload_router)
app.include_router(datasets_router)
app.include_router(data_retrieval_router)
app.include_router(download_router)
app.include_router(preprocess_router)
app.include_router(augment_router)
app.include_router(eda_router)
app.include_router(model_router)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        logger.info("Starting Garuda ML Pipeline API...")
        
        # Initialize Supabase manager
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_service_key:
            logger.warning("Supabase credentials not found in environment variables")
        else:
            init_supabase_manager(supabase_url, supabase_service_key)
            logger.info("Supabase manager initialized")
        
        # Initialize module runner
        workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        init_module_runner(workspace_root)
        logger.info("Module runner initialized")
        
        # Initialize mixup service (this will load the models)
        # get_mixup_service()  # Temporarily disabled due to NumPy compatibility
        # logger.info("Mixup service initialized")
        
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Garuda ML Pipeline API",
        "status": "running",
        "version": "2.0.0",
        "endpoints": {
            "upload": "/api/upload/dataset",
            "preprocess": "/api/preprocess/*",
            "augment": "/api/augment/*",
            "eda": "/api/eda/*",
            "models": {
                "logistic_regression": "/api/model/logistic-regression",
                "random_forest": "/api/model/random-forest",
                "gradient_boosting": "/api/model/gradient-boosting",
                "xgboost": "/api/model/xgboost",
                "distilbert": "/api/model/distilbert-finetune"
            },
            "download": "/download/{bucket_name}/{file_path:path}",
            "health": "/health"
        },
        "pipeline_stages": ["upload", "preprocess", "augment", "eda", "model"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Try to get the service to ensure models are loaded
        # service = get_mixup_service()  # Temporarily disabled due to NumPy compatibility
        return {
            "status": "healthy",
            "service": "mixup_service",
            "device": "cpu",  # Temporarily hardcoded
            "models_loaded": False  # Temporarily disabled
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

@app.post("/augment/mixup", response_model=MixupResponse)
async def mixup_augmentation(request: MixupRequest):
    """
    Perform mixup augmentation on text data
    
    Args:
        request: MixupRequest containing texts and parameters
        
    Returns:
        MixupResponse with augmented samples
    """
    try:
        logger.info(f"Processing mixup request with {len(request.texts)} texts")
        
        # Validate input
        if not request.texts or len(request.texts) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least 2 texts are required for mixup augmentation"
            )
        
        # Get the mixup service
        # service = get_mixup_service()  # Temporarily disabled due to NumPy compatibility
        
        # Process the batch
        # augmented_samples = service.process_mixup_batch(
        #     texts=request.texts,
        #     alpha=request.alpha,
        #     mix_labels=request.mix_labels
        # )
        
        # Temporary response since service is disabled
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mixup service temporarily disabled due to NumPy compatibility issues"
        )
        
        # logger.info(f"Successfully processed {len(augmented_samples)} samples")
        
        # Return response
        return MixupResponse(
            success=True,
            message=f"Mixup augmentation completed successfully. Generated {len(augmented_samples)} augmented samples.",
            augmented_samples=augmented_samples,
            total_samples=len(augmented_samples)
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in mixup augmentation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    # Run the server
    logger.info("Starting Garuda Data Augmentation API server...")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
