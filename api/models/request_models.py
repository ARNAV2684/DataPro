from pydantic import BaseModel
from typing import List, Optional

class MixupRequest(BaseModel):
    """Request model for mixup text augmentation"""
    texts: List[str]
    alpha: Optional[float] = 0.2
    mix_labels: Optional[bool] = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "This is a sample text for augmentation.",
                    "Another example text to demonstrate mixup.",
                    "Machine learning is fascinating."
                ],
                "alpha": 0.2,
                "mix_labels": True
            }
        }

class MixupResponse(BaseModel):
    """Response model for mixup text augmentation"""
    success: bool
    message: str
    augmented_samples: List[dict]
    total_samples: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Mixup augmentation completed successfully",
                "augmented_samples": [
                    {
                        "id": "mixup_0",
                        "original": "This is a sample text.",
                        "augmented": "This is a sample text.",
                        "alpha_used": 0.2
                    }
                ],
                "total_samples": 3
            }
        }
