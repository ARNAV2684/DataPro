import random
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MixupService:
    """Service class for text mixup augmentation"""
    
    def __init__(self):
        """Initialize the mixup service with required models"""
        try:
            logger.info("Loading sentence embedding model...")
            self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            
            logger.info("Loading language model for decoding...")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            logger.info(f"Models loaded successfully. Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def approximate_text_from_embedding(self, embedding: np.ndarray, texts: List[str], top_k: int = 5) -> str:
        """
        Try to approximate text from a mixed embedding using original texts
        
        Args:
            embedding: The mixed embedding
            texts: List of original texts used for reference
            top_k: Number of nearest neighbors to consider
            
        Returns:
            Approximated text
        """
        try:
            # Encode all original texts
            original_embeddings = self.sentence_model.encode(texts)
            
            # Find nearest neighbors
            knn = NearestNeighbors(n_neighbors=min(top_k, len(texts)), metric='cosine')
            knn.fit(original_embeddings)
            
            # Get the closest text to our mixed embedding
            distances, indices = knn.kneighbors([embedding], n_neighbors=min(top_k, len(texts)))
            
            # Get the closest match
            closest_idx = indices[0][0]
            closest_text = texts[closest_idx]
            
            return closest_text
            
        except Exception as e:
            logger.error(f"Error in text approximation: {str(e)}")
            return texts[0] if texts else "Error in approximation"
    
    def mixup_text_augmentation(self, text1: str, text2: str, alpha: float = 0.5) -> str:
        """
        Create a mixed-up version of two texts by interpolating their embeddings
        
        Args:
            text1: First text
            text2: Second text
            alpha: Mixing ratio between 0 and 1
            
        Returns:
            Approximated text from mixed embeddings
        """
        try:
            if not isinstance(text1, str) or not isinstance(text2, str):
                return text1  # Return original if not strings
            
            if len(text1.strip()) == 0 or len(text2.strip()) == 0:
                return text1  # Return original if empty
            
            # Encode both texts
            embedding1 = self.sentence_model.encode(text1)
            embedding2 = self.sentence_model.encode(text2)
            
            # Mix the embeddings
            mixed_embedding = alpha * embedding1 + (1 - alpha) * embedding2
            
            # Try to approximate text from the mixed embedding
            approx_text = self.approximate_text_from_embedding(
                mixed_embedding, 
                [text1, text2]
            )
            
            return approx_text
            
        except Exception as e:
            logger.error(f"Error in mixup augmentation: {str(e)}")
            return text1  # Return original on error
    
    def process_mixup_batch(self, texts: List[str], alpha: float = 0.2, mix_labels: bool = True) -> List[Dict[str, Any]]:
        """
        Process a batch of texts for mixup augmentation
        
        Args:
            texts: List of input texts
            alpha: Mixing ratio
            mix_labels: Whether to mix labels (placeholder for future use)
            
        Returns:
            List of augmented samples with metadata
        """
        try:
            if not texts or len(texts) < 2:
                raise ValueError("At least 2 texts are required for mixup augmentation")
            
            augmented_samples = []
            
            for i, text in enumerate(texts):
                # Randomly select another text for mixing
                other_indices = [j for j in range(len(texts)) if j != i]
                other_idx = random.choice(other_indices)
                other_text = texts[other_idx]
                
                # Random mixing ratio around the specified alpha
                actual_alpha = max(0.1, min(0.9, alpha + random.uniform(-0.1, 0.1)))
                
                # Create mixed-up text
                mixed_text = self.mixup_text_augmentation(text, other_text, actual_alpha)
                
                # Create sample record
                sample = {
                    "id": f"mixup_{i}",
                    "original": text,
                    "augmented": mixed_text,
                    "alpha_used": round(actual_alpha, 3),
                    "mixed_with_index": other_idx
                }
                
                augmented_samples.append(sample)
            
            return augmented_samples
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise

# Global instance
mixup_service = None

def get_mixup_service() -> MixupService:
    """Get or create mixup service instance"""
    global mixup_service
    if mixup_service is None:
        mixup_service = MixupService()
    return mixup_service
