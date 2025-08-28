import os
import pandas as pd
import random
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Initialize models
print("Loading sentence embedding model...")
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("Loading language model for decoding...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Models loaded. Using device: {device}")

def find_csv_files():
    """
    Search for CSV files in the 'data' folder
    
    Returns:
        list: List of CSV file paths
    """
    data_dir = os.path.join(os.getcwd(), 'data')
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory at: {data_dir}")
        print("Please add your CSV files to this folder and run the script again.")
        return []
    
    # Get CSV files from the data folder
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                if f.endswith('.csv')]
    
    return csv_files

def select_csv_file(csv_files):
    """
    Prompt user to select a CSV file from a list
    
    Args:
        csv_files (list): List of CSV file paths
        
    Returns:
        str: Selected CSV file path
    """
    if not csv_files:
        print("No CSV files found in the data directory.")
        return None
    
    print("\nAvailable CSV files:")
    for i, file_path in enumerate(csv_files):
        print(f"{i+1}: {os.path.basename(file_path)}")
    
    while True:
        try:
            choice = int(input(f"Select a file (1-{len(csv_files)}): ")) - 1
            if 0 <= choice < len(csv_files):
                return csv_files[choice]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(csv_files)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def approximate_text_from_embedding(embedding, texts, top_k=5):
    """
    Try to approximate text from a mixed embedding using original texts
    
    Args:
        embedding (numpy.ndarray): The mixed embedding
        texts (list): List of original texts used for reference
        top_k (int): Number of nearest neighbors to consider
        
    Returns:
        str: Approximated text
    """
    # Encode all original texts
    original_embeddings = sentence_model.encode(texts)
    
    # Find nearest neighbors
    knn = NearestNeighbors(n_neighbors=min(top_k, len(texts)), metric='cosine')
    knn.fit(original_embeddings)
    
    # Get the closest text to our mixed embedding
    distances, indices = knn.kneighbors([embedding], n_neighbors=min(top_k, len(texts)))
    
    # Get the closest match
    closest_idx = indices[0][0]
    closest_text = texts[closest_idx]
    
    return closest_text

def generate_text_from_embedding(embedding, max_length=64):
    """
    Generate text that might represent the mixed embedding
    This is an experimental approach and may not produce optimal results
    
    Args:
        embedding (numpy.ndarray): The mixed embedding
        max_length (int): Maximum length of generated text
        
    Returns:
        str: Generated text approximation
    """
    try:
        # Start with a [CLS] token
        input_ids = torch.tensor([[tokenizer.cls_token_id]]).to(device)
        
        # Convert embedding to tensor and project to model dimensions if needed
        embedding_tensor = torch.tensor(embedding).to(device)
        
        # Generate text auto-regressively
        for _ in range(max_length):
            with torch.no_grad():
                # Get the model's output
                outputs = model(input_ids)
                logits = outputs.logits
                
                # Get the next token prediction
                next_token_logits = logits[:, -1, :]
                
                # Sample from the distribution
                next_token_id = torch.multinomial(
                    torch.softmax(next_token_logits, dim=-1), 
                    num_samples=1
                ).item()
                
                # Check if we hit the end token
                if next_token_id == tokenizer.sep_token_id:
                    break
                
                # Add the token to our sequence
                input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]]).to(device)], dim=1)
        
        # Decode the generated ids
        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text
    
    except Exception as e:
        # If generation fails, return a placeholder
        return f"[Mixed Text - Could not generate]"

def mixup_text_augmentation(text1, text2, alpha=0.5):
    """
    Create a mixed-up version of two texts by interpolating their embeddings
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        alpha (float): Mixing ratio between 0 and 1
        
    Returns:
        str: Approximated text from mixed embeddings
    """
    if not isinstance(text1, str) or not isinstance(text2, str):
        return text1  # Return original if not strings
    
    if len(text1.strip()) == 0 or len(text2.strip()) == 0:
        return text1  # Return original if empty
    
    # Encode both texts
    embedding1 = sentence_model.encode(text1)
    embedding2 = sentence_model.encode(text2)
    
    # Mix the embeddings
    mixed_embedding = alpha * embedding1 + (1 - alpha) * embedding2
    
    # Try to approximate text from the mixed embedding
    # We'll use a simple nearest neighbor approach for the demo
    approx_text = approximate_text_from_embedding(
        mixed_embedding, 
        [text1, text2]
    )
    
    return approx_text

def augment_csv_file(input_file):
    """
    Perform mixup augmentation on text columns in a CSV file
    
    Args:
        input_file (str): Path to input CSV file
        
    Returns:
        str: Path to output CSV file
    """
    print(f"Reading CSV file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Identify text columns (object dtype)
    text_columns = df.select_dtypes(include=['object']).columns
    print(f"Found {len(text_columns)} text column(s): {', '.join(text_columns)}")
    
    # For each text column, create an augmented version
    for col in text_columns:
        new_col_name = f"{col}_mixup_augmented"
        print(f"Augmenting column: {col} → {new_col_name}")
        
        # Apply mixup augmentation to each row
        augmented_texts = []
        total_rows = len(df)
        
        for i, text in enumerate(df[col]):
            if (i + 1) % 10 == 0 or (i + 1) == total_rows:
                print(f"  Progress: {i + 1}/{total_rows} rows processed", end="\r")
            
            # Randomly select another row for mixing
            other_idx = random.choice(range(len(df)))
            while other_idx == i:  # Ensure we don't select the same row
                other_idx = random.choice(range(len(df)))
            
            other_text = df.iloc[other_idx][col]
            
            # Random mixing ratio
            alpha = random.uniform(0.3, 0.7)
            
            # Create mixed-up text
            mixed_text = mixup_text_augmentation(text, other_text, alpha)
            augmented_texts.append(mixed_text)
        
        print()  # New line after progress display
        df[new_col_name] = augmented_texts
    
    # Generate output filename
    output_dir = os.path.join(os.getcwd(), 'data')
    output_file = os.path.join(output_dir, 'text_aug_mixup.csv')
    
    # Save to new CSV file
    df.to_csv(output_file, index=False)
    print(f"Saved augmented data to: {output_file}")
    
    return output_file

def main():
    """
    Main function to run mixup-based text augmentation
    """
    print("="*60)
    print("TEXT AUGMENTATION WITH EMBEDDING MIXUP")
    print("="*60)
    
    # Find CSV files
    csv_files = find_csv_files()
    if not csv_files:
        return
    
    # Let user select a file
    input_file = select_csv_file(csv_files)
    if not input_file:
        return
    
    try:
        # Perform augmentation
        output_file = augment_csv_file(input_file)
        print(f"Augmentation complete! Output saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during augmentation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()