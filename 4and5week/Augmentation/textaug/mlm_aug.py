import os
import pandas as pd
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize BERT model and tokenizer
print("Loading BERT model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded. Using device: {device}")

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

def augment_text_with_mlm(text):
    """
    Augment text using BERT masked language modeling
    
    Args:
        text (str): Input text to augment
        
    Returns:
        str: Augmented text with one word replaced
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Filter out stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    valid_tokens = [token for token in tokens 
                   if token.isalpha() and token.lower() not in stop_words]
    
    # If no valid tokens, return original text
    if not valid_tokens:
        return text
    
    # Select a random word to mask
    word_to_mask = random.choice(valid_tokens)
    
    # Create masked text
    masked_text = text.replace(word_to_mask, tokenizer.mask_token, 1)
    
    # Check if tokenized input would exceed BERT's max length
    # If so, truncate around the masked token
    encoded = tokenizer.encode(masked_text)
    if len(encoded) > 512:
        # Find position of mask token
        try:
            mask_pos = encoded.index(tokenizer.mask_token_id)
            
            # Calculate start and end to keep mask token in the middle when possible
            start = max(0, mask_pos - 250)
            end = min(len(encoded), mask_pos + 250)
            
            # Make sure we don't exceed 512 tokens
            if end - start > 512:
                end = start + 512
            
            # Decode and re-encode the truncated text
            truncated_ids = encoded[start:end]
            truncated_text = tokenizer.decode(truncated_ids)
            
            # If we lost the mask token in truncation, return original text
            if tokenizer.mask_token not in truncated_text:
                return text
            
            masked_text = truncated_text
        except ValueError:
            # If mask token not found for some reason, truncate from beginning
            masked_text = tokenizer.decode(encoded[:512])
    
    # Tokenize for BERT
    inputs = tokenizer(masked_text, return_tensors="pt").to(device)
    
    # Get mask token index
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    
    # If mask token not found (should not happen), return original text
    if len(mask_token_index) == 0:
        return text
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predictions at the masked position
    logits = outputs.logits
    mask_token_logits = logits[0, mask_token_index, :]
    
    # Get top 5 predictions
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    
    # Convert tokens to words
    predicted_tokens = [tokenizer.decode([token]) for token in top_5_tokens]
    
    # Filter out subwords, special tokens, and the original word
    predicted_words = [word.strip() for word in predicted_tokens 
                      if word.strip() and word.strip().isalpha() 
                      and word.lower() != word_to_mask.lower()]
    
    # If no suitable predictions, return original text
    if not predicted_words:
        return text
    
    # Choose a random prediction
    replacement = random.choice(predicted_words)
    
    # Replace the masked token with the prediction
    augmented_text = masked_text.replace(tokenizer.mask_token, replacement)
    
    # If we truncated the text, we need to replace only the masked token in the original text
    if augmented_text != masked_text.replace(tokenizer.mask_token, replacement):
        # We truncated, so just replace the original masked word in the original text
        return text.replace(word_to_mask, replacement, 1)
    
    return augmented_text

def augment_csv_file(input_file):
    """
    Perform MLM augmentation on text columns in a CSV file
    
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
        new_col_name = f"{col}_mlm_augmented"
        print(f"Augmenting column: {col} → {new_col_name}")
        
        # Apply MLM augmentation to each row
        augmented_texts = []
        total_rows = len(df)
        
        for i, text in enumerate(df[col]):
            if (i + 1) % 10 == 0 or (i + 1) == total_rows:
                print(f"  Progress: {i + 1}/{total_rows} rows processed", end="\r")
            augmented_texts.append(augment_text_with_mlm(text))
        
        print()  # New line after progress display
        df[new_col_name] = augmented_texts
    
    # Generate output filename
    output_dir = os.path.join(os.getcwd(), 'data')
    output_file = os.path.join(output_dir, 'text_aug_mlm.csv')
    
    # Save to new CSV file
    df.to_csv(output_file, index=False)
    print(f"Saved augmented data to: {output_file}")
    
    return output_file

def main():
    """
    Main function to run MLM-based text augmentation
    """
    print("="*60)
    print("TEXT AUGMENTATION WITH MASKED LANGUAGE MODELING")
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