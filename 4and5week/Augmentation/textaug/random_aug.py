import os
import pandas as pd
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

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

def get_synonym(word):
    """
    Get a synonym for the given word using WordNet
    
    Args:
        word (str): The word to find a synonym for
        
    Returns:
        str: A synonym or the original word if no synonym found
    """
    synonyms = []
    
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word:
                synonyms.append(synonym)
    
    if synonyms:
        return random.choice(synonyms)
    else:
        return word

def random_insertion(text):
    """
    Insert a random contextually relevant word into the text
    
    Args:
        text (str): The original text
        
    Returns:
        str: Text with a random word inserted
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text
        
    # Tokenize
    tokens = word_tokenize(text)
    
    if len(tokens) <= 1:
        return text
        
    # Get a random position
    insert_pos = random.randint(0, len(tokens))
    
    # Select a random word that's not a stopword or punctuation
    content_words = [token for token in tokens if token.isalpha() and 
                    token.lower() not in stopwords.words('english')]
    
    if not content_words:
        return text
        
    # Get a synonym of a random content word
    random_word = get_synonym(random.choice(content_words))
    
    # Insert the synonym
    tokens.insert(insert_pos, random_word)
    
    # Reconstruct the text
    augmented_text = ' '.join(tokens)
    
    # Clean up spaces before punctuation
    for punct in ['.', ',', '!', '?', ':', ';']:
        augmented_text = augmented_text.replace(f' {punct}', punct)
    
    return augmented_text

def random_deletion(text):
    """
    Delete a random word from the text
    
    Args:
        text (str): The original text
        
    Returns:
        str: Text with a random word deleted
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text
        
    # Tokenize
    tokens = word_tokenize(text)
    
    if len(tokens) <= 1:
        return text
        
    # Try to avoid deleting stopwords if possible
    stop_words = set(stopwords.words('english'))
    content_tokens = [i for i, token in enumerate(tokens) 
                     if token.isalpha() and token.lower() not in stop_words]
    
    # If no content tokens, delete any random token
    if not content_tokens:
        delete_idx = random.randint(0, len(tokens) - 1)
    else:
        # Delete a random content token
        delete_idx = random.choice(content_tokens)
    
    # Delete the word
    del tokens[delete_idx]
    
    # Reconstruct the text
    augmented_text = ' '.join(tokens)
    
    # Clean up spaces before punctuation
    for punct in ['.', ',', '!', '?', ':', ';']:
        augmented_text = augmented_text.replace(f' {punct}', punct)
    
    return augmented_text

def random_swapping(text):
    """
    Swap two random words in the text
    
    Args:
        text (str): The original text
        
    Returns:
        str: Text with two random words swapped
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text
        
    # Tokenize
    tokens = word_tokenize(text)
    
    if len(tokens) <= 1:
        return text
        
    # Get indices of words (not punctuation)
    word_indices = [i for i, token in enumerate(tokens) if token.isalpha()]
    
    if len(word_indices) <= 1:
        return text
        
    # Get two random positions
    pos1, pos2 = random.sample(word_indices, 2)
    
    # Swap the words
    tokens[pos1], tokens[pos2] = tokens[pos2], tokens[pos1]
    
    # Reconstruct the text
    augmented_text = ' '.join(tokens)
    
    # Clean up spaces before punctuation
    for punct in ['.', ',', '!', '?', ':', ';']:
        augmented_text = augmented_text.replace(f' {punct}', punct)
    
    return augmented_text

def augment_csv_file(input_file):
    """
    Perform random augmentation on text columns in a CSV file
    
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
    
    # For each text column, create augmented versions
    for col in text_columns:
        # Create new column names
        insertion_col = f"{col}_random_insertion"
        deletion_col = f"{col}_random_deletion"
        swapping_col = f"{col}_random_swapping"
        
        print(f"Augmenting column: {col}")
        
        # Apply augmentations
        total_rows = len(df)
        
        print("  Applying random insertion...")
        df[insertion_col] = [random_insertion(text) for text in df[col]]
        
        print("  Applying random deletion...")
        df[deletion_col] = [random_deletion(text) for text in df[col]]
        
        print("  Applying random swapping...")
        df[swapping_col] = [random_swapping(text) for text in df[col]]
    
    # Generate output filename
    output_dir = os.path.join(os.getcwd(), 'data')
    output_file = os.path.join(output_dir, 'text_aug_random.csv')
    
    # Save to new CSV file
    df.to_csv(output_file, index=False)
    print(f"Saved augmented data to: {output_file}")
    
    return output_file

def main():
    """
    Main function to run random text augmentation
    """
    print("="*60)
    print("TEXT AUGMENTATION WITH RANDOM OPERATIONS")
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