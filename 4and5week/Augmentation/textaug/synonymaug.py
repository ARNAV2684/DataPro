import os
import pandas as pd
import nltk
import random
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

def get_wordnet_pos(treebank_tag):
    """
    Convert Penn Treebank POS tags to WordNet POS tags
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # For other POS tags, return None to indicate no replacement
        return None

def get_synonym(word, pos=None):
    """
    Get a synonym for the given word with the specified part of speech
    """
    synsets = wordnet.synsets(word, pos=pos) if pos else wordnet.synsets(word)
    if not synsets:
        return word  # Return original word if no synonyms found
    
    # Get all lemmas (word forms) from all synsets
    all_lemmas = []
    for synset in synsets:
        all_lemmas.extend(synset.lemmas())
    
    # Get unique synonyms (excluding the original word)
    synonyms = set()
    for lemma in all_lemmas:
        synonym = lemma.name().replace('_', ' ')
        if synonym.lower() != word.lower():  # Exclude the original word
            synonyms.add(synonym)
    
    if not synonyms:
        return word  # Return original word if no synonyms found
    
    # Return a random synonym
    return random.choice(list(synonyms))

def augment_text(text, replacement_prob=0.3):
    """
    Replace some words in the text with their synonyms
    Only replace nouns, verbs, adjectives, and adverbs
    Skip stopwords and words with no synonyms
    
    Args:
        text (str): The input text to augment
        replacement_prob (float): Probability of replacing a word with its synonym (0-1)
    """
    if not isinstance(text, str):
        return text  # Return as is if not a string
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Get POS tags
    tagged_tokens = pos_tag(tokens)
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Replace words with synonyms based on POS
    augmented_tokens = []
    for word, tag in tagged_tokens:
        # Skip stopwords and punctuation
        if word.lower() in stop_words or not word.isalpha():
            augmented_tokens.append(word)
            continue
        
        # Get WordNet POS
        wordnet_pos = get_wordnet_pos(tag)
        
        # Only replace if it's a noun, verb, adjective, or adverb
        if wordnet_pos and random.random() < replacement_prob:
            synonym = get_synonym(word, wordnet_pos)
            augmented_tokens.append(synonym)
        else:
            augmented_tokens.append(word)
    
    # Join tokens back into text
    augmented_text = ' '.join(augmented_tokens)
    return augmented_text

def augment_csv_file(input_file, replacement_prob=0.3):
    """
    Augment text data in a CSV file and save to a new file
    
    Args:
        input_file (str): Path to the input CSV file
        replacement_prob (float): Probability of replacing a word with its synonym (0-1)
    """
    # Read CSV file
    print(f"Reading CSV file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Identify text columns (object dtype)
    text_columns = df.select_dtypes(include=['object']).columns
    print(f"Found {len(text_columns)} text column(s): {', '.join(text_columns)}")
    
    # For each text column, create an augmented version
    for col in text_columns:
        new_col_name = f"{col}_augmented"
        print(f"Augmenting column: {col} → {new_col_name}")
        df[new_col_name] = df[col].apply(lambda x: augment_text(x, replacement_prob))
    
    # Generate output filename
    file_path, file_extension = os.path.splitext(input_file)
    output_file = f"{file_path}_augmented{file_extension}"
    
    # Save to new CSV file
    df.to_csv(output_file, index=False)
    print(f"Saved augmented data to: {output_file}")
    
    return output_file

def main():
    """
    Main function to handle user input and process the CSV file
    """
    print("="*60)
    print("TEXT AUGMENTATION WITH SYNONYMS")
    print("="*60)
    
    # Get input file from user
    input_file = input("Enter the path to the CSV file: ")
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' does not exist.")
        return
    
    # Get replacement probability
    replacement_prob = 0.3  # Default value
    prob_input = input("Enter replacement probability (0.1-0.5, default 0.3): ")
    if prob_input:
        try:
            replacement_prob = float(prob_input)
            if replacement_prob < 0.1 or replacement_prob > 0.5:
                print("Invalid probability. Using default value 0.3.")
                replacement_prob = 0.3
        except ValueError:
            print("Invalid input. Using default value 0.3.")
    
    try:
        output_file = augment_csv_file(input_file, replacement_prob)
        print(f"Augmentation complete! Output saved to: {output_file}")
    except Exception as e:
        print(f"Error during augmentation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()