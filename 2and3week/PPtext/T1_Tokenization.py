"""
Advanced Tokenization and Text Preprocessing Script
This script applies various tokenization methods on text data using the latest modules and tools.
"""

import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import warnings
import argparse
warnings.filterwarnings('ignore')

# Standard libraries for tokenization
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag

# Advanced tokenization with spaCy
import spacy

# Transformers for modern tokenization
from transformers import AutoTokenizer, BertTokenizer, GPT2Tokenizer

# Scikit-learn for additional text processing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

class TextTokenizer:
    """
    Advanced text tokenization class with multiple preprocessing methods
    """
    
    def __init__(self):
        # Download required NLTK data
        self.download_nltk_requirements()
        
        # Initialize tokenizers
        self.porter_stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.tweet_tokenizer = TweetTokenizer()
        
        # Load spaCy model (try different models)
        self.nlp = self.load_spacy_model()
        
        # Initialize transformer tokenizers
        self.bert_tokenizer = None
        self.gpt2_tokenizer = None
        self.load_transformer_tokenizers()
        
        # Load stopwords
        self.stop_words = set(stopwords.words('english'))
    
    def download_nltk_requirements(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        try:
            nltk.data.find('averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
    
    def load_spacy_model(self):
        """Load spaCy model with fallback options"""
        models_to_try = ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg']
        
        for model in models_to_try:
            try:
                return spacy.load(model)
            except OSError:
                continue
        
        print("Warning: No spaCy model found. Please install with: python -m spacy download en_core_web_sm")
        return None
    
    def load_transformer_tokenizers(self):
        """Load transformer tokenizers"""
        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            print("BERT tokenizer loaded successfully")
        except Exception as e:
            print(f"Could not load BERT tokenizer: {e}")
        
        try:
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            print("GPT-2 tokenizer loaded successfully")
        except Exception as e:
            print(f"Could not load GPT-2 tokenizer: {e}")
    
    def basic_preprocessing(self, text):
        """Basic text preprocessing"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', '', text)
        
        return text.strip()
    
    def advanced_preprocessing(self, text):
        """Advanced text preprocessing"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers (optional)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text.strip()
    
    def nltk_word_tokenize(self, text):
        """NLTK word tokenization"""
        if pd.isna(text):
            return []
        return word_tokenize(text)
    
    def nltk_sentence_tokenize(self, text):
        """NLTK sentence tokenization"""
        if pd.isna(text):
            return []
        return sent_tokenize(text)
    
    def tweet_tokenize(self, text):
        """Twitter-aware tokenization"""
        if pd.isna(text):
            return []
        return self.tweet_tokenizer.tokenize(text)
    
    def spacy_tokenize(self, text):
        """spaCy tokenization with additional features"""
        if pd.isna(text) or self.nlp is None:
            return []
        
        doc = self.nlp(text)
        tokens = []
        
        for token in doc:
            if not token.is_space:
                tokens.append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'pos': token.pos_,
                    'tag': token.tag_,
                    'is_alpha': token.is_alpha,
                    'is_stop': token.is_stop,
                    'is_punct': token.is_punct
                })
        
        return tokens
    
    def bert_tokenize(self, text):
        """BERT tokenization"""
        if pd.isna(text) or self.bert_tokenizer is None:
            return []
        
        # Tokenize and return tokens
        tokens = self.bert_tokenizer.tokenize(text)
        return tokens
    
    def gpt2_tokenize(self, text):
        """GPT-2 tokenization"""
        if pd.isna(text) or self.gpt2_tokenizer is None:
            return []
        
        # Tokenize and return tokens
        tokens = self.gpt2_tokenizer.tokenize(text)
        return tokens
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from token list"""
        if isinstance(tokens, list):
            return [token for token in tokens if token.lower() not in self.stop_words]
        return tokens
    
    def apply_stemming(self, tokens):
        """Apply Porter stemming"""
        if isinstance(tokens, list):
            return [self.porter_stemmer.stem(token) for token in tokens]
        return tokens
    
    def apply_lemmatization(self, tokens):
        """Apply lemmatization"""
        if isinstance(tokens, list):
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        return tokens
    
    def get_pos_tags(self, tokens):
        """Get part-of-speech tags"""
        if isinstance(tokens, list):
            return pos_tag(tokens)
        return tokens

def load_and_explore_data(file_path, text_column=None):
    """Load and explore the dataset"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Only show label distribution if Label column exists
    if 'Label' in df.columns:
        print(f"\nLabel distribution:")
        print(df['Label'].value_counts())
    
    # Auto-detect text column if not provided
    if text_column is None:
        text_columns = ['text', 'Text', 'content', 'Content', 'message', 'Message', 'description', 'Description']
        text_column = None
        for col in text_columns:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            # Use the first string column
            string_columns = df.select_dtypes(include=['object']).columns
            if len(string_columns) > 0:
                text_column = string_columns[0]
    
    if text_column and text_column in df.columns:
        print(f"\nText length statistics for column '{text_column}':")
        df['text_length'] = df[text_column].astype(str).str.len()
        print(df['text_length'].describe())
    
    return df

def apply_tokenization_methods(df, tokenizer, text_column=None):
    """Apply various tokenization methods to the dataset"""
    print("\nApplying tokenization methods...")
    
    # Auto-detect text column if not provided
    if text_column is None:
        # Look for common text column names
        text_columns = ['text', 'Text', 'content', 'Content', 'message', 'Message', 'description', 'Description']
        for col in text_columns:
            if col in df.columns:
                text_column = col
                break
        
        # If still not found, use the first string column
        if text_column is None:
            string_columns = df.select_dtypes(include=['object']).columns
            if len(string_columns) > 0:
                text_column = string_columns[0]
            else:
                raise ValueError("No text column found. Please specify --text_column argument.")
    
    print(f"Using text column: '{text_column}'")
    
    # Verify the column exists
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataset. Available columns: {list(df.columns)}")
    
    # Create a sample for demonstration (first 100 rows for speed)
    sample_df = df.head(100).copy()
    
    # Basic preprocessing
    print("1. Basic preprocessing...")
    sample_df['text_basic_cleaned'] = sample_df[text_column].apply(tokenizer.basic_preprocessing)
    
    # Advanced preprocessing
    print("2. Advanced preprocessing...")
    sample_df['text_advanced_cleaned'] = sample_df[text_column].apply(tokenizer.advanced_preprocessing)
    
    # NLTK word tokenization
    print("3. NLTK word tokenization...")
    sample_df['nltk_word_tokens'] = sample_df['text_basic_cleaned'].apply(tokenizer.nltk_word_tokenize)
    
    # NLTK sentence tokenization
    print("4. NLTK sentence tokenization...")
    sample_df['nltk_sent_tokens'] = sample_df[text_column].apply(tokenizer.nltk_sentence_tokenize)
    
    # Tweet tokenization
    print("5. Tweet tokenization...")
    sample_df['tweet_tokens'] = sample_df['text_basic_cleaned'].apply(tokenizer.tweet_tokenize)
    
    # spaCy tokenization
    print("6. spaCy tokenization...")
    sample_df['spacy_tokens'] = sample_df['text_basic_cleaned'].apply(tokenizer.spacy_tokenize)
    
    # BERT tokenization
    print("7. BERT tokenization...")
    sample_df['bert_tokens'] = sample_df['text_basic_cleaned'].apply(tokenizer.bert_tokenize)
    
    # GPT-2 tokenization
    print("8. GPT-2 tokenization...")
    sample_df['gpt2_tokens'] = sample_df['text_basic_cleaned'].apply(tokenizer.gpt2_tokenize)
    
    # Apply additional processing
    print("9. Applying stopword removal, stemming, and lemmatization...")
    sample_df['tokens_no_stopwords'] = sample_df['nltk_word_tokens'].apply(tokenizer.remove_stopwords)
    sample_df['tokens_stemmed'] = sample_df['tokens_no_stopwords'].apply(tokenizer.apply_stemming)
    sample_df['tokens_lemmatized'] = sample_df['tokens_no_stopwords'].apply(tokenizer.apply_lemmatization)
    sample_df['pos_tags'] = sample_df['tokens_no_stopwords'].apply(tokenizer.get_pos_tags)
    
    return sample_df

def create_visualizations(df, tokenized_df):
    """Create visualizations for tokenization analysis"""
    print("\nCreating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Text length distribution
    axes[0, 0].hist(df['text_length'], bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Distribution of Text Lengths')
    axes[0, 0].set_xlabel('Text Length (characters)')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Label distribution
    df['Label'].value_counts().plot(kind='bar', ax=axes[0, 1], color=['lightcoral', 'lightgreen'])
    axes[0, 1].set_title('Label Distribution')
    axes[0, 1].set_xlabel('Label')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=0)
    
    # 3. Token count distribution (NLTK)
    token_counts = tokenized_df['nltk_word_tokens'].apply(len)
    axes[1, 0].hist(token_counts, bins=30, alpha=0.7, color='lightgreen')
    axes[1, 0].set_title('Distribution of Token Counts (NLTK)')
    axes[1, 0].set_xlabel('Number of Tokens')
    axes[1, 0].set_ylabel('Frequency')
    
    # 4. Comparison of tokenization methods
    methods = ['NLTK', 'Tweet', 'BERT', 'GPT-2']
    avg_tokens = [
        tokenized_df['nltk_word_tokens'].apply(len).mean(),
        tokenized_df['tweet_tokens'].apply(len).mean(),
        tokenized_df['bert_tokens'].apply(len).mean(),
        tokenized_df['gpt2_tokens'].apply(len).mean()
    ]
    
    axes[1, 1].bar(methods, avg_tokens, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    axes[1, 1].set_title('Average Token Count by Method')
    axes[1, 1].set_xlabel('Tokenization Method')
    axes[1, 1].set_ylabel('Average Token Count')
    
    plt.tight_layout()
    plt.savefig('tokenization_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create word cloud
    try:
        all_text = ' '.join(tokenized_df['text_advanced_cleaned'].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Processed Text')
        plt.savefig('wordcloud.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Could not create word cloud: {e}")

def analyze_tokenization_results(tokenized_df, text_column):
    """Analyze and compare tokenization results"""
    print("\n" + "="*50)
    print("TOKENIZATION ANALYSIS RESULTS")
    print("="*50)
    
    # Sample text for comparison
    sample_text = tokenized_df[text_column].iloc[0][:200] + "..."
    print(f"\nSample text: {sample_text}")
    
    # Compare different tokenization methods
    print(f"\nTokenization Method Comparison:")
    print(f"Original text length: {len(tokenized_df[text_column].iloc[0])}")
    print(f"NLTK tokens: {len(tokenized_df['nltk_word_tokens'].iloc[0])}")
    print(f"Tweet tokens: {len(tokenized_df['tweet_tokens'].iloc[0])}")
    
    if not tokenized_df['bert_tokens'].iloc[0]:
        print("BERT tokens: Not available")
    else:
        print(f"BERT tokens: {len(tokenized_df['bert_tokens'].iloc[0])}")
    
    if not tokenized_df['gpt2_tokens'].iloc[0]:
        print("GPT-2 tokens: Not available")
    else:
        print(f"GPT-2 tokens: {len(tokenized_df['gpt2_tokens'].iloc[0])}")
    
    # Show effect of preprocessing
    original_tokens = len(tokenized_df['nltk_word_tokens'].iloc[0])
    no_stopwords = len(tokenized_df['tokens_no_stopwords'].iloc[0])
    stemmed = len(tokenized_df['tokens_stemmed'].iloc[0])
    lemmatized = len(tokenized_df['tokens_lemmatized'].iloc[0])
    
    print(f"\nPreprocessing Effects:")
    print(f"Original tokens: {original_tokens}")
    print(f"After stopword removal: {no_stopwords} ({((no_stopwords/original_tokens)*100):.1f}%)")
    print(f"After stemming: {stemmed}")
    print(f"After lemmatization: {lemmatized}")
    
    # Show sample tokens
    print(f"\nSample tokens (first 10):")
    print(f"Original: {tokenized_df['nltk_word_tokens'].iloc[0][:10]}")
    print(f"No stopwords: {tokenized_df['tokens_no_stopwords'].iloc[0][:10]}")
    print(f"Stemmed: {tokenized_df['tokens_stemmed'].iloc[0][:10]}")
    print(f"Lemmatized: {tokenized_df['tokens_lemmatized'].iloc[0][:10]}")
    
    # Most common words
    all_tokens = []
    for tokens in tokenized_df['tokens_no_stopwords']:
        all_tokens.extend(tokens)
    
    most_common = Counter(all_tokens).most_common(10)
    print(f"\nMost common words (after preprocessing):")
    for word, count in most_common:
        print(f"  {word}: {count}")

def save_results(tokenized_df):
    """Save tokenization results to files"""
    print("\nSaving results...")
    
    # Save the tokenized dataset
    tokenized_df.to_csv('tokenized_data.csv', index=False)
    print("Tokenized data saved to 'tokenized_data.csv'")
    
    # Save token statistics
    stats = {
        'method': ['NLTK', 'Tweet', 'BERT', 'GPT-2', 'No Stopwords', 'Stemmed', 'Lemmatized'],
        'avg_tokens': [
            tokenized_df['nltk_word_tokens'].apply(len).mean(),
            tokenized_df['tweet_tokens'].apply(len).mean(),
            tokenized_df['bert_tokens'].apply(lambda x: len(x) if x else 0).mean(),
            tokenized_df['gpt2_tokens'].apply(lambda x: len(x) if x else 0).mean(),
            tokenized_df['tokens_no_stopwords'].apply(len).mean(),
            tokenized_df['tokens_stemmed'].apply(len).mean(),
            tokenized_df['tokens_lemmatized'].apply(len).mean()
        ]
    }
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv('tokenization_stats.csv', index=False)
    print("Statistics saved to 'tokenization_stats.csv'")

def main():
    """Main function to run the tokenization pipeline"""
    # Add argument parser for API compatibility
    parser = argparse.ArgumentParser(description="Advanced Text Tokenization and Preprocessing")
    parser.add_argument("--input", required=True, help="Path to input CSV or Excel file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--text_column", default=None, help="Name of the text column to process")
    
    args = parser.parse_args()
    
    print("="*60)
    print("ADVANCED TEXT TOKENIZATION AND PREPROCESSING")
    print("="*60)
    
    try:
        # Load data with Excel support
        if args.input.lower().endswith('.csv'):
            df = pd.read_csv(args.input)
        elif args.input.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(args.input)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel files.")
        
        print(f"Successfully loaded data from: {args.input}")
        print(f"Dataset shape: {df.shape}")
        
        # Initialize tokenizer
        tokenizer = TextTokenizer()
        
        # Apply tokenization methods
        tokenized_df = apply_tokenization_methods(df, tokenizer, args.text_column)
        
        # Analyze results
        analyze_tokenization_results(tokenized_df, args.text_column)
        
        # Save results to specified output path
        tokenized_df.to_csv(args.output, index=False)
        print(f"Tokenized data saved to: {args.output}")
        
        print("\n" + "="*60)
        print("TOKENIZATION COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the required packages are installed:")
        print("pip install pandas numpy nltk spacy transformers scikit-learn matplotlib seaborn wordcloud")
        raise  # Re-raise the exception for proper error handling

if __name__ == "__main__":
    main()