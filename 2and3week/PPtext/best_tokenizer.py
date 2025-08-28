import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
warnings.filterwarnings('ignore')

# Try importing transformers (optional)
try:
    from transformers import AutoTokenizer
    transformers_available = True
except ImportError:
    transformers_available = False

# Try importing spacy (optional)
try:
    import spacy
    spacy_available = True
except ImportError:
    spacy_available = False

def get_recommendations():
    """Read the recommendations.txt file and extract recommendations"""
    try:
        with open('recommendations.txt', 'r') as f:
            content = f.read()
        
        recommendations = {
            'best_overall': None,
            'speed': None,
            'accuracy': None,
            'preprocessing': None
        }
        
        lines = content.split('\n')
        for line in lines:
            if line.startswith('Best Overall:'):
                recommendations['best_overall'] = line.split(':')[1].strip()
            elif 'Speed:' in line:
                recommendations['speed'] = line.split(':')[1].strip()
            elif 'Accuracy:' in line:
                recommendations['accuracy'] = line.split(':')[1].strip()
            elif 'Preprocessing:' in line:
                recommendations['preprocessing'] = line.split(':')[1].strip()
        
        return recommendations
    except Exception as e:
        print(f"Error reading recommendations file: {e}")
        return None

def display_recommendations_and_get_choice():
    """Display recommendations and get user's choice"""
    recommendations = get_recommendations()
    
    if not recommendations:
        print("Could not load recommendations. Available tokenizers:")
        available_tokenizers = ['NLTK', 'Tweet', 'BERT', 'GPT-2', 'spaCy']
    else:
        print("\n" + "="*60)
        print("🎯 TOKENIZER RECOMMENDATIONS FOR YOUR DATASET")
        print("="*60)
        
        if recommendations['best_overall']:
            print(f"🏆 RECOMMENDED (Best Overall): {recommendations['best_overall']}")
        
        print(f"\n📋 Other Options:")
        if recommendations['speed']:
            print(f"   🚀 For Speed: {recommendations['speed']}")
        if recommendations['accuracy']:
            print(f"   🎯 For Accuracy: {recommendations['accuracy']}")
        if recommendations['preprocessing']:
            print(f"   🔧 For Preprocessing: {recommendations['preprocessing']}")
    
    print(f"\n💡 Available Tokenizers:")
    tokenizer_options = {
        '1': 'NLTK',
        '2': 'Tweet',
        '3': 'spaCy',
        '4': 'BERT',
        '5': 'GPT-2'
    }
    
    # Check availability and display options
    available_options = {}
    for key, tokenizer in tokenizer_options.items():
        status = "✅"
        if tokenizer == 'BERT' or tokenizer == 'GPT-2':
            if not transformers_available:
                status = "❌ (transformers not installed)"
        elif tokenizer == 'spaCy':
            if not spacy_available:
                status = "❌ (spacy not installed)"
        
        available_options[key] = tokenizer
        print(f"   {key}. {tokenizer} {status}")
    
    # Get user choice
    while True:
        choice = input(f"\nSelect tokenizer (1-5): ").strip()
        if choice in available_options:
            selected_tokenizer = available_options[choice]
            
            # Check if it's available
            if selected_tokenizer == 'BERT' or selected_tokenizer == 'GPT-2':
                if not transformers_available:
                    print("❌ Transformers library not available. Please install with: pip install transformers")
                    continue
            elif selected_tokenizer == 'spaCy':
                if not spacy_available:
                    print("❌ spaCy not available. Please install with: pip install spacy && python -m spacy download en_core_web_sm")
                    continue
            
            # Show recommendation if user didn't choose the best one
            if recommendations and recommendations['best_overall']:
                if selected_tokenizer != recommendations['best_overall']:
                    print(f"\n⚠️  Note: You selected {selected_tokenizer}, but {recommendations['best_overall']} was recommended as the best overall choice for your dataset.")
                    confirm = input("Do you want to continue with your selection? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                else:
                    print(f"✅ Great choice! {selected_tokenizer} is the recommended tokenizer for your dataset.")
            
            return selected_tokenizer
        else:
            print("Invalid choice. Please select 1-5.")

def download_nltk_requirements():
    """Download required NLTK data"""
    required_packages = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    
    for package in required_packages:
        try:
            if 'punkt' in package:
                nltk.data.find(f'tokenizers/{package}')
            else:
                nltk.data.find(f'corpora/{package}')
        except LookupError:
            print(f"Downloading {package}...")
            nltk.download(package, quiet=True)

def basic_preprocessing(text):
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

def apply_selected_tokenizer(text, tokenizer_name):
    """Apply the selected tokenizer to the text"""
    text = basic_preprocessing(text)
    
    if tokenizer_name.upper() == 'NLTK':
        download_nltk_requirements()
        return word_tokenize(text)
    
    elif tokenizer_name.upper() == 'TWEET':
        tweet_tokenizer = TweetTokenizer()
        return tweet_tokenizer.tokenize(text)
    
    elif tokenizer_name.upper() == 'SPACY':
        try:
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(text)
            return [token.text for token in doc if not token.is_space]
        except OSError:
            print("spaCy model not found. Please install with: python -m spacy download en_core_web_sm")
            return text.split()
    
    elif tokenizer_name.upper() == 'BERT':
        try:
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            tokens = tokenizer.encode(text, max_length=512, truncation=True, add_special_tokens=True)
            return tokenizer.convert_ids_to_tokens(tokens)
        except Exception as e:
            print(f"BERT tokenization error: {e}")
            return text.split()
    
    elif tokenizer_name.upper() == 'GPT-2':
        try:
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            tokens = tokenizer.encode(text, max_length=1024, truncation=True)
            return tokenizer.convert_ids_to_tokens(tokens)
        except Exception as e:
            print(f"GPT-2 tokenization error: {e}")
            return text.split()
    
    else:
        # Fallback to simple whitespace tokenization
        return text.split()

def apply_postprocessing(tokens, tokenizer_name):
    """Apply all post-processing methods to tokens"""
    if not isinstance(tokens, list):
        tokens = str(tokens).split()
    
    processed_tokens = tokens.copy()
    
    # For BERT/GPT-2, filter out special tokens first
    if tokenizer_name.upper() in ['BERT', 'GPT-2']:
        special_tokens = ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '<|endoftext|>', 'Ġ']
        processed_tokens = [token for token in processed_tokens if not any(st in token for st in special_tokens)]
        # Clean up BERT/GPT-2 specific prefixes
        processed_tokens = [token.replace('##', '').replace('Ġ', '') for token in processed_tokens]
        processed_tokens = [token for token in processed_tokens if token.strip() and len(token) > 1]
    
    # Store original tokens before processing
    original_count = len(processed_tokens)
    
    # Apply stopword removal
    try:
        download_nltk_requirements()
        stop_words = set(stopwords.words('english'))
        processed_tokens = [token for token in processed_tokens if token.lower() not in stop_words]
    except Exception as e:
        print(f"Warning: Could not apply stopword removal: {e}")
    
    # Apply lemmatization
    try:
        download_nltk_requirements()
        lemmatizer = WordNetLemmatizer()
        processed_tokens = [lemmatizer.lemmatize(token.lower()) for token in processed_tokens]
    except Exception as e:
        print(f"Warning: Could not apply lemmatization: {e}")
    
    return processed_tokens, original_count

def apply_spacy_postprocessing(text):
    """Apply post-processing using spaCy's built-in capabilities"""
    try:
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        tokens = []
        original_count = 0
        
        for token in doc:
            if token.is_space:
                continue
            original_count += 1
            
            # Apply filters - remove stopwords
            if token.is_stop:
                continue
            
            # Apply lemmatization
            token_text = token.lemma_.lower()
            tokens.append(token_text)
        
        return tokens, original_count
    except Exception as e:
        print(f"Warning: spaCy post-processing failed: {e}")
        return text.split(), len(text.split())

def create_visualizations(df, selected_tokenizer):
    """Create visual representations of tokenizer statistics"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{selected_tokenizer} Tokenizer Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Token Count Distribution
    axes[0, 0].hist(df['token_count'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Token Count Distribution (After Processing)')
    axes[0, 0].set_xlabel('Number of Tokens')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df['token_count'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["token_count"].mean():.1f}')
    axes[0, 0].legend()
    
    # 2. Original vs Processed Token Count Comparison
    if 'original_token_count' in df.columns:
        axes[0, 1].scatter(df['original_token_count'], df['token_count'], alpha=0.6, color='green')
        axes[0, 1].plot([0, df['original_token_count'].max()], [0, df['original_token_count'].max()], 
                        'r--', label='y=x line')
        axes[0, 1].set_title('Original vs Processed Token Count')
        axes[0, 1].set_xlabel('Original Token Count')
        axes[0, 1].set_ylabel('Processed Token Count')
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'No comparison data\navailable', 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].set_title('Token Count Comparison')
    
    # 3. Stopwords Removed Distribution
    if 'stopwords_removed' in df.columns:
        axes[0, 2].hist(df['stopwords_removed'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_title('Stopwords Removed Distribution')
        axes[0, 2].set_xlabel('Number of Stopwords Removed')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(df['stopwords_removed'].mean(), color='red', linestyle='--', 
                           label=f'Mean: {df["stopwords_removed"].mean():.1f}')
        axes[0, 2].legend()
    else:
        axes[0, 2].text(0.5, 0.5, 'No stopword removal\ndata available', 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=axes[0, 2].transAxes, fontsize=12)
        axes[0, 2].set_title('Stopwords Removed')
    
    # 4. Most Common Tokens (Top 20)
    all_tokens = []
    for token_list in df['tokens']:
        if isinstance(token_list, list):
            all_tokens.extend(token_list)
        else:
            all_tokens.extend(str(token_list).split())
    
    token_freq = Counter(all_tokens)
    top_tokens = token_freq.most_common(20)
    
    if top_tokens:
        tokens, frequencies = zip(*top_tokens)
        y_pos = np.arange(len(tokens))
        axes[1, 0].barh(y_pos, frequencies, color='lightcoral')
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(tokens)
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_title('Top 20 Most Common Tokens')
        axes[1, 0].set_xlabel('Frequency')
    
    # 5. Text Length vs Token Count
    if 'Text' in df.columns:
        text_lengths = df['Text'].astype(str).str.len()
        axes[1, 1].scatter(text_lengths, df['token_count'], alpha=0.6, color='purple')
        axes[1, 1].set_title('Text Length vs Token Count')
        axes[1, 1].set_xlabel('Text Length (characters)')
        axes[1, 1].set_ylabel('Token Count')
        
        # Add correlation coefficient
        correlation = np.corrcoef(text_lengths, df['token_count'])[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=axes[1, 1].transAxes, fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:
        axes[1, 1].text(0.5, 0.5, 'No text column\navailable for analysis', 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Text Length vs Token Count')
    
    # 6. Processing Efficiency Stats
    stats_text = f"""
    TOKENIZER: {selected_tokenizer}
    
    📊 DATASET STATISTICS:
    • Total Documents: {len(df):,}
    • Avg Tokens/Doc: {df['token_count'].mean():.1f}
    • Min Tokens: {df['token_count'].min()}
    • Max Tokens: {df['token_count'].max()}
    • Std Dev: {df['token_count'].std():.1f}
    
    🔧 PROCESSING RESULTS:
    • Total Tokens: {df['token_count'].sum():,}
    """
    
    if 'stopwords_removed' in df.columns:
        stats_text += f"• Stopwords Removed: {df['stopwords_removed'].sum():,}\n"
        stats_text += f"• Avg Stopwords/Doc: {df['stopwords_removed'].mean():.1f}\n"
    
    stats_text += f"• Lemmatization: ✅ Applied\n"
    stats_text += f"• Case Normalization: ✅ Applied\n"
    
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Processing Statistics')
    
    plt.tight_layout()
    
    # Save the visualization
    viz_filename = f'{selected_tokenizer.lower()}_tokenizer_analysis.png'
    plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved as: {viz_filename}")
    
    # Show the plot
    plt.show()

def main():
    print("="*60)
    print("AUTOMATED TOKENIZER WITH FULL POST-PROCESSING")
    print("="*60)
    print("🔧 Auto-applying: Stopword removal, Lemmatization & Case normalization")
    print("📊 Generating comprehensive visual analysis")
    
    # Display recommendations and get user choice
    selected_tokenizer = display_recommendations_and_get_choice()
    
    print(f"\n🔄 Processing with {selected_tokenizer} tokenizer...")
    print("🔧 Auto-applying all post-processing methods...")
    
    # Read your data
    try:
        df = pd.read_csv('df_file.csv')
        print(f"📊 Dataset loaded: {len(df)} rows")
        
        # Check if 'Text' column exists
        if 'Text' not in df.columns:
            print(f"Available columns: {df.columns.tolist()}")
            text_column = input("Enter the name of the text column: ").strip()
        else:
            text_column = 'Text'
        
        # Apply the selected tokenizer with processing
        print(f"🔄 Applying {selected_tokenizer} tokenization...")
        
        # Initialize lists to store results
        tokens_list = []
        original_counts = []
        
        # Process each text
        for idx, text in enumerate(df[text_column]):
            if idx % 100 == 0:
                print(f"   Processing {idx}/{len(df)} documents...")
            
            # Special handling for spaCy
            if selected_tokenizer.upper() == 'SPACY':
                tokens, original_count = apply_spacy_postprocessing(str(text))
            else:
                # Standard tokenization + post-processing
                raw_tokens = apply_selected_tokenizer(str(text), selected_tokenizer)
                tokens, original_count = apply_postprocessing(raw_tokens, selected_tokenizer)
            
            tokens_list.append(tokens)
            original_counts.append(original_count)
        
        # Add results to dataframe
        df['tokens'] = tokens_list
        df['original_token_count'] = original_counts
        df['token_count'] = df['tokens'].apply(len)
        df['stopwords_removed'] = df['original_token_count'] - df['token_count']
        
        # Convert tokens to string format for CSV storage
        df['tokens_str'] = df['tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        
        # Save the results
        output_filename = f'final_tokenized_output_{selected_tokenizer.lower()}_full_processing.csv'
        df.to_csv(output_filename, index=False)
        
        # Display results
        print(f"\n✅ Tokenization complete!")
        print(f"📁 Results saved to: {output_filename}")
        print(f"📊 Total rows processed: {len(df)}")
        print(f"📈 Average tokens per text: {df['token_count'].mean():.2f}")
        print(f"📊 Token count range: {df['token_count'].min()} - {df['token_count'].max()}")
        print(f"🗑️ Total stopwords removed: {df['stopwords_removed'].sum():,}")
        print(f"🗑️ Average stopwords per document: {df['stopwords_removed'].mean():.2f}")
        
        # Show sample result
        print(f"\n📝 Sample tokenization result:")
        sample_text = df[text_column].iloc[0][:100] + "..."
        sample_tokens = df['tokens'].iloc[0][:10]
        print(f"Original: {sample_text}")
        print(f"Tokens: {sample_tokens}")
        print(f"Original token count: {df['original_token_count'].iloc[0]}")
        print(f"Processed token count: {df['token_count'].iloc[0]}")
        print(f"Stopwords removed: {df['stopwords_removed'].iloc[0]}")
        
        # Create comprehensive visualizations
        print(f"\n📊 Generating visual analysis...")
        create_visualizations(df, selected_tokenizer)
        
        print(f"\n🎉 Analysis complete! Check the generated visualization and CSV file.")
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        print("\nMake sure:")
        print("- df_file.csv exists in the current directory")
        print("- Required packages are installed: pandas, nltk, matplotlib, seaborn")
        print("- Install missing packages with: pip install pandas nltk matplotlib seaborn")

if __name__ == "__main__":
    main()