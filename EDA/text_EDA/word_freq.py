import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

def read_csv_file():
    """
    Scans for CSV files in the text data folder and prompts the user to select one.
    Also identifies potential text columns in the selected file.
    
    Returns:
        tuple: (DataFrame, text_column, filename)
    """
    # Define the path to the text data folder
    text_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'text')
    
    # Check if the folder exists, if not, create it
    if not os.path.exists(text_data_path):
        os.makedirs(text_data_path)
        print(f"Created text data folder at: {text_data_path}")
        print("Please add your CSV files to this folder and run the script again.")
        return None, None, None
    
    # Get CSV files from the text folder
    csv_files = [f for f in os.listdir(text_data_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in the text data folder: {text_data_path}")
        return None, None, None

    print(f"CSV files found in: {text_data_path}")
    print("\nPlease select a CSV file for word frequency analysis:")
    for i, filename in enumerate(csv_files):
        print(f"{i + 1}: {filename}")

    while True:
        try:
            choice = int(input(f"Enter the number (1-{len(csv_files)}): ")) - 1
            if 0 <= choice < len(csv_files):
                selected_file = os.path.join(text_data_path, csv_files[choice])
                filename = csv_files[choice]
                break
            else:
                print(f"Invalid number. Please enter a number between 1 and {len(csv_files)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Load the selected CSV file
    try:
        df = pd.read_csv(selected_file)
        print(f"\nLoaded dataset: {filename} ({len(df)} rows x {len(df.columns)} columns)")
        
        # Identify potential text columns (string/object columns with reasonable text length)
        text_columns = []
        for col in df.select_dtypes(include=['object']).columns:
            # Check if column has string values with reasonable length
            avg_len = df[col].astype(str).apply(len).mean()
            if avg_len > 10:  # Assuming text should be at least 10 chars on average
                text_columns.append(col)
        
        if not text_columns:
            print("No suitable text columns found for word frequency analysis.")
            return None, None, None
        
        print("\nPotential text columns found:")
        for i, col in enumerate(text_columns):
            sample = df[col].iloc[0]
            preview = (sample[:50] + '...') if len(str(sample)) > 50 else sample
            print(f"{i + 1}: {col} - Example: \"{preview}\"")
        
        while True:
            try:
                col_choice = int(input(f"Select text column (1-{len(text_columns)}): ")) - 1
                if 0 <= col_choice < len(text_columns):
                    text_column = text_columns[col_choice]
                    return df, text_column, filename
                else:
                    print(f"Invalid number. Please enter a number between 1 and {len(text_columns)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
                
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None, None

def word_frequency_analysis(text_data, title="Word Frequency Analysis"):
    """
    Analyzes and visualizes word frequencies in text data
    
    Parameters:
    text_data (list or Series): Collection of text documents
    title (str): Title for the visualizations
    """
    # Combine all text if it's a list/series
    if isinstance(text_data, (list, pd.Series)):
        all_text = ' '.join([str(text) for text in text_data])
    else:
        all_text = str(text_data)
    
    # Tokenize and clean
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in all_text.split() 
             if word.isalpha() and word.lower() not in stop_words]
    
    # Count frequencies
    word_counts = Counter(words)
    most_common = word_counts.most_common(20)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    plt.bar([word for word, count in most_common], 
            [count for word, count in most_common])
    plt.title(f'Top 20 Words in {title}')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save bar chart
    bar_chart_file = os.path.join(output_dir, f"word_frequency_bar_{title.replace(' ', '_')}.png")
    plt.savefig(bar_chart_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create word cloud
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white', 
                         max_words=100).generate(all_text)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {title}')
    
    # Save word cloud
    wordcloud_file = os.path.join(output_dir, f"word_cloud_{title.replace(' ', '_')}.png")
    plt.savefig(wordcloud_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nBar chart saved to: {bar_chart_file}")
    print(f"Word cloud saved to: {wordcloud_file}")
    
    return word_counts

def main():
    """Main function to run word frequency analysis."""
    print("="*60)
    print("TEXT WORD FREQUENCY ANALYSIS")
    print("="*60)
    
    # Read CSV file
    df, text_column, filename = read_csv_file()
    if df is None:
        return
    
    # Run word frequency analysis
    title = f"{filename} - {text_column}"
    word_counts = word_frequency_analysis(df[text_column], title)
    
    # Print summary
    print("\nWord Frequency Analysis Summary:")
    print(f"Total unique words: {len(word_counts)}")
    
    print("\nTop 20 most frequent words:")
    for word, count in word_counts.most_common(20):
        print(f"{word}: {count}")

if __name__ == "__main__":
    main()