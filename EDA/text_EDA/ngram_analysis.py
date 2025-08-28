import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import networkx as nx

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
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
    print("\nPlease select a CSV file for n-gram analysis:")
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
            print("No suitable text columns found for n-gram analysis.")
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

def ngram_analysis(text_data, n=2, top_n=20, title="N-gram Analysis"):
    """
    Analyzes and visualizes n-grams in text data
    
    Parameters:
    text_data (list or Series): Collection of text documents
    n (int): Size of n-grams (2 for bigrams, 3 for trigrams, etc.)
    top_n (int): Number of top n-grams to display
    title (str): Title for the visualizations
    """
    # Combine all text if it's a list/series
    if isinstance(text_data, (list, pd.Series)):
        all_text = ' '.join([str(text) for text in text_data])
    else:
        all_text = str(text_data)
    
    # Tokenize
    tokens = word_tokenize(all_text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    # Generate n-grams
    n_grams = list(ngrams(filtered_tokens, n))
    ngram_freq = Counter(n_grams)
    
    # Get top n-grams
    top_ngrams = ngram_freq.most_common(top_n)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Bar chart
    plt.figure(figsize=(12, 6))
    plt.bar([' '.join(gram) for gram, count in top_ngrams], 
            [count for gram, count in top_ngrams])
    plt.title(f'Top {top_n} {n}-grams in {title}')
    plt.xlabel(f'{n}-grams')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save bar chart
    bar_chart_file = os.path.join(output_dir, f"ngram_{n}_bar_{title.replace(' ', '_')}.png")
    plt.savefig(bar_chart_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Network visualization for bigrams
    if n == 2:
        # Create network graph
        plt.figure(figsize=(12, 12))
        G = nx.Graph()
        
        # Add edges with weights based on frequency
        for (word1, word2), count in top_ngrams:
            G.add_edge(word1, word2, weight=count)
        
        # Position nodes using force-directed layout
        pos = nx.spring_layout(G, k=0.3)
        
        # Get edge weights for line thickness
        edge_weights = [G[u][v]['weight']*0.1 for u, v in G.edges()]
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color='skyblue')
        nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='gray', alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        plt.axis('off')
        plt.title(f'Bigram Network for {title}')
        plt.tight_layout()
        
        # Save network visualization
        network_file = os.path.join(output_dir, f"ngram_network_{title.replace(' ', '_')}.png")
        plt.savefig(network_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nBar chart saved to: {bar_chart_file}")
        print(f"Network visualization saved to: {network_file}")
    else:
        print(f"\nBar chart saved to: {bar_chart_file}")
    
    return ngram_freq

def main():
    """Main function to run n-gram analysis."""
    print("="*60)
    print("TEXT N-GRAM ANALYSIS")
    print("="*60)
    
    # Read CSV file
    df, text_column, filename = read_csv_file()
    if df is None:
        return
    
    # Get n-gram size from user
    while True:
        try:
            n = int(input("\nEnter n-gram size (2 for bigrams, 3 for trigrams, etc.): "))
            if n >= 1:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    # Get number of top n-grams to display
    while True:
        try:
            top_n = int(input("Enter number of top n-grams to display (5-50): "))
            if 5 <= top_n <= 50:
                break
            else:
                print("Please enter a number between 5 and 50.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Run n-gram analysis
    title = f"{filename} - {text_column}"
    ngram_freq = ngram_analysis(df[text_column], n=n, top_n=top_n, title=title)
    
    # Print summary
    print("\nN-gram Analysis Summary:")
    print(f"N-gram size: {n}")
    print(f"Total unique {n}-grams: {len(ngram_freq)}")
    
    print(f"\nTop {top_n} most frequent {n}-grams:")
    for gram, count in ngram_freq.most_common(top_n):
        print(f"{' '.join(gram)}: {count}")

if __name__ == "__main__":
    main()