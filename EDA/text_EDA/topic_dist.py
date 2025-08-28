import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import matplotlib.colors as mcolors

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
    print("\nPlease select a CSV file for topic modeling:")
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
            print("No suitable text columns found for topic modeling.")
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

def topic_modeling_visualization(text_data, n_topics=5, n_top_words=10, title="Topic Modeling"):
    """
    Performs topic modeling and visualizes the results
    
    Parameters:
    text_data (list or Series): Collection of text documents
    n_topics (int): Number of topics to extract
    n_top_words (int): Number of top words to display per topic
    title (str): Title for the visualizations
    """
    # Create TF-IDF representation
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(text_data)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Create NMF model
    nmf = NMF(n_components=n_topics, random_state=42).fit(tfidf)
    
    # Create LDA model
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42).fit(tfidf)
    
    # Function to display top words
    def display_topics(model, feature_names, n_top_words, model_name):
        topic_dict = {}
        for topic_idx, topic in enumerate(model.components_):
            topic_dict[topic_idx] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        return topic_dict
    
    # Get topics
    nmf_topics = display_topics(nmf, tfidf_feature_names, n_top_words, 'NMF')
    lda_topics = display_topics(lda, tfidf_feature_names, n_top_words, 'LDA')
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Visualize topics
    fig, axes = plt.subplots(n_topics, 2, figsize=(15, n_topics * 3))
    
    # Handle case where n_topics is 1
    if n_topics == 1:
        axes = axes.reshape(1, 2)
    
    # Colors for visualization
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    for i in range(n_topics):
        # NMF topics
        x_pos = np.arange(len(nmf_topics[i]))
        axes[i, 0].bar(x_pos, height=np.ones(len(nmf_topics[i])), width=0.8, 
                      color=colors[i % len(colors)])
        axes[i, 0].set_xticks(x_pos)
        axes[i, 0].set_xticklabels(nmf_topics[i], rotation=45, ha='right')
        axes[i, 0].set_title(f'Topic {i+1} (NMF)')
        axes[i, 0].set_yticks([])
        
        # LDA topics
        x_pos = np.arange(len(lda_topics[i]))
        axes[i, 1].bar(x_pos, height=np.ones(len(lda_topics[i])), width=0.8, 
                      color=colors[i % len(colors)])
        axes[i, 1].set_xticks(x_pos)
        axes[i, 1].set_xticklabels(lda_topics[i], rotation=45, ha='right')
        axes[i, 1].set_title(f'Topic {i+1} (LDA)')
        axes[i, 1].set_yticks([])
    
    plt.suptitle(f'Topic Modeling: {title}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save figure
    output_file = os.path.join(output_dir, f"topic_modeling_{title.replace(' ', '_')}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to: {output_file}")
    
    return {
        'nmf_model': nmf,
        'lda_model': lda,
        'nmf_topics': nmf_topics,
        'lda_topics': lda_topics,
        'tfidf_vectorizer': tfidf_vectorizer
    }

def main():
    """Main function to run topic modeling."""
    print("="*60)
    print("TEXT TOPIC MODELING")
    print("="*60)
    
    # Read CSV file
    df, text_column, filename = read_csv_file()
    if df is None:
        return
    
    # Get number of topics from user
    while True:
        try:
            n_topics = int(input("\nEnter number of topics to extract (1-10): "))
            if 1 <= n_topics <= 10:
                break
            else:
                print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Run topic modeling
    title = f"{filename} - {text_column}"
    topic_results = topic_modeling_visualization(df[text_column], n_topics=n_topics, title=title)
    
    # Print summary
    print("\nTopic Modeling Summary:")
    print(f"Number of topics: {n_topics}")
    print(f"Number of documents: {len(df)}")
    
    # Display NMF topics
    print("\nNMF Topics:")
    for i, words in topic_results['nmf_topics'].items():
        print(f"Topic {i+1}: {', '.join(words)}")
    
    # Display LDA topics
    print("\nLDA Topics:")
    for i, words in topic_results['lda_topics'].items():
        print(f"Topic {i+1}: {', '.join(words)}")

if __name__ == "__main__":
    main()