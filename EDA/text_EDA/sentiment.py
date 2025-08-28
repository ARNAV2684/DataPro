import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns

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
    print("\nPlease select a CSV file for sentiment analysis:")
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
            print("No suitable text columns found for sentiment analysis.")
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

def sentiment_analysis_visualization(text_data, title="Sentiment Analysis"):
    """
    Analyzes and visualizes sentiment in text data
    
    Parameters:
    text_data (list or Series): Collection of text documents
    title (str): Title for the visualizations
    """
    # Calculate sentiment scores
    polarities = []
    subjectivities = []
    
    for text in text_data:
        blob = TextBlob(str(text))
        polarities.append(blob.sentiment.polarity)
        subjectivities.append(blob.sentiment.subjectivity)
    
    # Create DataFrame
    sentiment_df = pd.DataFrame({
        'Text': text_data,
        'Polarity': polarities,
        'Subjectivity': subjectivities
    })
    
    # Count positive and negative texts
    positive_count = sum(1 for pol in polarities if pol > 0.2)
    negative_count = sum(1 for pol in polarities if pol < -0.2)
    neutral_count = len(polarities) - positive_count - negative_count
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Polarity distribution
    sns.histplot(sentiment_df['Polarity'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Sentiment Polarity')
    axes[0, 0].set_xlabel('Polarity (-1: Negative, 1: Positive)')
    
    # Subjectivity distribution
    sns.histplot(sentiment_df['Subjectivity'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of Sentiment Subjectivity')
    axes[0, 1].set_xlabel('Subjectivity (0: Objective, 1: Subjective)')
    
    # Scatter plot
    sns.scatterplot(x='Polarity', y='Subjectivity', data=sentiment_df, ax=axes[1, 0], alpha=0.6)
    axes[1, 0].set_title('Polarity vs Subjectivity')
    axes[1, 0].set_xlabel('Polarity')
    axes[1, 0].set_ylabel('Subjectivity')
    
    # Word cloud colored by sentiment
    # Generate text with sentiment info for word cloud
    pos_words = ' '.join([str(text) for text, pol in zip(text_data, polarities) if pol > 0.2])
    neg_words = ' '.join([str(text) for text, pol in zip(text_data, polarities) if pol < -0.2])
    
    axes[1, 1].axis('off')
    if len(pos_words) > 0 and len(neg_words) > 0:
        axes[1, 1].text(0.5, 0.7, f"Positive Words: {positive_count}", fontsize=14, ha='center')
        axes[1, 1].text(0.5, 0.5, f"Negative Words: {negative_count}", fontsize=14, ha='center', color='red')
        axes[1, 1].text(0.5, 0.3, f"Neutral Words: {neutral_count}", fontsize=14, ha='center', color='gray')
        axes[1, 1].text(0.5, 0.1, f"Avg. Polarity: {np.mean(polarities):.2f}", 
                       fontsize=14, ha='center', color='green')
    else:
        axes[1, 1].text(0.5, 0.5, "Insufficient data for sentiment word clouds", 
                       fontsize=12, ha='center')
    
    plt.suptitle(f'Sentiment Analysis: {title}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure
    output_file = os.path.join(output_dir, f"sentiment_analysis_{title.replace(' ', '_')}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to: {output_file}")
    
    # Add the counts to the DataFrame for use in the main function
    sentiment_df.attrs['pos_count'] = positive_count
    sentiment_df.attrs['neg_count'] = negative_count
    sentiment_df.attrs['neutral_count'] = neutral_count
    
    return sentiment_df

def main():
    """Main function to run sentiment analysis."""
    print("="*60)
    print("TEXT SENTIMENT ANALYSIS")
    print("="*60)
    
    # Read CSV file
    df, text_column, filename = read_csv_file()
    if df is None:
        return
    
    # Run sentiment analysis
    title = f"{filename} - {text_column}"
    sentiment_df = sentiment_analysis_visualization(df[text_column], title)
    
    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, f"sentiment_results_{filename}")
    sentiment_df.to_csv(output_file, index=False)
    print(f"\nSentiment analysis results saved to: {output_file}")
    
    # Print summary
    print("\nSentiment Analysis Summary:")
    print(f"Average Polarity: {sentiment_df['Polarity'].mean():.4f} (-1: Negative, 1: Positive)")
    print(f"Average Subjectivity: {sentiment_df['Subjectivity'].mean():.4f} (0: Objective, 1: Subjective)")
    
    polarity_counts = {
        "Positive (> 0.05)": (sentiment_df['Polarity'] > 0.05).sum(),
        "Neutral (-0.05 to 0.05)": ((sentiment_df['Polarity'] >= -0.05) & (sentiment_df['Polarity'] <= 0.05)).sum(),
        "Negative (< -0.05)": (sentiment_df['Polarity'] < -0.05).sum()
    }
    
    print("\nSentiment Distribution:")
    for category, count in polarity_counts.items():
        percentage = (count / len(sentiment_df)) * 100
        print(f"{category}: {count} ({percentage:.2f}%)")
    
    # Display word count by sentiment
    print("\nWord Count by Sentiment:")
    print(f"Positive Words (> 0.2): {sentiment_df.attrs['pos_count']}")
    print(f"Negative Words (< -0.2): {sentiment_df.attrs['neg_count']}")
    print(f"Neutral Words: {sentiment_df.attrs['neutral_count']}")
    
    # Display first few rows of the sentiment DataFrame
    print("\nSample Sentiment Analysis Results:")
    print(sentiment_df.head())

if __name__ == "__main__":
    main()