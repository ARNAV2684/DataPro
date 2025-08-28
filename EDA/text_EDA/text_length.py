import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    print("\nPlease select a CSV file for text length analysis:")
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
            print("No suitable text columns found for text length analysis.")
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

def text_length_analysis(text_data, title="Text Length Analysis"):
    """
    Analyzes and visualizes the distribution of text lengths
    
    Parameters:
    text_data (list or Series): Collection of text documents
    title (str): Title for the visualizations
    """
    # Calculate lengths
    char_lengths = [len(str(text)) for text in text_data]
    word_lengths = [len(str(text).split()) for text in text_data]
    
    # Create a DataFrame for easier plotting
    lengths_df = pd.DataFrame({
        'Character Count': char_lengths,
        'Word Count': word_lengths
    })
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histograms
    sns.histplot(lengths_df['Character Count'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Character Counts')
    
    sns.histplot(lengths_df['Word Count'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of Word Counts')
    
    # Box plots
    sns.boxplot(y=lengths_df['Character Count'], ax=axes[1, 0])
    axes[1, 0].set_title('Box Plot of Character Counts')
    
    sns.boxplot(y=lengths_df['Word Count'], ax=axes[1, 1])
    axes[1, 1].set_title('Box Plot of Word Counts')
    
    plt.suptitle(f'Text Length Analysis: {title}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure
    output_file = os.path.join(output_dir, f"text_length_analysis_{title.replace(' ', '_')}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to: {output_file}")
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(lengths_df.describe())
    
    return lengths_df

def main():
    """Main function to run text length analysis."""
    print("="*60)
    print("TEXT LENGTH ANALYSIS")
    print("="*60)
    
    # Read CSV file
    df, text_column, filename = read_csv_file()
    if df is None:
        return
    
    # Run text length analysis
    title = f"{filename} - {text_column}"
    lengths_df = text_length_analysis(df[text_column], title)
    
    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, f"text_length_results_{filename}")
    lengths_df.to_csv(output_file, index=False)
    print(f"\nText length analysis results saved to: {output_file}")
    
    # Print additional insights
    print("\nText Length Insights:")
    print(f"Average Character Count: {lengths_df['Character Count'].mean():.1f} characters")
    print(f"Average Word Count: {lengths_df['Word Count'].mean():.1f} words")
    print(f"Longest Text: {lengths_df['Character Count'].max()} characters")
    print(f"Shortest Text: {lengths_df['Character Count'].min()} characters")
    
    # Calculate percentiles for text length
    percentiles = [5, 25, 50, 75, 95]
    char_percentiles = np.percentile(lengths_df['Character Count'], percentiles)
    
    print("\nCharacter Count Percentiles:")
    for p, val in zip(percentiles, char_percentiles):
        print(f"  • {p}th percentile: {val:.1f} characters")

if __name__ == "__main__":
    main()