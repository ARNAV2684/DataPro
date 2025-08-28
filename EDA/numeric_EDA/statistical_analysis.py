import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def select_csv_file():
    """Scans for CSV files in the current directory and prompts the user to select one."""
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the current directory.")
        return None

    print("Please select a CSV file for EDA:")
    for i, filename in enumerate(csv_files):
        print(f"{i + 1}: {filename}")

    while True:
        try:
            choice = int(input(f"Enter the number (1-{len(csv_files)}): ")) - 1
            if 0 <= choice < len(csv_files):
                return csv_files[choice]
            else:
                print(f"Invalid number. Please enter a number between 1 and {len(csv_files)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def basic_info_analysis(df, filename):
    """Perform basic dataset information analysis."""
    print("="*80)
    print(f"BASIC DATASET INFORMATION - {filename}")
    print("="*80)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\nColumn Information:")
    print("-" * 50)
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        print(f"{col:20} | {str(dtype):10} | Nulls: {null_count:5} ({null_pct:5.1f}%) | Unique: {unique_count:5}")
    
    print("\nData Types Summary:")
    print(df.dtypes.value_counts())
    
    print("\nMissing Values Summary:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing Percentage': (missing.values / len(df)) * 100
        })
        print(missing_df[missing_df['Missing Count'] > 0].to_string(index=False))
    else:
        print("No missing values found!")

def statistical_summary(df):
    """Generate comprehensive statistical summary."""
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        print("\nDescriptive Statistics:")
        print("-" * 50)
        desc_stats = df[numeric_cols].describe()
        print(desc_stats.round(4))
        
        print("\nAdditional Statistics:")
        print("-" * 50)
        additional_stats = pd.DataFrame({
            'Skewness': df[numeric_cols].skew(),
            'Kurtosis': df[numeric_cols].kurtosis(),
            'Variance': df[numeric_cols].var(),
            'IQR': df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
        })
        print(additional_stats.round(4))
        
        # Outlier detection using IQR method
        print("\nOutlier Analysis (IQR Method):")
        print("-" * 50)
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            print(f"{col:20} | Outliers: {len(outliers):5} ({len(outliers)/len(df)*100:5.1f}%)")
    
    # Categorical analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print("\nCategorical Variables Analysis:")
        print("-" * 50)
        for col in categorical_cols:
            print(f"\n{col}:")
            value_counts = df[col].value_counts()
            print(value_counts.head(10))

def create_distribution_plots(df, filename):
    """Create distribution plots for numeric variables."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("No numeric columns found for distribution plots.")
        return
    
    n_cols = min(4, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    fig.suptitle(f'Distribution Plots - {filename}', fontsize=16, y=0.98)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(numeric_cols):
        row = i // n_cols
        col_idx = i % n_cols
        
        # Histogram with KDE
        axes[row, col_idx].hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[row, col_idx].set_title(f'{col}\nMean: {df[col].mean():.2f}, Std: {df[col].std():.2f}')
        axes[row, col_idx].set_xlabel(col)
        axes[row, col_idx].set_ylabel('Frequency')
        axes[row, col_idx].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(numeric_cols), n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{os.path.splitext(filename)[0]}_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_box_plots(df, filename):
    """Create box plots to visualize outliers."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("No numeric columns found for box plots.")
        return
    
    n_cols = min(4, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    fig.suptitle(f'Box Plots - Outlier Detection - {filename}', fontsize=16, y=0.98)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(numeric_cols):
        row = i // n_cols
        col_idx = i % n_cols
        
        axes[row, col_idx].boxplot(df[col].dropna())
        axes[row, col_idx].set_title(f'{col}')
        axes[row, col_idx].set_ylabel('Values')
        axes[row, col_idx].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(numeric_cols), n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{os.path.splitext(filename)[0]}_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run basic statistical EDA."""
    filename = select_csv_file()
    if not filename:
        return
    
    try:
        df = pd.read_csv(filename)
        print(f"\nLoaded dataset: {filename}")
        
        # Perform analyses
        basic_info_analysis(df, filename)
        statistical_summary(df)
        create_distribution_plots(df, filename)
        create_box_plots(df, filename)
        
        print(f"\nBasic statistical EDA completed for {filename}")
        print("Distribution and box plot visualizations saved as PNG files.")
        
    except Exception as e:
        print(f"Error loading or processing file: {e}")

if __name__ == "__main__":
    main()