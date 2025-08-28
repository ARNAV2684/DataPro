import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def select_csv_file():
    """Scans for CSV files in the current directory and prompts the user to select one."""
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the current directory.")
        return None

    print("Please select a CSV file for correlation analysis:")
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

def correlation_matrix_analysis(df, filename):
    """Perform comprehensive correlation analysis."""
    print("="*80)
    print(f"CORRELATION ANALYSIS - {filename}")
    print("="*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        print("Need at least 2 numeric columns for correlation analysis.")
        return None, None
    
    # Calculate correlation matrices
    pearson_corr = df[numeric_cols].corr(method='pearson')
    spearman_corr = df[numeric_cols].corr(method='spearman')
    
    print(f"\nPearson Correlation Matrix:")
    print("-" * 50)
    print(pearson_corr.round(4))
    
    print(f"\nSpearman Correlation Matrix:")
    print("-" * 50)
    print(spearman_corr.round(4))
    
    # Find strong correlations
    print(f"\nStrong Correlations (|r| > 0.7):")
    print("-" * 50)
    strong_corr = []
    for i in range(len(pearson_corr.columns)):
        for j in range(i+1, len(pearson_corr.columns)):
            corr_val = pearson_corr.iloc[i, j]
            if abs(corr_val) > 0.7:
                strong_corr.append((pearson_corr.columns[i], pearson_corr.columns[j], corr_val))
                print(f"{pearson_corr.columns[i]} - {pearson_corr.columns[j]}: {corr_val:.4f}")
    
    if not strong_corr:
        print("No strong correlations found.")
    
    return pearson_corr, spearman_corr

def create_correlation_heatmap(pearson_corr, spearman_corr, filename):
    """Create correlation heatmaps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Pearson correlation heatmap
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('Pearson Correlation Matrix', fontsize=14, fontweight='bold')
    
    # Spearman correlation heatmap
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title('Spearman Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.suptitle(f'Correlation Analysis - {filename}', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(f'{os.path.splitext(filename)[0]}_correlation_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_pairplot(df, filename):
    """Create pairplot for numeric variables."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 6:
        print(f"Too many numeric columns ({len(numeric_cols)}). Using first 6 for pairplot.")
        numeric_cols = numeric_cols[:6]
    
    if len(numeric_cols) < 2:
        print("Need at least 2 numeric columns for pairplot.")
        return
    
    plt.figure(figsize=(15, 15))
    sns.pairplot(df[numeric_cols], diag_kind='hist', plot_kws={'alpha': 0.6})
    plt.suptitle(f'Pairplot - {filename}', fontsize=16, y=0.98)
    plt.savefig(f'{os.path.splitext(filename)[0]}_pairplot.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_scatter_plots(df, filename):
    """Create scatter plots for highly correlated variables."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        return
    
    corr_matrix = df[numeric_cols].corr()
    
    # Find top correlations
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            correlations.append((corr_matrix.columns[i], corr_matrix.columns[j], abs(corr_val)))
    
    # Sort by absolute correlation and take top 6
    correlations.sort(key=lambda x: x[2], reverse=True)
    top_correlations = correlations[:6]
    
    if not top_correlations:
        return
    
    n_plots = len(top_correlations)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    fig.suptitle(f'Top Correlations Scatter Plots - {filename}', fontsize=16, y=0.98)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (col1, col2, corr_val) in enumerate(top_correlations):
        row = i // n_cols
        col_idx = i % n_cols
        
        axes[row, col_idx].scatter(df[col1], df[col2], alpha=0.6, color='blue')
        
        # Add trend line
        z = np.polyfit(df[col1].dropna(), df[col2].dropna(), 1)
        p = np.poly1d(z)
        axes[row, col_idx].plot(df[col1], p(df[col1]), "r--", alpha=0.8)
        
        axes[row, col_idx].set_xlabel(col1)
        axes[row, col_idx].set_ylabel(col2)
        axes[row, col_idx].set_title(f'{col1} vs {col2}\nCorrelation: {corr_matrix.loc[col1, col2]:.4f}')
        axes[row, col_idx].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(top_correlations), n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{os.path.splitext(filename)[0]}_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def feature_correlation_ranking(df):
    """Rank features by their average correlation with other features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        return
    
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Calculate average correlation for each feature (excluding self-correlation)
    avg_correlations = {}
    for col in corr_matrix.columns:
        # Exclude self-correlation (diagonal = 1)
        other_correlations = corr_matrix[col].drop(col)
        avg_correlations[col] = other_correlations.mean()
    
    print("\nFeature Correlation Ranking:")
    print("-" * 50)
    print("(Average absolute correlation with other features)")
    
    sorted_features = sorted(avg_correlations.items(), key=lambda x: x[1], reverse=True)
    for feature, avg_corr in sorted_features:
        print(f"{feature:25} | Average |r|: {avg_corr:.4f}")

def main():
    """Main function to run correlation analysis."""
    filename = select_csv_file()
    if not filename:
        return
    
    try:
        df = pd.read_csv(filename)
        print(f"\nLoaded dataset: {filename}")
        
        # Perform correlation analysis
        pearson_corr, spearman_corr = correlation_matrix_analysis(df, filename)
        
        if pearson_corr is not None:
            create_correlation_heatmap(pearson_corr, spearman_corr, filename)
            create_pairplot(df, filename)
            create_scatter_plots(df, filename)
            feature_correlation_ranking(df)
        
        print(f"\nCorrelation analysis completed for {filename}")
        print("Correlation visualizations saved as PNG files.")
        
    except Exception as e:
        print(f"Error loading or processing file: {e}")

if __name__ == "__main__":
    main()