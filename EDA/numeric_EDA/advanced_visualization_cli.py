#!/usr/bin/env python3
"""
Advanced Numeric Visualization CLI
Creates comprehensive visualizations for numeric datasets with command-line interface
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_violin_plots(df, output_path):
    """Create violin plots for numeric variables."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("No numeric columns found for violin plots.")
        return False
    
    print(f"Creating violin plots for {len(numeric_cols)} numeric columns...")
    
    n_cols = min(4, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    fig.suptitle('Violin Plots - Distribution Analysis', fontsize=16, y=0.98)
    
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(numeric_cols):
        row = i // n_cols
        col_idx = i % n_cols
        
        ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
        
        # Create violin plot
        data = df[col].dropna()
        ax.violinplot([data], positions=[0], widths=0.6)
        ax.set_title(f'{col}\\nMean: {data.mean():.2f}')
        ax.set_ylabel('Values')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
    
    # Hide empty subplots
    for i in range(len(numeric_cols), n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
        ax.set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path.replace('.csv', '_violin_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Violin plots saved: {plot_path}")
    return True

def create_pair_plots(df, output_path, sample_size=1000):
    """Create pair plots for numeric variables."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        print("Need at least 2 numeric columns for pair plots.")
        return False
    
    # Limit to maximum 6 columns for readability
    if len(numeric_cols) > 6:
        print(f"Too many numeric columns ({len(numeric_cols)}). Using first 6 columns for pair plot.")
        numeric_cols = numeric_cols[:6]
    
    print(f"Creating pair plot for {len(numeric_cols)} columns...")
    
    # Sample data if too large
    df_sample = df[numeric_cols].copy()
    if len(df_sample) > sample_size:
        df_sample = df_sample.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} rows for pair plot visualization")
    
    # Create pair plot
    plt.figure(figsize=(15, 15))
    g = sns.pairplot(df_sample, diag_kind='hist', plot_kws={'alpha': 0.6})
    g.fig.suptitle('Pair Plot - Variable Relationships', y=1.02, fontsize=16)
    
    # Save plot
    plot_path = output_path.replace('.csv', '_pair_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Pair plot saved: {plot_path}")
    return True

def create_pca_analysis(df, output_path, n_components=None):
    """Perform PCA analysis and create visualizations."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        print("Need at least 2 numeric columns for PCA analysis.")
        return {}
    
    print(f"Performing PCA analysis on {len(numeric_cols)} numeric columns...")
    
    # Prepare data
    df_numeric = df[numeric_cols].dropna()
    
    if len(df_numeric) == 0:
        print("No complete cases found for PCA analysis.")
        return {}
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)
    
    # Determine number of components
    if n_components is None:
        n_components = min(len(numeric_cols), len(df_numeric))
    else:
        n_components = min(n_components, len(numeric_cols), len(df_numeric))
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate results
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print(f"\\nPCA Results:")
    print("-" * 50)
    for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
        print(f"PC{i+1}: {var_ratio:.4f} ({cum_var:.4f} cumulative)")
    
    # Create PCA visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Principal Component Analysis', fontsize=16)
    
    # 1. Scree plot
    axes[0, 0].plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
    axes[0, 0].set_title('Scree Plot')
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Cumulative variance plot
    axes[0, 1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
    axes[0, 1].axhline(y=0.8, color='g', linestyle='--', alpha=0.7, label='80% threshold')
    axes[0, 1].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
    axes[0, 1].set_title('Cumulative Explained Variance')
    axes[0, 1].set_xlabel('Principal Component')
    axes[0, 1].set_ylabel('Cumulative Explained Variance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. PC1 vs PC2 scatter plot
    if X_pca.shape[1] >= 2:
        axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
        axes[1, 0].set_title(f'PC1 vs PC2\\n({explained_variance_ratio[0]:.2%} vs {explained_variance_ratio[1]:.2%})')
        axes[1, 0].set_xlabel('First Principal Component')
        axes[1, 0].set_ylabel('Second Principal Component')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Need at least 2 PCs', ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # 4. Feature contributions (loadings) for PC1 and PC2
    if len(numeric_cols) <= 20:  # Only show if not too many features
        loadings = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])
        
        x_pos = np.arange(len(numeric_cols))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, loadings[:, 0], width, label='PC1', alpha=0.8)
        if loadings.shape[1] > 1:
            axes[1, 1].bar(x_pos + width/2, loadings[:, 1], width, label='PC2', alpha=0.8)
        
        axes[1, 1].set_title('Feature Loadings')
        axes[1, 1].set_xlabel('Features')
        axes[1, 1].set_ylabel('Loading')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(numeric_cols, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Too many features\\nfor loading plot', ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path.replace('.csv', '_pca_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PCA analysis plot saved: {plot_path}")
    
    return {
        'explained_variance_ratio': explained_variance_ratio.tolist(),
        'cumulative_variance': cumulative_variance.tolist(),
        'n_components': n_components,
        'feature_names': list(numeric_cols)
    }

def create_distribution_comparison(df, output_path):
    """Create comprehensive distribution comparison plots."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("No numeric columns found for distribution comparison.")
        return False
    
    print(f"Creating distribution comparison for {len(numeric_cols)} columns...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distribution Analysis Comparison', fontsize=16)
    
    # 1. Box plot comparison
    df[numeric_cols].boxplot(ax=axes[0, 0])
    axes[0, 0].set_title('Box Plot Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Violin plot comparison (for first few columns if too many)
    cols_for_violin = numeric_cols[:8] if len(numeric_cols) > 8 else numeric_cols
    df[cols_for_violin].plot(kind='density', ax=axes[0, 1], alpha=0.7)
    axes[0, 1].set_title('Density Plot Comparison')
    axes[0, 1].set_xlabel('Values')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Histogram comparison
    for i, col in enumerate(cols_for_violin):
        axes[1, 0].hist(df[col].dropna(), alpha=0.5, label=col, bins=20)
    axes[1, 0].set_title('Histogram Comparison')
    axes[1, 0].set_xlabel('Values')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # 4. Q-Q plots for normality check (first 4 columns)
    cols_for_qq = numeric_cols[:4] if len(numeric_cols) > 4 else numeric_cols
    
    if len(cols_for_qq) == 1:
        from scipy import stats
        stats.probplot(df[cols_for_qq[0]].dropna(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'Q-Q Plot: {cols_for_qq[0]}')
    else:
        # Create a mini subplot for multiple Q-Q plots
        axes[1, 1].text(0.5, 0.5, f'Q-Q plots for\\n{len(cols_for_qq)} variables\\n(see individual files)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Q-Q Plot Reference')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path.replace('.csv', '_distribution_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution comparison saved: {plot_path}")
    return True

def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Create advanced visualizations for numeric datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python advanced_visualization_cli.py --input data.csv --output results.csv
  python advanced_visualization_cli.py --input data.csv --output results.csv --pca-components 5
  python advanced_visualization_cli.py --input data.csv --output results.csv --skip-pca --sample-size 500
        '''
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output', '-o', required=True,
                       help='Path to output CSV file')
    parser.add_argument('--pca-components', type=int, default=None,
                       help='Number of PCA components to compute (default: all)')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Sample size for pair plots (default: 1000)')
    parser.add_argument('--skip-pca', action='store_true',
                       help='Skip PCA analysis')
    parser.add_argument('--skip-pairs', action='store_true',
                       help='Skip pair plot generation')
    parser.add_argument('--plots-only', action='store_true',
                       help='Generate only plots, skip result CSV')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    
    try:
        # Load dataset
        print(f"Loading dataset: {args.input}")
        df = pd.read_csv(args.input)
        print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            print("Error: No numeric columns found for visualization.")
            sys.exit(1)
        
        print(f"Found {len(numeric_cols)} numeric columns for visualization")
        
        results = {}
        
        # Generate visualizations
        print("\\nGenerating advanced visualizations...")
        
        # 1. Violin plots
        create_violin_plots(df, args.output)
        
        # 2. Distribution comparison
        create_distribution_comparison(df, args.output)
        
        # 3. Pair plots (if not skipped)
        if not args.skip_pairs:
            create_pair_plots(df, args.output, args.sample_size)
        
        # 4. PCA analysis (if not skipped)
        if not args.skip_pca:
            pca_results = create_pca_analysis(df, args.output, args.pca_components)
            results['pca'] = pca_results
        
        if not args.plots_only:
            # Create summary output
            summary_data = []
            
            # Basic dataset info
            summary_data.append({
                'metric': 'total_rows',
                'value': len(df),
                'description': 'Total number of rows in dataset'
            })
            summary_data.append({
                'metric': 'total_columns',
                'value': len(df.columns),
                'description': 'Total number of columns in dataset'
            })
            summary_data.append({
                'metric': 'numeric_columns',
                'value': len(numeric_cols),
                'description': 'Number of numeric columns'
            })
            
            # PCA summary if performed
            if not args.skip_pca and 'pca' in results:
                pca_data = results['pca']
                for i, var_ratio in enumerate(pca_data['explained_variance_ratio']):
                    summary_data.append({
                        'metric': f'pc{i+1}_variance_explained',
                        'value': var_ratio,
                        'description': f'Variance explained by Principal Component {i+1}'
                    })
            
            # Save summary
            output_df = pd.DataFrame(summary_data)
            output_df.to_csv(args.output, index=False)
            print(f"\\nVisualization summary saved: {args.output}")
        
        print("\\nAdvanced visualization completed successfully!")
        print("Generated plots:")
        print("- Violin plots for distribution analysis")
        print("- Distribution comparison plots")
        if not args.skip_pairs:
            print("- Pair plots for variable relationships")
        if not args.skip_pca:
            print("- PCA analysis plots")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
