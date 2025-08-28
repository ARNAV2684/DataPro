#!/usr/bin/env python3
"""
Numeric Correlation Analysis CLI
Performs comprehensive correlation analysis on numeric datasets with command-line interface
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def correlation_matrix_analysis(df):
    """Perform comprehensive correlation analysis."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        print("Need at least 2 numeric columns for correlation analysis.")
        return {}
    
    print("="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    # Calculate correlation matrices
    pearson_corr = df[numeric_cols].corr(method='pearson')
    spearman_corr = df[numeric_cols].corr(method='spearman')
    
    print(f"\\nPearson Correlation Matrix:")
    print("-" * 50)
    print(pearson_corr.round(4))
    
    print(f"\\nSpearman Correlation Matrix:")
    print("-" * 50)
    print(spearman_corr.round(4))
    
    return {
        'pearson_correlation': pearson_corr,
        'spearman_correlation': spearman_corr,
        'numeric_columns': list(numeric_cols)
    }

def find_high_correlations(corr_matrix, threshold=0.7):
    """Find pairs of variables with high correlation."""
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                high_corr_pairs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_value
                })
    
    if high_corr_pairs:
        print(f"\\nHigh Correlations (|r| >= {threshold}):")
        print("-" * 50)
        for pair in high_corr_pairs:
            print(f"{pair['var1']} <-> {pair['var2']}: {pair['correlation']:.4f}")
    else:
        print(f"\\nNo high correlations found (|r| >= {threshold})")
    
    return high_corr_pairs

def create_correlation_heatmap(corr_matrix, title, output_path):
    """Create correlation heatmap visualization."""
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdYlBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={'shrink': 0.8})
    
    plt.title(f'{title} Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path.replace('.csv', f'_{title.lower()}_correlation.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{title} correlation heatmap saved: {plot_path}")
    return plot_path

def create_scatter_plots(df, high_corr_pairs, output_path):
    """Create scatter plots for highly correlated variable pairs."""
    if not high_corr_pairs:
        print("No high correlation pairs found for scatter plots.")
        return []
    
    print(f"Creating scatter plots for {len(high_corr_pairs)} high correlation pairs...")
    
    n_pairs = len(high_corr_pairs)
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle('Scatter Plots - High Correlations', fontsize=16, y=0.98)
    
    if n_rows == 1:
        axes = np.array(axes).reshape(1, -1)
    elif n_pairs == 1:
        axes = np.array([axes])
    
    plot_paths = []
    
    for i, pair in enumerate(high_corr_pairs):
        row = i // n_cols
        col_idx = i % n_cols
        
        x_data = df[pair['var1']].dropna()
        y_data = df[pair['var2']].dropna()
        
        # Create scatter plot
        axes[row, col_idx].scatter(x_data, y_data, alpha=0.6, color='steelblue')
        axes[row, col_idx].set_xlabel(pair['var1'])
        axes[row, col_idx].set_ylabel(pair['var2'])
        axes[row, col_idx].set_title(f"r = {pair['correlation']:.3f}")
        axes[row, col_idx].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        axes[row, col_idx].plot(x_data, p(x_data), "r--", alpha=0.8)
    
    # Hide empty subplots
    for i in range(n_pairs, n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        if n_rows > 1 or n_cols > 1:
            axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path.replace('.csv', '_scatter_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scatter plots saved: {plot_path}")
    plot_paths.append(plot_path)
    
    return plot_paths

def correlation_strength_analysis(corr_matrix):
    """Analyze correlation strength distribution."""
    # Flatten correlation matrix (excluding diagonal)
    corr_values = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_values.append(abs(corr_matrix.iloc[i, j]))
    
    if not corr_values:
        return {}
    
    # Categorize correlations
    weak = sum(1 for r in corr_values if r < 0.3)
    moderate = sum(1 for r in corr_values if 0.3 <= r < 0.7)
    strong = sum(1 for r in corr_values if r >= 0.7)
    
    total = len(corr_values)
    
    print("\\nCorrelation Strength Analysis:")
    print("-" * 50)
    print(f"Weak correlations (|r| < 0.3): {weak} ({weak/total*100:.1f}%)")
    print(f"Moderate correlations (0.3 ≤ |r| < 0.7): {moderate} ({moderate/total*100:.1f}%)")
    print(f"Strong correlations (|r| ≥ 0.7): {strong} ({strong/total*100:.1f}%)")
    print(f"Total variable pairs: {total}")
    
    return {
        'weak_count': weak,
        'moderate_count': moderate,
        'strong_count': strong,
        'total_pairs': total,
        'max_correlation': max(corr_values),
        'min_correlation': min(corr_values),
        'avg_correlation': np.mean(corr_values)
    }

def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Perform comprehensive correlation analysis on numeric datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python correlation_analysis_cli.py --input data.csv --output results.csv
  python correlation_analysis_cli.py --input data.csv --output results.csv --threshold 0.8
  python correlation_analysis_cli.py --input data.csv --output results.csv --no-plots
        '''
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output', '-o', required=True,
                       help='Path to output CSV file')
    parser.add_argument('--threshold', '-t', type=float, default=0.7,
                       help='Correlation threshold for identifying high correlations (default: 0.7)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating visualization plots')
    parser.add_argument('--method', choices=['pearson', 'spearman', 'both'], default='both',
                       help='Correlation method to use (default: both)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    
    # Validate threshold
    if not 0 <= args.threshold <= 1:
        print(f"Error: Threshold must be between 0 and 1.")
        sys.exit(1)
    
    try:
        # Load dataset
        print(f"Loading dataset: {args.input}")
        df = pd.read_csv(args.input)
        print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            print("Error: Need at least 2 numeric columns for correlation analysis.")
            sys.exit(1)
        
        print(f"Found {len(numeric_cols)} numeric columns for analysis")
        
        # Perform correlation analysis
        print("\\nPerforming correlation analysis...")
        corr_results = correlation_matrix_analysis(df)
        
        # Analyze correlation strengths
        strength_analysis = correlation_strength_analysis(corr_results['pearson_correlation'])
        
        # Find high correlations
        high_corr_pairs = find_high_correlations(corr_results['pearson_correlation'], args.threshold)
        
        if not args.no_plots:
            # Create visualizations
            print("\\nGenerating visualizations...")
            
            if args.method in ['pearson', 'both']:
                create_correlation_heatmap(corr_results['pearson_correlation'], 'Pearson', args.output)
            
            if args.method in ['spearman', 'both']:
                create_correlation_heatmap(corr_results['spearman_correlation'], 'Spearman', args.output)
            
            # Create scatter plots for high correlations
            create_scatter_plots(df, high_corr_pairs, args.output)
        
        # Prepare output data
        output_data = []
        
        # Add correlation matrix data
        for i, col1 in enumerate(corr_results['pearson_correlation'].columns):
            for j, col2 in enumerate(corr_results['pearson_correlation'].columns):
                if i != j:  # Skip diagonal
                    output_data.append({
                        'variable_1': col1,
                        'variable_2': col2,
                        'pearson_correlation': corr_results['pearson_correlation'].iloc[i, j],
                        'spearman_correlation': corr_results['spearman_correlation'].iloc[i, j],
                        'abs_pearson': abs(corr_results['pearson_correlation'].iloc[i, j]),
                        'high_correlation': abs(corr_results['pearson_correlation'].iloc[i, j]) >= args.threshold
                    })
        
        # Save results
        output_df = pd.DataFrame(output_data)
        output_df = output_df.sort_values('abs_pearson', ascending=False)
        output_df.to_csv(args.output, index=False)
        
        print(f"\\nCorrelation analysis results saved: {args.output}")
        print(f"Found {len(high_corr_pairs)} high correlation pairs (threshold: {args.threshold})")
        print("\\nCorrelation analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
