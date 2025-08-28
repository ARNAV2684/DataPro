#!/usr/bin/env python3
"""
Numeric Statistical Analysis CLI
Performs comprehensive statistical analysis on numeric datasets with command-line interface
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def basic_info_analysis(df):
    """Perform basic dataset information analysis."""
    analysis_results = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'columns_info': [],
        'data_types_summary': df.dtypes.value_counts().to_dict()
    }
    
    print("="*80)
    print("BASIC DATASET INFORMATION")
    print("="*80)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Memory Usage: {analysis_results['memory_usage_mb']:.2f} MB")
    
    print("\nColumn Information:")
    print("-" * 50)
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        
        col_info = {
            'column': col,
            'dtype': str(dtype),
            'null_count': null_count,
            'null_percentage': null_pct,
            'unique_count': unique_count
        }
        analysis_results['columns_info'].append(col_info)
        
        print(f"{col:20} | {str(dtype):10} | Nulls: {null_count:5} ({null_pct:5.1f}%) | Unique: {unique_count:5}")
    
    print("\nData Types Summary:")
    print(df.dtypes.value_counts())
    
    return analysis_results

def statistical_summary(df):
    """Generate comprehensive statistical summary."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("No numeric columns found for statistical analysis.")
        return {}
    
    print("="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    
    # Basic descriptive statistics
    desc_stats = df[numeric_cols].describe()
    print("\nDescriptive Statistics:")
    print("-" * 50)
    print(desc_stats.round(4))
    
    # Additional statistical measures
    print("\nAdditional Statistical Measures:")
    print("-" * 50)
    
    stats_results = {}
    for col in numeric_cols:
        data = df[col].dropna()
        
        col_stats = {
            'mean': data.mean(),
            'median': data.median(),
            'mode': data.mode().iloc[0] if not data.mode().empty else np.nan,
            'std': data.std(),
            'variance': data.var(),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'min': data.min(),
            'max': data.max(),
            'range': data.max() - data.min(),
            'q1': data.quantile(0.25),
            'q3': data.quantile(0.75),
            'iqr': data.quantile(0.75) - data.quantile(0.25)
        }
        
        stats_results[col] = col_stats
        
        print(f"\n{col}:")
        print(f"  Skewness: {col_stats['skewness']:.4f}")
        print(f"  Kurtosis: {col_stats['kurtosis']:.4f}")
        print(f"  Range: {col_stats['range']:.4f}")
        print(f"  IQR: {col_stats['iqr']:.4f}")
    
    return stats_results

def create_distribution_plots(df, output_path):
    """Create distribution plots for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("No numeric columns found for distribution plots.")
        return False
    
    print(f"\nCreating distribution plots for {len(numeric_cols)} numeric columns...")
    
    n_cols = min(4, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    fig.suptitle('Distribution Plots', fontsize=16, y=0.98)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(numeric_cols):
        row = i // n_cols
        col_idx = i % n_cols
        
        # Histogram with KDE
        axes[row, col_idx].hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[row, col_idx].set_title(f'{col}\\nMean: {df[col].mean():.2f}, Std: {df[col].std():.2f}')
        axes[row, col_idx].set_xlabel(col)
        axes[row, col_idx].set_ylabel('Frequency')
        axes[row, col_idx].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(numeric_cols), n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path.replace('.csv', '_distributions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to prevent display in CLI mode
    
    print(f"Distribution plots saved: {plot_path}")
    return True

def create_box_plots(df, output_path):
    """Create box plots to visualize outliers."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("No numeric columns found for box plots.")
        return False
    
    print(f"Creating box plots for outlier detection...")
    
    n_cols = min(4, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    fig.suptitle('Box Plots - Outlier Detection', fontsize=16, y=0.98)
    
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
    
    # Save plot
    plot_path = output_path.replace('.csv', '_boxplots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to prevent display in CLI mode
    
    print(f"Box plots saved: {plot_path}")
    return True

def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Perform comprehensive statistical analysis on numeric datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python statistical_analysis_cli.py --input data.csv --output results.csv
  python statistical_analysis_cli.py --input data.csv --output results.csv --no-plots
        '''
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output', '-o', required=True,
                       help='Path to output CSV file')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating visualization plots')
    parser.add_argument('--plots-only', action='store_true',
                       help='Generate only plots, skip statistical analysis')
    
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
        
        results = {}
        
        if not args.plots_only:
            # Perform statistical analyses
            print("\\nPerforming statistical analysis...")
            results['basic_info'] = basic_info_analysis(df)
            results['statistical_summary'] = statistical_summary(df)
        
        if not args.no_plots:
            # Create visualizations
            print("\\nGenerating visualizations...")
            create_distribution_plots(df, args.output)
            create_box_plots(df, args.output)
        
        if not args.plots_only:
            # Save results to output file
            output_df = pd.DataFrame({
                'metric': ['dataset_rows', 'dataset_columns', 'memory_usage_mb'],
                'value': [df.shape[0], df.shape[1], df.memory_usage(deep=True).sum() / 1024**2]
            })
            
            # Add statistical summary for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                col_data = df[col].dropna()
                stats_data = pd.DataFrame({
                    'metric': [f'{col}_mean', f'{col}_median', f'{col}_std', f'{col}_min', f'{col}_max'],
                    'value': [col_data.mean(), col_data.median(), col_data.std(), col_data.min(), col_data.max()]
                })
                output_df = pd.concat([output_df, stats_data], ignore_index=True)
            
            output_df.to_csv(args.output, index=False)
            print(f"\\nStatistical analysis results saved: {args.output}")
        
        print("\\nStatistical analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
