#!/usr/bin/env python3
"""
EDA Manager CLI
Unified CLI interface for running all EDA techniques on numeric datasets
"""

import os
import pandas as pd
import argparse
import sys
import subprocess
from pathlib import Path
import json

# Map of technique names to their CLI script files
TECHNIQUE_MAP = {
    'statistical_analysis': 'statistical_analysis_cli.py',
    'correlation_analysis': 'correlation_analysis_cli.py', 
    'advanced_visualization': 'advanced_visualization_cli.py',
    'new_av': 'new_av.py'  # Already has CLI interface
}

def get_available_techniques():
    """Get available EDA techniques."""
    return list(TECHNIQUE_MAP.keys())

def run_technique(technique, input_file, output_file, **kwargs):
    """Run a specific EDA technique using its CLI script."""
    if technique not in TECHNIQUE_MAP:
        print(f"Error: Unknown technique '{technique}'")
        return False
    
    script_name = TECHNIQUE_MAP[technique]
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"Error: Script '{script_name}' not found at {script_path}")
        return False
    
    # Build command based on technique
    if technique == 'new_av':
        # new_av.py has different interface - only takes input file
        cmd = [sys.executable, str(script_path), str(input_file)]
    else:
        # Standard CLI interface
        cmd = [sys.executable, str(script_path), '--input', str(input_file), '--output', str(output_file)]
        
        # Add technique-specific arguments
        if technique == 'correlation_analysis':
            if kwargs.get('threshold'):
                cmd.extend(['--threshold', str(kwargs['threshold'])])
        elif technique == 'advanced_visualization':
            if kwargs.get('pca_components'):
                cmd.extend(['--pca-components', str(kwargs['pca_components'])])
            if kwargs.get('sample_size'):
                cmd.extend(['--sample-size', str(kwargs['sample_size'])])
            if kwargs.get('skip_pca'):
                cmd.append('--skip-pca')
            if kwargs.get('skip_pairs'):
                cmd.append('--skip-pairs')
    
    print(f"Running {technique}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {technique} completed successfully!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {technique}: {e}")
        if e.stdout:
            print("Stdout:", e.stdout)
        if e.stderr:
            print("Stderr:", e.stderr)
        return False

def run_all_techniques(input_file, output_dir, **kwargs):
    """Run all available EDA techniques."""
    results = {}
    total_techniques = len(TECHNIQUE_MAP)
    successful = 0
    
    print(f"\\nRunning all {total_techniques} EDA techniques...")
    print("-" * 60)
    
    for i, technique in enumerate(TECHNIQUE_MAP.keys(), 1):
        print(f"\\n[{i}/{total_techniques}] {technique.replace('_', ' ').title()}")
        
        # Create technique-specific output file
        if technique == 'new_av':
            # new_av.py generates files in the same directory as input
            success = run_technique(technique, input_file, None, **kwargs)
        else:
            output_file = Path(output_dir) / f"{technique}_results.csv"
            success = run_technique(technique, input_file, output_file, **kwargs)
        
        results[technique] = success
        if success:
            successful += 1
    
    print("\\n" + "=" * 60)
    print(f"EDA Analysis Complete: {successful}/{total_techniques} techniques successful")
    print("=" * 60)
    
    # Print summary
    for technique, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {technique.replace('_', ' ').title()}")
    
    return results

def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run EDA analysis techniques on numeric datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Available techniques:
  - statistical_analysis: Basic statistics, distributions, box plots
  - correlation_analysis: Correlation matrices, heatmaps, scatter plots  
  - advanced_visualization: Violin plots, pair plots, PCA analysis
  - new_av: Alternative visualization suite (KDE, summary dashboard)

Examples:
  # Run specific technique
  python eda_manager_cli.py --input data.csv --technique statistical_analysis --output results.csv
  
  # Run all techniques
  python eda_manager_cli.py --input data.csv --all --output-dir ./results/
  
  # Run with custom parameters
  python eda_manager_cli.py --input data.csv --technique correlation_analysis --output results.csv --threshold 0.7
        '''
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Path to input CSV file')
    parser.add_argument('--technique', '-t', choices=get_available_techniques(),
                       help='Specific EDA technique to run')
    parser.add_argument('--output', '-o',
                       help='Path to output CSV file (required for specific techniques)')
    parser.add_argument('--output-dir', '-d', default='./eda_results',
                       help='Output directory for results (default: ./eda_results)')
    parser.add_argument('--all', action='store_true',
                       help='Run all available EDA techniques')
    
    # Technique-specific arguments
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Correlation threshold for correlation_analysis (default: 0.5)')
    parser.add_argument('--pca-components', type=int,
                       help='Number of PCA components for advanced_visualization')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Sample size for pair plots in advanced_visualization (default: 1000)')
    parser.add_argument('--skip-pca', action='store_true',
                       help='Skip PCA analysis in advanced_visualization')
    parser.add_argument('--skip-pairs', action='store_true',
                       help='Skip pair plots in advanced_visualization')
    parser.add_argument('--list-techniques', action='store_true',
                       help='List all available techniques and exit')
    
    args = parser.parse_args()
    
    # List techniques and exit
    if args.list_techniques:
        print("Available EDA techniques:")
        for technique in get_available_techniques():
            print(f"  - {technique}: {technique.replace('_', ' ').title()}")
        sys.exit(0)
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    
    # Validate arguments
    if not args.all and not args.technique:
        print("Error: Must specify either --technique or --all")
        sys.exit(1)
    
    if args.technique and not args.output and args.technique != 'new_av':
        print(f"Error: --output is required when using --technique {args.technique}")
        sys.exit(1)
    
    try:
        # Load dataset to validate
        print(f"Loading dataset: {args.input}")
        df = pd.read_csv(args.input)
        print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) == 0:
            print("Warning: No numeric columns found for analysis.")
        else:
            print(f"Found {len(numeric_cols)} numeric columns for analysis")
        
        # Prepare technique arguments
        technique_kwargs = {
            'threshold': args.threshold,
            'pca_components': args.pca_components,
            'sample_size': args.sample_size,
            'skip_pca': args.skip_pca,
            'skip_pairs': args.skip_pairs
        }
        
        if args.all:
            # Create output directory
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Output directory: {output_dir.absolute()}")
            
            # Run all techniques
            results = run_all_techniques(args.input, output_dir, **technique_kwargs)
            
            # Save summary
            summary_file = output_dir / 'eda_summary.json'
            with open(summary_file, 'w') as f:
                json.dump({
                    'input_file': args.input,
                    'dataset_shape': df.shape,
                    'numeric_columns': len(numeric_cols),
                    'results': results,
                    'parameters': technique_kwargs
                }, f, indent=2)
            print(f"\\nSummary saved: {summary_file}")
            
        else:
            # Run specific technique
            success = run_technique(args.technique, args.input, args.output, **technique_kwargs)
            
            if success:
                print(f"\\n✅ {args.technique} analysis completed successfully!")
                if args.output and os.path.exists(args.output):
                    print(f"Results saved: {args.output}")
            else:
                print(f"\\n❌ {args.technique} analysis failed!")
                sys.exit(1)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
