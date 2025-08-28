"""
Scale and Jitter-based Data Augmentation Script (Command Line Interface)
========================================================================
This script performs scale and jitter augmentation on numeric columns of CSV files.
It applies random scaling factors and adds Gaussian noise to simulate realistic variations.
"""

import pandas as pd
import numpy as np
import argparse
import os
from typing import List, Tuple


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect all numeric columns (int and float dtypes) in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        List[str]: List of numeric column names
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_columns


def apply_scale_and_jitter(df: pd.DataFrame, numeric_columns: List[str], 
                          scale_range: Tuple[float, float], jitter_factor: float) -> pd.DataFrame:
    """
    Apply scale and jitter augmentation to the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        numeric_columns (List[str]): List of numeric columns to augment
        scale_range (Tuple[float, float]): Range for random scaling (min, max)
        jitter_factor (float): Standard deviation for Gaussian noise
        
    Returns:
        pd.DataFrame: Augmented dataframe
    """
    # Create a copy of the original dataframe
    augmented_df = df.copy()
    
    # Apply scaling and jittering to each numeric column
    for column in numeric_columns:
        if column in df.columns:
            # Get the original values
            original_values = df[column].values
            
            # Generate random scaling factors for each row
            scale_factors = np.random.uniform(scale_range[0], scale_range[1], size=len(original_values))
            
            # Apply scaling
            scaled_values = original_values * scale_factors
            
            # Add Gaussian noise (jitter)
            # Calculate noise based on the standard deviation of the scaled values
            noise_std = np.std(scaled_values) * jitter_factor
            noise = np.random.normal(0, noise_std, size=len(scaled_values))
            
            # Apply the augmentation
            augmented_df[column] = scaled_values + noise
    
    return augmented_df


def save_augmented_data(df: pd.DataFrame, output_path: str) -> str:
    """
    Save the augmented dataframe to a CSV file.
    
    Args:
        df (pd.DataFrame): Augmented dataframe to save
        output_path (str): Path where to save the file
        
    Returns:
        str: Path to the saved file
    """
    df.to_csv(output_path, index=False)
    return output_path


def main():
    """
    Main function for command-line scale and jitter augmentation.
    """
    parser = argparse.ArgumentParser(description="Apply scale and jitter augmentation to numeric data")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--scale_min", type=float, default=0.8, help="Minimum scaling factor (default: 0.8)")
    parser.add_argument("--scale_max", type=float, default=1.2, help="Maximum scaling factor (default: 1.2)")
    parser.add_argument("--jitter", type=float, default=0.1, help="Jitter factor for noise (default: 0.1)")
    parser.add_argument("--columns", nargs='+', help="Specific columns to augment (optional)")
    
    args = parser.parse_args()
    
    try:
        # Read the input CSV file
        print(f"Reading CSV file: {args.input}")
        df = pd.read_csv(args.input)
        print(f"Successfully loaded dataframe with shape: {df.shape}")
        
        # Detect numeric columns or use specified columns
        if args.columns:
            numeric_columns = [col for col in args.columns if col in df.columns and df[col].dtype in [np.number]]
            if not numeric_columns:
                print(f"Error: None of the specified columns {args.columns} are numeric")
                return
        else:
            numeric_columns = detect_numeric_columns(df)
        
        if not numeric_columns:
            print("No numeric columns found. Nothing to augment.")
            return
        
        print(f"Augmenting {len(numeric_columns)} numeric columns: {numeric_columns}")
        
        # Set parameters
        scale_range = (args.scale_min, args.scale_max)
        jitter_factor = args.jitter
        
        # Apply scale and jitter augmentation
        augmented_df = apply_scale_and_jitter(df, numeric_columns, scale_range, jitter_factor)
        
        # Save the augmented dataframe
        save_augmented_data(augmented_df, args.output)
        
        print(f"Scale and jitter augmentation completed successfully!")
        print(f"Input shape: {df.shape}")
        print(f"Output shape: {augmented_df.shape}")
        print(f"Output file: {args.output}")
        print(f"Scale range: {scale_range}")
        print(f"Jitter factor: {jitter_factor}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the input file '{args.input}'")
    except pd.errors.EmptyDataError:
        print(f"Error: The input file is empty or corrupted")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    main()
