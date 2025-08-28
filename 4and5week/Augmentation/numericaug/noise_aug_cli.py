"""
Noise-based Data Augmentation Script (Command Line Interface)
=============================================================
This script performs noise augmentation on numeric columns of CSV files.
It adds Gaussian noise to simulate realistic variations in the data.
"""

import pandas as pd
import numpy as np
import argparse
import os
from typing import List


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


def apply_noise_augmentation(df: pd.DataFrame, numeric_columns: List[str], 
                           noise_factor: float, augmentation_factor: int) -> pd.DataFrame:
    """
    Apply noise augmentation to the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        numeric_columns (List[str]): List of numeric columns to augment
        noise_factor (float): Factor controlling noise intensity
        augmentation_factor (int): Number of augmented samples per original
        
    Returns:
        pd.DataFrame: Augmented dataframe containing original + augmented samples
    """
    # Start with the original dataframe
    augmented_frames = [df.copy()]
    
    # Generate additional augmented copies
    for i in range(augmentation_factor):
        augmented_copy = df.copy()
        
        # Apply noise to each numeric column
        for column in numeric_columns:
            if column in df.columns:
                # Get the original values
                original_values = df[column].values
                
                # Calculate noise standard deviation based on the data's std and noise factor
                data_std = np.std(original_values)
                noise_std = data_std * noise_factor
                
                # Generate Gaussian noise
                noise = np.random.normal(0, noise_std, size=len(original_values))
                
                # Apply the noise
                augmented_copy[column] = original_values + noise
        
        augmented_frames.append(augmented_copy)
    
    # Concatenate all frames
    result_df = pd.concat(augmented_frames, ignore_index=True)
    
    return result_df


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
    Main function for command-line noise augmentation.
    """
    parser = argparse.ArgumentParser(description="Apply noise augmentation to numeric data")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--noise_factor", type=float, default=0.1, help="Factor controlling noise intensity (default: 0.1)")
    parser.add_argument("--augmentation_factor", type=int, default=2, help="Number of augmented samples per original (default: 2)")
    parser.add_argument("--columns", nargs='+', help="Specific columns to augment (optional)")
    
    args = parser.parse_args()
    
    try:
        # Read the input CSV file
        print(f"Reading CSV file: {args.input}")
        df = pd.read_csv(args.input)
        print(f"Successfully loaded dataframe with shape: {df.shape}")
        
        # Detect numeric columns or use specified columns
        if args.columns:
            numeric_columns = [col for col in args.columns if col in df.columns and 
                             df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
            if not numeric_columns:
                print(f"Error: None of the specified columns {args.columns} are numeric")
                return
        else:
            numeric_columns = detect_numeric_columns(df)
        
        if not numeric_columns:
            print("No numeric columns found. Nothing to augment.")
            return
        
        print(f"Augmenting {len(numeric_columns)} numeric columns: {numeric_columns}")
        
        # Validate parameters
        if args.noise_factor <= 0:
            print("Error: noise_factor must be greater than 0")
            return
        
        if args.augmentation_factor < 1:
            print("Error: augmentation_factor must be at least 1")
            return
        
        # Apply noise augmentation
        augmented_df = apply_noise_augmentation(df, numeric_columns, args.noise_factor, args.augmentation_factor)
        
        # Save the augmented dataframe
        save_augmented_data(augmented_df, args.output)
        
        print(f"Noise augmentation completed successfully!")
        print(f"Input shape: {df.shape}")
        print(f"Output shape: {augmented_df.shape}")
        print(f"Output file: {args.output}")
        print(f"Noise factor: {args.noise_factor}")
        print(f"Augmentation factor: {args.augmentation_factor}")
        print(f"Original samples: {len(df)}")
        print(f"Total samples after augmentation: {len(augmented_df)}")
        print(f"New samples generated: {len(augmented_df) - len(df)}")
        
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
