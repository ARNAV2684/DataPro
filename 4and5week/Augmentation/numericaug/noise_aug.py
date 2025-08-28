"""
Noise-based Data Augmentation Script
====================================
This script performs noise augmentation on numeric columns of CSV files.
It adds Gaussian noise to simulate realistic variations in the data.
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import List, Tuple


def find_csv_files(data_folder: str = "data") -> List[str]:
    """
    Search for all CSV files in the specified data folder.
    
    Args:
        data_folder (str): Path to the data folder
        
    Returns:
        List[str]: List of CSV file paths found in the folder
    """
    # Get the current working directory and construct the data folder path
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, data_folder)
    
    # Search for all CSV files in the data folder
    csv_pattern = os.path.join(data_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found in '{data_folder}' folder.")
        return []
    
    print(f"Found {len(csv_files)} CSV file(s) in '{data_folder}' folder:")
    return csv_files


def display_csv_options(csv_files: List[str]) -> None:
    """
    Display the available CSV files with numbered options.
    
    Args:
        csv_files (List[str]): List of CSV file paths
    """
    for i, file_path in enumerate(csv_files, 1):
        filename = os.path.basename(file_path)
        print(f"{i}. {filename}")


def get_user_selection(csv_files: List[str]) -> str:
    """
    Prompt the user to select a CSV file from the available options.
    
    Args:
        csv_files (List[str]): List of CSV file paths
        
    Returns:
        str: Path to the selected CSV file
    """
    while True:
        try:
            print("\nPlease select a CSV file to augment:")
            display_csv_options(csv_files)
            
            choice = int(input(f"\nEnter your choice (1-{len(csv_files)}): "))
            
            if 1 <= choice <= len(csv_files):
                selected_file = csv_files[choice - 1]
                filename = os.path.basename(selected_file)
                print(f"\nSelected: {filename}")
                return selected_file
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(csv_files)}.")
                
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect all numeric columns (int and float dtypes) in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        List[str]: List of numeric column names
    """
    # Select columns with numeric dtypes (int and float)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\nDetected {len(numeric_columns)} numeric column(s):")
    for col in numeric_columns:
        print(f"  - {col} (dtype: {df[col].dtype})")
    
    return numeric_columns


def add_gaussian_noise(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """
    Add Gaussian noise to numeric columns and create new noisy columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        numeric_columns (List[str]): List of numeric column names
        
    Returns:
        pd.DataFrame: Dataframe with added noisy columns
    """
    # Create a copy of the original dataframe
    augmented_df = df.copy()
    
    print(f"\nAdding Gaussian noise to numeric columns...")
    
    for col in numeric_columns:
        # Calculate the standard deviation of the column
        col_std = df[col].std()
        
        # Set noise standard deviation as 1% of the column's standard deviation
        noise_std = 0.01 * col_std
        
        # Generate Gaussian noise with mean=0 and calculated std deviation
        noise = np.random.normal(loc=0, scale=noise_std, size=len(df))
        
        # Create new column name for the noisy version
        noisy_col_name = f"{col}_noisy"
        
        # Add noise to the original column values
        augmented_df[noisy_col_name] = df[col] + noise
        
        print(f"  - Created '{noisy_col_name}' (noise std: {noise_std:.6f})")
    
    return augmented_df


def save_augmented_data(df: pd.DataFrame, output_filename: str = "numeric_aug_noise.csv") -> str:
    """
    Save the augmented dataframe to a CSV file in the data folder.
    
    Args:
        df (pd.DataFrame): Augmented dataframe
        output_filename (str): Name of the output CSV file
        
    Returns:
        str: Path to the saved file
    """
    # Construct the output file path
    output_path = os.path.join("data", output_filename)
    
    # Save the dataframe to CSV
    df.to_csv(output_path, index=False)
    
    print(f"\nAugmented data saved as: {output_path}")
    print(f"Original shape: {df.shape}")
    
    return output_path


def display_summary(original_df: pd.DataFrame, augmented_df: pd.DataFrame, 
                   numeric_columns: List[str]) -> None:
    """
    Display a summary of the augmentation process.
    
    Args:
        original_df (pd.DataFrame): Original dataframe
        augmented_df (pd.DataFrame): Augmented dataframe
        numeric_columns (List[str]): List of numeric columns that were augmented
    """
    print("\n" + "="*60)
    print("AUGMENTATION SUMMARY")
    print("="*60)
    
    print(f"Original columns: {len(original_df.columns)}")
    print(f"Augmented columns: {len(augmented_df.columns)}")
    print(f"New noisy columns added: {len(numeric_columns)}")
    
    print(f"\nNew columns created:")
    for col in numeric_columns:
        noisy_col = f"{col}_noisy"
        print(f"  - {noisy_col}")
    
    print(f"\nDataframe shape: {augmented_df.shape}")
    print("="*60)


def main():
    """
    Main function that orchestrates the noise augmentation process.
    """
    print("Noise-based Data Augmentation Tool")
    print("="*40)
    
    # Step 1: Search for CSV files in the data folder
    csv_files = find_csv_files("data")
    
    if not csv_files:
        print("Exiting: No CSV files found.")
        return
    
    # Step 2: Prompt user to select a CSV file
    selected_file = get_user_selection(csv_files)
    
    try:
        # Step 3: Read the selected CSV file
        print(f"\nReading CSV file: {os.path.basename(selected_file)}")
        df = pd.read_csv(selected_file)
        print(f"Successfully loaded dataframe with shape: {df.shape}")
        
        # Step 4: Detect numeric columns
        numeric_columns = detect_numeric_columns(df)
        
        if not numeric_columns:
            print("No numeric columns found. Nothing to augment.")
            return
        
        # Step 5: Add Gaussian noise to numeric columns
        augmented_df = add_gaussian_noise(df, numeric_columns)
        
        # Step 6: Save the augmented dataframe
        output_path = save_augmented_data(augmented_df)
        
        # Step 7: Display summary
        display_summary(df, augmented_df, numeric_columns)
        
        print(f"\nAugmentation completed successfully!")
        print(f"Output file: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{selected_file}'")
    except pd.errors.EmptyDataError:
        print(f"Error: The selected file is empty or corrupted")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    main()