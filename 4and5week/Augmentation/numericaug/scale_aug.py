"""
Scale and Jitter-based Data Augmentation Script
===============================================
This script performs scale and jitter augmentation on numeric columns of CSV files.
It applies random scaling factors and adds Gaussian noise to simulate realistic variations.
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


def apply_scale_and_jitter(df: pd.DataFrame, numeric_columns: List[str], 
                          scale_range: Tuple[float, float] = (0.95, 1.05),
                          jitter_factor: float = 0.01) -> pd.DataFrame:
    """
    Apply random scaling and Gaussian jitter to numeric columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        numeric_columns (List[str]): List of numeric column names
        scale_range (Tuple[float, float]): Range for random scaling factors (min, max)
        jitter_factor (float): Factor to determine jitter noise std (as fraction of column std)
        
    Returns:
        pd.DataFrame: Dataframe with added scaled and jittered columns
    """
    # Create a copy of the original dataframe
    augmented_df = df.copy()
    
    print(f"\nApplying scale and jitter augmentation to numeric columns...")
    print(f"  - Scale range: {scale_range[0]:.3f} to {scale_range[1]:.3f}")
    print(f"  - Jitter factor: {jitter_factor} * column standard deviation")
    
    for col in numeric_columns:
        # Calculate the standard deviation of the column for jitter
        col_std = df[col].std()
        
        # Set jitter noise standard deviation
        jitter_std = jitter_factor * col_std
        
        # Generate random scaling factors uniformly sampled from the specified range
        scaling_factors = np.random.uniform(
            low=scale_range[0], 
            high=scale_range[1], 
            size=len(df)
        )
        
        # Generate Gaussian jitter noise
        jitter_noise = np.random.normal(
            loc=0, 
            scale=jitter_std, 
            size=len(df)
        )
        
        # Create new column name for the scaled and jittered version
        scaled_jittered_col_name = f"{col}_scaled_jittered"
        
        # Apply scaling and jitter: (original_value * scaling_factor) + jitter_noise
        scaled_values = df[col] * scaling_factors
        final_values = scaled_values + jitter_noise
        
        # Add the new column to the dataframe
        augmented_df[scaled_jittered_col_name] = final_values
        
        # Calculate statistics for reporting
        original_mean = df[col].mean()
        original_std = df[col].std()
        new_mean = final_values.mean()
        new_std = final_values.std()
        
        print(f"  - Created '{scaled_jittered_col_name}':")
        print(f"    * Original - Mean: {original_mean:.6f}, Std: {original_std:.6f}")
        print(f"    * Augmented - Mean: {new_mean:.6f}, Std: {new_std:.6f}")
        print(f"    * Jitter noise std: {jitter_std:.6f}")
    
    return augmented_df


def get_augmentation_parameters() -> Tuple[Tuple[float, float], float]:
    """
    Prompt the user to specify augmentation parameters or use defaults.
    
    Returns:
        Tuple containing scale range and jitter factor
    """
    print("\nAugmentation Parameters:")
    print("You can customize the augmentation parameters or use defaults.")
    
    use_defaults = input("Use default parameters? (y/n) [default: y]: ").lower().strip()
    
    if use_defaults == 'n':
        print("\nCustom parameter setup:")
        
        # Get scaling range
        while True:
            try:
                scale_min = float(input("Enter minimum scaling factor [default: 0.95]: ") or "0.95")
                scale_max = float(input("Enter maximum scaling factor [default: 1.05]: ") or "1.05")
                
                if scale_min < scale_max and scale_min > 0:
                    scale_range = (scale_min, scale_max)
                    break
                else:
                    print("Invalid range. Minimum must be positive and less than maximum.")
            except ValueError:
                print("Invalid input. Please enter numeric values.")
        
        # Get jitter factor
        while True:
            try:
                jitter_factor = float(input("Enter jitter factor (as fraction of std) [default: 0.01]: ") or "0.01")
                
                if jitter_factor >= 0:
                    break
                else:
                    print("Jitter factor must be non-negative.")
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
        
        print(f"\nUsing custom parameters:")
        print(f"  - Scale range: {scale_range[0]:.3f} to {scale_range[1]:.3f}")
        print(f"  - Jitter factor: {jitter_factor}")
        
    else:
        # Use default parameters
        scale_range = (0.95, 1.05)
        jitter_factor = 0.01
        
        print(f"\nUsing default parameters:")
        print(f"  - Scale range: {scale_range[0]:.3f} to {scale_range[1]:.3f}")
        print(f"  - Jitter factor: {jitter_factor}")
    
    return scale_range, jitter_factor


def save_augmented_data(df: pd.DataFrame, output_filename: str = "numeric_aug_scaled_jittered.csv") -> str:
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
    print(f"Shape: {df.shape}")
    
    return output_path


def analyze_augmentation_quality(original_df: pd.DataFrame, augmented_df: pd.DataFrame, 
                               numeric_columns: List[str]) -> None:
    """
    Analyze the quality of augmentation by comparing original and augmented columns.
    
    Args:
        original_df (pd.DataFrame): Original dataframe
        augmented_df (pd.DataFrame): Augmented dataframe
        numeric_columns (List[str]): List of numeric columns that were augmented
    """
    print(f"\nAugmentation Quality Analysis:")
    print("-" * 50)
    
    for col in numeric_columns:
        augmented_col = f"{col}_scaled_jittered"
        
        if augmented_col in augmented_df.columns:
            # Calculate correlation between original and augmented columns
            correlation = original_df[col].corr(augmented_df[augmented_col])
            
            # Calculate percentage change in mean and std
            orig_mean = original_df[col].mean()
            aug_mean = augmented_df[augmented_col].mean()
            mean_change_pct = ((aug_mean - orig_mean) / orig_mean) * 100 if orig_mean != 0 else 0
            
            orig_std = original_df[col].std()
            aug_std = augmented_df[augmented_col].std()
            std_change_pct = ((aug_std - orig_std) / orig_std) * 100 if orig_std != 0 else 0
            
            print(f"\n{col} -> {augmented_col}:")
            print(f"  - Correlation with original: {correlation:.6f}")
            print(f"  - Mean change: {mean_change_pct:+.3f}%")
            print(f"  - Std deviation change: {std_change_pct:+.3f}%")
            
            # Quality assessment
            if correlation > 0.95:
                quality = "Excellent"
            elif correlation > 0.90:
                quality = "Good"
            elif correlation > 0.80:
                quality = "Fair"
            else:
                quality = "Poor"
            
            print(f"  - Quality assessment: {quality}")


def display_summary(original_df: pd.DataFrame, augmented_df: pd.DataFrame, 
                   numeric_columns: List[str], scale_range: Tuple[float, float], 
                   jitter_factor: float) -> None:
    """
    Display a summary of the scale and jitter augmentation process.
    
    Args:
        original_df (pd.DataFrame): Original dataframe
        augmented_df (pd.DataFrame): Augmented dataframe
        numeric_columns (List[str]): List of numeric columns that were augmented
        scale_range (Tuple[float, float]): Scale range used
        jitter_factor (float): Jitter factor used
    """
    print("\n" + "="*70)
    print("SCALE AND JITTER AUGMENTATION SUMMARY")
    print("="*70)
    
    print(f"Parameters used:")
    print(f"  - Scale range: {scale_range[0]:.3f} to {scale_range[1]:.3f}")
    print(f"  - Jitter factor: {jitter_factor}")
    
    print(f"\nDataframe information:")
    print(f"  - Original columns: {len(original_df.columns)}")
    print(f"  - Augmented columns: {len(augmented_df.columns)}")
    print(f"  - New columns added: {len(numeric_columns)}")
    print(f"  - Shape: {augmented_df.shape}")
    
    print(f"\nNew columns created:")
    for col in numeric_columns:
        scaled_jittered_col = f"{col}_scaled_jittered"
        print(f"  - {scaled_jittered_col}")
    
    print(f"\nAugmentation technique:")
    print(f"  - Each value is multiplied by a random scaling factor")
    print(f"  - Gaussian jitter noise is added to scaled values")
    print(f"  - Preserves approximate distribution while adding variation")
    
    print("="*70)


def main():
    """
    Main function that orchestrates the scale and jitter augmentation process.
    """
    print("Scale and Jitter Data Augmentation Tool")
    print("="*45)
    
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
        
        # Step 5: Get augmentation parameters
        scale_range, jitter_factor = get_augmentation_parameters()
        
        # Step 6: Apply scale and jitter augmentation
        augmented_df = apply_scale_and_jitter(df, numeric_columns, scale_range, jitter_factor)
        
        # Step 7: Analyze augmentation quality
        analyze_augmentation_quality(df, augmented_df, numeric_columns)
        
        # Step 8: Save the augmented dataframe
        output_path = save_augmented_data(augmented_df)
        
        # Step 9: Display summary
        display_summary(df, augmented_df, numeric_columns, scale_range, jitter_factor)
        
        print(f"\nScale and jitter augmentation completed successfully!")
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