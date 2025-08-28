"""
SMOTE-based Data Augmentation Script
===================================
This script performs SMOTE (Synthetic Minority Oversampling Technique) augmentation
on numeric features to address class imbalance in classification datasets.
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import List, Tuple, Optional
from collections import Counter

try:
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import LabelEncoder
except ImportError as e:
    print("Error: Required libraries not found.")
    print("Please install the required packages:")
    print("pip install imbalanced-learn scikit-learn")
    print(f"Missing: {e}")
    exit(1)


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


def display_column_options(columns: List[str]) -> None:
    """
    Display the available columns with numbered options.
    
    Args:
        columns (List[str]): List of column names
    """
    for i, col in enumerate(columns, 1):
        print(f"{i}. {col}")


def get_target_column_selection(df: pd.DataFrame) -> str:
    """
    Prompt the user to select the target column (class labels) for classification.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        str: Name of the selected target column
    """
    all_columns = df.columns.tolist()
    
    while True:
        try:
            print("\nPlease select the target column (class labels) for classification:")
            display_column_options(all_columns)
            
            choice = int(input(f"\nEnter your choice (1-{len(all_columns)}): "))
            
            if 1 <= choice <= len(all_columns):
                target_column = all_columns[choice - 1]
                print(f"\nSelected target column: {target_column}")
                
                # Display class distribution
                class_counts = df[target_column].value_counts()
                print(f"\nClass distribution in '{target_column}':")
                for class_label, count in class_counts.items():
                    print(f"  - {class_label}: {count} samples")
                
                return target_column
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(all_columns)}.")
                
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def validate_target_column(df: pd.DataFrame, target_column: str) -> bool:
    """
    Validate if the target column is suitable for classification.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        
    Returns:
        bool: True if target is valid for classification
    """
    # Check for missing values in target column
    if df[target_column].isna().any():
        print(f"Warning: Target column '{target_column}' contains missing values.")
        print("SMOTE requires complete data. Please handle missing values first.")
        return False
    
    # Check number of unique classes
    unique_classes = df[target_column].nunique()
    if unique_classes < 2:
        print(f"Error: Target column '{target_column}' has only {unique_classes} unique class.")
        print("Classification requires at least 2 classes.")
        return False
    
    # Check if target has too many unique values (might be continuous)
    total_samples = len(df)
    if unique_classes > total_samples * 0.5:
        print(f"Warning: Target column '{target_column}' has {unique_classes} unique values.")
        print("This might be a continuous variable rather than categorical classes.")
        
        confirm = input("Continue anyway? (y/n): ").lower().strip()
        return confirm == 'y'
    
    print(f"Target column validation passed: {unique_classes} classes detected.")
    return True


def prepare_features_and_target(df: pd.DataFrame, numeric_columns: List[str], 
                              target_column: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare features and target for SMOTE application.
    
    Args:
        df (pd.DataFrame): Input dataframe
        numeric_columns (List[str]): List of numeric column names
        target_column (str): Name of the target column
        
    Returns:
        Tuple containing features dataframe, target series, and other column names
    """
    # Remove target column from numeric columns if present
    feature_columns = [col for col in numeric_columns if col != target_column]
    
    # Get non-numeric, non-target columns
    other_columns = [col for col in df.columns if col not in numeric_columns and col != target_column]
    
    # Prepare features (only numeric columns excluding target)
    X = df[feature_columns].copy()
    
    # Prepare target
    y = df[target_column].copy()
    
    print(f"\nFeature preparation:")
    print(f"  - Numeric features for SMOTE: {len(feature_columns)} columns")
    print(f"  - Target column: {target_column}")
    print(f"  - Other columns (unchanged): {len(other_columns)} columns")
    
    return X, y, other_columns


def apply_smote(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to generate synthetic samples for minority classes.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Target series
        
    Returns:
        Tuple containing resampled features and target
    """
    print(f"\nApplying SMOTE...")
    
    # Display original class distribution
    original_distribution = Counter(y)
    print(f"Original class distribution:")
    for class_label, count in original_distribution.items():
        print(f"  - {class_label}: {count} samples")
    
    # Initialize SMOTE with default parameters
    # random_state for reproducible results
    smote = SMOTE(random_state=42)
    
    try:
        # Apply SMOTE
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Display new class distribution
        new_distribution = Counter(y_resampled)
        print(f"\nAfter SMOTE class distribution:")
        for class_label, count in new_distribution.items():
            print(f"  - {class_label}: {count} samples")
        
        # Convert back to pandas objects with proper column names
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)
        
        print(f"\nSMOTE completed successfully!")
        print(f"Original samples: {len(X)}")
        print(f"Resampled samples: {len(X_resampled)}")
        print(f"Synthetic samples generated: {len(X_resampled) - len(X)}")
        
        return X_resampled, y_resampled
        
    except Exception as e:
        print(f"Error applying SMOTE: {str(e)}")
        print("This might be due to:")
        print("- Insufficient samples for minority classes")
        print("- High dimensionality relative to sample size")
        print("- Invalid data types")
        raise


def reconstruct_dataframe(X_resampled: pd.DataFrame, y_resampled: pd.Series,
                         original_df: pd.DataFrame, other_columns: List[str],
                         target_column: str) -> pd.DataFrame:
    """
    Reconstruct the final dataframe by combining resampled data with original non-numeric columns.
    
    Args:
        X_resampled (pd.DataFrame): Resampled features
        y_resampled (pd.Series): Resampled target
        original_df (pd.DataFrame): Original dataframe
        other_columns (List[str]): Names of non-numeric, non-target columns
        target_column (str): Name of the target column
        
    Returns:
        pd.DataFrame: Final balanced dataframe
    """
    print(f"\nReconstructing dataframe...")
    
    # Start with resampled features
    final_df = X_resampled.copy()
    
    # Add resampled target column
    final_df[target_column] = y_resampled
    
    # Handle other (non-numeric, non-target) columns
    if other_columns:
        print(f"Handling {len(other_columns)} non-numeric columns...")
        
        # For original samples, keep original values
        original_length = len(original_df)
        
        for col in other_columns:
            # Create a new column filled with NaN
            final_df[col] = np.nan
            
            # Fill original samples with their original values
            final_df.loc[:original_length-1, col] = original_df[col].values
            
            # For synthetic samples, use forward fill from the last original sample
            # or use the most common value
            if len(final_df) > original_length:
                # Use the most frequent value for categorical columns
                if original_df[col].dtype == 'object':
                    most_common = original_df[col].mode()
                    if len(most_common) > 0:
                        final_df.loc[original_length:, col] = most_common.iloc[0]
                else:
                    # For numeric non-feature columns, use median
                    median_val = original_df[col].median()
                    final_df.loc[original_length:, col] = median_val
        
        print(f"Non-numeric columns filled for synthetic samples.")
    
    return final_df


def save_balanced_data(df: pd.DataFrame, output_filename: str = "numeric_aug_smote.csv") -> str:
    """
    Save the balanced dataframe to a CSV file in the data folder.
    
    Args:
        df (pd.DataFrame): Balanced dataframe
        output_filename (str): Name of the output CSV file
        
    Returns:
        str: Path to the saved file
    """
    # Construct the output file path
    output_path = os.path.join("data", output_filename)
    
    # Save the dataframe to CSV
    df.to_csv(output_path, index=False)
    
    print(f"\nBalanced data saved as: {output_path}")
    print(f"Final shape: {df.shape}")
    
    return output_path


def display_summary(original_df: pd.DataFrame, balanced_df: pd.DataFrame,
                   target_column: str, feature_columns: List[str]) -> None:
    """
    Display a summary of the SMOTE augmentation process.
    
    Args:
        original_df (pd.DataFrame): Original dataframe
        balanced_df (pd.DataFrame): Balanced dataframe
        target_column (str): Name of the target column
        feature_columns (List[str]): List of feature columns used in SMOTE
    """
    print("\n" + "="*60)
    print("SMOTE AUGMENTATION SUMMARY")
    print("="*60)
    
    print(f"Target column: {target_column}")
    print(f"Features used for SMOTE: {len(feature_columns)}")
    
    # Original class distribution
    original_dist = original_df[target_column].value_counts().sort_index()
    print(f"\nOriginal class distribution:")
    for class_label, count in original_dist.items():
        print(f"  - {class_label}: {count} samples")
    
    # Balanced class distribution
    balanced_dist = balanced_df[target_column].value_counts().sort_index()
    print(f"\nBalanced class distribution:")
    for class_label, count in balanced_dist.items():
        print(f"  - {class_label}: {count} samples")
    
    print(f"\nDataframe shape:")
    print(f"  - Original: {original_df.shape}")
    print(f"  - Balanced: {balanced_df.shape}")
    print(f"  - Synthetic samples added: {len(balanced_df) - len(original_df)}")
    
    print("="*60)


def main():
    """
    Main function that orchestrates the SMOTE augmentation process.
    """
    print("SMOTE-based Data Augmentation Tool")
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
        
        if len(numeric_columns) < 2:  # Need at least 1 feature + 1 target
            print("Error: Need at least 2 numeric columns (1 for features, 1 for target).")
            return
        
        # Step 5: Prompt user to select target column
        target_column = get_target_column_selection(df)
        
        # Step 6: Validate target column
        if not validate_target_column(df, target_column):
            print("Exiting: Target column validation failed.")
            return
        
        # Step 7: Prepare features and target
        X, y, other_columns = prepare_features_and_target(df, numeric_columns, target_column)
        
        if X.empty:
            print("Error: No numeric features available for SMOTE after removing target column.")
            return
        
        # Step 8: Apply SMOTE
        X_resampled, y_resampled = apply_smote(X, y)
        
        # Step 9: Reconstruct the complete dataframe
        balanced_df = reconstruct_dataframe(X_resampled, y_resampled, df, other_columns, target_column)
        
        # Step 10: Save the balanced dataframe
        output_path = save_balanced_data(balanced_df)
        
        # Step 11: Display summary
        display_summary(df, balanced_df, target_column, X.columns.tolist())
        
        print(f"\nSMOTE augmentation completed successfully!")
        print(f"Output file: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{selected_file}'")
    except pd.errors.EmptyDataError:
        print(f"Error: The selected file is empty or corrupted")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")
        print("Please check your data and try again.")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    main()