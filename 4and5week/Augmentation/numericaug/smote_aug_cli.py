"""
SMOTE-based Data Augmentation Script (Command Line Interface)
============================================================
This script performs SMOTE (Synthetic Minority Oversampling Technique) augmentation
on numeric features to address class imbalance in classification datasets.
"""

import pandas as pd
import numpy as np
import argparse
import os
from typing import List, Tuple
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


def validate_target_column(df: pd.DataFrame, target_column: str) -> bool:
    """
    Validate if the target column is suitable for classification.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        
    Returns:
        bool: True if target is valid for classification
    """
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in dataset")
        return False
    
    # Check for missing values in target column
    if df[target_column].isna().any():
        print(f"Warning: Target column '{target_column}' contains missing values.")
        print("SMOTE requires complete data. Please handle missing values first.")
        return False
    
    # Check if target is continuous (likely regression, not classification)
    target_values = df[target_column]
    unique_values = target_values.nunique()
    total_values = len(target_values)
    
    # If more than 50% of values are unique, it's likely continuous
    uniqueness_ratio = unique_values / total_values
    if uniqueness_ratio > 0.5:
        print(f"Error: Target column '{target_column}' appears to be continuous (regression target).")
        print(f"SMOTE is designed for classification with discrete classes.")
        print(f"Unique values: {unique_values} out of {total_values} samples ({uniqueness_ratio:.2%} unique)")
        print(f"Consider:")
        print(f"  1. Using a categorical target column for classification")
        print(f"  2. Converting continuous target to classes (e.g., High/Medium/Low)")
        print(f"  3. Using noise augmentation instead for regression tasks")
        return False
    
    # Check number of unique classes
    if unique_values < 2:
        print(f"Error: Target column '{target_column}' has only {unique_values} unique class.")
        print("Classification requires at least 2 classes.")
        return False
    
    # Check if we have enough samples per class for SMOTE
    class_counts = target_values.value_counts()
    min_class_count = class_counts.min()
    if min_class_count < 2:
        print(f"Error: Smallest class has only {min_class_count} sample(s).")
        print("SMOTE requires at least 2 samples per class.")
        return False
    
    print(f"Target column validation passed:")
    print(f"  - Classes: {unique_values}")
    print(f"  - Samples per class: {dict(class_counts)}")
    print(f"  - Smallest class: {min_class_count} samples")
    return True


def prepare_features_and_target(df: pd.DataFrame, numeric_columns: List[str], 
                              target_column: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare features and target for SMOTE application.
    
    Args:
        df (pd.DataFrame): Input dataframe
        numeric_columns (List[str]): List of numeric columns
        target_column (str): Name of target column
        
    Returns:
        Tuple containing features, target, and feature column names
    """
    # Remove target column from features if it's numeric
    feature_columns = [col for col in numeric_columns if col != target_column]
    
    if not feature_columns:
        raise ValueError("No numeric feature columns available for SMOTE")
    
    # Extract features and target
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    # Check for missing values in features
    if X.isna().any().any():
        print("Warning: Missing values detected in features. Removing rows with missing values.")
        # Get indices where both X and y have no missing values
        complete_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[complete_mask]
        y = y[complete_mask]
        print(f"Rows after removing missing values: {len(X)}")
        
        if len(X) == 0:
            raise ValueError("No complete rows available after removing missing values")
    
    # Encode target if it's not numeric
    label_encoder = None
    if not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), name=y.name)
    
    print(f"Features prepared for SMOTE:")
    print(f"  - Numeric features: {len(feature_columns)} columns")
    print(f"  - Target column: {target_column}")
    print(f"  - Total samples: {len(X)}")
    
    return X, y, feature_columns


def apply_smote(X: pd.DataFrame, y: pd.Series, k_neighbors: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to generate synthetic samples for minority classes.
    
    Args:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Target series
        k_neighbors (int): Number of nearest neighbors for SMOTE
        
    Returns:
        Tuple containing resampled features and target
    """
    print(f"\nApplying SMOTE...")
    
    # Display original class distribution
    original_distribution = Counter(y)
    print(f"Original class distribution:")
    for class_label, count in original_distribution.items():
        print(f"  - {class_label}: {count} samples")
    
    # Validate k_neighbors parameter
    min_class_size = min(original_distribution.values())
    if k_neighbors >= min_class_size:
        # Adjust k_neighbors to be safe
        k_neighbors = max(1, min_class_size - 1)
        print(f"Warning: k_neighbors adjusted to {k_neighbors} (must be less than smallest class size: {min_class_size})")
    
    # Check if SMOTE is needed
    if len(set(original_distribution.values())) == 1:
        print("All classes have equal size. No SMOTE needed.")
        return X, y
    
    # Initialize SMOTE with specified parameters
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    
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
        
    except ValueError as e:
        if "k_neighbors" in str(e).lower():
            print(f"Error: k_neighbors parameter issue: {str(e)}")
            print(f"Try reducing k_neighbors value. Current value: {k_neighbors}")
            print(f"Minimum class size: {min_class_size}")
        else:
            print(f"Error applying SMOTE: {str(e)}")
        raise
    except Exception as e:
        print(f"Error applying SMOTE: {str(e)}")
        print(f"Data shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"k_neighbors: {k_neighbors}")
        raise


def combine_features_and_target(X: pd.DataFrame, y: pd.Series, 
                              original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine resampled features and target back into a single dataframe.
    
    Args:
        X (pd.DataFrame): Resampled features
        y (pd.Series): Resampled target
        original_df (pd.DataFrame): Original dataframe for reference
        
    Returns:
        pd.DataFrame: Combined resampled dataframe
    """
    # Start with the resampled features
    result_df = X.copy()
    
    # Add the target column
    result_df[y.name] = y
    
    # Add any non-numeric columns from original (using first occurrence)
    non_numeric_cols = [col for col in original_df.columns 
                       if col not in result_df.columns]
    
    for col in non_numeric_cols:
        # For non-numeric columns, repeat the first value for all new samples
        original_values = original_df[col].dropna()
        if len(original_values) > 0:
            # For original samples, keep original values
            original_length = len(original_df)
            if len(result_df) >= original_length:
                # Fill with original values for original samples
                result_df[col] = [original_df[col].iloc[i % len(original_df)] 
                                for i in range(len(result_df))]
    
    return result_df


def main():
    """
    Main function for command-line SMOTE augmentation.
    """
    parser = argparse.ArgumentParser(description="Apply SMOTE augmentation for class imbalance")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--target", required=True, help="Name of the target column for classification")
    parser.add_argument("--k_neighbors", type=int, default=5, help="Number of nearest neighbors for SMOTE (default: 5)")
    parser.add_argument("--columns", nargs='+', help="Specific feature columns to use (optional)")
    
    args = parser.parse_args()
    
    try:
        # Read the input CSV file
        print(f"Reading CSV file: {args.input}")
        df = pd.read_csv(args.input)
        print(f"Successfully loaded dataframe with shape: {df.shape}")
        
        # Validate target column
        if not validate_target_column(df, args.target):
            return
        
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
        
        print(f"Using {len(numeric_columns)} numeric feature columns: {numeric_columns}")
        
        # Prepare features and target for SMOTE
        X, y, feature_columns = prepare_features_and_target(df, numeric_columns, args.target)
        
        # Apply SMOTE augmentation
        X_resampled, y_resampled = apply_smote(X, y, args.k_neighbors)
        
        # Combine features and target back into dataframe
        augmented_df = combine_features_and_target(X_resampled, y_resampled, df)
        
        # Save the augmented dataframe
        augmented_df.to_csv(args.output, index=False)
        
        print(f"\nSMOTE augmentation completed successfully!")
        print(f"Input shape: {df.shape}")
        print(f"Output shape: {augmented_df.shape}")
        print(f"Output file: {args.output}")
        print(f"Target column: {args.target}")
        print(f"K-neighbors: {args.k_neighbors}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the input file '{args.input}'")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The input file is empty or corrupted")
        exit(1)
    except ValueError as e:
        print(f"Error: {str(e)}")
        exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    main()
