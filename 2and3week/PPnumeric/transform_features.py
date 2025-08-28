import pandas as pd
import argparse
from scipy.stats import boxcox
import numpy as np
import os

def select_csv_file():
    """Scans for CSV files in the current directory and prompts the user to select one."""
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the current directory.")
        return None

    print("Please select a CSV file to process:")
    for i, filename in enumerate(csv_files):
        print(f"{i + 1}: {filename}")

    while True:
        try:
            choice = int(input(f"Enter the number (1-{len(csv_files)}): ")) - 1
            if 0 <= choice < len(csv_files):
                return csv_files[choice]
            else:
                print(f"Invalid number. Please enter a number between 1 and {len(csv_files)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def transform_features(input_file, output_file, transform_type='log', columns=None):
    """
    Applies transformations to numeric features in a CSV file.

    :param input_file: Path to the input CSV file.
    :param output_file: Path to save the processed CSV file.
    :param transform_type: Type of transformation ('log' or 'boxcox').
    :param columns: List of specific columns to transform. If None, transforms all numeric columns.
    """
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return

    if columns:
        numeric_cols = columns
    else:
        numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        if transform_type == 'log':
            # Add 1 to handle zero values before log transformation
            if (df[col] <= 0).any():
                print(f"Warning: Column '{col}' contains non-positive values. Adding 1 to apply log transform.")
                df[col] = np.log1p(df[col])
            else:
                df[col] = np.log(df[col])
            print(f"Applied log transformation to column '{col}'.")

        elif transform_type == 'boxcox':
            # Box-Cox requires positive values
            if (df[col] <= 0).any():
                print(f"Error: Box-Cox transformation requires all values to be positive in column '{col}'.")
                continue
            df[col], _ = boxcox(df[col])
            print(f"Applied Box-Cox transformation to column '{col}'.")
        else:
            print(f"Error: Invalid transformation type '{transform_type}'. Choose from 'log' or 'boxcox'.")
            return

    df.to_csv(output_file, index=False)
    print(f"Processed file saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform numeric features in a CSV file.")
    parser.add_argument("--transform", default="log", choices=['log', 'boxcox'],
                        help="Transformation to use (default: log).")
    parser.add_argument("--columns", nargs='+', help="Specific columns to transform (optional).")

    args = parser.parse_args()
    
    input_file = select_csv_file()
    if input_file:
        output_file = f"{os.path.splitext(input_file)[0]}_transformed.csv"
        transform_features(input_file, output_file, args.transform, args.columns)
