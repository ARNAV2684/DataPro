import pandas as pd
import argparse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

def scale_numeric_features(input_file, output_file, scaler_type='minmax', columns=None):
    """
    Scales numeric features in a CSV file using MinMaxScaler or StandardScaler.

    :param input_file: Path to the input CSV file.
    :param output_file: Path to save the processed CSV file.
    :param scaler_type: Type of scaler ('minmax' or 'standard').
    :param columns: List of specific columns to scale. If None, scales all numeric columns.
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

    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
        print("Using MinMaxScaler.")
    elif scaler_type == 'standard':
        scaler = StandardScaler()
        print("Using StandardScaler.")
    else:
        print(f"Error: Invalid scaler type '{scaler_type}'. Choose from 'minmax' or 'standard'.")
        return

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f"Scaled columns: {list(numeric_cols)}")

    df.to_csv(output_file, index=False)
    print(f"Processed file saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale numeric features in a CSV file.")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--scaler", default="minmax", choices=['minmax', 'standard'],
                        help="Scaler to use (default: minmax).")
    parser.add_argument("--columns", nargs='+', help="Specific columns to scale (optional).")

    args = parser.parse_args()
    
    # Use provided input/output files instead of interactive selection
    scale_numeric_features(args.input, args.output, args.scaler, args.columns)
