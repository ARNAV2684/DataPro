import pandas as pd
import argparse
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

def handle_missing_values(input_file, output_file, strategy='mean', columns=None):
    """
    Handles missing values in a CSV or Excel file using different strategies.

    :param input_file: Path to the input CSV or Excel file.
    :param output_file: Path to save the processed CSV file.
    :param strategy: Imputation strategy ('mean', 'median', 'mode').
    :param columns: List of specific columns to apply imputation. If None, applies to all numeric columns.
    """
    try:
        # Check file extension and read accordingly
        if input_file.lower().endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(input_file)
        else:
            print(f"Error: Unsupported file format. Please use CSV or Excel files.")
            return
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if columns:
        # If specific columns are provided, use them
        numeric_cols_to_process = columns
    else:
        # Otherwise, select all numeric columns
        numeric_cols_to_process = df.select_dtypes(include=['number']).columns

    for col in numeric_cols_to_process:
        if df[col].isnull().sum() > 0:
            if strategy == 'mean':
                fill_value = df[col].mean()
                print(f"Filled missing values in column '{col}' with mean ({fill_value:.2f}).")
            elif strategy == 'median':
                fill_value = df[col].median()
                print(f"Filled missing values in column '{col}' with median ({fill_value:.2f}).")
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
                print(f"Filled missing values in column '{col}' with mode ({fill_value}).")
            else:
                print(f"Error: Invalid strategy '{strategy}'. Choose from 'mean', 'median', or 'mode'.")
                return
            df[col].fillna(fill_value, inplace=True)

    df.to_csv(output_file, index=False)
    print(f"Processed file saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handle missing values in a CSV or Excel file.")
    parser.add_argument("--input", required=True, help="Path to input CSV or Excel file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--strategy", default="mean", choices=['mean', 'median', 'mode'],
                        help="Imputation strategy to use (default: mean).")
    parser.add_argument("--columns", nargs='+', help="Specific columns to process (optional).")

    args = parser.parse_args()
    
    # Use provided input/output files instead of interactive selection
    handle_missing_values(args.input, args.output, args.strategy, args.columns)
