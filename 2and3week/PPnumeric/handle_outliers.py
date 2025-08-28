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

def handle_outliers(input_file, output_file, columns=None, threshold=1.5):
    """
    Detects and removes outliers from a CSV or Excel file using the IQR method.

    :param input_file: Path to the input CSV or Excel file.
    :param output_file: Path to save the processed CSV file.
    :param columns: List of specific columns to process. If None, processes all numeric columns.
    :param threshold: IQR threshold for outlier detection.
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
        numeric_cols = columns
    else:
        numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            print(f"Detected {len(outliers)} outliers in column '{col}'.")
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            print(f"Removed outliers from column '{col}'.")
        else:
            print(f"No outliers detected in column '{col}'.")

    df.to_csv(output_file, index=False)
    print(f"Processed file saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handle outliers in a CSV or Excel file.")
    parser.add_argument("--input", required=True, help="Path to input CSV or Excel file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--columns", nargs='+', help="Specific columns to process (optional).")
    parser.add_argument("--threshold", type=float, default=1.5, help="IQR threshold (default: 1.5).")

    args = parser.parse_args()

    # Use provided input/output files instead of interactive selection
    handle_outliers(args.input, args.output, args.columns, args.threshold)
