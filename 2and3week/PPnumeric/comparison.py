import pandas as pd
import os
import sys

def select_csv_file(prompt_message, file_list):
    """Prompts the user to select a file from a given list."""
    print(prompt_message, file=sys.stderr)
    for i, filename in enumerate(file_list):
        print(f"{i + 1}: {filename}", file=sys.stderr)

    while True:
        try:
            choice = int(input(f"Enter the number (1-{len(file_list)}): ")) - 1
            if 0 <= choice < len(file_list):
                return file_list[choice]
            else:
                print(f"Invalid number. Please enter a number between 1 and {len(file_list)}.", file=sys.stderr)
        except ValueError:
            print("Invalid input. Please enter a number.", file=sys.stderr)

def find_processed_file(original_filename):
    """Finds processed files corresponding to the original file."""
    base_name = os.path.splitext(original_filename)[0]
    # Ensure we don't select another original file if it has a similar name
    base_name_with_underscore = base_name + "_"
    possible_suffixes = ['missing_handled', 'outliers_handled', 'scaled', 'transformed']
    
    processed_files = [
        f for f in os.listdir('.') 
        if f.startswith(base_name_with_underscore) and any(f.endswith(suffix + '.csv') for suffix in possible_suffixes)
    ]
    
    if not processed_files:
        return None
    if len(processed_files) == 1:
        return processed_files[0]
    
    return select_csv_file(
        "Multiple processed files found. Please select which one to compare:",
        processed_files
    )

def compare_files(original_file, processed_file):
    """Compares two CSV files and prints a summary of their differences."""
    print("\n" + "="*60)
    print(f"Comparing '{original_file}' with '{processed_file}'")
    print("="*60)

    try:
        df_original = pd.read_csv(original_file)
        df_processed = pd.read_csv(processed_file)
    except FileNotFoundError as e:
        print(f"Error: Could not read file. {e}", file=sys.stderr)
        return

    # --- Basic Information ---
    print("\n--- File Information ---")
    print(f"Original file shape:    {df_original.shape}")
    print(f"Processed file shape:   {df_processed.shape}")
    print(f"Original missing values:  {df_original.isnull().sum().sum()}")
    print(f"Processed missing values: {df_processed.isnull().sum().sum()}")

    # --- Column-level Comparison ---
    print("\n--- Column Comparison ---")
    original_cols = set(df_original.columns)
    processed_cols = set(df_processed.columns)

    if original_cols != processed_cols:
        print("Columns have been added or removed.")
        print(f"Columns only in original: {original_cols - processed_cols}")
        print(f"Columns only in processed: {processed_cols - original_cols}")
    
    common_columns = list(original_cols & processed_cols)
    modified_cols = []
    for col in common_columns:
        # Use try-except to handle cases where columns might not be comparable
        try:
            if not df_original[col].equals(df_processed[col]):
                modified_cols.append(col)
        except:
            # This can happen with mixed types, consider it modified
            modified_cols.append(col)

    if not modified_cols:
        print("No columns were modified.")
    else:
        print(f"The following columns were modified: {modified_cols}")
        for col in modified_cols:
            # --- Numeric Column Analysis ---
            if pd.api.types.is_numeric_dtype(df_original[col]) and pd.api.types.is_numeric_dtype(df_processed[col]):
                print("\n" + "-"*20 + f" Numeric Stats for '{col}' " + "-"*20)
                stats = pd.DataFrame({
                    'Original': df_original[col].describe(),
                    'Processed': df_processed[col].describe()
                })
                print(stats.round(2))
            # --- Text/Categorical Column Analysis ---
            else:
                print("\n" + "-"*20 + f" Text/Category Changes for '{col}' " + "-"*20)
                original_unique = df_original[col].nunique()
                processed_unique = df_processed[col].nunique()
                print(f"Original unique values: {original_unique}")
                print(f"Processed unique values: {processed_unique}")

    print("\n" + "="*60)
    print("Comparison complete.")
    print("="*60)


if __name__ == "__main__":
    possible_suffixes = ['_missing_handled.csv', '_outliers_handled.csv', '_scaled.csv', '_transformed.csv']
    all_csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    # Filter to find only original files (those that don't have the processed suffixes)
    original_files = [f for f in all_csv_files if not any(f.endswith(suffix) for suffix in possible_suffixes)]

    if not original_files:
        print("No original CSV files found to validate.", file=sys.stderr)
        print("An original file is one that does not end with '_missing_handled.csv', '_outliers_handled.csv', etc.", file=sys.stderr)
    else:
        original_file = select_csv_file(
            "Please select the ORIGINAL CSV file to validate:",
            original_files
        )
        
        if original_file:
            processed_file = find_processed_file(original_file)
            if processed_file:
                compare_files(original_file, processed_file)
            else:
                print(f"No corresponding processed file found for '{original_file}'.", file=sys.stderr)
