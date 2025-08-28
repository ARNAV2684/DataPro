import os
import pandas as pd
import importlib
import sys

def find_csv_files():
    """Find all CSV files in the numeric folder."""
    numeric_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'numeric')
    
    # Create numeric folder if it doesn't exist
    if not os.path.exists(numeric_folder):
        os.makedirs(numeric_folder)
        print(f"Created numeric folder at: {numeric_folder}")
        print("Please add your CSV files to this folder and run again.")
        return []
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(numeric_folder) if f.endswith('.csv')]
    return csv_files, numeric_folder

def select_csv_file(csv_files, numeric_folder):
    """Let user select a CSV file."""
    if not csv_files:
        print("No CSV files found in the numeric folder.")
        return None
    
    print("\nAvailable CSV files:")
    for i, filename in enumerate(csv_files):
        print(f"{i+1}. {filename}")
    
    while True:
        try:
            choice = int(input(f"Select a file (1-{len(csv_files)}): ")) - 1
            if 0 <= choice < len(csv_files):
                selected_file = os.path.join(numeric_folder, csv_files[choice])
                return selected_file, csv_files[choice]
            else:
                print(f"Please enter a number between 1 and {len(csv_files)}.")
        except ValueError:
            print("Please enter a valid number.")

def get_available_eda_techniques():
    """Get available EDA techniques based on the Python files in the current directory."""
    current_dir = os.path.dirname(__file__)
    py_files = [f for f in os.listdir(current_dir) 
                if f.endswith('.py') and f != 'eda_manager.py' and f != '__init__.py']
    
    # Remove .py extension for display
    techniques = [os.path.splitext(f)[0] for f in py_files]
    return techniques, py_files

def select_eda_technique(techniques):
    """Let user select an EDA technique."""
    print("\nAvailable EDA techniques:")
    for i, technique in enumerate(techniques):
        # Format the technique name for better display
        display_name = technique.replace('_', ' ').title()
        print(f"{i+1}. {display_name}")
    
    while True:
        try:
            choice = int(input(f"Select a technique (1-{len(techniques)}): ")) - 1
            if 0 <= choice < len(techniques):
                return techniques[choice]
            else:
                print(f"Please enter a number between 1 and {len(techniques)}.")
        except ValueError:
            print("Please enter a valid number.")

def run_eda_technique(technique_name, filepath, filename):
    """Run the selected EDA technique on the selected file."""
    try:
        # Import the module dynamically
        module = importlib.import_module(technique_name)
        
        # Load the CSV file
        df = pd.read_csv(filepath)
        print(f"\nLoaded dataset: {filename} ({len(df)} rows x {len(df.columns)} columns)")
        
        # Run the analysis
        print(f"\nRunning {technique_name.replace('_', ' ').title()}...")
        
        # Call the run_analysis function from the imported module
        module.run_analysis(df, filename)
        
        print(f"\n✅ {technique_name.replace('_', ' ').title()} completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    """Main function to manage EDA workflow."""
    print("="*60)
    print("EDA MANAGER - DATA ANALYSIS TOOLKIT")
    print("="*60)
    
    # Find and select CSV file
    csv_files, numeric_folder = find_csv_files()
    if not csv_files:
        return
    
    selected_file, filename = select_csv_file(csv_files, numeric_folder)
    if not selected_file:
        return
    
    # Get and select EDA technique
    techniques, _ = get_available_eda_techniques()
    if not techniques:
        print("No EDA techniques found. Please add EDA scripts to the numeric_EDA folder.")
        return
    
    selected_technique = select_eda_technique(techniques)
    
    # Run the selected technique
    run_eda_technique(selected_technique, selected_file, filename)

if __name__ == "__main__":
    main()