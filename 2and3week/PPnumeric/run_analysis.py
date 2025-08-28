import subprocess
import sys
import os

def run_analysis():
    """
    Runs the full analysis pipeline:
    1. Executes comparison.py to generate a technical report.
    2. Pipes the report to summarize_changes.py.
    3. Prints only the final summary from the Gemini model.
    """
    # Ensure we are in the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("Starting analysis... Please follow the prompts to select a file.", file=sys.stderr)

    try:
        # --- Step 1: Run comparison.py and capture its output ---
        # We run it as a subprocess. The user will see the prompts (from stderr)
        # and can provide input. We capture the report data (from stdout).
        comparison_process = subprocess.run(
            [sys.executable, "comparison.py"],
            stdout=subprocess.PIPE,
            text=True,
            check=True  # This will raise an error if comparison.py fails
        )
        
        comparison_report = comparison_process.stdout
        
        if not comparison_report.strip():
            print("\nThe comparison script did not produce a report. Exiting.", file=sys.stderr)
            return

        print("\nComparison complete. Generating summary...", file=sys.stderr)

        # --- Step 2: Run summarize_changes.py with the captured report as input ---
        # We pipe the report to the stdin of the summarizer script.
        # Its output (the Gemini summary) will be printed directly to the console.
        summarize_process = subprocess.run(
            [sys.executable, "summarize_changes.py"],
            input=comparison_report,
            text=True,
            check=True
        )

    except FileNotFoundError as e:
        print(f"Error: Script not found. Make sure all scripts are in the same directory.", file=sys.stderr)
        print(e, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print("\nAn error occurred while running a script.", file=sys.stderr)
        # Print the error output from the failed script for debugging
        print("\n--- Error Details ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        print("---------------------", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    run_analysis()