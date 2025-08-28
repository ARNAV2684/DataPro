import sys
import os
import google.generativeai as genai

def summarize_text(input_text):
    """
    Uses the Gemini API to summarize the input text.

    :param input_text: The text to be summarized.
    """
    try:
        # --- IMPORTANT --- #
        # This script uses an environment variable to secure the API key.
        # Before running, set the GEMINI_API_KEY environment variable in your terminal:
        # In Windows CMD: setx GEMINI_API_KEY "YOUR_API_KEY"
        # In PowerShell: $env:GEMINI_API_KEY="YOUR_API_KEY"
        # In Linux/macOS: export GEMINI_API_KEY="YOUR_API_KEY"
        # You must restart your terminal after setting the variable.
        # --- IMPORTANT --- #
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY environment variable not set.")
            print("Please set the variable with your Google AI Studio API key.")
            return

        genai.configure(api_key=api_key)

        # Create a prompt for the model
        prompt = f"""
        You are a helpful data analyst. Your task is to interpret a technical comparison of two data files and explain the differences in simple, easy-to-understand terms for a non-technical person.

        Here is the output from a script that compared the two files:
        --- SCRIPT OUTPUT ---
        {input_text}
        --- END SCRIPT OUTPUT ---

        Please provide a brief, one or two-paragraph summary of what has changed between the two files based on the script's output.
        also with the brief summary add some technical details in it like it should be for both a beginner and a techinal person. 
        Give some technical details too 
        """

        # Call the Gemini API
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)

        # Print the summary
        print("\n--- Gemini Summary ---")
        print(response.text)
        print("----------------------")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Read the input piped from the previous command
    comparison_output = sys.stdin.read()
    if not comparison_output.strip():
        print("No input received from the comparison script.")
    else:
        summarize_text(comparison_output)
