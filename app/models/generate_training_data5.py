# File: app/models/generation_training_data.py

import os
import json
import pandas as pd
from glob import glob

def generate_summary(text: str) -> str:
    """
    Generate a summary for the given text using a heuristic approach.
    This function extracts the first complete sentence from the text.

    Args:
        text (str): The full text.

    Returns:
        str: The summary (first sentence) of the text. If no clear sentence is found,
             returns the first 100 characters followed by an ellipsis.
    """
    # Split the text by period and filter out very short sentences.
    sentences = text.split('.')
    for sentence in sentences:
        summary = sentence.strip()
        if len(summary) > 20:  # Ensure a minimal length for a valid sentence.
            return summary + "."
    # Fallback: return a truncated version if no proper sentence is found.
    return text[:100] + "..."

def process_txt_file(file_path: str) -> list:
    """
    Process a plain text file by splitting it into paragraphs and generating summaries.

    Args:
        file_path (str): Path to the .txt file.

    Returns:
        list: A list of dictionaries, each with keys "text" and "summary".
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading file {file_path} with utf-8: {e}")
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
        except Exception as e2:
            print(f"Error reading file {file_path} with latin-1: {e2}")
            return []
    
    # Split text into paragraphs using double newlines and filter out very short ones.
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 100]
    data = []
    for paragraph in paragraphs:
        summary = generate_summary(paragraph)
        data.append({"text": paragraph, "summary": summary})
    return data

def process_csv_file(file_path: str, text_column: str, summary_column: str) -> list:
    """
    Process a CSV file by extracting the specified text and summary columns.
    If the summary column is missing, generate summaries using the heuristic.

    Args:
        file_path (str): Path to the CSV file.
        text_column (str): Column name containing the full text.
        summary_column (str): Column name containing the summary.

    Returns:
        list: A list of dictionaries with keys "text" and "summary".
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8', error_bad_lines=False)
    except Exception as e:
        print(f"Error reading '{file_path}' with utf-8: {e}. Trying latin-1 encoding...")
        try:
            df = pd.read_csv(file_path, encoding='latin-1', error_bad_lines=False)
        except Exception as e2:
            print(f"Error reading '{file_path}' with latin-1: {e2}")
            return []
    
    if text_column not in df.columns:
        print(f"Skipping '{file_path}' as it lacks the required text column '{text_column}'.")
        return []
    
    # If the summary column is missing, generate summaries using the heuristic.
    if summary_column not in df.columns:
        print(f"'{file_path}' lacks the summary column '{summary_column}'. Generating summaries using heuristic.")
        df[summary_column] = df[text_column].apply(lambda x: generate_summary(x) if isinstance(x, str) else "")
    
    # Rename columns to standard keys.
    df = df.rename(columns={text_column: "text", summary_column: "summary"})
    # Verify that both 'text' and 'summary' columns exist after renaming.
    if not all(col in df.columns for col in ["text", "summary"]):
        print(f"Skipping '{file_path}' as required columns 'text' and 'summary' are missing after renaming.")
        return []
    
    return df[["text", "summary"]].to_dict(orient="records")

def create_training_data(input_folder: str, output_file: str, text_column: str = "article", summary_column: str = "highlights"):
    """
    Combine all .txt and CSV files from the input folder into a single JSON file of paired training data.
    
    For each .txt file, the file is split into paragraphs and a summary is generated.
    For CSV files (e.g. the CNN/DailyMail dataset), the specified text column is used and if the summary column is missing,
    a heuristic is applied.
    
    Args:
        input_folder (str): Directory containing your files (.txt and .csv).
        output_file (str): Output path for the generated JSON file.
        text_column (str): Column name for text in CSV files (default "article").
        summary_column (str): Column name for summary in CSV files (default "highlights").
    """
    all_data = []

    # Process all .txt files.
    txt_files = glob(os.path.join(input_folder, "*.txt"))
    for txt_file in txt_files:
        print(f"Processing text file: {txt_file}")
        all_data.extend(process_txt_file(txt_file))
    
    # Process all CSV files.
    csv_files = glob(os.path.join(input_folder, "*.csv"))
    for csv_file in csv_files:
        print(f"Processing CSV file: {csv_file}")
        records = process_csv_file(csv_file, text_column, summary_column)
        all_data.extend(records)
    
    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    print(f"Training data created at: {output_file}")

if __name__ == "__main__":
    # Set the folder where your raw text and CSV files are located.
    input_folder = "app/models/data/text/public_texts"
    # Set the output JSON file path.
    output_file = "app/models/data/text/training_data.json"
    # For the CNN/DailyMail dataset, we use "article" for the text and "highlights" for the summary.
    text_column = "article"     
    summary_column = "highlights"  
    create_training_data(input_folder, output_file, text_column, summary_column)
