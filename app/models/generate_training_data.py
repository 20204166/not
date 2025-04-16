# File: generate_training_data.py

from datasets import load_dataset
import json
import re
import os

def clean_text(text: str) -> str:
    """
    Clean the input text by removing unwanted characters and extra whitespace.
    """
    text = re.sub(r"[^a-zA-Z0-9\s\.,;:!?'\-]", '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def truncate_text(text: str, max_words: int) -> str:
    """
    Truncate the text to the first max_words words.
    """
    words = text.split()
    return " ".join(words[:max_words])

def truncate_summary_complete(text: str, max_words: int) -> str:
    """
    Truncate the summary text to the first max_words words, ensuring that the output 
    ends with a sentence-ending punctuation if possible.
    """
    valid_endings = {'.', '?', '!'}
    words = text.split()
    if len(words) <= max_words:
        return text if text and text[-1] in valid_endings else text + '.'
    truncated = " ".join(words[:max_words])
    if truncated and truncated[-1] in valid_endings:
        return truncated
    last_pos = -1
    for punct in valid_endings:
        pos = truncated.rfind(punct)
        if pos > last_pos:
            last_pos = pos
    if last_pos != -1:
        return truncated[:last_pos+1].strip()
    else:
        return truncated + '.'

def process_cnn_dailymail() -> list:
    """
    Load and process the CNN/DailyMail dataset (v3.0.0) for training data.
    
    Uses the 'article' and 'highlights' keys from each sample. Outputs dictionaries
    with consistent keys: "text" for the article and "summary" for the summary.
    """
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
    processed_data = []
    for sample in dataset:
        article = clean_text(sample["article"])
        summary = clean_text(sample["highlights"])
        processed_data.append({
            "text":     truncate_text(article, 50),
            "summary":  truncate_summary_complete(summary, 20)
        })
    return processed_data

def process_reddit_tifu() -> list:
    """
    Load and process the Reddit TIFU dataset (short version) for training data.
    
    Checks for 'text'/'summary', otherwise falls back to 'document'/'tldr'.
    """
    dataset = load_dataset("reddit_tifu", "short", split="train", trust_remote_code=True)
    processed_data = []
    for sample in dataset:
        input_text = clean_text(sample.get("text", sample.get("document", "")))
        output_summary = clean_text(sample.get("summary", sample.get("tldr", "")))
        processed_data.append({
            "text":    truncate_text(input_text, 50),
            "summary": truncate_summary_complete(output_summary, 20)
        })
    return processed_data

def process_billsum() -> list:
    """
    Load and process the BillSum dataset for training data.
    
    Uses 'bill_text' (or 'bill') and 'summary' keys.
    """
    try:
        dataset = load_dataset("billsum", split="train")
    except Exception as e:
        raise ValueError("Could not load the BillSum dataset. "
                         "Please ensure it is available or adjust the dataset identifier.") from e

    processed_data = []
    for sample in dataset:
        bill_text = clean_text(sample.get("bill_text", sample.get("bill", "")))
        summary   = clean_text(sample.get("summary", ""))
        processed_data.append({
            "text":    truncate_text(bill_text, 50),
            "summary": truncate_summary_complete(summary, 20)
        })
    return processed_data

def process_newsroom() -> list:
    """
    Load and process the Newsroom dataset for training data.
    
    Checks for 'text' (or falls back to 'document') and uses 'summary'.
    """
    try:
        # allow execution of the Newsroom repo’s custom loading code
        dataset = load_dataset("newsroom", split="train", trust_remote_code=True)
    except Exception as e:
        raise ValueError("Could not load the Newsroom dataset. "
                         "Please ensure the dataset is available or adjust the dataset identifier.") from e

    processed_data = []
    for sample in dataset:
        input_text     = clean_text(sample.get("text", sample.get("document", "")))
        output_summary = clean_text(sample.get("summary", ""))
        processed_data.append({
            "text":    truncate_text(input_text, 50),
            "summary": truncate_summary_complete(output_summary, 20)
        })
    return processed_data

def save_combined_data(output_file: str):
    """
    Load and process CNN/DailyMail, Reddit TIFU, BillSum, and Newsroom,
    combine into one list, and save as JSON with keys 'text' and 'summary'.
    """
    combined = (
        process_cnn_dailymail()
      + process_reddit_tifu()
      + process_billsum()
      + process_newsroom()
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=4)
    print(f"\nCombined training data saved to: {output_file}")

if __name__ == "__main__":
    save_combined_data("app/models/data/text/training_data.json")
