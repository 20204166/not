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
    """
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
    processed_data = []
    for sample in dataset:
        article = clean_text(sample["article"])
        summary = clean_text(sample["highlights"])
        truncated_article = truncate_text(article, 50)      # 50 words for article
        truncated_summary = truncate_summary_complete(summary, 20)  # 20 words for summary
        processed_data.append({
            "text": truncated_article,
            "summary": truncated_summary
        })
    return processed_data

def process_reddit_tifu() -> list:
    """
    Load and process the Reddit TIFU dataset (short version) for training data.
    
    This dataset uses the keys 'document' for the input text and 'tldr' for the summary.
    """
    dataset = load_dataset("reddit_tifu", "short", split="train")
    processed_data = []
    for sample in dataset:
        document = clean_text(sample.get("document", ""))
        summary = clean_text(sample.get("tldr", ""))
        truncated_document = truncate_text(document, 50)
        truncated_summary = truncate_summary_complete(summary, 20)
        processed_data.append({
            "text": truncated_document,
            "summary": truncated_summary
        })
    return processed_data

def save_combined_data(output_file: str):
    """
    Load and process both the CNN/DailyMail and Reddit TIFU datasets,
    combine them into one list, and save the resulting data to a JSON file.
    """
    cnn_data = process_cnn_dailymail()
    reddit_data = process_reddit_tifu()
    combined_data = cnn_data + reddit_data

    # Create the directory if it doesn't exist.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
    print(f"\nCombined training data saved to: {output_file}")

if __name__ == "__main__":
    # Specify the output file for the combined data.
    output_file = "app/models/data/text/training_data_combined.json"
    save_combined_data(output_file)
