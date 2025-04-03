# File: generate_cnn_dailymail_training_data.py

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
        return text if text[-1] in valid_endings else text + '.'
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

def save_cnn_dailymail_data(output_file: str):
    """
    Load the CNN/DailyMail dataset (version 3.0.0), clean and truncate the articles and summaries,
    and save the resulting list of dictionaries to a JSON file.
    """
    # Load the dataset.
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
    
    # Process a sample for display.
    sample = dataset[0]
    sample_article = clean_text(sample["article"])
    sample_summary = clean_text(sample["highlights"])
    truncated_article = truncate_text(sample_article, 50)  # 50 words for input.
    truncated_summary = truncate_summary_complete(sample_summary, 20)  # 20 words for summary.
    
    print("Sample Article (first 50 words):")
    print(truncated_article + "...")
    print("\nSample Summary (complete sentence if possible):")
    print(truncated_summary)
    
    # Process all samples.
    data = []
    for sample in dataset:
        cleaned_article = clean_text(sample["article"])
        cleaned_summary = clean_text(sample["highlights"])
        truncated_article = truncate_text(cleaned_article, 50)
        truncated_summary = truncate_summary_complete(cleaned_summary, 20)
        data.append({
            "text": truncated_article,
            "summary": truncated_summary
        })
    
    # Create the directory if it doesn't exist.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"\nTraining data saved to: {output_file}")

if __name__ == "__main__":
    output_file = "app/models/data/text/training_data.json"
    save_cnn_dailymail_data(output_file)
