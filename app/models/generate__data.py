# File: generate_training_data.py

from datasets import load_dataset
import json
import re
import os
import sys
import urllib.request
import tarfile

def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s\.,;:!?'\-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def truncate_text(text: str, max_words: int) -> str:
    return " ".join(text.split()[:max_words])

def truncate_summary_complete(text: str, max_words: int) -> str:
    valid_endings = {".", "?", "!"}
    words = text.split()
    if len(words) <= max_words:
        return text if text and text[-1] in valid_endings else text + "."
    truncated = " ".join(words[:max_words])
    if truncated and truncated[-1] in valid_endings:
        return truncated
    last = max(truncated.rfind(p) for p in valid_endings)
    return truncated[: last + 1].strip() if last != -1 else truncated + "."

def process_cnn_dailymail() -> list:
    ds = load_dataset("cnn_dailymail", "3.0.0", split="train")
    out = []
    for s in ds:
        a = clean_text(s["article"])
        h = clean_text(s["highlights"])
        out.append({
            "text":    truncate_text(a, 50),
            "summary": truncate_summary_complete(h, 20)
        })
    return out

def process_reddit_tifu() -> list:
    ds = load_dataset("reddit_tifu", "short", split="train", trust_remote_code=True)
    out = []
    for s in ds:
        inp = clean_text(s.get("text", s.get("document", "")))
        tgt = clean_text(s.get("summary", s.get("tldr", "")))
        out.append({
            "text":    truncate_text(inp, 50),
            "summary": truncate_summary_complete(tgt, 20),
        })
    return out

def process_billsum() -> list:
    try:
        ds = load_dataset("billsum", split="train")
    except Exception:
        print("⚠️  BillSum not found—skipping.", file=sys.stderr)
        return []
    out = []
    for s in ds:
        bill = clean_text(s.get("bill_text", s.get("bill", "")))
        summ = clean_text(s.get("summary", ""))
        out.append({
            "text":    truncate_text(bill, 50),
            "summary": truncate_summary_complete(summ, 20),
        })
    return out


def save_combined_data(output_file: str):
    parts = []
    parts += process_cnn_dailymail()
    parts += process_reddit_tifu()
    parts += process_billsum()


    if not parts:
        raise RuntimeError("No data processed—check your datasets.")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parts, f, ensure_ascii=False, indent=2)
    print(f"\n Combined training data written to {output_file}")

if __name__ == "__main__":
    save_combined_data("app/models/data/text/training_data.json")
