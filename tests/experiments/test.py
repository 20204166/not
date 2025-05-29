import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def load_tokenizer(tokenizer_path: str):
    """
    Load a Tokenizer from a JSON file.
    Args:
        tokenizer_path (str): Path to the tokenizer JSON file.

    Returns:
        Tokenizer: The loaded tokenizer.
    """
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_json = f.read()
    return tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

def generate_summary_inference(model, tokenizer_input, tokenizer_target, input_text: str,
                               max_length_input: int, max_length_target: int) -> str:
    """
    Generate a summary for a given input text using greedy decoding.
    """
    seq = tokenizer_input.texts_to_sequences([input_text])
    encoder_input = pad_sequences(seq, maxlen=max_length_input, padding="post")

    decoder_input = np.zeros((1, max_length_target), dtype="int32")
    start_token = tokenizer_target.word_index.get("<start>", 1)
    end_token = tokenizer_target.word_index.get("<end>", 2)
    decoder_input[0, 0] = start_token

    summary_generated = []
    for t in range(1, max_length_target):
        predictions = model.predict([encoder_input, decoder_input], verbose=0)
        next_token_probs = predictions[0, t - 1, :]
        next_token = np.argmax(next_token_probs)

        if next_token == end_token:
            break

        word = tokenizer_target.index_word.get(next_token, "")
        if not word:
            break

        summary_generated.append(word)
        decoder_input[0, t] = next_token

    return " ".join(summary_generated)


def interactive_review(model, tokenizer_input, tokenizer_target,
                       max_length_input: int, max_length_target: int):
    """
    Run an interactive loop to generate summaries.
    """
    print("Interactive Summarization Mode. Type 'exit' to quit.")

    while True:
        user_input = input("\nEnter text to summarize: ")
        if user_input.lower() == "exit":
            break

        summary = generate_summary_inference(
            model, tokenizer_input, tokenizer_target,
            user_input, max_length_input, max_length_target
        )
        print("\nOriginal Text:")
        print(user_input)
        print("\nGenerated Summary:")
        print(summary)


def qualitative_review(model_path: str, tokenizer_input_path: str,
                       tokenizer_target_path: str, max_length_input: int,
                       max_length_target: int):
    """
    Load the model and run sample + interactive summary generation.
    """
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"Attention": tf.keras.layers.Attention}
    )

    tokenizer_input = load_tokenizer(tokenizer_input_path)
    tokenizer_target = load_tokenizer(tokenizer_target_path)

    sample_texts = [
        "The Project Gutenberg eBook of Great Expectations is a classic "
        "novel by Charles Dickens. It tells the story of Pip, an orphan "
        "with a mysterious benefactor who changes his life.",
        "Recent advancements in artificial intelligence have "
        "revolutionized data processing. AI-driven applications now "
        "provide real-time insights and have transformed industries."
    ]

    print("Qualitative Review of Sample Generated Summaries:\n")

    for i, text in enumerate(sample_texts):
        summary = generate_summary_inference(
            model, tokenizer_input, tokenizer_target,
            text, max_length_input, max_length_target
        )
        print(f"Sample {i + 1}:")
        print("Original Text:")
        print(text)
        print("\nGenerated Summary:")
        print(summary)
        print("\n" + "-" * 50 + "\n")

    interactive_review(
        model, tokenizer_input, tokenizer_target,
        max_length_input, max_length_target
    )

if __name__ == "__main__":
    model_path = "app/models/saved_model/summarization_model.keras"
    tokenizer_input_path = "app/models/saved_model/tokenizer_input.json"
    tokenizer_target_path = "app/models/saved_model/tokenizer_target.json"
    max_length_input = 50
    max_length_target = 20
    
    qualitative_review(
        model_path,
        tokenizer_input_path,
        tokenizer_target_path,
        max_length_input,
        max_length_target
    )

