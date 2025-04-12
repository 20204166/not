# File: test.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

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

def generate_summary(model, tokenizer_input, tokenizer_target, input_text: str, max_length_input: int, max_length_target: int) -> str:
    """
    Generate a summary for the provided input text using a greedy decoding approach.
    
    This function assumes the start token index is 1 and the end token index is 2.
    
    Args:
        model: The trained seq2seq model.
        tokenizer_input: The tokenizer used for the input text.
        tokenizer_target: The tokenizer used for the target text.
        input_text (str): The input text to summarize.
        max_length_input (int): Maximum sequence length for input.
        max_length_target (int): Maximum sequence length for the generated summary.
    
    Returns:
        str: The generated summary.
    """
    # Tokenize and pad the input text.
    seq = tokenizer_input.texts_to_sequences([input_text])
    encoder_input = pad_sequences(seq, maxlen=max_length_input, padding='post')
    
    # Initialize decoder input with zeros and set the start token.
    decoder_input = np.zeros((1, max_length_target), dtype='int32')
    start_token = 1  # Assumed index for <start>
    end_token = 2    # Assumed index for <end>
    decoder_input[0, 0] = start_token

    summary_generated = []
    # Iteratively generate tokens.
    for t in range(1, max_length_target):
        predictions = model.predict([encoder_input, decoder_input], verbose=0)
        # Use the prediction from the last filled time step.
        next_token_probs = predictions[0, t-1, :]
        next_token = np.argmax(next_token_probs)
        # Stop if the end token is predicted.
        if next_token == end_token:
            break
        word = tokenizer_target.index_word.get(next_token, "")
        if not word:
            break
        summary_generated.append(word)
        # Update decoder_input at time step t.
        decoder_input[0, t] = next_token

    return " ".join(summary_generated)

def main():
    # Define paths for your model and tokenizers.
    model_path = "app/models/saved_model/summarization_model.keras"  # Correct model filename
    tokenizer_input_path = "app/models/saved_model/tokenizer_input.json"
    tokenizer_target_path = "app/models/saved_model/tokenizer_target.json"
    
    # Maximum sequence lengths (should match your training configuration).
    max_length_input = 50
    max_length_target = 20

    # Load the trained model and tokenizers.
    model = tf.keras.models.load_model(model_path)
    tokenizer_input = load_tokenizer(tokenizer_input_path)
    tokenizer_target = load_tokenizer(tokenizer_target_path)

    # Define some sample texts to test summary generation.
    sample_texts = [
        "The Project Gutenberg eBook of Great Expectations is a classic novel by Charles Dickens, telling the story of Pip and his mysterious benefactor.",
        "Recent advancements in artificial intelligence are revolutionizing data processing, with AI-driven applications offering real-time insights."
    ]
    
    # Generate and print summaries for the sample texts.
    for i, text in enumerate(sample_texts, start=1):
        summary = generate_summary(model, tokenizer_input, tokenizer_target, text, max_length_input, max_length_target)
        print(f"Sample {i}:")
        print("Original Text:")
        print(text)
        print("\nGenerated Summary:")
        print(summary)
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()
