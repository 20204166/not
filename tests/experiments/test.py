<<<<<<< HEAD
<<<<<<< HEAD
# File: test.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
=======
=======
>>>>>>> 1786abc (Remove app/__init__.py and add tests directory)
# File: experiments/qualitative_review.py

import tensorflow as tf
# Force CPU if desired:
tf.config.set_visible_devices([], 'GPU')
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os
<<<<<<< HEAD
>>>>>>> 1786abc (Remove app/__init__.py and add tests directory)
=======
>>>>>>> 1786abc (Remove app/__init__.py and add tests directory)

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

<<<<<<< HEAD
<<<<<<< HEAD
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
=======
=======
>>>>>>> 1786abc (Remove app/__init__.py and add tests directory)
def generate_summary_inference(model, tokenizer_input, tokenizer_target, input_text: str, max_length_input: int, max_length_target: int) -> str:
    """
    Generate a summary for a given input text using a simple iterative (greedy) decoding loop.
    
    This function assumes that the start token is represented by index 1 and the end token by index 2.
    
    Args:
        model: The trained seq2seq model.
        tokenizer_input: Tokenizer for input texts.
        tokenizer_target: Tokenizer for target texts.
        input_text (str): The text to summarize.
        max_length_input (int): Maximum length for input sequences.
        max_length_target (int): Maximum length for target sequences.
<<<<<<< HEAD
>>>>>>> 1786abc (Remove app/__init__.py and add tests directory)
=======
>>>>>>> 1786abc (Remove app/__init__.py and add tests directory)
    
    Returns:
        str: The generated summary.
    """
    # Tokenize and pad the input text.
    seq = tokenizer_input.texts_to_sequences([input_text])
    encoder_input = pad_sequences(seq, maxlen=max_length_input, padding='post')
    
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
=======
>>>>>>> 1786abc (Remove app/__init__.py and add tests directory)
    # Initialize the decoder input array with all zeros.
    decoder_input = np.zeros((1, max_length_target), dtype='int32')
    start_token = 1  # assumed index for <start>
    end_token = 2    # assumed index for <end>
    decoder_input[0, 0] = start_token
    
    summary_generated = []
    # Iteratively predict each time step.
    for t in range(1, max_length_target):
        predictions = model.predict([encoder_input, decoder_input], verbose=0)
        # Use the prediction from the last filled time step (t-1)
        next_token_probs = predictions[0, t-1, :]
        next_token = np.argmax(next_token_probs)
        # If the end token is predicted, stop.
        if next_token == end_token:
            break
        # Get the word for the predicted token.
<<<<<<< HEAD
>>>>>>> 1786abc (Remove app/__init__.py and add tests directory)
=======
>>>>>>> 1786abc (Remove app/__init__.py and add tests directory)
        word = tokenizer_target.index_word.get(next_token, "")
        if not word:
            break
        summary_generated.append(word)
<<<<<<< HEAD
<<<<<<< HEAD
        # Update decoder_input at time step t.
=======
        # Update the decoder input for the next time step.
>>>>>>> 1786abc (Remove app/__init__.py and add tests directory)
=======
        # Update the decoder input for the next time step.
>>>>>>> 1786abc (Remove app/__init__.py and add tests directory)
        decoder_input[0, t] = next_token

    return " ".join(summary_generated)

<<<<<<< HEAD
<<<<<<< HEAD
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
=======
=======
>>>>>>> 1786abc (Remove app/__init__.py and add tests directory)
def interactive_review(model, tokenizer_input, tokenizer_target, max_length_input: int, max_length_target: int):
    """
    Run an interactive loop that prompts the user for input text, generates a summary,
    and prints both the original text and its generated summary.
    
    Args:
        model: The trained seq2seq model.
        tokenizer_input: Tokenizer for input texts.
        tokenizer_target: Tokenizer for target texts.
        max_length_input (int): Maximum input sequence length.
        max_length_target (int): Maximum target sequence length.
    """
    print("Interactive Summarization Mode. Type 'exit' to quit.")
    while True:
        user_input = input("\nEnter text to summarize: ")
        if user_input.lower() == "exit":
            break
        summary = generate_summary_inference(model, tokenizer_input, tokenizer_target, user_input, max_length_input, max_length_target)
        print("\nGenerated Summary:")
        print(summary)

def qualitative_review(model_path: str, tokenizer_input_path: str, tokenizer_target_path: str, max_length_input: int, max_length_target: int):
    """
    Load the trained model and tokenizers, and perform a qualitative review
    by printing sample outputs and entering interactive mode.
    
    Args:
        model_path (str): Path to the saved summarization model.
        tokenizer_input_path (str): Path to the saved input tokenizer JSON.
        tokenizer_target_path (str): Path to the saved target tokenizer JSON.
        max_length_input (int): Maximum input sequence length.
        max_length_target (int): Maximum target sequence length.
    """
    # Load the trained model.
    model = tf.keras.models.load_model(model_path)
    # Load tokenizers.
    tokenizer_input = load_tokenizer(tokenizer_input_path)
    tokenizer_target = load_tokenizer(tokenizer_target_path)
    
    # Display sample summaries.
    sample_texts = [
        "The Project Gutenberg eBook of Great Expectations is a classic novel by Charles Dickens. It tells the story of Pip, an orphan with a mysterious benefactor who changes his life.",
        "Recent advancements in artificial intelligence have revolutionized data processing. AI-driven applications now provide real-time insights and have transformed industries across the board."
    ]
    
    print("Qualitative Review of Sample Generated Summaries:\n")
    for i, text in enumerate(sample_texts):
        summary = generate_summary_inference(model, tokenizer_input, tokenizer_target, text, max_length_input, max_length_target)
        print(f"Sample {i+1}:")
<<<<<<< HEAD
>>>>>>> 1786abc (Remove app/__init__.py and add tests directory)
=======
>>>>>>> 1786abc (Remove app/__init__.py and add tests directory)
        print("Original Text:")
        print(text)
        print("\nGenerated Summary:")
        print(summary)
<<<<<<< HEAD
<<<<<<< HEAD
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()
=======
=======
>>>>>>> 1786abc (Remove app/__init__.py and add tests directory)
        print("\n" + "-"*50 + "\n")
    
    # Enter interactive mode.
    interactive_review(model, tokenizer_input, tokenizer_target, max_length_input, max_length_target)

if __name__ == "__main__":
    # Define paths for the saved model and tokenizers.
    model_path = "app/models/saved_model/summarization_model.h5"
    tokenizer_input_path = "app/models/saved_model/tokenizer_input.json"
    tokenizer_target_path = "app/models/saved_model/tokenizer_target.json"
    
    # Set maximum sequence lengths (must match training configuration).
    max_length_input = 50
    max_length_target = 20
    
    qualitative_review(model_path, tokenizer_input_path, tokenizer_target_path, max_length_input, max_length_target)

<<<<<<< HEAD
>>>>>>> 1786abc (Remove app/__init__.py and add tests directory)
=======
>>>>>>> 1786abc (Remove app/__init__.py and add tests directory)
