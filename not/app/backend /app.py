from flask import Flask, request, jsonify, current_app
from typing import Any, Dict
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import speech_recognition as sr

app = Flask(__name__)

# ----- Load Custom Summarization Model and Tokenizers -----
MODEL_PATH = "app/models/saved_model/summarization_model.h5"
TOKENIZER_INPUT_PATH = "app/models/saved_model/tokenizer_input.json"
TOKENIZER_TARGET_PATH = "app/models/saved_model/tokenizer_target.json"

# Load the custom-trained summarization model.
summarization_model = load_model(MODEL_PATH)

def load_tokenizer(tokenizer_path: str):
    """Load a Keras Tokenizer from a JSON file."""
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_json = f.read()
    return tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

# Load tokenizers.
tokenizer_input = load_tokenizer(TOKENIZER_INPUT_PATH)
tokenizer_target = load_tokenizer(TOKENIZER_TARGET_PATH)

# Set maximum sequence lengths (must match your training configuration)
MAX_LENGTH_INPUT = 50
MAX_LENGTH_TARGET = 20

def speech_to_text(file_path: str) -> str:
    """
    Convert an audio file to text using the SpeechRecognition library.
    
    Args:
        file_path (str): The path to the audio file.
        
    Returns:
        str: The transcribed text.
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        # Using Google's free speech-to-text API for demonstration.
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Unable to transcribe audio: unclear input."
    except sr.RequestError as e:
        current_app.logger.error("Speech-to-text service error: %s", e)
        return "Speech-to-text service unavailable."

def generate_summary(text: str) -> str:
    """
    Generate a summary for the provided text using the custom-trained summarization model.
    
    Args:
        text (str): The text to summarize.
        
    Returns:
        str: The generated summary.
    """
    # Tokenize and pad the input text.
    sequence = tokenizer_input.texts_to_sequences([text])
    padded_seq = pad_sequences(sequence, maxlen=MAX_LENGTH_INPUT, padding='post')
    
    # Predict the output sequence.
    predictions = summarization_model.predict(padded_seq)
    # For each time step, choose the index with highest probability.
    predicted_indices = predictions.argmax(axis=-1)[0]
    
    # Convert indices back to words using the target tokenizer.
    summary_words = []
    for idx in predicted_indices:
        # Stop if the end-of-sequence token is encountered (assumed to be <end>).
        if tokenizer_target.index_word.get(idx) == "<end>":
            break
        # Skip padding tokens (assumed to be index 0).
        if idx == 0:
            continue
        word = tokenizer_target.index_word.get(idx, "")
        if word:
            summary_words.append(word)
    
    return " ".join(summary_words)

@app.route('/process_note', methods=['POST'])
def process_note() -> Any:
    """
    Process note-taking requests by converting speech to text (if an audio file is provided)
    and summarizing the resulting text. Accepts either:
      - A file upload with the key 'audio_file'
      - Or a JSON payload with the key 'text_input'
    
    Returns:
        A JSON response containing:
            - transcription: The transcribed text (or provided text input).
            - summary: A summarized version of the text.
        If input is missing or invalid, returns a JSON error with a 400 status code.
    """
    transcription: str = ""
    
    # Check if an audio file was provided in the request.
    if 'audio_file' in request.files:
        audio_file = request.files['audio_file']
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, audio_file.filename)
        audio_file.save(temp_path)
        transcription = speech_to_text(temp_path)
        os.remove(temp_path)
    else:
        # Otherwise, expect a JSON payload with a 'text_input' field.
        data: Dict[str, Any] = request.get_json(silent=True)
        if not data or 'text_input' not in data:
            return jsonify({'error': 'Missing audio_file or text_input parameter'}), 400
        transcription = data['text_input']
    
    # Generate a summary for the transcribed (or provided) text.
    summary = generate_summary(transcription)
    
    response: Dict[str, str] = {
        'transcription': transcription,
        'summary': summary
    }
    return jsonify(response), 200

if __name__ == '__main__':
    # Run the Flask development server on port 5000 with debug mode enabled.
    app.run(debug=True, port=5000)
