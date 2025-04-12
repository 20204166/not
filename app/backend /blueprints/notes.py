# app/blueprints/notes.py

import os
import logging
import json
import numpy as np
import tensorflow as tf
import speech_recognition as sr

from flask import Blueprint, request, jsonify, current_app
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

notes_bp = Blueprint('notes_bp', __name__)

# ------------------------------------------------------------------
#  Load Model / Tokenizers with .keras extension
# ------------------------------------------------------------------
MODEL_PATH = "app/models/saved_model/summarization_model.keras"  # Updated extension!
TOKENIZER_INPUT_PATH = "app/models/saved_model/tokenizer_input.json"
TOKENIZER_TARGET_PATH = "app/models/saved_model/tokenizer_target.json"

MAX_LENGTH_INPUT = 50
MAX_LENGTH_TARGET = 20

try:
    summarization_model = load_model(MODEL_PATH)
    current_app.logger.info("Loaded custom summarization model.")
except Exception as e:
    summarization_model = None
    current_app.logger.error("Error loading summarization model: %s", e)

def load_tokenizer(tokenizer_path: str):
    """
    Load a Keras Tokenizer from a JSON file.
    """
    if not os.path.exists(tokenizer_path):
        current_app.logger.warning(f"Tokenizer file not found: {tokenizer_path}")
        return None
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_json = f.read()
    return tokenizer_from_json(tokenizer_json)

tokenizer_input = load_tokenizer(TOKENIZER_INPUT_PATH)
tokenizer_target = load_tokenizer(TOKENIZER_TARGET_PATH)

# ------------------------------------------------------------------
#  Shared Utility Functions
# ------------------------------------------------------------------
def speech_to_text(file_path: str) -> str:
    """
    Convert an audio file to text using the SpeechRecognition library.
    Uses Google's free API for demonstration.
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Unable to transcribe audio: unclear input."
    except sr.RequestError as e:
        current_app.logger.error("Speech-to-text service error: %s", e)
        return "Speech-to-text service unavailable."

def generate_summary_inference(input_text: str) -> str:
    """
    Generate a summary using the loaded model and tokenizers with a greedy decoding approach.
    """
    if not summarization_model or not tokenizer_input or not tokenizer_target:
        return "Model or tokenizers not available."

    seq = tokenizer_input.texts_to_sequences([input_text])
    encoder_input = pad_sequences(seq, maxlen=MAX_LENGTH_INPUT, padding='post')

    target_seq = np.array([[1]])  # Start token assumed to be index 1.
    summary_generated = []

    for _ in range(MAX_LENGTH_TARGET):
        preds = summarization_model.predict([encoder_input, target_seq], verbose=0)
        token_index = np.argmax(preds[0, -1, :])
        if token_index == 2:  # End token (adjust this index as needed)
            break
        if token_index == 0:
            continue
        word = tokenizer_target.index_word.get(token_index, "")
        if not word:
            break
        summary_generated.append(word)
        target_seq = np.concatenate([target_seq, [[token_index]]], axis=1)

    return " ".join(summary_generated)

# ------------------------------------------------------------------
#  Blueprint Routes
# ------------------------------------------------------------------
@notes_bp.route('/process', methods=['POST'])
def process_note():
    """
    Process a note by either converting an audio file to text or summarizing a text input.
    """
    transcription = ""
    try:
        if 'audio_file' in request.files:
            audio_file = request.files['audio_file']
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, audio_file.filename)
            audio_file.save(temp_path)
            transcription = speech_to_text(temp_path)
            os.remove(temp_path)
        else:
            data = request.get_json(silent=True) or {}
            if 'text_input' not in data:
                return jsonify({'error': 'Missing audio_file or text_input'}), 400
            transcription = data['text_input']

        summary = generate_summary_inference(transcription)
        response = {
            'transcription': transcription,
            'summary': summary
        }
        return jsonify(response), 200

    except Exception as e:
        current_app.logger.exception("Error in /process endpoint: %s", e)
        return jsonify({'error': str(e)}), 500

@notes_bp.route('/evaluate', methods=['GET'])
def evaluate_model():
    """
    Evaluate the model on a hold-out dataset by returning token-level accuracy.
    """
    from app.models.training_text_summarization import evaluate_holdout_accuracy
    try:
        accuracy = evaluate_holdout_accuracy()
        return jsonify({"validation_token_accuracy": accuracy}), 200
    except Exception as e:
        current_app.logger.error("Error evaluating model: %s", e)
        return jsonify({'error': 'Failed to evaluate model'}), 500
