import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, current_app
import speech_recognition as sr

# Constants for model & tokenizer paths
MODEL_PATH = "app/models/saved_model/summarization_model.keras"
TOKENIZER_INPUT_PATH = "app/models/saved_model/tokenizer_input.json"
TOKENIZER_TARGET_PATH = "app/models/saved_model/tokenizer_target.json"
MAX_LENGTH_INPUT = 50
MAX_LENGTH_TARGET = 20

def load_tokenizer(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return tf.keras.preprocessing.text.tokenizer_from_json(f.read())

def create_app():
    app = Flask(__name__)

    # ── Load model & tokenizers once on startup ───────────────────────────────
    try:
        app.config["model"] = load_model(MODEL_PATH)
    except Exception as e:
        app.logger.error(f"Failed to load model: {e}")
        app.config["model"] = None

    app.config["tokenizer_input"] = load_tokenizer(TOKENIZER_INPUT_PATH)
    app.config["tokenizer_target"] = load_tokenizer(TOKENIZER_TARGET_PATH)

    # ── Helper functions use app.config so they can access model/tokenizers ──
    def speech_to_text(file_path: str) -> str:
        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Unable to transcribe audio: unclear input."
        except sr.RequestError as e:
            current_app.logger.error("Speech-to-text error: %s", e)
            return "Speech-to-text service unavailable."

    def generate_summary(text: str) -> str:
        model = current_app.config["model"]
        ti = current_app.config["tokenizer_input"]
        tt = current_app.config["tokenizer_target"]

        # Tokenize + pad
        seq = ti.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LENGTH_INPUT, padding="post")

        # If model failed to load, return a placeholder
        if model is None:
            return "[Model unavailable]"

        preds = model.predict(padded)
        indices = preds.argmax(axis=-1)[0]

        words = []
        for idx in indices:
            if tt.index_word.get(idx) == "<end>":
                break
            if idx == 0:
                continue
            word = tt.index_word.get(idx, "")
            if word:
                words.append(word)
        return " ".join(words)

    # ── Register your routes ───────────────────────────────────────────────────
    @app.route("/process_note", methods=["POST"])
    def process_note():
        # handle audio upload vs JSON text_input
        if "audio_file" in request.files:
            f = request.files["audio_file"]
            tmp = os.path.join("temp", f.filename)
            os.makedirs("temp", exist_ok=True)
            f.save(tmp)
            transcription = speech_to_text(tmp)
            os.remove(tmp)
        else:
            data = request.get_json(silent=True) or {}
            if "text_input" not in data:
                return jsonify(error="Missing audio_file or text_input"), 400
            transcription = data["text_input"]

        summary = generate_summary(transcription)
        return jsonify(transcription=transcription, summary=summary), 200

    @app.route("/health", methods=["GET"])
    def health_check():
        return jsonify(status="ok"), 200

    return app
