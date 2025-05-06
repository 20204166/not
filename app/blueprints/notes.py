# run.py

import os
import json
import tensorflow as tf
from flask import Flask, jsonify
from tensorflow.keras.preprocessing.text import tokenizer_from_json

def load_tokenizer(path):
    """Load a Keras tokenizer from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    return tokenizer_from_json(data)

def create_app():
    app = Flask(__name__, instance_relative_config=False)

    # 1) Load configuration
    #    Pick the config class you want (DevelopmentConfig, ProductionConfig, etc.)
    app.config.from_object("config.Config")

    # 2) Load the model
    model_path = app.config["MODEL_PATH"]
    app.logger.info(f"Loading summarization model from {model_path}...")
    app.config["SUMMARY_MODEL"] = tf.keras.models.load_model(model_path)

    # 3) Load tokenizers
    ti_path = app.config["TOKENIZER_INPUT_PATH"]
    tt_path = app.config["TOKENIZER_TARGET_PATH"]
    app.logger.info(f"Loading input tokenizer from {ti_path}...")
    app.config["TOK_INPUT"]   = load_tokenizer(ti_path)
    app.logger.info(f"Loading target tokenizer from {tt_path}...")
    app.config["TOK_TARGET"]  = load_tokenizer(tt_path)

    # 4) Copy over any other numeric config the blueprint needs
    app.config["MAX_LENGTH_INPUT"]  = app.config.get("MAX_LENGTH_INPUT", 50)
    app.config["MAX_LENGTH_TARGET"] = app.config.get("MAX_LENGTH_TARGET", 20)
    app.config["START_TOKEN_INDEX"] = app.config.get("START_TOKEN_INDEX", 1)
    app.config["END_TOKEN_INDEX"]   = app.config.get("END_TOKEN_INDEX", 2)

    # 5) Health check
    @app.route("/health")
    def health():
        return jsonify(status="ok"), 200

    # 6) Register your notes blueprint at /notes
    from app.blueprints.notes import notes_bp
    app.register_blueprint(notes_bp, url_prefix="/notes")

    return app

# Create the Flask app
app = create_app()

if __name__ == "__main__":
    # For local dev / debugging
    debug = app.config.get("DEBUG", False)
    app.run(host="0.0.0.0", port=5000, debug=debug)
