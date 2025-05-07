# app/__init__.py

import os
import tensorflow as tf
from flask import Flask, jsonify

# 1) Import *only* your Config from app/config.py
from app.config import Config

def load_tokenizer(path: str):
    """Load a Keras Tokenizer from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return tf.keras.preprocessing.text.tokenizer_from_json(f.read())

def create_app():
    # 2) Create Flask and load the base Config class
    app = Flask(__name__)
    app.config.from_object(Config)

    # 3) Normalize a few numeric settings (so they're ints, not strings)
    app.config["MAX_LENGTH_INPUT"]  = int(app.config.get("MAX_LENGTH_INPUT", 50))
    app.config["MAX_LENGTH_TARGET"] = int(app.config.get("MAX_LENGTH_TARGET", 20))
    app.config["START_TOKEN_INDEX"] = int(app.config.get("START_TOKEN_INDEX", 1))
    app.config["END_TOKEN_INDEX"]   = int(app.config.get("END_TOKEN_INDEX", 2))

    # 4) Load your model & tokenizers once, stash them in config
    try:
        mpath = app.config["MODEL_PATH"]
        ipath = app.config["TOKENIZER_INPUT_PATH"]
        tpath = app.config["TOKENIZER_TARGET_PATH"]

        app.logger.info(f"Loading model from {mpath}")
        summarization_model = tf.keras.models.load_model(mpath)

        app.logger.info(f"Loading input tokenizer from {ipath}")
        tokenizer_input = load_tokenizer(ipath)

        app.logger.info(f"Loading target tokenizer from {tpath}")
        tokenizer_target = load_tokenizer(tpath)

    except Exception as e:
        app.logger.error("Model/tokenizer load failed: %s", e)
        summarization_model = tokenizer_input = tokenizer_target = None

    app.config["SUMMARY_MODEL"] = summarization_model
    app.config["TOK_INPUT"]     = tokenizer_input
    app.config["TOK_TARGET"]    = tokenizer_target

    # 5) Register your notes blueprint
    from app.blueprints.notes import notes_bp
    app.register_blueprint(notes_bp, url_prefix="/notes")

    # 6) Add a simple health endpoint
    @app.route("/health")
    def health():
        return jsonify(status="ok"), 200

    return app
