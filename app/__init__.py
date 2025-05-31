import logging
import os
import tensorflow as tf
from flask import Flask, jsonify

from .config import Config
from .extensions import db, ma

def load_tokenizer(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return tf.keras.preprocessing.text.tokenizer_from_json(f.read())

def create_app():
    app = Flask(__name__, instance_relative_config=False)

    # Load configuration
    app.config.from_object(Config)

    model_path    = app.config["MODEL_PATH"]
    tok_in_path   = app.config["TOKENIZER_INPUT_PATH"]
    tok_out_path  = app.config["TOKENIZER_TARGET_PATH"]

    summarization_model = None
    tok_input = None
    tok_target = None
    start_i = None
    end_i = None

    # Load ML model and tokenizers if available
    if os.path.exists(model_path):
        try:
            summarization_model = tf.keras.models.load_model(model_path)
            tok_input  = load_tokenizer(tok_in_path)
            tok_target = load_tokenizer(tok_out_path)

            widx = tok_target.word_index
            start_i = widx.get("<start>", widx.get("start"))
            end_i   = widx.get("<end>",   widx.get("end"))

            app.logger.info("Model and tokenizers loaded successfully.")

        except Exception as e:
            app.logger.error(f"Failed to load model/tokenizers: {e}")
    else:
        app.logger.warning("Summarizer model not found. Skipping model/tokenizer load.")

    # Update app config (conditionally)
    app.config.update({
        "SUMMARIZER":         summarization_model,
        "TOK_INPUT":          tok_input,
        "TOK_TARGET":         tok_target,
        "START_TOKEN_INDEX":  start_i,
        "END_TOKEN_INDEX":    end_i,
        "MAX_INPUT_LEN":      int(app.config.get("MAX_LENGTH_INPUT", 50)),
        "MAX_TARGET_LEN":     int(app.config.get("MAX_LENGTH_TARGET", 20)),
    })

    # Initialize Flask extensions
    db.init_app(app)
    ma.init_app(app)

    # Register blueprints
    from app.blueprints.notes import notes_bp
    from app.blueprints.visual_ai import visual_ai_bp
    app.register_blueprint(notes_bp, url_prefix="/api/notes")
    app.register_blueprint(visual_ai_bp)

    # Health check endpoint
    @app.route("/healthz", methods=["GET"])
    def healthz():
        return jsonify(status="ok"), 200

    return app
