# app/__init__.py

import os
import tensorflow as tf
from flask import Flask, jsonify

def load_tokenizer(path: str):
    """Load a Keras Tokenizer from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return tf.keras.preprocessing.text.tokenizer_from_json(f.read())

def create_app():
    # Create the Flask app
    app = Flask(__name__, instance_relative_config=False)

    # ─── 1) Load the correct Config class from app/config.py ─────────────
    env = os.environ.get("FLASK_ENV", "development").lower()
    cfg_map = {
        "development": "app.config.DevelopmentConfig",
        "testing":     "app.config.TestingConfig",
        "production":  "app.config.ProductionConfig",
    }
    config_path = cfg_map.get(env, "app.config.Config")
    app.config.from_object(config_path)

    # ─── 2) Normalize numeric settings for your blueprint ────────────────
    app.config["MAX_LENGTH_INPUT"]  = int(app.config.get("MAX_LENGTH_INPUT", 50))
    app.config["MAX_LENGTH_TARGET"] = int(app.config.get("MAX_LENGTH_TARGET", 20))
    app.config["START_TOKEN_INDEX"] = int(app.config.get("START_TOKEN_INDEX", 1))
    app.config["END_TOKEN_INDEX"]   = int(app.config.get("END_TOKEN_INDEX", 2))

    # ─── 3) Load your model & tokenizers into the app config ────────────
    try:
        model_path    = app.config["MODEL_PATH"]
        tok_in_path   = app.config["TOKENIZER_INPUT_PATH"]
        tok_targ_path = app.config["TOKENIZER_TARGET_PATH"]

        app.logger.info(f"Loading model from {model_path}")
        summarization_model = tf.keras.models.load_model(model_path)

        app.logger.info(f"Loading input tokenizer from {tok_in_path}")
        tokenizer_input = load_tokenizer(tok_in_path)

        app.logger.info(f"Loading target tokenizer from {tok_targ_path}")
        tokenizer_target = load_tokenizer(tok_targ_path)

    except Exception as e:
        app.logger.error("Failed loading model/tokenizers: %s", e)
        summarization_model = None
        tokenizer_input    = None
        tokenizer_target   = None

    # Make them available to your blueprint
    app.config["SUMMARY_MODEL"] = summarization_model
    app.config["TOK_INPUT"]     = tokenizer_input
    app.config["TOK_TARGET"]    = tokenizer_target

    # ─── 4) Register your Note‐processing Blueprint ──────────────────────
    from app.blueprints.notes import notes_bp
    app.register_blueprint(notes_bp, url_prefix="/notes")

    # ─── 5) Health Check ─────────────────────────────────────────────────
    @app.route("/health")
    def health():
        return jsonify(status="ok"), 200

    return app
