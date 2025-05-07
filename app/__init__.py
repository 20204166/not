# app/__init__.py

import tensorflow as tf
import os 
from flask import Flask, jsonify
from app.config import Config

def load_tokenizer(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return tf.keras.preprocessing.text.tokenizer_from_json(f.read())

def create_app():
    # 1) Create the Flask app
    app = Flask(__name__, instance_relative_config=False)
    
    # 2) Always load the base Config class (DEBUG=False, TESTING=False)
    cfg_path = os.environ.get("APP_CONFIG", "app.config.Config")
    app.config.from_object(cfg_path)

    # 3) Normalize any numeric settings your blueprint needs
    app.config["MAX_LENGTH_INPUT"]  = int(app.config.get("MAX_LENGTH_INPUT", 50))
    app.config["MAX_LENGTH_TARGET"] = int(app.config.get("MAX_LENGTH_TARGET", 20))
    app.config["START_TOKEN_INDEX"] = int(app.config.get("START_TOKEN_INDEX", 1))
    app.config["END_TOKEN_INDEX"]   = int(app.config.get("END_TOKEN_INDEX", 2))

    # 4) Load your model & tokenizers
    try:
        mpath = app.config["MODEL_PATH"]
        ip   = app.config["TOKENIZER_INPUT_PATH"]
        tp   = app.config["TOKENIZER_TARGET_PATH"]

        app.logger.info(f"Loading model from {mpath}")
        summarization_model = tf.keras.models.load_model(mpath)

        app.logger.info(f"Loading input tokenizer from {ip}")
        tokenizer_input = load_tokenizer(ip)

        app.logger.info(f"Loading target tokenizer from {tp}")
        tokenizer_target = load_tokenizer(tp)

    except Exception as e:
        app.logger.error("Model/tokenizer load failed: %s", e)
        summarization_model = tokenizer_input = tokenizer_target = None

    # 5) Make them available to your blueprint
    app.config["SUMMARY_MODEL"] = summarization_model
    app.config["TOK_INPUT"]     = tokenizer_input
    app.config["TOK_TARGET"]    = tokenizer_target

    # 6) Register blueprint and health check
    from app.blueprints.notes import notes_bp
    app.register_blueprint(notes_bp, url_prefix="/notes")

    @app.route("/health")
    def health():
        return jsonify(status="ok"), 200

    return app
