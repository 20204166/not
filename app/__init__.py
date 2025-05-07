# app/__init__.py
import os
import tensorflow as tf
from flask import Flask, jsonify
from app.config import Config, DevelopmentConfig, TestingConfig, ProductionConfig

def load_tokenizer(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return tf.keras.preprocessing.text.tokenizer_from_json(f.read())

def create_app():
    app = Flask(__name__, instance_relative_config=False)

    # pick your config class via APP_CONFIG env var or FLASK_ENV
    cfg = os.environ.get("APP_CONFIG")
    if cfg:
        app.config.from_object(cfg)
    else:
        env = os.environ.get("FLASK_ENV", "production").lower()
        klass = {
            "development": DevelopmentConfig,
            "testing":     TestingConfig,
            "production":  ProductionConfig,
        }.get(env, Config)
        app.config.from_object(klass)

    # normalize numeric settings
    app.config["MAX_LENGTH_INPUT"]  = int(app.config["MAX_LENGTH_INPUT"])
    app.config["MAX_LENGTH_TARGET"] = int(app.config["MAX_LENGTH_TARGET"])
    app.config["START_TOKEN_INDEX"] = int(app.config["START_TOKEN_INDEX"])
    app.config["END_TOKEN_INDEX"]   = int(app.config["END_TOKEN_INDEX"])

    # load model + tokenizers
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

    app.config["SUMMARY_MODEL"] = summarization_model
    app.config["TOK_INPUT"]     = tokenizer_input
    app.config["TOK_TARGET"]    = tokenizer_target

    # register blueprint
    from app.blueprints.notes import notes_bp
    app.register_blueprint(notes_bp, url_prefix="/notes")

    @app.route("/health")
    def health():
        return jsonify(status="ok"), 200

    return app
