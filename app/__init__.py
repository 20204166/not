# app/__init__.py
import os
import tensorflow as tf
from flask import Flask, jsonify
from app.config import Config

def load_tokenizer(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return tf.keras.preprocessing.text.tokenizer_from_json(f.read())

def create_app():
    app = Flask(__name__, instance_relative_config=False)

    # load your base config class
    cfg_path = os.environ.get("APP_CONFIG", "app.config.Config")
    app.config.from_object(cfg_path)
    
    app.logger.debug("CONFIG START_TOKEN_INDEX=%r END_TOKEN_INDEX=%r",
                 app.config.get("START_TOKEN_INDEX"),
                 app.config.get("END_TOKEN_INDEX"))


    # load model + tokenizers
    model_path  = app.config["MODEL_PATH"]
    tok_in_path = app.config["TOKENIZER_INPUT_PATH"]
    tok_out_path= app.config["TOKENIZER_TARGET_PATH"]

    summarization_model = tf.keras.models.load_model(model_path)
    tokenizer_input     = load_tokenizer(tok_in_path)
    tokenizer_target    = load_tokenizer(tok_out_path)

    # **dynamically** figure out your start/end indices
    start_idx = tokenizer_target.word_index.get("<start>")
    end_idx   = tokenizer_target.word_index.get("<end>")

    app.config["SUMMARY_MODEL"]     = summarization_model
    app.config["TOK_INPUT"]         = tokenizer_input
    app.config["TOK_TARGET"]        = tokenizer_target
    app.config["START_TOKEN_INDEX"] = start_idx
    app.config["END_TOKEN_INDEX"]   = end_idx

    # bring through the rest of your numeric settings
    app.config["MAX_LENGTH_INPUT"]  = int(app.config["MAX_LENGTH_INPUT"])
    app.config["MAX_LENGTH_TARGET"] = int(app.config["MAX_LENGTH_TARGET"])

    # register blueprint
    from app.blueprints.notes import notes_bp
    app.register_blueprint(notes_bp, url_prefix="/notes")

    @app.route("/health")
    def health():
        return jsonify(status="ok"), 200

    return app
