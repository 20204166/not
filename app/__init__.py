# app/__init__.py
import os
import tensorflow as tf
from flask import Flask, jsonify


def load_tokenizer(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return tf.keras.preprocessing.text.tokenizer_from_json(f.read())

def create_app():
    app = Flask(__name__, instance_relative_config=False)

    # load your base config class
    cfg_path = os.environ.get("APP_CONFIG", "app.config.Config")
    app.config.from_object(cfg_path)

    # load model + tokenizers
        # ─── Tell your blueprint how long to pad/truncate for inference ───────────
    app.config["MAX_LENGTH_INPUT"]  = int(app.config.get("MAX_LENGTH_INPUT", 50))
    app.config["MAX_LENGTH_TARGET"] = int(app.config.get("MAX_LENGTH_TARGET", 20))

    model_path  = app.config["MODEL_PATH"]
    tok_in_path = app.config["TOKENIZER_INPUT_PATH"]
    tok_out_path= app.config["TOKENIZER_TARGET_PATH"]

    summarization_model = tf.keras.models.load_model(model_path)
    tok_input     = load_tokenizer(tok_in_path)
    tok_target    = load_tokenizer(tok_out_path)
    # ─── Build a Hugging-Face summarization pipeline ─────────────────────────
    from transformers import pipeline
    summarizer = pipeline(
        "summarization",
        model=summarization_model,
        tokenizer=tok_input,
        framework="tf",
        device=-1            # CPU only
    )
    app.config["SUMMARIZER"] = summarizer



    # in process_note()
    widx = tok_target.word_index
    if "<start>" in widx:
        start_i = widx["<start>"]
        end_i   = widx["<end>"]
    else:
        start_i = widx["start"]
        end_i   = widx["end"]
    


    app.config.update({
        "SUMMARY_MODEL":     summarization_model,
        "TOK_INPUT":         tok_input,
        "TOK_TARGET":        tok_target,
        "START_TOKEN_INDEX": start_i,
        "END_TOKEN_INDEX":   end_i,
    })



 
    # register blueprint
    from app.blueprints.notes import notes_bp
    app.register_blueprint(notes_bp, url_prefix="/notes")

    @app.route("/health")
    def health():
        return jsonify(status="ok"), 200

    return app
