from flask import Flask, jsonify
import os

def create_app():
    app = Flask(__name__)

    #  ─── Load Config ─────────────────────────────────────────────────────
    # Choose the config class based on FLASK_ENV (or an env var of your choosing)
    env = os.environ.get("FLASK_ENV", "development").title() + "Config"
    app.config.from_object(f"app.config.{env}")

    #  ─── Initialize Logging / DB / etc. ──────────────────────────────────
    # e.g. set logging level from app.config["LOGGING_LEVEL"]

    #  ─── Load your summarization model & tokenizers ──────────────────────
    import tensorflow as tf
    def load_tokenizer(path):
        with open(path, 'r', encoding='utf-8') as f:
            return tf.keras.preprocessing.text.tokenizer_from_json(f.read())

    model_path = app.config["MODEL_PATH"]
    tok_in_p  = app.config["TOKENIZER_INPUT_PATH"]
    tok_targ  = app.config["TOKENIZER_TARGET_PATH"]

    try:
        summarization_model = tf.keras.models.load_model(model_path)
        tokenizer_input     = load_tokenizer(tok_in_p)
        tokenizer_target    = load_tokenizer(tok_targ)
        app.logger.info("Model and tokenizers loaded.")
    except Exception as e:
        app.logger.error("Failed loading model/tokenizers: %s", e)
        summarization_model = tokenizer_input = tokenizer_target = None

    app.config["SUMMARY_MODEL"] = summarization_model
    app.config["TOK_INPUT"]     = tokenizer_input
    app.config["TOK_TARGET"]    = tokenizer_target

    #  ─── Register Blueprints ────────────────────────────────────────────
    from app.blueprints.notes import notes_bp
    app.register_blueprint(notes_bp, url_prefix="/notes")

    #  ─── Health Check ────────────────────────────────────────────────────
    @app.route("/health")
    def health():
        return jsonify(status="ok"), 200

    return app
