import os
import tensorflow as tf
from flask import Flask, jsonify

def load_tokenizer(path: str):
    """Load a Keras tokenizer from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return tf.keras.preprocessing.text.tokenizer_from_json(f.read())

def create_app():
    app = Flask(__name__, instance_relative_config=False)

    # ─── Load the right config class ───────────────────────────────────────
    # FLASK_ENV should be one of 'development', 'testing', or 'production'
    env = os.environ.get("FLASK_ENV", "development").lower()
    # Map to your class names: DevelopmentConfig, TestingConfig, ProductionConfig
    config_class = {
        "development": "app.config.DevelopmentConfig",
        "testing":     "app.config.TestingConfig",
        "production":  "app.config.ProductionConfig",
    }.get(env, "app.config.Config")
    app.config.from_object(config_class)

    # ─── Bring numeric settings into config for the blueprint ─────────────
    app.config["MAX_LENGTH_INPUT"]  = int(app.config.get("MAX_LENGTH_INPUT", 50))
    app.config["MAX_LENGTH_TARGET"] = int(app.config.get("MAX_LENGTH_TARGET", 20))
    app.config["START_TOKEN_INDEX"] = int(app.config.get("START_TOKEN_INDEX", 1))
    app.config["END_TOKEN_INDEX"]   = int(app.config.get("END_TOKEN_INDEX", 2))

    # ─── Load your model & tokenizers ──────────────────────────────────────
    try:
        # Paths from your config.py
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

    # ─── Register Blueprints & Health ─────────────────────────────────────
    from app.blueprints.notes import notes_bp
    app.register_blueprint(notes_bp, url_prefix="/notes")

    @app.route("/health")
    def health():
        return jsonify(status="ok"), 200

    return app

# so run.py can do: from app import create_app; app = create_app()
