import os
import json
import tensorflow as tf
from flask import Flask, jsonify

def create_app():
    app = Flask(__name__)

    # --- Load summarization model ---
    model_path = os.path.join(app.root_path, 'models', 'saved_model', 'summarization_model.keras')
    try:
        summarization_model = tf.keras.models.load_model(model_path)
        app.logger.info("Summarization model loaded from %s", model_path)
    except Exception as e:
        app.logger.error("Failed to load summarization model: %s", e)
        summarization_model = None

    # --- Load tokenizers from JSON ---
    def load_tokenizer(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = f.read()
        return tf.keras.preprocessing.text.tokenizer_from_json(data)

    tok_in_path = os.path.join(app.root_path, 'models', 'saved_model', 'tokenizer_input.json')
    tok_tar_path = os.path.join(app.root_path, 'models', 'saved_model', 'tokenizer_target.json')
    try:
        tokenizer_input  = load_tokenizer(tok_in_path)
        tokenizer_target = load_tokenizer(tok_tar_path)
        app.logger.info("Tokenizers loaded.")
    except Exception as e:
        app.logger.error("Failed to load tokenizers: %s", e)
        tokenizer_input = tokenizer_target = None

    # Store on config
    app.config['SUMMARY_MODEL']   = summarization_model
    app.config['TOK_INPUT']       = tokenizer_input
    app.config['TOK_TARGET']      = tokenizer_target

    # Register blueprint
    from app.blueprints.notes import notes_bp
    app.register_blueprint(notes_bp, url_prefix='/notes')

    # Health check
    @app.route('/health')
    def health():
        return jsonify(status='ok'), 200

    return app
