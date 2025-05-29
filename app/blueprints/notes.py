from flask import Blueprint, request, jsonify, current_app, abort
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

notes_bp = Blueprint("notes", __name__)

def extract_text():
    """Extract and validate input text."""
    data = request.get_json(silent=True) or {}
    text = data.get("text_input", "").strip()
    if not text:
        abort(400, "No input provided")
    return text

def summarize_lstm(text: str) -> str:
    """Generate summary using the LSTM model."""
    model = current_app.config["SUMMARIZER"]
    tok_input = current_app.config["TOK_INPUT"]
    tok_target = current_app.config["TOK_TARGET"]
    max_input_len = current_app.config["MAX_INPUT_LEN"]
    max_target_len = current_app.config["MAX_TARGET_LEN"]

    seq = tok_input.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_input_len, padding='post')

    pred = model.predict(np.array(padded))[0]
    output_seq = np.argmax(pred, axis=1)
    output_text = tok_target.sequences_to_texts([output_seq])[0]

    return output_text.strip()

@notes_bp.route("/process", methods=["POST"])
def process_note():
    text = extract_text()
    summary = summarize_lstm(text)
    return jsonify({
        "status": "success",
        "data": {
            "transcription": text,
            "summary": summary
        }
    }), 200

@notes_bp.route("/evaluate", methods=["POST"])
def evaluate_summary():
    data = request.get_json(silent=True) or {}
    summary = data.get("summary", "").strip()
    original = data.get("original", "").strip()
    if not summary or not original:
        abort(400, "Both summary and original are required")

    # Placeholder metrics
    results = {
        "rouge1":    0.42,
        "rouge2":    0.17,
        "bert_score": 0.88
    }

    return jsonify({
        "status": "success",
        "data": results
    }), 200
