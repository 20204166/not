from flask import Blueprint, request, jsonify, current_app, abort
from tensorflow.keras.preprocessing.sequence import pad_sequences

notes_bp = Blueprint("notes", __name__)


def get_summarizer():
    """Retrieve the HF summarization pipeline from app config."""
    summarizer = current_app.config.get("SUMMARIZER")
    if not summarizer:
        current_app.logger.error("Summarizer not configured")
        abort(500, "Summarizer unavailable")
    return summarizer


def extract_text():
    """Parse and validate incoming JSON for the `/process` endpoint."""
    data = request.get_json(silent=True) or {}
    text = data.get("text_input", "").strip()
    if not text:
        abort(400, "No input provided")
    return text


def summarize_text(text: str) -> str:
    """Run the HF pipeline and return the generated summary."""
    pipe = get_summarizer()
    try:
        out = pipe(
            text,
            max_length=current_app.config["MAX_LENGTH_TARGET"],
            min_length=int(current_app.config.get("MIN_LENGTH_TARGET", 5)),
            do_sample=False
        )
        return out[0].get("summary_text", "")
    except Exception as e:
        current_app.logger.error(f"Summarization error: {e}")
        abort(500, "Error during summarization")


@notes_bp.route("/process", methods=["POST"])
def process_note():
    """
    1) Extract and validate input text.
    2) Summarize via HF pipeline.
    3) Return a consistent JSON envelope.
    """
    text = extract_text()
    summary = summarize_text(text)
    return jsonify({
        "status": "success",
        "data": {
            "transcription": text,
            "summary": summary
        }
    }), 200


@notes_bp.route("/evaluate", methods=["POST"])
def evaluate_summary():
    """
    1) Parse and validate both `summary` and `original`.
    2) (Stub) compute metrics.
    3) Return them in the same envelope style.
    """
    data = request.get_json(silent=True) or {}
    summary = data.get("summary", "").strip()
    original = data.get("original", "").strip()
    if not summary or not original:
        abort(400, "Both summary and original are required")

    # TODO: replace with real metric calculations (e.g., rouge, bert-score)
    results = {
        "rouge1":    0.42,
        "rouge2":    0.17,
        "bert_score": 0.88
    }

    return jsonify({
        "status": "success",
        "data": results
    }), 200
