from flask import Blueprint, request, jsonify, current_app
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

notes_bp = Blueprint("notes", __name__)

@notes_bp.route("/process", methods=["POST"])
def process_note():
    # 1) grab everything from config
    model    = current_app.config["SUMMARY_MODEL"]
    tok_in   = current_app.config["TOK_INPUT"]
    tok_targ = current_app.config["TOK_TARGET"]
    max_in   = current_app.config["MAX_LENGTH_INPUT"]
    max_out  = current_app.config["MAX_LENGTH_TARGET"]
    start_i  = current_app.config["START_TOKEN_INDEX"]
    end_i    = current_app.config["END_TOKEN_INDEX"]

    if not all([model, tok_in, tok_targ]):
        return jsonify(error="Model or tokenizers not loaded"), 500

    # 2) pull text out
    data = request.get_json(silent=True) or {}
    text = data.get("text_input", "").strip()
    if not text:
        return jsonify(error="No input provided"), 400

    # 3) tokenize and (always) pad encoder input
    seqs = tok_in.texts_to_sequences([text])
    # if the sequence is empty, force it to at least one OOV token
    if not seqs or not seqs[0]:
        oov_idx = tok_in.word_index.get(tok_in.oov_token, 1)
        seqs = [[oov_idx]]

    enc_in  = pad_sequences(seqs, maxlen=max_in, padding="post", dtype="int32")

    # 4) initialize decoder sequence (start token) and result buffer
    dec_seq = np.array([[start_i]], dtype="int32")
    result  = []

    
    # 5) Summarize via the HF pipeline you registered on startup
    summarizer = current_app.config.get("SUMMARIZER")
    if summarizer is None:
       return jsonify(error="No summarizer available"), 500
    # call the pipeline, respecting your max length
    out = summarizer(
        text,
        max_length=max_out,
        min_length=5,
        do_sample=False
    )
    # pipeline returns [{"summary_text": "..."}]
    summary_text = out[0]["summary_text"]
    
    return jsonify(transcription=text, summary=summary_text), 200


@notes_bp.route("/evaluate", methods=["POST"])
def evaluate_summary():
    data = request.get_json() or {}
    summary  = data.get("summary", "")
    original = data.get("original", "")
    if not summary or not original:
        return jsonify(error="Must provide both summary and original text"), 400

    # TODO: replace with your actual evaluation logic
    results = {
      "rouge1": 0.42,
      "rouge2": 0.17,
      "bert_score": 0.88
    }
    return jsonify(results), 200
