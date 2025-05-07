# app/blueprints/notes.py

from flask import Blueprint, request, jsonify, current_app
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
# … other imports …

notes_bp = Blueprint("notes", __name__)

@notes_bp.route("/process", methods=["POST"])
def process_note():
    model     = current_app.config.get("SUMMARY_MODEL")
    tok_in    = current_app.config.get("TOK_INPUT")
    tok_targ  = current_app.config.get("TOK_TARGET")
    max_in    = current_app.config.get("MAX_LENGTH_INPUT")
    max_out   = current_app.config.get("MAX_LENGTH_TARGET")
    start_i   = current_app.config.get("START_TOKEN_INDEX", 1)
    end_i     = current_app.config.get("END_TOKEN_INDEX", 2)

    if not all([model, tok_in, tok_targ]):
        return jsonify(error="Model/tokenizers missing"), 500

    data = request.get_json(silent=True) or {}
    text = data.get("text_input", "")
    if not text:
        return jsonify(error="No input provided"), 400

    # Greedy decoding loop…
    enc_seq = tok_in.texts_to_sequences([text])
    enc_in  = pad_sequences(enc_seq, maxlen=max_in, padding="post")
    dec_seq = np.array([[start_i]])
    result  = []

    for _ in range(max_out):
        preds = model.predict([enc_in, dec_seq], verbose=0)
        idx   = np.argmax(preds[0, -1, :])
        if idx == end_i:
            break
        word = tok_targ.index_word.get(idx, "")
        if not word:
            break
        result.append(word)
        dec_seq = np.concatenate([dec_seq, [[idx]]], axis=1)

    return jsonify(transcription=text, summary=" ".join(result)), 200
