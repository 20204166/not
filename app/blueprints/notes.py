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

    # 5) greedy decode
    for _ in range(max_out):
        preds = model.predict([enc_in, dec_seq], verbose=0)
        idx   = int(np.argmax(preds[0, -1, :]))
        if idx == end_i:
            break
        word = tok_targ.index_word.get(idx, "")
        if not word:
            break
        result.append(word)
        # append predicted token and keep dtype
        dec_seq = np.concatenate([dec_seq, [[idx]]], axis=1).astype("int32")

    return jsonify(transcription=text, summary=" ".join(result)), 200
