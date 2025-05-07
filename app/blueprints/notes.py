from flask import Blueprint, request, jsonify, current_app
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

notes_bp = Blueprint("notes", __name__)

@notes_bp.route("/process", methods=["POST"])
def process_note():
    # 1) grab everything from config
    model   = current_app.config.get("SUMMARY_MODEL")
    tok_in  = current_app.config.get("TOK_INPUT")
    tok_targ= current_app.config.get("TOK_TARGET")
    max_in  = current_app.config.get("MAX_LENGTH_INPUT", 50)
    max_out = current_app.config.get("MAX_LENGTH_TARGET", 20)
    start_i = current_app.config.get("START_TOKEN_INDEX", 1)
    end_i   = current_app.config.get("END_TOKEN_INDEX", 2)

    if not all([model, tok_in, tok_targ]):
        return jsonify(error="Model or tokenizers not loaded"), 500

    # 2) get text
    data = request.get_json(silent=True) or {}
    text = data.get("text_input", "").strip()
    if not text:
        return jsonify(error="No input provided"), 400

    # 3) tokenize & pad the encoder input
    seqs = tok_in.texts_to_sequences([text])
    # if you get back an empty list or None, force it to your OOV index (usually 1)
    if not seqs or seqs[0] is None or len(seqs[0]) == 0:
        oov = getattr(tok_in, "oov_token", None)
        seqs = [[ tok_in.word_index.get(oov, 1) ]]

    enc_in = pad_sequences(
        seqs,
        maxlen=max_in,
        padding="post",
        dtype="int32"        # <–– force numeric dtype
    )

    # 4) initialize the decoder
    dec_seq = np.array([[start_i]], dtype="int32")

    # 5) greedy decode loop
    result = []
    for _ in range(max_out):
        try:
            preds = model.predict([enc_in, dec_seq], verbose=0)
        except Exception as e:
            current_app.logger.error("Prediction failed: %s", e)
            return jsonify(error="Prediction error", details=str(e)), 500

        idx = int(np.argmax(preds[0, -1, :]))
        if idx == end_i:
            break

        word = tok_targ.index_word.get(idx, "")
        if not word:
            break

        result.append(word)
        # append to decoder input & re-cast to int32
        dec_seq = np.concatenate([dec_seq, [[idx]]], axis=1).astype("int32")

    return jsonify(transcription=text, summary=" ".join(result)), 200
