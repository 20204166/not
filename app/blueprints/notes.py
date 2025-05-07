# app/blueprints/notes.py

import os
import numpy as np
import speech_recognition as sr
from flask import Blueprint, request, jsonify, current_app
from tensorflow.keras.preprocessing.sequence import pad_sequences

notes_bp = Blueprint("notes", __name__)

def _speech_to_text(path: str) -> str:
    """Convert an audio file at `path` into text via Google Web Speech."""
    recog = sr.Recognizer()
    with sr.AudioFile(path) as src:
        audio = recog.record(src)
    try:
        return recog.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        # API was unreachable or unresponsive
        return ""

@notes_bp.route("/process", methods=["POST"])
def process_note():
    # 1) Grab everything out of the config
    model     = current_app.config.get("SUMMARY_MODEL")
    tok_in    = current_app.config.get("TOK_INPUT")
    tok_targ  = current_app.config.get("TOK_TARGET")
    max_in    = current_app.config.get("MAX_LENGTH_INPUT")
    max_out   = current_app.config.get("MAX_LENGTH_TARGET")
    start_i   = current_app.config.get("START_TOKEN_INDEX")
    end_i     = current_app.config.get("END_TOKEN_INDEX")

    if not all([model, tok_in, tok_targ]):
        return jsonify(error="Model or tokenizers missing"), 500

    # 2) Branch: audio upload vs JSON text
    if "audio_file" in request.files:
        f = request.files["audio_file"]
        os.makedirs("/tmp/uploads", exist_ok=True)
        tmp_path = os.path.join("/tmp/uploads", f.filename)
        f.save(tmp_path)
        text = _speech_to_text(tmp_path)
        os.remove(tmp_path)
    else:
        payload = request.get_json(silent=True) or {}
        text = payload.get("text_input", "").strip()
        if not text:
            return jsonify(error="No input provided"), 400

    # 3) Run greedy decoding
    seq    = tok_in.texts_to_sequences([text])
    enc_in = pad_sequences(seq, maxlen=max_in, padding="post")
    dec_seq = np.array([[start_i]])
    words   = []

    for _ in range(max_out):
        preds = model.predict([enc_in, dec_seq], verbose=0)
        idx   = np.argmax(preds[0, -1, :])
        if idx == end_i:
            break
        w = tok_targ.index_word.get(idx, "")
        if not w:
            break
        words.append(w)
        dec_seq = np.concatenate([dec_seq, [[idx]]], axis=1)

    return jsonify(transcription=text, summary=" ".join(words)), 200
