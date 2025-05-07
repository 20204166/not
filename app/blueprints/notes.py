# app/blueprints/notes.py

import os
import numpy as np
import speech_recognition as sr
from flask import Blueprint, request, jsonify, current_app
from tensorflow.keras.preprocessing.sequence import pad_sequences

notes_bp = Blueprint("notes", __name__)

@notes_bp.route("/process", methods=["POST"])
def process_note():
    # pull everything out of your properly-loaded config
    model       = current_app.config.get("SUMMARY_MODEL")
    tok_in      = current_app.config.get("TOK_INPUT")
    tok_targ    = current_app.config.get("TOK_TARGET")
    max_in      = current_app.config.get("MAX_LENGTH_INPUT")
    max_out     = current_app.config.get("MAX_LENGTH_TARGET")
    start_i     = current_app.config.get("START_TOKEN_INDEX")
    end_i       = current_app.config.get("END_TOKEN_INDEX")

    if not all([model, tok_in, tok_targ]):
        return jsonify(error="Model or tokenizers missing"), 500

    # 1) Audio branch
    if "audio_file" in request.files:
        f = request.files["audio_file"]
        tmp_dir = "tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, f.filename)
        f.save(tmp_path)

        recog = sr.Recognizer()
        with sr.AudioFile(tmp_path) as src:
            audio = recog.record(src)
        try:
            text = recog.recognize_google(audio)
        except Exception:
            text = ""
        finally:
            os.remove(tmp_path)

    # 2) JSON/text branch
    else:
        payload = request.get_json(silent=True) or {}
        text = payload.get("text_input", "")
        if not text:
            return jsonify(error="No input provided"), 400

    # 3) Greedy decode
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
