from flask import Blueprint, request, jsonify, current_app
import os, numpy as np, speech_recognition as sr
from tensorflow.keras.preprocessing.sequence import pad_sequences

notes_bp = Blueprint("notes", __name__)

@notes_bp.route("/process", methods=["POST"])
def process_note():
    # 1) Load the configured objects
    model = current_app.config.get("SUMMARY_MODEL")
    tok_in = current_app.config.get("TOK_INPUT")
    tok_targ = current_app.config.get("TOK_TARGET")
    max_in  = current_app.config.get("MAX_LENGTH_INPUT")
    max_out = current_app.config.get("MAX_LENGTH_TARGET")
    start_i = current_app.config.get("START_TOKEN_INDEX", 1)
    end_i   = current_app.config.get("END_TOKEN_INDEX", 2)

    if not all([model, tok_in, tok_targ]):
        return jsonify(error="Model or tokenizers not available"), 500

    # 2) Accept either audio_file or text_input
    if "audio_file" in request.files:
        f = request.files["audio_file"]
        tmp = os.path.join("tmp", f.filename)
        os.makedirs("tmp", exist_ok=True)
        f.save(tmp)
        text = _speech_to_text(tmp)
        os.remove(tmp)
    else:
        data = request.get_json(silent=True) or {}
        text = data.get("text_input", "")
        if not text:
            return jsonify(error="No input provided"), 400

    # 3) Run your greedy decoding loop
    seq = tok_in.texts_to_sequences([text])
    enc_in = pad_sequences(seq, maxlen=max_in, padding="post")
    dec_seq = np.array([[start_i]])
    result = []

    for _ in range(max_out):
        preds = model.predict([enc_in, dec_seq], verbose=0)
        idx = np.argmax(preds[0, -1, :])
        if idx == end_i:
            break
        word = tok_targ.index_word.get(idx, "")
        if not word:
            break
        result.append(word)
        dec_seq = np.concatenate([dec_seq, [[idx]]], axis=1)

    return jsonify(transcription=text, summary=" ".join(result)), 200

# reuse your old function
def _speech_to_text(path):
    recog = sr.Recognizer()
    with sr.AudioFile(path) as src:
        audio = recog.record(src)
    try:
        return recog.recognize_google(audio)
    except:
        return ""
