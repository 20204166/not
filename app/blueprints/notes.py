import os
import numpy as np
from flask import Blueprint, request, jsonify, current_app
import speech_recognition as sr
from tensorflow.keras.preprocessing.sequence import pad_sequences

notes_bp = Blueprint('notes', __name__)

# Constants: must match your training config
MAX_INPUT_LEN  = 50
MAX_TARGET_LEN = 20
START_TOKEN_IDX = 1
END_TOKEN_IDX   = 2

def speech_to_text(file_path: str) -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as src:
        audio = recognizer.record(src)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""

def generate_summary(text: str) -> str:
    model = current_app.config['SUMMARY_MODEL']
    tok_in  = current_app.config['TOK_INPUT']
    tok_tar = current_app.config['TOK_TARGET']
    if not all([model, tok_in, tok_tar]):
        return "Model or tokenizers not available."

    seq = tok_in.texts_to_sequences([text])
    encoder_input = pad_sequences(seq, maxlen=MAX_INPUT_LEN, padding='post')

    # Start with the <start> token
    target_seq = np.array([[START_TOKEN_IDX]])
    result = []

    for _ in range(MAX_TARGET_LEN):
        preds = model.predict([encoder_input, target_seq], verbose=0)
        sampled_idx = np.argmax(preds[0, -1, :])
        if sampled_idx == END_TOKEN_IDX:
            break
        word = tok_tar.index_word.get(sampled_idx, "")
        if not word:
            break
        result.append(word)
        target_seq = np.concatenate([target_seq, [[sampled_idx]]], axis=1)

    return " ".join(result)

@notes_bp.route('/process', methods=['POST'])
def process_note():
    # 1) Get either uploaded audio or text_input
    if 'audio_file' in request.files:
        f = request.files['audio_file']
        tmp = os.path.join('tmp', f.filename)
        os.makedirs('tmp', exist_ok=True)
        f.save(tmp)
        text = speech_to_text(tmp)
        os.remove(tmp)
    else:
        data = request.get_json(silent=True) or {}
        text = data.get('text_input', '')
        if not text:
            return jsonify(error='No audio_file or text_input'), 400

    # 2) Run summarization
    summary = generate_summary(text)
    return jsonify(transcription=text, summary=summary), 200

@notes_bp.route('/evaluate', methods=['GET'])
def evaluate_model():
    # (Optional) re-introduce your load_training_data + create_dataset logic here
    # For now, just sanity-check the model by summarizing a fixed prompt
    sample = "This is a quick test of the evaluation endpoint."
    return jsonify(test_summary=generate_summary(sample)), 200
