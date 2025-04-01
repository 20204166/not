import os
from flask import Blueprint, request, jsonify, current_app
import speech_recognition as sr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# Create the Blueprint for note processing
notes_bp = Blueprint('notes', __name__)

# Paths for the saved model and tokenizers.
MODEL_PATH = "app/models/saved_model/summarization_model.h5"
TOKENIZER_INPUT_PATH = "app/models/saved_model/tokenizer_input.json"
TOKENIZER_TARGET_PATH = "app/models/saved_model/tokenizer_target.json"
TRAINING_DATA_PATH = "app/models/data/text/training_data.json"  # Assumed location of your training data

# Load the custom summarization model and tokenizers.
try:
    summarization_model = tf.keras.models.load_model(MODEL_PATH)
    current_app.logger.info("Custom summarization model loaded.")
except Exception as e:
    current_app.logger.error("Error loading summarization model: %s", e)
    summarization_model = None

def load_tokenizer(tokenizer_path: str):
    """
    Load a Keras Tokenizer from a JSON file.
    """
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_json = f.read()
    return tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

try:
    tokenizer_input = load_tokenizer(TOKENIZER_INPUT_PATH)
    tokenizer_target = load_tokenizer(TOKENIZER_TARGET_PATH)
    current_app.logger.info("Tokenizers loaded successfully.")
except Exception as e:
    current_app.logger.error("Error loading tokenizers: %s", e)
    tokenizer_input = None
    tokenizer_target = None

# Define sequence length parameters (must match training configuration)
MAX_LENGTH_INPUT = 50
MAX_LENGTH_TARGET = 20

def generate_summary_inference(input_text: str, max_length_input: int = MAX_LENGTH_INPUT, max_length_target: int = MAX_LENGTH_TARGET) -> str:
    """
    Generate a summary for the given text using the custom summarization model with greedy decoding.
    """
    if summarization_model is None or tokenizer_input is None or tokenizer_target is None:
        return "Summarization model or tokenizers not available."
    
    # Preprocess the input text.
    seq = tokenizer_input.texts_to_sequences([input_text])
    encoder_input = pad_sequences(seq, maxlen=max_length_input, padding='post')
    
    # Initialize the target sequence with the start token (assumed index 1).
    target_seq = np.array([[1]])
    
    summary_generated = []
    
    for _ in range(max_length_target):
        predictions = summarization_model.predict([encoder_input, target_seq], verbose=0)
        sampled_token_index = np.argmax(predictions[0, -1, :])
        if sampled_token_index == 2:  # End token index (assumed to be 2)
            break
        sampled_word = tokenizer_target.index_word.get(sampled_token_index, "")
        if not sampled_word:
            break
        summary_generated.append(sampled_word)
        target_seq = np.concatenate([target_seq, np.array([[sampled_token_index]])], axis=1)
    
    return " ".join(summary_generated)

def speech_to_text(file_path: str) -> str:
    """
    Convert an audio file to text using the SpeechRecognition library.
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Unable to transcribe audio: unclear input."
    except sr.RequestError as e:
        current_app.logger.error("Speech-to-text service error: %s", e)
        return "Speech-to-text service unavailable."

def load_training_data(data_path: str):
    """
    Load training data from a JSON file.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if len(data) == 0 or not isinstance(data[0], dict):
        raise ValueError("Training data must be a non-empty list of objects.")
    if "article" in data[0] and "highlights" in data[0]:
        input_texts = [item["article"] for item in data]
        target_texts = [item["highlights"] for item in data]
    elif "text" in data[0] and "summary" in data[0]:
        input_texts = [item["text"] for item in data]
        target_texts = [item["summary"] for item in data]
    else:
        raise ValueError("Training data must contain keys 'article'/'highlights' or 'text'/'summary'.")
    target_texts = [f"<start> {summary} <end>" for summary in target_texts]
    return input_texts, target_texts

def create_dataset(input_texts, target_texts, batch_size, tokenizer_input, tokenizer_target):
    """
    Creates a tf.data.Dataset that yields ((encoder_input, decoder_input), decoder_target)
    for each sample.
    """
    dataset = tf.data.Dataset.from_tensor_slices((input_texts, target_texts))
    
    def process_sample(input_text, target_text):
        def _process_sample(input_text_str, target_text_str):
            input_str = input_text_str.numpy().decode('utf-8')
            target_str = target_text_str.numpy().decode('utf-8')
            encoder_seq = tokenizer_input.texts_to_sequences([input_str])[0]
            decoder_seq = tokenizer_target.texts_to_sequences([target_str])[0]
            encoder_seq = pad_sequences([encoder_seq], maxlen=MAX_LENGTH_INPUT, padding='post')[0]
            decoder_input_seq = pad_sequences([decoder_seq], maxlen=MAX_LENGTH_TARGET, padding='post')[0]
            decoder_target_seq = np.zeros_like(decoder_input_seq)
            decoder_target_seq[:-1] = decoder_input_seq[1:]
            decoder_target_seq[-1] = 0
            return encoder_seq.astype(np.int32), decoder_input_seq.astype(np.int32), decoder_target_seq.astype(np.int32)
        
        encoder_seq, decoder_input_seq, decoder_target_seq = tf.py_function(
            _process_sample, [input_text, target_text],
            [tf.int32, tf.int32, tf.int32]
        )
        encoder_seq.set_shape([MAX_LENGTH_INPUT])
        decoder_input_seq.set_shape([MAX_LENGTH_TARGET])
        decoder_target_seq.set_shape([MAX_LENGTH_TARGET])
        return (encoder_seq, decoder_input_seq), decoder_target_seq

    dataset = dataset.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

@notes_bp.route('/process', methods=['POST'])
def process_note():
    """
    Process note-taking requests by converting speech to text (if an audio file is provided)
    and summarizing the resulting text.
    """
    transcription = ""
    
    if 'audio_file' in request.files:
        audio_file = request.files['audio_file']
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, audio_file.filename)
        audio_file.save(temp_path)
        transcription = speech_to_text(temp_path)
        os.remove(temp_path)
    else:
        data = request.get_json(silent=True)
        if not data or 'text_input' not in data:
            return jsonify({'error': 'Missing audio_file or text_input parameter'}), 400
        transcription = data['text_input']
    
    summary = generate_summary_inference(transcription)
    
    response = {
        'transcription': transcription,
        'summary': summary
    }
    return jsonify(response), 200

# --- New Endpoint for Model Evaluation ---
@notes_bp.route('/evaluate', methods=['GET'])
def evaluate_model():
    """
    Evaluate the model on a hold-out validation set (using a 90/10 split from the training data)
    and return the token-level accuracy.
    """
    try:
        input_texts, target_texts = load_training_data(TRAINING_DATA_PATH)
    except Exception as e:
        current_app.logger.error("Error loading training data for evaluation: %s", e)
        return jsonify({'error': 'Failed to load training data.'}), 500

    split_index = int(len(input_texts) * 0.9)
    val_inputs = input_texts[split_index:]
    val_targets = target_texts[split_index:]
    val_dataset = create_dataset(val_inputs, val_targets, batch_size=32, tokenizer_input=tokenizer_input, tokenizer_target=tokenizer_target)
    
    all_predictions = []
    all_true = []
    for (encoder_inputs, decoder_inputs), decoder_targets in val_dataset:
        predictions = summarization_model.predict([encoder_inputs, decoder_inputs], verbose=0)
        predicted_indices = np.argmax(predictions, axis=-1)
        all_predictions.extend(predicted_indices.tolist())
        all_true.extend(decoder_targets.numpy().tolist())
    
    total_tokens = 0
    correct_tokens = 0
    for pred_seq, true_seq in zip(all_predictions, all_true):
        for p, t in zip(pred_seq, true_seq):
            if t != 0:
                total_tokens += 1
                if p == t:
                    correct_tokens += 1
    val_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    current_app.logger.info("Validation Token Accuracy: %.4f", val_accuracy)
    return jsonify({"validation_token_accuracy": val_accuracy}), 200

if __name__ == "__main__":
    # For testing purposes, run a simple Flask app.
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(notes_bp, url_prefix='/notes')
    app.run(debug=True)
