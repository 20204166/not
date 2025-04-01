import os
# Force all operations to run on CPU by disabling all GPUs.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import numpy as np
import matplotlib.pyplot as plt  # Ensure matplotlib is installed
import re  # For cleaning text if needed
import tensorflow as tf

# Enable XLA (Accelerated Linear Algebra) to optimize and fuse operations.
tf.config.optimizer.set_jit(True)

# Adjust threading to balance workload on CPU:
num_threads = os.cpu_count() or 6 # Fallback to 6 if os.cpu_count() returns None
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Attention, LSTMCell
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam

print("Running on CPU only.")

# Global settings for maximum sequence lengths.
max_length_input = 50
max_length_target = 20

def load_training_data(data_path: str):
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
    # Wrap target texts with start/end tokens.
    target_texts = [f"<start> {summary} <end>" for summary in target_texts]
    return input_texts, target_texts

def create_tokenizer(texts: list, oov_token="<OOV>"):
    tokenizer = Tokenizer(oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_json = f.read()
    return tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

def create_dataset(input_texts, target_texts, batch_size, tokenizer_input, tokenizer_target):
    """
    Creates a tf.data.Dataset that yields ((encoder_input, decoder_input), decoder_target)
    for each sample.
    """
    dataset = tf.data.Dataset.from_tensor_slices((input_texts, target_texts))
    
    def process_sample(input_text, target_text):
        """
        Tokenize and pad a single sample, and create the decoder target by shifting.
        """
        def _process_sample(input_text_str, target_text_str):
            # Convert EagerTensors to numpy then decode.
            input_str = input_text_str.numpy().decode('utf-8')
            target_str = target_text_str.numpy().decode('utf-8')
            # Convert to sequences using the loaded tokenizers.
            encoder_seq = tokenizer_input.texts_to_sequences([input_str])[0]
            decoder_seq = tokenizer_target.texts_to_sequences([target_str])[0]
            # Pad sequences.
            encoder_seq = pad_sequences([encoder_seq], maxlen=max_length_input, padding='post')[0]
            decoder_input_seq = pad_sequences([decoder_seq], maxlen=max_length_target, padding='post')[0]
            # Create decoder target sequence: shift left by one.
            decoder_target_seq = np.zeros_like(decoder_input_seq)
            decoder_target_seq[:-1] = decoder_input_seq[1:]
            decoder_target_seq[-1] = 0
            return encoder_seq.astype(np.int32), decoder_input_seq.astype(np.int32), decoder_target_seq.astype(np.int32)
        
        encoder_seq, decoder_input_seq, decoder_target_seq = tf.py_function(
            _process_sample, [input_text, target_text],
            [tf.int32, tf.int32, tf.int32]
        )
        # Set static shapes.
        encoder_seq.set_shape([max_length_input])
        decoder_input_seq.set_shape([max_length_target])
        decoder_target_seq.set_shape([max_length_target])
        return (encoder_seq, decoder_input_seq), decoder_target_seq

    dataset = dataset.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def build_seq2seq_model(vocab_size_input: int, vocab_size_target: int,
                        embedding_dim: int, max_length_input: int, max_length_target: int) -> tf.keras.Model:
    # Encoder
    encoder_inputs = Input(shape=(max_length_input,), name="encoder_inputs")
    enc_emb = Embedding(vocab_size_input, embedding_dim, name="encoder_embedding")(encoder_inputs)
    encoder_lstm_cell1 = LSTMCell(64, name="encoder_lstm_cell1")
    encoder_rnn1 = tf.keras.layers.RNN(encoder_lstm_cell1, return_sequences=True, return_state=True, name="encoder_rnn1")
    encoder_outputs1, state_h1, state_c1 = encoder_rnn1(enc_emb)
    encoder_lstm_cell2 = LSTMCell(64, name="encoder_lstm_cell2")
    encoder_rnn2 = tf.keras.layers.RNN(encoder_lstm_cell2, return_sequences=True, return_state=True, name="encoder_rnn2")
    encoder_outputs2, state_h2, state_c2 = encoder_rnn2(encoder_outputs1)
    encoder_outputs = encoder_outputs2  # For attention.
    encoder_states = [state_h2, state_c2]

    # Decoder
    decoder_inputs = Input(shape=(max_length_target,), name="decoder_inputs")
    dec_emb = Embedding(vocab_size_target, embedding_dim, name="decoder_embedding")(decoder_inputs)
    decoder_lstm_cell1 = LSTMCell(64, name="decoder_lstm_cell1")
    decoder_rnn1 = tf.keras.layers.RNN(decoder_lstm_cell1, return_sequences=True, return_state=True, name="decoder_rnn1")
    decoder_outputs1, _, _ = decoder_rnn1(dec_emb, initial_state=encoder_states)
    decoder_lstm_cell2 = LSTMCell(64, name="decoder_lstm_cell2")
    decoder_rnn2 = tf.keras.layers.RNN(decoder_lstm_cell2, return_sequences=True, return_state=True, name="decoder_rnn2")
    decoder_outputs2, _, _ = decoder_rnn2(decoder_outputs1)

    # Attention and Dense
    attention_layer = Attention(name="attention_layer")
    context_vector = attention_layer([decoder_outputs2, encoder_outputs])
    decoder_combined_context = Concatenate(axis=-1, name="concat_layer")([context_vector, decoder_outputs2])
    decoder_dense = Dense(vocab_size_target, activation='softmax', name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_combined_context)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # Set a custom learning rate.
    custom_learning_rate = 0.001
    optimizer = Adam(learning_rate=custom_learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history, save_dir: str):
    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'bo-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], 'go-', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_progress.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Training history plot saved to: {plot_path}")

# --- Custom Callback for Additional Evaluation ---
class CustomEvaluationCallback(Callback):
    def __init__(self, val_dataset, tokenizer_target):
        super().__init__()
        self.val_dataset = val_dataset
        self.tokenizer_target = tokenizer_target

    def on_epoch_end(self, epoch, logs=None):
        all_predictions = []
        all_true = []
        for (encoder_inputs, decoder_inputs), decoder_targets in self.val_dataset:
            predictions = self.model.predict([encoder_inputs, decoder_inputs])
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
        print(f"Custom Evaluation - Validation Token Accuracy: {val_accuracy:.4f}")

def train_model(data_path: str, epochs: int = 10,
                max_length_input: int = 50, max_length_target: int = 20,
                embedding_dim: int = 50, force_rebuild: bool = True,
                batch_size: int = 64):
    model_path = "app/models/saved_model/summarization_model.h5"
    tokenizer_input_path = "app/models/saved_model/tokenizer_input.json"
    tokenizer_target_path = "app/models/saved_model/tokenizer_target.json"

    # Load the raw training data.
    input_texts, target_texts = load_training_data(data_path)

    # --- Split data into training and validation sets (90/10 split) ---
    split_index = int(len(input_texts) * 0.9)
    train_inputs = input_texts[:split_index]
    train_targets = target_texts[:split_index]
    val_inputs = input_texts[split_index:]
    val_targets = target_texts[split_index:]

    # Load or create tokenizers to ensure tokenizer consistency.
    if os.path.exists(tokenizer_input_path) and os.path.exists(tokenizer_target_path) and not force_rebuild:
        print("Loading tokenizers from saved files...")
        tokenizer_input = load_tokenizer(tokenizer_input_path)
        tokenizer_target = load_tokenizer(tokenizer_target_path)
    else:
        print("Creating new tokenizers...")
        tokenizer_input = create_tokenizer(input_texts)
        tokenizer_target = create_tokenizer(target_texts)

    vocab_size_input = len(tokenizer_input.word_index) + 1
    vocab_size_target = len(tokenizer_target.word_index) + 1

    # Create tf.data.Dataset pipelines for training and validation.
    train_dataset = create_dataset(train_inputs, train_targets, batch_size, tokenizer_input, tokenizer_target)
    val_dataset = create_dataset(val_inputs, val_targets, batch_size, tokenizer_input, tokenizer_target)

    if os.path.exists(model_path) and not force_rebuild:
        print("Loading previously saved model...")
        model = load_model(model_path)
    else:
        print("Building a new model...")
        model = build_seq2seq_model(vocab_size_input, vocab_size_target, embedding_dim, max_length_input, max_length_target)

    early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_path, monitor='loss', save_best_only=True, verbose=1)
    custom_eval = CustomEvaluationCallback(val_dataset, tokenizer_target)

    print("Training will run on CPU using the tf.data pipeline.")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        verbose=2,
        callbacks=[early_stop, checkpoint, custom_eval],
        validation_data=val_dataset
    )

    # Save tokenizers to JSON.
    with open(tokenizer_input_path, "w", encoding="utf-8") as f:
        f.write(tokenizer_input.to_json())
    with open(tokenizer_target_path, "w", encoding="utf-8") as f:
        f.write(tokenizer_target.to_json())

    plot_training_history(history, os.path.dirname(model_path))

    return model, tokenizer_input, tokenizer_target, history

if __name__ == "__main__":
    training_data_path = "app/models/data/text/training_data.json"  # Your training data file.
    model, tokenizer_input, tokenizer_target, history = train_model(
        training_data_path,
        epochs=30,
        force_rebuild=True,
        batch_size=64
    )
    print("Model training complete. Model saved to app/models/saved_model/summarization_model.h5")
    print("Tokenizers saved to app/models/saved_model/tokenizer_input.json and app/models/saved_model/tokenizer_target.json")
    print("Training history and plots saved.")
