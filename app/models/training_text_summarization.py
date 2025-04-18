import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import tensorflow as tf

# 1) Discover and configure GPUs before any TF ops
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled memory growth on GPUs:", gpus)
    except RuntimeError as e:
        print("Error setting GPU memory growth:", e)
else:
    print("No GPUs found; using CPU.")

# 2) Enable XLA now that memory growth is set
tf.config.optimizer.set_jit(True)

# 3) Safe to do other TF operations
print("TensorFlow version:", tf.__version__)
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
print("Operation result shape:", c.shape)
os.system("nvidia-smi")

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Attention, LSTMCell
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam

import json
import numpy as np
import matplotlib.pyplot as plt

# Global settings for maximum sequence lengths
max_length_input = 50
max_length_target = 20

def load_training_data(data_path: str):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not data or not isinstance(data[0], dict):
        raise ValueError("Training data must be a non-empty list of objects.")
    if "article" in data[0] and "highlights" in data[0]:
        inputs = [item["article"] for item in data]
        targets = [item["highlights"] for item in data]
    elif "text" in data[0] and "summary" in data[0]:
        inputs = [item["text"] for item in data]
        targets = [item["summary"] for item in data]
    else:
        raise ValueError("Training data must contain 'article'/'highlights' or 'text'/'summary'.")
    # Wrap target texts
    targets = [f"<start> {t} <end>" for t in targets]
    return inputs, targets

def create_tokenizer(texts, oov_token="<OOV>"):
    tok = Tokenizer(oov_token=oov_token)
    tok.fit_on_texts(texts)
    return tok

def load_tokenizer(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return tf.keras.preprocessing.text.tokenizer_from_json(f.read())

def create_dataset(input_texts, target_texts, batch_size, tok_in, tok_tgt):
    dataset = tf.data.Dataset.from_tensor_slices((input_texts, target_texts))
    def _process(input_str, target_str):
        inp = input_str.numpy().decode('utf-8')
        tgt = target_str.numpy().decode('utf-8')
        enc_seq = tok_in.texts_to_sequences([inp])[0]
        dec_seq = tok_tgt.texts_to_sequences([tgt])[0]
        enc_seq = pad_sequences([enc_seq], maxlen=max_length_input, padding='post')[0]
        dec_in_seq = pad_sequences([dec_seq], maxlen=max_length_target, padding='post')[0]
        # Shift for decoder target
        dec_tgt_seq = np.zeros_like(dec_in_seq)
        dec_tgt_seq[:-1] = dec_in_seq[1:]
        return (enc_seq.astype(np.int32), dec_in_seq.astype(np.int32)), dec_tgt_seq.astype(np.int32)
    def _tf_wrap(inp, tgt):
        return tf.py_function(_process, [inp, tgt], [tf.int32, tf.int32, tf.int32])
    dataset = dataset.map(_tf_wrap, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def build_seq2seq_model(vocab_in, vocab_tgt, emb_dim, max_in, max_tgt):
    # Encoder
    enc_inputs = Input(shape=(max_in,), name="enc_inputs")
    enc_emb = Embedding(vocab_in, emb_dim, name="enc_emb")(enc_inputs)
    enc_cell1 = LSTMCell(64, name="enc_cell1")
    enc_rnn1 = tf.keras.layers.RNN(enc_cell1, return_sequences=True, return_state=True, name="enc_rnn1")
    out1, h1, c1 = enc_rnn1(enc_emb)
    enc_cell2 = LSTMCell(64, name="enc_cell2")
    enc_rnn2 = tf.keras.layers.RNN(enc_cell2, return_sequences=True, return_state=True, name="enc_rnn2")
    enc_outs, h2, c2 = enc_rnn2(out1)
    enc_states = [h2, c2]

    # Decoder
    dec_inputs = Input(shape=(max_tgt,), name="dec_inputs")
    dec_emb = Embedding(vocab_tgt, emb_dim, name="dec_emb")(dec_inputs)
    dec_cell1 = LSTMCell(64, name="dec_cell1")
    dec_rnn1 = tf.keras.layers.RNN(dec_cell1, return_sequences=True, return_state=True, name="dec_rnn1")
    dec_out1, _, _ = dec_rnn1(dec_emb, initial_state=enc_states)
    dec_cell2 = LSTMCell(64, name="dec_cell2")
    dec_rnn2 = tf.keras.layers.RNN(dec_cell2, return_sequences=True, return_state=True, name="dec_rnn2")
    dec_out2, _, _ = dec_rnn2(dec_out1)

    # Attention & output
    attn = Attention(name="attn_layer")([dec_out2, enc_outs])
    concat = Concatenate(name="concat_layer")([attn, dec_out2])
    outputs = Dense(vocab_tgt, activation='softmax', name="decoder_dense")(concat)

    model = Model([enc_inputs, dec_inputs], outputs)
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0005,
        decay_steps=20000,
        decay_rate=0.98,
        staircase=True
    )
    model.compile(Adam(lr_sched), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(hist, save_dir):
    epochs = range(1, len(hist.history['loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, hist.history['loss'], 'bo-', label='Loss')
    plt.plot(epochs, hist.history['accuracy'], 'go-', label='Accuracy')
    plt.title('Training Metrics')
    plt.xlabel('Epochs')
    plt.legend()
    out_path = os.path.join(save_dir, "training_progress.png")
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

class CustomEval(Callback):
    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds
    def on_epoch_end(self, epoch, logs=None):
        total, correct = 0, 0
        for (enc_in, dec_in), dec_tgt in self.val_ds:
            preds = self.model.predict([enc_in, dec_in])
            idx = np.argmax(preds, axis=-1)
            mask = dec_tgt != 0
            correct += np.sum((idx == dec_tgt) & mask)
            total += np.sum(mask)
        print(f"Validation Token Accuracy: {correct/total:.4f}")

def train_model(data_path, epochs=10, batch_size=64, emb_dim=50):
    inputs, targets = load_training_data(data_path)
    split = int(0.9 * len(inputs))
    train_in, train_tgt = inputs[:split], targets[:split]
    val_in, val_tgt = inputs[split:], targets[split:]

    tok_in_path = "app/models/saved_model/tokenizer_input.json"
    tok_tgt_path = "app/models/saved_model/tokenizer_target.json"
    if os.path.exists(tok_in_path) and os.path.exists(tok_tgt_path):
        tok_in = load_tokenizer(tok_in_path)
        tok_tgt = load_tokenizer(tok_tgt_path)
    else:
        tok_in = create_tokenizer(inputs)
        tok_tgt = create_tokenizer(targets)

    vs_in = len(tok_in.word_index) + 1
    vs_tgt = len(tok_tgt.word_index) + 1

    train_ds = create_dataset(train_in, train_tgt, batch_size, tok_in, tok_tgt)
    val_ds   = create_dataset(val_in,   val_tgt,   batch_size, tok_in, tok_tgt)

    model_path = "app/models/saved_model/summarization_model.keras"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        print("Loading model from disk")
        model = load_model(model_path)
    else:
        model = build_seq2seq_model(vs_in, vs_tgt, emb_dim, max_length_input, max_length_target)

    callbacks = [
        EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True, verbose=1),
        CustomEval(val_ds)
    ]

    history = model.fit(train_ds, epochs=epochs, verbose=2, callbacks=callbacks, validation_data=val_ds)

    # Save tokenizers
    with open(tok_in_path, 'w', encoding='utf-8') as f:
        f.write(tok_in.to_json())
    with open(tok_tgt_path, 'w', encoding='utf-8') as f:
        f.write(tok_tgt.to_json())

    plot_history(history, os.path.dirname(model_path))
    return model

if __name__ == "__main__":
    model = train_model("app/models/data/text/training_data.json")
    print("Training complete.")
