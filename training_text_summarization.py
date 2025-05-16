import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import tensorflow as tf

from tensorflow.keras.mixed_precision import Policy, set_global_policy
set_global_policy(Policy("mixed_float16"))



gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        # first try the “stable” API…
        try:
            tf.config.set_memory_growth(gpu, True)
        except AttributeError:
            # fallback to the experimental API in older TF versions
            tf.config.experimental.set_memory_growth(gpu, True)

    # make sure TensorFlow only “sees” those real GPUs
    tf.config.set_visible_devices(gpus, "GPU")

print("Physical GPUs:", gpus)
print("Logical GPUs:", tf.config.list_logical_devices("GPU"))


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
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model 
from tensorflow.keras.optimizers.schedules import ExponentialDecay  
from rouge_score import rouge_scorer
import psutil
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt




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
    targets = [f"<start> {t} <end>" for t in targets]
    return inputs, targets

MAX_VOCAB = 30_000
def create_tokenizer(texts, oov_token="<OOV>", max_words=MAX_VOCAB, add_special_tokens=True):
    tok = Tokenizer(num_words=max_words, oov_token=oov_token)
    if add_special_tokens:
        texts = [f"<start> {t} <end>" for t in texts]
    tok.fit_on_texts(texts)
    return tok

def load_tokenizer(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return tf.keras.preprocessing.text.tokenizer_from_json(f.read())

def preprocess_texts(texts, tokenizer, max_length, max_vocab):
    sequences = tokenizer.texts_to_sequences(texts)
    arr = pad_sequences(sequences, maxlen=max_length, padding='post',truncating='post')
    arr = np.where(arr >= max_vocab, 1, arr)
    return arr


def prepare_decoder_sequences(sequences):
    dec_in = sequences[:, :-1]
    dec_tgt = sequences[:, 1:]
    pad_width = max_length_target - dec_in.shape[1]
    if pad_width > 0:
        dec_in = np.pad(dec_in, ((0, 0), (0, pad_width)), mode='constant')
        dec_tgt = np.pad(dec_tgt, ((0, 0), (0, pad_width)), mode='constant')
    return dec_in, dec_tgt

def build_seq2seq_model(vocab_in, vocab_tgt, emb_dim, max_in, max_tgt):
    enc_inputs = Input(shape=(max_in,), name="enc_inputs")
    enc_emb = Embedding(vocab_in, emb_dim, name="enc_emb")(enc_inputs)
    enc_cell1 = LSTMCell(64, name="enc_cell1")
    enc_rnn1 = tf.keras.layers.RNN(enc_cell1, return_sequences=True, return_state=True, name="enc_rnn1")
    out1, h1, c1 = enc_rnn1(enc_emb)
    enc_cell2 = LSTMCell(64, name="enc_cell2")
    enc_rnn2 = tf.keras.layers.RNN(enc_cell2, return_sequences=True, return_state=True, name="enc_rnn2")
    enc_outs, h2, c2 = enc_rnn2(out1)
    enc_states = [h2, c2]

    dec_inputs = Input(shape=(max_tgt,), name="dec_inputs")
    dec_emb = Embedding(vocab_tgt, emb_dim, name="dec_emb")(dec_inputs)
    dec_cell1 = LSTMCell(64, name="dec_cell1")
    dec_rnn1 = tf.keras.layers.RNN(dec_cell1, return_sequences=True, return_state=True, name="dec_rnn1")
    dec_out1, _, _ = dec_rnn1(dec_emb, initial_state=enc_states)
    dec_cell2 = LSTMCell(64, name="dec_cell2")
    dec_rnn2 = tf.keras.layers.RNN(dec_cell2, return_sequences=True, return_state=True, name="dec_rnn2")
    dec_out2, _, _ = dec_rnn2(dec_out1)

    attn = Attention(name="attn_layer")([dec_out2, enc_outs])
    concat = Concatenate(name="concat_layer")([attn, dec_out2])
    outputs = Dense(vocab_tgt, activation='softmax', name="decoder_dense", dtype='float32')(concat)

    model = Model([enc_inputs, dec_inputs], outputs)
   
    return model
    
def plot_history(history, save_dir):
    h = history.history
    keys = h.keys()
    epochs = range(1, len(h.get("loss", [])) + 1)

    # 1) Loss
    plt.figure()
    plt.plot(epochs, h["loss"],     label="Train loss")
    plt.plot(epochs, h["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_full.png"))
    plt.close()

    # 2) Accuracy
    acc_key     = "token_accuracy" if "token_accuracy" in keys else "sparse_categorical_accuracy"
    val_acc_key = "val_" + acc_key

    plt.figure()
    plt.plot(epochs, h[acc_key],     label="Train acc")
    plt.plot(epochs, h[val_acc_key], label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs. Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "acc_full.png"))
    plt.close()



class SnapshotCallback(Callback):
    def __init__(self, save_dir, interval_epochs=10):
        super().__init__()
        self.save_dir = save_dir
        self.interval = interval_epochs
        os.makedirs(self.save_dir, exist_ok=True)
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = {}

    def _get_gpu_utils(self):
        try:
            raw = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits"
            ])
            return [float(l) for l in raw.decode().splitlines() if l.strip()]
        except Exception:
            return []

    def _plot_metrics(self, upto, suffix):
        h = self.model.history.history
        upto = min(upto, len(h["loss"]))
        epochs = range(1, upto + 1)

        # 1) Loss
        plt.figure()
        plt.plot(epochs, h["loss"][:upto],     label="Train loss")
        plt.plot(epochs, h["val_loss"][:upto], label="Val loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title(f"Loss (1–{upto})"); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"loss_1to{upto}{suffix}.png"))
        plt.close()

        # 2) Token‐accuracy (your real metric)
        plt.figure()
        plt.plot(epochs, h["token_accuracy"][:upto],     label="Train token-acc")
        plt.plot(epochs, h["val_token_accuracy"][:upto], label="Val token-acc")
        plt.xlabel("Epoch"); plt.ylabel("Token Accuracy")
        plt.title(f"Token Accuracy (1–{upto})"); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"token_acc_1to{upto}{suffix}.png"))
        plt.close()

        # 3) ROUGE
        if "val_rouge1" in h:
            plt.figure()
            plt.plot(epochs, h["val_rouge1"][:upto], label="ROUGE-1")
            plt.plot(epochs, h["val_rouge2"][:upto], label="ROUGE-2")
            plt.plot(epochs, h["val_rougeL"][:upto], label="ROUGE-L")
            plt.xlabel("Epoch"); plt.ylabel("F1 Score")
            plt.title(f"ROUGE (1–{upto})"); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"rouge_1to{upto}{suffix}.png"))
            plt.close()



    def _plot_resources(self, upto, suffix):
        epochs = range(1, upto + 1)

        # CPU & RAM
        plt.figure()
        plt.plot(epochs, self.cpu_usage[:upto], label="CPU %")
        plt.plot(epochs, self.ram_usage[:upto], label="RAM %")
        plt.xlabel("Epoch"); plt.ylabel("Percent")
        plt.title(f"CPU & RAM (1–{upto})"); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"cpu_ram_1to{upto}{suffix}.png"))
        plt.close()

        # Per-GPU
        for i, util in self.gpu_usage.items():
            plt.figure()
            plt.plot(epochs, util[:upto], label=f"GPU{i} %")
            plt.xlabel("Epoch"); plt.ylabel("GPU Util %")
            plt.title(f"GPU {i} (1–{upto})"); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"gpu{i}_1to{upto}{suffix}.png"))
            plt.close()

    def on_epoch_end(self, epoch, logs=None):
        # record resources
        self.cpu_usage.append(psutil.cpu_percent())
        self.ram_usage.append(psutil.virtual_memory().percent)
        for idx, u in enumerate(self._get_gpu_utils()):
            self.gpu_usage.setdefault(idx, []).append(u)

        if (epoch + 1) % self.interval == 0:
            suffix = f"_ep{epoch+1}"
            self._plot_metrics(epoch+1, suffix)
            self._plot_resources(epoch+1, suffix)

    def on_train_end(self, logs=None):
        total = len(self.model.history.history["loss"])
        self._plot_metrics(total, "_final")
        self._plot_resources(total, "_final")
        
class SamplePrediction(Callback):
    def __init__(self, val_ds, tokenizer, max_len, samples=1, save_path="sample_pred.png"):
        super().__init__()
        
        self.val_ds = val_ds.take(1).unbatch().batch(samples)
        self.tokenizer = tokenizer
        self.start_id   = tgt_tokenizer.word_index.get('<start>', tgt_tokenizer.word_index[tgt_tokenizer.oov_token])
        self.end_id     = tgt_tokenizer.word_index.get('<end>',   tgt_tokenizer.word_index[tgt_tokenizer.oov_token])
        self.max_length = max_len
        self.save_path = save_path

    def on_train_end(self, logs=None):
      
        for (enc, _), _ in self.val_ds:
            
            dec_in = tf.fill([enc.shape[0], 1], self.start_id)
            result = []
         
            for _ in range(self.max_length):
                pad_len = self.max_length - dec_in.shape[1]
                logits = self.model([enc, tf.pad(dec_in, [[0,0],[0,pad_len]])],
                                    training=False)
                next_tok = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)
                dec_in = tf.concat([dec_in, next_tok[:,None]], axis=1)
                result.append(next_tok)
            preds = tf.stack(result, axis=1).numpy()

            # only take the first sample for display
            pred_seq = preds[0]
            # cut off at the first <end> token
            if self.end_id in pred_seq:
                cut = list(pred_seq).index(self.end_id)
                pred_seq = pred_seq[:cut]

            
            words = [ self.tokenizer.index_word.get(int(w), "<OOV>")
                      for w in pred_seq
                      if w not in (0, self.start_id) ]
            text = " ".join(words)

            
            plt.figure(figsize=(8, 1.5))
            plt.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=12)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(self.save_path, dpi=150)
            plt.close()

            print(f" Saved sample prediction to {self.save_path}")
            break

class RougeCallback(Callback):
    def __init__(self, val_ds, tgt_tokenizer, max_length_target, n_samples):
        super().__init__()
        self.val_ds     = val_ds
        self.tokenizer  = tgt_tokenizer
        self.start_id   = tgt_tokenizer.word_index.get('<start>', tgt_tokenizer.word_index[tgt_tokenizer.oov_token])
        self.end_id     = tgt_tokenizer.word_index.get('<end>',   tgt_tokenizer.word_index[tgt_tokenizer.oov_token])
        self.max_length = max_length_target
        self.scorer     = rouge_scorer.RougeScorer(
            ['rouge1','rouge2','rougeL'], use_stemmer=True
        )
        self.n_samples  = n_samples
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        (enc_batch, _), dec_tgt_batch = next(iter(self.val_ds))
        total = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        batch = tf.shape(enc_batch)[0]

        dec_input = tf.fill([batch, 1], self.start_id)
        result    = tf.zeros([batch, 0], dtype=tf.int32)
        for t in range(self.max_length):
            pad_amt    = self.max_length - tf.shape(dec_input)[1]
            dec_padded = tf.pad(dec_input, [[0,0],[0,pad_amt]])
            logits     = self.model([enc_batch, dec_padded], training=False)
            next_token = tf.cast(tf.argmax(logits[:, t, :], axis=-1), tf.int32)
            dec_input  = tf.concat([dec_input, next_token[:, None]], axis=1)
            result     = tf.concat([result,    next_token[:, None]], axis=1)

   
        count = 0
        for ref_seq, pred_seq in zip(dec_tgt_batch.numpy(), result.numpy()):
            ref_txt  = " ".join(
                self.tokenizer.index_word[w]
                for w in ref_seq 
                if w not in (0, self.start_id, self.end_id)
                )
            pred_txt = " ".join(
                self.tokenizer.index_word.get(w, "") 
                for w in pred_seq 
                if w not in (0, self.start_id, self.end_id)
                )
            sc = self.scorer.score(ref_txt, pred_txt)
            total['rouge1'] += sc['rouge1'].fmeasure
            total['rouge2'] += sc['rouge2'].fmeasure
            total['rougeL'] += sc['rougeL'].fmeasure
            count += 1
        avg = {k: (total[k] / count if count else 0.0) for k in total}
        logs.update({f'val_{k}': v for k, v in avg.items()})
    


class SaveOnAnyImprovement(tf.keras.callbacks.Callback):
    def __init__(self, model_path, monitor="val_rouge1"):
        super().__init__()
        self.model_path = model_path
        self.monitor = monitor
        self.best_loss = float("inf")
        self.best_acc = 0.0
        self.best_rouges = [0.0, 0.0, 0.0]  # [rouge1, rouge2, rougeL]

    def on_epoch_end(self, epoch, logs=None):
        new_loss = logs.get("val_loss")
        new_acc = logs.get("val_token_accuracy")
        new_rouges = [logs.get(f"val_rouge{i}") for i in [1, 2, "L"]]
        
        # Save if val_rouge1 or val_rougeL improves, or significant val_token_accuracy gain
        should_save = False
        save_reasons = []
        
        # Primary: Check val_rouge1 and val_rougeL
        if new_rouges[0] > self.best_rouges[0]:
            should_save = True
            save_reasons.append(f"val_rouge1 ↑ {new_rouges[0]:.4f}")
        if new_rouges[2] > self.best_rouges[2]:
            should_save = True
            save_reasons.append(f"val_rougeL ↑ {new_rouges[2]:.4f}")
        
        # Secondary: Check val_token_accuracy with loss constraint
        if new_acc > self.best_acc + 0.001 and new_loss < self.best_loss * 1.1:
            should_save = True
            save_reasons.append(f"val_token_accuracy ↑ {new_acc:.4f}")
        
        if should_save or not os.path.exists(self.model_path):
            self.model.save(self.model_path, overwrite=True)
            print(f"✔️ Saved model at epoch {epoch + 1} because {', '.join(save_reasons)}")
        
        # Update best metrics
        self.best_loss = min(self.best_loss, new_loss)
        self.best_acc = max(self.best_acc, new_acc)
        self.best_rouges = [max(nr, br) for nr, br in zip(new_rouges, self.best_rouges)]

class CustomEval(Callback):
    def __init__(self, val_ds, strategy):
        super().__init__()
        self.val_ds = val_ds
        self.strategy = strategy

    @tf.function
    def _eval_step(self, enc, dec_in, dec_tgt):
        preds = self.model([enc, dec_in], training=False)
        mask  = tf.cast(tf.not_equal(dec_tgt, 0), tf.float32)
        acc   = tf.keras.metrics.sparse_categorical_accuracy(dec_tgt, preds)
        return tf.reduce_sum(acc * mask), tf.reduce_sum(mask)

    def on_epoch_end(self, epoch, logs=None):
        total_correct = 0.0
        total_count   = 0.0
        for (enc, dec_in), dec_tgt in self.val_ds:
            c, n = self._eval_step(enc, dec_in, dec_tgt)
            total_correct += c
            total_count   += n
        token_acc = total_correct / total_count
        logs = logs or {}
        logs['val_token_accuracy'] = token_acc
        print(f"Validation token accuracy: {token_acc:.4f}")


def train_model(data_path, epochs=20, batch_size=240, emb_dim=50, train_from_scratch = False):
    inputs, targets = load_training_data(data_path)
    split = int(0.9 * len(inputs))
    save_dir     = "app/models/saved_model"
    tok_in_path  = f"{save_dir}/tokenizer_input.json"
    tok_tgt_path = f"{save_dir}/tokenizer_target.json"
    model_path   = f"{save_dir}/summarization_model.keras"
    
    os.makedirs(save_dir, exist_ok=True)
    
    train_in, train_tgt = inputs[:split], targets[:split]
    val_in, val_tgt = inputs[split:], targets[split:]
    
    if os.path.exists(tok_in_path) and os.path.exists(tok_tgt_path):
        tok_in = load_tokenizer(tok_in_path)
        tok_tgt = load_tokenizer(tok_tgt_path)
    else:
        tok_in = create_tokenizer(inputs, max_words=MAX_VOCAB, add_special_tokens=False)
        tok_tgt = create_tokenizer(targets, max_words=MAX_VOCAB, add_special_tokens=True)

    vs_in = min(len(tok_in.word_index) + 1, MAX_VOCAB + 1)
    vs_tgt = min(len(tok_tgt.word_index) + 1, MAX_VOCAB + 1)

    train_enc = preprocess_texts(train_in, tok_in,  max_length_input,  vs_in)
    train_dec = preprocess_texts(train_tgt, tok_tgt, max_length_target, vs_tgt)
    train_dec_in, train_dec_tgt = prepare_decoder_sequences(train_dec)

    val_enc = preprocess_texts(val_in,  tok_in,  max_length_input,  vs_in)
    val_dec = preprocess_texts(val_tgt, tok_tgt, max_length_target, vs_tgt)
    val_dec_in, val_dec_tgt = prepare_decoder_sequences(val_dec)


    num_train = len(train_enc)
    steps_per_epoch = max(1, num_train // batch_size)
    train_ds = (
        tf.data.Dataset
        .from_tensor_slices(((train_enc, train_dec_in), train_dec_tgt))
        .shuffle(buffer_size=steps_per_epoch, seed=42)
        .batch(batch_size, drop_remainder=True)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset
          .from_tensor_slices(((val_enc, val_dec_in), val_dec_tgt))
          .batch(batch_size, drop_remainder=False)
          .prefetch(tf.data.AUTOTUNE)
    )
    val_steps = max(1, len(val_enc) // batch_size
                   + (1 if len(val_enc) % batch_size else 0))
    
    n_rouge = 100
    rouge_ds = (
        tf.data.Dataset
        .from_tensor_slices(((val_enc, val_dec_in), val_dec_tgt))
        .shuffle(len(val_enc))            
        .take(n_rouge) 
        .cache()
        .batch(n_rouge, drop_remainder=False) 
        .prefetch(tf.data.AUTOTUNE)
    )
    
    
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope(): 
        if (not train_from_scratch) and os.path.exists(model_path):
            print("Loading model from disk")
            model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'Attention': Attention,
                'ExponentialDecay': ExponentialDecay,
            }
            )
        else:
            model = build_seq2seq_model(
            vs_in, vs_tgt, emb_dim,
            max_length_input, max_length_target
            )

        lr_schedule = ExponentialDecay(
            initial_learning_rate=5e-5,
            decay_steps=20_000,
            decay_rate=0.98,
            staircase=True
            )

        base_opt = Adam(learning_rate=lr_schedule)
        opt = tf.keras.mixed_precision.LossScaleOptimizer(base_opt, dynamic=True)
        model.compile(
            optimizer=opt,
            loss="sparse_categorical_crossentropy",
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="token_accuracy")
                ]      
            )
        (enc_batch, dec_in_batch), dec_tgt_batch = next(iter(train_ds))
        logits = model([enc_batch, dec_in_batch], training=False)
        print("Output logits stats:",
              tf.reduce_min(logits).numpy(),
              tf.reduce_max(logits).numpy(),
              "any NaN?", tf.reduce_any(tf.math.is_nan(logits)).numpy())
        print(">>> Global policy:", tf.keras.mixed_precision.global_policy().name)
        print(">>> Optimizer class:", type(model.optimizer).__name__)
     
        snap_cb = SnapshotCallback(
            save_dir="app/models/saved_model/plots",
            interval_epochs=10
            )
        rouge_cb = RougeCallback(
            val_ds=rouge_ds,
            tgt_tokenizer=tok_tgt,         
            max_length_target=max_length_target,
            n_samples=n_rouge
        )

        

        save_cb  = SaveOnAnyImprovement(model_path)
        

        callbacks = [
            rouge_cb,
            EarlyStopping(
                monitor='val_token_accuracy',   
                mode='max',
                patience=5,
                restore_best_weights=True
                ),
                save_cb,
                snap_cb,         
        ]
        history = model.fit(
            train_ds,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=val_steps
            )



    # after training: save tokenizers, plot history, return model
    with open(tok_in_path, 'w', encoding='utf-8') as f:
        f.write(tok_in.to_json())
        f.flush()                # push Python buffer to OS
        os.fsync(f.fileno()) 
    with open(tok_tgt_path, 'w', encoding='utf-8') as f:
        f.write(tok_tgt.to_json())
        f.flush()                # push Python buffer to OS
        os.fsync(f.fileno()) 

    plot_history(history, os.path.dirname(model_path))
    return model

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "app/models/data/text/training_data.json"
    model = train_model(data_path)
    print("Training complete.")
    print("Model saved to:", "app/models/saved_model/summarization_model.keras")
    print("Input tokenizer saved to:", "app/models/saved_model/tokenizer_input.json") 
