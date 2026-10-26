"""Part 2 - Deep Learning: 2D CNN on Mel-spectrograms.

Run:  python dl_model.py                 (about 10-30 min on a laptop CPU)
Needs: data/genres_original/<genre>/*.wav   (from the Kaggle GTZAN download)

What it does
  1. Splits the SONGS 70/15/15 into train/val/test (no song is in two sets).
  2. Cuts each 30 s song into ten 3 s clips and turns each clip into a
     Mel-spectrogram image (10x more training data).
  3. Trains a CNN, evaluates it, saves models/music_genre_cnn.h5 + charts.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models

import audio_utils as au

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--rebuild", action="store_true", help="ignore cached spectrograms")
args = parser.parse_args()

DATA = Path(args.data_dir)
CACHE = DATA / "mel_cache.npz"
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)
tf.keras.utils.set_random_seed(42)


# ---------- 1. Data ----------
def list_tracks(audio_dir):
    genres = sorted(p.name for p in audio_dir.iterdir() if p.is_dir())
    files, labels = [], []
    for gi, g in enumerate(genres):
        for f in sorted((audio_dir / g).glob("*.wav")):
            files.append(f)
            labels.append(gi)
    return genres, files, np.array(labels)


def extract(files, labels, name):
    X, y, track = [], [], []
    for i, (f, lab) in enumerate(zip(files, labels)):
        try:
            batch = au.audio_to_batch(au.load_audio(f))
        except Exception as e:  # GTZAN has one corrupted file (jazz.00054.wav)
            print(f"  skipped {f.name}: {e}")
            continue
        X.append(batch)
        y.extend([lab] * len(batch))
        track.extend([i] * len(batch))
        if (i + 1) % 100 == 0:
            print(f"  [{name}] {i + 1}/{len(files)} songs processed")
    return np.concatenate(X), np.array(y), np.array(track)


def get_data():
    if CACHE.exists() and not args.rebuild:
        print(f"Loading cached spectrograms from {CACHE}")
        d = np.load(CACHE, allow_pickle=True)
        return list(d["genres"]), {k: d[k] for k in d.files if k != "genres"}

    audio_dir = DATA / "genres_original"
    if not audio_dir.exists():
        raise SystemExit(f"Cannot find {audio_dir}. Put the GTZAN 'genres_original' folder inside data/.")
    genres, files, labels = list_tracks(audio_dir)
    print(f"Found {len(files)} songs in {len(genres)} genres: {genres}")

    idx = np.arange(len(files))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=labels, random_state=42)
    va, te = train_test_split(rest, test_size=0.50, stratify=labels[rest], random_state=42)

    out = {}
    for name, ids in (("train", tr), ("val", va), ("test", te)):
        print(f"Extracting {name} spectrograms ...")
        X, y, t = extract([files[i] for i in ids], labels[ids], name)
        out[f"X_{name}"], out[f"y_{name}"], out[f"t_{name}"] = X, y, t
    np.savez_compressed(CACHE, genres=np.array(genres), **out)
    return genres, out


# ---------- 2. Model ----------
def build_model(n_classes):
    inp = layers.Input(shape=au.INPUT_SHAPE)
    x = layers.Rescaling(1 / 255.0)(inp)          # 0-255 -> 0-1
    for filters in (32, 64, 128, 128):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out, name="music_genre_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


genres, d = get_data()
print(f"Train {len(d['X_train'])}  Val {len(d['X_val'])}  Test {len(d['X_test'])} spectrogram clips")
model = build_model(len(genres))
model.summary()

# ---------- 3. Train ----------
history = model.fit(
    d["X_train"].astype("float32"), d["y_train"],
    validation_data=(d["X_val"].astype("float32"), d["y_val"]),
    epochs=args.epochs, batch_size=args.batch_size,
    callbacks=[callbacks.EarlyStopping(patience=8, restore_best_weights=True),
               callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)],
)

# ---------- 4. Evaluate ----------
Xte = d["X_test"].astype("float32")
probs = model.predict(Xte, verbose=0)
clip_pred = probs.argmax(1)
clip_acc = accuracy_score(d["y_test"], clip_pred)

# song-level: average the clip probabilities of each song, then pick the best genre
song_ids = np.unique(d["t_test"])
song_true = np.array([d["y_test"][d["t_test"] == s][0] for s in song_ids])
song_pred = np.array([probs[d["t_test"] == s].mean(0).argmax() for s in song_ids])
song_acc = accuracy_score(song_true, song_pred)

print(f"\nTest accuracy (3 s clips): {clip_acc:.3f}")
print(f"Test accuracy (whole songs): {song_acc:.3f}")
print(classification_report(song_true, song_pred, target_names=genres, zero_division=0))

# ---------- 5. Save model + report charts ----------
model.save("models/music_genre_cnn.h5")
json.dump(genres, open("models/labels.json", "w"))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history.history["accuracy"], label="train")
ax[0].plot(history.history["val_accuracy"], label="validation")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(history.history["loss"], label="train")
ax[1].plot(history.history["val_loss"], label="validation")
ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
fig.tight_layout(); fig.savefig("reports/cnn_training_curves.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay.from_predictions(song_true, song_pred, display_labels=genres,
                                        ax=ax, xticks_rotation=45, cmap="Blues")
ax.set_title(f"CNN - confusion matrix (whole songs, acc {song_acc:.0%})")
fig.tight_layout(); fig.savefig("reports/cnn_confusion_matrix.png", dpi=150)

print("Saved: models/music_genre_cnn.h5, models/labels.json, reports/cnn_*.png")
