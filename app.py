"""Part 3 - Streamlit web app + (optional) AWS S3 storage.

Run locally:  streamlit run app.py
"""
import io
import json
import time
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

import audio_utils as au

HERE = Path(__file__).parent
MODEL_PATH = HERE / "models" / "music_genre_cnn.h5"
LABELS_PATH = HERE / "models" / "labels.json"

st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵")


@st.cache_resource  # load once, reuse on every upload
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model, json.loads(LABELS_PATH.read_text())


def save_to_s3(file_bytes, name):
    """Cloud storage: keep every uploaded sample in an AWS S3 bucket."""
    import boto3
    cfg = st.secrets["aws"]
    s3 = boto3.client("s3",
                      aws_access_key_id=cfg["access_key_id"],
                      aws_secret_access_key=cfg["secret_access_key"],
                      region_name=cfg["region"])
    key = f"uploads/{int(time.time())}_{name}"
    s3.put_object(Bucket=cfg["bucket"], Key=key, Body=file_bytes)
    return key


st.title("🎵 Music Genre Classifier")
st.caption("Upload a song sample. A CNN trained on the GTZAN dataset predicts its genre.")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python dl_model.py` first to create models/music_genre_cnn.h5")
    st.stop()

model, genres = load_model_and_labels()
file = st.file_uploader("Upload audio (wav / mp3, at least 3 seconds, 30 s works best)",
                        type=["wav", "mp3"])

if file:
    data = file.getvalue()
    st.audio(data)

    try:
        y = au.load_audio(io.BytesIO(data))
        batch = au.audio_to_batch(y).astype("float32")  # one spectrogram per 3 s clip
    except Exception as e:
        st.error(f"Could not read this audio: {e}")
        st.stop()

    fig, ax = plt.subplots(figsize=(6, 3))
    img = librosa.display.specshow(au.mel_db(y), sr=au.SR, hop_length=au.HOP_LENGTH,
                                   x_axis="time", y_axis="mel", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel-spectrogram")
    st.pyplot(fig)

    probs = model.predict(batch, verbose=0).mean(axis=0)   # average over all clips
    top = int(np.argmax(probs))
    st.success(f"Predicted genre: **{genres[top].title()}** ({probs[top]:.0%} confidence)")
    st.bar_chart({g: float(p) for g, p in zip(genres, probs)})

    if "aws" in st.secrets:  # only when S3 secrets are configured
        st.caption(f"Saved to S3: {save_to_s3(data, file.name)}")
