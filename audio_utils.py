"""Shared audio preprocessing.

Used by BOTH training (dl_model.py) and the web app (app.py), so the model
always sees exactly the same kind of input at train time and at prediction time.
"""
import numpy as np
import librosa

SR = 22050                        # sample rate (Hz)
SEG_SECONDS = 3                   # each 30 s song is cut into 3 s clips
SEG_SAMPLES = SR * SEG_SECONDS
N_MELS = 128                      # height of the spectrogram image
HOP_LENGTH = 512
N_FRAMES = 1 + SEG_SAMPLES // HOP_LENGTH      # width of the image (= 130)
INPUT_SHAPE = (N_MELS, N_FRAMES, 1)


def load_audio(path_or_buffer, duration=30):
    """Load audio as mono at SR, max `duration` seconds."""
    y, _ = librosa.load(path_or_buffer, sr=SR, mono=True, duration=duration)
    return y


def mel_db(y):
    """Waveform -> Mel-spectrogram in decibels (values in [-80, 0])."""
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
    return librosa.power_to_db(mel, ref=np.max)


def to_uint8(db):
    """Scale dB values to 0-255 integers (keeps the cached dataset small)."""
    return np.round(np.clip((db + 80.0) / 80.0, 0, 1) * 255).astype(np.uint8)


def audio_to_batch(y):
    """Waveform -> array of shape (n_clips, 128, 130, 1), one image per 3 s clip."""
    n = len(y) // SEG_SAMPLES
    if n == 0:
        raise ValueError(f"Audio is shorter than {SEG_SECONDS} seconds.")
    clips = [y[i * SEG_SAMPLES:(i + 1) * SEG_SAMPLES] for i in range(n)]
    batch = np.stack([to_uint8(mel_db(c)) for c in clips])[..., None]
    return batch[:, :, :N_FRAMES, :]
