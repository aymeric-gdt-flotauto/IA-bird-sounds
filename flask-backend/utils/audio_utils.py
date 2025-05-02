import os
from pydub import AudioSegment
import numpy as np
import tensorflow as tf
import librosa

UPLOAD_FOLDER = "uploads"
CONVERTED_FOLDER = "converted"
TARGET_SAMPLE_RATE = 32000  # 32 kHz

os.makedirs(CONVERTED_FOLDER, exist_ok=True)

def convert_to_ogg(input_path):
    filename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(CONVERTED_FOLDER, f"{filename}.ogg")

    # Convert to mono and resample to 32kHz
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1)
    audio.export(output_path, format="ogg")

    return output_path

def preprocess_audio(audio_path):
    # Chargement avec librosa (mono, 32 kHz)
    y, sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)

    # Padding/cropping à 5 secondes (à adapter à la durée attendue)
    desired_length = TARGET_SAMPLE_RATE * 5  # 5 secondes
    if len(y) < desired_length:
        y = np.pad(y, (0, desired_length - len(y)))
    else:
        y = y[:desired_length]

    # Exemple de transformation en MFCCs (à adapter à ton modèle)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = mfccs.T  # (temps, 40)

    # Ajouter une dimension pour batch et channel si nécessaire
    return np.expand_dims(mfccs, axis=0)  # (1, temps, 40)
