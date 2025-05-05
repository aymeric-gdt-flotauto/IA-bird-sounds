import os
import sys
from pydub import AudioSegment
import numpy as np
import tensorflow as tf
import librosa
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from create_spectrograms import create_mel_spectrogram_for_model

UPLOAD_FOLDER = "uploads"
CONVERTED_FOLDER = "static/converted"
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
    # Étape 1 : Conversion en mono 32kHz
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    converted_path = os.path.join(CONVERTED_FOLDER, f"{base_name}_converted_32k.wav")
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1)
    audio.export(converted_path, format="wav")

    # Étape 2 : Création du spectrogramme JPEG
    spectrogram_img_path = os.path.join(CONVERTED_FOLDER, f"{base_name}_spectrogram.jpg")
    denoised_audio_path = os.path.join(CONVERTED_FOLDER, f"{base_name}_denoised.wav")
    create_mel_spectrogram_for_model(converted_path, spectrogram_img_path, audio_output_path=denoised_audio_path)

    # Étape 3 : Chargement du JPEG avec les bonnes dimensions
    img = Image.open(spectrogram_img_path).convert("RGB")
    img = img.resize((148, 388))  # largeur, hauteur → donne (388, 148, 3) après np.array()
    spectrogram = np.array(img) / 255.0  # normalisation

    # Étape 4 : Ajout de la dimension batch
    spectrogram = np.expand_dims(spectrogram, axis=0)  # (1, 388, 148, 3)

    return spectrogram

