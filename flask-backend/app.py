from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from utils.audio_utils import convert_to_ogg, preprocess_audio
import tensorflow as tf

app = Flask(__name__)
CORS(app)

app.static_folder = 'static'
model = tf.keras.models.load_model('model/model_01.keras')
labels = {'barswa' : 'Hirondelle rustique', 'comsan' : 'Chevalier guignette', 'eaywag1' : 'Bergeronnette printanière', 'thrnig1' : 'Rossignol progné', 'wlwwar' : 'Pouillot fitis', 'woosan': 'Chevalier sylvain'}  # adapte à ton modèle

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['audio']
    filename = secure_filename(file.filename)
    path = os.path.join('uploads', filename)
    file.save(path)

    ogg_path = convert_to_ogg(path)
    features = preprocess_audio(ogg_path)

    prediction = model.predict(features)[0]  # shape: (num_classes,)
    
    class_ids = list(labels.keys())  # ['barswa', 'comsan', ...]
    top_indices = prediction.argsort()[-5:][::-1]  # indices des 5 plus grandes valeurs

    top_predictions = []
    for idx in top_indices:
        class_id = class_ids[idx]
        top_predictions.append({
            'id': class_id,
            'label': labels[class_id],
            'confidence': float(prediction[idx])
        })

    spectrogram_filename = f"{os.path.splitext(filename)[0]}_spectrogram.jpg"
    spectrogram_url = f"http://localhost:5000/static/converted/{spectrogram_filename}"

    return jsonify({
        'top_predictions': top_predictions,
        'spectrogram_url': spectrogram_url
    })
