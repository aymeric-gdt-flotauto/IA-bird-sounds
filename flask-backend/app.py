from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from utils.audio_utils import convert_to_ogg, preprocess_audio
import tensorflow as tf

app = Flask(__name__)
CORS(app)

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

    prediction = model.predict(features)
    
    # Supposons que la sortie est un vecteur de probabilités
    predicted_index = prediction.argmax(axis=1)[0]
    
    class_ids = list(labels.keys())  # ['barswa', 'comsan', ...]
    predicted_class_id = class_ids[predicted_index]
    predicted_bird = labels[predicted_class_id]

    return jsonify({'bird': predicted_bird})
