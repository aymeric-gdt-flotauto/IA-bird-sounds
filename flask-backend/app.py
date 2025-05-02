from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from utils.audio_utils import convert_to_ogg, preprocess_audio
import tensorflow as tf

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('model/bird_model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['audio']
    filename = secure_filename(file.filename)
    path = os.path.join('uploads', filename)
    file.save(path)

    ogg_path = convert_to_ogg(path)
    features = preprocess_audio(ogg_path)  # à adapter à ton modèle
    prediction = model.predict(features)
    predicted_bird = ...  # à extraire selon ta sortie

    return jsonify({'bird': predicted_bird})

if __name__ == '__main__':
    app.run(debug=True)
