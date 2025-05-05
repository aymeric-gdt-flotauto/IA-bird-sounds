import React, { useState } from 'react';
import axios from 'axios';

const AudioUploader = () => {
  const [file, setFile] = useState(null);
  const [topPredictions, setTopPredictions] = useState([]);
  const [spectrogramUrl, setSpectrogramUrl] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files?.[0] || null);
    setTopPredictions([]);  // Correction ici
    setSpectrogramUrl('');
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('audio', file);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setTopPredictions(response.data.top_predictions);
      setSpectrogramUrl(response.data.spectrogram_url);
    } catch (error) {
      console.error(error);
      setTopPredictions ("Erreur lors de la prédiction.");
    } finally {
      setLoading(false);
    }
  };


  return (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded-2xl shadow-md mt-10 space-y-6">
      <h1 className="text-2xl font-semibold text-center">Reconnaissance de chant d'oiseau</h1>

      <div>
        <input
          type="file"
          accept="audio/*"
          onChange={handleFileChange}
          className="w-full text-sm text-gray-500
            file:mr-4 file:py-2 file:px-4
            file:rounded-full file:border-0
            file:text-sm file:font-semibold
            file:bg-blue-50 file:text-blue-700
            hover:file:bg-blue-100"
        />
      </div>

      <div className="text-center">
        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="px-4 py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? 'Analyse en cours...' : 'Envoyer'}
        </button>
      </div>

      {topPredictions.length > 0 && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <div>
              <h2 className="text-lg font-semibold mb-2 text-center">Top 5 des prédictions</h2>
              <ul className="space-y-2">
                {topPredictions.map((pred, index) => (
                  <li key={index} className="flex justify-between px-4 py-1 bg-gray-100 rounded">
                    <span>{pred.label}</span>
                    <span className="text-gray-600">{(pred.confidence * 100).toFixed(1)}%</span>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h2 className="text-lg font-semibold mb-2 text-center">Spectrogramme</h2>
              <img
                src={spectrogramUrl}
                alt="Spectrogramme"
                className="w-full rounded-xl shadow"
              />
            </div>
          </div>

          <div className="mt-6 text-center">
            <h3 className="font-semibold mb-2">Image de l'espèce la plus probable :</h3>
            <img
              src={`/images/${topPredictions[0].label}.jpg`}
              alt={topPredictions[0].label}
              className="w-2/3 mx-auto rounded-xl shadow"
              onError={(e) => e.currentTarget.style.display = 'none'}
            />
          </div>
        </>
      )}
    </div>
  );
};

export default AudioUploader;
