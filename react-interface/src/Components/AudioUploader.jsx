import React, { useState } from 'react';
import axios from 'axios';

const AudioUploader = () => {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files?.[0] || null);
    setPrediction(null);
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
      setPrediction(response.data.bird);
    } catch (error) {
      console.error(error);
      setPrediction("Erreur lors de la prédiction.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto p-6 bg-white rounded-2xl shadow-md mt-10 text-center space-y-4">
      <h1 className="text-xl font-semibold">Reconnaissance de chant d'oiseau</h1>
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
      <button
        onClick={handleUpload}
        disabled={!file || loading}
        className="px-4 py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50"
      >
        {loading ? 'Analyse en cours...' : 'Envoyer'}
      </button>
      {prediction && (
        <div className="mt-4 text-green-700 font-medium">
          Oiseau prédit : {prediction}
        </div>
      )}
    </div>
  );
};

export default AudioUploader;
