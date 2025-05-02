import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil
import sys

def create_spectrogram(audio_path, output_path):
    """
    Crée un spectrogramme à partir d'un fichier audio et le sauvegarde en tant qu'image
    """
    try:
        # Charger le fichier audio
        y, sr = librosa.load(audio_path)
        
        # Calculer le spectrogramme
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        # Créer la figure
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogramme de {os.path.basename(audio_path)}')
        plt.tight_layout()
        
        # Sauvegarder l'image
        plt.savefig(output_path)
        plt.close()
        
        print(f"Spectrogramme créé: {output_path}")
        return True
    except Exception as e:
        print(f"Erreur lors de la création du spectrogramme pour {audio_path}: {e}")
        return False

def supprimer_spectrograms():
    """
    Supprime tous les dossiers 'spectrogram' dans les sous-dossiers de bird_audio
    """
    base_dir = Path("dataset/bird_audio")
    
    # Vérifier si le répertoire existe
    if not base_dir.exists():
        print(f"Le répertoire {base_dir} n'existe pas.")
        return
    
    # Parcourir tous les sous-dossiers
    for bird_folder in base_dir.iterdir():
        if bird_folder.is_dir():
            spectrogram_dir = bird_folder / "spectrogram"
            
            # Supprimer le dossier spectrogram s'il existe
            if spectrogram_dir.exists():
                try:
                    shutil.rmtree(spectrogram_dir)
                    print(f"Dossier supprimé: {spectrogram_dir}")
                except Exception as e:
                    print(f"Erreur lors de la suppression de {spectrogram_dir}: {e}")

def main():
    # Chemin de base des données
    base_dir = Path("dataset/bird_audio")
    
    # Vérifier si le répertoire existe
    if not base_dir.exists():
        print(f"Le répertoire {base_dir} n'existe pas.")
        return
    
    # Parcourir tous les sous-dossiers
    for bird_folder in base_dir.iterdir():
        if bird_folder.is_dir():
            print(f"Traitement du dossier: {bird_folder.name}")
            
            # Créer le dossier de sortie des spectrogrammes s'il n'existe pas
            spectrogram_dir = bird_folder / "spectrogram"
            spectrogram_dir.mkdir(exist_ok=True)
            
            # Traiter tous les fichiers audio .ogg
            for audio_file in bird_folder.glob("*.ogg"):
                # Définir le chemin de sortie pour le spectrogramme
                output_path = spectrogram_dir / f"{audio_file.stem}_spectrogram.png"
                
                # Créer le spectrogramme
                create_spectrogram(str(audio_file), str(output_path))

if __name__ == "__main__":
    # Si l'argument "nettoyer" est passé, supprimer tous les dossiers spectrogram
    if len(sys.argv) > 1 and sys.argv[1] == "nettoyer":
        supprimer_spectrograms()
    else:
        main() 