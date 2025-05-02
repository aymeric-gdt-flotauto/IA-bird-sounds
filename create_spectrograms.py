import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil
import sys

def create_raw_spectrogram(audio_path, output_path, clean_display=False):
    """
    Crée un spectrogramme brut (raw) à partir d'un fichier audio et le sauvegarde en tant qu'image
    
    Parameters:
    -----------
    audio_path : str
        Chemin du fichier audio à analyser
    output_path : str
        Chemin où sauvegarder le spectrogramme
    clean_display : bool, default=False
        Si True, génère un spectrogramme sans axes, titre ni colorbar
    """
    try:
        # Charger le fichier audio
        y, sr = librosa.load(audio_path)
        
        # Calculer le spectrogramme
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        # Créer la figure
        plt.figure(figsize=(10, 4))
        
        # Afficher le spectrogramme
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        
        if not clean_display:
            # Ajouter les éléments visuels (si non clean)
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogramme brut de {os.path.basename(audio_path)}')
            plt.tight_layout()
        else:
            # Version propre sans axes, titre ou colorbar
            plt.axis('off')
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
        
        # Sauvegarder l'image
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0 if clean_display else 0.1)
        plt.close()
        
        print(f"Spectrogramme brut créé: {output_path}")
        return True
    except Exception as e:
        print(f"Erreur lors de la création du spectrogramme brut pour {audio_path}: {e}")
        return False

def supprimer_spectrograms():
    """
    Supprime tous les dossiers 'spectrograms' dans les sous-dossiers de bird_audio
    """
    base_dir = Path("dataset/bird_audio")
    
    # Vérifier si le répertoire existe
    if not base_dir.exists():
        print(f"Le répertoire {base_dir} n'existe pas.")
        return
    
    # Parcourir tous les sous-dossiers
    for bird_folder in base_dir.iterdir():
        if bird_folder.is_dir():
            spectrograms_dir = bird_folder / "spectrograms"
            
            # Supprimer le dossier spectrograms s'il existe
            if spectrograms_dir.exists():
                try:
                    shutil.rmtree(spectrograms_dir)
                    print(f"Dossier supprimé: {spectrograms_dir}")
                except Exception as e:
                    print(f"Erreur lors de la suppression de {spectrograms_dir}: {e}")

def main():
    # Chemin de base des données
    base_dir = Path("dataset/bird_audio")
    
    # Vérifier si le répertoire existe
    if not base_dir.exists():
        print(f"Le répertoire {base_dir} n'existe pas.")
        return
    
    # Parcourir tous les sous-dossiers d'espèces d'oiseaux
    for bird_folder in base_dir.iterdir():
        if bird_folder.is_dir():
            print(f"Traitement du dossier: {bird_folder.name}")
            
            # Créer le dossier principal des spectrogrammes
            spectrograms_dir = bird_folder / "spectrograms"
            spectrograms_dir.mkdir(exist_ok=True)
            
            # Créer le sous-dossier pour les spectrogrammes bruts
            raw_dir = spectrograms_dir / "raw"
            raw_dir.mkdir(exist_ok=True)
            
            # Créer le sous-dossier pour les spectrogrammes bruts sans éléments visuels
            raw_clean_dir = spectrograms_dir / "raw_clean"
            raw_clean_dir.mkdir(exist_ok=True)
            
            # Traiter tous les fichiers audio .ogg
            for audio_file in bird_folder.glob("*.ogg"):
                # Définir le chemin de sortie pour le spectrogramme brut standard
                output_path = raw_dir / f"{audio_file.stem}_raw.png"
                
                # Créer le spectrogramme brut standard
                create_raw_spectrogram(str(audio_file), str(output_path), clean_display=False)
                
                # Définir le chemin de sortie pour le spectrogramme brut sans éléments visuels
                output_clean_path = raw_clean_dir / f"{audio_file.stem}_raw_clean.png"
                
                # Créer le spectrogramme brut sans éléments visuels
                create_raw_spectrogram(str(audio_file), str(output_clean_path), clean_display=True)

if __name__ == "__main__":
    # Si l'argument "nettoyer" est passé, supprimer tous les dossiers spectrograms
    if len(sys.argv) > 1 and sys.argv[1] == "nettoyer":
        supprimer_spectrograms()
    else:
        main() 