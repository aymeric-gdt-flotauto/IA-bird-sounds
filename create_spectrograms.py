import os
import sys
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend non-interactif Agg
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image
import noisereduce as nr
import soundfile as sf
import albumentations as A
import random

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

def create_threshold_spectrogram(audio_path, output_path):
    """
    Crée un spectrogramme ne conservant que les signaux au-dessus du 3ème quartile des niveaux sonores
    puis le crop en supprimant 150 pixels du bas
    
    Parameters:
    -----------
    audio_path : str
        Chemin du fichier audio à analyser
    output_path : str
        Chemin où sauvegarder le spectrogramme
    """
    try:
        # Charger le fichier audio
        y, sr = librosa.load(audio_path)
        
        # Calculer le spectrogramme
        stft = librosa.stft(y)
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        
        # Calculer le 3ème quartile des amplitudes
        q3 = np.percentile(stft_db, 75)
        
        # Créer un masque pour ne garder que les valeurs au-dessus du 3ème quartile
        mask = stft_db >= q3
        
        # Appliquer le masque (mettre à une valeur très basse les points en-dessous du seuil)
        stft_db_filtered = stft_db.copy()
        stft_db_filtered[~mask] = -80  # Valeur très basse en dB, presque silencieuse
        
        # Créer la figure
        plt.figure(figsize=(10, 4))
        
        # Afficher le spectrogramme filtré
        librosa.display.specshow(stft_db_filtered, sr=sr, x_axis='time', y_axis='log')
        
        # Version propre sans axes, titre ou colorbar
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        
        # Chemin temporaire pour l'image avant cropping
        temp_path = str(output_path) + ".temp.png"
        
        # Sauvegarder l'image temporaire
        plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Ouvrir l'image avec PIL pour la cropper
        with Image.open(temp_path) as img:
            width, height = img.size
            # Cropper l'image en supprimant 150 pixels du bas
            cropped_img = img.crop((0, 0, width, max(1, height - 150)))
            # Sauvegarder l'image croppée
            cropped_img.save(output_path)
        
        # Supprimer le fichier temporaire
        os.remove(temp_path)
        
        print(f"Spectrogramme à seuil Q3 (croppé) créé: {output_path}")
        return True
    except Exception as e:
        print(f"Erreur lors de la création du spectrogramme à seuil Q3 pour {audio_path}: {e}")
        return False

def create_noisereduced_spectrogram(audio_path, output_path, audio_output_path=None):
    """
    Crée un spectrogramme à partir d'un fichier audio après réduction de bruit
    et sauvegarde également le fichier audio traité
    
    Parameters:
    -----------
    audio_path : str
        Chemin du fichier audio à analyser
    output_path : str
        Chemin où sauvegarder le spectrogramme
    audio_output_path : str, optional
        Chemin où sauvegarder le fichier audio traité, si fourni
    """
    try:
        # Charger le fichier audio
        y, sr = librosa.load(audio_path, sr=32000)
        
        # Appliquer la réduction de bruit
        # On suppose que les premières secondes de l'audio contiennent principalement du bruit
        noise_sample = y[:int(sr)]  # Première seconde comme échantillon de bruit
        y_reduced = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, stationary=True)
        
        # Sauvegarder le fichier audio traité si un chemin est fourni
        if audio_output_path:
            sf.write(audio_output_path, y_reduced, sr)
            print(f"Fichier audio avec réduction de bruit créé: {audio_output_path}")
        
        # Calculer le spectrogramme
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_reduced)), ref=np.max)
        
        # Créer la figure
        plt.figure(figsize=(10, 4))
        
        # Afficher le spectrogramme
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        
        # Version propre sans axes, titre ou colorbar
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        
        # Sauvegarder l'image
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"Spectrogramme avec réduction de bruit créé: {output_path}")
        return True
    except Exception as e:
        print(f"Erreur lors de la création du spectrogramme avec réduction de bruit pour {audio_path}: {e}")
        return False

def apply_audio_augmentation(y, sr, seed=None):
    """
    Applique des augmentations de données au signal audio
    
    Parameters:
    -----------
    y : np.ndarray
        Signal audio
    sr : int
        Taux d'échantillonnage
    seed : int, optional
        Graine aléatoire pour la reproductibilité
        
    Returns:
    --------
    np.ndarray
        Signal audio augmenté
    """
    # Fixer la graine si fournie
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    augmented_y = y.copy()
    
    # Probabilité d'appliquer chaque augmentation
    prob_threshold = 0.5
    
    # 1. Time shifting (décalage temporel) - décale le signal dans le temps
    if random.random() < prob_threshold:
        shift_factor = int(random.uniform(-0.1, 0.1) * len(y))
        if shift_factor > 0:
            augmented_y = np.pad(augmented_y[shift_factor:], (0, shift_factor))
        else:
            augmented_y = np.pad(augmented_y[:shift_factor], (-shift_factor, 0))
    
    # 2. Pitch shifting (changement de hauteur) - pour simuler différents individus
    if random.random() < prob_threshold:
        # Pas plus de 2 demi-tons pour garder un son naturel
        n_steps = random.uniform(-2, 2)
        augmented_y = librosa.effects.pitch_shift(augmented_y, sr=sr, n_steps=n_steps)
    
    # 3. Time stretching (étirement temporel) - pour simuler différentes vitesses de chant
    if random.random() < prob_threshold:
        # Entre 0.9x et 1.1x pour ne pas trop déformer
        rate = random.uniform(0.9, 1.1)
        augmented_y = librosa.effects.time_stretch(augmented_y, rate=rate)
        # Ajuster la longueur si nécessaire
        if len(augmented_y) > len(y):
            augmented_y = augmented_y[:len(y)]
        elif len(augmented_y) < len(y):
            augmented_y = np.pad(augmented_y, (0, len(y) - len(augmented_y)))
    
    # 4. Ajout d'un léger bruit ambiant
    if random.random() < prob_threshold:
        noise_factor = random.uniform(0.001, 0.005)
        noise = np.random.randn(len(augmented_y)) * noise_factor
        augmented_y = augmented_y + noise
        # Normaliser si nécessaire
        if np.abs(augmented_y).max() > 1.0:
            augmented_y = augmented_y / np.abs(augmented_y).max()
    
    return augmented_y

def create_mel_spectrogram_for_model(audio_path, output_path, audio_output_path=None, apply_augmentation=False, num_augmentations=1, index=0):
    """
    Crée un spectrogramme Mel optimisé pour l'entrée dans un modèle de deep learning
    avec normalisation ImageNet et éventuellement application d'augmentation de données
    
    Parameters:
    -----------
    audio_path : str
        Chemin du fichier audio à analyser
    output_path : str
        Chemin où sauvegarder le spectrogramme
    audio_output_path : str, optional
        Chemin où sauvegarder le fichier audio traité, si fourni
    apply_augmentation : bool, default=False
        Si True, applique des techniques d'augmentation de données
    num_augmentations : int, default=1
        Nombre d'augmentations à générer (utilisé seulement si apply_augmentation=True)
    index : int, default=0
        Index de l'augmentation actuelle (utilisé pour la reproductibilité)
    """
    try:
        # Charger le fichier audio avec un taux d'échantillonnage fixe
        y, sr = librosa.load(audio_path, sr=32000)
        
        # Appliquer la réduction de bruit
        noise_sample = y[:int(sr)]  # Première seconde comme échantillon de bruit
        y_reduced = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, stationary=True)
        
        # Appliquer les augmentations de données audio si demandé
        if apply_augmentation:
            # Utiliser l'index comme graine pour avoir des augmentations reproductibles mais différentes
            seed = index if index > 0 else None
            y_processed = apply_audio_augmentation(y_reduced, sr, seed=seed)
        else:
            y_processed = y_reduced
        
        # Sauvegarder le fichier audio traité si un chemin est fourni
        if audio_output_path:
            sf.write(audio_output_path, y_processed, sr)
            print(f"Fichier audio traité créé: {audio_output_path}")
        
        # S'assurer que le signal audio a une longueur de 1,000,000 échantillons
        # Ajuster par zero-padding ou troncature
        target_length = 1000000
        if len(y_processed) < target_length:
            # Padding - ajouter des zéros à la fin
            y_processed = np.pad(y_processed, (0, target_length - len(y_processed)))
        else:
            # Tronquer si trop long
            y_processed = y_processed[:target_length]
        
        # Calculer le spectrogramme Mel
        n_mels = 256  # Nombre de bandes Mel demandé
        mel_spec = librosa.feature.melspectrogram(
            y=y_processed, 
            sr=sr,
            n_mels=n_mels,
            fmax=sr/2
        )
        
        # Convertir à l'échelle dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normaliser entre 0 et 1 pour la conversion en image
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        # Convertir en image 8-bit
        mel_img = (mel_spec_norm * 255).astype(np.uint8)
        
        # Augmentation du spectrogramme si demandée
        if apply_augmentation:
            # Fixer la graine pour les transformations
            if index > 0:
                A.ReplayCompose.seed = index
            
            # Appliquer l'augmentation au niveau du spectrogramme
            aug_transform = A.Compose([
                # Time masking : masque des segments temporels
                A.CoarseDropout(
                    max_holes=4, max_height=4, max_width=20, min_width=5,
                    fill_value=0, p=0.5
                ),
                # Frequency masking : masque des bandes de fréquences
                A.CoarseDropout(
                    max_holes=4, max_height=10, max_width=8, min_height=5,
                    fill_value=0, p=0.5
                ),
                # Petites transformations géométriques
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=0,
                    border_mode=0, value=0, p=0.5
                ),
                # Légères modifications de contraste
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.5
                ),
            ])
            
            mel_img = aug_transform(image=mel_img)["image"]
        
        # Convertir en 3 canaux (répéter le canal pour RGB)
        mel_img_rgb = np.stack([mel_img, mel_img, mel_img], axis=-1)
        
        # Appliquer la normalisation ImageNet avec albumentations
        transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])
        
        # Appliquer la transformation
        transform(image=mel_img_rgb)  # On applique la transformation mais on ne la sauvegarde pas
        
        # Créer une image pour la visualisation (avant normalisation)
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_img, cmap='viridis', origin='lower', aspect='auto')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Si augmentation, indiquer dans le message
        aug_text = f" (augmentation {index}/{num_augmentations})" if apply_augmentation else ""
        print(f"Spectrogramme Mel pour modèle créé{aug_text}: {output_path}")
        return True
    except Exception as e:
        print(f"Erreur lors de la création du spectrogramme Mel pour modèle pour {audio_path}: {e}")
        return False

def create_multiple_augmented_spectrograms(audio_path, output_dir, base_filename, num_augmentations=5):
    """
    Crée plusieurs versions augmentées d'un spectrogramme Mel à partir d'un seul fichier audio
    
    Parameters:
    -----------
    audio_path : str
        Chemin du fichier audio à analyser
    output_dir : str ou Path
        Dossier où sauvegarder les spectrogrammes
    base_filename : str
        Nom de base pour les fichiers générés
    num_augmentations : int, default=5
        Nombre d'augmentations différentes à générer
    """
    output_dir = Path(output_dir)
    
    # Créer le dossier de sortie s'il n'existe pas
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Générer plusieurs versions augmentées
    for i in range(1, num_augmentations + 1):
        # Définir le chemin de sortie pour cette augmentation
        output_path = output_dir / f"{base_filename}_aug{i}.png"
        
        # Créer le spectrogramme augmenté
        create_mel_spectrogram_for_model(
            audio_path, 
            str(output_path), 
            None, 
            apply_augmentation=True,
            num_augmentations=num_augmentations,
            index=i
        )
    
    print(f"Généré {num_augmentations} versions augmentées pour {os.path.basename(audio_path)}")
    return True

def supprimer_spectrograms():
    """
    Supprime tous les dossiers 'spectrograms' et 'audio_processed' dans les sous-dossiers de bird_audio
    """
    base_dir = Path("dataset2/dataset-2")
    
    # Vérifier si le répertoire existe
    if not base_dir.exists():
        print(f"Le répertoire {base_dir} n'existe pas.")
        return
    
    # Parcourir tous les sous-dossiers
    for bird_folder in base_dir.iterdir():
        if bird_folder.is_dir():
            # Supprimer le dossier spectrograms s'il existe
            spectrograms_dir = bird_folder / "spectrograms"
            if spectrograms_dir.exists():
                try:
                    shutil.rmtree(spectrograms_dir)
                    print(f"Dossier supprimé: {spectrograms_dir}")
                except Exception as e:
                    print(f"Erreur lors de la suppression de {spectrograms_dir}: {e}")
            
            # Supprimer le dossier audio_processed s'il existe
            audio_processed_dir = bird_folder / "audio_processed"
            if audio_processed_dir.exists():
                try:
                    shutil.rmtree(audio_processed_dir)
                    print(f"Dossier supprimé: {audio_processed_dir}")
                except Exception as e:
                    print(f"Erreur lors de la suppression de {audio_processed_dir}: {e}")

def main():
    # Chemin de base des données
    base_dir = Path("dataset2/dataset-2")
    
    # Vérifier si le répertoire existe
    if not base_dir.exists():
        print(f"Le répertoire {base_dir} n'existe pas.")
        return
    
    # Nombre d'augmentations à générer pour chaque fichier audio
    num_augmentations = 2
    
    # Parcourir tous les sous-dossiers d'espèces d'oiseaux
    for bird_folder in base_dir.iterdir():
        if bird_folder.is_dir():
            print(f"Traitement du dossier: {bird_folder.name}")
            
            # Créer le dossier principal des spectrogrammes
            spectrograms_dir = bird_folder / "spectrograms"
            spectrograms_dir.mkdir(exist_ok=True)
            
            # Créer les sous-dossiers pour les différents types de spectrogrammes
            #raw_dir = spectrograms_dir / "raw"
            #raw_dir.mkdir(exist_ok=True)
            
            #raw_clean_dir = spectrograms_dir / "raw_clean"
            #raw_clean_dir.mkdir(exist_ok=True)
            
            #threshold_q3_dir = spectrograms_dir / "threshold_q3"
            #threshold_q3_dir.mkdir(exist_ok=True)
            
            #noise_reduced_dir = spectrograms_dir / "noise_reduced"
            #noise_reduced_dir.mkdir(exist_ok=True)
            
            mel_model_dir = spectrograms_dir / "mel_model"
            mel_model_dir.mkdir(exist_ok=True)
            
            # Créer le dossier pour les fichiers audio traités
            #audio_processed_dir = bird_folder / "audio_processed"
            #audio_processed_dir.mkdir(exist_ok=True)
            
            # Traiter tous les fichiers audio .ogg
            for audio_file in bird_folder.glob("*.ogg"):
                # Chemin de base pour les noms de fichiers
                audio_stem = audio_file.stem
                
                # 1. Spectrogramme brut standard
                #output_path = raw_dir / f"{audio_stem}_raw.png"
                #create_raw_spectrogram(str(audio_file), str(output_path), clean_display=False)
                
                # 2. Spectrogramme brut sans éléments visuels
                #output_clean_path = raw_clean_dir / f"{audio_stem}_raw_clean.png"
                #create_raw_spectrogram(str(audio_file), str(output_clean_path), clean_display=True)
                
                # 3. Spectrogramme avec seuil du 3ème quartile (version croppée)
                #threshold_path = threshold_q3_dir / f"{audio_stem}_threshold_q3.png"
                #create_threshold_spectrogram(str(audio_file), str(threshold_path))
                
                # 4. Spectrogramme avec réduction de bruit et sauvegarde du fichier audio traité
                #noise_reduced_path = noise_reduced_dir / f"{audio_stem}_noise_reduced.png"
                #audio_output_path = audio_processed_dir / f"{audio_stem}_noise_reduced.wav"
                #create_noisereduced_spectrogram(str(audio_file), str(noise_reduced_path), str(audio_output_path))
                
                # 5. Spectrogramme Mel pour modèle d'IA (version standard)
                mel_model_path = mel_model_dir / f"{audio_stem}_mel_model.png"
                create_mel_spectrogram_for_model(str(audio_file), str(mel_model_path), None, apply_augmentation=False)
                
                # 6. Générer plusieurs versions augmentées du spectrogramme Mel dans le même dossier
                create_multiple_augmented_spectrograms(
                    str(audio_file),
                    mel_model_dir,  # Même dossier que les spectrogrammes standards
                    audio_stem,
                    num_augmentations=num_augmentations
                )

if __name__ == "__main__":
    # Si l'argument "nettoyer" est passé, supprimer tous les dossiers spectrograms
    if len(sys.argv) > 1 and sys.argv[1] == "nettoyer":
        supprimer_spectrograms()
    else:
        main() 