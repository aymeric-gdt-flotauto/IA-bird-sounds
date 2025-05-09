# Générateur de Spectrogrammes pour Audios d'Oiseaux

Ce script Python permet de générer différents types de spectrogrammes à partir de fichiers audio d'oiseaux au format .ogg.

## Prérequis

- Python 3.6 ou supérieur
- Les bibliothèques listées dans `requirements.txt`

## Installation

1. Installez les dépendances requises:

```bash
pip install -r requirements.txt
```

## Utilisation

1. Assurez-vous que vos fichiers audio sont organisés dans la structure suivante:
   ```
   dataset/
     └── bird_audio/
         ├── espece1/
         │   ├── audio1.ogg
         │   ├── audio2.ogg
         │   └── ...
         ├── espece2/
         │   ├── audio1.ogg
         │   └── ...
         └── ...
   ```

2. Exécutez le script:

```bash
python create_spectrograms.py
```

3. Les spectrogrammes seront créés dans une structure de dossiers organisée:
   ```
   dataset/
     └── bird_audio/
         ├── espece1/
         │   ├── audio1.ogg
         │   ├── audio2.ogg
         │   ├── audio_processed/
         │   │   ├── audio1_noise_reduced.wav
         │   │   ├── audio2_noise_reduced.wav
         │   │   └── ...
         │   └── spectrograms/
         │       ├── raw/
         │       │   ├── audio1_raw.png
         │       │   ├── audio2_raw.png
         │       │   └── ...
         │       ├── raw_clean/
         │       │   ├── audio1_raw_clean.png
         │       │   ├── audio2_raw_clean.png
         │       │   └── ...
         │       ├── threshold_q3/
         │       │   ├── audio1_threshold_q3.png
         │       │   ├── audio2_threshold_q3.png
         │       │   └── ...
         │       ├── noise_reduced/
         │       │   ├── audio1_noise_reduced.png
         │       │   ├── audio2_noise_reduced.png
         │       │   └── ...
         │       └── mel_model/
         │           ├── audio1_mel_model.png
         │           ├── audio1_aug1.png
         │           ├── audio1_aug2.png
         │           ├── audio1_aug3.png
         │           ├── audio1_aug4.png
         │           ├── audio1_aug5.png
         │           ├── audio2_mel_model.png
         │           ├── audio2_aug1.png
         │           ├── audio2_aug2.png
         │           └── ...
         └── ...
   ```

## Types de Spectrogrammes

Le script génère les types de spectrogrammes suivants:

- **Raw (Brut)**: Spectrogramme standard montrant la distribution des fréquences au fil du temps, avec axes, titre et barre de couleur.
- **Raw Clean (Brut épuré)**: Version épurée du spectrogramme brut, sans axes, titre ni barre de couleur. Idéal pour le traitement d'image ou la visualisation minimaliste.
- **Threshold Q3**: Spectrogramme épuré ne conservant que les signaux dont le niveau sonore est supérieur au 3ème quartile des amplitudes. Cette version est croppée en supprimant 150 pixels du bas de l'image pour se concentrer sur les fréquences les plus pertinentes. Sans éléments visuels périphériques.
- **Noise Reduced**: Spectrogramme généré après application d'une réduction de bruit sur l'audio. La réduction de bruit utilise la bibliothèque noisereduce qui estime le profil de bruit à partir de la première seconde de l'enregistrement. Ce spectrogramme permet de mieux isoler les vocalisations des oiseaux en réduisant le bruit de fond.
- **Mel Model**: Spectrogrammes Mel optimisés pour les modèles de deep learning. Deux types sont générés dans le même dossier:
  - Version standard: `audio_name_mel_model.png`
  - Versions augmentées: `audio_name_aug1.png`, `audio_name_aug2.png`, etc.
  
  Caractéristiques communes:
  - Réduction de bruit appliquée sur l'audio
  - Signal audio standardisé à une longueur de 1,000,000 échantillons
  - 256 bandes Mel
  - Conversion en image 3 canaux (RGB)
  - Normalisation ImageNet classique appliquée pour le traitement

  Les versions augmentées incluent diverses transformations:
  - **Augmentations au niveau du signal audio**:
    - Time shifting: Décalage temporel aléatoire du signal pour simuler différents moments d'enregistrement
    - Pitch shifting: Modification légère de la hauteur (±2 demi-tons) pour simuler différents individus
    - Time stretching: Étirement temporel (±10%) pour simuler différentes vitesses de chant
    - Ajout de bruit: Légère addition de bruit aléatoire pour améliorer la robustesse
  - **Augmentations au niveau du spectrogramme**:
    - Time masking: Masquage aléatoire de segments temporels pour simuler des interruptions
    - Frequency masking: Masquage aléatoire de bandes de fréquences pour simuler des occlusions
    - Transformations géométriques légères: Petits décalages et mises à l'échelle
    - Ajustements de luminosité/contraste: Pour simuler différentes conditions d'enregistrement

## Fichiers audio traités

En plus des spectrogrammes, le script génère également les fichiers audio traités:

- **Fichiers avec réduction de bruit**: Les fichiers audio après traitement par l'algorithme de réduction de bruit sont sauvegardés au format WAV dans le dossier `audio_processed/`. Ces fichiers peuvent être utilisés pour l'analyse audio ou pour l'écoute des vocalisations avec moins de bruit de fond.

## Nettoyage des fichiers générés

Pour supprimer tous les dossiers de spectrogrammes et d'audio traités:

```bash
python create_spectrograms.py nettoyer
```

Cette commande supprimera tous les dossiers `spectrograms` et `audio_processed` dans chaque dossier d'espèce d'oiseau.

## Personnalisation

Vous pouvez facilement modifier le nombre d'augmentations générées pour chaque fichier audio en ajustant la variable `num_augmentations` dans la fonction `main()` du script.

## Résultat

Pour chaque fichier audio .ogg, un spectrogramme au format PNG sera généré, montrant la distribution des fréquences au fil du temps.
