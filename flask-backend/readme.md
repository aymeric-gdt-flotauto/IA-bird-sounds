# Comment lancer le server Flask

## 🔧 Installation des dépendances Python (Backend Flask)

Ce projet utilise Flask pour le backend et un modèle d'IA `.keras` pour reconnaître les chants d'oiseaux à partir de fichiers audio. Voici les dépendances nécessaires.

### 🐍 Installation via `pip`

Installe tous les packages requis avec la commande suivante :

```bash
pip install flask flask-cors tensorflow pydub librosa numpy soundfile
```

📦 Packages utilisés
| Package      | Description                                                       |
| ------------ | ----------------------------------------------------------------- |
| `flask`      | Serveur web léger pour gérer les requêtes HTTP                    |
| `flask-cors` | Autorise les requêtes Cross-Origin depuis le frontend React       |
| `tensorflow` | Permet de charger et exécuter le modèle d\'IA `.keras`            |
| `pydub`      | Utilisé pour convertir les fichiers audio en `.ogg`               |
| `librosa`    | Extraction de caractéristiques audio (MFCC, spectrogrammes, etc.) |
| `numpy`      | Manipulation des tableaux numériques                              |
| `soundfile`  | Backend utilisé par `librosa` pour la lecture des fichiers audio  |

#### ⚠️ Dépendance système : ffmpeg

Le module pydub nécessite ffmpeg pour convertir les fichiers audio (ex : mp3, wav → ogg).
📥 Installation de ffmpeg

    Ubuntu / Debian :

        sudo apt update
        sudo apt install ffmpeg

macOS (Homebrew) :

    brew install ffmpeg

    Windows :

        🪟 Installation de FFmpeg sur Windows

            Le module `pydub` utilise `ffmpeg` pour lire et convertir les fichiers audio (ex : mp3, wav, m4a → ogg). Voici comment l’installer correctement sur Windows :

        📥 Étape 1 : Télécharger FFmpeg

            1. Rendez-vous sur le site officiel de téléchargement :  
            👉 [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

            2. Cliquez sur **Windows** > puis **Windows builds from gyan.dev**

            3. Sur la page, téléchargez la version recommandée :  
            **"ffmpeg-release-essentials.zip"**

        📁 Étape 2 : Extraire les fichiers

            1. Une fois le `.zip` téléchargé, fais un clic droit dessus et **extrais** tout (clic droit → Extraire tout...).

            2. Renomme le dossier extrait en `ffmpeg` pour simplifier.

            3. Place ce dossier dans un endroit stable, par exemple :  
            `C:\Program Files\ffmpeg`

        🛠️ Étape 3 : Ajouter FFmpeg au PATH

            1. Ouvre le menu **Démarrer**, cherche **Variables d’environnement** et clique sur :

                `Modifier les variables d’environnement système`

            2. Dans la fenêtre **Propriétés système**, clique sur **Variables d’environnement...**

            3. Dans la section **Variables système**, trouve et sélectionne la variable `Path`, puis clique sur **Modifier...**

            4. Clique sur **Nouveau**, et ajoute le chemin vers le dossier `bin` de FFmpeg, par exemple :

                C:\Program Files\ffmpeg\bin

            5. Clique sur **OK** pour fermer toutes les fenêtres.

        ✅ Étape 4 : Vérifier l’installation

            1. Ouvre une **Invite de commandes (cmd)**

            2. Tape :

                ```bash
                ffmpeg -version
                ```

                Si tout est bien configuré, tu verras la version installée s’afficher.

            FFmpeg est maintenant installé et prêt à être utilisé par pydub dans ton backend Flask.


Une fois toutes les dépendances installées, vous pouvez lancer le serveur Flask avec :

        ```bash
        cd flask-backend
        $env:FLASK_APP = "app.py" #(Pour Powershell,i.e terminal VSCode. Si sur cmd windows ou autre, c'est une autre commande)
        flask run
        ```

Et accéder au backend sur http://localhost:5000.

