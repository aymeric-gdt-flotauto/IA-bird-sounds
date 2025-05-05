import torch
import torch.nn as nn
from torchvision.models import resnet34
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

# === CONFIG ===
MODEL_PATH = r"C:\Users\FabienETHEVE\OneDrive - ARTIMON\Bureau\bird\IA-bird-sounds\models\resnet.pt"
DATA_PATH = r"C:\Users\FabienETHEVE\OneDrive - ARTIMON\Bureau\bird\IA-bird-sounds\dataset2\train_data"
NUM_CLASSES = 6  # Tu as 6 classes

# === CLASSE MODELE ===
class BirdNet(nn.Module):
    def __init__(self, f, o):
        super(BirdNet, self).__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.dense_output = nn.Linear(f, o)
        self.resnet = resnet34(pretrained=True)
        self.resnet_head = nn.Sequential(*list(self.resnet.children())[:-1])  # Utiliser toutes les couches sauf la dernière

    def forward(self, x):
        x = self.resnet_head(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.dense_output(self.dropout(x))

# === INITIALISATION DU MODELE ===
print("[INFO] Initialisation du modèle...")
netword = BirdNet(f=512, o=NUM_CLASSES)  # 6 classes

print(f"[INFO] Chargement du modèle depuis {MODEL_PATH}...")
state_dict = torch.load(MODEL_PATH, map_location='cpu')

# Ignorer les poids de la couche finale 'dense_output' du modèle pré-existant
del state_dict['dense_output.weight']
del state_dict['dense_output.bias']

# Charger le reste des poids
netword.load_state_dict(state_dict, strict=False)  # strict=False permet de ne pas échouer pour la couche dense_output
netword.eval()
print("[INFO] Modèle chargé avec succès !")

# === TRANSFORMATIONS ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === CHARGEMENT DES DONNÉES ===
dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
classes = dataset.classes
print(f"[INFO] {len(classes)} classes détectées : {classes[:5]}...")

# === TEST DE PRÉDICTION ===
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloader):
        outputs = netword(inputs)
        
        # Calcul des probabilités via softmax
        probs = F.softmax(outputs, dim=1)
        
        # Obtenir l'indice de la classe prédite et la probabilité
        _, preds = torch.max(probs, 1)
        prob = probs[0, preds].item() * 100  # Probabilité en pourcentage

        # Vérifier que l'indice de l'étiquette réelle est dans les limites des classes
        if labels.item() < len(classes):
            true_class = classes[labels.item()]
        else:
            print(f"[ERROR] L'étiquette réelle {labels.item()} est hors des limites des classes.")
            continue  # Passer à l'image suivante

        # Vérifier que l'indice de la prédiction est dans les limites des classes
        if preds.item() < len(classes):
            predicted_class = classes[preds.item()]
        else:
            print(f"[ERROR] L'indice de prédiction {preds.item()} est hors des limites des classes.")
            continue  # Passer à l'image suivante

        print(f"[INFO] Vrai : {true_class} | Prédiction : {predicted_class} (Précision : {prob:.2f}%)")

        # Affichage de l'image
        img = inputs[0].permute(1, 2, 0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # dé-normalisation
        img = img.clip(0, 1)
        
        # Affichage de l'image avec la précision en pourcentage dans le titre
        plt.imshow(img)
        plt.title(f"Prédiction : {predicted_class} ({prob:.2f}%)")
        plt.axis('off')

        # Attente de l'interaction avec la touche 'q' pour quitter
        plt.show()
        key = plt.waitforbuttonpress(timeout=0)  # Attente d'un clic ou d'une touche
        if key:
            print("[INFO] Arrêt de la visualisation.")
            break  # Arrêter la boucle après un clic ou une touche

        plt.close()  # Fermer la fenêtre d'affichage pour la prochaine image
