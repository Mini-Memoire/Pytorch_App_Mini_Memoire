from generate_data_json import generate_data_json
from load_and_prepare_datasets import load_and_prepare_dataset
from train_model import train_model
from test_model import test_model

# Génération du fichier JSON des données
generate_data_json("reconnaissance_vocale/files/data", "data.json")

# Chargement et préparation des données
dataset, processor = load_and_prepare_dataset("data.json")

# Entraînement du modèle
model = train_model(dataset, processor)

# Test du modèle sur un fichier audio
test_model(model, processor, "reconnaissance_vocale/files/data/Solo_1_valiha.wav")
