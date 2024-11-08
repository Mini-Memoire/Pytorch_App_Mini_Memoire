from generate_data_json import generate_data_json
from load_and_prepare_datasets import load_and_prepare_dataset
from train_model import train_model
from test_model import test_model

# Génération du fichier JSON des données
generate_data_json("reconnaissance_vocale/files/data", "data.json")
