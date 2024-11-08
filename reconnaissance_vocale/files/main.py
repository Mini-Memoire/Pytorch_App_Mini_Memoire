from load_and_prepare_datasets import load_and_prepare_data
from train_model import train_model
from test_model import test_model

def main():
    data_dir = "reconnaissance_vocale/files/data"
    json_file = "data.json"
    
    # Charger et préparer les données
    print("Chargement et préparation des données...")
    dataset, labels = load_and_prepare_data(data_dir, json_file)
    print("Données chargées et préparées.")
    
    # Entraîner le modèle
    print("Entraînement du modèle...")
    train_model(dataset, labels)
    print("Modèle entraîné.")
    
    # Tester le modèle
    print("Test du modèle...")
    test_model(dataset, labels)
    print("Modèle testé.")

if __name__ == "__main__":
    main()