from pathlib import Path
import json

def generate_data_json(directory, output_file):
    data = []
    
    # Parcours tous les fichiers dans le dossier spécifié
    for filepath in Path(directory).glob("*.wav"):
        # Obtient le nom du fichier sans l'extension
        filename = filepath.stem
        
        # Extrait le mot après le dernier underscore "_"
        word = filename.split('_')[-1]
        
        # Ajoute l'entrée au format souhaité dans la liste
        data.append({"path": str(filepath), "sentence": word})
    
    # Enregistre la liste au format JSON
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Données enregistrées dans {output_file}")

