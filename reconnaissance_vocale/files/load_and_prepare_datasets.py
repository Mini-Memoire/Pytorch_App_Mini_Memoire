import json
from pathlib import Path
from datasets import Dataset, DatasetDict, Audio

def load_and_prepare_data(data_dir, json_file):
    data_dir = Path(data_dir)
    json_file = Path(json_file)

    with json_file.open('r') as f:
        data = json.load(f)

    audio_files = [item['path'] for item in data]
    labels = [item['sentence'] for item in data]

    # Création d'un dataset à partir des fichiers audio et des labels
    dataset = Dataset.from_dict({
        'audio': audio_files,
        'sentence': labels
    })

    # Chargement des fichiers audio
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Division du dataset en train et test
    train_test_split = dataset.train_test_split(test_size=0.2)
    dataset = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })

    return dataset, labels

if __name__ == "__main__":
    data_dir = "reconnaissance_vocale/files/data"
    json_file = "data.json"
    dataset, labels = load_and_prepare_data(data_dir, json_file)
    print("Données chargées et préparées.")