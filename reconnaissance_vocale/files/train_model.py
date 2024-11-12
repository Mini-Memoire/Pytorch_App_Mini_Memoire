import os
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments

def train_model(dataset, labels):
    model_name = "facebook/mms-1b-all"
    
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        print("Processeur chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du processeur : {e}")
        return
    
    try:
        num_labels = len(processor.tokenizer)
        print(f"Nombre de labels : {num_labels}")
        model = Wav2Vec2ForCTC.from_pretrained(model_name, num_labels=num_labels)
        print("Modèle chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return

    def preprocess_function(examples):
        audio = examples["audio"]
        examples["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"], padding=True).input_values[0]
        examples["labels"] = processor.tokenizer(examples["sentence"], padding=True).input_ids
        return examples

    print("Prétraitement des données...")
    dataset = dataset.map(preprocess_function, remove_columns=["audio"])
    print("Prétraitement terminé.")

    # Vérifiez les dimensions des labels
    def check_labels(examples):
        labels = examples["labels"]
        input_values = examples["input_values"]
        if len(labels) != len(input_values):
            print(f"Label length {len(labels)} does not match input length {len(input_values)}")
        return examples

    dataset = dataset.map(check_labels)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor.feature_extractor,
        data_collator=lambda data: processor.pad(data, padding=True)
    )

    print("Début de l'entraînement...")
    trainer.train()
    print("Entraînement terminé.")

    # Vérifiez que le répertoire ./results existe
    if not os.path.exists("./results"):
        os.makedirs("./results")
        print("Répertoire ./results créé.")
    else:
        print("Répertoire ./results existe déjà.")

    # Sauvegarder le modèle et le processeur
    try:
        print("Sauvegarde du modèle...")
        model.save_pretrained("./results")
        print("Modèle sauvegardé avec succès.")
        print("Sauvegarde du processeur...")
        processor.save_pretrained("./results")
        print("Processeur sauvegardé avec succès.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle et du processeur : {e}")

if __name__ == "__main__":
    from load_and_prepare_datasets import load_and_prepare_data
    data_dir = "reconnaissance_vocale/files/data"
    json_file = "data.json"
    print("Chargement des données...")
    dataset, labels = load_and_prepare_data(data_dir, json_file)
    print("Données chargées.")
    train_model(dataset, labels)