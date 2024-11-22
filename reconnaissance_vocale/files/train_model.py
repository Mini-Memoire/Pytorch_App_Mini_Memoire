import os
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
from datasets import Dataset

def verify_labels(labels, vocab_size, pad_token_id):
    """Vérifie que toutes les étiquettes sont dans la taille du vocabulaire et remplace les étiquettes invalides 
    par le jeton de padding"""
    return [label if label < vocab_size else pad_token_id for label in labels]

def preprocess_function(examples):
    input_values = []
    labels = []
    for audio, sentence in zip(examples["audio"], examples["sentence"]):
        input_value = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_values[0]
        label = processor.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).input_ids[0]
        label = verify_labels(label.tolist(), processor.tokenizer.vocab_size, processor.tokenizer.pad_token_id)
        input_values.append(input_value)
        labels.append(torch.tensor(label))
    
    return {"input_values": input_values, "labels": labels}

def train_model(dataset):
    model_name = "facebook/mms-1b-all"
    
    try:
        # Chargement du processeur
        global processor
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        print("Processeur chargé.")
    except Exception as e:
        print(f"Erreur lors du chargement du processeur : {e}")
        return
    
    try:
        num_labels = len(processor.tokenizer)
        print(f"Nombre de labels : {num_labels}")
        # Chargement du modèle
        model = Wav2Vec2ForCTC.from_pretrained(model_name, num_labels=num_labels)
        print("Modèle chargé.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return

    # Prétraitement de données
    print("Début du prétraitement.")
    dataset = dataset.map(preprocess_function, remove_columns=["audio"], batched=True)
    print("Prétraitement terminé.")

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5, 
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
    )

    def data_collator(features):
        # Convert lists to tensors and pad
        input_values = torch.nn.utils.rnn.pad_sequence([torch.tensor(feature["input_values"], dtype=torch.float32) for feature in features], batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(feature["labels"], dtype=torch.int64) for feature in features], batch_first=True, padding_value=processor.tokenizer.pad_token_id)

        batch = {
            "input_values": input_values,
            "labels": labels
        }
        return batch

    print("Préparation de l'entraîneur.")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor.feature_extractor,
        data_collator=data_collator
    )
    print("Entraîneur prêt.")

    # Entraînement du modèle
    print("Début de l'entraînement.")
    trainer.train()
    print("Entraînement terminé.")

    # Sauvegarde du modèle et du processeur dans un dossier results
    try:
        model.save_pretrained("./results")
        processor.save_pretrained("./results")
        print("Fin de la sauvegarde")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle et du processeur : {e}")

if __name__ == "__main__":
    from load_and_prepare_datasets import load_and_prepare_data
    data_dir = "reconnaissance_vocale/files/data"
    json_file = "data.json"
    dataset, labels = load_and_prepare_data(data_dir, json_file)
    print("Données chargées.")
    train_model(dataset)
