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
        examples["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        examples["labels"] = processor.tokenizer(examples["sentence"]).input_ids
        return examples

    dataset = dataset.map(preprocess_function, remove_columns=["audio"])

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
    )

    trainer.train()

if __name__ == "__main__":
    from load_and_prepare_datasets import load_and_prepare_data
    data_dir = "reconnaissance_vocale/files/data"
    json_file = "data.json"
    dataset, labels = load_and_prepare_data(data_dir, json_file)
    train_model(dataset, labels)