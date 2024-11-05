from transformers import Wav2Vec2ForCTC, Trainer, TrainingArguments

def train_model(dataset, processor):
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
    training_args = TrainingArguments(
        output_dir="./mms-malagasy",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=10,
        fp16=True,
        learning_rate=1e-4,
        logging_dir="./logs"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor.tokenizer,
    )
    trainer.train()
    return model
