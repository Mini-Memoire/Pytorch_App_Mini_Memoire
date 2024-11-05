from datasets import load_dataset
from transformers import Wav2Vec2Processor
import librosa

def load_and_prepare_dataset(json_path):
    dataset = load_dataset("json", data_files=json_path, split="train")
    processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all")
    
    def preprocess_function(examples):
        audio, _ = librosa.load(examples["path"], sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_values[0]
        return {"input_values": inputs, "labels": processor.tokenizer(examples["sentence"]).input_ids}
    
    dataset = dataset.map(preprocess_function)
    return dataset, processor
