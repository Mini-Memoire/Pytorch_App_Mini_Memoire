import os
import torchaudio
from datasets import Dataset, DatasetDict, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer

def verify_and_correct_audio(audio_dir):
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                waveform, sample_rate = torchaudio.load(file_path)

                # Vérification et correction pour l'échantillonnage à 16 kHz
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    waveform = resampler(waveform)
                    sample_rate = 16000

                # Vérification et correction du nombre de canaux à mono
                if waveform.shape[0] != 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Sauvegarde du fichier corrigé
                torchaudio.save(file_path, waveform, sample_rate)

def verify_label_input_length(dataset, processor):
    for split in ['train', 'test']:
        for i, batch in enumerate(dataset[split]):
            audio = batch["audio"]
            labels = batch["sentence"]
            input_values = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
            label_ids = processor.tokenizer(labels).input_ids

            print(f"Batch {i} in split {split}: label length {len(label_ids)}, input length {len(input_values)}")

def main():
    audio_dir = "reconnaissance_vocale/files/data"
    verify_and_correct_audio(audio_dir)

    # Chargement des fichiers audios et des labels
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]
    labels = ["label"] * len(audio_files)

    # Création d'un dataset à partir des fichiers audio et des labels
    dataset = Dataset.from_dict({
        'audio': audio_files,
        'sentence': labels
    })

    # Chargement des fichiers audios
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Division du dataset en train et test
    train_test_split = dataset.train_test_split(test_size=0.2)
    dataset = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })

    processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all")
    verify_label_input_length(dataset, processor)

if __name__ == "__main__":
    main()
