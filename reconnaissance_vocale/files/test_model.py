import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def test_model(dataset, labels):
    model_name = "facebook/mms-1b-all"
    
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        print("Processeur chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du processeur : {e}")
        return
    
    try:
        print("Chargement du modèle...")
        model = Wav2Vec2ForCTC.from_pretrained(model_name, num_labels=len(set(labels)))
        print("Modèle chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return
    
    def predict(audio_file):
        try:
            print(f"Chargement du fichier audio : {audio_file}")
            audio_input, sample_rate = torchaudio.load(audio_file)
            print(f"Fichier audio chargé avec succès. Sample rate: {sample_rate}")
            print(f"Audio input shape: {audio_input.shape}")

            # Rééchantillonnage à 16000 Hz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                audio_input = resampler(audio_input)
                sample_rate = 16000
                print(f"Audio rééchantillonné à {sample_rate} Hz")

            input_values = processor(audio_input.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_values
            print(f"Valeurs d'entrée préparées. Input values shape: {input_values.shape}")
            logits = model(input_values).logits
            print("Logits calculés.")
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
            print(f"Transcription brute: {transcription}")
            return transcription
        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}")
            return None

    test_audio_file = "reconnaissance_vocale/files/data/Tino_1_valiha.wav"
    print(f"Test du fichier audio : {test_audio_file}")
    transcription = predict(test_audio_file)
    if transcription:
        print(f"Transcription: {transcription}")
    else:
        print("La transcription a échoué.")

if __name__ == "__main__":
    from load_and_prepare_datasets import load_and_prepare_data
    data_dir = "reconnaissance_vocale/files/data"
    json_file = "data.json"
    print("Chargement des données...")
    dataset, labels = load_and_prepare_data(data_dir, json_file)
    print("Données chargées.")
    test_model(dataset, labels)