import torch
import librosa

def test_model(model, processor, audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred_ids)[0]
    print("Transcription:", transcription)
