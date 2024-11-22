import pyaudio
import wave
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

print("Importations réussies")

model_name = "facebook/mms-1b-all"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

print("Modèle chargé")

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5

audio = pyaudio.PyAudio()

print("PyAudio configuré")

stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Enregistrement en cours...")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Enregistrement terminé.")

stream.stop_stream()
stream.close()
audio.terminate()

print("Audio terminé")

# Convertion des données audio en tableau numpy
audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

# Convertion des données audio en float32
audio_data = audio_data.astype(np.float32)

# Normalisation des données audio
input_values = processor(audio_data, sampling_rate=RATE, return_tensors="pt").input_values

print("Données audio normalisées")

# Transcription
with torch.no_grad():
    logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

print("Transcription : ", transcription)