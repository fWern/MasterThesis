from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch
from tqdm import tqdm
import librosa
from utils import device

model_name = "facebook/wav2vec2-large-960h-lv60-self"
model = Wav2Vec2Model.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)
model.to(device)

def extract_features_wav(audio_paths):
    audio_features = []
    model.eval()
    for audio_path in tqdm(audio_paths):
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            inputs = processor(audio, sampling_rate=sr, return_tensors='pt', padding=True).to(device)

            with torch.no_grad():
                features = model(**inputs).last_hidden_state
            print(features.shape)
            features = features.mean(dim=1)
            features = features.squeeze(0)
            print(features)
            audio_features.append(features)
        except:
            audio_features.append(torch.zeros(1024).to(device))
    return torch.stack(audio_features)

