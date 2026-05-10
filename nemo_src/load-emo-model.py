import os
import torch
import soundfile as sf
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
import matplotlib.pyplot as plt

EMOTION_TO_ID = {
    "neutral": 0.0,
    "amused": 1.0,
    "angry": 2.0,
    "disgust": 3.0,
    "happy": 4.0,
    "sad": 5.0,
    "ps": 6.0,
}

output_dir = "/home/alex/Documents/KNN/out_dir"
sample_rate = 22050

spec_generator = FastPitchModel.restore_from(
    #"/home/alex/Documents/KNN/2026-05-10_08-51-25/checkpoints/FastPitch.nemo"
    "/home/alex/Documents/KNN/first-model/FastPitch.nemo"
).eval().cuda()

vocoder = HifiGanModel.restore_from(
    "/home/alex/Documents/KNN/2026-05-10_08-51-25/checkpoints/hifigan_finetuned.nemo"
).eval().cuda()


def synthesize(text, output_filename, pace=1.0, emotion=None):
    with torch.no_grad():
        tokens = spec_generator.parse(text)

        # Create per-token emotion tensor
        if emotion is not None:
            emotion_id = EMOTION_TO_ID.get(emotion, 0.0)
            emotion_tensor = torch.full((1, tokens.shape[1]), emotion_id).cuda()
        else:
            emotion_tensor = None

        spectrogram = spec_generator.generate_spectrogram(
            tokens=tokens,
            pace=pace,
            emotion=emotion_tensor,
        )
        audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

    audio_np = audio.squeeze().cpu().numpy()
    output_path = os.path.join(output_dir, output_filename)
    sf.write(output_path, audio_np, sample_rate)
    print(f"Saved: {output_path} ({len(audio_np)/sample_rate:.1f}s)")
    return audio_np


tests = [
    ("Hello, I am happy to meet you!", "01_angry.wav", "angry"),
    ("Hello, I am happy to meet you!", "02_neutral.wav", "neutral"),
    ("Hello, I am happy to meet you!", "03_amused.wav", "amused"),
    ("Hello, I am happy to meet you!", "04_sad.wav", "sad"),
    ("Hello, I am happy to meet you!", "05_no_emotion.wav", None),
]

for text, filename, emotion in tests:
    synthesize(text, filename, emotion=emotion)