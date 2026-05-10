import os
import torch
import soundfile as sf
from nemo.collections.tts.models import FastPitchModel, HifiGanModel

emotion_dir = "/home/dzejn/PycharmProjects/KNN_Project_TTS/sup_data_22050/emotion"
output_dir = "/home/dzejn/PycharmProjects/KNN_Project_TTS/out_dir"
sample_rate = 22050

spec_generator = FastPitchModel.restore_from(
    "/home/dzejn/PycharmProjects/NeMo/nemo_experiments/FastPitch/2026-05-09_16-11-55/checkpoints/FastPitch.nemo",
    map_location='cuda'
).eval().cuda()

vocoder = HifiGanModel.from_pretrained('tts_en_hifigan')

def load_emotion(emotion_file: str):
    path = os.path.join(emotion_dir, emotion_file)
    emotion = torch.load(path, weights_only=True).float()
    return emotion.unsqueeze(0).cuda()  # add batch dim

def synthesize(text, output_filename, pace=1.0, emotion_file=None):
    emotion = load_emotion(emotion_file) if emotion_file else None

    with torch.no_grad():
        tokens = spec_generator.parse(text)
        spectrogram = spec_generator.generate_spectrogram(
            tokens=tokens,
            pace=pace,
            emotion=emotion,
        )
        audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

    audio_np = audio.squeeze().cpu().numpy()
    output_path = os.path.join(output_dir, output_filename)
    sf.write(output_path, audio_np, sample_rate)
    print(f"Saved: {output_path} ({len(audio_np)/sample_rate:.1f}s)")
    return audio_np

tests = [
    ("Hello, I am happy to meet you!", "01_amused.wav",  "amused_1-28_0003_josh.pt"),
    ("Hello, I am happy to meet you!", "02_neutral.wav", "neutral_1-28_0002_jenie.pt"),
    ("Hello, I am happy to meet you!", "03_predicted.wav", None),
]

for text, filename, emotion_file in tests:
    synthesize(text, filename, emotion_file=emotion_file)