import os
import torch
import soundfile as sf
import argparse

from nemo.collections.tts.models import FastPitchModel, HifiGanModel

MODEL_PATHS = {
    "angry": {
        "spec": "angry_fastpitch_finetuned.nemo",
        "vocoder": "angry_hifigan_finetuned.nemo"
    },
    "neutral": {
        "spec": "neutral_fastpitch_finetuned.nemo",
        "vocoder": "neutral_hifigan_finetuned.nemo"
    },
    "happy": {
        "spec": "new_happy_fastpitch_finetuned.nemo",
        "vocoder": "new_happy_hifigan_finetuned.nemo"
    },
    "sad": {
        "spec": "new_sad_fastpitch_finetuned.nemo",
        "vocoder": "new_sad_hifigan_finetuned.nemo"
    },
    "surprised": {
        "spec": "new_surprised_fastpitch_finetuned.nemo",
        "vocoder": "new_surprised_hifigan_finetuned.nemo"
    }
}


def load_models(model_path, emotion):
    if emotion not in MODEL_PATHS:
        raise ValueError(f"Emotion '{emotion}' not supported. Available emotions: {list(MODEL_PATHS.keys())}")
    paths =MODEL_PATHS[emotion]

    spec_generator = FastPitchModel.restore_from(
        model_path + paths["spec"],
        map_location='cuda',
        strict=False
    ).eval().cuda()

    vocoder = HifiGanModel.restore_from(
        model_path + paths["vocoder"],
        strict=False
    ).eval().cuda()

    return spec_generator, vocoder


def synthesize(spec_generator, vocoder, text, output_path="output.wav", pace=1.0):
    with torch.no_grad():
        tokens = spec_generator.parse(text)
        spectrogram = spec_generator.generate_spectrogram(tokens=tokens, pace=pace)
        audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

    audio_np = audio.squeeze().cpu().numpy()

    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    sf.write(output_path, audio_np, 22050)
    duration = len(audio_np) / 22050
    print(f"Saved: {output_path} ({duration:.1f}s)")
    return audio_np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EmoTTSe – Emotion-based Text-to-Speech System")
    parser.add_argument("-m", "--model_path", required=True, help="Path to all models")
    parser.add_argument("-s", "--sentence", required=True, help="Text to synthesize")
    parser.add_argument("-e", "--emotion", required=True, help="Emotion (angry, neutral)")
    parser.add_argument("-o", "--output", default="output.wav", help="Output file path")

    args = parser.parse_args()

    try:
        print("Loading models...")
        spec_gen, voc = load_models(args.model_path, args.emotion)
        print("Synthesize voice...")
        synthesize(spec_gen, voc, args.sentence, args.output)

        del spec_gen, voc
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)