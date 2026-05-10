import torchaudio
import json
import os
import glob
from transformers import pipeline

def predict_emotion(audio_path: str, pipe):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    waveform = waveform.mean(dim=0).numpy()  # mono

    results = pipe(waveform, sampling_rate=16000, top_k=8)
    return results

def main():
    audio_dir = "/home/alex/Documents/KNN/baseline_data/surprised"
    output_file = "new_suprised.jsonl"

    pipe = pipeline("audio-classification", model="/home/alex/Documents/KNN/speech-emotion-recognition-with-openai-whisper-large-v3",device=0)

    with open(output_file, "w") as f:
        for audio_path in sorted(glob.glob(os.path.join(audio_dir, "test_*.wav"))):
            print(f"Processing: {audio_path}")
            res = pipe(audio_path)
            top_emotion = res[0]["label"]  # top-1 prediction

            entry = {
                "audio": os.path.basename(audio_path),
                "emotion": top_emotion,
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Done. Results saved to {output_file}")

if __name__ == "__main__":
    main()