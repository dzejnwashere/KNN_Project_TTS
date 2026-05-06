import torchaudio
from transformers import pipeline
import json


def predict_emotion(audio_path: str, pipe):
    # Load and resample to 16kHz
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    waveform = waveform.mean(dim=0).numpy()  # mono

    results = pipe(waveform, sampling_rate=16000, top_k=8)

    #print("\n🎙️ Emotion Predictions:")
    #for r in results:
    #    bar = "█" * int(r['score'] * 30)
    #    print(f"  {r['label']:<12} {r['score']:.2%}  {bar}")

    #print(f"\n✅ Detected: {results[0]['label'].upper()}")
    return results

def main():
    """pipe1 = pipeline(
        "audio-classification",
        model="/home/alex/Documents/KNN/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )

    pipe2 = pipeline(
        "audio-classification",
        model="/home/alex/Documents/KNN/wav2vec2-base-superb-er"
    )"""

    pipe3 = pipeline(
        "audio-classification",
        model="/home/alex/Documents/KNN/speech-emotion-recognition-with-openai-whisper-large-v3"
    )

    with open("emo_base.jsonl", "r") as input_file:
        for line in input_file:
            json_line = json.loads(line)
            """
            res1 = predict_emotion(json_line["audio_path"], pipe1)
            res2 = predict_emotion(json_line["audio_path"], pipe2)
            res3 = predict_emotion(json_line["audio_path"], pipe3)

            json_line["pipe1"] = res1
            json_line["pipe2"] = res2
            json_line["pipe3"] = res3"""

            res = predict_emotion(json_line["audio_path"], pipe3)


            with open("new_processed.jsonl", "a") as output_file:
                output_file.write(json.dumps(json_line) + "\n")

if __name__ == "__main__":
    main()