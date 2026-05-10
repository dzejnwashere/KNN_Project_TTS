# save as: extract_emotion_from_filenames.py

import torch
import os
import re
from pathlib import Path

EMOTION_TO_ID = {
    "neutral": 0.0,
    "amused": 1.0,
    "anger": 2.0,
    "angry": 2.0,
    "disgust": 3.0,
    "happy": 4.0,
    "sad": 5.0,
    "ps": 6.0,
}

def parse_emotion_from_filename(filename):
    basename = Path(filename).stem.lower()
    # OAF/YAF format: OAF_word_emotion.wav
    if basename.startswith(("oaf_", "yaf_")):
        emotion = basename.rsplit("_", 1)[-1]
    else:
        # e.g. "amused_113-136_0113_jenie" → "amused"
        emotion = basename.split("_", 1)[0]
    return EMOTION_TO_ID.get(emotion, 0.0)

def main():
    import json
    import argparse
    import librosa

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to manifest .json file")
    parser.add_argument("--sup_data_path", required=True, help="Path to sup_data folder")
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--sample_rate", type=int, default=22050)
    args = parser.parse_args()

    emotion_dir = Path(args.sup_data_path) / "emotion"
    emotion_dir.mkdir(parents=True, exist_ok=True)

    # Read manifest to get audio paths
    manifest_entries = []
    with open(args.manifest, "r") as f:
        for line in f:
            manifest_entries.append(json.loads(line))

    # Determine base_data_dir (common prefix of all audio paths)
    all_paths = [e["audio_filepath"] for e in manifest_entries]
    base_dir = os.path.commonpath(all_paths)
    if not os.path.isdir(base_dir):
        base_dir = os.path.dirname(base_dir)

    for entry in manifest_entries:
        audio_path = entry["audio_filepath"]

        # Build the same rel_audio_path_as_text_id that TTSDataset uses
        rel_path = Path(audio_path).relative_to(base_dir).with_suffix("")
        rel_audio_path_as_text_id = str(rel_path).replace("/", "_")

        out_path = emotion_dir / f"{rel_audio_path_as_text_id}.pt"
        if out_path.exists():
            continue

        # Parse emotion label from filename
        emotion_id = parse_emotion_from_filename(audio_path)

        # Get audio duration to compute number of spectrogram frames
        duration = entry.get("duration", None)
        if duration is not None:
            n_frames = int(duration * args.sample_rate / args.hop_length) + 1
        else:
            import soundfile as sf
            info = sf.info(audio_path)
            n_frames = int(info.frames / args.hop_length) + 1

        # Create constant emotion tensor (same value for every frame)
        emotion_tensor = torch.full((n_frames,), emotion_id, dtype=torch.float32)
        torch.save(emotion_tensor, out_path)

    print(f"Done! Saved {len(manifest_entries)} emotion files to {emotion_dir}")

if __name__ == "__main__":
    main()