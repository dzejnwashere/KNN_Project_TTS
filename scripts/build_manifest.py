"""
Usage:
    python build_manifest.py \
        --wav_dir /path/to/wavs \
        --output_dir /path/to/output \
        --val_ratio 0.05 \
        --asr_model "nvidia/parakeet-tdt-0.6b-v2" \
        --batch_size 16
"""

import argparse
import json
import os
import random
from pathlib import Path

import librosa
import torch
from tqdm import tqdm

KNOWN_EMOTIONS = {"happy", "amused", "angry", "neutral", "sad", "surprised"}


def extract_emotion_from_filename(filepath):
    stem = Path(filepath).stem.lower()
    segments = stem.split("_")
    for segment in segments:
        if segment in KNOWN_EMOTIONS:
            return segment
    return None


def get_duration(audio_path, sample_rate):
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    return len(audio) / sr


def transcribe_batch(asr_model, audio_paths):
    outputs = asr_model.transcribe(audio_paths)
    if isinstance(outputs, list) and len(outputs) > 0:
        if hasattr(outputs[0], 'text'):
            return [o.text for o in outputs]
        else:
            return outputs
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Build Parakeet TTS manifest"
    )
    parser.add_argument(
        "--wav_dir", type=Path, required=True,
        help="Directory containing .wav files (searched recursively)"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Directory where manifest files will be saved"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=44100,
        help="Sample rate of your audio files"
    )
    parser.add_argument(
        "--asr_model", type=str, default="nvidia/parakeet-tdt-0.6b-v2",
        help="Pretrained ASR model name or path"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for ASR transcription"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.05,
        help="Fraction of data to use for validation (default 5%%)"
    )
    parser.add_argument(
        "--min_duration", type=float, default=0.5,
        help="Skip audio files shorter than this (seconds)"
    )
    parser.add_argument(
        "--max_duration", type=float, default=15.0,
        help="Skip audio files longer than this (seconds)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning for .wav files...")
    wav_files = sorted(args.wav_dir.rglob("*.wav"))
    print(f"  Found {len(wav_files)} .wav files")

    if not wav_files:
        print("ERROR: No .wav files found. Check --wav_dir path.")
        return

    valid_files = []
    skipped_emotion = 0
    for wav_path in wav_files:
        emotion = extract_emotion_from_filename(wav_path)
        if emotion is None:
            skipped_emotion += 1
            continue
        valid_files.append((str(wav_path.resolve()), emotion))

    print(f"  Valid emotion prefix: {len(valid_files)}")
    if skipped_emotion:
        print(f"  Skipped (no emotion match): {skipped_emotion}")

    if not valid_files:
        print("ERROR: No files matched known emotions. Check filenames and KNOWN_EMOTIONS.")
        return

    print("\nComputing durations...")
    entries = []
    skipped_duration = 0
    for audio_path, emotion in tqdm(valid_files, desc="Duration check"):
        try:
            dur = get_duration(audio_path, args.sample_rate)
        except Exception as e:
            print(f"  WARNING: Could not read {audio_path}: {e}")
            continue

        if dur < args.min_duration or dur > args.max_duration:
            skipped_duration += 1
            continue

        entries.append({
            "audio_filepath": audio_path,
            "emotion": emotion,
            "duration": round(dur, 4),
        })

    print(f"  Kept: {len(entries)}")
    if skipped_duration:
        print(f"  Skipped (duration filter): {skipped_duration}")

    print(f"\nLoading ASR model: {args.asr_model}")
    import nemo.collections.asr as nemo_asr
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.asr_model)
    asr_model.eval()

    print(f"Transcribing {len(entries)} files in batches of {args.batch_size}...")
    audio_paths = [e["audio_filepath"] for e in entries]

    all_transcriptions = []
    for i in tqdm(range(0, len(audio_paths), args.batch_size), desc="Transcribing"):
        batch_paths = audio_paths[i : i + args.batch_size]
        batch_texts = transcribe_batch(asr_model, batch_paths)
        all_transcriptions.extend(batch_texts)

    skipped_empty = 0
    final_entries = []
    for entry, text in zip(entries, all_transcriptions):
        text = text.strip()
        if not text:
            skipped_empty += 1
            continue
        entry["text"] = text
        final_entries.append(entry)

    if skipped_empty:
        print(f"  Skipped (empty transcription): {skipped_empty}")

    del asr_model
    torch.cuda.empty_cache()

    all_manifest = args.output_dir / "manifest_all.json"
    with open(all_manifest, "w", encoding="utf-8") as f:
        for entry in final_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nFull manifest: {all_manifest} ({len(final_entries)} entries)")

    random.seed(args.seed)
    shuffled = final_entries.copy()
    random.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * args.val_ratio))
    val_entries = shuffled[:val_count]
    train_entries = shuffled[val_count:]

    train_manifest = args.output_dir / "manifest_train.json"
    val_manifest = args.output_dir / "manifest_val.json"

    with open(train_manifest, "w", encoding="utf-8") as f:
        for entry in train_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(val_manifest, "w", encoding="utf-8") as f:
        for entry in val_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Train manifest: {train_manifest} ({len(train_entries)} entries)")
    print(f"Val manifest:   {val_manifest} ({len(val_entries)} entries)")

    emotion_counts = {}
    for e in final_entries:
        emotion_counts[e["emotion"]] = emotion_counts.get(e["emotion"], 0) + 1

    print("\nEmotion distribution:")
    for emo, count in sorted(emotion_counts.items()):
        print(f"  {emo}: {count}")

    print("\nDone! Next steps:")
    print("  1. Run extract_sup_data.py for pitch + alignment priors")
    print("  2. Run extract_emotion_data.py for emotion .pt files")
    print("  3. Update your YAML config with manifest and sup_data paths")
    print("  4. Train!")


if __name__ == "__main__":
    main()