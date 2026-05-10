#!/usr/bin/env python3
"""
Batch audio transcription using NVIDIA Parakeet (NeMo ASR).

Scans a directory for audio files, transcribes each one using the
Parakeet-TDT model, and writes a CSV with columns: audio_name, text.

Usage:
    python transcribe_audio.py --audio_dir /path/to/audio
    python transcribe_audio.py --audio_dir /path/to/audio --output results.csv
    python transcribe_audio.py --audio_dir /path/to/audio --model nvidia/parakeet-tdt-0.6b-v2

Requirements:
    pip install nemo_toolkit['asr']
    (A CUDA-capable GPU is strongly recommended for reasonable speed.)
"""

import argparse
import csv
import os
import sys
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".wma", ".webm"}


def find_audio_files(directory: str) -> list[Path]:
    """Return sorted list of audio files in *directory* (non-recursive)."""
    audio_dir = Path(directory)
    if not audio_dir.is_dir():
        print(f"Error: '{directory}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    files = sorted(
        p for p in audio_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Batch-transcribe audio files with NVIDIA Parakeet."
    )
    parser.add_argument(
        "--audio_dir",
        required=True,
        help="Directory containing audio files to transcribe.",
    )
    parser.add_argument(
        "--output",
        default="transcriptions.csv",
        help="Output CSV path (default: transcriptions.csv).",
    )
    parser.add_argument(
        "--model",
        default="nvidia/parakeet-tdt-0.6b-v2",
        help="Pretrained model name (default: nvidia/parakeet-tdt-0.6b-v2).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for transcription (default: 16).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for audio files recursively in subdirectories.",
    )
    args = parser.parse_args()

    # ── Discover audio files ────────────────────────────────────────────
    if args.recursive:
        audio_dir = Path(args.audio_dir)
        audio_files = sorted(
            p for p in audio_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        )
    else:
        audio_files = find_audio_files(args.audio_dir)

    if not audio_files:
        print(f"No audio files found in '{args.audio_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(audio_files)} audio file(s) in '{args.audio_dir}'.")

    # ── Load model ──────────────────────────────────────────────────────
    print(f"Loading model '{args.model}' …")
    import nemo.collections.asr as nemo_asr

    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)
    print("Model loaded.\n")

    # ── Transcribe ──────────────────────────────────────────────────────
    file_paths = [str(f) for f in audio_files]
    file_names = [f.name for f in audio_files]

    print(f"Transcribing {len(file_paths)} file(s) (batch_size={args.batch_size}) …")
    outputs = asr_model.transcribe(file_paths, batch_size=args.batch_size)

    # ── Write CSV ───────────────────────────────────────────────────────
    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["audio_name", "text"])
        for name, result in zip(file_names, outputs):
            text = result.text if hasattr(result, "text") else str(result)
            writer.writerow([name, text])

    print(f"\nDone! Transcriptions saved to '{args.output}'.")
    print(f"  • Files processed : {len(file_paths)}")
    print(f"  • Output CSV      : {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()