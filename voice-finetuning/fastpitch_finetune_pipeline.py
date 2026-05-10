"""
=============================================================================
FastPitch + HiFi-GAN Fine-Tuning Pipeline for Custom Voice
=============================================================================

Prerequisites:
    pip install nemo_toolkit[tts] librosa soundfile tqdm

Audio requirements:
    - 22050 Hz sample rate (will be resampled if different)
    - Mono channel
    - Clean, no background noise

Usage:
    1. Edit the CONFIG section below
    2. Run steps sequentially:
       python fastpitch_finetune_pipeline.py --step prepare
       python fastpitch_finetune_pipeline.py --step extract_sup_data
       python fastpitch_finetune_pipeline.py --step train_fastpitch
       python fastpitch_finetune_pipeline.py --step generate_mels
       python fastpitch_finetune_pipeline.py --step train_hifigan
       python fastpitch_finetune_pipeline.py --step inference

    Or run everything:
       python fastpitch_finetune_pipeline.py --step all
=============================================================================
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


# =============================================================================
# CONFIG — EDIT THESE
# =============================================================================
class Config:
    # --- Paths ---
    MANIFEST_FILE = "/home/alex/Documents/KNN/KNN_Project_TTS/voice-finetuning/transcriptions.csv"  # your manifest file
    OUTPUT_DIR = "./new_surprised_finetuned_output"  # where all outputs go
    AUDIO_DIR = "/home/alex/Documents/KNN/datasets/Surprise"  # folder with wav files directly

    # --- Metadata format ---
    # Your metadata.csv should have lines like:
    #   utterance_001|Hello, this is a test.
    # Set the delimiter and column positions:
    METADATA_FILE = "transcriptions.csv"
    METADATA_DELIMITER = ","
    METADATA_FILENAME_COL = 0  # column index for filename (without .wav)
    METADATA_TEXT_COL = 1  # column index for transcription

    # --- Audio ---
    SAMPLE_RATE = 22050
    WAV_SUBDIR = "wavs"  # subdirectory containing wav files

    # --- Train/Val split ---
    VAL_SIZE = 50  # number of utterances for validation
    RANDOM_SEED = 42

    # --- FastPitch training ---
    FP_PRETRAINED = "tts_en_fastpitch"  # or "tts_en_fastpitch_ipa" if using IPA
    FP_MAX_STEPS = 5000  # ~5000 for 2hrs data, increase for better quality
    FP_BATCH_SIZE = 16  # reduce if OOM
    FP_LR = 2e-4
    FP_NUM_GPUS = 1

    # --- HiFi-GAN training ---
    HG_PRETRAINED = "tts_en_hifigan"
    HG_MAX_STEPS = 5000
    HG_BATCH_SIZE = 16
    HG_LR = 1e-5
    HG_NUM_GPUS = 1


cfg = Config()


# =============================================================================
# STEP 1: Prepare Data — Create NeMo manifests
# =============================================================================
def prepare_data():
    """
    Converts your metadata file + audio folder into NeMo JSON-lines manifest files.
    Audio and manifest can live in completely different locations.
    Also resamples audio to 22050 Hz mono if needed.
    """
    print("=" * 60)
    print("STEP 1: Preparing data and creating manifests")
    print("=" * 60)

    audio_dir = Path(cfg.AUDIO_DIR)
    manifest_file = Path(cfg.MANIFEST_FILE)
    output_dir = Path(cfg.OUTPUT_DIR)
    processed_wav_dir = output_dir / "wavs_processed"

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_wav_dir.mkdir(parents=True, exist_ok=True)

    # Read metadata
    entries = []

    print(f"Reading metadata from: {manifest_file}")
    print(f"Looking for audio in:  {audio_dir}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(cfg.METADATA_DELIMITER)
            try:
                filename = parts[cfg.METADATA_FILENAME_COL].strip()
                text = parts[cfg.METADATA_TEXT_COL].strip()
            except IndexError:
                print(f"  WARNING: Skipping line {line_num}: {line}")
                continue

            # Find the wav file in AUDIO_DIR
            wav_path = audio_dir / f"{filename}.wav"
            if not wav_path.exists():
                # Try without adding .wav if filename already has extension
                wav_path = audio_dir / filename
                if not wav_path.exists():
                    print(f"  WARNING: Audio not found for '{filename}', skipping")
                    continue

            entries.append({"filename": filename, "text": text, "wav_path": str(wav_path)})

    print(f"Found {len(entries)} valid entries")

    # Process audio files and build manifest entries
    manifest_entries = []
    print("Processing audio files...")
    for entry in tqdm(entries):
        src_path = entry["wav_path"]
        dst_path = processed_wav_dir / f"{entry['filename']}.wav"

        # Load and resample if needed
        audio, sr = librosa.load(src_path, sr=cfg.SAMPLE_RATE, mono=True)

        # Remove silence from beginning and end
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)

        # Save processed audio
        sf.write(str(dst_path), audio_trimmed, cfg.SAMPLE_RATE)

        duration = len(audio_trimmed) / cfg.SAMPLE_RATE

        # Skip very short or very long utterances
        if duration < 0.5:
            print(f"  Skipping {entry['filename']}: too short ({duration:.1f}s)")
            continue
        if duration > 20.0:
            print(f"  Skipping {entry['filename']}: too long ({duration:.1f}s)")
            continue

        manifest_entries.append({
            "audio_filepath": str(dst_path.resolve()),
            "text": entry["text"],
            "duration": round(duration, 3),
            "speaker": 0  # single speaker
        })

    print(f"\nKept {len(manifest_entries)} utterances after filtering")

    # Compute total duration
    total_dur = sum(e["duration"] for e in manifest_entries)
    print(f"Total audio duration: {total_dur / 3600:.2f} hours")

    # Split into train/val
    random.seed(cfg.RANDOM_SEED)
    random.shuffle(manifest_entries)

    val_entries = manifest_entries[:cfg.VAL_SIZE]
    train_entries = manifest_entries[cfg.VAL_SIZE:]

    print(f"Train: {len(train_entries)} utterances")
    print(f"Val:   {len(val_entries)} utterances")

    # Write manifests
    train_manifest = output_dir / "manifest_train.json"
    val_manifest = output_dir / "manifest_val.json"
    full_manifest = output_dir / "manifest_full.json"

    for path, data in [
        (train_manifest, train_entries),
        (val_manifest, val_entries),
        (full_manifest, manifest_entries),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Written: {path}")

    print("\n✅ Data preparation complete!")
    return train_manifest, val_manifest


# =============================================================================
# STEP 2: Extract supplementary data (pitch, alignment priors)
# =============================================================================
def extract_supplementary_data():
    """
    Extracts pitch contours and alignment priors needed for FastPitch training.
    Also computes pitch statistics (mean, std) for your speaker.
    """
    print("=" * 60)
    print("STEP 2: Extracting supplementary data (pitch, priors)")
    print("=" * 60)

    output_dir = Path(cfg.OUTPUT_DIR)
    sup_data_path = output_dir / "sup_data"
    sup_data_path.mkdir(parents=True, exist_ok=True)

    train_manifest = output_dir / "manifest_train.json"
    val_manifest = output_dir / "manifest_val.json"

    # --------------------------------------------------------------------------
    # Option A: Use NeMo's built-in extract_sup_data.py script (RECOMMENDED)
    # --------------------------------------------------------------------------
    print("\nAttempting to use NeMo's extract_sup_data.py script...")

    # First, let's compute pitch stats manually as a fallback
    print("Computing pitch statistics from your data...")
    all_pitches = []

    with open(train_manifest, "r") as f:
        for line in tqdm(list(f), desc="Analyzing pitch"):
            entry = json.loads(line)
            audio, sr = librosa.load(entry["audio_filepath"], sr=cfg.SAMPLE_RATE)

            # Extract pitch using pyin (same as NeMo uses)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                sr=sr,
                frame_length=1024,
                hop_length=256
            )

            # Keep only voiced frames
            voiced_f0 = f0[voiced_flag]
            if len(voiced_f0) > 0:
                all_pitches.extend(voiced_f0.tolist())

    all_pitches = np.array(all_pitches)
    pitch_mean = float(np.mean(all_pitches))
    pitch_std = float(np.std(all_pitches))
    pitch_fmin = float(np.percentile(all_pitches, 1))
    pitch_fmax = float(np.percentile(all_pitches, 99))

    pitch_stats = {
        "pitch_mean": round(pitch_mean, 2),
        "pitch_std": round(pitch_std, 2),
        "pitch_fmin": round(pitch_fmin, 2),
        "pitch_fmax": round(pitch_fmax, 2),
    }

    stats_path = output_dir / "pitch_stats.json"
    with open(stats_path, "w") as f:
        json.dump(pitch_stats, f, indent=2)

    print(f"\n📊 Pitch statistics for your speaker:")
    print(f"   pitch_mean = {pitch_stats['pitch_mean']}")
    print(f"   pitch_std  = {pitch_stats['pitch_std']}")
    print(f"   pitch_fmin = {pitch_stats['pitch_fmin']}")
    print(f"   pitch_fmax = {pitch_stats['pitch_fmax']}")
    print(f"   Saved to: {stats_path}")

    # --------------------------------------------------------------------------
    # Now run NeMo's supplementary data extraction
    # --------------------------------------------------------------------------
    print("\nRunning NeMo supplementary data extraction...")
    print("This pre-computes alignment priors and pitch for training.\n")

    # We build the command for NeMo's script
    cmd = f"""python -m nemo.collections.tts.data.extract_sup_data \\
    manifest_filepath={train_manifest} \\
    sup_data_path={sup_data_path} \\
    sup_data_types="[align_prior_matrix,pitch]" \\
    pitch_fmin={pitch_stats['pitch_fmin']:.0f} \\
    pitch_fmax={pitch_stats['pitch_fmax']:.0f}"""

    print(f"Command:\n{cmd}\n")

    ret = os.system(cmd)

    if ret != 0:
        # Fallback: try the script path approach
        print("\nDirect module call failed. Trying script path...")

        # Try to find NeMo's script
        try:
            import nemo
            nemo_path = Path(nemo.__file__).parent.parent
            script = nemo_path / "scripts" / "dataset_processing" / "tts" / "extract_sup_data.py"
            if script.exists():
                cmd = f"""python {script} \\
    manifest_filepath={train_manifest} \\
    sup_data_path={sup_data_path} \\
    sup_data_types="[align_prior_matrix,pitch]" \\
    pitch_fmin={pitch_stats['pitch_fmin']:.0f} \\
    pitch_fmax={pitch_stats['pitch_fmax']:.0f}"""
                print(f"Command:\n{cmd}\n")
                os.system(cmd)
            else:
                print(f"Script not found at {script}")
                print("Supplementary data will be computed on-the-fly during training.")
                print("This is slower but works fine.")
        except Exception as e:
            print(f"Could not locate NeMo scripts: {e}")
            print("Supplementary data will be computed on-the-fly during training.")

    # Also extract for validation
    cmd_val = f"""python -m nemo.collections.tts.data.extract_sup_data \\
    manifest_filepath={val_manifest} \\
    sup_data_path={sup_data_path} \\
    sup_data_types="[align_prior_matrix,pitch]" \\
    pitch_fmin={pitch_stats['pitch_fmin']:.0f} \\
    pitch_fmax={pitch_stats['pitch_fmax']:.0f}"""
    os.system(cmd_val)

    print("\n✅ Supplementary data extraction complete!")
    return pitch_stats


# =============================================================================
# STEP 3: Fine-tune FastPitch
# =============================================================================
def train_fastpitch():
    """
    Fine-tunes the pretrained FastPitch model on your speaker's data.
    """
    print("=" * 60)
    print("STEP 3: Fine-tuning FastPitch")
    print("=" * 60)

    output_dir = Path(cfg.OUTPUT_DIR)

    # Load pitch stats
    stats_path = output_dir / "pitch_stats.json"
    with open(stats_path, "r") as f:
        pitch_stats = json.load(f)

    train_manifest = output_dir / "manifest_train.json"
    val_manifest = output_dir / "manifest_val.json"
    sup_data_path = output_dir / "sup_data"
    exp_dir = output_dir / "fastpitch_exp"

    sup_data_path.mkdir(parents=True, exist_ok=True)

    # Strategy flag
    strategy = "auto" if cfg.FP_NUM_GPUS <= 1 else "ddp"

    # Determine which config to use based on pretrained model
    if "ipa" in cfg.FP_PRETRAINED:
        config_name = "fastpitch_align_44100_ipa.yaml"
        tokenizer_cfg = "+model.text_tokenizer.add_blank_at=true"
    else:
        config_name = "fastpitch_align_v1.05.yaml"
        tokenizer_cfg = "+model.text_tokenizer.add_blank_at=true"

    cmd = f"""python -m nemo.collections.tts.models.fastpitch \\
    --config-name={config_name} \\
    train_dataset={train_manifest} \\
    validation_datasets={val_manifest} \\
    sup_data_path={sup_data_path} \\
    exp_manager.exp_dir={exp_dir} \\
    +init_from_pretrained_model={cfg.FP_PRETRAINED} \\
    +trainer.max_steps={cfg.FP_MAX_STEPS} \\
    ~trainer.max_epochs \\
    trainer.check_val_every_n_epoch=25 \\
    trainer.devices={cfg.FP_NUM_GPUS} \\
    trainer.strategy={strategy} \\
    model.train_ds.dataloader_params.batch_size={cfg.FP_BATCH_SIZE} \\
    model.validation_ds.dataloader_params.batch_size={cfg.FP_BATCH_SIZE} \\
    model.n_speakers=1 \\
    model.pitch_mean={pitch_stats['pitch_mean']} \\
    model.pitch_std={pitch_stats['pitch_std']} \\
    model.pitch_fmin={pitch_stats['pitch_fmin']:.0f} \\
    model.pitch_fmax={pitch_stats['pitch_fmax']:.0f} \\
    model.optim.lr={cfg.FP_LR} \\
    ~model.optim.sched \\
    model.optim.name=adam \\
    {tokenizer_cfg}"""

    print(f"\n🚀 Training command:\n{cmd}\n")
    print("This will take a while depending on your GPU...")
    print(f"Checkpoints saved to: {exp_dir}\n")

    # --------------------------------------------------------------------------
    # ALTERNATIVE: Programmatic training (if the CLI approach fails)
    # --------------------------------------------------------------------------
    print("=" * 40)
    print("If the CLI command above doesn't work for your NeMo version,")
    print("use this programmatic approach instead:")
    print("=" * 40)

    programmatic_code = f'''
import pytorch_lightning as pl
from nemo.collections.tts.models import FastPitchModel
from omegaconf import OmegaConf, open_dict

# Load pretrained model
model = FastPitchModel.from_pretrained("{cfg.FP_PRETRAINED}")

# Update config for fine-tuning
with open_dict(model.cfg):
    # Data
    model.cfg.train_ds.manifest_filepath = "{train_manifest}"
    model.cfg.validation_ds.manifest_filepath = "{val_manifest}"
    model.cfg.sup_data_path = "{sup_data_path}"
    model.cfg.sup_data_types = ["align_prior_matrix", "pitch"]

    # Speaker
    model.cfg.n_speakers = 1

    # Pitch stats from YOUR data
    model.cfg.pitch_mean = {pitch_stats['pitch_mean']}
    model.cfg.pitch_std = {pitch_stats['pitch_std']}
    model.cfg.pitch_fmin = {pitch_stats['pitch_fmin']:.0f}
    model.cfg.pitch_fmax = {pitch_stats['pitch_fmax']:.0f}

    # Optimizer — lower LR, no scheduler, adam
    model.cfg.optim.name = "adam"
    model.cfg.optim.lr = {cfg.FP_LR}
    if hasattr(model.cfg.optim, "sched"):
        model.cfg.optim.pop("sched")

    # Batch size
    model.cfg.train_ds.dataloader_params.batch_size = {cfg.FP_BATCH_SIZE}
    model.cfg.validation_ds.dataloader_params.batch_size = {cfg.FP_BATCH_SIZE}

# Setup data loaders
model.setup_training_data(model.cfg.train_ds)
model.setup_validation_data(model.cfg.validation_ds)

# Trainer
trainer = pl.Trainer(
    max_steps={cfg.FP_MAX_STEPS},
    accelerator="gpu",
    devices={cfg.FP_NUM_GPUS},
    strategy="auto",
    check_val_every_n_epoch=25,
    default_root_dir="{exp_dir}",
    enable_checkpointing=True,
)

# Train!
trainer.fit(model)

# Save the final model
model.save_to("{output_dir}/fastpitch_finetuned.nemo")
print("✅ FastPitch fine-tuned model saved!")
'''

    programmatic_path = output_dir / "train_fastpitch_programmatic.py"
    with open(programmatic_path, "w") as f:
        f.write(programmatic_code)

    print(f"\nProgrammatic training script saved to: {programmatic_path}")
    print(f"Run it with: python {programmatic_path}\n")

    # Run the CLI command
    ret = os.system(cmd)
    if ret != 0:
        print("\n⚠️  CLI training failed. Try the programmatic approach:")
        print(f"   python {programmatic_path}")

    print("\n✅ FastPitch training step complete!")


# =============================================================================
# STEP 4: Generate mel spectrograms for HiFi-GAN fine-tuning
# =============================================================================
def generate_mels():
    """
    Uses the fine-tuned FastPitch to generate mel spectrograms (GTA mels)
    from training audio. These are used to fine-tune HiFi-GAN.
    """
    print("=" * 60)
    print("STEP 4: Generating mel spectrograms for HiFi-GAN")
    print("=" * 60)

    output_dir = Path(cfg.OUTPUT_DIR)
    mel_dir = output_dir / "mels"
    mel_dir.mkdir(parents=True, exist_ok=True)

    # Find the best FastPitch checkpoint
    fp_model_path = output_dir / "fastpitch_finetuned.nemo"
    if not fp_model_path.exists():
        # Look for checkpoint in exp dir
        exp_dir = output_dir / "fastpitch_exp"
        nemo_files = list(exp_dir.rglob("*.nemo"))
        ckpt_files = list(exp_dir.rglob("*.ckpt"))

        if nemo_files:
            fp_model_path = sorted(nemo_files)[-1]
            print(f"Found .nemo checkpoint: {fp_model_path}")
        elif ckpt_files:
            fp_model_path = sorted(ckpt_files)[-1]
            print(f"Found .ckpt checkpoint: {fp_model_path}")
        else:
            print("❌ No FastPitch checkpoint found!")
            print(f"   Expected at: {output_dir}/fastpitch_finetuned.nemo")
            print(f"   Or in: {exp_dir}/")
            return

    generate_mels_code = f'''
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from nemo.collections.tts.models import FastPitchModel

# Load fine-tuned FastPitch
print("Loading fine-tuned FastPitch...")
model = FastPitchModel.restore_from("{fp_model_path}")
model.eval()
model.cuda()

manifest_path = "{output_dir}/manifest_full.json"
output_manifest = "{mel_dir}/manifest_mels.json"
mel_output_dir = Path("{mel_dir}")

entries = []
with open(manifest_path, "r") as f:
    for line in f:
        entries.append(json.loads(line))

print(f"Generating mels for {{len(entries)}} utterances...")
new_entries = []

with torch.no_grad():
    for entry in tqdm(entries):
        try:
            # Parse text to tokens
            tokens = model.parse(entry["text"])

            # Generate mel spectrogram
            mel = model.generate_spectrogram(tokens=tokens)

            # Save mel
            mel_filename = Path(entry["audio_filepath"]).stem + "_mel.npy"
            mel_path = mel_output_dir / mel_filename
            np.save(str(mel_path), mel.cpu().numpy())

            new_entry = {{
                "audio_filepath": entry["audio_filepath"],
                "mel_filepath": str(mel_path),
                "duration": entry["duration"],
                "text": entry["text"],
            }}
            new_entries.append(new_entry)
        except Exception as e:
            print(f"Error processing {{entry['audio_filepath']}}: {{e}}")

# Write HiFi-GAN manifest
with open(output_manifest, "w") as f:
    for entry in new_entries:
        f.write(json.dumps(entry) + "\\n")

# Split into train/val for HiFi-GAN
val_size = min(50, len(new_entries) // 10)
hg_val = new_entries[:val_size]
hg_train = new_entries[val_size:]

with open("{mel_dir}/manifest_hifigan_train.json", "w") as f:
    for e in hg_train:
        f.write(json.dumps(e) + "\\n")

with open("{mel_dir}/manifest_hifigan_val.json", "w") as f:
    for e in hg_val:
        f.write(json.dumps(e) + "\\n")

print(f"✅ Generated mels: {{len(new_entries)}} total")
print(f"   HiFi-GAN train: {{len(hg_train)}}")
print(f"   HiFi-GAN val:   {{len(hg_val)}}")
'''

    script_path = output_dir / "generate_mels.py"
    with open(script_path, "w") as f:
        f.write(generate_mels_code)

    print(f"Mel generation script saved to: {script_path}")
    print(f"Run: python {script_path}")

    os.system(f"python {script_path}")
    print("\n✅ Mel generation complete!")


# =============================================================================
# STEP 5: Fine-tune HiFi-GAN
# =============================================================================
def train_hifigan():
    """
    Fine-tunes HiFi-GAN vocoder on your speaker's data for better quality.
    """
    print("=" * 60)
    print("STEP 5: Fine-tuning HiFi-GAN")
    print("=" * 60)

    output_dir = Path(cfg.OUTPUT_DIR)
    mel_dir = output_dir / "mels"
    exp_dir = output_dir / "hifigan_exp"

    hg_train = mel_dir / "manifest_hifigan_train.json"
    hg_val = mel_dir / "manifest_hifigan_val.json"

    if not hg_train.exists():
        print(f"❌ HiFi-GAN train manifest not found: {hg_train}")
        print("   Run step 'generate_mels' first.")
        return

    train_hifigan_code = f'''
import pytorch_lightning as pl
from nemo.collections.tts.models import HifiGanModel
from omegaconf import open_dict

# Load pretrained HiFi-GAN
print("Loading pretrained HiFi-GAN...")
model = HifiGanModel.from_pretrained("{cfg.HG_PRETRAINED}")

# Update config for fine-tuning
with open_dict(model.cfg):
    model.cfg.train_ds.manifest_filepath = "{hg_train}"
    model.cfg.validation_ds.manifest_filepath = "{hg_val}"

    # Lower learning rate for fine-tuning
    model.cfg.optim.lr = {cfg.HG_LR}
    if hasattr(model.cfg.optim, "sched"):
        model.cfg.optim.pop("sched")

    model.cfg.train_ds.dataloader_params.batch_size = {cfg.HG_BATCH_SIZE}
    model.cfg.validation_ds.dataloader_params.batch_size = {cfg.HG_BATCH_SIZE}

# Setup data
model.setup_training_data(model.cfg.train_ds)
model.setup_validation_data(model.cfg.validation_ds)

# Trainer
trainer = pl.Trainer(
    max_steps={cfg.HG_MAX_STEPS},
    accelerator="gpu",
    devices={cfg.HG_NUM_GPUS},
    strategy="auto",
    check_val_every_n_epoch=10,
    default_root_dir="{exp_dir}",
    enable_checkpointing=True,
)

trainer.fit(model)
model.save_to("{output_dir}/hifigan_finetuned.nemo")
print("✅ HiFi-GAN fine-tuned model saved!")
'''

    script_path = output_dir / "train_hifigan.py"
    with open(script_path, "w") as f:
        f.write(train_hifigan_code)

    print(f"HiFi-GAN training script saved to: {script_path}")
    print(f"Run: python {script_path}")

    os.system(f"python {script_path}")
    print("\n✅ HiFi-GAN training complete!")


# =============================================================================
# STEP 6: Inference — Use your custom voice!
# =============================================================================
def inference():
    """
    Generates speech with your fine-tuned custom voice.
    """
    print("=" * 60)
    print("STEP 6: Inference with your custom voice")
    print("=" * 60)

    output_dir = Path(cfg.OUTPUT_DIR)

    inference_code = f'''
import torch
import soundfile as sf
from nemo.collections.tts.models import FastPitchModel, HifiGanModel

# Load your fine-tuned models
print("Loading fine-tuned FastPitch...")
spec_generator = FastPitchModel.restore_from(
    "{output_dir}/fastpitch_finetuned.nemo"
).eval().cuda()

# Use fine-tuned HiFi-GAN if available, otherwise pretrained
hifigan_path = "{output_dir}/hifigan_finetuned.nemo"
import os
if os.path.exists(hifigan_path):
    print("Loading fine-tuned HiFi-GAN...")
    vocoder = HifiGanModel.restore_from(hifigan_path).eval().cuda()
else:
    print("Loading pretrained HiFi-GAN (fine-tuned not found)...")
    vocoder = HifiGanModel.from_pretrained("{cfg.HG_PRETRAINED}").eval().cuda()

def synthesize(text, output_path="output.wav", pace=1.0):
    """Generate speech from text and save to wav file."""
    with torch.no_grad():
        tokens = spec_generator.parse(text)
        spectrogram = spec_generator.generate_spectrogram(tokens=tokens, pace=pace)
        audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

    audio_np = audio.squeeze().cpu().numpy()
    sf.write(output_path, audio_np, 22050)
    print(f"Saved: {{output_path}} ({{len(audio_np)/22050:.1f}}s)")
    return audio_np

# --- Test sentences ---
test_sentences = [
    "Hello! This is my custom voice speaking.",
    "The quick brown fox jumps over the lazy dog.",
    "I can now speak with emotion and personality.",
    "This text to speech system was fine tuned on just two hours of data.",
]

print("\\nGenerating test sentences...\\n")
for i, text in enumerate(test_sentences):
    synthesize(text, f"{output_dir}/test_{{i+1:02d}}.wav")

print("\\n✅ All test sentences generated!")
print(f"Check output files in: {output_dir}/")
'''

    script_path = output_dir / "inference.py"
    with open(script_path, "w") as f:
        f.write(inference_code)

    print(f"Inference script saved to: {script_path}")
    print(f"Run: python {script_path}")

    os.system(f"python {script_path}")


# =============================================================================
# Notebook-friendly version (for Jupyter / Colab)
# =============================================================================
def generate_notebook_code():
    """Generates a single-cell notebook version of the full pipeline."""

    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    notebook_code = '''# ===========================================================================
# FastPitch + HiFi-GAN Fine-Tuning — Jupyter Notebook Version
# ===========================================================================
# Run each cell in order. Adjust paths in Cell 1.

# %% Cell 1: Configuration
AUDIO_DIR = "/path/to/your/audio"          # folder with wav files
MANIFEST_FILE = "/path/to/your/manifest.csv"  # metadata file (separate location)
OUTPUT_DIR = "./finetune_output"        
SAMPLE_RATE = 22050
VAL_SIZE = 50
FP_MAX_STEPS = 5000
HG_MAX_STEPS = 5000
BATCH_SIZE = 16

# %% Cell 2: Install dependencies
# !pip install nemo_toolkit[tts] librosa soundfile

# %% Cell 3: Prepare manifests
import json, os, random
from pathlib import Path
import librosa, soundfile as sf, numpy as np
from tqdm.notebook import tqdm

audio_dir = Path(AUDIO_DIR)
manifest_file = Path(MANIFEST_FILE)
output_dir = Path(OUTPUT_DIR)
proc_dir = output_dir / "wavs_processed"
output_dir.mkdir(parents=True, exist_ok=True)
proc_dir.mkdir(parents=True, exist_ok=True)

entries = []
with open(manifest_file) as f:
    for line in f:
        if not line.strip(): continue
        parts = line.strip().split("|")
        entries.append({"filename": parts[0], "text": parts[1]})

manifest_entries = []
for e in tqdm(entries):
    src = audio_dir / f"{e['filename']}.wav"
    if not src.exists(): continue
    dst = proc_dir / f"{e['filename']}.wav"
    audio, sr = librosa.load(str(src), sr=SAMPLE_RATE, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=25)
    sf.write(str(dst), audio, SAMPLE_RATE)
    dur = len(audio) / SAMPLE_RATE
    if 0.5 < dur < 20:
        manifest_entries.append({
            "audio_filepath": str(dst.resolve()),
            "text": e["text"], "duration": round(dur, 3), "speaker": 0
        })

random.seed(42)
random.shuffle(manifest_entries)
val_entries = manifest_entries[:VAL_SIZE]
train_entries = manifest_entries[VAL_SIZE:]

for name, data in [("train", train_entries), ("val", val_entries), ("full", manifest_entries)]:
    with open(output_dir / f"manifest_{name}.json", "w") as f:
        for e in data: f.write(json.dumps(e) + "\\n")

print(f"Train: {len(train_entries)}, Val: {len(val_entries)}")

# %% Cell 4: Compute pitch statistics
all_pitches = []
with open(output_dir / "manifest_train.json") as f:
    for line in tqdm(list(f)):
        entry = json.loads(line)
        audio, sr = librosa.load(entry["audio_filepath"], sr=SAMPLE_RATE)
        f0, voiced, _ = librosa.pyin(audio, fmin=65, fmax=2093, sr=sr,
                                      frame_length=1024, hop_length=256)
        all_pitches.extend(f0[voiced].tolist())

pitch_mean = np.mean(all_pitches)
pitch_std = np.std(all_pitches)
pitch_fmin = np.percentile(all_pitches, 1)
pitch_fmax = np.percentile(all_pitches, 99)
print(f"mean={pitch_mean:.1f}, std={pitch_std:.1f}, fmin={pitch_fmin:.0f}, fmax={pitch_fmax:.0f}")

# %% Cell 5: Fine-tune FastPitch
import pytorch_lightning as pl
from nemo.collections.tts.models import FastPitchModel
from omegaconf import open_dict

model = FastPitchModel.from_pretrained("tts_en_fastpitch")

with open_dict(model.cfg):
    model.cfg.train_ds.manifest_filepath = str(output_dir / "manifest_train.json")
    model.cfg.validation_ds.manifest_filepath = str(output_dir / "manifest_val.json")
    model.cfg.sup_data_path = str(output_dir / "sup_data")
    model.cfg.sup_data_types = ["align_prior_matrix", "pitch"]
    model.cfg.n_speakers = 1
    model.cfg.pitch_mean = float(pitch_mean)
    model.cfg.pitch_std = float(pitch_std)
    model.cfg.pitch_fmin = int(pitch_fmin)
    model.cfg.pitch_fmax = int(pitch_fmax)
    model.cfg.optim.name = "adam"
    model.cfg.optim.lr = 2e-4
    if hasattr(model.cfg.optim, "sched"): model.cfg.optim.pop("sched")
    model.cfg.train_ds.dataloader_params.batch_size = BATCH_SIZE
    model.cfg.validation_ds.dataloader_params.batch_size = BATCH_SIZE

model.setup_training_data(model.cfg.train_ds)
model.setup_validation_data(model.cfg.validation_ds)

trainer = pl.Trainer(max_steps=FP_MAX_STEPS, accelerator="gpu", devices=1,
                     strategy="auto", check_val_every_n_epoch=25)
trainer.fit(model)
model.save_to(str(output_dir / "fastpitch_finetuned.nemo"))

# %% Cell 6: Fine-tune HiFi-GAN (optional but recommended)
from nemo.collections.tts.models import HifiGanModel

vocoder = HifiGanModel.from_pretrained("tts_en_hifigan")
with open_dict(vocoder.cfg):
    vocoder.cfg.train_ds.manifest_filepath = str(output_dir / "manifest_train.json")
    vocoder.cfg.validation_ds.manifest_filepath = str(output_dir / "manifest_val.json")
    vocoder.cfg.optim.lr = 1e-5
    if hasattr(vocoder.cfg.optim, "sched"): vocoder.cfg.optim.pop("sched")
    vocoder.cfg.train_ds.dataloader_params.batch_size = BATCH_SIZE
    vocoder.cfg.validation_ds.dataloader_params.batch_size = BATCH_SIZE

vocoder.setup_training_data(vocoder.cfg.train_ds)
vocoder.setup_validation_data(vocoder.cfg.validation_ds)

trainer2 = pl.Trainer(max_steps=HG_MAX_STEPS, accelerator="gpu", devices=1, strategy="auto")
trainer2.fit(vocoder)
vocoder.save_to(str(output_dir / "hifigan_finetuned.nemo"))

# %% Cell 7: Inference!
import IPython.display as ipd
import torch

spec_gen = FastPitchModel.restore_from(str(output_dir / "fastpitch_finetuned.nemo")).eval().cuda()
voc = HifiGanModel.restore_from(str(output_dir / "hifigan_finetuned.nemo")).eval().cuda()

def speak(text, pace=1.0):
    with torch.no_grad():
        tokens = spec_gen.parse(text)
        mel = spec_gen.generate_spectrogram(tokens=tokens, pace=pace)
        audio = voc.convert_spectrogram_to_audio(spec=mel)
    return ipd.Audio(audio.squeeze().cpu().numpy(), rate=22050)

speak("Hello! This is my custom voice with emotion.")
'''

    notebook_path = output_dir / "finetune_notebook.py"
    with open(notebook_path, "w") as f:
        f.write(notebook_code)

    print(f"Notebook-friendly code saved to: {notebook_path}")


# =============================================================================
# Main entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="FastPitch + HiFi-GAN Fine-Tuning Pipeline")
    parser.add_argument(
        "--step",
        choices=["prepare", "extract_sup_data", "train_fastpitch",
                 "generate_mels", "train_hifigan", "inference",
                 "notebook", "all"],
        required=True,
        help="Which step to run"
    )
    args = parser.parse_args()

    if args.step == "prepare" or args.step == "all":
        prepare_data()

    if args.step == "extract_sup_data" or args.step == "all":
        extract_supplementary_data()

    if args.step == "train_fastpitch" or args.step == "all":
        train_fastpitch()

    if args.step == "generate_mels" or args.step == "all":
        generate_mels()

    if args.step == "train_hifigan" or args.step == "all":
        train_hifigan()

    if args.step == "inference" or args.step == "all":
        inference()

    if args.step == "notebook":
        generate_notebook_code()


if __name__ == "__main__":
    main()