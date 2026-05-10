import lightning.pytorch as pl
import torch
import tempfile
from nemo.collections.tts.models import FastPitchModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.app_state import AppState


@hydra_runner(config_path="../conf", config_name="fastpitch_emotion")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Create fresh model with your config (includes emotion layers)
    model = FastPitchModel(cfg=cfg.model, trainer=trainer)

    # Extract state dict directly from .nemo file (it's a tar archive)
    import tarfile
    import io

    #nemo_path = "/home/alex/Documents/KNN/NeMo/tts_en_fastpitch.nemo"
    #nemo_path = "/mnt/matylda6/xokruc00/knn/first_nemo_tts/NeMo/tts_en_fastpitch.nemo"
    nemo_path = "/mnt/matylda6/xokruc00/knn/second_nemo_tts/NeMo/tts_en_fastpitch.nemo"
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(nemo_path, "r") as tar:
            tar.extractall(tmpdir)

        # Find the model weights file
        import glob
        ckpt_files = glob.glob(f"{tmpdir}/**/model_weights.ckpt", recursive=True)
        if not ckpt_files:
            ckpt_files = glob.glob(f"{tmpdir}/**/*.ckpt", recursive=True)

        print(f"Found checkpoint files: {ckpt_files}")
        pretrained_state = torch.load(ckpt_files[0], map_location="cpu")

    model_state = model.state_dict()

    # Only load weights that match in shape
    filtered = {
        k: v for k, v in pretrained_state.items()
        if k in model_state and v.shape == model_state[k].shape
    }

    print(f"Loading {len(filtered)}/{len(model_state)} layers from pretrained")
    skipped = set(model_state.keys()) - set(filtered.keys())
    print(f"Skipped (new or mismatched): {skipped}")

    model_state.update(filtered)
    model.load_state_dict(model_state)

    del pretrained_state, filtered

    trainer.fit(model)


if __name__ == "__main__":
    main()