from nemo.collections.tts.models import HifiGanModel
model = HifiGanModel.load_from_checkpoint(
    '/home/alex/Documents/KNN/2026-05-10_08-51-25/checkpoints/HifiGan_finetune--val_loss=0.3690-epoch=129.ckpt',
    weights_only=False
)
model.save_to('/home/alex/Documents/KNN/2026-05-10_08-51-25/checkpoints/hifigan_finetuned.nemo')
print('Saved!')