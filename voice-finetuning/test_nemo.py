from nemo.collections.tts.models import FastPitchModel, HifiGanModel
import soundfile as sf

# Your fine-tuned FastPitch
model = FastPitchModel.restore_from("sam_angry_finetune_output/fastpitch_finetuned.nemo").eval().cuda()

# Pretrained HiFi-GAN (for inference it works fine without fine-tuning)
vocoder = HifiGanModel.from_pretrained("tts_en_hifigan").eval().cuda()

tokens = model.parse("Hello, this is a test of my custom voice.")
spec = model.generate_spectrogram(tokens=tokens)
audio = vocoder.convert_spectrogram_to_audio(spec=spec)
sf.write("test_output.wav", audio.squeeze().cpu().detach().numpy(), 22050)
print("Saved test_output.wav")