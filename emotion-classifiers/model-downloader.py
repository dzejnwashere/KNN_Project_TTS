from huggingface_hub import snapshot_download

# Download entire model repo to a specific path
snapshot_download(
    repo_id="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
    local_dir="/home/alex/Documents/KNN/speech-emotion-recognition-with-openai-whisper-large-v3/"
)