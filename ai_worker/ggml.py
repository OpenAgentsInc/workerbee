from huggingface_hub import hf_hub_download

def download_ggml(name):
    base = f"ggml-{name}.bin"
    repo = "ggerganov/whisper.cpp"
    path = hf_hub_download(repo_id=repo, filename=base, resume_download=True)
    return path