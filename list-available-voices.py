from huggingface_hub import list_repo_files

repo_id = "hexgrad/Kokoro-82M"
files = list_repo_files(repo_id)
voices = [f for f in files if f.startswith("voices/") and f.endswith(".pt")]
print("Available voices:", voices) 