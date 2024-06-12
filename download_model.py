import os
import shutil
import fnmatch
# from tensorize import serialize_model # TODO: Add back when tensorizer is implemented
from huggingface_hub import snapshot_download
from huggingface_hub import HfFileSystem, snapshot_download
from tqdm.auto import tqdm


class DisabledTqdm(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)
        

def download_weights_from_hf(model_name_or_path,
                             cache_dir,
                             allow_patterns,
                             revision):
    """Download model weights from Hugging Face Hub.
    
    Args:
        model_name_or_path (str): The model name or path.
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        allow_patterns (List[str]): The allowed patterns for the
            weight files. Files matched by any of the patterns will be
            downloaded.
        revision (Optional[str]): The revision of the model.

    Returns:
        str: The path to the downloaded model weights.
    """
    # Before we download we look at that is available:
    fs = HfFileSystem()
    file_list = fs.ls(model_name_or_path, detail=False, revision=revision)

    # depending on what is available we download different things
    for pattern in allow_patterns:
        matching = fnmatch.filter(file_list, pattern)
        if len(matching) > 0:
            allow_patterns = [pattern]
            break

    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with get_lock(model_name_or_path, cache_dir):
        hf_folder = snapshot_download(model_name_or_path,
                                      allow_patterns=allow_patterns,
                                      cache_dir=cache_dir,
                                      tqdm_class=DisabledTqdm,
                                      revision=revision)
    return hf_folder



def download_lora_weight(model_name):
    folder = snapshot_download(
        model_name,
        local_dir='/lora',
        tqdm_class=DisabledTqdm,
    )
    return folder

def download_extras_or_tokenizer(model_name, cache_dir, revision, extras=False):
    """Download model or tokenizer and prepare its weights, returning the local folder path."""
    pattern = ["*token*", "*.json"] if extras else None
    extra_dir = "/extras" if extras else ""
    folder = snapshot_download(
        model_name,
        cache_dir=cache_dir + extra_dir,
        revision=revision,
        tqdm_class=DisabledTqdm,
        allow_patterns=pattern if extras else None,
        ignore_patterns=["*.safetensors", "*.bin", "*.pt"] if not extras else None
    )
    return folder


def move_files(src_dir, dest_dir):
    """Move files from source to destination directory."""
    for f in os.listdir(src_dir):
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(dest_dir, f)
        if os.path.abspath(src_path) != os.path.abspath(dst_path) and os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            os.remove(src_path)


if __name__ == "__main__":
    model, download_dir = os.getenv("MODEL_NAME"), os.getenv("HF_HOME")
    tokenizer = os.getenv("TOKENIZER_NAME") or model

    lora = os.getenv('LORA') or None

    revisions = {
        "model": os.getenv("MODEL_REVISION") or None,
        "tokenizer": os.getenv("TOKENIZER_REVISION") or None
    }

    if not model or not download_dir:
        raise ValueError(f"Must specify model and download_dir. Model: {model}, download_dir: {download_dir}")

    os.makedirs(download_dir, exist_ok=True)
    model_folder = download_weights_from_hf(model_name_or_path=model,allow_patterns = ["*.safetensors", "*.bin"],
                                                                               revision=revisions["model"],
                                                                               cache_dir=download_dir)

    if lora:
        folder = download_lora_weight(lora)
        move_files(folder, '/lora')
    model_extras_folder = download_extras_or_tokenizer(model, download_dir, revisions["model"], extras=True)
    move_files(model_extras_folder, model_folder)

    # if os.environ.get("TENSORIZE_MODEL"): TODO: Add back when tensorizer is implemented

    with open("/local_model_path.txt", "w") as f:
        f.write(model_folder)

    tokenizer_folder = download_extras_or_tokenizer(tokenizer, download_dir, revisions["tokenizer"])
    with open("/local_tokenizer_path.txt", "w") as f:
        f.write(tokenizer_folder)
