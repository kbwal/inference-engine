import glob, torch
from safetensors import safe_open
from huggingface_hub import snapshot_download


def load_weights(
    model_path: str | None = None, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    if model_path is None:
        model_path = snapshot_download("Qwen/Qwen3-4B", local_files_only=True)
    shard_files = sorted(glob.glob(model_path + "/*.safetensors"))
    weights = {}
    for file in shard_files:
        with safe_open(filename=file, framework="pt", device=device) as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    return weights
