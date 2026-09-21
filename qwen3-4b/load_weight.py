import glob, torch
from safetensors import safe_open
from huggingface_hub import snapshot_download


def load_weight(model_path: str, weight_name: str) -> torch.Tensor | None:
    assert torch.mps.is_available(), "MPS couldn't be found"
    shard_files = sorted(glob.glob(model_path + "/*.safetensors"))
    for file in shard_files:
        w = safe_open(filename=file, framework="pt", device="mps")
        if weight_name in w.keys():
            return w.get_tensor(weight_name)
