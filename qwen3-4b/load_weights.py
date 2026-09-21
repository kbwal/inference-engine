import glob, torch
from safetensors import safe_open


def load_weights(model_path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    shard_files = sorted(glob.glob(model_path + "/*.safetensors"))
    weights = {}
    for file in shard_files:
        with safe_open(filename=file, framework="pt", device=device) as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    return weights
