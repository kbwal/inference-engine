import glob, torch
from safetensors import safe_open
from huggingface_hub import snapshot_download


def load_weights(
    model_path: str | None = None,
    device: str = "cuda",
    tie_word_embeddings: bool = True,
) -> dict[str, torch.Tensor]:
    if model_path is None:
        model_path = snapshot_download("Qwen/Qwen3-1.7B", local_files_only=True)
    shard_files = sorted(glob.glob(model_path + "/*.safetensors"))
    weights = {}
    for file in shard_files:
        with safe_open(filename=file, framework="pt", device=device) as f:
            for key in f.keys():
                if tie_word_embeddings and key == "lm_head.weight":
                    continue
                weights[key] = f.get_tensor(key)

    for key in [k for k in weights if k.endswith("self_attn.q_proj.weight")]:
        prefix = key.removesuffix("q_proj.weight")
        weights[prefix + "qkv_proj.weight"] = torch.cat(
            [
                weights.pop(prefix + "q_proj.weight"),
                weights.pop(prefix + "k_proj.weight"),
                weights.pop(prefix + "v_proj.weight"),
            ],
            dim=0,
        )
    return weights
