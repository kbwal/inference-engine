from transformers import AutoTokenizer, Qwen2Tokenizer
import torch


# the type for qwen3-4b's tokenizer is Qwen2Tokenizer. i assume they didn't change the tokenizer!
def gen_tokenizer():
    tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    tokenizer.padding_side = "left"
    return tokenizer


tokenizer = gen_tokenizer()


def encode(
    msgs: list[list[dict[str, str]]], device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    assert all(
        list(msgs[0][i].keys()) == ["role", "content"] for i in range(len(msgs[0]))
    ), "make sure your messages have a 'role' and a 'content' field"
    formatted_prompt = tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        padding=True,
        return_tensors="pt",
    )
    # (B, seq_len)
    token_ids: torch.Tensor = formatted_prompt.input_ids  # type: ignore
    mask: torch.Tensor = formatted_prompt.attention_mask  # type: ignore
    return token_ids.to(device), mask.to(device)


def decode(token_ids: torch.Tensor | list[torch.Tensor]) -> list[str]:
    return tokenizer.batch_decode(token_ids, skip_special_tokens=True)

