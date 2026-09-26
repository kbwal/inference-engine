from architecture import Qwen3_1_7B, make_qwen_1_7
import torch
from load_weights import load_weights
from run_tokenization import encode, decode


@torch.inference_mode()
def batch_greedy_decode(
    model: Qwen3_1_7B,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    raw = model.forward(token_ids, attention_mask=attention_mask)
    predicted_tokens = torch.argmax(raw, -1)
    next_ids = torch.cat((token_ids, predicted_tokens.unsqueeze(dim=1)), dim=1)
    return (
        decode(predicted_tokens),
        predicted_tokens,
        next_ids,
    )


@torch.inference_mode()
def batch_temperature_sampling(
    model: Qwen3_1_7B,
    token_ids: torch.Tensor,
    tau: float,
    attention_mask: torch.Tensor | None = None,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    assert tau >= 0.0, "make sure temperature is positive!"

    if tau == 0.0:
        return batch_greedy_decode(
            model=model, token_ids=token_ids, attention_mask=attention_mask
        )

    raw = model.forward(token_ids, attention_mask=attention_mask)
    probs = (raw / tau).softmax(-1)
    sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
    next_ids = torch.cat((token_ids, sampled_indices.unsqueeze(dim=1)), dim=1)
    return (
        decode(sampled_indices),
        sampled_indices,
        next_ids,
    )


@torch.inference_mode()
def autoregress(
    model: Qwen3_1_7B,
    inputs: list[str],
    tau: float,
    max_new_tokens: int,
    device,
) -> list[str]:
    model = model.to(device)

    msgs = []
    for input in inputs:
        msgs.append([{"role": "user", "content": input}])
    token_ids, attention_mask = encode(msgs)
    token_ids = token_ids.to(device)
    attention_mask = attention_mask.to(device)

    prompt_len = token_ids.shape[1]

    for _ in range(max_new_tokens):
        _, _, next_ids = batch_temperature_sampling(
            model=model, token_ids=token_ids, tau=tau, attention_mask=attention_mask
        )
        token_ids = next_ids
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (token_ids.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=device,
                ),
            ),
            dim=-1,
        )
    return decode(token_ids[:, prompt_len:])


if __name__ == "__main__":
    model = make_qwen_1_7()
    model.load_state_dict(load_weights(device="cpu"))
    res = autoregress(
        model=model,
        inputs=["2*3=", "What is your name?"],
        tau=0.3,
        max_new_tokens=32,
        device="mps",
    )
    print(res)
