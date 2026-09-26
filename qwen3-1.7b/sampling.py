from architecture import Qwen3_1_7B
import torch
from run_tokenization import encode, decode


@torch.inference_mode()
def batch_greedy_decode(
    model: Qwen3_1_7B,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    raw = model.forward(token_ids, attention_mask=attention_mask)
    predicted_tokens = torch.argmax(raw, -1, keepdim=True)
    return predicted_tokens


@torch.inference_mode()
def batch_temperature_sampling(
    model: Qwen3_1_7B,
    token_ids: torch.Tensor,
    tau: float,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    assert tau >= 0.0, "make sure temperature is positive!"

    if tau == 0.0:
        return batch_greedy_decode(
            model=model, token_ids=token_ids, attention_mask=attention_mask
        )

    raw = model(token_ids, attention_mask=attention_mask)
    probs = (raw / tau).softmax(-1)
    predicted_tokens = torch.multinomial(probs, num_samples=1)
    return predicted_tokens


@torch.inference_mode()
def autoregress(
    model: Qwen3_1_7B,
    inputs: list[str],
    tau: float,
    max_new_tokens: int,
    stop_token_id: int,
    device: str,
    drop_stopped: bool = True,
) -> list[str]:
    model = model.to(device)

    msgs = []
    for input in inputs:
        msgs.append([{"role": "user", "content": input}])
    token_ids, attention_mask = encode(msgs, device=device)

    B = token_ids.shape[0]
    prompt_len = token_ids.shape[1]

    finished_sequences = torch.zeros((token_ids.shape[0], 1), device=device).bool()
    active_indices = torch.arange(B, device=device)
    outputs: list[torch.Tensor | None] = [None] * B

    for _ in range(max_new_tokens):
        if token_ids.shape[0] == 0:
            break

        predicted_tokens = batch_temperature_sampling(
            model=model, token_ids=token_ids, tau=tau, attention_mask=attention_mask
        )
        stopped = predicted_tokens == stop_token_id
        finished_sequences = torch.logical_or(stopped, finished_sequences)
        tokens_to_add = torch.where(
            finished_sequences,
            stop_token_id,
            predicted_tokens,
        )
        token_ids = torch.cat((token_ids, tokens_to_add), dim=1)
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones((token_ids.size(0), 1), device=device),
            ),
            dim=-1,
        )

        if drop_stopped and stopped.any():
            stopped_indexes = torch.where(stopped.squeeze(-1))[0]
            for idx in stopped_indexes:
                outputs[int(active_indices[idx].item())] = token_ids[idx, prompt_len:]
            keep = ~(stopped.squeeze(-1))
            token_ids = token_ids[keep]
            attention_mask = attention_mask[keep]
            finished_sequences = finished_sequences[keep]
            active_indices = active_indices[keep]

    for idx in range(token_ids.shape[0]):
        outputs[int(active_indices[idx].item())] = token_ids[idx, prompt_len:]

    return decode([out for out in outputs if out is not None])
