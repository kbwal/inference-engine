from architecture import Qwen3_1_7B
import time
from dataclasses import dataclass
import torch
from run_tokenization import encode, decode


@dataclass
class GenerationStats:
    prompt_tokens: int  # non-pad
    ttft_s: float  # time to first token
    prefill_tok_s: float  # input_tokens / ttft_s
    decode_steps: int  # max over all in batch, even if some are dropped
    decode_s: float  # overall_time - ttft
    decode_tok_s_per_seq: float  # decode_steps / decode_seconds
    decode_tok_s_batch: float  # decode_tokens / decode_seconds


def sync():
    torch.mps.synchronize()


@torch.inference_mode()
def batch_greedy_decode(
    model: Qwen3_1_7B,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    raw, new_kv_cache = model.forward(
        token_ids, attention_mask=attention_mask, kv_cache=kv_cache
    )
    predicted_tokens = torch.argmax(raw, -1, keepdim=True)
    return predicted_tokens, new_kv_cache


@torch.inference_mode()
def batch_temperature_sampling(
    model: Qwen3_1_7B,
    token_ids: torch.Tensor,
    tau: float,
    attention_mask: torch.Tensor | None = None,
    kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    assert tau >= 0.0, "make sure temperature is positive!"

    if tau == 0.0:
        return batch_greedy_decode(
            model=model,
            token_ids=token_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )

    raw, new_kv_cache = model(
        token_ids, attention_mask=attention_mask, kv_cache=kv_cache
    )
    probs = (raw / tau).softmax(-1)
    predicted_tokens = torch.multinomial(probs, num_samples=1)
    return predicted_tokens, new_kv_cache


@torch.inference_mode()
def autoregress(
    model: Qwen3_1_7B,
    inputs: list[str],
    tau: float,
    max_new_tokens: int,
    stop_token_id: int,
    device: str,
    drop_stopped: bool = True,
    use_kv_cache: bool = True,
) -> tuple[list[str], GenerationStats]:
    model = model.to(device)
    sync()
    t_start = time.perf_counter()

    msgs = []
    for input in inputs:
        msgs.append([{"role": "user", "content": input}])
    token_ids, attention_mask = encode(msgs, device=device)

    B = token_ids.shape[0]
    prompt_len = token_ids.shape[1]

    finished_sequences = torch.zeros((token_ids.shape[0], 1), device=device).bool()
    active_indices = torch.arange(B, device=device)
    outputs: list[torch.Tensor | None] = [None] * B
    prompt_tokens = attention_mask.sum()

    t_first = t_start
    num_steps = 0
    decode_tokens = 0
    kv_cache = None
    for step in range(max_new_tokens):
        if token_ids.shape[0] == 0:
            break

        predicted_tokens, new_kv_cache = batch_temperature_sampling(
            model=model,
            token_ids=token_ids,
            tau=tau,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = new_kv_cache if use_kv_cache else None
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
            if kv_cache is not None:
                kv_cache = [(k[keep], v[keep]) for k, v in kv_cache]

        num_steps += 1
        if step == 0:
            sync()
            t_first = time.perf_counter()
        else:
            decode_tokens += predicted_tokens.shape[0]

    sync()
    t_end = time.perf_counter()

    for idx in range(token_ids.shape[0]):
        outputs[int(active_indices[idx].item())] = token_ids[idx, prompt_len:]

    n_prompt = int(prompt_tokens.item())
    ttft_s = t_first - t_start
    decode_s = t_end - t_first
    decode_steps = max(num_steps - 1, 0)
    stats = GenerationStats(
        prompt_tokens=n_prompt,
        ttft_s=ttft_s,
        prefill_tok_s=n_prompt / ttft_s,
        decode_steps=decode_steps,
        decode_s=decode_s,
        decode_tok_s_per_seq=decode_steps / decode_s if decode_steps else 0.0,
        decode_tok_s_batch=decode_tokens / decode_s if decode_steps else 0.0,
    )
    return decode([out for out in outputs if out is not None]), stats
