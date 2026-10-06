from architecture import Granite3_1_1B_400M, AttentionLayer
import time
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class GenerationStats:
    prompt_tokens: int  # non-pad
    ttft_s: float  # time to first token
    prefill_tok_s: float  # input_tokens / ttft_s
    decode_steps: int  # max over all in batch
    capture_s: float  # cuda graph capture, excluded from both ttft and decode
    decode_s: float  # overall_time - ttft - capture
    decode_tok_s_per_seq: float  # decode_steps / decode_seconds
    decode_tok_s_batch: float  # decode_tokens / decode_seconds


def sync():
    torch.cuda.synchronize()


def capture_decode_graph(
    model: Granite3_1_1B_400M,
    kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
    next_input: torch.Tensor,
    cache_pos: torch.Tensor,
    positions: torch.Tensor,
    key_mask: torch.Tensor,
):
    static_input = next_input.clone()
    static_cache_pos = cache_pos.clone()
    static_positions = positions.clone()

    def run() -> torch.Tensor:
        return model(
            static_input,
            kv_cache=kv_cache,
            cache_pos=static_cache_pos,
            positions=static_positions,
            key_mask=key_mask,
        )

    # warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_logits = run()

    return graph, static_input, static_cache_pos, static_positions, static_logits


def sample_from_logits(logits: torch.Tensor, tau: float) -> torch.Tensor:
    if tau == 0:
        predicted_tokens = torch.argmax(logits, -1, keepdim=True)
    else:
        probs = (logits / tau).softmax(-1)
        predicted_tokens = torch.multinomial(probs, num_samples=1)
    return predicted_tokens


@torch.inference_mode()
def autoregress(
    model: Granite3_1_1B_400M,
    tokenizer,
    inputs: list[str],
    tau: float,
    max_new_tokens: int,
    stop_token_id: int,
    device: str,
) -> tuple[list[str], GenerationStats]:
    model = model.to(device)
    sync()
    t_start = time.perf_counter()

    enc: dict[str, torch.Tensor] = tokenizer(
        inputs, padding=True, return_tensors="pt"
    ).to(device)
    token_ids, attention_mask = enc["input_ids"], enc["attention_mask"]

    assert type(token_ids) == torch.Tensor and type(attention_mask) == torch.Tensor

    B = token_ids.shape[0]
    prompt_len = token_ids.shape[1]

    finished_sequences = torch.zeros((B, 1), device=device).bool()
    prompt_tokens = attention_mask.sum()

    t_first = t_start
    t_decode_start = t_start
    num_steps = 0
    decode_tokens = 0

    max_len = prompt_len + max_new_tokens
    attn: AttentionLayer = model.model.layers[0].self_attn  # type: ignore

    kv_cache = [
        (
            torch.zeros(
                B,
                attn.num_kv_heads,
                max_len,
                attn.head_dim,
                device=device,
                dtype=model.dtype,
            ),
            torch.zeros(
                B,
                attn.num_kv_heads,
                max_len,
                attn.head_dim,
                device=device,
                dtype=model.dtype,
            ),
        )
        for _ in model.model.layers
    ]
    cache_pos = torch.arange(prompt_len, device=kv_cache[0][0].device)
    positions = (attention_mask.cumsum(dim=1) - 1).clamp(0)
    key_mask = F.pad(attention_mask, (0, max_len - prompt_len), value=1).bool()
    next_input = token_ids

    graph: torch.cuda.CUDAGraph | None = None
    static_input: torch.Tensor | None = None
    static_cache_pos: torch.Tensor | None = None
    static_positions: torch.Tensor | None = None
    static_logits: torch.Tensor | None = None

    for step in range(max_new_tokens):
        if step == 0:
            # no need to capture a graph on prefill cuz it's variable length
            # plus prefill is high arithmetic intensity, so the overhead doesn't matter as much
            logits = model(
                token_ids,
                kv_cache=kv_cache,
                cache_pos=cache_pos,
                positions=positions,
                key_mask=key_mask,
            )
        else:
            assert (
                graph is not None
                and static_input is not None
                and static_cache_pos is not None
                and static_positions is not None
                and static_logits is not None
            )
            static_input.copy_(next_input)
            static_cache_pos.copy_(cache_pos)
            static_positions.copy_(positions)
            graph.replay()
            logits = static_logits

        predicted_tokens = sample_from_logits(logits=logits, tau=tau)

        cache_pos = (
            cache_pos[-1:]
            + 1
            + torch.arange(predicted_tokens.shape[1], device=cache_pos.device)
        )
        positions = (
            positions[:, -1:]
            + 1
            + torch.arange(predicted_tokens.shape[1], device=positions.device)
        )
        stopped = predicted_tokens == stop_token_id
        finished_sequences = torch.logical_or(stopped, finished_sequences)
        tokens_to_add = torch.where(
            finished_sequences,
            stop_token_id,
            predicted_tokens,
        )
        next_input = tokens_to_add
        token_ids = torch.cat((token_ids, tokens_to_add), dim=1)

        num_steps += 1
        if step == 0:
            sync()
            t_first = time.perf_counter()
            graph, static_input, static_cache_pos, static_positions, static_logits = (
                capture_decode_graph(
                    model, kv_cache, next_input, cache_pos, positions, key_mask
                )
            )
            sync()
            t_decode_start = time.perf_counter()
        else:
            decode_tokens += predicted_tokens.shape[0]

    sync()
    t_end = time.perf_counter()

    n_prompt = int(prompt_tokens.item())
    ttft_s = t_first - t_start
    capture_s = t_decode_start - t_first
    decode_s = t_end - t_decode_start
    decode_steps = max(num_steps - 1, 0)
    stats = GenerationStats(
        prompt_tokens=n_prompt,
        ttft_s=ttft_s,
        prefill_tok_s=n_prompt / ttft_s,
        decode_steps=decode_steps,
        capture_s=capture_s,
        decode_s=decode_s,
        decode_tok_s_per_seq=decode_steps / decode_s if decode_steps else 0.0,
        decode_tok_s_batch=decode_tokens / decode_s if decode_steps else 0.0,
    )
    return (
        tokenizer.batch_decode(token_ids[:, prompt_len:], skip_special_tokens=True),
        stats,
    )
