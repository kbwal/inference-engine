import argparse
import gc
import math
import random
import time
import torch
from engine import Engine, Request, enable_tunableop
from run_tokenization import gen_tokenizer, encode
from architecture import make_qwen_1_7, Qwen3_1_7B, AttentionLayer
from load_weights import load_weights

FILLER = (
    "The history of computing is full of ideas that were invented long before the "
    "hardware existed to make them practical. Early mechanical calculators could add "
    "and subtract, but it took decades of work on vacuum tubes, transistors, and "
    "integrated circuits before general purpose machines became cheap enough for "
    "ordinary people to own. Along the way, researchers developed compilers, operating "
    "systems, networks, and databases, each of which changed how software was written. "
)
QUESTIONS = [
    "how does the light dependent reaction work?",
    "write me a binary search in cpp.",
    "what is euclid's infinite primes proof?",
    "my name is colonel mustard. which game am i from? am i from a game at all? who do you think killed me.",
    "your reply to this message should be one word. do you like cats or dogs? one word.",
]


class Prompts:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.overhead = len(self.ids(""))
        self.filler_ids = tokenizer.encode(FILLER)

    def ids(self, text: str) -> list[int]:
        return encode([[{"role": "user", "content": text}]], device="cpu")[0][
            0
        ].tolist()

    def of_length(self, n_tokens: int, i: int = 0) -> list[int]:
        question = QUESTIONS[i % len(QUESTIONS)]
        n_filler = n_tokens - self.overhead - len(self.tokenizer.encode(question)) - 8
        if n_filler <= 0:
            return self.ids(question)
        filler = (self.filler_ids * (n_filler // len(self.filler_ids) + 1))[:n_filler]
        return self.ids(
            f"Context: {self.tokenizer.decode(filler)}\n\nQuestion: {question}"
        )


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def tpot(r: Request) -> float | None:
    # mean time between a request's output tokens, None if it only made one
    n = len(r.output)
    return (r.t_finish - r.t_first_token) / (n - 1) if n > 1 else None  # type: ignore


def kv_bytes_per_token(model: Qwen3_1_7B) -> int:
    attn: AttentionLayer = model.model.layers[0].self_attn  # type: ignore
    per_layer = 2 * attn.num_kv_heads * attn.head_dim + 2 * attn.num_kv_heads * 4  # type: ignore
    return per_layer * len(model.model.layers)


def make_engine(
    model: Qwen3_1_7B, max_batch_size: int, max_len: int, max_tokens_per_step: int
) -> Engine:
    # stop_token_id=-1 is never sampled, so every request generates exactly max_new_tokens
    return Engine(
        model,
        max_batch_size=max_batch_size,
        max_len=max_len,
        stop_token_id=-1,
        tau=0.0,
        max_tokens_per_step=max(max_tokens_per_step, max_batch_size + 64),
    )


def free_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


# decode


def time_decode(model, prompt: list[int], B: int, args) -> tuple[float, float]:
    # returns (engine build + graph capture seconds, seconds per decode step)
    L, warmup = len(prompt), 2
    # with budget >= B + L, every step finishes at least one prompt's prefill, so all B are
    # decoding within B steps. the earliest request has made at most B tokens by then, so with
    # this max_new_tokens nobody finishes (and shrinks the batch) before timing is done
    max_new = B + warmup + args.decode_tokens + 1
    t = time.perf_counter()
    engine = make_engine(
        model, B, L + max_new + 1, max(args.max_tokens_per_step, B + L)
    )
    capture_s = time.perf_counter() - t

    for _ in range(B):
        engine.submit_request(prompt, max_new)
    while engine.waiting_queue or any(
        r.cached_len < len(r.prompt_ids) for r in engine.currently_processing_list
    ):
        engine.step()
    for _ in range(warmup):
        engine.step()
    t = time.perf_counter()
    for _ in range(args.decode_tokens):
        engine.step()  # ends with a .tolist(), so the gpu work is done when it returns
    step_s = (time.perf_counter() - t) / args.decode_tokens
    assert len(engine.currently_processing_list) == B, "batch shrank while timing"
    return capture_s, step_s


def bench_decode(model, prompts: Prompts, args) -> None:
    print("\n== decode: B requests at the same context, all decoding in lockstep")
    print(f"   ms/step averaged over {args.decode_tokens} steps\n")
    print(
        f"{'batch':>5} | {'context':>7} | {'capture':>7} | {'ms/step':>7} | {'tok/s/seq':>9} | {'tok/s total':>11}"
    )
    print(f"{'':>5} | {'':>7} | {'(s)':>7} | {'':>7} | {'':>9} | {'':>11}")
    per_token = kv_bytes_per_token(model)
    for ctx in args.decode_contexts:
        prompt = prompts.of_length(ctx)
        for B in args.decode_batches:
            row = f"{B:>5} | {len(prompt):>7} | "
            max_len = len(prompt) + B + args.decode_tokens + 4
            cache_gb = (B + 1) * max_len * per_token / 1e9
            if cache_gb > args.max_cache_gb:
                print(row + f"skipped: kv cache would be {cache_gb:.1f} GB")
                continue
            try:
                capture_s, step_s = time_decode(model, prompt, B, args)
                print(
                    row + f"{capture_s:>7.1f} | {step_s * 1000:>7.2f} | "
                    f"{1 / step_s:>9.1f} | {B / step_s:>11.0f}"
                )
            except torch.OutOfMemoryError:
                print(row + "out of memory")
            free_gpu_memory()


# prefill


def time_prefill(model, prompt: list[int], args) -> tuple[int, float]:
    # returns (steps the prompt was chunked into, median ttft seconds)
    engine = make_engine(model, 1, len(prompt) + 1, args.max_tokens_per_step)
    engine.generate(
        [prompt], max_new_tokens=1
    )  # warmup: autotune for these chunk sizes
    ttfts = []
    for _ in range(args.prefill_reps):
        (r,) = engine.generate([prompt], max_new_tokens=1)
        ttfts.append(r.t_first_token - r.t_submit)  # type: ignore
    return math.ceil(len(prompt) / engine.max_tokens_per_step), pct(ttfts, 50)


def bench_prefill(model, prompts: Prompts, args) -> None:
    print("\n== prefill: one request on an idle engine, time to its first token")
    print(
        f"   max_tokens_per_step={args.max_tokens_per_step}, so longer prompts take several steps\n"
    )
    print(f"{'prompt':>7} | {'steps':>5} | {'ttft':>8} | {'prefill tok/s':>13}")
    print(f"{'':>7} | {'':>5} | {'(ms)':>8} | {'':>13}")
    for n in args.prefill_lengths:
        prompt = prompts.of_length(n)
        try:
            steps, ttft = time_prefill(model, prompt, args)
            print(
                f"{len(prompt):>7} | {steps:>5} | {ttft * 1000:>8.1f} | {len(prompt) / ttft:>13,.0f}"
            )
        except torch.OutOfMemoryError:
            print(f"{len(prompt):>7} | out of memory")
        free_gpu_memory()


# serve


def make_workload(
    n: int, rng: random.Random, prompts: Prompts
) -> list[tuple[list[int], int]]:
    # mostly short chat turns, some documents, a few long ones. output lengths vary independently
    work = []
    for i in range(n):
        r = rng.random()
        target = (
            rng.randint(32, 256)
            if r < 0.70
            else rng.randint(512, 2048) if r < 0.95 else 4096
        )
        work.append((prompts.of_length(target, i), rng.randint(32, 256)))
    return work


def poisson_arrivals(n: int, rate: float, seed: int) -> list[float]:
    # rate inf means everything arrives at t=0
    rng, t, arrivals = random.Random(seed), 0.0, []
    for _ in range(n):
        t += 0.0 if math.isinf(rate) else rng.expovariate(rate)
        arrivals.append(t)
    return arrivals


def is_graph_step(engine: Engine) -> bool:
    # mirrors Engine.step: it replays a cuda graph iff every running request is decoding and
    # nothing gets admitted. admission happens whenever something waits and a slot is free,
    # since the budget left after decodes is always >= min_first_chunk
    running = engine.currently_processing_list
    return (
        bool(running)
        and all(r.cached_len >= len(r.prompt_ids) for r in running)
        and not (engine.waiting_queue and engine.free_slots)
    )


def serve(engine: Engine, work, arrivals: list[float]):
    # submit each request once its arrival time has passed, step whenever there's work, idle otherwise.
    # returns (requests, elapsed seconds, per step: (was a graph replay, batch size, seconds))
    reqs: list[Request] = []
    steps: list[tuple[bool, int, float]] = []
    start = time.perf_counter()
    i = 0
    while i < len(work) or engine.waiting_queue or engine.currently_processing_list:
        now = time.perf_counter() - start
        while i < len(work) and arrivals[i] <= now:
            req = engine.submit_request(*work[i])
            req.t_submit = (
                start + arrivals[i]
            )  # from when it arrived, not when we noticed
            reqs.append(req)
            i += 1
        if engine.waiting_queue or engine.currently_processing_list:
            graph = is_graph_step(engine)
            batch = len(engine.currently_processing_list)
            t = time.perf_counter()
            engine.step()  # ends with a .tolist(), so the gpu work is done when it returns
            steps.append((graph, batch, time.perf_counter() - t))
        elif i < len(work):
            time.sleep(max(0.0, arrivals[i] - (time.perf_counter() - start)))
    return reqs, time.perf_counter() - start, steps


def warmup_prefill_buckets(engine: Engine, prompt: list[int]) -> None:
    # prefill_attention autotunes once per power-of-2 token count (T_BUCKET). hit every bucket
    # up front so the first rate in the sweep doesn't pay for it
    cap = min(engine.max_tokens_per_step, engine.max_len - 1, len(prompt))
    for b in range((cap - 1).bit_length() + 1):
        engine.generate([prompt[: min(1 << b, cap)]], max_new_tokens=1)


def bench_serve(model, prompts: Prompts, args) -> None:
    work = make_workload(args.requests, random.Random(args.seed), prompts)
    lens = [len(p) for p, _ in work]
    outs = [m for _, m in work]
    engine = make_engine(
        model,
        args.max_batch_size,
        max(len(p) + m for p, m in work) + 1,
        args.max_tokens_per_step,
    )
    print(
        f"\n== serve: {args.requests} requests arriving at random (poisson) times, rate 'inf' = all at once"
    )
    print(
        f"   prompts {min(lens)}-{max(lens)} tokens (mean {mean(lens):.0f}), outputs {min(outs)}-{max(outs)} "  # type: ignore
        f"(mean {mean(outs):.0f}). max_batch_size={args.max_batch_size}, "  # type: ignore
        f"max_tokens_per_step={engine.max_tokens_per_step}"
    )
    print(
        f"   goodput = requests/s that met the slo: ttft <= {args.slo_ttft_ms:g} ms and tpot <= {args.slo_tpot_ms:g} ms\n"
    )
    warmup_prefill_buckets(engine, prompts.of_length(engine.max_tokens_per_step + 64))

    print(
        f"{'rate/s':>6} | {'ttft p50':>8} | {'ttft p99':>8} | {'tpot p50':>8} | {'tpot p99':>8} | "
        f"{'out tok/s':>9} | {'req/s':>5} | {'goodput':>12} || {'graph':>5} | {'graph':>5} | {'graph':>5} | {'eager':>5}"
    )
    print(
        f"{'':>6} | {'(ms)':>8} | {'(ms)':>8} | {'(ms)':>8} | {'(ms)':>8} | {'':>9} | {'':>5} | {'req/s (%)':>12} || "
        f"{'steps':>5} | {'batch':>5} | {'ms':>5} | {'ms':>5}"
    )
    for rate in args.rates:
        arrivals = poisson_arrivals(len(work), rate, args.seed + 1)
        reqs, elapsed, steps = serve(engine, work, arrivals)

        ttfts = [r.t_first_token - r.t_submit for r in reqs]  # type: ignore
        tpots = [tpot(r) for r in reqs]
        good = sum(
            1
            for a, b in zip(ttfts, tpots)
            if a * 1000 <= args.slo_ttft_ms
            and (b is None or b * 1000 <= args.slo_tpot_ms)
        )
        graph = [(batch, s) for is_graph, batch, s in steps if is_graph]
        eager = [s for is_graph, _, s in steps if not is_graph]
        tpots = [t for t in tpots if t is not None]
        print(
            f"{rate:>6g} | {pct(ttfts, 50) * 1000:>8.0f} | {pct(ttfts, 99) * 1000:>8.0f} | "
            f"{pct(tpots, 50) * 1000:>8.1f} | {pct(tpots, 99) * 1000:>8.1f} | "
            f"{sum(len(r.output) for r in reqs) / elapsed:>9.0f} | {len(reqs) / elapsed:>5.1f} | "
            f"{good / elapsed:>5.1f} ({good / len(reqs) * 100:>3.0f}%) || "
            f"{len(graph) / len(steps) * 100:>4.0f}% | {mean([b for b, _ in graph]):>5.1f} | "
            f"{mean([s for _, s in graph]) * 1000:>5.1f} | {mean(eager) * 1000:>5.1f}"
        )
    print(
        "\n   right of ||: share of steps that replayed a cuda graph, their mean batch size, and mean step time "
        "for graph vs eager (prefill-carrying) steps"
    )


# main

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--sections",
        nargs="+",
        default=["decode", "prefill", "serve"],
        choices=["decode", "prefill", "serve"],
    )
    ap.add_argument(
        "--quick", action="store_true", help="smaller sweeps and fewer requests"
    )
    ap.add_argument("--max-tokens-per-step", type=int, default=1024)
    # decode
    ap.add_argument(
        "--decode-batches", type=int, nargs="+", default=[1, 4, 16, 32, 64, 128, 256]
    )
    ap.add_argument("--decode-contexts", type=int, nargs="+", default=[128, 1024, 4096])
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument(
        "--max-cache-gb",
        type=float,
        default=14.0,
        help="skip decode configs whose kv cache is bigger",
    )
    # prefill
    ap.add_argument(
        "--prefill-lengths", type=int, nargs="+", default=[128, 512, 2048, 8192]
    )
    ap.add_argument("--prefill-reps", type=int, default=3)
    # serve
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=[2, 4, 8, 16, math.inf],
        help="mean arrivals per second",
    )
    ap.add_argument("--max-batch-size", type=int, default=32)
    ap.add_argument("--slo-ttft-ms", type=float, default=1000)
    ap.add_argument("--slo-tpot-ms", type=float, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.quick:
        args.decode_batches, args.decode_contexts = [1, 16], [128, 2048]
        args.prefill_lengths, args.prefill_reps = [128, 2048], 2
        args.requests, args.rates = 32, [8, math.inf]

    model = make_qwen_1_7(device="meta")
    model.load_state_dict(load_weights(device="cuda"), assign=True)
    enable_tunableop()
    prompts = Prompts(gen_tokenizer())

    sections = {"decode": bench_decode, "prefill": bench_prefill, "serve": bench_serve}
    for name in args.sections:
        sections[name](model, prompts, args)
        free_gpu_memory()
