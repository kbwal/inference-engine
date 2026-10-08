import torch
from sampling import autoregress, enable_tunableop
from run_tokenization import gen_tokenizer, encode
from architecture import make_qwen_1_7
from load_weights import load_weights

FILLER = (
    "The history of computing is full of ideas that were invented long before the "
    "hardware existed to make them practical. Early mechanical calculators could add "
    "and subtract, but it took decades of work on vacuum tubes, transistors, and "
    "integrated circuits before general purpose machines became cheap enough for "
    "ordinary people to own. Along the way, researchers developed compilers, operating "
    "systems, networks, and databases, each of which changed how software was written. "
)


def make_prompt(question: str, n_tokens: int, tokenizer, template_overhead: int) -> str:
    n_filler = n_tokens - template_overhead - len(tokenizer.encode(question)) - 8
    if n_filler <= 0:
        return question
    filler_ids = tokenizer.encode(FILLER)
    filler_ids = (filler_ids * (n_filler // len(filler_ids) + 1))[:n_filler]
    return f"Context: {tokenizer.decode(filler_ids)}\n\nQuestion: {question}"


if __name__ == "__main__":
    model = make_qwen_1_7(device="meta")
    tokenizer = gen_tokenizer()
    stop_token_id = tokenizer.eos_token_id
    assert type(stop_token_id) == int
    model.load_state_dict(load_weights(device="cuda"), assign=True)
    enable_tunableop()

    MAX_NEW_TOKENS = 128
    BATCH_SIZES = [1, 4, 16, 32, 64, 128, 256, 512]
    CONTEXTS = [128, 512, 2048, 4096, 8192]  # prompt length
    MAX_BATCH = {128: 512, 512: 512, 2048: 128, 4096: 64, 8192: 32}
    prompts = [
        "how does the light dependent reaction work?",
        "write me a binary search in cpp.",
        "what is euclid's infinite primes proof?",
        "my name is colonel mustard. which game am i from? am i from a game at all? who do you think killed me.",
        "your reply to this message should be one word. do you like cats or dogs? one word.",
    ]
    probe = [{"role": "user", "content": ""}]
    template_overhead = encode([probe], device="cpu")[0].shape[1]

    print(
        f"{'batch':>5} | {'ctx':>5} | {'prompt tok':>10} | {'ttft (ms)':>9} | {'prefill tok/s':>13} | "
        f"{'capture (ms)':>12} | {'step (ms)':>9} | {'decode tok/s/seq':>16} | {'decode tok/s total':>18}"
    )
    grid: dict[tuple[int, int], float | None] = {}
    res = None
    for ctx in CONTEXTS:
        ctx_prompts = [
            make_prompt(p, ctx, tokenizer, template_overhead) for p in prompts
        ]
        for batch_size in BATCH_SIZES:
            if batch_size > MAX_BATCH[ctx]:
                break
            inputs = [ctx_prompts[i % len(ctx_prompts)] for i in range(batch_size)]
            oom = False
            try:
                # dummy call for triton autotuning, warmup
                autoregress(
                    model=model,
                    inputs=inputs,
                    tau=0.3,
                    max_new_tokens=MAX_NEW_TOKENS,
                    stop_token_id=stop_token_id,
                    device="cuda",
                )
                res, stats = autoregress(
                    model=model,
                    inputs=inputs,
                    tau=0.3,
                    max_new_tokens=MAX_NEW_TOKENS,
                    stop_token_id=stop_token_id,
                    device="cuda",
                )
            except torch.OutOfMemoryError:
                oom = True
                stats = None
            if oom:
                torch.cuda.empty_cache()
                grid[(batch_size, ctx)] = None
                print(f"{batch_size:>5} | {ctx:>5} | out of memory")
                continue
            assert stats is not None
            # release this config's cached blocks so fragmentation doesn't OOM the near-full ones
            torch.cuda.empty_cache()
            grid[(batch_size, ctx)] = stats.decode_tok_s_batch
            print(
                f"{batch_size:>5} | {ctx:>5} | {stats.prompt_tokens // batch_size:>10} | "
                f"{stats.ttft_s * 1000:>9.1f} | {stats.prefill_tok_s:>13.1f} | "
                f"{stats.capture_s * 1000:>12.1f} | "
                f"{stats.decode_s / stats.decode_steps * 1000:>9.2f} | "
                f"{stats.decode_tok_s_per_seq:>16.2f} | "
                f"{stats.decode_tok_s_batch:>18.2f}"
            )

    print("\nsample output (last run, first seq):", repr(res[0]) if res else None)
