from sampling import autoregress
from transformers import AutoTokenizer
from architecture import make_granite_1b_400m
from load_weights import load_weights

if __name__ == "__main__":
    model = make_granite_1b_400m(device="meta")
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.1-1b-a400m-base")
    assert type(tokenizer.eos_token_id) == int
    model.load_state_dict(load_weights(device="cuda"), assign=True)

    MAX_NEW_TOKENS = 128
    BATCH_SIZES = [1, 4, 16, 64]
    prompts = [
        "how does the light dependent reaction work?",
        "write me a binary search in cpp.",
        "what is euclid's infinite primes proof?",
        "my name is colonel mustard. which game am i from? am i from a game at all? who do you think killed me.",
        "your reply to this message should be one word. do you like cats or dogs? one word.",
    ]

    print(
        f"{'batch':>5} | {'prompt tok':>10} | {'ttft (ms)':>9} | {'prefill tok/s':>13} | "
        f"{'capture (ms)':>12} | {'decode tok/s/seq':>16} | {'decode tok/s total':>18}"
    )
    for batch_size in BATCH_SIZES:
        inputs = [prompts[i % len(prompts)] for i in range(batch_size)]
        # dummy call per batch size so cuda init / cublas kernel selection for
        # these shapes doesn't pollute ttft
        autoregress(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
            tau=0.3,
            max_new_tokens=8,
            stop_token_id=tokenizer.eos_token_id,
            device="cuda",
        )
        res, stats = autoregress(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
            tau=0.3,
            max_new_tokens=MAX_NEW_TOKENS,
            stop_token_id=tokenizer.eos_token_id,
            device="cuda",
        )
        print(
            f"{batch_size:>5} | {stats.prompt_tokens:>10} | {stats.ttft_s * 1000:>9.1f} | "
            f"{stats.prefill_tok_s:>13.1f} | {stats.capture_s * 1000:>12.1f} | "
            f"{stats.decode_tok_s_per_seq:>16.2f} | "
            f"{stats.decode_tok_s_batch:>18.2f}"
        )

    print("\nsample output (last batch, first seq):", repr(res[0]))  # type: ignore
