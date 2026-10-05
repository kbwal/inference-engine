from sampling import autoregress
from transformers import AutoTokenizer
from architecture import make_granite_1b_400m
from load_weights import load_weights

if __name__ == "__main__":
    model = make_granite_1b_400m(device="meta")
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.1-1b-a400m-base")
    assert type(tokenizer.eos_token_id) == int
    model.load_state_dict(load_weights(device="cuda"), assign=True)

    MAX_NEW_TOKENS = 64
    inputs = [
        "how does the light dependent reaction work?",
        "write me a binary search in cpp.",
        "what is euclid's infinite primes proof?",
        "my name is colonel mustard. which game am i from? am i from a game at all? who do you think killed me.",
        "your reply to this message should be one word. do you like cats or dogs? one word.",
    ]
    res, stats = autoregress(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        tau=0.3,
        max_new_tokens=MAX_NEW_TOKENS,
        stop_token_id=tokenizer.eos_token_id,
        device="cuda",
        drop_stopped=False,
    )
    print("result: ", res)
    print(f"prompt tokens:     {stats.prompt_tokens}")
    print(f"ttft:              {stats.ttft_s * 1000:.1f} ms")
    print(f"prefill:           {stats.prefill_tok_s:.1f} tok/s")
    print(f"decode (per seq):  {stats.decode_tok_s_per_seq:.2f} tok/s")
    print(f"decode (batch):    {stats.decode_tok_s_batch:.2f} tok/s")
