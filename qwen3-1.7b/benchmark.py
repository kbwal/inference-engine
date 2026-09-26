from sampling import autoregress
from architecture import make_qwen_1_7
from load_weights import load_weights
import time

if __name__ == "__main__":
    model = make_qwen_1_7()
    model.load_state_dict(load_weights(device="cpu"))
    MAX_NEW_TOKENS = 64
    inputs = [
        "how does the light dependent reaction work?",
        "write me a binary search in cpp.",
        "what is euclid's infinite primes proof?",
        "my name is colonel mustard. which game am i from? am i from a game at all? who do you think killed me.",
    ]
    t1 = time.time()
    res = autoregress(
        model=model,
        inputs=inputs,
        tau=0.3,
        max_new_tokens=MAX_NEW_TOKENS,
        device="mps",
    )
    print(f"throughput: {MAX_NEW_TOKENS * len(inputs) / (time.time() - t1)} tok/s")
