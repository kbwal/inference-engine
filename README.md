A custom inference engine to learn about how model inference works.

I first built this on Qwen3-1.7B (a vanilla transformer) on an RTX 3090. I might consider moving on to other things later, like MoE or different types of attention.

## Benchmark vs vLLM

Qwen3-1.7B bf16 on one RTX 3090, against vLLM 0.32.0.

**Matched settings:** 8-bit KV cache, CUDA graphs, chunked prefill with a 1024 token/step budget, max concurrency 32 (also 128), greedy sampling, token ids in/out, prefix caching off, distinct prompts.

**vLLM tuning:** each KV format runs on its fastest attention backend: fp8 on FlashInfer, int8_per_token_head on Triton. Inductor max_autotune is on with `compile_sizes`, which cut vLLM's step time by 5-12%. Block size, custom ops, budget and graph sizes made no difference.

### Serving (ShareGPT, concurrency 32)

SLO capacity is the highest arrival rate where 90% of requests meet TTFT <= 1 s and TPOT <= 50 ms.

Higher is better in every row.

| metric | mine | best vLLM (fp8 KV, tuned) | vLLM int8 KV (same format as mine) |
|---|---|---|---|
| SLO capacity (req/s) | **16.8** | 15.2 | ~11.0 |
| goodput @ 16 req/s | **12.5** | 10.2 | not recorded |
| peak output tok/s | **3257** | 3037 | not recorded |

My engine came out ahead: it sustains 10% more requests/s within the SLO than the best tuned vLLM, and 53% more than vLLM using the same int8 KV format. Its peak throughput is 7% higher.

- Mixed workload (benchmark.py's chat/docs/long mix), concurrency 32: mine has 3% more SLO capacity and 7% more throughput than fp8 vLLM, and 57% more capacity than int8 vLLM.
- Concurrency 128: roughly tied (mine is 1-3% higher).
- vLLM has the better p99 inter-token latency. A 512 token budget brings mine from 42 to 34 ms with the same goodput.

### Decode (40 shapes, B 1-256, context 128-4096)

- vs stock vLLM: mine is >2% faster on 34/40 shapes; at the median shape mine is 1.18x faster.
- vs best tuned vLLM: mine wins 23, ties 12 (within 2%), loses 5; at the median shape mine is 1.04x faster.
  - B=2-8: mine is 1.06-1.11x faster. B=1 is a tie.
  - Mine is 2-5% slower at B>=16 with ctx 4096, and at B=256.
- Profile at B=8 (mine vs vLLM): skinny matmuls 4.23 vs 4.32 ms, attention 0.85 vs 0.96 ms.
- Profile at B=64, ctx 4K: FlashInfer attention ~6% faster, my matmuls 14% faster.

### Prefill (single request, TTFT)

Mine is 1.40x faster at 128 tokens; parity at >= 1K tokens.
