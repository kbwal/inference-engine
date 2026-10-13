from architecture import Qwen3_1_7B, AttentionLayer, LayerKVCache, PrefixScratch
import itertools
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
import torch
import torch.cuda.tunable as tunable
from torch.cuda import nvtx
from kernels import sample
from collections import deque

TUNABLEOP_FILE = os.path.join(os.path.dirname(__file__), "tunableop_results.csv")


def enable_tunableop(path: str = TUNABLEOP_FILE):
    tunable.enable(True)
    tunable.tuning_enable(False)
    tunable.set_filename(path)


@dataclass
class Request:
    id: int
    prompt_ids: list[int]
    max_new_tokens: int
    slot: int | None = None  # if slot is None, it'll be assigned a slot
    cached_len: int = 0  # tokens currently in the kv cache
    output: list[int] = field(
        default_factory=list
    )  # generated tokens. the first one is appended when the prompt's last chunk is prefilled
    t_submit: float = 0.0
    t_first_token: float | None = None
    t_finish: float | None = None


class Engine:
    def __init__(
        self,
        model: Qwen3_1_7B,
        max_batch_size: int,
        max_len: int,  # prompt + continuation max tokens, cache will run out post this
        stop_token_id: int,
        tau: float,
        max_tokens_per_step: int = 1024,  # tokens in one step
        min_first_chunk: int = 64,  # a prompt's first chunk is >= this (or the whole prompt)
        # steps with prefill up to this many tokens replay a cuda graph. bigger ones keep the gpu
        # busy long enough that launching their kernels from python costs little
        max_graph_tokens: int = 512,
        device: str = "cuda",
        seed: int = 0,
    ):
        assert (
            max_tokens_per_step >= max_batch_size + min_first_chunk
        ), "budget must fit every decoding request plus one first chunk, or prefill can stall forever"
        self.model = model.to(device)
        self.max_batch_size = max_batch_size
        self.max_len = max_len
        self.stop_token_id = stop_token_id
        self.max_tokens_per_step = max_tokens_per_step
        self.min_first_chunk = min_first_chunk
        self.device = device
        # one extra cache row that no request ever owns. padded rows in a graph replay point here,
        # so the kv they write lands somewhere harmless
        self.scratch_slot = max_batch_size
        self.currently_processing_list: list[Request] = (
            []
        )  # every request holding a slot: prefilling or decoding, until it finishes
        self.waiting_queue: deque[Request] = (
            deque()
        )  # submitted but no slot yet. admitted first come first serve when a slot and budget free up
        self.kv_cache = [self.make_layer_cache() for _ in self.model.model.layers]
        self.free_slots: list[int] = list(
            range(max_batch_size)
        )  # which cache slots are now free? initially, every slot is free
        self.tau = tau
        self.seed = torch.tensor([seed], device=device, dtype=torch.int64)
        self.id = 0

        # a step with prefill is padded to fixed shapes (see fill_inputs): num_entries sequence
        # entries, enough for every slot plus one padding sequence, and a bucketed token count.
        # buckets stop at max_len, so the padding sequence always fits in the scratch slot's row
        self.num_entries = max_batch_size + 1
        self.token_buckets = self.make_token_buckets(
            min(max_graph_tokens, max_tokens_per_step, max_len)
        )
        # all of that step's inputs live in one int32 buffer, so they reach the gpu in one copy
        capacity = sum(self.input_layout(max_tokens_per_step))
        self.host_inputs = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
        self.dev_inputs = torch.empty(capacity, dtype=torch.int32, device=device)
        # where every layer dequantizes the prefilling sequences' cached prefixes. one fixed buffer
        # instead of one per step. a step whose prefixes don't fit gets a one-off buffer
        attn: AttentionLayer = self.model.model.layers[0].self_attn  # type: ignore
        self.prefix_capacity = 2 * max_len
        self.prefix_k = torch.empty(
            attn.num_kv_heads,
            self.prefix_capacity,
            attn.head_dim,
            device=device,
            dtype=self.model.dtype,
        )
        self.prefix_v = torch.empty_like(self.prefix_k)

        # every graph shares one memory pool, so they reuse each other's scratch memory. fine
        # because only one ever runs at a time
        self.graph_pool = torch.cuda.graph_pool_handle()
        # pure decode steps, by batch size
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self.capture_decode_graphs()
        # steps with prefill, by token bucket
        self.prefill_graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self.capture_prefill_graphs()

    def submit_request(self, prompt_ids: list[int], max_new_tokens: int) -> Request:
        assert max_new_tokens >= 1 and len(prompt_ids) > 0
        assert (
            len(prompt_ids) + max_new_tokens <= self.max_len
        ), "please make sure your request is below max_len"
        request = Request(
            id=self.id,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            slot=None,
            cached_len=0,
            output=[],
            t_submit=time.perf_counter(),
        )
        self.waiting_queue.append(request)
        self.id += 1
        return request

    def generate(self, prompts: list[list[int]], max_new_tokens: int) -> list[Request]:
        requests = [self.submit_request(p, max_new_tokens) for p in prompts]
        while self.waiting_queue or self.currently_processing_list:
            self.step()
        return requests

    @torch.inference_mode()
    def step(self) -> list[Request]:
        with nvtx.range("plan"):
            num_seqs_decoding = 0
            budget = self.max_tokens_per_step
            # plan for this singular step
            plan: list[tuple[Request, list]] = []

            for running_request in self.currently_processing_list:
                if running_request.cached_len >= len(running_request.prompt_ids):
                    # it's decoding now, don't starve these, and they come at the beginning for continuity
                    plan.append((running_request, [running_request.output[-1]]))
                    budget -= 1
                    num_seqs_decoding += 1
            for running_request in self.currently_processing_list:
                if (
                    running_request.cached_len < len(running_request.prompt_ids)
                    and budget > 0
                ):
                    # prefilling
                    n = min(
                        len(running_request.prompt_ids) - running_request.cached_len,
                        budget,
                    )
                    plan.append(
                        (
                            running_request,
                            running_request.prompt_ids[
                                running_request.cached_len : running_request.cached_len
                                + n
                            ],
                        )
                    )
                    budget -= n

            while self.waiting_queue and self.free_slots and budget > 0:
                new_request_to_admit = self.waiting_queue[0]
                n = min(len(new_request_to_admit.prompt_ids), budget)
                if n < min(len(new_request_to_admit.prompt_ids), self.min_first_chunk):
                    # break not continue for first come first serve
                    break
                req = self.waiting_queue.popleft()
                slot_to_alloc = self.free_slots.pop()
                req.slot = slot_to_alloc
                self.currently_processing_list.append(req)
                plan.append((req, req.prompt_ids[:n]))
                budget -= n

            if not plan:
                return []

            decode_only = all(
                len(toks) == 1 and req.cached_len >= len(req.prompt_ids)
                for req, toks in plan
            )
        if self.graphs and decode_only:
            tokens = self.replay_decode_graph(plan)
        else:
            n_tokens = self.pick_bucket(sum(len(toks) for _, toks in plan))
            with nvtx.range("inputs"):
                prefix_tokens = self.fill_inputs(plan, num_seqs_decoding, n_tokens)
            graph = self.prefill_graphs.get(n_tokens)
            if graph is not None and prefix_tokens <= self.prefix_capacity:
                with nvtx.range("forward"):
                    graph[0].replay()  # runs on whatever fill_inputs just wrote
                sampled = graph[1]
            else:
                # no graph for this many tokens, or the cached prefixes don't fit the fixed buffer
                with nvtx.range("forward"):
                    logits = self.model(
                        kv_cache=self.kv_cache,
                        **self.input_views(n_tokens, prefix_tokens),
                    )
                with nvtx.range("sample"):
                    sampled = sample(logits, self.tau, self.seed)
            with nvtx.range("readback"):
                # a token for every entry, padding and unused ones included. the zip in
                # bookkeeping only reads the first len(plan)
                tokens: list[int] = sampled.view(-1).tolist()

        with nvtx.range("bookkeeping"):
            now = time.perf_counter()
            finished: list[Request] = []
            for (req, toks), tok in zip(plan, tokens):
                req.cached_len += len(toks)
                if req.cached_len >= len(req.prompt_ids):
                    req.output.append(tok)
                    if len(req.output) == 1:
                        req.t_first_token = now
                    if (
                        tok == self.stop_token_id
                        or len(req.output) >= req.max_new_tokens
                        or req.cached_len + 1 >= self.max_len
                    ):
                        self.free_slots.append(req.slot)  # type: ignore
                        self.currently_processing_list.remove(req)
                        req.slot = None
                        req.t_finish = now
                        finished.append(req)
            return finished

    def make_token_buckets(self, M: int) -> list[int]:
        if M <= 0:
            return []
        return sorted(
            {b for b in (16, 32, 64) if b < M} | set(range(128, M, 128)) | {M}
        )

    def pick_bucket(self, N: int) -> int:
        return next((b for b in self.token_buckets if b >= N), N)

    def input_layout(self, n_tokens: int) -> list[int]:
        # how long each field of dev_inputs is for a step of n_tokens (padded) tokens, in order:
        # ids, positions, seq_ids (one per token), cu_seqlens, cached_lens, slot_ids, cu_cached
        # (one or two per entry), tile_start, num_decoding. the same n_tokens always gives the
        # same layout, so a bucket's fields always sit at the same addresses
        S = self.num_entries
        max_tiles = math.ceil(n_tokens / self.model.model.prefill_block_m) + S
        return [n_tokens, n_tokens, n_tokens, S + 1, S, S, S + 1, max_tiles, 1]

    def input_views(self, n_tokens: int, prefix_tokens: int = 0) -> dict:
        # the model's inputs for a step of n_tokens tokens, as views into dev_inputs
        (
            ids,
            positions,
            seq_ids,
            cu_seqlens,
            cached_lens,
            slot_ids,
            cu_cached,
            tile_start,
            num_decoding,
        ) = self.dev_inputs[: sum(self.input_layout(n_tokens))].split(
            self.input_layout(n_tokens)
        )
        prefix_k, prefix_v = self.prefix_k, self.prefix_v
        if prefix_tokens > self.prefix_capacity:  # a one-off buffer for this step
            Hkv, _, D = self.prefix_k.shape
            prefix_k = torch.empty(
                Hkv, prefix_tokens, D, device=self.device, dtype=self.prefix_k.dtype
            )
            prefix_v = torch.empty_like(prefix_k)
        return dict(
            x=ids,
            num_decoding=num_decoding,
            cached_lens=cached_lens,
            slot_ids=slot_ids,
            positions=positions,
            cu_seqlens=cu_seqlens,
            seq_ids=seq_ids,
            tile_start=tile_start,
            prefix_scratch=PrefixScratch(prefix_k, prefix_v, cu_cached),
        )

    def fill_inputs(
        self, plan: list[tuple[Request, list]], num_seqs_decoding: int, N_pad: int
    ) -> int:  # -> how many cached prefix tokens the step dequantizes
        # writes every model input of a step with prefill into dev_inputs: built on the host from
        # the plan (so the model never reads anything back), padded to fixed shapes, one copy.
        # entries: the plan's requests, decoding ones first, then one padding sequence on the
        # scratch slot that tops the tokens up to N_pad, then unused entries with no tokens.
        # e.g. A and B decoding, C prefilling 5 tokens, bucket 16, 5 entries:
        #   chunk_lens [1, 1, 5, 9, 0]  slot_ids [A, B, C, scratch, scratch]
        S, B = self.num_entries, len(plan)
        N = sum(len(toks) for _, toks in plan)
        block_m = self.model.model.prefill_block_m

        chunk_lens = [len(toks) for _, toks in plan] + [N_pad - N] + [0] * (S - B - 1)
        cached_lens = [r.cached_len for r, _ in plan] + [0] * (S - B)
        slot_ids = [r.slot for r, _ in plan] + [self.scratch_slot] * (S - B)
        ids = [tok for _, toks in plan for tok in toks] + [0] * (N_pad - N)
        cu_seqlens = [0, *itertools.accumulate(chunk_lens)]
        positions: list[int] = []
        seq_ids: list[int] = []
        tile_start: list[int] = []
        for b, (start, L) in enumerate(zip(cu_seqlens, chunk_lens)):
            # a chunk continues after whatever is already cached
            positions += range(cached_lens[b], cached_lens[b] + L)
            seq_ids += [b] * L
            # first row of each prefill attention program. every entry starts on a fresh tile,
            # so no tile holds rows from two entries. decoding entries get none
            if b >= num_seqs_decoding:
                tile_start += range(start, start + L, block_m)
        # -1 = a program with no work, so the tile count is the same for every step in a bucket
        tile_start += [-1] * (math.ceil(N_pad / block_m) + S - len(tile_start))
        # each entry's cached prefix, laid end to end in the prefix scratch. decoders read the
        # int8 cache directly, so theirs are empty
        prefix_lens = [0] * num_seqs_decoding + cached_lens[num_seqs_decoding:]
        cu_cached = [0, *itertools.accumulate(prefix_lens)]

        fields = [
            ids,
            positions,
            seq_ids,
            cu_seqlens,
            cached_lens,
            slot_ids,
            cu_cached,
            tile_start,
            [num_seqs_decoding],
        ]
        assert [len(f) for f in fields] == self.input_layout(N_pad)
        flat = list(itertools.chain.from_iterable(fields))
        # the last step ended with a readback, so nothing still reads these buffers
        host, dev = self.host_inputs[: len(flat)], self.dev_inputs[: len(flat)]
        host.copy_(torch.tensor(flat, dtype=torch.int32))
        dev.copy_(host, non_blocking=True)
        return cu_cached[-1]

    @torch.inference_mode()
    def capture_decode_graphs(self) -> None:
        B = self.max_batch_size
        # inputs live at fixed addresses
        # the graph is captured on the slices [:b], which start at the same address
        self.graph_ids = torch.zeros(B, device=self.device, dtype=torch.int64)
        self.graph_cached_lens = torch.zeros(B, device=self.device, dtype=torch.int32)
        self.graph_slot_ids = torch.full(
            (B,), self.scratch_slot, device=self.device, dtype=torch.int32
        )
        self.graph_buckets = sorted({min(1 << i, B) for i in range(B.bit_length() + 1)})
        # every row of bucket b decodes (padded rows on the scratch slot), so its count is the
        # constant b. one tensor per bucket, made before capture since a graph can't create one
        # from a python number, and kept here because the graph reads it at a fixed address
        self.graph_num_decoding: dict[int, torch.Tensor] = {}

        for b in reversed(self.graph_buckets):
            ids, cached_lens, slot_ids = (
                self.graph_ids[:b],
                self.graph_cached_lens[:b],
                self.graph_slot_ids[:b],
            )
            num_decoding = torch.full((1,), b, device=self.device, dtype=torch.int32)
            self.graph_num_decoding[b] = num_decoding

            def run() -> torch.Tensor:
                logits = self.model(
                    ids,
                    self.kv_cache,
                    num_decoding=num_decoding,
                    cached_lens=cached_lens,
                    slot_ids=slot_ids,
                )
                return sample(logits, self.tau, self.seed)  # [b, 1]

            # every row points at the scratch slot with cached_len 0, so the kv the warmup
            # writes is harmless
            self.graphs[b] = self.capture_graph(run)

    @torch.inference_mode()
    def capture_prefill_graphs(self) -> None:
        # one graph per token bucket, for steps that carry prefill. a bucket's inputs always sit
        # at the same addresses (its views into dev_inputs), and no kernel's grid depends on what's
        # in them, so replaying it runs whatever fill_inputs wrote: any mix of decoding and
        # prefilling entries that fits in the bucket
        for n in reversed(self.token_buckets):
            # a dummy step: no requests, just a padding sequence of n tokens on the scratch slot,
            # so the kv the warmup writes is harmless
            self.fill_inputs([], 0, n)
            inputs = self.input_views(n)

            def run() -> torch.Tensor:
                logits = self.model(kv_cache=self.kv_cache, **inputs)
                return sample(logits, self.tau, self.seed)  # [num_entries, 1]

            self.prefill_graphs[n] = self.capture_graph(run)

    def capture_graph(
        self, run: Callable[[], torch.Tensor]
    ) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
        # warmup runs for real so triton autotune (and tunableop) happen before capture. then
        # records run's gpu work, and the tensor it returns, which each replay overwrites
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        tunable.tuning_enable(tunable.is_enabled())
        with torch.cuda.stream(s):
            for _ in range(2):
                run()
        tunable.tuning_enable(False)
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool):
            out = run()
        return graph, out

    def replay_decode_graph(self, plan: list[tuple[Request, list]]) -> list[int]:
        B = len(plan)
        b = next(x for x in self.graph_buckets if x >= B)
        pad = b - B

        with nvtx.range("inputs"):
            ids = [toks[0] for _, toks in plan] + [0] * pad
            cached_lens = [req.cached_len for req, _ in plan] + [0] * pad
            slot_ids = [req.slot for req, _ in plan] + [self.scratch_slot] * pad

            self.graph_ids[:b].copy_(torch.tensor(ids, dtype=torch.int64))
            self.graph_cached_lens[:b].copy_(
                torch.tensor(cached_lens, dtype=torch.int32)
            )
            self.graph_slot_ids[:b].copy_(torch.tensor(slot_ids, dtype=torch.int32))
        with nvtx.range("forward"):
            graph, tokens = self.graphs[b]
            graph.replay()
        with nvtx.range("readback"):
            return tokens[:B].view(-1).tolist()

    def make_layer_cache(self) -> LayerKVCache:
        # one kv cache row per slot, each long enough for a whole request, plus the scratch row
        attn: AttentionLayer = self.model.model.layers[0].self_attn  # type: ignore
        pool, Hkv, D = self.max_batch_size + 1, attn.num_kv_heads, attn.head_dim
        return LayerKVCache(
            k=torch.zeros(
                pool, Hkv, self.max_len, D, device=self.device, dtype=torch.int8
            ),
            v=torch.zeros(
                pool, Hkv, self.max_len, D, device=self.device, dtype=torch.int8
            ),
            k_scale=torch.zeros(
                pool, Hkv, self.max_len, device=self.device, dtype=torch.float32
            ),
            v_scale=torch.zeros(
                pool, Hkv, self.max_len, device=self.device, dtype=torch.float32
            ),
            k_calibration=torch.ones(
                pool, Hkv, 1, D, device=self.device, dtype=torch.float32
            ),
        )
