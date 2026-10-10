from architecture import Qwen3_1_7B, AttentionLayer, LayerKVCache
import os
import time
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
import torch.cuda.tunable as tunable
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
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self.capture_decode_graphs()

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
        budget = self.max_tokens_per_step
        # plan for this singular step
        plan: list[tuple[Request, list]] = []

        for running_request in self.currently_processing_list:
            if running_request.cached_len >= len(running_request.prompt_ids):
                # it's decoding now, don't starve these, and they come at the beginning for continuity
                plan.append((running_request, [running_request.output[-1]]))
                budget -= 1
        for running_request in self.currently_processing_list:
            if (
                running_request.cached_len < len(running_request.prompt_ids)
                and budget > 0
            ):
                # prefilling
                n = min(
                    len(running_request.prompt_ids) - running_request.cached_len, budget
                )
                plan.append(
                    (
                        running_request,
                        running_request.prompt_ids[
                            running_request.cached_len : running_request.cached_len + n
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
            ids = (
                torch.tensor([tok for _, toks in plan for tok in toks])
                .to(torch.int64)
                .to(self.device)
            )
            cu_seqlens = F.pad(
                torch.tensor([len(toks) for _, toks in plan])
                .cumsum(dim=0)
                .to(torch.int32)
                .to(self.device),
                (1, 0),
            )
            cached_lens = (
                torch.tensor([r.cached_len for r, _ in plan])
                .to(torch.int32)
                .to(self.device)
            )
            slot_ids = (
                torch.tensor([r.slot for r, _ in plan]).to(torch.int32).to(self.device)
            )
            logits = self.model(
                ids,
                self.kv_cache,
                cu_seqlens=cu_seqlens,
                cached_lens=cached_lens,
                slot_ids=slot_ids,
            )
            tokens: list[int] = sample(logits, self.tau, self.seed).view(-1).tolist()

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
        pool = (
            torch.cuda.graph_pool_handle()
        )  # shared, so buckets reuse each other's scratch memory

        for b in reversed(self.graph_buckets):
            ids, cached_lens, slot_ids = (
                self.graph_ids[:b],
                self.graph_cached_lens[:b],
                self.graph_slot_ids[:b],
            )

            def run() -> torch.Tensor:
                logits = self.model(
                    ids, self.kv_cache, cached_lens=cached_lens, slot_ids=slot_ids
                )
                return sample(logits, self.tau, self.seed)  # [b, 1]

            # warmup runs for real so triton autotune (and tunableop) happen before capture.
            # every row points at the scratch slot with cached_len 0, so the kv it writes is harmless
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            tunable.tuning_enable(tunable.is_enabled())
            with torch.cuda.stream(s):
                for _ in range(2):
                    run()
            tunable.tuning_enable(False)
            torch.cuda.current_stream().wait_stream(s)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                tokens = run()
            self.graphs[b] = (graph, tokens)

    def replay_decode_graph(self, plan: list[tuple[Request, list]]) -> list[int]:
        B = len(plan)
        b = next(x for x in self.graph_buckets if x >= B)
        pad = b - B

        ids = [toks[0] for _, toks in plan] + [0] * pad
        cached_lens = [req.cached_len for req, _ in plan] + [0] * pad
        slot_ids = [req.slot for req, _ in plan] + [self.scratch_slot] * pad

        self.graph_ids[:b].copy_(torch.tensor(ids, dtype=torch.int64))
        self.graph_cached_lens[:b].copy_(torch.tensor(cached_lens, dtype=torch.int32))
        self.graph_slot_ids[:b].copy_(torch.tensor(slot_ids, dtype=torch.int32))
        graph, tokens = self.graphs[b]
        graph.replay()
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
