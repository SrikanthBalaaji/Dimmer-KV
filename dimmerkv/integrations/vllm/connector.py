"""vLLM v1 KV connector: compressed storage of chat history between turns, and of old conversations.

When a request finishes, the KV of every token it computed (prompt and
answer) beyond what the store already holds is compressed with DimmerKV
(~3-5% of FP16) and appended to the conversation's segment chain on disk.
When a later prompt starts with a stored conversation (the next chat turn, a
returning user, a shared system prompt or document) and vLLM's own GPU
prefix cache no longer has it, the longest stored prefix is decoded straight
into the request's paged blocks, so only the new tokens are prefilled.

Enable with::

    vllm serve <model> --kv-transfer-config '{
        "kv_connector": "DimmerKVConnector",
        "kv_connector_module_path": "dimmerkv.integrations.vllm.connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"checkpoint": "/path/calibration.pt",
                                      "store_path": "/var/lib/dimmerkv"}}'

(``kv_connector_module_path`` is optional once the package is installed: its
``vllm.general_plugins`` entry point registers the name.)

``kv_connector_extra_config`` keys: ``checkpoint`` (required), ``store_path``
(default ``~/.cache/dimmerkv``), ``budget`` (0.03 = 3% of FP16),
``recent`` (32 exact tokens per segment), ``floor_share`` (1.0),
``min_tokens`` (256: shorter conversations are not stored), ``ram_cache_mb``
(512, worker-side LRU of compressed blobs), ``max_disk_gb`` (no limit),
``max_rung_fraction`` (drop finer rungs to shrink the basis on the GPU),
``codec_device`` (default: the KV cache's device), ``save`` / ``load``
(default from ``kv_role``: producer saves, consumer loads, both does both).

Limits: one KV cache group of full attention (no sliding window / hybrid
models), tensor and pipeline parallel size 1 (the basis spans all heads), no
FP8 KV cache, text-only requests. Restored KV is lossy and also enters vLLM's
GPU prefix cache like any loaded block.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

from ...codec import DimmerKVCodec, read_fingerprint
from ...engine import CompressionSettings, DimmerKV
from ...store import ConversationStore, Match
from .paged import gather, geometry, ordered_layers, scatter, slot_mapping

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class LoadSpec:
    req_id: str
    segment: str
    tokens: int  # Decode the conversation's first ``tokens`` tokens ...
    start: int  # ... and write positions [start, tokens) (vLLM computed the rest itself).
    slots: torch.Tensor


@dataclass
class SaveSpec:
    req_id: str
    token_ids: list[int]  # The conversation up to the segment's end.
    start: int  # The segment covers [start, len(token_ids)).
    parent: str | None
    namespace: str | None
    slots: torch.Tensor


@dataclass
class DimmerKVConnectorMetadata(KVConnectorMetadata):
    loads: list[LoadSpec] = field(default_factory=list)
    saves: list[SaveSpec] = field(default_factory=list)


def _namespace(request: "Request") -> tuple[bool, str | None]:
    """(eligible, namespace): LoRA adapters change the KV, multimodal / embedding prompts are skipped."""
    if getattr(request, "mm_features", None) or getattr(request, "prompt_embeds", None) is not None:
        return False, None
    lora = getattr(request, "lora_request", None)
    return True, (f"lora:{lora.lora_name}" if lora is not None else None)


class DimmerKVConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig | None" = None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        extra = dict(self._kv_transfer_config.kv_connector_extra_config or {})
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise NotImplementedError("DimmerKVConnector needs tensor_parallel_size = pipeline_parallel_size = 1: "
                                      "its PCA basis spans every layer and KV head.")
        if str(vllm_config.cache_config.cache_dtype).startswith("fp8"):
            raise NotImplementedError("DimmerKVConnector does not support FP8 KV caches.")
        if "checkpoint" not in extra:
            raise ValueError("Set kv_connector_extra_config.checkpoint to a `dimmerkv calibrate` file.")
        self._extra = extra
        self._block_size = vllm_config.cache_config.block_size
        kv_role = self._kv_transfer_config.kv_role
        self._save_enabled = bool(extra.get("save", kv_role in ("kv_producer", "kv_both")))
        self._load_enabled = bool(extra.get("load", kv_role in ("kv_consumer", "kv_both")))
        self._min_tokens = int(extra.get("min_tokens", 256))
        max_disk = extra.get("max_disk_gb")
        ram = int(extra.get("ram_cache_mb", 512)) << 20 if role == KVConnectorRole.WORKER else 0
        self._store = ConversationStore(
            Path(extra.get("store_path", "~/.cache/dimmerkv")).expanduser(), read_fingerprint(extra["checkpoint"]),
            block_size=16, ram_bytes=ram, max_disk_bytes=None if max_disk is None else int(float(max_disk) * 2**30))

        # Scheduler side.
        self._matches: dict[str, Match | None] = {}
        self._pending_loads: dict[str, tuple[Match, int]] = {}
        self._pending_saves: list[SaveSpec] = []
        # Worker side.
        self._layers: list[torch.Tensor] = []
        self._kv: DimmerKV | None = None
        self._failed_blocks: set[int] = set()
        self._saved: set[str] = set()
        self._finished_seen: set[str] = set()
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stream = None

    # ==================== scheduler side ====================
    def _lookup(self, request: "Request") -> Match | None:
        if request.request_id not in self._matches:
            eligible, namespace = _namespace(request)
            prompt = list(request.prompt_token_ids or [])
            match = None
            if eligible and len(prompt) > 1:
                match = self._store.longest_prefix(prompt, len(prompt) - 1, self._store.namespaced(namespace))
            if match is not None:
                # vLLM loads whole blocks; the store indexes every 16 tokens.
                match = Match(match.segment, match.tokens // self._block_size * self._block_size)
            self._matches[request.request_id] = match if match and match.tokens else None
        return self._matches[request.request_id]

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int | None, bool]:
        if not self._load_enabled:
            return 0, False
        match = self._lookup(request)
        if match is None or match.tokens <= num_computed_tokens:
            return 0, False
        return match.tokens - num_computed_tokens, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        if num_external_tokens > 0:
            match = self._matches[request.request_id]
            self._pending_loads[request.request_id] = (match, match.tokens - num_external_tokens)

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> KVConnectorMetadata:
        meta = DimmerKVConnectorMetadata()

        def add_load(req_id: str, block_ids: list[int]) -> None:
            match, start = self._pending_loads.pop(req_id)
            meta.loads.append(LoadSpec(req_id, match.segment, match.tokens, start,
                                       slot_mapping(block_ids, start, match.tokens, self._block_size)))

        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self._pending_loads:
                add_load(new_req.req_id, new_req.block_ids[0])
        cached = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached.req_ids):
            # A request resumed after preemption gets all of its block ids here.
            if req_id in self._pending_loads and req_id in cached.resumed_req_ids and cached.new_block_ids[index]:
                add_load(req_id, cached.new_block_ids[index][0])
        if self._pending_loads:
            logger.warning("DimmerKV: %d planned loads were not scheduled this step.", len(self._pending_loads))
        meta.saves, self._pending_saves = self._pending_saves, []
        return meta

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        self._matches.pop(request.request_id, None)
        self._pending_loads.pop(request.request_id, None)
        eligible, namespace = _namespace(request)
        if not (self._save_enabled and eligible):
            return False, None
        computed = request.num_computed_tokens - getattr(request, "num_in_flight_tokens", 0)
        computed = min(computed, len(block_ids) * self._block_size, len(request.all_token_ids))
        end = computed // self._store.block_size * self._store.block_size
        if end < self._min_tokens:
            return False, None
        token_ids = list(request.all_token_ids)[:end]
        match = self._store.longest_prefix(token_ids, end, self._store.namespaced(namespace))
        start = match.tokens if match else 0
        if start >= end:
            return False, None
        self._pending_saves.append(SaveSpec(request.request_id, token_ids, start, match.segment if match else None,
                                            namespace, slot_mapping(block_ids, start, end, self._block_size)))
        # Keep the blocks until the worker has copied them out (reported via get_finished).
        return True, None

    # ==================== worker side ====================
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._layers = ordered_layers(kv_caches)
        heads, head_size = geometry(self._layers[0])
        device = self._extra.get("codec_device") or self._layers[0].device
        max_rung = self._extra.get("max_rung_fraction")
        codec = DimmerKVCodec(self._extra["checkpoint"], device=device,
                              max_rung_fraction=None if max_rung is None else float(max_rung))
        if (codec.layers, codec.heads, codec.head_dim) != (len(self._layers), heads, head_size):
            raise ValueError(f"Calibration {codec.model_id} has {codec.layers} layers x {codec.heads} KV heads x "
                             f"{codec.head_dim}, the served model {len(self._layers)} x {heads} x {head_size}.")
        settings = CompressionSettings(budget=float(self._extra.get("budget", 0.03)),
                                       recent=int(self._extra.get("recent", 32)),
                                       floor_share=float(self._extra.get("floor_share", 1.0)))
        self._kv = DimmerKV(codec, self._store, settings)
        if self._layers[0].device.type == "cuda":
            self._stream = torch.cuda.Stream(device=self._layers[0].device)
        self._thread = threading.Thread(target=self._save_loop, name="dimmerkv-save", daemon=True)
        self._thread.start()
        logger.info("DimmerKV: %s, budget %.1f%%, store %s", codec.model_id, 100 * settings.budget, self._store.root)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        if not self.has_connector_metadata():
            return
        meta = self._get_connector_metadata()
        assert isinstance(meta, DimmerKVConnectorMetadata)
        for load in meta.loads:
            try:
                keys, values = self._kv.load(Match(load.segment, load.tokens), dtype=self._layers[0].dtype,
                                             device=self._layers[0].device)
                scatter(self._layers, keys[:, :, load.start :], values[:, :, load.start :], load.slots,
                        self._block_size)
            except Exception:
                logger.exception("DimmerKV: restoring %s failed; vLLM will recompute it.", load.req_id)
                self._failed_blocks |= set((load.slots // self._block_size).tolist())

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata",
                      **kwargs: Any) -> None:
        return  # Saves happen once the request is finished, from all layers at once (wait_for_save).

    def wait_for_save(self):
        if not self.has_connector_metadata():
            return
        meta = self._get_connector_metadata()
        assert isinstance(meta, DimmerKVConnectorMetadata)
        for save in meta.saves:
            try:
                # Copy out now (stream-ordered before any reuse of the blocks); compress in the background.
                keys, values = gather(self._layers, save.slots, self._block_size)
                event = None
                if self._stream is not None:
                    event = torch.cuda.Event()
                    event.record()
                    keys.record_stream(self._stream)
                    values.record_stream(self._stream)
                self._queue.put((save, keys, values, event))
            except Exception:
                logger.exception("DimmerKV: copying %s out failed; not stored.", save.req_id)
            with self._lock:
                self._saved.add(save.req_id)

    def _save_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            save, keys, values, event = item
            try:
                if self._stream is not None:
                    with torch.cuda.stream(self._stream):
                        self._stream.wait_event(event)
                        self._kv.save(save.token_ids, keys, values, save.start, save.parent, save.namespace)
                else:
                    self._kv.save(save.token_ids, keys, values, save.start, save.parent, save.namespace)
            except Exception:
                logger.exception("DimmerKV: compressing %s failed; not stored.", save.req_id)
            finally:
                del keys, values
                self._queue.task_done()

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        with self._lock:
            self._finished_seen |= finished_req_ids
            done = self._saved & self._finished_seen
            self._saved -= done
            self._finished_seen -= done
        return (done or None), None

    def get_block_ids_with_load_errors(self) -> set[int]:
        failed, self._failed_blocks = self._failed_blocks, set()
        return failed

    def shutdown(self):
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=60)
