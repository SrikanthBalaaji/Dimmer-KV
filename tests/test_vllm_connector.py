"""Drive DimmerKVConnector through two chat turns with stub vLLM modules and CPU paged caches.

This checks the connector's own logic (what is saved and loaded, slots,
block release, partial local hits) against the vLLM v1 connector API as read
from vLLM's source; it is not a substitute for running inside vLLM.
"""

from __future__ import annotations

import enum
import logging
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from dimmerkv.integrations.vllm.paged import gather, scatter, slot_mapping

BLOCK = 16


def install_stub_vllm(monkeypatch):
    class KVConnectorRole(enum.Enum):
        SCHEDULER = 0
        WORKER = 1

    class KVConnectorMetadata:
        pass

    class KVConnectorBase_V1:
        def __init__(self, vllm_config, role, kv_cache_config=None):
            self._kv_transfer_config = vllm_config.kv_transfer_config
            self._role = role
            self._connector_metadata = None

        def bind_connector_metadata(self, metadata):
            self._connector_metadata = metadata

        def clear_connector_metadata(self):
            self._connector_metadata = None

        def has_connector_metadata(self):
            return self._connector_metadata is not None

        def _get_connector_metadata(self):
            return self._connector_metadata

    base = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")
    base.KVConnectorBase_V1, base.KVConnectorMetadata, base.KVConnectorRole = (
        KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = logging.getLogger
    for name in ("vllm", "vllm.distributed", "vllm.distributed.kv_transfer",
                 "vllm.distributed.kv_transfer.kv_connector", "vllm.distributed.kv_transfer.kv_connector.v1"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)
    monkeypatch.delitem(sys.modules, "dimmerkv.integrations.vllm.connector", raising=False)
    from dimmerkv.integrations.vllm import connector

    return connector, KVConnectorRole


def request(req_id, prompt, output=(), computed=None):
    tokens = list(prompt) + list(output)
    return SimpleNamespace(request_id=req_id, prompt_token_ids=list(prompt), all_token_ids=tokens,
                           num_computed_tokens=len(tokens) - 1 if computed is None else computed,
                           mm_features=[], prompt_embeds=None, lora_request=None)


def step(scheduler, worker, new_reqs=(), finished=()):
    output = SimpleNamespace(scheduled_new_reqs=list(new_reqs),
                             scheduled_cached_reqs=SimpleNamespace(req_ids=[], resumed_req_ids=set(), new_block_ids=[]))
    worker.bind_connector_metadata(scheduler.build_connector_meta(output))
    worker.start_load_kv(None)
    worker.wait_for_save()
    done, _ = worker.get_finished(set(finished))
    worker.clear_connector_metadata()
    worker._queue.join()  # Background compression finished.
    return done or set()


@pytest.mark.parametrize("legacy_layout", [False, True])
def test_two_turns(monkeypatch, tmp_path, checkpoint, synthetic, legacy_layout):
    connector, Role = install_stub_vllm(monkeypatch)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role="kv_both", kv_connector_extra_config={
            "checkpoint": str(checkpoint), "store_path": str(tmp_path), "budget": 0.2, "min_tokens": 64}),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
        cache_config=SimpleNamespace(block_size=BLOCK, cache_dtype="auto"))
    scheduler = connector.DimmerKVConnector(config, Role.SCHEDULER)
    worker = connector.DimmerKVConnector(config, Role.WORKER)
    shape = (2, 80, BLOCK, 2, 32) if legacy_layout else (80, 2, BLOCK, 64)
    caches = {f"model.layers.{i}.self_attn.attn": torch.zeros(shape, dtype=torch.float16) for i in range(3)}
    worker.register_kv_caches(caches)
    layers = [caches[f"model.layers.{i}.self_attn.attn"] for i in range(3)]

    # Turn 1 runs normally: 400 prompt tokens + 50 generated; vLLM wrote their KV into blocks 0..28.
    keys, values = synthetic.cache(480, seed=42)
    tokens = list(range(1000, 1480))
    turn1_blocks = list(range(29))
    scatter(layers, keys[:, :, :449], values[:, :, :449], slot_mapping(turn1_blocks, 0, 449, BLOCK), BLOCK)
    first = request("A", tokens[:400], tokens[400:450])
    assert scheduler.get_num_new_matched_tokens(first, 0) == (0, False)
    assert scheduler.request_finished(first, turn1_blocks) == (True, None)  # Blocks held until copied out.
    assert step(scheduler, worker, finished={"A"}) == {"A"}
    assert worker._store.stats()["tokens"] == 448

    # Blocks are reused by others; turn 2 arrives after the GPU prefix cache lost the conversation.
    for cache in layers:
        cache.zero_()
    second = request("B", tokens[:480], computed=0)
    assert scheduler.get_num_new_matched_tokens(second, 0) == (448, False)
    turn2_blocks = list(range(40, 70))
    scheduler.update_state_after_alloc(second, None, 448)
    step(scheduler, worker, new_reqs=[SimpleNamespace(req_id="B", block_ids=(turn2_blocks,))])
    k, v = gather(layers, slot_mapping(turn2_blocks, 0, 448, BLOCK), BLOCK)
    error = float((k.float() - keys[:, :, :448].float()).square().sum() / keys[:, :, :448].float().square().sum())
    assert error < 0.1
    assert torch.equal(v[:, :, :4], values[:, :, :4])  # Sinks exact.

    # Turn 2 finishes with 30 more tokens: only tokens 448.. are compressed, chained to turn 1.
    scatter(layers, keys[:, :, 448:], values[:, :, 448:], slot_mapping(turn2_blocks, 448, 480, BLOCK), BLOCK)
    done = request("B", tokens[:480], [7] * 10, computed=489)
    assert scheduler.request_finished(done, turn2_blocks) == (True, None)
    assert step(scheduler, worker, finished={"B"}) == {"B"}
    assert worker._store.stats()["segments"] == 2

    # A third request whose first 160 tokens are still in vLLM's GPU prefix cache loads only the rest.
    for cache in layers:
        cache.zero_()
    third = request("C", tokens[:480] + [7] * 10 + [5] * 20, computed=0)
    assert scheduler.get_num_new_matched_tokens(third, 160) == (480 - 160, False)
    scheduler.update_state_after_alloc(third, None, 320)
    blocks = list(range(31))
    step(scheduler, worker, new_reqs=[SimpleNamespace(req_id="C", block_ids=(blocks,))])
    k, _ = gather(layers, slot_mapping(blocks, 0, 480, BLOCK), BLOCK)
    assert k[:, :, :160].abs().sum() == 0  # Left to vLLM's own (exact) blocks.
    assert k[:, :, 160:].abs().sum() > 0
    worker.shutdown()


def test_load_failure_reports_blocks(monkeypatch, tmp_path, checkpoint):
    connector, Role = install_stub_vllm(monkeypatch)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role="kv_both", kv_connector_extra_config={
            "checkpoint": str(checkpoint), "store_path": str(tmp_path)}),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
        cache_config=SimpleNamespace(block_size=BLOCK, cache_dtype="auto"))
    worker = connector.DimmerKVConnector(config, Role.WORKER)
    worker.register_kv_caches({f"model.layers.{i}.self_attn.attn": torch.zeros(8, 2, BLOCK, 64, dtype=torch.float16)
                               for i in range(3)})
    meta = connector.DimmerKVConnectorMetadata(loads=[connector.LoadSpec(
        "X", "missing", 32, 0, slot_mapping([3, 5], 0, 32, BLOCK))])
    worker.bind_connector_metadata(meta)
    worker.start_load_kv(None)
    assert worker.get_block_ids_with_load_errors() == {3, 5}
    worker.shutdown()


def test_rejects_tensor_parallel(monkeypatch, tmp_path, checkpoint):
    connector, Role = install_stub_vllm(monkeypatch)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role="kv_both", kv_connector_extra_config={"checkpoint": str(checkpoint)}),
        parallel_config=SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1),
        cache_config=SimpleNamespace(block_size=BLOCK, cache_dtype="auto"))
    with pytest.raises(NotImplementedError, match="tensor_parallel"):
        connector.DimmerKVConnector(config, Role.SCHEDULER)
