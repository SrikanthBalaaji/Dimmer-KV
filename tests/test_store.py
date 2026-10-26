from __future__ import annotations

import torch

from dimmerkv.engine import DimmerKV
from dimmerkv.store import ConversationStore, Match

NS = b"n" * 16


def test_prefix_lookup_and_branches(tmp_path):
    store = ConversationStore(tmp_path, NS, block_size=4)
    turn1 = list(range(16))
    first = store.put(turn1, 0, b"one", None)
    turn2 = turn1 + list(range(100, 108))
    second = store.put(turn2, 16, b"two", first)

    assert store.longest_prefix(turn2 + [7, 7, 7]) == Match(second, 24)
    assert store.longest_prefix(turn2, max_tokens=23) == Match(second, 20)  # Block ending at 20 lives in second.
    branch = turn1 + [9, 9, 9, 9, 1]  # Edited second turn: only turn 1 is shared.
    assert store.longest_prefix(branch) == Match(first, 16)
    assert store.longest_prefix([5] + turn1) is None
    assert store.longest_prefix(turn1, namespace=store.namespaced("lora-a")) is None

    pieces = store.pieces(Match(second, 20))
    assert [(blob, start, stop) for blob, start, stop in pieces] == [(b"one", 0, 16), (b"two", 16, 20)]
    assert store.stats()["conversations"] == 1 and store.stats()["segments"] == 2


def test_delete_cascades_and_eviction(tmp_path):
    store = ConversationStore(tmp_path, NS, block_size=4, ram_bytes=8)
    a = store.put(list(range(8)), 0, b"x" * 10, None)
    b = store.put(list(range(12)), 8, b"y" * 10, a)
    c = store.put([50] * 8, 0, b"z" * 10, None)
    store.get(c)  # c is used most recently.
    assert store.evict(15) == 20  # Drops b (leaf, older), then a.
    assert store.longest_prefix(list(range(12))) is None
    assert store.longest_prefix([50] * 8) == Match(c, 8)
    assert store.delete(c) == 10 and store.stats()["segments"] == 0


def test_shared_across_store_instances(tmp_path):
    """Two processes (vLLM scheduler and worker) see the same index."""
    writer = ConversationStore(tmp_path, NS, block_size=4)
    reader = ConversationStore(tmp_path, NS, block_size=4)
    segment = writer.put(list(range(8)), 0, b"blob", None)
    assert reader.longest_prefix(list(range(9))) == Match(segment, 8)
    assert reader.get(segment) == b"blob"


def test_engine_turns_append_segments(checkpoint, synthetic, tmp_path):
    kv = DimmerKV.open(checkpoint, tmp_path, block_size=16, budget=0.2)
    keys, values = synthetic.cache(700, seed=21)
    tokens = list(range(700))

    first = kv.save_conversation(tokens[:300], keys[:, :, :300], values[:, :, :300])
    assert kv.lookup(tokens[:300]) == Match(first, 288)
    # Next turn: only tokens 288.. are compressed, chained to the first segment.
    second = kv.save_conversation(tokens, keys, values)
    assert [s.start for s in kv.store.chain(second)] == [0, 288]
    assert kv.save_conversation(tokens, keys, values) is None  # Nothing new.

    match = kv.lookup(tokens + [1, 2, 3])
    assert match == Match(second, 688)
    k, v = kv.load(match)
    assert k.shape == (3, 2, 688, 32)
    error = (k.float() - keys[:, :, :688].float()).square().sum() / keys[:, :, :688].float().square().sum()
    assert error < 0.1
    # Each segment keeps its own recent window exact, capped at half its budget (0.5 * 0.2 * 288 = 28 tokens).
    assert torch.equal(v[:, :, 288 - 28 : 288], values[:, :, 288 - 28 : 288])
    assert not torch.equal(v[:, :, 288 - 29 : 288], values[:, :, 288 - 29 : 288])
