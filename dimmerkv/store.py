"""Compressed conversation store with a token-prefix index, shared by every engine.

A conversation is stored as a chain of append-only **segments**: each turn
compresses only the tokens it added (``[start, end)``) and points at the
segment holding the tokens before it. Earlier turns are never re-encoded, so
quality does not drift turn after turn and compression costs only the new
tokens.

Lookups are by token prefix, like a prefix cache: every ``block_size``
tokens a chained hash of all tokens so far is indexed, and a new prompt is
matched against the longest indexed prefix. Branches (an edited earlier
message) therefore still reuse the shared part.

Blobs live on disk (one file per segment) with an in-process LRU of recent
blobs in RAM; the index is SQLite (WAL), so several processes (vLLM's
scheduler and workers) can share one store directory. ``max_disk_bytes``
evicts least recently used conversations, dropping a segment together with
every segment built on it.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    id TEXT PRIMARY KEY, parent TEXT, start INTEGER NOT NULL, end INTEGER NOT NULL,
    bytes INTEGER NOT NULL, created REAL NOT NULL, last_used REAL NOT NULL);
CREATE INDEX IF NOT EXISTS segments_parent ON segments(parent);
CREATE TABLE IF NOT EXISTS prefixes (hash BLOB PRIMARY KEY, segment TEXT NOT NULL, tokens INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS prefixes_segment ON prefixes(segment);
"""
QUERY_CHUNK = 500  # Hashes per IN (...) query, below SQLite's variable limit.


@dataclass(frozen=True)
class Segment:
    id: str
    parent: str | None
    start: int
    end: int
    bytes: int


@dataclass(frozen=True)
class Match:
    """The longest stored prefix of a token sequence: ``tokens`` tokens, ending in ``segment``."""

    segment: str
    tokens: int


def prefix_hashes(token_ids, block_size: int, namespace: bytes, limit: int | None = None) -> list[bytes]:
    """Chained hash after every full block: ``hashes[i]`` covers ``token_ids[:(i + 1) * block_size]``."""
    ids = np.asarray(token_ids, dtype=np.int64)
    if limit is not None:
        ids = ids[:limit]
    hashes, previous = [], namespace
    for begin in range(0, len(ids) - block_size + 1, block_size):
        previous = hashlib.blake2b(previous + ids[begin : begin + block_size].tobytes(), digest_size=16).digest()
        hashes.append(previous)
    return hashes


class ConversationStore:
    """Prefix-indexed, segment-chained store of compressed KV blobs.

    ``namespace`` must identify everything that changes the KV values for the
    same tokens: the calibration fingerprint, and e.g. a LoRA adapter name.
    """

    def __init__(self, root: str | Path, namespace: bytes, block_size: int = 16, ram_bytes: int = 256 << 20,
                 max_disk_bytes: int | None = None) -> None:
        self.root = Path(root).expanduser()
        (self.root / "blobs").mkdir(parents=True, exist_ok=True)
        self.namespace, self.block_size = namespace, block_size
        self.ram_bytes, self.max_disk_bytes = ram_bytes, max_disk_bytes
        self._ram: OrderedDict[str, bytes] = OrderedDict()
        self._ram_used = 0
        self._lock = threading.Lock()
        self._local = threading.local()
        with self._db() as db:
            db.executescript(SCHEMA)

    # ----- plumbing -----
    def _db(self) -> sqlite3.Connection:
        db = getattr(self._local, "db", None)
        if db is None:
            db = sqlite3.connect(self.root / "index.sqlite", timeout=30, isolation_level=None)
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            self._local.db = db
        return db

    def _path(self, segment: str) -> Path:
        return self.root / "blobs" / f"{segment}.dkv"

    def _remember(self, segment: str, blob: bytes) -> None:
        with self._lock:
            if segment in self._ram or len(blob) > self.ram_bytes:
                return
            self._ram[segment] = blob
            self._ram_used += len(blob)
            while self._ram_used > self.ram_bytes:
                _, dropped = self._ram.popitem(last=False)
                self._ram_used -= len(dropped)

    def namespaced(self, extra: str | None) -> bytes:
        """Namespace for one adapter / tenant on top of the store's own."""
        return self.namespace if not extra else hashlib.blake2b(self.namespace + extra.encode(), digest_size=16).digest()

    # ----- lookups -----
    def longest_prefix(self, token_ids, max_tokens: int | None = None, namespace: bytes | None = None) -> Match | None:
        """Longest stored prefix of ``token_ids`` (at most ``max_tokens`` tokens), at block granularity."""
        hashes = prefix_hashes(token_ids, self.block_size, namespace or self.namespace, max_tokens)
        if not hashes:
            return None
        best: Match | None = None
        db = self._db()
        # Hashes are chained, so a hit at block i implies the prefix through i is stored: scan from the end.
        for stop in range(len(hashes), 0, -QUERY_CHUNK):
            chunk = hashes[max(0, stop - QUERY_CHUNK) : stop]
            rows = db.execute(f"SELECT segment, tokens FROM prefixes WHERE hash IN ({','.join('?' * len(chunk))})",
                              chunk).fetchall()
            if rows:
                segment, tokens = max(rows, key=lambda row: row[1])
                best = Match(segment, tokens)
                break
        return best

    def chain(self, segment: str) -> list[Segment]:
        """Segments from the conversation start to ``segment``, oldest first."""
        db, out = self._db(), []
        current: str | None = segment
        while current is not None:
            row = db.execute("SELECT id, parent, start, end, bytes FROM segments WHERE id = ?", (current,)).fetchone()
            if row is None:
                raise KeyError(f"Segment {current} is missing (evicted?).")
            out.append(Segment(*row))
            current = row[1]
        return out[::-1]

    def get(self, segment: str) -> bytes:
        with self._lock:
            blob = self._ram.get(segment)
            if blob is not None:
                self._ram.move_to_end(segment)
        if blob is None:
            blob = self._path(segment).read_bytes()
            self._remember(segment, blob)
        self._db().execute("UPDATE segments SET last_used = ? WHERE id = ?", (time.time(), segment))
        return blob

    def pieces(self, match: Match) -> list[tuple[bytes, int, int]]:
        """(blob, start, stop) per segment covering ``match.tokens`` tokens; decode each blob and keep
        tokens ``[start, stop)`` of the conversation (the blob itself begins at its segment's start)."""
        segments = self.chain(match.segment)
        out = []
        for index, segment in enumerate(segments):
            stop = segments[index + 1].start if index + 1 < len(segments) else match.tokens
            out.append((self.get(segment.id), segment.start, min(stop, segment.end)))
        return out

    # ----- writes -----
    def put(self, token_ids, start: int, blob: bytes, parent: str | None, namespace: bytes | None = None) -> str:
        """Store a segment covering ``token_ids[start:]`` (``token_ids`` is the whole conversation so far,
        block-aligned at the end), chained to ``parent`` which must cover ``token_ids[:start]``."""
        end = len(token_ids)
        if start % self.block_size or end % self.block_size:
            raise ValueError("Segments must start and end on block boundaries.")
        if (parent is None) != (start == 0):
            raise ValueError("A segment has a parent exactly when it does not start at token 0.")
        segment = uuid.uuid4().hex
        path = self._path(segment)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(blob)
        os.replace(temporary, path)
        hashes = prefix_hashes(token_ids, self.block_size, namespace or self.namespace)
        now = time.time()
        db = self._db()
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute("INSERT INTO segments VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (segment, parent, start, end, len(blob), now, now))
            first = start // self.block_size
            db.executemany("INSERT OR IGNORE INTO prefixes VALUES (?, ?, ?)",
                           [(h, segment, (first + i + 1) * self.block_size) for i, h in enumerate(hashes[first:])])
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            path.unlink(missing_ok=True)
            raise
        self._remember(segment, blob)
        if self.max_disk_bytes is not None:
            self.evict(self.max_disk_bytes)
        return segment

    def delete(self, segment: str) -> int:
        """Delete a segment and everything chained after it; returns bytes freed."""
        db, freed = self._db(), 0
        pending = [segment]
        while pending:
            current = pending.pop()
            pending += [row[0] for row in db.execute("SELECT id FROM segments WHERE parent = ?", (current,))]
            row = db.execute("SELECT bytes FROM segments WHERE id = ?", (current,)).fetchone()
            db.execute("DELETE FROM prefixes WHERE segment = ?", (current,))
            db.execute("DELETE FROM segments WHERE id = ?", (current,))
            self._path(current).unlink(missing_ok=True)
            with self._lock:
                dropped = self._ram.pop(current, None)
                if dropped is not None:
                    self._ram_used -= len(dropped)
            freed += row[0] if row else 0
        return freed

    def evict(self, max_bytes: int) -> int:
        """Drop least recently used conversation tails until the store fits ``max_bytes``."""
        db, freed = self._db(), 0
        total = db.execute("SELECT COALESCE(SUM(bytes), 0) FROM segments").fetchone()[0]
        while total - freed > max_bytes:
            # Leaves first (nothing chained after them), oldest use first.
            row = db.execute("SELECT id FROM segments s WHERE NOT EXISTS (SELECT 1 FROM segments c WHERE c.parent = s.id) "
                             "ORDER BY last_used LIMIT 1").fetchone()
            if row is None:
                break
            freed += self.delete(row[0])
        return freed

    def stats(self) -> dict:
        db = self._db()
        segments, stored, tokens = db.execute(
            "SELECT COUNT(*), COALESCE(SUM(bytes), 0), COALESCE(SUM(end - start), 0) FROM segments").fetchone()
        roots = db.execute("SELECT COUNT(*) FROM segments WHERE parent IS NULL").fetchone()[0]
        return {"conversations": roots, "segments": segments, "stored_bytes": stored, "tokens": tokens,
                "ram_cached_bytes": self._ram_used}
