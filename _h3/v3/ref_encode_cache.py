"""Process-local, VAE-scoped cache for V3.8 reference encode tensors.

Only detached, contiguous CPU tensor copies are retained.  The cache never
owns source images, video frames, audio waveforms, conditioning objects, or
sampling outputs.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import threading
from typing import Any, Callable
import weakref

import torch


REF_ENCODE_CACHE_VERSION = 1
DEFAULT_MAX_ENTRIES = 16
DEFAULT_MAX_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class RefEncodeCacheKey:
    kind: str
    preprocess_version: int
    identity: str
    cache_version: int = REF_ENCODE_CACHE_VERSION


@dataclass(frozen=True, slots=True)
class RefEncodeCacheEvent:
    kind: str
    action: str
    reason: str = ""
    payload_bytes: int = 0


@dataclass(slots=True)
class _CacheEntry:
    tensor: torch.Tensor
    payload_bytes: int
    last_used: int


CacheEventSink = Callable[[RefEncodeCacheEvent], None]


def make_ref_encode_cache_key(
    kind: str,
    preprocess_version: int,
    identity: str,
) -> RefEncodeCacheKey:
    return RefEncodeCacheKey(
        kind=str(kind),
        preprocess_version=int(preprocess_version),
        identity=str(identity),
    )


class RefEncodeCache:
    """Small process-local LRU split into weak VAE object namespaces."""

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        if int(max_entries) < 1:
            raise ValueError("max_entries must be positive")
        if int(max_bytes) < 1:
            raise ValueError("max_bytes must be positive")
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._lock = threading.RLock()
        self._namespaces: weakref.WeakKeyDictionary[
            Any, OrderedDict[RefEncodeCacheKey, _CacheEntry]
        ] = weakref.WeakKeyDictionary()
        self._clock = 0
        self._stats: Counter[str] = Counter()
        self._kind_stats: dict[str, Counter[str]] = {}

    @staticmethod
    def _vae_cacheable(vae: Any) -> bool:
        try:
            weakref.ref(vae)
            hash(vae)
        except TypeError:
            return False
        return True

    def supports_vae(self, vae: Any) -> bool:
        return self._vae_cacheable(vae)

    def _next_clock_locked(self) -> int:
        self._clock += 1
        return self._clock

    def _record_locked(
        self,
        *,
        key: RefEncodeCacheKey,
        action: str,
        reason: str = "",
        payload_bytes: int = 0,
    ) -> RefEncodeCacheEvent:
        self._stats[str(action)] += 1
        self._kind_stats.setdefault(key.kind, Counter())[str(action)] += 1
        return RefEncodeCacheEvent(
            kind=key.kind,
            action=str(action),
            reason=str(reason),
            payload_bytes=int(payload_bytes),
        )

    @staticmethod
    def _emit(event: RefEncodeCacheEvent, sink: CacheEventSink | None) -> None:
        if sink is not None:
            sink(event)

    def _resident_locked(self) -> tuple[int, int]:
        entries = 0
        payload_bytes = 0
        for namespace in list(self._namespaces.values()):
            entries += len(namespace)
            payload_bytes += sum(entry.payload_bytes for entry in namespace.values())
        return entries, payload_bytes

    def _evict_locked(self) -> None:
        while True:
            entries, payload_bytes = self._resident_locked()
            if entries <= self.max_entries and payload_bytes <= self.max_bytes:
                return
            oldest_namespace = None
            oldest_key = None
            oldest_clock = None
            for namespace in list(self._namespaces.values()):
                for key, entry in namespace.items():
                    if oldest_clock is None or entry.last_used < oldest_clock:
                        oldest_namespace = namespace
                        oldest_key = key
                        oldest_clock = entry.last_used
            if oldest_namespace is None or oldest_key is None:
                return
            oldest_namespace.pop(oldest_key, None)
            self._stats["evictions"] += 1

    def lookup(
        self,
        vae: Any,
        key: RefEncodeCacheKey,
        *,
        event_sink: CacheEventSink | None = None,
    ) -> torch.Tensor | None:
        if not self._vae_cacheable(vae):
            with self._lock:
                event = self._record_locked(
                    key=key,
                    action="bypass",
                    reason="vae_not_weakrefable_or_hashable",
                )
            self._emit(event, event_sink)
            return None

        master = None
        with self._lock:
            namespace = self._namespaces.get(vae)
            entry = namespace.get(key) if namespace is not None else None
            if entry is None:
                event = self._record_locked(key=key, action="miss")
            else:
                entry.last_used = self._next_clock_locked()
                namespace.move_to_end(key)
                master = entry.tensor
                event = self._record_locked(
                    key=key,
                    action="hit",
                    payload_bytes=entry.payload_bytes,
                )
        self._emit(event, event_sink)
        return None if master is None else master.clone()

    def store(
        self,
        vae: Any,
        key: RefEncodeCacheKey,
        tensor: Any,
        *,
        event_sink: CacheEventSink | None = None,
    ) -> bool:
        if not self._vae_cacheable(vae):
            with self._lock:
                event = self._record_locked(
                    key=key,
                    action="bypass",
                    reason="vae_not_weakrefable_or_hashable",
                )
            self._emit(event, event_sink)
            return False
        if not torch.is_tensor(tensor):
            with self._lock:
                event = self._record_locked(
                    key=key,
                    action="bypass",
                    reason="payload_not_tensor",
                )
            self._emit(event, event_sink)
            return False
        if tensor.device.type != "cpu":
            with self._lock:
                event = self._record_locked(
                    key=key,
                    action="bypass",
                    reason="payload_not_cpu",
                )
            self._emit(event, event_sink)
            return False
        payload_bytes = int(tensor.numel()) * int(tensor.element_size())
        if payload_bytes > self.max_bytes:
            with self._lock:
                event = self._record_locked(
                    key=key,
                    action="bypass",
                    reason="entry_exceeds_byte_limit",
                    payload_bytes=payload_bytes,
                )
            self._emit(event, event_sink)
            return False
        try:
            master = tensor.detach().contiguous().clone()
        except Exception:
            with self._lock:
                event = self._record_locked(
                    key=key,
                    action="bypass",
                    reason="payload_copy_failed",
                    payload_bytes=payload_bytes,
                )
            self._emit(event, event_sink)
            return False

        with self._lock:
            namespace = self._namespaces.get(vae)
            if namespace is None:
                namespace = OrderedDict()
                self._namespaces[vae] = namespace
            namespace[key] = _CacheEntry(
                tensor=master,
                payload_bytes=payload_bytes,
                last_used=self._next_clock_locked(),
            )
            namespace.move_to_end(key)
            self._evict_locked()
            event = self._record_locked(
                key=key,
                action="store",
                payload_bytes=payload_bytes,
            )
        self._emit(event, event_sink)
        return True

    def clear(self) -> None:
        with self._lock:
            self._namespaces = weakref.WeakKeyDictionary()
            self._clock = 0
            self._stats.clear()
            self._kind_stats.clear()

    def inspect(self) -> dict[str, Any]:
        with self._lock:
            entries, payload_bytes = self._resident_locked()
            return {
                "cache_version": REF_ENCODE_CACHE_VERSION,
                "resident_entries": entries,
                "resident_bytes": payload_bytes,
                "max_entries": self.max_entries,
                "max_bytes": self.max_bytes,
                "stats": dict(self._stats),
                "kind_stats": {
                    kind: dict(values)
                    for kind, values in sorted(self._kind_stats.items())
                },
                "vae_namespaces": len(self._namespaces),
            }


_REF_ENCODE_CACHE = RefEncodeCache()


def get_ref_encode_cache() -> RefEncodeCache:
    return _REF_ENCODE_CACHE


def clear_ref_encode_cache() -> None:
    _REF_ENCODE_CACHE.clear()


def inspect_ref_encode_cache() -> dict[str, Any]:
    return _REF_ENCODE_CACHE.inspect()


def format_ref_encode_cache_diagnostics(
    events: list[RefEncodeCacheEvent] | tuple[RefEncodeCacheEvent, ...],
) -> str:
    if not events:
        return ""
    labels = {
        "reference_image": "image",
        "video_guide": "video",
        "reference_audio": "audio",
    }
    by_kind: dict[str, Counter[str]] = {}
    totals: Counter[str] = Counter()
    for event in events:
        by_kind.setdefault(event.kind, Counter())[event.action] += 1
        totals[event.action] += 1
    components = []
    for kind in ("reference_image", "video_guide", "reference_audio"):
        values = by_kind.get(kind)
        if not values:
            continue
        components.append(
            f"{labels[kind]} hit={values['hit']}/miss={values['miss']}"
            f"/store={values['store']}/bypass={values['bypass']}"
        )
    state = inspect_ref_encode_cache()
    resident_mib = int(state["resident_bytes"]) / float(1024**2)
    limit_mib = int(state["max_bytes"]) / float(1024**2)
    components.append(
        "total "
        f"hits={totals['hit']}, misses={totals['miss']}, "
        f"stores={totals['store']}, bypasses={totals['bypass']}"
    )
    components.append(
        f"resident={resident_mib:.1f} MiB/{limit_mib:.1f} MiB "
        f"({int(state['resident_entries'])}/{int(state['max_entries'])} entries)"
    )
    return "Reference encode cache: " + "; ".join(components) + "."
