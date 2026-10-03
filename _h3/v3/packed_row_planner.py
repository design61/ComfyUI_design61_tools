"""Read-only A7a planning from actual ComfyUI H3 PackedLayout rows.

The planner observes layouts that ComfyUI Core already built for Sampling.  It
does not estimate VRAM, change execution policy, mutate a layout, or influence
the physical-group order.  Memory counters are point-in-time observations only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


PACKED_ROW_PLAN_VERSION = 1
MAX_CONCURRENT_PHYSICAL_GROUPS = 1

_VIDEO_CONDITION_KINDS = frozenset(("cond", "ref_img"))
_AUDIO_CONDITION_KINDS = frozenset(("cond_audio", "ref_audio"))


class PackedRowPlanningError(ValueError):
    """An observed object is not an internally consistent H3 PackedLayout."""


@dataclass(frozen=True)
class PackedRowBreakdown:
    seq_len: int
    text_rows: int
    condition_video_rows: int
    condition_audio_rows: int
    target_video_rows: int
    target_audio_rows: int
    unknown_rows: int
    segment_count: int
    segment_signature: tuple[tuple[int, int, str], ...]

    @property
    def condition_rows(self) -> int:
        return int(self.condition_video_rows + self.condition_audio_rows)

    @property
    def target_rows(self) -> int:
        return int(self.target_video_rows + self.target_audio_rows)

    @property
    def condition_to_target_ratio(self) -> float | None:
        if self.target_rows <= 0:
            return None
        return float(self.condition_rows / self.target_rows)


@dataclass(frozen=True)
class MemoryObservation:
    system_available_bytes: int | None = None
    cuda_allocated_bytes: int | None = None
    cuda_reserved_bytes: int | None = None
    core_visible_free_bytes: int | None = None


def analyze_packed_layout(layout: Any) -> PackedRowBreakdown:
    """Account actual rows without importing or recreating Core PackedLayout."""

    if layout is None:
        raise PackedRowPlanningError("Core PackedLayout is unavailable")
    try:
        seq_len = int(layout.seq_len)
        raw_segments = tuple(layout.segments)
    except Exception as exc:
        raise PackedRowPlanningError(
            "Core PackedLayout does not expose seq_len/segments"
        ) from exc
    if seq_len <= 0 or not raw_segments:
        raise PackedRowPlanningError("Core PackedLayout is empty")

    signature: list[tuple[int, int, str]] = []
    previous_stop = 0
    counts = {
        "text": 0,
        "condition_video": 0,
        "condition_audio": 0,
        "target_video": 0,
        "target_audio": 0,
        "unknown": 0,
    }
    target_video_segments = 0
    target_audio_segments = 0
    for item in raw_segments:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise PackedRowPlanningError("PackedLayout segment is not (start, stop, kind)")
        start, stop, kind = int(item[0]), int(item[1]), str(item[2])
        if start != previous_stop or stop <= start or stop > seq_len:
            raise PackedRowPlanningError(
                "PackedLayout segments are not a contiguous in-range partition"
            )
        rows = stop - start
        signature.append((start, stop, kind))
        if kind == "text":
            counts["text"] += rows
        elif kind in _VIDEO_CONDITION_KINDS:
            counts["condition_video"] += rows
        elif kind in _AUDIO_CONDITION_KINDS:
            counts["condition_audio"] += rows
        elif kind == "video":
            counts["target_video"] += rows
            target_video_segments += 1
        elif kind == "audio":
            counts["target_audio"] += rows
            target_audio_segments += 1
        else:
            counts["unknown"] += rows
        previous_stop = stop
    if previous_stop != seq_len:
        raise PackedRowPlanningError("PackedLayout segment rows do not sum to seq_len")
    if target_video_segments != 1 or target_audio_segments != 1:
        raise PackedRowPlanningError(
            "H3 PackedLayout must contain exactly one target Video and Audio segment"
        )
    return PackedRowBreakdown(
        seq_len=seq_len,
        text_rows=int(counts["text"]),
        condition_video_rows=int(counts["condition_video"]),
        condition_audio_rows=int(counts["condition_audio"]),
        target_video_rows=int(counts["target_video"]),
        target_audio_rows=int(counts["target_audio"]),
        unknown_rows=int(counts["unknown"]),
        segment_count=len(signature),
        segment_signature=tuple(signature),
    )


def read_memory_observation() -> MemoryObservation:
    """Read counters without CUDA synchronization, cache clearing, or peak reset."""

    system_available = None
    cuda_allocated = None
    cuda_reserved = None
    core_free = None
    try:
        import psutil

        system_available = int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            cuda_allocated = int(torch.cuda.memory_allocated())
            cuda_reserved = int(torch.cuda.memory_reserved())
    except Exception:
        pass
    try:
        import comfy.model_management as model_management

        device = model_management.get_torch_device()
        core_free = int(model_management.get_free_memory(device))
    except Exception:
        pass
    return MemoryObservation(
        system_available_bytes=system_available,
        cuda_allocated_bytes=cuda_allocated,
        cuda_reserved_bytes=cuda_reserved,
        core_visible_free_bytes=core_free,
    )


def _safe_memory_read(reader: Callable[[], MemoryObservation]) -> MemoryObservation:
    try:
        value = reader()
        return value if isinstance(value, MemoryObservation) else MemoryObservation()
    except Exception:
        return MemoryObservation()


class PackedRowPlanner:
    """Collect one immutable actual-layout plan record per sampled group."""

    def __init__(
        self,
        *,
        memory_reader: Callable[[], MemoryObservation] = read_memory_observation,
    ) -> None:
        self._memory_reader = memory_reader
        self._active: dict[str, Any] | None = None
        self._records: list[dict[str, Any]] = []
        self._advisories: list[str] = []

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._records)

    def begin_group(
        self,
        *,
        physical_group: int,
        logical_chunks: tuple[int, ...],
        terminal_atomic: bool,
    ) -> None:
        if self._active is not None:
            self._advisories.append(
                "a group began before the prior observation was finalized"
            )
            self.finish_group()
        self._active = {
            "physical_group": int(physical_group),
            "logical_chunks": tuple(int(value) for value in logical_chunks),
            "terminal_atomic": bool(terminal_atomic),
            "pre_memory": _safe_memory_read(self._memory_reader),
            "breakdown": None,
            "unavailable": None,
        }

    def observe_layout(self, layout: Any) -> None:
        if self._active is None:
            return
        try:
            observed = analyze_packed_layout(layout)
        except Exception as exc:
            if self._active["breakdown"] is None:
                self._active["unavailable"] = f"{type(exc).__name__}: {exc}"
            return
        current = self._active["breakdown"]
        if current is not None and current != observed:
            self._active["unavailable"] = (
                "PackedLayout topology changed within one physical group"
            )
            self._active["breakdown"] = None
            return
        if self._active["unavailable"] is None:
            self._active["breakdown"] = observed

    def note_observation_failure(self, exc: Exception) -> None:
        if self._active is not None and self._active["breakdown"] is None:
            self._active["unavailable"] = f"{type(exc).__name__}: {exc}"

    def finish_group(self) -> None:
        if self._active is None:
            return
        record = dict(self._active)
        record["post_memory"] = _safe_memory_read(self._memory_reader)
        if record["breakdown"] is None and record["unavailable"] is None:
            record["unavailable"] = "Core PackedLayout was not observed"
        self._records.append(record)
        self._active = None

    @staticmethod
    def _mib(value: int | None) -> str:
        return "n/a" if value is None else f"{int(value) / 1024**2:.1f} MiB"

    def report_lines(self) -> list[str]:
        if self._active is not None:
            self.finish_group()
        valid = [record for record in self._records if record["breakdown"] is not None]
        peak = None
        if valid:
            peak = max(valid, key=lambda record: record["breakdown"].seq_len)
        order = [int(record["physical_group"]) for record in self._records]
        lines = [
            "Packed-row plan [A7a v1]: "
            f"execution_order={order}, "
            f"max_concurrent_physical_groups={MAX_CONCURRENT_PHYSICAL_GROUPS}, "
            f"observed_groups={len(valid)}/{len(self._records)}, "
            f"peak_candidate={int(peak['physical_group']) if peak else 'unavailable'}, "
            f"peak_total_rows={peak['breakdown'].seq_len if peak else 'unavailable'}; "
            "actual Core rows only, VRAM/time prediction=not provided, "
            "execution policy=unchanged."
        ]
        for record in self._records:
            group = int(record["physical_group"])
            logical = list(record["logical_chunks"])
            breakdown = record["breakdown"]
            if breakdown is None:
                lines.append(
                    "Packed-row group [A7a v1]: "
                    f"physical_group={group}, logical_chunks={logical}, "
                    f"terminal_atomic={str(bool(record['terminal_atomic'])).lower()}, "
                    f"unavailable/advisory={record['unavailable']}."
                )
                continue
            pre = record["pre_memory"]
            post = record["post_memory"]
            ratio = breakdown.condition_to_target_ratio
            ratio_text = "n/a" if ratio is None else f"{ratio:.6f}"
            lines.append(
                "Packed-row group [A7a v1]: "
                f"physical_group={group}, logical_chunks={logical}, "
                f"terminal_atomic={str(bool(record['terminal_atomic'])).lower()}, "
                f"total={breakdown.seq_len}, text={breakdown.text_rows}, "
                f"condition_video={breakdown.condition_video_rows}, "
                f"condition_audio={breakdown.condition_audio_rows}, "
                f"target_video={breakdown.target_video_rows}, "
                f"target_audio={breakdown.target_audio_rows}, "
                f"unknown={breakdown.unknown_rows}, "
                f"condition_to_target_ratio={ratio_text}, "
                f"pre_system_available={self._mib(pre.system_available_bytes)}, "
                f"pre_CUDA_allocated={self._mib(pre.cuda_allocated_bytes)}, "
                f"pre_CUDA_reserved={self._mib(pre.cuda_reserved_bytes)}, "
                f"pre_Core_free={self._mib(pre.core_visible_free_bytes)}, "
                f"post_system_available={self._mib(post.system_available_bytes)}, "
                f"post_CUDA_allocated={self._mib(post.cuda_allocated_bytes)}, "
                f"post_CUDA_reserved={self._mib(post.cuda_reserved_bytes)}, "
                f"post_Core_free={self._mib(post.core_visible_free_bytes)}."
            )
        lines.extend(f"Packed-row advisory [A7a v1]: {item}." for item in self._advisories)
        return lines


def observe_layout_fail_soft(planner: Any, payload: Any) -> None:
    """Diagnostics must never replace a Sampling result."""

    if planner is None:
        return
    try:
        layout = payload.get("layout") if isinstance(payload, dict) else None
        planner.observe_layout(layout)
    except Exception as exc:
        try:
            planner.note_observation_failure(exc)
        except Exception:
            pass


def planner_event_fail_soft(planner: Any, method: str, **kwargs: Any) -> Any:
    """Invoke one lifecycle event without allowing diagnostics to stop Sampling."""

    if planner is None:
        return None
    try:
        return getattr(planner, str(method))(**kwargs)
    except Exception as exc:
        try:
            planner.note_observation_failure(exc)
        except Exception:
            pass
        return None


__all__ = [
    "MAX_CONCURRENT_PHYSICAL_GROUPS",
    "MemoryObservation",
    "PACKED_ROW_PLAN_VERSION",
    "PackedRowBreakdown",
    "PackedRowPlanner",
    "PackedRowPlanningError",
    "analyze_packed_layout",
    "observe_layout_fail_soft",
    "planner_event_fail_soft",
    "read_memory_observation",
]
