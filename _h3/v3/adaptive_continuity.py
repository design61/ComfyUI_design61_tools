"""Read-only Adaptive Continuity observation and decision trace for A8a.

The observer records the continuation contract that Production already chose.
It never changes context depth, transport, masks, tensors, conditioning, MODEL,
Seed, SIGMAS, physical grouping, or execution policy.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


ADAPTIVE_CONTINUITY_OBSERVER_VERSION = 1
EXECUTION_APPLIED = False

INITIAL_TRANSPORT = "initial"
REFERENCE_CONTEXT_V1 = "reference_context_v1"
MASKED_VIDEO_PREFIX_V1 = "masked_video_prefix_v1"
MASKED_AV_PREFIX_22_V1 = "masked_av_prefix_22_v1"
MASKED_AV_PREFIX_39_V1 = "masked_av_prefix_39_v1"

_KNOWN_TRANSPORTS = frozenset(
    (
        INITIAL_TRANSPORT,
        REFERENCE_CONTEXT_V1,
        MASKED_VIDEO_PREFIX_V1,
        MASKED_AV_PREFIX_22_V1,
        MASKED_AV_PREFIX_39_V1,
    )
)
_VALID_CONTEXT_FRAMES = frozenset((0, 5, 22, 39))
_PLANNER_CONTRACT = {
    "planner_version": ADAPTIVE_CONTINUITY_OBSERVER_VERSION,
    "mode": "read_only_observer",
    "execution_applied": EXECUTION_APPLIED,
    "confidence_scope": "observation_completeness_not_mitigation_efficacy",
    "issue_13_mitigation_claim": False,
    "mutations": [],
    "observed_fields": [
        "motion_score",
        "current_context_frames",
        "resolved_transport",
        "terminal_atomic",
        "boundary_index",
        "boundary_count",
    ],
}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


ADAPTIVE_CONTINUITY_PLANNER_HASH = _canonical_hash(_PLANNER_CONTRACT)


class AdaptiveContinuityObservationError(ValueError):
    """One diagnostic observation is malformed or internally inconsistent."""


def _recommendation(
    *,
    current_context_frames: int,
    resolved_transport: str,
    terminal_atomic: bool,
) -> str:
    if int(current_context_frames) == 0:
        return "no_continuation_action_initial_group"
    if bool(terminal_atomic):
        return "keep_fixed_terminal_merge_contract"
    if str(resolved_transport) == REFERENCE_CONTEXT_V1:
        return "keep_current_reference_context_contract"
    return "keep_current_masked_prefix_contract"


def make_adaptive_continuity_decision(
    *,
    physical_group: int,
    logical_chunks: tuple[int, ...],
    observed_motion_score: float,
    current_context_frames: int,
    resolved_transport: str,
    terminal_atomic: bool,
    boundary_index: int,
    boundary_count: int,
    reason: str,
    fallback_reason: str | None = None,
    reused: bool = False,
) -> dict[str, Any]:
    """Build one immutable, JSON-safe observation without retaining tensors."""

    group = int(physical_group)
    logical = tuple(int(value) for value in logical_chunks)
    context = int(current_context_frames)
    boundary = int(boundary_index)
    boundaries = int(boundary_count)
    transport = str(resolved_transport)
    motion = float(observed_motion_score)
    reason_text = str(reason).strip()

    if group < 1 or not logical or any(value < 1 for value in logical):
        raise AdaptiveContinuityObservationError(
            "physical group and logical chunks must be positive"
        )
    if tuple(sorted(set(logical))) != logical:
        raise AdaptiveContinuityObservationError(
            "logical chunks must be ordered and unique"
        )
    if context not in _VALID_CONTEXT_FRAMES:
        raise AdaptiveContinuityObservationError(
            f"unsupported observed context frame count: {context}"
        )
    if transport not in _KNOWN_TRANSPORTS:
        raise AdaptiveContinuityObservationError(
            f"unknown resolved continuation transport: {transport!r}"
        )
    if context == 0 and transport != INITIAL_TRANSPORT:
        raise AdaptiveContinuityObservationError(
            "zero-context observations must use the initial transport marker"
        )
    if context > 0 and transport == INITIAL_TRANSPORT:
        raise AdaptiveContinuityObservationError(
            "continuation observations require a resolved transport"
        )
    if boundary < 0 or boundaries < 0 or boundary > boundaries:
        raise AdaptiveContinuityObservationError(
            "boundary index/count are inconsistent"
        )
    if not math.isfinite(motion) or motion < 0.0:
        raise AdaptiveContinuityObservationError(
            "observed motion score must be finite and non-negative"
        )

    confidence = 1.0 if reason_text else 0.5
    payload: dict[str, Any] = {
        "planner_version": ADAPTIVE_CONTINUITY_OBSERVER_VERSION,
        "planner_hash": ADAPTIVE_CONTINUITY_PLANNER_HASH,
        "physical_group": group,
        "logical_chunks": list(logical),
        "observed_motion_score": motion,
        "current_context_frames": context,
        "resolved_transport": transport,
        "terminal_atomic": bool(terminal_atomic),
        "boundary_index": boundary,
        "boundary_count": boundaries,
        "recommendation": _recommendation(
            current_context_frames=context,
            resolved_transport=transport,
            terminal_atomic=bool(terminal_atomic),
        ),
        "confidence": confidence,
        "confidence_scope": "observation completeness only; not mitigation efficacy",
        "reason": reason_text or "selection reason unavailable",
        "fallback_reason": (
            str(fallback_reason).strip() if fallback_reason is not None else None
        ),
        "reused": bool(reused),
        "execution_applied": EXECUTION_APPLIED,
        "issue_13_mitigation_claim": False,
    }
    payload["decision_hash"] = _canonical_hash(payload)
    return payload


class AdaptiveContinuityObserver:
    """Collect fail-soft A8a observations in deterministic physical-group order."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._advisories: list[str] = []

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(record) for record in self._records)

    def observe(self, **kwargs: Any) -> dict[str, Any]:
        record = make_adaptive_continuity_decision(**kwargs)
        self._records.append(record)
        return record

    def note_observation_failure(self, exc: Exception) -> None:
        self._advisories.append(f"{type(exc).__name__}: {exc}")

    def replay_prefix(
        self,
        *,
        entries: list[dict[str, Any]],
        total_chunks: int,
        terminal_merge_enabled: bool,
        resolved_transport: str,
    ) -> None:
        """Reconstruct trace-only records from accepted CPU chunk metadata."""

        chunk_count = int(total_chunks)
        completed = len(entries)
        if chunk_count < 1 or completed > chunk_count:
            raise AdaptiveContinuityObservationError(
                "replayed prefix length is outside the configured sequence"
            )
        boundary_count = max(0, chunk_count - 1)
        normal_count = completed
        terminal_complete = False
        pair_start = max(0, chunk_count - 2)
        if bool(terminal_merge_enabled):
            if pair_start < completed < chunk_count:
                raise AdaptiveContinuityObservationError(
                    "terminal merged prefix contains only one logical half"
                )
            terminal_complete = completed == chunk_count
            normal_count = pair_start if terminal_complete else completed

        for index in range(normal_count):
            entry = entries[index]
            context = int(entry.get("context_frames", 0))
            self.observe(
                physical_group=index + 1,
                logical_chunks=(index + 1,),
                observed_motion_score=float(entry.get("motion_score", 0.0)),
                current_context_frames=context,
                resolved_transport=(
                    INITIAL_TRANSPORT if context == 0 else str(resolved_transport)
                ),
                terminal_atomic=False,
                boundary_index=index,
                boundary_count=boundary_count,
                reason="replayed from accepted chunk metadata",
                fallback_reason=None,
                reused=True,
            )

        if terminal_complete:
            entry = entries[pair_start]
            context = int(entry.get("context_frames", 0))
            self.observe(
                physical_group=pair_start + 1,
                logical_chunks=(pair_start + 1, chunk_count),
                observed_motion_score=float(entry.get("motion_score", 0.0)),
                current_context_frames=context,
                resolved_transport=(
                    INITIAL_TRANSPORT if context == 0 else str(resolved_transport)
                ),
                terminal_atomic=True,
                boundary_index=pair_start,
                boundary_count=boundary_count,
                reason="replayed from accepted Terminal Merge metadata",
                fallback_reason=None,
                reused=True,
            )

    def report_lines(self) -> list[str]:
        lines = [
            "Adaptive Continuity observer [A8a v1]: "
            f"planner_hash={ADAPTIVE_CONTINUITY_PLANNER_HASH}, "
            f"observed_groups={len(self._records)}, "
            "execution_applied=false; recommendation trace only, "
            "Issue #13 mitigation claim=none, Production continuity unchanged."
        ]
        for record in self._records:
            lines.append(
                "Adaptive Continuity decision [A8a v1]: "
                f"physical_group={record['physical_group']}, "
                f"logical_chunks={record['logical_chunks']}, "
                f"boundary={record['boundary_index']}/{record['boundary_count']}, "
                f"motion={record['observed_motion_score']:.6f}, "
                f"context={record['current_context_frames']}, "
                f"transport={record['resolved_transport']}, "
                f"terminal_atomic={str(record['terminal_atomic']).lower()}, "
                f"reused={str(record['reused']).lower()}, "
                f"recommendation={record['recommendation']}, "
                f"confidence={record['confidence']:.2f}, "
                f"reason={record['reason']}, "
                f"fallback_reason={record['fallback_reason'] or 'none'}, "
                f"decision_hash={record['decision_hash']}, "
                "execution_applied=false."
            )
        lines.extend(
            f"Adaptive Continuity advisory [A8a v1]: {item}; execution_applied=false."
            for item in self._advisories
        )
        return lines


def adaptive_observer_event_fail_soft(
    observer: Any,
    method: str,
    **kwargs: Any,
) -> Any:
    """Invoke one observer event without allowing diagnostics to stop Sampling."""

    if observer is None:
        return None
    try:
        return getattr(observer, str(method))(**kwargs)
    except Exception as exc:
        try:
            observer.note_observation_failure(exc)
        except Exception:
            pass
        return None


__all__ = [
    "ADAPTIVE_CONTINUITY_OBSERVER_VERSION",
    "ADAPTIVE_CONTINUITY_PLANNER_HASH",
    "AdaptiveContinuityObservationError",
    "AdaptiveContinuityObserver",
    "EXECUTION_APPLIED",
    "INITIAL_TRANSPORT",
    "adaptive_observer_event_fail_soft",
    "make_adaptive_continuity_decision",
]
