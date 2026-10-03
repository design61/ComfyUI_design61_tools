"""Pure execution-policy resolvers for V3.8 chunk review.

This module deliberately owns no ComfyUI, torch, filesystem, Run Storage, or
sampling behavior.  It turns already-validated storage facts into an execution
decision that later phases can apply to the existing Production engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..branch_provenance import TAKE_ACTION_CONTINUE, TAKE_ACTION_USE


GENERATION_MODE_FULL_RUN = "Full Run"
GENERATION_MODE_REVIEW = "Review Each Chunk"
GENERATION_MODE_OPTIONS = (
    GENERATION_MODE_FULL_RUN,
    GENERATION_MODE_REVIEW,
)

REVIEW_ACTION_CONTINUE = "Continue / Next"
REVIEW_ACTION_REGENERATE_CURRENT = "Regenerate Current"
REVIEW_ACTION_FINISH_REMAINING = "Finish Remaining"
REVIEW_ACTION_OPTIONS = (
    REVIEW_ACTION_CONTINUE,
    REVIEW_ACTION_REGENERATE_CURRENT,
    REVIEW_ACTION_FINISH_REMAINING,
)

RUN_STORAGE_OFF = "Off"
RUN_STORAGE_SAVE_AUTO_RESUME = "Save + Auto Resume"

REVISION_STATUS_IN_PROGRESS = "in_progress"
REVISION_STATUS_INTERRUPTED = "interrupted"
REVISION_STATUS_REVIEW_READY = "review_ready"
REVISION_STATUS_COMPLETE = "complete"
REVISION_STATUSES = (
    REVISION_STATUS_IN_PROGRESS,
    REVISION_STATUS_INTERRUPTED,
    REVISION_STATUS_REVIEW_READY,
    REVISION_STATUS_COMPLETE,
)

EXECUTION_MODE_FULL_RUN = "full_run"
EXECUTION_MODE_REVIEW_CONTINUE = "review_continue"
EXECUTION_MODE_REVIEW_REGENERATE = "review_regenerate_current"
EXECUTION_MODE_REVIEW_FINISH = "review_finish_remaining"
EXECUTION_MODE_REVIEW_USE_TAKE = "review_use_take"
EXECUTION_MODE_REVIEW_CONTINUE_FROM_TAKE = "review_continue_from_take"

NONCE_POLICY_INACTIVE = "inactive"
NONCE_POLICY_INHERIT = "inherit_branch"
NONCE_POLICY_ADVANCE = "advance_variation"
NONCE_POLICY_MANUAL = "manual"

REVIEW_PAUSE_REASON = "review_each_chunk"


class ReviewControlError(ValueError):
    """Raised when a requested review policy is structurally ambiguous."""


@dataclass(frozen=True)
class ReviewUnit:
    """One atomic physical sampling group expressed in logical chunk numbers."""

    start: int
    end: int
    physical_group: int

    def as_metadata(self) -> dict[str, int]:
        return {
            "start": self.start,
            "end": self.end,
            "physical_group": self.physical_group,
        }


@dataclass(frozen=True)
class ReviewExecution:
    """Side-effect-free decision consumed by later execution phases."""

    execution_mode: str
    requires_run_storage: bool
    effective_regenerate_from: int
    max_new_physical_groups: int | None
    next_review_unit_start: int | None
    next_review_unit_end: int | None
    next_review_physical_group: int | None
    finish_remaining: bool
    smart_regenerate: bool
    one_shot_reset_required: bool
    partial_review: bool
    projected_prefix_count: int
    nonce_policy: str
    branch_regenerate_from: int
    base_effective_nonce: int
    requested_effective_nonce: int | None
    status_hint: str


def _integer(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReviewControlError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ReviewControlError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _manual_regenerate_from(value: Any, *, configured_chunks: int) -> int:
    if value is None or value == "Auto" or value == 0:
        return 0
    if isinstance(value, str) and value.startswith("Chunk "):
        try:
            value = int(value[6:])
        except ValueError as exc:
            raise ReviewControlError(
                "manual_regenerate_from must be Auto or a valid chunk"
            ) from exc
    return _integer(
        value,
        name="manual_regenerate_from",
        minimum=1,
        maximum=configured_chunks,
    )


def _terminal_pair_start(
    *,
    configured_chunks: int,
    terminal_merge_enabled: bool,
    terminal_pair_start: Any,
) -> int | None:
    if type(terminal_merge_enabled) is not bool:
        raise ReviewControlError("terminal_merge_enabled must be a boolean")
    if not terminal_merge_enabled:
        return None
    if configured_chunks < 2:
        raise ReviewControlError(
            "Terminal Merge requires at least two configured chunks"
        )
    start = _integer(
        terminal_pair_start,
        name="terminal_pair_start",
        minimum=1,
        maximum=configured_chunks - 1,
    )
    if start != configured_chunks - 1:
        raise ReviewControlError(
            "Terminal Merge must cover the final two configured chunks"
        )
    return start


def _review_geometry(
    *,
    configured_chunks: Any,
    validated_prefix_count: Any,
    terminal_merge_enabled: bool,
    terminal_pair_start: Any,
) -> tuple[int, int, int | None]:
    chunks = _integer(
        configured_chunks,
        name="configured_chunks",
        minimum=1,
        maximum=0x7FFFFFFF,
    )
    prefix = _integer(
        validated_prefix_count,
        name="validated_prefix_count",
        minimum=0,
        maximum=chunks,
    )
    pair_start = _terminal_pair_start(
        configured_chunks=chunks,
        terminal_merge_enabled=terminal_merge_enabled,
        terminal_pair_start=terminal_pair_start,
    )
    if pair_start is not None and pair_start <= prefix < chunks:
        raise ReviewControlError(
            "validated prefix ends inside the atomic Terminal Merge pair"
        )
    return chunks, prefix, pair_start


def resolve_next_review_unit(
    *,
    configured_chunks: int,
    validated_prefix_count: int,
    terminal_merge_enabled: bool,
    terminal_pair_start: int | None,
) -> ReviewUnit | None:
    """Return the next atomic physical group after a validated logical prefix."""

    chunks, prefix, pair_start = _review_geometry(
        configured_chunks=configured_chunks,
        validated_prefix_count=validated_prefix_count,
        terminal_merge_enabled=terminal_merge_enabled,
        terminal_pair_start=terminal_pair_start,
    )
    if prefix == chunks:
        return None
    start = prefix + 1
    if pair_start is not None and start == pair_start:
        return ReviewUnit(
            start=pair_start,
            end=chunks,
            physical_group=pair_start,
        )
    return ReviewUnit(start=start, end=start, physical_group=start)


def _review_unit_for_chunk(
    chunk: int,
    *,
    configured_chunks: int,
    terminal_pair_start: int | None,
) -> ReviewUnit:
    if terminal_pair_start is not None and chunk >= terminal_pair_start:
        return ReviewUnit(
            start=terminal_pair_start,
            end=configured_chunks,
            physical_group=terminal_pair_start,
        )
    return ReviewUnit(start=chunk, end=chunk, physical_group=chunk)


def _validated_review_unit(
    value: Any,
    *,
    configured_chunks: int,
    validated_prefix_count: int,
    terminal_pair_start: int | None,
) -> ReviewUnit | None:
    if value is None:
        return None
    if isinstance(value, ReviewUnit):
        unit = value
    elif isinstance(value, Mapping):
        try:
            unit = ReviewUnit(
                start=value["start"],
                end=value["end"],
                physical_group=value["physical_group"],
            )
        except KeyError as exc:
            raise ReviewControlError(
                "latest_review_unit is missing required metadata"
            ) from exc
    else:
        raise ReviewControlError(
            "latest_review_unit must be review metadata or None"
        )
    start = _integer(
        unit.start,
        name="latest_review_unit.start",
        minimum=1,
        maximum=configured_chunks,
    )
    end = _integer(
        unit.end,
        name="latest_review_unit.end",
        minimum=start,
        maximum=configured_chunks,
    )
    physical_group = _integer(
        unit.physical_group,
        name="latest_review_unit.physical_group",
        minimum=1,
        maximum=configured_chunks,
    )
    if physical_group != start:
        raise ReviewControlError(
            "latest_review_unit physical group must equal its first logical chunk"
        )
    if terminal_pair_start is not None and start == terminal_pair_start:
        if end != configured_chunks:
            raise ReviewControlError(
                "latest_review_unit splits the atomic Terminal Merge pair"
            )
    elif end != start:
        raise ReviewControlError(
            "a non-terminal review unit must contain exactly one logical chunk"
        )
    if terminal_pair_start is not None and start > terminal_pair_start:
        raise ReviewControlError(
            "latest_review_unit starts inside the atomic Terminal Merge pair"
        )
    if end != validated_prefix_count:
        raise ReviewControlError(
            "latest_review_unit is not the head of the validated prefix"
        )
    return ReviewUnit(start=start, end=end, physical_group=physical_group)


def make_review_pause_metadata(review_unit: ReviewUnit) -> dict[str, Any]:
    """Build non-identity manifest metadata for a successful review pause."""

    if not isinstance(review_unit, ReviewUnit):
        raise ReviewControlError("review_unit must be a ReviewUnit")
    start = _integer(
        review_unit.start,
        name="review_unit.start",
        minimum=1,
        maximum=0x7FFFFFFF,
    )
    end = _integer(
        review_unit.end,
        name="review_unit.end",
        minimum=start,
        maximum=0x7FFFFFFF,
    )
    physical_group = _integer(
        review_unit.physical_group,
        name="review_unit.physical_group",
        minimum=1,
        maximum=0x7FFFFFFF,
    )
    if physical_group != start:
        raise ReviewControlError(
            "review_unit physical group must equal its first logical chunk"
        )
    return {
        "review_unit": {
            "start": start,
            "end": end,
            "physical_group": physical_group,
        },
        "review_pause_reason": REVIEW_PAUSE_REASON,
    }


def _revision_status(value: Any, *, prefix: int, chunks: int) -> str | None:
    if value is None or value == "":
        if prefix:
            raise ReviewControlError(
                "a non-empty validated prefix requires a revision status"
            )
        return None
    status = str(value)
    if status not in REVISION_STATUSES:
        raise ReviewControlError(f"unknown latest revision status: {status!r}")
    if status == REVISION_STATUS_COMPLETE and prefix != chunks:
        raise ReviewControlError(
            "a complete revision must contain every configured chunk"
        )
    if status == REVISION_STATUS_REVIEW_READY and not 0 < prefix < chunks:
        raise ReviewControlError(
            "review_ready requires a non-empty incomplete prefix"
        )
    return status


def _branch_contract(
    *,
    latest_effective_nonce: Any,
    latest_branch_regenerate_from: Any,
    validated_prefix_count: int,
    configured_chunks: int,
    latest_revision_status: str | None,
) -> tuple[int, int]:
    nonce = _integer(
        latest_effective_nonce,
        name="latest_effective_nonce",
        minimum=0,
        maximum=0xFFFFFFFF,
    )
    branch_start = _manual_regenerate_from(
        latest_branch_regenerate_from,
        configured_chunks=configured_chunks,
    )
    if nonce == 0 and branch_start != 0:
        raise ReviewControlError(
            "a regenerate branch requires a positive effective nonce"
        )
    if nonce > 0 and branch_start == 0:
        raise ReviewControlError(
            "a positive effective nonce requires its regenerate branch boundary"
        )
    if branch_start > validated_prefix_count + 1:
        raise ReviewControlError(
            "regenerate branch boundary skips the validated prefix head"
        )
    if (
        latest_revision_status
        in (REVISION_STATUS_REVIEW_READY, REVISION_STATUS_COMPLETE)
        and branch_start > validated_prefix_count
    ):
        raise ReviewControlError(
            "a ready regenerate branch has not committed its boundary chunk"
        )
    return nonce, branch_start


def validate_review_prefix_metadata(
    *,
    configured_chunks: int,
    validated_prefix_count: int,
    terminal_merge_enabled: bool,
    terminal_pair_start: int | None,
    latest_review_unit: ReviewUnit | Mapping[str, Any] | None,
    latest_revision_status: str | None,
    latest_effective_nonce: int,
    latest_branch_regenerate_from: Any = 0,
) -> ReviewUnit | None:
    """Validate persisted Review metadata without creating an execution decision.

    Repository reads and finalization use this boundary after raw prefix
    validation.  It deliberately returns no ``ReviewExecution`` so those paths
    cannot become a second policy owner.
    """

    chunks, prefix, pair_start = _review_geometry(
        configured_chunks=configured_chunks,
        validated_prefix_count=validated_prefix_count,
        terminal_merge_enabled=terminal_merge_enabled,
        terminal_pair_start=terminal_pair_start,
    )
    status = _revision_status(
        latest_revision_status,
        prefix=prefix,
        chunks=chunks,
    )
    unit = _validated_review_unit(
        latest_review_unit,
        configured_chunks=chunks,
        validated_prefix_count=prefix,
        terminal_pair_start=pair_start,
    )
    _branch_contract(
        latest_effective_nonce=latest_effective_nonce,
        latest_branch_regenerate_from=latest_branch_regenerate_from,
        validated_prefix_count=prefix,
        configured_chunks=chunks,
        latest_revision_status=status,
    )
    return unit


def _nonce_decision(
    *,
    manual_regenerate_from: int,
    smart_regenerate: bool,
    latest_effective_nonce: int,
    latest_branch_regenerate_from: int,
    smart_target: int = 0,
) -> tuple[str, int, int | None]:
    if smart_regenerate:
        if latest_effective_nonce >= 0xFFFFFFFF:
            raise ReviewControlError(
                "effective reroll nonce variation space is exhausted"
            )
        return (
            NONCE_POLICY_ADVANCE,
            smart_target,
            latest_effective_nonce + 1,
        )
    if manual_regenerate_from:
        return NONCE_POLICY_MANUAL, manual_regenerate_from, None
    if latest_effective_nonce:
        return (
            NONCE_POLICY_INHERIT,
            latest_branch_regenerate_from,
            latest_effective_nonce,
        )
    return NONCE_POLICY_INACTIVE, 0, 0


def _unit_status(unit: ReviewUnit, *, chunks: int, verb: str) -> str:
    if unit.start == unit.end:
        return f"{verb}: Chunk {unit.start} / {chunks}."
    return (
        f"{verb}: Chunks {unit.start}-{unit.end} / {chunks}; "
        "Terminal Merge atomic group."
    )


def resolve_review_execution(
    *,
    generation_mode: str,
    review_action: str,
    configured_chunks: int,
    validated_prefix_count: int,
    terminal_merge_enabled: bool,
    terminal_pair_start: int | None,
    manual_regenerate_from: Any,
    run_storage_mode: str,
    latest_review_unit: ReviewUnit | Mapping[str, Any] | None,
    latest_revision_status: str | None,
    latest_effective_nonce: int,
    latest_branch_regenerate_from: Any = 0,
) -> ReviewExecution:
    """Resolve V3.8 review intent without touching Production or storage state."""

    if generation_mode not in GENERATION_MODE_OPTIONS:
        raise ReviewControlError(f"unknown Generation Mode: {generation_mode!r}")
    if review_action not in REVIEW_ACTION_OPTIONS:
        raise ReviewControlError(f"unknown Review Action: {review_action!r}")
    chunks, prefix, pair_start = _review_geometry(
        configured_chunks=configured_chunks,
        validated_prefix_count=validated_prefix_count,
        terminal_merge_enabled=terminal_merge_enabled,
        terminal_pair_start=terminal_pair_start,
    )
    status = _revision_status(
        latest_revision_status,
        prefix=prefix,
        chunks=chunks,
    )
    manual_from = _manual_regenerate_from(
        manual_regenerate_from,
        configured_chunks=chunks,
    )
    nonce, existing_branch_start = _branch_contract(
        latest_effective_nonce=latest_effective_nonce,
        latest_branch_regenerate_from=latest_branch_regenerate_from,
        validated_prefix_count=prefix,
        configured_chunks=chunks,
        latest_revision_status=status,
    )

    if generation_mode == GENERATION_MODE_FULL_RUN:
        nonce_policy, branch_start, requested_nonce = _nonce_decision(
            manual_regenerate_from=manual_from,
            smart_regenerate=False,
            latest_effective_nonce=nonce,
            latest_branch_regenerate_from=existing_branch_start,
        )
        return ReviewExecution(
            execution_mode=EXECUTION_MODE_FULL_RUN,
            requires_run_storage=False,
            effective_regenerate_from=(manual_from or branch_start),
            max_new_physical_groups=None,
            next_review_unit_start=None,
            next_review_unit_end=None,
            next_review_physical_group=None,
            finish_remaining=False,
            smart_regenerate=False,
            one_shot_reset_required=False,
            partial_review=False,
            projected_prefix_count=chunks,
            nonce_policy=nonce_policy,
            branch_regenerate_from=branch_start,
            base_effective_nonce=nonce,
            requested_effective_nonce=requested_nonce,
            status_hint="Full Run: Review execution limit disabled.",
        )

    if run_storage_mode != RUN_STORAGE_SAVE_AUTO_RESUME:
        raise ReviewControlError(
            "Review Each Chunk requires Run Storage = Save + Auto Resume."
        )

    latest_unit = _validated_review_unit(
        latest_review_unit,
        configured_chunks=chunks,
        validated_prefix_count=prefix,
        terminal_pair_start=pair_start,
    )

    if review_action == REVIEW_ACTION_REGENERATE_CURRENT:
        if manual_from:
            raise ReviewControlError(
                "Regenerate Current cannot be combined with manual Regenerate From. "
                "Set Regenerate From to Auto."
            )
        if prefix == 0:
            raise ReviewControlError(
                "Regenerate Current requires an existing reviewed unit"
            )
        if latest_unit is None:
            raise ReviewControlError(
                "Regenerate Current requires latest review_unit metadata"
            )
        if status not in (
            REVISION_STATUS_REVIEW_READY,
            REVISION_STATUS_COMPLETE,
        ):
            raise ReviewControlError(
                "Regenerate Current requires a review_ready revision or a complete "
                "revision with review_unit metadata"
            )
        nonce_policy, branch_start, requested_nonce = _nonce_decision(
            manual_regenerate_from=0,
            smart_regenerate=True,
            latest_effective_nonce=nonce,
            latest_branch_regenerate_from=existing_branch_start,
            smart_target=latest_unit.start,
        )
        projected = latest_unit.end
        return ReviewExecution(
            execution_mode=EXECUTION_MODE_REVIEW_REGENERATE,
            requires_run_storage=True,
            effective_regenerate_from=latest_unit.start,
            max_new_physical_groups=1,
            next_review_unit_start=latest_unit.start,
            next_review_unit_end=latest_unit.end,
            next_review_physical_group=latest_unit.physical_group,
            finish_remaining=False,
            smart_regenerate=True,
            one_shot_reset_required=True,
            partial_review=projected < chunks,
            projected_prefix_count=projected,
            nonce_policy=nonce_policy,
            branch_regenerate_from=branch_start,
            base_effective_nonce=nonce,
            requested_effective_nonce=requested_nonce,
            status_hint=_unit_status(
                latest_unit,
                chunks=chunks,
                verb="Smart Regenerate",
            ),
        )

    nonce_policy, branch_start, requested_nonce = _nonce_decision(
        manual_regenerate_from=manual_from,
        smart_regenerate=False,
        latest_effective_nonce=nonce,
        latest_branch_regenerate_from=existing_branch_start,
    )
    if review_action == REVIEW_ACTION_FINISH_REMAINING:
        return ReviewExecution(
            execution_mode=EXECUTION_MODE_REVIEW_FINISH,
            requires_run_storage=True,
            effective_regenerate_from=(manual_from or branch_start),
            max_new_physical_groups=None,
            next_review_unit_start=None,
            next_review_unit_end=None,
            next_review_physical_group=None,
            finish_remaining=True,
            smart_regenerate=False,
            one_shot_reset_required=True,
            partial_review=False,
            projected_prefix_count=chunks,
            nonce_policy=nonce_policy,
            branch_regenerate_from=branch_start,
            base_effective_nonce=nonce,
            requested_effective_nonce=requested_nonce,
            status_hint=(
                f"Review Mode: finish remaining after accepted prefix "
                f"{prefix} / {chunks}."
            ),
        )

    if manual_from:
        next_unit = _review_unit_for_chunk(
            manual_from,
            configured_chunks=chunks,
            terminal_pair_start=pair_start,
        )
    else:
        next_unit = resolve_next_review_unit(
            configured_chunks=chunks,
            validated_prefix_count=prefix,
            terminal_merge_enabled=terminal_merge_enabled,
            terminal_pair_start=pair_start,
        )
    if next_unit is None:
        projected = chunks
        status_hint = "Review Mode: Sequence complete."
        unit_start = unit_end = physical_group = None
    else:
        projected = next_unit.end
        status_hint = _unit_status(
            next_unit,
            chunks=chunks,
            verb="Review next unit",
        )
        unit_start = next_unit.start
        unit_end = next_unit.end
        physical_group = next_unit.physical_group
    return ReviewExecution(
        execution_mode=EXECUTION_MODE_REVIEW_CONTINUE,
        requires_run_storage=True,
        effective_regenerate_from=(manual_from or branch_start),
        max_new_physical_groups=1,
        next_review_unit_start=unit_start,
        next_review_unit_end=unit_end,
        next_review_physical_group=physical_group,
        finish_remaining=False,
        smart_regenerate=False,
        one_shot_reset_required=False,
        partial_review=projected < chunks,
        projected_prefix_count=projected,
        nonce_policy=nonce_policy,
        branch_regenerate_from=branch_start,
        base_effective_nonce=nonce,
        requested_effective_nonce=requested_nonce,
        status_hint=status_hint,
    )


def resolve_take_execution(
    *,
    take_action: str,
    configured_chunks: int,
    selected_prefix_count: int,
    terminal_merge_enabled: bool,
    terminal_pair_start: int | None,
    selected_review_unit: ReviewUnit | Mapping[str, Any],
    selected_effective_nonce: int,
    selected_branch_regenerate_from: int,
    next_effective_nonce: int,
) -> ReviewExecution:
    """Resolve an already-validated immutable Take selection.

    The selected prefix is a Run Storage fact.  Continuing from it starts a
    fresh downstream nonce boundary without changing the Sampling Contract.
    """

    if take_action not in (TAKE_ACTION_USE, TAKE_ACTION_CONTINUE):
        raise ReviewControlError(f"unknown Take action: {take_action!r}")
    chunks, prefix, pair_start = _review_geometry(
        configured_chunks=configured_chunks,
        validated_prefix_count=selected_prefix_count,
        terminal_merge_enabled=terminal_merge_enabled,
        terminal_pair_start=terminal_pair_start,
    )
    unit = _validated_review_unit(
        selected_review_unit,
        configured_chunks=chunks,
        validated_prefix_count=prefix,
        terminal_pair_start=pair_start,
    )
    if unit is None:
        raise ReviewControlError("selected Take is missing its physical review unit")
    selected_nonce, selected_branch = _branch_contract(
        latest_effective_nonce=selected_effective_nonce,
        latest_branch_regenerate_from=selected_branch_regenerate_from,
        validated_prefix_count=prefix,
        configured_chunks=chunks,
        latest_revision_status=(
            REVISION_STATUS_COMPLETE if prefix == chunks else REVISION_STATUS_REVIEW_READY
        ),
    )
    if take_action == TAKE_ACTION_USE:
        return ReviewExecution(
            execution_mode=EXECUTION_MODE_REVIEW_USE_TAKE,
            requires_run_storage=True,
            effective_regenerate_from=selected_branch,
            max_new_physical_groups=0,
            next_review_unit_start=None,
            next_review_unit_end=None,
            next_review_physical_group=None,
            finish_remaining=False,
            smart_regenerate=False,
            one_shot_reset_required=True,
            partial_review=prefix < chunks,
            projected_prefix_count=prefix,
            nonce_policy=NONCE_POLICY_INHERIT if selected_nonce else NONCE_POLICY_INACTIVE,
            branch_regenerate_from=selected_branch,
            base_effective_nonce=selected_nonce,
            requested_effective_nonce=selected_nonce,
            status_hint=_unit_status(unit, chunks=chunks, verb="Use This Take"),
        )

    next_unit = resolve_next_review_unit(
        configured_chunks=chunks,
        validated_prefix_count=prefix,
        terminal_merge_enabled=terminal_merge_enabled,
        terminal_pair_start=pair_start,
    )
    if next_unit is None:
        return ReviewExecution(
            execution_mode=EXECUTION_MODE_REVIEW_USE_TAKE,
            requires_run_storage=True,
            effective_regenerate_from=selected_branch,
            max_new_physical_groups=0,
            next_review_unit_start=None,
            next_review_unit_end=None,
            next_review_physical_group=None,
            finish_remaining=False,
            smart_regenerate=False,
            one_shot_reset_required=True,
            partial_review=False,
            projected_prefix_count=chunks,
            nonce_policy=NONCE_POLICY_INHERIT if selected_nonce else NONCE_POLICY_INACTIVE,
            branch_regenerate_from=selected_branch,
            base_effective_nonce=selected_nonce,
            requested_effective_nonce=selected_nonce,
            status_hint="Continue From Here: selected Take is already complete.",
        )
    nonce = _integer(
        next_effective_nonce,
        name="next_effective_nonce",
        minimum=1,
        maximum=0xFFFFFFFF,
    )
    return ReviewExecution(
        execution_mode=EXECUTION_MODE_REVIEW_CONTINUE_FROM_TAKE,
        requires_run_storage=True,
        effective_regenerate_from=next_unit.start,
        max_new_physical_groups=1,
        next_review_unit_start=next_unit.start,
        next_review_unit_end=next_unit.end,
        next_review_physical_group=next_unit.physical_group,
        finish_remaining=False,
        smart_regenerate=False,
        one_shot_reset_required=True,
        partial_review=next_unit.end < chunks,
        projected_prefix_count=next_unit.end,
        nonce_policy=NONCE_POLICY_ADVANCE,
        branch_regenerate_from=next_unit.start,
        base_effective_nonce=selected_nonce,
        requested_effective_nonce=nonce,
        status_hint=_unit_status(next_unit, chunks=chunks, verb="Continue From Here"),
    )
