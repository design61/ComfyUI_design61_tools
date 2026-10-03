"""Pure physical-group planner for the staged V3.8 runtime redesign.

The caller owns continuation-source priority and passes exactly one selected
``ContinuationSourceFacts``.  This module performs no repository, Session,
State, tensor, ComfyUI, or sampling work.
"""

from __future__ import annotations

from ..branch_provenance import BranchProvenanceError, physical_groups
from .planning_types import (
    ContinuationSourceFacts,
    ExecutionPlan,
    PhysicalGroupDescriptor,
    ProjectionDecision,
)
from .review_control import (
    REVISION_STATUS_COMPLETE,
    REVISION_STATUS_INTERRUPTED,
    REVISION_STATUS_REVIEW_READY,
    ReviewExecution,
    ReviewUnit,
)


PROJECTION_FULL_ACCEPTED_PREFIX = "full_accepted_prefix"
PROJECTION_CURRENT_REVIEW_UNIT = "current_review_unit"


class ExecutionPlanningError(ValueError):
    """Raised when already-selected planning facts are internally inconsistent."""


def resolve_projection_decision(
    *,
    review_execution: ReviewExecution | None,
    effective_result_status: str,
    accepted_chunks: int,
    physical_group_facts: tuple[tuple[int, ...], ...],
) -> ProjectionDecision:
    """Resolve output-only projection from finalized, source-neutral facts."""

    full = ProjectionDecision(PROJECTION_FULL_ACCEPTED_PREFIX)
    if str(effective_result_status) != REVISION_STATUS_REVIEW_READY:
        return full
    if review_execution is None or bool(
        getattr(review_execution, "finish_remaining", False)
    ):
        return full
    if not (
        bool(getattr(review_execution, "partial_review", False))
        or bool(getattr(review_execution, "smart_regenerate", False))
    ):
        return full
    start = getattr(review_execution, "next_review_unit_start", None)
    end = getattr(review_execution, "next_review_unit_end", None)
    if start is None or end is None:
        return full
    start, end = int(start), int(end)
    if start < 1 or end < start or end > int(accepted_chunks):
        return full

    selected = set(range(start, end + 1))
    if not physical_group_facts:
        return full
    for group in physical_group_facts:
        group_indices = set(group)
        if selected.intersection(group_indices) and not group_indices.issubset(selected):
            return full
    return ProjectionDecision(PROJECTION_CURRENT_REVIEW_UNIT, start, end)


def build_execution_plan(
    *,
    continuation_source_facts: ContinuationSourceFacts,
    configured_chunks: int,
    terminal_merge_enabled: bool,
    max_new_physical_groups: int | None,
    review_execution: ReviewExecution | None,
    retained_review_unit: ReviewUnit | None = None,
) -> ExecutionPlan:
    """Plan physical groups without selecting or validating runtime sources."""

    if not isinstance(continuation_source_facts, ContinuationSourceFacts):
        raise ExecutionPlanningError(
            "continuation_source_facts must be one selected ContinuationSourceFacts"
        )
    if continuation_source_facts.kind not in {
        "run_storage",
        "explicit_session",
        "initial_state",
        "none",
    }:
        raise ExecutionPlanningError(
            f"unknown continuation source kind: {continuation_source_facts.kind!r}"
        )
    if isinstance(configured_chunks, bool) or not isinstance(configured_chunks, int):
        raise ExecutionPlanningError("configured_chunks must be an integer")
    if configured_chunks < 1:
        raise ExecutionPlanningError("configured_chunks must be positive")
    if type(terminal_merge_enabled) is not bool:
        raise ExecutionPlanningError("terminal_merge_enabled must be a boolean")
    if max_new_physical_groups is not None and (
        isinstance(max_new_physical_groups, bool)
        or not isinstance(max_new_physical_groups, int)
        or max_new_physical_groups < 0
    ):
        raise ExecutionPlanningError(
            "max_new_physical_groups must be None or a non-negative integer"
        )

    accepted = continuation_source_facts.accepted_chunks
    if isinstance(accepted, bool) or not isinstance(accepted, int):
        raise ExecutionPlanningError("accepted_chunks must be an integer")
    if not 0 <= accepted <= configured_chunks:
        raise ExecutionPlanningError(
            "accepted_chunks must be within the configured sequence"
        )

    try:
        runtime_groups = physical_groups(
            chunks=configured_chunks,
            terminal_merge_enabled=terminal_merge_enabled,
        )
    except BranchProvenanceError as exc:
        raise ExecutionPlanningError(str(exc)) from exc
    descriptors = tuple(
        PhysicalGroupDescriptor(
            physical_group=int(group.physical_group),
            logical_chunks=tuple(range(int(group.start), int(group.end) + 1)),
            terminal_atomic=int(group.start) != int(group.end),
        )
        for group in runtime_groups
    )
    accepted_boundaries = tuple(
        (descriptor.logical_chunks[0], descriptor.logical_chunks[-1])
        for descriptor in descriptors
        if descriptor.logical_chunks[-1] <= accepted
    )
    if accepted_boundaries and accepted_boundaries[-1][1] != accepted:
        raise ExecutionPlanningError("accepted prefix ends inside a physical group")
    if accepted and not accepted_boundaries:
        raise ExecutionPlanningError("accepted prefix has no complete physical group")
    if continuation_source_facts.physical_group_boundaries != accepted_boundaries:
        raise ExecutionPlanningError(
            "selected continuation source physical groups do not match its prefix"
        )

    pending = tuple(
        descriptor
        for descriptor in descriptors
        if descriptor.logical_chunks[0] > accepted
    )
    groups_to_generate = (
        pending
        if max_new_physical_groups is None
        else pending[:max_new_physical_groups]
    )
    expected_chunks = accepted + sum(
        len(descriptor.logical_chunks) for descriptor in groups_to_generate
    )
    expected_review_unit = retained_review_unit
    if groups_to_generate and max_new_physical_groups is not None:
        final_group = groups_to_generate[-1]
        expected_review_unit = ReviewUnit(
            start=final_group.logical_chunks[0],
            end=final_group.logical_chunks[-1],
            physical_group=final_group.physical_group,
        )

    if expected_chunks == configured_chunks:
        expected_status = REVISION_STATUS_COMPLETE
    elif max_new_physical_groups is not None and expected_review_unit is not None:
        expected_status = REVISION_STATUS_REVIEW_READY
    else:
        expected_status = REVISION_STATUS_INTERRUPTED

    projection = resolve_projection_decision(
        review_execution=review_execution,
        effective_result_status=expected_status,
        accepted_chunks=expected_chunks,
        physical_group_facts=tuple(
            descriptor.logical_chunks for descriptor in descriptors
        ),
    )
    return ExecutionPlan(
        groups_to_generate=groups_to_generate,
        projection=projection,
        expected_status=expected_status,
        expected_review_unit=expected_review_unit,
    )
