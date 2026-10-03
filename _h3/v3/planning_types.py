"""Immutable facts shared by the pure Continuum execution planner.

This module must remain free of torch and ComfyUI imports.  Runtime payloads
stay in their existing adapters and are projected into these facts only after
the active continuation source has already been selected.
"""

from __future__ import annotations

from dataclasses import dataclass

from .review_control import ReviewUnit


@dataclass(frozen=True, slots=True)
class PrefixFacts:
    """Immutable planning projection of one fully validated storage prefix."""

    accepted_chunks: int
    configured_chunks: int
    revision_id: str | None
    status: str | None
    review_unit: ReviewUnit | None
    effective_nonce: int
    branch_regenerate_from: int
    physical_group_boundaries: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class ExplicitSessionFacts:
    """Characterized facts for an explicit Session runtime input."""

    present: bool
    compatible: bool
    accepted_chunks: int
    reroll_from_chunk: int | None
    physical_group_boundaries: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class InitialStateFacts:
    """Characterized facts for an initial State runtime input."""

    present: bool
    usable: bool


@dataclass(frozen=True, slots=True)
class ContinuationSourceFacts:
    """Planning projection of the single source selected by the runtime."""

    kind: str
    accepted_chunks: int
    reroll_from_chunk: int | None
    physical_group_boundaries: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class ProjectionDecision:
    """Describe which accepted chunks may be exposed to external decode."""

    kind: str
    start_chunk: int | None = None
    end_chunk: int | None = None


@dataclass(frozen=True, slots=True)
class PhysicalGroupDescriptor:
    """One atomic sampling group expressed in one-based logical chunks."""

    physical_group: int
    logical_chunks: tuple[int, ...]
    terminal_atomic: bool


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Pure plan consumed by the unchanged runtime execution loops."""

    groups_to_generate: tuple[PhysicalGroupDescriptor, ...]
    projection: ProjectionDecision
    expected_status: str
    expected_review_unit: ReviewUnit | None
