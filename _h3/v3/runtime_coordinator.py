"""Internal owner of Continuum runtime continuation-source priority.

The pure execution planner receives only the source selected here.  Runtime
payload validation remains outside the planner so Run Storage, explicit
Session, and initial State can never become competing planner inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from ..branch_provenance import physical_groups
from ..run_storage import RunStorageError
from ..state import validate_state
from ..v2.session import SessionValidationError, entry_to_state
from .planning_types import ContinuationSourceFacts


LOG = logging.getLogger("h3_continuum_join")


class RuntimeCoordinatorError(ValueError):
    """Raised when the characterized runtime source contract is invalid."""


@dataclass(slots=True)
class RuntimeContinuationSelection:
    """Selected runtime payload plus its single planning projection."""

    entries: list[dict[str, Any]]
    previous_state: dict[str, Any] | None
    continuation_source_facts: ContinuationSourceFacts
    notes: tuple[str, ...]


def project_selected_continuation_source(
    *,
    kind: str,
    selected_source: dict[str, Any] | None,
) -> ContinuationSourceFacts:
    """Project one already-selected runtime source into planning-only facts."""

    normalized_kind = str(kind)
    if normalized_kind == "none":
        if selected_source is not None:
            raise RuntimeCoordinatorError("the none continuation source has no payload")
        return ContinuationSourceFacts(
            kind="none",
            accepted_chunks=0,
            reroll_from_chunk=0,
            physical_group_boundaries=(),
        )
    if normalized_kind not in {
        "run_storage",
        "explicit_session",
        "initial_state",
    }:
        raise RuntimeCoordinatorError(
            f"unknown selected continuation source: {normalized_kind!r}"
        )
    if not isinstance(selected_source, dict):
        raise RuntimeCoordinatorError(
            "the selected continuation source requires normalized runtime facts"
        )
    try:
        accepted_chunks = int(selected_source["accepted_chunks"])
        reroll_from_chunk = int(selected_source["reroll_from_chunk"])
        boundaries = tuple(
            (int(start), int(end))
            for start, end in selected_source["physical_group_boundaries"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeCoordinatorError(
            "selected continuation source facts are invalid"
        ) from exc
    return ContinuationSourceFacts(
        kind=normalized_kind,
        accepted_chunks=accepted_chunks,
        reroll_from_chunk=reroll_from_chunk,
        physical_group_boundaries=boundaries,
    )


class InternalRuntimeCoordinator:
    """Own the established Storage / Session / State priority contract."""

    def __init__(
        self,
        *,
        storage_controller: Any,
        input_session: dict[str, Any] | None,
        initial_state: dict[str, Any] | None,
    ) -> None:
        self.storage_controller = storage_controller
        self.input_session = input_session
        self.initial_state = initial_state
        self.session_present = input_session is not None

    @property
    def storage_enabled(self) -> bool:
        return self.storage_controller is not None

    def prepare_inputs(
        self,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Suppress State as soon as an explicit Session is present."""

        if self.session_present and self.initial_state is not None:
            LOG.warning(
                "Both session and initial_state were supplied; using the session "
                "and ignoring initial_state"
            )
            self.initial_state = None
        return self.input_session, self.initial_state

    def assert_storage_session_compatible(self) -> None:
        """Preserve the existing Run Storage plus explicit Session hard error."""

        if self.storage_enabled and self.session_present:
            raise RunStorageError(
                "Run Storage cannot be combined with an explicit Session"
            )

    def validate_initial_state_reroll(self, reroll_from_chunk: int) -> None:
        """Apply the existing reroll restriction to the effective State only."""

        if self.initial_state is not None and int(reroll_from_chunk) not in (0, 1):
            raise RuntimeCoordinatorError(
                "with initial_state, reroll_from_chunk can only be 0 or 1"
            )

    def select_continuation_source(
        self,
        *,
        preserved: list[dict[str, Any]],
        initial_state: dict[str, Any] | None,
        chunks: int,
        width: int,
        height: int,
        reroll_from_chunk: int,
        effective_reroll_from_chunk: int,
        terminal_merge_enabled: bool,
    ) -> RuntimeContinuationSelection:
        """Select exactly one runtime continuation source, then project it."""

        entries = preserved[:]
        previous_state = None
        notes: list[str] = []

        if entries:
            try:
                previous_state = entry_to_state(entries[-1])
            except (SessionValidationError, ValueError) as exc:
                notes.append(
                    "saved continuation state was rejected; generated a fresh run "
                    f"({exc})"
                )
                entries = []
                previous_state = None

        # An initial State is considered only if no usable Storage/Session prefix
        # survived. Explicit Session presence already suppressed it in
        # prepare_inputs(), before Session validity was known.
        if not entries and initial_state is not None:
            try:
                candidate = validate_state(initial_state)
                if int(candidate["width"]) != int(width) or int(
                    candidate["height"]
                ) != int(height):
                    notes.append(
                        "initial_state resolution differs; generated a fresh run"
                    )
                else:
                    previous_state = candidate
            except ValueError as exc:
                notes.append(
                    f"initial_state was rejected; generated a fresh run ({exc})"
                )

        if entries:
            selected_kind = (
                "run_storage" if self.storage_enabled else "explicit_session"
            )
            source_reroll = (
                int(reroll_from_chunk)
                if self.storage_enabled
                else int(effective_reroll_from_chunk)
            )
            accepted_boundaries = tuple(
                (int(group.start), int(group.end))
                for group in physical_groups(
                    chunks=int(chunks),
                    terminal_merge_enabled=bool(terminal_merge_enabled),
                )
                if int(group.end) <= len(entries)
            )
            selected_source = {
                "accepted_chunks": len(entries),
                "reroll_from_chunk": source_reroll,
                "physical_group_boundaries": accepted_boundaries,
            }
        elif previous_state is not None and initial_state is not None:
            selected_kind = "initial_state"
            selected_source = {
                "accepted_chunks": 0,
                "reroll_from_chunk": int(reroll_from_chunk),
                "physical_group_boundaries": (),
            }
        else:
            selected_kind = "none"
            selected_source = None

        return RuntimeContinuationSelection(
            entries=entries,
            previous_state=previous_state,
            continuation_source_facts=project_selected_continuation_source(
                kind=selected_kind,
                selected_source=selected_source,
            ),
            notes=tuple(notes),
        )
