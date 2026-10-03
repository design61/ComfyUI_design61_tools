"""Pure RR-R3 plan joining fixed Reference identities to Core Picture numbers.

The caller supplies only immutable routing and Reference metadata. This module
does not encode images, rewrite prompts, select runtime sources, or sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .reference_routing import REFERENCE_SLOT_IDS, ReferenceRoutingSchedule


_SLOT_ORDER = {slot_id: index for index, slot_id in enumerate(REFERENCE_SLOT_IDS)}


@dataclass(frozen=True, slots=True)
class EffectivePlanIssue:
    code: str
    message: str
    source_slot_id: str | None = None


@dataclass(frozen=True, slots=True)
class ReferencePicture:
    """A fixed source slot, compact asset, content hash, and Qwen Picture number."""

    source_slot_id: str
    asset_index: int
    image_sha256: str
    picture_number: int

    @property
    def picture_tag(self) -> str:
        return f"<Picture {self.picture_number}>"


@dataclass(frozen=True, slots=True)
class PictureMap:
    """Core's image presentation order; First/Last are not Reference assets."""

    first_picture_number: int | None
    last_picture_number: int | None
    references: tuple[ReferencePicture, ...]

    @property
    def source_slot_ids(self) -> tuple[str, ...]:
        return tuple(item.source_slot_id for item in self.references)


@dataclass(frozen=True, slots=True)
class LogicalEffectiveRoute:
    logical_chunk: int
    requested_source_slot_ids: tuple[str, ...]
    picture_map: PictureMap


@dataclass(frozen=True, slots=True)
class PhysicalEffectiveRoute:
    physical_group: int
    logical_chunks: tuple[int, ...]
    terminal_atomic: bool
    routing_status: str
    status: str
    requested_source_slot_ids: tuple[str, ...] | None
    picture_map: PictureMap | None


@dataclass(frozen=True, slots=True)
class EffectiveReferencePlan:
    mode: str
    legacy_fallback: bool
    connected_source_slot_ids: tuple[str, ...]
    logical_routes: tuple[LogicalEffectiveRoute, ...] = ()
    physical_routes: tuple[PhysicalEffectiveRoute, ...] = ()
    issues: tuple[EffectivePlanIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.issues

    @property
    def has_route_conflicts(self) -> bool:
        return any(group.status == "conflict" for group in self.physical_routes)


def _schedule_issues(schedule: ReferenceRoutingSchedule) -> tuple[EffectivePlanIssue, ...]:
    if not schedule.valid:
        return tuple(
            EffectivePlanIssue(f"routing_{issue.code}", issue.message, issue.slot_id)
            for issue in schedule.issues
        )
    total = schedule.total_chunks
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        return (EffectivePlanIssue("invalid_routing_schedule", "missing logical chunk count"),)
    if tuple(route.logical_chunk for route in schedule.logical_routes) != tuple(range(1, total + 1)):
        return (EffectivePlanIssue("invalid_routing_schedule", "logical chunks are not the full project sequence"),)

    logical_slots = {route.logical_chunk: route.reference_slot_ids for route in schedule.logical_routes}
    for route in schedule.logical_routes:
        if any(slot_id not in _SLOT_ORDER for slot_id in route.reference_slot_ids):
            return (EffectivePlanIssue("invalid_routing_schedule", "logical route has an unknown Reference slot"),)
        if len(set(route.reference_slot_ids)) != len(route.reference_slot_ids):
            return (EffectivePlanIssue("invalid_routing_schedule", "logical route repeats a Reference slot"),)
        if tuple(sorted(route.reference_slot_ids, key=_SLOT_ORDER.get)) != route.reference_slot_ids:
            return (EffectivePlanIssue("invalid_routing_schedule", "logical Reference slots are not ordered"),)

    covered = tuple(chunk for group in schedule.physical_routes for chunk in group.logical_chunks)
    if covered != tuple(range(1, total + 1)):
        return (EffectivePlanIssue("invalid_routing_schedule", "physical groups do not cover the logical sequence"),)
    for group in schedule.physical_routes:
        if group.terminal_atomic != (len(group.logical_chunks) > 1):
            return (EffectivePlanIssue("invalid_routing_schedule", "physical group atomicity disagrees with its chunks"),)
        members = tuple(logical_slots[chunk] for chunk in group.logical_chunks)
        if group.logical_reference_slot_ids != members:
            return (EffectivePlanIssue("invalid_routing_schedule", "physical group member routes disagree with logical routes"),)
        consistent = all(member == members[0] for member in members[1:])
        if consistent:
            expected_status = "empty" if not members[0] else "consistent"
            if group.status != expected_status or group.reference_slot_ids != members[0]:
                return (EffectivePlanIssue("invalid_routing_schedule", "physical group route is not its shared member route"),)
        elif group.status != "conflict" or group.reference_slot_ids is not None:
            return (EffectivePlanIssue("invalid_routing_schedule", "conflicting physical group has an effective route"),)
    return ()


def _source_issues(
    source_slot_ids: Any, image_hashes: Any, asset_count: Any
) -> tuple[EffectivePlanIssue, ...]:
    issues: list[EffectivePlanIssue] = []
    if isinstance(asset_count, bool) or not isinstance(asset_count, int) or asset_count < 0:
        issues.append(EffectivePlanIssue("invalid_asset_count", "Reference asset count must be a non-negative integer"))
    if not isinstance(source_slot_ids, tuple):
        issues.append(EffectivePlanIssue("invalid_source_slots", "source_slot_ids must be a tuple"))
        return tuple(issues)
    if not isinstance(image_hashes, tuple):
        issues.append(EffectivePlanIssue("invalid_image_hashes", "Reference image hashes must be a tuple"))
    elif len(image_hashes) != len(source_slot_ids):
        issues.append(EffectivePlanIssue("hash_count_mismatch", "Reference image hashes do not align with fixed source slots"))
    elif any(not isinstance(value, str) or len(value) != 64 or
             any(char not in "0123456789abcdef" for char in value) for value in image_hashes):
        issues.append(EffectivePlanIssue("invalid_image_hash", "Reference image hashes must be lowercase SHA-256 values"))
    if isinstance(asset_count, int) and not isinstance(asset_count, bool):
        if asset_count > 0 and not source_slot_ids:
            issues.append(EffectivePlanIssue("missing_source_slots", "connected Reference assets have no fixed source slot IDs"))
        elif asset_count != len(source_slot_ids):
            issues.append(EffectivePlanIssue("asset_count_mismatch", "source slot IDs do not align with compact Reference assets"))
    for slot_id in source_slot_ids:
        if not isinstance(slot_id, str) or slot_id not in _SLOT_ORDER:
            issues.append(EffectivePlanIssue("unknown_source_slot", f"unknown Reference source slot: {slot_id!r}", str(slot_id)))
    if not any(issue.code == "unknown_source_slot" for issue in issues):
        if len(set(source_slot_ids)) != len(source_slot_ids):
            issues.append(EffectivePlanIssue("duplicate_source_slot", "a fixed Reference source slot occurs more than once"))
        elif tuple(sorted(source_slot_ids, key=_SLOT_ORDER.get)) != source_slot_ids:
            issues.append(EffectivePlanIssue("source_slot_order", "source slot IDs do not follow the compact R1-R9 order"))
    return tuple(issues)


def _picture_map(
    requested_source_slot_ids: tuple[str, ...],
    connected_asset_indices: dict[str, int],
    image_hashes: tuple[str, ...],
    *,
    has_first_image: bool,
    has_last_image: bool,
) -> PictureMap:
    first_number = 1 if has_first_image else None
    last_number = 1 + int(has_first_image) if has_last_image else None
    picture_offset = int(has_first_image) + int(has_last_image)
    selected = tuple(slot_id for slot_id in requested_source_slot_ids if slot_id in connected_asset_indices)
    references = tuple(
        ReferencePicture(
            slot_id, connected_asset_indices[slot_id],
            image_hashes[connected_asset_indices[slot_id]], picture_offset + index,
        )
        for index, slot_id in enumerate(selected, start=1)
    )
    return PictureMap(first_number, last_number, references)


def compile_group_picture_map(
    *,
    source_slot_ids: tuple[str, ...],
    reference_image_hashes: tuple[str, ...],
    has_first_image: bool,
    has_last_image: bool,
) -> PictureMap:
    """Use RR-R3's Picture numbering for one already-selected physical group."""

    issues = _source_issues(source_slot_ids, reference_image_hashes, len(source_slot_ids))
    if issues or type(has_first_image) is not bool or type(has_last_image) is not bool:
        raise ValueError("invalid internal Reference group identity or anchor flags")
    return _picture_map(
        source_slot_ids,
        {slot_id: index for index, slot_id in enumerate(source_slot_ids)},
        reference_image_hashes,
        has_first_image=has_first_image,
        has_last_image=has_last_image,
    )


def compile_effective_reference_plan(
    *,
    routing_schedule: ReferenceRoutingSchedule,
    source_slot_ids: tuple[str, ...],
    reference_image_hashes: tuple[str, ...],
    reference_asset_count: int,
    has_first_image: bool,
    has_last_image: bool,
) -> EffectiveReferencePlan:
    """Map full logical routes to compact assets and Core Qwen Picture numbers.

    Anchor flags describe the presentation being planned. The caller retains
    responsibility for choosing when First/Last are supplied to conditioning.
    """

    if not isinstance(routing_schedule, ReferenceRoutingSchedule):
        return EffectiveReferencePlan(
            mode="invalid", legacy_fallback=False, connected_source_slot_ids=(),
            issues=(EffectivePlanIssue("invalid_routing_schedule", "expected an RR-R2 routing schedule"),),
        )
    issues = [
        *_schedule_issues(routing_schedule),
        *_source_issues(source_slot_ids, reference_image_hashes, reference_asset_count),
    ]
    if type(has_first_image) is not bool or type(has_last_image) is not bool:
        issues.append(EffectivePlanIssue("invalid_anchor_flags", "First/Last presence flags must be booleans"))
    if issues:
        return EffectiveReferencePlan(
            mode=routing_schedule.mode,
            legacy_fallback=routing_schedule.legacy_fallback,
            connected_source_slot_ids=source_slot_ids if isinstance(source_slot_ids, tuple) else (),
            issues=tuple(issues),
        )

    asset_indices = {slot_id: index for index, slot_id in enumerate(source_slot_ids)}
    logical_routes = tuple(
        LogicalEffectiveRoute(
            logical_chunk=route.logical_chunk,
            requested_source_slot_ids=route.reference_slot_ids,
            picture_map=_picture_map(
                route.reference_slot_ids, asset_indices, reference_image_hashes,
                has_first_image=has_first_image, has_last_image=has_last_image,
            ),
        )
        for route in routing_schedule.logical_routes
    )
    physical_routes: list[PhysicalEffectiveRoute] = []
    for group in routing_schedule.physical_routes:
        if group.status == "conflict":
            physical_routes.append(
                PhysicalEffectiveRoute(
                    physical_group=group.physical_group,
                    logical_chunks=group.logical_chunks,
                    terminal_atomic=group.terminal_atomic,
                    routing_status=group.status,
                    status="conflict",
                    requested_source_slot_ids=None,
                    picture_map=None,
                )
            )
            continue
        requested = group.reference_slot_ids
        assert requested is not None
        picture_map = _picture_map(
            requested, asset_indices, reference_image_hashes,
            has_first_image=has_first_image, has_last_image=has_last_image,
        )
        physical_routes.append(
            PhysicalEffectiveRoute(
                physical_group=group.physical_group,
                logical_chunks=group.logical_chunks,
                terminal_atomic=group.terminal_atomic,
                routing_status=group.status,
                status="active" if picture_map.references else "empty",
                requested_source_slot_ids=requested,
                picture_map=picture_map,
            )
        )
    return EffectiveReferencePlan(
        mode=routing_schedule.mode,
        legacy_fallback=routing_schedule.legacy_fallback,
        connected_source_slot_ids=source_slot_ids,
        logical_routes=logical_routes,
        physical_routes=tuple(physical_routes),
    )
