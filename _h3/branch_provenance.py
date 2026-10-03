"""Pure Branch Provenance Contract v1 helpers.

The contract describes immutable physical-group revisions independently from
the sampling contract that produced their tensors.  A display label such as
``Take B`` is deliberately not persisted as identity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


BRANCH_PROVENANCE_VERSION = 1
TAKE_ACTION_AUTOMATIC = "Automatic"
TAKE_ACTION_USE = "Use This Take"
TAKE_ACTION_CONTINUE = "Continue From Here"
TAKE_ACTION_OPTIONS = (
    TAKE_ACTION_AUTOMATIC,
    TAKE_ACTION_USE,
    TAKE_ACTION_CONTINUE,
)


class BranchProvenanceError(ValueError):
    """Raised when persisted branch ancestry is ambiguous or inconsistent."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PhysicalGroup:
    start: int
    end: int
    physical_group: int

    def as_metadata(self) -> dict[str, int]:
        return {
            "start": self.start,
            "end": self.end,
            "physical_group": self.physical_group,
        }


def physical_groups(*, chunks: int, terminal_merge_enabled: bool) -> tuple[PhysicalGroup, ...]:
    if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks < 1:
        raise BranchProvenanceError("chunks must be a positive integer")
    if type(terminal_merge_enabled) is not bool:
        raise BranchProvenanceError("terminal_merge_enabled must be a boolean")
    if terminal_merge_enabled and chunks < 2:
        raise BranchProvenanceError("Terminal Merge requires at least two chunks")
    terminal_start = chunks - 1 if terminal_merge_enabled else None
    groups: list[PhysicalGroup] = []
    chunk = 1
    while chunk <= chunks:
        if terminal_start is not None and chunk == terminal_start:
            groups.append(PhysicalGroup(chunk, chunks, chunk))
            break
        groups.append(PhysicalGroup(chunk, chunk, chunk))
        chunk += 1
    return tuple(groups)


def _record_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    entry = record.get("entry") or {}
    return {
        "sequence_index": int(record.get("sequence_index", -1)),
        "storage_revision_id": str(record.get("storage_revision_id", "")),
        "filename": str(record.get("filename", "")),
        "file_size": int(record.get("file_size", -1)),
        "file_sha256": str(record.get("file_sha256", "")),
        "seed": int(record.get("seed", entry.get("seed", -1))),
        "prompt_hash": str(record.get("prompt_hash", entry.get("prompt_hash", ""))),
    }


def make_group_revision(
    *,
    parent_revision_id: str | None,
    group: PhysicalGroup,
    variation_nonce: int,
    created_utc: str,
    storage_revision_id: str,
    generation_contract_sha256: str,
    lineage_sha256: str,
    records: Iterable[Mapping[str, Any]],
    branch_cut: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(group, PhysicalGroup):
        raise BranchProvenanceError("group must be a PhysicalGroup")
    nonce = int(variation_nonce)
    if nonce < 0 or nonce > 0xFFFFFFFF:
        raise BranchProvenanceError("variation_nonce is outside uint32")
    record_values = [_record_identity(record) for record in records]
    expected_positions = list(range(group.start - 1, group.end))
    if [record["sequence_index"] for record in record_values] != expected_positions:
        raise BranchProvenanceError("physical group records are incomplete or non-atomic")
    if not all(record["storage_revision_id"] for record in record_values):
        raise BranchProvenanceError("physical group record source is missing")
    seed_identity = _hash(
        [
            {
                "sequence_index": record["sequence_index"],
                "seed": record["seed"],
                "prompt_hash": record["prompt_hash"],
            }
            for record in record_values
        ]
    )
    identity = {
        "branch_provenance_version": BRANCH_PROVENANCE_VERSION,
        "parent_revision_id": None if parent_revision_id is None else str(parent_revision_id),
        "group": group.as_metadata(),
        "variation_nonce": nonce,
        "seed_identity": seed_identity,
        "storage_revision_id": str(storage_revision_id),
        "generation_contract_sha256": str(generation_contract_sha256),
        "lineage_sha256": str(lineage_sha256),
        "records": record_values,
    }
    revision_id = _hash(identity)[:24]
    result = {
        "revision_id": revision_id,
        **identity,
        "created_utc": str(created_utc),
        "revision_order": f"{created_utc}|{revision_id}",
    }
    if branch_cut is not None:
        result["branch_cut"] = {
            "selected_revision_id": str(branch_cut.get("selected_revision_id", "")),
            "after_physical_group": int(branch_cut.get("after_physical_group", 0)),
        }
    return result


def validate_group_revision(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BranchProvenanceError("group revision must be an object")
    if int(value.get("branch_provenance_version", -1)) != BRANCH_PROVENANCE_VERSION:
        raise BranchProvenanceError("group revision provenance version is unsupported")
    group_value = value.get("group")
    if not isinstance(group_value, Mapping):
        raise BranchProvenanceError("group revision physical group is missing")
    group = PhysicalGroup(
        start=int(group_value.get("start", 0)),
        end=int(group_value.get("end", 0)),
        physical_group=int(group_value.get("physical_group", 0)),
    )
    if group.start < 1 or group.end < group.start or group.physical_group != group.start:
        raise BranchProvenanceError("group revision physical group is invalid")
    rebuilt = make_group_revision(
        parent_revision_id=value.get("parent_revision_id"),
        group=group,
        variation_nonce=int(value.get("variation_nonce", -1)),
        created_utc=str(value.get("created_utc", "")),
        storage_revision_id=str(value.get("storage_revision_id", "")),
        generation_contract_sha256=str(value.get("generation_contract_sha256", "")),
        lineage_sha256=str(value.get("lineage_sha256", "")),
        records=value.get("records") or [],
        branch_cut=value.get("branch_cut"),
    )
    if str(value.get("revision_id", "")) != rebuilt["revision_id"]:
        raise BranchProvenanceError("group revision immutable identity is invalid")
    if str(value.get("seed_identity", "")) != rebuilt["seed_identity"]:
        raise BranchProvenanceError("group revision seed identity is invalid")
    rebuilt["revision_order"] = str(
        value.get("revision_order", rebuilt["revision_order"])
    )
    return rebuilt


def resolve_chain(
    catalog: Mapping[str, Mapping[str, Any]],
    head_revision_id: str,
) -> list[dict[str, Any]]:
    head = str(head_revision_id)
    if not head:
        raise BranchProvenanceError("selected revision ID is empty")
    reverse: list[dict[str, Any]] = []
    seen: set[str] = set()
    current: str | None = head
    while current is not None:
        if current in seen:
            raise BranchProvenanceError("group revision ancestry contains a cycle")
        seen.add(current)
        raw = catalog.get(current)
        if raw is None:
            raise BranchProvenanceError(
                f"group revision parent is missing: {current}"
            )
        revision = validate_group_revision(raw)
        reverse.append(revision)
        parent = revision.get("parent_revision_id")
        current = None if parent is None else str(parent)
    chain = list(reversed(reverse))
    expected_start = 1
    lineage = str(chain[0].get("lineage_sha256", ""))
    for revision in chain:
        group = revision["group"]
        if int(group["start"]) != expected_start:
            raise BranchProvenanceError("group revision ancestry is not a contiguous prefix")
        if str(revision.get("lineage_sha256", "")) != lineage:
            raise BranchProvenanceError("group revision ancestry crosses sampling lineages")
        expected_start = int(group["end"]) + 1
    return chain


def active_revision_map(chain: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    active: dict[str, str] = {}
    for revision in chain:
        validated = validate_group_revision(revision)
        active[str(validated["group"]["physical_group"])] = validated["revision_id"]
    return active
