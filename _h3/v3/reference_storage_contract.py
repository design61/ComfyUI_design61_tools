"""Private RR-R5 physical-group Reference persistence contract.

The stored descriptor records *effective* conditioning, not selector spelling.
No VAE, CLIP, Sampling, or saved-run operation is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

from ..branch_provenance import physical_groups
from ..reference import HYBRID_PRESENTATION_VERSION, REFERENCE_PREPROCESS_VERSION
from .reference_effective_plan import PictureMap, compile_group_picture_map
from .reference_routing import REFERENCE_SLOT_IDS


REFERENCE_ROUTING_CONTRACT_VERSION = 1
SESSION_ROUTING_IDENTITY_VERSION = 1
_HEX = frozenset("0123456789abcdef")


class ReferenceStorageContractError(ValueError):
    """A routed producer or accepted-prefix contract cannot be trusted."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def routed_session_identity(base_identity_hash: str) -> str:
    return canonical_sha256({
        "rr_r5_session_identity_version": SESSION_ROUTING_IDENTITY_VERSION,
        "base_identity_hash": str(base_identity_hash),
    })


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX for c in value)


def _anchor(value: str, present: bool) -> str | None:
    if not present:
        return None
    if not _valid_sha(value):
        raise ReferenceStorageContractError("routed First/Last image fingerprint is invalid")
    return value


def _present(value: str) -> bool:
    if str(value).lower() in {"", "none", "null"}:
        return False
    if not _valid_sha(value):
        raise ReferenceStorageContractError("routed First/Last image fingerprint is invalid")
    return True


def descriptor_from_facts(
    *, physical_group: int, logical_chunks: tuple[int, ...],
    selected_source_slot_ids: tuple[str, ...],
    prepared_image_hashes: tuple[str, ...], picture_map: PictureMap,
    first_frame_hash: str, last_frame_hash: str,
    effective_prompt: str, size_mode: str,
    terminal_prompt_policy: str | None,
) -> dict[str, Any]:
    if len(selected_source_slot_ids) != len(prepared_image_hashes):
        raise ReferenceStorageContractError("selected Reference slots and image hashes differ")
    if picture_map.source_slot_ids != selected_source_slot_ids:
        raise ReferenceStorageContractError("Reference Picture map differs from selected source slots")
    if any(slot not in REFERENCE_SLOT_IDS for slot in selected_source_slot_ids):
        raise ReferenceStorageContractError("selected Reference slot is unknown")
    if tuple(sorted(selected_source_slot_ids, key=REFERENCE_SLOT_IDS.index)) != selected_source_slot_ids:
        raise ReferenceStorageContractError("selected Reference slots are not in fixed R1-R9 order")
    if len(set(selected_source_slot_ids)) != len(selected_source_slot_ids):
        raise ReferenceStorageContractError("selected Reference slot is duplicated")
    if any(not _valid_sha(value) for value in prepared_image_hashes):
        raise ReferenceStorageContractError("prepared Reference image hash is invalid")
    if any(item.image_sha256 != prepared_image_hashes[index]
           for index, item in enumerate(picture_map.references)):
        raise ReferenceStorageContractError("Reference Picture map image hash is inconsistent")
    if not logical_chunks or tuple(range(logical_chunks[0], logical_chunks[-1] + 1)) != logical_chunks:
        raise ReferenceStorageContractError("physical group logical chunks are not contiguous")
    if int(physical_group) != logical_chunks[0]:
        raise ReferenceStorageContractError("physical group number differs from its first logical chunk")
    if (len(logical_chunks) > 1) != (terminal_prompt_policy is not None):
        raise ReferenceStorageContractError("terminal prompt policy does not match physical group")
    first_present = picture_map.first_picture_number is not None
    last_present = picture_map.last_picture_number is not None
    order = []
    if first_present:
        order.append("first_frame")
    if last_present:
        order.append("last_frame")
    order.extend(selected_source_slot_ids)
    descriptor: dict[str, Any] = {
        "version": REFERENCE_ROUTING_CONTRACT_VERSION,
        "physical_group": int(physical_group),
        "logical_chunks": list(logical_chunks),
        "terminal_atomic": len(logical_chunks) > 1,
        "references": [
            {
                "source_slot_id": slot_id,
                "prepared_sha256": prepared_image_hashes[index],
                "picture_number": picture_map.references[index].picture_number,
            }
            for index, slot_id in enumerate(selected_source_slot_ids)
        ],
        "first_anchor": _anchor(first_frame_hash, first_present),
        "last_anchor": _anchor(last_frame_hash, last_present),
        "picture_order": order,
        "picture_presentation_version": HYBRID_PRESENTATION_VERSION,
        "effective_prompt_sha256": hashlib.sha256(str(effective_prompt).encode("utf-8")).hexdigest(),
        "terminal_prompt_policy": str(terminal_prompt_policy) if len(logical_chunks) > 1 else None,
    }
    if selected_source_slot_ids:
        descriptor["reference_size_mode"] = str(size_mode)
        descriptor["reference_preprocess_version"] = REFERENCE_PREPROCESS_VERSION
    return descriptor


@dataclass(frozen=True, slots=True)
class ReferenceStoragePlan:
    """Canonical JSON is immutable even if the caller's source dicts are not."""

    descriptor_jsons: tuple[str, ...]
    routing_config_sha256: str
    requested_routes: tuple[tuple[str, ...], ...]
    _group_index: dict[int, int] = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        index: dict[int, int] = {}
        for position, payload in enumerate(self.descriptor_jsons):
            descriptor = json.loads(payload)
            # Preserve the former first-match lookup for duplicate group IDs.
            index.setdefault(int(descriptor["physical_group"]), position)
        object.__setattr__(self, "_group_index", index)

    def _record_at(self, position: int) -> dict[str, Any]:
        payload = self.descriptor_jsons[position]
        return {
            "descriptor": json.loads(payload),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }

    @property
    def group_contracts(self) -> list[dict[str, Any]]:
        return [self._record_at(position) for position in range(len(self.descriptor_jsons))]

    def group_for(self, physical_group: int) -> dict[str, Any]:
        position = self._group_index.get(int(physical_group))
        if position is None:
            raise ReferenceStorageContractError(
                f"physical group {physical_group} has no frozen Reference contract"
            )
        return self._record_at(position)

    def audit(self) -> dict[str, Any]:
        return {
            "version": REFERENCE_ROUTING_CONTRACT_VERSION,
            "routing_config_sha256": self.routing_config_sha256,
            "requested_routes": [list(route) for route in self.requested_routes],
        }


def build_reference_storage_plan(
    *, runtime: Any, prompts: list[str], first_frame_hash: str,
    last_frame_hash: str, terminal_prompt: str | None = None,
    terminal_prompt_policy: str | None = None,
    through_chunk: int | None = None,
) -> ReferenceStoragePlan:
    """Prehash only connected R slots selected by the requested group prefix."""
    from .reference_runtime import ReferenceRoutingRuntime, resolve_reference_group_prompt

    if not isinstance(runtime, ReferenceRoutingRuntime):
        raise ReferenceStorageContractError("routed storage requires ReferenceRoutingRuntime")
    schedule = runtime.schedule
    total = int(schedule.total_chunks or 0)
    if total < 1 or len(prompts) != total:
        raise ReferenceStorageContractError("Reference schedule and prompt count differ")
    runtime.check_physical_contract(
        chunks=total, terminal_merge_enabled=bool(schedule.terminal_merge_enabled),
    )
    limit = total if through_chunk is None else int(through_chunk)
    if limit < 1 or limit > total:
        raise ReferenceStorageContractError("Reference plan prefix is outside configured chunks")
    routes = [route for route in schedule.physical_routes if route.logical_chunks[-1] <= limit]
    if not routes or routes[-1].logical_chunks[-1] != limit:
        raise ReferenceStorageContractError("Reference plan prefix ends inside a physical group")
    connected = set(runtime.inputs.connected_source_slot_ids)
    selected = {slot for route in routes for slot in (route.reference_slot_ids or ()) if slot in connected}
    image_hashes: dict[str, str] = {}
    for slot in REFERENCE_SLOT_IDS:
        if slot not in selected:
            continue
        assets = runtime.inputs.prepare_selected((slot,))
        if assets is None or assets.source_slot_ids != (slot,) or len(assets.image_hashes) != 1:
            raise ReferenceStorageContractError(f"selected {slot} cannot be prepared")
        image_hashes[slot] = assets.image_hashes[0]
        del assets
    descriptor_jsons: list[str] = []
    requested_routes: list[tuple[str, ...]] = []
    pending_group_warnings: dict[int, tuple[str, ...]] = {}
    for route in routes:
        requested = route.reference_slot_ids
        if requested is None:
            raise ReferenceStorageContractError("conflicting terminal Reference route")
        requested_routes.append(requested)
        source_slots = tuple(slot for slot in requested if slot in connected)
        hashes = tuple(image_hashes[slot] for slot in source_slots)
        final = route.logical_chunks[-1] == total
        has_first = _present(first_frame_hash)
        has_last = _present(last_frame_hash) and (bool(source_slots) or final)
        picture_map = compile_group_picture_map(
            source_slot_ids=source_slots,
            reference_image_hashes=hashes,
            has_first_image=has_first,
            has_last_image=has_last,
        )
        prompt = (
            terminal_prompt if len(route.logical_chunks) > 1
            else prompts[route.logical_chunks[0] - 1]
        )
        if prompt is None:
            raise ReferenceStorageContractError("terminal physical prompt is unavailable")
        effective_prompt, warnings = resolve_reference_group_prompt(
            physical_group=int(route.physical_group), prompt=prompt,
            requested_source_slot_ids=requested,
            connected_source_slot_ids=runtime.inputs.connected_source_slot_ids,
            picture_map=picture_map,
        )
        pending_group_warnings[int(route.physical_group)] = warnings
        descriptor = descriptor_from_facts(
            physical_group=route.physical_group,
            logical_chunks=route.logical_chunks,
            selected_source_slot_ids=source_slots,
            prepared_image_hashes=hashes,
            picture_map=picture_map,
            first_frame_hash=first_frame_hash,
            last_frame_hash=last_frame_hash,
            effective_prompt=effective_prompt,
            size_mode=runtime.inputs.size_mode,
            terminal_prompt_policy=terminal_prompt_policy if len(route.logical_chunks) > 1 else None,
        )
        descriptor_jsons.append(canonical_json(descriptor))
    routing_config_sha256 = canonical_sha256({
        "version": REFERENCE_ROUTING_CONTRACT_VERSION,
        "mode": schedule.mode,
        "logical_routes": [
            [route.logical_chunk, list(route.reference_slot_ids)]
            for route in schedule.logical_routes
        ],
    })
    plan = ReferenceStoragePlan(
        tuple(descriptor_jsons), routing_config_sha256, tuple(requested_routes)
    )
    runtime.group_warnings.update({
        group: warnings for group, warnings in pending_group_warnings.items()
        if group not in runtime.observed_group_contracts
    })
    return plan


def validate_reference_group_contracts(
    records: Any, *, chunk_count: int, terminal_merge_enabled: bool,
) -> list[dict[str, Any]]:
    """Validate the complete routed stored plan, not just its declared hashes."""
    expected = physical_groups(chunks=chunk_count, terminal_merge_enabled=terminal_merge_enabled)
    if not isinstance(records, list) or len(records) != len(expected):
        raise ReferenceStorageContractError("stored Reference group count does not match physical topology")
    clean: list[dict[str, Any]] = []
    for record, group in zip(records, expected):
        if not isinstance(record, dict) or not isinstance(record.get("descriptor"), dict):
            raise ReferenceStorageContractError("stored Reference group descriptor is missing")
        descriptor = record["descriptor"]
        if descriptor.get("version") != REFERENCE_ROUTING_CONTRACT_VERSION:
            raise ReferenceStorageContractError("stored Reference group version is unsupported")
        if (descriptor.get("physical_group") != group.physical_group
                or descriptor.get("logical_chunks") != list(range(group.start, group.end + 1))
                or descriptor.get("terminal_atomic") is not (group.start != group.end)):
            raise ReferenceStorageContractError("stored Reference group boundary is invalid")
        if not _valid_sha(record.get("sha256")) or canonical_sha256(descriptor) != record["sha256"]:
            raise ReferenceStorageContractError("stored Reference group SHA-256 is invalid")
        refs = descriptor.get("references")
        if not isinstance(refs, list):
            raise ReferenceStorageContractError("stored Reference group source list is invalid")
        slots = [ref.get("source_slot_id") for ref in refs if isinstance(ref, dict)]
        if (len(slots) != len(refs) or any(slot not in REFERENCE_SLOT_IDS for slot in slots)
                or slots != sorted(set(slots), key=REFERENCE_SLOT_IDS.index)
                or any(not _valid_sha(ref.get("prepared_sha256")) for ref in refs)):
            raise ReferenceStorageContractError("stored Reference group source identity is invalid")
        expected_order = (["first_frame"] if descriptor.get("first_anchor") else [])
        expected_order += (["last_frame"] if descriptor.get("last_anchor") else [])
        expected_order += slots
        if descriptor.get("picture_order") != expected_order:
            raise ReferenceStorageContractError("stored Reference Picture order is invalid")
        if descriptor.get("picture_presentation_version") != HYBRID_PRESENTATION_VERSION:
            raise ReferenceStorageContractError("stored Reference Picture presentation version is invalid")
        offset = int(descriptor.get("first_anchor") is not None) + int(
            descriptor.get("last_anchor") is not None
        )
        if any(ref.get("picture_number") != offset + index + 1
               for index, ref in enumerate(refs)):
            raise ReferenceStorageContractError("stored Reference Picture numbers are invalid")
        if (group.start != group.end) != (descriptor.get("terminal_prompt_policy") is not None):
            raise ReferenceStorageContractError("stored terminal prompt policy is invalid")
        if any(not _valid_sha(descriptor.get(name)) for name in ("effective_prompt_sha256",)):
            raise ReferenceStorageContractError("stored Reference effective Prompt hash is invalid")
        if any(descriptor.get(name) is not None and not _valid_sha(descriptor[name])
               for name in ("first_anchor", "last_anchor")):
            raise ReferenceStorageContractError("stored Reference anchor hash is invalid")
        if refs and (descriptor.get("reference_preprocess_version") != REFERENCE_PREPROCESS_VERSION
                     or not isinstance(descriptor.get("reference_size_mode"), str)):
            raise ReferenceStorageContractError("stored Reference preprocessing contract is invalid")
        if not refs and ("reference_preprocess_version" in descriptor or "reference_size_mode" in descriptor):
            raise ReferenceStorageContractError("empty Reference group has preprocessing metadata")
        clean.append(record)
    return clean


def session_routing_settings(plan: ReferenceStoragePlan, accepted_chunks: int) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    for record in plan.group_contracts:
        descriptor = record["descriptor"]
        end = int(descriptor["logical_chunks"][-1])
        if end > accepted_chunks:
            break
        accepted.append({
            "physical_group": int(descriptor["physical_group"]),
            "logical_chunks": list(descriptor["logical_chunks"]),
            "sha256": record["sha256"],
        })
    if not accepted or accepted[-1]["logical_chunks"][-1] != accepted_chunks:
        raise ReferenceStorageContractError("accepted Session ends inside a physical group")
    return {
        "version": REFERENCE_ROUTING_CONTRACT_VERSION,
        "accepted_groups": accepted,
        "routing_config_sha256": plan.routing_config_sha256,
    }


def compatible_session_prefix(
    evidence: Any, plan: ReferenceStoragePlan, *, saved_chunks: int,
) -> int:
    """Return a complete-group accepted prefix; malformed evidence is rejected."""
    if not isinstance(evidence, dict) or evidence.get("version") != REFERENCE_ROUTING_CONTRACT_VERSION:
        raise ReferenceStorageContractError("saved Session has no RR-R5 group evidence")
    groups = evidence.get("accepted_groups")
    if not isinstance(groups, list) or not groups:
        raise ReferenceStorageContractError("saved Session group evidence is missing")
    last_end = 0
    for position, item in enumerate(groups):
        if not isinstance(item, dict) or not _valid_sha(item.get("sha256")):
            raise ReferenceStorageContractError("saved Session group hash is invalid")
        chunks = item.get("logical_chunks")
        if (not isinstance(chunks, list) or not chunks or chunks[0] != last_end + 1
                or chunks != list(range(chunks[0], chunks[-1] + 1))
                or item.get("physical_group") != chunks[0]):
            raise ReferenceStorageContractError("saved Session group boundaries are invalid")
        last_end = chunks[-1]
    if last_end != saved_chunks:
        raise ReferenceStorageContractError("saved Session group evidence does not cover its chunks")
    preserved = 0
    current = plan.group_contracts
    for position, item in enumerate(groups):
        if position >= len(current):
            break
        now = current[position]
        if (item["physical_group"] != now["descriptor"]["physical_group"]
                or item["logical_chunks"] != now["descriptor"]["logical_chunks"]
                or item["sha256"] != now["sha256"]):
            break
        preserved = item["logical_chunks"][-1]
    return preserved
