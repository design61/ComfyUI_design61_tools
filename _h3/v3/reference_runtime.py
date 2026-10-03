"""V3.9 Custom group-local Reference conditioning.

The V3.9 caller supplies this adapter after routing is resolved. V3.8 and
V3.9 All retain the legacy path; saved-run compatibility is handled by the
separate Reference storage contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import uuid
from typing import Any

import torch

from ..reference import (
    ReferenceAssets,
    ReferenceImageBundle,
    encode_reference_latents,
    encode_reference_latents_cached,
    encode_reference_prompt,
    prepare_reference_assets,
    resolve_reference_image_inputs,
)
from ..v2.h3_builder import encode_prompt_conditioning, encode_prompt_conditioning_cached
from .reference_effective_plan import PictureMap, compile_group_picture_map
from .reference_routing import REFERENCE_SLOT_IDS, ReferenceRoutingSchedule


_REFERENCE_TAG = re.compile(r"(?<![A-Za-z0-9_])@R([1-9])(?![A-Za-z0-9_])")


class ReferenceRuntimeContractError(ValueError):
    """An internal RR-R4 caller supplied an unusable routing contract."""


@dataclass(slots=True)
class ReferenceRoutingRequest:
    """Private V3.9 node-to-V2 request; never part of a saved node schema."""

    selectors_by_slot: dict[str, str]
    inputs: ReferenceInputSet
    runtime: ReferenceRoutingRuntime | None = None


@dataclass(frozen=True, slots=True)
class ReferenceInputSet:
    """Raw R1-R9 sockets; no image is validated, resized, or VAE-encoded here."""

    images: tuple[torch.Tensor | None, ...]
    output_width: int
    output_height: int
    size_mode: str

    @classmethod
    def from_inputs(
        cls,
        *,
        reference_image_1: torch.Tensor | None = None,
        reference_image_2: torch.Tensor | None = None,
        reference_image_3: torch.Tensor | None = None,
        reference_image_4: torch.Tensor | None = None,
        reference_image_5: torch.Tensor | None = None,
        image_references: ReferenceImageBundle | None = None,
        output_width: int,
        output_height: int,
        size_mode: str,
    ) -> ReferenceInputSet:
        images = resolve_reference_image_inputs(
            reference_image_1, reference_image_2, reference_image_3,
            reference_image_4, reference_image_5, image_references,
        )
        return cls(images, int(output_width), int(output_height), str(size_mode))

    @property
    def connected_source_slot_ids(self) -> tuple[str, ...]:
        return tuple(
            slot_id for slot_id, image in zip(REFERENCE_SLOT_IDS, self.images, strict=True)
            if image is not None
        )

    def prepare_selected(self, selected_source_slot_ids: tuple[str, ...]) -> ReferenceAssets | None:
        return prepare_reference_assets(
            reference_image_1=self.images[0],
            reference_image_2=self.images[1],
            reference_image_3=self.images[2],
            image_references=ReferenceImageBundle(self.images[3:]),
            output_width=self.output_width,
            output_height=self.output_height,
            size_mode=self.size_mode,
            selected_source_slot_ids=selected_source_slot_ids,
        )


@dataclass(frozen=True, slots=True)
class GroupReferenceConditioning:
    physical_group: int
    logical_chunks: tuple[int, ...]
    requested_source_slot_ids: tuple[str, ...]
    selected_assets: ReferenceAssets | None
    picture_map: PictureMap
    effective_prompt: str
    conditioning: list[list[Any]]
    warnings: tuple[str, ...]


def rewrite_reference_tags(prompt: str, picture_map: PictureMap) -> tuple[str, tuple[str, ...]]:
    """Rewrite active @R tags only; inactive tags are diagnostic, never blocking."""

    tags = {item.source_slot_id: item.picture_tag for item in picture_map.references}
    inactive: set[str] = set()

    def replace_tag(match: re.Match[str]) -> str:
        slot_id = f"R{match.group(1)}"
        replacement = tags.get(slot_id)
        if replacement is None:
            inactive.add(slot_id)
            return match.group(0)
        return replacement

    effective = _REFERENCE_TAG.sub(replace_tag, str(prompt))
    warnings = tuple(
        f"{slot_id} is not active in this physical group; {slot_id}'s @R tag was left unchanged"
        for slot_id in REFERENCE_SLOT_IDS if slot_id in inactive
    )
    return effective, warnings


def resolve_reference_group_prompt(
    *, physical_group: int, prompt: str,
    requested_source_slot_ids: tuple[str, ...],
    connected_source_slot_ids: tuple[str, ...],
    picture_map: PictureMap,
) -> tuple[str, tuple[str, ...]]:
    """Use the same prompt and warning rules for planned and sampled groups."""

    effective_prompt, prompt_warnings = rewrite_reference_tags(prompt, picture_map)
    connected = set(connected_source_slot_ids)
    disconnected_warnings = tuple(
        f"{slot_id} is routed to physical group {physical_group} but no image is connected"
        for slot_id in requested_source_slot_ids if slot_id not in connected
    )
    return effective_prompt, disconnected_warnings + prompt_warnings


class ReferenceRoutingRuntime:
    """Prepare and encode only the images selected by a physical group."""

    def __init__(self, *, schedule: ReferenceRoutingSchedule, inputs: ReferenceInputSet):
        if not isinstance(schedule, ReferenceRoutingSchedule) or not schedule.valid:
            raise ReferenceRuntimeContractError("RR-R4 requires a valid RR-R2 routing schedule")
        if not isinstance(inputs, ReferenceInputSet) or len(inputs.images) != len(REFERENCE_SLOT_IDS):
            raise ReferenceRuntimeContractError("RR-R4 requires nine fixed raw Reference slots")
        self.schedule = schedule
        self.inputs = inputs
        # RR-R4 has no saved-run/prefix contract. Keep its session identity
        # queue-local without reading images assigned only to later groups.
        self._run_nonce = uuid.uuid4().hex
        self.storage_plan: Any = None
        self.generated_groups: tuple[int, ...] = ()
        self.accepted_chunks = 0
        self.observed_group_contracts: dict[int, dict[str, Any]] = {}
        self.group_warnings: dict[int, tuple[str, ...]] = {}

    def group_contract_for(self, physical_group: int) -> dict[str, Any]:
        group = int(physical_group)
        observed = self.observed_group_contracts.get(group)
        if observed is not None:
            return observed
        if self.storage_plan is not None:
            return self.storage_plan.group_for(group)
        raise ReferenceRuntimeContractError(
            f"physical group {group} has no observed Reference contract"
        )

    def check_physical_contract(self, *, chunks: int, terminal_merge_enabled: bool) -> None:
        if self.schedule.total_chunks != int(chunks) or self.schedule.terminal_merge_enabled != bool(terminal_merge_enabled):
            raise ReferenceRuntimeContractError(
                "RR-R4 routing schedule does not match the current physical group topology"
            )
        if any(route.status == "conflict" for route in self.schedule.physical_routes):
            raise ReferenceRuntimeContractError(
                "RR-R4 cannot select one Reference set for a conflicting terminal physical group"
            )

    @property
    def has_selected_references(self) -> bool:
        connected = set(self.inputs.connected_source_slot_ids)
        return any(
            slot_id in connected
            for route in self.schedule.physical_routes
            for slot_id in (route.reference_slot_ids or ())
        )

    def global_identity(self, keyframe_identity_hash: str) -> str:
        """Queue-local identity; RR-R5 will define persistent prefix reuse."""

        payload = {
            "rr_r4_identity_version": 1,
            "run_nonce": self._run_nonce,
            "keyframe_identity_hash": str(keyframe_identity_hash),
            "mode": self.schedule.mode,
            "logical_routes": [
                (route.logical_chunk, route.reference_slot_ids)
                for route in self.schedule.logical_routes
            ],
            "physical_groups": [
                (route.physical_group, route.logical_chunks)
                for route in self.schedule.physical_routes
            ],
            "connected_source_slot_ids": self.inputs.connected_source_slot_ids,
            "size_mode": self.inputs.size_mode,
            "output_size": (self.inputs.output_width, self.inputs.output_height),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _route(self, physical_group: int):
        for route in self.schedule.physical_routes:
            if route.physical_group == physical_group:
                if route.status == "conflict" or route.reference_slot_ids is None:
                    raise ReferenceRuntimeContractError(
                        f"physical group {physical_group} has no single Reference route"
                    )
                return route
        raise ReferenceRuntimeContractError(f"unknown physical group {physical_group}")

    def prepare_group(
        self,
        *,
        physical_group: int,
        prompt: str,
        clip: Any,
        video_vae: Any,
        first_image: torch.Tensor | None,
        last_image: torch.Tensor | None,
        include_last_image: bool = True,
        reference_audio_assets: Any = None,
        timeline_video_assets: Any = None,
        reference_encode_cache: bool = False,
        cache_event: Any = None,
        prompt_conditioning_cache: bool = False,
        prompt_cache_event: Any = None,
        expected_group_contract: dict[str, Any] | None = None,
        first_frame_hash: str = "none",
        last_frame_hash: str = "none",
        terminal_prompt_policy: str | None = None,
    ) -> GroupReferenceConditioning:
        route = self._route(int(physical_group))
        requested = route.reference_slot_ids
        connected = set(self.inputs.connected_source_slot_ids)
        selected = tuple(slot_id for slot_id in requested if slot_id in connected)
        assets = self.inputs.prepare_selected(selected)
        if assets is not None and assets.source_slot_ids != selected:
            raise ReferenceRuntimeContractError("prepared Reference slots differ from the selected route")
        # The legacy no-Reference path presents Last Image only to the final
        # group. With References, Core's hybrid Qwen presentation includes it
        # in every group even though the Last keyframe is attached only at end.
        presented_last_image = last_image if assets is not None or include_last_image else None
        picture_map = compile_group_picture_map(
            source_slot_ids=assets.source_slot_ids if assets is not None else (),
            reference_image_hashes=assets.image_hashes if assets is not None else (),
            has_first_image=first_image is not None,
            has_last_image=presented_last_image is not None,
        )
        effective_prompt, warnings = resolve_reference_group_prompt(
            physical_group=int(physical_group), prompt=prompt,
            requested_source_slot_ids=requested,
            connected_source_slot_ids=self.inputs.connected_source_slot_ids,
            picture_map=picture_map,
        )
        # Direct RR-R4 callers predate the saved-plan fingerprints. The normal
        # V3.9 Sequence supplies both hashes; for an unsaved direct call only,
        # derive a truthful anchor from the exact image being presented.
        if expected_group_contract is None and self.storage_plan is None:
            from ..v2.h3_builder import _tensor_fingerprint
            if first_image is not None and first_frame_hash == "none":
                first_frame_hash = _tensor_fingerprint(first_image)
            if presented_last_image is not None and last_frame_hash == "none":
                last_frame_hash = _tensor_fingerprint(presented_last_image)
        from .reference_storage_contract import (
            ReferenceStorageContractError, canonical_sha256, descriptor_from_facts,
        )
        descriptor = descriptor_from_facts(
            physical_group=int(physical_group),
            logical_chunks=route.logical_chunks,
            selected_source_slot_ids=assets.source_slot_ids if assets is not None else (),
            prepared_image_hashes=assets.image_hashes if assets is not None else (),
            picture_map=picture_map,
            first_frame_hash=first_frame_hash,
            last_frame_hash=last_frame_hash,
            effective_prompt=effective_prompt,
            size_mode=self.inputs.size_mode,
            terminal_prompt_policy=terminal_prompt_policy,
        )
        group_contract = {"descriptor": descriptor, "sha256": canonical_sha256(descriptor)}
        if expected_group_contract is not None:
            if (canonical_sha256(descriptor) != expected_group_contract.get("sha256")
                    or descriptor != expected_group_contract.get("descriptor")):
                raise ReferenceStorageContractError(
                    f"physical group {physical_group} Reference inputs changed after the "
                    "saved-run preflight; no VAE, Prompt, or Sampling was started"
                )
        if assets is not None:
            assets = (
                encode_reference_latents_cached(video_vae, assets, cache_event=cache_event)
                if reference_encode_cache else encode_reference_latents(video_vae, assets)
            )
            conditioning = encode_reference_prompt(
                clip, effective_prompt, assets,
                first_image=first_image,
                last_image=presented_last_image,
                reference_audio_assets=reference_audio_assets,
                timeline_video_assets=timeline_video_assets,
            )
        elif (prompt_conditioning_cache and not self.has_selected_references
              and reference_audio_assets is None and timeline_video_assets is None
              and terminal_prompt_policy is None):
            # The Sequence caller enables this only for Fixed, reference-free
            # runs. Reuse the V3.8 cache and its conservative CLIP guards; keep
            # route/storage validation above and group metadata below intact.
            conditioning = encode_prompt_conditioning_cached(
                clip, effective_prompt,
                first_image=first_image,
                last_image=presented_last_image,
                first_image_fingerprint=first_frame_hash,
                last_image_fingerprint=(last_frame_hash
                                        if presented_last_image is not None else "none"),
                cache_enabled=True,
                cache_event=prompt_cache_event,
            )
        else:
            conditioning = encode_prompt_conditioning(
                clip, effective_prompt,
                first_image=first_image,
                last_image=presented_last_image,
                reference_audio_assets=reference_audio_assets,
                timeline_video_assets=timeline_video_assets,
            )
        existing = self.observed_group_contracts.get(int(physical_group))
        if existing is not None and existing != group_contract:
            raise ReferenceRuntimeContractError(
                f"physical group {physical_group} Reference contract changed within one Queue"
            )
        self.observed_group_contracts[int(physical_group)] = group_contract
        self.group_warnings[int(physical_group)] = warnings
        return GroupReferenceConditioning(
            physical_group=int(physical_group),
            logical_chunks=route.logical_chunks,
            requested_source_slot_ids=requested,
            selected_assets=assets,
            picture_map=picture_map,
            effective_prompt=effective_prompt,
            conditioning=conditioning,
            warnings=warnings,
        )
