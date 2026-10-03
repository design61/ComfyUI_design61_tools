"""MiniMax H3 image-reference adapter for the V3 production sampler.

This module only builds the public Core conditioning contract.  It does not
own model loading, sampling, VAE decoding, or any optimizer implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Any

import torch


REFERENCE_CONTRACT_VERSION = 1
REFERENCE_PREPROCESS_VERSION = 1
HYBRID_PRESENTATION_VERSION = 1
REFERENCE_IMAGES_TYPE = "H3_CONTINUUM_IMAGE_REFERENCES"
REFERENCE_SIZE_MATCH_OUTPUT = "Match Output"
REFERENCE_SIZE_MAX_IDENTITY = "Max Identity"
REFERENCE_SIZE_OPTIONS = (
    REFERENCE_SIZE_MATCH_OUTPUT,
    REFERENCE_SIZE_MAX_IDENTITY,
)
_CANVAS_MULTIPLE = 32
_MAX_IDENTITY_SHORT_EDGE = 2048
_PICTURE_TAG = re.compile(r"<Picture\s+(\d+)>")


class ReferenceConditioningError(ValueError):
    pass


@dataclass(frozen=True)
class ReferenceImageBundle:
    images: tuple[torch.Tensor | None, ...]


def resolve_reference_image_inputs(
    reference_image_1=None,
    reference_image_2=None,
    reference_image_3=None,
    reference_image_4=None,
    reference_image_5=None,
    image_references=None,
) -> tuple[torch.Tensor | None, ...]:
    extra = [None] * 6
    if image_references is not None:
        if not isinstance(image_references, ReferenceImageBundle) or len(image_references.images) != 6:
            raise ReferenceConditioningError("expected a Reference Images bundle for slots 4–9")
        extra = list(image_references.images)
    for index, legacy in enumerate((reference_image_4, reference_image_5)):
        if legacy is not None:
            if extra[index] is not None:
                raise ReferenceConditioningError(
                    f"Reference Image {index + 4} is connected through both the legacy input and the bundle"
                )
            extra[index] = legacy
    return (reference_image_1, reference_image_2, reference_image_3, *extra)


@dataclass(frozen=True)
class ReferenceAssets:
    images: tuple[torch.Tensor, ...]
    latents: tuple[torch.Tensor | None, ...]
    image_hashes: tuple[str, ...]
    combined_hash: str
    size_mode: str
    # Internal source identity, aligned with the compact images/hash/latent tuples.
    # An empty tuple keeps manually constructed legacy assets compatible.
    source_slot_ids: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return len(self.images)

    @property
    def contract(self) -> dict[str, Any]:
        result = {
            "reference_contract_version": REFERENCE_CONTRACT_VERSION,
            "count": self.count,
            "size_mode": self.size_mode,
            "image_hashes": list(self.image_hashes),
            "combined_hash": self.combined_hash,
        }
        # Keep the existing zero-to-three-reference contracts unchanged.
        for index in range(2, self.count):
            image = self.images[index]
            result[f"reference_image_{index + 1}"] = {
                "reference_position": index + 1,
                "shape": [int(value) for value in image.shape],
                "dtype": str(image.dtype),
                "sha256": self.image_hashes[index],
                "preprocess_version": REFERENCE_PREPROCESS_VERSION,
            }
        return result


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(int(v) for v in tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _round_canvas(value: float) -> int:
    return max(_CANVAS_MULTIPLE, int(round(value / _CANVAS_MULTIPLE)) * _CANVAS_MULTIPLE)


def _validate_image(name: str, image: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(image) or image.ndim != 4:
        raise ReferenceConditioningError(f"{name} must be a ComfyUI IMAGE tensor")
    if int(image.shape[0]) < 1 or int(image.shape[1]) < 1 or int(image.shape[2]) < 1:
        raise ReferenceConditioningError(f"{name} is empty")
    if int(image.shape[0]) != 1:
        raise ReferenceConditioningError(
            f"{name} must contain exactly one image (batch size B=1)"
        )
    if int(image.shape[-1]) < 3:
        raise ReferenceConditioningError(f"{name} must have RGB channels")
    return image[:, :, :, :3]


def resolve_reference_image_size(
    source_width: int,
    source_height: int,
    *,
    output_width: int,
    output_height: int,
    size_mode: str,
) -> tuple[int, int]:
    """Resolve Reference Image geometry without resizing or reading pixels."""

    if size_mode not in REFERENCE_SIZE_OPTIONS:
        raise ReferenceConditioningError(f"unknown Reference Size: {size_mode!r}")
    source_width = int(source_width)
    source_height = int(source_height)
    if size_mode == REFERENCE_SIZE_MATCH_OUTPUT:
        target_area = float(int(output_width) * int(output_height))
        scale = min(1.0, math.sqrt(target_area / float(source_width * source_height)))
    else:
        scale = min(
            1.0,
            float(_MAX_IDENTITY_SHORT_EDGE) / min(source_width, source_height),
        )
    return (
        _round_canvas(source_width * scale),
        _round_canvas(source_height * scale),
    )


def _resize_reference(
    image: torch.Tensor,
    *,
    output_width: int,
    output_height: int,
    size_mode: str,
) -> torch.Tensor:
    source_height = int(image.shape[1])
    source_width = int(image.shape[2])
    target_width, target_height = resolve_reference_image_size(
        source_width,
        source_height,
        output_width=int(output_width),
        output_height=int(output_height),
        size_mode=size_mode,
    )
    if target_width == source_width and target_height == source_height:
        return image.contiguous()
    try:
        import comfy.utils
    except Exception as exc:
        raise ReferenceConditioningError(
            "ComfyUI Core image resize support is unavailable"
        ) from exc
    channels_first = image.movedim(-1, 1)
    resized = comfy.utils.common_upscale(
        channels_first,
        target_width,
        target_height,
        "lanczos",
        "disabled",
    )
    return resized.movedim(1, -1).contiguous()


def prepare_reference_assets(
    *,
    reference_image_1: torch.Tensor | None,
    reference_image_2: torch.Tensor | None,
    output_width: int,
    output_height: int,
    size_mode: str,
    reference_image_3: torch.Tensor | None = None,
    reference_image_4: torch.Tensor | None = None,
    reference_image_5: torch.Tensor | None = None,
    image_references: ReferenceImageBundle | None = None,
    selected_source_slot_ids: tuple[str, ...] | None = None,
) -> ReferenceAssets | None:
    # Match Core's dynamic Reference inputs: bypassed or otherwise absent
    # sockets are ignored, then active images are numbered contiguously in
    # connection order as Picture 1..N.
    resolved_inputs = resolve_reference_image_inputs(
        reference_image_1, reference_image_2, reference_image_3,
        reference_image_4, reference_image_5, image_references,
    )
    selected_slots = (
        None if selected_source_slot_ids is None else set(selected_source_slot_ids)
    )
    active_slots = [
        (f"R{slot_number}", image)
        for slot_number, image in enumerate(resolved_inputs, start=1)
        if image is not None
        and (selected_slots is None or f"R{slot_number}" in selected_slots)
    ]
    if not active_slots:
        return None
    images = tuple(
        _resize_reference(
            _validate_image(f"Reference Image {index}", image),
            output_width=int(output_width),
            output_height=int(output_height),
            size_mode=size_mode,
        )
        for index, (_, image) in enumerate(active_slots, start=1)
    )
    image_hashes = tuple(_tensor_hash(image) for image in images)
    combined_hash = _canonical_hash(
        {
            "reference_contract_version": REFERENCE_CONTRACT_VERSION,
            "size_mode": size_mode,
            "image_hashes": list(image_hashes),
        }
    )
    return ReferenceAssets(
        images=images,
        latents=tuple(None for _ in images),
        image_hashes=image_hashes,
        combined_hash=combined_hash,
        size_mode=size_mode,
        source_slot_ids=tuple(slot_id for slot_id, _ in active_slots),
    )


def encode_reference_latents(video_vae: Any, assets: ReferenceAssets) -> ReferenceAssets:
    if all(latent is not None for latent in assets.latents):
        return assets
    latents = tuple(video_vae.encode(image) for image in assets.images)
    return replace(assets, latents=latents)


def encode_reference_latents_cached(
    video_vae: Any,
    assets: ReferenceAssets,
    *,
    cache_event=None,
) -> ReferenceAssets:
    """Encode V3.8 Reference Images with the process-local VAE cache."""

    if all(latent is not None for latent in assets.latents):
        return assets
    from .v3.ref_encode_cache import (
        get_ref_encode_cache,
        make_ref_encode_cache_key,
    )

    cache = get_ref_encode_cache()
    latents: list[torch.Tensor] = []
    for image, image_hash, existing in zip(
        assets.images,
        assets.image_hashes,
        assets.latents,
        strict=True,
    ):
        if existing is not None:
            latents.append(existing)
            continue
        key = make_ref_encode_cache_key(
            "reference_image",
            REFERENCE_PREPROCESS_VERSION,
            image_hash,
        )
        latent = cache.lookup(video_vae, key, event_sink=cache_event)
        if latent is None:
            latent = video_vae.encode(image)
            if cache.supports_vae(video_vae):
                cache.store(video_vae, key, latent, event_sink=cache_event)
        latents.append(latent)
    return replace(assets, latents=tuple(latents))


def build_hybrid_presentation_items(
    reference_items: list[dict[str, Any]],
    *,
    first_image: torch.Tensor | None = None,
    last_image: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    """Build the Core Qwen Picture list without changing DiT payloads."""
    items: list[dict[str, Any]] = []
    if first_image is not None:
        items.append({"type": "image", "data": first_image})
    if last_image is not None:
        items.append({"type": "image", "data": last_image})
    items.extend(dict(item) for item in reference_items)
    return items


def combine_hybrid_visual_identity(
    *,
    keyframe_identity_hash: str,
    reference_assets: ReferenceAssets | None,
    has_first: bool,
    has_last: bool,
) -> str:
    """Preserve pure-mode identity and version only the hybrid contract."""
    if reference_assets is None:
        return str(keyframe_identity_hash)
    if not has_first and not has_last:
        return reference_assets.combined_hash
    return _canonical_hash(
        {
            "hybrid_visual_identity_version": 1,
            "hybrid_presentation_version": HYBRID_PRESENTATION_VERSION,
            "keyframe_identity_hash": str(keyframe_identity_hash),
            "reference_hash": reference_assets.combined_hash,
        }
    )


def validate_reference_prompts(
    prompts: list[str],
    reference_count: int,
    picture_offset: int = 0,
) -> str:
    reference_count = max(0, int(reference_count))
    picture_offset = max(0, int(picture_offset))
    found: set[int] = set()
    for prompt in prompts:
        found.update(int(value) for value in _PICTURE_TAG.findall(str(prompt)))

    first_reference = picture_offset + 1
    last_reference = picture_offset + reference_count
    found_references = {
        value for value in found if first_reference <= value <= last_reference
    }
    invalid = sorted(value for value in found if value < 1 or value > last_reference)
    warnings: list[str] = []
    if invalid:
        unavailable = ", ".join(f"<Picture {value}>" for value in invalid)
        warnings.append(
            f"H3C-P102 Warning: prompt references unavailable {unavailable}; only "
            f"{reference_count} active reference image(s) reached the Sampler. "
            "ComfyUI Core permits this and generation will continue, but the "
            "unavailable reference may be ignored or hallucinated. Enable the "
            "corresponding Reference Image input or remove the unavailable tag."
        )

    expected_references = set(range(first_reference, last_reference + 1))
    missing = sorted(expected_references - found_references)
    if missing:
        labels = ", ".join(f"<Picture {value}>" for value in missing)
        detail = (
            "the prompt contains no <Picture N> tag"
            if picture_offset == 0 and not found
            else f"connected reference {labels} is not explicitly mentioned in the prompt"
        )
        warnings.append(
            f"H3C-P103 Warning: {detail}; images still condition generation, but "
            "explicit identity references are recommended."
        )
    return "\n".join(warnings)


def encode_reference_prompt(
    clip: Any,
    prompt: str,
    assets: ReferenceAssets,
    first_image: torch.Tensor | None = None,
    last_image: torch.Tensor | None = None,
    reference_audio_assets=None,
    timeline_video_assets=None,
) -> list[list[Any]]:
    if any(latent is None for latent in assets.latents):
        raise ReferenceConditioningError("reference latents were not encoded")
    reference_items = [
        {"type": "image", "data": image}
        for image in assets.images
    ]
    if timeline_video_assets is not None:
        reference_items.append(dict(timeline_video_assets.item))
    if reference_audio_assets is not None:
        from .reference_audio import (
            reference_audio_asset_items,
            reference_audio_item,
        )

        reference_items.extend(
            reference_audio_item()
            for _ in reference_audio_asset_items(reference_audio_assets)
        )
    presentation_items = build_hybrid_presentation_items(
        reference_items,
        first_image=first_image,
        last_image=last_image,
    )
    try:
        tokens = clip.tokenize(str(prompt), minimax_ref_items=presentation_items)
    except TypeError as exc:
        raise ReferenceConditioningError(
            "This ComfyUI Core/CLIP does not support MiniMax H3 image references"
        ) from exc
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    refs = []
    for image, latent in zip(assets.images, assets.latents, strict=True):
        refs.append(
            {
                "kind": "image",
                "latent_h": int(image.shape[1]) // 16,
                "latent_w": int(image.shape[2]) // 16,
                "latent": latent,
            }
        )
    if timeline_video_assets is not None:
        refs.append(dict(timeline_video_assets.block))
    if reference_audio_assets is not None:
        from .reference_audio import (
            reference_audio_asset_items,
            reference_audio_block,
        )

        refs.extend(
            dict(reference_audio_block(item))
            for item in reference_audio_asset_items(reference_audio_assets)
        )
    return [
        [tensor, {**dict(metadata), "minimax_refs": list(refs)}]
        for tensor, metadata in conditioning
    ]


class H3ContinuumReferenceImages:
    CATEGORY = "MiniMax H3/Continuum/Input"
    DESCRIPTION = "Bundle optional Reference Images 4–9 for the Continuum Sampler. Encoding remains in the Sampler."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                f"reference_image_{index}": (
                    "IMAGE",
                    {"display_name": f"Reference Image {index} (Optional)",
                     "tooltip": "One optional appearance or identity image. Empty slots are skipped in prompt order."},
                )
                for index in range(4, 10)
            }
        }

    RETURN_TYPES = (REFERENCE_IMAGES_TYPE,)
    RETURN_NAMES = ("image_references",)
    FUNCTION = "pack"

    def pack(
        self, reference_image_4=None, reference_image_5=None,
        reference_image_6=None, reference_image_7=None,
        reference_image_8=None, reference_image_9=None,
    ):
        images = (
            reference_image_4, reference_image_5, reference_image_6,
            reference_image_7, reference_image_8, reference_image_9,
        )
        return (ReferenceImageBundle(images) if any(image is not None for image in images) else None,)
