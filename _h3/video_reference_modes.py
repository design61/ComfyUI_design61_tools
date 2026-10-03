"""Experimental Follow Timeline over the existing 24-fps IMAGE input.

Repeat Reference deliberately continues through reference_video.py unchanged.
A Follow source has a different identity, but its identity does not depend on
Chunks. A physical window is selected from the existing retained-frame plan,
not a Queue-local counter. No sampling, continuation or audio code lives here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import torch

from .reference_video import (
    REFERENCE_VIDEO_PREPROCESS_VERSION,
    REFERENCE_VIDEO_SIZE_EFFICIENT,
    REFERENCE_VIDEO_SIZE_OPTIONS,
    ReferenceVideoAssets,
    ReferenceVideoError,
    ReferenceVideoSource,
    _canonical_hash,
    _prepare_reference_frames,
    encode_reference_video,
    encode_reference_video_cached,
    resolve_reference_video_size,
)
from .temporal import align_frame_count_up

REPEAT_REFERENCE = "Repeat Reference"
FOLLOW_TIMELINE = "Follow Timeline"
VIDEO_REFERENCE_MODES = (FOLLOW_TIMELINE, REPEAT_REFERENCE)
FOLLOW_VERSION = 1
FPS = 24
FOLLOW_PLAN_KEY = "_h3_continuum_video_reference_v1"
WINDOW_POLICY = "natural_visible_frames_excluding_context_v1"
END_POLICY = "remaining_interval_then_no_reference_v1"


def mode_input_definition() -> tuple:
    """Append this widget; never insert it into the old serialized prefix."""
    return (
        VIDEO_REFERENCE_MODES,
        {
            "default": REPEAT_REFERENCE,
            "display_name": "Video Reference Mode",
            "tooltip": (
                "Experimental. Follow Timeline selects the source interval for each "
                "physical generation group, then continues without video reference "
                "after the source ends. Repeat Reference preserves the previous "
                "Video Guide behavior: the same bounded prefix for every group. "
                "This is independent of Prompt Format. Input frames must be 24 fps."
            ),
        },
    )


@dataclass(frozen=True)
class FollowVideoSource:
    """Run-scoped, normalized source. Do not mutate frames after preparation."""

    frames: torch.Tensor
    source_shape: tuple[int, ...]
    source_dtype: str
    source_sha256: str
    size_mode: str
    target_width: int
    target_height: int
    chunk_seconds: float
    combined_hash: str

    @property
    def frame_count(self) -> int:
        return int(self.source_shape[0])

    @property
    def contract(self) -> dict[str, Any]:
        # No configured Chunks or total duration: adding future groups must not
        # invalidate an otherwise unchanged saved prefix. Mode is nevertheless
        # part of the reference identity, so Follow and Repeat never collide.
        return {
            "follow_video_contract_version": FOLLOW_VERSION,
            "mode": FOLLOW_TIMELINE,
            "source_shape": list(self.source_shape),
            "source_dtype": self.source_dtype,
            "source_sha256": self.source_sha256,
            "source_fps": FPS,
            "size_mode": self.size_mode,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "chunk_seconds": self.chunk_seconds,
            "preprocess_version": REFERENCE_VIDEO_PREPROCESS_VERSION,
            "window_policy": WINDOW_POLICY,
            "end_policy": END_POLICY,
            "combined_hash": self.combined_hash,
        }


@dataclass(frozen=True)
class FollowVideoWindow:
    """Half-open source frame range, before H3's short alignment padding."""

    requested_start: int
    requested_stop: int
    source_start: int
    source_stop: int
    encoded_frames: int
    source_frames: int
    identity: str

    @property
    def available_frames(self) -> int:
        return self.source_stop - self.source_start

    @property
    def active(self) -> bool:
        return self.available_frames >= 5

    @property
    def status(self) -> str:
        if self.available_frames == 0:
            return "source_exhausted"
        if not self.active:
            return "short_remainder_skipped"
        if self.source_stop < self.requested_stop:
            return "partial_source"
        return "active"

    @property
    def contract(self) -> dict[str, Any]:
        return {
            "version": FOLLOW_VERSION,
            "mode": FOLLOW_TIMELINE,
            "requested_start_frame": self.requested_start,
            "requested_stop_frame": self.requested_stop,
            "source_start_frame": self.source_start,
            "source_stop_frame": self.source_stop,
            "source_frames": self.source_frames,
            "encoded_frames": self.encoded_frames,
            "padding_frames": self.encoded_frames - self.available_frames if self.active else 0,
            "fps": FPS,
            "status": self.status,
            "slice_sha256": self.identity,
        }

    def report(self) -> str:
        if not self.active:
            return (
                f"Timeline Video: output start={self.requested_start / FPS:.3f}s; "
                f"{self.status}; no video reference for this physical group."
            )
        return (
            "Timeline Video: source "
            f"[{self.source_start / FPS:.3f}, {self.source_stop / FPS:.3f})s "
            f"({self.available_frames} frames, H3={self.encoded_frames}); {self.status}."
        )


def prepare_follow_video_source(
    frames: torch.Tensor,
    *,
    chunk_seconds: float,
    output_width: int,
    output_height: int,
    size_mode: str = REFERENCE_VIDEO_SIZE_EFFICIENT,
) -> FollowVideoSource:
    """Keep the full normalized source for future windows, without a GPU copy.

    This IMAGE-input implementation is not a lazy media-file reader. Its CPU
    source copy is intentional: hashing and encoding must see the same pixels.
    It makes no claim of reducing long-input RAM use.
    """
    if not torch.is_tensor(frames) or frames.ndim != 4 or int(frames.shape[-1]) < 3:
        raise ReferenceVideoError("Timeline Video Frames requires IMAGE [T,H,W,C] with RGB")
    if int(frames.shape[1]) < 1 or int(frames.shape[2]) < 1:
        raise ReferenceVideoError("Timeline Video Frames has empty spatial dimensions")
    seconds = float(chunk_seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid internal chunk duration for Timeline Video")
    mode = str(size_mode) if str(size_mode) in REFERENCE_VIDEO_SIZE_OPTIONS else REFERENCE_VIDEO_SIZE_EFFICIENT
    normalized, shape, dtype, fingerprint = _prepare_reference_frames(
        frames, frame_count=int(frames.shape[0])
    )
    width, height = resolve_reference_video_size(
        int(shape[2]), int(shape[1]),
        output_width=int(output_width), output_height=int(output_height), size_mode=mode,
    )
    descriptor = {
        "follow_video_contract_version": FOLLOW_VERSION,
        "mode": FOLLOW_TIMELINE,
        "source_shape": list(shape),
        "source_dtype": dtype,
        "source_sha256": fingerprint,
        "source_fps": FPS,
        "size_mode": mode,
        "target_width": width,
        "target_height": height,
        "chunk_seconds": seconds,
        "preprocess_version": REFERENCE_VIDEO_PREPROCESS_VERSION,
        "window_policy": WINDOW_POLICY,
        "end_policy": END_POLICY,
    }
    return FollowVideoSource(
        normalized, shape, dtype, fingerprint, mode, width, height, seconds,
        _canonical_hash(descriptor),
    )


def select_follow_window(
    source: FollowVideoSource, *, start_frame: int, visible_frames: int,
) -> FollowVideoWindow:
    """Select the existing physical plan's visible range, never its context.

    Use the natural (pre-final-crop) output range. Clamping this to the current
    requested total would change an old final group's conditioning on extension.
    Only the available source and H3's <=16-frame grid padding limit the window.
    """
    if isinstance(start_frame, bool) or isinstance(visible_frames, bool):
        raise ValueError("Physical video window indices must be integers")
    if int(start_frame) != start_frame or int(visible_frames) != visible_frames:
        raise ValueError("Physical video window indices must be integers")
    start, count = int(start_frame), int(visible_frames)
    if start < 0 or count < 1:
        raise ValueError("Invalid internal visible-frame window")
    stop = start + count
    first = min(start, source.frame_count)
    last = min(stop, source.frame_count)
    available = last - first
    encoded = align_frame_count_up(available) if available >= 5 else 0
    identity = _canonical_hash({
        "source": source.combined_hash,
        "requested_start": start,
        "requested_stop": stop,
        "source_start": first,
        "source_stop": last,
        "encoded_frames": encoded,
        "window_policy": WINDOW_POLICY,
    })
    return FollowVideoWindow(start, stop, first, last, encoded, source.frame_count, identity)


def encode_follow_video_group(
    video_vae: Any,
    source: FollowVideoSource,
    *,
    start_frame: int,
    visible_frames: int,
    cache_enabled: bool = False,
    cache_event: Callable | None = None,
) -> tuple[ReferenceVideoAssets | None, FollowVideoWindow]:
    window = select_follow_window(source, start_frame=start_frame, visible_frames=visible_frames)
    if not window.active:
        return None, window
    selected = source.frames[window.source_start:window.source_stop]
    # Both the Qwen item and native Video VAE latent are made by the SAME
    # existing encoder from this one selected frame batch. Slice identity is
    # included in the old cache's source.combined_hash slot.
    selected_source = ReferenceVideoSource(
        frames=selected,
        source_shape=tuple(int(v) for v in selected.shape),
        source_dtype=source.source_dtype,
        source_sha256=window.identity,
        size_mode=source.size_mode,
        target_width=source.target_width,
        target_height=source.target_height,
        frame_count=window.encoded_frames,
        combined_hash=window.identity,
    )
    if cache_enabled:
        assets = encode_reference_video_cached(video_vae, selected_source, cache_event=cache_event)
    else:
        assets = encode_reference_video(video_vae, selected_source)
    return assets, window


def build_follow_group_conditioning(
    source: FollowVideoSource,
    video_vae: Any,
    *,
    start_frame: int,
    visible_frames: int,
    conditioning_builder: Callable,
    conditioning_kwargs: dict[str, Any],
    cache_enabled: bool = False,
    cache_event: Callable | None = None,
) -> tuple[dict, FollowVideoWindow]:
    """Use a fresh prompt-cache dictionary per physical reference interval.

    Never reuse a prior group's text conditioning merely because prompt text
    matches. The VAE cache remains reusable because its key includes the slice.
    """
    assets, window = encode_follow_video_group(
        video_vae, source, start_frame=start_frame, visible_frames=visible_frames,
        cache_enabled=cache_enabled, cache_event=cache_event,
    )
    kwargs = dict(conditioning_kwargs)
    kwargs["timeline_video_assets"] = assets
    return conditioning_builder(**kwargs), window
