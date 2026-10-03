"""Explicit, sampling-free adapters for externally sampled H3 segments.

The only geometry conversion here is an opt-in projection of a selected
continuation reference. The accepted State and generated latent are never resized.
"""

from __future__ import annotations

import copy
import hashlib

import torch

from ..constants import CONTINUITY_FRAMES, FPS
from ..continuation import POLICY_REPLACE, clone_conditioning, new_h3_latent, prepare_conditioning
from ..model_patch import clone_model_for_chunk
from ..state import StateValidationError, extract_av_streams, make_plan, select_context, validate_plan, validate_state
from ..temporal import audio_latent_t, largest_context_capacity, make_extension_shape, pixel_frames_for_latent_t, video_latent_t
from ..v2.session import entry_to_state, make_chunk_entry, validate_chunk_entry
from .plan import prepare_physical_decode_entries
from .external_conditioning import DRIVING_GUIDE_KEY
from ..driving_audio import attach_driving_audio

KEEP_GEOMETRY = "Keep State resolution"
PROJECT_CONTEXT = "Project context to template resolution"
CONTEXT_GEOMETRIES = (KEEP_GEOMETRY, PROJECT_CONTEXT)
SEQUENCE_MAGIC = "H3_CONTINUUM_EXTERNAL_SEQUENCE"


def _project_reference(video, target_h, target_w):
    """Explicit spatial projection; keep time, audio and original State intact."""
    batch, channels, temporal, _, _ = video.shape
    flat = video.permute(0, 2, 1, 3, 4).reshape(batch * temporal, channels, *video.shape[-2:])
    resized = torch.nn.functional.interpolate(
        flat.float(), size=(target_h, target_w), mode="bilinear", align_corners=False,
    ).to(video.dtype)
    return resized.reshape(batch, temporal, channels, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous()


def prepare_external_segment(*, model, conditioning, latent, continuity, extend_seconds,
                             audio_continuity, context_geometry=KEEP_GEOMETRY,
                             previous_state=None, source_plan=None, sequence_flow=None):
    """Prepare a single physical segment. No noise generation or sampling."""
    if sequence_flow is not None:
        continuity = sequence_flow["continuity"]
        extend_seconds = sequence_flow.get("extend_seconds", sequence_flow["seconds"])
    video, _ = extract_av_streams(latent)
    width, height = int(video.shape[-1]) * 16, int(video.shape[-2]) * 16
    current_frames = pixel_frames_for_latent_t(int(video.shape[2]))
    state = validate_state(previous_state) if previous_state is not None else None
    context_frames = CONTINUITY_FRAMES[continuity] if state is not None else 0
    clip_index = int(state["clip_index"]) + 1 if state is not None else 1
    report = []

    if source_plan is not None:
        # Both stages share one temporal/overlap contract, at independent geometry.
        plan = copy.deepcopy(validate_plan(source_plan))
        if int(plan["clip_index"]) != clip_index or int(plan["trim_frames"]) != context_frames:
            raise StateValidationError("external source_plan and previous State describe different segment/overlap topology")
        plan.update(width=width, height=height)
        target_latent = latent if current_frames == plan["total_frames"] else new_h3_latent(
            latent, video_t=video_latent_t(plan["total_frames"]), audio_t=audio_latent_t(plan["total_frames"]),
        )
    elif state is None:
        plan = make_plan(
            continuation=False, clip_index=1, total_frames=current_frames, trim_frames=0,
            width=width, height=height, context_frames=5,
            state_capacity_frames=largest_context_capacity(current_frames),
            requested_extend_seconds=current_frames / FPS, debug=False,
        )
        target_latent = latent
    else:
        shape = make_extension_shape(context_frames, float(extend_seconds))
        plan = make_plan(
            continuation=True, clip_index=clip_index, total_frames=shape.total_frames,
            trim_frames=context_frames, width=width, height=height, context_frames=context_frames,
            state_capacity_frames=largest_context_capacity(shape.net_new_frames),
            requested_extend_seconds=float(extend_seconds), debug=False,
        )
        target_latent = latent if current_frames == shape.total_frames else new_h3_latent(
            latent, video_t=shape.video_latent_t, audio_t=shape.audio_latent_t,
        )

    if target_latent is not latent and latent.get("noise_mask") is not None:
        report.append("New temporal template created; input noise_mask was not extended. Rebuild any time-dependent audio/guide masks externally for physical_frames before sampling.")

    validate_plan(plan)
    output_conditioning = clone_conditioning(conditioning)
    if state is not None:
        context_video, context_audio, offset = select_context(state, context_frames, include_audio=bool(audio_continuity))
        if (state["width"], state["height"]) != (width, height):
            if context_geometry != PROJECT_CONTEXT:
                raise StateValidationError(
                    f"state is {state['width']}x{state['height']} but template is {width}x{height}; "
                    "automatic State resizing is disabled. Explicit context projection is available."
                )
            context_video = _project_reference(context_video, int(video.shape[-2]), int(video.shape[-1]))
            report.append(f"Explicit reference-only projection: {state['width']}x{state['height']} -> {width}x{height}; original State unchanged.")
        output_conditioning = prepare_conditioning(
            output_conditioning, video_context=context_video, audio_context=context_audio,
            audio_grid_offset=offset, context_frames=context_frames, new_frame_count=plan["total_frames"],
            first_frame_policy=POLICY_REPLACE, preserve_last_frame=True, source_frame_count=current_frames,
        )
    for index, (tensor, metadata) in enumerate(output_conditioning):
        driving_guide = metadata.pop(DRIVING_GUIDE_KEY, None)
        if state is not None and driving_guide is not None:
            # Core's frame-zero audio guide is attached AFTER continuation,
            # exactly as in Main; its first-keyframe replacement must not drop it.
            output_conditioning[index] = attach_driving_audio([[tensor, metadata]], driving_guide)[0]
        output_conditioning[index][1]["minimax_frame_count"] = int(plan["total_frames"])
    output_model = clone_model_for_chunk(
        model, strict=False, debug=False, chunk_index=clip_index,
        context_frames=context_frames if state is not None else None,
    )
    report.insert(0, f"External segment {clip_index}: {width}x{height}, {plan['total_frames']} physical frames, trim {plan['trim_frames']}. Sampling remains external; Reference Context transport.")
    return output_model, output_conditioning, target_latent, plan, int(plan["total_frames"]), "\n".join(report)


def capture_external_segment(*, samples, plan, seed=0, previous_sequence=None):
    """Accept the caller's final sampling output and expose CPU decode groups."""
    plan = validate_plan(plan)
    video, _ = extract_av_streams(samples)
    # Reuse accepted schema/topology validation before exposing anything as State.
    entry = make_chunk_entry(
        latent=samples, plan=plan, prompt="", prompt_hash=hashlib.sha256(b"").hexdigest(),
        seed=int(seed), context_frames=int(plan["trim_frames"]), motion_score=0.0, reused=False,
    )
    entries = []
    if previous_sequence is not None:
        if not isinstance(previous_sequence, dict) or previous_sequence.get("magic") != SEQUENCE_MAGIC or previous_sequence.get("version") != 1:
            raise StateValidationError("invalid external sequence schema")
        stored_entries = previous_sequence.get("entries")
        if not isinstance(stored_entries, tuple) or not stored_entries:
            raise StateValidationError("external sequence has no accepted CPU entries")
        entries = [validate_chunk_entry(item) for item in stored_entries]
        if any(right["clip_index"] != left["clip_index"] + 1 for left, right in zip(entries, entries[1:])):
            raise StateValidationError("external sequence stored clip order is not contiguous")
    if entries and int(plan["clip_index"]) != entries[-1]["clip_index"] + 1:
        raise StateValidationError("external sequence clip order is not contiguous")
    if (plan["width"], plan["height"]) != (int(video.shape[-1]) * 16, int(video.shape[-2]) * 16):
        raise StateValidationError("external completed latent spatial topology does not match its plan; connect the high-resolution plan")
    entries = [*entries, validate_chunk_entry(entry)]
    retained_frames = sum(item["plan"]["net_frames"] for item in entries)
    decode_entries, assembly_plan = prepare_physical_decode_entries(
        entries, chunk_seconds=retained_frames / FPS / len(entries),
        preserve_final_frame=False, terminal_merged=False,
    )
    state = entry_to_state(entry)
    from ..v2.sampling import latent_from_cpu
    av_latent = latent_from_cpu(entry["video"], entry["audio"])
    sequence = {"magic": SEQUENCE_MAGIC, "version": 1, "entries": tuple(entries)}
    return (
        av_latent, state, [{"samples": item["video"]} for item in decode_entries],
        [{"samples": item["audio"]} for item in decode_entries], assembly_plan, sequence,
        f"Accepted external segment {len(entries)} on CPU; {retained_frames} retained frames. Caller must connect completed second-pass output; noise level is not inferred from LATENT.",
    )
