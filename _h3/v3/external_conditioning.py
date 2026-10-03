"""Sampling-free native H3 conditioning with the original V3.9 image router."""
from __future__ import annotations

from ..reference import REFERENCE_SIZE_MATCH_OUTPUT
from ..constants import CONTINUITY_FRAMES, CONTINUITY_OPTIONS, FPS
from ..driving_audio import (DrivingAudioAssets, prepare_driving_audio_source,
                            encode_driving_audio, slice_driving_audio_latent, attach_driving_audio)
from ..reference_audio import ReferenceAudioBundle, resolve_reference_audio_input, encode_reference_audio_input
from ..reference_video import (REFERENCE_VIDEO_SIZE_EFFICIENT, prepare_reference_video_source,
                               encode_reference_video_cached)
from ..video_reference_modes import (REPEAT_REFERENCE, FOLLOW_TIMELINE,
                                     prepare_follow_video_source, encode_follow_video_group)
from ..temporal import align_frame_count_up
from ..v2.h3_builder import attach_keyframes, empty_h3_latent, prepare_identity_assets
from .reference_images_v39 import ReferenceImagesV39Bundle, effective_reference_inputs
from .reference_routing import compile_reference_routing_schedule
from .reference_runtime import ReferenceInputSet, ReferenceRoutingRuntime

DRIVING_GUIDE_KEY = "_h3_external_driving_guide_v1"


def _driving_assets(source, vae):
    if source is None:
        return None
    from .ref_encode_cache import get_ref_encode_cache, make_ref_encode_cache_key
    cache = get_ref_encode_cache()
    key = make_ref_encode_cache_key("external_driving_audio", 1, source.combined_hash)
    latent = cache.lookup(vae, key)
    if latent is None:
        latent = encode_driving_audio(source, vae).audio_latent.detach().to("cpu").contiguous()
        if cache.supports_vae(vae):
            cache.store(vae, key, latent)
    return DrivingAudioAssets(source, latent)


def build_external_conditioning(*, clip, video_vae, prompt, width, height, length,
                                chunks=4, chunk_index=1,
                                reference_size=REFERENCE_SIZE_MATCH_OUTPUT,
                                reference_images=None, sequence_flow=None,
                                first_frame=None, last_frame=None, driving_audio=None,
                                audio_vae=None, audio_references=None, reference_video_1=None,
                                chunk_seconds=5.0, video_reference_size=REFERENCE_VIDEO_SIZE_EFFICIENT,
                                video_reference_mode=REPEAT_REFERENCE):
    """Encode the current chunk's selected images separately for each resolution.

    No sampler, upscaler, decode or stored-State mutation occurs.
    Continuation remains in External Prepare, after this native conditioning.
    """
    if sequence_flow is not None:
        chunks = int(sequence_flow["chunks"])
        chunk_index = len(sequence_flow["entries"]) + 1
        prompt = sequence_flow["prompts"][min(chunk_index - 1, chunks - 1)]
        chunk_seconds = sequence_flow.get("seconds", chunk_seconds)
        length = sequence_flow.get("physical_frames", length)
    # Original helpers belong to a different Python package; adapt their public
    # payload fields without changing source tensors, selectors, VAE or hashes.
    if reference_images is not None and not isinstance(reference_images, ReferenceImagesV39Bundle):
        reference_images = ReferenceImagesV39Bundle(tuple(reference_images.images), reference_images.reference_use, tuple(reference_images.selectors))
    if audio_references is not None and not isinstance(audio_references, ReferenceAudioBundle):
        audio_references = ReferenceAudioBundle(tuple(audio_references.sources), audio_references.reference_audio_vae, audio_references.combined_hash)
    frames = align_frame_count_up(int(length))
    retained = sum(entry["plan"]["net_frames"] for entry in sequence_flow["entries"]) if sequence_flow and sequence_flow["entries"] and isinstance(sequence_flow["entries"][0], dict) else round((chunk_index - 1) * chunk_seconds * FPS)
    trim = CONTINUITY_FRAMES[sequence_flow.get("continuity", CONTINUITY_OPTIONS[0])] if sequence_flow and sequence_flow["entries"] else 0
    audio_source, reference_vae = resolve_reference_audio_input(None, None, audio_references)
    audio_assets = encode_reference_audio_input(reference_vae, audio_source, cache_enabled=True) if audio_source is not None else None
    drive_source = prepare_driving_audio_source(driving_audio, audio_vae, target_frames=round(chunks * chunk_seconds * FPS), fps=FPS)
    drive = slice_driving_audio_latent(_driving_assets(drive_source, audio_vae), cumulative_retained_before=retained, total_frames=frames, trim_frames=trim, fps=FPS)
    video_assets, video_report = None, ""
    if reference_video_1 is not None:
        if video_reference_mode == FOLLOW_TIMELINE:
            source = prepare_follow_video_source(reference_video_1, chunk_seconds=chunk_seconds, output_width=int(width), output_height=int(height), size_mode=video_reference_size)
            video_assets, window = encode_follow_video_group(video_vae, source, start_frame=retained, visible_frames=frames-trim, cache_enabled=True)
            video_report = window.report()
        else:
            source = prepare_reference_video_source(reference_video_1, target_frames=align_frame_count_up(round(chunk_seconds*FPS)), output_width=int(width), output_height=int(height), size_mode=video_reference_size)
            video_assets = encode_reference_video_cached(video_vae, source)
            video_report = "Video Frames: Repeat Reference."
    direct, extra, selectors = effective_reference_inputs(reference_images, chunks=int(chunks))
    schedule = compile_reference_routing_schedule(
        total_chunks=int(chunks), terminal_merge_enabled=False,
        mode="Custom", selectors_by_slot=selectors,
    )
    runtime = ReferenceRoutingRuntime(
        schedule=schedule,
        inputs=ReferenceInputSet.from_inputs(
            **{f"reference_image_{index}": image for index, image in enumerate(direct, 1)},
            image_references=extra, output_width=int(width), output_height=int(height),
            size_mode=reference_size,
        ),
    )
    identity = prepare_identity_assets(
        video_vae, width=int(width), height=int(height),
        first_frame=first_frame, last_frame=last_frame,
    )
    group = runtime.prepare_group(
        physical_group=int(chunk_index), prompt=str(prompt), clip=clip, video_vae=video_vae,
        first_image=identity.first_image, last_image=identity.last_image,
        include_last_image=int(chunk_index) == int(chunks),
        reference_audio_assets=audio_assets, timeline_video_assets=video_assets,
    )
    conditioning = attach_keyframes(
        group.conditioning, frame_count=frames, first_latent=identity.first_latent,
        last_latent=identity.last_latent if int(chunk_index) == int(chunks) else None,
    )
    conditioning = attach_driving_audio(conditioning, drive)
    if drive is not None:
        for _, metadata in conditioning:
            metadata[DRIVING_GUIDE_KEY] = drive
    selected = group.picture_map.source_slot_ids
    mappings = ", ".join(f"{item.source_slot_id}={item.picture_tag}" for item in group.picture_map.references)
    report = f"External Conditioning chunk {chunk_index}/{chunks}: {width}x{height}, {frames} frames; selected {', '.join(selected) or 'none'}."
    if mappings:
        report += "\n" + mappings
    if group.warnings:
        report += "\n" + "\n".join(group.warnings)
    if drive_source is not None:
        report += f"\nDriving Audio: native guide, physical slice starts at frame {max(0, retained-trim)}; original waveform is available to Sequence End/Finalize."
    if audio_assets is not None:
        report += "\nReference Audios: original ordered bundle encoded with its own Audio VAE."
    if video_report:
        report += "\n" + video_report
    return conditioning, empty_h3_latent(int(width), int(height), frames), report
