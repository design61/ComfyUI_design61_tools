"""User-requested external Conditioning / Sampling separation."""

from ..constants import CONTINUITY_OPTIONS
from ..reference import REFERENCE_SIZE_MATCH_OUTPUT, REFERENCE_SIZE_OPTIONS
from ..reference_audio import REFERENCE_AUDIOS_TYPE
from ..reference_video import REFERENCE_VIDEO_SIZE_OPTIONS, REFERENCE_VIDEO_SIZE_EFFICIENT
from ..video_reference_modes import mode_input_definition
from .external_conditioning import build_external_conditioning
from .reference_images_v39 import REFERENCE_IMAGES_V39_TYPE
from .external_sampling import CONTEXT_GEOMETRIES, KEEP_GEOMETRY, capture_external_segment, prepare_external_segment

CATEGORY = "MiniMax H3/Continuum/External Sampling"


class H3ContinuumExternalConditioning_design61:
    DESCRIPTION = "Native H3 text/image Conditioning and empty joint AV latent for external sampling. Connect the original Reference Images V3.9 helper; Sequence Start flow selects the current chunk automatically. Use a separate instance at each stage's resolution. Sampling and learned upscale remain external."
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP", {"tooltip": "Native H3 CLIP. Uses the same Qwen image presentation and @R tag routing as V3.9."}),
            "video_vae": ("VAE", {"tooltip": "H3 video VAE used only to encode current selected reference images and optional keyframes."}),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Manual chunk prompt. Sequence flow supplies its current parsed prompt when connected. Inactive @R tags are diagnostic only."}),
            "width": ("INT", {"default": 416, "min": 32, "max": 16384, "step": 32, "tooltip": "This stage's canvas width. Encode LOW and HIGH references independently at their own resolution."}),
            "height": ("INT", {"default": 736, "min": 32, "max": 16384, "step": 32, "tooltip": "This stage's canvas height. It does not resize stored continuation State."}),
            "length": ("INT", {"default": 124, "min": 5, "step": 17, "tooltip": "Physical H3 frame count. Connect Sequence Start physical_frames for LOW, and LOW Prepare physical_frames for HIGH."}),
            "chunks": ("INT", {"default": 4, "min": 1, "max": 16, "tooltip": "Manual total chunks; automatic Sequence flow overrides this value without rewriting the widget."}),
            "chunk_index": ("INT", {"default": 1, "min": 1, "max": 16, "tooltip": "Manual current chunk, numbered from 1. Sequence flow derives it from accepted segments, including Retry and Resume."}),
            "reference_size": (REFERENCE_SIZE_OPTIONS, {"default": REFERENCE_SIZE_MATCH_OUTPUT, "tooltip": "Original V3.9 reference sizing. Match Output uses this stage's canvas; Max Identity retains its original sizing policy."}),
        }, "optional": {
            "reference_images": (REFERENCE_IMAGES_V39_TYPE, {"tooltip": "Connect original H3 Continuum Reference Images V3.9. All chunks / Per chunk and fixed @R1–@R9 keep their existing semantics."}),
            "sequence_flow": ("H3_CONTINUUM_EXTERNAL_FLOW", {"tooltip": "Sequence Start flow: current chunk, total chunks and prompt for both LOW and HIGH Conditioning."}),
            "first_frame": ("IMAGE", {"tooltip": "Optional first keyframe, resized and encoded at this stage. External Prepare handles its replacement by continuation context in later segments."}),
            "last_frame": ("IMAGE", {"tooltip": "Optional final keyframe; attached only to the final chunk. Original V3.9 hybrid image presentation is retained."}),
            "reference_video_1": ("IMAGE", {"display_name": "Timeline Video Frames", "tooltip": "Original 24-fps video frame batch. Repeat Reference keeps the bounded guide in each chunk; Follow Timeline selects the current visible interval."}),
            "driving_audio": ("AUDIO", {"display_name": "Driving Audio", "tooltip": "Original audio timeline, sliced by accepted frames and context as native H3 guide. Also connect it to Sequence End driving_audio for original final sound."}),
            "audio_vae": ("VAE", {"display_name": "Driving Audio VAE", "tooltip": "Required when Driving Audio is connected. Uses the existing native H3 audio encoder."}),
            "audio_references": (REFERENCE_AUDIOS_TYPE, {"display_name": "Reference Audios (Optional)", "tooltip": "Original H3 Continuum Reference Audios helper, including its VAE and ordered Audio 1–3 references."}),
            "chunk_seconds": ("FLOAT", {"default": 5.0, "min": 4.0, "max": 15.0, "tooltip": "Manual media timeline duration per chunk; automatic Sequence flow supplies the actual configured duration."}),
            "video_reference_size": (REFERENCE_VIDEO_SIZE_OPTIONS, {"default": REFERENCE_VIDEO_SIZE_EFFICIENT, "tooltip": "Original video-reference sizing policy, resolved independently at each stage."}),
            "video_reference_mode": mode_input_definition(),
        }}
    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("conditioning", "av_latent", "report")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    def build(self, **kwargs):
        result = build_external_conditioning(**kwargs)
        flow = kwargs.get("sequence_flow")
        current = len(flow["entries"]) + 1 if flow is not None else kwargs.get("chunk_index", 1)
        total = flow["chunks"] if flow is not None else kwargs.get("chunks", 4)
        return {"ui": {"external_conditioning": [{"current": current, "chunks": total, "automatic": flow is not None}]}, "result": result}


class H3ContinuumExternalPrepare_design61:
    DESCRIPTION = (
        "Prepare MODEL, CONDITIONING and joint AV LATENT for an external sampler. "
        "Connect your existing H3 conditioning and template latent. Call once before "
        "each stage; connect the low stage's source_plan to the high stage. "
        "For later segments, use the completed high-stage State for both stages. "
        "Low-stage context projection must be explicitly enabled. No sampling, "
        "upscaling, noise generation, or sigma changes occur here."
    )
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "Stage-specific model. External Sage/Sol/Spectrum wrappers are preserved on a call-local clone."}),
            "conditioning": ("CONDITIONING", {"tooltip": "Your external H3 text/image/audio conditioning, already encoded at this stage's resolution."}),
            "latent": ("LATENT", {"tooltip": "H3 joint AV template defining this stage's resolution. On the initial low stage it passes through unchanged."}),
            "continuity": (CONTINUITY_OPTIONS, {"default": CONTINUITY_OPTIONS[0], "tooltip": "Overlap context for subsequent segments. Set identically on both stages; this adapter uses Reference Context transport."}),
            "extend_seconds": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 15.0, "step": 0.1, "tooltip": "New duration per subsequent segment, snapped by the existing H3 temporal helpers. Initial length comes from the template."}),
            "audio_continuity": ("BOOLEAN", {"default": True, "tooltip": "Include the completed previous segment's audio tail as a context reference. Both current AV streams remain externally sampled."}),
            "context_geometry": (list(CONTEXT_GEOMETRIES), {"default": KEEP_GEOMETRY, "tooltip": "Keep resolution is default. Explicit projection resizes only the selected video reference to the template grid; original State and audio remain unchanged. Use projection for the low stage after a high-resolution segment."}),
        }, "optional": {
            "previous_state": ("H3_CONTINUUM_STATE", {"tooltip": "Completed second-pass State from External Capture. Do not connect an unfinished first-pass latent."}),
            "source_plan": ("H3_CONTINUUM_PLAN", {"tooltip": "For the high stage, connect the low stage's plan to share exactly its physical duration and overlap at the high stage's resolution."}),
            "sequence_flow": ("H3_CONTINUUM_EXTERNAL_FLOW", {"tooltip": "Optional automatic Sequence Start flow; supplies the controller's continuity and chunk duration to both stages."}),
        }}
    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "H3_CONTINUUM_PLAN", "INT", "STRING")
    RETURN_NAMES = ("model", "conditioning", "latent", "plan", "physical_frames", "report")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY

    def prepare(self, **kwargs):
        return prepare_external_segment(**kwargs)


class H3ContinuumExternalCapture_design61:
    DESCRIPTION = (
        "Capture completed external sampling output as CPU continuation State and "
        "append a segment to a separate in-memory external sequence. Connect the "
        "SECOND sampler's completed output and the HIGH stage plan. Exposes video/audio "
        "LATENT lists and Assembly Plan for external VAE Decode and Continuum Finalize. "
        "No sampling or intermediate decoding. This does not write Main Run Storage "
        "or provide its Review/Take/resume controls."
    )
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT", {"tooltip": "Completed joint AV output from the second external sampler. LATENT does not carry reliable sigma metadata; this node cannot detect unfinished denoising."}),
            "plan": ("H3_CONTINUUM_PLAN", {"tooltip": "Plan from the high-stage External Prepare. Spatial and temporal topology must match the final samples."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Record the seed used by your external sampler; this node does not generate or change noise."}),
        }, "optional": {
            "previous_sequence": ("H3_CONTINUUM_EXTERNAL_SEQUENCE", {"tooltip": "Previous External Capture's sequence for CPU-only accumulation and final decode/assembly. Use its matching State in Prepare."}),
        }}
    RETURN_TYPES = ("LATENT", "H3_CONTINUUM_STATE", "LATENT", "LATENT", "H3_CONTINUUM_ASSEMBLY_PLAN", "H3_CONTINUUM_EXTERNAL_SEQUENCE", "STRING")
    RETURN_NAMES = ("av_latent", "state", "video_latents", "audio_latents", "assembly_plan", "sequence", "report")
    OUTPUT_IS_LIST = (False, False, True, True, False, False, False)
    FUNCTION = "capture"
    CATEGORY = CATEGORY

    def capture(self, **kwargs):
        return capture_external_segment(**kwargs)


NODE_CLASS_MAPPINGS = {
    "H3ContinuumExternalConditioning_design61": H3ContinuumExternalConditioning_design61,
    "H3ContinuumExternalPrepare_design61": H3ContinuumExternalPrepare_design61,
    "H3ContinuumExternalCapture_design61": H3ContinuumExternalCapture_design61,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuumExternalConditioning_design61": "H3 Continuum External Conditioning",
    "H3ContinuumExternalPrepare_design61": "H3 Continuum External Prepare",
    "H3ContinuumExternalCapture_design61": "H3 Continuum External Capture",
}
