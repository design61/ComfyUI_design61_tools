"""Visible automatic-sequence and Review controls around editable external nodes."""
from ..constants import CONTINUITY_OPTIONS, PROMPT_FORMAT_AUTO, PROMPT_FORMAT_OPTIONS
from .external_sequence import ACTIONS, LEGACY_FINISH, FULL, REVIEW, begin_sequence, end_sequence, sequence_progress, expanded_output_packet
from .frame_storage import LATENTS, FRAMES, end_frame_sequence
from .external_control import guard_stop

CATEGORY = "MiniMax H3/Continuum/External Sampling"


class H3ContinuumExternalSequenceStart_design61:
    DESCRIPTION = "Automatic external-sampling sequence control. Connect prompt, seed, frame count and State to the editable dual-sampling graph, and flow to Sequence End. Sampling remains in the external nodes. Review sessions are stored separately from Main Run Storage."
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sequence_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Fixed, List, Timeline or JSON with Fixed fallback. Review Retry/Continue rereads edited text while keeping the accepted prefix. The blue Queue button starts a new sequence."}),
            "prompt_mode": (PROMPT_FORMAT_OPTIONS, {"default": PROMPT_FORMAT_AUTO, "tooltip": "Existing Continuum prompt interpretation; content never blocks this controller."}),
            "chunks": ("INT", {"default": 4, "min": 1, "max": 16, "tooltip": "Number of physical segments. Each segment executes the external high-noise/upscale/low-noise graph before the next starts."}),
            "chunk_seconds": ("FLOAT", {"default": 5.0, "min": 4.0, "max": 15.0, "step": 0.1, "tooltip": "Duration per segment before native H3 temporal-grid alignment; uses the existing Continuum prompt-plan range."}),
            "continuity": (CONTINUITY_OPTIONS, {"default": CONTINUITY_OPTIONS[0], "tooltip": "Use this same continuity setting in both External Prepare nodes."}),
            "base_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Existing deterministic chunk-seed derivation. Connect output seed to external RandomNoise and any restart seed."}),
            "generation_mode": ([FULL, REVIEW], {"default": REVIEW, "display_name": "Generation Mode", "tooltip": "Full Video runs every chunk automatically; End offers Stop to preserve completed chunks and switch to Review. Review Each Chunk pauses after every chunk for Continue, Finish here, Retry or Restart all."}),
            "review_action": ([*ACTIONS, LEGACY_FINISH], {"default": ACTIONS[0], "tooltip": "Internal one-shot command set by Sequence End. Continue all runs the remaining chunks automatically. Regenerate from selected chunk keeps its earlier prefix and branches with new Takes. Prompt edits are reread; normal Queue starts fresh."}),
            "run_name": ("STRING", {"default": "external_dual_sampling", "tooltip": "Separate external-review run name. Review buttons preserve this lineage across prompt edits. Latents retains old Takes; Frames removes superseded media only after replacement succeeds."}),
            "expected_revision": ("STRING", {"default": "", "tooltip": "Internal backend lineage/revision reference submitted only by Review buttons; old plain revision references still load. Normal Queue leaves this empty."}),
        }, "optional": {
            "iteration": ("H3_CONTINUUM_EXTERNAL_FLOW", {"tooltip": "Internal dynamic-expansion continuation payload; leave unconnected in the visible workflow."}),
            "storage_mode": ([LATENTS, FRAMES], {"default": LATENTS, "tooltip": "Latents preserves the existing Decode/Finalize workflow. Frames decodes only the current chunk, saves PNGs and continuation tail State, and outputs a folder for the design61 FFmpeg node. Choose before starting a new run; Review stays in its original storage mode."}),
            "frame_root": ("STRING", {"default": "", "tooltip": "Frames storage directory. Empty uses ComfyUI output/design61_sequences; relative paths are beneath output. Store on a disk with space. Regeneration deletes superseded frames/tails after the replacement commits; existing latent Sessions are untouched."}),
        },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID", "execution_list": "EXECUTION_LIST"}}
    RETURN_TYPES = ("H3_CONTINUUM_EXTERNAL_FLOW", "H3_CONTINUUM_STATE", "H3_CONTINUUM_EXTERNAL_SEQUENCE", "STRING", "INT", "INT")
    RETURN_NAMES = ("flow", "previous_state", "previous_sequence", "prompt", "seed", "physical_frames")
    FUNCTION = "start"
    CATEGORY = CATEGORY
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
    def start(self, **kwargs):
        execution_list = kwargs.pop("execution_list", None)
        iteration = kwargs.get("iteration")
        if isinstance(iteration, dict) and "_external_output_source" in iteration:
            return expanded_output_packet(iteration, execution_list, kwargs.get("unique_id"))
        result = begin_sequence(**kwargs)
        return {"ui": {"external_progress": [sequence_progress(result[0])]}, "result": result}


class H3ContinuumExternalSequenceEnd_design61:
    DESCRIPTION = "Receive the completed external AV output and matching HIGH plan. Latents mode saves CPU Takes for Decode/Finalize. Frames mode decodes only this chunk, stores effective PNG frames and native continuation tail State, and outputs a folder/audio for the design61 FFmpeg node. Review and automatic expansion keep the sampling/upscale graph external."
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "flow": ("H3_CONTINUUM_EXTERNAL_FLOW", {"tooltip": "Flow from matching Sequence Start."}),
            "samples": ("LATENT", {"lazy": True, "tooltip": "Completed second-pass AV output. Cached Review/completion queries do not evaluate the external sampling chain."}),
            "plan": ("H3_CONTINUUM_PLAN", {"lazy": True, "tooltip": "HIGH External Prepare plan matching the completed AV output."}),
        }, "optional": {
            "driving_audio": ("AUDIO", {"display_name": "Driving Audio", "tooltip": "Same original timeline connected to Conditioning. Frames mode outputs original audio trimmed/padded to accepted duration, without requiring Audio VAE. Latent mode passes it to Finalize."}),
            "video_vae": ("VAE", {"lazy": True, "display_name": "Video VAE", "tooltip": "Connect the same H3 Video VAE in Frames mode. Decode stays within the current physical chunk; overlap frames are removed before PNGs are saved. Unused in Latents mode."}),
            "audio_vae": ("VAE", {"lazy": True, "display_name": "Audio VAE", "tooltip": "Optional generated-audio decode in Frames mode. Driving Audio takes precedence. Without either, the frame folder has no audio; audio continuation tail is still saved."}),
        }, "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("LATENT", "LATENT", "H3_CONTINUUM_ASSEMBLY_PLAN", "H3_CONTINUUM_STATE", "STRING", "STRING", "AUDIO")
    RETURN_NAMES = ("video_latents", "audio_latents", "assembly_plan", "state", "status", "frames_path", "audio")
    OUTPUT_IS_LIST = (True, True, False, False, False, False, False)
    FUNCTION = "finish"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    def check_lazy_status(self, flow, samples=None, plan=None, **kwargs):
        guard_stop(flow)
        if not flow["active"]:
            return []
        required = [("samples", samples), ("plan", plan)]
        if flow.get("storage_mode") == FRAMES:
            if "video_vae" in kwargs:
                required.append(("video_vae", kwargs.get("video_vae")))
            if "audio_vae" in kwargs and kwargs.get("audio_vae") is None and kwargs.get("driving_audio") is None:
                required.append(("audio_vae", None))
        return [name for name, value in required if value is None]
    def finish(self, **kwargs):
        if kwargs["flow"].get("storage_mode") == FRAMES:
            return end_frame_sequence(**kwargs)
        kwargs.pop("video_vae", None)
        kwargs.pop("audio_vae", None)
        result = end_sequence(**kwargs)
        # The first five slots stay byte-for-byte compatible with saved graphs.
        from comfy_execution.graph import ExecutionBlocker
        result["result"] = (*result["result"], ExecutionBlocker(None), None)
        return result


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (H3ContinuumExternalSequenceStart_design61, H3ContinuumExternalSequenceEnd_design61)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuumExternalSequenceStart_design61": "H3 Continuum External Sequence Start",
    "H3ContinuumExternalSequenceEnd_design61": "H3 Continuum External Sequence End",
}
