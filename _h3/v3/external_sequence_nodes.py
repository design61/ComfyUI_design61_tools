"""Visible automatic-sequence and Review controls around editable external nodes."""
from ..constants import CONTINUITY_OPTIONS, PROMPT_FORMAT_AUTO, PROMPT_FORMAT_OPTIONS
from .external_sequence import ACTIONS, LEGACY_FINISH, FULL, REVIEW, begin_sequence, end_sequence, sequence_progress

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
            "generation_mode": ([FULL, REVIEW], {"default": REVIEW, "display_name": "Generation Mode", "tooltip": "Full Video runs every chunk without Review buttons. Review Each Chunk pauses after every chunk for Continue, Finish here, Retry or Restart all."}),
            "review_action": ([*ACTIONS, LEGACY_FINISH], {"default": ACTIONS[0], "tooltip": "Internal one-shot command set by Sequence End buttons. Review actions retain the selected run after prompt edits. Normal Queue starts fresh. Finish here finalizes completed chunks; Restart all preserves old Takes."}),
            "run_name": ("STRING", {"default": "external_dual_sampling", "tooltip": "Separate external-review run name. The submitted graph defines its lineage; old Takes are retained when settings change."}),
            "expected_revision": ("STRING", {"default": "", "tooltip": "Internal backend lineage/revision reference submitted only by Review buttons; old plain revision references still load. Normal Queue leaves this empty."}),
        }, "optional": {"iteration": ("H3_CONTINUUM_EXTERNAL_FLOW", {"tooltip": "Internal dynamic-expansion continuation payload; leave unconnected in the visible workflow."})},
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("H3_CONTINUUM_EXTERNAL_FLOW", "H3_CONTINUUM_STATE", "H3_CONTINUUM_EXTERNAL_SEQUENCE", "STRING", "INT", "INT")
    RETURN_NAMES = ("flow", "previous_state", "previous_sequence", "prompt", "seed", "physical_frames")
    FUNCTION = "start"
    CATEGORY = CATEGORY
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
    def start(self, **kwargs):
        result = begin_sequence(**kwargs)
        return {"ui": {"external_progress": [sequence_progress(result[0])]}, "result": result}


class H3ContinuumExternalSequenceEnd_design61:
    DESCRIPTION = "Receive the final external second-pass output and HIGH plan. Saves one immutable CPU Take, exposes canonical Review status, and dynamically expands the editable external graph for the next segment. It does not invoke a sampler or learned upscaler internally."
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "flow": ("H3_CONTINUUM_EXTERNAL_FLOW", {"tooltip": "Flow from matching Sequence Start."}),
            "samples": ("LATENT", {"lazy": True, "tooltip": "Completed second-pass AV output. Cached Review/completion queries do not evaluate the external sampling chain."}),
            "plan": ("H3_CONTINUUM_PLAN", {"lazy": True, "tooltip": "HIGH External Prepare plan matching the completed AV output."}),
        }, "optional": {"driving_audio": ("AUDIO", {"display_name": "Driving Audio", "tooltip": "Same original timeline connected to both Conditioning stages. Passed through Assembly Plan to Finalize and trimmed to completed duration, including Finish here."})}, "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("LATENT", "LATENT", "H3_CONTINUUM_ASSEMBLY_PLAN", "H3_CONTINUUM_STATE", "STRING")
    RETURN_NAMES = ("video_latents", "audio_latents", "assembly_plan", "state", "status")
    OUTPUT_IS_LIST = (True, True, False, False, False)
    FUNCTION = "finish"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    def check_lazy_status(self, flow, samples=None, plan=None, **kwargs):
        return [name for name, value in (("samples", samples), ("plan", plan)) if value is None] if flow["active"] else []
    def finish(self, **kwargs):
        return end_sequence(**kwargs)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (H3ContinuumExternalSequenceStart_design61, H3ContinuumExternalSequenceEnd_design61)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuumExternalSequenceStart_design61": "H3 Continuum External Sequence Start",
    "H3ContinuumExternalSequenceEnd_design61": "H3 Continuum External Sequence End",
}
