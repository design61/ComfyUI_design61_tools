"""Run external loop adapters through real ComfyUI PromptExecutor on CPU tensors.

No weights, real media, sampler kernels, learned VAE/upscale or GPU Queue is used.
Synthetic external node classes live only in this short-lived verifier process.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--reference-gate", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    sys.path.insert(0, str(args.comfy_root))
    import torch
    import nodes
    from execution import PromptExecutor
    from comfy.nested_tensor import NestedTensor
    name = "h3_external_cpu_gate"
    spec = importlib.util.spec_from_file_location(name, ROOT / "_h3" / "__init__.py", submodule_search_locations=[str(ROOT / "_h3")])
    sys.modules[name] = importlib.util.module_from_spec(spec)
    from h3_external_cpu_gate.v3 import external_sequence as seq
    from h3_external_cpu_gate.v3 import external_sampling as sampling
    from h3_external_cpu_gate.v3.external_sequence_nodes import NODE_CLASS_MAPPINGS as controls
    from h3_external_cpu_gate.v3.external_sampling_nodes import NODE_CLASS_MAPPINGS as adapters
    from h3_external_cpu_gate.temporal import video_latent_t, audio_latent_t
    from h3_external_cpu_gate.v2 import session_io
    from h3_external_cpu_gate.constants import CONTINUITY_OPTIONS
    calls = []
    reference_calls = []
    def matches(actual, expected):
        # Core's Lanczos image path quantizes to RGB8 before synthetic encoding.
        return len(actual)==len(expected) and all(len(a)==len(b) and all(abs(x-y)<1/255 for x,y in zip(a,b)) for a,b in zip(actual,expected))
    class SyntheticClip:
        def tokenize(self, text, **kwargs):
            images = [item["data"] for item in kwargs.get("minimax_ref_items", []) if item["type"] == "image"]
            reference_calls.append([float(image.mean()) for image in images])
            return text
        def encode_from_tokens_scheduled(self, tokens):
            return [[torch.ones(1, 2, 3), {"prompt": tokens}]]
    class SyntheticVAE:
        def encode(self, image):
            return torch.full((1, 24, 1, image.shape[1] // 16, image.shape[2] // 16), float(image.mean()))
    class ReferenceAssets:
        @classmethod
        def INPUT_TYPES(cls): return {"required": {}}
        RETURN_TYPES = ("MODEL", "CLIP", "VAE", "H3_CONTINUUM_REFERENCE_IMAGES_V39")
        FUNCTION = "build"
        def build(self):
            from h3_external_cpu_gate.v3.reference_images_v39 import H3ContinuumReferenceImagesV39_design61
            bundle = H3ContinuumReferenceImagesV39_design61().pack(
                reference_use="Per chunk", reference_image_1=torch.full((1,256,256,3),0.25), reference_r1_chunks="1",
                reference_image_4=torch.full((1,256,256,3),0.75), reference_r4_chunks="2",
            )[0]
            return "synthetic_model", SyntheticClip(), SyntheticVAE(), bundle
    class CpuUpscale:
        @classmethod
        def INPUT_TYPES(cls): return {"required": {"latent": ("LATENT",)}}
        RETURN_TYPES = ("LATENT",)
        FUNCTION = "upscale"
        def upscale(self, latent):
            video, audio = latent["samples"].unbind()
            b,c,t,h,w = video.shape
            flat = video.permute(0,2,1,3,4).reshape(b*t,c,h,w)
            flat = torch.nn.functional.interpolate(flat, scale_factor=2, mode="bilinear", align_corners=False)
            video = flat.reshape(b,t,c,h*2,w*2).permute(0,2,1,3,4).contiguous()
            return ({"samples": NestedTensor((video, audio))},)
    class Template:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"frames": ("INT",), "text": ("STRING",)}}
        RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT")
        FUNCTION = "build"
        def build(self, frames, text):
            video = torch.zeros(1, 24, video_latent_t(frames), 4, 4)
            audio = torch.zeros(1, 32, 2, audio_latent_t(frames))
            return "synthetic_model", [[torch.ones(1, 2, 3), {}]], {"samples": NestedTensor((video, audio))}
    class ExternalPass:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"latent": ("LATENT",), "seed": ("INT",), "stage": ("STRING",)}, "optional": {"conditioning": ("CONDITIONING",)}}
        RETURN_TYPES = ("LATENT",)
        FUNCTION = "sample"
        def sample(self, latent, seed, stage, conditioning=None):
            calls.append(stage)
            parts = list(latent["samples"].unbind())
            return ({"samples": NestedTensor(tuple(part + 1 for part in parts))},)
    class Observer:
        @classmethod
        def INPUT_TYPES(cls): return {"required": {"samples": ("LATENT",)}}
        RETURN_TYPES = ()
        FUNCTION = "observe"
        OUTPUT_NODE = True
        INPUT_IS_LIST = True
        def observe(self, samples): return ()
    class Server:
        client_id = None
        last_node_id = None
        def send_sync(self, *args, **kwargs): pass
    nodes.NODE_CLASS_MAPPINGS.update(controls)
    nodes.NODE_CLASS_MAPPINGS.update(adapters)
    nodes.NODE_CLASS_MAPPINGS.update(H3CpuTemplate=Template, H3CpuExternalPass=ExternalPass, H3CpuObserver=Observer)
    nodes.NODE_CLASS_MAPPINGS.update(H3CpuReferenceAssets=ReferenceAssets, H3CpuUpscale=CpuUpscale)
    sampling.clone_model_for_chunk = lambda model, **kwargs: model
    graph = {
        "start": {"class_type": seq.START, "inputs": dict(sequence_prompt="CPU simulation", prompt_mode="Auto", chunks=3, chunk_seconds=5.0, continuity=CONTINUITY_OPTIONS[0], base_seed=123, generation_mode=seq.FULL, review_action=seq.ACTIONS[0], run_name="cpu_gate", expected_revision="")},
        "template": {"class_type": "H3CpuTemplate", "inputs": {"frames": ["start", 5], "text": ["start", 3]}},
        "prepare": {"class_type": "H3ContinuumExternalPrepare_design61", "inputs": dict(model=["template", 0], conditioning=["template", 1], latent=["template", 2], continuity=CONTINUITY_OPTIONS[0], extend_seconds=5.0, audio_continuity=True, context_geometry="Keep State resolution", previous_state=["start", 1], sequence_flow=["start", 0])},
        "first": {"class_type": "H3CpuExternalPass", "inputs": {"latent": ["prepare", 2], "seed": ["start", 4], "stage": "first"}},
        "second": {"class_type": "H3CpuExternalPass", "inputs": {"latent": ["first", 0], "seed": ["start", 4], "stage": "second"}},
        "end": {"class_type": seq.END, "inputs": {"flow": ["start", 0], "samples": ["second", 0], "plan": ["prepare", 3]}},
        "observe": {"class_type": "H3CpuObserver", "inputs": {"samples": ["end", 0]}},
    }
    with tempfile.TemporaryDirectory(prefix="external-core-gate-", dir=ROOT) as temporary:
        session_io.session_directory = lambda: Path(temporary)
        seq.session_directory = lambda: Path(temporary)
        executor = PromptExecutor(Server(), cache_args={"ram": 0, "ram_inactive": 0, "lru": 0})
        executor.execute(graph, "external-cpu-full", execute_outputs=["end", "observe"])
        assert executor.success, executor.status_messages
        assert calls == ["first", "second"] * 3, calls
        # Completion query must skip the external graph entirely, even with cache reuse.
        executor.execute(graph, "external-cpu-complete-query", execute_outputs=["end", "observe"])
        assert executor.success, executor.status_messages
        assert len(calls) == 6, calls
        review_graph = copy.deepcopy(graph)
        review_graph["start"]["inputs"].update(generation_mode=seq.REVIEW, run_name="cpu_review")
        executor.execute(review_graph, "external-cpu-review", execute_outputs=["end", "observe"])
        assert executor.success, executor.status_messages
        assert len(calls) == 8, calls
        executor.execute(review_graph, "external-cpu-review-query", execute_outputs=["end", "observe"])
        assert executor.success, executor.status_messages
        assert len(calls) == 8, calls
        review_graph["start"]["inputs"]["review_action"] = seq.ACTIONS[1]
        executor.execute(review_graph, "external-cpu-review-continue", execute_outputs=["end", "observe"])
        assert executor.success, executor.status_messages
        assert len(calls) == 10, calls
        review_graph["start"]["inputs"]["review_action"] = seq.ACTIONS[2]
        executor.execute(review_graph, "external-cpu-review-retry", execute_outputs=["end", "observe"])
        assert executor.success, executor.status_messages
        assert len(calls) == 12, calls
        review_graph["start"]["inputs"]["review_action"] = seq.ACTIONS[3]
        executor.execute(review_graph, "external-cpu-review-finish", execute_outputs=["end", "observe"])
        assert executor.success, executor.status_messages
        assert len(calls) == 12, calls
        review_graph["start"]["inputs"]["review_action"] = seq.ACTIONS[1]
        executor.execute(review_graph, "external-cpu-finished-continue-query", execute_outputs=["end", "observe"])
        assert executor.success and len(calls) == 12
        review_graph["start"]["inputs"]["review_action"] = seq.ACTIONS[4]
        executor.execute(review_graph, "external-cpu-review-restart-all", execute_outputs=["end", "observe"])
        assert executor.success and len(calls) == 14
        reference_graph = None
        if args.reference_gate:
            reference_graph = copy.deepcopy(graph)
            reference_graph["start"]["inputs"].update(run_name="cpu_native_references", prompt_mode="List", sequence_prompt="@R1 walks\n---\n@R4 walks\n---\nA quiet corridor")
            reference_graph.pop("template")
            reference_graph["assets"] = {"class_type":"H3CpuReferenceAssets", "inputs":{}}
            for key, size in [("low_cond",64),("high_cond",128)]:
                reference_graph[key] = {"class_type":"H3ContinuumExternalConditioning_design61", "inputs":dict(
                    clip=["assets",1], video_vae=["assets",2], prompt=["start",3], width=size, height=size,
                    length=["start",5] if key=="low_cond" else ["prepare",4], chunks=7, chunk_index=3,
                    reference_size="Match Output", reference_images=["assets",3], sequence_flow=["start",0])}
            reference_graph["prepare"]["inputs"].update(model=["assets",0], conditioning=["low_cond",0], latent=["low_cond",1], context_geometry="Project context to template resolution")
            reference_graph["high_prepare"] = copy.deepcopy(reference_graph["prepare"])
            reference_graph["high_prepare"]["inputs"].update(conditioning=["high_cond",0], latent=["high_cond",1], source_plan=["prepare",3], context_geometry="Keep State resolution")
            reference_graph["upscale"] = {"class_type":"H3CpuUpscale", "inputs":{"latent":["first",0]}}
            reference_graph["first"]["inputs"]["conditioning"] = ["prepare",1]
            reference_graph["second"]["inputs"].update(latent=["upscale",0], conditioning=["high_prepare",1])
            reference_graph["end"]["inputs"]["plan"] = ["high_prepare",3]
            before = len(calls)
            executor.execute(reference_graph, "external-cpu-native-reference-full", execute_outputs=["end","observe"])
            assert executor.success, executor.status_messages
            assert len(calls)==before+6
            assert matches(reference_calls, [[0.25],[0.25],[0.75],[0.75],[],[]]), reference_calls
            executor.execute(reference_graph, "external-cpu-native-reference-complete-query", execute_outputs=["end","observe"])
            assert executor.success and len(calls)==before+6 and len(reference_calls)==6
            reference_graph["start"]["inputs"].update(run_name="cpu_native_reference_review", generation_mode=seq.REVIEW)
            for action, expected in [(seq.ACTIONS[0],[[0.25],[0.25]]),(seq.ACTIONS[1],[[0.75],[0.75]]),(seq.ACTIONS[2],[[0.75],[0.75]]),(seq.ACTIONS[3],None),(seq.ACTIONS[4],[[0.25],[0.25]])]:
                reference_graph["start"]["inputs"]["review_action"]=action
                previous_calls, previous_refs = len(calls), len(reference_calls)
                executor.execute(reference_graph, "external-cpu-native-reference-review", execute_outputs=["end","observe"])
                assert executor.success, executor.status_messages
                if expected is None:
                    assert len(calls) == previous_calls and len(reference_calls) == previous_refs
                else:
                    assert matches(reference_calls[-2:], expected), reference_calls
            assert len(calls)==28 and len(reference_calls)==14
        print(json.dumps({"gate": "real ComfyUI PromptExecutor, synthetic CPU payloads", "result": "PASS",
            "reference_workflow": reference_graph, "reference_qwen_image_means": reference_calls,
            "synthetic_reference_inputs": {"R1":{"value":0.25,"shape":[1,256,256,3],"chunks":[1]}, "R4":{"value":0.75,"shape":[1,256,256,3],"chunks":[2]}} if reference_graph else {},
            "reference_stage_dimensions": [[64,64],[128,128]] if reference_graph else [],
            "external_pass_calls": len(calls), "weights_or_gpu_sampling": False, "elapsed_seconds": round(time.perf_counter()-started, 3),
            "workflow": graph, "prompt": "CPU simulation", "media_inputs": [], "model": "synthetic_model", "lora": None,
            "sampler": "synthetic CPU add-one per stage; no SIGMAS", "steps": "not diffusion sampling", "dimensions": [64,64],
            "chunks": 3, "seconds_per_chunk": 5, "observed": "Full3, completion query, Review pause/query/continue/retry/finish-here/restart PASS; Finish calls zero samplers or encoders. Optional native reference gate checks LOW/HIGH automatic R1/R4/empty routing despite deliberately incorrect manual indices, plus restart from R1."}))


if __name__ == "__main__": main()
