from __future__ import annotations

import copy
import sys
import types

import pytest
import torch

from ComfyUI_design61_tools._h3.constants import CONTINUITY_OPTIONS, MARK_VIDEO_CONTEXT
from ComfyUI_design61_tools._h3.state import StateValidationError, extract_av_streams
from ComfyUI_design61_tools._h3.temporal import audio_latent_t, video_latent_t
from ComfyUI_design61_tools._h3.v3 import external_sampling as runtime
from ComfyUI_design61_tools._h3.v3.external_sampling_nodes import NODE_CLASS_MAPPINGS


class Nested:
    def __init__(self, parts):
        self.parts = tuple(parts)
    def unbind(self):
        return self.parts


def latent(frames=124, size=4, value=0.0):
    return {"samples": Nested((
        torch.full((1, 24, video_latent_t(frames), size, size), value),
        torch.full((1, 32, 2, audio_latent_t(frames)), value),
    ))}


@pytest.fixture(autouse=True)
def fake_core(monkeypatch):
    nested = types.ModuleType("comfy.nested_tensor")
    nested.NestedTensor = Nested
    management = types.ModuleType("comfy.model_management")
    management.intermediate_device = lambda: torch.device("cpu")
    comfy = types.ModuleType("comfy")
    comfy.nested_tensor = nested
    comfy.model_management = management
    for name, value in [("comfy", comfy), ("comfy.nested_tensor", nested), ("comfy.model_management", management)]:
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(runtime, "clone_model_for_chunk", lambda model, **kwargs: {"original": model, **kwargs})


def prepare(**overrides):
    kwargs = dict(model=object(), conditioning=[[torch.ones(1, 2, 3), {"minimax_frame_count": 124}]],
                  latent=latent(), continuity=CONTINUITY_OPTIONS[0], extend_seconds=5.0,
                  audio_continuity=True)
    kwargs.update(overrides)
    return runtime.prepare_external_segment(**kwargs)


def test_first_segment_has_no_sampler_or_noise_and_preserves_template():
    source = latent()
    source["noise_mask"] = object()
    cond = [[torch.ones(1, 2, 3), {"minimax_frame_count": 124, "custom": "kept"}]]
    result = prepare(latent=source, conditioning=cond)
    assert result[2] is source
    assert result[1][0][1] is not cond[0][1]
    assert result[1][0][1]["custom"] == "kept"
    assert result[3]["trim_frames"] == 0
    assert result[4] == 124
    assert result[0]["strict"] is False


def test_two_resolutions_then_next_segment_uses_completed_high_output():
    low = prepare()
    high_template = latent(size=8)
    high = prepare(latent=high_template, source_plan=low[3])
    assert high[2] is high_template
    assert high[3]["total_frames"] == low[3]["total_frames"]
    assert high[3]["width"] == low[3]["width"] * 2
    completed = latent(size=8, value=3.0)
    first = runtime.capture_external_segment(samples=completed, plan=high[3], seed=123)
    state = first[1]
    before_video = state["video_tail"].clone()
    before_audio = state["audio_tail"].clone()
    low_next = prepare(previous_state=state, context_geometry=runtime.PROJECT_CONTEXT)
    high_next = prepare(latent=latent(size=8), previous_state=state, source_plan=low_next[3])
    for result, size in [(low_next, 4), (high_next, 8)]:
        ref = next(item for item in result[1][0][1]["minimax_refs"] if item[MARK_VIDEO_CONTEXT])
        assert ref["latent"].shape[-2:] == (size, size)
        assert torch.all(ref["latent"] == 3)
        assert torch.equal(ref["audio_latent"], state["audio_tail"][..., -audio_latent_t(22):])
        assert result[3]["trim_frames"] == 22
    assert low_next[4] == high_next[4]
    assert torch.equal(state["video_tail"], before_video)
    assert torch.equal(state["audio_tail"], before_audio)
    frames = high_next[4]
    second = runtime.capture_external_segment(samples=latent(frames=frames, size=8, value=4), plan=high_next[3], previous_sequence=first[5])
    assert len(second[2]) == len(second[3]) == 2
    assert second[4]["target_frames"] == 124 + frames - 22
    assert second[4]["chunks"][1]["trim_frames"] == 22
    assert all(item["samples"].device.type == "cpu" for item in second[2] + second[3])
    assert len(first[5]["entries"]) == 1
    assert torch.all(second[1]["video_tail"] == 4)


def test_state_geometry_requires_explicit_reference_projection():
    high = prepare(latent=latent(size=8))
    state = runtime.capture_external_segment(samples=latent(size=8), plan=high[3])[1]
    with pytest.raises(StateValidationError, match="automatic State resizing is disabled"):
        prepare(previous_state=state)


def test_high_plan_and_state_must_describe_same_physical_segment():
    low = prepare()
    bad = copy.deepcopy(low[3])
    bad["clip_index"] = 2
    with pytest.raises(StateValidationError, match="segment/overlap topology"):
        prepare(source_plan=bad)


def test_capture_rejects_corrupt_topology_before_exposing_state():
    plan = prepare()[3]
    with pytest.raises(StateValidationError, match="spatial topology"):
        runtime.capture_external_segment(samples=latent(size=8), plan=plan)
    with pytest.raises(ValueError, match="frames"):
        runtime.capture_external_segment(samples=latent(frames=141), plan=plan)
    bad = latent()
    extract_av_streams(bad)[0][..., -1, -1] = float("nan")
    with pytest.raises(RuntimeError, match="NaN"):
        runtime.capture_external_segment(samples=bad, plan=plan)


def test_capture_copies_cpu_output_and_preserves_audio():
    source = latent(value=1)
    result = runtime.capture_external_segment(samples=source, plan=prepare()[3])
    original_video, original_audio = extract_av_streams(source)
    output_video, output_audio = extract_av_streams(result[0])
    assert torch.equal(original_audio, output_audio)
    assert original_video.data_ptr() != output_video.data_ptr()
    assert original_audio.data_ptr() != output_audio.data_ptr()
    original_video.zero_()
    assert torch.all(result[1]["video_tail"] == 1)
    assert torch.all(result[2][0]["samples"] == 1)


def test_append_rejects_wrong_order_and_mixed_geometry():
    plan = prepare()[3]
    first = runtime.capture_external_segment(samples=latent(), plan=plan)
    with pytest.raises(StateValidationError, match="order"):
        runtime.capture_external_segment(samples=latent(), plan=plan, previous_sequence=first[5])
    next_plan = dict(plan, clip_index=2, width=128, height=128)
    with pytest.raises(ValueError, match="dimensions changed"):
        runtime.capture_external_segment(samples=latent(size=8), plan=next_plan, previous_sequence=first[5])


def test_all_new_public_inputs_have_help_and_no_sampling_controls():
    for cls in NODE_CLASS_MAPPINGS.values():
        schema = cls.INPUT_TYPES()
        for section in schema.values():
            for name, spec in section.items():
                assert spec[1]["tooltip"], name
        assert not {"sigmas", "sampler", "steps", "noise"} & set(schema["required"])
