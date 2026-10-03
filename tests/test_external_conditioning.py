"""Reference helper selection reaches Qwen, VAE and DiT in both external stages."""
import sys
import types

import pytest
import torch

from ComfyUI_design61_tools._h3.constants import CONTINUITY_OPTIONS, CONTINUITY_FRAMES, MARK_VIDEO_CONTEXT, FPS
from ComfyUI_design61_tools._h3.reference_audio import H3ContinuumReferenceAudios_design61
from ComfyUI_design61_tools._h3.temporal import video_latent_t
from ComfyUI_design61_tools._h3.video_reference_modes import FOLLOW_TIMELINE, REPEAT_REFERENCE
from ComfyUI_design61_tools._h3.reference_video import REFERENCE_VIDEO_SIZE_MATCH_OUTPUT
from ComfyUI_design61_tools._h3.state import extract_av_streams
from ComfyUI_design61_tools._h3.v3.external_conditioning import build_external_conditioning
from ComfyUI_design61_tools._h3.v3.reference_images_v39 import H3ContinuumReferenceImagesV39_design61
from ComfyUI_design61_tools._h3.v3 import external_sampling


class Nested:
    def __init__(self, parts): self.parts = tuple(parts)
    def unbind(self): return self.parts


class Clip:
    def __init__(self): self.calls = []
    def tokenize(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return prompt
    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.ones(1, 2, 3), {"prompt": tokens}]]


class VAE:
    def __init__(self): self.calls = []
    def encode(self, image):
        self.calls.append((float(image.mean()), *image.shape[1:3]))
        return torch.full((1, 24, 1, image.shape[1] // 16, image.shape[2] // 16), float(image.mean()))


@pytest.fixture(autouse=True)
def core(monkeypatch):
    comfy = types.ModuleType("comfy")
    comfy.utils = types.ModuleType("comfy.utils")
    comfy.utils.common_upscale = lambda x, w, h, *args: torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    comfy.model_management = types.ModuleType("comfy.model_management")
    comfy.model_management.intermediate_device = lambda: torch.device("cpu")
    comfy.nested_tensor = types.ModuleType("comfy.nested_tensor")
    comfy.nested_tensor.NestedTensor = Nested
    for name, value in [("comfy", comfy), ("comfy.utils", comfy.utils), ("comfy.model_management", comfy.model_management), ("comfy.nested_tensor", comfy.nested_tensor)]:
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(external_sampling, "clone_model_for_chunk", lambda model, **kwargs: model)


def image(value): return torch.full((1, 256, 256, 3), value)


def bundle():
    return H3ContinuumReferenceImagesV39_design61().pack(
        reference_use="Per chunk", reference_image_1=image(0.25), reference_r1_chunks="1",
        reference_image_4=image(0.75), reference_r4_chunks="2",
        reference_image_9=object(), reference_r9_chunks="off",
    )[0]


def build(**overrides):
    inputs = dict(clip=Clip(), video_vae=VAE(), prompt="@R1 walks", width=64, height=64, length=124,
                  chunks=3, reference_images=bundle())
    inputs.update(overrides)
    return build_external_conditioning(**inputs)


def test_both_resolutions_use_only_current_chunk_reference_and_reencode_it():
    references, vae, clip = bundle(), VAE(), Clip()
    for chunk, source, value in [(1, "R1", 0.25), (2, "R4", 0.75)]:
        flow = {"chunks": 3, "entries": (None,) * (chunk - 1), "prompts": ["@R1 walks", "@R4 walks", "no references"]}
        for size in (64, 128):
            result = build(clip=clip, video_vae=vae, reference_images=references, sequence_flow=flow,
                           width=size, height=size, prompt="ignored manual prompt", chunk_index=3)
            metadata = result[0][0][1]
            assert metadata["prompt"] == "<Picture 1> walks"
            assert vae.calls[-1] == pytest.approx((value, size, size))
            assert len(metadata["minimax_refs"]) == 1
            assert float(metadata["minimax_refs"][0]["latent"].mean()) == pytest.approx(value)
            assert len(clip.calls[-1][1]["minimax_ref_items"]) == 1
            assert f"selected {source}" in result[2]
            video, audio = extract_av_streams(result[1])
            assert video.shape[-2:] == (size // 16, size // 16)
            assert video.device.type == audio.device.type == "cpu"
    assert [item[0] for item in vae.calls] == pytest.approx([0.25, 0.25, 0.75, 0.75])
    assert references.selectors[0] == "1" and references.selectors[3] == "2"


def test_empty_chunk_does_not_fall_back_to_all_and_inactive_tags_are_nonblocking():
    vae, clip = VAE(), Clip()
    result = build(clip=clip, video_vae=vae, chunk_index=3, prompt="@R1 unavailable; malformed [")
    assert not vae.calls
    assert clip.calls[0][0] == "@R1 unavailable; malformed ["
    assert not result[0][0][1].get("minimax_refs")
    assert "not active" in result[2]


def test_all_chunks_and_disconnected_helper_keep_native_text_path():
    references = H3ContinuumReferenceImagesV39_design61().pack(
        reference_use="All chunks", reference_image_4=image(0.5), reference_r4_chunks="off",
    )[0]
    for chunk in (1, 3):
        result = build(reference_images=references, chunk_index=chunk, prompt="@R4 walks")
        assert result[0][0][1]["prompt"] == "<Picture 1> walks"
    result = build(reference_images=None, prompt="  exact fallback text  ", length=120)
    assert result[0][0][1]["prompt"] == "  exact fallback text  "
    assert result[0][0][1]["minimax_frame_count"] == 124


def test_retry_and_resume_derive_route_from_accepted_prefix_not_seed_or_manual_index():
    flow = {"chunks": 3, "entries": (None,), "prompts": ["@R1", "@R4", "none"], "nonce": 0}
    original = build(sequence_flow=flow, chunk_index=1)
    retry = build(sequence_flow={**flow, "nonce": 8}, chunk_index=3)
    assert original[0][0][1]["prompt"] == retry[0][0][1]["prompt"] == "<Picture 1>"
    assert original[2] == retry[2]


def test_keyframe_picture_offset_and_final_chunk_attachment():
    first, last = image(0.1), image(0.9)
    initial = build(first_frame=first, last_frame=last)
    assert initial[0][0][1]["prompt"] == "<Picture 3> walks"
    assert [item["resolved_frame_index"] for item in initial[0][0][1]["minimax_keyframes"]] == [0]
    final = build(first_frame=first, last_frame=last, chunk_index=3)
    assert [item["resolved_frame_index"] for item in final[0][0][1]["minimax_keyframes"]] == [0, 123]


def test_next_chunk_keeps_new_image_reference_alongside_previous_av_context():
    first = build(width=128, height=128)
    def prepare(values, **kwargs):
        return external_sampling.prepare_external_segment(
            model=object(), conditioning=values[0], latent=values[1], continuity=CONTINUITY_OPTIONS[0],
            extend_seconds=5.0, audio_continuity=True, **kwargs,
        )
    initial = prepare(first)
    captured = external_sampling.capture_external_segment(samples=initial[2], plan=initial[3])
    state = captured[1]
    before = state["video_tail"].clone()
    low = prepare(build(chunk_index=2, prompt="@R4 walks", length=141), previous_state=state,
                  context_geometry=external_sampling.PROJECT_CONTEXT)
    high = prepare(build(chunk_index=2, prompt="@R4 walks", length=141, width=128, height=128),
                   previous_state=state, source_plan=low[3])
    for stage in (low, high):
        refs = stage[1][0][1]["minimax_refs"]
        images = [ref for ref in refs if ref["kind"] == "image"]
        assert len(images) == 1
        assert float(images[0]["latent"].mean()) == pytest.approx(0.75)
        assert any(ref.get(MARK_VIDEO_CONTEXT) for ref in refs)
    assert torch.equal(before, state["video_tail"])


class AudioVAE:
    audio_sample_rate = 32000
    def encode(self, waveform):
        length = round(waveform.shape[1] / self.audio_sample_rate * 40)
        return torch.arange(length, dtype=torch.float32).reshape(1, 1, 1, -1).expand(1, 32, 2, -1).clone()


class VideoVAE(VAE):
    def encode(self, frames):
        self.calls.append((float(frames.mean()), *frames.shape[1:3]))
        return torch.full((1, 24, video_latent_t(len(frames)), frames.shape[1]//16, frames.shape[2]//16), float(frames.mean()))


def audio(seconds):
    return {"waveform": torch.zeros(1, 2, round(seconds*32000)), "sample_rate": 32000}


def test_original_reference_audio_bundle_and_driver_survive_both_continuation_stages():
    vae, driver = AudioVAE(), audio(20)
    references = H3ContinuumReferenceAudios_design61().pack(reference_audio_1=audio(2), reference_audio_2=audio(3), reference_audio_vae=vae)[0]
    source = driver["waveform"].clone()
    first = build(width=128, height=128, driving_audio=driver, audio_vae=vae, audio_references=references)
    prepared = external_sampling.prepare_external_segment(model=object(), conditioning=first[0], latent=first[1], continuity=CONTINUITY_OPTIONS[0], extend_seconds=5, audio_continuity=True)
    captured = external_sampling.capture_external_segment(samples=prepared[2], plan=prepared[3])
    entry = captured[5]["entries"][0]
    flow = {"chunks":3, "entries":(entry,), "prompts":["first", "<Audio 1> and <Audio 2>", "third"], "seconds":5, "continuity":CONTINUITY_OPTIONS[0], "physical_frames":141}
    start = round((entry["plan"]["net_frames"]-CONTINUITY_FRAMES[CONTINUITY_OPTIONS[0]])*40/FPS)
    low_plan = None
    for size in (64,128):
        inputs = build(width=size, height=size, sequence_flow=flow, driving_audio=driver, audio_vae=vae, audio_references=references)
        stage = external_sampling.prepare_external_segment(model=object(), conditioning=inputs[0], latent=inputs[1], continuity=CONTINUITY_OPTIONS[0], extend_seconds=5, audio_continuity=True, previous_state=captured[1], context_geometry=external_sampling.PROJECT_CONTEXT, source_plan=low_plan)
        low_plan = stage[3]
        meta = stage[1][0][1]
        reference_blocks = [item for item in meta["minimax_refs"] if item["kind"] == "audio" and not item.get("_h3_continuum_audio_context")]
        assert [item["ref_audio_t"] for item in reference_blocks] == [80,120]
        guides = [item["audio_latent"] for item in meta["minimax_keyframes"] if "audio_latent" in item]
        assert len(guides) == 1
        assert float(guides[0][0,0,0,0]) == start
        assert guides[0].shape[-1] == round((entry["plan"]["net_frames"]-CONTINUITY_FRAMES[CONTINUITY_OPTIONS[0]]+141)*40/FPS)-start
        assert "_h3_external_driving_guide_v1" not in meta
        assert meta["minimax_frame_count"] == stage[4]
    assert torch.equal(driver["waveform"], source)


def test_timeline_video_follow_advances_source_window_while_repeat_keeps_original_prefix():
    frames = torch.cat([torch.full((124,64,64,3),0.2), torch.full((120,64,64,3),0.8)])
    entry = {"plan":{"net_frames":124}}
    flow = {"chunks":3, "entries":(entry,), "prompts":["first","second","third"], "seconds":5, "continuity":CONTINUITY_OPTIONS[0], "physical_frames":141}
    values = {}
    for mode in (FOLLOW_TIMELINE, REPEAT_REFERENCE):
        result = build(video_vae=VideoVAE(), reference_images=None, reference_video_1=frames, video_reference_mode=mode,
                       video_reference_size=REFERENCE_VIDEO_SIZE_MATCH_OUTPUT, sequence_flow=flow)
        blocks = [item for item in result[0][0][1]["minimax_refs"] if item["kind"] == "video"]
        assert len(blocks) == 1
        values[mode] = float(blocks[0]["latent"].mean())
    assert values[FOLLOW_TIMELINE] == pytest.approx(0.8)
    assert values[REPEAT_REFERENCE] == pytest.approx(0.2)
    flow["entries"] = (entry, {"plan":{"net_frames":124}})
    exhausted = build(video_vae=VideoVAE(), reference_images=None, reference_video_1=frames, video_reference_mode=FOLLOW_TIMELINE,
                      video_reference_size=REFERENCE_VIDEO_SIZE_MATCH_OUTPUT, sequence_flow=flow)
    assert not exhausted[0][0][1].get("minimax_refs")
