from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import pytest
import torch

from ComfyUI_design61_tools._h3.constants import CONTINUITY_OPTIONS
from ComfyUI_design61_tools._h3.v2 import session_io
from ComfyUI_design61_tools._h3.v3 import external_sequence as sequence
from ComfyUI_design61_tools._h3.v3 import external_sampling as sampling
from ComfyUI_design61_tools._h3.v3.external_sequence_nodes import H3ContinuumExternalSequenceEnd_design61
from ComfyUI_design61_tools._h3.temporal import video_latent_t, audio_latent_t


class Nested:
    def __init__(self, parts): self.parts = parts
    def unbind(self): return self.parts


def samples(frames, value=1.0):
    return {"samples": Nested((torch.full((1, 24, video_latent_t(frames), 4, 4), value), torch.full((1, 32, 2, audio_latent_t(frames)), value)))}


@pytest.fixture(autouse=True)
def local_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(session_io, "session_directory", lambda: tmp_path)
    monkeypatch.setattr(sequence, "session_directory", lambda: tmp_path)
    comfy = types.ModuleType("comfy")
    nested = types.ModuleType("comfy.nested_tensor")
    nested.NestedTensor = Nested
    management = types.ModuleType("comfy.model_management")
    management.intermediate_device = lambda: torch.device("cpu")
    comfy.nested_tensor, comfy.model_management = nested, management
    for key, value in [("comfy", comfy), ("comfy.nested_tensor", nested), ("comfy.model_management", management)]:
        monkeypatch.setitem(sys.modules, key, value)
    monkeypatch.setattr(sampling, "clone_model_for_chunk", lambda model, **kwargs: model)


def begin(**changes):
    kwargs = dict(chunks=4, chunk_seconds=5.0, continuity=CONTINUITY_OPTIONS[0], base_seed=123,
                  sequence_prompt="Walk through a corridor.", prompt_mode="Auto", generation_mode=sequence.REVIEW,
                  review_action=sequence.ACTIONS[0], run_name="test", prompt={"1": {"class_type": sequence.START, "inputs": {"chunks": 4}}}, unique_id="1")
    kwargs.update(changes)
    return sequence.begin_sequence(**kwargs)


def finish(values, value=1.0):
    flow, state, _, prompt, seed, frames = values
    prepared = sampling.prepare_external_segment(model=object(), conditioning=[[torch.ones(1, 2, 3), {}]], latent=samples(frames, 0),
        continuity=CONTINUITY_OPTIONS[0], extend_seconds=5.0, audio_continuity=True, previous_state=state, sequence_flow=flow)
    return sequence.end_sequence(flow=flow, samples=samples(frames, value), plan=prepared[3])


def test_review_pause_query_continue_and_completion_are_backend_authoritative():
    first = finish(begin(chunks=2))
    payload = first["ui"]["external_review"][0]
    assert payload["status"] == "review_ready" and payload["accepted"] == 1
    query = begin(chunks=2)
    assert query[0]["active"] is False
    assert H3ContinuumExternalSequenceEnd_design61().check_lazy_status(query[0]) == []
    replay = sequence.end_sequence(flow=query[0])
    assert replay["ui"]["external_review"][0]["revision"] == payload["revision"]
    second = finish(begin(chunks=2, review_action=sequence.ACTIONS[1], expected_revision=payload["revision"]), 2)
    complete = second["ui"]["external_review"][0]
    assert complete["status"] == "complete" and complete["accepted"] == 2
    assert complete["review_unit"] == {"chunk": 2}
    assert len(second["result"][0]) == 2


def test_retry_retains_old_take_and_preserves_prefix(tmp_path):
    first = finish(begin(chunks=2))
    head = first["ui"]["external_review"][0]["revision"]
    old_files = list(tmp_path.glob("*.safetensors"))
    retry = begin(chunks=2, review_action=sequence.ACTIONS[2], expected_revision=head)
    assert retry[0]["nonce"] == 1 and len(retry[0]["entries"]) == 0
    reroll = finish(retry, 3)
    payload = reroll["ui"]["external_review"][0]
    assert len(payload["history"]) == 2 and payload["revision"] != head
    assert all(path.exists() for path in old_files)
    assert len(list(tmp_path.glob("*.safetensors"))) == 2
    next_values = begin(chunks=2, review_action=sequence.ACTIONS[1])
    assert torch.all(next_values[1]["video_tail"] == 3)


def test_full_video_expands_one_completed_segment_at_a_time(monkeypatch):
    pending = []
    def expansion(dynprompt, unique_id, flow):
        pending.append(flow)
        return {"next": {}}, (["next", 0],) * 5
    monkeypatch.setattr(sequence, "expand_next_iteration", expansion)
    values = begin(generation_mode=sequence.FULL)
    for number in range(4):
        result = finish(values, number + 1)
        if number < 3:
            assert "expand" in result
            assert pending[-1]["entries"][-1]["clip_index"] == number + 1
            values = begin(iteration=pending[-1])
        else:
            assert result["ui"]["external_review"][0]["status"] == "complete"
            assert result["ui"]["external_review"][0]["review_unit"] is None
            assert len(result["result"][0]) == 4
    assert len(pending) == 3


def test_stale_review_action_is_read_only_and_graph_changes_keep_old_lineage(tmp_path):
    first = finish(begin())
    stale = begin(review_action=sequence.ACTIONS[2], expected_revision="stale")
    assert not stale[0]["active"]
    changed = begin(prompt={"1": {"class_type": sequence.START, "inputs": {"chunks": 3}}})
    assert changed[0]["active"] and not changed[0]["entries"]
    assert list(tmp_path.glob("*.safetensors"))


def test_fixed_fallback_preserves_entered_prompt_and_duration_plan():
    entered = "  [malformed timeline]  "
    values = begin(sequence_prompt=entered, prompt_mode="Timeline", chunks=2)
    assert values[3] == entered
    first = finish(values)
    assert first["result"][2]["target_frames"] == 120
    next_values = begin(sequence_prompt=entered, prompt_mode="Timeline", chunks=2, review_action=sequence.ACTIONS[1])
    assert next_values[0]["extend_seconds"] == (240 - 124) / 24


def test_long_sequence_corrects_cumulative_temporal_rounding(monkeypatch):
    pending = []
    monkeypatch.setattr(sequence, "expand_next_iteration", lambda dyn, uid, flow: (pending.append(flow) or {}, (["next", 0],) * 5))
    values = begin(chunks=16, generation_mode=sequence.FULL)
    lengths = []
    for index in range(16):
        lengths.append(values[5])
        result = finish(values)
        if index < 15:
            values = begin(iteration=pending[-1])
    assert 158 in lengths
    assert result["result"][2]["target_frames"] == 16 * 5 * 24


def test_canonical_head_compare_and_swap_protects_concurrent_acceptance():
    first_values, competing = begin(), begin()
    finish(first_values)
    with pytest.raises(RuntimeError, match="head changed"):
        finish(competing)


def test_finish_here_finalizes_only_prefix_without_sampling_and_restart_retains_takes(tmp_path):
    first = finish(begin())
    revision = first["ui"]["external_review"][0]["revision"]
    cutoff = begin(review_action=sequence.ACTIONS[3], expected_revision=revision)
    assert not cutoff[0]["active"]
    assert H3ContinuumExternalSequenceEnd_design61().check_lazy_status(cutoff[0]) == []
    completed = sequence.end_sequence(flow=cutoff[0])
    payload = completed["ui"]["external_review"][0]
    assert payload["status"] == "complete" and payload["accepted"] == 1
    assert payload["revision"] != revision
    assert payload["progress"]["skipped"] == 3
    assert completed["result"][2]["target_frames"] == 120
    assert len(list(tmp_path.glob("*.safetensors"))) == 1
    # Retry the finished current chunk stays finished at that same cutoff.
    retry = begin(review_action=sequence.ACTIONS[2], expected_revision=payload["revision"])
    assert retry[0]["target_chunks"] == 1
    retried = finish(retry)
    assert retried["ui"]["external_review"][0]["status"] == "complete"
    restarted = begin(review_action=sequence.ACTIONS[4], expected_revision=retried["ui"]["external_review"][0]["revision"])
    assert restarted[0]["active"] and len(restarted[0]["entries"]) == 0
    assert restarted[0]["chunk_index"] == 1 and restarted[0]["target_chunks"] == 4
    result = finish(restarted)
    assert result["ui"]["external_review"][0]["status"] == "review_ready"
    assert len(list(tmp_path.glob("*.safetensors"))) == 3


def test_automatic_manual_reserves_and_core_cache_metadata_do_not_change_run_identity():
    graph={"1":{"class_type":sequence.START,"inputs":{"chunks":4}},"2":{"class_type":"H3ContinuumExternalConditioning_design61","inputs":{"sequence_flow":["1",0],"chunks":2,"chunk_index":1,"prompt":"manual reserve","length":124,"width":64}}}
    changed=copy.deepcopy(graph)
    changed["2"]["inputs"].update(chunk_index=2,chunks=3,length=141,prompt="changed unused reserve")
    changed["2"]["is_changed"]=123
    assert sequence.graph_contract(graph,"1")==sequence.graph_contract(changed,"1")
    changed["2"]["inputs"]["width"]=128
    assert sequence.graph_contract(graph,"1")!=sequence.graph_contract(changed,"1")


def test_full_mode_ignores_hidden_review_action_and_runs_remaining_groups(monkeypatch):
    finish(begin(chunks=2))
    next_values=begin(chunks=2,generation_mode=sequence.FULL,review_action=sequence.ACTIONS[3])
    assert next_values[0]["active"] and next_values[0]["chunk_index"]==2
    result=finish(next_values)
    assert result["ui"]["external_review"][0]["mode"]==sequence.FULL
    assert result["ui"]["external_review"][0]["review_unit"] is None


def test_finish_control_revision_prevents_sampling_that_was_inflight_overwriting_cutoff():
    first=finish(begin())
    pending=begin(review_action=sequence.ACTIONS[1])
    begin(review_action=sequence.ACTIONS[3],expected_revision=first["ui"]["external_review"][0]["revision"])
    with pytest.raises(RuntimeError,match="head changed"):
        finish(pending)


def test_finish_here_passes_original_sound_to_finalize_without_serializing_it(tmp_path):
    finish(begin())
    cutoff = begin(review_action=sequence.ACTIONS[3])
    original = {"waveform":torch.arange(32000*20,dtype=torch.float32).reshape(1,1,-1),"sample_rate":32000}
    result = sequence.end_sequence(flow=cutoff[0], driving_audio=original)
    plan = result["result"][2]
    assert plan["target_frames"] == 5*24
    saved_audio = plan["_h3_continuum_driving_audio_v1"]
    assert torch.equal(saved_audio["waveform"], original["waveform"])
    assert saved_audio["waveform"].device.type == "cpu"
    assert all("waveform" not in file.read_text(encoding="utf-8") for file in tmp_path.glob("*.json"))


def test_expansion_clones_only_start_dependent_sampling_nodes(monkeypatch):
    core = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(core))
    class DynamicPrompt:
        def __init__(self, graph): self.graph = graph
        def get_node(self, node_id): return self.graph[node_id]
        def get_display_node_id(self, node_id): return node_id
    graph = {
        "model": {"class_type": "UNETLoader", "inputs": {"unet_name": "any-model"}},
        "start": {"class_type": sequence.START, "inputs": {"chunks": 4}},
        "cond": {"class_type": "AnyConditioning", "inputs": {"prompt": ["start", 3]}},
        "sample": {"class_type": "AnyExternalSampler", "inputs": {"model": ["model", 0], "conditioning": ["cond", 0]}},
        "end": {"class_type": sequence.END, "inputs": {"flow": ["start", 0], "samples": ["sample", 0]}},
    }
    dynprompt = DynamicPrompt(graph)
    expanded, outputs = sequence.expand_next_iteration(dynprompt, "end", {"active": True})
    assert len(expanded) == 4
    assert all(node["class_type"] != "UNETLoader" for node in expanded.values())
    sample = next(node for node in expanded.values() if node["class_type"] == "AnyExternalSampler")
    assert sample["inputs"]["model"] == ["model", 0]
    assert graph["start"]["inputs"] == {"chunks": 4}
    assert len(outputs) == 5
