"""Persistent review and public ComfyUI graph expansion for external sampling.

This namespace is independent of Main Run Storage. Each immutable Take uses
the accepted Session safetensors+JSON writer; the small index is atomic JSON.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import uuid

from ..constants import CONTINUITY_FRAMES, FPS, PROMPT_MODE_FIXED
from ..run_storage import RunStorageError, _RunLock, _write_json
from ..state import StateValidationError
from ..temporal import align_frame_count_up, make_extension_shape
from ..v2.prompts import make_prompt_plan
from ..v2.seeds import derive_chunk_seed
from ..v2.session import entry_to_state, make_session
from ..v2.session_io import load_session, save_session, session_directory
from .external_sampling import SEQUENCE_MAGIC, capture_external_segment
from .plan import prepare_physical_decode_entries

START = "H3ContinuumExternalSequenceStart_design61"
END = "H3ContinuumExternalSequenceEnd_design61"
REVIEW = "Review Each Chunk"
FULL = "Full Video"
ACTIONS = ("Start / Resume", "Use it and continue", "Try this chunk again", "Finish here", "Start again from Chunk 1",
           "Continue all remaining chunks", "Regenerate from selected chunk")
LEGACY_FINISH = "Use it and finish the rest"


def canonical_revision(index):
    return index.get("revision", index["head"])


def sequence_progress(flow, index=None):
    completed = len(flow["entries"])
    active = bool(flow["active"])
    status = "running" if active else flow["status"]
    if index is not None:
        completed = index["records"][index["head"]]["chunk"] if index["head"] else 0
        active = False
        status = index["status"]
    return {"mode": flow["mode"], "status": status, "chunks": flow["chunks"],
            "completed": completed, "current": completed + 1 if active else completed if status == "review_ready" else None,
            "remaining": max(0, flow["chunks"]-completed) if status != "complete" else 0,
            "skipped": max(0, flow["chunks"]-completed) if status == "complete" else 0,
            "seconds": flow["seconds"], "total_seconds": flow["chunks"]*flow["seconds"],
            "completed_seconds": completed*flow["seconds"]}


def graph_contract(prompt, start_id):
    """Snapshot execution inputs, ignoring only review presentation/actions."""
    descriptor = copy.deepcopy(prompt)
    for node in descriptor.values():
        node.pop("_meta", None)
        node.pop("is_changed", None)  # Core cache annotation is not a graph input.
        if node.get("class_type") == START:
            for key in ("review_action", "expected_revision", "generation_mode", "run_name", "iteration"):
                node.get("inputs", {}).pop(key, None)
        if node.get("class_type") == "H3ContinuumExternalConditioning_design61" and "sequence_flow" in node.get("inputs", {}):
            # These manual reserves are superseded by the actual controller.
            for key in ("chunks", "chunk_index", "prompt", "length", "chunk_seconds"):
                node["inputs"].pop(key, None)
    raw = json.dumps(descriptor, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256((str(start_id) + raw).encode("utf-8")).hexdigest()


def review_reference(value):
    """Read a button's lineage reference; old plain revision strings still load."""
    try:
        reference = json.loads(value)
    except (TypeError, ValueError):
        return str(value or ""), None, None
    if not isinstance(reference, dict):
        return str(value or ""), None, None
    contract = reference.get("contract")
    if not isinstance(contract, str) or len(contract) != 64 or any(c not in "0123456789abcdef" for c in contract):
        contract = None
    return str(reference.get("revision") or ""), contract, reference.get("run_name")


def review_store(run_name, execution_contract, expected_revision):
    revision, anchor, anchor_name = review_reference(expected_revision)
    if anchor:
        name = str(anchor_name) if anchor_name is not None else run_name
        return ExternalReviewStore(name, anchor), name, revision
    store = ExternalReviewStore(run_name, execution_contract)
    if revision and canonical_revision(store.index()) != revision:
        # A panel drawn by the previous frontend carries only a revision. Find
        # that existing lineage by its saved revision, without loading tensors
        # or changing any index. UUID revisions uniquely identify their store.
        prefix = hashlib.sha256(str(run_name).encode("utf-8")).hexdigest()[:16]
        matches = []
        for path in (session_directory() / "external_sampling").glob(prefix + "_*/index.json"):
            try:
                index = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(index, dict) or index.get("magic") != "H3_EXTERNAL_REVIEW":
                continue
            if revision == index.get("revision", index.get("head")) or revision in index.get("records", {}):
                contract = index.get("contract", "")
                if isinstance(contract, str) and len(contract) == 64 and all(c in "0123456789abcdef" for c in contract):
                    matches.append(ExternalReviewStore(run_name, contract))
        if len(matches) == 1:
            store = matches[0]
    return store, run_name, revision


def selected_review_chunk(reference, accepted):
    """Read the explicit End selection; malformed/stale choices are read-only."""
    try:
        value = json.loads(reference).get("restart_chunk")
    except (TypeError, ValueError, AttributeError):
        return None
    if type(value) is int and 1 <= value <= accepted:
        return value
    return None


class ExternalReviewStore:
    def __init__(self, run_name, contract):
        self.contract = contract
        self.key = hashlib.sha256(str(run_name).encode("utf-8")).hexdigest()[:16] + "_" + contract[:16]
        self.root = session_directory() / "external_sampling" / self.key
        self.path = self.root / "index.json"

    def index(self):
        if not self.path.exists():
            return {"magic": "H3_EXTERNAL_REVIEW", "version": 1, "contract": self.contract, "head": "", "status": "in_progress", "records": {}}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if value.get("magic") != "H3_EXTERNAL_REVIEW" or value.get("version") != 1 or value.get("contract") != self.contract:
            raise StateValidationError("external review index schema/contract is corrupt")
        return value

    def entries(self, index, head=None):
        cursor = index["head"] if head is None else head
        records, entries, visited = index["records"], [], set()
        while cursor:
            if cursor in visited or cursor not in records:
                raise StateValidationError("external review Take ancestry is corrupt")
            visited.add(cursor)
            record = records[cursor]
            session = load_session(prefix=record["prefix"], slot=1)
            settings = session["settings"]
            if settings.get("external_contract") != self.contract or settings.get("external_revision") != cursor or settings.get("external_parent") != record["parent"] or len(session["chunks"]) != 1:
                raise StateValidationError("external Take Session does not match its index contract")
            entry = session["chunks"][0]
            if entry["clip_index"] != record["chunk"] or entry["seed"] != record["seed"]:
                raise StateValidationError("external Take Session/index provenance differs")
            entries.append(entry)
            cursor = record["parent"]
        entries.reverse()
        if any(entry["clip_index"] != i for i, entry in enumerate(entries, 1)):
            raise StateValidationError("external review Take chain is not contiguous")
        return entries

    def commit(self, flow, entry):
        self.root.mkdir(parents=True, exist_ok=True)
        lock = _RunLock(self.root / ".lock")
        lock.acquire()
        try:
            index = self.index()
            if index["head"] != flow["expected_head"] or canonical_revision(index) != flow["expected_revision"]:
                raise RunStorageError("external review head changed during Sampling; refusing to overwrite another accepted Take")
            revision = uuid.uuid4().hex
            prefix = f"external_{self.key}_{revision}"
            plan = entry["plan"]
            session = make_session(
                chunks=[entry], width=plan["width"], height=plan["height"],
                chunk_seconds=plan["requested_extend_seconds"], identity_hash=self.contract,
                model_fingerprint_value=self.contract, parent_session_id=None,
                reroll_from_chunk=0, settings={"external_contract": self.contract, "external_revision": revision, "external_parent": flow["parent_head"],
                                             "external_execution_contract": flow.get("execution_contract", self.contract)},
            )
            save_session(session, prefix=prefix, slot=1)
            index["records"][revision] = {"prefix": prefix, "parent": flow["parent_head"], "chunk": entry["clip_index"], "seed": entry["seed"], "nonce": flow["nonce"],
                                           "execution_contract": flow.get("execution_contract", self.contract)}
            complete = entry["clip_index"] >= flow["target_chunks"]
            status = "complete" if complete else "review_ready" if flow["mode"] == REVIEW else "in_progress"
            index.update(head=revision, revision=revision, mode=flow["mode"], status=status,
                         review_unit={"chunk": entry["clip_index"]} if flow["mode"] == REVIEW else None,
                         finalized_chunks=flow["target_chunks"] if complete else None)
            _write_json(self.path, index)
            return index
        finally:
            lock.release()

    def finish_here(self, expected):
        """Finalize the accepted prefix atomically without invoking sampling."""
        self.root.mkdir(parents=True, exist_ok=True)
        lock = _RunLock(self.root / ".lock")
        lock.acquire()
        try:
            index = self.index()
            if canonical_revision(index) != expected:
                return index
            if index["head"]:
                index.update(status="complete", finalized_chunks=index["records"][index["head"]]["chunk"], revision=uuid.uuid4().hex)
                _write_json(self.path, index)
            return index
        finally:
            lock.release()


def begin_sequence(*, chunks, chunk_seconds, continuity, base_seed, sequence_prompt, prompt_mode,
                   generation_mode, review_action, run_name, expected_revision="", prompt=None,
                   unique_id=None, iteration=None):
    if iteration is not None:
        flow = dict(iteration)
    else:
        generation_mode = REVIEW if generation_mode in (REVIEW, "Review by Chunk") else FULL
        review_action = ACTIONS[3] if review_action == LEGACY_FINISH else review_action
        if generation_mode == FULL and review_action not in ACTIONS[5:]:
            review_action = ACTIONS[0]
        execution_contract = graph_contract(prompt or {}, unique_id)
        # Review commands belong to the panel's accepted lineage, even when
        # upstream text changes. Normal Queue has no reference and starts fresh.
        if review_action == ACTIONS[0] and not expected_revision:
            store, revision = ExternalReviewStore(run_name, execution_contract), ""
        else:
            store, run_name, revision = review_store(run_name, execution_contract, expected_revision)
        contract = store.contract
        index = store.index()
        entries = store.entries(index)
        head = index["head"]
        nonce = int(index["records"].get(head, {}).get("nonce", 0))
        parent_head = head
        active = len(entries) < int(chunks) and index["status"] not in ("review_ready", "complete")
        target_chunks = int(index.get("finalized_chunks") or chunks)
        if revision and revision != canonical_revision(index):
            active = False  # Stale browser action is diagnostic, never a destructive rewrite.
        elif review_action == ACTIONS[0] and not revision:
            entries, parent_head, target_chunks, nonce = [], "", int(chunks), 0
            active = True  # Blue Queue: a new sequence; old Takes stay immutable.
        elif review_action == ACTIONS[3]:
            index = store.finish_here(canonical_revision(index))
            target_chunks = int(index.get("finalized_chunks") or chunks)
            active = False
        elif review_action == ACTIONS[4]:
            entries, parent_head, target_chunks = [], "", int(chunks)
            nonce += 1
            active = True
        elif review_action == ACTIONS[5]:
            generation_mode = FULL
            active = len(entries) < target_chunks and index["status"] != "complete"
        elif review_action == ACTIONS[6]:
            selected = selected_review_chunk(expected_revision, len(entries))
            active = selected is not None
            if active:
                # Branch from the selected segment's predecessor. Old later
                # Takes leave the active chain only when the new Take commits.
                cursor = head
                while cursor and index["records"][cursor]["chunk"] >= selected:
                    cursor = index["records"][cursor]["parent"]
                entries, parent_head = entries[:selected - 1], cursor
                target_chunks, generation_mode = int(chunks), REVIEW
                nonce += 1
        elif head and review_action == ACTIONS[2] and index.get("review_unit"):
            entries = entries[:-1]
            parent_head = index["records"][head]["parent"]
            nonce += 1
            active = True
        elif generation_mode == FULL:
            active = len(entries) < target_chunks and index["status"] != "complete"
        elif review_action == ACTIONS[1]:
            active = len(entries) < target_chunks and index["status"] != "complete"
        plans = make_prompt_plan(mode=prompt_mode, script=sequence_prompt, chunks=int(chunks), chunk_seconds=float(chunk_seconds))
        if plans["mode"] == PROMPT_MODE_FIXED:
            plans["prompts"] = [str(sequence_prompt)] * int(chunks)
        flow = {"contract": contract, "execution_contract": execution_contract, "run_name": run_name, "chunks": int(chunks), "seconds": float(chunk_seconds),
                "continuity": continuity, "base_seed": int(base_seed), "prompts": plans["prompts"],
                "mode": generation_mode, "target_chunks": target_chunks, "entries": tuple(entries),
                "expected_head": head, "expected_revision": canonical_revision(index), "parent_head": parent_head, "nonce": nonce, "active": active,
                "status": index["status"], "index": index}
    entries = flow["entries"]
    position = len(entries)
    state = entry_to_state(entries[-1]) if entries else None
    sequence = {"magic": SEQUENCE_MAGIC, "version": 1, "entries": entries} if entries else None
    retained = sum(entry["plan"]["net_frames"] for entry in entries)
    requested_frames = max(1, round((position + 1) * flow["seconds"] * FPS) - retained)
    flow["extend_seconds"] = requested_frames / FPS
    frames = make_extension_shape(CONTINUITY_FRAMES[flow["continuity"]], flow["extend_seconds"]).total_frames if state else align_frame_count_up(round(flow["seconds"] * FPS))
    flow["physical_frames"] = frames
    flow["chunk_index"] = position + 1
    seed = derive_chunk_seed(flow["base_seed"], position, flow["nonce"])
    return flow, state, sequence, flow["prompts"][min(position, flow["chunks"] - 1)], seed, frames


def sequence_outputs(entries, chunk_seconds=None):
    if not entries:
        raise StateValidationError("external sequence has no completed payload to decode")
    duration = float(chunk_seconds) if chunk_seconds is not None else sum(entry["plan"]["net_frames"] for entry in entries) / FPS / len(entries)
    decode, plan = prepare_physical_decode_entries(entries, chunk_seconds=duration, preserve_final_frame=False, terminal_merged=False)
    return [{"samples": entry["video"]} for entry in decode], [{"samples": entry["audio"]} for entry in decode], plan, entry_to_state(entries[-1])


def review_payload(flow, index):
    head = index["head"]
    records = index["records"]
    return {"status": index["status"], "revision": canonical_revision(index), "contract": flow["contract"], "run_name": flow["run_name"], "chunks": flow["chunks"], "mode": flow["mode"],
            "accepted": records[head]["chunk"] if head else 0,
            "accepted_chunks": list(range(1, records[head]["chunk"] + 1)) if head else [],
            "review_unit": index.get("review_unit"),
            "progress": sequence_progress(flow, index),
            "history": [{"revision": key, "chunk": record["chunk"], "seed": record["seed"]} for key, record in records.items()]}


def expand_next_iteration(dynprompt, end_id, next_flow):
    """Replay the editable graph between Start and End through public Core APIs."""
    from comfy_execution.graph_utils import GraphBuilder, is_link
    end_node = dynprompt.get_node(end_id)
    start_id = end_node["inputs"]["flow"][0]
    ancestors, depends = {}, {}

    def visit(node_id):
        if node_id in depends:
            return depends[node_id]
        node = dynprompt.get_node(node_id)
        ancestors[node_id] = node
        if node_id == start_id:
            depends[node_id] = True
            return True
        parents = [value[0] for value in node["inputs"].values() if is_link(value)]
        flags = [visit(parent) for parent in parents]
        depends[node_id] = any(flags)
        return depends[node_id]

    visit(end_id)
    selected = {key for key, flag in depends.items() if flag}
    builder = GraphBuilder()
    clones = {key: builder.node(ancestors[key]["class_type"], id=key) for key in selected}
    for key, clone in clones.items():
        clone.set_override_display_id(dynprompt.get_display_node_id(key))
        for name, value in ancestors[key]["inputs"].items():
            clone.set_input(name, clones[value[0]].out(value[1]) if is_link(value) and value[0] in clones else value)
    clones[start_id].set_input("iteration", next_flow)
    return builder.finalize(), tuple(clones[end_id].out(slot) for slot in range(5))


def end_sequence(*, flow, samples=None, plan=None, dynprompt=None, unique_id=None, driving_audio=None):
    store = ExternalReviewStore(flow["run_name"], flow["contract"])
    def outputs(entries):
        result = sequence_outputs(entries, flow["seconds"])
        if driving_audio is not None:
            # Assembly-only side channel; never serialized into Session State.
            result[2]["_h3_continuum_driving_audio_v1"] = {"waveform": driving_audio["waveform"].detach().to("cpu"), "sample_rate": driving_audio["sample_rate"]}
        return result
    if not flow["active"]:
        index = store.index()
        entries = store.entries(index)
        if not entries:
            from comfy_execution.graph import ExecutionBlocker
            return {"ui": {"external_review": [review_payload(flow, index)]}, "result": (*[ExecutionBlocker(None) for _ in range(4)], "No accepted chunks to finalize; no Sampling was started.")}
        return {"ui": {"external_review": [review_payload(flow, index)]}, "result": (*outputs(entries), f"External sequence {index['status']}: {len(entries)}/{flow['chunks']} chunks.")}
    captured = capture_external_segment(
        samples=samples, plan=plan, seed=derive_chunk_seed(flow["base_seed"], len(flow["entries"]), flow["nonce"]),
        previous_sequence={"magic": SEQUENCE_MAGIC, "version": 1, "entries": flow["entries"]} if flow["entries"] else None,
    )
    entries = captured[5]["entries"]
    # Record the actual prompt for the newly sampled Take only. Accepted prefix
    # entries and older Take Session files remain byte-for-byte untouched.
    entry = dict(entries[-1])
    entry["prompt"] = flow["prompts"][len(flow["entries"])]
    entry["prompt_hash"] = hashlib.sha256(entry["prompt"].encode("utf-8")).hexdigest()
    entries = (*entries[:-1], entry)
    index = store.commit(flow, entry)
    ui = {"external_review": [review_payload(flow, index)]}
    if index["status"] == "in_progress":
        next_flow = dict(flow, entries=entries, expected_head=index["head"], expected_revision=canonical_revision(index), parent_head=index["head"], index=index, status=index["status"])
        expansion, result = expand_next_iteration(dynprompt, unique_id, next_flow)
        return {"ui": ui, "result": result, "expand": expansion}
    return {"ui": ui, "result": (*outputs(entries), f"External sequence {index['status']}: {len(entries)}/{flow['chunks']} chunks.")}
