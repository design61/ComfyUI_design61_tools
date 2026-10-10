"""Disk-backed external sequences; only the current decode is materialized.

This is a separate, disposable frame store. Main Run Storage and existing
full-latent Review Sessions are never migrated, resized, or deleted here.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import uuid

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ..constants import CONTINUITY_FRAMES, FPS
from ..run_storage import RunStorageError, _RunLock, _write_json, _pid_exists
from ..state import StateValidationError, validate_state
from ..v2.session import entry_to_state, make_chunk_entry, validate_chunk_entry
from ..v2.seeds import derive_chunk_seed
from .external_sequence import ExternalReviewStore, REVIEW, canonical_revision

LATENTS = "Latents (existing)"
FRAMES = "Frames + tail State (disk)"
HEADER = "h3_continuum_state_json"
LOG = logging.getLogger(__name__)


def frames_directory(value=""):
    import folder_paths
    output = Path(folder_paths.get_output_directory())
    candidate = Path(os.path.expandvars(os.path.expanduser(str(value)))) if value else Path("design61_sequences")
    return (candidate if candidate.is_absolute() else output / candidate).resolve()


def _owned(root, relative):
    """Reject corrupted paths before reading or deleting store-owned assets."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root or Path(relative).is_absolute():
        raise StateValidationError("frame store asset escapes its owned directory")
    return path


def _remove(root, relative):
    path = _owned(root, relative)
    if path.exists():
        # Windows may temporarily hold files open. Cleanup failure is diagnostic;
        # the already-published active chain must remain usable.
        try:
            shutil.rmtree(path)
        except OSError:
            LOG.warning("Could not remove discarded frame assets: %s", path, exc_info=True)


def save_tail(path, state, revision, contract):
    state = validate_state(state)
    metadata = {key: value for key, value in state.items() if key not in ("video_tail", "audio_tail")}
    metadata.update(_state_id=revision, _external_contract=contract)
    temporary = path.with_suffix(".tmp")
    try:
        _write_json(path.with_suffix(".json"), metadata)
        save_file({key: state[key].contiguous() for key in ("video_tail", "audio_tail")}, str(temporary),
                  metadata={HEADER: json.dumps(metadata, sort_keys=True)})
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_tail(entry):
    path = _owned(Path(entry["store_root"]), entry["asset"] + "/tail.safetensors")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        state = json.loads((handle.metadata() or {}).get(HEADER, "null"))
        if not isinstance(state, dict) or state.get("_state_id") != entry["revision"] or state.get("_external_contract") != entry["contract"]:
            raise StateValidationError("frame tail State provenance does not match its index")
        state.update(video_tail=handle.get_tensor("video_tail"), audio_tail=handle.get_tensor("audio_tail"))
    state = validate_state(state)
    if state["clip_index"] != entry["clip_index"] or (state["width"], state["height"]) != (entry["plan"]["width"], entry["plan"]["height"]):
        raise StateValidationError("frame tail State topology/index differs")
    return state


def decode_video(video_vae, entry):
    from nodes import VAEDecode
    return VAEDecode().decode(video_vae, {"samples": entry["video"]})[0]


def decode_audio(audio_vae, entry):
    from comfy_extras.nodes_audio import vae_decode_audio
    return vae_decode_audio(audio_vae, {"samples": entry["audio"]})


def _fit_audio(waveform, length):
    if waveform.shape[-1] >= length:
        return waveform[..., :length].contiguous()
    return torch.nn.functional.pad(waveform, (0, length - waveform.shape[-1])).contiguous()


class FrameReviewStore(ExternalReviewStore):
    def __init__(self, run_name, contract, frame_root=""):
        self.contract = contract
        self.key = hashlib.sha256(str(run_name).encode("utf-8")).hexdigest()[:16] + "_" + contract[:16]
        self.root = frames_directory(frame_root) / self.key
        self.path = self.root / "index.json"

    def index(self):
        index = super().index()
        if self.path.exists() and index.get("storage_mode") != FRAMES:
            raise StateValidationError("frame store index has an incompatible storage contract")
        index["storage_mode"] = FRAMES
        return index

    def entries(self, index, head=None):
        cursor = index["head"] if head is None else head
        result, visited = [], set()
        while cursor:
            if cursor in visited or cursor not in index["records"]:
                raise StateValidationError("frame Take ancestry is corrupt")
            visited.add(cursor)
            record = index["records"][cursor]
            asset = _owned(self.root, record["asset"])
            if record.get("discarded") or not (asset / "tail.safetensors").is_file() or not (asset / "frames").is_dir():
                raise StateValidationError("active frame Take is missing required stored payloads")
            result.append(dict(record, clip_index=record["chunk"], revision=cursor, contract=self.contract, store_root=str(self.root)))
            cursor = record["parent"]
        result.reverse()
        if any(entry["clip_index"] != i for i, entry in enumerate(result, 1)):
            raise StateValidationError("active frame Take chain is not contiguous")
        return result

    def _view(self, entries, seconds, revision):
        """A flat FFmpeg folder of hard links; never load historical pixels."""
        relative = "views/" + revision
        view = _owned(self.root, relative)
        view.mkdir(parents=True)
        target = round(len(entries) * seconds * FPS)
        count, last = 0, None
        for entry in entries:
            source = _owned(self.root, entry["asset"] + "/frames")
            for number in range(entry["plan"]["net_frames"]):
                if count >= target:
                    break
                last = source / f"{number:08d}.png"
                self._link(last, view / f"{count:08d}.png")
                count += 1
        while count < target and last is not None:
            self._link(last, view / f"{count:08d}.png")
            count += 1
        if count != target:
            raise StateValidationError("frame sequence cannot assemble its declared duration")
        _write_json(view / ".design61-frame-view.json", {"store_root": str(self.root), "revision": revision, "frames": target, "fps": FPS})
        return relative

    @staticmethod
    def _link(source, destination):
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)

    def _prune(self, index, retained):
        for revision, record in index["records"].items():
            if revision not in retained and not record.get("discarded"):
                _remove(self.root, record["asset"])
                record["discarded"] = True
        active_view = index.get("view")
        views = self.root / "views"
        if views.exists():
            for view in views.iterdir():
                if not view.is_dir() or view.name == Path(active_view or "").name:
                    continue
                # The folder encoder pins managed views while FFmpeg reads them.
                pinned = False
                for pin in view.glob(".reader-*.json"):
                    try:
                        pid = int(json.loads(pin.read_text(encoding="utf-8"))["pid"])
                        pinned |= _pid_exists(pid)
                    except (OSError, ValueError, KeyError):
                        pinned = True
                if not pinned:
                    _remove(self.root, "views/" + view.name)

    def commit_frames(self, flow, entry, video_vae, audio_vae=None, driving_audio=None):
        from PIL import Image
        from .external_control import guard_stop
        guard_stop(flow)
        entry = validate_chunk_entry(entry)
        plan = entry["plan"]
        if entry["clip_index"] != len(flow["entries"]) + 1 or (plan["width"], plan["height"]) != (entry["video"].shape[-1]*16, entry["video"].shape[-2]*16):
            raise StateValidationError("external frame latent topology does not match the current HIGH plan")
        revision = uuid.uuid4().hex
        asset = "takes/" + revision
        stage = _owned(self.root, asset)
        stage.mkdir(parents=True)
        published, view = False, None
        try:
            frames = stage / "frames"
            frames.mkdir()
            images = decode_video(video_vae, entry)
            if not torch.is_tensor(images) or images.ndim != 4 or tuple(images.shape[:3]) != (plan["total_frames"], plan["height"], plan["width"]) or images.shape[-1] not in (3, 4):
                raise StateValidationError("decoded video cannot be assembled with its declared physical frame plan")
            for number, frame in enumerate(images[plan["trim_frames"]:plan["total_frames"]]):
                guard_stop(flow)
                if not bool(torch.isfinite(frame).all()):
                    raise StateValidationError("decoded video frame contains NaN or Inf")
                pixels = frame.detach().to("cpu").clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
                Image.fromarray(pixels).save(frames / f"{number:08d}.png", compress_level=1)
            del images
            tail = entry_to_state(entry, capacity_frames=CONTINUITY_FRAMES[flow["continuity"]])
            save_tail(stage / "tail.safetensors", tail, revision, self.contract)
            sound = None
            cursor = sum(item["plan"]["net_frames"] for item in flow["entries"])
            if driving_audio is not None:
                rate = int(driving_audio["sample_rate"])
                start, stop = round(cursor / FPS * rate), round((cursor + plan["net_frames"]) / FPS * rate)
                sound = _fit_audio(driving_audio["waveform"].detach().to("cpu")[..., start:stop], stop - start)
            elif audio_vae is not None:
                decoded = decode_audio(audio_vae, entry)
                rate = int(decoded["sample_rate"])
                trim = round(plan["trim_frames"] / FPS * rate)
                length = round((cursor + plan["net_frames"]) / FPS * rate) - round(cursor / FPS * rate)
                sound = _fit_audio(decoded["waveform"].detach().to("cpu")[..., trim:], length)
            if sound is not None:
                if sound.ndim != 3 or sound.shape[0] != 1 or not bool(torch.isfinite(sound).all()) or rate <= 0:
                    raise StateValidationError("decoded audio cannot be assembled safely")
                save_file({"waveform": sound.contiguous()}, str(stage / "audio.safetensors"), metadata={"sample_rate": str(rate)})
            record = {"asset": asset, "parent": flow["parent_head"], "chunk": entry["clip_index"], "seed": entry["seed"], "nonce": flow["nonce"],
                      "plan": plan, "prompt": entry["prompt"], "prompt_hash": entry["prompt_hash"],
                      "execution_contract": flow.get("execution_contract", self.contract), "audio_rate": rate if sound is not None else None,
                      "audio_channels": sound.shape[1] if sound is not None else None}
            if flow.get("_control"):
                record["execution_token"] = flow["_control"]
            lock = _RunLock(self.root / ".lock")
            lock.acquire()
            try:
                guard_stop(flow)
                index = self.index()
                if index["head"] != flow["expected_head"] or canonical_revision(index) != flow["expected_revision"]:
                    raise RunStorageError("external review head changed during Sampling; refusing to overwrite another accepted Take")
                index["records"][revision] = record
                index.pop("stopped", None)
                complete = entry["clip_index"] >= flow["target_chunks"]
                status = "complete" if complete else "review_ready" if flow["mode"] == REVIEW else "in_progress"
                index.update(head=revision, revision=revision, mode=flow["mode"], status=status,
                             review_unit={"chunk": entry["clip_index"]} if flow["mode"] == REVIEW else None,
                             finalized_chunks=flow["target_chunks"] if complete else None, seconds=flow["seconds"], chunks=flow["chunks"])
                active = self.entries(index)
                if status != "in_progress":
                    view = "views/" + revision
                    self._view(active, flow["seconds"], revision)
                index["view"] = view
                # Publish the complete Take/view/index before deleting any old branch.
                retained = {item["revision"] for item in active}
                for key, item in index["records"].items():
                    if key not in retained:
                        item["discarded"] = True
                _write_json(self.path, index)
                published = True
                for key, item in index["records"].items():
                    if key not in retained:
                        _remove(self.root, item["asset"])
                self._prune(index, retained)
                return index
            finally:
                lock.release()
        finally:
            if not published:
                _remove(self.root, asset)
                if view:
                    _remove(self.root, view)

    def ensure_view(self, expected):
        lock = _RunLock(self.root / ".lock")
        lock.acquire()
        try:
            index = self.index()
            if canonical_revision(index) != expected or not index["head"]:
                return index
            path = _owned(self.root, index["view"]) if index.get("view") else None
            target = round(index["records"][index["head"]]["chunk"] * index["seconds"] * FPS)
            # Recreate a view after the user intentionally clears its files.
            if path is None or not path.is_dir() or len(list(path.glob("*.png"))) != target:
                index["view"] = self._view(self.entries(index), index["seconds"], uuid.uuid4().hex)
                _write_json(self.path, index)
            return index
        finally:
            lock.release()

    def assembled_audio(self, entries, seconds, driving_audio=None):
        if not entries:
            return None
        if driving_audio is not None:
            rate = int(driving_audio["sample_rate"])
            return {"sample_rate": rate, "waveform": _fit_audio(driving_audio["waveform"].detach().to("cpu"), round(len(entries)*seconds*rate))}
        sound_entries = [entry for entry in entries if entry.get("audio_rate")]
        if not sound_entries:
            return None
        rate, channels = sound_entries[0]["audio_rate"], sound_entries[0]["audio_channels"]
        waveform = torch.zeros((1, channels, round(len(entries)*seconds*rate)), dtype=torch.float32)
        cursor = 0
        for entry in entries:
            stop = min(waveform.shape[-1], round((cursor + entry["plan"]["net_frames"])/FPS*rate))
            start = min(waveform.shape[-1], round(cursor/FPS*rate))
            if entry.get("audio_rate"):
                if (entry["audio_rate"], entry["audio_channels"]) != (rate, channels):
                    raise StateValidationError("stored chunk audio rates/channels cannot be assembled")
                sound = load_file(str(_owned(self.root, entry["asset"] + "/audio.safetensors")), device="cpu")["waveform"]
                waveform[..., start:stop] = _fit_audio(sound, stop-start)
            cursor += entry["plan"]["net_frames"]
        return {"sample_rate": rate, "waveform": waveform}


def end_frame_sequence(*, flow, samples=None, plan=None, video_vae=None, audio_vae=None, driving_audio=None, dynprompt=None, unique_id=None):
    from comfy_execution.graph import ExecutionBlocker
    from .external_sequence import expand_next_iteration, review_payload
    from .external_control import guard_stop, track_commit
    guard_stop(flow)
    store = FrameReviewStore(flow["run_name"], flow["contract"], flow["frame_root"])
    if flow["active"]:
        text = flow["prompts"][len(flow["entries"])]
        entry = make_chunk_entry(latent=samples, plan=plan, prompt=text, prompt_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                 seed=derive_chunk_seed(flow["base_seed"], len(flow["entries"]), flow["nonce"]),
                                 context_frames=int(plan["trim_frames"]), motion_score=0.0, reused=False)
        index = store.commit_frames(flow, entry, video_vae, audio_vae, driving_audio)
        track_commit(flow, index)
        guard_stop(flow)
    else:
        index = store.index()
    entries = tuple(store.entries(index))
    ui = {"external_review": [review_payload(flow, index)]}
    if index["status"] == "in_progress" and flow["active"]:
        next_flow = dict(flow, entries=entries, expected_head=index["head"], expected_revision=canonical_revision(index), parent_head=index["head"], index=index, status=index["status"])
        expansion, result = expand_next_iteration(dynprompt, unique_id, next_flow)
        return {"ui": ui, "result": result, "expand": expansion}
    index = store.ensure_view(canonical_revision(index))
    state = load_tail(entries[-1]) if entries else ExecutionBlocker(None)
    path = str(_owned(store.root, index["view"])) if index.get("view") else ExecutionBlocker(None)
    sound = store.assembled_audio(entries, flow["seconds"], driving_audio)
    result = (*[ExecutionBlocker(None) for _ in range(3)], state,
              f"Frames on disk: {len(entries)}/{flow['chunks']} chunks. {index['status']}.", path, sound)
    return {"ui": ui, "result": result}


def pin_frame_view(path):
    """Protect a managed view during an external FFmpeg reader; ordinary folders unchanged."""
    path = Path(path).resolve()
    marker = path / ".design61-frame-view.json"
    if not marker.is_file():
        return None
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    root = Path(metadata["store_root"]).resolve()
    if path.parent != root / "views" or not (root / "index.json").is_file():
        return None
    lock = _RunLock(root / ".lock")
    lock.acquire()
    try:
        pin = path / (".reader-" + uuid.uuid4().hex + ".json")
        _write_json(pin, {"pid": os.getpid()})
        return pin
    finally:
        lock.release()
