"""Run-scoped Full Video interruption; stores metadata, never latent tensors.

Uses Core's atomic interrupt_if_running(prompt_id), without patching Core or
interrupting a successor prompt. The canonical Review checkpoint is on disk.
"""
from __future__ import annotations

import asyncio
import threading
import uuid
import sys

_lock = threading.RLock()
_runs = {}
_epoch = uuid.uuid4().hex
_order = 0


def _server():
    module = sys.modules.get("server")
    return getattr(getattr(module, "PromptServer", None), "instance", None)


def _running(server, prompt_id):
    return any(item[1] == prompt_id for item in server.prompt_queue.get_current_queue_volatile()[0])


def guard_stop(flow):
    token = flow.get("_control")
    with _lock:
        stopped = bool(token and _runs.get(token, {}).get("stop_requested"))
    if stopped:
        from comfy.model_management import InterruptProcessingException
        raise InterruptProcessingException()


def _metadata(flow):
    names = ("contract", "run_name", "chunks", "seconds", "mode", "target_chunks", "storage_mode", "frame_root",
             "expected_head", "expected_revision", "parent_head", "chunk_index", "nonce")
    return {name: flow[name] for name in names if name in flow}


def _state(run):
    if run.get("paused"):
        return run["paused"]
    if run.get("complete"):
        return run["last_state"]
    from .external_sequence import sequence_progress
    flow = dict(run["flow"], entries=[None] * run["accepted"], active=True, status="in_progress", _control=run["token"])
    return {"mode": "Full Video", "status": "in_progress", "accepted": run["accepted"],
            "chunks": flow["chunks"], "progress": sequence_progress(flow), "control_token": run["token"]}


def _packet(run, server):
    return {"token": run["token"], "epoch": _epoch, "order": run["order"], "serial": run["serial"], "prompt_id": run["prompt_id"], "start_id": run["start_id"], "end_ids": run["end_ids"],
            "running": _running(server, run["prompt_id"]), "stop_requested": run["stop_requested"],
            "ready": bool(run.get("paused")), "complete": run.get("complete", False), "state": _state(run)}


def _emit(run, server):
    server.send_sync("design61.external_control", _packet(run, server), run["client_id"])


def track_start(flow, prompt, start_id):
    global _order
    if not flow["active"] or flow["mode"] != "Full Video":
        return
    server = _server()
    if server is None or not getattr(server, "prompt_queue", None):
        return  # Offline CPU callers have no live prompt to interrupt.
    with _lock:
        token = flow.get("_control")
        if token in _runs:
            guard_stop(flow)
            run = _runs[token]
            run["flow"] = _metadata(flow)
            run["accepted"] = len(flow["entries"])
            run["serial"] += 1
        else:
            from .external_sequence import START, END, graph_contract
            candidates = [item for item in server.prompt_queue.get_current_queue_volatile()[0]
                          if item[2].get(str(start_id), {}).get("class_type") == START
                          and graph_contract(item[2], start_id) == graph_contract(prompt or {}, start_id)]
            if len(candidates) != 1:
                return
            item = candidates[0]
            # End flow links are part of the expansion adapter's existing contract.
            end_ids = [key for key, node in item[2].items() if node.get("class_type") == END
                       and node.get("inputs", {}).get("flow") == [str(start_id), 0]]
            if not end_ids:
                return
            for old in list(_runs):
                if len(_runs) < 64:
                    break
                if not _running(server, _runs[old]["prompt_id"]):
                    del _runs[old]
            token = uuid.uuid4().hex
            _order += 1
            flow["_control"] = token
            run = {"token": token, "order": _order, "serial": 1, "prompt_id": item[1], "start_id": str(start_id), "end_ids": end_ids,
                   "client_id": item[3].get("client_id"), "stop_requested": False,
                   "flow": _metadata(flow), "accepted": len(flow["entries"])}
            _runs[token] = run
        _emit(run, server)


def track_commit(flow, index):
    token = flow.get("_control")
    server = _server() if token else None
    with _lock:
        run = _runs.get(token)
        if run is None or server is None:
            return
        # Stop may already have read this committed record under the store lock.
        # Never replace its canonical paused payload with a delayed Full event.
        if not run["stop_requested"]:
            from .external_sequence import review_payload
            run["accepted"] = index["records"][index["head"]]["chunk"]
            run["complete"] = index["status"] == "complete"
            run["last_state"] = review_payload(flow, index)
            run["serial"] += 1
            _emit(run, server)


def status(token):
    server = _server()
    with _lock:
        if server is None or token not in _runs:
            raise ValueError("This Full Video run is no longer available.")
        packet = _packet(_runs[token], server)
        if not packet["stop_requested"] and not packet["running"]:
            packet["state"] = _runs[token].get("last_state", packet["state"])
        return packet


def request_stop(token):
    server = _server()
    with _lock:
        run = _runs.get(token)
        if server is None or run is None:
            raise ValueError("This Full Video run is no longer available.")
        if run["stop_requested"]:
            return _packet(run, server)
        if run.get("complete"):
            return _packet(run, server)
        run["stop_requested"] = True
        run["serial"] += 1
        if not server.prompt_queue.interrupt_if_running(run["prompt_id"]):
            run["stop_requested"] = False
            raise ValueError("This prompt has already ended; no other prompt was interrupted.")
        metadata = dict(run["flow"])
    # Do not hold the registry lock while waiting for the transaction lock.
    from .external_sequence import sequence_store, review_payload
    store = sequence_store(metadata["run_name"], metadata["contract"], metadata.get("storage_mode"), metadata.get("frame_root", ""))
    index = store.pause_for_review(metadata, token)
    state = review_payload(dict(metadata, entries=(), active=False, status=index["status"]), index)
    state["control_token"] = token
    with _lock:
        run["paused"] = state
        run["serial"] += 1
        _emit(run, server)
        return _packet(run, server)


def install_control_routes():
    from aiohttp import web
    server = _server()
    if server is None or getattr(server, "_design61_control_installed", False):
        return

    @server.routes.get("/design61/external-sequence/control/{token}")
    async def get_control(request):
        try:
            return web.json_response(status(request.match_info["token"]))
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=409)

    @server.routes.post("/design61/external-sequence/stop")
    async def stop_control(request):
        try:
            body = await request.json()
            token = body.get("token")
            if not isinstance(token, str):
                raise ValueError("Missing Full Video run token.")
            return web.json_response(await asyncio.to_thread(request_stop, token))
        except (ValueError, TypeError, AttributeError) as error:
            return web.json_response({"error": str(error)}, status=409)

    server._design61_control_installed = True
