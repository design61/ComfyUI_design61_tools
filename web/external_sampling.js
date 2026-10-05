import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { refreshV39ReferenceImagesForSampler } from "./upstream_reference_images.js";

const END = "H3ContinuumExternalSequenceEnd_design61";
const START = "H3ContinuumExternalSequenceStart_design61";
const CONDITIONING = "H3ContinuumExternalConditioning_design61";
const REVIEW = "Review Each Chunk";
const RESUME = "Start / Resume";
const CONTINUE = "Use it and continue";
const RETRY = "Try this chunk again";
const FINISH = "Finish here";
const RESTART = "Start again from Chunk 1";
const CONTINUE_ALL = "Continue all remaining chunks";
const REGENERATE_FROM = "Regenerate from selected chunk";
const FRAME_STORAGE = "Frames + tail State (disk)";
const widget = (node, name) => node.widgets?.find((item) => item.name === name);

export function findExternalSequenceStart(node) {
    // Saved graphs may place legacy Reroute nodes on the flow wire. Trace that
    // wire, never choose an unrelated Start by scanning the entire canvas.
    const visited = new Set();
    let current = node;
    while (current && !visited.has(current)) {
        visited.add(current);
        if ((current.comfyClass || current.type) === START) return current;
        const inputs = current.inputs || [];
        let slot = inputs.findIndex((input) => input.name === "flow" || input.name === "sequence_flow");
        if (slot < 0) {
            const linked = inputs.map((input, index) => ({ input, index })).filter(({ input }) => input.link != null);
            if (linked.length !== 1) return null;
            slot = linked[0].index;
        }
        const input = inputs[slot];
        const edge = current.graph?.links?.[input?.link];
        current = current.getInputNode?.(slot) || current.graph?.getNodeById?.(edge?.origin_id);
    }
    return null;
}

function reviewError(node, message) {
    const status = widget(node, "Review status");
    if (status) status.value = message;
    node.setDirtyCanvas?.(true, true);
    console.error("[design61 External Review]", message);
}

export function externalReviewActions(state) {
    if (state?.mode === "Full Video") return [];
    if (state?.status === "review_ready") return [CONTINUE, CONTINUE_ALL, ...(state.stopped && !state.accepted ? [] : [FINISH]), RETRY, RESTART];
    if (state?.status === "complete" && state.review_unit) return [RETRY, RESTART];
    return [];
}

export function externalReviewReference(state, restartChunk) {
    return state.contract || restartChunk != null ? JSON.stringify({ revision: state.revision, contract: state.contract, run_name: state.run_name, restart_chunk: restartChunk, ...(state.storage_mode ? {storage_mode: state.storage_mode} : {}), ...(state.storage_mode === FRAME_STORAGE ? {frame_root: state.frame_root} : {}) }) : state.revision;
}

export function bindExternalConditioningFlow(node) {
    if (node?.comfyClass !== CONDITIONING && node?.type !== CONDITIONING) return false;
    const slot = node.inputs?.findIndex((item) => item.name === "sequence_flow");
    if (slot == null || slot < 0 || node.inputs[slot].link != null) return false;
    const sources = new Map();
    for (const output of node.outputs || []) for (const id of output.links || []) {
        const edge = node.graph?.links?.[id];
        const prepare = node.graph?.getNodeById?.(edge?.target_id);
        if ((prepare?.comfyClass || prepare?.type) !== "H3ContinuumExternalPrepare_design61") continue;
        const input = prepare.inputs?.find((item) => item.name === "sequence_flow");
        const flow = node.graph?.links?.[input?.link];
        const start = node.graph?.getNodeById?.(flow?.origin_id);
        if ((start?.comfyClass || start?.type) === START) sources.set(start.id, start);
    }
    if (sources.size !== 1) return false;
    const start = [...sources.values()][0];
    start.connect(0, node, slot);
    return true;
}

function hidden(node, name, value) {
    const item = widget(node, name);
    if (!item) return;
    item._externalOriginal ||= { type: item.type, computeSize: item.computeSize, hidden: item.hidden };
    item.hidden = value ? true : item._externalOriginal.hidden;
    item.type = value ? "hidden" : item._externalOriginal.type;
    item.computeSize = value ? () => [0, -4] : item._externalOriginal.computeSize;
}

function statusPanel(node, state, preview = false) {
    node._externalProgress = state;
    if (typeof document === "undefined" || typeof node.addDOMWidget !== "function") return;
    if (!node._externalStatusHost) {
        const host = document.createElement("div");
        host.style.cssText = "box-sizing:border-box;width:100%;padding:6px 4px;pointer-events:auto";
        const item = node.addDOMWidget("sequence_status", "h3_external_status", host, { serialize: false, tooltip: "Planned duration and canonical backend chunk progress. Finish here omits remaining chunks." });
        item.serialize = false;
        item.options ||= {}; item.options.serialize = false;
        item.computeSize = () => [300, 176];
        node._externalStatusHost = host;
        node.setSize?.(node.computeSize());
    }
    const host = node._externalStatusHost;
    const create = (tag, text = "", css = "") => {
        const item = document.createElement(tag); item.textContent = text; item.style.cssText = css; return item;
    };
    host.replaceChildren();
    const running = state.status === "running" || state.status === "in_progress";
    const color = state.status === "review_ready" ? "#e6b45d" : state.status === "complete" ? "#8cd5aa" : "#98cce5";
    const panel = create("div", "", "box-sizing:border-box;padding:12px;border:1px solid #496353;border-radius:10px;background:#1d2c24;color:#e6efe9;font:12px system-ui,sans-serif");
    panel.title = "Planned duration uses Chunks × Seconds. Progress comes from backend execution; remaining includes an executing chunk. Finish here cancels ungenerated chunks.";
    const top = create("div", "", "display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:10px");
    top.append(create("span", "TOTAL LENGTH", "font-size:10px;letter-spacing:1px;color:#a6b9ae"));
    top.append(create("strong", Number.isFinite(state.total_seconds) ? `${Number(state.total_seconds.toFixed(2))} s` : "Queue to resolve", "font-size:18px;color:#eff8f1"));
    panel.append(top);
    const grid = create("div", "", "display:grid;grid-template-columns:repeat(4,1fr);gap:5px");
    for (const [label, value] of [["TOTAL",state.chunks],["DONE",state.completed],["CURRENT",state.current],["LEFT",state.remaining]]) {
        const cell = create("div", "", "padding:6px 2px;text-align:center;background:#293e31;border-radius:6px");
        cell.append(create("div", value == null ? "—" : String(value), "font-size:17px;font-weight:650;color:"+color));
        cell.append(create("div", label, "margin-top:2px;font-size:9px;letter-spacing:.6px;color:#a6b9ae"));
        grid.append(cell);
    }
    panel.append(grid);
    const bar = create("div", "", "height:5px;background:#344c3d;border-radius:4px;margin:9px 0 7px;overflow:hidden");
    const ratio = state.chunks ? Math.min(1, (state.completed || 0) / state.chunks) : 0;
    bar.append(create("div", "", `height:100%;width:${ratio*100}%;background:${color};border-radius:4px`)); panel.append(bar);
    const label = preview ? "Ready · choose a mode and Queue" : state.stopped ? `Stopped at chunk ${state.current ?? state.completed} · kept ${state.completed}` : state.status === "complete" ? `Complete · ${state.completed_seconds ?? ""} s${state.skipped ? ` · ${state.skipped} chunks omitted` : ""}` : state.status === "review_ready" ? `Chunk ${state.completed} ready · choose an action below` : running ? `Running chunk ${state.current ?? "—"} / ${state.chunks}` : "Waiting for backend status";
    panel.append(create("div", label + (state.storage_mode === FRAME_STORAGE ? " · Frames on disk" : ""), "font-size:11px;color:"+color));
    host.append(panel);
    node.setDirtyCanvas?.(true, true);
}

function refreshStart(node) {
    for (const name of ["review_action", "expected_revision"]) hidden(node, name, true);
    const mode = widget(node, "generation_mode");
    const storage = widget(node, "storage_mode")?.value;
    hidden(node, "frame_root", storage !== FRAME_STORAGE);
    if (mode?.value === "Review by Chunk") mode.value = REVIEW;
    if (widget(node, "review_action")?.value === "Use it and finish the rest") widget(node, "review_action").value = FINISH;
    const linked = (name) => node.inputs?.some((item) => item.name === name && item.link != null);
    const count = linked("chunks") ? NaN : Number(widget(node, "chunks")?.value);
    const seconds = linked("chunk_seconds") ? NaN : Number(widget(node, "chunk_seconds")?.value);
    const key = JSON.stringify([count, seconds, mode?.value, storage]);
    if (node._externalPreviewKey !== key) {
        node._externalPreviewKey = key;
        statusPanel(node, { chunks: Number.isFinite(count) ? count : null, completed: 0, current: 1, remaining: Number.isFinite(count) ? count : null, total_seconds: count*seconds, mode: mode?.value, storage_mode: storage }, true);
        const pending = [node];
        const visited = new Set();
        while (pending.length) {
            const source = pending.pop();
            if (!source || visited.has(source)) continue;
            visited.add(source);
            for (const output of source.outputs || []) for (const id of output.links || []) {
                const edge = source.graph?.links?.[id];
                const target = source.graph?.getNodeById?.(edge?.target_id);
                if ((target?.comfyClass || target?.type) === END) {
                    if (target._externalReviewState && findExternalSequenceStart(target) === node) updatePanel(target, target._externalReviewState, true);
                } else if (target?.inputs?.length === 1 && target.inputs[0].link === id) {
                    pending.push(target); // Single-input reroute/pass-through.
                }
            }
        }
    }
}

function updatePanel(node, state, presentationOnly = false) {
    if (!state || !["review_ready", "complete", "in_progress"].includes(state.status)) return;
    const control = node._externalControl;
    if (control?.stop_requested && (control.running || !control.ready || (state.control_token === control.token && !state.stopped))) return;
    if (control?.stop_requested && !state.control_token && !state.stopped) node._externalControl = null;
    node._externalReviewState = state;
    node.widgets = (node.widgets || []).filter((widget) => !widget._externalReview);
    const add = (type, name, value, callback, tooltip, extra = {}) => {
        const widget = node.addWidget(type, name, value, callback, { ...extra, serialize: false, tooltip });
        widget._externalReview = true;
        widget.serialize = false;
        widget.options.serialize = false;
        return widget;
    };
    const linkedStart = findExternalSequenceStart(node);
    const framesOnDisk = state.storage_mode === FRAME_STORAGE;
    const historyHelp = framesOnDisk ? "Frame-mode history records. Superseded media/tails are deleted after replacement commits; earlier active chunks remain." : "Read-only immutable Take history for this external graph lineage. This panel does not select or overwrite a Take.";
    const find = (name) => linkedStart?.widgets?.find((widget) => widget.name === name);
    const action = find("review_action");
    const revision = find("expected_revision");
    if (action) action.value = "Start / Resume";
    if (revision) revision.value = ""; // Only an explicit button supplies a Review reference.
    if (state.progress && !presentationOnly) {
        statusPanel(node, state.progress);
        if (linkedStart) statusPanel(linkedStart, state.progress);
    }
    const label = state.stopped && state.review_unit?.pending ? `Stopped during Chunk ${state.review_unit.chunk} · kept ${state.accepted} chunks` : state.status === "complete" ? "Saved sequence is complete" : state.status === "review_ready" ? `Chunk ${state.accepted} is ready for review` : `Generating ${state.accepted}/${state.chunks} chunks`;
    add("text", "Review status", label, () => {}, "Canonical saved backend status. Review controls are never inferred from visible widget values.");
    if (control?.running && !control.complete) {
        const button = add("button", control.stop_requested ? "Stopping…" : "Stop", null, async () => {
            if (node._externalControl?.stop_requested) return;
            const token = node._externalControl.token;
            node._externalControl = { ...node._externalControl, stop_requested: true };
            button.name = "Stopping…";
            widget(node, "Review status").value = "Stopping this run · preserving completed chunks…";
            node.setDirtyCanvas?.(true, true);
            try {
                const response = await api.fetchApi("/design61/external-sequence/stop", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ token }) });
                const packet = await response.json();
                if (!response.ok) throw Error(packet.error || "Stop request failed");
                applyControlPacket(packet, token);
            } catch (error) {
                reviewError(node, `Stop: ${error?.message || error}`);
                node._externalControl = { ...node._externalControl, stop_requested: false };
                button.name = "Stop";
            }
        }, "Interrupt only this Full Video run. Preserve committed chunks and continuation tails, discard the unfinished chunk, then switch to Review. Wait for Core to finish interrupting before Retry/Continue.");
        node.setSize(node.computeSize());
        node.setDirtyCanvas(true, true);
        return;
    }
    const selectedMode = presentationOnly ? find("generation_mode")?.value || state.mode : state.mode || find("generation_mode")?.value;
    const queueAction = async (label, restartChunk) => {
            // Resolve again at click time: connections/widgets can be replaced
            // after the backend Review panel was drawn.
            const controller = findExternalSequenceStart(node);
            const command = controller && widget(controller, "review_action");
            const expected = controller && widget(controller, "expected_revision");
            if (!command || !expected) {
                reviewError(node, "Cannot find Sequence Start controls through flow. Check the flow connection.");
                return;
            }
            command.value = label;
            const reference = externalReviewReference(state, restartChunk);
            expected.value = reference;
            const previousControl = node._externalControl;
            if (previousControl) {
                clearTimeout(controlTimers.get(previousControl.token));
                controlTimers.delete(previousControl.token);
                node._externalControl = null;
            }
            node.setDirtyCanvas?.(true, true);
            try {
                await app.queuePrompt(0, 1);
            } catch (error) {
                node._externalControl ||= previousControl;
                reviewError(node, `Could not queue ${label}: ${error?.message || error}`);
            } finally {
                // Do not leave a Review command armed for the blue Queue button.
                // Avoid clearing a newer command submitted by another click.
                if (command.value === label && expected.value === reference) {
                    command.value = RESUME;
                    expected.value = "";
                }
            }
    };
    for (const label of externalReviewActions({ ...state, mode: selectedMode })) {
        add("button", label, null, () => queueAction(label), label === CONTINUE_ALL ? "Keep all accepted chunks and automatically generate every remaining chunk without Review pauses. Uses updated upstream prompts. Normal blue Queue still starts fresh." : label === FINISH ? "Finalize only accepted chunks now. No remaining chunk is sampled." : label === RESTART ? (framesOnDisk ? "Generate again from chunk 1. After the new first chunk commits, previous frame-mode media/tails are removed." : "Generate again from chunk 1 with a new seed nonce; all old Takes remain stored.") : label === RETRY ? (framesOnDisk ? "Regenerate this chunk, preserving earlier chunks. Replace its old frames/tail only after the new chunk commits." : "Generate a new Take for this chunk, preserving the accepted prefix and old Take.") : "Accept this chunk and execute exactly the next chunk. Its index and media window advance automatically.");
    }
    const chunks = state.accepted_chunks || [];
    if (selectedMode !== "Full Video" && ["review_ready", "complete"].includes(state.status) && chunks.length) {
        const selected = chunks.includes(node._externalRestartChunk) ? node._externalRestartChunk : chunks.at(-1);
        node._externalRestartChunk = selected;
        const choice = add("combo", "Restart from chunk", `Chunk ${selected}`, value => {
            node._externalRestartChunk = Number(String(value).replace("Chunk ", ""));
        }, framesOnDisk ? "Choose an accepted chunk. Keep earlier chunks; after replacement succeeds, delete the old selected/later chunk frames and tails." : "Choose an already accepted chunk. Regeneration keeps only its earlier prefix in the active sequence; old later Takes remain in History.", { values: chunks.map(number => `Chunk ${number}`) });
        add("button", REGENERATE_FROM, null, () => {
            const target = Number(String(choice.value).replace("Chunk ", ""));
            return queueAction(REGENERATE_FROM, target);
        }, "Regenerate the selected chunk using current prompts, then pause for Review. Later old chunks leave the active sequence when this Take succeeds; continue to regenerate following chunks. " + historyHelp);
    }
    if (state.history?.length && selectedMode !== "Full Video") {
        add("button", `Render History — ${state.history.length} Takes`, null, () => {
            const existing = node.widgets.find((widget) => widget._externalHistory);
            if (existing) node.widgets = node.widgets.filter((widget) => widget !== existing);
            else {
                const history = add("text", "Saved Takes", state.history.map((item) => `Chunk ${item.chunk}: ${item.revision.slice(0, 8)} / seed ${item.seed}${item.discarded ? " / media removed" : ""}`).join(" | "), () => {}, historyHelp);
                history._externalHistory = true;
            }
            node.setSize(node.computeSize());
            node.setDirtyCanvas(true, true);
        }, historyHelp);
    }
    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}

const controlTimers = new Map();
export function applyControlPacket(packet, expectedToken = null) {
    const graph = app.graph;
    if (!packet?.token || !graph) return;
    for (const id of packet.end_ids || []) {
        const node = graph.getNodeById?.(id);
        const start = node && findExternalSequenceStart(node);
        if (!node || (node.comfyClass || node.type) !== END || String(start?.id) !== packet.start_id) continue;
        // A late poll/event from a stopped run cannot replace a newer run.
        const old = node._externalControl;
        if (expectedToken && old?.token !== expectedToken) continue;
        if (packet.epoch && packet.epoch === old?.epoch && packet.order < old.order) continue;
        if (old?.token === packet.token && packet.serial < old.serial) continue;
        if (old?.token !== packet.token && old?.running && packet.stop_requested) continue;
        node._externalControl = packet;
        if (packet.stop_requested) {
            if (packet.running || !packet.ready) {
                node.widgets = (node.widgets || []).filter(item => !item._externalReview);
                const item = node.addWidget("button", "Stopping…", null, () => {}, {serialize:false,tooltip:"Core is interrupting this run. Completed chunks remain saved. Review actions become available after the old prompt exits."});
                item._externalReview = true; item.serialize = false;
                node.setSize?.(node.computeSize()); node.setDirtyCanvas?.(true, true);
                continue;
            }
            const mode = widget(start, "generation_mode");
            if (mode) mode.value = REVIEW;
            refreshStart(start);
        }
        updatePanel(node, packet.state);
    }
    clearTimeout(controlTimers.get(packet.token));
    controlTimers.delete(packet.token);
    if (packet.running || (packet.stop_requested && !packet.ready)) {
        controlTimers.set(packet.token, setTimeout(async () => {
            controlTimers.delete(packet.token);
            try {
                const response = await api.fetchApi(`/design61/external-sequence/control/${packet.token}`);
                const next = await response.json();
                if (!response.ok) throw Error(next.error || "Could not read run status");
                applyControlPacket(next, packet.token);
            } catch (error) {
                for (const id of packet.end_ids || []) {
                    const node = graph.getNodeById?.(id);
                    if (node?._externalControl?.token === packet.token) reviewError(node, `Run status: ${error?.message || error}`);
                }
            }
        }, 1000));
    }
}

app.registerExtension({
    name: "design61.ExternalSamplingReview",
    setup() {
        api.addEventListener("design61.external_control", event => applyControlPacket(event.detail));
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === CONDITIONING) {
            for (const hook of ["onNodeCreated", "onConfigure", "onConnectionsChange", "onDrawForeground"]) {
                const previous = nodeType.prototype[hook];
                nodeType.prototype[hook] = function () {
                    const result = previous?.apply(this, arguments);
                    if (!this._externalBindingPending) {
                        this._externalBindingPending = true;
                        queueMicrotask(() => {
                            this._externalBindingPending = false;
                            bindExternalConditioningFlow(this);
                            const automatic = this.inputs?.some((item) => item.name === "sequence_flow" && item.link != null);
                            for (const name of ["chunks", "chunk_index", "chunk_seconds"]) hidden(this, name, automatic);
                        });
                    }
                    refreshV39ReferenceImagesForSampler(this);
                    return result;
                };
            }
            const previousExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                previousExecuted?.apply(this, arguments);
                const state = message?.external_conditioning?.[0];
                if (!state) return;
                let display = this.widgets?.find((item) => item.name === "Current Chunk");
                if (!display) display = this.addWidget("text", "Current Chunk", "", () => {}, {serialize:false,tooltip:"Actual backend chunk used for this Conditioning. Automatic mode derives it from the accepted prefix; manual reserves are hidden."});
                display.serialize = false;
                display.options.serialize = false;
                display.value = `${state.current} / ${state.chunks}${state.automatic ? " · automatic" : " · manual"}`;
                this.setSize?.(this.computeSize());
            };
            return;
        }
        if (nodeData.name === START) {
            for (const hook of ["onNodeCreated", "onConfigure", "onDrawForeground"]) {
                const previous = nodeType.prototype[hook];
                nodeType.prototype[hook] = function () {
                    const result = previous?.apply(this, arguments);
                    if (hook !== "onDrawForeground") {
                        if (widget(this, "review_action")) widget(this, "review_action").value = RESUME;
                        if (widget(this, "expected_revision")) widget(this, "expected_revision").value = "";
                    }
                    refreshStart(this);
                    return result;
                };
            }
            const executed = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                executed?.apply(this, arguments);
                const progress = message?.external_progress?.[0];
                const old = this._externalProgress;
                if (progress && !(progress.control_token && progress.control_token === old?.control_token && progress.completed < old.completed)) statusPanel(this, progress);
                if (widget(this, "review_action")) widget(this, "review_action").value = RESUME;
                if (widget(this, "expected_revision")) widget(this, "expected_revision").value = "";
            };
            return;
        }
        if (nodeData.name !== END) return;
        const previous = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            previous?.apply(this, arguments);
            const state = message?.external_review?.[0];
            if (state?.control_token && this._externalControl && state.control_token !== this._externalControl.token) return;
            if (state?.mode === "Full Video" && this._externalControl?.running && state.accepted < this._externalControl.state?.accepted) return;
            updatePanel(this, state);
        };
    },
});
