import { app } from "../../scripts/app.js";
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
const widget = (node, name) => node.widgets?.find((item) => item.name === name);

export function externalReviewActions(state) {
    if (state?.mode === "Full Video") return [];
    if (state?.status === "review_ready") return [CONTINUE, FINISH, RETRY, RESTART];
    if (state?.status === "complete" && state.review_unit) return [RETRY, RESTART];
    return [];
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
        item.computeSize = () => [300, 152];
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
    const label = preview ? "Ready · choose a mode and Queue" : state.status === "complete" ? `Complete · ${state.completed_seconds ?? ""} s${state.skipped ? ` · ${state.skipped} chunks omitted` : ""}` : state.status === "review_ready" ? `Chunk ${state.completed} ready · choose an action below` : running ? `Running chunk ${state.current ?? "—"} / ${state.chunks}` : "Waiting for backend status";
    panel.append(create("div", label, "font-size:11px;color:"+color));
    host.append(panel);
    node.setDirtyCanvas?.(true, true);
}

function refreshStart(node) {
    for (const name of ["review_action", "expected_revision"]) hidden(node, name, true);
    const mode = widget(node, "generation_mode");
    if (mode?.value === "Review by Chunk") mode.value = REVIEW;
    if (widget(node, "review_action")?.value === "Use it and finish the rest") widget(node, "review_action").value = FINISH;
    const linked = (name) => node.inputs?.some((item) => item.name === name && item.link != null);
    const count = linked("chunks") ? NaN : Number(widget(node, "chunks")?.value);
    const seconds = linked("chunk_seconds") ? NaN : Number(widget(node, "chunk_seconds")?.value);
    const key = JSON.stringify([count, seconds, mode?.value]);
    if (node._externalPreviewKey !== key) {
        node._externalPreviewKey = key;
        statusPanel(node, { chunks: Number.isFinite(count) ? count : null, completed: 0, current: 1, remaining: Number.isFinite(count) ? count : null, total_seconds: count*seconds, mode: mode?.value }, true);
        for (const output of node.outputs || []) for (const id of output.links || []) {
            const edge = node.graph?.links?.[id];
            const end = node.graph?.getNodeById?.(edge?.target_id);
            if ((end?.comfyClass || end?.type) === END && end._externalReviewState) updatePanel(end, end._externalReviewState, true);
        }
    }
}

function updatePanel(node, state, presentationOnly = false) {
    if (!state || !["review_ready", "complete", "in_progress"].includes(state.status)) return;
    node._externalReviewState = state;
    node.widgets = (node.widgets || []).filter((widget) => !widget._externalReview);
    const add = (type, name, value, callback, tooltip) => {
        const widget = node.addWidget(type, name, value, callback, { serialize: false, tooltip });
        widget._externalReview = true;
        widget.serialize = false;
        widget.options.serialize = false;
        return widget;
    };
    const start = node.getInputNode?.(0);
    const linkedStart = start?.comfyClass === START || start?.type === START ? start : null;
    const find = (name) => linkedStart?.widgets?.find((widget) => widget.name === name);
    const action = find("review_action");
    const revision = find("expected_revision");
    if (action) action.value = "Start / Resume";
    if (revision) revision.value = state.revision || "";
    if (state.progress && !presentationOnly) {
        statusPanel(node, state.progress);
        if (linkedStart) statusPanel(linkedStart, state.progress);
    }
    const label = state.status === "complete" ? "Saved sequence is complete" : state.status === "review_ready" ? `Chunk ${state.accepted} is ready for review` : `Generating ${state.accepted}/${state.chunks} chunks`;
    add("text", "Review status", label, () => {}, "Canonical saved backend status. Review controls are never inferred from visible widget values.");
    const selectedMode = find("generation_mode")?.value || state.mode;
    for (const label of externalReviewActions({ ...state, mode: selectedMode })) {
        add("button", label, null, async () => {
            if (!action || !revision) return;
            action.value = label;
            revision.value = state.revision;
            await app.queuePrompt(0, 1);
        }, label === FINISH ? "Finalize only accepted chunks now. No remaining chunk is sampled." : label === RESTART ? "Generate again from chunk 1 with a new seed nonce; all old Takes remain stored." : label === RETRY ? "Generate a new Take for this chunk, preserving the accepted prefix and old Take." : "Accept this chunk and execute exactly the next chunk. Its index and media window advance automatically.");
    }
    if (state.history?.length && selectedMode !== "Full Video") {
        add("button", `Render History — ${state.history.length} Takes`, null, () => {
            const existing = node.widgets.find((widget) => widget._externalHistory);
            if (existing) node.widgets = node.widgets.filter((widget) => widget !== existing);
            else {
                const history = add("text", "Saved Takes", state.history.map((item) => `Chunk ${item.chunk}: ${item.revision.slice(0, 8)} / seed ${item.seed}`).join(" | "), () => {}, "Read-only immutable Take history for this external graph lineage. This panel does not select or overwrite a Take.");
                history._externalHistory = true;
            }
            node.setSize(node.computeSize());
            node.setDirtyCanvas(true, true);
        }, "Show or hide the saved external sequence's immutable Take history.");
    }
    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "design61.ExternalSamplingReview",
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
                    refreshStart(this);
                    return result;
                };
            }
            const executed = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                executed?.apply(this, arguments);
                if (message?.external_progress?.[0]) statusPanel(this, message.external_progress[0]);
                if (widget(this, "review_action")) widget(this, "review_action").value = RESUME;
            };
            return;
        }
        if (nodeData.name !== END) return;
        const previous = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            previous?.apply(this, arguments);
            updatePanel(this, message?.external_review?.[0]);
        };
    },
});
