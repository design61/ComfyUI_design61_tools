import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TYPE = "NInferAgentSharingSettings_design61";
const widget = (node, name) => node.widgets?.find((item) => item.name === name);

app.registerExtension({
    name: "design61.NInferSharingSettings",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TYPE) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            const node = this;
            const saved = node.addWidget("text", "保存状态", "尚未保存", () => {}, { serialize: false });
            node.addWidget("button", "保存参数（删除节点后仍生效）", null, async () => {
                try {
                    const response = await api.fetchApi("/design61/ninfer-sharing/settings", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ enabled: Boolean(widget(node, "enabled")?.value), bridge_path: widget(node, "bridge_path")?.value ?? "" }),
                    });
                    const data = await response.json();
                    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
                    saved.value = data.enabled ? "已保存 · 显存交接已启用" : "已保存 · 显存交接已关闭";
                } catch (error) {
                    saved.value = `保存失败：${error.message}`;
                }
                node.setDirtyCanvas(true, true);
            }, { serialize: false });
            // New settings nodes show the persisted config. Workflow loading
            // subsequently restores its widgets; only the Save button commits.
            api.fetchApi("/design61/ninfer-sharing/settings").then((response) => {
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                return response.json();
            }).then((data) => {
                if (!node.graph) return;
                widget(node, "enabled").value = data.enabled;
                widget(node, "bridge_path").value = data.bridge_path;
                saved.value = data.enabled ? "已保存 · 显存交接已启用" : "已保存 · 显存交接已关闭";
                node.setDirtyCanvas(true, true);
            }).catch(() => { saved.value = "读取配置失败，可修改后保存"; });
            node.setSize([410, Math.max(node.size[1], 165)]);
            return result;
        };
    },
});
