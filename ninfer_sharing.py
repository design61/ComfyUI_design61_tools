"""Persistent, opt-in NInfer GPU handoff before ComfyUI prompt submission."""
import asyncio
import http.client
import json
import logging
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit

import folder_paths
from aiohttp import web


def settings_path():
    return Path(folder_paths.get_user_directory()) / "design61_tools" / "ninfer-sharing.json"


def load_settings():
    path = settings_path()
    if not path.exists():
        return {"enabled": False, "bridge_path": ""}
    return json.loads(path.read_text(encoding="utf-8"))


def save_settings(values):
    if not isinstance(values, dict) or type(values.get("enabled")) is not bool or not isinstance(values.get("bridge_path"), str):
        raise ValueError("enabled must be boolean and bridge_path must be text")
    config = {"enabled": values["enabled"], "bridge_path": values["bridge_path"].strip()}
    if config["enabled"] and not config["bridge_path"]:
        raise ValueError("启用交接时请填写 agent-sharing-bridge.json 的完整路径")
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".ninfer-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return config


def control(config, action, body=None):
    url = urlsplit(config["url"])
    if url.hostname not in ("127.0.0.1", "localhost", "::1") or url.scheme != "http":
        raise ValueError("NInfer sharing only supports local HTTP")
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=20)
    try:
        connection.request("POST", "/_agent/" + action, json.dumps(body or {}),
                           {"Content-Type": "application/json", "X-Agent-Token": config["token"]})
        response = connection.getresponse()
        data = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(f"NInfer handoff failed: HTTP {response.status}")
        return data
    finally:
        connection.close()


@web.middleware
async def release_before_prompt(request, handler):
    if request.method != "POST" or request.path not in ("/prompt", "/api/prompt"):
        return await handler(request)
    try:
        settings = load_settings()
        config = None
        if settings.get("enabled") and Path(settings["bridge_path"]).is_file():
            # Secrets stay in the local bridge file, never in node widgets/API replies.
            config = json.loads(Path(settings["bridge_path"]).read_text(encoding="utf-8"))
            result = await asyncio.to_thread(control, config, "yield")
    except ConnectionRefusedError:
        return await handler(request)
    except Exception as error:
        logging.error("[design61 NInfer] GPU handoff failed: %s", error)
        return web.json_response({"error": {"type": "ninfer_handoff_failed",
            "message": "NInfer 显存交接失败，请先停止 start-agent 再提交任务。",
            "details": str(error), "extra_info": {}}, "node_errors": {}}, status=503)
    if config is None:
        return await handler(request)
    try:
        return await handler(request)
    finally:
        try:
            await asyncio.to_thread(control, config, "settle", {"lease": result["lease"]})
        except Exception as error:
            logging.warning("[design61 NInfer] Submission lease retained until timeout: %s", error)


def install_hooks():
    from server import PromptServer
    instance = getattr(PromptServer, "instance", None)
    if instance is None or getattr(instance, "_design61_ninfer_installed", False):
        return

    @instance.routes.get("/design61/ninfer-sharing/settings")
    async def get_settings(request):
        return web.json_response(load_settings())

    @instance.routes.post("/design61/ninfer-sharing/settings")
    async def put_settings(request):
        try:
            return web.json_response(save_settings(await request.json()))
        except (ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)

    @instance.routes.get("/design61/ninfer-sharing/status")
    async def status(request):
        values = load_settings()
        return web.json_response({"installed": True, "gate": "before_prompt_queue",
            "enabled": bool(values.get("enabled")),
            "supervisor_active": bool(values.get("bridge_path")) and Path(values["bridge_path"]).is_file()})

    # Preserve the standalone hook's read-only probe for existing NInfer tools.
    # Its active directory must be disabled before this alias can be installed.
    if not any((Path(root) / "ninfer_agent_sharing").is_dir()
               for root in folder_paths.get_folder_paths("custom_nodes")):
        instance.routes.get("/ninfer-sharing")(status)

    instance.app.middlewares.append(release_before_prompt)
    instance._design61_ninfer_installed = True


class NInferAgentSharingSettings_design61:
    DESCRIPTION = "设置 NInfer 与 ComfyUI 的显存交接。点击节点内保存按钮后全局生效；删除节点仍保留设置。无需连接或运行此节点。令牌从本地 bridge 文件读取，不保存到工作流。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "enabled": ("BOOLEAN", {"default": False, "tooltip": "保存后启用任务入队前的 NInfer 显存交接。关闭后不请求 NInfer。"}),
            "bridge_path": ("STRING", {"default": "", "tooltip": "NInfer 的 agent-sharing-bridge.json 完整路径。配置保存到 ComfyUI 用户目录，删除此节点仍有效。"}),
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_status",)
    FUNCTION = "status"
    CATEGORY = "design61/NInfer"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def status(self, **kwargs):
        settings = load_settings()
        return ("NInfer sharing: " + ("enabled" if settings.get("enabled") else "disabled"),)
