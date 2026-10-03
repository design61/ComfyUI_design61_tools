# MiniMax H3 自定义单采长视频案例

[下载工作流 JSON](MiniMax_H3_single_sampler_design61.json)，在 ComfyUI 中打开。

由 design61 提供。每个 chunk 使用一次外部采样，Sequence Start / End 管理分段续写和 Review。保留用户原有的接线、提示词、参数和素材文件名；公开副本仅恢复 Core 节点的标准显示名称，原始文件不变。

## 需要准备

- 本工具包的四个 H3 拆分节点。
- 原版 [ComfyUI-H3-Continuum](https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum) 的素材与拼接辅助节点。
- 案例包含 `H3ContinuumAssembleSeamV35`、`H3DecodeCacheHelper` 和 `H3ContinuumReferenceImagesV39`。不同原版版本的公开节点可能不同；如果版本不提供相应 ID，请使用匹配版本或按其官方说明连接对应 Finalize / Decode 节点。不要把未知节点当成本工具包漏装。
- 含 `comfy_extras.nodes_lora_debug` 的 ComfyUI（提供 `LoraLoaderBypassModelOnly`），以及提供 `VHS_VideoCombine` 的 [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)。
- 支持 MiniMax H3 的 ComfyUI、H3 模型、CLIP、音视频 VAE 和所选 LoRA。按自己的环境重新选择模型和输入图片；这些文件没有附带。

案例当前保存为两段、每段 5 秒、Review Each Chunk，外部 Euler / simple 调度器共 8 步。每段采样完成后，在 End 接受并继续或选择其他操作。Review 按钮会沿 Reroute 找到关联的 Start，不需要拆掉转接节点。

Start 的 `seed` 可接到外部 RandomNoise，使用控制器的分段种子；案例当前保留用户已有的独立固定噪声种子设置。不要在同一轮 Review 中自动随机修改采样参数，否则执行图的身份会改变。

这个案例说明单采接线，并不构成无限时长 GPU 验证。实际可生成的总长度受到所设段数、模型行为、磁盘空间和运行环境影响。
