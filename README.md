# ComfyUI_design61_tools

**让采样链自由，让视频续写有序。**

design61 的个人 ComfyUI 节点合集。将 H3 的分段续写控制与采样拆开，支持每段使用外部双采和中间 latent 放大；同时收纳文件夹序列帧转视频工具。以后自制功能继续在这个包里扩展。

[![介绍动画](docs/intro/preview.gif)](https://github.com/design61/ComfyUI_design61_tools/releases/download/v0.1.0/design61-intro.mp4)

[观看 36 秒介绍视频](https://github.com/design61/ComfyUI_design61_tools/releases/download/v0.1.0/design61-intro.mp4) · [下载插件 ZIP](https://github.com/design61/ComfyUI_design61_tools/releases/latest) · [动画源码](docs/intro)

> 视频是功能示意动画，由 HTML/Canvas 和本地 FFmpeg 制作；不是模型生成结果或 GPU 测试录屏。

## 为什么做这个包

想用 H3 连续生成多个 chunk，又想自己决定采样器、噪声分段和中间放大？这个包把 Conditioning、latent 准备与序列控制单独开放，采样全部交给画布上的外部节点。

例如，总共 8 步：第一采运行前 3 步、保留中间噪声 latent → 外部 **Upstream-default learned 3D latent upscale · 2x** → 第二采运行剩余 5 步。这里的 3 + 5 是用法示例；采样器、SIGMAS、噪声衔接和放大器需要在外部正确配置。

- **自动多段**：每段执行完整的外部双采链，随后自动进入下一段。
- **逐段 Review**：接受并继续、在当前段结束、重试当前段、从第一段重新开始。
- **保留素材输入**：首尾帧、原版按 chunk 选择的参考图、视频序列帧、驱动音频与其 VAE、参考音频合集。
- **进度清晰**：Start 实时显示计划总时长，以及总段数、已完成、当前段、剩余段数。
- **文件夹转视频**：序列帧通过 FFmpeg 编码；另有独立的文件夹序列帧清理节点。

## 六个公开节点

| 节点名称 | 用途 |
| --- | --- |
| H3 Continuum External Sequence Start_design61 | 设定段数、每段时长、种子和运行模式；提供当前段的 flow、prompt、seed 和前段 State |
| H3 Continuum External Conditioning_design61 | 按当前段生成 H3 Conditioning 和联合音视频模板 latent；两个分辨率阶段分别使用 |
| H3 Continuum External Prepare_design61 | 为各阶段准备 model、conditioning、latent 和 plan，接入前段续写上下文 |
| H3 Continuum External Sequence End_design61 | 接收第二采完成的 latent，保存当前 Take，驱动循环或 Review，输出拼接所需数据 |
| 文件夹序列帧直接FFmpeg视频_design61 | 将文件夹内的序列帧直接编码为视频 |
| 清空文件夹序列帧_design61 | 按原工具的规则清理指定文件夹的序列帧 |

所有公开节点 ID 和显示名都带 `_design61`。内部 Capture 和 `_h3` 模块只是运行依赖，不注册成额外节点。**本包没有复制注册原版 Sampler、Reference Images、Reference Audios、Video Adapter 或 Finalize。**

## 安装

在 ComfyUI 的 `custom_nodes` 目录执行：

```bash
git clone https://github.com/design61/ComfyUI_design61_tools.git
```

或下载 Release ZIP，将其中的 `ComfyUI_design61_tools` 文件夹解压到 `custom_nodes`，重启 ComfyUI 并刷新页面。

H3 功能需要支持 MiniMax H3 / 原生 PackedLayout 的 ComfyUI、对应的模型及音视频 VAE。使用原版素材辅助节点和拼接功能时，还需要安装 [ComfyUI-H3-Continuum](https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum)。采样器、learned 3D latent upscale 和其他外部节点需要自行安装；本包不包含模型、VAE 或 FFmpeg 二进制。

文件夹工具按「节点指定路径 → 插件的 ffmpeg 目录 → 系统 PATH」查找 FFmpeg。已经安装旧 `ComfyUI_FolderFramesToVideo_design61` 的用户，应禁用旧目录以免相同节点 ID 重复注册，并保留需要的配置和 FFmpeg 文件。

## H3 连接要点

这里仅说明节点契约，不附带或修改工作流。

1. Start 的 `flow` 连接两个阶段的 Conditioning / Prepare 和 End。连接 `sequence_flow` 后，当前 chunk 自动选择，不需要手动修改 `chunk_index`。
2. 低分辨率和高分辨率阶段各自使用 Conditioning / Prepare；素材在各阶段按对应尺寸编码。第二阶段的 `source_plan` 使用第一阶段的 plan。
3. 前 3 步的中间噪声 latent 经外部 learned upscale 后交给第二采；**End 接收第二采完全完成的联合 AV latent 和高分辨率 plan**。
4. 后续段使用已接受的高分辨率 State。低分辨率 Prepare 需显式启用参考上下文投影；投影不改变已保存 State 的分辨率。
5. 原版 Reference Images V3.9 / Reference Audios 的输出直接接入两个 Conditioning。按段素材选择通过原 bundle 保留。原版参考图表格的自动段数展示取决于其前端版本；本包后端使用 Start 的实际段数与当前段执行选择。
6. 驱动音频接入两个 Conditioning，并同时接到 End 的 `driving_audio`，供原版 Finalize 拼接时使用；提前结束时按已完成时长裁切。
7. End 输出的视频 / 音频 latent 列表及 assembly plan，交给原版 Finalize / 对应外部解码保存节点。

`Full Video` 自动跑完整个序列，End 不出现逐段操作按钮。`Review Each Chunk` 每完成一段暂停，End 提供以下操作：

| 操作 | 行为 |
| --- | --- |
| Use it and continue | 接受当前段，自动运行下一段 |
| Finish here | 以截至当前段的已接受片段结束，不再生成后面的段 |
| Retry this chunk | 重做当前段，保留历史 Take |
| Start again from Chunk 1 | 从第一段重来，保留已有 Takes |

计划总时长是段数 × 每段秒数；实际输出时长还会受到原生 H3 帧网格和上下文拼接影响。修改连接或设置会影响序列 lineage，已有 Review 操作应以当前后端 revision 为准。

## 验证范围

初版完成了 **30 项 CPU 单元测试**、真实 ComfyUI Core PromptExecutor 的合成 Full / Review / Continue / Retry / Finish / Restart 检查、安装目录六节点加载检查、原版参考图和音频 bundle 互通、原版 Finalize / PackedLayout 检查，以及 FFmpeg 四帧实际编码和清理检查。

这些检查没有加载生成模型或执行 GPU 扩散采样；实际 GPU 质量、浏览器交互及你的模型和外部采样器组合还需要实际工作流验证。暂未发布到 ComfyUI Registry。

## 开发与动画

新增个人功能放到独立模块，在根 `__init__.py` 合并映射，并为公开 ID / 显示名加 `_design61`。不要改变现有节点 ID。私有 H3 依赖仍需保留，不能因为没有公开节点就删去。

`docs/intro/index.html` 可直接在浏览器播放或拖动时间轴。动画使用确定性的 `renderAt(t)`，便于本地逐帧导出；渲染方法见 [动画说明](docs/intro/README.md)。

## 来源与许可

H3 部分基于 [ComfyUI-H3-Continuum](https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum) 的 MIT 代码二次开发，保留上游署名和 [原始许可证](LICENSE_H3_CONTINUUM)。这是 design61 的独立工具包，不是上游官方发布。文件夹工具由 design61 提供。

本项目使用 [MIT License](LICENSE)。详细来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

---

**English:** Personal ComfyUI tools by design61. Four modular H3 nodes expose conditioning, latent preparation and automatic sequence/review control for external two-stage sampling with an intermediate learned upscale. Two additional tools encode folder frames with FFmpeg and clear frame folders. Original H3 helpers remain in the upstream plugin. No workflows or model weights are bundled. Initial validation is CPU-only.
