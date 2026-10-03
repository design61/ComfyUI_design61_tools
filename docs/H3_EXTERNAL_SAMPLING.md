# H3 自定义采样：进阶连接说明

先通过[单采工作流案例](../examples/MiniMax_H3_single_sampler_design61.json)了解基本连接。长视频续写原理请阅读[原版 ComfyUI-H3-Continuum](https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum)。这里仅记录拆分节点的连接契约。

## H3 连接要点

单采案例由 design61 提供；双采与 latent 二采放大需要自行配置外部采样链。

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
