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
| Continue all remaining chunks | 保留已接受的全部片段，自动生成剩余所有段，不再逐段暂停；无需修改 Start 模式 |
| Finish here | 以截至当前段的已接受片段结束，不再生成后面的段 |
| Retry this chunk | 重做当前段，保留历史 Take |
| Start again from Chunk 1 | 从第一段重来，保留已有 Takes |
| Restart from chunk + Regenerate from selected chunk | 下拉选择当前已接受的段，从该段重新生成并暂停 Review；更早的段保留，旧的该段及后续段退出当前序列，历史 Takes 保留 |

Review 时修改上游提示词后，使用 End 的按钮继续操作，按钮会指向当前已接受的序列，并重新解析修改后的分块文本。例如在第三段 Review 时，重试保留第一、二段并用新提示词生成第三段；继续保留前三段并读取新的第四段提示词。已接受的片段和历史 Takes 不会因提示词修改而被覆盖。

点击 ComfyUI 蓝色运行按钮会从第一段开始新一轮；End 的「Start again from Chunk 1」也会明确重来，并使用新的 seed nonce。按钮指令只用于本次提交，不会留给后续蓝色运行。此机制不自动调整已有 State 的分辨率或布局。

例如计划十段、已完成五段：点击 `Continue all remaining chunks` 会从第六段自动跑到第十段；下拉选第三段并执行重做，会以第二段的已接受 State 为上下文，使用当前提示词生成新第三段。成功后当前序列只包含第 1、2、新 3 段，再点继续会重新生成第四段。新第三段失败时，原有五段的已接受链仍保留；成功提交后才切换到新链。下拉框随后只列出当前链中的第 1～3 段。

计划总时长是段数 × 每段秒数；实际输出时长还会受到原生 H3 帧网格和上下文拼接影响。已有 Review 操作以当前后端 revision 为准，过期按钮不会覆盖新提交的 Take。
