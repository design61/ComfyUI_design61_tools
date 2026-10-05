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
7. 默认 `Latents (existing)` 模式中，End 输出的视频 / 音频 latent 列表及 assembly plan，交给原版 Finalize / 对应外部解码保存节点。分段落盘模式使用下面的 FFmpeg 接线。

`Full Video` 自动跑完整个序列，End 不出现逐段操作按钮。`Review Each Chunk` 每完成一段暂停，End 提供以下操作：

| 操作 | 行为 |
| --- | --- |
| Use it and continue | 接受当前段，自动运行下一段 |
| Continue all remaining chunks | 保留已接受的全部片段，自动生成剩余所有段，不再逐段暂停；无需修改 Start 模式 |
| Finish here | 以截至当前段的已接受片段结束，不再生成后面的段 |
| Retry this chunk | 重做当前段 |
| Start again from Chunk 1 | 从第一段重来 |
| Restart from chunk + Regenerate from selected chunk | 下拉选择当前已接受的段，从该段重新生成并暂停 Review；更早的段保留，成功后旧的该段及后续段退出当前序列 |

Review 时修改上游提示词后，使用 End 的按钮继续操作，按钮会指向当前已接受的序列，并重新解析修改后的分块文本。例如在第三段 Review 时，重试保留第一、二段并用新提示词生成第三段；继续保留前三段并读取新的第四段提示词。单独修改文本不会删除已接受的片段。

点击 ComfyUI 蓝色运行按钮会从第一段开始新一轮；End 的「Start again from Chunk 1」也会明确重来，并使用新的 seed nonce。按钮指令只用于本次提交，不会留给后续蓝色运行。此机制不自动调整已有 State 的分辨率或布局。

例如计划十段、已完成五段：点击 `Continue all remaining chunks` 会从第六段自动跑到第十段；下拉选第三段并执行重做，会以第二段的已接受 State 为上下文，使用当前提示词生成新第三段。成功后当前序列只包含第 1、2、新 3 段，再点继续会重新生成第四段。新第三段失败时，原有五段的已接受链仍保留；成功提交后才切换到新链。下拉框随后只列出当前链中的第 1～3 段。

计划总时长是段数 × 每段秒数；实际输出时长还会受到原生 H3 帧网格和上下文拼接影响。已有 Review 操作以当前后端 revision 为准，过期按钮不会覆盖新提交的 Take。

## 分段存帧与 FFmpeg 合成

生成前在 Start 的 `storage_mode` 选择：

| 模式 | 保存与输出 |
| --- | --- |
| Latents (existing) | 原有完整 CPU latent Take，End 前五个输出保持兼容，使用 Decode／Finalize；历史 Takes 保留 |
| Frames + tail State (disk) | 每段解码一次，将去除上下文重叠的 PNG 帧、音频和尾部 State 存盘；End 输出帧目录及音频给 FFmpeg；替代段的旧媒体会清理 |

落盘模式接线：

| 来源 | 接到 |
| --- | --- |
| 原来的视频 VAE | End 的 `video_vae`，必须连接 |
| 音频 VAE（需要生成音频时） | End 的 `audio_vae` |
| 原始驱动音频（使用时） | End 的 `driving_audio`，优先采用原音频，无需 End 的音频 VAE |
| End 的 `frames_path` | 文件夹序列帧直接FFmpeg视频_design61 的 `frames_path`（右键转换为输入） |
| End 的 `audio` | 文件夹序列帧直接FFmpeg视频_design61 的 `audio` |

FFmpeg 节点设 `fps = 24`、`start_frame = 1`、`end_frame = 0`，不连接 `image`。生成期间建议 `delete_source_frames = false`。帧目录本身是输入依赖，不必额外连接触发器。落盘模式会阻断 End 的前三个旧解码输出，因此不能再通过它们生成视频；使用 FFmpeg 节点的输出预览。单采、双采和外部 latent 放大链保持原样。

Start 的 `frame_root` 留空时，目录位于 ComfyUI 的 `output/design61_sequences/<运行键>/`。可填绝对路径改用其他盘，或填相对路径放在 output 下。`takes/<版本>/frames` 是各段的有效帧；`tail.safetensors` 和同名 JSON 保存当前连续性设置所需的原生视频、音频尾部及网格元数据；默认 22 个画面帧对应 **7 个视频 latent 时间位置**，不是直接截取 22 个 latent。`views/<版本>` 是供 FFmpeg 读取的平面帧目录，优先使用硬链接，避免再复制整段图片。

`Full Video` 逐段存盘，完成后一次编码；`Review Each Chunk` 每段完成后输出截至该段的帧目录，可以编码预览。已知总时长用于裁剪原生网格多出的末尾帧，或补齐不足的最后一帧；内部尾部 State 使用真实采样网格，不受输出帧裁剪影响。

例如完成到第 5 段后重做第 3 段：先读取第 2 段的 State，生成、解码和保存新的第 3 段，全部成功并提交后才删除旧第 3～5 段的帧与尾部 State，保留第 1、2 段。失败或过期的并发操作不会覆盖原来的已接受链。历史面板保留操作记录，被清理的旧媒体不能恢复。原有 latent 模式的 Session／State 文件不受此清理影响。

开始新任务时选择存储模式；End 的 Review 按钮会保持当前任务原来的模式与目录，避免中途把一条序列切到不兼容的存储。若希望改用另一模式，用蓝色运行开始新任务。增大连续性上下文长度也需要从具有足够尾部的段重新生成；本模式不会补造或静默扩大 State。

此路径复用 Core VAE 解码，并按计划去除每段重叠帧，不执行原版 Finalize 的自动视频／音频接缝修正。需要那些修正时保留原有 latent 模式。未接驱动音频或音频 VAE 时，FFmpeg 输出无声视频；续写用的音频 latent 尾部仍保存。

内存主要用于当前段 VAE 解码、模型与少量音频，PNG 历史帧通过磁盘交给 FFmpeg，不会作为整段 IMAGE 输出积累。它减少历史解码图像的内存增长，不代表完全不使用内存。PNG 会占用磁盘；只清理明确不再需要的运行目录，正在生成或 Review 的任务不要清理 `takes`。
