# 介绍动画源码

双击 `index.html`，在浏览器里播放、暂停或拖动时间轴。页面没有外部字体、素材或服务请求，视频不使用模型生成接口。

画布尺寸 1920 × 1080，36 秒，导出 24 fps，无音轨。六个场景说明产品用途、3 + 5 外部双采、自动序列 / Review、素材输入、文件夹工具及安装地址。它是功能示意，不是实际 ComfyUI 操作录屏或生成效果。

## 本地导出

需要 Node.js、Playwright 的 Chromium 运行环境和 FFmpeg（带 libx264）。这些仅用于渲染视频，**不是插件的运行依赖**。在本目录：

```bash
npm install --no-save playwright
npx playwright install chromium
node render.cjs
```

FFmpeg 不在 PATH 时，用环境变量 `FFMPEG_PATH` 指定完整路径；希望使用已有 Chrome 时设置 `CHROME_PATH`。Playwright 已在别处安装时，可用 `PLAYWRIGHT_MODULE` 指定模块路径。`VIDEO_OUTPUT` 可改变 MP4 输出位置。

脚本调用 Canvas 的确定性 `renderAt(seconds)`，逐帧传给 FFmpeg，不创建大量临时图片，也不打开或操作你的日常浏览器窗口。MP4 使用 H.264 / yuv420p / faststart，默认不进入 Git；完整视频在 Release 附件中。

README 的轻量 GIF 可用 FFmpeg 从 MP4 提取：

```bash
ffmpeg -i design61-intro.mp4 -vf "fps=6,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" -loop 0 preview.gif
```

修改台词、颜色或场景后重新导出即可。全部画面由代码绘制，无额外图片版权来源。
