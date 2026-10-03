# Changelog

## 0.1.0 — 2026-10-03

- Initial independent design61 tools package with six public nodes: four modified H3 external sequence/conditioning/prepare/end nodes and two personal folder-frame tools.
- External sampling and learned latent upscale remain under the user's workflow control.
- Automatic multi-chunk sequence and per-chunk Review support Continue, Finish here, Retry and Restart while retaining previous Takes.
- Original H3 image/audio bundles are adapted without duplicating upstream helper registrations; driving audio and timeline video inputs are retained.
- Added live planned duration and sequence progress display.
- Included a 36-second introduction animation and reusable offline HTML/Canvas source.
- Validation: 30 CPU unit tests; synthetic Core execution and native layout/helper/finalize checks; real CPU FFmpeg encoding. GPU diffusion and browser workflow acceptance remain unverified.

No model weights, workflows, FFmpeg binaries or ComfyUI Registry publication are included.
