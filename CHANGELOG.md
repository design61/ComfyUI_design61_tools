# Changelog

## 0.1.1 — 2026-10-04

- Fixed Review buttons silently returning when the End flow wire passes through Reroute nodes. Resolve the connected Start at click time and preserve the canonical revision.
- Full/Review presentation now follows the same routed flow connection. Queue failures and missing controls are shown in Review status rather than ignored.
- Added multi-reroute, replaced-widget, history toggle, cycle and queue-failure regression checks, plus a read-only saved-workflow verifier.
- Added design61's single-sampler H3 long-video example. Public-copy Core display titles are standard; original user's file and all inputs/settings/wiring are preserved.
- Rewrote the introduction around two tool functions: folder frames to video by path, and custom sampling for long video with ComfyUI-H3-Continuum. Advanced sampling details are in a separate document.
- Updated the local HTML/Canvas introduction video to match this overview.
- Validation: 30 CPU tests passed; the supplied saved flow route was verified; all four Review buttons and History responded to native clicks in the live ComfyUI frontend, with every server mutation blocked and zero model prompts submitted.

The repair changes only frontend Review routing; H3 sampling, State/Session and stored Takes are unchanged.

## 0.1.0 — 2026-10-03

- Initial independent design61 tools package with six public nodes: four modified H3 external sequence/conditioning/prepare/end nodes and two personal folder-frame tools.
- External sampling and learned latent upscale remain under the user's workflow control.
- Automatic multi-chunk sequence and per-chunk Review support Continue, Finish here, Retry and Restart while retaining previous Takes.
- Original H3 image/audio bundles are adapted without duplicating upstream helper registrations; driving audio and timeline video inputs are retained.
- Added live planned duration and sequence progress display.
- Included a 36-second introduction animation and reusable offline HTML/Canvas source.
- Validation: 30 CPU unit tests; synthetic Core execution and native layout/helper/finalize checks; real CPU FFmpeg encoding. GPU diffusion and browser workflow acceptance remain unverified.

No model weights, workflows, FFmpeg binaries or ComfyUI Registry publication are included.
