"""One-physical-group Sampling Engine for the Continuum runtime."""

from __future__ import annotations

from typing import Any, Callable


def sample_physical_group(
    *,
    sample_callable: Callable[..., Any],
    model: Any,
    conditioning: Any,
    latent: Any,
    sampler: Any,
    sigmas: Any,
    seed: int,
    enable_preview: bool,
    physical_group: int,
    logical_chunks: tuple[int, ...],
    terminal_atomic: bool,
    layout_validation_profiler: Any = None,
    packed_row_planner: Any = None,
    planner_event: Any = None,
    diagnostic_policy: Any = None,
    context_frames: int = 0,
) -> tuple[Any, Any]:
    """Run exactly one physical sample with unchanged observer ordering."""

    normalized_chunks = tuple(int(value) for value in logical_chunks)
    layout_profile_token = (
        layout_validation_profiler.begin_sampling_group(
            physical_group=int(physical_group),
            logical_chunks=normalized_chunks,
            sampling_steps=max(0, int(sigmas.shape[-1]) - 1),
        )
        if layout_validation_profiler is not None
        else None
    )
    diagnostic_sample_token = None
    if diagnostic_policy is not None:
        diagnostic_sample_token = diagnostic_policy.before_sampling(
            physical_group=int(physical_group),
            logical_chunks=normalized_chunks,
            context_frames=int(context_frames),
            latent=latent,
            conditioning=conditioning,
            seed=int(seed),
            sigmas=sigmas,
        )
    if planner_event is not None:
        planner_event(
            packed_row_planner,
            "begin_group",
            physical_group=int(physical_group),
            logical_chunks=normalized_chunks,
            terminal_atomic=bool(terminal_atomic),
        )
    sampled = sample_callable(
        model=model,
        conditioning=conditioning,
        latent=latent,
        sampler=sampler,
        sigmas=sigmas,
        seed=int(seed),
        enable_preview=bool(enable_preview),
    )
    if planner_event is not None:
        planner_event(packed_row_planner, "finish_group")
    if layout_validation_profiler is not None and layout_profile_token is not None:
        layout_validation_profiler.finish_sampling_group(layout_profile_token)
    return sampled, diagnostic_sample_token
