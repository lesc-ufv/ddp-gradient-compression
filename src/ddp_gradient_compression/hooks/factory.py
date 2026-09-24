from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD

from .topk import TopKState, topk_hook


CommHook = Callable[
    [Any, dist.GradBucket],
    torch.futures.Future[torch.Tensor],
]


@dataclass
class HookBundle:
    name: str
    state: Any
    hook: CommHook
    topk_state: TopKState | None = None


def build_hook(
    name: str,
    topk_ratio: float,
    topk_error_feedback: bool,
    powersgd_rank: int,
    powersgd_start_iter: int,
) -> HookBundle:
    process_group = dist.group.WORLD

    if name == "dense":
        return HookBundle(
            name=name,
            state=process_group,
            hook=default_hooks.allreduce_hook,
        )

    if name == "fp16":
        return HookBundle(
            name=name,
            state=process_group,
            hook=default_hooks.fp16_compress_hook,
        )

    if name == "topk":
        state = TopKState(
            keep_ratio=topk_ratio,
            process_group=process_group,
            error_feedback=topk_error_feedback,
        )
        return HookBundle(
            name=name,
            state=state,
            hook=topk_hook,
            topk_state=state,
        )

    if name == "powersgd":
        state = powerSGD.PowerSGDState(
            process_group=process_group,
            matrix_approximation_rank=powersgd_rank,
            start_powerSGD_iter=powersgd_start_iter,
            use_error_feedback=True,
            warm_start=True,
        )
        return HookBundle(
            name=name,
            state=state,
            hook=powerSGD.powerSGD_hook,
        )

    raise ValueError(f"Unknown hook: {name}")
