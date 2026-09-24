from __future__ import annotations

import os
from dataclasses import dataclass

import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    backend: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_local_main(self) -> bool:
        return self.local_rank == 0


def setup_distributed(backend: str) -> DistributedContext:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")

    # torchrun supplies RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR and MASTER_PORT.
    dist.init_process_group(backend=backend, init_method="env://")

    return DistributedContext(
        rank=dist.get_rank(),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=dist.get_world_size(),
        backend=backend,
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
