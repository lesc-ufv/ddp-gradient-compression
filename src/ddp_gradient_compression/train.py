from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import TrainConfig, parse_args
from .data import build_loaders, ensure_mnist
from .distributed import cleanup_distributed, setup_distributed
from .hooks import build_hook
from .metrics import CsvLogger, EpochMetrics
from .models import build_model


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader) -> tuple[float, float]:
    model.eval()

    loss_sum = 0.0
    correct = 0
    samples = 0

    for x, y in loader:
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        samples += y.numel()

    return loss_sum / samples, 100.0 * correct / samples


def logical_payload_for_epoch(
    hook_name: str,
    num_parameters: int,
    steps: int,
    topk_state,
) -> tuple[int, int | None]:
    dense_reference_bytes = num_parameters * 4 * steps

    if hook_name == "dense":
        return dense_reference_bytes, dense_reference_bytes

    if hook_name == "fp16":
        return dense_reference_bytes, num_parameters * 2 * steps

    if hook_name == "topk":
        snapshot = topk_state.snapshot_and_reset()
        return snapshot.dense_bytes, snapshot.payload_bytes

    # PowerSGD communication volume depends on per-parameter shapes and internal decisions.
    # Do not report a misleading estimate.
    if hook_name == "powersgd":
        return dense_reference_bytes, None

    raise ValueError(hook_name)


def train(config: TrainConfig) -> None:
    ctx = setup_distributed(config.backend)

    try:
        # torchrun may set OMP_NUM_THREADS itself. We do not override user/cluster policy here.
        seed_everything(config.seed)

        if config.dataset == "mnist":
            ensure_mnist(
                data_dir=str(config.data_dir),
                download=config.download,
                ctx=ctx,
            )
        else:
            # Keep startup ordering symmetric across ranks for smoke-test datasets too.
            dist.barrier()

        train_loader, train_sampler, test_loader = build_loaders(
            dataset_name=config.dataset,
            data_dir=str(config.data_dir),
            batch_size=config.batch_size,
            seed=config.seed,
            train_subset=config.train_subset,
            test_subset=config.test_subset,
            ctx=ctx,
        )

        model = build_model(config.hidden_size)

        # CPU/Gloo path: no device_ids are supplied.
        ddp_model = DDP(
            model,
            bucket_cap_mb=config.bucket_cap_mb,
            broadcast_buffers=False,
        )

        hook_bundle = build_hook(
            name=config.hook,
            topk_ratio=config.topk_ratio,
            topk_error_feedback=config.topk_error_feedback,
            powersgd_rank=config.powersgd_rank,
            powersgd_start_iter=config.powersgd_start_iter,
        )

        ddp_model.register_comm_hook(
            state=hook_bundle.state,
            hook=hook_bundle.hook,
        )

        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=config.lr)

        num_parameters = sum(p.numel() for p in ddp_model.module.parameters())

        logger = None
        if ctx.is_main:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = (
                config.results_dir
                / f"{timestamp}-{config.hook}-world{ctx.world_size}.csv"
            )
            logger = CsvLogger(output_path)

            print("=" * 72)
            print("DDP Gradient Compression Lab")
            print(f"backend            : {config.backend}")
            print(f"hook               : {config.hook}")
            print(f"dataset            : {config.dataset}")
            print(f"world_size         : {ctx.world_size}")
            print(f"parameters         : {num_parameters:,}")
            print(f"local batch        : {config.batch_size}")
            print(f"approx global batch: {config.batch_size * ctx.world_size}")
            print(f"bucket_cap_mb      : {config.bucket_cap_mb}")
            if config.hook == "topk":
                print(f"topk_ratio         : {config.topk_ratio}")
                print(f"error_feedback     : {config.topk_error_feedback}")
            if config.hook == "powersgd":
                print(f"powersgd_rank      : {config.powersgd_rank}")
                print(f"powersgd_start_iter: {config.powersgd_start_iter}")
            print(f"results            : {output_path}")
            print("=" * 72)

        dist.barrier()

        for epoch in range(1, config.epochs + 1):
            train_sampler.set_epoch(epoch)
            ddp_model.train()

            dist.barrier()
            epoch_start = time.perf_counter()

            local_loss_sum = 0.0
            local_samples = 0
            steps = 0

            for x, y in train_loader:
                optimizer.zero_grad(set_to_none=True)

                logits = ddp_model(x)
                loss = F.cross_entropy(logits, y)

                # DDP invokes the registered communication hook while backward progresses.
                loss.backward()

                optimizer.step()

                local_loss_sum += loss.item() * y.numel()
                local_samples += y.numel()
                steps += 1

            # Aggregate loss statistics across workers.
            stats = torch.tensor(
                [local_loss_sum, float(local_samples)],
                dtype=torch.float64,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            train_loss = stats[0].item() / stats[1].item()

            dist.barrier()
            epoch_seconds = time.perf_counter() - epoch_start

            dense_bytes, payload_bytes = logical_payload_for_epoch(
                hook_name=config.hook,
                num_parameters=num_parameters,
                steps=steps,
                topk_state=hook_bundle.topk_state,
            )

            dense_mb = dense_bytes / (1024 ** 2)
            payload_mb = (
                payload_bytes / (1024 ** 2)
                if payload_bytes is not None
                else None
            )
            compression_ratio = (
                dense_bytes / payload_bytes
                if payload_bytes not in (None, 0)
                else None
            )

            # Only rank 0 evaluates. Use the underlying module so other ranks do not need
            # to participate in DDP forward-time synchronization.
            if ctx.is_main:
                assert test_loader is not None
                test_loss, test_accuracy = evaluate(ddp_model.module, test_loader)

                metrics = EpochMetrics(
                    epoch=epoch,
                    hook=config.hook,
                    world_size=ctx.world_size,
                    train_loss=train_loss,
                    test_loss=test_loss,
                    test_accuracy=test_accuracy,
                    epoch_seconds=epoch_seconds,
                    dense_reference_mb_per_worker=dense_mb,
                    logical_payload_mb_per_worker=payload_mb,
                    logical_compression_ratio=compression_ratio,
                )

                assert logger is not None
                logger.write(metrics)

                payload_text = (
                    f"{payload_mb:.3f} MiB"
                    if payload_mb is not None
                    else "not estimated"
                )
                compression_text = (
                    f"{compression_ratio:.2f}x"
                    if compression_ratio is not None
                    else "n/a"
                )

                print(
                    f"epoch={epoch:02d} "
                    f"train_loss={train_loss:.4f} "
                    f"test_loss={test_loss:.4f} "
                    f"accuracy={test_accuracy:.2f}% "
                    f"time={epoch_seconds:.2f}s "
                    f"payload={payload_text} "
                    f"compression={compression_text}"
                )

            # Keep all workers aligned before the next epoch.
            dist.barrier()

    finally:
        cleanup_distributed()


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
