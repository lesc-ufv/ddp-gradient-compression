from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    backend: str
    hook: str
    dataset: str
    data_dir: Path
    results_dir: Path
    download: bool
    epochs: int
    batch_size: int
    lr: float
    seed: int
    train_subset: int
    test_subset: int
    hidden_size: int
    bucket_cap_mb: float
    topk_ratio: float
    topk_error_feedback: bool
    powersgd_rank: int
    powersgd_start_iter: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PyTorch DDP gradient-compression laboratory."
    )

    parser.add_argument("--backend", default="gloo", choices=["gloo", "nccl", "mpi"])
    parser.add_argument(
        "--hook",
        default="dense",
        choices=["dense", "fp16", "topk", "powersgd"],
    )
    parser.add_argument(
        "--dataset",
        default="mnist",
        choices=["mnist", "synthetic"],
        help="synthetic is useful for offline DDP smoke tests.",
    )

    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Download MNIST on LOCAL_RANK=0 of every node.",
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--train-subset",
        type=int,
        default=12000,
        help="0 uses the entire training set.",
    )
    parser.add_argument(
        "--test-subset",
        type=int,
        default=2000,
        help="0 uses the entire test set.",
    )

    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument(
        "--bucket-cap-mb",
        type=float,
        default=1.0,
        help="DDP gradient bucket capacity in MiB.",
    )

    parser.add_argument("--topk-ratio", type=float, default=0.01)
    parser.add_argument(
        "--topk-error-feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--powersgd-rank", type=int, default=1)
    parser.add_argument("--powersgd-start-iter", type=int, default=10)

    return parser


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    args = build_parser().parse_args(argv)

    if not (0.0 < args.topk_ratio <= 1.0):
        raise ValueError("--topk-ratio must be in the interval (0, 1].")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.bucket_cap_mb <= 0:
        raise ValueError("--bucket-cap-mb must be positive.")
    if args.powersgd_rank <= 0:
        raise ValueError("--powersgd-rank must be positive.")
    if args.powersgd_start_iter < 0:
        raise ValueError("--powersgd-start-iter cannot be negative.")

    return TrainConfig(
        backend=args.backend,
        hook=args.hook,
        dataset=args.dataset,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        download=args.download,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        hidden_size=args.hidden_size,
        bucket_cap_mb=args.bucket_cap_mb,
        topk_ratio=args.topk_ratio,
        topk_error_feedback=args.topk_error_feedback,
        powersgd_rank=args.powersgd_rank,
        powersgd_start_iter=args.powersgd_start_iter,
    )
