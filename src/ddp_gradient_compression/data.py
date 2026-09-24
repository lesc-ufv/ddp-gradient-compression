from __future__ import annotations

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from .distributed import DistributedContext


def _subset(dataset: Dataset, size: int, seed: int) -> Dataset:
    if size <= 0 or size >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def ensure_mnist(
    data_dir: str,
    download: bool,
    ctx: DistributedContext,
) -> None:
    """
    Download once per node, not once per global job.

    With one rank per node, every rank has LOCAL_RANK=0 and downloads its local copy.
    With multiple ranks per node, only LOCAL_RANK=0 touches the local dataset first.
    """
    transform = transforms.ToTensor()

    if download and ctx.is_local_main:
        datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    dist.barrier()


def _make_synthetic_dataset(size: int, seed: int) -> TensorDataset:
    """
    Deterministic MNIST-shaped classification data.

    This is not a benchmark dataset. It exists only for distributed smoke tests and CI.
    Every rank independently generates exactly the same tensors from the same seed.
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(size, 1, 28, 28, generator=generator)

    # Create labels from a deterministic projection so the problem has learnable structure.
    projection = torch.randn(28 * 28, 10, generator=generator)
    logits = x.flatten(1) @ projection
    y = logits.argmax(dim=1)

    return TensorDataset(x, y)


def build_loaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    seed: int,
    train_subset: int,
    test_subset: int,
    ctx: DistributedContext,
) -> tuple[DataLoader, DistributedSampler, DataLoader | None]:
    if dataset_name == "mnist":
        transform = transforms.ToTensor()

        train_dataset: Dataset = datasets.MNIST(
            data_dir,
            train=True,
            download=False,
            transform=transform,
        )
        test_dataset: Dataset = datasets.MNIST(
            data_dir,
            train=False,
            download=False,
            transform=transform,
        )

        train_dataset = _subset(train_dataset, train_subset, seed=seed)
        test_dataset = _subset(test_dataset, test_subset, seed=seed + 1)

    elif dataset_name == "synthetic":
        train_size = train_subset if train_subset > 0 else 12000
        test_size = test_subset if test_subset > 0 else 2000
        train_dataset = _make_synthetic_dataset(train_size, seed=seed)
        test_dataset = _make_synthetic_dataset(test_size, seed=seed + 1)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )

    test_loader = None
    if ctx.is_main:
        test_loader = DataLoader(
            test_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=0,
        )

    return train_loader, sampler, test_loader
