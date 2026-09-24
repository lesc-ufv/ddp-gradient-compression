import math
import threading
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


@dataclass
class TopKStats:
    dense_bytes: int = 0
    payload_bytes: int = 0
    buckets: int = 0

    def reset(self) -> None:
        self.dense_bytes = 0
        self.payload_bytes = 0
        self.buckets = 0


@dataclass
class TopKState:
    keep_ratio: float
    process_group: dist.ProcessGroup | None = None
    error_feedback: bool = True
    residuals: dict[int, torch.Tensor] = field(default_factory=dict)
    stats: TopKStats = field(default_factory=TopKStats)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, numel: int, k: int) -> None:
        # Dense FP32 reference: 4 bytes/element.
        dense_bytes = numel * 4
        # Sparse payload: FP32 value + INT32 index = 8 bytes/selected element.
        payload_bytes = k * 8

        with self._lock:
            self.stats.dense_bytes += dense_bytes
            self.stats.payload_bytes += payload_bytes
            self.stats.buckets += 1

    def snapshot_and_reset(self) -> TopKStats:
        with self._lock:
            snapshot = TopKStats(
                dense_bytes=self.stats.dense_bytes,
                payload_bytes=self.stats.payload_bytes,
                buckets=self.stats.buckets,
            )
            self.stats.reset()
            return snapshot


def select_topk_with_error_feedback(
    gradient: torch.Tensor,
    residual: torch.Tensor,
    keep_ratio: float,
    use_error_feedback: bool,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Pure local Top-K step.

    Returns:
        indices_int32: selected flattened positions
        values_fp32: selected values
        k: number of selected entries

    residual is updated in place.
    """
    if gradient.dtype != torch.float32:
        raise TypeError(
            f"Custom Top-K hook currently expects FP32 buckets, got {gradient.dtype}."
        )

    flat = gradient.reshape(-1)
    corrected = flat + residual

    numel = corrected.numel()
    k = max(1, math.ceil(numel * keep_ratio))

    indices64 = torch.topk(
        corrected.abs(),
        k,
        largest=True,
        sorted=False,
    ).indices

    values = corrected.index_select(0, indices64).contiguous()

    if use_error_feedback:
        # e_(t+1) = u_t - C(u_t)
        residual.copy_(corrected)
        residual.index_fill_(0, indices64, 0.0)
    else:
        residual.zero_()

    if numel > torch.iinfo(torch.int32).max:
        raise ValueError(
            "Bucket is too large for the current INT32 index representation."
        )

    indices32 = indices64.to(dtype=torch.int32).contiguous()
    return indices32, values, k


def pack_sparse_payload(
    indices32: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """
    Pack FP32 values and INT32 indices into one FP32 tensor.

    The index half is not numerically converted to float. Its raw INT32 bit pattern is
    reinterpreted as FP32, preserving all 32 index bits while allowing one tensor collective.
    """
    if indices32.dtype != torch.int32:
        raise TypeError("indices32 must have dtype torch.int32.")
    if values.dtype != torch.float32:
        raise TypeError("values must have dtype torch.float32.")
    if indices32.numel() != values.numel():
        raise ValueError("indices and values must have the same number of elements.")

    k = values.numel()
    payload = torch.empty(2 * k, dtype=torch.float32, device=values.device)
    payload[:k].copy_(values)
    payload[k:].copy_(indices32.view(torch.float32))
    return payload


def unpack_sparse_payload(
    payload: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if payload.dtype != torch.float32:
        raise TypeError("payload must have dtype torch.float32.")
    if payload.numel() % 2 != 0:
        raise ValueError("payload length must be even.")

    k = payload.numel() // 2
    values = payload[:k]
    indices32 = payload[k:].contiguous().view(torch.int32)
    return indices32, values


def topk_hook(
    state: TopKState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """
    DDP communication hook implementing local Top-K + error feedback + all_gather.

    Every rank chooses K entries from the same-size bucket, so all sparse payloads have the
    same shape and can be exchanged with all_gather.

    The all_gather itself is asynchronous. The returned Future reconstructs and averages the
    sparse contributions once communication completes.
    """
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)

    gradient = bucket.buffer()
    bucket_index = bucket.index()

    residual = state.residuals.get(bucket_index)
    if residual is None or residual.shape != gradient.shape:
        residual = torch.zeros_like(gradient)
        state.residuals[bucket_index] = residual

    indices32, values, k = select_topk_with_error_feedback(
        gradient=gradient,
        residual=residual,
        keep_ratio=state.keep_ratio,
        use_error_feedback=state.error_feedback,
    )

    payload = pack_sparse_payload(indices32, values)
    gathered_payloads = [torch.empty_like(payload) for _ in range(world_size)]

    work = dist.all_gather(
        gathered_payloads,
        payload,
        group=group,
        async_op=True,
    )

    future = work.get_future()

    def reconstruct(_: torch.futures.Future) -> torch.Tensor:
        aggregated = torch.zeros_like(gradient.reshape(-1))

        for rank_payload in gathered_payloads:
            rank_indices32, rank_values = unpack_sparse_payload(rank_payload)
            aggregated.index_add_(
                0,
                rank_indices32.to(dtype=torch.int64),
                rank_values,
            )

        aggregated.div_(world_size)
        state.record(numel=gradient.numel(), k=k)
        return aggregated.view_as(gradient)

    return future.then(reconstruct)
