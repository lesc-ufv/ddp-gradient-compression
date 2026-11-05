"""PowerSGD with asynchronous Gloo collectives and local error feedback."""

from dataclasses import dataclass, field
import threading

import torch
import torch.distributed as dist


@dataclass
class PowerSGDStats:
    """Logical tensor bytes per worker, not physical network traffic."""

    dense_bytes: int = 0
    payload_bytes: int = 0
    buckets: int = 0


@dataclass
class PowerSGDState:
    matrix_approximation_rank: int = 1
    start_powerSGD_iter: int = 10
    process_group: dist.ProcessGroup | None = None
    use_error_feedback: bool = True
    warm_start: bool = True
    random_seed: int = 42
    iteration: int = 0
    residuals: dict[int, torch.Tensor] = field(default_factory=dict)
    q_factors: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    layouts: dict[int, tuple] = field(default_factory=dict)

    stats: PowerSGDStats = field(default_factory=PowerSGDStats)
    _stats_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _tail: torch.futures.Future | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.matrix_approximation_rank < 1:
            raise ValueError("PowerSGD rank must be positive.")
        if self.start_powerSGD_iter < 0:
            raise ValueError("PowerSGD start iteration cannot be negative.")

    def record(self, dense_bytes: int, payload_bytes: int) -> None:
        with self._stats_lock:
            self.stats.dense_bytes += dense_bytes
            self.stats.payload_bytes += payload_bytes
            self.stats.buckets += 1

    def snapshot_and_reset(self) -> PowerSGDStats:
        with self._stats_lock:
            snapshot = self.stats
            self.stats = PowerSGDStats()
            return snapshot


def orthogonalize(matrix: torch.Tensor) -> torch.Tensor:
    """Reduced QR also supplies a finite orthonormal basis for zero inputs."""
    return torch.linalg.qr(matrix, mode="reduced").Q


def _powersgd_hook(
    state: PowerSGDState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Approximate matrices by P Q^T; average vectors and small matrices densely.

    All ranks start with identical Q. Reduce P=M Q, orthogonalize it, then
    reduce Q=M^T P. The residual is the *local* projection error, computed
    before averaging Q, so it does not accumulate differences between workers.
    Callbacks launch the next collective without waiting on unfinished futures.
    """
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    buffer = bucket.buffer()
    if buffer.dtype not in (torch.float32, torch.float64):
        raise TypeError("Custom PowerSGD expects FP32 or FP64 buckets.")
    iteration = state.iteration
    if bucket.is_last():
        state.iteration += 1

    if iteration < state.start_powerSGD_iter:
        def average(future):
            future.value()  # Propagate collective errors.
            buffer.div_(world_size)
            dense_bytes = buffer.numel() * buffer.element_size()
            state.record(dense_bytes, dense_bytes)
            return buffer

        return dist.all_reduce(buffer, group=group, async_op=True).get_future().then(average)

    index = bucket.index()
    gradients = bucket.gradients()
    layout = (tuple(tuple(t.shape) for t in gradients), buffer.dtype, buffer.device)
    if state.layouts.get(index) != layout:
        # DDP can rebuild buckets after the first backward, even at equal numel.
        state.residuals.pop(index, None)
        state.q_factors.pop(index, None)
        state.layouts[index] = layout

    corrected = buffer.detach().clone()
    if state.use_error_feedback and index in state.residuals:
        corrected.add_(state.residuals[index])
    residual = torch.zeros_like(buffer)
    matrices, outputs, dense_inputs, dense_outputs = [], [], [], []
    offset = 0
    for gradient in gradients:
        tensor = corrected[offset:offset + gradient.numel()].view_as(gradient)
        error = residual[offset:offset + gradient.numel()].view_as(gradient)
        offset += gradient.numel()
        n = gradient.shape[0] if gradient.ndim else 1
        m = gradient.numel() // n if n else 0
        rank = min(n, m, state.matrix_approximation_rank)
        if gradient.ndim > 1 and rank > 0 and (n + m) * rank < n * m:
            matrices.append((tensor.reshape(n, m), error.reshape(n, m), rank))
            outputs.append(gradient)
        else:
            dense_inputs.append(tensor.reshape(-1))
            dense_outputs.append(gradient)

    qs = state.q_factors.get(index) if state.warm_start else None
    if qs is None:
        generator = torch.Generator(device=buffer.device)
        generator.manual_seed(state.random_seed + index)
        qs = [torch.randn(m.shape[1], r, generator=generator,
                          dtype=buffer.dtype, device=buffer.device)
              for m, _, r in matrices]
    qs = [orthogonalize(q) for q in qs]
    ps = [matrix @ q for (matrix, _, _), q in zip(matrices, qs)]
    dense = torch.cat(dense_inputs) if dense_inputs else buffer.new_empty(0)
    p_payload = torch.cat([p.reshape(-1) for p in ps]) if ps else buffer.new_empty(0)
    result = torch.futures.Future()

    def reduce_async(payload, continuation):
        if payload.numel() == 0:
            continuation()
            return

        def done(future):
            try:
                future.value()
                continuation()
            except Exception as exc:
                result.set_exception(exc)

        dist.all_reduce(payload, group=group, async_op=True).get_future().then(done)

    def finish(q_payload, local_qs):
        q_payload.div_(world_size)
        pos = 0
        averaged_qs = []
        for p, local_q, output in zip(ps, local_qs, outputs):
            q = q_payload[pos:pos + local_q.numel()].view_as(local_q)
            pos += local_q.numel()
            output.copy_((p @ q.T).view_as(output))
            averaged_qs.append(q.clone())
        if state.warm_start:
            state.q_factors[index] = averaged_qs
        else:
            state.q_factors.pop(index, None)
        if state.use_error_feedback:
            state.residuals[index] = residual
        else:
            state.residuals.pop(index, None)
        # Count each collective input once: dense fallback + P + Q. No
        # world-size multiplier or all-reduce topology/network overhead.
        state.record(
            dense_bytes=buffer.numel() * buffer.element_size(),
            payload_bytes=sum(t.numel() * t.element_size()
                              for t in (dense, p_payload, q_payload)),
        )
        result.set_result(buffer)

    def after_p():
        pos = 0
        local_qs = []
        for i, (matrix, error, _) in enumerate(matrices):
            p = p_payload[pos:pos + ps[i].numel()].view_as(ps[i])
            pos += ps[i].numel()
            ps[i] = orthogonalize(p)
            q = matrix.T @ ps[i]
            local_qs.append(q)
            if state.use_error_feedback:
                error.copy_(matrix - ps[i] @ q.T)
        q_payload = (torch.cat([q.reshape(-1) for q in local_qs])
                     if local_qs else buffer.new_empty(0))
        reduce_async(q_payload, lambda: finish(q_payload, local_qs))

    def after_dense():
        dense.div_(world_size)
        pos = 0
        for output in dense_outputs:
            output.copy_(dense[pos:pos + output.numel()].view_as(output))
            pos += output.numel()
        reduce_async(p_payload, after_p)

    try:
        reduce_async(dense, after_dense)
    except Exception as exc:
        result.set_exception(exc)
    return result


def powersgd_hook(
    state: PowerSGDState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Serialize bucket pipelines without blocking the autograd thread.

    DDP invokes hooks in bucket order, but collective callbacks may run while
    the next bucket's hook is being invoked. Chaining complete pipelines keeps
    dense/P/Q collectives in the same order on every worker. The returned future
    holds the bucket until its queued pipeline finishes.
    """
    result = torch.futures.Future()
    previous = state._tail
    # Publish before attaching callbacks: a completed future may run them inline.
    state._tail = result

    def forward(completed):
        try:
            result.set_result(completed.value())
        except Exception as exc:
            result.set_exception(exc)

    def start(completed=None):
        try:
            if completed is not None:
                completed.value()  # Propagate errors from the preceding bucket.
            _powersgd_hook(state, bucket).then(forward)
        except Exception as exc:
            result.set_exception(exc)

    if previous is None:
        start()
    else:
        previous.then(start)
    return result
