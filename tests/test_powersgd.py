from datetime import timedelta
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ddp_gradient_compression.hooks.powersgd import (
    PowerSGDState,
    PowerSGDStats,
    orthogonalize,
    powersgd_hook,
)


class Bucket:
    def __init__(self, tensors, index=0, last=True):
        self.data = torch.cat([t.flatten() for t in tensors])
        self.views = []
        offset = 0
        for tensor in tensors:
            self.views.append(self.data[offset:offset + tensor.numel()].view_as(tensor))
            offset += tensor.numel()
        self.idx, self.last = index, last

    def buffer(self):
        return self.data

    def gradients(self):
        return self.views

    def index(self):
        return self.idx

    def is_last(self):
        return self.last


def test_orthogonalize_handles_zero_and_dependent_columns():
    for matrix in (torch.zeros(8, 2), torch.ones(8, 2)):
        q = orthogonalize(matrix)
        assert torch.isfinite(q).all()
        torch.testing.assert_close(q.T @ q, torch.eye(2), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('kwargs', [dict(matrix_approximation_rank=0),
                                   dict(start_powerSGD_iter=-1)])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        PowerSGDState(**kwargs)


def distributed_checks(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{path}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        state = PowerSGDState(start_powerSGD_iter=1)
        # Warmup averages and counts iterations only for the final bucket.
        for last in (False, True):
            bucket = Bucket([torch.full((8, 8), float(rank + 1))], last=last)
            output = powersgd_hook(state, bucket).wait()
            torch.testing.assert_close(output, torch.full((64,), 1.5))
            assert state.iteration == int(last)
        assert not state.residuals

        # Exact rank-one recovery, dense fallback, and repeated warm starts.
        for step in range(3):
            matrix = torch.outer(torch.arange(1., 9.), torch.arange(1., 9.)) * (rank + 1)
            bucket = Bucket([matrix, torch.full((3,), float(rank + 1))])
            future = powersgd_hook(state, bucket)
            assert isinstance(future, torch.futures.Future)
            output = future.wait()
            expected = torch.cat([(matrix / (rank + 1) * 1.5).flatten(), torch.full((3,), 1.5)])
            torch.testing.assert_close(output, expected, atol=1e-4, rtol=1e-5)
            assert output.data_ptr() == bucket.buffer().data_ptr()
            assert state.q_factors[0][0].shape == (8, 1)

        # Mixed epoch includes both warmup buckets and all compressed buckets.
        snapshot = state.snapshot_and_reset()
        assert snapshot == PowerSGDStats(
            dense_bytes=(2 * 64 + 3 * 67) * 4,
            payload_bytes=(2 * 64 + 3 * (8 + 8 + 3)) * 4,
            buckets=5,
        )
        assert state.snapshot_and_reset() == PowerSGDStats()
        assert state.iteration == 4 and state.q_factors and state.residuals

        # Higher-rank projection and local residual conserve each worker's input.
        state = PowerSGDState(matrix_approximation_rank=2, start_powerSGD_iter=0)
        for step in range(2):
            generator = torch.Generator().manual_seed(100 + rank + step)
            matrix = torch.randn(8, 8, generator=generator)
            previous = state.residuals.get(0, torch.zeros(64)).clone()
            corrected = matrix.flatten() + previous
            bucket = Bucket([matrix])
            output = powersgd_hook(state, bucket).wait()
            mean_corrected = corrected.clone()
            mean_residual = state.residuals[0].clone()
            dist.all_reduce(mean_corrected)
            dist.all_reduce(mean_residual)
            torch.testing.assert_close(output + mean_residual / 2, mean_corrected / 2,
                                       atol=1e-6, rtol=1e-5)
            other_output = output.clone()
            dist.broadcast(other_output, src=0)
            torch.testing.assert_close(output, other_output)

        assert state.snapshot_and_reset() == PowerSGDStats(
            dense_bytes=2 * 64 * 4, payload_bytes=2 * (8 + 8) * 2 * 4, buckets=2,
        )

        # Equal-numel bucket rebuild must invalidate residuals and Q.
        state.residuals[0].fill_(100)
        bucket = Bucket([torch.full((64,), float(rank + 1))])
        torch.testing.assert_close(powersgd_hook(state, bucket).wait(), torch.full((64,), 1.5))
        assert state.q_factors[0] == []
        assert torch.count_nonzero(state.residuals[0]) == 0
        assert state.snapshot_and_reset() == PowerSGDStats(64 * 4, 64 * 4, 1)

        # FP64 accounting and small-matrix fallback, with no world-size factor.
        for dtype in (torch.float32, torch.float64):
            counter_state = PowerSGDState(start_powerSGD_iter=0, matrix_approximation_rank=2)
            bucket = Bucket([torch.ones(8, 8, dtype=dtype), torch.ones(2, 2, dtype=dtype)])
            powersgd_hook(counter_state, bucket).wait()
            element_bytes = bucket.buffer().element_size()
            assert counter_state.snapshot_and_reset() == PowerSGDStats(
                68 * element_bytes, ((8 + 8) * 2 + 4) * element_bytes, 1,
            )

        state = PowerSGDState(start_powerSGD_iter=0, warm_start=False, use_error_feedback=False)
        for _ in range(2):
            bucket = Bucket([torch.zeros(8, 8)])
            torch.testing.assert_close(powersgd_hook(state, bucket).wait(), torch.zeros(64))
            assert not state.residuals and not state.q_factors

        # Real DDP registration, bucket rebuild and optimizer steps on CPU.
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(8, 8))
        state = PowerSGDState(start_powerSGD_iter=2)
        model.register_comm_hook(state, powersgd_hook)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(4):
            optimizer.zero_grad()
            model(torch.ones(4, 8) * (rank + 1)).square().mean().backward()
            optimizer.step()
        for parameter in model.parameters():
            reference = parameter.detach().clone()
            dist.broadcast(reference, src=0)
            torch.testing.assert_close(parameter, reference)
        assert state.q_factors
    finally:
        dist.destroy_process_group()


def test_distributed_powersgd(tmp_path):
    context = mp.spawn(distributed_checks, args=(str(tmp_path / 'store'),),
                       nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail('PowerSGD distributed test timed out')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def overlapping_bucket_checks(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{path}', rank=rank,
                            world_size=4, timeout=timedelta(seconds=30))
    try:
        state = PowerSGDState(start_powerSGD_iter=0)
        for step in range(4):
            first = Bucket([torch.full((8, 8), float(rank + 1)),
                            torch.full((3,), float(rank + 1))], index=0, last=False)
            second = Bucket([torch.full((16, 16), float(rank + 1)),
                             torch.full((5,), float(rank + 1))], index=1)
            pending_first = powersgd_hook(state, first)
            # Let callbacks of bucket 0 advance on some workers while others
            # invoke bucket 1. Different payload sizes expose ordering bugs.
            if rank == step % 4:
                time.sleep(0.15)
            pending_second = powersgd_hook(state, second)
            for pending, bucket in ((pending_first, first), (pending_second, second)):
                torch.testing.assert_close(pending.wait(), torch.full_like(bucket.buffer(), 2.5))
            assert state.iteration == step + 1
            assert state.snapshot_and_reset() == PowerSGDStats(
                (64 + 3 + 256 + 5) * 4, (8 + 8 + 3 + 16 + 16 + 5) * 4, 2,
            )
    finally:
        dist.destroy_process_group()


def test_overlapping_buckets_four_workers(tmp_path):
    context = mp.spawn(overlapping_bucket_checks, args=(str(tmp_path / 'store'),),
                       nprocs=4, join=False)
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail('Overlapping PowerSGD buckets timed out')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def test_epoch_payload_uses_hook_counts_and_resets():
    from ddp_gradient_compression.hooks.factory import build_hook
    from ddp_gradient_compression.train import logical_payload_for_epoch

    bundle = build_hook('powersgd', 0.01, True, 1, 10)
    state = bundle.powersgd_state
    assert state is bundle.state
    state.record(dense_bytes=1024, payload_bytes=128)
    kwargs = dict(hook_name='powersgd', num_parameters=999, steps=99,
                  topk_state=None, powersgd_state=state)
    assert logical_payload_for_epoch(**kwargs) == (1024, 128)
    assert logical_payload_for_epoch(**kwargs) == (0, 0)
