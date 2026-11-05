"""Contract tests using an explicit test double, NOT ACCL emulator benchmarks."""

from datetime import timedelta
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ddp_gradient_compression.backends.accl import ProcessGroupACCL, _create_process_group
from ddp_gradient_compression.backends.accl_config import ACCLConfig
from ddp_gradient_compression.backends.accl_runtime import PyACCLRuntime
from ddp_gradient_compression.config import parse_args


def config_dict(size=2):
    return dict(mode='emulator', protocol='TCP', nbufs=16, bufsize=64, timeout_s=15,
                ranks=[dict(ip='127.0.0.1', port=9000+i, session_id=i, max_segment_size=64)
                       for i in range(size)],
                devices=[dict(sim_sock=f'tcp://127.0.0.1:{5500+i}') for i in range(size)])


def test_config_and_backend_selection(tmp_path, monkeypatch):
    assert parse_args(['--backend', 'accl']).backend == 'accl'
    assert parse_args([]).backend == 'gloo'
    path = tmp_path / 'accl.json'
    path.write_text(json.dumps(config_dict()))
    assert ACCLConfig.load(str(path), 2).mode == 'emulator'
    with pytest.raises(ValueError, match='WORLD_SIZE'):
        ACCLConfig.load(str(path), 4)
    bad = config_dict()
    bad['mode'] = 'mock'
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='mode'):
        ACCLConfig.load(str(path), 2)
    monkeypatch.delenv('ACCL_CONFIG', raising=False)
    with pytest.raises(RuntimeError, match='ACCL_CONFIG'):
        _create_process_group(dist.HashStore(), 0, 2, timedelta(seconds=1))


def test_no_silent_fallback_without_pyaccl(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyaccl', None)
    with pytest.raises(RuntimeError, match='Gloo is not used as a fallback'):
        PyACCLRuntime(ACCLConfig(**config_dict()), 0, 2)


class Buffer(np.ndarray):
    def __new__(cls, shape, dtype, events):
        obj = np.zeros(shape, dtype=dtype).view(cls)
        obj.events = events
        return obj

    def __array_finalize__(self, parent):
        self.events = getattr(parent, 'events', [])

    def sync_to_device(self):
        self.events.append('to_device')

    def sync_from_device(self):
        self.events.append('from_device')


class DriverDouble:
    def __init__(self, **kwargs):
        self.events = []
        self.calls = []

    def allocate(self, shape, dtype):
        return Buffer(shape, dtype, self.events)

    def handle(self, name, source, output, count, **kwargs):
        assert kwargs == dict(from_fpga=True, to_fpga=True, compress_dtype=None, run_async=True)
        assert source.nbytes % 64 == 0
        self.calls.append((name, count))

        def wait():
            self.events.append('wait')
            if name == 'allreduce':
                output[:] = source * 2
            elif name == 'allgather':
                # Preserve payload bits, including NaNs used to store indices.
                output.view(np.uint8)[:] = np.tile(source.view(np.uint8), 2)
        return SimpleNamespace(wait=wait)

    def allreduce(self, source, output, count, function, **kwargs):
        assert function == 0
        return self.handle('allreduce', source, output, count, **kwargs)

    def allgather(self, source, output, count, **kwargs):
        return self.handle('allgather', source, output, count, **kwargs)

    def bcast(self, source, count, root, **kwargs):
        assert root == 0
        return self.handle('broadcast', source, source, count, **kwargs)

    def check_return_value(self, name):
        self.events.append('check')

    def barrier(self):
        self.events.append('barrier')


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.float64, torch.int32, torch.int64])
def test_pyaccl_adapter_chunks_syncs_and_preserves_shape(monkeypatch, dtype):
    monkeypatch.setitem(sys.modules, 'pyaccl', SimpleNamespace(
        accl=DriverDouble, ACCLReduceFunctions=SimpleNamespace(SUM=0)))
    runtime = PyACCLRuntime(ACCLConfig(**config_dict()), 0, 2)
    tensor = torch.arange(35, dtype=dtype)
    runtime.allreduce(tensor)
    torch.testing.assert_close(tensor, torch.arange(35, dtype=dtype) * 2)
    assert runtime.driver.events[-3:] == ['wait', 'check', 'from_device']
    assert len(runtime.driver.calls) > 1
    runtime.broadcast(tensor, 0)
    torch.testing.assert_close(tensor, torch.arange(35, dtype=dtype) * 2)
    runtime.barrier()
    runtime.close()


def test_pyaccl_adapter_preserves_topk_index_bits(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyaccl', SimpleNamespace(
        accl=DriverDouble, ACCLReduceFunctions=SimpleNamespace(SUM=0)))
    runtime = PyACCLRuntime(ACCLConfig(**config_dict()), 0, 2)
    bits = torch.tensor([0, 1, 123456, 2147483646], dtype=torch.int32).repeat(11)
    payload = bits.view(torch.float32)
    outputs = [torch.empty_like(payload) for _ in range(2)]
    runtime.allgather(outputs, payload)
    for output in outputs:
        assert torch.equal(output.view(torch.int32), bits)
    # Native refusal must never look like a successful empty operation.
    runtime.driver.allgather = lambda *args, **kwargs: None
    with pytest.raises(RuntimeError, match='did not launch'):
        runtime.allgather(outputs, payload)
    runtime.close()


class LocalRuntime:
    def __init__(self, *args):
        pass

    def close(self):
        pass


def test_async_timeout_poisoning():
    entered, release = threading.Event(), threading.Event()

    class BlockedRuntime(LocalRuntime):
        def allreduce(self, tensor):
            entered.set()
            release.wait(5)

    config = SimpleNamespace(timeout_s=0.15)
    pg = ProcessGroupACCL(dist.HashStore(), 0, 1, timedelta(seconds=1), config, BlockedRuntime)
    try:
        work = pg.allreduce([torch.ones(1)])
        assert entered.wait(1)
        # Future completion also fails on timeout, without requiring Work.wait.
        with pytest.raises(Exception, match='timed out'):
            work.get_future().wait()
        with pytest.raises(RuntimeError, match='failed'):
            pg.barrier()
    finally:
        release.set()
        pg.shutdown()


def test_validation_rejects_unsupported_operations():
    pg = ProcessGroupACCL(dist.HashStore(), 0, 1, timedelta(seconds=1),
                          SimpleNamespace(timeout_s=1), LocalRuntime)
    try:
        with pytest.raises(TypeError, match='dtype'):
            pg.allreduce([torch.ones(1, dtype=torch.bool)])
        with pytest.raises(ValueError, match='contiguous'):
            pg.allreduce([torch.ones(4)[::2]])
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.MAX
        with pytest.raises(NotImplementedError, match='SUM'):
            pg.allreduce([torch.ones(1)], opts)
    finally:
        pg.shutdown()


class StoreRuntimeDouble:
    """Test-only cross-process arithmetic; never selectable by ACCL_CONFIG."""

    def __init__(self, store, rank, size):
        self.store, self.rank, self.size, self.sequence = store, rank, size, 0

    def exchange(self, name, tensor):
        prefix = f'op/{self.sequence}'
        self.sequence += 1
        keys = [f'{prefix}/{i}' for i in range(self.size)]
        metadata = json.dumps([name, str(tensor.dtype), list(tensor.shape)])
        self.store.set(f'{prefix}/meta/{self.rank}', metadata)
        self.store.set(keys[self.rank], tensor.numpy().tobytes())
        self.store.wait(keys, timedelta(seconds=20))
        for rank in range(self.size):
            assert self.store.get(f'{prefix}/meta/{rank}').decode() == metadata
        return [torch.from_numpy(np.frombuffer(self.store.get(key), dtype=tensor.numpy().dtype).copy())
                .reshape(tensor.shape) for key in keys]

    def allreduce(self, tensor):
        values = self.exchange('sum', tensor)
        tensor.copy_(torch.stack(values).sum(0).to(tensor.dtype))

    def broadcast(self, tensor, root):
        tensor.copy_(self.exchange(f'broadcast/{root}', tensor)[root])

    def allgather(self, outputs, tensor):
        for output, value in zip(outputs, self.exchange('gather', tensor)):
            output.copy_(value)

    def barrier(self):
        self.exchange('barrier', torch.zeros(1, dtype=torch.int32))

    def close(self):
        pass


def distributed_backend_checks(rank, path):
    torch.set_num_threads(1)
    config = SimpleNamespace(timeout_s=20)

    def creator(store, group_rank, size, timeout):
        return ProcessGroupACCL(store, group_rank, size, timeout, config,
                                lambda c, r, w: StoreRuntimeDouble(store, r, w))

    dist.Backend.register_backend('accl', creator, devices=['cpu'])
    dist.init_process_group('accl', store=dist.FileStore(path, 2), rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    try:
        from ddp_gradient_compression.hooks import build_hook
        assert dist.get_backend() == 'accl'
        value = torch.tensor([float(rank + 1)], dtype=torch.float64)
        future = dist.all_reduce(value, async_op=True).get_future()
        assert torch.equal(future.wait()[0], torch.tensor([3.], dtype=torch.float64))
        bits = torch.tensor([rank, 2147483646], dtype=torch.int32)
        gathered = [torch.empty_like(bits) for _ in range(2)]
        dist.all_gather(gathered, bits)
        assert gathered[0].tolist() == [0, 2147483646]
        assert gathered[1].tolist() == [1, 2147483646]
        output = torch.empty(4, dtype=torch.int32)
        dist.all_gather_into_tensor(output, bits)
        assert output.tolist() == [0, 2147483646, 1, 2147483646]
        dist.barrier()
        for name in ('dense', 'fp16', 'topk', 'powersgd'):
            torch.manual_seed(42 + rank)  # DDP must broadcast rank 0's parameters.
            model = torch.nn.parallel.DistributedDataParallel(torch.nn.Sequential(
                torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8)),
                bucket_cap_mb=0.001)
            hook = build_hook(name, 0.25, True, 1, 2)
            model.register_comm_hook(hook.state, hook.hook)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            for _ in range(5):
                optimizer.zero_grad()
                model(torch.ones(4, 16) * (rank + 1)).square().mean().backward()
                optimizer.step()
            for parameter in model.parameters():
                reference = parameter.detach().clone()
                dist.broadcast(reference, 0)
                torch.testing.assert_close(parameter, reference)
            # Same FP64 loss-metric reduction used by train.py.
            stats = torch.tensor([rank + 1., 4.], dtype=torch.float64)
            dist.all_reduce(stats)
            torch.testing.assert_close(stats, torch.tensor([3., 8.], dtype=torch.float64))
            dist.barrier()
    finally:
        dist.group.WORLD.shutdown()
        dist.destroy_process_group()


def test_process_group_ddp_with_two_workers(tmp_path):
    context = mp.spawn(distributed_backend_checks, args=(str(tmp_path / 'store'),), nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail('ACCL ProcessGroup contract test timed out')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def test_hardware_mapping_requires_distinct_boards(tmp_path):
    raw = config_dict()
    raw.update(mode='hardware', driver_factory='cluster_accl:create_driver')
    raw['devices'] = [dict(host='node0', board_idx=0, xclbin='design.xclbin') for _ in range(2)]
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='distinct'):
        ACCLConfig.load(str(path), 2)
    raw['devices'][1]['board_idx'] = 1
    path.write_text(json.dumps(raw))
    parsed = ACCLConfig.load(str(path), 2)
    assert parsed.devices[0]['xclbin'] == str(tmp_path / 'design.xclbin')


def test_driver_failure_propagates_to_future_and_wait():
    class FailedRuntime(LocalRuntime):
        def allreduce(self, tensor):
            raise RuntimeError('CCLO rejected operation')

    pg = ProcessGroupACCL(dist.HashStore(), 0, 1, timedelta(seconds=1),
                          SimpleNamespace(timeout_s=1), FailedRuntime)
    try:
        work = pg.allreduce([torch.ones(1)])
        with pytest.raises(Exception, match='CCLO rejected operation'):
            work.get_future().wait()
        with pytest.raises(Exception, match='CCLO rejected operation'):
            work.wait()
        assert not work.is_success()
    finally:
        pg.shutdown()
