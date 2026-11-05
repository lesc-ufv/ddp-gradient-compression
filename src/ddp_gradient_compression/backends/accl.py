"""Experimental CPU ProcessGroup backed by ACCL collectives, never by Gloo.

One daemon I/O thread owns the PyACCL instance and serializes submissions.
Completion futures contain CPU tensors only after device-to-host synchronization.
"""

from datetime import timedelta
import os
import queue
import threading
import weakref

import torch
import torch.distributed as dist

from .accl_config import ACCLConfig
from .accl_runtime import DTYPES, PyACCLRuntime


_active_group = None


class ACCLWork(dist.Work):
    def __init__(self, timeout: float, fail_group):
        super().__init__()
        self._future = torch.futures.Future()
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._finished = False
        self._error = None
        self._timeout = timeout
        self._fail_group = fail_group

    def finish(self, tensors=None, error=None):
        with self._lock:
            if self._finished:
                return
            self._finished = True
            self._error = error
        try:
            if error is not None:
                self._future.set_exception(error)
            else:
                self._future.set_result(tensors)
        finally:
            self._event.set()

    def get_future(self):
        return self._future

    def wait(self, timeout=None):
        seconds = self._timeout if timeout is None else timeout.total_seconds()
        if seconds <= 0:
            seconds = self._timeout
        if not self._event.wait(seconds):
            self._fail_group(TimeoutError('ACCL Work timed out; process group cannot be reused.'))
        self._future.wait()  # Propagate the original runtime exception.
        return True

    def is_completed(self):
        return self._future.done()

    def is_success(self):
        return self.is_completed() and self._error is None

    def result(self):
        self.wait()
        return self._future.value()

    def synchronize(self):
        self.wait()


class ProcessGroupACCL(dist.ProcessGroup):
    def __init__(self, store, rank, world_size, timeout, config, runtime_factory=PyACCLRuntime):
        super().__init__(store, rank, world_size)
        self._rank, self._size = rank, world_size
        self.backend_mode = getattr(config, "mode", "test-double")
        self._timeout_seconds = min(timeout.total_seconds(), config.timeout_s)
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._pending = set()
        self._fatal = None
        self._closed = False
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(runtime_factory, config), name=f'accl-io-{rank}', daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(self._timeout_seconds):
            self._fail(TimeoutError('ACCL driver initialization timed out.'))
            self._queue.put(None)
        if self._fatal is not None:
            raise self._fatal

    def getBackendName(self):
        return 'accl'

    def _fail(self, error):
        with self._lock:
            if self._fatal is None:
                self._fatal = error
            pending = list(self._pending)
            failure = self._fatal
        for work in pending:
            work.finish(error=failure)

    def _run(self, factory, config):
        runtime = None
        try:
            runtime = factory(config, self._rank, self._size)
        except Exception as exc:
            self._fail(exc)
        finally:
            self._ready.set()
        if runtime is None:
            return
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                work, operation, tensors, timer = item
                try:
                    if self._fatal is not None:
                        work.finish(error=self._fatal)
                    else:
                        operation(runtime)
                        work.finish(tensors=tensors)
                except Exception as exc:
                    self._fail(exc)
                finally:
                    timer.cancel()
                    with self._lock:
                        self._pending.discard(work)
        finally:
            # Only the I/O thread frees resources, and only after a native call
            # has returned. A timed-out native request is not safely cancellable.
            try:
                runtime.close()
            except Exception as exc:
                self._fail(exc)

    def _submit(self, operation, tensors):
        work = ACCLWork(self._timeout_seconds, self._fail)
        timer = threading.Timer(
            self._timeout_seconds, self._deadline, args=(work,),
        )
        timer.daemon = True
        with self._lock:
            if self._fatal is not None:
                raise RuntimeError('ACCL process group has failed.') from self._fatal
            if self._closed:
                raise RuntimeError('ACCL process group is closed.')
            self._pending.add(work)
            self._queue.put((work, operation, tensors, timer))
        timer.start()
        return work

    def _deadline(self, work):
        if not work.is_completed():
            self._fail(TimeoutError('ACCL collective timed out; restart the job and check/reset the device.'))

    @staticmethod
    def _validate(tensors):
        for tensor in tensors:
            if tensor.device.type != 'cpu' or tensor.layout != torch.strided or not tensor.is_contiguous():
                raise ValueError('ACCL currently requires contiguous strided CPU tensors.')
            if tensor.dtype not in DTYPES:
                raise TypeError(f'ACCL does not support dtype {tensor.dtype}.')

    def allreduce(self, tensors, opts=None):
        tensors = list(tensors)
        self._validate(tensors)
        if opts is not None and opts.reduceOp != dist.ReduceOp.SUM:
            raise NotImplementedError('ACCL backend currently supports only ReduceOp.SUM.')
        return self._submit(lambda runtime: [runtime.allreduce(t) for t in tensors], tensors)

    def allreduce_coalesced(self, tensors, opts=None):
        return self.allreduce(tensors, opts)

    def broadcast(self, tensors, opts=None):
        tensors = list(tensors)
        self._validate(tensors)
        root = opts.rootRank if opts is not None else 0
        if not 0 <= root < self._size or (opts is not None and opts.rootTensor != 0):
            raise ValueError('Invalid ACCL broadcast root.')
        return self._submit(lambda runtime: [runtime.broadcast(t, root) for t in tensors], tensors)

    def allgather(self, output_tensors, input_tensors, opts=None):
        inputs = list(input_tensors)
        outputs = [list(tensors) for tensors in output_tensors]
        self._validate(inputs)
        if len(inputs) != len(outputs):
            raise ValueError('ACCL allgather input/output lists must match.')
        for tensor, destinations in zip(inputs, outputs):
            self._validate(destinations)
            if len(destinations) != self._size or any(
                t.shape != tensor.shape or t.dtype != tensor.dtype for t in destinations
            ):
                raise ValueError('ACCL allgather requires one same-shape output per rank.')
        return self._submit(
            lambda runtime: [runtime.allgather(out, src) for out, src in zip(outputs, inputs)],
            [tensor for group in outputs for tensor in group],
        )

    def _allgather_base(self, output, tensor, opts=None):
        self._validate([output, tensor])
        if output.numel() != tensor.numel() * self._size or output.dtype != tensor.dtype:
            raise ValueError('Invalid ACCL all_gather_into_tensor output.')
        outputs = list(output.reshape(self._size, *tensor.shape).unbind(0))
        return self.allgather([outputs], [tensor], opts)

    # PyTorch names this operation differently across supported releases.
    all_gather_single = _allgather_base

    def barrier(self, opts=None):
        return self._submit(lambda runtime: runtime.barrier(), [])

    def shutdown(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(None)
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=self._timeout_seconds)
            if self._thread.is_alive():
                raise TimeoutError('ACCL native request is still running; device buffers were not freed.')

    def abort(self):
        self._fail(RuntimeError('ACCL process group aborted.'))
        self.shutdown()


def _create_process_group(store, rank, world_size, timeout):
    global _active_group
    active = _active_group() if _active_group is not None else None
    if active is not None and not active._closed:
        raise NotImplementedError('Only one ACCL process group per process is supported.')
    path = os.environ.get('ACCL_CONFIG')
    if not path:
        raise RuntimeError('--backend accl requires ACCL_CONFIG=/path/to/config.json; see docs/accl.md.')
    config = ACCLConfig.load(path, world_size)
    # The first backend supports only WORLD, with one dedicated device per rank.
    # Reject unsupported subgroups instead of accidentally reprogramming a board.
    if rank != int(os.environ.get('RANK', rank)) or world_size != int(os.environ.get('WORLD_SIZE', world_size)):
        raise NotImplementedError('ACCL currently supports only the WORLD communicator.')
    group = ProcessGroupACCL(store, rank, world_size, timeout, config)
    _active_group = weakref.ref(group)
    return group


def register_accl_backend():
    if not dist.is_backend_available('accl'):
        dist.Backend.register_backend('accl', _create_process_group, devices=['cpu'])
