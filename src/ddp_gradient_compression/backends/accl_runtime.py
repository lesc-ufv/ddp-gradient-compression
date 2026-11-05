"""PyACCL adapter. All methods run on the process group's single I/O thread.

Targets Xilinx/pyaccl ff876197ac476f302ed485269b6926ae11df4b81 (0.2 API).
No Gloo, MPI or host-store transport is used for collective payloads here.
"""

import importlib
from pathlib import Path
import socket

import numpy as np
import torch

from .accl_config import ACCLConfig


DTYPES = {
    torch.float16: np.dtype('float16'), torch.float32: np.dtype('float32'),
    torch.float64: np.dtype('float64'), torch.int32: np.dtype('int32'),
    torch.int64: np.dtype('int64'),
}


class PyACCLRuntime:
    def __init__(self, config: ACCLConfig, rank: int, world_size: int):
        self.config, self.rank, self.world_size = config, rank, world_size
        self.buffers = {}
        try:
            import pyaccl
        except ImportError as exc:
            raise RuntimeError(
                "--backend accl requires PyACCL/PYNQ and a compatible ACCL design or emulator. "
                "See docs/accl.md. Gloo is not used as a fallback."
            ) from exc
        device = config.devices[rank]
        if config.mode == 'hardware':
            if device['host'] not in {socket.gethostname(), socket.getfqdn()}:
                raise ValueError(f"ACCL rank {rank} is assigned to host {device['host']!r}, "
                                 f"but runs on {socket.gethostname()!r}.")
            if not Path(device['xclbin']).is_file():
                raise FileNotFoundError(f"ACCL xclbin not found: {device['xclbin']}")
        if config.driver_factory:
            module, function = config.driver_factory.split(':', 1)
            self.driver = getattr(importlib.import_module(module), function)(config, rank, world_size)
        else:
            if config.mode == 'hardware':
                raise ValueError(
                    "Hardware requires driver_factory to load the overlay and configure its Ethernet stack. "
                    "An xclbin path alone does not configure the U55C network. See docs/accl.md."
                )
            self.driver = pyaccl.accl(
                nranks=world_size, local_rank=rank, ranks=config.ranks,
                protocol=config.protocol, nbufs=config.nbufs, bufsize=config.bufsize,
                sim_mode=True, sim_sock=device['sim_sock'],
            )
        self.sum_function = pyaccl.ACCLReduceFunctions.SUM
        self.driver.ignore_safety_checks = False
        # Hardware timeout units vary with the CCLO revision. The process group
        # enforces a host deadline; the board factory must configure HW timeout.

    def _buffer(self, slot, count, dtype):
        # Bound cached storage to input/output slots per dtype. Grow only when
        # needed; queued collectives never use these buffers concurrently.
        key = (slot, dtype.str)
        buffer = self.buffers.get(key)
        if buffer is None or buffer.size < count:
            if buffer is not None and hasattr(buffer, 'freebuffer'):
                buffer.freebuffer()
            buffer = self.driver.allocate((count,), dtype=dtype)
            self.buffers[key] = buffer
        return buffer[:count]

    def _complete(self, handle, output, operation):
        if handle is None:
            # PyACCL can decline a collective (e.g. insufficient RX buffers).
            raise RuntimeError(f"PyACCL did not launch {operation}; check RX buffers and configuration.")
        handle.wait()
        self.driver.check_return_value(operation)
        output.sync_from_device()

    def _chunks(self, tensor):
        # Keep every chunk within one RX segment, and pad DMA to a 64-byte word.
        # Padding is internal, excluded from the hooks' logical payload counters.
        dtype = DTYPES[tensor.dtype]
        capacity = self.config.bufsize // dtype.itemsize
        word = 64 // dtype.itemsize
        flat = tensor.detach().reshape(-1).numpy()
        for start in range(0, flat.size, capacity):
            length = min(capacity, flat.size - start)
            count = ((length + word - 1) // word) * word
            source = self._buffer('input', count, dtype)
            source.fill(0)
            # Raw byte copy preserves NaN payloads / INT32 index bit patterns.
            source[:length].view(np.uint8)[:] = flat[start:start + length].view(np.uint8)
            source.sync_to_device()
            yield start, length, count, dtype, source

    @staticmethod
    def _copy_to_tensor(destination, source):
        array = destination.detach().numpy()
        array.view(np.uint8).reshape(-1)[:] = np.asarray(source).view(np.uint8).reshape(-1)

    def allreduce(self, tensor):
        target = tensor.reshape(-1)
        for start, length, count, dtype, source in self._chunks(tensor):
            output = self._buffer('output', count, dtype)
            handle = self.driver.allreduce(source, output, count, self.sum_function,
                                          from_fpga=True, to_fpga=True,
                                          compress_dtype=None, run_async=True)
            self._complete(handle, output, 'allreduce')
            self._copy_to_tensor(target[start:start + length], output[:length])

    def broadcast(self, tensor, root):
        target = tensor.reshape(-1)
        for start, length, count, _, source in self._chunks(tensor):
            handle = self.driver.bcast(source, count, root, from_fpga=True, to_fpga=True,
                                      compress_dtype=None, run_async=True)
            self._complete(handle, source, 'broadcast')
            self._copy_to_tensor(target[start:start + length], source[:length])

    def allgather(self, outputs, tensor):
        targets = [output.reshape(-1) for output in outputs]
        for start, length, count, dtype, source in self._chunks(tensor):
            output = self._buffer('output', count * self.world_size, dtype)
            handle = self.driver.allgather(source, output, count, from_fpga=True, to_fpga=True,
                                          compress_dtype=None, run_async=True)
            self._complete(handle, output, 'allgather')
            for rank, target in enumerate(targets):
                self._copy_to_tensor(target[start:start + length], output[rank * count:rank * count + length])

    def barrier(self):
        self.driver.barrier()
        self.driver.check_return_value('barrier')

    def close(self):
        for buffer in self.buffers.values():
            if hasattr(buffer, 'freebuffer'):
                buffer.freebuffer()
        self.buffers.clear()
        # Site factories may supply additional teardown; do not reset/reprogram
        # a board implicitly, especially after a timed-out operation.
        close = getattr(self.driver, 'close', None)
        if close is not None:
            close()
        cclo = getattr(self.driver, 'cclo', None)
        connection = getattr(cclo, 'socket', None)
        if connection is not None:
            connection.close(linger=0)
