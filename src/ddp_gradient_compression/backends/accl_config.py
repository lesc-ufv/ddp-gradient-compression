"""Explicit rank-to-device mapping for the ACCL WORLD communicator."""

from dataclasses import dataclass
import ipaddress
import json
from pathlib import Path


@dataclass(frozen=True)
class ACCLConfig:
    mode: str
    ranks: list[dict]
    devices: list[dict]
    protocol: str = "TCP"
    nbufs: int = 16
    bufsize: int = 4096
    timeout_s: float = 120.0
    driver_factory: str | None = None

    @classmethod
    def load(cls, path: str, world_size: int) -> "ACCLConfig":
        source = Path(path).resolve()
        with source.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        config = cls(**raw)
        if config.mode not in ("hardware", "emulator"):
            raise ValueError("ACCL mode must be 'hardware' or 'emulator'; no mock/Gloo fallback.")
        if config.protocol not in ("TCP", "UDP", "AXIS"):
            raise ValueError("PyACCL supports TCP, UDP or AXIS in this integration.")
        if len(config.ranks) != world_size or len(config.devices) != world_size:
            raise ValueError("ACCL ranks/devices must have exactly WORLD_SIZE entries, in global rank order.")
        if not 0 < config.timeout_s < 86400:
            raise ValueError("ACCL timeout_s must be positive and less than one day.")
        if config.nbufs < world_size or config.bufsize < 64 or config.bufsize % 64:
            raise ValueError("ACCL requires nbufs >= WORLD_SIZE and bufsize a positive multiple of 64.")
        if config.driver_factory is not None and ':' not in config.driver_factory:
            raise ValueError("driver_factory must be module:function.")
        endpoints = set()
        hardware_devices = set()
        for rank, (peer, device) in enumerate(zip(config.ranks, config.devices)):
            ipaddress.IPv4Address(peer['ip'])
            if not 0 < peer['port'] < 65536 or peer['session_id'] < 0:
                raise ValueError(f"Invalid ACCL network descriptor for rank {rank}.")
            if peer['max_segment_size'] != config.bufsize:
                raise ValueError("All max_segment_size fields must equal bufsize.")
            endpoint = (peer['ip'], peer['port'])
            if endpoint in endpoints:
                raise ValueError("ACCL network endpoints must be unique.")
            endpoints.add(endpoint)
            if config.mode == "hardware":
                # PyACCL loads an overlay per process: do not allow two ranks
                # to reload/share the same physical board in this first backend.
                key = (device['host'], device['board_idx'])
                if key in hardware_devices or device['board_idx'] < 0 or device.get('cclo_idx', 0) < 0:
                    raise ValueError("Each hardware rank needs a distinct host/board_idx and nonnegative indices.")
                hardware_devices.add(key)
                bitstream = Path(device['xclbin'])
                if not bitstream.is_absolute():
                    device['xclbin'] = str(source.parent / bitstream)
            elif not device.get('sim_sock', '').startswith('tcp://'):
                raise ValueError("Each emulator rank requires an explicit tcp:// sim_sock.")
        return config
