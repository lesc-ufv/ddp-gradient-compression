from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EpochMetrics:
    epoch: int
    hook: str
    world_size: int
    train_loss: float
    test_loss: float
    test_accuracy: float
    epoch_seconds: float
    dense_reference_mb_per_worker: float
    logical_payload_mb_per_worker: float | None
    logical_compression_ratio: float | None


class CsvLogger:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    def write(self, metrics: EpochMetrics) -> None:
        row = asdict(metrics)
        with self.output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            if not self._initialized and self.output_path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)
        self._initialized = True
