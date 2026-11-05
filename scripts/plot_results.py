#!/usr/bin/env python3
"""Compare experiment CSVs without importing PyTorch or requiring a display."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


REQUIRED = (
    "epoch", "hook", "world_size", "train_loss", "test_loss",
    "test_accuracy", "epoch_seconds",
)
OPTIONAL = ("logical_payload_mb_per_worker", "logical_compression_ratio")


@dataclass
class Run:
    path: Path
    hook: str
    world_size: int
    values: dict[str, list[float]]
    backend: str = "gloo"
    backend_mode: str = ""

    @property
    def label(self) -> str:
        return self.path.stem

    def cumulative_payload(self) -> list[float] | None:
        epochs = self.values["epoch"]
        payload = self.values["logical_payload_mb_per_worker"]
        # Never treat unknown bytes or missing epochs as zero traffic.
        if epochs != list(range(1, len(epochs) + 1)) or not all(map(math.isfinite, payload)):
            return None
        total = 0.0
        accumulated = []
        for value in payload:
            total += value
            accumulated.append(total)
        return accumulated


def read_run(path: Path) -> Run:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(REQUIRED).issubset(reader.fieldnames or []):
            raise ValueError("colunas obrigatórias ausentes")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV sem épocas registradas")
    hooks = {row["hook"] for row in rows}
    worlds = {int(row["world_size"]) for row in rows}
    if len(hooks) != 1 or len(worlds) != 1 or min(worlds) < 1:
        raise ValueError("hook/world_size inconsistentes")
    rows.sort(key=lambda row: float(row["epoch"]))
    values = {}
    for name in ("epoch", "train_loss", "test_loss", "test_accuracy", "epoch_seconds", *OPTIONAL):
        series = []
        for row in rows:
            raw = row.get(name, "")
            value = float(raw) if raw and raw.strip() else math.nan
            if name in OPTIONAL and math.isnan(value):
                series.append(math.nan)
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"valor inválido em {name}: {raw!r}")
            series.append(value)
        values[name] = series
    epochs = values["epoch"]
    if any(epoch < 1 or not epoch.is_integer() for epoch in epochs) or len(set(epochs)) != len(epochs):
        raise ValueError("épocas devem ser inteiras, positivas e únicas")
    backends = {row.get("backend") or "gloo" for row in rows}
    modes = {row.get("backend_mode") or "" for row in rows}
    if len(backends) != 1 or len(modes) != 1:
        raise ValueError("backend/backend_mode inconsistentes")
    return Run(path, hooks.pop(), worlds.pop(), values, backends.pop(), modes.pop())


def select_runs(paths: list[Path], world_size: int | None, all_runs: bool) -> list[Run]:
    runs = []
    for path in sorted(set(paths)):
        try:
            run = read_run(path)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"Ignorando {path}: {exc}", file=sys.stderr)
            continue
        if world_size is None or run.world_size == world_size:
            runs.append(run)
    if not all_runs:
        # Training filenames begin with YYYYMMDD-HHMMSS. Keep the newest valid
        # file for each hook/world size, not a concatenation of separate runs.
        latest = {}
        for run in sorted(runs, key=lambda run: run.path.name):
            latest[run.hook, run.backend, run.backend_mode, run.world_size] = run
        runs = list(latest.values())
    return sorted(runs, key=lambda run: (run.hook, run.world_size, run.path.name))


def plot_runs(runs: list[Run], output_dir: Path, file_format: str, dpi: int) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10")
    line_styles = ("-", "--", "-.", ":")
    markers = ("o", "s", "^", "D")
    specs = (
        ("loss", "Loss por época", "Loss", ("train_loss", "test_loss")),
        ("accuracy", "Accuracy de teste por época", "Accuracy (%)", ("test_accuracy",)),
        ("epoch_time", "Tempo por época", "Tempo (s)", ("epoch_seconds",)),
        ("logical_payload", "Payload lógico por época", "Payload lógico (MiB / worker)", (OPTIONAL[0],)),
        ("compression_ratio", "Razão de compressão lógica por época", "Referência densa / payload (×)", (OPTIONAL[1],)),
        ("accuracy_vs_payload", "Accuracy versus payload lógico acumulado", "Accuracy (%)", ("test_accuracy",)),
    )
    outputs = []
    for filename, title, ylabel, metrics in specs:
        fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
        cumulative = filename == "accuracy_vs_payload"
        plotted = False
        for index, run in enumerate(runs):
            x = run.cumulative_payload() if cumulative else run.values["epoch"]
            if x is None:
                print(f"{run.path.name}: payload acumulado indisponível (lacunas ou bytes desconhecidos).",
                      file=sys.stderr)
                continue
            for metric in metrics:
                y = run.values[metric]
                if not any(map(math.isfinite, y)):
                    continue
                label = run.label
                if filename == "loss":
                    label += " · treino" if metric == "train_loss" else " · teste"
                # Distinct dashes and hollow markers keep almost identical curves
                # visible (e.g. dense and FP16 accuracy), without shifting data.
                ax.plot(x, y, label=label, color=colors(index % 10),
                        linestyle="--" if metric == "train_loss" else line_styles[index % 4],
                        marker=markers[index % 4], markersize=7 if index % 4 == 0 else 5,
                        markerfacecolor="none", markeredgewidth=1.2,
                        linewidth=2.4 if index % 4 == 0 else 1.6)
                plotted = True
        ax.set(title=title, ylabel=ylabel,
               xlabel="Payload lógico acumulado (MiB / worker)" if cumulative else "Época")
        ax.grid(True, alpha=0.25)
        if not cumulative:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if plotted:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        else:
            ax.text(0.5, 0.5, "Sem dados disponíveis", transform=ax.transAxes, ha="center")
        if filename in ("logical_payload", "compression_ratio", "accuracy_vs_payload"):
            fig.supxlabel("Payload lógico não representa tráfego físico nem ganho de velocidade.", fontsize=9)
        destination = output_dir / f"{filename}.{file_format}"
        fig.savefig(destination, dpi=dpi)
        plt.close(fig)
        outputs.append(destination)
    manifest = output_dir / "selected_runs.json"
    manifest.write_text(json.dumps([
        {"file": str(run.path.resolve()), "hook": run.hook, "world_size": run.world_size,
         "epochs": run.values["epoch"], "backend": run.backend, "backend_mode": run.backend_mode} for run in runs
    ], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_files", nargs="*", type=Path,
                        help="CSVs explícitos; todos os arquivos informados são comparados.")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results",
                        help="Pasta de entrada; padrão: results/ na raiz do projeto.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "plots",
                        help="Pasta de saída; padrão: results/plots/ na raiz do projeto.")
    parser.add_argument("--world-size", type=int, help="Filtrar por número de processos.")
    parser.add_argument("--all-runs", action="store_true",
                        help="Comparar todas as execuções; padrão: mais recente por hook/backend/modo/world_size.")
    parser.add_argument("--format", choices=("png", "svg", "pdf"), default="png")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    if args.dpi < 1 or (args.world_size is not None and args.world_size < 1):
        parser.error("--dpi e --world-size devem ser positivos")
    paths = args.csv_files or list(args.results_dir.glob("*.csv"))
    runs = select_runs(paths, args.world_size, args.all_runs or bool(args.csv_files))
    if not runs:
        parser.error(
            "Nenhuma execução válida encontrada para os filtros informados. "
            + ("Confira os CSVs explícitos." if args.csv_files
               else f"Pasta consultada: {args.results_dir.resolve()}")
        )
    print("Execuções selecionadas:")
    for run in runs:
        print(f"  {run.path} ({len(run.values['epoch'])} épocas)")
    print("Os CSVs não registram dataset/hiperparâmetros completos; confira se as execuções são comparáveis.")
    try:
        outputs = plot_runs(runs, args.output_dir, args.format, args.dpi)
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        parser.error('Instale a dependência: python -m pip install -e ".[plots]"')
    for path in outputs:
        print(f"Gráfico: {path}")
    print(f"Seleção registrada em {args.output_dir / 'selected_runs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
