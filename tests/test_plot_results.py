import csv
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


script = Path(__file__).resolve().parents[1] / 'scripts' / 'plot_results.py'
module = runpy.run_path(str(script))
read_run = module['read_run']
select_runs = module['select_runs']


def write_csv(path, epochs=(1, 2, 3), payloads=(10, 2, 2), hook='powersgd', world=4):
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow((*module['REQUIRED'], *module['OPTIONAL']))
        for epoch, payload in zip(epochs, payloads):
            writer.writerow((epoch, hook, world, 1 / epoch, 2 / epoch,
                             60 + epoch, 1.5, payload, '' if payload == '' else 10 / payload))
    return path


def test_payload_accumulates_in_epoch_order(tmp_path):
    path = write_csv(tmp_path / 'run.csv', epochs=(3, 1, 2), payloads=(2, 10, 2))
    run = read_run(path)
    assert run.values['epoch'] == [1, 2, 3]
    assert run.cumulative_payload() == [10, 12, 14]


@pytest.mark.parametrize('epochs,payloads', [((1, 2, 3), (10, '', 2)), ((1, 3), (10, 2))])
def test_unknown_payload_or_missing_epoch_has_no_cumulative_curve(tmp_path, epochs, payloads):
    run = read_run(write_csv(tmp_path / 'run.csv', epochs, payloads))
    assert run.cumulative_payload() is None


def test_latest_selection_keeps_hooks_and_world_sizes_separate(tmp_path):
    old = write_csv(tmp_path / '20260101-powersgd-world4.csv')
    new = write_csv(tmp_path / '20260102-powersgd-world4.csv')
    other_world = write_csv(tmp_path / '20260103-powersgd-world2.csv', world=2)
    other_hook = write_csv(tmp_path / '20260101-dense-world4.csv', hook='dense')
    empty = tmp_path / '20260104-empty.csv'
    empty.write_text('')
    paths = [new, empty, other_world, old, other_hook]
    assert {r.path for r in select_runs(paths, None, False)} == {new, other_world, other_hook}
    assert {r.path for r in select_runs(paths, 4, False)} == {new, other_hook}
    assert {r.path for r in select_runs(paths, 4, True)} == {old, new, other_hook}


def test_duplicate_epoch_rejected(tmp_path):
    path = write_csv(tmp_path / 'run.csv', epochs=(1, 1), payloads=(10, 2))
    with pytest.raises(ValueError, match='únicas'):
        read_run(path)


def test_cli_exports_six_graphs_and_manifest(tmp_path):
    pytest.importorskip('matplotlib')
    path = write_csv(tmp_path / 'run.csv')
    output = tmp_path / 'plots'
    completed = subprocess.run([sys.executable, str(script), str(path), '--output-dir', str(output),
                                '--dpi', '60'], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    images = list(output.glob('*.png'))
    assert len(images) == 6
    for image in images:
        assert image.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert str(path) in (output / 'selected_runs.json').read_text()


def test_cli_defaults_work_from_scripts_directory(tmp_path):
    pytest.importorskip('matplotlib')
    scripts_dir = tmp_path / 'scripts'
    scripts_dir.mkdir()
    copied_script = scripts_dir / 'plot_results.py'
    copied_script.write_text(script.read_text())
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    selected = write_csv(results_dir / 'world4.csv', world=4)
    write_csv(results_dir / 'world2.csv', world=2)
    completed = subprocess.run(
        [sys.executable, 'plot_results.py', '--world-size', '4', '--dpi', '60'],
        cwd=scripts_dir, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    output = results_dir / 'plots'
    assert len(list(output.glob('*.png'))) == 6
    manifest = (output / 'selected_runs.json').read_text()
    assert str(selected) in manifest
    assert 'world2.csv' not in manifest
    assert not (scripts_dir / 'results').exists()


def test_latest_selection_separates_communication_backends(tmp_path):
    paths = []
    for timestamp, backend, mode in ((1, 'gloo', ''), (2, 'accl', 'hardware'), (3, 'accl', 'emulator')):
        path = write_csv(tmp_path / f'2026010{timestamp}-dense-world4.csv', hook='dense')
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row.update(backend=backend, backend_mode=mode)
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    runs = select_runs(paths, 4, False)
    assert {(run.backend, run.backend_mode) for run in runs} == {
        ('gloo', ''), ('accl', 'hardware'), ('accl', 'emulator'),
    }
