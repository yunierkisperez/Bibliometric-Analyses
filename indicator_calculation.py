"""Calculate selected scientific families without exporting Excel or figures."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from product_catalog import ANALYSES, YEARS, analysis_selection
from workflow_utils import SRC, digest, new_directory, run_command


def annual_inputs(folder, analyses):
    folder = Path(folder).expanduser().resolve()
    suffixes = sorted({suffix for name in analyses for suffix in ANALYSES[name][2]})
    result = {}
    for year in YEARS:
        for suffix in suffixes:
            filename = f'{year}_{suffix}.csv'
            found = list(folder.rglob(filename))
            if len(found) != 1:
                raise ValueError(f'Missing or duplicate input: {filename}')
            result[filename] = found[0]
    return result


def calculate(derived_dir, mapping_file, output, analyses, mapping_encoding='latin1', jobs=2):
    analyses = analysis_selection(analyses=analyses)
    files = annual_inputs(derived_dir, analyses)
    mapping = Path(mapping_file).expanduser().resolve()
    if not mapping.is_file():
        raise FileNotFoundError('A local private display mapping is required')
    before = {name: digest(path) for name, path in files.items()}
    mapping_before = digest(mapping)
    output = new_directory(output)
    logs = output / 'private_logs'; logs.mkdir()
    methodology = {}
    env = dict(os.environ, PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', REPRO_DATA_ONLY='1', MPLBACKEND='Agg')
    with tempfile.TemporaryDirectory(prefix='repro_calculation_inputs_') as temp:
        flat = Path(temp)
        for name, path in files.items():
            try:
                (flat / name).symlink_to(path)
            except OSError:
                shutil.copyfile(path, flat / name)
        def execute(name):
            number, script, _ = ANALYSES[name]
            dest = output / ('supp' if name == 'temporal' else str(number))
            dest.mkdir()
            command = [sys.executable, SRC / script, '--input-dir', flat, '--mapping-file', mapping,
                       '--mapping-encoding', mapping_encoding, '--output-dir', dest]
            if name == 'temporal':
                command += ['--main-signals-file', output / '2/figure2_robust_d_temporal_exposure_profiles_annual_robust_deviations.csv']
            run_command(command, logs / f'{name}.log', env)
            marker = 'METHOD_DIAGNOSTICS_JSON '
            entries = [line[len(marker):] for line in (logs / f'{name}.log').read_text().splitlines()
                       if line.startswith(marker)]
            if len(entries) > 1:
                raise ValueError(f'Multiple methodology diagnostic records in {name}')
            if entries:
                methodology[name] = json.loads(entries[0])
            print(f'Calculated {name}', flush=True)
        independent = [name for name in analyses if name != 'temporal']
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(execute, independent))
        if 'temporal' in analyses:
            execute('temporal')
    if before != {name: digest(path) for name, path in files.items()} or mapping_before != digest(mapping):
        raise ValueError('A private input changed during calculation')
    report = {'analyses': analyses, 'years': list(YEARS), 'inputs_unchanged': True,
              'methodology_diagnostics': methodology,
              'input_sha256': before, 'mapping_sha256': mapping_before,
              'calculated_csv_sha256': {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*.csv'))},
              'excel_export_performed': False, 'figure_generation_performed': False,
              'historical_comparison_performed': False, 'distribution': 'Private calculation files and logs'}
    (output / 'calculation_report.json').write_text(json.dumps(report, indent=2))
    return report
