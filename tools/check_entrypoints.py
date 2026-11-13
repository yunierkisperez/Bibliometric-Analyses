#!/usr/bin/env python3
"""Check independent command help without loading research inputs."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from product_catalog import FIGURES


def main():
    scripts = sorted(ROOT.glob('reproduce_figure_*.py'))
    if len(scripts) != 17:
        raise ValueError('Expected 17 independent figure entry points')
    scripts += [ROOT / x for x in ['process_json.py', 'calculate_indicators.py', 'export_tables.py', 'validate_tables.py']]
    for script in scripts:
        result = subprocess.run([sys.executable, '-B', script, '--help'], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f'Command help failed: {script.name}\n{result.stdout}{result.stderr}')
        if script.name.startswith('reproduce_figure_'):
            suffix = script.stem.removeprefix('reproduce_figure_')
            key = 'Fig-S' + suffix[1:] if suffix.startswith('s') else 'Fig' + suffix
            expected = FIGURES[key][1]
            for product in expected:
                for extension in ['xlsx', 'csv']:
                    if f'Table-{product}.{extension}' not in result.stdout:
                        raise RuntimeError(f'{script.name} help omits required {product} {extension} input')
            if 'No other table is read.' not in result.stdout or 'PRIVATE JSON ROUTE' not in result.stdout:
                raise RuntimeError(f'{script.name} help does not distinguish the two routes')
    print(f'Command-help checks passed: {len(scripts)}. No research inputs were read.')


if __name__ == '__main__':
    main()
