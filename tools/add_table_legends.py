#!/usr/bin/env python3
"""Add the text-only Legend sheet to existing supplementary Excel tables.

Reads Table-S<n>.xlsx from one folder and writes copies with a second sheet
named in config/table_legends.json to a new folder. The data sheet keeps its
exact bytes, so formulas and cached results are unchanged. Inputs are never
modified, and a workbook that already has the legend sheet is refused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def data_parts(path):
    """Digest of every worksheet part, to show that the data sheet was not rewritten."""
    with ZipFile(path) as package:
        return {name: hashlib.sha256(package.read(name)).hexdigest()
                for name in package.namelist() if name.startswith('xl/worksheets/')}


def run(tables_dir, output_dir, tables=None):
    from table_legends import add_legend
    source = Path(tables_dir).expanduser().resolve(); target = Path(output_dir).expanduser().resolve()
    if source == target:
        raise ValueError('Write the tables with legend to a different folder')
    selected = list(range(1, 16)) if not tables else [int(str(x).upper().lstrip('S')) for x in tables]
    if len(set(selected)) != len(selected) or any(n not in range(1, 16) for n in selected):
        raise ValueError('Select distinct supplementary tables within S1-S15')
    for n in selected:
        if not (source / f'Table-S{n}.xlsx').is_file():
            raise ValueError(f'Missing workbook: Table-S{n}.xlsx')
        if (target / f'Table-S{n}.xlsx').exists():
            raise ValueError(f'Output already exists: {target / f"Table-S{n}.xlsx"}')
    report = []
    for n in selected:
        before = data_parts(source / f'Table-S{n}.xlsx')
        result = add_legend(source / f'Table-S{n}.xlsx', target / f'Table-S{n}.xlsx', n)
        after = data_parts(target / f'Table-S{n}.xlsx')
        result['data_sheet_unchanged'] = all(after.get(name) == digest for name, digest in before.items())
        if not result['data_sheet_unchanged'] or len(after) != len(before) + 1:
            (target / f'Table-S{n}.xlsx').unlink()
            raise RuntimeError(f'Table S{n}: the data sheet changed; output is not valid')
        report.append(result)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables-dir', required=True, help='Folder with Table-S<n>.xlsx without legend sheet')
    parser.add_argument('--output-dir', required=True, help='Different folder for the copies with legend sheet')
    parser.add_argument('--tables', nargs='+', help='Default: all 15; examples: S1 S13 S14')
    arguments = parser.parse_args()
    try:
        outcome = run(arguments.tables_dir, arguments.output_dir, arguments.tables)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise SystemExit(f'Error: {error}')
    print(json.dumps(outcome, indent=2))
