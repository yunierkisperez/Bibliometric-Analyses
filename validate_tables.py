#!/usr/bin/env python3
"""Step 5: independently compare selected supplementary Excel tables, then stop."""
import argparse
import json
from pathlib import Path
from product_catalog import table_selection
from workflow_utils import invoke


def run(args):
    from table_validation import validate_tables
    selected = table_selection(args.tables, extras=False)
    target = Path(args.report).expanduser().resolve()
    if target.exists():
        raise ValueError(f'Report already exists: {target}')
    report = validate_tables(Path(args.reference_dir), Path(args.tables_dir), [int(x[1:]) for x in selected])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Tables passing: {sum(x["pass"] for x in report)}/{len(report)}')
    if not all(x['pass'] for x in report):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables-dir', required=True)
    parser.add_argument('--reference-dir', required=True)
    parser.add_argument('--tables', nargs='+', help='Default: all 15; examples: S1 S13 S14')
    parser.add_argument('--report', required=True, help='New JSON comparison report')
    invoke(parser, run)
