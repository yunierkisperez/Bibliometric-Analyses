#!/usr/bin/env python3
"""Step 3: export selected anonymous tables from calculated CSVs, then stop."""
import argparse
import json
from pathlib import Path
from product_catalog import table_selection
from workflow_utils import digest, invoke, new_directory


def run(args):
    from table_export import build
    selected = table_selection(args.tables)
    generated = Path(args.calculated_dir).expanduser().resolve()
    before = {str(p.relative_to(generated)): digest(p) for p in sorted(generated.rglob('*.csv'))}
    output = new_directory(args.output)
    result = build(generated, output, tables=[int(x[1:]) for x in selected if x.startswith('S') and x[1:].isdigit()],
                   include_main='Main1' in selected, legend=not args.no_legend)
    if before != {str(p.relative_to(generated)): digest(p) for p in sorted(generated.rglob('*.csv'))}:
        raise ValueError('A calculated CSV changed during export')
    (output / 'table_build_report.json').write_text(json.dumps({
        'selected_products': selected, 'tables': result, 'inputs_unchanged': True, 'calculated_csv_sha256': before,
        'legend_sheet_included': not args.no_legend,
        'reference_values_used_for_calculation': False, 'historical_comparison_performed': False}, indent=2))
    print(f'Anonymous tables: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calculated-dir', required=True, help='Step 2 outputs, searched recursively')
    parser.add_argument('--tables', nargs='+', required=True, help='Explicit products; examples: S1 S2 S13 S14, Main1')
    parser.add_argument('--output', required=True, help='New anonymous table directory')
    parser.add_argument('--no-legend', action='store_true', help='Write workbooks without the text-only Legend sheet')
    invoke(parser, run)
