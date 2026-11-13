"""Run one figure and its associated table products as an independent workflow.

Entry points cover Figures 1-7 and S1-S10 individually. Shared
orchestration avoids duplicating scientific calculations across entry points.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from product_catalog import FIGURES, analysis_selection, figure_products
from workflow_utils import digest, invoke, new_directory


def reproduce(figure, args):
    from table_export import build
    if args.tables_dir and (args.only_tables or args.minimal_tables):
        raise ValueError('--tables-dir renders existing tables; --only-tables and --minimal-tables apply to annual or calculated inputs')
    figures_only = bool(args.tables_dir) or getattr(args, 'figures_only', False)
    figures, products = figure_products(figure, args.only_tables, args.minimal_tables or figures_only)
    analyses = analysis_selection(tables=products)
    if args.tables_dir:
        if args.mapping_file or args.work_dir:
            raise ValueError('--mapping-file and --work-dir apply only to --derived-dir')
        source = Path(args.tables_dir).expanduser().resolve()
        mode = 'anonymous_tables'
    elif args.calculated_dir:
        if args.mapping_file or args.work_dir:
            raise ValueError('--mapping-file and --work-dir apply only to --derived-dir')
        source = Path(args.calculated_dir).expanduser().resolve()
        mode = 'calculated_csvs'
    else:
        if not args.mapping_file:
            raise ValueError('--derived-dir requires --mapping-file with the private associations')
        source = Path(args.derived_dir).expanduser().resolve()
        mode = 'private_annual_csvs'
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        raise ValueError(f'Output directory already exists: {output_path}')
    if figures:
        from runtime_check import check
        check(figures=True)
    private_temp = None
    table_temp = None
    try:
        if mode == 'private_annual_csvs':
            from indicator_calculation import calculate
            if args.work_dir:
                work = Path(args.work_dir).expanduser().resolve()
                if work == output_path or work.is_relative_to(output_path) or output_path.is_relative_to(work):
                    raise ValueError('Private --work-dir and anonymous --output must be separate trees')
            else:
                private_temp = Path(tempfile.mkdtemp(prefix='repro_private_calculation_'))
                work = private_temp / 'calculated'
            try:
                calculate(source, args.mapping_file, work, analyses, args.mapping_encoding, args.jobs)
            except Exception:
                if private_temp is not None:
                    print(f'Private diagnostic files retained in: {private_temp}', flush=True)
                    private_temp = None
                raise
            source = work
        output = new_directory(output_path)
        table_output = output / 'tables'
        if mode == 'anonymous_tables':
            paths = [source / f'Table-{key}.{args.source_format}' for key in products]
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(f'Missing required input for {figure}: {path.name}')
            provenance = {'input_sha256': {p.name: digest(p) for p in paths},
                          'table_source': 'Supplied anonymous tables', 'table_copies_written': False,
                          'inputs_unchanged': True}
            table_output = source
            source_format = args.source_format
        else:
            if figures_only:
                table_temp = tempfile.TemporaryDirectory(prefix='repro_figure_construction_')
                table_output = Path(table_temp.name) / 'tables'
            manifest = build(source, table_output,
                             tables=[int(t[1:]) for t in products if t.startswith('S') and t[1:].isdigit()],
                             include_main='Main1' in products)
            provenance = {'table_export': manifest, 'reference_workbook_values_used': False,
                          'construction_tables_temporary': figures_only,
                          'table_deliverables_written': not figures_only}
            source_format = 'xlsx'
        if figures:
            from figure_rendering import render
            figure_output = output / 'figures'; figure_output.mkdir()
            render(table_output, figure_output, source_format, figures=figures)
        report = {
            'figure_entry_point': figure, 'mode': mode, 'selected_figures': figures,
            'selected_table_inputs': products, 'selected_table_products': [] if figures_only else products,
            'analysis_families': analyses if mode == 'private_annual_csvs' else [],
            'table_provenance': provenance,
            'output_sha256': {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
            'private_inputs_copied_to_result': False,
            'historical_comparison_performed': False,
            'notice': 'Use validate_tables.py against historical Excel references; execution alone is not empirical equivalence.'}
        (output / 'reproduction_report.json').write_text(json.dumps(report, indent=2))
        print(f'{figure}: {len(figures)} figure(s), table products {", ".join([] if figures_only else products) or "none"}')
        print(f'Anonymous results: {output}')
    finally:
        if table_temp is not None:
            table_temp.cleanup()
        if private_temp is not None:
            shutil.rmtree(private_temp)


def main(figure):
    tables = FIGURES[figure][2]
    required = FIGURES[figure][1]
    xlsx = ', '.join(f'Table-{key}.xlsx' for key in required)
    csv = ', '.join(f'Table-{key}.csv' for key in required)
    entry = 'reproduce_figure_' + (figure.replace('Fig-S', 's') if figure.startswith('Fig-S') else figure.replace('Fig', '')) + '.py'
    help_text = (
        f'TABLE INPUTS FOR {figure}\n'
        f'  Required Excel files: {xlsx}\n'
        f'  Or exact-schema CSV files: {csv}\n'
        '  Download only these supplementary tables. No other table is read.\n'
        '  --tables-dir writes this figure and reports, without copying tables.\n\n'
        f'  Example: python {entry} --tables-dir /path/to/tables --output /path/to/new/result\n\n'
        'PRIVATE JSON ROUTE\n'
        '  Process JSONs first with process_json.py; then use --derived-dir or\n'
        '  --calculated-dir. No published supplementary table is required.\n'
        f'  Default calculated table products: {", ".join(tables)}.\n'
        '  --figures-only: figure and reports, using temporary construction tables.\n'
        '  --only-tables: selected associated tables, with no figure rendering.\n'
        '  Omit both flags: this figure and its associated tables.\n'
        '  See USAGE.md for complete, separate steps and all output choices.'
    )
    parser = argparse.ArgumentParser(
        description=f'Reproduce {figure} independently.',
        epilog=help_text, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--derived-dir', help='Private annual CSVs; calculate this figure and export its associated tables')
    source.add_argument('--calculated-dir', help='Previously calculated CSVs; export this figure without recalculating')
    source.add_argument('--tables-dir', help=f'Article tables: {xlsx}; render this figure only, with no private inputs')
    select = parser.add_mutually_exclusive_group()
    select.add_argument('--minimal-tables', action='store_true', help='Include only direct construction tables, omitting additional associated tables')
    select.add_argument('--only-tables', nargs='+', help='Selected associated table products; do not generate the figure')
    select.add_argument('--figures-only', action='store_true', help='Write only this figure and its reports; calculate temporary construction tables when using annual or calculated inputs')
    parser.add_argument('--source-format', choices=['xlsx', 'csv'], default='xlsx', help='Format read by --tables-dir')
    parser.add_argument('--mapping-file', help='Private display mapping, required only with --derived-dir')
    parser.add_argument('--mapping-encoding', default='latin1')
    parser.add_argument('--work-dir', help='Optional new private calculation directory outside the anonymous output tree')
    parser.add_argument('--jobs', type=int, choices=[1, 2, 3], default=2)
    parser.add_argument('--output', required=True, help='New anonymous result directory for this figure or table subset')
    invoke(parser, lambda args: reproduce(figure, args))
