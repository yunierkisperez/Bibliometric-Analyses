#!/usr/bin/env python3
"""Step 2: calculate selected analysis families from private annual CSVs, then stop."""
import argparse
from product_catalog import ANALYSES, analysis_selection
from indicator_calculation import calculate
from workflow_utils import invoke


def run(args):
    selected = analysis_selection(args.analyses, args.for_tables, args.for_figures)
    calculate(args.derived_dir, args.mapping_file, args.output, selected, args.mapping_encoding, args.jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--derived-dir', required=True, help='Private annual CSV root, searched recursively')
    parser.add_argument('--mapping-file', required=True, help='Private display mapping')
    parser.add_argument('--mapping-encoding', default='latin1')
    parser.add_argument('--output', required=True, help='New private calculation directory')
    parser.add_argument('--jobs', type=int, choices=[1, 2, 3], default=2)
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument('--analyses', nargs='+', choices=list(ANALYSES), help='Explicitly selected analysis families')
    select.add_argument('--for-tables', nargs='+', help='Calculate only families needed for these table products')
    select.add_argument('--for-figures', nargs='+', help='Calculate only families needed for these figures and related tables')
    invoke(parser, run)
