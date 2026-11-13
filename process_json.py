#!/usr/bin/env python3
"""Step 1: process private JSONs for all years or a selected year, then stop."""
import argparse
from workflow_utils import invoke
from json_processing import process

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--inputs-manifest', help='Private annual JSON manifest')
    source.add_argument('--input-json', help='One local annual Abstract Retrieval JSON')
    parser.add_argument('--years', type=int, nargs='+', help='Selected manifest years; omitted means all 2011-2025')
    parser.add_argument('--year', type=int, help='PUBYEAR for --input-json')
    parser.add_argument('--institutions-file', required=True, help='Private institution/AF-ID definitions')
    parser.add_argument('--output', required=True, help='New private output directory')
    invoke(parser, process)
