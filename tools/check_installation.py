#!/usr/bin/env python3
"""Check installed Python libraries, native Cairo and the bundled open fonts."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_check import check

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables-only', action='store_true', help='Skip PDF/SVG dependencies')
    args = parser.parse_args()
    try:
        print(json.dumps({'pass': True, **check(figures=not args.tables_only)}, indent=2))
    except RuntimeError as error:
        parser.exit(1, str(error) + '\n')
