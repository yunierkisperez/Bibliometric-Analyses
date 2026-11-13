#!/usr/bin/env python3
"""Check code-only distribution boundaries and optional publication metadata.

This is a defined file/contract check, not proof of licensing or anonymity.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
JSON_FILES = {
    'config/table_adapters.json', 'config/retained_csv.json', 'config/table_schemas.json',
    'config/palette.json', 'config/inputs.example.json',
    'config/institutions.example.json', 'designs/layout.json', 'assets/fonts/source_sans_3/font_manifest.json',
    'PUBLICATION_MANIFEST.json',
    'config/annual_outputs.json',
}
FORBIDDEN_EXTENSIONS = {'.xlsx', '.xls', '.tif', '.tiff', '.pdf', '.png', '.jpg', '.jpeg',
    '.zip', '.tar', '.gz', '.bz2', '.jsonl', '.ndjson', '.parquet', '.feather', '.pkl',
    '.pickle', '.otf', '.ttf', '.woff', '.woff2', '.key', '.pem'}
OPEN_FONT_FILES = {f'assets/fonts/source_sans_3/SourceSans3-{face}.ttf' for face in ['Regular', 'Semibold', 'Bold']}
SKIP_PARTS = {'.git', '.venv', '__pycache__', '.pytest_cache'}


def check(publication=False):
    errors = []
    files = []
    for path in ROOT.rglob('*'):
        rel = path.relative_to(ROOT)
        if set(rel.parts).intersection(SKIP_PARTS):
            continue
        if path.is_symlink():
            errors.append(f'Symlink is not part of this distribution: {rel}')
        elif path.is_file():
            name = rel.as_posix(); files.append(name)
            if path.suffix.lower() in FORBIDDEN_EXTENSIONS and name not in OPEN_FONT_FILES:
                errors.append(f'Excluded file format: {name}')
            if path.suffix.lower() == '.json' and name not in JSON_FILES:
                errors.append(f'Unlisted JSON file: {name}')
            if path.suffix.lower() == '.csv' and name != 'config/mapping.example.csv':
                errors.append(f'Only the fictitious mapping CSV belongs here: {name}')
            if path.name.startswith('.env'):
                errors.append(f'Environment/credential file: {name}')
            if set(rel.parts).intersection({'private', 'private_logs', 'annual', 'audit', 'outputs', 'data'}):
                errors.append(f'Private or generated-data directory: {name}')
    institutions = json.loads((ROOT / 'config/institutions.example.json').read_text())
    if not institutions or any(not str(r['afid']).startswith('990000') or
            not r['institution_id'].startswith('EXAMPLE_') or not r['name'].startswith('Fictitious ') for r in institutions):
        errors.append('Institution examples no longer match the fictitious-example convention')
    with (ROOT / 'config/mapping.example.csv').open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if row['institution_id'] != 'NATIONAL' and (not row['institution_id'].startswith('EXAMPLE_')
                    or not row['name'].startswith('Fictitious ')):
                errors.append('The mapping example contains a non-fictitious association')
    sys.path.insert(0, str(ROOT))
    from font_resources import verify_font_bundle
    try:
        verify_font_bundle()
    except (OSError, ValueError, KeyError) as error:
        errors.append(f'Open font bundle failed verification: {error}')
    if publication:
        if (ROOT / 'DO_NOT_PUBLISH_TEMPLATE.md').exists():
            errors.append('This is still the review template; run the external preparation tool')
        for required in ['LICENSE', 'CITATION.cff', 'ASSET_RIGHTS.md', 'PUBLICATION_MANIFEST.json', 'FILES_SHA256.txt', 'VERSION']:
            if not (ROOT / required).is_file():
                errors.append(f'Missing publication file: {required}')
        if (ROOT / 'PUBLICATION_MANIFEST.json').is_file():
            metadata = json.loads((ROOT / 'PUBLICATION_MANIFEST.json').read_text())
            if (metadata.get('font_family'), metadata.get('font_license'), metadata.get('font_bundle_verified')) != ('Source Sans 3', 'OFL-1.1', True):
                errors.append('Publication metadata does not identify the verified open font bundle')
            for field in ['code_license', 'copyright_holder', 'software_authors']:
                if not metadata.get(field):
                    errors.append(f'Unspecified publication metadata: {field}')
    if (ROOT / 'FILES_SHA256.txt').is_file():
        tracked = set()
        for line in (ROOT / 'FILES_SHA256.txt').read_text().splitlines():
            expected, name = line.split('  ', 1)
            path = ROOT / name; tracked.add(name)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(f'Manifest mismatch: {name}')
        if tracked != set(files).difference({'FILES_SHA256.txt'}):
            errors.append('Manifest and distribution file inventory differ')
    return {'pass': not errors, 'publication_mode': publication, 'files_checked': len(files),
            'errors': errors, 'scope': 'Defined distribution contracts; not licensing advice, empirical validation or a general anonymity guarantee'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication', action='store_true')
    result = check(parser.parse_args().publication)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['pass'] else 1)
