"""Run only private annual JSON processing, without table or figure generation."""
import json
from pathlib import Path
import sys
from product_catalog import YEARS
from workflow_utils import SRC, digest, new_directory


def json_jobs(manifest, years=None):
    def unique(pairs):
        result = {}
        for key, val in pairs:
            if key in result:
                raise ValueError('Duplicate year in JSON manifest')
            result[key] = val
        return result
    manifest = Path(manifest).expanduser().resolve()
    obj = json.loads(manifest.read_text(), object_pairs_hook=unique)
    jobs = obj.get('inputs', {})
    allowed = {str(y) for y in YEARS}
    if years is None and set(jobs) != allowed:
        raise ValueError('Expected exactly the years 2011-2025; use --years for a selected subset')
    if not jobs or set(jobs).difference(allowed):
        raise ValueError('Manifest years must be within 2011-2025')
    requested = list(YEARS) if years is None else sorted(set(years))
    if not requested or len(requested) != len(years if years is not None else YEARS):
        raise ValueError('Year selection is empty or contains duplicates')
    if any(str(y) not in jobs for y in requested):
        raise ValueError('The manifest does not contain every selected year')
    paths = {}
    for year in requested:
        path = Path(jobs[str(year)]).expanduser()
        if not path.is_absolute():
            path = manifest.parent / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Missing local JSON for {year}')
        paths[year] = path
    if len(set(paths.values())) != len(paths):
        raise ValueError('Each selected year needs a different annual JSON')
    return paths


def process(args):
    if args.input_json:
        if args.year not in YEARS or args.years is not None:
            raise ValueError('--input-json requires one --year within 2011-2025')
        path = Path(args.input_json).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError('The local JSON file does not exist')
        jobs = {args.year: path}
    else:
        if args.year is not None:
            raise ValueError('Use --years with --inputs-manifest')
        jobs = json_jobs(args.inputs_manifest, args.years)
    config = Path(args.institutions_file).expanduser().resolve()
    sys.path.insert(0, str(SRC))
    import scopus_process
    scopus_process.INSTITUTIONS = scopus_process.load_institutions(config)
    before = {str(year): digest(path) for year, path in jobs.items()}
    config_before = digest(config)
    output = new_directory(args.output)
    annual = output / 'annual'; annual.mkdir()
    inventory = {}
    for year, path in jobs.items():
        print(f'Processing {year}', flush=True)
        tables = scopus_process.process_input(path, analysis_year=year, include_audit=False)
        from annual_outputs import write
        inventory[str(year)] = write(tables, annual / f'{year}.csv')
    if before != {str(y): digest(p) for y, p in jobs.items()} or config_before != digest(config):
        raise ValueError('A private input changed during processing')
    (output / 'processing_report.json').write_text(json.dumps({
        'years': list(jobs), 'inputs_unchanged': True, 'input_sha256': before,
        'institution_configuration_sha256': config_before, 'scientific_comparison_performed': False,
        'annual_outputs': inventory, 'detailed_audit_files_written': False,
        'notice': 'Execution and integrity only; compare historical tables for empirical equivalence.'}, indent=2))
    print(f'Private annual CSVs: {annual}')
