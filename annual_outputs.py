"""The complete serialization contract for private JSON-processing outputs."""
import json
from pathlib import Path
import pandas as pd

CONTRACT = json.loads((Path(__file__).resolve().parent / 'config/annual_outputs.json').read_text())


def project(tables):
    """Keep only columns with a documented downstream consumer."""
    result = {}
    for name, spec in CONTRACT.items():
        if name not in tables:
            raise ValueError(f'Missing required processing result: {name}')
        frame = tables[name]
        missing = set(spec['columns']).difference(frame.columns)
        if missing and len(frame):
            raise ValueError(f'{name}: missing required columns {sorted(missing)}')
        selected = frame.reindex(columns=spec['columns']).copy()
        if name == 'top_author_concentration':
            selected = selected.loc[pd.to_numeric(selected['n_top_authors'], errors='raise').isin(spec['retained_top_k'])
                                    & selected['institution_id'].ne('NATIONAL')].copy()
        result[name] = selected
    return result


def write(tables, prefix):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    result = project(tables)
    for name, frame in result.items():
        frame.to_csv(f'{prefix.with_suffix("")}_{name}.csv', index=False, encoding='utf-8-sig')
    return {name: {'rows': len(frame), 'columns': list(frame.columns)} for name, frame in result.items()}
