"""Output selection and stable anonymous sequence identifiers."""
import json
import math
import os
from pathlib import Path
CONFIG = Path(__file__).resolve().parents[1] / 'config'
POLICY = {Path(x).name for x in json.loads((CONFIG/'retained_csv.json').read_text())}
POLICY.add('Source-Fig-S1.csv')
def keep(path):
    return Path(path).name in POLICY
def data_only():
    return os.environ.get('REPRO_DATA_ONLY') == '1'
def persistence_ids(frame):
    """Apply the original generator's sequential labels to its sorted rows.

    persistence_sequences sorts by decreasing sequence length, decreasing
    median CTA-1, then increasing anonymous institution code. The identifiers
    describe rows in that calculated result; they are not permanent author IDs.
    No published table, label registry or external environment variable is read.
    """
    return [f'EP-{number:03d}' for number in range(1, len(frame) + 1)]


def report_methodology(**components):
    """Send calculated diagnostics to the parent calculation report.

    The existing private log is the transport. No supplementary reference file
    is read and no extra result file or workbook column is created.
    """
    import pandas as pd
    def clean(value):
        if value is pd.NA or value is pd.NaT:
            return None
        if hasattr(value, 'columns') and hasattr(value, 'to_dict'):
            return clean(value.to_dict(orient='records'))
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if value is None or isinstance(value, (str, int, bool)):
            return value
        if hasattr(value, 'item'):
            return clean(value.item())
        raise TypeError(f'Unsupported methodology diagnostic type: {type(value).__name__}')
    print('METHOD_DIAGNOSTICS_JSON ' + json.dumps(clean(components), allow_nan=False), flush=True)
