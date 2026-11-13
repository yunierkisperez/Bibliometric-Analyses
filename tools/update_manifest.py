#!/usr/bin/env python3
"""Refresh file hashes after an intentional review; does not validate science."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '.venv', '__pycache__', '.pytest_cache'}


if __name__ == '__main__':
    rows=[]
    for path in sorted(ROOT.rglob('*')):
        relative=path.relative_to(ROOT)
        if not path.is_file() or set(relative.parts).intersection(SKIP) or relative.as_posix()=='FILES_SHA256.txt':
            continue
        if path.is_symlink():raise ValueError('Do not include symlinks in the distribution')
        rows.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative.as_posix()}\n')
    (ROOT/'FILES_SHA256.txt').write_text(''.join(rows))
    print(f'Hashed {len(rows)} files. Run the repository check and scientific tests separately.')
