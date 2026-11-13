"""File integrity and command helpers shared by independent entry points."""
import hashlib
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def new_directory(path):
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise ValueError(f'Output directory already exists: {path}')
    path.mkdir(parents=True, mode=0o700)
    return path


def run_command(command, log, env=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.run([str(x) for x in command], stdout=stream, stderr=subprocess.STDOUT, env=env)
    if process.returncode:
        raise RuntimeError(f'Command failed ({process.returncode}); see {log}')


def invoke(parser, function):
    args = parser.parse_args()
    os.umask(0o077)
    try:
        return function(args)
    except Exception as error:
        parser.exit(1, f'{type(error).__name__}: {error}\n')
