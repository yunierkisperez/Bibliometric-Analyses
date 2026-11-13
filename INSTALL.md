# Installation

Use an isolated Python environment. `environment.yml` specifies Python 3.12.14,
native libraries and the Python packages pinned in `requirements.txt`.

## Conda

From the repository directory:

```bash
conda env create -f environment.yml
conda activate bibliometric-analyses
python tools/check_installation.py
```

The environment includes Cairo and libffi from conda-forge. Activate it whenever
you open a new shell to run the scripts.

## Python virtual environment

Install Python 3.12 and the native Cairo library first. On Debian or Ubuntu,
the Cairo package is `libcairo2`. Then run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/check_installation.py
```

On Windows, activate with `.venv\Scripts\activate`. Conda provides a convenient
way to install Cairo on that platform. Installing CairoSVG alone does not supply
the native library required for PDF export.

## Check the environment

```bash
python -m pip check
python tools/check_entrypoints.py
python -m unittest discover -s tests -v
```

`check_installation.py` imports dependencies, verifies the bundled font hashes
and tests vector PDF conversion in memory. For table processing without the PDF
conversion dependencies, run `python tools/check_installation.py --tables-only`.

Source Sans 3 fonts are bundled and loaded directly from their files. System
font installation is not needed to generate figures. Install the fonts in your
vector editor or operating system to edit the text SVG companions consistently.

Follow [USAGE.md](USAGE.md) after installation. All study data must already
exist locally; the software does not download records from Scopus.
