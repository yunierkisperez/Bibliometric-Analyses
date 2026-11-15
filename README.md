# Bibliometric-Analyses

Python scripts for the bibliometric analysis of 30 Ecuadorian institutions,
2011–2025. Each figure and each processing stage can be run independently.

## Choose a starting point

| Available inputs | Workflow | Products |
| --- | --- | --- |
| Supplementary tables distributed with the article | Read the required Excel or CSV files | Any of Figures 1–7 or S1–S10 |
| Locally downloaded Scopus JSON records and private study settings | Process records, calculate indicators, then select outputs | Tables S1–S15, main-text Table 1, figures, or both tables and figures |

The JSON workflow calculates the supplementary tables from the records. It
does not use article tables as inputs. Each figure has its own script; there
is no requirement to generate every figure or table.

## Get started

1. Create the environment in [INSTALL.md](INSTALL.md).
2. Obtain the inputs for your chosen workflow. Research data are not bundled.
3. Follow the commands in [USAGE.md](USAGE.md).

For example, Figure 3 requires only `Table-S5.xlsx`:

```bash
python reproduce_figure_3.py --help
python reproduce_figure_3.py \
  --tables-dir /absolute/path/to/article_tables \
  --output /absolute/path/to/new/figure_3
```

Use `--source-format csv` for CSV input. Table filenames, sheet names, column
order, identifiers and stored numerical precision must be preserved.

Each workbook holds the data sheet, named `Table S1` to `Table S15`, optionally
followed by a text-only `Legend` sheet with the table title, the meaning of
every column, abbreviations and notes. The figure scripts read the data sheet
and accept the workbook with or without the legend; any other sheet layout is
rejected.

## Tables required by each figure

These are the complete input sets for `--tables-dir`. Download only the tables
needed for your figures. Each script's `--help` lists the exact filenames.

| Figure | Script | Required supplementary tables |
| --- | --- | --- |
| 1 | `reproduce_figure_1.py` | S1 |
| 2 | `reproduce_figure_2.py` | S2 |
| 3 | `reproduce_figure_3.py` | S5 |
| 4 | `reproduce_figure_4.py` | S7 |
| 5 | `reproduce_figure_5.py` | S10, S11 |
| 6 | `reproduce_figure_6.py` | S13 |
| 7 | `reproduce_figure_7.py` | S15 |
| S1 | `reproduce_figure_s1.py` | S2 |
| S2 | `reproduce_figure_s2.py` | S3 |
| S3 | `reproduce_figure_s3.py` | S5 |
| S4 | `reproduce_figure_s4.py` | S6 |
| S5 | `reproduce_figure_s5.py` | S6 |
| S6 | `reproduce_figure_s6.py` | S8 |
| S7 | `reproduce_figure_s7.py` | S11 |
| S8 | `reproduce_figure_s8.py` | S12 |
| S9 | `reproduce_figure_s9.py` | S13, S14 |
| S10 | `reproduce_figure_s10.py` | S13 |

For example, S13 means `Table-S13.xlsx` or `Table-S13.csv`. Table S9 and Figure
S9 are separate products. Tables S4 and S9 are methodological outputs generated
by the JSON workflow; they are not direct inputs to these figure scripts.

## Outputs

Figures include vector PDF and SVG, a text-editable SVG companion, a PNG preview
and execution reports. Source Sans 3 fonts are included. Table generation
preserves the article's Excel sheets, columns, identifiers, formulas and number
formats, adds the `Legend` sheet and also exports CSV versions.

The JSON workflow supports tables only, figures only, or both. It also supports
individual figures and selected tables. See [USAGE.md](USAGE.md) for commands
and [OUTPUTS.md](OUTPUTS.md) for file formats.

## Data access and reproducibility

Tables S1–S15 are distributed with the article, separately from this repository.
Scopus JSONs, credentials and the private institution-to-code mapping are not
distributed. The scripts process existing local downloads and do not call the
Scopus API. Configuration examples contain fictitious institutions.

Reproducing the study from JSON requires its data snapshot and authorized
private settings. Researchers can apply the methods to their own data and
settings, but matching the anonymous study institutions requires the private
code assignments. The table-based figure workflow needs no mapping.

See [DATA_ACCESS.md](DATA_ACCESS.md) for input requirements and
[VALIDATION.md](VALIDATION.md) for checks, comparison rules and limitations.
Successful execution alone does not establish agreement with the article.

## Software checks and support

```bash
python tools/check_installation.py
python tools/check_entrypoints.py
python -m unittest discover -s tests -v
```

Tests use fictitious records. For a software problem, report the version,
command, Python environment and relevant error. Remove credentials, private
paths, research records and institution-code associations before sharing logs.

Font and artwork notices are in [ASSET_RIGHTS.md](ASSET_RIGHTS.md). Code and data
licence boundaries are described in [LICENSE_SCOPE.md](LICENSE_SCOPE.md).

## Software citation and licensing

See [CITATION.md](CITATION.md) for available software citation information and
[LICENSE_SCOPE.md](LICENSE_SCOPE.md) for licensing status. No code licence or
software author list is specified. Bundled fonts retain their OFL-1.1 licence.
