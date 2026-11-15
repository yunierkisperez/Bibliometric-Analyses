# Checking a reproduction run

Run software checks before processing data, compare generated tables when
reference workbooks are available, and inspect figures separately. These checks
answer different questions.

## Software checks

```bash
python tools/check_installation.py
python tools/check_entrypoints.py
python -m unittest discover -s tests -v
```

The installation check verifies dependencies, font integrity and native Cairo
conversion. The entry-point check verifies command help, including required
figure inputs. Unit tests use fictitious records and check calculation and
file-format contracts. They do not establish agreement with research data.

For the code distribution and file-integrity check, also run:

```bash
python tools/check_repository.py
```

This checks the defined distribution policy and file hashes. It is not a
general detector of private information; review files before distributing them.
The optional `--publication` check also requires a declared code licence and
complete software citation metadata. Unspecified publication metadata does not
prevent local calculations.

## Compare generated supplementary tables

After generating tables from JSONs, compare them with the article workbooks:

```bash
python validate_tables.py \
  --reference-dir /absolute/path/to/article_tables \
  --tables-dir /absolute/path/to/generated_tables \
  --tables all \
  --report /absolute/path/to/table_validation.json
```

For a subset, replace `all` with table identifiers such as `S13 S14`. References
are read only by this comparison command, not by the generation workflow.
Main-text Table 1 requires a separate comparison of `Main-Table1.csv`.

The comparison checks sheet order, dimensions, column order, identifiers,
categories, value types, formulas, formula caches and number formats. The
optional text-only `Legend` sheet is not compared; the report states whether
each workbook has one. Counts
and identifiers must agree exactly. Only the rate and robust-D columns listed
in `table_validation.py` permit this numerical tolerance:

```text
abs(candidate - reference) <= 1e-12 + 5e-14 * abs(reference)
```

Missing values remain missing. Formula expressions and cached values in S5/S6
are checked separately. S2 must include its `IMFA or EC-CIS` rows for Figure S1;
a table missing that indicator is not a complete reference for this workflow.

If a comparison fails, inspect the reported file, sheet and cell. Check input
years, record completeness, focal AF-ID aliases, private code assignments and
parameters before changing any output. A later Scopus snapshot may legitimately
produce different results.

## Inspect figures

Check the PDF or SVG at its intended publication size against the corresponding
article figure. Inspect values, axes, scales, colour assignments, legends,
annotations, panel placement and tied-row ordering. Confirm that the PDF/SVG
contains vector marks and that the text SVG can be edited with Source Sans 3.

Full visual agreement with the article has not been certified. Areas requiring
particular inspection are colour categories in Figure S4, tied-row ordering
and the footer in Figure S9, and styling or point placement in Figures 3 and 4.
Font-dependent text spacing also needs visual inspection. Automated vector
checks do not establish graphical agreement.

## Interpret execution reports

- `processing_report.json` records input hashes, selected years and annual files.
- `calculation_report.json` records selected analyses and private diagnostic results.
- Table-build reports record generated products and their contracts.
- `reproduction_report.json` records the figure inputs, output selection and hashes.
- The separate table-validation report records reference comparisons.

Confirm input-integrity fields and keep private records and logs outside the
code directory. A successful command establishes that the command completed;
it does not establish that its scientific outputs match the article.

Record the tested starting point: fictitious records, real annual intermediates,
article tables, or raw Scopus JSONs. Validation from one starting point does not
validate the preceding stages. A full JSON-based reproduction requires the
study data snapshot and authorized private configuration.
