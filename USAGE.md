# Usage

Activate the environment in [INSTALL.md](INSTALL.md) and run commands from the
directory containing the scripts. The examples use Bash. Replace paths with
your local paths and use a new output directory for each command. Keep input
data and generated results outside the code repository.

Two workflows are available:

- **A. Figures from article tables:** start with supplementary Excel or CSV files.
- **B. Methods from Scopus JSONs:** process local records, calculate results,
  then generate tables, figures, or both.

## A. Generate figures from supplementary tables

Obtain the required tables from the article's supplementary material. The
input set for every figure is listed in [README.md](README.md) and its script's
help. No JSON records or private institution mapping are needed.

### One figure

Figure 3 requires `Table-S5.xlsx`:

```bash
python reproduce_figure_3.py --help
python reproduce_figure_3.py \
  --tables-dir /absolute/path/to/article_tables \
  --output /absolute/path/to/new/figure_3
```

For CSV input, supply `Table-S5.csv`:

```bash
python reproduce_figure_3.py \
  --tables-dir /absolute/path/to/article_csv \
  --source-format csv \
  --output /absolute/path/to/new/figure_3_csv
```

Preserve filenames, workbook sheets, column order, anonymous codes and full
stored numerical precision. Exporting a sheet to CSV must preserve its values
and headers; changing an Excel filename extension does not convert the file.

A workbook may carry a second, text-only sheet named `Legend` after the data
sheet. It is ignored when figures are rendered. To add it to existing workbooks
without rewriting the data sheet, write copies to a new folder:

```bash
python tools/add_table_legends.py \
  --tables-dir /absolute/path/to/article_tables \
  --output-dir /absolute/path/to/new/article_tables_with_legend
```

The legend text is defined in `config/table_legends.json`.

Figure 6 requires only S13. Figure S9 requires S13 and S14. Figure S1 uses the
`IMFA or EC-CIS` metric in S2, which represents the union of the two conditions.
The union count cannot be obtained by adding their separate counts.

With `--tables-dir`, a command writes the selected figure and reports without
copying or modifying supplied tables. `--figures-only` is optional in this mode.
Table-generation flags apply to workflow B.

### All figures

```bash
set -euo pipefail
TABLES_DIR="/absolute/path/to/article_tables"
FIGURES_DIR="/absolute/path/to/new/figures_from_tables"

for figure in 1 2 3 4 5 6 7 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10; do
  python "reproduce_figure_${figure}.py" \
    --tables-dir "$TABLES_DIR" \
    --source-format xlsx \
    --output "$FIGURES_DIR/figure_${figure}"
done
```

Use `--source-format csv` for CSV inputs. Remove figure identifiers from the
loop to generate a subset. Across all 17 figures, 13 supplementary tables are
read; S4 and S9 are not required for rendering.

## B. Generate results from Scopus JSON records

This workflow calculates results from local JSON files. Published supplementary
tables are not inputs to processing, calculation, table export or rendering.

### 1. Configure the inputs

Create three private files outside the repository, using the examples in
`config/` as guides:

| Private file | Example | Contents |
| --- | --- | --- |
| `inputs.json` | `inputs.example.json` | One JSON path for each analysis year, 2011–2025 |
| `institutions.json` | `institutions.example.json` | Scopus affiliation IDs and their canonical focal institution IDs and names |
| `mapping.csv` | `mapping.example.csv` | Canonical institution IDs, names, anonymous codes and colours |

The JSON files must contain complete Abstract Retrieval records, rather than
only Search API summaries. Each manifest year must point to its corresponding
annual download. Relative paths are resolved from the manifest directory. The
manifest year determines the analysis publication year, including when a
record's `coverDate` differs.

For an institution with several Scopus affiliation IDs, add one definition per
AF-ID with the same canonical `institution_id` and `name`. For example:

```json
[
  {"afid": "99000001", "institution_id": "EXAMPLE_01", "name": "Fictitious University 01"},
  {"afid": "99000004", "institution_id": "EXAMPLE_01", "name": "Fictitious University 01"}
]
```

Use unique digit strings for AF-IDs and exactly these three fields per definition.
`NATIONAL` is reserved and added by processing. Nonfocal AF-IDs remain separate
entities; institution names are not used to merge them.

The mapping CSV has exactly four columns:
`institution_id,name,id-paper,color_hex`. Include one row per canonical focal
institution and one `NATIONAL` row. Do not repeat mapping rows for aliases.
Study figures use codes `U-01` through `U-30` and `National`, with the study
palette. All supplied examples are fictitious and must be replaced locally.

Exact study reproduction requires the same data snapshot and authorized
institution and code assignments. Later Scopus downloads may contain different
records. Confidential assignments are not recoverable from the examples or
anonymous tables; see [DATA_ACCESS.md](DATA_ACCESS.md).

Define these paths in your shell before running the following steps:

```bash
set -euo pipefail
PRIVATE_DIR="/absolute/path/to/private_inputs"
RUN_DIR="/absolute/path/to/new_run"
MAPPING_ENCODING="utf-8-sig"
```

Use `MAPPING_ENCODING="latin1"` if your mapping is encoded in Latin-1.

### 2. Process the records

```bash
python process_json.py \
  --inputs-manifest "$PRIVATE_DIR/inputs.json" \
  --institutions-file "$PRIVATE_DIR/institutions.json" \
  --output "$RUN_DIR/private/processed"
```

This writes 12 annual CSV families per year and `processing_report.json`.
It does not generate final tables or figures. The report records input hashes
and the output inventory. Intermediate schemas are in [OUTPUTS.md](OUTPUTS.md).

For one year only:

```bash
python process_json.py \
  --input-json "$PRIVATE_DIR/2025_download.json" \
  --year 2025 \
  --institutions-file "$PRIVATE_DIR/institutions.json" \
  --output "$RUN_DIR/private/year_2025"
```

Alternatively, select manifest years with `--years 2011 2012`. The study's
figure layouts require their 2011–2025 series. Annual results may be stored in
separate year directories; downstream searches are recursive and reject duplicates.

### 3. Calculate indicators

```bash
python calculate_indicators.py \
  --derived-dir "$RUN_DIR/private/processed/annual" \
  --mapping-file "$PRIVATE_DIR/mapping.csv" \
  --mapping-encoding "$MAPPING_ENCODING" \
  --for-tables all \
  --output "$RUN_DIR/private/calculated"
```

This computes the inputs for Tables S1–S15, main-text Table 1 and all 17 figures,
without exporting final workbooks or rendering figures. The same calculations
can be reused for any of the output choices below.

`calculation_report.json` contains execution details and methodology diagnostics,
including correlations, threshold and rank statistics, sharing summaries, paired
comparisons and changes between selected years. Undefined values are `null`.
Keep this report and the calculated files private.

### 4. Select the final products

#### Tables only

```bash
python export_tables.py \
  --calculated-dir "$RUN_DIR/private/calculated" \
  --tables all \
  --output "$RUN_DIR/tables"
```

Outputs are `Table-S1.xlsx` through `Table-S15.xlsx`, corresponding CSV files,
`Main-Table1.csv` and a build report. Each workbook includes the `Legend`
sheet unless `--no-legend` is given. No figures are rendered. To exclude
main-text Table 1, replace `all` with
`S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S12 S13 S14 S15`.

#### Figures only

```bash
for figure in 1 2 3 4 5 6 7 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10; do
  python "reproduce_figure_${figure}.py" \
    --calculated-dir "$RUN_DIR/private/calculated" \
    --figures-only \
    --output "$RUN_DIR/figures/figure_${figure}"
done
```

Each output directory contains vector PDF/SVG, a text-editable SVG companion,
a PNG preview and reports. Rendering tables are built temporarily from the
calculations and removed afterwards; no supplementary table files are retained.

#### Tables and figures

Run the table-export command and then the figure loop above. Reuse the same
calculated directory. JSON processing and indicator calculation need not be
repeated. This produces one table directory and independent figure directories.

### Generate selected results independently

After step 2, a figure script can calculate its own inputs. Step 3 is not
required for these examples.

Figure 4 and its associated Tables S7 and S9:

```bash
python reproduce_figure_4.py \
  --derived-dir "$RUN_DIR/private/processed/annual" \
  --mapping-file "$PRIVATE_DIR/mapping.csv" \
  --mapping-encoding "$MAPPING_ENCODING" \
  --output "$RUN_DIR/selected_figure_4_and_tables"
```

Use `--figures-only` for the figure without final tables, or `--only-tables S7 S9`
for those tables without the figure. Use a different output directory for each
run. `--minimal-tables` generates the figure with only its direct rendering
tables. Each script's `--help` lists its associated table products.

Calculations run in temporary storage by default. Add `--work-dir` with a new
private directory outside the result directory to retain them for reuse.
Already calculated inputs can be supplied with `--calculated-dir` instead;
that mode does not require a mapping argument and supports the same output choices.

Selected tables can also be calculated and exported as separate steps:

```bash
python calculate_indicators.py \
  --derived-dir "$RUN_DIR/private/processed/annual" \
  --mapping-file "$PRIVATE_DIR/mapping.csv" \
  --mapping-encoding "$MAPPING_ENCODING" \
  --for-tables S13 S14 \
  --output "$RUN_DIR/private/selected_calculations"
python export_tables.py \
  --calculated-dir "$RUN_DIR/private/selected_calculations" \
  --tables S13 S14 \
  --output "$RUN_DIR/selected_tables_s13_s14"
```

## Check the generated tables

Comparison is a separate, optional step. This command reads the article
workbooks as references after generation:

```bash
python validate_tables.py \
  --reference-dir /absolute/path/to/article_tables \
  --tables-dir "$RUN_DIR/tables" \
  --tables all \
  --report "$RUN_DIR/table_validation.json"
```

Main-text Table 1 is checked separately. See [VALIDATION.md](VALIDATION.md) for
comparison rules, interpretation and figure inspection.
