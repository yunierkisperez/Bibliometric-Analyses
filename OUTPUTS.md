# Processing outputs and their use

process_json.py serializes exactly 12 CSV families per selected year, plus
one processing_report.json describing input hashes, row/column inventory and
execution scope. The annual files remain private. They are intermediates for
the methodology, not an additional dataset for GitHub or the article.

The serialization contract is `config/annual_outputs.json`. Processing writes
only the fields consumed by the calculation workflow, as listed below.

| Annual suffix | Analysis consumers | Use |
| --- | --- | --- |
| `article_affiliations.csv` | sharing | Resolve national article sharing and nonfocal partners (Tables S10-S12). |
| `author_affiliations.csv` | sharing | Distinguish sharing mechanisms using author-affiliation links (Tables S10-S12). |
| `corresponding_author_affiliations.csv` | sharing | Determine corresponding-author leadership of shared publications (Tables S10-S12). |
| `article_institution_flags.csv` | composition, sharing | Reconstruct union and three-state composition; identify national sharing (Tables S4-S6, S10-S12). |
| `institution_year_metrics.csv` | growth, exposure, composition, concentration, affiliations, signals, temporal | Institution-year denominators, signals and concentration needed for Figures 1-7, S1-S10 and their tables. |
| `national_baseline.csv` | composition | Check unique national publication accounting and leadership (Tables S4-S6). |
| `top_author_concentration.csv` | concentration | Validate focal CTA-1, CTA-5 and CTA-10 unions; only k=1,5,10 and focal rows are retained (Tables S7-S9). |
| `author_productivity.csv` | exposure, composition, concentration | Compute HPA-MFA/EC-CIS union coverage, concentration profiles and leading-author persistence. |
| `hyperprolific_author_flags.csv` | exposure, composition, concentration | Select hyperprolific authors with IMFA or EC-CIS for output dependency and threshold checks. |
| `imfa_author_partner_institutions.csv` | affiliations | Reconstruct article and author partner distributions, retaining focal alias consolidation (Tables S13-S14). |
| `cis_author_partner_institutions.csv` | affiliations | Reconstruct article and author partner distributions, retaining focal alias consolidation (Tables S13-S14). |
| `institutions.csv` | sharing | Consolidate multiple focal AF-IDs before national sharing calculations. |

Filenames start with the analysis year, for example
`2025_institution_year_metrics.csv`. Fifteen years produce 180 annual CSVs.
A selected-year run produces only 12 per selected year. Raw JSONs and configuration files
are checked for changes and are never overwritten.

## Exact retained columns

These are private intermediate schemas. They do not alter the Excel schemas
of Tables S1–S15. Author names/IDs and affiliation links below are retained
only where required by the current scientific loaders and methods.

### article_affiliations

```text
eid
afid
affiliation_name
affiliation_country
```

### author_affiliations

```text
eid
author_id
afid
```

### corresponding_author_affiliations

```text
eid
afid
```

### article_institution_flags

```text
eid
institution_id
name
year
analysis_year
n_total_authors_article
CA_known
flag_IMFA
flag_CIS
flag_HPA
flag_HAO
flag_NLO
flag_any_risk
```

### institution_year_metrics

```text
AnyRisk_rate
CA_known
CA_unknown_or_unmatched
CIS_rate
HAO_rate
HPA_rate
IMFA_or_CIS_rate
IMFA_rate
NLO_rate
N_articles_CA_led
N_articles_CIS
N_articles_IMFA
N_articles_IMFA_or_CIS
N_articles_any_risk
N_articles_attributed
N_articles_hyperauthored
N_articles_non_led
N_articles_with_hyperprolific_author
N_authors_institution
N_hyperprolific_authors
N_hyperprolific_authors_MFA_or_CIS
N_top_authors_needed_50_percent_output_union
institution_id
name
top_10_authors_article_count
top_10_authors_article_share
top_1_authors_article_count
top_1_authors_article_share
top_5_authors_article_count
top_5_authors_article_share
year
```

### national_baseline

```text
AnyRisk_rate
CA_known
CA_unknown_or_unmatched
CIS_rate
HAO_rate
HPA_rate
IMFA_rate
NLO_rate
N_articles_CA_led
N_articles_CIS
N_articles_IMFA
N_articles_any_risk
N_articles_attributed
N_articles_hyperauthored
N_articles_non_led
N_articles_with_hyperprolific_author
N_hyperprolific_authors
N_hyperprolific_authors_MFA_or_CIS
institution_id
name
year
```

### top_author_concentration

```text
institution_id
year
n_top_authors
N_articles_attributed
n_articles_by_top_authors_union
top_authors_article_share
```

### author_productivity

```text
institution_id
year
author_rank
author_id
indexed_name
p_author_institution_year
author_article_eids
```

### hyperprolific_author_flags

```text
institution_id
year
author_id
indexed_name
p_author_institution_year
is_hyperprolific_author
is_multi_affiliated_same_publication
is_cross_institution_signer
is_multi_affiliated_or_cross_institution
```

### imfa_author_partner_institutions

```text
institution_id
author_id
partner_afid
imfa_article_eids
partner_affiliation_name
partner_affiliation_country
partner_focal_institution_id
partner_focal_institution_name
```

### cis_author_partner_institutions

```text
institution_id
author_id
partner_afid
external_article_eids
partner_affiliation_name
partner_affiliation_country
partner_focal_institution_id
partner_focal_institution_name
```

### institutions

```text
afid
institution_id
name
```

## Calculations and final products

calculate_indicators.py runs only selected scientific families in data-only
mode. Retained calculation files are enumerated in config/retained_csv.json;
they supply the table adapters or a dependency of another selected family.
The internal Source-Fig-S1.csv supplies the IMFA-or-EC-CIS union rows in Table S2.
It remains a private calculation intermediate and is not exported as a new
supplementary product. Tables S2 and S3 require the exposure and temporal
dependencies; their union and recurrence statistics are not guessed from
marginal counts.

export_tables.py writes selected Table-Sn.xlsx and Table-Sn.csv files using
the sheet, header, type, number-format and formula contracts in config/.
Each workbook has the data sheet followed by a text-only Legend sheet built
from config/table_legends.json; --no-legend omits it. The legend holds the
table title, column meanings, abbreviations and notes, and no data.
Figure entry points can perform this export locally for their own products.
With --tables-dir, only direct construction inputs are read and no table is
copied or re-exported. With annual or calculated inputs, --figures-only builds
construction tables temporarily and removes them after rendering; the result
contains no XLSX/CSV table deliverables.

Each figure writes a vector PDF, an outlined SVG, a live-text _text.svg,
an inspection preview, layout metadata and reports. The vector check detects
embedded raster images and verifies that the editable text companion exists.
It does not prove visual identity with manuscript artwork.

Retained records are not necessarily anonymous merely because the final
figures are. Do not copy processed/annual, calculation logs, mappings or
any other record-level research data into a public repository. See DATA_ACCESS.md.

## Private methodology diagnostics

The calculation_report.json records computed diagnostic summaries
under methodology_diagnostics. Available components depend on selected analysis
families and cover author/partner size correlations, HPA threshold and rank
statistics, sharing/leadership summaries, matched changes, paired differences
and signal phase summaries. Counts and finite floating-point values retain
their native JSON numeric values; undefined diagnostics are null. These
diagnostics are report fields, not supplementary table columns. Keep the report private.
