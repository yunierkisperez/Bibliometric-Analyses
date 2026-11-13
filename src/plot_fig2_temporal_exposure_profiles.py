#!/usr/bin/env python3
"""
Create Figure 2 for the temporal reconfiguration of Ecuador's institutional
exposure profile, 2011-2025.

The main figure contains six panels. Standard indicators are represented by
annual box plots calculated only from individual universities with a sufficient
denominator. Robust deviations are overlaid using the institution codes and
colors in MAPPING_with_colors.csv. For each indicator-year, D is calculated as
(x - annual median) / (annual IQR / 1.349). Upper signals satisfy D >= 2 and
extreme upper signals satisfy D >= 3. Descriptive lower deviations (D <= -2)
are retained only for IMFA, CIS, and NLO; structural zeros are therefore not
classified as lower deviations for HPA, HAO, or HPA-MFA/CIS dependence.
NATIONAL is always plotted as a separate continuous black line and never enters
an institutional box, median, IQR, or robust-deviation calculation.

HAO is treated separately because it is zero-inflated. Its panel shows annual
boxes for positive university observations and uses a symmetric-log scale;
zero observations are retained in the output tables but are not drawn. Sparse
positive years are shown as points rather than boxes.

The article-level HPA-MFA/CIS panel shows the percentage of institutional output
attributable to hyperprolific authors who maintain simultaneous affiliations
(MFA) or sign for other institutions (CIS). Articles are counted once within an
institution-year even when several selected authors sign the same paper. Because
this metric is also strongly zero-inflated, its boxes summarize positive eligible
university observations; zero observations remain in the output tables but are
not drawn. The metric is derived automatically from the annual input directory
and validated against the annual article denominators and HPA author counts. An
existing derived CSV can optionally be supplied for an audited rerun.

HPA, HAO, and HPA-MFA/CIS dependence use positive eligible observations to
define their annual robust reference distributions. All eligible institutions,
including zeros, are nevertheless evaluated for upper signals. A reference is
valid only when it contains at least eight institutions, at least two distinct
values, and a positive IQR.

The screening is descriptive. It is not a test of misconduct or a formal
hypothesis test. The analysis year is always the year encoded in the exact
annual filename (YYYY_institution_year_metrics.csv), never a year derived from
coverDate. Disagreements with the table's year column are retained in an audit.
"""

from __future__ import annotations

import figure_style
import output_utils

import argparse
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


SCRIPT_VERSION = "6.0-robust-D-2026-08-20"
NATIONAL_ID = "NATIONAL"
ROBUST_NORMAL_CONSISTENCY = 1.349
REQUIRED_ANNUAL_SUFFIXES = (
    "institution_year_metrics.csv",
    "hyperprolific_author_flags.csv",
    "author_productivity.csv",
)

INVALID_TEXT_VALUES = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "missing",
    "not available",
}

# Configuration order is also the panel order.
METRICS = (
    {
        "key": "IMFA",
        "title": "IMFA",
        "description": "Articles with institutional multi-affiliation",
        "value_type": "rate",
        "value_col": "IMFA_rate",
        "count_col": "N_articles_IMFA",
        "denominator_col": "N_articles_attributed",
        "eligibility_col": "N_articles_attributed",
        "secondary_min_col": None,
        "display_kind": "box",
        "classification_basis": "all_eligible_institutions",
        "lower_tail_evaluated": True,
        "unit_label": "Articles (%)",
        "ylim": (0.0, 80.0),
        "ytick": 20.0,
    },
    {
        "key": "CIS",
        "title": "EC-CIS",
        "description": "Articles with cross-institution signers",
        "value_type": "rate",
        "value_col": "CIS_rate",
        "count_col": "N_articles_CIS",
        "denominator_col": "N_articles_attributed",
        "eligibility_col": "N_articles_attributed",
        "secondary_min_col": None,
        "display_kind": "box",
        "classification_basis": "all_eligible_institutions",
        "lower_tail_evaluated": True,
        "unit_label": "Articles (%)",
        "ylim": (0.0, 65.0),
        "ytick": 20.0,
    },
    {
        "key": "HPA",
        "title": "HPA",
        "description": "Articles with at least one hyperprolific author",
        "value_type": "rate",
        "value_col": "HPA_rate",
        "count_col": "N_articles_with_hyperprolific_author",
        "denominator_col": "N_articles_attributed",
        "eligibility_col": "N_articles_attributed",
        "secondary_min_col": None,
        "display_kind": "box",
        "classification_basis": "positive_eligible_institutions",
        "lower_tail_evaluated": False,
        "unit_label": "Articles (%)",
        "ylim": (0.0, 55.0),
        "ytick": 10.0,
    },
    {
        "key": "HAO",
        "title": "HAO",
        "description": "Hyperauthored articles",
        "value_type": "rate",
        "value_col": "HAO_rate",
        "count_col": "N_articles_hyperauthored",
        "denominator_col": "N_articles_attributed",
        "eligibility_col": "N_articles_attributed",
        "secondary_min_col": None,
        "display_kind": "positive_box",
        "classification_basis": "positive_eligible_institutions",
        "lower_tail_evaluated": False,
        "unit_label": "Articles (%; symlog)",
        "ylim": (0.0, 60.0),
        "ytick": None,
    },
    {
        "key": "NLO",
        "title": "Non-led output (NLO)",
        "description": "Articles not led by the focal institution",
        "value_type": "rate",
        "value_col": "NLO_rate",
        "count_col": "N_articles_non_led",
        "denominator_col": "CA_known",
        "eligibility_col": "CA_known",
        "secondary_min_col": None,
        "display_kind": "box",
        "classification_basis": "all_eligible_institutions",
        "lower_tail_evaluated": True,
        "unit_label": "CA-known articles (%)",
        "ylim": (0.0, 85.0),
        "ytick": 20.0,
    },
    {
        "key": "HPA_MFA_or_CIS_articles",
        "title": "Output dependency on HPA with\nMFA or EC-CIS",
        "description": (
            "Share of institutional output attributable to hyperprolific authors "
            "who maintain simultaneous affiliations (MFA) or sign for other "
            "institutions (CIS)"
        ),
        "external": True,
        "value_type": "rate",
        "value_col": "HPA_MFA_or_CIS_article_rate",
        "count_col": "N_articles_by_HPA_MFA_or_CIS",
        "denominator_col": "N_articles_attributed",
        "eligibility_col": "N_articles_attributed",
        "secondary_min_col": None,
        "display_kind": "positive_box",
        "classification_basis": "positive_eligible_institutions",
        "lower_tail_evaluated": False,
        "unit_label": "Attributed articles (%)",
        "ylim": (0.0, 50.0),
        "ytick": 10.0,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the six-panel Figure 2 with annual university box plots, "
            "mapped robust deviations, and separate national lines."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing exact YYYY_institution_year_metrics.csv files.",
    )
    parser.add_argument(
        "--mapping-file",
        required=True,
        help="CSV containing institution IDs, manuscript codes, and colors.",
    )
    parser.add_argument(
        "--hpa-article-file",
        default=None,
        help=(
            "Optional pre-derived CSV containing institution-year "
            "N_articles_by_HPA_MFA_or_CIS and HPA_MFA_or_CIS_article_rate. "
            "If omitted, the file is derived automatically from --input-dir "
            "and written below --output-dir."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--mapping-encoding", default="latin1")
    parser.add_argument(
        "--min-denominator",
        type=int,
        default=50,
        help=(
            "Minimum annual article denominator for institutional comparison. "
            "NLO uses CA_known; the other panels use N_articles_attributed."
        ),
    )
    parser.add_argument(
        "--min-box-institutions",
        type=int,
        default=5,
        help="Minimum eligible universities required to draw an annual box.",
    )
    parser.add_argument(
        "--min-reference-institutions",
        "--min-outlier-institutions",
        dest="min_reference_institutions",
        type=int,
        default=8,
        help=(
            "Minimum observations required in an annual robust reference "
            "distribution. The legacy --min-outlier-institutions name is "
            "accepted as an alias."
        ),
    )
    parser.add_argument(
        "--elevated-d",
        type=float,
        default=2.0,
        help="Absolute robust-D threshold for an elevated deviation.",
    )
    parser.add_argument(
        "--extreme-d",
        type=float,
        default=3.0,
        help="Absolute robust-D threshold for an extreme deviation.",
    )
    parser.add_argument("--box-width", type=float, default=0.58)
    parser.add_argument(
        "--x-tick-step",
        type=int,
        default=2,
        help="Year-label interval. Two gives equally spaced 2011, 2013, ..., 2025.",
    )
    parser.add_argument("--figure-width", type=float, default=7.5)
    parser.add_argument("--figure-height", type=float, default=8.4)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--output-stem",
        default="figure2_robust_d_temporal_exposure_profiles",
    )
    args = parser.parse_args()

    if args.end_year < args.start_year:
        parser.error("--end-year must be greater than or equal to --start-year.")
    if args.min_denominator < 1:
        parser.error("--min-denominator must be positive.")
    if args.min_box_institutions < 2:
        parser.error("--min-box-institutions must be at least 2.")
    if args.min_reference_institutions < 4:
        parser.error("--min-reference-institutions must be at least 4.")
    if args.elevated_d <= 0:
        parser.error("--elevated-d must be positive.")
    if args.extreme_d < args.elevated_d:
        parser.error("--extreme-d must be greater than or equal to --elevated-d.")
    if not 0 < args.box_width <= 1.0:
        parser.error("--box-width must be in (0, 1].")
    if args.x_tick_step < 1:
        parser.error("--x-tick-step must be positive.")
    if args.figure_width <= 0 or args.figure_height <= 0:
        parser.error("Figure dimensions must be positive.")
    if args.dpi < 150:
        parser.error("--dpi must be at least 150; 300 is recommended.")
    return args


def is_invalid_text(value: object) -> bool:
    return str(value).strip().lower() in INVALID_TEXT_VALUES


def clean_column_names(columns: Iterable[object]) -> list[str]:
    return [str(column).replace("\ufeff", "").strip() for column in columns]


def parse_code_order(code: object) -> int:
    match = re.fullmatch(r"U-(\d+)", str(code).strip(), flags=re.IGNORECASE)
    return int(match.group(1)) if match else 10_000


def load_mapping(path: Path, encoding: str) -> pd.DataFrame:
    mapping = pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False)
    mapping.columns = clean_column_names(mapping.columns)
    required = {"institution_id", "name", "id-paper", "color_hex"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Mapping file is missing required columns: {sorted(missing)}")

    for column in required:
        mapping[column] = mapping[column].astype(str).str.strip()
        bad = mapping[column].map(is_invalid_text)
        if bad.any():
            raise ValueError(
                f"Mapping column {column!r} contains invalid values at rows "
                f"{(mapping.index[bad] + 2).tolist()}"
            )

    for column in ("institution_id", "id-paper"):
        duplicated = mapping.loc[mapping[column].duplicated(keep=False), column]
        if not duplicated.empty:
            raise ValueError(
                f"Mapping contains duplicated {column} values: "
                f"{sorted(set(duplicated.astype(str)))}"
            )

    valid_color = mapping["color_hex"].str.fullmatch(r"#[0-9A-Fa-f]{6}")
    if not valid_color.all():
        bad = mapping.loc[~valid_color, ["institution_id", "color_hex"]]
        raise ValueError(
            "Every color_hex must use #RRGGBB. Invalid values:\n"
            + bad.to_string(index=False)
        )
    mapping["color_hex"] = mapping["color_hex"].str.upper()

    if (mapping["institution_id"] == NATIONAL_ID).sum() != 1:
        raise ValueError("Mapping must contain exactly one NATIONAL row.")

    if "legend_label" not in mapping.columns:
        mapping["legend_label"] = mapping["id-paper"]
    else:
        mapping["legend_label"] = mapping["legend_label"].astype(str).str.strip()
        invalid = mapping["legend_label"].map(is_invalid_text)
        mapping.loc[invalid, "legend_label"] = mapping.loc[invalid, "id-paper"]

    mapping["code_order"] = mapping["id-paper"].map(parse_code_order)
    return mapping


def required_data_columns() -> set[str]:
    required = {
        "institution_id",
        "name",
        "year",
        "N_articles_attributed",
        "CA_known",
    }
    for metric in METRICS:
        if metric.get("external", False):
            continue
        required.update(
            {
                metric["count_col"],
                metric["denominator_col"],
                metric["eligibility_col"],
            }
        )
        if metric["value_col"] is not None:
            required.add(metric["value_col"])
        if metric["secondary_min_col"] is not None:
            required.add(metric["secondary_min_col"])
    return required


def load_annual_data(
    input_dir: Path,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pattern = re.compile(r"^(\d{4})_institution_year_metrics\.csv$")
    selected: dict[int, Path] = {}
    ignored_rows: list[dict[str, object]] = []

    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if match:
            year = int(match.group(1))
            if start_year <= year <= end_year:
                if year in selected:
                    raise ValueError(f"More than one annual metrics file was found for {year}.")
                selected[year] = path
            else:
                ignored_rows.append(
                    {"file": path.name, "reason": "year_outside_requested_range"}
                )

    expected_years = list(range(start_year, end_year + 1))
    missing_years = sorted(set(expected_years) - set(selected))
    if missing_years:
        raise FileNotFoundError(
            "Missing exact annual institution metrics files for years: "
            f"{missing_years}"
        )

    required = required_data_columns()
    frames: list[pd.DataFrame] = []
    loaded_rows: list[dict[str, object]] = []
    off_year_frames: list[pd.DataFrame] = []
    numeric_columns = sorted(required - {"institution_id", "name"})

    for analysis_year in expected_years:
        path = selected[analysis_year]
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        frame.columns = clean_column_names(frame.columns)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

        frame["institution_id"] = frame["institution_id"].astype(str).str.strip()
        frame["name"] = frame["name"].astype(str).str.strip()
        if frame["institution_id"].map(is_invalid_text).any():
            raise ValueError(f"{path.name} contains an invalid institution_id.")
        duplicated = frame.loc[
            frame["institution_id"].duplicated(keep=False), "institution_id"
        ]
        if not duplicated.empty:
            raise ValueError(
                f"{path.name} contains duplicated institution IDs: "
                f"{sorted(set(duplicated))}"
            )
        if (frame["institution_id"] == NATIONAL_ID).sum() != 1:
            raise ValueError(f"{path.name} must contain exactly one NATIONAL row.")

        original_numeric = frame[numeric_columns].copy()
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            newly_missing = original_numeric[column].notna() & frame[column].isna()
            if newly_missing.any():
                raise ValueError(
                    f"{path.name}: nonnumeric values found in {column} at data rows "
                    f"{(frame.index[newly_missing] + 2).tolist()}"
                )

        source_year = frame["year"].copy()
        off_year = source_year.notna() & source_year.ne(analysis_year)
        if off_year.any():
            audit = frame.loc[off_year, ["institution_id", "name", "year"]].copy()
            audit.insert(0, "source_file", path.name)
            audit.insert(1, "analysis_year", analysis_year)
            audit = audit.rename(columns={"year": "table_year"})
            off_year_frames.append(audit)

        frame["table_year"] = source_year
        frame["analysis_year"] = analysis_year
        frame["source_file"] = path.name

        for metric in METRICS:
            if metric.get("external", False):
                continue
            count_col = metric["count_col"]
            denominator_col = metric["denominator_col"]
            for column in {count_col, denominator_col, metric["eligibility_col"]}:
                if (frame[column].dropna() < 0).any():
                    raise ValueError(f"{path.name}: {column} contains negative values.")

            impossible = (
                frame[count_col].notna()
                & frame[denominator_col].notna()
                & frame[count_col].gt(frame[denominator_col])
            )
            if impossible.any():
                raise ValueError(
                    f"{path.name}: {count_col} exceeds {denominator_col} for "
                    f"{metric['key']}."
                )

            if metric["value_type"] == "rate":
                rate_col = metric["value_col"]
                bad_rate = frame[rate_col].notna() & ~frame[rate_col].between(0, 1)
                if bad_rate.any():
                    raise ValueError(
                        f"{path.name}: {rate_col} contains values outside [0, 1]."
                    )
                valid_ratio = (
                    frame[denominator_col].gt(0)
                    & frame[count_col].notna()
                    & frame[rate_col].notna()
                )
                expected_rate = frame[count_col] / frame[denominator_col]
                mismatch = valid_ratio & (frame[rate_col] - expected_rate).abs().gt(1e-9)
                if mismatch.any():
                    raise ValueError(
                        f"{path.name}: {rate_col} is inconsistent with "
                        f"{count_col}/{denominator_col}."
                    )

        frames.append(frame)
        loaded_rows.append(
            {
                "analysis_year": analysis_year,
                "file": path.name,
                "rows": len(frame),
                "institution_rows": int((frame["institution_id"] != NATIONAL_ID).sum()),
                "off_year_rows": int(off_year.sum()),
            }
        )

    data = pd.concat(frames, ignore_index=True)
    loaded = pd.DataFrame(loaded_rows)
    ignored = pd.DataFrame(ignored_rows, columns=["file", "reason"])
    if off_year_frames:
        off_year = pd.concat(off_year_frames, ignore_index=True)
    else:
        off_year = pd.DataFrame(
            columns=["source_file", "analysis_year", "institution_id", "name", "table_year"]
        )
    return data, loaded, ignored, off_year


def load_hpa_article_metric(
    path: Path,
    annual_data: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and merge the uniformly reconstructed article-level metric."""
    required = {
        "analysis_year",
        "institution_id",
        "N_articles_attributed",
        "N_hyperprolific_authors",
        "N_hyperprolific_authors_MFA_or_CIS",
        "N_selected_HPA_MFA_or_CIS_authors",
        "N_articles_by_HPA_MFA_or_CIS",
        "HPA_MFA_or_CIS_article_rate",
        "derivation_method",
        "source_hpa_flags_file",
        "source_author_productivity_file",
        "source_metrics_file",
        "script_version",
    }
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame.columns = clean_column_names(frame.columns)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing derived-metric columns: {sorted(missing)}"
        )

    frame["institution_id"] = frame["institution_id"].astype(str).str.strip()
    if frame["institution_id"].map(is_invalid_text).any():
        raise ValueError(f"{path.name} contains an invalid institution_id.")

    numeric_columns = [
        "analysis_year",
        "N_articles_attributed",
        "N_hyperprolific_authors",
        "N_hyperprolific_authors_MFA_or_CIS",
        "N_selected_HPA_MFA_or_CIS_authors",
        "N_articles_by_HPA_MFA_or_CIS",
        "HPA_MFA_or_CIS_article_rate",
    ]
    for column in numeric_columns:
        original = frame[column].copy()
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        bad = original.notna() & frame[column].isna()
        if bad.any():
            raise ValueError(
                f"{path.name}: nonnumeric values found in {column} at data rows "
                f"{(frame.index[bad] + 2).tolist()}"
            )
    noninteger_years = frame["analysis_year"].notna() & frame[
        "analysis_year"
    ].mod(1).ne(0)
    if noninteger_years.any():
        raise ValueError(f"{path.name} contains noninteger analysis years.")
    frame["analysis_year"] = frame["analysis_year"].astype(int)

    keys = ["analysis_year", "institution_id"]
    if frame.duplicated(keys).any():
        duplicated = frame.loc[frame.duplicated(keys, keep=False), keys]
        raise ValueError(
            f"{path.name} contains duplicated institution-years:\n"
            + duplicated.to_string(index=False)
        )

    expected_years = set(range(start_year, end_year + 1))
    observed_years = set(frame["analysis_year"])
    if observed_years != expected_years:
        raise ValueError(
            f"{path.name} has years {sorted(observed_years)}; expected "
            f"{sorted(expected_years)}."
        )

    annual_pairs = annual_data[keys].drop_duplicates()
    derived_pairs = frame[keys].drop_duplicates()
    pair_audit = annual_pairs.merge(derived_pairs, on=keys, how="outer", indicator=True)
    if not pair_audit["_merge"].eq("both").all():
        raise ValueError(
            f"{path.name} does not have exact institution-year coverage:\n"
            + pair_audit.loc[pair_audit["_merge"].ne("both")].to_string(index=False)
        )

    denominators = annual_data[
        keys
        + [
            "N_articles_attributed",
            "N_hyperprolific_authors",
            "N_hyperprolific_authors_MFA_or_CIS",
        ]
    ].merge(
        frame[
            keys
            + [
                "N_articles_attributed",
                "N_hyperprolific_authors",
                "N_hyperprolific_authors_MFA_or_CIS",
            ]
        ],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_annual", "_derived"),
    )
    comparison_columns = [
        "N_articles_attributed",
        "N_hyperprolific_authors",
        "N_hyperprolific_authors_MFA_or_CIS",
    ]
    mismatch = np.zeros(len(denominators), dtype=bool)
    for column in comparison_columns:
        mismatch |= denominators[f"{column}_annual"].ne(
            denominators[f"{column}_derived"]
        )
    if mismatch.any():
        raise ValueError(
            f"{path.name} does not reproduce annual denominators/author counts:\n"
            + denominators.loc[mismatch].to_string(index=False)
        )

    counts = frame["N_articles_by_HPA_MFA_or_CIS"]
    denominators_derived = frame["N_articles_attributed"]
    invalid_counts = counts.isna() | counts.lt(0) | counts.gt(denominators_derived)
    if invalid_counts.any():
        raise ValueError(f"{path.name} contains invalid article numerators.")
    invalid_rates = frame["HPA_MFA_or_CIS_article_rate"].notna() & ~frame[
        "HPA_MFA_or_CIS_article_rate"
    ].between(0, 1)
    if invalid_rates.any():
        raise ValueError(f"{path.name} contains rates outside [0, 1].")
    expected_rate = counts.div(denominators_derived.replace(0, np.nan))
    rate_mismatch = (
        frame["HPA_MFA_or_CIS_article_rate"] - expected_rate
    ).abs().gt(1e-9)
    if rate_mismatch.any():
        raise ValueError(
            f"{path.name}: HPA_MFA_or_CIS_article_rate is inconsistent with "
            "its numerator and denominator."
        )

    year_text = frame["analysis_year"].astype(str)
    expected_sources = {
        "source_hpa_flags_file": year_text + "_hyperprolific_author_flags.csv",
        "source_author_productivity_file": year_text + "_author_productivity.csv",
        "source_metrics_file": year_text + "_institution_year_metrics.csv",
    }
    bad_sources = np.zeros(len(frame), dtype=bool)
    for column, expected in expected_sources.items():
        bad_sources |= frame[column].astype(str).ne(expected)
    if bad_sources.any():
        raise ValueError(
            f"{path.name} does not identify the exact annual HPA/productivity source "
            "for every row."
        )
    methods = sorted(frame["derivation_method"].astype(str).unique())
    if methods != ["uniform_union_from_hpa_flags_and_author_productivity"]:
        raise ValueError(
            f"{path.name} does not use one uniform derivation method: {methods}"
        )

    merge_columns = keys + [
        "N_articles_by_HPA_MFA_or_CIS",
        "HPA_MFA_or_CIS_article_rate",
    ]
    merged = annual_data.merge(
        frame[merge_columns],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    validation = pd.DataFrame(
        [
            {
                "derived_metric_file": path.name,
                "rows": len(frame),
                "start_year": int(frame["analysis_year"].min()),
                "end_year": int(frame["analysis_year"].max()),
                "institution_year_coverage_exact": True,
                "annual_denominators_match": True,
                "annual_HPA_counts_match": True,
                "annual_HPA_MFA_or_CIS_counts_match": True,
                "article_numerator_bounds_valid": True,
                "rate_identity_valid": True,
                "derivation_method": methods[0],
                "derivation_script_versions": ";".join(
                    sorted(frame["script_version"].astype(str).unique())
                ),
            }
        ]
    )
    return merged, validation


def audit_and_merge_mapping(
    data: pd.DataFrame,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_ids = set(data["institution_id"])
    mapping_ids = set(mapping["institution_id"])
    missing_in_mapping = sorted(data_ids - mapping_ids)
    if missing_in_mapping:
        raise ValueError(
            "Data contain institution IDs absent from the mapping: "
            f"{missing_in_mapping}"
        )

    names = (
        data[["institution_id", "name", "analysis_year"]]
        .sort_values("analysis_year")
        .drop_duplicates("institution_id", keep="last")
        .rename(columns={"name": "data_name"})
    )
    name_audit = mapping[["institution_id", "name"]].merge(
        names[["institution_id", "data_name"]],
        on="institution_id",
        how="left",
    )
    name_audit["normalized_mapping_name"] = (
        name_audit["name"].str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    name_audit["normalized_data_name"] = (
        name_audit["data_name"]
        .fillna("")
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    name_audit["name_matches"] = (
        name_audit["normalized_mapping_name"] == name_audit["normalized_data_name"]
    )
    name_audit = name_audit.loc[
        ~name_audit["name_matches"],
        ["institution_id", "name", "data_name", "name_matches"],
    ]

    merge_columns = [
        "institution_id",
        "id-paper",
        "color_hex",
        "legend_label",
        "code_order",
    ]
    merged = data.merge(
        mapping[merge_columns],
        on="institution_id",
        how="left",
        validate="many_to_one",
    )
    return merged, name_audit


def ordered_unique(values: Iterable[str | None]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value is not None and value not in result:
            result.append(value)
    return result


def build_long_metric_data(data: pd.DataFrame, min_denominator: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    base_columns = [
        "institution_id",
        "name",
        "id-paper",
        "color_hex",
        "legend_label",
        "code_order",
        "analysis_year",
        "table_year",
        "source_file",
    ]

    for metric_order, metric in enumerate(METRICS, start=1):
        raw_columns = ordered_unique(
            [
                metric["value_col"],
                metric["count_col"],
                metric["denominator_col"],
                metric["eligibility_col"],
                metric["secondary_min_col"],
            ]
        )
        frame = data[base_columns + raw_columns].copy()
        frame["count"] = frame[metric["count_col"]]
        frame["denominator"] = frame[metric["denominator_col"]]
        frame["eligibility_denominator"] = frame[metric["eligibility_col"]]
        if metric["secondary_min_col"] is None:
            frame["secondary_denominator"] = np.nan
        else:
            frame["secondary_denominator"] = frame[metric["secondary_min_col"]]

        if metric["value_type"] == "rate":
            frame["value_raw"] = frame[metric["value_col"]]
        elif metric["value_type"] == "ratio_per_100":
            frame["value_raw"] = frame["count"].div(frame["denominator"].replace(0, np.nan))
        elif metric["value_type"] == "count":
            frame["value_raw"] = frame["count"]
        else:
            raise ValueError(f"Unknown value_type: {metric['value_type']}")
        if metric["value_type"] == "count":
            frame["value_plot"] = frame["value_raw"]
        else:
            frame["value_plot"] = 100.0 * frame["value_raw"]

        frame["metric_order"] = metric_order
        frame["metric_key"] = metric["key"]
        frame["metric_title"] = metric["title"]
        frame["metric_description"] = metric["description"]
        frame["value_type"] = metric["value_type"]
        frame["value_column"] = metric["value_col"] or "derived_count_divided_by_denominator"
        frame["count_column"] = metric["count_col"]
        frame["denominator_column"] = metric["denominator_col"]
        frame["eligibility_column"] = metric["eligibility_col"]
        frame["secondary_min_column"] = metric["secondary_min_col"] or ""
        frame["unit_label"] = metric["unit_label"]
        frame["display_kind"] = metric["display_kind"]
        frame["classification_basis"] = metric["classification_basis"]

        national = frame.loc[
            frame["institution_id"] == NATIONAL_ID,
            ["analysis_year", "value_raw", "value_plot", "count", "denominator"],
        ].rename(
            columns={
                "value_raw": "national_value_raw",
                "value_plot": "national_value_plot",
                "count": "national_count",
                "denominator": "national_denominator",
            }
        )
        frame = frame.merge(national, on="analysis_year", how="left", validate="many_to_one")
        frame["gap_display_units"] = frame["value_plot"] - frame["national_value_plot"]

        eligible = (
            frame["institution_id"].ne(NATIONAL_ID)
            & frame["eligibility_denominator"].ge(min_denominator)
            & frame["value_plot"].notna()
        )
        if metric["secondary_min_col"] is not None:
            eligible &= frame["secondary_denominator"].ge(1)
            rule = (
                f"{metric['eligibility_col']} >= {min_denominator} and "
                f"{metric['secondary_min_col']} >= 1"
            )
        else:
            rule = f"{metric['eligibility_col']} >= {min_denominator}"
        frame["eligibility_rule"] = rule
        frame["eligible"] = eligible
        frames.append(frame)

    long_data = pd.concat(frames, ignore_index=True)
    return long_data.sort_values(
        ["metric_order", "code_order", "institution_id", "analysis_year"]
    ).reset_index(drop=True)


def empty_distribution_stats() -> dict[str, float]:
    return {
        "minimum_value": np.nan,
        "q1_value": np.nan,
        "median_value": np.nan,
        "q3_value": np.nan,
        "maximum_value": np.nan,
        "iqr_value": np.nan,
        "robust_scale_iqr_over_1_349": np.nan,
    }


def classify_annual_distributions(
    long_data: pd.DataFrame,
    min_box_institutions: int,
    min_reference_institutions: int,
    elevated_d: float,
    extreme_d: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize distributions and identify annual robust-D deviations."""
    years = sorted(long_data["analysis_year"].unique())
    distribution_rows: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []

    for metric_order, metric in enumerate(METRICS, start=1):
        metric_data = long_data.loc[
            (long_data["metric_key"] == metric["key"]) & long_data["eligible"]
        ]
        for year in years:
            eligible = metric_data.loc[
                metric_data["analysis_year"] == year
            ].sort_values(["value_plot", "code_order", "institution_id"])
            zero_institutions = int(eligible["value_plot"].eq(0).sum())
            positive_institutions = int(eligible["value_plot"].gt(0).sum())

            if metric["classification_basis"] == "positive_eligible_institutions":
                reference = eligible.loc[eligible["value_plot"].gt(0)].copy()
            else:
                reference = eligible.copy()

            if metric["display_kind"] == "positive_box":
                box_basis = eligible.loc[eligible["value_plot"].gt(0)].copy()
            else:
                box_basis = eligible.copy()

            values = reference["value_plot"].dropna()
            box_values = box_basis["value_plot"].dropna()
            n_eligible = int(len(eligible))
            n_reference = int(len(values))
            n_reference_distinct = int(values.nunique())
            n_box = int(len(box_values))
            n_box_distinct = int(box_values.nunique())
            box_median = float(box_values.median()) if n_box else np.nan

            if n_reference:
                q1 = float(values.quantile(0.25))
                median = float(values.median())
                q3 = float(values.quantile(0.75))
                iqr = q3 - q1
                robust_scale = iqr / ROBUST_NORMAL_CONSISTENCY if iqr > 0 else np.nan
                stats = {
                    "minimum_value": float(values.min()),
                    "q1_value": q1,
                    "median_value": median,
                    "q3_value": q3,
                    "maximum_value": float(values.max()),
                    "iqr_value": iqr,
                    "robust_scale_iqr_over_1_349": robust_scale,
                }
            else:
                stats = empty_distribution_stats()
                median = np.nan
                iqr = np.nan
                robust_scale = np.nan

            box_drawn = bool(
                metric["display_kind"] in {"box", "positive_box"}
                and n_box >= min_box_institutions
                and n_box_distinct >= 2
            )
            reference_rule_applied = bool(
                n_reference >= min_reference_institutions
                and n_reference_distinct >= 2
                and pd.notna(iqr)
                and iqr > 0
            )
            evaluated = eligible.copy()
            if reference_rule_applied:
                evaluated["robust_D"] = (
                    evaluated["value_plot"] - median
                ) / robust_scale
                upper = evaluated["robust_D"].ge(elevated_d)
                if metric["lower_tail_evaluated"]:
                    lower = evaluated["robust_D"].le(-elevated_d)
                else:
                    lower = pd.Series(False, index=evaluated.index)
                signals = evaluated.loc[upper | lower].copy()
            else:
                evaluated["robust_D"] = np.nan
                signals = evaluated.iloc[0:0].copy()

            if not signals.empty:
                signals["deviation_side"] = np.where(
                    signals["robust_D"].ge(elevated_d), "upper", "lower"
                )
                signals["extreme_deviation"] = signals["robust_D"].abs().ge(
                    extreme_d
                )
                signals["deviation_strength"] = np.where(
                    signals["extreme_deviation"], "extreme", "elevated"
                )
                signals["eligible_institutions"] = n_eligible
                signals["reference_institutions"] = n_reference
                signals["reference_distinct_values"] = n_reference_distinct
                signals["box_institutions"] = n_box
                signals["box_distinct_values"] = n_box_distinct
                signals["box_drawn"] = box_drawn
                signals["reference_rule_applied"] = reference_rule_applied
                signals["lower_tail_evaluated"] = metric["lower_tail_evaluated"]
                signals["q1_value"] = stats["q1_value"]
                signals["median_value"] = stats["median_value"]
                signals["q3_value"] = stats["q3_value"]
                signals["iqr_value"] = stats["iqr_value"]
                signals["robust_scale_iqr_over_1_349"] = robust_scale
                signals["elevated_D_threshold"] = elevated_d
                signals["extreme_D_threshold"] = extreme_d
                signal_frames.append(signals)

            upper_signals = int(
                signals["deviation_side"].eq("upper").sum()
            ) if not signals.empty else 0
            lower_deviations = int(
                signals["deviation_side"].eq("lower").sum()
            ) if not signals.empty else 0
            extreme_deviations = int(
                signals["extreme_deviation"].sum()
            ) if not signals.empty else 0

            distribution_rows.append(
                {
                    "metric_order": metric_order,
                    "metric_key": metric["key"],
                    "metric_title": metric["title"],
                    "analysis_year": year,
                    "display_kind": metric["display_kind"],
                    "classification_basis": metric["classification_basis"],
                    "lower_tail_evaluated": metric["lower_tail_evaluated"],
                    "eligible_institutions": n_eligible,
                    "reference_institutions": n_reference,
                    "reference_distinct_values": n_reference_distinct,
                    "box_institutions": n_box,
                    "box_distinct_values": n_box_distinct,
                    "box_median_value": box_median,
                    "zero_institutions": zero_institutions,
                    "positive_institutions": positive_institutions,
                    "zero_share_pct": (
                        100.0 * zero_institutions / n_eligible if n_eligible else np.nan
                    ),
                    **stats,
                    "normal_consistency_constant": ROBUST_NORMAL_CONSISTENCY,
                    "elevated_D_threshold": elevated_d,
                    "extreme_D_threshold": extreme_d,
                    "box_drawn": box_drawn,
                    "reference_rule_applied": reference_rule_applied,
                    "annual_upper_signals": upper_signals,
                    "annual_lower_deviations": lower_deviations,
                    "annual_extreme_deviations": extreme_deviations,
                    "annual_robust_deviations": int(len(signals)),
                }
            )

    distribution = pd.DataFrame(distribution_rows)
    if signal_frames:
        signals = pd.concat(signal_frames, ignore_index=True)
    else:
        signals = pd.DataFrame(
            columns=[
                *long_data.columns,
                "robust_D",
                "deviation_side",
                "extreme_deviation",
                "deviation_strength",
                "eligible_institutions",
                "reference_institutions",
                "reference_distinct_values",
                "box_institutions",
                "box_distinct_values",
                "box_drawn",
                "reference_rule_applied",
                "lower_tail_evaluated",
                "q1_value",
                "median_value",
                "q3_value",
                "iqr_value",
                "robust_scale_iqr_over_1_349",
                "elevated_D_threshold",
                "extreme_D_threshold",
            ]
        )
    return distribution, signals


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.015,
        0.97,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        fontweight="bold",
    )


def validate_saved_figure(path: Path) -> None:
    """Fully reopen a generated figure before it is promoted to final output."""
    suffix = path.suffix.lower()
    if path.stat().st_size == 0:
        raise OSError(f"Generated an empty figure file: {path}")
    if suffix == ".png":
        # imread decodes the complete PNG and therefore catches truncated data.
        plt.imread(str(path))
    elif suffix == ".svg":
        ET.parse(path)
    elif suffix == ".pdf":
        with path.open("rb") as handle:
            header = handle.read(5)
            handle.seek(max(0, path.stat().st_size - 2048))
            trailer = handle.read()
        if header != b"%PDF-" or b"%%EOF" not in trailer:
            raise OSError(f"Generated an incomplete PDF file: {path}")


def save_figure_atomically(
    figure: plt.Figure,
    output_path: Path,
    **savefig_kwargs: object,
) -> None:
    """Save, validate, and atomically replace one figure-format output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=output_path.suffix,
        dir=str(output_path.parent),
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(temporary_path, **savefig_kwargs)
        validate_saved_figure(temporary_path)
        os.replace(str(temporary_path), str(output_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def symmetric_offsets(count: int, maximum: float) -> np.ndarray:
    if count <= 1:
        return np.zeros(count, dtype=float)
    return np.linspace(-maximum, maximum, count)


def style_boxplot(boxplot: dict[str, object]) -> None:
    for box in boxplot["boxes"]:
        box.set_facecolor("#D6D6D6")
        box.set_edgecolor("#777777")
        box.set_linewidth(0.65)
        box.set_alpha(0.72)
    for whisker in boxplot["whiskers"]:
        whisker.set_color("#777777")
        whisker.set_linewidth(0.65)
    for cap in boxplot["caps"]:
        cap.set_color("#777777")
        cap.set_linewidth(0.65)
    for median in boxplot["medians"]:
        median.set_color("#333333")
        median.set_linewidth(1.45)


def draw_box_panel_year(
    axis: plt.Axes,
    year: int,
    year_data: pd.DataFrame,
    stats: pd.Series,
    metric: dict[str, object],
    box_width: float,
) -> None:
    if year_data.empty:
        return
    values = year_data["value_plot"].to_numpy(dtype=float)
    box_drawn = bool(stats["box_drawn"])
    if box_drawn:
        boxplot = axis.boxplot(
            [values],
            positions=[year],
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            whis=(0, 100),
            manage_ticks=False,
            zorder=2,
        )
        style_boxplot(boxplot)

    if not box_drawn:
        year_data = year_data.sort_values(
            ["value_plot", "code_order", "institution_id"]
        )
        offsets = symmetric_offsets(len(year_data), 0.16)
        axis.scatter(
            year + offsets,
            year_data["value_plot"],
            s=11,
            color="#9B9B9B",
            edgecolor="white",
            linewidth=0.3,
            alpha=0.86,
            zorder=3,
        )

    if not box_drawn and pd.notna(stats["box_median_value"]):
        axis.hlines(
            float(stats["box_median_value"]),
            year - 0.19,
            year + 0.19,
            color="#3F3F3F",
            linewidth=1.4,
            zorder=4,
        )


def draw_positive_box_year(
    axis: plt.Axes,
    year: int,
    year_data: pd.DataFrame,
    stats: pd.Series,
    box_width: float,
) -> None:
    if year_data.empty:
        return
    positive = year_data.loc[year_data["value_plot"].gt(0)].sort_values(
        ["value_plot", "code_order", "institution_id"]
    )
    if not positive.empty:
        if bool(stats["box_drawn"]):
            boxplot = axis.boxplot(
                [positive["value_plot"].to_numpy(dtype=float)],
                positions=[year],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                whis=(0, 100),
                manage_ticks=False,
                zorder=2,
            )
            style_boxplot(boxplot)
        else:
            offsets = symmetric_offsets(len(positive), 0.20)
            axis.scatter(
                year + offsets,
                positive["value_plot"],
                s=13,
                color="#9B9B9B",
                edgecolor="white",
                linewidth=0.3,
                alpha=0.82,
                zorder=3,
            )
        if not bool(stats["box_drawn"]) and pd.notna(stats["box_median_value"]):
            axis.hlines(
                float(stats["box_median_value"]),
                year - 0.18,
                year + 0.18,
                color="#4A4A4A",
                linewidth=1.35,
                zorder=4,
            )

def draw_robust_deviations(
    axis: plt.Axes,
    metric_signals: pd.DataFrame,
) -> None:
    for year, year_signals in metric_signals.groupby("analysis_year", sort=True):
        year_signals = year_signals.sort_values(
            ["value_plot", "code_order", "institution_id"]
        ).copy()
        year_signals["x_plot"] = float(year) + symmetric_offsets(
            len(year_signals), 0.13
        )
        for row in year_signals.itertuples(index=False):
            extreme = bool(row.extreme_deviation)
            marker = "o" if row.deviation_side == "upper" else "v"
            axis.scatter(
                row.x_plot,
                row.value_plot,
                s=32.0 if extreme else 27.0,
                c=[row.color_hex],
                marker=marker,
                edgecolor="#111111" if extreme else "white",
                linewidth=1.05 if extreme else 0.5,
                alpha=0.99,
                zorder=8 if extreme else 7,
            )


def build_figure(
    long_data: pd.DataFrame,
    distribution: pd.DataFrame,
    signals: pd.DataFrame,
    mapping: pd.DataFrame,
    output_base: Path,
    start_year: int,
    end_year: int,
    box_width: float,
    x_tick_step: int,
    width: float,
    height: float,
    dpi: int,
) -> None:
    if output_utils.data_only():
        return
    plt.rcParams.update(
        {
            "font.family": "Source Sans 3",
            "font.size": 7.0,
            "axes.titlesize": 7.8,
            "axes.labelsize": 7.1,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    years = list(range(start_year, end_year + 1))
    ticks = list(range(start_year, end_year + 1, x_tick_step))
    if not ticks:
        ticks = [start_year]

    figure = plt.figure(figsize=(width, height))
    grid = figure.add_gridspec(3, 2)
    axes_flat = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[2, 0]),
        figure.add_subplot(grid[2, 1]),
    ]
    panel_labels = [f"({letter})" for letter in "abcdef"]
    signal_ids = set(signals["institution_id"]) if not signals.empty else set()

    for axis, metric, panel_label in zip(axes_flat, METRICS, panel_labels):
        metric_key = metric["key"]
        metric_data = long_data.loc[long_data["metric_key"] == metric_key]
        institutional = metric_data.loc[metric_data["eligible"]].copy()
        national = metric_data.loc[metric_data["institution_id"] == NATIONAL_ID]
        metric_distribution = distribution.loc[
            distribution["metric_key"] == metric_key
        ].set_index("analysis_year")

        for year in years:
            year_data = institutional.loc[institutional["analysis_year"] == year]
            if year not in metric_distribution.index:
                continue
            stats = metric_distribution.loc[year]
            if metric["display_kind"] == "positive_box":
                draw_positive_box_year(
                    axis,
                    year,
                    year_data,
                    stats,
                    box_width,
                )
            else:
                draw_box_panel_year(
                    axis,
                    year,
                    year_data,
                    stats,
                    metric,
                    box_width,
                )

        national_series = national.set_index("analysis_year")["value_plot"].reindex(years)
        axis.plot(
            years,
            national_series,
            color="#111111",
            linewidth=1.9,
            zorder=6,
        )

        metric_signals = signals.loc[signals["metric_key"] == metric_key]
        if not metric_signals.empty:
            draw_robust_deviations(axis, metric_signals)

        if metric_key == "HAO":
            axis.set_yscale("symlog", linthresh=1.0, linscale=1.0, base=10)
            hao_ticks = [0, 0.5, 1, 2, 5, 10, 20, 50]
            axis.set_yticks(hao_ticks)
            axis.set_yticklabels(["0", "0.5", "1", "2", "5", "10", "20", "50"])
        elif metric["ytick"] is not None:
            axis.yaxis.set_major_locator(MultipleLocator(float(metric["ytick"])))

        axis.set_title(str(metric["title"]), pad=4.2)
        axis.set_ylabel(str(metric["unit_label"]), labelpad=2.0)
        axis.set_xlim(start_year - 0.55, end_year + 0.55)
        axis.set_ylim(*metric["ylim"])
        axis.set_xticks(ticks)
        axis.set_xticklabels([str(year) for year in ticks], rotation=45, ha="right")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.78)
        # Positional visibility is compatible with Matplotlib 3.3; the
        # ``visible=`` keyword was standardized in later releases.
        axis.grid(False, axis="x")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#555555")
        axis.spines["bottom"].set_color("#555555")
        axis.tick_params(length=2.3, width=0.55, color="#555555")
        add_panel_label(axis, panel_label)

    for axis in axes_flat:
        axis.tick_params(axis="x", labelbottom=True)

    style_handles = [
        Line2D(
            [0],
            [0],
            color="#111111",
            linewidth=1.9,
            label="National",
        ),
        Patch(
            facecolor="#D6D6D6",
            alpha=0.72,
            edgecolor="#777777",
            label="University IQR; whiskers = min-max",
        ),
        Line2D(
            [0],
            [0],
            color="#9B9B9B",
            marker="o",
            linestyle="None",
            markersize=3.5,
            label="Sparse year (points; no box)",
        ),
        Line2D(
            [0],
            [0],
            color="#666666",
            marker="o",
            markerfacecolor="#888888",
            linestyle="None",
            markersize=4,
            markeredgecolor="white",
            label="Upper robust signal (D ≥ 2)",
        ),
        Line2D(
            [0],
            [0],
            color="#666666",
            marker="v",
            markerfacecolor="#888888",
            linestyle="None",
            markersize=4,
            markeredgecolor="white",
            label="Lower robust deviation (D ≤ −2)",
        ),
        Line2D(
            [0],
            [0],
            color="#111111",
            marker="o",
            markerfacecolor="#BBBBBB",
            linestyle="None",
            markersize=4.5,
            markeredgewidth=1.0,
            label="Black outline: |D| ≥ 3",
        ),
    ]
    figure.legend(
        handles=style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
        labelspacing=0.45,
    )

    signal_mapping = mapping.loc[mapping["institution_id"].isin(signal_ids)].sort_values(
        ["code_order", "id-paper"]
    )
    institution_handles = [
        Line2D(
            [0],
            [0],
            color=row.color_hex,
            marker="o",
            linestyle="None",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=row.legend_label,
        )
        for row in signal_mapping.itertuples(index=False)
    ]
    if institution_handles:
        figure.legend(
            handles=institution_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.006),
            ncol=min(10, len(institution_handles)),
            frameon=False,
            title="Institutions with annual robust deviations",
            title_fontsize=6.4,
            handlelength=0.8,
            handletextpad=0.2,
            columnspacing=0.65,
            labelspacing=0.32,
        )

    legend_rows = max(1, int(np.ceil(len(institution_handles) / 10)))
    bottom = 0.105 + 0.019 * legend_rows
    # Figure.supxlabel() was introduced in Matplotlib 3.4. Figure.text() gives
    # the same figure-level label while remaining compatible with 3.3.
    figure.text(
        0.5,
        bottom - 0.055,
        "Year",
        ha="center",
        va="bottom",
        fontsize=7.4,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.99,
        top=0.895,
        bottom=bottom,
        wspace=0.25,
        hspace=0.40,
    )

    figure_style.finish(figure, output_base)
    for suffix in (".pdf", ".svg"):
        save_figure_atomically(
            figure,
            output_base.with_suffix(suffix),
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )
    save_figure_atomically(
        figure,
        output_base.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    plt.close(figure)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    if not output_utils.keep(path):
        return
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def validate_required_annual_inputs(
    input_dir: Path,
    start_year: int,
    end_year: int,
) -> None:
    required = [
        input_dir / f"{year}_{suffix}"
        for year in range(start_year, end_year + 1)
        for suffix in REQUIRED_ANNUAL_SUFFIXES
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The annual input directory is missing required files:\n"
            + "\n".join(missing)
        )
    print(f"Validated {len(required)} required annual input files in {input_dir}")


def derive_hpa_article_metric_file(
    input_dir: Path,
    output_dir: Path,
    start_year: int,
    end_year: int,
) -> Path:
    """Derive the HPA-MFA/CIS article metric when no audit CSV is supplied."""
    try:
        from derive_hpa_mfa_cis_article_share import derive_year
    except ImportError as error:
        raise ImportError(
            "Automatic HPA-MFA/CIS derivation requires "
            "derive_hpa_mfa_cis_article_share.py in the same directory as this script."
        ) from error

    frames = [
        derive_year(input_dir, year)
        for year in range(start_year, end_year + 1)
    ]
    derived = pd.concat(frames, ignore_index=True).sort_values(
        ["analysis_year", "institution_id"]
    )
    derived_path = (
        output_dir / "derived" / "hpa_mfa_cis_article_share_uniform.csv"
    )
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(derived, derived_path)
    print(f"Automatically derived HPA-MFA/CIS article metric: {derived_path}")
    return derived_path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    mapping_path = Path(args.mapping_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.is_dir():
        raise NotADirectoryError(f'Input directory does not exist: {input_dir}')
    validate_required_annual_inputs(input_dir, args.start_year, args.end_year)
    if not mapping_path.is_file():
        raise FileNotFoundError(f'Mapping file does not exist: {mapping_path}')
    hpa_article_path = Path(args.hpa_article_file) if args.hpa_article_file else derive_hpa_article_metric_file(input_dir, output_dir, args.start_year, args.end_year)
    if not hpa_article_path.is_file():
        raise FileNotFoundError(f'Derived HPA article file does not exist: {hpa_article_path}')
    mapping = load_mapping(mapping_path, args.mapping_encoding)
    data, loaded_files, ignored_files, off_year_rows = load_annual_data(input_dir, args.start_year, args.end_year)
    data, hpa_article_validation = load_hpa_article_metric(hpa_article_path, data, args.start_year, args.end_year)
    data, mapping_name_mismatches = audit_and_merge_mapping(data, mapping)
    long_data = build_long_metric_data(data, args.min_denominator)
    distribution, signals = classify_annual_distributions(long_data, min_box_institutions=args.min_box_institutions, min_reference_institutions=args.min_reference_institutions, elevated_d=args.elevated_d, extreme_d=args.extreme_d)
    distribution_columns = ['metric_key', 'analysis_year', 'eligible_institutions', 'reference_institutions', 'reference_distinct_values', 'box_institutions', 'box_distinct_values', 'box_median_value', 'zero_institutions', 'positive_institutions', 'zero_share_pct', 'q1_value', 'median_value', 'q3_value', 'iqr_value', 'robust_scale_iqr_over_1_349', 'box_drawn', 'reference_rule_applied', 'annual_upper_signals', 'annual_lower_deviations', 'annual_extreme_deviations', 'annual_robust_deviations']
    long_data = long_data.merge(distribution[distribution_columns], on=['metric_key', 'analysis_year'], how='left', validate='many_to_one')
    long_data['robust_D'] = np.where(long_data['eligible'] & long_data['reference_rule_applied'], (long_data['value_plot'] - long_data['median_value']) / long_data['robust_scale_iqr_over_1_349'], np.nan)
    signal_lookup = signals[['metric_key', 'analysis_year', 'institution_id', 'deviation_side', 'deviation_strength', 'extreme_deviation']].copy()
    signal_lookup['annual_robust_deviation'] = True
    long_data = long_data.merge(signal_lookup, on=['metric_key', 'analysis_year', 'institution_id'], how='left', validate='one_to_one')
    long_data['annual_robust_deviation'] = long_data['annual_robust_deviation'].eq(True)
    long_data['extreme_deviation'] = long_data['extreme_deviation'].eq(True)
    prefix = args.output_stem
    write_csv(long_data, output_dir / f'{prefix}_data_used.csv')
    write_csv(signals, output_dir / f'{prefix}_annual_robust_deviations.csv')
    print(f'Calculated exposure sources in {output_dir}')


if __name__ == "__main__":
    main()
