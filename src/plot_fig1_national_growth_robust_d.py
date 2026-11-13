#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1. Attributed publication output and institutional growth, 2011-2025.

This version retains the contiguous-year corrections from the previous
combined script, replaces the institutional line panels with annual
distributional boxplots, combines national and institutional growth in one
panel, and adds descriptive robust-D screening of institutional growth.

Core rules
----------
1. The analysis year is the Scopus query year encoded in each annual filename:
   YYYY_institution_year_metrics.csv. The record year derived from coverDate is
   retained only for audit purposes.
2. Institutional year series are expanded to a complete annual grid before
   growth is calculated. Missing institution-year rows remain NaN by default,
   so lines are interrupted and no non-consecutive observations are treated as
   year-over-year changes.
3. A missing row can be interpreted as zero only with the explicit option
   --missing-institution-year zero. Growth from a previous value of zero is
   undefined and is therefore not plotted.
4. Colors are read exclusively from the mapping file and must use #RRGGBB.
5. Main y-axis limits are validated before plotting. The default policy raises
   an error rather than silently clipping data.
6. Institutional-output boxplots are descriptive and do not classify
   publication volume as an exposure signal.
7. Institutional growth is screened with
   D = (growth - eligible annual median) / (eligible annual IQR / 1.349).
   The main robust reference is restricted to valid consecutive-year
   comparisons with at least 50 publications in the preceding year. Both
   tails are descriptive: D >= 2 identifies an upper deviation, D <= -2 a
   lower deviation, and |D| >= 3 marks either direction as strong.
8. The national aggregate is excluded from all institutional boxes and robust
   references. It is plotted as a separate black line.
9. The three panels are exported as one PDF, SVG and 300-dpi PNG figure.

Required annual files
---------------------
    2011_institution_year_metrics.csv
    ...
    2025_institution_year_metrics.csv

Required mapping columns
------------------------
    institution_id, name, id-paper, color_hex

Optional mapping columns
------------------------
    line_width, legend_label

Main outputs
------------
    figure1_output_and_robust_d_growth.pdf
    figure1_output_and_robust_d_growth.svg
    figure1_output_and_robust_d_growth.png
    fig1_data_used.csv
    fig1_growth_robust_d_annual_reference.csv
    fig1_growth_robust_d_signals.csv
    fig1_growth_robust_d_recurrence.csv
    fig1_growth_robust_d_sensitivity_summary.csv
    fig1_growth_robust_d_sensitivity_signals.csv
    fig1_output_distribution.csv
    fig1_growth_distribution_all_valid.csv
    section_4_1_annual_summary.csv
    section_4_1_key_statistics.csv
    fig1_growth_robust_d_run_parameters.csv
    fig1_missing_institution_years.csv
    fig1_nonconsecutive_observed_rows.csv
    fig1_undefined_growth_rows.csv
    fig1_axis_range_audit.csv
    fig1_loaded_metric_files.csv
    fig1_ignored_metric_like_files.csv
    fig1_off_year_rows_from_annual_files.csv
    fig1_year_source_audit.csv
    fig1_duplicate_rows_within_annual_files.csv

SCRIPT_VERSION = 4.0-robust-D-growth-boxplots-2026-08-24
"""

from __future__ import annotations

import figure_style
import output_utils

import argparse
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator


SCRIPT_VERSION = "4.0-robust-D-growth-boxplots-2026-08-24"

NATIONAL_ID = "NATIONAL"
DEFAULT_NATIONAL_WIDTH = 2.6
DEFAULT_INST_WIDTH = 0.9
ROBUST_NORMAL_CONSISTENCY = 1.349

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

Y_AXIS_SETTINGS = {
    "national_output": {"ymin": 0.0, "ymax": 9000.0, "ystep": 1000.0},
    "institutional_output": {"ymin": 0.0, "ymax": 850.0, "ystep": 100.0},
    "combined_growth": {"ymin": -100.0, "ymax": 1000.0, "ystep": 100.0},
}

INSET_SETTINGS = {
    "start_year": 2019,
    "ymin": -100.0,
    "ymax": 220.0,
    "ystep": 100.0,
}

PANEL_TITLES = {
    "national_output": "National attributed output",
    "institutional_output": "Institutional attributed output",
    "combined_growth": "Year-on-year growth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the three-panel Figure 1 with output boxplots, combined "
            "national/institutional growth and robust-D growth deviations."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing exact annual YYYY_institution_year_metrics.csv files.",
    )
    parser.add_argument(
        "--mapping-file",
        required=True,
        help="CSV containing institution IDs, manuscript codes and colors.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for figures and audits.")
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--mapping-encoding", default="latin1")
    parser.add_argument(
        "--missing-institution-year",
        choices=("nan", "zero"),
        default="nan",
        help=(
            "Interpret absent institution-year rows as missing (default) or zero. "
            "Use zero only after confirming that annual files omit true zeros."
        ),
    )
    parser.add_argument(
        "--axis-policy",
        choices=("error", "warn"),
        default="error",
        help="Action when a main-panel value falls outside its configured y-axis.",
    )
    parser.add_argument(
        "--min-growth-denominator",
        type=float,
        default=50.0,
        help=(
            "Minimum preceding-year institutional output used to construct "
            "and evaluate the primary robust-D growth reference (default: 50)."
        ),
    )
    parser.add_argument(
        "--min-reference-institutions",
        type=int,
        default=8,
        help="Minimum eligible institutions required for annual D (default: 8).",
    )
    parser.add_argument(
        "--elevated-d",
        type=float,
        default=2.0,
        help="Absolute D threshold for an elevated descriptive deviation.",
    )
    parser.add_argument(
        "--strong-d",
        type=float,
        default=3.0,
        help="Absolute D threshold for a strong descriptive deviation.",
    )
    parser.add_argument(
        "--sensitivity-growth-denominators",
        default="1,10,50",
        help=(
            "Comma-separated preceding-output thresholds for denominator "
            "sensitivity tables (default: 1,10,50)."
        ),
    )
    parser.add_argument("--box-width", type=float, default=0.58)
    parser.add_argument("--figure-width", type=float, default=7.5)
    parser.add_argument("--figure-height", type=float, default=7.7)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--output-stem",
        default="figure1_output_and_robust_d_growth",
    )
    args = parser.parse_args()

    if args.end_year <= args.start_year:
        parser.error("--end-year must be greater than --start-year.")
    if args.figure_width <= 0 or args.figure_height <= 0:
        parser.error("Figure dimensions must be positive.")
    if args.dpi < 150:
        parser.error("--dpi must be at least 150; 300 is recommended for submission.")
    if args.min_growth_denominator <= 0:
        parser.error("--min-growth-denominator must be positive.")
    if args.min_reference_institutions < 4:
        parser.error("--min-reference-institutions must be at least 4.")
    if args.elevated_d <= 0 or args.strong_d <= args.elevated_d:
        parser.error("Require 0 < --elevated-d < --strong-d.")
    try:
        sensitivity = [
            float(value.strip())
            for value in args.sensitivity_growth_denominators.split(",")
            if value.strip()
        ]
    except ValueError:
        parser.error("--sensitivity-growth-denominators must contain numbers.")
    if not sensitivity or any(value <= 0 for value in sensitivity):
        parser.error("Sensitivity growth denominators must be positive.")
    if args.min_growth_denominator not in sensitivity:
        sensitivity.append(float(args.min_growth_denominator))
    args.sensitivity_growth_denominators = sorted(set(sensitivity))
    return args


def is_invalid_text(value: object) -> bool:
    return str(value).strip().lower() in INVALID_TEXT_VALUES


def parse_idpaper_order(value: object) -> int:
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else 999999


def join_unique(values: Iterable[object]) -> str:
    clean = sorted({str(value).strip() for value in values if not is_invalid_text(value)})
    return " | ".join(clean)


def fmt_thousands(value: float, _position: int) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.1f}"


def fmt_percent(value: float, _position: int) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}%"
    return f"{value:.1f}%"


def load_mapping(mapping_path: Path, encoding: str) -> pd.DataFrame:
    mapping = pd.read_csv(
        mapping_path,
        encoding=encoding,
        dtype=str,
        keep_default_na=False,
    )
    mapping.columns = [str(column).strip() for column in mapping.columns]

    required = {"institution_id", "name", "id-paper", "color_hex"}
    missing_columns = required - set(mapping.columns)
    if missing_columns:
        raise ValueError(
            f"Mapping file is missing required columns: {sorted(missing_columns)}"
        )

    for column in ("institution_id", "name", "id-paper", "color_hex"):
        mapping[column] = mapping[column].astype(str).str.strip()

    invalid_required = []
    for column in ("institution_id", "name", "id-paper", "color_hex"):
        bad = mapping[column].map(is_invalid_text)
        if bad.any():
            invalid_required.append(
                {"column": column, "rows": (mapping.index[bad] + 2).tolist()}
            )
    if invalid_required:
        raise ValueError(f"Mapping contains empty or invalid required values: {invalid_required}")

    duplicate_ids = mapping.loc[
        mapping["institution_id"].duplicated(keep=False), "institution_id"
    ].tolist()
    if duplicate_ids:
        raise ValueError(
            f"Mapping contains duplicated institution_id values: {sorted(set(duplicate_ids))}"
        )

    duplicate_codes = mapping.loc[
        mapping["id-paper"].duplicated(keep=False), "id-paper"
    ].tolist()
    if duplicate_codes:
        raise ValueError(
            f"Mapping contains duplicated id-paper values: {sorted(set(duplicate_codes))}"
        )

    valid_color = mapping["color_hex"].str.fullmatch(r"#[0-9A-Fa-f]{6}")
    if not valid_color.all():
        bad = mapping.loc[~valid_color, ["institution_id", "color_hex"]]
        raise ValueError(
            "Every color_hex must use #RRGGBB. Invalid values:\n"
            + bad.to_string(index=False)
        )
    mapping["color_hex"] = mapping["color_hex"].str.upper()

    if "line_width" not in mapping.columns:
        mapping["line_width"] = ""
    widths = []
    for row in mapping.itertuples(index=False):
        raw = str(getattr(row, "line_width")).strip()
        institution_id = str(getattr(row, "institution_id"))
        if is_invalid_text(raw):
            width = (
                DEFAULT_NATIONAL_WIDTH
                if institution_id == NATIONAL_ID
                else DEFAULT_INST_WIDTH
            )
        else:
            try:
                width = float(raw)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid line_width for {institution_id}: {raw!r}"
                ) from exc
            if not np.isfinite(width) or width <= 0:
                raise ValueError(
                    f"line_width must be positive for {institution_id}: {raw!r}"
                )
        widths.append(width)
    mapping["line_width"] = widths

    if "legend_label" not in mapping.columns:
        mapping["legend_label"] = ""
    mapping["legend_label"] = mapping["legend_label"].astype(str).str.strip()
    blank_legend = mapping["legend_label"].map(is_invalid_text)
    mapping.loc[blank_legend, "legend_label"] = mapping.loc[blank_legend, "id-paper"]

    if (mapping["institution_id"] == NATIONAL_ID).sum() != 1:
        raise ValueError(
            f"Mapping must contain exactly one {NATIONAL_ID!r} institution_id row."
        )
    return mapping


def discover_annual_metric_files(
    input_dir: Path, start_year: int, end_year: int
) -> Tuple[List[Path], pd.DataFrame, pd.DataFrame]:
    exact_files: List[Path] = []
    loaded_rows = []
    ignored_rows = []

    for path in sorted(input_dir.glob("*.csv")):
        match = re.fullmatch(r"(\d{4})_institution_year_metrics\.csv", path.name)
        if match:
            query_year = int(match.group(1))
            if start_year <= query_year <= end_year:
                exact_files.append(path)
                loaded_rows.append({"source_file": path.name, "query_year": query_year})
            else:
                ignored_rows.append(
                    {"source_file": path.name, "reason": "outside_requested_year_range"}
                )
        elif "institution_year_metrics" in path.name:
            ignored_rows.append(
                {
                    "source_file": path.name,
                    "reason": "not_exact_annual_filename_pattern",
                }
            )

    loaded_years = {int(path.name[:4]) for path in exact_files}
    expected_years = set(range(start_year, end_year + 1))
    missing_years = sorted(expected_years - loaded_years)
    if missing_years:
        raise FileNotFoundError(
            "Missing required annual metric files for query years: "
            f"{missing_years}. Expected exact names YYYY_institution_year_metrics.csv."
        )

    return (
        exact_files,
        pd.DataFrame(loaded_rows, columns=["source_file", "query_year"]),
        pd.DataFrame(ignored_rows, columns=["source_file", "reason"]),
    )


def load_annual_metrics(
    files: List[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows = []
    off_year_rows = []
    audit_rows = []

    for path in files:
        query_year = int(path.name[:4])
        frame = pd.read_csv(path, dtype={"institution_id": str})
        required = {"institution_id", "name", "N_articles_attributed"}
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{path.name} is missing required columns: {sorted(missing_columns)}"
            )

        frame["institution_id"] = frame["institution_id"].astype(str).str.strip()
        if frame["institution_id"].map(is_invalid_text).any():
            rows = (frame.index[frame["institution_id"].map(is_invalid_text)] + 2).tolist()
            raise ValueError(f"{path.name} has invalid institution_id values at rows {rows}")

        numeric_output = pd.to_numeric(frame["N_articles_attributed"], errors="coerce")
        if numeric_output.isna().any():
            rows = (frame.index[numeric_output.isna()] + 2).tolist()
            raise ValueError(
                f"{path.name} has nonnumeric N_articles_attributed values at rows {rows}"
            )
        if (numeric_output < 0).any():
            rows = (frame.index[numeric_output < 0] + 2).tolist()
            raise ValueError(
                f"{path.name} has negative N_articles_attributed values at rows {rows}"
            )
        if not np.allclose(numeric_output, np.round(numeric_output)):
            raise ValueError(f"{path.name} contains noninteger article counts.")

        frame["N_articles_attributed"] = numeric_output.astype(int)
        frame["source_file"] = path.name
        frame["analysis_year"] = query_year

        if "year" in frame.columns:
            frame["record_year"] = pd.to_numeric(frame["year"], errors="coerce")
            off_mask = frame["record_year"].notna() & frame["record_year"].ne(query_year)
            if off_mask.any():
                off_year_rows.append(frame.loc[off_mask].copy())
            n_off_year = int(off_mask.sum())
        else:
            frame["record_year"] = pd.NA
            n_off_year = pd.NA

        audit_rows.append(
            {
                "source_file": path.name,
                "query_year": query_year,
                "n_rows_total": len(frame),
                "n_rows_off_year": n_off_year,
            }
        )
        all_rows.append(frame)

    raw = pd.concat(all_rows, ignore_index=True)

    duplicate_audit = (
        raw.groupby(["source_file", "analysis_year", "institution_id"], as_index=False)
        .agg(
            n_rows=("institution_id", "size"),
            summed_articles=("N_articles_attributed", "sum"),
            source_names=("name", join_unique),
        )
    )
    duplicate_audit = duplicate_audit[duplicate_audit["n_rows"] > 1].copy()

    metrics = (
        raw.groupby(["institution_id", "analysis_year"], as_index=False)
        .agg(
            N_articles_attributed=("N_articles_attributed", "sum"),
            source_names=("name", join_unique),
        )
        .rename(columns={"analysis_year": "year"})
    )

    off_year = (
        pd.concat(off_year_rows, ignore_index=True)
        if off_year_rows
        else pd.DataFrame(columns=raw.columns)
    )
    year_audit = pd.DataFrame(audit_rows)
    return metrics, off_year, year_audit, duplicate_audit


def prepare_plot_data(
    metrics: pd.DataFrame,
    mapping: pd.DataFrame,
    start_year: int,
    end_year: int,
    missing_policy: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    years = list(range(start_year, end_year + 1))

    metric_ids = set(metrics["institution_id"].astype(str))
    mapping_ids = set(mapping["institution_id"].astype(str))
    unmapped_ids = sorted(metric_ids - mapping_ids)
    if unmapped_ids:
        raise ValueError(f"Metric files contain institution IDs absent from mapping: {unmapped_ids}")

    national = metrics[metrics["institution_id"] == NATIONAL_ID].copy()
    if national["year"].duplicated().any():
        raise ValueError("National reference contains duplicated years after aggregation.")
    missing_national_years = sorted(set(years) - set(national["year"].astype(int)))
    if missing_national_years:
        raise ValueError(
            "National reference must contain every requested year. Missing years: "
            f"{missing_national_years}"
        )

    national = national[national["year"].isin(years)].copy()
    national["originally_observed"] = True
    national["missing_output_cell"] = False

    institution_ids = mapping.loc[
        mapping["institution_id"] != NATIONAL_ID, "institution_id"
    ].tolist()
    complete_index = pd.MultiIndex.from_product(
        [institution_ids, years], names=["institution_id", "year"]
    )
    institutional = (
        metrics[metrics["institution_id"] != NATIONAL_ID]
        .set_index(["institution_id", "year"])
        .reindex(complete_index)
        .reset_index()
    )
    institutional["originally_observed"] = institutional[
        "N_articles_attributed"
    ].notna()
    institutional["missing_output_cell"] = ~institutional["originally_observed"]

    missing_audit = institutional.loc[
        institutional["missing_output_cell"],
        ["institution_id", "year"],
    ].copy()
    missing_audit["applied_policy"] = missing_policy

    if missing_policy == "zero":
        institutional["N_articles_attributed"] = institutional[
            "N_articles_attributed"
        ].fillna(0.0)
        institutional["source_names"] = institutional["source_names"].fillna("")

    data = pd.concat([national, institutional], ignore_index=True, sort=False)
    data = data.merge(
        mapping[
            [
                "institution_id",
                "name",
                "id-paper",
                "color_hex",
                "line_width",
                "legend_label",
            ]
        ],
        on="institution_id",
        how="left",
        validate="many_to_one",
    )
    data["missing_policy"] = missing_policy
    data = data.sort_values(["institution_id", "year"]).reset_index(drop=True)

    data["previous_year"] = data.groupby("institution_id")["year"].shift()
    data["previous_output"] = data.groupby("institution_id")[
        "N_articles_attributed"
    ].shift()
    data["annual_growth_pct"] = np.nan
    data["growth_status"] = "not_evaluated"

    first_year = data["year"] == start_year
    current_missing = data["N_articles_attributed"].isna() & ~first_year
    previous_missing = data["previous_output"].isna() & ~first_year & ~current_missing
    previous_zero = (
        data["previous_output"].eq(0)
        & ~first_year
        & ~current_missing
        & ~previous_missing
    )
    valid_growth = (
        ~first_year
        & ~current_missing
        & ~previous_missing
        & ~previous_zero
        & data["previous_output"].gt(0)
        & data["previous_year"].eq(data["year"] - 1)
    )

    data.loc[first_year, "growth_status"] = "first_analysis_year"
    data.loc[current_missing, "growth_status"] = "missing_current_year"
    data.loc[previous_missing, "growth_status"] = "missing_previous_year"
    data.loc[previous_zero, "growth_status"] = "undefined_previous_zero"
    data.loc[valid_growth, "annual_growth_pct"] = (
        data.loc[valid_growth, "N_articles_attributed"]
        / data.loc[valid_growth, "previous_output"]
        - 1.0
    ) * 100.0
    data.loc[valid_growth, "growth_status"] = "computed_consecutive_years"

    unresolved = data["growth_status"] == "not_evaluated"
    if unresolved.any():
        bad = data.loc[
            unresolved,
            ["institution_id", "year", "previous_year", "previous_output"],
        ]
        raise RuntimeError(
            "Unexpected growth rows remained unevaluated:\n" + bad.to_string(index=False)
        )

    return data, missing_audit


def distribution_stats(values: pd.Series) -> dict:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return {
            "n": 0,
            "n_distinct": 0,
            "minimum": np.nan,
            "q1": np.nan,
            "median": np.nan,
            "q3": np.nan,
            "maximum": np.nan,
            "iqr": np.nan,
            "robust_scale_iqr_over_1_349": np.nan,
        }
    q1 = float(clean.quantile(0.25))
    median = float(clean.median())
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    return {
        "n": int(len(clean)),
        "n_distinct": int(clean.nunique()),
        "minimum": float(clean.min()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "maximum": float(clean.max()),
        "iqr": iqr,
        "robust_scale_iqr_over_1_349": (
            iqr / ROBUST_NORMAL_CONSISTENCY if iqr > 0 else np.nan
        ),
    }


def classify_growth_robust_d(
    institutional: pd.DataFrame,
    start_year: int,
    end_year: int,
    minimum_previous_output: float,
    min_reference_institutions: int,
    elevated_d: float,
    strong_d: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate annual robust-D growth deviations for one denominator rule."""
    evaluated_data = institutional.copy()
    evaluated_data["growth_reference_eligible"] = (
        evaluated_data["growth_status"].eq("computed_consecutive_years")
        & evaluated_data["annual_growth_pct"].notna()
        & evaluated_data["previous_output"].ge(minimum_previous_output)
    )
    evaluated_data["growth_D"] = np.nan
    evaluated_data["growth_deviation_side"] = ""
    evaluated_data["growth_deviation_strength"] = ""
    evaluated_data["growth_elevated_deviation"] = False
    evaluated_data["growth_strong_deviation"] = False

    reference_rows = []
    signal_frames = []
    for year in range(start_year + 1, end_year + 1):
        eligible_mask = (
            evaluated_data["year"].eq(year)
            & evaluated_data["growth_reference_eligible"]
        )
        eligible = evaluated_data.loc[eligible_mask].copy()
        stats = distribution_stats(eligible["annual_growth_pct"])
        reference_applied = bool(
            stats["n"] >= min_reference_institutions
            and stats["n_distinct"] >= 2
            and pd.notna(stats["iqr"])
            and stats["iqr"] > 0
        )
        if reference_applied:
            d_values = (
                eligible["annual_growth_pct"] - stats["median"]
            ) / stats["robust_scale_iqr_over_1_349"]
            evaluated_data.loc[eligible.index, "growth_D"] = d_values
            elevated = d_values.abs().ge(elevated_d)
            strong = d_values.abs().ge(strong_d)
            evaluated_data.loc[eligible.index, "growth_elevated_deviation"] = elevated
            evaluated_data.loc[eligible.index, "growth_strong_deviation"] = strong
            sides = np.where(d_values.ge(elevated_d), "upper", "")
            sides = np.where(d_values.le(-elevated_d), "lower", sides)
            strengths = np.where(strong, "strong", np.where(elevated, "elevated", ""))
            evaluated_data.loc[eligible.index, "growth_deviation_side"] = sides
            evaluated_data.loc[eligible.index, "growth_deviation_strength"] = strengths

            signals = evaluated_data.loc[
                eligible.index[elevated],
                [
                    "institution_id",
                    "id-paper",
                    "name",
                    "year",
                    "N_articles_attributed",
                    "previous_output",
                    "annual_growth_pct",
                    "growth_D",
                    "growth_deviation_side",
                    "growth_deviation_strength",
                    "growth_strong_deviation",
                    "color_hex",
                ],
            ].copy()
            if not signals.empty:
                signals["minimum_previous_output"] = minimum_previous_output
                signals["reference_institutions"] = stats["n"]
                signals["reference_median_growth_pct"] = stats["median"]
                signals["reference_iqr_growth_pct"] = stats["iqr"]
                signals["robust_scale_iqr_over_1_349"] = stats[
                    "robust_scale_iqr_over_1_349"
                ]
                signals["elevated_D_threshold"] = elevated_d
                signals["strong_D_threshold"] = strong_d
                signal_frames.append(signals)

        year_signals = evaluated_data.loc[
            evaluated_data["year"].eq(year)
            & evaluated_data["growth_elevated_deviation"]
        ]
        reference_rows.append(
            {
                "year": year,
                "minimum_previous_output": minimum_previous_output,
                **stats,
                "minimum_reference_institutions": min_reference_institutions,
                "reference_applied": reference_applied,
                "upper_deviations": int(
                    year_signals["growth_deviation_side"].eq("upper").sum()
                ),
                "lower_deviations": int(
                    year_signals["growth_deviation_side"].eq("lower").sum()
                ),
                "elevated_deviations_total": int(len(year_signals)),
                "strong_deviations_total": int(
                    year_signals["growth_strong_deviation"].sum()
                ),
                "normal_consistency_constant": ROBUST_NORMAL_CONSISTENCY,
                "elevated_D_threshold": elevated_d,
                "strong_D_threshold": strong_d,
            }
        )

    reference = pd.DataFrame(reference_rows)
    if signal_frames:
        signals = pd.concat(signal_frames, ignore_index=True)
        signals = signals.sort_values(
            ["year", "growth_D", "id-paper"], ascending=[True, True, True]
        ).reset_index(drop=True)
    else:
        signals = pd.DataFrame(
            columns=[
                "institution_id",
                "id-paper",
                "name",
                "year",
                "N_articles_attributed",
                "previous_output",
                "annual_growth_pct",
                "growth_D",
                "growth_deviation_side",
                "growth_deviation_strength",
                "growth_strong_deviation",
                "color_hex",
                "minimum_previous_output",
            ]
        )
    return evaluated_data, reference, signals


def get_axis_setting(panel_key: str) -> Tuple[float, float, float]:
    config = Y_AXIS_SETTINGS[panel_key]
    return float(config["ymin"]), float(config["ymax"]), float(config["ystep"])


def axis_range_audit(panel_key: str, values: pd.Series) -> dict:
    ymin, ymax, _ = get_axis_setting(panel_key)
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {
            "panel": panel_key,
            "configured_ymin": ymin,
            "configured_ymax": ymax,
            "observed_min": pd.NA,
            "observed_max": pd.NA,
            "n_points_below_ymin": 0,
            "n_points_above_ymax": 0,
        }
    return {
        "panel": panel_key,
        "configured_ymin": ymin,
        "configured_ymax": ymax,
        "observed_min": float(numeric.min()),
        "observed_max": float(numeric.max()),
        "n_points_below_ymin": int((numeric < ymin).sum()),
        "n_points_above_ymax": int((numeric > ymax).sum()),
    }


def validate_axis_ranges(axis_audit: pd.DataFrame, policy: str) -> None:
    outside = axis_audit[
        (axis_audit["n_points_below_ymin"] > 0)
        | (axis_audit["n_points_above_ymax"] > 0)
    ]
    if outside.empty:
        return
    message = (
        "Main-panel data fall outside configured y-axis limits. Adjust "
        "Y_AXIS_SETTINGS before final export:\n"
        + outside.to_string(index=False)
    )
    if policy == "error":
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning)


def ordered_institution_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    institutions = mapping[mapping["institution_id"] != NATIONAL_ID].copy()
    institutions["sort_order"] = institutions["id-paper"].map(parse_idpaper_order)
    return institutions.sort_values(["sort_order", "id-paper", "institution_id"])


def style_axis(
    axis: plt.Axes,
    panel_key: str,
    y_label: str,
    x_label: str,
    x_ticks: List[int],
    x_limits: Tuple[float, float],
    is_percent: bool,
) -> None:
    ymin, ymax, ystep = get_axis_setting(panel_key)
    axis.set_facecolor("white")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.7)
    axis.spines["bottom"].set_linewidth(0.7)
    axis.tick_params(axis="both", labelsize=7.2, length=3.0, width=0.65)
    axis.set_title(PANEL_TITLES[panel_key], fontsize=9.3, fontweight="bold", pad=6)
    axis.set_ylabel(y_label, fontsize=8.2)
    axis.set_xlabel(x_label, fontsize=8.2)
    axis.set_xlim(*x_limits)
    axis.set_xticks(x_ticks)
    axis.set_ylim(ymin, ymax)
    axis.yaxis.set_major_locator(MultipleLocator(ystep))
    axis.yaxis.set_major_formatter(
        FuncFormatter(fmt_percent if is_percent else fmt_thousands)
    )
    if is_percent:
        axis.axhline(0, color="#777777", linewidth=0.7, linestyle="--", zorder=1)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.14,
        1.055,
        label,
        transform=axis.transAxes,
        fontsize=9.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def plot_national_series(
    axis: plt.Axes,
    national: pd.DataFrame,
    value_column: str,
    mapping: pd.DataFrame,
) -> None:
    national_row = mapping[mapping["institution_id"] == NATIONAL_ID].iloc[0]
    axis.plot(
        national["year"],
        national[value_column],
        color=national_row["color_hex"],
        linewidth=float(national_row["line_width"]),
        solid_capstyle="round",
        zorder=3,
    )


def style_boxplot(boxplot: dict) -> None:
    for box in boxplot["boxes"]:
        box.set_facecolor("#D6D6D6")
        box.set_edgecolor("#777777")
        box.set_linewidth(0.65)
        box.set_alpha(0.75)
    for whisker in boxplot["whiskers"]:
        whisker.set_color("#777777")
        whisker.set_linewidth(0.65)
    for cap in boxplot["caps"]:
        cap.set_color("#777777")
        cap.set_linewidth(0.65)
    for median in boxplot["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.35)


def draw_annual_boxplots(
    axis: plt.Axes,
    data: pd.DataFrame,
    value_column: str,
    start_year: int,
    end_year: int,
    box_width: float,
) -> None:
    for year in range(start_year, end_year + 1):
        values = pd.to_numeric(
            data.loc[data["year"].eq(year), value_column], errors="coerce"
        ).dropna()
        if len(values) >= 2 and values.nunique() >= 2:
            boxplot = axis.boxplot(
                [values.to_numpy(dtype=float)],
                positions=[year],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                whis=(0, 100),
                manage_ticks=False,
                zorder=2,
            )
            style_boxplot(boxplot)
        elif len(values) == 1:
            axis.scatter(
                [year],
                [float(values.iloc[0])],
                s=12,
                c="#888888",
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )


def symmetric_offsets(count: int, maximum: float) -> np.ndarray:
    if count <= 1:
        return np.zeros(count, dtype=float)
    return np.linspace(-maximum, maximum, count)


def draw_growth_deviations(axis: plt.Axes, signals: pd.DataFrame) -> None:
    if signals.empty:
        return
    for year, year_signals in signals.groupby("year", sort=True):
        year_signals = year_signals.sort_values(
            ["annual_growth_pct", "id-paper", "institution_id"]
        ).copy()
        year_signals["x_plot"] = float(year) + symmetric_offsets(
            len(year_signals), 0.13
        )
        for row in year_signals.itertuples(index=False):
            strong = bool(row.growth_strong_deviation)
            marker = "o" if row.growth_deviation_side == "upper" else "v"
            axis.scatter(
                row.x_plot,
                row.annual_growth_pct,
                s=34.0 if strong else 28.0,
                c=[row.color_hex],
                marker=marker,
                edgecolor="#111111" if strong else "white",
                linewidth=1.05 if strong else 0.5,
                alpha=0.99,
                zorder=8 if strong else 7,
            )


def add_growth_inset(
    axis: plt.Axes,
    institutional_growth: pd.DataFrame,
    national: pd.DataFrame,
    signals: pd.DataFrame,
    mapping: pd.DataFrame,
    end_year: int,
    box_width: float,
) -> None:
    inset_start = max(
        int(INSET_SETTINGS["start_year"]),
        int(institutional_growth["year"].min()),
    )
    inset_ymin = float(INSET_SETTINGS["ymin"])
    inset_ymax = float(INSET_SETTINGS["ymax"])
    inset_ystep = float(INSET_SETTINGS["ystep"])

    inset = axis.inset_axes([0.57, 0.49, 0.41, 0.45])
    zoom_data = institutional_growth.loc[
        institutional_growth["year"].between(inset_start, end_year)
    ]
    draw_annual_boxplots(
        inset,
        zoom_data,
        "annual_growth_pct",
        inset_start,
        end_year,
        box_width * 0.78,
    )
    national_zoom = national.loc[national["year"].between(inset_start, end_year)]
    plot_national_series(inset, national_zoom, "annual_growth_pct", mapping)
    inset_signals = signals.loc[signals["year"].between(inset_start, end_year)]
    draw_growth_deviations(inset, inset_signals)
    inset.set_facecolor((1.0, 1.0, 1.0, 0.97))
    inset.grid(axis="y", color="#E1E1E1", linewidth=0.4, zorder=0)
    inset.axhline(0, color="#777777", linewidth=0.55, linestyle="--", zorder=1)
    inset.set_xlim(inset_start - 0.45, end_year + 0.45)
    inset.set_ylim(inset_ymin, inset_ymax)
    inset.set_xticks(list(range(inset_start, end_year + 1, 2)))
    inset.yaxis.set_major_locator(MultipleLocator(inset_ystep))
    inset.yaxis.set_major_formatter(FuncFormatter(fmt_percent))
    inset.tick_params(axis="both", labelsize=5.7, length=2.0, width=0.5, pad=1.5)
    for spine in inset.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(0.6)
    inset.set_title(f"{inset_start}-{end_year} detail", fontsize=6.4, pad=2)

    selection = Rectangle(
        (inset_start - 0.25, inset_ymin),
        end_year - inset_start + 0.5,
        inset_ymax - inset_ymin,
        fill=False,
        edgecolor="#666666",
        linewidth=0.65,
        linestyle=":",
        zorder=4,
    )
    axis.add_patch(selection)


def build_combined_figure(
    data: pd.DataFrame,
    growth_signals: pd.DataFrame,
    mapping: pd.DataFrame,
    output_base: Path,
    start_year: int,
    end_year: int,
    width: float,
    height: float,
    dpi: int,
    box_width: float,
) -> None:
    if output_utils.data_only():
        return
    plt.rcParams.update(
        {
            "font.family": "Source Sans 3",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    national = data[data["institution_id"] == NATIONAL_ID].copy()
    institutional = data[data["institution_id"] != NATIONAL_ID].copy()
    institutional_growth = institutional.loc[
        institutional["growth_status"].eq("computed_consecutive_years")
        & institutional["annual_growth_pct"].notna()
    ].copy()
    institution_mapping = ordered_institution_mapping(mapping)

    figure = plt.figure(figsize=(width, height))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.18])
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, :])

    output_ticks = list(range(start_year, end_year + 1, 2))
    if output_ticks[-1] != end_year:
        output_ticks.append(end_year)
    growth_start = start_year + 1
    growth_ticks = list(range(growth_start, end_year + 1, 2))
    if end_year - growth_ticks[-1] >= 2:
        growth_ticks.append(end_year)

    plot_national_series(axis_a, national, "N_articles_attributed", mapping)
    style_axis(
        axis_a,
        "national_output",
        "Publications per year",
        "",
        output_ticks,
        (start_year - 0.2, end_year + 0.2),
        False,
    )
    add_panel_label(axis_a, "(a)")

    draw_annual_boxplots(
        axis_b,
        institutional,
        "N_articles_attributed",
        start_year,
        end_year,
        box_width,
    )
    style_axis(
        axis_b,
        "institutional_output",
        "Publications per year",
        "",
        output_ticks,
        (start_year - 0.2, end_year + 0.2),
        False,
    )
    add_panel_label(axis_b, "(b)")

    draw_annual_boxplots(
        axis_c,
        institutional_growth,
        "annual_growth_pct",
        growth_start,
        end_year,
        box_width,
    )
    plot_national_series(axis_c, national, "annual_growth_pct", mapping)
    draw_growth_deviations(axis_c, growth_signals)
    style_axis(
        axis_c,
        "combined_growth",
        "Annual growth (%)",
        "Year",
        growth_ticks,
        (growth_start - 0.45, end_year + 0.45),
        True,
    )
    add_growth_inset(
        axis_c,
        institutional_growth,
        national,
        growth_signals,
        mapping,
        end_year,
        box_width,
    )
    add_panel_label(axis_c, "(c)")

    key_color = "#6E6E6E"
    national_row = mapping.loc[mapping["institution_id"].eq(NATIONAL_ID)].iloc[0]
    marker_key = [
        Line2D(
            [0],
            [0],
            color=national_row["color_hex"],
            linewidth=2.0,
            label="National aggregate",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=key_color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5.0,
            label="Upper D >= 2",
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            linestyle="None",
            markerfacecolor=key_color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5.0,
            label="Lower D <= -2",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=key_color,
            markeredgecolor="#111111",
            markeredgewidth=1.05,
            markersize=5.3,
            label="Black outline: |D| >= 3",
        ),
    ]
    axis_c.legend(
        handles=marker_key,
        loc="upper left",
        bbox_to_anchor=(0.012, 0.985),
        ncol=2,
        frameon=False,
        fontsize=6.3,
        handlelength=1.6,
        handletextpad=0.45,
        columnspacing=0.9,
        borderaxespad=0.0,
    )

    legend_handles_row_major = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=row.color_hex,
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=4.0,
            label=row.legend_label,
        )
        for row in institution_mapping.itertuples(index=False)
    ]
    legend_columns = 15
    legend_rows = int(np.ceil(len(legend_handles_row_major) / legend_columns))
    legend_handles = []
    for column in range(legend_columns):
        for row in range(legend_rows):
            index = row * legend_columns + column
            if index < len(legend_handles_row_major):
                legend_handles.append(legend_handles_row_major[index])
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=legend_columns,
        frameon=False,
        fontsize=6.2,
        handlelength=0.8,
        handletextpad=0.15,
        columnspacing=0.55,
        labelspacing=0.35,
        borderaxespad=0.0,
    )

    figure.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.955,
        bottom=0.12,
        wspace=0.25,
        hspace=0.30,
    )

    figure_style.finish(figure, output_base)
    figure.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    figure.savefig(
        output_base.with_suffix(".svg"),
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    figure.savefig(
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


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    mapping_path = Path(args.mapping_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.is_dir():
        raise NotADirectoryError(f'Input directory does not exist: {input_dir}')
    if not mapping_path.is_file():
        raise FileNotFoundError(f'Mapping file does not exist: {mapping_path}')
    mapping = load_mapping(mapping_path, args.mapping_encoding)
    files, loaded_files, ignored_files = discover_annual_metric_files(input_dir, args.start_year, args.end_year)
    metrics, off_year, year_audit, duplicate_audit = load_annual_metrics(files)
    data, missing_audit = prepare_plot_data(metrics, mapping, args.start_year, args.end_year, args.missing_institution_year)
    national = data[data['institution_id'] == NATIONAL_ID]
    institutional = data[data['institution_id'] != NATIONAL_ID]
    institutional_with_d, growth_reference, growth_signals = classify_growth_robust_d(institutional, args.start_year, args.end_year, args.min_growth_denominator, args.min_reference_institutions, args.elevated_d, args.strong_d)
    data_with_d = pd.concat([national.copy(), institutional_with_d], ignore_index=True, sort=False).sort_values(['institution_id', 'year']).reset_index(drop=True)
    combined_growth_values = pd.concat([national['annual_growth_pct'], institutional['annual_growth_pct']], ignore_index=True)
    axis_audit = pd.DataFrame([axis_range_audit('national_output', national['N_articles_attributed']), axis_range_audit('institutional_output', institutional['N_articles_attributed']), axis_range_audit('combined_growth', combined_growth_values)])
    data_columns = ['institution_id', 'id-paper', 'name', 'year', 'N_articles_attributed', 'annual_growth_pct', 'previous_year', 'previous_output', 'growth_status', 'originally_observed', 'missing_output_cell', 'missing_policy', 'color_hex', 'line_width', 'legend_label', 'source_names']
    data_columns.extend(['growth_reference_eligible', 'growth_D', 'growth_deviation_side', 'growth_deviation_strength', 'growth_elevated_deviation', 'growth_strong_deviation'])
    for column in data_columns:
        if column not in data_with_d.columns:
            data_with_d[column] = np.nan
    write_csv(data_with_d[data_columns], output_dir / 'fig1_data_used.csv')
    validate_axis_ranges(axis_audit, args.axis_policy)
    print(f'Calculated growth sources in {output_dir}')


if __name__ == "__main__":
    main()
