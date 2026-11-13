#!/usr/bin/env python3
"""Create the main-text institution-year matrix of exposure signals (Figure 7).

The matrix is a descriptive screening display for five primary indicators:
IMFA, CIS, HPA, HAO, and NLO. It does not construct a composite risk score.
For each indicator and year, an upper robust deviation is defined as

    D = (x - annual institutional median) / (IQR / 1.349)

An elevated screening signal is assigned when D is at least 2.0 by default,
and a strong deviation is marked when D is at least 3.0 by default. These are
descriptive upper-tail thresholds, not formal hypothesis tests or evidence of
misconduct.
HPA and HAO use the positive eligible institutional distribution as their
annual reference because structural zeros otherwise collapse robust scales.
All eligible observations, including zeros, are nevertheless evaluated. NLO
uses CA_known as its eligibility denominator; the other indicators use total
attributed output.
"""

from __future__ import annotations

import figure_style
import output_utils

import argparse
import hashlib
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


SCRIPT_VERSION = "3.1-recent-first-aligned-profile-matrix-2026-08-17"
NATIONAL_ID = "NATIONAL"
METRIC_SPECS = (
    {
        "metric_key": "IMFA",
        "rate_column": "IMFA_rate",
        "count_column": "N_articles_IMFA",
        "denominator_column": "N_articles_attributed",
        "positive_reference": False,
    },
    {
        "metric_key": "CIS",
        "rate_column": "CIS_rate",
        "count_column": "N_articles_CIS",
        "denominator_column": "N_articles_attributed",
        "positive_reference": False,
    },
    {
        "metric_key": "HPA",
        "rate_column": "HPA_rate",
        "count_column": "N_articles_with_hyperprolific_author",
        "denominator_column": "N_articles_attributed",
        "positive_reference": True,
    },
    {
        "metric_key": "HAO",
        "rate_column": "HAO_rate",
        "count_column": "N_articles_hyperauthored",
        "denominator_column": "N_articles_attributed",
        "positive_reference": True,
    },
    {
        "metric_key": "NLO",
        "rate_column": "NLO_rate",
        "count_column": "N_articles_non_led",
        "denominator_column": "CA_known",
        "positive_reference": False,
    },
)
METRIC_ORDER = tuple(spec["metric_key"] for spec in METRIC_SPECS)
METRIC_ORDER_LOOKUP = {metric: index for index, metric in enumerate(METRIC_ORDER)}




def build_phase_summary(
    details: pd.DataFrame,
    start_year: int,
    end_year: int,
    recent_start_year: int,
) -> pd.DataFrame:
    phase_definitions = (
        ("expansion_phase", start_year, min(2017, end_year)),
        ("slower_growth_phase", max(2018, start_year), end_year),
        ("recent_period", max(recent_start_year, start_year), end_year),
    )
    frames = []
    for phase_name, phase_start, phase_end in phase_definitions:
        if phase_start > phase_end:
            continue
        phase = details.loc[
            details["analysis_year"].between(phase_start, phase_end)
        ].copy()
        summary = (
            phase.groupby(
                [
                    "institution_id",
                    "name",
                    "id-paper",
                    "color_hex",
                    "code_order",
                ],
                as_index=False,
            )
            .agg(
                evaluable_indicator_years=("scorable", "sum"),
                elevated_signal_instances=("elevated_signal", "sum"),
                strong_deviation_instances=("strong_signal", "sum"),
            )
        )
        evaluable_years = (
            phase.groupby(["institution_id", "analysis_year"])["scorable"]
            .max()
            .groupby("institution_id")
            .sum()
            .rename("evaluable_years")
            .reset_index()
        )
        signal_years = (
            phase.groupby(["institution_id", "analysis_year"])["elevated_signal"]
            .max()
            .groupby("institution_id")
            .sum()
            .rename("signal_years")
            .reset_index()
        )
        signal_metrics = (
            phase.loc[phase["elevated_signal"]]
            .groupby("institution_id")["metric_key"]
            .apply(ordered_metric_list)
            .rename("elevated_metrics")
            .reset_index()
        )
        summary = summary.merge(
            evaluable_years,
            on="institution_id",
            how="left",
            validate="one_to_one",
        ).merge(
            signal_years,
            on="institution_id",
            how="left",
            validate="one_to_one",
        ).merge(
            signal_metrics,
            on="institution_id",
            how="left",
            validate="one_to_one",
        )
        summary["elevated_metrics"] = summary["elevated_metrics"].fillna("")
        summary["distinct_elevated_metrics"] = summary["elevated_metrics"].map(
            lambda value: 0 if not value else len(value.split(";"))
        )
        summary["signal_fraction_of_evaluable_indicator_years"] = summary[
            "elevated_signal_instances"
        ].div(summary["evaluable_indicator_years"].replace(0, np.nan))
        summary["phase"] = phase_name
        summary["phase_start_year"] = phase_start
        summary["phase_end_year"] = phase_end
        frames.append(summary)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["phase_start_year", "elevated_signal_instances", "signal_years", "code_order"],
        ascending=[True, False, False, True],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--mapping-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--mapping-encoding", default="latin1")
    parser.add_argument("--min-denominator", type=int, default=50)
    parser.add_argument("--min-reference-institutions", type=int, default=8)
    parser.add_argument("--deviation-threshold", type=float, default=2.0)
    parser.add_argument("--strong-deviation-threshold", type=float, default=3.0)
    parser.add_argument("--recent-start-year", type=int, default=2023)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--include-internal-identifiers",
        action="store_true",
        help=(
            "Retain internal institution_id/name fields in audit CSVs. By default, "
            "outputs contain only deidentified manuscript codes."
        ),
    )
    parser.add_argument(
        "--output-stem",
        default="Figure7_institution_year_exposure_signal_matrix_2011_2025",
    )
    args = parser.parse_args()
    if args.end_year < args.start_year:
        parser.error("--end-year must not precede --start-year")
    if not args.start_year <= args.recent_start_year <= args.end_year:
        parser.error("--recent-start-year must fall inside the displayed period")
    if min(args.min_denominator, args.min_reference_institutions, args.dpi) < 1:
        parser.error("Count thresholds and --dpi must be positive")
    if args.deviation_threshold <= 0:
        parser.error("--deviation-threshold must be positive")
    if args.strong_deviation_threshold <= args.deviation_threshold:
        parser.error(
            "--strong-deviation-threshold must exceed --deviation-threshold"
        )
    return args


def parse_code_order(value: object) -> int:
    match = re.search(r"(\d+)$", str(value).strip())
    return int(match.group(1)) if match else 10**9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mapping(path: Path, encoding: str) -> pd.DataFrame:
    mapping = pd.read_csv(path, encoding=encoding)
    required = {"institution_id", "name", "id-paper", "color_hex"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    mapping = mapping.copy()
    for column in ("institution_id", "name", "id-paper", "color_hex"):
        mapping[column] = mapping[column].astype(str).str.strip()
    mapping["color_hex"] = mapping["color_hex"].str.upper()
    if mapping["institution_id"].duplicated().any():
        raise ValueError("The mapping contains duplicated institution IDs")
    if not mapping["color_hex"].str.fullmatch(r"#[0-9A-F]{6}").all():
        raise ValueError("All mapping colors must use #RRGGBB")
    if mapping["institution_id"].eq(NATIONAL_ID).sum() != 1:
        raise ValueError("The mapping must contain exactly one NATIONAL row")
    mapping["code_order"] = mapping["id-paper"].map(parse_code_order)
    return mapping


def load_annual_metrics(
    input_dir: Path,
    mapping: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {"institution_id", "name", "year"}
    for spec in METRIC_SPECS:
        required_columns.update(
            (
                spec["rate_column"],
                spec["count_column"],
                spec["denominator_column"],
            )
        )
    frames = []
    files = []
    for analysis_year in range(start_year, end_year + 1):
        path = input_dir / f"{analysis_year}_institution_year_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing exact annual file: {path}")
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        frame = frame[list(required_columns)].copy()
        frame["institution_id"] = frame["institution_id"].astype(str).str.strip()
        frame["name"] = frame["name"].astype(str).str.strip()
        if frame["institution_id"].duplicated().any():
            raise ValueError(f"{path.name} contains duplicated institution IDs")
        if frame["institution_id"].eq(NATIONAL_ID).sum() != 1:
            raise ValueError(f"{path.name} must contain exactly one NATIONAL row")
        for column in required_columns - {"institution_id", "name"}:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        frame["analysis_year"] = analysis_year
        frame["source_file"] = path.name
        frames.append(frame)
        files.append(
            {
                "analysis_year": analysis_year,
                "source_file": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rows": len(frame),
                "off_year_rows": int(frame["year"].ne(analysis_year).sum()),
            }
        )
    data = pd.concat(frames, ignore_index=True)
    missing_ids = sorted(set(data["institution_id"]) - set(mapping["institution_id"]))
    if missing_ids:
        raise ValueError(f"Institutions absent from mapping: {missing_ids}")
    data = data.merge(
        mapping[
            ["institution_id", "id-paper", "color_hex", "code_order"]
        ],
        on="institution_id",
        how="left",
        validate="many_to_one",
    )
    return data, pd.DataFrame(files)


def validate_metric_values(data: pd.DataFrame) -> None:
    for spec in METRIC_SPECS:
        rate = data[spec["rate_column"]]
        count = data[spec["count_column"]]
        denominator = data[spec["denominator_column"]]
        invalid = (
            count.lt(0)
            | denominator.lt(0)
            | count.gt(denominator)
            | (~rate.between(0, 1) & rate.notna())
        )
        if invalid.any():
            sample = data.loc[
                invalid, ["analysis_year", "institution_id", spec["rate_column"]]
            ].head()
            raise ValueError(
                f"Invalid values for {spec['metric_key']}:\n{sample.to_string(index=False)}"
            )
        expected = count.div(denominator.replace(0, np.nan))
        mismatch = rate.sub(expected).abs().gt(1e-9) & denominator.gt(0)
        if mismatch.any():
            raise ValueError(f"Rate identity failed for {spec['metric_key']}")


def build_indicator_details(
    data: pd.DataFrame,
    min_denominator: int,
    min_reference_institutions: int,
    deviation_threshold: float,
    strong_deviation_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    institutional = data.loc[data["institution_id"].ne(NATIONAL_ID)].copy()
    detail_frames = []
    reference_rows = []
    for analysis_year in sorted(institutional["analysis_year"].unique()):
        annual = institutional.loc[
            institutional["analysis_year"].eq(analysis_year)
        ].copy()
        for spec in METRIC_SPECS:
            metric = spec["metric_key"]
            rate_column = spec["rate_column"]
            denominator_column = spec["denominator_column"]
            frame = annual[
                [
                    "analysis_year",
                    "institution_id",
                    "name",
                    "id-paper",
                    "color_hex",
                    "code_order",
                    "source_file",
                    rate_column,
                    spec["count_column"],
                    denominator_column,
                ]
            ].copy()
            frame = frame.rename(
                columns={
                    rate_column: "rate",
                    spec["count_column"]: "indicator_article_count",
                    denominator_column: "eligibility_denominator",
                }
            )
            frame["metric_key"] = metric
            frame["metric_order"] = METRIC_ORDER_LOOKUP[metric] + 1
            frame["rate_pp"] = 100.0 * frame["rate"]
            frame["eligible"] = (
                frame["eligibility_denominator"].ge(min_denominator)
                & frame["rate_pp"].notna()
            )
            eligible = frame.loc[frame["eligible"]].copy()
            if spec["positive_reference"]:
                reference = eligible.loc[eligible["rate_pp"].gt(0), "rate_pp"]
                reference_basis = "positive eligible institutions"
            else:
                reference = eligible["rate_pp"]
                reference_basis = "all eligible institutions"
            n_reference = int(len(reference))
            distinct_reference = int(reference.nunique())
            q1 = float(reference.quantile(0.25)) if n_reference else np.nan
            median = float(reference.median()) if n_reference else np.nan
            q3 = float(reference.quantile(0.75)) if n_reference else np.nan
            iqr = q3 - q1 if n_reference else np.nan
            robust_scale = iqr / 1.349 if np.isfinite(iqr) and iqr > 0 else np.nan
            reference_valid = bool(
                n_reference >= min_reference_institutions
                and distinct_reference >= 2
                and np.isfinite(robust_scale)
                and robust_scale > 0
            )
            frame["reference_basis"] = reference_basis
            frame["n_reference_institutions"] = n_reference
            frame["n_distinct_reference_values"] = distinct_reference
            frame["annual_q1_pp"] = q1
            frame["annual_median_pp"] = median
            frame["annual_q3_pp"] = q3
            frame["annual_iqr_pp"] = iqr
            frame["robust_scale_pp"] = robust_scale
            frame["reference_valid"] = reference_valid
            frame["scorable"] = frame["eligible"] & reference_valid
            frame["robust_deviation"] = np.where(
                frame["scorable"],
                (frame["rate_pp"] - median) / robust_scale,
                np.nan,
            )
            frame["elevated_signal"] = (
                frame["scorable"]
                & frame["robust_deviation"].ge(deviation_threshold)
            )
            frame["strong_signal"] = (
                frame["scorable"]
                & frame["robust_deviation"].ge(strong_deviation_threshold)
            )
            frame["deviation_threshold"] = deviation_threshold
            frame["strong_deviation_threshold"] = strong_deviation_threshold
            frame["screening_reason"] = "not elevated"
            frame.loc[~frame["eligible"], "screening_reason"] = (
                "denominator below threshold or rate unavailable"
            )
            if not reference_valid:
                frame.loc[frame["eligible"], "screening_reason"] = (
                    "annual reference distribution not scorable"
                )
            frame.loc[frame["elevated_signal"], "screening_reason"] = (
                "upper robust deviation"
            )
            frame.loc[frame["strong_signal"], "screening_reason"] = (
                "strong upper robust deviation"
            )
            detail_frames.append(frame)
            reference_rows.append(
                {
                    "analysis_year": analysis_year,
                    "metric_key": metric,
                    "reference_basis": reference_basis,
                    "eligible_institutions": int(frame["eligible"].sum()),
                    "reference_institutions": n_reference,
                    "distinct_reference_values": distinct_reference,
                    "q1_pp": q1,
                    "median_pp": median,
                    "q3_pp": q3,
                    "iqr_pp": iqr,
                    "robust_scale_pp": robust_scale,
                    "reference_valid": reference_valid,
                    "elevated_institutions": int(frame["elevated_signal"].sum()),
                    "strong_deviation_institutions": int(frame["strong_signal"].sum()),
                }
            )
    details = pd.concat(detail_frames, ignore_index=True)
    details = details.sort_values(
        ["analysis_year", "metric_order", "code_order", "institution_id"]
    )
    reference = pd.DataFrame(reference_rows).sort_values(
        ["analysis_year", "metric_key"]
    )
    return details, reference


def ordered_metric_list(values: pd.Series) -> str:
    observed = set(values.dropna().astype(str))
    return ";".join(metric for metric in METRIC_ORDER if metric in observed)


def build_cell_summary(
    details: pd.DataFrame,
    mapping: pd.DataFrame,
    start_year: int,
    end_year: int,
    recent_start_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    institutions = mapping.loc[mapping["institution_id"].ne(NATIONAL_ID)].copy()
    grid = pd.MultiIndex.from_product(
        [institutions["institution_id"], range(start_year, end_year + 1)],
        names=["institution_id", "analysis_year"],
    ).to_frame(index=False)
    grouped = (
        details.groupby(["institution_id", "analysis_year"], as_index=False)
        .agg(
            evaluable_indicator_count=("scorable", "sum"),
            elevated_signal_count=("elevated_signal", "sum"),
            strong_signal_count=("strong_signal", "sum"),
            max_robust_deviation=("robust_deviation", "max"),
        )
    )
    drivers = (
        details.loc[details["elevated_signal"]]
        .groupby(["institution_id", "analysis_year"])["metric_key"]
        .apply(ordered_metric_list)
        .rename("elevated_metrics")
        .reset_index()
    )
    strong_drivers = (
        details.loc[details["strong_signal"]]
        .groupby(["institution_id", "analysis_year"])["metric_key"]
        .apply(ordered_metric_list)
        .rename("strong_metrics")
        .reset_index()
    )
    cells = grid.merge(
        institutions[
            ["institution_id", "name", "id-paper", "color_hex", "code_order"]
        ],
        on="institution_id",
        how="left",
        validate="many_to_one",
    )
    cells = cells.merge(
        grouped,
        on=["institution_id", "analysis_year"],
        how="left",
        validate="one_to_one",
    ).merge(
        drivers,
        on=["institution_id", "analysis_year"],
        how="left",
        validate="one_to_one",
    ).merge(
        strong_drivers,
        on=["institution_id", "analysis_year"],
        how="left",
        validate="one_to_one",
    )
    for column in (
        "evaluable_indicator_count",
        "elevated_signal_count",
        "strong_signal_count",
    ):
        cells[column] = cells[column].fillna(0).astype(int)
    cells["elevated_metrics"] = cells["elevated_metrics"].fillna("")
    cells["strong_metrics"] = cells["strong_metrics"].fillna("")
    cells["has_strong_signal"] = cells["strong_signal_count"].gt(0)
    cells["cell_label"] = cells.apply(
        lambda row: (
            f"{row['elevated_signal_count']}/{row['evaluable_indicator_count']}"
            if row["evaluable_indicator_count"] > 0
            else "—"
        ),
        axis=1,
    )
    cells["all_five_evaluable"] = cells["evaluable_indicator_count"].eq(
        len(METRIC_ORDER)
    )

    complete = (
        cells.groupby(
            ["institution_id", "name", "id-paper", "color_hex", "code_order"],
            as_index=False,
        )
        .agg(
            complete_period_signal_instances=("elevated_signal_count", "sum"),
            complete_period_strong_signal_instances=("strong_signal_count", "sum"),
            complete_period_signal_years=(
                "elevated_signal_count",
                lambda values: int((values > 0).sum()),
            ),
            complete_period_evaluable_years=(
                "evaluable_indicator_count",
                lambda values: int((values > 0).sum()),
            ),
        )
    )
    recent = cells.loc[cells["analysis_year"].ge(recent_start_year)]
    recent_summary = (
        recent.groupby("institution_id", as_index=False)
        .agg(
            recent_signal_instances=("elevated_signal_count", "sum"),
            recent_strong_signal_instances=("strong_signal_count", "sum"),
            recent_peak_signals=("elevated_signal_count", "max"),
            recent_signal_years=(
                "elevated_signal_count",
                lambda values: int((values > 0).sum()),
            ),
        )
    )
    row_order = complete.merge(
        recent_summary, on="institution_id", how="left", validate="one_to_one"
    ).sort_values(
        [
            "recent_signal_instances",
            "complete_period_signal_instances",
            "complete_period_signal_years",
            "complete_period_strong_signal_instances",
            "code_order",
        ],
        ascending=[False, False, False, False, True],
    )
    row_order["display_order"] = range(1, len(row_order) + 1)
    cells = cells.merge(
        row_order[
            [
                "institution_id",
                "recent_signal_instances",
                "recent_strong_signal_instances",
                "recent_peak_signals",
                "recent_signal_years",
                "complete_period_signal_instances",
                "complete_period_strong_signal_instances",
                "complete_period_signal_years",
                "display_order",
            ]
        ],
        on="institution_id",
        how="left",
        validate="many_to_one",
    ).sort_values(["display_order", "analysis_year"])
    return cells, row_order


def build_institution_metric_profiles(
    details: pd.DataFrame,
    row_order: pd.DataFrame,
) -> pd.DataFrame:
    profiles = (
        details.groupby(
            [
                "institution_id",
                "name",
                "id-paper",
                "color_hex",
                "code_order",
                "metric_key",
                "metric_order",
            ],
            as_index=False,
        )
        .agg(
            evaluable_years=("scorable", "sum"),
            elevated_years=("elevated_signal", "sum"),
            strong_deviation_years=("strong_signal", "sum"),
            maximum_robust_deviation=("robust_deviation", "max"),
        )
    )
    profiles["evaluable_years"] = profiles["evaluable_years"].astype(int)
    profiles["elevated_years"] = profiles["elevated_years"].astype(int)
    profiles["strong_deviation_years"] = profiles[
        "strong_deviation_years"
    ].astype(int)
    profiles["elevated_fraction_of_evaluable_years"] = profiles[
        "elevated_years"
    ].div(profiles["evaluable_years"].replace(0, np.nan))
    profiles["profile_label"] = profiles.apply(
        lambda row: (
            f"{row['elevated_years']}/{row['evaluable_years']}"
            if row["evaluable_years"] > 0
            else "—"
        ),
        axis=1,
    )
    profiles = profiles.merge(
        row_order[["institution_id", "display_order"]],
        on="institution_id",
        how="left",
        validate="many_to_one",
    ).sort_values(["display_order", "metric_order"])
    return profiles


def save_figure(figure: plt.Figure, output_base: Path, dpi: int) -> None:
    figure_style.finish(figure, output_base)
    for suffix in (".pdf", ".svg"):
        figure.savefig(
            output_base.with_suffix(suffix),
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
        )
    figure.savefig(
        output_base.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
    )


def plot_matrix(
    cells: pd.DataFrame,
    row_order: pd.DataFrame,
    profiles: pd.DataFrame,
    annual_reference: pd.DataFrame,
    output_base: Path,
    start_year: int,
    end_year: int,
    recent_start_year: int,
    deviation_threshold: float,
    strong_deviation_threshold: float,
    dpi: int,
) -> None:
    if output_utils.data_only():
        return
    years = list(range(start_year, end_year + 1))
    ordered_ids = row_order.sort_values("display_order")["institution_id"].tolist()
    metadata = row_order.set_index("institution_id").loc[ordered_ids]
    count_matrix = (
        cells.pivot(
            index="institution_id",
            columns="analysis_year",
            values="elevated_signal_count",
        )
        .reindex(index=ordered_ids, columns=years)
        .to_numpy(dtype=float)
    )
    eval_matrix = (
        cells.pivot(
            index="institution_id",
            columns="analysis_year",
            values="evaluable_indicator_count",
        )
        .reindex(index=ordered_ids, columns=years)
        .to_numpy(dtype=int)
    )
    label_matrix = (
        cells.pivot(index="institution_id", columns="analysis_year", values="cell_label")
        .reindex(index=ordered_ids, columns=years)
        .to_numpy(dtype=object)
    )
    strong_matrix = (
        cells.pivot(
            index="institution_id",
            columns="analysis_year",
            values="strong_signal_count",
        )
        .reindex(index=ordered_ids, columns=years)
        .to_numpy(dtype=int)
    )
    masked_counts = np.ma.masked_where(eval_matrix == 0, count_matrix)
    colors = (
        "#F5F5F5",
        "#FEE8C8",
        "#FDBB84",
        "#FC8D59",
        "#D7301F",
        "#7F0000",
    )
    cmap = ListedColormap(colors)
    cmap.set_bad("#D9D9D9")
    norm = BoundaryNorm(np.arange(-0.5, 6.0, 1.0), cmap.N)

    profile_fraction_matrix = (
        profiles.pivot(
            index="institution_id",
            columns="metric_key",
            values="elevated_fraction_of_evaluable_years",
        )
        .reindex(index=ordered_ids, columns=METRIC_ORDER)
        .to_numpy(dtype=float)
    )
    profile_evaluable_matrix = (
        profiles.pivot(
            index="institution_id",
            columns="metric_key",
            values="evaluable_years",
        )
        .reindex(index=ordered_ids, columns=METRIC_ORDER)
        .to_numpy(dtype=int)
    )
    profile_label_matrix = (
        profiles.pivot(
            index="institution_id",
            columns="metric_key",
            values="profile_label",
        )
        .reindex(index=ordered_ids, columns=METRIC_ORDER)
        .to_numpy(dtype=object)
    )
    profile_strong_matrix = (
        profiles.pivot(
            index="institution_id",
            columns="metric_key",
            values="strong_deviation_years",
        )
        .reindex(index=ordered_ids, columns=METRIC_ORDER)
        .to_numpy(dtype=int)
    )
    masked_profile_fraction = np.ma.masked_where(
        profile_evaluable_matrix == 0,
        profile_fraction_matrix,
    )
    profile_cmap = LinearSegmentedColormap.from_list(
        "profile_fraction",
        ("#F5F5F5", "#D9EDF0", "#8CC4C7", "#2A7F83", "#084C52"),
    )
    profile_cmap.set_bad("#D9D9D9")

    valid_metric_counts = (
        annual_reference.loc[annual_reference["reference_valid"]]
        .groupby("analysis_year")["metric_key"]
        .nunique()
        .reindex(years, fill_value=0)
        .astype(int)
    )

    height = max(8.4, 0.235 * len(ordered_ids) + 2.15)
    figure = plt.figure(figsize=(10.8, height))
    grid = figure.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[15.0, 5.2],
        wspace=0.065,
    )
    axis = figure.add_subplot(grid[0, 0])
    profile_axis = figure.add_subplot(grid[0, 1], sharey=axis)
    image = figure_style.heatmap(axis,masked_counts, cmap=cmap, norm=norm, aspect="auto")
    axis.set_xticks(range(len(years)))
    axis.set_xticklabels(
        [f"{year}\nm={valid_metric_counts.loc[year]}" for year in years]
    )
    axis.set_yticks(range(len(ordered_ids)))
    axis.set_yticklabels(metadata["id-paper"].tolist())
    axis.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
    axis.tick_params(axis="x", labelsize=7.5)
    axis.tick_params(axis="y", length=0, pad=5, labelsize=7.5)
    axis.set_xticks(np.arange(-0.5, len(years), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(ordered_ids), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.15)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    if start_year <= 2018 < end_year:
        axis.axvline(2018 - start_year + 0.5, color="#333333", linewidth=1.6)
    recent_column = recent_start_year - start_year
    if recent_column > 0:
        axis.axvline(
            recent_column - 0.5,
            color="#6A6A6A",
            linewidth=1.25,
            linestyle="--",
        )
    for index, label in enumerate(axis.get_xticklabels()):
        if years[index] >= recent_start_year:
            label.set_fontweight("bold")
            label.set_color("#8B1A1A")

    for row in range(len(ordered_ids)):
        for column in range(len(years)):
            if eval_matrix[row, column] == 0:
                continue
            count = int(count_matrix[row, column])
            text_color = "white" if count >= 3 else "#222222"
            axis.text(
                column,
                row,
                str(label_matrix[row, column]),
                ha="center",
                va="center",
                color=text_color,
                fontsize=6.8,
                fontweight="bold" if count >= 2 else "normal",
            )
            if strong_matrix[row, column] > 0:
                axis.scatter(
                    column + 0.29,
                    row - 0.29,
                    s=11,
                    marker="o",
                    facecolor="#111111",
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=5,
                )

    for row, institution_id in enumerate(ordered_ids):
        color = metadata.loc[institution_id, "color_hex"]
        axis.add_patch(
            Rectangle(
                (-0.65, row - 0.26),
                0.08,
                0.52,
                facecolor=color,
                edgecolor="none",
                clip_on=False,
            )
        )

    axis.set_title(
        "A  Institution-year signal breadth",
        loc="left",
        fontsize=9.0,
        fontweight="bold",
        pad=28,
    )
    if start_year <= 2018:
        expansion_start = max(start_year, 2011)
        expansion_end = min(end_year, 2018)
        if expansion_start <= expansion_end:
            expansion_midpoint = (
                expansion_start + expansion_end
            ) / 2.0 - start_year
            axis.text(
                expansion_midpoint,
                len(ordered_ids) + 0.42,
                "Expansion phase",
                ha="center",
                va="top",
                fontsize=6.8,
                color="#4A4A4A",
                clip_on=False,
            )
    if end_year >= 2019:
        later_start = max(start_year, 2019)
        later_midpoint = (later_start + end_year) / 2.0 - start_year
        axis.text(
            later_midpoint,
            len(ordered_ids) + 0.42,
            "Slower-growth phase",
            ha="center",
            va="top",
            fontsize=6.8,
            color="#4A4A4A",
            clip_on=False,
        )

    profile_image = figure_style.heatmap(profile_axis,
        masked_profile_fraction,
        cmap=profile_cmap,
        norm=Normalize(vmin=0.0, vmax=1.0),
        aspect="auto",
    )
    profile_axis.set_xticks(range(len(METRIC_ORDER)))
    profile_axis.set_xticklabels(METRIC_ORDER)
    profile_axis.tick_params(
        axis="x",
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        labelsize=7.5,
    )
    profile_axis.tick_params(axis="y", left=False, labelleft=False)
    profile_axis.set_xticks(np.arange(-0.5, len(METRIC_ORDER), 1), minor=True)
    profile_axis.set_yticks(np.arange(-0.5, len(ordered_ids), 1), minor=True)
    profile_axis.grid(which="minor", color="white", linewidth=1.15)
    profile_axis.tick_params(which="minor", bottom=False, left=False)
    for spine in profile_axis.spines.values():
        spine.set_visible(False)
    profile_axis.set_title(
        "B  Full-period profile by indicator",
        loc="left",
        fontsize=9.0,
        fontweight="bold",
        pad=28,
    )
    for row in range(len(ordered_ids)):
        for column in range(len(METRIC_ORDER)):
            if profile_evaluable_matrix[row, column] == 0:
                continue
            fraction = float(profile_fraction_matrix[row, column])
            profile_axis.text(
                column,
                row,
                str(profile_label_matrix[row, column]),
                ha="center",
                va="center",
                color="white" if fraction >= 0.58 else "#222222",
                fontsize=6.7,
                fontweight="bold" if fraction >= 0.5 else "normal",
            )
            if profile_strong_matrix[row, column] > 0:
                profile_axis.scatter(
                    column + 0.29,
                    row - 0.29,
                    s=11,
                    marker="o",
                    facecolor="#111111",
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=5,
                )

    profile_axis.set_ylim(axis.get_ylim())

    figure.suptitle(
        "Concurrent elevated exposure signals by institution and year",
        y=0.992,
        fontsize=11.0,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.965,
        (
            "Panel A: elevated/evaluable indicators (k/m); panel B: elevated/evaluable years (s/y). "
            f"Elevated D ≥ {deviation_threshold:g}; dot marks D ≥ "
            f"{strong_deviation_threshold:g}."
        ),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#444444",
    )

    # Apply the final subplot geometry before creating the colorbar. Passing
    # profile_axis directly to figure.colorbar() would shrink only panel B and
    # make its rows physically shorter than the corresponding rows in panel A.
    figure.subplots_adjust(left=0.075, right=0.985, top=0.885, bottom=0.16)
    axis_position = axis.get_position()
    profile_position = profile_axis.get_position()
    if (
        abs(axis_position.y0 - profile_position.y0) > 1e-9
        or abs(axis_position.height - profile_position.height) > 1e-9
    ):
        raise RuntimeError("Panels A and B do not have identical row geometry")

    legend_handles = [
        Patch(facecolor=colors[count], edgecolor="#FFFFFF", label=str(count))
        for count in range(6)
    ]
    legend_handles.append(
        Patch(facecolor="#D9D9D9", edgecolor="#FFFFFF", label="Not evaluable")
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#111111",
            markeredgecolor="white",
            markersize=4.5,
            label=f"Strong (D ≥ {strong_deviation_threshold:g})",
        )
    )
    axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.072),
        ncol=8,
        frameon=False,
        title="Panel A: number of elevated indicators",
        fontsize=6.5,
        title_fontsize=6.8,
        handlelength=1.0,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    profile_colorbar_axis = figure.add_axes(
        [profile_position.x0, 0.084, profile_position.width, 0.014]
    )
    profile_colorbar = figure.colorbar(
        profile_image,
        cax=profile_colorbar_axis,
        orientation="horizontal",
        ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    profile_colorbar.ax.tick_params(labelsize=6.2, length=2)
    profile_colorbar.set_label(
        "Panel B: elevated years / evaluable years",
        fontsize=6.6,
        labelpad=2,
    )
    figure.text(
        0.5,
        0.008,
        (
            "m = valid annual indicators; annual references exclude NATIONAL. "
            f"Rows: {recent_start_year}–{end_year} signals ↓, full-period signals ↓, "
            "signal years ↓, strong signals ↓, then code ↑. This display order is not an institutional ranking."
        ),
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#555555",
    )
    save_figure(figure, output_base, dpi)
    plt.close(figure)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    if not output_utils.keep(path):
        return
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def deidentify_for_export(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output = output.drop(
        columns=["institution_id", "name", "color_hex"], errors="ignore"
    )
    if "id-paper" in output.columns:
        output = output.rename(columns={"id-paper": "institution_code"})
    return output


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    mapping_path = Path(args.mapping_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if not mapping_path.is_file():
        raise FileNotFoundError(mapping_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_mapping(mapping_path, args.mapping_encoding)
    data, files = load_annual_metrics(input_dir, mapping, args.start_year, args.end_year)
    validate_metric_values(data)
    details, annual_reference = build_indicator_details(data, args.min_denominator, args.min_reference_institutions, args.deviation_threshold, args.strong_deviation_threshold)
    export_details = details if args.include_internal_identifiers else deidentify_for_export(details)
    write_csv(export_details, output_dir / f'{args.output_stem}_indicator_details.csv')
    phase_summary = build_phase_summary(details, args.start_year, args.end_year, args.recent_start_year)
    output_utils.report_methodology(signal_phase_summary=deidentify_for_export(phase_summary))
    print(f'Calculated signal-matrix sources in {output_dir}')


if __name__ == "__main__":
    main()
