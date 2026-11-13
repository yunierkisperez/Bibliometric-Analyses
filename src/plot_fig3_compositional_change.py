#!/usr/bin/env python3
"""
Figure 3: contemporaneous composition of publication-output change.

The script tests whether national and institutional changes in attributed
publication output coincided with disproportionate changes in the number or
share of articles displaying the study indicators. For every indicator and
interval it applies the exact identity

    change in total output
    = change in indicator-positive output
    + change in remaining output.

It also decomposes the indicator-positive change into the amount expected if
the initial output composition had remained constant and the exact
compositional-reallocation term. Analyses are calculated for the national
aggregate, every institution, both study phases, and all consecutive annual
transitions.

Institutional compositional changes are screened with a two-sided robust
deviation, D = (change - median change) / (IQR change / 1.349), within each
indicator and interval. Values at |D| >= 2 are signals and values at |D| >= 3
are strong signals. This screening is separate from the exact accounting
classification and is not applied to national aggregates.

Indicators overlap and their contributions cannot be summed. The script
reconstructs the union excluding NLO and a three-state NLO/union partition so
that changes in corresponding-author coverage are not mistaken for changes in
indicator prevalence. It also audits every author crossing the annual
25-article HPA threshold.

The analysis year is always the year encoded in each annual filename. Missing
institution-year rows are interpreted as zero output only for the count-based
accounting. All results are descriptive accounting of the observed corpus and
do not estimate causal effects.
"""

from __future__ import annotations

import figure_style
import output_utils

import argparse
import importlib.util
import inspect
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib_fig3_dependence_cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

SCRIPT_VERSION = "4.1.0-2026-08-20"
DEFAULT_PREFIX = "fig3_compositional_change"
CORE_FILENAME = "fig3_data_core.py"

ROBUST_D_NORMAL_CONSISTENCY = 1.349
ROBUST_D_ELEVATED = 2.0
ROBUST_D_STRONG = 3.0
ROBUST_D_MIN_DENOMINATOR = 50
ROBUST_D_MIN_REFERENCE_N = 8
ROBUST_D_ACTIVE_REFERENCE_METRICS = {"HPA", "HAO", "HPA_MFA_CIS"}

PHASE_COLORS = {
    "Expansion (2012-2018)": "#C46A1A",
    "Stabilization (2019-2025)": "#2E7D5A",
}
PHASE_MARKERS = {
    "Expansion (2012-2018)": "o",
    "Stabilization (2019-2025)": "s",
}

ACCOUNTING_METRICS = (
    {
        "key": "ANY_RISK",
        "label": "At least one indicator",
        "numerator": "N_articles_any_risk",
        "denominator": "N_articles_attributed",
        "rate": "AnyRisk_rate",
    },
    {
        "key": "ANY_EXCL_NLO",
        "label": "At least one, excluding NLO",
        "numerator": "N_articles_any_excluding_NLO",
        "denominator": "N_articles_attributed",
        "rate": "AnyExcludingNLO_rate",
    },
    {
        "key": "IMFA",
        "label": "IMFA",
        "numerator": "N_articles_IMFA",
        "denominator": "N_articles_attributed",
        "rate": "IMFA_rate",
    },
    {
        "key": "CIS",
        "label": "CIS",
        "numerator": "N_articles_CIS",
        "denominator": "N_articles_attributed",
        "rate": "CIS_rate",
    },
    {
        "key": "HPA",
        "label": "HPA",
        "numerator": "N_articles_with_hyperprolific_author",
        "denominator": "N_articles_attributed",
        "rate": "HPA_rate",
    },
    {
        "key": "HAO",
        "label": "HAO",
        "numerator": "N_articles_hyperauthored",
        "denominator": "N_articles_attributed",
        "rate": "HAO_rate",
    },
    {
        "key": "NLO",
        "label": "NLO",
        "numerator": "N_articles_non_led",
        "denominator": "CA_known",
        "rate": "NLO_rate",
    },
    {
        "key": "HPA_MFA_CIS",
        "label": "HPA-MFA/CIS",
        "numerator": "N_articles_by_HPA_MFA_or_CIS",
        "denominator": "N_articles_attributed",
        "rate": "HPA_MFA_or_CIS_article_rate",
    },
)
ACCOUNTING_ORDER = [metric["key"] for metric in ACCOUNTING_METRICS]
PRIMARY_ACCOUNTING_ORDER = [
    "ANY_EXCL_NLO",
    "IMFA",
    "CIS",
    "HPA",
    "HAO",
    "NLO",
    "HPA_MFA_CIS",
]
ACCOUNTING_LABELS = {
    metric["key"]: metric["label"] for metric in ACCOUNTING_METRICS
}


def load_core_module():
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / CORE_FILENAME,
        script_dir / "previous" / CORE_FILENAME,
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"Required helper module is missing: {CORE_FILENAME}"
        )
    spec = importlib.util.spec_from_file_location("fig3_analysis_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose national and institutional output change into "
            "indicator-positive and remaining components."
        )
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--mapping-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--expansion-end-year", type=int, default=2018)
    parser.add_argument("--mapping-encoding", default="latin1")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    if not output_utils.keep(path):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters = inspect.signature(pd.DataFrame.to_csv).parameters
    if "lineterminator" in parameters:
        newline_keyword = "lineterminator"
    elif "line_terminator" in parameters:
        newline_keyword = "line_terminator"
    else:
        raise RuntimeError(
            "Unsupported pandas DataFrame.to_csv signature: no line-terminator "
            "keyword was found."
        )
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        **{newline_keyword: "\n"},
    )


def save_figure_formats(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> None:
    figure_style.finish(figure, output_dir / stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f"{stem}_",
            suffix=f".{extension}",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            figure.savefig(
                temporary_path,
                dpi=dpi if extension == "png" else None,
                bbox_inches="tight",
            )
            payload = temporary_path.read_bytes()
            if not payload:
                raise RuntimeError(
                    f"Temporary figure export is empty: {temporary_path}"
                )
            if extension == "png" and not payload.startswith(b"\x89PNG"):
                raise RuntimeError("Temporary PNG has an invalid signature")
            if extension == "pdf" and not payload.startswith(b"%PDF"):
                raise RuntimeError("Temporary PDF has an invalid signature")
            if extension == "svg":
                ET.parse(temporary_path)
            destination = output_dir / f"{stem}.{extension}"
            staging = output_dir / f".{stem}.{extension}.part"
            written = False
            for _ in range(3):
                staging.write_bytes(payload)
                if staging.stat().st_size != len(payload):
                    continue
                os.replace(staging, destination)
                if destination.stat().st_size == len(payload):
                    written = True
                    break
            if not written:
                raise RuntimeError(
                    f"Could not write a complete figure file: {destination}"
                )
        finally:
            temporary_path.unlink(missing_ok=True)


def required_revised_inputs(
    core,
    input_dir: Path,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    _, audit = core.required_input_paths(input_dir, start_year, end_year)
    extra_rows: list[dict[str, object]] = []
    for year in range(start_year, end_year + 1):
        for file_type in ("national_baseline", "article_institution_flags"):
            path = input_dir / f"{year}_{file_type}.csv"
            extra_rows.append(
                {
                    "analysis_year": year,
                    "file_type": file_type,
                    "path": str(path),
                    "exists": path.is_file(),
                    "size_bytes": (
                        path.stat().st_size if path.is_file() else np.nan
                    ),
                }
            )
    extra = pd.DataFrame(extra_rows)
    missing = extra.loc[~extra["exists"], "path"].tolist()
    if missing:
        raise FileNotFoundError(
            "Missing required revised-analysis files:\n" + "\n".join(missing)
        )
    return pd.concat([audit, extra], ignore_index=True, sort=False)


def national_required_columns(core) -> set[str]:
    return core.metric_required_columns() | {
        "N_articles_any_risk",
        "AnyRisk_rate",
        "CA_unknown_or_unmatched",
        "N_articles_CA_led",
    }


def load_national_baselines(
    core,
    input_dir: Path,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    off_year_rows: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    required = national_required_columns(core)
    numeric = sorted(
        {
            "year",
            "N_articles_attributed",
            "N_articles_any_risk",
            "AnyRisk_rate",
            "CA_unknown_or_unmatched",
            "N_articles_CA_led",
            "N_hyperprolific_authors",
            "N_hyperprolific_authors_MFA_or_CIS",
            *[
                column
                for metric in core.METRICS
                if metric["key"] != "HPA_MFA_CIS"
                for column in (
                    metric["numerator"],
                    metric["denominator"],
                    metric["rate"],
                )
            ],
        }
    )
    for analysis_year in range(start_year, end_year + 1):
        path = input_dir / f"{analysis_year}_national_baseline.csv"
        frame = core.read_csv(path)
        core.require_columns(frame, required, path.name)
        if len(frame) != 1:
            raise ValueError(f"{path.name}: expected exactly one national row")
        frame = core.coerce_numeric(frame, numeric, path.name)
        frame["institution_id"] = frame["institution_id"].astype(str).str.strip()
        frame["name"] = frame["name"].astype(str).str.strip()
        if not frame["institution_id"].eq("NATIONAL").all():
            raise ValueError(f"{path.name}: institution_id must be NATIONAL")
        frame["reported_year"] = frame["year"]
        frame["analysis_year"] = analysis_year
        frame["is_off_year_table_row"] = frame["reported_year"].ne(analysis_year)
        if frame["is_off_year_table_row"].any():
            audit = frame[
                ["institution_id", "name", "reported_year", "analysis_year"]
            ].copy()
            audit["source_file"] = path.name
            audit["source_type"] = "national_baseline"
            off_year_rows.append(audit)
        for metric in core.METRICS:
            if metric["key"] == "HPA_MFA_CIS":
                continue
            core.validate_rate(
                frame,
                metric["numerator"],
                metric["denominator"],
                metric["rate"],
                path.name,
            )
        core.validate_rate(
            frame,
            "N_articles_any_risk",
            "N_articles_attributed",
            "AnyRisk_rate",
            path.name,
        )
        if not (
            frame["CA_known"]
            + frame["CA_unknown_or_unmatched"]
            == frame["N_articles_attributed"]
        ).all():
            raise ValueError(
                f"{path.name}: known and unknown corresponding-author counts "
                "do not sum to national output"
            )
        if not (
            frame["N_articles_CA_led"] + frame["N_articles_non_led"]
            == frame["CA_known"]
        ).all():
            raise ValueError(
                f"{path.name}: led and non-led counts do not sum to CA_known"
            )
        audits.append(
            {
                "analysis_year": analysis_year,
                "source_file": path.name,
                "national_rows": len(frame),
                "off_year_rows": int(frame["is_off_year_table_row"].sum()),
                "N_articles_attributed": int(frame["N_articles_attributed"].iloc[0]),
                "rate_checks_passed": True,
                "leadership_partition_valid": True,
            }
        )
        frames.append(frame)
    national = pd.concat(frames, ignore_index=True, sort=False)
    off_year = (
        pd.concat(off_year_rows, ignore_index=True, sort=False)
        if off_year_rows
        else pd.DataFrame(
            columns=[
                "institution_id",
                "name",
                "reported_year",
                "analysis_year",
                "source_file",
                "source_type",
            ]
        )
    )
    return national, off_year, pd.DataFrame(audits)


def load_article_flag_aggregates(
    core,
    input_dir: Path,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "eid",
        "institution_id",
        "name",
        "year",
        "analysis_year",
        "CA_known",
        "flag_IMFA",
        "flag_CIS",
        "flag_HPA",
        "flag_HAO",
        "flag_NLO",
        "flag_any_risk",
    }
    binary_columns = [
        "CA_known",
        "flag_IMFA",
        "flag_CIS",
        "flag_HPA",
        "flag_HAO",
        "flag_NLO",
        "flag_any_risk",
    ]
    aggregate_frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    off_year_frames: list[pd.DataFrame] = []
    for analysis_year in range(start_year, end_year + 1):
        path = input_dir / f"{analysis_year}_article_institution_flags.csv"
        frame = core.read_csv(path)
        core.require_columns(frame, required, path.name)
        frame["institution_id"] = frame["institution_id"].astype(str).str.strip()
        frame["eid"] = frame["eid"].astype(str).str.strip()
        if frame[["institution_id", "eid"]].isna().any().any():
            raise ValueError(f"{path.name}: missing institution_id or eid")
        for column in ["year", "analysis_year", *binary_columns]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[["analysis_year", *binary_columns]].isna().any().any():
            raise ValueError(f"{path.name}: invalid analysis year or binary flag")
        for column in binary_columns:
            values = set(frame[column].astype(int).unique().tolist())
            if not values.issubset({0, 1}):
                raise ValueError(
                    f"{path.name}: {column} contains values outside 0/1"
                )
            frame[column] = frame[column].astype(int)
        duplicate_rows = frame.duplicated(
            subset=["institution_id", "eid"], keep=False
        )
        if duplicate_rows.any():
            raise ValueError(
                f"{path.name}: duplicate article-institution rows detected"
            )
        off_year = frame.loc[
            frame["analysis_year"].ne(analysis_year),
            ["institution_id", "name", "eid", "year", "analysis_year"],
        ].copy()
        if not off_year.empty:
            off_year["reported_analysis_year"] = off_year["analysis_year"]
            off_year["analysis_year"] = analysis_year
            off_year["source_file"] = path.name
            off_year["source_type"] = "article_institution_flags"
            off_year_frames.append(off_year)
        frame["analysis_year"] = analysis_year
        non_nlo_flags = ["flag_IMFA", "flag_CIS", "flag_HPA", "flag_HAO"]
        frame["flag_any_excluding_NLO"] = (
            frame[non_nlo_flags].max(axis=1).gt(0).astype(int)
        )
        recomputed_any = (
            frame[[*non_nlo_flags, "flag_NLO"]].max(axis=1).gt(0).astype(int)
        )
        if not recomputed_any.eq(frame["flag_any_risk"]).all():
            raise ValueError(
                f"{path.name}: flag_any_risk disagrees with its components"
            )
        frame["flag_confirmed_no_indicator"] = (
            frame["flag_any_excluding_NLO"].eq(0)
            & frame["CA_known"].eq(1)
            & frame["flag_NLO"].eq(0)
        ).astype(int)
        frame["flag_union_NLO_unknown"] = (
            frame["flag_any_excluding_NLO"].eq(0)
            & frame["CA_known"].eq(0)
        ).astype(int)
        frame["flag_confirmed_non_NLO"] = (
            frame["CA_known"].eq(1) & frame["flag_NLO"].eq(0)
        ).astype(int)
        frame["flag_NLO_unknown"] = frame["CA_known"].eq(0).astype(int)
        grouped = (
            frame.groupby(
                ["institution_id", "name", "analysis_year"],
                as_index=False,
                dropna=False,
            )
            .agg(
                N_articles_attributed_flags=("eid", "nunique"),
                N_articles_IMFA_flags=("flag_IMFA", "sum"),
                N_articles_CIS_flags=("flag_CIS", "sum"),
                N_articles_HPA_flags=("flag_HPA", "sum"),
                N_articles_HAO_flags=("flag_HAO", "sum"),
                N_articles_NLO_flags=("flag_NLO", "sum"),
                N_articles_any_risk_flags=("flag_any_risk", "sum"),
                N_articles_any_excluding_NLO=(
                    "flag_any_excluding_NLO",
                    "sum",
                ),
                N_articles_confirmed_no_indicator=(
                    "flag_confirmed_no_indicator",
                    "sum",
                ),
                N_articles_union_NLO_unknown=(
                    "flag_union_NLO_unknown",
                    "sum",
                ),
                N_articles_confirmed_non_NLO=(
                    "flag_confirmed_non_NLO",
                    "sum",
                ),
                N_articles_NLO_unknown=("flag_NLO_unknown", "sum"),
                CA_known_flags=("CA_known", "sum"),
            )
        )
        union_partition = (
            grouped["N_articles_any_risk_flags"]
            + grouped["N_articles_confirmed_no_indicator"]
            + grouped["N_articles_union_NLO_unknown"]
        )
        nlo_partition = (
            grouped["N_articles_NLO_flags"]
            + grouped["N_articles_confirmed_non_NLO"]
            + grouped["N_articles_NLO_unknown"]
        )
        if not union_partition.eq(
            grouped["N_articles_attributed_flags"]
        ).all():
            raise ValueError(f"{path.name}: three-state union partition failed")
        if not nlo_partition.eq(
            grouped["N_articles_attributed_flags"]
        ).all():
            raise ValueError(f"{path.name}: three-state NLO partition failed")
        grouped["AnyExcludingNLO_rate"] = (
            grouped["N_articles_any_excluding_NLO"]
            / grouped["N_articles_attributed_flags"]
        )
        aggregate_frames.append(grouped)
        audits.append(
            {
                "analysis_year": analysis_year,
                "source_file": path.name,
                "source_rows": len(frame),
                "article_institution_rows": len(frame),
                "unique_units": frame["institution_id"].nunique(),
                "national_rows": int(
                    frame["institution_id"].eq("NATIONAL").sum()
                ),
                "duplicate_article_institution_rows": int(
                    duplicate_rows.sum()
                ),
                "off_year_analysis_rows": len(off_year),
                "union_partition_valid": True,
                "nlo_partition_valid": True,
            }
        )
    aggregates = pd.concat(aggregate_frames, ignore_index=True, sort=False)
    off_year_rows = (
        pd.concat(off_year_frames, ignore_index=True, sort=False)
        if off_year_frames
        else pd.DataFrame(
            columns=[
                "institution_id",
                "name",
                "eid",
                "year",
                "reported_analysis_year",
                "analysis_year",
                "source_file",
                "source_type",
            ]
        )
    )
    return aggregates, pd.DataFrame(audits), off_year_rows


def attach_and_validate_article_flag_counts(
    annual: pd.DataFrame,
    national: pd.DataFrame,
    flag_counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed = pd.concat(
        [
            annual.assign(analysis_level="institution"),
            national.assign(analysis_level="national"),
        ],
        ignore_index=True,
        sort=False,
    )
    comparisons = (
        observed[
            [
                "institution_id",
                "analysis_year",
                "analysis_level",
                "N_articles_attributed",
                "N_articles_IMFA",
                "N_articles_CIS",
                "N_articles_with_hyperprolific_author",
                "N_articles_hyperauthored",
                "N_articles_non_led",
                "N_articles_any_risk",
                "CA_known",
            ]
        ]
        .merge(
            flag_counts,
            on=["institution_id", "analysis_year"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_flags_source"),
        )
    )
    pairs = {
        "N_articles_attributed": "N_articles_attributed_flags",
        "N_articles_IMFA": "N_articles_IMFA_flags",
        "N_articles_CIS": "N_articles_CIS_flags",
        "N_articles_with_hyperprolific_author": "N_articles_HPA_flags",
        "N_articles_hyperauthored": "N_articles_HAO_flags",
        "N_articles_non_led": "N_articles_NLO_flags",
        "N_articles_any_risk": "N_articles_any_risk_flags",
        "CA_known": "CA_known_flags",
    }
    audit_rows: list[pd.DataFrame] = []
    for metric_column, flags_column in pairs.items():
        audit = comparisons[
            ["institution_id", "analysis_year", "analysis_level"]
        ].copy()
        audit["metric"] = metric_column
        audit["metrics_table_value"] = comparisons[metric_column]
        audit["article_flags_value"] = comparisons[flags_column]
        audit["difference"] = (
            comparisons[flags_column] - comparisons[metric_column]
        )
        audit["equal"] = audit["difference"].abs().le(1e-9)
        audit_rows.append(audit)
    validation = pd.concat(audit_rows, ignore_index=True, sort=False)
    if validation["article_flags_value"].isna().any():
        raise ValueError("Article flags are missing for an observed unit-year")
    if not validation["equal"].all():
        bad = validation.loc[~validation["equal"]]
        raise ValueError(
            "Article-level flags disagree with annual metrics:\n"
            + bad.head(30).to_string(index=False)
        )
    add_columns = [
        "institution_id",
        "analysis_year",
        "N_articles_any_excluding_NLO",
        "AnyExcludingNLO_rate",
        "N_articles_confirmed_no_indicator",
        "N_articles_union_NLO_unknown",
        "N_articles_confirmed_non_NLO",
        "N_articles_NLO_unknown",
    ]
    annual_augmented = annual.merge(
        flag_counts[add_columns],
        on=["institution_id", "analysis_year"],
        how="left",
        validate="one_to_one",
    )
    national_augmented = national.merge(
        flag_counts[add_columns],
        on=["institution_id", "analysis_year"],
        how="left",
        validate="one_to_one",
    )
    return annual_augmented, national_augmented, validation


def compare_national_sources(
    national_baseline: pd.DataFrame,
    national_from_metrics: pd.DataFrame,
) -> pd.DataFrame:
    relevant = sorted(
        {
            "N_articles_attributed",
            "N_articles_any_risk",
            "AnyRisk_rate",
            "CA_known",
            "CA_unknown_or_unmatched",
            "N_articles_CA_led",
            "N_articles_non_led",
            *[
                column
                for metric in ACCOUNTING_METRICS
                if metric["key"] not in {"ANY_RISK", "HPA_MFA_CIS"}
                for column in (
                    metric["numerator"],
                    metric["denominator"],
                    metric["rate"],
                )
            ],
        }
    )
    rows: list[dict[str, object]] = []
    left = national_baseline.set_index("analysis_year")
    right = national_from_metrics.set_index("analysis_year")
    for year in sorted(left.index):
        for column in relevant:
            if column not in left.columns or column not in right.columns:
                continue
            baseline_value = left.at[year, column]
            metrics_value = right.at[year, column]
            if pd.isna(baseline_value) and pd.isna(metrics_value):
                equal = True
                difference = np.nan
            elif pd.api.types.is_number(baseline_value) and pd.api.types.is_number(
                metrics_value
            ):
                difference = float(baseline_value) - float(metrics_value)
                equal = bool(abs(difference) <= 1e-12)
            else:
                difference = np.nan
                equal = str(baseline_value) == str(metrics_value)
            rows.append(
                {
                    "analysis_year": year,
                    "column": column,
                    "national_baseline_value": baseline_value,
                    "institution_metrics_national_value": metrics_value,
                    "difference": difference,
                    "equal_within_tolerance": equal,
                }
            )
    audit = pd.DataFrame(rows)
    if not audit["equal_within_tolerance"].all():
        bad = audit.loc[~audit["equal_within_tolerance"]]
        raise ValueError(
            "National baseline and NATIONAL rows in institution metrics differ:\n"
            + bad.to_string(index=False)
        )
    return audit


def complete_institution_year_counts(
    annual: pd.DataFrame,
    derived: pd.DataFrame,
    mapping: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    derived_columns = [
        "institution_id",
        "analysis_year",
        "N_articles_by_HPA_MFA_or_CIS",
        "HPA_MFA_or_CIS_article_rate",
    ]
    frame = annual.merge(
        derived[derived_columns],
        on=["institution_id", "analysis_year"],
        how="left",
        validate="one_to_one",
    )
    grid = pd.MultiIndex.from_product(
        [
            mapping["institution_id"].tolist(),
            range(start_year, end_year + 1),
        ],
        names=["institution_id", "analysis_year"],
    ).to_frame(index=False)
    frame = grid.merge(
        frame,
        on=["institution_id", "analysis_year"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    frame["row_imputed_as_zero_output"] = frame["_merge"].eq("left_only")
    frame = frame.drop(columns="_merge")
    frame = frame.drop(
        columns=["id-paper", "color_hex"], errors="ignore"
    ).merge(
        mapping[["institution_id", "name", "id-paper", "color_hex"]],
        on="institution_id",
        how="left",
        suffixes=("_data", ""),
        validate="many_to_one",
    )
    frame = frame.drop(columns=["name_data"], errors="ignore")
    count_columns = sorted(
        {
            "N_articles_attributed",
            "N_articles_any_risk",
            "CA_known",
            "CA_unknown_or_unmatched",
            "N_articles_CA_led",
            "N_articles_non_led",
            "N_articles_confirmed_no_indicator",
            "N_articles_union_NLO_unknown",
            "N_articles_confirmed_non_NLO",
            "N_articles_NLO_unknown",
            *[metric["numerator"] for metric in ACCOUNTING_METRICS],
        }
    )
    for column in count_columns:
        if column not in frame.columns:
            raise ValueError(f"Accounting input lacks required count: {column}")
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    if (frame[count_columns] < 0).any().any():
        raise ValueError("Negative counts found after completing institution years")
    return frame.sort_values(["institution_id", "analysis_year"]).reset_index(
        drop=True
    )


def complete_national_counts(
    national: pd.DataFrame,
    derived: pd.DataFrame,
) -> pd.DataFrame:
    frame = national.merge(
        derived[
            [
                "institution_id",
                "analysis_year",
                "N_articles_by_HPA_MFA_or_CIS",
                "HPA_MFA_or_CIS_article_rate",
            ]
        ],
        on=["institution_id", "analysis_year"],
        how="left",
        validate="one_to_one",
    )
    frame["row_imputed_as_zero_output"] = False
    frame["id-paper"] = "National"
    frame["color_hex"] = "#111111"
    return frame


def accounting_phases(
    start_year: int,
    expansion_end_year: int,
    end_year: int,
) -> tuple[dict[str, object], dict[str, object]]:
    return (
        {
            "phase": "Expansion (2012-2018)",
            "display_period": f"{start_year}-{expansion_end_year}",
            "start_year": start_year,
            "end_year": expansion_end_year,
        },
        {
            "phase": "Stabilization (2019-2025)",
            "display_period": f"{expansion_end_year}-{end_year}",
            "start_year": expansion_end_year,
            "end_year": end_year,
        },
    )


def classify_accounting_pattern(
    output_start: float,
    output_end: float,
    net_change: float,
    indicator_change: float,
    other_change: float,
) -> str:
    if output_start == 0 and output_end > 0:
        return "entry_from_zero_output"
    if output_start > 0 and output_end == 0:
        return "exit_to_zero_output"
    if net_change > 0:
        if indicator_change > 0 and other_change <= 0:
            return "growth_strictly_supported_by_indicator_positive_output"
        if indicator_change > 0 and other_change > 0:
            if indicator_change > other_change:
                return "growth_indicator_positive_component_dominant"
            if indicator_change == other_change:
                return "growth_components_equal"
            return "growth_remaining_output_component_dominant"
        if indicator_change == 0:
            return "growth_without_change_in_indicator_positive_output"
        return "growth_despite_decline_in_indicator_positive_output"
    if net_change < 0:
        if indicator_change < 0 and other_change >= 0:
            return "contraction_strictly_concentrated_in_indicator_positive_output"
        if indicator_change < 0 and other_change < 0:
            if abs(indicator_change) > abs(other_change):
                return "contraction_indicator_positive_component_dominant"
            if abs(indicator_change) == abs(other_change):
                return "contraction_components_equal"
            return "contraction_remaining_output_component_dominant"
        if indicator_change == 0:
            return "contraction_without_change_in_indicator_positive_output"
        return "contraction_despite_increase_in_indicator_positive_output"
    if indicator_change == 0:
        return "no_net_or_compositional_change"
    return "reconfiguration_without_net_output_change"


def accounting_record(
    institution_id: str,
    unit_type: str,
    start: pd.Series,
    end: pd.Series,
    metric: dict[str, object],
    first: int,
    last: int,
    phase: str,
    display_period: str,
    interval_type: str,
) -> dict[str, object]:
    numerator = str(metric["numerator"])
    denominator = str(metric["denominator"])
    output_start = float(start["N_articles_attributed"])
    output_end = float(end["N_articles_attributed"])
    net_change = output_end - output_start
    positive_start = float(start[numerator])
    positive_end = float(end[numerator])
    positive_change = positive_end - positive_start
    other_start = output_start - positive_start
    other_end = output_end - positive_end
    other_change = other_end - other_start
    eligible_start = float(start[denominator])
    eligible_end = float(end[denominator])
    output_share_start = (
        positive_start / output_start if output_start > 0 else np.nan
    )
    output_share_end = (
        positive_end / output_end if output_end > 0 else np.nan
    )
    contribution = (
        100.0 * positive_change / net_change if net_change != 0 else np.nan
    )
    relative_change = (
        100.0 * net_change / output_start if output_start > 0 else np.nan
    )
    expected_constant_composition = (
        output_share_start * net_change
        if not pd.isna(output_share_start)
        else np.nan
    )
    compositional_reallocation = (
        positive_change - expected_constant_composition
        if not pd.isna(expected_constant_composition)
        else np.nan
    )
    direction_consistent = bool(
        net_change != 0
        and positive_change != 0
        and np.sign(net_change) == np.sign(positive_change)
    )
    strict_growth = bool(
        net_change > 0 and positive_change > 0 and other_change <= 0
    )
    strict_contraction = bool(
        net_change < 0 and positive_change < 0 and other_change >= 0
    )
    return {
        "unit_type": unit_type,
        "institution_id": institution_id,
        "name": start["name"],
        "id-paper": start["id-paper"],
        "color_hex": start["color_hex"],
        "interval_type": interval_type,
        "phase": phase,
        "display_period": display_period,
        "start_year": first,
        "end_year": last,
        "metric_key": metric["key"],
        "metric_label": metric["label"],
        "output_start": output_start,
        "output_end": output_end,
        "net_output_change": net_change,
        "relative_output_change_pct": relative_change,
        "absolute_output_change": abs(net_change),
        "indicator_positive_start": positive_start,
        "indicator_positive_end": positive_end,
        "indicator_positive_change": positive_change,
        "remaining_output_start": other_start,
        "remaining_output_end": other_end,
        "remaining_output_change": other_change,
        "other_output_start": other_start,
        "other_output_end": other_end,
        "indicator_share_of_net_change_pct": contribution,
        "indicator_output_share_start_pct": (
            100.0 * output_share_start
            if not pd.isna(output_share_start)
            else np.nan
        ),
        "indicator_output_share_end_pct": (
            100.0 * output_share_end
            if not pd.isna(output_share_end)
            else np.nan
        ),
        "indicator_output_share_change_pp": (
            100.0 * (output_share_end - output_share_start)
            if not pd.isna(output_share_start)
            and not pd.isna(output_share_end)
            else np.nan
        ),
        "contribution_minus_start_output_share_pp": (
            contribution - 100.0 * output_share_start
            if not pd.isna(contribution)
            and not pd.isna(output_share_start)
            else np.nan
        ),
        "expected_indicator_change_at_start_composition": (
            expected_constant_composition
        ),
        "indicator_change_due_to_compositional_reallocation": (
            compositional_reallocation
        ),
        "eligible_denominator_start": eligible_start,
        "eligible_denominator_end": eligible_end,
        "indicator_rate_start_pct": (
            100.0 * positive_start / eligible_start
            if eligible_start > 0
            else np.nan
        ),
        "indicator_rate_end_pct": (
            100.0 * positive_end / eligible_end
            if eligible_end > 0
            else np.nan
        ),
        "indicator_rate_change_pp": (
            100.0 * positive_end / eligible_end
            - 100.0 * positive_start / eligible_start
            if eligible_start > 0 and eligible_end > 0
            else np.nan
        ),
        "observed_output_growth_pct": (
            100.0 * (output_end / output_start - 1.0)
            if output_start > 0
            else np.nan
        ),
        "output_growth_without_indicator_pct": (
            100.0 * (other_end / other_start - 1.0)
            if other_start > 0
            else np.nan
        ),
        "change_direction_consistent_with_indicator": direction_consistent,
        "indicator_majority_of_net_change": bool(
            direction_consistent
            and not pd.isna(contribution)
            and contribution >= 50.0
        ),
        "indicator_at_least_75pct_of_net_change": bool(
            direction_consistent
            and not pd.isna(contribution)
            and contribution >= 75.0
        ),
        "indicator_at_least_80pct_of_net_change": bool(
            direction_consistent
            and not pd.isna(contribution)
            and contribution >= 80.0
        ),
        "indicator_at_least_90pct_of_net_change": bool(
            direction_consistent
            and not pd.isna(contribution)
            and contribution >= 90.0
        ),
        "strict_indicator_supported_growth": strict_growth,
        "strict_indicator_concentrated_contraction": strict_contraction,
        "accounting_pattern": classify_accounting_pattern(
            output_start,
            output_end,
            net_change,
            positive_change,
            other_change,
        ),
        "both_endpoint_outputs_positive": bool(
            output_start > 0 and output_end > 0
        ),
        "small_net_change_abs_lt_10": bool(abs(net_change) < 10),
        "contribution_defined": bool(net_change != 0),
        "CA_known_start": float(start["CA_known"]),
        "CA_known_end": float(end["CA_known"]),
        "CA_unknown_start": float(start["CA_unknown_or_unmatched"]),
        "CA_unknown_end": float(end["CA_unknown_or_unmatched"]),
        "CA_coverage_start_pct": (
            100.0 * float(start["CA_known"]) / output_start
            if output_start > 0
            else np.nan
        ),
        "CA_coverage_end_pct": (
            100.0 * float(end["CA_known"]) / output_end
            if output_end > 0
            else np.nan
        ),
        "locally_led_start": float(start["N_articles_CA_led"]),
        "locally_led_end": float(end["N_articles_CA_led"]),
        "union_confirmed_negative_start": float(
            start["N_articles_confirmed_no_indicator"]
        ),
        "union_confirmed_negative_end": float(
            end["N_articles_confirmed_no_indicator"]
        ),
        "union_NLO_unknown_start": float(
            start["N_articles_union_NLO_unknown"]
        ),
        "union_NLO_unknown_end": float(
            end["N_articles_union_NLO_unknown"]
        ),
        "NLO_confirmed_negative_start": float(
            start["N_articles_confirmed_non_NLO"]
        ),
        "NLO_confirmed_negative_end": float(
            end["N_articles_confirmed_non_NLO"]
        ),
        "NLO_unknown_start": float(start["N_articles_NLO_unknown"]),
        "NLO_unknown_end": float(end["N_articles_NLO_unknown"]),
        "start_row_imputed_as_zero_output": bool(
            start["row_imputed_as_zero_output"]
        ),
        "end_row_imputed_as_zero_output": bool(
            end["row_imputed_as_zero_output"]
        ),
        "contribution_interpretation": (
            "indicator-specific descriptive accounting; overlapping "
            "indicators are not additive"
        ),
        "script_version": SCRIPT_VERSION,
    }


def build_phase_accounting(
    counts: pd.DataFrame,
    unit_ids: Sequence[str],
    unit_type: str,
    start_year: int,
    expansion_end_year: int,
    end_year: int,
) -> pd.DataFrame:
    indexed = counts.set_index(["institution_id", "analysis_year"])
    rows: list[dict[str, object]] = []
    for institution_id in unit_ids:
        for phase in accounting_phases(
            start_year, expansion_end_year, end_year
        ):
            first = int(phase["start_year"])
            last = int(phase["end_year"])
            start = indexed.loc[(institution_id, first)]
            end = indexed.loc[(institution_id, last)]
            for metric in ACCOUNTING_METRICS:
                rows.append(
                    accounting_record(
                        institution_id=institution_id,
                        unit_type=unit_type,
                        start=start,
                        end=end,
                        metric=metric,
                        first=first,
                        last=last,
                        phase=str(phase["phase"]),
                        display_period=str(phase["display_period"]),
                        interval_type="phase_endpoint",
                    )
                )
    accounting = pd.DataFrame(rows)
    identity_error = (
        accounting["indicator_positive_change"]
        + accounting["remaining_output_change"]
        - accounting["net_output_change"]
    ).abs()
    if identity_error.gt(1e-9).any():
        raise ValueError("Phase accounting identity failed")
    return accounting


def build_annual_transition_accounting(
    counts: pd.DataFrame,
    unit_ids: Sequence[str],
    unit_type: str,
    start_year: int,
    end_year: int,
    expansion_end_year: int,
) -> pd.DataFrame:
    indexed = counts.set_index(["institution_id", "analysis_year"])
    rows: list[dict[str, object]] = []
    for institution_id in unit_ids:
        for current_year in range(start_year + 1, end_year + 1):
            previous_year = current_year - 1
            start = indexed.loc[(institution_id, previous_year)]
            end = indexed.loc[(institution_id, current_year)]
            phase = (
                f"Expansion ({start_year + 1}-{expansion_end_year})"
                if current_year <= expansion_end_year
                else f"Stabilization ({expansion_end_year + 1}-{end_year})"
            )
            for metric in ACCOUNTING_METRICS:
                rows.append(
                    accounting_record(
                        institution_id=institution_id,
                        unit_type=unit_type,
                        start=start,
                        end=end,
                        metric=metric,
                        first=previous_year,
                        last=current_year,
                        phase=phase,
                        display_period=f"{previous_year}-{current_year}",
                        interval_type="annual_transition",
                    )
                )
    accounting = pd.DataFrame(rows)
    identity_error = (
        accounting["indicator_positive_change"]
        + accounting["remaining_output_change"]
        - accounting["net_output_change"]
    ).abs()
    if identity_error.gt(1e-9).any():
        raise ValueError("Annual transition accounting identity failed")
    return accounting


def add_robust_change_screening(
    accounting: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add two-sided robust D screening to institutional composition changes.

    The screened quantity is the change in the indicator rate in percentage
    points. For all metrics except NLO this is also the change in share of
    attributed output. For NLO, the rate uses publications with identified
    corresponding-author affiliations as its denominator.

    Reference distributions are interval- and metric-specific. Both endpoint
    denominators must contain at least 50 publications. For HPA, HAO, and
    HPA-MFA/CIS, transitions with a positive numerator at either endpoint
    define the reference distribution so structural zero-to-zero transitions
    do not collapse the robust scale; every otherwise eligible transition is
    still evaluated.
    """
    data = accounting.copy()
    if not data["unit_type"].eq("institution").all():
        raise ValueError(
            "Robust change screening must be calculated from institutional "
            "rows only."
        )

    data["robust_change_value_pp"] = data["indicator_rate_change_pp"]
    data["robust_change_denominator"] = np.where(
        data["metric_key"].eq("NLO"),
        "publications with identified corresponding-author affiliation",
        "attributed publications",
    )
    data["robust_change_eligible"] = (
        data["both_endpoint_outputs_positive"]
        & data["eligible_denominator_start"].ge(
            ROBUST_D_MIN_DENOMINATOR
        )
        & data["eligible_denominator_end"].ge(
            ROBUST_D_MIN_DENOMINATOR
        )
        & data["robust_change_value_pp"].notna()
    )
    active_transition = (
        data["indicator_positive_start"].gt(0)
        | data["indicator_positive_end"].gt(0)
    )
    active_reference_required = data["metric_key"].isin(
        ROBUST_D_ACTIVE_REFERENCE_METRICS
    )
    data["robust_change_active_reference_required"] = (
        active_reference_required
    )
    data["robust_change_reference_member"] = (
        data["robust_change_eligible"]
        & (~active_reference_required | active_transition)
    )

    group_keys = [
        "interval_type",
        "start_year",
        "end_year",
        "metric_key",
    ]
    statistic_columns = [
        "robust_change_reference_n",
        "robust_change_evaluation_n",
        "robust_change_reference_distinct_n",
        "robust_change_reference_median_pp",
        "robust_change_reference_q1_pp",
        "robust_change_reference_q3_pp",
        "robust_change_reference_iqr_pp",
        "robust_change_reference_scale_pp",
        "robust_change_reference_status",
        "robust_change_reference_valid",
    ]
    for column in statistic_columns:
        data[column] = np.nan
    data["robust_change_reference_status"] = "not_calculated"
    data["robust_change_reference_valid"] = False
    data["robust_change_D"] = np.nan

    for _, group in data.groupby(group_keys, sort=False):
        reference = group.loc[
            group["robust_change_reference_member"],
            "robust_change_value_pp",
        ].dropna()
        evaluation_n = int(group["robust_change_eligible"].sum())
        reference_n = int(len(reference))
        distinct_n = int(reference.nunique())
        median = float(reference.median()) if reference_n else np.nan
        q1 = float(reference.quantile(0.25)) if reference_n else np.nan
        q3 = float(reference.quantile(0.75)) if reference_n else np.nan
        iqr = q3 - q1 if reference_n else np.nan
        if reference_n < ROBUST_D_MIN_REFERENCE_N:
            status = "insufficient_reference_n"
        elif distinct_n < 2:
            status = "insufficient_distinct_values"
        elif pd.isna(iqr) or iqr <= 0:
            status = "nonpositive_iqr"
        else:
            status = "valid"
        valid = status == "valid"
        scale = (
            iqr / ROBUST_D_NORMAL_CONSISTENCY if valid else np.nan
        )
        indexes = group.index
        values = {
            "robust_change_reference_n": reference_n,
            "robust_change_evaluation_n": evaluation_n,
            "robust_change_reference_distinct_n": distinct_n,
            "robust_change_reference_median_pp": median,
            "robust_change_reference_q1_pp": q1,
            "robust_change_reference_q3_pp": q3,
            "robust_change_reference_iqr_pp": iqr,
            "robust_change_reference_scale_pp": scale,
            "robust_change_reference_status": status,
            "robust_change_reference_valid": valid,
        }
        for column, value in values.items():
            data.loc[indexes, column] = value
        if valid:
            eligible_indexes = group.index[
                group["robust_change_eligible"]
            ]
            data.loc[eligible_indexes, "robust_change_D"] = (
                data.loc[eligible_indexes, "robust_change_value_pp"]
                - median
            ) / scale

    data["robust_change_signal"] = (
        data["robust_change_D"].abs().ge(ROBUST_D_ELEVATED)
    )
    data["robust_change_strong_signal"] = (
        data["robust_change_D"].abs().ge(ROBUST_D_STRONG)
    )
    data["robust_change_signal_direction"] = np.select(
        [
            data["robust_change_D"].ge(ROBUST_D_ELEVATED),
            data["robust_change_D"].le(-ROBUST_D_ELEVATED),
            data["robust_change_D"].notna(),
        ],
        ["upper", "lower", "none"],
        default="not_evaluated",
    )
    data["robust_change_signal_level"] = np.select(
        [
            data["robust_change_D"].abs().ge(ROBUST_D_STRONG),
            data["robust_change_D"].abs().ge(ROBUST_D_ELEVATED),
            data["robust_change_D"].notna(),
        ],
        ["strong", "elevated", "none"],
        default="not_evaluated",
    )
    data["robust_change_signal_label"] = np.select(
        [
            data["robust_change_D"].ge(ROBUST_D_STRONG),
            data["robust_change_D"].ge(ROBUST_D_ELEVATED),
            data["robust_change_D"].le(-ROBUST_D_STRONG),
            data["robust_change_D"].le(-ROBUST_D_ELEVATED),
            data["robust_change_D"].notna(),
        ],
        [
            "strong_upper",
            "elevated_upper",
            "strong_lower",
            "elevated_lower",
            "none",
        ],
        default="not_evaluated",
    )

    summary_rows: list[dict[str, object]] = []
    for keys, group in data.groupby(group_keys, sort=False):
        interval_type, first, last, metric_key = keys
        summary_rows.append(
            {
                "interval_type": interval_type,
                "start_year": first,
                "end_year": last,
                "display_period": group["display_period"].iloc[0],
                "phase": group["phase"].iloc[0],
                "metric_key": metric_key,
                "metric_label": group["metric_label"].iloc[0],
                "screened_quantity": "indicator_rate_change_pp",
                "reference_n": int(
                    group["robust_change_reference_n"].iloc[0]
                ),
                "evaluation_n": int(
                    group["robust_change_evaluation_n"].iloc[0]
                ),
                "reference_distinct_n": int(
                    group["robust_change_reference_distinct_n"].iloc[0]
                ),
                "reference_median_pp": group[
                    "robust_change_reference_median_pp"
                ].iloc[0],
                "reference_q1_pp": group[
                    "robust_change_reference_q1_pp"
                ].iloc[0],
                "reference_q3_pp": group[
                    "robust_change_reference_q3_pp"
                ].iloc[0],
                "reference_iqr_pp": group[
                    "robust_change_reference_iqr_pp"
                ].iloc[0],
                "reference_scale_pp": group[
                    "robust_change_reference_scale_pp"
                ].iloc[0],
                "reference_status": group[
                    "robust_change_reference_status"
                ].iloc[0],
                "reference_valid": bool(
                    group["robust_change_reference_valid"].iloc[0]
                ),
                "upper_signals_D_ge_2": int(
                    group["robust_change_D"].ge(ROBUST_D_ELEVATED).sum()
                ),
                "strong_upper_signals_D_ge_3": int(
                    group["robust_change_D"].ge(ROBUST_D_STRONG).sum()
                ),
                "lower_signals_D_le_minus_2": int(
                    group["robust_change_D"].le(-ROBUST_D_ELEVATED).sum()
                ),
                "strong_lower_signals_D_le_minus_3": int(
                    group["robust_change_D"].le(-ROBUST_D_STRONG).sum()
                ),
                "active_reference_required": bool(
                    group[
                        "robust_change_active_reference_required"
                    ].iloc[0]
                ),
                "minimum_endpoint_denominator": (
                    ROBUST_D_MIN_DENOMINATOR
                ),
                "minimum_reference_n": ROBUST_D_MIN_REFERENCE_N,
                "normal_consistency_constant": (
                    ROBUST_D_NORMAL_CONSISTENCY
                ),
                "script_version": SCRIPT_VERSION,
            }
        )
    reference_summary = pd.DataFrame(summary_rows)
    return data, reference_summary


def build_three_state_accounting(
    accounting: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in accounting.loc[
        accounting["metric_key"].isin(["ANY_RISK", "NLO"])
    ].iterrows():
        if row["metric_key"] == "ANY_RISK":
            confirmed_start = row["union_confirmed_negative_start"]
            confirmed_end = row["union_confirmed_negative_end"]
            unknown_start = row["union_NLO_unknown_start"]
            unknown_end = row["union_NLO_unknown_end"]
            classification = "union_of_five_indicators"
        else:
            confirmed_start = row["NLO_confirmed_negative_start"]
            confirmed_end = row["NLO_confirmed_negative_end"]
            unknown_start = row["NLO_unknown_start"]
            unknown_end = row["NLO_unknown_end"]
            classification = "NLO"
        confirmed_change = confirmed_end - confirmed_start
        unknown_change = unknown_end - unknown_start
        positive_change = row["indicator_positive_change"]
        identity_error = (
            positive_change
            + confirmed_change
            + unknown_change
            - row["net_output_change"]
        )
        if abs(identity_error) > 1e-9:
            raise ValueError(
                "Three-state accounting identity failed for "
                f"{row['institution_id']} {row['display_period']} "
                f"{row['metric_key']}"
            )
        rows.append(
            {
                "unit_type": row["unit_type"],
                "institution_id": row["institution_id"],
                "name": row["name"],
                "id-paper": row["id-paper"],
                "interval_type": row["interval_type"],
                "phase": row["phase"],
                "display_period": row["display_period"],
                "start_year": row["start_year"],
                "end_year": row["end_year"],
                "classification": classification,
                "positive_start": row["indicator_positive_start"],
                "positive_end": row["indicator_positive_end"],
                "positive_change": positive_change,
                "confirmed_negative_start": confirmed_start,
                "confirmed_negative_end": confirmed_end,
                "confirmed_negative_change": confirmed_change,
                "unknown_start": unknown_start,
                "unknown_end": unknown_end,
                "unknown_change": unknown_change,
                "output_start": row["output_start"],
                "output_end": row["output_end"],
                "net_output_change": row["net_output_change"],
                "positive_share_of_net_change_pct": (
                    row["indicator_share_of_net_change_pct"]
                ),
                "unknown_share_of_net_change_pct": (
                    100.0 * unknown_change / row["net_output_change"]
                    if row["net_output_change"] != 0
                    else np.nan
                ),
                "CA_coverage_start_pct": row["CA_coverage_start_pct"],
                "CA_coverage_end_pct": row["CA_coverage_end_pct"],
                "CA_coverage_change_pp": (
                    row["CA_coverage_end_pct"] - row["CA_coverage_start_pct"]
                    if not pd.isna(row["CA_coverage_start_pct"])
                    and not pd.isna(row["CA_coverage_end_pct"])
                    else np.nan
                ),
                "indicator_rate_start_pct": row["indicator_rate_start_pct"],
                "indicator_rate_end_pct": row["indicator_rate_end_pct"],
                "indicator_rate_change_pp": row["indicator_rate_change_pp"],
                "three_state_identity_error": identity_error,
                "script_version": SCRIPT_VERSION,
            }
        )
    return pd.DataFrame(rows)


def main_figure_data(
    accounting: pd.DataFrame,
    annual_accounting: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    national = accounting.loc[
        accounting["unit_type"].eq("national")
        & accounting["metric_key"].isin(PRIMARY_ACCOUNTING_ORDER)
    ].copy()
    national_annual = annual_accounting.loc[
        annual_accounting["unit_type"].eq("national")
        & annual_accounting["metric_key"].eq("ANY_EXCL_NLO")
    ].copy()
    return national, national_annual


def create_main_figure(
    accounting: pd.DataFrame,
    annual_accounting: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    dpi: int,
) -> None:
    if output_utils.data_only():
        return
    national, national_annual = main_figure_data(
        accounting, annual_accounting
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.6, 5.55),
        gridspec_kw={"width_ratios": [1.0, 1.22], "wspace": 0.42},
    )
    offsets = {
        "Expansion (2012-2018)": 0.15,
        "Stabilization (2019-2025)": -0.15,
    }

    axis = axes[1]
    annual = national_annual.sort_values("end_year")
    x = np.arange(len(annual))
    exposed_change = annual["indicator_positive_change"].to_numpy(float)
    remaining_change = annual["remaining_output_change"].to_numpy(float)
    positive_bottom = np.zeros(len(annual))
    negative_bottom = np.zeros(len(annual))
    for values, color, label in [
        (
            exposed_change,
            "#C46A1A",
            "Positive for IMFA, CIS, HPA, or HAO",
        ),
        (remaining_change, "#B7BDC5", "Remaining articles"),
    ]:
        positive = np.clip(values, 0, None)
        negative = np.clip(values, None, 0)
        axis.bar(
            x,
            positive,
            bottom=positive_bottom,
            width=0.72,
            color=color,
            label=label,
            linewidth=0,
        )
        axis.bar(
            x,
            negative,
            bottom=negative_bottom,
            width=0.72,
            color=color,
            linewidth=0,
        )
        positive_bottom += positive
        negative_bottom += negative
    axis.scatter(
        x,
        annual["net_output_change"],
        marker="D",
        s=16,
        color="#222222",
        zorder=4,
        label="Net change",
    )
    axis.axhline(0, color="#666666", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(
        annual["end_year"].astype(int).astype(str), rotation=45, ha="right"
    )
    axis.set_ylabel("Annual change in article count")
    axis.set_xlabel("Ending year of transition")
    axis.set_title(
        "(b) Annual decomposition of national output change",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
    )
    axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper left",
        frameon=False,
        fontsize=7.3,
        handlelength=1.2,
        borderaxespad=0.2,
    )

    axis = axes[0]
    y = np.arange(len(PRIMARY_ACCOUNTING_ORDER))[::-1]
    for phase in PHASE_COLORS:
        subset = (
            national.loc[national["phase"].eq(phase)]
            .set_index("metric_key")
            .reindex(PRIMARY_ACCOUNTING_ORDER)
        )
        initial_share = subset[
            "indicator_output_share_start_pct"
        ].to_numpy(float)
        final_share = subset[
            "indicator_output_share_end_pct"
        ].to_numpy(float)
        contribution = subset[
            "indicator_share_of_net_change_pct"
        ].to_numpy(float)
        y_values = y + offsets[phase]
        for start_value, end_value, yi in zip(
            initial_share, final_share, y_values
        ):
            axis.plot(
                [start_value, end_value],
                [yi, yi],
                color=PHASE_COLORS[phase],
                linewidth=1.0,
                alpha=0.75,
                zorder=2,
            )
        axis.scatter(
            initial_share,
            y_values,
            s=29,
            facecolor="white",
            edgecolor=PHASE_COLORS[phase],
            linewidth=1.0,
            marker=PHASE_MARKERS[phase],
            zorder=3,
        )
        axis.scatter(
            final_share,
            y_values,
            s=31,
            color=PHASE_COLORS[phase],
            marker=PHASE_MARKERS[phase],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        axis.scatter(
            contribution,
            y_values,
            s=29,
            facecolor="#222222",
            edgecolor="white",
            linewidth=0.45,
            marker="D",
            zorder=4,
        )
    axis.axvline(0, color="#666666", linewidth=0.8)
    axis.set_yticks(y)
    axis.set_yticklabels(
        [ACCOUNTING_LABELS[key] for key in PRIMARY_ACCOUNTING_ORDER]
    )
    axis.set_xlabel(
        "Share of national output or net output change (%)"
    )
    axis.set_title(
        "(a) Phase-level composition",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
    )
    axis.grid(axis="x", color="#E2E2E2", linewidth=0.6)
    axis.set_axisbelow(True)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8.5)

    phase_legend = [
        Line2D(
            [0],
            [0],
            marker=PHASE_MARKERS[phase],
            color="none",
            markerfacecolor=PHASE_COLORS[phase],
            markeredgecolor="white",
            markersize=6,
            label=phase,
        )
        for phase in PHASE_COLORS
    ]
    symbol_legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#555555",
            markersize=6,
            label="Initial output share",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#777777",
            markeredgecolor="white",
            markersize=6,
            label="Final output share",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="#222222",
            markeredgecolor="white",
            markersize=5.5,
            label="Contribution to net change",
        ),
    ]
    fig.legend(
        handles=phase_legend + symbol_legend,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=7.8,
    )
    fig.subplots_adjust(bottom=0.17, top=0.93, left=0.09, right=0.985)
    save_figure_formats(fig, output_dir, prefix, dpi)
    plt.close(fig)


def create_accounting_heatmap(
    accounting: pd.DataFrame,
    mapping: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    dpi: int,
) -> None:
    if output_utils.data_only():
        return
    data = accounting.loc[
        accounting["unit_type"].eq("institution")
        & accounting["metric_key"].isin(ACCOUNTING_ORDER)
    ].copy()
    institutions = mapping.sort_values("id-paper")["institution_id"].tolist()
    labels = (
        mapping.set_index("institution_id")
        .loc[institutions]
        .apply(
            lambda row: f"{row['id-paper']} · {row.name}",
            axis=1,
        )
        .tolist()
    )
    phases = list(PHASE_COLORS)
    cmap = ListedColormap(
        ["#6B8EAA", "#E8EEF2", "#F3D9A4", "#E9A66B", "#C75B45"]
    )
    cmap.set_bad("#F7F7F7")
    bounds = [-1000, 0, 25, 50, 100, 1000]
    norm = BoundaryNorm(bounds, cmap.N)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 10.2),
        sharey=False,
        gridspec_kw={"wspace": 0.25},
    )
    short_labels = {
        "ANY_RISK": "Any",
        "ANY_EXCL_NLO": "Any excl. NLO",
        "IMFA": "IMFA",
        "CIS": "CIS",
        "HPA": "HPA",
        "HAO": "HAO",
        "NLO": "NLO",
        "HPA_MFA_CIS": "HPA-MFA/CIS",
    }
    phase_titles = {
        "Expansion (2012-2018)": "Expansion: 2011–2018",
        "Stabilization (2019-2025)": "Stabilization: 2018–2025",
    }
    image = None
    for axis, phase in zip(axes, phases):
        phase_data = data.loc[data["phase"].eq(phase)]
        contribution_matrix = (
            phase_data
            .pivot(
                index="institution_id",
                columns="metric_key",
                values="indicator_share_of_net_change_pct",
            )
            .reindex(index=institutions, columns=ACCOUNTING_ORDER)
        )
        rate_change_matrix = (
            phase_data
            .pivot(
                index="institution_id",
                columns="metric_key",
                values="robust_change_value_pp",
            )
            .reindex(index=institutions, columns=ACCOUNTING_ORDER)
        )
        robust_d_matrix = (
            phase_data
            .pivot(
                index="institution_id",
                columns="metric_key",
                values="robust_change_D",
            )
            .reindex(index=institutions, columns=ACCOUNTING_ORDER)
        )
        net_change = (
            phase_data.loc[
                phase_data["metric_key"].eq("ANY_EXCL_NLO")
            ]
            .set_index("institution_id")["net_output_change"]
            .reindex(institutions)
        )
        phase_labels = [
            (
                f"{label} ↓"
                if change < 0
                else f"{label} 0"
                if change == 0
                else label
            )
            for label, change in zip(labels, net_change)
        ]
        image = figure_style.heatmap(axis,
            contribution_matrix.to_numpy(float),
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        axis.set_xticks(np.arange(len(ACCOUNTING_ORDER)))
        axis.set_xticklabels(
            [short_labels[key] for key in ACCOUNTING_ORDER],
            rotation=45,
            ha="right",
            fontsize=7.5,
        )
        axis.set_yticks(np.arange(len(institutions)))
        axis.set_yticklabels(phase_labels, fontsize=7.2)
        axis.set_title(phase_titles[phase], fontsize=10, fontweight="bold")
        axis.tick_params(length=0)
        for row_index in range(contribution_matrix.shape[0]):
            for column_index in range(contribution_matrix.shape[1]):
                contribution = contribution_matrix.iat[
                    row_index, column_index
                ]
                rate_change = rate_change_matrix.iat[
                    row_index, column_index
                ]
                robust_d = robust_d_matrix.iat[row_index, column_index]
                contribution_text = (
                    "NA"
                    if pd.isna(contribution)
                    else "0"
                    if abs(contribution) < 0.5
                    else f"{contribution:.0f}"
                )
                rate_change_text = (
                    "NA"
                    if pd.isna(rate_change)
                    else "+0"
                    if abs(rate_change) < 0.5
                    else f"{rate_change:+.0f}"
                )
                text_color = (
                    "white"
                    if not pd.isna(contribution)
                    and (contribution < 0 or contribution >= 100)
                    else "#303030"
                )
                axis.text(
                    column_index,
                    row_index,
                    f"{contribution_text}\n{rate_change_text}",
                    ha="center",
                    va="center",
                    fontsize=5.0,
                    linespacing=0.92,
                    color=text_color,
                )
                if not pd.isna(robust_d) and abs(robust_d) >= ROBUST_D_ELEVATED:
                    edge_color = "#7F1D1D" if robust_d > 0 else "#163A5F"
                    line_width = (
                        1.55
                        if abs(robust_d) >= ROBUST_D_STRONG
                        else 0.75
                    )
                    axis.add_patch(
                        Rectangle(
                            (column_index - 0.48, row_index - 0.48),
                            0.96,
                            0.96,
                            fill=False,
                            edgecolor=edge_color,
                            linewidth=line_width,
                            zorder=5,
                        )
                    )
        for spine in axis.spines.values():
            spine.set_visible(False)
    if image is not None:
        colorbar_axis = fig.add_axes([0.22, 0.035, 0.60, 0.022])
        colorbar = fig.colorbar(
            image,
            cax=colorbar_axis,
            orientation="horizontal",
            ticks=[-500, 12.5, 37.5, 75, 550],
        )
        colorbar.ax.set_xticklabels(
            ["≤0", "0–25", "25–50", "50–100", "≥100"]
        )
        colorbar.set_label(
            "Contribution to net change (%); cells show contribution above "
            "change in indicator rate (pp)",
            fontsize=8.5,
        )
        colorbar.ax.tick_params(labelsize=8)
    signal_legend = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#7F1D1D",
            markeredgewidth=0.9,
            markersize=6,
            label="Upper change signal (D ≥ 2)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#163A5F",
            markeredgewidth=0.9,
            markersize=6,
            label="Lower change signal (D ≤ −2)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#333333",
            markeredgewidth=1.8,
            markersize=6,
            label="Thick border: |D| ≥ 3",
        ),
    ]
    fig.legend(
        handles=signal_legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7.2,
        bbox_to_anchor=(0.5, 0.125),
    )
    fig.text(
        0.5,
        0.102,
        "↓ indicates net contraction; 0 indicates no net output change; "
        "NA indicates an undefined contribution. D was not estimable for "
        "2011–2018 because fewer than eight institutions met both endpoint "
        "eligibility criteria.",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.96, bottom=0.23)
    save_figure_formats(
        fig,
        output_dir,
        f"{prefix}_supplementary_institution_heatmap",
        dpi,
    )
    plt.close(fig)


def create_national_annual_indicator_heatmap(
    annual_accounting: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    dpi: int,
) -> None:
    if output_utils.data_only():
        return
    data = annual_accounting.loc[
        annual_accounting["unit_type"].eq("national")
        & annual_accounting["metric_key"].isin(ACCOUNTING_ORDER)
    ].copy()
    transition_order = (
        data[["start_year", "end_year", "display_period"]]
        .drop_duplicates()
        .sort_values("end_year")["display_period"]
        .tolist()
    )
    contribution_matrix = (
        data.pivot(
            index="metric_key",
            columns="display_period",
            values="indicator_share_of_net_change_pct",
        )
        .reindex(index=ACCOUNTING_ORDER, columns=transition_order)
    )
    initial_share_matrix = (
        data.pivot(
            index="metric_key",
            columns="display_period",
            values="indicator_output_share_start_pct",
        )
        .reindex(index=ACCOUNTING_ORDER, columns=transition_order)
    )
    share_change_matrix = (
        data.pivot(
            index="metric_key",
            columns="display_period",
            values="indicator_output_share_change_pp",
        )
        .reindex(index=ACCOUNTING_ORDER, columns=transition_order)
    )
    cmap = ListedColormap(
        ["#6B8EAA", "#E8EEF2", "#F3E5BC", "#F3C47A", "#E98B62", "#B84335"]
    )
    cmap.set_bad("#F7F7F7")
    bounds = [-1000, 0, 25, 50, 75, 100, 1000]
    norm = BoundaryNorm(bounds, cmap.N)
    fig, axis = plt.subplots(figsize=(12.0, 5.2))
    image = figure_style.heatmap(axis,
        contribution_matrix.to_numpy(float),
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(len(transition_order)))
    axis.set_xticklabels(
        [period.replace("-", "–") for period in transition_order],
        rotation=45,
        ha="right",
        fontsize=7.2,
    )
    axis.set_yticks(np.arange(len(ACCOUNTING_ORDER)))
    axis.set_yticklabels(
        [ACCOUNTING_LABELS[key] for key in ACCOUNTING_ORDER],
        fontsize=8.2,
    )
    for row_index in range(contribution_matrix.shape[0]):
        for column_index in range(contribution_matrix.shape[1]):
            contribution = contribution_matrix.iat[row_index, column_index]
            initial_share = initial_share_matrix.iat[
                row_index, column_index
            ]
            share_change = share_change_matrix.iat[
                row_index, column_index
            ]
            color = (
                "white"
                if not pd.isna(contribution)
                and (contribution >= 100 or contribution < 0)
                else "#303030"
            )
            contribution_text = (
                "NA"
                if pd.isna(contribution)
                else "0"
                if abs(contribution) < 0.5
                else f"{contribution:.0f}"
            )
            initial_text = (
                "NA" if pd.isna(initial_share) else f"{initial_share:.0f}"
            )
            change_text = (
                "NA"
                if pd.isna(share_change)
                else "+0"
                if abs(share_change) < 0.5
                else f"{share_change:+.0f}"
            )
            axis.text(
                column_index,
                row_index,
                f"{contribution_text}; {initial_text}; {change_text}",
                ha="center",
                va="center",
                fontsize=5.45,
                color=color,
            )
    axis.set_title(
        "Annual national composition: contribution; initial share; "
        "change in share",
        fontsize=10.5,
        fontweight="bold",
    )
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(
        image,
        ax=axis,
        orientation="horizontal",
        fraction=0.07,
        pad=0.24,
        ticks=[-500, 12.5, 37.5, 62.5, 87.5, 550],
    )
    colorbar.ax.set_xticklabels(
        ["<0", "0–25", "25–50", "50–75", "75–100", ">100"]
    )
    colorbar.set_label(
        "Contribution to national net output change (%)",
        fontsize=8.5,
    )
    colorbar.ax.tick_params(labelsize=7.5)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.89, bottom=0.31)
    save_figure_formats(
        fig,
        output_dir,
        f"{prefix}_supplementary_national_annual_indicators",
        dpi,
    )
    plt.close(fig)


def create_institution_transition_scatterplots(
    annual_accounting: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    dpi: int,
) -> None:
    if output_utils.data_only():
        return
    data = annual_accounting.loc[
        annual_accounting["unit_type"].eq("institution")
        & annual_accounting["metric_key"].isin(ACCOUNTING_ORDER)
        & annual_accounting["both_endpoint_outputs_positive"]
    ].copy()
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(11.2, 6.8),
        sharex=False,
        sharey=False,
        gridspec_kw={"wspace": 0.28, "hspace": 0.34},
    )
    for axis, metric_key in zip(axes.flat, ACCOUNTING_ORDER):
        metric = data.loc[data["metric_key"].eq(metric_key)]
        for phase in PHASE_COLORS:
            subset = metric.loc[metric["phase"].eq(phase)]
            axis.scatter(
                subset["remaining_output_change"],
                subset["indicator_positive_change"],
                s=9,
                marker=PHASE_MARKERS[phase],
                color="#AEB7BF",
                alpha=0.38,
                linewidth=0,
                rasterized=True,
            )
            for direction, color in [
                ("upper", "#B64034"),
                ("lower", "#2F6F9F"),
            ]:
                signal = subset.loc[
                    subset["robust_change_signal_direction"].eq(direction)
                ]
                if signal.empty:
                    continue
                sizes = np.where(
                    signal["robust_change_strong_signal"], 30, 19
                )
                widths = np.where(
                    signal["robust_change_strong_signal"], 0.95, 0.45
                )
                axis.scatter(
                    signal["remaining_output_change"],
                    signal["indicator_positive_change"],
                    s=sizes,
                    marker=PHASE_MARKERS[phase],
                    facecolor=color,
                    edgecolor="#111111",
                    linewidth=widths,
                    alpha=0.90,
                    zorder=5,
                    rasterized=True,
                )
        strict = metric.loc[
            metric["strict_indicator_supported_growth"]
            | metric["strict_indicator_concentrated_contraction"]
        ]
        axis.scatter(
            strict["remaining_output_change"],
            strict["indicator_positive_change"],
            s=42,
            facecolor="none",
            edgecolor="#1E1E1E",
            linewidth=0.45,
            zorder=6,
        )
        values = pd.concat(
            [
                metric["remaining_output_change"].abs(),
                metric["indicator_positive_change"].abs(),
            ],
            ignore_index=True,
        )
        limit = max(float(values.max()), 10.0)
        diagonal = np.linspace(-limit, limit, 301)
        axis.plot(
            diagonal,
            -diagonal,
            color="#9A9A9A",
            linewidth=0.7,
            linestyle=":",
            zorder=1,
        )
        axis.axhline(0, color="#777777", linewidth=0.55)
        axis.axvline(0, color="#777777", linewidth=0.55)
        axis.set_xscale("symlog", linthresh=10)
        axis.set_yscale("symlog", linthresh=10)
        axis.set_title(
            ACCOUNTING_LABELS[metric_key],
            fontsize=9.2,
            fontweight="bold",
        )
        axis.grid(color="#E5E5E5", linewidth=0.45)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    for axis in axes[-1, :]:
        axis.set_xlabel("Change in remaining articles", fontsize=8)
    for axis in axes[:, 0]:
        axis.set_ylabel("Change in indicator-positive articles", fontsize=8)
    legend = [
        Line2D(
            [0],
            [0],
            marker=PHASE_MARKERS[phase],
            color="none",
            markerfacecolor="#AEB7BF",
            markersize=5,
            label=phase,
        )
        for phase in PHASE_COLORS
    ]
    legend.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#B64034",
                markeredgecolor="#111111",
                markeredgewidth=0.45,
                markersize=5,
                label="Upper change signal (D ≥ 2)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#2F6F9F",
                markeredgecolor="#111111",
                markeredgewidth=0.45,
                markersize=5,
                label="Lower change signal (D ≤ −2)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#B64034",
                markeredgecolor="#111111",
                markeredgewidth=1.2,
                markersize=6,
                label="Strong change signal (|D| ≥ 3)",
            ),
        ]
    )
    legend.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#1E1E1E",
            markersize=6,
            label="Strict accounting pattern",
        )
    )
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7.4,
        bbox_to_anchor=(0.5, -0.018),
    )
    fig.suptitle(
        "Institution-year changes in indicator-positive and remaining output",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.135)
    save_figure_formats(
        fig,
        output_dir,
        f"{prefix}_supplementary_institution_year_changes",
        dpi,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    core = load_core_module()
    input_dir = Path(args.input_dir).expanduser().resolve()
    mapping_path = Path(args.mapping_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.start_year >= args.end_year:
        raise ValueError('--start-year must be smaller than --end-year')
    if not args.start_year <= args.expansion_end_year < args.end_year:
        raise ValueError('--expansion-end-year must fall between start and end years')
    if not input_dir.is_dir():
        raise FileNotFoundError(f'Input directory does not exist: {input_dir}')
    if not mapping_path.is_file():
        raise FileNotFoundError(f'Mapping file does not exist: {mapping_path}')
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded_files = required_revised_inputs(core, input_dir, args.start_year, args.end_year)
    mapping, _ = core.validate_mapping(mapping_path, args.mapping_encoding)
    annual, off_year_metrics, metric_validation, national_from_metrics = core.load_annual_metrics(input_dir, args.start_year, args.end_year)
    annual, name_mismatches = core.attach_mapping(annual, mapping)
    national, off_year_national, national_validation = load_national_baselines(core, input_dir, args.start_year, args.end_year)
    national_comparison = compare_national_sources(national, national_from_metrics)
    article_flag_counts, article_flag_audit, off_year_article_flags = load_article_flag_aggregates(core, input_dir, args.start_year, args.end_year)
    annual, national, article_flag_validation = attach_and_validate_article_flag_counts(annual, national, article_flag_counts)
    institutional_dependency, off_year_hpa_institution, hpa_validation_institution = core.derive_hpa_dependency(input_dir, annual, args.start_year, args.end_year)
    national_dependency, off_year_hpa_national, hpa_validation_national = core.derive_hpa_dependency(input_dir, national, args.start_year, args.end_year)
    institutional_counts = complete_institution_year_counts(annual, institutional_dependency, mapping, args.start_year, args.end_year)
    national_counts = complete_national_counts(national, national_dependency)
    institutional_accounting = build_phase_accounting(institutional_counts, mapping['institution_id'].tolist(), 'institution', args.start_year, args.expansion_end_year, args.end_year)
    national_accounting = build_phase_accounting(national_counts, ['NATIONAL'], 'national', args.start_year, args.expansion_end_year, args.end_year)
    institutional_annual_accounting = build_annual_transition_accounting(institutional_counts, mapping['institution_id'].tolist(), 'institution', args.start_year, args.end_year, args.expansion_end_year)
    national_annual_accounting = build_annual_transition_accounting(national_counts, ['NATIONAL'], 'national', args.start_year, args.end_year, args.expansion_end_year)
    institutional_accounting, phase_robust_change_reference_summary = add_robust_change_screening(institutional_accounting)
    institutional_annual_accounting, annual_robust_change_reference_summary = add_robust_change_screening(institutional_annual_accounting)
    accounting = pd.concat([national_accounting, institutional_accounting], ignore_index=True, sort=False)
    annual_accounting = pd.concat([national_annual_accounting, institutional_annual_accounting], ignore_index=True, sort=False)
    three_state_accounting = build_three_state_accounting(pd.concat([accounting, annual_accounting], ignore_index=True, sort=False))
    outputs = {f'{args.prefix}_phase_accounting_national.csv': national_accounting, f'{args.prefix}_phase_accounting_institutions.csv': institutional_accounting, f'{args.prefix}_annual_transition_accounting_national.csv': national_annual_accounting, f'{args.prefix}_annual_transition_accounting_institutions.csv': institutional_annual_accounting, f'{args.prefix}_three_state_NLO_union_accounting.csv': three_state_accounting}
    for filename, frame in outputs.items():
        write_csv(frame, output_dir / filename)
    print(f'Calculated composition sources in {output_dir}')


if __name__ == "__main__":
    main()
