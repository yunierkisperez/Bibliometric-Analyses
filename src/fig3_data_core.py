#!/usr/bin/env python3
"""
Validated data-loading helpers for the Figure 3 compositional analysis.

The module validates the institution mapping and annual institution-year
metrics, audits rows whose cover-date year differs from the annual filename,
and reconstructs the HPA-MFA/CIS article numerator uniformly for 2011–2025.
The analysis year is always the year encoded in the annual filename.
"""

from __future__ import annotations


import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION = "4.0.0-2026-07-29"

INVALID_TEXT = {"", "nan", "none", "null", "na", "n/a", "<na>"}
COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")

METRICS = (
    {
        "key": "IMFA",
        "label": "IMFA",
        "description": "Articles with institutional multi-affiliation",
        "numerator": "N_articles_IMFA",
        "denominator": "N_articles_attributed",
        "rate": "IMFA_rate",
        "source": "annual institution-year metrics",
    },
    {
        "key": "CIS",
        "label": "CIS",
        "description": "Articles with cross-institution signers",
        "numerator": "N_articles_CIS",
        "denominator": "N_articles_attributed",
        "rate": "CIS_rate",
        "source": "annual institution-year metrics",
    },
    {
        "key": "HPA",
        "label": "HPA",
        "description": "Articles with at least one hyperprolific author",
        "numerator": "N_articles_with_hyperprolific_author",
        "denominator": "N_articles_attributed",
        "rate": "HPA_rate",
        "source": "annual institution-year metrics",
    },
    {
        "key": "HAO",
        "label": "HAO",
        "description": "Hyperauthored articles",
        "numerator": "N_articles_hyperauthored",
        "denominator": "N_articles_attributed",
        "rate": "HAO_rate",
        "source": "annual institution-year metrics",
    },
    {
        "key": "NLO",
        "label": "NLO",
        "description": "Articles not led by the focal institution",
        "numerator": "N_articles_non_led",
        "denominator": "CA_known",
        "rate": "NLO_rate",
        "source": "annual institution-year metrics",
    },
    {
        "key": "HPA_MFA_CIS",
        "label": "HPA-MFA/CIS",
        "description": (
            "Share of institutional output attributable to hyperprolific authors "
            "with simultaneous or cross-institutional affiliations"
        ),
        "numerator": "N_articles_by_HPA_MFA_or_CIS",
        "denominator": "N_articles_attributed",
        "rate": "HPA_MFA_or_CIS_article_rate",
        "source": "derived annual HPA flags and author-productivity tables",
    },
)

def clean_column_names(columns: Iterable[object]) -> list[str]:
    return [str(column).replace("\ufeff", "").strip() for column in columns]


def invalid_text(value: object) -> bool:
    return str(value).strip().lower() in INVALID_TEXT


def read_csv(path: Path, encoding: str = "utf-8-sig") -> pd.DataFrame:
    frame = pd.read_csv(path, encoding=encoding, low_memory=False)
    frame.columns = clean_column_names(frame.columns)
    return frame




def require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def coerce_numeric(
    frame: pd.DataFrame, columns: Sequence[str], source: str
) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        original = frame[column].copy()
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        bad = original.notna() & frame[column].isna()
        if bad.any():
            rows = (frame.index[bad] + 2).tolist()
            raise ValueError(
                f"{source}: nonnumeric values in {column} at data rows {rows}"
            )
    return frame


def validate_mapping(
    path: Path, encoding: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = read_csv(path, encoding=encoding)
    required = {"institution_id", "name", "id-paper", "color_hex"}
    require_columns(mapping, required, path.name)
    mapping = mapping[list(required)].copy()
    for column in ["institution_id", "name", "id-paper", "color_hex"]:
        mapping[column] = mapping[column].astype(str).str.strip()
        if mapping[column].map(invalid_text).any():
            rows = (mapping.index[mapping[column].map(invalid_text)] + 2).tolist()
            raise ValueError(
                f"{path.name}: invalid or blank {column} at data rows {rows}"
            )
    if mapping["institution_id"].duplicated().any():
        duplicate_ids = sorted(
            mapping.loc[
                mapping["institution_id"].duplicated(keep=False), "institution_id"
            ].unique()
        )
        raise ValueError(
            f"{path.name}: institution_id must be unique; duplicates: {duplicate_ids}"
        )
    if mapping["id-paper"].duplicated().any():
        duplicate_labels = sorted(
            mapping.loc[mapping["id-paper"].duplicated(keep=False), "id-paper"].unique()
        )
        raise ValueError(
            f"{path.name}: id-paper must be unique; duplicates: {duplicate_labels}"
        )
    bad_colors = ~mapping["color_hex"].str.match(COLOR_RE)
    if bad_colors.any():
        bad = mapping.loc[bad_colors, ["institution_id", "color_hex"]]
        raise ValueError(
            f"{path.name}: color_hex must use #RRGGBB:\n{bad.to_string(index=False)}"
        )
    mapping["color_hex"] = mapping["color_hex"].str.upper()
    return mapping.sort_values("id-paper").reset_index(drop=True), pd.DataFrame()


def required_input_paths(
    input_dir: Path, start_year: int, end_year: int
) -> tuple[list[Path], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    required: list[Path] = []
    suffixes = (
        "institution_year_metrics.csv",
        "hyperprolific_author_flags.csv",
        "author_productivity.csv",
    )
    for year in range(start_year, end_year + 1):
        for suffix in suffixes:
            path = input_dir / f"{year}_{suffix}"
            required.append(path)
            rows.append(
                {
                    "analysis_year": year,
                    "file_type": suffix[:-4] if suffix.endswith(".csv") else suffix,
                    "path": str(path),
                    "exists": path.is_file(),
                    "size_bytes": path.stat().st_size if path.is_file() else np.nan,
                }
            )
    audit = pd.DataFrame(rows)
    missing = audit.loc[~audit["exists"], "path"].tolist()
    if missing:
        raise FileNotFoundError("Missing required annual files:\n" + "\n".join(missing))
    return required, audit


def metric_required_columns() -> set[str]:
    required = {
        "institution_id",
        "name",
        "year",
        "N_articles_attributed",
        "N_hyperprolific_authors",
        "N_hyperprolific_authors_MFA_or_CIS",
    }
    for metric in METRICS:
        if metric["key"] == "HPA_MFA_CIS":
            continue
        required.update(
            {metric["numerator"], metric["denominator"], metric["rate"]}
        )
    return required


def validate_rate(
    frame: pd.DataFrame,
    numerator: str,
    denominator: str,
    rate: str,
    source: str,
    tolerance: float = 1e-9,
) -> None:
    n = frame[numerator]
    d = frame[denominator]
    r = frame[rate]
    invalid_counts = n.isna() | d.isna() | n.lt(0) | d.lt(0) | n.gt(d)
    if invalid_counts.any():
        rows = (frame.index[invalid_counts] + 2).tolist()
        raise ValueError(
            f"{source}: invalid numerator/denominator for {rate} at rows {rows}"
        )
    invalid_rates = r.notna() & ~r.between(0, 1)
    if invalid_rates.any():
        rows = (frame.index[invalid_rates] + 2).tolist()
        raise ValueError(f"{source}: {rate} outside [0, 1] at rows {rows}")
    expected = n.div(d.replace(0, np.nan))
    mismatch = r.notna() & expected.notna() & r.sub(expected).abs().gt(tolerance)
    missing_when_defined = r.isna() & expected.notna()
    if mismatch.any() or missing_when_defined.any():
        rows = (
            frame.index[mismatch | missing_when_defined].to_series().add(2).tolist()
        )
        raise ValueError(
            f"{source}: {rate} is inconsistent with {numerator}/{denominator} "
            f"at rows {rows}"
        )


def load_annual_metrics(
    input_dir: Path, start_year: int, end_year: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    off_year: list[pd.DataFrame] = []
    validations: list[dict[str, object]] = []
    national_rows: list[pd.DataFrame] = []
    required = metric_required_columns()
    numeric = sorted(
        {
            "year",
            "N_articles_attributed",
            "N_hyperprolific_authors",
            "N_hyperprolific_authors_MFA_or_CIS",
            *[
                column
                for metric in METRICS
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
        path = input_dir / f"{analysis_year}_institution_year_metrics.csv"
        frame = read_csv(path)
        require_columns(frame, required, path.name)
        frame = coerce_numeric(frame, numeric, path.name)
        frame["institution_id"] = frame["institution_id"].astype(str).str.strip()
        frame["name"] = frame["name"].astype(str).str.strip()
        if frame["institution_id"].map(invalid_text).any():
            raise ValueError(f"{path.name}: invalid institution_id")
        if frame["name"].map(invalid_text).any():
            raise ValueError(f"{path.name}: invalid institution name")
        if frame["institution_id"].duplicated().any():
            duplicate_ids = sorted(
                frame.loc[
                    frame["institution_id"].duplicated(keep=False), "institution_id"
                ].unique()
            )
            raise ValueError(
                f"{path.name}: duplicate institution rows: {duplicate_ids}"
            )
        frame["reported_year"] = frame["year"]
        frame["analysis_year"] = analysis_year
        frame["is_off_year_table_row"] = frame["reported_year"].ne(analysis_year)
        if frame["is_off_year_table_row"].any():
            audit = frame.loc[
                frame["is_off_year_table_row"],
                ["institution_id", "name", "reported_year", "analysis_year"],
            ].copy()
            audit["source_file"] = path.name
            audit["source_type"] = "institution_year_metrics"
            off_year.append(audit)
        for metric in METRICS:
            if metric["key"] == "HPA_MFA_CIS":
                continue
            validate_rate(
                frame,
                metric["numerator"],
                metric["denominator"],
                metric["rate"],
                path.name,
            )
        national = frame.loc[frame["institution_id"].eq("NATIONAL")].copy()
        if len(national) > 1:
            raise ValueError(f"{path.name}: more than one NATIONAL row")
        if len(national):
            national["source_file"] = path.name
            national_rows.append(national)
        frame = frame.loc[~frame["institution_id"].eq("NATIONAL")].copy()
        validations.append(
            {
                "analysis_year": analysis_year,
                "source_file": path.name,
                "focal_institution_rows": len(frame),
                "unique_focal_institutions": frame["institution_id"].nunique(),
                "national_rows_excluded": len(national),
                "off_year_rows": int(frame["is_off_year_table_row"].sum()),
                "minimum_output": int(frame["N_articles_attributed"].min()),
                "maximum_output": int(frame["N_articles_attributed"].max()),
                "rate_checks_passed": True,
            }
        )
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True, sort=False)
    if panel.duplicated(["institution_id", "analysis_year"]).any():
        bad = panel.loc[
            panel.duplicated(["institution_id", "analysis_year"], keep=False),
            ["institution_id", "analysis_year"],
        ]
        raise ValueError(
            "Duplicate institution-analysis-year rows:\n" + bad.to_string(index=False)
        )
    off_year_frame = (
        pd.concat(off_year, ignore_index=True)
        if off_year
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
    national_frame = (
        pd.concat(national_rows, ignore_index=True, sort=False)
        if national_rows
        else pd.DataFrame(columns=list(panel.columns) + ["source_file"])
    )
    return panel, off_year_frame, pd.DataFrame(validations), national_frame


def split_eids(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {
        eid.strip()
        for eid in str(value).split(";")
        if eid.strip() and eid.strip().lower() not in INVALID_TEXT
    }


def derive_hpa_dependency(
    input_dir: Path,
    annual: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_flags = {
        "institution_id",
        "year",
        "author_id",
        "is_hyperprolific_author",
        "is_multi_affiliated_or_cross_institution",
    }
    required_productivity = {
        "institution_id",
        "year",
        "author_id",
        "author_article_eids",
    }
    derived_rows: list[dict[str, object]] = []
    off_year: list[pd.DataFrame] = []
    validation_rows: list[dict[str, object]] = []
    for analysis_year in range(start_year, end_year + 1):
        annual_year = annual.loc[annual["analysis_year"].eq(analysis_year)].copy()
        expected_ids = set(annual_year["institution_id"])
        flags_path = input_dir / f"{analysis_year}_hyperprolific_author_flags.csv"
        productivity_path = input_dir / f"{analysis_year}_author_productivity.csv"
        flags = read_csv(flags_path)
        productivity = read_csv(productivity_path)
        require_columns(flags, required_flags, flags_path.name)
        require_columns(productivity, required_productivity, productivity_path.name)
        flags = coerce_numeric(
            flags,
            [
                "year",
                "is_hyperprolific_author",
                "is_multi_affiliated_or_cross_institution",
            ],
            flags_path.name,
        )
        productivity = coerce_numeric(
            productivity, ["year"], productivity_path.name
        )
        for frame in (flags, productivity):
            frame["institution_id"] = frame["institution_id"].astype(str).str.strip()
            frame["author_id"] = frame["author_id"].astype(str).str.strip()
        flags["reported_year"] = flags["year"]
        productivity["reported_year"] = productivity["year"]
        flags["analysis_year"] = analysis_year
        productivity["analysis_year"] = analysis_year
        for frame, source_path, source_type in (
            (flags, flags_path, "hyperprolific_author_flags"),
            (productivity, productivity_path, "author_productivity"),
        ):
            mask = frame["reported_year"].ne(analysis_year)
            if mask.any():
                audit = frame.loc[
                    mask,
                    [
                        "institution_id",
                        "author_id",
                        "reported_year",
                        "analysis_year",
                    ],
                ].copy()
                audit["source_file"] = source_path.name
                audit["source_type"] = source_type
                off_year.append(audit)
        flags = flags.loc[flags["institution_id"].isin(expected_ids)].copy()
        productivity = productivity.loc[
            productivity["institution_id"].isin(expected_ids)
        ].copy()
        if flags.duplicated(["institution_id", "author_id"]).any():
            bad = flags.loc[
                flags.duplicated(["institution_id", "author_id"], keep=False),
                ["institution_id", "author_id"],
            ]
            raise ValueError(
                f"{flags_path.name}: duplicate institution-author flags:\n"
                + bad.to_string(index=False)
            )
        if productivity.duplicated(["institution_id", "author_id"]).any():
            bad = productivity.loc[
                productivity.duplicated(
                    ["institution_id", "author_id"], keep=False
                ),
                ["institution_id", "author_id"],
            ]
            raise ValueError(
                f"{productivity_path.name}: duplicate institution-author rows:\n"
                + bad.to_string(index=False)
            )
        invalid_binary = ~flags[
            ["is_hyperprolific_author", "is_multi_affiliated_or_cross_institution"]
        ].isin([0, 1])
        if invalid_binary.any().any():
            raise ValueError(f"{flags_path.name}: nonbinary HPA/MFA-or-CIS flags")
        selected = flags.loc[
            flags["is_hyperprolific_author"].eq(1)
            & flags["is_multi_affiliated_or_cross_institution"].eq(1),
            ["institution_id", "author_id"],
        ].copy()
        joined = selected.merge(
            productivity[
                ["institution_id", "author_id", "author_article_eids"]
            ],
            on=["institution_id", "author_id"],
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        if not joined["_merge"].eq("both").all():
            bad = joined.loc[
                joined["_merge"].ne("both"), ["institution_id", "author_id"]
            ]
            raise ValueError(
                f"{analysis_year}: selected HPA-MFA/CIS authors missing from "
                f"author productivity:\n{bad.to_string(index=False)}"
            )
        selected_author_counts = selected.groupby("institution_id").size().to_dict()
        article_unions: dict[str, set[str]] = {institution_id: set() for institution_id in expected_ids}
        for row in joined.itertuples(index=False):
            article_unions[row.institution_id].update(split_eids(row.author_article_eids))
        for row in annual_year.itertuples(index=False):
            institution_id = row.institution_id
            numerator = len(article_unions[institution_id])
            denominator = int(row.N_articles_attributed)
            if numerator < 0 or numerator > denominator:
                raise ValueError(
                    f"{analysis_year} {institution_id}: derived dependency numerator "
                    f"{numerator} exceeds attributed output {denominator}"
                )
            selected_count = int(selected_author_counts.get(institution_id, 0))
            expected_hpa = int(row.N_hyperprolific_authors)
            expected_selected = int(row.N_hyperprolific_authors_MFA_or_CIS)
            flag_hpa_count = int(
                flags.loc[
                    flags["institution_id"].eq(institution_id),
                    "is_hyperprolific_author",
                ].sum()
            )
            if flag_hpa_count != expected_hpa:
                raise ValueError(
                    f"{analysis_year} {institution_id}: HPA flag count "
                    f"{flag_hpa_count} does not match annual metric {expected_hpa}"
                )
            if selected_count != expected_selected:
                raise ValueError(
                    f"{analysis_year} {institution_id}: selected HPA-MFA/CIS count "
                    f"{selected_count} does not match annual metric "
                    f"{expected_selected}"
                )
            derived_rows.append(
                {
                    "analysis_year": analysis_year,
                    "institution_id": institution_id,
                    "N_articles_attributed": denominator,
                    "N_hyperprolific_authors": expected_hpa,
                    "N_hyperprolific_authors_MFA_or_CIS": expected_selected,
                    "N_selected_HPA_MFA_or_CIS_authors": selected_count,
                    "N_articles_by_HPA_MFA_or_CIS": numerator,
                    "HPA_MFA_or_CIS_article_rate": (
                        numerator / denominator if denominator > 0 else np.nan
                    ),
                    "derivation_method": (
                        "uniform_union_from_hpa_flags_and_author_productivity"
                    ),
                    "source_hpa_flags_file": flags_path.name,
                    "source_author_productivity_file": productivity_path.name,
                    "source_metrics_file": (
                        f"{analysis_year}_institution_year_metrics.csv"
                    ),
                    "script_version": SCRIPT_VERSION,
                }
            )
        validation_rows.append(
            {
                "analysis_year": analysis_year,
                "institution_rows": len(annual_year),
                "HPA_flag_rows": len(flags),
                "selected_HPA_MFA_or_CIS_authors": len(selected),
                "selected_authors_joined_to_productivity": len(joined),
                "derived_institution_rows": len(annual_year),
                "author_counts_match_annual_metrics": True,
                "article_union_bounds_valid": True,
            }
        )
    derived = pd.DataFrame(derived_rows)
    expected_pairs = annual[["institution_id", "analysis_year"]].drop_duplicates()
    observed_pairs = derived[["institution_id", "analysis_year"]].drop_duplicates()
    coverage = expected_pairs.merge(
        observed_pairs,
        on=["institution_id", "analysis_year"],
        how="outer",
        indicator=True,
    )
    if not coverage["_merge"].eq("both").all():
        raise ValueError(
            "Derived dependency metric lacks exact panel coverage:\n"
            + coverage.loc[coverage["_merge"].ne("both")].to_string(index=False)
        )
    off_year_frame = (
        pd.concat(off_year, ignore_index=True, sort=False)
        if off_year
        else pd.DataFrame(
            columns=[
                "institution_id",
                "author_id",
                "reported_year",
                "analysis_year",
                "source_file",
                "source_type",
            ]
        )
    )
    return derived, off_year_frame, pd.DataFrame(validation_rows)


def attach_mapping(
    annual: pd.DataFrame, mapping: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_ids = sorted(set(annual["institution_id"]) - set(mapping["institution_id"]))
    if missing_ids:
        raise ValueError(
            "Mapping file does not cover all analysed institutions: "
            + ", ".join(missing_ids)
        )
    unused_ids = sorted(set(mapping["institution_id"]) - set(annual["institution_id"]))
    name_reference = (
        annual.sort_values("analysis_year")
        .groupby("institution_id", as_index=False)
        .agg(data_name=("name", "last"))
    )
    comparison = mapping.merge(name_reference, on="institution_id", how="left")
    comparison["mapping_name_normalized"] = (
        comparison["name"].str.replace(r"\s+", " ", regex=True).str.strip()
    )
    comparison["data_name_normalized"] = (
        comparison["data_name"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    )
    mismatches = comparison.loc[
        comparison["data_name"].notna()
        & comparison["mapping_name_normalized"].ne(
            comparison["data_name_normalized"]
        ),
        ["institution_id", "id-paper", "name", "data_name"],
    ].rename(columns={"name": "mapping_name"})
    merged = annual.merge(
        mapping[["institution_id", "id-paper", "color_hex"]],
        on="institution_id",
        how="left",
        validate="many_to_one",
    )
    merged["mapping_unused_in_analysis"] = merged["institution_id"].isin(unused_ids)
    return merged, mismatches.reset_index(drop=True)


