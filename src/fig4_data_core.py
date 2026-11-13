#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared data utilities for Section 4.4.

The module is intentionally compatible with Python 3.8.  The analysis year is
always taken from the annual filename, never reassigned from the ``year`` or
``cover_date`` columns inside a file.
"""

from __future__ import absolute_import, division, print_function

import output_utils

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_MAPPING_COLUMNS = (
    "institution_id",
    "name",
    "id-paper",
    "color_hex",
)

REQUIRED_SUFFIXES = (
    "institution_year_metrics.csv",
    "top_author_concentration.csv",
    "author_productivity.csv",
    "hyperprolific_author_flags.csv",
)

METRIC_COLUMNS = (
    "N_articles_attributed",
    "N_authors_institution",
    "N_hyperprolific_authors",
    "N_articles_with_hyperprolific_author",
    "HPA_rate",
    "top_1_authors_article_count",
    "top_1_authors_article_share",
    "top_5_authors_article_count",
    "top_5_authors_article_share",
    "top_10_authors_article_count",
    "top_10_authors_article_share",
    "N_top_authors_needed_50_percent_output_union",
)


def read_csv(path):
    """Read CSV values as text while tolerating an UTF-8 BOM."""
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )


def require_columns(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError("{} is missing columns: {}".format(label, ", ".join(missing)))


def numeric(frame, columns, label, integer=False):
    """Convert selected columns to numeric values with explicit errors."""
    for column in columns:
        if column not in frame.columns:
            raise ValueError("{} is missing numeric column {}".format(label, column))
        converted = pd.to_numeric(frame[column], errors="coerce")
        bad = frame[column].astype(str).str.strip().ne("") & converted.isna()
        if bad.any():
            examples = frame.loc[bad, column].astype(str).head(5).tolist()
            raise ValueError(
                "{} contains non-numeric values in {}: {}".format(
                    label, column, examples
                )
            )
        if integer:
            converted = converted.fillna(0).astype(int)
        frame[column] = converted
    return frame


def clean_identifier(series):
    """Normalize identifiers without using Python 3.9-only string methods."""
    values = series.astype(str).str.strip()
    values = values.str.replace(r"\.0$", "", regex=True)
    return values


def load_mapping(path, encoding="utf-8-sig"):
    mapping = pd.read_csv(
        path,
        encoding=encoding,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    require_columns(mapping, REQUIRED_MAPPING_COLUMNS, Path(path).name)
    mapping = mapping[list(REQUIRED_MAPPING_COLUMNS)].copy()
    for column in REQUIRED_MAPPING_COLUMNS:
        mapping[column] = mapping[column].astype(str).str.strip()
    if mapping["institution_id"].eq("").any():
        raise ValueError("Mapping contains an empty institution_id")
    if mapping["institution_id"].duplicated().any():
        duplicated = mapping.loc[
            mapping["institution_id"].duplicated(False), "institution_id"
        ].tolist()
        raise ValueError("Duplicated institution_id values in mapping: {}".format(duplicated))
    invalid_codes = ~mapping["id-paper"].str.match(r"^(U-[0-9]{2}|National)$")
    if invalid_codes.any():
        raise ValueError(
            "Invalid manuscript codes in mapping: {}".format(
                mapping.loc[invalid_codes, "id-paper"].tolist()
            )
        )
    invalid_colors = ~mapping["color_hex"].str.match(r"^#[0-9A-Fa-f]{6}$")
    if invalid_colors.any():
        raise ValueError(
            "Invalid color_hex values in mapping: {}".format(
                mapping.loc[invalid_colors, "color_hex"].tolist()
            )
        )
    if mapping["color_hex"].str.lower().isin({"nan", "none", "null", ""}).any():
        raise ValueError("Mapping contains an empty/null color_hex")
    return mapping.sort_values("id-paper").reset_index(drop=True)


def required_paths(input_dir, start_year, end_year):
    input_dir = Path(input_dir)
    rows = []
    missing = []
    for year in range(start_year, end_year + 1):
        for suffix in REQUIRED_SUFFIXES:
            path = input_dir / "{}_{}".format(year, suffix)
            rows.append(
                {
                    "analysis_year": year,
                    "file_type": suffix[:-4] if suffix.endswith(".csv") else suffix,
                    "file_name": path.name,
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else np.nan,
                }
            )
            if not path.exists():
                missing.append(str(path))
    audit = pd.DataFrame(rows)
    if missing:
        raise FileNotFoundError("Missing required annual inputs:\n" + "\n".join(missing))
    return audit


def _audit_year(frame, analysis_year, file_name, file_type):
    if "year" not in frame.columns:
        return pd.DataFrame(
            [
                {
                    "analysis_year": analysis_year,
                    "file_name": file_name,
                    "file_type": file_type,
                    "row_number": np.nan,
                    "embedded_year": np.nan,
                    "status": "year_column_absent",
                }
            ]
        )
    embedded = pd.to_numeric(frame["year"], errors="coerce")
    off = embedded.ne(analysis_year) | embedded.isna()
    if not off.any():
        return pd.DataFrame(
            [
                {
                    "analysis_year": analysis_year,
                    "file_name": file_name,
                    "file_type": file_type,
                    "row_number": np.nan,
                    "embedded_year": analysis_year,
                    "status": "all_rows_match_filename_year",
                }
            ]
        )
    result = frame.loc[off, ["year"]].copy()
    result["row_number"] = result.index + 2
    result["embedded_year"] = embedded.loc[off]
    result["analysis_year"] = analysis_year
    result["file_name"] = file_name
    result["file_type"] = file_type
    result["status"] = "off_year_row_audited_not_reassigned"
    return result[
        [
            "analysis_year",
            "file_name",
            "file_type",
            "row_number",
            "embedded_year",
            "status",
        ]
    ]


def parse_eids(value):
    if value is None:
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {token.strip() for token in text.split(";") if token.strip()}


def load_annual_data(input_dir, mapping, start_year, end_year):
    """Load and validate the four annual table families."""
    input_dir = Path(input_dir)
    metric_frames = []
    productivity_frames = []
    hyper_frames = []
    top_frames = []
    year_audits = []

    for analysis_year in range(start_year, end_year + 1):
        metric_path = input_dir / "{}_institution_year_metrics.csv".format(analysis_year)
        metric = read_csv(metric_path)
        require_columns(
            metric,
            {"institution_id", "name", "year"}.union(METRIC_COLUMNS),
            metric_path.name,
        )
        year_audits.append(
            _audit_year(metric, analysis_year, metric_path.name, "institution_year_metrics")
        )
        metric["institution_id"] = metric["institution_id"].astype(str).str.strip()
        metric = metric.loc[metric["institution_id"].ne("NATIONAL")].copy()
        metric = numeric(metric, METRIC_COLUMNS, metric_path.name)
        if metric["institution_id"].duplicated().any():
            raise ValueError("{} contains duplicated institutions".format(metric_path.name))
        metric["analysis_year"] = analysis_year
        metric_frames.append(metric)

        productivity_path = input_dir / "{}_author_productivity.csv".format(analysis_year)
        productivity = read_csv(productivity_path)
        require_columns(
            productivity,
            {
                "institution_id",
                "year",
                "author_rank",
                "author_id",
                "indexed_name",
                "p_author_institution_year",
                "author_article_eids",
            },
            productivity_path.name,
        )
        year_audits.append(
            _audit_year(productivity, analysis_year, productivity_path.name, "author_productivity")
        )
        productivity["institution_id"] = productivity["institution_id"].astype(str).str.strip()
        productivity = productivity.loc[
            productivity["institution_id"].ne("NATIONAL")
        ].copy()
        productivity["author_id"] = clean_identifier(productivity["author_id"])
        productivity = numeric(
            productivity,
            ["author_rank", "p_author_institution_year"],
            productivity_path.name,
            integer=True,
        )
        productivity["analysis_year"] = analysis_year
        productivity["article_eid_set"] = productivity["author_article_eids"].map(parse_eids)
        duplicates = productivity.duplicated(
            ["institution_id", "analysis_year", "author_id"], keep=False
        )
        if duplicates.any():
            raise ValueError("{} contains duplicated institution-author rows".format(productivity_path.name))
        productivity_frames.append(productivity)

        hyper_path = input_dir / "{}_hyperprolific_author_flags.csv".format(analysis_year)
        hyper = read_csv(hyper_path)
        require_columns(
            hyper,
            {
                "institution_id",
                "year",
                "author_id",
                "p_author_institution_year",
                "is_hyperprolific_author",
                "is_multi_affiliated_or_cross_institution",
            },
            hyper_path.name,
        )
        year_audits.append(
            _audit_year(hyper, analysis_year, hyper_path.name, "hyperprolific_author_flags")
        )
        hyper["institution_id"] = hyper["institution_id"].astype(str).str.strip()
        hyper = hyper.loc[hyper["institution_id"].ne("NATIONAL")].copy()
        hyper["author_id"] = clean_identifier(hyper["author_id"])
        hyper = numeric(
            hyper,
            [
                "p_author_institution_year",
                "is_hyperprolific_author",
                "is_multi_affiliated_or_cross_institution",
            ],
            hyper_path.name,
            integer=True,
        )
        hyper["analysis_year"] = analysis_year
        hyper_frames.append(hyper)

        top_path = input_dir / "{}_top_author_concentration.csv".format(analysis_year)
        top = read_csv(top_path)
        require_columns(
            top,
            {
                "institution_id",
                "year",
                "n_top_authors",
                "N_articles_attributed",
                "n_articles_by_top_authors_union",
                "top_authors_article_share",
            },
            top_path.name,
        )
        year_audits.append(
            _audit_year(top, analysis_year, top_path.name, "top_author_concentration")
        )
        top["institution_id"] = top["institution_id"].astype(str).str.strip()
        top = top.loc[top["institution_id"].ne("NATIONAL")].copy()
        top = numeric(
            top,
            [
                "n_top_authors",
                "N_articles_attributed",
                "n_articles_by_top_authors_union",
                "top_authors_article_share",
            ],
            top_path.name,
        )
        top["n_top_authors"] = top["n_top_authors"].astype(int)
        top["analysis_year"] = analysis_year
        top_frames.append(top)

    metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    productivity = pd.concat(productivity_frames, ignore_index=True, sort=False)
    hyper = pd.concat(hyper_frames, ignore_index=True, sort=False)
    top = pd.concat(top_frames, ignore_index=True, sort=False)
    year_audit = pd.concat(year_audits, ignore_index=True, sort=False)

    mapping_fields = mapping[["institution_id", "name", "id-paper", "color_hex"]].copy()
    data_ids = set(metrics["institution_id"])
    mapping_ids = set(mapping_fields["institution_id"])
    missing_mapping = sorted(data_ids - mapping_ids)
    if missing_mapping:
        raise ValueError("Institutions missing from mapping: {}".format(missing_mapping))

    metrics = metrics.drop(columns=["name"], errors="ignore").merge(
        mapping_fields,
        on="institution_id",
        how="left",
        validate="many_to_one",
    )
    for frame in (productivity, hyper, top):
        unknown = sorted(set(frame["institution_id"]) - mapping_ids)
        if unknown:
            raise ValueError("Unmapped institution IDs in annual inputs: {}".format(unknown))

    return metrics, productivity, hyper, top, year_audit


def validate_concentration(metrics, productivity, top, tolerance=1e-10):
    """Recompute CTA unions from author-level EIDs and compare all three sources."""
    rows = []
    metric_lookup = metrics.set_index(["institution_id", "analysis_year"])
    top_lookup = top.set_index(["institution_id", "analysis_year", "n_top_authors"])
    for (institution_id, analysis_year), group in productivity.groupby(
        ["institution_id", "analysis_year"], sort=True
    ):
        group = group.sort_values(["author_rank", "author_id"])
        key = (institution_id, analysis_year)
        if key not in metric_lookup.index:
            raise ValueError("Author productivity row lacks matching metric row: {}".format(key))
        metric_row = metric_lookup.loc[key]
        denominator = int(metric_row["N_articles_attributed"])
        cumulative = set()
        union_by_n = {}
        for _, author in group.iterrows():
            cumulative.update(author["article_eid_set"])
            union_by_n[int(author["author_rank"])] = len(cumulative)
        for n in (1, 5, 10):
            metric_count = int(metric_row["top_{}_authors_article_count".format(n)])
            metric_share = float(metric_row["top_{}_authors_article_share".format(n)])
            recomputed_count = union_by_n.get(n, len(cumulative))
            top_key = (institution_id, analysis_year, n)
            if top_key not in top_lookup.index:
                raise ValueError("Missing top-author concentration row: {}".format(top_key))
            top_row = top_lookup.loc[top_key]
            top_count = int(top_row["n_articles_by_top_authors_union"])
            top_share = float(top_row["top_authors_article_share"])
            recomputed_share = recomputed_count / denominator if denominator else np.nan
            valid = (
                metric_count == recomputed_count == top_count
                and abs(metric_share - recomputed_share) <= tolerance
                and abs(top_share - recomputed_share) <= tolerance
            )
            rows.append(
                {
                    "institution_id": institution_id,
                    "analysis_year": analysis_year,
                    "n_top_authors": n,
                    "N_articles_attributed": denominator,
                    "metric_count": metric_count,
                    "top_table_count": top_count,
                    "recomputed_union_count": recomputed_count,
                    "metric_share": metric_share,
                    "top_table_share": top_share,
                    "recomputed_share": recomputed_share,
                    "validation_pass": valid,
                }
            )
    audit = pd.DataFrame(rows)
    if not audit["validation_pass"].all():
        raise ValueError("CTA validation failed; inspect concentration_validation.csv")
    return audit


def derive_hpa_coverage(metrics, productivity, hyper, threshold=25, tolerance=1e-10):
    """Derive HPA and exposed-HPA article unions from author-level records."""
    flags = hyper[
        [
            "institution_id",
            "analysis_year",
            "author_id",
            "is_hyperprolific_author",
            "is_multi_affiliated_or_cross_institution",
        ]
    ].drop_duplicates()
    authors = productivity.merge(
        flags,
        on=["institution_id", "analysis_year", "author_id"],
        how="left",
        validate="one_to_one",
    )
    authors["is_hyperprolific_author"] = (
        authors["is_hyperprolific_author"].replace("", np.nan).fillna(0).astype(int)
    )
    authors["is_multi_affiliated_or_cross_institution"] = (
        authors["is_multi_affiliated_or_cross_institution"]
        .replace("", np.nan)
        .fillna(0)
        .astype(int)
    )
    authors["is_HPA_from_productivity"] = (
        authors["p_author_institution_year"].ge(threshold).astype(int)
    )
    mismatch = authors["is_HPA_from_productivity"].ne(authors["is_hyperprolific_author"])
    if mismatch.any():
        raise ValueError("HPA flags disagree with the productivity threshold")

    metric_lookup = metrics.set_index(["institution_id", "analysis_year"])
    rows = []
    for (institution_id, analysis_year), group in authors.groupby(
        ["institution_id", "analysis_year"], sort=True
    ):
        metric_row = metric_lookup.loc[(institution_id, analysis_year)]
        denominator = int(metric_row["N_articles_attributed"])
        hpa = group.loc[group["is_HPA_from_productivity"].eq(1)]
        exposed_hpa = hpa.loc[
            hpa["is_multi_affiliated_or_cross_institution"].eq(1)
        ]
        hpa_union = set()
        exposed_union = set()
        for eid_set in hpa["article_eid_set"]:
            hpa_union.update(eid_set)
        for eid_set in exposed_hpa["article_eid_set"]:
            exposed_union.update(eid_set)
        metric_count = int(metric_row["N_articles_with_hyperprolific_author"])
        metric_rate = float(metric_row["HPA_rate"])
        valid = (
            metric_count == len(hpa_union)
            and abs(metric_rate - (len(hpa_union) / denominator if denominator else 0.0))
            <= tolerance
        )
        rows.append(
            {
                "institution_id": institution_id,
                "analysis_year": analysis_year,
                "N_articles_attributed": denominator,
                "N_HPA_authors": len(hpa),
                "N_exposed_HPA_authors": len(exposed_hpa),
                "N_articles_HPA_union_derived": len(hpa_union),
                "HPA_article_share_derived": len(hpa_union) / denominator if denominator else np.nan,
                "N_articles_exposed_HPA_union_derived": len(exposed_union),
                "exposed_HPA_article_share_derived": (
                    len(exposed_union) / denominator if denominator else np.nan
                ),
                "metric_HPA_article_count": metric_count,
                "metric_HPA_rate": metric_rate,
                "validation_pass": valid,
            }
        )
    coverage = pd.DataFrame(rows)
    if not coverage["validation_pass"].all():
        raise ValueError("HPA coverage validation failed")
    return coverage, authors


def add_analysis_fields(metrics, hpa_coverage, expansion_end_year):
    data = metrics.merge(
        hpa_coverage.drop(
            columns=[
                "N_articles_attributed",
                "metric_HPA_article_count",
                "metric_HPA_rate",
                "validation_pass",
            ]
        ),
        on=["institution_id", "analysis_year"],
        how="left",
        validate="one_to_one",
    )
    data = data.rename(
        columns={
            "N_articles_attributed": "output",
            "N_authors_institution": "institution_authors",
            "top_1_authors_article_share": "CTA_1",
            "top_5_authors_article_share": "CTA_5",
            "top_10_authors_article_share": "CTA_10",
            "N_top_authors_needed_50_percent_output_union": "A_top50",
        }
    )
    data["phase"] = np.where(
        data["analysis_year"].le(expansion_end_year),
        "Expansion (2011-2018)",
        "Stabilization (2019-2025)",
    )
    return data


def save_csv(frame, path):
    if not output_utils.keep(path):
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
