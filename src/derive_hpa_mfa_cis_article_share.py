#!/usr/bin/env python3
"""Derive the annual share of articles produced by HPA authors with MFA or CIS.

The calculation uses three annual output tables located directly in the input
directory:

* YYYY_hyperprolific_author_flags.csv identifies hyperprolific authors (HPA)
  and whether each author has multi-affiliation in the same publication (MFA)
  or is a cross-institution signer (CIS).
* YYYY_author_productivity.csv supplies the exact article EIDs attributed to
  every author within each institution and at NATIONAL level.
* YYYY_institution_year_metrics.csv supplies the article denominator and the
  published HPA author counts used for exact reconciliation.

For each institution-year, the numerator is the union of article EIDs for HPA
authors whose ``is_multi_affiliated_or_cross_institution`` flag equals one.
Thus, an article is counted once even when several selected authors sign it.
The denominator is ``N_articles_attributed``.  The same algorithm is applied
without year-specific branches from 2011 through 2025.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


NATIONAL_ID = "NATIONAL"
SCRIPT_VERSION = "2.0-uniform-precomputed-hpa-files-2026-07-16"
DERIVATION_METHOD = "uniform_union_from_hpa_flags_and_author_productivity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive institution-year article shares using annual institution "
            "metrics, HPA flags, and author-productivity files."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Directory containing YYYY_hyperprolific_author_flags.csv, "
            "YYYY_author_productivity.csv, and "
            "YYYY_institution_year_metrics.csv."
        ),
    )
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()
    if args.end_year < args.start_year:
        parser.error("--end-year must be greater than or equal to --start-year.")
    return args


def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        str(column).replace("\ufeff", "").strip() for column in frame.columns
    ]
    return frame


def read_csv(path: Path) -> pd.DataFrame:
    return clean_columns(
        pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    )


def require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")


def numeric(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    for column in columns:
        original = frame[column].copy()
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        bad = original.notna() & frame[column].isna()
        if bad.any():
            raise ValueError(
                f"{path.name}: nonnumeric {column} at rows "
                f"{(frame.index[bad] + 2).tolist()}"
            )


def validate_year_column(frame: pd.DataFrame, year: int, path: Path) -> None:
    numeric(frame, ["year"], path)
    mismatch = frame["year"].notna() & frame["year"].ne(year)
    if mismatch.any():
        raise ValueError(
            f"{path.name}: table year does not match filename year {year}."
        )


def parse_eids(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return tuple()
    return tuple(item.strip() for item in str(value).split(";") if item.strip())


def read_metrics(path: Path, year: int) -> pd.DataFrame:
    required = {
        "institution_id",
        "name",
        "year",
        "N_articles_attributed",
        "N_hyperprolific_authors",
        "N_hyperprolific_authors_MFA_or_CIS",
    }
    frame = read_csv(path)
    require_columns(frame, required, path)
    frame = frame[list(required)].copy()
    frame["institution_id"] = frame["institution_id"].astype(str).str.strip()
    validate_year_column(frame, year, path)
    numeric(
        frame,
        [
            "N_articles_attributed",
            "N_hyperprolific_authors",
            "N_hyperprolific_authors_MFA_or_CIS",
        ],
        path,
    )
    if frame["institution_id"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicated institution IDs.")
    if frame["institution_id"].eq(NATIONAL_ID).sum() != 1:
        raise ValueError(f"{path.name} must contain exactly one NATIONAL row.")
    if (
        frame[
            [
                "N_articles_attributed",
                "N_hyperprolific_authors",
                "N_hyperprolific_authors_MFA_or_CIS",
            ]
        ]
        .lt(0)
        .any()
        .any()
    ):
        raise ValueError(f"{path.name} contains negative counts.")
    return frame


def read_flags(path: Path, year: int) -> pd.DataFrame:
    required = {
        "institution_id",
        "year",
        "author_id",
        "indexed_name",
        "p_author_institution_year",
        "is_hyperprolific_author",
        "is_multi_affiliated_same_publication",
        "is_cross_institution_signer",
        "is_multi_affiliated_or_cross_institution",
    }
    frame = read_csv(path)
    require_columns(frame, required, path)
    frame = frame[list(required)].copy()
    for column in ("institution_id", "author_id"):
        frame[column] = frame[column].astype(str).str.strip()
    validate_year_column(frame, year, path)
    binary_columns = [
        "is_hyperprolific_author",
        "is_multi_affiliated_same_publication",
        "is_cross_institution_signer",
        "is_multi_affiliated_or_cross_institution",
    ]
    numeric(frame, ["p_author_institution_year", *binary_columns], path)
    if frame.duplicated(["institution_id", "author_id"]).any():
        raise ValueError(f"{path.name} contains duplicated institution-author rows.")
    for column in binary_columns:
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"{path.name}: {column} must be binary.")
        frame[column] = frame[column].astype(int)
    expected_hpa = frame["p_author_institution_year"].ge(25).astype(int)
    if not frame["is_hyperprolific_author"].eq(expected_hpa).all():
        raise ValueError(
            f"{path.name}: HPA flag is inconsistent with the >=25 threshold."
        )
    expected_union = frame[
        [
            "is_multi_affiliated_same_publication",
            "is_cross_institution_signer",
        ]
    ].max(axis=1)
    if not frame["is_multi_affiliated_or_cross_institution"].eq(
        expected_union
    ).all():
        raise ValueError(f"{path.name}: MFA-or-CIS flag is inconsistent.")
    return frame


def read_productivity(path: Path, year: int) -> pd.DataFrame:
    required = {
        "institution_id",
        "year",
        "author_id",
        "indexed_name",
        "p_author_institution_year",
        "author_article_eids",
    }
    frame = read_csv(path)
    require_columns(frame, required, path)
    frame = frame[list(required)].copy()
    for column in ("institution_id", "author_id"):
        frame[column] = frame[column].astype(str).str.strip()
    validate_year_column(frame, year, path)
    numeric(frame, ["p_author_institution_year"], path)
    if frame.duplicated(["institution_id", "author_id"]).any():
        raise ValueError(f"{path.name} contains duplicated institution-author rows.")
    frame["article_eids"] = frame["author_article_eids"].map(parse_eids)
    duplicate_eids = frame["article_eids"].map(lambda values: len(values) != len(set(values)))
    if duplicate_eids.any():
        raise ValueError(f"{path.name} contains duplicated EIDs within an author row.")
    eid_count = frame["article_eids"].map(len)
    mismatch = frame["p_author_institution_year"].ne(eid_count)
    if mismatch.any():
        raise ValueError(
            f"{path.name}: p_author_institution_year does not equal the EID count."
        )
    return frame


def derive_year(input_dir: Path, year: int) -> pd.DataFrame:
    metrics_path = input_dir / f"{year}_institution_year_metrics.csv"
    flags_path = input_dir / f"{year}_hyperprolific_author_flags.csv"
    productivity_path = input_dir / f"{year}_author_productivity.csv"
    for path in (metrics_path, flags_path, productivity_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metrics = read_metrics(metrics_path, year)
    flags = read_flags(flags_path, year)
    productivity = read_productivity(productivity_path, year)

    keys = ["institution_id", "author_id"]
    joined = flags.merge(
        productivity[keys + ["p_author_institution_year", "article_eids"]],
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("_flags", "_productivity"),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(
            f"{year}: at least one flagged author is absent from author_productivity."
        )
    p_mismatch = joined["p_author_institution_year_flags"].ne(
        joined["p_author_institution_year_productivity"]
    )
    if p_mismatch.any():
        raise ValueError(
            f"{year}: productivity differs between the flag and productivity files."
        )

    hpa = joined.loc[joined["is_hyperprolific_author"].eq(1)].copy()
    selected = hpa.loc[
        hpa["is_multi_affiliated_or_cross_institution"].eq(1)
    ].copy()
    reconstructed = hpa.groupby("institution_id").size().rename("reconstructed_HPA")
    selected_counts = selected.groupby("institution_id").size().rename(
        "reconstructed_HPA_MFA_or_CIS"
    )
    check = (
        metrics.set_index("institution_id")
        .join(reconstructed, how="left")
        .join(selected_counts, how="left")
        .fillna(
            {
                "reconstructed_HPA": 0,
                "reconstructed_HPA_MFA_or_CIS": 0,
            }
        )
    )
    mismatch = check["N_hyperprolific_authors"].ne(check["reconstructed_HPA"]) | check[
        "N_hyperprolific_authors_MFA_or_CIS"
    ].ne(check["reconstructed_HPA_MFA_or_CIS"])
    if mismatch.any():
        raise ValueError(
            f"{year}: flag counts do not reproduce annual HPA metrics:\n"
            + check.loc[
                mismatch,
                [
                    "N_hyperprolific_authors",
                    "reconstructed_HPA",
                    "N_hyperprolific_authors_MFA_or_CIS",
                    "reconstructed_HPA_MFA_or_CIS",
                ],
            ].to_string()
        )

    article_unions: dict[str, set[str]] = {}
    for row in selected.itertuples(index=False):
        article_unions.setdefault(row.institution_id, set()).update(row.article_eids)
    article_counts = {
        institution_id: len(eids) for institution_id, eids in article_unions.items()
    }

    output = metrics.copy()
    output["analysis_year"] = year
    output["N_selected_HPA_MFA_or_CIS_authors"] = (
        output["institution_id"].map(selected_counts).fillna(0).astype(int)
    )
    output["N_articles_by_HPA_MFA_or_CIS"] = (
        output["institution_id"].map(article_counts).fillna(0).astype(int)
    )
    impossible = output["N_articles_by_HPA_MFA_or_CIS"].gt(
        output["N_articles_attributed"]
    )
    if impossible.any():
        raise ValueError(
            f"{year}: selected-author article union exceeds N_articles_attributed."
        )
    output["HPA_MFA_or_CIS_article_rate"] = output[
        "N_articles_by_HPA_MFA_or_CIS"
    ].div(output["N_articles_attributed"].replace(0, np.nan))
    output["HPA_MFA_or_CIS_article_rate_pct"] = (
        100.0 * output["HPA_MFA_or_CIS_article_rate"]
    )
    output["derivation_method"] = DERIVATION_METHOD
    output["source_hpa_flags_file"] = flags_path.name
    output["source_author_productivity_file"] = productivity_path.name
    output["source_metrics_file"] = metrics_path.name
    output["script_version"] = SCRIPT_VERSION

    columns = [
        "analysis_year",
        "institution_id",
        "name",
        "N_articles_attributed",
        "N_hyperprolific_authors",
        "N_hyperprolific_authors_MFA_or_CIS",
        "N_selected_HPA_MFA_or_CIS_authors",
        "N_articles_by_HPA_MFA_or_CIS",
        "HPA_MFA_or_CIS_article_rate",
        "HPA_MFA_or_CIS_article_rate_pct",
        "derivation_method",
        "source_hpa_flags_file",
        "source_author_productivity_file",
        "source_metrics_file",
        "script_version",
    ]
    return output[columns]


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    frames = [derive_year(input_dir, year) for year in range(args.start_year, args.end_year + 1)]
    output = pd.concat(frames, ignore_index=True).sort_values(['analysis_year', 'institution_id'])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f'Created {output_file}')
    print(f"Rows: {len(output)}; years: {output['analysis_year'].min()}-{output['analysis_year'].max()}")
    print('Uniform method: union of author_article_eids for HPA rows with is_multi_affiliated_or_cross_institution = 1')


if __name__ == "__main__":
    main()
