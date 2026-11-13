#!/usr/bin/env python3
"""Analyse national sharing of focal-university output and generate Figure 5.

The input directory contains the annual CSV files produced by
process_json.py. The analysis distinguishes four mutually
exclusive classifications for articles attributed to a focal institution and
also carrying at least one additional Ecuadorian institution:

* same_author_only: at least one author carries the focal institution and a
  national partner, without a strict distinct-author link;
* distinct_author_only: at least one focal-only author and one partner-only
  author occur for a focal-partner pair, without a same-author link;
* both: both mechanisms occur in the article;
* undetermined: the article-author-affiliation links are incomplete.

Known AF-ID aliases of the 30 focal institutions are consolidated by
institution_id. Other Ecuadorian AF-IDs remain distinct Scopus affiliation
entities. The national denominator is the set of unique articles attributed to
at least one focal institution, not all Ecuadorian output.

"""

from __future__ import print_function

import figure_style
import output_utils

import argparse
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-section-3-5")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.3.1-marker-harmonization-2026-08-25"
YEARS = list(range(2011, 2026))
NATIONAL_ID = "NATIONAL"
FOCAL_PREFIX = "INST:"
NONFOCAL_PREFIX = "AFID:"
MECHANISMS = [
    "distinct_author_only",
    "same_author_only",
    "both",
    "undetermined",
]
MECHANISM_LABELS = {
    "distinct_author_only": "Distinct-author only",
    "same_author_only": "Same-author only",
    "both": "Both mechanisms",
    "undetermined": "Undetermined",
}
MECHANISM_COLORS = {
    "distinct_author_only": "#3B6FB6",
    "same_author_only": "#E69F4B",
    "both": "#8A6FA8",
    "undetermined": "#B5B5B5",
}
# Annual values remain equally spaced, but labels follow the two-year interval
# used by the other longitudinal figures in the manuscript.
TIME_TICKS = list(range(2011, 2026, 2))
MIN_SHARED_CA_KNOWN_FOR_FILLED_POINT = 20
ROBUST_NORMAL_IQR = 1.349
ROBUST_D_SIGNAL = 2.0
ROBUST_D_STRONG = 3.0
MIN_ROBUST_REFERENCE_N = 8
INPUT_FILE_PATTERN = re.compile(
    r"^(20(?:1[1-9]|2[0-5]))_(article_affiliations|article_institution_flags|author_affiliations|corresponding_author_affiliations|institutions)\.csv$"
)
INPUT_COLUMNS = {
    "article_affiliations": [
        "eid",
        "afid",
        "affiliation_name",
        "affiliation_country",
    ],
    "article_institution_flags": [
        "eid",
        "institution_id",
        "name",
        "n_total_authors_article",
    ],
    "author_affiliations": ["eid", "author_id", "afid"],
    "corresponding_author_affiliations": ["eid", "afid"],
    "institutions": ["afid", "institution_id", "name"],
}




def build_reciprocal_pairs(pair_public):
    focal = pair_public[pair_public["partner_type"] == "focal university"].copy()
    lookup = {}
    for row in focal.itertuples(index=False):
        lookup[(int(row.year), row.institution_code, row.partner_code)] = row
    rows = []
    for key, row_ab in lookup.items():
        year, code_a, code_b = key
        if code_a >= code_b:
            continue
        row_ba = lookup.get((year, code_b, code_a))
        if row_ba is None:
            continue
        rows.append(
            {
                "year": year,
                "institution_a": code_a,
                "institution_b": code_b,
                "a_to_b_shared_articles": int(row_ab.shared_articles),
                "b_to_a_shared_articles": int(row_ba.shared_articles),
                "absolute_shared_article_difference": abs(
                    int(row_ab.shared_articles) - int(row_ba.shared_articles)
                ),
                "a_total_articles": int(row_ab.focal_total_articles),
                "b_total_articles": int(row_ba.focal_total_articles),
                "a_to_b_percent": float(row_ab.dependency_percent),
                "b_to_a_percent": float(row_ba.dependency_percent),
                "absolute_directional_difference_pp": abs(
                    float(row_ab.dependency_percent)
                    - float(row_ba.dependency_percent)
                ),
                "larger_direction": (
                    code_a + " to " + code_b
                    if float(row_ab.dependency_percent)
                    >= float(row_ba.dependency_percent)
                    else code_b + " to " + code_a
                ),
            }
        )
    return pd.DataFrame(rows)

def dataframe_records_clean(frame):
    records = []
    for record in frame.to_dict(orient="records"):
        clean = {}
        for key, value in record.items():
            if isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating,)):
                clean[key] = None if not np.isfinite(value) else float(value)
            elif pd.isna(value):
                clean[key] = None
            else:
                clean[key] = value
        records.append(clean)
    return records

def make_summary(
    national,
    institution,
    distribution,
    reciprocal,
    robust_reference_table,
    robust_signals,
    robust_recurrence,
    min_output,
    min_shared_ca_known,
):
    nat = national.set_index("year")
    dist = distribution.set_index("year")
    eligible = institution[institution["total_articles"] >= min_output]
    summary = {
        "script_version": SCRIPT_VERSION,
        "minimum_institutional_output": int(min_output),
        "national": {},
        "institutional": {},
        "partner_dependence": {},
        "mechanism_coverage": {},
        "shared_output_leadership": {},
        "robust_D": {},
    }
    for year in [2011, 2018, 2025]:
        row = nat.loc[year]
        summary["national"][str(year)] = {
            "total_articles": int(row["total_unique_focal_attributed_articles"]),
            "shared_articles": int(row["articles_with_two_or_more_national_entities"]),
            "shared_percent": float(row["shared_percent"]),
            "mechanism_percent_of_shared": {
                mechanism: float(row[mechanism + "_percent_of_shared"])
                for mechanism in MECHANISMS
            },
            "permissive_mechanism_percent_of_shared": {
                mechanism: float(
                    row["permissive_" + mechanism + "_percent_of_shared"]
                )
                for mechanism in MECHANISMS
            },
            "strict_vs_permissive_classification_changes": int(
                row["strict_vs_permissive_classification_changes"]
            ),
            "partner_scope_percent_of_shared": {
                "other_focal_only": float(
                    row["shared_with_other_focal_only_percent_of_shared"]
                ),
                "nonfocal_only": float(
                    row["shared_with_nonfocal_only_percent_of_shared"]
                ),
                "both_scopes": float(
                    row["shared_with_both_partner_scopes_percent_of_shared"]
                ),
            },
        }
        drow = dist.loc[year]
        summary["institutional"][str(year)] = {
            "eligible_institutions": int(drow["n_eligible_institutions"]),
            "shared_median": float(drow["shared_median"]),
            "shared_q1": float(drow["shared_q1"]),
            "shared_q3": float(drow["shared_q3"]),
            "top1_median": float(drow["top1_median"]),
            "top5_union_median": float(drow["top5_union_median"]),
        }
    summary["national"]["change_2011_2025_pp"] = float(
        nat.loc[2025, "shared_percent"] - nat.loc[2011, "shared_percent"]
    )
    summary["national"]["change_2018_2025_pp"] = float(
        nat.loc[2025, "shared_percent"] - nat.loc[2018, "shared_percent"]
    )

    paired = eligible[eligible["year"].isin([2018, 2025])].pivot(
        index="institution_code", columns="year", values="shared_percent"
    ).dropna()
    summary["institutional"]["paired_2018_2025"] = {
        "n_institutions": int(len(paired)),
        "median_change_pp": float((paired[2025] - paired[2018]).median()),
        "increased": int(((paired[2025] - paired[2018]) > 0).sum()),
        "decreased": int(((paired[2025] - paired[2018]) < 0).sum()),
        "unchanged": int(((paired[2025] - paired[2018]) == 0).sum()),
    }

    data_2025 = eligible[eligible["year"] == 2025].copy()
    max_shared = data_2025.sort_values(
        ["shared_percent", "institution_code"], ascending=[False, True]
    ).iloc[0]
    max_top1 = data_2025.sort_values(
        ["leading_partner_dependency_percent", "institution_code"],
        ascending=[False, True],
    ).iloc[0]
    summary["partner_dependence"]["2025"] = {
        "eligible_institutions": int(len(data_2025)),
        "top_partner_is_focal": int(
            (data_2025["leading_partner_type"] == "focal university").sum()
        ),
        "top_partner_is_nonfocal": int(
            (
                data_2025["leading_partner_type"]
                == "other Ecuadorian affiliation entity"
            ).sum()
        ),
        "maximum_shared_output": {
            "institution_code": max_shared["institution_code"],
            "shared_percent": float(max_shared["shared_percent"]),
            "total_articles": int(max_shared["total_articles"]),
        },
        "maximum_top_partner_dependence": {
            "institution_code": max_top1["institution_code"],
            "partner_code": max_top1["leading_partner_code"],
            "dependency_percent": float(
                max_top1["leading_partner_dependency_percent"]
            ),
            "shared_articles": int(max_top1["leading_partner_articles"]),
            "total_articles": int(max_top1["total_articles"]),
        },
    }

    leadership_2025 = institution[
        (institution["year"] == 2025)
        & institution["shared_NLO_percent"].notna()
    ].copy()
    high_shared = leadership_2025["shared_percent"] >= leadership_2025["shared_percent"].median()
    high_nlo = leadership_2025["shared_NLO_percent"] >= leadership_2025["shared_NLO_percent"].median()
    summary["shared_output_leadership"]["2025"] = {
        "institutions_with_shared_CA_known": int(len(leadership_2025)),
        "median_shared_CA_coverage_percent": float(
            leadership_2025["shared_CA_coverage_percent"].median()
        ),
        "median_shared_CAL_percent": float(
            leadership_2025["shared_CAL_percent"].median()
        ),
        "median_shared_NLO_percent": float(
            leadership_2025["shared_NLO_percent"].median()
        ),
        "shared_NLO_q1": quantile(leadership_2025["shared_NLO_percent"], 0.25),
        "shared_NLO_q3": quantile(leadership_2025["shared_NLO_percent"], 0.75),
        "institutions_at_or_above_both_medians": int((high_shared & high_nlo).sum()),
        "median_shared_minus_nonshared_NLO_pp": float(
            leadership_2025["shared_minus_nonshared_NLO_pp"].median()
        ),
        "minimum_filled_point_shared_CA_known": int(
            min_shared_ca_known
        ),
        "spearman_shared_vs_shared_NLO": float(
            leadership_2025["shared_percent"]
            .rank(method="average")
            .corr(leadership_2025["shared_NLO_percent"].rank(method="average"))
        ),
    }
    for year in [2011, 2018]:
        leadership_year = institution[
            (institution["year"] == year)
            & institution["shared_NLO_percent"].notna()
        ].copy()
        summary["shared_output_leadership"][str(year)] = {
            "institutions_with_shared_CA_known": int(len(leadership_year)),
            "median_shared_CA_coverage_percent": float(
                leadership_year["shared_CA_coverage_percent"].median()
            ),
            "median_shared_CAL_percent": float(
                leadership_year["shared_CAL_percent"].median()
            ),
            "median_shared_NLO_percent": float(
                leadership_year["shared_NLO_percent"].median()
            ),
            "shared_NLO_q1": quantile(
                leadership_year["shared_NLO_percent"], 0.25
            ),
            "shared_NLO_q3": quantile(
                leadership_year["shared_NLO_percent"], 0.75
            ),
            "median_shared_minus_nonshared_NLO_pp": float(
                leadership_year["shared_minus_nonshared_NLO_pp"].median()
            ),
        }

    reciprocal_2025 = reciprocal[
        (reciprocal["year"] == 2025)
        & (reciprocal["a_to_b_shared_articles"] >= 10)
        & (reciprocal["b_to_a_shared_articles"] >= 10)
    ].copy()
    if not reciprocal_2025.empty:
        largest = reciprocal_2025.sort_values(
            ["absolute_directional_difference_pp", "institution_a", "institution_b"],
            ascending=[False, True, True],
        ).iloc[0]
        summary["partner_dependence"]["2025"][
            "largest_focal_pair_asymmetry_min_10_shared"
        ] = {
            "institution_a": largest["institution_a"],
            "institution_b": largest["institution_b"],
            "a_to_b_shared_articles": int(largest["a_to_b_shared_articles"]),
            "b_to_a_shared_articles": int(largest["b_to_a_shared_articles"]),
            "a_to_b_percent": float(largest["a_to_b_percent"]),
            "b_to_a_percent": float(largest["b_to_a_percent"]),
            "absolute_difference_pp": float(
                largest["absolute_directional_difference_pp"]
            ),
        }

    shared_total = int(national["articles_with_two_or_more_national_entities"].sum())
    undetermined = int(national["undetermined_articles"].sum())
    summary["mechanism_coverage"] = {
        "shared_articles_all_years": shared_total,
        "undetermined_shared_articles_all_years": undetermined,
        "undetermined_percent_of_shared_all_years": 100.0
        * undetermined
        / shared_total,
        "permissive_not_strict_pair_article_links_all_years": int(
            national["permissive_not_strict_pair_article_links"].sum()
        ),
        "strict_vs_permissive_national_article_classification_changes_all_years": int(
            national["strict_vs_permissive_classification_changes"].sum()
        ),
    }

    share_signals = robust_signals[
        robust_signals["metric_column"] == "shared_percent"
    ].copy()
    nlo_signals = robust_signals[
        robust_signals["metric_column"] == "shared_NLO_percent"
    ].copy()
    share_reference = robust_reference_table[
        robust_reference_table["metric_column"] == "shared_percent"
    ]
    nlo_reference = robust_reference_table[
        robust_reference_table["metric_column"] == "shared_NLO_percent"
    ]
    share_2025 = share_signals[share_signals["year"] == 2025]
    nlo_2025 = nlo_signals[nlo_signals["year"] == 2025]
    summary["robust_D"] = {
        "formula": "D = (x - annual median) / (annual IQR / 1.349)",
        "minimum_reference_institutions": int(MIN_ROBUST_REFERENCE_N),
        "signal_threshold": float(ROBUST_D_SIGNAL),
        "strong_threshold": float(ROBUST_D_STRONG),
        "national_sharing": {
            "interpretation": "two-sided descriptive; no risk direction assigned",
            "valid_reference_years": dataframe_records_clean(
                share_reference.loc[
                    share_reference["reference_valid"], ["year", "reference_n"]
                ]
            ),
            "signal_observations": int(len(share_signals)),
            "upper_signal_observations": int(
                (share_signals["signal_direction"] == "upper").sum()
            ),
            "lower_signal_observations": int(
                (share_signals["signal_direction"] == "lower").sum()
            ),
            "strong_signal_observations": int(
                (share_signals["signal_level"] == "strong").sum()
            ),
            "institutions_with_signal": int(
                share_signals["institution_code"].nunique()
            ),
            "signals_2025": dataframe_records_clean(share_2025),
        },
        "shared_output_NLO": {
            "interpretation": "upper non-leadership deviation; descriptive, not an integrity classification",
            "minimum_classifiable_shared_articles": int(min_shared_ca_known),
            "valid_reference_years": dataframe_records_clean(
                nlo_reference.loc[
                    nlo_reference["reference_valid"], ["year", "reference_n"]
                ]
            ),
            "upper_signal_observations": int(len(nlo_signals)),
            "strong_signal_observations": int(
                (nlo_signals["signal_level"] == "strong").sum()
            ),
            "institutions_with_signal": int(
                nlo_signals["institution_code"].nunique()
            ),
            "signals_2025": dataframe_records_clean(nlo_2025),
        },
        "recurrence": dataframe_records_clean(
            robust_recurrence[
                (robust_recurrence["national_sharing_signal_years"] > 0)
                | (robust_recurrence["shared_NLO_upper_signal_years"] > 0)
            ]
        ),
    }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Section 4.5 national-sharing outputs and figures."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the annual analysis CSV files.",
    )
    parser.add_argument(
        "--mapping-file",
        required=True,
        help="Institution mapping with institution_id, id-paper, and color_hex",
    )
    parser.add_argument(
        "--mapping-encoding",
        default="auto",
        help="CSV encoding; use latin1 for MAPPING_with_colors.csv (default: auto)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--min-output",
        type=int,
        default=50,
        help="Minimum attributed articles for institutional distributions (default: 50)",
    )
    parser.add_argument(
        "--min-shared-ca-known",
        type=int,
        default=MIN_SHARED_CA_KNOWN_FOR_FILLED_POINT,
        help=(
            "Minimum classifiable shared articles for the annual shared-NLO "
            "robust-D reference (default: 20)"
        ),
    )
    return parser.parse_args()


def ensure_dirs(root):
    root = Path(root)
    paths = {'root': root, 'tables': root / 'tables'}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def read_mapping(path, encoding):
    encodings = [encoding] if encoding != "auto" else ["utf-8-sig", "latin1"]
    last_error = None
    frame = None
    used = None
    for enc in encodings:
        try:
            frame = pd.read_csv(path, encoding=enc, dtype=str)
            used = enc
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    if frame is None:
        raise last_error
    required = {"institution_id", "id-paper", "color_hex"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Mapping file lacks columns: " + ", ".join(sorted(missing)))
    frame = frame.copy()
    for col in ["institution_id", "id-paper", "color_hex"]:
        frame[col] = frame[col].astype(str).str.strip()
    if "name" not in frame.columns:
        frame["name"] = frame["institution_id"]
    frame["name"] = frame["name"].astype(str).str.strip()
    frame = frame.drop_duplicates("institution_id", keep="first")
    return frame, used


def read_csv_path(path, usecols):
    return pd.read_csv(
        path,
        usecols=usecols,
        dtype=str,
        low_memory=False,
        encoding="utf-8-sig",
    )


def clean_string_series(series):
    out = series.astype("string").str.strip()
    return out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def mode_text(values):
    vals = [str(v).strip() for v in values if pd.notna(v) and str(v).strip()]
    if not vals:
        return ""
    counts = Counter(vals)
    top = max(counts.values())
    candidates = [v for v, n in counts.items() if n == top]
    return sorted(candidates, key=lambda v: (-len(v), v.casefold(), v))[0]


def quantile(values, q):
    series = pd.Series(list(values), dtype=float).dropna()
    if series.empty:
        return float("nan")
    return float(series.quantile(q))


def classify_pair(holders, focal_token, partner_token):
    focal_holders = holders.get(focal_token, set())
    partner_holders = holders.get(partner_token, set())
    same = bool(focal_holders.intersection(partner_holders))
    distinct_strict = bool(
        focal_holders.difference(partner_holders)
        and partner_holders.difference(focal_holders)
    )
    distinct_permissive = bool(
        focal_holders and partner_holders and len(focal_holders.union(partner_holders)) > 1
    )
    return same, distinct_strict, distinct_permissive


def mechanism_label(same, distinct):
    if same and distinct:
        return "both"
    if same:
        return "same_author_only"
    if distinct:
        return "distinct_author_only"
    return "undetermined"


def process_year(year, frames, partner_name_counter):
    required = {
        "article_affiliations",
        "article_institution_flags",
        "author_affiliations",
        "corresponding_author_affiliations",
        "institutions",
    }
    missing = required.difference(frames)
    if missing:
        raise ValueError(
            "Missing required tables for %s: %s" % (year, ", ".join(sorted(missing)))
        )

    article_aff = frames["article_affiliations"].copy()
    flags = frames["article_institution_flags"].copy()
    author_aff = frames["author_affiliations"].copy()
    corr_aff = frames["corresponding_author_affiliations"].copy()
    institutions = frames["institutions"].copy()

    for frame in [article_aff, flags, author_aff, corr_aff, institutions]:
        for col in ["eid", "afid", "author_id", "institution_id"]:
            if col in frame.columns:
                frame[col] = clean_string_series(frame[col])

    institutions = institutions.dropna(subset=["afid", "institution_id"])
    # The exported table contains the synthetic national aggregate used by the
    # upstream metrics script. It is a denominator, not a Scopus institution,
    # and must never enter the set of national affiliation entities.
    institutions = institutions[institutions["institution_id"] != NATIONAL_ID]
    institutions = institutions.drop_duplicates(["afid", "institution_id"])
    afid_to_inst = dict(zip(institutions["afid"], institutions["institution_id"]))
    inst_name = (
        institutions.groupby("institution_id")["name"].agg(mode_text).to_dict()
    )
    focal_tokens = set(FOCAL_PREFIX + x for x in inst_name)

    corr_aff = corr_aff[corr_aff["eid"].notna()].copy()
    corr_aff = corr_aff.dropna(subset=["eid", "afid"])
    corr_aff = corr_aff.drop_duplicates(["eid", "afid"])
    corr_aff["focal_institution_id"] = corr_aff["afid"].map(afid_to_inst)
    ca_known_eids = set(corr_aff["eid"])
    ca_focal_by_eid = defaultdict(set)
    for row in corr_aff.dropna(subset=["focal_institution_id"])[
        ["eid", "focal_institution_id"]
    ].itertuples(index=False):
        ca_focal_by_eid[row.eid].add(row.focal_institution_id)

    flags = flags.dropna(subset=["eid", "institution_id"])
    focal_flags = flags[flags["institution_id"] != NATIONAL_ID].copy()
    focal_flags = focal_flags.drop_duplicates(["eid", "institution_id"])
    national_flags = flags[flags["institution_id"] == NATIONAL_ID].copy()
    national_flags = national_flags.drop_duplicates("eid")
    national_eids = set(national_flags["eid"])
    focal_union_eids = set(focal_flags["eid"])
    if national_eids != focal_union_eids:
        raise ValueError(
            "%s: NATIONAL EIDs do not equal the union of focal-institution EIDs" % year
        )

    total_authors = {}
    if "n_total_authors_article" in national_flags.columns:
        totals = pd.to_numeric(
            national_flags["n_total_authors_article"], errors="coerce"
        ).fillna(0)
        total_authors = dict(zip(national_flags["eid"], totals.astype(int)))

    article_aff = article_aff[article_aff["eid"].isin(national_eids)].copy()
    article_aff = article_aff.dropna(subset=["eid", "afid"])
    article_aff["affiliation_country"] = article_aff[
        "affiliation_country"
    ].fillna("").astype(str).str.strip()
    mapped_inst = article_aff["afid"].map(afid_to_inst)
    is_focal = mapped_inst.notna()
    is_ecuador = article_aff["affiliation_country"].str.casefold().eq("ecuador")
    national_aff = article_aff[is_focal | is_ecuador].copy()
    mapped_inst = national_aff["afid"].map(afid_to_inst)
    national_aff["entity"] = np.where(
        mapped_inst.notna(),
        FOCAL_PREFIX + mapped_inst.fillna("").astype(str),
        NONFOCAL_PREFIX + national_aff["afid"].astype(str),
    )
    national_aff = national_aff.drop_duplicates(["eid", "afid", "entity"])

    external_aff = national_aff[national_aff["entity"].str.startswith(NONFOCAL_PREFIX)]
    for row in external_aff[["entity", "affiliation_name"]].itertuples(index=False):
        name = "" if pd.isna(row.affiliation_name) else str(row.affiliation_name).strip()
        if name:
            partner_name_counter[(row.entity, name)] += 1

    article_entities = defaultdict(set)
    for row in national_aff[["eid", "entity"]].itertuples(index=False):
        article_entities[row.eid].add(row.entity)
    for row in focal_flags[["eid", "institution_id"]].itertuples(index=False):
        article_entities[row.eid].add(FOCAL_PREFIX + row.institution_id)

    author_aff = author_aff[author_aff["eid"].isin(national_eids)].copy()
    author_aff = author_aff.dropna(subset=["eid", "author_id", "afid"])
    author_aff = author_aff.drop_duplicates(["eid", "author_id", "afid"])
    linked_author_count = (
        author_aff.groupby("eid")["author_id"].nunique().astype(int).to_dict()
    )

    aa = author_aff.copy()
    aa["focal_institution_id"] = aa["afid"].map(afid_to_inst)
    external_map = national_aff[
        national_aff["entity"].str.startswith(NONFOCAL_PREFIX)
    ][["eid", "afid", "entity"]].drop_duplicates()
    aa = aa.merge(
        external_map.rename(columns={"entity": "external_entity"}),
        on=["eid", "afid"],
        how="left",
    )
    aa["entity"] = pd.NA
    focal_mask = aa["focal_institution_id"].notna()
    aa.loc[focal_mask, "entity"] = (
        FOCAL_PREFIX + aa.loc[focal_mask, "focal_institution_id"].astype(str)
    )
    external_mask = (~focal_mask) & aa["external_entity"].notna()
    aa.loc[external_mask, "entity"] = aa.loc[external_mask, "external_entity"]
    aa_nat = aa.dropna(subset=["entity"])[["eid", "author_id", "entity"]]
    aa_nat = aa_nat.drop_duplicates()

    holders_by_eid = defaultdict(lambda: defaultdict(set))
    author_entities_by_eid = defaultdict(lambda: defaultdict(set))
    for row in aa_nat.itertuples(index=False):
        holders_by_eid[row.eid][row.entity].add(row.author_id)
        author_entities_by_eid[row.eid][row.author_id].add(row.entity)

    classification_complete = {}
    entity_coverage = {}
    author_coverage = {}
    for eid in national_eids:
        article_set = article_entities[eid]
        represented = set(holders_by_eid[eid])
        entity_ok = article_set.issubset(represented)
        expected_authors = total_authors.get(eid, linked_author_count.get(eid, 0))
        linked_authors = linked_author_count.get(eid, 0)
        authors_ok = expected_authors > 0 and linked_authors >= expected_authors
        entity_coverage[eid] = entity_ok
        author_coverage[eid] = authors_ok
        classification_complete[eid] = entity_ok and authors_ok

    pair_article_records = []
    institution_article_records = []
    national_article_records = []
    permissive_difference_pairs = 0

    focal_by_eid = defaultdict(set)
    for row in focal_flags[["eid", "institution_id"]].itertuples(index=False):
        focal_by_eid[row.eid].add(row.institution_id)

    strict_sensitivity_records = []
    for eid in sorted(national_eids):
        entities = article_entities[eid]
        holders = holders_by_eid[eid]
        complete = classification_complete[eid]
        is_shared = len(entities) >= 2
        same_national = False
        distinct_national = False
        permissive_national = False
        if is_shared and complete:
            entity_list = sorted(entities)
            for idx, entity_a in enumerate(entity_list):
                for entity_b in entity_list[idx + 1 :]:
                    same, distinct, permissive = classify_pair(
                        holders, entity_a, entity_b
                    )
                    same_national = same_national or same
                    distinct_national = distinct_national or distinct
                    permissive_national = permissive_national or permissive
        national_mechanism = (
            mechanism_label(same_national, distinct_national)
            if is_shared and complete
            else ("undetermined" if is_shared else "not_shared")
        )
        national_mechanism_permissive = (
            mechanism_label(same_national, permissive_national)
            if is_shared and complete
            else ("undetermined" if is_shared else "not_shared")
        )
        n_focal_entities = sum(1 for x in entities if x.startswith(FOCAL_PREFIX))
        n_nonfocal_entities = sum(
            1 for x in entities if x.startswith(NONFOCAL_PREFIX)
        )
        national_article_records.append(
            {
                "year": year,
                "eid": eid,
                "n_national_entities": len(entities),
                "n_focal_entities": n_focal_entities,
                "n_nonfocal_entities": n_nonfocal_entities,
                "shared": int(is_shared),
                "mechanism": national_mechanism,
                "mechanism_permissive": national_mechanism_permissive,
                "classification_complete": int(complete),
                "entity_coverage_complete": int(entity_coverage[eid]),
                "author_coverage_complete": int(author_coverage[eid]),
            }
        )
        if is_shared and national_mechanism != national_mechanism_permissive:
            strict_sensitivity_records.append(
                {
                    "year": year,
                    "analysis_level": "national_article",
                    "institution_id": "",
                    "eid": eid,
                    "strict_mechanism": national_mechanism,
                    "permissive_mechanism": national_mechanism_permissive,
                }
            )

        for institution_id in sorted(focal_by_eid[eid]):
            focal_token = FOCAL_PREFIX + institution_id
            partners = sorted(entities.difference({focal_token}))
            same_any = False
            distinct_any = False
            permissive_any = False
            for partner_token in partners:
                same, distinct, permissive = classify_pair(
                    holders, focal_token, partner_token
                )
                if permissive and not distinct:
                    permissive_difference_pairs += 1
                same_any = same_any or same
                distinct_any = distinct_any or distinct
                permissive_any = permissive_any or permissive
                pair_mechanism = (
                    mechanism_label(same, distinct)
                    if complete
                    else "undetermined"
                )
                pair_article_records.append(
                    {
                        "year": year,
                        "eid": eid,
                        "institution_id": institution_id,
                        "partner_token": partner_token,
                        "mechanism": pair_mechanism,
                        "same_author": int(same and complete),
                        "distinct_author_strict": int(distinct and complete),
                        "distinct_author_permissive": int(permissive and complete),
                        "classification_complete": int(complete),
                    }
                )
            institution_mechanism = "not_shared"
            institution_mechanism_permissive = "not_shared"
            if partners:
                institution_mechanism = (
                    mechanism_label(same_any, distinct_any)
                    if complete
                    else "undetermined"
                )
                institution_mechanism_permissive = (
                    mechanism_label(same_any, permissive_any)
                    if complete
                    else "undetermined"
                )
            institution_article_records.append(
                {
                    "year": year,
                    "eid": eid,
                    "institution_id": institution_id,
                    "shared": int(bool(partners)),
                    "mechanism": institution_mechanism,
                    "mechanism_permissive": institution_mechanism_permissive,
                    "classification_complete": int(complete),
                    "has_other_focal": int(
                        any(x.startswith(FOCAL_PREFIX) for x in partners)
                    ),
                    "has_nonfocal_national": int(
                        any(x.startswith(NONFOCAL_PREFIX) for x in partners)
                    ),
                    "CA_known": int(eid in ca_known_eids),
                    "flag_CAL": int(
                        eid in ca_known_eids
                        and institution_id in ca_focal_by_eid.get(eid, set())
                    ),
                    "flag_NLO": int(
                        eid in ca_known_eids
                        and institution_id not in ca_focal_by_eid.get(eid, set())
                    ),
                }
            )
            if (
                partners
                and institution_mechanism
                != institution_mechanism_permissive
            ):
                strict_sensitivity_records.append(
                    {
                        "year": year,
                        "analysis_level": "institution_article",
                        "institution_id": institution_id,
                        "eid": eid,
                        "strict_mechanism": institution_mechanism,
                        "permissive_mechanism": institution_mechanism_permissive,
                    }
                )

    national_articles = pd.DataFrame(national_article_records)
    institution_articles = pd.DataFrame(institution_article_records)
    pair_articles = pd.DataFrame(pair_article_records)

    total_national = int(len(national_articles))
    shared_national = int(national_articles["shared"].sum())
    focal_scope = national_articles["n_focal_entities"] >= 2
    nonfocal_scope = (
        (national_articles["shared"] == 1)
        & (national_articles["n_nonfocal_entities"] >= 1)
    )
    focal_only_scope = focal_scope & (~nonfocal_scope)
    nonfocal_only_scope = nonfocal_scope & (~focal_scope)
    both_scopes = focal_scope & nonfocal_scope
    national_row = {
        "year": year,
        "total_unique_focal_attributed_articles": total_national,
        "articles_with_one_national_entity": total_national - shared_national,
        "articles_with_two_or_more_national_entities": shared_national,
        "shared_percent": 100.0 * shared_national / total_national,
        "articles_with_two_or_more_focal_institutions": int(
            focal_scope.sum()
        ),
        "articles_with_nonfocal_ecuadorian_partner": int(
            nonfocal_scope.sum()
        ),
        "shared_with_other_focal_only_articles": int(focal_only_scope.sum()),
        "shared_with_nonfocal_only_articles": int(nonfocal_only_scope.sum()),
        "shared_with_both_partner_scopes_articles": int(both_scopes.sum()),
        "distinct_national_entities": int(
            len(set().union(*[article_entities[eid] for eid in national_eids]))
        ),
        "distinct_focal_institutions": int(len(focal_tokens)),
        "distinct_nonfocal_ecuadorian_afids": int(
            len(
                {
                    x
                    for eid in national_eids
                    for x in article_entities[eid]
                    if x.startswith(NONFOCAL_PREFIX)
                }
            )
        ),
        "all_author_links_complete_articles": int(
            national_articles["author_coverage_complete"].sum()
        ),
        "all_national_entities_linked_articles": int(
            national_articles["entity_coverage_complete"].sum()
        ),
        "fully_classifiable_articles": int(
            national_articles["classification_complete"].sum()
        ),
        "permissive_not_strict_pair_article_links": int(
            permissive_difference_pairs
        ),
    }
    for scope in [
        "shared_with_other_focal_only",
        "shared_with_nonfocal_only",
        "shared_with_both_partner_scopes",
    ]:
        count = national_row[scope + "_articles"]
        national_row[scope + "_percent_of_total"] = 100.0 * count / total_national
        national_row[scope + "_percent_of_shared"] = (
            100.0 * count / shared_national if shared_national else float("nan")
        )
    for mechanism in MECHANISMS:
        count = int(
            (
                (national_articles["shared"] == 1)
                & (national_articles["mechanism"] == mechanism)
            ).sum()
        )
        national_row[mechanism + "_articles"] = count
        national_row[mechanism + "_percent_of_total"] = (
            100.0 * count / total_national
        )
        national_row[mechanism + "_percent_of_shared"] = (
            100.0 * count / shared_national if shared_national else float("nan")
        )
        permissive_count = int(
            (
                (national_articles["shared"] == 1)
                & (national_articles["mechanism_permissive"] == mechanism)
            ).sum()
        )
        national_row["permissive_" + mechanism + "_articles"] = permissive_count
        national_row[
            "permissive_" + mechanism + "_percent_of_shared"
        ] = (
            100.0 * permissive_count / shared_national
            if shared_national
            else float("nan")
        )
    national_row["strict_vs_permissive_classification_changes"] = int(
        (
            (national_articles["shared"] == 1)
            & (
                national_articles["mechanism"]
                != national_articles["mechanism_permissive"]
            )
        ).sum()
    )

    institution_rows = []
    pair_rows = []
    if not pair_articles.empty:
        grouped = pair_articles.groupby(
            ["institution_id", "partner_token"], sort=True, dropna=False
        )
        for (institution_id, partner_token), group in grouped:
            row = {
                "year": year,
                "institution_id": institution_id,
                "partner_token": partner_token,
                "shared_articles": int(group["eid"].nunique()),
            }
            for mechanism in MECHANISMS:
                row[mechanism + "_articles"] = int(
                    group.loc[group["mechanism"] == mechanism, "eid"].nunique()
                )
            row["same_author_articles"] = int(
                group.loc[group["same_author"] == 1, "eid"].nunique()
            )
            row["distinct_author_strict_articles"] = int(
                group.loc[group["distinct_author_strict"] == 1, "eid"].nunique()
            )
            row["distinct_author_permissive_articles"] = int(
                group.loc[
                    group["distinct_author_permissive"] == 1, "eid"
                ].nunique()
            )
            pair_rows.append(row)
    pair_annual = pd.DataFrame(pair_rows)

    for institution_id, group in institution_articles.groupby("institution_id"):
        total = int(group["eid"].nunique())
        shared = int(group.loc[group["shared"] == 1, "eid"].nunique())
        row = {
            "year": year,
            "institution_id": institution_id,
            "institution_name": inst_name.get(institution_id, institution_id),
            "total_articles": total,
            "shared_articles": shared,
            "shared_percent": 100.0 * shared / total if total else float("nan"),
            "articles_with_other_focal": int(
                group.loc[group["has_other_focal"] == 1, "eid"].nunique()
            ),
            "articles_with_nonfocal_national": int(
                group.loc[group["has_nonfocal_national"] == 1, "eid"].nunique()
            ),
            "fully_classifiable_articles": int(
                group.loc[group["classification_complete"] == 1, "eid"].nunique()
            ),
        }
        for subset_name, subset_mask in [
            ("shared", group["shared"] == 1),
            ("nonshared", group["shared"] == 0),
            ("all", pd.Series(True, index=group.index)),
        ]:
            subset = group.loc[subset_mask]
            subset_total = int(subset["eid"].nunique())
            ca_known = int(subset.loc[subset["CA_known"] == 1, "eid"].nunique())
            ca_led = int(subset.loc[subset["flag_CAL"] == 1, "eid"].nunique())
            ca_nonled = int(subset.loc[subset["flag_NLO"] == 1, "eid"].nunique())
            row[subset_name + "_CA_known_articles"] = ca_known
            row[subset_name + "_CA_unknown_articles"] = subset_total - ca_known
            row[subset_name + "_CA_coverage_percent"] = (
                100.0 * ca_known / subset_total if subset_total else float("nan")
            )
            row[subset_name + "_CA_led_articles"] = ca_led
            row[subset_name + "_CAL_percent"] = (
                100.0 * ca_led / ca_known if ca_known else float("nan")
            )
            row[subset_name + "_NLO_articles"] = ca_nonled
            row[subset_name + "_NLO_percent"] = (
                100.0 * ca_nonled / ca_known if ca_known else float("nan")
            )
        row["shared_minus_nonshared_NLO_pp"] = (
            row["shared_NLO_percent"] - row["nonshared_NLO_percent"]
            if np.isfinite(row["shared_NLO_percent"])
            and np.isfinite(row["nonshared_NLO_percent"])
            else float("nan")
        )
        focal_only_group = (
            (group["has_other_focal"] == 1)
            & (group["has_nonfocal_national"] == 0)
        )
        nonfocal_only_group = (
            (group["has_other_focal"] == 0)
            & (group["has_nonfocal_national"] == 1)
        )
        both_scope_group = (
            (group["has_other_focal"] == 1)
            & (group["has_nonfocal_national"] == 1)
        )
        row["shared_with_other_focal_only_articles"] = int(
            group.loc[focal_only_group, "eid"].nunique()
        )
        row["shared_with_nonfocal_only_articles"] = int(
            group.loc[nonfocal_only_group, "eid"].nunique()
        )
        row["shared_with_both_partner_scopes_articles"] = int(
            group.loc[both_scope_group, "eid"].nunique()
        )
        for mechanism in MECHANISMS:
            count = int(
                group.loc[
                    (group["shared"] == 1) & (group["mechanism"] == mechanism),
                    "eid",
                ].nunique()
            )
            row[mechanism + "_articles"] = count
            row[mechanism + "_percent_of_total"] = 100.0 * count / total
            row[mechanism + "_percent_of_shared"] = (
                100.0 * count / shared if shared else float("nan")
            )
        row["strict_vs_permissive_classification_changes"] = int(
            group.loc[
                (group["shared"] == 1)
                & (group["mechanism"] != group["mechanism_permissive"]),
                "eid",
            ].nunique()
        )

        inst_pairs = pair_annual[pair_annual["institution_id"] == institution_id]
        if inst_pairs.empty:
            row.update(
                {
                    "leading_partner_token": "",
                    "leading_partner_articles": 0,
                    "leading_partner_dependency_percent": 0.0,
                    "top_five_partner_union_articles": 0,
                    "top_five_partner_union_percent": 0.0,
                    "n_national_partners": 0,
                }
            )
        else:
            ranked = inst_pairs.sort_values(
                ["shared_articles", "partner_token"], ascending=[False, True]
            )
            leading = ranked.iloc[0]
            top_tokens = list(ranked.head(5)["partner_token"])
            top_eids = set(
                pair_articles.loc[
                    (pair_articles["institution_id"] == institution_id)
                    & (pair_articles["partner_token"].isin(top_tokens)),
                    "eid",
                ]
            )
            row.update(
                {
                    "leading_partner_token": leading["partner_token"],
                    "leading_partner_articles": int(leading["shared_articles"]),
                    "leading_partner_dependency_percent": 100.0
                    * float(leading["shared_articles"])
                    / total,
                    "top_five_partner_union_articles": int(len(top_eids)),
                    "top_five_partner_union_percent": 100.0
                    * len(top_eids)
                    / total,
                    "n_national_partners": int(len(ranked)),
                }
            )
        institution_rows.append(row)

    institution_annual = pd.DataFrame(institution_rows)
    if not pair_annual.empty:
        totals = institution_annual.set_index("institution_id")["total_articles"]
        pair_annual["focal_total_articles"] = pair_annual["institution_id"].map(totals)
        pair_annual["dependency_percent"] = (
            100.0 * pair_annual["shared_articles"] / pair_annual["focal_total_articles"]
        )

    audit_row = {
        "year": year,
        "national_articles": total_national,
        "focal_article_links": int(len(focal_flags)),
        "institution_article_output_rows": int(len(institution_articles)),
        "pair_article_rows": int(len(pair_articles)),
        "institution_output_sum": int(institution_annual["total_articles"].sum()),
        "shared_institution_article_sum": int(institution_annual["shared_articles"].sum()),
        "national_mechanism_sum": int(
            sum(national_row[x + "_articles"] for x in MECHANISMS)
        ),
        "national_permissive_mechanism_sum": int(
            sum(
                national_row["permissive_" + x + "_articles"]
                for x in MECHANISMS
            )
        ),
        "national_shared_articles": shared_national,
        "national_mechanisms_equal_shared": int(
            sum(national_row[x + "_articles"] for x in MECHANISMS)
            == shared_national
        ),
        "national_permissive_mechanisms_equal_shared": int(
            sum(
                national_row["permissive_" + x + "_articles"]
                for x in MECHANISMS
            )
            == shared_national
        ),
        "national_partner_scopes_equal_shared": int(
            sum(
                national_row[x + "_articles"]
                for x in [
                    "shared_with_other_focal_only",
                    "shared_with_nonfocal_only",
                    "shared_with_both_partner_scopes",
                ]
            )
            == shared_national
        ),
        "institution_mechanism_mismatches": int(
            (
                institution_annual["shared_articles"]
                != institution_annual[
                    [x + "_articles" for x in MECHANISMS]
                ].sum(axis=1)
            ).sum()
        ),
        "institution_partner_scope_mismatches": int(
            (
                institution_annual["shared_articles"]
                != institution_annual[
                    [
                        "shared_with_other_focal_only_articles",
                        "shared_with_nonfocal_only_articles",
                        "shared_with_both_partner_scopes_articles",
                    ]
                ].sum(axis=1)
            ).sum()
        ),
        "institution_all_CA_partition_mismatches": int(
            (
                institution_annual["all_CA_known_articles"]
                != institution_annual["all_CA_led_articles"]
                + institution_annual["all_NLO_articles"]
            ).sum()
        ),
        "institution_shared_CA_partition_mismatches": int(
            (
                institution_annual["shared_CA_known_articles"]
                != institution_annual["shared_CA_led_articles"]
                + institution_annual["shared_NLO_articles"]
            ).sum()
        ),
        "institution_nonshared_CA_partition_mismatches": int(
            (
                institution_annual["nonshared_CA_known_articles"]
                != institution_annual["nonshared_CA_led_articles"]
                + institution_annual["nonshared_NLO_articles"]
            ).sum()
        ),
    }

    return {
        "national": pd.DataFrame([national_row]),
        "institution": institution_annual,
        "pair": pair_annual,
        "audit": pd.DataFrame([audit_row]),
        "strict_sensitivity": pd.DataFrame(strict_sensitivity_records),
    }


def combine_processed_outputs(all_outputs, partner_name_counter, seen_files, source):
    if not all_outputs["national"]:
        raise ValueError("No annual input tables were processed from %s" % source)

    processed_years = set(
        pd.concat(all_outputs["national"], ignore_index=True)["year"].astype(int)
    )
    missing_years = sorted(set(YEARS).difference(processed_years))
    if missing_years:
        raise ValueError("Input lacks processed years: " + repr(missing_years))

    combined = {
        key: pd.concat(frames, ignore_index=True, sort=False)
        for key, frames in all_outputs.items()
    }
    combined["loaded_files"] = pd.DataFrame(seen_files)

    names_by_token = {}
    counts_by_token = defaultdict(Counter)
    for (token, name), count in partner_name_counter.items():
        counts_by_token[token][name] += count
    for token, counts in counts_by_token.items():
        top = max(counts.values())
        candidates = [name for name, count in counts.items() if count == top]
        names_by_token[token] = sorted(
            candidates, key=lambda v: (-len(v), v.casefold(), v)
        )[0]
    combined["partner_names"] = names_by_token
    return combined


def load_and_process_directory(input_dir):
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError("--input-dir is not a directory: %s" % input_dir)

    selected = {}
    for path in sorted(input_dir.rglob("*.csv")):
        match = INPUT_FILE_PATTERN.match(path.name)
        if not match:
            continue
        key = (int(match.group(1)), match.group(2))
        if key in selected:
            raise ValueError(
                "Duplicate input table for %s %s: %s and %s"
                % (key[0], key[1], selected[key], path)
            )
        selected[key] = path

    all_outputs = {
        "national": [],
        "institution": [],
        "pair": [],
        "audit": [],
        "strict_sensitivity": [],
    }
    partner_name_counter = Counter()
    seen_files = []
    required_tables = set(INPUT_COLUMNS)

    for year in YEARS:
        available = {table for input_year, table in selected if input_year == year}
        missing = sorted(required_tables.difference(available))
        if missing:
            raise ValueError(
                "Missing required tables for %s in %s: %s"
                % (year, input_dir, ", ".join(missing))
            )
        frames = {}
        for table in INPUT_COLUMNS:
            path = selected[(year, table)]
            frame = read_csv_path(path, INPUT_COLUMNS[table])
            frames[table] = frame
            seen_files.append(
                {
                    "year": year,
                    "table": table,
                    "member": str(path.relative_to(input_dir)),
                    "bytes": int(path.stat().st_size),
                    "rows": int(len(frame)),
                }
            )
        result = process_year(year, frames, partner_name_counter)
        for key in all_outputs:
            all_outputs[key].append(result[key])
        print("Processed %s" % year, flush=True)

    return combine_processed_outputs(
        all_outputs, partner_name_counter, seen_files, input_dir
    )


def add_codes(outputs, mapping):
    inst_to_code = dict(zip(mapping["institution_id"], mapping["id-paper"]))
    inst_to_color = dict(zip(mapping["institution_id"], mapping["color_hex"]))
    inst_to_name = dict(zip(mapping["institution_id"], mapping["name"]))
    nonfocal_tokens = sorted(
        set(
            outputs["pair"].loc[
                outputs["pair"]["partner_token"].str.startswith(NONFOCAL_PREFIX),
                "partner_token",
            ]
        ),
        key=lambda x: (len(x), x),
    )
    nonfocal_codes = {
        token: "N-%04d" % (idx + 1) for idx, token in enumerate(nonfocal_tokens)
    }

    def partner_code(token):
        if str(token).startswith(FOCAL_PREFIX):
            institution_id = str(token)[len(FOCAL_PREFIX) :]
            return inst_to_code.get(institution_id, institution_id)
        return nonfocal_codes.get(token, str(token))

    institution = outputs["institution"].copy()
    institution["institution_code"] = institution["institution_id"].map(inst_to_code)
    institution["institution_color"] = institution["institution_id"].map(inst_to_color)
    institution["leading_partner_code"] = institution["leading_partner_token"].map(
        partner_code
    )
    institution["leading_partner_type"] = np.where(
        institution["leading_partner_token"].str.startswith(FOCAL_PREFIX),
        "focal university",
        np.where(
            institution["leading_partner_token"].eq(""),
            "none",
            "other Ecuadorian affiliation entity",
        ),
    )

    pair = outputs["pair"].copy()
    pair["institution_code"] = pair["institution_id"].map(inst_to_code)
    pair["partner_code"] = pair["partner_token"].map(partner_code)
    pair["partner_type"] = np.where(
        pair["partner_token"].str.startswith(FOCAL_PREFIX),
        "focal university",
        "other Ecuadorian affiliation entity",
    )
    pair["partner_name"] = pair["partner_token"].map(outputs["partner_names"])
    focal_partner_mask = pair["partner_token"].str.startswith(FOCAL_PREFIX)
    pair.loc[focal_partner_mask, "partner_name"] = pair.loc[
        focal_partner_mask, "partner_token"
    ].str[len(FOCAL_PREFIX) :].map(inst_to_name)

    nonfocal_codebook_rows = []
    for token in nonfocal_tokens:
        nonfocal_codebook_rows.append(
            {
                "partner_code": nonfocal_codes[token],
                "partner_token": token,
                "scopus_afid": token[len(NONFOCAL_PREFIX) :],
                "partner_name": outputs["partner_names"].get(token, ""),
            }
        )
    nonfocal_codebook = pd.DataFrame(nonfocal_codebook_rows)
    return institution, pair, nonfocal_codebook


def make_distribution_table(institution, min_output):
    eligible = institution[institution["total_articles"] >= min_output].copy()
    rows = []
    for year, group in eligible.groupby("year"):
        row = {"year": int(year), "n_eligible_institutions": int(len(group))}
        for column, prefix in [
            ("shared_percent", "shared"),
            ("leading_partner_dependency_percent", "top1"),
            ("top_five_partner_union_percent", "top5_union"),
        ]:
            row[prefix + "_q1"] = quantile(group[column], 0.25)
            row[prefix + "_median"] = quantile(group[column], 0.50)
            row[prefix + "_q3"] = quantile(group[column], 0.75)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year")


def robust_reference(values):
    series = pd.Series(list(values), dtype=float).dropna()
    n = int(len(series))
    distinct = int(series.nunique())
    median = quantile(series, 0.50)
    q1 = quantile(series, 0.25)
    q3 = quantile(series, 0.75)
    iqr = q3 - q1 if np.isfinite(q1) and np.isfinite(q3) else float("nan")
    scale = (
        iqr / ROBUST_NORMAL_IQR
        if np.isfinite(iqr) and iqr > 0
        else float("nan")
    )
    valid = bool(
        n >= MIN_ROBUST_REFERENCE_N
        and distinct >= 2
        and np.isfinite(scale)
        and scale > 0
    )
    if n < MIN_ROBUST_REFERENCE_N:
        reason = "fewer than %s eligible institutions" % MIN_ROBUST_REFERENCE_N
    elif distinct < 2:
        reason = "fewer than two distinct values"
    elif not np.isfinite(iqr) or iqr <= 0:
        reason = "non-positive IQR"
    else:
        reason = "valid"
    return {
        "reference_n": n,
        "reference_distinct_values": distinct,
        "reference_median": median,
        "reference_q1": q1,
        "reference_q3": q3,
        "reference_iqr": iqr,
        "reference_robust_scale": scale,
        "reference_valid": valid,
        "reference_status": reason,
    }


def maximum_consecutive_years(years):
    ordered = sorted(set(int(year) for year in years))
    if not ordered:
        return 0
    longest = 1
    current = 1
    for previous, present in zip(ordered[:-1], ordered[1:]):
        if present == previous + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return int(longest)


def make_robust_d_tables(institution, min_output, min_shared_ca_known):
    """Calculate annual robust-D values and signal recurrence.

    National sharing is interpreted two-sidedly because neither high nor low
    sharing is assigned an intrinsic risk direction. Shared-output NLO retains
    an upper direction because larger values denote less corresponding-author
    leadership, but it remains a descriptive metadata-based signal.
    """

    data = institution.copy()
    data["shared_D_reference_eligible"] = data["total_articles"] >= min_output
    data["shared_NLO_D_reference_eligible"] = (
        (data["total_articles"] >= min_output)
        & (data["shared_CA_known_articles"] >= min_shared_ca_known)
        & data["shared_NLO_percent"].notna()
    )

    for prefix in ["shared", "shared_NLO"]:
        data[prefix + "_D_reference_valid"] = False
        data[prefix + "_D_reference_n"] = np.nan
        data[prefix + "_D_reference_median"] = np.nan
        data[prefix + "_D_reference_q1"] = np.nan
        data[prefix + "_D_reference_q3"] = np.nan
        data[prefix + "_D_reference_iqr"] = np.nan
        data[prefix + "_D_robust_scale"] = np.nan
        data[prefix + "_D"] = np.nan
        data[prefix + "_D_signal_direction"] = ""
        data[prefix + "_D_signal_level"] = ""

    reference_rows = []
    signal_rows = []
    specifications = [
        {
            "metric": "shared_percent",
            "prefix": "shared",
            "label": "National sharing",
            "eligibility": "shared_D_reference_eligible",
            "denominator": "total_articles",
            "interpretation": "two-sided descriptive",
        },
        {
            "metric": "shared_NLO_percent",
            "prefix": "shared_NLO",
            "label": "Shared-output NLO",
            "eligibility": "shared_NLO_D_reference_eligible",
            "denominator": "shared_CA_known_articles",
            "interpretation": "upper non-leadership deviation",
        },
    ]

    for year, group in data.groupby("year", sort=True):
        for specification in specifications:
            metric = specification["metric"]
            prefix = specification["prefix"]
            eligible = group[specification["eligibility"]] & group[metric].notna()
            reference = robust_reference(group.loc[eligible, metric])
            reference_rows.append(
                dict(
                    {
                        "year": int(year),
                        "metric": specification["label"],
                        "metric_column": metric,
                        "eligibility_rule": (
                            "total_articles >= %s" % min_output
                            if prefix == "shared"
                            else "total_articles >= %s and shared_CA_known_articles >= %s"
                            % (min_output, min_shared_ca_known)
                        ),
                        "signal_interpretation": specification["interpretation"],
                    },
                    **reference
                )
            )

            all_year_indices = group.index
            data.loc[all_year_indices, prefix + "_D_reference_valid"] = bool(
                reference["reference_valid"]
            )
            for suffix, key in [
                ("_D_reference_n", "reference_n"),
                ("_D_reference_median", "reference_median"),
                ("_D_reference_q1", "reference_q1"),
                ("_D_reference_q3", "reference_q3"),
                ("_D_reference_iqr", "reference_iqr"),
                ("_D_robust_scale", "reference_robust_scale"),
            ]:
                data.loc[all_year_indices, prefix + suffix] = reference[key]

            if not reference["reference_valid"]:
                continue

            eligible_indices = group.index[eligible]
            d_values = (
                data.loc[eligible_indices, metric]
                - reference["reference_median"]
            ) / reference["reference_robust_scale"]
            data.loc[eligible_indices, prefix + "_D"] = d_values

            if prefix == "shared":
                lower = d_values <= -ROBUST_D_SIGNAL
                upper = d_values >= ROBUST_D_SIGNAL
                strong = d_values.abs() >= ROBUST_D_STRONG
                data.loc[d_values.index[lower], prefix + "_D_signal_direction"] = "lower"
                data.loc[d_values.index[upper], prefix + "_D_signal_direction"] = "upper"
                signal_mask = lower | upper
            else:
                signal_mask = d_values >= ROBUST_D_SIGNAL
                strong = d_values >= ROBUST_D_STRONG
                data.loc[d_values.index[signal_mask], prefix + "_D_signal_direction"] = "upper"

            elevated_indices = d_values.index[signal_mask & ~strong]
            strong_indices = d_values.index[signal_mask & strong]
            data.loc[elevated_indices, prefix + "_D_signal_level"] = "elevated"
            data.loc[strong_indices, prefix + "_D_signal_level"] = "strong"

            for index in d_values.index[signal_mask]:
                row = data.loc[index]
                signal_rows.append(
                    {
                        "year": int(year),
                        "metric": specification["label"],
                        "metric_column": metric,
                        "institution_code": row["institution_code"],
                        "value_percent": float(row[metric]),
                        "denominator_name": specification["denominator"],
                        "denominator": int(row[specification["denominator"]]),
                        "D": float(row[prefix + "_D"]),
                        "signal_direction": row[prefix + "_D_signal_direction"],
                        "signal_level": row[prefix + "_D_signal_level"],
                        "signal_interpretation": specification["interpretation"],
                        "reference_n": int(reference["reference_n"]),
                        "reference_median": float(reference["reference_median"]),
                        "reference_q1": float(reference["reference_q1"]),
                        "reference_q3": float(reference["reference_q3"]),
                        "reference_iqr": float(reference["reference_iqr"]),
                        "reference_robust_scale": float(
                            reference["reference_robust_scale"]
                        ),
                    }
                )

    signals = pd.DataFrame(signal_rows)
    reference_table = pd.DataFrame(reference_rows)

    recurrence_rows = []
    for code, group in data.groupby("institution_code", sort=True):
        share_evaluable = group[group["shared_D"].notna()]
        share_signals = share_evaluable[
            share_evaluable["shared_D"].abs() >= ROBUST_D_SIGNAL
        ]
        share_strong = share_signals[
            share_signals["shared_D"].abs() >= ROBUST_D_STRONG
        ]
        nlo_evaluable = group[group["shared_NLO_D"].notna()]
        nlo_signals = nlo_evaluable[
            nlo_evaluable["shared_NLO_D"] >= ROBUST_D_SIGNAL
        ]
        nlo_strong = nlo_signals[
            nlo_signals["shared_NLO_D"] >= ROBUST_D_STRONG
        ]
        recurrence_rows.append(
            {
                "institution_code": code,
                "national_sharing_evaluable_years": int(len(share_evaluable)),
                "national_sharing_signal_years": int(len(share_signals)),
                "national_sharing_upper_signal_years": int(
                    (share_signals["shared_D"] >= ROBUST_D_SIGNAL).sum()
                ),
                "national_sharing_lower_signal_years": int(
                    (share_signals["shared_D"] <= -ROBUST_D_SIGNAL).sum()
                ),
                "national_sharing_strong_years": int(len(share_strong)),
                "national_sharing_max_consecutive_signal_years": maximum_consecutive_years(
                    share_signals["year"]
                ),
                "national_sharing_signal_year_list": ";".join(
                    str(int(value)) for value in sorted(share_signals["year"])
                ),
                "shared_NLO_evaluable_years": int(len(nlo_evaluable)),
                "shared_NLO_upper_signal_years": int(len(nlo_signals)),
                "shared_NLO_strong_years": int(len(nlo_strong)),
                "shared_NLO_max_consecutive_signal_years": maximum_consecutive_years(
                    nlo_signals["year"]
                ),
                "shared_NLO_signal_year_list": ";".join(
                    str(int(value)) for value in sorted(nlo_signals["year"])
                ),
            }
        )
    recurrence = pd.DataFrame(recurrence_rows)
    recurrence = recurrence.sort_values(
        [
            "national_sharing_signal_years",
            "national_sharing_strong_years",
            "shared_NLO_upper_signal_years",
            "institution_code",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return data, reference_table, signals, recurrence


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "Source Sans 3",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def tidy_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6, zorder=0)


def style_time_axis(ax):
    ax.set_xticks(TIME_TICKS)
    ax.set_xticklabels([str(year) for year in TIME_TICKS], rotation=45, ha="right")
    ax.set_xlim(2010.35, 2025.65)


def add_panel_label(ax, label):
    ax.text(
        0.01,
        0.92,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="#202020",
        zorder=10,
    )


def save_figure(fig, base_path):
    # Write each format to a temporary sibling and publish atomically. This
    # prevents a partially written raster from replacing a valid prior output
    # if a run is interrupted.
    figure_style.finish(fig, base_path)
    specs = [(".pdf", {}), (".svg", {}), (".png", {"dpi": 300})]
    for extension, options in specs:
        final_path = Path(str(base_path) + extension)
        temporary_path = Path(str(base_path) + ".tmp" + extension)
        fig.savefig(str(temporary_path), bbox_inches="tight", **options)
        os.replace(str(temporary_path), str(final_path))


def plot_main_figure(
    national,
    institution,
    distribution,
    robust_signals,
    min_output,
    min_shared_ca_known,
    path,
):
    if output_utils.data_only():
        return
    configure_style()
    years = np.array(YEARS)
    fig = plt.figure(figsize=(7.35, 7.15))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    nat = national.set_index("year").loc[YEARS].reset_index()
    eligible = institution[institution["total_articles"] >= min_output].copy()
    values_by_year = {
        year: list(
            eligible.loc[
                eligible["year"] == year, "shared_percent"
            ].dropna()
        )
        for year in YEARS
    }
    box_years = [year for year in YEARS if len(values_by_year[year]) >= 4]
    box_values = [values_by_year[year] for year in box_years]
    bp = ax_a.boxplot(
        box_values,
        positions=box_years,
        widths=0.58,
        whis=(0, 100),
        manage_ticks=False,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 0.9},
        boxprops={"facecolor": "#D9D9D9", "edgecolor": "#777777", "linewidth": 0.7},
        whiskerprops={"color": "#777777", "linewidth": 0.7},
        capprops={"color": "#777777", "linewidth": 0.7},
    )
    del bp
    for year in YEARS:
        if 0 < len(values_by_year[year]) < 4:
            ax_a.scatter(
                [year] * len(values_by_year[year]),
                values_by_year[year],
                s=13,
                color="#8A8A8A",
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
            )
    ax_a.plot(
        years,
        nat["shared_percent"],
        color="#202020",
        linewidth=1.5,
        label="National aggregate",
        zorder=4,
    )
    color_lookup = institution.drop_duplicates("institution_code").set_index(
        "institution_code"
    )["institution_color"].to_dict()
    sharing_signals = robust_signals[
        robust_signals["metric_column"] == "shared_percent"
    ].copy()
    if not sharing_signals.empty:
        for row in sharing_signals.itertuples(index=False):
            ax_a.scatter(
                row.year,
                row.value_percent,
                s=24,
                marker="o" if row.signal_direction == "upper" else "v",
                color=color_lookup.get(row.institution_code, "#555555"),
                edgecolor="#202020" if row.signal_level == "strong" else "white",
                linewidth=1.05 if row.signal_level == "strong" else 0.35,
                zorder=5,
            )
    ax_a.set_title("National sharing across institutional portfolios", pad=7)
    add_panel_label(ax_a, "(a)")
    ax_a.set_ylabel("Shared institutional output (%)")
    style_time_axis(ax_a)
    ax_a.set_ylim(bottom=0)
    tidy_axis(ax_a)

    bottom = np.zeros(len(nat))
    for mechanism in MECHANISMS:
        values_m = nat[mechanism + "_percent_of_shared"].to_numpy(dtype=float)
        ax_b.bar(
            years,
            values_m,
            bottom=bottom,
            width=0.78,
            color=MECHANISM_COLORS[mechanism],
            label=MECHANISM_LABELS[mechanism],
            edgecolor="white",
            linewidth=0.25,
        )
        bottom += np.nan_to_num(values_m)
    ax_b.set_title("Mechanisms within nationally shared output", pad=7)
    add_panel_label(ax_b, "(b)")
    ax_b.set_ylabel("Composition of shared articles (%)")
    style_time_axis(ax_b)
    ax_b.set_ylim(0, 100)
    ax_b.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        columnspacing=1.0,
        handletextpad=0.5,
    )
    tidy_axis(ax_b)

    leadership = institution[institution["year"] == 2025].copy()
    leadership = leadership.dropna(subset=["shared_percent", "shared_NLO_percent"])
    if leadership["institution_color"].isna().any():
        missing_codes = sorted(
            leadership.loc[
                leadership["institution_color"].isna(), "institution_code"
            ].astype(str)
        )
        raise ValueError(
            "Institution colors are missing for panel C: " + ", ".join(missing_codes)
        )
    known = leadership["shared_CA_known_articles"].to_numpy(dtype=float)
    sizes = 20.0 + 3.0 * np.sqrt(np.maximum(known, 1.0))
    stable = known >= min_shared_ca_known
    size_by_index = dict(zip(leadership.index, sizes))
    stable_by_index = dict(zip(leadership.index, stable))
    for index, row in leadership.iterrows():
        share_direction = row.get("shared_D_signal_direction", "")
        share_level = row.get("shared_D_signal_level", "")
        marker = "o"
        if share_direction == "lower":
            marker = "v"
        filled = bool(stable_by_index[index])
        facecolor = row["institution_color"] if filled else "none"
        edgecolor = (
            "#202020"
            if share_level == "strong"
            else ("#555555" if filled else row["institution_color"])
        )
        linewidth = 1.15 if share_level == "strong" else (0.4 if filled else 1.1)
        ax_c.scatter(
            row["shared_percent"],
            row["shared_NLO_percent"],
            s=size_by_index[index],
            marker=marker,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=0.92,
            zorder=3,
        )
        if row.get("shared_NLO_D_signal_level", ""):
            ax_c.scatter(
                row["shared_percent"],
                row["shared_NLO_percent"],
                s=size_by_index[index] + 42,
                marker="s",
                facecolor="none",
                edgecolor="#202020",
                linewidth=(
                    1.8
                    if row["shared_NLO_D_signal_level"] == "strong"
                    else 1.05
                ),
                zorder=4,
            )
    x_median = float(leadership["shared_percent"].median())
    y_median = float(leadership["shared_NLO_percent"].median())
    ax_c.axvline(x_median, color="#8A8A8A", linewidth=0.8, linestyle="--", zorder=1)
    ax_c.axhline(y_median, color="#8A8A8A", linewidth=0.8, linestyle="--", zorder=1)
    x_offsets = {
        "U-03": -4,
        "U-04": -4,
        "U-11": 4,
        "U-12": -4,
        "U-14": -4,
        "U-16": 4,
        "U-18": -4,
        "U-20": 4,
        "U-22": 4,
        "U-25": 4,
    }
    y_offsets = {
        "U-03": -7,
        "U-04": 6,
        "U-11": 4,
        "U-12": 8,
        "U-14": -4,
        "U-16": 8,
        "U-18": 7,
        "U-20": -10,
        "U-22": -6,
        "U-25": 8,
    }
    for row in leadership.itertuples(index=False):
        dx = 4 if row.shared_percent >= x_median else -4
        dy = 2 if row.shared_NLO_percent >= y_median else -2
        dx = x_offsets.get(row.institution_code, dx)
        dy = y_offsets.get(row.institution_code, dy)
        ax_c.annotate(
            row.institution_code,
            (row.shared_percent, row.shared_NLO_percent),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
            fontsize=5.8,
            color="#303030",
        )
    ax_c.set_title(
        "National sharing and corresponding-author leadership, 2025", pad=7
    )
    add_panel_label(ax_c, "(c)")
    ax_c.set_xlabel("Output shared with another Ecuadorian entity (%)")
    ax_c.set_ylabel("Shared output not led by focal institution (%)")
    ax_c.set_ylim(0, 100)
    ax_c.text(
        0.995,
        0.02,
        "Dashed lines: institutional medians",
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#666666",
    )
    tidy_axis(ax_c)

    overview_handles = [
        Line2D(
            [0],
            [0],
            color="#202020",
            linewidth=1.5,
            label="National",
        ),
        Patch(
            facecolor="#D9D9D9",
            edgecolor="#777777",
            linewidth=0.7,
            label="University IQR; whiskers = min-max",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=4.5,
            markerfacecolor="#8A8A8A",
            markeredgecolor="white",
            markeredgewidth=0.3,
            label="Sparse year (points; no box)",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=5.0,
            markerfacecolor="#666666",
            markeredgecolor="white",
            markeredgewidth=0.35,
            label="Upper robust signal (D >= 2)",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="v",
            markersize=5.0,
            markerfacecolor="#666666",
            markeredgecolor="white",
            markeredgewidth=0.35,
            label="Lower robust deviation (D <= -2)",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=5.0,
            markerfacecolor="none",
            markeredgecolor="#202020",
            markeredgewidth=1.1,
            label="Black outline: |D| >= 3",
        ),
    ]
    fig.legend(
        handles=overview_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=3,
        frameon=False,
        fontsize=7.0,
        handlelength=1.5,
        columnspacing=1.1,
        handletextpad=0.45,
    )

    if not sharing_signals.empty:
        signal_codes = sorted(
            sharing_signals["institution_code"].dropna().astype(str).unique(),
            key=lambda value: int(value.split("-")[-1]),
        )
        signal_handles = [
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=4.5,
                markerfacecolor=color_lookup.get(code, "#555555"),
                markeredgecolor="white",
                markeredgewidth=0.3,
                label=code,
            )
            for code in signal_codes
        ]
        signal_legend = fig.legend(
            handles=signal_handles,
            title="Institutions with at least one two-sided national-sharing D signal",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.004),
            ncol=min(6, len(signal_handles)),
            frameon=False,
            handlelength=0.8,
            columnspacing=0.8,
            handletextpad=0.25,
        )
        plt.setp(signal_legend.get_title(), fontsize=7.4)

    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.155,
        top=0.87,
        wspace=0.29,
        hspace=0.58,
    )
    save_figure(fig, path)
    plt.close(fig)


def plot_supplementary_top_partners(institution, min_output, path):
    if output_utils.data_only():
        return
    data = institution[
        (institution["year"] == 2025) & (institution["total_articles"] >= min_output)
    ].copy()
    data = data.sort_values(
        ["leading_partner_dependency_percent", "institution_code"],
        ascending=[True, True],
    ).reset_index(drop=True)
    configure_style()
    height = max(5.7, 0.22 * len(data) + 1.2)
    fig, ax = plt.subplots(figsize=(7.2, height))
    y = np.arange(len(data))
    ax.hlines(
        y,
        data["leading_partner_dependency_percent"],
        data["top_five_partner_union_percent"],
        color="#B9C8D8",
        linewidth=1.4,
        zorder=1,
    )
    ax.scatter(
        data["leading_partner_dependency_percent"],
        y,
        color="#555555",
        s=22,
        label="Leading partner",
        zorder=3,
    )
    ax.scatter(
        data["top_five_partner_union_percent"],
        y,
        color="#2F6DAE",
        s=22,
        label="Top-five partner union",
        zorder=3,
    )
    for idx, row in data.iterrows():
        ax.annotate(
            str(row["leading_partner_code"]),
            (row["leading_partner_dependency_percent"], idx),
            xytext=(-5, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=6.8,
            color="#444444",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(data["institution_code"])
    ax.set_xlabel("Attributed output covered by partner set (%)")
    ax.set_title("Leading-partner and top-five-partner dependence, 2025")
    ax.legend(frameon=False, loc="lower right")
    tidy_axis(ax)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def plot_supplementary_matrix(pair_public, path):
    if output_utils.data_only():
        return
    data = pair_public[
        (pair_public["year"] == 2025)
        & (pair_public["partner_type"] == "focal university")
    ].copy()
    codes = sorted(
        set(data["institution_code"]).union(set(data["partner_code"])),
        key=lambda x: int(str(x).split("-")[-1]),
    )
    matrix = data.pivot_table(
        index="institution_code",
        columns="partner_code",
        values="dependency_percent",
        aggfunc="first",
        fill_value=0.0,
    ).reindex(index=codes, columns=codes, fill_value=0.0)
    values = matrix.to_numpy(dtype=float)
    np.fill_diagonal(values, np.nan)
    configure_style()
    fig, ax = plt.subplots(figsize=(7.4, 6.5))
    cmap = LinearSegmentedColormap.from_list(
        "dependency", ["#FFFFFF", "#D8E7F5", "#75A9D5", "#245A8D"]
    )
    image = figure_style.heatmap(ax,values, cmap=cmap, aspect="equal", interpolation="nearest")
    ax.set_xticks(np.arange(len(codes)))
    ax.set_xticklabels(codes, rotation=90)
    ax.set_yticks(np.arange(len(codes)))
    ax.set_yticklabels(codes)
    ax.set_xlabel("Partner focal university")
    ax.set_ylabel("Source focal university")
    ax.set_title("Directed dependence among focal universities, 2025")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Source output shared with partner (%)")
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def write_csv(frame, path):
    if not output_utils.keep(path):
        return
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def validate_calculated_results(national, institution, audit, robust_reference_table, robust_signals):
    """Check national-sharing accounting and denominator consistency."""
    errors = []
    if not audit["national_mechanisms_equal_shared"].eq(1).all():
        errors.append("National mechanism categories do not sum to shared output.")
    if not audit["institution_mechanism_mismatches"].eq(0).all():
        errors.append("At least one institutional mechanism decomposition is inconsistent.")
    if not audit["national_partner_scopes_equal_shared"].eq(1).all():
        errors.append("National partner-scope categories do not sum to shared output.")
    if not audit["national_permissive_mechanisms_equal_shared"].eq(1).all():
        errors.append("Permissive national mechanism categories do not sum to shared output.")
    if not audit["institution_partner_scope_mismatches"].eq(0).all():
        errors.append("At least one institutional partner-scope decomposition is inconsistent.")
    for column, label in [
        ("institution_all_CA_partition_mismatches", "all output"),
        ("institution_shared_CA_partition_mismatches", "shared output"),
        ("institution_nonshared_CA_partition_mismatches", "nonshared output"),
    ]:
        if not audit[column].eq(0).all():
            errors.append(
                "Corresponding-author leadership does not partition %s."
                % label
            )
    if not national["year"].tolist() == YEARS:
        errors.append("National table does not contain exactly 2011-2025 in order.")
    if institution["institution_code"].isna().any():
        errors.append("At least one focal institution lacks a manuscript code.")
    share_signal_rows = robust_signals[
        robust_signals["metric_column"] == "shared_percent"
    ]
    if not share_signal_rows.empty and not share_signal_rows["D"].abs().ge(
        ROBUST_D_SIGNAL
    ).all():
        errors.append("A national-sharing D signal does not satisfy |D| >= 2.")
    nlo_signal_rows = robust_signals[
        robust_signals["metric_column"] == "shared_NLO_percent"
    ]
    if not nlo_signal_rows.empty and not nlo_signal_rows["D"].ge(
        ROBUST_D_SIGNAL
    ).all():
        errors.append("A shared-output NLO signal does not satisfy D >= 2.")
    strong_rows = robust_signals[robust_signals["signal_level"] == "strong"]
    if not strong_rows.empty:
        strong_ok = np.where(
            strong_rows["metric_column"].eq("shared_percent"),
            strong_rows["D"].abs().ge(ROBUST_D_STRONG),
            strong_rows["D"].ge(ROBUST_D_STRONG),
        )
        if not bool(np.all(strong_ok)):
            errors.append("A strong robust-D signal does not satisfy the D >= 3 rule.")
    if not robust_reference_table.loc[
        robust_reference_table["reference_valid"], "reference_n"
    ].ge(MIN_ROBUST_REFERENCE_N).all():
        errors.append("A robust-D reference has fewer than eight institutions.")
    if errors:
        raise ValueError('Validation failed: ' + ' '.join(errors))


def main():
    args = parse_args()
    paths = ensure_dirs(args.output_dir)
    mapping, mapping_encoding = read_mapping(args.mapping_file, args.mapping_encoding)
    input_path = Path(args.input_dir)
    if not input_path.is_dir():
        raise ValueError('Expected an annual CSV directory')
    outputs = load_and_process_directory(input_path)
    national = outputs['national'].sort_values('year').reset_index(drop=True)
    institution, pair, nonfocal_codebook = add_codes(outputs, mapping)
    institution = institution.sort_values(['year', 'institution_code']).reset_index(drop=True)
    pair = pair.sort_values(['year', 'institution_code', 'shared_articles', 'partner_code'], ascending=[True, True, False, True]).reset_index(drop=True)
    institution, robust_reference_table, robust_signals, robust_recurrence = make_robust_d_tables(institution, args.min_output, args.min_shared_ca_known)
    public_institution_columns = ['year', 'institution_code', 'total_articles', 'shared_articles', 'shared_percent', 'articles_with_other_focal', 'articles_with_nonfocal_national', 'shared_with_other_focal_only_articles', 'shared_with_nonfocal_only_articles', 'shared_with_both_partner_scopes_articles', 'distinct_author_only_articles', 'same_author_only_articles', 'both_articles', 'undetermined_articles', 'strict_vs_permissive_classification_changes', 'leading_partner_code', 'leading_partner_type', 'leading_partner_articles', 'leading_partner_dependency_percent', 'top_five_partner_union_articles', 'top_five_partner_union_percent', 'n_national_partners', 'shared_CA_known_articles', 'shared_CA_unknown_articles', 'shared_CA_coverage_percent', 'shared_CA_led_articles', 'shared_CAL_percent', 'shared_NLO_articles', 'shared_NLO_percent', 'nonshared_CA_known_articles', 'nonshared_CA_unknown_articles', 'nonshared_CA_coverage_percent', 'nonshared_CA_led_articles', 'nonshared_CAL_percent', 'nonshared_NLO_articles', 'nonshared_NLO_percent', 'shared_minus_nonshared_NLO_pp', 'all_CA_known_articles', 'all_CA_unknown_articles', 'all_CA_coverage_percent', 'all_CA_led_articles', 'all_CAL_percent', 'all_NLO_articles', 'all_NLO_percent', 'shared_D_reference_eligible', 'shared_D_reference_valid', 'shared_D_reference_n', 'shared_D_reference_median', 'shared_D_reference_q1', 'shared_D_reference_q3', 'shared_D_reference_iqr', 'shared_D_robust_scale', 'shared_D', 'shared_D_signal_direction', 'shared_D_signal_level', 'shared_NLO_D_reference_eligible', 'shared_NLO_D_reference_valid', 'shared_NLO_D_reference_n', 'shared_NLO_D_reference_median', 'shared_NLO_D_reference_q1', 'shared_NLO_D_reference_q3', 'shared_NLO_D_reference_iqr', 'shared_NLO_D_robust_scale', 'shared_NLO_D', 'shared_NLO_D_signal_direction', 'shared_NLO_D_signal_level']
    institution_public = institution[public_institution_columns].copy()
    public_pair_columns = ['year', 'institution_code', 'partner_code', 'partner_type', 'focal_total_articles', 'shared_articles', 'dependency_percent', 'distinct_author_only_articles', 'same_author_only_articles', 'both_articles', 'undetermined_articles', 'same_author_articles', 'distinct_author_strict_articles', 'distinct_author_permissive_articles']
    pair_public = pair[public_pair_columns].copy()
    write_csv(national, paths['tables'] / 'Table_S10_national_annual_sharing.csv')
    write_csv(institution_public, paths['tables'] / 'Table_S11_institution_annual_sharing.csv')
    # Table S12 publishes only the focal-university pairs of 2025 that underlie
    # Figure S8; finer partner-level rows remain private calculations.
    s12_public = pair_public[
        (pair_public['year'] == 2025)
        & (pair_public['partner_type'] == 'focal university')
    ].copy()
    write_csv(s12_public, paths['tables'] / 'Table_S12_pairwise_dependencies.csv')
    distribution = make_distribution_table(institution, args.min_output)
    reciprocal = build_reciprocal_pairs(pair_public)
    if not reciprocal.empty:
        reciprocal = reciprocal.sort_values(['year', 'a_to_b_shared_articles', 'b_to_a_shared_articles', 'institution_a', 'institution_b'], ascending=[True, False, False, True, True]).reset_index(drop=True)
    summary = make_summary(national, institution, distribution, reciprocal,
        robust_reference_table, robust_signals, robust_recurrence, args.min_output, args.min_shared_ca_known)
    output_utils.report_methodology(national_sharing_summary=summary)
    validate_calculated_results(national, institution, outputs['audit'], robust_reference_table, robust_signals)
    print(f'Calculated national-sharing sources in {paths["root"]}')


if __name__ == "__main__":
    main()
