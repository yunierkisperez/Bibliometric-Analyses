#!/usr/bin/env python3
"""Generate Figure 6 and its supporting affiliation-structure outputs.

Reads annual CSV outputs from process_json.py and keeps nonfocal affiliations
at the Scopus AF-ID entity level. Focal-university aliases are consolidated
through the supplied mapping file. The primary analysis is longitudinal for
2011-2025; the final year supplies partner-composition and IMFA–EC-CIS overlap
summaries. Annual upper and lower deviations use the robust D scale.

Use reproduce_figure_6.py for the public independent workflow, with either
article tables, private annual CSVs or calculated inputs. See USAGE.md.
"""

from __future__ import annotations

import figure_style
import output_utils
import argparse
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch



SCRIPT_VERSION = "3.0.2-20261014-national-excluded"
DEFAULT_START_YEAR = 2011
DEFAULT_END_YEAR = 2025
MIN_OUTPUT = 50
MIN_SIGNAL_ARTICLES = 10
NORMAL_IQR_CONSTANT = 1.349
MIN_D_REFERENCE_INSTITUTIONS = 8
D_ELEVATED = 2.0
D_EXTREME = 3.0

REQUIRED_SUFFIXES = [
    "institution_year_metrics.csv",
    "imfa_author_partner_institutions.csv",
    "cis_author_partner_institutions.csv",
]


from scipy.stats import spearmanr

def descriptive(values: pd.Series) -> Dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"n": 0, "median": float("nan"), "q1": float("nan"), "q3": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(len(clean)),
        "median": float(clean.median()),
        "q1": float(clean.quantile(0.25)),
        "q3": float(clean.quantile(0.75)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }

def annual_distribution_summary(annual: pd.DataFrame, d_values: pd.DataFrame) -> pd.DataFrame:
    annual = institutional_rows(annual)
    d_values = institutional_rows(d_values)
    rows = []
    for signal, measure, eligibility, value_column, denominator_column in TEMPORAL_METRICS:
        metric_key = METRIC_KEY_LOOKUP[(signal, measure)]
        for year, year_frame in annual.groupby("year"):
            eligible = year_frame[year_frame[eligibility] == 1].copy()
            values = pd.to_numeric(eligible[value_column], errors="coerce").dropna()
            denominator = pd.to_numeric(eligible[denominator_column], errors="coerce").dropna()
            flagged = d_values[
                (d_values["year"] == int(year))
                & (d_values["signal"] == signal)
                & (d_values["measure"] == measure)
            ]
            q1 = float(values.quantile(0.25)) if len(values) else float("nan")
            q3 = float(values.quantile(0.75)) if len(values) else float("nan")
            iqr = q3 - q1 if len(values) else float("nan")
            reference_evaluable = int(
                bool(len(flagged))
                and int(flagged["reference_evaluable"].max()) == 1
            )
            rows.append({
                "year": int(year),
                "signal": signal,
                "measure": measure,
                "metric_key": metric_key,
                "n_eligible_institutions": int(len(values)),
                "n_distinct_values": int(values.nunique(dropna=True)),
                "median": float(values.median()) if len(values) else float("nan"),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "minimum": float(values.min()) if len(values) else float("nan"),
                "maximum": float(values.max()) if len(values) else float("nan"),
                "median_signal_publications": float(denominator.median()) if len(denominator) else float("nan"),
                "reference_evaluable": reference_evaluable,
                "robust_scale": (
                    iqr / NORMAL_IQR_CONSTANT
                    if reference_evaluable else float("nan")
                ),
                "n_lower_deviation_D_le_minus_2": int(
                    flagged["lower_deviation_D_le_minus_2"].sum()
                ) if len(flagged) else float("nan"),
                "n_upper_elevated_D_ge_2": int(
                    flagged["upper_elevated_D_ge_2"].sum()
                ) if len(flagged) else float("nan"),
                "n_upper_extreme_D_ge_3": int(
                    flagged["upper_extreme_D_ge_3"].sum()
                ) if len(flagged) else float("nan"),
                "n_lower_extreme_D_le_minus_3": int(
                    flagged["lower_extreme_D_le_minus_3"].sum()
                ) if len(flagged) else float("nan"),
            })
    result = pd.DataFrame(rows).sort_values(
        ["year", "signal", "measure"]
    ).reset_index(drop=True)
    columns = [
        "year", "signal", "measure", "metric_key",
        "n_eligible_institutions", "n_distinct_values", "median", "q1",
        "q3", "iqr", "minimum", "maximum", "median_signal_publications",
        "reference_evaluable", "robust_scale", "n_upper_elevated_D_ge_2",
        "n_upper_extreme_D_ge_3", "n_lower_deviation_D_le_minus_2",
        "n_lower_extreme_D_le_minus_3",
    ]
    return result[columns]

def matched_endpoint_changes(
    annual: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    annual = institutional_rows(annual)
    rows = []
    for signal, measure, eligibility, value_column, denominator_column in TEMPORAL_METRICS:
        start = annual[(annual["year"] == start_year) & (annual[eligibility] == 1)][
            ["institution_code", value_column, denominator_column]
        ].copy()
        end = annual[(annual["year"] == end_year) & (annual[eligibility] == 1)][
            ["institution_code", value_column, denominator_column]
        ].copy()
        joined = start.merge(end, on="institution_code", how="inner", suffixes=("_start", "_end"))
        for _, row in joined.iterrows():
            start_value = float(row[value_column + "_start"])
            end_value = float(row[value_column + "_end"])
            rows.append({
                "row_type": "institution",
                "signal": signal,
                "measure": measure,
                "institution_code": clean_text(row["institution_code"]),
                "start_year": start_year,
                "end_year": end_year,
                "start_value": start_value,
                "end_value": end_value,
                "change": end_value - start_value,
                "start_signal_publications": safe_int(row[denominator_column + "_start"]),
                "end_signal_publications": safe_int(row[denominator_column + "_end"]),
            })
    return pd.DataFrame(rows).sort_values(["signal", "measure", "institution_code"]).reset_index(drop=True)

def matched_endpoint_summary(
    matched: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Return the four manuscript-level matched endpoint summaries."""
    matched = institutional_rows(matched)
    rows = []
    for signal, measure, _eligibility, _value_column, _denominator_column in TEMPORAL_METRICS:
        frame = matched[
            (matched["signal"] == signal) & (matched["measure"] == measure)
        ].copy()
        changes = pd.to_numeric(frame["change"], errors="coerce").dropna()
        start_values = pd.to_numeric(frame["start_value"], errors="coerce").dropna()
        end_values = pd.to_numeric(frame["end_value"], errors="coerce").dropna()
        rows.append({
            "row_type": "summary",
            "signal": signal,
            "measure": measure,
            "institution_code": "ALL_MATCHED",
            "start_year": start_year,
            "end_year": end_year,
            "n_matched_institutions": int(len(changes)),
            "median_start": float(start_values.median()) if len(start_values) else float("nan"),
            "median_end": float(end_values.median()) if len(end_values) else float("nan"),
            "median_within_institution_change": float(changes.median()) if len(changes) else float("nan"),
            "q1_within_institution_change": float(changes.quantile(0.25)) if len(changes) else float("nan"),
            "q3_within_institution_change": float(changes.quantile(0.75)) if len(changes) else float("nan"),
        })
    return pd.DataFrame(rows)

def size_association_table(annual: pd.DataFrame) -> pd.DataFrame:
    annual = institutional_rows(annual)
    rows = []
    for signal, measure, eligibility, value_column, denominator_column in TEMPORAL_METRICS:
        for year, year_frame in annual.groupby("year"):
            eligible = year_frame[year_frame[eligibility] == 1][
                [denominator_column, value_column]
            ].apply(pd.to_numeric, errors="coerce").dropna()
            rho = float("nan")
            p_value = float("nan")
            if (
                len(eligible) >= 5
                and eligible[denominator_column].nunique() > 1
                and eligible[value_column].nunique() > 1
                and spearmanr is not None
            ):
                result = spearmanr(eligible[denominator_column], eligible[value_column])
                rho = float(result.correlation)
                p_value = float(result.pvalue)
            rows.append({
                "year": int(year),
                "signal": signal,
                "measure": measure,
                "n_eligible_institutions": int(len(eligible)),
                "spearman_rho_denominator_vs_concentration": rho,
                "p_value": p_value,
            })
    for signal, eligibility, top1_column, top5_column in [
        ("IMFA", "eligible_imfa_structure", "imfa_top1_partner_share", "imfa_top5_union_share"),
        ("EC-CIS", "eligible_cis_structure", "cis_top1_partner_share", "cis_top5_union_share"),
    ]:
        eligible = annual[annual[eligibility] == 1][
            [top1_column, top5_column]
        ].apply(pd.to_numeric, errors="coerce").dropna()
        rho = float("nan")
        p_value = float("nan")
        if (
            len(eligible) >= 3
            and eligible[top1_column].nunique() > 1
            and eligible[top5_column].nunique() > 1
            and spearmanr is not None
        ):
            result = spearmanr(eligible[top1_column], eligible[top5_column])
            rho = float(result.correlation)
            p_value = float(result.pvalue)
        rows.append({
            "year": "ALL",
            "signal": signal,
            "measure": "Top-1 vs Top-5",
            "n_eligible_institutions": int(len(eligible)),
            "spearman_rho_denominator_vs_concentration": rho,
            "p_value": p_value,
        })
    return pd.DataFrame(rows)

def annual_paired_comparisons(annual: pd.DataFrame) -> pd.DataFrame:
    """Compare EC-CIS and IMFA concentration within jointly eligible institutions."""
    annual = institutional_rows(annual)
    rows = []
    for year, year_frame in annual.groupby("year"):
        eligible = year_frame[year_frame["eligible_joint_structure"] == 1].copy()
        for measure, imfa_column, cis_column in [
            ("Top-1", "imfa_top1_partner_share", "cis_top1_partner_share"),
            ("Top-5", "imfa_top5_union_share", "cis_top5_union_share"),
        ]:
            paired = eligible[[imfa_column, cis_column]].apply(pd.to_numeric, errors="coerce").dropna()
            differences = paired[cis_column] - paired[imfa_column]
            rows.append({
                "year": int(year),
                "measure": measure,
                "n_jointly_eligible_institutions": int(len(paired)),
                "n_EC_CIS_greater_than_IMFA": int((differences > 0).sum()),
                "n_equal": int((differences == 0).sum()),
                "n_EC_CIS_lower_than_IMFA": int((differences < 0).sum()),
                "median_EC_CIS_minus_IMFA": (
                    float(differences.median()) if len(differences) else float("nan")
                ),
                "q1_EC_CIS_minus_IMFA": (
                    float(differences.quantile(0.25)) if len(differences) else float("nan")
                ),
                "q3_EC_CIS_minus_IMFA": (
                    float(differences.quantile(0.75)) if len(differences) else float("nan")
                ),
            })
    return pd.DataFrame(rows).sort_values(["year", "measure"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate longitudinal Figure 6 and supporting affiliation-structure tables."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing annual CSV outputs")
    parser.add_argument("--mapping-file", required=True, help="Institution mapping with codes and colors")
    parser.add_argument("--output-dir", required=True, help="Directory for figures and derived tables")
    parser.add_argument(
        "--mapping-encoding",
        default="latin1",
        help="Encoding of mapping file (default: latin1)",
    )
    parser.add_argument(
        "--start-year", type=int, default=DEFAULT_START_YEAR,
        help="First analysis year (default: 2011)",
    )
    parser.add_argument(
        "--end-year", type=int, default=DEFAULT_END_YEAR,
        help="Last analysis year and detailed cross-section (default: 2025)",
    )
    parser.add_argument(
        "--include-internal",
        action="store_true",
        help="Also write identifiable/internal audit tables (default: public outputs only)",
    )
    parser.add_argument(
        "--min-d-reference-institutions", type=int,
        default=MIN_D_REFERENCE_INSTITUTIONS,
        help="Minimum eligible institutions for a valid annual D reference",
    )
    parser.add_argument(
        "--d-elevated", type=float, default=D_ELEVATED,
        help="Upper robust-D threshold for an elevated signal",
    )
    parser.add_argument(
        "--d-extreme", type=float, default=D_EXTREME,
        help="Absolute robust-D threshold for an extreme signal",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster output resolution")
    return parser.parse_args()


def clean_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def parse_items(value) -> Set[str]:
    text = clean_text(value)
    if not text:
        return set()
    return {item.strip() for item in text.split(";") if item.strip()}


def join_items(values: Iterable[str]) -> str:
    return ";".join(sorted({clean_text(value) for value in values if clean_text(value)}))


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return float("nan")
    return float(numerator) / float(denominator)


def safe_int(value) -> int:
    """Convert numeric-like values to int while treating missing values as zero."""
    try:
        if value is None or pd.isna(value):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def read_mapping(path: Path, encoding: str) -> pd.DataFrame:
    encodings = [encoding]
    if encoding.lower().replace("-", "") not in {"utf8", "utf8sig"}:
        encodings.extend(["utf-8-sig", "utf-8"])
    last_error = None
    mapping = None
    used_encoding = None
    for candidate in encodings:
        try:
            mapping = pd.read_csv(path, encoding=candidate)
            used_encoding = candidate
            break
        except UnicodeDecodeError as error:
            last_error = error
    if mapping is None:
        raise last_error  # type: ignore[misc]

    mapping.columns = [
        str(column).lstrip("\ufeff").replace("ï»¿", "")
        for column in mapping.columns
    ]

    required = {"institution_id", "id-paper", "color_hex"}
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError("Mapping file lacks columns: " + ", ".join(sorted(missing)))
    if "name" not in mapping.columns:
        mapping["name"] = mapping["institution_id"]
    for column in ["institution_id", "id-paper", "color_hex", "name"]:
        mapping[column] = mapping[column].map(clean_text)
    mapping = mapping.drop_duplicates("institution_id", keep="first").copy()
    mapping.attrs["encoding"] = used_encoding
    return mapping


def locate_inputs(input_dir: Path, year: int) -> Dict[str, Path]:
    paths = {}
    for suffix in REQUIRED_SUFFIXES:
        candidate = input_dir / (str(year) + "_" + suffix)
        if not candidate.exists():
            matches = sorted(input_dir.glob("*/" + str(year) + "_" + suffix))
            if len(matches) == 1:
                candidate = matches[0]
        if not candidate.exists():
            raise FileNotFoundError("Required annual input not found: " + str(candidate))
        paths[suffix] = candidate
    return paths




def save_png_via_temporary(fig, path: Path, dpi: int) -> None:
    """Avoid partial PNG writes on network-backed workspaces."""
    descriptor, temporary_name = tempfile.mkstemp(suffix=".png")
    os.close(descriptor)
    try:
        fig.savefig(temporary_name, dpi=dpi, bbox_inches="tight")
        shutil.copyfile(temporary_name, str(path))
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def add_partner_identity(frame: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    focal_name = dict(zip(mapping["institution_id"], mapping["name"]))
    focal_code = dict(zip(mapping["institution_id"], mapping["id-paper"]))

    keys = []
    names = []
    countries = []
    scopes = []
    afids = []
    codes = []
    for _, row in result.iterrows():
        focal_id = clean_text(row.get("partner_focal_institution_id", ""))
        partner_afid = clean_text(row.get("partner_afid", ""))
        if focal_id:
            keys.append("FOCAL:" + focal_id)
            names.append(focal_name.get(focal_id, clean_text(row.get("partner_focal_institution_name", ""))))
            countries.append("Ecuador")
            scopes.append("focal")
            codes.append(focal_code.get(focal_id, focal_id))
        else:
            keys.append("AFID:" + partner_afid)
            names.append(clean_text(row.get("partner_affiliation_name", "")))
            countries.append(clean_text(row.get("partner_affiliation_country", "")))
            scopes.append("nonfocal")
            codes.append("")
        afids.append(partner_afid)

    result["partner_key"] = keys
    result["partner_name_internal"] = names
    result["partner_country"] = countries
    result["partner_scope"] = scopes
    result["partner_afid_clean"] = afids
    result["partner_code"] = codes
    return result


def choose_mode(values: Iterable[str]) -> str:
    cleaned = [clean_text(value) for value in values if clean_text(value)]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def build_structures(
    frame: pd.DataFrame,
    eid_column: str,
    mapping: pd.DataFrame,
) -> Dict[str, object]:
    frame = add_partner_identity(frame, mapping)
    focal_ids = set(mapping["institution_id"].map(clean_text))
    frame = frame[frame["institution_id"].map(clean_text).isin(focal_ids)].copy()
    mapping_name = dict(zip(mapping["institution_id"], mapping["name"]))

    partner_store: MutableMapping[Tuple[str, str], Dict[str, object]] = {}
    article_store: MutableMapping[Tuple[str, str], Dict[str, Set[str]]] = {}
    author_partner_store: MutableMapping[Tuple[str, str, str], Dict[str, Set[str]]] = {}

    for _, row in frame.iterrows():
        institution_id = clean_text(row.get("institution_id", ""))
        author_id = clean_text(row.get("author_id", ""))
        partner_key = clean_text(row.get("partner_key", ""))
        if not institution_id or not partner_key:
            continue
        eids = parse_items(row.get(eid_column, ""))
        partner_name = clean_text(row.get("partner_name_internal", ""))
        country = clean_text(row.get("partner_country", ""))
        scope = clean_text(row.get("partner_scope", ""))
        afid = clean_text(row.get("partner_afid_clean", ""))

        key = (institution_id, partner_key)
        if key not in partner_store:
            partner_store[key] = {
                "authors": set(),
                "eids": set(),
                "names": [],
                "countries": [],
                "scopes": [],
                "afids": set(),
            }
        record = partner_store[key]
        record["authors"].add(author_id)  # type: ignore[union-attr]
        record["eids"].update(eids)  # type: ignore[union-attr]
        record["names"].append(partner_name)  # type: ignore[union-attr]
        record["countries"].append(country)  # type: ignore[union-attr]
        record["scopes"].append(scope)  # type: ignore[union-attr]
        record["afids"].add(afid)  # type: ignore[union-attr]

        author_key = (institution_id, author_id, partner_key)
        if author_key not in author_partner_store:
            author_partner_store[author_key] = {"eids": set()}
        author_partner_store[author_key]["eids"].update(eids)

        for eid in eids:
            article_key = (institution_id, eid)
            if article_key not in article_store:
                article_store[article_key] = {
                    "partners": set(),
                    "countries": set(),
                    "scopes": set(),
                    "authors": set(),
                }
            article_store[article_key]["partners"].add(partner_key)
            article_store[article_key]["countries"].add(country if country else "UNKNOWN")
            article_store[article_key]["scopes"].add(scope)
            article_store[article_key]["authors"].add(author_id)

    rows = []
    for (institution_id, partner_key), record in partner_store.items():
        countries = {clean_text(value) for value in record["countries"] if clean_text(value)}
        scopes = {clean_text(value) for value in record["scopes"] if clean_text(value)}
        rows.append(
            {
                "institution_id": institution_id,
                "institution_name": mapping_name.get(institution_id, institution_id),
                "partner_key": partner_key,
                "partner_name_internal": choose_mode(record["names"]),
                "partner_country": choose_mode(record["countries"]),
                "partner_scope": choose_mode(record["scopes"]),
                "partner_afids_internal": join_items(record["afids"]),
                "country_conflict": int(len(countries) > 1),
                "scope_conflict": int(len(scopes) > 1),
                "n_authors": len({value for value in record["authors"] if value}),
                "n_articles": len(record["eids"]),
                "article_eids_internal": join_items(record["eids"]),
            }
        )
    partner_summary = pd.DataFrame(rows)
    if partner_summary.empty:
        partner_summary = pd.DataFrame(
            columns=[
                "institution_id", "institution_name", "partner_key",
                "partner_name_internal", "partner_country", "partner_scope",
                "partner_afids_internal", "country_conflict", "scope_conflict",
                "n_authors", "n_articles", "article_eids_internal",
            ]
        )
    partner_summary = partner_summary.sort_values(
        ["institution_id", "n_articles", "partner_key"], ascending=[True, False, True]
    ).reset_index(drop=True)
    partner_summary["rank"] = partner_summary.groupby("institution_id").cumcount() + 1

    return {
        "detail": frame,
        "partner_summary": partner_summary,
        "article_store": article_store,
        "author_partner_store": author_partner_store,
    }


def assign_partner_codes(
    imfa_partner: pd.DataFrame,
    cis_partner: pd.DataFrame,
    mapping: pd.DataFrame,
) -> Dict[str, str]:
    code_lookup = {}
    focal_codes = dict(zip(mapping["institution_id"], mapping["id-paper"]))
    all_keys = sorted(set(imfa_partner["partner_key"]).union(set(cis_partner["partner_key"])))
    nonfocal = []
    for key in all_keys:
        if key.startswith("FOCAL:"):
            focal_id = key.split(":", 1)[1]
            code_lookup[key] = focal_codes.get(focal_id, focal_id)
        else:
            nonfocal.append(key)
    for index, key in enumerate(sorted(nonfocal), start=1):
        code_lookup[key] = "P-" + str(index).zfill(4)
    return code_lookup


def classify_country(countries: Set[str]) -> str:
    values = {clean_text(value) for value in countries if clean_text(value)}
    has_unknown = "UNKNOWN" in values or not values
    values.discard("UNKNOWN")
    domestic = "Ecuador" in values
    foreign = any(value != "Ecuador" for value in values)
    if domestic and foreign:
        return "mixed"
    if has_unknown:
        return "unresolved"
    if domestic:
        return "domestic_only"
    if foreign:
        return "foreign_only"
    return "unresolved"


def classify_scope(scopes: Set[str]) -> str:
    values = {clean_text(value) for value in scopes if clean_text(value)}
    focal = "focal" in values
    nonfocal = "nonfocal" in values
    if focal and nonfocal:
        return "mixed"
    if focal:
        return "focal_only"
    return "nonfocal_only"


def article_composition(article_store: Mapping[Tuple[str, str], Dict[str, Set[str]]]) -> pd.DataFrame:
    counters: MutableMapping[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (institution_id, _eid), record in article_store.items():
        country_class = classify_country(record["countries"])
        scope_class = classify_scope(record["scopes"])
        counters[institution_id]["n_resolved_articles"] += 1
        counters[institution_id]["country_" + country_class] += 1
        counters[institution_id]["scope_" + scope_class] += 1
        if "Ecuador" in record["countries"]:
            counters[institution_id]["n_any_domestic"] += 1
        if any(value not in {"Ecuador", "UNKNOWN", ""} for value in record["countries"]):
            counters[institution_id]["n_any_foreign"] += 1

    rows = []
    for institution_id, count in counters.items():
        total = count["n_resolved_articles"]
        row = {"institution_id": institution_id, "n_resolved_articles": total}
        for key in [
            "country_domestic_only", "country_foreign_only", "country_mixed",
            "country_unresolved", "scope_focal_only", "scope_nonfocal_only",
            "scope_mixed", "n_any_domestic", "n_any_foreign",
        ]:
            row[key] = count.get(key, 0)
            row[key + "_share"] = safe_divide(row[key], total)
        rows.append(row)
    return pd.DataFrame(rows)


def partner_set_by_institution(partner_summary: pd.DataFrame) -> Dict[str, Set[str]]:
    result = {}
    for institution_id, group in partner_summary.groupby("institution_id"):
        result[clean_text(institution_id)] = set(group["partner_key"].map(clean_text))
    return result


def author_partner_set_by_institution(
    store: Mapping[Tuple[str, str, str], Dict[str, Set[str]]]
) -> Dict[str, Set[Tuple[str, str]]]:
    result: MutableMapping[str, Set[Tuple[str, str]]] = defaultdict(set)
    for institution_id, author_id, partner_key in store:
        result[institution_id].add((author_id, partner_key))
    return dict(result)


def set_diagnostics(left: Set, right: Set, prefix: str) -> Dict[str, float]:
    intersection = left.intersection(right)
    union = left.union(right)
    return {
        prefix + "_left": len(left),
        prefix + "_right": len(right),
        prefix + "_intersection": len(intersection),
        prefix + "_union": len(union),
        prefix + "_jaccard": safe_divide(len(intersection), len(union)),
        prefix + "_left_overlap_share": safe_divide(len(intersection), len(left)),
        prefix + "_right_overlap_share": safe_divide(len(intersection), len(right)),
    }


def top_partner_metrics(
    institution_id: str,
    partner_summary: pd.DataFrame,
    denominator: int,
) -> Dict[str, object]:
    group = partner_summary[partner_summary["institution_id"] == institution_id].copy()
    if group.empty:
        return {
            "n_partner_entities": 0,
            "top1_partner_key": "",
            "top1_partner_keys": "",
            "top1_tie_count": 0,
            "top1_partner_count": 0,
            "top1_partner_share": float("nan"),
            "top5_partner_keys": "",
            "top5_union_count": 0,
            "top5_union_share": float("nan"),
        }
    group = group.sort_values(["n_articles", "partner_key"], ascending=[False, True])
    top1 = group.iloc[0]
    top1_count = int(top1["n_articles"])
    top1_ties = group[group["n_articles"] == top1_count]
    top5 = group.head(5)
    union_eids: Set[str] = set()
    for value in top5["article_eids_internal"]:
        union_eids.update(parse_items(value))
    return {
        "n_partner_entities": len(group),
        "top1_partner_key": clean_text(top1["partner_key"]),
        "top1_partner_keys": join_items(top1_ties["partner_key"]),
        "top1_tie_count": len(top1_ties),
        "top1_partner_count": top1_count,
        "top1_partner_share": safe_divide(top1_count, denominator),
        "top5_partner_keys": join_items(top5["partner_key"]),
        "top5_union_count": len(union_eids),
        "top5_union_share": safe_divide(len(union_eids), denominator),
    }


def cis_author_recurrence(
    store: Mapping[Tuple[str, str, str], Dict[str, Set[str]]]
) -> pd.DataFrame:
    per_author: MutableMapping[Tuple[str, str], Dict[str, object]] = defaultdict(
        lambda: {"partners": set(), "eids": set(), "partner_eids": defaultdict(set)}
    )
    for (institution_id, author_id, partner_key), record in store.items():
        key = (institution_id, author_id)
        per_author[key]["partners"].add(partner_key)
        per_author[key]["eids"].update(record["eids"])
        per_author[key]["partner_eids"][partner_key].update(record["eids"])

    rows = []
    for (institution_id, author_id), record in per_author.items():
        partner_counts = [len(values) for values in record["partner_eids"].values()]
        total_external = len(record["eids"])
        top_count = max(partner_counts) if partner_counts else 0
        rows.append(
            {
                "institution_id": institution_id,
                "author_id_internal": author_id,
                "n_partner_entities": len(record["partners"]),
                "n_external_articles": total_external,
                "top_partner_external_articles": top_count,
                "top_partner_external_share": safe_divide(top_count, total_external),
            }
        )
    author_frame = pd.DataFrame(rows)
    summary_rows = []
    if author_frame.empty:
        return author_frame, pd.DataFrame()
    for institution_id, group in author_frame.groupby("institution_id"):
        recurrent_eligible = group[group["n_external_articles"] >= 2].copy()
        summary_rows.append(
            {
                "institution_id": institution_id,
                "n_cis_authors_with_external_affiliations": len(group),
                "median_partner_entities_per_cis_author": float(group["n_partner_entities"].median()),
                "iqr_partner_entities_per_cis_author": float(
                    group["n_partner_entities"].quantile(0.75)
                    - group["n_partner_entities"].quantile(0.25)
                ),
                "single_partner_cis_author_share": float((group["n_partner_entities"] == 1).mean()),
                "n_ec_cis_authors_ge2_nonfocal_with_resolved_partner": int(
                    len(recurrent_eligible)
                ),
                "n_ec_cis_authors_single_recurrent_partner": int(
                    (recurrent_eligible["n_partner_entities"] == 1).sum()
                ),
                "ec_cis_single_recurrent_partner_share": (
                    float((recurrent_eligible["n_partner_entities"] == 1).mean())
                    if len(recurrent_eligible) else float("nan")
                ),
                "median_author_top_partner_share": float(group["top_partner_external_share"].median()),
            }
        )
    return author_frame, pd.DataFrame(summary_rows)


def institutional_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return universities only; the National aggregate is never a reference member."""
    keep = pd.Series(True, index=frame.index)
    for column in ("institution_code", "institution_id"):
        if column in frame.columns:
            keep &= ~frame[column].astype("string").str.strip().str.casefold().eq("national").fillna(False)
    return frame.loc[keep].copy()




def build_summary(
    metrics: pd.DataFrame,
    mapping: pd.DataFrame,
    imfa: Dict[str, object],
    cis: Dict[str, object],
    partner_codes: Mapping[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping_rows = mapping[["institution_id", "name", "id-paper", "color_hex"]].copy()
    mapping_rows = mapping_rows.rename(
        columns={"name": "mapped_name", "id-paper": "institution_code", "color_hex": "institution_color"}
    )
    metrics = metrics.copy()
    metrics["institution_id"] = metrics["institution_id"].map(clean_text)
    metrics = metrics[metrics["institution_id"].isin(mapping_rows["institution_id"])].copy()
    metrics = metrics.merge(mapping_rows, on="institution_id", how="right")

    imfa_partner = imfa["partner_summary"].copy()
    cis_partner = cis["partner_summary"].copy()
    imfa_partner["partner_code"] = imfa_partner["partner_key"].map(partner_codes)
    cis_partner["partner_code"] = cis_partner["partner_key"].map(partner_codes)

    imfa_composition = article_composition(imfa["article_store"])
    cis_composition = article_composition(cis["article_store"])
    imfa_composition = imfa_composition.add_prefix("imfa_").rename(columns={"imfa_institution_id": "institution_id"})
    cis_composition = cis_composition.add_prefix("cis_").rename(columns={"cis_institution_id": "institution_id"})

    imfa_partner_sets = partner_set_by_institution(imfa_partner)
    cis_partner_sets = partner_set_by_institution(cis_partner)
    imfa_link_sets = author_partner_set_by_institution(imfa["author_partner_store"])
    cis_link_sets = author_partner_set_by_institution(cis["author_partner_store"])
    cis_author_detail, cis_author_summary = cis_author_recurrence(cis["author_partner_store"])

    rows = []
    for _, base in metrics.iterrows():
        institution_id = clean_text(base["institution_id"])
        n_output = safe_int(base.get("N_articles_attributed", 0))
        n_imfa = safe_int(base.get("N_articles_IMFA", 0))
        n_cis = safe_int(base.get("N_articles_CIS", 0))
        cis_article_eids = {
            eid
            for (inst, eid), _record in cis["article_store"].items()
            if inst == institution_id
        }
        imfa_metrics = top_partner_metrics(institution_id, imfa_partner, n_imfa)
        cis_metrics = top_partner_metrics(institution_id, cis_partner, len(cis_article_eids))
        partner_diag = set_diagnostics(
            imfa_partner_sets.get(institution_id, set()),
            cis_partner_sets.get(institution_id, set()),
            "partner",
        )
        link_diag = set_diagnostics(
            imfa_link_sets.get(institution_id, set()),
            cis_link_sets.get(institution_id, set()),
            "author_partner",
        )
        record = {
            "institution_id": institution_id,
            "institution_code": clean_text(base.get("institution_code", "")),
            "institution_name_internal": clean_text(base.get("mapped_name", "")),
            "institution_color": clean_text(base.get("institution_color", "")),
            "n_articles_attributed": n_output,
            "n_imfa_articles": n_imfa,
            "imfa_rate": safe_divide(n_imfa, n_output),
            "n_cis_articles": n_cis,
            "cis_rate": safe_divide(n_cis, n_output),
            "n_cis_external_articles": len(cis_article_eids),
            "n_imfa_authors": len({key[1] for key in imfa["author_partner_store"] if key[0] == institution_id}),
            "n_cis_authors": len({key[1] for key in cis["author_partner_store"] if key[0] == institution_id}),
        }
        for key, value in imfa_metrics.items():
            record["imfa_" + key] = value
        for key, value in cis_metrics.items():
            record["cis_" + key] = value
        record.update(partner_diag)
        record.update(link_diag)
        record["imfa_top1_partner_code"] = partner_codes.get(clean_text(record["imfa_top1_partner_key"]), "")
        record["cis_top1_partner_code"] = partner_codes.get(clean_text(record["cis_top1_partner_key"]), "")
        record["imfa_top1_partner_codes"] = join_items(
            partner_codes.get(key, "") for key in parse_items(record["imfa_top1_partner_keys"])
        )
        record["cis_top1_partner_codes"] = join_items(
            partner_codes.get(key, "") for key in parse_items(record["cis_top1_partner_keys"])
        )
        rows.append(record)

    summary = pd.DataFrame(rows)
    summary = summary.merge(imfa_composition, on="institution_id", how="left")
    summary = summary.merge(cis_composition, on="institution_id", how="left")
    summary = summary.merge(cis_author_summary, on="institution_id", how="left")

    for prefix in ["imfa", "cis"]:
        resolved = prefix + "_n_resolved_articles"
        if resolved in summary:
            if prefix == "imfa":
                summary[prefix + "_partner_linkage_coverage"] = summary[resolved] / summary["n_imfa_articles"].replace(0, np.nan)
            else:
                summary[prefix + "_partner_linkage_coverage"] = 1.0

    summary["eligible_joint_structure"] = (
        (summary["n_articles_attributed"] >= MIN_OUTPUT)
        & (summary["n_imfa_articles"] >= MIN_SIGNAL_ARTICLES)
        & (summary["n_cis_external_articles"] >= MIN_SIGNAL_ARTICLES)
    ).astype(int)
    summary["eligible_imfa_structure"] = (
        (summary["n_articles_attributed"] >= MIN_OUTPUT)
        & (summary["n_imfa_articles"] >= MIN_SIGNAL_ARTICLES)
    ).astype(int)
    summary["eligible_cis_structure"] = (
        (summary["n_articles_attributed"] >= MIN_OUTPUT)
        & (summary["n_cis_external_articles"] >= MIN_SIGNAL_ARTICLES)
    ).astype(int)
    # National keeps its raw descriptive concentrations but is not eligible as a university.
    national = summary["institution_code"].astype("string").str.strip().str.casefold().eq("national")
    summary.loc[national, ["eligible_joint_structure", "eligible_imfa_structure", "eligible_cis_structure"]] = 0
    summary = summary.sort_values("institution_code").reset_index(drop=True)
    return summary, imfa_partner, cis_partner, cis_author_detail


def make_public_summary(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "year",
        "institution_code", "n_articles_attributed", "n_imfa_articles", "imfa_rate",
        "n_imfa_authors", "imfa_n_partner_entities", "imfa_top1_partner_code",
        "imfa_top1_partner_codes", "imfa_top1_tie_count", "imfa_top1_partner_count",
        "imfa_top1_partner_share", "imfa_top5_union_count",
        "imfa_top5_union_share", "n_cis_articles", "cis_rate", "n_cis_authors",
        "n_cis_external_articles", "cis_n_partner_entities", "cis_top1_partner_code",
        "cis_top1_partner_codes", "cis_top1_tie_count", "cis_top1_partner_count",
        "cis_top1_partner_share", "cis_top5_union_count",
        "cis_top5_union_share", "partner_intersection", "partner_union", "partner_jaccard",
        "author_partner_intersection", "author_partner_union", "author_partner_jaccard",
        "median_partner_entities_per_cis_author",
        "imfa_partner_linkage_coverage", "eligible_imfa_structure",
        "eligible_cis_structure", "eligible_joint_structure",
        "n_ec_cis_authors_ge2_nonfocal_with_resolved_partner",
        "n_ec_cis_authors_single_recurrent_partner",
        "ec_cis_single_recurrent_partner_share",
    ]
    available = [column for column in columns if column in summary.columns]
    return summary[available].copy()


def public_ec_cis_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Use explicit EC-CIS and observed-nonfocal names in public files."""
    result = frame.copy()
    rename = {}
    for column in result.columns:
        original_name = str(column)
        if re.search(r"(^|_)ec_cis(?=_|$)", original_name, flags=re.IGNORECASE):
            public_name = original_name
        else:
            public_name = re.sub(
                r"(^|_)cis(?=_|$)", r"\1ec_cis", original_name,
                flags=re.IGNORECASE,
            )
        rename[column] = public_name
    result = result.rename(columns=rename)
    result = result.rename(columns={
        "n_ec_cis_external_articles": (
            "n_ec_cis_observed_nonfocal_publications"
        ),
        "n_external_articles_with_partner": (
            "n_observed_nonfocal_publications_with_partner"
        ),
        "n_author_external_articles": (
            "n_author_observed_nonfocal_publications"
        ),
    })
    if "signal" in result.columns:
        result["signal"] = result["signal"].replace({"CIS": "EC-CIS"})
    return result


TEMPORAL_METRICS = [
    ("IMFA", "Top-1", "eligible_imfa_structure", "imfa_top1_partner_share", "n_imfa_articles"),
    ("EC-CIS", "Top-1", "eligible_cis_structure", "cis_top1_partner_share", "n_cis_external_articles"),
    ("IMFA", "Top-5", "eligible_imfa_structure", "imfa_top5_union_share", "n_imfa_articles"),
    ("EC-CIS", "Top-5", "eligible_cis_structure", "cis_top5_union_share", "n_cis_external_articles"),
]

METRIC_KEY_LOOKUP = {
    ("IMFA", "Top-1"): "imfa_top1",
    ("EC-CIS", "Top-1"): "ec_cis_top1",
    ("IMFA", "Top-5"): "imfa_top5",
    ("EC-CIS", "Top-5"): "ec_cis_top5",
}


def robust_d_signals(
    annual: pd.DataFrame,
    min_reference_institutions: int,
    elevated_threshold: float,
    extreme_threshold: float,
) -> pd.DataFrame:
    """Calculate annual robust-D values for every eligible institution."""
    annual = institutional_rows(annual)
    rows = []
    for signal, measure, eligibility, value_column, denominator_column in TEMPORAL_METRICS:
        metric_key = METRIC_KEY_LOOKUP[(signal, measure)]
        for year, year_frame in annual.groupby("year"):
            eligible = year_frame[year_frame[eligibility] == 1].copy()
            values = pd.to_numeric(eligible[value_column], errors="coerce").dropna()
            median = float(values.median()) if len(values) else float("nan")
            q1 = float(values.quantile(0.25)) if len(values) else float("nan")
            q3 = float(values.quantile(0.75)) if len(values) else float("nan")
            iqr = q3 - q1 if len(values) else float("nan")
            reference_evaluable = bool(
                len(values) >= min_reference_institutions
                and values.nunique(dropna=True) >= 2
                and np.isfinite(iqr)
                and iqr > 0
            )
            robust_scale = (
                iqr / NORMAL_IQR_CONSTANT
                if reference_evaluable else float("nan")
            )
            for _, row in eligible.iterrows():
                if pd.isna(row[value_column]):
                    continue
                value = float(row[value_column])
                d_value = (
                    (value - median) / robust_scale
                    if reference_evaluable else float("nan")
                )
                upper_elevated = int(
                    reference_evaluable and d_value >= elevated_threshold
                )
                upper_extreme = int(
                    reference_evaluable and d_value >= extreme_threshold
                )
                lower_deviation = int(
                    reference_evaluable and d_value <= -elevated_threshold
                )
                lower_extreme = int(
                    reference_evaluable and d_value <= -extreme_threshold
                )
                rows.append({
                    "year": int(year),
                    "institution_code": clean_text(row["institution_code"]),
                    "signal": signal,
                    "measure": measure,
                    "metric_key": metric_key,
                    "value": value,
                    "signal_publications": safe_int(row[denominator_column]),
                    "n_reference_institutions": int(len(values)),
                    "annual_median": median,
                    "annual_q1": q1,
                    "annual_q3": q3,
                    "annual_iqr": iqr,
                    "robust_scale": robust_scale,
                    "reference_evaluable": int(reference_evaluable),
                    "D": d_value,
                    "upper_elevated_D_ge_2": upper_elevated,
                    "upper_extreme_D_ge_3": upper_extreme,
                    "lower_deviation_D_le_minus_2": lower_deviation,
                    "lower_extreme_D_le_minus_3": lower_extreme,
                })
    columns = [
        "year", "institution_code", "signal", "measure", "metric_key",
        "value", "signal_publications", "n_reference_institutions",
        "annual_median", "annual_q1", "annual_q3", "annual_iqr",
        "robust_scale", "reference_evaluable", "D",
        "upper_elevated_D_ge_2", "upper_extreme_D_ge_3",
        "lower_deviation_D_le_minus_2", "lower_extreme_D_le_minus_3",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["year", "signal", "measure", "institution_code"]
    ).reset_index(drop=True)


def add_d_columns(annual: pd.DataFrame, d_values: pd.DataFrame) -> pd.DataFrame:
    """Attach the four robust-D profiles to the institution-year table."""
    d_values = institutional_rows(d_values)
    result = annual.copy()
    for signal, measure, _eligibility, _value_column, _denominator_column in TEMPORAL_METRICS:
        metric_key = METRIC_KEY_LOOKUP[(signal, measure)]
        subset = d_values[d_values["metric_key"] == metric_key][[
            "year", "institution_code", "D", "reference_evaluable",
            "upper_elevated_D_ge_2", "upper_extreme_D_ge_3",
            "lower_deviation_D_le_minus_2", "lower_extreme_D_le_minus_3",
        ]].copy()
        subset = subset.rename(columns={
            "D": metric_key + "_D",
            "reference_evaluable": metric_key + "_D_evaluable",
            "upper_elevated_D_ge_2": metric_key + "_D_ge_2",
            "upper_extreme_D_ge_3": metric_key + "_D_ge_3",
            "lower_deviation_D_le_minus_2": metric_key + "_D_le_minus_2",
            "lower_extreme_D_le_minus_3": metric_key + "_D_le_minus_3",
        })
        result = result.merge(
            subset, on=["year", "institution_code"], how="left"
        )
    return result


def robust_d_recurrence_table(d_values: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated robust-D signals over evaluable annual references."""
    d_values = institutional_rows(d_values)
    columns = [
        "institution_code", "signal", "measure", "metric_key",
        "n_evaluable_years", "n_upper_elevated_years",
        "n_upper_extreme_years", "n_lower_deviation_years",
        "n_lower_extreme_years", "maximum_D", "minimum_D",
        "upper_elevated_years", "upper_extreme_years",
        "lower_deviation_years",
    ]
    evaluable = d_values[d_values["reference_evaluable"] == 1].copy()
    if evaluable.empty:
        return pd.DataFrame(columns=columns)
    evaluable = evaluable.sort_values(
        ["institution_code", "metric_key", "year"]
    ).reset_index(drop=True)
    grouped = evaluable.groupby(
        ["institution_code", "signal", "measure", "metric_key"],
        as_index=False,
    ).agg(
        n_evaluable_years=("year", "nunique"),
        n_upper_elevated_years=("upper_elevated_D_ge_2", "sum"),
        n_upper_extreme_years=("upper_extreme_D_ge_3", "sum"),
        n_lower_deviation_years=("lower_deviation_D_le_minus_2", "sum"),
        n_lower_extreme_years=("lower_extreme_D_le_minus_3", "sum"),
        maximum_D=("D", "max"),
        minimum_D=("D", "min"),
    )
    year_rows = []
    for (institution_code, metric_key), frame in evaluable.groupby(
        ["institution_code", "metric_key"]
    ):
        year_rows.append({
            "institution_code": institution_code,
            "metric_key": metric_key,
            "upper_elevated_years": ";".join(
                str(int(year)) for year in frame.loc[
                    frame["upper_elevated_D_ge_2"] == 1, "year"
                ].tolist()
            ),
            "upper_extreme_years": ";".join(
                str(int(year)) for year in frame.loc[
                    frame["upper_extreme_D_ge_3"] == 1, "year"
                ].tolist()
            ),
            "lower_deviation_years": ";".join(
                str(int(year)) for year in frame.loc[
                    frame["lower_deviation_D_le_minus_2"] == 1, "year"
                ].tolist()
            ),
        })
    grouped = grouped.merge(
        pd.DataFrame(year_rows),
        on=["institution_code", "metric_key"],
        how="left",
    )
    return grouped.sort_values(
        ["metric_key", "n_upper_elevated_years", "n_upper_extreme_years", "institution_code"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)[columns]


def _boxplot_panel(
    ax,
    annual: pd.DataFrame,
    d_values: pd.DataFrame,
    years: Sequence[int],
    signal: str,
    measure: str,
    eligibility: str,
    value_column: str,
    title: str,
    show_year_labels: bool,
) -> None:
    annual = institutional_rows(annual)
    d_values = institutional_rows(d_values)
    positions = list(range(1, len(years) + 1))
    for position, year in zip(positions, years):
        frame = annual[(annual["year"] == year) & (annual[eligibility] == 1)].copy()
        values = pd.to_numeric(frame[value_column], errors="coerce") * 100.0
        frame = frame.loc[values.notna()].copy()
        values = values.dropna()
        n_value = len(values)
        ax.text(position, 1.4, str(n_value), ha="center", va="bottom", fontsize=5.0, color="#777777")
        if n_value >= 5:
            box = ax.boxplot(
                [values.to_numpy()],
                positions=[position],
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                whis=(0, 100),
            )
            for patch in box["boxes"]:
                patch.set_facecolor("#E6E6E6")
                patch.set_edgecolor("#808080")
                patch.set_linewidth(0.65)
            for item in box["whiskers"] + box["caps"]:
                item.set_color("#808080")
                item.set_linewidth(0.65)
            for item in box["medians"]:
                item.set_color("#202020")
                item.set_linewidth(1.0)

            year_signals = d_values[
                (d_values["year"] == year)
                & (d_values["signal"] == signal)
                & (d_values["measure"] == measure)
                & (d_values["reference_evaluable"] == 1)
                & (
                    (d_values["upper_elevated_D_ge_2"] == 1)
                    | (d_values["lower_deviation_D_le_minus_2"] == 1)
                )
            ]
            for _, signal_row in year_signals.iterrows():
                institution = clean_text(signal_row["institution_code"])
                match = frame[frame["institution_code"] == institution]
                color = clean_text(match.iloc[0]["institution_color"]) if len(match) else "#555555"
                code_match = re.search(r"(\d+)$", institution)
                code_number = int(code_match.group(1)) if code_match else 0
                x_value = position + ((code_number % 7) - 3) * 0.012
                y_value = 100.0 * float(signal_row["value"])
                d_value = float(signal_row["D"])
                marker = "o" if d_value >= 2.0 else "v"
                is_extreme = abs(d_value) >= 3.0
                ax.scatter(
                    x_value,
                    y_value,
                    s=29 if is_extreme else 22,
                    marker=marker,
                    color=color or "#555555",
                    edgecolor="#111111" if is_extreme else "white",
                    linewidth=0.85 if is_extreme else 0.45,
                    zorder=6 if is_extreme else 5,
                    clip_on=False,
                )
        elif n_value > 0:
            offsets = np.linspace(-0.13, 0.13, n_value) if n_value > 1 else np.array([0.0])
            ordered = frame.sort_values("institution_code").reset_index(drop=True)
            for index, (_, row) in enumerate(ordered.iterrows()):
                ax.scatter(
                    position + float(offsets[index]),
                    100.0 * float(row[value_column]),
                    s=17,
                    color="#777777",
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=4,
                    clip_on=False,
                )

    ax.set_title(title, fontsize=9, loc="left", pad=5)
    ax.set_xlim(0.45, len(years) + 0.55)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="#E3E3E3", linewidth=0.55)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    tick_positions = positions[::2]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(years[index - 1]) for index in tick_positions] if show_year_labels else [])
    ax.tick_params(axis="both", labelsize=7)
    ax.text(-0.025, 0.015, "n", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.3, color="#777777")


def make_temporal_figure(
    annual: pd.DataFrame,
    d_values: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    dpi: int,
) -> None:
    if output_utils.data_only():
        return
    years = list(range(start_year, end_year + 1))
    plt.rcParams.update({
        "font.family": "Source Sans 3",
        "font.size": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharey=True)
    panel_specs = [
        (axes[0, 0], "IMFA", "Top-1", "eligible_imfa_structure", "imfa_top1_partner_share", "(a) IMFA Top-1"),
        (axes[0, 1], "EC-CIS", "Top-1", "eligible_cis_structure", "cis_top1_partner_share", "(b) EC-CIS Top-1"),
        (axes[1, 0], "IMFA", "Top-5", "eligible_imfa_structure", "imfa_top5_union_share", "(c) IMFA Top-5 union"),
        (axes[1, 1], "EC-CIS", "Top-5", "eligible_cis_structure", "cis_top5_union_share", "(d) EC-CIS Top-5 union"),
    ]
    for index, (ax, signal, measure, eligibility, value_column, title) in enumerate(panel_specs):
        _boxplot_panel(
            ax,
            annual,
            d_values,
            years,
            signal,
            measure,
            eligibility,
            value_column,
            title,
            show_year_labels=index >= 2,
        )
    axes[0, 0].set_ylabel("Partner concentration (%)", fontsize=8)
    axes[1, 0].set_ylabel("Partner concentration (%)", fontsize=8)
    axes[1, 0].set_xlabel("Year", fontsize=8)
    axes[1, 1].set_xlabel("Year", fontsize=8)
    legend = [
        Patch(facecolor="#E6E6E6", edgecolor="#808080", label="Median, IQR, and range"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666",
               markeredgecolor="white", markersize=5.2, label=r"Upper robust signal ($D\geq2$)"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="#666666",
               markeredgecolor="white", markersize=5.2, label=r"Lower robust deviation ($D\leq-2$)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#B0B0B0",
               markeredgecolor="#111111", markeredgewidth=1.0, markersize=5.2,
               label=r"Black outline: $|D|\geq3$"),
    ]
    fig.legend(
        handles=legend, loc="lower center", ncol=4, frameon=False,
        fontsize=6.2, bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle("Temporal distribution of affiliation-partner concentration", fontsize=10.5, y=0.995)
    fig.subplots_adjust(top=0.925, bottom=0.135, left=0.09, right=0.985, hspace=0.28, wspace=0.12)
    stem = output_dir / "Figure_6_affiliation_partner_concentration_robust_D"
    figure_style.finish(fig, stem)
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".svg", bbox_inches="tight")
    save_png_via_temporary(fig, Path(str(stem) + ".png"), dpi)
    fig.savefig(str(stem) + ".tiff", dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def make_recurrence_figure(
    recurrence: pd.DataFrame,
    institution_codes: Sequence[str],
    output_dir: Path,
    start_year: int,
    end_year: int,
    dpi: int,
) -> None:
    """Plot elevated/evaluable years for every institution-metric profile."""
    if output_utils.data_only():
        return
    recurrence = institutional_rows(recurrence)
    metrics = ["imfa_top1", "ec_cis_top1", "imfa_top5", "ec_cis_top5"]
    labels = ["IMFA\nTop-1", "EC-CIS\nTop-1", "IMFA\nTop-5", "EC-CIS\nTop-5"]

    def code_sort_key(code: str) -> Tuple[str, int, str]:
        match = re.search(r"^(.*?)(\d+)$", clean_text(code))
        if match:
            return match.group(1), int(match.group(2)), clean_text(code)
        return clean_text(code), 0, clean_text(code)

    codes = sorted(
        {clean_text(code) for code in institution_codes
         if clean_text(code) and clean_text(code).casefold() != "national"},
        key=code_sort_key,
    )
    if not codes:
        return

    if recurrence.empty:
        pivot_elevated = pd.DataFrame(0, index=codes, columns=metrics)
        pivot_extreme = pd.DataFrame(0, index=codes, columns=metrics)
        pivot_evaluable = pd.DataFrame(0, index=codes, columns=metrics)
    else:
        pivot_elevated = recurrence.pivot(
            index="institution_code", columns="metric_key",
            values="n_upper_elevated_years",
        ).reindex(index=codes, columns=metrics).fillna(0)
        pivot_extreme = recurrence.pivot(
            index="institution_code", columns="metric_key",
            values="n_upper_extreme_years",
        ).reindex(index=codes, columns=metrics).fillna(0)
        pivot_evaluable = recurrence.pivot(
            index="institution_code", columns="metric_key",
            values="n_evaluable_years",
        ).reindex(index=codes, columns=metrics).fillna(0)

    order = pd.DataFrame({
        "code": codes,
        "total": pivot_elevated.sum(axis=1).reindex(codes).to_numpy(),
        "extreme": pivot_extreme.sum(axis=1).reindex(codes).to_numpy(),
    }).sort_values(
        ["total", "extreme", "code"], ascending=[False, False, True]
    )["code"].tolist()
    manuscript_order = ['U-17', 'U-03', 'U-12', 'U-13', 'U-11', 'U-23', 'U-05', 'U-10', 'U-24', 'U-15', 'U-21', 'U-18', 'U-06', 'U-09', 'U-22', 'U-02', 'U-04', 'U-14', 'U-16', 'U-25', 'U-28', 'U-01', 'U-07', 'U-08', 'U-20', 'U-19', 'U-26', 'U-27', 'U-29', 'U-30']
    if set(order) == set(manuscript_order):
        order = manuscript_order
    elevated_values = pivot_elevated.loc[order].to_numpy(dtype=float)
    extreme_values = pivot_extreme.loc[order].to_numpy(dtype=float)
    evaluable_values = pivot_evaluable.loc[order].to_numpy(dtype=float)
    cells = []
    for i, code in enumerate(order):
        for j, metric in enumerate(metrics):
            n = int(evaluable_values[i, j])
            e = int(elevated_values[i, j])
            s = int(extreme_values[i, j])
            cells.append({"institution_code": code, "metric_key": metric,
                          "display_row": i, "display_column": j,
                          "n_upper_elevated_years": e, "n_evaluable_years": n,
                          "n_upper_extreme_years": s,
                          "annotation": "%d/%d" % (e, n) if n else "--",
                          "has_extreme_dot": int(s > 0), "color_value_elevated_years": e})
    pd.DataFrame(cells).to_csv(output_dir / "Figure_S10_cells_exact.csv", index=False, encoding="utf-8-sig")

    plt.rcParams.update({
        "font.family": "Source Sans 3",
        "font.size": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure_height = max(4.6, 0.205 * len(order) + 1.25)
    fig, ax = plt.subplots(figsize=(5.6, figure_height))
    vmax = max(1.0, float(np.nanmax(elevated_values)))
    image_handle = figure_style.heatmap(ax,
        elevated_values, cmap="Oranges", vmin=0, vmax=vmax, aspect="auto"
    )
    for row_index in range(len(order)):
        for column_index in range(len(metrics)):
            elevated_count = int(elevated_values[row_index, column_index])
            evaluable_count = int(evaluable_values[row_index, column_index])
            extreme_count = int(extreme_values[row_index, column_index])
            label = (
                "%d/%d" % (elevated_count, evaluable_count)
                if evaluable_count else "--"
            )
            color = "white" if elevated_count >= max(2, 0.62 * vmax) else "#202020"
            ax.text(
                column_index, row_index, label,
                ha="center", va="center", fontsize=6.0, color=color,
            )
            if extreme_count:
                # A separate marker remains black even when count text is white.
                ax.scatter(column_index + 0.35, row_index, s=7, marker="o", color="black", zorder=5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6.4)
    ax.tick_params(
        top=True, bottom=False, labeltop=True, labelbottom=False, length=0
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image_handle, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label("Years with elevated concentration (D >= 2)", fontsize=7.0)
    colorbar.ax.tick_params(labelsize=6.2)
    ax.set_title(
        "Recurrence of robust-D partner-concentration signals, %d-%d"
        % (start_year, end_year),
        fontsize=9.2,
        pad=22,
    )
    fig.text(
        0.01, 0.012,
        "Cells report elevated years/evaluable years. Black dot: at least one extreme signal (D >= 3).\n"
        "Years without an evaluable institutional reference are excluded from the denominator.",
        fontsize=6.2, ha="left", va="bottom",
    )
    fig.subplots_adjust(left=0.17, right=0.90, top=0.90, bottom=0.055)
    stem = output_dir / "Figure_S10_robust_D_recurrence_partner_concentration"
    figure_style.finish(fig, stem)
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".svg", bbox_inches="tight")
    save_png_via_temporary(fig, Path(str(stem) + ".png"), dpi)
    fig.savefig(str(stem) + ".tiff", dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def plot_dumbbell(
    ax,
    frame: pd.DataFrame,
    prefix: str,
    title: str,
    denominator_column: str,
) -> None:
    frame = institutional_rows(frame)
    ordered = frame.sort_values(prefix + "_top5_union_share", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    top1 = 100.0 * ordered[prefix + "_top1_partner_share"].astype(float)
    top5 = 100.0 * ordered[prefix + "_top5_union_share"].astype(float)
    sufficient = ordered[denominator_column] >= MIN_SIGNAL_ARTICLES

    for index, row in ordered.iterrows():
        color = row["institution_color"]
        ax.plot([top1.iloc[index], top5.iloc[index]], [index, index], color="#B7B7B7", linewidth=0.65, zorder=1)
        if sufficient.iloc[index]:
            ax.scatter(top1.iloc[index], index, s=18, marker="o", color=color, edgecolor="white", linewidth=0.35, zorder=3)
            ax.scatter(top5.iloc[index], index, s=22, marker="s", color=color, edgecolor="white", linewidth=0.35, zorder=3)
        else:
            ax.scatter(top1.iloc[index], index, s=18, marker="o", facecolor="none", edgecolor=color, linewidth=0.8, zorder=3)
            ax.scatter(top5.iloc[index], index, s=22, marker="s", facecolor="none", edgecolor=color, linewidth=0.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["institution_code"], fontsize=6.2)
    ax.set_xlabel("Partner concentration (%)", fontsize=8)
    ax.set_title(title, fontsize=9, loc="left", pad=5)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=7)


def annotate_selected(
    ax,
    frame: pd.DataFrame,
    x: str,
    y: str,
    selected_codes: Set[str],
    offsets: Mapping[str, Tuple[int, int]],
) -> None:
    for _, row in frame[frame["institution_code"].isin(selected_codes)].iterrows():
        code = clean_text(row["institution_code"])
        dx, dy = offsets.get(code, (5, 4))
        ax.annotate(
            code,
            (100.0 * row[x], 100.0 * row[y]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.2,
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
            color="#303030",
            arrowprops={"arrowstyle": "-", "color": "#707070", "linewidth": 0.35},
        )


def make_supplementary_2025_figure(
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    output_dir: Path,
    year: int,
    dpi: int,
) -> None:
    if output_utils.data_only():
        return
    summary = institutional_rows(summary)
    selected = institutional_rows(selected)
    plt.rcParams.update({
        "font.family": "Source Sans 3",
        "font.size": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.2, 9.1))
    grid = fig.add_gridspec(2, 2, height_ratios=[2.35, 1.0], hspace=0.30, wspace=0.34)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    plot_dumbbell(ax_a, summary, "imfa", "(a) IMFA affiliation-partner concentration", "n_imfa_articles")
    plot_dumbbell(ax_b, summary, "cis", "(b) EC-CIS nonfocal-publication concentration", "n_cis_external_articles")

    eligible = summary[summary["eligible_joint_structure"] == 1].copy()
    colors = eligible["institution_color"].tolist()
    sizes = 16.0 + 2.2 * np.sqrt(eligible["n_articles_attributed"].astype(float))
    selected_codes = set(selected["institution_code"]) if not selected.empty else set()

    recurrent = eligible.dropna(
        subset=["ec_cis_single_recurrent_partner_share"]
    ).sort_values("institution_code").reset_index(drop=True)
    recurrent_y = np.arange(len(recurrent))
    ax_c.scatter(
        100.0 * recurrent["ec_cis_single_recurrent_partner_share"],
        recurrent_y,
        s=24,
        marker="^",
        c=recurrent["institution_color"].tolist(),
        edgecolor="white",
        linewidth=0.4,
        alpha=0.95,
    )
    ax_c.set_yticks(recurrent_y)
    ax_c.set_yticklabels(recurrent["institution_code"], fontsize=5.9)
    ax_c.set_xlim(0, 100)
    ax_c.set_xlabel("Single recurrent partner (%)", fontsize=7.5)
    ax_c.set_title("(c) EC-CIS author recurrence", fontsize=9, loc="left", pad=5)
    ax_c.tick_params(axis="y", length=0)

    ax_d.scatter(
        100.0 * eligible["partner_jaccard"],
        100.0 * eligible["author_partner_jaccard"],
        s=sizes,
        c=colors,
        edgecolor="#505050",
        linewidth=0.45,
        alpha=0.92,
    )
    ax_d.set_xlabel("Partner-entity Jaccard overlap (%)", fontsize=7.5)
    ax_d.set_ylabel("Author-partner Jaccard overlap (%)", fontsize=7.5)
    ax_d.set_title("(d) Within-corpus IMFA–EC-CIS overlap", fontsize=9, loc="left", pad=5)
    offsets_d = {
        "U-01": (5, 5), "U-05": (5, -8), "U-10": (6, -9),
        "U-11": (-5, -10), "U-12": (-8, 10), "U-16": (5, -8),
        "U-24": (-8, -10),
    }
    annotate_selected(
        ax_d, eligible, "partner_jaccard", "author_partner_jaccard",
        selected_codes, offsets_d,
    )

    for ax in [ax_c, ax_d]:
        ax.grid(color="#E6E6E6", linewidth=0.55)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=7)

    from matplotlib.lines import Line2D

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markeredgecolor="white", markersize=5, label="Top-1"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#666666", markeredgecolor="white", markersize=5, label="Top-5 union"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#666666", markeredgecolor="white", markersize=5, label="Single recurrent partner"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=7.2)
    fig.suptitle("Institution-level affiliation structures in " + str(year), fontsize=10.5, y=1.018)
    fig.text(
        0.01, 0.012,
        "EC-CIS is restricted to observed same-year nonfocal publications within the Ecuadorian institutional-output corpus. "
        "Panel c includes author–focal-university records with at least two observed nonfocal publications and a resolved partner affiliation.",
        fontsize=5.8, ha="left",
    )
    fig.subplots_adjust(top=0.956, bottom=0.085, left=0.085, right=0.985)

    stem = output_dir / (
        "Figure_S9_affiliation_structures_" + str(year) + "_EC_CIS"
    )
    figure_style.finish(fig, stem)
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".svg", bbox_inches="tight")
    save_png_via_temporary(fig, Path(str(stem) + ".png"), dpi)
    fig.savefig(str(stem) + ".tiff", dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def public_institution_year_table(annual: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "year", "institution_code", "n_articles_attributed",
        "n_imfa_articles", "imfa_rate", "n_imfa_authors", "imfa_n_partner_entities",
        "imfa_top1_partner_count", "imfa_top1_partner_share",
        "imfa_top5_union_count", "imfa_top5_union_share",
        "n_cis_articles", "cis_rate", "n_cis_authors", "n_cis_external_articles",
        "cis_n_partner_entities", "cis_top1_partner_count", "cis_top1_partner_share",
        "cis_top5_union_count", "cis_top5_union_share",
        "eligible_imfa_structure", "eligible_cis_structure", "eligible_joint_structure",
    ]
    return annual[[column for column in columns if column in annual.columns]].copy()


def write_temporal_outputs(annual, d_values, final_summary, output_dir, end_year):
    """Write only the two affiliation sources consumed by Tables S13 and S14."""
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched_annual = add_d_columns(public_institution_year_table(annual), d_values)
    public_annual = public_ec_cis_columns(enriched_annual)
    public_annual.to_csv(output_dir / 'Table_S14_institution_year_concentration_robust_D.csv', index=False, encoding='utf-8-sig')
    final_public = public_ec_cis_columns(make_public_summary(final_summary))
    if 'year' not in final_public.columns:
        final_public.insert(0, 'year', end_year)
    final_public.to_csv(output_dir / ('Table_S20_affiliation_structures_%d_updated.csv' % end_year), index=False, encoding='utf-8-sig')


def main() -> None:
    """Calculate annual and final-year affiliation results, without figures."""
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    mapping_path = Path(args.mapping_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.end_year < args.start_year:
        raise ValueError('--end-year must be greater than or equal to --start-year')
    if args.min_d_reference_institutions < 4:
        raise ValueError('--min-d-reference-institutions must be at least 4')
    if args.d_elevated <= 0 or args.d_extreme <= args.d_elevated:
        raise ValueError('Require 0 < --d-elevated < --d-extreme')
    mapping = read_mapping(mapping_path, args.mapping_encoding)
    annual_frames = []
    final_summary = None
    for year in range(args.start_year, args.end_year + 1):
        input_paths = locate_inputs(input_dir, year)
        metrics = read_csv(input_paths['institution_year_metrics.csv'])
        imfa_raw = read_csv(input_paths['imfa_author_partner_institutions.csv'])
        cis_raw = read_csv(input_paths['cis_author_partner_institutions.csv'])
        imfa = build_structures(imfa_raw, 'imfa_article_eids', mapping)
        cis = build_structures(cis_raw, 'external_article_eids', mapping)
        partner_codes = assign_partner_codes(imfa['partner_summary'], cis['partner_summary'], mapping)
        summary, _, _, _ = build_summary(metrics, mapping, imfa, cis, partner_codes)
        summary.insert(0, 'year', year)
        annual_frames.append(summary)
        if year == args.end_year:
            final_summary = summary.copy()
    if final_summary is None:
        raise RuntimeError('Final-year structures were not created')
    annual = pd.concat(annual_frames, ignore_index=True, sort=False)
    d_values = robust_d_signals(annual, args.min_d_reference_institutions, args.d_elevated, args.d_extreme)
    write_temporal_outputs(annual, d_values, final_summary, output_dir, args.end_year)
    annual_summary = annual_distribution_summary(annual, d_values)
    matched_start_year = 2018 if args.start_year <= 2018 <= args.end_year else args.start_year
    matched = matched_endpoint_changes(annual, matched_start_year, args.end_year)
    eligible_final = institutional_rows(final_summary)
    eligible_final = eligible_final.loc[eligible_final['eligible_joint_structure'].eq(1)]
    final_descriptive = {variable: descriptive(eligible_final[variable]) for variable in [
        'imfa_top1_partner_share', 'imfa_top5_union_share', 'cis_top1_partner_share',
        'cis_top5_union_share', 'partner_jaccard', 'author_partner_jaccard']}
    output_utils.report_methodology(annual_partner_concentration=annual_summary,
        matched_endpoint_changes=matched_endpoint_summary(matched, matched_start_year, args.end_year),
        partner_concentration_size_sensitivity=size_association_table(annual),
        paired_ec_cis_minus_imfa=annual_paired_comparisons(annual),
        final_year_joint_descriptive=final_descriptive)
    print(f'Calculated affiliation sources in {output_dir}')


if __name__ == "__main__":
    main()
