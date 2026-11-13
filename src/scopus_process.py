#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local Scopus processing for the bibliometric study. See README.md for output contracts and validation.
Reads local records and private institution definitions without network requests.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------
# 1. Local institution configuration
# ---------------------------------------------------------------------
# Associations are read from a private JSON file; see config/institutions.example.json.
SCRIPT_VERSION = "2.9.1-canonical-affiliation-metrics-2026-07-14"
PROCESSOR_REFACTOR_VERSION = "phase2-1.0"
RETAINED_OUTPUTS = ('article_affiliations', 'author_affiliations', 'corresponding_author_affiliations', 'article_institution_flags', 'institution_year_metrics', 'national_baseline', 'top_author_concentration', 'author_productivity', 'hyperprolific_author_flags', 'imfa_author_partner_institutions', 'cis_author_partner_institutions', 'institutions')
AUDIT_OUTPUTS = ('article_year_audit', 'duplicate_record_report', 'excluded_record_report', 'record_processing_summary', 'author_affiliation_canonicalization', 'affiliation_alias_correction_audit')
NATIONAL_INSTITUTION_ID = "NATIONAL"
NATIONAL_INSTITUTION_NAME = "Ecuador - all focal institutions"

# Document types excluded from the analysis by default.
# Matching is case-insensitive and is applied to coredata["subtypeDescription"].
DEFAULT_EXCLUDED_SUBTYPE_DESCRIPTIONS = {"erratum", "retracted"}

INSTITUTIONS = []  # Loaded only from the user's local configuration.

def as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def get_nested(d: Any, path: Iterable[str], default: Any = None) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def text_value(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float)):
        return str(x)
    if isinstance(x, dict) and "$" in x:
        return text_value(x.get("$"))
    return None


def normalize_name(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s).lower().strip())


def organization_to_text(org: Any) -> Optional[str]:
    parts = []
    for item in as_list(org):
        val = text_value(item)
        if val:
            parts.append(val)
    return "; ".join(parts) if parts else None


def extract_year(cover_date: Optional[str]) -> Optional[int]:
    if not cover_date:
        return None
    m = re.match(r"^(\d{4})", str(cover_date))
    return int(m.group(1)) if m else None


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(x)
    except Exception:
        return None


def get_record_eid(record: Dict[str, Any]) -> Optional[str]:
    """Return EID from a Scopus Abstract Retrieval record."""
    return get_nested(
        record,
        ["abstracts-retrieval-response", "coredata", "eid"],
        default=None,
    )


def get_record_subtype_description(record: Dict[str, Any]) -> Optional[str]:
    """Return Scopus document subtype description, e.g. Article, Review, Erratum."""
    return get_nested(
        record,
        ["abstracts-retrieval-response", "coredata", "subtypeDescription"],
        default=None,
    )


def get_record_subtype_code(record: Dict[str, Any]) -> Optional[str]:
    """Return Scopus subtype code, e.g. ar, re, cp, er."""
    return get_nested(
        record,
        ["abstracts-retrieval-response", "coredata", "subtype"],
        default=None,
    )


def is_excluded_document_type(
    record: Dict[str, Any],
    excluded_subtype_descriptions: Optional[set] = None,
) -> bool:
    """Return True when record should be excluded by document type."""
    if excluded_subtype_descriptions is None:
        excluded_subtype_descriptions = DEFAULT_EXCLUDED_SUBTYPE_DESCRIPTIONS
    desc = get_record_subtype_description(record)
    desc_norm = str(desc).strip().lower() if desc is not None else ""
    return desc_norm in excluded_subtype_descriptions


def record_completeness_score(record: Dict[str, Any]) -> int:
    """
    Score metadata richness for choosing between duplicate records with the same EID.

    Higher score means richer metadata. This is useful when the same EID appears
    more than once because it was retrieved through several institutional AF-ID
    searches. The script keeps the highest-scoring record.
    """
    arr = record.get("abstracts-retrieval-response", {})
    core = arr.get("coredata", {}) or {}

    authors = as_list(get_nested(arr, ["authors", "author"], default=[]))
    affiliations = as_list(arr.get("affiliation"))
    author_groups = as_list(get_nested(arr, ["item", "bibrecord", "head", "author-group"], default=[]))
    correspondences = as_list(get_nested(arr, ["item", "bibrecord", "head", "correspondence"], default=[]))
    subject_areas = as_list(get_nested(arr, ["subject-areas", "subject-area"], default=[]))
    keywords = as_list(get_nested(arr, ["authkeywords", "author-keyword"], default=[]))

    author_affiliation_links = 0
    for a in authors:
        if isinstance(a, dict):
            author_affiliation_links += len(as_list(a.get("affiliation")))

    score = 0
    score += 1000 * len(authors)
    score += 100 * author_affiliation_links
    score += 50 * len(author_groups)
    score += 25 * len(affiliations)
    score += 25 * len(correspondences)
    score += 10 * len(subject_areas)
    score += 5 * len(keywords)
    score += 5 if core.get("prism:doi") else 0
    score += 5 if core.get("dc:title") else 0
    score += 5 if core.get("prism:publicationName") else 0
    score += 5 if core.get("subtype") else 0
    return score


def unpack_json_records(data: Any) -> List[Dict[str, Any]]:
    """Unpack supported JSON structures into a flat list of candidate records."""
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict) and "abstracts-retrieval-response" in data:
        return [data]
    if isinstance(data, dict):
        return [v for v in data.values() if isinstance(v, dict)]
    raise ValueError("Unsupported JSON structure.")


def filter_and_deduplicate_records(
    records: List[Dict[str, Any]],
    excluded_subtype_descriptions: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Validate, filter and deduplicate records.

    Returns:
        deduped_records:
            Records that will be processed.
        duplicate_report:
            One row per non-excluded valid input record, marking duplicate EIDs and
            the chosen record.
        excluded_report:
            Records excluded because of document type or missing EID.
        processing_summary:
            Counts for audit.
    """
    if excluded_subtype_descriptions is None:
        excluded_subtype_descriptions = DEFAULT_EXCLUDED_SUBTYPE_DESCRIPTIONS

    valid_rows = []
    excluded_rows = []
    best = {}

    n_candidate_records = len(records)
    n_invalid_records = 0

    for input_index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("abstracts-retrieval-response"), dict):
            n_invalid_records += 1
            excluded_rows.append({
                "input_index": input_index,
                "eid": None,
                "doi": None,
                "title": None,
                "year": None,
                "subtype": None,
                "subtype_description": None,
                "exclusion_reason": "invalid_or_not_abstract_retrieval_record",
            })
            continue

        eid = get_record_eid(record)
        core = get_nested(record, ["abstracts-retrieval-response", "coredata"], default={}) or {}
        subtype_desc = get_record_subtype_description(record)
        subtype_code = get_record_subtype_code(record)
        year = extract_year(core.get("prism:coverDate"))

        base_row = {
            "input_index": input_index,
            "eid": eid,
            "doi": core.get("prism:doi"),
            "title": core.get("dc:title"),
            "year": year,
            "subtype": subtype_code,
            "subtype_description": subtype_desc,
        }

        if not eid:
            excluded_rows.append({**base_row, "exclusion_reason": "missing_eid"})
            continue

        if is_excluded_document_type(record, excluded_subtype_descriptions):
            excluded_rows.append({**base_row, "exclusion_reason": "excluded_document_type"})
            continue

        score = record_completeness_score(record)
        valid_rows.append({**base_row, "completeness_score": score})

        if eid not in best or score > best[eid]["score"]:
            best[eid] = {
                "score": score,
                "input_index": input_index,
                "record": record,
            }

    deduped_records = [v["record"] for _, v in sorted(best.items(), key=lambda item: item[1]["input_index"])]

    duplicate_report = pd.DataFrame(valid_rows)
    if not duplicate_report.empty:
        counts = duplicate_report.groupby("eid").size().rename("n_records_with_same_eid").reset_index()
        chosen = pd.DataFrame([
            {
                "eid": eid,
                "chosen_input_index": v["input_index"],
                "chosen_completeness_score": v["score"],
            }
            for eid, v in best.items()
        ])
        duplicate_report = duplicate_report.merge(counts, on="eid", how="left").merge(chosen, on="eid", how="left")
        duplicate_report["is_duplicate_eid"] = (duplicate_report["n_records_with_same_eid"] > 1).astype(int)
        duplicate_report["is_chosen_record"] = (duplicate_report["input_index"] == duplicate_report["chosen_input_index"]).astype(int)
    else:
        duplicate_report = pd.DataFrame(columns=[
            "input_index", "eid", "doi", "title", "year", "subtype", "subtype_description",
            "completeness_score", "n_records_with_same_eid", "chosen_input_index",
            "chosen_completeness_score", "is_duplicate_eid", "is_chosen_record"
        ])

    excluded_report = pd.DataFrame(excluded_rows)
    if excluded_report.empty:
        excluded_report = pd.DataFrame(columns=[
            "input_index", "eid", "doi", "title", "year", "subtype",
            "subtype_description", "exclusion_reason"
        ])

    n_valid_non_excluded = len(valid_rows)
    n_duplicate_records_removed = n_valid_non_excluded - len(deduped_records)
    n_excluded_document_type = int((excluded_report.get("exclusion_reason") == "excluded_document_type").sum()) if not excluded_report.empty else 0
    n_missing_eid = int((excluded_report.get("exclusion_reason") == "missing_eid").sum()) if not excluded_report.empty else 0

    processing_summary = pd.DataFrame([{
        "candidate_records_in_input": n_candidate_records,
        "invalid_records": n_invalid_records,
        "excluded_missing_eid": n_missing_eid,
        "excluded_document_type": n_excluded_document_type,
        "valid_non_excluded_records_before_eid_dedup": n_valid_non_excluded,
        "duplicate_records_removed_by_eid": n_duplicate_records_removed,
        "records_processed_after_filtering_and_dedup": len(deduped_records),
        "unique_eids_processed": len({get_record_eid(r) for r in deduped_records}),
    }])

    return deduped_records, duplicate_report, excluded_report, processing_summary


def load_scopus_records(path: Path) -> Tuple[List[Dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load valid JSON records, exclude unwanted document types and deduplicate by EID."""
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    candidate_records = unpack_json_records(data)
    return filter_and_deduplicate_records(candidate_records)


def infer_analysis_year_from_path(path: Path) -> Optional[int]:
    """Infer the Scopus search/PUBYEAR year from the input JSON filename.

    The analysis year is intentionally different from the publication year parsed
    from coredata["prism:coverDate"]. Annual Scopus downloads may contain records
    whose coverDate belongs to a different calendar year than the PUBYEAR query used
    to download the JSON. For annual analyses, the correct year is the search year
    encoded in the JSON filename or supplied via --analysis-year.
    """
    candidates = []
    for match in re.finditer(r"(?<!\d)(20\d{2})(?!\d)", path.name):
        year = int(match.group(1))
        if 1900 <= year <= 2100:
            candidates.append(year)
    if not candidates:
        return None
    return candidates[0]


def apply_analysis_year(
    tables: Dict[str, pd.DataFrame],
    analysis_year: int,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Set the article year used by all metrics to the JSON/PUBYEAR year.

    The original year extracted from coverDate is preserved as publication_year.
    All downstream metric calculations continue to use the column named year, so
    overwriting year here forces institution_year_metrics and all auxiliary
    author/partner tables to use the intended annual analysis year.
    """
    out = {name: df.copy() for name, df in tables.items()}
    articles = out.get("articles", pd.DataFrame()).copy()

    if articles.empty:
        audit = pd.DataFrame(columns=[
            "eid", "title", "cover_date", "publication_year", "analysis_year",
            "is_off_year_cover_date",
        ])
        return out, audit

    if "year" in articles.columns:
        articles["publication_year"] = pd.to_numeric(articles["year"], errors="coerce").astype("Int64")
    else:
        articles["publication_year"] = pd.NA

    articles["analysis_year"] = int(analysis_year)
    articles["year"] = int(analysis_year)
    articles["is_off_year_cover_date"] = (
        articles["publication_year"].notna()
        & (articles["publication_year"].astype("Int64") != int(analysis_year))
    ).astype(int)

    audit_cols = [
        "eid", "title", "cover_date", "publication_year", "analysis_year",
        "is_off_year_cover_date",
    ]
    audit = articles[[c for c in audit_cols if c in articles.columns]].copy()
    out["articles"] = articles
    return out, audit


def apply_analysis_year_to_record_reports(
    duplicate_record_report: pd.DataFrame,
    excluded_record_report: pd.DataFrame,
    processing_summary: pd.DataFrame,
    analysis_year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Annotate pre-extraction audit reports with analysis year and publication year."""

    def _annotate(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if out.empty:
            return out
        if "year" in out.columns:
            out = out.rename(columns={"year": "publication_year"})
            out["publication_year"] = pd.to_numeric(out["publication_year"], errors="coerce").astype("Int64")
            out["analysis_year"] = int(analysis_year)
            out["is_off_year_cover_date"] = (
                out["publication_year"].notna()
                & (out["publication_year"].astype("Int64") != int(analysis_year))
            ).astype(int)
        return out

    duplicate = _annotate(duplicate_record_report)
    excluded = _annotate(excluded_record_report)
    summary = processing_summary.copy()
    if not summary.empty:
        summary["analysis_year"] = int(analysis_year)
        summary["off_year_cover_date_records_processed"] = int(
            duplicate.loc[
                (duplicate.get("is_chosen_record", 0) == 1)
                & (duplicate.get("is_off_year_cover_date", 0) == 1)
            ].shape[0]
        ) if not duplicate.empty else 0
    return duplicate, excluded, summary


def extract_article(record: Dict[str, Any]) -> Dict[str, Any]:
    arr = record.get("abstracts-retrieval-response", {})
    core = arr.get("coredata", {}) or {}
    cover_date = core.get("prism:coverDate")

    return {
        "eid": core.get("eid"),
        "scopus_id": str(core.get("dc:identifier", "")).replace("SCOPUS_ID:", "") or None,
        "doi": core.get("prism:doi"),
        "title": core.get("dc:title"),
        # Initial extraction from coverDate. This value is preserved later as
        # publication_year; the analysis year is applied after the input JSON
        # filename or --analysis-year is known.
        "year": extract_year(cover_date),
        "cover_date": cover_date,
        "source_title": core.get("prism:publicationName"),
        "source_id": core.get("source-id"),
        "srctype": core.get("srctype"),
        "aggregation_type": core.get("prism:aggregationType"),
        "subtype": core.get("subtype"),
        "subtype_description": core.get("subtypeDescription"),
        "citedby_count": safe_int(core.get("citedby-count")),
        "issn": core.get("prism:issn"),
        "isbn": core.get("prism:isbn"),
        "volume": core.get("prism:volume"),
        "issue": core.get("prism:issueIdentifier"),
        "page_range": core.get("prism:pageRange"),
        "starting_page": core.get("prism:startingPage"),
        "ending_page": core.get("prism:endingPage"),
        "article_number": core.get("article-number"),
        "openaccess": core.get("openaccess"),
        "openaccess_flag": core.get("openaccessFlag"),
        "pubmed_id": core.get("pubmed-id"),
        "pii": core.get("pii"),
    }


def extract_article_affiliations(record: Dict[str, Any], eid: str) -> List[Dict[str, Any]]:
    arr = record.get("abstracts-retrieval-response", {})
    rows = []
    for aff in as_list(arr.get("affiliation")):
        if not isinstance(aff, dict):
            continue
        rows.append({
            "eid": eid,
            "afid": aff.get("@id"),
            "affiliation_name": aff.get("affilname"),
            "affiliation_city": aff.get("affiliation-city"),
            "affiliation_country": aff.get("affiliation-country"),
            "affiliation_href": aff.get("@href"),
        })
    return rows


def extract_authors_flat(record: Dict[str, Any], eid: str) -> List[Dict[str, Any]]:
    arr = record.get("abstracts-retrieval-response", {})
    author_list = get_nested(arr, ["authors", "author"], default=[])
    rows = []

    for author in as_list(author_list):
        if not isinstance(author, dict):
            continue
        seen_afids = set()
        afids = []
        aff_hrefs = []
        for aff in as_list(author.get("affiliation")):
            if not isinstance(aff, dict):
                continue
            afid = aff.get("@id") or aff.get("@afid")
            href = aff.get("@href")
            if afid and str(afid) not in seen_afids:
                seen_afids.add(str(afid))
                afids.append(str(afid))
                if href:
                    aff_hrefs.append(href)

        rows.append({
            "eid": eid,
            "author_id": author.get("@auid"),
            "author_seq": safe_int(author.get("@seq")),
            "indexed_name": author.get("ce:indexed-name"),
            "given_name": author.get("ce:given-name"),
            "surname": author.get("ce:surname"),
            "initials": author.get("ce:initials"),
            "orcid": author.get("@orcid"),
            "author_url": author.get("author-url"),
            "n_distinct_author_afids": len(afids),
            "author_afids": ";".join(afids),
            "author_affiliation_hrefs": ";".join(aff_hrefs),
            "is_multi_affiliated_author": int(len(afids) > 1),
        })
    return rows


def extract_author_affiliations_from_authors(record: Dict[str, Any], eid: str) -> List[Dict[str, Any]]:
    arr = record.get("abstracts-retrieval-response", {})
    author_list = get_nested(arr, ["authors", "author"], default=[])
    rows = []

    for author in as_list(author_list):
        if not isinstance(author, dict):
            continue
        seen_afids = set()
        for aff in as_list(author.get("affiliation")):
            if not isinstance(aff, dict):
                continue
            afid = aff.get("@id") or aff.get("@afid")
            if not afid:
                continue
            afid = str(afid)
            if afid in seen_afids:
                continue
            seen_afids.add(afid)
            rows.append({
                "eid": eid,
                "author_id": author.get("@auid"),
                "author_seq": safe_int(author.get("@seq")),
                "indexed_name": author.get("ce:indexed-name"),
                "afid": afid,
                "affiliation_href": aff.get("@href"),
                "source": "authors.author.affiliation",
            })
    return rows


def extract_correspondences(record: Dict[str, Any], eid: str) -> List[Dict[str, Any]]:
    arr = record.get("abstracts-retrieval-response", {})
    corr = get_nested(arr, ["item", "bibrecord", "head", "correspondence"], default=[])
    rows = []

    for idx, c in enumerate(as_list(corr), start=1):
        if not isinstance(c, dict):
            continue
        person = c.get("person") if isinstance(c.get("person"), dict) else {}
        aff = c.get("affiliation") if isinstance(c.get("affiliation"), dict) else {}
        rows.append({
            "eid": eid,
            "correspondence_index": idx,
            "corr_indexed_name": person.get("ce:indexed-name"),
            "corr_given_name": person.get("ce:given-name"),
            "corr_surname": person.get("ce:surname"),
            "corr_initials": person.get("ce:initials"),
            "corr_degrees": person.get("ce:degrees"),
            "corr_country": aff.get("country"),
            "corr_country_code": aff.get("@country"),
            "corr_city": aff.get("city"),
            "corr_organization_text": organization_to_text(aff.get("organization")),
            "corr_source_text": aff.get("ce:source-text"),
            "corr_affid_direct": aff.get("@afid") or get_nested(aff, ["affiliation-id", "@afid"]),
        })
    return rows


def match_corresponding_authors(correspondences: pd.DataFrame, authors: pd.DataFrame, author_affiliations: pd.DataFrame) -> pd.DataFrame:
    cols = ["eid", "correspondence_index", "corr_indexed_name", "corr_source_text", "author_id", "indexed_name", "afid", "match_method", "is_matched"]
    if correspondences.empty:
        return pd.DataFrame(columns=cols)

    authors2 = authors.copy()
    authors2["name_norm"] = authors2["indexed_name"].map(normalize_name)
    authors2["surname_norm"] = authors2["surname"].map(normalize_name)
    authors2["initials_norm"] = authors2["initials"].map(normalize_name)

    aff_by_author = (
        author_affiliations.dropna(subset=["eid", "author_id", "afid"])
        .groupby(["eid", "author_id"])["afid"]
        .apply(lambda x: sorted(set(map(str, x))))
        .to_dict()
    )

    rows = []
    for _, c in correspondences.iterrows():
        eid = c["eid"]
        sub = authors2[authors2["eid"] == eid]
        corr_name_norm = normalize_name(c.get("corr_indexed_name"))
        corr_surname_norm = normalize_name(c.get("corr_surname"))
        corr_initials_norm = normalize_name(c.get("corr_initials"))

        matched = pd.DataFrame()
        method = None
        if corr_name_norm:
            matched = sub[sub["name_norm"] == corr_name_norm]
            if not matched.empty:
                method = "indexed_name_exact"
        if matched.empty and corr_surname_norm and corr_initials_norm:
            matched = sub[(sub["surname_norm"] == corr_surname_norm) & (sub["initials_norm"] == corr_initials_norm)]
            if not matched.empty:
                method = "surname_initials_exact"

        if matched.empty:
            rows.append({
                "eid": eid,
                "correspondence_index": c["correspondence_index"],
                "corr_indexed_name": c.get("corr_indexed_name"),
                "corr_source_text": c.get("corr_source_text"),
                "author_id": None,
                "indexed_name": None,
                "afid": c.get("corr_affid_direct"),
                "match_method": "direct_corr_afid_only" if c.get("corr_affid_direct") else None,
                "is_matched": 0,
            })
            continue

        for _, a in matched.iterrows():
            afids = aff_by_author.get((eid, a["author_id"]), [])
            if not afids and c.get("corr_affid_direct"):
                afids = [str(c.get("corr_affid_direct"))]
            if not afids:
                afids = [None]
            for afid in afids:
                rows.append({
                    "eid": eid,
                    "correspondence_index": c["correspondence_index"],
                    "corr_indexed_name": c.get("corr_indexed_name"),
                    "corr_source_text": c.get("corr_source_text"),
                    "author_id": a["author_id"],
                    "indexed_name": a["indexed_name"],
                    "afid": afid,
                    "match_method": method,
                    "is_matched": 1,
                })
    return pd.DataFrame(rows, columns=cols)


def compute_top_author_concentration(
    focal_author_year: pd.DataFrame,
    max_top_n: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute institutional production concentration among the n most productive authors.

    For each institution-year, authors are ranked by the number of distinct articles
    they sign for the focal institution. For the top n authors, the numerator is the
    union of EIDs signed by those authors. This avoids overestimating their contribution
    when highly productive authors coauthor the same papers.

    Returns:
        topn_wide:
            One row per institution-year with count/share columns for n = 1..max_top_n.
        topn_long:
            One row per institution-year-n with the top author set and union article count.
        author_productivity:
            One row per institution-year-author with rank and productivity.
    """
    wide_cols = ["institution_id", "year"]
    for n in range(1, max_top_n + 1):
        wide_cols.extend([
            f"top_{n}_authors_article_count",
            f"top_{n}_authors_article_share",
        ])
    wide_cols.append("N_top_authors_needed_50_percent_output_union")

    long_cols = [
        "institution_id", "year", "n_top_authors", "N_articles_attributed",
        "top_author_ids", "top_author_names", "n_articles_by_top_authors_union",
        "top_authors_article_share",
    ]
    prod_cols = [
        "institution_id", "year", "author_rank", "author_id", "indexed_name",
        "p_author_institution_year", "author_article_eids",
    ]

    if focal_author_year.empty:
        return (
            pd.DataFrame(columns=wide_cols),
            pd.DataFrame(columns=long_cols),
            pd.DataFrame(columns=prod_cols),
        )

    fay = focal_author_year.dropna(subset=["eid", "institution_id", "year", "author_id"]).copy()
    if fay.empty:
        return (
            pd.DataFrame(columns=wide_cols),
            pd.DataFrame(columns=long_cols),
            pd.DataFrame(columns=prod_cols),
        )

    fay = fay.drop_duplicates(subset=["eid", "institution_id", "year", "author_id"])

    rows_wide = []
    rows_long = []
    rows_prod = []

    for (institution_id, year), g in fay.groupby(["institution_id", "year"], dropna=False):
        total_eids = set(g["eid"].dropna().astype(str))
        total_articles = len(total_eids)
        if total_articles == 0:
            continue

        author_rows = []
        for author_id, ga in g.groupby("author_id", dropna=False):
            eids = set(ga["eid"].dropna().astype(str))
            names = sorted(set(ga["indexed_name"].dropna().astype(str))) if "indexed_name" in ga.columns else []
            author_rows.append({
                "author_id": str(author_id),
                "indexed_name": names[0] if names else None,
                "p_author_institution_year": len(eids),
                "eids": eids,
            })

        # Rank: productivity desc, then author_id for reproducibility.
        author_rows = sorted(
            author_rows,
            key=lambda r: (-r["p_author_institution_year"], str(r["author_id"])),
        )

        wide = {"institution_id": str(institution_id), "year": year}
        needed_50 = None
        cumulative_union = set()

        for rank, ar in enumerate(author_rows, start=1):
            cumulative_union |= ar["eids"]
            if needed_50 is None and len(cumulative_union) / total_articles >= 0.5:
                needed_50 = rank
            rows_prod.append({
                "institution_id": str(institution_id),
                "year": year,
                "author_rank": rank,
                "author_id": ar["author_id"],
                "indexed_name": ar["indexed_name"],
                "p_author_institution_year": ar["p_author_institution_year"],
                "author_article_eids": ";".join(sorted(ar["eids"])),
            })

        for n in range(1, max_top_n + 1):
            top = author_rows[:n]
            top_eids = set()
            for ar in top:
                top_eids |= ar["eids"]
            count = len(top_eids)
            share = count / total_articles if total_articles else pd.NA
            wide[f"top_{n}_authors_article_count"] = count
            wide[f"top_{n}_authors_article_share"] = share
            rows_long.append({
                "institution_id": str(institution_id),
                "year": year,
                "n_top_authors": n,
                "N_articles_attributed": total_articles,
                "top_author_ids": ";".join([ar["author_id"] for ar in top]),
                "top_author_names": ";".join([ar["indexed_name"] or "" for ar in top]),
                "n_articles_by_top_authors_union": count,
                "top_authors_article_share": share,
            })

        wide["N_top_authors_needed_50_percent_output_union"] = needed_50
        rows_wide.append(wide)

    return (
        pd.DataFrame(rows_wide, columns=wide_cols),
        pd.DataFrame(rows_long, columns=long_cols),
        pd.DataFrame(rows_prod, columns=prod_cols),
    )


def compute_hyperprolific_author_diagnostics(
    author_prod: pd.DataFrame,
    focal_author_year: pd.DataFrame,
    focal_author_rows_cis: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify whether hyperprolific authors also show same-article multi-affiliation
    or cross-institution signing.
    """
    detail_cols = [
        "institution_id", "year", "author_id", "indexed_name",
        "p_author_institution_year", "is_hyperprolific_author",
        "is_multi_affiliated_same_publication", "is_cross_institution_signer",
        "is_multi_affiliated_or_cross_institution",
    ]
    summary_cols = [
        "institution_id", "year",
        "N_hyperprolific_authors_multi_affiliated_same_publication",
        "N_hyperprolific_authors_cross_institution_signers",
        "N_hyperprolific_authors_MFA_or_CIS",
    ]

    if author_prod.empty:
        return pd.DataFrame(columns=detail_cols), pd.DataFrame(columns=summary_cols)

    hyper = author_prod[author_prod["is_hyperprolific_author"] == 1].copy()
    if hyper.empty:
        return pd.DataFrame(columns=detail_cols), pd.DataFrame(columns=summary_cols)

    # Same-publication multi-affiliation for the author while signing the focal institution.
    mfa = (
        focal_author_year[focal_author_year["is_multi_affiliated_author"] == 1]
        .drop_duplicates(subset=["institution_id", "year", "author_id"])
        [["institution_id", "year", "author_id"]]
        .copy()
    )
    mfa["is_multi_affiliated_same_publication"] = 1

    # Cross-institution signer flag, propagated to the institution-year-author level.
    cis = focal_author_rows_cis.copy()
    if not cis.empty:
        cis = (
            cis.merge(
                focal_author_year[["eid", "institution_id", "year", "author_id"]].drop_duplicates(),
                on=["eid", "institution_id", "author_id"],
                how="left",
            )
            .dropna(subset=["year"])
            .groupby(["institution_id", "year", "author_id"], as_index=False)["is_cross_institution_signer"]
            .max()
        )
    else:
        cis = pd.DataFrame(columns=["institution_id", "year", "author_id", "is_cross_institution_signer"])

    names = (
        focal_author_year.dropna(subset=["author_id"])
        .drop_duplicates(subset=["institution_id", "year", "author_id"])
        [["institution_id", "year", "author_id", "indexed_name"]]
    )

    detail = (
        hyper.merge(names, on=["institution_id", "year", "author_id"], how="left")
        .merge(mfa, on=["institution_id", "year", "author_id"], how="left")
        .merge(cis, on=["institution_id", "year", "author_id"], how="left")
    )
    detail["is_multi_affiliated_same_publication"] = detail["is_multi_affiliated_same_publication"].fillna(0).astype(int)
    detail["is_cross_institution_signer"] = detail["is_cross_institution_signer"].fillna(0).astype(int)
    detail["is_multi_affiliated_or_cross_institution"] = (
        (detail["is_multi_affiliated_same_publication"] == 1)
        | (detail["is_cross_institution_signer"] == 1)
    ).astype(int)

    summary = (
        detail.groupby(["institution_id", "year"], as_index=False)
        .agg(
            N_hyperprolific_authors_multi_affiliated_same_publication=("is_multi_affiliated_same_publication", "sum"),
            N_hyperprolific_authors_cross_institution_signers=("is_cross_institution_signer", "sum"),
            N_hyperprolific_authors_MFA_or_CIS=("is_multi_affiliated_or_cross_institution", "sum"),
        )
    )

    detail = detail[[c for c in detail_cols if c in detail.columns]]
    summary = summary[[c for c in summary_cols if c in summary.columns]]
    return detail, summary


def normalize_institutions(institutions: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize institution definitions and validate canonical AF-ID mapping.

    Required column:
        afid
    Optional columns:
        institution_id: canonical grouping key. If absent, afid is used.
        name: display name. If absent, institution_id is used.

    This allows one institution to be represented by several Scopus AF-IDs.
    Example: one fictitious university may have several AF-IDs but one institution_id.
    """
    inst = institutions.copy()
    if "afid" not in inst.columns:
        raise ValueError("INSTITUTIONS must contain an 'afid' field.")

    inst["afid"] = inst["afid"].astype(str).str.strip()
    if "institution_id" not in inst.columns:
        inst["institution_id"] = inst["afid"]
    inst["institution_id"] = inst["institution_id"].astype(str).str.strip()
    if "name" not in inst.columns:
        inst["name"] = inst["institution_id"]
    inst["name"] = inst["name"].astype(str).str.strip()

    inst = inst[["afid", "institution_id", "name"]].drop_duplicates()

    afid_to_inst = inst.groupby("afid")["institution_id"].nunique()
    bad_afids = afid_to_inst[afid_to_inst > 1]
    if not bad_afids.empty:
        raise ValueError(
            "The same AF-ID is assigned to more than one institution_id: "
            + ", ".join(bad_afids.index.astype(str))
        )

    return inst


def compute_metrics(
    articles: pd.DataFrame,
    authors: pd.DataFrame,
    author_affiliations: pd.DataFrame,
    article_affiliations: pd.DataFrame,
    corr_author_affiliations: pd.DataFrame,
    institutions: pd.DataFrame,
    hyperauthor_threshold: int,
    hyperprolific_threshold: int,
    include_audit: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    if articles.empty or authors.empty or author_affiliations.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    # Work on local copies and normalize key columns. This prevents duplicated or
    # fragmented author-level outputs caused by mixed dtypes or indexed-name variants.
    articles = articles.copy()
    authors = authors.copy()
    author_affiliations = author_affiliations.copy()
    article_affiliations = article_affiliations.copy() if article_affiliations is not None else pd.DataFrame()
    corr_author_affiliations = corr_author_affiliations.copy()

    for df in [articles, authors, author_affiliations, article_affiliations, corr_author_affiliations]:
        if isinstance(df, pd.DataFrame):
            for col in ["eid", "author_id", "afid"]:
                if col in df.columns:
                    df[col] = df[col].astype("string")

    institutions = normalize_institutions(institutions)
    all_focal_afids = sorted(set(institutions["afid"].astype(str)))

    institution_meta = (
        institutions
        .sort_values(["institution_id", "name"])
        .groupby("institution_id", as_index=False)
        .agg(name=("name", "first"))
    )
    # Add a synthetic national aggregate. It is not a real Scopus institution;
    # it represents all focal institutions pooled as one national system.
    institution_meta = pd.concat([
        institution_meta,
        pd.DataFrame([{
            "institution_id": NATIONAL_INSTITUTION_ID,
            "name": NATIONAL_INSTITUTION_NAME,
        }]),
    ], ignore_index=True)

    institution_afids = (
        institutions.groupby("institution_id")["afid"]
        .apply(lambda x: sorted(set(map(str, x))))
        .rename("institution_afids")
        .reset_index()
    )
    institution_afids = pd.concat([
        institution_afids,
        pd.DataFrame([{
            "institution_id": NATIONAL_INSTITUTION_ID,
            "institution_afids": all_focal_afids,
        }]),
    ], ignore_index=True)
    institution_afid_sets = {
        row["institution_id"]: set(row["institution_afids"])
        for _, row in institution_afids.iterrows()
    }

    # PATCH_CANONICAL_AFFILIATIONS_V1
    # Canonicalize known focal AF-IDs before counting affiliations. Several
    # Scopus AF-IDs can represent the same institution_id and must count as a
    # single affiliation. Unknown/external AF-IDs remain separate tokens.
    afid_to_canonical_institution = dict(
        zip(institutions["afid"].astype(str), institutions["institution_id"].astype(str))
    )

    canonical_author_affiliations = (
        author_affiliations
        .dropna(subset=["eid", "author_id", "afid"])
        [["eid", "author_id", "afid"]]
        .drop_duplicates()
        .copy()
    )
    canonical_author_affiliations["afid"] = (
        canonical_author_affiliations["afid"].astype(str)
    )
    mapped_institution = canonical_author_affiliations["afid"].map(
        afid_to_canonical_institution
    )
    canonical_author_affiliations["canonical_affiliation_id"] = (
        "INST:" + mapped_institution.astype("string")
    )
    external_mask = mapped_institution.isna()
    canonical_author_affiliations.loc[
        external_mask, "canonical_affiliation_id"
    ] = (
        "AFID:"
        + canonical_author_affiliations.loc[external_mask, "afid"].astype("string")
    )

    def _join_affiliation_tokens(values) -> str:
        return ";".join(
            sorted({str(v) for v in values if pd.notna(v) and str(v).strip()})
        )

    author_affiliation_canonicalization = (
        canonical_author_affiliations
        .groupby(["eid", "author_id"], dropna=False)
        .agg(
            n_distinct_raw_afids=("afid", "nunique"),
            raw_afids=("afid", _join_affiliation_tokens),
            n_distinct_canonical_affiliations=(
                "canonical_affiliation_id", "nunique"
            ),
            canonical_affiliation_ids=(
                "canonical_affiliation_id", _join_affiliation_tokens
            ),
        )
        .reset_index()
    )
    author_affiliation_canonicalization["is_multi_affiliated_author_raw_afid"] = (
        author_affiliation_canonicalization["n_distinct_raw_afids"] > 1
    ).astype(int)
    author_affiliation_canonicalization["is_multi_affiliated_author"] = (
        author_affiliation_canonicalization[
            "n_distinct_canonical_affiliations"
        ] > 1
    ).astype(int)
    author_affiliation_canonicalization["is_alias_only_multi_affiliation"] = (
        (author_affiliation_canonicalization[
            "is_multi_affiliated_author_raw_afid"
        ] == 1)
        & (author_affiliation_canonicalization[
            "is_multi_affiliated_author"
        ] == 0)
    ).astype(int)

    # Article-level author count and hyperauthorship.
    n_authors = (
        authors.dropna(subset=["eid", "author_id"])
        .groupby("eid")["author_id"]
        .nunique()
        .rename("n_total_authors_article")
        .reset_index()
    )
    articles_unique = articles.dropna(subset=["eid"]).drop_duplicates(subset=["eid"], keep="first")
    articles2 = articles_unique.merge(n_authors, on="eid", how="left")
    articles2["n_total_authors_article"] = articles2["n_total_authors_article"].fillna(0).astype(int)
    articles2["flag_HAO"] = (articles2["n_total_authors_article"] >= hyperauthor_threshold).astype(int)

    # Article-level MFA after canonicalization. The raw-AF-ID version is
    # retained only for auditing the size of the alias correction.
    article_any_mfa = (
        author_affiliation_canonicalization
        .groupby("eid", as_index=False)
        .agg(
            flag_MFA_raw_afid=(
                "is_multi_affiliated_author_raw_afid", "max"
            ),
            flag_MFA=("is_multi_affiliated_author", "max"),
        )
    )

    # Join author-AFID rows to the focal institution list. This maps AF-ID to canonical institution_id.
    aa = author_affiliations.copy()
    aa["afid"] = aa["afid"].astype(str)
    aa_focal_institutions = aa.merge(institutions, on="afid", how="inner")
    if aa_focal_institutions.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    # Add national pooled rows. If one author signs two Ecuadorian focal AF-IDs
    # in the same article, downstream drop_duplicates() makes that author/article
    # count only once for NATIONAL.
    aa_national = aa_focal_institutions.copy()
    aa_national["institution_id"] = NATIONAL_INSTITUTION_ID
    aa_national["name"] = NATIONAL_INSTITUTION_NAME
    aa_focal = pd.concat([aa_focal_institutions, aa_national], ignore_index=True)

    # Attributed article-institution links. One EID counts only once per canonical institution.
    article_inst = (
        aa_focal[["eid", "institution_id"]]
        .drop_duplicates()
        .merge(institution_meta, on="institution_id", how="left")
        .merge(articles2, on="eid", how="left")
        .merge(article_any_mfa, on="eid", how="left")
    )
    article_inst["flag_MFA_raw_afid"] = (
        article_inst["flag_MFA_raw_afid"].fillna(0).astype(int)
    )
    article_inst["flag_MFA"] = article_inst["flag_MFA"].fillna(0).astype(int)
    article_inst["flag_MFA_alias_only"] = (
        (article_inst["flag_MFA_raw_afid"] == 1)
        & (article_inst["flag_MFA"] == 0)
    ).astype(int)

    # Focal institutional authors. Deduplicate by author_id, not by indexed_name.
    # Scopus may return several indexed-name variants for the same Author ID;
    # all author-level outputs must therefore use author_id as the grouping key.
    def _choose_author_name(values) -> Optional[str]:
        vals = [str(v).strip() for v in values if pd.notna(v) and str(v).strip()]
        if not vals:
            return None
        counts = pd.Series(vals).value_counts()
        max_count = counts.max()
        # Reproducible tie-breaker: choose the longest name, then alphabetically.
        candidates = sorted(
            [name for name, count in counts.items() if count == max_count],
            key=lambda x: (-len(x), x),
        )
        return candidates[0]

    def _join_author_name_variants(values) -> str:
        vals = sorted({str(v).strip() for v in values if pd.notna(v) and str(v).strip()})
        return ";".join(vals)

    author_name_sources = []
    if "author_id" in authors.columns and "indexed_name" in authors.columns:
        author_name_sources.append(authors[["author_id", "indexed_name"]])
    if "author_id" in aa_focal.columns and "indexed_name" in aa_focal.columns:
        author_name_sources.append(aa_focal[["author_id", "indexed_name"]])

    if author_name_sources:
        author_name_lookup = (
            pd.concat(author_name_sources, ignore_index=True)
            .dropna(subset=["author_id"])
            .drop_duplicates()
            .groupby("author_id", as_index=False)
            .agg(
                indexed_name=("indexed_name", _choose_author_name),
                indexed_name_variants=("indexed_name", _join_author_name_variants),
            )
        )
    else:
        author_name_lookup = pd.DataFrame(columns=["author_id", "indexed_name", "indexed_name_variants"])

    focal_author_rows = (
        aa_focal[["eid", "institution_id", "name", "author_id"]]
        .dropna(subset=["eid", "institution_id", "author_id"])
        .drop_duplicates(subset=["eid", "institution_id", "author_id"])
        .merge(author_name_lookup, on="author_id", how="left")
    )

    author_multi = author_affiliation_canonicalization[[
        "eid", "author_id",
        "is_multi_affiliated_author_raw_afid",
        "is_multi_affiliated_author",
    ]].copy()
    focal_author_rows = focal_author_rows.merge(
        author_multi,
        on=["eid", "author_id"],
        how="left",
    )
    for col in [
        "is_multi_affiliated_author_raw_afid",
        "is_multi_affiliated_author",
    ]:
        focal_author_rows[col] = focal_author_rows[col].fillna(0).astype(int)

    # IMFA: at least one author signing for the focal institution has more
    # than one distinct canonical affiliation. Multiple AF-IDs mapped to the
    # same institution_id count once.
    imfa = (
        focal_author_rows
        .groupby(["eid", "institution_id"], as_index=False)
        .agg(
            flag_IMFA_raw_afid=(
                "is_multi_affiliated_author_raw_afid", "max"
            ),
            flag_IMFA=("is_multi_affiliated_author", "max"),
        )
    )
    article_inst = article_inst.merge(
        imfa,
        on=["eid", "institution_id"],
        how="left",
    )
    article_inst["flag_IMFA_raw_afid"] = (
        article_inst["flag_IMFA_raw_afid"].fillna(0).astype(int)
    )
    article_inst["flag_IMFA"] = article_inst["flag_IMFA"].fillna(0).astype(int)
    article_inst["flag_IMFA_alias_only"] = (
        (article_inst["flag_IMFA_raw_afid"] == 1)
        & (article_inst["flag_IMFA"] == 0)
    ).astype(int)

    # Corresponding authorship.
    # Denominator: articles where correspondence can be mapped to at least one AF-ID.
    if corr_author_affiliations.empty:
        article_inst["CA_known"] = 0
        article_inst["flag_CAL"] = 0
    else:
        ca_with_afid = (
            corr_author_affiliations
            .dropna(subset=["afid"])
            .assign(afid=lambda d: d["afid"].astype(str))
            .merge(institutions, on="afid", how="left")
        )
        ca_any = ca_with_afid[["eid"]].drop_duplicates()
        ca_any["CA_known"] = 1
        ca_led = (
            ca_with_afid.dropna(subset=["institution_id"])[["eid", "institution_id"]]
            .drop_duplicates()
        )
        ca_led_national = (
            ca_with_afid[ca_with_afid["afid"].isin(all_focal_afids)][["eid"]]
            .drop_duplicates()
        )
        if not ca_led_national.empty:
            ca_led_national["institution_id"] = NATIONAL_INSTITUTION_ID
            ca_led = pd.concat([ca_led, ca_led_national], ignore_index=True).drop_duplicates()
        ca_led["flag_CAL"] = 1

        article_inst = article_inst.merge(ca_any, on="eid", how="left")
        article_inst["CA_known"] = article_inst["CA_known"].fillna(0).astype(int)
        article_inst = article_inst.merge(ca_led, on=["eid", "institution_id"], how="left")
        article_inst["flag_CAL"] = article_inst["flag_CAL"].fillna(0).astype(int)

    article_inst["flag_NLO"] = (
        (article_inst["CA_known"] == 1) & (article_inst["flag_CAL"] == 0)
    ).astype(int)

    # Add year to focal author rows.
    article_years = articles_unique[["eid", "year"]].drop_duplicates(subset=["eid"], keep="first")
    focal_author_year = (
        focal_author_rows
        .merge(article_years, on="eid", how="left")
        .dropna(subset=["year"])
    )

    # Hyperprolific authors by institution-year.
    author_prod = (
        focal_author_year
        .drop_duplicates(subset=["eid", "institution_id", "author_id"])
        .groupby(["institution_id", "year", "author_id"])
        .size()
        .rename("p_author_institution_year")
        .reset_index()
    )
    author_prod["is_hyperprolific_author"] = (
        author_prod["p_author_institution_year"] >= hyperprolific_threshold
    ).astype(int)
    hyper_authors = author_prod[author_prod["is_hyperprolific_author"] == 1][
        ["institution_id", "year", "author_id"]
    ].drop_duplicates()
    hpa_articles = (
        focal_author_year
        .merge(hyper_authors, on=["institution_id", "year", "author_id"], how="inner")
        [["eid", "institution_id"]]
        .drop_duplicates()
    )
    hpa_articles["flag_HPA"] = 1
    article_inst = article_inst.merge(hpa_articles, on=["eid", "institution_id"], how="left")
    article_inst["flag_HPA"] = article_inst["flag_HPA"].fillna(0).astype(int)

    # Cross-institution signer (CIS), revised definition:
    # An author is CIS for a focal institution if the author signs at least one
    # publication with the focal institution and at least one other publication
    # without the focal institution. CIS is an author-institution property.
    # All attributed publications of that author for the focal institution are
    # counted as CIS output for that institution.

    aa_author_eid_afid = (
        aa.dropna(subset=["author_id", "eid", "afid"])[["author_id", "eid", "afid"]]
        .drop_duplicates()
    )
    aa_author_eid_afid["afid"] = aa_author_eid_afid["afid"].astype(str)

    author_eid_afids = (
        aa_author_eid_afid
        .groupby(["author_id", "eid"])["afid"]
        .apply(lambda x: set(map(str, x)))
        .rename("afids_in_article")
        .reset_index()
    )

    author_eid_map = defaultdict(list)
    for _, row in author_eid_afids.iterrows():
        author_eid_map[row["author_id"]].append(
            (row["eid"], set(row["afids_in_article"]))
        )

    def author_has_article_without_institution(author_id: str, institution_id: str) -> int:
        """
        Return 1 if an author linked to the focal institution has at least one
        publication in which none of the focal institution AF-IDs appear.
        """
        focal_set = institution_afid_sets.get(str(institution_id), set())
        if not focal_set:
            return 0

        for _, afids_in_article in author_eid_map.get(author_id, []):
            if len(afids_in_article & focal_set) == 0:
                return 1
        return 0

    # Author-level CIS classification by institution.
    # This table contains only authors who have signed at least one publication
    # with the focal institution, because focal_author_rows was built from
    # author-affiliation rows matched to focal institutions.
    cis_authors = (
        focal_author_rows[["institution_id", "author_id"]]
        .dropna(subset=["institution_id", "author_id"])
        .drop_duplicates(subset=["institution_id", "author_id"])
        .merge(author_name_lookup, on="author_id", how="left")
        .copy()
    )

    # National aggregate:
    # Recommended rule: NATIONAL-CIS means the author has at least one publication
    # with any focal Ecuadorian institution and at least one publication with none
    # of the focal Ecuadorian institutions. This detects movement outside the
    # focal national system, not switching among Ecuadorian focal universities.
    # If instead you want NATIONAL to detect switching among Ecuadorian universities,
    # it needs a separate definition.
    cis_authors["is_cross_institution_signer"] = cis_authors.apply(
        lambda r: author_has_article_without_institution(
            r["author_id"],
            r["institution_id"],
        ),
        axis=1,
    )

    # Propagate author-institution CIS status to all attributed articles in which
    # that author signs for the focal institution.
    focal_author_rows_cis = (
        focal_author_rows
        .merge(
            cis_authors[["institution_id", "author_id", "is_cross_institution_signer"]],
            on=["institution_id", "author_id"],
            how="left",
        )
    )
    focal_author_rows_cis["is_cross_institution_signer"] = (
        focal_author_rows_cis["is_cross_institution_signer"].fillna(0).astype(int)
    )

    # Article-level CIS flag:
    # an attributed article is CIS output if at least one focal-institution author
    # in that article is CIS for that institution.
    cis_articles = (
        focal_author_rows_cis
        .groupby(["eid", "institution_id"])["is_cross_institution_signer"]
        .max()
        .rename("flag_CIS")
        .reset_index()
    )

    article_inst = article_inst.merge(cis_articles, on=["eid", "institution_id"], how="left")
    article_inst["flag_CIS"] = article_inst["flag_CIS"].fillna(0).astype(int)

    # ------------------------------------------------------------
    # Partner institutions for IMFA and CIS authors
    # ------------------------------------------------------------
    # Goal:
    #   1) For IMFA authors: identify the other institutions that appear
    #      in the same article where the author signs for the focal institution.
    #   2) For CIS authors: identify the institutions that appear in articles
    #      where the CIS author signs without the focal institution.
    #
    # Output tables:
    #   imfa_author_partner_institutions
    #   cis_author_partner_institutions


    # Lookup for affiliation names and countries.
    if article_affiliations is not None and not article_affiliations.empty:
        affiliation_lookup = (
            article_affiliations[["afid", "affiliation_name", "affiliation_country"]]
            .dropna(subset=["afid"])
            .assign(afid=lambda d: d["afid"].astype(str))
            .drop_duplicates(subset=["afid"], keep="first")
            .rename(columns={
                "afid": "partner_afid",
                "affiliation_name": "partner_affiliation_name",
                "affiliation_country": "partner_affiliation_country",
            })
        )
    else:
        affiliation_lookup = pd.DataFrame(columns=[
            "partner_afid", "partner_affiliation_name", "partner_affiliation_country"
        ])

    # Identify whether a partner AF-ID is one of the focal institutions.
    focal_partner_lookup = (
        institutions[["afid", "institution_id", "name"]]
        .drop_duplicates()
        .rename(columns={
            "afid": "partner_afid",
            "institution_id": "partner_focal_institution_id",
            "name": "partner_focal_institution_name",
        })
    )
    focal_partner_lookup["partner_afid"] = focal_partner_lookup["partner_afid"].astype(str)

    def _add_partner_metadata(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["partner_afid"] = df["partner_afid"].astype(str)
        df = df.merge(affiliation_lookup, on="partner_afid", how="left")
        df = df.merge(focal_partner_lookup, on="partner_afid", how="left")
        df["is_partner_focal_institution"] = (
            df["partner_focal_institution_id"].notna().astype(int)
        )
        return df

    # ------------------------------------------------------------
    # IMFA author-partner institutions
    # ------------------------------------------------------------
    # Unit before aggregation:
    #   focal institution + year + article + author + partner AF-ID
    #
    # A partner AF-ID is any author AF-ID in the same article that does not
    # belong to the focal institution AF-ID set.

    author_article_afid_sets = author_eid_afids.copy()

    imfa_author_article = (
        focal_author_year[
            (focal_author_year["is_multi_affiliated_author"] == 1)
            & focal_author_year["author_id"].notna()
        ][["eid", "year", "institution_id", "name", "author_id"]]
        .drop_duplicates(subset=["eid", "year", "institution_id", "author_id"])
        .merge(author_article_afid_sets, on=["author_id", "eid"], how="left")
    )

    def _imfa_partner_afids(row):
        focal_set = institution_afid_sets.get(str(row["institution_id"]), set())
        afids = row["afids_in_article"] if isinstance(row["afids_in_article"], set) else set()
        return sorted(afids - focal_set)

    if not imfa_author_article.empty:
        imfa_author_article["partner_afids"] = imfa_author_article.apply(
            _imfa_partner_afids,
            axis=1,
        )

        imfa_partner_detail = (
            imfa_author_article
            .explode("partner_afids")
            .dropna(subset=["partner_afids"])
            .rename(columns={
                "name": "institution_name",
                "partner_afids": "partner_afid",
            })
        )

        imfa_author_partner_institutions = (
            imfa_partner_detail
            .groupby(
                [
                    "institution_id", "institution_name", "year",
                    "author_id", "partner_afid",
                ],
                dropna=False,
            )
            .agg(
                n_imfa_articles_with_partner=("eid", "nunique"),
                imfa_article_eids=("eid", lambda x: ";".join(sorted(set(map(str, x))))),
            )
            .reset_index()
            .merge(author_name_lookup, on="author_id", how="left")
        )
    else:
        imfa_author_partner_institutions = pd.DataFrame(columns=[
            "institution_id", "institution_name", "year",
            "author_id", "partner_afid", "indexed_name", "indexed_name_variants",
            "n_imfa_articles_with_partner", "imfa_article_eids",
        ])

    imfa_author_partner_institutions = _add_partner_metadata(
        imfa_author_partner_institutions
    )


    # ------------------------------------------------------------
    # CIS author-partner institutions
    # ------------------------------------------------------------
    # Unit before aggregation:
    #   focal institution + external article year + CIS author + partner AF-ID
    #
    # For CIS, partner AF-IDs come from articles in which the author signs
    # without the focal institution.

    cis_author_ids = (
        cis_authors[cis_authors["is_cross_institution_signer"] == 1]
        [["institution_id", "author_id"]]
        .drop_duplicates(subset=["institution_id", "author_id"])
        .merge(author_name_lookup, on="author_id", how="left")
    )

    if not cis_author_ids.empty:
        cis_external_rows = (
            cis_author_ids
            .merge(author_eid_afids, on="author_id", how="left")
            .merge(article_years, on="eid", how="left")
        )

        def _article_lacks_focal_institution(row):
            focal_set = institution_afid_sets.get(str(row["institution_id"]), set())
            afids = row["afids_in_article"] if isinstance(row["afids_in_article"], set) else set()
            return len(afids & focal_set) == 0

        cis_external_rows["article_without_focal_institution"] = (
            cis_external_rows.apply(_article_lacks_focal_institution, axis=1)
        )

        cis_external_rows = cis_external_rows[
            cis_external_rows["article_without_focal_institution"] == True
        ].copy()

        cis_external_rows = (
            cis_external_rows
            .explode("afids_in_article")
            .dropna(subset=["afids_in_article"])
            .rename(columns={
                "afids_in_article": "partner_afid",
                "year": "external_article_year",
            })
        )

        cis_author_partner_institutions = (
            cis_external_rows
            .groupby(
                [
                    "institution_id", "external_article_year",
                    "author_id", "partner_afid",
                ],
                dropna=False,
            )
            .agg(
                n_external_articles_with_partner=("eid", "nunique"),
                external_article_eids=("eid", lambda x: ";".join(sorted(set(map(str, x))))),
            )
            .reset_index()
            .merge(author_name_lookup, on="author_id", how="left")
        )

        # Add how many focal-institution articles the CIS author contributes.
        focal_articles_by_cis_author = (
            focal_author_year
            .merge(
                cis_author_ids[["institution_id", "author_id"]],
                on=["institution_id", "author_id"],
                how="inner",
            )
            .groupby(["institution_id", "author_id"], as_index=False)
            .agg(
                n_articles_with_focal_institution=("eid", "nunique"),
                focal_article_eids=("eid", lambda x: ";".join(sorted(set(map(str, x))))),
            )
        )

        cis_author_partner_institutions = cis_author_partner_institutions.merge(
            focal_articles_by_cis_author,
            on=["institution_id", "author_id"],
            how="left",
        )

        cis_author_partner_institutions = cis_author_partner_institutions.merge(
            institution_meta.rename(columns={"name": "institution_name"}),
            on="institution_id",
            how="left",
        )
    else:
        cis_author_partner_institutions = pd.DataFrame(columns=[
            "institution_id", "institution_name", "external_article_year",
            "author_id", "partner_afid", "indexed_name", "indexed_name_variants",
            "n_external_articles_with_partner", "external_article_eids",
            "n_articles_with_focal_institution", "focal_article_eids",
        ])

    cis_author_partner_institutions = _add_partner_metadata(
        cis_author_partner_institutions
    )


    # Final safeguard: exactly one row per article-canonical institution before
    # summing article-level flags. This prevents duplicated input records from
    # inflating numerators such as N_articles_IMFA or CA_known.
    article_inst = article_inst.drop_duplicates(subset=["eid", "institution_id"], keep="first")

    affiliation_alias_correction_audit = pd.DataFrame()
    if include_audit:
        # Quantify the correction without changing the primary metrics schema.
        affiliation_alias_correction_audit = (
            article_inst
            .groupby(["institution_id", "name", "year"], dropna=False)
            .agg(
                N_articles_attributed=("eid", "nunique"),
                N_articles_MFA_raw_afid=("flag_MFA_raw_afid", "sum"),
                N_articles_MFA_corrected=("flag_MFA", "sum"),
                N_articles_MFA_alias_only=("flag_MFA_alias_only", "sum"),
                N_articles_IMFA_raw_afid=("flag_IMFA_raw_afid", "sum"),
                N_articles_IMFA_corrected=("flag_IMFA", "sum"),
                N_articles_IMFA_alias_only=("flag_IMFA_alias_only", "sum"),
            )
            .reset_index()
        )
        for prefix in ["MFA", "IMFA"]:
            for version in ["raw_afid", "corrected", "alias_only"]:
                count_col = f"N_articles_{prefix}_{version}"
                rate_col = f"{prefix}_{version}_rate"
                affiliation_alias_correction_audit[rate_col] = (
                    affiliation_alias_correction_audit[count_col]
                    / affiliation_alias_correction_audit["N_articles_attributed"]
                )

    # Composite descriptive flags use one multi-affiliation construct only.
    # IMFA is institution-specific and is the selected construct. Including
    # both MFA and IMFA would count every IMFA article twice because IMFA is
    # logically nested within MFA.
    risk_cols = ["flag_IMFA", "flag_HAO", "flag_HPA", "flag_CIS", "flag_NLO"]
    article_inst["risk_count"] = article_inst[risk_cols].sum(axis=1)
    article_inst["flag_any_risk"] = (article_inst["risk_count"] >= 1).astype(int)
    article_inst["flag_two_or_more_risks"] = (article_inst["risk_count"] >= 2).astype(int)
    article_inst["flag_IMFA_or_CIS"] = (
        (article_inst["flag_IMFA"] == 1) | (article_inst["flag_CIS"] == 1)
    ).astype(int)

    group_cols = ["institution_id", "name", "year"]
    base = (
        article_inst.groupby(group_cols, dropna=False)
        .agg(
            N_articles_attributed=("eid", "nunique"),
            N_articles_MFA=("flag_MFA", "sum"),
            N_articles_IMFA=("flag_IMFA", "sum"),
            N_articles_hyperauthored=("flag_HAO", "sum"),
            CA_known=("CA_known", "sum"),
            N_articles_CA_led=("flag_CAL", "sum"),
            N_articles_non_led=("flag_NLO", "sum"),
            N_articles_with_hyperprolific_author=("flag_HPA", "sum"),
            N_articles_CIS=("flag_CIS", "sum"),
            N_articles_IMFA_or_CIS=("flag_IMFA_or_CIS", "sum"),
            N_articles_any_risk=("flag_any_risk", "sum"),
            N_articles_two_or_more_risks=("flag_two_or_more_risks", "sum"),
        )
        .reset_index()
    )

    author_counts = (
        focal_author_year
        .drop_duplicates(subset=["eid", "institution_id", "year", "author_id"])
        .groupby(["institution_id", "year"])
        .agg(
            N_authors_institution=("author_id", "nunique"),
            N_author_article_signatures=("author_id", "size"),
        )
        .reset_index()
    )

    multi_auth_counts = (
        focal_author_year[focal_author_year["is_multi_affiliated_author"] == 1]
        .drop_duplicates(subset=["institution_id", "year", "author_id"])
        .groupby(["institution_id", "year"])
        .agg(N_multi_affiliated_authors_institution=("author_id", "nunique"))
        .reset_index()
    )

    hyper_counts = (
        author_prod[author_prod["is_hyperprolific_author"] == 1]
        .groupby(["institution_id", "year"])
        .agg(N_hyperprolific_authors=("author_id", "nunique"))
        .reset_index()
    )

    cis_author_counts = (
        focal_author_year
        .drop_duplicates(subset=["eid", "institution_id", "year", "author_id"])
        .merge(
            focal_author_rows_cis[["eid", "institution_id", "author_id", "is_cross_institution_signer"]],
            on=["eid", "institution_id", "author_id"],
            how="left"
        )
    )
    cis_author_counts = (
        cis_author_counts[cis_author_counts["is_cross_institution_signer"] == 1]
        .drop_duplicates(subset=["institution_id", "year", "author_id"])
        .groupby(["institution_id", "year"])
        .agg(N_cross_institution_signers=("author_id", "nunique"))
        .reset_index()
    )

    # New union-based top-author concentration.
    author_concentration_wide, top_author_concentration, author_productivity = compute_top_author_concentration(
        focal_author_year,
        max_top_n=10,
    )

    # Hyperprolific author diagnostics: overlap with MFA and CIS.
    hyperprolific_author_flags, hyperprolific_author_summary = compute_hyperprolific_author_diagnostics(
        author_prod=author_prod,
        focal_author_year=focal_author_year,
        focal_author_rows_cis=focal_author_rows_cis,
    )

    metrics = (
        base
        .merge(author_counts, on=["institution_id", "year"], how="left")
        .merge(multi_auth_counts, on=["institution_id", "year"], how="left")
        .merge(hyper_counts, on=["institution_id", "year"], how="left")
        .merge(cis_author_counts, on=["institution_id", "year"], how="left")
        .merge(hyperprolific_author_summary, on=["institution_id", "year"], how="left")
        .merge(author_concentration_wide, on=["institution_id", "year"], how="left")
        .merge(institution_afids, on="institution_id", how="left")
    )
    metrics["institution_afids"] = metrics["institution_afids"].apply(
        lambda x: ";".join(x) if isinstance(x, list) else x
    )

    for col in [
        "N_authors_institution", "N_author_article_signatures",
        "N_multi_affiliated_authors_institution", "N_hyperprolific_authors",
        "N_cross_institution_signers",
        "N_hyperprolific_authors_multi_affiliated_same_publication",
        "N_hyperprolific_authors_cross_institution_signers",
        "N_hyperprolific_authors_MFA_or_CIS",
    ]:
        if col in metrics.columns:
            metrics[col] = metrics[col].fillna(0).astype(int)

    # Rates.
    metrics["MFA_rate"] = metrics["N_articles_MFA"] / metrics["N_articles_attributed"]
    metrics["IMFA_rate"] = metrics["N_articles_IMFA"] / metrics["N_articles_attributed"]
    metrics["HAO_rate"] = metrics["N_articles_hyperauthored"] / metrics["N_articles_attributed"]
    metrics["HPA_rate"] = metrics["N_articles_with_hyperprolific_author"] / metrics["N_articles_attributed"]
    metrics["CIS_rate"] = metrics["N_articles_CIS"] / metrics["N_articles_attributed"]
    metrics["IMFA_or_CIS_rate"] = metrics["N_articles_IMFA_or_CIS"] / metrics["N_articles_attributed"]
    metrics["AnyRisk_rate"] = metrics["N_articles_any_risk"] / metrics["N_articles_attributed"]
    metrics["TwoOrMoreRisk_rate"] = metrics["N_articles_two_or_more_risks"] / metrics["N_articles_attributed"]
    metrics["CAL_rate"] = metrics.apply(
        lambda r: r["N_articles_CA_led"] / r["CA_known"] if r["CA_known"] else pd.NA,
        axis=1
    )
    metrics["NLO_rate"] = metrics.apply(
        lambda r: r["N_articles_non_led"] / r["CA_known"] if r["CA_known"] else pd.NA,
        axis=1
    )
    metrics["CA_unknown_or_unmatched"] = metrics["N_articles_attributed"] - metrics["CA_known"]

    metrics["hyperprolific_authors_multi_affiliated_same_publication_rate"] = metrics.apply(
        lambda r: r["N_hyperprolific_authors_multi_affiliated_same_publication"] / r["N_hyperprolific_authors"] if r.get("N_hyperprolific_authors", 0) else pd.NA,
        axis=1,
    )
    metrics["hyperprolific_authors_cross_institution_signer_rate"] = metrics.apply(
        lambda r: r["N_hyperprolific_authors_cross_institution_signers"] / r["N_hyperprolific_authors"] if r.get("N_hyperprolific_authors", 0) else pd.NA,
        axis=1,
    )
    metrics["hyperprolific_authors_MFA_or_CIS_rate"] = metrics.apply(
        lambda r: r["N_hyperprolific_authors_MFA_or_CIS"] / r["N_hyperprolific_authors"] if r.get("N_hyperprolific_authors", 0) else pd.NA,
        axis=1,
    )

    preferred = [
        "institution_id", "name", "institution_afids", "year",
        "N_articles_attributed", "N_authors_institution", "N_author_article_signatures",
        "N_articles_MFA", "MFA_rate", "N_articles_IMFA", "IMFA_rate",
        "N_multi_affiliated_authors_institution", "N_articles_hyperauthored", "HAO_rate",
        "N_hyperprolific_authors", "N_articles_with_hyperprolific_author", "HPA_rate",
        "N_hyperprolific_authors_multi_affiliated_same_publication",
        "hyperprolific_authors_multi_affiliated_same_publication_rate",
        "N_hyperprolific_authors_cross_institution_signers",
        "hyperprolific_authors_cross_institution_signer_rate",
        "N_hyperprolific_authors_MFA_or_CIS",
        "hyperprolific_authors_MFA_or_CIS_rate",
        "N_cross_institution_signers", "N_articles_CIS", "CIS_rate",
        "N_articles_IMFA_or_CIS", "IMFA_or_CIS_rate",
        "CA_known", "CA_unknown_or_unmatched", "N_articles_CA_led", "CAL_rate",
        "N_articles_non_led", "NLO_rate",
    ]
    for n in range(1, 11):
        preferred.extend([f"top_{n}_authors_article_count", f"top_{n}_authors_article_share"])
    preferred.append("N_top_authors_needed_50_percent_output_union")
    preferred.extend([
        "N_articles_any_risk", "AnyRisk_rate",
        "N_articles_two_or_more_risks", "TwoOrMoreRisk_rate",
    ])
    metrics = metrics[[c for c in preferred if c in metrics.columns]]

    # Include AF-ID list in article-level flags for audit.
    article_inst = article_inst.merge(institution_afids, on="institution_id", how="left")
    article_inst["institution_afids"] = article_inst["institution_afids"].apply(
        lambda x: ";".join(x) if isinstance(x, list) else x
    )

    aux_tables = {
        "author_affiliation_canonicalization": author_affiliation_canonicalization,
        "affiliation_alias_correction_audit": affiliation_alias_correction_audit,
        "top_author_concentration": top_author_concentration,
        "author_productivity": author_productivity,
        "hyperprolific_author_flags": hyperprolific_author_flags,
        "imfa_author_partner_institutions": imfa_author_partner_institutions,
        "cis_author_partner_institutions": cis_author_partner_institutions,
    }

    return article_inst, metrics, aux_tables

def make_author_key(df: pd.DataFrame) -> pd.Series:
    """
    Create a stable author key for deduplication.

    Prefer Scopus Author ID. If missing, use author sequence and indexed name.
    """
    def get_col(name: str) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series([pd.NA] * len(df), index=df.index)

    author_id = get_col("author_id")
    seq = get_col("author_seq")
    name = get_col("indexed_name")

    keys = []
    for aid, sq, nm in zip(author_id, seq, name):
        if pd.notna(aid) and str(aid).strip() != "":
            keys.append(str(aid).strip())
        else:
            seq_txt = "" if pd.isna(sq) else str(sq)
            name_txt = "" if pd.isna(nm) else str(nm)
            keys.append(f"SEQ:{seq_txt}|NAME:{name_txt}")
    return pd.Series(keys, index=df.index)


def deduplicate_extracted_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Remove duplicated rows from extracted tables before matching, metrics and export.

    This is a second safeguard after input-record deduplication. It prevents duplicated
    rows from inflating audit tables or metrics if a JSON source contains repeated
    blocks within the same record.
    """
    out = {k: v.copy() for k, v in tables.items()}

    if not out.get("articles", pd.DataFrame()).empty and "eid" in out["articles"].columns:
        out["articles"] = out["articles"].drop_duplicates(subset=["eid"], keep="first")

    if not out.get("article_affiliations", pd.DataFrame()).empty:
        df = out["article_affiliations"].copy()
        subset = [c for c in ["eid", "afid"] if c in df.columns]
        if subset:
            df = df.drop_duplicates(subset=subset, keep="first")
        out["article_affiliations"] = df

    if not out.get("article_authors", pd.DataFrame()).empty:
        df = out["article_authors"].copy()
        df["_author_key"] = make_author_key(df)
        df = df.drop_duplicates(subset=["eid", "_author_key"], keep="first")
        out["article_authors"] = df.drop(columns=["_author_key"])

    if not out.get("author_affiliations", pd.DataFrame()).empty:
        df = out["author_affiliations"].copy()
        df["_author_key"] = make_author_key(df)
        df = df.drop_duplicates(subset=["eid", "_author_key", "afid"], keep="first")
        out["author_affiliations"] = df.drop(columns=["_author_key"])


    if not out.get("correspondences_raw", pd.DataFrame()).empty:
        df = out["correspondences_raw"].copy()
        subset = [c for c in ["eid", "correspondence_index", "corr_indexed_name", "corr_source_text"] if c in df.columns]
        if subset:
            df = df.drop_duplicates(subset=subset, keep="first")
        out["correspondences_raw"] = df


    return out

def build_tables(records: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    article_rows, article_aff_rows, author_rows, author_aff_rows = [], [], [], []
    corr_rows = []

    for rec in records:
        art = extract_article(rec)
        eid = art.get("eid")
        if not eid:
            continue
        article_rows.append(art)
        article_aff_rows.extend(extract_article_affiliations(rec, eid))
        author_rows.extend(extract_authors_flat(rec, eid))
        author_aff_rows.extend(extract_author_affiliations_from_authors(rec, eid))
        corr_rows.extend(extract_correspondences(rec, eid))

    tables = {
        "articles": pd.DataFrame(article_rows),
        "article_affiliations": pd.DataFrame(article_aff_rows),
        "article_authors": pd.DataFrame(author_rows),
        "author_affiliations": pd.DataFrame(author_aff_rows),
        "correspondences_raw": pd.DataFrame(corr_rows),
    }

    tables = deduplicate_extracted_tables(tables)

    tables["corresponding_author_affiliations"] = match_corresponding_authors(
        tables["correspondences_raw"],
        tables["article_authors"],
        tables["author_affiliations"],
    )
    if not tables["corresponding_author_affiliations"].empty:
        tables["corresponding_author_affiliations"] = tables["corresponding_author_affiliations"].drop_duplicates(
            subset=["eid", "correspondence_index", "author_id", "afid"],
            keep="first",
        )

    return tables


def process_input(
    input_path: Path,
    analysis_year: Optional[int] = None,
    hyperauthor_threshold: int = 100,
    hyperprolific_threshold: int = 25,
    include_audit: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Read, deduplicate, attribute, compute, and return tables without exporting."""
    input_path = Path(input_path)
    analysis_year = analysis_year if analysis_year is not None else infer_analysis_year_from_path(input_path)
    if analysis_year is None:
        raise ValueError(
            "Could not infer analysis year from input JSON filename. "
            "Rename the file with the PUBYEAR/search year, e.g. 2023_abstracts.json, "
            "or pass --analysis-year 2023."
        )
    institutions = pd.DataFrame(INSTITUTIONS)
    institutions["afid"] = institutions["afid"].astype(str)
    records, duplicate_record_report, excluded_record_report, processing_summary = load_scopus_records(input_path)
    if not records:
        raise ValueError("No valid Scopus records were found after filtering and deduplication.")
    duplicate_record_report, excluded_record_report, processing_summary = apply_analysis_year_to_record_reports(
        duplicate_record_report=duplicate_record_report,
        excluded_record_report=excluded_record_report,
        processing_summary=processing_summary,
        analysis_year=analysis_year,
    )
    tables = build_tables(records)
    tables, article_year_audit = apply_analysis_year(tables, analysis_year=analysis_year)
    tables["article_year_audit"] = article_year_audit
    tables["duplicate_record_report"] = duplicate_record_report
    tables["excluded_record_report"] = excluded_record_report
    tables["record_processing_summary"] = processing_summary
    article_institution_flags, institution_year_metrics, metric_aux_tables = compute_metrics(
        articles=tables["articles"], authors=tables["article_authors"],
        author_affiliations=tables["author_affiliations"],
        article_affiliations=tables["article_affiliations"],
        corr_author_affiliations=tables["corresponding_author_affiliations"],
        institutions=institutions,
        hyperauthor_threshold=hyperauthor_threshold,
        hyperprolific_threshold=hyperprolific_threshold,
        include_audit=include_audit,
    )
    tables["article_institution_flags"] = article_institution_flags
    tables["institution_year_metrics"] = institution_year_metrics
    tables["national_baseline"] = institution_year_metrics[
        institution_year_metrics["institution_id"] == NATIONAL_INSTITUTION_ID
    ].copy() if not institution_year_metrics.empty else pd.DataFrame()
    tables.update(metric_aux_tables)
    institutions_normalized = normalize_institutions(institutions)
    tables["institutions"] = pd.concat([
        institutions_normalized,
        pd.DataFrame([{"afid": "ALL_FOCAL_AFIDS", "institution_id": NATIONAL_INSTITUTION_ID,
                       "name": NATIONAL_INSTITUTION_NAME}]),
    ], ignore_index=True)
    return tables


def load_institutions(path):
    """Read local associations; the supplied example uses fictitious institutions.

    Multiple AF-IDs may share one institution_id. Use the exact same internal
    institution_id in the private display mapping supplied to the table stage.
    Neither local configuration is copied to the public figure directory.
    """
    records = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(records, list) or not records:
        raise ValueError("Institution definitions must be a nonempty JSON array")
    result = []
    seen = set()
    names = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"afid", "institution_id", "name"}:
            raise ValueError("Each institution definition needs afid, institution_id and name")
        row = {k: str(v).strip() for k, v in record.items()}
        if not all(row.values()) or not row['afid'].isdigit() or row['afid'] in seen:
            raise ValueError("AF-IDs must be unique digit strings and fields must be nonempty")
        if row['institution_id'] == NATIONAL_INSTITUTION_ID:
            raise ValueError("NATIONAL is reserved for the pooled aggregate")
        if row['institution_id'] in names and names[row['institution_id']] != row['name']:
            raise ValueError("Aliases of one institution must have the same name")
        seen.add(row['afid']); names[row['institution_id']] = row['name']; result.append(row)
    return result


