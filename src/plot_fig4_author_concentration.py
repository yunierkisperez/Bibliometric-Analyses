#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Section 4.4: concentration of output and dependence on leading authors.

The script creates the main Figure 4, supplementary figures, manuscript-ready
tables, and validation audits from the annual Scopus-derived CSV tables.

Python compatibility: 3.8 or later.
"""

from __future__ import absolute_import, division, print_function

import figure_style
import output_utils

import argparse
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image

import fig4_data_core as core


SCRIPT_VERSION = "2.0.1-2026-08-25"
PLOT_METRICS = (
    ("CTA_1", "CTA-1", "CTA-1", 100.0, "national_CTA_1"),
    ("CTA_5", "CTA-5", "CTA-5", 100.0, "national_CTA_5"),
    ("CTA_10", "CTA-10", "CTA-10", 100.0, "national_CTA_10"),
    (
        "A_top50",
        "A^(Top-50%)",
        r"$A^{(\mathrm{Top\!\!-\!50\%})}$",
        1.0,
        None,
    ),
)
SELECTED_2025_CODES = ("U-12", "U-24", "U-10", "U-30")


from scipy.stats import spearmanr

def quantile_summary(values):
    values = pd.Series(values).dropna().astype(float)
    if values.empty:
        return np.nan, np.nan, np.nan
    return (
        float(values.quantile(0.25)),
        float(values.median()),
        float(values.quantile(0.75)),
    )

def annual_summary(data, threshold):
    rows = []
    eligible = data.loc[data["output"].ge(threshold)].copy()
    for year in sorted(data["analysis_year"].unique()):
        group = eligible.loc[eligible["analysis_year"].eq(year)]
        row = {
            "minimum_output": threshold,
            "analysis_year": year,
            "eligible_institution_years": len(group),
            "eligible_institutions": group["institution_id"].nunique(),
        }
        for key in ("output", "CTA_1", "CTA_5", "CTA_10", "A_top50"):
            q1, median, q3 = quantile_summary(group[key])
            row["{}_q1".format(key)] = q1
            row["{}_median".format(key)] = median
            row["{}_q3".format(key)] = q3
        rows.append(row)
    return pd.DataFrame(rows)

def correlation_table(data, threshold):
    eligible = data.loc[data["output"].ge(threshold)].copy()
    rows = []
    for metric in ("CTA_1", "CTA_5", "CTA_10", "A_top50"):
        rho, p_value = spearmanr(eligible["output"], eligible[metric])
        rows.append(
            {
                "minimum_output": threshold,
                "metric": metric,
                "n_institution_years": len(eligible),
                "n_institutions": eligible["institution_id"].nunique(),
                "spearman_rho_with_output": rho,
                "two_sided_p_value_descriptive": p_value,
            }
        )
    return pd.DataFrame(rows)

def hpa_threshold_crossings(authors, mapping, start_year, end_year, threshold):
    panel = authors[
        [
            "institution_id",
            "analysis_year",
            "author_id",
            "indexed_name",
            "p_author_institution_year",
            "is_HPA_from_productivity",
            "is_multi_affiliated_or_cross_institution",
        ]
    ].copy()
    rows = []
    for end in range(start_year + 1, end_year + 1):
        start = end - 1
        left = panel.loc[panel["analysis_year"].eq(start)].rename(
            columns={
                "indexed_name": "indexed_name_start",
                "p_author_institution_year": "productivity_start",
                "is_HPA_from_productivity": "is_HPA_start",
                "is_multi_affiliated_or_cross_institution": "exposure_start",
            }
        )
        right = panel.loc[panel["analysis_year"].eq(end)].rename(
            columns={
                "indexed_name": "indexed_name_end",
                "p_author_institution_year": "productivity_end",
                "is_HPA_from_productivity": "is_HPA_end",
                "is_multi_affiliated_or_cross_institution": "exposure_end",
            }
        )
        keep_left = [
            "institution_id",
            "author_id",
            "indexed_name_start",
            "productivity_start",
            "is_HPA_start",
            "exposure_start",
        ]
        keep_right = [
            "institution_id",
            "author_id",
            "indexed_name_end",
            "productivity_end",
            "is_HPA_end",
            "exposure_end",
        ]
        joined = left[keep_left].merge(
            right[keep_right],
            on=["institution_id", "author_id"],
            how="outer",
            validate="one_to_one",
        )
        for column in (
            "productivity_start",
            "productivity_end",
            "is_HPA_start",
            "is_HPA_end",
            "exposure_start",
            "exposure_end",
        ):
            joined[column] = joined[column].fillna(0).astype(int)
        joined = joined.loc[joined["is_HPA_start"].ne(joined["is_HPA_end"])].copy()
        if joined.empty:
            continue
        joined["start_year"] = start
        joined["end_year"] = end
        joined["direction"] = np.where(joined["is_HPA_end"].eq(1), "entered_HPA", "exited_HPA")
        joined["productivity_change"] = joined["productivity_end"] - joined["productivity_start"]
        joined["near_threshold"] = np.where(
            joined["direction"].eq("entered_HPA"),
            joined["productivity_start"].between(threshold - 5, threshold - 1),
            joined["productivity_end"].between(threshold - 5, threshold - 1),
        )
        joined["affiliation_exposed_on_HPA_side"] = np.where(
            joined["direction"].eq("entered_HPA"),
            joined["exposure_end"],
            joined["exposure_start"],
        )
        rows.append(joined)
    crossings = pd.concat(rows, ignore_index=True, sort=False)
    crossings = crossings.merge(
        mapping[["institution_id", "id-paper", "name", "color_hex"]],
        on="institution_id",
        how="left",
        validate="many_to_one",
    )
    crossings = crossings.sort_values(
        ["start_year", "end_year", "id-paper", "author_id"]
    ).reset_index(drop=True)
    crossings.insert(
        0,
        "crossing_record_id",
        ["HPA-X{:03d}".format(i) for i in range(1, len(crossings) + 1)],
    )
    return crossings

def hpa_rank_summary(authors, year=2025):
    hpa = authors.loc[authors["analysis_year"].eq(year) & authors["is_HPA_from_productivity"].eq(1)].copy()
    return pd.DataFrame(
        [
            {
                "analysis_year": year,
                "HPA_author_institution_records": len(hpa),
                "institutions_with_HPA": hpa["institution_id"].nunique(),
                "HPA_records_ranked_top_5": int(hpa["author_rank"].le(5).sum()),
                "maximum_author_rank_among_HPA": int(hpa["author_rank"].max()),
                "affiliation_exposed_HPA_records": int(
                    hpa["is_multi_affiliated_or_cross_institution"].eq(1).sum()
                ),
            }
        ]
    )

def create_key_statistics(
    data,
    annual,
    national,
    evaluations,
    signals,
    persistence,
    crossings,
    threshold,
    elevated_threshold=2.0,
    extreme_threshold=3.0,
):
    rows = []
    eligible = data.loc[data["output"].ge(threshold)]
    rows.extend(
        [
            {"statistic": "Eligible institution-years at primary threshold", "value": len(eligible), "unit": "institution-years"},
            {"statistic": "Institutions represented at primary threshold", "value": eligible["institution_id"].nunique(), "unit": "institutions"},
            {"statistic": "HPA threshold crossings", "value": len(crossings), "unit": "author-institution transitions"},
            {"statistic": "HPA entries", "value": int(crossings["direction"].eq("entered_HPA").sum()), "unit": "records"},
            {"statistic": "HPA exits", "value": int(crossings["direction"].eq("exited_HPA").sum()), "unit": "records"},
            {"statistic": "HPA crossings near threshold", "value": int(crossings["near_threshold"].sum()), "unit": "records"},
            {"statistic": "HPA crossings affiliation-exposed on HPA side", "value": int(crossings["affiliation_exposed_on_HPA_side"].sum()), "unit": "records"},
            {"statistic": "Institutions with a >=3-year tie-aware leading-author sequence", "value": persistence.loc[persistence["sequence_length_years"].ge(3), "institution_id"].nunique(), "unit": "institutions"},
            {"statistic": "Institutions with a >=5-year tie-aware leading-author sequence", "value": persistence.loc[persistence["sequence_length_years"].ge(5), "institution_id"].nunique(), "unit": "institutions"},
            {"statistic": "Annual robust-D concentration signals (D>=2)", "value": len(signals), "unit": "institution-metric-years"},
            {"statistic": "Annual extreme concentration signals (D>=3)", "value": int(signals["D"].ge(extreme_threshold).sum()), "unit": "institution-metric-years"},
            {"statistic": "CTA robust-D concentration signals (D>=2)", "value": int(signals["metric"].str.startswith("CTA").sum()), "unit": "institution-metric-years"},
            {"statistic": "A_top50 reversed-direction signals (D>=2)", "value": int(signals["metric"].eq("A^(Top-50%)").sum()), "unit": "institution-years"},
            {"statistic": "Institutions with annual concentration signals (D>=2)", "value": signals["id-paper"].nunique(), "unit": "institutions"},
            {"statistic": "Years with evaluable robust-D concentration references", "value": evaluations.loc[evaluations["reference_status"].eq("evaluable"), "analysis_year"].nunique(), "unit": "years"},
        ]
    )
    for metric_label in [spec[1] for spec in PLOT_METRICS]:
        metric_signals = signals.loc[signals["metric"].eq(metric_label)]
        rows.extend(
            [
                {"statistic": "{} signals (D>=2)".format(metric_label), "value": len(metric_signals), "unit": "institution-years"},
                {"statistic": "{} extreme signals (D>=3)".format(metric_label), "value": int(metric_signals["D"].ge(extreme_threshold).sum()), "unit": "institution-years"},
            ]
        )
    for year in (2018, 2025):
        row = annual.loc[annual["analysis_year"].eq(year)].iloc[0]
        for metric, unit in (("CTA_1", "%"), ("CTA_5", "%"), ("CTA_10", "%"), ("A_top50", "authors")):
            value = row["{}_median".format(metric)]
            if metric.startswith("CTA"):
                value *= 100
            rows.append({"statistic": "Median {} in {}".format(metric, year), "value": value, "unit": unit})
        national_row = national.loc[national["analysis_year"].eq(year)].iloc[0]
        for metric in ("CTA_1", "CTA_5", "CTA_10"):
            rows.append(
                {
                    "statistic": "National {} in {}".format(metric, year),
                    "value": 100 * national_row["national_{}".format(metric)],
                    "unit": "%",
                }
            )
    return pd.DataFrame(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--mapping-file", required=True, type=Path)
    parser.add_argument("--mapping-encoding", default="utf-8-sig")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-year", default=2011, type=int)
    parser.add_argument("--end-year", default=2025, type=int)
    parser.add_argument("--expansion-end-year", default=2018, type=int)
    parser.add_argument("--minimum-output", default=50, type=int)
    parser.add_argument("--sensitivity-thresholds", default="20,100")
    parser.add_argument("--hpa-threshold", default=25, type=int)
    parser.add_argument("--minimum-box-institutions", default=5, type=int)
    parser.add_argument("--minimum-reference-institutions", default=8, type=int)
    parser.add_argument("--d-elevated-threshold", default=2.0, type=float)
    parser.add_argument("--d-extreme-threshold", default=3.0, type=float)
    parser.add_argument("--x-tick-step", default=2, type=int)
    parser.add_argument("--dpi", default=400, type=int)
    parser.add_argument("--prefix", default="section_4_4")
    return parser.parse_args(argv)


def configure_matplotlib():
    mpl.rcParams.update(
        {
            "font.family": "Source Sans 3",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def national_concentration(authors):
    """Compute national CTA from unique EID unions across all focal institutions."""
    rows = []
    for analysis_year, group in authors.groupby("analysis_year", sort=True):
        author_eids = defaultdict(set)
        national_eids = set()
        for _, row in group.iterrows():
            eid_set = set(row["article_eid_set"])
            author_eids[str(row["author_id"])].update(eid_set)
            national_eids.update(eid_set)
        ranked = sorted(
            author_eids.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
        cumulative = set()
        coverage = {}
        for rank, (_, eid_set) in enumerate(ranked, 1):
            cumulative.update(eid_set)
            if rank in (1, 5, 10):
                coverage[rank] = len(cumulative) / len(national_eids)
            if rank >= 10:
                break
        rows.append(
            {
                "analysis_year": int(analysis_year),
                "national_output": len(national_eids),
                "national_authors": len(author_eids),
                "national_CTA_1": coverage.get(1, np.nan),
                "national_CTA_5": coverage.get(5, np.nan),
                "national_CTA_10": coverage.get(10, np.nan),
            }
        )
    return pd.DataFrame(rows)


def annual_robust_d_screen(
    data,
    threshold,
    minimum_reference_institutions,
    elevated_threshold,
    extreme_threshold,
):
    """Calculate annual direction-aligned robust D values for concentration.

    Positive D always points toward greater concentration. CTA uses the upper
    direction; A_top50 uses the lower direction because fewer authors imply a
    narrower institutional author base.
    """
    eligible = data.loc[data["output"].ge(threshold)].copy()
    rows = []
    for metric, metric_label, _, _, _ in PLOT_METRICS:
        direction = -1.0 if metric == "A_top50" else 1.0
        direction_label = "lower_is_more_concentrated" if direction < 0 else "upper_is_more_concentrated"
        for analysis_year, group in eligible.groupby("analysis_year", sort=True):
            values = group[metric].dropna().astype(float)
            median = float(values.median()) if len(values) else np.nan
            q1 = float(values.quantile(0.25)) if len(values) else np.nan
            q3 = float(values.quantile(0.75)) if len(values) else np.nan
            iqr = q3 - q1 if len(values) else np.nan
            valid_reference = bool(
                len(values) >= minimum_reference_institutions
                and values.nunique() >= 2
                and np.isfinite(iqr)
                and iqr > 0
            )
            scale = iqr / 1.349 if valid_reference else np.nan
            if len(values) < minimum_reference_institutions:
                reference_status = "fewer_than_{}_institutions".format(minimum_reference_institutions)
            elif values.nunique() < 2:
                reference_status = "fewer_than_two_distinct_values"
            elif not np.isfinite(iqr) or iqr <= 0:
                reference_status = "nonpositive_IQR"
            else:
                reference_status = "evaluable"

            for _, row in group.iterrows():
                value = float(row[metric]) if pd.notna(row[metric]) else np.nan
                robust_d = direction * (value - median) / scale if valid_reference and np.isfinite(value) else np.nan
                if np.isfinite(robust_d) and robust_d >= extreme_threshold:
                    signal_level = "extreme"
                elif np.isfinite(robust_d) and robust_d >= elevated_threshold:
                    signal_level = "elevated"
                else:
                    signal_level = ""
                rows.append(
                    {
                        "analysis_year": int(analysis_year),
                        "metric": metric_label,
                        "metric_key": metric,
                        "id-paper": row["id-paper"],
                        "value": value,
                        "median": median,
                        "q1": q1,
                        "q3": q3,
                        "IQR": iqr,
                        "robust_scale_IQR_over_1_349": scale,
                        "D": robust_d,
                        "signal_level": signal_level,
                        "concentration_direction": direction_label,
                        "eligible_institutions": len(values),
                        "reference_status": reference_status,
                        "color_hex": row["color_hex"],
                    }
                )

    evaluations = pd.DataFrame(rows)
    signals = evaluations.loc[evaluations["signal_level"].ne("")].copy()
    return evaluations, signals


def robust_d_wide(evaluations):
    """Return one row per institution-year with D and signal level by metric."""
    value_wide = evaluations.pivot(
        index=["id-paper", "analysis_year"], columns="metric_key", values="D"
    ).rename(columns=lambda key: "D_{}".format(key))
    level_wide = evaluations.pivot(
        index=["id-paper", "analysis_year"], columns="metric_key", values="signal_level"
    ).rename(columns=lambda key: "D_level_{}".format(key))
    return value_wide.join(level_wide).reset_index()


def leading_author_panel(data, authors, threshold):
    eligible_keys = set(
        zip(
            data.loc[data["output"].ge(threshold), "institution_id"],
            data.loc[data["output"].ge(threshold), "analysis_year"],
        )
    )
    rows = []
    metric_lookup = data.set_index(["institution_id", "analysis_year"])
    for key, group in authors.groupby(["institution_id", "analysis_year"], sort=True):
        if key not in eligible_keys:
            continue
        group = group.sort_values(["p_author_institution_year", "author_id"], ascending=[False, True])
        maximum = int(group["p_author_institution_year"].max())
        leaders = group.loc[group["p_author_institution_year"].eq(maximum)]
        metric_row = metric_lookup.loc[key]
        for _, author in leaders.iterrows():
            rows.append(
                {
                    "institution_id": key[0],
                    "id-paper": metric_row["id-paper"],
                    "analysis_year": key[1],
                    "output": metric_row["output"],
                    "author_id": author["author_id"],
                    "indexed_name": author["indexed_name"],
                    "leading_author_productivity": maximum,
                    "leading_author_tie_count": len(leaders),
                    "is_HPA": int(author["is_HPA_from_productivity"]),
                    "is_affiliation_exposed_if_HPA": int(
                        author["is_multi_affiliated_or_cross_institution"]
                    ),
                    "CTA_1": metric_row["CTA_1"],
                    "CTA_5": metric_row["CTA_5"],
                    "CTA_10": metric_row["CTA_10"],
                    "A_top50": metric_row["A_top50"],
                }
            )
    return pd.DataFrame(rows)


def _consecutive_runs(years):
    years = sorted(set(int(y) for y in years))
    if not years:
        return []
    runs = []
    current = [years[0]]
    for year in years[1:]:
        if year == current[-1] + 1:
            current.append(year)
        else:
            runs.append(current)
            current = [year]
    runs.append(current)
    return runs


def persistence_sequences(leader_panel, minimum_length=3):
    rows = []
    for (institution_id, author_id), group in leader_panel.groupby(
        ["institution_id", "author_id"], sort=True
    ):
        for run in _consecutive_runs(group["analysis_year"]):
            if len(run) < minimum_length:
                continue
            episode = group.loc[group["analysis_year"].isin(run)].sort_values("analysis_year")
            hpa_years = int(episode["is_HPA"].sum())
            exposed_hpa_years = int(
                (
                    episode["is_HPA"].eq(1)
                    & episode["is_affiliation_exposed_if_HPA"].eq(1)
                ).sum()
            )
            rows.append(
                {
                    "institution_id": institution_id,
                    "id-paper": episode["id-paper"].iloc[0],
                    "author_id": author_id,
                    "indexed_name": episode["indexed_name"].iloc[-1],
                    "start_year": run[0],
                    "end_year": run[-1],
                    "sequence_length_years": len(run),
                    "median_CTA_1_pct": 100 * episode["CTA_1"].median(),
                    "minimum_CTA_1_pct": 100 * episode["CTA_1"].min(),
                    "maximum_CTA_1_pct": 100 * episode["CTA_1"].max(),
                    "HPA_years": hpa_years,
                    "affiliation_exposed_HPA_years": exposed_hpa_years,
                    "all_sequence_years_HPA": hpa_years == len(run),
                    "all_HPA_years_affiliation_exposed": (
                        hpa_years > 0 and exposed_hpa_years == hpa_years
                    ),
                    "tie_aware_definition": True,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["sequence_length_years", "median_CTA_1_pct", "id-paper"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def cumulative_selected_profiles(data, authors, codes, year=2025):
    lookup = data.set_index(["institution_id", "analysis_year"])
    rows = []
    code_to_id = data[["institution_id", "id-paper"]].drop_duplicates().set_index("id-paper")["institution_id"]
    for code in codes:
        institution_id = code_to_id.loc[code]
        group = authors.loc[
            authors["institution_id"].eq(institution_id)
            & authors["analysis_year"].eq(year)
        ].sort_values(["author_rank", "author_id"])
        metric = lookup.loc[(institution_id, year)]
        union = set()
        for _, author in group.iterrows():
            union.update(author["article_eid_set"])
            rank = int(author["author_rank"])
            rows.append(
                {
                    "institution_id": institution_id,
                    "id-paper": code,
                    "analysis_year": year,
                    "author_rank": rank,
                    "cumulative_article_count": len(union),
                    "cumulative_article_share": len(union) / metric["output"],
                    "A_top50": int(metric["A_top50"]),
                    "output": int(metric["output"]),
                    "color_hex": metric["color_hex"],
                }
            )
            if rank >= int(metric["A_top50"]):
                break
    return pd.DataFrame(rows)


def draw_panel_label(axis, label):
    axis.text(
        0.015,
        0.98,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    axis.set_axisbelow(True)


def save_figure_set(fig, output_base, dpi):
    """Stage, validate, and copy complete figure files to the output directory."""
    figure_style.finish(fig, output_base)
    for extension in ("pdf", "svg"):
        final_path = Path(str(output_base) + "." + extension)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="section34_figure_", suffix="." + extension
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            fig.savefig(str(temporary_path), format=extension)
            payload = temporary_path.read_bytes()
            if extension == "pdf":
                valid = (
                    len(payload) >= 1024
                    and payload.startswith(b"%PDF")
                    and b"%%EOF" in payload[-1024:]
                )
            else:
                valid = len(payload) >= 1024 and b"</svg>" in payload[-1024:]
            if not valid:
                raise IOError("Incomplete {} figure: {}".format(extension, temporary_path))
            final_path.write_bytes(payload)
            if final_path.read_bytes() != payload:
                raise IOError("Figure copy verification failed: {}".format(final_path))
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    final_png = Path(str(output_base) + ".png")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="section34_figure_", suffix=".png"
    )
    os.close(descriptor)
    temporary_png = Path(temporary_name)
    try:
        fig.savefig(str(temporary_png), format="png", dpi=dpi)
        with Image.open(str(temporary_png)) as rendered:
            rendered.verify()
        payload = temporary_png.read_bytes()
        final_png.write_bytes(payload)
        with Image.open(str(final_png)) as rendered:
            rendered.verify()
    finally:
        if temporary_png.exists():
            temporary_png.unlink()


def plot_main_figure(
    data,
    national,
    signals,
    threshold,
    minimum_box_institutions,
    start_year,
    end_year,
    x_tick_step,
    output_base,
    dpi,
):
    """Match Figure 2: annual boxes, robust-D signals, and CTA national lines."""
    if output_utils.data_only():
        return
    configure_matplotlib()
    eligible = data.loc[data["output"].ge(threshold)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 6.15))
    axes = axes.ravel()
    years = list(range(start_year, end_year + 1))
    tick_years = list(range(start_year, end_year + 1, x_tick_step))
    if tick_years[-1] != end_year:
        tick_years.append(end_year)

    for panel_index, (axis, metric_spec) in enumerate(zip(axes, PLOT_METRICS)):
        metric, metric_label, title, scale, national_column = metric_spec
        box_years = []
        box_values = []
        sparse = []
        for year in years:
            group = eligible.loc[eligible["analysis_year"].eq(year), metric].dropna()
            values = scale * group.to_numpy(dtype=float)
            if len(values) >= minimum_box_institutions:
                box_years.append(year)
                box_values.append(values)
            elif len(values):
                offsets = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else np.array([0.0])
                sparse.extend(zip(year + offsets, values))

        if box_values:
            artists = axis.boxplot(
                box_values,
                positions=box_years,
                widths=0.58,
                patch_artist=True,
                whis=(0, 100),
                showfliers=False,
                manage_ticks=False,
                medianprops={"color": "#2F2F2F", "linewidth": 1.1},
                boxprops={"facecolor": "#D9D9D9", "edgecolor": "#8C8C8C", "linewidth": 0.8},
                whiskerprops={"color": "#8C8C8C", "linewidth": 0.8},
                capprops={"color": "#8C8C8C", "linewidth": 0.8},
            )
            for box in artists["boxes"]:
                box.set_alpha(0.9)

        if sparse:
            sx, sy = zip(*sparse)
            axis.scatter(sx, sy, s=11, color="#9A9A9A", edgecolor="none", zorder=3)

        metric_signals = signals.loc[signals["metric"].eq(metric_label)]
        for _, row in metric_signals.iterrows():
            is_extreme = row["signal_level"] == "extreme"
            axis.scatter(
                row["analysis_year"],
                scale * row["value"],
                s=34,
                marker="o",
                color=row["color_hex"],
                edgecolor="#202020" if is_extreme else "#F7F7F7",
                linewidth=1.05 if is_extreme else 0.45,
                zorder=5,
            )

        maxima = [float((scale * eligible[metric]).max())]
        if national_column is not None:
            axis.plot(
                national["analysis_year"],
                scale * national[national_column],
                color="#111111",
                linewidth=2.35,
                solid_capstyle="round",
                zorder=4,
            )
            maxima.append(float((scale * national[national_column]).max()))
        maximum = max(maxima)
        padding = 4.0 if scale == 100.0 else 5.0
        axis.set_ylim(0, max(10.0, np.ceil((maximum + padding) / 10.0) * 10.0))
        axis.set_xlim(start_year - 0.55, end_year + 0.55)
        axis.set_xticks(tick_years)
        axis.set_xticklabels(tick_years, rotation=45, ha="right")
        axis.set_title(title)
        if scale == 100.0:
            axis.set_ylabel("Publications covered (%)")
        else:
            axis.set_ylabel("Authors required for 50% coverage")
        style_axis(axis)
        draw_panel_label(axis, "({})".format(chr(ord("a") + panel_index)))

    fig.text(0.5, 0.185, "Year", ha="center", fontsize=9.0)

    summary_handles = [
        Line2D([0], [0], color="#111111", linewidth=2.35, label="National (CTA panels)"),
        Patch(facecolor="#D9D9D9", edgecolor="#8C8C8C", label="University IQR; whiskers = min-max"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=5, color="#9A9A9A", label="Sparse year (points; no box)"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=5, markerfacecolor="#777777", markeredgecolor="#F7F7F7", label=r"Elevated concentration ($D\geq2$)"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=5.5, markerfacecolor="#777777", markeredgecolor="#202020", markeredgewidth=1.1, label=r"Black outline: $D\geq3$"),
    ]
    fig.legend(
        handles=summary_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.003),
        ncol=3,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.0,
    )

    if not signals.empty:
        institution_colors = (
            signals[["id-paper", "color_hex"]]
            .drop_duplicates()
            .sort_values("id-paper")
        )
        institution_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=4.5,
                color=row["color_hex"],
                label=row["id-paper"],
            )
            for _, row in institution_colors.iterrows()
        ]
        fig.legend(
            handles=institution_handles,
            title=r"Institutions with annual $D\geq2$ concentration signals",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.045),
            ncol=min(7, len(institution_handles)),
            frameon=False,
            handlelength=0.6,
            handletextpad=0.3,
            columnspacing=0.8,
            fontsize=7.5,
            title_fontsize=7.8,
        )

    fig.subplots_adjust(
        left=0.105,
        right=0.99,
        top=0.87,
        bottom=0.27,
        wspace=0.30,
        hspace=0.48,
    )
    save_figure_set(fig, output_base, dpi)
    plt.close(fig)


def plot_cumulative_profiles(cumulative, output_base, dpi):
    """Supplementary profile curves formerly shown in main Figure 4d."""
    if output_utils.data_only():
        return
    configure_matplotlib()
    fig, axis = plt.subplots(figsize=(7.35, 4.2))
    max_rank = int(cumulative["author_rank"].max())
    for code in SELECTED_2025_CODES:
        subset = cumulative.loc[cumulative["id-paper"].eq(code)]
        color = subset["color_hex"].iloc[0]
        axis.plot(
            subset["author_rank"],
            100.0 * subset["cumulative_article_share"],
            color=color,
            marker="o",
            markersize=3.3,
            markeredgecolor="#222222",
            markeredgewidth=0.35,
            linewidth=1.8,
            label=code,
        )
        crossing = subset.loc[subset["author_rank"].eq(subset["A_top50"])]
        if not crossing.empty:
            axis.scatter(
                crossing["author_rank"],
                100.0 * crossing["cumulative_article_share"],
                color=color,
                edgecolor="#111111",
                linewidth=0.8,
                s=34,
                zorder=5,
            )
    axis.axhline(50, color="#555555", linestyle="--", linewidth=0.9)
    axis.set_xlim(0.5, max_rank + 1)
    axis.set_ylim(0, 61)
    axis.set_xlabel("Number of leading authors")
    axis.set_ylabel("Cumulative publication coverage (%)")
    axis.set_title("Contrasting institutional profiles, 2025")
    axis.legend(frameon=False, ncol=2, loc="lower right")
    style_axis(axis)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.90, bottom=0.15)
    save_figure_set(fig, output_base, dpi)
    plt.close(fig)


def deidentify_persistence(persistence):
    if persistence.empty:
        return persistence.copy()
    result = persistence.drop(columns=["institution_id", "author_id", "indexed_name"], errors="ignore").copy()
    result.insert(0, "sequence_record_id", output_utils.persistence_ids(result))
    return result


def main(argv=None):
    """Calculate only the four sources consumed by Tables S7-S9 and Figure S6."""
    args = parse_args(argv)
    output_dir = args.output_dir
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    mapping = core.load_mapping(args.mapping_file, args.mapping_encoding)
    loaded_files = core.required_paths(args.input_dir, args.start_year, args.end_year)
    metrics, productivity, hyper, top, year_audit = core.load_annual_data(
        args.input_dir, mapping, args.start_year, args.end_year)
    core.validate_concentration(metrics, productivity, top)
    hpa_coverage, authors = core.derive_hpa_coverage(metrics, productivity, hyper, args.hpa_threshold)
    data = core.add_analysis_fields(metrics, hpa_coverage, args.expansion_end_year)
    national = national_concentration(authors)
    d_evaluations, d_signals = annual_robust_d_screen(
        data, args.minimum_output, args.minimum_reference_institutions,
        args.d_elevated_threshold, args.d_extreme_threshold)
    d_wide = robust_d_wide(d_evaluations)
    leader_panel = leading_author_panel(data, authors, args.minimum_output)
    persistence = persistence_sequences(leader_panel, minimum_length=3)
    cumulative = cumulative_selected_profiles(data, authors, SELECTED_2025_CODES, args.end_year)
    public_panel = data[['id-paper', 'analysis_year', 'phase', 'output', 'institution_authors',
        'CTA_1', 'CTA_5', 'CTA_10', 'A_top50', 'N_HPA_authors', 'HPA_article_share_derived',
        'N_exposed_HPA_authors', 'exposed_HPA_article_share_derived']].merge(
        d_wide, on=['id-paper', 'analysis_year'], how='left', validate='one_to_one').sort_values(['id-paper', 'analysis_year'])
    outputs = {
        'Table_S9_leading_author_persistence_deidentified.csv': deidentify_persistence(persistence),
        'section_4_4_complete_institution_year_panel_deidentified.csv': public_panel,
        'section_4_4_cumulative_selected_profiles_2025.csv': cumulative.drop(columns=['institution_id', 'color_hex'], errors='ignore'),
        'section_4_4_national_concentration_series.csv': national,
    }
    for filename, frame in outputs.items():
        frame = frame.copy()
        frame['script_version'] = SCRIPT_VERSION
        core.save_csv(frame, tables_dir / filename)
    annual_primary = annual_summary(data, args.minimum_output).merge(national, on='analysis_year', how='left', validate='many_to_one')
    correlations = correlation_table(data, args.minimum_output)
    crossings = hpa_threshold_crossings(authors, mapping, args.start_year, args.end_year, args.hpa_threshold)
    key_stats = create_key_statistics(data, annual_primary, national, d_evaluations, d_signals,
        persistence, crossings, args.minimum_output, args.d_elevated_threshold, args.d_extreme_threshold)
    output_utils.report_methodology(author_concentration_size_sensitivity=correlations,
        hpa_threshold_statistics=key_stats, hpa_final_year_ranks=hpa_rank_summary(authors, args.end_year))
    print(f'Validated {len(loaded_files)} annual inputs; calculated concentration sources in {output_dir}')


if __name__ == "__main__":
    main()
