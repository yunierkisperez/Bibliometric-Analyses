#!/usr/bin/env python3
"""Create the supplementary temporal-deviation figures for Figure 2.

Figure S1 moves the derived IMFA-or-CIS indicator out of the main text. It uses
the same annual university-only robust-D screening, denominator threshold,
mapped colors, and separate NATIONAL line as Figure 2. Upper signals satisfy
D >= 2, lower deviations satisfy D <= -2, and a black outline identifies
absolute D >= 3.

Figure S2 summarizes persistent deviations in the six main-text indicators.
Rows are institutions with an annual robust deviation in at least two years for
at least one indicator. Cell labels give upper/lower counts and mark any extreme
deviation; cell colors give the median signed robust D across deviation years.
"""

from __future__ import annotations

import figure_style
import output_utils

import argparse
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


SCRIPT_VERSION = "2.0-robust-D-2026-08-20"
NATIONAL_ID = "NATIONAL"
ROBUST_NORMAL_CONSISTENCY = 1.349
MAIN_METRICS = (
    "IMFA",
    "CIS",
    "HPA",
    "HAO",
    "NLO",
    "HPA_MFA_or_CIS_articles",
)
HEATMAP_LABELS = {
    "IMFA": "IMFA",
    "CIS": "EC-CIS",
    "HPA": "HPA",
    "HAO": "HAO",
    "NLO": "NLO",
    "HPA_MFA_or_CIS_articles": "HPA-MFA/EC-CIS\noutput dependency",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--mapping-file", required=True)
    parser.add_argument(
        "--main-signals-file",
        "--main-outliers-file",
        dest="main_signals_file",
        required=True,
        help=(
            "Annual robust-deviation CSV produced by the Figure 2 script. "
            "The legacy --main-outliers-file option is accepted as an alias."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--mapping-encoding", default="latin1")
    parser.add_argument("--min-denominator", type=int, default=50)
    parser.add_argument("--min-box-institutions", type=int, default=5)
    parser.add_argument(
        "--min-reference-institutions",
        "--min-outlier-institutions",
        dest="min_reference_institutions",
        type=int,
        default=8,
    )
    parser.add_argument("--elevated-d", type=float, default=2.0)
    parser.add_argument("--extreme-d", type=float, default=3.0)
    parser.add_argument("--persistent-min-years", type=int, default=2)
    parser.add_argument("--x-tick-step", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.end_year < args.start_year:
        parser.error("--end-year must not precede --start-year")
    if min(
        args.min_denominator,
        args.min_box_institutions,
        args.min_reference_institutions,
        args.persistent_min_years,
        args.x_tick_step,
    ) < 1:
        parser.error("Thresholds and --x-tick-step must be positive")
    if args.elevated_d <= 0:
        parser.error("--elevated-d must be positive")
    if args.extreme_d < args.elevated_d:
        parser.error("--extreme-d must be at least --elevated-d")
    return args


def parse_code_order(value: object) -> int:
    match = re.search(r"(\d+)$", str(value).strip())
    return int(match.group(1)) if match else 10**9


def load_mapping(path: Path, encoding: str) -> pd.DataFrame:
    mapping = pd.read_csv(path, encoding=encoding)
    required = {"institution_id", "name", "id-paper", "color_hex"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    mapping = mapping.copy()
    mapping["institution_id"] = mapping["institution_id"].astype(str).str.strip()
    mapping["id-paper"] = mapping["id-paper"].astype(str).str.strip()
    mapping["color_hex"] = mapping["color_hex"].astype(str).str.strip().str.upper()
    if not mapping["color_hex"].str.fullmatch(r"#[0-9A-F]{6}").all():
        raise ValueError("All mapping colors must use #RRGGBB")
    if mapping["institution_id"].duplicated().any():
        raise ValueError("The mapping contains duplicated institution IDs")
    mapping["code_order"] = mapping["id-paper"].map(parse_code_order)
    return mapping


def load_imfa_or_cis(
    input_dir: Path,
    mapping: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = (
        "institution_id",
        "name",
        "year",
        "N_articles_attributed",
        "N_articles_IMFA_or_CIS",
        "IMFA_or_CIS_rate",
    )
    frames: list[pd.DataFrame] = []
    loaded: list[dict[str, object]] = []
    for year in range(start_year, end_year + 1):
        path = input_dir / f"{year}_institution_year_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing exact annual file: {path}")
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        missing = set(required_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        frame = frame[list(required_columns)].copy()
        frame["institution_id"] = frame["institution_id"].astype(str).str.strip()
        frame["name"] = frame["name"].astype(str).str.strip()
        if frame["institution_id"].duplicated().any():
            raise ValueError(f"{path.name} contains duplicated institution IDs")
        if frame["institution_id"].eq(NATIONAL_ID).sum() != 1:
            raise ValueError(f"{path.name} must contain exactly one NATIONAL row")
        for column in (
            "year",
            "N_articles_attributed",
            "N_articles_IMFA_or_CIS",
            "IMFA_or_CIS_rate",
        ):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        invalid = (
            frame["N_articles_IMFA_or_CIS"].lt(0)
            | frame["N_articles_IMFA_or_CIS"].gt(frame["N_articles_attributed"])
            | ~frame["IMFA_or_CIS_rate"].between(0, 1)
        )
        if invalid.any():
            raise ValueError(f"{path.name} contains invalid IMFA-or-CIS values")
        expected = frame["N_articles_IMFA_or_CIS"].div(
            frame["N_articles_attributed"].replace(0, np.nan)
        )
        mismatch = frame["IMFA_or_CIS_rate"].sub(expected).abs().gt(1e-9)
        if mismatch.any():
            raise ValueError(f"{path.name} fails the rate identity check")
        frame["analysis_year"] = year
        frame["table_year"] = frame["year"]
        frame["source_file"] = path.name
        frames.append(frame)
        loaded.append(
            {
                "analysis_year": year,
                "source_file": path.name,
                "rows": len(frame),
                "institution_rows": int(frame["institution_id"].ne(NATIONAL_ID).sum()),
                "off_year_rows": int(frame["year"].ne(year).sum()),
            }
        )
    data = pd.concat(frames, ignore_index=True)
    missing_ids = sorted(set(data["institution_id"]) - set(mapping["institution_id"]))
    if missing_ids:
        raise ValueError(f"Institutions absent from mapping: {missing_ids}")
    data = data.merge(
        mapping[["institution_id", "id-paper", "color_hex", "code_order"]],
        on="institution_id",
        how="left",
        validate="many_to_one",
    )
    data["value_plot"] = 100.0 * data["IMFA_or_CIS_rate"]
    national = data.loc[
        data["institution_id"].eq(NATIONAL_ID),
        ["analysis_year", "value_plot"],
    ].rename(columns={"value_plot": "national_value_plot"})
    data = data.merge(national, on="analysis_year", how="left", validate="many_to_one")
    data["gap_display_units"] = data["value_plot"] - data["national_value_plot"]
    return data, pd.DataFrame(loaded)


def classify_imfa_or_cis(
    data: pd.DataFrame,
    min_denominator: int,
    min_box_institutions: int,
    min_reference_institutions: int,
    elevated_d: float,
    extreme_d: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    institutional = data.loc[
        data["institution_id"].ne(NATIONAL_ID)
        & data["N_articles_attributed"].ge(min_denominator)
        & data["value_plot"].notna()
    ].copy()
    institutional["eligible"] = True
    rows: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    for year in sorted(data["analysis_year"].unique()):
        frame = institutional.loc[institutional["analysis_year"].eq(year)].copy()
        values = frame["value_plot"]
        n = len(values)
        distinct = int(values.nunique())
        q1 = float(values.quantile(0.25)) if n else np.nan
        median = float(values.median()) if n else np.nan
        q3 = float(values.quantile(0.75)) if n else np.nan
        iqr = q3 - q1 if n else np.nan
        robust_scale = iqr / ROBUST_NORMAL_CONSISTENCY if pd.notna(iqr) and iqr > 0 else np.nan
        box_drawn = bool(n >= min_box_institutions and distinct >= 2)
        rule_applied = bool(
            n >= min_reference_institutions and distinct >= 2 and pd.notna(iqr) and iqr > 0
        )
        if rule_applied:
            frame["robust_D"] = (frame["value_plot"] - median) / robust_scale
            signals = frame.loc[
                frame["robust_D"].ge(elevated_d)
                | frame["robust_D"].le(-elevated_d)
            ].copy()
        else:
            frame["robust_D"] = np.nan
            signals = frame.iloc[:0].copy()
        if not signals.empty:
            signals["deviation_side"] = np.where(
                signals["robust_D"].ge(elevated_d), "upper", "lower"
            )
            signals["extreme_deviation"] = signals["robust_D"].abs().ge(extreme_d)
            signals["deviation_strength"] = np.where(
                signals["extreme_deviation"], "extreme", "elevated"
            )
            signals["reference_institutions"] = n
            signals["reference_distinct_values"] = distinct
            signals["median_value"] = median
            signals["iqr_value"] = iqr
            signals["robust_scale_iqr_over_1_349"] = robust_scale
            signals["elevated_D_threshold"] = elevated_d
            signals["extreme_D_threshold"] = extreme_d
            signal_frames.append(signals)
        rows.append(
            {
                "metric_key": "IMFA_or_CIS",
                "analysis_year": year,
                "eligible_institutions": n,
                "distinct_values": distinct,
                "minimum_value": float(values.min()) if n else np.nan,
                "q1_value": q1,
                "median_value": median,
                "q3_value": q3,
                "maximum_value": float(values.max()) if n else np.nan,
                "iqr_value": iqr,
                "robust_scale_iqr_over_1_349": robust_scale,
                "normal_consistency_constant": ROBUST_NORMAL_CONSISTENCY,
                "elevated_D_threshold": elevated_d,
                "extreme_D_threshold": extreme_d,
                "box_drawn": box_drawn,
                "reference_rule_applied": rule_applied,
                "annual_upper_signals": int(
                    signals["deviation_side"].eq("upper").sum()
                ) if not signals.empty else 0,
                "annual_lower_deviations": int(
                    signals["deviation_side"].eq("lower").sum()
                ) if not signals.empty else 0,
                "annual_extreme_deviations": int(
                    signals["extreme_deviation"].sum()
                ) if not signals.empty else 0,
                "annual_robust_deviations": len(signals),
            }
        )
    signals = (
        pd.concat(signal_frames, ignore_index=True)
        if signal_frames
        else institutional.iloc[:0].assign(
            robust_D=pd.Series(dtype=float),
            deviation_side=pd.Series(dtype=str),
            extreme_deviation=pd.Series(dtype=bool),
            deviation_strength=pd.Series(dtype=str),
        )
    )
    return pd.DataFrame(rows), signals


def style_boxplot(boxplot: dict[str, object]) -> None:
    for box in boxplot["boxes"]:
        box.set_facecolor("#D6D6D6")
        box.set_edgecolor("#777777")
        box.set_linewidth(0.7)
        box.set_alpha(0.72)
    for key in ("whiskers", "caps"):
        for artist in boxplot[key]:
            artist.set_color("#777777")
            artist.set_linewidth(0.7)
    for median in boxplot["medians"]:
        median.set_color("#3F3F3F")
        median.set_linewidth(1.45)


def symmetric_offsets(count: int, maximum: float) -> np.ndarray:
    return np.zeros(count) if count <= 1 else np.linspace(-maximum, maximum, count)


def validate_saved_figure(path: Path) -> None:
    """Fully reopen a generated figure before it is promoted to final output."""
    suffix = path.suffix.lower()
    if path.stat().st_size == 0:
        raise OSError(f"Generated an empty figure file: {path}")
    if suffix == ".png":
        plt.imread(str(path))
    elif suffix == ".svg":
        ET.parse(path)
    elif suffix == ".pdf":
        with path.open("rb") as handle:
            header = handle.read(5)
            handle.seek(max(0, path.stat().st_size - 2048))
            trailer = handle.read()
        if header != b"%PDF-" or b"%%EOF" not in trailer:
            raise OSError(f"Generated an incomplete PDF file: {path}")


def save_figure_atomically(
    figure: plt.Figure,
    output_path: Path,
    **savefig_kwargs: object,
) -> None:
    """Save, validate, and atomically replace one figure-format output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=output_path.suffix,
        dir=str(output_path.parent),
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(temporary_path, **savefig_kwargs)
        validate_saved_figure(temporary_path)
        os.replace(str(temporary_path), str(output_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_figure(figure: plt.Figure, output_base: Path, dpi: int) -> None:
    figure_style.finish(figure, output_base)
    for suffix in (".pdf", ".svg"):
        save_figure_atomically(
            figure,
            output_base.with_suffix(suffix),
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )
    save_figure_atomically(
        figure,
        output_base.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )


def plot_imfa_or_cis(
    data: pd.DataFrame,
    distribution: pd.DataFrame,
    signals: pd.DataFrame,
    mapping: pd.DataFrame,
    output_base: Path,
    start_year: int,
    end_year: int,
    min_denominator: int,
    x_tick_step: int,
    dpi: int,
) -> None:
    if output_utils.data_only():
        return
    years = list(range(start_year, end_year + 1))
    ticks = list(range(start_year, end_year + 1, x_tick_step))
    institutional = data.loc[
        data["institution_id"].ne(NATIONAL_ID)
        & data["N_articles_attributed"].ge(min_denominator)
    ].copy()
    national = data.loc[data["institution_id"].eq(NATIONAL_ID)].set_index(
        "analysis_year"
    )["value_plot"]
    stats = distribution.set_index("analysis_year")

    figure, axis = plt.subplots(figsize=(7.4, 3.65))
    for year in years:
        frame = institutional.loc[institutional["analysis_year"].eq(year)]
        row = stats.loc[year]
        if bool(row["box_drawn"]):
            box = axis.boxplot(
                [frame["value_plot"].to_numpy(dtype=float)],
                positions=[year],
                widths=0.58,
                patch_artist=True,
                showfliers=False,
                whis=(0, 100),
                manage_ticks=False,
                zorder=2,
            )
            style_boxplot(box)
        elif not frame.empty:
            offsets = symmetric_offsets(len(frame), 0.20)
            axis.scatter(
                year + offsets,
                frame["value_plot"],
                s=14,
                color="#9B9B9B",
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
            )
            axis.hlines(
                float(row["median_value"]),
                year - 0.18,
                year + 0.18,
                color="#4A4A4A",
                linewidth=1.35,
                zorder=4,
            )

    axis.plot(years, national.reindex(years), color="#111111", linewidth=2.0, zorder=6)
    for year, frame in signals.groupby("analysis_year", sort=True):
        frame = frame.sort_values(
            ["value_plot", "code_order", "institution_id"]
        ).copy()
        frame["x_plot"] = float(year) + symmetric_offsets(len(frame), 0.13)
        for row in frame.itertuples(index=False):
            extreme = bool(row.extreme_deviation)
            axis.scatter(
                row.x_plot,
                row.value_plot,
                s=32 if extreme else 28,
                c=[row.color_hex],
                marker="o" if row.deviation_side == "upper" else "v",
                edgecolor="#111111" if extreme else "white",
                linewidth=1.05 if extreme else 0.5,
                zorder=8 if extreme else 7,
            )

    axis.set_title("IMFA or EC-CIS", pad=5)
    axis.set_ylabel("Articles (%)")
    axis.set_xlabel("Year")
    axis.set_xlim(start_year - 0.55, end_year + 0.55)
    axis.set_ylim(0, 80)
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(year) for year in ticks], rotation=45, ha="right")
    axis.yaxis.set_major_locator(MultipleLocator(20))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    # Iterating over individual spines is compatible with Matplotlib 3.3;
    # list-based SpineDict indexing is only available in newer releases.
    for spine_name in ("top", "right"):
        axis.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        axis.spines[spine_name].set_color("#555555")
    axis.tick_params(length=2.4, width=0.6, color="#555555")

    style_handles = [
        Line2D([0], [0], color="#111111", linewidth=2.0, label="National"),
        Patch(
            facecolor="#D6D6D6",
            alpha=0.72,
            edgecolor="#777777",
            label="University IQR; whiskers = min-max",
        ),
        Line2D(
            [0],
            [0],
            color="#9B9B9B",
            marker="o",
            linestyle="None",
            markersize=4,
            label="Sparse year (points; no box)",
        ),
        Line2D(
            [0],
            [0],
            color="#777777",
            marker="o",
            markerfacecolor="#999999",
            linestyle="None",
            markersize=4,
            markeredgecolor="white",
            label="Upper robust signal (D ≥ 2)",
        ),
        Line2D(
            [0],
            [0],
            color="#777777",
            marker="v",
            markerfacecolor="#999999",
            linestyle="None",
            markersize=4,
            markeredgecolor="white",
            label="Lower robust deviation (D ≤ −2)",
        ),
        Line2D(
            [0],
            [0],
            color="#111111",
            marker="o",
            markerfacecolor="#BBBBBB",
            linestyle="None",
            markersize=4.5,
            markeredgewidth=1.0,
            label="Black outline: |D| ≥ 3",
        ),
    ]
    figure.legend(
        handles=style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=6.4,
        handlelength=2.0,
        columnspacing=1.0,
    )

    signal_ids = set(signals["institution_id"])
    signal_mapping = mapping.loc[mapping["institution_id"].isin(signal_ids)].sort_values(
        "code_order"
    )
    institution_handles = [
        Line2D(
            [0],
            [0],
            color=row.color_hex,
            marker="o",
            markeredgecolor="white",
            markeredgewidth=0.4,
            linestyle="None",
            markersize=4,
            label=getattr(row, "_2"),
        )
        for row in signal_mapping[
            ["institution_id", "color_hex", "id-paper"]
        ].itertuples(index=False)
    ]
    if institution_handles:
        figure.legend(
            handles=institution_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.002),
            ncol=min(10, len(institution_handles)),
            frameon=False,
            title="Institutions with annual robust deviations",
            fontsize=6.2,
            title_fontsize=6.3,
            handlelength=0.7,
            handletextpad=0.2,
            columnspacing=0.7,
        )
    figure.subplots_adjust(left=0.09, right=0.99, top=0.74, bottom=0.25)
    save_figure(figure, output_base, dpi)
    plt.close(figure)


def validate_main_signals(path: Path) -> pd.DataFrame:
    signals = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {
        "metric_key",
        "analysis_year",
        "institution_id",
        "name",
        "id-paper",
        "color_hex",
        "code_order",
        "deviation_side",
        "robust_D",
        "extreme_deviation",
        "gap_display_units",
    }
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    unexpected = sorted(set(signals["metric_key"]) - set(MAIN_METRICS))
    if unexpected:
        raise ValueError(
            "The main-signal input contains metrics not in the six-panel figure: "
            f"{unexpected}. Regenerate Figure 2 first."
        )
    duplicated = signals.duplicated(
        ["metric_key", "analysis_year", "institution_id"]
    )
    if duplicated.any():
        raise ValueError("The main-signal input contains duplicated annual records")
    return signals


def build_persistent_summary(
    signals: pd.DataFrame,
    persistent_min_years: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    grouped_rows: list[dict[str, object]] = []
    group_cols = [
        "institution_id",
        "name",
        "id-paper",
        "color_hex",
        "code_order",
        "metric_key",
    ]
    for keys, frame in signals.groupby(group_cols, sort=False, dropna=False):
        row = dict(zip(group_cols, keys))
        years = sorted(frame["analysis_year"].astype(int).unique())
        grouped_rows.append(
            {
                **row,
                "deviation_years": len(years),
                "lower_deviation_years": int(
                    frame["deviation_side"].eq("lower").sum()
                ),
                "upper_signal_years": int(
                    frame["deviation_side"].eq("upper").sum()
                ),
                "extreme_deviation_years": int(frame["extreme_deviation"].sum()),
                "first_deviation_year": min(years),
                "last_deviation_year": max(years),
                "median_robust_D": float(frame["robust_D"].median()),
                "maximum_absolute_robust_D": float(frame["robust_D"].abs().max()),
                "median_signed_gap_from_national_pp": float(
                    frame["gap_display_units"].median()
                ),
                "maximum_absolute_gap_from_national_pp": float(
                    frame["gap_display_units"].abs().max()
                ),
                "deviation_year_list": ";".join(str(year) for year in years),
            }
        )
    recurrence = pd.DataFrame(grouped_rows)
    persistent_ids = set(
        recurrence.loc[
            recurrence["deviation_years"].ge(persistent_min_years), "institution_id"
        ]
    )
    if not persistent_ids:
        raise ValueError("No institution meets the persistent-deviation threshold")

    institution_rows = (
        recurrence.loc[recurrence["institution_id"].isin(persistent_ids)]
        .groupby(
            ["institution_id", "name", "id-paper", "color_hex", "code_order"],
            as_index=False,
        )
        .agg(
            maximum_metric_deviation_years=("deviation_years", "max"),
            total_metric_deviation_years=("deviation_years", "sum"),
        )
        .sort_values(
            [
                "maximum_metric_deviation_years",
                "total_metric_deviation_years",
                "code_order",
            ],
            ascending=[False, False, True],
        )
    )
    ordered_ids = institution_rows["institution_id"].tolist()

    complete_rows: list[dict[str, object]] = []
    lookup = recurrence.set_index(["institution_id", "metric_key"])
    for institution in institution_rows.itertuples(index=False):
        for metric in MAIN_METRICS:
            key = (institution.institution_id, metric)
            base = {
                "institution_id": institution.institution_id,
                "name": institution.name,
                "id-paper": getattr(institution, "_2"),
                "color_hex": institution.color_hex,
                "code_order": institution.code_order,
                "metric_key": metric,
                "metric_label": HEATMAP_LABELS[metric].replace("\n", " "),
            }
            if key in lookup.index:
                values = lookup.loc[key]
                complete_rows.append(
                    {
                        **base,
                        "deviation_years": int(values["deviation_years"]),
                        "lower_deviation_years": int(
                            values["lower_deviation_years"]
                        ),
                        "upper_signal_years": int(values["upper_signal_years"]),
                        "extreme_deviation_years": int(
                            values["extreme_deviation_years"]
                        ),
                        "first_deviation_year": int(values["first_deviation_year"]),
                        "last_deviation_year": int(values["last_deviation_year"]),
                        "median_robust_D": float(values["median_robust_D"]),
                        "maximum_absolute_robust_D": float(
                            values["maximum_absolute_robust_D"]
                        ),
                        "median_signed_gap_from_national_pp": float(
                            values["median_signed_gap_from_national_pp"]
                        ),
                        "maximum_absolute_gap_from_national_pp": float(
                            values["maximum_absolute_gap_from_national_pp"]
                        ),
                        "deviation_year_list": values["deviation_year_list"],
                    }
                )
            else:
                complete_rows.append(
                    {
                        **base,
                        "deviation_years": 0,
                        "lower_deviation_years": 0,
                        "upper_signal_years": 0,
                        "extreme_deviation_years": 0,
                        "first_deviation_year": np.nan,
                        "last_deviation_year": np.nan,
                        "median_robust_D": np.nan,
                        "maximum_absolute_robust_D": np.nan,
                        "median_signed_gap_from_national_pp": np.nan,
                        "maximum_absolute_gap_from_national_pp": np.nan,
                        "deviation_year_list": "",
                    }
                )
    complete = pd.DataFrame(complete_rows)
    d_matrix = complete.pivot(
        index="institution_id",
        columns="metric_key",
        values="median_robust_D",
    ).reindex(index=ordered_ids, columns=MAIN_METRICS)
    count_matrix = complete.pivot(
        index="institution_id", columns="metric_key", values="deviation_years"
    ).reindex(index=ordered_ids, columns=MAIN_METRICS)
    return complete, d_matrix, count_matrix, ordered_ids


def plot_persistent_heatmap(
    summary: pd.DataFrame,
    d_matrix: pd.DataFrame,
    count_matrix: pd.DataFrame,
    ordered_ids: list[str],
    output_base: Path,
    persistent_min_years: int,
    dpi: int,
) -> None:
    if output_utils.data_only():
        return
    metadata = (
        summary[["institution_id", "id-paper", "color_hex", "code_order"]]
        .drop_duplicates("institution_id")
        .set_index("institution_id")
        .loc[ordered_ids]
    )
    values = d_matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    finite = np.abs(values[np.isfinite(values)])
    limit = max(3.0, float(np.ceil(finite.max())))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    height = max(4.0, 0.36 * len(ordered_ids) + 1.9)
    figure, axis = plt.subplots(figsize=(7.4, height))
    image = figure_style.heatmap(axis,masked, cmap="RdBu_r", norm=norm, aspect="auto")
    image.cmap.set_bad("#F3F3F3")
    axis.set_xticks(range(len(MAIN_METRICS)))
    axis.set_xticklabels([HEATMAP_LABELS[key] for key in MAIN_METRICS])
    axis.set_yticks(range(len(ordered_ids)))
    axis.set_yticklabels(metadata["id-paper"].tolist())
    axis.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
    axis.tick_params(axis="y", length=0)
    axis.set_xticks(np.arange(-0.5, len(MAIN_METRICS), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(ordered_ids), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.3)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for label, color in zip(axis.get_yticklabels(), metadata["color_hex"]):
        label.set_color(color)
        label.set_fontweight("bold")

    for row in range(len(ordered_ids)):
        for column in range(len(MAIN_METRICS)):
            count = int(count_matrix.iloc[row, column])
            if count == 0:
                continue
            robust_d = values[row, column]
            text_color = "white" if abs(robust_d) >= 0.55 * limit else "#222222"
            institution_id = ordered_ids[row]
            metric_key = MAIN_METRICS[column]
            cell = summary.loc[
                summary["institution_id"].eq(institution_id)
                & summary["metric_key"].eq(metric_key)
            ].iloc[0]
            upper = int(cell["upper_signal_years"])
            lower = int(cell["lower_deviation_years"])
            extreme = int(cell["extreme_deviation_years"])
            if upper and lower:
                label = f"{upper}U/{lower}L"
            elif upper:
                label = f"{upper}U"
            else:
                label = f"{lower}L"
            if extreme:
                label += "*"
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=7.4,
                fontweight="bold",
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
    colorbar.set_label("Median robust deviation D across deviation years")
    colorbar.outline.set_linewidth(0.5)
    axis.set_title(
        "Persistent institutional robust deviations across Figure 2 indicators",
        pad=36,
        fontsize=9.2,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.025,
        (
            "Labels: U = upper, L = lower, * = at least one |D| ≥ 3. "
            f"Rows require at least {persistent_min_years} deviation years in "
            "at least one indicator; color = median signed D."
        ),
        ha="center",
        va="bottom",
        fontsize=6.6,
    )
    figure.subplots_adjust(left=0.12, right=0.91, top=0.80, bottom=0.11)
    save_figure(figure, output_base, dpi)
    plt.close(figure)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    if not output_utils.keep(path):
        return
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    mapping_path = Path(args.mapping_file)
    main_signals_path = Path(args.main_signals_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    mapping = load_mapping(mapping_path, args.mapping_encoding)
    data, loaded = load_imfa_or_cis(input_dir, mapping, args.start_year, args.end_year)
    distribution, s1_signals = classify_imfa_or_cis(data, args.min_denominator, args.min_box_institutions, args.min_reference_institutions, args.elevated_d, args.extreme_d)
    main_signals = validate_main_signals(main_signals_path)
    summary, d_matrix, count_matrix, ordered_ids = build_persistent_summary(main_signals, args.persistent_min_years)
    annotated = data.merge(distribution[['analysis_year', 'median_value', 'iqr_value', 'robust_scale_iqr_over_1_349', 'reference_rule_applied']], on='analysis_year', how='left', validate='many_to_one')
    annotated['eligible'] = annotated['institution_id'].ne(NATIONAL_ID) & annotated['N_articles_attributed'].ge(args.min_denominator)
    annotated['robust_D'] = np.where(annotated['eligible'] & annotated['reference_rule_applied'], (annotated['value_plot'] - annotated['median_value']) / annotated['robust_scale_iqr_over_1_349'], np.nan)
    signal_lookup = s1_signals[['analysis_year', 'institution_id', 'deviation_side', 'deviation_strength', 'extreme_deviation']].copy()
    signal_lookup['annual_robust_deviation'] = True
    annotated = annotated.merge(signal_lookup, on=['analysis_year', 'institution_id'], how='left', validate='one_to_one')
    annotated['annual_robust_deviation'] = annotated['annual_robust_deviation'].eq(True)
    annotated['extreme_deviation'] = annotated['extreme_deviation'].eq(True)
    write_csv(annotated[['id-paper', 'analysis_year', 'N_articles_attributed', 'N_articles_IMFA_or_CIS', 'IMFA_or_CIS_rate', 'robust_D', 'deviation_side', 'deviation_strength']], output_dir / 'Source-Fig-S1.csv')
    write_csv(s1_signals, output_dir / 'figureS1_robust_d_imfa_or_cis_annual_deviations.csv')
    write_csv(summary, output_dir / 'figureS2_persistent_robust_deviation_table.csv')
    print(f'Calculated supplementary temporal sources in {output_dir}')


if __name__ == "__main__":
    main()
