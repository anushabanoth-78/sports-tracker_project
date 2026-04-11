"""
analysis.py
-----------
Generates graphs and a summary report from the CSV outputs of tracker_v2.py.

Usage:
    python analysis.py --out-dir output/

Produces:
    output/graph_player_count.png      — active players visible per frame
    output/graph_speed_distribution.png — speed histogram per player
    output/graph_speed_timeline.png    — each player's speed over time
    output/graph_id_lifetime.png       — how long each track ID lived (ID switch analysis)
    output/summary_report.txt          — plain-text summary for your technical report

Dependencies:
    pip install matplotlib pandas seaborn --break-system-packages
"""

import os
import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# ── Style ────────────────────────────────────────────────────────────────────
# Clean, publication-ready look that matches the assignment's professional tone
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
TRACKER_COLOR  = "#1f77b4"   # blue  — ByteTrack
COMPARE_COLOR  = "#ff7f0e"   # orange — BoT-SORT (used only if comparison CSV exists)
ACCENT         = "#d62728"   # red   — highlights / peaks
FIG_DPI        = 150


def load_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"  [skip] not found: {path}")
        return None
    df = pd.read_csv(path)
    print(f"  loaded {len(df):,} rows  ←  {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Graph 1 — Active player count over time
# Answers: "How many subjects were visible throughout the video?"
# ─────────────────────────────────────────────────────────────────────────────
def graph_player_count(df: pd.DataFrame, fps: float, out_path: str,
                       tracker_name: str = "ByteTrack") -> None:
    seconds  = df["frame"] / fps
    count    = df["active_count"]

    # Rolling average to smooth noise (window = 1 second of frames)
    window   = max(1, int(fps))
    smoothed = count.rolling(window=window, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.fill_between(seconds, count, alpha=0.18, color=TRACKER_COLOR)
    ax.plot(seconds, count,    color=TRACKER_COLOR, alpha=0.35, linewidth=0.7,
            label="Raw count")
    ax.plot(seconds, smoothed, color=TRACKER_COLOR, linewidth=1.8,
            label=f"1-second rolling avg")

    # Mark peak
    peak_idx = smoothed.idxmax()
    ax.axvline(seconds[peak_idx], color=ACCENT, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.annotate(
        f"Peak: {int(count[peak_idx])} players\n@ {seconds[peak_idx]:.0f}s",
        xy=(seconds[peak_idx], smoothed[peak_idx]),
        xytext=(seconds[peak_idx] + 2, smoothed[peak_idx] + 0.3),
        fontsize=9, color=ACCENT,
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1),
    )

    ax.set_xlabel("Video time (seconds)")
    ax.set_ylabel("Visible players")
    ax.set_title(f"Active Player Count Over Time  [{tracker_name}]", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 2 — Speed distribution (histogram + KDE)
# Answers: "What was the movement profile of players?"
# ─────────────────────────────────────────────────────────────────────────────
def graph_speed_distribution(df: pd.DataFrame, out_path: str,
                              tracker_name: str = "ByteTrack",
                              max_plausible_kmh: float = 40.0) -> None:
    # Filter out stationary (speed < 0.5) and physically impossible values
    filtered = df[(df["speed_kmh"] > 0.5) & (df["speed_kmh"] <= max_plausible_kmh)]
    speeds   = filtered["speed_kmh"]

    if speeds.empty:
        print(f"  [skip] no plausible speed data for distribution graph")
        return

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.hist(speeds, bins=40, color=TRACKER_COLOR, alpha=0.55,
            edgecolor="white", linewidth=0.4, density=True, label="Histogram")

    # KDE overlay
    from scipy.stats import gaussian_kde
    kde_x = np.linspace(speeds.min(), speeds.max(), 300)
    kde   = gaussian_kde(speeds, bw_method=0.3)
    ax.plot(kde_x, kde(kde_x), color=TRACKER_COLOR, linewidth=2, label="KDE")

    mean_speed   = speeds.mean()
    median_speed = speeds.median()
    ax.axvline(mean_speed,   color=ACCENT,   linestyle="--", linewidth=1.3,
               label=f"Mean: {mean_speed:.1f} km/h")
    ax.axvline(median_speed, color="#2ca02c", linestyle=":",  linewidth=1.3,
               label=f"Median: {median_speed:.1f} km/h")

    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Density")
    ax.set_title(f"Player Speed Distribution  [{tracker_name}]", fontweight="bold")
    ax.legend(fontsize=9)

    # Annotation: percentile bands
    p75 = speeds.quantile(0.75)
    p90 = speeds.quantile(0.90)
    ax.axvspan(p75, p90, alpha=0.08, color="orange",
               label=f"75th–90th pct ({p75:.1f}–{p90:.1f} km/h)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 3 — Speed timeline per player (top N most active players)
# Answers: "Which players were most active and when?"
# ─────────────────────────────────────────────────────────────────────────────
def graph_speed_timeline(df: pd.DataFrame, fps: float, out_path: str,
                         tracker_name: str = "ByteTrack",
                         top_n: int = 6,
                         max_plausible_kmh: float = 40.0) -> None:
    df = df[(df["speed_kmh"] > 0.5) & (df["speed_kmh"] <= max_plausible_kmh)].copy()
    df["seconds"] = df["frame"] / fps

    # Select top N players by total observation count (most consistently tracked)
    top_ids = (
        df.groupby("track_id")["speed_kmh"]
        .count()
        .nlargest(top_n)
        .index.tolist()
    )

    if not top_ids:
        print("  [skip] no sufficient speed data for timeline graph")
        return

    palette = sns.color_palette("tab10", n_colors=len(top_ids))

    fig, ax = plt.subplots(figsize=(13, 4))

    for tid, color in zip(top_ids, palette):
        sub = df[df["track_id"] == tid].sort_values("seconds")
        # Smooth per-player speed
        smooth = sub["speed_kmh"].rolling(window=10, min_periods=1).mean()
        ax.plot(sub["seconds"], smooth, linewidth=1.5,
                label=f"Player #{tid}", color=color, alpha=0.85)
        ax.fill_between(sub["seconds"], smooth, alpha=0.07, color=color)

    ax.set_xlabel("Video time (seconds)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(f"Speed Timeline — Top {top_n} Most-Tracked Players  [{tracker_name}]",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.5, ncol=2)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 4 — ID lifetime bar chart  (most important for assignment)
# Answers: "How stable were the track IDs? How many were short-lived (ID switches)?"
# This graph directly shows the marker: short-lived tracks < 8 frames = likely ID switch
# ─────────────────────────────────────────────────────────────────────────────
def graph_id_lifetime(count_df: pd.DataFrame, speed_df: pd.DataFrame,
                      out_path: str, tracker_name: str = "ByteTrack",
                      short_lived_threshold: int = 8) -> None:
    """
    Build ID lifetime from count CSV by tracking which IDs appear per frame.
    Since count_df only has totals, we use speed_df track_ids as a proxy for
    which IDs appeared. Combine both sources for maximum coverage.
    """
    id_frame_counts: dict[int, int] = defaultdict(int)

    # Count frames each track_id appears in speed log
    if speed_df is not None and not speed_df.empty:
        for tid, grp in speed_df.groupby("track_id"):
            id_frame_counts[int(tid)] = len(grp)

    if not id_frame_counts:
        print("  [skip] not enough data for ID lifetime graph")
        return

    ids       = sorted(id_frame_counts.keys())
    lifetimes = [id_frame_counts[i] for i in ids]

    colors = [
        ACCENT if lt < short_lived_threshold else TRACKER_COLOR
        for lt in lifetimes
    ]

    fig, ax = plt.subplots(figsize=(max(10, len(ids) * 0.35), 4))

    bars = ax.bar(range(len(ids)), lifetimes, color=colors, width=0.7, edgecolor="none")

    ax.axhline(short_lived_threshold, color=ACCENT, linestyle="--",
               linewidth=1.3, label=f"Short-lived threshold ({short_lived_threshold} frames)")

    # Count short vs long lived
    n_short = sum(1 for lt in lifetimes if lt < short_lived_threshold)
    n_long  = len(lifetimes) - n_short

    ax.set_xlabel("Track ID")
    ax.set_ylabel("Frames tracked (lifetime)")
    ax.set_title(
        f"Track ID Lifetimes  [{tracker_name}]\n"
        f"{n_long} stable tracks  |  {n_short} short-lived (likely ID switches, shown in red)",
        fontweight="bold"
    )

    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([f"#{i}" for i in ids], rotation=90, fontsize=7)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Bonus Graph 5 — Tracker comparison bar chart  (only if comparison CSV exists)
# ─────────────────────────────────────────────────────────────────────────────
def graph_tracker_comparison(comp_csv: str, out_path: str) -> None:
    df = load_csv(comp_csv)
    if df is None:
        return

    # Select metrics that are numeric and meaningful to compare
    metrics_to_plot = {
        "Total unique IDs":       "Total unique IDs\n(lower = better)",
        "Short-lived tracks":     "Short-lived tracks\n(lower = better)",
        "Est. true objects":      "Est. true objects\n(higher = better)",
        "Peak detections":        "Peak detections",
        "Processing speed (fps)": "Processing speed\n(fps, higher = better)",
    }

    rows = df[df["metric"].isin(metrics_to_plot.keys())].copy()
    if rows.empty:
        print("  [skip] comparison CSV doesn't have expected metrics")
        return

    rows["bytetrack"] = pd.to_numeric(rows["bytetrack"], errors="coerce")
    rows["botsort"]   = pd.to_numeric(rows["botsort"],   errors="coerce")
    rows = rows.dropna(subset=["bytetrack", "botsort"])

    n       = len(rows)
    x       = np.arange(n)
    width   = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width/2, rows["bytetrack"], width, color=TRACKER_COLOR,
           label="ByteTrack", alpha=0.85)
    ax.bar(x + width/2, rows["botsort"],   width, color=COMPARE_COLOR,
           label="BoT-SORT",  alpha=0.85)

    labels = [metrics_to_plot.get(m, m) for m in rows["metric"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Value")
    ax.set_title("ByteTrack vs BoT-SORT — Key Metric Comparison", fontweight="bold")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plain-text summary report  (paste into technical report)
# ─────────────────────────────────────────────────────────────────────────────
def write_summary_report(
    count_df: pd.DataFrame | None,
    speed_df: pd.DataFrame | None,
    out_path: str,
    tracker_name: str = "ByteTrack",
    fps: float = 30.0,
    short_lived_threshold: int = 8,
) -> None:
    lines = []
    lines.append("=" * 60)
    lines.append(f"TRACKING PIPELINE — ANALYSIS SUMMARY")
    lines.append(f"Tracker : {tracker_name}")
    lines.append("=" * 60)

    if count_df is not None:
        total_frames = count_df["frame"].max()
        video_dur_s  = total_frames / fps
        avg_count    = count_df["active_count"].mean()
        peak_count   = count_df["active_count"].max()
        peak_frame   = count_df.loc[count_df["active_count"].idxmax(), "frame"]
        lines.append(f"\n--- Detection Stats ---")
        lines.append(f"Total frames analysed : {total_frames:,}")
        lines.append(f"Video duration        : {video_dur_s:.1f}s ({video_dur_s/60:.2f} min)")
        lines.append(f"Avg players/frame     : {avg_count:.2f}")
        lines.append(f"Peak players visible  : {peak_count}  (frame {peak_frame}, "
                     f"{peak_frame/fps:.1f}s)")
        zero_frames = (count_df["active_count"] == 0).sum()
        lines.append(f"Frames with 0 players : {zero_frames:,} "
                     f"({100*zero_frames/len(count_df):.1f}%)")

    if speed_df is not None and not speed_df.empty:
        max_plausible = 40.0
        filtered = speed_df[
            (speed_df["speed_kmh"] > 0.5) & (speed_df["speed_kmh"] <= max_plausible)
        ]
        all_speeds   = filtered["speed_kmh"]
        unique_ids   = speed_df["track_id"].nunique()
        id_counts    = speed_df.groupby("track_id").size()
        short_lived  = (id_counts < short_lived_threshold).sum()
        stable       = unique_ids - short_lived

        lines.append(f"\n--- Tracking / ID Stats ---")
        lines.append(f"Total unique IDs      : {unique_ids}")
        lines.append(f"Stable tracks (≥{short_lived_threshold}f) : {stable}")
        lines.append(f"Short-lived tracks    : {short_lived}  "
                     f"(proxy for ID switches)")
        lines.append(f"ID stability rate     : {100*stable/unique_ids:.1f}%")

        lines.append(f"\n--- Speed Stats (plausible range: 0.5–{max_plausible} km/h) ---")
        lines.append(f"Observations          : {len(all_speeds):,}")
        lines.append(f"Mean speed            : {all_speeds.mean():.2f} km/h")
        lines.append(f"Median speed          : {all_speeds.median():.2f} km/h")
        lines.append(f"Std deviation         : {all_speeds.std():.2f} km/h")
        lines.append(f"Min (filtered)        : {all_speeds.min():.2f} km/h")
        lines.append(f"Max (filtered)        : {all_speeds.max():.2f} km/h")
        lines.append(f"75th percentile       : {all_speeds.quantile(0.75):.2f} km/h")
        lines.append(f"90th percentile       : {all_speeds.quantile(0.90):.2f} km/h")

        # Per-player breakdown (top 10)
        lines.append(f"\n--- Per-Player Speed Summary (top 10 most tracked) ---")
        top10 = (
            filtered.groupby("track_id")["speed_kmh"]
            .agg(["count", "mean", "max"])
            .nlargest(10, "count")
            .reset_index()
        )
        lines.append(f"{'ID':<8} {'Obs':>6} {'Avg km/h':>10} {'Max km/h':>10}")
        lines.append("-" * 38)
        for _, row in top10.iterrows():
            lines.append(
                f"#{int(row['track_id']):<7} {int(row['count']):>6} "
                f"{row['mean']:>10.1f} {row['max']:>10.1f}"
            )

    lines.append("\n" + "=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n  report saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate graphs + summary report from tracker CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir",  default="output",
                        help="Directory containing the tracker CSV outputs")
    parser.add_argument("--tracker",  default="ByteTrack",
                        choices=["ByteTrack", "BoT-SORT"])
    parser.add_argument("--fps",      type=float, default=30.0,
                        help="FPS of the original video (used for time axis)")
    parser.add_argument("--max-kmh",  type=float, default=40.0,
                        help="Max plausible speed — filters out bad calibration spikes")
    parser.add_argument("--top-n",    type=int,   default=6,
                        help="Number of top players to show in speed timeline")
    args = parser.parse_args()

    out = args.out_dir
    t   = args.tracker
    fps = args.fps

    print(f"\nGenerating analysis graphs from: {out}/")
    print(f"Tracker: {t}   FPS: {fps}\n")

    # Load CSVs
    count_df = load_csv(os.path.join(out, f"count_{t}.csv"))
    speed_df = load_csv(os.path.join(out, f"speed_{t}.csv"))

    # Graph 1 — Player count over time
    if count_df is not None:
        graph_player_count(
            count_df, fps,
            out_path     = os.path.join(out, "graph_player_count.png"),
            tracker_name = t,
        )

    # Graph 2 — Speed distribution
    if speed_df is not None:
        graph_speed_distribution(
            speed_df,
            out_path        = os.path.join(out, "graph_speed_distribution.png"),
            tracker_name    = t,
            max_plausible_kmh = args.max_kmh,
        )

    # Graph 3 — Speed timeline per player
    if speed_df is not None:
        graph_speed_timeline(
            speed_df, fps,
            out_path          = os.path.join(out, "graph_speed_timeline.png"),
            tracker_name      = t,
            top_n             = args.top_n,
            max_plausible_kmh = args.max_kmh,
        )

    # Graph 4 — ID lifetime
    if speed_df is not None:
        graph_id_lifetime(
            count_df, speed_df,
            out_path     = os.path.join(out, "graph_id_lifetime.png"),
            tracker_name = t,
        )

    # Graph 5 — Tracker comparison (only if you ran --compare)
    comp_csv = os.path.join(out, "tracker_comparison.csv")
    if os.path.exists(comp_csv):
        graph_tracker_comparison(
            comp_csv,
            out_path = os.path.join(out, "graph_tracker_comparison.png"),
        )
    else:
        print(f"  [info] no comparison CSV found — skipping graph 5")
        print(f"         (run tracker_v2.py --compare to generate it)")

    # Summary report
    write_summary_report(
        count_df     = count_df,
        speed_df     = speed_df,
        out_path     = os.path.join(out, "summary_report.txt"),
        tracker_name = t,
        fps          = fps,
    )

    print("\nAll done.")


if __name__ == "__main__":
    main()