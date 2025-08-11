import json
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter

def generate_difficulty_chart(base_path: str, output_folder: str) -> None:
    """
    Scan iteration_*/iteration_data.jsonl for `difficulty`, compute mean ± sd per round,
    and render a single professional-looking chart (PDF).

    Parameters
    ----------
    base_path : str
        Directory containing iteration_1, iteration_2, ... subfolders.
    output_folder : str
        Directory to save the chart files.
    """
    iterations: List[Dict[str, float]] = []

    i = 1
    while True:
        iteration_path = os.path.join(base_path, f"iteration_{i}")
        if not os.path.isdir(iteration_path):
            break

        candidate_examples_path = os.path.join(iteration_path, "iteration_data.jsonl")
        difficulties: List[float] = []

        if os.path.exists(candidate_examples_path):
            with open(candidate_examples_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if "difficulty" in rec and isinstance(rec["difficulty"], (int, float)):
                            difficulties.append(float(rec["difficulty"]))
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue

        if difficulties:
            arr = np.array(difficulties, dtype=float)
            iterations.append({
                "round": i,
                "mean": float(np.mean(arr)),
                "sd": float(np.std(arr, ddof=0)),  # population SD to match original
                "n": len(difficulties),
            })

        i += 1

    if not iterations:
        print("No difficulty data found in any iteration.")
        return

    # Prepare series
    rounds = [d["round"] for d in iterations]
    means = np.array([d["mean"] for d in iterations], dtype=float)
    sds   = np.array([d["sd"]   for d in iterations], dtype=float)

    # --- Plot (single, professional look; no explicit colors) ---
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 110,
    })

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Main line
    line_plot, = ax.plot(rounds, means, marker="o", linewidth=2, label="Mean difficulty")

    # Confidence band: ±1 SD
    ax.fill_between(rounds, means - sds, means + sds, alpha=0.15, label="±1 SD")

    # Annotate mean values above markers
    for x, y in zip(rounds, means):
        ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=line_plot.get_color())

    # Titles & labels
    ax.set_title("Alice – Mean Difficulty Score by Round (from Bob's Evaluation)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Difficulty")

    # Y range similar to original
    ax.set_ylim(0, 5)
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Grid (y-only), subtle; keep default color
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    # Ticks as integers for rounds
    ax.set_xticks(rounds)

    # Finance-style axes: clean spines and outward ticks
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", direction="out", length=5, width=1)

    # Legend directly below title, centered
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    # Save as high-quality PDF
    pdf_path = os.path.join(output_folder, "difficulty_chart.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Chart saved:\n- {pdf_path}")

def generate_status_distribution_charts(base_path: str, output_folder: str) -> None:
    """
    For each iteration_i/iteration_data.jsonl, collect difficulties by status and
    generate two professional charts:
      1) Equivalent: boxplot of difficulty distribution per round + bar for count
      2) Inequivalent: boxplot of difficulty distribution per round + bar for count

    Both charts are saved as PDF in output_folder.
    """
    # Normalize known labels (lower-cased) for robustness
    EQUIV_LABELS = {
        "equivalent", "proved", "eq", "equiv", "equal", "proved_equivalent"
    }
    INEQUIV_LABELS = {
        "inequivalent", "not_equivalent", "different", "differs", "refuted", "counterexample"
    }

    aggregated = []  # List of {round, equivalent_difficulties, inequivalent_difficulties}

    i = 1
    while True:
        iteration_path = os.path.join(base_path, f"iteration_{i}")
        if not os.path.isdir(iteration_path):
            break

        eq_difficulties: List[float] = []
        ineq_difficulties: List[float] = []
        data_path = os.path.join(iteration_path, "iteration_data.jsonl")

        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    status_value = str(rec.get("status", "")).strip().lower()
                    diff_val = rec.get("difficulty", None)

                    if isinstance(diff_val, (int, float)):
                        diff = float(diff_val)
                        if status_value in EQUIV_LABELS:
                            eq_difficulties.append(diff)
                        elif status_value in INEQUIV_LABELS:
                            ineq_difficulties.append(diff)

        aggregated.append({
            "round": i,
            "equivalent_difficulties": eq_difficulties,
            "inequivalent_difficulties": ineq_difficulties,
        })
        i += 1

    if not aggregated:
        print("No iterations found under the base path.")
        return

    # Use the same visual style as the mean-difficulty chart
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 110,
    })

    def _draw(status_name: str, key: str, filename_stem: str) -> None:
        # Professional, distinctive palette (axis-color coded)
        BOX_EDGE = "#1F4B99"      # deep navy (left axis)
        BOX_FACE = "#1F4B99"
        MEAN_COLOR = "#E07A5F"    # burnt orange for mean marker
        BAR_COLOR = "#6A994E"     # muted green (right axis)

        # Layout params for side-by-side grouping
        offset = 0.30
        box_width = 0.55
        bar_width = 0.50

        rounds = [entry["round"] for entry in aggregated]
        datasets = [entry[key] for entry in aggregated]
        counts = [len(ds) for ds in datasets]

        # Compute positions: boxes left, bars right of each round
        box_positions = [r - offset for r, ds in zip(rounds, datasets) if len(ds) > 0]
        box_data = [ds for ds in datasets if len(ds) > 0]
        bar_positions = [r + offset for r in rounds]

        fig, ax = plt.subplots(figsize=(10, 5.75))

        # Boxplot (difficulty) on left axis
        if box_data:
            bp = ax.boxplot(
                box_data,
                positions=box_positions,
                widths=box_width,
                showmeans=True,
                meanline=False,  # distinct mean marker
                showfliers=False,
                patch_artist=True,
                whis=(0, 100),  # whiskers at min/max
                meanprops={
                    "marker": "D",
                    "markerfacecolor": MEAN_COLOR,
                    "markeredgecolor": MEAN_COLOR,
                    "markersize": 5,
                    "alpha": 0.95,
                    "zorder": 3,
                },
                medianprops={"color": BOX_EDGE, "linewidth": 1.6, "zorder": 3},
            )
            for patch in bp["boxes"]:
                patch.set(facecolor=BOX_FACE, alpha=0.16, edgecolor=BOX_EDGE, linewidth=1.2, zorder=3)
            for whisker in bp["whiskers"]:
                whisker.set(color=BOX_EDGE, linewidth=1, zorder=3)
            for cap in bp["caps"]:
                cap.set(color=BOX_EDGE, linewidth=1, zorder=3)

            # Connect mean diamonds with a line
            mean_y = [float(np.mean(ds)) for ds in box_data]
            ax.plot(box_positions, mean_y, color=MEAN_COLOR, linewidth=1.4, alpha=0.95, zorder=2)
            for x, y in zip(box_positions, mean_y):
                ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 5), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9, color=MEAN_COLOR)

        # Primary axis: difficulty (left)
        ax.set_ylabel("Difficulty", color=BOX_EDGE)
        ax.set_ylim(0, 10)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.set_xlabel("Round")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="y", colors=BOX_EDGE)
        # Ensure both side-by-side elements fit and ticks show integer rounds only
        left_limit = (rounds[0] - offset) - max(box_width, bar_width) * 0.6
        right_limit = (rounds[-1] + offset) + max(box_width, bar_width) * 0.6
        ax.set_xlim(left_limit, right_limit)
        ax.set_xticks(rounds)
        ax.set_xticklabels([str(r) for r in rounds])

        # Secondary axis: counts per round (right)
        ax2 = ax.twinx()
        max_count = max(counts) if counts else 0
        ylim2_max = max_count * 1.15 if max_count > 0 else 1
        ax2.set_ylim(0, max(ylim2_max, 12))
        ax2.set_ylabel("Count (right axis)", color=BAR_COLOR)
        ax2.tick_params(axis="y", colors=BAR_COLOR)
        ax2.bar(bar_positions, counts, width=bar_width, color=BAR_COLOR, alpha=0.28, linewidth=0, zorder=1)

        # Annotate counts above bars
        offset_text = 0.02 * ylim2_max
        for x, c in zip(bar_positions, counts):
            if c > 0:
                pct = (c / 250.0) * 100.0
                ax2.text(x, c + offset_text, f"{c} ({pct:.1f}%)", ha="center", va="bottom", fontsize=9, color=BAR_COLOR)

        # Title and aesthetics
        ax.set_title(f"Alice – {status_name.capitalize()}: Count and Difficulty Distribution by Round")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", direction="out", length=5, width=1)

        # Legend directly beneath the title
        box_proxy = Patch(facecolor=BOX_FACE, alpha=0.16, edgecolor=BOX_EDGE, label="Difficulty (box; whiskers=min/max, left axis)")
        mean_proxy = Line2D([0], [0], color=MEAN_COLOR, marker='D', linestyle='None', label='Mean')
        count_proxy = Patch(facecolor=BAR_COLOR, alpha=0.28, edgecolor=BAR_COLOR, label="Count (right axis)")
        ax.legend(handles=[box_proxy, mean_proxy, count_proxy], frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=3)

        # Leave room for the external legend and suptitle
        fig.tight_layout(rect=[0, 0, 1, 0.90])

        pdf_path = os.path.join(output_folder, f"{filename_stem}.pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Chart saved:\n- {pdf_path}")

    _draw("equivalent", "equivalent_difficulties", "equivalent_difficulty_distribution")
    _draw("inequivalent", "inequivalent_difficulties", "inequivalent_difficulty_distribution")

def generate_status_charts_separate(base_path: str, output_folder: str) -> None:
    """
    Generate separate charts per status:
      - Difficulty-only boxplots by round (whiskers=min/max, mean markers)
      - Count-only bars by round

    Outputs four PDF files: equivalent_difficulty_only.pdf, equivalent_count_only.pdf,
    inequivalent_difficulty_only.pdf, inequivalent_count_only.pdf
    """
    EQUIV_LABELS = {"equivalent", "proved", "eq", "equiv", "equal", "proved_equivalent"}
    INEQUIV_LABELS = {"inequivalent", "not_equivalent", "different", "differs", "refuted", "counterexample"}

    aggregated = []
    i = 1
    while True:
        iteration_path = os.path.join(base_path, f"iteration_{i}")
        if not os.path.isdir(iteration_path):
            break
        eq_difficulties: List[float] = []
        ineq_difficulties: List[float] = []
        data_path = os.path.join(iteration_path, "iteration_data.jsonl")
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    status_value = str(rec.get("status", "")).strip().lower()
                    diff_val = rec.get("difficulty", None)
                    if isinstance(diff_val, (int, float)):
                        diff = float(diff_val)
                        if status_value in EQUIV_LABELS:
                            eq_difficulties.append(diff)
                        elif status_value in INEQUIV_LABELS:
                            ineq_difficulties.append(diff)
        aggregated.append({
            "round": i,
            "equivalent_difficulties": eq_difficulties,
            "inequivalent_difficulties": ineq_difficulties,
        })
        i += 1

    if not aggregated:
        print("No iterations found under the base path.")
        return

    plt.rcParams.update({
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 110,
    })

    BOX_EDGE = "#1F4B99"   # deep navy
    BOX_FACE = "#1F4B99"
    MEAN_COLOR = "#E07A5F" # burnt orange
    BAR_COLOR = "#6A994E"  # muted green

    def _draw_difficulty(status_name: str, key: str, filename_stem: str) -> None:
        rounds = [entry["round"] for entry in aggregated]
        datasets = [entry[key] for entry in aggregated]
        positions = [r for r, ds in zip(rounds, datasets) if len(ds) > 0]
        box_data = [ds for ds in datasets if len(ds) > 0]

        fig, ax = plt.subplots(figsize=(10, 5.5))
        if box_data:
            bp = ax.boxplot(
                box_data,
                positions=positions,
                widths=0.6,
                showmeans=True,
                meanline=False,
                showfliers=False,
                patch_artist=True,
                whis=(0, 100),
                meanprops={
                    "marker": "D",
                    "markerfacecolor": MEAN_COLOR,
                    "markeredgecolor": MEAN_COLOR,
                    "markersize": 5,
                    "alpha": 0.95,
                },
                medianprops={"color": BOX_EDGE, "linewidth": 1.6},
            )
            for patch in bp["boxes"]:
                patch.set(facecolor=BOX_FACE, alpha=0.16, edgecolor=BOX_EDGE, linewidth=1.2)
            for whisker in bp["whiskers"]:
                whisker.set(color=BOX_EDGE, linewidth=1)
            for cap in bp["caps"]:
                cap.set(color=BOX_EDGE, linewidth=1)

            # Connect mean diamonds with a line
            mean_y = [float(np.mean(ds)) for ds in box_data]
            ax.plot(positions, mean_y, color=MEAN_COLOR, linewidth=1.4, alpha=0.95)
            for x, y in zip(positions, mean_y):
                ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 5), textcoords="offset points",
                            ha="left", va="bottom", fontsize=9, color=MEAN_COLOR)

        ax.set_title(f"Alice – {status_name}: Difficulty Distribution by Round")
        ax.set_xlabel("Round")
        ax.set_ylabel("Difficulty")
        ax.set_ylim(0, 10)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.set_xticks(rounds)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        box_proxy = Patch(facecolor=BOX_FACE, alpha=0.16, edgecolor=BOX_EDGE, label="Box (whiskers=min/max)")
        mean_proxy = Line2D([0], [0], color=MEAN_COLOR, marker='D', linestyle='None', label='Mean')
        ax.legend(handles=[box_proxy, mean_proxy], frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=2)

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        pdf_path = os.path.join(output_folder, f"{filename_stem}.pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Chart saved:\n- {pdf_path}")

    def _draw_count(status_name: str, key: str, filename_stem: str) -> None:
        rounds = [entry["round"] for entry in aggregated]
        counts = [len(entry[key]) for entry in aggregated]
        fig, ax = plt.subplots(figsize=(10, 5.0))
        ax.bar(rounds, counts, width=0.6, color=BAR_COLOR, alpha=0.35, linewidth=0)
        for x, c in zip(rounds, counts):
            if c > 0:
                pct = (c / 250.0) * 100.0
                ax.text(x, c + 0.1, f"{c} ({pct:.1f}%)", ha="center", va="bottom", fontsize=9, color=BAR_COLOR)
        ax.set_title(f"Alice – {status_name}: Count of validated Alice generations by Round")
        ax.set_xlabel("Round")
        ax.set_ylabel("Count")
        ax.set_ylim(0, 150)
        ax.set_xticks(rounds)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        pdf_path = os.path.join(output_folder, f"{filename_stem}.pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Chart saved:\n- {pdf_path}")

    _draw_difficulty("SEQ", "equivalent_difficulties", "equivalent_difficulty_only")
    _draw_count("SEQ", "equivalent_difficulties", "equivalent_count_only")
    _draw_difficulty("SINQ", "inequivalent_difficulties", "inequivalent_difficulty_only")
    _draw_count("SINQ", "inequivalent_difficulties", "inequivalent_count_only")

# NEW: SEQ vs SINQ counts side-by-side with % labels

def generate_seq_sinq_counts_side_by_side(base_path: str, output_folder: str) -> None:
    """
    Grouped bar chart per round with SEQ (equivalent) and SINQ (inequivalent) counts side-by-side.
    Each bar is annotated with its count and percentage (of 250) for quick comparison.
    """
    EQUIV_LABELS = {"equivalent", "proved", "eq", "equiv", "equal", "proved_equivalent"}
    INEQUIV_LABELS = {"inequivalent", "not_equivalent", "different", "differs", "refuted", "counterexample"}

    per_round = []  # {round, seq_count, sinq_count}
    i = 1
    while True:
        iteration_path = os.path.join(base_path, f"iteration_{i}")
        if not os.path.isdir(iteration_path):
            break

        seq_count = 0
        sinq_count = 0
        data_path = os.path.join(iteration_path, "iteration_data.jsonl")
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    status_value = str(rec.get("status", "")).strip().lower()
                    diff_val = rec.get("difficulty", None)
                    if isinstance(diff_val, (int, float)):
                        if status_value in EQUIV_LABELS:
                            seq_count += 1
                        elif status_value in INEQUIV_LABELS:
                            sinq_count += 1

        per_round.append({
            "round": i,
            "seq": seq_count,
            "sinq": sinq_count,
        })
        i += 1

    if not per_round:
        print("No iterations found under the base path.")
        return

    plt.rcParams.update({
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 110,
    })

    rounds = [r["round"] for r in per_round]
    seq_counts = [r["seq"] for r in per_round]
    sinq_counts = [r["sinq"] for r in per_round]

    SEQ_COLOR = "#6A994E"   # muted green
    SINQ_COLOR = "#1F4B99"  # deep navy
    BAR_ALPHA = 0.35
    bar_width = 0.38

    positions_seq = [r - bar_width/2 for r in rounds]
    positions_sinq = [r + bar_width/2 for r in rounds]

    fig, ax = plt.subplots(figsize=(10, 5.25))

    ax.bar(positions_seq, seq_counts, width=bar_width, color=SEQ_COLOR, alpha=BAR_ALPHA, linewidth=0, label="SEQ")
    ax.bar(positions_sinq, sinq_counts, width=bar_width, color=SINQ_COLOR, alpha=BAR_ALPHA, linewidth=0, label="SINQ")

    y_max = max(seq_counts + sinq_counts + [0])
    y_top = max(int(y_max * 1.15) if y_max > 0 else 1, 1)
    ax.set_ylim(0, y_top)
    ax.set_xlabel("Round")
    ax.set_ylabel("Count")
    ax.set_xticks(rounds)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate counts and percentage of 250 above bars
    denom = 250.0
    offset_text = 0.02 * y_top
    for x, c in zip(positions_seq, seq_counts):
        if c > 0:
            pct = (c / denom) * 100.0
            ax.text(x, c + offset_text, f"{c} ({pct:.1f}%)", ha="center", va="bottom", fontsize=9, color=SEQ_COLOR)
    for x, c in zip(positions_sinq, sinq_counts):
        if c > 0:
            pct = (c / denom) * 100.0
            ax.text(x, c + offset_text, f"{c} ({pct:.1f}%)", ha="center", va="bottom", fontsize=9, color=SINQ_COLOR)

    ax.set_title("Alice – SEQ vs SINQ: Counts by Round")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf_path = os.path.join(output_folder, "seq_sinq_counts_side_by_side.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved:\n- {pdf_path}")

# Example usage:
# Adjust the base_output_dir to your specific path
base_output_dir = "/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SEQ_v2/output/SEQ_deepseek-ai/DeepSeek-R1-Distill-Qwen-7B_TIME20250806_SEQ_PROGRAMS500_ITERATIONS7_no_initial_adapter_random_dataset_withdiffbiasing_LR2e-4_EPOCHS3"
output_chart_dir = base_output_dir

generate_difficulty_chart(base_output_dir, output_chart_dir)

# generate_status_distribution_charts(base_output_dir, output_chart_dir)

generate_status_charts_separate(base_output_dir, output_chart_dir)

generate_seq_sinq_counts_side_by_side(base_output_dir, output_chart_dir)
