import os
import json
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


EQUIV_LABELS = {"equivalent", "proved", "eq", "equiv", "equal", "proved_equivalent"}
INEQUIV_LABELS = {"inequivalent", "not_equivalent", "different", "differs", "refuted", "counterexample"}

# Visual style consistent with charts.py
DEFAULT_RCPARAMS = {
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 110,
}

# Base colors
SEQ_COLOR = "#6A994E"   # muted green (A)
SINQ_COLOR = "#1F4B99"  # deep navy (A)
BAR_ALPHA = 0.35

# Finance-style shade variants for experiment B (no hatch)
SEQ_COLOR_B = "#A7C957"   # lighter green
SINQ_COLOR_B = "#5A7AD6"  # lighter blue


def _collect_seq_sinq_counts(base_path: str) -> List[Dict[str, int]]:
    """
    Walk iteration_*/iteration_data.jsonl under base_path and count entries with a numeric
    difficulty, split by status (equivalent vs inequivalent).

    Returns list of dicts with keys: round, seq, sinq
    """
    results: List[Dict[str, int]] = []
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

        results.append({"round": i, "seq": seq_count, "sinq": sinq_count})
        i += 1

    return results


def _label_from_path(path: str) -> str:
    tail = os.path.basename(os.path.normpath(path))
    return tail if tail else path


def generate_stacked_seq_sinq_comparison(
    base_path_a: str,
    base_path_b: str,
    output_folder: str,
    chart_title: str | None = None,
    label_a_override: str | None = None,
    label_b_override: str | None = None,
) -> None:
    """
    Create a PDF chart with stacked bars (SEQ + SINQ) for two experiments placed side-by-side per round.

    - base_path_a, base_path_b: folders containing iteration_*/iteration_data.jsonl
    - output_folder: where the resulting PDF will be saved
    - chart_title: optional custom title (placed at the very top)
    - label_a_override / label_b_override: optional legend/display labels for the two experiments
    """
    data_a = _collect_seq_sinq_counts(base_path_a)
    data_b = _collect_seq_sinq_counts(base_path_b)

    if not data_a and not data_b:
        print("No iterations found in either base path.")
        return

    # Build unified set of rounds
    rounds_a = {d["round"] for d in data_a}
    rounds_b = {d["round"] for d in data_b}
    rounds = sorted(rounds_a.union(rounds_b))

    # Map rounds to counts (default 0 if missing)
    map_a = {d["round"]: (d["seq"], d["sinq"]) for d in data_a}
    map_b = {d["round"]: (d["seq"], d["sinq"]) for d in data_b}

    seq_a = [map_a.get(r, (0, 0))[0] for r in rounds]
    sinq_a = [map_a.get(r, (0, 0))[1] for r in rounds]
    seq_b = [map_b.get(r, (0, 0))[0] for r in rounds]
    sinq_b = [map_b.get(r, (0, 0))[1] for r in rounds]

    # Visual configuration
    plt.rcParams.update(DEFAULT_RCPARAMS)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    bar_width = 0.38
    positions_a = [r - bar_width / 2 for r in rounds]
    positions_b = [r + bar_width / 2 for r in rounds]

    # Bars for experiment A (solid fill)
    ax.bar(positions_a, seq_a, width=bar_width, color=SEQ_COLOR, alpha=BAR_ALPHA, linewidth=0, label=None)
    ax.bar(positions_a, sinq_a, width=bar_width, color=SINQ_COLOR, alpha=BAR_ALPHA, linewidth=0, bottom=seq_a, label=None)

    # Bars for experiment B (finance-style shade variants, no hatch)
    ax.bar(positions_b, seq_b, width=bar_width, color=SEQ_COLOR_B, alpha=BAR_ALPHA, linewidth=0, label=None)
    ax.bar(positions_b, sinq_b, width=bar_width, color=SINQ_COLOR_B, alpha=BAR_ALPHA, linewidth=0, bottom=seq_b, label=None)

    # Y limits and grid
    totals = [sa + ia for sa, ia in zip(seq_a, sinq_a)] + [sb + ib for sb, ib in zip(seq_b, sinq_b)]
    y_max = max(totals + [0])
    y_top = max(int(y_max * 1.15) if y_max > 0 else 1, 1)
    ax.set_ylim(0, y_top)

    # Axes labels and ticks
    ax.set_xlabel("Round")
    ax.set_ylabel("Count")
    ax.set_xticks(rounds)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # X limits to ensure bars are fully visible
    left_limit = (rounds[0] - bar_width) if rounds else -0.5
    right_limit = (rounds[-1] + bar_width) if rounds else 0.5
    ax.set_xlim(left_limit, right_limit)

    # Annotations: counts and percent of 250 above segments
    denom = 250.0
    offset_text = 0.02 * y_top

    for x, c_eq, c_in in zip(positions_a, seq_a, sinq_a):
        if c_eq > 0:
            pct = (c_eq / denom) * 100.0
            ax.text(x, c_eq + offset_text, f"{c_eq} ({pct:.1f}%)", ha="center", va="bottom", fontsize=9, color=SEQ_COLOR)
        if c_in > 0:
            top = c_eq + c_in
            pct = (c_in / denom) * 100.0
            ax.text(x, top + offset_text, f"{c_in} ({pct:.1f}%)", ha="center", va="bottom", fontsize=9, color=SINQ_COLOR)

    for x, c_eq, c_in in zip(positions_b, seq_b, sinq_b):
        if c_eq > 0:
            pct = (c_eq / denom) * 100.0
            ax.text(x, c_eq + offset_text, f"{c_eq} ({pct:.1f}%)", ha="center", va="bottom", fontsize=9, color=SEQ_COLOR_B)
        if c_in > 0:
            top = c_eq + c_in
            pct = (c_in / denom) * 100.0
            ax.text(x, top + offset_text, f"{c_in} ({pct:.1f}%)", ha="center", va="bottom", fontsize=9, color=SINQ_COLOR_B)

    # Labels and title
    label_a = label_a_override or _label_from_path(base_path_a)
    label_b = label_b_override or _label_from_path(base_path_b)
    print(label_a, label_b)
    title_text = chart_title or f"Alice – Stacked SEQ/SINQ Counts by Round: {label_a} vs {label_b}"

    # Legend at the very top (2 rows, 2 columns for compactness)
    handles = [
        Patch(facecolor=SEQ_COLOR, alpha=BAR_ALPHA, label=f"{label_a} – SEQ"),
        Patch(facecolor=SINQ_COLOR, alpha=BAR_ALPHA, label=f"{label_a} – SINQ"),
        Patch(facecolor=SEQ_COLOR_B, alpha=BAR_ALPHA, label=f"{label_b} – SEQ"),
        Patch(facecolor=SINQ_COLOR_B, alpha=BAR_ALPHA, label=f"{label_b} – SINQ"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2)

    # Suptitle at the very top
    fig.suptitle(title_text, y=0.995)

    # Leave room for legend and title
    fig.tight_layout(rect=[0, 0, 1, 0.86])

    os.makedirs(output_folder, exist_ok=True)
    pdf_path = os.path.join(output_folder, "stacked_seq_sinq_counts_comparison.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Chart saved:\n- {pdf_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate stacked SEQ/SINQ count comparison chart for two experiments.")
    parser.add_argument("--base_path_a", default="/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SEQ_v2/output/SEQ_deepseek-ai/DeepSeek-R1-Distill-Qwen-7B_TIME20250810_SEQ_PROGRAMS70_ITERATIONS7_no_initial_adapter_random_dataset_withdiffbiasing_inq_LR2e-4_EPOCHS3_DIFF3", help="First experiment base folder (contains iteration_* subfolders)")
    parser.add_argument("--base_path_b", default="/home/s2652867/llm-self-play-liquidhaskell/src/Self-Play/SEQ_v2/output/SEQ_deepseek-ai/DeepSeek-R1-Distill-Qwen-7B_TIME20250810_SEQ_PROGRAMS500_ITERATIONS7_no_initial_adapter_random_dataset_withdiffbiasing_eq_LR2e-4_EPOCHS3_DIFF3", help="Second experiment base folder (contains iteration_* subfolders)")
    parser.add_argument("--out", dest="output_folder", default="", help="Output folder for the PDF (defaults to CWD)")
    parser.add_argument("--title", dest="chart_title", default="", help="Custom chart title placed at the very top")
    parser.add_argument("--label_a", dest="label_a", default="E2", help="Legend/display label for experiment A")
    parser.add_argument("--label_b", dest="label_b", default="E3", help="Legend/display label for experiment B")

    args = parser.parse_args()
    out_dir = args.output_folder or os.getcwd()
    generate_stacked_seq_sinq_comparison(
        args.base_path_a,
        args.base_path_b,
        out_dir,
        chart_title=args.chart_title,
        label_a_override=args.label_a,
        label_b_override=args.label_b,
    )
