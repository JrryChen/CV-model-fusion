#!/usr/bin/env python3
"""Overlay mean importance (HRNet / OpenPose / MediaPipe [+ ViTPose]) across fusion methods.

Loads *_importance_weights.json files from an eval_results directory and saves
importance_comparison_mean_std.png (bar chart with error bars). Supports weights
with last dimension 3 or 4 (4-model fusion including ViTPose).

Usage:
  python plot_importance_comparison.py \\
    --eval-results /path/to/eval_results \\
    --output importance_comparison_mean_std.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_METHOD_FILES: Tuple[Tuple[str, str], ...] = (
    ("MLP fusion", "mlp_fusion_importance_weights.json"),
    ("Transformer fusion", "transformer_fusion_importance_weights.json"),
    ("Internal transformer", "internal_transformer_fusion_importance_weights.json"),
    ("MLP fusion (4-way)", "mlp_fusion_4_importance_weights.json"),
    ("Transformer fusion (4-way)", "transformer_fusion_4_importance_weights.json"),
    ("Internal transformer (4-way)", "internal_transformer_fusion_4_importance_weights.json"),
    ("Uniform", "uniform_fusion_importance_weights.json"),
    ("Confidence-weighted", "confidence_weighted_fusion_importance_weights.json"),
)


def load_mean_per_baseline(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    with open(path, "r") as f:
        data = json.load(f)
    w = np.asarray(data["importance_weights"], dtype=np.float64)
    if w.ndim != 3 or w.shape[-1] not in (3, 4):
        raise ValueError(
            f"Expected (N, K, 3) or (N, K, 4) weights in {path}, got {w.shape}"
        )
    c = int(w.shape[-1])
    default_names = ["hrnet", "openpose", "mediapipe"]
    if c == 4:
        default_names = ["hrnet", "openpose", "mediapipe", "vitpose"]
    names = list(data.get("model_names", default_names))
    if len(names) != c:
        names = default_names[:c]
    avg = w.mean(axis=(0, 1))
    std = w.reshape(-1, c).std(axis=0)
    return avg, std, names, c


def _plot_group(
    methods: List[str],
    means: List[np.ndarray],
    stds: List[np.ndarray],
    baseline_labels: List[str],
    out_path: Path,
    title_suffix: str,
) -> None:
    x = np.arange(len(baseline_labels))
    n = len(methods)
    width = min(0.14, 0.9 / max(n + 1, 1))
    fig_w = max(8.0, 1.2 * len(baseline_labels) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(n, 1)))

    for i, (lab, mu, sig) in enumerate(zip(methods, means, stds)):
        offset = width * (i - (n - 1) / 2)
        ax.bar(
            x + offset,
            mu,
            width,
            yerr=sig,
            capsize=3,
            label=lab,
            color=colors[i],
            ecolor="0.35",
        )

    ax.set_ylabel("Mean importance (± std over frames×joints)")
    ax.set_xticks(x)
    ax.set_xticklabels(baseline_labels)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Fusion importance by method (test set){title_suffix}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare mean fusion importance across methods")
    p.add_argument(
        "--eval-results",
        type=Path,
        required=True,
        help="Directory containing *_importance_weights.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <eval-results>/importance_comparison_mean_std.png)",
    )
    args = p.parse_args()
    out_dir = args.eval_results
    default_out = out_dir / "importance_comparison_mean_std.png"
    out_path = args.output or default_out

    # Per-method rows: (label, mean, std, labels, num_models)
    rows: List[Tuple[str, np.ndarray, np.ndarray, List[str], int]] = []

    for label, fname in DEFAULT_METHOD_FILES:
        fp = out_dir / fname
        if not fp.is_file():
            print(f"Skip (missing): {fp}")
            continue
        m, s, bl_names, c = load_mean_per_baseline(fp)
        bl = [n.replace("_", " ").title() for n in bl_names]
        rows.append((label, m, s, bl, c))

    if not rows:
        raise SystemExit(f"No importance JSON files found under {out_dir}")

    by_c: dict[int, List[Tuple[str, np.ndarray, np.ndarray, List[str], int]]] = {}
    for row in rows:
        by_c.setdefault(row[4], []).append(row)

    if len(by_c) == 1:
        c0 = next(iter(by_c.keys()))
        methods = [r[0] for r in by_c[c0]]
        means = [r[1] for r in by_c[c0]]
        stds = [r[2] for r in by_c[c0]]
        baseline_labels = by_c[c0][0][3]
        suffix = "" if c0 == 3 else f" ({c0} models)"
        _plot_group(methods, means, stds, baseline_labels, out_path, suffix)
        return

    # Mixed 3- and 4-model JSONs: write one figure per group
    stem = out_path.stem
    parent = out_path.parent
    for c, group in sorted(by_c.items()):
        methods = [r[0] for r in group]
        means = [r[1] for r in group]
        stds = [r[2] for r in group]
        baseline_labels = group[0][3]
        path_c = parent / f"{stem}_n{c}.png"
        _plot_group(
            methods,
            means,
            stds,
            baseline_labels,
            path_c,
            f" ({c} models)",
        )
    print(
        "Note: wrote separate PNGs for mixed 3- and 4-model importance files "
        f"(see {stem}_n3.png / {stem}_n4.png under {parent})."
    )


if __name__ == "__main__":
    main()
