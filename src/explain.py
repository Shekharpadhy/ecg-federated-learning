"""
Explainability module — GradientExplainer + ECG-annotated SHAP visualisation.

The 200-sample beat window maps to:
  0–49   → Pre-beat baseline
  50–79  → P-wave
  80–94  → PR interval
  95–109 → QRS complex  (R-peak at centre ≈ sample 100)
  110–139→ ST segment
  140–169→ T-wave
  170–199→ Post-beat baseline
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from .config import RESULTS_DIR
from .train_baseline import run_baseline

# ── ECG anatomy ───────────────────────────────────────────────────────────────
_REGIONS = [
    (0,   50,  "Pre-beat baseline", "#aab4be"),
    (50,  80,  "P-wave",            "#4a90d9"),
    (80,  95,  "PR interval",       "#9b59b6"),
    (95,  110, "QRS complex",       "#e74c3c"),
    (110, 140, "ST segment",        "#e67e22"),
    (140, 170, "T-wave",            "#27ae60"),
    (170, 200, "Post-beat baseline","#aab4be"),
]

def _region_label(idx: int) -> tuple[str, str]:
    for start, end, name, color in _REGIONS:
        if start <= idx < end:
            return name, color
    return "Unknown", "#cccccc"


def run_explainability():
    model, X_train, X_test, _, label_encoder = run_baseline()
    model.eval()

    background = torch.tensor(X_train[:30], dtype=torch.float32)
    samples    = torch.tensor(X_test[:30],  dtype=torch.float32)

    explainer   = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(samples)

    # Average absolute SHAP across all classes → shape (200,)
    if isinstance(shap_values, list):
        mean_shap = np.abs(np.array(shap_values)).mean(axis=(0, 1))   # (200,)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)

    # ── Figure: two panels ────────────────────────────────────────────────────
    fig, (ax_ecg, ax_bar) = plt.subplots(
        1, 2, figsize=(16, 7),
        gridspec_kw={"width_ratios": [1.2, 1]}
    )
    fig.patch.set_facecolor("#f8f9fa")

    # ── Panel 1 : mean ECG beat with SHAP heat-overlay ───────────────────────
    mean_beat = X_test[:30].mean(axis=0)          # average ECG shape
    xs = np.arange(200)

    ax_ecg.set_facecolor("#f0f2f5")

    # Shade regions
    for start, end, name, color in _REGIONS:
        ax_ecg.axvspan(start, end, alpha=0.12, color=color, label=name)

    # SHAP importance as a filled area behind the signal
    shap_norm = mean_shap / mean_shap.max()
    ax_ecg.fill_between(xs, mean_beat - shap_norm * 0.3,
                         mean_beat + shap_norm * 0.3,
                         alpha=0.35, color="#e74c3c", label="SHAP importance")

    ax_ecg.plot(xs, mean_beat, color="#1a1a2e", linewidth=1.8, zorder=5)

    # Annotate key ECG landmarks
    r_peak = int(np.argmax(mean_beat[85:115])) + 85
    ax_ecg.annotate("R-peak\n(QRS)", xy=(r_peak, mean_beat[r_peak]),
                    xytext=(r_peak + 15, mean_beat[r_peak] + 0.15),
                    arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
                    fontsize=9, color="#e74c3c", fontweight="bold")

    ax_ecg.set_xlabel("ECG Sample Index (0 = beat start, 100 = R-peak)", fontsize=10)
    ax_ecg.set_ylabel("Normalised Amplitude", fontsize=10)
    ax_ecg.set_title("Mean ECG Beat + SHAP Importance Overlay", fontsize=12, fontweight="bold", pad=12)
    ax_ecg.set_xlim(0, 199)
    ax_ecg.grid(axis="y", linestyle="--", alpha=0.4)

    # Region legend
    region_patches = [mpatches.Patch(color=c, alpha=0.55, label=n)
                      for _, _, n, c in _REGIONS if n != "Pre-beat baseline"]
    region_patches.append(mpatches.Patch(color="#e74c3c", alpha=0.4, label="SHAP importance"))
    ax_ecg.legend(handles=region_patches, loc="upper left", fontsize=8, framealpha=0.8)

    # ── Panel 2 : Top-20 features bar chart, coloured by ECG region ──────────
    top_n = 20
    top_idx  = np.argsort(mean_shap)[::-1][:top_n]
    top_vals = mean_shap[top_idx]
    top_labels = []
    top_colors = []

    for idx in top_idx:
        region, color = _region_label(idx)
        top_labels.append(f"S{idx:03d}  [{region}]")
        top_colors.append(color)

    y_pos = np.arange(top_n)
    ax_bar.set_facecolor("#f0f2f5")
    bars = ax_bar.barh(y_pos, top_vals[::-1], color=top_colors[::-1],
                       edgecolor="white", linewidth=0.5, height=0.7)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(top_labels[::-1], fontsize=8.5)
    ax_bar.set_xlabel("mean(|SHAP value|)  — impact on model output", fontsize=9)
    ax_bar.set_title(f"Top {top_n} Most Important ECG Signal Points", fontsize=12,
                     fontweight="bold", pad=12)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.4)
    ax_bar.spines[["top", "right"]].set_visible(False)

    # Region colour legend
    seen = {}
    for _, _, name, color in _REGIONS:
        if name not in seen:
            seen[name] = color
    bar_patches = [mpatches.Patch(color=c, label=n) for n, c in seen.items()]
    ax_bar.legend(handles=bar_patches, loc="lower right", fontsize=7.5, framealpha=0.85)

    # ── Title & save ──────────────────────────────────────────────────────────
    fig.suptitle("ECG Arrhythmia Classification — SHAP Explainability",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = RESULTS_DIR / "shap_summary.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"Saved SHAP summary → {out_path}")

    # Also save a compact ECG-region summary
    region_importance = {}
    for start, end, name, _ in _REGIONS:
        region_importance[name] = float(mean_shap[start:end].sum())

    import json
    with open(RESULTS_DIR / "shap_region_summary.json", "w") as f:
        json.dump(region_importance, f, indent=2)
    print("Saved region importance → results/shap_region_summary.json")
