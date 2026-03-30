# =============================================================================
# CELL — Sanity vs Best LSTM Comparison Table
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
rows = [
    # Section header, (sanity value, best value)
    ("PERFORMANCE",               "",          ""),
    ("MAE (Val)",                 "—",         f"{mae_v:.3f}"),
    ("RMSE (Val)",                "—",         f"{rmse_v:.3f}"),
    ("R² (Val)",                  "—",         f"{r2_v:.4f}"),
    ("MAE (Test)",                f"{mae_sanity:.3f}",  f"{mae_t:.3f}"),
    ("RMSE (Test)",               f"{rmse_sanity:.3f}", f"{rmse_t:.3f}"),
    ("R² (Test)",                 f"{r2_sanity:.4f}",   f"{r2_t:.4f}"),
    ("Val→Test R² Drop",          "—",         f"{r2_v - r2_t:.4f}"),
    ("% Within 10 Cycles (Test)", f"{np.mean(np.abs(y_true_sanity - y_pred_sanity) <= 10)*100:.1f}%",
                                  f"{np.mean(np.abs(y_test_lstm   - y_pred_test_lstm) <= 10)*100:.1f}%"),

    ("COMPLEXITY",                "",          ""),
    ("Parameters",                f"{count_parameters(sanity_lstm):,}",
                                  f"{count_parameters(best_lstm_model):,}"),

    ("CONFIGURATION",             "",          ""),
    ("Seq Length",                "30",        f"{best_lstm_config['seq_len']}"),
    ("Hidden Size",               "64",        f"{best_lstm_config['hidden_size']}"),
    ("Num Layers",                "1",         f"{best_lstm_config['num_layers']}"),
    ("Dropout",                   "0.2",       f"{best_lstm_config['dropout_rate']}"),
    ("Learning Rate",             "1e-3",      f"{best_lstm_config['lr']:.0e}"),
    ("Weight Decay",              "0",         f"{best_lstm_config['weight_decay']:.0e}"),
    ("Attention",                 "False",     f"{best_lstm_config['use_attention']}"),
    ("Loss Alpha",                "1.0",       f"{best_lstm_config['alpha']}"),
]

# ── Layout ────────────────────────────────────────────────────────────────────
col_labels  = ["", "Sanity LSTM", "Best LSTM"]
n_rows      = len(rows)

fig, ax = plt.subplots(figsize=(9, n_rows * 0.42 + 0.8))
ax.axis("off")

# Colours
HEADER_BG   = "#000000"
HEADER_FG   = "#FFFFFF"
SECTION_BG  = "#555555"
SECTION_FG  = "#FFFFFF"
ROW_ODD     = "#FFFFFF"
ROW_EVEN    = "#EBEBEB"
BEST_FG     = "#000000"
COL_WIDTHS  = [0.42, 0.29, 0.29]
col_x       = [0.0, 0.42, 0.71]  # left edges of each column

def cell_text(ax, x, y, w, h, text, bg, fg, bold=False, fontsize=11, align="center"):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="square,pad=0",
        facecolor=bg, edgecolor="#CCCCCC", linewidth=0.5,
        transform=ax.transAxes, clip_on=False
    ))
    ax.text(
        x + (0.01 if align == "left" else w / 2),
        y + h / 2, text,
        ha=align, va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color=fg,
        transform=ax.transAxes, clip_on=False
    )

row_h = 1.0 / (n_rows + 1)  # +1 for header

# Header row
y = 1.0 - row_h
for i, (label, cw) in enumerate(zip(col_labels, COL_WIDTHS)):
    cell_text(ax, col_x[i], y, cw, row_h, label, HEADER_BG, HEADER_FG,
              bold=True, fontsize=12, align="left" if i == 0 else "center")

# Data rows
for r_idx, (metric, sanity_val, best_val) in enumerate(rows):
    y = 1.0 - row_h * (r_idx + 2)
    is_section = sanity_val == "" and best_val == ""

    if is_section:
        bg = SECTION_BG
        cell_text(ax, col_x[0], y, 1.0, row_h, f"  {metric}", bg, SECTION_FG,
                  bold=True, fontsize=11, align="left")
    else:
        bg = ROW_ODD if r_idx % 2 == 0 else ROW_EVEN
        cell_text(ax, col_x[0], y, COL_WIDTHS[0], row_h, f"  {metric}", bg, "#000000",
                  fontsize=11, align="left")
        cell_text(ax, col_x[1], y, COL_WIDTHS[1], row_h, sanity_val, bg, "#000000",
                  fontsize=11)
        cell_text(ax, col_x[2], y, COL_WIDTHS[2], row_h, best_val, bg, "#000000",
                  bold=True, fontsize=11)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Sanity LSTM vs Best LSTM Full Comparison",
             fontsize=13, fontweight="bold", pad=12, color="#1A1A1A")

plt.tight_layout()
plt.savefig("lstm_comp_graphs/comparison_table.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: lstm_comp_graphs/comparison_table.png")

# ── Export to CSV ─────────────────────────────────────────────────────────────
import csv

csv_rows = [r for r in rows if not (r[1] == "" and r[2] == "")]  # drop section headers

with open("lstm_comp_graphs/comparison_table.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Sanity LSTM", "Best LSTM"])
    writer.writerows(csv_rows)

print("Saved: lstm_comp_graphs/comparison_table.csv")
