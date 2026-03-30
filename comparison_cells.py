# =============================================================================
# CELL 1 — Setup: Get Sanity Test Predictions
# =============================================================================

_, _, (X_test_sanity, y_test_sanity) = get_tensors(30)  # sanity used seq_len=30
mae_sanity, rmse_sanity, r2_sanity, y_pred_sanity = evaluate_lstm(
    sanity_lstm, X_test_sanity, y_test_sanity
)
y_true_sanity = y_test_sanity.numpy().flatten()

print(f"Sanity LSTM  — Test  MAE={mae_sanity:.3f}  RMSE={rmse_sanity:.3f}  R²={r2_sanity:.4f}")
print(f"Best LSTM    — Test  MAE={mae_t:.3f}       RMSE={rmse_t:.3f}       R²={r2_t:.4f}")


# =============================================================================
# CELL 2 — Scatter: Predicted vs Actual RUL (Overlaid)
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 7))

# Sanity — translucent gray, behind
ax.scatter(y_true_sanity, y_pred_sanity, alpha=0.15, s=8, color="#AAAAAA", label=f"Sanity LSTM  (R²={r2_sanity:.4f})")
# Best — blue, on top
ax.scatter(y_test_lstm, y_pred_test_lstm, alpha=0.35, s=8, color="#2E86AB", label=f"Best LSTM    (R²={r2_t:.4f})")

lims = [0, max(y_true_sanity.max(), y_test_lstm.max(), y_pred_sanity.max(), y_pred_test_lstm.max()) + 5]
ax.plot(lims, lims, "k--", linewidth=1.2, label="Perfect")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Actual RUL", fontsize=12)
ax.set_ylabel("Predicted RUL", fontsize=12)
ax.set_title("Predicted vs Actual RUL — Test Set", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig("comparison_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: comparison_scatter.png")


# =============================================================================
# CELL 3 — Residual Distribution
# =============================================================================

res_sanity = y_true_sanity - y_pred_sanity
res_best   = y_test_lstm   - y_pred_test_lstm

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(res_sanity, bins=60, alpha=0.5, color="#888888", label=f"Sanity LSTM  (std={res_sanity.std():.2f})")
ax.hist(res_best,   bins=60, alpha=0.6, color="#2E86AB", label=f"Best LSTM    (std={res_best.std():.2f})")
ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
ax.set_xlabel("Residual (Actual − Predicted)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Residual Distribution (Test Set)", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("comparison_residuals_sanity_vs_best_lstm.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: comparison_residuals_sanity_vs_best_lstm.png")


# =============================================================================
# CELL 4 — CDF of Absolute Errors
# =============================================================================

abs_sanity = np.abs(res_sanity)
abs_best   = np.abs(res_best)

THRESHOLD = 10

fig, ax = plt.subplots(figsize=(10, 5))
for abs_err, label, color in [
    (abs_sanity, "Sanity LSTM", "#888888"),
    (abs_best,   "Best LSTM",   "#2E86AB"),
]:
    sorted_err = np.sort(abs_err)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    pct_within = np.mean(abs_err <= THRESHOLD) * 100
    ax.plot(sorted_err, cdf, linewidth=2, color=color, label=f"{label}  ({pct_within:.1f}% within {THRESHOLD} cycles)")

ax.axvline(10, color="black", linewidth=1, linestyle="--", alpha=0.5, label="10-cycle threshold")
ax.set_xlabel("Absolute Error (cycles)", fontsize=12)
ax.set_ylabel("Cumulative Proportion", fontsize=12)
ax.set_title("CDF of Absolute Errors — Test Set", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.set_xlim(left=0)
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig("comparison_cdf_sanity_vs_best_lstm.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: comparison_cdf_sanity_vs_best_lstm.png")


# =============================================================================
# D2 — Asymmetric Loss Function vs. MSE
# =============================================================================

alpha = 1.5
errors = np.linspace(-40, 40, 500)

mse_loss  = errors ** 2
asym_loss = np.where(errors > 0, alpha * errors**2, errors**2)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(errors, mse_loss,  color="#888888", linewidth=2.0, linestyle="--", label="Symmetric MSE  (α = 1.0)")
ax.plot(errors, asym_loss, color="#2E86AB", linewidth=2.5, label=f"Asymmetric Loss  (α = {alpha})")
ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
ax.fill_between(errors, mse_loss, asym_loss,
                where=(errors > 0), alpha=0.12, color="#E63946",
                label="Extra penalty (overprediction)")
ax.annotate("Overprediction\n(ŷ > y  →  underestimates\ntime-to-failure risk)",
            xy=(20, alpha * 20**2), xytext=(5, 1200),
            arrowprops=dict(arrowstyle="->", color="#E63946"),
            fontsize=10, color="#E63946")
ax.annotate("Underprediction\n(ŷ < y  →  conservative)",
            xy=(-20, 20**2), xytext=(-38, 1200),
            arrowprops=dict(arrowstyle="->", color="#555555"),
            fontsize=10, color="#555555")
ax.set_xlabel("Prediction Error  (ŷ − y)", fontsize=12)
ax.set_ylabel("Loss Value", fontsize=12)
ax.set_title("Asymmetric Loss vs. Symmetric MSE", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.set_xlim(-40, 40)
ax.set_ylim(0, 2600)
plt.tight_layout()
plt.savefig("asymmetric_loss_curve.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: asymmetric_loss_curve.png")


# =============================================================================
# D3 — Window Size Context Illustration
# =============================================================================

np.random.seed(42)
n_cycles = 200
cycles = np.arange(n_cycles)

# Piecewise: healthy plateau then degradation onset ~cycle 75
rul_true = np.where(cycles < 75, 125, 125 - (cycles - 75) * 1.0)
rul_true = np.clip(rul_true, 0, 125)
sensor   = 125 - rul_true + np.random.normal(0, 2, n_cycles)

# Observation point near cycle 160 (RUL ~40)
obs = 160
w30 = (obs - 30, obs)
w75 = (obs - 75, obs)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cycles, sensor, color="#555555", linewidth=1.5, alpha=0.8, label="Sensor signal (proxy)")
ax.axvline(75, color="#E63946", linewidth=1.2, linestyle="--", alpha=0.6, label="Degradation onset (~cycle 75)")

ax.axvspan(w30[0], w30[1], alpha=0.15, color="#F4A261", label=f"Window = 30  (cycles {w30[0]}–{w30[1]})")
ax.annotate("", xy=(w30[1], 95), xytext=(w30[0], 95),
            arrowprops=dict(arrowstyle="<->", color="#F4A261", lw=2))
ax.text((w30[0]+w30[1])/2, 97, "30 cycles", ha="center", fontsize=10, color="#C77A32")

ax.axvspan(w75[0], w75[1], alpha=0.08, color="#2E86AB", label=f"Window = 75  (cycles {w75[0]}–{w75[1]})")
ax.annotate("", xy=(w75[1], 108), xytext=(w75[0], 108),
            arrowprops=dict(arrowstyle="<->", color="#2E86AB", lw=2))
ax.text((w75[0]+w75[1])/2, 110, "75 cycles", ha="center", fontsize=10, color="#1a5f7a")

ax.set_xlabel("Operational Cycle", fontsize=12)
ax.set_ylabel("Sensor Signal (a.u.)", fontsize=12)
ax.set_title("Effect of Window Size on Captured Degradation Context", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
plt.tight_layout()
plt.savefig("window_size_illustration.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: window_size_illustration.png")


# =============================================================================
# D4 — Updated All-Models Bar Chart (RF / MLP / Sanity LSTM / Best LSTM)
# =============================================================================

models    = ["Random\nForest", "MLP", "Sanity\nLSTM", "Best\nLSTM"]
colors    = ["#A8DADC", "#A8DADC", "#7FB3C8", "#2E86AB"]
rmse_test = [17.18,  17.285, 16.960, 12.951]
r2_test   = [0.612,  0.607,  0.674,  0.839]
mae_test  = [12.378, 13.39,  12.066,  9.942]

x = np.arange(len(models))

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Test-Set Performance: All Models", fontsize=14, fontweight="bold")

for ax, vals, ylabel, title, fmt in zip(
    axes,
    [rmse_test, r2_test, mae_test],
    ["RMSE (cycles)", "R²", "MAE (cycles)"],
    ["Test RMSE ↓", "Test R² ↑", "Test MAE ↓"],
    [".2f", ".3f", ".2f"]
):
    bars = ax.bar(x, vals, color=colors, edgecolor="#333333", linewidth=0.8, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(vals),
                f"{v:{fmt}}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("all_models_bar_comparison_final.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: all_models_bar_comparison_final.png")
