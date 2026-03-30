# =============================================================================
# MARKDOWN CELL — What is RUL?
# (paste as a Markdown cell, not a Code cell)
# =============================================================================

"""
## What is Remaining Useful Life (RUL)?

Every jet engine wears out a little more with each flight cycle (takeoff → cruise → landing).
**RUL** answers a simple question:

> *How many cycles does this engine have left before it needs maintenance?*

If the model predicts **RUL = 50**, the engine has roughly **50 flights remaining**.

---

### Why it matters

| Prediction | Consequence |
|------------|-------------|
| RUL too **low** | Engine grounded early — wasted cost |
| RUL too **high** | Failing engine kept in service — **dangerous** |

Getting it wrong in the dangerous direction (too high) costs more than being conservative.
This is why the model uses **asymmetric loss** — underestimating remaining life is
penalized more heavily than overestimating it.

---

### What the model sees

The LSTM receives the **last 75 cycles** of sensor readings (temperature, pressure,
vibration, etc.) for a given engine and predicts how many cycles remain.
The NASA C-MAPS FD001 dataset caps RUL at **125 cycles** — engines are treated
as healthy before degradation begins.
"""
