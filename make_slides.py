from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Palette ──────────────────────────────────────────────────────────────────
BG      = RGBColor(0x0D, 0x1B, 0x2A)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
DIM     = RGBColor(0x7A, 0x96, 0xB2)
H_ACC   = RGBColor(0x00, 0xB4, 0xD8)   # cyan  — Hunter
F_ACC   = RGBColor(0x6F, 0xD0, 0x8C)   # green — Felix
CARD_BG = RGBColor(0x15, 0x28, 0x3C)

W, H = Inches(13.33), Inches(7.5)


def new_prs():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs

def blank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    return s

def rect(slide, x, y, w, h, color):
    sh = slide.shapes.add_shape(1, x, y, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = color
    sh.line.fill.background()

def txt(slide, text, x, y, w, h, size, color, bold=False, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    p  = tf.paragraphs[0]; p.alignment = align
    r  = p.add_run(); r.text = text
    r.font.size = size; r.font.color.rgb = color; r.font.bold = bold

def bullets(slide, items, x, y, w, h, size, color):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before = Pt(6)
        r = p.add_run(); r.text = "•  " + item
        r.font.size = size; r.font.color.rgb = color

def set_notes(slide, text):
    from pptx.oxml.ns import qn
    from lxml import etree
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = text

def header(slide, title, acc, owner):
    rect(slide, 0, 0, Inches(0.07), H, acc)
    txt(slide, title,
        Inches(0.22), Inches(0.15), Inches(10.5), Inches(0.75),
        Pt(28), WHITE, bold=True)
    rect(slide, W - Inches(1.7), Inches(0.15), Inches(1.5), Inches(0.45), acc)
    txt(slide, owner,
        W - Inches(1.7), Inches(0.15), Inches(1.5), Inches(0.45),
        Pt(13), BG, bold=True, align=PP_ALIGN.CENTER)
    rect(slide, Inches(0.22), Inches(0.92), Inches(12.9), Pt(1),
         RGBColor(0x25, 0x3C, 0x52))

def cards(slide, items, x, y, cw, ch, acc):
    gap = Inches(0.15)
    for i, (val, lbl) in enumerate(items):
        cx = x + i * (cw + gap)
        rect(slide, cx, y, cw, ch, CARD_BG)
        txt(slide, val, cx, y + Inches(0.05), cw, Inches(0.55),
            Pt(28), acc, bold=True, align=PP_ALIGN.CENTER)
        txt(slide, lbl, cx, y + Inches(0.58), cw, Inches(0.32),
            Pt(14), DIM, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════════════════
#   SLIDES
# ═══════════════════════════════════════════════════════════════════════════

def s1_title(prs):
    s = blank(prs)
    rect(s, 0, 0, Inches(0.07), H, H_ACC)
    txt(s, "Remaining Useful Life Estimation",
        Inches(0.3), Inches(1.5), Inches(12.7), Inches(1.0),
        Pt(40), WHITE, bold=True)
    txt(s, "of Turbofan Engines Using Machine Learning",
        Inches(0.3), Inches(2.4), Inches(12.7), Inches(0.85),
        Pt(34), H_ACC, bold=True)
    txt(s, "Felix Ikokwu  ·  Hunter Antal   |   Lakehead University",
        Inches(0.3), Inches(3.5), Inches(9), Inches(0.5),
        Pt(20), DIM)
    bullets(s, [
        "Predict cycles remaining before engine failure",
        "NASA C-MAPSS FD001 dataset",
        "Models compared: Random Forest, MLP, LSTM",
    ], Inches(0.3), Inches(4.4), Inches(12.5), Inches(2.0), Pt(21), RGBColor(0xD0,0xE8,0xF5))
    set_notes(s,
        "Welcome everyone. This project asks a simple question: given sensor "
        "readings from a jet engine, how many cycles does it have left? "
        "We tested three models of increasing complexity to find out.")

def s2_dataset(prs):
    s = blank(prs)
    header(s, "Dataset & Problem", H_ACC, "HUNTER  1/5")
    # Left card
    rect(s, Inches(0.3), Inches(1.1), Inches(5.9), Inches(5.7), CARD_BG)
    txt(s, "NASA C-MAPSS FD001",
        Inches(0.5), Inches(1.2), Inches(5.5), Inches(0.55),
        Pt(21), H_ACC, bold=True)
    bullets(s, [
        "~100 engines, run-to-failure",
        "21 sensors + 3 operational settings per cycle",
        "Training: full histories   |   Test: partial",
    ], Inches(0.5), Inches(1.85), Inches(5.5), Inches(2.5), Pt(20), RGBColor(0xD0,0xE8,0xF5))
    # Right card
    rect(s, Inches(6.9), Inches(1.1), Inches(6.1), Inches(5.7), CARD_BG)
    txt(s, "Problem",
        Inches(7.1), Inches(1.2), Inches(5.7), Inches(0.55),
        Pt(21), H_ACC, bold=True)
    bullets(s, [
        "Supervised regression — predict RUL (cycles)",
        "RUL capped at 125  (healthy region)",
        "Split by engine ID — no data leakage",
    ], Inches(7.1), Inches(1.85), Inches(5.7), Inches(2.5), Pt(20), RGBColor(0xD0,0xE8,0xF5))
    set_notes(s,
        "The dataset simulates turbofan engines degrading until failure. "
        "Each row is one operational cycle with 24 raw features. "
        "We cap RUL at 125 because before that point engines show no degradation signal. "
        "Engine IDs are kept separate between train and test to prevent leakage.")

def s3_preprocessing(prs):
    s = blank(prs)
    header(s, "Data Preprocessing", H_ACC, "HUNTER  2/5")
    steps = [
        ("Feature Selection",
         ["9 features removed (low variance / low correlation)",
          "15 features retained"]),
        ("Normalization",
         ["Z-score: mean = 0, std = 1",
          "Prevents any feature dominating training"]),
        ("Windowing (LSTM)",
         ["30-cycle sliding windows",
          "RF & MLP flatten to a 450-dim vector"]),
    ]
    cw = Inches(4.05)
    for i, (title, pts) in enumerate(steps):
        cx = Inches(0.3) + i * (cw + Inches(0.15))
        rect(s, cx, Inches(1.1), cw, Inches(5.7), CARD_BG)
        rect(s, cx, Inches(1.1), cw, Inches(0.48), RGBColor(0x00,0x7A,0x99))
        txt(s, title, cx + Inches(0.12), Inches(1.13),
            cw - Inches(0.2), Inches(0.42), Pt(19), WHITE, bold=True)
        bullets(s, pts, cx + Inches(0.12), Inches(1.68),
                cw - Inches(0.2), Inches(4.8), Pt(20), RGBColor(0xD0,0xE8,0xF5))
    set_notes(s,
        "Three preprocessing steps. Feature selection removes noise. "
        "Normalization ensures fair training. Windowing gives the LSTM temporal context — "
        "it sees the last 30 cycles at once, rather than just the current cycle.")

def s4_models(prs):
    s = blank(prs)
    header(s, "Model Architectures", H_ACC, "HUNTER  3/5")
    models = [
        ("Random Forest", "Baseline",
         ["Ensemble of decision trees",
          "1 M params  |  52 M FLOPs"]),
        ("MLP", "Intermediate",
         ["3 hidden layers: 128 → 64 → 32",
          "BatchNorm + Dropout  |  18 K params"]),
        ("LSTM", "Best",
         ["128-unit recurrent layer",
          "Preserves 30-cycle temporal order",
          "83 K params  |  22.6 B FLOPs"]),
    ]
    chip_colors = [RGBColor(0x3A,0x5A,0x7E),
                   RGBColor(0x2A,0x75,0x55),
                   RGBColor(0x00,0x7A,0x99)]
    cw = Inches(4.05)
    for i, (name, tag, pts) in enumerate(models):
        cx = Inches(0.3) + i * (cw + Inches(0.15))
        rect(s, cx, Inches(1.1), cw, Inches(5.7), CARD_BG)
        rect(s, cx + Inches(0.12), Inches(1.18),
             Inches(1.6), Inches(0.32), chip_colors[i])
        txt(s, tag.upper(),
            cx + Inches(0.12), Inches(1.18), Inches(1.6), Inches(0.32),
            Pt(12), WHITE, bold=True, align=PP_ALIGN.CENTER)
        txt(s, name, cx + Inches(0.12), Inches(1.58),
            cw - Inches(0.2), Inches(0.5), Pt(22), H_ACC, bold=True)
        bullets(s, pts, cx + Inches(0.12), Inches(2.15),
                cw - Inches(0.2), Inches(4.3), Pt(20), RGBColor(0xD0,0xE8,0xF5))
    set_notes(s,
        "Three models of increasing complexity. "
        "Random Forest is our baseline — powerful but treats every cycle independently. "
        "MLP adds nonlinear transformations but still uses a flat input. "
        "LSTM is the only model that actually sees time — it processes 30 cycles in sequence.")

def s5_metrics(prs):
    s = blank(prs)
    header(s, "Evaluation Framework", H_ACC, "HUNTER  4/5")
    rect(s, Inches(0.3), Inches(1.1), Inches(5.9), Inches(5.7), CARD_BG)
    txt(s, "Accuracy Metrics",
        Inches(0.5), Inches(1.2), Inches(5.5), Inches(0.5),
        Pt(21), H_ACC, bold=True)
    bullets(s, [
        "MAE  —  avg error in cycles",
        "RMSE  —  penalises large errors",
        "R²  —  variance explained (0–1)",
    ], Inches(0.5), Inches(1.8), Inches(5.5), Inches(2.5), Pt(21), RGBColor(0xD0,0xE8,0xF5))

    rect(s, Inches(6.9), Inches(1.1), Inches(6.1), Inches(5.7), CARD_BG)
    txt(s, "Residual & Cost",
        Inches(7.1), Inches(1.2), Inches(5.7), Inches(0.5),
        Pt(21), H_ACC, bold=True)
    bullets(s, [
        "Residual = Actual − Predicted",
        "Overprediction = engine seems healthier than it is → dangerous",
        "Asymmetric loss penalises overprediction 50% harder (α = 1.5)",
        "Efficiency: parameter count & FLOPs",
    ], Inches(7.1), Inches(1.8), Inches(5.7), Inches(4.5), Pt(20), RGBColor(0xD0,0xE8,0xF5))
    set_notes(s,
        "We care most about RMSE and R² for overall accuracy. "
        "But residual direction matters too — if we overestimate RUL we might delay maintenance on an engine that's about to fail. "
        "That's why we use asymmetric loss in the final LSTM: wrong in the dangerous direction costs more.")

def s6_rf(prs):
    s = blank(prs)
    header(s, "Random Forest — Results", F_ACC, "FELIX  1/5")
    cards(s, [
        ("12.38", "MAE (test)"),
        ("17.18", "RMSE (test)"),
        ("0.612", "R² (test)"),
    ], Inches(0.3), Inches(1.1), Inches(4.15), Inches(1.0), F_ACC)
    bullets(s, [
        "Solid baseline — captures nonlinear feature relationships",
        "Compresses predictions at high RUL (early degradation) — underestimates severity",
        "Residual variance grows as engine approaches failure",
        "Hyperparameter tuning plateaus at RMSE ≈ 19.6  — structural limit, not a tuning issue",
    ], Inches(0.3), Inches(2.35), Inches(12.6), Inches(4.3), Pt(21), RGBColor(0xD0,0xE8,0xF5))
    set_notes(s,
        "Random Forest is a strong starting point. It handles nonlinearity well. "
        "But it sees each cycle as an isolated snapshot — it has no idea what happened 10 cycles ago. "
        "That's why it struggles early in degradation when the signal is subtle and slow-building. "
        "More trees or deeper trees don't help — the bottleneck is the flat input, not the model complexity.")

def s7_mlp(prs):
    s = blank(prs)
    header(s, "MLP — Results", F_ACC, "FELIX  2/5")
    cards(s, [
        ("12.07", "MAE (test)"),
        ("16.89", "RMSE (test)"),
        ("0.625", "R² (test)"),
    ], Inches(0.3), Inches(1.1), Inches(4.15), Inches(1.0), F_ACC)
    bullets(s, [
        "Marginal improvement over RF in error metrics",
        "Tighter clustering around the diagonal vs. RF",
        "Test R² drops from 0.792 → 0.625  — generalisation limited by flat input",
        "Architecture [64, 128, 64] + LR 0.0005 was optimal",
    ], Inches(0.3), Inches(2.35), Inches(12.6), Inches(4.3), Pt(21), RGBColor(0xD0,0xE8,0xF5))
    set_notes(s,
        "MLP adds learnable nonlinear transformations on top of the flat window. "
        "It does improve slightly, and the scatter plot looks a bit tighter. "
        "But the validation-to-test drop in R² tells us it's not generalising well to new engines. "
        "The root cause is the same as RF: no temporal structure in the input.")

def s8_lstm(prs):
    s = blank(prs)
    header(s, "LSTM — Results", F_ACC, "FELIX  3/5")
    cards(s, [
        ("10.34", "MAE (test)"),
        ("14.76", "RMSE (test)"),
        ("0.754", "R² (test)"),
    ], Inches(0.3), Inches(1.1), Inches(4.15), Inches(1.0), F_ACC)
    bullets(s, [
        "Best model — explicitly models temporal degradation sequences",
        "Substantially lower RMSE vs. RF and MLP on both val and test",
        "Narrower residual distribution — fewer extreme errors",
        "Val → test R² drop of only 0.14 pts (RF/MLP: ~0.17)",
    ], Inches(0.3), Inches(2.35), Inches(12.6), Inches(4.3), Pt(21), RGBColor(0xD0,0xE8,0xF5))
    set_notes(s,
        "LSTM is a clear winner. By processing 30 cycles in order it can detect gradual drift "
        "that a single-cycle model would miss. "
        "The smaller val-to-test drop tells us it generalises better to engines it's never seen. "
        "This makes sense — temporal patterns of degradation are more universal than any single-point feature snapshot.")

def s9_comparison(prs):
    s = blank(prs)
    header(s, "Model Comparison", F_ACC, "FELIX  4/5")

    col_xs = [Inches(0.3), Inches(3.55), Inches(6.8), Inches(10.05)]
    col_ws = [Inches(3.2), Inches(3.2), Inches(3.2), Inches(3.0)]
    headers = ["Metric", "Random Forest", "MLP", "LSTM"]
    rows = [
        ("RMSE  (val / test)", "19.58 / 17.18", "18.95 / 16.89", "13.35 / 14.76"),
        ("R²      (val / test)", "0.778 / 0.612", "0.792 / 0.625", "0.898 / 0.754"),
        ("MAE   (val / test)", "14.01 / 12.38", "13.50 / 12.07",  "9.21 / 10.34"),
        ("Parameters",          "~1 M",          "18 K",           "83 K"),
        ("FLOPs (test)",        "52 M",          "456 M",          "22.6 B"),
    ]
    row_h = Inches(0.56)
    hdr_y = Inches(1.1)
    rect(s, Inches(0.25), hdr_y, Inches(12.85), row_h, RGBColor(0x00,0x7A,0x99))
    for j, (cx, cw, ch) in enumerate(zip(col_xs, col_ws, headers)):
        txt(s, ch, cx, hdr_y, cw, row_h, Pt(20), WHITE, bold=True)

    for i, row in enumerate(rows):
        ry = hdr_y + row_h + i * row_h
        bg = CARD_BG if i % 2 == 0 else BG
        rect(s, Inches(0.25), ry, Inches(12.85), row_h, bg)
        for j, (cx, cw, val) in enumerate(zip(col_xs, col_ws, row)):
            color = F_ACC if j == 3 else RGBColor(0xD0,0xE8,0xF5)
            txt(s, val, cx, ry, cw, row_h, Pt(19), color, bold=(j==3))

    set_notes(s,
        "This table tells the full story. LSTM leads on every accuracy metric. "
        "What's interesting is the efficiency angle: RF has a million parameters but cheap inference. "
        "LSTM has 83K parameters but expensive sequential computation — 22 billion FLOPs. "
        "For a safety-critical maintenance system, the accuracy gain is worth the compute cost.")

def s10_conclusion(prs):
    s = blank(prs)
    header(s, "Optimisation & Conclusions", F_ACC, "FELIX  5/5")

    rect(s, Inches(0.3), Inches(1.1), Inches(5.9), Inches(5.7), CARD_BG)
    txt(s, "Best LSTM Config",
        Inches(0.5), Inches(1.2), Inches(5.5), Inches(0.5),
        Pt(21), F_ACC, bold=True)
    bullets(s, [
        "Hidden = 256  |  Sequence = 75 cycles",
        "Dropout = 0.4  |  Asymmetric loss (α = 1.5)",
        "ReduceLROnPlateau scheduler",
        "RMSE 13.33  |  R² 0.839  |  62.4% within 10 cycles",
    ], Inches(0.5), Inches(1.8), Inches(5.5), Inches(4.5), Pt(20), RGBColor(0xD0,0xE8,0xF5))

    rect(s, Inches(6.9), Inches(1.1), Inches(6.1), Inches(5.7), CARD_BG)
    txt(s, "Takeaways",
        Inches(7.1), Inches(1.2), Inches(5.7), Inches(0.5),
        Pt(21), F_ACC, bold=True)
    bullets(s, [
        "Temporal modelling drives accuracy  —  LSTM > MLP > RF",
        "More non-temporal complexity doesn't close the gap",
        "Asymmetric loss reduces dangerous overpredictions",
        "Future: attention, ensembling, multi-condition datasets",
    ], Inches(7.1), Inches(1.8), Inches(5.7), Inches(4.5), Pt(20), RGBColor(0xD0,0xE8,0xF5))

    set_notes(s,
        "After selecting LSTM we ran a 150-configuration random search to tune it. "
        "The biggest wins came from longer sequences (75 vs 30 cycles) and asymmetric loss. "
        "The key conclusion: the ability to model time is what matters. "
        "If you can only pick one thing to improve a predictive maintenance model, add temporal structure.")


def main():
    prs = new_prs()
    s1_title(prs)
    s2_dataset(prs)
    s3_preprocessing(prs)
    s4_models(prs)
    s5_metrics(prs)
    s6_rf(prs)
    s7_mlp(prs)
    s8_lstm(prs)
    s9_comparison(prs)
    s10_conclusion(prs)
    out = "/home/hunterantal/Dev/4011_RUL_Est_ of_Turbofan_Engines/RUL_Presentation.pptx"
    prs.save(out)
    print(f"Saved → {out}")

if __name__ == "__main__":
    main()
