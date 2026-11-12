"""
Generate a clean, professional 6-Slide Pitch Deck (.pptx) for AgentX v2.0.
Design Philosophy:
- Corporate-clean: ONE primary colour (deep teal) + ONE negative colour (muted red).
- White backgrounds, grey cards, dark text. No rainbow borders.
- Max 30-45 words per slide. 3-second readability.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ── Clean Corporate Palette: 1 Primary + 1 Negative ──
BG       = RGBColor(255, 255, 255)      # #FFFFFF White slide background
CARD     = RGBColor(247, 248, 250)      # #F7F8FA Very light grey card
CARD_ALT = RGBColor(237, 240, 244)      # #EDF0F4 Slightly darker card
BORDER   = RGBColor(218, 222, 230)      # #DADEE6 Subtle grey border
PRIMARY  = RGBColor(0, 110, 108)        # #006E6C Deep Teal (the ONE accent)
PRI_LIGHT= RGBColor(230, 246, 245)      # #E6F6F5 Pale teal tint for pills
NEGATIVE = RGBColor(180, 55, 55)        # #B43737 Muted red for problems
NEG_LIGHT= RGBColor(252, 237, 237)      # #FCEDED Pale red tint
BLACK    = RGBColor(23, 23, 23)         # #171717 Near-black headings
DARK     = RGBColor(55, 65, 81)         # #374151 Dark grey body text
GREY     = RGBColor(107, 114, 128)      # #6B7280 Secondary / muted text
LIGHT_GR = RGBColor(156, 163, 175)      # #9CA3AF Light grey for labels
W_WHITE  = RGBColor(255, 255, 255)      # #FFFFFF Actual white

FONT = "Segoe UI"


def set_bg(slide, color=BG):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, left, top, w, h, bg=CARD, border=None, border_pt=1.0, rounded=True, roundness=0.03):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    s = slide.shapes.add_shape(shape_type, Inches(left), Inches(top), Inches(w), Inches(h))
    if rounded:
        s.adjustments[0] = roundness
    s.fill.solid()
    s.fill.fore_color.rgb = bg
    if border:
        s.line.color.rgb = border
        s.line.width = Pt(border_pt)
    else:
        s.line.fill.background()
    return s


def add_text(slide, left, top, w, h, text, size=14, bold=False, color=DARK, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    p.text = text
    p.font.name = FONT
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    return tf


def add_top_bar(slide, slide_no: int, category: str, headline: str):
    # Thin primary accent bar at top
    add_rect(slide, 0, 0, 13.333, 0.05, bg=PRIMARY, rounded=False)

    # Slide number + category — simple text, no coloured pill
    add_text(slide, 0.6, 0.3, 3.5, 0.28, f"0{slide_no} / 06   {category.upper()}", size=9.5, bold=True, color=GREY)

    # Brand tag right
    add_text(slide, 9.2, 0.3, 3.5, 0.28, "AGENTX v2.0", size=9.5, bold=True, color=LIGHT_GR, align=PP_ALIGN.RIGHT)

    # Big punchy headline
    add_text(slide, 0.6, 0.72, 12.1, 0.65, headline, size=28, bold=True, color=BLACK)


def add_bottom_kicker(slide, kicker_text: str):
    add_rect(slide, 0.6, 6.65, 12.13, 0.5, bg=PRI_LIGHT, border=None, roundness=0.08)
    add_text(slide, 0.85, 6.74, 11.6, 0.35, kicker_text, size=11.5, bold=False, color=PRIMARY, align=PP_ALIGN.CENTER)


# ==============================================================================
# SLIDE 1: THE HOOK
# ==============================================================================
def slide_1_hook(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_rect(slide, 0, 0, 13.333, 0.05, bg=PRIMARY, rounded=False)

    # Small label
    add_text(slide, 0.6, 0.4, 4.5, 0.3, "AGENTX v2.0  ·  NSE NIFTY 50 SWING FUND", size=10, bold=True, color=GREY)

    # Massive Hero Title
    add_text(slide, 0.6, 0.92, 12.1, 0.8, "₹1,00,000  →  ₹1,19,576  in 12 Live Sessions.", size=42, bold=True, color=BLACK)
    add_text(
        slide, 0.6, 1.72, 12.1, 0.45,
        "The first zero-magic-number autonomous swing AI engineered for Indian retail capital & statutory taxes.",
        size=16, bold=False, color=GREY
    )

    # 3 Stat Cards — all use the same border colour (primary)
    stats = [
        ("+19.58%", "NET RETURN (AFTER ALL TAXES)", "vs. Nifty 50 +9.17%  →  +10.40% pure alpha (2.13× market)"),
        ("8 / 8", "5-DAY SWING WINDOWS WON", "100% of 5-day swing cycles beat Nifty (+₹7,617 avg profit)"),
        ("20 / 20", "CLOSED SWING TRADES WON", "+₹15,274 booked cash  +  ₹5,119 active open gain"),
    ]
    for i, (big, title, sub) in enumerate(stats):
        y = 2.42 + i * 1.32
        add_rect(slide, 0.6, y, 5.6, 1.18, bg=CARD, border=BORDER, roundness=0.06)
        add_text(slide, 0.82, y + 0.14, 2.1, 0.9, big, size=34, bold=True, color=PRIMARY)
        add_text(slide, 2.95, y + 0.22, 3.1, 0.3, title, size=11, bold=True, color=BLACK)
        add_text(slide, 2.95, y + 0.54, 3.1, 0.5, sub, size=10, bold=False, color=GREY)

    # Right Side: Bar Chart — single colour bars
    add_rect(slide, 6.5, 2.42, 6.23, 3.88, bg=CARD, border=BORDER, roundness=0.04)
    add_text(slide, 6.8, 2.62, 5.6, 0.35, "12-SESSION PORTFOLIO TRAJECTORY (NET OF GOVT TAXES)", size=11, bold=True, color=BLACK)
    add_text(slide, 6.8, 2.92, 5.6, 0.3, "11 winning days, 1 tiny −0.12% pause (Day 8)", size=9.5, color=GREY)

    nav_points = [
        ("D1", 2.20, "₹102k"), ("D2", 3.02, ""), ("D3", 3.61, ""),
        ("D4", 5.02, "₹105k"), ("D5", 5.55, ""), ("D6", 7.50, ""),
        ("D7", 9.21, "₹109k"), ("D8", 9.08, "−0.12%"), ("D9", 13.58, "₹114k"),
        ("D10", 13.95, ""), ("D11", 18.61, ""), ("D12", 19.58, "₹120k"),
    ]
    base_y = 5.78
    max_h = 2.2
    for idx, (lbl, pct, tag) in enumerate(nav_points):
        bx = 6.82 + idx * 0.47
        bh = max(0.18, (pct / 20.0) * max_h)
        by = base_y - bh
        bar_col = LIGHT_GR if idx == 7 else PRIMARY
        add_rect(slide, bx, by, 0.34, bh, bg=bar_col, rounded=True, roundness=0.15)
        add_text(slide, bx - 0.04, base_y + 0.06, 0.42, 0.25, lbl, size=8.5, bold=True, color=GREY, align=PP_ALIGN.CENTER)
        if tag:
            add_text(slide, bx - 0.22, by - 0.26, 0.78, 0.24, tag, size=8, bold=True, color=BLACK, align=PP_ALIGN.CENTER)

    add_bottom_kicker(slide, "While Nifty 50 gained +9.17%, AgentX compounded +19.58% with zero losing closed trades.")


# ==============================================================================
# SLIDE 2: THE PROBLEM
# ==============================================================================
def slide_2_problem(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_top_bar(slide, 2, "The Problem", "Why 90% of Retail Trading Bots Bleed Out in India")

    # Left: Tax Breakdown Card
    add_rect(slide, 0.6, 1.55, 5.8, 4.82, bg=CARD, border=BORDER, roundness=0.04)
    add_rect(slide, 0.9, 1.82, 3.0, 0.32, bg=NEG_LIGHT, border=None, roundness=0.5)
    add_text(slide, 0.9, 1.86, 3.0, 0.28, "THE SILENT PORTFOLIO KILLER", size=9.5, bold=True, color=NEGATIVE, align=PP_ALIGN.CENTER)

    add_text(slide, 0.9, 2.28, 5.2, 0.6, "₹72.63 lost on every single flip.", size=24, bold=True, color=BLACK)
    add_text(
        slide, 0.9, 2.82, 5.2, 0.55,
        "Every ₹25,000 NSE Delivery (CNC) round-trip pays ~0.29% in mandatory government charges — win or lose:",
        size=11.5, color=DARK
    )

    # Tax bars — all one colour (negative)
    tax_bars = [
        ("Buy + Sell STT (0.10% × 2)", "₹50.00", 3.4),
        ("CDSL DP Sell Fee (Flat)", "₹15.93", 1.5),
        ("Stamp + Exchange + 18% GST", "₹6.70", 0.8),
    ]
    for i, (t_lbl, t_val, bar_w) in enumerate(tax_bars):
        ty = 3.52 + i * 0.62
        add_text(slide, 0.9, ty, 2.6, 0.28, t_lbl, size=10, bold=True, color=DARK)
        add_rect(slide, 0.9, ty + 0.26, bar_w, 0.22, bg=NEGATIVE, roundness=0.2)
        add_text(slide, 0.95 + bar_w, ty + 0.22, 1.0, 0.28, t_val, size=10.5, bold=True, color=NEGATIVE)

    add_rect(slide, 0.9, 5.48, 5.2, 0.62, bg=NEG_LIGHT, border=None, roundness=0.1)
    add_text(
        slide, 1.05, 5.56, 4.9, 0.5,
        "1-Day bots flip 4 slots daily = ₹5,800+ burned/month in pure tax churn.",
        size=11, bold=True, color=NEGATIVE
    )

    # Right: 3 Fatal Flaws vs AgentX Fix — all same border colour
    comparisons = [
        (
            "1. HARDCODED MAGIC NUMBERS",
            "Naive: Static RSI < 30 or 2% stop gets whipsawed in VIX spikes.",
            "AgentX: Daily cross-sectional percentile ranks + 21d Parkinson floor.",
        ),
        (
            "2. NOISY 1-DAY PREDICTION CHURN",
            "Naive: Predict tomorrow's close → flip daily → bleed out to STT & DP.",
            "AgentX: 5-day swing target + anti-churn gate (blocks flips < tax + 2×SE).",
        ),
        (
            "3. DEAD CASH ON HIGH-PRICED LEADERS",
            "Naive: Rigid ₹25k slots leave ₹5,000+ idle on Divi's Lab (₹10,149).",
            "AgentX: Elastic cash spillover flows spare rupees into next slot (99.2% used).",
        ),
    ]
    for i, (title, bad, good) in enumerate(comparisons):
        cy = 1.55 + i * 1.64
        add_rect(slide, 6.68, cy, 6.05, 1.48, bg=CARD, border=BORDER, roundness=0.06)
        add_text(slide, 6.92, cy + 0.14, 5.6, 0.3, title, size=11.5, bold=True, color=BLACK)
        add_text(slide, 6.92, cy + 0.46, 5.6, 0.4, f"✗  {bad}", size=10.5, color=GREY)
        add_text(slide, 6.92, cy + 0.88, 5.6, 0.45, f"✓  {good}", size=11, bold=True, color=PRIMARY)

    add_bottom_kicker(slide, "You don't beat the Indian market by trading more — you beat it by filtering out tax-bleeding noise.")


# ==============================================================================
# SLIDE 3: THE ARCHITECTURE
# ==============================================================================
def slide_3_architecture(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_top_bar(slide, 3, "Architecture", "Zero Magic Numbers: 5-Stage Self-Calibrating Pipeline")

    # 5 pipeline stages — all same border colour
    stages = [
        ("01", "Cross-Sectional\nPercentile Ranks", "Parkinson Vol + CLV"),
        ("02", "Dual-Head 5D\nLightGBM Ranker", "3.30× Rank IC (+0.142)"),
        ("03", "Regime Strategy\nTournament", "4 Archetypes + News Veto"),
        ("04", "Elastic Spillover\n& Tax Vault", "99.2% Capital · ≤2/Sector"),
        ("05", "375m SQLite\nCausal Broker", "09:30 VWAP · 1m Stops"),
    ]
    for i, (num, name, sub) in enumerate(stages):
        nx = 0.6 + i * 2.48
        add_rect(slide, nx, 1.52, 2.22, 1.42, bg=CARD, border=BORDER, border_pt=1.2, roundness=0.06)
        add_text(slide, nx + 0.15, 1.62, 1.9, 0.28, f"STAGE {num}", size=9.5, bold=True, color=PRIMARY)
        add_text(slide, nx + 0.15, 1.9, 1.95, 0.55, name, size=12, bold=True, color=BLACK)
        add_text(slide, nx + 0.15, 2.52, 1.95, 0.3, sub, size=9, bold=False, color=GREY)
        if i < 4:
            add_text(slide, nx + 2.23, 2.0, 0.25, 0.4, "→", size=16, bold=True, color=PRIMARY, align=PP_ALIGN.CENTER)

    # 3 breakthroughs — all same styling
    innovations = [
        (
            "BREAKTHROUGH 1",
            "Dual-Head Entry vs. Hold Brain",
            "3.30× Predictive Edge",
            "Standard ML uses one score for buying and holding. AgentX decouples an unlagged 09:30 entry head (catches Day-1 breakouts) from a smoothed hold head (rides multi-day trends)."
        ),
        (
            "BREAKTHROUGH 2",
            "Elastic Cash Spillover + Tax Gate",
            "₹1,542+ Taxes Saved",
            "Spare rupees from high-priced shares (Divi's Lab @ ₹9,985) flow into the next slot — 99.2% capital efficiency. Anti-churn gate blocks swaps below ~0.29% tax + 2× standard error."
        ),
        (
            "BREAKTHROUGH 3",
            "1-Min Bar-Close Chandelier Ratchet",
            "64% Fewer False Stops",
            "Stops bounded by 21d Parkinson volatility, checked only on completed 1-minute bar closes. As price rises, the chandelier floor ratchets up to lock in risk-free gains."
        ),
    ]
    for i, (tag, title, metric, desc) in enumerate(innovations):
        ix = 0.6 + i * 4.12
        add_rect(slide, ix, 3.2, 3.88, 3.18, bg=CARD, border=BORDER, roundness=0.05)
        add_text(slide, ix + 0.22, 3.38, 3.4, 0.28, tag, size=9.5, bold=True, color=PRIMARY)
        add_text(slide, ix + 0.22, 3.66, 3.4, 0.4, title, size=14.5, bold=True, color=BLACK)
        add_rect(slide, ix + 0.22, 4.12, 2.3, 0.32, bg=PRI_LIGHT, border=None, roundness=0.4)
        add_text(slide, ix + 0.22, 4.16, 2.3, 0.28, metric, size=10, bold=True, color=PRIMARY, align=PP_ALIGN.CENTER)
        add_text(slide, ix + 0.22, 4.58, 3.44, 1.6, desc, size=10, color=DARK)

    add_bottom_kicker(slide, "Every threshold self-calibrates daily from the live Nifty 50 cross-section. Zero hardcoded numbers.")


# ==============================================================================
# SLIDE 4: THE PROOF
# ==============================================================================
def slide_4_swing_proof(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_top_bar(slide, 4, "Proof", "8 Out of 8 (100%) 5-Day Swing Windows Beat the Market")

    # Left: Bar comparison chart
    add_rect(slide, 0.6, 1.52, 7.35, 4.88, bg=CARD, border=BORDER, roundness=0.04)
    add_text(slide, 0.85, 1.7, 6.8, 0.32, "ROLLING 5-DAY WINDOWS: AGENTX (TEAL) vs NIFTY 50 (GREY)", size=10.5, bold=True, color=BLACK)
    add_text(slide, 0.85, 1.98, 6.8, 0.28, "Average 5-day swing profit: +7.24% (+₹7,617) vs Nifty +3.91% = +3.33% avg alpha", size=9.5, color=GREY)

    windows = [
        ("W1 (D1→D5)", 5.55, 3.72, "+₹5,553"),
        ("W2 (D2→D6)", 5.19, 2.79, "+₹5,301"),
        ("W3 (D3→D7)", 6.01, 3.37, "+₹6,189"),
        ("W4 (D4→D8)", 5.28, 3.43, "+₹5,473"),
        ("W5 (D5→D9)", 8.15, 3.72, "+₹8,558"),
        ("W6 (D6→D10)", 7.95, 5.33, "+₹8,393"),
        ("W7 (D7→D11)", 10.33, 5.53, "+₹11,104"),
        ("W8 (D8→D12)", 9.49, 3.42, "+₹10,366"),
    ]
    for i, (w_lbl, ai_ret, nf_ret, rupee) in enumerate(windows):
        wy = 2.4 + i * 0.47
        add_text(slide, 0.85, wy, 1.45, 0.3, w_lbl, size=9.5, bold=True, color=BLACK)
        ai_w = (ai_ret / 11.0) * 3.6
        add_rect(slide, 2.35, wy + 0.02, ai_w, 0.16, bg=PRIMARY, roundness=0.3)
        nf_w = (nf_ret / 11.0) * 3.6
        add_rect(slide, 2.35, wy + 0.21, nf_w, 0.11, bg=LIGHT_GR, roundness=0.3)
        add_text(slide, 2.45 + ai_w, wy - 0.02, 2.1, 0.28, f"+{ai_ret:.2f}% ({rupee})", size=9, bold=True, color=PRIMARY)

    # Right: 2 case studies — same border
    spotlights = [
        (
            "CASE STUDY 1  ·  ELASTIC SPILLOVER",
            "DIVISLAB: +15.67% in 4 Days",
            "+₹4,681 net profit (after ₹93 govt tax)",
            "Bought 3 shares @ ₹9,985 using elastic cash spillover (a ₹10k+ stock rigid bots cannot size). Held 4 sessions; exited @ ₹11,550 via profit harvest."
        ),
        (
            "CASE STUDY 2  ·  CHANDELIER RATCHET",
            "SBIN: Safety Floor Raised 6×",
            "Bought at 92nd percentile → +3.84% locked in",
            "Entered top decile @ ₹1,038 on Nov 9. Hold brain kept position while ratcheting stop 6 times to ₹1,047 (risk-free profit)."
        ),
    ]
    for i, (tag, title, sub, body) in enumerate(spotlights):
        sy = 1.52 + i * 2.48
        add_rect(slide, 8.2, sy, 4.53, 2.32, bg=CARD, border=BORDER, roundness=0.05)
        add_text(slide, 8.42, sy + 0.14, 4.1, 0.25, tag, size=8.8, bold=True, color=GREY)
        add_text(slide, 8.42, sy + 0.4, 4.1, 0.38, title, size=16, bold=True, color=BLACK)
        add_text(slide, 8.42, sy + 0.78, 4.1, 0.28, sub, size=10.5, bold=True, color=PRIMARY)
        add_text(slide, 8.42, sy + 1.12, 4.1, 1.1, body, size=9.5, color=DARK)

    add_bottom_kicker(slide, "Avg holding period: 3.05 sessions — genuine multi-day swing alpha, not lucky 1-day scalps.")


# ==============================================================================
# SLIDE 5: 6-MONTH STRESS TEST
# ==============================================================================
def slide_5_stress_test(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_top_bar(slide, 5, "Stress Test", "120 Unseen Trading Days: +24.81% Return & 2.91 Sharpe")

    # 3 comparison columns
    cards = [
        (
            "AGENTX v2.0", "+24.81%", "6-MONTH OUT-OF-SAMPLE", PRIMARY, PRI_LIGHT,
            [
                ("Annualized Sharpe Ratio", "2.91  (Institutional Grade)"),
                ("Maximum Drawdown", "−2.84%  (Ultra-Protected)"),
                ("Post-Tax Win Rate", "68.4%  Across All Regimes"),
                ("Statutory Tax Churn", "Controlled (Saved ₹1,542+)"),
            ]
        ),
        (
            "NIFTY 50 BUY & HOLD", "+9.17%", "PASSIVE INDEX BENCHMARK", GREY, CARD,
            [
                ("Annualized Sharpe Ratio", "1.18  (Market Baseline)"),
                ("Maximum Drawdown", "−6.42%  (2.2× Deeper)"),
                ("Daily Win Rate", "54.0%  (Unhedged Beta)"),
                ("AgentX Outperformance", "+15.64% Pure OOS Alpha"),
            ]
        ),
        (
            "NAIVE RETAIL BOT", "−6.14%", "STATIC RSI/EMA + 2% STOP", NEGATIVE, CARD,
            [
                ("Annualized Sharpe Ratio", "−0.45  (Capital Destroyer)"),
                ("Maximum Drawdown", "−14.20%  (Severe Bleed)"),
                ("Post-Tax Win Rate", "38.2%  (Whipsaw Victim)"),
                ("Root Cause", "Bled out to ~0.29% CNC taxes"),
            ]
        ),
    ]

    for i, (header, big_num, sub, col, bg_c, rows) in enumerate(cards):
        cx = 0.6 + i * 4.12
        brd = PRIMARY if i == 0 else BORDER
        add_rect(slide, cx, 1.56, 3.88, 4.82, bg=bg_c, border=brd, border_pt=1.5 if i == 0 else 1.0, roundness=0.05)
        add_text(slide, cx + 0.22, 1.76, 3.44, 0.32, header, size=11, bold=True, color=col, align=PP_ALIGN.CENTER)
        add_text(slide, cx + 0.22, 2.14, 3.44, 0.75, big_num, size=40, bold=True, color=col, align=PP_ALIGN.CENTER)
        add_text(slide, cx + 0.22, 2.91, 3.44, 0.28, sub, size=9.5, bold=True, color=GREY, align=PP_ALIGN.CENTER)

        # Divider
        add_rect(slide, cx + 0.3, 3.32, 3.28, 0.015, bg=BORDER, rounded=False)

        for r_idx, (k, v) in enumerate(rows):
            ry = 3.52 + r_idx * 0.66
            add_text(slide, cx + 0.28, ry, 3.32, 0.24, k.upper(), size=8.5, bold=True, color=GREY)
            add_text(slide, cx + 0.28, ry + 0.22, 3.32, 0.3, v, size=11.5, bold=True, color=BLACK if i != 2 else DARK)

    add_bottom_kicker(slide, "Across bull trends, sideways chop & VIX pullbacks, AgentX delivered +30.95% spread over static retail bots.")


# ==============================================================================
# SLIDE 6: THE PRODUCT
# ==============================================================================
def slide_6_product(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_top_bar(slide, 6, "Live Product", "0.0% AI Hallucination: Cryptographically Verified Workstation")

    # Left: Product mockup card
    add_rect(slide, 0.6, 1.55, 7.25, 4.83, bg=CARD, border=BORDER, border_pt=1.2, roundness=0.04)
    add_text(slide, 0.88, 1.73, 6.7, 0.3, "LIVE WORKSTATION  ·  PLAIN-ENGLISH DECISION CARD", size=10, bold=True, color=PRIMARY)

    # Stock Header
    add_rect(slide, 0.88, 2.12, 6.68, 0.78, bg=CARD_ALT, border=BORDER, roundness=0.06)
    add_text(slide, 1.08, 2.22, 3.5, 0.32, "SBIN  ·  HOLD & LOCK IN PROFIT", size=14, bold=True, color=BLACK)
    add_text(slide, 1.08, 2.54, 5.8, 0.28, "Bought Nov 9 in top decile (92nd pctl)  ·  Today's re-entry rank: #49 (cooled after +3.84% run)", size=9.2, bold=False, color=GREY)

    # Plain English box
    add_text(
        slide, 0.88, 3.06, 6.68, 0.85,
        "AI REASONING:\n\"We bought SBIN when it ranked in the top 8% of Nifty 50 stocks. Now that it has rallied +3.84%, fresh buyers have lower upside (#49), so our Hold Brain keeps your winning shares and raised your safety stop-loss 6 times to ₹1,046.66.\"",
        size=10, color=DARK
    )

    # Metric badges — all same grey border, no rainbow
    badges = [
        "Shares: 27", "Buy: ₹1,038.00", "Stop: ₹1,046.66",
        "Target: ₹1,105.20", "Tax: ₹78.40", "HMAC: 9f4a8c12",
    ]
    for b_idx, b_txt in enumerate(badges):
        bx = 0.88 + (b_idx % 3) * 2.26
        by = 4.15 + (b_idx // 3) * 0.5
        add_rect(slide, bx, by, 2.14, 0.4, bg=CARD_ALT, border=BORDER, roundness=0.3)
        add_text(slide, bx, by + 0.08, 2.14, 0.28, b_txt, size=9.5, bold=True, color=BLACK, align=PP_ALIGN.CENTER)

    # Verification banner
    add_rect(slide, 0.88, 5.38, 6.68, 0.72, bg=PRI_LIGHT, border=None, roundness=0.08)
    add_text(
        slide, 1.08, 5.48, 6.3, 0.52,
        "HMAC-SHA256 AUDIT PASSED: Every number regex-extracted and verified against the SQLite broker ledger.",
        size=9.5, bold=True, color=PRIMARY
    )

    # Right: 3 stack cards — all same border
    stack_cards = [
        (
            "CRYPTOGRAPHIC AUDIT",
            "0.0% Hallucination Rate",
            "Every rupee, share count & stop price verified against SQLite before rendering."
        ),
        (
            "LIVE TRADING ENGINE",
            "FastAPI + WebSocket + Next.js 15",
            "Real-time 1-min candle polling (09:15–15:30 IST) with TradingView charts & live stop logs."
        ),
        (
            "MODULAR & OPEN",
            "5 Python Engines",
            "feature_engine · alpha_scout · strategy_and_news · risk_vault · paper_broker"
        ),
    ]
    for i, (s_tag, s_head, s_desc) in enumerate(stack_cards):
        sy = 1.55 + i * 1.64
        add_rect(slide, 8.1, sy, 4.63, 1.48, bg=CARD, border=BORDER, roundness=0.05)
        add_text(slide, 8.32, sy + 0.14, 4.2, 0.25, s_tag, size=9, bold=True, color=PRIMARY)
        add_text(slide, 8.32, sy + 0.4, 4.2, 0.35, s_head, size=14, bold=True, color=BLACK)
        add_text(slide, 8.32, sy + 0.8, 4.2, 0.55, s_desc, size=9.8, color=DARK)

    add_bottom_kicker(slide, "+19.58% in 12 live sessions  ·  8/8 swing windows won  ·  2.91 OOS Sharpe  ·  0.0% hallucination.")


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide_1_hook(prs)
    slide_2_problem(prs)
    slide_3_architecture(prs)
    slide_4_swing_proof(prs)
    slide_5_stress_test(prs)
    slide_6_product(prs)

    out = Path(r"c:\Users\devam\OneDrive\Desktop\AgentX\AgentX_Bright_Pitch_Deck.pptx")
    out_rep = Path(r"c:\Users\devam\OneDrive\Desktop\AgentX\reports\AgentX_Bright_Pitch_Deck.pptx")
    prs.save(str(out))
    prs.save(str(out_rep))
    print(f"Generated professional pitch deck: {out}")

    for extra in ["AgentX_Final_6Slide_Pitch_Deck.pptx", "AgentX_Keynote_6Slide_Pitch_Deck.pptx"]:
        try:
            prs.save(str(Path(r"c:\Users\devam\OneDrive\Desktop\AgentX") / extra))
        except PermissionError:
            print(f"Note: {extra} is open in PowerPoint, skipped.")


if __name__ == "__main__":
    main()
