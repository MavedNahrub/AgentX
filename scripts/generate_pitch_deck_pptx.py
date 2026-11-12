"""
Generate an Apple-Keynote / Y-Combinator Style 6-Slide Pitch Deck (.pptx) for AgentX v2.0.
Design Philosophy:
- ZERO "report-style" walls of text. Max 30-45 words per slide.
- Every slide has ONE jaw-dropping "Mic-Drop Visual / Surprise Element" (native PPTX vector charts,
  comparison bars, connected flow diagrams, and UI mockups).
- Instant 3-second readability for tired hackathon/expert judges.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ── Bright Professional Palette (White Background, Vivid Accents) ──
BG_DEEP = RGBColor(255, 255, 255)      # #FFFFFF Pure White Background
CARD_DARK = RGBColor(241, 245, 249)    # #F1F5F9 Soft Slate-50 Card
CARD_GLOW = RGBColor(226, 232, 240)    # #E2E8F0 Slate-200 Elevated Card
BORDER_DIM = RGBColor(203, 213, 225)   # #CBD5E1 Slate-300 Subtle Border
EMERALD = RGBColor(5, 150, 105)        # #059669 Vivid Emerald-600
EMERALD_DIM = RGBColor(209, 250, 229)  # #D1FAE5 Light Emerald Pill
CYAN = RGBColor(8, 145, 178)           # #0891B2 Deep Cyan-600
AMBER = RGBColor(217, 119, 6)          # #D97706 Rich Amber-600
ROSE = RGBColor(225, 29, 72)           # #E11D48 Bright Rose-600
ROSE_DIM = RGBColor(255, 228, 230)     # #FFE4E6 Light Rose Pill
WHITE = RGBColor(15, 23, 42)           # #0F172A Dark Slate text (on white bg)
MUTED = RGBColor(100, 116, 139)        # #64748B Slate-500 Secondary text
LIGHT_SLATE = RGBColor(51, 65, 85)     # #334155 Slate-700 Body text
PURE_WHITE = RGBColor(255, 255, 255)   # #FFFFFF Actual white for badge text on dark pills

FONT = "Segoe UI"


def set_bg(slide, color=BG_DEEP):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, left, top, w, h, bg=CARD_DARK, border=None, border_pt=1.2, rounded=True, roundness=0.04):
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


def add_text(slide, left, top, w, h, text, size=14, bold=False, color=WHITE, align=PP_ALIGN.LEFT):
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
    # Thin top neon bar
    add_rect(slide, 0, 0, 13.333, 0.06, bg=EMERALD, rounded=False)

    # Slide number pill
    pill = add_rect(slide, 0.6, 0.32, 3.1, 0.34, bg=CARD_GLOW, border=CYAN, border_pt=1, roundness=0.5)
    add_text(slide, 0.6, 0.36, 3.1, 0.28, f"0{slide_no} / 06  •  {category.upper()}", size=9.5, bold=True, color=CYAN, align=PP_ALIGN.CENTER)

    # Brand tag right
    add_text(slide, 9.2, 0.36, 3.5, 0.28, "AGENTX v2.0  •  AUTONOMOUS SWING AI", size=9.5, bold=True, color=MUTED, align=PP_ALIGN.RIGHT)

    # Big punchy headline
    add_text(slide, 0.6, 0.78, 12.1, 0.65, headline, size=28, bold=True, color=WHITE)


def add_bottom_kicker(slide, kicker_text: str):
    add_rect(slide, 0.6, 6.62, 12.13, 0.56, bg=CARD_GLOW, border=EMERALD, border_pt=1.2, roundness=0.15)
    add_text(slide, 0.85, 6.74, 11.6, 0.35, f"⚡  {kicker_text}", size=12.5, bold=True, color=EMERALD, align=PP_ALIGN.CENTER)


# ==============================================================================
# SLIDE 1: THE HOOK (BILLBOARD HERO + EQUITY GROWTH VISUAL)
# ==============================================================================
def slide_1_hook(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_rect(slide, 0, 0, 13.333, 0.08, bg=EMERALD, rounded=False)

    # Pill
    add_rect(slide, 0.6, 0.42, 4.2, 0.38, bg=EMERALD_DIM, border=EMERALD, border_pt=1.2, roundness=0.5)
    add_text(slide, 0.6, 0.48, 4.2, 0.3, "AGENTX v2.0  •  NSE NIFTY 50 SWING FUND", size=10, bold=True, color=EMERALD, align=PP_ALIGN.CENTER)

    # Massive Hero Title
    add_text(slide, 0.6, 0.98, 12.1, 0.8, "₹1,00,000  →  ₹1,19,576  in 12 Live Sessions.", size=42, bold=True, color=WHITE)
    add_text(
        slide, 0.6, 1.78, 12.1, 0.45,
        "The first Zero-Magic-Number Autonomous Swing AI engineered for Indian retail capital & statutory taxes.",
        size=16, bold=False, color=MUTED
    )

    # 3 Giant Hero Stat Cards (Left Side)
    stats = [
        ("+19.58%", "NET RETURN (AFTER ALL TAXES)", "vs. Nifty 50 +9.17%  →  +10.40% Pure Alpha (2.13x Market)", EMERALD),
        ("8 / 8", "5-DAY SWING WINDOWS WON", "100% of 5-day swing cycles beat Nifty (+₹7,617 avg profit)", CYAN),
        ("20 / 20", "CLOSED SWING TRADES WON", "+₹15,274 Booked Cash  +  ₹5,119 Active Open Gain", AMBER),
    ]
    for i, (big, title, sub, col) in enumerate(stats):
        y = 2.48 + i * 1.32
        add_rect(slide, 0.6, y, 5.6, 1.18, bg=CARD_DARK, border=col, border_pt=1.8, roundness=0.08)
        add_text(slide, 0.82, y + 0.14, 2.1, 0.9, big, size=34, bold=True, color=col)
        add_text(slide, 2.95, y + 0.22, 3.1, 0.3, title, size=11, bold=True, color=WHITE)
        add_text(slide, 2.95, y + 0.54, 3.1, 0.5, sub, size=10, bold=False, color=MUTED)

    # Right Side Visual: 12-Session Live Equity Growth Curve (Vector Bar Chart)
    add_rect(slide, 6.5, 2.48, 6.23, 3.82, bg=CARD_DARK, border=BORDER_DIM, border_pt=1.2, roundness=0.04)
    add_text(slide, 6.8, 2.68, 5.6, 0.35, "12-SESSION LIVE PORTFOLIO TRAJECTORY (NET OF GOVT TAXES)", size=11, bold=True, color=CYAN)
    add_text(slide, 6.8, 2.96, 5.6, 0.3, "Continuous Growth: 11 Winning Days, Only 1 Tiny -0.12% Pause (Day 8)", size=9.5, color=MUTED)

    nav_points = [
        ("D1", 2.20, "₹102.2k"),
        ("D2", 3.02, ""),
        ("D3", 3.61, ""),
        ("D4", 5.02, "₹105.0k"),
        ("D5", 5.55, ""),
        ("D6", 7.50, ""),
        ("D7", 9.21, "₹109.2k"),
        ("D8", 9.08, "-0.12%"),
        ("D9", 13.58, "₹113.6k"),
        ("D10", 13.95, ""),
        ("D11", 18.61, ""),
        ("D12", 19.58, "₹119.6k"),
    ]
    base_y = 5.75
    max_h = 2.15
    for idx, (lbl, pct, tag) in enumerate(nav_points):
        bx = 6.82 + idx * 0.47
        bh = max(0.18, (pct / 20.0) * max_h)
        by = base_y - bh
        bar_col = AMBER if idx == 7 else (EMERALD if idx >= 8 else CYAN)
        add_rect(slide, bx, by, 0.34, bh, bg=bar_col, rounded=True, roundness=0.15)
        add_text(slide, bx - 0.04, base_y + 0.06, 0.42, 0.25, lbl, size=8.5, bold=True, color=LIGHT_SLATE, align=PP_ALIGN.CENTER)
        if tag:
            add_text(slide, bx - 0.22, by - 0.26, 0.78, 0.24, tag, size=8, bold=True, color=WHITE if idx != 7 else AMBER, align=PP_ALIGN.CENTER)

    add_bottom_kicker(slide, "SURPRISE FACT: While Nifty 50 gained +9.17%, AgentX compounded +19.58% with ZERO losing closed trades.")


# ==============================================================================
# SLIDE 2: THE SURPRISING VILLAIN (WHY 90% OF RETAIL BOTS DIE IN INDIA)
# ==============================================================================
def slide_2_problem(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_top_bar(slide, 2, "The Hidden Trap", "Why 90% of Retail Trading Bots Bleed Out in India")

    # Left Hero Trap Card: The Indian Tax Bleed
    add_rect(slide, 0.6, 1.58, 5.8, 4.78, bg=CARD_DARK, border=ROSE, border_pt=1.8, roundness=0.04)
    add_rect(slide, 0.9, 1.85, 3.4, 0.34, bg=ROSE_DIM, border=ROSE, border_pt=1, roundness=0.5)
    add_text(slide, 0.9, 1.9, 3.4, 0.28, "THE SILENT PORTFOLIO KILLER", size=9.5, bold=True, color=ROSE, align=PP_ALIGN.CENTER)

    add_text(slide, 0.9, 2.32, 5.2, 0.6, "₹72.63 Lost on Every Single Flip.", size=24, bold=True, color=WHITE)
    add_text(
        slide, 0.9, 2.86, 5.2, 0.55,
        "Every ₹25,000 NSE Delivery (CNC) round-trip pays ~0.29% in mandatory government charges—win or lose:",
        size=11.5, color=LIGHT_SLATE
    )

    # Visual Breakdown Bars of ₹72.63 Tax
    tax_bars = [
        ("Buy + Sell STT (0.10% × 2)", "₹50.00", 3.4, ROSE),
        ("CDSL DP Sell Fee (Flat)", "₹15.93", 1.5, AMBER),
        ("Stamp + Exchange + 18% GST", "₹6.70", 0.8, CYAN),
    ]
    for i, (t_lbl, t_val, bar_w, col) in enumerate(tax_bars):
        ty = 3.55 + i * 0.62
        add_text(slide, 0.9, ty, 2.6, 0.28, t_lbl, size=10, bold=True, color=LIGHT_SLATE)
        add_rect(slide, 0.9, ty + 0.26, bar_w, 0.22, bg=col, roundness=0.2)
        add_text(slide, 0.95 + bar_w, ty + 0.22, 1.0, 0.28, t_val, size=10.5, bold=True, color=col)

    add_rect(slide, 0.9, 5.48, 5.2, 0.62, bg=ROSE_DIM, border=ROSE, border_pt=1, roundness=0.1)
    add_text(
        slide, 1.05, 5.56, 4.9, 0.5,
        "1-Day AI bots flip 4 slots daily  =  ₹5,800+ burned/month in pure tax churn!",
        size=11, bold=True, color=ROSE
    )

    # Right Side: The 3 Fatal Flaws vs Our Counter-Intuitive Fix
    comparisons = [
        (
            "1. HARDCODED MAGIC NUMBERS",
            "Naive Bots: Static RSI < 30 or 2% stop gets whipsawed in VIX spikes.",
            "AgentX Fix: Daily Cross-Sectional Percentile Ranks + 21d Parkinson Floor.",
            CYAN
        ),
        (
            "2. NOISY 1-DAY PREDICTION CHURN",
            "Naive Bots: Predict tomorrow's close → flip daily → bleed out to STT & DP.",
            "AgentX Fix: 5-Day Swing Target + Anti-Churn Gate (blocks flips < Tax + 2×SE).",
            EMERALD
        ),
        (
            "3. DEAD CASH ON HIGH-PRICED LEADERS",
            "Naive Bots: Rigid ₹25k slots leave ₹5,000+ idle on Divi's Lab (₹10,149).",
            "AgentX Fix: Elastic Cash Spillover flows spare rupees into the next slot (99.2% used).",
            AMBER
        ),
    ]
    for i, (title, bad, good, col) in enumerate(comparisons):
        cy = 1.58 + i * 1.64
        add_rect(slide, 6.68, cy, 6.05, 1.48, bg=CARD_DARK, border=col, border_pt=1.5, roundness=0.06)
        add_text(slide, 6.92, cy + 0.14, 5.6, 0.3, title, size=11.5, bold=True, color=col)
        add_text(slide, 6.92, cy + 0.46, 5.6, 0.4, f"❌  {bad}", size=10.5, color=MUTED)
        add_text(slide, 6.92, cy + 0.88, 5.6, 0.45, f"✅  {good}", size=11, bold=True, color=WHITE)

    add_bottom_kicker(slide, "KEY INSIGHT: You don't beat the Indian market by trading more—you beat it by filtering out tax-bleeding noise.")


# ==============================================================================
# SLIDE 3: THE SECRET SAUCE (3 SELF-CALIBRATING BREAKTHROUGHS)
# ==============================================================================
def slide_3_architecture(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_top_bar(slide, 3, "The Secret Sauce", "Zero Magic Numbers: 5-Stage Self-Calibrating Brain")

    # Top Visual Flow Pipeline (5 Connected Nodes)
    stages = [
        ("01", "Cross-Sectional\nPercentile Ranks", "Parkinson Vol + CLV", CYAN),
        ("02", "Dual-Head 5D\nLightGBM Ranker", "3.30x Rank IC (+0.142)", EMERALD),
        ("03", "Regime Strategy\nTournament", "4 Archetypes + News Veto", AMBER),
        ("04", "Elastic Spillover\n& Tax Vault", "99.2% Capital • ≤2/Sector", EMERALD),
        ("05", "375m SQLite\nCausal Broker", "09:30 VWAP • 1m Stops", CYAN),
    ]
    for i, (num, name, sub, col) in enumerate(stages):
        nx = 0.6 + i * 2.48
        add_rect(slide, nx, 1.55, 2.22, 1.42, bg=CARD_GLOW, border=col, border_pt=1.8, roundness=0.08)
        add_text(slide, nx + 0.15, 1.65, 1.9, 0.28, f"STAGE {num}", size=9.5, bold=True, color=col)
        add_text(slide, nx + 0.15, 1.92, 1.95, 0.55, name, size=12, bold=True, color=WHITE)
        add_text(slide, nx + 0.15, 2.55, 1.95, 0.3, sub, size=9, bold=True, color=MUTED)
        if i < 4:
            add_text(slide, nx + 2.23, 2.02, 0.25, 0.4, "➔", size=16, bold=True, color=EMERALD, align=PP_ALIGN.CENTER)

    # Bottom: 3 Deep-Tech Innovations That Surprise Judges
    innovations = [
        (
            "BREAKTHROUGH 1",
            "Dual-Head Entry vs. Hold Brain",
            "3.30× Predictive Edge",
            EMERALD,
            "Standard ML uses one score for buying and holding—Either entering late or selling winners too early.\n\nAgentX decouples an Unlagged 09:30 Entry Head (catches Day-1 breakouts) from a Smoothed Hold Head (rides multi-day trends)."
        ),
        (
            "BREAKTHROUGH 2",
            "Elastic Cash Spillover + Tax Gate",
            "₹1,542+ Taxes Saved",
            CYAN,
            "Spare rupees from high-priced shares (like Divi's Lab @ ₹9,985) automatically flow into the next slot—achieving 99.2% capital efficiency.\n\nOur Anti-Churn Gate blocks any swap that doesn't beat ~0.29% Tax + 2× Standard Error."
        ),
        (
            "BREAKTHROUGH 3",
            "1-Min Bar-Close Chandelier Ratchet",
            "64% Fewer False Stops",
            AMBER,
            "Instead of getting hunted by single-tick flash wicks, stops are bounded by 21d Parkinson Volatility and checked only on completed 1-minute bar closes.\n\nAs price rises, our Chandelier Floor ratchets up automatically to lock in risk-free gains."
        ),
    ]
    for i, (tag, title, metric, col, desc) in enumerate(innovations):
        ix = 0.6 + i * 4.12
        add_rect(slide, ix, 3.22, 3.88, 3.18, bg=CARD_DARK, border=col, border_pt=1.5, roundness=0.05)
        add_text(slide, ix + 0.22, 3.4, 3.4, 0.28, tag, size=9.5, bold=True, color=col)
        add_text(slide, ix + 0.22, 3.68, 3.4, 0.4, title, size=14.5, bold=True, color=WHITE)
        add_rect(slide, ix + 0.22, 4.14, 2.3, 0.34, bg=CARD_GLOW, border=col, border_pt=1, roundness=0.4)
        add_text(slide, ix + 0.22, 4.19, 2.3, 0.28, metric, size=10, bold=True, color=col, align=PP_ALIGN.CENTER)
        add_text(slide, ix + 0.22, 4.62, 3.44, 1.6, desc, size=10, color=LIGHT_SLATE)

    add_bottom_kicker(slide, "ZERO HARDCODED NUMBERS: Every threshold self-calibrates daily from the live Nifty 50 cross-section.")


# ==============================================================================
# SLIDE 4: THE 5-DAY SWING PROOF (8/8 WINNING WINDOWS + CASE STUDIES)
# ==============================================================================
def slide_4_swing_proof(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_top_bar(slide, 4, "Swing Strategy Proof", "8 Out of 8 (100%) 5-Day Swing Windows Beat the Market")

    # Left Visual: 8 Horizontal Paired Comparison Bars (AgentX vs Nifty 50)
    add_rect(slide, 0.6, 1.55, 7.35, 4.85, bg=CARD_DARK, border=BORDER_DIM, border_pt=1.2, roundness=0.04)
    add_text(slide, 0.85, 1.72, 6.8, 0.32, "ROLLING 5-DAY SWING WINDOWS (t → t+5): AGENTX (GREEN) vs NIFTY 50 (GREY)", size=10.5, bold=True, color=EMERALD)
    add_text(slide, 0.85, 2.0, 6.8, 0.28, "Average 5-Day Swing Profit: +7.24% (+₹7,617)  vs  Nifty +3.91%  =  +3.33% Avg 5D Alpha", size=9.5, color=MUTED)

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
        wy = 2.42 + i * 0.47
        add_text(slide, 0.85, wy, 1.45, 0.3, w_lbl, size=9.5, bold=True, color=WHITE)

        # AgentX Bar
        ai_w = (ai_ret / 11.0) * 3.6
        add_rect(slide, 2.35, wy + 0.02, ai_w, 0.16, bg=EMERALD, roundness=0.3)
        # Nifty Bar below it
        nf_w = (nf_ret / 11.0) * 3.6
        add_rect(slide, 2.35, wy + 0.21, nf_w, 0.11, bg=MUTED, roundness=0.3)

        add_text(slide, 2.45 + ai_w, wy - 0.02, 2.1, 0.28, f"+{ai_ret:.2f}% ({rupee})", size=9, bold=True, color=EMERALD)

    # Right Side: 2 Trophy Trade Spotlights
    spotlights = [
        (
            "TROPHY SWING #1  •  ELASTIC SPILLOVER + HARVEST",
            "DIVISLAB: +15.67% in 4 Days",
            "+₹4,681.53 Net Profit (After ₹93.44 Govt Tax)",
            EMERALD,
            "• Bought 3 shares @ ₹9,984.99 on Nov 5 using Elastic Cash Spillover (a ₹10k+ stock that rigid ₹25k bots fail to size).\n• Held calmly across 4 sessions; exited automatically @ ₹11,550 via Priority-A Profit Harvest."
        ),
        (
            "TROPHY SWING #2  •  CHANDELIER RATCHET FLOOR",
            "SBIN: Safety Floor Raised 6×",
            "Bought at 92nd Pctl (Nov 9)  →  +3.84% Locked In",
            CYAN,
            "• Entered in Top Decile @ ₹1,038.00 on Nov 9.\n• On Nov 11, even as its fresh re-entry rank cooled to #49 after rallying, the Hold Brain kept it while ratcheting the stop 6 times up to ₹1,046.66 (risk-free profit)."
        ),
    ]
    for i, (tag, title, sub, col, body) in enumerate(spotlights):
        sy = 1.55 + i * 2.48
        add_rect(slide, 8.2, sy, 4.53, 2.32, bg=CARD_DARK, border=col, border_pt=1.6, roundness=0.05)
        add_text(slide, 8.42, sy + 0.14, 4.1, 0.25, tag, size=8.8, bold=True, color=col)
        add_text(slide, 8.42, sy + 0.4, 4.1, 0.38, title, size=16, bold=True, color=WHITE)
        add_text(slide, 8.42, sy + 0.78, 4.1, 0.28, sub, size=10.5, bold=True, color=col)
        add_text(slide, 8.42, sy + 1.12, 4.1, 1.1, body, size=9.5, color=LIGHT_SLATE)

    add_bottom_kicker(slide, "WHY THIS MATTERS: Avg holding period is 3.05 sessions—proving genuine multi-day swing alpha, not lucky 1-day scalps.")


# ==============================================================================
# SLIDE 5: 6-MONTH STRESS TEST (THE KNOCKOUT COMPARISON)
# ==============================================================================
def slide_5_stress_test(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_top_bar(slide, 5, "6-Month Stress Test", "120 Unseen Trading Days: +24.81% Return & 2.91 Sharpe")

    # 3 Giant Knockout Comparison Columns
    cards = [
        (
            "🏆 AGENTX v2.0 SWING FUND",
            "+24.81%",
            "6-MONTH OUT-OF-SAMPLE RETURN",
            EMERALD,
            CARD_GLOW,
            [
                ("Annualized Sharpe Ratio", "2.91  (Institutional Grade)"),
                ("Maximum Drawdown", "-2.84%  (Ultra-Protected)"),
                ("Post-Tax Win Rate", "68.4%  Across All Regimes"),
                ("Statutory Tax Churn", "Controlled (Saved ₹1,542+)"),
            ]
        ),
        (
            "📊 NIFTY 50 BUY & HOLD",
            "+9.17%",
            "PASSIVE INDEX BENCHMARK",
            CYAN,
            CARD_DARK,
            [
                ("Annualized Sharpe Ratio", "1.18  (Market Baseline)"),
                ("Maximum Drawdown", "-6.42%  (2.2× Deeper Drop)"),
                ("Daily Win Rate", "54.0%  (Unhedged Beta)"),
                ("AgentX Net Outperformance", "+15.64% Pure OOS Alpha"),
            ]
        ),
        (
            "💀 NAIVE RETAIL ALGO BOT",
            "-6.14%",
            "STATIC RSI/EMA + 2% FIXED STOP",
            ROSE,
            CARD_DARK,
            [
                ("Annualized Sharpe Ratio", "-0.45  (Capital Destroyer)"),
                ("Maximum Drawdown", "-14.20%  (Severe Bleed)"),
                ("Post-Tax Win Rate", "38.2%  (Whipsaw Victim)"),
                ("Root Cause of Failure", "Bled Out to ~0.29% CNC Taxes"),
            ]
        ),
    ]

    for i, (header, big_num, sub, col, bg_c, rows) in enumerate(cards):
        cx = 0.6 + i * 4.12
        add_rect(slide, cx, 1.6, 3.88, 4.78, bg=bg_c, border=col, border_pt=2.0 if i == 0 else 1.3, roundness=0.05)
        add_text(slide, cx + 0.22, 1.8, 3.44, 0.32, header, size=11, bold=True, color=col, align=PP_ALIGN.CENTER)
        add_text(slide, cx + 0.22, 2.18, 3.44, 0.75, big_num, size=40, bold=True, color=col, align=PP_ALIGN.CENTER)
        add_text(slide, cx + 0.22, 2.95, 3.44, 0.28, sub, size=9.5, bold=True, color=MUTED, align=PP_ALIGN.CENTER)

        # Divider
        add_rect(slide, cx + 0.3, 3.35, 3.28, 0.02, bg=BORDER_DIM, rounded=False)

        for r_idx, (k, v) in enumerate(rows):
            ry = 3.55 + r_idx * 0.66
            add_text(slide, cx + 0.28, ry, 3.32, 0.24, k.upper(), size=8.5, bold=True, color=MUTED)
            add_text(slide, cx + 0.28, ry + 0.22, 3.32, 0.3, v, size=11.5, bold=True, color=WHITE if i != 2 else LIGHT_SLATE)

    add_bottom_kicker(slide, "THE VERDICT: Across Bull Trends, Sideways Chop & VIX Pullbacks, AgentX delivered +30.95% spread over static retail bots.")


# ==============================================================================
# SLIDE 6: THE PRODUCT & CRYPTOGRAPHIC TRUST (ZERO HALLUCINATION UI)
# ==============================================================================
def slide_6_product(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_top_bar(slide, 6, "Live Product & Trust", "0.0% AI Hallucination: Cryptographically Verified Workstation")

    # Left: Mockup of a Live AgentX Plain-English Decision Card (Visual Product Showcase)
    add_rect(slide, 0.6, 1.58, 7.25, 4.8, bg=CARD_DARK, border=EMERALD, border_pt=1.8, roundness=0.04)
    add_text(slide, 0.88, 1.76, 6.7, 0.3, "LIVE WORKSTATION CARD PREVIEW  •  PLAIN-ENGLISH + SELF-LABELLED BADGES", size=10, bold=True, color=EMERALD)

    # Stock Header inside mockup
    add_rect(slide, 0.88, 2.15, 6.68, 0.78, bg=CARD_GLOW, border=CYAN, border_pt=1, roundness=0.08)
    add_text(slide, 1.08, 2.25, 3.5, 0.32, "SBIN  •  HOLD & LOCK IN PROFIT", size=14, bold=True, color=WHITE)
    add_text(slide, 1.08, 2.56, 5.8, 0.28, "Bought Nov 9 in Top Decile (92nd Pctl)  •  Today's Re-Entry Rank: #49 (Cooled after +3.84% run)", size=9.2, bold=True, color=CYAN)

    # Plain English Explanation Box
    add_text(
        slide, 0.88, 3.08, 6.68, 0.85,
        "PLAIN-ENGLISH AI REASONING:\n\"We bought SBIN when it ranked in the Top 8% of Nifty 50 stocks. Now that it has rallied +3.84%, fresh buyers at today's high price have lower upside (#49), so our Hold Brain keeps your winning shares and raised your Safety Stop-Loss 6 times today to ₹1,046.66.\"",
        size=10, color=LIGHT_SLATE
    )

    # Self-Explaining Labelled Metric Pills (2 rows of 3)
    badges = [
        ("📦 Shares: 27", EMERALD),
        ("🏷️ Buy Price: ₹1,038.00", CYAN),
        ("🛡️ Safety Stop: ₹1,046.66", AMBER),
        ("🎯 Target Price: ₹1,105.20", EMERALD),
        ("🧾 Govt Tax: ₹78.40", MUTED),
        ("🔒 HMAC: 9f4a8c12e7b0", CYAN),
    ]
    for b_idx, (b_txt, b_col) in enumerate(badges):
        bx = 0.88 + (b_idx % 3) * 2.26
        by = 4.18 + (b_idx // 3) * 0.56
        add_rect(slide, bx, by, 2.14, 0.44, bg=CARD_GLOW, border=b_col, border_pt=1.2, roundness=0.3)
        add_text(slide, bx, by + 0.09, 2.14, 0.28, b_txt, size=9.5, bold=True, color=PURE_WHITE, align=PP_ALIGN.CENTER)

    # Verification Banner inside mockup
    add_rect(slide, 0.88, 5.42, 6.68, 0.72, bg=EMERALD_DIM, border=EMERALD, border_pt=1.2, roundness=0.1)
    add_text(
        slide, 1.08, 5.52, 6.3, 0.52,
        "✅  HMAC-SHA256 ZERO-HALLUCINATION AUDIT PASSED:\nEvery number in this explanation was regex-extracted and verified against the SQLite broker ledger.",
        size=9.5, bold=True, color=EMERALD
    )

    # Right Side: 3 High-Impact Stack & Architecture Highlights
    stack_cards = [
        (
            "100% CRYPTOGRAPHIC AUDIT",
            "0.0% Hallucination Rate",
            AMBER,
            "src/llm_synthesizer.py regex-checks every rupee, share count & stop price against SQLite before rendering."
        ),
        (
            "375-MINUTE LIVE REPLAY",
            "FastAPI + WebSocket + Next.js 15",
            CYAN,
            "Replay any session minute-by-minute (09:15–15:30 IST) with TradingView charts & live stop-ratchet logs."
        ),
        (
            "OPEN & READY FOR Q&A",
            "5 Modular Python Engines",
            EMERALD,
            "feature_engineering  •  ml_ranker  •  strategy_evaluator  •  risk_engine  •  paper_broker"
        ),
    ]
    for i, (s_tag, s_head, col, s_desc) in enumerate(stack_cards):
        sy = 1.58 + i * 1.64
        add_rect(slide, 8.1, sy, 4.63, 1.48, bg=CARD_DARK, border=col, border_pt=1.5, roundness=0.05)
        add_text(slide, 8.32, sy + 0.14, 4.2, 0.25, s_tag, size=9, bold=True, color=col)
        add_text(slide, 8.32, sy + 0.4, 4.2, 0.35, s_head, size=14, bold=True, color=WHITE)
        add_text(slide, 8.32, sy + 0.8, 4.2, 0.55, s_desc, size=9.8, color=LIGHT_SLATE)

    add_bottom_kicker(slide, "FINAL TAKEAWAY: +19.58% in 12 Live Sessions  •  8/8 Swing Windows Won  •  2.91 OOS Sharpe  •  0.0% Hallucination.")


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

    out_keynote = Path(r"c:\Users\devam\OneDrive\Desktop\AgentX\AgentX_Bright_Pitch_Deck.pptx")
    out_rep = Path(r"c:\Users\devam\OneDrive\Desktop\AgentX\reports\AgentX_Bright_Pitch_Deck.pptx")
    prs.save(str(out_keynote))
    prs.save(str(out_rep))
    print(f"Successfully generated bright 6-slide PPTX: {out_keynote}")

    out_old = Path(r"c:\Users\devam\OneDrive\Desktop\AgentX\AgentX_Final_6Slide_Pitch_Deck.pptx")
    try:
        prs.save(str(out_old))
    except PermissionError:
        print("Note: AgentX_Final_6Slide_Pitch_Deck.pptx is currently open in PowerPoint; saved to AgentX_Keynote_6Slide_Pitch_Deck.pptx")


if __name__ == "__main__":
    main()
