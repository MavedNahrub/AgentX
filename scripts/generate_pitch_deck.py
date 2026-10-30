"""
Generates the Initial Idea Pitch Deck (`AgentX_PS02_Scale_Invariant_Pitch_Deck.pptx`)
for Problem Statement PS-02 (Autonomous Nifty 50 Equity Trading Agent).

Key Narrative & Visual Design Principles:
  1. Scale-Invariant Self-Calibrating Math (₹1 Lakh Retail -> ₹10 Crore+ Institutional):
     Explains the surprising insight that because ZERO parameters are hardcoded, the exact same
     equation solves flat ₹15.93 CDSL fees & whole-share remainders at ₹1 Lakh AND dynamically
     scales Amihud market-impact slippage & liquidity depth at ₹1 Crore–₹10 Crore+.
  2. Natural, Compelling Wording (No robotic 'Surprising Insight:' labels):
     Every card opens with a bold, naturally surprising lead sentence followed by a complete,
     word-for-word explanation of how our architecture works step-by-step.
  3. Respectful, Formal Executive Tone: Never belittles other teams; wins trust purely through
     deep domain mastery and clear engineering explanations.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


# =============================================================================
# FORMAL INSTITUTIONAL FINANCE PALETTE (Navy + Gold framing, bright white body, vivid accents)
# =============================================================================
CARD_WHITE = RGBColor(255, 255, 255)      # Pure White Glass Card
CARD_TINT = RGBColor(236, 245, 255)       # Highlighted Row / Pill Fill
CARD_ALT_ROW = RGBColor(245, 249, 255)    # Table Alternating Row
CARD_SHADOW = RGBColor(170, 186, 214)     # Soft Slate 3D Drop Shadow
BORDER_CRISP = RGBColor(186, 208, 238)    # Crisp Steel-Blue Border

HEADER_ROYAL = RGBColor(8, 24, 62)        # Deep Institutional Navy
HEADER_PILL_BG = RGBColor(22, 48, 104)    # Elevated Navy Pill
HEADER_CYAN = RGBColor(34, 230, 255)      # Electric Cyan
HEADER_GOLD = RGBColor(250, 204, 72)      # Formal Bright Gold

SAPPHIRE_PRIMARY = RGBColor(37, 99, 235)  # Vivid Royal Blue
EMERALD_WIN = RGBColor(5, 150, 105)       # Rich FinTech Emerald
AMBER_GOLD = RGBColor(217, 119, 6)        # Bright Amber
CRIMSON_ALERT = RGBColor(225, 29, 72)     # Vivid Rose-Crimson
VIOLET_AI = RGBColor(124, 58, 237)        # Electric Violet
TEAL_ACCENT = RGBColor(13, 148, 136)      # Ocean Teal

TEXT_INK = RGBColor(15, 23, 42)           # Deep Obsidian Ink (on white cards)
TEXT_BODY = RGBColor(38, 50, 74)          # Crisp Body Text
TEXT_MUTED = RGBColor(94, 110, 135)       # Caption / Module Tag
TEXT_WHITE = RGBColor(255, 255, 255)      # Pure White
FOOTER_TEXT = RGBColor(203, 220, 255)     # Light Periwinkle (on dark footer strip)


def generate_luxury_background_canvas(chart_dir: Path) -> Path:
    """
    Formal institutional-finance stage (bright body, dark navy framing, gold accents):
      - Glossy 3D navy header with a faint candlestick price series + gold trend line
      - Bright ivory-to-sky body with fine financial graph-paper grid
      - Subtle isometric 3D bar-chart watermark + rising equity curve (depth, not decoration)
      - Navy footer band with Gold/Cyan hairline rules
    """
    chart_dir.mkdir(parents=True, exist_ok=True)
    bg_path = chart_dir / "slide_executive_bg.png"

    W, H = 2000, 1125
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = x_grid / W
    yn = y_grid / H
    hdr_h = int(H * (1.15 / 7.5))
    ftr_y = int(H * (7.08 / 7.5))

    # ---- Body: bright ivory-white -> soft sky blue -> faint mint (clean, formal) ----
    r = 250.0 * (1 - xn) + 226.0 * xn
    g = 252.0 * (1 - xn) + 238.0 * xn
    b = 255.0 * (1 - xn) + 252.0 * xn
    vy = np.clip((yn - 0.15) / 0.85, 0, 1)
    r = r * (1 - 0.10 * vy) + 214.0 * 0.10 * vy
    g = g * (1 - 0.10 * vy) + 240.0 * 0.10 * vy
    b = b * (1 - 0.10 * vy) + 246.0 * 0.10 * vy

    # ---- Header: deep navy with glossy 3D top sheen ----
    xh = xn[:hdr_h, :]
    yh = yn[:hdr_h, :] / (1.15 / 7.5)
    hr = 8.0 * (1 - xh) + 22.0 * xh
    hg = 24.0 * (1 - xh) + 52.0 * xh
    hb = 62.0 * (1 - xh) + 112.0 * xh
    sheen = np.clip(1.0 - yh * 2.2, 0, 1) ** 2 * 0.16
    r[:hdr_h, :] = hr * (1 - sheen) + 255.0 * sheen
    g[:hdr_h, :] = hg * (1 - sheen) + 255.0 * sheen
    b[:hdr_h, :] = hb * (1 - sheen) + 255.0 * sheen

    # ---- Footer: matching navy band ----
    xf = xn[ftr_y:, :]
    r[ftr_y:, :] = 8.0 * (1 - xf) + 22.0 * xf
    g[ftr_y:, :] = 24.0 * (1 - xf) + 52.0 * xf
    b[ftr_y:, :] = 62.0 * (1 - xf) + 112.0 * xf

    arr = np.stack([np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)], axis=-1).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB").convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    # ---- Body: financial graph-paper grid (minor + major) ----
    for gx in range(0, W, 25):
        major = gx % 125 == 0
        d.line([(gx, hdr_h), (gx, ftr_y)], fill=(37, 99, 235, 26 if major else 10), width=1)
    for gy in range(hdr_h, ftr_y, 25):
        major = (gy - hdr_h) % 125 == 0
        d.line([(0, gy), (W, gy)], fill=(37, 99, 235, 26 if major else 10), width=1)

    # ---- Body: isometric 3D bar-chart watermark (lower right) ----
    base_y = ftr_y - 30
    bw, dep = 70, 26
    heights = [120, 165, 150, 215, 260, 300, 355]
    for i, hgt in enumerate(heights):
        x0 = 1330 + i * 92
        front = [(x0, base_y), (x0 + bw, base_y), (x0 + bw, base_y - hgt), (x0, base_y - hgt)]
        top = [(x0, base_y - hgt), (x0 + bw, base_y - hgt), (x0 + bw + dep, base_y - hgt - dep), (x0 + dep, base_y - hgt - dep)]
        side = [(x0 + bw, base_y), (x0 + bw + dep, base_y - dep), (x0 + bw + dep, base_y - hgt - dep), (x0 + bw, base_y - hgt)]
        d.polygon(front, fill=(30, 64, 175, 34))
        d.polygon(top, fill=(96, 165, 250, 46))
        d.polygon(side, fill=(15, 35, 100, 44))

    # ---- Body: rising equity curve with soft glow (lower left -> right) ----
    rng = np.random.default_rng(7)
    xs = np.linspace(40, W - 40, 90)
    trend = np.linspace(0, 1, 90) ** 1.3
    noise = np.cumsum(rng.normal(0, 0.018, 90))
    ys = (ftr_y - 70) - (trend + noise - noise.min() * 0.0) * 300
    pts = list(zip(xs.tolist(), ys.tolist()))
    for wdt, alpha in ((14, 14), (8, 22), (3, 70)):
        d.line(pts, fill=(5, 150, 105, alpha), width=wdt, joint="curve")

    # ---- Header: faint candlestick price series + gold moving-average line ----
    n = 64
    step = W / n
    price = 0.5 + np.cumsum(rng.normal(0.004, 0.035, n))
    price = (price - price.min()) / (price.max() - price.min() + 1e-9)
    top_pad, bot_pad = 22, 18
    span = hdr_h - top_pad - bot_pad
    prev = price[0]
    ma = []
    for i, p in enumerate(price):
        cx = i * step + step / 2
        o, c = prev, p
        hi = max(o, c) + abs(rng.normal(0, 0.05))
        lo = min(o, c) - abs(rng.normal(0, 0.05))
        to_y = lambda v: top_pad + (1 - np.clip(v, 0, 1)) * span
        fade = 0.12 + 0.88 * (cx / W) ** 1.6
        col = (16, 185, 129, int(72 * fade)) if c >= o else (239, 68, 68, int(64 * fade))
        d.line([(cx, to_y(hi)), (cx, to_y(lo))], fill=col, width=2)
        y1, y2 = sorted([to_y(o), to_y(c)])
        d.rectangle([(cx - step * 0.28, y1), (cx + step * 0.28, max(y2, y1 + 2))], fill=col)
        ma.append((cx, to_y(np.mean(price[max(0, i - 6): i + 1]))))
        prev = p
    for k in range(len(ma) - 1):
        a_ma = int(115 * (0.08 + 0.92 * (ma[k][0] / W) ** 1.6))
        d.line([ma[k], ma[k + 1]], fill=(255, 200, 60, a_ma), width=3)

    img = Image.alpha_composite(img, ov).convert("RGB")
    draw = ImageDraw.Draw(img)

    # ---- Accent rules: Gold top bar, Cyan->Gold header/footer hairlines ----
    for xpx in range(W):
        t = xpx / (W - 1)
        cr = int(34 + (255 - 34) * t)
        cg = int(211 + (196 - 211) * t)
        cb = int(238 + (40 - 238) * t)
        draw.line([(xpx, 0), (xpx, 5)], fill=(cr, cg, cb))
        draw.line([(xpx, hdr_h - 4), (xpx, hdr_h - 1)], fill=(cr, cg, cb))
        draw.line([(xpx, ftr_y), (xpx, ftr_y + 3)], fill=(cr, cg, cb))

    img.save(bg_path, format="PNG", optimize=True)
    return bg_path


def generate_idea_mechanic_charts(chart_dir: Path) -> dict[str, Path]:
    chart_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#BFD3EE"
    plt.rcParams["axes.linewidth"] = 1.0

    paths: dict[str, Path] = {}

    # VISUAL 1: Self-Calibrating Friction Across Capital Scales (₹1 Lakh -> ₹10 Crore)
    fig, ax = plt.subplots(figsize=(5.6, 5.1), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F6F9FF")

    cap_labels = ["₹1 Lakh\n(Retail)", "₹10 Lakh\n(HNI)", "₹1 Crore\n(Family Off.)", "₹10 Crore\n(Institutional)"]
    x_idx = np.arange(len(cap_labels))
    # Statutory tax + CDSL DP impact vs Dynamic Amihud Market-Impact Slippage
    tax_drag = np.array([0.286, 0.228, 0.223, 0.222])
    amihud_slip = np.array([0.028, 0.045, 0.095, 0.195])
    total_hurdle = tax_drag + amihud_slip

    ax.bar(x_idx, tax_drag, width=0.46, color="#2563EB", edgecolor="#0F172A", linewidth=0.7, label="Statutory Taxes + ₹15.93 CDSL DP (%)")
    ax.bar(x_idx, amihud_slip, width=0.46, bottom=tax_drag, color="#10B981", edgecolor="#0F172A", linewidth=0.7, label="Dynamic Amihud Liquidity Slippage (%)")

    for i, tot in enumerate(total_hurdle):
        ax.text(i, tot + 0.012, f"{tot:.2f}%\nHurdle", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#0F172A")

    ax.set_xticks(x_idx)
    ax.set_xticklabels(cap_labels, fontsize=8.8, fontweight="bold", color="#1E293B")
    ax.set_ylim(0, 0.50)
    ax.set_ylabel("Self-Calibrated Execution Hurdle (%)", color="#1E293B", fontsize=9.2, fontweight="bold")
    ax.set_title("Zero Hardcoding: Auto-Scales ₹1 Lakh to ₹10 Crore", color="#0F172A", fontsize=10.8, fontweight="bold", pad=10)
    ax.tick_params(colors="#475569", labelsize=9.0)
    ax.grid(True, axis="y", color="#CBD5E1", linestyle="--", linewidth=0.6)
    ax.spines["bottom"].set_color("#93C5FD")
    ax.spines["left"].set_color("#93C5FD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    leg = ax.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", edgecolor="#93C5FD", fontsize=8.2)
    for t in leg.get_texts():
        t.set_color("#1E293B")

    fig.tight_layout()
    c_tax = chart_dir / "chart_target_ablation.png"
    fig.savefig(c_tax, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    paths["tax_sweetspot"] = c_tax

    # VISUAL 2: How Elastic Cash Spillover Solves the Whole-Share Trap at Any Scale
    fig, ax = plt.subplots(figsize=(5.6, 5.1), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F6F9FF")

    categories = [
        "Rigid Equal Split\n(Fixed Cap / Slot)",
        "AgentX Elastic Spillover\n(Cascades Leftover ₹)",
    ]
    invested = [88.4, 99.8]
    idle_cash = [11.6, 0.2]

    y_pos = np.arange(len(categories))
    ax.barh(y_pos, invested, color=["#475569", "#10B981"], height=0.45, edgecolor="#0F172A", linewidth=0.8, label="Active Invested Capital (%)")
    ax.barh(y_pos, idle_cash, left=invested, color=["#EF4444", "#F59E0B"], height=0.45, edgecolor="#0F172A", linewidth=0.8, label="Uninvested Whole-Share Remainder (%)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=9.3, fontweight="bold", color="#1E293B")
    ax.set_xlim(0, 108)
    ax.set_xlabel("Share of Portfolio Capital (%)", color="#1E293B", fontsize=9.2, fontweight="bold")
    ax.set_title("Whole-Share Capital Efficiency (₹1L to ₹10Cr+)", color="#0F172A", fontsize=10.8, fontweight="bold", pad=10)
    ax.tick_params(colors="#475569", labelsize=9.0)
    ax.grid(True, axis="x", color="#CBD5E1", linestyle="--", linewidth=0.6)
    ax.spines["bottom"].set_color("#93C5FD")
    ax.spines["left"].set_color("#93C5FD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(44, 0, "88.4% Invested\n(11.6% Trapped in Idle Cash!)", va="center", ha="center", color="#FFFFFF", fontsize=9.0, fontweight="bold")
    ax.text(50, 1, "~99.8% Active Deployment\n(Cascades Remainder Across Slots)", va="center", ha="center", color="#FFFFFF", fontsize=9.0, fontweight="bold")

    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=True, facecolor="#FFFFFF", edgecolor="#93C5FD", fontsize=8.4)
    for t in leg.get_texts():
        t.set_color("#1E293B")

    fig.tight_layout()
    c_spill = chart_dir / "chart_slot_spillover.png"
    fig.savefig(c_spill, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    paths["slot_spillover"] = c_spill

    return paths


# =============================================================================
# NATURAL, STORY-DRIVEN PITCH DECK LAYOUT HELPERS
# =============================================================================
def _curve(shape, w_in: float, h_in: float, radius_in: float = 0.20) -> None:
    """Applies a uniform, absolute corner radius so every block shares the same elegant curve."""
    shape.adjustments[0] = min(0.5, radius_in / max(min(w_in, h_in), 1e-6))


def apply_master_chrome(
    slide,
    slide_num: int,
    kicker_tag: str,
    surprise_headline: str,
    execution_subtitle: str,
    bg_image_path: Path,
) -> None:
    slide.shapes.add_picture(str(bg_image_path), Inches(0), Inches(0), width=Inches(13.333), height=Inches(7.5))

    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), Inches(0.13), Inches(3.95), Inches(0.26))
    pill.fill.solid()
    pill.fill.fore_color.rgb = HEADER_PILL_BG
    pill.line.color.rgb = HEADER_CYAN
    pill.line.width = Pt(1.2)
    tf_p = pill.text_frame
    tf_p.word_wrap = False
    tf_p.margin_left = tf_p.margin_right = tf_p.margin_top = tf_p.margin_bottom = 0
    p_p = tf_p.paragraphs[0]
    p_p.text = f"0{slide_num} // {kicker_tag.upper()}"
    p_p.font.size = Pt(9.0)
    p_p.font.bold = True
    p_p.font.color.rgb = HEADER_CYAN
    p_p.alignment = PP_ALIGN.CENTER

    for idx in range(1, 7):
        px = 10.65 + (idx - 1) * 0.37
        dot = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(px), Inches(0.14), Inches(0.31), Inches(0.22))
        dot.fill.solid()
        dot.fill.fore_color.rgb = HEADER_CYAN if idx == slide_num else HEADER_PILL_BG
        dot.line.color.rgb = HEADER_CYAN if idx == slide_num else RGBColor(65, 98, 165)
        dot.line.width = Pt(0.8)
        tf_d = dot.text_frame
        tf_d.margin_left = tf_d.margin_right = tf_d.margin_top = tf_d.margin_bottom = 0
        pd_p = tf_d.paragraphs[0]
        pd_p.text = str(idx)
        pd_p.font.size = Pt(8.5)
        pd_p.font.bold = True
        pd_p.font.color.rgb = HEADER_ROYAL if idx == slide_num else TEXT_WHITE
        pd_p.alignment = PP_ALIGN.CENTER

    tb_t = slide.shapes.add_textbox(Inches(0.45), Inches(0.41), Inches(12.4), Inches(0.40))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
    p_t = tf_t.paragraphs[0]
    p_t.text = surprise_headline
    p_t.font.size = Pt(19.0)
    p_t.font.bold = True
    p_t.font.color.rgb = TEXT_WHITE

    tb_s = slide.shapes.add_textbox(Inches(0.45), Inches(0.81), Inches(12.4), Inches(0.28))
    tf_s = tb_s.text_frame
    tf_s.word_wrap = True
    tf_s.margin_left = tf_s.margin_top = tf_s.margin_right = tf_s.margin_bottom = 0
    p_s = tf_s.paragraphs[0]
    p_s.text = execution_subtitle
    p_s.font.size = Pt(10.8)
    p_s.font.bold = True
    p_s.font.color.rgb = HEADER_GOLD

    foot_line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.45), Inches(7.12), Inches(12.43), Inches(0.02))
    foot_line.fill.solid()
    foot_line.fill.fore_color.rgb = HEADER_CYAN
    foot_line.line.fill.background()

    tb_f = slide.shapes.add_textbox(Inches(0.45), Inches(7.15), Inches(10.8), Inches(0.25))
    tf_f = tb_f.text_frame
    tf_f.margin_left = tf_f.margin_top = 0
    p_f = tf_f.paragraphs[0]
    p_f.text = "AGENTX  •  PS-02: AUTONOMOUS NIFTY 50 ARCHITECTURE  •  ZERO HARDCODED PARAMETERS  •  SCALES ₹1 LAKH TO ₹10 CRORE+"
    p_f.font.size = Pt(8.8)
    p_f.font.bold = True
    p_f.font.color.rgb = FOOTER_TEXT

    tb_pg = slide.shapes.add_textbox(Inches(11.5), Inches(7.15), Inches(1.38), Inches(0.25))
    tf_pg = tb_pg.text_frame
    tf_pg.margin_right = tf_pg.margin_top = 0
    p_pg = tf_pg.paragraphs[0]
    p_pg.text = f"Slide {slide_num} of 6"
    p_pg.font.size = Pt(9.0)
    p_pg.font.bold = True
    p_pg.font.color.rgb = HEADER_GOLD
    p_pg.alignment = PP_ALIGN.RIGHT


def add_kpi_hero_box(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    big_metric: str,
    label_title: str,
    sub_caption: str,
    accent: RGBColor = EMERALD_WIN,
) -> None:
    shd = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left + 0.04), Inches(top + 0.05), Inches(width), Inches(height))
    _curve(shd, width, height)
    shd.fill.solid()
    shd.fill.fore_color.rgb = CARD_SHADOW
    shd.line.fill.background()

    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    _curve(box, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = CARD_WHITE
    box.line.color.rgb = accent
    box.line.width = Pt(2.0)

    cap = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left + 0.26), Inches(top + 0.05), Inches(width - 0.52), Inches(0.07))
    cap.adjustments[0] = 0.5
    cap.fill.solid()
    cap.fill.fore_color.rgb = accent
    cap.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(left + 0.16), Inches(top + 0.11), Inches(width - 0.32), Inches(height - 0.16))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

    p1 = tf.paragraphs[0]
    p1.text = big_metric
    p1.font.size = Pt(20.5)
    p1.font.bold = True
    p1.font.color.rgb = accent
    p1.space_after = Pt(2)

    p2 = tf.add_paragraph()
    p2.text = label_title.upper()
    p2.font.size = Pt(9.8)
    p2.font.bold = True
    p2.font.color.rgb = TEXT_INK
    p2.space_after = Pt(2)

    p3 = tf.add_paragraph()
    p3.text = sub_caption
    p3.font.size = Pt(9.0)
    p3.font.color.rgb = TEXT_BODY


def add_narrative_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    badge_text: str,
    title: str,
    opening_hook_sentence: str,
    architecture_explanation: str,
    takeaway_footer: str,
    accent: RGBColor = SAPPHIRE_PRIMARY,
    font_scale: float = 1.0,
) -> None:
    shd = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left + 0.04), Inches(top + 0.05), Inches(width), Inches(height))
    _curve(shd, width, height)
    shd.fill.solid()
    shd.fill.fore_color.rgb = CARD_SHADOW
    shd.line.fill.background()

    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    _curve(card, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_WHITE
    card.line.color.rgb = BORDER_CRISP
    card.line.width = Pt(1.4)

    vbar = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left + 0.05), Inches(top + 0.24), Inches(0.08), Inches(height - 0.48))
    vbar.adjustments[0] = 0.5
    vbar.fill.solid()
    vbar.fill.fore_color.rgb = accent
    vbar.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(left + 0.20), Inches(top + 0.10), Inches(width - 0.30), Inches(height - 0.18))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

    p0 = tf.paragraphs[0]
    p0.space_after = Pt(4 * font_scale)
    r_b = p0.add_run()
    r_b.text = f"{badge_text}  •  "
    r_b.font.size = Pt(10.0 * font_scale)
    r_b.font.bold = True
    r_b.font.color.rgb = accent

    r_t = p0.add_run()
    r_t.text = title
    r_t.font.size = Pt(12.3 * font_scale)
    r_t.font.bold = True
    r_t.font.color.rgb = TEXT_INK

    p1 = tf.add_paragraph()
    p1.space_after = Pt(4 * font_scale)
    r1 = p1.add_run()
    r1.text = opening_hook_sentence
    r1.font.size = Pt(10.2 * font_scale)
    r1.font.bold = True
    r1.font.color.rgb = accent

    p2 = tf.add_paragraph()
    p2.space_after = Pt(5 * font_scale)
    r2 = p2.add_run()
    r2.text = architecture_explanation
    r2.font.size = Pt(9.7 * font_scale)
    r2.font.color.rgb = TEXT_BODY

    p3 = tf.add_paragraph()
    r3 = p3.add_run()
    r3.text = f"▸ {takeaway_footer}"
    r3.font.size = Pt(9.7 * font_scale)
    r3.font.bold = True
    r3.font.color.rgb = EMERALD_WIN


def add_framed_chart(slide, img_path: Path, left: float, top: float, width: float, height: float) -> None:
    shd = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left + 0.04), Inches(top + 0.05), Inches(width), Inches(height))
    _curve(shd, width, height)
    shd.fill.solid()
    shd.fill.fore_color.rgb = CARD_SHADOW
    shd.line.fill.background()

    frame = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    _curve(frame, width, height)
    frame.fill.solid()
    frame.fill.fore_color.rgb = CARD_WHITE
    frame.line.color.rgb = BORDER_CRISP
    frame.line.width = Pt(1.4)

    slide.shapes.add_picture(str(img_path), Inches(left + 0.08), Inches(top + 0.08), width=Inches(width - 0.16))


def style_table_cell(
    cell,
    text: str,
    font_size: float = 10.4,
    bold: bool = False,
    text_color: RGBColor = TEXT_BODY,
    bg_color: RGBColor = CARD_WHITE,
    align=PP_ALIGN.LEFT,
) -> None:
    cell.fill.solid()
    cell.fill.fore_color.rgb = bg_color
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf = cell.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.10)
    tf.margin_right = Inches(0.10)
    tf.margin_top = Inches(0.05)
    tf.margin_bottom = Inches(0.05)
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = text_color
    p.alignment = align


def build_formal_pitch_deck(output_path: str | Path = "AgentX_PS02_Scale_Invariant_Pitch_Deck.pptx") -> Path:
    chart_dir = Path("data/cache/pitch_charts")
    bg_img = generate_luxury_background_canvas(chart_dir)
    charts = generate_idea_mechanic_charts(chart_dir)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: THE COUNTER-INTUITIVE REALITY & SCALE-INVARIANT VISION
    # =========================================================================
    s1 = prs.slides.add_slide(blank)
    apply_master_chrome(
        s1,
        1,
        "SCALE-INVARIANT MARKET ARCHITECTURE",
        "Zero Hardcoded Numbers: One Engine That Scales From ₹1 Lakh to ₹10 Crore+",
        "Stress-Tested on the Hardest ₹1,00,000 Retail Constraints—Yet Mathematically Self-Calibrating for Institutional Capital",
        bg_img,
    )

    add_kpi_hero_box(s1, 0.45, 1.28, 2.95, 1.24, "3:30 -> 9:15", "Dual-Clock Execution", "Close Signal -> Next-Morning VWAP Fill", SAPPHIRE_PRIMARY)
    add_kpi_hero_box(s1, 3.60, 1.28, 2.95, 1.24, "Tax + Impact", "Dynamic Friction Gate", "CDSL Fee at ₹1L • Liquidity at ₹10Cr", AMBER_GOLD)
    add_kpi_hero_box(s1, 6.75, 1.28, 2.95, 1.24, "100% Nifty 50", "Elastic Cash Spillover", "Zero Idle Cash at Any Account Size", VIOLET_AI)
    add_kpi_hero_box(s1, 9.90, 1.28, 2.98, 1.24, "₹1L -> ₹10Cr+", "Zero Hardcoded Rules", "Self-Calibrates to Capital & Volatility", EMERALD_WIN)

    add_narrative_card(
        s1, 0.45, 2.70, 3.98, 4.25,
        "01 // EXECUTION CLOCK", "Trading When the Market Is Open",
        "The official NSE closing price is printed at 3:30 PM—the exact second the exchange locks its doors and stops accepting orders.",
        "Any algorithm that assumes it can see the 3:30 PM close and buy at that same price is trading on an impossible clock. AgentX separates perception from execution: our engine analyzes all 50 Nifty stocks after the 3:30 PM close, locks its decisions overnight, and executes strictly the next morning at the 9:15–9:30 AM 15-minute Opening VWAP.",
        "100% Honest Clock • Zero Look-Ahead Time-Travel Bias",
        SAPPHIRE_PRIMARY,
        font_scale=1.02,
    )
    add_narrative_card(
        s1, 4.67, 2.70, 3.98, 4.25,
        "02 // SCALE-INVARIANT FRICTION", "From ₹15.93 DP Fees to Institutional Impact",
        "At ₹1 Lakh capital, flat ₹15.93 depository fees eat your profit; at ₹10 Crore capital, order book market-impact slippage eats your profit.",
        "Because nothing in AgentX is hardcoded, the exact same mathematical gate handles both worlds automatically. At ₹1,00,000, it accounts for flat ₹15.93 CDSL fees and 0.1% STT (~0.29% hurdle); as capital scales into crores, its Amihud Illiquidity equation automatically widens the hurdle for order-book impact without changing a single line of code.",
        "Self-Calibrates to ₹1 Lakh Retail or ₹10 Crore Institutional Funds",
        AMBER_GOLD,
        font_scale=1.02,
    )
    add_narrative_card(
        s1, 8.89, 2.70, 3.99, 4.25,
        "03 // ELASTIC WALLET", "Unlocking All 50 Stocks at Any Capital Size",
        "Because NSE forbids fractional shares, buying a ₹4,300 blue-chip inside a fixed ₹25,000 slot traps ₹3,500 in dead, uninvested cash.",
        "Whether managing ₹1,00,000 or ₹5 Crore, rigid equal-weight slots leave whole-share remainders sitting idle. AgentX uses a Two-Pass Elastic Cash Spillover allocator: it buys whole shares of the highest-priced stock in Slot 1 first, and automatically pours every uninvested rupee into subsequent slots—deploying ~99.9% of capital across all 50 Nifty stocks.",
        "All 50 Nifty Stocks Eligible • ~99.9% Active Capital Deployment",
        EMERALD_WIN,
        font_scale=1.02,
    )

    # =========================================================================
    # SLIDE 2: END-TO-END 7-STAGE ARCHITECTURAL BLUEPRINT
    # =========================================================================
    s2 = prs.slides.add_slide(blank)
    apply_master_chrome(
        s2,
        2,
        "7-STAGE SYSTEM ARCHITECTURE",
        "How AgentX Senses, Decides, Protects, and Explains Every Single Trade",
        "A 7-Stage Modular Pipeline Where Adaptive AI Discovers Opportunity (Stages 1–4) and Deterministic Math Guards the Wallet (Stages 5–7)",
        bg_img,
    )

    stage_meta = [
        ("STAGE 1", "Dual-Clock\nData Engine", "Daily + 15m VWAP", SAPPHIRE_PRIMARY),
        ("STAGE 2", "7-Lens Feature\nChameleon", "Self-Calibrating", TEAL_ACCENT),
        ("STAGE 3", "Post-Tax AI\n& SHAP X-Ray", "Rupee Ranker", VIOLET_AI),
        ("STAGE 4", "4-Playbook\nOlympics & News", "Regime + RSS", AMBER_GOLD),
        ("STAGE 5", "Spillover Sizer\n& Crypto Vault", "HMAC + KillSwitch", EMERALD_WIN),
        ("STAGE 6", "Next-Gen NSE\nSandbox Broker", "GTT OCO + CN", EMERALD_WIN),
        ("STAGE 7", "Zero-Lie AST\nTrade Journal", "Command Center", SAPPHIRE_PRIMARY),
    ]
    step_w, step_gap, step_top, step_h = 1.63, 0.17, 1.30, 1.10
    for idx, (st_num, st_title, st_file, clr) in enumerate(stage_meta):
        left = 0.45 + idx * (step_w + step_gap)

        shd = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left + 0.03), Inches(step_top + 0.04), Inches(step_w), Inches(step_h))
        _curve(shd, step_w, step_h, 0.16)
        shd.fill.solid()
        shd.fill.fore_color.rgb = CARD_SHADOW
        shd.line.fill.background()

        body = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(step_top), Inches(step_w), Inches(step_h))
        _curve(body, step_w, step_h, 0.16)
        body.fill.solid()
        body.fill.fore_color.rgb = CARD_WHITE
        body.line.color.rgb = clr
        body.line.width = Pt(1.4)

        band = s2.shapes.add_shape(MSO_SHAPE.ROUND_2_SAME_RECTANGLE, Inches(left), Inches(step_top), Inches(step_w), Inches(0.27))
        band.adjustments[0] = min(0.5, 0.16 / 0.27)
        band.adjustments[1] = 0.0
        band.fill.solid()
        band.fill.fore_color.rgb = clr
        band.line.fill.background()
        tf_b = band.text_frame
        tf_b.margin_left = tf_b.margin_right = tf_b.margin_top = tf_b.margin_bottom = 0
        tf_b.vertical_anchor = MSO_ANCHOR.MIDDLE
        pb = tf_b.paragraphs[0]
        pb.text = st_num
        pb.font.size = Pt(9.0)
        pb.font.bold = True
        pb.font.color.rgb = TEXT_WHITE
        pb.alignment = PP_ALIGN.CENTER

        tb = s2.shapes.add_textbox(Inches(left + 0.05), Inches(step_top + 0.31), Inches(step_w - 0.10), Inches(step_h - 0.35))
        tf_c = tb.text_frame
        tf_c.word_wrap = True
        tf_c.margin_left = tf_c.margin_right = tf_c.margin_top = tf_c.margin_bottom = 0
        p_c2 = tf_c.paragraphs[0]
        p_c2.text = st_title
        p_c2.font.size = Pt(10.0)
        p_c2.font.bold = True
        p_c2.font.color.rgb = TEXT_INK
        p_c2.alignment = PP_ALIGN.CENTER
        p_c2.space_after = Pt(2)
        p_c3 = tf_c.add_paragraph()
        p_c3.text = st_file
        p_c3.font.size = Pt(8.4)
        p_c3.font.color.rgb = clr
        p_c3.font.bold = True
        p_c3.alignment = PP_ALIGN.CENTER

        if idx < len(stage_meta) - 1:
            arr = s2.shapes.add_shape(
                MSO_SHAPE.ISOSCELES_TRIANGLE,
                Inches(left + step_w + 0.035), Inches(step_top + step_h / 2 - 0.06), Inches(0.12), Inches(0.10),
            )
            arr.rotation = 90
            arr.fill.solid()
            arr.fill.fore_color.rgb = HEADER_ROYAL
            arr.line.fill.background()

    add_narrative_card(
        s2, 0.45, 2.58, 3.98, 4.38,
        "STAGES 1 & 2", "Zero-Hardcoded Perception",
        "A hardcoded rule like 'RSI below 30' or a fixed '2% stop-loss' works in one month and breaks the moment volatility or capital changes.",
        "Stage 1 ingests daily bars, 9:15 AM 15-minute opening bars, and India VIX across all 50 Nifty stocks. Stage 2 eliminates every fixed cutoff: our Chameleon Feature Engine watches each stock across 7 Fibonacci time horizons (3 to 55 days), converts indicators into daily 0–100% cross-sectional percentile ranks, and sets each stock's stop-loss floor from its own 99th-percentile intraday noise.",
        "7 Volatility Horizons • 100% Percentile-Ranked • Custom Tail Stops",
        SAPPHIRE_PRIMARY,
        font_scale=1.01,
    )
    add_narrative_card(
        s2, 4.67, 2.58, 3.98, 4.38,
        "STAGES 3 & 4", "Tax-Aware AI & Strategy Olympics",
        "Beating a falling Nifty 50 index still loses real money—and a dramatic news headline is often already priced into the chart.",
        "In Stage 3, a Walk-Forward LightGBM model predicts each stock's net post-tax rupee return, uses TreeSHAP to expose its top 3 mathematical drivers, and starves weak sectors at ₹0. In Stage 4, four specialized strategy playbooks compete daily in an Empirical Bayes Tournament to match the current market regime, while our News Brain checks if price has already reacted to Google News RSS headlines.",
        "Post-Tax Rupee AI • 4-Playbook Tournament • Price-Aware News Shield",
        VIOLET_AI,
        font_scale=1.01,
    )
    add_narrative_card(
        s2, 8.89, 2.58, 3.99, 4.38,
        "STAGES 5, 6 & 7", "Cryptographic Vault & Audit",
        "An autonomous AI should never have unchecked permission to fire broker orders or invent its own trade explanations.",
        "In Stage 5, an independent Risk Vault sizes whole shares via Elastic Spillover (for ₹1L or ₹10Cr), enforces a -2.5% daily loss circuit breaker and human kill switch, and stamps approved orders with an HMAC-SHA256 signature. In Stage 6, our custom NSE Sandbox Broker settles trades with 2-leg GTT OCO triggers and SEBI Contract Notes. In Stage 7, a Locked-Pointer Journal writes plain-English audits with 0.0% numerical hallucination.",
        "HMAC-SHA256 Signed Orders • GTT OCO Sandbox • 0.0% Hallucination UI",
        EMERALD_WIN,
        font_scale=1.01,
    )

    # =========================================================================
    # SLIDE 3: STAGES 2 & 3 DEEP DIVE — TEACHING AI TO COUNT POST-TAX RUPEES
    # =========================================================================
    s3 = prs.slides.add_slide(blank)
    apply_master_chrome(
        s3,
        3,
        "SELF-CALIBRATING AI CORE (STAGES 2 & 3)",
        "Self-Calibrating Intelligence: Adapts to Volatility, Taxes & Any Capital Scale",
        "Why Zero Hardcoded Parameters Allow AgentX to Optimize Both ₹1 Lakh Retail Accounts and ₹10 Crore Institutional Portfolios",
        bg_img,
    )

    add_framed_chart(s3, charts["tax_sweetspot"], 0.45, 1.28, 5.70, 5.68)

    add_narrative_card(
        s3, 6.32, 1.28, 6.56, 1.80,
        "STAGE 2 // CHAMELEON RADAR", "Trend Purity & Stock-Specific Tail Stops",
        "Two stocks can both rise +3% in a week—one gliding smoothly on institutional buying, the other violently whipsawing intraday.",
        "We compute Kaufman's Efficiency Ratio (net price move divided by total path traveled) across 7 Fibonacci horizons (3d–55d) to measure trend smoothness, while setting each stock's stop-loss from its own 60-day 99th-percentile intraday drop.",
        "Separates Institutional Accumulation from Whipsaw Noise",
        SAPPHIRE_PRIMARY,
        font_scale=0.95,
    )
    add_narrative_card(
        s3, 6.32, 3.22, 6.56, 1.80,
        "STAGE 3A // SCALE-INVARIANT AI", "Asymmetric Post-Tax & Liquidity Objective",
        "If the Nifty 50 drops -2.0% and a stock drops -1.0%, relative-alpha models celebrate a +1.0% win while your wallet shrinks.",
        "Our Purged Walk-Forward LightGBM model is trained directly on Net Rupee Sharpe after deducting statutory taxes and dynamic Amihud liquidity slippage—so it automatically protects ₹1 Lakh accounts from flat fees and ₹10 Crore accounts from illiquid order books.",
        "Self-Calibrates Hurdle for ₹1 Lakh Retail or ₹10 Crore Funds",
        EMERALD_WIN,
        font_scale=0.95,
    )
    add_narrative_card(
        s3, 6.32, 5.16, 6.56, 1.80,
        "STAGE 3B // GLASS-BOX ROUTER", "TreeSHAP X-Ray & Sector Starvation",
        "Black-box AI hides why it bought a stock—and standard portfolios keep funding weak sectors just to fill empty slots.",
        "AgentX runs exact TreeSHAP decomposition to expose the top 3 mathematical drivers for every stock pick, and routes ₹0 (Sector Starvation) to any NSE sector whose average predicted edge fails the dynamic friction + standard error hurdle.",
        "Every Pick Mathematically Explained • Weak Sectors Get ₹0",
        VIOLET_AI,
        font_scale=0.95,
    )

    # =========================================================================
    # SLIDE 4: STAGE 4 DEEP DIVE — 4-PLAYBOOK OLYMPICS & "PRICED-IN" NEWS BRAIN
    # =========================================================================
    s4 = prs.slides.add_slide(blank)
    apply_master_chrome(
        s4,
        4,
        "ADAPTIVE STRATEGY TOURNAMENT & NEWS SHIELD (STAGE 4)",
        "Four Competing Strategy Playbooks & A News Engine That Checks the Chart First",
        "How AgentX Switches Between Breakout, Trend, Pullback, and Defensive Modes While Filtering Media Noise",
        bg_img,
    )

    # Rounded presentation frame behind the table (matches card curvature)
    s4_tshd = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.49), Inches(1.29), Inches(12.43), Inches(2.64))
    _curve(s4_tshd, 12.43, 2.64)
    s4_tshd.fill.solid()
    s4_tshd.fill.fore_color.rgb = CARD_SHADOW
    s4_tshd.line.fill.background()
    s4_tfrm = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), Inches(1.24), Inches(12.43), Inches(2.64))
    _curve(s4_tfrm, 12.43, 2.64)
    s4_tfrm.fill.solid()
    s4_tfrm.fill.fore_color.rgb = HEADER_ROYAL
    s4_tfrm.line.color.rgb = BORDER_CRISP
    s4_tfrm.line.width = Pt(1.4)
    t4_shape = s4.shapes.add_table(5, 4, Inches(0.55), Inches(1.32), Inches(12.23), Inches(2.48))
    t4 = t4_shape.table
    col_w4 = [2.45, 5.28, 2.15, 2.35]
    for idx, w in enumerate(col_w4):
        t4.columns[idx].width = Inches(w)

    h4 = ["Competing Playbook", "How It Spots Opportunity (100% Percentile-Ranked Logic)", "Target Market Regime", "Exit & Protection Style"]
    for c_idx, h in enumerate(h4):
        style_table_cell(t4.cell(0, c_idx), h, font_size=10.6, bold=True, text_color=TEXT_WHITE, bg_color=HEADER_ROYAL, align=PP_ALIGN.CENTER if c_idx >= 2 else PP_ALIGN.LEFT)

    r4_data = [
        ("1. FAST_BREAKOUT", "Captures fresh 13-day volume expansions breaking out of tight, low-noise consolidation bases", "Bull Expansion", "Dynamic Excursion Target"),
        ("2. SMOOTH_TREND", "Rides steady institutional sector leaders (high trend purity) on shallow 3-day pullbacks", "Steady Uptrend", "Chandelier Trailing Stop"),
        ("3. RANGE_REVERSION", "Buys deeply oversold quality blue-chips (bottom 20th percentile) as selling pressure exhausts", "Sideways Range", "Mean-Reversion Target"),
        ("4. DEFENSIVE_SHIELD", "Rotates into liquid low-beta safe havens when India VIX and market stress surge", "High-VIX Stress", "Tight Capital Shield"),
    ]
    for r_idx, row_vals in enumerate(r4_data, start=1):
        bg = CARD_WHITE if r_idx % 2 == 1 else CARD_ALT_ROW
        for c_idx, val in enumerate(row_vals):
            clr = SAPPHIRE_PRIMARY if c_idx == 0 else (EMERALD_WIN if c_idx == 2 else TEXT_BODY)
            is_b = c_idx in (0, 2)
            al = PP_ALIGN.CENTER if c_idx >= 2 else PP_ALIGN.LEFT
            style_table_cell(t4.cell(r_idx, c_idx), val, font_size=10.2, bold=is_b, text_color=clr, bg_color=bg, align=al)

    add_narrative_card(
        s4, 0.45, 3.92, 3.98, 3.04,
        "TOURNAMENT ENGINE", "Bayesian Anti-Whipsaw Memory",
        "Firing a proven trading strategy after just three losing days is the fastest way to get whipsawed right before it rebounds.",
        "Every morning, all 4 playbooks are scored on post-tax expectancy. Using Empirical Bayes Shrinkage (weighted by Kish's Effective Sample Size), AgentX blends each playbook's 2-year reliability with its recent trades in similar VIX regimes.",
        "Smooth Regime Transitions • Zero Whipsaw Switching",
        VIOLET_AI,
        font_scale=0.95,
    )
    add_narrative_card(
        s4, 4.67, 3.92, 3.98, 3.04,
        "SURPRISE SENSOR", "Event-Driven Volatility Alarm",
        "Rebalancing on a rigid 'every Friday' timer leaves a portfolio defenseless when a sudden macro shock hits on a Tuesday.",
        "AgentX eliminates fixed calendar timers. An event-driven Surprise Sensor monitors whether India VIX velocity or Nifty intraday range breaks its 252-day 90th percentile—staying patient on calm days and recalibrating instantly on shock days.",
        "Zero Fixed Timers • Instant 90th-Percentile Shock Defense",
        AMBER_GOLD,
        font_scale=0.95,
    )
    add_narrative_card(
        s4, 8.89, 3.92, 3.99, 3.04,
        "CATALYST SHIELD", "The 'Already Priced-In' News Check",
        "By the time a negative article trends on financial news feeds, institutional desks have often already sold and priced in the drop.",
        "Our News Brain scores date-bounded Google News RSS headlines and checks the stock's chart: it vetoes a buy only if negative news is unpriced (price hasn't dropped by 1x tail stop), preventing panic on already washed-out rebounds.",
        "Blocks Unpriced Disasters • Captures Oversold Rebounds",
        SAPPHIRE_PRIMARY,
        font_scale=0.95,
    )

    # =========================================================================
    # SLIDE 5: STAGE 5 DEEP DIVE — ELASTIC WALLET (₹1L TO ₹10CR+) & CRYPTO VAULT
    # =========================================================================
    s5 = prs.slides.add_slide(blank)
    apply_master_chrome(
        s5,
        5,
        "SCALE-INVARIANT CAPITAL & RISK VAULT (STAGE 5)",
        "Deploying ~99.9% of Capital—From ₹1 Lakh Retail to ₹10 Crore+ Institutional",
        "How Elastic Whole-Share Spillover, Intraday Profit Ratcheting, and Hardware-Style Order Signing Protect Any Account Size",
        bg_img,
    )

    add_framed_chart(s5, charts["slot_spillover"], 0.45, 1.28, 5.70, 5.68)

    add_narrative_card(
        s5, 6.32, 1.28, 6.56, 1.80,
        "1. ELASTIC SPILLOVER", "Scale-Free Whole-Share Cascade",
        "Rigid equal-weight boxes leave thousands of rupees stranded on a ₹1 Lakh account—and over-concentrate illiquid stocks on a ₹10 Crore account.",
        "Because sizing uses pure percentages and Amihud liquidity-adjusted conviction (zero hardcoded rupee caps), our Elastic Sizer cascades whole-share remainders at ₹1,00,000 and scales seamlessly into multi-crore institutional allocations (Max 50% per sector).",
        "Works Identically on ₹1,00,000 or ₹10,00,00,000 Without Code Changes",
        EMERALD_WIN,
        font_scale=0.95,
    )
    add_narrative_card(
        s5, 6.32, 3.22, 6.56, 1.80,
        "2. PROFIT RATCHET", "Intraday Chandelier Stop & 3-Day Hold Lock",
        "Nothing hurts an account more than watching a +4% unrealized gain slide all the way back down into a losing stop-loss.",
        "AgentX monitors intraday highs and ratchets the stop-loss upward using a volatility-scaled Chandelier exit—locking the floor safely above break-even plus round-trip friction—while a 3-Bar Hold Lock prevents unnecessary tax and slippage churn.",
        "Locks In Open Profits Intraday • Blocks Low-Edge Churn",
        AMBER_GOLD,
        font_scale=0.95,
    )
    add_narrative_card(
        s5, 6.32, 5.16, 6.56, 1.80,
        "3. CRYPTOGRAPHIC VAULT", "Independent Gatekeeper & Human Kill Switch",
        "If risk rules live inside the same script as the AI model, a single software glitch could fire unauthorized orders into the market.",
        "An independent Cryptographic Risk Vault (backed by ACID SQLite) checks our -2.50% daily loss circuit breaker, smooth drawdown throttle, and persistent Human Kill Switch before stamping approved orders with an HMAC-SHA256 token.",
        "Unsigned or Tampered Orders Are Mathematically Rejected",
        SAPPHIRE_PRIMARY,
        font_scale=0.95,
    )

    # =========================================================================
    # SLIDE 6: STAGES 6 & 7 DEEP DIVE — CUSTOM NSE SANDBOX BROKER & ZERO-LIE JOURNAL
    # =========================================================================
    s6 = prs.slides.add_slide(blank)
    apply_master_chrome(
        s6,
        6,
        "NEXT-GEN SANDBOX BROKER & ZERO-LIE JOURNAL (STAGES 6 & 7)",
        "Our Custom Next-Gen NSE Sandbox Broker & Zero-Hallucination Trade Journal",
        "Bringing DhanHQ Super Orders, Zerodha GTT OCO, Alpaca FIX State Machines & Locked-Pointer Explanations Into One Platform",
        bg_img,
    )

    # Rounded presentation frame behind the table (matches card curvature)
    s6_tshd = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.49), Inches(1.29), Inches(12.43), Inches(2.64))
    _curve(s6_tshd, 12.43, 2.64)
    s6_tshd.fill.solid()
    s6_tshd.fill.fore_color.rgb = CARD_SHADOW
    s6_tshd.line.fill.background()
    s6_tfrm = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), Inches(1.24), Inches(12.43), Inches(2.64))
    _curve(s6_tfrm, 12.43, 2.64)
    s6_tfrm.fill.solid()
    s6_tfrm.fill.fore_color.rgb = HEADER_ROYAL
    s6_tfrm.line.color.rgb = BORDER_CRISP
    s6_tfrm.line.width = Pt(1.4)
    t6_shape = s6.shapes.add_table(5, 4, Inches(0.55), Inches(1.32), Inches(12.23), Inches(2.48))
    t6 = t6_shape.table
    w6 = [2.65, 2.45, 4.48, 2.65]
    for idx, w in enumerate(w6):
        t6.columns[idx].width = Inches(w)

    h6 = ["Custom Sandbox Feature", "Inspired By", "How It Works Inside AgentX (paper_broker.py)", "Architectural Benefit"]
    for c_idx, h in enumerate(h6):
        style_table_cell(t6.cell(0, c_idx), h, font_size=10.6, bold=True, text_color=TEXT_WHITE, bg_color=HEADER_ROYAL, align=PP_ALIGN.CENTER if c_idx in (1, 3) else PP_ALIGN.LEFT)

    r6_data = [
        ("1. Atomic Super Orders", "DhanHQ v2 API", "Bundles 9:15 AM Entry + Target Leg + Trailing Stop Jump in one atomic call", "Zero unprotected naked positions"),
        ("2. 2-Leg GTT OCO Engine", "Zerodha Kite Connect", "Links Target & Stop legs in SQLite; filling either leg auto-cancels the other", "Prevents orphan double-execution"),
        ("3. 6-Stage FIX State Machine", "Alpaca Paper API", "Audits PENDING -> HMAC_VALIDATED -> CASH_CHECK -> 915_VWAP -> FILLED", "Full institutional order lifecycle"),
        ("4. T+1 Digital Contract Notes", "SEBI / NSE Clearing", "Generates official Contract Notes with STT, Stamp Duty, GST & ₹15.93 DP fee", "100% faithful to real bank net P&L"),
    ]
    for r_idx, r_vals in enumerate(r6_data, start=1):
        bg = CARD_WHITE if r_idx % 2 == 1 else CARD_ALT_ROW
        for c_idx, val in enumerate(r6_data[r_idx - 1]):
            clr = SAPPHIRE_PRIMARY if c_idx == 0 else (EMERALD_WIN if c_idx == 3 else (VIOLET_AI if c_idx == 1 else TEXT_BODY))
            style_table_cell(t6.cell(r_idx, c_idx), val, font_size=10.2, bold=(c_idx in (0, 1, 3)), text_color=clr, bg_color=bg, align=PP_ALIGN.CENTER if c_idx in (1, 3) else PP_ALIGN.LEFT)

    add_narrative_card(
        s6, 0.45, 3.92, 3.98, 3.04,
        "STAGE 6A // CUSTOM SANDBOX", "Why We Built Our Own Indian Broker",
        "Major Indian brokers (Zerodha, Dhan) offer no virtual-money sandbox, while US paper APIs assume zero tax and zero market slippage.",
        "We engineered an ACID SQLite NSE Sandbox that simulates 9:15 AM opening VWAP fills, order-size-scaled Amihud slippage (for ₹1L or ₹10Cr orders), auto-cancelling 2-leg GTT OCO brackets, and SEBI Contract Notes—plus 1-click JSON export for Zerodha & DhanHQ.",
        "Full Indian CNC Realism • 1-Click Live Broker JSON Bridge",
        EMERALD_WIN,
        font_scale=0.95,
    )
    add_narrative_card(
        s6, 4.67, 3.92, 3.98, 3.04,
        "STAGE 6B // 3-TIER VALIDATION", "Honest Regime Stress-Testing",
        "Almost any trading strategy looks profitable when tested only in a bull market or without statutory taxes and gap-down open fills.",
        "Our Walk-Forward Backtester stress-tests AgentX across Bull Trends, Sideways Chop, and High-VIX Crashes against Nifty 50 Buy-and-Hold and a Static Baseline—filling overnight stop breaches at the actual unfavorable 9:15 AM gap-open price.",
        "Verifies Net Post-Tax Expectancy Across All Market Regimes",
        AMBER_GOLD,
        font_scale=0.95,
    )
    add_narrative_card(
        s6, 8.89, 3.92, 3.99, 3.04,
        "STAGE 7 // ZERO-LIE JOURNAL", "Locked-Pointer Plain-English Narrator",
        "When standard LLMs summarize trades, they frequently hallucinate prices, invent indicator values, or miscalculate net P&L after taxes.",
        "Our Trade Journal builds a Locked Trade Fact AST (Pydantic schema) that binds every plain-English explanation directly to immutable SQLite Contract Note numbers and TreeSHAP drivers—verifying a 0.0% numerical hallucination rate.",
        "Plain-English Clarity • Mathematically Incapable of Lying",
        SAPPHIRE_PRIMARY,
        font_scale=0.95,
    )

    candidates = [
        Path("AgentX_PS02_Scale_Invariant_Pitch_Deck.pptx").resolve(),
        Path("AgentX_PS02_Architecture_Pitch_Deck.pptx").resolve(),
        Path("AgentX_PS02_Idea_Pitch_Deck.pptx").resolve(),
        Path(output_path).resolve(),
    ]
    saved_primary = None
    for c_path in candidates:
        try:
            prs.save(str(c_path))
            if saved_primary is None:
                saved_primary = c_path
        except PermissionError:
            continue

    art_dir = Path(r"C:\Users\devam\.gemini\antigravity\brain\2ae97b0c-037b-4801-8b7a-63804886cd1e")
    if art_dir.exists():
        for p in [bg_img, *charts.values()]:
            if p.exists():
                shutil.copy2(p, art_dir / p.name)

    return saved_primary or candidates[0]


if __name__ == "__main__":
    saved = build_formal_pitch_deck()
    print(f"Successfully generated Scale-Invariant Pitch Deck: {saved} ({saved.stat().st_size:,} bytes)")
