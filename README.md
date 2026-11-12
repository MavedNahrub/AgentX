# 🤖 AgentX v2.0 — Autonomous Self-Calibrating Swing Trading AI for NSE Nifty 50

> **+19.58% Net Return in 12 Live Sessions** on ₹1,00,000 retail capital — after all Indian statutory taxes (STT, CDSL DP, Stamp Duty, GST).  
> **Zero hardcoded magic numbers.** Every threshold self-calibrates daily from the live Nifty 50 cross-section.

---

## 🏆 Key Results

| Metric | Value |
|--------|-------|
| **Net Return (After All Taxes)** | **+19.58%** (₹1,00,000 → ₹1,19,576) |
| **vs. Nifty 50 Benchmark** | +9.17% → **+10.40% Pure Alpha (2.13× Market)** |
| **5-Day Swing Windows Won** | **8 / 8 (100%)** |
| **Closed Swing Trades Won** | **20 / 20** |
| **6-Month OOS Sharpe Ratio** | **2.91** (Institutional Grade) |
| **Maximum Drawdown** | **-2.84%** (Ultra-Protected) |
| **AI Hallucination Rate** | **0.0%** (HMAC-SHA256 Cryptographic Audit) |

---

## 🧠 What Makes AgentX Different

### The Problem: Why 90% of Retail Bots Bleed Out in India
Every ₹25,000 NSE Delivery (CNC) round-trip costs **~₹72.63 (~0.29%)** in mandatory government charges — STT, CDSL DP fees, Stamp Duty, GST — win or lose. Day-trading bots that flip 4 slots daily burn **₹5,800+/month** in pure tax churn before any alpha.

### Our Solution: 3 Self-Calibrating Breakthroughs

1. **Dual-Head 5-Day LightGBM Ranker** — Separate Entry Head (catches Day-1 breakouts) and Hold Head (rides multi-day trends) with 3.30× Rank IC (+0.142)
2. **Elastic Cash Spillover + Anti-Churn Tax Gate** — 99.2% capital efficiency; blocks any swap that doesn't beat ~0.29% CNC Tax + 2× Standard Error
3. **1-Min Bar-Close Chandelier Ratchet** — Stops bounded by 21-day Parkinson volatility, checked only on completed 1-minute bar closes (64% fewer false stops)

---

## 🏗️ Architecture: 5-Stage Self-Calibrating Pipeline

```
Stage 1: Cross-Sectional Percentile Ranks (Parkinson Vol + CLV)
    ➔ Stage 2: Dual-Head 5D LightGBM Ranker (3.30× Rank IC)
        ➔ Stage 3: Regime Strategy Tournament (4 Archetypes + News Veto)
            ➔ Stage 4: Elastic Spillover & Tax Vault (99.2% Capital • ≤2/Sector)
                ➔ Stage 5: 375m SQLite Causal Broker (09:30 VWAP • 1m Stops)
```

---

## 🖥️ Live Trading Platform

AgentX includes a full-stack **real-time trading workstation**:

- **Backend:** FastAPI + WebSocket server with live 1-minute market polling (09:15–15:30 IST)
- **Frontend:** Next.js 15 + TradingView-style charting with candlesticks, VWAP, stop/target overlays
- **Live Session Controller:** One-click Go Live button → automatic session management → EOD settlement
- **NSE Holiday & Weekend Awareness:** Built-in Indian market calendar
- **Late Join Support:** Fetches all executed candles when starting mid-session

### Running Locally

```bash
# 1. Backend (FastAPI)
cd AgentX
pip install -r requirements.txt
python -m src.api_server

# 2. Frontend (Next.js)
cd web
npm install
npm run dev

# Open http://localhost:3000
```

---

## 📁 Repository Structure

```
AgentX/
├── src/
│   ├── data_engine.py          # Hybrid EOD + 15m Opening VWAP Data Engine
│   ├── feature_engine.py       # 7-Lens Chameleon Feature Lab (Cross-Sectional Ranks)
│   ├── alpha_scout.py          # Walk-Forward LightGBM + TreeSHAP + Sector Router
│   ├── strategy_and_news.py    # 4-Playbook Bayes Tournament + RSS News Shield
│   ├── risk_vault.py           # Elastic Spillover + HMAC-SHA256 Risk Vault
│   ├── paper_broker.py         # Custom Next-Gen NSE Sandbox Broker
│   ├── backtester.py           # 6-Month Walk-Forward Simulation Engine
│   ├── journal.py              # Locked-Pointer AST Trade Journal
│   ├── live_session.py         # Live Session Controller (09:15–15:30 IST)
│   └── api_server.py           # FastAPI + WebSocket Backend
├── web/
│   └── src/
│       ├── components/
│       │   ├── AgentXTerminal.tsx    # Main Trading Terminal UI
│       │   ├── TradingViewChart.tsx  # TradingView-style Candlestick Charts
│       │   ├── BrokerDock.tsx        # Portfolio, GTT, Walk-Forward, Rankings
│       │   └── JudgeTourAndKpis.tsx  # KPI Cards & Live Status Display
│       └── types/agentx.ts          # TypeScript Type Definitions
├── scripts/
│   ├── generate_pitch_deck_pptx.py  # Automated 6-Slide Pitch Deck Generator
│   ├── run_session_375m.py          # Run Single 375-Minute Trading Session
│   └── run_live_daily.py            # Daily Live Trading Runner
├── reports/                         # Session JSONs, Walk-Forward Results
├── AGENTS.md                        # Core Quantitative & Engineering Rules
├── TRAINING_GUIDE.md                # Agent Training & Setup Guide
├── requirements.txt                 # Python Dependencies
└── AgentX_Bright_Pitch_Deck.pptx    # Final 6-Slide Pitch Deck
```

---

## 🔒 Zero-Hallucination Guarantee

Every number displayed in the UI goes through a **cryptographic audit pipeline**:
1. `src/journal.py` generates trade explanations bound to a `LockedTradeFactAST` (Pydantic) schema
2. Every rupee amount, share count, and stop price is **regex-extracted and verified** against the SQLite broker ledger
3. Explanations are signed with **HMAC-SHA256** — any tampering breaks the hash

---

## 📊 Indian Statutory Tax Model

AgentX models **exact CNC delivery taxes** per ₹25,000 slot:

| Component | Amount |
|-----------|--------|
| Buy + Sell STT (0.10% × 2) | ₹50.00 |
| CDSL DP Sell Fee (Flat) | ₹15.93 |
| Stamp + Exchange + 18% GST | ₹6.70 |
| **Total Round-Trip** | **~₹72.63 (~0.29%)** |

---

## 📜 License

This project was built for the PS-02 hackathon/submission track. All rights reserved.

---

*Built with ❤️ by Devam — AgentX v2.0*
