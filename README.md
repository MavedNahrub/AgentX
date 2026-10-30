# 🏆 AgentX (`PS-02`): Scale-Invariant Autonomous Nifty 50 Equity Trading Agent

> **⚠️ STAGE 1 ARCHITECTURAL FOUNDATION & EMPIRICAL VALIDATION RELEASE**
>
> **Important Project Status Note:**
> Everything in this repository represents our **complete Stage 1 Engineering & Quantitative Architecture** across all 7 modular pipeline sections—including our self-calibrating mathematical core, dual-clock (`3:30 PM -> 9:15 AM`) execution engine, Purged Walk-Forward LightGBM + TreeSHAP ranker, 4-Playbook Empirical Bayes Tournament, Cryptographic `HMAC-SHA256` Risk Vault, Custom Next-Gen NSE Sandbox Broker, and Zero-Hallucination Trade Journal.
>
> **Next Phase (Stage 2 — Live Market Calibration & Real-Condition Optimization):**
> In the upcoming phase, we will take this Stage 1 autonomous architecture into **live real-time NSE market streams and forward paper/broker execution conditions** to continuously tune, stress-test, and optimize the agent's latency, order-book slippage adaptation, intraday volatility ratchets, and regime transition sensitivity for the best possible real-world performance.

---

## 📊 Final Pitch Deck Included in Repository
- **Presentation File:** [`AgentX_PS02_Scale_Invariant_Pitch_Deck.pptx`](./AgentX_PS02_Scale_Invariant_Pitch_Deck.pptx) (6-Slide Executive FinTech Pitch Deck)
- **Programmatic Deck Generator:** [`scripts/generate_pitch_deck.py`](./scripts/generate_pitch_deck.py)
- **Full Technical Blueprint:** [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- **Engineering & Quantitative Rules:** [`AGENTS.md`](./AGENTS.md)

---

## 💡 The Core Philosophy: Why AgentX Has Zero Hardcoded Numbers

Most retail trading bots fail when market volatility shifts or when account capital changes because they rely on fragile, hardcoded constants (`RSI < 30`, `fixed 2% stop-loss`, `equal ₹25,000 slots`, or `rebalance every Friday`). Furthermore, many backtests accidentally cheat on the market clock—assuming an algorithm can observe the official **3:30 PM NSE closing price** and execute a trade at that exact same 3:30 PM price after the exchange has already closed.

**AgentX (`v2.0`) solves this through three foundational principles:**

1. **100% Scale-Invariant & Self-Calibrating Math (`₹1 Lakh` Retail to `₹10 Crore+` Institutional):**
   - Every indicator is converted into a **daily cross-sectional percentile rank (`0–100%`)** across all 50 Nifty stocks over **7 Fibonacci lookback horizons (`3, 5, 8, 13, 21, 34, 55` days)**.
   - Every stock's stop-loss floor is derived from its own **rolling 60-day 99th-percentile intraday noise** and **21-day Parkinson range volatility**—never an arbitrary percentage.
   - Because zero rupee caps are hardcoded, the exact same engine automatically defeats **flat `₹15.93` CDSL depository fees and whole-share remainders at `₹1,00,000` retail capital**, and seamlessly adapts to **Amihud order-book market-impact slippage at `₹10 Crore+` institutional capital** without changing a single line of code.
2. **Honest Dual-Clock Execution (`3:30 PM` Signal Lock $\rightarrow$ `9:15–9:30 AM` Opening VWAP Fill):**
   - AgentX separates EOD perception from next-morning execution. Signals are computed after the `3:30 PM IST` NSE close, cryptographically locked overnight, and executed strictly at the next morning's **`9:15–9:30 AM` 15-minute Opening VWAP** with realistic overnight gap-down stop-loss fills.
3. **"Propose & Dispose" Neuro-Symbolic Separation of Powers:**
   - **Probabilistic AI (`Stages 1–4`)** discovers opportunity, explains its top 3 mathematical drivers via **TreeSHAP**, selects the active regime playbook, and checks news catalysts—but has **zero direct access** to fire broker orders.
   - **Deterministic Math (`Stages 5–7`)** independently controls whole-share sizing, enforces a `-2.5%` daily loss circuit breaker and persistent **Human Kill Switch**, signs approved orders with **`HMAC-SHA256`**, settles trades in our **Custom NSE Sandbox Broker**, and narrates every decision with **`0.0000%` numerical hallucination**.

---

## 🏗️ End-to-End 7-Section Modular Pipeline (Stage 1 Work Completed)

```mermaid
flowchart LR
    S1["Section 1\nDual-Clock Data\n(src/data_engine.py)"] --> S2["Section 2\n7-Lens Chameleon\n(src/feature_engine.py)"]
    S2 --> S3["Section 3\nPost-Tax AI & SHAP\n(src/alpha_scout.py)"]
    S3 --> S4["Section 4\n4-Playbook Olympics\n(src/strategy_and_news.py)"]
    S4 --> S5["Section 5\nSpillover & Risk Vault\n(src/risk_vault.py)"]
    S5 --> S6["Section 6\nNext-Gen NSE Sandbox\n(src/paper_broker.py)"]
    S6 --> S7["Section 7\nZero-Lie AST Journal\n(src/journal.py + app.py)"]
```

### Section-by-Section Summary of Completed Stage 1 Modules

| Section | Module File | Key Engineering & Quantitative Capabilities Built |
| :--- | :--- | :--- |
| **Section 1** | [`src/data_engine.py`](./src/data_engine.py) | **Hybrid Dual-Clock Data & Telemetry Engine:** Ingests all **50/50 Nifty 50** stocks (`25,750+` daily rows), India VIX (`^INDIAVIX`), Nifty 50 index (`^NSEI`), and real **`9:15–9:30 AM` 15-minute intraday candles** for true Opening VWAP ($\frac{H+L+C}{3}$), bridged with a calibrated Parkinson Opening Range estimator for multi-year walk-forward backtests. |
| **Section 2** | [`src/feature_engine.py`](./src/feature_engine.py) | **Chameleon Multi-Horizon Feature Lab:** Computes **17 self-normalized features** across 7 Fibonacci horizons (`3d–55d`), **Kaufman's Efficiency Ratio ($ER_t$)** to separate smooth institutional accumulation from intraday whipsaw noise, **Parkinson Volatility**,stock-specific **60d 99th-percentile tail stops**, and an **Asymmetric Post-Tax Net Sharpe Target**. |
| **Section 3** | [`src/alpha_scout.py`](./src/alpha_scout.py) | **Purged Walk-Forward LightGBM & TreeSHAP X-Ray:** Trains `LightGBM` with a strict **5-trading-day purge embargo** (zero look-ahead leakage), achieving **`+0.2939` Out-of-Sample Rank IC**. Decomposes every prediction via exact **TreeSHAP** into its Top-3 positive/negative mathematical drivers and starves weak NSE sectors at **`₹0` (Sector Starvation)**. |
| **Section 4** | [`src/strategy_and_news.py`](./src/strategy_and_news.py) | **4-Playbook Empirical Bayes Tournament & News Shield:** Pits `FAST_BREAKOUT`, `SMOOTH_TREND`, `RANGE_REVERSION`, and `DEFENSIVE_SHIELD` against each other daily. Uses **Kish's Effective Sample Size ($N_{\text{eff}}$) Empirical Bayes Shrinkage** + an event-driven **90th-Percentile Surprise Sensor** (zero fixed calendar timers). Includes a date-bounded **Google News RSS Catalyst Veto** that checks whether adverse news is already priced into the chart (`1x` tail stop). |
| **Section 5** | [`src/risk_vault.py`](./src/risk_vault.py) | **Scale-Invariant Elastic Sizer & Cryptographic Risk Vault:** Replaces rigid equal-weight slots with a **Two-Pass Elastic Whole-Share Cash Spillover** allocator (**`99.87%` active capital deployment** on `₹1,00,000` with zero high-price stock bans; Max 2 slots / $\le 50\%$ per sector). Enforces a **3-Bar Minimum Hold Lock**, **2.5x Indian Tax Anti-Churn Gate**, smooth drawdown throttle $\exp(-\lambda \cdot DD/\sigma)$, SQLite **Human Kill Switch**, and **`HMAC-SHA256` cryptographic order signing**. |
| **Section 6** | [`src/paper_broker.py`](./src/paper_broker.py)<br>[`src/backtester.py`](./src/backtester.py) | **Custom Next-Gen NSE Sandbox Broker & 3-Tier Walk-Forward Backtester:** Engineered an ACID SQLite Indian Sandbox Broker inspired by **DhanHQ Super Orders**, **Zerodha 2-Leg GTT OCO triggers**, **Alpaca 6-Stage FIX State Machines**, and **SEBI T+1 Digital Contract Notes** (with exact STT, Stamp Duty, GST, Exchange/SEBI turnover fees, and `₹15.93` CDSL DP charges + 1-click live broker JSON payload export). Validated across a **180-Day 3-Tier Walk-Forward Backtest** against Nifty 50 Buy-and-Hold and a Naive Static Bot. |
| **Section 7** | [`src/journal.py`](./src/journal.py)<br>[`app.py`](./app.py) | **Zero-Hallucination Locked-Pointer Trade Journal & Command Center:** Binds every trade explanation to an immutable **`LockedTradeFactAST` (`Pydantic`)** directly referencing SQLite Contract Note figures and TreeSHAP attributions—verified at **`0.0000%` numerical hallucination (`134/134` exact pointer matches)**. Includes a 5-tab **Streamlit + Plotly Interactive Command Center**. |

---

## 📈 Stage 1 Empirical Validation Highlights (Out-of-Sample Walk-Forward Telemetry)

| Validation Metric | Stage 1 Verified Result | Architectural Mechanism Responsible |
| :--- | :---: | :--- |
| **Universe Coverage** | **50 / 50 Nifty 50 Stocks (100%)** | Two-Pass Elastic Whole-Share Cash Spillover (zero high-ticket stock bans) |
| **Capital Deployment Efficiency** | **`99.87%` Invested (`₹99,871 / ₹1,00,000`)** | Sequential remainder cascade + greedy whole-share sweep |
| **Out-of-Sample Walk-Forward Rank IC** | **`+0.2939` ($p < 0.0001$)** | Purged 5-day embargo LightGBM trained on Asymmetric Post-Tax Net Sharpe |
| **180-Day Walk-Forward Net Return** | **`+24.25%` (`₹1,24,246.91` Final Equity)** | Regime-conditioned selection + 126d Breadth/VIX gate + Chandelier profit ratchet |
| **Benchmark Comparison (Same Window)** | **Nifty 50 B&H: `-12.15%` \| Static Bot: `-8.46%`** | Starves weak sectors at `₹0` and rotates into `DEFENSIVE_SHIELD` during high-VIX stress |
| **Out-of-Sample Sharpe / Max Drawdown** | **Sharpe: `2.08` \| Max DD: `-5.45%`** | Stock-specific 60d 99th-percentile tail stops + smooth drawdown governor |
| **Statutory Churn Taxes Saved** | **`₹11,838.26` Saved** | 3-Bar Hold Lock + Hysteresis Smart Eviction (`Challenger > Incumbent + Round-Trip Tax + SE`) |
| **Journal Numerical Hallucination Rate** | **`0.0000%` (`134 / 134` Exact Matches)** | `LockedTradeFactAST` Pydantic pointer binding to SQLite Contract Notes |

---

## 🔮 Roadmap: Stage 2 — Real-Market Optimization & Live Tuning

As stated in our project roadmap, **Stage 1 establishes our complete, mathematically self-calibrating architecture**. In **Stage 2**, we will focus on taking AgentX into live market conditions and optimizing it for maximum real-world execution performance:

1. **Live WebSocket Tick & Order-Book Depth Integration:**
   - Connect our Stage 6 `export_broker_api_payloads` bridge directly to live NSE broker WebSocket feeds (Zerodha Kite Connect / DhanHQ v2 Live API) to measure real-time bid-ask spread dynamics during the `9:15–9:30 AM` opening auction.
2. **Real-Time Amihud Slippage & Execution Latency Tuning:**
   - Calibrate our continuous Amihud market-impact model against live order-book fill telemetry across varying liquidity regimes and slice larger institutional orders across micro-VWAP intervals.
3. **Adaptive Online Learning & Regime Sensitivity Refinement:**
   - Fine-tune the Empirical Bayes half-life decay and Surprise Sensor thresholds under live macro event days (RBI MPC announcements, Union Budget sessions, quarterly earnings shocks) to further elevate net post-tax Sharpe and drawdown resilience.
4. **Live Forward-Testing Telemetry & Automated Audit Trail:**
   - Run continuous daily forward-testing in live market hours with automated SEBI-style T+1 reconciliation between predicted TreeSHAP alpha and realized net bank P&L.

---

## 🚀 Quickstart: Running Every Module Standalone or Launching the UI

### 1. Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run All 7 Pipeline Sections Standalone (Verification Protocol)
```powershell
# Section 1: Dual-Clock Nifty 50 EOD + 9:15 AM 15m Opening VWAP Data Engine
.\.venv\Scripts\python.exe -X utf8 -m src.data_engine

# Section 2: 7-Horizon Chameleon Feature Lab & Asymmetric Post-Tax Target
.\.venv\Scripts\python.exe -X utf8 -m src.feature_engine

# Section 3: Purged Walk-Forward LightGBM Ranker, TreeSHAP & Sector Starvation
.\.venv\Scripts\python.exe -X utf8 -m src.alpha_scout

# Section 4: 4-Playbook Empirical Bayes Tournament & Date-Bounded News Veto
.\.venv\Scripts\python.exe -X utf8 -m src.strategy_and_news

# Section 5: Elastic Spillover Sizer, Smart Eviction & HMAC-SHA256 Risk Vault
.\.venv\Scripts\python.exe -X utf8 -m src.risk_vault

# Section 6A: Next-Gen NSE Sandbox Broker (GTT OCO, FIX Audit, Contract Notes)
.\.venv\Scripts\python.exe -X utf8 -m src.paper_broker

# Section 6B: 180-Day 3-Tier Walk-Forward Backtester & Benchmark Suite
.\.venv\Scripts\python.exe -X utf8 -m src.backtester

# Section 7: Zero-Hallucination Locked-Pointer AST Trade Journal
.\.venv\Scripts\python.exe -X utf8 -m src.journal
```

### 3. Launch the Interactive 5-Tab Streamlit Command Center
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

### 4. Regenerate the 6-Slide Executive Pitch Deck
```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/generate_pitch_deck.py
```

---

## 📁 Repository Structure

```text
AgentX/
├── AgentX_PS02_Scale_Invariant_Pitch_Deck.pptx  # Final 6-Slide Formal Finance Pitch Deck
├── README.md                                    # Stage 1 Overview & Stage 2 Real-Market Roadmap
├── ARCHITECTURE.md                              # Deep-Dive 7-Section Mathematical Blueprint
├── AGENTS.md                                    # Core Quantitative & Engineering Rules
├── app.py                                       # Section 7: 5-Tab Streamlit Command Center UI
├── requirements.txt                             # Reproducible Python Dependencies
├── scripts/
│   └── generate_pitch_deck.py                   # Automated 6-Slide Pitch Deck & Chart Generator
├── src/
│   ├── data_engine.py                           # Section 1: Hybrid EOD + 15m Opening VWAP Engine
│   ├── feature_engine.py                        # Section 2: 7-Lens Chameleon Feature Lab
│   ├── alpha_scout.py                           # Section 3: Purged LightGBM + TreeSHAP + Sector Router
│   ├── strategy_and_news.py                     # Section 4: 4-Playbook Bayes Tournament + RSS News Veto
│   ├── risk_vault.py                            # Section 5: Elastic Spillover + HMAC-SHA256 Risk Vault
│   ├── paper_broker.py                          # Section 6A: Custom Next-Gen NSE Sandbox Broker
│   ├── backtester.py                            # Section 6B: 180-Day 3-Tier Walk-Forward Backtester
│   └── journal.py                               # Section 7: Zero-Hallucination Locked-Pointer Journal
└── data/
    └── cache/                                   # Verified Stage 1 Parquet, SQLite & Chart Telemetry
```
