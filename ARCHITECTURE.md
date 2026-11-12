# PS-02: Autonomous Nifty 50 Equity Trading Agent — Stage 1 Architecture Blueprint

> **Stage 1 Status Note:** This document outlines the **Stage 1 architectural design** built across all 7 modular sections. In **Stage 2**, we will take this architecture into real market conditions and interactive UI verification to tune, stress-test, and optimize the agent for the best possible real-world performance.

## 1. Executive Summary & Core Philosophy
Retail traders often act on noise, overlook statutory Indian transaction fees, skip risk rules, and struggle to explain their own trades. Standard bots fail when they rely on rigid, hardcoded cutoffs (`20-day high`, `fixed 2% stop-loss`, `equal ₹25,000 slots`) or let raw LLMs make unverified financial decisions.

Our **Stage 1 Neuro-Symbolic Autonomous Agent (`AgentX`)** enforces a strict **"Propose and Dispose" Separation of Powers**:
- **Probabilistic AI (Alpha Scout — Stages 1–4)** senses multi-horizon momentum/reversion patterns, ranks opportunities on a post-tax basis, and extracts exact `TreeSHAP` mathematical attributions, but has **zero direct access** to fire broker orders.
- **Deterministic Math (Cryptographic Risk Vault — Stages 5–7)** holds the `HMAC-SHA256` order-signing key, enforces Indian statutory fee-efficiency rules across any capital scale (`₹1 Lakh` to `₹10 Crore+`), scales exposure via a smooth drawdown governor, and routes signed orders to our custom NSE Sandbox Broker.

---

## 2. End-to-End 7-Section Modular Pipeline (Stage 1 Build)

```mermaid
flowchart TD
    subgraph S1["Section 1: Hybrid Data & Market Telemetry Engine"]
        D1["Nifty 50 EOD + India VIX (yfinance / NSE)"]
        D2["9:15–9:30 AM 15m Intraday VWAP + OHLC Bridge"]
        D3["Full 50-Stock Eligibility & NSE Sector Map"]
        D4["Local Parquet / DuckDB Cache"]
        D1 --> D4
        D2 --> D4
        D3 --> D4
    end

    subgraph S2["Section 2: Chameleon Multi-Horizon Feature Lab"]
        F1["7 Fibonacci Horizons (3, 5, 8, 13, 21, 34, 55d Percentile Ranks)"]
        F2["Kaufman's Efficiency Ratio (ER_t) & Parkinson Intraday Vol"]
        F3["Asymmetric Post-Tax Net Rupee Sharpe Target"]
        D4 --> F1 & F2 --> F3
    end

    subgraph S3["Section 3: Walk-Forward AI Scout & SHAP Explainer"]
        A1["Purged Walk-Forward LightGBM Ranker (5-Day Embargo)"]
        A2["Cross-Sectional Expected Alpha Ranking & Sector Starvation (₹0)"]
        A3["Exact TreeExplainer Top-3 SHAP Feature Attributions"]
        F3 --> A1 --> A2 & A3
    end

    subgraph S4["Section 4: Event-Driven Strategy Tournament & News Shield"]
        T1["Surprise-Triggered Recalibration (90th-Pctl VIX / Range Shock)"]
        T2["Empirical Bayes Shrinkage Over Recent Regime Signals"]
        T3["4 Strategy Playbooks: FAST_BREAKOUT | SMOOTH_TREND | RANGE_REVERSION | DEFENSIVE_SHIELD"]
        N1["Date-Bounded Google News RSS -> Price-Aware Catalyst Check"]
        A2 --> T1 --> T2 --> T3 --> N1
    end

    subgraph S5["Section 5: Cryptographic Risk Vault & Elastic Sizer"]
        R1["Two-Pass Elastic Whole-Share Cash Spillover (₹1L to ₹10Cr+)"]
        R2["Smooth Drawdown Throttle + Daily Loss Circuit Breaker + SQLite Kill Switch"]
        R3["Hysteresis Smart Eviction (3-Day Hold Lock + Indian Tax Hurdle)"]
        R4["HMAC-SHA256 Digital Signature Generation"]
        N1 --> R1 --> R2 --> R3 --> R4
    end

    subgraph S6["Section 6: Custom Next-Gen NSE Sandbox Broker & Simulation"]
        B1["HMAC Signature Verification & 9:15–9:30 AM 15m VWAP Execution"]
        B2["2-Leg GTT OCO Triggers, 6-Stage FIX State Machine & SEBI Contract Notes"]
        B3["Walk-Forward Simulation Framework Across Market Regimes"]
        R4 --> B1 --> B2 --> B3
    end

    subgraph S7["Section 7: Locked-Pointer AST Journal & Web Terminal"]
        J1["LockedTradeFactAST Pydantic Schema (Plain-English Audit Narrator)"]
        J2["Next.js 15 + TradingView Financial Terminal (FastAPI + WebSocket)"]
        B3 & A3 --> J1 --> J2
    end
```

---

## 3. Core Stage 1 Architectural Mechanisms

1. **Hybrid `9:15–9:30 AM` 15-Minute Opening VWAP Engine (`src/data_engine.py` & `src/paper_broker.py`):**
   -Separates EOD signal generation (`3:30 PM IST`) from next-morning execution (`9:15–9:30 AM IST`). Uses 15-minute intraday candles to compute the opening VWAP ($\frac{H_{9:15} + L_{9:15} + C_{9:15}}{3}$) and bridges historical dates using a calibrated Parkinson Opening Range estimator.
2. **Scale-Invariant Elastic Whole-Share Cash Spillover & Sector Routing (`src/risk_vault.py` & `src/alpha_scout.py`):**
   - Keeps all **50 Nifty 50 stocks** eligible without hardcoded share-price bans.
   - Cascades uninvested whole-share rupee remainders from high-ticket slots into subsequent slots (`Max 2 slots` / $\le 50\%$ per NSE sector), while starving weak sectors at **₹0** when their average edge fails the dynamic friction hurdle.
3. **Zero-Hardcoded Chameleon Features & Asymmetric Post-Tax AI (`src/feature_engine.py` & `src/alpha_scout.py`):**
   - Evaluates each stock across 7 Fibonacci horizons (`3d–55d`) using daily `0–100%` cross-sectional percentile ranks and **Kaufman's Efficiency Ratio** (trend purity), while setting each stock's stop-loss floor from its own rolling 60-day 99th-percentile intraday noise.
   - Trains a **Purged Walk-Forward LightGBM** model on net post-tax return potential and uses **TreeSHAP** to expose the top 3 mathematical drivers for every candidate.
4. **4-Playbook Empirical Bayes Tournament & Price-Aware News Shield (`src/strategy_and_news.py`):**
   - Dynamically selects between `FAST_BREAKOUT`, `SMOOTH_TREND`, `RANGE_REVERSION`, and `DEFENSIVE_SHIELD` using **Empirical Bayes Shrinkage** and an event-driven **Surprise Sensor** (eliminating fixed calendar timers).
   - Screens Google News RSS headlines and checks whether the stock's price has already absorbed the move before vetoing an entry.
5. **Cryptographic Risk Vault, Next-Gen NSE Sandbox Broker & Locked-AST Journal (`src/risk_vault.py`, `src/paper_broker.py`, `src/journal.py`):**
   - Enforces a 3-Bar Hold Lock, anti-churn tax hurdle, and persistent SQLite **Human Kill Switch** before stamping approved orders with an **`HMAC-SHA256`** signature.
   - Settles orders inside an ACID SQLite **NSE Sandbox Broker** featuring **DhanHQ-style Atomic Super Orders**, **Zerodha-style 2-Leg GTT OCO triggers**, **Alpaca-style 6-Stage FIX State Machines**, and **SEBI T+1 Digital Contract Notes**, paired with a **Locked-Pointer Pydantic AST Journal**.
