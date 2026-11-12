# AgentX Master End-to-End Training & Calibration Guide

This document is the **single, authoritative, non-deviating training specification** for **AgentX**. Every training run must execute this exact sequence from start to finish without skipping steps, introducing hardcoded magic numbers, or leaking future information.

---

## Part 1: Exact Data Fetching Specification (What Must Be Fetched & How)

All raw market and catalyst data is ingested exclusively by **Module 1 (`src/data_engine.py`)** and **Module 4 (`src/strategy_and_news.py`)** before any mathematical transformation or model training begins.

### 1.1 Equity Universe & Macro Benchmarks (`src/data_engine.py`)
| Data Stream | Ticker / Source | Lookback & Granularity | Exact Columns Required | Purpose in Training |
| :--- | :--- | :--- | :--- | :--- |
| **1. Nifty 50 Equity Daily Tape** | 50 Official NSE Constituents (`RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `MARUTI.NS`, etc.) | **5 Years (~1,257 Trading Days = 62,850 Stock-Days)**, Daily (`1d`) | `date`, `symbol`, `sector`, `open`, `high`, `low`, `close`, `volume` | Core multi-horizon feature engineering (`60,100` active stock-days), rolling 60d tail risk, 5-year strategy tournament (`9,951` signals), and 180d out-of-sample walk-forward backtesting. |
| **2. Intraday Opening Auction Tape** | Same 50 NSE Constituents | **Trailing 55 Trading Days**, 15-Minute Candles (`15m`, `09:15–09:30 IST`) | `open_15m`, `high_15m`, `low_15m`, `close_15m`, `volume_15m` | Computes true institutional Opening VWAP: $\text{VWAP}_{9:15} = \frac{O_{15m} + H_{15m} + L_{15m} + C_{15m}}{4}$. |
| **3. Historical Opening VWAP Bridge (>55d)** | Derived from Daily `open`, `high`, `low`, `close` | Applied to days older than the 55-day 15m Yahoo window | `exec_vwap_915` | Bridges historical opening VWAP using the theoretical Brownian-Parkinson time-scaling law: $O_t + \text{sign}(C_t - O_t)(H_t - L_t)\sqrt{\frac{15}{375}}\sqrt{\frac{\pi}{4\ln 2}}$. |
| **4. Nifty 50 Market Benchmark** | `^NSEI` (Nifty 50 Index) | **5 Years (~1,257 Trading Days)**, Daily (`1d`) | `nifty_close`, `nifty_ret_1d`, `nifty_ret_21d` | Computes cross-sectional relative strength (`rel_strength_vs_nifty_21d`) and excess return vs. benchmark. |
| **5. India VIX Volatility Index** | `^INDIAVIX` | **5 Years (~1,257 Trading Days)**, Daily (`1d`) | `india_vix_level`, `india_vix_zscore_21d` | Drives the 126-day quantile Market-Breadth & VIX Regime Shield and triggers emergency walk-forward retraining when $\|Z_{\text{VIX}}\| > Q_{90}$. |
| **6. Dual-Gate Catalyst & News Feed** | Google News RSS (`https://news.google.com/rss/search?q={SYMBOL}+NSE+stock`) + Intraday Price-Volume Tape | **Trailing 72 Hours** (Live) + Daily RVOL & 1d Return (Historical) | `headline_title`, `pub_date`, `ret_1d`, `volume_shock_21d`, `empirical_tail_stop_pct` | Vetoes regulatory raids, earnings shocks, fraud probes, and unindexed institutional flash crashes (`1d drop <= -tail_noise_pct` on `RVOL >= 1.75`). |

### 1.2 Strict Data Hygiene & Anti-Leakage Policies
1. **Zero Backward Filling (`.bfill()` Strictly Prohibited):** Missing early rows in macro series (`^INDIAVIX`, `^NSEI`) or stock rolling windows must **never** use `.bfill()`. Only forward-filling (`.ffill()`) or strictly causal per-stock expanding quantiles (`.expanding(min_periods=2)`) are permitted.
2. **100% Universe Eligibility on ₹1,00,000 Capital:** High-ticket Nifty 50 stocks (`MARUTI`, `BAJAJ-AUTO`, `ULTRATECH`) must **never** be filtered out during data ingestion, provided at least 1 whole share (`close <= ₹1,00,000`) is purchasable.

---

## Part 2: Module-by-Module Input/Output Contract & Training Strategy

The training pipeline executes sequentially in **6 deterministic stages**. Each module consumes only the verified output of the preceding stage.

```
[Stage 1: Data Engine] ──> nifty50_hybrid_ohlcv.parquet (23,750 stock-days)
       │
[Stage 2: Feature Lab] ──> nifty50_chameleon_features.parquet (18 features + leak-free 5d target)
       │
[Stage 3: Alpha Scout] ──> Purged Walk-Forward LightGBM + OOS Validation IC + TreeSHAP Ranks
       │
[Stage 4: Strategy & News Tournament] ──> Learned Stop/Target Multipliers + Catalyst Veto
       │
[Stage 5: Risk Vault & Sandbox Broker] ──> 4-Slot Elastic Spillover + Sector Starvation + OCO Fills
       │
[Stage 6: 180d OOS Walk-Forward Lock & AST Journal] ──> Verified Performance & 0.0000% Hallucination
```

---

### Stage 1: Hybrid Data & Opening VWAP Ingestion (`src/data_engine.py`)
* **Input Required:** Raw NSE Yahoo Finance daily (`2y`) and 15-minute (`55d`) feeds for all 50 Nifty constituents, `^NSEI`, and `^INDIAVIX`.
* **Training Action:**
  1. Align all 50 stocks to the master `^NSEI` trading calendar (`f_fill` up to 3 days for isolated holidays).
  2. Compute exact `exec_vwap_915` for every stock-day.
  3. Verify `100%` universe eligibility (`50 / 50` stocks eligible on ₹1,00,000 via Elastic Cash Spillover).
* **Output Produced:** `data/cache/nifty50_hybrid_ohlcv.parquet`, `data/cache/macro_regime_ohlcv.parquet`, `data/cache/universe_1lakh_audit.parquet`.

---

### Stage 2: Self-Normalizing Multi-Horizon Feature & Label Engineering (`src/feature_engine.py`)
* **Input Required:** `nifty50_hybrid_ohlcv.parquet` and `macro_regime_ohlcv.parquet` from Stage 1.
* **Training Action (18 Features + 1 Target):**
  1. **7 Fibonacci Volatility-Normalized Momentum Lenses (`3, 5, 8, 13, 21, 34, 55` days):**
     $$Z_{\text{Fib}, w}(i, t) = \text{clip}\left(\frac{R_{w}(i, t)}{\sigma_{21}(i, t)\sqrt{w}}, -4.0, +4.0\right)$$
  2. **2 Kaufman Efficiency Ratios (`ER_8d`, `ER_21d`):** Measures path smoothness $\in [0, 1]$ to separate clean trends from noisy sideways chop.
  3. **3 Intraday & Overnight Volatility Estimators:** 21-day Parkinson High-Low Range Volatility (`parkinson_vol_21d`), 5d/21d Volatility Compression Coil (`vol_compression_5_21`), and Overnight Gap-Down Shock Ratio (`overnight_gap_shock`).
  4. **4 Microstructure & Relative Strength Lenses:** 13-day Bollinger Mean-Reversion Z-Score (`mean_rev_zscore_13d`), 21-day Median Relative Volume Shock (`volume_shock_21d`), Amihud Illiquidity Z-Score (`amihud_illiquidity_z`), and 21-day Relative Strength vs. Nifty 50 (`rel_strength_vs_nifty_21d`).
  5. **2 Macro Stress Features:** `india_vix_level` and `india_vix_zscore_21d` (with 95th-percentile stress threshold computed strictly along the unique calendar `date` index).
  6. **Cross-Sectional Micro-Regime Assignment (`micro_regime`):** Uses daily cross-sectional percentile ranks (`rank(pct=True)`) combined with directional Z-score confirmation (`SMOOTH_TREND`, `COILED_BREAKOUT`, `RANGE_REVERSION`, `DEFENSIVE_SHIELD`).
  7. **Zero-Look-Ahead 5-Day Forward Information Ratio Target (`target_5d_info_ratio`):**
     * Entry anchored at Day $t+1$ Opening VWAP (`exec_vwap_915.shift(-1)`).
     * Exit anchored at Day $t+6$ Opening VWAP (`exec_vwap_915.shift(-6)`).
     * Gross return clipped asymmetrically between each stock's causal 60-day 5th-percentile downside tail (`-stop_dist`) and 95th-percentile $\times \sqrt{5}$ upside tail (`+target_dist`).
     * Deducts exact statutory Indian CNC round-trip tax (`~0.29%`) + square-root order-size market impact (`2 * parkinson_vol_21d * sqrt(slot_notional / adv_rupees_21d)`).
     * Blends post-tax absolute return and excess return vs. Nifty 50 using `ER_21d` as the natural $[0, 1]$ weight, divided by 5-day realized volatility ($\sigma_{21}\sqrt{5}$).
* **Mandatory Training Gate:** `0` NaNs across all 18 features after the 55-day warmup; zero `.bfill()` calls across stock boundaries.
* **Output Produced:** `data/cache/nifty50_chameleon_features.parquet`.

---

### Stage 3: Purged Walk-Forward LightGBM & Out-of-Sample IC Calibration (`src/alpha_scout.py`)
* **Input Required:** `nifty50_chameleon_features.parquet` from Stage 2.
* **Training Action:**
  1. **Strict 6-Trading-Day Embargo Purge:** When training at date $T$, discard the most recent 6 trading days ($t > T - 6$) from the training set because their 5-day forward labels (`t+1 -> t+6`) have not settled yet at date $T$.
  2. **126-Day Exponential Half-Life Sample Weighting:** Weight training observations by $w_t = \exp\left(-\frac{\ln 2}{126} \cdot \Delta \text{days}_t\right)$ so recent market regimes dominate without discarding structural crisis memory.
  3. **Purged Out-of-Sample Validation IC Blending (Anti-Overfitting Policy):**
     * Split the purged training window into a sub-train fold and a held-out 15-trading-day validation fold (separated by its own 6-day embargo gap).
     * Fit `val_model` on the sub-train fold and compute out-of-sample Spearman Rank IC (`val_ic`) on the held-out 15-day validation fold.
     * Set `ml_weight = clip(val_ic, 0.08, 0.25)` and `struct_weight = 1.0 - ml_weight` before refitting the final `LGBMRegressor` (`max_depth=4`, `num_leaves=12`, `min_child_samples=45`, `reg_alpha=0.8`, `reg_lambda=2.0`) on the full purged window.
  4. **Precision-Weighted 4-Factor Structural Rank:** Combine the 4 orthogonal cross-sectional percentile ranks (`1 - r_fib_mom_z_55d`, `1 - r_mean_rev_zscore_13d`, `r_efficiency_ratio_21d`, `r_parkinson_vol_21d`) weighted by their square-root Fibonacci horizon ratios ($\sqrt{55}, \sqrt{34}, \sqrt{21}, \sqrt{13}$).
  5. **Exact TreeSHAP Attribution:** Extract the 18-dimensional SHAP vector for all 50 stocks on the target date to identify the top 3 positive drivers and top negative drag.
* **Mandatory Training Gate:** Out-of-Sample Validation Rank IC must be positive (`> +0.10`), and only stocks clearing `dynamic_alpha_hurdle_pct` (`Statutory Tax + Amihud Impact + Cross-Sectional SE`) are marked `passes_tax_hurdle = True`.
* **Output Produced:** `data/cache/alpha_scout_latest_ranked.parquet`, `data/cache/alpha_scout_shap_matrix.parquet`.

---

### Stage 4: Post-Tax Strategy Tournament & Learned Excursion Calibration (`src/strategy_and_news.py`)
* **Input Required:** `nifty50_chameleon_features.parquet` and `alpha_scout_latest_ranked.parquet`.
* **Training Action:**
  1. **252-Day Walk-Forward Strategy Tournament:** Simulate all 4 strategies (`SMOOTH_TREND`, `COILED_BREAKOUT`, `RANGE_REVERSION`, `DEFENSIVE_SHIELD`) over the trailing 252 trading days with structural entry confirmation and full Indian CNC tax + Amihud slippage deduction.
  2. **Self-Calibrating Stop & Target Multipliers:** Measure the empirical distribution of realized 5-day excursions relative to `empirical_tail_stop_pct` across completed tournament signals:
     * `learned_stop_mult` = 55th percentile of downside excursion ratios (`~0.55x–0.65x` tail stop).
     * `learned_target_mult` = 93.5th percentile of upside excursion ratios (`~1.95x` tail stop, maintaining a ~3:1 asymmetric reward-to-risk ratio).
  3. **Dual-Gate Catalyst & Silent Crash Veto:** Filter all top-ranked candidates through live Google News RSS sentiment + the deterministic price-volume shock detector (`SILENT_PRICE_VOLUME_CRASH_SHOCK`).
* **Mandatory Training Gate:** Every active strategy must exhibit **Post-Tax Expectancy > `0.00%`** and **Profit Factor > `1.20`** over the 252-day tournament.
* **Output Produced:** `data/cache/strategy_tournament_scorecard.parquet`, `data/cache/strategy_news_paired_candidates.parquet`.

---

### Stage 5: ₹1,00,000 Dynamic Sector Router, 4-Slot Elastic Spillover & Sandbox Execution (`src/risk_vault.py` & `src/paper_broker.py`)
* **Input Required:** `strategy_news_paired_candidates.parquet` and `nifty50_chameleon_features.parquet`.
* **Training Action:**
  1. **Dynamic Sector Starvation at ₹0:** Compute each NSE sector's top candidate alpha against `Statutory Round-Trip Tax (~0.29%) + Cross-Sectional Standard Error (~0.11%)`. Any sector below this hurdle receives **₹0** allocation.
  2. **Max 2 Slots ($\le 50\%$) Per Sector Guardrail:** Cap any active sector at a maximum of 2 slots out of 4.
  3. **Two-Pass Conviction-to-Tail-Risk Sizer with Elastic Cash Spillover:**
     * Weight slots by $\frac{\text{Conviction}_i}{\text{TailStop}_i}$.
     * Sort selected candidates from highest share price to lowest so high-ticket stocks (`MARUTI`, `APOLLOHOSP`, `LT`) are sized first, and every rupee of uninvested whole-share remainder cascades into the next slot.
     * Sweep any final leftover cash into affordable slots in Pass 2 to achieve **98%–99.5% capital efficiency** with zero fractional shares.
  4. **3-Bar Hysteresis & Smart Eviction Gate:** Block portfolio rotation if `bars_held < 3` or if the challenger's alpha gain is $\le \text{Round-Trip Switch Tax} + \text{Alpha SE}$.
  5. **Cryptographic Signing & Multi-Session Sandbox Verification:** Stamp every order with `HMAC-SHA256`, execute at `9:15–9:30 AM` Opening VWAP in `IndiaPaperBroker`, arm Two-Leg GTT OCO triggers, verify Chandelier stop ratcheting above breakeven, and generate `T+1 Business Day` SEBI Digital Contract Notes.
* **Mandatory Training Gate:** Capital deployment $\ge 95\%$ when 4 slots are active; 0 sector cap violations; 100% tampered HMAC rejection.
* **Output Produced:** `data/cache/risk_vault_4slot_allocations.parquet`, `data/cache/paper_broker_sandbox.db`.

---

### Stage 6: 180-Day Out-of-Sample Walk-Forward Benchmark & Zero-Hallucination Audit (`src/backtester.py` & `src/journal.py`)
* **Input Required:** All artifacts from Stages 1–5.
* **Training Action:**
  1. **180-Day 3-Tier Walk-Forward Backtest:** Run `WalkForwardBacktester` across the 180-trading-day out-of-sample window with:
     * Rolling 126-day quantile Market-Breadth & VIX Regime Shield + Standard-Error Equilibrium Gate (`mkt_mrev <= -sqrt(pi/2) * mkt_mrev_se`).
     * Retraining every 15 trading days (or immediately when $\|Z_{\text{VIX}}\| > Q_{90}$) with a 6-day purge embargo.
     * Exact Indian CNC taxes + $\sqrt{2/15}$ 15-minute VWAP time-sliced Amihud slippage on every entry and exit.
  2. **Locked-Pointer AST Journal Verification:** Synthesize plain-English SHAP trade narratives in `src/journal.py` and run `verify_zero_hallucination()` to confirm every number in the text matches its locked tensor pointer (`0.0000%` hallucination rate).
* **Mandatory Final Selection Gate:**
  * **Tier 3 (AgentX v2.0)** must beat **Tier 1 (Nifty 50 Buy-and-Hold)** and **Tier 2 (Naive Fixed-Rule Baseline)** on Net Return, Sharpe Ratio, Max Drawdown, and Profit Factor.
  * **Hallucination Rate** must equal **`0.0000%`**.

---

## Part 3: Single-Command Non-Deviating Execution Sequence

To execute this entire training and calibration guide end-to-end on historical data in ~60 seconds without waiting for market hours:

```powershell
.\.venv\Scripts\python.exe -m src.data_engine; `
.\.venv\Scripts\python.exe -m src.feature_engine; `
.\.venv\Scripts\python.exe -m src.alpha_scout; `
.\.venv\Scripts\python.exe -m src.strategy_and_news; `
.\.venv\Scripts\python.exe -m src.risk_vault; `
.\.venv\Scripts\python.exe -m src.paper_broker; `
.\.venv\Scripts\python.exe -m src.backtester; `
.\.venv\Scripts\python.exe -m src.journal
```
