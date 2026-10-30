# AgentX Quantitative Architecture & Engineering Rules

You must follow these core engineering and quantitative finance rules at all times when designing, coding, or modifying any module in this repository:

## 1. Zero Hardcoded Magic Numbers (Self-Calibrating Math Only)
- **No Static Indicator or Regime Cutoffs:** Never hardcode fixed indicator thresholds (e.g., `ER > 0.36`, `RVOL > 1.25`, `VIX > 1.5`, `-4.5% drop`, `20-day lookback`, or `5d/60d timers`).
- **Use Empirical & Cross-Sectional Scaling:**
  - Use **Daily Cross-Sectional Percentile Ranks (`rank(pct=True)`)** combined with directional/structural confirmation checks to select candidates.
  - Bound stop-losses and volatility triggers using each stock's own **Rolling Empirical Quantiles** (e.g., 21d Parkinson intraday range floor, 60d 95th/99th percentile tail excursion) rather than arbitrary percentage clips.
  - Set minimum alpha hurdles dynamically using **Exact Indian CNC Statutory Round-Trip Taxes (`~0.29%` on ₹25k slot = STT + ₹15.93 CDSL DP + Stamp + GST) + Cross-Sectional Standard Error**.
  - Learn strategy exit/target multipliers directly from the **Empirical Excursion Distribution** of recent completed signals.

## 2. Capital Efficiency & Diversification on ₹1,00,000 Retail Capital
- **100% Universe Eligibility (No High-Price Bans):** Never ban high-ticket Nifty 50 stocks (e.g., Maruti, Bajaj Auto, UltraTech) via hardcoded share-price caps. Use **Elastic Cash Spillover** so any uninvested whole-share rupee remainder from one slot automatically flows into the next slot.
- **Starve Weak Sectors at ₹0, Diversify Active Sectors:** Never lock slots to fixed sectors. Route ₹0 to sectors that fail the post-tax + standard-error alpha hurdle, and enforce a **Maximum 2 Slots ($\le 50\%$ capital) per NSE Sector** guardrail so capital is never 100% concentrated in a single sector.

## 3. Market-Condition Validation (Keep Only What Wins in Real Regimes)
- Before finalizing any mathematical change, verify how it behaves across real market conditions (Bull Trend, Sideways Chop, and High-VIX Crash).
- If a pure cross-sectional rank would force trades when the *entire market* is crashing or catch falling knives in `RANGE_REVERSION`, pair cross-sectional ranks with **Regime-Conditioned Structural Confirmation** and a **Post-Tax Expectancy Gate (> 0)** so only techniques that empirically improve post-tax expectancy and profit factor are kept.

## 4. Modular Section-by-Section Execution Protocol
For every pipeline section:
1. **Pre-Build Review:** Audit the section design before coding to eliminate hardcoded parameters and edge-case flaws.
2. **Standalone Verification Output:** Execute the module standalone and show the user the real data/telemetry output before connecting the next module.
3. **Post-Build Explanation:** Explicitly explain **"Why the naive/old approach was bad in real markets"** and **"How our new self-calibrating fix solves it."**
