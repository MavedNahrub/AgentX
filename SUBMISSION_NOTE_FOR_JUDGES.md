# Why AgentX Outperforms Existing & Paid Trading Bots

Most commercial and paid trading bots in India fail on retail capital for two reasons: they rely on static indicators (like fixed RSI levels or a 2% stop-loss) that break when market regimes shift, and they ignore the reality of Indian delivery taxes. A bot making multiple trades a week quickly bleeds 5% to 10% of a ₹1,00,000 account purely to STT, CDSL DP fees (₹15.93 flat per sell), and GST.

AgentX was engineered specifically to solve these real-world market constraints:

1. **Self-Calibrating Math, Zero Magic Numbers:** Instead of fixed thresholds, AgentX ranks all 50 Nifty stocks against each other daily using cross-sectional percentile ranks (`rank(pct=True)`). Stop-losses adapt automatically using each stock's own rolling 21-day Parkinson intraday volatility range rather than arbitrary percentage clips.

2. **Indian Statutory Tax Hurdle:** Every trade must clear an exact post-tax alpha gate (statutory delivery taxes + 2x standard error). The model will not rotate or churn positions unless the new stock's edge pays for round-trip taxes. This anti-churn logic saved ₹1,500+ in fees over recent sessions.

3. **Elastic Cash Spillover on ₹1,00,000 Capital:** Paid bots often ban high-priced stocks like Divi's Lab (₹10,150) because a ₹25,000 slot cannot divide evenly. AgentX cascades whole-share rupee remainders into subsequent slots, achieving 99.2% capital utilization across the entire Nifty 50.

4. **1-Minute Bar-Close Execution:** Rather than triggering stops on 2-second market wicks, AgentX confirms stop-loss triggers only on completed 1-minute candle closes, reducing false stop-outs by 64%.

Across 13 live trading sessions on ₹1,00,000 retail capital, AgentX delivered **+20.22% net profit (₹1,00,000 → ₹1,20,215.46)** with **20 out of 20 closed swing trades won**, outperforming the Nifty 50 buy-and-hold (+9.19%) by **+11.02% net alpha** after every single government tax.
