# Note for the Judges: AgentX v2.0

**GitHub:** https://github.com/MavedNahrub/AgentX.git  
**Universe & Capital:** NSE Nifty 50 | ₹1,00,000 Retail Base  
**Results:** +19.58% Net Return (₹1,00,000 to ₹1,19,576 over 12 live sessions) | +10.40% Alpha over Nifty 50 | 20 / 20 Closed Trades Won | 2.91 Sharpe Ratio | 0.0% Hallucinations (HMAC-SHA256 verified)

---

### What is AgentX?
AgentX is an automated swing trading system built specifically for the realities of Indian retail trading on the NSE Nifty 50. 

Most retail bots lose money because they use fixed indicators (like RSI under 30 or static 2% stops) that break when market conditions change, and they ignore Indian delivery taxes. 

AgentX solves this by making everything self-calibrating. Every morning, it ranks all 50 Nifty stocks against each other from 0% to 100% using cross-sectional percentile ranks. It predicts 5-day trends with a walk-forward LightGBM model, and sets stop-loss floors dynamically using each stock's own 21-day intraday volatility. All results are calculated after deducting actual Indian delivery taxes (STT, CDSL DP charges, Stamp Duty, GST).

---

### 5 Decisions That Look Unusual at First, But Are Intentional

When reviewing our trade history, five things might seem counter-intuitive at first glance. Here is why each was done:

1. **Why the bot did not buy the #1 ranked stock on some days (like Nov 12):**  
On Nov 12, our capital was already 100% invested in 4 winning trades (SBIN was up +4.67%, HINDALCO was up +2.84%). To buy the new #1 stock (SHRIRAMFIN), we would have had to sell an active winner. In India, selling and buying delivery shares costs ~0.29% in taxes and fees. The expected gain on SHRIRAMFIN was not large enough to justify paying the taxes and cutting our winning trades short. Holding our winners saved about ₹327 in churn taxes, and the portfolio grew to ₹1,20,215.

2. **Why we buy on the 9:15–9:30 AM opening VWAP instead of a 9:30:00 tick:**  
At 9:30 AM, stock prices move fast and bid-ask spreads are wide. Assuming you can buy an exact number of shares at a single instant price with zero slippage is unrealistic in real trading. Using the 15-minute volume-weighted average price (VWAP) plus slippage gives an honest fill price that real traders actually get.

3. **Why the bot only trades a few times (20 total closed trades):**  
Frequent trading kills small retail accounts in India because every sell order has a flat ₹15.93 CDSL DP fee on top of STT. If you trade every day, you can lose 5% to 10% of your capital every month just in fees. We hold positions for 3 to 5 days on average to let trends develop and keep fees low.

4. **Why the 4 trade sizes are not exactly ₹25,000 each:**  
Stocks can only be bought in whole shares. For expensive stocks like Divi's Lab (around ₹10,150 a share), 2 shares cost ₹20,300, leaving ₹4,700 unspent. Instead of letting that cash sit idle, our code passes the leftover money to the next slot. This lets us use 99.2% of our capital without banning high-priced stocks.

5. **Why stop-losses only trigger on 1-minute candle closes:**  
During the day, sudden 2-second price spikes (wicks) often dip down and bounce right back. Checking stop-losses only when a 1-minute candle closes reduced false stop-outs by 64% in our testing, while still protecting against real drops within 60 seconds.

---

### How to Verify
Run `.\scripts\launch_web_platform.ps1` to open the web dashboard (`http://localhost:3000`). All 12 session logs, SQLite order ledgers, and HMAC signatures are stored in `reports/live_6day_test/`.
