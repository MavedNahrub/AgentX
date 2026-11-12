"use client";

import React, { useEffect, useState } from "react";
import type { LiveSessionState, LiveTickMessage, PortfolioSummary, TickerSeriesPoint } from "@/types/agentx";

interface JudgeTourAndKpisProps {
  summary: PortfolioSummary;
  latestTradedDate: string;
  tickerSeries: TickerSeriesPoint[];
  showTour: boolean;
  liveTick: LiveTickMessage | null;
  liveSession?: LiveSessionState | null;
}

const TERMS = [
  "📐 Fair Average Price (VWAP): The volume-weighted average price across the day (Gold line on chart). Buying near this line avoids overpaying.",
  "🔒 Safety Stop-Loss (Floor): An automatic sell floor that protects your money if a stock drops. As the stock rises, this safety floor moves UP to lock in profit.",
  "🎯 Automatic Target & Safety Orders (GTT): A paired broker order that automatically sells at your Target Price for profit OR at your Safety Floor to prevent a big loss.",
  "🧠 5-Day AI Stock Ranker: Every morning, our AI scores all 50 Nifty stocks on their expected 5-day price trend and picks the strongest leaders.",
  "💧 Smart Cash Rollover: Any leftover cash from buying whole shares in Stock #1 automatically rolls into Stocks #2–#4 so your ₹1,00,000 doesn't sit idle.",
  "🧾 Govt Delivery Taxes (~0.29%): Every profit number shown here is 100% net after deducting real Indian delivery taxes (STT, ₹15.93 CDSL DP fee, Stamp Duty & GST).",
];

function fmtINR(v: number, sign = false): string {
  const s = v >= 0 ? (sign ? "+" : "") : "-";
  return (
    s +
    "₹" +
    Math.abs(v).toLocaleString("en-IN", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })
  );
}

export default function JudgeTourAndKpis({
  summary,
  latestTradedDate,
  tickerSeries,
  showTour,
  liveTick,
  liveSession,
}: JudgeTourAndKpisProps) {
  void tickerSeries;
  const [stepIdx, setStepIdx] = useState(0);
  const [termIdx, setTermIdx] = useState(0);
  const [flashDir, setFlashDir] = useState<"up" | "down" | null>(null);
  const liveStatus = liveSession?.status ?? liveTick?.live_status ?? "IDLE";
  const isSessionLive = ["STARTING", "MORNING_SCOUT", "CATCHING_UP", "LIVE"].includes(liveStatus);
  const isArmedPreOpen = liveStatus === "WAITING_FOR_0915_OPEN";
  const isSettling = liveStatus === "SETTLING";

  const [localTick, setLocalTick] = useState<{
    clock: string;
    eq: number;
    pos: number;
    cash: number;
    netPnl: number;
    netPct: number;
    unreal: number;
    totalPnl: number;
  }>({
    clock: "15:30 IST (Settled Close)",
    eq: summary.equity,
    pos: summary.invested_inr,
    cash: summary.cash,
    netPnl: summary.net_pnl_inr,
    netPct: summary.net_pct,
    unreal: summary.unrealized_pnl_inr,
    totalPnl: summary.total_pnl_inr,
  });

  // Sync strictly from verified backend portfolio summary & WebSocket ticks (no synthetic jitter)
  useEffect(() => {
    const nextEq = isSessionLive && liveTick ? liveTick.equity : summary.equity;
    const nextPos = isSessionLive && liveTick ? liveTick.invested_inr : summary.invested_inr;
    const nextCash = isSessionLive && liveTick ? liveTick.cash : summary.cash;
    const nextNetPnl = isSessionLive && liveTick ? liveTick.net_pnl_inr : summary.net_pnl_inr;
    const nextNetPct = isSessionLive && liveTick ? liveTick.net_pct : summary.net_pct;
    const nextUnreal = isSessionLive && liveTick ? liveTick.unrealized_pnl_inr : summary.unrealized_pnl_inr;
    const nextTotalPnl = isSessionLive && liveTick ? liveTick.total_pnl_inr : summary.total_pnl_inr;
    const nextClock = isSessionLive
      ? (liveTick?.clock_ist ?? `${liveSession?.market_clock_ist ?? "LIVE"} IST`)
      : isArmedPreOpen
      ? `Armed 09:15 IST (${liveSession?.market_clock_ist ?? ""})`
      : isSettling
      ? "15:30 IST (Settling)"
      : "15:30 IST (Settled Close)";

    setLocalTick((prev) => {
      if (Math.abs(nextEq - prev.eq) >= 0.01) {
        setFlashDir(nextEq >= prev.eq ? "up" : "down");
      }
      return {
        clock: nextClock,
        eq: nextEq,
        pos: nextPos,
        cash: nextCash,
        netPnl: nextNetPnl,
        netPct: nextNetPct,
        unreal: nextUnreal,
        totalPnl: nextTotalPnl,
      };
    });
    const timer = setTimeout(() => setFlashDir(null), 320);
    return () => clearTimeout(timer);
  }, [liveTick, liveSession, isSessionLive, isArmedPreOpen, isSettling, summary]);

  const STEPS = [
    {
      num: "Step 1 • Scan Market",
      lbl: "📡 Scan All 50 Nifty Stocks",
      title: "Step 1: Daily Scan of All 50 Nifty Stocks",
      desc: "Every morning, the system checks price trend, trading volume, and daily price swings across all 50 Nifty companies.",
      proof: `✅ 50/50 Nifty Stocks Scanned • Latest Trading Day: ${latestTradedDate}`,
      where: "👀 See it in: Top Stock Chart & Candlestick View",
    },
    {
      num: "Step 2 • Rank Best Picks",
      lbl: "🧠 Rank 5-Day Winners",
      title: "Step 2: AI Predicts the Best 5-Day Swing Stocks",
      desc: "Our AI ranks all 50 stocks by their expected 5-day gain and selects the strongest leaders (limiting any single sector to at most 2 stocks).",
      proof: `✅ Current #1 AI Pick: ${summary.top_ranked_symbol} (Expected 5D Gain: +${summary.top_ranked_alpha_pct.toFixed(2)}%)`,
      where: "👀 See it in: Bottom Tab #4 (Top 20 AI Stock Picks)",
    },
    {
      num: "Step 3 • Adaptive Strategy",
      lbl: "🏆 Match Trading Style",
      title: "Step 3: Match Each Stock with Its Best Trading Style",
      desc: "Picks the right style for the market mood (Smooth Uptrend Dip, Fast Breakout, or Safe Leader) and skips stocks with risky news.",
      proof: `✅ 6-Month Past Growth: +${summary.wf_6m_agent_ret.toFixed(2)}% (Beat Nifty 50 by +${summary.wf_6m_alpha.toFixed(2)}% after taxes)`,
      where: "👀 See it in: Bottom Tab #3 (6-Month Growth Proof)",
    },
    {
      num: "Step 4 • Protect Money",
      lbl: "🛡️ Safety Stop & Profit Lock",
      title: "Step 4: Invest ₹1,00,000 Across 4 Stocks with Automatic Safety Floors",
      desc: "Every stock gets an automatic Safety Stop-Loss and Target Sell Price. As a stock rises, its Safety Floor automatically moves UP to lock in profit.",
      proof: `✅ Closed Trade Wins: ${summary.win_count}/${summary.closed_trades_count} (${summary.win_rate_pct.toFixed(0)}%) • Taxes Saved by Avoiding Bad Swaps: ₹${Math.round(summary.churn_saved_inr).toLocaleString("en-IN")}`,
      where: "👀 See it in: Red/Green Lines on Chart & Center 'Price Plan' Card",
    },
    {
      num: "Step 5 • Clear Explanations",
      lbl: "📖 Simple Trade Stories",
      title: "Step 5: Every Buy, Sell & Tax Deducted Explained in Plain English",
      desc: "Tracks every rupee minute-by-minute from 9:15 AM to 3:30 PM IST with real Indian delivery taxes deducted and simple trade explanations.",
      proof: `✅ Govt Taxes Paid: ${fmtINR(summary.taxes_paid_inr)} • All ${summary.journal_audited_count} Trade Explanations Verified`,
      where: "👀 See it in: Bottom Tab #1 (Stocks Bought & Sold) & Tab #5 (Why We Traded)",
    },
  ];

  const depPct = (localTick.pos / Math.max(1, localTick.eq)) * 100;

  return (
    <div className="flex flex-col gap-3 mt-3">
      {/* Compact Collapsible Beginner Guide (starts closed to reduce visual noise) */}
      {showTour && (
        <details className="ax-card px-4 py-2.5">
          <summary className="cursor-pointer flex flex-wrap items-center justify-between gap-2 text-[13px] font-extrabold select-none">
            <span>💡 New here? Click to see how this AI Swing Fund works in 5 simple steps (+ Beginner Term Guide)</span>
            <span
              className="text-[11px] font-extrabold px-2.5 py-0.5 rounded-full"
              style={{ background: "var(--yellow-accent)", color: "#111827" }}
            >
              Show Simple Guide ▾
            </span>
          </summary>

          <div className="pt-3 mt-2.5 border-t" style={{ borderColor: "var(--grid-line)" }}>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-2 mb-2.5">
              {STEPS.map((s, idx) => {
                const active = idx === stepIdx;
                return (
                  <button
                    key={s.num}
                    type="button"
                    onClick={() => setStepIdx(idx)}
                    className="text-left rounded-xl p-2 px-2.5 border transition-all cursor-pointer"
                    style={{
                      background: active ? "var(--yellow-accent)" : "var(--input-bg)",
                      borderColor: active ? "var(--yellow-border)" : "var(--card-border-strong)",
                      color: active ? "#111827" : "var(--text-main)",
                    }}
                  >
                    <div
                      className="text-[10.5px] font-extrabold uppercase tracking-wide mb-0.5"
                      style={{ color: active ? "#111827" : "var(--text-muted)" }}
                    >
                      {s.num}
                    </div>
                    <div className="text-[12px] font-extrabold leading-tight">{s.lbl}</div>
                  </button>
                );
              })}
            </div>

            <div
              className="grid grid-cols-1 lg:grid-cols-[1.7fr_1.1fr] gap-3 rounded-xl p-2.5 px-3.5 mb-2.5 border"
              style={{ background: "var(--surface-subtle)", borderColor: "var(--card-border-strong)" }}
            >
              <div>
                <div className="text-[13px] font-extrabold mb-1">{STEPS[stepIdx].title}</div>
                <div className="text-[12px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
                  {STEPS[stepIdx].desc}
                </div>
              </div>
              <div
                className="flex flex-col justify-center gap-1 text-[11.5px] lg:border-l lg:pl-3"
                style={{ borderColor: "var(--card-border-strong)" }}
              >
                <div className="font-bold" style={{ color: "var(--green-main)" }}>
                  {STEPS[stepIdx].proof}
                </div>
                <div className="font-semibold" style={{ color: "var(--blue-accent)" }}>
                  {STEPS[stepIdx].where}
                </div>
              </div>
            </div>

            <div
              className="flex flex-wrap items-center gap-1.5 pt-2 border-t"
              style={{ borderColor: "var(--grid-line)" }}
            >
              <span className="text-[11px] font-extrabold mr-1" style={{ color: "var(--text-muted)" }}>
                📘 Plain-English Terms:
              </span>
              {[
                "Fair Price (VWAP)",
                "Safety Stop-Loss",
                "Auto Target & Stop (GTT)",
                "5-Day AI Ranker",
                "Smart Cash Rollover",
                `Govt Taxes (~${summary.statutory_rt_tax_pct}%)`,
              ].map((lbl, idx) => {
                const active = idx === termIdx;
                return (
                  <button
                    key={lbl}
                    type="button"
                    onClick={() => setTermIdx(idx)}
                    className="rounded-md px-2.5 py-1 text-[11px] font-bold border cursor-pointer transition-all"
                    style={{
                      background: active ? "var(--yellow-accent)" : "var(--input-bg)",
                      color: active ? "#111827" : "var(--text-muted)",
                      borderColor: active ? "var(--yellow-border)" : "var(--card-border-strong)",
                    }}
                  >
                    {lbl}
                  </button>
                );
              })}
              <div
                className="text-[11.5px] font-semibold rounded-md px-2.5 py-1 flex-1 min-w-[260px] border"
                style={{
                  background: "var(--yellow-soft)",
                  borderColor: "var(--yellow-border)",
                  color: "var(--text-main)",
                }}
              >
                {TERMS[termIdx] || TERMS[0]}
              </div>
            </div>
          </div>
        </details>
      )}

      {/* 4 Key Portfolio Summary Cards */}
      <section className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
        {/* KPI 1: Total Portfolio Value */}
        <div
          className="ax-card p-3.5 px-4 transition-colors duration-200"
          style={{
            borderColor:
              flashDir === "up"
                ? "var(--green-main)"
                : flashDir === "down"
                ? "var(--red-main)"
                : "var(--card-border)",
          }}
        >
          <div className="flex items-center justify-between text-[12px] font-bold mb-1" style={{ color: "var(--text-muted)" }}>
            <span>💰 Total Portfolio Value</span>
            <span
              className="rounded-md px-2 py-0.5 text-[10.5px] font-extrabold"
              title={liveSession?.message || "Official NSE session status"}
              style={{
                background: isSessionLive
                  ? "rgba(22, 163, 74, 0.16)"
                  : isArmedPreOpen
                  ? "rgba(234, 179, 8, 0.22)"
                  : "var(--input-bg)",
                color: isSessionLive
                  ? "var(--green-main)"
                  : isArmedPreOpen
                  ? "#B45309"
                  : "var(--text-main)",
              }}
            >
              {isSessionLive
                ? `🟢 LIVE • Min ${liveSession?.current_minute ?? liveTick?.current_minute ?? 0}/375`
                : isArmedPreOpen
                ? "⏳ ARMED 09:15 IST"
                : isSettling
                ? "⏳ SETTLING 15:30"
                : "🌙 15:30 IST CLOSED"}
            </span>
          </div>
          <div
            className="tabular-nums text-[22px] font-extrabold font-mono mb-1 transition-colors"
            style={{
              color:
                flashDir === "up"
                  ? "var(--green-main)"
                  : flashDir === "down"
                  ? "var(--red-main)"
                  : "var(--text-main)",
            }}
          >
            {fmtINR(localTick.eq)}
          </div>
          <div
            className="tabular-nums text-[11.8px] font-bold"
            style={{ color: localTick.netPnl >= 0 ? "var(--green-main)" : "var(--red-main)" }}
          >
            Total Gain: {fmtINR(localTick.netPnl, true)} ({localTick.netPct >= 0 ? "+" : ""}
            {localTick.netPct.toFixed(2)}% from ₹1,00,000)
          </div>
        </div>

        {/* KPI 2: Money in Stocks */}
        <div className="ax-card p-3.5 px-4">
          <div className="flex items-center justify-between text-[12px] font-bold mb-1" style={{ color: "var(--text-muted)" }}>
            <span>📦 Money in Stocks (4 Active)</span>
            <span className="tabular-nums font-mono font-extrabold text-[11px]" style={{ color: "var(--blue-accent)" }}>
              {localTick.clock}
            </span>
          </div>
          <div className="tabular-nums text-[22px] font-extrabold font-mono mb-1">
            {fmtINR(localTick.pos)}
          </div>
          <div className="tabular-nums text-[11.8px] font-bold" style={{ color: "var(--text-muted)" }}>
            Cash in Bank: {fmtINR(localTick.cash)} ({depPct.toFixed(1)}% Invested)
          </div>
        </div>

        {/* KPI 3: Total Profit Earned */}
        <div className="ax-card p-3.5 px-4">
          <div className="flex items-center justify-between text-[12px] font-bold mb-1" style={{ color: "var(--text-muted)" }}>
            <span>💵 Total Profit Earned</span>
            <span className="text-[11px] font-extrabold" style={{ color: "var(--green-main)" }}>
              {summary.win_count} of {summary.closed_trades_count} Trades Won ({summary.win_rate_pct.toFixed(0)}%)
            </span>
          </div>
          <div
            className="tabular-nums text-[22px] font-extrabold font-mono mb-1"
            style={{ color: localTick.totalPnl >= 0 ? "var(--green-main)" : "var(--red-main)" }}
          >
            {fmtINR(localTick.totalPnl, true)}
          </div>
          <div className="tabular-nums text-[11.8px] font-bold" style={{ color: "var(--green-main)" }}>
            Booked Cash: {fmtINR(summary.realized_pnl_inr, true)} | Open Gain: {fmtINR(localTick.unreal, true)}
          </div>
        </div>

        {/* KPI 4: 5-Day Swing Performance */}
        <div className="ax-card p-3.5 px-4">
          <div className="flex items-center justify-between text-[12px] font-bold mb-1" style={{ color: "var(--text-muted)" }}>
            <span>🎯 5-Day Swing Average</span>
            <span className="text-[11px] font-extrabold" style={{ color: "var(--green-main)" }}>
              {summary.settled_5d_alpha_wins} of {summary.settled_5d_windows_count} Beat Market
            </span>
          </div>
          <div className="tabular-nums text-[21px] font-extrabold font-mono mb-1" style={{ color: "var(--green-main)" }}>
            +{summary.avg_5d_ret_pct.toFixed(2)}% per 5 Days
          </div>
          <div className="tabular-nums text-[11.8px] font-bold" style={{ color: "var(--text-muted)" }}>
            Avg Profit: +₹{Math.round(summary.avg_5d_profit_inr).toLocaleString("en-IN")} (Beat Nifty by +{summary.avg_5d_alpha_pct.toFixed(2)}%)
          </div>
        </div>
      </section>
    </div>
  );
}
