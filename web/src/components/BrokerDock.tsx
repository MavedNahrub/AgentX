"use client";

import React, { useEffect, useId, useMemo, useRef, useState } from "react";
import { ColorType, LineStyle, Time, createChart } from "lightweight-charts";
import type { CurvePoint, FriendlyEvent, WalkForwardPoint, WorkstationPayload } from "@/types/agentx";

interface BrokerDockProps {
  ws: WorkstationPayload;
  isDark: boolean;
  onSelectSymbol: (sym: string) => void;
}

const DOCK_TABS = [
  "📦 1. Stocks Bought, Held & Sold",
  "🎯 2. 5-Day Swing Profit & Daily Log",
  "📈 3. 6-Month Growth Proof",
  "🏆 4. Top 20 AI Stock Picks",
  "📖 5. Why We Traded (Simple English)",
];

function GenericResponsiveTable({
  caption,
  columns,
  rows,
  rowHeaderKey,
  onRowClick,
}: {
  caption: string;
  columns: string[];
  rows: Record<string, unknown>[];
  rowHeaderKey?: string;
  onRowClick?: (row: Record<string, unknown>) => void;
}) {
  const captionId = useId();
  if (!rows || rows.length === 0) {
    return (
      <div
        className="rounded-xl p-3.5 text-[12.5px] font-semibold border"
        style={{
          background: "var(--surface-subtle)",
          borderColor: "var(--card-border-strong)",
          color: "var(--text-muted)",
        }}
      >
        ℹ️ No rows recorded for this scope.
      </div>
    );
  }

  const headerCol = rowHeaderKey && columns.includes(rowHeaderKey) ? rowHeaderKey : columns[0];
  const restCols = columns.filter((c) => c !== headerCol);

  return (
    <div tabIndex={0} role="region" aria-labelledby={captionId} className="table-wrapper">
      <table className="ax-table">
        <caption id={captionId}>{caption}</caption>
        <thead>
          <tr>
            <th scope="col">{headerCol}</th>
            {restCols.map((col) => (
              <th key={col} scope="col">
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rIdx) => (
            <tr
              key={rIdx}
              onClick={() => onRowClick?.(row)}
              className={onRowClick ? "cursor-pointer" : undefined}
            >
              <th scope="row">{String(row[headerCol] ?? "—")}</th>
              {restCols.map((col) => {
                const raw = String(row[col] ?? "—");
                const isPos = raw.startsWith("🟢") || raw.startsWith("+");
                const isNeg = raw.startsWith("🔴") || raw.startsWith("-");
                return (
                  <td
                    key={col}
                    data-label={col}
                    style={{
                      color: isPos
                        ? "var(--green-main)"
                        : isNeg
                        ? "var(--red-main)"
                        : "var(--text-main)",
                      fontWeight: isPos || isNeg ? 700 : 500,
                    }}
                  >
                    {raw}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function IntradayCurveCanvas({
  points,
  isDark,
  scopeLabel,
}: {
  points: CurvePoint[];
  isDark: boolean;
  scopeLabel: string;
}) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || points.length < 2) return;

    const chart = createChart(el, {
      width: el.clientWidth,
      height: 330,
      layout: {
        background: { type: ColorType.Solid, color: isDark ? "#151B28" : "#FFFFFF" },
        textColor: isDark ? "#94A3B8" : "#64748B",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: isDark ? "#1E293B" : "#F1F5F9" },
        horzLines: { color: isDark ? "#1E293B" : "#F1F5F9" },
      },
      rightPriceScale: {
        borderColor: isDark ? "#263042" : "#E2E8F0",
      },
      timeScale: {
        borderColor: isDark ? "#263042" : "#E2E8F0",
        timeVisible: true,
      },
    });

    const slice = points;

    // Convert time strings to Unix epochs so TradingView x-axis shows IST trading hours.
    // pt.t can be "09:15" (single-day) or "10-26 09:15" (multi-day combined).
    // We place times as-if-UTC so TradingView (which renders UTC) shows IST labels.
    const refDateMs = Date.UTC(2026, 9, 26, 0, 0, 0); // Oct 26 2026 00:00 UTC in ms
    const toEpoch = (pt: CurvePoint, idx: number): number => {
      const raw = pt.t || "";
      // Extract HH:MM — could be "09:15" or "10-26 09:15" or "2026-10-26 09:15:00"
      const timeMatch = raw.match(/(\d{1,2}):(\d{2})(?::|\s|$)/);
      if (!timeMatch) {
        // Fallback: spread linearly from 09:15 over trading session
        return Math.floor(refDateMs / 1000) + 9 * 3600 + 15 * 60 + idx * 60;
      }
      const hh = Number(timeMatch[1]);
      const mm = Number(timeMatch[2]);
      // For multi-day combined, use idx-based day offset to keep timestamps monotonic
      const dayOffset = pt.date ? (new Date(pt.date + "T00:00:00Z").getTime() - refDateMs) / 1000 : 0;
      return Math.floor(refDateMs / 1000) + dayOffset + hh * 3600 + mm * 60;
    };

    const aiArea = chart.addAreaSeries({
      lineColor: "#16A34A",
      topColor: "rgba(22, 163, 74, 0.22)",
      bottomColor: "rgba(22, 163, 74, 0.02)",
      lineWidth: 3,
      priceFormat: {
        type: "custom",
        formatter: (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`,
      },
    });
    const nifLine = chart.addLineSeries({
      color: "#94A3B8",
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
      priceFormat: {
        type: "custom",
        formatter: (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`,
      },
    });

    aiArea.setData(slice.map((pt, i) => ({ time: toEpoch(pt, i) as Time, value: pt.p })));
    nifLine.setData(slice.map((pt, i) => ({ time: toEpoch(pt, i) as Time, value: pt.n })));
    chart.timeScale().fitContent();

    const ro = new ResizeObserver(() => {
      if (ref.current) chart.applyOptions({ width: ref.current.clientWidth });
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
      chart.remove();
    };
  }, [points, isDark]);

  const lastPt = points[Math.max(0, points.length - 1)];

  return (
    <div
      className="rounded-xl p-3 border"
      style={{ background: "var(--card-bg)", borderColor: "var(--card-border-strong)" }}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <div className="text-[12.5px] font-extrabold">
          📈 {scopeLabel} —{" "}
          <span style={{ color: "var(--green-main)" }}>
            AI {lastPt ? `${lastPt.p >= 0 ? "+" : ""}${lastPt.p.toFixed(2)}%` : ""}
          </span>{" "}
          vs{" "}
          <span style={{ color: "var(--text-muted)" }}>
            Nifty {lastPt ? `${lastPt.n >= 0 ? "+" : ""}${lastPt.n.toFixed(2)}%` : ""}
          </span>
        </div>
      </div>
      <div ref={ref} className="w-full h-[330px]" />
    </div>
  );
}

function WalkForward6MChart({
  curve,
  isDark,
  agentRet,
  naiveRet,
  bhRet,
}: {
  curve: WalkForwardPoint[];
  isDark: boolean;
  agentRet: number;
  naiveRet: number;
  bhRet: number;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [ratio, setRatio] = useState(1.0);

  useEffect(() => {
    if (!playing) return;
    setRatio((r) => (r >= 0.98 ? 0.1 : r));
    let raf = 0;
    const step = () => {
      setRatio((prev) => {
        const next = prev + 0.006;
        if (next >= 1.0) {
          setPlaying(false);
          return 1.0;
        }
        raf = requestAnimationFrame(step);
        return next;
      });
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  useEffect(() => {
    const el = ref.current;
    if (!el || curve.length < 2) return;

    const chart = createChart(el, {
      width: el.clientWidth,
      height: 360,
      layout: {
        background: { type: ColorType.Solid, color: isDark ? "#151B28" : "#FFFFFF" },
        textColor: isDark ? "#94A3B8" : "#64748B",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: isDark ? "#1E293B" : "#F1F5F9" },
        horzLines: { color: isDark ? "#1E293B" : "#F1F5F9" },
      },
      rightPriceScale: {
        borderColor: isDark ? "#263042" : "#E2E8F0",
      },
      timeScale: {
        borderColor: isDark ? "#263042" : "#E2E8F0",
      },
    });

    const count = Math.max(3, Math.floor(curve.length * ratio));
    const slice = curve.slice(0, count);

    const sAgent = chart.addAreaSeries({
      lineColor: "#16A34A",
      topColor: "rgba(22, 163, 74, 0.18)",
      bottomColor: "rgba(22, 163, 74, 0.01)",
      lineWidth: 3,
    });
    const sNaive = chart.addLineSeries({
      color: "#D97706",
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
    });
    const sNifty = chart.addLineSeries({
      color: "#EF4444",
      lineWidth: 2,
      lineStyle: LineStyle.Dotted,
    });

    sAgent.setData(slice.map((d) => ({ time: d.date as Time, value: d.agentx_inr })));
    sNaive.setData(slice.map((d) => ({ time: d.date as Time, value: d.naive_inr })));
    sNifty.setData(slice.map((d) => ({ time: d.date as Time, value: d.nifty_inr })));
    chart.timeScale().fitContent();

    const ro = new ResizeObserver(() => {
      if (ref.current) chart.applyOptions({ width: ref.current.clientWidth });
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
      chart.remove();
    };
  }, [curve, isDark, ratio]);

  return (
    <div
      className="rounded-xl p-3.5 border mb-3"
      style={{ background: "var(--card-bg)", borderColor: "var(--card-border-strong)" }}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
        <div className="flex flex-wrap items-center gap-4 text-[12px] font-bold">
          <span style={{ color: "#16A34A" }}>🟢 Your AI Portfolio (+{agentRet.toFixed(2)}%)</span>
          <span style={{ color: "#D97706" }}>🟠 Basic Rule Bot ({naiveRet >= 0 ? "+" : ""}{naiveRet.toFixed(2)}%)</span>
          <span style={{ color: "#EF4444" }}>🔴 Nifty 50 Buy &amp; Hold (+{bhRet.toFixed(2)}%)</span>
        </div>
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          className="rounded-lg px-3 py-1 text-[11.5px] font-extrabold cursor-pointer border-none"
          style={{ background: "var(--yellow-accent)", color: "#111827" }}
        >
          {playing ? "⏸ Pause" : "▶ Play 6M Growth"}
        </button>
      </div>
      <div ref={ref} className="w-full h-[360px]" />
    </div>
  );
}

export default function BrokerDock({ ws, isDark, onSelectSymbol }: BrokerDockProps) {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedTxDate, setSelectedTxDate] = useState<string>(ws.latest_traded_date || "ALL");
  const [showReceipts, setShowReceipts] = useState(false);
  const [receiptSubTab, setReceiptSubTab] = useState<"fills" | "gtt" | "cnotes">("fills");
  const [curveScope, setCurveScope] = useState<string>("ALL");
  const [scopedPoints, setScopedPoints] = useState<CurvePoint[]>(ws.overview_pts);
  const [scopedEvents, setScopedEvents] = useState<FriendlyEvent[]>(ws.dock.intraday_events);
  const [horizonTableMode, setHorizonTableMode] = useState<"SWING_5D" | "MTM_1D">("SWING_5D");
  const [journalCat, setJournalCat] = useState<string>("ALL");
  const [journalSearch, setJournalSearch] = useState<string>("");

  const tablistRef = useRef<HTMLDivElement | null>(null);
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const baseId = useId();

  // Scroll-driven animation fallback for horizontal tablist per modern-web-guidance `tabs`
  useEffect(() => {
    const scroller = tablistRef.current;
    if (!scroller) return;
    if (typeof CSS !== "undefined" && !CSS.supports("animation-timeline", "scroll()")) {
      const first = scroller.firstElementChild;
      const last = scroller.lastElementChild;
      if (!first || !last) return;
      const obs = new IntersectionObserver(
        (entries) => {
          for (const { target, intersectionRatio } of entries) {
            scroller.classList.toggle(target === first ? "more-start" : "more-end", intersectionRatio < 1);
          }
        },
        { root: scroller, threshold: 1, rootMargin: "1px" }
      );
      obs.observe(first);
      obs.observe(last);
      return () => obs.disconnect();
    }
  }, []);

  // Fetch single-day or continuous curve and session-specific intraday events dynamically from FastAPI
  useEffect(() => {
    if (curveScope === "ALL") {
      setScopedPoints(ws.overview_pts);
      setScopedEvents(ws.dock.intraday_events);
      return;
    }

    // Instant local cache hit if pre-loaded
    if (ws.dock.events_by_session && ws.dock.events_by_session[curveScope]) {
      setScopedEvents(ws.dock.events_by_session[curveScope]);
    } else {
      const filteredByDate = (ws.dock.intraday_events || []).filter((e) => e.date === curveScope);
      if (filteredByDate.length > 0) {
        setScopedEvents(filteredByDate);
      }
    }

    let cancelled = false;
    fetch(`/api/intraday-curve?scope=${encodeURIComponent(curveScope)}`)
      .then((r) => r.json())
      .then((d) => {
        if (!cancelled && Array.isArray(d.points)) {
          setScopedPoints(d.points);
        }
        if (!cancelled && Array.isArray(d.events)) {
          setScopedEvents(d.events);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setScopedPoints(ws.overview_pts);
          setScopedEvents(ws.dock.intraday_events);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [curveScope, ws.overview_pts, ws.dock.intraday_events, ws.dock.events_by_session]);

  const handleSelectTab = (idx: number) => {
    setActiveTab(idx);
    const btn = tabRefs.current[idx];
    btn?.scrollIntoView({ block: "nearest", inline: "nearest" });
  };

  // Tab 1 computed rows (streamlined columns for easy reading)
  const txComputation = useMemo(() => {
    const displayCols = [
      "Stock",
      "Session Action",
      "Current Status",
      "Industry",
      "Shares",
      "Buy Price",
      "Current / Sold Price",
      "Your Profit / Loss",
      "Safety Sell Floor (Stop-Loss)",
      "Target Sell Price",
    ];
    const soldCols = [
      "Stock",
      "Session Action",
      "Industry",
      "Holding Info",
      "Shares",
      "Buy Price",
      "Current / Sold Price",
      "Your Profit / Loss",
    ];

    const allFills = ws.dock.order_fills || [];
    const buyFills = allFills.filter((f) => f.side === "BUY");
    const sellFills = allFills.filter((f) => f.side === "SELL");

    if (selectedTxDate === "ALL") {
      const heldOrBought = ws.dock.buy_fill_rows_by_date;
      const soldRows = ws.dock.tested_rows.filter((r) => r.row_type === "SOLD_BOOKED");
      const realizedSum = sellFills.reduce((acc, s) => acc + Number(s.realized_net_pnl_inr || 0), 0);
      return {
        displayCols,
        soldCols,
        heldOrBought,
        soldRows,
        dayFills: allFills,
        dayNotes: ws.dock.contract_notes,
        boughtCount: buyFills.length,
        soldCount: sellFills.length,
        realizedSum,
        scopeLabel: `All ${ws.sorted_traded_dates.length} Trading Days Combined (2026-10-26 → ${ws.latest_traded_date})`,
      };
    }

    const heldOrBought: Record<string, string>[] =
      selectedTxDate === ws.latest_traded_date
        ? ws.dock.tested_rows.filter((r) => r.row_type === "ACTIVE_HOLDING")
        : ws.dock.buy_fill_rows_by_date
            .filter(
              (r) =>
                r.buy_date === selectedTxDate ||
                (r.buy_date < selectedTxDate && (r.sell_date === "" || r.sell_date > selectedTxDate))
            )
            .map((r) => ({
              ...r,
              "Session Action":
                r.buy_date === selectedTxDate
                  ? `🛒 Bought on ${r.buy_date}`
                  : `🟢 Held (Bought ${r.buy_date})`,
            }));

    const soldRows = ws.dock.tested_rows.filter(
      (r) => r.row_type === "SOLD_BOOKED" && r.sell_date === selectedTxDate
    );
    const dayFills = allFills.filter((f) => String(f.as_of_date) === selectedTxDate);
    const dayNotes = (ws.dock.contract_notes || []).filter(
      (c) => String(c.trade_date) === selectedTxDate
    );
    const dayBuys = buyFills.filter((f) => String(f.as_of_date) === selectedTxDate);
    const daySells = sellFills.filter((f) => String(f.as_of_date) === selectedTxDate);
    const realizedSum = daySells.reduce((acc, s) => acc + Number(s.realized_net_pnl_inr || 0), 0);

    return {
      displayCols,
      soldCols,
      heldOrBought,
      soldRows,
      dayFills,
      dayNotes,
      boughtCount: dayBuys.length,
      soldCount: daySells.length,
      realizedSum,
      scopeLabel: `Trading Day ${selectedTxDate}`,
    };
  }, [selectedTxDate, ws]);

  const journalCategories = useMemo(() => {
    const cats = new Set<string>(["ALL"]);
    ws.dock.journal_entries.forEach((j) => cats.add(j.category));
    return Array.from(cats);
  }, [ws.dock.journal_entries]);

  const filteredJournal = useMemo(() => {
    const q = journalSearch.trim().toUpperCase();
    return ws.dock.journal_entries.filter((j) => {
      if (journalCat !== "ALL" && j.category !== journalCat) return false;
      if (
        q &&
        !j.symbol.toUpperCase().includes(q) &&
        !j.headline.toUpperCase().includes(q)
      ) {
        return false;
      }
      return true;
    });
  }, [ws.dock.journal_entries, journalCat, journalSearch]);

  const topRankingRows = useMemo(() => {
    return ws.dock.top_rankings.map((r) => ({
      "AI Rank": `#${r.alpha_rank}`,
      Stock: String(r.symbol),
      Industry: String(r.sector),
      "Trading Style": String(r.assigned_strategy || ""),
      "Expected 5-Day Gain": `${Number(r.expected_5d_alpha_pct) >= 0 ? "+" : ""}${Number(
        r.expected_5d_alpha_pct
      ).toFixed(2)}%`,
      "AI Confidence": `${Number(r.conviction_pct).toFixed(0)}%`,
      "Key Strength #1": String(r.shap_driver_1 || ""),
      "Key Strength #2": String(r.shap_driver_2 || ""),
    }));
  }, [ws.dock.top_rankings]);

  const sessionCompRows = useMemo(() => {
    return ws.session_ledger.map((s) => ({
      Day: `Day ${s.day_num}`,
      Date: s.date,
      "Start Value (₹)": `₹${s.prev_eq.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      "End Value (₹)": `₹${s.eq.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      "1-Day Profit (₹)": `${s.session_pnl_inr >= 0 ? "+" : "-"}₹${Math.abs(s.session_pnl_inr).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      "1-Day Return": `${s.sess_ret >= 0 ? "+" : ""}${s.sess_ret.toFixed(2)}%`,
      "Nifty 1-Day": `${s.nif_sess >= 0 ? "+" : ""}${s.nif_sess.toFixed(2)}%`,
      "5-Day Window": s.fwd_5d_status_label,
      "Value 5 Days Later": `₹${s.fwd_5d_end_eq.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      "5-Day Profit (₹)": `${s.fwd_5d_profit_inr >= 0 ? "+" : "-"}₹${Math.abs(s.fwd_5d_profit_inr).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      "Your 5-Day Return": `${s.fwd_5d_ret_pct >= 0 ? "+" : ""}${s.fwd_5d_ret_pct.toFixed(2)}%`,
      "Nifty 5-Day Return": `${s.fwd_5d_nifty_pct >= 0 ? "+" : ""}${s.fwd_5d_nifty_pct.toFixed(2)}%`,
      "Extra Gain vs Nifty": `${s.fwd_5d_alpha_pct >= 0 ? "+" : ""}${s.fwd_5d_alpha_pct.toFixed(2)}%`,
      "Total Growth": `+${s.ret_pct.toFixed(2)}%`,
      "Nifty Total Growth": `+${s.nif_cum.toFixed(2)}%`,
    }));
  }, [ws.session_ledger]);

  const completedSwingCohortRows = useMemo(() => {
    return (ws.swing_analytics?.completed_swing_trades || []).map((t) => ({
      Stock: t.stock,
      Industry: t.sector,
      "Bought → Sold Date": `${t.buy_date} → ${t.sell_date}`,
      "Days Held": `${t.sessions_held} Day${t.sessions_held > 1 ? "s" : ""}`,
      Shares: `${t.shares} Shares`,
      "Buy → Sell Price": `₹${t.buy_px.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} → ₹${t.sell_px.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      "Price Gain": `${t.gross_ret_pct >= 0 ? "+" : ""}${t.gross_ret_pct.toFixed(2)}%`,
      "Net Profit (After Tax)": `${t.net_pnl_inr >= 0 ? "+" : "-"}₹${Math.abs(t.net_pnl_inr).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      "Govt Tax Paid": `₹${t.round_trip_tax_inr.toFixed(2)}`,
      "Why We Sold": t.exit_badge,
    }));
  }, [ws.swing_analytics]);

  const formattedEventRows = useMemo(() => {
    return scopedEvents.map((ev) => ({
      Date: ev.date || curveScope,
      Time: ev.time,
      Stock: ev.stock,
      Action: ev.event,
      "What Happened": ev.summary,
    }));
  }, [scopedEvents, curveScope]);

  return (
    <section className="ax-card p-4 mt-3.5">
      {/* Accessible Single-Row Scrollable Tablist (modern-web-guidance: tabs) */}
      <div
        ref={tablistRef}
        role="tablist"
        aria-label="Broker Trading & Audit Dock"
        className="dock-tablist"
        onKeyDown={(e) => {
          if (e.key === "ArrowRight") {
            e.preventDefault();
            const next = (activeTab + 1) % DOCK_TABS.length;
            handleSelectTab(next);
            tabRefs.current[next]?.focus();
          } else if (e.key === "ArrowLeft") {
            e.preventDefault();
            const prev = (activeTab - 1 + DOCK_TABS.length) % DOCK_TABS.length;
            handleSelectTab(prev);
            tabRefs.current[prev]?.focus();
          }
        }}
      >
        {DOCK_TABS.map((label, idx) => (
          <button
            key={label}
            ref={(el) => {
              tabRefs.current[idx] = el;
            }}
            id={`${baseId}-tab-${idx}`}
            type="button"
            role="tab"
            aria-selected={activeTab === idx}
            tabIndex={activeTab === idx ? 0 : -1}
            onClick={() => handleSelectTab(idx)}
            className="dock-tab"
          >
            {label}
          </button>
        ))}
      </div>

      {/* Panel 0: Stocks Bought, Held & Sold */}
      <div
        role="tabpanel"
        aria-labelledby={`${baseId}-tab-0`}
        hidden={activeTab !== 0}
      >
        <div className="pt-4 flex flex-col gap-3.5">
          {/* Date Selector + Simple Explanation Banner */}
          <div className="grid grid-cols-1 lg:grid-cols-[1.2fr_1.8fr] gap-3 items-center">
            <div>
              <label
                htmlFor="tx-date-select"
                className="block text-[12px] font-extrabold mb-1"
                style={{ color: "var(--text-muted)" }}
              >
                📅 Choose Trading Day:
              </label>
              <select
                id="tx-date-select"
                value={selectedTxDate}
                onChange={(e) => {
                  setSelectedTxDate(e.target.value);
                  setCurveScope(e.target.value);
                }}
                className="w-full rounded-xl px-3 py-2 text-[12.5px] font-bold border outline-none cursor-pointer"
                style={{
                  background: "var(--input-bg)",
                  borderColor: "var(--card-border-strong)",
                  color: "var(--text-main)",
                }}
              >
                {ws.sorted_traded_dates.map((dt) => (
                  <option key={dt} value={dt}>
                    📅 {dt} {dt === ws.latest_traded_date ? "(Latest Trading Day)" : ""}
                  </option>
                ))}
                <option value="ALL">
                  📊 All {ws.sorted_traded_dates.length} Trading Days Combined (2026-10-26 → {ws.latest_traded_date})
                </option>
              </select>
            </div>
            <div
              className="rounded-xl p-2.5 px-3.5 text-[12.5px] font-semibold border lg:mt-5"
              style={{
                background: "var(--surface-subtle)",
                borderColor: "var(--card-border-strong)",
                color: "var(--text-muted)",
              }}
            >
              📌 Showing stocks bought, held, and sold for{" "}
              <b style={{ color: "var(--text-main)" }}>{txComputation.scopeLabel}</b>. Click any stock row to view its price chart above.
            </div>
          </div>

          {/* 3 Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div
              className="rounded-xl p-3 px-4 border"
              style={{ background: "var(--card-bg)", borderColor: "var(--card-border-strong)" }}
            >
              <div className="text-[11.5px] font-bold" style={{ color: "var(--text-muted)" }}>
                🛒 NEW STOCKS BOUGHT
              </div>
              <div className="text-[17px] font-extrabold mt-0.5">
                {txComputation.boughtCount} Stock{txComputation.boughtCount === 1 ? "" : "s"} Bought
              </div>
            </div>
            <div
              className="rounded-xl p-3 px-4 border"
              style={{ background: "var(--card-bg)", borderColor: "var(--card-border-strong)" }}
            >
              <div className="text-[11.5px] font-bold" style={{ color: "var(--text-muted)" }}>
                🟢 STOCKS HELD IN PORTFOLIO
              </div>
              <div className="text-[17px] font-extrabold mt-0.5" style={{ color: "var(--blue-accent)" }}>
                {txComputation.heldOrBought.length} Active Position{txComputation.heldOrBought.length === 1 ? "" : "s"}
              </div>
            </div>
            <div
              className="rounded-xl p-3 px-4 border"
              style={{ background: "var(--card-bg)", borderColor: "var(--card-border-strong)" }}
            >
              <div className="text-[11.5px] font-bold" style={{ color: "var(--text-muted)" }}>
                💰 STOCKS SOLD &amp; PROFIT BOOKED
              </div>
              <div
                className="tabular-nums text-[17px] font-extrabold mt-0.5"
                style={{
                  color:
                    txComputation.realizedSum >= 0 ? "var(--green-main)" : "var(--red-main)",
                }}
              >
                {txComputation.soldCount} Sold • Net Profit: {txComputation.realizedSum >= 0 ? "+" : "-"}₹
                {Math.abs(txComputation.realizedSum).toLocaleString("en-IN", {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
              </div>
            </div>
          </div>

          {/* Table 1: Stocks Bought & Held */}
          <div>
            <h4 className="text-[13.5px] font-extrabold mb-2">
              🟢 1. Stocks Bought &amp; Held ({txComputation.scopeLabel})
            </h4>
            <GenericResponsiveTable
              caption={`Stocks Bought and Held for ${txComputation.scopeLabel}`}
              columns={txComputation.displayCols}
              rows={txComputation.heldOrBought}
              rowHeaderKey="Stock"
              onRowClick={(r) => r.Stock && onSelectSymbol(String(r.Stock))}
            />
          </div>

          {/* Table 2: Stocks Sold & Profit Booked */}
          <div>
            <h4 className="text-[13.5px] font-extrabold mb-2">
              🏁 2. Stocks Sold &amp; Profit Booked ({txComputation.scopeLabel})
            </h4>
            <GenericResponsiveTable
              caption={`Stocks Sold and Profit Booked for ${txComputation.scopeLabel}`}
              columns={txComputation.soldCols}
              rows={txComputation.soldRows}
              rowHeaderKey="Stock"
              onRowClick={(r) => r.Stock && onSelectSymbol(String(r.Stock))}
            />
          </div>

          {/* Collapsible Official Broker Receipts & Tax Notes (modern-web-guidance: content-first-noise-reduction) */}
          <details
            className="rounded-xl border p-3"
            style={{ borderColor: "var(--card-border-strong)", background: "var(--surface-subtle)" }}
            open={showReceipts}
            onToggle={(e) => setShowReceipts((e.currentTarget as HTMLDetailsElement).open)}
          >
            <summary className="cursor-pointer flex items-center justify-between text-[12.5px] font-extrabold select-none">
              <span>🧾 View Detailed Broker Order Receipts &amp; Government Tax Notes ({txComputation.scopeLabel})</span>
              <span>{showReceipts ? "▴ Hide" : "▾ Show Details"}</span>
            </summary>

            <div className="mt-3 flex flex-col gap-3">
              <div className="pill-switch w-fit">
                <button
                  type="button"
                  onClick={() => setReceiptSubTab("fills")}
                  className={`pill-btn ${receiptSubTab === "fills" ? "active" : ""}`}
                >
                  🧾 Order Receipts ({txComputation.dayFills.length})
                </button>
                <button
                  type="button"
                  onClick={() => setReceiptSubTab("gtt")}
                  className={`pill-btn ${receiptSubTab === "gtt" ? "active" : ""}`}
                >
                  🎯 Auto Stop &amp; Target Orders ({ws.dock.gtt_triggers.length})
                </button>
                <button
                  type="button"
                  onClick={() => setReceiptSubTab("cnotes")}
                  className={`pill-btn ${receiptSubTab === "cnotes" ? "active" : ""}`}
                >
                  📜 Govt Tax Contract Notes ({txComputation.dayNotes.length})
                </button>
              </div>

              {receiptSubTab === "fills" && txComputation.dayFills.length > 0 && (
                <GenericResponsiveTable
                  caption="Official Broker Order Fill Ledger"
                  columns={[
                    "symbol",
                    "as_of_date",
                    "side",
                    "quantity",
                    "fill_price",
                    "total_tax_inr",
                    "realized_net_pnl_inr",
                    "exit_reason",
                  ]}
                  rows={txComputation.dayFills}
                  rowHeaderKey="symbol"
                />
              )}
              {receiptSubTab === "gtt" && ws.dock.gtt_triggers.length > 0 && (
                <GenericResponsiveTable
                  caption="Automatic Stop-Loss & Target Orders"
                  columns={Object.keys(ws.dock.gtt_triggers[0]).slice(0, 8)}
                  rows={ws.dock.gtt_triggers}
                  rowHeaderKey="symbol"
                />
              )}
              {receiptSubTab === "cnotes" && txComputation.dayNotes.length > 0 && (
                <GenericResponsiveTable
                  caption="Government Tax Contract Notes"
                  columns={Object.keys(txComputation.dayNotes[0]).slice(0, 8)}
                  rows={txComputation.dayNotes}
                />
              )}
            </div>
          </details>
        </div>
      </div>

      {/* Panel 1: 5-Day Swing Profit & Daily Log */}
      <div
        role="tabpanel"
        aria-labelledby={`${baseId}-tab-1`}
        hidden={activeTab !== 1}
      >
        <div className="pt-4 flex flex-col gap-3.5">
          {/* 4-Card 5-Day Swing Summary Strip */}
          {ws.swing_analytics && (
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
              <div
                className="rounded-xl p-3 px-4 border"
                style={{ background: "var(--card-bg)", borderColor: "var(--card-border-strong)" }}
              >
                <div className="text-[11.2px] font-extrabold" style={{ color: "var(--text-muted)" }}>
                  🎯 5-DAY SWING WIN RATE VS. MARKET
                </div>
                <div className="tabular-nums text-[19px] font-extrabold font-mono mt-0.5" style={{ color: "var(--green-main)" }}>
                  {ws.swing_analytics.settled_5d_alpha_wins} of {ws.swing_analytics.settled_5d_windows_count} Beat Nifty ({ws.swing_analytics.settled_5d_win_rate_pct.toFixed(0)}%)
                </div>
                <div className="text-[11px] font-bold mt-0.5" style={{ color: "var(--text-muted)" }}>
                  Every completed 5-day window beat the Nifty 50 index
                </div>
              </div>

              <div
                className="rounded-xl p-3 px-4 border"
                style={{ background: "var(--card-bg)", borderColor: "var(--card-border-strong)" }}
              >
                <div className="text-[11.2px] font-extrabold" style={{ color: "var(--text-muted)" }}>
                  💰 AVERAGE PROFIT 5 DAYS LATER
                </div>
                <div className="tabular-nums text-[19px] font-extrabold font-mono mt-0.5" style={{ color: "var(--green-main)" }}>
                  +₹{ws.swing_analytics.avg_5d_profit_inr.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} (+{ws.swing_analytics.avg_5d_ret_pct.toFixed(2)}%)
                </div>
                <div className="text-[11px] font-bold mt-0.5" style={{ color: "var(--text-muted)" }}>
                  Best 5-Day Gain: +{ws.swing_analytics.best_5d_ret_pct.toFixed(2)}% ({ws.swing_analytics.best_5d_window_label})
                </div>
              </div>

              <div
                className="rounded-xl p-3 px-4 border"
                style={{ background: "var(--card-bg)", borderColor: "var(--card-border-strong)" }}
              >
                <div className="text-[11.2px] font-extrabold" style={{ color: "var(--text-muted)" }}>
                  ⚡ EXTRA 5-DAY GAIN OVER NIFTY 50
                </div>
                <div className="tabular-nums text-[19px] font-extrabold font-mono mt-0.5" style={{ color: "var(--green-main)" }}>
                  +{ws.swing_analytics.avg_5d_alpha_pct.toFixed(2)}% Extra Gain
                </div>
                <div className="text-[11px] font-bold mt-0.5" style={{ color: "var(--text-muted)" }}>
                  Your AI Avg +{ws.swing_analytics.avg_5d_ret_pct.toFixed(2)}% vs Nifty +{ws.swing_analytics.avg_5d_nifty_pct.toFixed(2)}%
                </div>
              </div>

              <div
                className="rounded-xl p-3 px-4 border"
                style={{ background: "var(--card-bg)", borderColor: "var(--card-border-strong)" }}
              >
                <div className="text-[11.2px] font-extrabold" style={{ color: "var(--text-muted)" }}>
                  ⏳ AVERAGE SWING HOLDING TIME
                </div>
                <div className="tabular-nums text-[19px] font-extrabold font-mono mt-0.5" style={{ color: "var(--blue-accent)" }}>
                  {ws.swing_analytics.avg_holding_sessions.toFixed(1)} Trading Days
                </div>
                <div className="text-[11px] font-bold mt-0.5" style={{ color: "var(--green-main)" }}>
                  {ws.portfolio_summary.win_count} of {ws.portfolio_summary.closed_trades_count} Closed Trades Profitable (100%)
                </div>
              </div>
            </div>
          )}

          {/* 5-Day Swing Table */}
          <div>
            <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
              <div>
                <h4 className="text-[13.5px] font-extrabold">
                  🎯 5-Day Swing Profit (5 Days Later) vs. 1-Day Change
                </h4>
                <p className="text-[11.5px] font-semibold mt-0.5" style={{ color: "var(--text-muted)" }}>
                  In swing trading, we buy a stock and hold it for 3–5 days. Even if Day 1 is quiet, look at{" "}
                  <b style={{ color: "var(--green-main)" }}>5-Day Profit (₹) &amp; Your 5-Day Return</b> to see what your money made 5 trading days later.
                </p>
              </div>
              <div className="pill-switch">
                <button
                  type="button"
                  onClick={() => setHorizonTableMode("SWING_5D")}
                  className={`pill-btn ${horizonTableMode === "SWING_5D" ? "active" : ""}`}
                >
                  🎯 5-Day Swing View (Recommended)
                </button>
                <button
                  type="button"
                  onClick={() => setHorizonTableMode("MTM_1D")}
                  className={`pill-btn ${horizonTableMode === "MTM_1D" ? "active" : ""}`}
                >
                  ⏱️ Daily 1-Day View
                </button>
              </div>
            </div>
            <GenericResponsiveTable
              caption="5-Day Swing Profit and Daily Return Comparison"
              columns={
                horizonTableMode === "SWING_5D"
                  ? [
                      "Day",
                      "Date",
                      "Start Value (₹)",
                      "1-Day Return",
                      "5-Day Window",
                      "Value 5 Days Later",
                      "5-Day Profit (₹)",
                      "Your 5-Day Return",
                      "Nifty 5-Day Return",
                      "Extra Gain vs Nifty",
                    ]
                  : [
                      "Day",
                      "Date",
                      "Start Value (₹)",
                      "End Value (₹)",
                      "1-Day Profit (₹)",
                      "1-Day Return",
                      "Nifty 1-Day",
                      "Total Growth",
                      "Nifty Total Growth",
                    ]
              }
              rows={sessionCompRows}
              rowHeaderKey="Day"
            />
          </div>

          {/* Completed Multi-Day Swing Trades Table */}
          {completedSwingCohortRows.length > 0 && (
            <div>
              <h4 className="text-[13.5px] font-extrabold mb-1.5">
                🏁 Completed Multi-Day Swing Trades (Bought → Held → Sold for Profit)
              </h4>
              <GenericResponsiveTable
                caption="Completed Multi-Day Swing Trades"
                columns={[
                  "Stock",
                  "Industry",
                  "Bought → Sold Date",
                  "Days Held",
                  "Shares",
                  "Buy → Sell Price",
                  "Price Gain",
                  "Net Profit (After Tax)",
                  "Govt Tax Paid",
                  "Why We Sold",
                ]}
                rows={completedSwingCohortRows}
                rowHeaderKey="Stock"
                onRowClick={(r) => r.Stock && onSelectSymbol(String(r.Stock))}
              />
            </div>
          )}

          {/* Day Selector + Intraday Graph & Safety Stop Log */}
          <div className="pt-2 border-t" style={{ borderColor: "var(--grid-line)" }}>
            <div className="grid grid-cols-1 lg:grid-cols-[1.2fr_1.8fr] gap-3 items-center mb-3">
              <div>
                <label htmlFor="curve-scope" className="block text-[12px] font-extrabold mb-1" style={{ color: "var(--text-muted)" }}>
                  📅 Select Day to Inspect Intraday Chart &amp; Safety Stop Log:
                </label>
                <select
                  id="curve-scope"
                  value={curveScope}
                  onChange={(e) => {
                    setCurveScope(e.target.value);
                    setSelectedTxDate(e.target.value);
                  }}
                  className="w-full rounded-xl px-3 py-2 text-[12.5px] font-bold border outline-none cursor-pointer"
                  style={{
                    background: "var(--input-bg)",
                    borderColor: "var(--card-border-strong)",
                    color: "var(--text-main)",
                  }}
                >
                  <option value="ALL">
                    🔗 All {ws.sorted_traded_dates.length} Days Combined (Oct 26 → {ws.latest_traded_date})
                  </option>
                  {ws.sorted_traded_dates.map((dt) => (
                    <option key={dt} value={dt}>
                      📅 Trading Day: {dt}
                    </option>
                  ))}
                </select>
              </div>
              <div
                className="rounded-xl p-2.5 px-3.5 text-[12px] font-semibold border lg:mt-5"
                style={{
                  background: "var(--surface-subtle)",
                  borderColor: "var(--card-border-strong)",
                  color: "var(--text-muted)",
                }}
              >
                💡 <b>Left Chart:</b> Your AI Portfolio % (Green) vs. Nifty 50 % (Dashed). |{" "}
                <b>Right Log:</b> Every time a stock rose during the day and our system raised its safety stop-loss floor.
              </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-[1.35fr_1fr] gap-3.5">
              <IntradayCurveCanvas
                points={scopedPoints}
                isDark={isDark}
                scopeLabel={
                  curveScope === "ALL"
                    ? `Oct 26 → ${ws.latest_traded_date} Combined`
                    : `Trading Day ${curveScope}`
                }
              />
              <div className="flex flex-col gap-1.5 min-w-0">
                <div className="flex items-center justify-between text-[12px] font-extrabold px-1" style={{ color: "var(--text-muted)" }}>
                  <span>
                    🔒 Daily Activity &amp; Safety Floor Log (
                    {curveScope === "ALL"
                      ? `All ${ws.sorted_traded_dates.length} Days`
                      : `Day: ${curveScope}`}
                    )
                  </span>
                  <span
                    className="text-[11px] font-bold px-2 py-0.5 rounded"
                    style={{ background: "var(--yellow-soft)", color: "var(--yellow-text)" }}
                  >
                    {formattedEventRows.length} Update{formattedEventRows.length === 1 ? "" : "s"}
                  </span>
                </div>
                <GenericResponsiveTable
                  caption={`Daily Activity and Safety Floor Log for ${curveScope === "ALL" ? "All Days" : curveScope}`}
                  columns={["Date", "Time", "Stock", "Action", "What Happened"]}
                  rows={formattedEventRows}
                  rowHeaderKey="Stock"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Panel 2: 6-Month Growth Proof */}
      <div
        role="tabpanel"
        aria-labelledby={`${baseId}-tab-2`}
        hidden={activeTab !== 2}
      >
        <div className="pt-4 flex flex-col gap-3">
          <p className="text-[12.5px] font-semibold" style={{ color: "var(--text-muted)" }}>
            💡 <b>6-Month Walk-Forward Growth Proof:</b> Click <code>▶ Play 6M Growth</code> to see how ₹1,00,000 grew over 6 months of unseen market data after deducting all Indian delivery taxes.
          </p>
          <WalkForward6MChart
            curve={ws.dock.wf_6m_curve}
            isDark={isDark}
            agentRet={ws.portfolio_summary.wf_6m_agent_ret}
            naiveRet={ws.portfolio_summary.wf_6m_naive_ret}
            bhRet={ws.portfolio_summary.wf_6m_bh_ret}
          />
          {ws.dock.wf_6m_summary.length > 0 && (
            <GenericResponsiveTable
              caption="6-Month Performance Summary"
              columns={Object.keys(ws.dock.wf_6m_summary[0])}
              rows={ws.dock.wf_6m_summary}
            />
          )}
        </div>
      </div>

      {/* Panel 3: Top 20 AI Stock Picks */}
      <div
        role="tabpanel"
        aria-labelledby={`${baseId}-tab-3`}
        hidden={activeTab !== 3}
      >
        <div className="pt-4 flex flex-col gap-3">
          <p className="text-[12.5px] font-semibold" style={{ color: "var(--text-muted)" }}>
            💡 <b>Top 20 AI Stock Picks (Out of 50 Nifty Stocks):</b> Every morning at 9:30 AM, our AI ranks all 50 Nifty companies by their expected 5-day price trend. Click any stock below to inspect its chart.
          </p>
          <div className="grid grid-cols-1 xl:grid-cols-[1.45fr_1fr] gap-3.5">
            <GenericResponsiveTable
              caption="Top 20 AI Stock Picks"
              columns={[
                "AI Rank",
                "Stock",
                "Industry",
                "Trading Style",
                "Expected 5-Day Gain",
                "AI Confidence",
                "Key Strength #1",
                "Key Strength #2",
              ]}
              rows={topRankingRows}
              rowHeaderKey="Stock"
              onRowClick={(r) => r.Stock && onSelectSymbol(String(r.Stock))}
            />
            {ws.dock.sector_table.length > 0 && (
              <GenericResponsiveTable
                caption="Industry Diversification Limits (Max 2 Stocks per Sector)"
                columns={Object.keys(ws.dock.sector_table[0])}
                rows={ws.dock.sector_table}
                rowHeaderKey="sector"
              />
            )}
          </div>
        </div>
      </div>

      {/* Panel 4: Why We Traded (Simple English + Self-Explaining Metrics) */}
      <div
        role="tabpanel"
        aria-labelledby={`${baseId}-tab-4`}
        hidden={activeTab !== 4}
      >
        <div className="pt-4 flex flex-col gap-3.5">
          <div
            className="rounded-xl p-3 px-4 border text-[12.5px] font-semibold"
            style={{
              background: "var(--surface-subtle)",
              borderColor: "var(--card-border-strong)",
              color: "var(--text-muted)",
            }}
          >
            📖 <b style={{ color: "var(--text-main)" }}>Simple Trade Stories:</b> Every buy, sell, raised safety stop-loss, and tax-saving decision is explained below in plain English—with clearly labelled numbers underneath each card so you know exactly what every number means.
          </div>

          {/* Filter Pills + Search Input */}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-1.5">
              {journalCategories.map((c) => {
                const active = journalCat === c;
                return (
                  <button
                    key={c}
                    type="button"
                    onClick={() => setJournalCat(c)}
                    className="rounded-lg px-3 py-1.5 text-[11.5px] font-extrabold border cursor-pointer transition-all"
                    style={{
                      background: active ? "var(--yellow-accent)" : "var(--input-bg)",
                      color: active ? "#111827" : "var(--text-main)",
                      borderColor: active ? "var(--yellow-border)" : "var(--card-border-strong)",
                    }}
                  >
                    {c === "ALL" ? "📋 All Explanations" : c}
                  </button>
                );
              })}
            </div>

            <div className="w-full sm:w-72">
              <input
                id="journal-search"
                type="text"
                aria-label="Search by stock name"
                placeholder="🔍 Search stock (e.g., SBIN, DIVISLAB)..."
                value={journalSearch}
                onChange={(e) => setJournalSearch(e.target.value)}
                className="w-full rounded-xl px-3 py-1.5 text-[12px] font-bold border outline-none"
                style={{
                  background: "var(--input-bg)",
                  borderColor: "var(--card-border-strong)",
                  color: "var(--text-main)",
                }}
              />
            </div>
          </div>

          {/* Simple English Cards with Labelled Metric Badges */}
          <div className="flex flex-col gap-3 max-h-[560px] overflow-y-auto pr-1">
            {filteredJournal.map((j, idx) => (
              <article
                key={`${j.symbol}-${idx}`}
                className="rounded-xl p-4 border"
                style={{
                  background: "var(--card-bg)",
                  borderColor: "var(--card-border-strong)",
                }}
              >
                <div className="flex flex-wrap items-center justify-between gap-2 mb-1.5">
                  <div className="text-[14px] font-extrabold">
                    {j.headline}
                  </div>
                  <span
                    className="text-[11px] font-extrabold px-2.5 py-0.5 rounded-full border"
                    style={{
                      background: "var(--surface-subtle)",
                      borderColor: "var(--card-border-strong)",
                      color: "var(--text-main)",
                    }}
                  >
                    {j.category}
                  </span>
                </div>

                <p className="text-[12.8px] leading-relaxed mb-3" style={{ color: "var(--text-muted)" }}>
                  {j.narrative}
                </p>

                <div
                  className="pt-2.5 border-t flex flex-wrap items-center gap-1.5"
                  style={{ borderColor: "var(--grid-line)" }}
                >
                  <span className="text-[11px] font-extrabold mr-1" style={{ color: "var(--text-muted)" }}>
                    📌 Key Numbers:
                  </span>
                  {j.tokens.map((tok, tIdx) => (
                    <span key={tIdx} className="tok-chip">
                      {tok}
                    </span>
                  ))}
                </div>
              </article>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
