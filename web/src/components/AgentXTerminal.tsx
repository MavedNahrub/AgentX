"use client";

import React, { useCallback, useEffect, useState } from "react";
import TradingViewChart from "@/components/TradingViewChart";
import JudgeTourAndKpis from "@/components/JudgeTourAndKpis";
import BrokerDock from "@/components/BrokerDock";
import type {
  LiveSessionState,
  LiveTickMessage,
  OhlcvBar,
  WorkstationPayload,
} from "@/types/agentx";

export default function AgentXTerminal() {
  const [ws, setWs] = useState<WorkstationPayload | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [activeSym, setActiveSym] = useState<string>("SBIN");
  const [timeframe, setTimeframe] = useState<"1m" | "5m" | "15m" | "1D">("1D");
  const [activeBars, setActiveBars] = useState<OhlcvBar[]>([]);
  const [chartMode, setChartMode] = useState<"candles" | "overview">("candles");
  const [showOverlays, setShowOverlays] = useState<boolean>(true);
  const [liveSession, setLiveSession] = useState<LiveSessionState | null>(null);

  const [fillFilter, setFillFilter] = useState<"ALL" | "BUY" | "SELL">("ALL");
  const [mktFilter, setMktFilter] = useState<"ALL" | "Held" | "Sold" | "Watchlist">("ALL");
  const [centerTab, setCenterTab] = useState<"vault" | "shap" | "audit">("vault");
  const [rightBotMode, setRightBotMode] = useState<"SESSIONS" | "SECTORS">("SESSIONS");
  const [activeNav, setActiveNav] = useState<string>("candles");

  const [showTour, setShowTour] = useState<boolean>(true);
  const [showOpDrawer, setShowOpDrawer] = useState<boolean>(false);
  const [showCapitalModal, setShowCapitalModal] = useState<boolean>(false);
  const [depositAmount, setDepositAmount] = useState<number>(0);
  const [customAmountStr, setCustomAmountStr] = useState<string>("");
  const [opBusy, setOpBusy] = useState<boolean>(false);
  const [toastMsg, setToastMsg] = useState<string | null>(null);

  const [isDark, setIsDark] = useState<boolean>(false);
  const [liveTick, setLiveTick] = useState<LiveTickMessage | null>(null);

  const showToast = useCallback((msg: string) => {
    setToastMsg(msg);
  }, []);

  useEffect(() => {
    if (!toastMsg) return;
    const t = setTimeout(() => setToastMsg(null), 3400);
    return () => clearTimeout(t);
  }, [toastMsg]);

  // Initialize & sync color-scheme per modern-web-guidance `dark-mode-toggle`
  useEffect(() => {
    const computeRenderedDark = () => {
      const meta = document.querySelector('meta[name="color-scheme"]') as HTMLMetaElement | null;
      const content = meta?.content || "light dark";
      if (content === "dark") return true;
      if (content === "light") return false;
      return window.matchMedia("(prefers-color-scheme: dark)").matches;
    };
    setIsDark(computeRenderedDark());

    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onMq = () => setIsDark(computeRenderedDark());
    const onStorage = (e: StorageEvent) => {
      if (e.key === "color-scheme") {
        const meta = document.querySelector('meta[name="color-scheme"]') as HTMLMetaElement | null;
        if (meta) meta.content = e.newValue ?? "light dark";
        setIsDark(computeRenderedDark());
      }
    };
    mq.addEventListener("change", onMq);
    window.addEventListener("storage", onStorage);
    return () => {
      mq.removeEventListener("change", onMq);
      window.removeEventListener("storage", onStorage);
    };
  }, []);

  // Two-state override logic per `dark-mode-toggle`: check divergence against system only at storage time
  const applyColorSchemeTarget = (target: "light" | "dark") => {
    const sysPrefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const sysScheme = sysPrefersDark ? "dark" : "light";
    const meta = document.querySelector('meta[name="color-scheme"]') as HTMLMetaElement | null;
    if (target === sysScheme) {
      try {
        localStorage.removeItem("color-scheme");
      } catch {}
      if (meta) meta.content = "light dark";
    } else {
      try {
        localStorage.setItem("color-scheme", target);
      } catch {}
      if (meta) meta.content = target;
    }
    setIsDark(target === "dark");
  };

  // Load initial workstation payload
  const fetchWorkstation = useCallback(async (forceRefresh = false) => {
    try {
      setErrorMsg(null);
      const res = await fetch(`/api/workstation${forceRefresh ? "?refresh=true" : ""}`);
      if (!res.ok) {
        throw new Error(`API returned HTTP ${res.status}`);
      }
      const data: WorkstationPayload = await res.json();
      setWs(data);
      const firstSym =
        data.market_list.length > 0
          ? data.market_list[0].sym
          : Object.keys(data.stock_bundles)[0];
      setActiveSym((prev) => (data.stock_bundles[prev] ? prev : firstSym));
      const initialBundle = data.stock_bundles[firstSym];
      if (initialBundle) {
        setActiveBars(initialBundle.bars);
      }
    } catch (err) {
      setErrorMsg(
        err instanceof Error
          ? err.message
          : "Unable to connect to FastAPI server on port 8000."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWorkstation(false);
  }, [fetchWorkstation]);

  // Fetch bars dynamically when symbol or timeframe changes
  useEffect(() => {
    if (!ws) return;
    const bundle = ws.stock_bundles[activeSym];
    if (timeframe === "15m" && bundle?.bars?.length) {
      setActiveBars(bundle.bars);
      return;
    }
    let cancelled = false;
    const sessionParam = bundle?.session_date ? `&session_date=${encodeURIComponent(bundle.session_date)}` : "";
    fetch(`/api/bars/${encodeURIComponent(activeSym)}?timeframe=${timeframe}${sessionParam}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!cancelled && d?.bars?.length) {
          setActiveBars(d.bars);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [activeSym, timeframe, ws]);

  // Connect to FastAPI WebSocket `/ws/live` for real-time tick streaming
  useEffect(() => {
    if (!ws) return;
    const apiEnv = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
    const wsUrl = apiEnv.replace(/^http/, "ws") + "/ws/live";
    let socket: WebSocket | null = null;
    try {
      socket = new WebSocket(wsUrl);
      socket.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data) as LiveTickMessage;
          if (msg.type === "LIVE_TICK") {
            setLiveTick(msg);
          }
        } catch {}
      };
    } catch {}
    return () => {
      socket?.close();
    };
  }, [ws]);

  const handleSelectSymbol = useCallback(
    (sym: string) => {
      if (!ws) return;
      if (ws.stock_bundles[sym]) {
        setActiveSym(sym);
        setChartMode("candles");
      } else {
        // Symbol from top rankings not yet in bundles: fetch bars on the fly
        setActiveSym(sym);
        setChartMode("candles");
      }
    },
    [ws]
  );

  const handleKillSwitch = async (mode: "RUNNING" | "HALTED_KILL_SWITCH") => {
    setOpBusy(true);
    try {
      const res = await fetch("/api/kill-switch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode,
          reason:
            mode === "RUNNING"
              ? "Operator resumed autonomous execution via Next.js Terminal"
              : "Manual emergency kill switch engaged via Next.js Terminal",
        }),
      });
      if (res.ok) {
        await fetchWorkstation(false);
        showToast(
          mode === "RUNNING"
            ? "🟢 Autonomous Execution RESUMED in Risk Vault"
            : "🔴 Emergency Kill Switch HALTED all new BUY orders"
        );
      }
    } finally {
      setOpBusy(false);
    }
  };

  const fetchLiveStatus = useCallback(async (): Promise<LiveSessionState | null> => {
    try {
      const res = await fetch("/api/session/live-status", { cache: "no-store" });
      if (res.ok) {
        const st: LiveSessionState = await res.json();
        setLiveSession(st);
        return st;
      }
    } catch {
      // ignore transient polling error
    }
    return null;
  }, []);

  useEffect(() => {
    void fetchLiveStatus();
    let lastSig = "";
    const id = setInterval(async () => {
      const st = await fetchLiveStatus();
      if (!st) return;
      const sig = `${st.status}:${st.current_minute}:${st.date ?? ""}`;
      const isWorking = ["MORNING_SCOUT", "CATCHING_UP", "LIVE", "SETTLING", "SETTLED"].includes(st.status);
      if (isWorking && sig !== lastSig) {
        lastSig = sig;
        void fetchWorkstation(false);
      }
    }, 3000);
    return () => clearInterval(id);
  }, [fetchLiveStatus, fetchWorkstation]);

  const handleGoLiveClick = () => {
    setDepositAmount(0);
    setCustomAmountStr("");
    setShowCapitalModal(true);
  };

  const executeGoLive = async (topUpInr: number) => {
    setShowCapitalModal(false);
    setOpBusy(true);
    if (topUpInr > 0) {
      showToast(`💳 Depositing ₹${topUpInr.toLocaleString("en-IN")} & arming live session...`);
    } else {
      showToast("🟢 Arming live NSE trading session (09:15 AM - 03:30 PM IST)...");
    }
    try {
      const res = await fetch("/api/session/go-live", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ additional_capital_inr: topUpInr }),
      });
      if (res.ok) {
        const st: LiveSessionState = await res.json();
        setLiveSession(st);
        showToast(
          topUpInr > 0
            ? `✅ Deposited ₹${topUpInr.toLocaleString("en-IN")}! Live session active.`
            : (st.message || "Live trading session active!")
        );
        void fetchWorkstation(false);
      } else {
        showToast("⚠️ Could not start live trading session.");
      }
    } finally {
      setOpBusy(false);
    }
  };

  const handleGoLive = () => {
    handleGoLiveClick();
  };

  const handleStopLive = async () => {
    setOpBusy(true);
    try {
      const res = await fetch("/api/session/stop-live", { method: "POST" });
      if (res.ok) {
        const st: LiveSessionState = await res.json();
        setLiveSession(st);
        showToast(st.message || "Live session paused cleanly.");
        void fetchWorkstation(false);
      }
    } finally {
      setOpBusy(false);
    }
  };

  if (loading) {
    return (
      <main className="min-h-screen flex items-center justify-center p-6">
        <div className="ax-card p-8 max-w-md w-full text-center">
          <div className="text-3xl mb-3">🐷</div>
          <h1 className="text-[18px] font-extrabold mb-1.5">
            Loading AgentX v2.0 Quant Terminal...
          </h1>
          <p className="text-[12.5px] font-semibold" style={{ color: "var(--text-muted)" }}>
            Hydrating Dual-Head Future Swing Ranker, Cryptographic Risk Vault &amp; 375m NSE Broker Ledger...
          </p>
        </div>
      </main>
    );
  }

  if (errorMsg || !ws) {
    return (
      <main className="min-h-screen flex items-center justify-center p-6">
        <div className="ax-card p-7 max-w-lg w-full">
          <h1 className="text-[18px] font-extrabold mb-2" style={{ color: "var(--red-main)" }}>
            ⚠️ FastAPI Quant Server Not Reachable
          </h1>
          <p className="text-[13px] mb-4" style={{ color: "var(--text-muted)" }}>
            {errorMsg || "Start the Python FastAPI server on port 8000 first."}
          </p>
          <pre
            className="p-3 rounded-xl text-[11.5px] font-mono overflow-x-auto mb-4"
            style={{ background: "var(--input-bg)" }}
          >
            {`$env:PYTHONPATH="."; .\\.venv\\Scripts\\python.exe -m uvicorn src.api_server:app --port 8000`}
          </pre>
          <button
            type="button"
            onClick={() => {
              setLoading(true);
              fetchWorkstation(false);
            }}
            className="pill-btn active px-4 py-2"
          >
            🔄 Retry Connection
          </button>
        </div>
      </main>
    );
  }

  const summary = ws.portfolio_summary;
  const bundle =
    ws.stock_bundles[activeSym] ||
    ws.stock_bundles[Object.keys(ws.stock_bundles)[0]];
  const isVaultRunning =
    (liveTick?.kill_switch_mode || ws.operator_state.kill_switch_mode) === "RUNNING";

  const filteredFills = ws.last_trades.filter(
    (tr) => fillFilter === "ALL" || tr.side === fillFilter
  );
  const filteredMarket = ws.market_list.filter(
    (m) => mktFilter === "ALL" || m.status.includes(mktFilter)
  );

  // Stop-to-target progress & conviction dot calculations
  const riskPs = Math.max(0.1, bundle.buy_px - bundle.stop_px);
  const rewPs = Math.max(0.1, bundle.target_px - bundle.buy_px);
  const rrRatio = (rewPs / riskPs).toFixed(1);
  const priceSpan = Math.max(0.01, bundle.target_px - bundle.stop_px);
  const stopToTargetProg = Math.max(
    0,
    Math.min(1, (bundle.cur_px - bundle.stop_px) / priceSpan)
  );
  const convFrac = Math.max(0.25, bundle.conv_pct / 100.0);
  const notionalInr = Math.round(bundle.cur_px * bundle.qty);

  return (
    <main className="min-h-screen p-2.5 sm:p-3.5 max-w-[1920px] mx-auto">
      {/* Main Reference Workstation Shell (Left Vertical Nav Rail + 3-Row Right Workspace) */}
      <div className="grid grid-cols-1 lg:grid-cols-[54px_minmax(0,1fr)] gap-3.5 items-stretch">
        {/* 1. LEFT VERTICAL NAVIGATION RAIL */}
        <nav
          aria-label="Workstation Module Shortcuts"
          className="ax-card flex lg:flex-col items-center justify-between px-3 py-2.5 lg:px-0 lg:py-3.5"
        >
          <div className="flex lg:flex-col items-center gap-2.5 flex-wrap">
            <div
              className="w-[34px] h-[34px] rounded-[10px] flex items-center justify-center font-extrabold text-[16px] lg:mb-1.5"
              style={{ background: "#1E1B4B", color: "#FFFFFF" }}
              title="AgentX v2.0 Autonomous NSE Swing Fund"
            >
              🐷
            </div>
            {[
              { id: "candles", icon: "🏠", title: "Live Stock Candlesticks & GTT Brackets" },
              {
                id: "overview",
                icon: "📊",
                title: `${summary.sessions_count}-Session Portfolio vs. Nifty 50 (+${summary.net_pct.toFixed(2)}%)`,
              },
              { id: "vault", icon: "🛡️", title: "GTT OCO Safety Vault & Elastic Sizing" },
              { id: "shap", icon: "🧠", title: "Dual-Head Future Ranker & TreeSHAP" },
              { id: "audit", icon: "🔒", title: "Zero-Hallucination Audit Narrative" },
              { id: "sessions", icon: "📅", title: `${summary.sessions_count}-Session Compounding Ledger` },
              { id: "sectors", icon: "🏛️", title: "NSE Sector Diversification Guardrail (≤2 Slots)" },
              { id: "tour", icon: "🎓", title: "Toggle Interactive Architecture Tour" },
              { id: "operator", icon: "⚡", title: "Live Session Controller & Emergency Kill Switch" },
            ].map((item) => {
              const active = activeNav === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  title={item.title}
                  aria-label={item.title}
                  onClick={() => {
                    setActiveNav(item.id);
                    if (item.id === "candles") {
                      setChartMode("candles");
                      setShowOverlays(true);
                    } else if (item.id === "overview") {
                      setChartMode("overview");
                    } else if (item.id === "vault") {
                      setCenterTab("vault");
                    } else if (item.id === "shap") {
                      setCenterTab("shap");
                      showToast(
                        `🧠 NSE:${bundle.sym} TreeSHAP — #1: ${bundle.shap1} | #2: ${bundle.shap2}`
                      );
                    } else if (item.id === "audit") {
                      setCenterTab("audit");
                    } else if (item.id === "sessions") {
                      setRightBotMode("SESSIONS");
                    } else if (item.id === "sectors") {
                      setRightBotMode("SECTORS");
                    } else if (item.id === "tour") {
                      setShowTour((t) => !t);
                    } else if (item.id === "operator") {
                      setShowOpDrawer((o) => !o);
                    }
                  }}
                  className="w-[34px] h-[34px] rounded-[10px] flex items-center justify-center text-[15px] cursor-pointer border-none transition-all"
                  style={{
                    background: active ? "#111827" : "transparent",
                    color: active ? "#FFFFFF" : "var(--text-muted)",
                    boxShadow: active ? "0 4px 12px rgba(17, 24, 39, 0.22)" : "none",
                  }}
                >
                  {item.icon}
                </button>
              );
            })}
          </div>
          <button
            type="button"
            onClick={() => setShowOpDrawer((o) => !o)}
            title={`Cryptographic Risk Vault: ${isVaultRunning ? "RUNNING" : "HALTED"}`}
            className="w-[34px] h-[34px] rounded-[10px] flex items-center justify-center text-[15px] cursor-pointer border-none bg-transparent"
          >
            {isVaultRunning ? "🟢" : "🔴"}
          </button>
        </nav>

        {/* 2. RIGHT MAIN 3-ROW WORKSPACE */}
        <div className="flex flex-col gap-3.5 min-w-0">
          {/* Operator Drawer (Replaces Streamlit Sidebar cleanly inline when toggled) */}
          {showOpDrawer && (
            <div
              className="ax-card p-3.5 px-4 flex flex-wrap items-center justify-between gap-3 border-2"
              style={{ borderColor: "var(--yellow-border)" }}
            >
              <div className="flex flex-wrap items-center gap-4 text-[12px]">
                <span className="font-extrabold text-[13.5px]">⚡ Live NSE Session &amp; Risk Vault</span>
                <span>
                  Engine:{" "}
                  <b>{(liveSession ?? ws.operator_state.live_session)?.status ?? "IDLE"}</b>
                </span>
                <span>
                  IST Clock:{" "}
                  <b>{(liveSession ?? ws.operator_state.live_session)?.market_clock_ist ?? "15:30:00"} IST</b>
                </span>
                <span>
                  Completed: <b>{ws.session_ledger.length} Sessions</b>
                </span>
                <span>
                  Kill Switch:{" "}
                  <b style={{ color: isVaultRunning ? "var(--green-main)" : "var(--red-main)" }}>
                    {isVaultRunning ? "🟢 RUNNING" : "🔴 HALTED"}
                  </b>
                </span>
                {(liveSession ?? ws.operator_state.live_session)?.message && (
                  <span style={{ color: "var(--text-muted)" }}>
                    {(liveSession ?? ws.operator_state.live_session)?.message}
                  </span>
                )}
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {["WAITING_FOR_0915_OPEN", "STARTING", "MORNING_SCOUT", "CATCHING_UP", "LIVE", "SETTLING"].includes(
                  (liveSession ?? ws.operator_state.live_session)?.status ?? "IDLE"
                ) ? (
                  <button
                    type="button"
                    disabled={opBusy}
                    onClick={handleStopLive}
                    className="rounded-lg px-3 py-1.5 text-[11.5px] font-extrabold text-white cursor-pointer border-none"
                    style={{ background: "var(--red-btn)" }}
                  >
                    ⏹ End Live Session
                  </button>
                ) : (
                  <button
                    type="button"
                    disabled={opBusy}
                    onClick={handleGoLive}
                    className="rounded-lg px-3 py-1.5 text-[11.5px] font-extrabold text-white cursor-pointer border-none"
                    style={{ background: "var(--green-btn)" }}
                  >
                    {opBusy ? "⏳ Starting..." : "🟢 Start Live Trade (09:15–15:30)"}
                  </button>
                )}
                <button
                  type="button"
                  disabled={opBusy}
                  onClick={() => handleKillSwitch("HALTED_KILL_SWITCH")}
                  className="rounded-lg px-3 py-1.5 text-[11.5px] font-extrabold text-white cursor-pointer border-none"
                  style={{ background: "var(--red-btn)" }}
                >
                  🔴 HALT VAULT
                </button>
                <button
                  type="button"
                  disabled={opBusy}
                  onClick={() => handleKillSwitch("RUNNING")}
                  className="rounded-lg px-3 py-1.5 text-[11.5px] font-extrabold text-white cursor-pointer border-none"
                  style={{ background: "var(--green-btn)" }}
                >
                  🟢 RESUME VAULT
                </button>
                <button
                  type="button"
                  onClick={() => fetchWorkstation(true)}
                  className="pill-btn"
                  style={{ background: "var(--input-bg)" }}
                >
                  🔄 Refresh Cache
                </button>
              </div>
            </div>
          )}

          {/* TOP HEADER BAR (Streamlined — Zero Duplicate Metric Clutter) */}
          <header className="ax-card flex flex-wrap items-center justify-between gap-3 px-4 py-2.5 min-h-[54px]">
            <div className="flex flex-wrap items-center gap-3">
              <div className="pill-switch">
                <button
                  type="button"
                  onClick={() => {
                    setChartMode("candles");
                    setShowOverlays(true);
                  }}
                  className={`pill-btn ${chartMode === "candles" ? "active" : ""}`}
                >
                  📊 Stock Price Chart
                </button>
                <button
                  type="button"
                  onClick={() => setChartMode("overview")}
                  className={`pill-btn ${chartMode === "overview" ? "active" : ""}`}
                >
                  📈 {summary.sessions_count}-Day Portfolio Growth (+{summary.net_pct.toFixed(2)}%)
                </button>
              </div>

              <div className="flex items-center gap-2 pl-1">
                <span
                  className="w-7 h-7 rounded-full inline-flex items-center justify-center font-extrabold text-[11.5px] text-white"
                  style={{ background: "#F59E0B" }}
                >
                  {bundle.sym.slice(0, 2)}
                </span>
                <select
                  aria-label="Select NSE Symbol"
                  value={bundle.sym}
                  onChange={(e) => handleSelectSymbol(e.target.value)}
                  className="bg-transparent border-none font-extrabold text-[15px] cursor-pointer outline-none"
                  style={{ color: "var(--text-main)" }}
                >
                  {Object.keys(ws.stock_bundles).map((s) => (
                    <option key={s} value={s} style={{ background: "var(--card-bg)" }}>
                      NSE:{s}
                    </option>
                  ))}
                </select>
              </div>

              <div className="tabular-nums flex items-center gap-2">
                <span
                  className="text-[15px] font-extrabold font-mono"
                  style={{
                    color: bundle.move_pct >= 0 ? "var(--green-main)" : "var(--red-main)",
                  }}
                >
                  ₹{bundle.cur_px.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>
                <span
                  className="text-[11.5px] font-bold px-2 py-0.5 rounded-md"
                  style={{
                    background: "var(--input-bg)",
                    color: bundle.move_pct >= 0 ? "var(--green-main)" : "var(--red-main)",
                  }}
                >
                  {bundle.move_pct >= 0 ? "+" : ""}
                  {bundle.move_pct.toFixed(2)}% • {bundle.status}
                </span>
              </div>

              <div
                className="hidden xl:flex items-center gap-3 pl-3 border-l text-[12px] font-bold tabular-nums"
                style={{ borderColor: "var(--grid-line)" }}
              >
                <span>
                  💰 Total Value:{" "}
                  <b style={{ color: "var(--green-main)" }}>
                    ₹{summary.equity.toLocaleString("en-IN", { maximumFractionDigits: 0 })} (+{summary.net_pct.toFixed(2)}%)
                  </b>
                </span>
                <span>•</span>
                <span>
                  🎯 5-Day Swing Avg:{" "}
                  <b style={{ color: "var(--green-main)" }}>
                    +{summary.avg_5d_ret_pct.toFixed(2)}% ({summary.settled_5d_alpha_wins}/{summary.settled_5d_windows_count} Beat Market)
                  </b>
                </span>
              </div>
            </div>

            <div className="flex items-center gap-2 ml-auto">
              {/* Theme Switcher */}
              <div className="inline-flex rounded-lg p-[3px] gap-[3px]" style={{ background: "var(--input-bg)" }}>
                <button
                  type="button"
                  onClick={() => applyColorSchemeTarget("light")}
                  title="Switch to Light Theme"
                  aria-label="Switch to Light Theme"
                  className="w-[30px] h-[30px] rounded-[7px] border-none cursor-pointer inline-flex items-center justify-center text-[14px]"
                  style={{
                    background: !isDark ? "var(--yellow-accent)" : "transparent",
                    color: !isDark ? "#111827" : "var(--text-muted)",
                  }}
                >
                  ☀️
                </button>
                <button
                  type="button"
                  onClick={() => applyColorSchemeTarget("dark")}
                  title="Switch to Dark Theme"
                  aria-label="Switch to Dark Theme"
                  className="w-[30px] h-[30px] rounded-[7px] border-none cursor-pointer inline-flex items-center justify-center text-[14px]"
                  style={{
                    background: isDark ? "var(--yellow-accent)" : "transparent",
                    color: isDark ? "#111827" : "var(--text-muted)",
                  }}
                >
                  🌙
                </button>
              </div>

              {["WAITING_FOR_0915_OPEN", "STARTING", "MORNING_SCOUT", "CATCHING_UP", "LIVE", "SETTLING"].includes(
                liveSession?.status ?? ws.operator_state.live_session?.status ?? "IDLE"
              ) ? (
                <div className="flex items-center gap-1.5">
                  <span
                    className="h-[30px] px-2.5 rounded-[7px] inline-flex items-center text-[11px] font-extrabold tabular-nums"
                    title={(liveSession ?? ws.operator_state.live_session)?.message || "Live trading engine active"}
                    style={{
                      background:
                        (liveSession ?? ws.operator_state.live_session)?.status === "LIVE"
                          ? "rgba(22, 163, 74, 0.16)"
                          : "rgba(234, 179, 8, 0.22)",
                      color:
                        (liveSession ?? ws.operator_state.live_session)?.status === "LIVE"
                          ? "var(--green-main)"
                          : "#B45309",
                    }}
                  >
                    {(liveSession ?? ws.operator_state.live_session)?.status === "WAITING_FOR_0915_OPEN"
                      ? "⏳ Armed 09:15 IST"
                      : (liveSession ?? ws.operator_state.live_session)?.status === "SETTLING"
                      ? "⏳ Settling 15:30..."
                      : `🟢 LIVE • Min ${(liveSession ?? ws.operator_state.live_session)?.current_minute ?? 0}/375`}
                  </span>
                  <button
                    type="button"
                    disabled={opBusy}
                    onClick={handleStopLive}
                    title="Stop or pause live session safely (preserves today's recorded trades)"
                    className="h-[30px] px-2.5 rounded-[7px] border-none cursor-pointer text-[11px] font-extrabold text-white"
                    style={{ background: "var(--red-btn)" }}
                  >
                    ⏹ End Session
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  disabled={opBusy}
                  onClick={handleGoLive}
                  title="Arm or join today's live NSE session (09:15 AM start, auto-settles 03:30 PM IST, resumes mid-session from real candles)"
                  className="h-[30px] px-2.5 rounded-[7px] border-none cursor-pointer text-[11.5px] font-extrabold text-white"
                  style={{ background: "var(--green-btn)" }}
                >
                  {opBusy ? "⏳ Arming..." : "🟢 Start Live Trade (09:15–15:30)"}
                </button>
              )}
              <button
                type="button"
                onClick={() => setShowOpDrawer((o) => !o)}
                title="Open Session Controls"
                className="w-[30px] h-[30px] rounded-[7px] border-none cursor-pointer inline-flex items-center justify-center text-[14px]"
                style={{ background: "var(--input-bg)", color: "var(--text-main)" }}
              >
                ⚙️
              </button>
            </div>
          </header>

          {/* MIDDLE ROW: TRADINGVIEW CHART (LEFT) + RECENT ORDERS (RIGHT) */}
          <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_330px] gap-3.5 xl:h-[465px]">
            {/* TradingView Multi-Mode Chart Card */}
            <div className="ax-card p-3 px-3.5 flex flex-col overflow-hidden min-h-[420px]">
              <div
                className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b"
                style={{ borderColor: "var(--grid-line)" }}
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-[12.5px] font-extrabold">
                    NSE:{bundle.sym} ({bundle.sector})
                  </span>
                  <div className="inline-flex items-center gap-1 ml-1">
                    {(["1m", "5m", "15m", "1D"] as const).map((tf) => {
                      const active = timeframe === tf && chartMode === "candles";
                      return (
                        <button
                          key={tf}
                          type="button"
                          onClick={() => {
                            setTimeframe(tf);
                            setChartMode("candles");
                          }}
                          className="border-none text-[12px] font-bold px-2 py-1 rounded cursor-pointer"
                          style={{
                            color: active ? "var(--blue-accent)" : "var(--text-muted)",
                            background: active ? "rgba(2, 132, 199, 0.12)" : "transparent",
                          }}
                        >
                          {tf}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="pill-switch">
                  <button
                    type="button"
                    onClick={() => {
                      setChartMode("candles");
                      setShowOverlays(true);
                    }}
                    className={`pill-btn ${chartMode === "candles" && showOverlays ? "active" : ""}`}
                  >
                    🛡️ Stop &amp; Target Lines
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setChartMode("candles");
                      setShowOverlays(false);
                    }}
                    className={`pill-btn ${chartMode === "candles" && !showOverlays ? "active" : ""}`}
                  >
                    Candles Only
                  </button>
                  <button
                    type="button"
                    onClick={() => setChartMode("overview")}
                    className={`pill-btn ${chartMode === "overview" ? "active" : ""}`}
                  >
                    {summary.sessions_count}-Day Growth Curve
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-[28px_minmax(0,1fr)] gap-2 flex-1 min-h-0 pt-2">
                {/* Left Drawing Tool Strip */}
                <div
                  className="flex flex-col items-center justify-between py-1.5 border-r text-[12.5px]"
                  style={{ borderColor: "var(--grid-line)", color: "var(--text-soft)" }}
                >
                  {[
                    { icon: "✚", tip: "Crosshair" },
                    { icon: "╱", tip: "Show Fair Average Price (VWAP)", action: () => setShowOverlays(true) },
                    {
                      icon: "🧠",
                      tip: "Why AI Picked This Stock",
                      action: () =>
                        showToast(`🧠 ${bundle.sym}: #1 ${bundle.shap1} | #2 ${bundle.shap2}`),
                    },
                    {
                      icon: "🛡️",
                      tip: "Safety Stop-Loss Price",
                      action: () =>
                        showToast(`🛡️ ${bundle.sym} Safety Stop-Loss @ ₹${bundle.stop_px.toFixed(2)}`),
                    },
                    { icon: "📏", tip: `Reward vs Risk (${rrRatio}x)` },
                    { icon: "🔍", tip: "Scroll or Drag to Zoom" },
                    {
                      icon: "👁️",
                      tip: "Show/Hide Price Lines",
                      action: () => setShowOverlays((v) => !v),
                    },
                  ].map((tool) => (
                    <button
                      key={tool.tip}
                      type="button"
                      title={tool.tip}
                      onClick={tool.action}
                      className="p-1 rounded hover:opacity-80 cursor-pointer border-none bg-transparent"
                      style={{ color: "inherit" }}
                    >
                      {tool.icon}
                    </button>
                  ))}
                </div>

                {/* Official TradingView Lightweight Charts Component */}
                <TradingViewChart
                  bundle={bundle}
                  bars={activeBars}
                  timeframe={timeframe}
                  chartMode={chartMode}
                  showOverlays={showOverlays}
                  overviewPts={ws.overview_pts}
                  portfolioEquity={summary.equity}
                  sessionsCount={summary.sessions_count}
                  isDark={isDark}
                  playing={false}
                  onStopPlaying={() => {}}
                />
              </div>
            </div>

            {/* Upper-Right: Recent Buy & Sell Orders Card */}
            <aside className="ax-card contain-panel p-3.5 px-4 flex flex-col overflow-hidden min-h-[340px]">
              <div className="flex items-center justify-between gap-2 mb-2.5">
                <span className="text-[15px] font-extrabold">Recent Buy &amp; Sell Orders</span>
                <button
                  type="button"
                  onClick={() =>
                    setFillFilter((f) =>
                      f === "ALL" ? "BUY" : f === "BUY" ? "SELL" : "ALL"
                    )
                  }
                  className="rounded-[7px] px-2.5 py-1 text-[11px] font-bold border-none cursor-pointer"
                  style={{ background: "var(--input-bg)", color: "var(--text-muted)" }}
                >
                  Filter: {fillFilter === "ALL" ? "All" : fillFilter} ▾
                </button>
              </div>

              <div
                className="grid grid-cols-[1.05fr_0.95fr_1fr] gap-1.5 text-[10.6px] font-bold pb-1.5 mb-1 border-b"
                style={{ color: "var(--text-soft)", borderColor: "var(--grid-line)" }}
              >
                <span>Price (Action)</span>
                <span className="text-right">Shares (Stock)</span>
                <span className="text-right">Profit / Date</span>
              </div>

              <div className="flex flex-col gap-1 overflow-y-auto flex-1 min-h-0 pr-0.5">
                {filteredFills.map((tr, idx) => {
                  const isBuy = tr.side === "BUY";
                  const col = isBuy
                    ? "var(--green-main)"
                    : tr.pnl >= 0
                    ? "var(--green-main)"
                    : "var(--red-main)";
                  const pnlTxt = isBuy
                    ? `Bought`
                    : `${tr.pnl >= 0 ? "+₹" : "-₹"}${Math.abs(tr.pnl).toFixed(0)}`;
                  return (
                    <button
                      key={`${tr.sym}-${tr.date}-${idx}`}
                      type="button"
                      onClick={() => handleSelectSymbol(tr.sym)}
                      className="tabular-nums grid grid-cols-[1.05fr_0.95fr_1fr] gap-1.5 text-[11.2px] font-semibold items-center p-1 rounded cursor-pointer border-none bg-transparent text-left hover:opacity-85"
                      style={{ color: "var(--text-main)" }}
                    >
                      <span className="font-mono truncate" style={{ color: col }}>
                        ₹{tr.price.toFixed(1)} <small>({tr.side})</small>
                      </span>
                      <span className="text-right font-mono truncate">
                        {tr.qty} Sh ({tr.sym})
                      </span>
                      <span className="text-right text-[10.8px] truncate" style={{ color: col }}>
                        {pnlTxt} • {tr.date}
                      </span>
                    </button>
                  );
                })}
              </div>
            </aside>
          </div>

          {/* BOTTOM WORKSTATION ROW: WATCHLIST (LEFT) + STOCK PLAN & STORY (CENTER) + 12-DAY LOG (RIGHT) */}
          <div className="grid grid-cols-1 xl:grid-cols-[285px_minmax(0,1fr)_330px] gap-3.5 xl:h-[405px]">
            {/* Bottom-Left: AI Stock Watchlist */}
            <aside className="ax-card contain-panel p-3.5 px-4 flex flex-col overflow-hidden min-h-[330px]">
              <div className="flex items-center justify-between gap-2 mb-2.5">
                <span className="text-[15px] font-extrabold">AI Stock Watchlist</span>
                <button
                  type="button"
                  onClick={() =>
                    setMktFilter((f) =>
                      f === "ALL"
                        ? "Held"
                        : f === "Held"
                        ? "Sold"
                        : f === "Sold"
                        ? "Watchlist"
                        : "ALL"
                    )
                  }
                  className="rounded-[7px] px-2.5 py-1 text-[11px] font-bold border-none cursor-pointer"
                  style={{ background: "var(--input-bg)", color: "var(--text-muted)" }}
                >
                  Show: {mktFilter === "ALL" ? "All Stocks" : mktFilter} ▾
                </button>
              </div>

              <div className="flex flex-col gap-1 overflow-y-auto flex-1 min-h-0 pr-0.5">
                {filteredMarket.map((m) => {
                  const active = m.sym === bundle.sym;
                  const col = m.chg >= 0 ? "var(--green-main)" : "var(--red-main)";
                  return (
                    <button
                      key={m.sym}
                      type="button"
                      onClick={() => handleSelectSymbol(m.sym)}
                      className="tabular-nums flex items-center justify-between gap-2 p-1.5 px-2 rounded-lg cursor-pointer border-none text-left transition-colors"
                      style={{
                        background: active ? "var(--input-bg)" : "transparent",
                        color: "var(--text-main)",
                      }}
                    >
                      <div className="min-w-0">
                        <div className="text-[12.2px] font-extrabold truncate">
                          {m.sym}{" "}
                          <span
                            className="text-[10.5px] font-bold"
                            style={{
                              color: m.status.includes("Held")
                                ? "var(--green-main)"
                                : m.status.includes("Sold")
                                ? "var(--text-soft)"
                                : "var(--blue-accent)",
                            }}
                          >
                            {m.status.includes("Held")
                              ? "• 🟢 Held"
                              : m.status.includes("Sold")
                              ? "• 🏁 Sold"
                              : `• AI Rank #${m.rank}`}
                          </span>
                        </div>
                        <div
                          className="text-[10.2px] font-semibold truncate max-w-[165px]"
                          style={{ color: "var(--text-soft)" }}
                        >
                          {m.sub}
                        </div>
                      </div>
                      <div className="text-right shrink-0">
                        <div className="text-[12.5px] font-extrabold font-mono">
                          ₹{m.px.toFixed(2)}
                        </div>
                        <div className="text-[10.8px] font-bold" style={{ color: col }}>
                          {m.chg >= 0 ? "+" : ""}
                          {m.chg.toFixed(2)}%
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </aside>

            {/* Bottom-Center: Clean 3-Tab Stock Plan, AI Reasons & Simple Trade Story */}
            <section className="ax-card p-3.5 px-4.5 flex flex-col justify-between overflow-hidden">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="pill-switch">
                  <button
                    type="button"
                    onClick={() => setCenterTab("vault")}
                    className={`pill-btn ${centerTab === "vault" ? "active" : ""}`}
                  >
                    🛡️ Price &amp; Safety Plan
                  </button>
                  <button
                    type="button"
                    onClick={() => setCenterTab("shap")}
                    className={`pill-btn ${centerTab === "shap" ? "active" : ""}`}
                  >
                    🧠 Why AI Picked It
                  </button>
                  <button
                    type="button"
                    onClick={() => setCenterTab("audit")}
                    className={`pill-btn ${centerTab === "audit" ? "active" : ""}`}
                  >
                    📖 Simple Trade Story
                  </button>
                </div>
                <span className="text-[11.5px] font-bold" style={{ color: "var(--text-muted)" }}>
                  {bundle.sym} ({bundle.sector}) •{" "}
                  {bundle.status.includes("Held")
                    ? "🟢 Currently Held (Bought in Top Tier)"
                    : bundle.status.includes("Sold")
                    ? "🏁 Profit Booked & Sold"
                    : `AI Watchlist Rank #${bundle.rank} of 50`}{" "}
                  • {bundle.strategy}
                </span>
              </div>

              {centerTab === "audit" ? (
                <div
                  className="rounded-xl p-3.5 px-4 flex-1 mt-2.5 overflow-y-auto flex flex-col justify-between border"
                  style={{ background: "var(--surface-subtle)", borderColor: "var(--card-border-strong)" }}
                >
                  <div>
                    <div className="text-[13.5px] font-extrabold mb-1.5">
                      📖 {bundle.audit_headline}
                    </div>
                    <p
                      className="text-[12.5px] leading-relaxed"
                      style={{ color: "var(--text-muted)" }}
                    >
                      {bundle.audit_body}
                    </p>
                  </div>
                  <div className="mt-3 pt-2.5 border-t" style={{ borderColor: "var(--grid-line)" }}>
                    <div className="text-[11px] font-extrabold mb-1.5" style={{ color: "var(--text-muted)" }}>
                      📌 Key Trade Numbers Explained:
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {bundle.audit_tokens.map((t, i) => (
                        <span key={i} className="tok-chip">
                          {t}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ) : centerTab === "shap" ? (
                <div className="flex flex-col justify-between flex-1 mt-2.5 gap-3">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div
                      className="rounded-xl p-3 border"
                      style={{ background: "var(--input-bg)", borderColor: "var(--card-border-strong)" }}
                    >
                      <div className="text-[11px] font-bold" style={{ color: "var(--text-muted)" }}>
                        {bundle.status.includes("Held")
                          ? "🏆 RANK WHEN BOUGHT"
                          : "🏆 AI RANKING (OUT OF 50 NIFTY STOCKS)"}
                      </div>
                      <div className="text-[18px] font-extrabold mt-0.5">
                        {bundle.status.includes("Held") ? (
                          <>
                            Top Decile{" "}
                            <span className="text-[12px] font-bold" style={{ color: "var(--green-main)" }}>
                              (Ranked in Top 5 on Entry)
                            </span>
                          </>
                        ) : (
                          <>
                            Rank #{bundle.rank} of 50{" "}
                            <span className="text-[12px] font-bold" style={{ color: "var(--green-main)" }}>
                              ({bundle.conv_pct.toFixed(0)}% AI Confidence)
                            </span>
                          </>
                        )}
                      </div>
                    </div>

                    <div
                      className="rounded-xl p-3 border"
                      style={{ background: "var(--input-bg)", borderColor: "var(--card-border-strong)" }}
                    >
                      <div className="text-[11px] font-bold" style={{ color: "var(--text-muted)" }}>
                        {bundle.status.includes("Held")
                          ? "🔄 TODAY'S FRESH ENTRY RANK"
                          : "🔮 EXPECTED 5-DAY SWING GAIN"}
                      </div>
                      <div
                        className="tabular-nums text-[18px] font-extrabold font-mono mt-0.5"
                        style={{
                          color: bundle.status.includes("Held")
                            ? "var(--text-soft)"
                            : bundle.alpha_pct >= 0
                            ? "var(--green-main)"
                            : "var(--red-main)",
                        }}
                      >
                        {bundle.status.includes("Held") ? (
                          <>
                            Rank #{bundle.rank} of 50{" "}
                            <span className="text-[11px] font-sans font-bold" style={{ color: "var(--yellow-text)" }}>
                              (Cooled after {bundle.move_pct >= 0 ? "+" : ""}{bundle.move_pct.toFixed(1)}% run)
                            </span>
                          </>
                        ) : (
                          `${bundle.alpha_pct >= 0 ? "+" : ""}${bundle.alpha_pct.toFixed(2)}% over next 5 days`
                        )}
                      </div>
                    </div>

                    <div
                      className="rounded-xl p-3 border"
                      style={{ background: "var(--input-bg)", borderColor: "var(--card-border-strong)" }}
                    >
                      <div className="text-[11px] font-bold" style={{ color: "var(--text-muted)" }}>
                        ✅ MAIN REASON #1 AI LIKES THIS STOCK
                      </div>
                      <div className="text-[14px] font-extrabold mt-0.5" style={{ color: "var(--green-main)" }}>
                        {bundle.shap1}
                      </div>
                    </div>

                    <div
                      className="rounded-xl p-3 border"
                      style={{ background: "var(--input-bg)", borderColor: "var(--card-border-strong)" }}
                    >
                      <div className="text-[11px] font-bold" style={{ color: "var(--text-muted)" }}>
                        ✅ MAIN REASON #2 AI LIKES THIS STOCK
                      </div>
                      <div className="text-[14px] font-extrabold mt-0.5" style={{ color: "var(--blue-accent)" }}>
                        {bundle.shap2}
                      </div>
                    </div>
                  </div>

                  <div
                    className="rounded-xl p-2.5 px-3.5 text-[12px] font-semibold border flex flex-wrap items-center justify-between gap-2"
                    style={{ background: "var(--surface-subtle)", borderColor: "var(--card-border-strong)" }}
                  >
                    <span>
                      🧭 <b>Trading Style Used:</b> {bundle.strategy}
                    </span>
                    <span>
                      ⚖️ <b>Reward vs. Risk Ratio:</b> {rrRatio}x higher target reward than stop-loss risk
                    </span>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col justify-between flex-1 mt-2.5 gap-2.5">
                  {/* 4 Simple Price Boxes */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                    <div
                      className="rounded-xl p-2.5 px-3 border"
                      style={{ background: "var(--input-bg)", borderColor: "var(--card-border-strong)" }}
                    >
                      <div className="text-[10.8px] font-bold" style={{ color: "var(--text-muted)" }}>
                        🏷️ Buy Price
                      </div>
                      <div className="tabular-nums text-[16px] font-extrabold font-mono mt-0.5">
                        ₹{bundle.buy_px.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </div>
                      <div className="text-[10.5px] font-semibold" style={{ color: "var(--text-soft)" }}>
                        {bundle.qty} Shares (₹{notionalInr.toLocaleString("en-IN")})
                      </div>
                    </div>

                    <div
                      className="rounded-xl p-2.5 px-3 border"
                      style={{ background: "var(--input-bg)", borderColor: "var(--card-border-strong)" }}
                    >
                      <div className="text-[10.8px] font-bold" style={{ color: "var(--text-muted)" }}>
                        📈 Current Price
                      </div>
                      <div
                        className="tabular-nums text-[16px] font-extrabold font-mono mt-0.5"
                        style={{ color: bundle.move_pct >= 0 ? "var(--green-main)" : "var(--red-main)" }}
                      >
                        ₹{bundle.cur_px.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </div>
                      <div
                        className="text-[10.5px] font-bold"
                        style={{ color: bundle.move_pct >= 0 ? "var(--green-main)" : "var(--red-main)" }}
                      >
                        {bundle.move_pct >= 0 ? "+" : ""}
                        {bundle.move_pct.toFixed(2)}% ({bundle.status})
                      </div>
                    </div>

                    <div
                      className="rounded-xl p-2.5 px-3 border"
                      style={{ background: "var(--input-bg)", borderColor: "var(--card-border-strong)" }}
                    >
                      <div className="text-[10.8px] font-bold" style={{ color: "var(--text-muted)" }}>
                        🛡️ Safety Stop-Loss
                      </div>
                      <div className="tabular-nums text-[16px] font-extrabold font-mono mt-0.5" style={{ color: "var(--red-main)" }}>
                        ₹{bundle.stop_px.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </div>
                      <div className="text-[10.5px] font-semibold" style={{ color: "var(--text-soft)" }}>
                        Auto-sells if price drops here
                      </div>
                    </div>

                    <div
                      className="rounded-xl p-2.5 px-3 border"
                      style={{ background: "var(--input-bg)", borderColor: "var(--card-border-strong)" }}
                    >
                      <div className="text-[10.8px] font-bold" style={{ color: "var(--text-muted)" }}>
                        🎯 Target Sell Price
                      </div>
                      <div className="tabular-nums text-[16px] font-extrabold font-mono mt-0.5" style={{ color: "var(--green-main)" }}>
                        ₹{bundle.target_px.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </div>
                      <div className="text-[10.5px] font-semibold" style={{ color: "var(--text-soft)" }}>
                        Auto-books profit at target
                      </div>
                    </div>
                  </div>

                  {/* Visual Progress Bar from Safety Stop to Target Price */}
                  <div
                    className="rounded-xl p-2.5 px-3.5 border"
                    style={{ background: "var(--surface-subtle)", borderColor: "var(--card-border-strong)" }}
                  >
                    <div className="flex items-center justify-between text-[11.2px] font-bold mb-1.5">
                      <span style={{ color: "var(--red-main)" }}>
                        🛡️ Stop-Loss: ₹{bundle.stop_px.toFixed(2)}
                      </span>
                      <span style={{ color: "var(--text-main)" }}>
                        📍 Progress to Target: {(stopToTargetProg * 100).toFixed(0)}%
                      </span>
                      <span style={{ color: "var(--green-main)" }}>
                        🎯 Target: ₹{bundle.target_px.toFixed(2)}
                      </span>
                    </div>
                    <div
                      className="w-full h-2.5 rounded-full overflow-hidden"
                      style={{ background: "var(--input-bg)" }}
                    >
                      <div
                        className="h-full rounded-full transition-all"
                        style={{
                          width: `${Math.max(6, Math.min(100, stopToTargetProg * 100))}%`,
                          background: "linear-gradient(90deg, #F59E0B 0%, #16A34A 100%)",
                        }}
                      />
                    </div>
                  </div>

                  {/* 2 Clean Action Buttons */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                    <button
                      type="button"
                      onClick={() => {
                        setChartMode("candles");
                        setShowOverlays(true);
                        showToast(
                          `🛡️ Safety Stop-Loss for ${bundle.sym} is active at ₹${bundle.stop_px.toFixed(2)}`
                        );
                      }}
                      className="rounded-[9px] p-2.5 text-[12.5px] font-extrabold text-white border-none cursor-pointer hover:opacity-90"
                      style={{ background: "var(--green-btn)" }}
                    >
                      🛡️ Safety Stop-Loss: ₹{bundle.stop_px.toFixed(2)}
                    </button>

                    <button
                      type="button"
                      onClick={() => {
                        setChartMode("candles");
                        setShowOverlays(true);
                        showToast(
                          `🎯 Target Sell Price for ${bundle.sym} is set at ₹${bundle.target_px.toFixed(2)}`
                        );
                      }}
                      className="rounded-[9px] p-2.5 text-[12.5px] font-extrabold text-white border-none cursor-pointer hover:opacity-90"
                      style={{ background: "var(--red-btn)" }}
                    >
                      🎯 Target Sell Price: ₹{bundle.target_px.toFixed(2)}
                    </button>
                  </div>
                </div>
              )}
            </section>

            {/* Bottom-Right: 12-Day Growth Log OR Sector Limits Card */}
            <aside className="ax-card contain-panel p-3.5 px-4 flex flex-col overflow-hidden min-h-[330px]">
              <div className="flex items-center justify-between gap-2 mb-2">
                <span className="text-[15px] font-extrabold">
                  {rightBotMode === "SESSIONS"
                    ? `${ws.session_ledger.length}-Day Growth Log`
                    : "Sector Limits (Max 2)"}
                </span>
                <button
                  type="button"
                  onClick={() =>
                    setRightBotMode((m) => (m === "SESSIONS" ? "SECTORS" : "SESSIONS"))
                  }
                  className="rounded-[7px] px-2.5 py-1 text-[11px] font-bold border-none cursor-pointer"
                  style={{ background: "var(--input-bg)", color: "var(--text-muted)" }}
                >
                  {rightBotMode === "SESSIONS"
                    ? "🏛️ Sector Limits ▾"
                    : `📅 ${ws.session_ledger.length}-Day Log ▾`}
                </button>
              </div>

              {rightBotMode === "SESSIONS" ? (
                <>
                  <div
                    className="grid grid-cols-[1.05fr_0.95fr_1fr] gap-1.5 text-[10.6px] font-bold pb-1.5 mb-1 border-b"
                    style={{ color: "var(--text-soft)", borderColor: "var(--grid-line)" }}
                  >
                    <span>Day (Value)</span>
                    <span className="text-right">1-Day Change</span>
                    <span className="text-right">5-Day Swing</span>
                  </div>
                  <div
                    className="tabular-nums flex items-center justify-between py-1 px-1 mb-1 border-b text-[11px]"
                    style={{ borderColor: "var(--grid-line)" }}
                  >
                    <span className="font-extrabold font-mono" style={{ color: "var(--green-main)" }}>
                      5D Avg: +{summary.avg_5d_ret_pct.toFixed(2)}%
                    </span>
                    <span className="font-bold text-[10.4px]" style={{ color: "var(--text-soft)" }}>
                      {summary.settled_5d_alpha_wins}/{summary.settled_5d_windows_count} Beat Market • Avg Hold {summary.avg_holding_sessions.toFixed(1)}d
                    </span>
                  </div>
                  <div className="flex flex-col gap-1 overflow-y-auto flex-1 min-h-0 pr-0.5">
                    {ws.session_ledger
                      .slice()
                      .reverse()
                      .map((s) => {
                        const maxRet = Math.max(20, summary.net_pct + 1);
                        const wPct = Math.min(96, Math.max(12, (s.ret_pct / maxRet) * 95));
                        return (
                          <div
                            key={s.date}
                            className="tabular-nums relative grid grid-cols-[1.05fr_0.95fr_1fr] gap-1.5 text-[10.8px] font-semibold items-center p-1 rounded"
                          >
                            <div className="depth-bar-bid" style={{ width: `${wPct}%` }} />
                            <span className="font-mono z-10 truncate">
                              D{s.day_num} {s.date.slice(5)}{" "}
                              <small style={{ color: "var(--text-soft)" }}>
                                (₹{(s.eq / 1000).toFixed(1)}k)
                              </small>
                            </span>
                            <span className="text-right font-mono z-10 truncate">
                              <b
                                style={{
                                  color: s.sess_ret >= 0 ? "var(--green-main)" : "var(--red-main)",
                                }}
                              >
                                {s.sess_ret >= 0 ? "+" : ""}
                                {s.sess_ret.toFixed(2)}%
                              </b>
                            </span>
                            <span
                              className="text-right font-mono z-10 truncate"
                              style={{ color: "var(--green-main)" }}
                              title={s.fwd_5d_status_label}
                            >
                              <b>
                                {s.fwd_5d_ret_pct >= 0 ? "+" : ""}
                                {s.fwd_5d_ret_pct.toFixed(2)}%
                              </b>{" "}
                              <small style={{ color: "var(--text-soft)" }}>
                                {s.fwd_5d_is_settled ? "(5D)" : `(${s.fwd_5d_days_elapsed}/5D)`}
                              </small>
                            </span>
                          </div>
                        );
                      })}
                  </div>
                </>
              ) : (
                <>
                  <div
                    className="grid grid-cols-[1.05fr_0.95fr_1fr] gap-1.5 text-[10.6px] font-bold pb-1.5 mb-1 border-b"
                    style={{ color: "var(--text-soft)", borderColor: "var(--grid-line)" }}
                  >
                    <span>Industry</span>
                    <span className="text-right">Stocks Owned</span>
                    <span className="text-right">5D Outlook</span>
                  </div>
                  <div
                    className="flex items-center justify-between py-1 px-1 mb-1 border-b text-[11px]"
                    style={{ borderColor: "var(--grid-line)" }}
                  >
                    <span className="font-extrabold">Max 2 Stocks per Industry</span>
                    <span className="font-bold text-[10.4px]" style={{ color: "var(--green-main)" }}>
                      Weak Sectors → ₹0
                    </span>
                  </div>
                  <div className="flex flex-col gap-1 overflow-y-auto flex-1 min-h-0 pr-0.5">
                    {ws.sector_guardrails.map((sec) => {
                      const isPos = sec.alpha >= 0 && sec.slots > 0;
                      const wPct = Math.min(95, Math.max(15, Math.abs(sec.alpha) * 32));
                      return (
                        <div
                          key={sec.sector}
                          className="tabular-nums relative grid grid-cols-[1.05fr_0.95fr_1fr] gap-1.5 text-[11px] font-semibold items-center p-1 rounded"
                        >
                          <div
                            className={isPos ? "depth-bar-bid" : "depth-bar-ask"}
                            style={{ width: `${wPct}%` }}
                          />
                          <span className="z-10 truncate">{sec.sector}</span>
                          <span className="text-right font-mono z-10">{sec.slots} of 2</span>
                          <span
                            className="text-right font-mono z-10"
                            style={{
                              color: sec.alpha >= 0 ? "var(--green-main)" : "var(--red-main)",
                            }}
                          >
                            {sec.alpha >= 0 ? "+" : ""}
                            {sec.alpha.toFixed(2)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </>
              )}
            </aside>
          </div>
        </div>
      </div>

      {/* Interactive Judge Architecture Tour + 4 Live Ticking KPI Cards */}
      <JudgeTourAndKpis
        summary={summary}
        latestTradedDate={ws.latest_traded_date}
        tickerSeries={ws.intraday_ticker_series}
        showTour={showTour}
        liveTick={liveTick}
        liveSession={liveSession ?? ws.operator_state.live_session ?? null}
      />

      {/* Bottom 5-Tab Broker Trading & Audit Dock */}
      <BrokerDock ws={ws} isDark={isDark} onSelectSymbol={handleSelectSymbol} />

      {/* Capital Allocation & Top-Up Modal before Starting Live Session */}
      {showCapitalModal && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="capital-modal-title"
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/65 backdrop-blur-sm"
        >
          <div
            className="w-full max-w-lg rounded-2xl p-5 sm:p-6 border shadow-2xl flex flex-col gap-4 text-left transition-all"
            style={{
              background: "var(--card-bg)",
              borderColor: "var(--card-border-strong)",
            }}
          >
            {/* Modal Header */}
            <div className="flex items-center justify-between pb-3 border-b" style={{ borderColor: "var(--grid-line)" }}>
              <div className="flex items-center gap-2.5">
                <span className="text-[22px]">💼</span>
                <div>
                  <h3 id="capital-modal-title" className="text-[15px] font-black" style={{ color: "var(--text-main)" }}>
                    Start Live Trading Session
                  </h3>
                  <p className="text-[11.5px] font-semibold" style={{ color: "var(--text-muted)" }}>
                    Capital Allocation & Wallet Top-Up
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setShowCapitalModal(false)}
                className="w-7 h-7 rounded-lg flex items-center justify-center border-none cursor-pointer text-[14px] font-bold"
                style={{ background: "var(--surface-subtle)", color: "var(--text-muted)" }}
              >
                ✕
              </button>
            </div>

            {/* Current Account Balances */}
            <div
              className="rounded-xl p-3 px-3.5 border flex flex-col gap-1.5"
              style={{ background: "var(--surface-subtle)", borderColor: "var(--card-border-strong)" }}
            >
              <div className="text-[10.5px] font-black uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
                Current Trading Wallet
              </div>
              <div className="grid grid-cols-3 gap-2 text-[11.5px]">
                <div>
                  <div style={{ color: "var(--text-muted)" }}>Available Cash</div>
                  <div className="text-[13.5px] font-black" style={{ color: "var(--green-pnl)" }}>
                    ₹{(liveTick?.cash ?? summary.cash).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </div>
                </div>
                <div>
                  <div style={{ color: "var(--text-muted)" }}>Invested in Stocks</div>
                  <div className="text-[13.5px] font-black" style={{ color: "var(--text-main)" }}>
                    ₹{(liveTick?.invested_inr ?? summary.invested_inr).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </div>
                </div>
                <div>
                  <div style={{ color: "var(--text-muted)" }}>Total Portfolio Equity</div>
                  <div className="text-[13.5px] font-black" style={{ color: "var(--text-main)" }}>
                    ₹{(liveTick?.equity ?? summary.equity).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </div>
                </div>
              </div>
            </div>

            {/* Prompt & Explanation */}
            <div>
              <p className="text-[12.5px] font-bold mb-1.5 leading-snug" style={{ color: "var(--text-main)" }}>
                Would you like to invest more capital into your wallet for this session?
              </p>
              <p className="text-[11.5px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
                💡 <b>Why add capital?</b> If all 4 slots are currently held in winning swing trades, adding capital creates fresh slot capacity so the AI can buy today&apos;s new #1 ranked stock immediately without waiting for existing trades to close.
              </p>
            </div>

            {/* Quick Choice Buttons */}
            <div className="grid grid-cols-4 gap-2">
              {[
                { label: "₹0", amount: 0, sub: "Current Funds" },
                { label: "+₹25,000", amount: 25000, sub: "+1 Full Slot" },
                { label: "+₹50,000", amount: 50000, sub: "+2 Full Slots" },
                { label: "+₹1,00,000", amount: 100000, sub: "+4 Full Slots" },
              ].map((opt) => {
                const active = depositAmount === opt.amount && customAmountStr === "";
                return (
                  <button
                    key={opt.amount}
                    type="button"
                    onClick={() => {
                      setDepositAmount(opt.amount);
                      setCustomAmountStr("");
                    }}
                    className="rounded-xl p-2.5 border text-center cursor-pointer transition-all flex flex-col items-center justify-center gap-0.5"
                    style={{
                      background: active ? "var(--yellow-accent)" : "var(--input-bg)",
                      color: active ? "#111827" : "var(--text-main)",
                      borderColor: active ? "var(--yellow-border)" : "var(--card-border-strong)",
                    }}
                  >
                    <div className="text-[12.5px] font-black">{opt.label}</div>
                    <div className="text-[10px] opacity-80">{opt.sub}</div>
                  </button>
                );
              })}
            </div>

            {/* Custom Amount Input */}
            <div className="flex items-center gap-2">
              <label htmlFor="custom-deposit-inr" className="text-[11.5px] font-bold" style={{ color: "var(--text-muted)" }}>
                Or Custom Amount:
              </label>
              <div className="relative flex-1">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[12px] font-bold" style={{ color: "var(--text-muted)" }}>
                  ₹
                </span>
                <input
                  id="custom-deposit-inr"
                  type="number"
                  placeholder="e.g. 35000"
                  value={customAmountStr}
                  onChange={(e) => {
                    setCustomAmountStr(e.target.value);
                    const val = parseFloat(e.target.value);
                    setDepositAmount(isNaN(val) || val < 0 ? 0 : val);
                  }}
                  className="w-full rounded-xl pl-7 pr-3 py-1.5 text-[12px] font-bold border outline-none"
                  style={{
                    background: "var(--input-bg)",
                    borderColor: "var(--card-border-strong)",
                    color: "var(--text-main)",
                  }}
                />
              </div>
            </div>

            {/* Live Preview if depositing */}
            {depositAmount > 0 && (
              <div
                className="rounded-xl p-2.5 px-3 border text-[11.5px] font-bold flex items-center justify-between"
                style={{
                  background: "rgba(16, 185, 129, 0.08)",
                  borderColor: "rgba(16, 185, 129, 0.35)",
                  color: "var(--green-pnl)",
                }}
              >
                <span>💳 Adding: +₹{depositAmount.toLocaleString("en-IN")}</span>
                <span>
                  New Cash: ₹{((liveTick?.cash ?? summary.cash) + depositAmount).toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                </span>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex items-center justify-end gap-2.5 pt-3 border-t" style={{ borderColor: "var(--grid-line)" }}>
              <button
                type="button"
                onClick={() => setShowCapitalModal(false)}
                className="rounded-xl px-4 py-2 text-[12px] font-bold border cursor-pointer"
                style={{
                  background: "var(--input-bg)",
                  borderColor: "var(--card-border-strong)",
                  color: "var(--text-main)",
                }}
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={opBusy}
                onClick={() => executeGoLive(depositAmount)}
                className="rounded-xl px-4 py-2 text-[12px] font-black text-white border-none cursor-pointer transition-all shadow-md"
                style={{ background: "var(--green-btn)" }}
              >
                {opBusy
                  ? "⏳ Starting..."
                  : depositAmount > 0
                  ? `💳 Deposit ₹${depositAmount.toLocaleString("en-IN")} & Start Live Session`
                  : "🚀 Start Session with Current Balance"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Toast Notification */}
      {toastMsg && (
        <div
          role="status"
          aria-live="polite"
          className="fixed bottom-5 right-6 z-50 rounded-xl px-4 py-2.5 text-[12.5px] font-bold shadow-2xl border-l-4"
          style={{
            background: "#111827",
            color: "#FFFFFF",
            borderLeftColor: "var(--yellow-accent)",
          }}
        >
          {toastMsg}
        </div>
      )}
    </main>
  );
}
