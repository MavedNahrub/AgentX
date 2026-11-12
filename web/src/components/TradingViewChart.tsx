"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  ColorType,
  CrosshairMode,
  IChartApi,
  ISeriesApi,
  LineStyle,
  Time,
  createChart,
} from "lightweight-charts";
import type { CurvePoint, OhlcvBar, StockBundle } from "@/types/agentx";

interface TradingViewChartProps {
  bundle: StockBundle;
  bars: OhlcvBar[];
  timeframe: "1m" | "5m" | "15m" | "1D";
  chartMode: "candles" | "overview";
  showOverlays: boolean;
  overviewPts: CurvePoint[];
  portfolioEquity: number;
  sessionsCount: number;
  isDark: boolean;
  playing: boolean;
  onStopPlaying: () => void;
}

export default function TradingViewChart({
  bundle,
  bars,
  timeframe,
  chartMode,
  showOverlays,
  overviewPts,
  portfolioEquity,
  sessionsCount,
  isDark,
  playing,
  onStopPlaying,
}: TradingViewChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [playRatio, setPlayRatio] = useState<number>(1.0);
  const [hoveredBar, setHoveredBar] = useState<OhlcvBar | null>(null);
  const [hoveredOverview, setHoveredOverview] = useState<CurvePoint | null>(null);

  // Smooth 60-FPS Replay cursor
  useEffect(() => {
    if (!playing) {
      return;
    }
    setPlayRatio((prev) => (prev >= 0.98 ? 0.14 : prev));
    let rafId = 0;
    const step = () => {
      setPlayRatio((prev) => {
        const next = prev + 0.004;
        if (next >= 1.0) {
          onStopPlaying();
          return 1.0;
        }
        rafId = requestAnimationFrame(step);
        return next;
      });
    };
    rafId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafId);
  }, [playing, onStopPlaying]);

  // Reset cursor when switching symbol or mode while not playing
  useEffect(() => {
    if (!playing) {
      setPlayRatio(1.0);
    }
  }, [bundle.sym, timeframe, chartMode, playing]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const bgColor = isDark ? "#151B28" : "#FFFFFF";
    const gridColor = isDark ? "#1E293B" : "#F1F5F9";
    const textColor = isDark ? "#94A3B8" : "#64748B";
    const borderColor = isDark ? "#263042" : "#E2E8F0";

    const chart = createChart(el, {
      width: el.clientWidth,
      height: Math.max(260, el.clientHeight),
      layout: {
        background: { type: ColorType.Solid, color: bgColor },
        textColor,
        fontFamily: '-apple-system, BlinkMacSystemFont, "Inter", monospace',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: gridColor },
        horzLines: { color: gridColor },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor,
        scaleMargins: { top: 0.09, bottom: 0.12 },
      },
      timeScale: {
        borderColor,
        timeVisible: timeframe !== "1D" && chartMode === "candles",
        secondsVisible: false,
      },
    });
    chartRef.current = chart;

    if (chartMode === "overview" && overviewPts.length > 1) {
      const count = Math.max(2, Math.floor(overviewPts.length * playRatio));
      const slice = overviewPts.slice(0, count);
      const baseEpoch = 1761440400; // Monotonic synthetic timestamps for multi-session curve

      const aiSeries: ISeriesApi<"Area"> = chart.addAreaSeries({
        lineColor: "#16A34A",
        topColor: "rgba(22, 163, 74, 0.24)",
        bottomColor: "rgba(22, 163, 74, 0.02)",
        lineWidth: 3,
        priceFormat: {
          type: "custom",
          formatter: (val: number) => `${val >= 0 ? "+" : ""}${val.toFixed(2)}%`,
        },
      });

      const niftySeries: ISeriesApi<"Line"> = chart.addLineSeries({
        color: "#94A3B8",
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
        priceFormat: {
          type: "custom",
          formatter: (val: number) => `${val >= 0 ? "+" : ""}${val.toFixed(2)}%`,
        },
      });

      const aiData = slice.map((pt, i) => ({
        time: (baseEpoch + i * 300) as Time,
        value: pt.p,
      }));
      const nifData = slice.map((pt, i) => ({
        time: (baseEpoch + i * 300) as Time,
        value: pt.n,
      }));

      aiSeries.setData(aiData);
      niftySeries.setData(nifData);
      setHoveredOverview(slice[slice.length - 1]);

      const timeLookup = new Map<number, CurvePoint>();
      slice.forEach((pt, i) => timeLookup.set(baseEpoch + i * 300, pt));

      chart.subscribeCrosshairMove((param) => {
        if (!param.time) {
          setHoveredOverview(slice[slice.length - 1]);
          return;
        }
        const found = timeLookup.get(Number(param.time));
        if (found) setHoveredOverview(found);
      });

      chart.timeScale().fitContent();
    } else if (bars.length > 0) {
      const count = Math.max(5, Math.floor(bars.length * playRatio));
      const slice = bars.slice(0, count);

      const candleSeries = chart.addCandlestickSeries({
        upColor: "#16A34A",
        downColor: "#EF4444",
        borderUpColor: "#16A34A",
        borderDownColor: "#EF4444",
        wickUpColor: "#16A34A",
        wickDownColor: "#EF4444",
      });

      candleSeries.setData(
        slice.map((b) => ({
          time: b.time as Time,
          open: b.o,
          high: b.h,
          low: b.l,
          close: b.c,
        }))
      );

      if (showOverlays) {
        // Institutional VWAP (Gold)
        const vwapSeries = chart.addLineSeries({
          color: "#F59E0B",
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: false,
        });
        vwapSeries.setData(
          slice.map((b) => ({
            time: b.time as Time,
            value: b.vwap,
          }))
        );

        // Fast EMA 9 (Sky Blue)
        const ema9Series = chart.addLineSeries({
          color: "#0284C7",
          lineWidth: 1,
          lineStyle: LineStyle.Solid,
          priceLineVisible: false,
          lastValueVisible: false,
        });
        ema9Series.setData(
          slice.map((b) => ({
            time: b.time as Time,
            value: b.ema9,
          }))
        );

        // Step-Ratcheting Chandelier Stop Curve if available on intraday bars
        const ratchetPoints = slice.filter((b) => b.ratchet_stop !== null);
        if (ratchetPoints.length > 2) {
          const ratchetSeries = chart.addLineSeries({
            color: "#EA580C",
            lineWidth: 2,
            lineStyle: LineStyle.Dotted,
            priceLineVisible: false,
            lastValueVisible: false,
          });
          ratchetSeries.setData(
            ratchetPoints.map((b) => ({
              time: b.time as Time,
              value: Number(b.ratchet_stop),
            }))
          );
        }

        // GTT OCO Target Leg A PriceLine
        if (bundle.target_px > 0) {
          candleSeries.createPriceLine({
            price: bundle.target_px,
            color: "#16A34A",
            lineWidth: 2,
            lineStyle: LineStyle.Dashed,
            axisLabelVisible: true,
            title: "🎯 Target OCO",
          });
        }

        // Entry Fill PriceLine
        if (bundle.buy_px > 0) {
          candleSeries.createPriceLine({
            price: bundle.buy_px,
            color: "#0284C7",
            lineWidth: 1,
            lineStyle: LineStyle.Dotted,
            axisLabelVisible: true,
            title: "Entry Price",
          });
        }

        // Chandelier Safety Sell Floor PriceLine
        if (bundle.stop_px > 0) {
          candleSeries.createPriceLine({
            price: bundle.stop_px,
            color: "#EF4444",
            lineWidth: 2,
            lineStyle: LineStyle.Dashed,
            axisLabelVisible: true,
            title: "🔒 Safety Floor",
          });
        }
      }

      setHoveredBar(slice[slice.length - 1]);
      const barMap = new Map<string, OhlcvBar>();
      slice.forEach((b) => barMap.set(String(b.time), b));

      chart.subscribeCrosshairMove((param) => {
        if (!param.time) {
          setHoveredBar(slice[slice.length - 1]);
          return;
        }
        const hit = barMap.get(String(param.time));
        if (hit) setHoveredBar(hit);
      });

      chart.timeScale().fitContent();
    }

    const ro = new ResizeObserver(() => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: containerRef.current.clientWidth,
          height: Math.max(260, containerRef.current.clientHeight),
        });
      }
    });
    ro.observe(el);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [bars, bundle, chartMode, showOverlays, overviewPts, isDark, timeframe, playRatio]);

  const activeBar = hoveredBar || bars[bars.length - 1];
  const activeOv = hoveredOverview || overviewPts[overviewPts.length - 1];

  return (
    <div className="flex flex-col w-full h-full min-h-0 min-w-0">
      {/* OHLCV / Portfolio Telemetry Strip matching current UI */}
      <div
        className="tabular-nums flex flex-wrap items-center gap-x-3 gap-y-1 px-2.5 py-1.5 mb-1.5 rounded-lg text-[11px] font-bold"
        style={{ background: "var(--input-bg)" }}
      >
        {chartMode === "overview" && activeOv ? (
          <>
            <span>
              AgentX AI Portfolio ({sessionsCount} Sessions Continuous • {activeOv.t})
            </span>
            <span style={{ color: "var(--green-main)" }}>
              AI Return: {activeOv.p >= 0 ? "+" : ""}
              {activeOv.p.toFixed(2)}% (₹
              {portfolioEquity.toLocaleString("en-IN", {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
              })}
              )
            </span>
            <span style={{ color: "var(--text-muted)" }}>
              Nifty 50: {activeOv.n >= 0 ? "+" : ""}
              {activeOv.n.toFixed(2)}%
            </span>
            <span style={{ color: "#CA8A04" }}>
              Net Alpha: {activeOv.p - activeOv.n >= 0 ? "+" : ""}
              {(activeOv.p - activeOv.n).toFixed(2)}%
            </span>
          </>
        ) : activeBar ? (
          <>
            <span>
              NSE:{bundle.sym} • {timeframe} • {bundle.strategy}
            </span>
            <span>
              O <b style={{ color: "var(--green-main)" }}>{activeBar.o.toFixed(2)}</b>
            </span>
            <span>
              H <b style={{ color: "var(--green-main)" }}>{activeBar.h.toFixed(2)}</b>
            </span>
            <span>
              L <b style={{ color: "var(--green-main)" }}>{activeBar.l.toFixed(2)}</b>
            </span>
            <span>
              C <b style={{ color: "var(--green-main)" }}>{activeBar.c.toFixed(2)}</b>
            </span>
            <span>
              Vol <b style={{ color: "var(--blue-accent)" }}>{Math.round(activeBar.v).toLocaleString("en-IN")}</b>
            </span>
            <span>
              VWAP <b style={{ color: "#D97706" }}>₹{activeBar.vwap.toFixed(2)}</b>
            </span>
            <span>
              RSI(14) <b>{activeBar.rsi14.toFixed(1)}</b>
            </span>
            <span>
              Stop <b style={{ color: "var(--red-main)" }}>₹{bundle.stop_px.toFixed(2)}</b>
            </span>
            <span>
              Target <b style={{ color: "var(--green-main)" }}>₹{bundle.target_px.toFixed(2)}</b>
            </span>
          </>
        ) : null}
      </div>

      {/* Hardware-Accelerated TradingView Lightweight Charts Container */}
      <div ref={containerRef} className="w-full flex-1 min-h-0 rounded-lg overflow-hidden" />
    </div>
  );
}
