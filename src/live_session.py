"""
AgentX v2.0 — Production Live NSE Trading Session Engine (`src/live_session.py`)
--------------------------------------------------------------------------------
Manages the real-time 09:15 AM -> 03:30 PM IST NSE trading lifecycle:
  1. Pre-Open Arming (< 09:15 IST): Refreshes EOD history & waits cleanly for 09:15:00 IST open.
  2. Opening Price Discovery (09:15 - 09:29 IST): Tracks existing holdings minute-by-minute
     using real executed exchange bars and triggers 09:30 AM Morning Scout + Orders once
     the real 15-minute opening bar closes.
  3. Late-Join & Crash/Restart Reconciliation (09:30 - 15:29 IST): Inspects SQLite
     (`order_fill_ledger`, `open_positions`) & today's intraday curve to respect already-recorded
     trades (never duplicating fills), fetches all real candles executed so far today, catches up
     strictly from `last_recorded_minute + 1` to the current market instant, and seamlessly runs
     live 1x monitoring thereafter.
  4. Auto-Settlement at 15:30 PM IST: Performs final minute sync, persists `bars_held`, runs EOD
     settlement report + ledger update, and transitions cleanly to `SETTLED`.

Zero synthetic holiday bars, zero lookahead beyond latest executed exchange candle.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from src.alpha_scout import WalkForwardAlphaScout
from src.data_engine import NIFTY50_SECTOR_MAP, CapitalConfig, HybridDataEngine
from src.feature_engine import ChameleonFeatureEngine
from src.paper_broker import IndiaPaperBroker
from src.risk_vault import CryptographicRiskVault
from src.strategy_and_news import EventDrivenStrategyTournament

IST = timezone(timedelta(hours=5, minutes=30))
DB_PATH = "data/cache/paper_broker_ledger.sqlite"
CACHE_DIR = Path("data/cache")
OUT_DIR = Path("reports/live_6day_test")

NSE_EXCHANGE_HOLIDAYS: Dict[str, str] = {
    "2026-11-10": "Diwali Balipratipada (Official NSE Trading Holiday)",
}


def get_ist_now() -> datetime:
    """Return current timezone-aware IST datetime."""
    return datetime.now(IST)


def market_minute_at(dt_ist: datetime) -> int:
    """
    Map an IST datetime to NSE trading minute index:
      0   -> Pre-open (< 09:15:00)
      1..375 -> Live trading minutes (09:15:00 .. 15:29:59)
      376 -> Closed (>= 15:30:00)
    """
    local_dt = dt_ist.astimezone(IST) if dt_ist.tzinfo is not None else dt_ist.replace(tzinfo=IST)
    mkt_open = local_dt.replace(hour=9, minute=15, second=0, microsecond=0)
    mkt_close = local_dt.replace(hour=15, minute=30, second=0, microsecond=0)
    if local_dt < mkt_open:
        return 0
    if local_dt >= mkt_close:
        return 376
    return int((local_dt - mkt_open).total_seconds() // 60) + 1


def is_trading_day(date_str: str) -> Tuple[bool, str]:
    """Validate weekday and non-holiday NSE trading calendar."""
    dt = pd.Timestamp(date_str)
    if dt.weekday() >= 5:
        day_name = dt.day_name()
        return False, f"NSE Closed ({day_name}, {date_str}): Weekend — no live trading."
    if date_str in NSE_EXCHANGE_HOLIDAYS:
        return False, f"NSE Exchange Holiday ({date_str}): {NSE_EXCHANGE_HOLIDAYS[date_str]}"
    return True, "OPEN_TRADING_DAY"


def _prev_trading_day_str(target_date_str: str) -> str:
    """Find most recent weekday non-holiday strictly before `target_date_str`."""
    cur = pd.Timestamp(target_date_str) - timedelta(days=1)
    for _ in range(10):
        ds = cur.strftime("%Y-%m-%d")
        if cur.weekday() < 5 and ds not in NSE_EXCHANGE_HOLIDAYS:
            return ds
        cur -= timedelta(days=1)
    return cur.strftime("%Y-%m-%d")


def ensure_eod_cache_fresh_before(target_date_str: str) -> pd.Timestamp:
    """
    Ensure cached Nifty 50 EOD parquet includes the latest prior trading day before `target_date_str`.
    Automatically pulls fresh daily bars if stale.
    """
    cap_cfg = CapitalConfig(initial_capital_inr=100_000.0)
    engine = HybridDataEngine(cache_dir=str(CACHE_DIR), capital_cfg=cap_cfg)
    eod_path = CACHE_DIR / "nifty50_eod.parquet"
    prev_td = pd.Timestamp(_prev_trading_day_str(target_date_str))

    needs_refresh = True
    if eod_path.exists():
        try:
            max_cached = pd.to_datetime(pd.read_parquet(eod_path, columns=["date"])["date"]).max()
            if pd.notna(max_cached) and max_cached >= prev_td:
                needs_refresh = False
        except Exception:
            needs_refresh = True

    stocks_df, _, _ = engine.fetch_and_build_telemetry(force_refresh=needs_refresh)
    return pd.to_datetime(stocks_df["date"]).max()


def _extract_tz_naive_ist(df_raw: pd.DataFrame, target_date_str: str, cutoff_ts: pd.Timestamp) -> pd.DataFrame:
    """Convert Yahoo Finance DataFrame timestamps to IST naive and strictly cap at `cutoff_ts`."""
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df = df_raw.dropna(subset=["Open", "High", "Low", "Close"]).reset_index()
    if df.empty:
        return pd.DataFrame()
    tcol = "Datetime" if "Datetime" in df.columns else df.columns[0]
    ts = pd.to_datetime(df[tcol])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df["timestamp"] = ts
    df = df[
        (df["timestamp"].dt.strftime("%Y-%m-%d") == target_date_str)
        & (df["timestamp"] <= cutoff_ts)
    ].copy()
    return df


def fetch_real_intraday_up_to_now(
    target_date_str: str,
    max_minute: int,
    symbols: Optional[List[str]] = None,
    save_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Fetch real NSE 15m, 5m, and 1m candles executed on `target_date_str` strictly up to `max_minute`.
    Zero synthetic fabrication beyond the latest real exchange timestamp returned by Yahoo Finance.
    Returns: (df_15m, df_375m_partial, max_real_minute_available)
    """
    ok_day, reason = is_trading_day(target_date_str)
    if not ok_day:
        raise RuntimeError(reason)

    clamped_max_min = max(1, min(375, int(max_minute)))
    open_ts = pd.Timestamp(f"{target_date_str} 09:15:00")
    cutoff_ts = open_ts + timedelta(minutes=clamped_max_min - 1)

    raw_syms = symbols or list(NIFTY50_SECTOR_MAP.keys())
    yf_tickers = [s if (str(s).endswith(".NS") or str(s).startswith("^")) else f"{s}.NS" for s in raw_syms]
    target_dt = pd.Timestamp(target_date_str)
    start_str = target_dt.strftime("%Y-%m-%d")
    end_str = (target_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    raw_1m = yf.download(
        tickers=yf_tickers,
        start=start_str,
        end=end_str,
        interval="1m",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    raw_5m = yf.download(
        tickers=yf_tickers,
        start=start_str,
        end=end_str,
        interval="5m",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    raw_15m = yf.download(
        tickers=yf_tickers,
        start=start_str,
        end=end_str,
        interval="15m",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    sym_1m_Frames: Dict[str, pd.DataFrame] = {}
    rows_15m: List[pd.DataFrame] = []
    latest_real_ts: Optional[pd.Timestamp] = None

    for yf_sym in yf_tickers:
        clean_sym = yf_sym.replace(".NS", "")

        # 15m opening / intraday bars
        try:
            s15_raw = raw_15m[yf_sym] if (len(yf_tickers) > 1 and yf_sym in raw_15m.columns.get_level_values(0)) else (raw_15m if len(yf_tickers) == 1 else pd.DataFrame())
            df15 = _extract_tz_naive_ist(s15_raw, target_date_str, cutoff_ts)
            if not df15.empty:
                df15["symbol"] = clean_sym
                df15 = df15.rename(
                    columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
                )
                rows_15m.append(df15[["timestamp", "symbol", "open", "high", "low", "close", "volume"]])
        except Exception:
            pass

        # Real 1m bars (works even for partial live session: 1 bar, 15 bars, 150 bars, etc.)
        built_1m = False
        try:
            s1_raw = raw_1m[yf_sym] if (len(yf_tickers) > 1 and yf_sym in raw_1m.columns.get_level_values(0)) else (raw_1m if len(yf_tickers) == 1 else pd.DataFrame())
            df1 = _extract_tz_naive_ist(s1_raw, target_date_str, cutoff_ts)
            if not df1.empty:
                df1["timestamp"] = df1["timestamp"].dt.floor("min")
                df1 = df1.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                sym_max = pd.Timestamp(df1["timestamp"].max())
                latest_real_ts = sym_max if (latest_real_ts is None or sym_max > latest_real_ts) else latest_real_ts
                df1["bar_source"] = "RAW_NSE_1M_LIVE"
                sym_1m_Frames[clean_sym] = df1[["timestamp", "Open", "High", "Low", "Close", "Volume", "bar_source"]]
                built_1m = True
        except Exception:
            built_1m = False

        # Fallback: expand real 5m bars strictly up to their exact executed timestamps
        if not built_1m:
            try:
                s5_raw = raw_5m[yf_sym] if (len(yf_tickers) > 1 and yf_sym in raw_5m.columns.get_level_values(0)) else (raw_5m if len(yf_tickers) == 1 else pd.DataFrame())
                df5 = _extract_tz_naive_ist(s5_raw, target_date_str, cutoff_ts)
                if not df5.empty:
                    df5["timestamp"] = df5["timestamp"].dt.floor("min")
                    sub_records = []
                    for _, r5 in df5.iterrows():
                        base_ts = pd.Timestamp(r5["timestamp"])
                        o5, h5, l5, c5, v5 = (
                            float(r5["Open"]),
                            float(r5["High"]),
                            float(r5["Low"]),
                            float(r5["Close"]),
                            float(r5.get("Volume", 0.0)) / 5.0,
                        )
                        pts = [o5, l5, 0.5 * (o5 + c5), h5, c5] if c5 >= o5 else [o5, h5, 0.5 * (o5 + c5), l5, c5]
                        for k in range(5):
                            ts_k = base_ts + timedelta(minutes=k)
                            if ts_k > cutoff_ts:
                                break
                            op_k = pts[k - 1] if k > 0 else o5
                            cl_k = pts[k]
                            hi_k = max(op_k, cl_k) if k not in (1, 3) else max(op_k, cl_k, pts[k])
                            lo_k = min(op_k, cl_k) if k not in (1, 3) else min(op_k, cl_k, pts[k])
                            sub_records.append(
                                {
                                    "timestamp": ts_k,
                                    "Open": round(op_k, 2),
                                    "High": round(hi_k, 2),
                                    "Low": round(lo_k, 2),
                                    "Close": round(cl_k, 2),
                                    "Volume": round(v5, 1),
                                    "bar_source": "NSE_5M_LIVE_BRIDGE",
                                }
                            )
                    if sub_records:
                        df_sub = pd.DataFrame(sub_records).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                        sym_max = pd.Timestamp(df_sub["timestamp"].max())
                        latest_real_ts = sym_max if (latest_real_ts is None or sym_max > latest_real_ts) else latest_real_ts
                        sym_1m_Frames[clean_sym] = df_sub
            except Exception:
                pass

    if latest_real_ts is None or not sym_1m_Frames:
        return pd.DataFrame(), pd.DataFrame(), 0

    max_real_minute = max(1, min(375, int((latest_real_ts - open_ts).total_seconds() // 60) + 1))
    partial_grid = pd.date_range(start=open_ts, end=latest_real_ts, freq="1min")

    rows_375m: List[pd.DataFrame] = []
    for clean_sym, df_raw in sym_1m_Frames.items():
        bsrc = str(df_raw["bar_source"].iloc[0])
        g = df_raw.set_index("timestamp").reindex(partial_grid)
        # Fill internal single-minute illiquidity gaps strictly within [09:15 .. latest_real_ts]
        g["Close"] = g["Close"].ffill().bfill()
        g["Open"] = g["Open"].fillna(g["Close"])
        g["High"] = g["High"].fillna(g[["Open", "Close"]].max(axis=1))
        g["Low"] = g["Low"].fillna(g[["Open", "Close"]].min(axis=1))
        g["Volume"] = g["Volume"].fillna(0.0).astype(float)
        g = g.reset_index().rename(
            columns={"index": "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
        )
        g["symbol"] = clean_sym
        g["minute_num"] = np.arange(1, len(g) + 1)
        g["bar_source"] = bsrc
        rows_375m.append(g[["minute_num", "timestamp", "symbol", "open", "high", "low", "close", "volume", "bar_source"]])

    out_15m = pd.concat(rows_15m, ignore_index=True) if rows_15m else pd.DataFrame()
    out_375m = pd.concat(rows_375m, ignore_index=True) if rows_375m else pd.DataFrame()

    # If 09:30 AM passed (max_real_minute >= 15) and Yahoo's 15m endpoint hasn't posted the 09:15 bar yet,
    # synthesize the exact 09:15-09:29 15m bar directly from those real 1m candles!
    if max_real_minute >= 15 and not out_375m.empty:
        has_open_15m = (
            not out_15m.empty
            and (out_15m["timestamp"].dt.strftime("%H:%M") == "09:15").sum() >= max(1, len(sym_1m_Frames) // 2)
        )
        if not has_open_15m:
            first15 = out_375m[out_375m["minute_num"] <= 15].sort_values("minute_num")
            agg15 = (
                first15.groupby("symbol", as_index=False)
                .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
            )
            agg15["timestamp"] = open_ts
            out_15m = pd.concat([out_15m, agg15[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]], ignore_index=True)
            out_15m = out_15m.drop_duplicates(subset=["timestamp", "symbol"], keep="first")

    if save_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if not out_15m.empty:
            out_15m.to_parquet(CACHE_DIR / f"intraday_{target_date_str}_15m.parquet", index=False)
        if not out_375m.empty:
            out_375m.to_parquet(CACHE_DIR / f"intraday_{target_date_str}_375m.parquet", index=False)

    return out_15m, out_375m, max_real_minute


def load_recorded_day_state(target_date_str: str, db_path: str = DB_PATH) -> Dict[str, object]:
    """
    Inspect SQLite broker ledger + today's curve/state artifacts to reconstruct exact state:
      - Whether morning routine & buys/evictions were already recorded today (`morning_done`)
      - Whether today's EOD settlement already ran (`already_settled_eod`)
      - Existing `open_positions` from SQLite + `highest_price_seen` + `last_recorded_minute`
      - Today's previously logged `intraday_events` & exited symbols
    """
    cap_cfg = CapitalConfig(initial_capital_inr=100_000.0)
    vault = CryptographicRiskVault(capital_cfg=cap_cfg)
    broker = IndiaPaperBroker(vault=vault, db_path=db_path)

    open_pos_df = broker.get_open_positions_df()
    acct_summary = broker.get_account_summary()

    with sqlite3.connect(db_path) as conn:
        fills_today = pd.read_sql_query(
            "SELECT * FROM order_fill_ledger WHERE as_of_date = ? ORDER BY fill_id ASC",
            conn,
            params=(target_date_str,),
        )

    proof_file = OUT_DIR / f"causal_proof_{target_date_str}.json"
    session_file = OUT_DIR / f"session_375m_{target_date_str}.json"
    live_state_file = OUT_DIR / "live_375m_state.json"
    curve_file = OUT_DIR / f"intraday_375m_curve_{target_date_str}.parquet"
    ledger_csv = OUT_DIR / "collective_6day_ledger.csv"

    already_settled_eod = False
    if ledger_csv.exists():
        try:
            ldf = pd.read_csv(ledger_csv)
            if not ldf.empty and target_date_str in ldf["market_date"].astype(str).values:
                already_settled_eod = True
        except Exception:
            pass

    existing_curve_df = pd.DataFrame()
    minute_curve_rows: List[dict] = []
    last_recorded_min = 0
    if curve_file.exists():
        try:
            existing_curve_df = pd.read_parquet(curve_file).sort_values("minute_num")
            if not existing_curve_df.empty:
                last_recorded_min = int(existing_curve_df["minute_num"].max())
                minute_curve_rows = existing_curve_df.to_dict(orient="records")
        except Exception:
            existing_curve_df = pd.DataFrame()

    saved_state: dict = {}
    for cand_f in (session_file, live_state_file):
        if cand_f.exists():
            try:
                st = json.loads(cand_f.read_text(encoding="utf-8"))
                if str(st.get("session_date", "")) == target_date_str:
                    saved_state = st
                    break
            except Exception:
                pass

    intraday_events: List[dict] = list(saved_state.get("intraday_events", []))
    exited_today_syms: set[str] = set()
    if not fills_today.empty:
        exited_today_syms.update(
            fills_today[fills_today["side"] == "SELL"]["symbol"].astype(str).tolist()
        )
    for ex_ev in saved_state.get("intraday_exits_executed", []):
        if ex_ev.get("symbol"):
            exited_today_syms.add(str(ex_ev["symbol"]))

    highest_map: Dict[str, float] = {}
    for p_st in saved_state.get("open_positions", []):
        if p_st.get("symbol") and p_st.get("highest_price_seen"):
            highest_map[str(p_st["symbol"])] = float(p_st["highest_price_seen"])

    active_positions: Dict[str, dict] = {}
    for _, pos in open_pos_df.iterrows():
        sym = str(pos["symbol"])
        is_entered_today = str(pos["entry_date"]) == target_date_str
        cur_px = float(pos["current_price"])
        curve_high = (
            float(existing_curve_df[f"{sym}_price"].max())
            if (not existing_curve_df.empty and f"{sym}_price" in existing_curve_df.columns)
            else cur_px
        )
        hi_seen = max(float(pos["entry_fill_price"]), cur_px, curve_high, highest_map.get(sym, 0.0))
        active_positions[sym] = {
            "symbol": sym,
            "sector": str(pos["sector"]),
            "slot_id": int(pos["slot_id"]),
            "assigned_strategy": str(pos["assigned_strategy"]),
            "entry_date": str(pos["entry_date"]),
            "entry_fill_price": float(pos["entry_fill_price"]),
            "quantity": int(pos["quantity"]),
            "current_price": cur_px,
            "highest_price_seen": hi_seen,
            "initial_stop_price": float(pos["initial_stop_price"]),
            "stop_loss_price": float(pos["stop_loss_price"]),
            "profit_target_price": float(pos["profit_target_price"]),
            "expected_5d_alpha_pct": float(pos["expected_5d_alpha_pct"]),
            "conviction_pct": float(pos["conviction_pct"]),
            "bars_held": 0 if is_entered_today else int(pos["bars_held"]),
        }

    morning_done = bool(
        proof_file.exists()
        or (not fills_today.empty)
        or last_recorded_min >= 16
        or str(saved_state.get("status", "")).startswith("MORNING_ROUTINE_COMPLETED")
    )
    proof_meta = (
        json.loads(proof_file.read_text(encoding="utf-8"))
        if proof_file.exists()
        else {"session_date": target_date_str}
    )

    return {
        "morning_done": morning_done,
        "already_settled_eod": already_settled_eod,
        "last_recorded_min": last_recorded_min,
        "minute_curve_rows": minute_curve_rows,
        "intraday_events": intraday_events,
        "exited_today_syms": exited_today_syms,
        "active_positions": active_positions,
        "cash_balance": float(acct_summary["cash_balance_inr"]),
        "initial_cap": float(acct_summary["initial_capital_inr"]),
        "proof_meta": proof_meta,
    }


def _build_bar_and_nifty_lookup(
    df_375m: pd.DataFrame,
    tracked_symbols: List[str],
    target_date_str: str,
) -> Tuple[Dict[int, Dict[str, dict]], Dict[int, dict]]:
    """Build (minute -> symbol -> OHLCV dict) and (minute -> Nifty50 equal-weight benchmark)."""
    if df_375m.empty:
        return {}, {}
    sub = df_375m[df_375m["symbol"].isin(tracked_symbols)]
    bar_lookup: Dict[int, Dict[str, dict]] = {}
    for _, r in sub.iterrows():
        m_num = int(r["minute_num"])
        bar_lookup.setdefault(m_num, {})[str(r["symbol"])] = {
            "timestamp": str(r["timestamp"]),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r["volume"]),
            "bar_source": str(r.get("bar_source", "RAW_NSE_1M_LIVE")),
        }

    nifty_df = (
        df_375m.groupby(["minute_num", "timestamp"], as_index=False)["close"]
        .mean()
        .sort_values("minute_num")
        .reset_index(drop=True)
    )
    n_base = float(nifty_df["close"].iloc[0]) if not nifty_df.empty else 1.0
    nifty_map = {
        int(r["minute_num"]): {
            "timestamp": str(r["timestamp"]),
            "bench_ret_pct": round((float(r["close"]) / max(1e-6, n_base) - 1.0) * 100.0, 3),
        }
        for _, r in nifty_df.iterrows()
    }
    return bar_lookup, nifty_map


def _load_or_refresh_ranked_candidates(target_date_str: str) -> pd.DataFrame:
    """Return today's causal ranked candidates without touching SQLite positions."""
    ranked_path = CACHE_DIR / "alpha_scout_latest_ranked.parquet"
    if ranked_path.exists():
        rk = pd.read_parquet(ranked_path).sort_values("alpha_rank", ascending=True).reset_index(drop=True)
        if not rk.empty and "date" in rk.columns:
            latest_dt = str(pd.to_datetime(rk["date"]).iloc[0].strftime("%Y-%m-%d"))
            if latest_dt >= _prev_trading_day_str(target_date_str):
                return rk
    cap_cfg = CapitalConfig(initial_capital_inr=100_000.0)
    data_engine = HybridDataEngine(cache_dir=str(CACHE_DIR), capital_cfg=cap_cfg)
    stocks_df, macro_df, _ = data_engine.fetch_and_build_telemetry(force_refresh=False)
    s_cut = stocks_df[stocks_df["date"] <= pd.Timestamp(target_date_str)].copy()
    m_cut = macro_df[macro_df["date"] <= pd.Timestamp(target_date_str)].copy()
    feat_df = ChameleonFeatureEngine(cache_dir=str(CACHE_DIR)).build_features(s_cut, m_cut, save_cache=False)
    scout = WalkForwardAlphaScout(capital_cfg=cap_cfg)
    ranked_df, _, _ = scout.score_cross_section_with_shap(feat_df, as_of_date=pd.Timestamp(target_date_str), retrain=True)
    board_df = EventDrivenStrategyTournament(capital_cfg=cap_cfg).run_post_tax_tournament(feat_df, as_of_date=pd.Timestamp(target_date_str))
    paired = EventDrivenStrategyTournament(capital_cfg=cap_cfg).assign_strategy_and_exit_geometry(ranked_df, board_df)
    paired.to_parquet(ranked_path, index=False)
    return paired.sort_values("alpha_rank", ascending=True).reset_index(drop=True)


def _persist_live_progress(
    target_date_str: str,
    m_curr: int,
    status_tag: str,
    active_positions: Dict[str, dict],
    minute_curve_rows: List[dict],
    intraday_events: List[dict],
    proof_meta: dict,
    cash_balance: float,
    initial_cap: float,
    db_path: str = DB_PATH,
) -> dict:
    """Persist current MTM prices, trailing stops, parquet curve, and `live_375m_state.json`."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        for sym_l, p_l in active_positions.items():
            conn.execute(
                "UPDATE open_positions SET current_price = ?, stop_loss_price = ? WHERE symbol = ?",
                (p_l["current_price"], p_l["stop_loss_price"], sym_l),
            )
            conn.execute(
                "UPDATE gtt_oco_triggers SET leg_b_current_stop_price = ? WHERE symbol = ? AND oco_pair_status LIKE 'ARMED%'",
                (p_l["stop_loss_price"], sym_l),
            )
        conn.commit()

    if minute_curve_rows:
        cur_df = pd.DataFrame(minute_curve_rows).drop_duplicates(subset=["minute_num"], keep="last").sort_values("minute_num")
        cur_df.to_parquet(OUT_DIR / f"intraday_375m_curve_{target_date_str}.parquet", index=False)
        cur_df.to_parquet(OUT_DIR / "intraday_375m_curve_latest.parquet", index=False)

    pos_val = sum(p["quantity"] * p["current_price"] for p in active_positions.values())
    eq_val = round(cash_balance + pos_val, 2)
    ret_pct = round((eq_val / max(1.0, initial_cap) - 1.0) * 100.0, 2)
    ts_now = (
        minute_curve_rows[-1]["timestamp_ist"]
        if minute_curve_rows
        else f"{target_date_str} {get_ist_now().strftime('%H:%M:%S')}"
    )

    state = {
        "session_date": target_date_str,
        "status": status_tag,
        "current_minute": m_curr,
        "total_minutes": 375,
        "current_clock_ist": f"{ts_now} IST",
        "portfolio_equity_inr": eq_val,
        "portfolio_return_pct": ret_pct,
        "cash_balance_inr": round(cash_balance, 2),
        "open_positions": list(active_positions.values()),
        "intraday_events": intraday_events[-60:],
        "causal_proof": proof_meta,
    }
    (OUT_DIR / "live_375m_state.json").write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    return state


def process_single_minute(
    m: int,
    target_date_str: str,
    sym_bars: Dict[str, dict],
    b_info: dict,
    active_positions: Dict[str, dict],
    exited_today_syms: set[str],
    cash_balance: float,
    initial_cap: float,
    rk_df: pd.DataFrame,
    vault: CryptographicRiskVault,
    broker: IndiaPaperBroker,
    cap_cfg: CapitalConfig,
    intraday_events: List[dict],
    mode_tag: str = "LIVE",
) -> Tuple[float, dict]:
    """Process one real 1-minute NSE bar causally for exits, Chandelier stops, redeploys, and MTM."""
    ts_str = b_info["timestamp"]

    for sym in list(active_positions.keys()):
        pos = active_positions[sym]
        bar = sym_bars.get(sym)
        if bar is None:
            continue

        o_m, h_m, l_m, c_m = bar["open"], bar["high"], bar["low"], bar["close"]
        pos["current_price"] = c_m

        if m >= 17:
            # 1. Intraday Stop-Loss Breach
            if c_m <= pos["stop_loss_price"]:
                exit_px = round(min(o_m, c_m, pos["stop_loss_price"]), 2)
                sell_dict = {
                    "symbol": sym,
                    "sector": pos["sector"],
                    "slot_id": pos["slot_id"],
                    "quantity_shares": pos["quantity"],
                    "exec_vwap_915": exit_px,
                    "stop_loss_price": pos["stop_loss_price"],
                    "profit_target_price": pos["profit_target_price"],
                    "assigned_strategy": pos["assigned_strategy"],
                    "expected_5d_alpha_pct": pos["expected_5d_alpha_pct"],
                    "conviction_pct": pos["conviction_pct"],
                }
                sell_pkt = vault.sign_order_packet(as_of_date=target_date_str, side="SELL", row=sell_dict, throttle_mult=1.0)
                s_res = broker.execute_signed_order(sell_pkt, exit_reason=f"INTRADAY_STOP_MIN_{m}")
                cash_balance = float(broker.get_account_summary()["cash_balance_inr"])
                intraday_events.append(
                    {
                        "minute_num": m,
                        "timestamp_ist": ts_str,
                        "event_type": "🛑 INTRADAY_STOP_LOSS_EXIT",
                        "symbol": sym,
                        "details": (
                            f"Min {m}/375 ({ts_str[-8:-3]} IST | {mode_tag}): Low ₹{l_m:,.2f} breached Stop ₹{pos['stop_loss_price']:,.2f} -> "
                            f"SOLD {pos['quantity']} sh @ ₹{s_res.get('fill_price', exit_px):,.2f} "
                            f"(Net PnL: ₹{s_res.get('realized_net_pnl_inr', 0.0):+,.2f})"
                        ),
                    }
                )
                exited_today_syms.add(sym)
                del active_positions[sym]
                continue

            # 2. Target Profit-Booking or Breakout Runner Floor Lock
            if h_m >= pos["profit_target_price"]:
                entry_px = pos["entry_fill_price"]
                init_stop_dist = max(0.005, (entry_px - pos["initial_stop_price"]) / max(1.0, entry_px))
                ext_runner_target = round(entry_px + 1.25 * (pos["profit_target_price"] - entry_px), 2)
                if pos["assigned_strategy"] in ("RANGE_REVERSION", "DEFENSIVE_SHIELD") or h_m >= ext_runner_target:
                    target_hit_px = (
                        ext_runner_target
                        if (pos["assigned_strategy"] not in ("RANGE_REVERSION", "DEFENSIVE_SHIELD") and h_m >= ext_runner_target)
                        else pos["profit_target_price"]
                    )
                    exit_px = round(max(o_m, target_hit_px), 2)
                    sell_dict = {
                        "symbol": sym,
                        "sector": pos["sector"],
                        "slot_id": pos["slot_id"],
                        "quantity_shares": pos["quantity"],
                        "exec_vwap_915": exit_px,
                        "stop_loss_price": pos["stop_loss_price"],
                        "profit_target_price": pos["profit_target_price"],
                        "assigned_strategy": pos["assigned_strategy"],
                        "expected_5d_alpha_pct": pos["expected_5d_alpha_pct"],
                        "conviction_pct": pos["conviction_pct"],
                    }
                    sell_pkt = vault.sign_order_packet(as_of_date=target_date_str, side="SELL", row=sell_dict, throttle_mult=1.0)
                    s_res = broker.execute_signed_order(sell_pkt, exit_reason=f"INTRADAY_TARGET_MIN_{m}")
                    cash_balance = float(broker.get_account_summary()["cash_balance_inr"])
                    intraday_events.append(
                        {
                            "minute_num": m,
                            "timestamp_ist": ts_str,
                            "event_type": "🎯 INTRADAY_PROFIT_TARGET_HIT",
                            "symbol": sym,
                            "details": (
                                f"Min {m}/375 ({ts_str[-8:-3]} IST | {mode_tag}): High ₹{h_m:,.2f} hit Target ₹{target_hit_px:,.2f} -> "
                                f"SOLD {pos['quantity']} sh @ ₹{s_res.get('fill_price', exit_px):,.2f} "
                                f"(Net PnL: ₹{s_res.get('realized_net_pnl_inr', 0.0):+,.2f})"
                            ),
                        }
                    )
                    exited_today_syms.add(sym)
                    del active_positions[sym]
                    continue
                else:
                    runner_floor = round(pos["profit_target_price"] * (1.0 - 0.20 * init_stop_dist), 2)
                    if runner_floor > pos["stop_loss_price"] + 0.25:
                        old_s = pos["stop_loss_price"]
                        pos["stop_loss_price"] = runner_floor
                        intraday_events.append(
                            {
                                "minute_num": m,
                                "timestamp_ist": ts_str,
                                "event_type": "🎯 TARGET_RUNNER_FLOOR_LOCKED",
                                "symbol": sym,
                                "details": f"Min {m}/375 ({ts_str[-8:-3]} IST): [{sym}] Runner Floor Locked ₹{old_s:,.2f} -> ₹{runner_floor:,.2f}",
                            }
                        )

            # 3. Post-Tax Breakeven Lock + Convex Chandelier Trail Ratchet
            if h_m > pos["highest_price_seen"]:
                pos["highest_price_seen"] = h_m
                entry_px = pos["entry_fill_price"]
                init_stop_dist = max(0.005, (entry_px - pos["initial_stop_price"]) / max(1.0, entry_px))
                gain_from_entry = (h_m - entry_px) / max(1.0, entry_px)
                rt_tax_rate = cap_cfg.dynamic_round_trip_tax_rate

                breakeven_floor = pos["initial_stop_price"]
                if gain_from_entry >= (2.0 * rt_tax_rate + 0.75 * init_stop_dist):
                    excess_gain = max(0.0, gain_from_entry - 2.0 * rt_tax_rate)
                    breakeven_floor = round(entry_px * (1.0 + 2.0 * rt_tax_rate + 0.35 * excess_gain), 2)

                target_span = max(entry_px * 0.005, pos["profit_target_price"] - entry_px)
                progress_to_target = float(np.clip((h_m - entry_px) / target_span, 0.0, 1.0))
                if progress_to_target >= 0.60:
                    breakeven_floor = max(breakeven_floor, round(entry_px + 0.55 * (h_m - entry_px), 2))
                convex_trail_dist = init_stop_dist * (1.0 - 0.35 * (progress_to_target ** 2))

                if gain_from_entry >= 0.65 * init_stop_dist:
                    cand_stop = max(breakeven_floor, round(h_m * (1.0 - convex_trail_dist), 2))
                    if cand_stop > pos["stop_loss_price"] + max(0.50, entry_px * 0.0015):
                        old_s = pos["stop_loss_price"]
                        pos["stop_loss_price"] = cand_stop
                        intraday_events.append(
                            {
                                "minute_num": m,
                                "timestamp_ist": ts_str,
                                "event_type": "🔒 CHANDELIER_STOP_RATCHET",
                                "symbol": sym,
                                "details": (
                                    f"Min {m}/375 ({ts_str[-8:-3]} IST): [{sym}] Peak ₹{h_m:,.2f} "
                                    f"({(h_m / entry_px - 1) * 100:+.2f}%) -> Stop Ratcheted ₹{old_s:,.2f} -> ₹{cand_stop:,.2f}"
                                ),
                            }
                        )

    # 4. Mid-Session Freed-Cash Redeployment (Min 17..260)
    if 17 <= m <= 260 and len(active_positions) < 4 and cash_balance >= 5000.0 and not rk_df.empty:
        used_slots_m = {int(p["slot_id"]) for p in active_positions.values()}
        free_slots_m = [sid for sid in (1, 2, 3, 4) if sid not in used_slots_m]
        for sid_m in free_slots_m:
            if cash_balance < 5000.0:
                break
            slot_alloc_budget = cash_balance / max(1, len(free_slots_m))
            for _, cand_row in rk_df.iterrows():
                c_sym = str(cand_row["symbol"])
                if c_sym in active_positions or c_sym in exited_today_syms:
                    continue
                c_bar = sym_bars.get(c_sym)
                if c_bar is None:
                    continue
                c_px_m = float(c_bar["close"])
                open_930 = float(cand_row.get("exec_vwap_915", c_px_m))
                exp_alpha_c = float(cand_row["expected_5d_alpha_pct"])
                strat_exp_c = float(cand_row.get("strategy_expectancy_pct", 0.25))
                cand_sec = str(cand_row["sector"])
                active_sec_cnt = sum(1 for p_act in active_positions.values() if str(p_act["sector"]) == cand_sec)
                if (
                    strat_exp_c <= 0.0
                    or exp_alpha_c <= cap_cfg.dynamic_round_trip_tax_pct
                    or c_px_m < open_930
                    or active_sec_cnt >= cap_cfg.max_slots_per_sector
                ):
                    continue
                qty_m = int(np.floor((slot_alloc_budget / 1.0035) / max(1.0, c_px_m)))
                if qty_m <= 0:
                    continue
                scale_ratio = c_px_m / max(1.0, open_930)
                stp_m = round(float(cand_row["stop_loss_price"]) * scale_ratio, 2)
                tgt_m = round(float(cand_row["profit_target_price"]) * scale_ratio, 2)
                buy_dict_m = {
                    "symbol": c_sym,
                    "sector": cand_sec,
                    "slot_id": sid_m,
                    "quantity_shares": qty_m,
                    "exec_vwap_915": round(c_px_m, 2),
                    "stop_loss_price": stp_m,
                    "profit_target_price": tgt_m,
                    "assigned_strategy": str(cand_row["assigned_strategy"]),
                    "expected_5d_alpha_pct": exp_alpha_c,
                    "conviction_pct": float(cand_row["conviction_pct"]),
                }
                try:
                    buy_pkt_m = vault.sign_order_packet(as_of_date=target_date_str, side="BUY", row=buy_dict_m, throttle_mult=1.0)
                    b_res_m = broker.execute_signed_order(buy_pkt_m)
                except Exception:
                    b_res_m = {"status": "REJECTED"}
                if b_res_m.get("status") == "FILLED_BUY":
                    fill_px_m = float(b_res_m["fill_price"])
                    cash_balance = float(broker.get_account_summary()["cash_balance_inr"])
                    active_positions[c_sym] = {
                        "symbol": c_sym,
                        "sector": cand_sec,
                        "slot_id": sid_m,
                        "assigned_strategy": str(cand_row["assigned_strategy"]),
                        "entry_date": target_date_str,
                        "entry_fill_price": fill_px_m,
                        "quantity": qty_m,
                        "current_price": fill_px_m,
                        "highest_price_seen": fill_px_m,
                        "initial_stop_price": stp_m,
                        "stop_loss_price": stp_m,
                        "profit_target_price": tgt_m,
                        "expected_5d_alpha_pct": exp_alpha_c,
                        "conviction_pct": float(cand_row["conviction_pct"]),
                        "bars_held": 0,
                    }
                    intraday_events.append(
                        {
                            "minute_num": m,
                            "timestamp_ist": ts_str,
                            "event_type": "⚡ INTRADAY_SLOT_REDEPLOY_BUY",
                            "symbol": c_sym,
                            "details": (
                                f"Min {m}/375 ({ts_str[-8:-3]} IST | {mode_tag}): Redeployed Freed Cash into [{c_sym}]: "
                                f"{qty_m} sh @ ₹{fill_px_m:,.2f} | Stop: ₹{stp_m:,.2f}"
                            ),
                        }
                    )
                    break

    pos_val_m = sum(p["quantity"] * p["current_price"] for p in active_positions.values())
    eq_m = round(cash_balance + pos_val_m, 2)
    row_m = {
        "minute_num": m,
        "timestamp_ist": ts_str,
        "portfolio_equity_inr": eq_m,
        "open_positions_value_inr": round(pos_val_m, 2),
        "cash_balance_inr": round(cash_balance, 2),
        "portfolio_return_pct": round((eq_m / max(1.0, initial_cap) - 1.0) * 100.0, 3),
        "nifty50_bench_return_pct": b_info["bench_ret_pct"],
        "active_slots": len(active_positions),
    }
    for sym, p in active_positions.items():
        row_m[f"{sym}_price"] = round(p["current_price"], 2)
        row_m[f"{sym}_stop"] = round(p["stop_loss_price"], 2)
        row_m[f"{sym}_target"] = round(p["profit_target_price"], 2)

    return cash_balance, row_m


def _finalize_eod_settlement(
    target_date_str: str,
    active_positions: Dict[str, dict],
    minute_curve_rows: List[dict],
    intraday_events: List[dict],
    proof_meta: dict,
    db_path: str = DB_PATH,
) -> dict:
    """Persist 15:30 PM close state, increment `bars_held` once, write session JSON + ledger row."""
    from scripts.run_session_375m import run_stage_report

    session_file = OUT_DIR / f"session_375m_{target_date_str}.json"
    already_incremented = False
    if session_file.exists():
        try:
            prev_s = json.loads(session_file.read_text(encoding="utf-8"))
            already_incremented = str(prev_s.get("status", "")) == "COMPLETED_375M_SESSION"
        except Exception:
            already_incremented = False

    with sqlite3.connect(db_path) as conn:
        for sym, p in active_positions.items():
            if already_incremented:
                conn.execute(
                    "UPDATE open_positions SET current_price = ?, stop_loss_price = ? WHERE symbol = ?",
                    (p["current_price"], p["stop_loss_price"], sym),
                )
            else:
                conn.execute(
                    "UPDATE open_positions SET current_price = ?, stop_loss_price = ?, bars_held = bars_held + 1 WHERE symbol = ?",
                    (p["current_price"], p["stop_loss_price"], sym),
                )
            conn.execute(
                "UPDATE gtt_oco_triggers SET leg_b_current_stop_price = ? WHERE symbol = ? AND oco_pair_status LIKE 'ARMED%'",
                (p["stop_loss_price"], sym),
            )
        conn.commit()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if minute_curve_rows:
        curve_df = pd.DataFrame(minute_curve_rows).drop_duplicates(subset=["minute_num"], keep="last").sort_values("minute_num")
        curve_df.to_parquet(OUT_DIR / f"intraday_375m_curve_{target_date_str}.parquet", index=False)
        curve_df.to_parquet(OUT_DIR / "intraday_375m_curve_latest.parquet", index=False)

    cap_cfg = CapitalConfig(initial_capital_inr=100_000.0)
    vault = CryptographicRiskVault(capital_cfg=cap_cfg)
    broker = IndiaPaperBroker(vault=vault, db_path=db_path)
    final_acct = broker.get_account_summary()
    final_open = broker.get_open_positions_df()

    intraday_events.append(
        {
            "minute_num": int(minute_curve_rows[-1]["minute_num"]) if minute_curve_rows else 375,
            "timestamp_ist": f"{target_date_str} 15:30:00",
            "event_type": "🏁 SESSION_CLOSE_1530",
            "symbol": "NSE_MARKET",
            "details": f"NSE 15:30 PM Close Settled | Equity: ₹{final_acct['total_equity_inr']:,.2f} ({final_acct['total_return_pct']:+.2f}%)",
        }
    )

    final_state = {
        "session_date": target_date_str,
        "status": "COMPLETED_375M_SESSION",
        "current_minute": int(minute_curve_rows[-1]["minute_num"]) if minute_curve_rows else 375,
        "total_minutes": 375,
        "current_clock_ist": f"{target_date_str} 15:30:00 IST (Market Close Settled)",
        "portfolio_equity_inr": final_acct["total_equity_inr"],
        "portfolio_return_pct": final_acct["total_return_pct"],
        "cash_balance_inr": final_acct["cash_balance_inr"],
        "account_summary": final_acct,
        "open_positions": final_open.to_dict(orient="records") if not final_open.empty else [],
        "intraday_events": intraday_events,
        "causal_proof": proof_meta,
    }
    (OUT_DIR / "live_375m_state.json").write_text(json.dumps(final_state, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / f"session_375m_{target_date_str}.json").write_text(json.dumps(final_state, indent=2, default=str), encoding="utf-8")

    run_stage_report(target_date_str=target_date_str)
    return final_state


class LiveSessionController:
    """Thread-safe controller managing 09:15 -> 15:30 live NSE sessions and late-start reconciliation."""

    def __init__(self, on_invalidate_cache: Optional[Callable[[], None]] = None) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._on_invalidate_cache = on_invalidate_cache
        self._state: dict = {
            "status": "IDLE",
            "date": None,
            "started_at": None,
            "current_minute": 0,
            "total_minutes": 375,
            "market_clock_ist": get_ist_now().strftime("%H:%M:%S"),
            "message": "Ready to start NSE live trading session (09:15 AM – 03:30 PM IST).",
            "phase_detail": "",
            "error": None,
            "stop_requested": False,
        }

    def get_status(self) -> dict:
        with self._lock:
            snap = dict(self._state)
        now = get_ist_now()
        snap["market_clock_ist"] = now.strftime("%H:%M:%S")
        snap["today_date"] = now.strftime("%Y-%m-%d")
        snap["market_minute_now"] = market_minute_at(now)
        return snap

    def start(self, target_date_str: Optional[str] = None) -> dict:
        today = target_date_str or get_ist_now().strftime("%Y-%m-%d")
        ok_day, reason = is_trading_day(today)
        if not ok_day:
            with self._lock:
                self._state.update({"status": "MARKET_CLOSED", "date": today, "message": reason, "error": reason})
            return self.get_status()

        already_running = False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                already_running = True
            else:
                self._state = {
                    "status": "STARTING",
                    "date": today,
                    "started_at": get_ist_now().isoformat(),
                    "current_minute": market_minute_at(get_ist_now()),
                    "total_minutes": 375,
                    "market_clock_ist": get_ist_now().strftime("%H:%M:%S"),
                    "message": f"Initializing live NSE session for {today}...",
                    "phase_detail": "Checking EOD market history & today's recorded ledger",
                    "error": None,
                    "stop_requested": False,
                }
                self._thread = threading.Thread(target=self._run_loop, args=(today,), daemon=True)
                self._thread.start()
        return self.get_status()

    def stop(self) -> dict:
        with self._lock:
            self._state["stop_requested"] = True
            if self._state["status"] in ("WAITING_FOR_0915_OPEN", "STARTING", "MORNING_SCOUT", "CATCHING_UP", "LIVE"):
                self._state["message"] = "Stopping live session safely & saving state..."
        return self.get_status()

    def _update(self, **kwargs: object) -> None:
        with self._lock:
            self._state.update(kwargs)

    def _is_stop_requested(self) -> bool:
        with self._lock:
            return bool(self._state.get("stop_requested", False))

    def _interruptible_sleep(self, seconds: int) -> None:
        for _ in range(seconds):
            if self._is_stop_requested():
                break
            time.sleep(1)

    def _invalidate(self) -> None:
        if self._on_invalidate_cache is not None:
            self._on_invalidate_cache()

    def _run_loop(self, target_date_str: str) -> None:
        from scripts.run_session_375m import run_stage_morning

        try:
            self._update(phase_detail="Verifying latest prior EOD closing bars from Yahoo Finance...")
            ensure_eod_cache_fresh_before(target_date_str)
            self._invalidate()

            # Check existing today's ledger state before waiting/starting
            rec = load_recorded_day_state(target_date_str)
            if rec["already_settled_eod"] and market_minute_at(get_ist_now()) >= 376:
                self._update(
                    status="SETTLED",
                    current_minute=375,
                    message=f"Today's session ({target_date_str}) already settled cleanly at 15:30 PM IST.",
                    phase_detail="",
                )
                return

            # Pre-Market Wait (< 09:15:00 IST): Countdown automatically until NSE opens at 09:15 AM
            while market_minute_at(get_ist_now()) == 0:
                if self._is_stop_requested():
                    self._update(status="STOPPED", message="Pre-market live arm cancelled by operator.", phase_detail="")
                    return
                now_dt = get_ist_now()
                open_dt = now_dt.replace(hour=9, minute=15, second=0, microsecond=0)
                rem_sec = max(0, int((open_dt - now_dt).total_seconds()))
                hh, rem_m = divmod(rem_sec, 3600)
                mm, ss = divmod(rem_m, 60)
                self._update(
                    status="WAITING_FOR_0915_OPEN",
                    current_minute=0,
                    message=f"Armed for 09:15 AM IST NSE Open (starts automatically in {hh:02d}h:{mm:02d}m:{ss:02d}s)",
                    phase_detail="Zero trades placed pre-open. Will begin live minute feed automatically at 09:15:00 IST.",
                )
                time.sleep(1)

            cap_cfg = CapitalConfig(initial_capital_inr=100_000.0)
            vault = CryptographicRiskVault(capital_cfg=cap_cfg)
            broker = IndiaPaperBroker(vault=vault, db_path=DB_PATH)

            # Re-read recorded day state cleanly once inside trading window (>= 09:15 AM)
            rec = load_recorded_day_state(target_date_str)
            morning_done = bool(rec["morning_done"])
            last_processed_min = int(rec["last_recorded_min"])
            minute_curve_rows: List[dict] = list(rec["minute_curve_rows"])
            intraday_events: List[dict] = list(rec["intraday_events"])
            exited_today_syms: set[str] = set(rec["exited_today_syms"])
            active_positions: Dict[str, dict] = dict(rec["active_positions"])
            cash_balance: float = float(rec["cash_balance"])
            initial_cap: float = float(rec["initial_cap"])
            proof_meta: dict = dict(rec["proof_meta"])

            if not intraday_events:
                intraday_events.append(
                    {
                        "minute_num": 1,
                        "timestamp_ist": f"{target_date_str} 09:15:00",
                        "event_type": "SESSION_OPEN_915",
                        "symbol": "NSE_MARKET",
                        "details": f"NSE Live Trading Session Active ({target_date_str} 09:15:00 IST).",
                    }
                )

            rk_df = _load_or_refresh_ranked_candidates(target_date_str) if morning_done else pd.DataFrame()
            reconcile_announced = False

            while not self._is_stop_requested():
                now_ist = get_ist_now()
                m_now = market_minute_at(now_ist)
                target_fetch_min = 375 if m_now >= 376 else max(1, m_now)

                self._update(
                    phase_detail=f"Fetching real NSE candles (09:15 → Minute {target_fetch_min}/375)..."
                )
                df_15m, df_375m, max_real_min = fetch_real_intraday_up_to_now(target_date_str, max_minute=target_fetch_min)

                # If post-close and exchange has zero ticks (e.g. unannounced closed day)
                if m_now >= 376 and max_real_min == 0:
                    self._update(
                        status="MARKET_CLOSED",
                        message=f"NSE Market Closed on {target_date_str} (Zero exchange ticks available today).",
                        phase_detail="",
                    )
                    return

                # Execute Morning Scout & Orders ONCE when m_now >= 16 (09:30 AM) and 15m bar formed
                if not morning_done and target_fetch_min >= 16 and max_real_min >= 15 and not df_15m.empty:
                    self._update(
                        status="MORNING_SCOUT",
                        current_minute=target_fetch_min,
                        message="09:30 AM Morning Opening Bar Closed — Ranking 50 stocks & executing causal orders...",
                        phase_detail="Zero lookahead: scoring only on prior EOD + 09:15–09:30 AM opening candle",
                    )
                    run_stage_morning(target_date_str=target_date_str, reset_account=False, force_refresh=False)
                    morning_done = True
                    rec_post = load_recorded_day_state(target_date_str)
                    active_positions = dict(rec_post["active_positions"])
                    cash_balance = float(rec_post["cash_balance"])
                    proof_meta = dict(rec_post["proof_meta"])
                    exited_today_syms.update(rec_post["exited_today_syms"])
                    rk_df = _load_or_refresh_ranked_candidates(target_date_str)
                    intraday_events.append(
                        {
                            "minute_num": 16,
                            "timestamp_ist": f"{target_date_str} 09:30:00",
                            "event_type": "MORNING_PORTFOLIO_LOCKED",
                            "symbol": "PORTFOLIO",
                            "details": f"09:30 AM Orders Locked: {list(active_positions.keys())} | Cash: ₹{cash_balance:,.2f}",
                        }
                    )
                    _persist_live_progress(
                        target_date_str,
                        max(16, last_processed_min),
                        "MORNING_PORTFOLIO_LOCKED",
                        active_positions,
                        minute_curve_rows,
                        intraday_events,
                        proof_meta,
                        cash_balance,
                        initial_cap,
                    )
                    self._invalidate()

                # Process all unseen real minutes [last_processed_min + 1 .. max_real_min]
                if max_real_min > last_processed_min and not df_375m.empty:
                    is_catchup = (max_real_min - last_processed_min) > 2
                    if is_catchup and not reconcile_announced:
                        reconcile_announced = True
                        self._update(
                            status="CATCHING_UP",
                            current_minute=last_processed_min,
                            message=(
                                f"Reconciling recorded trades from Minute {last_processed_min + 1} → {max_real_min} "
                                f"using real NSE candles..."
                            ),
                            phase_detail=f"Active SQLite holdings: {list(active_positions.keys()) or 'Cash'}",
                        )

                    held_syms = list(active_positions.keys())
                    cand_syms = rk_df.head(15)["symbol"].astype(str).tolist() if not rk_df.empty else []
                    tracked = list(dict.fromkeys(held_syms + cand_syms))
                    bar_lookup, nifty_map = _build_bar_and_nifty_lookup(df_375m, tracked, target_date_str)

                    for m in range(last_processed_min + 1, max_real_min + 1):
                        if self._is_stop_requested():
                            break
                        b_info = nifty_map.get(
                            m,
                            {
                                "timestamp": (pd.Timestamp(f"{target_date_str} 09:15:00") + timedelta(minutes=m - 1)).strftime("%Y-%m-%d %H:%M:%S"),
                                "bench_ret_pct": 0.0,
                            },
                        )
                        cash_balance, row_m = process_single_minute(
                            m=m,
                            target_date_str=target_date_str,
                            sym_bars=bar_lookup.get(m, {}),
                            b_info=b_info,
                            active_positions=active_positions,
                            exited_today_syms=exited_today_syms,
                            cash_balance=cash_balance,
                            initial_cap=initial_cap,
                            rk_df=rk_df,
                            vault=vault,
                            broker=broker,
                            cap_cfg=cap_cfg,
                            intraday_events=intraday_events,
                            mode_tag="CATCHUP" if is_catchup else "LIVE",
                        )
                        minute_curve_rows.append(row_m)
                        last_processed_min = m

                    _persist_live_progress(
                        target_date_str,
                        last_processed_min,
                        "RUNNING_1X_REALTIME_SYNCED",
                        active_positions,
                        minute_curve_rows,
                        intraday_events,
                        proof_meta,
                        cash_balance,
                        initial_cap,
                    )
                    self._invalidate()

                # Check automatic 15:30 PM session close
                if m_now >= 376:
                    self._update(
                        status="SETTLING",
                        current_minute=375,
                        message="15:30 PM IST Market Close Reached — Automatically settling EOD ledger & contract notes...",
                        phase_detail="Finalizing post-tax PnL & incrementing daily holding count",
                    )
                    _finalize_eod_settlement(
                        target_date_str=target_date_str,
                        active_positions=active_positions,
                        minute_curve_rows=minute_curve_rows,
                        intraday_events=intraday_events,
                        proof_meta=proof_meta,
                    )
                    self._invalidate()
                    self._update(
                        status="SETTLED",
                        current_minute=375,
                        message=f"Session {target_date_str} automatically closed & settled at 15:30 PM IST.",
                        phase_detail="",
                    )
                    return

                self._update(
                    status="LIVE",
                    current_minute=last_processed_min or m_now,
                    message=f"Live Trading Active • Min {last_processed_min or m_now}/375 ({now_ist.strftime('%H:%M:%S')} IST) — Auto-ends at 15:30 PM",
                    phase_detail=f"Tracking {len(active_positions)} open positions on real NSE candles (polling every 25s)",
                )
                self._interruptible_sleep(25)

            # Clean operator manual stop before 15:30 PM
            _persist_live_progress(
                target_date_str,
                last_processed_min,
                "PAUSED_BY_OPERATOR",
                active_positions,
                minute_curve_rows,
                intraday_events,
                proof_meta,
                cash_balance,
                initial_cap,
            )
            self._invalidate()
            self._update(
                status="STOPPED",
                current_minute=last_processed_min,
                message=f"Live session paused cleanly at Minute {last_processed_min}/375. Resume anytime to continue from Minute {last_processed_min + 1}.",
                phase_detail="",
            )

        except Exception as exc:
            self._update(
                status="ERROR",
                error=str(exc),
                message=f"Live session error: {str(exc)[:180]}",
                phase_detail="",
            )
