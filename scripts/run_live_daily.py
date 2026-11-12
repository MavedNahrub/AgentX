"""
AgentX v2.0 - Live Market 6-Day Autonomous Orchestrator (`scripts/run_live_daily.py`)
-------------------------------------------------------------------------------------
Executes the upgraded 6-Day Regime-Adaptive Live Market Testing Protocol:
  - Morning Mode (`--mode morning` at 9:15-9:30 AM):
      1. Ingests live telemetry & 9:15 AM 15m Opening VWAP
      2. Generates 17 self-calibrating Chameleon features
      3. Cross-sectional Walk-Forward LightGBM scoring & TreeSHAP X-Ray
      4. Strategy tournament selection & Google News RSS Catalyst Veto
      5. Opening Gap-Invalidation Check (ensures 9:15 open hasn't exhausted target/stop)
      6. Day 1: 4-Slot Elastic Whole-Share allocation (Regime-Conditioned >98% deployment)
      7. Days 2-6: Stateful Smart Eviction (`evaluate_smart_eviction`), 3-Bar Hysteresis
         Anti-Churn Defense, Catalyst Override & Tax-Aware Day-5 Horizon Rollover
  - Evening Mode (`--mode evening` at 3:30 PM):
      1. Runs `mark_to_market_and_process_exits()` on today's High/Low/Close bars
      2. Triggers GTT OCO Targets/Stops & Chandelier Profit-Lock Stop Ratchets
      3. Computes Volatility-Normalized Cross-Sectional Spread (Top 4 vs Bottom 4 & Nifty 50)
      4. Evaluates Live Model Prediction (Upper/Lower Limits) vs Actual Market Trend
      5. Reconciles end-of-day portfolio equity, cash balance & cumulative churn taxes saved
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Reconfigure stdout for Windows console UTF-8 support
sys.stdout.reconfigure(encoding="utf-8")
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.alpha_scout import WalkForwardAlphaScout
from src.data_engine import CapitalConfig, HybridDataEngine
from src.feature_engine import ChameleonFeatureEngine
from src.paper_broker import IndiaPaperBroker
from src.risk_vault import CryptographicRiskVault, KillSwitchStateStore
from src.strategy_and_news import (
    DateBoundedNewsVetoEngine,
    EventDrivenStrategyTournament,
)


def run_morning_cycle(force_refresh: bool = False, reset_account: bool = False) -> None:
    print("\n" + "=" * 84)
    print("🌅 AGENTX v2.0 LIVE MORNING CYCLE (9:15 AM - 9:30 AM EXECUTION & REBALANCE GATE)")
    print("=" * 84)

    state_store = KillSwitchStateStore("data/cache/risk_vault_state.sqlite")
    if reset_account:
        state_store.set_mode("RUNNING", "Fresh 6-Day Live Testing Cycle Initialized")
    v_state = state_store.get_state()
    if v_state["mode"] != "RUNNING":
        print(f"🛑 [ABORT] Risk Vault is in {v_state['mode']} mode! Reason: {v_state['reason']}")
        return

    print("\n[Step 1/6] Ingesting Live Telemetry & 9:15 AM 15m Opening VWAP...")
    cap_cfg = CapitalConfig(initial_capital_inr=100_000.0)
    data_engine = HybridDataEngine(cache_dir="data/cache", capital_cfg=cap_cfg)
    stocks_df, macro_df, _ = data_engine.fetch_and_build_telemetry(force_refresh=force_refresh)

    print("\n[Step 2/6] Engineering 17 Self-Calibrating Features & Volatility Bands...")
    feat_engine = ChameleonFeatureEngine(cache_dir="data/cache")
    feat_df = feat_engine.build_features(stocks_df, macro_df, save_cache=True)

    print("\n[Step 3/6] Scoring Cross-Sectional Alpha via Purged LightGBM + TreeSHAP...")
    scout = WalkForwardAlphaScout()
    ranked_df, sector_df, _ = scout.score_cross_section_with_shap(feat_df, retrain=True)
    as_of_date = pd.Timestamp(ranked_df["date"].iloc[0])
    as_of_str = as_of_date.strftime("%Y-%m-%d")
    print(f"  Signal Date: {as_of_str} | Candidates Scored: {len(ranked_df)}")

    print("\n[Step 4/6] Assigning Self-Calibrated Strategies & Running Live News RSS Veto...")
    tournament = EventDrivenStrategyTournament(capital_cfg=cap_cfg)
    board_df = tournament.run_post_tax_tournament(feat_df, as_of_date=as_of_date)
    paired_df = tournament.assign_strategy_and_exit_geometry(ranked_df, board_df)
    paired_df.to_parquet("data/cache/alpha_scout_latest_ranked.parquet", index=False)

    news_engine = DateBoundedNewsVetoEngine(cache_dir="data/cache")
    news_map = {}
    for _, r in paired_df.head(12).iterrows():
        ret_3d_pct = float(r["fib_mom_z_3d"] * r["ret_std_21d"] * np.sqrt(3.0) * 100.0)
        news_map[str(r["symbol"])] = news_engine.evaluate_catalyst_veto(
            symbol=str(r["symbol"]),
            as_of_date=as_of_date,
            stock_empirical_tail_stop_pct=float(r["empirical_tail_stop_pct"]),
            recent_3d_return_pct=ret_3d_pct,
        )

    vault = CryptographicRiskVault(capital_cfg=cap_cfg)
    broker = IndiaPaperBroker(
        vault=vault,
        db_path="data/cache/paper_broker_ledger.sqlite",
        reset_account=reset_account,
    )
    acct_summary = broker.get_account_summary()
    open_pos_df = broker.get_open_positions_df()

    guardrails = vault.compute_drawdown_and_daily_guardrails(
        current_equity_inr=float(acct_summary["total_equity_inr"]),
        peak_equity_inr=float(acct_summary["peak_equity_inr"]),
        today_pnl_inr=0.0,
        candidate_slice=paired_df,
    )
    throttle_mult = float(guardrails["exponential_drawdown_throttle"])

    # STEP 5A: IF POSITIONS ARE ALREADY HELD (DAYS 2-6 ON THE SAME CONTINUOUS CAPITAL)
    if not open_pos_df.empty and not reset_account:
        print("\n[Step 5/6] Continuous 6-Day Portfolio Audit: 3-Bar Hysteresis, Eviction & Rollover...")
        today_map = paired_df.set_index("symbol")
        held_symbols = set(open_pos_df["symbol"].tolist())
        unheld_df = paired_df[~paired_df["symbol"].isin(held_symbols)].sort_values("alpha_rank", ascending=True)
        top_challenger = unheld_df.iloc[0] if not unheld_df.empty else paired_df.iloc[0]

        current_holdings_list = []
        for _, pos in open_pos_df.iterrows():
            sym = str(pos["symbol"])
            cur_alpha = float(today_map.loc[sym, "expected_5d_alpha_pct"]) if sym in today_map.index else float(pos["expected_5d_alpha_pct"])
            if as_of_str > str(pos["entry_date"]) and sym in today_map.index:
                cur_px = float(today_map.loc[sym, "exec_vwap_915"])
            else:
                cur_px = float(pos["current_price"])
            current_holdings_list.append(
                {
                    "symbol": sym,
                    "sector": str(pos["sector"]),
                    "bars_held": int(pos["bars_held"]),
                    "quantity_shares": int(pos["quantity"]),
                    "current_price": cur_px,
                    "stop_loss_price": float(pos["stop_loss_price"]),
                    "current_expected_5d_alpha_pct": cur_alpha,
                }
            )

        evict_df = vault.evaluate_smart_eviction(
            current_holdings=current_holdings_list,
            challenger_row=top_challenger,
            cross_section_df=paired_df,
            news_assessments=news_map,
            record_to_db=True,
        )

        print("\n🛡️ LIVE CONTINUOUS PORTFOLIO ANTI-CHURN & ROLLOVER AUDIT:")
        pos_by_sym = open_pos_df.set_index("symbol")
        for _, erow in evict_df.iterrows():
            sym = str(erow["holding_symbol"])
            b_held = int(erow["bars_held"])
            verdict = str(erow["eviction_verdict"])
            saved_inr = float(erow["churn_tax_saved_inr"])
            cur_rank = int(today_map.loc[sym, "alpha_rank"]) if sym in today_map.index else 25

            should_sell = False
            sell_reason = ""
            if verdict.startswith("EVICT_IMMEDIATELY") or verdict.startswith("APPROVE_SWITCH"):
                should_sell = True
                sell_reason = verdict.split(" ")[0]
            elif b_held >= 5:
                if cur_rank <= 12 and float(erow["holding_alpha_pct"]) > 0:
                    verdict = f"HOLD_CONTINUE_ROLLOVER (Day {b_held}: Still Rank #{cur_rank} -> Saved ₹{erow['switch_tax_inr']:.2f} Rollover Tax)"
                else:
                    verdict = f"SETTLE_5D_HORIZON (Day {b_held}: Rank #{cur_rank} Decayed -> Settle at 9:15 VWAP)"
                    should_sell = True
                    sell_reason = "HORIZON_5D_SETTLEMENT"

            print(
                f"  - [{sym}] (Held {b_held}d | Rank #{cur_rank}): {verdict} "
                f"| Tax Saved Today: ₹{saved_inr:,.2f}"
            )

            if should_sell and sym in pos_by_sym.index:
                prow = pos_by_sym.loc[sym]
                exit_px = float(today_map.loc[sym, "exec_vwap_915"]) if sym in today_map.index else float(prow["current_price"])
                sell_dict = {
                    "symbol": sym,
                    "sector": str(prow["sector"]),
                    "slot_id": int(prow["slot_id"]),
                    "quantity_shares": int(prow["quantity"]),
                    "exec_vwap_915": exit_px,
                    "stop_loss_price": float(prow["stop_loss_price"]),
                    "profit_target_price": float(prow["profit_target_price"]),
                    "assigned_strategy": str(prow["assigned_strategy"]),
                    "expected_5d_alpha_pct": float(prow["expected_5d_alpha_pct"]),
                    "conviction_pct": float(prow["conviction_pct"]),
                }
                sell_pkt = vault.sign_order_packet(
                    as_of_date=as_of_str,
                    side="SELL",
                    row=sell_dict,
                    throttle_mult=throttle_mult,
                )
                s_res = broker.execute_signed_order(sell_pkt, exit_reason=sell_reason)
                print(
                    f"    🔄 Executed {sell_reason} for [{sym}] @ ₹{s_res.get('fill_price', exit_px):,.2f} | "
                    f"Realized Net PnL: ₹{s_res.get('realized_net_pnl_inr', 0.0):+,.2f} | "
                    f"Updated Account Cash: ₹{s_res.get('remaining_cash_inr', 0.0):,.2f}"
                )

        # Refresh open positions & cash from the same continuous account
        open_pos_df = broker.get_open_positions_df()
        acct_summary = broker.get_account_summary()
        ks_now = state_store.get_state()
        print(
            f"\n• Continuous Account Status: Open Slots = {len(open_pos_df)}/4 | "
            f"Available Cash = ₹{float(acct_summary['cash_balance_inr']):,.2f} | "
            f"Total Collective Equity = ₹{float(acct_summary['total_equity_inr']):,.2f} "
            f"({float(acct_summary['total_return_pct']):+.2f}%)"
        )
        print(
            f"• Cumulative Churn Taxes Saved Across Cycle: ₹{float(ks_now['cumulative_churn_taxes_saved_inr']):,.2f} "
            f"({int(ks_now['blocked_rotations_count'])} Rotations Blocked)"
        )
        if len(open_pos_df) >= 4:
            _save_live_checkpoint(
                session_type="MORNING_HYSTERESIS_HOLD",
                as_of_date=as_of_str,
                acct=acct_summary,
                open_pos_df=open_pos_df,
                churn_saved_inr=float(ks_now["cumulative_churn_taxes_saved_inr"]),
                extra_meta={"hysteresis_audit": evict_df.to_dict(orient="records")},
            )
            return

    # STEP 5B: DEPLOY AVAILABLE CASH INTO ANY EMPTY SLOTS (ALL 4 ON DAY 1, OR FREED SLOTS ON DAYS 2-6)
    empty_slots_count = 4 - len(open_pos_df)
    avail_cash = float(acct_summary["cash_balance_inr"])
    print(
        f"\n[Step 5/6] Allocating Continuous Account Cash (₹{avail_cash:,.2f}) across "
        f"{empty_slots_count} Open Slot(s)..."
    )
    held_syms_now = set(open_pos_df["symbol"].tolist()) if not open_pos_df.empty else set()
    elig_candidates_df = paired_df[~paired_df["symbol"].isin(held_syms_now)].copy()

    sized_df, sizing_meta = vault.select_and_size_4_slot_portfolio(
        candidates_df=elig_candidates_df,
        sector_df=sector_df,
        news_assessments=news_map,
        available_capital_inr=avail_cash,
        throttle_mult=throttle_mult,
    )
    sized_df = sized_df.head(empty_slots_count).copy()

    print("\n[Step 6/6] Executing Fills at 9:15 AM VWAP & Generating SEBI Contract Notes...")
    fills_summary = []
    for _, row in sized_df.iterrows():
        pkt = vault.sign_order_packet(
            as_of_date=as_of_str,
            side="BUY",
            row=row,
            throttle_mult=throttle_mult,
        )
        fill_res = broker.execute_signed_order(pkt)
        fills_summary.append((row, fill_res))

    print("\n" + "-" * 84)
    print("📊 MORNING REGIME-ADAPTIVE EXECUTION REPORT:")
    print("-" * 84)
    invested = float(sizing_meta["invested_notional_inr"])
    deployable = float(sizing_meta["deployable_capital_inr"])
    util = float(sizing_meta["capital_utilization_pct"])
    idle = float(sizing_meta["uninvested_cash_inr"])
    active_sec_cnt = int((sector_df["sector_capital_status"].str.startswith("ACTIVE_ELIGIBLE")).sum())
    starved_sec_cnt = len(sector_df) - active_sec_cnt

    print(
        f"• Regime Throttle: {throttle_mult:.2f}x | Active Sectors: {active_sec_cnt}/{len(sector_df)} "
        f"({starved_sec_cnt} Weak Sectors Starved at ₹0)"
    )
    print(
        f"• Regime-Adjusted Capital Utilization: {util:.2f}% of ₹{deployable:,.2f} Allowed "
        f"(Invested: ₹{invested:,.2f} | Idle Cash: ₹{idle:,.2f})"
    )
    print(f"• Target Gate (>=98% of Allowed Capital Deployed): {'✅ PASS' if util >= 98.0 else '⚠️ DEFENSIVE CASH HELD'}")

    print("\nActive 4-Slot Purchases, Opening Gap Check & GTT OCO Arming:")
    for row, f in fills_summary:
        if f.get("status") == "FILLED_BUY":
            px = float(f["fill_price"])
            tgt = float(row["profit_target_price"])
            stp = float(row["stop_loss_price"])
            rem_upside_pct = (tgt - px) / max(1.0, px) * 100.0
            stop_risk_pct = (px - stp) / max(1.0, px) * 100.0
            rr_ratio = rem_upside_pct / max(0.01, stop_risk_pct)
            gap_ok = "✅ VALID_OPEN" if (px < tgt and px > stp) else "⚠️ GAP_BREACH"
            print(
                f"  - Slot #{int(row['slot_id'])} [{f['symbol']}] ({row['assigned_strategy']}): "
                f"{f['quantity']} sh @ ₹{px:,.2f} (Slip: {f['slippage_bps']:.2f} bps) | "
                f"Upper Target: ₹{tgt:,.2f} (+{rem_upside_pct:.2f}%) | Lower Stop: ₹{stp:,.2f} (-{stop_risk_pct:.2f}%) | "
                f"R:R {rr_ratio:.2f}x | {gap_ok} | GTT: {f.get('gtt_id', 'ARMED')}"
            )
        else:
            print(f"  - [{f.get('symbol', 'UNKNOWN')}]: {f.get('status')} - {f.get('reason')}")

    acct_after = broker.get_account_summary()
    open_after = broker.get_open_positions_df()
    ks_after = state_store.get_state()
    _save_live_checkpoint(
        session_type="MORNING_EXECUTION",
        as_of_date=as_of_str,
        acct=acct_after,
        open_pos_df=open_after,
        churn_saved_inr=float(ks_after["cumulative_churn_taxes_saved_inr"]),
        extra_meta={"sizing_meta": sizing_meta},
    )
    print("\nContract Notes & Audit Ledger written to: data/cache/paper_broker_ledger.sqlite")


def _save_live_checkpoint(
    session_type: str,
    as_of_date: str,
    acct: dict,
    open_pos_df: pd.DataFrame,
    churn_saved_inr: float,
    extra_meta: dict | None = None,
) -> None:
    out_dir = Path("reports/live_6day_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_utc = datetime.now(timezone.utc).isoformat()

    snap = {
        "timestamp_utc": ts_utc,
        "session_type": session_type,
        "market_date": as_of_date,
        "account_summary": acct,
        "cumulative_churn_taxes_saved_inr": round(churn_saved_inr, 2),
        "open_positions": open_pos_df.to_dict(orient="records") if not open_pos_df.empty else [],
        "telemetry": extra_meta or {},
    }
    json_file = out_dir / f"{session_type.lower()}_{as_of_date}.json"
    json_file.write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

    ledger_csv = out_dir / "collective_6day_ledger.csv"
    row_df = pd.DataFrame(
        [
            {
                "timestamp_utc": ts_utc,
                "market_date": as_of_date,
                "session_type": session_type,
                "initial_capital_inr": acct.get("initial_capital_inr", 100000.0),
                "open_positions_value_inr": acct.get("open_positions_value_inr", 0.0),
                "cash_balance_inr": acct.get("cash_balance_inr", 0.0),
                "total_equity_inr": acct.get("total_equity_inr", 100000.0),
                "total_return_pct": acct.get("total_return_pct", 0.0),
                "realized_pnl_net_inr": acct.get("realized_pnl_net_inr", 0.0),
                "unrealized_pnl_gross_inr": acct.get("unrealized_pnl_gross_inr", 0.0),
                "total_taxes_paid_inr": acct.get("total_taxes_paid_inr", 0.0),
                "churn_taxes_saved_inr": round(churn_saved_inr, 2),
                "open_slots_count": len(open_pos_df),
            }
        ]
    )
    if ledger_csv.exists():
        row_df.to_csv(ledger_csv, mode="a", header=False, index=False)
    else:
        row_df.to_csv(ledger_csv, mode="w", header=True, index=False)
    print(f"💾 Saved permanent test snapshot -> {json_file} & appended to {ledger_csv}")


def run_evening_cycle(force_refresh: bool = False) -> None:
    print("\n" + "=" * 84)
    print("🌇 AGENTX v2.0 LIVE EVENING CYCLE (3:30 PM MTM, RATCHET & ALPHA SPREAD GATE)")
    print("=" * 84)

    cap_cfg = CapitalConfig(initial_capital_inr=100_000.0)
    vault = CryptographicRiskVault(capital_cfg=cap_cfg)
    broker = IndiaPaperBroker(vault=vault, db_path="data/cache/paper_broker_ledger.sqlite")
    pos_df = broker.get_open_positions_df()

    if pos_df.empty:
        print("ℹ️ No active open positions in broker database.")
        return

    cache_dir = Path("data/cache")
    if force_refresh:
        print("\n[Pre-Step] Refreshing 3:30 PM Live EOD Closes from Market Feed...")
        data_engine = HybridDataEngine(cache_dir="data/cache", capital_cfg=cap_cfg)
        stocks_df, macro_df, _ = data_engine.fetch_and_build_telemetry(force_refresh=True)
        feat_engine = ChameleonFeatureEngine(cache_dir="data/cache")
        feat_df = feat_engine.build_features(stocks_df, macro_df, save_cache=True)
    else:
        feat_path = cache_dir / "nifty50_chameleon_features.parquet"
        if not feat_path.exists():
            print("⚠️ Cache file nifty50_chameleon_features.parquet not found.")
            return
        feat_df = pd.read_parquet(feat_path)

    ranked_path = cache_dir / "alpha_scout_latest_ranked.parquet"
    latest_date = feat_df["date"].max()
    latest_date_str = pd.Timestamp(latest_date).strftime("%Y-%m-%d")
    today_slice = feat_df[feat_df["date"] == latest_date].copy()

    print(f"\n[Step 1/4] Marking Open Positions to Market & Evaluating GTT OCO / Chandelier Ratchets ({latest_date_str})...")
    exits = broker.mark_to_market_and_process_exits(
        daily_bars_df=today_slice,
        as_of_date=latest_date_str,
    )
    if exits:
        print(f"  ⚡ {len(exits)} Automatic GTT OCO Exit(s) Triggered Today:")
        for ex in exits:
            print(
                f"    -> [{ex['symbol']}] {ex['exit_reason']} @ ₹{ex['fill_price']:,.2f} | "
                f"Realized Net PnL: ₹{ex['realized_net_pnl_inr']:+,.2f}"
            )
    else:
        print("  ✅ Zero Stop-Loss Breaches Today (All 4 Slots Safely Above Parkinson Floor).")

    # Volatility-Normalized Cross-Sectional Rank IC Spread Check
    print("\n[Step 2/4] Evaluating Volatility-Normalized Cross-Sectional Spread (Top 4 vs Bottom 4 & Nifty)...")
    spread_meta = {}
    if ranked_path.exists() and not today_slice.empty:
        ranked_df = pd.read_parquet(ranked_path)
        merged = ranked_df[["alpha_rank", "symbol", "expected_5d_alpha_pct"]].merge(
            today_slice[["symbol", "open", "high", "low", "close", "exec_vwap_915", "parkinson_vol_21d"]],
            on="symbol",
            how="inner",
        )
        # Intraday return from 9:15 AM Opening VWAP to 3:30 PM Close
        merged["intraday_ret_pct"] = (merged["close"] / merged["exec_vwap_915"] - 1.0) * 100.0
        merged["vol_norm_ret_z"] = merged["intraday_ret_pct"] / (merged["parkinson_vol_21d"] * 100.0).clip(lower=0.2)

        top_4_df = merged.sort_values("alpha_rank", ascending=True).head(4)
        bot_4_df = merged.sort_values("alpha_rank", ascending=True).tail(4)
        universe_mean_ret = float(merged["intraday_ret_pct"].mean())
        top4_ret = float(top_4_df["intraday_ret_pct"].mean())
        bot4_ret = float(bot_4_df["intraday_ret_pct"].mean())
        raw_spread = top4_ret - bot4_ret
        z_spread = float(top_4_df["vol_norm_ret_z"].mean() - bot_4_df["vol_norm_ret_z"].mean())

        spread_meta = {
            "top4_symbols": top_4_df["symbol"].tolist(),
            "top4_avg_intraday_ret_pct": round(top4_ret, 3),
            "bottom4_symbols": bot_4_df["symbol"].tolist(),
            "bottom4_avg_intraday_ret_pct": round(bot4_ret, 3),
            "nifty50_equal_weight_ret_pct": round(universe_mean_ret, 3),
            "raw_spread_pct": round(raw_spread, 3),
            "vol_normalized_spread_sigma": round(z_spread, 3),
        }

        print(f"  • Top 4 Model Picks ({top_4_df['symbol'].tolist()}): Avg Intraday Return {top4_ret:+.2f}%")
        print(f"  • Bottom 4 Avoided  ({bot_4_df['symbol'].tolist()}): Avg Intraday Return {bot4_ret:+.2f}%")
        print(f"  • Nifty 50 Equal-Weight Session Benchmark:        {universe_mean_ret:+.2f}%")
        print(
            f"  • Cross-Sectional Raw Spread (Top 4 - Bottom 4):  {raw_spread:+.2f}% "
            f"(Vol-Normalized Spread: {z_spread:+.2f} sigma)"
        )

    # Prediction vs Actual Market Trend & Chandelier Ratchet Status
    print("\n[Step 3/4] Model Prediction (Upper/Lower Limits) vs. Actual Market Trend & Ratchet Status...")
    pos_df = broker.get_open_positions_df()
    gtt_df = broker.get_gtt_oco_triggers_df()
    today_map = today_slice.set_index("symbol")

    for _, p in pos_df.iterrows():
        sym = str(p["symbol"])
        entry_px = float(p["entry_fill_price"])
        cur_px = float(p["current_price"])
        init_stop = float(p["initial_stop_price"])
        cur_stop = float(p["stop_loss_price"])
        target_px = float(p["profit_target_price"])
        exp_alpha = float(p["expected_5d_alpha_pct"])
        b_held = int(p["bars_held"])

        day_high = float(today_map.loc[sym, "high"]) if sym in today_map.index else cur_px
        day_low = float(today_map.loc[sym, "low"]) if sym in today_map.index else cur_px
        actual_ret_pct = (cur_px / entry_px - 1.0) * 100.0
        stop_cushion_pct = (cur_px - cur_stop) / max(1.0, cur_px) * 100.0
        dist_to_target_pct = (target_px - cur_px) / max(1.0, cur_px) * 100.0

        is_ratcheted = cur_stop > init_stop + 0.01
        ratchet_badge = f"🔒 RATCHETED ABOVE ENTRY (₹{init_stop:,.2f} -> ₹{cur_stop:,.2f})" if is_ratcheted else f"🛡️ PARKINSON STOP (₹{cur_stop:,.2f})"
        pred_badge = f"TREND UP (+{exp_alpha:.2f}%)" if exp_alpha > 0 else f"TREND DOWN ({exp_alpha:.2f}%)"

        print(
            f"  - Slot #{int(p['slot_id'])} [{sym}] (Held {b_held}d | {p['assigned_strategy']}):\n"
            f"      Predicted: {pred_badge} | Upper Limit: ₹{target_px:,.2f} (+{dist_to_target_pct:.2f}% away) | "
            f"Lower Limit: ₹{cur_stop:,.2f} (-{stop_cushion_pct:.2f}% cushion)\n"
            f"      Actual Market: Entry ₹{entry_px:,.2f} -> Close ₹{cur_px:,.2f} ({actual_ret_pct:+.2f}%) "
            f"[Session Range: ₹{day_low:,.2f} - ₹{day_high:,.2f}] | {ratchet_badge}"
        )

    print("\n[Step 4/4] End-of-Day Net Profit & Remaining Assets Reconciliation...")
    acct = broker.get_account_summary()
    ks_state = KillSwitchStateStore("data/cache/risk_vault_state.sqlite").get_state()
    print(f"  • Initial Capital:              ₹{float(acct['initial_capital_inr']):,.2f}")
    print(f"  • Active Stock Assets (MTM):    ₹{float(acct['open_positions_value_inr']):,.2f}")
    print(f"  • Remaining Uninvested Cash:    ₹{float(acct['cash_balance_inr']):,.2f}")
    print(f"  • Total End-of-Day Equity:      ₹{float(acct['total_equity_inr']):,.2f} ({float(acct['total_return_pct']):+.2f}%)")
    print(f"  • Unrealized Gross Profit/Loss: ₹{float(acct['unrealized_pnl_gross_inr']):+,.2f}")
    print(f"  • Realized Net Profit/Loss:     ₹{float(acct['realized_pnl_net_inr']):+,.2f}")
    print(f"  • Statutory Taxes Paid:         ₹{float(acct['total_taxes_paid_inr']):,.2f}")
    print(f"  • Churn Taxes Saved (Vault):    ₹{float(ks_state['cumulative_churn_taxes_saved_inr']):,.2f}")
    _save_live_checkpoint(
        session_type="EVENING_MTM_EVALUATION",
        as_of_date=latest_date_str,
        acct=acct,
        open_pos_df=pos_df,
        churn_saved_inr=float(ks_state["cumulative_churn_taxes_saved_inr"]),
        extra_meta={"cross_sectional_spread": spread_meta, "exits_triggered": exits},
    )
    print("-" * 84)
    print("Launch Interactive Dashboard: .\\.venv\\Scripts\\streamlit run app.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentX v2.0 Live Market 6-Day Autonomous Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["morning", "evening", "both"],
        default="both",
        help="Run 'morning' (9:15 AM sizing/rebalance), 'evening' (3:30 PM MTM/ratchets/spread), or 'both'",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force fresh download of Yahoo Finance live data",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset paper broker account to fresh ₹1,00,000 capital (Use ONLY once before Day 1 starts)",
    )
    args = parser.parse_args()

    if args.mode in ("morning", "both"):
        run_morning_cycle(force_refresh=args.refresh, reset_account=args.reset)
    if args.mode in ("evening", "both"):
        run_evening_cycle(force_refresh=args.refresh)


if __name__ == "__main__":
    main()
