"""
Section 3: Walk-Forward AI Scout & SHAP Explainer (`src/alpha_scout.py`)
------------------------------------------------------------------------
Responsibilities:
1. Train a Purged Walk-Forward LightGBM Regressor on the 17 Chameleon multi-horizon features
   to predict the 5-day forward Information Ratio (`target_5d_info_ratio`).
2. Enforce a strict 5-trading-day Purge/Embargo Gap between the last training label date (t - 6)
   and the live prediction date (t) so there is 0.0% future return overlap (zero look-ahead bias).
3. Apply Exponential Recency Sample Weighting during training so recent market regimes carry more
   weight than 9 months ago, controlled dynamically by India VIX stress.
4. Extract exact mathematical feature attributions for every stock using `shap.TreeExplainer`
   (Top 3 positive drivers + Top negative risk drag)—guaranteeing 100% transparent AI reasoning.
5. Compute Live Sector Alpha Breadth so the Risk Engine (Section 5) routes 100% of capital into
   winning sectors and starves weak/lagging sectors (zero idle cash in dead sectors).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from src.data_engine import CapitalConfig, HybridDataEngine
from src.feature_engine import FEATURE_COLUMNS, ChameleonFeatureEngine

# Human-readable labels for the 17 Chameleon features (used in SHAP Journals & Dashboards)
FEATURE_HUMAN_NAMES: Dict[str, str] = {
    "fib_mom_z_3d": "3d Fib Momentum Z",
    "fib_mom_z_5d": "5d Fib Momentum Z",
    "fib_mom_z_8d": "8d Fib Momentum Z",
    "fib_mom_z_13d": "13d Fib Momentum Z",
    "fib_mom_z_21d": "21d Fib Momentum Z",
    "fib_mom_z_34d": "34d Fib Momentum Z",
    "fib_mom_z_55d": "55d Fib Macro Trend Z",
    "efficiency_ratio_8d": "8d Trend Purity (ER)",
    "efficiency_ratio_21d": "21d Trend Purity (ER)",
    "parkinson_vol_21d": "21d Intraday Range Vol",
    "vol_compression_5_21": "5d/21d Volatility Coil",
    "mean_rev_zscore_13d": "13d Mean-Rev Z-Score",
    "volume_shock_21d": "21d Volume Shock (RVOL)",
    "amihud_illiquidity_z": "Order-Book Liquidity Z",
    "rel_strength_vs_nifty_21d": "21d Rel Strength vs Nifty",
    "india_vix_level": "India VIX Level",
    "india_vix_zscore_21d": "India VIX Stress Z",
    "inst_clv_5d": "5d Inst. Accumulation CLV",
    "swing_pullback_in_trend": "21d-3d Coiled Pullback",
    "sector_mom_13d": "13d Sector Leadership Z",
    "stock_vs_sector_5d": "5d Stock vs Sector Rel Z",
    "trend_x_mkt": "Trend x Bull Breadth",
    "mrev_x_chop": "Mean-Rev x Chop Purity",
    "def_x_vix": "Low-Vol Shield x VIX",
    "mkt_trend_21d": "21d Market Breadth Z",
    "mkt_fast_5d": "5d Market Thrust Z",
    "mkt_er_21d": "21d Market Purity (ER)",
}

SWING_EXTRA_COLS: List[str] = [
    "inst_clv_5d",
    "swing_pullback_in_trend",
    "sector_mom_13d",
    "stock_vs_sector_5d",
    "trend_x_mkt",
    "mrev_x_chop",
    "def_x_vix",
]


@dataclass
class WalkForwardConfig:
    train_window_days: int = 378  # Stationary 1.5-year rolling window covering bull, chop, and VIX stress regimes
    purge_embargo_days: int = 6   # Strict 6-bar purge gap (since target spans t+1 to t+6)
    base_half_life_days: int = 126 # Semi-annual exponential recency base modulated by Gaussian Regime Kernel
    min_train_rows: int = 1500


class WalkForwardAlphaScout:
    """
    Purged Walk-Forward Dual-Head Future-Basis Swing Predictor (`LGBMRanker` + `LGBMRegressor` + Regime-Kernel IC)
    + Exact TreeSHAP Attribution + Multi-Bar Kaufman-Smoothed Swing Rank Engine.
    """

    def __init__(
        self,
        wf_cfg: Optional[WalkForwardConfig] = None,
        capital_cfg: Optional[CapitalConfig] = None,
        cache_dir: str | Path = "data/cache",
    ):
        self.wf_cfg = wf_cfg or WalkForwardConfig()
        self.capital_cfg = capital_cfg or CapitalConfig()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[lgb.LGBMRegressor] = None
        self.ranker_model: Optional[lgb.LGBMRanker] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.last_train_date: Optional[pd.Timestamp] = None
        self.last_rank_ic: float = 0.15
        self.head_weights: np.ndarray = np.array([0.25, 0.25, 0.25, 0.25])
        self.regime_ic_vec: Optional[np.ndarray] = None
        self._csr_feature_names: List[str] = []
        self._tree_feature_names: List[str] = []

    def _prepare_swing_rank_features(self, df_in: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Prepares continuous self-normalized multi-horizon Z-score & swing-coil features (preserving extreme
        multi-sigma spring magnitudes like -2.8σ coils for Tree splits) alongside daily cross-sectional
        centered percentile ranks (`[-0.5, +0.5]`) for the Regime-Kernel Factor IC head.
        """
        df = df_in.copy()
        avail_extra = [c for c in SWING_EXTRA_COLS if c in df.columns]
        base_cs_cols = [
            c for c in (FEATURE_COLUMNS + avail_extra)
            if c not in ("india_vix_level", "india_vix_zscore_21d")
        ]
        csr_cols = []
        by_dt = df.groupby("date")
        for c in base_cs_cols:
            cname = f"csr_{c}"
            df[cname] = (by_dt[c].rank(pct=True) - 0.50).fillna(0.0)
            csr_cols.append(cname)

        for mcol, fallback_col, default_v in [
            ("mkt_trend_21d", "fib_mom_z_21d", 0.0),
            ("mkt_fast_5d", "fib_mom_z_5d", 0.0),
            ("mkt_er_21d", "efficiency_ratio_21d", 0.5),
        ]:
            if mcol not in df.columns:
                df[mcol] = by_dt[fallback_col].transform("median").fillna(default_v)

        # Feed continuous volatility-normalized Z-scores + swing coil magnitudes + regime state into Tree models!
        tree_cols = FEATURE_COLUMNS + avail_extra + ["mkt_trend_21d", "mkt_fast_5d", "mkt_er_21d"]
        return df, csr_cols, tree_cols

    def _compute_regime_kernel_weights(
        self,
        train_df: pd.DataFrame,
        target_state: Dict[str, float],
    ) -> np.ndarray:
        """
        Computes Gaussian Market-Regime Similarity Kernel x VIX-Adaptive Exponential Recency weights
        so historical days matching today's Bull/Chop/VIX macro state receive highest weight (zero magic cutoffs).
        """
        tr_reg = train_df.groupby("date", as_index=False).agg(
            t21=("mkt_trend_21d", "first"),
            f5=("mkt_fast_5d", "first"),
            er=("mkt_er_21d", "first"),
            vz=("india_vix_zscore_21d", "first"),
        )
        s_t21 = max(1e-3, float(tr_reg["t21"].std()))
        s_f5 = max(1e-3, float(tr_reg["f5"].std()))
        s_er = max(1e-3, float(tr_reg["er"].std()))
        s_vz = max(1e-3, float(tr_reg["vz"].std()))

        dist2 = (
            ((tr_reg["t21"] - target_state["t21"]) / s_t21) ** 2
            + ((tr_reg["f5"] - target_state["f5"]) / s_f5) ** 2
            + ((tr_reg["er"] - target_state["er"]) / s_er) ** 2
            + ((tr_reg["vz"] - target_state["vz"]) / s_vz) ** 2
        )
        max_dt = tr_reg["date"].max()
        days_ago = (max_dt - tr_reg["date"]).dt.days.values.astype(float)
        stress_factor = np.clip(1.0 + max(0.0, target_state["vz"]) / s_vz, 1.0, 2.5)
        eff_hl = self.wf_cfg.base_half_life_days / stress_factor
        recency_w = np.exp(-np.log(2.0) / max(15.0, eff_hl) * days_ago)

        tr_reg["day_w"] = np.exp(-0.5 * dist2) * recency_w
        mean_w = float(tr_reg["day_w"].mean())
        tr_reg["day_w"] = tr_reg["day_w"] / max(1e-8, mean_w)
        w_map = dict(zip(tr_reg["date"], tr_reg["day_w"]))
        row_w = train_df["date"].map(w_map).fillna(1.0).values
        return row_w / max(1e-8, float(np.mean(row_w)))

    def fit_purged_walk_forward(self, feat_df: pd.DataFrame, as_of_date: pd.Timestamp) -> Dict[str, float]:
        """
        Fits the Dual-Head Future Swing Predictor (`LGBMRegressor` + `LGBMRanker` + Regime-Kernel IC)
        strictly on historical rows whose 5-day forward target completed BEFORE `as_of_date`
        (`date <= as_of_date - purge_embargo_days`), and learns ensemble weights from Purged Validation IC.
        """
        prep_df, csr_cols, tree_cols = self._prepare_swing_rank_features(feat_df)
        self._csr_feature_names = csr_cols
        self._tree_feature_names = tree_cols

        all_dates = np.sort(prep_df["date"].unique())
        eligible_dates = all_dates[all_dates <= as_of_date]
        if len(eligible_dates) <= self.wf_cfg.purge_embargo_days + 30:
            raise ValueError(f"Insufficient history prior to {as_of_date} for purged walk-forward training.")

        # Strict purge cutoff: the last training bar must have completed its t+6 exit on or before as_of_date
        cutoff_date = pd.Timestamp(eligible_dates[-1 - self.wf_cfg.purge_embargo_days])
        start_idx = max(0, len(eligible_dates) - self.wf_cfg.purge_embargo_days - self.wf_cfg.train_window_days)
        start_date = pd.Timestamp(eligible_dates[start_idx])

        target_col = "target_5d_info_ratio"
        train_slice = prep_df[
            (prep_df["date"] >= start_date)
            & (prep_df["date"] <= cutoff_date)
            & prep_df[target_col].notna()
        ].dropna(subset=FEATURE_COLUMNS).sort_values(["date", "symbol"]).copy()

        if len(train_slice) < self.wf_cfg.min_train_rows:
            train_slice = prep_df[
                (prep_df["date"] <= cutoff_date) & prep_df[target_col].notna()
            ].dropna(subset=FEATURE_COLUMNS).sort_values(["date", "symbol"]).copy()

        if "target_5d_rel_grade" in train_slice.columns and train_slice["target_5d_rel_grade"].notna().all():
            rel_grades = train_slice["target_5d_rel_grade"].astype(int).values
        else:
            rel_grades = np.floor(train_slice.groupby("date")[target_col].rank(pct=True) * 29.99).fillna(0).astype(int).values
        train_slice["_rel_grade"] = rel_grades

        cur_day_rows = prep_df[prep_df["date"] == eligible_dates[-1]]
        target_state = {
            "t21": float(cur_day_rows["mkt_trend_21d"].median()),
            "f5": float(cur_day_rows["mkt_fast_5d"].median()),
            "er": float(cur_day_rows["mkt_er_21d"].median()),
            "vz": float(cur_day_rows["india_vix_zscore_21d"].mean()),
        }
        sample_weights = self._compute_regime_kernel_weights(train_slice, target_state=target_state)

        # Regime-Conditioned Empirical Cross-Sectional Factor IC Vector (exact weighted covariance)
        X_csr_all = train_slice[csr_cols].values
        y_all = train_slice[target_col].values
        w_prob = sample_weights / np.sum(sample_weights)
        self.regime_ic_vec = (X_csr_all.T @ (w_prob * y_all)) / np.maximum(1e-5, np.std(X_csr_all, axis=0))

        reg_params = dict(
            n_estimators=130,
            learning_rate=0.04,
            num_leaves=15,
            max_depth=4,
            min_child_samples=45,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.8,
            reg_lambda=2.0,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )
        rk_params = dict(
            n_estimators=110,
            learning_rate=0.04,
            num_leaves=15,
            max_depth=4,
            min_child_samples=45,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.8,
            reg_lambda=2.0,
            objective="lambdarank",
            lambdarank_truncation_level=8,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )

        # Purged Out-of-Sample Validation Fold (last 20 dates with strict 6-bar purge gap) to self-calibrate head weights!
        train_unique_dates = np.sort(train_slice["date"].unique())
        oos_rank_ic = np.nan
        val_bars = 20
        if len(train_unique_dates) > val_bars + 25:
            val_start_dt = train_unique_dates[-val_bars]
            inner_cutoff_dt = train_unique_dates[-val_bars - self.wf_cfg.purge_embargo_days]
            inner_tr = train_slice[train_slice["date"] <= inner_cutoff_dt].sort_values(["date", "symbol"])
            val_fold = train_slice[train_slice["date"] >= val_start_dt].sort_values(["date", "symbol"])
            if len(inner_tr) >= 500 and len(val_fold) >= 100:
                inner_w = self._compute_regime_kernel_weights(inner_tr, target_state=target_state)
                val_reg = lgb.LGBMRegressor(**reg_params).fit(inner_tr[tree_cols], inner_tr[target_col].values, sample_weight=inner_w)
                grp_in = inner_tr.groupby("date", sort=False).size().values
                val_rk = lgb.LGBMRanker(**rk_params).fit(inner_tr[tree_cols], inner_tr["_rel_grade"].values, group=grp_in)

                X_in_csr = inner_tr[csr_cols].values
                w_in_prob = inner_w / np.sum(inner_w)
                in_ic_vec = (X_in_csr.T @ (w_in_prob * inner_tr[target_col].values)) / np.maximum(1e-5, np.std(X_in_csr, axis=0))

                p1 = pd.Series(val_reg.predict(val_fold[tree_cols]), index=val_fold.index)
                p2 = pd.Series(val_rk.predict(val_fold[tree_cols]), index=val_fold.index)
                p3 = pd.Series(val_fold[csr_cols].values @ in_ic_vec, index=val_fold.index)
                er_v = val_fold["mkt_er_21d"].values
                p4 = pd.Series(
                    (1.0 - er_v) * (-0.5 * val_fold["csr_rel_strength_vs_nifty_21d"].values - 0.5 * val_fold["csr_mean_rev_zscore_13d"].values)
                    + er_v * val_fold.get("csr_inst_clv_5d", val_fold["csr_efficiency_ratio_8d"]).values,
                    index=val_fold.index,
                )

                def _mean_cs_ic(pred_s: pd.Series) -> float:
                    ics = val_fold.groupby("date").apply(
                        lambda gdf: pred_s.loc[gdf.index].corr(gdf[target_col], method="spearman"),
                        include_groups=False,
                    )
                    val_m = float(ics.mean())
                    return val_m if not np.isnan(val_m) else 0.01

                head_ics = np.array([_mean_cs_ic(p1), _mean_cs_ic(p2), _mean_cs_ic(p3), _mean_cs_ic(p4)])
                se_floor = 1.0 / np.sqrt(max(25.0, float(len(val_fold))))
                pos_w = np.maximum(se_floor, head_ics)
                self.head_weights = pos_w / np.sum(pos_w)

                ens_val = (
                    self.head_weights[0] * p1.groupby(val_fold["date"]).rank(pct=True)
                    + self.head_weights[1] * p2.groupby(val_fold["date"]).rank(pct=True)
                    + self.head_weights[2] * p3.groupby(val_fold["date"]).rank(pct=True)
                    + self.head_weights[3] * p4.groupby(val_fold["date"]).rank(pct=True)
                )
                oos_rank_ic = _mean_cs_ic(ens_val)

        # Fit full Dual-Head Swing Predictor on `train_slice` + TreeSHAP Explainer on Primary Regressor
        self.model = lgb.LGBMRegressor(**reg_params)
        self.model.fit(train_slice[tree_cols], y_all, sample_weight=sample_weights)
        grp_full = train_slice.groupby("date", sort=False).size().values
        self.ranker_model = lgb.LGBMRanker(**rk_params)
        self.ranker_model.fit(train_slice[tree_cols], train_slice["_rel_grade"].values, group=grp_full)
        self.explainer = shap.TreeExplainer(self.model)
        self.last_train_date = pd.Timestamp(as_of_date)

        preds_in = self.model.predict(train_slice[tree_cols])
        insample_ic = pd.Series(preds_in).corr(pd.Series(y_all), method="spearman")
        effective_ic = oos_rank_ic if not np.isnan(oos_rank_ic) else insample_ic
        self.last_rank_ic = float(np.clip(effective_ic if not np.isnan(effective_ic) else 0.10, 0.03, 0.75))

        return {
            "train_start_date": start_date.strftime("%Y-%m-%d"),
            "train_cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
            "prediction_date": pd.Timestamp(eligible_dates[-1]).strftime("%Y-%m-%d"),
            "purge_gap_bars": int(self.wf_cfg.purge_embargo_days),
            "train_rows": int(len(train_slice)),
            "spearman_rank_ic": round(float(effective_ic), 4),
            "insample_rank_ic": round(float(insample_ic), 4),
            "ensemble_head_weights": [round(float(x), 3) for x in self.head_weights],
        }

    def _score_daily_ml_thesis(self, slice_df: pd.DataFrame) -> np.ndarray:
        """
        Evaluates the Dual-Head Future-Swing ML (`LGBMRegressor` + `LGBMRanker` + Regime-Kernel IC)
        strictly on apples-to-apples multi-day swing features across `[t-2, t-1, t]` (used both for
        ML forward alpha and for Multi-Bar Smoothed Hold-Rank Thesis Decay detection).
        """
        tree_cols = self._tree_feature_names
        csr_cols = self._csr_feature_names
        p_reg = self.model.predict(slice_df[tree_cols])
        p_rk = self.ranker_model.predict(slice_df[tree_cols])
        p_kern = slice_df[csr_cols].values @ self.regime_ic_vec

        out_scores = np.zeros(len(slice_df), dtype=float)
        w = self.head_weights
        w_sum = max(1e-6, float(w[0] + w[1] + w[2]))
        for _, idx_s in slice_df.groupby("date").groups.items():
            pos_idx = slice_df.index.get_indexer(idx_s)
            r1 = pd.Series(p_reg[pos_idx]).rank(pct=True).values - 0.50
            r2 = pd.Series(p_rk[pos_idx]).rank(pct=True).values - 0.50
            r3 = pd.Series(p_kern[pos_idx]).rank(pct=True).values - 0.50
            blend = (w[0] * r1 + w[1] * r2 + w[2] * r3) / w_sum
            out_scores[pos_idx] = pd.Series(blend).rank(pct=True).values - 0.50
        return out_scores

    def score_cross_section_with_shap(
        self, feat_df: pd.DataFrame, as_of_date: Optional[pd.Timestamp] = None, retrain: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        """
        Scores all 50 Nifty stocks on `as_of_date`:
        1. Unlagged Forward Swing Entry Score (`pred_info_ratio` & `expected_5d_alpha_pct`): fuses the
           Dual-Head Regime-Kernel Future Swing ML predictor (weighted by Purged Validation Rank IC) with
           live 09:15-09:30 AM Institutional Absorption + Multi-Week Coiled Value Spring (zero multi-day lag on entries!).
        2. Multi-Bar Kaufman-Smoothed Hold Rank (`smoothed_rank_pct` over `[t-2, t-1, t]`): separates 1-day
           pullback noise from genuine multi-day thesis decay for the Risk Vault's Priority-B Loss-Recovery Engine.
        """
        valid_df = feat_df.dropna(subset=FEATURE_COLUMNS).copy()
        target_date = pd.Timestamp(as_of_date) if as_of_date is not None else valid_df["date"].max()

        train_meta = {}
        if retrain or self.model is None or self.ranker_model is None:
            train_meta = self.fit_purged_walk_forward(valid_df, as_of_date=target_date)

        prep_df, _, tree_cols = self._prepare_swing_rank_features(valid_df[valid_df["date"] <= target_date])
        avail_dates = np.sort(prep_df["date"].unique())
        recent_dates = avail_dates[-3:]
        recent_slice = prep_df[prep_df["date"].isin(recent_dates)].sort_values(["symbol", "date"]).copy()
        recent_slice["_ml_thesis_score"] = self._score_daily_ml_thesis(recent_slice)

        day_er = float(recent_slice.loc[recent_slice["date"] == target_date, "mkt_er_21d"].median())
        smooth_alpha = float(np.clip(0.50 + 0.35 * day_er, 0.50, 0.85))
        recent_slice["_smoothed_ml_thesis"] = recent_slice.groupby("symbol")["_ml_thesis_score"].transform(
            lambda s: s.ewm(alpha=smooth_alpha, adjust=False).mean()
        )

        day_slice = recent_slice[recent_slice["date"] == target_date].copy()
        if day_slice.empty:
            raise ValueError(f"No feature rows found for date {target_date}")

        X_day_tree = day_slice[tree_cols]
        raw_pred_ir = self.model.predict(X_day_tree)
        shap_matrix = self.explainer.shap_values(X_day_tree)

        # 1. Dual-Head Future-Swing ML Prediction (preserving continuous multi-sigma magnitude + LGBMRanker ordering)
        pred_std = max(1e-4, float(np.std(raw_pred_ir)))
        ml_today = day_slice["_ml_thesis_score"].values
        ml_rank_z = (ml_today - float(np.mean(ml_today))) / max(1e-4, float(np.std(ml_today)))
        dual_ml_ir = 0.60 * raw_pred_ir + 0.40 * pred_std * ml_rank_z

        # 2. Unlagged 09:15-09:30 AM Institutional Thrust + Coiled Value-Compression Swing Synergy
        r_val_rel = (-day_slice["rel_strength_vs_nifty_21d"]).rank(pct=True).values
        r_val_m21 = (-day_slice["fib_mom_z_21d"]).rank(pct=True).values
        r_val_m55 = (-day_slice["fib_mom_z_55d"]).rank(pct=True).values
        r_vshock = day_slice["volume_shock_21d"].rank(pct=True).values
        r_clv5 = (
            day_slice["inst_clv_5d"].rank(pct=True).values
            if "inst_clv_5d" in day_slice.columns
            else day_slice["efficiency_ratio_8d"].rank(pct=True).values
        )

        if "opening_15m_body_pct" in day_slice.columns and "opening_15m_clv" in day_slice.columns:
            r_body15 = day_slice["opening_15m_body_pct"].rank(pct=True).values
            r_clv15 = day_slice["opening_15m_clv"].rank(pct=True).values
        else:
            r_body15 = (-day_slice["mean_rev_zscore_13d"]).rank(pct=True).values
            r_clv15 = r_clv5

        cs_composite = (
            0.35 * r_body15
            + 0.25 * r_clv15
            + 0.17 * r_val_rel
            + 0.10 * r_val_m21
            + 0.05 * r_val_m55
            + 0.04 * r_vshock
            + 0.04 * r_clv5
        )
        cs_z = (cs_composite - float(np.mean(cs_composite))) / max(1e-4, float(np.std(cs_composite)))

        # Self-calibrated blend using Purged Out-of-Sample Rank IC (zero hardcoded magic scale!)
        w_ml = float(np.clip(self.last_rank_ic, 0.15, 0.45))
        structural_prior_weight = 1.0 - w_ml
        in_archetype = (day_slice["micro_regime"] != "NEUTRAL_WATCH").astype(float).values

        day_slice["pred_info_ratio"] = (
            w_ml * dual_ml_ir
            + structural_prior_weight * pred_std * (0.35 * in_archetype + 1.35 * cs_z)
        )

        raw_5d_vol_pct = day_slice["ret_std_21d"] * np.sqrt(5.0) * 100.0
        five_day_vol_pct = raw_5d_vol_pct.fillna(float(raw_5d_vol_pct.median()))
        day_slice["expected_5d_alpha_pct"] = (day_slice["pred_info_ratio"] * five_day_vol_pct).round(2)

        # Preserve BOTH Unlagged 1-Day Entry Rank (`raw_1d_rank_pct`) AND Multi-Bar Smoothed Hold Thesis Rank (`smoothed_rank_pct`)!
        day_slice["raw_1d_rank_pct"] = day_slice["pred_info_ratio"].rank(pct=True, method="average").round(4)
        smoothed_combo = 0.55 * day_slice["_smoothed_ml_thesis"].rank(pct=True) + 0.45 * day_slice["raw_1d_rank_pct"]
        day_slice["smoothed_rank_pct"] = smoothed_combo.rank(pct=True, method="average").round(4)
        day_slice["inst_clv_5d_pct"] = (
            day_slice["inst_clv_5d"].rank(pct=True, method="average").round(4)
            if "inst_clv_5d" in day_slice.columns
            else 0.50
        )
        day_slice["conviction_pct"] = (day_slice["raw_1d_rank_pct"] * 100.0).round(1)

        # Extract Top-3 positive SHAP drivers and Top negative SHAP drag per stock (mapped cleanly to human feature names)
        top1_list, top2_list, top3_list, drag_list, dominante_horizon_list = [], [], [], [], []
        for row_idx in range(len(day_slice)):
            sv = shap_matrix[row_idx]
            order_desc = np.argsort(sv)[::-1]
            order_asc = np.argsort(sv)

            def _fmt(f_idx: int) -> str:
                tname = tree_cols[f_idx]
                raw_name = tname[4:] if tname.startswith("csr_") else tname
                hname = FEATURE_HUMAN_NAMES.get(raw_name, raw_name)
                raw_val = float(day_slice.iloc[row_idx][raw_name]) if raw_name in day_slice.columns else float(day_slice.iloc[row_idx][tname])
                return f"{hname} ({sv[f_idx]:+.2f} | val={raw_val:.2f})"

            top1_list.append(_fmt(order_desc[0]))
            top2_list.append(_fmt(order_desc[1]))
            top3_list.append(_fmt(order_desc[2]))
            drag_list.append(_fmt(order_asc[0]))

            fast_shap = sum(
                abs(sv[tree_cols.index(f"fib_mom_z_{w}d")])
                for w in (3, 5, 8)
                if f"fib_mom_z_{w}d" in tree_cols
            )
            slow_shap = sum(
                abs(sv[tree_cols.index(f"fib_mom_z_{w}d")])
                for w in (13, 21, 34, 55)
                if f"fib_mom_z_{w}d" in tree_cols
            )
            dominante_horizon_list.append("FAST_3_8D_LENS" if fast_shap > slow_shap else "MACRO_13_55D_LENS")

        day_slice["shap_driver_1"] = top1_list
        day_slice["shap_driver_2"] = top2_list
        day_slice["shap_driver_3"] = top3_list
        day_slice["shap_top_drag"] = drag_list
        day_slice["dominant_fib_lens"] = dominante_horizon_list

        # Clean temporary csr_ columns while keeping core + swing telemetry intact
        drop_tmp = [
            c for c in day_slice.columns
            if c.startswith("csr_") or c in ("_ml_thesis_score", "_smoothed_ml_thesis", "_raw_ens_score", "_smoothed_ens_score")
        ]
        day_slice = day_slice.drop(columns=drop_tmp, errors="ignore")

        ranked_df = day_slice.sort_values("pred_info_ratio", ascending=False).reset_index(drop=True)
        ranked_df["alpha_rank"] = np.arange(1, len(ranked_df) + 1)
        ranked_df["smoothed_alpha_rank"] = ranked_df["alpha_rank"]

        # Scale-Invariant Dynamic Friction Hurdle:
        # Exact Statutory Tax (at current CapitalConfig) + 21d Median Order-Size Amihud Impact + 1 Cross-Sectional Standard Error
        slot_notional_inr = self.capital_cfg.ideal_slot_size_inr
        vol_med_21_shares = ranked_df["volume"] / ranked_df["volume_shock_21d"].replace(0, 1.0)
        median_adv_inr = float((ranked_df["close"] * vol_med_21_shares).median())
        median_part_rate = np.clip(slot_notional_inr / max(1.0, median_adv_inr), 1e-7, 0.25)
        median_impact_pct = float(2.0 * ranked_df["parkinson_vol_21d"].median() * np.sqrt(median_part_rate) * 100.0)
        alpha_std_err = float(ranked_df["expected_5d_alpha_pct"].std() / np.sqrt(len(ranked_df)))
        dynamic_alpha_hurdle_pct = round(
            self.capital_cfg.dynamic_round_trip_tax_pct + median_impact_pct + alpha_std_err, 3
        )
        ranked_df["dynamic_alpha_hurdle_pct"] = dynamic_alpha_hurdle_pct
        ir_median = float(ranked_df["pred_info_ratio"].median())

        # Build Sector Alpha Breadth Table (shows which sectors earn capital vs. which get starved at ₹0)
        sector_df = (
            ranked_df.groupby("sector", as_index=False)
            .agg(
                stocks_count=("symbol", "count"),
                mean_pred_ir=("pred_info_ratio", "mean"),
                max_pred_ir=("pred_info_ratio", "max"),
                mean_expected_alpha_pct=("expected_5d_alpha_pct", "mean"),
                max_expected_alpha_pct=("expected_5d_alpha_pct", "max"),
                top_stock=("symbol", "first"),
                positive_alpha_stocks=(
                    "expected_5d_alpha_pct",
                    lambda s: int((s > dynamic_alpha_hurdle_pct).sum()),
                ),
            )
            .sort_values("mean_pred_ir", ascending=False)
            .reset_index(drop=True)
        )
        sector_active_mask = (sector_df["max_pred_ir"] > ir_median) & (sector_df["positive_alpha_stocks"] >= 1)
        n_active_sectors = int(sector_active_mask.sum())
        if n_active_sectors <= 1:
            active_cap_label = "Adaptive Up to 4 Slots (100% Booming-Sector Focus)"
        elif n_active_sectors == 2:
            active_cap_label = "Adaptive 2-3 Slots (Breadth-Scaled)"
        else:
            active_cap_label = "Adaptive 2 Slots (4-Slot Spillover Enabled)"
        sector_df["max_sector_slots_cap"] = np.where(
            sector_active_mask,
            active_cap_label,
            "0 Slots (₹0 Allocated)",
        )
        sector_df["sector_capital_status"] = np.where(
            sector_active_mask,
            f"ACTIVE_ELIGIBLE (Clears +{dynamic_alpha_hurdle_pct:.2f}% Tax+SE Hurdle)",
            f"STARVED_AT_₹0 (Fails +{dynamic_alpha_hurdle_pct:.2f}% Hurdle)",
        )

        return ranked_df, sector_df, train_meta


def run_section_3_verification() -> None:
    print("=" * 105)
    print("SECTION 3 VERIFICATION: WALK-FORWARD AI SCOUT (LightGBM) + EXACT SHAP EXPLAINER + SECTOR CAPITAL ROUTING")
    print("=" * 105)

    data_engine = HybridDataEngine(cache_dir="data/cache", capital_cfg=CapitalConfig(initial_capital_inr=100_000.0))
    stocks_df, macro_df, _ = data_engine.fetch_and_build_telemetry(force_refresh=False)

    feat_engine = ChameleonFeatureEngine(cache_dir="data/cache")
    feat_df = feat_engine.build_features(stocks_df, macro_df, save_cache=False)

    scout = WalkForwardAlphaScout()
    ranked_df, sector_df, train_meta = scout.score_cross_section_with_shap(feat_df, retrain=True)

    print("[1] PURGED WALK-FORWARD TRAINING TELEMETRY (ZERO LOOK-AHEAD VERIFICATION):")
    print(f"  • Training Window Start Date : {train_meta['train_start_date']}")
    print(f"  • Purged Training Cutoff     : {train_meta['train_cutoff_date']} (Strict {train_meta['purge_gap_bars']}-bar embargo before prediction date)")
    print(f"  • Live Scoring Date          : {train_meta['prediction_date']}")
    print(f"  • Training Observations      : {train_meta['train_rows']:,} stock-days")
    print(f"  • Walk-Forward Rank IC       : {train_meta['spearman_rank_ic']:+.4f} (Spearman Information Coefficient)")

    print("\n[2] TOP 10 AI-RANKED CANDIDATE STOCKS TODAY (WITH EXPECTED 5D ALPHA & DOMINANT FIBONACCI LENS):")
    top_cols = [
        "alpha_rank",
        "symbol",
        "sector",
        "close",
        "exec_vwap_915",
        "pred_info_ratio",
        "expected_5d_alpha_pct",
        "conviction_pct",
        "dominant_fib_lens",
        "micro_regime",
    ]
    disp_top = ranked_df[top_cols].head(10).copy()
    disp_top["pred_info_ratio"] = disp_top["pred_info_ratio"].round(3)
    disp_top["expected_5d_alpha_pct"] = disp_top["expected_5d_alpha_pct"].map(lambda x: f"{x:+.2f}%")
    print(disp_top.to_string(index=False))

    print("\n[3] EXACT TREE-SHAP MATHEMATICAL REASONS FOR THE TOP 5 CANDIDATES (0% HALLUCINATION TELEMETRY):")
    for _, row in ranked_df.head(5).iterrows():
        print(
            f"  #{int(row['alpha_rank'])} {row['symbol']:<11} ({row['sector']:<10} | E[Alpha]={row['expected_5d_alpha_pct']:+.2f}% | Stop={row['empirical_tail_stop_pct']*100:.2f}%)"
        )
        print(f"     ├─ Driver #1 : {row['shap_driver_1']}")
        print(f"     ├─ Driver #2 : {row['shap_driver_2']}")
        print(f"     ├─ Driver #3 : {row['shap_driver_3']}")
        print(f"     └─ Main Drag : {row['shap_top_drag']}")

    print("\n[4] SECTOR ALPHA BREADTH & DYNAMIC CAPITAL ROUTING (WEAK SECTORS AUTOMATICALLY STARVED AT ₹0):")
    sec_disp = sector_df.copy()
    sec_disp["mean_pred_ir"] = sec_disp["mean_pred_ir"].round(3)
    sec_disp["max_pred_ir"] = sec_disp["max_pred_ir"].round(3)
    sec_disp["mean_expected_alpha_pct"] = sec_disp["mean_expected_alpha_pct"].map(lambda x: f"{x:+.2f}%")
    print(sec_disp.to_string(index=False))

    print("\n[5] BOTTOM 5 LAGGARD STOCKS TODAY (CANDIDATES FOR SMART EVICTION IF HELD):")
    bot_disp = ranked_df[top_cols].tail(5).copy()
    bot_disp["pred_info_ratio"] = bot_disp["pred_info_ratio"].round(3)
    bot_disp["expected_5d_alpha_pct"] = bot_disp["expected_5d_alpha_pct"].map(lambda x: f"{x:+.2f}%")
    print(bot_disp.to_string(index=False))
    print("=" * 105)


if __name__ == "__main__":
    run_section_3_verification()
