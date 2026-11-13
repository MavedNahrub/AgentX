export interface OhlcvBar {
  time: number | string;
  t: string;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
  vwap: number;
  ema9: number;
  ema21: number;
  bb_upper: number;
  bb_lower: number;
  rsi14: number;
  ratchet_stop: number | null;
}

export interface StockBundle {
  sym: string;
  session_date: string;
  cur_px: number;
  buy_px: number;
  stop_px: number;
  target_px: number;
  qty: number;
  move_pct: number;
  status: string;
  strategy: string;
  sector: string;
  rank: number;
  smoothed_rank_pct: number;
  alpha_pct: number;
  conv_pct: number;
  shap1: string;
  shap2: string;
  audit_headline: string;
  audit_body: string;
  audit_tokens: string[];
  bars: OhlcvBar[];
}

export interface MarketItem {
  sym: string;
  rank: number;
  smoothed_rank_pct: number;
  status: string;
  sub: string;
  px: number;
  chg: number;
}

export interface ExecutedTradeItem {
  sym: string;
  side: "BUY" | "SELL" | string;
  price: number;
  qty: number;
  date: string;
  full_date: string;
  tax: number;
  pnl: number;
  exit_reason: string;
}

export interface CurvePoint {
  t: string;
  date: string;
  eq: number;
  p: number;
  n: number;
}

export interface SessionLedgerRow {
  day_num: number;
  date: string;
  prev_eq: number;
  eq: number;
  session_pnl_inr: number;
  sess_ret: number;
  nif_sess: number;
  sess_alpha: number;
  ret_pct: number;
  nif_cum: number;
  cum_alpha: number;
  realized_pnl: number;
  taxes_paid: number;
  churn_saved: number;
  fwd_5d_end_day_num: number;
  fwd_5d_end_date: string;
  fwd_5d_days_elapsed: number;
  fwd_5d_is_settled: boolean;
  fwd_5d_status_label: string;
  fwd_5d_end_eq: number;
  fwd_5d_profit_inr: number;
  fwd_5d_ret_pct: number;
  fwd_5d_nifty_pct: number;
  fwd_5d_alpha_pct: number;
}

export interface CompletedSwingTrade {
  trade_num: number;
  stock: string;
  sector: string;
  strategy: string;
  buy_date: string;
  sell_date: string;
  sessions_held: number;
  shares: number;
  buy_px: number;
  sell_px: number;
  gross_ret_pct: number;
  net_pnl_inr: number;
  round_trip_tax_inr: number;
  exit_reason: string;
  exit_badge: string;
}

export interface SwingAnalytics {
  horizon_sessions: number;
  settled_5d_windows_count: number;
  settled_5d_positive_count: number;
  settled_5d_alpha_wins: number;
  settled_5d_win_rate_pct: number;
  avg_5d_profit_inr: number;
  avg_5d_ret_pct: number;
  avg_5d_nifty_pct: number;
  avg_5d_alpha_pct: number;
  best_5d_ret_pct: number;
  best_5d_profit_inr: number;
  best_5d_window_label: string;
  min_5d_ret_pct: number;
  min_5d_profit_inr: number;
  avg_holding_sessions: number;
  multi_day_trades_count: number;
  core_3to5d_trades_count: number;
  multi_day_trades_pct: number;
  completed_swing_trades: CompletedSwingTrade[];
}

export interface SectorGuardrailRow {
  sector: string;
  slots: number;
  alpha: number;
  status: string;
}

export interface TickerSeriesPoint {
  t: string;
  eq: number;
  pos: number;
  cash: number;
}

export interface PortfolioSummary {
  initial_capital_inr: number;
  equity: number;
  invested_inr: number;
  entry_notional_inr: number;
  cash: number;
  deployed_pct: number;
  net_pnl_inr: number;
  net_pct: number;
  realized_pnl_inr: number;
  unrealized_pnl_inr: number;
  total_pnl_inr: number;
  taxes_paid_inr: number;
  churn_saved_inr: number;
  statutory_rt_tax_pct: number;
  sessions_count: number;
  latest_sess_ret: number;
  latest_nifty_sess: number;
  latest_nifty_cum: number;
  cum_alpha_pct: number;
  closed_trades_count: number;
  win_count: number;
  win_rate_pct: number;
  settled_5d_windows_count: number;
  settled_5d_alpha_wins: number;
  settled_5d_win_rate_pct: number;
  avg_5d_profit_inr: number;
  avg_5d_ret_pct: number;
  avg_5d_nifty_pct: number;
  avg_5d_alpha_pct: number;
  avg_holding_sessions: number;
  wf_6m_agent_ret: number;
  wf_6m_agent_eq: number;
  wf_6m_naive_ret: number;
  wf_6m_naive_eq: number;
  wf_6m_bh_ret: number;
  wf_6m_bh_eq: number;
  wf_6m_alpha: number;
  top_ranked_symbol: string;
  top_ranked_alpha_pct: number;
  journal_audited_count: number;
  hallucination_rate_pct: number;
}

export interface LiveSessionState {
  status: string;
  date: string | null;
  started_at: string | null;
  current_minute: number;
  total_minutes: number;
  market_clock_ist: string;
  today_date: string;
  market_minute_now: number;
  message: string;
  phase_detail: string;
  error: string | null;
  stop_requested: boolean;
}

export interface OperatorState {
  kill_switch_mode: string;
  kill_switch_reason: string;
  active_session_date?: string;
  live_session?: LiveSessionState;
  queue_state: {
    completed_sessions: string[];
    next_session_date: string;
    pending_fetch_date: string | null;
    status: string;
  };
}

export interface JournalEntry {
  date?: string;
  category: string;
  symbol: string;
  headline: string;
  narrative: string;
  tokens: string[];
}

export interface FriendlyEvent {
  date?: string;
  time: string;
  stock: string;
  event: string;
  summary: string;
}

export interface WalkForwardPoint {
  date: string;
  agentx_inr: number;
  naive_inr: number;
  nifty_inr: number;
}

export interface WorkstationPayload {
  generated_at_epoch: number;
  active_session_date: string;
  latest_traded_date: string;
  sorted_traded_dates: string[];
  portfolio_summary: PortfolioSummary;
  swing_analytics: SwingAnalytics;
  operator_state: OperatorState;
  stock_bundles: Record<string, StockBundle>;
  market_list: MarketItem[];
  last_trades: ExecutedTradeItem[];
  overview_pts: CurvePoint[];
  session_ledger: SessionLedgerRow[];
  sector_guardrails: SectorGuardrailRow[];
  intraday_ticker_series: TickerSeriesPoint[];
  dock: {
    tested_rows: Record<string, string>[];
    buy_fill_rows_by_date: Record<string, string>[];
    order_fills: Record<string, unknown>[];
    gtt_triggers: Record<string, unknown>[];
    contract_notes: Record<string, unknown>[];
    intraday_events: FriendlyEvent[];
    events_by_session?: Record<string, FriendlyEvent[]>;
    wf_6m_curve: WalkForwardPoint[];
    wf_6m_summary: Record<string, unknown>[];
    top_rankings: Record<string, unknown>[];
    sector_table: Record<string, unknown>[];
    journal_entries: JournalEntry[];
  };
}

export interface LiveTickMessage {
  type: "LIVE_TICK";
  clock_ist: string;
  equity: number;
  invested_inr: number;
  cash: number;
  net_pnl_inr: number;
  net_pct: number;
  unrealized_pnl_inr: number;
  total_pnl_inr: number;
  realized_pnl_inr: number;
  kill_switch_mode: string;
  live_status?: string;
  current_minute?: number;
  live_message?: string;
}
