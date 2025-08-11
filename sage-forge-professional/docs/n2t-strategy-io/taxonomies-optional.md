### Optional Taxonomies — Pick & Choose

Add columns to your tables using these tags. Fill Your_Rank and Include? per study.

| Acronym | Term                       | Purpose                   | Typical Bins                                                                    | Suggested Column      | Default_Rank | Your_Rank | Include? |
| ------- | -------------------------- | ------------------------- | ------------------------------------------------------------------------------- | --------------------- | ------------ | --------- | -------- |
| LCT     | Lifecycle Classification   | Stage in R→P flow         | Ideation/DataPrep/Modeling/Backtest/PaperTrade/Prod                             | lifecycle             | 6            |           |          |
| DSQ     | Data Source & Quality      | Provenance + reliability  | source: Primary/Ref/Derived/Synth; grade: A/B/C; latency: L0/L1/L2              | dsq                   | 9            |           |          |
| FCT     | Feature Category           | Feature family            | Price/Vol/Volume/Micro/XAsset/Exog/Calendar                                     | feature_category      | 8            |           |          |
| TTT     | Target & Task Type         | Learning task clarity     | Point/Quantile/Path/Class/Reg/Density; unit: price/return/logret                | task, target_unit     | 9            |           |          |
| HCT     | Horizon & Cadence          | T/k & sampling            | US(≤5)/S(6–64)/M(65–512)/L(>512); 1s/1m/5m/1h                                   | horizon, bar, T, k    | 9            |           |          |
| RST     | Regime & State             | Market condition tags     | vol: L/M/H; trend: U/D/F; liq: T/N/K; session: AS/EU/US; event: None/Macro/Exch | regime                | 7            |           |          |
| MCT     | Model Class & Topology     | Model family              | Stat/ML/DL/Hybrid; ARIMA/RF/Transformer/xLSTM                                   | model_class, topology | 7            |           |          |
| URT     | Uncertainty & Risk Type    | Noise vs ignorance        | aleatoric/epistemic; quantile/interval/ensemble                                 | uncertainty, repr     | 8            |           |          |
| EMT     | Evaluation Metric Type     | Metric selection          | MAE/RMSE, sMAPE, Pinball/CRPS/Coverage, PnL/Sharpe/MDD                          | metrics               | 9            |           |          |
| BAT     | Backtest Assumption Tags   | Assumption snapshot       | fees_bps, slip_bps, latency_ms, order: MKT/LMT, liq: Static/Book/Replay         | bt_assumptions        | 9            |           |          |
| SAT     | Signal Archetype           | Intent of rule            | Trend/MR/Breakout/Carry/Spread/VolTarget                                        | signal_archetype      | 8            |           |          |
| ECT     | Execution Conditions       | Executability constraints | order: MKT/LMT/IOC; venue: Spot/Perp; throttle_qps; killswitch_dd               | execution             | 8            |           |          |
| MAT     | Monitoring & Alerts        | Alerting rules            | DQ(missing/flat/outlier), Model(error/coverage), Trade(slip/reject)             | monitor               | 7            |           |          |
| FMET    | Failure Modes & Effects    | Pre/post-mortem tags      | Data/Model/Exec/Ops → PnL/Risk/Latency                                          | failure_mode, effect  | 7            |           |          |
| GRT     | Governance & Repro Track   | Compliance hooks          | pins/approvals/audit/rollback                                                   | governance            | 8            |           |          |
| CCT     | Complexity & Cost          | Budgeting                 | train_gpu_h, infer_ms_per_series, mem_mb, effort(S/M/L)                         | cost                  | 7            |           |          |
| AAT     | Asset & Instrument         | Instrument quirks         | class: Crypto/FX/Equity/Fut; contract: Spot/Perp/Fut; margin: USD/coin; tick    | asset                 | 8            |           |          |
| SCT     | Session & Calendar Tagging | Time handling             | 24x7/Exchange; tz=UTC; holidays=<ref>                                           | calendar              | 8            |           |          |
| IDT     | Integration & Dataflow     | Pipeline placement        | Ingest/Validate/Transform/Feature/Forecast/Signal/Exec/Log                      | dataflow_stage        | 7            |           |          |
| EDT     | Experiment Design          | Consistent IDs            | F{feature}\_M{model}\_H{k}\_SPL{scheme}\_SEED{s}                                | exp_id                | 8            |           |          |
