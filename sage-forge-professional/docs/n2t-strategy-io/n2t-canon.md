### N2T Canon — Strategy I/O & Research Taxonomy (Rank 0–10)

Fill Your_Rank (0–10) and Include? (Y/N) to drive build order.

| Acronym | Term                               | Purpose                          | Where it lives        | Suggested Columns                | Tier | Default_Rank | Your_Rank | Include? |
| ------- | ---------------------------------- | -------------------------------- | --------------------- | -------------------------------- | ---- | ------------ | --------- | -------- |
| DPF     | Data Provenance & Fidelity         | Source, timestamps, validation   | Data Dictionary       | source, checksum, latency_ms     | A    | 10           |           |          |
| SCU     | Schema, Contracts & Units          | Names, dtypes, units, tz=UTC     | Data Dictionary       | dtype, unit, tz                  | A    | 10           |           |          |
| CLC     | Causality & Leakage Controls       | Forbid future info               | Data/QC               | leakage_guard                    | A    | 10           |           |          |
| MCC     | Model Context Contract             | Input shape/length/normalization | Spec Meta / Model I/O | T, scaler, clip                  | A    | 10           |           |          |
| FOC     | Forecast Output Contract           | Outputs, horizon k, quantiles    | Model I/O             | k, quantiles, align              | A    | 10           |           |          |
| FTR     | Feature Registry                   | Derived features & lineage       | Data Dictionary       | formula, lineage                 | A    | 9            |           |          |
| BTP     | Backtest & Tuning Protocol         | Splits, metrics, fees            | Experiment            | split_id, metrics                | A    | 9            |           |          |
| SDL     | Signal Definition Language         | Entry/Exit/TP/SL rules           | Signal Table          | entry, exit, tp, sl, cooldown    | A    | 9            |           |          |
| RBR     | Risk Budget & Sizing Rules         | Position sizing & limits         | Risk                  | risk_budget, size_rule           | A    | 9            |           |          |
| CAL     | Calendar & Sessionization          | Sessions, gaps, DST              | Meta / Data           | session, gap_policy              | B    | 8            |           |          |
| NRM     | Normalization Policy               | Scaling/clipping policy          | Model In              | scaler, win, clip                | B    | 8            |           |          |
| UQC     | Uncertainty & Quantile Calibration | Coverage & calibration           | Evaluation            | coverage_target                  | B    | 8            |           |          |
| OSS     | Order Simulation Spec              | Order model, latency, slippage   | Backtest Sim          | order_type, slip_bps, latency_ms | B    | 8            |           |          |
| ARL     | Audit & Reproducibility Ledger     | Versions, hashes, seeds          | Run Manifest          | data_hash, model_hash, seed      | B    | 8            |           |          |
| IRP     | Inference Runtime Profile          | HW/SW stack, SLOs                | Ops                   | gpu, infer_ms                    | C    | 7            |           |          |
| MND     | Monitoring & Drift                 | Data/model/trade health          | Ops/Monitoring        | drift_thresholds                 | C    | 7            |           |          |
| DRG     | Deployment & Rollout Guardrails    | Canary, kill-switch, rollback    | Ops                   | canary_pct, killswitch_dd        | C    | 7            |           |          |
| RLB     | Research Logbook                   | Hypotheses, decisions, rationale | Docs                  | decision_note                    | C    | 6            |           |          |
