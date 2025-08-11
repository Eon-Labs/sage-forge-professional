### N2T Strategy I/O Docs

These docs standardize a portable, tabular Markdown I/O contract for zero-/few-shot time-series forecasters (e.g., `NX-AI/TiRex`) and downstream signal/risk design.

- **Scope**: research & design (R&D) specifications only; no code.
- **Design tenets**: DRY, causality-first, audit-ready, model-agnostic.
- **How to use**: pick acronyms from the taxonomies, then fill the Strategy I/O Contract template.

#### Files

- `n2t-canon.md` — Core governance taxonomy (ranks 0–10; Tier A/B/C)
- `taxonomies-optional.md` — Optional tagging/classification taxonomies
- `strategy-io-contract.md` — Tabular I/O Contract template to copy
- `tirex-quickref.md` — TiRex-specific facts to embed when applicable

#### Source of Truth for Column Names

DSM (`data-source-manager`) is the authoritative source for RAW column nomenclature and timestamp semantics. All Strategy I/O contracts MUST use DSM column names (e.g., `open_time`, `close_time`, `quote_asset_volume`, `taker_buy_*`) with UTC timestamps at millisecond precision, where `open_time` denotes the beginning of the bar.

#### Prompt (paste into any LLM)

"""
You are given the N2T Canon and a Strategy I/O Contract. Treat acronyms as binding terms: DPF, SCU, CLC, MCC, FOC, FTR, BTP, SDL, RBR, CAL, NRM, UQC, OSS, ARL, IRP, MND, DRG, RLB.

1. Do not invent new columns or names.
2. Enforce CLC.
3. Keep model inputs/outputs per MCC/FOC.
4. Only derive features listed in FTR.
5. Map forecasts to trades using SDL + RBR only.
6. Log ARL items for every run.
   """
