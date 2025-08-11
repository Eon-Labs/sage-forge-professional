### N2T Strategy I/O Docs

These docs standardize a portable, tabular Markdown I/O contract for zero-/few-shot time-series forecasters (e.g., `NX-AI/TiRex`) and downstream signal/risk design using **TiRex native terminology** aligned with actual architecture components.

- **Scope**: research & design (R&D) specifications only; no code.
- **Design tenets**: DRY, causality-first, audit-ready, TiRex-architecture-aligned.
- **How to use**: pick acronyms from the taxonomies, then fill the Strategy I/O Contract template.

#### Files

**Core Documentation**:
- `n2t-canon.md` — Core governance taxonomy (ranks 0–10; Tier A/B/C)
- `taxonomies-optional.md` — Optional tagging/classification taxonomies
- `strategy-io-contract.md` — **TiRex Native Pipeline Contract** with layer navigation (CONTEXT, TOKENIZED, PREDICTIONS, FEATURES, SIGNALS)
- `tirex-quickref.md` — TiRex-specific facts with native architecture terminology
- `tirex-vulnerability-analysis.md` — Comprehensive security analysis (52.8% TiRex safety assessment)
- `tirex-deployment-requirements.md` — Production deployment security with Guardian system integration

**TiRex Native Layer Architecture** (`layers/` subfolder):
- `context-layer.md` — Exchange data foundation (11 columns) - `context: torch.Tensor`
- `tokenized-layer.md` — **🎯 OPTIMIZATION FOCUS**: Input architecture analysis (2→8 features) - `PatchedUniTokenizer`
- `predictions-layer.md` — TiRex quantile outputs (4 columns) - `quantile_preds`  
- `features-layer.md` — Technical indicators (5 columns) - Post-processing
- `signals-layer.md` — Trading decisions (3 columns) - Trading logic
- `pipeline-dependencies.md` — Complete TiRex data flow analysis and dependency mapping

#### Source of Truth for Column Names

DSM (`data-source-manager`) is the authoritative source for **CONTEXT layer** column nomenclature and timestamp semantics. All Strategy I/O contracts MUST use DSM column names (e.g., `open_time`, `close_time`, `quote_asset_volume`, `taker_buy_*`) with UTC timestamps at millisecond precision, where `open_time` denotes the beginning of the bar.

#### TiRex Native Architecture

**Data Pipeline**: `CONTEXT → TOKENIZED → [sLSTM Processing] → PREDICTIONS → FEATURES → SIGNALS`

**Key Optimization**: TOKENIZED layer currently uses only 25% of TiRex's native `PatchedUniTokenizer` capacity (2/8 features). Full utilization provides **2-4x performance improvement**.

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
