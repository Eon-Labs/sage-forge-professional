# PPO: Project Prime Objective

- Domain: financial time series forecasting ("FTSF") for intraday directional trading.
- Design OHLCV-turning self-adaptive Nonparametric Predictive Alpha Factor ("NPPAF") that's Self-Adaptive Generative Evaluation ("SAGE").
- SAGE is quantitatively and adaptively assessed through parameter-free, regime-aware evaluation frameworks that discover optimal performance criteria from market structure rather than relying on fixed thresholds, ensuring robust nonparametric out-of-sample viability across market variations.

## Project Resources

The following should include minimalistic description and pointers to the paths that containts the resources only.

## DSM: Data Source Manager

- **Location**: `repos/data-source-manager` - Private Eon-Labs FCP-based Binance historical OHLCV data retrieval
- **Integration**: ✅ NT-native via `ArrowDataManager` + Apache Arrow MMAP optimization

## Research Documentation

- **Research Folder**: `docs/research/` - Theoretical foundation and algorithm taxonomy for NPAF/SAGE framework
  - **Motivation**: Research genesis and problem statement documentation
  - **Algorithm Taxonomy**: State-of-the-art parameter-free adaptive algorithms categorization (2024-2025)
  - **Implementation Pointers**: Future references for updating `docs/roadmap/` with practical NT-native methods

## Planning Documentation

- **Planning Folder**: `sage-forge-professional/docs/planning/` - Comprehensive implementation plans and architecture documentation
  - **TiRex-Native ODEB Architecture**: Complete Phase 3B implementation plan with canonicalized concepts
  - **Phase Implementation Tracking**: Structured development approach across multiple phases

## Project Glossary

- **Glossary Folder**: `sage-forge-professional/docs/glossary/` - Canonicalized concept definitions for precise boundaries
  - **TiRex-Native ODEB Concepts**: Comprehensive glossary of all "Name-It-To-Tame-It" terms with cross-references
  - **18 Canonicalized Terms**: Unique, representative, acronymizable concepts across 6 architectural layers
  - **Usage Guidelines**: Naming conventions, boundary enforcement, implementation standards

## FPPA: FinPlot Pattern Alignment

- FinPlot pattern is in the `~/eon/nt/repos/finplot`.
- Prefer `/Users/terryli/eon/nt/repos/finplot/finplot/examples/complicated.py` as the default template.
- Proactively conform to the native paradigm of FinPlot—including its provided classes, idiomatic patterns, and native conventions.

## NTPA: NautilusTrader Pattern Alignment

- NautilusTrader pattern is in the `~/eon/nt/repos/nautilus_trader`.
- Proactively conform to the native paradigm of NautilusTrader—including its provided classes, idiomatic patterns, and native conventions.

## CFUP: Claude-Flow Usage Pattern

- Claude-Flow officially recommended usage pattern is in the `~/eon/nt/repos/claude-flow`.
- **Command**: `npx claude-flow@alpha swarm "<objective>" --strategy research` for PPO enhancement
- **Hive-Mind**: `npx claude-flow@alpha hive-mind spawn "<complex-project>" --claude` for persistent sessions
- Proactively conform to Claude-Flow's multi-agent orchestration paradigm with specialized worker agents

## CCSS: Claude Code Session Sync

- **GPU Workstation Connection**: `zerotier-remote` via SSH (user: `tca`)
- **SSH Key**: `~/.ssh/id_ed25519_zerotier_np`
- **Enhanced Sync Tool**: `gpu-ws sync-all`, `gpu-ws push-all`, `gpu-ws pull-all`
- **Connection Test**: `ssh zerotier-remote "echo test"` or `gpu-ws` command
- **Comprehensive Sync**: Sessions, config, git, workspace, and environment files
- **Complete Documentation**: `docs/infrastructure/claude-code-session-sync-guide.md`
- **TiRex Integration Guide**: `docs/infrastructure/tirex-gpu-workstation-integration.md`
- **GPU-WS Design**: `docs/infrastructure/gpu-ws-comprehensive-sync-design.md`

## Financial Time Series Trading Optimization

- Recommend 2025 state-of-the-art, benchmark-validated, top-ranked algorithms implemented in off-the-shelf, future-proof, turnkey Python libraries that require minimal or no manual tuning—avoiding hardcoded thresholds or magic numbers. In other words, prioritize generalizability, auto-tuning capabilities, and integration-friendliness.

- Proactively research recent best practices for the host framework's native paradigm—including its provided classes, idiomatic patterns, and native conventions—and present your findings as a concise, reproducible, cookbook‑style reference.

- Proactively conform to the native paradigm of NautilusTrader—including its provided classes, idiomatic patterns, and native conventions.

## Output

- Minimize the number of lines by using Rich Progress and related functions.

## APCF: Audit-Proof Commit Format for SR&ED Evidence Generation

- **Usage**: Request "APCF" or "apcf" to trigger automated SR&ED-compliant commit message generation.
- **Third-Party Protection**: APCF automatically excludes third-party submodules from commits to prevent tampering.
- **Protected Repositories**: `repos/nautilus_trader`, `repos/finplot`, `repos/claude-flow`
- **Allowed Repositories**: `repos/data-source-manager` (Eon-Labs private)
- **Full Documentation**: See `/apcf` command for complete specifications, templates, and usage guidelines.

## AFPOE: Advices from Panel of Experts

- You are to ultrathink like "Quant of the Year Award" winners Jean-Philippe Bouchaud, Maureen O'Hara, Riccardo Rebonato, Petter Kolm, Campbell R. Harvey. and Marcos López de Prado to recommand state of the art novel follow-up design actions for intraday positions without ultra-low latency infrastructure or order-book data.
