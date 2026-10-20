# Third-party sources and notices

This project uses FinRL's actual Stable-Baselines3 agent construction interface and
draws design ideas from RLPortfolio. The local vendor directories are unmodified
Git checkouts kept for reproducible research; this project does not republish
either complete framework as its own implementation.

## FinRL

- Repository: [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL)
- Commit: `8511e7c15fbe1fbe5eb25e9f89932ac1ecd546a0`
- Copyright (c) 2024 AI4Finance Foundation Inc.
- License: MIT; complete upstream text is preserved in [docs/licenses/FinRL.txt](docs/licenses/FinRL.txt).
- Integration: `src/rlasset/finrl_adapter.py` loads the original
  `finrl/agents/stablebaselines3/models.py` and its dependencies. It skips the
  top-level package initializer to avoid unrelated live-broker/data dependencies.
  It does not modify the upstream agent implementation.
- FinRL's upstream README separately identifies FinRL and its logo as trademarks.
  This project is not presented as an official FinRL product or endorsement.

## RLPortfolio

- Repository: [CaioSBC/RLPortfolio](https://github.com/CaioSBC/RLPortfolio)
- Commit: `071cf75da9ddf95ec62210627b971086eec6f18f`
- Copyright (c) 2024 Caio de Souza Barbosa Costa.
- License: MIT; complete upstream text is preserved in [docs/licenses/RLPortfolio.txt](docs/licenses/RLPortfolio.txt).
- Design references: portfolio-weight drift, fee-aware log returns, portfolio
  state memory, and shared per-asset temporal encoding. The project's
  `SharedTemporalExtractor` is EIIE-inspired, not a complete EIIE reproduction;
  its SB3 dense actor/critic does not inherit full permutation equivariance.
  The project does not claim to reproduce RLPortfolio's PVM training loop,
  direct policy-gradient optimizer, or complete graph-policy framework.

Runtime dependencies, including Stable-Baselines3, PyTorch, Gymnasium and AKShare,
retain their respective upstream licenses. Research-paper citations and source
comparisons are recorded in `docs/research/`. Market-data snapshots retain source
provenance in adjacent metadata; the software's license does not grant ownership
of third-party data or independent data redistribution rights.
