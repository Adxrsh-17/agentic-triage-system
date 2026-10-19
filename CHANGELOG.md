# Changelog

All notable changes to the **Agentic Clinical Triage Copilot** project will be documented in this file.

## [2.0.0] - 2026-10-19

### Added
- **Phase 0 (Audit):** Added `docs/audit.md` documenting node behaviors, vendor integrations, and isolating legacy booking/dispatch tools to `archive/legacy/`.
- **Phase 1 (Hybrid Intake & Risk):** Added `agent/schemas.py` (`RiskLevel`, `IntakeResult`, `RiskResult`), `agent/rules.py` (deterministic safety floor with NegEx conjunction and clause boundary parsing), and `agent/llm.py` (ChatGroq client with structured output and fail-fast fallbacks).
- **Phase 2 (Embeddings & Memory):** Integrated `sentence-transformers/all-MiniLM-L6-v2` (384-dim) in `agent/memory/embeddings.py` and implemented `PatientMemory` protocol with `LocalMemory` and `PineconeMemory` in `agent/memory/store.py`.
- **Phase 3 (Evaluation):** Created 150-case benchmark `eval/cases.jsonl` with dev/test split, `eval/run_eval.py` evaluation harness, and `docs/error_analysis.md` systematic autopsy.
- **Phase 4 (Tests & CI):** Added `tests/test_rules.py`, `tests/test_risk_hybrid.py`, `tests/test_memory.py`, `tests/test_graph_routing.py`, `tests/test_safety_validator.py`, `requirements-dev.txt`, and `.github/workflows/ci.yml`.
- **Phase 5 (Repo Polish):** Added `Makefile`, `CONTRIBUTING.md`, `docs/architecture.md`, and refreshed `README.md`.
- **Phase 6 (Safety & Ethics):** Added `docs/safety.md` detailing clinical governance, failure modes, and privacy mitigations.

### Changed
- Updated `.env.example` with `PINECONE_DIMENSION=384` and `LLM_MODEL=llama-3.3-70b-versatile`.
- Configured Ruff linter and pytest in `pyproject.toml`.
