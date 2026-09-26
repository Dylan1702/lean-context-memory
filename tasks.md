# Implementation Plan

- [ ] 1. Project scaffolding
  - Create project structure: `lean_context/` package with `router.py`, `storage.py`, `nodes.py`, `graph.py`, `cli.py`; `tests/`; `bench/`
  - Create `requirements.txt`: `langgraph`, `langchain-core`, `chromadb`, `ollama`, `sentence-transformers` (or confirm Chroma's default embedder is sufficient), `psutil` (for RSS memory sampling in benchmarks), `matplotlib` (for benchmark charts), `pytest`
  - Create `.gitignore` for `*.db`, `chroma_data/`, `__pycache__/`, venv
  - _Requirements: 10.1, 10.3_

- [ ] 2. SQLite storage layer (Tier 2)
  - Implement `storage.py::init_sqlite()` creating `memory_store.db` with WAL mode and `facts` table
  - Implement `get_fact(key)`, `set_fact(key, val)`, `get_all_facts()`
  - Write unit tests for insert/overwrite/read
  - _Requirements: 3.1, 3.2, 3.4, 3.5_

- [ ] 3. ChromaDB storage layer (Tier 3)
  - Implement `storage.py::init_chroma()` returning a persisted local collection `lean_context_narrative`
  - Implement `add_narrative(text, turn_index)` and `query_narrative(query, n_results=3)`
  - Verify embeddings run fully locally with no network calls
  - _Requirements: 4.1, 4.2, 4.3_

- [ ] 4. Zero-LLM router
  - Implement `router.py::cpu_router_edge(text) -> Literal["evict","persist","active"]`
  - Implement evict regex/stop-list, persist assignment-pattern regex
  - Write table-driven unit tests covering evict/persist/active cases and edge cases (empty, punctuation-only, mixed case)
  - Measure and assert classification time is well under the 5ms budget in a test
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 5. Graph state schema
  - Define `GraphState(TypedDict)` exactly as in design.md
  - _Requirements: 8.1_

- [ ] 6. Evict node — implements lifecycle state 5 (FORGET)
  - Implement `evict_node`: returns canned acknowledgment, touches no storage
  - Unit test: confirm no writes occur to RAM/SQLite/Chroma when this node runs
  - _Requirements: 5.1, 5.2_

- [ ] 7. Persist node — implements lifecycle state 3 (ARCHIVE)
  - Implement structured key-value extraction regex; on match, write via `storage.set_fact`
  - On no structured match, write narrative via `storage.add_narrative`
  - Unit tests for both extraction and narrative-fallback paths
  - _Requirements: 3.1, 4.1_

- [ ] 8. Active RAM node — implements lifecycle states 1 (KEEP) and 4 (RETRIEVE)
  - Implement RAM buffer read/append, `get_all_facts()` call, `query_narrative()` call
  - Cap and stage results into `retrieved_facts`
  - Unit test with a pre-seeded SQLite/Chroma fixture
  - _Requirements: 2.1, 2.2, 3.3, 4.2_

- [ ] 9. Minimal context assembly node
  - Implement the exact prompt template from design.md (system + facts + last-3 RAM turns + current query)
  - Unit test: assert prompt length does not grow when RAM history exceeds 3 turns (older turns excluded)
  - _Requirements: 6.1, 6.2_

- [ ] 10. Local inference node — implements lifecycle states 1 (KEEP) and 2 (COMPRESS)
  - Implement `ollama.generate()` call against `llama3.2:3b`
  - Implement `OllamaUnavailableError` handling for connection failures
  - Implement RAM buffer append (KEEP) + FIFO eviction at N>3 (COMPRESS)
  - Unit test with Ollama call mocked
  - _Requirements: 7.1, 7.2, 7.3, 2.2_

- [ ] 11. Graph assembly (LangGraph wiring)
  - Wire all nodes into `StateGraph` with conditional entry point per design.md diagram
  - Add `SqliteSaver` checkpointing on a separate checkpoint DB
  - Integration test: scripted 5-turn conversation (Ollama mocked) asserting RAM cap and correct routing at each turn
  - _Requirements: 8.1, 8.2, 8.3_

- [ ] 12. CLI / minimal chat interface
  - Implement `cli.py`: simple REPL loop calling the compiled graph per turn, printing `final_output`
  - Handle `OllamaUnavailableError` with a clear message instead of a crash/hang
  - _Requirements: 10.2_

- [ ] 13. Benchmark suite — 3-way comparison
  - Implement `bench/run_benchmark.py` with a `--mode` flag supporting three run modes:
    - `unmanaged`: full conversation history appended to every prompt, unbounded (Baseline A)
    - `fifo`: RAM truncated to last 3 turns only, no SQLite/ChromaDB backing (Baseline B — isolates the value of Tier 2/3)
    - `lean`: the full Lean-Context pipeline (all 3 tiers active)
  - For each mode, run the same scripted 50-turn conversation and record, per turn: TTFT (ms), total generation time (ms), process RSS (MB)
  - Output one combined CSV with a `mode` column, plus a simple line chart (TTFT vs turn, RSS vs turn) with all three modes overlaid, matching the shape of the graphs in the source doc (Section 9)
  - _Requirements: 9.1, 9.2, 9.4_

- [ ] 14. Recall accuracy test — 3-way comparison, multi-fact
  - Reuse the same scripted 50-turn conversation from Task 13, with 4 distinct facts planted at different turns (e.g. turn 2: project target, turn 10: dataset name, turn 25: a numeric parameter, turn 35: a preference) — vary turn distance so recall difficulty isn't uniform
  - At turn 50, ask 4 separate questions, one per planted fact, for all three modes (`unmanaged`, `fifo`, `lean`)
  - Score exact-match recall per fact per mode; report as a fraction (e.g. `lean: 4/4`, `fifo: 0/4`, `unmanaged: 2/4`) rather than a single pass/fail
  - Add this table to the benchmark script's output (or a separate `bench/recall_test.py`) so it's reproducible for the demo/defense
  - _Requirements: 9.3_

- [ ] 15. README and setup docs
  - Document Python version, `pip install -r requirements.txt`, Ollama model pull (`ollama pull llama3.2:3b`), and how to run the CLI and benchmark
  - _Requirements: 10.1, 10.3_

- [ ] 16. End-to-end manual smoke test
  - Run the CLI locally, exercise all three routing paths (say "hi", state a fact, ask a question), confirm behavior matches design.md
  - _Requirements: 1.1–1.4, 5.1, 3.1, 7.1_
