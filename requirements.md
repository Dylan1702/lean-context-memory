# Requirements Document

## Introduction

Lean-Context is a local, CPU-only chatbot memory system that keeps prompt size bounded (O(1)) regardless of conversation length, while retaining 100% recall of explicitly stated facts. It replaces LLM-based memory summarization (which doubles inference cost) with deterministic, sub-millisecond Python heuristics, and segregates memory into three tiers by volatility and structure: an in-RAM sliding buffer, a SQLite key-value store, and a ChromaDB vector store. It runs entirely offline against a local Ollama-served Llama 3.2 3B model.

## Requirements

### Requirement 1: Zero-LLM Input Router
**User Story:** As a system, I want to classify each incoming user message without calling an LLM, so that routing adds negligible latency.

#### Acceptance Criteria
1. WHEN a user message is received THEN the system SHALL classify it as EVICT, PERSIST, or ACTIVE using only regex/lexical rules, with no model call.
2. WHEN the message matches a social-filler pattern (e.g. "hi", "thanks", "ok", word count ≤ 2 from a known stop-list) THEN the system SHALL classify it as EVICT.
3. WHEN the message contains an assignment pattern (e.g. "remember that...", "my X is Y", "set X to Y", "key: value") THEN the system SHALL classify it as PERSIST.
4. WHEN the message matches neither EVICT nor PERSIST patterns THEN the system SHALL classify it as ACTIVE.
5. WHEN classification runs THEN the system SHALL complete it in under 5ms on commodity CPU hardware.

### Requirement 2: Volatile Active Memory (Tier 1)
**User Story:** As a system, I want to hold only the most recent conversational turns in RAM, so that prompt size stays bounded.

#### Acceptance Criteria
1. THE system SHALL maintain an active RAM buffer capped at N ≤ 3 turns (user+assistant pairs).
2. WHEN a new turn is added AND the buffer length exceeds 3 THEN the system SHALL evict the oldest turn (FIFO).
3. THE buffer SHALL be held in an in-memory Python structure (list or deque), not persisted to disk.

### Requirement 3: Structured Relational Memory (Tier 2)
**User Story:** As a user, I want facts I explicitly state (names, parameters, preferences) to be remembered exactly, no matter how long the conversation runs.

#### Acceptance Criteria
1. WHEN a message is classified PERSIST AND contains an extractable key-value pair THEN the system SHALL write it to a local SQLite database (`memory_store.db`).
2. THE SQLite store SHALL use WAL (write-ahead logging) mode.
3. WHEN an ACTIVE message is processed THEN the system SHALL query SQLite for all stored facts and include them in context.
4. Fact lookups SHALL complete in under 5ms.
5. WHEN the same key is persisted twice THEN the system SHALL overwrite (INSERT OR REPLACE), not duplicate.

### Requirement 4: Unstructured Semantic Memory (Tier 3)
**User Story:** As a system, I want to store and later retrieve narrative/contextual information that isn't a discrete key-value fact.

#### Acceptance Criteria
1. WHEN a PERSIST-classified message contains narrative content without a clean key-value match THEN the system SHALL store it as a document in a local, disk-persisted ChromaDB collection.
2. WHEN an ACTIVE message is processed THEN the system SHALL run a similarity query against ChromaDB and include the top-k relevant chunks in context.
3. Embeddings SHALL be generated locally (CPU-compatible embedding model), with no external API calls.

### Requirement 5: Context Purge (Evict Channel)
**User Story:** As a system, I want to discard social filler so it never inflates RAM, SQLite, or ChromaDB.

#### Acceptance Criteria
1. WHEN a message is classified EVICT THEN the system SHALL NOT write it to RAM buffer, SQLite, or ChromaDB.
2. WHEN a message is classified EVICT THEN the system SHALL return a lightweight canned acknowledgment without invoking the LLM.

### Requirement 6: Minimal Context Assembly
**User Story:** As a system, I want to assemble the smallest possible prompt that still contains everything needed to answer correctly.

#### Acceptance Criteria
1. THE assembled prompt SHALL consist of: static system instructions + retrieved SQLite/ChromaDB facts + at most the last 3 RAM turns + the current query.
2. THE assembled prompt token count SHALL NOT grow as a function of total conversation turns (t); it SHALL be bounded by a constant independent of t.

### Requirement 7: Local LLM Inference
**User Story:** As a user, I want responses generated entirely on my laptop with no cloud dependency.

#### Acceptance Criteria
1. WHEN an ACTIVE-routed message has its context assembled THEN the system SHALL send the prompt to a local Ollama instance running `llama3.2:3b`.
2. IF Ollama is not reachable at `localhost:11434` THEN the system SHALL surface a clear error rather than hanging indefinitely.
3. THE system SHALL append the resulting (user, assistant) turn to the Tier 1 RAM buffer.

### Requirement 8: LangGraph Orchestration
**User Story:** As a developer, I want the pipeline expressed as an explicit, inspectable state graph.

#### Acceptance Criteria
1. THE system SHALL implement routing, eviction, persistence, retrieval, context assembly, and inference as distinct LangGraph nodes.
2. THE graph SHALL use a conditional entry point keyed off the router's output (evict / persist / active).
3. THE graph state SHALL be checkpointed to disk (SqliteSaver or MemorySaver) so sessions survive application restarts.

### Requirement 9: Benchmarking & Evaluation
**User Story:** As the project owner, I want measurable proof that latency stays flat and recall stays perfect over long conversations.

#### Acceptance Criteria
1. THE system SHALL provide a benchmark script that runs ≥50 simulated turns and records TTFT per turn.
2. THE system SHALL provide a benchmark mode that measures process RSS memory over the same run.
3. THE system SHALL provide a recall-accuracy test that plants multiple facts (at least 4, at varying turn distances) early in a conversation and checks exact-match recall for each of them at turn 50, reporting a recall fraction per mode rather than a single pass/fail.
4. THE benchmark SHALL be runnable in three modes for comparison: (a) the Lean-Context pipeline, (b) an unmanaged baseline that appends full history indefinitely, and (c) a naive FIFO sliding-window baseline (N=3 turns, no SQLite/ChromaDB) that isolates whether Tier 2/3 persistence is actually necessary.

### Requirement 10: Local-Only Deployment
**User Story:** As a user on a personal laptop, I want to run the whole system with one command and no cloud accounts.

#### Acceptance Criteria
1. THE system SHALL run fully offline after initial model pull (no external API keys required).
2. THE system SHALL expose a simple CLI or minimal local chat interface for interactive use.
3. THE system SHALL document exact setup steps (Python version, dependencies, Ollama model pull) in a README.
