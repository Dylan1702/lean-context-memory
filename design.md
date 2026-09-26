# Design Document

## Overview

Lean-Context is a LangGraph state machine that sits between a chat UI/CLI and a local Ollama runtime. Every user turn passes through a zero-LLM router, then one of three branches (evict / persist / active), then (for active/persist) a retrieval + minimal-context-assembly step, then local inference. State is bounded: RAM holds only the last 3 turns; everything durable lives in SQLite (facts) or ChromaDB (narrative), never in the prompt itself beyond what's retrieved for the current query.

## Architecture

```
User Turn → cpu_router_edge (regex, <5ms)
              ├─ evict   → evict_node → END
              ├─ persist → persist_node → active_ram_node → assemble_context → llm_infer → END
              └─ active  →                active_ram_node → assemble_context → llm_infer → END
```

Single LangGraph `StateGraph` with a conditional entry point. No node calls an LLM except `llm_infer`.

## The 5-State Memory Lifecycle

The source document frames every turn as passing through five named states. Each maps directly onto a component below — code and comments should use these names so the implementation stays traceable to the design:

| State | What it does | Implemented in |
|---|---|---|
| **1. KEEP** | Appends the incoming user query + assistant reply to active RAM | `active_ram_node` (append) / `local_inference_node` (append after generation) |
| **2. COMPRESS** | When RAM buffer length > 3, ejects the oldest turn (FIFO) | `local_inference_node` (buffer trim step) |
| **3. ARCHIVE** | Writes structured facts to SQLite or narrative chunks to ChromaDB | `persist_node` |
| **4. RETRIEVE** | Queries SQLite + ChromaDB for context relevant to the current query | `active_ram_node` (facts + narrative lookup) |
| **5. FORGET** | Drops zero-entropy social filler before it ever reaches RAM or storage | `evict_node`, gated by `cpu_router_edge` |

Note the router (Requirement 1) decides EVICT vs PERSIST vs ACTIVE *before* any state runs; FORGET short-circuits the other four states entirely for that turn.

## Components and Interfaces

### 1. Router (`cpu_router_edge`)
- Pure function: `str -> Literal["evict", "persist", "active"]`
- Regex-based, no I/O, no LLM call.
- Order of checks matters: evict-pattern check first (cheapest, most common), then persist-assignment check, else active.

### 2. Evict Node
- Returns a static acknowledgment string.
- Never touches RAM buffer, SQLite, or ChromaDB.

### 3. Persist Node
- Attempts structured extraction first: `my (\w+) is (.+)`, `set (\w+) to (.+)`, `key: value`.
  - Match → write to SQLite `facts` table (`INSERT OR REPLACE`).
  - No match but content is substantive (length/sentence heuristic) → write as a document to ChromaDB with turn metadata (timestamp, turn index).
- Falls through to `active_ram_node` so the persisted turn is still acknowledged conversationally.

### 4. Active RAM Node
- Reads current RAM buffer from graph state.
- Queries SQLite for all facts (small table, full scan is fine at this scale; index on `key` regardless).
- Queries ChromaDB (`collection.query(query_texts=[user_input], n_results=3)`) for narrative chunks relevant to the current input.
- Stages both into `retrieved_facts` in state.

### 5. Context Assembly Node
- Implements: `P = S_system + F_retrieved + M(t-N..t)`
- Truncates RAM history to last 3 turns even if buffer briefly holds more mid-transition.
- Deterministic string template (see requirements doc's "Assembled Prompt Layout").

### 6. Inference Node
- Calls `ollama.generate(model="llama3.2:3b", prompt=lean_prompt)`.
- On connection failure, raises a typed `OllamaUnavailableError` caught at the API/CLI boundary and surfaced as a clear user-facing message (not a raw stack trace or silent hang).
- On success, appends `{user, assistant}` to RAM buffer, evicting oldest if length > 3.

### 7. Graph State (`TypedDict`)
As specified in the source document section 7 — `user_input`, `routing_flag`, `ram_buffer`, `extracted_facts`, `retrieved_facts`, `lean_prompt`, `final_output`.

### 8. Checkpointing
- `SqliteSaver` (LangGraph built-in) pointed at a separate checkpoint DB file (distinct from `memory_store.db`) so graph state — not just facts — survives restarts.

## Data Models

**SQLite `facts` table**
```sql
CREATE TABLE IF NOT EXISTS facts (
    key TEXT PRIMARY KEY,
    val TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```
PRAGMA `journal_mode=WAL` set on connection open.

**ChromaDB collection**
- Name: `lean_context_narrative`
- Document = raw narrative text
- Metadata: `{"turn_index": int, "timestamp": iso8601}`
- Embedding function: a lightweight local CPU model (e.g. `all-MiniLM-L6-v2` via `sentence-transformers`, or Chroma's default local embedding function) — no OpenAI/cloud embeddings.

**RAM buffer element**
```python
{"user": str, "assistant": str}
```

## Error Handling

- Ollama unreachable → typed exception, caught at CLI boundary, user sees "Local model unavailable — is `ollama serve` running?" rather than a crash.
- SQLite write failure (e.g. locked file) → retry once with short backoff, then log and continue without blocking the conversation.
- ChromaDB query failure → degrade gracefully: proceed with empty `retrieved_facts` from that tier rather than failing the whole turn.
- Regex extraction failure on a PERSIST-routed message → falls back to ChromaDB narrative storage rather than silently dropping the message.

## Testing Strategy

- **Unit tests** for `cpu_router_edge`: table-driven tests covering evict/persist/active classification, including edge cases (empty string, punctuation-only, mixed case).
- **Unit tests** for extraction regex in `persist_node`.
- **Integration test**: run a scripted 5-turn conversation through the compiled graph with Ollama mocked, assert RAM buffer never exceeds 3 turns and facts land in SQLite.
- **Benchmark script** (per Requirement 9): 50-turn run measuring TTFT and RSS across three modes — `unmanaged` (full history), `fifo` (RAM-only, N=3, no persistence), and `lean` (full pipeline) — output as a combined CSV/plot for direct comparison.
- **Recall test**: plant a fact at turn 2, ask about it at turn 50, run under all three modes and compare exact-match results (expect `fifo` to fail, `unmanaged` to degrade or hit context limits, `lean` to score 100%).
