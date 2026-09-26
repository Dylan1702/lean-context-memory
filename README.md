# Lean-Context Memory Management for Fast Local AI Chatbots

A local, CPU-only chatbot memory system that keeps LLM prompt size **O(1)** regardless of conversation length, while retaining **100% recall** of explicitly stated facts.

Runs entirely offline against a local [Ollama](https://ollama.com) instance serving `llama3.2:3b`. No cloud APIs, no GPU required.

---

## How it works

Every user turn passes through a zero-LLM regex router, then one of three memory branches, before a minimal-context prompt is assembled and sent to the local LLM.

```
User Turn → cpu_router_edge (regex, <5 ms)
              ├─ evict   → canned ack, no storage       (FORGET)
              ├─ persist → SQLite / ChromaDB write       (ARCHIVE)
              └─ active  → retrieve + assemble + infer  (KEEP + RETRIEVE)
```

### 5-State Memory Lifecycle

| State | What it does | Where |
|---|---|---|
| **KEEP** | Appends user+assistant turn to RAM buffer | `active_ram_node`, `local_inference_node` |
| **COMPRESS** | Evicts oldest RAM turn when buffer > 3 (FIFO) | `local_inference_node` |
| **ARCHIVE** | Writes facts to SQLite or narrative to ChromaDB | `persist_node` |
| **RETRIEVE** | Queries SQLite + ChromaDB for current context | `active_ram_node` |
| **FORGET** | Drops social filler before it reaches any storage | `evict_node` |

### 3-Tier Memory

| Tier | Store | What goes there |
|---|---|---|
| 1 — RAM buffer | Python list (in-memory) | Last 3 user+assistant turns |
| 2 — SQLite | `memory_store.db` (WAL mode) | Explicit key-value facts (`my X is Y`) |
| 3 — ChromaDB | `chroma_data/` (on-disk) | Narrative chunks, similarity-searched |

---

## Requirements

- **Python 3.11 or 3.13** (tested on 3.13.7)
- **[Ollama](https://ollama.com/download)** installed and running (`ollama serve`)
- ~2 GB disk space for the `llama3.2:3b` model weights
- ~90 MB for the `all-MiniLM-L6-v2` embedding model (auto-downloaded on first run)

---

## Setup

### 1. Install Ollama and pull the model

Download Ollama from https://ollama.com/download and install it, then:

```powershell
ollama pull llama3.2:3b
ollama serve          # keep this running in a separate terminal
```

### 2. Create a virtual environment (recommended)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
# or: source .venv/bin/activate  (Linux/macOS)
```

### 3. Install Python dependencies

```powershell
pip install -r requirements.txt
```

Key packages and their roles:

| Package | Role |
|---|---|
| `langgraph` | State machine orchestration |
| `langchain-core` | LangGraph dependency |
| `langgraph-checkpoint-sqlite` | Persistent graph checkpointing |
| `chromadb` | Local vector store (Tier 3) |
| `ollama` | Python client for local Ollama inference |
| `sentence-transformers` | CPU-only embeddings (`all-MiniLM-L6-v2`) |
| `psutil` | RSS memory sampling in benchmarks |
| `matplotlib` | Benchmark charts |
| `pytest`, `pytest-mock` | Test suite |

---

## Running the chatbot

```powershell
python -m lean_context.cli
```

or equivalently:

```powershell
python lean_context/cli.py
```

### CLI commands

| Input | What happens |
|---|---|
| Any text | Routed through the pipeline, response printed |
| `hi`, `thanks`, `ok` | Evicted immediately — no LLM call, canned ack |
| `my name is Alice` | Persisted to SQLite; fact recalled in future turns |
| `/facts` | Prints all stored SQLite facts |
| `/clear` | Resets the conversation thread (new checkpoint) |
| `quit` / `exit` | Exits the REPL |

### Example session

```
You: my name is Alice
Bot: Got it, Alice! I've noted that down.

You: my project is lean-context memory
Bot: Understood — your project is lean-context memory.

You: What is my name?
Bot: Your name is Alice.

You: /facts
[Stored facts]
  name: Alice
  project: lean-context memory

You: hi
Bot: Got it!

You: quit
Bye!
```

---

## Running the tests

```powershell
python -m pytest tests/ -v
```

Expected: **113 tests, all passing** in ~40 seconds.

To run a single test file:

```powershell
python -m pytest tests/test_router.py -v          # router classification + timing
python -m pytest tests/test_graph_integration.py -v  # 5-turn integration test
```

---

## Benchmarking

### 3-way latency + memory benchmark

Runs a scripted 50-turn conversation in three modes and produces a combined CSV and two line charts.

```powershell
# Run all three modes and generate charts
python bench/run_benchmark.py --mode all

# Run a single mode
python bench/run_benchmark.py --mode lean
python bench/run_benchmark.py --mode fifo
python bench/run_benchmark.py --mode unmanaged
```

Output files (in `bench/output/`):

| File | Contents |
|---|---|
| `benchmark_results.csv` | Per-turn: mode, turn, ttft_ms, rss_mb |
| `ttft_chart.png` | TTFT vs turn, all modes overlaid |
| `rss_chart.png` | RSS vs turn, all modes overlaid |

**Modes:**
- `lean` — full pipeline (SQLite + ChromaDB + RAM cap). Prompt size stays flat.
- `fifo` — RAM sliding window only, N=3, no persistence. Isolates Tier 2/3 value.
- `unmanaged` — full unbounded history. TTFT and RSS grow with conversation length.

### Recall accuracy test

Plants 4 facts at turns 2, 10, 25, 35 of a 50-turn conversation, then asks recall questions at turn 50+.

```powershell
python bench/recall_test.py
```

Expected output shape:

```
RECALL ACCURACY RESULTS
────────────────────────────────────────
Fact               lean      fifo    unmanaged
────────────────────────────────────────
project_target      ✓         ✗          ✓
dataset             ✓         ✗          ✓
learning_rate       ✓         ✗          ✗
batch_size          ✓         ✗          ✗
────────────────────────────────────────
TOTAL              4/4       0/4        2/4
────────────────────────────────────────
```

Results also saved to `bench/output/recall_results.json`.

---

## Project structure

```
lean-context-memory/
├── lean_context/
│   ├── __init__.py
│   ├── router.py      # Zero-LLM regex router (cpu_router_edge)
│   ├── storage.py     # SQLite (Tier 2) + ChromaDB (Tier 3) layers
│   ├── nodes.py       # LangGraph nodes + GraphState + 5-state lifecycle
│   ├── graph.py       # StateGraph assembly + SqliteSaver checkpointing
│   └── cli.py         # Interactive REPL
├── tests/
│   ├── test_router.py              # 57 router unit tests + timing
│   ├── test_storage_sqlite.py      # 7 SQLite unit tests
│   ├── test_storage_chroma.py      # 6 ChromaDB unit tests
│   ├── test_nodes.py               # 23 node unit tests
│   ├── test_context_assembly.py    # 8 context assembly tests
│   ├── test_inference_node.py      # 7 inference node tests (Ollama mocked)
│   └── test_graph_integration.py   # 5 end-to-end integration tests
├── bench/
│   ├── run_benchmark.py   # 3-way latency + RSS benchmark
│   └── recall_test.py     # 3-way, 4-fact recall accuracy test
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Troubleshooting

**"Local model unavailable — is `ollama serve` running?"**
→ Start Ollama: `ollama serve` in a separate terminal.

**"No module named 'sentence_transformers'"**
→ Run: `pip install sentence-transformers`

**ChromaDB symlink warning on Windows**
→ Harmless; occurs because Windows requires Developer Mode for symlinks.
   The cache still works correctly without symlinks.

**Slow first run**
→ The `all-MiniLM-L6-v2` embedding model (~90 MB) is downloaded automatically
   on first use and cached at `~/.cache/huggingface/`. Subsequent runs are fast.

---

## Design traceability

Every function that implements a lifecycle state carries a comment referencing
the state name (`KEEP`, `COMPRESS`, `ARCHIVE`, `RETRIEVE`, `FORGET`) so the
implementation stays traceable to the design document for project review.

See `design.md` for the full architecture, data models, and error handling spec.
