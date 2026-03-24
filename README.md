# openai-agents-context-compaction

[![PyPI version](https://img.shields.io/pypi/v/openai-agents-context-compaction.svg)](https://pypi.org/project/openai-agents-context-compaction/)
[![CI](https://github.com/damianoneill/openai-agents-context-compaction/actions/workflows/ci.yml/badge.svg)](https://github.com/damianoneill/openai-agents-context-compaction/actions/workflows/ci.yml)
[![Compatibility](https://github.com/damianoneill/openai-agents-context-compaction/actions/workflows/compatibility.yml/badge.svg)](https://github.com/damianoneill/openai-agents-context-compaction/actions/workflows/compatibility.yml)
[![Status](https://img.shields.io/badge/status-alpha-yellow.svg)](https://github.com/damianoneill/openai-agents-context-compaction)

Context compaction support for the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python), enabling intelligent management of conversation history token size for multi-turn agent interactions.

> **Note:** This package was created in response to [openai/openai-agents-python#2244](https://github.com/openai/openai-agents-python/issues/2244) to provide runner-level context compaction capabilities.

---

## Problem

As agent conversations grow longer, token counts sent to the LLM increase, leading to:

- **Higher costs** – LLM pricing is token-based
- **Degraded reasoning** – Models can lose focus with very long contexts
- **Context window limits** – Models have maximum token limits
- **Latency increases** – Longer prompts take more time to process

---

## Solution

This package extends the existing OpenAI Agents SDK `Session` protocol with **local compaction strategies**, making it **provider-agnostic** and independent of the OpenAI Responses API compaction endpoint.

> ⚠️ **Early-stage alpha:** The current release implements a minimal sliding window approach. Future releases will add token-aware, LLM-based, and pluggable strategies (see [Roadmap](#roadmap)).

---

## Installation

```bash
pip install openai-agents-context-compaction
```

### Optional: accurate token counting

By default, token counts are estimated using a simple ~4 chars/token heuristic. For accurate counts, install the `tiktoken` extra:

```bash
pip install 'openai-agents-context-compaction[tiktoken]'
```

**First-run download:** tiktoken downloads a small vocabulary file (~1 MB) from OpenAI's CDN on first use and caches it locally — subsequent runs are fully offline. If you use Docker, add both lines to your `Dockerfile` so the download happens at build time, not at runtime:

```dockerfile
RUN pip install 'openai-agents-context-compaction[tiktoken]'
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
```

Token counts are logged for observability. Token-budget compaction (keeping as many recent items as fit within N tokens) is also supported via the `token_budget` parameter — see [Token-budget compaction](#token-budget-compaction) below.

---

## Usage

Wrap any existing session with `LocalCompactionSession` to add automatic compaction support:

```python
from agents import Agent, Runner, SQLiteSession
from openai_agents_context_compaction import LocalCompactionSession

# Create your agent
agent = Agent(name="Assistant", instructions="You are a helpful assistant.")

# Wrap an existing session with compaction support
underlying = SQLiteSession("conversation_123")
session = LocalCompactionSession(underlying, window_size=30)

# Use normally - compaction happens automatically when needed
result = await Runner.run(agent, "Hello!", session=session)
```

- Compaction is **boundary-aware** – preserves function call pairs atomically
- **`limit` parameter**: Also boundary-aware — not a simple tail slice; function call pairs are kept atomic even when using `limit`
- When both `window_size` and `token_budget` are set, compaction stops when **either** limit is reached

### Token-budget compaction

Use `token_budget` to limit context by token count rather than (or in addition to) item count:

```python
from openai_agents_context_compaction import LocalCompactionSession, TiktokenCounter

# Token-budget only — keeps the most recent items that fit within 8000 tokens
session = LocalCompactionSession(
    underlying,
    token_budget=8000,
    token_counter=TiktokenCounter(),  # accurate OpenAI counts
)

# Both constraints — compaction stops when either is exhausted
session = LocalCompactionSession(
    underlying,
    window_size=50,
    token_budget=8000,
    token_counter=TiktokenCounter(),
)

# Custom tokenizer (e.g. Anthropic) — adapt to your SDK version
def my_counter(text: str) -> int:
    response = client.beta.messages.count_tokens(
        model="claude-haiku-4-5-20251001",  # any valid model works
        messages=[{"role": "user", "content": text}],
    )
    return response.input_tokens

session = LocalCompactionSession(underlying, token_budget=8000, token_counter=my_counter)
```

**Illustrative `token_budget` starting points (tune for your workload):**

- **Tight budget / small models**: `token_budget=4096`
- **Moderate**: `token_budget=16384`
- **Large models**: `token_budget=32768`

The default `token_counter` uses ~4 chars/token (no dependencies). For accurate counts pass `TiktokenCounter()` (requires `pip install 'openai-agents-context-compaction[tiktoken]'`).

### Choosing `window_size`

`window_size` is measured in **items**, not tokens. Each conversation turn adds multiple items:

| Scenario         | Items per turn                         | Guidance                                   |
| ---------------- | -------------------------------------- | ------------------------------------------ |
| Simple Q&A       | 2 (user + assistant)                   | `window_size=20` keeps ~10 exchanges       |
| Single tool call | 4 (user + fc + fco + assistant)        | `window_size=20` keeps ~5 tool-using turns |
| Batch tool calls | 2n+2 (user + n×fc + n×fco + assistant) | 3 parallel tools = 8 items/turn            |

**Illustrative starting points (not recommendations — tune for your workload):**

- **Light tool usage**: `window_size=30–50`
- **Heavy tool usage**: `window_size=50–100`

**Example:** If your agent typically calls 2 tools per turn, each turn produces ~6 items. With `window_size=30`, you retain roughly 5 recent exchanges.

### Technical Note

The OpenAI Agents SDK stores session data in [Responses API format](https://platform.openai.com/docs/api-reference/responses). Tool calls appear as separate `function_call` and `function_call_output` items matched by `call_id`. This package handles this transparently.

### Performance Considerations

For very large sessions (thousands of items), compaction runs on every `get_items()` call. No in-process cache is kept — each call fetches from the underlying session to avoid stale reads when the session is backed by a shared database with concurrent writers. The compaction algorithm itself is O(n) where n is the total session size. If performance becomes a concern:

- Consider periodic session pruning at the storage layer
- Use a reasonable `window_size` that balances context retention with processing cost

---

## Roadmap

| Feature                       | Status                                    |
| ----------------------------- | ----------------------------------------- |
| Sliding window compaction     | ✅ Implemented                            |
| Token-based limits            | ✅ Implemented (`token_budget` parameter) |
| LLM-based summarization       | 🟡 Planned                                |
| Write-time compaction         | 🟡 Planned                                |
| Pluggable compaction policies | 🟡 Planned                                |

---

## Write-time Compaction Caveats

| Aspect             | Read-time (current)          | Write-time            |
| ------------------ | ---------------------------- | --------------------- |
| Full history       | ✅ Preserved for audit/debug | ❌ Lost forever       |
| Change window_size | ✅ Retroactive               | ❌ Requires re-import |
| Storage            | ❌ Unbounded growth          | ✅ Bounded            |
| Read cost          | ❌ Compaction on every read  | ✅ Cheap reads        |

**Write-time compaction caveat:**
Function call pairs arrive in stages:

1. `add_items([function_call])` — incomplete (no output yet)
2. `add_items([function_call_output])` — now complete

If compaction runs at write-time, incomplete pairs may be dropped, breaking session integrity.

**Recommendation:** Use read-time compaction (current default) to guarantee atomic function call pairs unless storage size is critical. Write-time compaction requires redesign of `add_items` to handle incomplete pairs safely.

---

## Future Considerations

The following ideas are documented for future reference. Build them when there's a concrete need:

| Feature                        | Add when...                                                  |
| ------------------------------ | ------------------------------------------------------------ |
| **Time-based window**          | Sessions span days and old context becomes stale             |
| **Importance scoring**         | Tool outputs vary wildly in value                            |
| **Pluggable policy interface** | Multiple policies need swapping                              |
| **Role-based prioritization**  | Certain messages must never be evicted                       |
| **Hybrid window**              | Last N items + always keep M most recent function call pairs |

A pluggable policy interface would look like:

```python
class CompactionPolicy(Protocol):
    def compact(self, items: list[TResponseInputItem]) -> list[TResponseInputItem]:
        """Return compacted items. Must preserve function call pair atomicity."""
        ...
```

---

## Compatibility

Tested weekly against the latest OpenAI Agents SDK to ensure compatibility.

---

## License

MIT
