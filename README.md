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

> ⚠️ **Early-stage alpha:** The current release implements sliding window and LLM summarisation strategies behind a pluggable policy interface (see [Roadmap](#roadmap)).

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

### LLM-based summarisation

Use `SummarizingPolicy` to compress older conversation history into a single summary item while keeping the most recent turns raw. The policy requires a `HistorySummarizer` implementation that you provide — any LLM or service that can condense a list of messages works.

```python
from agents import TResponseInputItem
from openai_agents_context_compaction import (
    LocalCompactionSession,
    SummarizingPolicy,
)


class MyOpenAISummarizer:
    """Example summarizer using the OpenAI Chat API."""

    async def summarize(
        self,
        items: list[TResponseInputItem],
        *,
        session_id: str,
        target_tokens: int | None = None,
        existing_summary: str | None = None,
    ) -> TResponseInputItem:
        # Call your LLM to produce a summary of `items`.
        # Return a single synthetic message item.
        summary_text = await call_openai(items, max_tokens=target_tokens)
        return {
            "role": "assistant",
            "type": "message",
            "content": [{"type": "output_text", "text": summary_text}],
        }


policy = SummarizingPolicy(
    MyOpenAISummarizer(),
    retain_recent_turns=2,              # keep at least 2 recent turns raw (floor)
    retain_recent_token_budget=20000,   # expand tail up to 20k tokens (ceiling)
    summary_target_tokens=1500,         # hint for summary length
    summarizer_input_token_limit=60000, # truncate prefix before sending to summariser
    summary_timeout_seconds=5.0,        # cancel if summariser is too slow
)

session = LocalCompactionSession(underlying, policy=policy)
```

When `get_items()` is called, the policy:

1. Selects a recent tail using turn-aware retention (floor + ceiling: at least N turns, expand if token budget allows)
2. Truncates the older prefix to `summarizer_input_token_limit` tokens (dropping oldest items first, preserving pair atomicity)
3. Sends the prefix to the summariser
4. Returns `[summary_item] + recent_tail`

If the summariser raises or times out, the policy falls back to `SlidingWindowPolicy` with the same tail budget (set `fallback_to_sliding_window=False` to propagate the exception instead).

### Custom policies

You can pass any object implementing the `CompactionPolicy` protocol:

```python
session = LocalCompactionSession(underlying, policy=my_custom_policy)
```

See `CompactionPolicy` and `CompactionResult` in the [Future Considerations](#future-considerations) section for the full interface.

### Performance considerations

For very large sessions (thousands of items), compaction runs on every `get_items()` call. No in-process cache is kept — each call fetches from the underlying session to avoid stale reads when the session is backed by a shared database with concurrent writers. The compaction algorithm itself is O(n) where n is the total session size. If performance becomes a concern:

- Consider periodic session pruning at the storage layer
- Use a reasonable `window_size` or `retain_recent_turns` that balances context retention with processing cost

---

## Roadmap

| Feature                       | Status                                              |
| ----------------------------- | --------------------------------------------------- |
| Sliding window compaction     | ✅ Implemented (`SlidingWindowPolicy`)              |
| Token-based limits            | ✅ Implemented (`token_budget` parameter)           |
| Pluggable compaction policies | ✅ Implemented (`CompactionPolicy` protocol)        |
| LLM-based summarisation       | ✅ Implemented (`SummarizingPolicy`)                |
| Write-time compaction         | 🟡 Planned                                          |

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

| Feature                        | Add when...                                              |
| ------------------------------ | -------------------------------------------------------- |
| **Time-based window**          | Sessions span days and old context becomes stale         |
| **Importance scoring**         | Tool outputs vary wildly in value                        |
| **Role-based prioritisation**  | Certain messages must never be evicted                   |
| **Hybrid window**              | Last N items + always keep M most recent function pairs  |
| **Incremental summarisation**  | Long-running sessions need repeated re-summarisation     |

### Policy interface

The library exposes an async `CompactionPolicy` protocol. The built-in policies (`SlidingWindowPolicy`, `SummarizingPolicy`) implement it, and custom policies can do the same:

```python
class CompactionResult(TypedDict):
    items: list[TResponseInputItem]
    original_count: int
    returned_count: int
    limiting_factor: str | None
    summary_generated: bool
    metadata: dict[str, object]


class CompactionPolicy(Protocol):
    async def compact(
        self,
        items: list[TResponseInputItem],
        *,
        session_id: str,
        limit: int | None = None,
    ) -> CompactionResult:
        """Return compacted items. Must preserve function call pair atomicity."""

    def get_config(self) -> dict[str, object]:
        """Return a serializable description of the policy configuration."""
        ...


class HistorySummarizer(Protocol):
    async def summarize(
        self,
        items: list[TResponseInputItem],
        *,
        session_id: str,
        target_tokens: int | None = None,
        existing_summary: str | None = None,
    ) -> TResponseInputItem:
        """Return one synthetic history item summarising the provided items."""
```

---

## Compatibility

Tested weekly against the latest OpenAI Agents SDK to ensure compatibility.

---

## License

MIT
