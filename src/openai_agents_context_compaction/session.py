"""Local compaction session wrapper.

Provides a Session wrapper that delegates calls to an underlying session
and optionally applies a sliding window compaction strategy.

COMPACTION PIPELINE:
====================

    ┌───────────────────────────────────────────────────────────────────────────┐
    │  Session Items (chronological order)                                      │
    │  [user_msg, func_call(a), func_call(b), output(a), output(b), asst_msg,   │
    │   func_call(orphan), user_msg]                                            │
    └───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌───────────────────────────────────────────────────────────────────────────┐
    │  1. INDEX MAPPING (boundary_aware_compact)                                │
    │     Build call_id → {call_index, output_index} map                        │
    │     Identify complete pairs (both call and output present)                │
    └───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌───────────────────────────────────────────────────────────────────────────┐
    │  2. BACKWARD WALK WITH BUDGET (boundary_aware_compact)                    │
    │     Walk backwards from most recent item                                  │
    │     Track budget (remaining window capacity)                              │
    │     - function_call_output (complete pair): reserve budget for BOTH       │
    │       output and its matching call (budget -= 2), mark both indices       │
    │     - function_call (already marked): skip (already budgeted)             │
    │     - function_call (orphan/pair not selected): skip                      │
    │     - regular message / unknown type: include if budget > 0              │
    │     Items are marked at their ORIGINAL indices                            │
    └───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌───────────────────────────────────────────────────────────────────────────┐
    │  3. COLLECT IN ORIGINAL ORDER                                             │
    │     Gather marked items by ascending index                                │
    │     Original chronological ordering is preserved                          │
    └───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌───────────────────────────────────────────────────────────────────────────┐
    │  4. DROP ORPHANS (drop_orphaned_tool_outputs)                             │
    │     Safety net: remove any function_call without output, or vice versa    │
    └───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌───────────────────────────────────────────────────────────────────────────┐
    │  Result (chronological order — original ordering preserved)               │
    │  [asst_msg, func_call(b), output(b), user_msg]                            │
    └───────────────────────────────────────────────────────────────────────────┘

Order Preservation:
- Unlike a pair-bundling approach that groups fc/fco together then reassembles,
  this algorithm marks items at their original indices and collects them
  in order. This means items that appeared between a function_call and its
  output remain in their original position relative to both.
- Example: [fc(a), fc(b), output(a), output(b)] stays in that order,
  NOT reordered to [fc(a), output(a), fc(b), output(b)].

Responses API Format (used by OpenAI Agents SDK):
- function_call: {"type": "function_call", "call_id": "...", "name": "...", "arguments": "..."}
- function_call_output: {"type": "function_call_output", "call_id": "...", "output": "..."}
- user message: {"role": "user", "content": "..."}
- assistant message: {"role": "assistant", "type": "message", "content": [...]}

Note on System Instructions:
- In the OpenAI Agents SDK, system instructions (Agent.instructions) are ephemeral
- They are passed directly to the model at runtime, NOT stored in session history
- Session items never include {"role": "system", ...} messages
- This compaction code handles all item types that actually appear in sessions

Why backwards? Because we want to keep the MOST RECENT items, so we start
from the end and work back until we've filled the window.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypedDict

from agents import Session, TResponseInputItem

# NOTE: Verbose DEBUG logging is intentional during alpha; callers control via logging config.
logger = logging.getLogger(__name__)


class DefaultTokenCounter:
    """Default token counter: ~4 chars/token estimate (model-agnostic).

    Used when no token_counter is specified. Provides a rough estimate
    suitable for observability without requiring tiktoken.
    """

    def __call__(self, text: str) -> int:
        """Count tokens using ~4 chars/token estimate."""
        return max(1, len(text) // 4)


# Singleton instance for default parameter
_default_token_counter = DefaultTokenCounter()


class TiktokenCounter:
    """Token counter using tiktoken (requires ``pip install tiktoken``).

    Args:
        encoding_name: Tiktoken encoding name. Common values:
            'o200k_base' (GPT-4o/4.1), 'cl100k_base' (GPT-3.5/4).

    Raises:
        ImportError: If tiktoken is not installed.
        ValueError: If ``encoding_name`` is not a recognised tiktoken encoding.
        RuntimeError: If tiktoken fails to download the vocabulary file on first
            use (e.g. network unavailable in an air-gapped environment).

    Example:
        >>> counter = TiktokenCounter("o200k_base")
        >>> session = LocalCompactionSession(underlying, token_counter=counter)
    """

    def __init__(self, encoding_name: str = "o200k_base") -> None:
        import tiktoken

        self.encoding_name = encoding_name
        self._enc = tiktoken.get_encoding(encoding_name)

    def __call__(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._enc.encode(text))


# =============================================================================
# Token counting helpers
# =============================================================================
#
# FUTURE: These helpers lay the groundwork for token-based compaction.
# Currently, token counts are only logged (observability / baseline data).
# The next step is a compaction mode that enforces a token budget rather than
# an item-count window — keeping as many recent items as fit within N tokens.
# =============================================================================


def _extract_text(item: TResponseInputItem) -> str:
    """Extract all text content from a Responses API item for token counting."""
    parts: list[str] = []

    if _is_function_call(item):
        # name + JSON-encoded arguments
        parts.append(str(item.get("name", "")))
        parts.append(str(item.get("arguments", "")))
        return " ".join(parts)

    if _is_function_call_output(item):
        # output can be a plain string or a list of {"type": "input_text", "text": "..."} objects
        output = item.get("output", "")
        if isinstance(output, str):
            parts.append(output)
        elif isinstance(output, list):
            for part in output:
                if isinstance(part, dict):
                    parts.append(part.get("text", ""))
        return " ".join(parts)

    # user / assistant message: content can be a plain string or a list of content blocks
    content = item.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(block["text"])
                elif "input_text" in block:
                    parts.append(block["input_text"])
                elif "output_text" in block:
                    parts.append(block["output_text"])
                else:
                    logger.debug("Unknown content block shape: %r", block)
    elif content is None and ("type" in item or "role" in item):
        logger.warning(
            "No content field on item type=%r role=%r. Item: %r",
            item.get("type"),
            item.get("role"),
            item,
        )

    return " ".join(parts)


def _count_tokens(
    items: list[TResponseInputItem],
    token_counter: Callable[[str], int] = _default_token_counter,
) -> int:
    """Estimate total token count for a list of Responses API items.

    Each item's text is passed to ``token_counter`` to get raw token count,
    then a +4 overhead is added per item for message-structure tokens
    (role, separators, etc.).

    Args:
        items: List of Responses API items to count tokens for.
        token_counter: A callable that maps text to a token count.
            Defaults to ~4 chars/token estimate (model-agnostic).

    Returns:
        Estimated total token count.
    """
    total = 0
    for item in items:
        text = _extract_text(item)
        total += token_counter(text) + 4
    return total


# =============================================================================
# Helper functions for Responses API format detection
# =============================================================================


def _is_function_call(item: TResponseInputItem) -> bool:
    """Check if item is a function call (Responses API format)."""
    return item.get("type") == "function_call"


def _is_function_call_output(item: TResponseInputItem) -> bool:
    """Check if item is a function call output (Responses API format)."""
    return item.get("type") == "function_call_output"


def _get_call_id(item: TResponseInputItem) -> str | None:
    """Get call_id from function_call or function_call_output."""
    call_id = item.get("call_id")
    return str(call_id) if call_id is not None else None


def _is_conversation_message(item: TResponseInputItem) -> bool:
    """Check if item is a conversation message (user/assistant, not tool calls)."""
    # User messages have role but no type, or type != function_call
    # Assistant messages have role=assistant and type=message
    # Note: system messages don't appear in session history (SDK invariant)
    if _is_function_call(item) or _is_function_call_output(item):
        return False
    return item.get("role") in ("user", "assistant")


class _CallIndices(TypedDict):
    """Indices for a function call pair (call and output).

    Used internally by _boundary_aware_compact to track positions.
    Values are None until filled during index mapping; only entries
    where both are int make it into complete_pairs.
    """

    call: int | None
    output: int | None


def _boundary_aware_compact_with_indices(
    items: list[TResponseInputItem],
    window_size: int,
) -> tuple[list[TResponseInputItem], list[int]]:
    """Internal: compact history and return both items and their original indices.

    Returns:
        Tuple of (compacted_items, kept_indices) where kept_indices are the
        original positions of the kept items.
    """
    if window_size <= 0:
        return [], []

    if len(items) <= window_size:
        # All items fit — return as-is with all indices
        return list(items), list(range(len(items)))

    # Build call_id -> indices map
    call_to_indices: dict[str, _CallIndices] = {}

    for idx, item in enumerate(items):
        call_id = _get_call_id(item)
        if call_id is None:
            continue

        if call_id not in call_to_indices:
            call_to_indices[call_id] = {"call": None, "output": None}

        if _is_function_call(item):
            if call_to_indices[call_id]["call"] is not None:
                logger.debug("Duplicate function_call call_id=%s (overwriting)", call_id)
            call_to_indices[call_id]["call"] = idx
        elif _is_function_call_output(item):
            if call_to_indices[call_id]["output"] is not None:
                logger.debug("Duplicate function_call_output call_id=%s (overwriting)", call_id)
            call_to_indices[call_id]["output"] = idx

    # Identify complete pairs
    complete_pairs: dict[str, tuple[int, int]] = {}
    for call_id, indices in call_to_indices.items():
        if indices["call"] is not None and indices["output"] is not None:
            complete_pairs[call_id] = (indices["call"], indices["output"])

    # Walk backwards, marking indices
    included_indices: set[int] = set()
    required_call_ids: set[str] = set()
    budget = window_size

    for idx in range(len(items) - 1, -1, -1):
        current_item = items[idx]

        if _is_function_call_output(current_item):
            call_id = _get_call_id(current_item)
            if call_id and call_id in complete_pairs:
                if budget >= 2:
                    included_indices.add(idx)
                    call_idx = complete_pairs[call_id][0]
                    included_indices.add(call_idx)
                    required_call_ids.add(call_id)
                    budget -= 2
                elif not included_indices:
                    logger.debug(
                        "Window size %d smaller than function call pair size 2, "
                        "keeping pair anyway",
                        window_size,
                    )
                    included_indices.add(idx)
                    call_idx = complete_pairs[call_id][0]
                    included_indices.add(call_idx)
                    required_call_ids.add(call_id)
                    budget = 0
                else:
                    logger.debug(
                        "Skipping function call pair call_id=%s: budget %d < 2",
                        call_id,
                        budget,
                    )
            else:
                logger.debug("Dropping orphaned function_call_output with call_id=%s", call_id)
            continue

        if _is_function_call(current_item):
            call_id = _get_call_id(current_item)
            if call_id and call_id in required_call_ids:
                continue
            logger.debug("Dropping orphaned function_call with call_id=%s", call_id)
            continue

        if _is_conversation_message(current_item):
            if budget > 0:
                included_indices.add(idx)
                budget -= 1
            continue

        if budget > 0:
            logger.warning(
                "Unknown item type encountered: %r. Including as single item.",
                current_item.get("type"),
            )
            included_indices.add(idx)
            budget -= 1

    sorted_indices = sorted(included_indices)
    return [items[i] for i in sorted_indices], sorted_indices


def boundary_aware_compact(
    items: list[TResponseInputItem],
    window_size: int,
) -> list[TResponseInputItem]:
    """Compact history while preserving function call pairs and original ordering.

    Uses an index-marking algorithm that preserves the original chronological
    ordering of items. Unlike a pair-bundling approach, items that appear between
    a function_call and its output retain their original position.

    Responses API Function Call Pairs:
    - A function call pair = function_call + its matching function_call_output (same call_id)
    - Multiple consecutive function_calls each need their matching output
    - Pairs are ATOMIC: keep both function_call and output, or drop both

    Algorithm:
    1. Build a map of call_id -> {call_index, output_index} to identify complete pairs
    2. Walk backwards through items, tracking a budget (remaining window capacity)
    3. For function_call_output with a complete pair: if budget >= 2, mark BOTH the
       output and its matching function_call at their original indices (budget -= 2)
    4. For function_call already marked: skip (already budgeted)
    5. For regular messages/unknown types: mark if budget > 0 (budget -= 1)
    6. Collect marked items in ascending index order (preserving original ordering)

    Edge cases:
    - window_size <= 0: returns []
    - window_size=1: most recent single message kept; function call pairs (needing 2 slots)
      are skipped unless they're the only thing available (soft limit to avoid empty result)
    - Function call pairs that don't fit the budget are dropped entirely
    - Orphaned function_call (no output) is dropped during backward walk
    - Orphaned function_call_output (no call) is dropped during backward walk

    Args:
        items: List of conversation items to compact (Responses API format).
        window_size: Maximum number of items to keep.

    Returns:
        Compacted list of items with function call pairs preserved, in original order.
    """
    compacted_items, _ = _boundary_aware_compact_with_indices(items, window_size)
    return drop_orphaned_tool_outputs(compacted_items)


def drop_orphaned_tool_outputs(
    items: list[TResponseInputItem],
) -> list[TResponseInputItem]:
    """Remove orphaned function_calls and function_call_outputs (Responses API format).

    Safety Validator - Defense in Depth:
    This ensures every function_call has a matching function_call_output (same call_id)
    and vice versa. Orphaned items are dropped.

    Args:
        items: List of conversation items to validate (Responses API format).

    Returns:
        Cleaned list with orphaned function call items removed.
    """
    # Build map of call_id -> presence of call and output
    call_ids_with_call: set[str] = set()
    call_ids_with_output: set[str] = set()

    for item in items:
        call_id = _get_call_id(item)
        if call_id is None:
            continue

        if _is_function_call(item):
            call_ids_with_call.add(call_id)
        elif _is_function_call_output(item):
            call_ids_with_output.add(call_id)

    # Complete pairs have both call and output
    complete_call_ids = call_ids_with_call & call_ids_with_output

    # Filter items
    cleaned: list[TResponseInputItem] = []

    for item in items:
        call_id = _get_call_id(item)

        if _is_function_call(item):
            if call_id in complete_call_ids:
                cleaned.append(item)
            else:
                logger.debug(
                    "Safety validator: dropping orphaned function_call with call_id=%s",
                    call_id,
                )
            continue

        if _is_function_call_output(item):
            if call_id in complete_call_ids:
                cleaned.append(item)
            else:
                logger.debug(
                    "Safety validator: dropping orphaned function_call_output with call_id=%s",
                    call_id,
                )
            continue

        # Normal message or unknown type — always keep.
        # NOTE: Unknown types bypass validation; if future types introduce relationships
        # (like function call pairs), they'll need explicit handling here.
        cleaned.append(item)

    return cleaned


class LocalCompactionSession(Session):
    """Wraps a Session, optionally applying a sliding window on retrieval.

    Write operations delegate directly; `get_items` can limit results to
    the last `window_size` items.

    Note:
        This wrapper deliberately does **not** cache compaction results.
        The underlying session may be backed by a shared store (e.g., Postgres)
        with multiple application servers writing concurrently. An in-process
        cache would go stale when another server modifies the session.

    Args:
        session: The underlying session to wrap.
        window_size: Max items to keep when retrieving. None disables compaction.
            **Soft limit**: may be exceeded by 1 item to preserve atomic function
            call pairs (a pair requires 2 slots; if window_size=1 and the most
            recent item is a function_call_output, both call and output are kept).
        token_counter: A callable ``(str) -> int`` used to estimate token counts
            for observability logging. Defaults to a ~4 chars/token estimate
            (model-agnostic). Pass ``None`` to disable token counting entirely.
            For accurate OpenAI counts, use ``TiktokenCounter()``.
            For other models (e.g. Anthropic Claude), supply a tokenizer via the provider SDK.

    Example:
        >>> from agents import SQLiteSession
        >>> underlying = SQLiteSession(session_id="my-session")
        >>> compacting_session = LocalCompactionSession(underlying, window_size=100)
        >>> # get_items() returns at most 100 most recent items
        >>> # For accurate OpenAI token counts:
        >>> from openai_agents_context_compaction import TiktokenCounter
        >>> session_oai = LocalCompactionSession(
        ...     underlying, window_size=100, token_counter=TiktokenCounter(),
        ... )
        >>> # For non-OpenAI models, define a token counter function:
        >>> def my_token_counter(text: str) -> int:
        ...     return client.count_tokens(text, model="claude-sonnet-4-20250514")
        >>> session = LocalCompactionSession(
        ...     underlying, window_size=100, token_counter=my_token_counter,
        ... )
        >>> # To disable token counting entirely:
        >>> session_no_tokens = LocalCompactionSession(
        ...     underlying, window_size=100, token_counter=None,
        ... )
    """

    def __init__(
        self,
        session: Session,
        window_size: int | None = None,
        token_counter: Callable[[str], int] | None = _default_token_counter,
    ) -> None:
        """Initialize the LocalCompactionSession.

        Args:
            session: The underlying session to delegate calls to.
            window_size: Maximum number of items to keep in the sliding window.
                         If None, no compaction is applied.
            token_counter: Callable that maps text to token count for logging.
                          Defaults to ~4 chars/token. Pass None to disable.
        """
        self._session = session
        self._window_size = window_size
        self._token_counter = token_counter

        if token_counter is None:
            logger.info("[session=%s] Token counting disabled", session.session_id)
        else:
            encoding = getattr(token_counter, "encoding_name", None)
            counter_name = type(token_counter).__name__
            if encoding:
                logger.info(
                    "[session=%s] Token counting: %s (%s)",
                    session.session_id,
                    counter_name,
                    encoding,
                )
            else:
                logger.info(
                    "[session=%s] Token counting: %s",
                    session.session_id,
                    counter_name,
                )

    @property
    def session_id(self) -> str:
        """Return the session ID from the underlying session."""
        return self._session.session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        """Set the session ID on the underlying session."""
        self._session.session_id = value

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        """Retrieve the conversation history from the underlying session.

        Applies boundary-aware compaction to maintain function call pair atomicity.
        When both window_size and limit are set, uses min(window_size, limit)
        to ensure function call pairs are never split.

        Args:
            limit: Maximum number of items to retrieve. If None, retrieves all items
                   (subject to window_size constraint).
                   Combined with window_size via effective_size = min(window, limit).
                   Note: even when window_size is None, passing a limit will trigger
                   boundary_aware_compact() to ensure function call pair atomicity.

        Returns:
            List of input items representing the conversation history.
            Function call pairs (function_call + function_call_output) are always atomic.
        """
        # Combine window_size and limit to maintain atomicity
        effective_size: int | None
        if self._window_size is not None and limit is not None:
            effective_size = min(self._window_size, limit)
        elif self._window_size is not None:
            effective_size = self._window_size
        elif limit is not None:
            effective_size = limit
        else:
            effective_size = None

        # Fetch from underlying — needed for compaction
        # NOTE: We fetch ALL items because boundary-aware compaction needs the full
        # history to correctly identify and preserve function call pairs.
        items = await self._session.get_items()

        if effective_size is not None:
            original_count = len(items)
            # Token counts are logged here to build observability baseline data.
            # FUTURE: these will drive token-budget compaction (see Token counting helpers above).
            # Compute tokens by position (index) — robust even if compaction copies items.
            item_tokens_by_index: list[int] | None = None
            original_tokens = 0
            if self._token_counter is not None:
                item_tokens_by_index = []
                for item in items:
                    text = _extract_text(item)
                    tokens = self._token_counter(text) + 4
                    item_tokens_by_index.append(tokens)
                original_tokens = sum(item_tokens_by_index)

            # Use internal function to get both items and their original indices
            items, kept_indices = _boundary_aware_compact_with_indices(items, effective_size)
            dropped = original_count - len(items)

            if self._token_counter is not None and item_tokens_by_index is not None:
                # Sum tokens for kept indices. Note: drop_orphaned_tool_outputs runs
                # after this and may remove additional items, so compacted_tokens can
                # be slightly inflated if orphans are present. Intentional approximation
                # — this is observability only, not a correctness concern.
                compacted_tokens = sum(item_tokens_by_index[i] for i in kept_indices)
                token_reduction_pct = (
                    (original_tokens - compacted_tokens) / original_tokens * 100
                    if original_tokens > 0
                    else 0.0
                )
                logger.info(
                    "[session=%s] Compacted %d → %d items"
                    " (dropped %d, window=%d) | tokens: %d → %d (%.1f%% reduction)",
                    self.session_id,
                    original_count,
                    len(items),
                    dropped,
                    effective_size,
                    original_tokens,
                    compacted_tokens,
                    token_reduction_pct,
                )
            else:
                logger.info(
                    "[session=%s] Compacted %d → %d items (dropped %d, window=%d)",
                    self.session_id,
                    original_count,
                    len(items),
                    dropped,
                    effective_size,
                )

        # NOTE: This call is intentional as a defense-in-depth safety validator.
        # It ensures that any edge cases missed by boundary_aware_compact are caught.
        # Runs even when no compaction happened (len(items) <= window_size) because
        # orphans could exist in the source data regardless of compaction.
        return drop_orphaned_tool_outputs(items)

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """Add new items to the underlying session's conversation history.

        Args:
            items: List of input items to add to the history.
        """
        await self._session.add_items(items)

    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return the most recent item from the underlying session.

        Returns:
            The most recent item if it exists, None if the session is empty.
        """
        return await self._session.pop_item()

    async def clear_session(self) -> None:
        """Clear all items from the underlying session."""
        await self._session.clear_session()

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return (
            f"<LocalCompactionSession("
            f"session_id={self.session_id!r}, "
            f"window_size={self._window_size})>"
        )
