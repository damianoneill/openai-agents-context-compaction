"""Local compaction session wrapper.

Provides a Session wrapper that delegates calls to an underlying session
and optionally applies compaction by item count (window_size), token budget
(token_budget), or both.

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
    │  1. INDEX MAPPING (_boundary_aware_compact)                               │
    │     Build call_id → {call_index, output_index} map                        │
    │     Identify complete pairs (both call and output present)                │
    └───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌───────────────────────────────────────────────────────────────────────────┐
    │  2. BACKWARD WALK WITH DUAL BUDGET (_boundary_aware_compact)              │
    │     Walk backwards from most recent item                                  │
    │     Track item budget (window_size) and token budget (token_budget)       │
    │     Either budget exhausted → stop; the tighter constraint wins           │
    │     - function_call_output (complete pair): include if item budget >= 2   │
    │       AND token budget >= pair token cost; item budget -= 2,              │
    │       token budget -= pair token cost; mark both indices                  │
    │     - function_call (already marked): skip (already budgeted)             │
    │     - function_call (orphan/pair not selected): skip                      │
    │     - regular message / unknown type: include if item budget >= 1         │
    │       AND token budget >= item token cost; decrement both budgets         │
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

Soft Limit:
- If the most recent item exceeds both budgets and nothing has been included yet,
  it is kept anyway to prevent an empty or invalid result.
- For function call pairs: if the pair cost exceeds the budget but nothing has been
  included yet, both items are kept regardless (pair atomicity is preserved).
- After soft-limit inclusion both budgets are set to zero; no further items are added.

Token Budget:
- token_counter is called per item to estimate token cost (no built-in caching).
- Each item's cost = token_counter(text) + 4 (the +4 covers per-item structural
  overhead: role, separators, etc.).
- If token_counter returns 0 or negative it is clamped to 1 with a WARNING.
- Pass token_counter=None to disable token counting and token_budget enforcement.

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


# Singleton used as the default parameter for token_counter. Safe because
# DefaultTokenCounter is stateless — all LocalCompactionSession instances share
# this object without interference. If you subclass DefaultTokenCounter and need
# per-instance state, pass an explicit instance rather than relying on the default.
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

    if not parts:
        logger.debug(
            "No text extractable from item (type=%r, role=%r, keys=%r); "
            "token count will be clamped to 1",
            item.get("type"),
            item.get("role"),
            list(item.keys()),
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
    window_size: int | None,
    item_tokens_by_index: list[int] | None = None,
    token_budget: int | None = None,
) -> tuple[list[TResponseInputItem], list[int], bool]:
    """Internal: compact history and return both items and their original indices.

    Args:
        items: Full conversation history.
        window_size: Max items to keep (None = no item-count constraint).
        item_tokens_by_index: Pre-computed token cost per item (same indexing as items).
            Each value must include the +4 per-item structural overhead (role, separators)
            so costs are consistent with the budget accounting in the backward walk.
            Required when token_budget is set; ignored otherwise.
        token_budget: Max tokens to keep (None = no token constraint).

    Returns:
        Tuple of (compacted_items, kept_indices, soft_limit_fired) where
        kept_indices are the original positions of the kept items and
        soft_limit_fired is True when an item was kept despite exceeding the budget
        (to avoid returning an empty result).

    Raises:
        ValueError: If token_budget is set but item_tokens_by_index is None.
            Note: get_items() never triggers this — constructor validation ensures
            token_budget requires token_counter, and item_tokens_by_index is always
            computed when token_counter is present. This guard protects direct callers
            of this private function.
    """
    if token_budget is not None and item_tokens_by_index is None:
        raise ValueError(
            "item_tokens_by_index is required when token_budget is set; "
            "token budget would be silently ignored without it"
        )

    # window_size=0 or negative → empty result
    if window_size is not None and window_size <= 0:
        return [], [], False

    # No constraints → return everything
    if window_size is None and token_budget is None:
        return list(items), list(range(len(items))), False

    # Item-count only: short-circuit if all items already fit
    if token_budget is None and window_size is not None and len(items) <= window_size:
        return list(items), list(range(len(items))), False

    # Pre-compute total token cost once for use in the short-circuit checks below.
    # CONTRACT: item_tokens_by_index values must include the +4 per-item structural
    # overhead (role, separators, etc.), matching the cost model used in the backward
    # walk below. get_items() adds this overhead when building the list. A caller
    # passing raw token counts (without overhead) will get a subtly wrong short-circuit.
    total_tokens = sum(item_tokens_by_index) if item_tokens_by_index is not None else None

    # Both constraints: short-circuit if all items fit within both
    if (
        window_size is not None
        and token_budget is not None
        and total_tokens is not None
        and len(items) <= window_size
        and total_tokens <= token_budget
    ):
        return list(items), list(range(len(items))), False

    # Token-budget only: short-circuit if total tokens already fit.
    if (
        window_size is None
        and token_budget is not None
        and total_tokens is not None
        and total_tokens <= token_budget
    ):
        return list(items), list(range(len(items))), False

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

    # Walk backwards, marking indices to include
    # Two independent budgets: item count and token count (either may be None = unlimited)
    included_indices: set[int] = set()
    required_call_ids: set[str] = set()
    item_budget: int | None = window_size
    token_remaining: int | None = token_budget
    soft_limit_fired: bool = False

    for idx in range(len(items) - 1, -1, -1):
        # Early termination: either budget exhausted — nothing more can be included
        # (intersection semantics: both constraints must be satisfied to include any item)
        if (item_budget is not None and item_budget == 0) or (
            token_remaining is not None and token_remaining == 0
        ):
            break

        current_item = items[idx]

        if _is_function_call_output(current_item):
            call_id = _get_call_id(current_item)
            if call_id and call_id in complete_pairs:
                call_idx = complete_pairs[call_id][0]
                pair_token_cost: int | None = None
                if item_tokens_by_index is not None:
                    pair_token_cost = item_tokens_by_index[idx] + item_tokens_by_index[call_idx]

                fits_items = item_budget is None or item_budget >= 2
                fits_tokens = (
                    token_remaining is None
                    or pair_token_cost is None
                    or token_remaining >= pair_token_cost
                )

                if fits_items and fits_tokens:
                    included_indices.add(idx)
                    included_indices.add(call_idx)
                    required_call_ids.add(call_id)
                    if item_budget is not None:
                        item_budget -= 2
                    if token_remaining is not None and pair_token_cost is not None:
                        token_remaining -= pair_token_cost
                elif not included_indices:
                    # Soft limit: nothing included yet — keep this pair regardless of budget
                    # to avoid returning empty/stale data.
                    logger.debug(
                        "Budget too small for function call pair call_id=%s "
                        "(item_budget=%s, token_remaining=%s, pair_cost=%s), "
                        "keeping pair anyway (soft limit)",
                        call_id,
                        item_budget,
                        token_remaining,
                        pair_token_cost,
                    )
                    included_indices.add(idx)
                    included_indices.add(call_idx)
                    required_call_ids.add(call_id)
                    soft_limit_fired = True
                    if item_budget is not None:
                        item_budget = 0
                    if token_remaining is not None:
                        token_remaining = 0
                else:
                    logger.debug(
                        "Skipping function call pair call_id=%s: "
                        "item_budget=%s, token_remaining=%s, pair_cost=%s",
                        call_id,
                        item_budget,
                        token_remaining,
                        pair_token_cost,
                    )
            else:
                if not included_indices:
                    # Most recent item is an orphan — compaction may return [].
                    # This indicates bad session data; warn so operators can investigate.
                    logger.warning(
                        "Most recent item is an orphaned function_call_output "
                        "(call_id=%s, no matching function_call). "
                        "Compaction may return an empty result.",
                        call_id,
                    )
                else:
                    logger.debug("Dropping orphaned function_call_output with call_id=%s", call_id)
            continue

        if _is_function_call(current_item):
            call_id = _get_call_id(current_item)
            if call_id and call_id in required_call_ids:
                continue
            if call_id and call_id in complete_pairs:
                # Part of a complete pair but the output hasn't been processed yet
                # (fc appears after fco in the list — unusual but legal). The pair will
                # be included when the fco is encountered; skip here without "orphaned" log.
                logger.debug(
                    "Skipping function_call call_id=%s (will be budgeted with its output)",
                    call_id,
                )
            else:
                # Truly orphaned — no matching output in the session.
                # drop_orphaned_tool_outputs will also catch it as a defense-in-depth pass.
                logger.debug("Dropping orphaned function_call with call_id=%s", call_id)
            continue

        # Regular message or unknown type — single-slot item.
        # NOTE: kept as a separate branch from the pair case above intentionally.
        # Pairs require extra bookkeeping (call_idx, required_call_ids) that doesn't
        # apply here; merging the two branches would add abstraction without simplifying.
        item_token_cost: int | None = (
            item_tokens_by_index[idx] if item_tokens_by_index is not None else None
        )
        fits_items = item_budget is None or item_budget >= 1
        fits_tokens = (
            token_remaining is None or item_token_cost is None or token_remaining >= item_token_cost
        )

        is_unknown = not _is_conversation_message(current_item)

        if fits_items and fits_tokens:
            if is_unknown:
                logger.warning(
                    "Unknown item type encountered: %r. Including as single item.",
                    current_item.get("type"),
                )
            included_indices.add(idx)
            if item_budget is not None:
                item_budget -= 1
            if token_remaining is not None and item_token_cost is not None:
                token_remaining -= item_token_cost
        elif not included_indices:
            # Soft limit: nothing included yet — keep this item regardless of budget
            # (e.g. a single oversized message should not produce an empty result).
            logger.debug(
                "Budget too small for single item at idx=%d "
                "(item_budget=%s, token_remaining=%s, item_cost=%s), "
                "keeping anyway (soft limit)",
                idx,
                item_budget,
                token_remaining,
                item_token_cost,
            )
            if is_unknown:
                logger.warning(
                    "Unknown item type encountered: %r. Including as single item.",
                    current_item.get("type"),
                )
            included_indices.add(idx)
            soft_limit_fired = True
            if item_budget is not None:
                item_budget = 0
            if token_remaining is not None:
                token_remaining = 0

    sorted_indices = sorted(included_indices)
    return [items[i] for i in sorted_indices], sorted_indices, soft_limit_fired


def _determine_limiting_factor(
    items_before: int,
    items_after: int,
    item_budget: int | None,
    tokens_before: int,
    token_budget: int | None,
    soft_limit_fired: bool = False,
) -> str:
    """Return the constraint that caused compaction.

    Returns one of: ``'window_size'``, ``'token_budget'``, ``'both'``, ``'soft_limit'``.

    When both constraints are technically exceeded, ``items_after`` is used to
    determine which was actually tighter: if fewer items were kept than the
    item_budget allows, the token budget stopped the walk before the window
    was filled.

    ``soft_limit_fired`` must be passed from the compaction result; inferring it
    from items_before/items_after alone is unreliable when constraints are also active.
    """
    if soft_limit_fired:
        return "soft_limit"
    item_constrained = item_budget is not None and items_before > item_budget
    token_constrained = token_budget is not None and tokens_before > token_budget
    if item_constrained and token_constrained:
        # Determine which constraint was actually the binding one.
        # If items_after < item_budget, the token budget stopped the walk before
        # the window was filled — tokens were tighter.
        # If items_after == item_budget, the window was filled first (or both hit
        # simultaneously on the same item).
        if item_budget is not None and items_after < item_budget:
            return "token_budget"
        elif item_budget is not None and items_after == item_budget:
            return "window_size"
        else:
            return "both"  # defensive — should not occur in practice
    if item_constrained:
        return "window_size"
    if token_constrained:
        return "token_budget"
    return "soft_limit"


def _boundary_aware_compact(
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
    compacted_items, _, _ = _boundary_aware_compact_with_indices(items, window_size)
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
        The underlying session may be backed by a shared database
        with multiple application servers writing concurrently. An in-process
        cache would go stale when another server modifies the session.
        For single-process deployments calling ``get_items`` frequently on large
        histories, a TTL or invalidation-hook cache could reduce the O(n)
        compaction overhead; this is left as a future extension.

    Args:
        session: The underlying session to wrap.
        window_size: Max items to keep when retrieving. None disables item-count compaction.
            **Soft limit**: may be exceeded by 1 item to preserve atomic function
            call pairs (a pair requires 2 slots; if window_size=1 and the most
            recent item is a function_call_output, both call and output are kept).
        token_counter: A callable ``(str) -> int`` used to estimate token counts.
            Defaults to a ~4 chars/token estimate (model-agnostic).
            Pass ``None`` to disable token counting (also disables ``token_budget``).
            For accurate OpenAI counts, use ``TiktokenCounter()``.
            For other models (e.g. Anthropic Claude), supply a tokenizer via the provider SDK.
        token_budget: Max tokens to keep when retrieving. None disables token-budget
            compaction. When set alongside ``window_size``, both constraints apply and
            compaction stops when **either** is exhausted. Requires ``token_counter``
            to be set (raises ``ValueError`` if ``token_counter=None``).

    Example:
        >>> from agents import SQLiteSession
        >>> underlying = SQLiteSession(session_id="my-session")
        >>> # Item-count compaction only:
        >>> session = LocalCompactionSession(underlying, window_size=50)
        >>> # Token-budget compaction only (accurate OpenAI counts):
        >>> from openai_agents_context_compaction import TiktokenCounter
        >>> session = LocalCompactionSession(
        ...     underlying, token_budget=8000, token_counter=TiktokenCounter(),
        ... )
        >>> # Both constraints — compaction stops when either is exhausted:
        >>> session = LocalCompactionSession(
        ...     underlying, window_size=50, token_budget=8000, token_counter=TiktokenCounter(),
        ... )
        >>> # Custom tokenizer (e.g. Anthropic):
        >>> def my_counter(text: str) -> int:
        ...     return client.count_tokens(text, model="claude-sonnet-4-20250514")
        >>> session = LocalCompactionSession(
        ...     underlying, token_budget=8000, token_counter=my_counter
        ... )
        >>> # Disable token counting entirely:
        >>> session = LocalCompactionSession(underlying, window_size=50, token_counter=None)
    """

    def __init__(
        self,
        session: Session,
        window_size: int | None = None,
        token_counter: Callable[[str], int] | None = _default_token_counter,
        token_budget: int | None = None,
    ) -> None:
        """Initialize the LocalCompactionSession.

        Args:
            session: The underlying session to delegate calls to.
            window_size: Maximum number of items to keep. None disables item-count compaction.
            token_counter: Callable that maps text to token count. Defaults to ~4 chars/token.
                          Pass None to disable (also disables token_budget enforcement).
            token_budget: Maximum tokens to keep. None disables token-budget compaction.
                         Requires token_counter to be set.

        Raises:
            ValueError: If token_budget is set but token_counter is None.
            ValueError: If token_budget is <= 0.
        """
        if token_budget is not None and token_budget <= 0:
            raise ValueError(f"token_budget must be a positive integer, got {token_budget!r}")
        if token_budget is not None and token_counter is None:
            raise ValueError(
                "token_budget requires a token_counter; "
                "pass token_counter=... or remove token_budget"
            )

        self._session = session
        self._window_size = window_size
        self._token_counter = token_counter
        self._token_budget = token_budget

        if token_counter is None:
            logger.debug("[session=%s] Token counting disabled", session.session_id)
        else:
            encoding = getattr(token_counter, "encoding_name", None)
            counter_name = type(token_counter).__name__
            budget_info = f" | token_budget={token_budget}" if token_budget is not None else ""
            if encoding:
                logger.debug(
                    "[session=%s] Token counting: %s (%s)%s",
                    session.session_id,
                    counter_name,
                    encoding,
                    budget_info,
                )
            else:
                logger.debug(
                    "[session=%s] Token counting: %s%s",
                    session.session_id,
                    counter_name,
                    budget_info,
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
                   _boundary_aware_compact() to ensure function call pair atomicity.

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

        should_compact = effective_size is not None or self._token_budget is not None
        if should_compact:
            original_count = len(items)
            # Compute per-item token costs when either token logging or token_budget needs them.
            # Keyed by original index — robust even if compaction copies/reorders items.
            item_tokens_by_index: list[int] | None = None
            original_tokens = 0
            if self._token_counter is not None:
                item_tokens_by_index = []
                for item in items:
                    text = _extract_text(item)
                    raw = self._token_counter(text)
                    if raw <= 0:
                        # Counter returned an invalid value. Note: unrecognisable item
                        # shapes (empty text from _extract_text) are logged at DEBUG in
                        # _extract_text; this WARNING covers a misbehaving counter, not
                        # an unknown item shape (DefaultTokenCounter clamps internally).
                        logger.warning(
                            "token_counter returned %d for text %r; clamping to 1",
                            raw,
                            text[:50],
                        )
                        raw = 1
                    item_tokens_by_index.append(raw + 4)
                original_tokens = sum(item_tokens_by_index)

            # Use internal function to get both items and their original indices
            items, kept_indices, soft_limit_fired = _boundary_aware_compact_with_indices(
                items,
                window_size=effective_size,
                item_tokens_by_index=item_tokens_by_index,
                token_budget=self._token_budget,
            )
            dropped = original_count - len(items)

            limiting_factor = _determine_limiting_factor(
                items_before=original_count,
                items_after=len(items),
                item_budget=effective_size,
                tokens_before=original_tokens,
                token_budget=self._token_budget,
                soft_limit_fired=soft_limit_fired,
            )

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
                if dropped > 0:
                    logger.info(
                        "[session=%s] Compacted %d → %d items"
                        " (dropped %d, limiting_factor=%s, window=%s, token_budget=%s)"
                        " | tokens: %d → %d (%.1f%% reduction)",
                        self.session_id,
                        original_count,
                        len(items),
                        dropped,
                        limiting_factor,
                        effective_size,
                        self._token_budget,
                        original_tokens,
                        compacted_tokens,
                        token_reduction_pct,
                    )
                else:
                    logger.debug(
                        "[session=%s] Compaction check: %d items, nothing dropped"
                        " (window=%s, token_budget=%s) | tokens: %d",
                        self.session_id,
                        original_count,
                        effective_size,
                        self._token_budget,
                        original_tokens,
                    )
            else:
                if dropped > 0:
                    logger.info(
                        "[session=%s] Compacted %d → %d items"
                        " (dropped %d, limiting_factor=%s, window=%s, token_budget=%s)",
                        self.session_id,
                        original_count,
                        len(items),
                        dropped,
                        limiting_factor,
                        effective_size,
                        self._token_budget,
                    )
                else:
                    logger.debug(
                        "[session=%s] Compaction check: %d items, nothing dropped"
                        " (window=%s, token_budget=%s)",
                        self.session_id,
                        original_count,
                        effective_size,
                        self._token_budget,
                    )

        # NOTE: This call is intentional as a defense-in-depth safety validator.
        # It ensures that any edge cases missed by _boundary_aware_compact are caught.
        # Runs even when no compaction happened (len(items) <= window_size) because
        # orphans could exist in the source data regardless of compaction.
        # drop_orphaned_tool_outputs emits DEBUG logs for anything it removes,
        # so drops are visible in logs regardless of whether compaction was active.
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
            f"window_size={self._window_size}, "
            f"token_budget={self._token_budget})>"
        )

    def get_compaction_config(self) -> dict[str, object]:
        """Get current compaction configuration.

        Returns:
            Dict with keys: session_id, window_size, token_budget, token_counter_type.
        """
        return {
            "session_id": self.session_id,
            "window_size": self._window_size,
            "token_budget": self._token_budget,
            "token_counter_type": type(self._token_counter).__name__
            if self._token_counter
            else None,
        }
