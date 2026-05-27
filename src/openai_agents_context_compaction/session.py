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

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Protocol, TypedDict

from agents import Session, TResponseInputItem

# NOTE: Verbose DEBUG logging is intentional during alpha; callers control via logging config.
logger = logging.getLogger(__name__)


class CompactionResult(TypedDict):
    """Structured result returned by compaction policies."""

    items: list[TResponseInputItem]
    original_count: int
    returned_count: int
    limiting_factor: str | None
    summary_generated: bool
    metadata: dict[str, object]


class CompactionPolicy(Protocol):
    """Async interface for read-time history compaction policies."""

    async def compact(
        self,
        items: list[TResponseInputItem],
        *,
        session_id: str,
        limit: int | None = None,
    ) -> CompactionResult:
        """Return a compacted history view for the provided items."""

    def get_config(self) -> dict[str, object]:
        """Return a serializable description of the policy configuration."""


class HistorySummarizer(Protocol):
    """Provider-agnostic interface for compressing older history into one item."""

    async def summarize(
        self,
        items: list[TResponseInputItem],
        *,
        session_id: str,
        target_tokens: int | None = None,
        existing_summary: str | None = None,
    ) -> TResponseInputItem:
        """Return one synthetic history item summarising the provided items.

        Args:
            items: The prefix items to compress into a summary.
            session_id: The session identifier, passed for logging or tracing.
            target_tokens: Approximate token budget for the returned summary item.
                The summariser should aim to stay within this budget. Callers treat
                it as a hint; exceeding it produces a warning, not an error.
            existing_summary: Text of a prior summary produced for this session, if
                one exists. Implementations may use it to produce an incremental update
                rather than summarising from scratch. Not currently passed by
                ``SummarizingPolicy``; reserved for a future incremental-summarisation
                extension.
        """


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

    Note: values <= 0 from ``token_counter`` are clamped to 1, matching the
    behaviour of ``_build_item_token_costs``.

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
        raw = token_counter(text)
        if raw <= 0:
            raw = 1
        total += raw + 4
    return total


def _build_item_token_costs(
    items: list[TResponseInputItem],
    token_counter: Callable[[str], int] | None,
) -> tuple[list[int] | None, int]:
    """Build per-item token costs, including structural overhead."""
    if token_counter is None:
        return None, 0

    item_tokens_by_index: list[int] = []
    for item in items:
        text = _extract_text(item)
        raw = token_counter(text)
        if raw <= 0:
            logger.warning(
                "token_counter returned %d for text %r; clamping to 1",
                raw,
                text[:50],
            )
            raw = 1
        item_tokens_by_index.append(raw + 4)

    return item_tokens_by_index, sum(item_tokens_by_index)


def _log_sliding_window_policy_configuration(
    session_id: str,
    token_counter: Callable[[str], int] | None,
    token_budget: int | None,
) -> None:
    """Emit the legacy constructor log line for the default sliding-window policy."""
    if token_counter is None:
        logger.debug("[session=%s] Token counting disabled", session_id)
        return

    encoding = getattr(token_counter, "encoding_name", None)
    counter_name = type(token_counter).__name__
    budget_info = f" | token_budget={token_budget}" if token_budget is not None else ""
    if encoding:
        logger.debug(
            "[session=%s] Token counting: %s (%s)%s",
            session_id,
            counter_name,
            encoding,
            budget_info,
        )
        return

    logger.debug(
        "[session=%s] Token counting: %s%s",
        session_id,
        counter_name,
        budget_info,
    )


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


def _is_user_message(item: TResponseInputItem) -> bool:
    """Check if item is a user message (start of a turn)."""
    return item.get("role") == "user"


def _find_turn_start_indices(items: list[TResponseInputItem]) -> list[int]:
    """Find indices where each turn begins (i.e., user message positions).

    A turn is defined as a user message plus all subsequent items
    (assistant messages, tool calls, tool outputs) until the next user message.

    Returns a list of indices in ascending order. Each index marks the start
    of a turn.
    """
    return [i for i, item in enumerate(items) if _is_user_message(item)]


def _turn_aware_tail_selection(
    items: list[TResponseInputItem],
    retain_recent_turns: int,
    item_tokens_by_index: list[int] | None = None,
    token_budget: int | None = None,
) -> tuple[int, int, str | None]:
    """Select tail items based on turn boundaries with floor+ceiling semantics.

    Args:
        items: Full conversation history.
        retain_recent_turns: MINIMUM turns to keep (floor - always guaranteed).
        item_tokens_by_index: Pre-computed token cost per item.
        token_budget: Maximum tokens for expansion beyond minimum (ceiling).

    Returns:
        Tuple of (split_index, kept_turns, limiting_factor) where:
        - split_index: The index where the tail starts (items[split_index:] are kept)
        - kept_turns: Number of complete turns in the tail
        - limiting_factor: "turn_floor" if minimum applied, "token_budget" if
          budget limited expansion, None if all history kept.

    Algorithm:
        1. Always include at least `retain_recent_turns` complete turns (FLOOR)
        2. Walk backward through older turns, adding each if cumulative
           tokens <= token_budget
        3. Stop when next turn would exceed budget
        4. Boundary always snaps to user message (never mid-turn)
    """
    if not items:
        return 0, 0, None

    turn_starts = _find_turn_start_indices(items)
    if not turn_starts:
        # No user messages = no turns; keep everything
        return 0, 0, None

    total_turns = len(turn_starts)

    # Start with minimum floor turns
    floor_turns = min(retain_recent_turns, total_turns)

    # If floor already covers everything, return all
    if floor_turns >= total_turns:
        return 0, total_turns, None

    # Calculate the split point for minimum floor
    # We want the last `floor_turns` turns, so start from turn_starts[-floor_turns]
    floor_split = turn_starts[-floor_turns] if floor_turns > 0 else len(items)

    # If no token budget, just use the floor
    if token_budget is None or item_tokens_by_index is None:
        return floor_split, floor_turns, "turn_floor"

    # Calculate tokens for floor turns
    floor_tokens = sum(item_tokens_by_index[floor_split:])

    # Try to expand beyond floor if budget allows
    kept_turns = floor_turns
    current_split = floor_split
    current_tokens = floor_tokens

    # Walk backward through earlier turns
    for extra_turns in range(1, total_turns - floor_turns + 1):
        candidate_turns = floor_turns + extra_turns
        candidate_split = turn_starts[-candidate_turns]

        # Calculate tokens for this turn (from candidate_split to current_split)
        turn_tokens = sum(item_tokens_by_index[candidate_split:current_split])

        if current_tokens + turn_tokens <= token_budget:
            # This turn fits, include it
            current_tokens += turn_tokens
            current_split = candidate_split
            kept_turns = candidate_turns
        else:
            # Budget exhausted, stop expansion
            return current_split, kept_turns, "token_budget"

    # All turns fit within budget
    return 0, total_turns, None


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
                    logger.debug(
                        "Dropping orphaned function_call_output with call_id=%s",
                        call_id,
                    )
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

    Returns one of: ``'window_size'``, ``'token_budget'``, ``'soft_limit'``.

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
        # item_budget is not None — guaranteed by item_constrained's definition
        if items_after < item_budget:  # type: ignore[operator]
            return "token_budget"
        return "window_size"
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


class SlidingWindowPolicy:
    """Boundary-aware sliding-window compaction policy."""

    def __init__(
        self,
        window_size: int | None = None,
        token_counter: Callable[[str], int] | None = _default_token_counter,
        token_budget: int | None = None,
    ) -> None:
        """Initialize the sliding-window policy.

        Args:
            window_size: Maximum number of items to keep. None disables item-count compaction.
            token_counter: Callable that maps text to token count. Defaults to ~4 chars/token.
                Pass None to disable token counting (also disables token_budget enforcement).
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

        self._window_size = window_size
        self._token_counter = token_counter
        self._token_budget = token_budget

    async def compact(
        self,
        items: list[TResponseInputItem],
        *,
        session_id: str,
        limit: int | None = None,
    ) -> CompactionResult:
        """Apply boundary-aware sliding-window compaction to history items."""
        effective_size: int | None
        if self._window_size is not None and limit is not None:
            effective_size = min(self._window_size, limit)
        elif self._window_size is not None:
            effective_size = self._window_size
        elif limit is not None:
            effective_size = limit
        else:
            effective_size = None

        original_count = len(items)
        original_tokens = 0
        returned_tokens: int | None = None
        limiting_factor: str | None = None
        soft_limit_fired = False
        kept_indices: list[int] = list(range(len(items)))
        compacted_items = list(items)

        should_compact = effective_size is not None or self._token_budget is not None
        if should_compact:
            item_tokens_by_index, original_tokens = _build_item_token_costs(
                compacted_items,
                self._token_counter,
            )

            compacted_items, kept_indices, soft_limit_fired = _boundary_aware_compact_with_indices(
                compacted_items,
                window_size=effective_size,
                item_tokens_by_index=item_tokens_by_index,
                token_budget=self._token_budget,
            )
            dropped = original_count - len(compacted_items)
            if dropped > 0 or soft_limit_fired:
                limiting_factor = _determine_limiting_factor(
                    items_before=original_count,
                    items_after=len(compacted_items),
                    item_budget=effective_size,
                    tokens_before=original_tokens,
                    token_budget=self._token_budget,
                    soft_limit_fired=soft_limit_fired,
                )

            if self._token_counter is not None and item_tokens_by_index is not None:
                compacted_tokens = sum(item_tokens_by_index[index] for index in kept_indices)
                token_reduction_pct = (
                    (original_tokens - compacted_tokens) / original_tokens * 100
                    if original_tokens > 0
                    else 0.0
                )
                returned_tokens = compacted_tokens
                if dropped > 0:
                    logger.info(
                        "[session=%s] Compacted %d → %d items"
                        " (dropped %d, limiting_factor=%s, window=%s, token_budget=%s)"
                        " | tokens: %d → %d (%.1f%% reduction)",
                        session_id,
                        original_count,
                        len(compacted_items),
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
                        session_id,
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
                        session_id,
                        original_count,
                        len(compacted_items),
                        dropped,
                        limiting_factor,
                        effective_size,
                        self._token_budget,
                    )
                else:
                    logger.debug(
                        "[session=%s] Compaction check: %d items, nothing dropped"
                        " (window=%s, token_budget=%s)",
                        session_id,
                        original_count,
                        effective_size,
                        self._token_budget,
                    )

        return {
            "items": compacted_items,
            "original_count": original_count,
            "returned_count": len(compacted_items),
            "limiting_factor": limiting_factor,
            "summary_generated": False,
            "metadata": {
                "policy_type": "sliding_window",
                "window_size": self._window_size,
                "effective_window_size": effective_size,
                "token_budget": self._token_budget,
                "token_counter_type": type(self._token_counter).__name__
                if self._token_counter
                else None,
                "original_tokens": original_tokens if should_compact else None,
                "returned_tokens": returned_tokens,
                "soft_limit_fired": soft_limit_fired,
            },
        }

    def get_config(self) -> dict[str, object]:
        """Return the current policy configuration."""
        return {
            "policy_type": "sliding_window",
            "window_size": self._window_size,
            "token_budget": self._token_budget,
            "token_counter_type": type(self._token_counter).__name__
            if self._token_counter
            else None,
        }


class SummarizingPolicy:
    """Boundary-aware summarising compaction policy.

    Keeps a recent raw tail of history and summarises the older prefix into one
    synthetic history item returned by the injected ``HistorySummarizer``.

    Falls back to ``SlidingWindowPolicy`` (token-budget only) when:
    - summarisation raises an exception or times out

    The fallback is always bounded (sliding-window), never unbounded raw history.

    Turn-Based Retention (floor + ceiling semantics):
        - A "turn" is a user message plus all subsequent items (assistant messages,
          tool calls, tool outputs) until the next user message.
        - ``retain_recent_turns`` is the MINIMUM FLOOR: this many turns are always
          guaranteed, regardless of token budget.
        - ``retain_recent_token_budget`` is the CEILING: used to expand beyond the
          minimum floor if budget allows, but does not reduce below the floor.

        Example with retain_recent_turns=2, token_budget=20000:
        - Turn 1 (5K) + Turn 2 (5K) → keep both (10K, minimum met)
        - Turn 1 (21K) + Turn 2 (5K) → keep both (26K, minimum guaranteed even if over)
        - Turn 1 (3K) + Turn 2 (3K) + Turn 3 (15K) → keep 2+3, try Turn 1: exceeds → keep 2

    Args:
        summarizer: Provider-agnostic callable that turns a prefix list into one
            synthetic history item.
        retain_recent_turns: Minimum turns to keep (floor). A turn is a user
            message plus all subsequent items until the next user message.
        retain_recent_token_budget: Max tokens for tail expansion (ceiling).
            Caps expansion beyond the minimum floor. Requires ``token_counter``.
        summary_target_tokens: Hint passed to the summariser for how long to make
            the summary. Logged as a warning if the returned item exceeds this.
        summary_timeout_seconds: Seconds before the summariser call is cancelled.
        summarizer_input_token_limit: Maximum tokens to send to the summariser.
            If the prefix exceeds this limit, the oldest items are dropped
            (keeping boundary-aware pairs intact) until the prefix fits. This
            prevents sending a prefix that exceeds the summariser model's context
            window. ``None`` disables truncation. Requires ``token_counter``.
        token_counter: Callable mapping text to token count. Defaults to ~4 chars/token.
            Pass ``None`` to disable token counting (also disables token-budget retention).
        fallback_to_sliding_window: When ``True`` (default), a summariser exception or
            timeout falls back to ``SlidingWindowPolicy``. When ``False``, the exception
            propagates to the caller instead.
    """

    def __init__(
        self,
        summarizer: HistorySummarizer,
        *,
        retain_recent_turns: int = 2,
        retain_recent_token_budget: int | None = 20000,
        summary_target_tokens: int = 1500,
        summary_timeout_seconds: float = 5.0,
        summarizer_input_token_limit: int | None = 60_000,
        token_counter: Callable[[str], int] | None = _default_token_counter,
        fallback_to_sliding_window: bool = True,
    ) -> None:
        if retain_recent_token_budget is not None and token_counter is None:
            raise ValueError(
                "retain_recent_token_budget requires a token_counter; "
                "pass token_counter=... or remove retain_recent_token_budget"
            )
        if summarizer_input_token_limit is not None and token_counter is None:
            raise ValueError(
                "summarizer_input_token_limit requires a token_counter; "
                "pass token_counter=... or remove summarizer_input_token_limit"
            )

        self._summarizer = summarizer
        self._retain_recent_turns = retain_recent_turns
        self._retain_recent_token_budget = retain_recent_token_budget
        self._summary_target_tokens = summary_target_tokens
        self._summary_timeout_seconds = summary_timeout_seconds
        self._summarizer_input_token_limit = summarizer_input_token_limit
        self._token_counter = token_counter
        self._fallback_to_sliding_window = fallback_to_sliding_window

    def _build_fallback_policy(self) -> SlidingWindowPolicy:
        return SlidingWindowPolicy(
            window_size=None,  # No item limit, rely on token budget
            token_counter=self._token_counter,
            token_budget=self._retain_recent_token_budget,
        )

    async def compact(
        self,
        items: list[TResponseInputItem],
        *,
        session_id: str,
        limit: int | None = None,
    ) -> CompactionResult:
        """Summarise the older prefix and return ``[summary_item] + recent_tail``."""
        original_count = len(items)

        if not items:
            return {
                "items": [],
                "original_count": 0,
                "returned_count": 0,
                "limiting_factor": None,
                "summary_generated": False,
                "metadata": {"policy_type": "summary_window"},
            }

        item_tokens_by_index, _ = _build_item_token_costs(items, self._token_counter)

        # Turn-aware tail selection with floor+ceiling semantics
        split_point, kept_turns, limiting_factor = _turn_aware_tail_selection(
            items,
            retain_recent_turns=self._retain_recent_turns,
            item_tokens_by_index=item_tokens_by_index,
            token_budget=self._retain_recent_token_budget,
        )
        tail_items = items[split_point:]

        # Nothing dropped — no summarisation needed.
        if split_point == 0:
            return {
                "items": list(items),
                "original_count": original_count,
                "returned_count": original_count,
                "limiting_factor": None,
                "summary_generated": False,
                "metadata": {
                    "policy_type": "summary_window",
                    "summary_skipped_reason": "nothing_to_summarize",
                    "turns_kept": kept_turns,
                },
            }

        items_to_summarize = items[:split_point]

        # Truncate prefix to summarizer_input_token_limit by dropping oldest items.
        # This prevents sending a prefix that exceeds the summariser model's
        # context window, which would waste latency and trigger fallback every turn.
        if self._summarizer_input_token_limit is not None and self._token_counter is not None:
            prefix_tokens_by_index, prefix_total_tokens = _build_item_token_costs(
                items_to_summarize, self._token_counter
            )
            if (
                prefix_total_tokens > self._summarizer_input_token_limit
                and prefix_tokens_by_index is not None
            ):
                original_prefix_count = len(items_to_summarize)
                # Reuse boundary-aware compaction: walk backward (keeping newest),
                # dropping oldest items first while preserving pair atomicity.
                items_to_summarize, _, _ = _boundary_aware_compact_with_indices(
                    items_to_summarize,
                    window_size=None,
                    item_tokens_by_index=prefix_tokens_by_index,
                    token_budget=self._summarizer_input_token_limit,
                )
                logger.info(
                    "[session=%s] Truncated summariser prefix %d → %d items "
                    "(token limit=%d, original tokens=%d)",
                    session_id,
                    original_prefix_count,
                    len(items_to_summarize),
                    self._summarizer_input_token_limit,
                    prefix_total_tokens,
                )

        # Attempt summarisation with timeout.
        start = time.monotonic()
        try:
            summary_item = await asyncio.wait_for(
                self._summarizer.summarize(
                    items_to_summarize,
                    session_id=session_id,
                    target_tokens=self._summary_target_tokens,
                ),
                timeout=self._summary_timeout_seconds,
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            if not self._fallback_to_sliding_window:
                logger.warning(
                    "[session=%s] Summarisation failed after %.1fms (%s: %s); "
                    "re-raising (fallback_to_sliding_window=False)",
                    session_id,
                    latency_ms,
                    type(exc).__name__,
                    exc,
                )
                raise
            logger.warning(
                "[session=%s] Summarisation failed after %.1fms (%s: %s); "
                "falling back to sliding window",
                session_id,
                latency_ms,
                type(exc).__name__,
                exc,
            )
            return await self._build_fallback_policy().compact(
                items, session_id=session_id, limit=limit
            )

        latency_ms = (time.monotonic() - start) * 1000

        # Warn if summary item is larger than the target.
        if self._token_counter is not None:
            summary_tokens = self._token_counter(_extract_text(summary_item)) + 4
            if summary_tokens > self._summary_target_tokens:
                logger.warning(
                    "[session=%s] Summary token count %d exceeds target %d; proceeding anyway",
                    session_id,
                    summary_tokens,
                    self._summary_target_tokens,
                )
        else:
            summary_tokens = 0

        result_items = [summary_item] + tail_items
        returned_count = len(result_items)

        logger.info(
            "[session=%s] Summarised %d → %d items "
            "(prefix=%d summarized, tail=%d kept raw, latency=%.1fms)",
            session_id,
            original_count,
            returned_count,
            len(items_to_summarize),
            len(tail_items),
            latency_ms,
        )

        metadata: dict[str, object] = {
            "policy_type": "summary_window",
            "items_summarized": len(items_to_summarize),
            "tail_items_kept": len(tail_items),
            "turns_kept": kept_turns,
            "summary_latency_ms": latency_ms,
            "summary_output_tokens": summary_tokens,
            "summary_target_tokens": self._summary_target_tokens,
        }
        if limiting_factor is not None:
            metadata["tail_limiting_factor"] = limiting_factor

        return {
            "items": result_items,
            "original_count": original_count,
            "returned_count": returned_count,
            "limiting_factor": limiting_factor,
            "summary_generated": True,
            "metadata": metadata,
        }

    def get_config(self) -> dict[str, object]:
        """Return the current policy configuration."""
        return {
            "policy_type": "summary_window",
            "retain_recent_turns": self._retain_recent_turns,
            "retain_recent_token_budget": self._retain_recent_token_budget,
            "summary_target_tokens": self._summary_target_tokens,
            "summary_timeout_seconds": self._summary_timeout_seconds,
            "summarizer_input_token_limit": self._summarizer_input_token_limit,
            "token_counter_type": type(self._token_counter).__name__
            if self._token_counter
            else None,
        }


class LocalCompactionSession(Session):
    """Wraps a Session, optionally applying a compaction policy on retrieval.

    Write operations delegate directly; `get_items` can limit results to
    the active policy's returned history.

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
        policy: Explicit compaction policy to use. When provided, the legacy
            ``window_size``, ``token_counter``, and ``token_budget`` arguments are ignored.

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
        >>> # Custom tokenizer (e.g. Anthropic) — adapt to your SDK version:
        >>> def my_counter(text: str) -> int:
        ...     response = client.beta.messages.count_tokens(
        ...         model="claude-haiku-4-5-20251001",  # any valid model works
        ...         messages=[{"role": "user", "content": text}],
        ...     )
        ...     return response.input_tokens
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
        policy: CompactionPolicy | None = None,
    ) -> None:
        """Initialize the LocalCompactionSession.

        Args:
            session: The underlying session to delegate calls to.
            window_size: Maximum number of items to keep. None disables item-count compaction.
            token_counter: Callable that maps text to token count. Defaults to ~4 chars/token.
                          Pass None to disable (also disables token_budget enforcement).
            token_budget: Maximum tokens to keep. None disables token-budget compaction.
                         Requires token_counter to be set unless ``policy`` is provided.
            policy: Explicit compaction policy. When provided, legacy constructor
                    arguments are ignored.
        """
        self._session = session
        self._policy: CompactionPolicy
        if policy is None:
            self._policy = SlidingWindowPolicy(
                window_size=window_size,
                token_counter=token_counter,
                token_budget=token_budget,
            )
            _log_sliding_window_policy_configuration(
                session.session_id,
                token_counter,
                token_budget,
            )
        else:
            self._policy = policy
            _legacy_args: list[str] = []
            if window_size is not None:
                _legacy_args.append(f"window_size={window_size!r}")
            if token_budget is not None:
                _legacy_args.append(f"token_budget={token_budget!r}")
            if token_counter is not _default_token_counter:
                _legacy_args.append("token_counter=<custom>")
            if _legacy_args:
                logger.warning(
                    "[session=%s] policy= was provided; ignoring legacy arguments: %s",
                    session.session_id,
                    ", ".join(_legacy_args),
                )
            policy_type = self._policy.get_config().get(
                "policy_type",
                type(self._policy).__name__,
            )
            logger.debug("[session=%s] Compaction policy: %s", session.session_id, policy_type)

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

        Fetches the full history from the underlying session, then forwards it to the
        active policy. The returned list is validated to ensure function call pair
        atomicity before being exposed to callers.

        Args:
            limit: Optional caller-supplied limit forwarded to the active policy.

        Returns:
            List of input items representing the conversation history.
            Function call pairs (function_call + function_call_output) are always atomic.
        """
        # Fetch from underlying — needed for compaction
        # NOTE: We fetch ALL items because boundary-aware compaction needs the full
        # history to correctly identify and preserve function call pairs.
        items = await self._session.get_items()
        result = await self._policy.compact(items, session_id=self.session_id, limit=limit)

        # NOTE: This call is intentional as a defense-in-depth safety validator.
        # It ensures that any edge cases missed by the active policy are caught.
        # Runs even when the active policy already validates history because orphans
        # could exist in source data or be introduced by a custom policy.
        # drop_orphaned_tool_outputs emits DEBUG logs for anything it removes,
        # so drops are visible in logs regardless of whether compaction was active.
        return drop_orphaned_tool_outputs(result["items"])

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
        policy_config = self._policy.get_config()
        policy_type = policy_config.get("policy_type", type(self._policy).__name__)
        parts = [f"session_id={self.session_id!r}", f"policy_type={policy_type!r}"]
        if "window_size" in policy_config:
            parts.append(f"window_size={policy_config['window_size']}")
        if "token_budget" in policy_config:
            parts.append(f"token_budget={policy_config['token_budget']}")
        return f"<LocalCompactionSession({', '.join(parts)})>"

    def get_compaction_config(self) -> dict[str, object]:
        """Get current compaction configuration.

        Returns:
            Dict with the session_id and the active policy's public configuration.
        """
        return {"session_id": self.session_id, **self._policy.get_config()}
