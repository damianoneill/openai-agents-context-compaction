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

from agents import Session, TResponseInputItem

# NOTE: Verbose DEBUG logging is intentional during alpha; callers control via logging config.
logger = logging.getLogger(__name__)


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
    - Orphaned function_call (no output) is dropped
    - Orphaned function_call_output (no call) is dropped

    Args:
        items: List of conversation items to compact (Responses API format).
        window_size: Maximum number of items to keep.

    Returns:
        Compacted list of items with function call pairs preserved, in original order.
    """
    if window_size <= 0:
        return []

    if len(items) <= window_size:
        return drop_orphaned_tool_outputs(items)

    # =========================================================================
    # STEP 1: Build a map of call_id -> (function_call_index, function_call_output_index)
    # This allows us to identify complete pairs
    # NOTE: call_id is assumed unique per type (SDK invariant). Duplicates overwrite.
    # =========================================================================
    call_to_indices: dict[str, dict[str, int | None]] = {}

    for idx, item in enumerate(items):
        call_id = _get_call_id(item)
        if call_id is None:
            continue

        if call_id not in call_to_indices:
            call_to_indices[call_id] = {"call": None, "output": None}

        if _is_function_call(item):
            call_to_indices[call_id]["call"] = idx
        elif _is_function_call_output(item):
            call_to_indices[call_id]["output"] = idx

    # Identify complete pairs (both call and output present)
    complete_pairs: dict[str, tuple[int, int]] = {}
    for call_id, indices in call_to_indices.items():
        if indices["call"] is not None and indices["output"] is not None:
            complete_pairs[call_id] = (indices["call"], indices["output"])

    # =========================================================================
    # STEP 2: Walk backwards, marking indices to include while tracking budget
    # Budget represents remaining window capacity. Including a complete pair
    # costs 2 from the budget (reserved upfront when the output is encountered).
    # =========================================================================
    included_indices: set[int] = set()
    required_call_ids: set[str] = set()  # call_ids whose function_call we must include
    budget = window_size

    for idx in range(len(items) - 1, -1, -1):
        current_item = items[idx]

        # ----- CASE A: function_call_output -----
        if _is_function_call_output(current_item):
            call_id = _get_call_id(current_item)

            if call_id and call_id in complete_pairs:
                if budget >= 2:
                    # Reserve budget for both output AND its matching call
                    included_indices.add(idx)
                    call_idx = complete_pairs[call_id][0]
                    included_indices.add(call_idx)
                    required_call_ids.add(call_id)
                    budget -= 2
                elif not included_indices:
                    # Soft limit: nothing included yet, so this pair is the most
                    # recent actionable content. Keep it to avoid returning
                    # empty/stale data even though it exceeds window_size.
                    logger.debug(
                        "Window size %d smaller than function call pair size 2, "
                        "keeping pair anyway",
                        window_size,
                    )
                    included_indices.add(idx)
                    call_idx = complete_pairs[call_id][0]
                    included_indices.add(call_idx)
                    required_call_ids.add(call_id)
                    budget = 0  # prevent further inclusions
                else:
                    # Not enough budget for the pair — skip entirely
                    logger.debug(
                        "Skipping function call pair call_id=%s: budget %d < 2",
                        call_id,
                        budget,
                    )
            else:
                # Orphaned output (no matching call) - skip
                logger.debug("Dropping orphaned function_call_output with call_id=%s", call_id)
            continue

        # ----- CASE B: function_call -----
        if _is_function_call(current_item):
            call_id = _get_call_id(current_item)

            if call_id and call_id in required_call_ids:
                # Already marked and budgeted when we processed its output — skip
                continue

            # Orphaned function_call (no output, or pair wasn't selected) - skip
            logger.debug("Dropping orphaned function_call with call_id=%s", call_id)
            continue

        # ----- CASE C: Normal message (user/assistant) -----
        if _is_conversation_message(current_item):
            if budget > 0:
                included_indices.add(idx)
                budget -= 1
            continue

        # ----- CASE D: Unknown item type -----
        # Treat as single item to preserve future Responses API types
        # (e.g. reasoning, file_search_call) rather than silently dropping them.
        if budget > 0:
            logger.warning(
                "Unknown item type encountered: %r. Including as single item.",
                current_item.get("type"),
            )
            included_indices.add(idx)
            budget -= 1

    # =========================================================================
    # STEP 3: Collect items in original chronological order
    # =========================================================================
    return [items[i] for i in sorted(included_indices)]


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

        # Normal message - always keep
        cleaned.append(item)

    return cleaned


class LocalCompactionSession(Session):
    """Wraps a Session, optionally applying a sliding window on retrieval.

    Write operations delegate directly; `get_items` can limit results to
    the last `window_size` items.

    Args:
        session: The underlying session to wrap.
        window_size: Max items to keep when retrieving. None disables compaction.

    Example:
        >>> from agents import SQLiteSession
        >>> underlying = SQLiteSession(session_id="my-session")
        >>> compacting_session = LocalCompactionSession(underlying, window_size=100)
        >>> # get_items() returns at most 100 most recent items
    """

    def __init__(self, session: Session, window_size: int | None = None) -> None:
        """Initialize the LocalCompactionSession.

        Args:
            session: The underlying session to delegate calls to.
            window_size: Maximum number of items to keep in the sliding window.
                         If None, no compaction is applied.
        """
        self._session = session
        self._window_size = window_size

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
        # NOTE: We always fetch ALL items from the underlying session, ignoring
        # its `limit` parameter, because boundary-aware compaction needs the full
        # history to correctly identify and preserve function call pairs. This means
        # memory usage scales with total session size, not window_size.
        items = await self._session.get_items()

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

        if effective_size is not None:
            original_count = len(items)
            items = boundary_aware_compact(items, effective_size)
            dropped = original_count - len(items)
            logger.debug(
                "[session=%s] Compacted %d → %d items (dropped %d, window=%d)",
                self.session_id,
                original_count,
                len(items),
                dropped,
                effective_size,
            )

        # NOTE: This call is intentional as a defense-in-depth safety validator.
        # It ensures that any edge cases missed by boundary_aware_compact are caught.
        items = drop_orphaned_tool_outputs(items)
        return items

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
