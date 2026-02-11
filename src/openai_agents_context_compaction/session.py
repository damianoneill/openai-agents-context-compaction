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
    │  1. BACKWARD WALK (boundary_aware_compact)                                │
    │     Start from most recent, walk backwards                                │
    │     Group into "chunks": single messages OR function call pairs           │
    └───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌───────────────────────────────────────────────────────────────────────────┐
    │  2. CHUNKS (reverse chronological order)                                  │
    │     [user_msg], [func_call(orphan)], [asst_msg],                          │
    │     [func_call(b), output(b)], [func_call(a), output(a)], [user_msg]      │
    │           ↑ dropped                  ↑ complete pairs                     │
    └───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌───────────────────────────────────────────────────────────────────────────┐
    │  3. PACK INTO WINDOW (_pack_chunks_into_window)                           │
    │     Fill window with chunks using strict recency                          │
    │     window_size=4: [user_msg] + [func_call(b), output(b)] + [asst_msg]    │
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
    │  Result (chronological order)                                             │
    │  [asst_msg, func_call(b), output(b), user_msg]                            │
    └───────────────────────────────────────────────────────────────────────────┘

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


def _is_message(item: TResponseInputItem) -> bool:
    """Check if item is a regular message (user/assistant without function call)."""
    # User messages have role but no type, or type != function_call
    # Assistant messages have role=assistant and type=message
    if _is_function_call(item) or _is_function_call_output(item):
        return False
    return item.get("role") in ("user", "assistant")


def _pack_chunks_into_window(
    chunks: list[list[TResponseInputItem]], window_size: int
) -> list[TResponseInputItem]:
    """Pack chunks into window using strict recency, never splitting function call pairs.

    Strategy:
    - Fill window with chunks in recency order (most recent first)
    - Function call pairs that don't fit are dropped (recency wins over completeness)
    - If window is smaller than a function call pair and result is empty, keep the pair
      anyway to ensure we return meaningful data rather than nothing

    Note: window_size is a "soft" limit only for the edge case where the most recent
    item is a function call pair (2 items) and window_size=1. In this case, the pair
    is kept anyway rather than returning an empty result.

    Args:
        chunks: List of chunks in reverse chronological order (most recent first).
        window_size: Maximum number of items the window can hold.

    Returns:
        Flattened list of items in chronological order.
    """
    if window_size <= 0:
        return []

    result_chunks: list[list[TResponseInputItem]] = []
    total_items = 0

    for chunk in chunks:
        if total_items + len(chunk) <= window_size:
            # This chunk fits - add it
            result_chunks.append(chunk)
            total_items += len(chunk)
        elif total_items == 0 and len(chunk) > 1:
            # Window too small for this function call pair, but result is empty.
            # Keep the most recent pair anyway to return meaningful data.
            logger.debug(
                f"Window size {window_size} smaller than function call pair size {len(chunk)}, "
                "keeping pair anyway"
            )
            result_chunks.append(chunk)
            total_items += len(chunk)
        # If chunk doesn't fit and result is not empty, skip it (strict recency)

    # Reverse and flatten to chronological order
    return [item for chunk in reversed(result_chunks) for item in chunk]


def boundary_aware_compact(
    items: list[TResponseInputItem],
    window_size: int,
) -> list[TResponseInputItem]:
    """Compact history while preserving function call pairs (Responses API format).

    Responses API Function Call Pairs:
    - A function call pair = function_call + its matching function_call_output (same call_id)
    - Multiple consecutive function_calls each need their matching output
    - Pairs are ATOMIC: keep both function_call and output, or drop both

    Algorithm:
    1. Walk backwards through items, grouping them into "chunks"
    2. Each chunk is either: [single_message] OR [function_call, function_call_output]
    3. Pack chunks into window using strict recency (no prioritization)

    Edge cases:
    - window_size <= 0: returns []
    - window_size=1: most recent single message kept; function call pairs dropped unless
      they're the only item (then kept to avoid empty result)
    - Function call pairs that don't fit the window are dropped (recency wins)
    - Orphaned function_call (no output) is dropped
    - Orphaned function_call_output (no call) is dropped

    Args:
        items: List of conversation items to compact (Responses API format).
        window_size: Maximum number of items to keep.

    Returns:
        Compacted list of items with function call pairs preserved.
    """
    if window_size <= 0:
        return []

    if len(items) <= window_size:
        return drop_orphaned_tool_outputs(items)

    # =========================================================================
    # STEP 1: Build a map of call_id -> (function_call_index, function_call_output_index)
    # This allows us to identify complete pairs
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
    complete_pairs: set[str] = {
        call_id
        for call_id, indices in call_to_indices.items()
        if indices["call"] is not None and indices["output"] is not None
    }

    # =========================================================================
    # STEP 2: Walk backwards through items, grouping them into "chunks"
    # A chunk is either: [single_message] OR [function_call, function_call_output]
    # =========================================================================
    chunks: list[list[TResponseInputItem]] = []
    idx = len(items) - 1  # Start from the end (most recent)
    processed_call_ids: set[str] = set()

    while idx >= 0:
        current_item = items[idx]

        # ----- CASE A: function_call_output -----
        if _is_function_call_output(current_item):
            call_id = _get_call_id(current_item)

            if call_id and call_id in complete_pairs and call_id not in processed_call_ids:
                # Find the matching function_call
                call_idx = call_to_indices[call_id]["call"]
                if call_idx is not None:
                    function_call = items[call_idx]
                    # Bundle as atomic chunk: [function_call, function_call_output]
                    chunks.append([function_call, current_item])
                    processed_call_ids.add(call_id)
            else:
                # Orphaned output - skip it
                logger.debug(f"Dropping orphaned function_call_output with call_id={call_id}")

            idx -= 1
            continue

        # ----- CASE B: function_call -----
        if _is_function_call(current_item):
            call_id = _get_call_id(current_item)

            if call_id and call_id in processed_call_ids:
                # Already processed as part of a complete pair - skip
                idx -= 1
                continue

            # Orphaned function_call (no output) - skip it
            logger.debug(f"Dropping orphaned function_call with call_id={call_id}")
            idx -= 1
            continue

        # ----- CASE C: Normal message (user/assistant) -----
        if _is_message(current_item):
            chunks.append([current_item])
            idx -= 1
            continue

        # ----- CASE D: Unknown item type -----
        # Treat as single-item chunk to preserve future Responses API types
        # (e.g. reasoning, file_search_call) rather than silently dropping them.
        logger.warning(
            f"Unknown item type encountered: {current_item.get('type')!r}. "
            "Treating as single-item chunk."
        )
        chunks.append([current_item])
        idx -= 1

    # Pack chunks into window using strict recency
    return _pack_chunks_into_window(chunks, window_size)


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
                    f"Safety validator: dropping orphaned function_call with call_id={call_id}"
                )
            continue

        if _is_function_call_output(item):
            if call_id in complete_call_ids:
                cleaned.append(item)
            else:
                logger.debug(
                    f"Safety validator: dropping orphaned function_call_output "
                    f"with call_id={call_id}"
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
            # NOTE: Always log during alpha for visibility. Later, consider:
            # - Move to DEBUG level
            # - Only log when dropped > 0
            logger.info(
                f"[session={self.session_id}] Compacted {original_count} → {len(items)} items "
                f"(dropped {dropped}, window={effective_size})"
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
