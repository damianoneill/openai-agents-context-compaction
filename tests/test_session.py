"""Condensed tests for LocalCompactionSession with Responses API format support."""

from __future__ import annotations

import logging

import pytest
from agents import Session, TResponseInputItem

from openai_agents_context_compaction import LocalCompactionSession

# -------------------------------------------------------------------
# Mock Session
# -------------------------------------------------------------------


class MockSession(Session):
    """A minimal mock session for testing."""

    def __init__(self, session_id: str = "test-session") -> None:
        self.session_id = session_id
        self._items: list[TResponseInputItem] = []

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        items = self._items if limit is None else self._items[-limit:]
        return list(items)  # return a copy to avoid accidental mutation

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        self._items.extend(items)

    async def pop_item(self) -> TResponseInputItem | None:
        return self._items.pop() if self._items else None

    async def clear_session(self) -> None:
        self._items.clear()


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def user_msg(content: str) -> TResponseInputItem:
    return {"role": "user", "content": content}


def assistant_msg(content: str) -> TResponseInputItem:
    return {
        "role": "assistant",
        "type": "message",
        "content": [{"text": content, "type": "output_text"}],
    }


def function_call(
    call_id: str, name: str = "test_func", arguments: str = "{}"
) -> TResponseInputItem:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}


def function_call_output(call_id: str, output: str = "result") -> TResponseInputItem:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


# -------------------------------------------------------------------
# Base Test Class
# -------------------------------------------------------------------


class TestLocalCompactionSession:
    """Tests for sliding window, function call pair atomicity, and Responses API safety.

    Critical Invariants (enforced by _validate_invariants):
    - Every function_call MUST have a matching function_call_output (same call_id)
    - Every function_call_output MUST have a matching function_call
    - Violating these causes OpenAI Agents SDK to fail at runtime

    Why Function Call Pairs Must Be Atomic:
    - Responses API requires function_call + function_call_output pairs
    - Splitting them produces invalid conversation history
    - The SDK will reject orphaned function call items
    """

    def _validate_invariants(self, items: list[TResponseInputItem]) -> None:
        """Validate Responses API invariants that must NEVER be violated.

        Checks:
        - All function_calls have matching function_call_outputs (by call_id)
        - All function_call_outputs have matching function_calls
        - All role values are valid (user/assistant/system)

        Why this matters:
        - OpenAI Agents SDK rejects malformed conversation history
        - Orphaned function call items cause runtime errors
        """
        calls = {i["call_id"] for i in items if i.get("type") == "function_call"}
        outputs = {i["call_id"] for i in items if i.get("type") == "function_call_output"}
        assert calls <= outputs, f"Orphaned function_call(s): {calls - outputs}"
        assert outputs <= calls, f"Orphaned function_call_output(s): {outputs - calls}"
        for i in items:
            if i.get("role"):
                assert i["role"] in ("user", "assistant", "system")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("window_size", [None, 0, 1, 2, 5])
    async def test_sliding_window_and_safety(self, window_size: int | None) -> None:
        """Verify sliding window behavior preserves Responses API safety.

        Safety guarantees validated:
        1. Function call pair atomicity - function_call + output never split
        2. Orphaned function call items removed (function_call without output, vice versa)
        3. Valid message roles preserved

        Edge cases covered by parameterization:
        - window_size=None: No compaction, all items returned
        - window_size=0: Empty result (valid edge case)
        - window_size=1: Only single messages fit, function call pairs (size 2) kept anyway
        - window_size=2: Exactly one function call pair fits
        - window_size=5: Multiple items fit, tests recency ordering
        """
        mock = MockSession()
        await mock.add_items(
            [
                user_msg("start"),
                function_call("a", "tool_a"),
                function_call("b", "tool_b"),
                function_call_output("a", "out_a"),
                function_call_output("b", "out_b"),
                assistant_msg("mid"),
                function_call("orphan", "oops"),
                user_msg("end"),
            ]
        )

        compacting = LocalCompactionSession(mock, window_size=window_size)
        items = await compacting.get_items()
        self._validate_invariants(items)

        # Verify window size is respected (with soft limit exception for fc pairs)
        # window_size=1 may be exceeded by fc pairs (size 2) — that's by design
        if window_size is not None and window_size > 1:
            assert len(items) <= window_size, f"window_size={window_size}, got {len(items)}"

        # For larger windows, expect user and assistant messages
        if window_size is None or window_size >= 5:
            roles = [i.get("role") for i in items if i.get("role")]
            assert "user" in roles
            assert "assistant" in roles

    @pytest.mark.asyncio
    async def test_fc_pair_preserved_when_window_too_small(self) -> None:
        """Function call pairs kept even when window_size < pair size (safety net).

        Why this matters:
        - A function call pair is 2 items (function_call + output)
        - If window_size=1, we can't fit it, but returning empty is worse
        - Better to exceed window slightly than return broken/empty data

        This test ensures we never silently drop valid function call pairs just
        because the window is too small. The SDK needs complete pairs.
        """
        mock = MockSession()
        fc = function_call("call_1")
        fco = function_call_output("call_1")
        await mock.add_items([user_msg("hi"), fc, fco])
        compacting = LocalCompactionSession(mock, window_size=1)
        items = await compacting.get_items()
        assert items == [fc, fco]

    @pytest.mark.asyncio
    async def test_orphaned_items_dropped(self) -> None:
        """Orphaned function_calls and outputs are removed (defense in depth).

        Why orphans must be dropped:
        - function_call without output: SDK expects a response
        - function_call_output without call: SDK can't match it to anything
        - Both cases cause runtime errors in OpenAI Agents SDK

        This test verifies drop_orphaned_tool_outputs() catches these cases.
        The function acts as a safety validator after boundary_aware_compact().
        """
        mock = MockSession()
        await mock.add_items(
            [
                function_call("x"),  # no output
                function_call_output("y", "result"),  # no call
                user_msg("msg"),
            ]
        )
        compacting = LocalCompactionSession(mock, window_size=10)
        items = await compacting.get_items()
        self._validate_invariants(items)
        # Only valid user message remains
        assert len(items) == 1
        assert items[0]["content"] == "msg"

    @pytest.mark.asyncio
    async def test_limit_and_window_interaction(self) -> None:
        """Verify limit and window_size combine correctly (uses min).

        Behavior:
        - effective_size = min(window_size, limit) when both set
        - Most recent items kept (strict recency)
        - Function call pairs that don't fit are dropped

        Why this matters:
        - Callers may set limit independently of window_size
        - Recency is prioritized; older function call pairs are dropped if they don't fit
        """
        mock = MockSession()
        msg1 = user_msg("msg1")
        fc = function_call("call_1")
        fco = function_call_output("call_1")
        msg2 = user_msg("msg2")
        await mock.add_items([msg1, fc, fco, msg2])
        compacting = LocalCompactionSession(mock, window_size=2)
        items = await compacting.get_items(limit=2)
        # effective_size=min(2,2)=2, most recent is msg2 (size 1)
        # Then function call pair (size 2) doesn't fit, dropped
        # Then msg1 (size 1) fits, total = 2
        # Result: [msg1, msg2] in chronological order
        assert msg2 in items, "most recent message must be kept (strict recency)"
        assert msg1 in items, "older message fits in window"
        assert len(items) == 2, "window should hold exactly 2 items"
        # Function call pair doesn't fit in remaining space, so it's dropped
        assert fc not in items, "function call pair dropped (doesn't fit)"
        assert fco not in items, "function call pair dropped (doesn't fit)"
        self._validate_invariants(items)

    @pytest.mark.asyncio
    async def test_multiple_fc_pairs_strict_recency(self) -> None:
        """Stress test: multiple function call pairs with strict recency.

        Setup: 2 complete function call pairs (2 items each) + 2 user messages = 6 items
        Window size: 4 — strict recency fills from most recent.

        Expected: most recent items kept; function call pairs preserved atomically.
        The backward walk encounters user_msg("end") first, then pair "b",
        then user_msg("start"), then pair "a". With window=4, it keeps
        user_msg("end")(1) + pair_b(2) = 3, then user_msg("start")(1) = 4. Fits.
        Pair "a" (2 more) would exceed 4, so dropped.
        """
        mock = MockSession()
        fc_a = function_call("a", "tool_a")
        fco_a = function_call_output("a", "out_a")
        fc_b = function_call("b", "tool_b")
        fco_b = function_call_output("b", "out_b")
        await mock.add_items([user_msg("start"), fc_a, fco_a, fc_b, fco_b, user_msg("end")])
        compacting = LocalCompactionSession(mock, window_size=4)
        items = await compacting.get_items()
        self._validate_invariants(items)
        assert len(items) <= 4, f"should respect window_size=4, got {len(items)}"
        # Most recent function call pair (b) must be kept
        assert fc_b in items, "most recent function call pair must be preserved"
        assert fco_b in items, "most recent function call pair output must be preserved"
        # Older function call pair (a) should be dropped to fit window
        assert fc_a not in items, "older function call pair should be dropped"
        assert fco_a not in items, "older function call pair output should be dropped"

    @pytest.mark.asyncio
    async def test_only_newest_fc_pair_when_window_is_2(self) -> None:
        """Window=2 can hold exactly one function call pair; older pair must be dropped."""
        mock = MockSession()
        fc_old = function_call("old")
        fco_old = function_call_output("old")
        fc_new = function_call("new")
        fco_new = function_call_output("new")
        await mock.add_items([fc_old, fco_old, fc_new, fco_new])
        compacting = LocalCompactionSession(mock, window_size=2)
        items = await compacting.get_items()
        self._validate_invariants(items)
        assert items == [fc_new, fco_new], "only the most recent function call pair should survive"

    @pytest.mark.asyncio
    async def test_pass_through_behavior(self) -> None:
        """window_size=None disables window compaction, but limit still triggers
        boundary_aware_compact() to ensure function call pair atomicity.

        Behavior:
        - limit triggers boundary_aware_compact(items, limit)
        - Chronological order preserved
        - Function call pair atomicity maintained even with limit alone

        Use case: When caller wants full history but SDK wants last N items.
        """
        mock = MockSession()
        await mock.add_items([user_msg(f"msg{i}") for i in range(5)])
        compacting = LocalCompactionSession(mock, window_size=None)
        items = await compacting.get_items(limit=3)
        assert [i["content"] for i in items] == ["msg2", "msg3", "msg4"]

    @pytest.mark.asyncio
    async def test_delegation_methods(self) -> None:
        """Verify write operations delegate to underlying session unchanged.

        LocalCompactionSession is a read-side wrapper:
        - add_items, pop_item, clear_session pass through directly
        - session_id proxies to underlying session
        - Only get_items applies compaction logic

        This ensures the wrapper is transparent for writes.
        """
        mock = MockSession(session_id="abc")
        compacting = LocalCompactionSession(mock)
        item = user_msg("hi")
        await compacting.add_items([item])
        assert await compacting.pop_item() == item
        await compacting.clear_session()
        assert await mock.get_items() == []
        assert compacting.session_id == "abc"
        compacting.session_id = "xyz"
        assert compacting.session_id == "xyz"
        assert mock.session_id == "xyz"

    def test_repr(self) -> None:
        """Verify __repr__ includes session_id and window_size."""
        mock = MockSession(session_id="s1")
        compacting = LocalCompactionSession(mock, window_size=42)
        r = repr(compacting)
        assert "s1" in r
        assert "42" in r

    @pytest.mark.asyncio
    async def test_unknown_item_types_preserved_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unknown item types are preserved (with warning) for forward compatibility.

        The Responses API may add new item types (e.g., reasoning, file_search_call).
        Rather than silently dropping them, we:
        - Log a warning about the unknown type
        - Keep the item as a single-item chunk

        This ensures forward compatibility when OpenAI adds new response types.
        """
        mock = MockSession()
        unknown_item = {"type": "future_reasoning_item", "content": "thinking..."}
        valid_msg = user_msg("valid message")
        # Need more items than window_size to trigger compaction where warning is logged
        await mock.add_items([user_msg("older"), unknown_item, valid_msg])
        compacting = LocalCompactionSession(mock, window_size=2)

        with caplog.at_level(logging.WARNING, logger="openai_agents_context_compaction.session"):
            items = await compacting.get_items()

        # Unknown item and most recent valid message should be preserved
        assert len(items) == 2
        assert unknown_item in items
        assert valid_msg in items
        # Warning should be logged about unknown type
        assert any(
            "unknown item type encountered" in record.message.lower() for record in caplog.records
        ), "Expected warning about unknown item type"
