"""Condensed tests for LocalCompactionSession with Responses API format support."""

from __future__ import annotations

import logging

import pytest
from agents import Session, TResponseInputItem

from openai_agents_context_compaction import LocalCompactionSession
from openai_agents_context_compaction.session import (
    _count_tokens,
    _determine_limiting_factor,
    _extract_text,
)

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


def _validate_invariants(items: list[TResponseInputItem]) -> None:
    """Validate Responses API invariants.

    Every function_call must have a matching output and vice versa.
    """
    calls = {i["call_id"] for i in items if i.get("type") == "function_call"}
    outputs = {i["call_id"] for i in items if i.get("type") == "function_call_output"}
    assert calls <= outputs, f"Orphaned function_call(s): {calls - outputs}"
    assert outputs <= calls, f"Orphaned function_call_output(s): {outputs - calls}"


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

        Delegates pair-atomicity check to the module-level _validate_invariants,
        then adds a role-value check specific to the sliding-window test suite.
        """
        _validate_invariants(items)
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

        # window_size=0 must return empty list
        if window_size == 0:
            assert items == [], "window_size=0 should return empty list"

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
        The function acts as a safety validator after _boundary_aware_compact().
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
        # effective_size=min(2,2)=2. Backward walk:
        #   msg2 (size 1): fits, budget 2→1
        #   fco  (pair):   needs 2 slots, only 1 remaining → dropped
        #   fc   (pair):   skipped (not in required_call_ids)
        #   msg1 (size 1): fits, budget 1→0
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
    async def test_limit_alone_triggers_boundary_aware_compact(self) -> None:
        """window_size=None disables window compaction, but limit still triggers
        _boundary_aware_compact() to ensure function call pair atomicity.

        Behavior:
        - limit triggers _boundary_aware_compact(items, limit)
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
    async def test_unknown_item_types_preserved(self) -> None:
        """Unknown item types are kept as independent atomic items (forward compatibility).

        The Responses API may add new item types (e.g., reasoning, file_search_call).
        Rather than silently dropping them, we keep the item and treat it as a
        single-slot item. This ensures the library doesn't break when OpenAI adds
        new response types.

        Note: logging behaviour (WARNING for unknown types) is tested separately
        in test_unknown_item_types_warning_log so that a wording change in the log
        message doesn't mask a regression in the behavioural assertion here.
        """
        mock = MockSession()
        unknown_item = {"type": "future_reasoning_item", "content": "thinking..."}
        valid_msg = user_msg("valid message")
        await mock.add_items([user_msg("older"), unknown_item, valid_msg])
        compacting = LocalCompactionSession(mock, window_size=2)

        items = await compacting.get_items()

        assert len(items) == 2
        assert unknown_item in items
        assert valid_msg in items

    @pytest.mark.asyncio
    async def test_unknown_item_types_warning_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unknown item types emit a WARNING log during compaction.

        Kept separate from the behavioural test so a log wording change doesn't
        obscure whether the item was actually preserved. WARNING level ensures
        operators see this in production without enabling debug logging.
        """
        mock = MockSession()
        unknown_item = {"type": "future_reasoning_item", "content": "thinking..."}
        await mock.add_items([user_msg("older"), unknown_item, user_msg("valid message")])
        compacting = LocalCompactionSession(mock, window_size=2)

        with caplog.at_level(logging.WARNING, logger="openai_agents_context_compaction.session"):
            await compacting.get_items()

        assert any(
            "unknown item type encountered" in record.message.lower() for record in caplog.records
        ), "Expected warning log about unknown item type"

    @pytest.mark.asyncio
    async def test_order_preserved_with_interleaved_messages(self) -> None:
        """Items between a function_call and its output retain their original position.

        Scenario: A user message sits between a function_call and its output
        (e.g., user commented while a tool was running). After compaction the
        original chronological order must be preserved — the user message must
        NOT be displaced before or after the pair.

        This verifies the index-marking algorithm preserves original ordering,
        unlike a pair-bundling approach that would group fc/fco together and
        displace the interleaved message.
        """
        mock = MockSession()
        fc = function_call("a", "slow_tool")
        msg = user_msg("why so slow?")
        fco = function_call_output("a", "done")
        await mock.add_items([fc, msg, fco])

        compacting = LocalCompactionSession(mock, window_size=3)
        items = await compacting.get_items()
        self._validate_invariants(items)
        # Original order must be preserved: fc, msg, fco — NOT msg, fc, fco
        assert items == [fc, msg, fco], (
            "Interleaved message must stay between function_call and its output"
        )

    @pytest.mark.asyncio
    async def test_order_preserved_with_batch_function_calls(self) -> None:
        """Batch function calls retain their original SDK ordering after compaction.

        The OpenAI Agents SDK emits batch tool calls as:
            [fc(a), fc(b), fco(a), fco(b)]
        i.e., all calls first, then all outputs. Compaction must preserve this
        ordering and NOT regroup into [fc(a), fco(a), fc(b), fco(b)].
        """
        mock = MockSession()
        fc_a = function_call("a", "tool_a")
        fc_b = function_call("b", "tool_b")
        fco_a = function_call_output("a", "result_a")
        fco_b = function_call_output("b", "result_b")
        await mock.add_items([fc_a, fc_b, fco_a, fco_b])

        compacting = LocalCompactionSession(mock, window_size=4)
        items = await compacting.get_items()
        self._validate_invariants(items)
        # Original SDK batch order must be preserved
        assert items == [fc_a, fc_b, fco_a, fco_b], (
            "Batch function call order must be preserved: all calls then all outputs"
        )

    @pytest.mark.asyncio
    async def test_order_preserved_partial_window_with_interleaved(self) -> None:
        """When window is too small for all items, kept items preserve original order.

        Setup: [msg1, fc(a), msg2, fco(a), msg3] — 5 items, window=4.
        The oldest item (msg1) is dropped, but the remaining 4 items must
        retain their original relative ordering including msg2 between the pair.
        """
        mock = MockSession()
        msg1 = user_msg("old")
        fc = function_call("a")
        msg2 = user_msg("middle")
        fco = function_call_output("a")
        msg3 = user_msg("recent")
        await mock.add_items([msg1, fc, msg2, fco, msg3])

        compacting = LocalCompactionSession(mock, window_size=4)
        items = await compacting.get_items()
        self._validate_invariants(items)
        assert len(items) <= 4
        # msg3 (most recent) must be present
        assert msg3 in items
        # The pair must be kept atomically
        assert fc in items
        assert fco in items
        # If msg2 is present, it must be between fc and fco (original order)
        if msg2 in items:
            fc_idx = items.index(fc)
            msg2_idx = items.index(msg2)
            fco_idx = items.index(fco)
            assert fc_idx < msg2_idx < fco_idx, (
                "Interleaved message must remain between function_call and output"
            )


# -------------------------------------------------------------------
# Token counting tests
# -------------------------------------------------------------------


class TestExtractText:
    """Tests for _extract_text — text extraction from Responses API items."""

    def test_function_call(self) -> None:
        item = function_call("c1", name="my_func", arguments='{"key": "value"}')
        text = _extract_text(item)
        assert "my_func" in text
        assert '{"key": "value"}' in text

    def test_function_call_output_string(self) -> None:
        item = function_call_output("c1", output="hello world")
        assert "hello world" in _extract_text(item)

    def test_function_call_output_list(self) -> None:
        item: TResponseInputItem = {
            "type": "function_call_output",
            "call_id": "c1",
            "output": [
                {"type": "input_text", "text": "part one"},
                {"type": "input_text", "text": "part two"},
            ],
        }
        text = _extract_text(item)
        assert "part one" in text
        assert "part two" in text

    def test_user_message_string_content(self) -> None:
        assert "hello" in _extract_text(user_msg("hello"))

    def test_assistant_message_list_content(self) -> None:
        assert "reply" in _extract_text(assistant_msg("reply"))

    def test_empty_item_returns_string(self) -> None:
        result = _extract_text({})  # type: ignore[arg-type]
        assert isinstance(result, str)


class TestCountTokens:
    """Tests for _count_tokens — token estimation for lists of items."""

    def test_empty_list(self) -> None:
        assert _count_tokens([]) == 0

    def test_single_item_has_overhead(self) -> None:
        # Each item has +4 overhead; even an empty-content item must count > 0.
        result = _count_tokens([user_msg("")])
        assert result >= 4

    def test_more_text_means_more_tokens(self) -> None:
        short = _count_tokens([user_msg("hi")])
        long = _count_tokens([user_msg("hello " * 100)])
        assert long > short

    def test_multiple_items_sum(self) -> None:
        one = _count_tokens([user_msg("hello")])
        two = _count_tokens([user_msg("hello"), user_msg("hello")])
        assert two == one * 2

    def test_fallback_tiny_text_nonzero(self) -> None:
        # Test default token counting for very short strings (edge case: single char).
        # Default uses max(1, len(text)//4) + 4 overhead.
        result = _count_tokens([user_msg("x")])
        # Default: max(1, len("x")//4) + 4
        #        = max(1, 1//4) + 4
        #        = max(1, 0) + 4
        #        = 1 + 4
        #        = 5
        assert result == 5

    def test_custom_token_counter(self) -> None:
        """A user-supplied token_counter callable is used instead of the default."""

        # Simple counter: 1 token per word
        def word_counter(text: str) -> int:
            return len(text.split())

        result = _count_tokens([user_msg("one two three")], token_counter=word_counter)
        # 3 words + 4 overhead = 7
        assert result == 7


class TestCustomTokenCounter:
    """Tests for pluggable token_counter on LocalCompactionSession."""

    @pytest.mark.asyncio
    async def test_custom_counter_used_in_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        """A custom token_counter is invoked during compaction logging."""
        call_count = 0

        def counting_counter(text: str) -> int:
            nonlocal call_count
            call_count += 1
            return len(text)  # 1 token per char

        mock = MockSession()
        await mock.add_items([user_msg(f"msg{i}") for i in range(5)])
        compacting = LocalCompactionSession(mock, window_size=3, token_counter=counting_counter)

        with caplog.at_level(logging.INFO, logger="openai_agents_context_compaction.session"):
            await compacting.get_items()

        assert call_count > 0, "Custom token_counter must be called during compaction"
        assert any("tokens:" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_token_counter_none_disables_counting(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Passing token_counter=None disables token counting in log output.

        Also verifies that the no-token-counter path still preserves function call
        pair atomicity — _boundary_aware_compact_with_indices and drop_orphaned_tool_outputs
        both run regardless of whether a token_counter is present.
        """
        mock = MockSession()
        fc = function_call("c1")
        fco = function_call_output("c1")
        # 5 items total; window_size=4 keeps the pair plus one older message
        await mock.add_items([user_msg("old"), user_msg("also_old"), fc, fco, user_msg("recent")])
        compacting = LocalCompactionSession(mock, window_size=4, token_counter=None)

        with caplog.at_level(logging.INFO, logger="openai_agents_context_compaction.session"):
            items = await compacting.get_items()

        # Should still log compaction but without token counts
        assert any("Compacted" in r.message for r in caplog.records)
        assert not any("tokens:" in r.message for r in caplog.records)

        # Pair atomicity must hold even without a token_counter
        _validate_invariants(items)
        assert fc in items
        assert fco in items

    @pytest.mark.asyncio
    async def test_get_items_returns_independent_copy(self) -> None:
        """Mutating the list returned by get_items() must not affect subsequent calls.

        The cache was removed, but the contract that callers receive an independent
        list still holds — drop_orphaned_tool_outputs returns a new list each time.
        """
        mock = MockSession()
        await mock.add_items([user_msg("a"), user_msg("b")])
        compacting = LocalCompactionSession(mock, window_size=10)

        result1 = await compacting.get_items()
        result1.clear()  # mutate the returned list

        result2 = await compacting.get_items()
        assert len(result2) == 2, "Mutating a returned list must not affect subsequent calls"


# -------------------------------------------------------------------
# Token budget compaction tests
# -------------------------------------------------------------------


def _char_counter(text: str) -> int:
    """Deterministic token counter: 1 token per character (no tiktoken dependency)."""
    return len(text)


def _token_cost(text: str) -> int:
    """Mirror the +4 overhead applied inside _count_tokens / get_items."""
    return len(text) + 4


class TestTokenBudgetCompaction:
    """Tests for the token_budget parameter on LocalCompactionSession."""

    # ------------------------------------------------------------------
    # A — Basic enforcement
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_token_budget_keeps_recent_items(self) -> None:
        """Token budget retains the most recent items that fit."""
        mock = MockSession()
        # Each user_msg("msgN") has content "msgN" → 4 chars → cost = 4+4 = 8 tokens
        await mock.add_items([user_msg(f"msg{i}") for i in range(5)])
        # Budget for exactly 2 items: 2 * 8 = 16
        compacting = LocalCompactionSession(mock, token_budget=16, token_counter=_char_counter)

        items = await compacting.get_items()

        assert len(items) == 2
        assert items[0] == user_msg("msg3")
        assert items[1] == user_msg("msg4")

    @pytest.mark.asyncio
    async def test_token_budget_none_disables_compaction(self) -> None:
        """token_budget=None leaves all items untouched."""
        mock = MockSession()
        await mock.add_items([user_msg(f"msg{i}") for i in range(5)])
        compacting = LocalCompactionSession(mock, token_budget=None, token_counter=_char_counter)

        items = await compacting.get_items()

        assert len(items) == 5

    @pytest.mark.asyncio
    async def test_all_items_fit_no_compaction(self) -> None:
        """When total tokens <= token_budget, all items are returned unchanged."""
        mock = MockSession()
        await mock.add_items([user_msg("hi"), user_msg("ok")])
        compacting = LocalCompactionSession(mock, token_budget=10000, token_counter=_char_counter)

        items = await compacting.get_items()

        assert len(items) == 2

    # ------------------------------------------------------------------
    # B — Function call pair atomicity
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_pair_kept_atomically_when_budget_covers_both(self) -> None:
        """A function call pair is kept when the budget covers both items."""
        mock = MockSession()
        fc = function_call("x", "my_tool", arguments='{"a":1}')
        fco = function_call_output("x", "result")
        await mock.add_items([user_msg("older older older"), fc, fco])
        # Budget: cost(fc) + cost(fco) but not older message
        budget = sum(_token_cost(_extract_text(i)) for i in [fc, fco])
        compacting = LocalCompactionSession(mock, token_budget=budget, token_counter=_char_counter)

        items = await compacting.get_items()

        _validate_invariants(items)
        assert fc in items
        assert fco in items

    @pytest.mark.asyncio
    async def test_pair_dropped_atomically_when_budget_too_small(self) -> None:
        """A pair is dropped entirely when the budget cannot cover both items."""
        mock = MockSession()
        fc = function_call("y", "tool", arguments="{}")
        fco = function_call_output("y", "out")
        recent = user_msg("recent")
        await mock.add_items([fc, fco, recent])
        # Budget: enough for recent only (pair cost > budget)
        recent_cost = len("recent") + 4
        compacting = LocalCompactionSession(
            mock, token_budget=recent_cost, token_counter=_char_counter
        )

        items = await compacting.get_items()

        _validate_invariants(items)
        assert fc not in items
        assert fco not in items
        assert recent in items

    @pytest.mark.asyncio
    async def test_soft_limit_keeps_first_pair_below_budget(self) -> None:
        """First pair is kept even when its cost exceeds the token budget (soft limit)."""
        mock = MockSession()
        fc = function_call("z", "big_tool", arguments='{"key": "value"}')
        fco = function_call_output("z", "big result output")
        await mock.add_items([fc, fco])
        # Budget of 1 is impossibly small — soft limit must fire
        compacting = LocalCompactionSession(mock, token_budget=1, token_counter=_char_counter)

        items = await compacting.get_items()

        _validate_invariants(items)
        assert fc in items
        assert fco in items

    @pytest.mark.asyncio
    async def test_soft_limit_keeps_oversized_standalone_message(self) -> None:
        """A single oversized message is kept even when it exceeds the token budget.

        Without soft-limit on standalone messages, token_budget=1 with a 500-token
        recent message would return [] — worse than no compaction at all.
        """
        mock = MockSession()
        big_msg = user_msg("x" * 200)  # cost = 200 + 4 = 204 tokens
        await mock.add_items([big_msg])
        compacting = LocalCompactionSession(mock, token_budget=1, token_counter=_char_counter)

        items = await compacting.get_items()

        assert items == [big_msg]

    @pytest.mark.asyncio
    async def test_soft_limit_includes_only_first_oversized_item(self) -> None:
        """After soft-limit inclusion, no further items are added."""
        mock = MockSession()
        old = user_msg("old message")
        recent = user_msg("x" * 200)  # oversized
        await mock.add_items([old, recent])
        compacting = LocalCompactionSession(mock, token_budget=1, token_counter=_char_counter)

        items = await compacting.get_items()

        assert recent in items
        assert old not in items

    @pytest.mark.asyncio
    async def test_pair_fits_exactly_at_budget(self) -> None:
        """A pair whose cost exactly equals the budget is kept; older items are excluded."""
        mock = MockSession()
        fc = function_call("e", "tool", arguments="{}")
        fco = function_call_output("e", "res")
        old = user_msg("older")
        await mock.add_items([old, fc, fco])
        budget = sum(_token_cost(_extract_text(i)) for i in [fc, fco])
        compacting = LocalCompactionSession(mock, token_budget=budget, token_counter=_char_counter)

        items = await compacting.get_items()

        _validate_invariants(items)
        assert fc in items
        assert fco in items
        assert old not in items

    @pytest.mark.asyncio
    async def test_standalone_fits_exactly_at_budget(self) -> None:
        """A standalone message whose cost exactly equals token_budget is included."""
        mock = MockSession()
        msg = user_msg("hello")
        old = user_msg("this is an older and longer message")
        await mock.add_items([old, msg])
        budget = _token_cost(_extract_text(msg))
        compacting = LocalCompactionSession(mock, token_budget=budget, token_counter=_char_counter)

        items = await compacting.get_items()

        assert msg in items
        assert old not in items

    @pytest.mark.asyncio
    async def test_standalone_fits_but_older_pair_does_not(self) -> None:
        """Standalone message is kept when the older pair cannot fit in the budget."""
        mock = MockSession()
        fc = function_call("f", "big_tool", arguments='{"key": "value"}')
        fco = function_call_output("f", "big result")
        recent = user_msg("hi")
        await mock.add_items([fc, fco, recent])
        # Budget only covers the standalone message
        budget = _token_cost(_extract_text(recent))
        compacting = LocalCompactionSession(mock, token_budget=budget, token_counter=_char_counter)

        items = await compacting.get_items()

        _validate_invariants(items)
        assert recent in items
        assert fc not in items
        assert fco not in items

    @pytest.mark.asyncio
    async def test_older_pair_dropped_when_only_recent_pair_fits(self) -> None:
        """When only the most recent pair fits, the older pair is dropped entirely."""
        mock = MockSession()
        fc1 = function_call("a", "tool1", arguments="{}")
        fco1 = function_call_output("a", "result1")
        fc2 = function_call("b", "tool2", arguments="{}")
        fco2 = function_call_output("b", "result2")
        await mock.add_items([fc1, fco1, fc2, fco2])
        # Budget: cost of pair 2 only
        budget = sum(_token_cost(_extract_text(i)) for i in [fc2, fco2])
        compacting = LocalCompactionSession(mock, token_budget=budget, token_counter=_char_counter)

        items = await compacting.get_items()

        _validate_invariants(items)
        assert fc1 not in items
        assert fco1 not in items
        assert fc2 in items
        assert fco2 in items

    # ------------------------------------------------------------------
    # C — Dual constraint (window_size + token_budget)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_item_count_wins_over_large_token_budget(self) -> None:
        """window_size is the binding constraint when token_budget is large."""
        mock = MockSession()
        await mock.add_items([user_msg(f"msg{i}") for i in range(10)])
        compacting = LocalCompactionSession(
            mock, window_size=2, token_budget=1_000_000, token_counter=_char_counter
        )

        items = await compacting.get_items()

        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_token_budget_wins_over_large_window_size(self) -> None:
        """token_budget is the binding constraint when window_size is large."""
        mock = MockSession()
        # Each msg costs len("msgN") + 4 = 8 tokens
        await mock.add_items([user_msg(f"msg{i}") for i in range(10)])
        compacting = LocalCompactionSession(
            mock, window_size=100, token_budget=8, token_counter=_char_counter
        )

        items = await compacting.get_items()

        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_dual_constraint_log_contains_both_values(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Compaction log includes both window and token_budget values when token is binding.

        5 items; window_size=3 would keep 3 (24 tokens), but token_budget=16 (= 2 * 8)
        is the tighter constraint and reduces the result to 2 items. The log must
        include both constraint values for observability regardless of which won.
        """
        mock = MockSession()
        # Each user_msg("msgN") costs len("msgN") + 4 = 8 tokens with _char_counter
        await mock.add_items([user_msg(f"msg{i}") for i in range(5)])
        compacting = LocalCompactionSession(
            mock, window_size=3, token_budget=16, token_counter=_char_counter
        )

        with caplog.at_level(logging.INFO, logger="openai_agents_context_compaction.session"):
            items = await compacting.get_items()

        assert len(items) == 2, "token_budget=16 is the binding constraint (2 × 8 tokens)"
        log_msgs = " ".join(r.message for r in caplog.records)
        assert "window=3" in log_msgs
        assert "token_budget=16" in log_msgs

    @pytest.mark.asyncio
    async def test_limit_and_token_budget_intersect(self) -> None:
        """limit passed to get_items() acts as item-count constraint alongside token_budget.

        With window_size=None, effective_size = limit. Both limit (item count) and
        token_budget (token count) apply; the tighter constraint wins.
        """
        mock = MockSession()
        # Each msg costs 8 tokens; limit=3 keeps 3, token_budget=8 keeps 1 — budget wins
        await mock.add_items([user_msg(f"msg{i}") for i in range(5)])
        compacting = LocalCompactionSession(mock, token_budget=8, token_counter=_char_counter)

        items = await compacting.get_items(limit=3)

        assert len(items) == 1
        assert items[0] == user_msg("msg4")

    @pytest.mark.asyncio
    async def test_interleaved_message_between_pair_dropped_by_token_budget(self) -> None:
        """A message interleaved between fc and fco is independently token-budget-accounted.

        mid_msg sits between fc and fco in the list, but is NOT treated as part of the pair.
        The pair's cost is computed as cost(fc) + cost(fco) only; mid_msg must compete for
        budget on its own. Here the budget covers the pair and recent but not mid_msg, so
        mid_msg is dropped while fc, fco, and recent are all retained.

        This tests the subtle property that positional proximity to a pair does not grant
        immunity from independent budget accounting.
        """
        mock = MockSession()
        fc = function_call("t", "tool", arguments="{}")
        fco = function_call_output("t", "res")
        mid_msg = user_msg("interleaved")
        recent = user_msg("hi")
        await mock.add_items([fc, mid_msg, fco, recent])
        # Budget covers pair + recent but not mid_msg
        budget = sum(_token_cost(_extract_text(i)) for i in [fc, fco, recent])
        compacting = LocalCompactionSession(mock, token_budget=budget, token_counter=_char_counter)

        items = await compacting.get_items()

        _validate_invariants(items)
        assert fc in items
        assert fco in items
        assert recent in items
        assert mid_msg not in items

    # ------------------------------------------------------------------
    # D — Validation / error cases
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_window_size_zero_with_token_budget_returns_empty(self) -> None:
        """window_size=0 short-circuits to [] even when token_budget has capacity.

        window_size <= 0 is the hard exit: one constraint says "nothing", result is empty.
        """
        mock = MockSession()
        await mock.add_items([user_msg(f"msg{i}") for i in range(5)])
        compacting = LocalCompactionSession(
            mock, window_size=0, token_budget=1_000_000, token_counter=_char_counter
        )

        items = await compacting.get_items()

        assert items == []

    def test_token_budget_zero_raises(self) -> None:
        """token_budget=0 raises ValueError at construction."""
        mock = MockSession()
        with pytest.raises(ValueError, match="token_budget"):
            LocalCompactionSession(mock, token_budget=0)

    def test_token_budget_negative_raises(self) -> None:
        """token_budget < 0 raises ValueError at construction."""
        mock = MockSession()
        with pytest.raises(ValueError, match="token_budget"):
            LocalCompactionSession(mock, token_budget=-1)

    def test_token_budget_without_token_counter_raises(self) -> None:
        """token_budget with token_counter=None raises ValueError at construction."""
        mock = MockSession()
        with pytest.raises(ValueError, match="token_counter"):
            LocalCompactionSession(mock, token_budget=8000, token_counter=None)

    @pytest.mark.asyncio
    async def test_zero_returning_token_counter_clamped_to_one(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A token_counter that returns 0 is clamped to 1 with a WARNING.

        After clamping each item costs 1+4=5 tokens. With token_budget=5, exactly
        one item fits — confirming no infinite inclusion and correct budget arithmetic.
        """

        def zero_counter(text: str) -> int:
            return 0

        mock = MockSession()
        await mock.add_items([user_msg(f"msg{i}") for i in range(5)])
        # Clamped cost per item = 1 + 4 = 5; budget = 5 keeps exactly the most recent item
        compacting = LocalCompactionSession(mock, token_budget=5, token_counter=zero_counter)

        with caplog.at_level(logging.WARNING, logger="openai_agents_context_compaction.session"):
            items = await compacting.get_items()

        assert any("clamping to 1" in r.message for r in caplog.records)
        assert len(items) == 1
        assert items[0] == user_msg("msg4")

    @pytest.mark.asyncio
    async def test_token_budget_only_triggers_compaction_and_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """token_budget alone (no window_size) triggers compaction and INFO log."""
        mock = MockSession()
        await mock.add_items([user_msg(f"msg{i}") for i in range(5)])
        compacting = LocalCompactionSession(mock, token_budget=16, token_counter=_char_counter)

        with caplog.at_level(logging.INFO, logger="openai_agents_context_compaction.session"):
            items = await compacting.get_items()

        assert len(items) < 5
        assert any("Compacted" in r.message for r in caplog.records)

    # ------------------------------------------------------------------
    # E — Invariants and ordering
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_order_preserved_after_token_budget_compaction(self) -> None:
        """Chronological order is preserved after token-budget compaction."""
        mock = MockSession()
        fc = function_call("p", "tool", arguments="{}")
        fco = function_call_output("p", "res")
        msg = user_msg("recent")
        await mock.add_items([user_msg("old1"), user_msg("old2"), fc, fco, msg])
        # Budget for pair + recent msg — use _extract_text to get exact costs
        budget = sum(_token_cost(_extract_text(i)) for i in [fc, fco, msg])
        compacting = LocalCompactionSession(mock, token_budget=budget, token_counter=_char_counter)

        items = await compacting.get_items()

        _validate_invariants(items)
        # fc must appear before fco, fco before msg
        assert items.index(fc) < items.index(fco) < items.index(msg)

    @pytest.mark.asyncio
    async def test_large_history_tight_budget_returns_only_recent(self) -> None:
        """Tight budget on a large history returns only the most recent items."""
        mock = MockSession()
        # 100 messages — cost per msg = len("msg_NNN") + 4 = 11 tokens
        await mock.add_items([user_msg(f"msg_{i:03d}") for i in range(100)])
        # Budget for exactly 2 messages
        budget = 2 * (len("msg_099") + 4)
        compacting = LocalCompactionSession(mock, token_budget=budget, token_counter=_char_counter)

        items = await compacting.get_items()

        assert len(items) == 2
        assert items[0] == user_msg("msg_098")
        assert items[1] == user_msg("msg_099")

    @pytest.mark.asyncio
    async def test_orphaned_output_as_most_recent_item_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An orphaned function_call_output as the most recent item logs a WARNING.

        This indicates bad session data. Compaction cannot include an orphan (it would
        be removed by drop_orphaned_tool_outputs anyway), so the result may be empty.
        Operators need visibility into this via a WARNING rather than a silent DEBUG log.
        """
        mock = MockSession()
        orphan_output = function_call_output("no_matching_call", "result")
        await mock.add_items([user_msg("older"), orphan_output])
        compacting = LocalCompactionSession(mock, window_size=1, token_counter=_char_counter)

        with caplog.at_level(logging.WARNING, logger="openai_agents_context_compaction.session"):
            items = await compacting.get_items()

        assert any("most recent item is an orphaned" in r.message.lower() for r in caplog.records)
        # The orphan is not in the result (drop_orphaned_tool_outputs removes it)
        assert orphan_output not in items

    # ------------------------------------------------------------------
    # F — Repr
    # ------------------------------------------------------------------

    def test_token_budget_in_repr(self) -> None:
        """token_budget value appears in repr output."""
        mock = MockSession(session_id="s1")
        compacting = LocalCompactionSession(
            mock, window_size=50, token_budget=8000, token_counter=_char_counter
        )
        r = repr(compacting)
        assert "8000" in r
        assert "50" in r


# -------------------------------------------------------------------
# Observability: limiting_factor, get_compaction_config
# -------------------------------------------------------------------


class TestDeterminingLimitingFactor:
    """Unit tests for _determine_limiting_factor — pure function, no async needed."""

    @pytest.mark.parametrize(
        "items_before, items_after, item_budget, tokens_before, token_budget, soft_limit_fired, expected",  # noqa: E501
        [
            # tight window, fits in token budget → window_size
            (10, 5, 5, 400, 100_000, False, "window_size"),
            # fits in window, token budget exceeded → token_budget
            (10, 2, 20, 80, 16, False, "token_budget"),
            # both exceeded, items_after < item_budget → token was the tighter constraint
            (10, 3, 5, 300, 100, False, "token_budget"),
            # both exceeded, items_after == item_budget → window hit first (or simultaneously)
            (10, 5, 5, 300, 100, False, "window_size"),
            # no item budget — window never binding → token_budget
            (10, 2, None, 80, 16, False, "token_budget"),
            # soft_limit_fired=True overrides any constraint inference → soft_limit
            # (constraints technically exceeded, but the signal from the walk wins)
            (10, 1, 5, 300, 100, True, "soft_limit"),
            # soft_limit_fired=True when nothing was exceeded (original observable case)
            (1, 1, 10, 8, 100, True, "soft_limit"),
        ],
    )
    def test_limiting_factor(
        self,
        items_before: int,
        items_after: int,
        item_budget: int | None,
        tokens_before: int,
        token_budget: int | None,
        soft_limit_fired: bool,
        expected: str,
    ) -> None:
        assert (
            _determine_limiting_factor(
                items_before,
                items_after,
                item_budget,
                tokens_before,
                token_budget,
                soft_limit_fired=soft_limit_fired,
            )
            == expected
        )


class TestObservability:
    """Tests for get_compaction_config."""

    def test_get_compaction_config(self) -> None:
        """get_compaction_config returns current configuration."""
        mock = MockSession(session_id="cfg-test")
        compacting = LocalCompactionSession(
            mock, window_size=10, token_budget=5000, token_counter=_char_counter
        )
        config = compacting.get_compaction_config()
        assert config == {
            "session_id": "cfg-test",
            "window_size": 10,
            "token_budget": 5000,
            "token_counter_type": "function",  # plain functions report type.__name__ == "function"
        }

    def test_get_compaction_config_defaults(self) -> None:
        """get_compaction_config reflects None values and default counter name."""
        mock = MockSession()
        assert (
            LocalCompactionSession(mock, token_counter=None).get_compaction_config()[
                "token_counter_type"
            ]
            is None
        )
        assert (
            LocalCompactionSession(mock).get_compaction_config()["token_counter_type"]
            == "DefaultTokenCounter"
        )
