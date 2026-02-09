"""Tests for LocalCompactionSession."""

from __future__ import annotations

import pytest
from agents import Session, TResponseInputItem

from openai_agents_context_compaction import LocalCompactionSession


class MockSession(Session):
    """A mock session implementing the Session protocol for testing."""

    def __init__(self, session_id: str = "test-session") -> None:
        self.session_id = session_id
        self._items: list[TResponseInputItem] = []

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        if limit is None:
            return list(self._items)
        return list(self._items[-limit:])

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        self._items.extend(items)

    async def pop_item(self) -> TResponseInputItem | None:
        if self._items:
            return self._items.pop()
        return None

    async def clear_session(self) -> None:
        self._items.clear()


class TestLocalCompactionSession:
    """Tests for LocalCompactionSession pass-through behavior using public API only."""

    @pytest.mark.asyncio
    async def test_session_id_delegates(self) -> None:
        mock = MockSession(session_id="my-custom-id")
        compacting = LocalCompactionSession(mock)
        assert compacting.session_id == "my-custom-id"

    @pytest.mark.asyncio
    async def test_get_items_delegates(self) -> None:
        mock = MockSession()
        test_items: list[TResponseInputItem] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        await mock.add_items(test_items)

        compacting = LocalCompactionSession(mock)
        items = await compacting.get_items()

        assert items == test_items

    @pytest.mark.asyncio
    async def test_get_items_with_limit_delegates(self) -> None:
        mock = MockSession()
        test_items: list[TResponseInputItem] = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
        ]
        await mock.add_items(test_items)

        compacting = LocalCompactionSession(mock)
        items = await compacting.get_items(limit=2)

        assert items == test_items[-2:]

    @pytest.mark.asyncio
    async def test_add_items_delegates(self) -> None:
        mock = MockSession()
        compacting = LocalCompactionSession(mock)

        test_items: list[TResponseInputItem] = [
            {"role": "user", "content": "Hello"},
        ]
        await compacting.add_items(test_items)
        items = await mock.get_items()
        assert items == test_items

    @pytest.mark.asyncio
    async def test_pop_item_delegates(self) -> None:
        mock = MockSession()
        test_item: TResponseInputItem = {"role": "user", "content": "Hello"}
        await mock.add_items([test_item])

        compacting = LocalCompactionSession(mock)
        popped = await compacting.pop_item()

        assert popped == test_item
        items_after = await mock.get_items()
        assert items_after == []

    @pytest.mark.asyncio
    async def test_pop_item_returns_none_when_empty(self) -> None:
        mock = MockSession()
        compacting = LocalCompactionSession(mock)

        popped = await compacting.pop_item()
        assert popped is None

    @pytest.mark.asyncio
    async def test_clear_session_delegates(self) -> None:
        mock = MockSession()
        await mock.add_items([{"role": "user", "content": "Hello"}])

        compacting = LocalCompactionSession(mock)
        await compacting.clear_session()
        items_after = await mock.get_items()
        assert items_after == []

    @pytest.mark.asyncio
    async def test_importable_from_package(self) -> None:
        from openai_agents_context_compaction import LocalCompactionSession as Imported

        assert Imported is LocalCompactionSession

    def test_implements_session_protocol(self) -> None:
        mock = MockSession()
        compacting = LocalCompactionSession(mock)
        assert isinstance(compacting, Session)
