"""Local compaction session wrapper.

Provides a Session wrapper that delegates all calls to an underlying session.
This is the foundation for implementing context compaction strategies.
"""

from __future__ import annotations

from agents import Session, TResponseInputItem


class LocalCompactionSession(Session):
    """A Session wrapper that delegates all calls to an underlying session.

    This class implements the Session protocol by wrapping another Session
    instance and delegating all method calls to it. This provides a foundation
    for implementing context compaction strategies in future iterations.

    Currently, this is a pure pass-through implementation with no compaction.

    Args:
        session: The underlying session to wrap.

    Example:
        >>> from agents import SQLiteSession
        >>> underlying = SQLiteSession(session_id="my-session")
        >>> compacting_session = LocalCompactionSession(underlying)
        >>> # Use compacting_session wherever a Session is expected
    """

    def __init__(self, session: Session) -> None:
        """Initialize the LocalCompactionSession.

        Args:
            session: The underlying session to delegate calls to.
        """
        self._session = session

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

        Args:
            limit: Maximum number of items to retrieve. If None, retrieves all items.
                   When specified, returns the latest N items in chronological order.

        Returns:
            List of input items representing the conversation history.
        """
        return await self._session.get_items(limit)

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
