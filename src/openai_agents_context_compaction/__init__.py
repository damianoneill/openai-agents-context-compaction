"""OpenAI Agents Context Compaction.

Context compaction support for the OpenAI Agents SDK Runner loop.
Enables automatic context compaction when conversation history grows too large.
"""

from openai_agents_context_compaction._version import __version__
from openai_agents_context_compaction.session import LocalCompactionSession

__all__ = [
    "__version__",
    "LocalCompactionSession",
]
