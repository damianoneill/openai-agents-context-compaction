# openai-agents-context-compaction

[![PyPI version](https://img.shields.io/pypi/v/openai-agents-context-compaction.svg)](https://pypi.org/project/openai-agents-context-compaction/)
[![CI](https://github.com/damianoneill/openai-agents-context-compaction/actions/workflows/ci.yml/badge.svg)](https://github.com/damianoneill/openai-agents-context-compaction/actions/workflows/ci.yml)
[![Compatibility](https://github.com/damianoneill/openai-agents-context-compaction/actions/workflows/compatibility.yml/badge.svg)](https://github.com/damianoneill/openai-agents-context-compaction/actions/workflows/compatibility.yml)

Context compaction support for the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python).

Enables intelligent management of conversation history token size for multi-turn agent interactions.

> **Note:** This package was created in response to [openai/openai-agents-python#2244](https://github.com/openai/openai-agents-python/issues/2244) to provide runner-level context compaction capabilities.

## Problem

As agent conversations grow longer, the token count sent to the LLM increases, leading to:

- **Higher costs** - LLM pricing is token-based
- **Degraded reasoning** - Models can lose focus with very long contexts
- **Context window limits** - Models have maximum token limits that cannot be exceeded
- **Latency increases** - Longer prompts take more time to process

## Solution

This package extends the existing OpenAI Agents SDK `Session` protocol with compaction strategies that operate locally without requiring the OpenAI Responses API compaction endpoint, making it provider-agnostic.

## Installation

```bash
pip install openai-agents-context-compaction
```

## Usage

Wrap any existing session with `LocalCompactionSession` to add automatic compaction support:

```python
from agents import Agent, Runner, SQLiteSession
from openai_agents_context_compaction import LocalCompactionSession, SlidingWindowStrategy

# Create your agent
agent = Agent(name="Assistant", instructions="You are a helpful assistant.")

# Wrap an existing session with compaction support
underlying = SQLiteSession("conversation_123")
session = LocalCompactionSession(
    session_id="conversation_123",
    underlying_session=underlying,
    strategy=SlidingWindowStrategy(window_size=30),
    max_tokens=128_000,
)

# Use normally - compaction happens automatically when needed
result = await Runner.run(agent, "Hello!", session=session)
```

The `SlidingWindowStrategy` keeps only the most recent N messages, automatically discarding older ones when the conversation grows too long.

## Compatibility

This package is tested weekly against the latest OpenAI Agents SDK to ensure compatibility.

## License

MIT
