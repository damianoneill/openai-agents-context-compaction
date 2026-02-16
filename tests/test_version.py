"""Version tests."""

import re

from openai_agents_context_compaction import __version__


def test_version():
    """Test that version follows semver format."""
    assert re.match(r"^\d+\.\d+\.\d+$", __version__)
