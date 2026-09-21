"""Shared pytest fixtures.

The `src/` path injection here makes `import arnlp` work without
`pip install -e .`. Production users should `pip install -e .` and
remove the sys.path hack.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import json
import pytest


@pytest.fixture(scope="session")
def sample_articles() -> list[dict]:
    """Tiny hand-picked article set covering preprocessing edge cases."""
    path = Path(__file__).parent / "fixtures" / "sample_articles.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
