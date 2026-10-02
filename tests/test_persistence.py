"""The checkpointer choice: in-memory by default, SQLite when CHECKPOINT_DB is set, never a crash."""
import pytest
from langgraph.checkpoint.memory import MemorySaver

from insight_copilot import graph as G


def _is_memory(saver) -> bool:
    """Newer langgraph renamed the class to InMemorySaver and kept MemorySaver as an alias, so compare by type, not name."""
    return isinstance(saver, MemorySaver)


def test_default_is_in_memory(monkeypatch):
    monkeypatch.delenv("CHECKPOINT_DB", raising=False)
    assert _is_memory(G.default_checkpointer())


def test_disabled_values_use_memory(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_DB", "off")
    assert _is_memory(G.default_checkpointer())


def test_unwritable_path_falls_back_to_memory(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("CHECKPOINT_DB", str(blocker / "cp.db"))           # a file cannot be a directory
    assert _is_memory(G.default_checkpointer())


def test_sqlite_checkpointer_when_configured(monkeypatch, tmp_path):
    pytest.importorskip("langgraph.checkpoint.sqlite")                     # optional package
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "cp.db"))
    assert G.default_checkpointer().__class__.__name__ == "SqliteSaver"
