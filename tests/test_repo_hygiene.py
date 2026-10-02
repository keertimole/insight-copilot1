"""Guards the 'do not commit API keys' rule from the brief: no secret-looking strings in any tracked source/config file."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".venv", ".git", "__pycache__", ".pytest_cache", "data"}
PATTERN = re.compile(r"(gsk_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|sk-ant-[A-Za-z0-9_-]{20,})")


def test_no_api_keys_in_repo_files():
    bad = []
    for p in ROOT.rglob("*"):
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP) and p.name != ".env" and p.suffix in {
                ".py", ".md", ".toml", ".example", ".json", ".txt", ".mmd", ".yml", ".yaml", ""}:
            try:
                if PATTERN.search(p.read_text(errors="ignore")):
                    bad.append(str(p.relative_to(ROOT)))
            except OSError:
                pass
    assert not bad, f"secret-looking strings found in: {bad}"


def test_gitignore_covers_secret_files():
    gi = (ROOT / ".gitignore").read_text()
    for entry in (".env", ".streamlit/secrets.toml", ".venv/"):
        assert entry in gi
