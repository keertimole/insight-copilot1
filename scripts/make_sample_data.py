"""Generate the synthetic fallback dataset:  python scripts/make_sample_data.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insight_copilot.data import SAMPLE_PATH  # noqa: E402
from insight_copilot.sample_data import generate  # noqa: E402

print("wrote", generate(SAMPLE_PATH))
