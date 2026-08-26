import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_historical_holdouts_v2_through_v6_are_byte_identical_to_checkpoint() -> None:
    manifest_path = ROOT / "evals/configs/historical-holdouts-v2-v6.sha256.json"
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))

    observed = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in expected
    }

    assert observed == expected
