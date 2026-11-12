import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db


def test_hash_ignores_whitespace():
    assert db.content_hash("a  b\n", "c d") == db.content_hash("a b", "c   d")


def test_hash_differs_for_different_inputs():
    assert db.content_hash("a", "b") != db.content_hash("a", "c")


def test_update_status_rejects_unknown_status():
    try:
        db.update_status(1, "Ghosted")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
