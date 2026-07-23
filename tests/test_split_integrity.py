"""Split-integrity tests: mechanically enforce the leak-certification invariant that
docs/case_study.md certifies by audit — no cluster ID may appear in more than one
split. Runs from the committed splits files alone (stdlib only, no data downloads).
"""

import json
from itertools import combinations
from pathlib import Path

SPLITS = Path(__file__).resolve().parent.parent / "data" / "features" / "splits_v2.json"


def _load_splits():
    raw = json.loads(SPLITS.read_text())
    # split -> set of cluster ids, ignoring bookkeeping keys like _image_index
    return {
        name: {cid for ids in per_brand.values() for cid in ids}
        for name, per_brand in raw.items()
        if not name.startswith("_")
    }


def test_expected_splits_present_and_nonempty():
    splits = _load_splits()
    for name in ("train", "val", "test", "temporal_test"):
        assert name in splits, f"missing split: {name}"
        assert splits[name], f"empty split: {name}"


def test_no_cluster_appears_in_two_splits():
    splits = _load_splits()
    for a, b in combinations(splits, 2):
        overlap = splits[a] & splits[b]
        assert not overlap, f"{len(overlap)} clusters shared between {a} and {b}: " \
                            f"{sorted(overlap)[:5]}"


def test_temporal_holdout_is_rhode_only():
    raw = json.loads(SPLITS.read_text())
    assert set(raw["temporal_test"].keys()) == {"rhode"}
