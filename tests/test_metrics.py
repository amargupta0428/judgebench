"""Adversarial regression tests for the report-card metric implementations.

Each test encodes a bug class that has actually bitten this repo (or nearly did):
tied scores fabricating AUC/Spearman discrimination (July 23 correction #1),
exact-1.0 scores falling out of the final ECE bin, and percentile detection
thresholds silently operating far below the target FPR on discrete scorers
(July 23 correction #2). If one of these fails, a metric change reintroduced a
known bug — do not ship the number.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.report_card import auc, det_threshold, ece, spearman


def test_auc_all_tied_is_half():
    # a judge scoring everything identically has no discrimination
    assert auc([3, 3, 3], [3, 3, 3, 3]) == 0.5


def test_auc_tied_pairs_count_half():
    # one pos above all negs, one pos tied with one neg: (1 + 0.5 + 1 + 1) / 4
    assert abs(auc([1.0, 0.5], [0.5, 0.0]) - 0.875) < 1e-12


def test_auc_perfect_separation():
    assert auc([2, 3], [0, 1]) == 1.0


def test_spearman_constant_scorer_is_zero():
    assert spearman([1, 2, 3, 4], [7, 7, 7, 7]) == 0.0


def test_spearman_ties_match_scipy_convention():
    # monotone with a tie block still yields positive, not fabricated-perfect, rho
    r = spearman([1, 2, 3, 4], [0.1, 0.5, 0.5, 0.9])
    assert 0 < r < 1


def test_ece_includes_exact_one():
    # a confident wrong prediction at exactly 1.0 must contribute error;
    # the old half-open final bin dropped it and returned 0.0
    assert ece([1.0], [0]) == 1.0


def test_ece_perfect_calibration_at_one():
    assert ece([1.0, 1.0], [1, 1]) == 0.0


def test_det_threshold_reports_realized_fpr_on_ties():
    # 94% of positives tied at the 5th-percentile value: strict < detects none
    pos = [0.3] * 94 + [0.5] * 6
    thr, realized = det_threshold(pos, 0.05)
    assert realized == 0.0  # must be REPORTED, not assumed to be 5%


def test_det_threshold_continuous_scores_near_target():
    pos = [i / 1000 for i in range(1000)]
    _, realized = det_threshold(pos, 0.05)
    assert abs(realized - 0.05) < 0.01
