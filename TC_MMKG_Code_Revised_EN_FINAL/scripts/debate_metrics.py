# -*- coding: utf-8 -*-
"""Metrics used in manuscript Sec. 3.2 for adversarial-debate evaluation."""
from __future__ import annotations

from typing import Iterable, Sequence
from scripts.common import canonical_triple, jaccard


def debate_stability_score(round_triple_sets: Sequence[Iterable[Sequence[str]]]) -> float:
    """DSS = mean Jaccard similarity between consecutive debate rounds."""
    if len(round_triple_sets) < 2:
        return 1.0
    vals = [jaccard(round_triple_sets[i], round_triple_sets[i+1]) for i in range(len(round_triple_sets)-1)]
    return sum(vals) / len(vals)


def consistency_gain(initial_accuracy_a: float, initial_accuracy_b: float, final_accuracy: float) -> float:
    """CG = Acc_final - mean(Acc_initial producers)."""
    return float(final_accuracy) - (float(initial_accuracy_a) + float(initial_accuracy_b)) / 2.0


def hallucination_suppression_rate(before_incorrect: int, after_incorrect: int) -> float:
    """HSR = (H_before - H_after) / H_before."""
    if before_incorrect <= 0:
        return 1.0 if after_incorrect <= 0 else 0.0
    return max(0.0, min(1.0, (before_incorrect - after_incorrect) / before_incorrect))


def conflict_resolution_accuracy(correct_arbitrations: int, triggered_conflicts: int) -> float:
    """CRA = N_correct / N_conflict."""
    if triggered_conflicts <= 0:
        return 1.0
    return max(0.0, min(1.0, correct_arbitrations / triggered_conflicts))
