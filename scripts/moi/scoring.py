#!/usr/bin/env python3
"""Shared MoI-Eval question and sample score aggregation.

The protocol is intentionally independent of a particular VLM backend:

1. every atomic question receives one score in ``{0, 0.6, 1}``;
2. questions are averaged within each applicable L2 criterion;
3. applicable L2 criterions are averaged within IF, VQ, and DP;
4. the available L1 scores are combined with the reviewed L1 weights using a
   weighted geometric mean.

Missing model outputs and unsupported capabilities are handled by the caller,
which can assign zero to every active question and then use the same function.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable, Mapping


SCORE_VALUES = (0.0, 0.6, 1.0)
GROUP_ORDER = ("IF", "VQ", "DP")
EVAL_PROTOCOL_VERSION = "moi-eval-v1"


def normalize_score(value: Any) -> float:
    """Return a canonical MoI score or raise for an out-of-protocol value."""
    if isinstance(value, bool):
        raise ValueError("boolean is not a score")
    score = float(value)
    for allowed in SCORE_VALUES:
        if abs(score - allowed) < 1e-8:
            return allowed
    raise ValueError(f"score must be one of {SCORE_VALUES}, got {value!r}")


def group_from_criterion(criterion: str) -> str:
    prefix = str(criterion).split("-", 1)[0]
    if prefix not in GROUP_ORDER:
        raise ValueError(f"unknown MoI L2 criterion: {criterion!r}")
    return prefix


def aggregate_question_scores(
    question_results: Iterable[Mapping[str, Any]],
    group_weights: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate atomic scores into L2, L1, and sample-level quality."""
    per_l2: dict[str, list[float]] = defaultdict(list)
    per_question: dict[str, float] = {}
    for index, result in enumerate(question_results, 1):
        criterion = str(result.get("criterion") or "")
        group_from_criterion(criterion)
        score = normalize_score(result.get("score"))
        question_id = str(result.get("question_id") or result.get("qid") or index)
        if question_id in per_question:
            raise ValueError(f"duplicate question id: {question_id}")
        per_question[question_id] = score
        per_l2[criterion].append(score)

    if not per_l2:
        raise ValueError("cannot aggregate an empty question set")

    criterion_scores = {criterion: mean(values) for criterion, values in per_l2.items()}
    l1_members: dict[str, list[float]] = defaultdict(list)
    for criterion, score in criterion_scores.items():
        l1_members[group_from_criterion(criterion)].append(score)
    group_scores = {
        l1: mean(l1_members[l1])
        for l1 in GROUP_ORDER
        if l1_members.get(l1)
    }

    active_weights = {
        l1: float(group_weights.get(l1, 0.0))
        for l1 in group_scores
    }
    if any(not math.isfinite(weight) or weight <= 0 for weight in active_weights.values()):
        raise ValueError(f"every active L1 must have a positive weight: {active_weights}")
    raw_weights = active_weights
    weight_sum = sum(raw_weights.values())
    if weight_sum <= 0:
        raise ValueError(f"active L1 weights sum to zero: {raw_weights}")
    normalized_weights = {
        l1: weight / weight_sum for l1, weight in raw_weights.items()
    }

    if any(group_scores[l1] <= 0.0 for l1 in normalized_weights):
        usable_score = 0.0
    else:
        usable_score = math.exp(
            sum(
                normalized_weights[l1] * math.log(group_scores[l1])
                for l1 in normalized_weights
            )
        )

    return {
        "score_values": list(SCORE_VALUES),
        "per_question": per_question,
        "criterion_scores": criterion_scores,
        "group_scores": group_scores,
        "normalized_group_weights": normalized_weights,
        "usable_score": usable_score,
        "aggregation": "question_mean_then_criterion_mean_then_weighted_geometric_group",
    }


def zero_question_results(question_record: Mapping[str, Any], reason: str) -> list[dict[str, Any]]:
    """Create traceable zero-valued results for an unsupported/missing sample."""
    return [
        {
            "question_id": str(question.get("question_id") or ""),
            "criterion": str(question.get("criterion") or ""),
            "question": str(question.get("question") or ""),
            "question_zh": str(question.get("question_zh") or ""),
            "grounding": dict(question.get("grounding") or {}),
            "success_condition": str(question.get("success_condition") or ""),
            "partial_condition": str(question.get("partial_condition") or ""),
            "failure_condition": str(question.get("failure_condition") or ""),
            "score": 0.0,
            "raw": reason,
        }
        for question in question_record.get("questions") or []
    ]
