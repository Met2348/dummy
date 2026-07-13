#!/usr/bin/env python3
"""Run the 5 x 20 TRACE-H synthetic transport kill-test matrix."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from traceh_core.transport import (
    balanced_plan,
    conservative_actions,
    pairwise_squared_euclidean,
    transport_responses,
    unbalanced_partial_plan,
)


ACTION_COSTS = np.array([0.0, 0.02, 0.02, 0.04])
warnings.filterwarnings("error", message=".*Sinkhorn did not converge.*")
warnings.filterwarnings("error", message=".*Unbalanced Sinkhorn did not converge.*")
CLASS_ADVANTAGES = np.array(
    [
        [0.0, 0.80, -0.20, 0.10],
        [0.0, -0.20, 0.70, 0.10],
        [0.0, 0.10, -0.10, 0.80],
        [0.0, -0.60, -0.50, -0.40],
    ]
)
CENTERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.9], [8.0, 8.0]])


def sample_classes(rng: np.random.Generator, probabilities: list[float], count: int) -> np.ndarray:
    return rng.choice(len(probabilities), size=count, p=probabilities)


def sample_features(
    rng: np.random.Generator,
    labels: np.ndarray,
    noise: float = 0.03,
    scale: float = 1.0,
) -> np.ndarray:
    return CENTERS[labels] * scale + rng.normal(0.0, noise, size=(len(labels), 2))


def evaluate(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    *,
    partial: bool,
    cost: np.ndarray | None = None,
) -> dict[str, float]:
    matrix = pairwise_squared_euclidean(source_features, target_features) if cost is None else cost
    plan = (
        unbalanced_partial_plan(matrix, reg=0.10, reg_m=0.25, max_match_cost=0.18)
        if partial
        else balanced_plan(matrix, reg=0.10)
    )
    response = transport_responses(plan, CLASS_ADVANTAGES[source_labels])
    selected, _ = conservative_actions(
        response,
        ACTION_COSTS,
        coverage_threshold=0.30,
        z_value=0.25,
    )
    oracle = np.argmax(CLASS_ADVANTAGES[target_labels] - ACTION_COSTS[None, :], axis=1)
    chosen_utility = CLASS_ADVANTAGES[target_labels, selected] - ACTION_COSTS[selected]
    private = target_labels == 3
    return {
        "accuracy": float(np.mean(selected == oracle)),
        "mean_utility": float(np.mean(chosen_utility)),
        "mean_coverage": float(np.mean(response.coverage)),
        "private_none_rate": float(np.mean(selected[private] == 0)) if np.any(private) else 1.0,
        "private_coverage": float(np.mean(response.coverage[private])) if np.any(private) else 0.0,
    }


def standard_scenario(seed: int, name: str) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    source_labels = sample_classes(rng, [1 / 3, 1 / 3, 1 / 3], 60)
    if name == "identical_support":
        target_labels = sample_classes(rng, [1 / 3, 1 / 3, 1 / 3], 60)
        scale = 1.0
    elif name == "frequency_shift":
        target_labels = sample_classes(rng, [0.75, 0.20, 0.05], 60)
        scale = 1.0
    elif name == "missing_target_class":
        target_labels = sample_classes(rng, [0.65, 0.35], 60)
        scale = 1.0
    elif name == "private_30pct":
        target_labels = sample_classes(rng, [0.24, 0.23, 0.23, 0.30], 60)
        scale = 1.0
    else:
        raise ValueError(name)
    source = sample_features(rng, source_labels)
    target = sample_features(rng, target_labels, scale=scale)
    return {
        "balanced": evaluate(source, target, source_labels, target_labels, partial=False),
        "partial_lcb": evaluate(source, target, source_labels, target_labels, partial=True),
    }


def semantic_conflict(seed: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    source_labels = sample_classes(rng, [0.5, 0.5], 60)
    target_labels = sample_classes(rng, [0.5, 0.5], 60)

    source_structure = source_labels.astype(float) + rng.normal(0, 0.02, len(source_labels))
    target_structure = target_labels.astype(float) + rng.normal(0, 0.02, len(target_labels))
    source_semantic = 1.0 - source_labels + rng.normal(0, 0.02, len(source_labels))
    target_semantic = target_labels + rng.normal(0, 0.02, len(target_labels))
    source = np.column_stack([source_structure, source_semantic])
    target = np.column_stack([target_structure, target_semantic])

    semantic_cost = pairwise_squared_euclidean(source[:, 1:2], target[:, 1:2])
    response_cost = pairwise_squared_euclidean(source[:, 0:1], target[:, 0:1])
    return {
        "semantic_only": evaluate(
            source, target, source_labels, target_labels, partial=True, cost=semantic_cost
        ),
        "response_aware": evaluate(
            source, target, source_labels, target_labels, partial=True, cost=response_cost
        ),
    }


def aggregate(rows: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    methods = rows[0].keys()
    return {
        method: {
            metric: float(np.mean([row[method][metric] for row in rows]))
            for metric in rows[0][method]
        }
        for method in methods
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenarios: dict[str, dict[str, dict[str, float]]] = {}
    for name in (
        "identical_support",
        "frequency_shift",
        "missing_target_class",
        "private_30pct",
    ):
        scenarios[name] = aggregate([standard_scenario(seed, name) for seed in range(20)])
    scenarios["semantic_conflict"] = aggregate([semantic_conflict(seed) for seed in range(20)])

    assertions = {
        "identical_partial_not_worse": (
            scenarios["identical_support"]["partial_lcb"]["accuracy"]
            >= scenarios["identical_support"]["balanced"]["accuracy"] - 0.05
        ),
        "private_none_rate_ge_90pct": (
            scenarios["private_30pct"]["partial_lcb"]["private_none_rate"] >= 0.90
        ),
        "private_coverage_le_5pct": (
            scenarios["private_30pct"]["partial_lcb"]["private_coverage"] <= 0.05
        ),
        "response_aware_beats_semantic_by_30pp": (
            scenarios["semantic_conflict"]["response_aware"]["accuracy"]
            >= scenarios["semantic_conflict"]["semantic_only"]["accuracy"] + 0.30
        ),
    }
    report = {
        "experiment_id": "L4.1",
        "scenario_count": 5,
        "seeds_per_scenario": 20,
        "simulation_count": 100,
        "scenarios": scenarios,
        "assertions": assertions,
        "ok": all(assertions.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
