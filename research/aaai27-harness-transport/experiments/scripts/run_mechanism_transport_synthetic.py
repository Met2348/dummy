#!/usr/bin/env python3
"""Synthetic transport probe using structured mechanism variants as actions."""

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


warnings.filterwarnings("error", message=".*Sinkhorn did not converge.*")
warnings.filterwarnings("error", message=".*Unbalanced Sinkhorn did not converge.*")

ACTION_NAMES = (
    "NONE",
    "natural_replan",
    "anti_loop_retry",
    "precondition_check",
    "subgoal_ledger",
    "bundle_conservative",
)
NONE_INDEX = 0
CLASS_NAMES = ("loop_escape", "missing_take", "missing_open", "two_object", "private")
SOURCE_CLASS_ADVANTAGES = np.array(
    [
        [0.0, 0.96, 0.98, 0.00, 0.00, 0.96],
        [0.0, -0.34, -0.32, 0.98, 0.97, 0.96],
        [0.0, -0.34, -0.32, 0.98, -0.33, 0.96],
        [0.0, -0.34, -0.32, 0.98, 0.97, 0.96],
    ],
    dtype=np.float64,
)
TARGET_CLASS_ADVANTAGES = np.vstack(
    [
        SOURCE_CLASS_ADVANTAGES,
        np.array([[0.0, -0.44, -0.42, -0.40, -0.40, -0.40]], dtype=np.float64),
    ]
)
CENTERS = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [5.0, 5.0],
    ],
    dtype=np.float64,
)


def sample_labels(rng: np.random.Generator, probabilities: list[float], count: int) -> np.ndarray:
    return rng.choice(len(probabilities), size=count, p=probabilities)


def sample_features(
    rng: np.random.Generator,
    labels: np.ndarray,
    *,
    semantic_flip: bool = False,
) -> np.ndarray:
    centers = CENTERS[labels].copy()
    if semantic_flip:
        flipped = centers.copy()
        flipped[:, 0] = 1.0 - flipped[:, 0]
        centers = np.where(labels[:, None] == 4, centers, flipped)
    return centers + rng.normal(0.0, 0.035, size=(len(labels), 2))


def evaluate_selected(target_labels: np.ndarray, selected: np.ndarray) -> dict[str, float]:
    oracle = np.argmax(TARGET_CLASS_ADVANTAGES[target_labels], axis=1)
    utility = TARGET_CLASS_ADVANTAGES[target_labels, selected]
    private = target_labels == 4
    return {
        "accuracy": float(np.mean(selected == oracle)),
        "mean_utility": float(np.mean(utility)),
        "negative_rate": float(np.mean(utility < -1e-12)),
        "private_none_rate": float(np.mean(selected[private] == NONE_INDEX)) if np.any(private) else 1.0,
        "private_negative_rate": float(np.mean(utility[private] < -1e-12)) if np.any(private) else 0.0,
    }


def nearest_neighbor_policy(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_labels: np.ndarray,
) -> np.ndarray:
    cost = pairwise_squared_euclidean(source_features, target_features)
    nearest = np.argmin(cost, axis=0)
    return np.argmax(SOURCE_CLASS_ADVANTAGES[source_labels[nearest]], axis=1)


def transported_policy(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_labels: np.ndarray,
    *,
    partial: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cost = pairwise_squared_euclidean(source_features, target_features)
    plan = (
        unbalanced_partial_plan(cost, reg=0.10, reg_m=0.25, max_match_cost=0.16)
        if partial
        else balanced_plan(cost, reg=0.10)
    )
    response = transport_responses(plan, SOURCE_CLASS_ADVANTAGES[source_labels])
    selected, _lcb = conservative_actions(
        response,
        action_costs=np.zeros(len(ACTION_NAMES)),
        coverage_threshold=0.30,
        z_value=0.25,
        none_index=NONE_INDEX,
    )
    return selected, response.coverage


def run_scenario(seed: int, name: str) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    if name == "matched_known_support":
        source_probs = [0.25, 0.25, 0.25, 0.25]
        target_probs = [0.25, 0.25, 0.25, 0.25, 0.00]
        semantic_flip = False
    elif name == "private_25pct":
        source_probs = [0.25, 0.25, 0.25, 0.25]
        target_probs = [0.20, 0.20, 0.20, 0.15, 0.25]
        semantic_flip = False
    elif name == "source_skew_loop_rare":
        source_probs = [0.05, 0.35, 0.30, 0.30]
        target_probs = [0.35, 0.20, 0.20, 0.10, 0.15]
        semantic_flip = False
    elif name == "semantic_conflict":
        source_probs = [0.25, 0.25, 0.25, 0.25]
        target_probs = [0.25, 0.25, 0.20, 0.10, 0.20]
        semantic_flip = True
    else:
        raise ValueError(name)

    source_labels = sample_labels(rng, source_probs, 96)
    target_labels = sample_labels(rng, target_probs, 96)
    source = sample_features(rng, source_labels)
    target = sample_features(rng, target_labels, semantic_flip=semantic_flip)

    source_mean = SOURCE_CLASS_ADVANTAGES[source_labels].mean(axis=0)
    best_fixed_action = int(np.argmax(source_mean))
    best_fixed = np.full(len(target_labels), best_fixed_action, dtype=np.int64)
    bundle_fixed = np.full(len(target_labels), ACTION_NAMES.index("bundle_conservative"), dtype=np.int64)
    knn = nearest_neighbor_policy(source, target, source_labels)
    balanced, balanced_coverage = transported_policy(source, target, source_labels, partial=False)
    partial, partial_coverage = transported_policy(source, target, source_labels, partial=True)

    rows = {
        "best_fixed_source_mean": evaluate_selected(target_labels, best_fixed),
        "fixed_bundle": evaluate_selected(target_labels, bundle_fixed),
        "knn_response": evaluate_selected(target_labels, knn),
        "balanced_ot_lcb": evaluate_selected(target_labels, balanced),
        "partial_ot_lcb": evaluate_selected(target_labels, partial),
    }
    rows["balanced_ot_lcb"]["mean_coverage"] = float(np.mean(balanced_coverage))
    rows["partial_ot_lcb"]["mean_coverage"] = float(np.mean(partial_coverage))
    rows["partial_ot_lcb"]["private_coverage"] = (
        float(np.mean(partial_coverage[target_labels == 4])) if np.any(target_labels == 4) else 0.0
    )
    return rows


def aggregate(rows: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    methods = rows[0].keys()
    return {
        method: {
            metric: float(np.mean([row[method].get(metric, 0.0) for row in rows]))
            for metric in sorted({metric for row in rows for metric in row[method]})
        }
        for method in methods
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()

    scenario_names = (
        "matched_known_support",
        "private_25pct",
        "source_skew_loop_rare",
        "semantic_conflict",
    )
    scenarios = {
        name: aggregate([run_scenario(seed, name) for seed in range(args.seeds)])
        for name in scenario_names
    }
    assertions = {
        "partial_beats_best_fixed_under_private_shift": (
            scenarios["private_25pct"]["partial_ot_lcb"]["mean_utility"]
            >= scenarios["private_25pct"]["best_fixed_source_mean"]["mean_utility"] + 0.10
        ),
        "partial_private_none_ge_90pct": (
            scenarios["private_25pct"]["partial_ot_lcb"]["private_none_rate"] >= 0.90
        ),
        "balanced_private_negative_ge_20pct": (
            scenarios["private_25pct"]["balanced_ot_lcb"]["private_negative_rate"] >= 0.20
        ),
        "partial_beats_fixed_bundle_when_source_skewed": (
            scenarios["source_skew_loop_rare"]["partial_ot_lcb"]["mean_utility"]
            >= scenarios["source_skew_loop_rare"]["fixed_bundle"]["mean_utility"] + 0.015
        ),
    }
    report = {
        "experiment_id": "L4.3",
        "purpose": "mechanism_variant_transport_stress",
        "action_names": ACTION_NAMES,
        "class_names": CLASS_NAMES,
        "seed_count": args.seeds,
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
