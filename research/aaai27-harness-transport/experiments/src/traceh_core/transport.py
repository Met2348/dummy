"""Response transport and conservative action compilation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ot


def pairwise_squared_euclidean(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source and target must be 2D with equal feature dimension")
    cost = ((source[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
    scale = float(cost.max())
    return cost / scale if scale > 0 else cost


def balanced_plan(cost: np.ndarray, reg: float = 0.05) -> np.ndarray:
    cost = np.asarray(cost, dtype=np.float64)
    a = np.full(cost.shape[0], 1.0 / cost.shape[0])
    b = np.full(cost.shape[1], 1.0 / cost.shape[1])
    return np.asarray(
        ot.sinkhorn(
            a,
            b,
            cost,
            reg=reg,
            method="sinkhorn_log",
            numItermax=5000,
            stopThr=1e-9,
        )
    )


def unbalanced_partial_plan(
    cost: np.ndarray,
    reg: float = 0.05,
    reg_m: float = 0.1,
    max_match_cost: float | None = 0.35,
) -> np.ndarray:
    cost = np.asarray(cost, dtype=np.float64)
    a = np.full(cost.shape[0], 1.0 / cost.shape[0])
    b = np.full(cost.shape[1], 1.0 / cost.shape[1])
    plan = np.asarray(
        ot.unbalanced.sinkhorn_unbalanced(
            a,
            b,
            cost,
            reg=reg,
            reg_m=reg_m,
            method="sinkhorn_stabilized",
            numItermax=5000,
            stopThr=1e-9,
        )
    )
    if max_match_cost is not None:
        plan = np.where(cost <= max_match_cost, plan, 0.0)
    return plan


@dataclass(frozen=True)
class TransportedResponse:
    mean: np.ndarray
    standard_error: np.ndarray
    coverage: np.ndarray
    effective_support: np.ndarray


def transport_responses(plan: np.ndarray, source_advantages: np.ndarray) -> TransportedResponse:
    plan = np.asarray(plan, dtype=np.float64)
    advantages = np.asarray(source_advantages, dtype=np.float64)
    if plan.ndim != 2 or advantages.ndim != 2 or plan.shape[0] != advantages.shape[0]:
        raise ValueError("plan/source advantage dimensions disagree")

    n_target = plan.shape[1]
    n_actions = advantages.shape[1]
    means = np.zeros((n_target, n_actions), dtype=np.float64)
    errors = np.full((n_target, n_actions), np.inf, dtype=np.float64)
    coverage = np.zeros(n_target, dtype=np.float64)
    effective = np.zeros(n_target, dtype=np.float64)
    expected_target_mass = 1.0 / n_target

    for target_index in range(n_target):
        column = plan[:, target_index]
        mass = float(column.sum())
        coverage[target_index] = np.clip(mass / expected_target_mass, 0.0, 1.0)
        if mass <= 1e-15:
            continue
        weights = column / mass
        effective[target_index] = 1.0 / float(np.square(weights).sum())
        means[target_index] = weights @ advantages
        centered = advantages - means[target_index]
        variance = weights @ np.square(centered)
        errors[target_index] = np.sqrt(variance / max(effective[target_index], 1.0))

    return TransportedResponse(means, errors, coverage, effective)


def conservative_actions(
    response: TransportedResponse,
    action_costs: np.ndarray,
    coverage_threshold: float = 0.5,
    z_value: float = 1.0,
    none_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    costs = np.asarray(action_costs, dtype=np.float64)
    if costs.shape != (response.mean.shape[1],):
        raise ValueError("action costs must have one value per action")
    lcb = response.mean - z_value * response.standard_error - costs[None, :]
    lcb[:, none_index] = 0.0
    selected = np.full(response.mean.shape[0], none_index, dtype=np.int64)
    for index in range(response.mean.shape[0]):
        if response.coverage[index] < coverage_threshold:
            continue
        candidate = int(np.argmax(lcb[index]))
        if candidate != none_index and lcb[index, candidate] > 0:
            selected[index] = candidate
    return selected, lcb
