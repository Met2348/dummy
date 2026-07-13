from __future__ import annotations

import numpy as np

from traceh_core.transport import (
    balanced_plan,
    conservative_actions,
    pairwise_squared_euclidean,
    transport_responses,
    unbalanced_partial_plan,
)


def test_balanced_plan_forces_target_mass() -> None:
    source = np.array([[0.0], [1.0]])
    target = np.array([[0.0], [1.0], [10.0]])
    cost = pairwise_squared_euclidean(source, target)
    plan = balanced_plan(cost)
    response = transport_responses(plan, np.array([[0.0, 1.0], [0.0, -1.0]]))
    assert np.all(response.coverage > 0.99)


def test_partial_plan_leaves_private_target_unmatched() -> None:
    source = np.array([[0.0], [1.0]])
    target = np.array([[0.0], [1.0], [10.0]])
    cost = pairwise_squared_euclidean(source, target)
    plan = unbalanced_partial_plan(cost, max_match_cost=0.2)
    response = transport_responses(plan, np.array([[0.0, 1.0], [0.0, -1.0]]))
    assert response.coverage[2] == 0.0
    selected, _ = conservative_actions(
        response,
        action_costs=np.array([0.0, 0.01]),
        coverage_threshold=0.5,
    )
    assert selected[2] == 0


def test_supported_positive_action_can_be_selected() -> None:
    source = np.array([[0.0], [0.1], [1.0]])
    target = np.array([[0.05]])
    cost = pairwise_squared_euclidean(source, target)
    plan = unbalanced_partial_plan(cost, reg=0.1, reg_m=1.0, max_match_cost=0.2)
    advantages = np.array([[0.0, 0.8], [0.0, 0.7], [0.0, -0.5]])
    response = transport_responses(plan, advantages)
    selected, lcb = conservative_actions(
        response,
        action_costs=np.array([0.0, 0.05]),
        coverage_threshold=0.2,
        z_value=0.5,
    )
    assert response.coverage[0] >= 0.2
    assert lcb[0, 1] > 0
    assert selected[0] == 1

