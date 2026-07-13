#!/usr/bin/env python3
"""Synthetic PK for structured harness-action mechanism variants.

This is a design probe, not a paper result.  It stress-tests whether candidate
mechanisms have distinct failure modes before running expensive ALFWorld/Qwen
branches.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from traceh_core.mechanisms import (
    MechanismVariant,
    PrefixContext,
    propose_mechanism_action,
)


ACTION_COSTS = {
    MechanismVariant.NATURAL_REPLAN: 0.04,
    MechanismVariant.ANTI_LOOP_RETRY: 0.02,
    MechanismVariant.PRECONDITION_CHECK: 0.02,
    MechanismVariant.SUBGOAL_LEDGER: 0.03,
    MechanismVariant.BUNDLE_CONSERVATIVE: 0.04,
}


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    scenario: str
    context: PrefixContext
    gold_command: str | None


def _shuffle(rng: random.Random, items: tuple[str, ...]) -> tuple[str, ...]:
    values = list(items)
    rng.shuffle(values)
    return tuple(values)


def build_cases(seed: int) -> list[SyntheticCase]:
    rng = random.Random(seed)
    cases: list[SyntheticCase] = []
    objects = ("apple", "mug", "pen", "book", "egg")
    places = ("table", "sofa", "desk", "counter")

    for index, obj in enumerate(objects):
        cases.append(
            SyntheticCase(
                case_id=f"loop_escape_{seed}_{index}",
                scenario="loop_escape",
                context=PrefixContext(
                    task=f"Find the {obj}.",
                    observation="Hallway.",
                    history=(
                        ("go to kitchen", "Kitchen."),
                        ("go to hallway", "Hallway."),
                        ("go to kitchen", "Kitchen."),
                        ("go to hallway", "Hallway."),
                    ),
                    admissible_commands=_shuffle(
                        rng,
                        ("look", "go to kitchen", "open drawer 1", "inventory"),
                    ),
                ),
                gold_command="open drawer 1",
            )
        )

    for index, obj in enumerate(objects):
        place = places[index % len(places)]
        cases.append(
            SyntheticCase(
                case_id=f"missing_take_{seed}_{index}",
                scenario="missing_precondition_take",
                context=PrefixContext(
                    task=f"Put the {obj} on the {place}.",
                    observation=f"Kitchen.\nYou see {obj} 1 on counter 1.",
                    history=(("go to counter 1", f"Kitchen.\nYou see {obj} 1 on counter 1."),),
                    admissible_commands=_shuffle(
                        rng,
                        (
                            "look",
                            f"put {obj} 1 on {place} 1",
                            f"take {obj} 1 from counter 1",
                            f"go to {place} 1",
                        ),
                    ),
                ),
                gold_command=f"take {obj} 1 from counter 1",
            )
        )

    for index, obj in enumerate(("egg", "mug", "apple", "book")):
        cases.append(
            SyntheticCase(
                case_id=f"missing_open_{seed}_{index}",
                scenario="missing_precondition_open",
                context=PrefixContext(
                    task=f"Heat the {obj} and put it on the counter.",
                    observation=f"Kitchen.\nYou are carrying {obj} 1. Microwave 1 is closed.",
                    history=((f"take {obj} 1 from table 1", f"You are carrying {obj} 1."),),
                    admissible_commands=_shuffle(
                        rng,
                        (
                            "look",
                            "open microwave 1",
                            f"put {obj} 1 in microwave 1",
                            "go to counter 1",
                        ),
                    ),
                ),
                gold_command="open microwave 1",
            )
        )

    for index, obj in enumerate(("pen", "book", "mug", "apple")):
        cases.append(
            SyntheticCase(
                case_id=f"two_object_{seed}_{index}",
                scenario="two_object_memory",
                context=PrefixContext(
                    task=f"Put two {obj}s on the desk.",
                    observation=f"Office.\nYou see {obj} 2 on shelf 1.",
                    history=((f"put {obj} 1 on desk 1", f"Office.\n{obj} 1 is on desk 1."),),
                    admissible_commands=_shuffle(
                        rng,
                        (
                            "look",
                            f"take {obj} 1 from desk 1",
                            f"take {obj} 2 from shelf 1",
                            "go to desk 1",
                        ),
                    ),
                ),
                gold_command=f"take {obj} 2 from shelf 1",
            )
        )

    for index in range(6):
        cases.append(
            SyntheticCase(
                case_id=f"private_{seed}_{index}",
                scenario="private_no_support",
                context=PrefixContext(
                    task="Solve the task.",
                    observation=f"Room {index}.\nThere is no visible task object.",
                    history=(),
                    admissible_commands=_shuffle(rng, ("look", "inventory", "wait")),
                ),
                gold_command=None,
            )
        )
    return cases


def score_case(case: SyntheticCase, variant: MechanismVariant) -> dict[str, object]:
    proposal = propose_mechanism_action(case.context, variant)
    cost = ACTION_COSTS[variant]
    if case.gold_command is None:
        correct = not proposal.intervene
        utility = 0.0 if correct else -0.40 - cost
        negative = proposal.intervene
    elif not proposal.intervene:
        correct = False
        utility = 0.0
        negative = False
    elif proposal.selected_command == case.gold_command:
        correct = True
        utility = 1.0 - cost
        negative = False
    else:
        correct = False
        utility = -0.30 - cost
        negative = True
    return {
        "case_id": case.case_id,
        "scenario": case.scenario,
        "variant": variant.value,
        "gold_command": case.gold_command,
        "intervene": proposal.intervene,
        "selected_command": proposal.selected_command,
        "confidence": proposal.confidence,
        "correct": correct,
        "negative": negative,
        "utility": utility,
        "reasons": list(proposal.reasons),
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    by_variant: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_pair: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)
        by_pair[(str(row["variant"]), str(row["scenario"]))].append(row)

    variants = {}
    for variant, values in by_variant.items():
        private = [row for row in values if row["scenario"] == "private_no_support"]
        actionable = [row for row in values if row["scenario"] != "private_no_support"]
        variants[variant] = {
            "case_count": len(values),
            "mean_utility": sum(float(row["utility"]) for row in values) / len(values),
            "accuracy": sum(bool(row["correct"]) for row in values) / len(values),
            "actionable_success_rate": sum(bool(row["correct"]) for row in actionable) / len(actionable),
            "intervention_rate": sum(bool(row["intervene"]) for row in values) / len(values),
            "negative_intervention_rate": sum(bool(row["negative"]) for row in values) / len(values),
            "private_abstain_rate": sum(not bool(row["intervene"]) for row in private) / len(private),
        }

    scenarios = {}
    for (variant, scenario), values in by_pair.items():
        scenarios.setdefault(scenario, {})[variant] = {
            "mean_utility": sum(float(row["utility"]) for row in values) / len(values),
            "accuracy": sum(bool(row["correct"]) for row in values) / len(values),
            "negative_intervention_rate": sum(bool(row["negative"]) for row in values) / len(values),
        }

    ranking = sorted(
        variants,
        key=lambda item: (-float(variants[item]["mean_utility"]), item),
    )
    return {
        "variants": variants,
        "scenarios": scenarios,
        "ranking_by_mean_utility": ranking,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    variants = (
        MechanismVariant.NATURAL_REPLAN,
        MechanismVariant.ANTI_LOOP_RETRY,
        MechanismVariant.PRECONDITION_CHECK,
        MechanismVariant.SUBGOAL_LEDGER,
        MechanismVariant.BUNDLE_CONSERVATIVE,
    )
    cases = [case for seed in range(args.seeds) for case in build_cases(seed)]
    rows = [score_case(case, variant) for case in cases for variant in variants]
    summary = aggregate(rows)
    variant_summary = summary["variants"]
    assertions = {
        "bundle_ranked_first": summary["ranking_by_mean_utility"][0] == "bundle_conservative",
        "bundle_private_abstain_ge_95pct": (
            variant_summary["bundle_conservative"]["private_abstain_rate"] >= 0.95
        ),
        "anti_loop_wins_loop_escape": (
            summary["scenarios"]["loop_escape"]["anti_loop_retry"]["accuracy"] >= 0.95
        ),
        "precondition_wins_missing_take": (
            summary["scenarios"]["missing_precondition_take"]["precondition_check"]["accuracy"] >= 0.95
        ),
        "ledger_wins_two_object": (
            summary["scenarios"]["two_object_memory"]["subgoal_ledger"]["accuracy"] >= 0.95
        ),
        "natural_replan_has_high_negative_rate": (
            variant_summary["natural_replan"]["negative_intervention_rate"] >= 0.20
        ),
    }
    report = {
        "experiment_id": "L4.2",
        "purpose": "structured_action_mechanism_design_probe",
        "seed_count": args.seeds,
        "case_count": len(cases),
        "row_count": len(rows),
        "summary": summary,
        "assertions": assertions,
        "ok": all(assertions.values()),
        "sample_rows": rows[:10],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
