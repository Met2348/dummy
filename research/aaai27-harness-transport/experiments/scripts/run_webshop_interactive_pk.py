#!/usr/bin/env python3
"""Run an interactive WebShop-small text-environment PK benchmark.

The benchmark drives the local WebShop ``WebAgentTextEnv`` through real actions:
search, optional pagination, product click, option selection, and Buy Now.  It
uses deterministic non-LLM policies so the run is cheap enough for local method
iteration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "color",
    "dollars",
    "find",
    "for",
    "from",
    "get",
    "i",
    "in",
    "instruction",
    "is",
    "it",
    "looking",
    "lower",
    "m",
    "me",
    "need",
    "of",
    "on",
    "or",
    "please",
    "price",
    "show",
    "size",
    "than",
    "that",
    "the",
    "this",
    "to",
    "want",
    "with",
}

METHOD_SPECS = [
    ("random_top10_full_search", "random top-10 after full search", "baseline"),
    ("bm25_full_instruction", "BM25 full instruction", "baseline"),
    ("bm25_attribute_query", "BM25 attribute query", "baseline"),
    ("bm25_core_query", "BM25 core query", "baseline"),
    ("title_overlap_full_search", "title overlap after full search", "baseline"),
    ("description_overlap_full_search", "description overlap after full search", "baseline"),
    ("all_text_overlap_full_search", "all-text overlap after full search", "baseline"),
    ("attribute_overlap_attr_search", "attribute overlap after attr search", "baseline"),
    ("rarest_attribute_anchor", "rarest-attribute anchor", "baseline"),
    ("traceh_constraint_query_ledger", "TRACE-H constraint+query ledger", "ours"),
]


@dataclass(frozen=True)
class ProductFields:
    asin: str
    index: int
    title_tokens: frozenset[str]
    description_tokens: frozenset[str]
    category_tokens: frozenset[str]
    attribute_tokens: frozenset[str]
    all_tokens: frozenset[str]
    attribute_phrases: frozenset[str]
    option_values: frozenset[str]
    price: float


def tokenize(text: object) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if token and token not in STOPWORDS
    )


def token_set(text: object) -> frozenset[str]:
    return frozenset(tokenize(text))


def phrase_token_set(values: list[object] | tuple[object, ...]) -> frozenset[str]:
    tokens: list[str] = []
    for value in values:
        tokens.extend(tokenize(value))
    return frozenset(tokens)


def idf_scores(documents: list[frozenset[str]]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(document)
    total = len(documents)
    return {
        token: math.log((total + 1.0) / (frequency + 1.0)) + 1.0
        for token, frequency in document_frequency.items()
    }


def phrase_idf_scores(products: dict[str, ProductFields]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for product in products.values():
        document_frequency.update(product.attribute_phrases)
    total = len(products)
    return {
        phrase: math.log((total + 1.0) / (frequency + 1.0)) + 1.0
        for phrase, frequency in document_frequency.items()
    }


def sum_idf(tokens: set[str] | frozenset[str], idf: dict[str, float]) -> float:
    return sum(idf.get(token, 0.0) for token in tokens)


def clean_instruction(instruction: str) -> str:
    text = instruction.lower()
    text = re.sub(r"\band price lower than [0-9.]+ dollars\b", " ", text)
    text = re.sub(r"\bprice lower than [0-9.]+ dollars\b", " ", text)
    return " ".join(tokenize(text))


def build_products(env: Any) -> dict[str, ProductFields]:
    products: dict[str, ProductFields] = {}
    for index, product in enumerate(env.server.all_products):
        asin = product["asin"]
        title_tokens = token_set(product.get("Title", ""))
        description_text = f"{product.get('Description') or ''} {' '.join(product.get('BulletPoints') or [])}"
        description_tokens = token_set(description_text)
        category_text = f"{product.get('category') or ''} {product.get('query') or ''} {product.get('product_category') or ''}"
        category_tokens = token_set(category_text)
        attributes = product.get("Attributes") or []
        attribute_tokens = phrase_token_set(attributes)
        option_values = []
        for values in (product.get("options") or {}).values():
            option_values.extend(str(value).lower() for value in values)
        all_tokens = frozenset(title_tokens | description_tokens | category_tokens | attribute_tokens)
        products[asin] = ProductFields(
            asin=asin,
            index=index,
            title_tokens=title_tokens,
            description_tokens=description_tokens,
            category_tokens=category_tokens,
            attribute_tokens=attribute_tokens,
            all_tokens=all_tokens,
            attribute_phrases=frozenset(str(attribute).lower() for attribute in attributes),
            option_values=frozenset(option_values),
            price=float(env.server.product_prices.get(asin, 1_000_000.0)),
        )
    return products


def goal_views(goal: dict[str, Any]) -> dict[str, Any]:
    instruction = str(goal["instruction_text"])
    attributes = tuple(str(attribute) for attribute in (goal.get("attributes") or []))
    goal_options_raw = goal.get("goal_options") or {}
    if isinstance(goal_options_raw, dict):
        goal_option_values = tuple(str(value).lower() for value in goal_options_raw.values())
    else:
        goal_option_values = tuple(str(value).lower() for value in goal_options_raw)
    instruction_tokens = token_set(instruction)
    required_tokens = phrase_token_set(attributes)
    option_tokens = phrase_token_set(goal_option_values)
    price_tokens = {"price", "lower", "than", "dollars"}
    query_tokens = frozenset(instruction_tokens - required_tokens - option_tokens - price_tokens)
    return {
        "instruction": instruction,
        "attributes": attributes,
        "goal_option_values": goal_option_values,
        "instruction_tokens": instruction_tokens,
        "required_tokens": required_tokens,
        "query_tokens": query_tokens,
        "attribute_query": " ".join(token for value in attributes for token in tokenize(value)) or clean_instruction(instruction),
        "core_query": " ".join(sorted(query_tokens)) or clean_instruction(instruction),
        "full_query": clean_instruction(instruction),
    }


def query_for_method(method_key: str, view: dict[str, Any]) -> str:
    if method_key in {
        "bm25_attribute_query",
        "attribute_overlap_attr_search",
        "rarest_attribute_anchor",
    }:
        return str(view["attribute_query"])
    if method_key == "bm25_core_query":
        return str(view["core_query"])
    if method_key == "traceh_constraint_query_ledger":
        query = f"{view['core_query']} {view['attribute_query']}".strip()
        return query or str(view["full_query"])
    return str(view["full_query"])


def search_candidates(env: Any, query: str, *, k: int = 50) -> list[str]:
    safe_query = query.strip() or "product"
    hits = env.server.search_engine.search(safe_query, k=k)
    asins: list[str] = []
    for hit in hits:
        document = env.server.search_engine.doc(hit.docid)
        asin = json.loads(document.raw())["id"]
        if asin in env.server.product_item_dict:
            asins.append(asin)
    return asins


def candidate_score(
    method_key: str,
    asin: str,
    view: dict[str, Any],
    products: dict[str, ProductFields],
    *,
    idf: dict[str, float],
    phrase_idf: dict[str, float],
    goal: dict[str, Any],
    rank_position: int,
) -> tuple[float, ...]:
    product = products[asin]
    instruction_tokens = view["instruction_tokens"]
    required_tokens = view["required_tokens"]
    query_tokens = view["query_tokens"]
    if method_key == "random_top10_full_search":
        return (-float(rank_position),)
    if method_key in {"bm25_full_instruction", "bm25_attribute_query", "bm25_core_query"}:
        return (-float(rank_position),)
    if method_key == "title_overlap_full_search":
        overlap = instruction_tokens & product.title_tokens
        return (float(len(overlap)), sum_idf(overlap, idf), -float(rank_position))
    if method_key == "description_overlap_full_search":
        overlap = instruction_tokens & product.description_tokens
        return (float(len(overlap)), sum_idf(overlap, idf), -float(rank_position))
    if method_key == "all_text_overlap_full_search":
        overlap = instruction_tokens & product.all_tokens
        return (float(len(overlap)), sum_idf(overlap, idf), -float(rank_position))
    if method_key == "attribute_overlap_attr_search":
        overlap = required_tokens & product.attribute_tokens
        return (float(len(overlap)), sum_idf(overlap, idf), -float(rank_position))
    if method_key == "rarest_attribute_anchor":
        overlap = required_tokens & product.attribute_tokens
        return (sum_idf(overlap, idf), float(len(overlap)), -float(rank_position))
    if method_key == "traceh_constraint_query_ledger":
        hard_attribute_overlap = required_tokens & product.attribute_tokens
        hard_anywhere_overlap = required_tokens & product.all_tokens
        query_category_overlap = query_tokens & product.category_tokens
        query_title_overlap = query_tokens & product.title_tokens
        query_all_overlap = query_tokens & product.all_tokens
        option_matches = sum(1.0 for value in view["goal_option_values"] if value in product.option_values)
        option_total = max(1.0, float(len(view["goal_option_values"])))
        option_rate = option_matches / option_total
        price_ok = 1.0 if product.price <= float(goal.get("price_upper", 1_000_000.0)) else 0.0
        exact_attribute_phrase_score = sum(
            phrase_idf.get(attribute.lower(), 0.0)
            for attribute in view["attributes"]
            if attribute.lower() in product.attribute_phrases
        )
        return (
            float(len(hard_attribute_overlap)),
            exact_attribute_phrase_score,
            option_rate,
            price_ok,
            sum_idf(query_category_overlap, idf),
            sum_idf(query_title_overlap, idf),
            sum_idf(query_all_overlap, idf),
            float(len(hard_anywhere_overlap)),
            -float(rank_position),
            -float(len(product.all_tokens)),
        )
    raise KeyError(method_key)


def choose_candidate(
    method_key: str,
    candidates: list[str],
    view: dict[str, Any],
    products: dict[str, ProductFields],
    *,
    idf: dict[str, float],
    phrase_idf: dict[str, float],
    goal: dict[str, Any],
    rng: random.Random,
) -> str | None:
    if not candidates:
        return None
    if method_key == "random_top10_full_search":
        return rng.choice(candidates[: min(10, len(candidates))])
    scored = [
        (
            candidate_score(
                method_key,
                asin,
                view,
                products,
                idf=idf,
                phrase_idf=phrase_idf,
                goal=goal,
                rank_position=index,
            ),
            -products[asin].index,
            asin,
        )
        for index, asin in enumerate(candidates, start=1)
    ]
    scored.sort(reverse=True)
    return scored[0][2]


def stable_method_offset(method_key: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(method_key))


def click_product(env: Any, asin: str) -> tuple[str | None, int]:
    target = asin.lower()
    steps = 0
    fallback: str | None = None
    for _page in range(5):
        actions = env.get_available_actions()["clickables"]
        product_links = [action for action in actions if re.fullmatch(r"b[0-9a-z]{9}", action)]
        if fallback is None and product_links:
            fallback = product_links[0]
        if target in actions:
            env.step(f"click[{target}]")
            return asin, steps + 1
        if "next >" not in actions:
            break
        env.step("click[next >]")
        steps += 1
    if fallback is not None:
        env.step(f"click[{fallback}]")
        return fallback.upper(), steps + 1
    return None, steps


def select_options_and_buy(env: Any, goal_option_values: tuple[str, ...]) -> tuple[float, bool, int, dict[str, Any]]:
    steps = 0
    for option_value in goal_option_values:
        actions = env.get_available_actions()["clickables"]
        option = option_value.lower()
        if option in actions:
            env.step(f"click[{option}]")
            steps += 1
            continue
        tokenized_option = set(tokenize(option))
        best = None
        best_score = 0
        for action in actions:
            action_tokens = set(tokenize(action))
            score = len(tokenized_option & action_tokens)
            if score > best_score:
                best = action
                best_score = score
        if best is not None and best_score > 0:
            env.step(f"click[{best}]")
            steps += 1
    actions = env.get_available_actions()["clickables"]
    if "buy now" not in actions:
        return 0.0, False, steps, {"error": "buy now not available", "clickables": actions}
    _obs, reward, done, _info = env.step("click[buy now]")
    steps += 1
    return float(reward), bool(done), steps, {}


def run_episode(
    env: Any,
    task_index: int,
    method_key: str,
    products: dict[str, ProductFields],
    *,
    idf: dict[str, float],
    phrase_idf: dict[str, float],
    seed: int,
) -> dict[str, Any]:
    env.reset(session=task_index)
    session_id = str(task_index)
    goal = env.server.user_sessions[session_id]["goal"]
    view = goal_views(goal)
    query = query_for_method(method_key, view)
    candidates = search_candidates(env, query, k=50)
    rng = random.Random(seed + task_index * 1009 + stable_method_offset(method_key))
    selected_asin = choose_candidate(
        method_key,
        candidates,
        view,
        products,
        idf=idf,
        phrase_idf=phrase_idf,
        goal=goal,
        rng=rng,
    )
    env.step(f"search[{query}]")
    action_count = 1
    clicked_asin = None
    if selected_asin is not None:
        clicked_asin, click_steps = click_product(env, selected_asin)
        action_count += click_steps
    if clicked_asin is None:
        return {
            "reward": 0.0,
            "done": False,
            "selected_asin": selected_asin,
            "clicked_asin": None,
            "target_asin": goal["asin"],
            "exact_purchase": False,
            "query": query,
            "candidate_count": len(candidates),
            "target_in_candidates": goal["asin"] in candidates,
            "action_count": action_count,
            "error": "no product clicked",
        }
    reward, done, option_steps, error = select_options_and_buy(env, view["goal_option_values"])
    action_count += option_steps
    old_session = env.server.user_sessions.get(session_id, {})
    return {
        "reward": reward,
        "done": done,
        "selected_asin": selected_asin,
        "clicked_asin": clicked_asin,
        "target_asin": goal["asin"],
        "exact_purchase": clicked_asin == goal["asin"],
        "query": query,
        "candidate_count": len(candidates),
        "target_in_candidates": goal["asin"] in candidates,
        "target_candidate_rank": candidates.index(goal["asin"]) + 1 if goal["asin"] in candidates else None,
        "action_count": action_count,
        "verbose_info": old_session.get("verbose_info"),
        **error,
    }


def sign_test(diffs: list[float], eps: float = 1e-12) -> dict[str, Any]:
    wins = sum(diff > eps for diff in diffs)
    losses = sum(diff < -eps for diff in diffs)
    ties = len(diffs) - wins - losses
    trials = wins + losses
    if trials == 0:
        p_value = 1.0
    else:
        smaller = min(wins, losses)
        p_value = 2.0 * sum(math.comb(trials, index) for index in range(smaller + 1)) / (2**trials)
        p_value = min(1.0, p_value)
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "trials": trials,
        "p_value": p_value,
        "significant_0_05": p_value < 0.05,
    }


def bootstrap_ci(values: list[float], *, seed: int, samples: int = 4000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if all(math.isclose(value, values[0], rel_tol=1e-12, abs_tol=1e-12) for value in values):
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _index in range(len(values))]
        means.append(mean(draw))
    means.sort()
    return float(means[int(0.025 * (samples - 1))]), float(means[int(0.975 * (samples - 1))])


def summarize(rows: list[dict[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    for index, (method_key, method_label, role) in enumerate(METHOD_SPECS):
        reward_values = [float(row["methods"][method_key]["reward"]) for row in rows]
        exact_values = [float(row["methods"][method_key]["exact_purchase"]) for row in rows]
        records.append(
            {
                "method_key": method_key,
                "method_label": method_label,
                "role": role,
                "unit_count": len(rows),
                "primary_metric": "mean_webshop_reward",
                "primary_value": float(mean(reward_values)),
                "primary_ci95": list(bootstrap_ci(reward_values, seed=seed + index * 101 + 11)),
                "secondary_metric": "exact_purchase_rate",
                "secondary_value": float(mean(exact_values)),
                "secondary_ci95": list(bootstrap_ci(exact_values, seed=seed + index * 101 + 23)),
                "target_retrieval_rate": float(
                    mean(float(row["methods"][method_key]["target_in_candidates"]) for row in rows)
                ),
                "mean_action_count": float(mean(float(row["methods"][method_key]["action_count"]) for row in rows)),
            }
        )

    for value_name, rank_name in (("primary_value", "primary_rank"), ("secondary_value", "secondary_rank")):
        ranked = sorted(records, key=lambda item: (-float(item[value_name]), item["method_label"]))
        previous_value: float | None = None
        previous_rank = 0
        for rank, record in enumerate(ranked, start=1):
            value = float(record[value_name])
            if previous_value is None or not math.isclose(value, previous_value, rel_tol=1e-12, abs_tol=1e-12):
                previous_value = value
                previous_rank = rank
            record[rank_name] = previous_rank

    comparisons = []
    ours_key = "traceh_constraint_query_ledger"
    for method_key, method_label, role in METHOD_SPECS:
        if method_key == ours_key:
            continue
        reward_diffs = [
            float(row["methods"][ours_key]["reward"]) - float(row["methods"][method_key]["reward"])
            for row in rows
        ]
        exact_diffs = [
            float(row["methods"][ours_key]["exact_purchase"]) - float(row["methods"][method_key]["exact_purchase"])
            for row in rows
        ]
        comparisons.append(
            {
                "ours_key": ours_key,
                "baseline_key": method_key,
                "baseline_label": method_label,
                "baseline_role": role,
                "primary_mean_delta": float(mean(reward_diffs)),
                "secondary_mean_delta": float(mean(exact_diffs)),
                "primary_sign_test": sign_test(reward_diffs),
                "secondary_sign_test": sign_test(exact_diffs),
            }
        )
    return records, comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webshop-root", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-goals", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    os.environ.setdefault("OPENAI_API_KEY", "sk-local-dummy")
    if "JAVA_HOME" not in os.environ and Path("/usr/lib/jvm/java-21-openjdk-amd64").exists():
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
    sys.path.insert(0, str(args.webshop_root))

    from web_agent_site.envs import WebAgentTextEnv  # noqa: PLC0415

    env = WebAgentTextEnv(
        observation_mode="text",
        file_path=str(args.items),
        attr_path=str(args.annotations),
        num_products=1000,
        human_goals=0,
        limit_goals=args.max_goals,
        seed=args.seed,
    )
    products = build_products(env)
    idf = idf_scores([product.all_tokens for product in products.values()])
    phrase_idf = phrase_idf_scores(products)
    goal_count = len(env.server.goals)

    rows: list[dict[str, Any]] = []
    for task_index in range(goal_count):
        if task_index > 0 and task_index % 50 == 0:
            print(f"[webshop-interactive] completed {task_index}/{goal_count} goals", file=sys.stderr)
        env.reset(session=task_index)
        goal = env.server.user_sessions[str(task_index)]["goal"]
        task_row = {
            "task_index": task_index,
            "target_asin": goal["asin"],
            "instruction": goal["instruction_text"],
            "attributes": list(goal.get("attributes") or []),
            "goal_options": goal.get("goal_options") or {},
            "price_upper": goal.get("price_upper"),
            "methods": {},
        }
        for method_key, _method_label, _role in METHOD_SPECS:
            task_row["methods"][method_key] = run_episode(
                env,
                task_index,
                method_key,
                products,
                idf=idf,
                phrase_idf=phrase_idf,
                seed=args.seed,
            )
        rows.append(task_row)

    records, comparisons = summarize(rows, seed=args.seed)
    paper_baseline_comparisons = [
        comparison for comparison in comparisons if comparison["baseline_role"] == "baseline"
    ]
    ours_record = next(record for record in records if record["method_key"] == "traceh_constraint_query_ledger")
    status = {
        "product_count": len(products),
        "goal_count": goal_count,
        "baseline_count": sum(1 for _key, _label, role in METHOD_SPECS if role == "baseline"),
        "ours_primary_rank_1": ours_record["primary_rank"] == 1,
        "ours_secondary_rank_1": ours_record["secondary_rank"] == 1,
        "ours_vs_all_baselines_primary_significant_0_05": all(
            comparison["primary_sign_test"]["significant_0_05"]
            and comparison["primary_mean_delta"] > 0.0
            for comparison in paper_baseline_comparisons
        ),
        "ours_vs_all_baselines_secondary_significant_0_05": all(
            comparison["secondary_sign_test"]["significant_0_05"]
            and comparison["secondary_mean_delta"] > 0.0
            for comparison in paper_baseline_comparisons
        ),
        "interactive_env_status": "runnable_local_text_env",
        "evidence_boundary": (
            "Interactive WebShop-small text environment over 1000 indexed products and synthetic goals; "
            "policies are deterministic non-LLM local baselines for mechanism iteration."
        ),
    }
    output = {
        "experiment_id": "L4-WEBSHOP-INTERACTIVE-PK",
        "inputs": {
            "webshop_root": str(args.webshop_root),
            "items": str(args.items),
            "annotations": str(args.annotations),
            "max_goals": args.max_goals,
            "seed": args.seed,
        },
        "method_specs": [
            {"method_key": key, "method_label": label, "role": role}
            for key, label, role in METHOD_SPECS
        ],
        "status": status,
        "records": records,
        "comparisons": comparisons,
        "rows": rows,
        "ok": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {
        "status": status,
        "records": [
            {
                "method_key": record["method_key"],
                "primary_value": record["primary_value"],
                "primary_rank": record["primary_rank"],
                "secondary_value": record["secondary_value"],
                "secondary_rank": record["secondary_rank"],
                "target_retrieval_rate": record["target_retrieval_rate"],
            }
            for record in records
        ],
        "primary_comparisons": [
            {
                "baseline_key": comparison["baseline_key"],
                "primary_mean_delta": comparison["primary_mean_delta"],
                **comparison["primary_sign_test"],
            }
            for comparison in comparisons
        ],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
