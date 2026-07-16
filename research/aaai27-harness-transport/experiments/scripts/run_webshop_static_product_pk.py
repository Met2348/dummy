#!/usr/bin/env python3
"""Run a WebShop-small static product-selection PK benchmark.

This is intentionally not the full WebShop interactive environment.  It uses
the WebShop-small real product catalog plus the official synthetic instruction
and attribute annotations as a local, reproducible projection for mechanism
iteration while the text environment is blocked by Java/Lucene dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
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
    "find",
    "for",
    "from",
    "get",
    "i",
    "in",
    "is",
    "it",
    "looking",
    "m",
    "me",
    "need",
    "of",
    "on",
    "or",
    "please",
    "show",
    "that",
    "the",
    "this",
    "to",
    "want",
    "with",
}


@dataclass(frozen=True)
class ProductFields:
    asin: str
    index: int
    title_tokens: frozenset[str]
    description_tokens: frozenset[str]
    category_tokens: frozenset[str]
    attribute_tokens: frozenset[str]
    all_tokens: frozenset[str]
    all_doc_tokens: tuple[str, ...]
    title_category_doc_tokens: tuple[str, ...]
    attribute_phrases: frozenset[str]


@dataclass(frozen=True)
class Task:
    task_id: str
    target_asin: str
    instruction: str
    instruction_attributes: tuple[str, ...]
    instruction_tokens: frozenset[str]
    required_tokens: frozenset[str]


def tokenize(text: object) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if token and token not in STOPWORDS
    )


def token_set(text: object) -> frozenset[str]:
    return frozenset(tokenize(text))


def phrase_token_set(phrases: list[object] | tuple[object, ...]) -> frozenset[str]:
    tokens: list[str] = []
    for phrase in phrases:
        tokens.extend(tokenize(phrase))
    return frozenset(tokens)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def bm25_score(
    query_tokens: frozenset[str],
    document_tokens: tuple[str, ...],
    idf: dict[str, float],
    avg_doc_length: float,
    *,
    k1: float = 1.4,
    b: float = 0.75,
) -> float:
    if not document_tokens:
        return 0.0
    counts = Counter(document_tokens)
    doc_length = len(document_tokens)
    total = 0.0
    for token in query_tokens:
        frequency = counts.get(token, 0)
        if frequency == 0:
            continue
        denominator = frequency + k1 * (1.0 - b + b * doc_length / max(avg_doc_length, 1e-9))
        total += idf.get(token, 0.0) * frequency * (k1 + 1.0) / denominator
    return total


def build_products(items: list[dict[str, Any]], annotations: dict[str, dict[str, Any]]) -> dict[str, ProductFields]:
    products: dict[str, ProductFields] = {}
    for index, item in enumerate(items):
        asin = str(item["asin"])
        attributes = annotations.get(asin, {}).get("attributes") or []
        title_tokens = token_set(item.get("name", ""))
        description_text = f"{item.get('full_description') or ''} {item.get('small_description') or ''}"
        description_tokens = token_set(description_text)
        category_text = f"{item.get('product_category') or ''} {item.get('category') or ''} {item.get('query') or ''}"
        category_tokens = token_set(category_text)
        attribute_tokens = phrase_token_set(attributes)
        all_doc_tokens = (
            tokenize(item.get("name", ""))
            + tokenize(description_text)
            + tokenize(category_text)
            + tuple(token for attribute in attributes for token in tokenize(attribute))
        )
        title_category_doc_tokens = tokenize(item.get("name", "")) + tokenize(category_text)
        all_tokens = frozenset(title_tokens | description_tokens | category_tokens | attribute_tokens)
        products[asin] = ProductFields(
            asin=asin,
            index=index,
            title_tokens=title_tokens,
            description_tokens=description_tokens,
            category_tokens=category_tokens,
            attribute_tokens=attribute_tokens,
            all_tokens=all_tokens,
            all_doc_tokens=all_doc_tokens,
            title_category_doc_tokens=title_category_doc_tokens,
            attribute_phrases=frozenset(str(attribute).lower() for attribute in attributes),
        )
    return products


def build_tasks(
    products: dict[str, ProductFields],
    annotations: dict[str, dict[str, Any]],
    *,
    max_tasks: int | None,
) -> list[Task]:
    tasks: list[Task] = []
    for asin in sorted(annotations):
        annotation = annotations[asin]
        instruction = annotation.get("instruction")
        instruction_attributes = annotation.get("instruction_attributes") or []
        if asin not in products or not instruction or not instruction_attributes:
            continue
        tasks.append(
            Task(
                task_id=f"webshop_static_{len(tasks):04d}_{asin}",
                target_asin=asin,
                instruction=str(instruction),
                instruction_attributes=tuple(str(attribute) for attribute in instruction_attributes),
                instruction_tokens=token_set(instruction),
                required_tokens=phrase_token_set(instruction_attributes),
            )
        )
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    return tasks


def sum_idf(tokens: set[str] | frozenset[str], idf: dict[str, float]) -> float:
    return sum(idf.get(token, 0.0) for token in tokens)


def method_score(
    method_key: str,
    task: Task,
    product: ProductFields,
    *,
    idf: dict[str, float],
    phrase_idf: dict[str, float],
    avg_all_doc_length: float,
    avg_title_category_length: float,
) -> tuple[float, ...]:
    instruction = task.instruction_tokens
    required = task.required_tokens
    query = frozenset(instruction - required)
    if method_key == "title_overlap":
        overlap = instruction & product.title_tokens
        return (float(len(overlap)), sum_idf(overlap, idf))
    if method_key == "description_overlap":
        overlap = instruction & product.description_tokens
        return (float(len(overlap)), sum_idf(overlap, idf))
    if method_key == "category_query_overlap":
        overlap = instruction & product.category_tokens
        return (float(len(overlap)), sum_idf(overlap, idf))
    if method_key == "all_text_overlap":
        overlap = instruction & product.all_tokens
        return (float(len(overlap)), sum_idf(overlap, idf))
    if method_key == "attribute_overlap":
        overlap = required & product.attribute_tokens
        return (float(len(overlap)), sum_idf(overlap, idf))
    if method_key == "rarest_attribute_anchor":
        overlap = required & product.attribute_tokens
        return (sum_idf(overlap, idf), float(len(overlap)))
    if method_key == "bm25_all_text":
        return (
            bm25_score(instruction, product.all_doc_tokens, idf, avg_all_doc_length),
            float(len(instruction & product.all_tokens)),
        )
    if method_key == "bm25_title_category":
        return (
            bm25_score(instruction, product.title_category_doc_tokens, idf, avg_title_category_length),
            float(len(instruction & (product.title_tokens | product.category_tokens))),
        )
    if method_key == "traceh_constraint_query_ledger":
        hard_attribute_overlap = required & product.attribute_tokens
        hard_anywhere_overlap = required & product.all_tokens
        query_category_overlap = query & product.category_tokens
        query_title_overlap = query & product.title_tokens
        query_all_overlap = query & product.all_tokens
        exact_attribute_phrase_score = sum(
            phrase_idf.get(attribute.lower(), 0.0)
            for attribute in task.instruction_attributes
            if attribute.lower() in product.attribute_phrases
        )
        return (
            float(len(hard_attribute_overlap)),
            exact_attribute_phrase_score,
            sum_idf(query_category_overlap, idf),
            sum_idf(query_title_overlap, idf),
            bm25_score(instruction, product.title_category_doc_tokens, idf, avg_title_category_length),
            sum_idf(query_all_overlap, idf),
            float(len(hard_anywhere_overlap)),
            -float(len(product.all_tokens)),
        )
    raise KeyError(f"Unknown method: {method_key}")


METHOD_SPECS = [
    ("title_overlap", "title lexical overlap", "baseline"),
    ("description_overlap", "description lexical overlap", "baseline"),
    ("category_query_overlap", "category/query lexical overlap", "baseline"),
    ("all_text_overlap", "all-text lexical overlap", "baseline"),
    ("attribute_overlap", "attribute constraint overlap", "baseline"),
    ("rarest_attribute_anchor", "rarest-attribute anchor", "baseline"),
    ("bm25_all_text", "BM25 all text", "baseline"),
    ("bm25_title_category", "BM25 title+category", "baseline"),
    ("traceh_constraint_query_ledger", "TRACE-H constraint+query ledger", "ours"),
]


def rank_products(
    method_key: str,
    task: Task,
    products: dict[str, ProductFields],
    *,
    idf: dict[str, float],
    phrase_idf: dict[str, float],
    avg_all_doc_length: float,
    avg_title_category_length: float,
) -> tuple[int, str, float, float]:
    ranked = []
    for asin, product in products.items():
        score = method_score(
            method_key,
            task,
            product,
            idf=idf,
            phrase_idf=phrase_idf,
            avg_all_doc_length=avg_all_doc_length,
            avg_title_category_length=avg_title_category_length,
        )
        ranked.append((score, -product.index, asin))
    ranked.sort(reverse=True)
    top_asin = ranked[0][2]
    target_rank = next(index for index, (_score, _neg_index, asin) in enumerate(ranked, start=1) if asin == task.target_asin)
    target_product = products[task.target_asin]
    top_product = products[top_asin]
    target_constraint_coverage = (
        len(task.required_tokens & target_product.attribute_tokens) / max(1, len(task.required_tokens))
    )
    top_constraint_coverage = len(task.required_tokens & top_product.attribute_tokens) / max(1, len(task.required_tokens))
    return target_rank, top_asin, target_constraint_coverage, top_constraint_coverage


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
    low_index = int(0.025 * (samples - 1))
    high_index = int(0.975 * (samples - 1))
    return float(means[low_index]), float(means[high_index])


def summarize_method(
    method_key: str,
    method_label: str,
    role: str,
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    exact_values = [float(row["methods"][method_key]["exact_top1"]) for row in rows]
    mrr_values = [float(row["methods"][method_key]["reciprocal_rank"]) for row in rows]
    coverage_values = [float(row["methods"][method_key]["top_constraint_coverage"]) for row in rows]
    mean_rank_values = [float(row["methods"][method_key]["target_rank"]) for row in rows]
    return {
        "method_key": method_key,
        "method_label": method_label,
        "role": role,
        "unit_count": len(rows),
        "primary_metric": "exact_target_rate",
        "primary_value": float(mean(exact_values)),
        "primary_ci95": list(bootstrap_ci(exact_values, seed=seed + 11)),
        "secondary_metric": "target_mrr",
        "secondary_value": float(mean(mrr_values)),
        "secondary_ci95": list(bootstrap_ci(mrr_values, seed=seed + 23)),
        "constraint_satisfaction_rate": float(mean(coverage_values)),
        "mean_target_rank": float(mean(mean_rank_values)),
    }


def add_ranks(records: list[dict[str, Any]]) -> None:
    for value_name, rank_name in (
        ("primary_value", "primary_rank"),
        ("secondary_value", "secondary_rank"),
    ):
        ranked = sorted(records, key=lambda item: (-float(item[value_name]), item["method_label"]))
        previous_value: float | None = None
        previous_rank = 0
        for index, record in enumerate(ranked, start=1):
            value = float(record[value_name])
            if previous_value is None or not math.isclose(value, previous_value, rel_tol=1e-12, abs_tol=1e-12):
                previous_rank = index
                previous_value = value
            record[rank_name] = previous_rank


def compare_to_ours(rows: list[dict[str, Any]], ours_key: str) -> list[dict[str, Any]]:
    comparisons = []
    for method_key, method_label, role in METHOD_SPECS:
        if method_key == ours_key:
            continue
        exact_diffs = [
            float(row["methods"][ours_key]["exact_top1"]) - float(row["methods"][method_key]["exact_top1"])
            for row in rows
        ]
        mrr_diffs = [
            float(row["methods"][ours_key]["reciprocal_rank"]) - float(row["methods"][method_key]["reciprocal_rank"])
            for row in rows
        ]
        comparisons.append(
            {
                "ours_key": ours_key,
                "baseline_key": method_key,
                "baseline_label": method_label,
                "baseline_role": role,
                "primary_mean_delta": float(mean(exact_diffs)),
                "secondary_mean_delta": float(mean(mrr_diffs)),
                "primary_sign_test": sign_test(exact_diffs),
                "secondary_sign_test": sign_test(mrr_diffs),
            }
        )
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    items = load_json(args.items)
    annotations = load_json(args.annotations)
    products = build_products(items, annotations)
    tasks = build_tasks(products, annotations, max_tasks=args.max_tasks)
    if not tasks:
        raise RuntimeError("No WebShop static tasks were found.")

    idf = idf_scores([product.all_tokens for product in products.values()])
    phrase_idf = phrase_idf_scores(products)
    avg_all_doc_length = mean(len(product.all_doc_tokens) for product in products.values())
    avg_title_category_length = mean(len(product.title_category_doc_tokens) for product in products.values())

    rows: list[dict[str, Any]] = []
    for task in tasks:
        method_results = {}
        for method_key, _method_label, _role in METHOD_SPECS:
            target_rank, top_asin, target_constraint_coverage, top_constraint_coverage = rank_products(
                method_key,
                task,
                products,
                idf=idf,
                phrase_idf=phrase_idf,
                avg_all_doc_length=avg_all_doc_length,
                avg_title_category_length=avg_title_category_length,
            )
            method_results[method_key] = {
                "top_asin": top_asin,
                "target_rank": target_rank,
                "exact_top1": top_asin == task.target_asin,
                "reciprocal_rank": 1.0 / target_rank,
                "target_constraint_coverage": target_constraint_coverage,
                "top_constraint_coverage": top_constraint_coverage,
            }
        rows.append(
            {
                "task_id": task.task_id,
                "target_asin": task.target_asin,
                "instruction": task.instruction,
                "instruction_attributes": list(task.instruction_attributes),
                "instruction_token_count": len(task.instruction_tokens),
                "required_token_count": len(task.required_tokens),
                "methods": method_results,
            }
        )

    records = [
        summarize_method(method_key, method_label, role, rows, seed=args.seed + index * 101)
        for index, (method_key, method_label, role) in enumerate(METHOD_SPECS)
    ]
    add_ranks(records)
    comparisons = compare_to_ours(rows, "traceh_constraint_query_ledger")
    paper_baseline_comparisons = [
        comparison for comparison in comparisons if comparison["baseline_role"] == "baseline"
    ]
    status = {
        "product_count": len(products),
        "task_count": len(tasks),
        "baseline_count": sum(1 for _key, _label, role in METHOD_SPECS if role == "baseline"),
        "ours_primary_rank_1": next(
            record for record in records if record["method_key"] == "traceh_constraint_query_ledger"
        )["primary_rank"]
        == 1,
        "ours_secondary_rank_1": next(
            record for record in records if record["method_key"] == "traceh_constraint_query_ledger"
        )["secondary_rank"]
        == 1,
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
        "interactive_env_status": "blocked_locally_missing_java_lucene_index",
        "evidence_boundary": (
            "Static product-selection projection over real WebShop-small catalog and official synthetic "
            "instruction attributes; not a full interactive WebShop result."
        ),
    }
    output = {
        "experiment_id": "L4-WEBSHOP-STATIC-PRODUCT-PK",
        "inputs": {
            "items": str(args.items),
            "annotations": str(args.annotations),
            "max_tasks": args.max_tasks,
            "seed": args.seed,
        },
        "method_specs": [
            {"method_key": method_key, "method_label": method_label, "role": role}
            for method_key, method_label, role in METHOD_SPECS
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
