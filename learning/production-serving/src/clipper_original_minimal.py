"""Small Clipper-style serving mechanisms.

This is not a full prediction-serving system. It isolates the core ideas from
the Clipper paper so they can be tested locally:

* prediction caching keyed by model and input
* adaptive batching with additive increase and multiplicative decrease
* model-container latency profiles
* best-effort ensemble prediction under a latency deadline
* a tiny Exp3-style model-selection update
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Tuple


class PredictionCache:
    def __init__(self, capacity: int = 128):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: OrderedDict[Tuple[str, str], str] = OrderedDict()

    def fetch(self, model_id: str, query: str) -> str | None:
        key = (model_id, query)
        if key not in self._items:
            return None
        value = self._items.pop(key)
        self._items[key] = value
        return value

    def put(self, model_id: str, query: str, prediction: str) -> None:
        key = (model_id, query)
        if key in self._items:
            self._items.pop(key)
        self._items[key] = prediction
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)


@dataclass
class AdaptiveBatcher:
    slo_ms: float
    max_batch_size: int = 1
    additive_step: int = 1
    multiplicative_backoff: float = 0.9

    def choose_batch_size(self, queue_len: int) -> int:
        return max(0, min(queue_len, self.max_batch_size))

    def observe(self, batch_size: int, latency_ms: float) -> None:
        if latency_ms <= self.slo_ms:
            self.max_batch_size = max(self.max_batch_size, batch_size + self.additive_step)
            return
        backed_off = int(math.floor(batch_size * self.multiplicative_backoff))
        self.max_batch_size = max(1, backed_off)


@dataclass(frozen=True)
class ModelContainer:
    name: str
    fixed_ms: float
    per_item_ms: float
    accuracy: float

    def latency_ms(self, batch_size: int = 1) -> float:
        return self.fixed_ms + self.per_item_ms * batch_size

    def predict(self, query: str) -> str:
        return f"{self.name}:{query}"


def best_effort_ensemble(
    models: Iterable[ModelContainer],
    query: str,
    deadline_ms: float,
) -> Dict[str, object]:
    used: List[str] = []
    missing: List[str] = []
    predictions: List[str] = []
    accuracies: List[float] = []

    for model in models:
        if model.latency_ms(batch_size=1) <= deadline_ms:
            used.append(model.name)
            predictions.append(model.predict(query))
            accuracies.append(model.accuracy)
        else:
            missing.append(model.name)

    total = len(used) + len(missing)
    confidence = len(used) / total if total else 0.0
    return {
        "prediction": predictions[0] if predictions else None,
        "used_models": used,
        "missing_models": missing,
        "confidence": confidence,
        "mean_accuracy_proxy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
    }


def exp3_probabilities(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must have positive sum")
    return {name: weight / total for name, weight in weights.items()}


def exp3_update(
    weights: Dict[str, float],
    chosen: str,
    reward: float,
    probability: float,
    learning_rate: float = 0.2,
) -> Dict[str, float]:
    if chosen not in weights:
        raise KeyError(chosen)
    if probability <= 0:
        raise ValueError("probability must be positive")
    updated = dict(weights)
    updated[chosen] *= math.exp(learning_rate * reward / probability)
    return updated
