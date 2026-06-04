"""Prometheus-style metrics — vendored counters so tests don't require the lib."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import time


@dataclass
class Counter:
    name: str
    help: str
    labels: List[str] = field(default_factory=list)
    values: Dict[tuple, float] = field(default_factory=dict)

    def inc(self, labels: tuple = (), value: float = 1.0) -> None:
        self.values[labels] = self.values.get(labels, 0.0) + value

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        for k, v in self.values.items():
            label_str = ",".join(f'{ln}="{lv}"' for ln, lv in zip(self.labels, k)) if k else ""
            lines.append(f"{self.name}{{{label_str}}} {v}" if label_str else f"{self.name} {v}")
        return "\n".join(lines)


@dataclass
class Histogram:
    name: str
    help: str
    buckets: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    counts: List[int] = field(default_factory=list)
    sum_: float = 0.0
    count_: int = 0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * len(self.buckets)

    def observe(self, x: float) -> None:
        for i, b in enumerate(self.buckets):
            if x <= b:
                self.counts[i] += 1
        self.sum_ += x
        self.count_ += 1

    def percentile(self, p: float) -> float:
        if self.count_ == 0:
            return 0.0
        target = self.count_ * p
        running = 0
        for i, b in enumerate(self.buckets):
            running += self.counts[i]
            if running >= target:
                return b
        return self.buckets[-1]

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for b, c in zip(self.buckets, self.counts):
            lines.append(f"{self.name}_bucket{{le=\"{b}\"}} {c}")
        lines.append(f"{self.name}_sum {self.sum_}")
        lines.append(f"{self.name}_count {self.count_}")
        return "\n".join(lines)


REQS = Counter("llm_requests_total", "Total LLM API requests", labels=["model"])
TTFT = Histogram("llm_ttft_seconds", "Time to first token")


def render_all() -> str:
    return "\n\n".join((REQS.render(), TTFT.render()))
