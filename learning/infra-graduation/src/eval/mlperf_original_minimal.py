"""Minimal MLPerf Training paper rules: time-to-quality and divisions."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    area: str
    dataset: str
    model: str
    metric: str
    target: float
    higher_is_better: bool = True
    required_runs: int = 10

    def reached(self, value: float) -> bool:
        if self.higher_is_better:
            return value >= self.target
        return value <= self.target


MLPERF_V05_TASKS = [
    BenchmarkTask(
        "image-classification",
        "vision",
        "ImageNet",
        "ResNet-50 v1.5",
        "Top-1 accuracy",
        74.9,
        required_runs=5,
    ),
    BenchmarkTask(
        "object-detection-light",
        "vision",
        "COCO 2017",
        "SSD-ResNet-34",
        "mAP",
        21.2,
        required_runs=5,
    ),
    BenchmarkTask(
        "instance-segmentation-heavy",
        "vision",
        "COCO 2017",
        "Mask R-CNN",
        "Box min AP",
        37.7,
        required_runs=5,
    ),
    BenchmarkTask(
        "translation-recurrent",
        "language",
        "WMT16 EN-DE",
        "GNMT",
        "SacreBLEU",
        21.8,
    ),
    BenchmarkTask(
        "translation-nonrecurrent",
        "language",
        "WMT17 EN-DE",
        "Transformer",
        "BLEU",
        25.0,
    ),
    BenchmarkTask(
        "recommendation",
        "recommendation",
        "MovieLens-20M",
        "NCF",
        "HR@10",
        0.635,
    ),
    BenchmarkTask(
        "reinforcement-learning",
        "rl",
        "9x9 Go",
        "MiniGo",
        "professional move prediction",
        40.0,
    ),
]


@dataclass(frozen=True)
class QualityEvent:
    elapsed_s: float
    value: float


@dataclass
class TrainingRun:
    seed: int
    events: list[QualityEvent]

    def time_to_quality(self, task: BenchmarkTask) -> float | None:
        for event in sorted(self.events, key=lambda e: e.elapsed_s):
            if task.reached(event.value):
                return event.elapsed_s
        return None


def reported_time_to_quality(task: BenchmarkTask, runs: list[TrainingRun]) -> float:
    """MLPerf-style result: drop fastest and slowest, average the rest."""

    if len(runs) < task.required_runs:
        raise ValueError(f"{task.name} requires {task.required_runs} timing runs")
    times = []
    for run in runs:
        t = run.time_to_quality(task)
        if t is None:
            raise ValueError(f"run seed={run.seed} never reached target")
        times.append(t)
    times = sorted(times)
    middle = times[1:-1]
    return sum(middle) / len(middle)


@dataclass
class Submission:
    division: str
    dataset: str
    metric: str
    model_equivalent: bool
    data_traversal_equivalent: bool
    modified_hyperparams: set[str] = field(default_factory=set)
    code_open: bool = True
    system_description_present: bool = True
    logs_present: bool = True


ALLOWED_CLOSED_HYPERPARAMS = {
    "batch_size",
    "learning_rate_schedule",
    "max_samples_per_training_patch",
    "image_candidates",
    "optimizer_choice",
    "learning_rate",
    "warmup_steps",
    "adam_beta1",
    "adam_beta2",
}


def validate_submission(task: BenchmarkTask, sub: Submission) -> tuple[bool, list[str]]:
    """Return MLPerf rule compliance reasons for a toy submission."""

    reasons: list[str] = []
    if sub.dataset != task.dataset:
        reasons.append("dataset differs from reference")
    if sub.metric != task.metric:
        reasons.append("quality metric differs from reference")
    if not sub.code_open:
        reasons.append("submission code is not public")
    if not sub.system_description_present:
        reasons.append("missing hardware/software system description")
    if not sub.logs_present:
        reasons.append("missing structured training logs")

    division = sub.division.lower()
    if division not in {"closed", "open"}:
        reasons.append("division must be closed or open")
    if division == "closed":
        if not sub.model_equivalent:
            reasons.append("closed division requires model equivalence")
        if not sub.data_traversal_equivalent:
            reasons.append("closed division requires equivalent data traversal")
        extra = sub.modified_hyperparams - ALLOWED_CLOSED_HYPERPARAMS
        if extra:
            reasons.append("closed division has disallowed hyperparameters")

    return (not reasons), reasons


def speedup(old_time_s: float, new_time_s: float) -> float:
    if old_time_s <= 0 or new_time_s <= 0:
        raise ValueError("times must be positive")
    return old_time_s / new_time_s


def _make_run(seed: int, seconds_to_target: float, target: float) -> TrainingRun:
    return TrainingRun(
        seed=seed,
        events=[
            QualityEvent(seconds_to_target * 0.5, target - 5.0),
            QualityEvent(seconds_to_target, target),
        ],
    )


def _self_test() -> None:
    assert len(MLPERF_V05_TASKS) == 7
    resnet = MLPERF_V05_TASKS[0]
    runs = [_make_run(i, t, resnet.target) for i, t in enumerate([120, 125, 128, 130, 200])]
    # Drop fastest 120 and slowest 200, average 125, 128, 130.
    assert round(reported_time_to_quality(resnet, runs), 2) == 127.67

    closed_bad = Submission(
        division="closed",
        dataset="ImageNet",
        metric="Top-1 accuracy",
        model_equivalent=True,
        data_traversal_equivalent=True,
        modified_hyperparams={"batch_size", "new_data_augmentation"},
    )
    ok, reasons = validate_submission(resnet, closed_bad)
    assert not ok
    assert "closed division has disallowed hyperparameters" in reasons

    open_good = Submission(
        division="open",
        dataset="ImageNet",
        metric="Top-1 accuracy",
        model_equivalent=False,
        data_traversal_equivalent=False,
        modified_hyperparams={"new_data_augmentation"},
    )
    ok, reasons = validate_submission(resnet, open_good)
    assert ok, reasons

    assert round(speedup(100.0, 76.9), 2) == 1.30
    print("[OK] mlperf_original_minimal (rules, trimmed mean, divisions)")


if __name__ == "__main__":
    _self_test()
