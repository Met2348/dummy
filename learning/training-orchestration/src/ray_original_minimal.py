"""Minimal Ray paper mechanisms: GCS, scheduling, objects, actors."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObjectRef:
    object_id: str


@dataclass
class ObjectRecord:
    ref: ObjectRef
    size_mb: float
    locations: set[str]
    creator_task: str | None = None
    ready: bool = True


@dataclass
class TaskSpec:
    task_id: str
    name: str
    input_refs: list[ObjectRef] = field(default_factory=list)
    duration_ms: float = 1.0
    cpus: int = 1
    gpus: int = 0
    actor_id: str | None = None
    stateful_dep: str | None = None
    assigned_node: str | None = None


@dataclass
class NodeState:
    node_id: str
    cpus_total: int
    gpus_total: int
    queued_ms: float = 0.0
    cpus_free: int | None = None
    gpus_free: int | None = None

    def __post_init__(self) -> None:
        if self.cpus_free is None:
            self.cpus_free = self.cpus_total
        if self.gpus_free is None:
            self.gpus_free = self.gpus_total

    def can_run(self, task: TaskSpec) -> bool:
        return self.cpus_free >= task.cpus and self.gpus_free >= task.gpus

    def reserve(self, task: TaskSpec) -> None:
        if not self.can_run(task):
            raise ValueError(f"node {self.node_id} lacks resources for {task.task_id}")
        self.cpus_free -= task.cpus
        self.gpus_free -= task.gpus
        self.queued_ms += task.duration_ms


@dataclass
class GlobalControlStore:
    """A tiny stand-in for Ray's sharded pub-sub control metadata store."""

    objects: dict[str, ObjectRecord] = field(default_factory=dict)
    tasks: dict[str, TaskSpec] = field(default_factory=dict)
    functions: set[str] = field(default_factory=set)
    actor_last_task: dict[str, str] = field(default_factory=dict)

    def register_function(self, name: str) -> None:
        self.functions.add(name)

    def put_object(
        self,
        object_id: str,
        size_mb: float,
        locations: set[str],
        creator_task: str | None = None,
        ready: bool = True,
    ) -> ObjectRef:
        ref = ObjectRef(object_id)
        self.objects[object_id] = ObjectRecord(
            ref=ref,
            size_mb=size_mb,
            locations=set(locations),
            creator_task=creator_task,
            ready=ready,
        )
        return ref

    def record_task(self, task: TaskSpec) -> None:
        self.tasks[task.task_id] = task

    def remote_input_mb(self, task: TaskSpec, node_id: str) -> float:
        total = 0.0
        for ref in task.input_refs:
            rec = self.objects[ref.object_id]
            if node_id not in rec.locations:
                total += rec.size_mb
        return total


def choose_node_bottom_up(
    task: TaskSpec,
    local_node_id: str,
    nodes: dict[str, NodeState],
    gcs: GlobalControlStore,
    queue_threshold_ms: float = 50.0,
    bandwidth_mb_per_ms: float = 10.0,
) -> str:
    """Ray paper scheduler: local first, global fallback with locality cost."""

    local = nodes[local_node_id]
    if local.can_run(task) and local.queued_ms <= queue_threshold_ms:
        return local_node_id

    candidates = [node for node in nodes.values() if node.can_run(task)]
    if not candidates:
        raise ValueError(f"no node can satisfy resources for {task.task_id}")

    def score(node: NodeState) -> float:
        transfer_ms = gcs.remote_input_mb(task, node.node_id) / bandwidth_mb_per_ms
        return node.queued_ms + transfer_ms

    return min(candidates, key=score).node_id


def submit_task(
    task: TaskSpec,
    local_node_id: str,
    nodes: dict[str, NodeState],
    gcs: GlobalControlStore,
    output_size_mb: float = 1.0,
) -> ObjectRef:
    node_id = choose_node_bottom_up(task, local_node_id, nodes, gcs)
    nodes[node_id].reserve(task)
    task.assigned_node = node_id
    gcs.record_task(task)
    return gcs.put_object(
        object_id=f"{task.task_id}:out",
        size_mb=output_size_mb,
        locations={node_id},
        creator_task=task.task_id,
    )


def ray_get(ref: ObjectRef, requester_node: str, gcs: GlobalControlStore) -> ObjectRecord:
    """Blocking get simplified to metadata: the object becomes local after copy."""

    rec = gcs.objects[ref.object_id]
    if not rec.ready:
        raise RuntimeError(f"object {ref.object_id} is not ready")
    rec.locations.add(requester_node)
    return rec


def ray_wait(
    refs: list[ObjectRef], k: int, gcs: GlobalControlStore
) -> tuple[list[ObjectRef], list[ObjectRef]]:
    ready = [ref for ref in refs if gcs.objects[ref.object_id].ready]
    ready_k = ready[:k]
    ready_ids = {ref.object_id for ref in ready_k}
    remaining = [ref for ref in refs if ref.object_id not in ready_ids]
    return ready_k, remaining


def actor_method_call(
    actor_id: str, method: str, gcs: GlobalControlStore, duration_ms: float = 1.0
) -> TaskSpec:
    """Record actor calls as normal tasks plus a stateful edge chain."""

    index = sum(1 for tid in gcs.tasks if tid.startswith(f"{actor_id}."))
    task = TaskSpec(
        task_id=f"{actor_id}.{method}.{index}",
        name=f"{actor_id}.{method}",
        duration_ms=duration_ms,
        actor_id=actor_id,
        stateful_dep=gcs.actor_last_task.get(actor_id),
    )
    gcs.record_task(task)
    gcs.actor_last_task[actor_id] = task.task_id
    return task


def reconstruct_lineage(ref: ObjectRef, gcs: GlobalControlStore) -> list[str]:
    """Return the tasks that must be replayed to reconstruct an object."""

    ordered: list[str] = []
    seen: set[str] = set()

    def visit_task(task_id: str) -> None:
        if task_id in seen:
            return
        task = gcs.tasks[task_id]
        if task.stateful_dep is not None:
            visit_task(task.stateful_dep)
        for input_ref in task.input_refs:
            rec = gcs.objects[input_ref.object_id]
            if rec.creator_task is not None:
                visit_task(rec.creator_task)
        seen.add(task_id)
        ordered.append(task_id)

    rec = gcs.objects[ref.object_id]
    if rec.creator_task is not None:
        visit_task(rec.creator_task)
    return ordered


def theoretical_task_throughput(n_nodes: int, cores_per_node: int, task_ms: float) -> float:
    """Paper's back-of-envelope: independent tasks per second at full utilization."""

    return n_nodes * cores_per_node * (1000.0 / task_ms)


def _self_test() -> None:
    gcs = GlobalControlStore()
    gcs.register_function("preprocess")
    big = gcs.put_object("big-image-batch", size_mb=1000.0, locations={"B"})

    nodes = {
        "A": NodeState("A", cpus_total=8, gpus_total=0, queued_ms=10.0),
        "B": NodeState("B", cpus_total=8, gpus_total=0, queued_ms=0.0),
    }
    local_task = TaskSpec("t-local", "preprocess", input_refs=[big])
    assert choose_node_bottom_up(local_task, "A", nodes, gcs) == "A"

    nodes["A"].queued_ms = 200.0
    remote_task = TaskSpec("t-locality", "preprocess", input_refs=[big])
    assert choose_node_bottom_up(remote_task, "A", nodes, gcs) == "B"

    out1 = submit_task(TaskSpec("t1", "make"), "B", nodes, gcs, output_size_mb=2.0)
    out2 = submit_task(TaskSpec("t2", "consume", input_refs=[out1]), "A", nodes, gcs)
    assert reconstruct_lineage(out2, gcs) == ["t1", "t2"]

    first = actor_method_call("trainer", "step", gcs)
    second = actor_method_call("trainer", "step", gcs)
    assert second.stateful_dep == first.task_id

    not_ready = gcs.put_object("slow", 1.0, {"A"}, ready=False)
    ready, remaining = ray_wait([out1, not_ready, out2], k=2, gcs=gcs)
    assert [ref.object_id for ref in ready] == [out1.object_id, out2.object_id]
    assert remaining == [not_ready]

    ray_get(out2, "A", gcs)
    assert "A" in gcs.objects[out2.object_id].locations

    tps = theoretical_task_throughput(n_nodes=200, cores_per_node=32, task_ms=5.0)
    assert tps == 1_280_000.0
    print("[OK] ray_original_minimal (GCS, scheduler, actors, lineage)")


if __name__ == "__main__":
    _self_test()
