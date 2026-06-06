"""Ray-style actor model for distributed coordination."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Actor:
    actor_id: int
    name: str
    state: dict = field(default_factory=dict)

    def call(self, method: str, *args, **kwargs):
        fn = getattr(self, method)
        return fn(*args, **kwargs)


@dataclass
class TrainerActor(Actor):
    def step(self, batch_size: int) -> dict:
        self.state["step"] = self.state.get("step", 0) + 1
        self.state["tokens_seen"] = self.state.get("tokens_seen", 0) + batch_size
        return {"step": self.state["step"], "loss": 1.0 / self.state["step"]}


@dataclass
class ParameterServer(Actor):
    def push(self, gradient: list[float]) -> None:
        params = self.state.setdefault("params", [0.0] * len(gradient))
        for i, g in enumerate(gradient):
            params[i] -= 0.001 * g

    def pull(self) -> list[float]:
        return list(self.state.get("params", []))


@dataclass
class ActorSystem:
    actors: dict[int, Actor] = field(default_factory=dict)

    def spawn(self, cls, name: str, **kwargs) -> Actor:
        actor_id = len(self.actors)
        a = cls(actor_id=actor_id, name=name, **kwargs)
        self.actors[actor_id] = a
        return a

    def call(self, actor_id: int, method: str, *args, **kwargs):
        return self.actors[actor_id].call(method, *args, **kwargs)


def _self_test() -> None:
    sys = ActorSystem()
    trainer = sys.spawn(TrainerActor, "trainer-0")
    ps = sys.spawn(ParameterServer, "ps-0")

    r1 = sys.call(trainer.actor_id, "step", 1024)
    assert r1["step"] == 1
    r2 = sys.call(trainer.actor_id, "step", 1024)
    assert r2["step"] == 2

    sys.call(ps.actor_id, "push", [1.0, 2.0, 3.0])
    params = sys.call(ps.actor_id, "pull")
    assert len(params) == 3
    assert params[0] == -0.001
    print("[OK] ray_actors (trainer + PS round-trip)")


if __name__ == "__main__":
    _self_test()
