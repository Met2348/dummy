"""Torchrun-style elastic — rendezvous + dynamic membership."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RendezvousState:
    min_nodes: int
    max_nodes: int
    current_members: set[int] = field(default_factory=set)
    generation: int = 0

    def is_quorum(self) -> bool:
        return len(self.current_members) >= self.min_nodes

    def can_admit(self) -> bool:
        return len(self.current_members) < self.max_nodes


def join(rdv: RendezvousState, node_id: int) -> bool:
    if not rdv.can_admit():
        return False
    if node_id in rdv.current_members:
        return False
    rdv.current_members.add(node_id)
    rdv.generation += 1
    return True


def leave(rdv: RendezvousState, node_id: int) -> bool:
    if node_id not in rdv.current_members:
        return False
    rdv.current_members.discard(node_id)
    rdv.generation += 1
    return True


def world_size(rdv: RendezvousState) -> int:
    return len(rdv.current_members)


def _self_test() -> None:
    rdv = RendezvousState(min_nodes=4, max_nodes=16)
    assert not rdv.is_quorum()
    for i in range(4):
        assert join(rdv, i)
    assert rdv.is_quorum()
    assert world_size(rdv) == 4
    assert rdv.generation == 4

    # Scale up to 12
    for i in range(4, 12):
        assert join(rdv, i)
    assert world_size(rdv) == 12

    # Can't admit beyond max
    for i in range(12, 20):
        ok = join(rdv, i)
        if i < 16:
            assert ok
        else:
            assert not ok

    # Leave: world shrinks but quorum holds
    assert leave(rdv, 0)
    assert rdv.is_quorum()
    # Mass leave → no quorum
    for i in range(1, 14):
        leave(rdv, i)
    assert not rdv.is_quorum()
    print(f"[OK] elastic_training (final world {world_size(rdv)}, "
          f"gen {rdv.generation})")


if __name__ == "__main__":
    _self_test()
