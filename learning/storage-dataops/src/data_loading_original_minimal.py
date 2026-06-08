"""Minimal mechanisms from the data loading paper."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    dataset_samples: int
    train_rate_per_node: float
    storage_rate: float
    preprocess_rate_per_node: float

    def training_time(self, n_nodes: int) -> float:
        return self.dataset_samples / (n_nodes * self.train_rate_per_node)

    def data_loading_time(self, n_nodes: int) -> float:
        sample_io = self.dataset_samples / self.storage_rate
        preprocess = self.dataset_samples / (n_nodes * self.preprocess_rate_per_node)
        return sample_io + preprocess

    def regular_epoch_time(self, n_nodes: int) -> float:
        return max(self.training_time(n_nodes), self.data_loading_time(n_nodes))


def distributed_cache_io_time(
    dataset_samples: int,
    n_nodes: int,
    storage_rate: float,
    remote_cache_rate: float,
    cached_ratio: float,
) -> float:
    """Paper equation 7 in samples/sec units."""

    storage_miss = (1.0 - cached_ratio) * dataset_samples / storage_rate
    remote_hits = cached_ratio * dataset_samples / remote_cache_rate
    remote_hits *= (n_nodes - 1) / n_nodes
    return storage_miss + remote_hits


def locality_aware_io_time(
    dataset_samples: int,
    storage_rate: float,
    balance_rate: float,
    cached_ratio: float,
    balance_ratio: float,
) -> float:
    """Paper equation 8 in samples/sec units."""

    storage_miss = (1.0 - cached_ratio) * dataset_samples / storage_rate
    balance_cost = cached_ratio * dataset_samples * balance_ratio / balance_rate
    return storage_miss + balance_cost


def sample_distribution(batch: list[int], cache_owner: dict[int, int], n_nodes: int) -> list[int]:
    counts = [0] * n_nodes
    for sample_id in batch:
        counts[cache_owner[sample_id]] += 1
    return counts


def balance_transfers(counts: list[int], target_per_node: int) -> list[tuple[int, int, int]]:
    """Greedy surplus-to-deficit schedule, mirroring Algorithm 1."""

    surplus = [[i, count - target_per_node] for i, count in enumerate(counts)
               if count > target_per_node]
    deficit = [[i, target_per_node - count] for i, count in enumerate(counts)
               if count < target_per_node]
    transfers: list[tuple[int, int, int]] = []
    surplus.sort(key=lambda x: x[1], reverse=True)
    deficit.sort(key=lambda x: x[1], reverse=True)

    si = 0
    di = 0
    while si < len(surplus) and di < len(deficit):
        src, extra = surplus[si]
        dst, need = deficit[di]
        moved = min(extra, need)
        transfers.append((src, dst, moved))
        surplus[si][1] -= moved
        deficit[di][1] -= moved
        if surplus[si][1] == 0:
            si += 1
        if deficit[di][1] == 0:
            di += 1
    return transfers


def imbalance_ratio(counts: list[int], target_per_node: int) -> float:
    deficit = sum(max(0, target_per_node - count) for count in counts)
    return deficit / max(1, sum(counts))


def gradient_sum(sample_ids: list[int], gradients: dict[int, float]) -> float:
    return sum(gradients[i] for i in sample_ids)


def partitions_equivalent(
    regular_partitions: list[list[int]],
    locality_partitions: list[list[int]],
    gradients: dict[int, float],
) -> bool:
    reg = sum(gradient_sum(part, gradients) for part in regular_partitions)
    loc = sum(gradient_sum(part, gradients) for part in locality_partitions)
    return abs(reg - loc) < 1e-9


def _self_test() -> None:
    model = CostModel(
        dataset_samples=1_000_000,
        train_rate_per_node=10_000,
        storage_rate=100_000,
        preprocess_rate_per_node=1_000_000,
    )
    assert model.regular_epoch_time(4) > model.regular_epoch_time(16)
    assert abs(model.regular_epoch_time(64) - model.regular_epoch_time(128)) < 0.1

    regular_cached = distributed_cache_io_time(
        dataset_samples=1_000_000,
        n_nodes=64,
        storage_rate=100_000,
        remote_cache_rate=200_000,
        cached_ratio=1.0,
    )
    local_aware = locality_aware_io_time(
        dataset_samples=1_000_000,
        storage_rate=100_000,
        balance_rate=1_000_000,
        cached_ratio=1.0,
        balance_ratio=0.05,
    )
    assert local_aware < regular_cached / 10, (regular_cached, local_aware)

    batch = list(range(12))
    owner = {
        0: 0, 1: 0,
        2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1,
        8: 2, 9: 2, 10: 2, 11: 2,
    }
    counts = sample_distribution(batch, owner, n_nodes=3)
    assert counts == [2, 6, 4]
    transfers = balance_transfers(counts, target_per_node=4)
    assert transfers == [(1, 0, 2)]
    assert round(imbalance_ratio(counts, target_per_node=4), 3) == 0.167

    gradients = {i: float(i + 1) for i in batch}
    regular = [batch[0:4], batch[4:8], batch[8:12]]
    locality = [[0, 1], [2, 3, 4, 5, 6, 7], [8, 9, 10, 11]]
    assert partitions_equivalent(regular, locality, gradients)
    print("[OK] data_loading_original_minimal (model, locality, balance)")


if __name__ == "__main__":
    _self_test()
