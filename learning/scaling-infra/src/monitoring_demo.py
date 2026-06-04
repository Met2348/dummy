"""Monitoring helpers - MFU 计算 / EMA / 健康检查."""
from __future__ import annotations


def compute_mfu(tokens_per_sec: float, n_params: int,
                n_gpu: int, gpu_tflops: float) -> float:
    """6N D / time = throughput (loss FLOPs)."""
    actual_flops = 6 * n_params * tokens_per_sec
    theoretical = n_gpu * gpu_tflops * 1e12
    return actual_flops / theoretical


class EmaLossTracker:
    def __init__(self, alpha: float = 0.99):
        self.ema = None
        self.alpha = alpha

    def update(self, loss: float) -> float:
        if self.ema is None:
            self.ema = loss
        else:
            self.ema = self.alpha * self.ema + (1 - self.alpha) * loss
        return self.ema

    def is_spike(self, loss: float, threshold: float = 3.0) -> bool:
        if self.ema is None:
            return False
        return loss > threshold * self.ema


def gpu_health(util: float, temp_c: float, mem_used_gb: float,
                mem_total_gb: float) -> str:
    issues = []
    if util < 80:
        issues.append(f"util={util}% low")
    if temp_c > 85:
        issues.append(f"temp={temp_c}°C high")
    if mem_used_gb / mem_total_gb > 0.95:
        issues.append(f"mem {mem_used_gb}/{mem_total_gb} GB near OOM")
    return "OK" if not issues else " | ".join(issues)


def ckpt_template():
    print("=== Checkpoint save (FSDP) ===")
    print("""
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)

cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "rng_cuda": torch.cuda.get_rng_state_all(),
        "rng_cpu": torch.get_rng_state(),
    }
    if rank == 0:
        torch.save(state, "ckpt.pt")
""")


if __name__ == "__main__":
    print("=== MFU (Llama-3 8B on 1 H100) ===")
    mfu = compute_mfu(tokens_per_sec=3000, n_params=8e9, n_gpu=1,
                       gpu_tflops=990)
    print(f"  3000 tok/s → MFU = {mfu:.1%}")

    print("\n=== EMA tracker ===")
    t = EmaLossTracker()
    for L in [2.5, 2.4, 2.3, 2.3, 9.0, 2.3]:
        ema = t.update(L if not t.is_spike(L) else t.ema)
        print(f"  loss={L} ema={ema:.3f} "
              f"spike={t.is_spike(L)}")

    print("\n=== GPU health ===")
    print("  ", gpu_health(util=95, temp_c=70, mem_used_gb=20, mem_total_gb=24))
    print("  ", gpu_health(util=40, temp_c=92, mem_used_gb=23, mem_total_gb=24))

    ckpt_template()
