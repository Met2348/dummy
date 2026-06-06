# L04 — Ray & Actor 模型

## 三大概念

- **Actor**：有状态的 process，方法通过 RPC 调用
- **Task**：无状态函数，scatter-gather 模式
- **Placement Group**：拓扑感知预留

## 适用场景

| 场景 | 选 Ray | 选 Slurm |
|------|-------|---------|
| 长跑训练 | △ | ✓ |
| 弹性推理服务 | ✓ | ✗ |
| RLHF (actor/critic/RM/ref) | ✓ | △ |
| 超参 sweep | ✓ | △ |
| 多模态多 stage pipeline | ✓ | △ |

## RLHF 经典布局

```python
trainer = TrainerActor.remote()      # actor 模型
critic  = CriticActor.remote()
rm      = RewardModelActor.remote()
ref     = RefModelActor.remote()

rollouts = trainer.generate.remote(prompts)
rewards = rm.score.remote(rollouts)
values = critic.score.remote(rollouts)
trainer.update.remote(rollouts, rewards, values, ref)
```

Ray 用于 RLHF 是 **verl / OpenRLHF / TRL 多卡** 的标准底盘。

## Ray Train + Ray Tune

- Ray Train：包 PyTorch DDP / FSDP，自动 rendezvous
- Ray Tune：超参 sweep + early stop (ASHA / PBT)
- Ray Serve：推理弹性服务

## 局限

- 调度公平性 / quota 弱于 Slurm → 共享 HPC 用 Slurm，专属云用 Ray
