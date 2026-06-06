# L01 — Slurm 与作业调度

## 核心模型

```
sbatch script → controller (slurmctld) → node daemons (slurmd)
```

- FIFO 队列 + priority + fairshare
- Backfill：让小作业插队到 head 后空窗
- Reservation：为关键 job 预留时间窗

## 关键命令

```bash
sbatch --gpus=8 --time=24:00:00 train.sh    # 提交
squeue -u $USER                              # 查队列
sacct -j 12345 --format=ElapsedRaw,MaxRSS   # 历史
scontrol show job 12345                      # 详情
```

## 实战配置

```bash
#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8        # 8 GPU per node
#SBATCH --gpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --partition=h100

srun python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=8 \
    train.py
```

## 调度公平性

- fairshare = (used_recently / fair_share_limit)^-1
- 一个用户连训 7 天 → 优先级自动衰减
- QoS：production/dev/preempt 三档
