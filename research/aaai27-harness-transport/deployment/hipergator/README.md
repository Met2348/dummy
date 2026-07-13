# TRACE-H 在 UF HiPerGator 上的运行说明

本目录只提供 scheduler、资源和 runner 接口模板。TRACE-H episode runner 尚未在本目录伪造实现；`TRACEH_BLOCK_RUNNER` 必须指向项目中经过本地 smoke 的可执行 block runner。

总资源决策见[本地与 HiPerGator 分阶段方案](../local-vs-hipergator-execution-plan-zh.md)。

## 1. 登录后先查 allocation

```bash
module load ufrc
slurmInfo -g <group>
showQos
blue_quota
orange_quota
nodeInfo
sinfo -p hpg-b200
```

不要根据默认 trial 猜 group 实际资源。array 的并发上限 `%K` 必须小于等于当前可用 GPU 数，并给其他组员留出余量。

## 2. 目录与环境

所有 active workloads 从 Blue 运行：

```bash
export HPG_ACCOUNT=<group>
export TRACEH_ROOT=/blue/$HPG_ACCOUNT/$USER/trace-h
mkdir -p "$TRACEH_ROOT"/{repo,envs,containers,models,data,manifests,artifacts,runs,logs}
cd "$TRACEH_ROOT"
```

Python/uv/conda cache 也放 Blue，不占 40GB Home：

```bash
export UV_CACHE_DIR="$TRACEH_ROOT/.cache/uv"
export HF_HOME="$TRACEH_ROOT/.cache/huggingface"
```

若使用 Apptainer：

```bash
apptainer exec --nv "$TRACEH_ROOT/containers/traceh.sif" python -c 'import torch; print(torch.cuda.get_device_name())'
```

多 GPU B200 job 还要加 `--ipc=host` 或 `--bind /dev/shm`。容器 SIF 放 `/blue`，不放 Home。

## 3. GPU preflight

提交前在当前目录创建日志位置不是必需的；模板把日志写到提交目录。

### L4

脚本请求 1 GPU、4 CPU、32GB RAM；不写 partition 时，当前 UF 规则会把普通 GPU request 调度到带 L4 的 `hpg-turin`。

```bash
export TRACEH_PROJECT_ROOT="$TRACEH_ROOT/repo"
export TRACEH_PYTHON="$TRACEH_ROOT/envs/traceh/bin/python"
sbatch --account="$HPG_ACCOUNT" slurm/gpu-preflight.sbatch
```

### B200

必须显式指定 `hpg-b200`。按当前硬件比例请求 14 CPU、112GB RAM：

```bash
sbatch \
  --account="$HPG_ACCOUNT" \
  --partition=hpg-b200 \
  --cpus-per-task=14 \
  --mem=112G \
  slurm/gpu-preflight.sbatch
```

preflight 必须报告 `torch.cuda.is_available() == True`，并显示预期 GPU 名称与 capability。L4 为 `sm_89`，B200 为 `sm_100`。

## 4. Block runner 接口

Slurm array 不直接理解 TRACE-H schema。它只约定 runner 接口：

```text
TRACEH_BLOCK_RUNNER \
  --manifest <absolute path> \
  --block-index <zero-based array index> \
  --output-dir <new directory>
```

runner 必须：

- 一个 block 内只加载一次模型；
- 从 manifest 的一行读取 phase/model/tasks/policy/seed；
- append-only 写 episode records；
- 已存在同 `run_id` 时拒绝覆盖；
- 失败时非零退出；
- 记录 git/model/env/GPU/Slurm metadata；
- target phase 遵守 `policy-transport-seal.md`。

[示例 block manifest](block-manifest.example.tsv) 只定义最小列，正式 manifest 还应有 checkpoint hash、task hash、prompt hash、action budget 和 target information class。

## 5. L4 rollout array

```bash
export TRACEH_PROJECT_ROOT="$TRACEH_ROOT/repo"
export TRACEH_BLOCK_MANIFEST="$TRACEH_ROOT/manifests/source-l4.tsv"
export TRACEH_BLOCK_RUNNER="$TRACEH_ROOT/repo/bin/traceh-run-block"
export TRACEH_OUTPUT_ROOT="$TRACEH_ROOT/runs/source_branch"
export TRACEH_PYTHON="$TRACEH_ROOT/envs/traceh/bin/python"

# 12 blocks，最多同时使用 2 GPU；按 slurmInfo 修改 %2。
sbatch --account="$HPG_ACCOUNT" --array=0-11%2 slurm/rollout-block-array-l4.sbatch
```

若要把 checkpoint stage 到 node-local flash：

```bash
export TRACEH_MODEL_SOURCE="$TRACEH_ROOT/models/Qwen3-8B"
```

模板会复制到 `$SLURM_TMPDIR/model` 并设置 `TRACEH_MODEL_PATH`。block 太短时不要 stage 大模型，复制成本可能超过计算。

## 6. B200 rollout array

```bash
export TRACEH_PROJECT_ROOT="$TRACEH_ROOT/repo"
export TRACEH_BLOCK_MANIFEST="$TRACEH_ROOT/manifests/target-b200.tsv"
export TRACEH_BLOCK_RUNNER="$TRACEH_ROOT/repo/bin/traceh-run-block"
export TRACEH_OUTPUT_ROOT="$TRACEH_ROOT/runs/target_final_blind"
export TRACEH_PYTHON="$TRACEH_ROOT/envs/traceh/bin/python"

sbatch \
  --account="$HPG_ACCOUNT" \
  --array=0-7%2 \
  slurm/rollout-block-array-b200.sbatch
```

不要把 `--array` 并发写死为 8。B200 需求高，且 group allocation 可能只有 2 NGU。多 GPU tensor parallel 必须另写并审查脚本；不要用 CLI 把单卡模板临时改成 8 卡后直接提交。

## 7. 监控与失败恢复

```bash
squeue -u "$USER"
sacct -X --starttime today -o jobid,jobname,state,elapsed,allocgres,maxrss,exitcode
jobnvtop <jobid>
```

恢复原则：

1. manifest block ID 不变；
2. 新 job 写新 attempt directory；
3. 成功记录不可覆盖；
4. parser failure、OOM、timeout 都保留为真实系统事件；
5. target final block 失败后只能按冻结配置重提，不能根据 outcome 改 policy。

## 8. 数据流

```text
/blue models + manifest
  -> optional $SLURM_TMPDIR staging
  -> model-loaded episode block
  -> /blue append-only raw results
  -> all blocks complete
  -> local/CPU unified analysis
  -> immutable bundle to /orange + external backup
```

大文件与大量小文件用 Globus 的 `UFRC HiPerGator` collection 传输。Orange 只用于归档，不作为并行 job 的活跃 I/O 路径。

## 9. 官方文档

- [GPU Access](https://docs.rc.ufl.edu/scheduler/gpu_access/)
- [SLURM Partition Limits](https://docs.rc.ufl.edu/scheduler/partition_limits/)
- [Computation Rules](https://docs.rc.ufl.edu/quickstart/computation/)
- [Practical Storage](https://docs.rc.ufl.edu/quickstart/practical_storage/)
- [CUDA Usage](https://docs.rc.ufl.edu/software/apps/cuda/usage/)
- [Apptainer Usage](https://docs.rc.ufl.edu/software/apps/apptainer/usage/)
- [Globus](https://docs.rc.ufl.edu/data_transfer/globus/)

