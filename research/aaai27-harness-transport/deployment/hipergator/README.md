# TRACE-H 在 UF HiPerGator 上的运行说明

**2026-07-16 状态：** GPU preflight、GitHub sparse clone、`uv.lock` 环境、Slurm array 与 ALFWorld `NONE` baseline block runner 已可迁移。当前 runner 只允许 `qwen_alfworld_baseline`，尚未接入 TRACE-H Ours 和完整 paper baselines，因此 H0/H1 可开始，sealed target-final 不可开始。

总资源决策见[本地与 HiPerGator 分阶段方案](../local-vs-hipergator-execution-plan-zh.md)。
完整迁移边界与发布门见[GitHub 到 HiPerGator 迁移验收](../github-hipergator-handoff-20260716-zh.md)。

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
cd "$TRACEH_ROOT/repo"
```

使用独立 private repository；不要把待投稿增量推送到当前公开的 `Met2348/dummy`：

```bash
export TRACEH_REPO_URL=<private-repository-url>
git clone --depth 1 --branch main "$TRACEH_REPO_URL" traceh
cd traceh

export TRACEH_PROJECT_ROOT="$TRACEH_ROOT/repo/traceh"
python3 "$TRACEH_PROJECT_ROOT/deployment/hipergator/validate_release.py"
```

Python/uv cache 也放 Blue，不占 40GB Home。环境按冻结 lock 创建：

```bash
cd "$TRACEH_PROJECT_ROOT/deployment/hipergator"
bash bootstrap-env.sh
export TRACEH_PYTHON="$TRACEH_ROOT/envs/traceh/bin/python"
```

H1 smoke 可下载 Qwen3-4B 与 ALFWorld，并生成内容哈希 manifest：

```bash
export TRACEH_MODEL_REPO=Qwen/Qwen3-4B
export TRACEH_MODEL_NAME=Qwen3-4B
export TRACEH_MODEL_REVISION=main  # H1 smoke；正式实验替换为 immutable commit
bash prepare-assets.sh

export TRACEH_MODEL_PATH="$TRACEH_ROOT/models/Qwen3-4B"
export TRACEH_CHECKPOINT_MANIFEST="$TRACEH_ROOT/manifests/assets/Qwen3-4B.manifest.json"
export TRACEH_ALFWORLD_DATA="$TRACEH_ROOT/data/alfworld"
```

若模型和数据已通过 Globus 放入 `/blue`，跳过下载，直接运行 `make_asset_manifest.py`。不要把 Hugging Face token 写进脚本或 manifest；`hf` 使用用户已有 credential。

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
export TRACEH_PROJECT_ROOT="$TRACEH_ROOT/repo/traceh"
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

Slurm array 通过 `experiments/scripts/run_cluster_block.py` 执行一个 TSV block：

```text
python run_cluster_block.py \
  --manifest <absolute path> \
  --block-index <zero-based array index> \
  --output-dir <new directory>
```

runner 当前保证：

- 一个 block 内只加载一次模型；
- 只允许登记过的 Python entrypoint，不接受任意 shell command；
- 校验 phase、information class、policy、precision、seed 与 action budget；
- append-only 写 episode records；
- output directory 非空时拒绝覆盖；
- 失败时非零退出；
- 记录 git/model/env/GPU/Slurm metadata；
- 冻结 SHA-256 必须与 job spec 指向的 checkpoint/task/prompt/freeze artifact 内容一致；
- target-final 还必须通过显式 freeze acknowledgement。

[示例 block manifest](block-manifest.example.tsv)和[示例 job spec](specs/alfworld-baseline.example.json)用于 30-episode baseline block。将它们复制到 `/blue/.../manifests` 后填写环境变量；正式 manifest 禁止保留 `AUTO`。

提交前先做不加载模型的命令构造检查：

```bash
DRY_RUN_DIR="$TRACEH_ROOT/runs/dry-run-$(date +%s)"
mkdir -p "$DRY_RUN_DIR"
"$TRACEH_PYTHON" "$TRACEH_PROJECT_ROOT/experiments/scripts/run_cluster_block.py" \
  --manifest "$TRACEH_BLOCK_MANIFEST" \
  --block-index 0 \
  --output-dir "$DRY_RUN_DIR" \
  --project-root "$TRACEH_PROJECT_ROOT" \
  --dry-run
```

## 5. L4 rollout array

```bash
export TRACEH_PROJECT_ROOT="$TRACEH_ROOT/repo/traceh"
export TRACEH_BLOCK_MANIFEST="$TRACEH_ROOT/manifests/source-l4.tsv"
export TRACEH_OUTPUT_ROOT="$TRACEH_ROOT/runs/source_branch"
export TRACEH_PYTHON="$TRACEH_ROOT/envs/traceh/bin/python"

# 12 blocks，最多同时使用 2 GPU；按 slurmInfo 修改 %2。
sbatch \
  --account="$HPG_ACCOUNT" \
  --array=0-11%2 \
  --output="$TRACEH_ROOT/logs/l4-%A_%a.out" \
  slurm/rollout-block-array-l4.sbatch
```

若要把 checkpoint stage 到 node-local flash：

```bash
export TRACEH_MODEL_SOURCE="$TRACEH_ROOT/models/Qwen3-8B"
```

模板会复制到 `$SLURM_TMPDIR/model` 并设置 `TRACEH_MODEL_PATH`。block 太短时不要 stage 大模型，复制成本可能超过计算。

## 6. B200 rollout array

```bash
export TRACEH_PROJECT_ROOT="$TRACEH_ROOT/repo/traceh"
export TRACEH_BLOCK_MANIFEST="$TRACEH_ROOT/manifests/target-b200.tsv"
export TRACEH_OUTPUT_ROOT="$TRACEH_ROOT/runs/target_final_blind"
export TRACEH_PYTHON="$TRACEH_ROOT/envs/traceh/bin/python"

sbatch \
  --account="$HPG_ACCOUNT" \
  --array=0-7%2 \
  --output="$TRACEH_ROOT/logs/b200-%A_%a.out" \
  slurm/rollout-block-array-b200.sbatch
```

正式 `target_final` 还必须设置：

```bash
export TRACEH_FROZEN_POLICY_SHA256=<manifest 中的 freeze_hash>
```

不匹配时 runner 会在加载模型前拒绝运行。

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
