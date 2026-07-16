# GitHub 到 HiPerGator 迁移验收

- 日期：2026-07-16
- 当前结论：H0/H1 已具备可迁移执行入口；正式 sealed target-final 尚未具备
- 当前 remote：`https://github.com/Met2348/dummy.git`，已核验为 public，不用于推送待投稿增量

## 1. 可立即迁移的部分

当前 GitHub 交付层包含：

1. `uv.lock` 与 Python 3.11/3.12 约束；
2. `/blue` 下的环境、缓存、模型、数据、manifest、runs 与 logs 目录约定；
3. L4/B200 GPU preflight；
4. 允许列表式 block runner，不接受 manifest 注入任意 shell 命令；
5. ALFWorld `NONE` baseline 的 BF16/NF4、offset 分块与 append-only 输出；
6. manifest phase/information-class 一致性检查；
7. target-final 的 checkpoint/task/prompt/freeze 四类 SHA-256 与显式 freeze acknowledgement；
8. git、Python package、GPU、Slurm、manifest 与 job-spec 元数据；
9. GitHub Actions 单测和 portability audit；
10. Windows 到 Linux 的 LF 换行约束。
11. 模型和 ALFWorld 数据的内容寻址 SHA-256 manifest。

因此可以立即在 HiPerGator 执行：allocation 核验、环境安装、B200 preflight、单模型 BF16 load、30-episode baseline throughput block 与 offset array。

## 2. 仍未完成的部分

当前 block runner 只允许 `qwen_alfworld_baseline`，即 `policy_id=NONE`。以下内容不能宣称已迁移完成：

- TRACE-H adaptive branch-race policy 的真实 LLM-agent target executor；
- MASA、SkillAdaptor、Offline-RL Harness 等公开系统的统一 B200 adapter；
- 预注册 task/prompt/checkpoint/freeze hashes 的正式 target-final manifest；
- 真正 freeze 后、对结果盲化的 sealed final；
- 集群上 16 B200 的实测吞吐和 GPU-hour 预算。

这不是文档性缺口，而是论文 final 的核心工程工作。H0/H1 通过后，下一开发任务应是把 `TRACE-H Ours + 6 个 paper baselines` 接入同一 block contract，而不是先扩 array 数量。

## 3. GitHub 传输策略

当前 monorepo 的本地 Git object pack 约 3.86 GiB，并包含其他学习项目和历史 PDF。独立 TRACE-H 交付物只有约十几 MiB，应发布到 private repository。AAAI-27 允许非匿名在线初稿，但正式投稿仍为 double blind；为集群迁移没有必要增加身份暴露。

本地先把本目录变成独立 branch：

```bash
git subtree split \
  --prefix=research/aaai27-harness-transport \
  -b traceh-private-release
git remote add traceh-private <private-repository-url>
git push traceh-private traceh-private-release:main
```

HiPerGator 只 clone 独立 private repo：

```bash
export HPG_ACCOUNT=<group>
export TRACEH_ROOT=/blue/$HPG_ACCOUNT/$USER/trace-h
mkdir -p "$TRACEH_ROOT/repo"
cd "$TRACEH_ROOT/repo"

git clone --depth 1 --branch main <private-repository-url> traceh
```

如果使用专门 release branch，把 `--branch main` 替换为冻结 branch/tag。正式实验不得追踪移动的 `main`；应在 preflight 之后创建 immutable tag，并把 commit SHA 写入 manifest bundle。

## 4. 上集群后的第一轮命令

```bash
export TRACEH_PROJECT_ROOT="$TRACEH_ROOT/repo/traceh"
python3 "$TRACEH_PROJECT_ROOT/deployment/hipergator/validate_release.py"

cd "$TRACEH_PROJECT_ROOT/deployment/hipergator"
bash bootstrap-env.sh

# H1 smoke 资产；正式实验把 revision 固定为模型仓库 commit
export TRACEH_MODEL_REPO=Qwen/Qwen3-4B
export TRACEH_MODEL_NAME=Qwen3-4B
export TRACEH_MODEL_REVISION=main
bash prepare-assets.sh

export TRACEH_PYTHON="$TRACEH_ROOT/envs/traceh/bin/python"
sbatch \
  --account="$HPG_ACCOUNT" \
  --partition=hpg-b200 \
  --cpus-per-task=14 \
  --mem=112G \
  --output="$TRACEH_ROOT/logs/preflight-%j.out" \
  slurm/gpu-preflight.sbatch
```

只有 preflight 输出 `cuda_available=True`、GPU 为 B200、capability 为 `(10, 0)`，并且 `torch/transformers/alfworld` 可导入，才进入 30-episode throughput block。

## 5. 发布门

迁移前必须同时满足：

- 需要迁移的源代码与 compact audit artifacts 已 commit；
- GitHub remote 上能看到目标 commit/tag；
- `validate_release.py` 通过；
- `pytest` 全通过；
- GitHub Actions 绿色；
- cluster clone 后 `git rev-parse HEAD` 等于冻结 SHA；
- formal manifest 不含 `AUTO`；
- target-final 只有在 `TRACEH_FROZEN_POLICY_SHA256` 与 manifest 一致时才能启动。

当前本地代码通过以后，仍需执行一次“只提交 TRACE-H 迁移文件”的 commit，并推到新建的 private repository。未完成这一步前，HiPerGator 还拉不到本轮迁移代码。
