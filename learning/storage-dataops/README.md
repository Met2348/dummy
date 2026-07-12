# Storage & DataOps — 存储分层 / Dataloader / Sharding / Checkpoint

> Module 8（系统与 Infra）7 专题系列的第 5 站。核心论文：Yang & Cong (IBM T. J. Watson) — *Accelerating
> Data Loading in Deep Neural Network Training*（arXiv:1910.01196，HiPC 2019）。
> 6 篇 lecture + 7 个纯 CPU 可直跑 self-test + 1 篇 637 行中文论文导读。
>
> ⚠️ **`src/` 下的 7 个脚本不是真实的存储系统/checkpoint I/O 代码**——没有真实 S3/Lustre/GPFS 网络调用，
> 没有 `torch.distributed.checkpoint`，**也没有向磁盘写入任何真实文件**（`checkpoint.py`/
> `capstone_ckpt_recovery.py` 的 `CkptCost` 是纯数值代价模型：只算 `sec`/`bytes_written`/`blocking`
> 三个数字，已用 grep 核实这两个文件源码里零 `open(`/`write(`/`os.` 调用），而是用可断言验证的纯 Python
> 解析模型去复现存储分层选型、dataloader 流水线加速、三种 sharding 策略的负载均衡、三代 checkpoint
> 策略的阻塞与故障恢复代价，以及《Accelerating Data Loading》论文里的 locality-aware 数据加载模型
> （distributed cache eq.7 / locality-aware eq.8 / Algorithm 1 load-balance）。
>
> 论文导读：[`paper/guide_01_accelerating_data_loading.md`](paper/guide_01_accelerating_data_loading.md)
> （637 行，原文 PDF 同目录）

---

## 专题概览

| # | Lecture | 主题 | 核心公式 / idea | 对应代码 |
|---|---------|------|----------------|---------|
| 01 | storage-tiers | RAM/NVMe/NVMe-RAID/Lustre-GPFS/S3 五层存储 BW/IOPS/延迟/容量 | `T=latency+bytes/BW`；反模式(训练数据直读S3/ckpt用NFS) | [`common.py`](src/common.py)（`TIERS` 表 6 条） |
| 02 | dataloader ⭐ | fetch→decode→augment→collate 4 阶段流水线 | 串行=Σstage×N；流水线=max(stage)/workers×N | [`dataloader.py`](src/dataloader.py) |
| 03 | sharding | Hash/Range/Round-robin 三种 sharding 策略 | `sha1(id)%N` 抗skew；`id//(total/N)` 局部性好但可能skew严重；`id%N` 简单均匀 | [`sharding.py`](src/sharding.py) |
| 04 | checkpoint | Full/Sharded/Async 三代 checkpoint 策略 | full=gather+write(blocking)；sharded=write(blocking)；async=PCIe暂存(non-blocking) | [`checkpoint.py`](src/checkpoint.py) |
| 05 | webdataset | tar shard 顺序读 vs 随机小文件 IOPS-bound | `random=n_files/iops`；`tar=total_bytes/BW`，顺序读碾压随机 IOPS | [`webdataset_style.py`](src/webdataset_style.py) |
| 06 | capstone | Capstone：70B/512GPU/7天训练 ckpt 经济学 | blocking_overhead + 故障恢复(MTBF)开销对比三种策略的 wasted% | [`capstone_ckpt_recovery.py`](src/capstone_ckpt_recovery.py) |
| — | （无对应 lecture，仅见论文导读全篇） | 复现《Accelerating Data Loading》的 cost model + locality-aware loading | `D/(pV)`训练时间、`D/R`存储下界、eq.7 distributed-cache、eq.8 locality-aware、Algorithm 1 surplus/deficit balance、梯度等价证明 | [`data_loading_original_minimal.py`](src/data_loading_original_minimal.py)（154 行，本模块最长脚本） |

**预计学时**：约 2.5–3 h（6 篇 lecture 精读，篇幅短小 + 637 行论文导读通读 + 7 个脚本全部跑一遍并读源码）。

---

## 学习路径

```
        L01 存储分层（common.py 的 6 层 TIERS：RAM→NVMe→NVMe-RAID→Lustre/GPFS→S3）
                |
        L02 Dataloader 流水线 ⭐（fetch/decode/augment/collate 4 阶段 + N-worker 并行）
                |
        L03 Sharding 策略（Hash/Range/Round-robin 的负载均衡权衡）
                |
        L04 Checkpoint 策略（Full/Sharded/Async 三代，阻塞时间递减 + 故障恢复经济学）
                |
        L05 WebDataset 流式格式（tar 顺序读 vs 随机小文件 IOPS-bound）
                |
        L06 Capstone：70B/512GPU/7天训练，三种 ckpt 策略的总浪费开销对比

   延伸阅读（论文导读专属，无对应 lecture）：
   data_loading_original_minimal.py — 复现论文 cost model + eq.7/eq.8 + Algorithm 1 load-balance
```

---

## 目录结构

```
storage-dataops/
├── README.md
├── runbook.yaml
├── paper/
│   ├── README.md                                 # 论文索引（1 篇）
│   ├── 01_accelerating_data_loading.pdf           # 原始论文 PDF
│   ├── guide_01_accelerating_data_loading.md      # 637 行中文导读
│   └── guide_01_accelerating_data_loading.pdf     # 导读渲染版 PDF
├── lectures/
│   └── 01..06-*.md                                # 6 篇 lecture markdown
└── src/
    ├── common.py                                  # Storage 数据类 + 6 层存储 BW/IOPS/延迟表
    ├── dataloader.py                               # fetch/decode/augment/collate 流水线加速模型
    ├── sharding.py                                 # hash(sha1)/range/round_robin 三种 sharding
    ├── checkpoint.py                                # full/sharded/async 三代 checkpoint 代价模型
    ├── webdataset_style.py                          # tar shard 顺序读 vs 随机小文件 IOPS-bound
    ├── data_loading_original_minimal.py             # 论文 cost model + eq.7/eq.8 + Algorithm 1（154行，最长）
    ├── capstone_ckpt_recovery.py                    # Capstone：ckpt 策略 + 故障恢复总开销
    └── tests/
        └── test_all.py                              # 聚合 7 模块 _self_test()（脚本式，无 test_ 前缀函数）
```

**没有** `environment/` 目录（也没有 `verify_env.py`）——本模块 `src/` 下所有代码只依赖标准库，见下节。

---

## 环境配置

`src/` 下 7 个脚本全部只依赖标准库（`dataclasses` / `hashlib` / `from __future__ import annotations`；
`sharding.py` 是本模块唯一额外 `import hashlib` 的脚本），**零第三方包、零网络、零 GPU 依赖**。已用
`grep -nE "^import |^from " src/*.py` 核实：

```text
common.py / dataloader.py / webdataset_style.py   → 只 dataclasses + __future__（不 import 同目录任何模块）
data_loading_original_minimal.py                  → 只 dataclasses + __future__（不 import 同目录任何模块）
sharding.py                                        → + import hashlib（标准库，仍不跨文件 import）
checkpoint.py                                       → + from common import Storage
capstone_ckpt_recovery.py                           → + from common import TIERS，
                                                        + from checkpoint import full_checkpoint, sharded_checkpoint, async_sharded
```

和 `gpu-architecture`/`cuda-essentials`/`cluster-networking` 同款模式（不同于零 import 的
`kernel-engineering`）：2 个脚本会 `from common import ...` / `from checkpoint import ...` 跨文件
import。**已实测**直接 `python learning/storage-dataops/src/X.py`（不设 `PYTHONPATH`、从 repo 根目录跑）
对全部 7 个脚本都可行——Python 会自动把脚本所在目录（`learning/storage-dataops/src/`）插入
`sys.path[0]`，同目录下的 `common.py`/`checkpoint.py` 都能被正常 import，不需要额外配置。

复用仓库根 `.venv`（Python 3.13）即可，不需要额外 `pip install`：

```powershell
python learning/storage-dataops/src/checkpoint.py
# [OK] checkpoint (full 0.70s, sharded 0.350s, async 0.009s)
```

---

## 横向对比：7 个脚本实测结果（本模块脚本直接产出，非手算）

| 脚本 | 建模对象 | 关键机制 | 本模块实测结果 |
|---|---|---|---|
| `common.py` | 6 层存储读时延 | `time_to_read = latency + bytes/BW` | 100GB 从 Lustre 读 ≈0.20s，从 S3 读 ≈100s（S3 慢 200 倍以上，符合断言 `t_s3 > 200*t`） |
| `dataloader.py` | 4 阶段流水线加速 | 串行=Σstage×N；流水线=max(stage)/workers×N | 1000 样本：naive 3400ms → pipelined(4 worker) 500ms，**speedup 6.8×**，瓶颈=decode_jpeg |
| `data_loading_original_minimal.py` | 论文 cost model + locality-aware | `D/(pV)` 训练时间 vs `D/R` 存储下界；eq.7 distributed-cache vs eq.8 locality-aware | 64→128 节点 `regular_epoch_time` 仅 10.016s→10.008s（几乎不降，验证 D/R 下界）；locality-aware(0.050s) 比 distributed-cache(4.922s) 快 **98.4×** |
| `sharding.py` | hash/range/round_robin 负载不均衡度 | `sha1(id)%N` / `id//(total/N)` / `id%N` | 10000 样本×8 shard：hash imbalance 5.1%，range **0.0%**（周期性 skew 与 shard 边界巧合对齐，非天生抗倾斜），round_robin 2.0% |
| `checkpoint.py` | full/sharded/async 三代 ckpt 代价 | full=gather+write(blocking)；sharded=write(blocking)；async=PCIe暂存(non-blocking) | 70B/512GPU：full 0.70s，sharded 0.350s，async **0.009s**（且 `blocking=False`） |
| `webdataset_style.py` | tar 顺序读 vs 随机小文件 | `random=n_files/iops`；`tar=total_bytes/BW` | 1M×100KB 样本：random 2.0s vs tar-shard 0.2s，**10.0× 加速**（无论 shard 切成 100 个 1GB 还是 10 个 10GB，总量不变则加速比不变） |
| `capstone_ckpt_recovery.py` | 70B/512GPU/7天训练 ckpt+恢复总开销 | blocking_overhead(仅计入 blocking=True 的策略) + 故障恢复(MTBF=24h) | full 2.1% vs sharded 2.09% vs async **2.08%** wasted；async 的 blocking(min) **精确等于 0.00**（非"很小"），三策略 recovery(h) 均 ≈3.50（恢复成本与 ckpt 策略无关，见下方 Runbook） |

**独立发现的洞察**（本次验证实测复算，非文档抄录）：

1. **`sharding.py` 的 range 策略 0.0% 不均衡是巧合，不是通用性质**——self-test 用的倾斜模式是
   `size = 100 + (i%50)*100`（周期 50），而 `10000/8=1250` 恰好是 50 的整数倍（25 个完整周期/shard），
   所以每个连续 range 区间看到的 size 分布完全相同，imbalance 精确为 0。我实测验证了如果把倾斜模式换成
   **按索引聚簇**（前 8000 个样本 100B + 后 2000 个样本 5000B，非周期性），range 的 imbalance 会飙到
   **363%**，而 hash(8.7%)/round_robin(0.0%，因为间隔采样天然跨越聚簇边界) 都保持在个位数——这才是
   lecture 03 表格里"Range: skew 严重"这个结论真正对应的场景。原 `_self_test()` 里一条误导性注释
   （"Range gets unlucky if data is sorted by size pattern"，但实际输出是 0.0% 最均衡）已改写为准确描述
   这个巧合成因（见 `src/sharding.py`）。
2. **`dataloader.py` 打印的 "6.8×" 不是 lecture 02 说的 "4 worker 4× 加速"，是两个效应相乘**：
   `naive_pipeline`(3400ms) 与 `pipelined`(500ms) 的比值 6.8× = **1.7×**(流水线把 fetch/augment/collate
   三个非瓶颈 stage 的耗时从关键路径上隐藏掉，3400/2000) × **4.0×**(decode_jpeg 瓶颈本身被 4 个 worker
   并行，2000/500)。lecture 02 的"4×"只讲了后半段（worker 并行瓶颈 stage），没提流水线隐藏其余 stage
   的额外收益，两者不矛盾但容易被误读成同一个数字。
3. **async checkpoint 的 blocking 开销是精确的 0.00 分钟，不是"很小的非零数"**——`capstone_ckpt_recovery.py`
   的 `total_overhead()` 用 `(cost.sec if cost.blocking else 0.0) * n_ckpts`，`async_sharded()` 把
   `blocking=False`，这个乘积项直接短路成 0，与 PCIe 暂存本身要花 0.009s 完全无关。lecture 06 原表格把这格
   写成 "0.03"（本次验证已用实测 stdout 订正为 "0.00"，见下方 Runbook 坑注记）。这也解释了为什么三种策略的
   `recovery(h)` 几乎相同（≈3.50h，差距在小数点后三位）：**async 省的是训练期间的 blocking，不是故障后的
   恢复时间**——恢复永远要重新加载一次 ckpt + 补跑半个 interval，这部分开销与 ckpt 写入策略无关。
4. **`hashlib.sha1` 的跨进程确定性已实测验证，不是"用了 hashlib 应该没事"的假设**——起 3 个独立 Python
   进程（含显式 `PYTHONHASHSEED=42` 一个变体）跑 `hashlib.sha1(str(i).encode()).hexdigest()`，5 个样本的
   十六进制摘要逐字节相同；对照组用内置 `hash("sample_0")` 才会随进程重启变化（这正是 `rag-essential`
   模块 `common.hash_embed` 踩过的坑）。`sharding.py` 的 `hash_shard()` 用 `hashlib.sha1`
   （不是内置 `hash()`），从设计上就不受 `PYTHONHASHSEED` 影响，`sharding.py` 三次独立进程重跑输出
   逐字节相同（`hash imbalance 5.1%, range 0.0%, rr 2.0%`）。

---

## 关键公式（cheatsheet）

```
存储读时延：
  T = latency_us/1e6 + (bytes/1e9) / read_gb_s

Dataloader 流水线：
  naive(串行)   T = Σ(per_sample_us for stage in stages) × N_samples
  pipelined     T = (max(per_sample_us for stage in stages) / n_workers) × N_samples
  speedup = naive / pipelined = (流水线隐藏非瓶颈stage的收益) × (worker并行瓶颈stage的收益)

Sharding：
  hash         shard = int(sha1(str(id)).hexdigest(), 16) % N     ← 确定性、跨进程稳定、抗skew
  range        shard = min(id * N // total, N-1)                  ← 局部性好，但对聚簇型倾斜数据可能严重不均
  round_robin  shard = id % N                                     ← 简单均匀，但周期性数据可能重复看同模式

Checkpoint 三代（70B BF16=140GB，512 GPU，Lustre write_gb_s=400）：
  full     gather_s = bytes/1e9/400 + 0.001；write_s = bytes/1e9/tier.write_gb_s；sec=gather+write；blocking=True
  sharded  write_s = bytes/1e9/tier.write_gb_s（OSS 聚合带宽仍是瓶颈）；sec=write_s；blocking=True
  async    per_rank = bytes // n_gpus；stage_s = per_rank/1e9/32(PCIe半速)；sec=stage_s；blocking=False

WebDataset 顺序 vs 随机：
  random(IOPS-bound)  T = n_files / iops
  tar-shard(BW-bound)  T = (n_shards × shard_bytes) / 1e9 / bw_gb_s

Capstone ckpt+恢复总开销（论文外，本模块自建模型）：
  blocking_overhead_s = (cost.sec if cost.blocking else 0.0) × n_ckpts     ← async 精确短路为 0
  recovery_s          = n_failures × (cost.sec + ckpt_interval_s / 2)      ← 三策略几乎相同
  wasted_pct           = 100 × (blocking_overhead_s + recovery_s) / (train_hours × 3600)

《Accelerating Data Loading》论文核心公式（data_loading_original_minimal.py）：
  training_time      = D / (p·V)
  data_loading_time   = D/R + D/(p·U)                      ← D/R 不随 p 下降，是常规 loader 的扩展性下界
  true_epoch_time     = max(training_time, data_loading_time)

  eq.7 distributed cache:   sample_io = (1-a)·D/R + a·D/Rc·(p-1)/p        ← 即使全缓存命中，跨节点搬运仍≈整个dataset
  eq.8 locality-aware:      sample_io = (1-a)·D/R + a·D·b/Rb              ← b(balance ratio)很小时，搬运量远小于eq.7
```

---

## 自测题

1. `common.py` 的 `time_to_read` 公式里，为什么 S3 比 Lustre 慢 200 倍以上主要是 `latency_us` 项
   （50ms vs 0.5ms）在小文件场景下的贡献，而不是 `read_gb_s` 的差距（1 vs 500 GB/s）？在什么样的
   `bytes_total` 规模下，带宽项会反过来主导？
2. `dataloader.py` 打印的 "speedup 6.8×" 可以分解成哪两个独立效应的乘积？如果 `n_workers` 从 4 提高到
   8，这两个效应各自会怎么变（提示：哪个效应有上限）？
3. Hash / Range / Round-robin 三种 sharding 各自的"最坏情况"是什么样的数据分布？为什么本模块
   `sharding.py` 自带的 `_self_test()` 数据没有暴露出 Range 的最坏情况（提示：周期 50 和 shard 数 8 的
   关系）？
4. `checkpoint.py` 的 `full_checkpoint`/`sharded_checkpoint`/`async_sharded` 三个函数都不写真实文件，
   它们各自返回的 `CkptCost.sec` 建模的是什么阶段的真实系统开销（gather？写盘？PCIe 暂存？）？
5. 为什么 `capstone_ckpt_recovery.py` 里三种 ckpt 策略的 `recovery(h)` 几乎相同（≈3.50h），但
   `blocking(min)` 差异巨大（1.96 vs 0.98 vs 0.00）？这对"选 ckpt 策略能不能降低故障恢复时间"这个问题
   说明了什么？
6. 论文的 `D/R` 存储下界具体指什么？`data_loading_original_minimal.py` 的 `CostModel` 用哪两个函数值的
   `max()` 来体现"计算和数据加载谁是瓶颈"？64→128 节点时 `regular_epoch_time` 几乎不变，对应论文的
   Figure 1 想说明什么？
7. Distributed cache（eq.7）和 locality-aware（eq.8）都假设 100% 缓存命中率（`cached_ratio=1.0`），
   为什么前者仍然要搬运"接近整个 dataset"量级的数据，后者却只需要搬 `balance_ratio` 那一小部分？
8. `balance_transfers`（对应论文 Algorithm 1）用什么贪心策略在 surplus/deficit learner 之间调度传输？
   为什么这个调度不会改变 `partitions_equivalent` 验证的全局梯度和？
9. WebDataset 的 "10×" 加速比在 `webdataset_style.py` 里对 100 个 1GB shard 和 10 个 10GB shard 给出了
   完全相同的结果——这是这个简化模型的局限（没建模什么真实开销），还是巧合？换成真实系统，shard
   太大或太小分别会引入什么额外成本？
10. 本模块 7 个脚本全部零第三方依赖、CPU 秒级跑完，且 `checkpoint.py`/`capstone_ckpt_recovery.py`
    从不写真实文件——这种"纯解析代价模型"能验证 checkpoint/data loading 系统设计的哪些性质（策略间的
    相对排序、扩展性下界的存在），又不能验证哪些（真实 Lustre OSS 拥塞、真实 GPU→host PCIe DMA 抖动、
    真实网络分区导致的故障模式、JPEG decode 的真实 CPU 成本）？

---

## Git 里程碑

| Tag | 内容 |
|-----|------|
| `storage-dataops-tiers` | L01-02：存储分层总览 + dataloader 流水线 |
| `storage-dataops-sharding` | L03：sharding 策略家族 |
| `storage-dataops-checkpoint` | L04-05：checkpoint 三代 + WebDataset 流式格式 |
| `storage-dataops` | L06：Capstone ckpt 经济学，模块完结 |

---

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上
> V0+V1 验证通过（7/7，纯 CPU 秒级，无需 GPU；V0 全部 `v0: false` 因为脚本无 argparse，跳过 `--help`
> 探针）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules storage-dataops
> ```

7 个脚本全部**无 argparse**（纯 `_self_test()` 直跑，无可调预算/flag，跑一次就是全部）：

```powershell
python learning/storage-dataops/src/common.py                          # [OK] storage_dataops.common (100GB Lustre 0.20s, S3 100s)
python learning/storage-dataops/src/dataloader.py                      # [OK] dataloader (speedup 6.8x, bottleneck=decode_jpeg)
python learning/storage-dataops/src/data_loading_original_minimal.py   # [OK] data_loading_original_minimal (model, locality, balance)
python learning/storage-dataops/src/sharding.py                        # [OK] sharding (hash imbalance 5.1%, range 0.0%, rr 2.0%)
python learning/storage-dataops/src/checkpoint.py                      # [OK] checkpoint (full 0.70s, sharded 0.350s, async 0.009s)
python learning/storage-dataops/src/webdataset_style.py                # [OK] webdataset (small shards 10.0x, large shards 10.0x)
```

**Capstone（L06）：70B/512GPU/7天训练 ckpt 经济学**

```powershell
python learning/storage-dataops/src/capstone_ckpt_recovery.py
# [OK] capstone_ckpt_recovery (full 2.1% vs async 2.08% wasted)
#
# Strategy | per-ckpt | blocking (min) | recovery (h) | wasted %
# ---------|----------|----------------|--------------|---------
# full     |    0.701 |           1.96 |          3.5 |     2.1%
# sharded  |     0.35 |           0.98 |          3.5 |    2.09%
# async    |    0.009 |            0.0 |          3.5 |    2.08%
```

**关键坑注记**

- **`lectures/06-capstone.md` 原有一处文档漂移，本次验证已修正**：结果表格 async 行的
  `blocking (min)` 原写 `0.03`，与 `capstone_ckpt_recovery.py` 当前实测（**精确 0.00**）不符——不是
  "很小的非零数"，是 `total_overhead()` 里 `(cost.sec if cost.blocking else 0.0)` 在 `blocking=False`
  时直接短路成 0，与 async 暂存本身要花的 0.009s 无关。已用实测 stdout 替换，并补充这个"精确为 0"背后
  的短路逻辑说明（而不只是改数字）。
- **`src/sharding.py` 的 `_self_test()` 原有一条误导性注释，本次验证已修正**：注释写"Range gets
  unlucky if data is sorted by size pattern"，但实际输出 range imbalance 是 **0.0%**（三种策略里最均衡的
  一个），与注释预期相反。原因是这次 toy 数据用的周期性倾斜（周期50）恰好整除每个 shard 的样本数
  （1250÷50=25 完整周期），使得每个连续 range 区间看到相同的 size 分布。已改写注释准确描述这个巧合成因，
  并补充实测过的"真正会让 range 失衡"的聚簇型倾斜场景（前 80% 小样本+后 20% 大样本 → imbalance 飙到
  363%，已用独立 Python 会话验证，未改动 `_self_test()` 本身的数据/断言——保留原数据是为了不改变已验证
  可复现的输出，只订正了对这份输出的描述）。同时清理了 `_self_test()` 里从未被用到的 `import random`/
  `random.seed(0)` 死代码（样本生成是确定性的 `i % 50`，从不调用任何 `random.*`）。
- **`src/dataloader.py` 的 `pipelined()` docstring 原有一处公式笔误，本次验证已修正**：docstring 第二行
  写 "Total time = max(stages) \* N"，遗漏了 `/ n_workers`，与函数体实际公式
  `max(...)/n_workers * n_samples` 不一致（纯 docstring 文字问题，函数体代码本身从未写错，行为无变化）。
- **`hashlib.sha1` 的跨进程确定性已实测验证**（3 个独立进程 + 显式 `PYTHONHASHSEED=42` 变体，输出逐字节
  相同），确认 `sharding.py` 的 `hash_shard()` **不是** `rag-essential` 模块 `common.hash_embed` 踩过的
  内置 `hash()` 随机加盐坑——`hashlib.*` 系列函数从设计上就与 `PYTHONHASHSEED` 无关。
- **`checkpoint.py`/`capstone_ckpt_recovery.py` 确认是纯数值代价模型，不写任何真实文件**——已读全部源码
  + grep 核实两个文件里零 `open(`/`write(`/`os.` 调用；`CkptCost` dataclass 只算 `sec`/`bytes_written`/
  `blocking` 三个数字，`bytes_written` 字段名有点误导（听起来像"已写入字节数"，实际只是"应该写入的模型总
  字节数"这个输入参数的回显，从未真正写盘）。
- **`data_loading_original_minimal.py` 确认忠实复现论文机制**：`CostModel`/`distributed_cache_io_time`/
  `locality_aware_io_time`/`balance_transfers`/`partitions_equivalent` 分别对应论文的 cost model、eq.7、
  eq.8、Algorithm 1、Theorem 1（梯度等价证明），论文导读 §14 有逐函数对应表，已核对一致，非臆造。
- **6 篇 lecture 逐一核查过，没有 `python ...` 命令**（含没有其它 M8 模块出现过的"CWD 依赖的
  `python -c "import sys; sys.path.insert(...)"` 一行流入口"写法）——6 篇都是纯 prose/表格，唯一的可跑
  命令在论文导读 §19（`tests/test_all.py`，已是仓库根相对路径直跑，无需改写）。
- 全部 7 个脚本零第三方依赖、CPU 秒级 PASS 属正常——这些是纯数值/dataclass self-test（非训练 demo），
  不存在"假成功"风险：没有网络、没有权重、没有 mock 捷径可走，每个 `[OK]` 后面的数字都是真算出来的。
- 不需要设置 `PYTHONPATH`，也不依赖 CWD：Python 自动把脚本所在目录插进 `sys.path[0]`；本模块 2 个脚本
  （`checkpoint.py`/`capstone_ckpt_recovery.py`）会互相 `from common import ...`/`from checkpoint import ...`
  跨文件 import（同 gpu-architecture/cuda-essentials/cluster-networking 模式），已实测直接
  `python learning/storage-dataops/src/X.py` 可行。

**测试（V2）**

```powershell
python learning/storage-dataops/src/tests/test_all.py    # 预期：=== 7/7 passed ===
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules storage-dataops --tests
```

> 注：`test_all.py` 是脚本式聚合器（只有 `def main()`，没有任何 `test_` 前缀函数），pytest 收集会得到
> `no tests ran`（rc=5）；audit harness 在这种情况下会**自动回退**成 `python tests/test_all.py` 直接跑
> （已实测：直跑输出 `=== 7/7 passed ===`，是真实 assert 执行，不是空心 PASS）。本轮验证**改了 2 处
> `src/*.py`**（`dataloader.py` docstring 笔误 + `sharding.py` 注释/死代码），均为纯注释/文档性质、零
> 行为改动，改后已重跑 7/7 self-test + `test_all.py` 全部逐字节输出不变，并复跑过 harness `--tests`
> （`/tmp` 输出，1/1 pass，`tests-script:test_all.py`）确认。

---

## 跨专题衔接

| 专题 | 衔接点 |
|---------|-------|
| ← `cluster-networking` | 上一站关心梯度/激活值怎么在 GPU 之间搬（all-reduce/SHARP）；本站接着看训练数据/checkpoint 怎么在存储和 GPU 之间搬（storage tiers/dataloader/sharding/checkpoint） |
| → `training-orchestration` | 本站的 dataloader 流水线、sharding、checkpoint 策略最终要被 `training-orchestration` 编排进真实分布式训练循环，才能体现出对 wall-clock/GPU 利用率的实际收益 |
| → `infra-graduation` | M8 毕业模块：把本站的存储/数据管线直觉和其余 6 站串成端到端系统设计 |

---

## 完成验收（自查）

- [ ] 6 篇 lecture 全过（01 存储分层 → 06 capstone）
- [ ] `paper/guide_01_accelerating_data_loading.md` 通读一遍，能回答文末「闭卷掌握检查」14 条
- [ ] 7 个 `src/*.py` self-test 全部亲自跑过一遍
- [ ] 能默写 dataloader 流水线的串行/流水线两条公式，并解释 6.8× 是哪两个效应相乘
- [ ] 能说出 Hash/Range/Round-robin 三种 sharding 各自在什么数据分布下表现最差
- [ ] 能解释为什么 async checkpoint 的 blocking 开销精确为 0，但故障恢复时间不受 ckpt 策略影响
- [ ] 能写出论文的 `D/R` 存储下界公式，并解释为什么它不随节点数下降
- [ ] `python scripts/eric_3080ti_env_audit.py --runbook --modules storage-dataops` 全绿（7/7）
- [ ] `python learning/storage-dataops/src/tests/test_all.py` 显示 `7/7 passed`

---

🎓 **Module 8 第 5 专题完成 → 进入 `training-orchestration`：从存储与数据管线扩展到训练编排。**
