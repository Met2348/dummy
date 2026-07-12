# Infra Graduation — Mini-Cluster 模拟器 / Topology Selector / MLPerf / TCO / Portfolio v3（Module 8 毕业 + 46-Topic 全系列收官）⭐⭐⭐⭐⭐⭐⭐⭐

> Module 8（系统与 Infra）7 专题系列的第 7 站（毕业站）。核心论文：Peter Mattson, Christine Cheng,
> Cody Coleman, Greg Diamos, Paulius Micikevicius, David Patterson, Matei Zaharia, et al. —
> *MLPerf Training Benchmark*（arXiv:1910.01500，MLSys 2020）。6 篇 lecture + 8 个纯 CPU 可直跑
> self-test（`sim/` 5 个 + `eval/` 2 个 + `portfolio_v3.py`）+ 1 篇 635 行中文论文导读。
>
> ⚠️ **`src/` 是包结构（`sim/`、`eval/` 两个子包），不是前 6 个 M8 模块的平铺脚本**：`sim/cost_model.py`
> / `sim/time_to_train.py` / `sim/topology_selector.py` / `sim/capstone_1.py` / `eval/mlperf_mock.py`
> 这 5 个用的是包内绝对导入 `from sim.common import ...`，**不能**像其它 M8 模块那样直接
> `python learning/infra-graduation/src/sim/xxx.py` 裸跑（会报 `ModuleNotFoundError: No module named
> 'sim'`，已实测复现）——这是本模块和前 6 个 M8 模块（gpu-architecture/cuda-essentials/
> kernel-engineering/cluster-networking/storage-dataops/training-orchestration）唯一的关键差异，
> 详见下方「环境配置」段的三种正确跑法，复制粘贴命令前请务必先读那一节。
>
> 论文导读：[`paper/guide_01_mlperf_training_benchmark.md`](paper/guide_01_mlperf_training_benchmark.md)（635 行，原文 PDF 同目录）

---

## 专题概览

| # | Lecture | 主题 | 核心公式 / idea | 对应代码 |
|---|---------|------|----------------|---------|
| 01 | grad-overview | M8 全系列 7 站总览 + 本模块（第 7 站/毕业站）定位 | M8 是"其他所有 module 的物理基础" | — （纯总览，无代码） |
| 02 | mini-cluster-sim ⭐ | Capstone-1：3 模型 × 6 集群 = 18 场景 time-to-train + TCO 矩阵 | `wall_days = 6·params·tokens/(peak·util)·overhead/86400` | [`sim/capstone_1.py`](src/sim/capstone_1.py) + [`sim/common.py`](src/sim/common.py) + [`sim/time_to_train.py`](src/sim/time_to_train.py) |
| 03 | topology-selector | `(model, budget, deadline)` → 最便宜可行 blueprint | 48 候选枚举 → budget/time 双过滤 → TCO 升序取最小 | [`sim/topology_selector.py`](src/sim/topology_selector.py) |
| 04 | mlperf | MLPerf Training/Inference 规则 + Capstone-2 硬件对比 | time-to-quality（非吞吐）；closed/open division | [`eval/mlperf_mock.py`](src/eval/mlperf_mock.py) + [`eval/mlperf_original_minimal.py`](src/eval/mlperf_original_minimal.py) |
| 05 | tco | TCO 三大成本 + buy-or-rent 决策框架 | capex(GPU+fabric+storage) + 3y opex | [`sim/cost_model.py`](src/sim/cost_model.py) |
| 06 | portfolio-v3 ⭐ | Capstone-3：46-topic Portfolio v3（6→7 大画像） | v2(39 题/6 画像) → v3(46 题/7 画像) | [`portfolio_v3.py`](src/portfolio_v3.py) |

**预计学时**：约 2.5-3 h（6 篇 lecture 精读约 213 行 + 635 行论文导读通读 + 8 个脚本全部跑一遍并读源码）。

---

## 学习路径

```
        L01 总览（M8 7 站定位：本模块是第 7 站/毕业站）
                |
        L02 Mini-Cluster Simulator ⭐（Capstone-1：18 场景 time-to-train + TCO）
                |
        L03 Topology Selector（(model, budget, deadline) → 最便宜可行集群）
                |
        L04 MLPerf Training/Inference（Capstone-2：5-task H100 vs B200）
                |
        L05 TCO（capex/opex 三大成本 + buy-or-rent 决策框架）
                |
        L06 Capstone-3：46-topic Portfolio v3（6→7 大画像，全系列毕业）
```

---

## 目录结构

```
infra-graduation/
├── README.md
├── runbook.yaml
├── paper/
│   ├── README.md
│   ├── 01_mlperf_training_benchmark.pdf
│   ├── guide_01_mlperf_training_benchmark.md        # 635 行中文导读
│   └── guide_01_mlperf_training_benchmark.pdf
├── lectures/
│   └── 01..06-*.md                                   # 6 篇 lecture markdown
└── src/
    ├── portfolio_v3.py             # Capstone-3：46-topic Portfolio v3 生成器（自包含，可裸跑）
    ├── sim/                        # 包：集群模拟器
    │   ├── __init__.py
    │   ├── common.py                # ClusterBlueprint/GPUSpec/FabricSpec/StorageSpec + 3 catalog（无跨包 import，可裸跑）
    │   ├── cost_model.py            # TCO：capex+opex+power（`from sim.common import ...`）
    │   ├── time_to_train.py         # time-to-train：FLOPs/peak/util+overhead（`from sim.common import ...`）
    │   ├── topology_selector.py     # 拓扑选型器（`from sim.{common,time_to_train,cost_model} import ...`）
    │   └── capstone_1.py            # Capstone-1：18 场景矩阵（同上三路 import）
    ├── eval/                        # 包：MLPerf 评测
    │   ├── __init__.py
    │   ├── mlperf_mock.py           # Capstone-2：5-task H100 vs B200（`from sim.* import ...`，跨包导入）
    │   └── mlperf_original_minimal.py  # MLPerf 论文规则复现（自包含，可裸跑）
    └── tests/
        └── test_all.py              # 聚合 8 模块 _self_test()（自己处理 sys.path，两种跑法都行）
```

**没有** `environment/` 目录（也没有 `verify_env.py`）——本模块 `src/` 下所有代码只依赖标准库，见下节。

---

## 环境配置

`src/` 下 8 个脚本全部只依赖标准库（`dataclasses` / `math` / `os` / `tempfile` / `from __future__
import annotations`），**零第三方包、零网络、零 GPU 依赖**。

**本模块和前 6 个 M8 模块（gpu-architecture/cuda-essentials/kernel-engineering/cluster-networking/
storage-dataops/training-orchestration）最大的差异**：那 6 个模块的 `src/` 是平铺脚本，互相
`from common import ...`（同目录裸导入），Python 自动把脚本所在目录插入 `sys.path[0]` 就能解析，不
需要 `PYTHONPATH`。**本模块是包结构**（`sim/`、`eval/` 两个子包），已用
`grep -nE "^from |^import " src/**/*.py` 核实 import 情况：

```text
sim/common.py                     → 无跨包 import，可裸跑
sim/cost_model.py                 → from sim.common import ClusterBlueprint
sim/time_to_train.py              → from sim.common import ClusterBlueprint
sim/topology_selector.py          → from sim.{common, time_to_train, cost_model} import ...
sim/capstone_1.py                 → from sim.{common, time_to_train, cost_model} import ...
eval/mlperf_mock.py               → from sim.{common, time_to_train, cost_model} import ...（跨包：eval 导入 sim）
eval/mlperf_original_minimal.py   → 无跨包 import，可裸跑
portfolio_v3.py                   → 无跨包 import，可裸跑
```

`sys.path[0]` 自动插入的只是**脚本自己所在的目录**（例如跑 `sim/cost_model.py` 时插入的是
`.../src/sim/`），不包含它的父目录 `.../src/`——而 `from sim.common import ...` 这行需要 `src/`
（`sim` 包的父目录）在 `sys.path` 上才能解析成"`sim` 是一个包，`common` 是它的子模块"。所以上面标
"跨包/包内绝对导入"的 5 个脚本**不能**直接裸跑（已实测复现）：

```powershell
python learning/infra-graduation/src/sim/cost_model.py
# Traceback (most recent call last):
#   File "...\sim\cost_model.py", line 3, in <module>
#     from sim.common import ClusterBlueprint
# ModuleNotFoundError: No module named 'sim'
```

`sim/time_to_train.py` / `sim/topology_selector.py` / `sim/capstone_1.py` / `eval/mlperf_mock.py`
裸跑同理报错；`sim/common.py` / `eval/mlperf_original_minimal.py` / `portfolio_v3.py` 这 3 个没有
跨包 import，直接裸跑没问题。

**三种正确跑法**（任选其一）：

1. **推荐：走 audit harness / `--runbook`**。`scripts/eric_3080ti_env_audit.py` 的 `_env_for()`
   （103-113 行）会给**每个**模块的所有命令自动注入 `PYTHONPATH=<module>/src`，所以
   `runbook.yaml` 里登记的 `python learning/infra-graduation/src/sim/cost_model.py` 这类命令通过
   harness 跑**完全没问题**（已用 `PYTHONPATH=.../src python .../sim/cost_model.py` 实测复现
   harness 行为成功）：
   ```powershell
   python scripts/eric_3080ti_env_audit.py --runbook --modules infra-graduation
   ```
2. **手动设 `PYTHONPATH` 后从仓库根跑**（PowerShell）：
   ```powershell
   $env:PYTHONPATH = "learning/infra-graduation/src"
   python learning/infra-graduation/src/sim/cost_model.py
   Remove-Item Env:\PYTHONPATH
   ```
3. **`cd` 进 `src/` 后用 `python -m` 模块调用**（不用碰 `PYTHONPATH`，已实测可行）：
   ```powershell
   cd learning/infra-graduation/src
   python -m sim.cost_model
   python -m sim.capstone_1
   python -m eval.mlperf_mock
   cd ../../..
   ```

复用仓库根 `.venv`（Python 3.13）即可，不需要额外 `pip install`。

---

## `sim/capstone_1.py` 和 `eval/mlperf_mock.py` 具体在演示什么

**`sim/capstone_1.py`（Capstone-1，18 场景矩阵）**：3 个模型规模（`8B-1T`=8B 参数训 1T token 的小
模型；`70B-5T`=70B 参数训 5T token 的中等模型；`405B-10T`=405B 参数训 10T token、对标 Llama-3 405B
规模的大模型）× 6 个集群配置（从 `8x H100` 到 `4096x B200`）交叉组合，对每个 `(model, cluster)` 都
调用 `time_to_train_days()` + `total_cost_3y()` 算出 wall-clock 训练天数和 3 年 TCO，一次性打印 18
行结果表。这是在演示一个 infra 工程师最核心的日常问题：**"给定一个要训的模型，在不同硬件配置上要
跑多久、花多少钱"**——把 L02（模拟器设计）+ L05（TCO 模型）串起来的端到端工作流。

**独立验证的洞察**：B200 vs H100 的 speedup 在 3 个交叉验证场景里都落在 2.27-2.28×，这不是巧合——
`wall_days` 公式里 `pure_compute_s = flops_total/(peak×util_pct)`，`util_pct`/`overhead_factor` 对
两种 GPU 相同、被比较的两个配置 `total_gpus` 也相同（如 `512x H100` vs `512x B200` 都是 64 节点），
比值精确约掉后只剩 `bf16_tflops` 之比：`2250/989 = 2.275`——和实测的 2.27-2.28× 精确吻合。也正因
如此，Capstone-2（`mlperf_mock.py`，5 个任务全部复用同一对 512-GPU H100/B200 集群）算出的**独立**
平均 speedup 恰好也是 2.28×，两个 capstone 的数字互相印证，不是巧合而是同一个 simulator 核心
（`sim.time_to_train`）的必然结果。

**`eval/mlperf_mock.py`（Capstone-2，MLPerf 风格 5-task 硬件对比）**：定义 5 个 MLPerf-Training 风格
任务（`llm-pretrain-8b` / `llm-pretrain-70b` / `llm-finetune-70b` / `llm-pretrain-405b` /
`llm-finetune-405b`，各带一个示意性 quality target 如 `loss<3.0` / `perplexity<5.0`，不实际训练到
该阈值——这是 mock，不是真训练），对每个任务在**两个固定集群**（`64 节点×8GPU=512 H100 + IB NDR`
vs 同规模 `512 B200 + IB XDR`）上各跑一次 `time_to_train_days` + `total_cost_3y`，报告
`speedup=h100_days/b200_days`。这是在演示真实 MLPerf Training 提交榜单"同一 benchmark 套件下比较
不同加速器代际"的方法论（对应 L04 讲的 time-to-quality 而非吞吐），但用本模块自己的模拟器算出数字，
不是真训练。5 个任务的 `days` 数值各不相同（因为 `n_params`/`n_tokens` 不同），但 `speedup` 比值
稳定在 ~2.28×，原因同上（GPU 峰值算力比值主导，与具体任务规模无关）。

---

## 关键公式（cheatsheet）

```
Time-to-train（L02，sim/time_to_train.py::time_to_train_days）：
  total_flops = 6 × n_params × n_tokens                              ← Chinchilla/Llama-3 FLOPs 公式
  wall_days = total_flops/(peak_pflops×1e15×util_pct) × overhead_factor / 86400
  （raw_comm_s 只是 informational，不计入 wall_days——假设现代 fabric >80% 通信被计算掩盖）

TCO 3y（L05，sim/cost_model.py::total_cost_3y）：
  power_kw = n_gpus×tdp_w×1.3(非GPU开销：CPU+NIC+冷却)×pue/1000
  tco_3y = gpu_capex + fabric_capex(n_nodes×$2k) + storage_capex(cap_pb×$50k) + 3×yearly_opex
  yearly_opex = power_kw × 8760h × $0.10/kWh
  → capex（gpu+fabric+storage）占 TCO ~90-91%，3y opex 仅占 ~9%（实测，见 L02 教学结论）

Topology selector（L03，sim/topology_selector.py::select）：
  48 候选 = 3 GPU(H100/H200/B200) × 4 n_nodes(16/64/256/1024) × 2 fabric × 2 storage
  过滤：tco_3y > budget 剔除；wall_days > max_days 剔除 → 按 (TCO↑, days↑, id) 排序取最小

MLPerf time-to-quality trimmed mean（L04，eval/mlperf_original_minimal.py::reported_time_to_quality）：
  required_runs 次独立 run，各自算 time_to_quality，掉最快+最慢，其余取平均
  （5-run 例：[120,125,128,130,200] 掉 120/200 → mean(125,128,130)=127.67）

B200 vs H100 speedup（本次验证独立核实）：
  speedup ≈ bf16_tflops(B200)/bf16_tflops(H100) = 2250/989 = 2.275
  ← 与 capstone_1.py 三场景实测 2.27-2.28× 、mlperf_mock.py 五任务平均 2.28× 全部吻合
```

---

## 46-Topic 全系列毕业 + 7 大画像

Portfolio v3（Capstone-3，[`portfolio_v3.py`](src/portfolio_v3.py)）汇总全部 8 个 Module 的 46 个
专题（Module 1 PEFT ×3 + Module 3 造大模型 ×8 + Module 4 改大模型 ×7 + Module 5 用大模型 ×7 +
Module 6 评测/安全 ×7 + Module 7 Agent 应用层 ×7 + Module 8 Infra/硬件层 ×7 = 46），完整清单见脚本
运行后生成的 `%TEMP%\infra_graduation_portfolio_v3.md` 或 [`lectures/06-portfolio-v3.md`](lectures/06-portfolio-v3.md)。

7 大画像（相对 Portfolio v2 的 6 大画像，新增"造 infra"）：

```
1. 造模型 — M3        2. 改模型 — M1+M4      3. 用模型 — M5        4. 评模型 — M6
5. 守模型 — M6         6. 造 agent — M7       7. 造 infra ⭐ — M8（本模块新增）
```

| | Portfolio v2（39 题，M7 毕业） | Portfolio v3（46 题，M8 毕业，本模块） |
|---|---|---|
| Modules 覆盖 | 1+3+4+5+6+7 | + **8**（GPU/CUDA/kernel/网络/存储/编排/本模块） |
| 画像数 | 6 | **7** |
| Career path | 5 | **7**（+ GPU/CUDA Engineer + HPC/Cluster Engineer） |

---

## Git 里程碑

| Tag | 内容 |
|-----|------|
| `infra-graduation` | L01-06：全模块完结 |
| `基础-graduation` | Module 8（系统与 Infra）收官 |
| `module8-complete` | Module 8 整体完成 |
| `series-v3-complete` | **全 46 专题 LLM 全栈学习马拉松完成 ⭐⭐⭐⭐⭐⭐⭐⭐** |

---

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti
> 16GB）上 V0+V1 验证通过（8/8，纯 CPU 秒级，无需 GPU；V0 全部 `v0: false` 因为脚本无 argparse，
> 跳过 `--help` 探针）。
> 一键复验本模块（harness 自动处理 PYTHONPATH，见上方「环境配置」）：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules infra-graduation
> ```

**3 个自包含脚本**（无跨包 import，可直接裸跑）：

```powershell
python learning/infra-graduation/src/sim/common.py
# [OK] sim.common (512x H100 = 506 PFLOPS, $15.4M)

python learning/infra-graduation/src/eval/mlperf_original_minimal.py
# [OK] mlperf_original_minimal (rules, trimmed mean, divisions)

python learning/infra-graduation/src/portfolio_v3.py
# [OK] portfolio_v3 (46 topics enumerated)
# ...(46-topic Portfolio v3 内容预览，前 1500 字符)...
# [full portfolio_v3.md written to C:\Users\<you>\AppData\Local\Temp\infra_graduation_portfolio_v3.md]
```

**5 个包内绝对导入脚本**（需要 `src/` 在 `PYTHONPATH` 上，见上方「环境配置」三种跑法；下面用手动
设 `PYTHONPATH` 演示）：

```powershell
$env:PYTHONPATH = "learning/infra-graduation/src"

python learning/infra-graduation/src/sim/cost_model.py
# [OK] sim.cost_model (512 H100 TCO 3y $18.1M, power 605.7kW)

python learning/infra-graduation/src/sim/time_to_train.py
# [OK] sim.time_to_train (512x H100 -> 70B-15T in 450.0d)

python learning/infra-graduation/src/sim/topology_selector.py
# [OK] topology_selector (picked 128x H100 + IB XDR 800G)

python learning/infra-graduation/src/sim/capstone_1.py
# [OK] capstone_1 (18 scenarios; 8B/8GPU 219.4d; 405B/4k H100 217.0d -> B200 95.4d)
# ...(18 行 Model | Cluster | Days | TCO 3y ($M) | Power (kW) 表)...

python learning/infra-graduation/src/eval/mlperf_mock.py
# [OK] mlperf_mock (5 tasks; avg B200/H100 speedup 2.28x)

Remove-Item Env:\PYTHONPATH
```

**关键坑注记**

- **包导入陷阱（本模块专属，前 6 个 M8 模块都没有）**：详见上方「环境配置」，5 个脚本裸跑会
  `ModuleNotFoundError: No module named 'sim'`；`runbook.yaml` 里的 `cmd` 仍按仓库惯例写成
  `python learning/infra-graduation/src/...py`（不是错误——audit harness 会自动注入
  `PYTHONPATH=<module>/src`，通过 `--runbook`/本文件跑完全没问题），但人类读者手动复制粘贴时必须
  按上面三种跑法之一来，不能直接照抄命令当裸跑用。
- **`portfolio_v3.py` 曾经不落盘任何交付物（本次已修复，真代码 bug）**：`_self_test()` 只写一个
  相对路径临时文件 `tmp_portfolio_v3.md`，读回校验内容后立即 `os.remove()` 删除；`__main__` 早先
  只调用 `_self_test()`，跑完不留下任何持久化文件——但 `lectures/06-portfolio-v3.md` 的退出条件
  checklist 明确要求"portfolio_v3.md 生成 + ≥46 topics enumerated"这个真实交付物。**仿照
  `eval-graduation/src/portfolio.py`、`agent-graduation/src/portfolio_v2.py` 的先例修复**：
  `__main__` 在 `_self_test()` 后新增真正落盘到 `tempfile.gettempdir()`
  （`%TEMP%\infra_graduation_portfolio_v3.md`）的逻辑，打印预览+完整路径，不污染仓库；
  `_self_test()` 内部"相对路径临时文件写入+立即删除"的自测机制本身不变（那是自测夹具，不是给用户
  看的交付命令）。旧版讲义命令 `python -c "sys.path.insert(...); ...write_portfolio_v3('portfolio_v3.md')"`
  除了 CWD 依赖，还会把产物写进**仓库根目录**（已实测复现：从仓库根跑该一行流，会在仓库根目录
  生成 `portfolio_v3.md`，且不受 `.gitignore` 拦截）——已改讲义为直接调用
  `python learning/infra-graduation/src/portfolio_v3.py`。
- **`lectures/06-portfolio-v3.md` "7 lectures" 计数漂移（已订正为 6）**：`01-grad-overview.md`
  "收官 7 件事"的"7"是 M8 全系列 7 个 Topic（Topic 1-6=其它 6 个 M8 模块，Topic 7=本模块自己），
  和"本模块 `lectures/` 目录有几篇文件"是两个不同的数字，早期版本的退出条件 checklist 把两者
  搞混写成"7 lectures"；`ls lectures/` 实际是 6 篇（01-06），已订正，Git 历史确认从未存在过第 7
  篇（首次提交 `3d8f12b` 就只有 6 篇）。
- **`lectures/02-mini-cluster-sim.md` 18-场景表格数字漂移（已订正）**：早期版本的表格数字是撰写时
  手估的草稿值，与 `capstone_1.py` 实测 stdout 有出入（`70B-5T`/`4096x H100` 那两行差 6 倍以上：
  文档写 120 天，实测 18.7 天；TCO/占比等数字也有出入）；已用真实 stdout 整表替换，并补充"B200/H100
  speedup 精确等于两代 GPU 峰值算力之比 2250/989=2.275"的独立验证（见上方「关键公式」）。
- **`portfolio_v3.py` 新增的落盘逻辑在 Windows 中文 locale 控制台会 `UnicodeEncodeError`（复核时独立
  发现并修复，不是原修复的一部分）**：`__main__` 打印的 preview 内容含 `⭐` 等字符，Windows 中文
  locale 下 PowerShell/cmd 默认控制台代码页是 GBK（cp936），不是 UTF-8，`print()` 直接崩
  （`'gbk' codec can't encode character '⭐'`，已在真实 PowerShell 复现，退出码 1）——这条崩溃
  只有在**不设 `PYTHONIOENCODING`/`PYTHONUTF8` 的裸终端**下才会触发，audit harness 会自动设置这两个
  环境变量（`_env_for()`），所以之前"harness 8/8 PASS"和"已在真实 PowerShell 测过"这两个说法字面
  都不算错，但都没有覆盖到"用户在自己中文 Windows 终端直接复制粘贴 README 命令"这个最常见的真实场景。
  已修：`__main__` 开头加 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`（Python
  3.7+ 可用），修复后已重新在真实 PowerShell 裸跑确认退出码 0。**同一个 bug 也存在于更早已提交的
  `agent-graduation/src/portfolio_v2.py`**（同款 `print(preview)` 模式、同款 `⭐` 内容），已用相同
  方式一并修复（独立 commit，不在本模块改动范围内，详见 `agent-graduation` 模块自己的 commit）。
  `eval-graduation/src/portfolio.py` 的 portfolio 内容不含 emoji，未命中此坑，未改动。
- 全部 8 个脚本零第三方依赖、CPU 秒级 PASS（0.12-0.17s）属正常——纯 dataclass/数值 self-test
  （非训练 demo），非假成功：`capstone_1`/`mlperf_mock` 逐项手工核实过是脚本实算（18/5 行结果表
  数字随参数变化、speedup 可独立用 GPU 峰值算力比值验证），非硬编码数字。

**测试（V2）**

```powershell
python learning/infra-graduation/src/tests/test_all.py    # 预期：=== 8/8 passed ===
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules infra-graduation --tests
```

> 注：`test_all.py` 自己在文件头部处理了 `sys.path`（`SRC = dirname(dirname(abspath(__file__)))`
> + `sys.path.insert(0, SRC)`，指向 `src/`），所以裸跑（`python
> learning/infra-graduation/src/tests/test_all.py`）和 `PYTHONPATH` 跑法两种都会成功（已实测两种
> 都是 `8/8 passed`），不受上面「包导入陷阱」影响。`test_all.py` 只有 `def main()`，没有任何
> `test_` 前缀函数 → pytest 收集会得到 `no tests ran`（已实测 rc=5，非 exit 0），audit harness 在
> 这种情况下会自动回退成 `python tests/test_all.py` 直接跑（`_run_test_command`，
> `scripts/eric_3080ti_env_audit.py` 166-185 行）。**本轮验证改过 `portfolio_v3.py`**，已用
> `--json-out`/`--md-out` 指向系统临时目录重跑确认（`tests-script:test_all.py PASS`，1/1 pass），
> 未覆盖任何已提交文件。

---

## 自测题

1. `sim/cost_model.py` / `sim/time_to_train.py` / `sim/topology_selector.py` / `sim/capstone_1.py` /
   `eval/mlperf_mock.py` 这 5 个脚本为什么不能直接 `python learning/infra-graduation/src/sim/xxx.py`
   裸跑？`sys.path[0]` 自动插入的是哪个目录，为什么不够？
2. `sim/common.py`、`eval/mlperf_original_minimal.py`、`portfolio_v3.py` 这 3 个为什么可以裸跑？
   它们和另外 5 个脚本在 import 语句上有什么本质区别？
3. `time_to_train_days()` 里 `raw_comm_s`（ring all-reduce 通信时间）为什么没有计入
   `wall_days`？这个建模假设合理吗（提示：现代 fabric 的通信/计算重叠比例）？
4. `capstone_1.py` 里 B200 vs H100 的 speedup 为什么在 3 个不同模型规模上都稳定在 2.27-2.28×？
   这个比值和哪两个具体数字直接相关（提示：不需要看 `n_params`/`n_tokens`）？
5. `topology_selector.py::select()` 枚举了多少个候选 blueprint？过滤顺序（先 budget 还是先 time）
   会不会影响最终选出的结果？
6. `total_cost_3y()` 里 storage capex 是固定 `cap_pb×$50k`，不随 `n_nodes` 变化——这对小集群
   （如 `8x H100`）的 TCO 占比有什么影响？（提示：算一下 8 GPU 的 gpu_capex 和 storage_capex 谁大）
7. `mlperf_original_minimal.py::reported_time_to_quality()` 的 trimmed mean 为什么要求
   `required_runs`（vision task 5 次，language/recommendation/rl 10 次）？丢弃最快+最慢分别在防
   什么风险？
8. `portfolio_v3.py` 修复前后，`_self_test()` 内部"相对路径临时文件写入+删除"的逻辑变了吗？为什么
   "自测机制用相对路径临时文件"可以接受，但"给用户看的交付命令用相对路径"就不行？

---

## 跨专题衔接

| 专题 | 衔接点 |
|---------|-------|
| ← `training-orchestration` | 上一站关心单集群内部怎么调度训练任务；本站把视角升到"选型层"——给定预算和 deadline，决定该造多大的集群、用什么 GPU/fabric/storage，是 M8 全系列的收束 |
| ← `eval-foundations`（M6） | L04 MLPerf 的 time-to-quality / closed-open division 方法论，和 M6 的 benchmark 设计是同一类"怎么公平比较系统"问题在不同 layer 的体现 |
| （本模块是 46 专题全系列终点，无下一站） | Portfolio v3（Capstone-3）汇总全部 8 个 Module，7 大画像收官 |

---

## 完成验收（自查）

- [ ] 6 篇 lecture 全过（01 总览 → 06 Capstone-3 Portfolio v3）
- [ ] `paper/guide_01_mlperf_training_benchmark.md` 通读一遍，能回答文末「闭卷掌握检查」14 条
- [ ] 8 个 `src/**/*.py` self-test 全部亲自跑过一遍（记得 5 个包导入脚本需要 `PYTHONPATH`/`-m`/harness，见「环境配置」）
- [ ] 能说清楚 `sim/capstone_1.py` 和 `eval/mlperf_mock.py` 分别在演示什么、为什么两者独立算出的 B200/H100 speedup 会一致收敛到 ~2.28×
- [ ] 能默写 time-to-train 公式（`6×params×tokens/(peak×util)×overhead`）和 TCO 成本占比（capex ~90%、3y opex ~9%）
- [ ] 能解释本模块比前 6 个 M8 模块多出的"包导入陷阱"是什么、三种正确跑法分别是什么
- [ ] `python scripts/eric_3080ti_env_audit.py --runbook --modules infra-graduation` 全绿（8/8）
- [ ] `python learning/infra-graduation/src/tests/test_all.py` 显示 `8/8 passed`
- [ ] `portfolio_v3.py` 真的在 `%TEMP%` 生成了 `infra_graduation_portfolio_v3.md`，打开确认 ≥46 个 topic 被枚举

---

🎓 **Module 8（系统与 Infra）7 站全部完成 → 全 46 专题 LLM 全栈学习马拉松完结（7 大画像：造模型 /
改模型 / 用模型 / 评模型 / 守模型 / 造 agent / 造 infra）。**
