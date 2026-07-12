# L06 — Capstone-3 Portfolio v3

## 完成清单

- 46 专题 (Module 1+3+4+5+6+7+8)
- 7 大画像 (从 6 大画像扩展，加入"造 infra")
- 90 git tags
- 47 learning 目录

## 7 大画像 (2026 LLM 全栈)

1. 造模型 — M3
2. 改模型 — M1+M4
3. 用模型 — M5
4. 评模型 — M6
5. 守模型 — M6
6. 造 agent — M7
7. **造 infra** ⭐ — M8 (新)

## 出门作品集

```powershell
# Capstone-3：Portfolio v3（直接跑；self-test 后真落盘到系统临时目录 %TEMP%，不写入仓库）
python learning/infra-graduation/src/portfolio_v3.py

# Capstone-1 + Capstone-2：sim/*.py（除 common.py 外）和 eval/mlperf_mock.py 用包内绝对导入
# `from sim.common import ...`，不能像上面 portfolio_v3.py 那样直接裸跑（会报
# `ModuleNotFoundError: No module named 'sim'`）。手动跑需要先把 src/ 加进 PYTHONPATH：
$env:PYTHONPATH = "learning/infra-graduation/src"
python learning/infra-graduation/src/sim/capstone_1.py      # Capstone-1：18 场景 time-to-train + TCO
python learning/infra-graduation/src/eval/mlperf_mock.py    # Capstone-2：5 task H100 vs B200
Remove-Item Env:\PYTHONPATH
```

> 为什么不能像其它 M8 模块一样"复制粘贴 `python xxx.py` 就行"、也不能再用旧版 `python -c
> "sys.path.insert(...)"` 一行流：后者从仓库根目录跑会把 `portfolio_v3.md` 直接写进仓库根目录污染仓库
> （`write_portfolio_v3()` 对相对路径没有防护）。完整原因和三种可选跑法（`$env:PYTHONPATH` / `cd src`
> 后 `python -m` / 走 `--runbook` harness）见 [`README.md`](../README.md) 的「环境配置」段。

## 与 Portfolio v2 的差异

| | v2 (39 题) | v3 (46 题) |
|---|----|----|
| Modules | 1+3+4+5+6+7 | + **8** |
| 画像 | 6 | **7** |
| career path | 5 | **7** (+ GPU/CUDA Engineer + HPC) |
| Hardware 知识 | inference layer | + roofline + CUDA + fabric + Slurm |
| 工业价值 | "我能造 agent" | + "我能 size 集群 + 写 kernel" |

## 退出条件 ⭐⭐⭐⭐⭐⭐⭐⭐

- [x] 6 lectures + sim/ + eval/ + portfolio_v3 全 PASS（`01-grad-overview.md`"收官 7 件事"里的
  "7"是 M8 全系列 7 个 Topic 计数，Topic 7 = 本模块自己，本模块内 `lectures/` 目录实际是 6 篇
  01-06；早期版本把两个"7"搞混，已订正为本模块真实文件数）
- [x] 7 tags (gpu-architecture / cuda-essentials / kernel-engineering /
  cluster-networking / storage-dataops / training-orchestration / infra-graduation)
- [x] 3 个总收尾 tag (`基础-graduation` + `module8-complete` + `series-v3-complete`)
- [x] portfolio_v3.md 生成 + ≥46 topics enumerated
