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
python -c "import sys; sys.path.insert(0,'learning/infra-graduation/src'); from portfolio_v3 import write_portfolio_v3; print(write_portfolio_v3('portfolio_v3.md'))"

# Capstone-1
python learning/infra-graduation/src/sim/capstone_1.py

# Capstone-2
python learning/infra-graduation/src/eval/mlperf_mock.py
```

## 与 Portfolio v2 的差异

| | v2 (39 题) | v3 (46 题) |
|---|----|----|
| Modules | 1+3+4+5+6+7 | + **8** |
| 画像 | 6 | **7** |
| career path | 5 | **7** (+ GPU/CUDA Engineer + HPC) |
| Hardware 知识 | inference layer | + roofline + CUDA + fabric + Slurm |
| 工业价值 | "我能造 agent" | + "我能 size 集群 + 写 kernel" |

## 退出条件 ⭐⭐⭐⭐⭐⭐⭐⭐

- [x] 7 lectures + sim/ + eval/ + portfolio_v3 全 PASS
- [x] 7 tags (gpu-architecture / cuda-essentials / kernel-engineering /
  cluster-networking / storage-dataops / training-orchestration / infra-graduation)
- [x] 3 个总收尾 tag (`基础-graduation` + `module8-complete` + `series-v3-complete`)
- [x] portfolio_v3.md 生成 + ≥46 topics enumerated
