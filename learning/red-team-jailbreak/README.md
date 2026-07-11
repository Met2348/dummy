# Topic 5: Red-Team / Jailbreak（红队与越狱攻击）

> Module 6「评」第 5 专题 · 12 lectures · 12 notebooks · ~13h
>
> ⚠️ **安全声明**：所有代码均为**教学 mock**，
> 不针对真实生产模型，不输出有效 jailbreak prompt。
> 学习目的：理解攻击 → 才能更好防御（Topic 6）。

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 红队历史 + Anthropic 框架 | `common.py` |
| L02 | Jailbreak 4 大分类 | (lecture only) |
| L03 | GCG (Zou 2023) | `gcg_minimal.py`（+ `gcg_original_minimal.py` 论文算法形状变体） |
| L04 | PAIR (Princeton 2024) | `pair_minimal.py` |
| L05 | AutoDAN (UMD 2023) | `autodan_minimal.py` |
| L06 | Crescendo (Microsoft 2024) | `crescendo_demo.py` |
| L07 | Many-shot (Anthropic 2024) | (lecture only) |
| L08 | Prompt Injection (OWASP #1) | `prompt_injection_demo.py` |
| L09 | Prefilling attack | (lecture only) |
| L10 | 多模态 jailbreak | (lecture only) |
| L11 | JailbreakBench / HarmBench | `jailbench_runner.py` |
| L12 | **Capstone: 4×3 ASR 矩阵** | `red_team_matrix.py` |

## Tag

- `red-team-jailbreak` — Topic 5 完结

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V1 验证通过（**纯 stdlib + 1 个 toy torch 用例，无需改代码**）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules red-team-jailbreak
> ```

9 个 lecture/capstone 直跑 demo（均无需传参，mock target 上跑攻击-防御循环，CPU 秒级出结果）：

```powershell
python learning/red-team-jailbreak/src/common.py                 # L01 红队历史 + Anthropic 框架工具箱（mock target/classifier/ASR）
python learning/red-team-jailbreak/src/gcg_minimal.py             # L03 GCG mock：随机 suffix 搜索攻击 mock target
python learning/red-team-jailbreak/src/gcg_original_minimal.py    # L03 GCG 论文算法形状：toy 可微目标上的真坐标梯度替换搜索（torch，~5s 含 import）
python learning/red-team-jailbreak/src/pair_minimal.py            # L04 PAIR mock：攻击者 LLM 用模板迭代重构 query
python learning/red-team-jailbreak/src/autodan_minimal.py         # L05 AutoDAN mock：遗传算法变异 seed prompt
python learning/red-team-jailbreak/src/crescendo_demo.py          # L06 Crescendo mock：多轮渐进升级脚本
python learning/red-team-jailbreak/src/prompt_injection_demo.py   # L08 Prompt Injection：direct + indirect（工具输出隐藏注入）
python learning/red-team-jailbreak/src/jailbench_runner.py        # L11 JailbreakBench/HarmBench 风格标准化评测（4 攻击法 × target ASR）
```

**Capstone：4 攻击 × 3 target ASR 矩阵红队报告卡**：

```powershell
python learning/red-team-jailbreak/src/red_team_matrix.py
```

> 注（demo 性质，非 bug）：全部 9 个脚本无 argparse（runbook 内标 `v0: false`，跳过 `--help` 探针，直接 smoke 直跑到完成）；均为教学 mock（见页首安全声明）——mock target 用关键词触发模拟"越狱成功"，不针对真实模型、不含可用 jailbreak 文本。`gcg_original_minimal.py` 是 L03 的补充脚本：在 toy 可微目标上跑**真实**坐标梯度替换（GCG 论文算法本身的形状），而 `gcg_minimal.py` 是攻击 mock target 的随机搜索版本——两者互补，共同还原"论文怎么做"与"攻击怎么打"。

**测试（V2）**：`test_redteam.py` 聚合跑 9 个模块的 `_self_test()`：

```powershell
python learning/red-team-jailbreak/src/tests/test_redteam.py
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules red-team-jailbreak --tests
```

预期：`9/9 modules passed`。（`test_redteam.py` 无 pytest `test_*` 函数，`pytest` 会 collect-0 → 审计 harness 自动回退直接跑脚本。）

## 关键文献

- Anthropic 2022 manual red-team
- Zou et al. 2023 GCG (CMU + CAIS)
- Chao et al. 2024 PAIR (Princeton)
- Liu et al. 2023 AutoDAN (UMD)
- Russinovich et al. 2024 Crescendo (Microsoft)
- Anil et al. 2024 Many-shot (Anthropic)
- OWASP LLM Top 10 (2024)
- Chao et al. 2024 JailbreakBench
- Mazeika et al. 2024 HarmBench (CMU)

## 与 Topic 6 衔接

Topic 6 (`safety-defense`) 用同样攻击方法 + 加防御，对比 ASR 降低效果。

## 一句话

> 红队 = 找洞，懂攻击才能造好防御（Topic 6）。
