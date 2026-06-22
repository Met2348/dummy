# ERIC-3080Ti Runbook 验证进度账本（活动状态）

> **这是跨会话的唯一活动状态源。中途会多次 compact——任何新会话从这里接续。**

## 🧭 RESUME PROTOCOL（compact 后从这里开始）

1. 读这三个文件：
   - 标准/方法：[`docs/superpowers/specs/2026-06-20-runbook-verification-design.md`](../superpowers/specs/2026-06-20-runbook-verification-design.md)
   - 计划/循环：[`docs/superpowers/plans/2026-06-20-runbook-verification.md`](../superpowers/plans/2026-06-20-runbook-verification.md)
   - 本账本（活动状态，本文件）
2. 在下方"模块矩阵"找**第一个未完成（V1 非 ✅/⏭）的模块**，那就是下一个目标。
3. 对该模块执行 spec §10 的"每模块循环"（写 `runbook.yaml` → V0 → V1 修复 → V2 → 修 README → 更新本账本 → commit）。
4. 每完成 ~3 个模块，更新本账本顶部"最近进度"并 `git commit`。
5. 工作分支固定：`ERIC-3080Ti/paper-guides`。验证环境：repo-local `.venv`（`.\.venv\Scripts\python.exe`）。

## 📌 最近进度（每 ~3 模块更新一次）

- 2026-06-20：建立 spec / plan / 账本三件套。
- 2026-06-22：**Phase 0 pilot 完成**。`--runbook` 工具就绪（6 单测）；rl-foundations 全绿（V0+V1 12/12，V2 tests PASS）。修了 4 个真实 bug（见 rl-foundations 行 + 系统性问题）。**下一步：Phase 1 fan-out，从 `prompt-tuning-family`(M1) 开始。**

## 状态图例

| 标记 | 含义 |
|---|---|
| ⬜ | TODO 未开始 |
| 🔧 | WIP 进行中 |
| ✅ | PASS 通过 |
| 🩹 | FIXED 修过后通过（见 Notes） |
| ⏭ | SKIP（本地不可真实运行，已文档标注 + mock 路径）|
| — | N/A 不适用 |

**列含义**：README=该模块 README 的"运行验证"段是否就绪；runbook=`runbook.yaml` 是否写好；V0=文档静态；V1=smoke 跑通；V2=`src/tests` 复核。

## 模块矩阵（课程序 M1→M8，46 个）

| # | Module | Grp | README | runbook | V0 | V1 | V2 | Notes / 修复 | Commit |
|---|--------|-----|:--:|:--:|:--:|:--:|:--:|---|---|
| 1 | prompt-tuning-family | M1 | ✅ | ✅ | — | ✅ | ✅ | 9 个 minimal/peft demo V1 全绿，**无需改码**；V0 N/A（无 argparse，v0:false）；V2 基线绿 | 41461e5.. |
| 2 | lora-family | M1 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | tests 280s（重）| |
| 3 | adapter-tuning-family | M1 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | tests 127s（重）| |
| 4 | data-curation | M3 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 5 | transformer-deep | M3 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | submodule: tensor2tensor | |
| 6 | moe-architecture | M3 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 7 | ssm-hybrid | M3 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 8 | long-context | M3 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 9 | scaling-infra | M3 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 10 | pretraining-recipe | M3 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 11 | small-model-graduation | M3 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 12 | rl-foundations | M4 | ✅ | ✅ | ✅ | 🩹 | ✅ | **PILOT 完成**。修：capstone trl 漂移→手写PPO回退；IMDb 裸id→stanfordnlp/imdb+离线回退；右填充→左填充；ppo_gpt2_trl 假成功→fail-fast | d4b5497.. |
| 13 | rlhf-classic | M4 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | submodule: instruct-hf | |
| 14 | dpo-family | M4 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | submodule: DPO | |
| 15 | process-reward | M4 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 16 | reasoning-r1 | M4 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | submodule: DeepSeek-R1；env 145s | |
| 17 | rl-sota-2026 | M4 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | verl 可选栈 | |
| 18 | multimodal-agent | M4 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | VLM 可选栈 | |
| 19 | inference-engine-core | M5 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 20 | sglang-radixattention | M5 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 21 | speculative-decoding | M5 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 22 | quantization-deploy | M5 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | bitsandbytes | |
| 23 | distributed-inference | M5 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 24 | production-serving | M5 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | fastapi/uvicorn | |
| 25 | serving-graduation | M5 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 26 | eval-foundations | M6 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 27 | reasoning-eval | M6 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 28 | agent-code-eval | M6 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 29 | llm-judge-arena | M6 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 30 | red-team-jailbreak | M6 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 31 | safety-defense | M6 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 32 | eval-graduation | M6 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 33 | agent-foundations | M7 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 34 | rag-essential | M7 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 35 | tool-use-mcp | M7 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 36 | multi-agent-orchestration | M7 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 37 | agent-memory-context | M7 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | blake2b mock embed | |
| 38 | agent-framework-stack | M7 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 39 | agent-graduation | M7 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
| 40 | gpu-architecture | M8 | ❌→写 | ⬜ | ⬜ | ⬜ | ⬜ | **缺 README**；test_all.py | |
| 41 | cuda-essentials | M8 | ❌→写 | ⬜ | ⬜ | ⬜ | ⬜ | **缺 README**；test_all.py | |
| 42 | kernel-engineering | M8 | ❌→写 | ⬜ | ⬜ | ⬜ | ⬜ | **缺 README**；submodule: flash-attn | |
| 43 | cluster-networking | M8 | ❌→写 | ⬜ | ⬜ | ⬜ | ⬜ | **缺 README** | |
| 44 | storage-dataops | M8 | ❌→写 | ⬜ | ⬜ | ⬜ | ⬜ | **缺 README** | |
| 45 | training-orchestration | M8 | ❌→写 | ⬜ | ⬜ | ⬜ | ⬜ | **缺 README**；ray 可选栈 | |
| 46 | infra-graduation | M8 | ❌→写 | ⬜ | ⬜ | ⬜ | ⬜ | **缺 README**；test_all.py | |

## 全局任务（非单模块）

| 任务 | 状态 | Notes |
|---|:--:|---|
| 根 `README.md` | ⬜ | Phase 2，知识组织框架 + 怎么跑/验证 |
| `--runbook` 模式（扩展 audit harness）| ⬜ | Phase 0 pilot 产出 |
| `runbook.yaml` 模板 | ⬜ | Phase 0 pilot 产出 |
| README"运行验证"段模板 | ⬜ | Phase 0 pilot 产出 |

## 发现的系统性问题（跨模块复用的坑，随时追加）

这些坑会在 fan-out 中**反复出现**，遇到同类直接套用同款修法：

1. **trl API 漂移**（dpo/rlhf/r1/rl-sota 高危）：trl 1.5.x 移除经典 `PPOConfig`/`PPOTrainer` 情感微调 API。
   - 症状：`cannot import name 'PPOConfig' from 'trl'`。`verify_env` 的 `trl>=0.11` 检查**会误判通过**。
   - 修法：① 静默 `except: print; return` 一律改 **fail-fast**（`raise SystemExit(1)`），杜绝"假成功 exit 0"；
     ② 优先回退到模块自带的手写/minimal 实现（真实可跑）；README 标注 trl 版本要求。
2. **datasets 裸别名失效**：`datasets 4.x+` 移除裸 id，`load_dataset("imdb")` 报 `Repository id must be 'namespace/name'`。
   - 修法：用命名空间 id（`stanfordnlp/imdb` 等）+ `try/except` 回退内置小样本（让 smoke 不依赖大下载）。
3. **decoder-only 批量生成右填充**：HF 警告 `right-padding detected, set padding_side='left'`。
   - 修法：批量 `generate` 前 `tokenizer.padding_side = "left"`。
4. **"假成功"反模式**：脚本捕获缺依赖后 `print + return` → exit 0 → harness 记 PASS、学生以为跑通实则 no-op。
   - 政策：缺依赖/缺数据一律显式失败或真实回退，**绝不静默 return 成功**。
5. **机器休眠污染超时**：`subprocess timeout` 是墙钟；挂起期间休眠会把快任务记成超 timeout（如 51920s）。
   - 应对：异常的超大 elapsed 不代表真慢，清醒态短超时复测即可判定。
6. **Bash 工具吃反斜杠**：`.\.venv\Scripts\python.exe` 在 Bash 工具里要写成 `.venv/Scripts/python.exe`（正斜杠）。
