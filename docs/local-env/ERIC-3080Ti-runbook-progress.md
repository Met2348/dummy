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

- 2026-06-20：建立 spec / plan / 账本三件套。下一步：Phase 0 在 `rl-foundations` 打磨 `--runbook` 工具 + 模板。**当前模块：rl-foundations（pilot，工具未就绪）。**

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
| 1 | prompt-tuning-family | M1 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | | |
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
| 12 | rl-foundations | M4 | ⬜ | 🔧 | ⬜ | ⬜ | ✅ | **PILOT**；tests 已绿 | |
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

- （暂无；pilot 后开始记录常见坑，如 Windows 路径、UTF-8、torch cu128 行为差异等）
