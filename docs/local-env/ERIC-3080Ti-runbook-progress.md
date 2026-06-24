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
- 2026-06-22：**Phase 0 pilot 完成**。`--runbook` 工具就绪（6 单测）；rl-foundations 全绿（V0+V1 12/12，V2 tests PASS）。修了 4 个真实 bug（见 rl-foundations 行 + 系统性问题）。
- 2026-06-22：**M1 PEFT 全部完成（3/3）**：prompt-tuning（9 demo，无需改码）、lora（19 demo，修 2 假成功，QLoRA 真跑）、adapter（13 demo + 9 skip，修文档漂移）。改用 **subagent 委派**（brief: `RUNBOOK-AGENT-BRIEF.md`），我审 diff+复验+提交。修了 brief 一个坑（V2 测试默认输出会覆盖基线，已强制 /tmp）。
- 2026-06-22：**M3 进度 5/8**：data-curation（修空语料假成功+spm崩溃+测试硬化）、transformer-deep（无改码）、moe-architecture（无改码）、ssm-hybrid（无改码）、long-context（无改码，12/12 绿；早期修的 RoPE shape/打包溢出 bug 无回归）。
- 2026-06-22：**M3 造模型全部完成（8/8）**：+ scaling-infra（修不可行案例 formatter 崩溃）、pretraining-recipe（无改码，capstone 真跑从零训练 smoke）、small-model-graduation（修 2 真 bug：capstone import 漂移 + train_variant 缺文档化 flag）。**累计 11/46。下一步：M4 改模型（6 个，trl 漂移高危），从 rlhf-classic 开始。**
- 2026-06-22：**M4 改模型进度 3/6**：rl-foundations（pilot，见上）、rlhf-classic（修 2 真 bug：reward_hacking 单元素 std→NaN 假成功变种 + capstone 缺 transformers 假成功）、dpo-family（修 1 真 bug：mock_step 对标量键 .clone() 硬崩 → 类型守卫）。**关键发现：预判的 trl API 漂移高危在 rlhf-classic/dpo-family 均未命中——两者 RLHF 三段与 8 个 PO 变体全手写、零 trl import**（grep 实证非假设）。datasets 均已用命名空间 id。**累计 14/46。下一步：process-reward（M4 #15）。**

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
| 2 | lora-family | M1 | ✅ | ✅ | — | ✅ | 🩹 | 19 demo V1 全绿；修 2 处假成功(qlora/loftq silent return→fail-fast)；**QLoRA 真能在 3080Ti 跑**(bnb 4bit NF4+TinyLlama)；V2 重跑绿(262s)。遗留：README 目录结构列了不存在文件(pissa_olora_extension 等)，待清理 | 2e67a5c.. |
| 3 | adapter-tuning-family | M1 | ✅ | ✅ | — | ✅ | ✅ | 13 demo V1 绿(12 minimal + ia3-peft)；9 个 *_adapters.py **tier:skip**(adapters 库与 transformers5.x 冲突，已 clean fail-fast)；修 README 文档漂移(漏列 adapter_original_minimal、错误 pip adapters 指示)；V2 基线绿 | 077de22.. |
| 4 | data-curation | M3 | ✅ | ✅ | ✅ | ✅ | 🩹 | V0 12 + V1 13 全绿。**重磅**：capstone mock 文档自相似→MinHash 坍缩→最终语料**空**却 exit 0(假成功)，V2 测试用 `if n>0` 守卫**掩盖**了空产出。修：mock 数据多样化 + spm vocab clamp(小语料崩溃) + 测试改硬断言。V2 重跑绿 | d879779.. |
| 5 | transformer-deep | M3 | ✅ | ✅ | ✅ | ✅ | ✅ | 18 demo V1 + 1 V0(capstone-train) 全绿，**无需改码**。GPU demo 真用 cuda；秒级 summary 是真 KV-cache 公式。tensor2tensor submodule 已排除。小漂移(README"7 tests"实6、papers/→paper/)待清理 | bfd46fd.. |
| 6 | moe-architecture | M3 | ✅ | ✅ | — | ✅ | ✅ | 13 demo V1 全绿，**无需改码**。capstone loss 60.6→12.2；秒级 summary 是真 config 计算。grouped_gemm 0.28×是诚实 CPU 结果。小漂移(README"14"实13)待清理 | 8ea09ac.. |
| 7 | ssm-hybrid | M3 | ✅ | ✅ | — | ✅ | ✅ | 10 demo V1 全绿，**无需改码**。selective-copy 真数值演示；capstone loss 62.7→6.9。mamba_lib [SKIP] 是诚实库缺失(非假成功)，手写 mamba_block 为可跑等价。V2 复跑绿。小漂移待清理 | b9e91ca.. |
| 8 | long-context | M3 | ✅ | ✅ | ✅ | ✅ | ✅ | 11 demo V1 + 1 V0(capstone --help) 全绿(12/12)，**无需改码**。RoPE/PI/NTK/YaRN/3D 纯数学；ring-naive online-softmax max diff 2.4e-07≈vanilla；NIAH/RULER 题目生成器(设计不跑模型)；秒级 PASS 均真。ring_attention_lib [SKIP] 诚实库缺失(非假成功)；capstone 默认 dry-run 诚实骨架(--train 需 HF-gated 权重+5090)。早期修的 RoPE shape/打包溢出 bug 无回归。V2 基线绿 | |
| 9 | scaling-infra | M3 | ✅ | ✅ | — | 🩹 | ✅ | 14 demo V1 全绿。修 capstone_train_estimator：175B/8×80GB 真不可行时 estimate() 返回 cost=None，但 report() 无条件 `${cost:.0f}` 崩溃(TypeError，硬崩非假成功)→加 feasibility 分支打印 n/a。ChinChilla 数学真实。V2 重跑绿 | 644c97b.. |
| 10 | pretraining-recipe | M3 | ✅ | ✅ | ✅ | ✅ | ✅ | 9 demo V1 + 1 V0 全绿，**无需改码**。capstone 用 `--max_step 3` 跑**真实从零训练**(bf16,loss~9.6,15.6s)非dry-run；3步不落盘。datasets 已用命名空间 id(fineweb-edu)。data-mixture 真采样收敛。V2 复跑绿(10) | c111564.. |
| 11 | small-model-graduation | M3 | ✅ | ✅ | ✅ | 🩹 | 🩹 | 8 V1 + 2 V0 全绿。修 2 真 bug：① graduation_capstone 从 visualize import 实际在 bench_matrix 的函数→文档命令一上来 ImportError 崩；② train_variant 缺讲义文档化的 --max_step 等 flag→`--max_step 3000` 崩，补 flag+回退 cfg+smoke 不落 ckpt。V2 重跑绿(14) | 31c158c.. |
| 12 | rl-foundations | M4 | ✅ | ✅ | ✅ | 🩹 | ✅ | **PILOT 完成**。修：capstone trl 漂移→手写PPO回退；IMDb 裸id→stanfordnlp/imdb+离线回退；右填充→左填充；ppo_gpt2_trl 假成功→fail-fast | d4b5497.. |
| 13 | rlhf-classic | M4 | ✅ | ✅ | ✅ | 🩹 | 🩹 | 6 V1 + 1 V0 全绿。修 2：① reward_hacking_demo 单元素 std→NaN→reward 全 NaN→demo 自己不演示 reward hacking(假成功变种)，改整轨迹向量化→detected:True；② capstone 缺 transformers 时 exit0 假成功→fail-fast。**trl 漂移未命中**(三段全手写无 trl)。datasets 已用 Anthropic/hh-rlhf。submodule 排除。V2 重跑绿 | ba32a98.. |
| 14 | dpo-family | M4 | ✅ | ✅ | ✅ | ✅ | 🩹 | 9 demo V1 全绿(1 argparse dpo-train + 8 纯数值 PO 变体 demo) + V0(dpo-train --help)。修 1 真 bug：capstone_dpo_comparison `mock_step` 第 2 步对上一步返回里夹带的标量键(loss/margin)做 `.clone()`→`AttributeError: 'float' object has no attribute 'clone'`(硬崩 exit1，非假成功)→加 `isinstance(v, torch.Tensor)` 类型守卫只 clone tensor。**trl 漂移未命中**(8 变体全手写无 trl)。datasets 已用 `Anthropic/hh-rlhf` 命名空间(合规)。submodule `official/repos/direct-preference-optimization` 排除。V2 重跑绿(14) | |
| 15 | process-reward | M4 | ✅ | ✅ | — | ✅ | ✅ | 7 demo V1 全绿(PRM/BoN/Math-Shepherd/MCTS/PRIME/RLVR/Capstone)，**src 无需改码**；V0 N/A(全无 argparse，v0:false)。秒级 PASS 逐一核实非 no-op(math-shepherd 打 rollout 成功率→label / mcts 打 UCT 最优路径+visits / rlvr 5 类规则奖励)。Capstone 真做 100 题×32 路 BoN 重排(mock 候选→正确答案恒高分使 BoN≈oracle，已诚实标注非 bug)。**trl 漂移未命中**(零高危 import)。**改 1 处 --env**：verify_env.py 裸 `load_dataset("gsm8k")`→`openai/gsm8k`(坑#2，原只 WARN 不阻塞，仍修正避免误导)。V2 基线绿 | |
| 16 | reasoning-r1 | M4 | ✅ | ✅ | — | ✅ | ✅ | **GRPO 中心⭐⭐⭐**。5 demo V1 全绿(grpo/rloo/reinforce++/track-a-countdown/track-b-gsm8k)，**src 无需改码**；V0 N/A(全无 argparse)。**GRPO group-baseline 实证为真**：grpo 打印 group mean≈0 + z-score adv ±0.935(mean0/std1)，与 rloo leave-one-out ±0.571 交叉验证两 baseline 数学各自正确(非 no-op)。track_a/b 是**诚实标注 mock**(不加载模型，~3.5s 纯 import 开销，复用真实 compute_group_advantage)→无需 smoke 化/skip。**trl 漂移未命中**(零高危 import)。submodule DeepSeek-R1 排除。V2 基线绿(31 测 collect 净)。**遗留文档漂移(待清理)**：讲义13/14 给 track_a/b 标了 `--algo/--total_steps` 等 flag 但脚本无 argparse→静默忽略(误导非崩)；讲义引用的 train_grpo.py/grpo_verl.py 本地不存在(完整版占位)；grpo_minimal docstring 提 --total-iters 但 import argparse 未用 | |
| 17 | rl-sota-2026 | M4 | ✅ | ✅ | — | ✅ | 🩹 | 5 demo V1 全绿(dapo/dr-grpo/vapo/genrm/capstone-ablation)。**DAPO 4 件套实证为真**(asymmetric_clip 解耦ε / dynamic_sampling 真重采 / token-level vs response-level 聚合 / overlong 分段线性)。**重磅修复(命中坑#9 算法不符论文)**：`dr_grpo.py` 原写成"MAD 归一+`−β·log|y|` 长度惩罚"，与 Sea AI Lab 论文**取向相反**(真 Dr.GRPO 是去 std 除法+去 1/\|o\| 长度归一)→**重写代码对齐论文**(advantage=R−mean 无 std；grpo vs dr 长度权重对照)+改 2 测试为验证真实性质+改 README。直跑实证:低方差组 GRPO ±1.225 vs Dr ±0.05；长度权重 GRPO[.1,.02,.005] vs Dr 恒[.005]。**trl/verl 漂移未命中**(src 零高危 import，verl 仅讲义)。mock(capstone/genrm)诚实标注。V2 重跑绿(9 测，含 3 新) | |
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
| `--runbook` 模式（扩展 audit harness）| ✅ | 已交付（d4b5497 + v0:false 增强，6 单测） |
| `runbook.yaml` 模板 | ✅ | rl-foundations（CLI）+ prompt-tuning（无参 demo）双模板 |
| README"运行验证"段模板 | ✅ | 见 rl-foundations / prompt-tuning README |
| subagent 委派 brief | ✅ | `docs/local-env/RUNBOOK-AGENT-BRIEF.md` |

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
7. **条件守卫掩盖空产出**（data-curation 实例）：测试写 `if last["n"]>0: assert ...` → 产出为空时**跳过断言**静默通过。capstone/流水线类要**硬断言最终产物非空**（`assert n>0` + token>0 + 文件存在），否则 #4 假成功会从测试侧漏过。
8. **harness 默认输出会覆盖基线**：任何 `--tests`/`--runbook` 调用必须显式 `--json-out`/`--md-out`（runbook→gitignored runbook 文件；V2 tests→`/tmp`）。默认值指向已提交的 `ERIC-3080Ti-test-{matrix.md,results.json}`，会就地覆盖。
9. **算法与所引论文不符（"能跑但教错"）**（rl-sota-2026 dr_grpo 实例；**用户已定策：修代码对齐论文**）：demo 能 exit 0、也符合自己的注释，但实现的算法与它声称复现的论文**相反/不符**（dr_grpo 写成"MAD+长度惩罚"，真 Dr.GRPO 是"去 std 除法+去长度归一"）。这是比崩溃更隐蔽的坑——跑得通所以没东西报警，学生却会内化错误概念。
   - 政策（**适用后续所有模块**）：发现"代码与所引论文矛盾"→**修代码 + 改测试 + 改文档**三件套对齐论文（不是只在 README 记差异）。先 grep 确认 blast radius（谁 import 了被改函数 + 哪些测试锁死了错行为），重写后跑直跑 demo + pytest 实证、再核 V0/V1。改测试要把"锁死错算法"的断言换成"验证论文真实性质"的断言。
   - 判据：仅当能明确指出"论文说 X、代码做了非 X"才算命中；风格/简化差异不算。拿不准 → 在报告 ESCALATE 留给编排者/用户定。
