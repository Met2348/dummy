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
- 2026-06-24：**M4 改模型全部完成（7/7）**：+ process-reward（无 src 改码，修 1 处 --env gsm8k 裸 id）、reasoning-r1（GRPO⭐⭐⭐ 无改码，group-baseline 实证为真）、rl-sota-2026（**重磅：dr_grpo 算法与论文取反→重写对齐论文**，DAPO 4 件套实证为真）、multimodal-agent（毕业，4 算法 WebSearch 核保真全过）。**新增系统性坑 #9「算法与所引论文不符」+ 用户定策「修代码对齐论文」**（dr_grpo 是迄今最隐蔽的坑：能跑、符合自注释，却教错算法）。trl 漂移 7/7 全未命中（全系列手写）。**累计 18/46。下一步：M5 用模型（7 个，inference-engine-core 起；vllm/sglang 重型栈高危，预计多个需缩小版或 skip）。**
- 2026-06-24：**M5 用模型全部完成（7/7）**：inference-engine-core（修 vllm_compare 假成功#4）、sglang-radixattention（**新增坑#10**：sglang_compare 两路径相同却报 +83% 增益→修；radix/jump-forward 保真）、speculative-decoding（只 2/11 有 __main__，库模块正确排除）、quantization-deploy（**坑#10 重写 capstone**：硬编码 paper 数字→真跑 6 量化器）、distributed-inference（disagg #10 重写为 interference 模型；7 脚本补 __main__）、production-serving（server 无挂起；5 脚本补 __main__）、serving-graduation（**整模块 0 入口**→8 个全补 __main__/兼容脚本跑）。**关键模式：vllm/sglang/triton 重型栈本地全没装，但本系列几乎全是手写模拟——重型栈漂移基本不命中，真正高发的是坑#10(mock 不真演示)+缺 __main__(整模块无入口)。**3 个 subagent 被 429 限额打断，distributed/serving 两个 inline 完成。**累计 25/46。下一步：M6 评测/安全（7 个，eval-foundations 起）。**

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
| 18 | multimodal-agent | M4 | ✅ | ✅ | — | ✅ | ✅ | **M4 毕业模块**。6 demo V1 全绿(vlm-r1/vision-r1/s1-budget/safe-rlhf/unified-view/capstone)，**src 无需改码**。**算法保真度逐一核(含 WebSearch 查 s1 arXiv:2501.19393 + Safe-RLHF Dai2024)：4 个全忠实于论文**(VLM-R1 三 reward 正确；Vision-R1 HFRRF 硬奖励+z-score GRPO+PTST；s1 双侧 budget forcing+Wait 注入；Safe-RLHF Lagrangian dual ascent 符号正确，仅"cost model"术语用 harmless reward 替代是简化非矛盾)→**#9 未命中**(与 dr_grpo 取反成对照)。**trl/VLM 漂移未命中**(真 import 仅 re/torch；transformers/PIL 仅在 setup_code 返回字符串里不执行)。mock/伪代码诚实标注。V2 基线绿。遗留小漂移：README"待续"段把 s1/safe-rlhf 列为未落地实则已有(待清理) | |
| 19 | inference-engine-core | M5 | ✅ | ✅ | ✅ | ✅ | 🩹 | 手写 mini-vLLM。9 demo V1 + 2 V0(mini-vllm/vllm-compare --help) 全绿。**修 1 假成功(坑#4)**：vllm_compare 缺 vllm 原 `print("skipping")+return[]→exit0`(harness 记假 PASS)→改 `raise SystemExit` fail-fast(指路 mini_vllm 无依赖主路径)。**paged attention 数值对齐 naive**(max abs diff=0.0)；**triton 缺失有真 torch 回退**(paged_attention_triton→paged_attention_torch，bit-identical 非假成功)。vllm-compare 设 tier:skip(本机无 vllm+需~1GB Qwen，Win 难装)，只 V0 探入口。scheduling_policies 无 __main__(纯库归 V2)。修 README"21测试"→27(实际 pytest)。V2 重跑绿(27) | |
| 20 | sglang-radixattention | M5 | ✅ | ✅ | — | ✅ | 🩹 | 9 demo V1 全绿(radix-tree/jump-forward/grammar-fsm/constrained/frontend/agent-patterns/agent-server/original-minimal/sglang-compare)，全无 argparse。radix_tree 真前缀匹配+分裂+LRU(hit_rate 实证)；jump_forward 真沿 FSM 确定边跳跃(保真过#9)；agent_server 是进程内 32-agent 模拟(非网络服务，正常退出)。**修 1 自相矛盾 bug(坑#10)**：sglang_compare 的 `cost_vllm`/`cost_sglang` 公式完全相同→两列恒相等，却硬编码 +83.3% gain(相等却报 83%)。修：cost_vllm fork 分支改 `fork_k·(prefix+suffix)`(独立重 prefill，与 gain_pct 分母一致)，gain 改为纯由两列推出(1−s/v)，未建模优势改定性列。现 tot_8way 真显 8400 vs 1400=+83.3%。加 1 测锁 fork 严格更省。README 测试数 31→32。V2 重跑绿(32) | |
| 21 | speculative-decoding | M5 | ✅ | ✅ | — | ✅ | ✅ | **src 无需改码**。11 脚本里**只 2 个有 `__main__`**(capstone_eagle3 / speculative_original_minimal)→runbook 只登记这 2 个 V1(2/2 绿)；其余 9 个无 __main__ 是库模块(直跑 no-op exit0=坑#4 假成功)→正确排除归 V2，README 点名解释。**预防性命中坑#4**(没误登记库模块)。classic spec-decode 接受率**真实**(code 0.75 非 1.0，medusa 0.144，eagle1 0.38)；original_minimal **枚举证明 exact output dist==target**(零 bias 定理)。EAGLE 确特征层(toy 无真模型，noisy-target 作合成代理)。#9 未命中。**修文档 bug**：README 原"运行"块用 CWD 依赖的 `python -c sys.path.insert` → 改 repo-root 相对 + 标准 pytest/--runbook。V2 基线绿(19 测) | |
| 22 | quantization-deploy | M5 | ✅ | ✅ | — | ✅ | 🩹 | 10 脚本只 2 个 `__main__`(capstone_quant_zoo/gptq_original)→runbook 登记这 2 个 V1(2/2 绿)，8 库模块归 V2(坑#4 预防)。**重磅修(坑#10)**：capstone 旧版是**一串硬编码 paper 数字、零量化调用**(自称"6-variant zoo"却没跑任何量化器)→**重写**：在 toy 权重层(含 salient 通道)真跑 6 个量化器(int8/GPTQ/AWQ/NF4/FP8/SmoothQuant)、打印**实算** recon-MSE、压缩比=16/bits 推出。实证排序合理(8bit int8 0.095/SmoothQuant 0.084 ≪ 4bit GPTQ12.1<AWQ16.2<NF4 17.3)。测试同步重写(旧只断言硬编码字面量在场→改断言真实性质:全调用/误差有限/8bit<4bit/GPTQ≤NF4/压缩比=16/bits)。**算法保真#9 全过**(GPTQ 真 Hessian+Cholesky+误差补偿/AWQ 真激活感知 scaling/SmoothQuant 真难度迁移)。bnb_int4=手写 NF4 非真 bnb。修文档 bug(CWD 依赖命令+测试数14→23)。V2 重跑绿(23) | |
| 23 | distributed-inference | M5 | ✅ | ✅ | — | ✅ | 🩹 | 8 demo V1 全绿(tp/pp/ep/disagg/kv-transfer/routing/distserve/capstone)，全单进程模拟(无 torch.distributed/多卡)。**8 脚本里 7 个原缺 `__main__`**(直跑无输出)，按"*_demo 名应可跑"判断→补 `demo()`+`__main__` 使名副其实(7 个均补)。**disaggregated_mock 原是 #10 弱 mock**(colocate/disagg 的 TPOT 相同，增益全靠硬编码 /4 vs /8 常数)→重写为**由 prefill/decode interference 模型推出增益**(colocate TPOT 9.4 被 prefill 拖慢，disagg 8.0 无干扰；增益随 prompt 104%→185%)，已有测试 test_disagg_near_better 实证。TP/PP/EP/DistServe 保真过#9。修文档(CWD 依赖命令+16→20测试数)。⚠️**subagent 跑到一半 429 限额中断**(只改完 7 src 未写 runbook/README)，我 inline 审 diff+验证(8 脚本真跑+20 测过)+补全 runbook/README+提交。V2 重跑绿(20) | |
| 24 | production-serving | M5 | ✅ | ✅ | — | ✅ | 🩹 | 6 demo V1 全绿(cost/metrics/sse/clipper/openai-protocol/trtllm)。**预判的 server 挂起风险未现**：openai_api_server 协议层与 FastAPI 解耦(`app=make_app()` 惰性构建，无 uvicorn.run/while True→不阻塞)；fastapi/uvicorn/prometheus/sse 实测都装了。**5 个原缺 `__main__`**(直跑无输出)→补 `demo()`+`__main__`(cost_calc 算 \$/M-token+缓存省+路由；metrics 渲染 Counter/Histogram+p50/p95；sse 编解码往返；clipper AIMD+ensemble+EXP3；openai 协议层 demo 不起服务)。trtllm_build 本有 __main__。修文档(CWD 依赖测试命令+错误 uvicorn 模块路径+22→27)。**inline 完成**(429 限额未复，避免再派 subagent)。V2 重跑绿(27) | |
| 25 | serving-graduation | M5 | ✅ | ✅ | — | ✅ | ✅ | **M5 毕业模块**。8 demo V1 全绿(6 支持 + 2 capstone)。**整模块原 0 可运行入口**：6 个支持脚本(agent/embedding/router/scorecard/budget/vlm)全缺 `__main__` + capstone1(r1_tiny_deploy/serve)也缺 → 逐个补 demo()/__main__；capstone2(graduation_e2e/run)原相对 import 只能 `python -m`，加 try/except 使其直接当脚本也能跑(默认写 tempdir 不污染 repo)。embedding 诚实标注 sha256 占位非语义。修文档(CWD 依赖+`python -c` hack→直跑命令，20→23 测试数)。**inline 完成**(429 未复)。V2 重跑绿(23) | |
| 26 | eval-foundations | M6 | ✅ | ✅ | — | ✅ | 🩹 | 10 demo V1 全绿(mmlu/mmlu-pro/bbh/commonsense/truthfulqa/contamination/helm/lm-eval-adapter/pipeline/common)，全有 __main__。**mock runner 真算 acc**(build_samples→model→抽字母→比 gold→accuracy 真流水线，无硬编码分数)；contamination 真检测(ngram_overlap clean 0.000 vs leaked 1.000 + canary 命中)。**#10 预防性硬化**：7 个脚本 __main__ 原只 print "self-test: OK"(弱于宣称)→补可见 demo 输出(random baseline/真 overlap)，纯加 print 零算法改动(51+/1-)。common.py 是库 self-test，登记 runbook 时注明性质。修文档(CWD 依赖 capstone 命令)。subagent 完成(429 已恢复)。V2 重跑绿(10) | |
| 27 | reasoning-eval | M6 | ✅ | ✅ | — | ✅ | 🩹 | 10 demo V1 全绿(gsm8k/math/aime/gpqa/hle/zebra/math-verify/tool-aug/capstone/common)，全有 __main__。**#10 预防同 #26**：9 脚本 __main__ 原只 print "self-test: OK"→补可见 demo(真 acc/验证)，纯加 print(155+/0-)。runner 真算 acc(dummy 恒0/oracle 恒1)；math_verify 真等价判断(1/2==0.5==\frac{1}{2}，无 sympy 是诚实简化非#9)；tool_aug 真沙箱 exec(tool 1.00 vs CoT 0.00，import/dunder 拦截实证)；**zebra 是 benchmark runner 非求解器**(subagent 未捏造求解器，诚实标注)。修文档(CWD 依赖)。V2 重跑绿(10) | |
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
10. **mock benchmark 没真演示出所宣称效果**（sglang_compare 实例；重型栈模块高发）：本地跑不动真引擎(vllm/sglang)→用合成 mock 对照，但 mock 的两条代码路径**实际相同/或"收益"是与模型解耦的硬编码** → 跑得通、也"诚实标注 mock"，却**没真演示出它要教的差异**(两列 cost 恒相等却报 +83.3% gain)。
    - 这是比"诚实 mock"更深一层的坑：mock 标注诚实 ≠ mock 逻辑正确。**审 mock 时要看它是否真算出了所宣称的对比**(两路径是否真分裂、"gain"是否真由展示的列推出)。
    - 修法：让对照的两条路径**真有机制差异**(如 vLLM fork 独立重 prefill vs SGLang 共享一次)，展示的收益**由展示的数推出**(`1−s/v`)而非另写硬编码；未建模的优势用**定性说明列**而非假百分比。加测试锁住"对照确有差异"。M5/M6 后续 mock 对照(quant/distributed/serving 的 benchmark)按此审。
