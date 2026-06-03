# Multimodal + Agent + Graduation Implementation Plan ⭐⭐⭐⭐⭐

**Goal**: 完整实现系列毕业专题学习包（14 lecture + minimal/库三轨 + 6 测试 + 14 notebook + 双 Capstone（VLM-R1 玩具 + 五线综合毕业作品））

**Architecture**: 14 lectures = 11 主线 + 1 多模态 Capstone + 1 五线综合 + 1 毕业作品。三轨代码：minimal + trl + verl + 各论文官方 repo。

**环境**: WSL2 继承专题 5-6

**Tech Stack**: WSL2 / torch cu130 / Qwen2-VL-2B / trl 0.13 / verl 0.4 / vllm 0.7 / qwen-vl-utils / miniwob / swe-gym / playwright

**Design 文档**: `docs/superpowers/specs/2026-06-03-multimodal-agent-design.md`

---

## Phase 1: 基础设施

### Task 1.1: 目录骨架
- WSL2 中 `learning/multimodal-agent/{environment,papers,lectures,src/tests,notebooks}/`

### Task 1.2: environment/requirements.txt
- qwen-vl-utils + miniwob + swe-gym + playwright + 继承专题 5-6

### Task 1.3: environment/verify_env.py
- Part A: 多模态库 + agent 库 import
- Part B: Qwen2-VL-2B 加载 smoke
- Part C: VLM-R1 GRPO 5-step smoke

### Task 1.4: src/common.py
- 多模态数据 loader / agent episode helpers / 5 ckpt 加载接口

### Task 1.5: papers/ 14 个占位 + README

### Commit: `chore: multimodal-agent scaffold`

---

## Phase 2: L1-L3 Multimodal RL

### Task 2.1: lectures/01-vision-r1.md (24 slides)
- cold-start + R1-style 多模态
- 数据 + 算法 + reward 设计

### Task 2.2: src/vision_r1_demo.py
- 简化版 Vision-R1（在 Qwen2-VL-2B 上 mock）

### Task 2.3: lectures/02-vlm-r1.md (24 slides)
- GRPO 直接训 VLM（无需 SFT）
- VLM-R1 关键工程细节

### Task 2.4: src/vlm_r1_minimal.py
- Qwen2-VL-2B + 4bit LoRA + GRPO
- CLEVR counting 数据接口

### Task 2.5: lectures/03-kimi-k1.5-vision.md (22 slides)
- long context + vision 联合 RL

### Task 2.6: src/kimi_vision_demo.py
- 简化演示

### Task 2.7: src/tests/test_vlm_r1.py
- 50 step 训练后 counting acc 上升

### Task 2.8: notebooks/01-03 三个

### Commit + Tag: `mm-vision-r1`

---

## Phase 3: L4-L7 Agent RL

### Task 3.1: lectures/04-webrl.md (22 slides)
- WebRL 自演化课程 + MiniWoB++

### Task 3.2: src/webrl_toy.py
- MiniWoB++ 玩具 + 简化 self-evolving curriculum

### Task 3.3: lectures/05-swe-gym.md (22 slides)
- 软件工程 RL benchmark
- issue → patch pipeline

### Task 3.4: src/swe_gym_demo.py
- 跑 1 个 issue 的完整 episode

### Task 3.5: lectures/06-computer-rl.md (22 slides)
- 智谱 ComputerRL / AutoGLM-OS
- OSWorld 48.9% 工程路径

### Task 3.6: src/computer_rl_demo.py
- 用智谱官方 repo 简化演示（docker）

### Task 3.7: lectures/07-tool-rl.md (22 slides)
- Tool-use RL

### Task 3.8: src/tool_rl_demo.py
- 简化 tool-use（calculator + search）

### Task 3.9: notebooks/04-07 四个

### Commit + Tag: `mm-agent`

---

## Phase 4: L8-L10 Long Context + TTC + Thinking Models

### Task 4.1: lectures/08-long-context-rl.md (22 slides)
- Long2Short / Token-PG / Overlong shaping

### Task 4.2: src/long_context_rl_demo.py
- 长文本 RL 简化 demo

### Task 4.3: lectures/09-test-time-scaling.md (28 slides)
- s1 budget forcing (Wait token)
- Don't Overthink 警示

### Task 4.4: src/s1_budget_forcing.py
- Wait token 强制延长机制
- Token budget 控制

### Task 4.5: src/dont_overthink_demo.py
- 演示 TTC 不是越长越好（accuracy vs length 曲线）

### Task 4.6: lectures/10-thinking-models-2026.md (22 slides)
- Claude 4 Extended Thinking
- Gemini 2.5 thinking_budget

### Task 4.7: src/thinking_model_api_demo.py
- API 调用示例 + 与本地模型对比

### Task 4.8: notebooks/08-10 三个

### Commit + Tag: `mm-long-ttc`

---

## Phase 5: L11 Safety + RL

### Task 5.1: lectures/11-safety-rlhf.md (24 slides)
- Constitutional AI 升级版
- Constitutional Classifiers (2025 jailbreak 防御)
- Safe-RLHF Lagrangian 多目标

### Task 5.2: src/safe_rlhf_minimal.py
- Lagrangian 多目标实现
- 简化 Constitutional Classifier

### Task 5.3: src/tests/test_safe_rlhf.py
- Lagrange 收敛性验证

### Task 5.4: notebooks/11-safety-rlhf.ipynb

### Commit + Tag: `mm-safety`

---

## Phase 6: L12 Capstone-1 VLM-R1 玩具复现

### Task 6.1: lectures/12-capstone-vlm-r1-toy.md (28 slides)
- 完整 walkthrough

### Task 6.2: src/capstone_vlm_r1/
- 数据: CLEVR counting (1k train / 200 val)
- 训练: Qwen2-VL-2B + 4bit LoRA + GRPO
- 评估: counting accuracy + response length

### Task 6.3: src/tests/test_capstone_vlm_r1.py
- 50 step 后 reward 上升验证

### Task 6.4: notebooks/12-capstone-vlm-r1-toy.ipynb
- 完整跑通展示 + counting acc 提升曲线

### Commit + Tag: `mm-capstone-vlm-r1`

---

## Phase 7: L13 ⭐⭐⭐ 五线综合 lecture

### Task 7.1: lectures/13-five-line-unification.md (32 slides) ⭐⭐⭐ 系列理论高峰

**Part I: 五线回顾 (8 slides, 20 min)**
- 每条线的切入点 + 典型方法
- 三个 PEFT 线（input/weight/structure）+ 两个 RL 线（distribution/trajectory）

**Part II: 统一公式 (12 slides, 35 min)**
- 核心命题：`p(y|x; θ_LM, φ)` 五线就是 φ 的不同安放位置
  - Prompt: φ = input embedding 扰动
  - LoRA: φ = weight 低秩扰动
  - Adapter: φ = structure 加层扰动
  - RLHF: φ 通过 reward 改 distribution shape
  - R1: φ 通过 verifier+rollout 改 trajectory
- 三句话一锤定音
- 跨主线等价对（Prefix ≡ Parallel Adapter / LoRA = Adapter w/o σ）

**Part III: 工程选型决策树 (8 slides, 20 min)**
- 4 真实场景：客服 / 数学竞赛 / SWE agent / 千用户 SaaS

**Part IV: 历史观 + 下一程 (4 slides, 15 min)**
- 大模型对齐 5 年史
- MoE / Long Context / Continuous Pretraining / World Model RL

### Task 7.2: src/unified_view.py
- 88 方法的统一公式数值验证
- 跨主线等价对验证（如 LoRA(σ=id) ≡ Parallel Adapter）

### Task 7.3: notebooks/13-five-line-unification.ipynb
- 统一公式数值演示 + 跨等价对验证

### Commit + Tag: `mm-five-line-unified`

---

## Phase 8: L14 ⭐⭐⭐⭐⭐ 毕业 Capstone

### Task 8.1: lectures/14-capstone-graduation.md (24 slides)
- 系列学习历程回顾
- 5 ckpt 同题对照 walkthrough

### Task 8.2: src/capstone_graduation/
- 加载 5 个 ckpt:
  1. Vanilla GPT-2-base
  2. LoRA (lora-family L01 ckpt) — 如已有
  3. Pfeiffer Adapter (adapter L01 ckpt)
  4. DPO (本系列专题 3 capstone ckpt)
  5. R1-Zero (本系列专题 5 capstone-A ckpt)
- 在同一道 GSM8K 题（Janet 鸡蛋）上跑 5 种 inference
- 输出对照表 + 推理 trace 可视化

### Task 8.3: src/ckpt_download.py
- 提供 fallback ckpt 下载脚本（学员可能某 ckpt 没训）

### Task 8.4: notebooks/14-capstone-graduation.ipynb
- 完整 5 ckpt 对照展示
- 性能表格 + 推理过程可视化

### Task 8.5: 输出"系列学习证书" markdown
- 自动生成证书：已完成 7 专题 / 88 方法 / 90 lectures / ~101h
- 自评测题完成情况

### Task 8.6: README.md
- 14 方法横向表 + 五线综合 cheat sheet + 整个系列总图
- 自测 14 题（含五线综合理论题）
- 系列总结

### Task 8.7: papers/ 补全

### Commit + Tag: `rl-graduation` ⭐⭐⭐⭐⭐ 系列收官

---

## Phase 9: 收尾验证

### Task 9.1: 全部 notebook 执行
### Task 9.2: 全部测试
### Task 9.3: Capstone-1 VLM-R1 counting acc 提升
### Task 9.4: Capstone-2 五线对照可视化完成
### Task 9.5: verify_env.py PASS

### Final commit: `docs: graduation README + complete series`

---

## 完成验收清单

- [ ] 14 lecture markdown
- [ ] 14 notebook 全跑
- [ ] VLM-R1 Capstone counting acc 提升
- [ ] 五线综合 lecture L13 完整 32 slides
- [ ] 五线综合 unified_view.py 数值验证 PASS
- [ ] 毕业 Capstone 5 ckpt 加载 + 同题对照
- [ ] 系列学习证书自动生成
- [ ] tag `rl-graduation` ⭐⭐⭐⭐⭐

**预计 git commits**: ~30
**预计实施时长**: 14 hours

---

## 系列收官总结（README 顶部用）

```
🎓 PEFT + RL 学习系列完整毕业

  专题 1: prompt-tuning-family   (5 方法,  ~6h)
  专题 2: lora-family             (12 方法, ~10h)
  专题 3: adapter-tuning-family   (11 方法, ~13h)
  专题 4: rl-foundations          (12 方法, ~14h)
  专题 5: rlhf-classic            (12 方法, ~15h)
  专题 6: dpo-family              (13 方法, ~14h)
  专题 7: process-reward          (12 方法, ~14h)
  专题 8: reasoning-r1            (15 方法, ~18h) ⭐ 系列高峰
  专题 9: rl-sota-2026            (12 方法, ~12h)
  专题 10: multimodal-agent       (14 方法, ~14h) ⭐ 系列毕业
  ─────────────────────────────────────────────────
  总计                            128 方法, ~130h

下一步推荐:
  ⭐ MoE (Mixtral / DeepSeek-MoE)
  ⭐ World Model RL
  ⭐ Continuous Pretraining
  ⭐ Test-time Scaling 深化
```
