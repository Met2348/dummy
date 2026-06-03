# Small Model + 五部曲毕业 Implementation Plan

**Goal**: 完整实现小模型 + 蒸馏 + BitNet + 毕业作品（14 lecture + 多个 distill 实现 + 8 测试 + 14 notebook + Capstone-1 270M→80M 蒸馏 + Capstone-2 ⭐⭐⭐⭐⭐ 五部曲毕业作品）

**Architecture**: 14 lectures = 11 主线 + 1 蒸馏 capstone + 1 综合 lecture + 1 毕业 capstone。WSL2 环境。

**Tech Stack**: torch + transformers + peft + bitsandbytes + accelerate + matplotlib + seaborn

**Design 文档**: `docs/superpowers/specs/2026-06-04-small-model-graduation-design.md`

**特殊地位**: 完成本 plan 后整个 Module 1+2+3 共 234 方法学习系列毕业 🎓

---

## Phase 1: 基础设施

### Task 1.1: 目录骨架
- Create `learning/small-model-graduation/{environment,papers,lectures,src/tests,notebooks}/`

### Task 1.2: environment/requirements.txt
- torch + transformers + peft + bitsandbytes + accelerate
- matplotlib + seaborn (雷达图)

### Task 1.3: environment/verify_env.py
- Part A: transformers + peft + bnb import
- Part B: GPU + sm_120 + 8GB 空闲
- Part C: 加载 Phi-3.5 mini 4bit smoke

### Task 1.4: src/common.py
- 加载 4bit 多 model helpers + radar chart helper

### Task 1.5: papers/ 11 个占位 + README index

### Commit: `chore: graduation scaffold`

---

## Phase 2: L01-L07 小模型 6 系列

### Task 2.1: lectures/01-small-model-era.md (16 slides)
- 边缘 + 隐私 + 成本 三大动力

### Task 2.2: lectures/02-phi-1-2-3-4.md (28 slides)
- Phi 1 → 1.5 → 2 → 3 → 3.5 → 4 完整演化
- 教科书数据 + 合成数据 trick

### Task 2.3: src/phi_4_load_test.py
- Phi-4 加载 + 推理 + 对比 GPT-2 / Llama

### Task 2.4: lectures/03-llama-3.2-edge.md (16 slides)
- Llama-3.2 1B/3B 边缘部署

### Task 2.5: src/llama32_load_test.py
- Llama-3.2-1B 加载 + 推理

### Task 2.6: lectures/04-qwen3-small.md (16 slides)
- Qwen3-0.6B/1.7B/4B 多语言

### Task 2.7: src/qwen3_small_load.py
- Qwen3-0.6B 加载 + 多语言 demo

### Task 2.8: lectures/05-smollm2.md (14 slides)
- HuggingFace SmolLM2 全开源

### Task 2.9: src/smollm2_load_test.py
- SmolLM2 加载 + 评测

### Task 2.10: lectures/06-tinyllama.md (12 slides)
- TinyLlama 1B 长训练

### Task 2.11: lectures/07-gemma-3.md (14 slides)
- Gemma-3 Google 小模型

### Task 2.12: notebooks/01-era.ipynb + 02-phi.ipynb + 03-llama32.ipynb + 04-qwen3.ipynb + 05-smollm2.ipynb + 06-tinyllama.ipynb + 07-gemma3.ipynb

### Commit + Tag: `small-models`

---

## Phase 3: L08-L09 蒸馏

### Task 3.1: lectures/08-h20-distillation.md (24 slides)
- KD 蒸馏（logit / hidden / sequence）

### Task 3.2: src/distill_logit.py
- 标准 KL 蒸馏 loss + temperature

### Task 3.3: src/distill_hidden.py
- hidden state MSE / cosine 对齐

### Task 3.4: lectures/09-soft-distill.md (20 slides)
- MiniLLM reverse KL + 经验

### Task 3.5: src/minillm_distill.py
- reverse KL 蒸馏实现

### Task 3.6: src/tests/test_distill_loss.py
- KL / reverse KL 数学正确

### Task 3.7: notebooks/08-distill.ipynb + 09-minillm.ipynb

### Commit + Tag: `distill`

---

## Phase 4: L10-L11 BitNet + QAT

### Task 4.1: lectures/10-bitnet.md (28 slides)
- BitNet b1.58 完整推导
- 三值量化 {-1, 0, 1}

### Task 4.2: src/bitnet_demo.py
- BitNet b1.58 forward 演示
- absmean quantization

### Task 4.3: src/tests/test_bitnet_forward.py
- 三值量化 forward 数值正确

### Task 4.4: lectures/11-quantization-aware-pretrain.md (20 slides)
- 训练阶段 QAT

### Task 4.5: src/qat_pretrain.py
- 教学版 QAT 训练循环

### Task 4.6: notebooks/10-bitnet.ipynb + 11-qat.ipynb

### Commit + Tag: `bitnet-qat`

---

## Phase 5: L12 Capstone-1 蒸馏

### Task 5.1: lectures/12-capstone-distill-phi-tiny.md (28 slides)
- 把专题 7 270M Phi-tiny 蒸馏到 80M GPT-mini

### Task 5.2: src/mini_distill_phi.py
- Teacher: 270M Phi-tiny (专题 7 ckpt)
- Student: 80M GPT-mini (专题 2 架构)
- logit + hidden 双蒸馏

### Task 5.3: src/tests/test_distill_student_ppl.py
- student val ppl < teacher × 1.3

### Task 5.4: notebooks/12-capstone-distill.ipynb

### Commit + Tag: `distill-capstone`

---

## Phase 6: L13-L14 五部曲毕业 ⭐⭐⭐⭐⭐

### Task 6.1: lectures/13-five-line-unification-revisit.md (32 slides)
- Part I (8): 五线回顾
- Part II (12): 统一公式 p(y|x; θ_data, θ_arch, θ_weight, φ)
- Part III (8): 工程选型决策树
- Part IV (4): 历史观 + 下一程

### Task 6.2: src/five_unification_theory.py
- 五线统一公式可视化 + 决策树

### Task 6.3: notebooks/13-unification.ipynb
- 公式推导 + 决策树交互式

### Commit + Tag: `five-unification`

---

### Task 6.4: lectures/14-capstone-graduation.md (32 slides)
- 同一道 GSM8K 题
- 5 个 ckpt 演化路径完整对照
- 雷达图：格式 / 准确 / 推理深度 / 自检 / 响应速度

### Task 6.5: src/capstone_five_module_graduation.py ⭐⭐⭐⭐⭐
- GSM8K_PROBLEM + GROUND_TRUTH
- 加载 5 ckpt（4bit 顺序加载避免 OOM）：
  - 1. Vanilla GPT-2 (公开 ckpt)
  - 2. LoRA 微调 (Module 1 ckpt)
  - 3. DPO 对齐 (Module 2 dpo-family ckpt)
  - 4. R1-Zero 推理 (Module 2 reasoning-r1 ckpt)
  - 5. 自训 Phi-tiny (Module 3 专题 7 ckpt) ⭐
- 同一道题生成 5 个 response
- 输出雷达图 + 性能表 + latency 表

### Task 6.6: src/tests/test_graduation_capstone.py
- 5 ckpt 全部加载成功
- 5 response 长度 / format 差异化
- export_for_notebook 返回结构正确
- 雷达图 5 个维度都有值

### Task 6.7: notebooks/14-capstone-graduation.ipynb
- 完整毕业作品交互式 notebook
- 5 response 文本对照
- 5 个 model latency / 显存表
- 雷达图最终图

### Task 6.8: learning/small-model-graduation/README.md
- 毕业宣言：234 方法 / 252h / 5 部曲完整 ✓

### Commit + Tag: `造改-graduation` ⭐⭐⭐⭐⭐

---

## 验证清单

```bash
python learning/small-model-graduation/environment/verify_env.py
python -m pytest learning/small-model-graduation/src/tests/ -v
jupyter nbconvert --execute --inplace learning/small-model-graduation/notebooks/*.ipynb
python learning/small-model-graduation/src/capstone_five_module_graduation.py
```

预期：
- env 三段 PASS
- 8 tests PASS
- 14 notebook 跑通
- Capstone-1 student ppl < teacher × 1.3
- Capstone-2 五部曲对照可视化完成（5 response + 雷达图）
- 🎓 整个 Module 1+2+3 共 234 方法学习系列毕业

---

## 总览

- 14 lectures × 平均 70 min = 16h slides
- 多个 distill + BitNet 实现
- 8 tests
- 14 notebooks
- 2 capstones (蒸馏 + 五部曲毕业)
- 预计 6 commit + 5 tag (含 `造改-graduation`)
- 总时长 12h（lecture + notebook + Capstone）

---

## 系列毕业里程碑

完成本 plan 后：

```
Module 1 (PEFT)          ✓ 28 方法 / ~29h
Module 2 (RL/对齐/推理)    ✓ 88 方法 / ~101h
Module 3 (造大模型)        ✓ 118 方法 / ~122h
─────────────────────
合计                       234 方法 / ~252h
```

**🎓 LLM 工程"硕士级"全套课程图谱完成**

后续可选：
- Module 4 (用大模型 — Inference + Deploy)
- Module 5 (扩大模型 — Multimodal Generation)
- Module 6 (包大模型 — Agent + Production)
