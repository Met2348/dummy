# LLM Learning Portfolio (v5) — 工程 + 研究 + 前沿 三腿

> v5 在 [v4](portfolio_v4.md) 的「工程腿 (M1-8, 48 专题) + 研究腿 (M9, 9 专题)」之上,
> 补齐**第三条腿: 前沿腿 (M10-13, 28 专题)** + **8 个真实模型旗舰例子** + **13 个旧专题 notebook 回补**。
> 体系终态 = **48 工程 + 9 研究 + 28 前沿 = 85 专题**, 全部 nbconvert 跑通、小尺度 CPU 可跑、离线确定性。

---

## 总览: 三腿站立

```
   工程腿 (M1-8, 48 专题)   —— 会造/改/用/评/守模型 + agent + infra (v4 详述)
   研究腿 (M9, 9 专题)      —— 会做研究 (找gap→实验→复现→出图→写投讲审) (v4 详述)
   前沿腿 (M10-13, 28 专题) —— 多模态 / 具身VLA / 可解释性 / 扩散世界模型 ⭐⭐⭐ NEW (本文)
   ─────────────────────────────────────────────────────────────
   = 2026 年 LLM 全栈工程师 + 准研究者 + 前沿四向 ID 卡 v5
```

> v4 的核心信念 (「前沿系统 = 已学部件的组合 + 规模」) 在 v5 全程兑现: 28 个前沿专题没有一个从零造, 全是把 48 工程 + 9 研究的存量**架到新范式上** (VLA 复用 transformer/RL/VLM, dLLM 复用 pretraining, interp 用 EE 数学)。

---

## Section A: Modules 10-13 — 前沿 4 大扩展 ✅ 全部完成 (2026-06)

> 2026-06 web 核对招聘后锁定的 4 个空白, 全部命中「真空白 × 2026 高薪 × 适配用户轨迹」。**4 模块 × 7 专题 = 28 专题, 全部落地** (研究生级课件 4 讲 + 2 notebook + 1-2 src + env/papers, nbconvert 0 报错)。

### M10 多模态 / VLM ✅ (用户明牌方向, 一切的地基)
| 专题 | 核心可跑产物 |
|---|---|
| vision-encoders | tiny ViT patchify + 对比学习 (CLIP/SigLIP/DINOv2) |
| vl-fusion-architectures | 投影 / cross-attn / early-fusion 三种 VL 融合 |
| vlm-training-recipe | mini-VLM (loss 1.77→0.07 acc→1.0) + 模态坍缩诊断 + 冻结食谱 |
| visual-tokenization-generation | VQ 视觉 token 化 + 自回归生图 |
| video-audio-omni | 视频时空 token + 音频 mel + omni 统一 |
| vlm-eval-hallucination | POPE 幻觉评测 |
| multimodal-graduation | capstone gap 雷达 |

### M11 具身 / VLA ✅ (用户已碰 IsaacLab; NLP 最自然转型)
| 专题 | 核心可跑产物 |
|---|---|
| embodied-foundations | `toy_env` 2D到达 (M11共享环境) + tokens-as-actions; next-action 策略成功率 1.00 |
| vla-architectures | `mini_vla` (backbone+动作头); 离散 vs 连续 (连续平滑 4×) |
| action-heads-diffusion-policy | 扩散动作头解多峰 (双峰 94上137下) vs 回归取均值直冲障碍; chunking 权衡 |
| robot-data-imitation | BC + 分布漂移; 数据 scaling 3条 0.09→200条 0.94 |
| world-action-models | 随机数据学世界模型 + MPC 规划 0.88 **零专家**; model-based 20样本 0.82 vs model-free 需 500 |
| sim2real-isaaclab | 域随机化弥合 gap 0.71→0.99; **吸收用户 WSL2失败→Windows原生 AntBot 踩坑** (`isaaclab_notes.md`) |
| embodied-graduation | 全栈装配 + mini-benchmark (BC/mini-VLA/世界模型MPC) |

### M12 机制可解释性 ✅ (MIT 2026 十大突破; **最可能转 PhD 题**; 重度用真实模型)
| 专题 | 核心可跑产物 (★ 真实 gpt2/TinyLlama) |
|---|---|
| interp-foundations | `tiny_transformer` 可hook基座 + ★真 gpt2 多义神经元 (superposition) |
| probing-and-activations | 线性探针 + logit lens; ★真 gpt2「Paris」排名 22679→0 逐层浮现 |
| causal-interventions | activation patching 因果定位最后位置 (恢复 1.0) + ablation 充要 |
| sparse-autoencoders | SAE 解叠加纯度 0.56 >> 原始 0.15; ★真 gpt2 特征 |
| circuits-attention | ★真 gpt2 induction head (层5头5 分数 0.95) + 逐头消融见冗余 |
| cot-faithfulness-oversight | ★真 TinyLlama 偏置敏感性 ~100% (CoT 不忠实) + weak-to-strong 0.73→0.99 |
| interp-graduation | 完整逆向工程 (探针→patching→SAE) + interp×reasoning gap 雷达 |

### M13 扩散 / 生成式媒体 / 世界模型 ✅ (范式级空白; dLLM 直连 NLP)
| 专题 | 核心可跑产物 |
|---|---|
| diffusion-foundations | DDPM 2D双月去噪 (`diffusion.py` M11 复用) |
| flow-matching-sota | flow matching + rectified flow (reflow std_err 0.24→0.05) |
| dit-latent-diffusion | DiT (transformer去噪) + CFG (g=0→0.23, g=1→1.0) + latent |
| video-generation | 时空 vs 逐帧视频扩散 (连贯 0.65 vs 抖动 0.87) + Sora 拆解 |
| world-models | 学环境动态 + 想象 rollout + 多步误差累积 (`world_model.py` M11 共享) |
| diffusion-language-models | masked dLLM 并行解码; **双向 infill dLLM 1.00 vs AR 0.17(瞎猜)** |
| generative-media-graduation | 跨专题 import 全6链装配 + 统一生成画廊 + 7 gap 雷达 |

**依赖与构建顺序**: M10(地基) → M13(扩散/世界模型, 供 M11 动作头/世界模型) → M11(用 M10+M13) → M12(独立横切)。
**跨专题 src 复用**: M13 `diffusion.py`/`world_model.py` 被 M11 复用; M11 `toy_env.py` 被 11.2-11.7 复用; M12 `tiny_transformer.py` 被 12.2-12.7 复用; `_shared/realmodels.py` 真实模型 helper 被 8 旗舰 + M12 复用。

---

## Section B: 8 个「小而真」真实模型旗舰例子 ⭐⭐ NEW

> 用户要求「toy 之外加小而真的例子」。用本地 HF 缓存的**真实模型** (gpt2-124M / TinyLlama-1.1B-Chat, CPU 离线确定性) 给 8 个旗舰专题各加 1 个真实 notebook。共享工具 `learning/_shared/realmodels.py` (优雅降级)。**所有数字诚实校准过。**

| 专题 | 真实例子 | 关键真实数字 |
|---|---|---|
| transformer-deep | 真实 gpt2 注意力 + KV cache | 下一token「Paris」6.4%; KV cache 加速 |
| eval-foundations | 真实困惑度 | 通顺 90 vs 打乱 6037 |
| reasoning-eval | 真实 CoT vs 直接答 | 小模型真实算错 (活教材) |
| llm-judge-arena | 真实 LLM 评委 | 位置偏置暴露 |
| red-team-jailbreak | 真实拒答行为 | 正常答/有害拒 (防御视角) |
| rag-essential | 真实检索+接地 | 闭卷瞎编 → 开卷答对 "Mira Chen" |
| quantization-deploy | per-channel 量化 | int8 +4.7% 近无损 / int4 +747% |
| lora-family | forward-hook LoRA | 0.01% 参数, 目标句困惑度 378→4, 无关句不变 |

> 价值: 这 8 个例子把课程从「纯 toy」升级到「toy 教机制 + 真模型验证」两条腿, 尤其 M12 (机制可解释性) **全程重度用真实 gpt2/TinyLlama** —— interp 在真模型上才有说服力。

---

## Section C: 13 个旧专题 notebook 回补 ⭐ NEW

> 用户要求「回补所有 nb=0 旧专题」。13 个有讲义+src 但缺 notebook 的旧专题, 全部补上可跑 notebook (复用既有 src + 可视化), nbconvert 全通过。

- **基础设施 7**: cuda-essentials (online softmax) / gpu-architecture (roofline) / kernel-engineering (flash 内存 O(S²)→O(S)) / cluster-networking (allreduce) / storage-dataops (checkpoint 策略) / training-orchestration (24h 容错集群) / infra-graduation (全栈)。
- **RL/后训练 6**: dpo-family (6 PO 变体, 修 capstone bug) / rlhf-classic (reward hacking) / process-reward (PRM+BoN 0.3→1.0) / reasoning-r1 (GRPO countdown 0.04→0.23) / rl-sota-2026 (DAPO 消融 +11.5pp) / multimodal-agent (毕业对比)。

---

## Section D: 12 大画像 v5 (7 工程 + 1 研究 + 4 前沿)

```
你已具备：

—— 工程腿 (M1-8, 48 专题) ——
1. 造模型   2. 改模型   3. 用模型   4. 评模型   5. 守模型   6. 造 agent   7. 造 infra

—— 研究腿 (M9, 9 专题) ——
8. 会做研究的人 — 找 gap → 实验 → 复现 → 出图 → 写/投/讲/审 (消费知识→生产知识)

—— 前沿腿 (M10-13, 28 专题) ⭐⭐⭐ NEW ——
9.  懂多模态/VLM    — ViT/CLIP/VL融合/VLM训练/VQ生成/视频音频/幻觉评测 (M10)
                    + 会拆 Stable Diffusion 每一块
10. 懂具身/VLA      — tokens-as-actions/VLA架构/扩散动作头/模仿学习/世界模型/sim2real (M11)
                    + 会拆 OpenVLA/π/GR00T; 有真实 IsaacLab 手感
11. 会解剖模型       — 探针/patching/SAE/circuits/CoT忠实性 (M12)
                    + 能对模型做完整逆向工程 + 看穿是否诚实 (在真 gpt2 上找到 induction head)
12. 懂扩散/世界模型  — DDPM/flow/DiT/视频/世界模型/dLLM (M13)
                    + 会拆 Sora; dLLM 双向并行解码直连你的 NLP

= 2026 年 LLM 全栈工程师 + 准研究者 + 前沿四向 ID 卡 v5
```

> 8 → 12 大画像 ⭐⭐⭐: 前沿腿把 ID 卡从「会造/会研究」扩到「**会造前沿系统 + 会理解前沿系统**」。
> **核心信念兑现**: 12 个画像里, 9-12 (前沿) 全是 1-8 (工程) + 9 (研究) 的**迁移**, 不是从零 —— 这正是用户 (NLP/EE, 迁移优势) 的最大杠杆。

---

## Section E: 前沿研究轨道 (给博 0 的路线)

> M12 的 capstone 明确指出: **interp × reasoning 是用户最可能直接转的 PhD 题**。各前沿模块的研究轨道:

| 研究轨道 | 切入点 | 为什么适配用户 |
|---|---|---|
| **interp × reasoning** ★最匹配 | CoT 忠实性的机制级验证 / 计算 vs 陈述一致性 (M12.6+12.3) | NLP+reasoning+EE 数学全用上; frontier 安全热点; toy 入口 |
| **dLLM / 扩散语言模型** | dLLM 少步高质量解码 / 对齐迁移 (M13.6) | 直连 NLP 本行; 范式新有红利 |
| **数据高效具身 / sim2real** | 世界模型样本效率 / DR 自动范围 (M11.5/11.6) | 接 NLP 数据直觉 + 真实 IsaacLab 手感 (稀缺) |
| **统一生成** | 跨模态一个扩散模型 (M13.7) | 接 M10 时空 token + M13 扩散 |

---

## Section F: Career Paths v5 (在 v4 基础上加前沿轨道)

| Path | Salary (2025 SF) | Key topics |
|------|-------:|-----------|
| LLM Infra Engineer ⭐ | $300k-$700k | M3 + M5 + M8 |
| ML Research Engineer | $300k-$1M+ | M3 + M4 |
| AI Safety Engineer | $200k-$500k | M6 + **M12 (interp)** ⭐ |
| PhD / Research Scientist ⭐⭐ | 学术/工业研究院 | M9 + **M12 (interp×reasoning)** ⭐⭐ |
| **多模态/VLM Engineer** ⭐ NEW | $300k-$800k | **M10** + M3 |
| **具身/机器人 (VLA) Engineer** ⭐ NEW | $300k-$900k (抢人) | **M11** + M13 + RL |
| **机制可解释性 Researcher** ⭐⭐ NEW | frontier 实验室硬通货 | **M12** + reasoning |
| **扩散/生成式媒体 Engineer** ⭐ NEW | $300k-$700k | **M13** + M10 |

> 前沿腿打开 4 条新职业/研究路径, 全部是 2026 bidding-war 方向。**M12 (interp) 同时强化 AI Safety Engineer + PhD 两条路** —— 这是用户初衷 (转 PhD) 的最强落点。

---

## What I Can Do (cover letter snippets v5 — 前沿新增)

- "I can build a mini-VLM from a ViT + projection + LLM, and explain every block of Stable Diffusion (M10)."
- "I can assemble a mini-VLA (VLM backbone + diffusion action head), and solve a control task with a world model + MPC using zero expert demos (M11)."
- "I can find an induction head in real gpt2, causally verify it by ablation, and reverse-engineer a model end-to-end (probe→patch→SAE) (M12)." ⭐⭐
- "I can demonstrate CoT unfaithfulness on a real model via bias-sensitivity, and frame a falsifiable interp×reasoning research question (M12). ⭐⭐"
- "I can build a diffusion language model with bidirectional parallel decoding, and show it beats AR at infilling (M13)."
- "I can derive DDPM/flow-matching, train a DiT with CFG, and decompose Sora into spacetime patches + DiT + latent (M13)."
- "I trained 8 flagship topics on real local models (gpt2/TinyLlama, CPU-offline) — real attention/KV, perplexity, CoT, judge, RAG, int8 quant, LoRA — not just toys." ⭐

---

> 设计/计划文件: `docs/superpowers/specs|plans/2026-06-24-module1{0,1,2,3}-*`
> 体系: 85 专题 (48 工程 + 9 研究 + 28 前沿) + 8 真实旗舰例子 + 13 回补; 全部 CPU 可跑、离线确定性、nbconvert 0 报错。
