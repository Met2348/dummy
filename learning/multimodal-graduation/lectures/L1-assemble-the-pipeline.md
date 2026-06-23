# L1 · 装配完整 mini-VLM 流水线: M10 全线串起来

> 30-min lecture · 目标: 把 M10.1-10.6 的所有组件串成一条完整的、能跑的 mini-VLM 流水线, 看清整个 Module 10 是怎么咬合的。

---

## 0. 回看你造了什么

走完 M10.1-10.6, 你手里有六块咬合的能力, 每块都有可跑的 src:

```
   10.1 vision-encoders     视觉塔: 图 → 视觉 token (tiny_vit, 对比学习)
        ↓
   10.2 vl-fusion           连接器: 视觉 token 接进 LLM (connectors, 三路线)
        ↓
   10.3 vlm-training        训练: 两阶段配方训出能问答的 VLM (mini_vlm, loss↓ acc↑)
        ↓
   10.4 visual-tokenization 生成: VQ 让 VLM 不只读图还能画图 (vq_tokenizer)
        ↓
   10.5 video-audio-omni    扩展: 视频/音频也 token 化, 通往 omni (temporal/audio)
        ↓
   10.6 vlm-eval            评测: POPE 测它真看图吗、会幻觉吗 (vlm_eval)
```

> 这不是六个孤立的练习, 是**一条能力链**: 每个专题的 src 都被下游真复用 (10.2 用 10.1 的 ViT, 10.3 用 10.1+10.2, ...)。`mm_capstone.assemble_pipeline_check` 会验证这六块都在、能串起来。

---

## 1. 一条端到端的 mini-VLM 流水线

把它们串成一个完整流程 (N1 会真跑):

```
   输入图 + 问题
      │  ① 10.1 tiny_vit: 图 → 视觉 token
      │  ② 10.2 connector: 视觉 token → LLM 空间, 拼进序列
      │  ③ 10.3 mini_vlm: tiny LLM 处理 [视觉 | 文本], 输出答案
   答案 (理解)
      │  ④ (可选) 10.4 vq + 自回归: 反过来从文本生成图 token → 画图
      │  ⑤ 10.6 vlm_eval: POPE 测它真看图吗、幻觉吗
   评测剖面
```

> 这条流水线就是一个**完整的多模态系统骨架**: 既能读图回答 (理解), 又能画图 (生成), 还能自我评测 (诚实)。虽然 tiny, 但**机制完整** —— 真实的 LLaVA/Chameleon 就是把每块换成大尺度版本。你理解了 tiny 版, 就理解了真版。

---

## 2. 从 tiny 到真实: 一一对应

| 你的 mini 组件 | 真实对应 | 换什么 |
|---|---|---|
| tiny_vit | CLIP/SigLIP ViT-L | 更大、预训练好的视觉塔 |
| connector (投影) | LLaVA MLP / Flamingo resampler | 同结构, 更大 |
| mini_vlm + tiny LLM | LLaVA + LLaMA/Qwen | 真 LLM 底座 |
| vq_tokenizer | VQGAN | 训练好的高质量码本 |
| vlm_eval (POPE) | 真 POPE/MMMU/MME | 真 benchmark |

> 关键: **架构和机制不变, 只是规模和预训练换了。** 这就是 tiny 教学的价值 —— 你在 CPU 上几秒钟跑通的流水线, 和几百卡训的真 VLM **是同一个东西的不同尺度**。你已经懂了 VLM 怎么造、怎么训、怎么测。

---

## 3. 这条流水线接回你的工程体系

mini-VLM 流水线不是孤岛, 它接回你已有的 48 工程专题:
- **视觉塔的 LLM 底座** ← 你的 transformer-deep / pretraining 专题。
- **VLM 微调** ← 你的 M1 PEFT (LoRA 抗遗忘, M10.3-L4)。
- **VLM 推理部署** ← 你的 M5 (vLLM/量化/投机, 实时 omni 用, M10.5-L4)。
- **VLM 评测** ← 你的 M6 (eval/judge/red-team, M10.6)。
- **VLM agent** ← 你的 multimodal-agent (现在你懂它底层的 VLM 了)。

> 一句话: **M10 给你的 48 工程专题装上了「眼睛和嘴」。** 你原来会造/改/用/评纯文本 LLM, 现在这套能力**全部扩展到多模态**。这是「迁移而非从零」的最好体现 (你的最大优势)。

---

## 4. 本讲小结 + 通往 L2

- M10.1-10.6 是**一条能力链**, 每块 src 被下游复用; `mm_capstone` 验证装配。
- 端到端 mini-VLM 流水线: 图→视觉 token→连接器→LLM→答案 (+画图 +评测), **机制完整**。
- tiny 和真实**一一对应**, 只换规模和预训练 —— 你已懂 VLM 怎么造/训/测。
- 这条流水线**接回你的 48 工程专题** (底座/PEFT/推理/评测/agent), 给它们装上眼睛和嘴。

> **下一讲 L2「VLM 研究前沿 + 找 gap」**: 你会造 VLM 了, 怎么把它变成 PhD 研究? 用 9.3 的 gap 雷达扫多模态前沿, 收敛出对你最友好的研究题目。L2 把 M10 变成你的研究入口。

**动手**: 去 `N1-end-to-end-vlm.ipynb`, 用 `mm_capstone` 检查装配, 然后真跑一条端到端 mini-VLM 流水线 (复用 10.1-10.4 的 src): 图 → 理解 → (画图) → 评测。亲手把 M10 全线跑通一次。
