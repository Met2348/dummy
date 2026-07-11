# Interview-Prep Track 蓝图（研究主线 / coding 保险）

> 2026-07-11 · 目标画像:**研究岗为主(前沿实验室 RS/RE + PhD 暑研),不走 heavy-coding;coding 作"技术面不翻车"的保险层**。
>
> 定位:这不是"第 86 个学习专题"。85 专题给了**广度**;本 track 治的是**把知识变成面试可演示的表现**——限时、口头、真框架。

## 诊断(为什么不是补内容)

| 面试考点 | 现状 | 缺口 |
|---------|------|------|
| ① 算法 coding(LeetCode) | 零 | 二元硬过滤,连研究岗也有 1–2 轮 |
| ② 真框架手写(live PyTorch) | 全 mock/stdlib | 要闭卷敲、讲清 shape |
| ③ ML 系统设计(格式) | 有素材没格式 | 需 45min 口述格式 |
| ④ ML/DL 基础 rapid-fire | EE 可能有洞 | 要秒答 + 说人话 |
| ⑤ 有真数字的 hero project | 自学 notebook | behavioral 挖 impact |
| ⑥ 论文复现 + 开放式科研推理 | Module 9 打底 | 真复现 + 口头 drill |

## 配比(研究 70% / 保险 30%)

**保险层(本 track 真建,自包含、无需 GPU/PhD 仓对接):**
- `mlcoding/` — 真 PyTorch 从零,**面试保险 × 研究肌肉双用**。每模块 `_self_test()` 真跑真反传(CPU)。
- `leetcode/` — 15 高频 pattern 的范例解 + 间隔复习 tracker(地板不是天花板)。
- `mlqa/` — ML/DL rapid-fire 题库 + `quiz()` 自测。

**研究主线(出 spec,不在本 track 硬建 —— 对接已有 PhD 仓 `research/`,避免重复):**
- Hero project:judge-internals / 可解释性方向在**真模型(GPT-2/Pythia)**上真复现(probing / activation patching / logit lens / 小 SAE),出真数字 + 公开 repo。
- 论文口头 drill 甲板(10 经典 + 10 前沿)。
- 开放式科研推理题库。

## `learning/interview-prep/` 结构

```
interview-prep/
├─ README.md
├─ environment/{requirements.txt, verify_env.py}   # 本 track 需要 torch(唯一破例)
├─ src/
│  ├─ mlcoding/
│  │   ├─ attention.py        scaled-dot-product + MHA(因果掩码,从零非 nn.MultiheadAttention)
│  │   ├─ norm.py             LayerNorm + RMSNorm 从零,对拍 torch
│  │   ├─ rope.py             RoPE 旋转位置编码,应用到 q/k
│  │   ├─ transformer_block.py 预归一化 block(attn + MLP + residual)
│  │   ├─ sampling.py         greedy/temperature/top-k/top-p/beam
│  │   ├─ kv_cache.py         增量解码 + KV cache,对拍全量重算
│  │   ├─ lora.py             LoRA 线性层包冻结 Linear,真反传 + 参数量核对
│  │   ├─ training_loop.py    真训练循环:warmup+cosine、grad clip、eval,拟合玩具任务
│  │   └─ bpe.py              最小 BPE:训练 merges + encode/decode 往返
│  ├─ leetcode/
│  │   ├─ patterns.py         15 pattern × 1 范例解(带复杂度注释)
│  │   └─ tracker.py          JSON 间隔复习追踪器(确定性,无 random)
│  ├─ mlqa/
│  │   └─ qbank.py            60+ 结构化 Q/A + quiz 自测
│  └─ tests/test_all.py       import 各模块调 _self_test()
└─ lectures/                  01..NN 面试策略/配比/各层打法
```

## 约定(沿用全仓)

- 每个 src 模块带 `_self_test()`;`tests/test_all.py` 汇总。
- **唯一破例**:本 track 需 `torch`(面试考真框架,mock 无意义)。CPU 可跑,tiny 张量。
- 无 emoji 于代码输出(用 `[PASS]/[FAIL]/[OK]`);`$env:PYTHONIOENCODING="utf-8"`。
- 收尾:commit + `git tag interview-prep`。

## 一句话

> 你不缺第 86 个专题,缺的是**把 85 个专题的知识,在限时 / 口头 / 真框架下演示出来**。研究是矛,coding 是盾——这个 track 把盾打厚,把矛的复现留给你的 PhD 仓。
