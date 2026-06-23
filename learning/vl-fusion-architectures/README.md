# 10.2 vl-fusion-architectures — 视觉怎么接进 LLM

> **Module 10「多模态/VLM 基础」· 第 2 专题 (VLM 最关键的架构决定)**
> 视觉塔 (10.1) 给了视觉 token, LLM 吃文本 token —— **两者怎么融合?** 这是造 VLM 第一个、也最该想清的决定。

---

## 这个专题要解决的真问题

```
   视觉 token ──→ ??? ──→ LLM
```

三条主流路线, 决定 VLM 的架构/成本/能力:
- **投影 (LLaVA)**: 一个 MLP 把视觉 token 投到 LLM 空间, 当普通 token 拼进序列。最简。
- **cross-attn (Flamingo)**: resampler 压缩视觉 + cross-attn 注入, 视觉不占序列。多图/视频友好。
- **early-fusion (Chameleon)**: 视觉文本同质 token 一锅煮, 解锁理解+生成一体。

> 两个区分维度: **视觉占不占 LLM 序列** (成本) + **改不改 LLM/能否复用文本预训练** (门槛)。没有最好, 只有适配约束 —— 本专题给你一张决策树。

## 学习路径 (4 讲)

| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-the-fusion-question.md` | 核心问题 + 三路线地图 + 两个区分维度 |
| L2 | `lectures/L2-cross-attention-flamingo.md` | cross-attn: resampler 压缩 + 文本查视觉, 视觉不占序列 |
| L3 | `lectures/L3-projection-llava.md` | 投影: 一个 MLP, 极简却出奇有效 (博0 起点) |
| L4 | `lectures/L4-early-fusion-decision-tree.md` | early-fusion 理解+生成一体 + 完整决策树 |

## 动手 (2 个 notebook)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-three-connectors.ipynb` | 用 `src/connectors.py` 跑三种连接器, 对比输出序列长度 (cross-attn 不增长!), 改视觉 token 数看差异, 用决策树给场景选路线 |
| `notebooks/N2-llava-projection.ipynb` | 用投影连接器把 10.1 的 mini-ViT 真接到 tiny LLM, 跑通「图+问题→输出」, 亲手搭最小 LLaVA |

## 工具 (`src/`)
- `connectors.py` — 三种连接器 (projection/cross_attn/early_fusion) torch 实现 + 对比表

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / torch (tiny CPU 秒跑) / numpy / pandas。

## 完成本专题后你应该能
- [ ] 说清三种 VL 融合路线各优化什么、各自得失
- [ ] 用决策树给一个 VLM 场景选对融合路线
- [ ] 解释 perceiver resampler 怎么把变长视觉压成固定 token
- [ ] 说清 LLaVA 投影为什么极简还 work (视觉塔已懂语言)
- [ ] 搭一个投影路线的 mini-LLaVA 前向

---
## 在 Module 10 中的位置
```
  10.1 vision-encoders ✅ → 10.2 vl-fusion ◄你在这 → 10.3 vlm-training (训出来) → 10.4-10.7
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module10-multimodal-vlm-design.md`
