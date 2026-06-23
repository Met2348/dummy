# L3 · 投影 adapter 路线 (LLaVA): 极简却出奇有效

> 30-min lecture · 目标: 理解 LLaVA 为什么用「一个 MLP」就造出了能用的 VLM, 以及这个极简路线为什么 work、何时够用。

---

## 0. 一个反直觉的成功

L2 的 Flamingo 又改结构又加门控, 复杂。2023 年 LLaVA 反其道而行, 用了一个**简单到让人怀疑**的方案:

> **把视觉塔 (CLIP) 输出的视觉 token, 用一个 MLP 投影到 LLM 的 embedding 空间, 然后当普通文本 token 一样拼进 LLM 的输入序列。完事。** 视觉塔冻结, LLM 也基本冻结/轻调, 只训那个小 MLP。就这样, 造出了当时 SOTA 级的开源 VLM。

这个极简路线现在是开源 VLM 的主流起点。本讲讲它为什么够用。

---

## 1. 机制: 一个投影, 把视觉 token「翻译」成 LLM 的母语

```
   图 → CLIP/SigLIP ViT (冻结) → N 个视觉 token (vis_dim 维)
        │  投影连接器: 一个 MLP   W2·GELU(W1·v)
   N 个「LLM 空间」的视觉 token (llm_dim 维)
        │  当普通 token 拼进输入序列
   LLM 输入: [视觉 token ×N | 文本 token ×M]
        │  统一的 LLM 自注意力处理 (视觉和文本互相 attend)
   输出文字
```

核心就一行 (`connectors.ProjectionConnector`):
$$v'_i = W_2\,\text{GELU}(W_1 v_i)$$
逐项: $v_i$ 是第 $i$ 个视觉 token (来自冻结视觉塔); $W_1, W_2$ 是 MLP 的两层 (唯一要训的); $\text{GELU}$ 是激活; $v'_i$ 是投影到 LLM embedding 空间的视觉 token。然后 $[v'_1,...,v'_N]$ 直接拼在文本 token 前面, 当作 LLM 输入序列的一部分。

> 为什么这么简单还 work: 关键是**视觉塔已经懂语言** (M10.1 的对比学习让 CLIP 表示和语言对齐)。所以视觉 token 离「LLM 能懂的东西」其实不远, 一个 MLP 就能架起这座小桥。**M10.1 的对比学习 + M10.3 的投影 = LLaVA 成功的两半。**

---

## 2. 投影路线的得与失

| 维度 | 投影 (LLaVA) |
|---|---|
| 视觉占输入序列 | **是** (+N token) |
| 上下文成本 | 随视觉 token 线性增长 (高分辨率会爆) |
| 改 LLM 结构 | **否** (复用现成 LLM) |
| 复用纯文本 LLM | **是** (冻结/轻调) |
| 要训的参数 | **极少** (一个 MLP) |
| 实现复杂度 | **极低** |
| 多图/视频 | 勉强 (token 线性堆, 容易爆) |

> 一句话权衡: **用「视觉吃上下文」换「极简 + 复用现成 LLM + 训练便宜」。** 单图问答场景, 视觉 token 不算多, 这个代价完全可接受。所以 LLaVA 路线成了「快速造一个能用 VLM」的默认起点。

---

## 3. 为什么 LLaVA 路线是博0 的最佳起点

- **便宜**: 视觉塔冻、LLM 冻, 只训一个 MLP。单卡能跑。
- **简单**: 没有改结构、没有门控、没有 resampler。代码几十行。
- **可复现**: 组件都是现成的 (CLIP + LLaMA + MLP), 容易搭。
- **够强**: 在单图问答上效果就很好, 是很多研究的 baseline。

> 给造 VLM 的现实建议: **第一个 VLM 就用投影路线** (你的 N2 会真搭一个: mini-ViT + MLP + tiny LLM)。先把最简单的跑通、理解透, 再考虑要不要为多图/视频上 cross-attn (L2) 或为生成上 early-fusion (L4)。**别一上来就上复杂架构** —— 这呼应你 9.2 的 MVE 精神 (最小验证先行)。

---

## 4. 投影路线的局限 (何时该换)

- **高分辨率 / 密集任务**: 视觉 token 太多, 撑爆上下文 → 考虑 resampler 压缩 (借 L2 的 perceiver) 或 cross-attn。
- **多图 / 视频**: token 线性堆叠, 很快爆 → cross-attn 更合适。
- **要生成图像**: 投影路线只「读」图不「画」图 → 需要 early-fusion + 视觉 token 化 (L4 + M10.4)。

这些局限正是 L4 (early-fusion) 和后续专题要补的。但对「读图问答」这个最常见场景, 投影路线就够。

---

## 5. 本讲小结 + 通往 L4

- LLaVA: **一个 MLP** 把视觉 token 投影到 LLM 空间, 当普通 token 拼进序列, 视觉塔+LLM 冻结, 只训 MLP。
- 能 work 是因为**视觉塔已懂语言** (M10.1 对比学习); MLP 只需架最后一小段桥。
- 得: 极简 + 复用现成 LLM + 训练便宜; 失: 视觉吃上下文, 多图/高分辨率会爆。
- **博0 造 VLM 的最佳起点** (最小验证先行)。

> **下一讲 L4「early-fusion + 决策树」**: 投影和 cross-attn 都把视觉文本当「两类东西」融合。early-fusion (Chameleon/Fuyu) 更激进: 从第一层就把视觉文本当**同一种 token**, 一锅煮 —— 这是「理解+生成一体」VLM 的基础 (通往 M10.4)。L4 还给你三路线的完整决策树。

**动手**: 去 `N2-llava-projection.ipynb`, 用投影连接器把 M10.1 的 mini-ViT 真的接到一个 tiny LLM 上, 跑通「图+问题→输出」的前向, 亲手搭一个最小 LLaVA。
