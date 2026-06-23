# L2 · cross-attention 路线 (Flamingo): 视觉旁路注入

> 30-min lecture · 目标: 理解 Flamingo 式融合 —— perceiver resampler 压缩视觉 token + cross-attention 让文本查视觉。这是处理多图/视频、且不撑爆上下文的经典方案。

---

## 0. cross-attn 路线要解决什么

L1 说过, 把视觉 token 直接拼进 LLM 序列 (投影路线) 有个硬伤: **视觉 token 多就撑爆上下文**。一张高分辨率图上千 patch token, 多张图/一段视频更甚。Flamingo 的方案: **让视觉待在旁路, 文本通过 cross-attention 去「查」它, 视觉不占输入序列。**

```
   文本序列 (占 LLM 上下文): [t1 t2 ... t10]
                              │ 每隔几层插一个 cross-attn 层
   视觉 (旁路, 不占序列):     [压缩后的视觉 token] ←─ 文本 cross-attend 它
```

---

## 1. 第一步: perceiver resampler 压缩视觉 token

视觉塔可能吐出几十上百个 token。Flamingo 先用一个 **perceiver resampler** 把它们压成固定的少量 (如 64 个) token:

```
   N 个视觉 token (变长, 可能很多)
        │  一组可学习的 query (固定 M 个, M << N)
        │  query 通过 cross-attn 从 N 个视觉 token 里「采样汇总」
   M 个压缩视觉 token (固定数量)
```

机制 (就是一次 cross-attention):
$$\text{compressed} = \text{CrossAttn}(Q=\text{query}, K=V=\text{视觉token})$$
逐项: **query** 是 $M$ 个可学习向量 (要多少压缩 token 就设多少); 它们当 Q, 视觉 token 当 K/V; cross-attn 让每个 query「从所有视觉 token 里加权汇总信息」。结果是**固定 $M$ 个**压缩视觉 token, 不管原来有多少 (一张图还是十张图, 都压成 $M$ 个)。

> 这就是 `connectors.CrossAttnConnector` 里的 `resampler`。它的价值: **把变长、可能巨量的视觉 token 压成固定的少量**, 解耦「视觉信息量」和「LLM 要处理的量」。多图/视频也能塞进来。

---

## 2. 第二步: 文本 cross-attend 压缩视觉

压好视觉后, 在 LLM 的层间插入 **cross-attention 层**, 让文本 token 去查视觉:
$$\text{fused} = \text{CrossAttn}(Q=\text{文本token}, K=V=\text{压缩视觉token})$$
逐项: 文本 token 当 Q (它想知道图里有什么), 压缩视觉当 K/V; 每个文本 token 从视觉里取它需要的信息; 结果**残差加回**文本表示。**注意: Q 是文本, 所以输出序列长度 = 文本长度, 视觉没有增加序列。** 这就是 L1 说的「视觉不占序列」。

> Flamingo 还在这些 cross-attn 层加了 **tanh 门控** (gating), 初始化为 0, 让模型从「纯文本 LLM」平滑地开始学着用视觉 —— 不破坏预训练的语言能力。这是「插新模块但不毁旧能力」的经典技巧 (呼应你 PEFT 专题的 adapter 思想)。

---

## 3. cross-attn 路线的得与失

| 维度 | cross-attn (Flamingo) |
|---|---|
| 视觉占输入序列 | **否** (resampler 压缩 + cross-attn 注入) |
| 上下文成本 | **低且恒定** (视觉压成固定 M 个, 不进序列) |
| 多图/视频 | **天然支持** (都压成 M 个) |
| 改 LLM 结构 | **是** (要插 cross-attn 层 + 门控) |
| 复用纯文本 LLM | 部分 (冻 LLM 主体, 训新插的 cross-attn 层) |
| 实现复杂度 | 中高 |

> 一句话权衡: **用「改 LLM 结构 + 实现复杂」换「视觉不撑爆上下文 + 多图视频友好」。** 如果你的 VLM 要吃很多图/视频 (如视频理解、多页文档), cross-attn 路线值得; 如果只是单图问答、想快速起步, 投影路线 (L3) 更省。

---

## 4. 什么时候选 cross-attn

- **多图 / 视频 / 长视觉上下文**: resampler 把任意量视觉压成固定 token, 不爆上下文。
- **想严格保护语言能力**: 门控 cross-attn 从 0 开始, 不动 LLM 主体。
- **不在乎实现复杂、改结构**: 接受工程成本。

反过来, 单图、想用现成 LLM 快速验证 → 别上 cross-attn, 用投影 (L3)。

---

## 5. 本讲小结 + 通往 L3

- cross-attn 路线让**视觉待旁路, 文本 cross-attend 查它**, 视觉不占输入序列。
- **perceiver resampler**: 用 $M$ 个可学习 query, cross-attn 把变长视觉 token 压成固定 $M$ 个 (多图/视频也压成 $M$)。
- **门控 cross-attn**: 插进 LLM 层间, tanh 门从 0 起, 不毁语言能力。
- 得: 上下文成本低恒定、多图视频友好; 失: 改 LLM 结构、实现复杂。

> **下一讲 L3「投影 adapter 路线」**: 与 cross-attn 相反, LLaVA 走极简路线 —— 一个 MLP 把视觉 token 投影成「LLM 看得懂的普通 token」, 拼进序列就完事。简单到不可思议, 效果却出奇好。L3 讲它为什么 work。

**动手**: 在 N1 里, 把视觉 token 数从 16 增到 64, 看 cross-attn 连接器的输出序列长度**是否变化** (应不变, 因为视觉被 resampler 压成固定 query 数), 对比投影连接器 (序列随视觉线性增长)。
