# L1 · 从像素到 token: Vision Transformer 怎么吃图

> 35-min lecture · 目标: 理解 ViT 如何把一张图变成一串 token, 让「为文本造的 transformer」也能处理图像。这是整个 VLM 的视觉地基。

---

## 0. 一个跨界的问题

你已经会 transformer 处理文本: 文本是天然的离散 token 序列 (词/子词)。但图像是连续的像素网格 (H×W×3), 没有现成的「token」。**怎么让吃 token 的 transformer 吃图?**

> 2020 年 ViT (Vision Transformer) 的答案简单到惊人: **把图切成小方块 (patch), 每个 patch 当一个 token。** 一张 224×224 的图切成 16×16 的 patch, 就是 14×14=196 个「视觉词」。然后……就当文本序列处理。这个「图像即 patch 序列」的想法, 是后来所有 VLM 视觉塔的起点。

---

## 1. ViT 的四步 (本讲的骨架)

```
   图像 (H,W,3)
      │ ① patchify: 切成 N 个 patch
   N 个 patch (每个 P×P×3 拉平)
      │ ② 线性投影: 每个 patch → d 维向量 (patch embedding)
   N 个 patch embedding (N, d)
      │ ③ 加 [CLS] token + 位置编码
   (N+1, d)
      │ ④ transformer 编码 (和文本一模一样)
   N+1 个视觉 token —— [CLS] 是图级表示, 其余是 patch 级
```

逐步交代:

### ① patchify (切块)
把 H×W×3 的图切成不重叠的 P×P patch, 共 $N = \frac{H}{P}\times\frac{W}{P}$ 个。每个 patch 拉平成一个 $P^2\cdot 3$ 维向量。`tiny_vit.patchify` 就做这个 (16×16 图、patch=4 → 16 个 patch, 每个 4×4×3=48 维)。

### ② patch embedding (投影成 token)
一个线性层 $W \in \mathbb{R}^{(P^2\cdot 3)\times d}$ 把每个 patch 投影到 $d$ 维:
$$e_i = W\, p_i + b$$
逐项: $p_i$ 是第 $i$ 个拉平的 patch ($P^2\cdot3$ 维); $W,b$ 是可学习投影; $e_i$ 是第 $i$ 个 **patch embedding** ($d$ 维), 它就是一个「视觉 token」。这一步把「一块像素」变成「transformer 能吃的向量」。

### ③ [CLS] token + 位置编码
- **[CLS] token**: 在序列最前面加一个可学习的特殊 token。transformer 跑完后, 这个位置的输出被当作**整张图的表示** (类似 BERT 的 [CLS])。VLM 里有时用它、有时用所有 patch token, 看连接器设计 (M10.2)。
- **位置编码**: transformer 本身不知道 patch 的空间位置 (它对顺序无感)。加位置编码告诉它「这个 patch 在左上、那个在右下」。ViT 通常用可学习的位置 embedding。

### ④ transformer 编码
此后**和文本 transformer 完全一样**: 多层 self-attention + FFN。每个 patch token 通过 attention「看」其它所有 patch, 聚合全图信息。输出是 $N+1$ 个**视觉 token** —— 这串 token 就是后面要喂给 LLM 的「视觉一半」。

> 一句话: **ViT = patchify + 线性投影 + 「图像版 BERT」。** 你已经会的 transformer 知识, 90% 直接迁移。新增的只有「怎么把图变成 token 序列」这一层。

---

## 2. 为什么这样能 work (反直觉之处)

新手疑惑: 把图粗暴切块、丢掉精细的卷积归纳偏置 (局部性/平移不变性), 凭什么比 CNN 强?

- **数据足够时, attention 能自己学到空间关系。** CNN 把「局部性」硬编码进结构; ViT 不假设, 让 attention 从数据里学。数据少时 CNN 占优 (偏置帮忙); 数据多时 ViT 反超 (不被偏置限制)。这是「归纳偏置 vs 数据规模」的经典权衡 (回忆你 transformer-deep 专题的同款讨论)。
- **统一架构的红利**: 图和文用同一种架构 (transformer), 才好**融合**成 VLM (M10.2)。如果视觉用 CNN、语言用 transformer, 拼接会别扭。ViT 的最大价值之一就是「让视觉和语言说同一种话 (token)」。

---

## 3. patch 大小的权衡

patch 大小 $P$ 是 ViT 的关键超参:
- **$P$ 小** (如 8): patch 多、序列长 → 计算贵 (attention 是 $O(N^2)$), 但**细节保留好**。
- **$P$ 大** (如 16/32): patch 少、序列短 → 快, 但**丢细节** (一个 patch 糊成一个 token)。

> 这直接影响 VLM 的成本和能力: 视觉 token 越多, LLM 要处理的上下文越长 (接你的 long-context 专题)。高分辨率/密集任务要小 patch, 但代价是 token 爆炸。VLM 工程里「怎么压缩视觉 token」是个真问题 (M10.5 会再碰)。

---

## 4. 本讲小结 + 通往 L2

- ViT 把图像变成 token 序列: **patchify → patch embedding → [CLS]+位置编码 → transformer**。
- patch embedding $e_i = W p_i + b$ 把「一块像素」变成「一个视觉 token」。
- 能 work 是因为「数据足够时 attention 能学空间关系」+「统一架构好融合」。
- patch 大小权衡细节 vs 成本; 视觉 token 数直接影响 VLM 上下文长度。

> **下一讲 L2「对比学习与 CLIP」**: ViT 给了视觉 token, 但它怎么「懂语言」? 答案是**对比学习**: 让图和它的文字描述在向量空间里靠近。CLIP 用 InfoNCE 损失做这件事, 是几乎所有 VLM 视觉塔的训练方式。

**动手**: 去 `N1-vit-patchify.ipynb`, 用 `tiny_vit` 把一张合成图 patchify, 可视化 16 个 patch, 再跑一遍 mini-ViT 看视觉 token 的形状。亲手确认「图 → token 序列」。
