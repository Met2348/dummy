# L3 · SigLIP 与 DINOv2: 两条改进视觉塔的路

> 30-min lecture · 目标: 理解 SigLIP 如何用 sigmoid 损失摆脱 CLIP 对大 batch 的依赖, 以及 DINOv2 如何完全不用文本就学出强视觉表示。这两个是 2026 VLM 最常用的视觉塔。

---

## 0. 为什么不止 CLIP

L2 的 CLIP 很强, 但有软肋 (大 batch 依赖)。2026 的 VLM 视觉塔通常是 CLIP 的两个进化方向之一:
- **SigLIP**: 还是图文对比, 但换损失, 摆脱大 batch。**LLaVA、多数开源 VLM 的默认视觉塔。**
- **DINOv2**: 完全不用文本, 纯**自监督**学视觉, 表示在密集任务 (分割/深度) 上更强。

理解这两条路, 你才能在造 VLM 时**选对视觉塔** (M10.2 的决策依赖它)。

---

## 1. SigLIP: 把对比损失从 softmax 换成 sigmoid

回忆 L2: InfoNCE 的分母要「batch 内所有负样本」做 softmax 归一化 → batch 越大越好 → 工程上贵。**SigLIP 的洞察: 不做全 batch 归一化, 把每个图文对当一个独立的二分类。**

$$\mathcal{L}_{\text{SigLIP}} = \frac{1}{B^2}\sum_{i=1}^{B}\sum_{j=1}^{B} \log\big(1 + \exp(-z_{ij}\cdot(S_{ij}/\tau + b))\big)$$

逐项交代:
- **$S_{ij}$**: 图 $i$、文 $j$ 的相似度 (同 L2)。
- **$z_{ij}$**: 标签。配对 ($i=j$) 时 $z_{ij}=+1$; 不配对时 $z_{ij}=-1$。
- **$\tau$ (温度)、$b$ (bias)**: 缩放和偏置。$b$ 是 SigLIP 特有的, 用来抵消「负对远多于正对」的不平衡 (一个 batch 里 B 个正对、$B^2-B$ 个负对)。
- **$\log(1+\exp(-\cdot))$**: 这是 **softplus**, 等价于 $-\log\text{sigmoid}(\cdot)$ —— 标准的二分类损失。
- **整体**: 对每个 $(i,j)$ 对独立判「配不配」, 不跨样本归一化。

**关键好处**: 因为每对独立, **不需要大 batch 来凑负样本**, 数值也更稳。这让 SigLIP 在更小 batch、更省资源下达到 CLIP 同等甚至更好的效果 —— 所以它成了开源 VLM 的默认视觉塔。

> `contrastive.sigmoid_loss` 实现了它。对比 N2 里 InfoNCE 和 sigmoid 两种损失, 体会「softmax 全局归一化 vs sigmoid 成对独立」的区别。

---

## 2. DINOv2: 完全不用文本的自监督视觉

CLIP/SigLIP 都需要「图文对」。但互联网的图文对有噪声、有覆盖盲区。**DINOv2 走另一条路: 只用图、不用任何文本/标签, 纯自监督学视觉表示。**

核心思想 (自蒸馏 self-distillation, 简化版):
```
   一张图 → 两种不同增强 (裁剪/颜色抖动) → view A, view B
   student 网络看 view A, teacher 网络看 view B
   目标: student(A) 的表示 ≈ teacher(B) 的表示
        (同一张图的不同视角应有一致的表示)
   teacher = student 的指数滑动平均 (EMA), 不直接训练
```

逐点交代为什么 work:
- **不变性**: 强迫「同一张图的不同增强」有相似表示 → 模型学到对增强不变的**语义**特征 (而非表面像素)。
- **避免坍缩**: 朴素地「让两个视角一样」会坍缩成「所有图都输出同一个常数」。DINO 用 centering + sharpening + EMA teacher 等技巧防止坍缩 (细节略, 知道有这个问题即可)。
- **结果**: DINOv2 的表示在**密集任务** (语义分割、深度估计、对应点) 上特别强, 因为它学的是细粒度的视觉结构, 而非只为图文对齐。

> 对 VLM 的意义: DINOv2 的视觉 token 空间细节更丰富, 但**不天然对齐语言** (没用文本训)。所以有些 VLM 用 SigLIP (懂语言) + DINOv2 (懂细节) **双塔**, 取长补短。

---

## 3. 三个视觉塔怎么选 (通往 M10.2)

| 视觉塔 | 训练信号 | 强在 | 弱在 | VLM 用法 |
|---|---|---|---|---|
| **CLIP** | 图文对比 (InfoNCE) | 语言对齐、zero-shot | 大 batch、密集任务一般 | 经典选择 |
| **SigLIP** | 图文对比 (sigmoid) | 语言对齐 + 省资源 | 同上但缓解 | **开源 VLM 默认** |
| **DINOv2** | 纯自监督 (无文本) | 密集/细粒度视觉 | 不天然对齐语言 | 补细节, 常和 SigLIP 双塔 |

> 选型逻辑: 要「懂语言」→ SigLIP/CLIP; 要「细节强」→ DINOv2; 都要 → 双塔。这是你造 VLM 时第一个架构决定 (M10.2 会接着讲怎么把选好的塔接进 LLM)。

---

## 4. 本讲小结 + 通往 L4

- **SigLIP**: 把 InfoNCE 的全 batch softmax 换成**成对 sigmoid (softplus)** 损失 + bias 抵消不平衡; 摆脱大 batch 依赖, 成开源 VLM 默认视觉塔。
- **DINOv2**: 纯**自监督** (自蒸馏, 同图不同增强表示一致), 不用文本; 密集/细粒度视觉强, 但不天然对齐语言。
- 三塔选型: 懂语言用 SigLIP/CLIP, 懂细节用 DINOv2, 兼顾用双塔。

> **下一讲 L4「视觉表示的性质」**: 有了视觉塔, 接进 LLM 前要回答几个工程问题 —— 用 [CLS] 还是所有 patch token? 哪一层的表示最适合喂 LLM? 冻结视觉塔还是一起微调? L4 讲这些「接口」决策, 直接通往 M10.2 的融合架构。

**动手**: 在 N2 里加一格, 把同一批图文对分别用 `info_nce_loss` 和 `sigmoid_loss` 算, 比较两者对噪声的敏感度 —— 体会两种损失的脾气差异。
