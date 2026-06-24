# L1 · 用 transformer 做扩散: Diffusion Transformer (DiT)

> 28-min lecture · 目标: 理解为什么现代扩散模型把去噪网络从 U-Net 换成 transformer (DiT), 这是你的本行 —— 你的 transformer 知识直接迁移到生成式媒体。

---

## 0. 去噪网络用什么骨架?

M13.1/13.2 的去噪/速度网络是个小 MLP (玩具够用)。但真实扩散模型 (高分辨率图/视频) 的去噪网络是什么?

- **2020-2022**: U-Net (卷积 + 下采样/上采样 + skip connection)。Stable Diffusion 1/2 用它。
- **2023-2026**: **Diffusion Transformer (DiT)** —— 把 U-Net 换成 transformer。SD3、Sora、Flux 都用 DiT。

> **这对你是天大的好消息**: 现代扩散的骨架就是你最熟的 transformer! 你的 transformer-deep / attention / scaling 知识**直接迁移**到图像/视频生成。DiT 让「会 transformer 的人」无缝进入扩散。本讲讲 DiT 怎么把扩散和 transformer 接起来。

---

## 1. DiT: 把数据当 token, transformer 去噪

DiT 的思想和 ViT (M10.1) 一脉相承: **把数据切成 patch/token, 用 transformer 处理。**

```
   图 (或 latent) → 切成 patch token → + 时间条件 + 类别/文本条件
        → transformer blocks (self-attention + FFN)
        → 预测每个 patch 的噪声
```

逐点 (对照 `dit.build_dit`):
- **数据 token 化**: 把带噪数据切成 patch, 每个 patch 当一个 token (玩具版里一个 2D 点就是一个 token)。
- **条件注入**: 扩散的去噪网络需要知道**当前噪声水平 ($t$)** 和**条件 (类别/文本)**。DiT 把它们做成 embedding, 当额外的「条件 token」或通过 adaptive LayerNorm 注入。
- **transformer 处理**: 标准 self-attention + FFN, 让 patch 之间互相 attend (全局感受野, 比 U-Net 的局部卷积更全局)。
- **输出**: 每个 patch token 输出它的预测噪声 (或速度)。

> 关键: **DiT = ViT 的结构 + 扩散的目标 (预测噪声)**。你已经会 ViT (M10.1) 和扩散 (M13.1), DiT 就是把两者拼起来。`dit.py` 的 DiT 把点当 token、把时间/类别当条件 token, 用 transformer 去噪 —— 机制和真 DiT 一致, 只是规模小。

---

## 2. 为什么 transformer 胜过 U-Net

| | U-Net (卷积) | DiT (transformer) |
|---|---|---|
| 感受野 | 局部 (卷积核) | 全局 (attention) |
| 归纳偏置 | 强 (局部性) | 弱 (数据驱动) |
| scaling | 一般 | **好** (transformer 的 scaling law) |
| 多模态条件 | 别扭 | **自然** (条件当 token) |

逐点交代为什么 DiT 赢:
- **scaling**: transformer 有干净的 scaling law (你的 scaling-infra 专题), 加参数/数据稳定变好。U-Net 的 scaling 不如它。这是 DiT 论文的核心发现 —— **扩散也遵循 transformer 的 scaling**。
- **全局感受野**: attention 让任意两个 patch 直接交互, 利于全局一致性 (图像/视频)。
- **多模态条件自然**: 文本条件、类别、时间都当 token 拼进序列, 和 M10 的多模态融合 (early-fusion) 一脉相承。

> 一句话: **DiT 把扩散接入了 transformer 的红利** (scaling、全局、多模态)。这也是为什么 2026 的大扩散模型几乎都是 DiT —— 它能像 LLM 一样 scale。你的 LLM scaling 直觉在这里再次变现。

---

## 3. 条件注入的细节 (adaLN)

DiT 注入条件 (时间 + 类别/文本) 有个常用技巧 **adaptive LayerNorm (adaLN)**: 用条件去**调制** transformer 每层 LayerNorm 的缩放和偏移。简化版 (本专题用) 是把条件当额外 token 拼进序列。

> 知道有 adaLN 这个细节即可 (DiT 论文证明 adaLN-Zero 效果最好)。核心思想不变: **让每一层都"知道"当前噪声水平和要生成什么**。本专题用「条件 token」简化演示, 抓住「条件如何进入 transformer」这个本质。

---

## 4. 本讲小结 + 通往 L2

- 现代扩散骨架从 **U-Net → DiT (Diffusion Transformer)**; 你的 transformer 知识直接迁移。
- **DiT = ViT 结构 + 扩散目标**: 数据切 token, transformer 去噪, 时间/条件当 token 注入。
- DiT 胜 U-Net: **scaling 好** (transformer scaling law) + 全局感受野 + 多模态条件自然。
- 条件注入用 adaLN (或简化的条件 token); 让每层知道噪声水平 + 要生成什么。

> **下一讲 L2「latent diffusion」**: DiT 直接在像素上扩散仍然贵 (高分辨率 patch 多)。latent diffusion 在一个压缩的 **latent 空间** (VAE/VQ 编码后) 做扩散, 大幅省算力。这接你 M10.4 的 VQ。

**动手**: 去 `N1-dit-conditional.ipynb`, 用 `dit.py` 在 4 类高斯团上训一个 DiT, 指定类别生成对应的团, 看 transformer 去噪器如何做条件生成。
