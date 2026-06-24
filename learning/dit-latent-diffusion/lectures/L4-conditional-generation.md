# L4 · 条件生成: 文本如何接进扩散 (文生图接口)

> 22-min lecture · 目标: 理解文本条件如何编码并注入 DiT, 完成「文生图」的接口。收口 M13.3。

---

## 0. 从「类别」到「文本」条件

L3 的 CFG 用**类别**当条件 (生成第几类)。但真实的文生图条件是**自由文本** (「一只戴帽子的红猫在月球上」)。文本怎么接进扩散? 本讲讲这个接口, 它把 M13.3 的所有部件 (DiT + CFG) 连成一个文生图系统。

---

## 1. 文本条件的三步

```
   文本 "a red cat" 
      │ ① 文本编码器 (如 CLIP text / T5): 文本 → 文本 embedding 序列
   文本 embedding
      │ ② 注入 DiT: 通过 cross-attention 或条件 token, 让去噪看到文本
   DiT 去噪 (每步都"看着"文本)
      │ ③ CFG: 用 guidance scale 控制贴合度 (L3)
   生成符合文本的图
```

逐步交代:
- **① 文本编码**: 用一个**文本编码器** (CLIP 的文本塔 M10.1, 或 T5) 把文本变成 embedding 序列。这接你 M10.1 (CLIP 文本塔) —— **文生图复用了 VLM 的文本编码**。
- **② 注入 DiT**: 文本 embedding 通过 **cross-attention** (M10.2 的 Flamingo 路线!) 注入 DiT 的每一层, 让去噪过程「看着文本」。或者像本专题简化版当条件 token。
- **③ CFG**: 用 L3 的 guidance scale 控制「生成多贴合文本」。

> 关键观察: **文生图 = 你已学的部件的组合**:
> - 文本编码 ← M10.1 (CLIP/文本塔)
> - 文本注入 ← M10.2 (cross-attention)
> - 去噪骨架 ← M13.3-L1 (DiT)
> - 强度控制 ← M13.3-L3 (CFG)
> - 压缩空间 ← M13.3-L2 (latent diffusion)
> **Stable Diffusion = 这些部件拼起来。** 你现在懂了文生图的每一块!

---

## 2. 一个完整文生图系统的样子 (Stable Diffusion)

```
   "a red cat"
      → CLIP/T5 文本编码器 → 文本 embedding (M10.1)
      → VAE 编码: 从噪声 latent 开始 (M13.3-L2)
      → DiT 去噪 T 步 (M13.3-L1):
           每步 cross-attend 文本 (M10.2) + CFG 放大文本方向 (M13.3-L3)
      → 去噪完成的 latent
      → VAE 解码 → 图 (M13.3-L2)
```

> 这就是一个真实文生图模型的完整流水线。每一块你都在 M10/M13 学过。**把它们拼起来, 就是 Stable Diffusion / SD3。** tiny 和真实只差规模和预训练 (同 M10.7 的 mini-VLM)。

---

## 3. 条件生成的其它形态

文本只是一种条件。扩散能接各种条件:
- **图像条件**: 给一张图 + 文本, 编辑/变体 (img2img、inpainting)。
- **结构条件**: ControlNet (给边缘图/姿态/深度, 控制生成结构)。
- **多条件组合**: 文本 + 区域 + 风格同时控制。

> 这些都是「往去噪过程注入更多控制信号」的变体。理解了「条件如何注入 + CFG 如何控制强度」这个核心 (本专题), 这些花样都是同一思想的扩展。可控生成 (更精细的条件控制) 是个活跃研究方向 (gap)。

---

## 4. M13.3 收口: 扩散的骨架与控制

```
   M13.3:
   L1 DiT: 去噪网络用 transformer (你的本行, scaling 好)
   L2 latent diffusion: 在压缩空间扩散, 省算力 (接 M10.4 VQ/VAE)
   L3 CFG: guidance scale 控制条件贴合度
   L4 条件接口: 文本编码 + cross-attn 注入 + CFG = 文生图
   ───────────────────────────────────────────
   = 一个完整的条件扩散生成系统 (Stable Diffusion 的全部部件)
```

> 你现在掌握了扩散的**机制 (M13.1) + 速度 (M13.2) + 骨架与控制 (M13.3)**。这三个专题让你从「不懂扩散」到「懂 Stable Diffusion 怎么造」。接下来 M13.4 加时间维 (视频), M13.5 当世界模型, M13.6 回文本 (dLLM)。

---

## 5. 本讲小结 (M13.3 收口) + 通往 M13.4

- 文本条件三步: **文本编码** (CLIP/T5, 接 M10.1) → **注入 DiT** (cross-attention, 接 M10.2) → **CFG 控制贴合** (L3)。
- **文生图 = 你已学部件的组合**: 文本塔 + cross-attn + DiT + CFG + latent diffusion = Stable Diffusion。
- 其它条件: 图像 (img2img/inpaint) / 结构 (ControlNet) / 多条件; 都是「注入更多控制信号」。
- M13.3 = 扩散的骨架 (DiT) + 省算力 (latent) + 控制 (CFG/条件)。

> **下一专题 M13.4「video-generation」**: 扩散从图扩到**视频** (加时间维)。时空扩散、Sora 式时空 patch、Open-Sora 成本教训。这接你 M10.5 的视频时空 token 化。

**动手**: 画出 (或写出) 一个完整文生图系统的数据流, 标出每一块来自你哪个专题 (M10.1 文本塔 / M10.2 cross-attn / M13.3 DiT+CFG+latent)。确认你真的懂了 Stable Diffusion 的每一块。
