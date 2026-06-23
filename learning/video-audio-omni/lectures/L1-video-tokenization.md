# L1 · 视频 token 化: 图像 + 时间

> 25-min lecture · 目标: 理解视频如何 token 化 —— 时空 patch (spatiotemporal) 如何同时编码空间和运动, 以及帧采样的权衡。把 M10.4 的 token 化扩到时序。

---

## 0. 视频 = 一串图 + 一个时间维

M10.4 把单图变 token。视频是「一串图 + 时间」。最朴素的想法: **每帧当一张图独立 token 化**。但这有两个问题:
1. **token 爆炸**: 一段 10 秒 30fps 的视频 = 300 帧, 每帧上百 token → 几万 token, 撑爆任何 LLM。
2. **丢运动**: 逐帧独立, 模型看不到「帧之间的变化」(运动), 而运动恰恰是视频的核心信息。

> 解法: **时空 patch (spatiotemporal tubelet)** —— 把「几帧 × 一块空间」当一个时空立方体, 切成一个 token。一个 token 同时编码「一小块空间在一小段时间里的样子」, 既压缩 token 数、又自带运动信息。这是 ViViT / Sora 式视频模型的核心。

---

## 1. 时空 patch: 一个 token = 一小段时空

```
   逐帧 patch (朴素):           时空 patch (好做法):
   每帧切成空间 patch            把 (tpatch 帧 × patch×patch 空间) 当一个立方体
   token 数 = T × 每帧patch       token 数 = (T/tpatch) × 每帧patch
   忽略时间                       一个 token 编码「一小段时间的一小块空间」= 含运动
```

机制 (`temporal_tokens.spatiotemporal_patchify`): 把视频切成 $\frac{T}{t_p} \times \frac{H}{p} \times \frac{W}{p}$ 个时空立方体, 每个立方体 ($t_p$ 帧 × $p$×$p$ 像素) 拉平成一个 token。

token 数对比 (你的 N1 会算):
$$\text{逐帧 token 数} = T \cdot \frac{HW}{p^2}, \quad \text{时空 token 数} = \frac{T}{t_p} \cdot \frac{HW}{p^2}$$
逐项: $T$ 帧数, $p$ 空间 patch 大小, $t_p$ 时间 patch 大小。时空 patch 把 token 数压缩了 $t_p$ 倍 (沿时间下采样), 同时每个 token 因为含 $t_p$ 帧, **天然编码了这 $t_p$ 帧间的运动**。

> 一举两得: **压缩 token 数 (省上下文) + 编码运动 (信息更全)**。这就是为什么视频模型用时空 patch 而非逐帧。$t_p$ 越大压缩越多、但时间分辨率越粗 —— 又一个权衡 (该消融, 9.4)。

---

## 2. 帧采样: 另一个压缩旋钮

除了时空 patch, 还有更粗的压缩: **帧采样** —— 不是每帧都要 (30fps 里很多帧几乎一样)。
- **均匀采样**: 每隔 N 帧取一帧 (如 1fps)。简单, 但可能漏掉快速动作。
- **关键帧采样**: 在变化大的地方多取、静止的地方少取。更聪明, 但要先检测变化。

> 帧采样和时空 patch 是两层压缩: 帧采样先粗筛 (取哪些帧), 时空 patch 再细压 (帧×空间立方体)。两者叠加, 把视频从「几万 token」降到 LLM 能吃的量。**视频理解的核心工程就是「怎么在不丢关键信息的前提下狠压 token」** (接 M10.2 的上下文成本、你的 long-context 专题)。

---

## 3. 时空注意力: 怎么处理时空 token

时空 token 进 transformer 后, attention 怎么算? 几种设计:
- **联合时空注意力**: 所有时空 token 互相 attend。最全, 但 $O(n^2)$ 在长视频上爆。
- **分解注意力 (factorized)**: 先空间内 attend、再时间上 attend (或反之)。省算力, ViViT 用这个。
- **窗口注意力**: 只在局部时空窗口内 attend。更省, 牺牲全局。

> 这和你 long-context 专题的「怎么省 attention」是同一类问题, 只是多了时间维。**视频的核心难点 = 时空的二次方爆炸**, 所有设计都在和它斗争。理解这点, 你就理解了视频模型架构的演化逻辑。

---

## 4. 本讲小结 + 通往 L2

- 视频 = 图 + 时间; 逐帧独立 token 化会**爆炸 + 丢运动**。
- **时空 patch**: 把「几帧×一块空间」当一个 token, **压缩 token 数 $t_p$ 倍 + 编码运动**。
- **帧采样** 是另一层压缩 (取哪些帧); 与时空 patch 叠加狠压 token。
- 时空注意力要对抗**时空二次方爆炸** (联合/分解/窗口), 同 long-context 的斗争。

> **下一讲 L2「音频与语音」**: 视频搞定了, 音频呢? 音频是 1D 波形, 直接当 token 太长。L2 讲怎么把波形变成 mel 谱 (音频版的「图」)、再 token 化 —— 这是 Whisper 和音频生成模型的输入处理。

**动手**: 去 `N1-video-tokens.ipynb`, 用 `temporal_tokens` 把合成「移动色块」视频做逐帧 vs 时空 patch, 对比 token 数 (时空压缩 $t_p$ 倍), 并验证运动信号被保留。
