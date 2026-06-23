# L4 · early-fusion + 决策树: 统一 token 与如何选

> 30-min lecture · 目标: 理解 early-fusion (Chameleon/Fuyu) 如何把视觉文本当同一种 token 一锅煮, 以及用一张决策树给你的 VLM 选对融合路线。

---

## 0. 第三条路: 干脆别分视觉和文本

投影 (L3) 和 cross-attn (L2) 都把视觉、文本当**两类东西**, 想办法融合。early-fusion 更激进:

> **从第一层就把视觉 token 和文本 token 当作同一种 token, 混在一个序列里, 用一个统一的 transformer 处理。没有「视觉塔 + 连接器 + LLM」的分层, 只有一个吃混合 token 的模型。** (Chameleon / Fuyu 路线)

```
   投影/cross-attn: [视觉塔]→[连接器]→[LLM]  (分阶段, 视觉文本分治)
   early-fusion:    图和文都 → token → [一个统一 transformer]  (同质, 一锅煮)
```

---

## 1. 机制: 一切皆 token

early-fusion 把图像也变成离散/连续 token (Fuyu 直接把 patch 线性投影当 token, Chameleon 用 VQ 把图变离散 token, 见 M10.4), 然后和文本 token 拼成一个序列, 喂给一个**从头训练的统一 transformer**:

```
   图 → patch/VQ → 视觉 token ┐
                              ├→ [一个序列] → 统一 transformer → 输出 token
   文本 → 文本 token ─────────┘            (视觉和文本无差别对待)
```

`connectors.EarlyFusionConnector` 演示了最简版: 视觉 token 投到同一空间, 直接和文本拼接, 没有特殊处理 —— 因为「特殊处理」恰恰是 early-fusion 想避免的。

---

## 2. early-fusion 的独特价值: 理解 + 生成一体

为什么要从头训这么贵的路线? 因为它解锁一件投影/cross-attn 难做的事: **同一个模型既能读图、又能画图。**

- 如果视觉是离散 token (VQ, M10.4), 那么「生成图」就是「自回归生成视觉 token」—— 和生成文本**完全一样的机制**。
- 于是一个 early-fusion 模型可以: 输入文字 → 生成图 token → 解码成图 (文生图); 也可以输入图 → 生成文字 (图生文)。**理解和生成统一在「自回归生成 token」下。**

> 这是 any-to-any 多模态模型 (Chameleon) 的基础, 也通往 M10.4 (视觉 token 化与生成)。投影路线只能「读」图 (视觉当输入), early-fusion 能「读」也能「写」。代价: 通常得从头训 (不能复用现成纯文本 LLM), 贵。

---

## 3. 三路线完整决策树

```
   要造 VLM, 怎么选融合路线?
   │
   ├─ 要让模型「生成图像」/ any-to-any?
   │     是 → early-fusion (Chameleon) + 视觉 token 化 (M10.4)  [贵, 从头训]
   │     否 ↓
   │
   ├─ 要处理「多图 / 视频 / 高分辨率」且不想撑爆上下文?
   │     是 → cross-attn (Flamingo) + perceiver resampler  [改结构, 实现复杂]
   │     否 ↓
   │
   └─ 单图问答 / 想用现成 LLM 快速起步?
         → 投影 (LLaVA), 一个 MLP  [最简, 最省, 博0 默认起点]
```

逐层决策的逻辑:
1. **先问要不要生成** (最大分叉): 要生成 → 只能 early-fusion (统一 token 才能自回归画图)。
2. **再问视觉规模** (成本分叉): 多图/视频/高分辨率 → cross-attn (视觉不占序列)。
3. **默认**: 单图理解 → 投影 (最省最简)。

> 这张决策树是本专题的核心交付。它呼应你 9.4 的实验设计: 架构选择是有约束的工程决定, 不是「哪个最强」。**先问清你的 VLM 要干什么 (读图? 多图? 生成?), 再选路线。**

---

## 4. 一个常见的混合现实

实践中边界没这么清:
- 很多 VLM 用**投影 + resampler** (LLaVA 的简单 + Flamingo 的压缩): 投影后再压缩视觉 token, 缓解上下文压力。
- early-fusion 模型也可能局部用 cross-attn。
- 三路线是**思想坐标**, 不是互斥教条。理解每条路线**优化什么**, 你才能在真实约束下组合。

---

## 5. 本讲小结 (M10.2 收口) + 通往 M10.3

- **early-fusion** (Chameleon/Fuyu): 视觉文本同质 token, 一个统一 transformer 一锅煮; 解锁**理解+生成一体** (自回归画图), 但通常从头训, 贵。
- **决策树**: ① 要生成?→early-fusion ② 多图/视频?→cross-attn ③ 默认单图→投影 (博0 起点)。
- 三路线是思想坐标, 实践常混合; 关键是理解每条**优化什么**。

> **下一专题 M10.3「VLM 训练配方」**: 选好了融合路线 (投影), 怎么真正训出一个 VLM? 两阶段配方 (对齐预训练 + 指令微调)、数据、冻结策略。M10.3 让你端到端训出一个能问答的 mini-VLM。

**动手**: 去 `N1-three-connectors.ipynb` 完成三连接器对比后, 用本讲决策树给三个场景 (单图问答 / 视频理解 / 文生图) 各选一条路线并写理由; 然后 N2 真搭一个投影路线的 mini-LLaVA。
