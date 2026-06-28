# L4 · 归因 patching: 规模化的因果归因

> 20-min lecture · 目标: 理解归因 patching 怎么用梯度一次性近似所有组件的因果贡献, 规模化 circuit 发现。收口 12.5。

---

## 0. 逐个 patching 不可扩展

M12.3 的 activation patching 要**逐个组件**做 (patch 一个、跑一次、测一次)。gpt2 有 144 个 head + 12 个 MLP; 大模型更多。逐个 patching 算力爆炸。**归因 patching (attribution patching)** 用一个聪明的近似, **一次反向传播**就估出所有组件的贡献。

---

## 1. 核心思想: 用梯度线性近似 patching

```
   patching 的效应 ≈ (clean 激活 − corrupt 激活) · (该激活处的梯度)
                     └─ 激活差 ──┘   └─ 度量对激活的梯度 ─┘
```

直觉: patching 把某激活从 corrupt 值改到 clean 值, 效应 ≈ 「改变量」× 「度量对它的敏感度 (梯度)」(一阶泰勒展开)。所以:
- 跑一次 corrupt (前向 + 反向), 拿到每个激活的**梯度**。
- 拿到 clean 激活 (一次前向)。
- 每个组件的归因 = 激活差 · 梯度 —— **一次反传估出全部组件**, 不用逐个 patch。

> 这把 O(组件数) 次 patching 降到 O(1) 次反传。**梯度让因果归因可扩展。** 你 N2 会用一个简化版 (逐头消融 = 精确但慢的归因) 给出每个 head 的贡献热图; 归因 patching 是它的梯度加速版 (用于大规模)。

---

## 2. 精确 vs 近似 (权衡)

| | 逐个 activation patching | 归因 patching |
|---|---|---|
| 精度 | 精确 (真 patch) | 近似 (一阶泰勒) |
| 成本 | O(组件数) 次前向 | O(1) 次前向+反向 |
| 适用 | 少量关键组件精查 | 全模型快速扫 |

> 实践: **归因 patching 先快速扫全模型** (找候选重要组件), 再用**精确 patching 复核**候选 (验证近似)。两者配合 = 规模化 + 严谨。归因近似在激活差大时可能不准 (泰勒一阶), 所以要复核。**别只信归因, 要精确 patching 兜底。**

---

## 3. 你 N2 做的: 逐头贡献热图

你 N2 对 gpt2 做**逐头消融**, 测每个 head 对 induction 的贡献 (消融它 induction loss 涨多少), 得一张 (层×头) 贡献热图:
- 亮的头 = 对 induction 重要 (消融损害大)。
- 你会看到**多个**头亮 (gpt2 的冗余 induction heads, L3) —— 不是单个头, 是一组。
- 这张图就是「induction circuit 的组件分布」(归因的可视化)。

> 注意 (接 L3 冗余): 因为冗余, 单头消融损害可能小, 但**一组头**的总贡献大。归因热图让你看到「整个 circuit 的分布」, 而非被单点消融误导。这是归因相对单点 patching 的价值: **看全貌**。

---

## 4. 收口: 从特征到电路 (12.5 全景)

```
   M12.5:
   L1 QK/OV: 一个 head = 看哪 (QK) + 搬什么 (OV)
   L2 induction head: 两 head 协作 → ICL 机制 (真 gpt2 找到)
   L3 circuit 分析: 干预定位组件+边, 连成算法 (冗余/完备性难点)
   L4 归因 patching: 梯度一次性估全部组件贡献 (规模化)
   ──────────────────────────────────────
   = 把单义特征 (M12.4) 连成可读、可验证的算法 (circuit)
```

> 这就是 mech interp 的「最终产物」: 不只是「有哪些特征」, 而是「它们怎么协作成算法」。你现在有了从 feature (M12.4) 到 circuit (M12.5) 的完整工具链。**12.7 capstone 会让你对一个行为做一次完整的 circuit 逆向。**

---

## 5. 本讲小结 (12.5 收口) + 通往 12.6

- 逐个 patching 不可扩展 → **归因 patching**: 激活差 · 梯度, **一次反传**估全部组件贡献 (一阶近似)。
- 精度 vs 成本: 归因快但近似 → 先归因扫全模型, 再精确 patching 复核候选。
- 你 N2 的逐头贡献热图 = induction circuit 的组件分布 (看到冗余的整组头)。
- 12.5 全景: QK/OV → induction head → circuit 分析 → 归因; 特征连成算法。

> **M12.5 收口**: 一个 head=QK+OV; induction head=ICL 机制 (真 gpt2); circuit=组件+信息流 (干预定位); 归因 patching 规模化。
> **下一专题 M12.6「cot-faithfulness-oversight」**: interp 用到推理模型 —— CoT 是真的「内心独白」吗? 忠实性/监控/欺骗检测 (对齐安全前沿, 接你 reasoning-r1)。下一专题 `cot-faithfulness-oversight`。

**动手**: 去 `N2`, 做 gpt2 逐头贡献热图, 看 induction 是**一组**头协作 (冗余), 不是单个。这解释了为什么 N1 消融单头损害小。
