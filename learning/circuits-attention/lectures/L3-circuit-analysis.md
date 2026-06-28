# L3 · 电路分析方法: 把组件连成算法

> 22-min lecture · 目标: 掌握把多个组件 (head/MLP) 系统连成一个 circuit 的方法论。

---

## 0. 从单个组件到完整电路

L1/L2 理解了单个 head。但一个行为通常是**多个组件协作** (induction 要两个 head)。这一讲讲怎么把它们系统地连成一个 **circuit** —— mech interp 的最终产物。

---

## 1. circuit = 组件 + 它们之间的信息流

```
   circuit = 一组组件 (head/MLP/特征) + 它们如何通过 residual stream 互相读写
   节点: 组件 (它算什么 — QK/OV/MLP feature)
   边:   信息流 (A 写入某 feature → B 读它)
```

> 逆向一个 circuit = 画出这张图: 哪些组件参与、信息怎么从一个流到另一个、最后怎么变成输出。induction circuit (L2) 的图: previous-token head → (经 residual) → induction head → 预测。**circuit 是网络内部的「数据流图」, 用人话标注每个节点和边。**

---

## 2. 找 circuit 的方法 (流程)

```
   ① 定义窄行为 (M12.1-L3): 如「induction」「IOI (间接宾语)」
   ② 定位关键组件: 逐组件 patching/ablation (M12.3), 找哪些重要
   ③ 找信息流的边: path patching (M12.3-L3), 区分「直接 vs 经某组件」
   ④ 理解每个节点: QK/OV (L1) — 每个 head 在算什么
   ⑤ 画出 circuit: 节点+边+人话; 验证它能预测/重建行为
   ⑥ 攻击: 找反例、查完备性 (是不是漏了组件) (M9.3)
```

> 核心是 ②③ 用**因果干预** (M12.3) 找节点和边 —— circuit 不是看注意力图猜出来的 (相关), 是干预验证出来的 (因果)。**circuit 分析 = 因果干预 (M12.3) + QK/OV 理解 (L1) + 信息流追踪。** 这是把前面所有工具串起来的综合技能。

---

## 3. 经典 circuit 案例 (知道它们)

- **induction circuit** (L2): previous-token head + induction head → ICL。
- **IOI circuit** (Indirect Object Identification): "John gave a drink to **Mary**" —— 十几个 head 协作识别间接宾语, 是最完整逆向的 circuit 之一 (Wang et al.)。
- **事实回忆** (ROME): MLP 层存储事实, attention 搬运到预测位置。

> 这些案例的共性: 从一个**窄行为**入手, 用干预定位组件, 连成可读的算法。它们证明了 mech interp 能对真实模型的真实行为给出**机制级**理解。IOI 尤其是「多组件复杂 circuit」的范例 (induction 只是两个 head, IOI 是一二十个)。

---

## 4. circuit 分析的难点 (诚实)

- **冗余**: 多个组件做同样的事 (gpt2 有多个 induction head!) → 消融一个效果小, 容易低估重要性。你 N1 会看到消融单个 induction head 损害不大 (冗余)。
- **组合爆炸**: 组件多 (gpt2: 144 个 head + MLP), 逐个/逐对 patching 不可扩展 → 需归因 patching (L4)。
- **完备性**: 找到一个 circuit 不代表它是唯一/全部路径。
- **跨层 MLP**: MLP 的作用比 attention 更难解读 (SAE M12.4 帮忙)。

> 这些是 circuit 分析的现实挑战。**冗余**尤其重要: 真实模型有备份路径, 单点消融会低估 —— 要消融「整组」或用归因 (L4) 才看得全。诚实地说, 完整逆向一个大模型的所有 circuit 仍远未做到 (M12.1-L4 的可扩展性 open 问题)。

---

## 5. 本讲小结 + 通往 L4

- **circuit = 组件 (节点) + residual 信息流 (边)**, 用人话标注的网络数据流图。
- 找 circuit 流程: 定义窄行为 → 干预定位组件/边 (M12.3) → QK/OV 理解节点 (L1) → 画图验证 → 攻击。
- 经典: induction (2 head) / IOI (十几 head) / 事实回忆 (MLP+attn)。
- 难点: **冗余** (单点消融低估) / 组合爆炸 (需归因 L4) / 完备性 / MLP 难解。

> **下一讲 L4「归因 patching」**: 逐个 patching 不可扩展 (144 个 head)。归因 patching 用梯度**一次性近似**所有组件的因果贡献, 规模化 circuit 发现。

**动手**: 完成 N1 (找 induction head) 后想: gpt2 有多个 induction head (冗余), 消融一个损害小。你怎么衡量「整组 induction head」的贡献? (答案: 消融整组 / 归因 patching, L4 + N2)。
