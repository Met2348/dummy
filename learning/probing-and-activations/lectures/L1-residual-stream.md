# L1 · residual stream 视角: 信息在残差流里读写

> 22-min lecture · 目标: 深入理解 residual stream 作为信息主干, 组件如何从它读写。Module 12.2 开篇。

---

## 0. 逆向工程的「数据总线」

M12.1-L3 说 residual stream 是信息主干。这一讲讲透它 —— 因为本模块所有工具 (probing/patching/SAE) 都在这条流上操作。理解 residual stream, 你才能读懂网络里信息怎么流。

---

## 1. residual stream 是什么

transformer 的每一层通过**残差连接**: `x = x + 组件(x)`。把所有层的 `x` 串起来, 就是一条贯穿网络的「信息流」:

```
   embed → resid → [attn 读, 算, +回] → resid → [mlp 读, 算, +回] → resid → ... → unembed
                      ↑ 每个组件从 resid 读, 把结果加回 resid
```

> 关键: **residual stream 是一条共享的「黑板」**。每个组件 (attention head / MLP) 从黑板上**读**信息、计算、把结果**写回**黑板 (加法)。信息在黑板上累积, 最后 unembed 读黑板出预测。你 12.1 的 `tiny_transformer` 缓存的 `resid_post_*` 就是这块黑板每层的状态。

---

## 2. 为什么「读写」视角这么有用

把组件看成「读写 residual」让逆向工程变可操作:
- **每个 head 读什么 (从哪些位置/方向)、写什么 (什么 feature)** → 这定义了它的功能。
- 信息流 = 一连串读写: feature A 被写入 → 下游组件读 A → 算出 B 写入 → ...
- 这就是 circuit (M12.5): 一串读写连成算法。

> 因为残差是**线性相加**, residual stream 有个漂亮性质: **不同组件的贡献可以线性分解** (resid = embed + 各 head 贡献 + 各 MLP 贡献)。这让「某个方向是谁写的」可追溯 —— 是 logit lens (L3) 和归因 (M12.5) 的基础。**线性结构 = 可分解 = 可逆向**, 你的线性代数直觉直接用上。

---

## 3. residual stream 的「带宽」与 superposition

residual stream 维度有限 (d_model), 但要承载的 feature 很多 → 又回到 superposition (M12.1-L2):
- residual 是个 d_model 维空间, 不同 feature 占不同**方向**。
- feature 数 > d_model → 方向必须**叠加** (superposition)。
- 所以读 feature 要读**方向** (投影), 不是读某一维 (那会混进多个 feature)。

> 这直接通向**线性探针** (L2): 既然 feature 是 residual 里的方向, 「读出某概念」= 找那个方向 + 看投影。探针就是学这个方向。**residual stream + 方向 = 探针的理论基础。**

---

## 4. 在你的玩具上看 residual stream

你 12.1-N2 已经看到: 玩具 transformer 的 residual (`resid_post_1`) 按「当前值」聚类 —— 说明「当前值」这个 feature 被编码在 residual 的某些方向上。本专题 N1 会用**线性探针**把这个方向学出来 (从 residual 读出当前值), N2 用 **logit lens** 看每层 residual 离最终预测多近。

> 你的 `tiny_transformer` 暴露了 `resid_pre / resid_post_0 / resid_post_1` —— 这是 residual stream 在三个时刻的快照。逆向工程就是看信息怎么从 `resid_pre` (刚 embed) 流到 `resid_post_1` (快出预测), 中间每个组件读写了什么。

---

## 5. 本讲小结 + 通往 L2

- **residual stream = 贯穿网络的共享黑板**; 组件从它读、算、写回 (加法)。
- 残差**线性相加** → 贡献可线性分解 → 可追溯「某方向是谁写的」(logit lens/归因基础)。
- feature 是 residual 里的**方向** (superposition 下叠加) → 读 feature = 读方向 (探针基础)。
- 你的玩具 `resid_post_*` 是黑板快照; 本专题用探针/logit lens 读它。

> **下一讲 L2「线性探针」**: 第一件读取工具 —— 训一个线性分类器, 看某概念是否被线性编码在 residual 里 (能读出=被编码)。

**动手**: 回看 12.1-N2 的「residual 按当前值聚类」图。L2 会把那个聚类**学成一个探针** (从 residual 读出当前值)。带着「feature=方向」的直觉进 L2。
