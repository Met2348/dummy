# L2 · activation patching: 因果定位的核心工具

> 24-min lecture · 目标: 掌握 activation patching —— mech interp 最核心的因果定位工具。

---

## 0. mech interp 的「主力武器」

activation patching (也叫 causal tracing / interchange intervention) 是 mech interp 用得最多的工具。一句话:

> **把「干净运行」的某个激活, 贴到「污染运行」里, 看污染运行的行为是否被恢复。恢复了 → 那个激活因果地携带了行为所需的信息。**

---

## 1. clean / corrupt 对照设计

patching 需要一对精心设计的输入:

```
   clean 输入:   产生行为 A (如预测 "Paris")
   corrupt 输入: 改一处, 产生行为 B (如预测 "Rome")
   要求: 两者尽量像, 只在「你关心的信息」上不同
```

> 设计 clean/corrupt 是 patching 的灵魂 (接 M9.4 实验设计: 单变量对照)。它们越像、只差你关心的那一点, patching 的定位就越干净。你 N1 的玩具: clean/corrupt 是同一 increment 序列, 只有**最后一个 token 不同** (→ 答案不同)。这样 patching 能精确定位「答案信息在哪」。

---

## 2. patching 的步骤

```
   ① 跑 clean, 缓存所有激活 (M12.2 的 run_with_cache)
   ② 跑 corrupt, 但在某 (层 L, 位置 p) 把 corrupt 的激活
      替换成 clean 缓存的激活 (patch!)
   ③ 看输出: 行为多大程度回到 clean (A)?
      完全恢复 → (L, p) 因果携带了 clean 行为所需信息
      没恢复   → (L, p) 不携带 (信息在别处)
   ④ 对每个 (L, p) 重复 → 因果定位热图
```

> 度量「恢复」常用 **logit difference**: clean答案与corrupt答案的 logit 差。corrupt 运行此差为负 (偏 B), patch 对了会把它拉回正 (偏 A)。恢复率 = (patch后 − corrupt) / (clean − corrupt), 0=没恢复, 1=完全恢复。你 N1 的热图就是这个恢复率。

---

## 3. 读 patching 热图: 信息流的地图

patching 热图 (层 × 位置) 是一张**因果信息流地图**:
- **哪个位置 patch 能恢复** → 信息在哪个 token 位置。
- **哪一层开始能恢复** → 信息在哪层就绪 (更早层 patch 无效说明还没算出来)。
- 热点的移动 → 信息怎么从一个位置/层流到另一个 (circuit 的雏形, M12.5)。

> 你 N1 的玩具会显示: 只有**最后位置**patch 能恢复 (恢复率≈1), 其它位置≈0 —— 因为 increment 任务里只有最后 token 决定答案。**热图干净地指出了因果位置。** 真实 gpt2 上同样: patching "capital of France" 任务能定位「国家名信息在哪个位置/层被搬运到答案」(经典 causal tracing)。

---

## 4. patching 的变体 (知道名字)

- **denoising (clean→corrupt)**: 把 clean 激活贴进 corrupt, 看恢复 (你 N1 做的; 找「哪里足以恢复」)。
- **noising (corrupt→clean)**: 把 corrupt 激活贴进 clean, 看破坏 (找「哪里必要」)。
- **patch 不同对象**: residual / 单个 attention head 的输出 / MLP 输出 —— 粒度越细, 定位越精 (M12.5 patch head)。
- **归因 patching** (M12.5): 用梯度近似 patching, 规模化到所有组件 (省去逐个 patch)。

> denoising 找「充分」(patch 这里就够恢复), noising 找「必要」(去掉这里就坏)。两者结合定位**最小充要因果集**。你 N2 的 ablation 就是 noising 的一种 (找必要组件)。

---

## 5. 本讲小结 + 通往 L3

- **activation patching = 把 clean 激活贴进 corrupt 运行, 看行为是否恢复** (恢复=因果携带信息)。
- 灵魂是 **clean/corrupt 对照设计**: 越像、只差你关心的那点, 定位越干净 (接 M9.4)。
- 度量: logit difference 恢复率 (0=没恢复, 1=完全恢复); 热图 = 因果信息流地图。
- 变体: denoising (找充分) / noising (找必要) / patch head / 归因 patching (M12.5)。

> **下一讲 L3「ablation / 因果路径」**: 另一种干预 —— 把激活置零 (ablation), 找「哪些组件是必要的」, 扫出最小因果子集。

**动手**: 完成 N1 (patching 热图) 后, 想: 如果两个位置 patch 都能恢复 (冗余路径), 你怎么判断哪个「必要」? (答案: ablation/noising, L3)。
