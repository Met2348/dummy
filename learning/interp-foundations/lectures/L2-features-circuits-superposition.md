# L2 · 核心概念: feature / circuit / superposition

> 24-min lecture · 目标: 掌握 mech interp 的三个基础概念, 尤其理解 superposition 为什么是核心难题。

---

## 0. 三个词撑起整门学科

mech interp 的几乎所有工作都围绕三个概念: **feature (特征)**、**circuit (电路)**、**superposition (叠加)**。理解这三个, 你就有了读这个领域的语言。

---

## 1. feature (特征): 网络的「概念」

> **feature = 网络内部表示的一个可理解的概念**, 通常对应激活空间里的一个**方向** (向量)。

例子:
- 「这个 token 是数字」可能是某个激活方向。
- 「金门大桥」(Anthropic 著名例子) 是 Claude 内部的一个方向。
- 「当前情绪是负面」可能是一个方向。

> 关键: feature 通常是**激活空间的方向**, 不是单个神经元 (见 superposition)。「读出某概念是否被编码」就是看激活在那个方向上的投影 —— 这就是**线性探针** (M12.2)。你 N1 会在玩具模型上看激活如何编码「当前值」这个 feature。

---

## 2. circuit (电路): 网络的「算法」

> **circuit = 一组协作的组件 (注意力头/神经元), 共同实现一个可理解的算法。**

例子:
- **induction head** (M12.5): 两个 attention head 协作, 实现「看到 AB...A 就预测 B」(in-context 复制)。这是 in-context learning 的机制基础。
- 「检测主谓一致」可能是若干 head + MLP 的电路。

> circuit 把 feature 连成计算: feature 是「名词」(概念), circuit 是「动词」(算法)。mech interp 的圣杯是**把一个模型行为完整逆向成一个 circuit** (像读懂一段代码)。M12.5 你会亲手找一个 induction circuit。

---

## 3. superposition (叠加): 核心难题

最反直觉也最关键的概念。朴素期待: 「一个神经元 = 一个概念」。**现实不是这样**:

> **superposition: 网络把「比神经元数量更多的 feature」压缩进有限的神经元里, 于是一个神经元会对多个不相关概念都激活 (多义神经元 / polysemantic)。**

为什么会这样?
- 真实世界的概念**远多于**网络的神经元数。
- 网络利用「大多数概念不同时出现」(稀疏), 把多个 feature **叠加**在同一组神经元上 (像超采样)。
- 代价: 单个神经元变「多义」—— 看一个神经元的激活, 分不清它在表示哪个概念。

> 你 N1 会在真 gpt2 上看到**多义神经元**: 同一个神经元对一堆不相关的 token/上下文都激活。这就是 superposition 的现象。**它是 mech interp 的头号障碍**: 神经元不是干净的概念单元, 所以不能「一个神经元一个神经元地读」。

---

## 4. superposition 为什么逼出了后面的工具

superposition 直接决定了本模块的工具链:
```
   神经元是多义的 (superposition)
     → 不能直接读神经元
     → 读「方向」而非神经元 (线性探针, M12.2)
     → 但探针是相关不是因果 → 需要干预 (patching, M12.3)
     → 想把叠加的 feature 拆开成单义的 → 稀疏自编码器 (SAE, M12.4)
     → 把单义 feature 连成算法 → circuit 分析 (M12.5)
```

> 整条工具链都是为了**对抗 superposition**: 网络把概念叠加压缩了, interp 要把它们解压、定位、连成电路。理解了 superposition 是难题, 你就懂了为什么需要 probing/patching/SAE/circuits 这一整套。**SAE (M12.4) 尤其是直接冲着「解叠加」去的。**

---

## 5. 本讲小结 + 通往 L3

- **feature**: 网络表示的可理解概念, 通常是激活空间的**方向** (→ 线性探针 M12.2)。
- **circuit**: 协作组件实现的可理解算法 (如 induction head, → M12.5)。
- **superposition**: 网络把多于神经元数的 feature 叠加压缩 → **多义神经元** (头号难题)。
- superposition 逼出整条工具链: 读方向 (probe) → 干预 (patch) → 解叠加 (SAE) → 连电路。

> **下一讲 L3「逆向工程纲领」**: 把这些概念组织成一个研究纲领 —— 怎么系统地「把神经网络当程序读」, 以及这个纲领的步骤。

**动手**: 在 `N1` 看到多义神经元后, 想一个问题: 如果神经元是多义的, 你怎么才能「读出」一个干净的概念? (答案: 读方向/probe M12.2, 或解叠加/SAE M12.4)。带着它进 L3。
