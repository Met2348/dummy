# L1 · attention head 在算什么: QK/OV 电路

> 22-min lecture · 目标: 理解一个 attention head 的两个独立电路 (QK 决定看哪、OV 决定搬什么)。Module 12.5 开篇。

---

## 0. 从特征到算法

M12.4 提取了单义特征 (网络的「名词」)。这一讲开始把特征**连成算法 (circuit)**, 从最关键的组件 —— attention head 入手。理解单个 head 在算什么, 才能把多个 head 连成 circuit (L3)。

---

## 1. 一个 head = 两个独立的电路 (QK + OV)

mech interp 的关键洞察 (Anthropic 数学框架): 一个 attention head 的计算可以分解成**两个独立的电路**:

```
   QK 电路 (Query-Key): 决定「每个位置该看哪些位置」(注意力模式)
   OV 电路 (Output-Value): 决定「从看到的位置搬运什么信息回来」
```

- **QK 电路**: $q = x W_Q$, $k = x W_K$, 注意力 $\propto q \cdot k$。它只决定**注意力图** (谁看谁)。
- **OV 电路**: $v = x W_V$, 输出 $= \text{attn} \cdot v \cdot W_O$。它决定**搬什么** (从被看的位置读什么 feature、写什么回 residual)。

> 关键: **QK 和 OV 是独立的** —— 「看哪」和「搬什么」是两件分开的事。这让逆向工程一个 head 变成两个可分别理解的问题: 它的注意力模式是什么 (QK)? 它搬运什么信息 (OV)? 你的线性代数背景在这里直接用: QK/OV 都是低秩线性映射, 可分析。

---

## 2. QK 电路: 注意力模式

QK 电路决定一个 head 的「注意力风格」。常见模式:
- **前一个 token** (previous-token head): 每个位置看前一个 (做局部 n-gram)。
- **句首** (BOS head): 都看句首 (一种「无操作」/默认)。
- **重复检测**: 看之前出现过的相同 token (induction 的前置, L2)。
- **句法**: 看主语/动词等。

> 看一个 head 的注意力图 (attention pattern), 就能猜它的 QK 风格。但记住 M12.2-L4: **看注意力图只是相关** (假设), 要因果验证 (ablation) 才算数。你 N1 会看真 gpt2 head 的注意力图, 找出 induction 风格的头。

---

## 3. OV 电路: 搬运什么

QK 决定看哪, OV 决定从那搬什么。OV 电路是一个线性映射: 从被看位置的 residual 读某 feature, 写另一个 feature 回当前位置的 residual。

- 例: induction head 的 OV「把被看 token 复制到当前位置的预测」(搬运=复制)。
- OV 是 $W_V W_O$ 的低秩映射, 可以分析它读什么方向、写什么方向 (接 M12.4 特征方向)。

> QK + OV 合起来描述一个 head 的完整功能: **「在什么情况下 (QK) 搬运什么 (OV)」**。这是 head 级逆向工程的语言。把多个这样的 head 连起来 (一个的输出是另一个的输入) = circuit (L3)。

---

## 4. 为什么这个分解如此有用

QK/OV 分解让 attention 可逆向:
- **降维**: 不用看整个 head 的复杂计算, 分成「看哪 (QK)」+「搬什么 (OV)」两个低秩问题。
- **可组合**: head 之间通过 residual stream 串联 (一个写、另一个读), QK/OV 让这种串联可追踪 (M12.5-L3 circuit)。
- **数学清晰**: QK/OV 都是线性映射, 能用线性代数 (特征值/低秩) 分析 (你的 EE 优势)。

> 这是 mech interp 把 attention 从「黑箱」变「可读」的关键工具。**一个 head = QK (看哪) + OV (搬什么)**, 记住这个分解, 你能读任何 attention head 的功能。

---

## 5. 本讲小结 + 通往 L2

- 一个 attention head = **两个独立电路**: **QK** (决定看哪/注意力模式) + **OV** (决定搬什么)。
- QK 模式: 前一个token / 句首 / 重复检测 / 句法; 看注意力图猜 (相关, 需 ablation 验证)。
- OV: 低秩线性映射, 从被看位置读 feature、写 feature 回当前 (如复制)。
- 分解优势: 降维 + 可组合 + 数学清晰 (你的线性代数优势)。

> **下一讲 L2「induction heads」**: 最经典的 circuit —— 两个 head 协作实现「看到 AB...A 预测 B」(in-context learning 的机制基础)。你会在真 gpt2 上找到它。

**动手**: 去 `N1`, 在真 gpt2 上喂「重复序列」, 逐头算 induction 分数, 找出 gpt2 的 induction head (它的 QK 看「上次出现的下一个」)。
