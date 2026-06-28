# L2 · induction heads: in-context learning 的机制

> 24-min lecture · 目标: 理解 induction head 这个经典 circuit, 它是 in-context learning 的机制基础。你在真 gpt2 上找到它。

---

## 0. mech interp 最著名的发现

induction head 是 mech interp 迄今最有名、最完整逆向的 circuit。它解释了一个深刻现象: **为什么 transformer 能 in-context learning (从上下文里现学)**。这一讲讲清它的机制, 你 N1 会在真 gpt2 上**亲手找到它**。

---

## 1. induction 行为: AB...A → B

induction head 实现一个简单但强大的规则:

```
   序列: ... A B ... A ?
   induction: ? 应该是 B (因为上次 A 后面跟的是 B)
```

例: 看到 "Harry Potter ... Harry" → 预测 "Potter" (上次 Harry 后是 Potter)。这就是**从上下文里学模式并复用** —— in-context learning 的最小核。

> 你 N1 喂一个「随机 token 重复两遍」的序列 (`[r1 r2 ... rk r1 r2 ... rk]`): 在第二遍, 一个 induction head 会在每个 token 处, 预测「上次它后面跟的那个」(= 正确的重复)。**随机 token 也能预测 → 不可能是记住的, 只能是 in-context 复制 → induction。**

---

## 2. 机制: 两个 head 协作 (一个 circuit)

induction 需要**两个 head 协作** (这是它作为 circuit 的精髓):

```
   ① previous-token head (前一个 token 头):
      在每个位置, 把「前一个 token」的信息写进当前 residual
      → 于是每个位置的 residual 标记了「我前面是谁」

   ② induction head:
      在第二遍的 A 处, 用 QK 找「之前哪个位置的'前一个token'是 A」
      → 找到第一遍 A 的下一个位置 (即 B 所在)
      → OV 把 B 复制到当前预测
```

> 这是一个**两步 circuit**: head ① 准备信息 (每个位置标记前驱), head ② 利用它做匹配+复制。**信息通过 residual stream 从 ① 流到 ②** (M12.2-L1 的黑板)。这是「把多个 head 连成算法」的范例 (L3 的 circuit 分析)。你 N1 先找 head ② (induction head, 分数最高的); 它依赖 head ① 在前面铺垫。

---

## 3. 怎么检测 induction head (你 N1 做的)

```
   喂重复序列 [BOS, r1..rk, r1..rk]
   对每个 (层, 头), 算 induction 分数:
     在第二遍位置 i (token = r_{i-k}), 看该头 attend 到「位置 (i-k)+1」(第一遍该 token 的下一个) 多少
     分数高 → 这个头在做 induction (attend 到'上次的下一个')
```

> 你 N1 会得到一张 (层×头) 的 induction 分数热图, 找出 gpt2 的 induction head (实测在中层, 如层5头5, 分数 ~0.9)。**这是真实大模型上的真实 circuit** —— 不是玩具, 是 gpt2 预训练涌现的。然后你**消融它**, 看预测重复的能力变差 (因果确认, L4/M12.3)。

---

## 4. 为什么 induction head 这么重要

- **in-context learning 的机制基础**: induction 是「从上下文学模式」的最简形式; 更复杂的 ICL 建在它之上。
- **涌现现象**: induction head 在训练中**突然出现** (伴随 ICL 能力跃升), 是「能力涌现」的一个可机制解释的例子。
- **mech interp 的胜利**: 它是「把一个模型行为完整逆向成 circuit」的标杆 —— 证明 mech interp 的纲领 (M12.1) 能成功。
- **跨模型普遍**: 几乎所有 transformer 都长出 induction head (普适机制)。

> induction head 是 mech interp「能成功」的存在性证明: 一个真实、重要的行为 (ICL) 被完整、因果地逆向成一个可理解的 circuit。这给整个领域信心。你亲手在 gpt2 上复现它, 是这个模块的高光。

---

## 5. 本讲小结 + 通往 L3

- **induction head**: 实现「AB...A→B」(从上下文复制), in-context learning 的机制基础。
- 机制 = **两个 head 协作 circuit**: previous-token head (标记前驱) + induction head (匹配+复制), 信息经 residual 串联。
- 检测: 喂重复序列, 算「attend 到上次下一个」的分数 (你 N1 在 gpt2 找到它)。
- 重要性: ICL 机制 + 涌现的可机制解释 + mech interp 的标杆成果。

> **下一讲 L3「电路分析方法」**: 怎么把多个组件 (head/MLP) 系统地连成一个 circuit? 从单个 head 到完整电路的方法论。

**动手**: 去 `N1`, 在真 gpt2 上找 induction head (induction 分数热图), 看它的注意力图确实 attend 到「上次出现的下一个 token」。这是真实大模型的真实 circuit。
