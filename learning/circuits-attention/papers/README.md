# 12.5 circuits-attention — 论文清单

> 电路分析 / induction head 论文。读法接 L3: 干预定位组件? 冗余? 完备?

## 必读 (核心)
- **A Mathematical Framework for Transformer Circuits** (Anthropic, 2021) — QK/OV 电路分解 (L1)。
- **In-context Learning and Induction Heads** (Anthropic, 2022) — induction head = ICL 机制 (L2)。
- **Interpretability in the Wild (IOI)** (Wang et al., 2022) — 完整逆向一个多组件 circuit (L3)。
- **Attribution Patching** (Neel Nanda) — 梯度近似 patching, 规模化归因 (L4)。

## 进阶
- **Towards Automated Circuit Discovery (ACDC)** — 自动找 circuit。
- **Function Vectors / Task Vectors** — ICL 的向量级机制。
- **Edge Attribution Patching (EAP)** — 边级归因。

## 怎么读 (接 L3/L4)
1. 用干预 (patching/ablation) 定位组件吗 (还是只看注意力图)?
2. 处理冗余了吗 (整组 vs 单点)?
3. 完备性 (是不是漏组件)?
4. 归因近似有精确复核吗 (L4)?

> 对照本专题: 真 gpt2 找到 induction head (层5头5 分数~0.9) + 逐头消融见冗余 (一组头协作)。
