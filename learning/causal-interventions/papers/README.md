# 12.3 causal-interventions — 论文清单

> activation patching / 因果干预论文。读法接 L4: 单变量? 对照干净? 分布内? 充要双验证?

## 必读 (核心)
- **Locating and Editing Factual Associations (ROME)** (Meng et al., 2022) — causal tracing 定位事实存在哪 + 编辑。
- **Interpretability in the Wild (IOI)** (Wang et al., 2022) — activation patching 逆向一个完整 circuit。
- **Causal Scrubbing** (Redwood, 2022) — 严谨验证 circuit 假说的干预方法。
- **Attribution Patching** (Neel Nanda) — 用梯度近似 patching, 规模化 (M12.5)。

## 进阶 (方法严谨性)
- **Towards Automated Circuit Discovery (ACDC)** — 自动找 circuit (干预驱动)。
- **Mean/Resample ablation vs Zero ablation** — 干预的分布外陷阱 (L4)。
- **Path Patching** — 区分直接/间接因果路径 (L3)。

## 怎么读 (接 L4)
1. clean/corrupt 对照干净吗 (单变量)?
2. 干预分布内吗 (mean/resample 还是 zero)?
3. 充分+必要都验证了吗?
4. 多样本 + 度量稳健吗?

> 对照本专题: 玩具 patching 干净定位「最后位置因果携带答案」(恢复率1.0), ablation 证必要 = 充要。
