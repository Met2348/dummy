# 统计深挖 —— 路线图与进度表

> 目标:约 116 个统计学知识点,由浅入深,深度对标 [dsa-deep-dive/](../dsa-deep-dive/00-roadmap.md)/[huggingface-deep-dive/](../huggingface-deep-dive/00-roadmap.md)(面试二三面深度,不是"这个公式怎么用")。
> 定位:仓库"独立技能系列"第 3 条(前两条是 [rhcsa-bash-deep-dive/](../rhcsa-bash-deep-dive/00-roadmap.md)、[dsa-deep-dive/](../dsa-deep-dive/00-roadmap.md))——和 `learning/` 没有源码对应关系(已核实全库无统计学专题模块),知识点是原创内容,不是给已有代码配讲解。

---

## 为什么是独立技能系列,不是专题精读系列

`learning/` 下 48 个模块(rl-foundations/lora-family/transformer-deep/moe-architecture/pretraining-recipe/rlhf-classic/dpo-family/eval-foundations/agent-\*/gpu-architecture/cluster-networking 等)全部核查过,没有任何统计学/概率推断专题模块——不构成"专题精读系列"要求的"直接对应同一份源码"关系。因此本系列和 dsa-deep-dive 同构:原创知识点,"AI研究/工程场景"步骤真实关联时引用 `llm-judge-arena`/`eval-foundations`/`dpo-family`/`alignment-algorithms-deep-dive`/`torch-deep-dive`/`huggingface-deep-dive` 已验证内容,没有真实关联时如实用通用工程场景,不编造关联。

## 和用户数学背景的关系

用户背景(见 [for_real_dummy/README.md](../README.md)):概率论 ✅ 已学,随机过程 ❌ 未学。据此:
- 01 类"概率论回顾"定位为**搭桥**,快速过一遍已学内容建立统一记号,不是重新教一遍。
- 19/20 类(时间序列)**现场建立最小必要直觉**——平稳性/自相关只用初等定义 + 数值模拟讲清楚,不引用鞅或马尔可夫链的形式化理论,不假设随机过程先修知识。

## 环境声明

运行环境:仓库根目录 `.venv`(Windows 原生),已实测确认 numpy 2.4.6、scipy 1.17.1 可用。**不新增任何第三方依赖**——已确认 `statsmodels` 未安装(`ModuleNotFoundError`),回归推断相关知识点全部手写正规方程 `(X'X)⁻¹X'y` + 解析标准误公式,简单线性场景用 `scipy.stats.linregress` 交叉验证。核心机制(MLE 优化、bootstrap、置换检验、Metropolis-Hastings、Gibbs 采样、BH 流程、2SLS、倾向得分匹配、Bradley-Terry MLE、Elo、KL 散度、PSI、Cohen's kappa、AR(1) 最小二乘等)全部**手写实现**,能和 `scipy.stats`/`scipy.optimize` 交叉验证数值一致的必须交叉验证,不依赖专门的统计/贝叶斯库(不装 PyMC/emcee/statsmodels)。

## 知识点结构模板(八步,相对既有七步新增独立"数学推导"步骤)

1. **定义与记号** 2. **一句话** 3. **数学推导**(真推公式,不是描述;用 unicode 数学符号 μ σ θ Σ √ ≈ ≤ ≥ ∂ ∑ ∫ 等写在正文里,**不用 LaTeX `$...$` 语法**,保证任何 markdown 渲染器下都能正确显示) 4. **底层机制/为什么这样设计** 5. **AI研究/工程场景** 6. **可运行例子**(assert 验证,数学推导给出的可检验结论必须被数值验证,不是只演示 API 调用) 7. **面试怎么问+追问链**(见下方五轴方法论) 8. **常见坑**

这次相对 dsa-deep-dive 的关键变化:七步模板"底层机制"步骤里原本混在一起的"数学推导"和"设计动机"两件事拆成独立的第 3/4 步分开讲透——响应"有数学"这一明确要求;"面试怎么问+追问链"从第一天起就用验证过的多轴深度方法论,不是像 dsa-deep-dive 20 类那样事后再补一篇独立追加文件。

## 五轴追问链方法论(dsa-deep-dive 20 类调研验证过的方法论,这次从第一天融入)

| 轴线 | 含义 | 统计学科例子 |
|------|------|------------|
| 规模递增轴 | 小样本→大样本→极限行为 | 精确检验→渐进正态→N→∞ |
| 工程约束递增轴 | 离线→在线→分布式 | 批量算CI→流式更新→多机汇总数值稳定性 |
| 方案批判迭代轴 | 面试官连续指出具体缺陷逼换方案 | "显著能上线吗"→功效不足→多重比较→换方法 |
| 决策依据追问轴 | 不纠错,只逼问选择依据 | "为什么用t检验不用Mann-Whitney" |
| 真实性验证轴 | 简历"做了A/B测试"被追问到具体数字 | "样本量怎么算的,遇到什么陷阱" |
| 诊断真实数据(新题型) | 给数据/日志摘录,要求诊断而非套公式 | 埋点延迟导致的伪显著 |

每个知识点挑 1~2 条最自然的轴线走 2~3 层深,不强行凑满 5 轴——凑不上就如实用规模递增轴或决策依据追问轴兜底,不编造牵强的追问。21 类模拟终面 capstone 是唯一要求同时用 3 条以上轴线的文件。

## 进度表

| # | 板块 | 分类 | 文件 | 知识点数(约) | 状态 |
|---|------|------|------|------------|------|
| 01 | I 经典推断统计基础 | 概率论回顾与描述统计 | [01-probability-recap-and-descriptive.md](01-probability-recap-and-descriptive.md) | 6 | ✅ 已完成 |
| 02 | I | 点估计理论 | [02-point-estimation.md](02-point-estimation.md) | 7 | ✅ 已完成 |
| 03 | I | 区间估计与假设检验框架 | [03-interval-estimation-and-testing-framework.md](03-interval-estimation-and-testing-framework.md) | 7 | ✅ 已完成 |
| 04 | I | 经典检验方法 | [04-classical-tests.md](04-classical-tests.md) | 8 | ✅ 已完成 |
| 05 | I | 多重检验与回归推断 | [05-multiple-testing-and-regression-inference.md](05-multiple-testing-and-regression-inference.md) | 7 | ✅ 已完成 |
| 06 | II 实验设计与因果推断 | A/B测试设计与功效分析 | [06-ab-test-design-and-power.md](06-ab-test-design-and-power.md) | 6 | ⬜ 待撰写 |
| 07 | II | 现代实验方法 | [07-modern-experimentation.md](07-modern-experimentation.md) | 5 | ⬜ 待撰写 |
| 08 | II | 因果推断基础 | [08-causal-inference-foundations.md](08-causal-inference-foundations.md) | 6 | ⬜ 待撰写 |
| 09 | II | 观察性因果推断方法 | [09-observational-causal-methods.md](09-observational-causal-methods.md) | 6 | ⬜ 待撰写 |
| 10 | II | 真实陷阱案例集 | [10-real-world-traps.md](10-real-world-traps.md) | 5 | ⬜ 待撰写 |
| 11 | III 贝叶斯方法 | 贝叶斯推断基础 | [11-bayesian-inference-foundations.md](11-bayesian-inference-foundations.md) | 6 | ⬜ 待撰写 |
| 12 | III | MCMC基础 | [12-mcmc-foundations.md](12-mcmc-foundations.md) | 5 | ⬜ 待撰写 |
| 13 | III | 贝叶斯应用 | [13-bayesian-applications.md](13-bayesian-applications.md) | 5 | ⬜ 待撰写 |
| 14 | IV AI/ML场景专属统计 | 模型评测统计 | [14-model-evaluation-statistics.md](14-model-evaluation-statistics.md) | 6 | ⬜ 待撰写 |
| 15 | IV | 排位系统 | [15-ranking-systems.md](15-ranking-systems.md) | 5 | ⬜ 待撰写 |
| 16 | IV | Scaling law与外推 | [16-scaling-laws-and-extrapolation.md](16-scaling-laws-and-extrapolation.md) | 5 | ⬜ 待撰写 |
| 17 | IV | 分布漂移与监控 | [17-distribution-shift-and-monitoring.md](17-distribution-shift-and-monitoring.md) | 5 | ⬜ 待撰写 |
| 18 | IV | 标注一致性与分析方法论 | [18-annotator-agreement-and-methodology.md](18-annotator-agreement-and-methodology.md) | 5 | ⬜ 待撰写 |
| 19 | V 时间序列基础 | 时间序列基础 | [19-time-series-foundations.md](19-time-series-foundations.md) | 6 | ⬜ 待撰写 |
| 20 | V | 简单预测方法 | [20-simple-forecasting.md](20-simple-forecasting.md) | 5 | ⬜ 待撰写 |
| 21 | 收尾 | 模拟终面capstone | [21-mock-interview-capstone.md](21-mock-interview-capstone.md) | — (不计入合计) | ⬜ 待撰写 |

**预计合计:约 116 个知识点,20 个分类文件 + 1 篇模拟终面 capstone。** 数字标"约",允许 ±10~15% 浮动(某知识点自然拆分/合并),不强行凑整数——沿用 huggingface(101≠整百)、dsa-deep-dive(140≠150)的既有纪律。

---

## 验证纪律

- 验证脚本 `_verify_md.py`(regex 提取 ` ```python ` 代码块,每块独立 subprocess 执行)直接拷贝自 `dsa-deep-dive/_verify_md.py`,不重新设计。
- 涉及随机抽样/MCMC/bootstrap 的代码固定随机种子,断言用统计意义上的容差,不用精确相等。
- 涉及计时对比的断言用 `best_of(fn, *args, trials=N)` 取多次采样最小值——dsa-deep-dive 01 类的教训:被测操作规模选得太小(个位数微秒级)时,单次 `perf_counter` 采样极易被系统调度噪声放大几十倍,只复测 1~3 次不足以确认稳定,需要 5~20 次不等的重复验证。
- 设计文档与实施计划见 [docs/superpowers/specs/2026-07-13-statistics-deep-dive-design.md](../../docs/superpowers/specs/2026-07-13-statistics-deep-dive-design.md)、[docs/superpowers/plans/2026-07-13-statistics-deep-dive.md](../../docs/superpowers/plans/2026-07-13-statistics-deep-dive.md)。

---

*创建:2026-07-13*
