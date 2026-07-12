# statistics-deep-dive 设计文档

> **承接**: for_real_dummy/ 下"独立技能系列"第 3 条(前两条是 rhcsa-bash-deep-dive、dsa-deep-dive)
> **触发动因**: 用户决定的"下一步系统学习方向 target frontier lab"两大专题之一,顺序在前(系统设计随后作为独立子项目另走一轮 brainstorm→spec→plan)
> **规模基准**: 参照 dsa-deep-dive(140点/commit 20a903d)、huggingface-deep-dive(101点),本专题预计约 116 点,是仓库里数学推导密度最高的一条

---

## 1. 专题定位

### 1.1 为什么是"独立技能系列"而不是"专题精读系列"

for_real_dummy/README.md 里现有三类系列边界清晰:"深挖系列"(numpy/torch/huggingface等,补读懂 `learning/` 代码需要的通用框架技能)、"专题精读系列"(long-context/kernel-gpu/alignment-algorithms/peft,**直接对应 `learning/` 下已有专题模块的同一份源码**,换成面试深度讲解)、"独立技能系列"(rhcsa-bash、dsa,和 `learning/` 无源码对应关系,是本仓库原创内容)。

已核实(Grep 全库 + 列出 `learning/` 48 个模块清单):`learning/` 下没有任何统计学/概率推断专题模块,只有两处误命中(dsa-deep-dive 的 DP 期望题、numpy 线性代数),不构成真实的"同一份源码"关系。因此 statistics-deep-dive 归类为**独立技能系列**,和 dsa-deep-dive 同构:不依赖 `learning/` 源码,知识点是原创内容,"AI研究/工程场景"步骤能真实关联时引用 `learning/`、其余系列的已验证内容(尤其 `llm-judge-arena`、`eval-foundations`、`dpo-family`/alignment-algorithms-deep-dive 里已经出现过的评测/优化场景),没有真实关联时如实使用通用工程场景,不强行编造关联。

### 1.2 与用户数学背景的关系

`for_real_dummy/README.md` 用户背景表:"概率论 ✅、随机过程 ❌(未学)"。据此:
- 板块 I"概率论回顾与描述统计"定位为**搭桥**(bridge),快速过一遍已学内容建立统一记号,不是重新教一遍概率论。
- 板块 V"时间序列基础"用户在澄清问题里被明确提醒"随机过程未学、这块会吃力",**用户仍选择包含**——设计上必须现场建立最小必要直觉(平稳性/自相关只用初等定义和数值模拟讲清楚,不引用鞅/马尔可夫链的形式化理论),不能假设随机过程先修知识,这是本板块的硬约束,写作时要反复对照检查。

### 1.3 范围决策记录(用户 AskUserQuestion 原始选择)

范围问题用户全选了四个可选板块(实验设计与因果推断、贝叶斯方法、AI/ML场景专属统计、时间序列基础)+ 默认包含的经典推断统计基础,做法问题用户选择"从头融入进阶深度"而不是"先出基础版后补"或"先建 learning/ 源码模块"。这两个选择是本设计文档所有后续决策的前提,不再重新论证。

---

## 2. 范围与规模(已在对话中向用户呈现并获批准,不再变更)

5 大板块 + 1 篇收尾 capstone,20 个分类文件,约 116 个知识点:

| 板块 | # | 分类文件 | 覆盖内容 | 知识点数(约) |
|------|---|---------|---------|------------|
| **I 经典推断统计基础** | 01 | 概率论回顾与描述统计 | 常见分布/矩/大数定律/中心极限定理/偏度峰度 | 6 |
| | 02 | 点估计理论 | MLE/MOM/无偏性一致性有效性/充分统计量/Cramér-Rao下界 | 7 |
| | 03 | 区间估计与假设检验框架 | 置信区间构造/Neyman-Pearson引理/Type I&II错误/检验力/p值真实含义与常见误用 | 7 |
| | 04 | 经典检验方法 | t检验族/卡方/ANOVA/Mann-Whitney/KS检验/置换检验/自助法bootstrap | 8 |
| | 05 | 多重检验与回归推断 | Bonferroni/FDR/BH流程/OLS假设与系数置信区间/残差诊断/逻辑回归推断 | 7 |
| **II 实验设计与因果推断** | 06 | A/B测试设计与功效分析 | 样本量计算/MDE/功效曲线/窥探问题/多重比较陷阱 | 6 |
| | 07 | 现代实验方法 | 序贯检验/always-valid p值/CUPED方差削减/新奇效应与学习效应 | 5 |
| | 08 | 因果推断基础 | potential outcomes框架/RCT为什么是金标准/混淆变量/DAG初步 | 6 |
| | 09 | 观察性因果推断方法 | DID/IV/倾向得分匹配PSM/断点回归RDD | 6 |
| | 10 | 真实陷阱案例集 | Simpson悖论/选择偏差/幸存者偏差/SUTVA违反/真实A-B测试事故复盘 | 5 |
| **III 贝叶斯方法** | 11 | 贝叶斯推断基础 | 先验似然后验/共轭先验族/频率派vs贝叶斯派对比 | 6 |
| | 12 | MCMC基础 | Metropolis-Hastings/Gibbs采样/收敛诊断/为什么需要采样 | 5 |
| | 13 | 贝叶斯应用 | 贝叶斯A/B测试/可信区间vs置信区间的真实语义差异/Bayes factor | 5 |
| **IV AI/ML场景专属统计** | 14 | 模型评测统计 | paired bootstrap比较模型分数/McNemar检验/评测集方差与置信区间 | 6 |
| | 15 | 排位系统 | Bradley-Terry/Elo(呼应llm-judge-arena)/TrueSkill简介 | 5 |
| | 16 | Scaling law与外推 | 幂律拟合/对数-对数回归/拟合不确定性/外推风险 | 5 |
| | 17 | 分布漂移与监控 | KL散度/KS检验做漂移检测/PSI/协变量偏移vs概念偏移 | 5 |
| | 18 | 标注一致性与分析方法论 | Cohen's kappa/Fleiss' kappa/呼应"经得起追问的具体数字"方法论 | 5 |
| **V 时间序列基础** | 19 | 时间序列基础 | 现场建立最小必要直觉:平稳性/ACF-PACF/白噪声/随机游走 | 6 |
| | 20 | 简单预测方法 | 移动平均/指数平滑/简单AR-MA概念,不深入完整ARIMA数学 | 5 |
| **收尾** | 21 | 模拟终面capstone | 跨板块场景叙事,非知识点列表,不计入 116 | — |

数字标"约",允许 ±10~15% 浮动(某知识点自然拆分/合并),沿用 huggingface(101≠整百)、dsa-deep-dive(140≠150)的既有纪律,不强行凑数。

---

## 3. 知识点模板:八步结构(本专题相对既有七步模板的核心变化)

既有七步(torch/huggingface/peft/dsa 系列通用):签名/是什么 → 一句话 → 底层机制/为什么这样设计 → AI研究/工程场景 → 可运行例子 → 面试怎么问+追问链 → 常见坑。

本专题扩展为**八步**,新增独立的"数学推导"步骤,不再和"底层机制"混在一起——这是响应用户"有数学"这一明确要求的具体落地方式(之前系列的"底层机制"步骤经常把数学推导和设计动机糅在一段话里带过,这次要求两者分开各自讲透):

1. **定义与记号**(对应既有"签名/是什么")—— 统计概念不总有函数签名,这一步写成形式化定义:符号、假设条件、适用前提。
2. **一句话** —— 直觉版本,不带公式。
3. **数学推导**(新增)—— 真正推公式:从定义出发一步步推导(如 MLE 的对数似然求导置零、CLT 的直觉证明思路、贝叶斯定理的共轭更新推导),用纯文本/unicode 数学符号(μ σ θ Σ √ ≈ ∂ ∑ ∫ 等),**不使用 LaTeX `$...$` 语法**——沿用 peft-deep-dive/alignment-algorithms-deep-dive 已验证的写法(如 `h=base(x)+α/r·BAx`),保证在任何 markdown 渲染器(GitHub/VSCode/纯文本)下都能正确显示,不依赖 MathJax。
4. **底层机制/为什么这样设计** —— 从数学推导"跳出来"讲设计取舍:为什么选这个假设检验而不是另一个、假设不满足时会怎么坏、和相邻方法的取舍关系。
5. **AI研究/工程场景** —— 真实关联优先引用仓库已验证内容(llm-judge-arena/eval-foundations/dpo-family 等),没有真实关联如实用通用工程场景,不编造。
6. **可运行例子** —— 待 §5 详细展开(不只是调库演示,要求数学结论本身被数值验证)。
7. **面试怎么问+追问链** —— 待 §4 详细展开(五轴方法论落地)。
8. **常见坑**。

---

## 4. 追问链方法论:五轴落地到统计学科

dsa-deep-dive 20 类调研出的 5 条轴线 + 1 种新题型,本专题的具体映射(**每个知识点按内容自然挑 1~2 条最贴切的轴线走 2~3 层深,不强行凑满 5 轴**——这是吸取 peft-deep-dive K-Adapter/MAD-X"如实标注,不强行解释成有意义设计"同一条纪律,避免形式主义):

| 轴线 | 统计学科映射 | 主要承载文件 |
|------|------------|------------|
| 规模递增轴 | 小样本精确检验→大样本渐进正态→N→∞极限行为;单次A/B→A/B/n→连续monitoring下多重比较爆炸 | 03、04、06、07 |
| 工程约束递增轴 | 离线批量算置信区间→在线流式更新→分布式多机汇总统计量的数值稳定性(呼应Welford算法) | 05、14、17 |
| 方案批判迭代轴 | 面试官给一个实验结果"显著,能上线吗"→候选人判断有误(功效不足/多重比较/新奇效应)→面试官指出具体缺陷→候选人换检验方法,重复 | 06、07、10 |
| 决策依据追问轴 | "为什么用t检验不用Mann-Whitney""为什么这里选贝叶斯不用频率派""为什么选DID不用PSM" | 04、09、11、13 |
| 真实性验证轴 | "简历写了'设计并分析了A/B测试'——功效分析具体怎么算的样本量,遇到什么统计陷阱" | 06、18、21 |
| **诊断真实数据/日志(新题型)** | 给一段真实实验数据/日志摘录(有几个可疑特征:分布不对称、样本量不均衡、时间戳聚集),要求诊断而非直接套公式 | 10、14、21 重点使用 |

capstone(21类)作为收尾,是唯一一篇要求**同时**用到 3 条以上轴线的文件——仿 dsa-deep-dive 19/20 类,场景是"候选人负责一个新训练方法要不要上线的判断,面试官从显著性解读一路追问到实验设计缺陷、多重比较、因果有效性(随机化是否干净)、贝叶斯视角的不确定性表达、最后给一段真实日志要求诊断异常模式",串联板块 I/II/III/IV。

---

## 5. 验证纪律与环境声明

- **环境**(已用 `.venv/Scripts/python.exe` 实测确认):numpy 2.4.6、scipy 1.17.1 均可用。scipy.stats 用于**交叉验证**(如验证自己手写的 t 检验统计量和 `scipy.stats.ttest_ind` 数值一致),但核心机制(MLE 优化、bootstrap 重采样、Metropolis-Hastings/Gibbs 采样)**手写实现**——沿用 peft-deep-dive"手写实现+和真实库交叉验证参数量/结果一致"的既有认识论,不用 PyMC/emcee 等专门 MCMC 库(避免新增重依赖,15 行 numpy 就能写清楚 Metropolis-Hastings 机制本身)。
- **数学结论必须被数值验证,不能只摆公式**——这是本专题验证纪律相对既有系列的实质加码:凡是"数学推导"步骤给出了一个可检验的结论(绝大多数知识点都属于这类),"可运行例子"步骤必须现场用蒙特卡洛模拟/数值实验验证该结论(如"验证 MLE 渐进正态性"要真的重复抽样几千次画出估计量分布并断言接近正态、"验证贝叶斯共轭更新公式"要真的用数值积分和解析公式对比);极少数纯定义性知识点(没有可数值验证的结论)不强行凑一个人造验证,但仍要满足既有系列"可运行例子"步骤的底线要求——真实构造数据、跑出具体数字,不能只放一段没有 assert 的 API 调用演示。这条纪律同源于 `00-roadmap.md`"复杂度不是断言出来的,是真的测出来的"和 huggingface"显存测量必须交叉核对"。
- **随机性代码固定种子**(`np.random.seed`/`random.seed`),涉及统计意义上的容差断言(不是精确相等)时参照 dsa-deep-dive 蓄水池抽样(2万次试验)、一致性哈希负载测量的既有容差设计经验。
- **计时类断言**(如果出现,比如比较频率派/贝叶斯计算耗时)沿用 `best_of(fn, trials=N)` 纪律,避免 dsa-deep-dive 01 类已经踩过的噪声坑。
- **验证脚本**:直接复用 `dsa-deep-dive/_verify_md.py`(regex 提取 ` ```python ` 代码块→独立 subprocess 执行),原样拷贝到 `statistics-deep-dive/_verify_md.py`,不重新设计。

---

## 6. 文件组织

```
for_real_dummy/statistics-deep-dive/
├── 00-roadmap.md                          ← 总规划+进度表+差异化声明+环境声明+八步模板+五轴映射表(本设计文档的浓缩版)
├── 01-probability-recap-and-descriptive.md
├── 02-point-estimation.md
├── 03-interval-estimation-and-testing-framework.md
├── 04-classical-tests.md
├── 05-multiple-testing-and-regression-inference.md
├── 06-ab-test-design-and-power.md
├── 07-modern-experimentation.md
├── 08-causal-inference-foundations.md
├── 09-observational-causal-methods.md
├── 10-real-world-traps.md
├── 11-bayesian-inference-foundations.md
├── 12-mcmc-foundations.md
├── 13-bayesian-applications.md
├── 14-model-evaluation-statistics.md
├── 15-ranking-systems.md
├── 16-scaling-laws-and-extrapolation.md
├── 17-distribution-shift-and-monitoring.md
├── 18-annotator-agreement-and-methodology.md
├── 19-time-series-foundations.md
├── 20-simple-forecasting.md
├── 21-mock-interview-capstone.md
└── _verify_md.py
```

## 7. README.md 集成

完成后更新 `for_real_dummy/README.md`:"独立技能系列"表格新增一行(和 rhcsa-bash-deep-dive、dsa-deep-dive 并列),目录结构树新增对应条目,顶部导航如有需要同步。不新建 `learning/` 内容(§1.1 已确认无源码依赖)。

---

## 8. 实施方案(供 writing-plans 展开)

按板块分批次,每批验证通过后单独提交(不是最后一次性提交)——原因:本专题体量与深度均超过既往任何一条系列,分批提交能在单批出问题时不影响已完成批次,也让用户在不需要主动确认的前提下仍能通过 git log 看到真实进度:

1. **Phase 0** 脚手架:创建目录、拷贝 `_verify_md.py`、写 `00-roadmap.md` 初版(全部⬜待撰写)
2. **Phase 1** 板块 I(01~05,经典推断统计基础,约35点)→ 验证 → 更新roadmap → 提交
3. **Phase 2** 板块 II(06~10,实验设计与因果推断,约28点)→ 验证 → 更新roadmap → 提交
4. **Phase 3** 板块 III(11~13,贝叶斯方法,约16点)→ 验证 → 更新roadmap → 提交
5. **Phase 4** 板块 IV(14~18,AI/ML场景专属统计,约26点)→ 验证 → 更新roadmap → 提交
6. **Phase 5** 板块 V(19~20,时间序列基础,约11点)→ 验证 → 更新roadmap → 提交
7. **Phase 6** 21类capstone → 验证 → 更新roadmap grand total → 提交
8. **Phase 7** 全库自查回归(七步/八步结构标记计数、assert覆盖率、进度表一致性)+ 更新 `README.md` + 更新 memory → 最终提交

每个 Phase 内部按"写→用 `_verify_md.py` 验证→不稳定的断言(计时/随机性)按 §5 纪律加固→更新对应 roadmap 行状态"的顺序推进,不把验证推迟到全部板块写完才做一次性检查(dsa-deep-dive 01 类的教训:间歇性 flaky 问题只有靠早发现早修才不会在后期大规模返工)。

---

## 设计签字

- **设计日期**: 2026-07-13
- **设计者**: Claude(brainstorming 流程,用户全程参与范围/做法两轮关键决策)
- **审阅者**: 用户已批准"整体结构与规模"(§2)及"实现方式"(§3~§5 的方向,即"从头融入进阶深度"),并明确指示后续细节由 Claude 自行决定、无需逐节确认("批准 正式开始 一口气分批次做完 中途无需我的反复确认")——§3~§8 的具体设计内容是本轮在此授权下由 Claude 直接定稿,未逐条呈现给用户复核,自查环节(见 spec 自查记录)已替代逐条确认。
