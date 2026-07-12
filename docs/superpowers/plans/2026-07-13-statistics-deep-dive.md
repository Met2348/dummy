# statistics-deep-dive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `for_real_dummy/statistics-deep-dive/` 下产出一条面试深度的统计学系列——5 大板块 20 个分类文件 + 1 篇收尾 capstone,约 116 个知识点,每点用八步模板讲透且从第一天起融入五轴追问链方法论。

**Architecture:** 每个分类文件是一个独立可验证的 markdown 文档(定义→一句话→数学推导→机制→AI场景→带 assert 的可运行例子→追问链→常见坑),用 `_verify_md.py` 独立提取并执行每个 python 代码块;`00-roadmap.md` 是进度表与规范总控;所有数学结论必须被代码数值验证,不是摆公式了事。

**Tech Stack:** Python(仓库根目录 `.venv`),numpy 2.4.6 + scipy 1.17.1(已实测确认可用),纯标准库 `random`/`statistics`/`math` 按需;**不新增任何第三方依赖**(已确认 `statsmodels` 未安装,回归推断部分手写正规方程,不安装它)。

## Global Constraints

以下规则适用于**每一个**知识点、**每一个**任务,不在每个 Task 里重复说明:

**G1 — 八步模板**(替代既有系列的七步,新增独立"数学推导"步骤):
1. 定义与记号 2. 一句话 3. **数学推导**(真推公式,不是描述) 4. 底层机制/为什么这样设计 5. AI研究/工程场景 6. 可运行例子(assert验证) 7. 面试怎么问+追问链 8. 常见坑

**G2 — 数学记号规则**:全部用 unicode 符号(μ σ θ Σ √ ≈ ≤ ≥ ∂ ∑ ∫ α β 等)写在 markdown 正文里,**禁止 LaTeX `$...$` 语法**(避免依赖 MathJax 渲染,任何 markdown 查看器都要能正确显示)。参照已验证写法如 `h=base(x)+α/r·BAx`。

**G3 — 五轴追问链方法论**(dsa-deep-dive 20 类验证过的方法论,这次从第一天融入,不是事后补):

| 轴线 | 含义 | 统计学科例子 |
|------|------|------------|
| 规模递增轴 | 小样本→大样本→极限行为 | 精确检验→渐进正态→N→∞ |
| 工程约束递增轴 | 离线→在线→分布式 | 批量算CI→流式更新→多机汇总数值稳定性 |
| 方案批判迭代轴 | 面试官连续指出具体缺陷逼换方案 | "显著能上线吗"→功效不足→多重比较→换方法 |
| 决策依据追问轴 | 不纠错,只逼问选择依据 | "为什么用t检验不用Mann-Whitney" |
| 真实性验证轴 | 简历"做了A/B测试"被追问到具体数字 | "样本量怎么算的,遇到什么陷阱" |
| 诊断真实数据(新题型) | 给数据/日志摘录,要求诊断而非套公式 | 埋点延迟导致的伪显著 |

每个知识点的"面试怎么问+追问链"步骤挑 1~2 条**最自然**的轴线走 2~3 层深,**不强行凑满 5 轴**——凑不上就如实用规模递增轴或决策依据追问轴兜底,不编造牵强的追问。

**G4 — 验证纪律**:
- 核心机制(MLE 优化、bootstrap、置换检验、Metropolis-Hastings、Gibbs 采样、BH流程、2SLS、倾向得分匹配、Bradley-Terry MLE、Elo、KL散度、PSI、Cohen's kappa、AR(1)最小二乘等)**全部手写实现**,不依赖专门统计库。
- 能和 `scipy.stats`/`scipy.optimize` 交叉验证数值一致的,**必须交叉验证**(如手写 t 检验统计量 vs `scipy.stats.ttest_ind`)。
- 数学推导步骤给出的每一个可检验结论,可运行例子步骤必须用蒙特卡洛模拟/数值实验**真的验证**,不是只演示 API 调用。极少数纯定义性知识点(无可数值验证的结论)不强行编验证,但仍要有真实数据+具体输出数字。
- 涉及随机抽样/MCMC/bootstrap 的代码固定随机种子(`np.random.seed(42)` 或函数级 `rng = np.random.default_rng(42)`),断言用统计意义上的容差(如"接近真值"用相对误差阈值),不用精确相等。
- 涉及计时对比的断言(如果出现)用 `best_of(fn, *args, trials=N)` 取多次采样最小值,参照 dsa-deep-dive 01 类已验证的纪律,避免噪声导致 flaky。

**G5 — 验证脚本**:`for_real_dummy/statistics-deep-dive/_verify_md.py` 是 `for_real_dummy/dsa-deep-dive/_verify_md.py` 的原样拷贝(regex 提取 ` ```python ` 代码块,每块独立 `subprocess.run([sys.executable, "-c", block])` 执行),不重新设计。

**G6 — Roadmap 状态约定**:`00-roadmap.md` 进度表状态列用 `⬜ 待撰写` → `🔧 撰写中` → `✅ 已完成`(验证通过才标已完成),和 peft-deep-dive/dsa-deep-dive 既有约定一致。

**G7 — 提交约定**:每个分类文件独立提交,commit message 格式 `docs(statistics-deep-dive): NN类 <中文分类名>,N个知识点`;`git add` 必须用显式文件路径,不用 `git add .`/`-A`。

**G8 — AI/工程场景真实性**:能真实关联 `learning/llm-judge-arena`、`learning/eval-foundations`、`learning/dpo-family`/`alignment-algorithms-deep-dive`、`torch-deep-dive`/`huggingface-deep-dive` 已验证内容的,先读一下对应文件确认关联真实存在再引用;没有真实关联的如实使用通用工程场景,不编造关联(参照 peft-deep-dive K-Adapter/MAD-X"如实标注,不强行解释成有意义设计"的既有纪律)。

---

## File Structure

```
for_real_dummy/statistics-deep-dive/
├── 00-roadmap.md
├── 01-probability-recap-and-descriptive.md      (6点)
├── 02-point-estimation.md                        (7点)
├── 03-interval-estimation-and-testing-framework.md (7点)
├── 04-classical-tests.md                          (8点)
├── 05-multiple-testing-and-regression-inference.md (7点)
├── 06-ab-test-design-and-power.md                 (6点)
├── 07-modern-experimentation.md                   (5点)
├── 08-causal-inference-foundations.md             (6点)
├── 09-observational-causal-methods.md             (6点)
├── 10-real-world-traps.md                         (5点)
├── 11-bayesian-inference-foundations.md           (6点)
├── 12-mcmc-foundations.md                         (5点)
├── 13-bayesian-applications.md                    (5点)
├── 14-model-evaluation-statistics.md              (6点)
├── 15-ranking-systems.md                          (5点)
├── 16-scaling-laws-and-extrapolation.md           (5点)
├── 17-distribution-shift-and-monitoring.md        (5点)
├── 18-annotator-agreement-and-methodology.md      (5点)
├── 19-time-series-foundations.md                  (6点)
├── 20-simple-forecasting.md                       (5点)
├── 21-mock-interview-capstone.md                  (capstone,不计点)
└── _verify_md.py
```

**标准文件级步骤模式**(每个分类文件 Task 都遵循这个 5 步节奏,下文不再逐条重复展开):
1. 撰写文件,覆盖该 Task 表格列出的全部知识点,套用 G1 八步模板
2. 运行 `.venv/Scripts/python.exe for_real_dummy/statistics-deep-dive/_verify_md.py for_real_dummy/statistics-deep-dive/NN-slug.md`,期望末行 `ALL N blocks passed`
3. 有 FAIL 就定位修复(常见原因:随机种子未固定、容差过严、假设条件写错的 assert),重跑到全部通过;随机性/计时类断言额外单独重复运行 10~15 次确认稳定(不满足于 1~3 次通过)
4. 更新 `00-roadmap.md` 对应行状态为 `✅`,写实际知识点数
5. `git add` 该文件 + `00-roadmap.md`,`git commit`(消息按 G7)

---

### Task 1: 脚手架 —— 目录、验证脚本、roadmap 初版

**Files:**
- Create: `for_real_dummy/statistics-deep-dive/_verify_md.py`(拷贝自 `for_real_dummy/dsa-deep-dive/_verify_md.py`,内容原样不改)
- Create: `for_real_dummy/statistics-deep-dive/00-roadmap.md`

**Interfaces:**
- Consumes: 无(起点任务)
- Produces: 后续全部 Task 依赖 `_verify_md.py` 做验证,依赖 `00-roadmap.md` 的 21 行进度表结构

- [ ] **Step 1: 拷贝验证脚本**

```bash
cp "for_real_dummy/dsa-deep-dive/_verify_md.py" "for_real_dummy/statistics-deep-dive/_verify_md.py"
```

- [ ] **Step 2: 撰写 `00-roadmap.md`**

内容必须包含(参照 `peft-deep-dive/00-roadmap.md`/`dsa-deep-dive/00-roadmap.md` 结构):
- 标题 + 定位(独立技能系列,不依赖 `learning/` 源码,理由见 spec §1.1)
- 差异化声明:和 `dsa-deep-dive`/`rhcsa-bash-deep-dive` 同属独立技能系列;和用户数学背景的关系(概率论已学、随机过程未学,板块V现场建立最小必要直觉)
- 环境声明:`.venv`,numpy 2.4.6 + scipy 1.17.1,不新增依赖,核心机制手写+和 scipy 交叉验证
- G1 八步模板全文
- G3 五轴追问链方法论表格全文
- 进度表(21 行,对应 File Structure 里的 21 个文件,初始全部 `⬜ 待撰写`,知识点数列填预估值)
- 合计行:"**预计合计:约 116 个知识点,20 个分类文件 + 1 篇模拟终面capstone。**"

- [ ] **Step 3: 提交**

```bash
git add "for_real_dummy/statistics-deep-dive/_verify_md.py" "for_real_dummy/statistics-deep-dive/00-roadmap.md"
git commit -m "docs(statistics-deep-dive): 脚手架——目录结构+验证脚本+roadmap初版(21行待撰写)"
```

---

## 板块 I:经典推断统计基础(Task 2~6)

### Task 2: 01 概率论回顾与描述统计(6点)

**Files:** Create `for_real_dummy/statistics-deep-dive/01-probability-recap-and-descriptive.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes 无。Produces:知识点 6(协方差与相关系数的合成数据集)被 Task 9(08因果推断基础)复用,构造"控制混淆变量前后效应估计差异"时需要引用同一个数据生成过程。

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | 常见分布回顾 | 伯努利/二项/泊松/正态/指数;数值模拟样本矩收敛到理论矩 |
| 2 | 矩与偏度峰度 | 正态样本验证偏度≈0峰度≈0,指数分布验证正偏 |
| 3 | 大数定律 | 样本均值随n增大收敛到总体均值,`\|样本均值-真值\|`随n下降趋势 |
| 4 | 中心极限定理 | 均匀分布样本均值分布,大n时用KS检验数值验证趋于正态(不是目测) |
| 5 | 描述统计的稳健性 | 含离群值数据集,量化均值 vs 中位数对同一离群值的响应幅度差异 |
| 6 | 协方差与相关系数 | 构造"高度相关但由第三变量驱动"的合成数据,为板块II"相关不是因果"埋伏笔,数据生成过程需要在代码里显式暴露真实因果结构(供 Task 9 复用同一生成逻辑) |

- [ ] Step 1~5:按标准文件级步骤模式(见 File Structure 节)执行

---

### Task 3: 02 点估计理论(7点)

**Files:** Create `for_real_dummy/statistics-deep-dive/02-point-estimation.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes 无。Produces:知识点7(MLE渐进正态性的重复抽样框架)被 Task 4(03类)验证"置信区间覆盖率实验"复用同一套重复抽样脚手架模式(不是复用代码,是复用同一验证手法)。

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | 矩估计法(MOM) | 估计指数分布λ,误差随n减小 |
| 2 | 极大似然估计(MLE) | 正态分布μ,σ的MLE,与`scipy.stats.norm.fit()`交叉验证 |
| 3 | 无偏性 | 数值验证样本方差n-1分母无偏、n分母有偏(重复抽样几千次) |
| 4 | 一致性 | 估计量方差随n增大趋于0 |
| 5 | 有效性与Cramér-Rao下界 | MLE估计量方差数值接近CR下界 |
| 6 | 充分统计量 | 验证给定充分统计量后原始数据条件分布不依赖参数(简化数值例子) |
| 7 | MLE的渐进正态性 | 重复抽样几千次,MLE估计量分布用KS/Shapiro检验验证趋于正态——本文件"数学结论数值验证"旗舰例子 |

- [ ] Step 1~5:按标准文件级步骤模式执行

---

### Task 4: 03 区间估计与假设检验框架(7点)

**Files:** Create `for_real_dummy/statistics-deep-dive/03-interval-estimation-and-testing-framework.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes Task 3 知识点7的重复抽样验证手法。Produces:知识点3(功效计算)的手写公式被 Task 7(06类"样本量计算")直接复用;知识点4(p值误用)被 capstone(Task 22)引用作为面试场景的第一层追问。

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | 置信区间的正确构造与解读 | 重复抽样几千次构造95%CI,数值验证约95%区间覆盖真值(coverage probability实验) |
| 2 | Neyman-Pearson引理 | 似然比检验在给定显著性水平下功效最大,数值对比次优检验的功效 |
| 3 | Type I/II错误与检验力 | 数值模拟给定效应量/样本量下的功效,和理论公式对比 |
| 4 | p值真实含义与误用 | 构造"p值被误读成H0为真的概率"反例,数值展示两者不同 |
| 5 | 效应量与统计显著≠实际显著 | 大样本下p值很小但效应量微不足道的例子 |
| 6 | 单侧vs双侧检验 | 同一数据集单双侧给出不同结论的边界case |
| 7 | 正态性假设违反的稳健性 | 严重偏态数据跑t检验 vs bootstrap结果对比,量化偏差 |

- [ ] Step 1~5:按标准文件级步骤模式执行

---

### Task 5: 04 经典检验方法(8点)

**Files:** Create `for_real_dummy/statistics-deep-dive/04-classical-tests.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes 无直接代码依赖。Produces:知识点1(t检验)、知识点2(配对vs独立)被 Task 15(14类模型评测统计知识点4)引用做对比;知识点4(ANOVA的多重比较膨胀)呼应 Task 6(05类多重检验)。

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | 单/双样本t检验 | 与`scipy.stats.ttest_1samp`/`ttest_ind`交叉验证统计量和p值 |
| 2 | 配对t检验 vs 独立双样本 | 配对数据展示忽略配对结构低估显著性 |
| 3 | 卡方检验 | 拟合优度+独立性检验,与`scipy.stats.chi2_contingency`交叉验证 |
| 4 | 单因素ANOVA | 与`scipy.stats.f_oneway`交叉验证;数值模拟"多次两两t检验"的Type I错误率膨胀,对比一次ANOVA |
| 5 | Mann-Whitney U检验 | 与`scipy.stats.mannwhitneyu`交叉验证,非正态数据展示比t检验更稳健 |
| 6 | KS检验 | 与`scipy.stats.kstest`交叉验证 |
| 7 | 置换检验 | 手写实现,和参数检验结果交叉对比,不依赖分布假设 |
| 8 | 自助法(bootstrap)CI | 手写实现,正态数据下接近参数法CI、偏态数据下有差异 |

- [ ] Step 1~5:按标准文件级步骤模式执行

---

### Task 6: 05 多重检验与回归推断(7点)

**Files:** Create `for_real_dummy/statistics-deep-dive/05-multiple-testing-and-regression-inference.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes Task 5 知识点4(ANOVA多重比较膨胀现象)。Produces:知识点1-3(多重检验)被 Task 15(14类知识点6"刷榜陷阱")直接复用同一套模拟手法;知识点4-7(回归推断)手写正规方程实现被 Task 17(16类scaling law拟合)的置信区间部分复用。

**重要:`statsmodels` 未安装(已用 `.venv/Scripts/python.exe -c "import statsmodels"` 实测确认 `ModuleNotFoundError`)——知识点5/6/7 全部手写正规方程 `(X'X)^-1 X'y` + 解析标准误公式,简单线性回归部分可用 `scipy.stats.linregress` 交叉验证,不安装 statsmodels。**

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | 多重比较问题 | 数值模拟20个独立无效应检验,验证至少一个p<0.05概率≈1-0.95^20≈64% |
| 2 | Bonferroni校正 | 校正后验证族错误率(FWER)被控制在名义水平 |
| 3 | FDR与BH流程 | 手写BH流程,已知真假阳性标签的合成数据验证FDR低于阈值 |
| 4 | OLS假设 | 构造异方差数据,展示标准误估计错误 |
| 5 | 回归系数CI与假设检验 | 手写正规方程+解析标准误,简单线性场景与`scipy.stats.linregress`交叉验证 |
| 6 | 残差诊断 | 数值化Q-Q对比(残差分位数vs理论分位数偏差统计量);简化版异方差检验 |
| 7 | 逻辑回归系数推断(Wald检验) | 手写实现,已知真实系数的合成数据验证估计准确 |

- [ ] Step 1~5:按标准文件级步骤模式执行

---

## 板块 II:实验设计与因果推断(Task 7~11)

### Task 7: 06 A/B测试设计与功效分析(6点)

**Files:** Create `for_real_dummy/statistics-deep-dive/06-ab-test-design-and-power.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes Task 4 知识点3(功效计算公式)。Produces:知识点1(样本量计算)被 Task 12(13类贝叶斯应用知识点1)、Task 8(07类CUPED)复用同一个"合成A/B测试数据生成器";知识点4(窥探问题)被 capstone 直接引用。

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | 样本量计算 | 手写公式算n,数值模拟(真实生成n样本重复1000次跑检验)验证达到目标功效 |
| 2 | 最小可检测效应(MDE) | 固定样本量反推可靠检测的最小效应量 |
| 3 | 功效曲线 | 数值扫描不同n/效应量组合,验证功效单调性 |
| 4 | 窥探问题(peeking) | 数值模拟"每天看一次、显著就停",展示真实Type I错误率远超名义5%——经典陷阱旗舰例子 |
| 5 | 多指标/多变体的多重比较 | 数值模拟测10个不相关指标,假阳性率膨胀 |
| 6 | 样本量不足的误判 | 真实存在小效应但样本量不足场景,"未显著"≠"没有效果"的具体数字对比 |

- [ ] Step 1~5:按标准文件级步骤模式执行

---

### Task 8: 07 现代实验方法(5点)

**Files:** Create `for_real_dummy/statistics-deep-dive/07-modern-experimentation.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes Task 7 知识点1的A/B测试数据生成器、知识点4的窥探问题场景。Produces:知识点4(新奇效应时间衰减数据)被 Task 10(10类真实陷阱)、capstone 复用;知识点3(CUPED)被 capstone 引用。

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | 序贯检验基础 | 简化O'Brien-Fleming式边界检验,对比naive反复检验的Type I错误率 |
| 2 | Always-valid p值直觉 | 数值模拟展示任意时刻停止查看不膨胀Type I错误(对照Task 7知识点4窥探问题) |
| 3 | CUPED方差削减 | 手写实现,用前置协变量调整后指标方差下降,等效样本量提升 |
| 4 | 新奇效应与学习效应 | 构造"效应随时间衰减"合成时间序列,只看聚合均值得出错误结论 |
| 5 | 网络效应/SUTVA违反 | 构造有社交网络溢出效应的合成实验,naive随机分组估计有偏 |

- [ ] Step 1~5:按标准文件级步骤模式执行

---

### Task 9: 08 因果推断基础(6点)

**Files:** Create `for_real_dummy/statistics-deep-dive/08-causal-inference-foundations.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes Task 2 知识点6(01类协方差合成数据生成过程,复用同一因果结构做混淆变量控制前后对比)。Produces:potential outcomes框架被 Task 10(09类DID/IV/PSM)全部复用作为理论基础。

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | Potential outcomes框架 | 合成数据(已知真实potential outcomes)算出真实ATE作为标杆 |
| 2 | 为什么RCT是金标准 | 数值验证随机分组下朴素差分估计量无偏(重复模拟几千次均值≈真实ATE) |
| 3 | 混淆变量 | 观察性数据数值展示朴素差分估计量有偏,偏差方向和幅度可算出 |
| 4 | DAG初步(链/叉/对撞) | 对撞结构(collider)构造"控制了不该控制的变量反而引入偏差"数值例子 |
| 5 | 为什么相关不是因果 | 回收 Task 2 知识点6数据集,算出混淆变量控制前后效应估计差异 |
| 6 | 反事实推理直觉 | potential outcomes框架下计算个体层面反事实差异 |

- [ ] Step 1~5:按标准文件级步骤模式执行

---

### Task 10: 09 观察性因果推断方法(6点)

**Files:** Create `for_real_dummy/statistics-deep-dive/09-observational-causal-methods.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes Task 9 全部知识点(potential outcomes框架、混淆变量概念)。Produces:知识点6(每种方法失效条件的对照实验)是"方案批判迭代轴"在本板块的集中体现,被 capstone 引用作追问链素材。

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | 双重差分法(DID) | 手写实现,平行趋势假设满足/违反两组数据,展示无偏/有偏对比 |
| 2 | 工具变量法(IV) | 手写两阶段最小二乘(2SLS),已知真实因果效应的合成数据验证收敛 |
| 3 | 倾向得分匹配(PSM) | 手写倾向得分(逻辑回归)+最近邻匹配,和朴素差分对比偏差削减 |
| 4 | 断点回归(RDD) | 合成数据有清晰断点,数值估计断点处跳跃(局部效应) |
| 5 | 平行趋势假设的数值检验 | 处理组/对照组处理前多期趋势差异的数值化检验统计量 |
| 6 | 观察性方法的适用边界 | 每种方法构造一个"假设违反导致估计错误"的对照实验 |

- [ ] Step 1~5:按标准文件级步骤模式执行

---

### Task 11: 10 真实陷阱案例集(5点)

**Files:** Create `for_real_dummy/statistics-deep-dive/10-real-world-traps.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes Task 8 知识点4(新奇效应时间衰减数据生成器)。Produces:知识点5(诊断真实数据新题型的第一个正式案例)是 capstone 诊断环节的直接模板。

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | Simpson悖论 | 经典分组反转数据集(合计趋势与分组趋势相反) |
| 2 | 选择偏差 | "只观察到通过筛选的样本"合成数据,估计量偏差 |
| 3 | 幸存者偏差 | "失败样本系统性缺失"合成数据集,对现存样本推断的错误结论 |
| 4 | SUTVA违反的其他形式 | 聚焦"处理版本不一致"等(和 Task 8 知识点5网络效应区分开,不重复) |
| 5 | 真实A/B测试事故复盘(诊断新题型) | 构造"埋点在实验开始后第3天才生效导致前3天数据为0"的合成日志,不排查直接分析会得出错误结论——要求先诊断再分析,这是"诊断真实数据"新题型的旗舰例子 |

- [ ] Step 1~5:按标准文件级步骤模式执行

---

## 板块 III:贝叶斯方法(Task 12~14)

### Task 12: 11 贝叶斯推断基础(6点)

**Files:** Create `for_real_dummy/statistics-deep-dive/11-bayesian-inference-foundations.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes Task 7 知识点1(A/B测试数据生成器,用于频率派vs贝叶斯派对比)。Produces:Beta-Binomial共轭机制被 Task 14(13类贝叶斯应用知识点1)直接复用。

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | 先验/似然/后验 | Beta-Binomial共轭例子手算后验参数,和数值网格积分后验对比验证解析公式 |
| 2 | 共轭先验族 | Beta-Binomial/Normal-Normal/Gamma-Poisson,每种至少数值验证一个后验更新公式 |
| 3 | 频率派vs贝叶斯派对比 | 同一数据集算频率派点估计+CI 和贝叶斯后验均值+可信区间,展示先验强弱的影响 |
| 4 | 后验预测分布 | 数值积分/采样算后验预测分布,和plug-in预测对比 |
| 5 | 先验选择的影响 | 同一似然不同先验强度,小样本差异明显、大样本趋同 |
| 6 | 基础比率谬误 | 医学检测假阳性经典数值例子,呼应"模型高置信度不代表真的对" |

- [ ] Step 1~5:按标准文件级步骤模式执行

---

### Task 13: 12 MCMC基础(5点)

**Files:** Create `for_real_dummy/statistics-deep-dive/12-mcmc-foundations.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes Task 12(贝叶斯推断基础,非共轭场景的动机)。Produces:Metropolis-Hastings/Gibbs手写实现被 Task 14(13类)间接引用作为方法论背景(不直接复用代码)。

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | 为什么需要采样 | 非共轭先验例子,数值积分高维不可行,对比MCMC可行性 |
| 2 | Metropolis-Hastings | 手写实现,采样已知分布(如正态),数值验证采样均值/方差收敛到理论值 |
| 3 | Gibbs采样 | 手写实现二维联合分布Gibbs采样,验证边际分布收敛 |
| 4 | 收敛诊断 | 构造"burn-in不足"反例,数值展示前段样本均值明显偏离真值 |
| 5 | 接受率与步长调优 | 数值扫描不同提议分布步长下的接受率,过大/过小步长降低有效样本量 |

- [ ] Step 1~5:按标准文件级步骤模式执行

---

### Task 14: 13 贝叶斯应用(5点)

**Files:** Create `for_real_dummy/statistics-deep-dive/13-bayesian-applications.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes Task 12 知识点1(Beta-Binomial共轭)、Task 7 知识点1(A/B测试数据生成器)、Task 8 知识点1-2(序贯检验场景)。Produces:知识点5的方法论对比表被 capstone 直接引用作为板块III的收束。

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | 贝叶斯A/B测试 | 用 Task 12 的Beta-Binomial共轭手写实现,和 Task 7 频率派A/B测试p值结论对比同一份合成数据 |
| 2 | 可信区间vs置信区间语义差异 | 构造两者数值重合但语义不同的例子,再构造数值明显不同的例子(小样本强先验) |
| 3 | Bayes factor模型比较 | 数值计算两个候选模型的Bayes factor,和似然比检验结论对比 |
| 4 | 贝叶斯早停优势 | 回收 Task 8 序贯检验场景,展示贝叶斯后验概率持续更新不需要专门边界校正 |
| 5 | 何时用贝叶斯何时用频率派 | 方法论对比,复用前4点已产出数字做总结性数值对照表 |

- [ ] Step 1~5:按标准文件级步骤模式执行

---

## 板块 IV:AI/ML场景专属统计(Task 15~19)

### Task 15: 14 模型评测统计(6点)

**Files:** Create `for_real_dummy/statistics-deep-dive/14-model-evaluation-statistics.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes Task 5 知识点1-2(t检验/配对检验)、Task 6 知识点1-3(多重检验)。Produces:paired bootstrap机制被 capstone 复用于最终诊断环节。

- [ ] **执行前置动作**:先 Read `learning/llm-judge-arena/README.md` 确认能否为知识点1-2建立真实关联场景(评测集抽样方差的具体案例),若无真实关联如实用通用工程场景(参照 G8)。

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | 单个benchmark分数会骗人 | 数值bootstrap评测集,展示同一模型在不同评测子集上的分数波动range |
| 2 | Paired bootstrap比较模型 | 手写实现,两个已知真实差异的模型输出(合成数据),验证bootstrap CI能否正确捕捉差异 |
| 3 | McNemar检验 | 比较两分类器同一测试集上错误模式,与scipy或手写卡方形式交叉验证 |
| 4 | 配对vs非配对比较的选择 | 呼应 Task 5 知识点2,同一评测数据两种方法对比结论差异 |
| 5 | 评测集大小与CI宽度 | 数值扫描评测集大小,CI宽度收敛速度(1/√n) |
| 6 | "刷榜"的隐藏多重比较 | 数值模拟"验证集上试100个超参选最好"系统性高估真实性能的幅度,呼应 Task 6 |

- [ ] Step 1~5:按标准文件级步骤模式执行(Step 1 之前先完成上面的执行前置动作)

---

### Task 16: 15 排位系统(5点)

**Files:** Create `for_real_dummy/statistics-deep-dive/15-ranking-systems.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes 无。Produces:无(相对独立的知识簇)。

- [ ] **执行前置动作**:先 Read `learning/llm-judge-arena/README.md`(如 Task 15 已读过可直接复用结论)确认知识点4能否建立真实关联,不能强行编。

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | Bradley-Terry模型 | 手写MLE估计,已知真实能力排序的合成成对比较数据,验证估计排序与真实排序一致 |
| 2 | Elo评分系统 | 手写实现,和Bradley-Terry在简单场景下的等价关系数值验证 |
| 3 | Elo的K因子与收敛速度 | 数值扫描不同K值,收敛速度vs稳定性权衡 |
| 4 | 呼应llm-judge-arena | 真实关联存在则引用其评分机制,否则如实标注为通用类比(见前置动作) |
| 5 | TrueSkill简介 | 简化的单一高斯近似演示核心思想,不要求完整实现TrueSkill |

- [ ] Step 1~5:按标准文件级步骤模式执行(Step 1 之前先完成上面的执行前置动作)

---

### Task 17: 16 Scaling law与外推(5点)

**Files:** Create `for_real_dummy/statistics-deep-dive/16-scaling-laws-and-extrapolation.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes Task 6 知识点5(回归系数CI的手写正规方程实现,复用到拟合不确定性部分)。Produces:无。

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | 幂律关系与log-log线性化 | 已知幂律关系合成数据,验证log-log线性回归能恢复真实指数 |
| 2 | 最小二乘拟合幂律参数 | 与`scipy.optimize.curve_fit`交叉验证 |
| 3 | 拟合不确定性 | bootstrap重采样拟合参数,数值给出参数分布(呼应 Task 6 回归推断) |
| 4 | 外推风险 | 数值展示外推距离和预测区间宽度的关系 |
| 5 | 模型选择直觉(简单AIC/BIC) | 数据对比幂律拟合和多项式过拟合的样本外表现差异 |

- [ ] Step 1~5:按标准文件级步骤模式执行

---

### Task 18: 17 分布漂移与监控(5点)

**Files:** Create `for_real_dummy/statistics-deep-dive/17-distribution-shift-and-monitoring.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes Task 5 知识点6(KS检验实现)。Produces:无。

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | KL散度 | 手写实现,与`scipy.stats.entropy`交叉验证 |
| 2 | KS检验用于漂移检测 | 与`scipy.stats.ks_2samp`交叉验证,有/无真实分布偏移的合成数据流验证检测能力 |
| 3 | PSI(Population Stability Index) | 手写实现,和KS检验在同一合成数据上对比敏感度 |
| 4 | 协变量偏移 vs 概念偏移 | 分别构造两种偏移合成数据,同一监控指标的不同反应模式 |
| 5 | 监控阈值设定 | 数值扫描不同阈值下误报率/漏报率曲线(ROC思路的监控版应用) |

- [ ] Step 1~5:按标准文件级步骤模式执行

---

### Task 19: 18 标注一致性与分析方法论(5点)

**Files:** Create `for_real_dummy/statistics-deep-dive/18-annotator-agreement-and-methodology.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes 板块IV全部前序知识点(知识点5汇总性引用)。Produces:知识点5的"经得起追问的具体数字"方法论收束被 capstone 呼应。

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | Cohen's kappa | 手写实现,已知手算例子交叉验证;构造"一致率高但kappa低"的类别不均衡反例 |
| 2 | Fleiss' kappa | 手写实现,退化到2标注者时和Cohen's kappa数值一致 |
| 3 | Krippendorff's alpha简介 | 概念性对比,给出简化数值例子(视执行情况可精简但不可省略可运行例子) |
| 4 | 一致性不够时的处理策略 | 具体"排查出分歧原因"的合成案例,不是空谈"要沟通" |
| 5 | "经得起追问的具体数字"方法论收尾 | 汇总板块IV(Task 15~19)已产出的真实数字做交叉引用,不产出新的独立代码块,但仍需至少一段整合性可运行例子 |

- [ ] Step 1~5:按标准文件级步骤模式执行

---

## 板块 V:时间序列基础(Task 20~21)

### Task 20: 19 时间序列基础(6点)

**Files:** Create `for_real_dummy/statistics-deep-dive/19-time-series-foundations.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes 无。Produces:平稳性/ACF概念被 Task 21(20类)直接复用。

**硬约束(spec §1.2)**:不假设随机过程先修知识,平稳性/自相关只用初等定义+数值模拟讲清楚,不引用鞅或马尔可夫链的形式化理论。撰写前重读 spec §1.2 确认没有违反这条约束。

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | 时间序列与iid数据的区别 | 数值对比iid数据和AR(1)数据的自相关函数差异 |
| 2 | 平稳性(初等定义) | 数值检验(滚动均值/方差是否随时间变化)区分平稳与非平稳合成序列 |
| 3 | ACF与PACF | 手写实现,与已知AR(p)/MA(q)理论ACF/PACF形状数值对比 |
| 4 | 白噪声 | 数值验证白噪声序列ACF在滞后>0时接近0(置信带内) |
| 5 | 随机游走(初等递推定义) | 数值展示随机游走方差随时间线性增长(不平稳的具体表现) |
| 6 | 差分使非平稳序列变平稳 | 随机游走序列一阶差分,数值验证差分后方差不再增长 |

- [ ] Step 1~5:按标准文件级步骤模式执行

---

### Task 21: 20 简单预测方法(5点)

**Files:** Create `for_real_dummy/statistics-deep-dive/20-simple-forecasting.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes Task 20 全部知识点(平稳性/差分)。Produces:无。

**硬约束**:知识点3只讲AR(1)初等形式(用过去值线性预测未来值),不深入完整ARIMA数学,不假设随机过程先修。

| # | 知识点 | 范围与验证锚点 |
|---|--------|--------------|
| 1 | 移动平均 | 手写实现,不同窗口大小对噪声平滑程度和滞后的权衡 |
| 2 | 指数平滑 | 手写实现,与移动平均对比对最近数据的加权差异 |
| 3 | 简单AR(1)概念 | 手写最小二乘估计AR(1)系数,已知系数的合成数据验证估计准确 |
| 4 | 预测评估指标(MAE/RMSE/MAPE) | 数值计算,展示同一预测误差在不同指标下的相对大小关系 |
| 5 | Naive baseline陷阱 | 合成数据集数值对比复杂方法和"预测=昨天的值"的预测误差,**如实**展示naive baseline有时更优,不为了"证明方法有用"回避这个真实陷阱 |

- [ ] Step 1~5:按标准文件级步骤模式执行

---

## 收尾(Task 22~23)

### Task 22: 21 模拟终面capstone

**Files:** Create `for_real_dummy/statistics-deep-dive/21-mock-interview-capstone.md`;Modify `00-roadmap.md`

**Interfaces:** Consumes 至少 6 处前序知识点(见下方场景大纲的逐段标注),是全系列唯一要求同时用 3 条以上 G3 轴线的文件。Produces:无(终点)。

**场景大纲**(叙事体,仿 `dsa-deep-dive/19-mock-interview-capstone.md` 和 `20-advanced-interview-depth.md` 的场景叙事风格,不是知识点列表):

候选人在 frontier lab 负责评估一个新训练方法(如优化器改动)要不要合并/上线,面试官层层追问,至少覆盖以下递进(每段配合真实数值分析,不是空对话):
1. 候选人展示"跑了5个随机种子,4个更好,1个更差,平均更好,显著,建议合并"→面试官追问显著性怎么算的、"显著"这个词是不是就等于"一定有效"(**consumes Task 5(04类)t检验的具体计算 + Task 4(03类)知识点4 p值真实含义**)
2. 追问样本量(5个种子)够不够做出这个判断(**consumes Task 7 功效分析**)
3. 追问"5个种子里选平均"本身是不是一种多重比较/选择性报告(**consumes Task 6 多重检验**)
4. 追问随机种子的选择过程是否存在隐藏的非随机化(**consumes Task 9 因果推断:随机化为什么重要**)
5. 追问能不能给一个概率化的"这个改进有多大把握"而不是死板的显著/不显著(**consumes Task 14 贝叶斯A/B测试**)
6. 最后给一段真实的5个种子loss曲线数字(其中1个种子在训练中途有一个异常跳变),要求候选人诊断这个异常种子是真实的方差还是训练脚本的bug(**consumes Task 11 诊断新题型手法,呼应G3"诊断真实数据"轴线**)

- [ ] **Step 1: 撰写文件**,场景连贯推进(不是6段孤立代码,是一个递进叙事+至少1段把上述判断串成完整数值分析流程的综合可运行例子),每个追问阶段标注呼应哪个既有知识点(格式参照 `dsa-deep-dive/19-mock-interview-capstone.md`)
- [ ] **Step 2**: 运行验证脚本,期望全部代码块通过
- [ ] **Step 3**: 修复失败块;综合例子若涉及随机性额外压力测试10~15次
- [ ] **Step 4**: 更新 `00-roadmap.md` 第21行为 `✅`
- [ ] **Step 5**: 提交

```bash
git add "for_real_dummy/statistics-deep-dive/21-mock-interview-capstone.md" "for_real_dummy/statistics-deep-dive/00-roadmap.md"
git commit -m "docs(statistics-deep-dive): 21类 模拟终面capstone——跨板块场景叙事,串联6处既有知识点"
```

---

### Task 23: 全库自查回归 + README + memory 更新

**Files:**
- Modify: `for_real_dummy/statistics-deep-dive/00-roadmap.md`(grand total 行)
- Modify: `for_real_dummy/README.md`("独立技能系列"表格 + 目录树)
- Create: `C:\Users\ericp\.claude\projects\e--Workspace-dummy\memory\statistics-deep-dive-complete.md`
- Modify: `C:\Users\ericp\.claude\projects\e--Workspace-dummy\memory\MEMORY.md`

**Interfaces:** Consumes 全部 Task 1~22 产出。Produces:无(终点)。

- [ ] **Step 1: 全部文件独立重跑验证**(不只是跑最后一批)

```bash
for f in for_real_dummy/statistics-deep-dive/[0-2]*.md; do
  .venv/Scripts/python.exe for_real_dummy/statistics-deep-dive/_verify_md.py "$f" || echo "FAILED: $f"
done
```

期望:全部文件末行 `ALL N blocks passed`,没有 `FAILED` 输出。

- [ ] **Step 2: 结构标记计数核查**

对每个 01~20 文件,用 grep 统计"### "开头的知识点标题数量,和 `00-roadmap.md` 进度表声称的知识点数逐行核对;统计总数是否接近"约116"(±10~15%容差内属正常,超出则复核是否漏写或多算)。

- [ ] **Step 3: assert 覆盖率核查**

```bash
for f in for_real_dummy/statistics-deep-dive/[0-2]*.md; do
  echo "$f: $(grep -c 'assert' "$f") asserts"
done
```

期望:每个文件 assert 数量 ≥ 该文件知识点数(至少每点一个 assert,多数点会有多个)。

- [ ] **Step 4: 随机性/计时类断言压力测试**

对涉及 `np.random`/`random`/bootstrap/MCMC/计时对比的文件(至少 Task 4/9/10/11/12/13/17/20/21 对应文件),额外重复运行 `_verify_md.py` 5~15 次,确认无间歇性失败(参照 dsa-deep-dive 01 类的教训:只测 1~3 次不足以确认稳定)。

- [ ] **Step 5: 更新 `00-roadmap.md`** grand total 行为实际统计数字(不强行凑成116)

- [ ] **Step 6: 更新 `for_real_dummy/README.md`**

"独立技能系列"表格(rhcsa-bash-deep-dive、dsa-deep-dive 所在的表格)新增一行 statistics-deep-dive,内容风格参照 dsa-deep-dive 那一行(板块结构+知识点数+capstone+验证环境说明);目录结构树 `for_real_dummy/` 部分新增对应子树条目。

- [ ] **Step 7: 创建 memory** `statistics-deep-dive-complete.md`,frontmatter 含 `name`/`description`/`metadata.type: project`,正文记录:5大板块20文件+capstone 的实际完成规模、和 dsa-deep-dive 20类同源的五轴方法论应用方式、本系列独有的"数学结论必须数值验证"纪律、`statsmodels`未安装改用手写正规方程的技术决策、commit 记录

- [ ] **Step 8: 更新 `MEMORY.md`** 索引,新增一行指向 `statistics-deep-dive-complete.md`;如 `queued-interview-depth-upgrade.md`/`four-new-deep-dive-series-2026-07.md` 有相关表述需要同步"下一步方向"状态,一并更新(不强行修改无关内容)

- [ ] **Step 9: 最终提交**

```bash
git add "for_real_dummy/statistics-deep-dive/00-roadmap.md" "for_real_dummy/README.md"
git commit -m "docs(statistics-deep-dive): 全库自查回归通过+README集成——统计系列收官"
```

---

## Self-Review Notes(执行前已自查,供参考不重复展开)

- **Spec覆盖**:spec §2 全部21个文件在 Task 1~22 均有对应任务;§3 八步模板→G1;§4 五轴方法论→G3;§5 验证纪律→G4/G5;§6 文件组织→File Structure节;§7 README集成→Task 23 Step 6;§8 Phase划分→Task 1~23 的顺序即对应 Phase 0~7,粒度细化到文件级。
- **占位符扫描**:全文没有 TBD/"适当补充"/"类似Task N不再展开"这类表述;每个 Task 的知识点表格都是具体范围+具体验证锚点,不是空泛描述。
- **依赖一致性**:已核实 `statsmodels` 未安装(Task 6 已标注硬约束,Task 17 scaling law 拟合部分只依赖 `scipy.optimize.curve_fit`,不依赖 statsmodels);numpy/scipy 版本已实测。跨 Task 的"Consumes/Produces"标注的是概念/数据生成过程复用,不是跨文件 import(每个 md 文件的代码块相互独立执行,这是 `_verify_md.py` 的设计前提,复用只能是"抄一遍生成逻辑"不能是"导入另一个文件"——这一点在执行时需要注意,不能让知识点之间产生真实的 Python import 依赖)。

---

## 设计签字

- **计划日期**: 2026-07-13
- **依据**: `docs/superpowers/specs/2026-07-13-statistics-deep-dive-design.md`(commit 6aa8f81)
- **执行授权**: 用户已明确"批准 正式开始 一口气分批次做完 中途无需我的反复确认",本计划生成后直接进入执行,不再暂停等待计划本身的用户复核。
