# 08 · 因果推断基础深挖(Causal Inference Foundations)

> 总览见 [00-roadmap.md](00-roadmap.md)

06-07类处理的是"能做随机化实验"这个理想情形下的设计与分析问题。但现实中大量重要问题根本没法做真正的A/B测试(不能强制随机分配用户是否使用一个已经上线多年的核心功能,不能随机分配病人吃不吃药,不能随机分配国家采不采用某个经济政策)——本文建立**potential outcomes框架**这套因果推断的通用理论语言,回答"随机化实验为什么天然可信、观察性数据为什么天然危险"这个根本问题,为09类"做不了随机化实验时,有哪些正规的补救方法"打基础。

**环境声明:** 全部代码在仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1)下真实跑通,随机抽样固定种子(`np.random.default_rng(42)`)。

---

## 1. Potential Outcomes框架 —— 因果推断的根本问题

**定义与记号:** potential outcomes(潜在结果)框架:对每个个体i,定义两个潜在结果Y_i(1)(如果ta接受处理会观测到的结果)和Y_i(0)(如果ta不接受处理会观测到的结果)。个体处理效应τ_i=Y_i(1)-Y_i(0)。但现实中每个个体只能观测到其中一个(实际分配到的那个),另一个是永远无法直接观测的"反事实"(counterfactual)——这被称为**因果推断的根本问题**(fundamental problem of causal inference)。平均处理效应ATE=E[τ_i]=E[Y(1)]-E[Y(0)]。

**一句话:** potential outcomes框架的核心洞察是——因果效应的定义本身不依赖任何具体的估计方法,是"如果同一个人分别经历两种情况,结果会差多少"这个思想实验;统计方法只是想办法在"每人只能观测一种情况"这个现实约束下,尽量逼近这个思想实验的答案。

**数学推导:** ATE=E[Y_i(1)-Y_i(0)]=E[Y_i(1)]-E[Y_i(0)](期望的线性性,这一步永远成立,和Y(1)、Y(0)是否独立无关)。但朴素差分估计量 Ȳ_treatment-Ȳ_control 在数学上等价于 E[Y(1)|Z=1]-E[Y(0)|Z=0](Z是处理分配指示变量),这和真正想要的 E[Y(1)]-E[Y(0)] 只有在"处理分配Z与潜在结果对(Y(1),Y(0))相互独立"时才相等——这个独立性条件正是知识点2要讲的"为什么RCT是金标准"的数学核心。

**底层机制/为什么这样设计:** 为什么需要"两个潜在结果"这么抽象的构造,而不是直接对观测数据做统计推断?因为如果不显式定义"没有发生的那个结果"(反事实),就无法把"因果效应"和"两组人本来就不一样(选择偏差)"这两件事在数学上区分开——potential outcomes框架强迫你承认每个个体原则上有一个确定的τ_i(哪怕永远测不到),这样"估计量的偏差从哪里来"才能被精确地写成数学表达式,而不是只能定性地说"可能有偏差"。

**AI研究/工程场景:** 这个框架是09类要讲的DID、IV、PSM、RDD等所有观察性因果推断方法的共同理论地基——不管具体技术手段是什么,这些方法最终都是在回答同一个问题:"在无法做真正随机实验的情况下,怎么用手头的观察性数据去逼近Y_i(1)和Y_i(0)这两个反事实量"。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)
n = 5000

# "上帝视角"合成数据: 每个个体的两个潜在结果都已知(现实中不可能获得)
baseline = rng.normal(50, 10, n)
true_tau = 8.0
Y1 = baseline + true_tau + rng.normal(0, 5, n)  # 潜在结果: 接受处理
Y0 = baseline + rng.normal(0, 5, n)              # 潜在结果: 不接受处理

true_ate = (Y1 - Y0).mean()
assert abs(true_ate - true_tau) < 0.5, f"true ATE should be close to true_tau={true_tau}, got {true_ate:.4f}"

# 模拟"现实限制": 随机分配处理, 每个个体只能观测到潜在结果中的一个
Z = rng.integers(0, 2, n)
Y_observed = np.where(Z == 1, Y1, Y0)

# 因果推断的根本问题: 每个个体缺失了另一个潜在结果(反事实)
n_missing_total = (Z == 0).sum() + (Z == 1).sum()
assert n_missing_total == n, "every individual contributes exactly one missing (counterfactual) potential outcome"

naive_diff = Y_observed[Z == 1].mean() - Y_observed[Z == 0].mean()
assert abs(naive_diff - true_ate) < 1.0, \
    f"with random assignment, naive diff ({naive_diff:.4f}) should already be close to true ATE ({true_ate:.4f})"

print(f"true ATE (god's-eye view, using both potential outcomes) = {true_ate:.4f}")
print(f"naive diff-in-means (only observing one outcome per person, as in reality) = {naive_diff:.4f}")
```

**面试怎么问+追问链**(决策依据追问轴):
- Q:"'因果推断的根本问题'具体指什么,为什么说它是'根本'问题?"
- 追问1:"如果我们真的造出时光机,能让同一个人重新经历一次不同的处理,这个问题会消失吗?"(理论上会消失——这说明这个问题的根源不是统计方法不够好,而是反事实数据在现实中physically不可获得,所有因果推断方法本质上都是在用不同的假设条件"曲线救国",不存在能完全绕开这个根本限制的"更好的统计方法")
- 深挖追问:"既然反事实永远无法在个体层面观测到,那ATE(总体平均处理效应)为什么还能被合理估计出来?"(个体层面的τ_i确实永远无法观测,但ATE是总体层面的期望,只需要"处理组的Y(1)分布"和"对照组的Y(0)分布"分别有代表性,不需要在同一个人身上同时观测两个潜在结果——这是从"个体不可知"到"总体可估计"的关键转折,知识点2的RCT正是制造这种代表性的机制)

**常见坑:**
- 把"无法观测个体反事实"和"无法估计总体ATE"混为一谈——前者是永远成立的根本限制,后者在合适的假设下(比如RCT)是可以做到的。
- 忽视potential outcomes本身也是一个建模假设(暗含"个体的潜在结果不受他人处理分配影响",即SUTVA,07类知识点5刚讲过)——如果这个假设不成立,整个框架的定义都需要修正。

---

## 2. 为什么RCT是金标准 —— 随机化如何自动消除混淆

**定义与记号:** RCT(Randomized Controlled Trial,随机对照试验):处理分配Z通过随机化机制决定,不依赖任何个体特征。随机化保证了处理分配Z与潜在结果对(Y(1),Y(0))统计独立(Z⊥(Y(1),Y(0)))——这是RCT被称为"金标准"的数学核心原因。

**一句话:** 随机分组这个动作本身,不需要任何额外假设或统计技巧,就自动让"朴素组间差分"这个最简单的估计量变成无偏估计——这是RCT相对所有观察性方法最大的优势:不需要"假设自己找全了所有混淆变量",随机化直接把这个问题连根拔起。

**数学推导:** 因为随机化保证Z⊥(Y(1),Y(0)),所以E[Y(1)|Z=1]=E[Y(1)|Z=0]=E[Y(1)](条件于Z不改变Y(1)的期望),同理E[Y(0)|Z=1]=E[Y(0)|Z=0]=E[Y(0)]。朴素差分估计量的期望:

E[Ȳ_treatment-Ȳ_control]=E[Y\|Z=1]-E[Y\|Z=0]=E[Y(1)\|Z=1]-E[Y(0)\|Z=0]=E[Y(1)]-E[Y(0)]=ATE

正是独立性让"以Z为条件"这个操作可以直接去掉,朴素差分才精确等于真实ATE,不多不少。

**底层机制/为什么这样设计:** 为什么随机化足以保证独立性,而"看起来很努力地控制变量"的观察性方法却做不到同样的保证?因为随机化是从机制上切断了处理分配Z和任何个体特征(不管是已经测量的还是压根没被测量到的)之间的关联——不管这个个体特征你有没有意识到、有没有收集数据,只要随机化机制本身不依赖它,它就不可能成为混淆变量。观察性方法只能"控制你已经想到并测量了的变量",永远无法排除"没想到的混淆变量"这个风险,这是两者本质的差距。

**AI研究/工程场景:** 这正是为什么A/B测试(06-07类)在能做的场景下永远是因果推断的第一选择,只有在业务、伦理、技术上确实无法做随机化实验时(比如不能强制随机分配用户使用/不使用某个已经上线很久的核心功能),才退而求其次使用09类的观察性方法,并且需要额外承担"混淆变量假设是否成立"这个无法完全验证的风险。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)
n = 2000
true_tau = 8.0

baseline = rng.normal(50, 10, n)
Y1 = baseline + true_tau + rng.normal(0, 5, n)
Y0 = baseline + rng.normal(0, 5, n)
true_ate = (Y1 - Y0).mean()

# 重复模拟: 随机分组下, 朴素差分估计量应该在重复实验中均值收敛到真实ATE(无偏)
n_repeats = 2000
naive_diffs = []
for _ in range(n_repeats):
    Z = rng.integers(0, 2, n)
    Y_obs = np.where(Z == 1, Y1, Y0)
    naive_diffs.append(Y_obs[Z == 1].mean() - Y_obs[Z == 0].mean())
naive_diffs = np.array(naive_diffs)

assert abs(naive_diffs.mean() - true_ate) < 0.1, \
    f"mean of {n_repeats} repeated naive diffs ({naive_diffs.mean():.4f}) should closely match true ATE ({true_ate:.4f})"

# 对照: 非随机分组(处理概率依赖baseline, 制造选择偏差), 朴素差分应该系统性偏离
p_treat = 1 / (1 + np.exp(-(baseline - 50) / 5))  # baseline越高越可能"被选中"接受处理
Z_biased = rng.binomial(1, p_treat)
Y_obs_biased = np.where(Z_biased == 1, Y1, Y0)
biased_diff = Y_obs_biased[Z_biased == 1].mean() - Y_obs_biased[Z_biased == 0].mean()

assert abs(biased_diff - true_ate) > 2.0, \
    f"non-random assignment should produce a naive diff ({biased_diff:.4f}) that clearly deviates from true ATE ({true_ate:.4f})"

print(f"true ATE = {true_ate:.4f}")
print(f"mean of {n_repeats} repeated RCT naive-diff estimates = {naive_diffs.mean():.4f}  (unbiased)")
print(f"single non-random-assignment naive diff = {biased_diff:.4f}  (badly biased)")
```

**面试怎么问+追问链**(真实性验证轴):
- Q:"业务方说'我们没有做真正的随机分组,但两组用户看起来差不多',这样能直接套用RCT的分析方法吗?"
- 追问1:"'看起来差不多'和'随机化保证的独立'有什么本质区别?"("看起来差不多"通常只是在你观测到的几个特征上分布相似,但完全无法保证在未观测特征上也相似;随机化保证的独立性覆盖所有变量,不管有没有被观测到,这是两者不可等价的根本原因)
- 深挖追问:"如果两组用户在所有你能想到、能测量的特征上都做了严格匹配,是不是就等价于随机化了?"(不等价——这正是09类PSM(倾向得分匹配)的核心假设"无混淆性"(unconfoundedness)的实质,这个假设是"假设没有遗漏未观测到的混淆变量",是一个无法从数据本身验证的假设,而随机化不需要做这个假设就能自动满足)

**常见坑:**
- 认为"样本量足够大"就能弥补"不是随机分组"这个问题——样本量只能降低方差,不能消除偏差,这是04-07类反复出现的同一个原则。
- 把"做了随机化"和"随机化执行正确"混为一谈——如果随机化过程本身有系统性问题(比如按访问顺序简单交替分组,恰好和某个时间相关的因素混杂),即使名义上是"随机分组",实际也可能不满足真正的独立性。

---

## 3. 混淆变量 —— 偏差可以被精确算出来,不是模糊的"可能"

**定义与记号:** 混淆变量(confounder):同时影响处理分配Z和结果Y的第三方变量。如果混淆变量存在且没有被控制,朴素差分估计量会同时混入"真实处理效应"和"两组人本来就因为混淆变量不同而产生的差异"这两部分,无法区分。

**一句话:** 混淆变量制造的偏差,本质上是"两组人在实验开始之前就已经不一样",朴素差分把这种"先天不一样"和"处理造成的改变"混在一起报告成了一个数字。

**数学推导:** 设Z是处理分配(不是随机的,依赖混淆变量C),结果Y=τZ+βC+ε(β是C对Y的直接效应)。朴素差分估计量的期望:

E[Ȳ_treatment-Ȳ_control]=E[Y\|Z=1]-E[Y\|Z=0]=τ+β·(E[C\|Z=1]-E[C\|Z=0])

只有当E[C\|Z=1]=E[C\|Z=0](两组C的分布相同,即C与Z无关)时,朴素差分才精确等于τ。偏差项恰好是β×(两组C均值之差)——偏差的方向和幅度完全由"C对Y的效应大小β"和"C在两组间的不平衡程度"这两个因素的乘积决定,不是模糊的"可能有偏差",是可以精确算出来的。

**底层机制/为什么这样设计:** 为什么偏差恰好是这个乘积形式?因为在这个线性设定下,"两组差异"这个统计量天然可以分解成"处理本身的效应"加上"两组在其他维度上原本就存在、且这个维度确实影响Y的那部分差异"——只要C不影响Y(β=0)或者C在两组间完全平衡(均值差=0),偏差项就会归零,这两个条件缺一不可,这也是"控制混淆变量"这个操作在数学上到底在做什么的精确刻画。

**AI研究/工程场景:** "上了这个新功能的用户,后续付费率比没上的用户高20%"这类观察性分析,几乎总是隐藏着混淆变量——比如"更活跃/更有粘性的用户"本来就更可能主动尝试新功能(混淆变量C=活跃度,同时影响"是否使用新功能"和"是否付费"),不控制这个变量直接得出"新功能带来20%付费提升"的结论,会严重高估真实效应。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)
n = 5000
true_tau = 5.0
beta = 3.0  # C对Y的直接效应

C = rng.normal(0, 2, n)  # 混淆变量, 比如用户活跃度
p_treat = 1 / (1 + np.exp(-C))  # C越大越可能接受处理, 制造Z和C的关联
Z = rng.binomial(1, p_treat)
Y = true_tau * Z + beta * C + rng.normal(0, 1, n)

naive_diff = Y[Z == 1].mean() - Y[Z == 0].mean()
c_diff = C[Z == 1].mean() - C[Z == 0].mean()
predicted_bias = beta * c_diff

# 核心断言1: 朴素差分和解析预测值(true_tau + beta*C组间差)吻合
assert abs(naive_diff - (true_tau + predicted_bias)) < 0.3, \
    f"naive diff ({naive_diff:.4f}) should match true_tau + predicted_bias ({true_tau + predicted_bias:.4f})"
assert naive_diff > true_tau + 1.0, \
    f"naive diff should be substantially inflated above the true effect due to confounding, got {naive_diff:.4f} vs true {true_tau}"

# 核心断言2: 控制C之后(回归调整), 效应估计应该回归到接近真实tau
X_design = np.column_stack([np.ones(n), Z, C])
beta_hat = np.linalg.solve(X_design.T @ X_design, X_design.T @ Y)
tau_adjusted = beta_hat[1]
assert abs(tau_adjusted - true_tau) < 0.3, \
    f"after controlling for C, estimate ({tau_adjusted:.4f}) should be close to true tau ({true_tau})"

print(f"true tau = {true_tau}")
print(f"naive diff = {naive_diff:.4f}  (predicted = true_tau + beta*C_diff = {true_tau + predicted_bias:.4f})")
print(f"adjusted (controlling for C) estimate = {tau_adjusted:.4f}  (close to true tau)")
```

**面试怎么问+追问链**(方案批判迭代轴):
- Q:"分析显示'用了新功能的用户留存率明显更高',这个结论有什么问题?"
- 追问1:"怎么证明这不是选择偏差(混淆变量)导致的?"(不能仅凭观察性数据本身证明——需要额外的领域知识去列出所有可能同时影响'是否使用新功能'和'留存率'的变量,并逐一控制,但永远无法保证列全了所有混淆变量,这也是为什么观察性因果推断的结论总是"在无混淆性假设下"成立,不是绝对结论)
- 深挖追问:"如果控制了活跃度这个变量之后,效应估计值从20%降到了5%,你会怎么汇报这个结果?"(应该如实汇报"控制混淆变量前后估计值差异巨大"这件事本身,而不是只挑一个数字汇报——这个巨大差异本身就是"混淆变量影响很强"的有力证据,分析报告里应该明确说明"5%是控制了已知混淆变量后的估计,仍然可能有未观测混淆变量残留偏差"这种诚实的不确定性表达)

**常见坑:**
- 只控制了"容易测量"的混淆变量,遗漏了"难测量但确实存在"的混淆变量(比如用户的主观意愿/动机水平),得出"已经控制了混淆变量所以结论可信"的虚假安全感。
- 把"控制变量后效应变小了"简单解读成"新功能其实没用",而不是更精确地表述为"部分观测到的效应可以由混淆变量解释,剩余部分可能是真实效应也可能仍有残留偏差"。

---

## 4. DAG初步(链/叉/对撞)—— 该不该控制这个变量,画个图就知道

**定义与记号:** DAG(Directed Acyclic Graph,有向无环图):用节点表示变量、有向边表示假设的因果方向来可视化因果结构。三种基本结构:**链**(chain,A→B→C,B是中介变量)、**叉**(fork,A←B→C,B是共同原因/混淆变量,知识点5要复用的温度就是这种结构)、**对撞**(collider,A→B←C,B同时被A和C影响,是两者的共同结果)。三种结构对"要不要控制中间节点B"的正确答案完全不同——这是DAG最重要的实践意义:直接目测就能判断该控制谁、不该控制谁。

**一句话:** 面对"要不要在回归里加这个变量"这个问题,直觉往往说"能加的都加上更保险",但对撞结构恰恰证明这个直觉是错的——控制一个对撞变量,反而会凭空制造出原本不存在的虚假关联。

**数学推导:** 对撞结构A→B←C(A和C独立,都直接影响B,A和C之间没有因果连接)。构造具体例子:A=才华,C=运气,B=是否被顶级期刊录用(才华和运气的某种组合超过阈值即录用)。在**全体人群**里,才华和运气边际独立(Cov(A,C)=0)。但如果**只看被录用的人**(相当于以B为条件筛选样本),才华和运气会呈现负相关——因为在"必须达到录用门槛"这个约束下,运气差的人只能靠才华撑起来,才华差的人只能靠运气,两者变成了互补关系。这个现象叫"对撞偏差"(collider bias)或"伯克森悖论"(Berkson's paradox)。

**底层机制/为什么这样设计:** 为什么"控制对撞变量"会制造虚假关联,而"控制混淆变量(叉结构)"却能消除虚假关联?两者的数学机制其实是同一件事的两面——条件化一个变量,会改变这个变量的"因"之间的联合分布。对叉结构,共同原因B已经在无条件情形下把A和C绑在一起制造了虚假关联,控制B之后这个绑定被切断,关联消失(好事)。对对撞结构,A和C本来独立、没有被任何东西绑在一起,但条件化B(它们共同的果)反而人为地把两者绑在了一起制造出负相关(坏事)。

**AI研究/工程场景:** 招聘漏斗分析是对撞偏差的经典重灾区——"能通过面试的候选人"本身就是一个对撞节点(技术能力和沟通能力共同决定是否通过),如果只在"通过面试的人"这个子样本里分析"技术能力和沟通能力的关系",几乎必然会观测到两者呈负相关(不管真实世界里两者是否独立甚至正相关),这是很多"看似反直觉的人才分析结论"背后真正的统计学陷阱来源。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)
n = 20_000

talent = rng.normal(0, 1, n)
luck = rng.normal(0, 1, n)  # talent, luck相互独立(设计上就是独立的)

corr_marginal = np.corrcoef(talent, luck)[0, 1]
assert abs(corr_marginal) < 0.03, f"talent and luck should be nearly uncorrelated in the full population, got {corr_marginal:.4f}"

# 录用规则: talent+luck超过阈值才被录用(对撞节点B)
score = talent + luck
threshold = np.quantile(score, 0.90)  # 只有top 10%被录用
hired = score > threshold

corr_conditional = np.corrcoef(talent[hired], luck[hired])[0, 1]

# 核心断言: 在录用子样本里, talent和luck呈现明显负相关(伯克森悖论), 尽管全体样本里两者独立
assert corr_conditional < -0.3, \
    f"conditioning on the collider (hired) should induce a strong negative correlation, got {corr_conditional:.4f}"

print(f"correlation(talent, luck) in full population = {corr_marginal:.4f}  (near 0, as designed)")
print(f"correlation(talent, luck) among the hired-only subsample = {corr_conditional:.4f}  (strongly negative -- collider bias)")
```

**面试怎么问+追问链**(诊断真实数据新题型):
- Q:"数据分析显示'我们录用的员工里,技术分数和沟通分数呈负相关',能不能得出'技术强的人沟通能力天生差'这个结论?"
- 追问1:"这个负相关有没有可能是分析方法本身制造出来的,而不是真实世界的规律?"(需要考虑分析所用的样本本身是否是一个对撞节点——"被录用"同时被技术分数和沟通分数影响,只在录用子样本里看两者关系,即使全体候选人里技术和沟通完全独立,录用子样本里也会呈现负相关,这正是DAG对撞结构预测的现象)
- 深挖追问:"怎么验证这确实是对撞偏差,而不是真实存在的负相关?"(理想情况下需要在**全体申请人**(而不是只有被录用的人)的数据里重新计算技术分数和沟通分数的相关性,如果全体申请人里两者不相关或正相关,而只在录用子样本里出现负相关,就是对撞偏差的直接证据)

**常见坑:**
- "多控制变量总是更保险"这个直觉在对撞结构下是错的——控制或以对撞节点为条件筛选样本,会人为制造出原本不存在的虚假关联,不是"最多没有用",是会主动引入新的错误。
- 混淆链结构(A→B→C,B是中介)如果被错误地当成需要控制的变量,会把"A通过B间接影响C"这部分真实的因果路径也一并切断,导致A对C的效应被低估甚至归零——这和对撞结构是不同的错误,但同样是"该不该控制这个变量"判断错误导致的。

---

## 5. 为什么相关不是因果 —— 回收01类的冰淇淋与溺水数据集

**定义与记号:** 复用01类知识点6"气温→冰淇淋销量、气温→溺水人数"这个经典**叉**结构合成数据集,本知识点把它明确接到因果推断的语言体系里:冰淇淋销量(X)和溺水人数(Y)之间存在强相关(01类已验证ρ>0.5),但两者之间没有任何直接因果边——DAG结构是 X←Z→Y(气温Z是共同原因),不是X→Y或Y→X。

**一句话:** 01类是从"描述统计"的角度展示"控制Z之后偏相关消失";本知识点从"因果推断"的角度,明确指出如果错误地把X当作"处理变量"、用朴素回归估计"X对Y的效应",这个估计量在数学上到底错在哪、错了多少,把01类的直觉观察升级成因果框架下可以精确量化的偏差。

**数学推导:** 复用01类的数据生成过程:temperature=Z,ice_cream_sales=X=50+3Z+ε_x,drowning_incidents=Y=2+0.15Z+ε_y。如果错误地把X当作处理变量拟合简单线性回归 Y=α+τ_naive·X+ε,可以证明:

τ_naive = Cov(X,Y)/Var(X) = Cov(3Z, 0.15Z)/Var(3Z) = (3×0.15×Var(Z))/(9×Var(Z)) = 0.45/9 = 0.05

这个0.05不是"处理效应",而是完全由X和Y各自与共同原因Z的关联强度决定的虚假斜率——和知识点3"偏差=β×(两组均值差)"公式(离散处理变量情形)本质相同:偏差项的大小完全由混淆结构的参数决定,和"真实因果效应"(此处为0)无关。

**底层机制/为什么这样设计:** 这个知识点存在的意义,是让"相关不是因果"这句人人都会说的口号,从一句正确的废话,变成一个可以精确计算、可以在具体数据集上复现、可以推广成通用诊断方法(控制住可疑的共同原因,看关联是否消失)的严谨工具——这正是因果推断这个学科相对朴素相关性分析的核心增量价值。

**AI研究/工程场景:** 大模型评测领域一个真实的类比:"模型规模"和"下游任务表现"高度相关,但如果训练数据量、训练计算量(这些经常和模型规模同步扩大的"共同原因")没有被控制,直接把"规模"当作"下游表现提升的原因"来做决策,就可能重复冰淇淋和溺水这个经典错误——16类scaling law会更系统地展开这个话题。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)
n = 10_000

# 完整复用01类知识点6的数据生成过程
temperature = rng.normal(25, 8, n)
ice_cream_sales = 50 + 3 * temperature + rng.normal(0, 10, n)
drowning_incidents = 2 + 0.15 * temperature + rng.normal(0, 1, n)

# 朴素回归: 把ice_cream_sales错误当作"处理变量", 估计对drowning_incidents的效应
X, Y = ice_cream_sales, drowning_incidents
tau_naive = np.cov(X, Y, ddof=0)[0, 1] / np.var(X)

# 解析预测值: tau_naive = Cov(3Z,0.15Z)/Var(3Z) = 0.45*Var(Z) / (9*Var(Z)) = 0.05
predicted_tau_naive = (3 * 0.15) / (3 ** 2)
assert abs(tau_naive - predicted_tau_naive) < 0.015, \
    f"naive slope ({tau_naive:.4f}) should match the analytical prediction ({predicted_tau_naive:.4f})"
assert tau_naive > 0.03, "naive slope should be a clear nonzero spurious effect, not noise around 0"

# 控制温度Z之后, 偏效应应该接近0(真实值 -- X对Y没有直接因果效应)
X_design = np.column_stack([np.ones(n), X, temperature])
beta_hat = np.linalg.solve(X_design.T @ X_design, X_design.T @ Y)
tau_adjusted = beta_hat[1]
assert abs(tau_adjusted) < 0.01, f"after controlling for temperature, the spurious effect should nearly vanish, got {tau_adjusted:.6f}"

print(f"naive 'ice cream -> drowning' slope = {tau_naive:.4f}  (analytically predicted spurious value = {predicted_tau_naive:.4f})")
print(f"slope after controlling for temperature = {tau_adjusted:.6f}  (should be ~0, the true causal effect)")
```

**面试怎么问+追问链**(真实性验证轴):
- Q:"两个业务指标的相关系数是0.85,产品经理据此提出'只要提升指标A,指标B就会跟着提升'的立项理由,你怎么评估这个论证?"
- 追问1:"光凭相关系数0.85,能不能判断这个论证站不站得住脚?"(不能——0.85只说明两者共同变化的程度很强,不包含任何方向信息,需要额外判断是否存在能同时驱动A和B的第三方变量,或者反过来是B驱动A)
- 深挖追问:"如果确实找到一个疑似共同原因的变量,怎么用数据验证'相关性主要是这个共同原因造成的'这个猜想?"(控制住这个疑似共同原因变量,重新计算A、B的偏相关或偏回归系数,如果偏相关/偏效应大幅缩小甚至消失,就是有力证据支持"共同原因假说";如果偏相关几乎不变,说明这个变量不是主要的混淆来源,需要继续寻找或重新考虑A、B之间可能确实存在直接因果关系)

**常见坑:**
- 满足于"控制了一个疑似混淆变量,偏相关变小了"就直接下定论——就像知识点3强调的,永远无法保证控制的变量列全了所有混淆因素,变小不等于消失,仍需谨慎表述结论的确定性程度。
- 反过来滥用"相关不是因果"这句话去否定所有基于观察性数据的结论,不区分"有明确合理机制且做过混淆变量控制的观察性证据"和"纯粹拍脑袋的相关性"这两种质量完全不同的证据。

---

## 6. 反事实推理直觉 —— 从"平均而言有没有用"到"这一次会怎样"

**定义与记号:** 反事实推理(counterfactual reasoning):在potential outcomes框架下,对某个具体观测到结果的个体i,问"如果ta当初被分配到另一种处理,结果会是什么"——即推断Y_i(1-Z_i)(实际未观测到的那个潜在结果)。个体层面反事实在没有额外结构假设时是根本不可识别的(知识点1"根本问题"的具体体现),但在简化假设下可以给出point estimate。

**一句话:** ATE回答的是"平均而言处理有没有用";反事实推理想回答一个更贴近直觉决策场景的问题——"这一个具体的人/这一次具体的情况,如果做了不同的选择,结果会不会不一样",这两个问题的难度完全不是一个量级。

**数学推导:** 在"同质处理效应"假设下(所有个体的τ_i都等于同一个常数τ,这是一个很强的简化假设),个体反事实可以直接点估计:

Ŷ_i(1-Z_i) = Y_i(Z_i) + (1-2Z_i)·τ̂

如果i在处理组(Z_i=1),估计其对照结果为Y_i-τ̂;如果i在对照组(Z_i=0),估计其处理结果为Y_i+τ̂——用ATE的估计值τ̂去"平移"这个个体的实际观测值。这个估计的关键局限在于:它假设τ_i对所有人都一样,完全没有捕捉个体处理效应的异质性(heterogeneous treatment effects,一个更前沿的因果推断子领域,超出本文范围,这里只点出这个局限存在)。

**底层机制/为什么这样设计:** 为什么"用ATE去平移个体观测值"是一个"能算但要谨慎"的估计,而不是精确答案?因为它悄悄地把"总体平均"套用到了"具体个体"身上——如果真实世界里处理效应对不同人差异很大(比如某种药物对一部分人有奇效、对另一部分人完全无效,平均下来是中等效应),用ATE去推断"这一个具体病人"吃药会怎样,可能和这个病人的真实反事实结果相去甚远,这是个体化决策(personalization)场景下因果推断面临的核心张力。

**AI研究/工程场景:** 推荐系统里的"这个用户如果没有看到这次推送,会不会依然完成购买"(用于计算增量价值/incremental value,避免把"反正都会发生的转化"错误地归功于推送),以及广告归因分析里的类似问题,都是反事实推理在工业界最直接的应用场景——这类分析通常需要更精细的异质处理效应建模(uplift modeling),而不是简单套用总体ATE,因为决策是针对每个具体用户是否要给予处理,不是关于总体平均效应的判断。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)
n = 3000

# 场景A: 真实处理效应对所有人同质(常数tau) -- "同质效应"假设确实成立
baseline = rng.normal(50, 10, n)
true_tau_homogeneous = 6.0
Y1_homo = baseline + true_tau_homogeneous + rng.normal(0, 3, n)
Y0_homo = baseline + rng.normal(0, 3, n)
Z = rng.integers(0, 2, n)
Y_obs_homo = np.where(Z == 1, Y1_homo, Y0_homo)
tau_hat_homo = Y_obs_homo[Z == 1].mean() - Y_obs_homo[Z == 0].mean()

# 用同质效应假设估计个体反事实: Y_hat_i(1-Z_i) = Y_i(Z_i) + (1-2*Z_i)*tau_hat
Y_cf_hat_homo = Y_obs_homo + (1 - 2 * Z) * tau_hat_homo
true_cf_homo = np.where(Z == 1, Y0_homo, Y1_homo)
error_homo = np.abs(Y_cf_hat_homo - true_cf_homo).mean()

# 场景B: 真实处理效应存在异质性(依赖个体特征feature) -- "同质效应"假设被违反
feature = rng.normal(0, 1, n)
true_tau_i = 6.0 + 8.0 * feature  # 效应因人而异, 不再是常数
Y1_hetero = baseline + true_tau_i + rng.normal(0, 3, n)
Y0_hetero = baseline + rng.normal(0, 3, n)
Y_obs_hetero = np.where(Z == 1, Y1_hetero, Y0_hetero)
tau_hat_hetero_ate = Y_obs_hetero[Z == 1].mean() - Y_obs_hetero[Z == 0].mean()  # 仍然只能估计出总体ATE

Y_cf_hat_hetero = Y_obs_hetero + (1 - 2 * Z) * tau_hat_hetero_ate
true_cf_hetero = np.where(Z == 1, Y0_hetero, Y1_hetero)
error_hetero = np.abs(Y_cf_hat_hetero - true_cf_hetero).mean()

# 核心断言: 同质效应假设下, 用ATE估计个体反事实的误差应该明显小于异质效应场景
assert error_homo < error_hetero, \
    f"counterfactual error should be smaller when the homogeneous-effect assumption actually holds: homo={error_homo:.4f} hetero={error_hetero:.4f}"
assert error_hetero > error_homo * 1.8, \
    f"heterogeneous-effect error should be substantially larger, not just marginally: homo={error_homo:.4f} hetero={error_hetero:.4f}"

print(f"true ATE (homogeneous scenario) = {true_tau_homogeneous}, estimated = {tau_hat_homo:.4f}")
print(f"counterfactual estimation error -- homogeneous effect (assumption holds): {error_homo:.4f}")
print(f"counterfactual estimation error -- heterogeneous effect (assumption violated): {error_hetero:.4f}")
```

**面试怎么问+追问链**(规模递增轴):
- Q:"我们能不能说'这个用户之所以流失,是因为没有收到这次促销推送'?"
- 追问1:"这句话背后依赖了什么因果假设?"(依赖对这个具体用户的反事实推理——需要知道"如果ta收到了推送"会不会留存,但这个反事实永远无法直接观测,只能借助总体或相似用户群体的处理效应做推断,推断的可靠性直接依赖处理效应异质性的大小)
- 深挖追问:"如果这个用户所在的细分人群整体的处理效应估计是+5%留存率提升,能不能直接说'这个用户的反事实结果就是ta本该被拯救的5%概率之一'?"(这是一个概念上合理但需要谨慎表述的推断——+5%是这个细分人群的平均效应,并不保证"这一个具体用户"确实属于被正向影响的那部分人,个体层面依然存在无法消除的不确定性,只能说"基于目前能获得的信息,这是对该用户反事实结果的最佳估计",不能说成确定性的因果断言)

**常见坑:**
- 把"总体ATE为正"直接套用到"这一个具体个体身上处理一定有正向效果"——总体平均为正,完全可能同时存在效应为负的子群体,个体反事实推理不能跳过异质性这一步。
- 忽视反事实估计依赖的假设(比如同质效应)有多强,把一个建立在简化假设上的点估计,当成和知识点1"上帝视角真实反事实"同等确定性的答案来使用。

---

下一篇:[09-observational-causal-methods.md](09-observational-causal-methods.md) —— 本文建立的potential outcomes框架是理论地基,09类正式展开"做不了RCT时"的具体补救方法:双重差分、工具变量、倾向得分匹配、断点回归。
