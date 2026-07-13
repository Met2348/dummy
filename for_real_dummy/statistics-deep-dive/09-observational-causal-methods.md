# 09 · 观察性因果推断方法深挖(Observational Causal Inference Methods)

> 总览见 [00-roadmap.md](00-roadmap.md)

08类建立了potential outcomes框架和"为什么RCT是金标准"这个基准。本文正式展开"做不了RCT时"业界最常用的四种正规补救方法——双重差分(DID)、工具变量(IV)、倾向得分匹配(PSM)、断点回归(RDD)。这四种方法有一个共同结构:都是用一个**无法被数据完全证明、只能靠业务理解去论证**的识别假设,换取"不需要随机化就能估计因果效应"的能力。知识点5给出"平行趋势假设"的具体数值检验方法,知识点6把前四种方法的失效边界放在一起对照展示,呼应08类"RCT为什么是金标准"这个结论——观察性方法都是不得已而为之的次优选择,选择时必须清楚自己在拿什么假设做交换。

**环境声明:** 全部代码在仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1)下真实跑通,随机抽样固定种子(`np.random.default_rng(42)`)。回归/逻辑回归全部手写正规方程/Newton-Raphson实现(仓库无statsmodels依赖,和05类环境声明一致)。

---

## 1. 双重差分法(DID)—— 比较变化量之差,不是水平之差

**定义与记号:** DID(Difference-in-Differences)比较处理组和对照组在处理前后的**变化量**之差,而不是处理后某一时刻的**水平**之差。设Y_it是个体i在时期t(t=0处理前,t=1处理后)的结果,估计量:

τ̂_DID = (Ȳ_treat,1 - Ȳ_treat,0) - (Ȳ_control,1 - Ȳ_control,0)

**一句话:** 单纯比较处理组处理后 vs 对照组处理后(横截面比较)会被两组"本来就不一样"的固定差异污染;DID通过"先各自算前后变化量,再比较变化量之差",把两组恒定不变的固定差异自动减掉了,只留下处理带来的额外变化。

**数学推导:** 设Y_it=α_i+λ_t+τ·Treat_i·Post_t+ε_it,α_i是个体固定效应(不随时间变化的个体特质)、λ_t是时间固定效应(不随个体变化的共同冲击)。则:

(Ȳ_treat,1-Ȳ_treat,0) = (α_treat+λ_1+τ)-(α_treat+λ_0) = λ_1-λ_0+τ
(Ȳ_control,1-Ȳ_control,0) = (α_control+λ_1)-(α_control+λ_0) = λ_1-λ_0

两者相减:τ̂_DID=τ——个体固定效应α_i和时间固定效应λ_t都被精确抵消,只剩真实处理效应τ。这个抵消依赖一个关键假设:如果没有处理,两组的时间趋势应该是**平行**的(parallel trends assumption,知识点5要专门数值检验)——如果不成立,时间效应对两组不对称,抵消就不完整。

**底层机制/为什么这样设计:** DID的精妙之处在于,它不要求"处理组和对照组在处理前的水平完全相同"(那是08类RCT才有的强要求),只要求"如果没有处理,两组的变化趋势会保持一致"——这是弱得多的假设,因为很多混淆变量即使让两组绝对水平不同,也不影响它们随时间变化的相对趋势。

**AI研究/工程场景:** 政策评估(某个城市率先实施新政策,其他城市作为对照组)、产品分阶段上线(某个市场先上线新功能,其他市场作为对照)都是DID的经典应用场景——当无法做真正的随机化实验,但存在"处理时点明确、有合理对照组"这个结构时,DID是观察性因果推断里最常用、最容易向业务方解释的方法之一。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)
n_per_group = 500
true_tau = 4.0

# 场景A: 平行趋势成立 -- 处理组/对照组处理前后的"自然趋势"完全相同, 只是水平不同
group_effect_treat, group_effect_control = 10.0, 3.0  # 固定的组间水平差异(不影响DID)
time_trend = 5.0  # 共同的时间趋势(两组共享, 不影响DID)

Y_treat_0 = group_effect_treat + rng.normal(0, 2, n_per_group)
Y_treat_1 = group_effect_treat + time_trend + true_tau + rng.normal(0, 2, n_per_group)
Y_control_0 = group_effect_control + rng.normal(0, 2, n_per_group)
Y_control_1 = group_effect_control + time_trend + rng.normal(0, 2, n_per_group)

did_estimate = (Y_treat_1.mean() - Y_treat_0.mean()) - (Y_control_1.mean() - Y_control_0.mean())
assert abs(did_estimate - true_tau) < 0.5, f"DID should closely recover true_tau when parallel trends holds, got {did_estimate:.4f}"

# 朴素横截面比较(只看处理后水平): 会被组间固定差异污染
naive_cross_section = Y_treat_1.mean() - Y_control_1.mean()
assert abs(naive_cross_section - true_tau) > 3.0, \
    f"naive cross-sectional comparison ({naive_cross_section:.4f}) should be badly contaminated by the fixed group gap"

# 场景B: 平行趋势违反 -- 处理组处理前趋势本身就和对照组不同(额外的反事实趋势差异diff_trend)
diff_trend = 6.0
Y_treat_1_violated = group_effect_treat + time_trend + diff_trend + true_tau + rng.normal(0, 2, n_per_group)
did_estimate_violated = (Y_treat_1_violated.mean() - Y_treat_0.mean()) - (Y_control_1.mean() - Y_control_0.mean())

# 核心断言: 平行趋势违反时, DID估计量的偏差应该约等于diff_trend
assert abs(did_estimate_violated - (true_tau + diff_trend)) < 0.5, \
    f"violated-parallel-trends bias should match diff_trend, got {did_estimate_violated:.4f} vs expected ~{true_tau + diff_trend}"
assert abs(did_estimate_violated - true_tau) > 4.0, "violated-parallel-trends DID should be far from the true effect"

print(f"parallel trends holds: DID estimate = {did_estimate:.4f}  (true tau = {true_tau})")
print(f"naive cross-section (contaminated by fixed group gap) = {naive_cross_section:.4f}")
print(f"parallel trends violated: DID estimate = {did_estimate_violated:.4f}  (bias ~= diff_trend = {diff_trend})")
```

**面试怎么问+追问链**(决策依据追问轴):
- Q:"为什么不直接比较处理组和对照组在政策实施后的水平差异,而要用DID这种更复杂的方法?"
- 追问1:"DID相对简单横截面比较到底多做对了什么?"(横截面比较没有办法区分"政策效果"和"两组本来就存在的固定差异",DID通过差分掉处理前的水平,把这部分固定差异消掉了)
- 深挖追问:"DID的核心假设'平行趋势'能不能被数据验证?"(不能被完全验证——平行趋势假设的是"处理后,如果没有处理,两组趋势会怎样",这本身是反事实,永远无法直接观测;能做的只是检验"处理前"两组趋势是否平行(知识点5),间接支持这个假设的合理性,但处理前平行不能逻辑上保证处理后也会平行)

**常见坑:**
- 只有两个时间点(处理前后各一个)时,没有能力检验平行趋势假设(需要至少处理前的多期数据才能观察趋势是否平行),此时平行趋势完全是一个无法验证的假设。
- 处理组和对照组的处理时点不一致(交错处理,staggered treatment timing)时,标准两期DID公式不再适用,简单套用会引入负权重偏差,是近年计量经济学文献重点纠正的常见误用。

---

## 2. 工具变量法(IV)—— 借一个"干净"的外部扰动切开混淆

**定义与记号:** 工具变量(instrument)Z'需满足:①**相关性**:Z'和处理变量X相关(Cov(Z',X)≠0);②**排他性约束**(exclusion restriction):Z'只能通过X影响Y,没有其他路径直接影响Y;③**外生性**:Z'不受任何混淆变量影响。两阶段最小二乘(2SLS):第一阶段用Z'回归X得到X̂;第二阶段用X̂回归Y得到τ̂_IV。

**一句话:** 如果有一个既让X变化、又完全不通过其他渠道影响Y的"外部扰动"Z',那么只需要看"Z'变化引起的X变化"对应"Z'变化引起的Y变化"的比例,就能算出X对Y的真实因果效应,完全不需要假设"控制了所有混淆变量"。

**数学推导:** 真实模型 Y=τX+βC+ε(C是未观测混淆变量)。工具变量Z'满足Cov(Z',C)=0、Cov(Z',ε)=0、Cov(Z',X)≠0。两边同时和Z'求协方差:

Cov(Z',Y) = τ·Cov(Z',X) + β·Cov(Z',C) + Cov(Z',ε) = τ·Cov(Z',X)

所以 τ_IV=Cov(Z',Y)/Cov(Z',X)——这是简单IV情形的解析形式,2SLS是这个思想在过度识别情形下的一般化:第一阶段回归X=γZ'+ν得到X̂=γ̂Z',第二阶段回归Y=τX̂+η得到τ̂,在恰好识别(工具变量数=内生变量数)情形下两者完全等价。

**底层机制/为什么这样设计:** 为什么"先用Z'预测出X̂,再用X̂而不是原始X去解释Y"能消除混淆偏差?因为X̂=γ̂Z'这部分变异完全来自Z',而Z'根据假设和混淆变量C无关——X̂里"纯净地不含被C污染的那部分信息",用X̂去解释Y,估计出来的效应就不会再混入C的干扰。这个操作本质上是"只保留X中被一个已知外生来源驱动的那部分变异,丢弃其他来源(包括被混淆变量驱动)的变异"。

**AI研究/工程场景:** 经典IV应用是"用自然实验作为工具变量"——研究"广告曝光对销量的因果效应"时,直接回归会被"公司在旺季本来就加大广告投放,而旺季销量本来就高"这类混淆污染,可以借助"竞争对手的广告投放中断"(和自身广告策略无关,但通过挤占媒体资源影响自身广告曝光)这类思路构造工具变量。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)
n = 5000
true_tau = 3.0
beta_c = 2.0  # 未观测混淆变量C对Y的效应

C = rng.normal(0, 1, n)              # 未观测混淆变量
Z_instrument = rng.normal(0, 1, n)   # 真正的工具变量: 只影响X, 和C无关, 不直接影响Y

X = 0.8 * Z_instrument + 1.5 * C + rng.normal(0, 1, n)
Y = true_tau * X + beta_c * C + rng.normal(0, 1, n)  # 工具变量没有直接路径到Y

# 朴素OLS: 应该因为C的混淆而有偏
tau_ols = np.cov(X, Y, ddof=0)[0, 1] / np.var(X)
assert abs(tau_ols - true_tau) > 0.5, f"naive OLS should be biased by unobserved confounding, got {tau_ols:.4f} vs true {true_tau}"

# 手写2SLS: 第一阶段用工具变量回归X, 第二阶段用X_hat回归Y
X_design_stage1 = np.column_stack([np.ones(n), Z_instrument])
gamma_hat = np.linalg.solve(X_design_stage1.T @ X_design_stage1, X_design_stage1.T @ X)
X_hat = X_design_stage1 @ gamma_hat

X_design_stage2 = np.column_stack([np.ones(n), X_hat])
beta_hat_stage2 = np.linalg.solve(X_design_stage2.T @ X_design_stage2, X_design_stage2.T @ Y)
tau_iv = beta_hat_stage2[1]

assert abs(tau_iv - true_tau) < 0.3, f"2SLS should closely recover the true tau, got {tau_iv:.4f} vs true {true_tau}"

# 简单IV公式交叉验证: tau_IV = Cov(Z,Y)/Cov(Z,X) (恰好识别情形下应与2SLS完全一致)
tau_iv_simple_formula = np.cov(Z_instrument, Y, ddof=0)[0, 1] / np.cov(Z_instrument, X, ddof=0)[0, 1]
assert abs(tau_iv - tau_iv_simple_formula) < 1e-6, "2SLS should exactly match the simple Cov(Z,Y)/Cov(Z,X) formula here"

print(f"true tau = {true_tau}")
print(f"naive OLS estimate (biased) = {tau_ols:.4f}")
print(f"2SLS IV estimate = {tau_iv:.4f}  (matches simple formula: {tau_iv_simple_formula:.4f})")
```

**面试怎么问+追问链**(方案批判迭代轴):
- Q:"你说用了工具变量法控制内生性,这个工具变量选得对不对,怎么验证?"
- 追问1:"'相关性'条件容易验证吗?"(相对容易——直接看第一阶段回归Z'对X的系数是否显著、F统计量是否够大(经验法则F>10),这是可以用数据直接检验的)
- 深挖追问:"那'排他性约束'呢?"(排他性约束本质上无法用数据完全证明,只能靠对业务机制的深入理解去论证,是IV方法里最容易被质疑、也最考验候选人对业务因果链条理解深度的地方——诚实的回答应该是详细讲清楚业务机制,而不是回避这个问题)

**常见坑:**
- 找到一个和X相关性很弱的"弱工具变量"(weak instrument)——第一阶段Cov(Z',X)接近0时,IV估计量的方差会急剧膨胀,置信区间可能宽到没有实际意义,是IV方法实践中最常见的技术性陷阱。
- 只验证了相关性条件就直接采用某个工具变量,没有认真论证排他性约束——这是IV方法里最容易被"偷懒"跳过、但恰恰最关键的假设。

---

## 3. 倾向得分匹配(PSM)—— 把高维匹配压缩成一维分数

**定义与记号:** 倾向得分(propensity score)e(X)=P(Z=1\|X):给定观测到的混淆变量X,个体接受处理的条件概率。Rosenbaum-Rubin定理:如果"给定所有混淆变量X,处理分配Z与潜在结果独立"(无混淆性/unconfoundedness假设)成立,那么"给定倾向得分e(X)这一个标量,Z与潜在结果也独立"——不需要在高维X空间里匹配,只需要在一维倾向得分上匹配处理组和对照组个体,就能达到和"完全匹配所有X"同样的去混淆效果。

**一句话:** 与其在很多个混淆变量维度上费力地给每个处理组个体找一个"各方面都像"的对照组个体(维度诅咒),不如先把"每个人有多可能被分到处理组"压缩成一个单一概率分数,再按这个分数配对——数学上被证明这样做和精确匹配所有维度的效果等价。

**数学推导:** 无混淆性假设:(Y(1),Y(0))⊥Z\|X。Rosenbaum-Rubin定理证明:若成立,则(Y(1),Y(0))⊥Z\|e(X)也成立(倾向得分是X的一个"平衡分数")。证明核心在于P(Z=1\|X,e(X))=P(Z=1\|e(X))=e(X)本身(倾向得分的定义决定了这一点),结合"给定X已满足无混淆性"的前提,可以推出"给定e(X)"也满足无混淆性——压缩到一维不损失匹配所需要的信息。

**底层机制/为什么这样设计:** 为什么"压缩成一维分数再匹配"不会损失信息?倾向得分e(X)已经是"给定X,有多大概率被分到处理组"这个信息的完整汇总——两个X完全不同、但e(X)恰好相同的个体,虽然具体特征不像,但在"被分配到处理组的可能性"这个决定性维度上是等价的,而无混淆性假设本质上只关心处理分配这一件事,所以只在这一个维度上匹配就足够,这是"维度诅咒"问题在因果推断匹配方法里的一个巧妙绕过。

**AI研究/工程场景:** 医疗健康数据分析、经济政策评估里PSM是最常用的观察性因果推断方法之一——当有大量观测到的用户/患者特征、但无法做随机化实验时,先用逻辑回归估计每个个体的倾向得分,再做最近邻匹配、构造"伪对照组",是business analytics里性价比很高的常规操作。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)
n = 2000
true_tau = 5.0

C1 = rng.normal(0, 1, n)  # 混淆变量1
C2 = rng.normal(0, 1, n)  # 混淆变量2
p_treat = 1 / (1 + np.exp(-(0.8 * C1 + 0.6 * C2)))
Z = rng.binomial(1, p_treat)
Y = true_tau * Z + 2.0 * C1 + 1.5 * C2 + rng.normal(0, 1, n)

naive_diff = Y[Z == 1].mean() - Y[Z == 0].mean()
assert abs(naive_diff - true_tau) > 1.5, f"naive diff should be visibly biased, got {naive_diff:.4f} vs true {true_tau}"

# 手写逻辑回归(Newton-Raphson/IRLS)估计倾向得分
X_design = np.column_stack([np.ones(n), C1, C2])
theta = np.zeros(3)
for _ in range(20):
    eta = X_design @ theta
    p = 1 / (1 + np.exp(-eta))
    W = p * (1 - p)
    grad = X_design.T @ (Z - p)
    H = -(X_design * W[:, None]).T @ X_design
    theta -= np.linalg.solve(H, grad)

propensity = 1 / (1 + np.exp(-(X_design @ theta)))

# 最近邻匹配: 对每个处理组个体, 在对照组里找倾向得分最接近的一个
treat_idx = np.where(Z == 1)[0]
control_idx = np.where(Z == 0)[0]
control_scores = propensity[control_idx]

matched_diffs = [
    Y[i] - Y[control_idx[np.argmin(np.abs(control_scores - propensity[i]))]]
    for i in treat_idx
]
psm_estimate = np.mean(matched_diffs)

assert abs(psm_estimate - true_tau) < 1.0, f"PSM estimate should be much closer to true tau, got {psm_estimate:.4f} vs true {true_tau}"
assert abs(psm_estimate - true_tau) < abs(naive_diff - true_tau) * 0.5, \
    f"PSM should cut the bias substantially: naive_bias={abs(naive_diff - true_tau):.4f} psm_bias={abs(psm_estimate - true_tau):.4f}"

print(f"true tau = {true_tau}")
print(f"naive diff (biased) = {naive_diff:.4f}")
print(f"PSM nearest-neighbor estimate = {psm_estimate:.4f}  (bias cut substantially)")
```

**面试怎么问+追问链**(工程约束递增轴):
- Q:"PSM听起来很好用,能不能替代随机化实验作为公司的标准做法?"
- 追问1:"PSM相对RCT,最本质的短板是什么?"(PSM的无混淆性假设永远无法被数据验证——它只能平衡你测量到的X,如果存在没有测量、但确实影响处理分配和结果的变量,PSM完全无能为力,这是它和RCT本质的差距)
- 深挖追问:"如果匹配后处理组和对照组在某些协变量上分布依然不平衡,该怎么办?"(需要做匹配质量诊断——比较匹配前后各协变量在两组间的标准化均值差,如果仍有明显不平衡,可能需要调整匹配方法(比如卡尺匹配限制配对距离),甚至说明这个混淆变量可能无法被现有数据充分平衡,需要谨慎解读结论)

**常见坑:**
- 匹配后不做协变量平衡性检验就直接报告效应估计——PSM的可信度完全建立在"匹配后两组协变量分布确实变得相似"这个前提上,跳过这一步验证等于假设了一个没有检验的前提。
- 把"倾向得分匹配消除了偏差"和"消除了所有偏差"混为一谈——PSM只能平衡观测到的协变量,对未观测混淆变量完全无能为力,这是所有基于无混淆性假设的方法共同的根本局限。

---

## 4. 断点回归(RDD)—— 阈值附近的胜负,近似是运气

**定义与记号:** 当处理分配由某个连续的"驱动变量"(running variable)R是否超过已知阈值c决定时,可以比较阈值附近"恰好没过线"和"恰好过线"的个体的结果差异,作为处理效应在阈值处的局部估计——因为在阈值附近,R稍微差一点点和稍微好一点点的个体在几乎所有其他方面都应该是相似的,近似构成"局部随机化"。

**一句话:** RDD的核心直觉是"阈值附近的胜负是接近随机的运气",两个R值只差0.001分的人,一个跨过了线一个没有,这一点点差距不太可能对应任何系统性的能力差异,所以可以把阈值两侧看作一个天然的、局部的随机对照实验。

**数学推导:** 设Y=f(R)+τ·1[R≥c]+ε,f(R)是R对Y的平滑"自然"影响,τ·1[R≥c]是处理效应。RDD估计量是阈值处的跳跃:τ̂_RDD=lim_{R→c⁺}E[Y\|R]-lim_{R→c⁻}E[Y\|R]。实践中用阈值两侧一个小带宽h内的数据分别做局部线性回归,外推到R=c处的截距,两个截距之差就是τ̂_RDD。这个估计量只识别**阈值处**的局部处理效应,如果处理效应本身随R变化(异质性),不能随意外推到远离阈值的个体。

**底层机制/为什么这样设计:** 为什么只信任阈值附近的数据,而不用全部数据?离阈值远的个体,R的取值本身可能和很多其他因素相关(比如R=能力测试分数,远高于阈值的人可能各方面能力都强),只有紧贴阈值的个体,f(R)这个"自然趋势"部分在两侧几乎连续、可以用局部线性外推抵消掉,剩下的跳跃才能干净地归因于处理本身——用"局部性"换取"可信度",代价是牺牲了对阈值以外个体的解释力。

**AI研究/工程场景:** 很多产品的分层运营策略天然构造了断点(活跃度评分超过某个阈值才能进入促活计划、模型评分超过阈值才触发人工审核),这些人为设定的阈值恰好为RDD提供了绝佳的自然实验场景——评估"进入促活计划是否真的提升了留存"时,与其直接比较计划内外用户(严重混淆),不如聚焦在评分刚好卡在阈值两侧的用户做RDD分析。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)
n = 4000
cutoff = 80.0
true_tau = 6.0

R = rng.uniform(60, 100, n)  # driving variable, 比如考试分数
f_R = 0.3 * (R - cutoff)     # 平滑的"自然"趋势(不含处理效应)
treated = (R >= cutoff).astype(float)
Y = 50 + f_R + true_tau * treated + rng.normal(0, 2, n)

# 朴素全量比较(不限制带宽): 会被f(R)本身的趋势污染
naive_diff = Y[R >= cutoff].mean() - Y[R < cutoff].mean()
assert abs(naive_diff - true_tau) > 2.0, f"naive full-sample comparison should be contaminated by the f(R) trend, got {naive_diff:.4f} vs true {true_tau}"

# 局部线性RDD: 只用阈值附近带宽h内的数据做线性拟合外推
bandwidth = 3.0
mask_left = (R >= cutoff - bandwidth) & (R < cutoff)
mask_right = (R >= cutoff) & (R < cutoff + bandwidth)

def local_linear_intercept(R_sub, Y_sub, cutoff):
    X_design = np.column_stack([np.ones(len(R_sub)), R_sub - cutoff])
    beta = np.linalg.solve(X_design.T @ X_design, X_design.T @ Y_sub)
    return beta[0]  # 截距就是外推到R=cutoff处的拟合值

intercept_left = local_linear_intercept(R[mask_left], Y[mask_left], cutoff)
intercept_right = local_linear_intercept(R[mask_right], Y[mask_right], cutoff)
rdd_estimate = intercept_right - intercept_left

assert abs(rdd_estimate - true_tau) < 1.0, f"local linear RDD should closely recover the true jump, got {rdd_estimate:.4f} vs true {true_tau}"
assert abs(rdd_estimate - true_tau) < abs(naive_diff - true_tau) * 0.5, \
    f"RDD should be much more accurate than naive full-sample comparison: naive_err={abs(naive_diff - true_tau):.4f} rdd_err={abs(rdd_estimate - true_tau):.4f}"

print(f"true tau (jump at cutoff) = {true_tau}")
print(f"naive full-sample comparison (contaminated by f(R) trend) = {naive_diff:.4f}")
print(f"local linear RDD estimate (bandwidth={bandwidth}) = {rdd_estimate:.4f}")
```

**面试怎么问+追问链**(方案批判迭代轴):
- Q:"评分超过80分的用户进入了促活计划,我们发现计划内用户留存率比计划外高很多,能直接归因于促活计划的效果吗?"
- 追问1:"这个比较的问题在哪?"(评分80分以上和以下的用户在很多方面本来就不一样(评分本身和留存相关),不限制范围的全量比较把"评分本身的自然影响"和"进入计划的处理效应"混在一起了)
- 深挖追问:"如果改成只比较79-81分区间的用户,这样够了吗?"(方向是对的,但"带宽"的选择本身是一个偏差-方差权衡:带宽太窄,数据量太少,估计噪声大;带宽太宽,又会重新引入f(R)本身趋势的污染,需要用数据驱动的带宽选择方法或做带宽敏感性分析,而不是随意选一个区间)

**常见坑:**
- 断点两侧样本量差异悬殊,或者存在"个体能够操纵自己刚好卡在阈值哪一侧"的现象(manipulation)——这会破坏"阈值附近接近随机"这个核心假设,是RDD方法在实践中最需要专门检验(密度检验,如McCrary test)的前提条件。
- 把RDD估计出的"阈值处局部效应"直接外推到"所有用户身上的平均效应"——RDD的可信度恰恰建立在"只在阈值附近做局部推断"这个限制上,外推到远离阈值的人群超出了这个方法本身能保证的范围。

---

## 5. 平行趋势假设的数值检验 —— 用事件研究法量化"处理前趋势是否一致"

**定义与记号:** 平行趋势假设是DID方法(知识点1)成立的核心前提,涉及反事实(处理组"如果没有处理"的趋势),无法直接验证,但可以通过检验**处理前**多期数据里两组趋势是否平行,作为这个假设合理性的间接支持证据。

**一句话:** 平行趋势假设检验回答不了"处理之后如果没处理会怎样"这个真正的问题(反事实,永远无法验证),只能回答弱得多、但至少可以验证的替代问题——"处理之前,这两组人的趋势本来就已经在朝着不同方向走了吗?"

**数学推导:** 构造"事件研究"(event study)设计:用处理前K期数据,拟合 Y_it=α_i+λ_t+Σ_{k=1}^{K}δ_k·Treat_i·1[t=t_0-k]+ε_it,每个δ_k代表"处理组相对对照组,在处理前第k期的额外趋势偏离"。如果平行趋势假设成立,所有δ_k理论上都应该接近0(标准误范围内波动)。检验方式:逐个检验δ_k是否显著偏离0,并观察δ_k是否随k呈现系统性模式(比如单调增长),而不是只看"处理前最后一期"这一个粗糙的数字。

**底层机制/为什么这样设计:** 为什么要看"处理前多期"而不是只看处理前一期?只比较处理前一期,即使两组水平相同,也无法排除"两组本来就在朝不同方向变化,只是恰好在临处理前那一刻数值接近"这种情况(两条曲线恰好交叉)——只有观察足够长的处理前窗口,才能有效识别出这种"趋势不同但某一时点恰好重合"的危险情形。

**AI研究/工程场景:** 严肃的政策评估报告/产品分阶段上线效果分析,几乎都会附上"事件研究图"(横轴是相对处理时点的时间、纵轴是δ_k的估计值和置信区间),让读者自己判断处理前的系数是否稳定地在0附近波动——这已经成为DID分析里事实上的标准披露要求。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)
n_per_group = 400
K = 5  # 处理前5期
group_gap = 8.0  # 固定组间水平差异(不影响delta_k, 每期都被差分掉)

# 场景A: 平行趋势成立 -- delta_k理论上都应该在0附近波动
deltas_true_zero = []
for k in range(K):
    common_trend_k = 2.0 * k  # 两组共享的时间趋势
    Y_treat_k = group_gap + common_trend_k + rng.normal(0, 1.5, n_per_group)
    Y_control_k = common_trend_k + rng.normal(0, 1.5, n_per_group)
    delta_k = (Y_treat_k.mean() - Y_control_k.mean()) - group_gap  # 减去固定水平差异, 只留"额外偏离"
    deltas_true_zero.append(delta_k)
deltas_true_zero = np.array(deltas_true_zero)

se_delta = 1.5 * np.sqrt(2 / n_per_group)  # 每期delta_k的近似标准误
assert np.all(np.abs(deltas_true_zero) < 4 * se_delta), \
    f"under true parallel trends, all delta_k should stay within a few SEs of 0: {deltas_true_zero}"

# 场景B: 平行趋势违反 -- 处理组处理前趋势里额外混入一个随k线性增长的偏离项
deltas_violated = []
for k in range(K):
    common_trend_k = 2.0 * k
    extra_drift_k = 1.2 * k  # 处理组独有的、随时间累积的额外趋势(平行趋势假设的违反)
    Y_treat_k = group_gap + common_trend_k + extra_drift_k + rng.normal(0, 1.5, n_per_group)
    Y_control_k = common_trend_k + rng.normal(0, 1.5, n_per_group)
    delta_k = (Y_treat_k.mean() - Y_control_k.mean()) - group_gap
    deltas_violated.append(delta_k)
deltas_violated = np.array(deltas_violated)

# 核心断言: 违反平行趋势时, delta_k应该呈现出随k单调增长的系统性模式, 且明显超出噪声范围
assert deltas_violated[-1] > deltas_violated[0] + 3 * se_delta, \
    f"violated parallel trends should show a clear systematic drift in delta_k: {deltas_violated}"
n_increasing = sum(deltas_violated[i] < deltas_violated[i + 1] for i in range(K - 1))
assert n_increasing >= K - 2, f"delta_k should show a mostly-monotonic increasing pattern under violation, got {deltas_violated}"

print(f"parallel trends holds: delta_k across {K} pre-periods = {[round(d, 3) for d in deltas_true_zero]}  (all near 0)")
print(f"parallel trends violated: delta_k across {K} pre-periods = {[round(d, 3) for d in deltas_violated]}  (systematic drift)")
```

**面试怎么问+追问链**(诊断真实数据新题型):
- Q:"给你处理前5期两组的δ_k估计值:-0.02, 0.01, -0.03, 0.02, -0.01(标准误约0.05),平行趋势假设站得住脚吗?"
- 追问1:"这组数字说明了什么?"(这5个δ_k都在0附近小幅波动,幅度明显小于标准误,没有随时间单调偏离0的系统性模式,支持平行趋势假设合理)
- 深挖追问:"如果换成这组数字:-0.15, -0.10, -0.06, -0.03, -0.01,你的判断会变吗?"(这组数字呈现出清晰的单调趋势(从-0.15逐渐收敛到接近0),暗示处理组相对对照组在处理前就存在系统性的趋势差异,即使最后一期恰好接近0,这种系统性模式也应该让人对平行趋势假设产生怀疑,不能只看最后一期数值)

**常见坑:**
- 只检验处理前"最后一期"两组水平是否接近,不看多期趋势的完整模式,漏掉"趋势本身不同、恰好某一刻数值重合"这种危险情形。
- 把"处理前平行趋势检验通过"等同于"处理后平行趋势假设一定成立"——处理前平行只是间接支持性证据,不是充分条件。

---

## 6. 观察性方法的适用边界 —— 每种方法都在拿一个假设换识别能力

**定义与记号:** 汇总本文四种观察性因果推断方法(DID、IV、PSM、RDD)各自的核心识别假设,以及假设违反时估计量失效的具体表现——不是新方法,而是把前四个知识点分别验证过的"正确使用"场景,反过来构造"错误使用"场景,让四种方法的适用边界具体化、可比较。

**一句话:** 面试官问"这个方法什么时候会失效"时,如果只能抽象地背出"假设不成立就会失效",说服力远不如直接展示"构造了一个具体场景,让这个假设明确被违反,亲眼看着估计量跑偏了多少"。

**数学推导/说明:** 四种方法的核心识别假设汇总:①DID依赖**平行趋势假设**——违反时偏差等于"两组反事实趋势之差"(知识点1已展示);②IV依赖**排他性约束**——违反时(工具变量本身直接泄漏影响Y),IV估计量会混入这条"直接路径"的虚假效应;③PSM依赖**无混淆性**(所有混淆变量都被观测并纳入模型)——遗漏关键混淆变量时,匹配后仍残留偏差,匹配质量"技术上做对了"但结论仍然错误;④RDD依赖**阈值处无操纵**(no manipulation)——个体能够精确操纵自己的R值卡在阈值某一侧时,阈值两侧不再是"局部随机",断点跳跃会同时包含真实处理效应和"能够/倾向于操纵的人本身就不一样"这两部分。

**底层机制/为什么这样设计:** 为什么把四种方法的失效场景放在同一个知识点里对照展示?放在一起对比才能看出更高层的规律——四种方法的失效模式表面不同,但都指向同一个结构:每种方法都是用一个"无法被数据完全验证的假设"去换取"不需要随机化就能识别因果效应"这个能力,假设的内容不同(平行趋势/排他性/无混淆性/无操纵),但"假设本身不可证伪,只能靠业务理解去论证或用间接证据支持"这个根本处境是所有观察性方法共享的,这也是08类"为什么RCT是金标准"这个结论在方法论层面的最终落脚点。

**AI研究/工程场景:** 在真实的因果推断项目汇报里,"我选择了什么方法"往往不是最容易被挑战的部分,"我为什么相信这个方法的识别假设在我的场景里成立"才是——一个经验丰富的分析师应该在动手之前就主动列出"如果我选的这个方法的核心假设不成立,我的结论会错成什么样",而不是等到被追问才第一次思考这个问题。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)
results = {}

# ---- DID: 平行趋势违反 ----
n = 500
true_tau_did = 4.0
diff_trend = 6.0
Y_t0 = 10 + rng.normal(0, 2, n)
Y_t1_ok = 10 + 5 + true_tau_did + rng.normal(0, 2, n)
Y_t1_bad = 10 + 5 + diff_trend + true_tau_did + rng.normal(0, 2, n)
Y_c0 = 3 + rng.normal(0, 2, n)
Y_c1 = 3 + 5 + rng.normal(0, 2, n)
did_ok = (Y_t1_ok.mean() - Y_t0.mean()) - (Y_c1.mean() - Y_c0.mean())
did_bad = (Y_t1_bad.mean() - Y_t0.mean()) - (Y_c1.mean() - Y_c0.mean())
results['DID'] = (abs(did_ok - true_tau_did), abs(did_bad - true_tau_did))

# ---- IV: 排他性约束违反(工具变量直接泄漏影响Y) ----
n = 3000
true_tau_iv = 3.0
C = rng.normal(0, 1, n)
Zinst = rng.normal(0, 1, n)
X = 0.8 * Zinst + 1.5 * C + rng.normal(0, 1, n)
Y_ok = true_tau_iv * X + 2.0 * C + rng.normal(0, 1, n)
Y_bad = true_tau_iv * X + 2.0 * C + 1.0 * Zinst + rng.normal(0, 1, n)  # 工具变量直接泄漏到Y

def iv_estimate(Z, X, Y):
    return np.cov(Z, Y, ddof=0)[0, 1] / np.cov(Z, X, ddof=0)[0, 1]

results['IV'] = (abs(iv_estimate(Zinst, X, Y_ok) - true_tau_iv), abs(iv_estimate(Zinst, X, Y_bad) - true_tau_iv))

# ---- PSM: 遗漏关键混淆变量C2 ----
n = 2000
true_tau_psm = 5.0
C1 = rng.normal(0, 1, n)
C2 = rng.normal(0, 1, n)
Zt = rng.binomial(1, 1 / (1 + np.exp(-(0.8 * C1 + 0.6 * C2))))
Yp = true_tau_psm * Zt + 2.0 * C1 + 1.5 * C2 + rng.normal(0, 1, n)

def psm_estimate(Z, Y, X_design):
    theta = np.zeros(X_design.shape[1])
    for _ in range(20):
        eta = X_design @ theta
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        grad = X_design.T @ (Z - p)
        H = -(X_design * W[:, None]).T @ X_design
        theta -= np.linalg.solve(H, grad)
    prop = 1 / (1 + np.exp(-(X_design @ theta)))
    treat_idx, control_idx = np.where(Z == 1)[0], np.where(Z == 0)[0]
    control_scores = prop[control_idx]
    diffs = [Y[i] - Y[control_idx[np.argmin(np.abs(control_scores - prop[i]))]] for i in treat_idx]
    return np.mean(diffs)

psm_ok = psm_estimate(Zt, Yp, np.column_stack([np.ones(n), C1, C2]))  # 完整控制C1, C2
psm_bad = psm_estimate(Zt, Yp, np.column_stack([np.ones(n), C1]))      # 遗漏C2
results['PSM'] = (abs(psm_ok - true_tau_psm), abs(psm_bad - true_tau_psm))

# ---- RDD: 阈值处存在操纵(manipulation) ----
n = 4000
cutoff, true_tau_rdd = 80.0, 6.0
R = rng.uniform(60, 100, n)
manipulators = (np.abs(R - cutoff) < 1.0) & (rng.random(n) < 0.5)
R_manip = np.where(manipulators, cutoff + 0.5, R)  # 一半"卡线"的人被人为推过阈值
treated_ok, treated_bad = (R >= cutoff).astype(float), (R_manip >= cutoff).astype(float)
ability_bonus = np.where(manipulators, 8.0, 0.0)  # 会操纵的人本身能力/资源更强, 独立于处理效应带来额外增益
Y_ok = 50 + 0.3 * (R - cutoff) + true_tau_rdd * treated_ok + rng.normal(0, 2, n)
Y_bad = 50 + 0.3 * (R_manip - cutoff) + true_tau_rdd * treated_bad + ability_bonus + rng.normal(0, 2, n)

def rdd_estimate(R, Y, cutoff, bandwidth=3.0):
    mask_l = (R >= cutoff - bandwidth) & (R < cutoff)
    mask_r = (R >= cutoff) & (R < cutoff + bandwidth)
    def intercept(Rs, Ys):
        Xd = np.column_stack([np.ones(len(Rs)), Rs - cutoff])
        return np.linalg.solve(Xd.T @ Xd, Xd.T @ Ys)[0]
    return intercept(R[mask_r], Y[mask_r]) - intercept(R[mask_l], Y[mask_l])

results['RDD'] = (
    abs(rdd_estimate(R, Y_ok, cutoff) - true_tau_rdd),
    abs(rdd_estimate(R_manip, Y_bad, cutoff) - true_tau_rdd),
)

# 核心断言: 每种方法, "假设违反"场景的估计误差都应该远大于"假设成立"场景
for method, (err_ok, err_bad) in results.items():
    assert err_bad > err_ok * 2.5, \
        f"{method}: violated-assumption error ({err_bad:.4f}) should be far larger than holds error ({err_ok:.4f})"

print("method | error when assumption holds | error when assumption violated | ratio")
for method, (err_ok, err_bad) in results.items():
    print(f"{method:5s} | {err_ok:.4f} | {err_bad:.4f} | {err_bad / err_ok:.1f}x")
```

**面试怎么问+追问链**(方案批判迭代轴,汇总全板块):
- Q:"给你一个观察性因果推断的场景(处理不是随机分配的,只能用观察数据),你会怎么选方法?"
- 追问1:"假设你选择了PSM,面试官反问'你怎么知道无混淆性假设成立',你怎么办?"(诚实地承认这个假设本质上无法被数据完全验证,但可以从两个角度增强可信度:①尽量详尽地列出所有理论上可能影响处理分配的变量并纳入模型;②做敏感性分析,量化"如果存在一个未观测混淆变量,它需要多强的效应才能推翻当前结论",这比假装无混淆性"显然成立"更专业)
- 深挖追问:"如果同一个因果问题,DID和PSM给出了不同的效应估计,该信哪个?"(不应该简单选一个"更可信"的,而应该回头检查两种方法各自的识别假设在这个具体场景下哪个更站得住脚——如果处理前趋势数据显示平行趋势明显不成立,DID的结果应该被打折扣;如果有理由怀疑存在未观测混淆变量,PSM的结果应该被打折扣;两种方法结论一致时互相印证,提升整体可信度)

**常见坑:**
- 选定一种方法后,把这个方法的核心假设当作"默认成立"而不主动去检验或论证——每种方法都有代价,不主动展示"认真考虑过这个方法可能在哪里失效"本身就是分析质量不足的信号。
- 认为"用了更复杂的方法(比如IV/RDD)"自动比"更简单的方法(比如朴素匹配)"更可信——复杂方法的识别假设往往更精巧、也更容易在不知不觉中被违反(比如排他性约束几乎无法用数据验证),方法复杂度和结论可信度不是单调递增关系,取决于假设在具体场景下是否真的成立。

---

下一篇:[10-real-world-traps.md](10-real-world-traps.md) —— 板块II收官,汇总Simpson悖论、选择偏差、幸存者偏差等真实分析陷阱案例集,并给出"诊断真实数据"新题型的旗舰示范。
