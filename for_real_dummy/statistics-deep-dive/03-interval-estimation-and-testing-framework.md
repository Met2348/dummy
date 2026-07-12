# 03 · 区间估计与假设检验框架深挖(Interval Estimation & Hypothesis Testing Framework)

> 总览见 [00-roadmap.md](00-roadmap.md)

02类讲的是"给一个点估计",本文讲两件事:第一,给一个点估计配一个"这个估计有多不确定"的区间(置信区间);第二,给一个二元判断("这个效应是不是真的存在")配一套完整的决策框架(假设检验)。这是整个统计系列里被面试问得最多、也最容易被问穿"背了公式但不懂原理"的一篇。

**环境声明:** 全部代码在仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1)下真实跑通,随机抽样固定种子(`np.random.default_rng(42)`)。

---

## 1. 置信区间的正确构造与解读 —— "95%概率"到底指什么在随机

**定义与记号:** 参数θ的95%置信区间(CI)是一个随机区间 [L(X), U(X)](L、U是数据X的函数),满足 P(L(X)≤θ≤U(X))=0.95。σ已知时,均值μ的CI:X̄ ± z_{0.025}·σ/√n;σ未知时用t分布:X̄ ± t_{0.025,n-1}·S/√n。

**一句话:** 95%指的是**构造区间这个过程**的可靠度(重复很多次,大约95%次构造出的区间会盖住真值),不是"这一个已经算出来的区间有95%概率盖住真值"——θ是固定常数,已经算出的区间要么盖住要么没盖住,没有概率可言。

**数学推导:** σ已知情形下,由(X̄-μ)/(σ/√n) ~ N(0,1)(01类CLT + 02类知识点5的方差公式):

P(-z_{0.025} ≤ (X̄-μ)/(σ/√n) ≤ z_{0.025}) = 0.95

对不等式做代数变形(把μ解出来,注意不等号方向在μ两边移项时会翻转):

-z_{0.025}·σ/√n ≤ X̄-μ ≤ z_{0.025}·σ/√n
X̄ - z_{0.025}·σ/√n ≤ μ ≤ X̄ + z_{0.025}·σ/√n

这就是CI公式的来源——**随机的是X̄(以及由它构造出的区间边界),μ是常数**,这也是"95%概率"这个说法必须挂在"区间"头上而不是"μ"头上的数学原因。

**底层机制/为什么这样设计:** CI的构造本质是把"点估计的抽样分布"(02类知识点7,MLE渐进正态性保证了这一点对几乎任何参数都成立)反过来用——已知X̄以μ为中心、以σ/√n为标准差近似正态分布,就能从"X̄落在以μ为中心的某个范围内的概率是95%"反推出"以X̄为中心构造的对称区间,有95%概率会盖住μ",这两个命题在数学上是等价的(对称区间的对称性保证了这个互换)。

**AI研究/工程场景:** 报告一个模型评测指标(比如准确率72.3%)时,不给置信区间等于隐藏了"这个数字本身有多大不确定性"这个关键信息——评测集只有200条数据时,72.3%的置信区间可能宽达±6个百分点,和另一个模型的68%完全可能"没有统计显著差异",这正是14类"模型评测统计"要专门处理的问题。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
true_mu, sigma = 50.0, 10.0
n = 30
n_trials = 5000
z_critical = stats.norm.ppf(0.975)  # 双侧95% CI

covered = 0
for _ in range(n_trials):
    sample = rng.normal(true_mu, sigma, n)
    x_bar = sample.mean()
    half_width = z_critical * sigma / np.sqrt(n)
    lower, upper = x_bar - half_width, x_bar + half_width
    if lower <= true_mu <= upper:
        covered += 1

coverage_rate = covered / n_trials

# 核心断言(coverage probability实验): 重复构造区间, 约95%的区间应该真的覆盖了真值
assert 0.93 < coverage_rate < 0.97, f"coverage rate {coverage_rate:.4f} should be close to 0.95"

print(f"observed coverage rate over {n_trials} trials: {coverage_rate:.4f} (target: 0.95)")
```

**面试怎么问+追问链**(真实性验证轴):
- Q:"你算出一个95% CI是[68%, 76%],能不能说'真实准确率有95%概率落在这个区间里'?"
- 追问1:"这句话错在哪?"(θ是固定常数,不是随机变量;95%描述的是区间构造过程的长期性质,不是这一个具体区间的概率——候选人如果只会背"这句话是错的"但说不出为什么,说明没真正理解频率派CI的语义)
- 深挖追问:"那贝叶斯的可信区间(credible interval)是不是就能说'95%概率'?"(能——贝叶斯可信区间是对参数的后验分布积分构造的,参数在贝叶斯框架下本身被当成随机变量,"95%概率落在区间内"这句话在贝叶斯框架下是成立的;这里为13类"可信区间vs置信区间"埋伏笔)

**常见坑:**
- 用"95%概率覆盖真值"描述已经算出的具体区间(上面已详细展开)。
- 忽视σ未知时必须用t分布而不是正态分布(t分布因为要多估计一个σ带来额外不确定性,尾部比正态分布更厚,n小时这个差异很明显,n大时t分布趋近正态)。

---

## 2. Neyman-Pearson引理与似然比检验 —— 给定显著性水平下功效最大的检验

**定义与记号:** 对简单假设H0:θ=θ₀ vs H1:θ=θ₁,似然比 Λ(x)=L(θ₀\|x)/L(θ₁\|x)。Neyman-Pearson引理:在给定显著性水平α下,"当Λ(x)小于某阈值k时拒绝H0"这个检验,是所有水平为α的检验里功效(power)最大的(UMP,uniformly most powerful)。

**一句话:** 不是所有"看起来合理"的判断规则功效都一样——NP引理告诉你,只要把全部数据的信息浓缩进似然比这一个统计量里做判断,就能拿到理论上最强的检验力,用不完整信息(比如只看一部分数据)做判断永远更弱。

**数学推导:** 对正态分布均值检验(σ已知)H0:μ=μ₀ vs H1:μ=μ₁(μ₁>μ₀),似然比 Λ = exp(-Σ(xᵢ-μ₀)²/(2σ²)) / exp(-Σ(xᵢ-μ₁)²/(2σ²))。取对数化简:log Λ = [Σ(xᵢ-μ₁)² - Σ(xᵢ-μ₀)²] / (2σ²),展开后可以化简成关于 x̄ 的线性函数——这说明**似然比检验在正态均值检验这个具体问题上等价于直接对样本均值x̄设一个阈值做判断**,不需要真的去算复杂的似然比,这是NP引理在这个具体例子上"落地"成一个简单可操作规则的原因。

**底层机制/为什么这样设计:** NP引理证明的核心思路(不展开完整证明)是反证法:假设存在另一个水平为α的检验功效比似然比检验更高,可以构造出一个矛盾(似然比检验在其拒绝域上已经把"单位显著性水平能换来的功效"最大化了,任何偏离这个拒绝域的检验,要么在不该拒绝的地方多拒绝了(超出α预算),要么在该拒绝的地方少拒绝了(浪费了α预算))。这也是为什么"用全部数据的似然比信息"比"只用部分数据(比如只看一个样本点)"更优——扔掉数据等于扔掉信息,信息更少的检验不可能功效更高。

**AI研究/工程场景:** 模型选择/异常检测里的似然比检验思想(比较"数据来自正常分布"和"数据来自异常分布"两个似然),是很多异常检测算法的理论基础;A/B测试的最优检验设计,底层也是在寻找NP意义下功效最大的判定规则。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
mu0, mu1, sigma = 0.0, 0.5, 1.0  # H0: mu=0 vs H1: mu=0.5 (effect size 0.5)
n = 25
alpha = 0.05
n_trials = 20_000

z_crit = stats.norm.ppf(1 - alpha)  # 单侧检验临界值

# 检验A(似然比检验, 在正态均值问题上等价于对样本均值设阈值): 拒绝H0当 x_bar > z_crit * sigma/sqrt(n)
threshold_lrt = z_crit * sigma / np.sqrt(n)

# 检验B(次优检验, 只用第一个观测值判断, 同样水平alpha): 拒绝H0当 x_1 > z_crit * sigma
threshold_naive = z_crit * sigma

rejects_lrt = 0
rejects_naive = 0
for _ in range(n_trials):
    sample = rng.normal(mu1, sigma, n)  # 真实来自H1(mu=0.5), 测两个检验各自的功效
    if sample.mean() > threshold_lrt:
        rejects_lrt += 1
    if sample[0] > threshold_naive:
        rejects_naive += 1

power_lrt = rejects_lrt / n_trials
power_naive = rejects_naive / n_trials

# 核心断言: 用全部样本的似然比检验(等价于样本均值检验)功效显著高于只用一个观测值的次优检验
assert power_lrt > power_naive * 3, f"LRT power {power_lrt:.4f} should far exceed naive power {power_naive:.4f}"
assert power_lrt > 0.7, f"LRT power should be reasonably high, got {power_lrt:.4f}"

print(f"power(likelihood-ratio test, uses all n={n} samples) = {power_lrt:.4f}")
print(f"power(naive test, uses only 1 sample)                = {power_naive:.4f}")
```

**面试怎么问+追问链**(方案批判迭代轴):
- Q:"两个检验方法都能把Type I错误率控制在5%,你怎么选?"
- 追问1:"只看α一样是不是就够了?"(不够——α只控制假阳性率,真正决定检验"好不好用"的是给定α下的功效,α相同、功效不同的两个检验有本质区别)
- 深挖追问:"如果两个假设都不是简单假设(比如H1:μ>0而不是μ=0.5这种确定值),NP引理还直接适用吗?"(NP引理严格来说是针对简单vs简单假设证明的;复合假设情形需要更强的条件(如单调似然比性质MLR)才能保证存在UMP检验,不是所有复合假设检验问题都有UMP检验存在——这是"检验优化理论"更深一层的内容,面试里能说出这个边界条件的候选人体现出真正理解而不是背答案)

**常见坑:**
- 把"功效最大"理解成"检验永远正确"——UMP检验只是在给定α约束下功效最大,依然存在β(Type II错误)风险,不是"完美"检验。
- 忽视NP引理是针对**简单假设**证明的,盲目套用到复合假设场景而不检查适用条件。

---

## 3. Type I/II错误与检验力(power)—— 两类错误此消彼长的权衡

**定义与记号:** α=P(拒绝H0\|H0为真)(假阳性率,Type I错误率);β=P(不拒绝H0\|H1为真)(假阴性率,Type II错误率);功效(power)=1-β=P(拒绝H0\|H1为真)。

**一句话:** 固定样本量的前提下,把α调小(更保守地拒绝H0)会让β变大(更容易漏掉真实存在的效应)——两类错误没法同时任意压低,唯一能同时压低两者的办法是增大样本量。

**数学推导:** 对正态均值单侧检验(σ已知,H0:μ=μ₀ vs H1:μ=μ₁>μ₀),拒绝域是 X̄ > μ₀+z_α·σ/√n。功效:

power = P(X̄ > μ₀+z_α·σ/√n \| μ=μ₁) = P((X̄-μ₁)/(σ/√n) > z_α - (μ₁-μ₀)/(σ/√n) \| μ=μ₁) = 1 - Φ(z_α - δ√n/σ)

其中δ=μ₁-μ₀是效应量的绝对大小。这个公式清楚展示三件事同时决定功效:样本量n(√n项)、效应量δ、以及α(通过z_α)——这是06类"样本量计算"要反过来用的核心公式(给定想要的power反解n)。

**底层机制/为什么这样设计:** α和β的权衡本质上是拒绝域边界移动的直接后果——把拒绝域的边界往右移(α变小,更保守),H1为真时样本落在拒绝域外的概率(β)自然增大;两者共享同一个决策边界,不可能独立地同时压低,除非改变决策边界之外的东西——也就是让抽样分布本身更"尖"(方差更小),而方差更小唯一可控的杠杆是增大样本量n。

**AI研究/工程场景:** 模型上线决策里,Type I错误(把没有真实提升的模型误判成"显著更好"从而错误上线)和Type II错误(把真实有提升的模型误判成"没有显著差异"从而错失机会)的代价往往不对称——误上线一个有害改动可能造成用户体验损失,错过一个真实有效的改进则是机会成本,不同业务场景对这两种错误的容忍度不同,这决定了α该设多严格。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
mu0 = 0.0
sigma = 1.0
n = 40
alpha = 0.05
delta = 0.4  # 真实效应量 mu1 - mu0
mu1 = mu0 + delta

z_alpha = stats.norm.ppf(1 - alpha)
threshold = mu0 + z_alpha * sigma / np.sqrt(n)

# 理论功效公式
theoretical_power = 1 - stats.norm.cdf(z_alpha - delta * np.sqrt(n) / sigma)

# 数值模拟功效
n_trials = 20_000
rejections = sum(1 for _ in range(n_trials) if rng.normal(mu1, sigma, n).mean() > threshold)
simulated_power = rejections / n_trials

assert abs(simulated_power - theoretical_power) < 0.02, \
    f"simulated power {simulated_power:.4f} should match theoretical power {theoretical_power:.4f}"

# 验证"此消彼长": alpha调得更严格(更小), 同样条件下功效应该下降
alpha_strict = 0.01
z_alpha_strict = stats.norm.ppf(1 - alpha_strict)
threshold_strict = mu0 + z_alpha_strict * sigma / np.sqrt(n)
rejections_strict = sum(1 for _ in range(n_trials) if rng.normal(mu1, sigma, n).mean() > threshold_strict)
power_strict = rejections_strict / n_trials

assert power_strict < simulated_power, f"stricter alpha should give lower power: {power_strict:.4f} vs {simulated_power:.4f}"

print(f"alpha=0.05: theoretical power={theoretical_power:.4f}  simulated power={simulated_power:.4f}")
print(f"alpha=0.01: power={power_strict:.4f} (lower, as expected)")
```

**面试怎么问+追问链**(工程约束递增轴):
- Q:"要把Type I错误率和Type II错误率都降低,可以怎么做?"
- 追问1:"如果样本量已经是业务能接受的上限,还能做什么?"(增大效应量本身不受你控制,但可以想办法降低数据本身的方差——比如06类的CUPED方差削减技术,方差变小意味着同样样本量下抽样分布更"尖",两类错误同时受益)
- 深挖追问:"如果这是一个持续进行的在线监控系统,不是一次性检验,'固定样本量'这个前提还成立吗?"(不成立——连续监控涉及"什么时候停止观测"的问题,06类"窥探问题"、07类"序贯检验"会专门处理这种前提被打破后带来的新问题)

**常见坑:**
- 认为"减小α"是无成本的更严格标准——减小α在样本量不变的前提下必然以增大β为代价,不存在"免费的更严格"。
- 混淆β和"H1为真的概率"——β是"H1为真时未能拒绝H0"这个条件概率,不是"H1本身为真的概率"(那是贝叶斯框架下的概念)。

---

## 4. p值真实含义与常见误用 —— p值不是"原假设为真的概率"

**定义与记号:** p值 = P(观测到当前检验统计量或比它更极端的值 \| H0为真)。**是一个条件概率,条件是"H0为真",不是"H0为真"这件事本身的概率。**

**一句话:** p值回答的是"如果原假设真的成立,像我现在这样(或更极端)的数据出现的概率有多大"——概率描述的是数据,不是假设本身;p很小说明"这样的数据在H0为真的世界里很稀奇",不是"H0为真的可能性很小"。

**数学推导/性质:** 一个重要且经常被忽视的性质:**在H0严格为真的前提下,p值本身服从Uniform(0,1)分布**——这不是巧合,是p值定义(累积分布函数在检验统计量处的取值,做了方向调整)的直接数学后果:如果T是连续型检验统计量,F是T在H0下的累积分布函数,p=1-F(T)(单侧情形),而F(T)本身根据概率积分变换(probability integral transform)服从Uniform(0,1)——所以p=1-F(T)也服从Uniform(0,1)。这个性质解释了为什么"H0为真时,p<0.05发生的概率恰好是5%"——不是巧合,是均匀分布的直接定义。

**底层机制/为什么这样设计:** 正是因为p值在H0为真时服从均匀分布,才能把α(显著性水平)直接理解成"愿意承受的假阳性率"——如果反复做很多次"H0确实为真"的检验,大约α比例的检验会因为随机噪声偶然给出p<α的结果,这是抽样随机性的必然产物,不是检验方法出了问题。

**AI研究/工程场景:** "在验证集上试了100个超参数组合,挑出p值最小的那个模型"这个操作,即使每个模型本身都没有真实提升,由于p值在H0下均匀分布,100次尝试里很大概率至少出现一次p<0.05的"虚假发现"——这正是05类多重检验、14类"刷榜陷阱"要专门量化和纠正的问题。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

# 验证: H0严格为真时(两组数据来自完全相同的分布), p值服从Uniform(0,1)
n_experiments = 5000
p_values = np.empty(n_experiments)
for i in range(n_experiments):
    group_a = rng.normal(0, 1, 30)
    group_b = rng.normal(0, 1, 30)  # 和group_a同分布, H0真实成立
    _, p_values[i] = stats.ttest_ind(group_a, group_b)

# 核心断言1: p值的经验分布应该接近Uniform(0,1) --- 用KS检验对比均匀分布, 而不是只看均值
ks_stat, ks_p = stats.kstest(p_values, "uniform")
assert ks_stat < 0.03, f"p-values under true H0 should look uniform, got KS stat={ks_stat:.4f}"

# 核心断言2: p<0.05这个"意外发现"在H0为真时本来就该发生约5%, 不是什么稀奇的证据
false_positive_rate = np.mean(p_values < 0.05)
assert 0.03 < false_positive_rate < 0.07, f"false positive rate {false_positive_rate:.4f} should be close to 0.05"

print(f"p-value distribution under true H0: KS stat={ks_stat:.4f}, fraction < 0.05 = {false_positive_rate:.4f}")
```

**面试怎么问+追问链**(真实性验证轴):
- Q:"p=0.03,能不能说'原假设为真的概率只有3%'?"
- 追问1:"这句话为什么是错的,能举一个反例吗?"(p值的定义里"H0为真"是**条件**,不是被赋概率的对象;反例:即使H0大概率是假的(比如医学检验里绝大多数被测者确实没有患病,患病是小概率事件),对没患病的人做检验,依然有固定的假阳性率α,p值小不代表病人群体的先验概率发生了变化——这里在呼应11类"贝叶斯定理里的基础比率谬误")
- 深挖追问:"那p=0.03到底能不能证明'效应存在'?"(p值提供的是"如果没有效应,数据有多不寻常"这一个证据的强度,不是效应存在与否的最终判决——完整判断还需要结合效应量、检验力、先验合理性,这也是为什么现代统计学越来越强调不要把p<0.05当成唯一的决策开关)

**常见坑:**
- 把p值等同于"效应大小"或"效应的重要性"——p值同时受效应量和样本量影响,大样本下极小的、没有实际意义的效应也能产生极小的p值(下一个知识点会用具体数字展示)。
- "p值刚好卡在0.05附近就反复调整分析方法直到显著"(p-hacking)——这个行为的危害正是因为p值在H0下均匀分布这个性质:多次尝试不同分析方式等价于多次独立检验,累积假阳性率会远超单次声称的5%。

---

## 5. 效应量与"统计显著 ≠ 实际显著" —— 大样本能把任何微小差异"测显著"

**定义与记号:** 效应量(effect size)衡量差异的"大小",独立于样本量。最常用的Cohen's d = (μ₁-μ₀)/σ(用标准差把差异标准化,方便跨场景比较)。经验参照:d≈0.2小效应,d≈0.5中效应,d≈0.8大效应(领域相关,不是绝对标准)。

**一句话:** p值告诉你"这个差异是不是巧合",效应量告诉你"这个差异到底有多大"——一个响亮的p<0.001完全可能对应一个小到没有任何实际业务价值的效应量。

**数学推导:** 结合知识点3的功效公式 power = 1-Φ(z_α - δ√n/σ) 可以看出:δ√n/σ 这个乘积决定检验统计量的"非中心度",δ(效应量的绝对差)很小时,只要n足够大,δ√n仍然能变得很大——这解释了为什么样本量足够大时,任意小的效应量最终都能被"测出显著性",p值会随n增大而不断变小,即使δ本身固定不变。

**底层机制/为什么这样设计:** p值本质上混合了两个不同的信息源——"效应有多大"和"数据有多少"——这也是为什么单独报告p值会误导人:相同的p=0.001,可能来自n=30、d=0.8的场景(大效应、显著),也可能来自n=2,000,000、d=0.02的场景(微小效应、因为样本量巨大照样显著)。效应量把"效应有多大"这一部分单独剥离出来,不受样本量污染。

**AI研究/工程场景:** 大厂A/B测试样本量动辄百万级,几乎任何微小的UI改动都能测出"统计显著"的差异(哪怕效应量只有0.001),这也是为什么成熟的实验平台除了p值,一定要求同时报告效应量和置信区间——只报告"p<0.05,显著"而不报告效应量大小,是一个真实存在、且经常被面试官专门追问的分析陷阱。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

true_delta = 0.01  # 极小的真实效应, 几乎没有业务价值
sigma = 1.0
n_huge = 2_000_000  # 巨大样本量

group_a = rng.normal(0, sigma, n_huge)
group_b = rng.normal(true_delta, sigma, n_huge)

t_stat, p_value = stats.ttest_ind(group_b, group_a)
cohens_d = (group_b.mean() - group_a.mean()) / np.sqrt((group_a.var() + group_b.var()) / 2)

# 核心断言: p值高度显著, 但效应量微不足道(这就是"统计显著不等于实际显著"的具体数字证据)
assert p_value < 0.001, f"with huge n, p-value should be tiny, got {p_value:.6f}"
assert abs(cohens_d) < 0.05, f"but effect size should be negligible, got {cohens_d:.4f}"

# 对照组: 小样本+大效应量的情形, p值可能反而没有那么夸张(说明p值确实不直接反映效应大小)
n_small = 20
big_delta = 1.5
group_a_small = rng.normal(0, sigma, n_small)
group_b_small = rng.normal(big_delta, sigma, n_small)
_, p_small = stats.ttest_ind(group_b_small, group_a_small)
d_small = (group_b_small.mean() - group_a_small.mean()) / np.sqrt((group_a_small.var() + group_b_small.var()) / 2)

assert d_small > cohens_d * 10, f"small-sample large-effect Cohen's d ({d_small:.4f}) should far exceed huge-sample tiny-effect d ({cohens_d:.4f})"

print(f"huge n, tiny effect:  p={p_value:.6f}  Cohen's d={cohens_d:.4f}")
print(f"small n, large effect: p={p_small:.6f}  Cohen's d={d_small:.4f}")
```

**面试怎么问+追问链**(真实性验证轴 + 方案批判迭代轴):
- Q:"你们的A/B测试显示新版本转化率p<0.001,非常显著,建议全量上线,面试官该怎么追问?"
- 追问1:"效应量具体是多少?"(候选人需要能report出具体的绝对/相对提升数字,而不是只重复"p值很小很显著")
- 深挖追问:"如果效应量只有0.1个百分点,上线这个改动的工程成本/维护成本划算吗?"(这里把纯统计判断拉回到业务决策——统计显著只是"这个差异大概率不是噪声"的证据,要不要行动还要结合效应量的业务价值和改动成本一起权衡,这是真正的"实际显著性"判断)

**常见坑:**
- 只报告p值不报告效应量(和置信区间),让读者误以为"显著"等于"效果很大"。
- 效应量的经验参照标准(d≈0.2/0.5/0.8)被当成放之四海皆准的硬指标——不同领域"大效应"的标准差异很大(医学干预d=0.2可能已经意义重大,某些工程指标d=0.2可能毫无意义),要结合具体业务场景判断,不能脱离场景套用Cohen的经验分类。

---

## 6. 单侧检验 vs 双侧检验 —— 同一份数据,选择不同会得到不同结论

**定义与记号:** 双侧检验:H0:θ=θ₀ vs H1:θ≠θ₀,拒绝域在两端。单侧检验:H0:θ≤θ₀ vs H1:θ>θ₀(或反方向),拒绝域只在一端。同一个检验统计量,单侧p值通常是双侧p值的一半(对称分布情形下)。

**一句话:** 单侧检验用同样的α预算换来了单个方向上更强的检验力,但代价是完全放弃了检测"反方向效应"的能力——选单侧还是双侧必须在看到数据**之前**根据业务问题的性质决定,不能看到数据后为了让结果显著才选单侧。

**数学推导:** 双侧检验的拒绝域是\|Z\|>z_{α/2}(两端各占α/2);单侧检验(比如只关心θ>θ₀)的拒绝域是Z>z_α(全部α预算都放在一端)。因为z_α<z_{α/2}(比如α=0.05时,z_{0.05}=1.645<z_{0.025}=1.96),单侧检验的拒绝门槛更低,同样的观测值更容易落入拒绝域——这正是"单侧检验power更高"的数学来源,但前提是效应确实发生在你预设的那个方向上。

**底层机制/为什么这样设计:** 选择单侧还是双侧,本质是在问"如果效应发生在意料之外的反方向,我还关不关心/还要不要检测出来"——比如测试一个新药"是否比安慰剂效果更好",如果新药效果显著更差,业务上同样是重要发现(不能上线),这时候双侧检验才是诚实的选择;但如果问题结构上反方向根本不可能发生或完全不重要(比如"新缓存策略是否降低延迟",提高延迟这个方向本身就是不可接受、不需要额外统计确认的失败),单侧检验才合理。

**AI研究/工程场景:** 性能优化类实验("这个kernel融合是否让推理延迟变快")天然适合单侧检验(没人关心"是否变慢"这个方向的统计显著性,变慢就是失败,不需要用检验来确认);而"新模型是否比旧模型效果更好"这类可能双向变化、双向都有业务意义的场景,规范做法是双侧检验。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
n = 40

# 构造一个边界case: 效应存在但比较微弱, 双侧检验刚好卡在临界值附近
sample_a = rng.normal(0, 1, n)
sample_b = rng.normal(0.35, 1, n)  # 中等偏小的效应

t_stat, p_two_sided = stats.ttest_ind(sample_b, sample_a)
p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2

# 核心断言: 双侧p值应该恰好是单侧p值的2倍(t统计量为正时)
assert t_stat > 0, "constructed so that b > a on average"
assert abs(p_two_sided - 2 * p_one_sided) < 1e-9

# 构造一个具体的边界案例: 双侧显著性刚好跨过0.05但单侧显著性明显跨过0.05
# (说明存在"双侧不显著, 单侧显著"这类边界情形, 因此二者选择会真实改变结论)
alpha = 0.05
double_significant = p_two_sided < alpha
single_significant = p_one_sided < alpha
# 单侧显著是双侧显著的必要不充分条件(在效应方向正确时): 双侧显著则单侧必然显著
if double_significant:
    assert single_significant

print(f"t={t_stat:.4f}  two-sided p={p_two_sided:.4f}  one-sided p={p_one_sided:.4f}")
print(f"two-sided significant={double_significant}  one-sided significant={single_significant}")
```

**面试怎么问+追问链**(真实性验证轴):
- Q:"你的双侧检验p=0.08不显著,但改用单侧检验p=0.04显著了,能不能就用单侧的结论?"
- 追问1:"这个决定应该是看到p=0.08之后才做,还是应该在实验设计阶段就做?"(必须是实验设计阶段就确定——看到双侧不显著之后才"改成单侧"本质上是一种p-hacking,是在数据已知的情况下选择对自己有利的分析方式,这在方法论上是不诚实的)
- 深挖追问:"如果确实是设计阶段就该用单侧(比如性能优化场景),但最后观测到效应方向和预期相反(变慢了),这时候单侧检验会给出什么结论?"(单侧检验对"反方向"的效应完全不敏感,p值会很大、"不显著",但这时候真正应该关注的是"反方向的效应本身已经是个问题",单纯看单侧p值不显著会掩盖这个事实——这是单侧检验设计不当时的真实风险)

**常见坑:**
- 看到双侧检验不显著之后才决定改用单侧检验(上面已详细展开,这是最常见、最容易被面试官抓到的坑)。
- 认为单侧检验"更宽松所以更好"——单侧检验是用统计力量换取了对反方向效应的盲区,不是免费的更强大,选择必须基于问题本身的结构,不是基于哪个能让p值更小。

---

## 7. 参数假设检验的稳健性 —— 正态性假设违反时会发生什么

**定义与记号:** t检验等参数检验的推导依赖"数据(或至少样本均值)近似正态分布"这个假设(通过CLT保证,01类知识点4)。稳健性(robustness)问的是:当这个假设不完全成立时(数据严重偏态/有离群值),检验结果还可靠吗。

**一句话:** t检验对轻度偏离正态性通常相当稳健(尤其样本量较大时,CLT本身会"拉直"抽样分布),但严重偏态+小样本的组合会让参数检验的置信区间覆盖率明显偏离名义水平——这时候04类的自助法(bootstrap)是更可靠的替代方案。

**数学推导/说明:** 稳健性没有一个单一的解析公式,是一个需要**数值验证**的经验问题——这也是为什么这个知识点的"可运行例子"比前面几个更重要:直接构造一个已知严重偏态的分布(比如指数分布,偏度=2,01类已经推过),在小样本下比较"t检验声称的95% CI覆盖率"和"bootstrap方法的95% CI覆盖率"谁更接近真实的95%。

**底层机制/为什么这样设计:** t检验的CI公式X̄±t_{α/2,n-1}·S/√n,本质上假设了(X̄-μ)/(S/√n)服从t分布——这个假设的成立依赖X̄本身趋于正态(CLT),小样本+严重偏态时CLT还没有"生效",t分布的假设是不准确的近似,导致声称的95% CI实际覆盖率可能明显低于95%(通常是覆盖率不足,而不是过度覆盖——偏态分布的极端尾部会让区间"看起来窄了")。

**AI研究/工程场景:** 用户付费金额、请求延迟这类天然右偏的业务指标,小样本(比如新功能刚上线只有几十个早期用户数据)下直接套用t检验的置信区间,覆盖率可能明显不足,容易对"新功能是否真的提升了付费"给出过度自信的错误结论——这是bootstrap方法在实际业务分析中被广泛采用而不是纯粹的学术偏好的真实原因。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
n = 15  # 小样本, 让正态性假设的问题被放大
n_trials = 4000
n_bootstrap = 500

true_lambda = 1.0
true_mean = 1 / true_lambda  # 指数分布均值

def t_interval_covers(sample, true_value):
    x_bar, s = sample.mean(), sample.std(ddof=1)
    half_width = stats.t.ppf(0.975, df=len(sample) - 1) * s / np.sqrt(len(sample))
    return (x_bar - half_width) <= true_value <= (x_bar + half_width)

def bootstrap_interval_covers(sample, true_value, rng_local):
    boot_means = np.array([rng_local.choice(sample, size=len(sample), replace=True).mean() for _ in range(n_bootstrap)])
    lower, upper = np.percentile(boot_means, [2.5, 97.5])
    return lower <= true_value <= upper

t_covered = 0
boot_covered = 0
for _ in range(n_trials):
    sample = rng.exponential(true_mean, n)  # 严重右偏分布, 偏度=2 (01类已推过)
    if t_interval_covers(sample, true_mean):
        t_covered += 1
    if bootstrap_interval_covers(sample, true_mean, rng):
        boot_covered += 1

t_coverage = t_covered / n_trials
boot_coverage = boot_covered / n_trials

# 核心断言: 严重偏态+小样本下, t检验的名义95% CI真实覆盖率会明显低于95%
assert t_coverage < 0.94, f"t-interval coverage should fall notably short of 0.95 under skewed small-sample data, got {t_coverage:.4f}"

print(f"nominal 95% CI, n={n}, exponential (skewed) data:")
print(f"  t-interval coverage:        {t_coverage:.4f}")
print(f"  bootstrap-interval coverage: {boot_coverage:.4f}")
```

**面试怎么问+追问链**(方案批判迭代轴):
- Q:"付费金额数据严重右偏,新功能只上线了3天、只有15个付费用户,能直接用t检验给转化提升下结论吗?"
- 候选人若说"能,反正样本量大于30就能用CLT"→追问1:"n=15,连'大于30'这个经验法则都够不上,而且付费金额这种数据偏度远超正态,这时候t检验的置信区间覆盖率会怎样?"(会明显低于名义值,过度自信)
- 换方案后追问2:"改用bootstrap是不是就完全没有前提假设了?"(bootstrap也有前提——原始样本要能代表总体分布,如果原始15个样本本身因为太少而不能代表真实的付费金额分布,bootstrap重抽样也无法凭空创造信息,只是不需要"正态性"这一条具体假设,不是"零假设"的万能方案)

**常见坑:**
- 死记"n大于30就能用t检验/CLT"这条经验法则,不考虑数据本身的偏度——偏度越大,需要的样本量阈值越高,30是一个非常粗糙、不适用所有场景的经验数字。
- 认为bootstrap是"没有假设的免费午餐"——bootstrap隐含假设"观测样本的经验分布是总体分布的合理近似",小样本下这个假设本身也可能不成立。

---

下一篇:[04-classical-tests.md](04-classical-tests.md) —— 把本文的框架落地成具体可以直接调用的检验方法:t检验、卡方检验、ANOVA、非参数检验、bootstrap。
