# 07 · 现代实验方法深挖(Modern Experimentation Methods)

> 总览见 [00-roadmap.md](00-roadmap.md)

06类建立了A/B测试设计的基本功(样本量/MDE/功效曲线)和最经典的操作性错误——窥探问题(想看就看,阈值不变,真实假阳性率远超名义5%)。本文聚焦成熟实验平台如何正规地解决这些痛点:序贯检验和always-valid p值是治窥探问题的两种正规方案(前者要求提前规划检验时间表,后者更彻底,允许真正随时查看);CUPED是几乎不增加任何成本就能提升功效的工程利器;新奇效应/学习效应与网络效应/SUTVA违反,则是即使实验设计和分析流程都做对了,也可能因为违反"处理效应恒定""个体互不干扰"这些隐含假设而得出错误结论的两类真实陷阱。

**环境声明:** 全部代码在仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1)下真实跑通,随机抽样固定种子(`np.random.default_rng(42)`)。

---

## 1. 序贯检验基础 —— 提前规划好"什么时候可以看"

**定义与记号:** 在实验进行过程中按预先设定的K个检验时间点(用信息量分数 t_k=k/K 表示,k=1,...,K)多次查看数据,每次用一个经过校正的显著性阈值(而不是每次都用0.05),使累积的Type I错误率仍然控制在名义α水平。O'Brien-Fleming边界是一种常见的校正方式:早期检验用非常严格的阈值(几乎不可能拒绝),越接近计划的最终样本量,阈值越接近常规水平。

**一句话:** 06类的窥探问题是"想看就看,阈值不变"导致的错误;序贯检验是"确实允许多次看,但把每次的阈值都主动调严"来抵消多次检验带来的膨胀——是治窥探问题的正规方法,不是禁止看数据,而是改变看数据的规则。

**数学推导:** 用简化版O'Brien-Fleming边界思想:在K个预先设定的检验点(信息分数t_k=k/K),用边界 z_k=z_{α/2}/√t_k 作为每次检验的临界值(t_k越早越小,临界值越大,越难拒绝;t_K=1时z_K=z_{α/2},退化成普通单次检验的阈值)。这不是精确的O'Brien-Fleming公式(精确公式需要求解多元正态分布的联合边界,涉及数值积分求解满足整体α约束的常数,超出本文范围),但完整捕捉了其核心形状:边界随t_k单调递减,首尾分别对应"几乎不可能拒绝"和"常规阈值"。

**底层机制/为什么这样设计:** 为什么早期边界要设得特别严格?因为早期数据量小、噪声大,如果允许"轻易拒绝",大部分"看起来显著"的早期波动其实是噪声,不是真实效应;O'Brien-Fleming这种边界形状的设计哲学是"把大部分显著性预算留到后期",因为后期数据量大、噪声小,此时的显著性证据更可信,也更符合实践中"早期不轻易叫停,除非效应大到不容置疑"的直觉。

**AI研究/工程场景:** 经过认真设计的实验平台几乎都内置某种形式的序贯检验或alpha-spending机制,允许业务方合规地"每天看一眼"而不破坏统计有效性,这也是为什么这些平台的仪表盘经常会显示"当前不能停止"或者一个动态调整的显著性阈值,而不是永远用固定的0.05。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

alpha = 0.05
K = 5  # 5个预先设定的检验点
info_fractions = np.array([1, 2, 3, 4, 5]) / K
z_alpha_half = stats.norm.ppf(1 - alpha / 2)

# 简化O'Brien-Fleming风格边界: z_k = z_alpha_half / sqrt(t_k)
of_boundaries = z_alpha_half / np.sqrt(info_fractions)

# 边界形状验证: 早期严格(临界值大), 最后一次退化成常规阈值
assert of_boundaries[0] > of_boundaries[-1], "boundary should be strictest at the earliest look"
assert abs(of_boundaries[-1] - z_alpha_half) < 1e-9, "boundary at the final look should equal the ordinary z_alpha/2"
assert all(of_boundaries[i] > of_boundaries[i + 1] for i in range(len(of_boundaries) - 1)), \
    "boundary should be monotonically decreasing across looks"

max_n_per_group = 2000
check_points = (info_fractions * max_n_per_group).astype(int)

n_trials = 2000
naive_stop_count = 0   # 06类知识点4的做法: 每次都用z_alpha_half
of_stop_count = 0      # 用O'Brien-Fleming风格边界

for _ in range(n_trials):
    group_a = rng.normal(0, 1, max_n_per_group)
    group_b = rng.normal(0, 1, max_n_per_group)  # H0严格成立, 数据不断累积不是重新抽取

    naive_stop, of_stop = False, False
    for k, n_cur in enumerate(check_points):
        se = np.sqrt(2 / n_cur)
        z_stat = (group_b[:n_cur].mean() - group_a[:n_cur].mean()) / se
        if abs(z_stat) > z_alpha_half:
            naive_stop = True
        if abs(z_stat) > of_boundaries[k]:
            of_stop = True
    if naive_stop:
        naive_stop_count += 1
    if of_stop:
        of_stop_count += 1

naive_fpr = naive_stop_count / n_trials
of_fpr = of_stop_count / n_trials

assert naive_fpr > alpha * 1.5, f"naive repeated testing FPR should be inflated, got {naive_fpr:.4f}"
assert of_fpr < naive_fpr, f"O'Brien-Fleming boundary FPR ({of_fpr:.4f}) should be far lower than naive ({naive_fpr:.4f})"
assert of_fpr < alpha * 1.5, f"O'Brien-Fleming boundary FPR should stay close to nominal alpha, got {of_fpr:.4f}"

print(f"5 boundaries (t=0.2..1.0): {[round(b, 3) for b in of_boundaries]}")
print(f"naive repeated testing FPR = {naive_fpr:.4f}  (inflated, target ~0.05)")
print(f"O'Brien-Fleming boundary FPR = {of_fpr:.4f}  (close to nominal 0.05)")
```

**面试怎么问+追问链**(方案批判迭代轴,直接衔接06类知识点4窥探问题):
- Q:"你说频繁看数据会导致假阳性膨胀,那业务方就是坚持要每天看,有什么正规的解决方案?"
- 追问1:"这个边界为什么早期比后期严格,而不是反过来?"(早期数据量小、单次波动的置信度低,如果早期也用常规阈值,少量噪声就足以触发误判;边界形状把"容易被批准显著"的额度集中在数据量已经足够大的后期)
- 深挖追问:"如果实验中途要临时增加一次计划外的检验(不在预先设定的K个点上),会发生什么?"(破坏了序贯检验方法预先设定检验次数/位置的前提——边界的整体α控制保证依赖于检验次数和位置在实验开始前就固定,中途随意加检验点又会重新引入类似窥探问题的膨胀,这也是序贯方法在实践中最容易被"合理化违反"的地方)

**常见坑:**
- 序贯边界的检验次数/位置必须在实验开始前确定,实验中途随意增加"顺便看一眼"的检验点会破坏边界的理论保证,等同于走回窥探问题。
- 把序贯检验和"贝叶斯早停"(13类会讲)混淆——频率派序贯检验仍然需要预先设定检验方案,贝叶斯方法原则上不需要提前设定停止规则,这是治疗同一个痛点的两条不同技术路线。

---

## 2. Always-valid p值直觉 —— 真正做到随时查看都不膨胀

**定义与记号:** Always-valid p值(随时有效的p值):一种特殊构造的p值序列,使得无论在什么时刻停止查看(不需要像知识点1那样预先设定检验次数或位置),"在整个观察过程中曾经看到p<α"这件事发生的概率在H0下始终不超过α——比序贯检验更彻底地解决窥探问题。

**一句话:** 知识点1要求你提前规划好"什么时候看";always-valid p值更进一步,允许你随时看、看多少次都行,理论保证依然成立——代价是通常需要构造基于似然比"鞅"(martingale)的检验统计量,不是普通t检验p值直接套用。

**数学推导:** 完整的严格证明依赖鞅论(超出用户当前"随机过程未学"的数学背景,这里只建立数值直觉,不做严格证明——呼应spec里19/20类"现场建最小必要直觉"的同一原则提前用在这里)。构造方式:假设观测X₁,...,Xₙ独立同分布N(θ,σ²),H0:θ=0。给θ一个"混合先验"N(0,τ²)(τ²代表"预期效应量级"的尺度参数,不是真的在做贝叶斯推断,只是借用这个数学构造),算出边际似然比(混合似然/H0似然):

Λₙ = √(σ²/(σ²+nτ²)) · exp(n²τ²X̄ₙ² / (2σ²(σ²+nτ²)))

这个Λₙ有一个关键性质:在H0(θ=0)下,{Λₙ}是一个非负鞅,且E[Λₙ]=1对所有n成立(标准的似然比鞅性质)。根据鞅论里的Ville不等式:P(sup_n Λₙ ≥ 1/α) ≤ α。这意味着,只要用规则"Λₙ首次≥1/α就拒绝H0",无论你在哪个n停下来查看,累积犯错概率永远不超过α——不需要像知识点1那样提前固定检验次数。

**底层机制/为什么这样设计:** 为什么普通p值"随时看都会膨胀",而这种特殊构造的p值不会?关键在于累积统计量是否具有"鞅"这个数学性质——直觉上,鞅是"公平游戏"的数学化:在任意时刻,给定过去所有信息,下一步的期望值等于当前值,不会系统性地朝任何方向漂移。普通的累积z统计量不满足这个性质(H0下它本身没有偏移,但"取绝对值后越过阈值"这件事的概率会随查看次数累积),但精心构造的似然比过程满足鞅性质,这正是Ville不等式能够生效的前提。

**AI研究/工程场景:** Always-valid推断是现代实验平台(支持"随时查看"而不用担心破坏统计有效性的A/B测试系统)的核心技术支撑,是从"数据科学团队被动规定检验时间表"到"业务方主动自由探索仪表盘"这一使用体验转变背后的统计学基础。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

alpha = 0.05
sigma = 1.0
tau2 = 1.0  # 混合先验方差, 代表"预期效应量级"的尺度参数
threshold = 1 / alpha  # Ville不等式给出的拒绝阈值: Lambda_n >= 1/alpha 就拒绝

def mixture_likelihood_ratio(x_bar_n, n, sigma, tau2):
    return np.sqrt(sigma ** 2 / (sigma ** 2 + n * tau2)) * np.exp(
        (n ** 2 * tau2 * x_bar_n ** 2) / (2 * sigma ** 2 * (sigma ** 2 + n * tau2))
    )

max_n = 2000
n_trials = 2000
step = 20  # 每隔20个观测查看一次(逐点检查计算量太大, 用步长近似"随时查看")
z_alpha_half = stats.norm.ppf(1 - alpha / 2)

always_valid_rejections = 0
naive_peeking_rejections = 0  # 对照: 06类知识点4式的naive反复检验(同样的查看频率)

for _ in range(n_trials):
    data = rng.normal(0, sigma, max_n)  # H0严格成立
    cum_sum = np.cumsum(data)
    av_rejected, naive_rejected = False, False
    for n in range(step, max_n + 1, step):
        x_bar_n = cum_sum[n - 1] / n
        lam = mixture_likelihood_ratio(x_bar_n, n, sigma, tau2)
        if lam >= threshold:
            av_rejected = True
        z_stat = x_bar_n / (sigma / np.sqrt(n))
        if abs(z_stat) > z_alpha_half:
            naive_rejected = True
    if av_rejected:
        always_valid_rejections += 1
    if naive_rejected:
        naive_peeking_rejections += 1

av_fpr = always_valid_rejections / n_trials
naive_fpr = naive_peeking_rejections / n_trials

# 核心断言1: always-valid规则的假阳性率应该控制在名义alpha附近(Ville不等式给出的是上界, 实际经常明显更低)
assert av_fpr <= alpha * 1.5, f"always-valid stopping rule's FPR should stay near nominal alpha, got {av_fpr:.4f}"

# 核心断言2: 同样查看100次(每20个观测看一次), naive规则的假阳性率应该严重膨胀, 远超always-valid
assert naive_fpr > 0.25, f"naive repeated peeking (100 looks) should be severely inflated, got {naive_fpr:.4f}"
assert av_fpr < naive_fpr / 5, \
    f"always-valid FPR ({av_fpr:.4f}) should be dramatically lower than naive peeking FPR ({naive_fpr:.4f})"

print(f"checking every {step} observations, up to n={max_n} (100 looks total):")
print(f"  always-valid FPR = {av_fpr:.4f}  (target <= {alpha})")
print(f"  naive repeated-peeking FPR = {naive_fpr:.4f}  (severely inflated)")
```

**面试怎么问+追问链**(决策依据追问轴):
- Q:"既然序贯检验(知识点1)已经能解决窥探问题,为什么还需要always-valid p值这种更复杂的方案?"
- 追问1:"两者的核心区别是什么?"(序贯检验需要提前固定检验的次数和时间点;always-valid允许完全不提前规划,真正做到随时查看随时可能停止,理论保证依然成立)
- 深挖追问:"既然always-valid看起来更灵活,为什么工业界不是所有场景都直接用它,还在用更简单的序贯边界?"(实现复杂度更高,需要构造合适的似然比过程,且需要为混合先验的尺度参数τ²做出合理选择;在检验次数确实能提前规划好的场景(比如固定周期发布的实验),序贯边界足够且更简单、更容易向非统计背景的同事解释;always-valid的价值在真正"完全不可预测查看时机"的场景更明显——这是典型的"没有免费午餐,方法选择要看场景约束"追问)

**常见坑:**
- 误以为"always-valid"意味着"可以无限制地查看无数次都完全没有代价"——虽然假阳性率不膨胀,但检验的功效(效率)通常比固定样本量、固定单次检验的方法更低,这是为换取"随时查看的自由"付出的代价,不是免费的。
- 把always-valid p值直接当成普通p值解读("p=0.03所以效应有97%的把握")——它的构造方式和语义都不同于普通p值,不能套用普通p值的直觉去解释具体数值大小。

---

## 3. CUPED方差削减 —— 用实验前数据"免费"提升功效

**定义与记号:** CUPED(Controlled-experiment Using Pre-Experiment Data,利用实验前数据的受控实验):利用实验开始前就已经观测到的、和实验期间指标相关的协变量(比如用户在实验开始前的历史行为数据),对实验期间的指标做线性调整,削减指标的方差,在不增加样本量的前提下提升统计功效。

**一句话:** 如果知道某个用户历史上就是重度用户,那ta在实验期间的指标天然会偏高,这部分"天生偏高"和实验本身(A还是B)无关,CUPED就是把这部分可预测的、和处理无关的方差扣掉,让剩下的方差更纯粹地反映处理效应的信号。

**数学推导:** 设Y为实验期间的指标,X为实验开始前观测到的协变量,构造调整后的指标:

Y_cuped = Y - θ(X - X̄), 其中 θ = Cov(Y,X)/Var(X)(和OLS回归系数完全同一形式)

可以证明 Var(Y_cuped) = Var(Y) - θ²Var(X) = Var(Y)·(1-Corr(X,Y)²)——方差削减的比例恰好等于X和Y相关系数的平方。同时调整不改变处理组和对照组之间指标均值之差的无偏性:因为X是实验开始前观测的,不受处理影响,两组X的分布本应相同,调整项θ(X-X̄)对两组的期望贡献相同,在做组间差分时相互抵消,E[Y_cuped,treatment]-E[Y_cuped,control] = E[Y_treatment]-E[Y_control]。

**底层机制/为什么这样设计:** 为什么CUPED不需要额外增加样本量就能提升"等效样本量"?本质上和"配对检验比独立样本检验更有效"(04类知识点2)是同一个原理——两者都利用了"每个用户自身的基线水平"这个额外信息,把和处理无关的、可预测的那部分变异从检验统计量的分母(标准误)里剔除出去,让分子(组间均值差)相对分母更突出,从而在完全不改变样本量的前提下提升功效,等效于"免费"获得了更多样本量的统计效力。

**AI研究/工程场景:** 大型互联网公司的实验平台几乎都内置CUPED或其推广版本,因为对于历史行为高度可预测、个体差异极大的业务指标(比如用户消费金额、活跃度),实验前的历史数据几乎总是和实验期指标高度相关,CUPED经常能带来30%以上甚至更高的方差削减,相当于同样的用户流量能跑出原本需要多招募不少用户才能达到的统计功效,是实验平台里投入产出比最高的技术优化之一。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

n_per_group = 1000
true_effect = 0.5

# 每个用户有一个"内在活跃度"水平, 同时影响实验前指标X和实验期指标Y(制造真实相关性)
user_baseline = rng.normal(10, 3, 2 * n_per_group)
X = user_baseline + rng.normal(0, 1, 2 * n_per_group)          # 实验前指标: 内在水平 + 噪声
treatment = np.array([0] * n_per_group + [1] * n_per_group)
Y = user_baseline + true_effect * treatment + rng.normal(0, 1, 2 * n_per_group)  # 实验期指标

# CUPED调整
theta = np.cov(Y, X)[0, 1] / np.var(X, ddof=1)
Y_cuped = Y - theta * (X - X.mean())

# 核心断言1: 方差确实按理论比例下降 Var(Y_cuped) ≈ Var(Y)*(1-corr(X,Y)^2)
corr_xy = np.corrcoef(X, Y)[0, 1]
var_y = np.var(Y, ddof=1)
var_y_cuped = np.var(Y_cuped, ddof=1)
theoretical_var_cuped = var_y * (1 - corr_xy ** 2)
assert abs(var_y_cuped - theoretical_var_cuped) / theoretical_var_cuped < 0.01, \
    "CUPED variance reduction should match the analytical (1 - corr^2) formula almost exactly"

# 核心断言2: 组间均值差估计量调整前后都应该无偏(都接近true_effect)
naive_diff = Y[treatment == 1].mean() - Y[treatment == 0].mean()
cuped_diff = Y_cuped[treatment == 1].mean() - Y_cuped[treatment == 0].mean()
assert abs(naive_diff - true_effect) < 0.15
assert abs(cuped_diff - true_effect) < 0.15

# 核心断言3: 用调整后指标做检验, 标准误明显更小(等效于更高功效)
se_naive = np.sqrt(np.var(Y[treatment == 1], ddof=1) / n_per_group + np.var(Y[treatment == 0], ddof=1) / n_per_group)
se_cuped = np.sqrt(np.var(Y_cuped[treatment == 1], ddof=1) / n_per_group + np.var(Y_cuped[treatment == 0], ddof=1) / n_per_group)
assert se_cuped < se_naive * 0.6, f"CUPED should meaningfully shrink standard error: naive={se_naive:.4f} cuped={se_cuped:.4f}"

print(f"corr(X,Y)={corr_xy:.4f}  variance reduction={1 - var_y_cuped / var_y:.4f} (theory={corr_xy ** 2:.4f})")
print(f"naive diff={naive_diff:.4f}  cuped diff={cuped_diff:.4f}  (true effect={true_effect}, both unbiased)")
print(f"SE naive={se_naive:.4f}  SE cuped={se_cuped:.4f}  (ratio={se_cuped / se_naive:.3f})")
```

**面试怎么问+追问链**(决策依据追问轴,真实性验证风格):
- Q:"简历上写'引入CUPED方差削减技术,实验所需样本量降低40%',这个40%具体是怎么来的?"
- 追问1:"为什么不直接把X当成一个额外的自变量放进回归里,而是用这种'调整后再做检验'的两步做法?"(CUPED本质上和"把X作为协变量做ANCOVA"是等价的做法,两步法在工程实现上更简单、更容易和现有的检验流程解耦——这里在考察候选人是否理解CUPED和更一般的协变量调整方法的关系,不是把CUPED当成一个孤立的黑盒技巧)
- 深挖追问:"如果选错了协变量X(比如X和Y的相关性很弱,或者X其实受到了处理的影响),会发生什么?"(相关性弱时方差削减效果有限,不会带来错误结论但收益很小;如果X本身受处理影响(比如错误地把"实验期间第一天的行为"当成协变量,但那已经被处理影响了),会破坏CUPED无偏性的前提条件,导致调整后的估计量有偏——这是CUPED在工程实践中最容易踩的坑)

**常见坑:**
- 用了实验开始之后(哪怕只是实验第一天)的数据作为协变量X——这违反了"X不受处理影响"的核心假设,一旦违反,CUPED调整后的估计量不再无偏。
- 误以为CUPED能提升所有指标的功效——如果指标本身和历史数据几乎不相关(比如全新功能的首次使用行为,没有对应的历史基线),CUPED的方差削减效果会接近于零,不是万能技巧。

---

## 4. 新奇效应与学习效应 —— 处理效应本身不是时间恒定的常数

**定义与记号:** 新奇效应(novelty effect):用户因为"这是新东西"这个新鲜感本身而产生的短期行为变化,通常表现为初期效应被高估、随时间衰减。学习效应(learning effect):用户需要时间适应新设计,初期效应被低估,随熟悉度提升效应逐渐显现或增强。两者都指向同一个更本质的问题——A/B测试"处理效应恒定"这个隐含假设,在现实中经常不成立。

**一句话:** 如果只看实验结束时的聚合平均效应,会把"前期新鲜感很高但快速衰减"和"效应从头到尾稳定"这两种性质完全不同的情况,误判成同一个数字。

**数学推导:** 构造效应随时间指数衰减的模型:δ(t)=δ_∞+(δ₀-δ_∞)e^(-t/τ),其中δ₀是初始效应(新奇效应导致偏高)、δ_∞是长期稳态效应、τ是衰减时间常数。如果实验只跑到时间T就结束并只看聚合平均效应 δ̄=(1/T)∫₀ᵀδ(t)dt,这个积分可以直接解出:

δ̄ = δ_∞ + (δ₀-δ_∞)·(τ/T)·(1-e^(-T/τ))

当T远大于τ时δ̄→δ_∞(实验跑得足够长,衰减效应被稀释掉,接近真实长期效应);但当T和τ相当时,δ̄会明显偏向δ₀(被前期的新奇效应主导)——这个公式量化了"实验跑多久"和"聚合均值有多失真"之间的关系,不是只能定性地说"可能有偏差"。

**底层机制/为什么这样设计:** 为什么只看聚合均值会踩这个坑?因为聚合平均值这个统计量本身丢失了"效应随时间如何变化"这个信息维度——两个效应曲线完全不同的场景(一个始终稳定在低水平,一个从高位快速衰减)可能产生几乎相同的聚合平均值,但对"这个改动值不值得永久上线"这个决策问题的含义截然不同:前者说明改动有稳定的长期价值,后者说明改动的价值主要来自短期新鲜感,长期可能所剩无几。

**AI研究/工程场景:** 推荐系统或UI改版类实验里,新奇效应是最常见的分析陷阱之一——一个新的排序算法或界面改版上线后,用户因为好奇会点击更多、停留更久,如果实验只跑一两周就下结论"这个改动带来了显著提升",很可能只是捕捉到了新奇效应,长期效果需要额外做"效应是否随时间衰减"的时序分析,或者用holdout(留一部分用户永久不接受新版本)长期跟踪的方式来验证效应的长期稳定性。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

delta_0 = 0.15    # 初始效应(新奇效应导致偏高)
delta_inf = 0.01  # 长期稳态效应
tau = 4.0         # 衰减时间常数(单位: 天)
T_total = 28      # 实验总共跑28天

def true_effect_at_t(t):
    return delta_inf + (delta_0 - delta_inf) * np.exp(-t / tau)

# 数值积分验证解析公式
t_grid = np.linspace(0, T_total, 100_000)
numerical_avg = np.trapezoid(true_effect_at_t(t_grid), t_grid) / T_total
analytical_avg = delta_inf + (delta_0 - delta_inf) * (tau / T_total) * (1 - np.exp(-T_total / tau))
assert abs(numerical_avg - analytical_avg) < 1e-3

# 核心断言1: 聚合平均效应系统性偏离长期稳态效应delta_inf, 明显被前期新奇效应拉高
assert analytical_avg > delta_inf * 2.5, \
    f"aggregate average ({analytical_avg:.4f}) should be pulled well above the true steady-state effect ({delta_inf})"
assert analytical_avg < delta_0 * 0.5, "aggregate average should still be well below the initial novelty-inflated effect"

# 按周分段重新估计效应, 应该能看出效应逐周衰减这个模式(而聚合均值看不出)
n_users_per_day_per_group = 3000
weekly_effects = []
for week in range(4):
    day_effects = []
    for day in range(week * 7, (week + 1) * 7):
        true_delta_today = true_effect_at_t(day + 0.5)  # 用当天中点时刻的效应
        control = rng.normal(0, 1, n_users_per_day_per_group)
        treat = rng.normal(true_delta_today, 1, n_users_per_day_per_group)
        day_effects.append(treat.mean() - control.mean())
    weekly_effects.append(float(np.mean(day_effects)))

# 核心断言2: 分周效应应该呈现明显的单调递减趋势, 聚合均值这一个数字看不出这个模式
assert all(weekly_effects[i] > weekly_effects[i + 1] for i in range(3)), \
    f"weekly effects should show a clear decaying trend: {weekly_effects}"
assert weekly_effects[0] > weekly_effects[-1] + 0.02, \
    f"week-1 vs week-4 gap should be substantial: week1={weekly_effects[0]:.4f} week4={weekly_effects[-1]:.4f}"

print(f"aggregate 28-day average effect = {analytical_avg:.4f}  (true steady-state={delta_inf}, initial novelty={delta_0})")
print(f"weekly effects: {[round(w, 4) for w in weekly_effects]}  (clearly decaying, invisible in the aggregate number)")
```

**面试怎么问+追问链**(诊断真实数据新题型):
- Q:"给你一份实验数据:第1周效应+6.5%,第2周+2.2%,第3周+2.0%,第4周+0.4%,你怎么解读这个模式?"
- 追问1:"只根据这4个数字,你能确定这是新奇效应衰减,还是别的原因?"(不能仅凭"效应在下降"这一个现象就直接下结论是新奇效应——还需要排除其他可能,比如实验分组本身随时间发生了污染、外部季节性因素同时影响了两组、或者第一周数据本身样本量小噪声大恰好偏高,需要结合样本量、置信区间的宽窄、以及是否存在理论上合理的新奇效应作用机制来综合判断)
- 深挖追问:"如果确认是新奇效应,实验该怎么处理,是不是直接按第4周的+0.4%作为最终结论?"(如果衰减曲线还没有明显趋于平稳(第4周相对第3周降幅还很大),直接采用第4周数字仍然可能低估或高估长期效应,更严谨的做法是继续延长实验直到效应曲线本身趋于平稳,或者用衰减模型外推稳态值,但外推本身要谨慎并给出不确定性区间,不能假装外推值就是确定的真相)

**常见坑:**
- 只跑很短的实验周期就下"长期结论",把新奇效应误判为稳定的长期效应,上线后长期数据打脸。
- 反过来走向另一个极端:任何前期效应强、后期效应减弱的模式都归咎于"新奇效应",不去检验是否存在其他更合理的解释(比如学习效应和新奇效应叠加导致的复杂曲线,不是单一机制能完全解释的)。

---

## 5. 网络效应与SUTVA违反 —— "A组和B组互不干扰"这个假设本身可能不成立

**定义与记号:** SUTVA(Stable Unit Treatment Value Assumption,个体处理值稳定假设):因果推断的基本假设之一,要求"一个个体的结果只取决于ta自己分配到的处理,不受其他个体分配到什么处理的影响"。网络效应/溢出效应(spillover effect):当个体之间存在社交连接、竞争关系或共享资源时,这个假设会被违反——比如实验组用户的行为会通过社交网络影响对照组用户。

**一句话:** A/B测试最基本的假设是"A组和B组互不干扰",但如果A组用户会在社交网络里把新功能安利给自己认识的B组好友,这个假设就已经被现实打破了,朴素的组间比较不再能得出正确的因果结论。

**数学推导:** 设真实的个体处理效应是τ,SUTVA成立时Y_i只取决于个体i自己的处理分配。SUTVA违反、存在正向溢出(实验组的存在提升了对照组的结果,比如社交安利)时,朴素估计量 τ̂_naive=Ȳ_treatment-Ȳ_control 会系统性地低估真实处理效应——因为对照组的观测值在现实中被"污染"抬高了(它本该代表"完全没有任何人接受处理时"的基准值,但实际观测到的是"周围有一部分好友接受了处理"时的值),导致两组之间原本该有的差距被人为缩小了。

**底层机制/为什么这样设计:** 为什么这个偏差的方向通常是"低估"而不是随机方向?因为常见的溢出机制(社交安利、正向网络外部性)大多是同向的——处理组的存在倾向于把对照组"往处理组的方向拉",缩小了两组之间本该观测到的差异,这也是为什么很多社交产品的实验设计者会专门强调"网络效应会让你的A/B测试结果看起来比真实效果更保守",需要额外的实验设计来隔离溢出。

**AI研究/工程场景:** 社交产品(好友推荐、内容分享功能)、双边市场(网约车、外卖平台的供需匹配算法实验)、以及任何存在"个体之间会互相影响"结构的场景,都需要在实验设计阶段就考虑SUTVA是否成立——常见的应对方案是按聚类随机化(cluster randomization,把整个社交群体/地理区域作为随机化单元而不是个体),这是因果推断理论对实验设计实践产生直接指导的经典例子。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

n_users = 2000
true_individual_effect = 1.0
spillover_strength = 0.4  # 溢出效应强度: 对照组好友受到的"污染"占真实效应的比例

# 简化社交网络: 用块状结构近似, 每10个用户组成一个"好友组"(组内互为好友)
group_size = 10
n_groups = n_users // group_size
friend_group = np.repeat(np.arange(n_groups), group_size)
rng.shuffle(friend_group)  # 打散, 让好友组和后续的个体随机分组没有系统性对应关系

treatment = rng.integers(0, 2, n_users)  # 个体随机化: 每个用户独立分配
baseline = rng.normal(0, 1, n_users)

# 计算每个用户所在好友组里, 处理组好友的比例(不含自己)
treated_neighbor_fraction = np.zeros(n_users)
for g in range(n_groups):
    idx = np.where(friend_group == g)[0]
    for i in idx:
        others = idx[idx != i]
        treated_neighbor_fraction[i] = treatment[others].mean() if len(others) > 0 else 0.0

# 真实结果生成过程: 处理组获得完整效应; 对照组因为好友里有处理组用户而被"污染"部分效应
Y = baseline + true_individual_effect * treatment \
    + spillover_strength * true_individual_effect * (1 - treatment) * treated_neighbor_fraction \
    + rng.normal(0, 0.5, n_users)

naive_diff = Y[treatment == 1].mean() - Y[treatment == 0].mean()

# 核心断言1: 朴素个体随机化估计量系统性低估真实处理效应
assert naive_diff < true_individual_effect * 0.85, \
    f"naive individual-randomization estimate ({naive_diff:.4f}) should be biased well below the true effect ({true_individual_effect})"
assert naive_diff > 0, "should still detect *some* positive effect, just an attenuated one"

# 对照: 聚类随机化(整个好友组一起分配), 组内不再有"部分处理部分对照"的污染
group_treatment = rng.integers(0, 2, n_groups)
treatment_cluster = group_treatment[friend_group]
Y_cluster = baseline + true_individual_effect * treatment_cluster + rng.normal(0, 0.5, n_users)
cluster_diff = Y_cluster[treatment_cluster == 1].mean() - Y_cluster[treatment_cluster == 0].mean()

# 核心断言2: 聚类随机化的估计量应该明显更接近真实效应
assert abs(cluster_diff - true_individual_effect) < abs(naive_diff - true_individual_effect), \
    f"cluster randomization ({cluster_diff:.4f}) should be closer to the true effect than naive individual randomization ({naive_diff:.4f})"

print(f"true individual effect = {true_individual_effect}")
print(f"naive individual-randomization estimate = {naive_diff:.4f}  (biased low due to spillover)")
print(f"cluster-randomization estimate = {cluster_diff:.4f}  (much closer to the true effect)")
```

**面试怎么问+追问链**(工程约束递增轴):
- Q:"一个新的'邀请好友'功能做A/B测试,个体随机分组,结果发现处理组和对照组几乎没有差异,你觉得这个结论可信吗?"
- 追问1:"这个场景有什么特殊之处,让个体随机化可能不适用?"(功能本身的机制就是"邀请好友"——处理组用户天然会去邀请对照组好友,直接违反SUTVA,对照组会被处理组的行为间接影响,个体随机化下的比较很可能严重低估真实效应)
- 深挖追问:"如果换成按地理区域整体分组(同一个城市的用户要么全部是处理组要么全部是对照组),能完全解决问题吗?"(能大幅削减同一地理区域内的溢出,但不能完全消除跨区域的社交连接影响,而且整体分组会大幅减少有效的随机化单元数量——原本几百万个体现在变成几十个城市,统计功效会大幅下降,需要更长的实验周期或更多区域才能达到同样的统计把握,这是一个需要权衡的真实工程代价,不是"换个分组方式就万事大吉")

**常见坑:**
- 看到处理组和对照组差异很小就直接下结论"这个功能没有效果",没有意识到功能本身的机制(社交传播/网络效应)可能导致个体随机化设计从一开始就无法准确估计真实效应,问题出在实验设计而不是功能本身。
- 简单地认为"只要样本量足够大,网络效应的影响就会被平均掉"——网络效应导致的是系统性偏差(bias),不是随机噪声(variance),增加样本量只能降低方差,不能修正偏差,这是04-05类反复强调的"偏差vs方差不是同一类问题"在这里的具体应用。

---

下一篇:[08-causal-inference-foundations.md](08-causal-inference-foundations.md) —— 从A/B测试(能做随机化实验的理想情形)转向更普遍的因果推断框架,为09类处理"做不了随机化实验时怎么办"打基础。
