# 06 · A/B测试设计与功效分析深挖(A/B Test Design & Power Analysis)

> 总览见 [00-roadmap.md](00-roadmap.md)

板块I建立了假设检验的完整理论框架,板块II从这里开始把它落地成业界最常见的应用场景——A/B测试。本文聚焦"实验开始之前要做的设计工作"(样本量怎么算)和"实验进行中最常见的操作性错误"(忍不住每天看结果)。

**环境声明:** 全部代码在仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1)下真实跑通,随机抽样固定种子(`np.random.default_rng(42)`)。

---

## 1. 样本量计算 —— 反过来用03类的功效公式

**定义与记号:** 给定baseline转化率p₀、想要检测的效应δ(即p₁=p₀+δ)、显著性水平α、目标功效power,反解每组所需样本量n。两比例检验的近似公式(pooled variance近似):

n ≈ 2(z_{α/2}+z_β)²·p̄(1-p̄) / δ²

其中p̄=(p₀+p₁)/2,z_β=Φ⁻¹(power)。

**一句话:** 这就是03类"给定n和δ求power"这个公式反过来解——不是新理论,是同一个功效公式换了一个未知数。

**数学推导:** 复用03类知识点3的功效公式框架:power=1-Φ(z_α - δ√n/σ)。要让power=目标值,即 z_α - δ√n/σ = Φ⁻¹(1-power) = -z_{power}(记z_β=Φ⁻¹(power)),整理:δ√n/σ = z_α + z_β,两边平方:n = (z_α+z_β)²σ²/δ²。对比例数据,单个观测的方差用p(1-p)近似(伯努利分布方差,01类推过),两组比较时方差项翻倍(两个独立样本的方差可加),代入整理后就是上面的公式。

**底层机制/为什么这样设计:** 这个公式清楚地展示了"样本量随什么变化"——δ在分母上被平方,意味着要检测的效应越小,需要的样本量按**平方反比**增长(检测效应减半,样本量要变成4倍);z_α、z_β(取决于α和目标power)决定了公式里的常数,α越严格、power要求越高,所需样本量都越大。

**AI研究/工程场景:** 产品上线一个新功能前,先用这个公式估算"要收集多少用户数据才能可靠判断这个功能是否有效",避免两种极端错误:样本量定得太小(实验做完了却测不出任何东西,浪费整个实验周期)或太大(不必要地拖长实验时间、让更多用户暴露在未经验证的变体下)。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

p0, p1 = 0.10, 0.12
alpha = 0.05
target_power = 0.80

z_alpha = stats.norm.ppf(1 - alpha / 2)
z_beta = stats.norm.ppf(target_power)
p_bar = (p0 + p1) / 2
delta = p1 - p0

n_per_group = int(np.ceil(2 * (z_alpha + z_beta) ** 2 * p_bar * (1 - p_bar) / delta ** 2))

# 数值模拟验证: 用这个样本量做实验, 重复多次, 真实达到的功效应该接近目标0.80
n_trials = 3000
rejections = 0
for _ in range(n_trials):
    group_a = rng.binomial(1, p0, n_per_group)
    group_b = rng.binomial(1, p1, n_per_group)
    p_pool = (group_a.sum() + group_b.sum()) / (2 * n_per_group)
    se_pool = np.sqrt(2 * p_pool * (1 - p_pool) / n_per_group)
    z_stat = (group_b.mean() - group_a.mean()) / se_pool
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    if p_value < alpha:
        rejections += 1

observed_power = rejections / n_trials
assert abs(observed_power - target_power) < 0.05, \
    f"observed power {observed_power:.4f} should be close to target {target_power} with n={n_per_group}/group"

# 验证平方反比关系: MDE减半, 所需样本量应该约变成4倍
delta_half = delta / 2
n_half_delta = int(np.ceil(2 * (z_alpha + z_beta) ** 2 * p_bar * (1 - p_bar) / delta_half ** 2))
ratio = n_half_delta / n_per_group
assert 3.5 < ratio < 4.5, f"halving MDE should roughly quadruple sample size, got ratio={ratio:.2f}"

print(f"required n per group = {n_per_group}, observed power = {observed_power:.4f} (target {target_power})")
print(f"halving delta: n={n_half_delta} (ratio={ratio:.2f}x, theory=4x)")
```

**面试怎么问+追问链**(规模递增轴):
- Q:"实验平台说'样本量不够,建议再跑两周',这个'不够'具体是怎么算出来的?"
- 追问1:"如果baseline转化率很低(比如1%),样本量需求会怎么变化?"(p̄(1-p̄)在p̄=0.5时最大,p̄越接近0或1这一项越小,但δ通常也会随之相应缩小(低baseline下能检测的绝对效应也小),实际样本量需求往往因为δ的影响占主导而显著增大——这里在考候选人是否理解各项因素的相对权重,不是死记公式)
- 深挖追问:"如果业务方说'不管样本量算出来多大,两周后必须出结果',你会怎么应对?"(如果两周内自然流量不够达到计算出的样本量,只能选择接受更低的功效(更可能漏掉真实存在的效应)、或者放宽MDE(只能可靠检测更大的效应)——这里在考候选人能否把统计约束转化为诚实的业务沟通,而不是假装约束不存在)

**常见坑:**
- 只套用公式算出一个数字,不检查这个样本量在业务实际流量下需要多久才能收集到——样本量计算的输出必须结合真实流量转化成"预计实验周期",否则只是一个孤立的数字。
- 忽视这个公式是**近似**公式(pooled variance近似),精确的两比例检验样本量公式项更复杂,近似公式在效应量较大时会有轻微低估,大规模严肃实验应该用精确公式或统计软件核实。

---

## 2. 最小可检测效应(MDE)—— 反过来问"这么多样本能测出多小的效应"

**定义与记号:** 固定样本量n、α、目标power,反解能可靠检测到的最小效应量δ_MDE。直接从知识点1的公式反解:δ_MDE = (z_α+z_β)·√(2p̄(1-p̄)/n)。

**一句话:** 样本量计算是"我要测δ这么大的效应,需要多少人";MDE是反过来问"我手头已经有n个人了,最小能测出多大的效应"——同一个约束关系,看你先固定哪个变量。

**数学推导:** 直接对知识点1公式 n=2(z_α+z_β)²p̄(1-p̄)/δ²(注意这个2——两组独立样本的方差相加带来的系数,和知识点1推导n公式时的来源相同,不是这里额外补的)关于δ求解:两边乘δ²、除以n,再开方——δ²=2(z_α+z_β)²p̄(1-p̄)/n,δ_MDE=(z_α+z_β)·√(2p̄(1-p̄)/n)。

**底层机制/为什么这样设计:** MDE这个视角在实验设计早期特别有用——很多时候样本量是被业务约束(流量规模、实验周期)硬性限定的,不是团队能自由选择的自由变量;这时候正确的问法不是"要多少样本",而是"给定这个样本量约束,我最多能可靠检测出多大的效应",如果算出的MDE远大于业务能接受的最小有意义提升,说明这次实验设计本身就注定测不出任何有价值的结论,应该在实验开始前就发现这一点,而不是做完实验才后知后觉。

**AI研究/工程场景:** 实验设计评审阶段,一个常见的检查项就是"这次实验的MDE是多少,业务方期望的提升幅度是否明显大于MDE"——如果MDE比预期效应还大,这次实验从设计上就该被驳回重新设计(增大样本量、延长周期,或者干脆放弃这次实验)。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

n_per_group = 1000
p0 = 0.10
alpha = 0.05
target_power = 0.80

z_alpha = stats.norm.ppf(1 - alpha / 2)
z_beta = stats.norm.ppf(target_power)
p_bar_approx = p0  # 效应未知时用baseline近似p_bar(效应通常不会大到显著改变这个近似)

mde = (z_alpha + z_beta) * np.sqrt(2 * p_bar_approx * (1 - p_bar_approx) / n_per_group)

# 数值验证: 真实效应恰好等于MDE时, 该样本量下的功效应该确实接近目标power
p1_at_mde = p0 + mde
n_trials = 3000
rejections = 0
for _ in range(n_trials):
    group_a = rng.binomial(1, p0, n_per_group)
    group_b = rng.binomial(1, p1_at_mde, n_per_group)
    p_pool = (group_a.sum() + group_b.sum()) / (2 * n_per_group)
    se_pool = np.sqrt(2 * p_pool * (1 - p_pool) / n_per_group)
    z_stat = (group_b.mean() - group_a.mean()) / se_pool
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    if p_value < alpha:
        rejections += 1

observed_power_at_mde = rejections / n_trials
assert abs(observed_power_at_mde - target_power) < 0.06, \
    f"power at exactly the MDE should be close to target {target_power}, got {observed_power_at_mde:.4f}"

# 效应量明显小于MDE时, 功效应该明显不足
p1_below_mde = p0 + mde * 0.4
rejections_below = sum(
    1 for _ in range(n_trials)
    if (lambda ga, gb: (2 * (1 - stats.norm.cdf(abs(
        (gb.mean() - ga.mean()) / np.sqrt(2 * ((ga.sum() + gb.sum()) / (2 * n_per_group)) * (1 - (ga.sum() + gb.sum()) / (2 * n_per_group)) / n_per_group))
    ))) < alpha)(rng.binomial(1, p0, n_per_group), rng.binomial(1, p1_below_mde, n_per_group))
)
power_below_mde = rejections_below / n_trials
assert power_below_mde < target_power - 0.2, f"power for a sub-MDE effect should be notably below target, got {power_below_mde:.4f}"

print(f"n={n_per_group}/group -> MDE={mde:.4f} (absolute)")
print(f"power at MDE={observed_power_at_mde:.4f} (target {target_power})   power at 0.4x MDE={power_below_mde:.4f} (should be much lower)")
```

**面试怎么问+追问链**(方案批判迭代轴):
- Q:"业务方说'我们只有5000个用户参与这次实验',你怎么判断这次实验设计靠不靠谱?"
- 追问1:"光看样本量数字能判断吗?"(不能孤立地看样本量大小,要结合baseline转化率和MDE公式算出这个样本量能检测的最小效应,再对比业务方期望看到的效应幅度)
- 深挖追问:"算出来MDE是3个百分点,但业务方说'哪怕提升0.5个百分点都很有价值',这时候怎么办?"(诚实地告知这次实验设计测不出0.5个百分点这么小的效应,需要增大样本量或延长实验周期;如果实在无法增加样本量,至少要让业务方清楚知道'实验结果不显著'在这种设计下不能被解读为'真的没有效果',这也是知识点6要专门展开的误判)

**常见坑:**
- MDE计算时用了实验开始前假设的p̄,但实际实验中的真实转化率和假设有较大偏差,导致实际功效和设计阶段计算的MDE不完全对应——严格来说应该在实验结束后用实际观测到的方差重新核实。
- 把MDE当成"能检测的效应下限"之外还误解成"效应一定至少有这么大"——MDE只是"这次实验设计下,能可靠检测的最小效应",不代表真实效应确实达到了这个量级。

---

## 3. 功效曲线 —— 样本量、效应量、功效三者的联动关系

**定义与记号:** 功效曲线(power curve):固定α,把功效表示成样本量n(或效应量δ)的函数,可视化(本文用数值扫描代替)功效如何随n或δ变化。

**一句话:** 功效曲线把"样本量该定多大"这个决策可视化成一条曲线而不是一个孤立数字——决策者能直观看到"再多收集20%的样本,功效能提升多少",而不是只有"够"或"不够"这个二元判断。

**数学推导:** 03类知识点3推的功效公式 power=1-Φ(z_α - δ√n/σ) 针对的是**单样本**情形(一个样本均值和一个已知基准比,标准误SE=σ/√n)。A/B测试比较的是**两个独立样本**的均值之差,和知识点1推导样本量公式时处理方差的逻辑完全一样——两个独立样本的方差可加,Var(X̄_b-X̄_a)=σ²/n+σ²/n=2σ²/n,标准误因此是SE=σ√(2/n),比单样本情形大√2倍。代入功效公式,真正的标准化效应量是δ/SE=δ√n/(σ√2),不是δ√n/σ——直接把单样本公式原样套过来而不做这个修正,会系统性地高估功效(分母少除了一个√2,标准化效应量被人为放大)。

除了标准误这处修正,公式还有第二处变化:03类推的是**单侧**检验(只有一个拒绝域,只在右尾),而A/B测试的常规做法是**双侧**检验(03类知识点6),两侧各有一个拒绝域,各自都可能贡献"拒绝H0"的概率,所以功效要把两条尾部概率加起来,而且拒绝阈值要用双侧的z_{α/2}(比如α=0.05时z_{0.025}≈1.96),不是单侧的z_α(比如z_{0.05}≈1.645)——这也是为什么下面代码里`z_alpha = stats.norm.ppf(1 - alpha / 2)`实际算的是z_{α/2}。两样本、双侧检验情形下正确的功效公式:

power(n,δ) = 1 - Φ(z_{α/2} - δ√n/(σ√2)) + Φ(-z_{α/2} - δ√n/(σ√2))

右边第一项是"真实效应δ>0时,统计量落进右侧拒绝域"的概率,第二项是"落进左侧拒绝域"的概率(δ>0时这一项概率很小,但双侧检验的定义决定了理论上必须把它算进去,不能因为数值小就直接省略这一项)。

固定δ扫描n,或固定n扫描δ,画出的曲线本质上都是同一个正态分布累积分布函数(经过参数变换)的不同切片——这也是为什么功效曲线总是呈现"S形"(sigmoid类似)的单调递增形状,不是巧合,是Φ函数本身的形状决定的。

**底层机制/为什么这样设计:** 功效曲线的边际收益是递减的——在"功效很低"的区间(比如从30%提升到50%),增加相同数量的样本能带来较大的功效提升;但在"功效已经很高"的区间(比如从95%提升到99%),同样数量的样本增量带来的功效提升非常有限(曲线趋于平坦,接近1的渐进线)。这是Φ函数(标准正态累积分布)本身尾部增长缓慢这个数学性质的直接体现,不是一个经验规律。

**AI研究/工程场景:** 实验设计评审时展示功效曲线,能帮助业务方直观理解"再等一周多收集样本"这个决策的边际价值——如果当前设计已经在曲线的平坦区间(功效已经很高),额外等待的收益很小;如果还在曲线陡峭上升的区间,继续收集样本的价值很大。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

alpha = 0.05
delta = 0.4
sigma = 2.0
z_alpha = stats.norm.ppf(1 - alpha / 2)

sample_sizes = [20, 50, 100, 200, 400, 800, 1600]
theoretical_powers = [
    1 - stats.norm.cdf(z_alpha - delta * np.sqrt(n) / (sigma * np.sqrt(2)))
    + stats.norm.cdf(-z_alpha - delta * np.sqrt(n) / (sigma * np.sqrt(2)))
    for n in sample_sizes
]

# 核心断言1: 功效应该随样本量单调递增
for i in range(len(theoretical_powers) - 1):
    assert theoretical_powers[i] < theoretical_powers[i + 1], \
        f"power should increase with n: {theoretical_powers}"

# 核心断言2: 边际收益递减 --- 正文举例的"30%->50%"区间(n:100->200)带来的功效提升,
# 应该明显大于"98%+->100%"区间(n:800->1600)带来的提升
early_gain = theoretical_powers[3] - theoretical_powers[2]  # n: 100->200
late_gain = theoretical_powers[6] - theoretical_powers[5]   # n: 800->1600
assert early_gain > late_gain * 3, \
    f"marginal power gain should shrink sharply once power is already high: early={early_gain:.4f} late={late_gain:.4f}"

# 数值模拟交叉验证其中一个样本量点的理论功效 --- 注意这里是两独立样本t检验(ttest_ind),
# 标准误是sigma*sqrt(2/n), 不是单样本情形的sigma/sqrt(n), 理论公式必须除以额外的sqrt(2)才能对上
n_check = 200
n_trials = 4000
rejections = sum(
    1 for _ in range(n_trials)
    if stats.ttest_ind(rng.normal(delta, sigma, n_check), rng.normal(0, sigma, n_check))[1] < alpha
)
simulated_power = rejections / n_trials
theory_at_check = theoretical_powers[sample_sizes.index(n_check)]
assert abs(simulated_power - theory_at_check) < 0.04, \
    f"simulated power {simulated_power:.4f} should match two-sample theory {theory_at_check:.4f} at n={n_check}"

print("power curve:", list(zip(sample_sizes, [round(p, 4) for p in theoretical_powers])))
print(f"n={n_check}: theory={theory_at_check:.4f}  simulated={simulated_power:.4f}")
print(f"gain 100->200 (~29%->52%): {early_gain:.4f}   gain 800->1600 (~98%->100%): {late_gain:.4f}")
```

**面试怎么问+追问链**(工程约束递增轴):
- Q:"现在功效只有60%,业务方问'再多跑多久能到90%',你怎么回答?"
- 追问1:"这个问题本身有没有陷阱?"("多跑多久"隐含假设了流量速率恒定,以及效应量本身不会变(07类会讨论新奇效应/学习效应导致效应量随时间变化这个更复杂的情形);在这些假设成立的前提下,可以用功效曲线反推需要的额外样本量,再结合流量速率换算成时间)
- 深挖追问:"如果功效曲线在当前样本量附近已经进入平坦区间,还要不要建议业务方继续等?"(如果已经接近曲线平坦区间,继续等待的边际收益很小,这时候更应该建议直接接受当前的功效水平做决策,而不是无止境地等待一个'完美'的高功效,这也是一个真实的工程/业务权衡)

**常见坑:**
- 把单样本情形的功效公式直接搬到两样本(A/B测试)场景,忘记两个独立样本的方差要相加(标准误从σ/√n变成σ√(2/n))——这个疏漏会让算出来的理论功效系统性偏高,本质是漏掉了"对照组自己也在抽样波动"这件事,只有实验组的波动被计入了。
- 把功效曲线的"边际收益递减"误解成"任何时候多收集数据都没有意义"——递减不等于零收益,在功效还不够高的区间,增加样本量依然有实质价值。
- 只关注"功效是否达到80%"这一个阈值判断,不看整条曲线的形状——曲线形状能揭示当前设计是"差一点就够"还是"还差得很远",这两种情况的应对策略应该不同。

---

## 4. 窥探问题(Peeking Problem)—— 每天看一次结果,真实假阳性率远超5%

**定义与记号:** 窥探(peeking):在实验尚未达到预先设定的样本量/截止时间之前,反复查看当前累积数据的显著性检验结果,一旦观测到p<α就提前停止并宣布"显著"。这种做法下,真实的总体Type I错误率(记为α_actual)会远超原本声称的单次检验α。

**一句话:** 每次"看一眼p值",都是在用当前累积的数据打一次赌;反复打赌、只要赢一次就收手,累积下来的"至少赢一次"的概率,远高于打一次赌的概率——这正是05类"多重比较问题"在时间维度上的具体化身。

**数学推导/直觉:** 这个问题本质上和05类知识点1(多重比较)、04类知识点4(ANOVA vs 多次两两检验)是同一个数学结构,只是这次的"多次检验"发生在**时间**维度上而不是**指标/组别**维度上——每天看一次,相当于对同一个假设做了T次相关的(不是独立的,因为后一天的数据包含前一天的数据)检验,累积的假阳性概率会显著超过单次检验的α,而且由于各次检验高度相关(共享大部分数据),膨胀程度介于"完全独立多重检验"(膨胀最严重)和"只做一次检验"(不膨胀)之间,不能简单套用1-(1-α)^T这个独立情形的公式,但定性结论(膨胀确实存在且可观)是一致的。

**底层机制/为什么这样设计:** 为什么窥探比普通多重比较更容易被忽视?因为它感觉起来"只做了一次分析"——很多人直觉上认为"我只是在等结果出来的过程中看了几眼",没有意识到"看了几眼、任何一眼显著就停"这个操作,在统计上和"做了好几次独立检验、任何一次显著就报告"是同一类问题,区别只是检验之间存在数据重叠导致的相关性,不是问题的有无。

**AI研究/工程场景:** 几乎所有自助式实验平台(允许实验负责人随时查看当前结果的dashboard)都天然诱导窥探行为——如果平台没有专门的序贯检验机制(07类会讲),团队"看到数据慢慢变得显著就迫不及待宣布胜利"是一个极其常见、真实发生的分析错误,是实验平台设计里最容易被低估的一类系统性偏差来源。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

n_checks = 20  # 比如实验持续20天, 每天累积样本量增加, 数据不断累积不是重新抽取
max_n_per_group = 2000
check_points = np.linspace(max_n_per_group // n_checks, max_n_per_group, n_checks).astype(int)

n_trials = 2000
peeking_false_positives = 0
single_look_false_positives = 0

for _ in range(n_trials):
    group_a_full = rng.normal(0, 1, max_n_per_group)
    group_b_full = rng.normal(0, 1, max_n_per_group)  # H0严格成立(两组真实同分布), 全程没有真实效应

    stopped_significant = False
    for n_cur in check_points:
        _, p = stats.ttest_ind(group_a_full[:n_cur], group_b_full[:n_cur])
        if p < 0.05:
            stopped_significant = True
            break
    if stopped_significant:
        peeking_false_positives += 1

    _, p_final = stats.ttest_ind(group_a_full, group_b_full)  # 只在预先设定的最终样本量看一次
    if p_final < 0.05:
        single_look_false_positives += 1

peeking_fpr = peeking_false_positives / n_trials
single_look_fpr = single_look_false_positives / n_trials

# 核心断言1: 只看一次(预先设定截止点)的假阳性率应该接近名义alpha=0.05
assert single_look_fpr < 0.08, f"single-look FPR should be near 0.05, got {single_look_fpr:.4f}"

# 核心断言2: 反复窥探、见显著就停的假阳性率应该明显膨胀, 远超5%
assert peeking_fpr > single_look_fpr * 2, \
    f"peeking FPR ({peeking_fpr:.4f}) should be far higher than single-look FPR ({single_look_fpr:.4f})"
assert peeking_fpr > 0.15, f"peeking FPR should be substantially above nominal 0.05, got {peeking_fpr:.4f}"

print(f"single predetermined look:  false positive rate = {single_look_fpr:.4f}  (target ~0.05)")
print(f"peeking {n_checks} times, stop at first p<0.05: false positive rate = {peeking_fpr:.4f}  (should be much higher)")
```

**面试怎么问+追问链**(方案批判迭代轴,这是A/B测试面试里最经典的追问场景之一):
- Q:"实验跑到第3天,你看到p=0.03,已经显著了,能不能提前结束实验、宣布获胜?"
- 追问1:"如果不能,为什么明明p<0.05还不能下结论?"(p<0.05是在'只检验一次'这个前提下才有5%的名义假阳性率保证;如果这已经是你查看的第N次、之前几次不显著这次显著就停,累积的真实假阳性率早已超过5%,这个p值不能按照它字面的数字直接采信)
- 深挖追问:"那如果业务方坚持要每天看数据监控实验健康度(不是为了做决策,只是想知道实验有没有跑歪),这样可以吗?"(可以看数据本身(比如样本量是否正常累积、有没有明显的数据质量问题),但看数据监控健康度和'看到显著就停止并当作最终结论'必须严格区分开——前者不影响决策规则,后者才是窥探问题的根源;如果做不到这种自我约束,应该采用专门设计的序贯检验方法(07类的always-valid p值),而不是靠自觉)

**常见坑:**
- 把"看了很多次数据、只在最后一次预先设定的时间点做决策"和"看了很多次数据、随时可能提前停止"混为一谈——前者完全没有窥探问题,后者才有,区分点在于"决策规则里是否包含'看到显著就提前停'这个选项",不是"看了几次数据"这个动作本身。
- 认为"只要不是天天看,一周看一次就没事"——窥探问题的膨胀程度和查看次数正相关,但只要存在"看到显著提前停"这个决策规则,查看次数再少也存在膨胀,只是程度较轻,不是"降低到某个频率就完全没有风险"的二元问题。

---

## 5. 多指标/多变体的多重比较 —— A/B测试里最常见的多重检验现场

**定义与记号:** A/B测试同时监控多个业务指标(点击率、转化率、留存率、客单价……)或同时测试多个变体(A/B/C/D……),对每个指标/每个变体分别做显著性检验,这正是05类多重比较问题最典型的现实原型。

**一句话:** "这次实验测了10个指标,有1个显著变好了"这句话,如果不做任何多重比较校正,信息量比听起来的要低得多——10个完全无效的指标里,"至少1个偶然显著"这件事本来就有很高的概率发生。

**数学推导:** 直接复用05类知识点1的公式:1-(1-α)^m,m是同时监控的指标数。10个独立指标、α=0.05:1-0.95^10≈0.401——四成概率至少有一个指标"意外"显著,即使没有任何真实效应存在。

**底层机制/为什么这样设计:** 和05类的区别在于这里的"多重"通常发生在**同一批数据**的不同切面上(不同指标、不同用户细分)——这意味着这些检验之间的相关性结构比"完全不同的数据集分别检验"更复杂(不同指标之间经常有业务逻辑上的相关性,比如"点击率"和"转化率"往往同向变动),实际的假阳性膨胀程度会因为这种相关性而和独立假设下的公式有偏差,但"存在膨胀"这个定性结论是稳健的。

**AI研究/工程场景:** 实验报告"上线后整体收入没有显著变化,但18-24岁用户群体的次日留存显著提升了3%"这类"事后从多个细分维度里挑出一个显著的"结论,是多重比较问题在用户分群分析里的经典化身,业界称为"数据挖掘式显著性"(data-dredged significance),需要专门标注为"探索性发现,需要独立数据验证"而不是当作确定性结论。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

n_metrics = 10
n_trials = 4000
alpha = 0.05

at_least_one_significant = 0
for _ in range(n_trials):
    p_values = []
    for _ in range(n_metrics):
        # 模拟10个完全无关联、真实没有任何效应的业务指标
        control = rng.normal(0, 1, 200)
        treatment = rng.normal(0, 1, 200)
        _, p = stats.ttest_ind(treatment, control)
        p_values.append(p)
    if any(p < alpha for p in p_values):
        at_least_one_significant += 1

observed_rate = at_least_one_significant / n_trials
theoretical_rate = 1 - (1 - alpha) ** n_metrics

assert abs(observed_rate - theoretical_rate) < 0.04
assert observed_rate > 0.30, f"with 10 null metrics, chance of at least one false 'win' should exceed 30%, got {observed_rate:.4f}"

print(f"{n_metrics} unrelated null metrics: theory={theoretical_rate:.4f}  observed={observed_rate:.4f}")
print("=> reporting 'metric X improved significantly' without correction, from among many tracked metrics, is misleading")
```

**面试怎么问+追问链**(真实性验证轴):
- Q:"你的实验报告说'整体没有显著提升,但发现某个用户细分群体显著提升了',这个结论能直接采纳吗?"
- 追问1:"这个细分维度是实验设计阶段就确定要看的,还是做完实验之后翻遍各种切法找到的?"(如果是实验前预先注册(pre-registered)的假设,可信度高;如果是事后翻查很多种切分方式才找到的这一个"显著"结果,本质上是05/06类反复强调的多重比较问题,需要用独立的新数据重新验证,不能直接当作定论)
- 深挖追问:"如果确实想探索性地看很多细分维度,有没有更严谨的做法?"(预先声明会检查哪些细分维度(限制m的规模)、对这些细分维度的检验做FDR或Bonferroni校正、或者把数据分成"探索集"和"验证集",探索集里找到的模式必须在独立的验证集上复现才采信——这是真实的工程实践,不是理论空谈)

**常见坑:**
- 把"预先设计好、会重点监控的少数几个核心指标"和"实验结束后临时想到、事后翻查的大量细分角度"混为一谈,对后者不做任何多重比较处理就直接采信。
- 报告实验结果时只挑出显著的指标汇报,不提及一共测了多少个指标(这种选择性报告即使不是有意为之,也会系统性地误导决策者)。

---

## 6. 样本量不足的误判 —— "未显著"不等于"没有效果"

**定义与记号:** 复用03类"Type II错误/功效"的框架:如果实验的真实功效很低(比如因为样本量不足、或者真实效应本身就小于设计时假设的MDE),观测到"不显著"这个结果,既可能是"真的没有效果"(H0为真),也可能是"有效果但这次实验没power测出来"(H1为真但β错误发生了)——这两种情况从单次实验的p值本身无法区分。

**一句话:** "p>0.05"翻译成人话应该是"这次实验没有找到充分证据证明效应存在",不是"证明了效应不存在"——03类已经建立了这个原则,本知识点用A/B测试里具体的数字展示这个原则被违反的后果有多严重。

**数学推导/说明:** 如果真实效应δ_true远小于设计阶段假定的MDE,实际功效会大幅低于设计时的目标(比如目标power=80%,但真实效应只有MDE的一半,功效可能只剩30%-40%左右,04-05类功效曲线的形状决定了这个关系不是线性的)——这意味着"该实验有60%-70%的概率,即使真实效应存在,也会给出'不显著'的错误结论",这个具体的量化是本知识点的核心。

**底层机制/为什么这样设计:** 这个误判之所以容易发生,是因为"不显著"这三个字听起来像是一个明确的、盖棺定论式的判决,但它实际传递的信息量,严重依赖这次实验本身的功效有多高——同样一句"不显著",在功效95%的实验里几乎能确认"真的没有效果",在功效30%的实验里几乎什么都说明不了,区分这两种情形必须回头看功效分析,不能只看p值本身。

**AI研究/工程场景:** "我们试过这个改动,A/B测试没有显著效果,所以放弃了这个方向"——这句话在很多实际业务复盘里出现,但如果没有人回头检查当时那次实验的功效有多高,完全可能是一个真实有效的改进方向因为样本量不足被错误放弃,这是一个真实存在、造成真实机会成本损失、但很少被追溯复盘的分析错误。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

p0 = 0.10
true_effect = 0.015  # 真实存在的小效应(绝对值), 但实验设计时样本量算少了
alpha = 0.05
n_underpowered = 400  # 明显不足的样本量(对应本文知识点2的MDE公式反推, 这个n下MDE远大于0.015)

n_trials = 3000
rejections = 0
for _ in range(n_trials):
    group_a = rng.binomial(1, p0, n_underpowered)
    group_b = rng.binomial(1, p0 + true_effect, n_underpowered)
    p_pool = (group_a.sum() + group_b.sum()) / (2 * n_underpowered)
    se_pool = np.sqrt(2 * p_pool * (1 - p_pool) / n_underpowered)
    z_stat = (group_b.mean() - group_a.mean()) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    if p_value < alpha:
        rejections += 1

observed_power = rejections / n_trials

# 核心断言: 真实效应确实存在(true_effect > 0), 但这个样本量下的功效很低, 大部分实验会错误地"未能拒绝H0"
assert observed_power < 0.5, \
    f"with this underpowered sample size, power should be low despite a real effect existing, got {observed_power:.4f}"
assert observed_power > 0.05, "power should still be meaningfully above the false-positive baseline, confirming a real (if hard to detect) effect"

# 对照组: 用知识点1的公式算出"真正够用"的样本量, 重新验证功效应该显著提升
z_alpha = stats.norm.ppf(1 - alpha / 2)
z_beta = stats.norm.ppf(0.80)
p_bar = p0 + true_effect / 2
n_adequate = int(np.ceil(2 * (z_alpha + z_beta) ** 2 * p_bar * (1 - p_bar) / true_effect ** 2))

rejections_adequate = 0
for _ in range(n_trials):
    group_a = rng.binomial(1, p0, n_adequate)
    group_b = rng.binomial(1, p0 + true_effect, n_adequate)
    p_pool = (group_a.sum() + group_b.sum()) / (2 * n_adequate)
    se_pool = np.sqrt(2 * p_pool * (1 - p_pool) / n_adequate)
    z_stat = (group_b.mean() - group_a.mean()) / se_pool
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    if p_value < alpha:
        rejections_adequate += 1

observed_power_adequate = rejections_adequate / n_trials
assert observed_power_adequate > observed_power + 0.3, \
    f"adequately-powered design (n={n_adequate}) should have much higher power than underpowered design (n={n_underpowered}): {observed_power_adequate:.4f} vs {observed_power:.4f}"

print(f"underpowered (n={n_underpowered}/group): power={observed_power:.4f}  (real effect exists but usually missed)")
print(f"adequately powered (n={n_adequate}/group): power={observed_power_adequate:.4f}")
```

**面试怎么问+追问链**(真实性验证轴):
- Q:"团队说'这个功能改动测试了,数据显示没有显著差异',你会怎么追问?"
- 追问1:"当时实验的样本量是多少,设计阶段算过MDE吗?"(候选人应该主动去追溯这次实验设计阶段的功效分析,而不是把'不显著'直接当成'没有效果'的证据)
- 深挖追问:"如果查出来当时功效只有30%,能不能补救?"(可以事后用观测到的效应量估计,结合当时的样本量重新算功效,如果确实偏低,建议用更大样本量重新做一次验证性实验,而不是仅凭一次低功效实验的'不显著'结果就永久放弃这个方向)

**常见坑:**
- 把任何"p>0.05"的结果都不加区分地解读为"没有效果",不回头检查当次实验的实际功效。
- 反过来走向另一个极端——因为"不显著不代表没效果"这句话,拒绝接受任何"不显著"的结果、坚持认为效应一定存在只是没测出来——如果实验设计本身功效已经足够高(比如90%+),"不显著"依然是相当强的证据说明真实效应即使存在也很小,不能用"功效不足"当借口无限期地不接受一个经过充分设计的实验结果。

---

下一篇:[07-modern-experimentation.md](07-modern-experimentation.md) —— 序贯检验、CUPED方差削减、新奇效应,处理"窥探问题"暴露出的这些更现实的实验痛点。
