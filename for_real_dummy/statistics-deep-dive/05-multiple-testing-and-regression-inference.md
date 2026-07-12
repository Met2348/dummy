# 05 · 多重检验与回归推断深挖(Multiple Testing & Regression Inference)

> 总览见 [00-roadmap.md](00-roadmap.md)

本文两条线:第一条正式量化04类ANOVA一节已经预告过的"多次检验会怎样系统性膨胀假阳性率"这个问题,给出具体的修正方法(Bonferroni、BH流程);第二条把假设检验框架用到回归系数上——一个回归系数"显著"是什么意思,置信区间怎么构造,以及回归模型自己的假设不成立时会发生什么。

**环境声明:** 全部代码在仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1)下真实跑通,随机抽样固定种子(`np.random.default_rng(42)`)。**`statsmodels` 未安装**(已用 `.venv/Scripts/python.exe -c "import statsmodels"` 实测确认 `ModuleNotFoundError`)——本文回归相关知识点全部手写正规方程 `(X'X)⁻¹X'y` + 解析标准误公式,不安装新依赖;简单线性场景用 `scipy.stats.linregress` 交叉验证。

---

## 1. 多重比较问题 —— 为什么测20次总有一次"显著"

**定义与记号:** 族错误率(Family-Wise Error Rate, FWER)= P(m次独立检验里至少有一次假阳性)。如果每次检验独立、都用水平α,FWER = 1-(1-α)^m。

**一句话:** 03类已经证明"H0为真时p值服从均匀分布",这个知识点是它最直接的推论——测的次数越多,"运气好蒙到一次p<0.05"的总概率越高,和任何一次检验本身的质量无关。

**数学推导:** m次独立检验,每次在H0为真时"不犯错"(p≥α)的概率是1-α,m次都不犯错的概率是(1-α)^m(独立事件概率相乘),所以"至少一次犯错"的概率是1-(1-α)^m。代入m=20,α=0.05:1-0.95^20≈1-0.3585≈**0.6415**——测20个完全无效应的指标,超过六成概率会"意外"测出至少一个显著结果。

**底层机制/为什么这样设计:** 这个公式假设了m次检验相互独立——现实中很多"测很多个指标"的场景,指标之间是相关的(比如同一批用户的多个转化率指标),这时候真实的FWER会比1-(1-α)^m更低(不是独立事件,公式是独立情形下的上界),但"多测总会撞见假阳性"这个定性结论依然成立,只是膨胀幅度会打折扣。

**AI研究/工程场景:** 评测一个新模型时同时报告十几个benchmark分数,如果不做任何多重比较校正,几乎必然有一两个benchmark"恰好"显著变好,这正是14类"刷榜陷阱"的数学根源。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

m = 20
alpha = 0.05
n_trials = 5000

at_least_one_false_positive = 0
for _ in range(n_trials):
    p_values = []
    for _ in range(m):
        group_a = rng.normal(0, 1, 30)
        group_b = rng.normal(0, 1, 30)  # 完全同分布, 全部H0严格为真
        _, p = stats.ttest_ind(group_a, group_b)
        p_values.append(p)
    if any(p < alpha for p in p_values):
        at_least_one_false_positive += 1

observed_fwer = at_least_one_false_positive / n_trials
theoretical_fwer = 1 - (1 - alpha) ** m

assert abs(observed_fwer - theoretical_fwer) < 0.03, \
    f"observed FWER {observed_fwer:.4f} should match theory {theoretical_fwer:.4f}"
assert observed_fwer > 0.55, "with m=20 independent null tests, FWER should be well over 50%"

print(f"m={m} independent null tests, alpha={alpha}: theoretical FWER={theoretical_fwer:.4f}  observed={observed_fwer:.4f}")
```

**面试怎么问+追问链**(规模递增轴):
- Q:"你的实验平台同时监控50个业务指标,每天自动跑显著性检验,这个系统设计有什么隐患?"
- 追问1:"具体隐患的量级有多大?"(m=50,α=0.05,FWER=1-0.95^50≈92%——几乎每天都会自动报出至少一个"显著"指标,即使系统完全没有任何真实变化)
- 深挖追问:"这些指标之间通常不是完全独立的(比如都和总收入相关),这会让问题变得更严重还是更轻?"(通常会让实际FWER比独立情形算出的公式值更低一些——但仍然显著高于单次检验的名义α,不能因为"指标相关"就忽视这个问题)

**常见坑:**
- 只在"正式做A/B测试"时才意识到多重比较问题,忽视日常监控大盘、探索性数据分析里同样存在(而且往往更严重,因为监控指标数量经常远超一次正式实验设计的假设数)。
- 认为多重比较问题只发生在"同时测多个指标"——04类已经展示过,"多次两两比较"(pairwise comparisons)是同一个问题的另一种常见形式。

---

## 2. Bonferroni校正 —— 最简单也最保守的FWER控制方法

**定义与记号:** Bonferroni校正:把每次检验的显著性阈值从α改成α/m,只有p<α/m才拒绝该次H0。

**一句话:** 用union bound(布尔不等式)保证FWER≤α——不要求检验之间独立,是最"皮实"、最容易实现、但也最保守(容易漏掉真实效应)的多重比较校正方法。

**数学推导:** 布尔不等式(union bound):P(A₁∪A₂∪...∪Aₘ) ≤ ΣP(Aᵢ),不要求事件Aᵢ独立。把Aᵢ设为"第i次检验假阳性"这个事件,P(Aᵢ)=α/m(每次用校正后的阈值),所以 P(至少一次假阳性) ≤ m×(α/m) = α——FWER被精确控制在α以内,而且这个证明**不依赖**各次检验相互独立这个假设,这正是Bonferroni校正相比"直接用1-(1-α)^m反解"更常用的原因(后者的推导依赖独立性,前者不依赖)。

**底层机制/为什么这样设计:** Bonferroni保守的代价在于它是"最坏情况"下也成立的保证——union bound不管检验之间是正相关、负相关还是独立,都成立,这种"无条件成立"的鲁棒性天然意味着在大多数(检验之间存在正相关的)实际场景下,真实FWER远低于α,校正比实际需要的更严格,牺牲了检验力(功效)。

**AI研究/工程场景:** 少量、彼此独立性存疑、后果严重不容有失的检验场景(比如药物安全性的多个终点指标),Bonferroni的保守是合理的代价;但m很大(几百上千个假设)时,Bonferroni的检验力损失会大到几乎测不出任何真实效应,这时候需要FDR方法(下一个知识点)。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

m = 20
alpha = 0.05
alpha_corrected = alpha / m
n_trials = 5000

fwer_uncorrected = 0
fwer_bonferroni = 0
for _ in range(n_trials):
    p_values = [stats.ttest_ind(rng.normal(0, 1, 30), rng.normal(0, 1, 30))[1] for _ in range(m)]
    if any(p < alpha for p in p_values):
        fwer_uncorrected += 1
    if any(p < alpha_corrected for p in p_values):
        fwer_bonferroni += 1

fwer_uncorrected_rate = fwer_uncorrected / n_trials
fwer_bonferroni_rate = fwer_bonferroni / n_trials

# 核心断言: Bonferroni校正后FWER被控制在alpha以内, 未校正的FWER远超alpha
assert fwer_bonferroni_rate < alpha * 1.5, f"Bonferroni-corrected FWER should stay near/below alpha={alpha}, got {fwer_bonferroni_rate:.4f}"
assert fwer_uncorrected_rate > alpha * 5, f"uncorrected FWER should be far higher, got {fwer_uncorrected_rate:.4f}"

print(f"uncorrected FWER={fwer_uncorrected_rate:.4f}  Bonferroni-corrected FWER={fwer_bonferroni_rate:.4f}  (target alpha={alpha})")
```

**面试怎么问+追问链**(方案批判迭代轴):
- Q:"你对1000个基因分别做假设检验,用Bonferroni校正,阈值变成0.05/1000=0.00005,这个方案合适吗?"
- 候选人若说"合适,FWER被严格控制住了"→追问1:"这么严格的阈值,会不会漏掉大量真实存在的效应?"(会——m很大时Bonferroni阈值极其苛刻,检验力会大幅下降,大量真实效应因为达不到这么严格的阈值而被漏掉,这在大规模假设检验场景(基因组学、A/B测试平台海量指标)里是真实存在的实用性问题)
- 换方案后追问2:"那有没有别的方法在控制某种错误率的同时,保留更多检验力?"(FDR方法——不追求"一个假阳性都不能有"这么严格的标准,转而控制"被拒绝的发现里,有多大比例是假的",在能接受少量假阳性存在的场景下换回更高的检验力,这正是下一个知识点)

**常见坑:**
- 不考虑m的规模就默认用Bonferroni——m很小(几个到十几个)时Bonferroni是合理选择,m很大(几百上千)时几乎总是应该考虑FDR方法。
- 混淆"控制FWER"和"控制FDR"是两种不同的错误率定义,不是同一个目标的两种实现方式(下一个知识点会展开这个区别)。

---

## 3. FDR与Benjamini-Hochberg(BH)流程 —— 控制"发现里有多少是假的"而不是"一个都不能错"

**定义与记号:** 错误发现率(False Discovery Rate, FDR)= E[假阳性数 / 拒绝总数](拒绝总数为0时定义FDR=0)。BH流程:把m个p值从小到大排序p₍₁₎≤p₍₂₎≤...≤p₍ₘ₎,找到最大的k使得 p₍ₖ₎ ≤ (k/m)·q(q是目标FDR水平),拒绝前k个(p值最小的k个)对应的H0。

**一句话:** FWER问的是"一个假阳性都不能有的概率",FDR问的是"我拒绝的这些发现里,平均有多大比例其实是假的"——后者是一个更宽松、但在"能接受少量假阳性换取更多真实发现"的场景下更实用的标准。

**数学推导:** BH流程的阈值 (k/m)·q 是关键——阈值随排序位置k线性增长,意味着排名越靠后(p值越大)的检验,允许通过的阈值也越宽松,这和Bonferroni"所有检验统一用α/m这一个最严格阈值"形成对比。BH流程能证明在检验统计量满足一定的正相关条件(或独立)下,严格保证 E[FDR] ≤ (m₀/m)·q ≤ q(m₀是真实H0成立的检验数量)——这是一个关于**平均**行为的保证,不是逐次实验都恰好等于q,所以验证这条性质需要多次重复实验取平均,不能只看一次结果。

**底层机制/为什么这样设计:** 为什么阈值要随排序位置递增?直觉上,如果m个检验中大部分p值都很小(意味着很可能存在大量真实效应),那么某个具体的p值排在靠后位置、但依然不太大,更可能是真实效应而不是偶然;BH流程的递增阈值正是在用"其他检验的p值分布信息"来调整对当前这个检验的判断标准,这是它比逐一检验独立设定阈值的方法(如Bonferroni)更有效地利用数据整体信息的地方。

**AI研究/工程场景:** 大规模在线实验平台同时监控成百上千个细分维度/指标的显著性,FDR方法(而不是逐个Bonferroni校正)是行业标准做法——能接受"报出的显著发现里有10%左右可能是假的"这个可控代价,换取远高于Bonferroni的检验力。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

def bh_procedure(p_values, q=0.10):
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    thresholds = (np.arange(1, m + 1) / m) * q
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(m, dtype=bool)
    k = np.max(np.where(below)[0]) + 1  # 最大的满足条件的排序位置(1-indexed)
    rejected_sorted = np.zeros(m, dtype=bool)
    rejected_sorted[:k] = True
    rejected = np.zeros(m, dtype=bool)
    rejected[order] = rejected_sorted
    return rejected

# 构造已知真假阳性标签的合成数据: 100个检验, 前20个有真实效应(true positive候选), 后80个是真实H0(true negative候选)
m_total = 100
n_true_effects = 20
q_target = 0.10
n_trials = 400

fdr_per_trial = []
power_per_trial = []
for _ in range(n_trials):
    p_values = []
    is_true_effect = [i < n_true_effects for i in range(m_total)]
    for i in range(m_total):
        a = rng.normal(0, 1, 30)
        b = rng.normal(0.8 if is_true_effect[i] else 0.0, 1, 30)
        _, p = stats.ttest_ind(b, a)
        p_values.append(p)

    rejected = bh_procedure(p_values, q=q_target)
    n_rejected = rejected.sum()
    n_false_discoveries = sum(1 for i in range(m_total) if rejected[i] and not is_true_effect[i])
    n_true_discoveries = sum(1 for i in range(m_total) if rejected[i] and is_true_effect[i])

    fdr_trial = n_false_discoveries / n_rejected if n_rejected > 0 else 0.0
    power_trial = n_true_discoveries / n_true_effects
    fdr_per_trial.append(fdr_trial)
    power_per_trial.append(power_trial)

mean_fdr = float(np.mean(fdr_per_trial))
mean_power = float(np.mean(power_per_trial))

# 核心断言: 平均FDR应该被控制在目标水平q附近(是平均性质, 不是逐次精确相等)
assert mean_fdr < q_target * 1.5, f"mean FDR {mean_fdr:.4f} should stay reasonably close to target q={q_target}"
# BH流程应该有不错的检验力(相比Bonferroni这种极端保守的方法, 能发现大部分真实效应)
assert mean_power > 0.5, f"BH should retain reasonable power to detect true effects, got {mean_power:.4f}"

print(f"BH procedure (q={q_target}) over {n_trials} trials: mean FDR={mean_fdr:.4f}  mean power={mean_power:.4f}")
```

**面试怎么问+追问链**(决策依据追问轴):
- Q:"什么场景你会选FWER控制(Bonferroni),什么场景选FDR控制(BH)?"
- 追问1:"能不能给一个具体的判断标准,而不是泛泛地说'看场景'?"(m小、每个假阳性代价极高不容许(比如药物安全性终点)→ FWER;m很大、能容忍一定比例假阳性换取更多真实发现(大规模指标监控、基因组学筛选)→ FDR)
- 深挖追问:"BH流程对p值之间的相关性结构有什么要求?"(原始BH证明假设独立或正相关依赖(positive regression dependency),检验统计量强烈负相关的场景需要更保守的变体如Benjamini-Yekutieli;这里在考候选人是否只会调库、还是知道方法背后的适用边界)

**常见坑:**
- 混淆FDR和FWER的定义,或者混淆"BH校正后的p值(adjusted p-value)"和"原始p值"直接比较(BH校正后的判断依据是排序位置和排序后的阈值,不是简单地把每个原始p值乘以某个固定倍数)。
- 认为FDR方法"完全没有代价"——FDR允许的假阳性数量随拒绝总数增长(FDR是比例不是绝对数量),拒绝的发现越多,假阳性的绝对数量也可能越多,这是"控制比例"和"控制绝对数量"两种不同保证的本质区别。

---

## 4. OLS假设 —— 同方差性被违反时,标准误估计会错

**定义与记号:** 简单线性回归 y=β₀+β₁x+ε,经典OLS(普通最小二乘)的几个核心假设:线性性(y和x的真实关系是线性的)、独立性(误差项之间不相关)、**同方差性**(homoscedasticity:Var(ε\|x)对所有x相同)、正态性(误差项近似正态,主要影响小样本推断)。

**一句话:** OLS点估计(斜率截距本身)在同方差假设不成立时依然是**无偏**的,但用来算标准误、置信区间、p值的经典公式全都假设了同方差——这个假设一旦不成立,置信区间的覆盖率会系统性地不准确(不是点估计错了,是"这个估计有多可信"这句话错了)。

**数学推导/说明:** 经典OLS标准误公式 Var(β̂)=σ²(X'X)⁻¹ 里的σ²是一个**单一**的常数,代表"所有观测点的误差方差都相同"这个假设的直接体现。如果真实的Var(εᵢ)随xᵢ变化(异方差),用同一个σ²(通常估计成残差的平均平方,本质是所有点方差的某种平均)去描述所有点的不确定性,对方差本来就小的点会高估不确定性、对方差本来就大的点会低估不确定性——这种"错配"会让基于σ²(X'X)⁻¹算出的标准误,系统性偏离β̂真实的抽样标准差。

**底层机制/为什么这样设计:** 为什么点估计β̂本身不受影响?OLS点估计的推导(正规方程)完全没有用到"方差是常数"这个假设,只用到"误差期望为0、和X不相关"这两个更弱的条件——高斯-马尔可夫定理里"OLS是最优线性无偏估计量(BLUE)"这个更强的结论(方差最小)才需要同方差假设,无偏性本身不需要。

**AI研究/工程场景:** 用户活跃度(x轴)预测付费金额(y轴)这类场景经常天然异方差——低活跃度用户付费金额普遍很小、方差也小,高活跃度用户付费金额差异巨大(有的转化成大R,有的仍然不付费),方差随x增大——直接套用OLS标准公式给出的置信区间会明显不准。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

true_beta0, true_beta1 = 2.0, 3.0
n = 200
n_trials = 3000

def fit_ols(x, y):
    X = np.column_stack([np.ones_like(x), x])
    beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
    residuals = y - X @ beta_hat
    sigma2_hat = np.sum(residuals ** 2) / (len(y) - 2)
    cov_beta = sigma2_hat * np.linalg.inv(X.T @ X)
    se_beta1 = np.sqrt(cov_beta[1, 1])
    return beta_hat, se_beta1

def repeated_fit(heteroscedastic, n_trials_local):
    # 直接对比"朴素公式声称的SE"和"重复抽样得到的真实抽样标准差", 比二元覆盖计数统计效率更高、噪声更小
    beta1_estimates = np.empty(n_trials_local)
    naive_se_estimates = np.empty(n_trials_local)
    for i in range(n_trials_local):
        x = rng.uniform(1, 10, n)
        if heteroscedastic:
            noise_sd = 0.3 * x ** 2  # 标准差随x平方增长(比线性更陡峭), 让高x端的大方差点主导整体估计, 避免低x端的高杠杆低方差点稀释效应
        else:
            noise_sd = np.full_like(x, 9.0)  # 常数方差 (同方差, 对照组, 量级和上面大致匹配)
        y = true_beta0 + true_beta1 * x + rng.normal(0, noise_sd)
        beta_hat, se_beta1 = fit_ols(x, y)
        beta1_estimates[i] = beta_hat[1]
        naive_se_estimates[i] = se_beta1
    return beta1_estimates, naive_se_estimates

beta1_homo, se_homo = repeated_fit(heteroscedastic=False, n_trials_local=n_trials)
beta1_het, se_het = repeated_fit(heteroscedastic=True, n_trials_local=n_trials)

empirical_se_homo = beta1_homo.std()
mean_naive_se_homo = se_homo.mean()
empirical_se_het = beta1_het.std()
mean_naive_se_het = se_het.mean()

# 核心断言1: 同方差数据下, 朴素公式给出的SE应该和真实抽样标准差高度一致
assert abs(mean_naive_se_homo - empirical_se_homo) / empirical_se_homo < 0.1, \
    f"under homoscedasticity, naive SE ({mean_naive_se_homo:.4f}) should closely match empirical SE ({empirical_se_homo:.4f})"

# 核心断言2: 异方差数据下, 朴素公式给出的SE应该明显偏离真实抽样标准差(公式假设不成立导致的系统性错误, 不是抽样噪声)
relative_mismatch = abs(mean_naive_se_het - empirical_se_het) / empirical_se_het
assert relative_mismatch > 0.15, \
    f"under heteroscedasticity, naive SE ({mean_naive_se_het:.4f}) should notably mismatch empirical SE ({empirical_se_het:.4f}), relative mismatch={relative_mismatch:.4f}"

print(f"homoscedastic:   naive SE={mean_naive_se_homo:.4f}  empirical SE={empirical_se_homo:.4f}  (should match)")
print(f"heteroscedastic: naive SE={mean_naive_se_het:.4f}  empirical SE={empirical_se_het:.4f}  (mismatch={relative_mismatch:.4f}, formula is wrong)")
```

**面试怎么问+追问链**(方案批判迭代轴):
- Q:"你用OLS拟合了一个回归,系数p值很显著,面试官提示你先看看残差图,为什么?"
- 追问1:"残差图能看出什么,和显著性判断有什么关系?"(残差图(本文用数值化方式代替,下一个知识点会具体讲)能直观看出残差方差是否随拟合值/自变量变化;如果存在明显的"喇叭口"形状(异方差),说明标准误公式的前提假设被违反,已经算出的p值和置信区间可信度存疑)
- 深挖追问:"发现异方差之后,除了reject整个模型重新分析,有没有更直接的修正方法?"(稳健标准误,如White/Huber-White异方差稳健标准误——不假设同方差,用残差本身估计一个更灵活的方差-协方差结构,是工业界处理异方差最常用的手段,不需要重新指定整个模型)

**常见坑:**
- 只看回归系数的p值就下结论,不检查残差是否呈现异方差模式(或其他假设违反的迹象)。
- 混淆"异方差导致点估计有偏"(错误说法)和"异方差导致标准误/推断有偏"(正确说法)——这是本知识点最容易被追问确认的细节。

---

## 5. 回归系数的置信区间与假设检验 —— 手写正规方程

**定义与记号:** 多元线性回归 y=Xβ+ε(X是n×p设计矩阵,含截距列)。OLS估计 β̂=(X'X)⁻¹X'y。残差方差估计 σ̂²=RSS/(n-p)(RSS=Σ残差²,n-p是自由度,p是参数个数——呼应02类"自由度损失"的一般化版本)。系数协方差矩阵 Cov(β̂)=σ̂²(X'X)⁻¹,第j个系数的标准误SE(β̂ⱼ)=√[Cov(β̂)]ⱼⱼ。t统计量 t=β̂ⱼ/SE(β̂ⱼ),自由度n-p。

**一句话:** 回归系数的置信区间/假设检验和02-03类讲的框架完全同源——只是把"一个均值参数"换成了"一组回归系数",标准误公式的形式从σ/√n变成了更一般的矩阵形式σ̂²(X'X)⁻¹,本质逻辑不变。

**数学推导:** β̂=(X'X)⁻¹X'y 的推导来自最小化残差平方和 RSS(β)=(y-Xβ)'(y-Xβ),对β求梯度置零:∂RSS/∂β=-2X'(y-Xβ)=0 → X'Xβ=X'y → β̂=(X'X)⁻¹X'y(假设X'X可逆,即X列满秩、不存在完全共线性)。β̂的协方差推导:β̂=(X'X)⁻¹X'y=(X'X)⁻¹X'(Xβ+ε)=β+(X'X)⁻¹X'ε,所以Cov(β̂)=(X'X)⁻¹X'·Cov(ε)·X(X'X)⁻¹,在同方差假设Cov(ε)=σ²I下化简为 σ²(X'X)⁻¹——这正是知识点4提到的"同方差假设"在这个协方差公式推导里被直接用到的地方。

**底层机制/为什么这样设计:** (X'X)⁻¹这个矩阵的对角线元素,本质上衡量的是"这一列自变量的信息量有多独特"——如果某个自变量和其他自变量高度相关(接近共线性),(X'X)会接近奇异,其逆矩阵对角线元素会变得很大,对应系数的标准误会被急剧放大,这是"多重共线性会让系数估计变得不稳定"这个工程直觉的精确数学来源。

**AI研究/工程场景:** 特征工程阶段做线性探索性分析,判断哪些特征和目标变量有统计显著的独立贡献(控制住其他特征之后);解释一个已训练好的线性/广义线性模型时,报告"哪个系数的置信区间不包含0"是标准的统计推断步骤。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

n = 300
true_beta = np.array([2.0, 3.0, -1.5])  # [截距, x1系数, x2系数]

x1 = rng.uniform(0, 10, n)
x2 = rng.normal(5, 2, n)
X = np.column_stack([np.ones(n), x1, x2])
y = X @ true_beta + rng.normal(0, 2.0, n)

# 手写正规方程求解
beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
p = X.shape[1]
residuals = y - X @ beta_hat
sigma2_hat = np.sum(residuals ** 2) / (n - p)
cov_beta = sigma2_hat * np.linalg.inv(X.T @ X)
se_beta = np.sqrt(np.diag(cov_beta))
t_stats = beta_hat / se_beta
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p))

# 估计值应该接近真值 --- 用每个系数自己的标准误做尺度(4倍SE, 极宽松但仍有意义的界), 而不是所有系数共用一个flat容差:
# 截距的SE天然远大于斜率的SE(x1/x2的均值离0较远, 相当于要外推到原点, 02类"充分统计量/Fisher信息"框架下这是预期行为不是bug)
assert np.all(np.abs(beta_hat - true_beta) < 4 * se_beta), \
    f"beta_hat={beta_hat} should be within ~4 SE of true_beta={true_beta}, se={se_beta}"
# 三个系数都应该高度显著(真实效应都不小, 样本量n=300足够大)
assert all(pv < 0.001 for pv in p_values), f"all coefficients should be highly significant, got p-values={p_values}"

# 简单线性回归(只用x1)的情形, 和 scipy.stats.linregress 交叉验证
X_simple = np.column_stack([np.ones(n), x1])
beta_simple = np.linalg.solve(X_simple.T @ X_simple, X_simple.T @ y)
lr = stats.linregress(x1, y)
assert abs(beta_simple[1] - lr.slope) < 1e-8
assert abs(beta_simple[0] - lr.intercept) < 1e-8

print(f"beta_hat={np.round(beta_hat, 4)}  true_beta={true_beta}")
print(f"SE={np.round(se_beta, 4)}  p-values={[f'{pv:.2e}' for pv in p_values]}")
print(f"simple regression cross-check: manual slope={beta_simple[1]:.6f}  scipy slope={lr.slope:.6f}")
```

**面试怎么问+追问链**(决策依据追问轴 + 真实性验证轴):
- Q:"你的回归里有两个高度相关的自变量,一个系数的p值突然从显著变成不显著,发生了什么?"
- 追问1:"能不能从(X'X)⁻¹的角度解释一下?"(高度相关的两个自变量让X'X接近奇异,逆矩阵对应的对角线元素(标准误的来源)急剧膨胀,系数标准误变大,即使点估计的效应本身没变,t统计量=β̂/SE会因为SE变大而缩小,p值随之变大——这是多重共线性的精确数学机制,不是"玄学"或"运气不好")
- 深挖追问:"这种情况下,还能相信任何一个系数的点估计本身吗?"(点估计依然是无偏的(高斯-马尔可夫定理不要求自变量之间无关),但方差会很大——单次拟合的点估计可能离真值很远,这也是为什么共线性问题的实际后果是"结果不稳定"而不是"结果系统性错误")

**常见坑:**
- 只看单个系数的p值判断这个变量"有没有用",忽视多重共线性可能让本来有真实效应的变量p值也变得不显著(这不是变量真的没用,是共线性掩盖了它)。
- 把回归系数的显著性检验当成"变量重要性排序"的唯一依据,忽视了p值同时受样本量、变量尺度(标准化与否)影响,不是纯粹的"重要性"度量(呼应03类"统计显著不等于实际显著"里效应量的教训)。

---

## 6. 残差诊断 —— 数值化的Q-Q对比与简化异方差检验

**定义与记号:** Q-Q图(分位数-分位数图)对比样本分位数和理论分布分位数,本文用**数值化**版本代替(不画图,直接算偏差统计量):把残差排序,对比每个残差的经验分位数位置和标准正态分布对应分位数的理论值,偏差越小说明残差越接近正态。简化异方差检验(Breusch-Pagan思路):用残差平方对自变量做一次辅助回归,如果这个辅助回归本身有显著的解释力(R²明显不为0),说明残差方差随自变量系统性变化。

**一句话:** 残差诊断问的是"OLS的两个关键假设(正态性、同方差性)到底成立不成立",不是走个形式——04类知识点6"KS检验"、03类知识点7"参数检验的稳健性"讲的是"假设不成立会怎样",本知识点讲"怎么现场查出假设是不是真的不成立"。

**数学推导/说明:** Breusch-Pagan检验的核心思想:在原假设(同方差)下,残差平方 ê² 的期望应该是常数σ²,不应该能被任何自变量系统性预测;把 ê² 对原始自变量做一次辅助回归,如果这个辅助回归的判定系数R²显著不为0(用n·R²近似服从卡方分布这个渐进检验,或者简化成看辅助回归系数本身是否显著),说明残差方差确实随自变量变化,同方差假设不成立。

**底层机制/为什么这样设计:** 数值化的Q-Q对比(而不是画图目测)的必要性,和03类"用KS统计量代替目测正态性"是同一个纪律的延续——"看起来还行"不构成验证,只有把偏差量化成一个具体数字并设定合理阈值,才能在自动化流水线里可靠地判断假设是否成立。

**AI研究/工程场景:** 上线一个基于线性模型系数做业务决策的分析之前,残差诊断是标准的"模型体检"步骤——不满足关键假设的模型给出的置信区间/p值不可信,残差诊断能在这类问题造成业务误判之前先被发现。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

def qq_deviation(residuals):
    standardized = (residuals - residuals.mean()) / residuals.std()
    n = len(standardized)
    sample_quantiles = np.sort(standardized)
    theoretical_quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    # 用样本分位数和理论分位数之间的相关系数衡量"有多接近一条完美的y=x直线"(数值化的Q-Q对比)
    corr = np.corrcoef(sample_quantiles, theoretical_quantiles)[0, 1]
    return corr  # 越接近1, 残差越接近正态

def breusch_pagan_r2(x, squared_residuals):
    X_aux = np.column_stack([np.ones_like(x), x])
    beta_aux = np.linalg.solve(X_aux.T @ X_aux, X_aux.T @ squared_residuals)
    fitted = X_aux @ beta_aux
    ss_res = np.sum((squared_residuals - fitted) ** 2)
    ss_tot = np.sum((squared_residuals - squared_residuals.mean()) ** 2)
    return 1 - ss_res / ss_tot

n = 300
x = rng.uniform(1, 10, n)

# 情形A: 正态同方差误差(良好模型)
y_good = 2.0 + 3.0 * x + rng.normal(0, 2.0, n)
X_good = np.column_stack([np.ones(n), x])
beta_good = np.linalg.solve(X_good.T @ X_good, X_good.T @ y_good)
resid_good = y_good - X_good @ beta_good

# 情形B: 异方差误差(方差随x增大)
y_bad = 2.0 + 3.0 * x + rng.normal(0, 0.4 * x, n)
X_bad = np.column_stack([np.ones(n), x])
beta_bad = np.linalg.solve(X_bad.T @ X_bad, X_bad.T @ y_bad)
resid_bad = y_bad - X_bad @ beta_bad

qq_good = qq_deviation(resid_good)
qq_bad = qq_deviation(resid_bad)
bp_r2_good = breusch_pagan_r2(x, resid_good ** 2)
bp_r2_bad = breusch_pagan_r2(x, resid_bad ** 2)

# 核心断言1: 正态同方差情形下, Q-Q相关系数应该非常接近1(残差确实接近正态)
assert qq_good > 0.98, f"good model residuals should look very normal, QQ corr={qq_good:.4f}"

# 核心断言2: 异方差情形下, 简化Breusch-Pagan的R^2应该明显更高(残差平方能被x系统性预测)
assert bp_r2_bad > bp_r2_good * 5, f"heteroscedastic case BP-R2 ({bp_r2_bad:.4f}) should far exceed homoscedastic case ({bp_r2_good:.4f})"
assert bp_r2_bad > 0.1, f"heteroscedastic case should show clear predictability of squared residuals, got R2={bp_r2_bad:.4f}"

print(f"homoscedastic: QQ-corr={qq_good:.4f}  BP-R2={bp_r2_good:.4f}")
print(f"heteroscedastic: QQ-corr={qq_bad:.4f}  BP-R2={bp_r2_bad:.4f}")
```

**面试怎么问+追问链**(工程约束递增轴):
- Q:"你的分析流水线每天要自动拟合几百个回归模型,没办法每个都人工看残差图,怎么办?"
- 追问1:"能不能把'看图'这件事自动化?"(能——本知识点演示的正是把Q-Q图、异方差检验都转化成具体的数值统计量+阈值判断,这样才能在自动化流水线里批量运行,不依赖人工目测)
- 深挖追问:"如果自动化检测到某个模型残差异方差,流水线应该怎么处理,直接报错终止吗?"(视场景而定——可以自动切换到稳健标准误(知识点4已提到的White标准误)、或者自动记录一个"结果可信度降级"的标记供下游消费者参考,不一定要终止整个流水线,这里在考候选人是否有把统计诊断结果转化成具体工程决策的思路)

**常见坑:**
- 只做残差诊断不采取任何后续行动(诊断出异方差却依然直接使用原来的置信区间)。
- 把"数值化的Q-Q相关系数"或"简化BP检验的R²"当成绝对标准套用固定阈值,不考虑样本量对这些统计量本身抽样波动性的影响(样本量小时,即使残差真的正态,QQ相关系数也可能明显小于1,不能用同一个阈值不加区分地套用到所有样本量)。

---

## 7. 逻辑回归系数推断(Wald检验)—— 分类模型的系数置信区间怎么来

**定义与记号:** 逻辑回归模型 P(Y=1\|x)=σ(β₀+β₁x),σ(z)=1/(1+e⁻ᶻ)。MLE通过数值优化(牛顿法/IRLS)求解(没有闭式解)。Wald检验:z=β̂ⱼ/SE(β̂ⱼ),SE来自Fisher信息矩阵的逆——和02类知识点5"Cramér-Rao下界"、知识点7"MLE渐进正态性"是同一套理论在逻辑回归这个具体模型上的应用。

**一句话:** 逻辑回归系数没有像线性回归那样的解析解,但系数的置信区间/假设检验框架完全复用02类"MLE渐进正态+Fisher信息给标准误"这套通用理论——这正是02类要把"渐进正态性"作为独立知识点强调的原因:它不是线性回归的专属性质,是几乎所有MLE都共享的大样本性质。

**数学推导:** 逻辑回归的对数似然 ℓ(β)=Σᵢ[yᵢlog(pᵢ)+(1-yᵢ)log(1-pᵢ)],pᵢ=σ(xᵢ'β)。Score(梯度):∂ℓ/∂β=X'(y-p)。Hessian(用于Newton法):∂²ℓ/∂β∂β'=-X'WX,W=diag(pᵢ(1-pᵢ))。Newton-Raphson更新:β_new=β_old+(X'WX)⁻¹X'(y-p),重复迭代到收敛。收敛后的(X'WX)⁻¹(在β̂处求值)就是Fisher信息矩阵的逆,即渐进协方差矩阵Cov(β̂)——这个矩阵的推导结构和02类知识点5(单参数Fisher信息)完全同源,只是从标量推广到了矩阵形式。

**底层机制/为什么这样设计:** W=diag(pᵢ(1-pᵢ))这个权重矩阵值得注意——pᵢ(1-pᵢ)在pᵢ=0.5时最大(=0.25),在pᵢ接近0或1时趋于0,这意味着"预测概率接近0.5(模型本身很不确定的样本点)"对Fisher信息的贡献最大,"预测概率接近0或1(模型很确定的样本点)"贡献很小——这个权重结构和加权最小二乘(WLS)的形式完全一致,逻辑回归的Newton-Raphson迭代因此也被称为IRLS(迭代重加权最小二乘)。

**AI研究/工程场景:** 逻辑回归系数的显著性检验广泛用于风控/医疗等需要"可解释性"的分类场景——不只是要模型预测准,还要能回答"哪个特征对预测结果有统计显著的独立贡献",这是逻辑回归相比黑盒模型在这些场景里依然被广泛使用的原因之一。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

def fit_logistic_newton(X, y, n_iter=50, tol=1e-10):
    beta = np.zeros(X.shape[1])
    for _ in range(n_iter):
        p = sigmoid(X @ beta)
        W = p * (1 - p)
        gradient = X.T @ (y - p)
        # 加一个极小的对角正则避免W接近0导致数值奇异(收敛后期p接近0/1时可能出现)
        hessian = X.T @ (X * W[:, None]) + np.eye(X.shape[1]) * 1e-10
        delta = np.linalg.solve(hessian, gradient)
        beta = beta + delta
        if np.max(np.abs(delta)) < tol:
            break
    p_final = sigmoid(X @ beta)
    W_final = p_final * (1 - p_final)
    cov_beta = np.linalg.inv(X.T @ (X * W_final[:, None]) + np.eye(X.shape[1]) * 1e-10)
    se_beta = np.sqrt(np.diag(cov_beta))
    return beta, se_beta

n = 3000
true_beta = np.array([-1.0, 2.0])  # [截距, x系数]
x = rng.normal(0, 1.5, n)
X = np.column_stack([np.ones(n), x])
p_true = sigmoid(X @ true_beta)
y = rng.binomial(1, p_true)

beta_hat, se_beta = fit_logistic_newton(X, y)
z_stats = beta_hat / se_beta
p_values = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))  # Wald检验用正态近似(渐进性质)

# 估计值应该接近真值(n=3000足够大, MLE渐进一致性生效)
assert np.allclose(beta_hat, true_beta, atol=0.25), f"beta_hat={beta_hat} should be close to true_beta={true_beta}"
# 两个系数都应该高度显著(真实效应明显, 样本量充足)
assert all(pv < 0.001 for pv in p_values), f"coefficients should be highly significant, got p={p_values}"

# 用重复抽样验证Wald检验给出的SE确实接近真实的抽样标准差(呼应02类Cramer-Rao下界的验证范式)
n_repeats = 300
beta1_estimates = np.empty(n_repeats)
for i in range(n_repeats):
    x_r = rng.normal(0, 1.5, n)
    X_r = np.column_stack([np.ones(n), x_r])
    y_r = rng.binomial(1, sigmoid(X_r @ true_beta))
    beta_r, _ = fit_logistic_newton(X_r, y_r)
    beta1_estimates[i] = beta_r[1]

empirical_se = beta1_estimates.std()
assert abs(empirical_se - se_beta[1]) / se_beta[1] < 0.25, \
    f"Wald SE ({se_beta[1]:.4f}) should roughly match empirical sampling SE ({empirical_se:.4f})"

print(f"beta_hat={np.round(beta_hat, 4)}  SE={np.round(se_beta, 4)}  true_beta={true_beta}")
print(f"Wald SE for x-coefficient={se_beta[1]:.4f}  empirical (repeated-sampling) SE={empirical_se:.4f}")
```

**面试怎么问+追问链**(规模递增轴,呼应02类知识点7):
- Q:"逻辑回归系数的p值是怎么算出来的,和线性回归系数的p值算法一样吗?"
- 追问1:"具体哪里一样,哪里不一样?"(不一样的是:线性回归系数的t分布是精确成立的(正态误差假设下），逻辑回归的Wald检验用的是**渐进**正态近似(02类知识点7),小样本下这个近似可能不准;一样的是:两者标准误的来源都是"参数协方差矩阵的对角线开根号",本质结构相同)
- 深挖追问:"小样本、且某个类别的正例数很少(比如y=1只有5个样本)时,Wald检验还可靠吗?"(不可靠——这是逻辑回归里一个真实存在的问题,称为"完全分离"或者稀有事件下Wald检验的已知失效场景,这时候似然比检验(比较有/无该变量的两个模型对数似然)通常比Wald检验更可靠,这是一个连很多从业者都不知道、但反映真正理解MLE推断局限性的深挖追问)

**常见坑:**
- 把逻辑回归系数当成线性回归系数一样直接解读为"x每增加1单位,y增加多少"——逻辑回归系数是对数几率(log-odds)的线性效应,不是概率本身的线性效应,解读时需要通过sigmoid函数转换,是一个非线性关系。
- 忽视Wald检验在小样本/稀有事件/类别分离场景下的已知不可靠性,盲目信任软件包默认输出的p值。

---

下一篇:[06-ab-test-design-and-power.md](06-ab-test-design-and-power.md) —— 板块II的开篇,把03类的功效理论具体落地成A/B测试的样本量计算。
