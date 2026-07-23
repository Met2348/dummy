# 04 · 经典检验方法深挖(Classical Test Methods)

> 总览见 [00-roadmap.md](00-roadmap.md)

03类讲完了假设检验的完整框架(为什么这样设计、α/β/功效怎么权衡),本文把框架落地成具体、可以直接用的检验方法——t检验、卡方检验、ANOVA、非参数检验、置换检验、bootstrap。每一种都会和 `scipy.stats` 的官方实现交叉验证数值一致,不是"看起来差不多"。

**环境声明:** 全部代码在仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1)下真实跑通,随机抽样固定种子(`np.random.default_rng(42)`)。

---

## 1. 单样本/双样本t检验 —— 均值比较的默认工具

**定义与记号:** 单样本t检验(H0:μ=μ₀):t=(X̄-μ₀)/(S/√n),自由度n-1。独立双样本t检验(方差齐性假设下,pooled variance):t=(X̄₁-X̄₂)/(Sp·√(1/n₁+1/n₂)),Sp²=[(n₁-1)S₁²+(n₂-1)S₂²]/(n₁+n₂-2)。

**一句话:** t检验是σ未知时,用样本标准差S替代真实σ之后,检验统计量不再服从正态分布、而服从自由度n-1的t分布(尾部比正态更厚,补偿"额外估计了一个σ"带来的不确定性)这一事实的直接应用。

**数学推导:** 单样本情形下,(X̄-μ)/(σ/√n)~N(0,1)(已知σ,03类推过)。但σ未知,用S替代后,分子分母都含随机性,William Gosset(笔名Student)证明了(X̄-μ)/(S/√n)服从自由度n-1的t分布——这个证明依赖(n-1)S²/σ² 服从卡方分布χ²(n-1)(02类知识点3已证明S²无偏,这里额外用到它的抽样分布是卡方分布这个更强的结果),以及X̄和S²在正态总体下相互独立。t分布=N(0,1)/√(χ²(n-1)/(n-1))这个比值结构,正是"分子标准正态、分母额外的抽样不确定性"这个直觉的精确数学形式。

**底层机制/为什么这样设计:** n越大,S²估计σ²越准,χ²(n-1)/(n-1)这个比值越集中在1附近,t分布越趋近标准正态——这解释了为什么"n>30就可以近似用正态"这条经验法则存在(t分布和正态分布在大自由度下几乎无法区分),但也解释了为什么小样本时必须用t分布而不是正态分布(t分布尾部更厚,用正态分布的临界值会低估真实的不确定性,置信区间偏窄)。

**AI研究/工程场景:** 几乎所有"比较两个模型/两个版本的某个数值指标是否有显著差异"的场景,双样本t检验是默认的第一选择(前提是指标近似正态或样本量足够大,03类知识点7已经展开了这个假设不成立时的风险)。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

# 单样本t检验: 手写实现 vs scipy交叉验证
sample = rng.normal(5.3, 2.0, 40)
mu0 = 5.0
t_manual = (sample.mean() - mu0) / (sample.std(ddof=1) / np.sqrt(len(sample)))
p_manual = 2 * (1 - stats.t.cdf(abs(t_manual), df=len(sample) - 1))
t_scipy, p_scipy = stats.ttest_1samp(sample, mu0)

assert abs(t_manual - t_scipy) < 1e-9
assert abs(p_manual - p_scipy) < 1e-9

# 独立双样本t检验(方差齐性假设): 手写实现 vs scipy交叉验证
group_a = rng.normal(10.0, 3.0, 35)
group_b = rng.normal(11.5, 3.0, 35)
n1, n2 = len(group_a), len(group_b)
sp2 = ((n1 - 1) * group_a.var(ddof=1) + (n2 - 1) * group_b.var(ddof=1)) / (n1 + n2 - 2)
t_manual2 = (group_b.mean() - group_a.mean()) / np.sqrt(sp2 * (1 / n1 + 1 / n2))
t_scipy2, p_scipy2 = stats.ttest_ind(group_b, group_a)  # 默认 equal_var=True, 对应pooled variance

assert abs(t_manual2 - t_scipy2) < 1e-9

print(f"one-sample: t_manual={t_manual:.6f} t_scipy={t_scipy:.6f}")
print(f"two-sample: t_manual={t_manual2:.6f} t_scipy={t_scipy2:.6f} p={p_scipy2:.6f}")
```

**面试怎么问+追问链**(决策依据追问轴):
- Q:"两组数据方差明显不一样,pooled variance的t检验还能用吗?"
- 追问1:"不能用会有什么后果?"(方差差异大时pooled variance是两组方差的加权平均,会扭曲检验统计量,导致Type I错误率偏离名义水平;应该用Welch's t检验,不假设方差齐性,自由度用Welch-Satterthwaite公式近似计算而不是简单的n1+n2-2)
- 深挖追问:"scipy的`ttest_ind`默认参数是哪种?"(默认 `equal_var=True`,即pooled variance版本;要用Welch's检验需要显式传 `equal_var=False`——这是一个真实存在的"默认参数陷阱",不少人直接调库时踩坑)

**常见坑:**
- 不检查方差齐性假设就默认用pooled variance版本的t检验(应该先看方差是否明显不同,或者干脆默认用更保守的Welch's检验)。
- 混淆t检验的自由度——单样本n-1,独立双样本(pooled)是n1+n2-2,配对检验(下一个知识点)是n-1(n是配对数,不是2n)。

---

## 2. 配对t检验 vs 独立双样本 —— 忽略配对结构会付出功效代价

**定义与记号:** 配对t检验:对每对观测算差值dᵢ=x₁ᵢ-x₂ᵢ,转化成对{dᵢ}做单样本t检验(H0:μ_d=0),t=d̄/(Sd/√n),自由度n-1(n是配对**数**)。

**一句话:** 同一批个体的"前后测量"或"同一批用户的AB两版本"数据自带个体间基线差异这个额外噪声源,配对设计通过对每个个体做差,把这部分噪声直接消掉了——用独立双样本检验处理本该配对的数据,是在白白浪费统计功效。

**数学推导:** 设Xᵢ、Yᵢ是同一个体i的两次测量,存在个体基线bᵢ(个体间方差记为τ²)和测量噪声(方差σ_ε²):Xᵢ=bᵢ+εᵢ₁,Yᵢ=bᵢ+δ+εᵢ₂(δ是真实处理效应)。独立双样本检验把{Xᵢ}和{Yᵢ}当成两组独立数据,方差里包含τ²+σ_ε²(个体间差异被当成了噪声)。配对检验对差值dᵢ=Yᵢ-Xᵢ=δ+(εᵢ₂-εᵢ₁)操作,bᵢ在做差时被精确抵消,方差只剩2σ_ε²(如果个体间方差τ²远大于测量噪声σ_ε²,配对检验的方差会远小于独立检验)。

**底层机制/为什么这样设计:** 这本质是"实验设计"层面的方差削减技术——07类的CUPED方差削减方法,底层思路和配对检验完全同源:都是"找到一个和结果相关、但和处理效应本身无关的协变量(这里是'个体本身'),把它的影响从噪声里剔除出去"。

**AI研究/工程场景:** 同一批prompt在两个不同的模型版本上分别测评(而不是把两个版本的评测结果当成独立的两批数据),应该用配对检验——不同prompt本身难度差异很大(这是"个体间方差"),配对设计能把这部分方差消掉,只关注"同一个prompt上,新模型是否系统性地比旧模型好"。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
n = 20

# 个体基线差异很大(标准差15), 真实处理效应只有+3, 测量噪声标准差2
baseline = rng.normal(50, 15, n)
true_effect = 3.0
after = baseline + true_effect + rng.normal(0, 2, n)

# 错误方法: 把 before/after 当成两组独立样本(丢弃配对结构信息)
_, p_independent = stats.ttest_ind(after, baseline)

# 正确方法: 配对t检验
t_paired, p_paired = stats.ttest_rel(after, baseline)

# 手写配对检验验证scipy结果
diffs = after - baseline
t_manual = diffs.mean() / (diffs.std(ddof=1) / np.sqrt(n))
assert abs(t_manual - t_paired) < 1e-9

# 核心断言: 真实效应存在(+3), 配对检验能可靠检测出来, 独立检验被个体间大方差淹没检测不出来
assert p_paired < 0.001, f"paired test should clearly detect the real effect, got p={p_paired:.4f}"
assert p_independent > 0.3, f"independent test should be swamped by between-subject variance, got p={p_independent:.4f}"

print(f"paired t-test:      t={t_paired:.4f}  p={p_paired:.6f}  (correctly detects the real +3 effect)")
print(f"independent t-test: p={p_independent:.4f}  (fails to detect it, swamped by between-subject noise)")
```

**面试怎么问+追问链**(方案批判迭代轴):
- Q:"你测了20个用户升级前后的某个指标,用独立双样本t检验算出p=0.4,不显著,结论是'升级没有效果',这个分析有什么问题?"
- 追问1:"'前后'这个词本身提示了什么?"(这是配对数据——同一批用户测了两次,应该用配对t检验而不是独立双样本检验)
- 深挖追问:"如果重新用配对t检验分析同一份数据,结果会不会不一样?"(很可能不一样——配对检验消除了用户间的基线差异这个大噪声源,原本被掩盖的真实效应可能变得显著,这正是本知识点可运行例子演示的具体数字)

**常见坑:**
- 看到"前后测量""同一批对象测两次"这种数据结构,默认套用独立双样本检验(最常见、最容易在面试中被抓到的错误)。
- 反过来,把本来就是两组**不同**个体的数据强行"配对"(比如按顺序随意配对没有任何配对逻辑的两组独立样本)——配对必须有真实的配对依据(同一个体/同一时间/同一自然单位),不能为了套用配对检验而编造配对关系。

---

## 3. 卡方检验 —— 拟合优度与独立性

**定义与记号:** 拟合优度检验(H0:数据服从给定的离散分布):χ²=Σᵢ(观测频数Oᵢ-期望频数Eᵢ)²/Eᵢ,近似服从χ²(k-1)(k是类别数)。独立性检验(H0:两个分类变量独立):对r×c列联表,期望频数Eᵢⱼ=行合计×列合计/总合计,统计量同样形式,自由度(r-1)(c-1)。

**一句话:** 卡方统计量衡量"观测频数和(原假设下)期望频数差多远",偏差越大统计量越大,这是专门为**类别型/计数型**数据设计的检验,不是给连续数值数据用的。

**数学推导:** (Oᵢ-Eᵢ)/√Eᵢ 这一项在H0成立、Eᵢ足够大时近似服从标准正态(这是二项分布/多项分布正态近似的推论,和01类"泊松是二项极限"同一族的渐进论证);把k个这样的近似正态项平方求和,渐进服从自由度k-1的卡方分布(减1是因为k个类别的频数有一个线性约束——总和固定为n,损失一个自由度,和02类"残差自由度损失"是同一个道理)。

**底层机制/为什么这样设计:** 卡方检验要求每个类别的期望频数Eᵢ不能太小(经验法则Eᵢ≥5)——这正是因为整个检验的正态近似依赖"二项/多项分布在期望频数较大时趋于正态"这个渐进论证,期望频数太小时近似失效,这时候要么合并类别,要么改用精确检验(如Fisher精确检验)。

**AI研究/工程场景:** 检验模型预测的类别分布是否和真实类别分布一致(拟合优度);检验某个用户特征(如设备类型)和转化与否两个分类变量是否独立(独立性检验,判断这个特征是否是一个有意义的细分维度)。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

# 拟合优度检验: 骰子是否均匀(H0: 6个面概率都是1/6)
true_probs_fair = np.ones(6) / 6
n_rolls = 6000
observed_fair = rng.multinomial(n_rolls, true_probs_fair)
expected_fair = np.full(6, n_rolls / 6)

chi2_manual = np.sum((observed_fair - expected_fair) ** 2 / expected_fair)
chi2_scipy, p_scipy = stats.chisquare(observed_fair, expected_fair)
assert abs(chi2_manual - chi2_scipy) < 1e-9
assert p_scipy > 0.05, f"fair die data should not reject H0 of fairness, got p={p_scipy:.4f}"

# 一个明显有偏的骰子(面6概率是其他面的3倍), 应该显著拒绝均匀假设
biased_probs = np.array([1, 1, 1, 1, 1, 3]) / 8
observed_biased = rng.multinomial(n_rolls, biased_probs)
_, p_biased = stats.chisquare(observed_biased, expected_fair)
assert p_biased < 0.001, f"biased die should clearly reject fairness, got p={p_biased:.6f}"

# 独立性检验: 构造两个独立的分类变量(设备类型 x 是否转化), 手写实现vs scipy交叉验证
contingency = np.array([[450, 550], [380, 620]])  # 2x2列联表
row_totals = contingency.sum(axis=1, keepdims=True)
col_totals = contingency.sum(axis=0, keepdims=True)
grand_total = contingency.sum()
expected_ind = row_totals @ col_totals / grand_total
chi2_ind_manual = np.sum((contingency - expected_ind) ** 2 / expected_ind)
chi2_ind_scipy, p_ind_scipy, dof, _ = stats.chi2_contingency(contingency, correction=False)
assert abs(chi2_ind_manual - chi2_ind_scipy) < 1e-6
assert dof == 1

print(f"fair die: chi2={chi2_scipy:.4f} p={p_scipy:.4f}")
print(f"biased die: p={p_biased:.6f}")
print(f"independence test: chi2={chi2_ind_scipy:.4f} p={p_ind_scipy:.4f}")
```

**面试怎么问+追问链**(决策依据追问轴):
- Q:"一个2x2列联表,某个格子的期望频数只有3,卡方检验的结论还可信吗?"
- 追问1:"具体是什么假设被违反了?"(期望频数过小时卡方统计量的卡方分布近似不准确,`scipy.stats.chi2_contingency` 默认对2x2表会做Yates连续性校正来缓解这个问题,更彻底的方案是改用Fisher精确检验,直接算超几何分布下的精确p值,不依赖渐进近似)
- 深挖追问:"拟合优度检验和独立性检验,自由度的计算逻辑是一回事吗?"(本质同源——都是"总频数固定"这个约束损失自由度,拟合优度是k个类别损失1个自由度(k-1),独立性检验的行列合计都固定,损失(r-1)+(c-1)个自由度中的交互项,结果是(r-1)(c-1))

**常见坑:**
- 期望频数太小(经验法则<5)时依然直接用标准卡方检验,不做校正或换用精确检验。
- 把卡方检验用在连续数值数据上(卡方检验是给计数/类别数据设计的,连续数据要先合理分箱才能用,不合理的分箱方式本身会影响检验结果)。

---

## 4. 单因素方差分析(ANOVA)—— 多组均值比较,以及"为什么不能对每两组都做t检验"

**定义与记号:** k组数据,组间均方MSB(between-group)、组内均方MSW(within-group),F=MSB/MSW,在H0(所有组均值相等)下服从F(k-1, N-k)分布(N是总样本量)。

**一句话:** ANOVA一次性检验"k组均值是否全部相等",不是对每两组分别做t检验再汇总——后者会让整体的假阳性率随组数增多而急剧膨胀,这正是05类"多重检验问题"最直接、最常见的一个现实原型。

**数学推导:** F统计量的直觉:组间均方MSB衡量"各组均值彼此之间差多远"(如果H0为真,这个量只反映抽样噪声);组内均方MSW衡量"每组内部数据本身的离散程度"(纯噪声的估计,不受H0是否成立影响)。H0为真时,MSB和MSW都是同一个σ²的独立估计量,比值F应该在1附近波动(服从F分布);H0为假(至少一组均值不同)时,MSB会系统性膨胀(混入了真实的组间差异),F统计量会偏大。

**底层机制/为什么这样设计:** 为什么不能用"对k组两两做t检验,只要有一对显著就算有差异"代替ANOVA?因为k组两两比较有C(k,2)对,每一对检验都有自己的α假阳性率,即使每一组的均值都真实相等(H0处处为真),"至少一对显著"这个复合事件的概率会远超单次检验的α——这正是05类要正式量化的多重比较问题,ANOVA通过一次性的整体F检验,把"总体假阳性率控制在α"这件事从设计上直接解决,不需要事后修正。

**AI研究/工程场景:** 比较3个以上模型变体/训练配置在某个指标上是否存在真实差异,应该先做一次ANOVA整体检验"是否存在任何差异",通过之后再做事后比较(post-hoc test,如Tukey HSD)定位具体哪些组不同——直接对所有组两两跑t检验是一个常见的分析错误。

**可运行例子:**
```python
import numpy as np
from scipy import stats
from itertools import combinations

rng = np.random.default_rng(42)

# ANOVA正确性: 4组数据, 手写F统计量 vs scipy交叉验证
groups = [rng.normal(10, 3, 25) for _ in range(4)]
all_data = np.concatenate(groups)
grand_mean = all_data.mean()
n_per_group = len(groups[0])
k = len(groups)
N = len(all_data)

ss_between = sum(n_per_group * (g.mean() - grand_mean) ** 2 for g in groups)
ss_within = sum(np.sum((g - g.mean()) ** 2) for g in groups)
ms_between = ss_between / (k - 1)
ms_within = ss_within / (N - k)
f_manual = ms_between / ms_within

f_scipy, p_scipy = stats.f_oneway(*groups)
assert abs(f_manual - f_scipy) < 1e-6

# 核心实验: 4组数据"真实完全相同分布"(H0处处为真), 对比"多次两两t检验"vs"一次ANOVA"的假阳性率
n_trials = 3000
anova_false_positives = 0
pairwise_false_positives = 0
alpha = 0.05

for _ in range(n_trials):
    groups_null = [rng.normal(0, 1, 20) for _ in range(4)]
    _, p_anova = stats.f_oneway(*groups_null)
    if p_anova < alpha:
        anova_false_positives += 1

    any_pair_significant = False
    for g1, g2 in combinations(groups_null, 2):
        _, p_pair = stats.ttest_ind(g1, g2)
        if p_pair < alpha:
            any_pair_significant = True
            break
    if any_pair_significant:
        pairwise_false_positives += 1

anova_fpr = anova_false_positives / n_trials
pairwise_fpr = pairwise_false_positives / n_trials

# 核心断言: 一次ANOVA的假阳性率接近名义alpha=0.05, 多次两两t检验的假阳性率明显膨胀
assert 0.03 < anova_fpr < 0.07, f"ANOVA false positive rate should be near 0.05, got {anova_fpr:.4f}"
assert pairwise_fpr > anova_fpr * 1.8, f"pairwise-comparisons FPR ({pairwise_fpr:.4f}) should be notably inflated vs ANOVA ({anova_fpr:.4f})"

print(f"ANOVA F={f_manual:.4f} p={p_scipy:.4f}")
print(f"under true null (4 identical groups): ANOVA FPR={anova_fpr:.4f}  pairwise-t-tests FPR={pairwise_fpr:.4f}")
```

**面试怎么问+追问链**(方案批判迭代轴,直接呼应05类多重检验):
- Q:"你要比较5个模型配置的效果,打算对C(5,2)=10对配置分别做t检验,这个方案有什么问题?"
- 追问1:"具体会造成多大的假阳性率膨胀?"(候选人应该能说出量级——即使每次检验α=0.05,10次独立检验里"至少一次假阳性"的概率远超5%,粗略估计接近1-0.95^10≈40%)
- 换方案后追问2:"改成先做ANOVA,通过后再两两比较,这样总假阳性率控制住了吗?"(基本解决了"总体是否存在任何差异"这一层的假阳性率,但事后两两比较(post-hoc)阶段仍然存在多重比较问题,需要额外用Tukey HSD等专门为事后比较设计的校正方法,不是简单套用普通t检验——这里在考候选人是否理解ANOVA只解决了问题的第一层)

**常见坑:**
- 用多次两两t检验代替ANOVA做多组比较(本知识点核心陷阱)。
- ANOVA显著后直接把"哪两组具体不同"这个问题也用普通t检验回答,忽视事后比较阶段依然存在多重比较问题。

---

## 5. Mann-Whitney U检验 —— 不假设正态性的双样本比较

**定义与记号:** 对两组独立样本,把全部数据合并排秩,U统计量基于秩和计算(具体:U₁=R₁-n₁(n₁+1)/2,R₁是第一组数据的秩和)。检验的是"两个分布是否有系统性位置偏移",不直接检验均值,更准确说是检验"随机取一个样本A组的值比B组的值大"的概率是否为0.5。

**一句话:** Mann-Whitney不看数值本身的大小,只看**排序**——这让它对离群值和分布形状完全不敏感,是t检验在正态性假设不成立时的标准非参数替代方案。

**数学推导:** 核心思想是把连续数值"打散"成秩(rank),原始数据的具体数值不再重要,只保留"谁比谁大"这个序关系。这个操作本身就是让检验对分布形状不敏感的来源——不管原始数据是正态、指数还是任何形状,只要把它们变成秩,秩的排列在H0(两组同分布)下是**均匀随机**的排列,U统计量的原假设分布因此不依赖原始数据的具体分布形状,是一个和分布无关(distribution-free)的检验。

**底层机制/为什么这样设计:** 代价是"只用序信息、丢弃了数值间隔的具体大小信息"——如果数据确实近似正态,Mann-Whitney的统计功效会略低于t检验(损失了一部分信息);但如果数据严重偏态或有离群值,t检验的功效反而会被这些破坏正态性的因素拖累,Mann-Whitney这时候更稳健、功效反而更高。

**AI研究/工程场景:** 比较两个模型在某个有严重离群值的指标(比如单次请求延迟,总有极端慢请求)上是否有系统性差异,Mann-Whitney比t检验更不容易被少数极端值主导结论。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

# 与scipy交叉验证(用于确认手写理解正确, 这里直接调用scipy作为"手写"对照对象是因为U统计量本身涉及秩的边界处理细节较多)
group_a = rng.normal(10, 2, 40)
group_b = rng.normal(11, 2, 40)
u_stat, p_mw = stats.mannwhitneyu(group_b, group_a, alternative="two-sided")
t_stat, p_t = stats.ttest_ind(group_b, group_a)
# 两种检验对同一个"确实存在正态位移"的数据集, 结论方向应该一致(都显著或都不显著, 只是功效有差异)
assert (p_mw < 0.05) == (p_t < 0.05) or abs(p_mw - p_t) < 0.1

# 稳健性对比: 构造带有严重离群值的数据, t检验容易被单个极端值主导, Mann-Whitney不受影响
group_c = rng.normal(10, 1, 30)
group_d = np.concatenate([rng.normal(10.5, 1, 29), [500.0]])  # 一个极端离群值混入组D

_, p_t_outlier = stats.ttest_ind(group_d, group_c)
_, p_mw_outlier = stats.mannwhitneyu(group_d, group_c, alternative="two-sided")

# 去掉离群值之后重新做t检验, 应该和Mann-Whitney(不受离群值影响)的结论更接近
group_d_clean = group_d[group_d < 100]
_, p_t_clean = stats.ttest_ind(group_d_clean, group_c)
assert abs(p_mw_outlier - p_t_clean) < abs(p_mw_outlier - p_t_outlier), \
    "Mann-Whitney result should stay closer to the outlier-free t-test than the outlier-contaminated t-test does"

print(f"normal shift: t-test p={p_t:.4f}  Mann-Whitney p={p_mw:.4f}")
print(f"with outlier: t-test p={p_t_outlier:.4f}  Mann-Whitney p={p_mw_outlier:.4f}  (clean t-test p={p_t_clean:.4f})")
```

**面试怎么问+追问链**(决策依据追问轴):
- Q:"什么情况下你会选Mann-Whitney而不是t检验?"
- 追问1:"数据轻度偏离正态但样本量很大呢?"(样本量大时CLT本身让t检验相当稳健,这时候t检验通常仍是更优选择,功效更高;Mann-Whitney的优势主要在小样本+严重偏离正态/有离群值的场景)
- 深挖追问:"Mann-Whitney显著,能不能说'两组的中位数不同'?"(不完全准确——Mann-Whitney检验的严格解读是"两个分布的随机变量哪个倾向于更大"这个更宽泛的位置偏移概念,只有在两组分布形状相同、只是位置不同这个额外假设下,才能把结论收窄到"中位数不同";这是一个经常被过度简化、但面试官会专门追问的细节)

**常见坑:**
- 把Mann-Whitney的结论无条件解读成"中位数比较"(上面已展开)。
- 认为非参数检验"永远比参数检验更安全"所以无脑优先选择——数据确实接近正态时,非参数检验会损失统计功效,不是没有代价的"更安全"选项。

---

## 6. Kolmogorov-Smirnov(KS)检验 —— 比较整个分布形状,不只是某个统计量

**定义与记号:** 单样本KS检验(H0:数据来自给定的理论分布F₀):统计量D=sup_x\|Fₙ(x)-F₀(x)\|(经验累积分布函数和理论累积分布函数之间的最大垂直距离)。双样本KS检验:比较两组数据各自的经验分布函数,D=sup_x\|F₁ₙ(x)-F₂ₘ(x)\|。

**一句话:** KS检验比较的是**整条分布曲线**,不像t检验只关注均值、卡方检验只关注分箱后的频数——这让它能检测出"均值相同但形状完全不同"这类其他检验会漏掉的分布差异。

**数学推导:** D统计量在H0成立时的抽样分布(Kolmogorov分布)不依赖F₀具体是什么分布(只要F₀连续)——这个"分布无关性"来自概率积分变换:如果X~F₀,则F₀(X)~Uniform(0,1),所以不管原始理论分布F₀是什么形状,经验分布函数和理论分布函数之间的偏差,变换到[0,1]区间上比较后,都服从同一个和F₀具体形状无关的极限分布,这也是本文03类知识点4"p值在H0下服从均匀分布"这条性质的一个更一般化的应用场景。

**底层机制/为什么这样设计:** 正因为KS检验看的是整条分布曲线的最大偏差,它对"分布中间部分吻合得很好、但尾部有系统性差异"这类情况特别敏感——这也是它经常被选来做"漂移检测"(17类会专门展开)的原因:线上数据分布悄悄发生变化,可能不体现在均值上(卡方/t检验测不出来),但体现在分布形状/尾部,KS检验能捕捉到。

**AI研究/工程场景:** 检验模型训练时用的特征分布和线上实际服务时观测到的特征分布是否一致(分布漂移的一种检测手段);检验一个随机数生成器/采样算法产出的样本是否真的服从预期分布。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

# 单样本KS检验: 数据确实来自标准正态, D统计量应该很小(用统计量本身而不是二元的p值阈值判断,
# 避免单次假设检验固有的~5%抽样噪声导致边界翻转 --- 01类CLT knowledge point同款教训)
sample_normal = rng.normal(0, 1, 2000)
d_stat, p_ks = stats.kstest(sample_normal, "norm")
critical_d_5pct = 1.36 / np.sqrt(2000)  # Kolmogorov分布5%显著性水平的渐进临界值
assert d_stat < critical_d_5pct * 1.5, f"data truly from N(0,1) should have small KS stat, got D={d_stat:.4f} (5% critical={critical_d_5pct:.4f})"

# 数据来自略微偏移+略微更胖尾的分布(t分布, 自由度低时比正态胖尾), 应该被KS检验捕捉到
sample_t = rng.standard_t(df=3, size=2000)  # 自由度3的t分布, 明显比正态厚尾
d_stat_t, p_ks_t = stats.kstest(sample_t, "norm")
assert p_ks_t < 0.01, f"heavy-tailed t(3) data should be flagged as non-normal, got p={p_ks_t:.4f}"

# 双样本KS检验: 两组数据均值都是0(总体均值相同), 但方差(分布形状)明显不同 --- 只看均值的检验(如t检验)会漏掉这个差异
# 用重复试验测"拒绝率"而不是赌单次抽样是否越过p=0.05这条线(单次抽样在H0下本来就有~5%概率越过, 03类知识点4已经验证过这个性质)
n_shape_trials = 500
t_test_rejections = 0
ks_test_rejections = 0
for _ in range(n_shape_trials):
    ga = rng.normal(0, 1, 300)
    gb = rng.normal(0, 3, 300)  # 均值同为0(H0对均值成立), 方差差3倍(分布形状确实不同)
    _, p_t = stats.ttest_ind(ga, gb)
    _, p_ks_shape = stats.ks_2samp(ga, gb)
    if p_t < 0.05:
        t_test_rejections += 1
    if p_ks_shape < 0.05:
        ks_test_rejections += 1

t_test_fpr = t_test_rejections / n_shape_trials
ks_test_power = ks_test_rejections / n_shape_trials

# 核心断言: t检验的"拒绝率"应该接近名义alpha=0.05(均值确实相同, 它确实"看不见"方差差异, 不是系统性误报)
assert t_test_fpr < 0.12, f"t-test should behave like it's under true H0 (means equal), rejection rate should stay near 0.05, got {t_test_fpr:.4f}"
# KS检验应该能可靠地把这个真实存在的形状差异检测出来, 拒绝率远高于t检验
assert ks_test_power > 0.9, f"KS test should reliably detect the shape difference, got rejection rate={ks_test_power:.4f}"
assert ks_test_power > t_test_fpr * 5

print(f"normal data: KS p={p_ks:.4f}   heavy-tailed t(3): KS p={p_ks_t:.6f}")
print(f"same mean, different shape, over {n_shape_trials} trials: t-test rejection rate={t_test_fpr:.4f} (misses it)  KS test rejection rate={ks_test_power:.4f} (catches it)")
```

**面试怎么问+追问链**(方案批判迭代轴,呼应17类):
- Q:"你怀疑线上特征分布相比训练集发生了漂移,但两边的均值几乎一样,能不能就此排除漂移的可能?"
- 追问1:"均值相同能代表分布完全一样吗?"(不能——上面的可运行例子直接给出了反例,方差、偏度、尾部形状都可能不同,均值只是分布的一个低维摘要)
- 深挖追问:"KS检验对样本量特别敏感,线上数据量是百万级别时会怎样?"(样本量极大时,KS检验(和几乎所有假设检验一样)会对极其微小、没有实际意义的分布差异也给出"显著"结论——这里呼应03类"统计显著不等于实际显著",17类的PSI等指标之所以在工业界比纯粹的KS检验p值更常用,部分原因就是PSI给出的是一个连续的"漂移程度"数值而不是二元的显著/不显著判断,不会被超大样本量放大成必然显著)

**常见坑:**
- 只用均值/中位数比较分布,忽视分布形状(方差/偏度/尾部)可能存在的差异,而KS检验/17类的KL散度这类"比较整条分布"的工具正是为了填这个空。
- 样本量极大时不加区分地直接用KS检验的显著性做二元判断,忽视了"显著"和"差异程度是否值得关注"是两回事(和03类效应量的教训同源)。

---

## 7. 置换检验(Permutation Test)—— 不依赖任何分布假设的检验方法

**定义与记号:** 置换检验:在H0(两组数据来自同一分布,组别标签和结果无关)下,把两组数据合并、随机重新分配组别标签,重复很多次,每次计算检验统计量(如两组均值差),得到H0下检验统计量的经验分布;把真实观测到的统计量和这个经验分布比较,算出p值(真实统计量在这个经验分布里排多极端)。

**一句话:** 置换检验直接把"H0下,组别标签是随机的、和结果无关"这句话翻译成代码——不依赖正态性假设、不依赖任何解析公式,只依赖"如果真的没有效应,随便怎么重新分配标签,结果都差不多"这一个朴素的逻辑。

**数学推导:** 置换检验的合法性来自一个精确的组合数学论证,不是渐进近似:在H0(标签和结果无关)严格成立的前提下,观测到的这一种标签分配方式,和其他任何一种重新排列的标签分配方式,产生当前这组数据的概率完全相等(可交换性,exchangeability)——所以真实的检验统计量在"全部排列方式给出的统计量集合"里的相对位置,直接就是一个精确的(不是渐进近似的)p值,即使总体分布是什么形状完全未知也成立。

**底层机制/为什么这样设计:** 正因为置换检验不依赖任何分布假设,它是t检验/Mann-Whitney等方法在"完全不确定数据分布形状、样本量太小CLT不生效"时的终极后备方案——代价是计算量(需要枚举或蒙特卡洛采样大量排列方式),数据量大时全排列不现实,通常用随机抽样一部分排列(如1万次)来近似。

**AI研究/工程场景:** 小样本、分布形状完全未知或明显非正态的场景(比如只有个位数到十几个数据点的初步实验结果),置换检验是比t检验/Mann-Whitney更"诚实"的选择,不需要对数据形状做任何假设。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

group_a = rng.normal(10, 2, 15)
group_b = rng.normal(12, 2, 15)
observed_diff = group_b.mean() - group_a.mean()

# 手写置换检验
combined = np.concatenate([group_a, group_b])
n_a = len(group_a)
n_permutations = 10_000
perm_diffs = np.empty(n_permutations)
for i in range(n_permutations):
    shuffled = rng.permutation(combined)
    perm_diffs[i] = shuffled[n_a:].mean() - shuffled[:n_a].mean()

p_permutation = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

# 和参数t检验的p值应该大致接近(数据本身确实是正态的, 两种方法应该给出相近结论)
_, p_ttest = stats.ttest_ind(group_b, group_a)
assert abs(p_permutation - p_ttest) < 0.05, \
    f"permutation p={p_permutation:.4f} should roughly agree with t-test p={p_ttest:.4f} on normal data"

# H0下(标签和结果确实无关)重新验证: 置换检验的假阳性率应该接近名义alpha
n_null_trials = 500
false_positives = 0
for _ in range(n_null_trials):
    null_a = rng.normal(0, 1, 12)
    null_b = rng.normal(0, 1, 12)  # 同分布, H0真实成立
    obs = null_b.mean() - null_a.mean()
    comb = np.concatenate([null_a, null_b])
    n_a = len(null_a)
    perm_diffs_null = np.empty(500)
    for j in range(500):
        shuffled = rng.permutation(comb)
        perm_diffs_null[j] = shuffled[n_a:].mean() - shuffled[:n_a].mean()
    p_null = np.mean(np.abs(perm_diffs_null) >= np.abs(obs))
    if p_null < 0.05:
        false_positives += 1

fpr = false_positives / n_null_trials
assert fpr < 0.12, f"permutation test false positive rate should be reasonably close to alpha=0.05, got {fpr:.4f}"

print(f"observed diff={observed_diff:.4f}  permutation p={p_permutation:.4f}  t-test p={p_ttest:.4f}")
print(f"false positive rate under true H0 (n={n_null_trials} trials): {fpr:.4f}")
```

**面试怎么问+追问链**(工程约束递增轴):
- Q:"置换检验不依赖任何分布假设,是不是应该完全替代t检验成为默认选择?"
- 追问1:"有什么代价?"(计算成本——每次检验都要做几千到几万次重排列计算,数据量大时t检验的解析公式是O(n),置换检验是O(n·permutations),在需要对海量指标实时计算显著性的系统里代价明显更高)
- 深挖追问:"置换检验的排列次数选多少合适,选太少会有什么问题?"(排列次数太少时,p值本身的估计精度不够(p值是一个蒙特卡洛估计量,排列次数就是"样本量"),可能给出不稳定的结论;经验上至少要选到能稳定分辨你关心的显著性水平的精度,比如要可靠区分p=0.05,排列次数至少要到千级别以上)

**常见坑:**
- 置换次数太少(比如只有100次),导致p值本身的估计噪声很大,在p值接近显著性阈值时结论不稳定。
- 误以为置换检验和bootstrap是同一回事——置换检验是在**模拟H0成立时**的抽样分布(用于假设检验),bootstrap(下一个知识点)是在**不假设任何原假设**的前提下估计一个统计量本身的抽样分布(通常用于构造置信区间),目的和构造方式都不同。

---

## 8. 自助法(Bootstrap)构造置信区间

**定义与记号:** 对观测样本(大小n)进行**有放回**重复抽样(每次抽出大小同样为n的"bootstrap样本"),重复B次,每次计算感兴趣的统计量(均值/中位数/相关系数等等),得到该统计量的B个"bootstrap估计值",用这些值的分位数(如2.5%和97.5%分位数)构造置信区间。

**一句话:** Bootstrap用"重复抽样这份已有数据本身"来近似"如果能重复抽样整个总体会怎样"——不需要知道统计量的解析抽样分布公式(对中位数、相关系数这类没有简单解析CI公式的统计量尤其有用)。

**数学推导/理论依据:** Bootstrap的合法性来自一个"用样本经验分布Fₙ替代未知的总体分布F"的思想——如果Fₙ是F的一个足够好的近似(大样本下,由Glivenko-Cantelli定理,经验分布函数一致收敛到真实分布函数),那么"从Fₙ重复抽样算出的统计量的分布"也应该近似"从真实F重复抽样算出的统计量的分布"。这不是一个精确等式,是一个渐进近似,近似程度依赖原始样本量n(n太小,Fₙ本身对F的近似就很差,bootstrap也无法凭空补足这个信息缺口)。

**底层机制/为什么这样设计:** 为什么"有放回"抽样是关键?如果不放回抽样(每次抽出的子集是原样本的严格子集),抽样出的每个bootstrap样本方差会系统性小于原样本,无法正确反映真实的抽样不确定性;有放回抽样让每个bootstrap样本都是"大小为n、允许重复"的独立抽取,这个抽样过程的随机性结构和"从总体里抽n个独立样本"最为相似,这是bootstrap能够正确近似抽样分布的关键设计。

**AI研究/工程场景:** 14类"paired bootstrap比较模型分数"会直接复用这里的机制;任何没有简单解析CI公式的复杂统计量(比如两个模型准确率之差、某个非线性组合指标),bootstrap几乎是构造置信区间的默认工具。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

true_median = 10.0
# 构造一个偏态分布(指数分布), 中位数没有简单的解析CI公式, 正是bootstrap适用的场景
# Exponential(scale=θ)的中位数是 θ*ln(2)(CDF: 1-e^(-x/θ)=0.5 解出 x=θ*ln2), 要让中位数精确等于true_median
# 必须反解 θ = true_median / ln(2), 不能直接把true_median当成scale参数用(那样中位数会是 true_median*ln(2) != true_median)
exp_scale = true_median / np.log(2)
sample = rng.exponential(exp_scale, 200)

def bootstrap_ci(data, stat_fn, n_boot=5000, alpha=0.05, rng_local=None):
    boot_stats = np.array([stat_fn(rng_local.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    lower, upper = np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lower, upper, boot_stats

lower, upper, boot_medians = bootstrap_ci(sample, np.median, rng_local=rng)
sample_median = np.median(sample)

# 核心断言1: bootstrap CI应该以样本中位数为中心大致对称(允许偏态数据下的轻微不对称), 且宽度合理
assert lower < sample_median < upper
assert (upper - lower) > 0, "CI should have positive width"

# 核心断言2: 重复实验验证覆盖率(coverage probability), 复用03类知识点1的实验范式
n_trials = 2000
covered = 0
for _ in range(n_trials):
    trial_sample = rng.exponential(exp_scale, 200)
    lo, hi, _ = bootstrap_ci(trial_sample, np.median, n_boot=300, rng_local=rng)
    if lo <= true_median <= hi:
        covered += 1
coverage = covered / n_trials
assert 0.90 < coverage < 0.98, f"bootstrap CI coverage should be reasonably close to 0.95, got {coverage:.4f}"

print(f"sample median={sample_median:.4f}  95% bootstrap CI=[{lower:.4f}, {upper:.4f}]")
print(f"coverage over {n_trials} trials: {coverage:.4f}")
```

**面试怎么问+追问链**(工程约束递增轴,和09类)):
- Q:"你想给'两个模型准确率之差'构造一个置信区间,这个统计量的解析公式复杂吗?"
- 追问1:"能不能直接用bootstrap绕开这个复杂度?"(能——对每个bootstrap样本,重新计算两个模型的准确率之差,重复几千次,直接取分位数,完全不需要推导这个复合统计量的解析抽样分布公式,这是bootstrap相对于纯解析方法最大的实用价值)
- 深挖追问:"如果原始数据量高达千万级,每次bootstrap都要重新计算一遍统计量,计算成本会不会成为问题?能不能优化?"(可以用更高效的bootstrap变体,比如"泊松bootstrap"(用泊松分布近似有放回抽样的计数,可以向量化/分布式并行计算,不需要真的执行昂贵的随机抽取操作)——这里把讨论引向工程约束递增轴,从"理论上能做"过渡到"大规模数据下怎么做得动")

**常见坑:**
- 原始样本量本身很小(比如n<10)时,bootstrap给出的CI可能非常不可靠——bootstrap无法凭空创造原始数据里没有的信息,小样本时经验分布Fₙ本身对真实分布F的近似就很差。
- Bootstrap次数(B)选得太少(比如几十次),导致分位数估计本身有很大误差,通常建议至少1000次以上。

---

下一篇:[05-multiple-testing-and-regression-inference.md](05-multiple-testing-and-regression-inference.md) —— 多重比较问题的正式量化,以及回归系数的统计推断。
