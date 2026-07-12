# 01 · 概率论回顾与描述统计深挖(Probability Recap & Descriptive Statistics)

> 总览见 [00-roadmap.md](00-roadmap.md)

本文是整个统计系列的地基,目的是**搭桥**而不是重新教一遍概率论——用户已经学过概率论(见仓库 README 背景表),这里快速过一遍已学内容,统一后续文件要用的记号和结论,重点放在"这些结果怎么用代码验证""哪些直觉容易在面试里被问穿"上,不是从零推导每一条概率论定理。

**环境声明:** 本文全部代码在仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1)下真实跑通,随机抽样类断言固定随机种子(`np.random.default_rng(42)`),数值结果是真实输出,不是手算。

---

## 1. 常见分布回顾 —— 二项是伯努利之和,泊松是二项的稀有事件极限

**定义与记号:**
- 伯努利分布 Bernoulli(p):单次试验,P(X=1)=p,P(X=0)=1-p
- 二项分布 Binomial(n,p):n次独立伯努利试验成功次数之和,P(X=k)=C(n,k)pᵏ(1-p)ⁿ⁻ᵏ
- 泊松分布 Poisson(λ):单位时间/空间内稀有事件发生次数,P(X=k)=e⁻λλᵏ/k!
- 正态分布 Normal(μ,σ²):密度 f(x)=1/(σ√(2π))·exp(-(x-μ)²/(2σ²))
- 指数分布 Exponential(λ):连续等待时间,f(x)=λe⁻λˣ (x≥0)

**一句话:** 这五个分布是后面假设检验、贝叶斯先验、时间序列噪声模型全部要用的地基——面试官默认你闭眼就能写出它们的均值方差。

**数学推导:**
- Bernoulli:E[X]=0·(1-p)+1·p=p;Var(X)=E[X²]-E[X]²=p-p²=p(1-p)
- Poisson:E[X]=Σ_{k=0}^∞ k·e⁻λλᵏ/k! = λ·Σ_{k=1}^∞ e⁻λλᵏ⁻¹/(k-1)! = λ·1 = λ(把k=0项去掉、提出一个λ,剩下的求和正好是Poisson(λ)的全概率求和=1)。方差需要先算E[X(X-1)]=λ²(同样的裂项技巧再做一次),Var(X)=E[X²]-E[X]²=(λ²+λ)-λ²=λ——泊松分布均值方差相等这个特征,17类"分布漂移"判断"数据是不是泊松过程"时会用到。
- 二项分布是n个独立Bernoulli(p)之和:E[X]=np,Var(X)=np(1-p)(独立随机变量求和,期望方差都可加)。
- **泊松是二项的极限**:固定np=λ,让n→∞、p→0,Binomial(n,p)→Poisson(λ)——这是"稀有事件在大量试验中发生次数"这个建模思路的数学基础(比如"日活10万用户里今天崩溃3次"更适合用泊松而不是二项建模,因为n很大但p很小)。

**底层机制/为什么这样设计:** 泊松分布不是凭空指定的公式,它是二项分布在"试验次数趋于无穷但成功概率趋于零、乘积保持有限"这个极限下的必然结果——这解释了为什么"稀有事件计数"场景(服务器故障次数、模型推理超时次数)默认用泊松而不是二项:试验次数(每秒/每次请求)极大,单次事件概率极小,直接用二项分布的组合数计算数值上会溢出/不稳定,泊松是这个极限场景下更合适的数学工具而不是近似技巧。

**AI研究/工程场景:** 伯努利/二项建模点击率、二分类准确率这类"发生或不发生"的事件;泊松建模系统故障率、API超时次数这类稀有计数事件;正态建模模型权重初始化、训练噪声;指数分布建模请求等待时间、生存分析里的"存活时长"(如"用户流失前的活跃天数")。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)
n_samples = 200_000

# 伯努利: 均值=p, 方差=p(1-p)
p = 0.3
bern = rng.binomial(1, p, n_samples)
assert abs(bern.mean() - p) < 0.01
assert abs(bern.var() - p * (1 - p)) < 0.01

# 泊松: 均值=方差=lambda
lam = 4.0
pois = rng.poisson(lam, n_samples)
assert abs(pois.mean() - lam) < 0.05
assert abs(pois.var() - lam) < 0.15

# 泊松是二项在 n->inf, p->0, np=lambda 固定 下的极限
n_binom, p_binom = 100_000, lam / 100_000
binom_approx = rng.binomial(n_binom, p_binom, n_samples)
assert abs(binom_approx.mean() - pois.mean()) < 0.1
assert abs(binom_approx.var() - pois.var()) < 0.2

# 指数分布均值=1/lambda, 方差=1/lambda^2
lam_exp = 2.0
expo = rng.exponential(1 / lam_exp, n_samples)
assert abs(expo.mean() - 1 / lam_exp) < 0.02
assert abs(expo.var() - 1 / lam_exp ** 2) < 0.03

print("all distribution moments verified")
```

**面试怎么问+追问链**(决策依据追问轴):
- Q:"给定日活10万用户,每天崩溃概率0.0001,你会用二项分布还是泊松分布建模每天崩溃次数?"
- 追问1:"如果直接用二项分布算P(崩溃次数=5)会有什么数值问题?"(C(100000,5)这种组合数计算在朴素实现下会溢出,泊松公式没有组合数项,数值更稳定)
- 深挖追问:"np=10不变,n从100000变成10,p从0.0001变成1,这时候泊松近似还准吗?"(不准——泊松近似要求n大p小,p=1时二项分布退化成确定性分布,和泊松分布形状完全不同;这里在考"你是背了'np=λ不变就能用泊松'这句话,还是真的理解了这个极限成立的前提条件")

**常见坑:**
- 把指数分布的"无记忆性"(P(X>s+t\|X>s)=P(X>t))和均匀分布搞混,以为"无记忆"意味着"任何时刻发生概率相同"——无记忆性说的是"已经等了s时间,不影响还要再等t时间的分布",不是概率密度本身是常数(指数分布密度是随x衰减的)。
- 泊松近似二项分布时忽视适用条件(n要足够大、p要足够小),盲目套公式。

---

## 2. 矩与偏度峰度 —— 三阶矩看不对称,四阶矩看尾部厚薄

**定义与记号:** 对随机变量X,k阶矩=E[Xᵏ],k阶中心矩=E[(X-μ)ᵏ]。标准化后:
- 偏度(skewness)γ₁=E[(X-μ)³]/σ³ —— 衡量分布不对称程度,右偏(长尾在右)>0,左偏<0,对称分布=0
- 峰度(kurtosis)γ₂=E[(X-μ)⁴]/σ⁴ —— 衡量尾部厚薄,正态分布峰度=3作为参照,超额峰度(excess kurtosis)=γ₂-3,>0叫"尖峰厚尾"(leptokurtic)

**一句话:** 均值方差只讲"中心在哪、多分散",偏度峰度讲"形状偏不偏、尾巴厚不厚"——这是02类"MLE渐进正态性"、17类"分布漂移检测"都要用到的形状描述工具。

**数学推导:** 正态分布峰度=3可以用矩生成函数(MGF)推:标准正态Z~N(0,1)的MGF是M(t)=E[e^{tZ}]=e^{t²/2}。对M(t)在t=0处求四阶导数就是E[Z⁴]:M'(t)=t·e^{t²/2},M''(t)=(1+t²)e^{t²/2},M'''(t)=(3t+t³)e^{t²/2},M''''(t)=(3+6t²+t⁴)e^{t²/2}。代入t=0:M''''(0)=3=E[Z⁴]。Z已经标准化(μ=0,σ=1),E[Z⁴]直接就是峰度,所以正态分布峰度恒等于3,这也是"超额峰度"要减3的原因——让正态分布作为参照点变成0。

**底层机制/为什么这样设计:** 为什么用三次方而不是二次方衡量不对称?因为(X-μ)²永远非负,任何分布的二阶中心矩(方差)都体现不出"往哪边歪"——只有奇数次方能保留符号信息,让正负两侧的偏离互相抵消或叠加,从而捕捉不对称性。四阶矩衡量尾部厚薄同理:一个"中间聚集、两头有极端离群值"的分布和一个"均匀分散"的分布可能方差相同,但四次方对远离均值的点的惩罚(加权)远大于二次方,所以四阶矩能把"尾部有没有极端值"这个方差看不出的信息暴露出来。

**AI研究/工程场景:** 训练数据里某个特征的偏度峰度异常,往往提示要做log变换或去除离群值再喂给模型;损失函数/梯度分布的峰度突然变化,是训练不稳定(比如学习率过大导致偶发的大梯度)的一个数值信号,比单看loss曲线更早发现问题。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
n_samples = 500_000

# 正态分布: 偏度约等于0, 超额峰度约等于0
normal_data = rng.normal(0, 1, n_samples)
skew_normal = stats.skew(normal_data)
kurt_normal_excess = stats.kurtosis(normal_data)  # scipy默认返回超额峰度(已减3)
assert abs(skew_normal) < 0.03
assert abs(kurt_normal_excess) < 0.08

# 指数分布: 理论偏度=2, 理论超额峰度=6 (与lambda无关)
exp_data = rng.exponential(1.0, n_samples)
skew_exp = stats.skew(exp_data)
kurt_exp_excess = stats.kurtosis(exp_data)
assert abs(skew_exp - 2.0) < 0.1
assert abs(kurt_exp_excess - 6.0) < 0.6

# 手写验证MGF四阶导数推出的"标准正态E[Z^4]=3"这个结论
z = rng.normal(0, 1, n_samples)
e_z4 = np.mean(z ** 4)
assert abs(e_z4 - 3.0) < 0.1

print(f"normal skew={skew_normal:.4f} kurt_excess={kurt_normal_excess:.4f}")
print(f"exp skew={skew_exp:.4f} kurt_excess={kurt_exp_excess:.4f}")
print(f"E[Z^4]={e_z4:.4f} (theory=3)")
```

**面试怎么问+追问链**(规模递增轴 + 决策依据追问轴):
- Q:"你有一份用户付费金额的数据,均值500元,标准差200元,只看这两个数你会怎么猜这份数据的分布形状?"
- 追问1:"如果告诉你偏度是3.5,这个猜测要怎么修正?"(付费金额几乎不可能是对称分布——大量小额付费+少量大额付费,右偏很正常;偏度3.5证实这一点,提示不能直接用正态假设做后续检验)
- 深挖追问:"偏度峰度都在正常范围,但样本量只有30,这个偏度估计本身可信吗?"(小样本下偏度峰度的估计量本身方差很大,这是规模递增轴的体现——同样的统计量,在n=30和n=50000下的可信度完全不同,03类"区间估计"会把这一点展开成专门的知识点)

**常见坑:**
- 混淆"峰度"和"超额峰度"——文献/库函数有的返回原始峰度(正态=3),有的返回超额峰度(正态=0),不确认口径直接对比数字会得出错误结论(`scipy.stats.kurtosis` 默认返回超额峰度,这是本文选用它的原因,但换一个库或手写公式时要重新确认)。
- 小样本下偏度峰度的估计噪声很大,却当成"分布形状的确定性描述"来使用。

---

## 3. 大数定律(LLN)—— "多测几次,平均值会越来越准"的证明

**定义与记号:** X₁,X₂,...,Xₙ独立同分布(iid),E[Xᵢ]=μ,Var(Xᵢ)=σ²<∞。样本均值X̄ₙ=(1/n)ΣXᵢ。弱大数定律:对任意ε>0,P(\|X̄ₙ-μ\|≥ε)→0(n→∞)。

**一句话:** "多测几次,平均值会越来越准"这句话的数学证明,后面所有"重复抽样验证某个结论"的可运行例子背后都在依赖这条定律成立。

**数学推导:** 用切比雪夫不等式直接证:Var(X̄ₙ)=Var((1/n)ΣXᵢ)=(1/n²)·ΣVar(Xᵢ)(iid独立可加)=(1/n²)·nσ²=σ²/n。切比雪夫不等式给出 P(\|X̄ₙ-μ\|≥ε)≤Var(X̄ₙ)/ε²=σ²/(nε²)。当n→∞时右边→0,所以左边也被夹逼到0——这就是弱大数定律,而且顺带证明了X̄ₙ的标准误是σ/√n,这是03类构造置信区间的核心公式来源。

**底层机制/为什么这样设计:** 大数定律成立的前提是Var(Xᵢ)<∞——如果分布方差无穷大(某些重尾分布,如柯西分布),样本均值不但不会稳定收敛,反而可能一直剧烈波动,这也是为什么"用平均值汇总数据"这个操作不是无条件安全的。

**AI研究/工程场景:** 训练时用mini-batch梯度均值估计真实梯度,batch size越大梯度估计越稳定,本质就是大数定律在起作用;A/B测试样本量越大,观测到的转化率越接近真实转化率,也是同一条定律。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)
true_mu = 10.0
true_sigma = 5.0

sample_sizes = [10, 100, 1_000, 10_000, 100_000]
errors = []
for n in sample_sizes:
    # 每个n重复抽样200次, 取误差的中位数减少单次抽样的随机噪声干扰
    trial_errors = [abs(rng.normal(true_mu, true_sigma, n).mean() - true_mu) for _ in range(200)]
    errors.append(float(np.median(trial_errors)))

# 核心断言: 误差整体趋势随n增大而大幅下降(理论上按1/sqrt(n)速率, 这里只要求首尾差距足够大, 不追求精确系数)
assert errors[0] > errors[-1] * 5, f"n=10 median error {errors[0]:.4f} should be much larger than n=100000 error {errors[-1]:.4f}"

# 误差量级应符合 sigma/sqrt(n) 这个理论标准误的数量级(切比雪夫给出的界, 量级对即可)
theoretical_se = [true_sigma / np.sqrt(n) for n in sample_sizes]
for n, err, se in zip(sample_sizes, errors, theoretical_se):
    assert err < se * 3, f"n={n}: observed error {err:.4f} far exceeds 3x theoretical SE {se:.4f}"

print("LLN convergence verified:", list(zip(sample_sizes, [round(e, 4) for e in errors])))
```

**面试怎么问+追问链**(规模递增轴):
- Q:"A/B测试里,为什么样本量太小的时候,即使观测到转化率差异很大,我们也不敢下结论?"
- 追问1:"能不能用大数定律的收敛速度(σ/√n)估算一下,样本量要多大误差才能压到可接受范围?"(误差按1/√n速率下降,想把误差减半,样本量要变成4倍——这个具体数量关系是很多候选人知道定律存在但答不出来的)
- 深挖追问:"如果这个指标的方差本身就很大(比如付费金额这种重尾分布),同样的样本量,置信区间会比转化率这种0/1指标更宽还是更窄?"(方差越大同样样本量下统计功效越低——为06类A/B测试功效分析埋伏笔)

**常见坑:**
- 把大数定律和中心极限定理搞混——LLN只保证样本均值收敛到真值这个"点"的性质,不描述收敛过程中样本均值本身的分布形状,那是CLT要回答的问题(下一个知识点)。
- 误以为"样本量越大越好"没有边际效应,忽视σ/√n是平方根关系——样本量翻4倍误差才减半,不是线性关系。

---

## 4. 中心极限定理(CLT)—— 不管原始分布多古怪,大样本均值都会"长成"正态

**定义与记号:** X₁,...,Xₙ iid,E[Xᵢ]=μ,Var(Xᵢ)=σ²<∞。标准化样本均值 Zₙ=(X̄ₙ-μ)/(σ/√n) 依分布收敛到标准正态N(0,1),不论Xᵢ原始分布是什么形状。

**一句话:** 不管原始数据长什么样,只要独立同分布、方差有限,大量样本的均值分布最终都会"长成"正态分布——这是t检验、置信区间、几乎所有经典假设检验成立的理论地基。

**数学推导:** 严格证明要用特征函数(证明Zₙ的特征函数逐点收敛到标准正态的特征函数e^{-t²/2},超出本文范围),这里给直觉版本:X̄ₙ可以看成"n个独立随机贡献的和除以n",任何独立随机变量之和,只要没有单个分量占主导地位,波动会互相抵消掉大部分"个性"、只留下"共性"的钟形分布特征——这也是为什么现实世界里"由大量独立小因素叠加而成"的量经常近似正态分布的直觉来源。

**底层机制/为什么这样设计:** CLT不要求Xᵢ本身是正态分布,只要求独立同分布+方差有限——这是它威力巨大的原因:不管指标原始分布多古怪(付费金额右偏、点击是0/1),只要样本量够大,样本均值/比例的分布都会趋于正态,t检验/z检验可以照常使用。但"方差有限"这个前提经常被忽视——某些重尾分布(如柯西分布)方差是无穷大,CLT在这种数据上根本不成立。

**AI研究/工程场景:** A/B测试里转化率的抽样分布用正态近似构造置信区间,不需要假设"单个用户是否转化"这个0/1变量本身是正态分布——CLT保证的是大样本下比例的分布趋于正态,不是原始数据。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

# 均匀分布本身明显不是正态分布(矩形/平顶形状)
low, high = 0.0, 10.0

def sample_means(n_per_sample, n_repeats):
    return np.array([rng.uniform(low, high, n_per_sample).mean() for _ in range(n_repeats)])

# 原始均匀分布本身: 标准化后用KS检验对比正态分布, 应该明确被拒绝(p值很小)
raw_uniform = rng.uniform(low, high, 5000)
raw_standardized = (raw_uniform - raw_uniform.mean()) / raw_uniform.std()
ks_stat_raw, p_raw = stats.kstest(raw_standardized, "norm")
assert p_raw < 0.01, "raw uniform data should NOT look normal"

# 样本量n=50的均值分布: 用KS统计量(而不是二元的p值阈值, 更抗测试本身的随机噪声)验证趋于正态
means_n50 = sample_means(n_per_sample=50, n_repeats=5000)
standardized_means = (means_n50 - means_n50.mean()) / means_n50.std()
ks_stat_clt, p_clt = stats.kstest(standardized_means, "norm")
assert ks_stat_clt < 0.03, f"CLT should make n=50 sample means close to normal, got KS stat={ks_stat_clt:.4f}"

# 标准误应该接近理论值 sigma/sqrt(n), 均匀分布方差=(b-a)^2/12
true_se = (high - low) / np.sqrt(12) / np.sqrt(50)
assert abs(means_n50.std() - true_se) / true_se < 0.1

print(f"raw data: KS stat={ks_stat_raw:.4f} p={p_raw:.6f} (NOT normal)")
print(f"n=50 sample means: KS stat={ks_stat_clt:.4f} p={p_clt:.4f} (close to normal)")
```

**面试怎么问+追问链**(方案批判迭代轴):
- Q:"你的评测指标是一个0/1的通过率,样本量只有20,能直接用z检验算置信区间吗?"
- 候选人若回答"能,CLT保证了正态近似"→追问1:"n=20,真实通过率只有2%,这时候CLT近似还准吗?"(不准——CLT的收敛速度依赖分布本身的偏度,通过率很低意味着0/1分布本身极度不对称,即使n=20正态近似也很差,需要用精确的二项分布置信区间而不是正态近似)
- 深挖追问:"那多大的n、多大的p才能放心用正态近似?"(经验法则np≥5且n(1-p)≥5——这里在考具体的、可操作的判断标准,不是泛泛地说"n要够大")

**常见坑:**
- 以为CLT保证"任何统计量"都趋于正态——CLT严格来说只针对(独立同分布的)**和/均值**这一类统计量,中位数、最大值等统计量的抽样分布有各自不同的极限定理,不能无差别套用。
- 用"画出来像钟形"这种目测判断代替真正的正态性检验(如KS检验),这也是本知识点可运行例子刻意用KS统计量而不是只凭肉眼的原因。

---

## 5. 描述统计的稳健性(均值vs中位数,标准差vs MAD)

**定义与记号:** 均值=Σxᵢ/n,中位数=排序后正中间的值。标准差σ=√(Σ(xᵢ-x̄)²/n),MAD(median absolute deviation)=median(\|xᵢ-median(x)\|)。

**一句话:** 均值和标准差对离群值极度敏感,中位数和MAD"看不见"离群值——选哪个取决于你到底想不想让极端值影响你的汇总统计量。

**数学推导:** 均值对单个数据点的"影响函数"(influence function)是无界的——把数据集里一个点从x改成x+Δ,均值直接漂移Δ/n,Δ可以取任意大的值,均值的漂移量也跟着任意大。中位数则不同:只要这个点改变后不越过"中间位置"这个排序边界,中位数完全不变;即使越过边界,中位数最多漂移到相邻数据点的位置,漂移量有界——这就是"稳健统计量"(robust statistic)的数学含义:对单个离群值的影响函数有界。

**底层机制/为什么这样设计:** 为什么中位数"抗打"?因为它只依赖数据的**排序位置**,不依赖具体数值大小——极端值不管多极端,排序位置的信息不会因此变化(除非极端到改变了谁排在中间)。均值则是所有数值本身的线性加权,数值本身的大小直接决定了它对结果的贡献。

**AI研究/工程场景:** 监控线上延迟(latency)指标时通常用P50(中位数)/P99而不是均值——个别慢请求(网络抖动/超时重试)会把均值拉得很难看,但用户体验主要由中位数/高分位数反映的"典型延迟"决定;训练数据清洗阶段用MAD而不是标准差做离群值检测,避免离群值本身把"正常范围"的标准差撑得过大导致离群值检测失灵(检测器被它要检测的东西污染的悖论)。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

clean_data = rng.normal(100, 10, 999)
outlier = np.array([100_000.0])  # 一个极端离群值
contaminated = np.concatenate([clean_data, outlier])

mean_clean, mean_contam = clean_data.mean(), contaminated.mean()
median_clean, median_contam = np.median(clean_data), np.median(contaminated)

mean_shift = abs(mean_contam - mean_clean)
median_shift = abs(median_contam - median_clean)

# 核心断言: 均值被离群值拖动的幅度远大于中位数
assert mean_shift > 90, f"mean should shift a lot, got {mean_shift:.2f}"
assert median_shift < 1.0, f"median should barely move, got {median_shift:.4f}"
assert mean_shift > median_shift * 100

# 标准差 vs MAD 的对比: MAD需要乘1.4826才能在正态分布下和标准差同尺度(渐进无偏修正常数, 来自标准正态0.75分位数的倒数)
std_clean, std_contam = clean_data.std(), contaminated.std()

def mad(x):
    med = np.median(x)
    return float(np.median(np.abs(x - med)) * 1.4826)

mad_clean, mad_contam = mad(clean_data), mad(contaminated)
std_shift = abs(std_contam - std_clean)
mad_shift = abs(mad_contam - mad_clean)
assert std_shift > 500, f"std should blow up, got {std_shift:.2f}"
assert mad_shift < 2.0, f"MAD should barely move, got {mad_shift:.4f}"

print(f"mean shift={mean_shift:.2f}  median shift={median_shift:.4f}")
print(f"std shift={std_shift:.2f}  MAD shift={mad_shift:.4f}")
```

**面试怎么问+追问链**(方案批判迭代轴):
- Q:"你要给公司的延迟监控大盘选一个汇总指标,用均值行不行?"
- 候选人若说"用均值"→追问1:"如果1%的请求因为超时重试导致延迟变成正常值的100倍,均值大盘会变成什么样?"(均值会被这1%的极端值显著拉高,给出"整体变慢了"的误导性结论,而实际上99%用户体验完全没变)
- 换方案后追问2:"改用P50就完全没问题了吗?"(P50确实不受这1%离群值影响,但也完全看不到这1%用户体验有多差——这是"稳健"的代价,通常P50+P99组合使用,单独一个分位数都不够)

**常见坑:**
- MAD直接用不乘1.4826的修正常数,导致和标准差的数值不能直接比较。
- 无脑认为"稳健统计量总是更好",忽视了稳健统计量"看不见"极端值这个特性在需要监控尾部风险时反而是缺点。

---

## 6. 协方差与相关系数 —— 为"相关不是因果"埋伏笔

**定义与记号:** 协方差 Cov(X,Y)=E[(X-μx)(Y-μy)]=E[XY]-E[X]E[Y]。皮尔逊相关系数 ρ=Cov(X,Y)/(σx·σy),取值范围[-1,1]。

**一句话:** 协方差告诉你两个变量"一起变大变小"的程度,相关系数把这个程度标准化到[-1,1]方便比较——但两者都只是**共同变化**的度量,不包含任何"谁导致谁"的信息。

**数学推导:** Cov(X,Y)=E[XY]-E[X]E[Y]的展开:E[(X-μx)(Y-μy)]=E[XY-Xμy-μxY+μxμy]=E[XY]-μyE[X]-μxE[Y]+μxμy=E[XY]-μxμy-μxμy+μxμy=E[XY]-μxμy。柯西-施瓦茨不等式保证\|Cov(X,Y)\|≤σx·σy,所以ρ必然落在[-1,1]区间——这是相关系数取值范围的数学来源,不是人为规定的。

**底层机制/为什么这样设计:** 相关系数为什么不能代表因果?因为它的数学定义完全**对称**——Cov(X,Y)=Cov(Y,X),ρ(X,Y)=ρ(Y,X),从公式本身无法区分"X导致Y"还是"Y导致X"还是"Z同时导致X和Y"。因果关系要求的是"干预X会不会改变Y的分布"(08类potential outcomes框架要正式定义的),而相关系数只是在描述"观测到的X和Y是否一起变化",这两件事在数学结构上是完全不同的对象。

**AI研究/工程场景:** 特征工程阶段用相关系数筛选和目标变量相关的特征是合理的(不需要因果,只需要预测力);但如果拿相关系数论证"改了这个特征/上线了这个功能,导致了指标提升",就是把"共同变化"偷换成了"因果关系"——这正是08类因果推断要专门解决的问题。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)
n = 10_000

# 数据生成过程(供08类因果推断基础复用同一套逻辑):
# Z(温度)独立同时驱动 X(冰淇淋销量)和 Y(溺水人数), X和Y之间没有任何直接因果连接
temperature = rng.normal(25, 8, n)  # Z: 气温
ice_cream_sales = 50 + 3 * temperature + rng.normal(0, 10, n)      # X: 由Z驱动
drowning_incidents = 2 + 0.15 * temperature + rng.normal(0, 1, n)  # Y: 由Z驱动, 和X没有直接联系

def covariance(x, y):
    return float(np.mean(x * y) - np.mean(x) * np.mean(y))

def correlation(x, y):
    return covariance(x, y) / (x.std() * y.std())

corr_xy = correlation(ice_cream_sales, drowning_incidents)

# 核心断言: X, Y之间相关系数很高(共同变化明显), 尽管X不导致Y、Y也不导致X
assert corr_xy > 0.5, f"expected strong spurious correlation, got {corr_xy:.4f}"

# 手写协方差公式和numpy内置结果交叉验证
cov_manual = covariance(ice_cream_sales, drowning_incidents)
cov_numpy = np.cov(ice_cream_sales, drowning_incidents, ddof=0)[0, 1]
assert abs(cov_manual - cov_numpy) < 1e-6

# 相关系数取值范围验证 [-1, 1] (柯西-施瓦茨不等式的数值验证)
assert -1.0 <= corr_xy <= 1.0

# 控制住Z(温度)之后, X和Y的偏相关应该显著更弱 --- 08类"混淆变量控制"要正式处理的手法, 这里先给出朴素验证
def residual_after_regress_on(y, z):
    slope = covariance(z, y) / z.var()
    intercept = y.mean() - slope * z.mean()
    return y - (intercept + slope * z)

resid_x = residual_after_regress_on(ice_cream_sales, temperature)
resid_y = residual_after_regress_on(drowning_incidents, temperature)
partial_corr = correlation(resid_x, resid_y)

assert abs(partial_corr) < 0.15, f"after controlling for Z, spurious correlation should nearly vanish, got {partial_corr:.4f}"

print(f"raw correlation(ice cream, drowning) = {corr_xy:.4f}  (spurious, driven by temperature)")
print(f"partial correlation after controlling temperature = {partial_corr:.4f}  (should be near 0)")
```

**面试怎么问+追问链**(真实性验证轴 + 决策依据追问轴):
- Q:"你发现某个特征和模型目标的相关系数是0.8,能直接说这个特征'有用'吗?"
- 追问1:"'有用'具体是指预测有用还是干预有用?你打算怎么用这个特征?"(如果只是加入模型做预测,相关性够了;但如果打算"改变这个特征的值来影响目标",就已经从预测问题跳到了因果问题,需要完全不同的论证)
- 深挖追问:"如果这个相关系数是在有严重混淆变量的观察性数据上算出来的,你怎么向面试官证明这不是虚假相关?"(要求候选人具体说出"控制混淆变量""找自然实验""做A/B测试"这类具体路径,不能只说"看着应该没问题")

**常见坑:**
- "相关不是因果"这句话大家都会说,但被问"具体怎么证明某个相关关系不是虚假的"时答不出可操作的方法(呼应08/09类要具体展开的因果推断工具箱)。
- 只算了皮尔逊相关系数就下结论"两个变量没关系"——皮尔逊相关只捕捉**线性**关系,两个变量可能有很强的非线性关系(比如Y=X²这种对称关系)但皮尔逊相关系数接近0。

---

下一篇:[02-point-estimation.md](02-point-estimation.md) —— 从"已知分布形状,怎么用数据估计具体参数"讲起。
