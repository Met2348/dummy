# 13 · 贝叶斯应用深挖(Bayesian Applications)

> 总览见 [00-roadmap.md](00-roadmap.md)

板块III收官篇。11类建立了贝叶斯推断的核心机制,12类建立了非共轭场景下的MCMC采样。本文把这些机制落地到更贴近AI/ML工程场景的具体应用:贝叶斯A/B测试怎么做、可信区间和置信区间在实际A/B测试场景下的语义差异、模型比较的贝叶斯工具(Bayes factor)、以及一个经常被想当然却并不完全正确的说法——"贝叶斯方法天然免疫窥探问题"——本文用数值直接检验这个说法,得到一个比预期更微妙的结论。知识点5汇总板块III全部证据构建方法论决策表,收束整个板块。

**环境声明:** 全部代码在仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1)下真实跑通,随机抽样固定种子(`np.random.default_rng(42)`)。

---

## 1. 贝叶斯A/B测试 —— 直接问"B比A好的概率是多少"

**定义与记号:** 贝叶斯A/B测试:用Beta-Binomial共轭机制(11类知识点1),分别对A组和B组的转化率θ_A、θ_B设定先验、各自独立更新为后验,然后回答"P(θ_B>θ_A\|data)"(B组转化率高于A组的后验概率)这个问题——不是频率派的"是否显著拒绝原假设",而是直接给出一个概率化的"B比A好的可能性有多大"。

**一句话:** 频率派A/B测试问的是"如果两组真的没有差异,观测到这么大差异的概率是多少"(p值,一个反直觉的间接问题);贝叶斯A/B测试直接问"根据现在的数据,B组真的比A组好的概率是多少"(一个更直接、更贴近业务决策直觉的问题)。

**数学推导:** P(θ_B>θ_A\|data)没有解析公式(两个独立Beta分布的比较没有闭式解),但可以直接用蒙特卡洛方法算:从后验Beta(α_A_post,β_A_post)和Beta(α_B_post,β_B_post)分别抽取大量样本,直接统计"θ_B样本>θ_A样本"这个事件的经验频率,随抽样数增加收敛到真实的P(θ_B>θ_A\|data)。

**底层机制/为什么这样设计:** 为什么贝叶斯A/B测试要用采样而不是解析公式,这不是和11类知识点1强调的"共轭先验能避免数值方法"矛盾吗?并不矛盾——每一组θ_A、θ_B各自的后验依然是解析的Beta分布(单变量共轭机制完全生效),数值方法只用在"比较两个独立后验分布的相对大小"这一步,这个比较操作本身超出了单变量共轭更新能覆盖的范围,需要额外的数值技巧——这体现了"共轭先验解决了更新这一步,不等于解决了后续所有需要的运算"这个更细致的认识。

**AI研究/工程场景:** 一些实验平台(尤其是强调业务方自助分析、不要求用户理解p值语义的产品)采用贝叶斯A/B测试框架,直接展示"B版本比A版本好的概率是87%"这种业务方更容易正确理解的表述,而不是"p=0.03,拒绝原假设"这种容易被误读的频率派表述(03类已讨论的误读问题)。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

n_per_group = 500
p_A, p_B = 0.10, 0.13  # 真实转化率, B组确实更好

data_A = rng.binomial(1, p_A, n_per_group)
data_B = rng.binomial(1, p_B, n_per_group)
k_A, k_B = data_A.sum(), data_B.sum()

# 贝叶斯: 独立更新两组后验(弱先验Beta(1,1))
alpha_A_post, beta_A_post = 1 + k_A, 1 + (n_per_group - k_A)
alpha_B_post, beta_B_post = 1 + k_B, 1 + (n_per_group - k_B)

n_mc = 500_000
theta_A_samples = rng.beta(alpha_A_post, beta_A_post, n_mc)
theta_B_samples = rng.beta(alpha_B_post, beta_B_post, n_mc)
prob_B_better = (theta_B_samples > theta_A_samples).mean()

# 频率派: 两比例z检验
p_pool = (k_A + k_B) / (2 * n_per_group)
se_pool = np.sqrt(2 * p_pool * (1 - p_pool) / n_per_group)
z_stat = (k_B / n_per_group - k_A / n_per_group) / se_pool
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# 核心断言: 两种方法在同一份数据上应该讲同一个故事(方向一致): 贝叶斯给出高胜率, 频率派显著拒绝
assert prob_B_better > 0.9, f"Bayesian P(B better) should be high given the true effect, got {prob_B_better:.4f}"
assert p_value < 0.05, f"frequentist test should also reject at this effect size, got p={p_value:.4f}"

print(f"observed: k_A={k_A}/{n_per_group}, k_B={k_B}/{n_per_group}")
print(f"Bayesian P(theta_B > theta_A | data) = {prob_B_better:.4f}")
print(f"frequentist z-test: z={z_stat:.4f}, p={p_value:.4f}")
```

**面试怎么问+追问链**(真实性验证轴):
- Q:"贝叶斯A/B测试报告'B版本比A版本好的概率是92%',这句话能不能保证B版本上线后一定更好?"
- 追问1:"'92%的概率更好'和'一定更好'有什么区别?"(92%不是100%,依然有8%的可能性B版本实际上不如A版本或者两者相当,贝叶斯概率化的表述提供的是一个基于当前数据和先验的置信程度,不是确定性保证)
- 深挖追问:"如果两个版本的贝叶斯后验概率是51% vs 49%,能不能说'两个版本几乎没有差别,该上哪个都行'?"(可以这样理解,但更完整的判断还应该结合具体的效应量大小(θ_B-θ_A的后验分布中心和宽度)一起看,51%vs49%这么接近的概率,通常也意味着效应量的后验分布本身跨越0附近很宽,不确定性很大,不只是"胜率接近"这一个信息)

**常见坑:**
- 把"贝叶斯概率P(θ_B>θ_A)=92%"误解成"效应量足够大、值得上线"——这个92%只回答了"方向"问题,不包含"好多少"这个业务同样关心的信息,需要额外看效应量的后验分布。
- 认为贝叶斯A/B测试"不需要考虑样本量"——虽然不需要像频率派那样提前算好样本量再做检验,但样本量太小时后验本身会很分散,P(θ_B>θ_A)这类概率估计的可靠性同样依赖数据量。

---

## 2. 可信区间vs置信区间语义差异 —— 在A/B测试差值场景里重新验证

**定义与记号:** 延续知识点1的贝叶斯A/B测试场景,进一步对比"两组转化率之差"δ=θ_B-θ_A(业务上最关心的"效应量")的频率派置信区间(Wald CI)和贝叶斯可信区间(基于后验样本差值的经验分位数)。

**一句话:** 11类知识点3已经在单个比例的场景下建立了"数值接近不等于语义等价"这个原则;本知识点把同样的原则用在A/B测试更实际关心的"差值"量上,并且构造出一个数值也明显分歧的具体反例(强先验+小样本流量),让"这不只是理论上的区别,数值上也会真的对不上"这件事更加具体可信。

**数学推导:** 频率派δ的Wald CI:δ̂±z·SE(δ̂),SE(δ̂)=√(p̂_A(1-p̂_A)/n_A+p̂_B(1-p̂_B)/n_B)(和06类样本量公式的方差结构完全一致)。贝叶斯δ的可信区间:从A、B两组后验各自采样,直接算出δ=θ_B-θ_A的采样分布,取2.5%和97.5%分位数。用弱先验+适中样本量时,两者数值上应该接近(和11类知识点3同样的极限逻辑);用强先验(比如Beta(50,50),相当于100个"虚拟观测")+很小的实际样本量,贝叶斯可信区间会被明显向"两组差不多"的方向压缩、且区间更窄,和频率派CI出现数值上的实质差异。

**底层机制/为什么这样设计:** 为什么要专门在"差值"这个衍生量上重新验证一遍?因为"差值"是两个随机变量的运算结果,其分布形状和单个比例的分布形状并不相同(即使两个原始比例的后验都是Beta分布,它们的差通常没有一个标准分布名称),数值计算的具体流程(蒙特卡洛差值 vs Wald正态近似)也和单变量情形不同,用一个业务上更常见、更贴近实际决策场景的例子重新验证同一个哲学原则,让它的适用范围从"教科书例子"扩展到"实际会遇到的分析场景"。

**AI研究/工程场景:** 汇报A/B测试结果时,"提升了3个百分点,95%区间是[1%, 5%]"这句话,不管背后用的是频率派还是贝叶斯方法计算出来的,业务方几乎总是按贝叶斯语义理解——这进一步印证了11类已经讨论过的现象,即使分析师用频率派方法计算,汇报语言也经常不自觉地滑向贝叶斯式的表述。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

n_per_group = 500
p_A, p_B = 0.10, 0.13
data_A = rng.binomial(1, p_A, n_per_group)
data_B = rng.binomial(1, p_B, n_per_group)
k_A, k_B = data_A.sum(), data_B.sum()
p_hat_A, p_hat_B = k_A / n_per_group, k_B / n_per_group
delta_hat = p_hat_B - p_hat_A

z = stats.norm.ppf(0.975)
se_delta = np.sqrt(p_hat_A * (1 - p_hat_A) / n_per_group + p_hat_B * (1 - p_hat_B) / n_per_group)
ci_freq = (delta_hat - z * se_delta, delta_hat + z * se_delta)

# 贝叶斯(弱先验 Beta(1,1)): 蒙特卡洛差值分布
a1, b1 = 1, 1
n_mc = 300_000
thA = rng.beta(a1 + k_A, b1 + (n_per_group - k_A), n_mc)
thB = rng.beta(a1 + k_B, b1 + (n_per_group - k_B), n_mc)
delta_samples_weak = thB - thA
ci_bayes_weak = (np.percentile(delta_samples_weak, 2.5), np.percentile(delta_samples_weak, 97.5))

# 核心断言1: 弱先验+适中样本量下, 两种区间数值上应该几乎重合
assert abs((ci_bayes_weak[1] - ci_bayes_weak[0]) - (ci_freq[1] - ci_freq[0])) < 0.01
assert abs(delta_samples_weak.mean() - delta_hat) < 0.01

# 小样本 + 强先验 Beta(50,50)(相当于100个"虚拟观测", 强烈拉向"两组一样")
n_small = 30
data_A_small = rng.binomial(1, p_A, n_small)
data_B_small = rng.binomial(1, p_B, n_small)
k_A_s, k_B_s = data_A_small.sum(), data_B_small.sum()
p_hat_A_s, p_hat_B_s = k_A_s / n_small, k_B_s / n_small
delta_hat_s = p_hat_B_s - p_hat_A_s
se_delta_s = np.sqrt(p_hat_A_s * (1 - p_hat_A_s) / n_small + p_hat_B_s * (1 - p_hat_B_s) / n_small)
ci_freq_s = (delta_hat_s - z * se_delta_s, delta_hat_s + z * se_delta_s)

a2, b2 = 50, 50
thA_s = rng.beta(a2 + k_A_s, b2 + (n_small - k_A_s), n_mc)
thB_s = rng.beta(a2 + k_B_s, b2 + (n_small - k_B_s), n_mc)
delta_samples_strong = thB_s - thA_s
ci_bayes_strong = (np.percentile(delta_samples_strong, 2.5), np.percentile(delta_samples_strong, 97.5))

# 核心断言2: 小样本+强先验下, 两种方法的中心估计出现实质性分歧
center_diff_strong = abs(delta_samples_strong.mean() - delta_hat_s)
assert center_diff_strong > 0.02, \
    f"with a strong prior and n=30, Bayesian and frequentist point estimates should diverge substantially, got {center_diff_strong:.4f}"

# 核心断言3: 强先验下, 贝叶斯可信区间应该明显更窄(强先验相当于额外贡献了100个虚拟观测, 削减了不确定性)
width_freq_s = ci_freq_s[1] - ci_freq_s[0]
width_bayes_s = ci_bayes_strong[1] - ci_bayes_strong[0]
assert width_bayes_s < width_freq_s, \
    f"the strong-prior credible interval should be narrower than the frequentist CI: bayes={width_bayes_s:.4f} freq={width_freq_s:.4f}"

print(f"weak prior, n={n_per_group}: freq CI={tuple(round(x,4) for x in ci_freq)}  bayes CI={tuple(round(x,4) for x in ci_bayes_weak)}  (nearly identical)")
print(f"strong prior, n={n_small}: freq point={delta_hat_s:.4f}  bayes mean={delta_samples_strong.mean():.4f}  (diverge)")
print(f"strong prior, n={n_small}: freq CI width={width_freq_s:.4f}  bayes CI width={width_bayes_s:.4f}")
```

**面试怎么问+追问链**(决策依据追问轴):
- Q:"小流量的实验(比如新市场刚上线,每组只有几十个样本),该用频率派方法还是贝叶斯方法分析A/B测试结果?"
- 追问1:"样本量小对两种方法各自会有什么影响?"(频率派方法在小样本下置信区间会很宽,诚实地反映出"数据不足以下结论"这个事实;贝叶斯方法在小样本下,如果先验设置得当(比如用历史同类市场的转化率数据作为合理先验),可以给出比"完全不用先验信息"更稳健的估计,但前提是这个先验本身要有可靠依据)
- 深挖追问:"如果这个'历史同类市场先验'其实并不完全适用于这个新市场,贝叶斯方法的这个'优势'会变成什么?"(会变成劣势——不恰当的先验会把后验估计系统性地拉向一个错误的方向,而且因为样本量小、数据本身的"纠正力"弱,这个偏差不会像11类知识点5展示的大样本场景那样被数据自动冲淡)

**常见坑:**
- 只在"弱先验、适中样本量"这一种情形下验证过频率派和贝叶斯方法数值接近,就想当然地认为"这两种方法反正结果差不多,用哪个无所谓"——样本量小或先验强时这个假设会失效。
- 汇报结果时把"可信区间"和"置信区间"这两个术语混用、不加区分——背后的概率陈述对象确实不同(11类知识点3已讲),严肃的分析报告应该准确使用对应的术语。

---

## 3. Bayes factor模型比较 —— 连续的证据强度,没有人为的显著/不显著断点

**定义与记号:** Bayes factor(贝叶斯因子)BF₁₀=p(data\|M₁)/p(data\|M₀):比较两个候选模型对观测数据的边际似然(marginal likelihood,对模型内部所有参数的不确定性积分掉之后的整体解释力)之比,BF₁₀>1支持M₁,数值大小对应支持强度(常用经验分级:BF>3算"温和证据",BF>10算"强证据")。

**一句话:** 似然比检验问的是"在原假设成立的世界里,观测到这么支持备择假设的似然比,是不是一个足够罕见的、值得怀疑原假设的巧合"(仍是p值式的反证逻辑);Bayes factor问的是"这份数据整体上更支持哪个模型,支持到什么程度"(直接对模型证据强弱做比较,不需要先假设某个模型为真再去反证)。

**数学推导:** 构造两个候选模型:M₀(原假设,θ固定等于0.5)vs M₁(θ自由参数,先验θ~Beta(1,1))。M₀的边际似然直接算出:p(data\|M₀)=C(n,k)·0.5ⁿ。M₁的边际似然需要对θ积分:p(data\|M₁)=∫C(n,k)θ^k(1-θ)^(n-k)dθ=C(n,k)·B(k+1,n-k+1)(用Beta函数的积分公式,先验均匀所以积分有解析解——这是共轭先验在模型比较这个新场景下的另一层价值)。BF₁₀=p(data\|M₁)/p(data\|M₀)。

**底层机制/为什么这样设计:** 为什么Bayes factor天然包含一种"奥卡姆剃刀"效应(倾向于惩罚过于灵活的模型),而似然比检验(不加额外惩罚项)不会?因为边际似然p(data\|M₁)是对**所有**可能的参数值(按先验加权)的似然求平均,如果M₁允许参数在很大范围内自由变化,大部分参数取值下的似然可能其实很差,平均下来反而会拉低M₁的边际似然;而似然比检验通常直接用最大似然估计(MLE)代入,只看"最优参数点"下的似然,不会因为模型"到处乱跑但只有一个点表现好"而受到惩罚——Bayes factor这种"对参数不确定性积分"的操作天然地对模型复杂度做了惩罚(虽然实践中对先验的选择很敏感,是这个方法本身的一个已知争议点)。

**AI研究/工程场景:** 模型选择场景(判断"这个指标的变化是纯噪声还是真的存在非零效应")用Bayes factor可以直接给出"数据支持存在效应的证据强度是多少倍",这个表述有时候比"p<0.05所以拒绝原假设"更能传达证据强度的连续性(p=0.049和p=0.051背后的证据强度其实非常接近,但二元的"显著/不显著"判定会把它们粗暴地划到两个不同结论,Bayes factor作为连续比值没有这个人为断点)。

**可运行例子:**
```python
import numpy as np
from scipy.special import betaln
from scipy import stats
from math import comb, log

n, k = 30, 22  # 明显偏离0.5

# M0(theta=0.5固定)和M1(theta~Beta(1,1))各自的边际似然
log_p_data_M0 = log(comb(n, k)) + k * np.log(0.5) + (n - k) * np.log(0.5)
log_p_data_M1 = log(comb(n, k)) + betaln(k + 1, n - k + 1)
BF10 = np.exp(log_p_data_M1 - log_p_data_M0)

# 数值积分独立验证M1边际似然的解析公式
theta_grid = np.linspace(1e-5, 1 - 1e-5, 500_000)
likelihood_vals = theta_grid ** k * (1 - theta_grid) ** (n - k)
p_data_M1_numerical = comb(n, k) * np.trapezoid(likelihood_vals, theta_grid)
relative_diff = abs(p_data_M1_numerical - np.exp(log_p_data_M1)) / np.exp(log_p_data_M1)
assert relative_diff < 0.01, f"numerical integration should match the analytical marginal likelihood, rel diff={relative_diff:.6f}"

# 核心断言: BF10应该显示至少"温和"强度的证据支持M1(数据明显偏离0.5)
assert BF10 > 3, f"Bayes factor should show at least moderate evidence for M1, got {BF10:.4f}"

# 传统似然比检验(MLE代入), 对比结论方向是否一致
theta_mle = k / n
log_lik_mle = k * np.log(theta_mle) + (n - k) * np.log(1 - theta_mle)
log_lik_null = k * np.log(0.5) + (n - k) * np.log(0.5)
lr_stat = 2 * (log_lik_mle - log_lik_null)
lr_pvalue = 1 - stats.chi2.cdf(lr_stat, df=1)
assert lr_pvalue < 0.05, f"likelihood ratio test should also reject theta=0.5, got p={lr_pvalue:.4f}"

print(f"data: n={n}, k={k} (observed rate={k/n:.3f})")
print(f"Bayes factor BF10 = {BF10:.4f}  (>3 = moderate evidence for M1)")
print(f"likelihood ratio test: stat={lr_stat:.4f}, p={lr_pvalue:.4f}  (same conclusion direction)")
```

**面试怎么问+追问链**(决策依据追问轴):
- Q:"Bayes factor和p值都是用来'判断证据强度'的工具,为什么还需要两套体系,不能统一成一个吗?"
- 追问1:"两者从数学结构上最本质的区别是什么?"(p值的计算前提是"假设原假设为真",在这个假设下去看观测数据有多"意外";Bayes factor直接比较两个模型对数据的解释力,不需要预先假设任何一个为真,是两个模型证据的直接对比,这是两者从出发点上就不同的设计哲学)
- 深挖追问:"Bayes factor对先验选择敏感这一点,具体是怎么体现的,会不会让模型比较结论变得不可靠?"(如果M₁的先验设置得极端宽泛,边际似然会被这些"不合理"的参数区域拉低,可能导致BF₁₀系统性地偏向M₀(哪怕真实效应确实存在)——这是Bayes factor方法一个广为人知的局限,先验的具体设置需要认真论证)

**常见坑:**
- 把Bayes factor的数值直接当成"概率"来解读(比如BF₁₀=5就理解成"M₁有5倍的概率是对的")——Bayes factor是似然比,不是后验概率比,要得到后验概率比还需要额外乘上M₀、M₁各自的先验概率之比。
- 忽视Bayes factor对模型先验设置的敏感性,直接把计算出的数值当作不容置疑的客观结论,而不去做类似11类知识点5建议的先验敏感性分析。

---

## 4. 贝叶斯早停优势?—— 一个需要被数值纠正的常见说法

**定义与记号:** 一个常见但不完全正确的直觉:"贝叶斯方法天然免疫窥探问题,不需要专门的边界校正,可以随时查看"。这个说法需要被精确拆解:①贝叶斯后验P(θ\|data)在任意固定时刻都是对"给定当前数据"的诚实描述,这一点始终成立,不因查看次数而改变含义(似然原理给出的保证);但②"用一个简单的后验概率阈值(比如P(μ_B>μ_A)越过97.5%/2.5%)反复检验、一旦越过阈值就停"这个朴素停止规则,并**不会**自动获得"整体错误率受控"这个性质——这是一个常被误解、需要用数值直接检验的说法。

**一句话:** "后验概率本身的含义不会因为查看次数而改变"和"用后验概率阈值做随时停止的决策规则,错误率不会膨胀"是两件不同的事——前者是真的,后者是一个需要被检验、而且在朴素实现下会被数值直接推翻的说法,这正是本知识点要澄清的核心区分。

**数学推导/说明:** 用和06类知识点4完全同构的H0场景(两组真实同分布,20次检验,最大样本量2000/组)对比三种规则的假阳性率:①单次预先设定截止点的检验(基准,应该≈5%);②频率派朴素反复t检验(06类知识点4已验证会显著膨胀);③朴素贝叶斯规则(Normal-Normal后验更新,弱先验,P(μ_B>μ_A)越过[2.5%,97.5%]区间外就停,阈值特意对齐频率派两侧α=0.05)。数值结果:**朴素贝叶斯规则的假阳性率和频率派朴素窥探几乎完全相同**——这不是巧合:11类知识点3已经验证过,弱先验下贝叶斯后验概率阈值在数值上就是频率派检验统计量阈值的另一种记法,换一套语言描述同一个数学对象,不会让它凭空获得鞅(martingale)性质、也不会让"反复检验+在首次越界时停止"这个操作变得无害。真正能够严格保证"随时停止都不膨胀错误率"的,是07类知识点2的mSPRT构造——用混合先验算出的似然比Λₙ被证明是非负鞅,Ville不等式P(sup_n Λₙ≥1/α)≤α给出的是这个**特定构造**的严格保证,不是"使用贝叶斯思维"这个标签自动附赠的性质。

**底层机制/为什么这样设计:** 为什么"贝叶斯"这个标签本身不能自动带来"随时停止不膨胀错误率"的保证?因为决定一个序贯停止规则是否安全的,不是这个统计量是用频率派语言还是贝叶斯语言描述出来的,而是这个统计量在原假设下是否具备鞅这个精确的数学性质——mSPRT的似然比之所以是鞅,是因为它对混合先验做了积分这个特定的构造步骤;简单的后验概率(尤其在弱先验、和频率派检验统计量数值等价的情形下)并不天然具备这个性质,只是换了个说法重新表述了同一个会被窥探问题坑害的统计量。这个区分——"数学结构(是否是鞅)才是关键,不是用哪个学派的语言描述"——是板块III最后要留下的、也是最容易被表面理解混淆的一个深层认识。

**AI研究/工程场景:** 一些实验平台可能错误地认为"只要把显著性判定换成贝叶斯后验概率,就可以放心地允许业务方随时查看结果",这个假设如果没有经过本知识点这样的数值验证,可能会把07类知识点4揭示的窥探问题原样带入贝叶斯包装的界面里,只是换了一套看起来更友好的数字(概率而不是p值),错误率膨胀的实际风险完全没有解决——这是一个真实存在、容易被"贝叶斯听起来更先进"这种表面印象误导的工程决策陷阱。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

n_checks = 20
max_n_per_group = 2000
check_points = np.linspace(max_n_per_group // n_checks, max_n_per_group, n_checks).astype(int)
sigma = 1.0
mu0, tau0_sq = 0.0, 100.0  # 弱先验

n_trials = 2000
bayes_false_positives = 0
freq_false_positives = 0
threshold_hi, threshold_lo = 0.975, 0.025  # 对齐频率派两侧alpha=0.05

for _ in range(n_trials):
    group_a_full = rng.normal(0, sigma, max_n_per_group)
    group_b_full = rng.normal(0, sigma, max_n_per_group)  # H0严格成立

    bayes_stopped, freq_stopped = False, False
    for n_cur in check_points:
        a_data, b_data = group_a_full[:n_cur], group_b_full[:n_cur]

        # 频率派朴素窥探(06类知识点4的做法)
        _, p_val = stats.ttest_ind(a_data, b_data)
        if p_val < 0.05:
            freq_stopped = True

        # 朴素贝叶斯: Normal-Normal后验更新, P(mu_B > mu_A | data)
        mu_A_post = (mu0 / tau0_sq + a_data.sum() / sigma ** 2) / (1 / tau0_sq + n_cur / sigma ** 2)
        var_A_post = 1 / (1 / tau0_sq + n_cur / sigma ** 2)
        mu_B_post = (mu0 / tau0_sq + b_data.sum() / sigma ** 2) / (1 / tau0_sq + n_cur / sigma ** 2)
        var_B_post = 1 / (1 / tau0_sq + n_cur / sigma ** 2)
        diff_mean = mu_B_post - mu_A_post
        diff_std = np.sqrt(var_A_post + var_B_post)
        prob_B_greater = 1 - stats.norm.cdf(0, loc=diff_mean, scale=diff_std)
        if prob_B_greater > threshold_hi or prob_B_greater < threshold_lo:
            bayes_stopped = True

    if freq_stopped:
        freq_false_positives += 1
    if bayes_stopped:
        bayes_false_positives += 1

freq_fpr = freq_false_positives / n_trials
bayes_fpr = bayes_false_positives / n_trials

# 核心断言1: 两条朴素规则的假阳性率都应该明显膨胀, 远超名义5%
assert freq_fpr > 0.15, f"naive frequentist peeking should be badly inflated, got {freq_fpr:.4f}"
assert bayes_fpr > 0.15, f"naive Bayesian threshold peeking should ALSO be badly inflated, got {bayes_fpr:.4f}"

# 核心断言2: 两者的假阳性率应该数值上几乎不可区分(同一现象的两种记法, 不是巧合)
assert abs(freq_fpr - bayes_fpr) < 0.03, \
    f"naive Bayesian and frequentist peeking should show essentially the same inflation: freq={freq_fpr:.4f} bayes={bayes_fpr:.4f}"

print(f"naive frequentist repeated t-test FPR = {freq_fpr:.4f}  (target ~0.05, badly inflated)")
print(f"naive Bayesian repeated posterior-threshold FPR = {bayes_fpr:.4f}  (ALSO badly inflated, nearly identical to frequentist)")
print("=> 'switching to a Bayesian posterior probability' does NOT by itself fix the peeking problem.")
print("   only a purpose-built construction like the mSPRT (07-class KP2) has the martingale property needed.")
```

**面试怎么问+追问链**(方案批判迭代轴,呼应07类知识点1-2和06类知识点4):
- Q:"有人提议'把显著性判定换成贝叶斯后验概率,这样业务方就可以随时查看结果、不用担心窥探问题了',这个方案靠谱吗?"
- 追问1:"'贝叶斯后验概率不依赖预先设定的检验次数'这句话本身没错,为什么这个方案还是有问题?"("后验概率的含义不因查看次数而改变"和"反复查看后验概率、越过阈值就停这个决策规则不会膨胀错误率"是两件不同的事——前者是似然原理给出的保证,后者需要这个被监控的统计量本身具备鞅性质,朴素的后验概率阈值(尤其在弱先验下)并不自动具备这个性质,数值实验直接展示了它和频率派朴素窥探几乎一样容易被坑)
- 深挖追问:"那什么样的贝叶斯构造才能真正解决这个问题?"(07类知识点2的mSPRT——不是直接监控"后验概率是否越过某个阈值",而是构造一个基于混合先验的似然比统计量,这个统计量在原假设下被证明是非负鞅,Ville不等式才能据此给出"任意停止时刻的错误率都有界"这个严格保证;这是一个需要专门数学构造的特定统计量,不是"用了贝叶斯方法"就自动获得的免费性质)

**常见坑:**
- 把"贝叶斯不需要预先设定检验方案"这个关于后验语义的正确论断,不加区分地推广成"任何基于贝叶斯量的随时停止规则都不会膨胀错误率"这个错误论断——这正是本知识点用数值直接推翻的常见误解。
- 反过来因为这个发现而完全否定贝叶斯方法在窥探问题上的价值——07类知识点2的mSPRT恰恰是一个真正有效的、基于贝叶斯机制(混合似然)的解决方案,问题不在于"贝叶斯思路不行",而在于"不是所有贝叶斯构造都天然具备所需的数学性质",需要用对具体的构造方法。

---

## 5. 何时用贝叶斯何时用频率派 —— 板块III的方法论收束

**定义与记号:** 汇总贝叶斯方法和频率派方法在不同场景下的相对优劣,基于本板块(11-13类)已经用代码验证过的具体证据构建一个决策表:①样本量大小如何影响两种方法的一致性(11类知识点5:n=10时不同先验分歧达0.079,n=5000时分歧仅0.008,相差10倍以上);②是否有可靠的先验信息可用(13类知识点2:合理先验下两种方法接近,不合理强先验下两者分道扬镳);③沟通对象是否是统计专业背景(11类知识点3:业务方几乎总是按贝叶斯语义理解频率派CI);④是否需要处理"随时查看"的实验场景(13类知识点4:只有mSPRT这种专门构造有效,不是"贴上贝叶斯标签"就自动获得);⑤是否需要模型比较而非单一参数估计(13类知识点3:Bayes factor给出连续证据强度)。

**一句话:** "贝叶斯还是频率派"不是一个需要选边站队的意识形态问题,而是一个应该基于具体场景的几个关键维度系统性回答的工程决策,本系列前12类和本类前4个知识点已经用真实代码验证过每个维度各自会带来什么具体后果。

**数学推导/说明:** 不是新的数学推导,而是把前面知识点已经产出的具体数值证据汇总成可执行的决策依据:样本量小时先验/方法选择的实质影响大(11-05);先验必须有独立依据,不能拍脑袋(13-02);面向非统计背景受众时贝叶斯表述沟通成本更低(11-03);"随时查看"需要专门构造而不是简单换个说法(13-04);模型比较场景Bayes factor避免了p值的人为断点(13-03)。

**底层机制/为什么这样设计:** 为什么要专门用一个总结性知识点,而不是让读者自己从前面知识点里归纳?真实面试/工作场景里,"你会怎么选择"这类综合性问题,恰恰要求候选人能够跳出单个知识点的细节,把多个独立的技术发现整合成一个连贯的决策框架——这个能力本身(把分散的技术证据整合成一个决策论证)是比记住任何单个知识点更高阶、也更容易在面试后期被专门考察的技能,呼应dsa-deep-dive调研发现的"55分钟系统设计轮次里,初版方案只占15分钟,其余40分钟都是追问"这个规律——本知识点是"追问被回答得好"的一次预演。

**AI研究/工程场景:** 一个成熟的数据科学团队,不会规定"所有实验分析必须用贝叶斯方法"或"必须用频率派方法"这种一刀切的政策,而是根据具体分析场景的特征(团队汇报对象、样本量规模、是否有可靠的历史先验)灵活选择,甚至同一个团队对不同类型的分析问题采用不同的默认方法,这种基于场景的灵活性本身就是团队统计素养成熟度的体现。

**可运行例子:**
```python
import numpy as np

def recommend_method(n, has_reliable_prior, needs_anytime_peeking, audience_technical):
    """基于板块III(11-13类)已验证过的4条具体证据打分, 返回(bayes_score, freq_score)"""
    score_bayes, score_freq = 0, 0

    # 依据1(11类知识点5): 小样本时不同先验/方法的分歧可达10倍以上, 选择需格外谨慎, 默认偏向频率派(除非先验确实可靠)
    if n < 100:
        score_freq += 1
    else:
        score_bayes += 1

    # 依据2(13类知识点2): 可靠先验下贝叶斯能有效利用历史信息; 不可靠先验会系统性地拉偏估计
    if has_reliable_prior:
        score_bayes += 1
    else:
        score_freq += 1

    # 依据3(13类知识点4): 需要支持随时查看时, 有效方案(mSPRT)属于贝叶斯机制家族, 倾向贝叶斯路线(但要用对具体构造)
    if needs_anytime_peeking:
        score_bayes += 1

    # 依据4(11类知识点3): 面向非技术受众时, 贝叶斯概率化表述更不容易被误读
    if audience_technical:
        score_freq += 1
    else:
        score_bayes += 1

    return score_bayes, score_freq

# 三个具体场景, 分别对应板块III验证过的不同证据组合
scenario_A = recommend_method(n=100_000, has_reliable_prior=False, needs_anytime_peeking=False, audience_technical=True)
scenario_B = recommend_method(n=20, has_reliable_prior=True, needs_anytime_peeking=False, audience_technical=False)
scenario_C = recommend_method(n=5000, has_reliable_prior=False, needs_anytime_peeking=True, audience_technical=False)

# 核心断言: 三个特征明显偏向一方的场景, 打分结果应该反映出清晰的方向性建议(不是模棱两可的平局)
assert scenario_A[1] > scenario_A[0], f"large-n, no-prior, technical audience should lean frequentist, got {scenario_A}"
assert scenario_B[0] > scenario_B[1], f"cold-start with reliable prior, non-technical audience should lean Bayesian, got {scenario_B}"
assert scenario_C[0] > scenario_C[1], f"anytime-peeking + non-technical audience should lean Bayesian, got {scenario_C}"

print("scenario A (large mature A/B test, no reliable prior, technical audience):")
print(f"  bayes_score={scenario_A[0]} freq_score={scenario_A[1]}  -> {'Bayesian' if scenario_A[0] > scenario_A[1] else 'Frequentist'}")
print("scenario B (cold-start item, reliable historical prior, non-technical audience):")
print(f"  bayes_score={scenario_B[0]} freq_score={scenario_B[1]}  -> {'Bayesian' if scenario_B[0] > scenario_B[1] else 'Frequentist'}")
print("scenario C (self-serve platform, anytime peeking needed, non-technical audience):")
print(f"  bayes_score={scenario_C[0]} freq_score={scenario_C[1]}  -> {'Bayesian' if scenario_C[0] > scenario_C[1] else 'Frequentist'}")
```

**面试怎么问+追问链**(决策依据追问轴,收束全板块):
- Q:"给你一个新的分析任务,你会怎么决定用贝叶斯方法还是频率派方法?"
- 追问1:"能不能具体说说你的决策依据?"(至少应该考虑:①样本量——小样本时方法选择和先验设置的影响更大,需要更谨慎;②是否有可靠的先验信息——如果有独立可信的历史数据,贝叶斯能更好地利用,但先验必须有依据不能拍脑袋;③沟通场景——面向非统计背景的受众,贝叶斯表述通常更直觉、更不容易被误读;④是否需要支持随时查看——需要专门构造(mSPRT),不是简单换个说法就能免费获得)
- 深挖追问:"如果这几个维度给出的建议相互矛盾(比如样本量建议用贝叶斯,但沟通场景要求必须用行业标准的频率派方法),该怎么权衡?"(需要明确不同维度的优先级——通常合规/行业标准要求(比如临床试验必须用监管机构认可的频率派方法)是硬约束,优先级最高;在硬约束满足的前提下,再考虑样本量、先验可靠性等其他维度做进一步优化;这种"先满足硬约束、再在剩余自由度里优化"的决策结构,是很多实际工程决策的通用模式)

**常见坑:**
- 把方法选择当成一次性的、一劳永逸的团队政策决定,而不是针对每个具体分析场景重新评估——不同分析任务的样本量、先验可靠性、沟通对象都可能不同,同一个团队完全可能对不同任务采用不同方法。
- 忽视"行业规范/合规要求"这类硬约束,单纯从统计学理论优劣的角度做方法选择——现实工作中方法选择经常不是纯技术决策,受既有惯例、监管要求、团队协作习惯等非纯技术因素制约。

---

板块III(贝叶斯方法,11-13类,共16个知识点)到此收官。下一篇:[14-model-evaluation-statistics.md](14-model-evaluation-statistics.md) —— 板块IV(AI/ML场景专属统计),把板块I-III建立的统计工具集中应用到模型评测、排位系统、scaling law等更贴近本仓库其余系列(torch/huggingface/peft-deep-dive等)的具体场景。
