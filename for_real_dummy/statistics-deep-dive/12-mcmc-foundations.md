# 12 · MCMC基础深挖(Markov Chain Monte Carlo Foundations)

> 总览见 [00-roadmap.md](00-roadmap.md)

11类的所有例子都依赖共轭先验——先验和似然相乘后恰好落回同一个分布族,后验有解析解,数值网格积分只是用来交叉验证解析公式。但绝大多数真实模型的先验和似然不构成共轭对,后验没有解析形式,网格积分在参数维度稍高时就会撞上维度诅咒(计算量指数爆炸)。本文建立MCMC(马尔可夫链蒙特卡洛)这一整套绕开归一化常数、绕开维度诅咒的采样方法——用手写的Metropolis-Hastings和Gibbs采样,数值验证它们确实能采出目标分布的样本,并直面MCMC实践里最容易被忽视的两个坑:收敛诊断(burn-in)和步长调优。

**环境声明:** 全部代码在仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1)下真实跑通,随机抽样固定种子(`np.random.default_rng(42)`)。MCMC核心算法(Metropolis-Hastings、Gibbs采样)全部手写实现,不依赖PyMC/emcee等专门贝叶斯库。

---

## 1. 为什么需要采样 —— 网格积分撞上维度诅咒

**定义与记号:** 先建立两个贯穿本文乃至整个MCMC话题的核心词的直觉——马尔可夫链(Markov chain)和平稳分布(stationary distribution)。两者完整的严格定义属于随机过程课程的范畴(超出用户当前"随机过程未学"的数学背景,这里和07类知识点2处理"鞅"概念时用的是同一个原则:只建立数值/直觉层面的理解,不做严格的测度论式定义)。马尔可夫链是一个"下一步去哪,只取决于你现在在哪,不取决于你是怎么走到现在这个位置的"的随机过程——这个"只看现在,不看历史"的性质叫马尔可夫性(Markov property);正因为下一步只依赖当前状态,整条链的行为可以完全由一个"从当前状态到下一个状态"的转移规则描述,不需要记住任何更早的历史。平稳分布是这条链的一个特殊性质:如果你从某个分布出发,按这条链的规则走一步,得到的分布还是同一个分布,这个分布就叫平稳分布——可以类比"人口在城市和郊区之间按固定比例迁移,迁移几年后两地人口比例会趋于一个不再变化的稳定值",这个稳定比例就是平稳分布。

有了这两个直觉,再看MCMC要解决的问题:当先验和似然不构成共轭对时,后验p(θ\|data)∝p(data\|θ)p(θ)通常没有解析形式。低维参数空间(1-2维)时,网格积分(11类各知识点用的方法)依然可行;但参数维度升高时,网格积分所需的计算量随维度**指数**增长(维度诅咒:d维、每维切N个格点,总计算量是N^d),很快变得不可行。MCMC通过构造一条以目标后验分布为平稳分布的马尔可夫链,采样得到一批服从后验分布的样本,不需要计算难以处理的归一化常数,每一步的计算量只随维度线性增长。

**一句话:** 网格积分是"把参数空间切成很细的格子,一格一格算出后验密度再拼起来",维度升高后格子数量爆炸性增长直接算不动;MCMC换了一个思路——不试图描绘出后验分布的完整形状,而是构造一个"随机游走",让它自动倾向于在密度高的地方多停留、密度低的地方少停留,跑得足够久,停留过的位置汇总起来就近似了后验分布,不需要遍历整个空间。

**数学推导:** 量化维度诅咒:假设每个维度切N=50个网格点,1维网格积分需要50次密度计算,5维需要50⁵≈3.1×10⁸次,10维需要50¹⁰≈9.8×10¹⁶次——即使现代计算机每秒能算10⁹次,10维网格积分也需要超过10⁸秒(超过3年)。而MCMC每一步只需要计算"当前位置"和"提议位置"两个点的(未归一化)密度值之比,每次密度评估本身是d维向量上的O(d)次运算,总计算量随维度只是**线性**增长——这是MCMC相对网格积分最核心的可扩展性优势,不是经验规律,是可以直接数值验证的复杂度差异。

**底层机制/为什么这样设计:** 为什么"构造一个以目标分布为平稳分布的马尔可夫链"这个抽象想法能work?核心洞察是:只需要知道后验密度的**比值**(p(θ')/p(θ),未知的归一化常数会直接约掉),不需要知道后验密度的绝对值(不需要算出那个难以处理的归一化积分),就能判断"应该以多大概率从θ移动到θ'"——这把"计算困难的积分问题"转化成了"计算容易的比值判断问题",是MCMC能够绕开归一化常数这个计算瓶颈的根本原因。

需要诚实地补充一句边界说明,避免读者以为上面这段已经完整回答了"MCMC为什么有效":以上只回答了"为什么不需要计算归一化常数Z"这一个问题——这是一个比"这条链的样本最终为什么会收敛到目标分布"更容易回答的相邻问题,两者不是同一件事。真正的收敛机制(链从任意起点出发、长期运行后样本分布为什么会趋近于目标分布p)依赖细致平衡(detailed balance)、不可约性(irreducibility)、非周期性(aperiodicity)这几个条件共同保证,知识点2会给出具体的推导和直觉,这里先留一个悬念,不要把"不需要算Z"和"链会收敛"这两个问题的答案混在一起。

**AI研究/工程场景:** 贝叶斯深度学习、层级贝叶斯模型(hierarchical Bayesian models,同时估计多个相关群体各自的参数、又假设这些群体参数本身来自一个共同的更高层分布)几乎必然涉及高维、非共轭的后验分布,网格积分从一开始就不是可选项,MCMC(或者更现代的变分推断,超出本文范围)是这类模型能够实际落地的计算基础。

**可运行例子:**
```python
import numpy as np

N_grid_per_dim = 50

def grid_evaluations_needed(d):
    return N_grid_per_dim ** d

def mcmc_evaluations_per_step(d):
    # 每一步: 评估候选点和当前点两处的(未归一化)密度, 每次密度评估是d维向量的O(d)次运算
    return 2 * d

for d in [1, 2, 3, 5, 8, 10]:
    g, m = grid_evaluations_needed(d), mcmc_evaluations_per_step(d)
    print(f"d={d:2d}: grid_evals={g:.3e}  mcmc_evals_per_step={m}  ratio={g / m:.3e}")

# 核心断言1: 网格积分所需计算量确实随维度指数增长(相邻维度比值恒等于N)
for d in range(1, 8):
    ratio = grid_evaluations_needed(d + 1) / grid_evaluations_needed(d)
    assert abs(ratio - N_grid_per_dim) < 1e-6, f"grid evaluations should grow by exactly N per added dimension, got ratio={ratio}"

# 核心断言2: d=10维时, 网格积分所需计算量已经是天文数字
assert grid_evaluations_needed(10) > 1e15, f"grid integration at d=10 should be utterly infeasible, got {grid_evaluations_needed(10):.3e}"

# 核心断言3: MCMC每一步所需计算量在同样维度下只是线性增长, 和网格积分的天文数字差距悬殊
assert mcmc_evaluations_per_step(10) < 100, "MCMC per-step cost should remain tiny even at d=10"
assert grid_evaluations_needed(10) / mcmc_evaluations_per_step(10) > 1e13, \
    "the gap between grid and MCMC cost should itself be astronomically large at d=10"

print("all assertions passed: grid integration is exponential in d, MCMC per-step cost is linear in d")
```

**面试怎么问+追问链**(工程约束递增轴):
- Q:"为什么不能简单地把网格积分扩展到高维参数模型(比如10个参数)?"
- 追问1:"具体会遇到什么问题?"(网格积分所需的格点数量随维度指数增长,10维模型即使每维只切适度精细的网格,总计算量也会超出任何现实计算资源的量级,这是维度诅咒的直接后果)
- 深挖追问:"MCMC是怎么绕开这个问题的?"(MCMC不试图对整个参数空间做穷举式的密度计算,而是构造一条马尔可夫链,让它按照与后验密度成正比的频率访问参数空间的不同区域,每一步只需要局部的密度比值计算,不需要遍历整个空间,计算量因此不随维度指数增长)

**常见坑:**
- 认为"MCMC总是比网格积分好"——对于真正低维(1-2维)且后验形状规则的问题,网格积分反而更简单、更容易验证正确性,也没有MCMC需要面对的收敛诊断问题,MCMC的优势体现在网格积分明确不可行的高维场景。
- 把"MCMC不需要计算归一化常数"误解成"MCMC不需要知道后验的具体形式"——MCMC仍然需要能够计算(未归一化的)后验密度在任意给定点的取值,只是不需要计算这个密度函数在整个空间上的积分。

---

## 2. Metropolis-Hastings —— 有时候也接受较差的候选点

**定义与记号:** Metropolis-Hastings(M-H)算法:给定当前状态θ,从提议分布(proposal distribution)q(θ'\|θ)抽取候选新状态θ',计算接受概率 A=min(1, [p(θ')q(θ\|θ')] / [p(θ)q(θ'\|θ)]),以概率A接受θ'(移动到新状态),否则拒绝(停留在原状态)。当提议分布对称(比如以θ为中心的正态分布)时,接受概率简化为A=min(1, p(θ')/p(θ))。

**一句话:** M-H算法的核心规则很直白——如果提议的新位置密度比当前位置更高,总是接受;如果提议的新位置密度更低,以一个概率(密度比值)接受,而不是总是拒绝,这个"有时候也接受较差位置"的设计正是让链条能够探索整个分布(而不是卡在局部最优)的关键。

**数学推导:** M-H算法构造的马尔可夫链之所以以目标分布p为平稳分布,核心是满足"细致平衡"(detailed balance)条件:p(θ)T(θ→θ')=p(θ')T(θ'→θ)。这里先补一句最简说明:T(θ→θ')就是转移概率(transition probability)——从状态θ跳到状态θ'的概率,在M-H算法里等于"提议×接受"两步复合而成:T(θ→θ')=q(θ'\|θ)·A(θ→θ')。代入这个式子,用接受概率的定义可以直接验证细致平衡两边相等——这是M-H接受概率公式的设计初衷:反推出"什么样的接受概率能保证细致平衡成立"。

"细致平衡蕴含平稳性"这一步值得把求和/积分这个动作真正展开写一遍,而不是只给结论。为了让求和动作看得更清楚,先把参数空间离散化成有限个状态θ₁,...,θₖ来看(连续情形只是把求和号Σ换成积分号∫,道理完全一样,不改变论证结构):细致平衡条件对每一对状态(θᵢ,θⱼ)都成立,即 p(θᵢ)T(θᵢ→θⱼ) = p(θⱼ)T(θⱼ→θᵢ)。现在把这个等式两边对所有的i求和(j先固定不动):

Σᵢ p(θᵢ)T(θᵢ→θⱼ) = Σᵢ [p(θⱼ)T(θⱼ→θᵢ)] = p(θⱼ)·Σᵢ T(θⱼ→θᵢ)

右边能把p(θⱼ)提到求和号外面,是因为它不依赖被求和的下标i;而Σᵢ T(θⱼ→θᵢ)的含义是"从θⱼ出发,转移到所有可能状态的概率之和",根据概率的完备性这个和恒等于1,所以右边整体化简为p(θⱼ)。左边Σᵢ p(θᵢ)T(θᵢ→θⱼ)的含义是:如果当前分布是p,按转移规则T走一步之后,恰好落在状态θⱼ的概率——这正是"分布p经过一步转移之后得到的新分布,在θⱼ这一点的取值"的定义。于是两边合起来就是 Σᵢ p(θᵢ)T(θᵢ→θⱼ) = p(θⱼ),也就是"分布p走一步之后,还是分布p"——这正是平稳分布的定义(连续情形写作p(θ')=∫p(θ)T(θ→θ')dθ,是同一个论证换成积分符号,求和变积分、下标i变积分变量θ,结构完全一致)。

但平稳性只保证"如果链已经处于分布p,走一步之后还是分布p",不保证"不管链从哪个起点出发,长期运行后一定会收敛到p"——这还需要两个额外条件。不可约性(irreducibility):链能从任意一个状态,通过有限步,到达任意其他状态,不会被"困"在某个子区域出不来(比如目标分布明明有两个分离的峰,但链的提议步长太小、永远跳不到另一个峰所在的区域,就违反了不可约性)。非周期性(aperiodicity):链不会陷入"每隔固定步数才可能回到某状态"这种死板的周期性循环。一个简单的病态反例能说清楚"周期性"是什么样子:设想一条只能在两个状态A、B之间确定性来回跳的链——在A必须跳到B,在B必须跳到A,没有第三种可能。不管从哪个状态出发,这条链在奇数步只能停在其中一个状态、偶数步只能停在另一个状态,样本分布永远在A和B之间机械交替、不会稳定到任何一个固定分布上——这就是周期为2的病态情形,正是非周期性条件要排除的对象。平稳性+不可约性+非周期性三者叠加,才共同保证了长期运行下链的样本分布收敛到目标分布p,这是MCMC核心收敛定理(马尔可夫链遍历定理)的直觉版本(完整的严格证明属于随机过程课程范畴,这里同样只建立直觉、不做严格证明,和07类知识点2处理"鞅"概念时的原则一致)。

**底层机制/为什么这样设计:** 为什么"细致平衡"这个看起来更强的条件是M-H算法实际采用的设计思路?因为细致平衡提供了一个**可以直接构造**的充分条件——直接设计转移机制去满足平稳性方程本身很困难(涉及对所有θ的积分),但细致平衡是一个"逐点"的局部条件,只涉及两个具体状态之间的关系,容易反推出满足它的接受概率公式,这是M-H算法能被具体构造出来的数学技巧核心。

**AI研究/工程场景:** 层级贝叶斯模型的参数估计、13类会涉及的贝叶斯A/B测试早停等场景,当后验没有解析解或共轭结构时,M-H是最基础、最容易手写实现、也最容易向别人解释清楚原理的MCMC算法,现代概率编程语言(PyMC、Stan)背后使用更高效的变体(Hamiltonian Monte Carlo/NUTS),但M-H是理解这些更复杂算法的必经起点。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

def log_target(theta):
    return -0.5 * theta ** 2  # 标准正态的log密度(未归一化, 忽略常数)

def metropolis_hastings(log_target, theta0, n_steps, step_size, rng):
    samples = np.zeros(n_steps)
    theta = theta0
    log_p_theta = log_target(theta)
    n_accept = 0
    for i in range(n_steps):
        theta_proposed = theta + rng.normal(0, step_size)
        log_p_proposed = log_target(theta_proposed)
        if np.log(rng.random()) < log_p_proposed - log_p_theta:  # log空间里的接受判定, 等价于 accept_prob=min(1, exp(diff))
            theta, log_p_theta = theta_proposed, log_p_proposed
            n_accept += 1
        samples[i] = theta
    return samples, n_accept / n_steps

n_steps = 50_000
samples, accept_rate = metropolis_hastings(log_target, 0.0, n_steps, 1.0, rng)
burn_in = 5000
samples_post_burnin = samples[burn_in:]

# 核心断言1: 采样均值/标准差应该收敛到标准正态的理论值(0, 1)
assert abs(samples_post_burnin.mean()) < 0.05, f"MCMC sample mean should be close to 0, got {samples_post_burnin.mean():.4f}"
assert abs(samples_post_burnin.std() - 1.0) < 0.05, f"MCMC sample std should be close to 1, got {samples_post_burnin.std():.4f}"

# 核心断言2: 用KS统计量(不用p值阈值, 避免单次抽样的阈值脆弱性)对比MCMC样本和真实标准正态样本的分布形状
from scipy import stats
true_samples = rng.normal(0, 1, len(samples_post_burnin))
ks_stat, _ = stats.ks_2samp(samples_post_burnin, true_samples)
assert ks_stat < 0.03, f"MCMC sample distribution should closely match a true standard normal, KS stat={ks_stat:.4f}"

print(f"acceptance rate = {accept_rate:.4f}")
print(f"MCMC sample mean = {samples_post_burnin.mean():.4f}, std = {samples_post_burnin.std():.4f}  (target: 0, 1)")
print(f"KS statistic vs true N(0,1) samples = {ks_stat:.4f}")
```

**面试怎么问+追问链**(决策依据追问轴):
- Q:"M-H算法'以一个概率接受较差的候选点'这个设计,如果改成'只接受更好的候选点、拒绝所有更差的'(贪心策略),会有什么问题?"
- 追问1:"贪心策略会让链条最终停在哪里?"(贪心策略会让链条快速收敛到密度的局部最大值附近就不再移动——一旦到达局部峰值,任何方向的候选点密度都更低,全部被拒绝,得到的不是"目标分布的样本",而是"目标分布众数附近的一个点",完全丢失了目标分布的形状信息,多峰分布下会完全错过其他峰)
- 深挖追问:"既然要探索整个分布,为什么不干脆均匀随机地在整个参数空间里跳,不做任何'偏向密度高处'的设计?"(纯随机游走(总是100%接受)虽然理论上最终也能覆盖整个空间,但会在密度很低的区域浪费大量时间,采样效率极低——M-H的接受概率设计,是在"必须偏向高密度区域(保证效率)"和"必须偶尔访问低密度区域(保证不遗漏,保证细致平衡)"这两个要求之间找到的精确数学平衡)

**常见坑:**
- 提议分布(步长)选得不合适,导致接受率过高或过低,采样效率低下——知识点5要专门数值展开这个调优问题。
- 忘记burn-in(链条从初始点出发,前面一段时间还没有"忘记"初始点的影响,不能代表目标分布)就直接把全部采样结果拿去估计统计量——知识点4要专门用数值例子展示这个问题的严重性。

---

## 3. Gibbs采样 —— 如果条件分布已知,不用来回猜

**定义与记号:** Gibbs采样:M-H算法的一个特例,专门用于多维参数场景——不是同时对所有维度提议新候选点,而是每次只更新**一个**维度,从"给定其他所有维度当前取值的条件下,这一个维度的条件分布"里直接采样(要求这个条件分布已知且容易采样),依次循环遍历所有维度。当条件分布可以直接采样时,Gibbs采样的接受概率恒等于1(总是接受)。

**一句话:** Gibbs采样是"如果知道给定其他变量、单独一个变量的完整条件分布该怎么采样,那就不用来回猜'要不要接受这个候选点',直接精确地从条件分布采样就行"——是M-H算法在"条件分布可直接采样"这个特殊但很常见场景下的简化和加速版本。

**数学推导:** 设联合分布p(θ₁,θ₂),Gibbs采样交替:θ₁^(t+1)~p(θ₁\|θ₂^(t)),θ₂^(t+1)~p(θ₂\|θ₁^(t+1))。这是M-H的特例:如果提议分布恰好选择为条件分布本身(q(θ₁'\|θ₁,θ₂)=p(θ₁'\|θ₂)),代入M-H接受概率公式:

A = min(1, [p(θ₁',θ₂)·p(θ₁\|θ₂)] / [p(θ₁,θ₂)·p(θ₁'\|θ₂)])

利用p(θ₁,θ₂)=p(θ₁\|θ₂)p(θ₂)(联合=条件×边际),代入化简后分子分母恰好完全抵消,A=1恒成立——这是"Gibbs采样总是接受"的代数证明,不是经验规律。

衔接知识点2:因为Gibbs采样只是M-H算法在"提议分布=条件分布"这一特定选择下的特例,知识点2里"细致平衡蕴含平稳性"的求和/积分推导、以及"平稳性+不可约性+非周期性→收敛到目标分布"的完整逻辑链条,对Gibbs采样同样成立,不需要重新证明一遍——这里只是额外确认了在这个特例下接受概率恒为1这一件事,转移概率T(θ→θ')的记号和含义也和知识点2完全一致。

**底层机制/为什么这样设计:** 为什么"从完整条件分布直接采样"能保证接受率恒为1?因为提议分布被特意选择为"目标联合分布在给定其他维度下的真实条件分布",提议出来的候选点已经完全符合目标分布在这个约束下应有的样子,没有任何"偏离目标"的成分需要通过接受/拒绝机制去纠正——这是一种"提议分布和目标分布高度耦合、量身定做"的特殊设计,代价是要求条件分布必须解析已知、容易直接采样(不满足时只能退回普通M-H)。

**AI研究/工程场景:** 层级贝叶斯模型(同时估计每个用户的个体参数和跨用户共享的超参数)经常具备"条件分布可直接采样"的结构(尤其是每一层都用共轭先验搭建时),Gibbs采样因此在这类模型里比通用M-H更高效、更常用;主题模型(LDA,超出本文范围)的经典训练算法之一也是Gibbs采样,是概率图模型推断里的基础工具。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

# 目标: 二维正态联合分布(已知协方差结构, 条件分布是解析已知的一元正态)
mu1, mu2 = 2.0, -1.0
sigma1, sigma2 = 1.5, 2.0
rho = 0.6

def sample_theta1_given_theta2(theta2, rng):
    cond_mean = mu1 + rho * (sigma1 / sigma2) * (theta2 - mu2)
    cond_std = sigma1 * np.sqrt(1 - rho ** 2)
    return rng.normal(cond_mean, cond_std)

def sample_theta2_given_theta1(theta1, rng):
    cond_mean = mu2 + rho * (sigma2 / sigma1) * (theta1 - mu1)
    cond_std = sigma2 * np.sqrt(1 - rho ** 2)
    return rng.normal(cond_mean, cond_std)

n_steps = 50_000
theta1_samples = np.zeros(n_steps)
theta2_samples = np.zeros(n_steps)
theta1, theta2 = 0.0, 0.0
for i in range(n_steps):
    theta1 = sample_theta1_given_theta2(theta2, rng)
    theta2 = sample_theta2_given_theta1(theta1, rng)
    theta1_samples[i], theta2_samples[i] = theta1, theta2

burn_in = 5000
t1, t2 = theta1_samples[burn_in:], theta2_samples[burn_in:]

# 核心断言: 两个维度各自的边际均值/方差, 以及两者的相关系数, 都应该收敛到目标联合分布的真实参数
assert abs(t1.mean() - mu1) < 0.1 and abs(t1.std() - sigma1) < 0.1, \
    f"theta1 marginal should match N({mu1},{sigma1}^2), got mean={t1.mean():.4f} std={t1.std():.4f}"
assert abs(t2.mean() - mu2) < 0.1 and abs(t2.std() - sigma2) < 0.1, \
    f"theta2 marginal should match N({mu2},{sigma2}^2), got mean={t2.mean():.4f} std={t2.std():.4f}"
assert abs(np.corrcoef(t1, t2)[0, 1] - rho) < 0.05, \
    f"sampled correlation should match true rho={rho}, got {np.corrcoef(t1, t2)[0, 1]:.4f}"

print(f"theta1: sampled mean={t1.mean():.4f} std={t1.std():.4f}  (true mu1={mu1} sigma1={sigma1})")
print(f"theta2: sampled mean={t2.mean():.4f} std={t2.std():.4f}  (true mu2={mu2} sigma2={sigma2})")
print(f"sampled correlation = {np.corrcoef(t1, t2)[0, 1]:.4f}  (true rho={rho})")
```

**面试怎么问+追问链**(决策依据追问轴):
- Q:"既然Gibbs采样'总是接受',是不是意味着它总是比M-H更好、应该优先选择?"
- 追问1:"Gibbs采样有什么前提条件是普通M-H不需要的?"(Gibbs采样要求每个维度在给定其他维度时的条件分布必须是解析已知、且容易直接采样的分布——这个条件在很多实际模型里不满足,这时候只能退回普通M-H,或者在Gibbs的每一步内部再嵌套一个M-H步骤,即所谓"Metropolis-within-Gibbs")
- 深挖追问:"即使条件分布都可以直接采样,Gibbs采样在什么情况下反而收敛得比较慢?"(如果多个维度之间高度相关(比如二维正态的相关系数接近1),Gibbs采样"一次只动一个维度"的更新方式会导致链条在高相关性方向上移动得非常缓慢——每次只能沿坐标轴方向小步挪动,而目标分布的"长轴"是斜着的,这种情形需要更多迭代次数才能充分探索空间,是Gibbs采样一个广为人知的局限)

**常见坑:**
- 把Gibbs采样的"总是接受"误解成"不会有收敛问题"——Gibbs采样仍然是一条马尔可夫链,同样需要burn-in、同样可能因为维度间强相关而收敛缓慢,"总是接受"只说明每一步提议设计得很巧妙,不代表整条链的收敛速度总是很快。
- 混淆条件分布p(θ₁\|θ₂)和边际分布p(θ₁)——Gibbs采样每一步用的是条件分布(依赖另一个变量当前的具体取值,每次采样都不同),不是边际分布(固定不变,和另一个变量取值无关)。

---

## 4. 收敛诊断 —— "最终会收敛"不等于"这500步已经收敛"

**定义与记号:** MCMC链条从任意初始点出发,需要经过一段"预热"时间才能真正开始按照目标分布的比例访问参数空间(burn-in/warm-up期)——如果初始点离目标分布的高密度区域很远,burn-in不充分就直接把早期样本纳入统计量计算,会系统性地污染估计结果。收敛诊断:判断MCMC链条是否已经进入"稳定态"的方法,常见实用手段包括目测轨迹图(trace plot)、分段计算统计量看结果是否已经稳定。

**一句话:** MCMC给出的理论保证是"链条跑得足够久,最终会收敛到目标分布",但这个保证不包含"从第一步开始就已经是目标分布的样本"这层含义,把还没收敛的早期样本也当作有效样本使用,是MCMC实践中最常见、也最容易被忽视的错误来源之一。

**数学推导/说明:** 构造具体反例:目标分布N(0,1),初始点故意设在θ₀=50(离目标均值极远)。链条需要经过若干步才能从50"游走"回到0附近——把采样序列切分成若干连续区间,分别计算各区间的样本均值:早期区间的均值会明显偏高(还在从50向0"下降"的过程中,带着初始点的痕迹),后期区间的均值会已经收敛到接近真值0——这个分段均值的变化轨迹,直接用数字量化了burn-in不足的具体危害,不是"这样不好"的定性说法。

下面是"目测轨迹图"具体在看什么的真实呈现(不是画出来的示意图:用下面"可运行例子"里同一个`metropolis_hastings`函数、相同参数(θ₀=50、step_size=1.0)、相同种子42真实采样后,按50步一个桶重新取均值——比代码块自带的10段×200步分法更细,能更清楚地看出下降的形状;字符数直接表示数值大小。这组数字和随后的10段均值表来自同一次真实运行,只是分桶粒度不同,读者用同样的函数、同样的种子重新采样后自行按50步分桶,会得到完全相同的数字):

```
step    0- 50: θ≈42.47 |##########################################  (刚出发, 仍在从50快速下降)
step   50-100: θ≈21.18 |#####################                       (继续快速下降)
step  100-150: θ≈ 3.92 |####                                        (已经跌入目标分布量级)
step  150-200: θ≈ 0.59 |#                                           (基本进入稳定波动区间)
step  200-2000: 此后到第2000步一共36个"50步桶", 桶均值全部落在[-0.62, 1.26]区间内,
                 不再有任何单向趋势, 是典型的"已收敛、只剩随机波动"的形态(而不是继续下降)
```

这就是trace plot要看的核心形状:前面一小段有明显的、单方向的下降趋势(还没收敛),后面一大段是围绕真值上下随机波动、没有趋势(已经收敛)——burn-in就是把前面那一小段有趋势的部分丢掉。换成"可运行例子"代码块里按10段(每段200步)算出来的分段均值(同一次真实运行,数字完全对应),更容易看出这个反差有多大:

```
第 1 段 [   0: 200) 均值 = 17.0427   <- 严重偏离真值0, 还带着起点θ₀=50的痕迹
第 2 段 [ 200: 400) 均值 = -0.1987
第 3 段 [ 400: 600) 均值 = -0.1667
第 4 段 [ 600: 800) 均值 =  0.0259
第 5 段 [ 800:1000) 均值 = -0.0330
第 6 段 [1000:1200) 均值 =  0.2746
第 7 段 [1200:1400) 均值 =  0.2798
第 8 段 [1400:1600) 均值 =  0.1958
第 9 段 [1600:1800) 均值 =  0.3105
第10段 [1800:2000) 均值 =  0.2768   <- 已经和真值0接近, 且和第2~10段量级一致(没有继续下降的趋势)
```

第1段的均值(17.04)比第2~10段里振幅最大的一段(第9段,0.31)还大了约55倍,这就是"前面一段和后面一段统计特性不同"这个现象的具体数字证据。也正因为这一段被起点污染的样本存在,下面代码里`naive_mean`(完全不做burn-in处理,真实跑出来是1.8008)比`clean_mean`(丢弃前20%即前400步,真实跑出来是0.1455)整整大了超过10倍——一小段还没收敛的早期样本,足以把全局均值估计拖到严重偏离真值的程度,这正是burn-in处理不能省略的直接原因。

**底层机制/为什么这样设计:** 为什么"链条最终会收敛"这个理论保证,不能自动确保任意一段实际运行的有限长度采样都是"高质量"的?因为MCMC的收敛性定理是渐进性质(n→∞时成立),不对任何具体的有限n给出精确的收敛程度保证——链条离目标分布还有多远,取决于目标分布形状、提议分布设计、初始点选择,这些理论定理本身不会告诉你,需要靠诊断方法(而不是理论保证)在具体应用中把关。

**AI研究/工程场景:** 任何使用MCMC做贝叶斯推断的实际项目,报告后验估计结果之前,检查trace plot、丢弃合理的burn-in样本、跑多条独立链条对比是否收敛到同一个分布(超出本文范围的Gelman-Rubin诊断),是标准的、不可省略的分析流程步骤——省略这一步直接报告"MCMC给出的后验均值是X",而不说明是否验证过收敛性,在严肃的贝叶斯分析工作里是一个明显的方法论缺陷。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

def log_target(theta):
    return -0.5 * theta ** 2

def metropolis_hastings(log_target, theta0, n_steps, step_size, rng):
    samples = np.zeros(n_steps)
    theta = theta0
    log_p_theta = log_target(theta)
    for i in range(n_steps):
        theta_proposed = theta + rng.normal(0, step_size)
        log_p_proposed = log_target(theta_proposed)
        if np.log(rng.random()) < log_p_proposed - log_p_theta:
            theta, log_p_theta = theta_proposed, log_p_proposed
        samples[i] = theta
    return samples

n_steps = 2000
extreme_start = 50.0  # 故意选一个离目标分布均值(0)极远的初始点
samples = metropolis_hastings(log_target, extreme_start, n_steps, 1.0, rng)

# 分段计算均值, 观察收敛轨迹
n_segments = 10
segment_size = n_steps // n_segments
segment_means = [samples[i * segment_size:(i + 1) * segment_size].mean() for i in range(n_segments)]

# 核心断言1: 第一段(链条还在从50游走回来的路上)应该明显偏离真值0
assert abs(segment_means[0]) > 5.0, f"the first segment should still show heavy contamination from the extreme start, got {segment_means[0]:.4f}"

# 核心断言2: 最后一段应该已经收敛到接近真值0
assert abs(segment_means[-1]) < 1.0, f"the last segment should have converged near 0, got {segment_means[-1]:.4f}"

# 核心断言3: 不做burn-in处理会显著污染整体均值估计, 丢弃前20%后估计明显改善
naive_mean = samples.mean()
burn_in = n_steps // 5
clean_mean = samples[burn_in:].mean()
assert abs(naive_mean) > abs(clean_mean) * 3, \
    f"discarding burn-in should substantially improve the mean estimate: naive={naive_mean:.4f} clean={clean_mean:.4f}"
assert abs(clean_mean) < 1.0, f"post-burn-in estimate should be reasonably close to the true value 0, got {clean_mean:.4f}"

print(f"segment means (10 segments across {n_steps} steps): {[round(m, 3) for m in segment_means]}")
print(f"naive mean (no burn-in) = {naive_mean:.4f}  vs  clean mean (20% burn-in discarded) = {clean_mean:.4f}  (true value = 0)")
```

**面试怎么问+追问链**(诊断真实数据新题型):
- Q:"给你一条MCMC采样得到的参数轨迹图,前200步和后1800步的样本呈现出明显不同的均值水平,你会怎么处理这份数据?"
- 追问1:"这个现象说明了什么?"(很可能是burn-in不充分——初始点选得离目标分布较远,链条前一段时间还在"游走"向目标分布靠近的过程中,还没有真正开始按目标分布比例采样;需要丢弃这部分早期样本,只用链条已经"稳定"之后的样本做统计推断)
- 深挖追问:"怎么判断'从第几步开始算稳定'这个截断点该怎么定,有没有更严谨的方法而不是凭肉眼看轨迹图?"(除了目测轨迹图,更严谨的做法包括跑多条从不同初始点出发的独立链条,比较链间方差和链内方差是否已经趋于一致(Gelman-Rubin诊断的核心思想);或者直接用统计量(比如分段均值的变化趋势)量化"轨迹是否还在明显变化",而不是单纯依赖主观判断)

**常见坑:**
- 完全不做任何burn-in处理,把从初始点开始的全部样本都纳入统计量计算,尤其在初始点选得比较随意(比如默认从0开始,而目标分布的高密度区域离0很远)时,这个错误会显著污染最终结果。
- burn-in丢弃得"过度保守"(比如丢弃了90%的样本),虽然不会引入偏差,但浪费了大量计算资源换来的有效样本,在计算资源有限的场景下也是一种需要权衡的成本。

---

## 5. 接受率与步长调优 —— 步子太大或太小都低效

**定义与记号:** M-H算法提议分布的"步长"(比如正态提议分布的标准差)直接决定接受率——步长过大,候选点经常落在密度很低的区域,大部分被拒绝,链条长时间停留原地不动;步长过小,候选点总是离当前位置很近,几乎总能被接受,但链条每一步移动的距离都很小,需要极多步数才能充分探索整个分布。存在一个"恰到好处"的步长区间,能让链条在"移动幅度"和"接受概率"之间取得平衡,最高效地探索目标分布。

**一句话:** 步长像走路的步幅——步子迈得太大,经常一脚踩空(候选点密度太低,被拒绝),原地不动;步子迈得太小,虽然每步都走得稳(容易被接受),但要走很远的路(充分探索整个分布)就要走非常多步,两种极端都低效,存在一个恰到好处的步幅。

**数学推导:** 用**有效样本量**(effective sample size, ESS)量化"探索效率":如果MCMC样本之间存在强的序列相关性(autocorrelation,相邻样本很像,步长过小时特别严重),那么N个MCMC样本携带的信息量远少于N个真正独立样本,ESS就是"这N个相关样本大约相当于多少个独立样本"这个折算值,ESS越接近N越好。步长过小时相邻样本几乎一样,自相关性极高,ESS远小于N;步长过大时虽然一旦接受移动距离较大,但大部分提议被拒绝(链条长时间停留原地),同样导致相邻样本高度重复,自相关性依然很高——用滞后1阶自相关系数可以直接数值验证这个"两头高、中间低"的U形关系。

**底层机制/为什么这样设计:** 为什么"接受率"和"探索效率"之间不是单调关系,而是存在一个中间最优点?因为探索效率同时依赖"能不能成功移动"(接受率)和"移动的时候能走多远"(步长)这两个因素的乘积效应——步长增大会提高单次成功移动的位移量,但同时降低接受概率,这两个效应此消彼长,数学上可以证明存在一个使"平均每步实际探索的期望距离"最大化的中间步长(严格推导超出本文范围,但优化问题的存在性和中间最优点的直觉可以直接数值演示)。

**AI研究/工程场景:** 现代概率编程框架(PyMC、Stan)在实际使用中通常会自动调优提议分布的步长(自适应MCMC,在采样早期动态调整步长直到接近目标接受率区间,再固定下来做正式采样),但理解"为什么需要调优步长、调优的目标是什么"这个基本原理,是正确使用这些自动化工具、诊断"采样效果不好是不是因为步长设置有问题"的必要知识基础,不能把这些工具完全当作不需要理解内部机制的黑箱使用。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

def log_target(theta):
    return -0.5 * theta ** 2

def metropolis_hastings(log_target, theta0, n_steps, step_size, rng):
    samples = np.zeros(n_steps)
    theta = theta0
    log_p_theta = log_target(theta)
    n_accept = 0
    for i in range(n_steps):
        theta_proposed = theta + rng.normal(0, step_size)
        log_p_proposed = log_target(theta_proposed)
        if np.log(rng.random()) < log_p_proposed - log_p_theta:
            theta, log_p_theta = theta_proposed, log_p_proposed
            n_accept += 1
        samples[i] = theta
    return samples, n_accept / n_steps

def lag1_autocorr(x):
    x = x - x.mean()
    return np.sum(x[:-1] * x[1:]) / np.sum(x ** 2)

step_sizes = [0.05, 0.3, 1.0, 3.0, 20.0]
n_steps = 20_000
results = {}
for s in step_sizes:
    samples, acc_rate = metropolis_hastings(log_target, 0.0, n_steps, s, rng)
    results[s] = (acc_rate, lag1_autocorr(samples[2000:]))

for s, (acc, ac) in results.items():
    print(f"step_size={s:5.2f}: accept_rate={acc:.4f}  lag1_autocorr={ac:.4f}")

# 核心断言1: 步长过小 -> 接受率很高, 但自相关性也很高(探索效率低)
assert results[0.05][0] > 0.95, f"very small step size should give a very high acceptance rate, got {results[0.05][0]:.4f}"
assert results[0.05][1] > 0.9, f"very small step size should give heavily autocorrelated samples, got {results[0.05][1]:.4f}"

# 核心断言2: 步长过大 -> 接受率很低, 自相关性同样很高(探索效率低, 长时间停留原地)
assert results[20.0][0] < 0.15, f"very large step size should give a very low acceptance rate, got {results[20.0][0]:.4f}"
assert results[20.0][1] > 0.8, f"very large step size should also give heavily autocorrelated samples, got {results[20.0][1]:.4f}"

# 核心断言3: 中间步长应该明显优于两个极端(自相关性明显更低)
best_middle_autocorr = min(results[1.0][1], results[3.0][1])
assert best_middle_autocorr < 0.8, \
    f"an intermediate step size should achieve notably lower autocorrelation than both extremes, got {best_middle_autocorr:.4f}"
assert best_middle_autocorr < results[0.05][1] and best_middle_autocorr < results[20.0][1], \
    "intermediate step sizes should outperform both the too-small and too-large extremes"

print("=> both step_size=0.05 (too small) and step_size=20.0 (too large) give high autocorrelation;")
print("   intermediate step sizes explore the distribution far more efficiently.")
```

**面试怎么问+追问链**(方案批判迭代轴):
- Q:"你的MCMC采样接受率是98%,是不是说明采样效果很好?"
- 追问1:"接受率高就一定好吗?"(不一定——接受率98%通常意味着提议的步长太小,候选点总是离当前位置很近、几乎总被接受,但这样链条每一步移动幅度很小,要真正探索完整个目标分布的形状需要非常多步,相邻样本之间高度相关,有效样本量会远小于名义上的采样步数)
- 深挖追问:"那接受率是不是应该越低越好,这样至少说明步长够大?"(也不是——接受率过低(比如5%)意味着绝大多数提议的候选点都被拒绝,链条长时间停留在原地不动,同样导致相邻样本高度相关、探索效率低下;真正想要的是一个中间区间的接受率,过高和过低都是问题,不是"接受率越低说明步子迈得越大就越好"这种单向的直觉)

**常见坑:**
- 把"接受率高"直接等同于"采样质量好"——接受率只是一个诊断线索,不是直接的质量指标,过高的接受率往往暗示步长过小、探索效率低下这个问题,而不是好消息。
- 只调优接受率这一个指标,不去看更直接的探索效率证据(自相关性、有效样本量、多条链是否收敛到一致结果)——接受率只是一个方便计算的代理指标,不是探索效率的直接度量,最终还是要看更本质的诊断量。

---

下一篇:[13-bayesian-applications.md](13-bayesian-applications.md) —— 板块III收官,把贝叶斯推断基础和MCMC采样落地到贝叶斯A/B测试、可信区间语义辨析、模型比较、贝叶斯早停这些更贴近AI/ML工程场景的具体应用。
