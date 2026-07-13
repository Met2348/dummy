# 17 · 分布漂移与监控深挖(Distribution Shift & Monitoring)

> 总览见 [00-roadmap.md](00-roadmap.md)

16类处理的是"训练时的规律,能不能预测更大规模会怎样";本文处理一个方向相反但同样重要的问题——"部署之后,真实世界的数据分布本身在变化,训练时学到的规律还适用吗"。这是MLOps里"模型监控"这个环节的统计基础。本文复用04类知识点7的KS检验方法论,引入KL散度和PSI两个新工具,并重点拆解一个容易被监控系统忽视的盲区——概念偏移。

**环境声明:** 全部代码在仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1)下真实跑通,随机抽样固定种子(`np.random.default_rng(42)`)。KL散度手写实现,与`scipy.stats.entropy`交叉验证;KS检验复用04类知识点7的方法论,与`scipy.stats.ks_2samp`交叉验证。

---

## 1. KL散度 —— 用一个错误的分布假设,要付出多少代价

**定义与记号:** KL散度(Kullback-Leibler divergence):衡量两个概率分布P、Q之间"差异"的非对称度量,D_KL(P\|\|Q)=Σ_x P(x)·log(P(x)/Q(x))。在分布漂移监控场景下,通常P是"当前生产环境观测到的分布",Q是"训练时的基准分布(reference distribution)",D_KL(P\|\|Q)量化"当前分布相对训练分布偏离了多少"。

**一句话:** KL散度可以理解成"如果假设数据来自Q分布,但实际来自P分布,你会为这个错误假设付出多少'信息量'的代价"——数值越大,说明P和Q差异越大,0代表两个分布完全相同。

**数学推导:** D_KL(P\|\|Q)=Σ P(x)log P(x) - Σ P(x)log Q(x)=-H(P)+H(P,Q)(H(P)是P的信息熵,H(P,Q)是交叉熵)。用Jensen不等式可以证明D_KL(P\|\|Q)≥0恒成立(当且仅当P=Q时取等号):-D_KL(P\|\|Q)=Σ P(x)log(Q(x)/P(x))≤log(Σ P(x)·Q(x)/P(x))=log(Σ Q(x))=log(1)=0(log是凹函数,直接套用Jensen不等式)。

**底层机制/为什么这样设计:** 为什么KL散度是**非对称**的,这个非对称性在监控场景下有什么实际含义?因为KL散度的定义按P(x)(当前分布)加权——如果当前分布P在某个区域有较高概率、但基准分布Q在这个区域概率极低,log(P(x)/Q(x))这一项会变得很大且被P(x)放大;反过来,如果是Q覆盖了一个P几乎不会出现的区域,这部分差异对D_KL(P\|\|Q)的贡献很小(按P(x)≈0加权)——这意味着D_KL(P\|\|Q)对"生产环境出现了训练时没见过的新模式"格外敏感,这正是监控场景最关心的方向性问题。

**AI研究/工程场景:** 特征/预测分布监控是MLOps里的标准实践——把当前生产环境一段时间窗口内的特征分布或模型预测分布,和训练时的基准分布做KL散度比较,如果散度超过某个阈值就触发告警,提示可能存在数据分布漂移,模型的预测可靠性可能已经打了折扣。

**可运行例子:**
```python
import numpy as np
from scipy.stats import entropy

P = np.array([0.1, 0.4, 0.3, 0.2])
Q = np.array([0.25, 0.25, 0.25, 0.25])

def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))

kl_pq, kl_qp = kl_divergence(P, Q), kl_divergence(Q, P)
scipy_kl_pq, scipy_kl_qp = entropy(P, Q), entropy(Q, P)  # scipy.stats.entropy(pk, qk) = KL(pk||qk)

# 核心断言1: 手写实现和scipy应该精确一致
assert abs(kl_pq - scipy_kl_pq) < 1e-9, f"hand-written KL should match scipy exactly: {kl_pq} vs {scipy_kl_pq}"
assert abs(kl_qp - scipy_kl_qp) < 1e-9, f"hand-written KL should match scipy exactly: {kl_qp} vs {scipy_kl_qp}"

# 核心断言2: 非对称性 -- 两个方向的散度应该明显不同
assert abs(kl_pq - kl_qp) > 0.01, f"KL divergence should be asymmetric: KL(P||Q)={kl_pq:.4f} KL(Q||P)={kl_qp:.4f}"

# 核心断言3: 非负性(大量随机分布对上验证)
rng = np.random.default_rng(42)
min_kl = min(kl_divergence(rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))) for _ in range(1000))
assert min_kl > -1e-9, f"KL divergence should always be non-negative, got min={min_kl:.6f}"

print(f"KL(P||Q) = {kl_pq:.4f}  (matches scipy: {scipy_kl_pq:.4f})")
print(f"KL(Q||P) = {kl_qp:.4f}  (matches scipy: {scipy_kl_qp:.4f})")
print(f"asymmetric: |KL(P||Q) - KL(Q||P)| = {abs(kl_pq - kl_qp):.4f}")
print(f"min KL over 1000 random distribution pairs = {min_kl:.6f}  (confirms non-negativity)")
```

**面试怎么问+追问链**(决策依据追问轴):
- Q:"监控数据漂移时,应该算D_KL(生产分布\|\|训练分布)还是D_KL(训练分布\|\|生产分布)?"
- 追问1:"这两个方向有什么本质区别?"(D_KL(生产\|\|训练)对"生产环境出现训练时没见过的新模式"格外敏感;D_KL(训练\|\|生产)则对"训练时常见但生产环境变少了的模式"更敏感,方向选择取决于监控真正关心的是哪种变化)
- 深挖追问:"如果生产环境出现了训练数据里完全没有见过的取值,D_KL(生产\|\|训练)会发生什么?"(log(P/Q)在Q=0处趋于无穷大,D_KL会变成无穷大或者在数值计算中直接产生NaN,这是KL散度在实践中一个真实的数值稳定性问题,常见的工程解决方案是给基准分布做平滑,或者用知识点3的PSI这种对0值更宽容的替代指标)

**常见坑:**
- 把KL散度当作对称的"距离"来使用,忽视了它本质上是非对称的,不满足距离度量的三角不等式等公理。
- 直接对包含0概率的分布计算KL散度而不做任何平滑处理,导致除0或log(0)的数值问题。

---

## 2. KS检验用于漂移检测 —— 不挑分布形状的通用检测器

**定义与记号:** 复用04类知识点7已建立的KS检验(Kolmogorov-Smirnov test)方法论,应用到分布漂移检测——比较当前生产环境数据和训练时基准数据的经验累积分布函数(ECDF)之间的最大距离(D统计量),D超过临界值就判定两个时间段的数据分布存在统计显著的差异,可能发生了分布漂移。

**一句话:** KS检验不需要对分布形状做任何参数假设,只是比较两条经验累积分布曲线"最大能拉开多远",这个"不挑分布形状"的特性使它成为分布漂移监控里应用广泛的通用工具,不需要针对每种特征的分布类型单独设计检验方法。

**数学推导/说明:** 构造模拟的"生产数据流",分成多个连续时间窗口:①没有真实分布偏移时,验证KS检验在各窗口之间比较的假阳性率控制在合理水平附近;②某个时间点后引入真实分布偏移(均值漂移),验证KS检验能正确地在偏移发生后的窗口检测出显著差异,量化检测能力。

**底层机制/为什么这样设计:** 为什么KS检验对分布漂移监控特别合适(相比只比较均值的t检验)?因为真实的分布漂移可能表现为多种形式——不仅是均值变化,也可能是方差、偏度或分布形状本身的改变,t检验对后面几种漂移形式可能完全不敏感(04类知识点4/6已讨论过t检验对形状差异不敏感、KS检验能捕捉形状差异这个对比),而KS检验因为比较整条累积分布曲线,能对更广泛类型的分布变化保持敏感,这是它被广泛用作"通用漂移检测器"的核心原因。

**AI研究/工程场景:** 特征监控系统里,对每个重要特征都维护一个训练时的基准分布,定期用KS检验比较当前数据和基准分布,是业界常见的自动化监控流程——不需要为每个特征单独设计检验逻辑,同一套KS检验框架可以应用到数值型特征的任何分布形状上。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

window_size = 200
shift_start_window = 10
n_windows = 20
n_trials = 500  # 重复多条模拟数据流, 取稳健的整体误报率/检出率

fp_count, fp_total = 0, 0
det_count, det_total = 0, 0

for _ in range(n_trials):
    reference = rng.normal(0, 1, window_size)  # 基准窗口(训练时的分布)
    for w in range(1, n_windows):
        if w < shift_start_window:
            data = rng.normal(0, 1, window_size)  # 无漂移
        else:
            data = rng.normal(0.5, 1, window_size)  # 均值漂移0.5
        _, p_val = stats.ks_2samp(reference, data)
        flagged = p_val < 0.05
        if w < shift_start_window:
            fp_total += 1
            fp_count += flagged
        else:
            det_total += 1
            det_count += flagged

fpr = fp_count / fp_total
detection_rate = det_count / det_total

# 核心断言1: 无漂移时的假阳性率应该接近名义alpha=0.05, 不应严重膨胀
assert fpr < 0.08, f"false positive rate should stay near nominal alpha, got {fpr:.4f}"

# 核心断言2: 真实漂移发生后, 检出率应该很高
assert detection_rate > 0.9, f"detection rate after a real mean shift should be high, got {detection_rate:.4f}"

print(f"false positive rate (no real drift, {fp_total} checks) = {fpr:.4f}")
print(f"detection rate (real mean-shift of 0.5, {det_total} checks) = {detection_rate:.4f}")
```

**面试怎么问+追问链**(诊断真实数据新题型):
- Q:"监控系统显示某个特征这周和基准分布相比KS检验p<0.01,应该立即触发告警吗?"
- 追问1:"在下结论之前应该先检查什么?"(和05-06类反复强调的原则一样,首先要考虑多重比较问题——如果同时监控几百个特征,即使所有特征真实都没有漂移,按5%的名义假阳性率,期望也会有一部分特征"意外"显著,需要对多特征同时监控做类似05类BH-FDR的校正)
- 深挖追问:"如果确认某个特征确实存在统计显著的分布偏移,但KS统计量D的绝对值很小(比如D=0.02),这个漂移在业务上重要吗?"(统计显著不等于业务重要,03类知识点5"效应量"原则在监控场景的复现——样本量很大时,极其微小的分布偏移也可能被判定为"统计显著",需要结合业务阈值综合判断)

**常见坑:**
- 同时监控大量特征,对每个特征分别做KS检验,不做多重比较校正,导致告警系统产生大量"狼来了"式的假阳性告警。
- 把"统计显著的分布偏移"直接等同于"需要立即处理的业务问题",不结合效应量和实际业务影响做进一步判断。

---

## 3. PSI(Population Stability Index)—— 分箱版本的对称化KL散度

**定义与记号:** PSI(群体稳定性指数):金融风控/信用评分领域历史上最常用的分布漂移监控指标之一——把特征值域切分成若干分箱,分别计算当前分布和基准分布落在每个分箱的比例(P_i和Q_i),PSI=Σ_i(P_i-Q_i)·ln(P_i/Q_i)。业界经验分级:PSI<0.1基本稳定,0.1≤PSI<0.25轻度漂移需关注,PSI≥0.25显著漂移建议采取行动。

**一句话:** PSI可以理解成"分箱版本的对称化KL散度"——它把连续特征离散化成若干区间再比较比例分布,并对P、Q做了对称处理,这让它比原始KL散度更适合在业务报告里直接使用一个单一数字配合几条容易解释的经验分级规则。

**数学推导:** PSI=Σ_i(P_i-Q_i)ln(P_i/Q_i)=Σ_i P_i·ln(P_i/Q_i)+Σ_i Q_i·ln(Q_i/P_i)=D_KL(P\|\|Q)+D_KL(Q\|\|P)——PSI恰好等于两个方向KL散度之和,这个恒等式可以直接代数验证,不是"类似"或"启发式相关"。PSI因为是两个方向散度的和,天然具备对称性(PSI(P,Q)=PSI(Q,P)),这是它相对单方向KL散度的一个实用优势。

**底层机制/为什么这样设计:** 为什么PSI选择先分箱、再计算离散分布上的散度,而不是直接处理连续分布?分箱操作是一种正则化/平滑手段——它天然规避了"某个具体数值点上概率密度为0"这种KL散度容易遇到的数值病态问题(只要每个箱子里都有一定数量的样本,箱子的比例就不会精确为0),同时让PSI计算对具体的数据点噪声不那么敏感,这是金融风控这类对可解释性、可复现性要求很高的行业偏爱PSI的实际原因。

**AI研究/工程场景:** PSI最初在信用评分卡领域被广泛使用,监控评分卡模型输出分数或关键特征分布是否随时间发生显著偏移;这套方法论近年被推广到更广泛的机器学习模型监控场景,因为它的经验分级规则提供了一个业务方不需要理解统计理论就能直接使用的、标准化的沟通语言。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

def compute_psi(reference, current, bins):
    ref_counts, edges = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_prop = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_prop = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
    return psi, ref_prop, cur_prop

def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))

reference = rng.normal(0, 1, 2000)
current_no_shift = rng.normal(0, 1, 2000)
current_shift = rng.normal(0.5, 1, 2000)
bin_edges = np.histogram_bin_edges(reference, bins=10)

psi_no_shift, _, _ = compute_psi(reference, current_no_shift, bin_edges)
psi_shift, ref_p, cur_p_s = compute_psi(reference, current_shift, bin_edges)

# 核心断言1: 无漂移时PSI应该低于"基本稳定"阈值0.1, 有漂移时应该超过"显著漂移"阈值0.25
assert psi_no_shift < 0.1, f"PSI with no real shift should be below the 'stable' threshold, got {psi_no_shift:.4f}"
assert psi_shift > 0.25, f"PSI with a real mean shift should exceed the 'significant drift' threshold, got {psi_shift:.4f}"

# 核心断言2: PSI = KL(P||Q) + KL(Q||P) 这个恒等式应该精确成立
kl_sum = kl_divergence(cur_p_s, ref_p) + kl_divergence(ref_p, cur_p_s)
assert abs(psi_shift - kl_sum) < 1e-6, f"PSI should exactly equal the sum of both-direction KL divergences: {psi_shift:.6f} vs {kl_sum:.6f}"

# 对比同一份数据上的KS检验
d_stat, p_val = stats.ks_2samp(reference, current_shift)

print(f"PSI (no shift) = {psi_no_shift:.4f}  (< 0.1, stable)")
print(f"PSI (real shift) = {psi_shift:.4f}  (> 0.25, significant) = KL(P||Q)+KL(Q||P) = {kl_sum:.4f}")
print(f"KS test on the same shifted data: D={d_stat:.4f}, p={p_val:.2e}")
```

**面试怎么问+追问链**(决策依据追问轴):
- Q:"PSI和KS检验都能用来监控分布漂移,该选哪个?"
- 追问1:"两者最本质的区别是什么?"(PSI是一个描述性的、连续的"距离数值"(配合经验分级规则),不直接对应一个假设检验框架下的p值;KS检验是一个正式的假设检验,有明确的原假设和基于渐进理论的p值,可以直接回答"这个差异在统计上是否显著"这个问题——PSI更像一个"体检指标",KS检验更像一个"正式的诊断结论")
- 深挖追问:"PSI的分箱方式会不会影响最终的PSI数值?"(会——分箱方式是PSI计算里一个需要人为决定的超参数,不同的分箱策略可能得出不同的PSI数值,这是PSI相对KS检验(不需要任何分箱)的一个实践弱点,业界通常有一些经验规则但缺乏严格的理论指导)

**常见坑:**
- 把PSI的经验分级阈值(0.1/0.25)当作放之四海而皆准的严格标准,不结合具体业务场景去调整判断——这些阈值来自金融风控领域的历史经验,不是从统计理论严格推导出来的普适常数。
- 忽视分箱方式对PSI数值的影响,不同团队/不同时间用不一致的分箱策略计算PSI,导致横向或纵向比较失去意义。

---

## 4. 协变量偏移 vs 概念偏移 —— 特征分布监控看不见的盲区

**定义与记号:** 协变量偏移(covariate shift):输入特征X的边际分布P(X)发生变化,但X和目标Y之间的条件关系P(Y\|X)保持不变。概念偏移(concept shift):P(X)可能不变,但X和Y之间的条件关系P(Y\|X)本身发生了变化。两者都会导致模型性能下降,但根源、检测方法和应对策略完全不同。

**一句话:** 协变量偏移是"考题的题型分布变了,但每种题型的正确解法没变";概念偏移是"考题看起来还是老样子,但正确答案本身变了"——前者理论上只要模型学到的P(Y\|X)是准确的,遇到协变量偏移依然能给出正确预测;后者是模型学到的映射关系本身已经过时,不管训练数据分布如何调整都无法回避需要更新模型这个事实。

**数学推导/说明:** 知识点1-3(KL散度/KS检验/PSI)监控的都是P(X)这个边际分布的变化,天然适合检测协变量偏移;但这些方法**无法直接检测**概念偏移,因为概念偏移可能发生在P(X)完全不变的情况下,纯粹监控特征分布对这类偏移是"盲"的——检测概念偏移需要额外监控模型的**预测准确率**随时间的变化,或者监控预测置信度分布是否发生系统性变化。

**底层机制/为什么这样设计:** 为什么区分这两种偏移在实践中很重要?因为最优应对策略不同——纯粹的协变量偏移,如果模型的函数形式本身足够灵活,有时候不需要立即重新训练,可以用重要性加权这类相对轻量的技术适配,不需要收集全新的标注数据;概念偏移则意味着历史标注数据里的Y标签所反映的规律已经过时,不管怎么调整训练数据的采样权重都无法弥补,必须重新收集反映当前真实P(Y\|X)的新标注数据才能解决问题——诊断错了偏移类型,可能选择成本高昂但没有针对性的应对方案。

**AI研究/工程场景:** 推荐系统里季节性的用户兴趣分布变化通常更接近协变量偏移(用户群体构成变了,但"喜欢这类商品的人还是倾向于购买"这个关系本身没变);而重大社会事件、平台规则改变导致的用户行为模式根本性转变,通常是概念偏移的典型案例——区分这两种情况,直接决定了监控团队应该采取"观察+轻量调整"还是"立即启动重新标注+重训"这两种成本天差地别的应对路径。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

n = 1000
true_coef = 2.0

# 训练数据(基准), 拟合一个简单线性模型
X_train = rng.normal(0, 1, n)
Y_train = true_coef * X_train + rng.normal(0, 0.5, n)
model_coef = np.sum(X_train * Y_train) / np.sum(X_train ** 2)

# 场景A: 协变量偏移 -- X分布变了(均值偏移), 但Y=f(X)关系不变
X_covshift = rng.normal(2.0, 1, n)
Y_covshift = true_coef * X_covshift + rng.normal(0, 0.5, n)  # 同样的真实关系
mse_covshift = np.mean((Y_covshift - model_coef * X_covshift) ** 2)

# 场景B: 概念偏移 -- X分布不变, 但Y=f(X)关系本身变了
X_conceptshift = rng.normal(0, 1, n)  # 和训练时同分布
Y_conceptshift = (true_coef + 3.0) * X_conceptshift + rng.normal(0, 0.5, n)  # 系数变了
mse_conceptshift = np.mean((Y_conceptshift - model_coef * X_conceptshift) ** 2)

# 基准: 无偏移
X_nodrift = rng.normal(0, 1, n)
Y_nodrift = true_coef * X_nodrift + rng.normal(0, 0.5, n)
mse_nodrift = np.mean((Y_nodrift - model_coef * X_nodrift) ** 2)

# 特征分布监控(KS检验)
_, p_cov = stats.ks_2samp(X_train, X_covshift)
_, p_concept = stats.ks_2samp(X_train, X_conceptshift)

# 核心断言1: 协变量偏移下, 特征分布监控能明确检测到变化, 但模型预测误差基本不受影响(关系没变)
assert p_cov < 0.001, f"KS test should clearly detect the covariate shift in X, got p={p_cov:.4f}"
assert mse_covshift < mse_nodrift * 1.5, \
    f"covariate shift alone shouldn't hurt predictions much when P(Y|X) is preserved: nodrift={mse_nodrift:.4f} covshift={mse_covshift:.4f}"

# 核心断言2: 概念偏移下, 特征分布监控完全检测不到异常(这是盲区), 但模型预测误差急剧上升
assert p_concept > 0.3, f"KS test on X should show NO significant difference under pure concept shift, got p={p_concept:.4f}"
assert mse_conceptshift > mse_nodrift * 10, \
    f"concept shift should severely degrade predictions despite unchanged feature distribution: nodrift={mse_nodrift:.4f} conceptshift={mse_conceptshift:.4f}"

print(f"baseline MSE (no drift) = {mse_nodrift:.4f}")
print(f"covariate shift: KS p={p_cov:.2e} (clearly flagged)  MSE={mse_covshift:.4f} (barely affected)")
print(f"concept shift:   KS p={p_concept:.4f} (NOT flagged -- blind spot!)  MSE={mse_conceptshift:.4f} (severely degraded)")
```

**面试怎么问+追问链**(诊断真实数据新题型):
- Q:"上线的模型精度下降了,监控显示所有特征的分布指标(PSI/KS)都在正常范围内,这说明模型没有遇到分布漂移问题吗?"
- 追问1:"特征分布正常但精度下降,可能是什么原因?"(很可能是概念偏移——特征本身的分布没有变化,但特征和目标之间的真实关系已经改变,纯粹的特征分布监控对这种情况是"盲"的,需要额外的监控手段,比如持续跟踪模型预测的实际准确率/校准度)
- 深挖追问:"如果没有及时的真实标签反馈,怎么尽早发现可能的概念偏移?"(可以监控代理指标——模型预测的置信度分布是否发生系统性变化、预测分数分布本身是否异常偏移,或者建立一个更快获得反馈的小规模影子标注流程)

**常见坑:**
- 只监控特征分布(PSI/KS/KL散度),就认为已经建立了完整的模型监控体系,忽视了这类方法对概念偏移完全不敏感这个根本局限。
- 一旦发现精度下降就不加区分地假设是协变量偏移(因为通常更容易通过重新加权解决,成本更低),而不去认真诊断是否其实是需要更彻底解决方案的概念偏移。

---

## 5. 监控阈值设定 —— ROC思路的监控版应用

**定义与记号:** 监控阈值设定:知识点1-3给出的量化指标本身是连续数值,需要设定一个阈值才能转化成"是否触发告警"这个二元决策——这个阈值的选择存在经典的误报(false positive)vs漏报(false negative)权衡,阈值越敏感(越低),误报率越高、漏报率越低;阈值越保守(越高),误报率越低、漏报率越高,这正是ROC曲线(receiver operating characteristic curve)思路在监控场景下的直接应用。

**一句话:** 监控告警系统本质上是一个二分类器(判断"当前是否存在真实的分布漂移"),和任何二分类器一样面临阈值选择的误报-漏报权衡,ROC曲线分析这套成熟的方法论可以直接搬过来用于系统性地选择监控阈值,而不是凭感觉设一个"看起来合理"的数字。

**数学推导/说明:** 构造数值实验:①无真实漂移的情形下重复模拟,对每个候选阈值统计触发告警的比例(误报率);②有真实漂移的情形下重复模拟,对每个候选阈值统计成功检测到漂移的比例(检出率)。这两组结果对应ROC曲线上不同的点,业务团队可以根据"更能容忍误报还是更能容忍漏报"这个实际需求,在曲线上选择合适的操作点(对应一个具体的阈值)。

**底层机制/为什么这样设计:** 为什么监控阈值的选择不能脱离具体业务场景、只依赖"业界经验值"?因为误报和漏报各自的实际代价在不同业务场景下差异巨大——医疗诊断、金融风控这类漏报代价极高的场景,应该倾向于更敏感(更低)的阈值,容忍更高的误报率;内容推荐这类误报代价相对较高(频繁误报导致监控团队"狼来了"式地失去警觉)、漏报代价相对温和的场景,应该倾向于更保守(更高)的阈值——业界经验值是一个合理的起点,不是不能根据具体场景调整的铁律。

**AI研究/工程场景:** 成熟的MLOps监控系统在部署告警规则之前,通常会用历史数据做"回测",模拟不同阈值下的误报率和检出率表现,结合具体业务对误报/漏报代价的实际容忍度系统性地选定阈值——这是04-06类"样本量计算/功效分析"这套方法论在"监控系统设计"这个新场景下的直接迁移应用,底层都是同一类"控制两类错误率、根据实际代价选择操作点"的统计决策问题。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

window_size = 200
n_trials = 800
candidate_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]

# 误报率: 无漂移情形下, D统计量超过阈值的比例
fpr_by_threshold = {t: 0 for t in candidate_thresholds}
for _ in range(n_trials):
    reference = rng.normal(0, 1, window_size)
    current = rng.normal(0, 1, window_size)
    d, _ = stats.ks_2samp(reference, current)
    for t in candidate_thresholds:
        fpr_by_threshold[t] += (d > t)
fpr_by_threshold = {t: v / n_trials for t, v in fpr_by_threshold.items()}

# 检出率: 中等幅度真实漂移下, D统计量超过阈值的比例
tpr_by_threshold = {t: 0 for t in candidate_thresholds}
for _ in range(n_trials):
    reference = rng.normal(0, 1, window_size)
    current = rng.normal(0.3, 1, window_size)  # 中等幅度漂移
    d, _ = stats.ks_2samp(reference, current)
    for t in candidate_thresholds:
        tpr_by_threshold[t] += (d > t)
tpr_by_threshold = {t: v / n_trials for t, v in tpr_by_threshold.items()}

fpr_list = [fpr_by_threshold[t] for t in candidate_thresholds]
tpr_list = [tpr_by_threshold[t] for t in candidate_thresholds]

# 核心断言1: 误报率和检出率都应该随阈值单调不增(经典的ROC式权衡; 阈值足够高时误报率可以精确降到0, 不必严格递减)
assert all(fpr_list[i] >= fpr_list[i + 1] for i in range(len(fpr_list) - 1)), f"FPR should be non-increasing with threshold: {fpr_list}"
assert all(tpr_list[i] >= tpr_list[i + 1] for i in range(len(tpr_list) - 1)), f"TPR should be non-increasing with threshold: {tpr_list}"

# 核心断言2: 最低阈值应该过于敏感(误报率高到不可用), 最高阈值应该过于保守(检出率很低)
assert fpr_list[0] > 0.5, f"the lowest threshold should be far too sensitive (high FPR), got {fpr_list[0]:.4f}"
assert tpr_list[-1] < 0.1, f"the highest threshold should be far too conservative (low TPR), got {tpr_list[-1]:.4f}"

for t in candidate_thresholds:
    print(f"threshold={t}: FPR={fpr_by_threshold[t]:.4f}  TPR={tpr_by_threshold[t]:.4f}")
print("=> no single threshold is 'correct' -- the choice depends on which error type the business can tolerate more")
```

**面试怎么问+追问链**(方案批判迭代轴,收束全文):
- Q:"监控系统的PSI告警阈值该设多少,直接用业界惯例的0.25可以吗?"
- 追问1:"0.25这个数字是怎么来的,直接套用有什么风险?"(0.25是金融风控领域长期实践积累下来的经验分级,不是针对当前具体业务场景推导出来的最优值——不同特征的PSI在"无漂移"情形下天然的波动幅度可能不同,直接套用一个通用阈值,可能对某些波动性本身就大的特征产生大量误报,对某些天生稳定的特征又不够敏感)
- 深挖追问:"如果想要一个更适合当前具体场景的阈值,应该怎么科学地确定?"(用本知识点的数值回测方法论——利用历史上"已知没有发生真实漂移"的时间段数据,模拟计算出监控指标在无漂移情形下的自然波动范围,以此为基准校准阈值,而不是照搬一个通用经验值;如果历史上有过已知的真实漂移事件记录,还可以进一步用这些事件验证候选阈值的检出率表现)

**常见坑:**
- 不做任何回测就直接采用业界惯例阈值,不验证这个阈值在当前具体场景下的误报率是否处于可接受范围。
- 只关注降低误报率,把阈值设得过于保守,而不评估这样做对应的漏报率会有多高、这个漏报率对应的业务风险是否可以承受——阈值选择必须同时考虑两类错误率,不能只优化其中一个而忽视另一个的代价。

---

下一篇:[18-annotator-agreement-and-methodology.md](18-annotator-agreement-and-methodology.md) —— 板块IV收官,标注一致性(Cohen's kappa)与"经得起追问的具体数字"这一贯穿全系列的方法论收束。
