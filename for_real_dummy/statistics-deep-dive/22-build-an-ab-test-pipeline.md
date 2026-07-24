# 22 · 手把手实战:搭一个完整的 A/B 测试分析流水线

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 22 个"知识点",不计入"116 个知识点"的统计——和 [21 类](21-mock-interview-capstone.md)是同一挂,但风格不一样:21 号文件里,你是**旁观者**,跟着面试官和候选人的往返追问把一条推理链条看一遍;这一篇里,你是**动手的人**——从一个空文件开始,一步步敲代码,每写一段就跑一次、看到真实效果,最后独立搭出一条完整能跑的 A/B 测试分析流水线。

## 为什么是"A/B 测试流水线"

不是要发明新知识点,是把 03/04/06 三类已经讲过的假设检验框架、经典检验方法、功效分析串成一个真实工程场景里天天在用的东西:一条从"合成实验数据"到"能不能上线"的完整分析流水线。

| 阶段 | 要让程序多会一件事 | 建立在哪个已有知识点之上 |
|------|------|------|
| 阶段 1 | 用不同的真实转化率合成对照组/实验组的 0/1 观测数据,现场看到"就算真实转化率完全相同,两组观测比例也几乎不会精确相等" | [01 类](01-probability-recap-and-descriptive.md) 伯努利分布与描述统计 |
| 阶段 2 | 对两组数据做双样本比例 z 检验,判断这个差异是不是噪声,并和卡方独立性检验交叉验证数值一致 | [03 类](03-interval-estimation-and-testing-framework.md) 假设检验框架(p 值/α/β/双侧检验)、[04 类](04-classical-tests.md) 卡方独立性检验 |
| 阶段 3 | 做功效分析:反解出给定显著性水平和目标检验力所需的最小样本量,和实际用的样本量对比,并用模拟验证功效公式本身、避免"事后功效"这个坑 | [06 类](06-ab-test-design-and-power.md) 样本量公式与 MDE |
| 阶段 4 | 把前三阶段组装成一个完整的 pipeline 函数,对同一份数据产出"能不能上线"的结论,并验证四种可能的结论分支都能被真实触发 | 阶段 1-3 全部组装 |

每个阶段的代码都能独立运行(本文件用系列自带的 [_verify_md.py](_verify_md.py) 校验,校验方式是把每个 ` ```python ` 代码块单独拎出来起一个新的 Python 子进程执行——块与块之间**不共享任何变量**,后面阶段用到前面阶段的函数/数据时会重新贴一遍,不是偷懒复制,是这套校验机制要求的)。

**环境声明:** 全部代码在仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1)下真实跑通,只用 numpy/scipy,不装新依赖,随机抽样全部固定种子(`np.random.default_rng(...)`),不访问网络、不读写真实文件。

---

## 阶段 1:合成实验数据——为什么不能只拿两个百分比比大小

场景设定贯穿全文:某电商网站要把结账按钮从旧版换成新版,想知道新版是否真的提升了转化率(下单/进入结账页的用户比例)。对照组(旧按钮)和实验组(新按钮)各自的转化行为,可以用 01 类讲过的伯努利分布建模——每个用户独立地以某个概率转化,`numpy` 的 `rng.binomial(1, p, n)` 就是 n 次独立伯努利试验。

在写任何检验代码之前,先现场验证一个容易被忽视的事实:**即使两组的真实转化率完全相同,观测到的样本比例也几乎不会精确相等**——这是"为什么不能直接拿两个百分比比大小、必须做正式检验"这件事最直接的动机。

```python
import numpy as np

def simulate_conversions(rng, true_rate, n):
    """模拟n个用户的0/1转化观测: 每个用户独立以true_rate的概率转化(伯努利试验, 01类知识点)"""
    return rng.binomial(1, true_rate, n)

# 先看一个容易被忽视的现象: 即使两组真实转化率完全相同, 观测到的样本比例也几乎不会精确相等
rng_demo = np.random.default_rng(1)
p_same = 0.080
demo_a = simulate_conversions(rng_demo, p_same, 2000)
demo_b = simulate_conversions(rng_demo, p_same, 2000)

assert demo_a.shape == (2000,) and demo_b.shape == (2000,)
assert set(np.unique(demo_a)).issubset({0, 1})
assert demo_a.mean() != demo_b.mean()  # 真实跑出来几乎必然不相等: 这就是抽样噪声, 不是bug

print(f"same true rate ({p_same}), but observed: group_a={demo_a.mean():.4f}  group_b={demo_b.mean():.4f}")
print("raw percentages differ even though the true underlying rate is identical -- eyeballing two numbers is not a test")

# 正式生成本篇要分析的实验数据: 电商结账按钮改版, 对照组(旧版)vs实验组(新版)
rng = np.random.default_rng(20260722)
p_control_true, p_treatment_true = 0.080, 0.095   # 实验组真实转化率比对照组高1.5个百分点(相对提升~18.75%)
n_per_group = 4000

control = simulate_conversions(rng, p_control_true, n_per_group)
treatment = simulate_conversions(rng, p_treatment_true, n_per_group)

assert control.shape == treatment.shape == (n_per_group,)
assert 0.05 < control.mean() < 0.11      # 样本比例应该落在真实概率附近的合理范围内, 不是精确相等
assert 0.07 < treatment.mean() < 0.12

print(f"control (old button)   observed conversion rate: {control.mean():.4f}  (true {p_control_true})")
print(f"treatment (new button) observed conversion rate: {treatment.mean():.4f}  (true {p_treatment_true})")
```

真实跑出来:两组真实转化率都是 8% 时,观测到的比例是 8.40% 和 8.25%——差了 0.15 个百分点,单看这两个数字,一个不知道抽样噪声的人完全可能误以为"组间有差异"。这就是整篇教程要解决的问题:观测到的差异,多少是真信号、多少是抽样噪声,不能靠肉眼判断,需要阶段 2 的正式检验。

正式实验数据里,对照组观测转化率 7.05%(真实值 8.0%),实验组观测转化率 8.40%(真实值 9.5%)——两个观测值都不精确等于各自的真实值,这是抽样的正常表现,不是数据生成出了问题。

---

## 阶段 2:双样本比例 z 检验——这个差异是真的,还是噪声

有了两组数据,正式检验来了。检验的是 H0: p_control = p_treatment(两个按钮转化率相同,观测到的差异纯属噪声)。这是 04 类知识点 1"pooled variance t 检验"同一个思路搬到比例数据上:H0 假设两组比例相等,那么估计"共同标准误"时应该把两组数据合并(pooled)来估计这个共同比例,而不是各自独立估计。

```python
import numpy as np
from scipy import stats

def simulate_conversions(rng, true_rate, n):
    return rng.binomial(1, true_rate, n)

def two_proportion_z_test(success_a, n_a, success_b, n_b):
    """双样本比例z检验(pooled variance版本, 04类知识点1"pooled variance"同一思路搬到比例数据上):
    H0: p_a == p_b。检验统计量分母用两组合并后的比例估计标准误(假设H0成立时两组比例本应相同)。"""
    p_a, p_b = success_a / n_a, success_b / n_b
    p_pool = (success_a + success_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z_stat = (p_b - p_a) / se_pool
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))   # 双侧检验, 03类知识点6的公式
    return z_stat, p_value

rng = np.random.default_rng(20260722)
p_control_true, p_treatment_true = 0.080, 0.095
n_per_group = 4000
control = simulate_conversions(rng, p_control_true, n_per_group)
treatment = simulate_conversions(rng, p_treatment_true, n_per_group)

success_a, success_b = int(control.sum()), int(treatment.sum())
z_stat, p_value = two_proportion_z_test(success_a, n_per_group, success_b, n_per_group)

# 交叉验证: 2x2列联表卡方独立性检验(04类知识点3), 不加连续性校正(correction=False)时,
# z统计量的平方应该精确等于卡方统计量, 两者p值也应该精确相等 --- 这不是巧合,
# 两个检验本质上是同一件事的两种等价写法(把"转化与否"和"分组"当成两个分类变量做独立性检验,
# 和直接检验两个比例是否相等, 数学上是同一个假设的两种表述)
contingency = np.array([
    [success_a, n_per_group - success_a],
    [success_b, n_per_group - success_b],
])
chi2_stat, chi2_p, dof, _ = stats.chi2_contingency(contingency, correction=False)

assert dof == 1
assert abs(z_stat ** 2 - chi2_stat) < 1e-8
assert abs(p_value - chi2_p) < 1e-8

alpha = 0.05
significant = p_value < alpha
assert significant  # 现场跑出来的真实结果: 这次合成数据确实显著(阶段3会诚实讨论这个结果有多"稳")

print(f"control observed={control.mean():.4f}  treatment observed={treatment.mean():.4f}")
print(f"z={z_stat:.4f}  p_value={p_value:.6f}  significant at alpha=0.05: {significant}")
print(f"cross-check: chi2={chi2_stat:.4f} (z^2={z_stat**2:.4f})  chi2_p={chi2_p:.6f}")
```

真实结果:z ≈ 2.2613,p ≈ 0.0237,在 α=0.05 下显著;和卡方独立性检验交叉验证,z² 和卡方统计量精确相等(差异在 1e-8 量级的浮点误差以内),两个检验的 p 值也精确相等——这是一个很好的交叉验证,如果两者对不上,说明至少有一处实现错了。

**先别急着下结论。** p=0.0237 只回答了"这个差异是不是纯属噪声"这一个问题(03 类知识点 4:p 值是"如果 H0 为真,观测到这么极端的数据的概率",不是"H0 为真的概率")。这次实验设计本身有没有问题、这个显著结果有多可信,还需要阶段 3 的功效分析来回答。

---

## 阶段 3:功效分析——这次实验的样本量,到底够不够

阶段 2 判定显著,是不是就能直接说这次实验设计没问题、可以放心上线了?**不能。** 06 类知识点 1 讲过:给定 α 和目标检验力(power),可以反解出需要多少样本量——反过来,已经用掉的样本量,也应该拿去和这个"应该用多少"的数字对比,而不是看到显著结果就默认样本量肯定够用。

这里有一个容易踩的坑,值得先说清楚:**算"应该用多少样本"时,不能拿阶段 2 里已经观测到的转化率(7.05%、8.40%)反推**——那是用已经看过的实验结果去论证"我的实验设计合理",在统计学里这个操作有名字,叫事后功效(post-hoc / observed power),它本质上只是 p 值的一个单调重新参数化,不能提供任何 p 值本身没有的新信息,拿它来"验证"样本量是否够用是没有意义的循环论证。真正该用的,是实验开始前就该定好的假设:根据历史数据估计的 baseline 转化率 p0,以及业务上认为有意义的最小提升(MDE,06 类知识点 2)——这两个数字必须在看到本次实验数据**之前**就写死,不能等数据来了再挑一个让结论好看的版本(这和 03 类知识点 6"看到双侧不显著才改单侧"是同一种不诚实操作的另一个变体)。

```python
import numpy as np
from scipy import stats

def required_sample_size(p0, p1, alpha=0.05, target_power=0.80):
    """06类知识点1的样本量公式, 原样复用: n ~= 2(z_{alpha/2}+z_beta)^2 * p_bar(1-p_bar) / delta^2"""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(target_power)
    p_bar = (p0 + p1) / 2
    delta = p1 - p0
    return int(np.ceil(2 * (z_alpha + z_beta) ** 2 * p_bar * (1 - p_bar) / delta ** 2))

def simulate_achieved_power(rng, p0, p1, n_per_group, alpha, n_trials=4000):
    """给定两组真实转化率和每组样本量, 蒙特卡洛模拟很多次独立实验, 统计"真的拒绝了H0"的比例
    (这是06类知识点1/2/6反复用到的"用模拟验证理论功效公式"同一套方法论)"""
    ca = rng.binomial(1, p0, size=(n_trials, n_per_group))
    cb = rng.binomial(1, p1, size=(n_trials, n_per_group))
    sa, sb = ca.sum(axis=1), cb.sum(axis=1)
    pool = (sa + sb) / (2 * n_per_group)
    se = np.sqrt(pool * (1 - pool) * (2 / n_per_group))
    z = (sb / n_per_group - sa / n_per_group) / se
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return float(np.mean(p < alpha))

# 这两个数字必须是实验开始前就定好的假设, 不能用上面提到的"事后功效"反推
assumed_p0 = 0.080       # 历史数据估计的baseline转化率
mde = 0.015               # 业务认为有意义的最小提升(绝对值), 对应p1=0.095
alpha = 0.05
target_power = 0.80

n_required = required_sample_size(assumed_p0, assumed_p0 + mde, alpha=alpha, target_power=target_power)
n_used = 4000   # 阶段1实际生成(相当于实际收集到)的每组样本量

assert n_required == 5571
print(f"required n per group for alpha={alpha}, power={target_power}, MDE={mde}: {n_required}")
print(f"n actually used: {n_used}  ({n_used / n_required:.1%} of the recommended minimum)")

# 用蒙特卡洛验证: 用实际用的样本量n_used, 真实达到的功效有多少
rng_power = np.random.default_rng(2026)
power_at_used = simulate_achieved_power(rng_power, assumed_p0, assumed_p0 + mde, n_used, alpha)
power_at_required = simulate_achieved_power(rng_power, assumed_p0, assumed_p0 + mde, n_required, alpha)

# 核心断言1: 用n_required这个样本量做实验, 真实功效应该确实接近目标0.80(验证公式本身是对的)
assert 0.75 < power_at_required < 0.85, f"power at n_required should be near target 0.80, got {power_at_required:.4f}"

# 核心断言2: 实际用的n_used明显低于n_required, 真实功效应该明显低于目标, 这次实验设计本身偏冒险
assert power_at_used < 0.75, f"power at underpowered n_used should fall clearly short of target, got {power_at_used:.4f}"
assert power_at_used > 0.5, "but should still be meaningfully above a coin flip, given the assumed effect is real"

print(f"achieved power at n_used={n_used}: {power_at_used:.4f}  (target {target_power})")
print(f"achieved power at n_required={n_required}: {power_at_required:.4f}  (target {target_power})")
```

真实数字:给定 α=0.05、目标功效 80%、MDE=1.5 个百分点,每组需要 **5571** 个样本;这次实际只用了 4000 个,是推荐值的 71.8%。蒙特卡洛模拟证实:用 5571 个样本时真实功效 ≈79.25%,和目标 80% 吻合(验证了公式本身是对的);但用实际的 4000 个样本时,真实功效只有 ≈66.27%。

这是一个诚实但不太舒服的结论:**阶段 2 那个 p=0.0237 的显著结果,是在一个真实功效只有约 66% 的实验设计下拿到的。** 如果真实效应确实是 +1.5 个百分点、把这个实验重新独立做很多次,大约三分之一会得到"不显著"的结论——这一次恰好落在显著的三分之二里,不是必然。这不代表阶段 2 的 p 值本身"有问题"或者不能信(p<0.05 的假阳性率保证只依赖 α,和功效高低无关,这是 03 类知识点 3 讲过的),但样本量没达到设计要求这件事本身是一个真实的设计缺陷,应该被诚实地记录并向业务方说明,而不是看到一次显著结果就假装设计没问题——这正是 06 类知识点 6"样本量不足的误判"要传达的教训,只是这里的具体表现形式不是"该拒绝时没拒绝"(false negative),而是"侥幸拒绝对了,但不代表设计本身可靠"。（更进一步、这里不展开的一个真实现象:统计学文献里管"underpowered 设计里侥幸显著的结果,报告出来的效应量往往系统性偏大"叫 winner's curse / Type M error——因为只有偏大的随机波动才够得着显著性门槛,只提一句,不在本系列展开。)

---

## 阶段 4:组装成完整的 pipeline,产出"能不能上线"的结论

把前三阶段拼进一个函数:输入两组原始 0/1 观测数据、加上实验设计时就该定好的假设(baseline 转化率 + MDE),输出一份完整报告——检验统计量、p 值、效应的置信区间、样本量是否达标、以及一句"能不能上线"的结论。

结论逻辑是一个 2×2 矩阵(显著 × 样本量是否达标),不是只有"显著就上线、不显著就不上线"这一条简单规则:

| | 样本量达标 | 样本量不达标(underpowered) |
|---|---|---|
| **显著** | SHIP——放心上线 | CAUTION——这次显著,但设计本身偏冒险,建议用达标样本量confirm一次 |
| **不显著** | DO_NOT_SHIP——比较有把握地说效应(如果有)小于MDE | INCONCLUSIVE——什么都说明不了,不是"没效果"的证据,只是"这次实验没设计好" |

置信区间这里额外有一处容易被忽视的细节:检验统计量的标准误用的是 **pooled SE**(阶段 2 已经解释过,H0 假设两组比例相等,所以合并估计一个共同比例);但构造置信区间时不能假设 H0 成立(CI 描述的是"两组比例之差"这个量本身有多不确定,不管 H0 是否成立都该有意义),所以要用两组**各自独立**的比例方差相加——这是 03 类知识点 1"CI = 估计值 ± z·SE"这个构造原理的直接应用,方差相加的道理和 06 类知识点 3"两个独立样本的方差可加"完全一致,只是这次分母换成了两比例之差的方差。

```python
import numpy as np
from scipy import stats

def simulate_conversions(rng, true_rate, n):
    return rng.binomial(1, true_rate, n)

def required_sample_size(p0, p1, alpha=0.05, target_power=0.80):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(target_power)
    p_bar = (p0 + p1) / 2
    delta = p1 - p0
    return int(np.ceil(2 * (z_alpha + z_beta) ** 2 * p_bar * (1 - p_bar) / delta ** 2))

def simulate_achieved_power(rng, p0, p1, n_per_group, alpha, n_trials=2000):
    ca = rng.binomial(1, p0, size=(n_trials, n_per_group))
    cb = rng.binomial(1, p1, size=(n_trials, n_per_group))
    sa, sb = ca.sum(axis=1), cb.sum(axis=1)
    pool = (sa + sb) / (2 * n_per_group)
    se = np.sqrt(pool * (1 - pool) * (2 / n_per_group))
    z = (sb / n_per_group - sa / n_per_group) / se
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return float(np.mean(p < alpha))

def run_ab_test_pipeline(control, treatment, assumed_p0, mde, alpha=0.05, target_power=0.80, power_rng=None):
    """完整的A/B测试分析流水线: 输入两组原始0/1观测 + 实验设计时的假设, 输出检验+功效+结论"""
    n_a, n_b = len(control), len(treatment)
    succ_a, succ_b = int(control.sum()), int(treatment.sum())
    p_a, p_b = succ_a / n_a, succ_b / n_b

    # 假设检验: pooled SE(检验统计量在H0下的标准误, H0假设两组比例相等)
    p_pool = (succ_a + succ_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z_stat = (p_b - p_a) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # 置信区间: unpooled SE(不假设H0成立, 两组各自独立的方差相加, 03类知识点1+06类知识点3同一原理)
    se_unpooled = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_b - p_a
    ci_lower, ci_upper = diff - z_crit * se_unpooled, diff + z_crit * se_unpooled

    # 功效分析: 必须用实验设计时的假设(assumed_p0, mde), 不能用刚算出的观测p_a/p_b反推
    # (那是阶段3详细说明过的事后功效/post-hoc power这个坑)
    n_required = required_sample_size(assumed_p0, assumed_p0 + mde, alpha=alpha, target_power=target_power)
    underpowered = n_a < n_required
    achieved_power = None
    if power_rng is not None:
        achieved_power = simulate_achieved_power(power_rng, assumed_p0, assumed_p0 + mde, n_a, alpha)

    significant = p_value < alpha
    if significant and not underpowered:
        verdict = "SHIP"
    elif significant and underpowered:
        verdict = "CAUTION"
    elif (not significant) and underpowered:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "DO_NOT_SHIP"

    return {
        "n_a": n_a, "n_b": n_b, "p_a": p_a, "p_b": p_b,
        "z_stat": z_stat, "p_value": p_value, "significant": significant,
        "diff": diff, "ci_lower": ci_lower, "ci_upper": ci_upper,
        "n_required": n_required, "underpowered": underpowered,
        "achieved_power": achieved_power, "verdict": verdict,
    }

# ---- 端到端跑一次真实实验(阶段1生成的那份结账按钮数据) ----
rng = np.random.default_rng(20260722)
control = simulate_conversions(rng, 0.080, 4000)
treatment = simulate_conversions(rng, 0.095, 4000)

power_rng = np.random.default_rng(2026)
report = run_ab_test_pipeline(control, treatment, assumed_p0=0.080, mde=0.015, power_rng=power_rng)

assert report["significant"]
assert report["underpowered"]
assert report["verdict"] == "CAUTION"
assert report["ci_lower"] > 0   # CI不包含0, 和"显著"这个结论一致
assert report["n_required"] == 5571

print("=== main experiment ===")
print(f"observed: control={report['p_a']:.4f}  treatment={report['p_b']:.4f}  diff={report['diff']:.4f}")
print(f"z={report['z_stat']:.4f}  p={report['p_value']:.6f}  95% CI for diff=[{report['ci_lower']:.4f}, {report['ci_upper']:.4f}]")
print(f"n_used={report['n_a']}  n_required={report['n_required']}  underpowered={report['underpowered']}  achieved_power={report['achieved_power']:.4f}")
print(f"VERDICT: {report['verdict']}")

# ---- 验证结论矩阵的另外三种分支确实都能被真实触发, 不是只讲了一种巧合场景 ----
rng2 = np.random.default_rng(123)
p0, mde = 0.10, 0.05
n_req_demo = required_sample_size(p0, p0 + mde)
n_big = n_req_demo + 2000

ship_c = simulate_conversions(rng2, p0, n_big)
ship_t = simulate_conversions(rng2, p0 + mde, n_big)
ship_report = run_ab_test_pipeline(ship_c, ship_t, assumed_p0=p0, mde=mde)
assert ship_report["verdict"] == "SHIP"

flat_c = simulate_conversions(rng2, p0, n_big)
flat_t = simulate_conversions(rng2, p0, n_big)   # 真实效应为0, 但样本量达标
noship_report = run_ab_test_pipeline(flat_c, flat_t, assumed_p0=p0, mde=mde)
assert noship_report["verdict"] == "DO_NOT_SHIP"

small_c = simulate_conversions(rng2, p0, 200)
small_t = simulate_conversions(rng2, p0, 200)    # 真实效应为0, 样本量也远不够
inconclusive_report = run_ab_test_pipeline(small_c, small_t, assumed_p0=p0, mde=mde)
assert inconclusive_report["verdict"] == "INCONCLUSIVE"

print("all four verdict branches confirmed reachable: SHIP, CAUTION, INCONCLUSIVE, DO_NOT_SHIP")
```

真实跑出来的完整报告:观测差异 1.35 个百分点(diff=0.0135),95% 置信区间 [0.18%, 2.52%](不包含 0,和"显著"一致),n_used=4000 < n_required=5571,达成的功效约 67%,最终判定 **CAUTION**——不是简单的"上线"或"不上线",而是一句更诚实的话:"这次结果显著、效应量在业务上也有意义(相对提升接近 19%),但样本量没有达到设计要求,建议用足量样本再跑一次确认,而不是直接当作最终结论全量上线"。四个分支(SHIP/CAUTION/INCONCLUSIVE/DO_NOT_SHIP)也都用构造出来的场景真实触发验证过,不是只讲了这一次巧合的场景。

到这里,`run_ab_test_pipeline` 已经是一个真实能用的小工具:喂给它任意一份对照组/实验组的 0/1 观测数据,加上实验设计时的假设,它能给出统计检验、置信区间、功效诊断、以及一句不回避"证据不够"这种可能性的诚实结论——这正是 03/04/06 三类知识点在真实业务场景里拼起来之后该有的样子。

## 可以怎么继续扩展(只指方向,不在本文实现)

- **窥探安全的持续监控**:本文的 pipeline 要求预先定好样本量、只看一次结果——如果业务方坚持要"每天看一眼进度",06 类知识点 4"窥探问题"已经证明这样做真实假阳性率会远超声称的 5%,要安全地做到"随时可以看",需要 [07 类](07-modern-experimentation.md)的序贯检验(always-valid p 值)机制,这里不实现。
- **CUPED 方差削减**:阶段 3 算出的 5571 这个样本量,是在"没有任何额外信息"的前提下算出来的;如果能拿到用户实验前的历史行为数据(比如过去 30 天的购买频率)当协变量,04 类知识点 2 提到的 CUPED 技术能显著压低方差,同样的 MDE 和目标功效,所需样本量可能明显低于 5571——具体机制见 [07 类](07-modern-experimentation.md)。
- **多指标/多变体场景下的多重比较校正**:本文只跟踪一个指标(转化率)。真实业务里同时监控多个指标(点击率、客单价、留存率……)是常态,05 类知识点 1 和 06 类知识点 5 已经证明不做校正会让"至少一个指标偶然显著"的概率远超预期——把 pipeline 扩展成支持多指标输入、自动做 FDR/Bonferroni 校正,是一个自然的下一步,这里不实现。
- **贝叶斯 A/B 测试框架**:本文全程是频率派框架(p 值、置信区间)。[13 类](13-bayesian-applications.md)知识点 1 讲过贝叶斯 A/B 测试能直接回答"给定观测数据,新版本转化率更高的概率是多少"这种业务方更直觉的问题,不需要功效分析这种预先设计步骤,但需要选择先验、后验没有解析解时可能要用 [12 类](12-mcmc-foundations.md)的 MCMC 采样——是完全不同的一套方法论,这里只指方向。

这四个方向都不实现,是为了让这篇教程聚焦在"03/04/06 三类已学知识点怎么拼成一条真实流水线"这一件事上——真要继续做下去,每个方向单独展开都够写一整篇。

## 这篇教程展示的方法论

这个"教程体"格式最早在 [dsa-deep-dive/21](../dsa-deep-dive/21-build-a-mini-search-engine.md) 试点验证,核心模式是:挑几个关联的已学知识点 → 设计一个真实有用、读者一看就懂价值的小工具 → 分阶段增量实现,每一步都跑起来看到真实效果,而不是一次性甩出完整代码 → 诚实呈现"不完美的真实输出"(本文阶段 3 的"显著但 underpowered"就是这样一处真实发现,不是编出来的教学案例)。这次应要求推广到 statistics-deep-dive 系列,落地成这一篇——选择 A/B 测试流水线作为主题,是因为 03(假设检验框架)、04(经典检验方法)、06(功效分析)三类知识点串起来,天然对应业务里最常见的一条真实分析链路,比空谈"这几个知识点有关联"更有说服力。

---

*创建:2026-07-24*
