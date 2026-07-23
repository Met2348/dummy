# 19 · 时间序列基础深挖(Time Series Foundations)

> 总览见 [00-roadmap.md](00-roadmap.md)

板块V开篇。01-18类建立的几乎所有推断工具,都隐含假设观测值之间相互独立(iid)。但真实世界大量重要的数据——日活用户数、股票价格、模型训练loss曲线——都是按时间顺序排列、前后相关的。本文建立理解时间序列数据的最小必要工具集。**本文定位是搭桥,不是随机过程课程**:平稳性、自相关这些概念只用初等的代数推导和数值模拟建立直觉,不引用鞅或马尔可夫链的形式化理论,不假设"随机过程"这门课的先修知识——需要的每一个结论,都能从"独立随机变量的方差可加"这类01类已经建立的初等事实直接推出。

**环境声明:** 全部代码在仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1)下真实跑通,随机抽样固定种子(`np.random.default_rng(42)`)。ACF/PACF全部手写实现。

---

## 1. 时间序列与iid数据的区别 —— 自相关函数是最直接的判据

**定义与记号:** 01类以来建立的推断工具几乎都隐含假设观测值相互独立(iid)。时间序列(time series):按时间顺序排列的观测值X₁,X₂,...,X_T,最本质的特点是**相邻观测值之间通常存在相关性**,不满足iid的"独立"这一条。自相关函数(ACF):衡量序列和它自身"滞后k期"版本之间的相关程度,ρ(k)=Cov(X_t,X_{t+k})/Var(X_t)。

**一句话:** iid数据的ACF在任何非零滞后处都应该接近0(独立意味着任何两个不同时刻的观测值都不相关);有真实时间依赖结构的数据(比如下面的AR(1)序列)的ACF在滞后较小时通常明显不为0——这个"非零自相关"是区分"这是一批独立同分布的普通数据"还是"这是一个真正的时间序列"最直接的数值判据。

**数学推导:** 构造递推序列(不需要任何随机过程的形式化定义,纯代数递推关系):X_t=φ·X_{t-1}+ε_t(ε_t是均值0、方差σ²、彼此独立的噪声,\|φ\|<1)。用初等的方差/协方差运算直接推导:Cov(X_t,X_{t+1})=Cov(X_t,φX_t+ε_{t+1})=φVar(X_t)(利用ε_{t+1}和X_t不相关这个独立性假设);依此类推,归纳可得ρ(k)=φ^\|k\|——一个纯代数结果,不需要测度论或者随机过程理论。

**底层机制/为什么这样设计:** 为什么"iid假设是否成立"对01-18类建立的几乎所有推断工具都至关重要?因为几乎所有标准误公式(比如样本均值的标准误σ/√n)推导时都用到了"各观测值方差可加"这个性质,而这个性质恰恰依赖独立性——如果观测值之间存在正相关,真实的方差会比"假装独立"算出的方差更大(相关的观测值"抱团",不能像独立观测值那样简单地互相抵消随机波动),直接套用01-18类的标准工具会系统性地低估不确定性、错误地过早宣称"显著"。

**AI研究/工程场景:** 监控一个业务指标(日活跃用户数)随时间的变化,如果把连续30天的数据当作30个独立观测值直接套用t检验比较"这个月和上个月是否有显著差异",完全忽视了"今天的用户数和昨天的用户数天然相关"这个事实,算出来的p值会严重失真——这正是17类分布漂移监控、以及本板块方法论和01-18类默认假设之间最根本的分界线。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

def acf(x, max_lag):
    x = x - x.mean()
    n = len(x)
    c0 = np.sum(x ** 2) / n
    return np.array([np.sum(x[:n - k] * x[k:]) / n / c0 for k in range(max_lag + 1)])

n = 2000
iid_data = rng.normal(0, 1, n)

phi = 0.7
ar1_data = np.zeros(n)
noise = rng.normal(0, 1, n)
for t in range(1, n):
    ar1_data[t] = phi * ar1_data[t - 1] + noise[t]

acf_iid = acf(iid_data, 10)
acf_ar1 = acf(ar1_data, 10)

# 核心断言1: iid数据的ACF在所有非零滞后处都应该接近0
assert np.max(np.abs(acf_iid[1:])) < 0.1, f"iid data's ACF should stay near 0 at all lags, got max={np.max(np.abs(acf_iid[1:])):.4f}"

# 核心断言2: AR(1)数据的ACF在滞后1处应该接近真实phi, 明显不为0
assert abs(acf_ar1[1] - phi) < 0.05, f"AR(1) lag-1 ACF should be close to phi={phi}, got {acf_ar1[1]:.4f}"
assert acf_ar1[1] > 0.3, "AR(1) ACF should be clearly nonzero at lag 1, unlike iid data"

print(f"iid data ACF (lags 1-5): {np.round(acf_iid[1:6], 4)}")
print(f"AR(1) data ACF (lags 1-5): {np.round(acf_ar1[1:6], 4)}  (theory phi^k: {np.round(phi**np.arange(1,6), 4)})")
```

**面试怎么问+追问链**(诊断真实数据新题型):
- Q:"给你一批连续30天的日活数据,直接用t检验比较'本月和上月是否有显著差异',这样做对吗?"
- 追问1:"问题出在哪?"(t检验假设各观测值独立,但日活数据几乎必然存在时间相关性,违反独立性假设会导致标准误被低估,p值虚假地显得更显著,容易把"纯粹的时间相关性波动"误判成"真实的月度差异")
- 深挖追问:"第一步应该怎么检验这批数据是否存在这种时间相关性?"(直接计算ACF,看滞后1、2、3期的自相关系数是否明显偏离0——如果确实存在显著自相关,说明不能直接套用假设独立性的标准检验方法)

**常见坑:**
- 拿到任何按时间顺序排列的数据,不做任何自相关检验就直接套用01-18类假设独立性的标准推断工具,可能严重低估真实的不确定性。
- 反过来看到"数据是按时间收集的"就不假思索地认为"一定需要专门的时间序列方法"——如果数据确实不存在有意义的时间相关性(ACF在各滞后都接近0),即使是按时间顺序收集的,也可以近似当作iid处理。

---

## 2. 平稳性(初等定义)—— 统计规律本身是否随时间变化

**定义与记号:** 平稳性(初等定义,只覆盖数值可直接验证的两条):一个时间序列的均值不随时间变化、方差不随时间变化,可以用一套固定不变的统计规律去描述;不满足这两条的序列称为非平稳。

**一句话:** 平稳性回答的是一个很朴素的问题——"这个序列的统计特性是不是'一直差不多',还是随着时间推移会系统性地往某个方向漂移或者越变越剧烈";平稳序列可以用一套固定规律描述,非平稳序列的"规律"本身就在变。

**数学推导/说明:** 数值检验思路(滚动统计量的数值比较,不需要形式化的假设检验):把序列切分成若干连续窗口,分别计算每个窗口的样本均值,①平稳序列(比如AR(1))各窗口均值应该在一个固定值附近**无系统性地**波动;②非平稳序列(比如带线性趋势的X_t=a+bt+ε_t)各窗口均值会表现出明显的、随时间**系统性**变化的模式。用"窗口序号和窗口均值之间的相关系数"量化这个区别——但要小心单次模拟的抽样噪声(AR(1)序列即使真实均值恒定为0,单条路径的窗口均值也可能因为随机波动偶然呈现出"看起来像趋势"的模式),稳妥的做法是重复多次独立模拟,看这个相关系数是否**系统性地**偏向某个方向,而不是只看一次模拟的结果。

**底层机制/为什么这样设计:** 为什么平稳性对后续时间序列分析方法(ACF/PACF、20类的预测方法)如此重要?因为几乎所有经典时间序列方法的理论基础都建立在"用序列的历史数据估计一套统计规律,再用这套规律预测未来"这个逻辑上——如果序列本身非平稳,"用过去的规律预测未来"这个操作的合理性就会被削弱,这也是为什么处理非平稳序列时几乎总是第一步先做某种变换(知识点6的差分)把它转化成平稳序列。

**AI研究/工程场景:** 判断"这个指标过去的历史规律是否还能用来指导未来预测",平稳性检验是第一步——如果发现指标本身的均值或波动幅度在系统性地变化,直接用历史数据拟合出的预测模型可能已经过时,需要先处理这个非平稳性,而不是不假思索地把全部历史数据丢进一个假设平稳的预测模型里。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

n = 2000
n_windows = 10
window_size = n // n_windows

def window_means(series):
    return np.array([series[w * window_size:(w + 1) * window_size].mean() for w in range(n_windows)])

# 重复多次独立模拟(避免单次抽样噪声导致的误判), 分别统计AR(1)和趋势序列的"窗口序号-窗口均值"相关系数
n_trials = 300
window_idx = np.arange(n_windows)
corrs_ar1, corrs_trend = [], []
for _ in range(n_trials):
    phi = 0.6
    ar1 = np.zeros(n)
    noise = rng.normal(0, 1, n)
    for t in range(1, n):
        ar1[t] = phi * ar1[t - 1] + noise[t]
    corrs_ar1.append(np.corrcoef(window_idx, window_means(ar1))[0, 1])

    trend = 0.05 * np.arange(n) + rng.normal(0, 1, n)
    corrs_trend.append(np.corrcoef(window_idx, window_means(trend))[0, 1])

corrs_ar1, corrs_trend = np.array(corrs_ar1), np.array(corrs_trend)

# 核心断言1: 平稳AR(1)序列, 平均而言窗口均值和窗口序号没有系统性关联(单次模拟可能有噪声, 但300次平均应该接近0)
assert abs(corrs_ar1.mean()) < 0.1, f"stationary AR(1) should show no systematic mean drift on average, got {corrs_ar1.mean():.4f}"

# 核心断言2: 非平稳趋势序列, 每一次模拟都应该显示出接近完美的正相关(确定性趋势主导, 极其稳健)
assert corrs_trend.min() > 0.99, f"the trending series should show near-perfect correlation in EVERY trial, got min={corrs_trend.min():.4f}"

print(f"AR(1) (stationary): mean correlation over {n_trials} trials = {corrs_ar1.mean():.4f}  (near 0, no systematic drift)")
print(f"trend (non-stationary): min correlation over {n_trials} trials = {corrs_trend.min():.4f}  (always ~1, robust systematic drift)")
```

**面试怎么问+追问链**(诊断真实数据新题型):
- Q:"给你一个业务指标过去两年的月度数据,怎么判断能不能直接用这两年的历史规律去预测下个月?"
- 追问1:"第一步应该检查什么?"(检查这个指标在过去两年里是否平稳——把两年数据切成几个季度窗口,看每个窗口的均值和波动幅度是否稳定,如果发现指标均值在系统性地上升或下降,说明历史规律本身在变化,不能简单假设"过去两年的平均水平"能代表未来预期)
- 深挖追问:"如果确实发现指标不平稳,这个'不平稳'本身对业务分析有什么价值,不只是一个需要被'处理掉'的技术障碍?"(不平稳本身往往就是业务里最重要的信号——持续增长趋势可能反映了产品的自然增长动能,处理非平稳性是为了后续技术分析的需要,不代表这部分信息在业务上不重要)

**常见坑:**
- 直接对一个明显存在趋势/波动幅度变化的非平稳序列,套用假设平稳性的分析方法(知识点3的ACF/PACF理论形状、20类的简单预测方法),得出的结论可能严重失真。
- 只看单次模拟/单条真实序列的窗口统计量就下结论——本知识点的可运行例子已经展示,平稳序列单次实现的窗口均值也可能因为抽样噪声呈现出"看起来像趋势"的假象,需要意识到这种噪声的存在,不能仅凭一次观察就断定平稳性。

---

## 3. ACF与PACF —— "整体相关"与"排除中间影响后的直接相关"

**定义与记号:** ACF(自相关函数,知识点1已定义)。PACF(偏自相关函数):衡量X_t和X_{t+k}之间"排除掉中间滞后期的影响之后"剩余的相关程度——类比05类"偏相关"(控制其他变量后的相关性)概念,PACF是把"控制中间变量"这个操作应用到时间序列自相关分析上。

**一句话:** ACF回答"X_t和X_{t+k}整体上相关多少"(可能包含通过中间时刻传递的间接相关性);PACF回答"排除掉中间时刻的传递效应之后,X_t和X_{t+k}还剩多少直接相关性"——AR(1)序列的这两个函数呈现出互补的形状:ACF按φ^k的速率缓慢衰减("拖尾"),PACF在滞后1之后迅速趋于0("截尾")。

**数学推导/说明:** 手写PACF的计算思路(和05类偏相关的计算完全同构,只是控制的变量换成了同一个序列自身的中间滞后项):PACF(k)是"用X_{t-1},...,X_{t-k+1}回归X_t"和"用同样这些变量回归X_{t-k}"这两个回归各自的**残差**之间的相关系数。对AR(1)序列(X_t=φX_{t-1}+ε_t),理论上PACF(1)=φ,PACF(k)=0(k≥2)——因为X_t只直接依赖X_{t-1},排除X_{t-1}的影响之后,X_t和更早的X_{t-2},X_{t-3}...之间不应该再剩下任何直接关联。

**底层机制/为什么这样设计:** 为什么ACF和PACF对AR过程呈现出这种互补的"拖尾vs截尾"形状?对AR(1)过程,X_t直接依赖前1期,超过1期之后没有直接依赖关系,所以排除中间期影响后的PACF在滞后1之后应该精确为0;但这种依赖关系会通过链式传导影响更远的时刻(X_t依赖X_{t-1},X_{t-1}又依赖X_{t-2},层层传递),没有排除中间期影响的原始ACF会在更长的滞后范围内保持非零(只是幅度按几何级数衰减)——"直接依赖vs间接传导"的区分,正是ACF和PACF分别刻画的两个不同层面。

**AI研究/工程场景:** 传统时间序列建模里,画出ACF和PACF图、根据"哪个截尾、哪个拖尾"判断该用AR模型还是MA模型、以及大致阶数,是自动化建模工具普及之前长期使用的经典人工诊断流程,理解这两个函数各自刻画的信息,是理解任何时间序列文献的必备语言。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

def acf(x, max_lag):
    x = x - x.mean()
    n = len(x)
    c0 = np.sum(x ** 2) / n
    return np.array([np.sum(x[:n - k] * x[k:]) / n / c0 for k in range(max_lag + 1)])

def pacf_via_regression(x, max_lag):
    n = len(x)
    a = acf(x, max_lag)
    pacf_vals = [1.0, a[1]]  # PACF(0)=1, PACF(1)=ACF(1)
    for k in range(2, max_lag + 1):
        Y_t, Y_tk = x[k:], x[:n - k]
        X_mid = np.column_stack([x[k - j:n - j] for j in range(1, k)])  # 中间k-1个滞后项
        X_design = np.column_stack([np.ones(len(Y_t)), X_mid])
        beta_t = np.linalg.lstsq(X_design, Y_t, rcond=None)[0]
        beta_tk = np.linalg.lstsq(X_design, Y_tk, rcond=None)[0]
        resid_t, resid_tk = Y_t - X_design @ beta_t, Y_tk - X_design @ beta_tk
        pacf_vals.append(np.corrcoef(resid_t, resid_tk)[0, 1])
    return np.array(pacf_vals)

n = 3000
phi = 0.6
ar1 = np.zeros(n)
noise = rng.normal(0, 1, n)
for t in range(1, n):
    ar1[t] = phi * ar1[t - 1] + noise[t]

acf_vals = acf(ar1, 8)
pacf_vals = pacf_via_regression(ar1, 8)

# 核心断言1: PACF(1)应该接近真实phi
assert abs(pacf_vals[1] - phi) < 0.05, f"PACF(1) should be close to phi={phi}, got {pacf_vals[1]:.4f}"

# 核心断言2: ACF"拖尾" -- 滞后2处依然明显非零(和理论phi^2=0.36吻合)
assert acf_vals[2] > 0.2, f"ACF should still be substantial at lag 2 ('tails off'), got {acf_vals[2]:.4f}"

# 核心断言3: PACF"截尾" -- 滞后2及以后迅速趋于0, 和ACF形成鲜明对比
assert np.max(np.abs(pacf_vals[2:])) < 0.05, f"PACF should cut off sharply after lag 1, got max={np.max(np.abs(pacf_vals[2:])):.4f}"

print(f"ACF (lags 0-4):  {np.round(acf_vals[:5], 4)}  (theory phi^k: {np.round(phi**np.arange(5), 4)}, decays gradually)")
print(f"PACF (lags 0-4): {np.round(pacf_vals[:5], 4)}  (cuts off sharply after lag 1)")
```

把上面这次真实运行的lags 0-8全部数值画成柱状图(柱长∝|数值|,同一次运行,数字和上面的print完全对应),"拖尾"和"截尾"的对比一眼就能看出来:

```
lag 0: ACF=+1.0000 |##############################  PACF=+1.0000 |##############################
lag 1: ACF=+0.6193 |###################             PACF=+0.6193 |###################
lag 2: ACF=+0.3679 |###########                     PACF=-0.0253 |#
lag 3: ACF=+0.2246 |#######                         PACF=+0.0111 |
lag 4: ACF=+0.1386 |####                            PACF=+0.0014 |
lag 5: ACF=+0.0702 |##                              PACF=-0.0247 |#
lag 6: ACF=+0.0169 |#                               PACF=-0.0265 |#
lag 7: ACF=+0.0142 |                                PACF=+0.0331 |#
lag 8: ACF=+0.0006 |                                PACF=-0.0195 |#
```

左边ACF的柱子从lag 1开始逐格缩短、缓慢消失到lag 8还没完全归零("拖尾");右边PACF的柱子在lag 1之后立刻掉到噪声水平、之后几根短柱纯粹是有限样本的抽样波动("截尾")——这正是AR(1)只直接依赖前1步、但通过链式传导间接影响所有更远滞后期,这两句话在数值上的直接体现。

**面试怎么问+追问链**(决策依据追问轴):
- Q:"拿到一个新的时间序列,怎么判断它更适合用AR模型还是MA模型来描述?"
- 追问1:"ACF和PACF各自能提供什么诊断信息?"(如果ACF拖尾、PACF在某个滞后p之后精确截尾,提示这是AR(p)过程;如果反过来ACF截尾、PACF拖尾,提示这是MA(q)过程,两者呈现互补形状,是从观测数据反推生成机制的经典诊断依据)
- 深挖追问:"如果ACF和PACF都呈现出拖尾,说明了什么?"(可能是同时包含AR和MA成分的更复杂过程(ARMA),纯AR或纯MA模型都无法完整刻画;或者序列本身可能还不平稳,需要先做差分处理再重新计算ACF/PACF,知识点6会展开这个处理流程)

**常见坑:**
- 只看ACF就直接判断序列的生成机制,不结合PACF的互补信息。
- 在序列本身还不平稳的情况下直接计算和解读ACF/PACF——非平稳序列的样本ACF形状会因为非平稳性本身而失真,需要先处理平稳性问题。

---

## 4. 白噪声 —— 时间结构被"提取干净"之后应有的样子

**定义与记号:** 白噪声:最简单的平稳时间序列——各观测值相互独立(或至少不相关)、均值恒定、方差恒定,ACF在任何非零滞后处理论上都精确为0。白噪声是知识点1"iid数据"这个概念在时间序列语境下的具体化身,也是几乎所有时间序列模型对"残差"部分的理想化假设——一个好的模型应该把序列里所有可预测的时间相关性都提取出来,剩下的残差应该表现得像白噪声。

**一句话:** 如果一个时间序列模型拟合得好,它的残差应该"看起来毫无规律"(白噪声);如果残差本身还表现出明显的自相关,说明模型还没有把序列里所有可预测的结构提取干净,这是判断模型"是否还有改进空间"的标准诊断依据。

**数学推导/说明:** 白噪声在有限样本下,即使真实ACF理论上为0,样本ACF不会精确等于0(和任何统计量一样存在抽样波动),但这个波动被限制在一个可计算的置信带内——大样本近似下,白噪声样本自相关系数近似服从均值0、标准差1/√n的正态分布,±1.96/√n给出近似95%置信带,落在带内属于正常抽样波动,持续落在带外才提示真实存在非零自相关。

**底层机制/为什么这样设计:** 为什么需要这个置信带,而不是要求样本ACF"精确等于0"?因为样本ACF本身是有限数据估计出来的统计量,和03类"任何统计量都有抽样分布"是同一个原则——±1.96/√n这个置信带是标准误公式配合正态近似给出的合理容忍范围,是"用统计意义上的容差,而不是精确相等"这条本系列贯穿全程的纪律在时间序列诊断场景下的又一次应用。

**AI研究/工程场景:** 拟合完一个时间序列预测模型之后,检查残差ACF是否落在置信带内,是判断模型是否"合格"的标准诊断步骤——如果残差ACF在某些滞后处显著超出置信带,说明还有可以被进一步利用的时间结构没有被模型捕捉到。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

def acf(x, max_lag):
    x = x - x.mean()
    n = len(x)
    c0 = np.sum(x ** 2) / n
    return np.array([np.sum(x[:n - k] * x[k:]) / n / c0 for k in range(max_lag + 1)])

n = 2000
white_noise = rng.normal(0, 1, n)
max_lag = 20
acf_vals = acf(white_noise, max_lag)
band = 1.96 / np.sqrt(n)

n_within = np.sum(np.abs(acf_vals[1:]) <= band)
n_outside = max_lag - n_within

# 核心断言1: 绝大多数滞后期应该落在置信带内
assert n_within >= 17, f"most lags should fall within the confidence band, got {n_within}/{max_lag}"

# 核心断言2(呼应05类多重比较原则): 20个滞后期各按5%名义水平检验, 偶尔有1-2个"意外"超出带外是正常现象, 不代表模型有问题
assert n_outside <= 3, f"a small number of chance exceedances is expected (multiple comparisons), got {n_outside}"

print(f"white noise ACF, {max_lag} lags tested: {n_within} within the 95% band, {n_outside} outside")
print(f"band = +/-{band:.4f}")
print("=> even genuine white noise will occasionally exceed the band by chance -- this is the multiple-comparisons phenomenon from 05-class, not evidence of real structure")
```

**面试怎么问+追问链**(诊断真实数据新题型):
- Q:"拟合了一个时间序列模型,残差在滞后1、5、12处的样本ACF都超出了置信带,滞后其他期都在带内,这说明什么?"
- 追问1:"这个模式有没有特别的含义?"(比随机噪声更值得关注的是"是否存在某种规律"——比如滞后12如果数据是月度的,可能提示存在年度季节性没有被模型捕捉到;滞后1超出置信带提示模型阶数可能不够;需要结合具体滞后期的业务含义来解读,不是所有"超出置信带"的滞后期都同等重要)
- 深挖追问:"如果残差ACF在20个滞后期里恰好有1个超出置信带,是不是说明模型有问题?"(不一定——这正是05类多重比较原则的直接应用,即使真实残差是完美的白噪声,期望也会有大约1个"偶然"超出置信带,不能仅凭一个孤立的超出点就断定模型有问题)

**常见坑:**
- 看到残差ACF有任何一个滞后期超出置信带就断定模型"不合格"需要重新调整,不考虑多重比较原则(多个滞后期同时检验,总会有小概率偶然超出)。
- 忽视置信带宽度本身依赖样本量(±1.96/√n),样本量很小时置信带很宽,即使残差存在真实的自相关也可能因为样本量不足而检测不出来。

---

## 5. 随机游走(初等递推定义)—— 方差随时间线性增长的非平稳典型

**定义与记号:** 随机游走(用最初等的递推方式定义,不涉及任何随机过程的形式化理论):X_t=X_{t-1}+ε_t,X_0是给定的初始值,ε_t是均值0、方差σ²、彼此独立的噪声——每一步都在前一步的基础上加一个随机增量。这是知识点2"非平稳序列"的一个具体、经典的例子。

**一句话:** 随机游走可以直观理解成"喝醉的人走路"——每一步往哪个方向走多远是随机的,但下一步永远是从"当前所在的位置"出发,不会"忘记"之前已经走到哪里(这和白噪声"每一步都完全独立于之前的位置"形成鲜明对比),这种"随机增量不断累积"的结构,正是随机游走非平稳性的直观来源。

**数学推导:** 通过递推展开X_t=X_0+Σ_{i=1}^{t}ε_i(每一步的随机增量简单累加,纯代数展开)。由独立噪声项方差可加(01类已建立的基本性质):Var(X_t)=Var(X_0)+t·σ²——**方差随时间t线性增长**,不是常数,直接违反了知识点2平稳性定义里"方差不随时间变化"这一条,是随机游走"非平稳"最直接、最具体的数值体现,增长速率恰好是σ²,是可以直接算出来的精确关系。

**底层机制/为什么这样设计:** 为什么"每一步的随机增量简单累加"会导致方差随时间无限增长?因为独立随机变量的方差可加(不会因为"更早的波动已经过去了"就被抵消或者遗忘),累积到时刻t的位置X_t本质上是t个独立随机增量的和,根据方差可加性,这个和的方差自然是各项方差之和;这个"没有回归到某个固定水平的力量"(没有类似AR(1)里\|φ\|<1那种"往回拉"的机制)是随机游走和知识点1-3讨论的平稳AR过程最本质的区别——随机游走可以理解成AR(1)在φ=1这个边界情形下的极限,完全没有回归力量,任何一次随机冲击的影响会永久保留、不会衰减。

**AI研究/工程场景:** 很多金融时间序列(股票价格、汇率)在经验上被发现更接近随机游走而不是均值回归的平稳过程,这个经验观察直接导致了金融时间序列分析里"不能直接对价格本身做平稳时间序列分析,几乎总是先取对数收益率(近似等价于一阶差分,知识点6会展开)"这个标准处理流程的由来。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

n_paths = 3000
sigma = 1.0
check_times = [50, 100, 200, 300, 400, 500]

variances_at_t = []
for t in check_times:
    # n_paths条独立的随机游走路径, 各自走t步(模拟"如果时间重新来过, 会是不同的具体路径, 但统计规律相同")
    increments = rng.normal(0, sigma, (n_paths, t))
    positions_at_t = increments.sum(axis=1)
    variances_at_t.append(positions_at_t.var())
variances_at_t = np.array(variances_at_t)

# 核心断言1: 方差应该随时间单调增大
assert all(variances_at_t[i] < variances_at_t[i + 1] for i in range(len(variances_at_t) - 1)), \
    f"variance should grow monotonically with t: {variances_at_t}"

# 核心断言2: 线性回归拟合variance vs t, 斜率应该接近理论值sigma^2=1, 且拟合应该接近完美的线性关系
slope, intercept = np.polyfit(check_times, variances_at_t, 1)
ss_res = np.sum((variances_at_t - (slope * np.array(check_times) + intercept)) ** 2)
ss_tot = np.sum((variances_at_t - variances_at_t.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

assert abs(slope - sigma ** 2) < 0.15, f"slope should be close to the theoretical sigma^2={sigma**2}, got {slope:.4f}"
assert r2 > 0.99, f"variance vs t should be almost perfectly linear, got R^2={r2:.4f}"

print(f"variance at t={check_times}: {np.round(variances_at_t, 2)}")
print(f"linear fit: slope={slope:.4f} (theory=sigma^2={sigma**2})  R^2={r2:.6f}")
print("=> a random walk's spread keeps growing without bound -- there is no force pulling it back to a fixed level")
```

**面试怎么问+追问链**(规模递增轴):
- Q:"一个业务指标看起来'不断随机漂移,没有固定的稳定水平',这种模式在统计上对应什么结构,有什么实际含义?"
- 追问1:"如果确实是随机游走结构,对'预测未来'这件事有什么具体含义?"(随机游走结构下,"明天最好的预测就是今天的值"——传统的"回归到历史均值"这类预测直觉在这里不适用,而且随着预测的时间跨度越远,预测的不确定性会线性增长,不是保持恒定)
- 深挖追问:"怎么用数据检验一个序列到底是随机游走还是有真实的均值回归倾向?"(可以估计AR(1)模型的φ系数,看它是否显著小于1(单位根检验是这个问题的标准形式化工具,严格的假设检验框架超出本文"初等定义"的范围);更直观的初步判断可以看方差是否确实随时间线性增长,本知识点的可运行例子演示的方法)

**常见坑:**
- 看到一个序列"不断变化、没有固定水平"就直接假设它是随机游走,不做任何数值验证——很多非平稳序列有其他结构(比如带确定性趋势),不是随机游走这一种非平稳形式。
- 忽视随机游走"预测不确定性随时间线性增长"这个特性,对随机游走结构的指标做长期预测时依然给出一个和短期预测同样窄的置信区间,严重低估长期预测的真实不确定性。

---

## 6. 差分使非平稳序列变平稳 —— 把累积过程逆过来

**定义与记号:** 一阶差分:ΔX_t=X_t-X_{t-1},把原始序列转换成"相邻观测值之差"构成的新序列。对随机游走X_t=X_{t-1}+ε_t,差分后ΔX_t=ε_t——差分后的序列恰好就是原始的噪声项本身,是一个白噪声序列(知识点4),天然满足平稳性。

**一句话:** 差分是处理"随机游走型非平稳性"最直接、最有效的手段——随机游走的非平稳性本质上来自"随机增量不断累积",差分这个操作恰好是在"把累积过程逆过来,把每一步的增量重新提取出来",提取出来的增量序列(如果原始序列真的是随机游走)天然就是平稳的。

**数学推导:** 对随机游走X_t=X_{t-1}+ε_t两边同时减去X_{t-1}:ΔX_t=X_t-X_{t-1}=ε_t——纯粹的代数移项,直接证明"随机游走的一阶差分精确等于原始噪声项",不是近似或启发式说法。ε_t按定义就是独立同分布的白噪声,差分后的序列自然继承白噪声的所有平稳性质,不需要额外证明,是差分定义和随机游走定义直接代数组合的必然结果。

**底层机制/为什么这样设计:** 为什么差分对"随机游走型"非平稳性特别有效,但不是对所有类型的非平稳性都同样有效?因为差分这个操作专门针对的是"单位根"(知识点5追问链提到的φ=1这个特殊情形)这一类特定的非平稳结构——如果非平稳性来自别的原因(比如方差本身随时间变化,而不是均值/水平的累积漂移),简单的一阶差分不一定能解决问题,可能需要别的变换(比如取对数来稳定方差);理解差分"具体解决的是哪一种非平稳性",而不是把它当作万能的"平稳化"操作,是正确使用这个工具的关键。

**AI研究/工程场景:** ARIMA模型(AutoRegressive Integrated Moving Average)里的"I"(Integrated)这个字母,指的就是"差分"操作的逆运算关系——ARIMA建模的标准流程是先对原始序列做适当阶数的差分直到平稳,再对差分后的平稳序列拟合ARMA模型(结合知识点3的ACF/PACF定阶),ARIMA(p,d,q)里的d就是差分的阶数,是经典时间序列建模流程里几乎必经的第一步预处理。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

def acf(x, max_lag):
    x = x - x.mean()
    n = len(x)
    c0 = np.sum(x ** 2) / n
    return np.array([np.sum(x[:n - k] * x[k:]) / n / c0 for k in range(max_lag + 1)])

n = 2000
sigma = 1.0
noise = rng.normal(0, sigma, n)
random_walk = np.cumsum(noise)  # X_t = X_{t-1} + eps_t, X_0=0

diff_series = np.diff(random_walk)

# 核心断言1: 差分后序列应该精确等于原始噪声项(纯代数恒等式)
assert np.allclose(diff_series, noise[1:]), "the first difference of a random walk should exactly equal the original noise"

# 原始序列: 累积方差应该持续增长(复用知识点5已验证的现象)
n_windows = 10
window_size = n // n_windows
def cumulative_variance(series):
    return np.array([series[:w * window_size].var() for w in range(1, n_windows + 1)])

cum_var_original = cumulative_variance(random_walk)
cum_var_diff = cumulative_variance(diff_series)

# 核心断言2: 原始随机游走的累积方差应该明显增长
assert cum_var_original[-1] > cum_var_original[0] * 5, \
    f"the original random walk's variance should grow substantially: {cum_var_original[0]:.2f} -> {cum_var_original[-1]:.2f}"

# 核心断言3: 差分后序列的累积方差应该稳定在sigma^2=1附近, 不再随时间增长
assert np.all(np.abs(cum_var_diff[3:] - sigma ** 2) < 0.15), \
    f"the differenced series' variance should stabilize near sigma^2={sigma**2}, got {cum_var_diff[3:]}"

# 核心断言4: 差分后序列的ACF应该接近白噪声(非零滞后处接近0)
acf_diff = acf(diff_series, 10)
band = 1.96 / np.sqrt(len(diff_series))
assert np.sum(np.abs(acf_diff[1:]) <= band) >= 8, "the differenced series' ACF should mostly fall within the white-noise confidence band"

print(f"original random walk: cumulative variance grows from {cum_var_original[0]:.2f} to {cum_var_original[-1]:.2f}")
print(f"differenced series:    cumulative variance stays near sigma^2={sigma**2}: {np.round(cum_var_diff, 3)}")
print(f"differenced series ACF (lags 1-5): {np.round(acf_diff[1:6], 4)}  (band=+/-{band:.4f}, looks like white noise)")
```

**面试怎么问+追问链**(方案批判迭代轴,收束时间序列基础板块):
- Q:"一个业务指标的原始序列不平稳,差分之后是不是就可以直接套用之前板块建立的、假设独立同分布的统计工具了?"
- 追问1:"差分之后的序列,一定会变成完全独立的白噪声吗?"(不一定——只有当原始序列恰好是纯粹的随机游走时,差分后才精确等于白噪声;如果原始序列是其他更复杂的非平稳结构,差分后的序列可能依然存在一定的自相关(仍然需要用ACF/PACF重新诊断),差分只是"处理了单位根这一种非平稳性来源",不保证处理完之后序列就一定完全独立)
- 深挖追问:"如果一阶差分之后序列依然不平稳,应该怎么办?"(可以尝试二阶差分,但过度差分(差分次数超过实际需要)会引入新的、人为的自相关结构,不是"差分越多越安全",需要每次差分后都重新检验平稳性,差分到刚好平稳为止就应该停止)

**常见坑:**
- 把差分当作"万能的平稳化操作",对任何类型的非平稳序列都不加区分地做差分,不理解差分特别针对的是单位根/随机游走这种特定结构。
- 差分次数选择不当——差分不够(残留的非平稳性没有被处理干净)或者过度差分(引入不必要的人工自相关结构),都需要每次差分后重新检验平稳性,不是"差分一次就万事大吉"的一次性操作。

---

下一篇:[20-simple-forecasting.md](20-simple-forecasting.md) —— 板块V收官,把本文建立的平稳性/ACF直觉应用到移动平均、指数平滑这些简单预测方法上。
