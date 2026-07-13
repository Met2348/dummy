# 16 · Scaling law与外推深挖(Scaling Laws & Extrapolation)

> 总览见 [00-roadmap.md](00-roadmap.md)

14-15类处理的是"给定当前的模型/评测,怎么公平比较";本文转向一个更前瞻性的问题——"能不能从已有的规模档位,预测更大规模会怎样"。这正是scaling law研究的核心价值主张,也是统计上风险最高的操作之一。本文用05类回归方法论、04类bootstrap、14类模型选择的思想,拆解幂律拟合、拟合不确定性、外推风险、模型选择这几个环节里容易被忽视的统计陷阱。

**环境声明:** 全部代码在仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1)下真实跑通,随机抽样固定种子(`np.random.default_rng(42)`)。回归全部手写正规方程,与`scipy.optimize.curve_fit`交叉验证(和05类环境声明一致,不依赖statsmodels)。

---

## 1. 幂律关系与log-log线性化 —— 弯曲的曲线,拉直成一条直线

**定义与记号:** 幂律关系(power law):y=a·x^b,AI研究里最常见的具体化身是scaling law——模型loss随参数量N、数据量D、计算量C按幂律衰减(Kaplan et al. 2020、Chinchilla scaling law)。log-log线性化:对y=a·x^b两边取对数,ln(y)=ln(a)+b·ln(x),把非线性关系转换成ln(x)和ln(y)之间的**线性**关系,可以直接用05类已建立的最小二乘回归求解截距ln(a)和斜率b。

**一句话:** 幂律关系在原始坐标系下画出来是一条弯曲的曲线,不容易用肉眼判断"这真的是幂律吗、指数是多少";但同样的数据换到log-log坐标系下画出来,如果真的是幂律关系,会变成一条完美的直线——这个坐标变换本身就是判断"是不是幂律"和提取指数b最经典的手段。

**数学推导:** y=a·x^b,两边取自然对数:ln(y)=ln(a)+b·ln(x)。令Y=ln(y), X=ln(x), A=ln(a),这就是标准线性模型Y=A+bX,可以直接用05类知识点5的正规方程(X'X)⁻¹X'Y求解A和b,再用a=exp(A)换算回原始参数——这把一个非线性最小二乘问题(直接对y=ax^b拟合需要非线性优化,知识点2会展开)转化成了线性最小二乘问题,后者有解析解、计算稳定、不需要初始值猜测。

**底层机制/为什么这样设计:** 为什么要专门提取出这个"指数b"?因为在scaling law语境下,这个指数(scaling exponent)直接决定了"投入产出比"这个最关键的工程决策信息——b的绝对值越大,意味着增加x带来的y下降的边际收益衰减得越快,不同任务/架构的scaling exponent差异,直接指导"该往哪个维度(参数量、数据量、计算量)投入资源更划算"这类真实的资源分配决策。

**AI研究/工程场景:** Kaplan et al. 2020和Chinchilla(Hoffmann et al. 2022)这两篇著名scaling law论文,核心贡献之一就是通过大量训练不同规模模型的实验数据,在log-log坐标系下拟合出loss vs 参数量/数据量/计算量的幂律关系和对应指数,这些指数直接指导了后续大模型训练"该把预算主要投向更大模型还是更多数据"这类价值数亿美元的资源分配决策。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

true_a, true_b = 100.0, -0.3  # y = a * x^b, 负指数(loss随规模增大而下降)
x = np.exp(rng.uniform(np.log(1e3), np.log(1e7), 30))  # x跨越多个数量级
noise_mult = rng.normal(1, 0.05, 30)  # 乘性噪声(scaling law场景常见)
y = true_a * x ** true_b * noise_mult

log_x, log_y = np.log(x), np.log(y)
X_design = np.column_stack([np.ones(len(x)), log_x])
beta = np.linalg.solve(X_design.T @ X_design, X_design.T @ log_y)
a_hat, b_hat = np.exp(beta[0]), beta[1]

pred_loglog = X_design @ beta
r2_loglog = 1 - np.sum((log_y - pred_loglog) ** 2) / np.sum((log_y - log_y.mean()) ** 2)

# 核心断言1: log-log线性回归应该精确恢复真实的a, b
assert abs(b_hat - true_b) < 0.02, f"recovered exponent should be close to true_b={true_b}, got {b_hat:.4f}"
assert abs(a_hat - true_a) / true_a < 0.1, f"recovered a should be close to true_a={true_a}, got {a_hat:.4f}"
assert r2_loglog > 0.95, f"log-log fit should be excellent, got R^2={r2_loglog:.4f}"

# 核心断言2: 错误方法(原始空间直接线性回归, 忽视非线性关系)拟合优度应该明显更差
X_wrong = np.column_stack([np.ones(len(x)), x])
beta_wrong = np.linalg.solve(X_wrong.T @ X_wrong, X_wrong.T @ y)
pred_wrong = X_wrong @ beta_wrong
r2_wrong = 1 - np.sum((y - pred_wrong) ** 2) / np.sum((y - y.mean()) ** 2)
assert r2_wrong < 0.5, f"naive linear regression in raw space should fit poorly, got R^2={r2_wrong:.4f}"

print(f"true: a={true_a}, b={true_b}")
print(f"log-log fit: a_hat={a_hat:.4f}, b_hat={b_hat:.4f}, R^2={r2_loglog:.4f}")
print(f"naive raw-space linear fit R^2 = {r2_wrong:.4f}  (badly misspecified)")
```

**面试怎么问+追问链**(决策依据追问轴):
- Q:"怎么判断一组(x,y)数据背后是不是幂律关系,而不是别的函数形式(比如指数关系y=a·e^(bx))?"
- 追问1:"log-log坐标系下是直线,能不能唯一确定是幂律?"(log-log下是直线是幂律关系的特征,但要注意区分:指数关系y=a·e^(bx)在**半对数**坐标系(x线性、y取log)下才是直线,在log-log坐标系下通常不是直线;通过尝试几种常见的变换坐标系、看哪种坐标系下数据最接近直线,是判断函数形式的一种实用诊断方法)
- 深挖追问:"如果数据在log-log坐标系下大致呈直线,但两端明显偏离直线,说明了什么?"(说明纯粹的幂律关系可能只是一个局部近似,不是全局精确成立,很多真实的scaling law论文也会讨论"在什么规模区间内幂律近似成立"这个边界问题,这正是知识点4"外推风险"要重点讨论的场景)

**常见坑:**
- 直接在原始坐标系下对(x,y)做普通线性回归,而不做log变换,当真实关系是幂律(非线性)时,拟合出的"线性关系"和真实曲线完全不匹配。
- 忽视log-log线性化对**噪声结构**的隐含假设——原始空间的加性噪声(y=ax^b+ε)在取log之后不再是简单的加性噪声,这会让log空间的最小二乘估计出现轻微偏差,知识点2会展开这个细节。

---

## 2. 最小二乘拟合幂律参数 —— 与curve_fit交叉验证

**定义与记号:** 除了知识点1的log-log线性化,还可以直接在原始空间对y=a·x^b+ε做非线性最小二乘(nonlinear least squares)拟合——最小化Σ(y_i-a·x_i^b)²,用迭代优化算法(`scipy.optimize.curve_fit`底层的Levenberg-Marquardt算法)直接求解a、b,不需要先取对数。

**一句话:** log-log线性化和非线性最小二乘,是拟合同一个幂律关系的两种不同数学操作——前者先变换问题使其容易解(有解析解),但改变了噪声的结构;后者直接在原始空间求解(保持噪声结构不变),但需要迭代优化、没有解析解,两种方法的估计结果在噪声较小时应该接近。

**数学推导:** log-log线性化最小化的是Σ(ln(y_i)-ln(a)-b·ln(x_i))²(log空间的平方误差),等价于假设原始空间的**乘性**噪声结构y=a·x^b·ε(取log后变成可加噪声);非线性最小二乘最小化的是Σ(y_i-a·x_i^b)²(原始空间的平方误差),对应**加性**噪声结构y=a·x^b+ε。这两种噪声结构假设不同——如果数据的噪声确实是乘性的(y值跨越多个数量级,相对误差稳定,这在scaling law场景很常见),log-log线性化通常拟合得更好(更接近该噪声假设下的MLE)。

**底层机制/为什么这样设计:** 为什么两种方法拟合出的参数可能不完全一样?因为它们在优化两个不同的目标函数,对数据里每个点的"重视程度"不同——log-log线性化因为压缩了大数值的尺度,相当于给小y值的数据点分配了更大的相对权重;非线性最小二乘在原始空间直接优化,大y值的数据点因为绝对误差可能更大,会被赋予更大的实际权重——这个"隐式加权方式不同"是两种方法结果分歧的根本原因,不是谁"算错了"。

**AI研究/工程场景:** 实践中很多scaling law论文的拟合过程会同时报告"log空间拟合"和"直接非线性拟合"两种结果作为交叉验证,或者显式讨论用哪种损失函数更适合当前数据的噪声特性——这是一个经常被简化(默认只用其中一种)、但严谨分析应该有所交代的细节。

**可运行例子:**
```python
import numpy as np
from scipy.optimize import curve_fit

rng = np.random.default_rng(42)

true_a, true_b = 100.0, -0.3
x = np.exp(rng.uniform(np.log(1e3), np.log(1e7), 30))
noise_mult = rng.normal(1, 0.05, 30)
y = true_a * x ** true_b * noise_mult

def power_law(x, a, b):
    return a * np.power(x, b)

# scipy非线性最小二乘
popt, _ = curve_fit(power_law, x, y, p0=[1.0, -0.1])
a_nls, b_nls = popt

# 手写log-log线性回归(知识点1的方法)
log_x, log_y = np.log(x), np.log(y)
X_design = np.column_stack([np.ones(len(x)), log_x])
beta = np.linalg.solve(X_design.T @ X_design, X_design.T @ log_y)
a_loglog, b_loglog = np.exp(beta[0]), beta[1]

# 核心断言: 噪声较小时, 两种方法给出的估计应该都准确、且彼此接近
assert abs(b_nls - true_b) < 0.03, f"NLS should recover b accurately, got {b_nls:.4f}"
assert abs(b_loglog - true_b) < 0.03, f"log-log should recover b accurately, got {b_loglog:.4f}"
assert abs(b_nls - b_loglog) < 0.02, f"the two methods should agree closely under mild noise: {b_nls:.4f} vs {b_loglog:.4f}"

print(f"true b = {true_b}")
print(f"scipy curve_fit (nonlinear LS): a={a_nls:.4f}, b={b_nls:.4f}")
print(f"hand-written log-log linear regression: a={a_loglog:.4f}, b={b_loglog:.4f}")
print(f"agreement: |b_nls - b_loglog| = {abs(b_nls - b_loglog):.4f}")
```

**面试怎么问+追问链**(方案批判迭代轴):
- Q:"拟合scaling law的参数,是应该做log-log线性回归还是直接用curve_fit做非线性拟合?"
- 追问1:"这两种方法的选择依据是什么?"(取决于数据的噪声结构更接近乘性还是加性——如果y值本身跨越多个数量级,而相对误差(不是绝对误差)在不同规模下保持稳定,更接近乘性噪声,log-log线性化更合适;如果绝对误差幅度本身在不同x取值下大致稳定,更接近加性噪声,直接非线性拟合更合适)
- 深挖追问:"如果不确定噪声结构是哪种,有没有办法用数据本身来判断?"(可以观察残差幅度是否随x或拟合值本身系统性变化——如果残差幅度随y的量级增大而成比例增大,支持乘性噪声假设;如果残差幅度和y的量级无关,支持加性噪声假设,这是05类知识点6残差诊断思路的复用)

**常见坑:**
- 不加说明地混用两种拟合方法的结果(比如用log-log线性化拟合参数,却用原始空间的R²评估拟合优度,两者的优化目标不一致)。
- 认为"非线性最小二乘更精确所以总是应该优先选择"——这个判断忽视了两种方法本质上对应不同的噪声假设,"哪个更精确"取决于哪个假设更符合真实数据特性。

---

## 3. 拟合不确定性 —— bootstrap量化scaling exponent的置信区间

**定义与记号:** 用bootstrap方法(04类bootstrap CI+05类回归系数CI+本文知识点1-2的拟合方法)量化scaling law拟合出的参数(a、b)本身的不确定性——对观测数据做bootstrap重采样,每次重新拟合出一组(a,b),重复多次得到参数的bootstrap分布,这个分布的宽度直接反映"如果重新收集一批同样规模的训练数据点,拟合出的scaling exponent大概会有多大波动"。

**一句话:** 论文里报告的"scaling exponent α=0.34"这个数字,如果不知道它的置信区间有多宽,几乎没有办法判断"这个具体数值有多可信、和另一篇论文报告的α=0.31是不是真的有意义的差异"——bootstrap给出的参数不确定性,是让scaling law的具体数值可以被严肃比较的前提。

**数学推导:** 复用知识点1-2的拟合方法作为bootstrap内部的"拟合函数":对原始n个数据点做B次有放回重采样,对每次重采样的数据独立重新拟合出(a_b,b_b),得到B组参数估计,这B组估计值的经验分布就是对真实参数抽样分布的bootstrap近似——这和05类知识点5"回归系数CI"处理的是同一类问题(回归系数是数据的函数,自然有抽样分布),只是这里的"回归"是log-log空间的线性回归,bootstrap方法论完全通用,不需要为每种回归形式重新发明一套不确定性量化方法。

**底层机制/为什么这样设计:** 为什么bootstrap对scaling law参数的不确定性量化特别有价值,而不是直接用回归系数的解析标准误公式?因为如果用非线性最小二乘拟合(知识点2),参数的解析标准误公式要么不存在闭式解、要么依赖复杂的渐近正态性近似,而bootstrap方法不需要关心具体的拟合方法是线性还是非线性,只需要能够"对任意一份重采样数据跑一次拟合流程",就能得到不确定性估计,这是bootstrap相对解析方法的通用性优势在这个更复杂拟合场景下的再次体现。

**AI研究/工程场景:** 严肃的scaling law研究论文在报告拟合出的指数时,应该同时报告置信区间或标准误(实际训练大模型做scaling law实验成本极高,数据点数量往往有限,只在几个规模档位各训练一两个模型),忽视这一点直接把拟合出的点估计当作精确值去指导后续资源分配决策,是一种常见但危险的简化。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

def fit_loglog(x, y):
    log_x, log_y = np.log(x), np.log(y)
    X_design = np.column_stack([np.ones(len(x)), log_x])
    beta = np.linalg.solve(X_design.T @ X_design, X_design.T @ log_y)
    return np.exp(beta[0]), beta[1]

true_a, true_b = 100.0, -0.3

# 场景A: 数据点较多(n=30)
n_points = 30
x = np.exp(rng.uniform(np.log(1e3), np.log(1e7), n_points))
y = true_a * x ** true_b * rng.normal(1, 0.05, n_points)

n_boot = 2000
boot_b = np.zeros(n_boot)
for i in range(n_boot):
    idx = rng.integers(0, n_points, n_points)
    _, b_b = fit_loglog(x[idx], y[idx])
    boot_b[i] = b_b
ci_b = np.percentile(boot_b, [2.5, 97.5])

# 核心断言1: bootstrap置信区间应该覆盖真实值
assert ci_b[0] <= true_b <= ci_b[1], f"bootstrap CI should cover the true b={true_b}, got CI={tuple(ci_b)}"

# 场景B: 数据点较少(n=10, 更接近真实scaling law实验的规模档位数量)
n_points_small = 10
x_small = np.exp(rng.uniform(np.log(1e3), np.log(1e7), n_points_small))
y_small = true_a * x_small ** true_b * rng.normal(1, 0.05, n_points_small)

boot_b_small = []
for i in range(n_boot):
    idx = rng.integers(0, n_points_small, n_points_small)
    if len(np.unique(idx)) < 2:  # 退化重采样(几乎所有点都相同), 无法拟合直线, 跳过
        continue
    _, b_b = fit_loglog(x_small[idx], y_small[idx])
    boot_b_small.append(b_b)
ci_b_small = np.percentile(boot_b_small, [2.5, 97.5])

# 核心断言2: 数据点更少时, bootstrap置信区间应该明显更宽
width_large = ci_b[1] - ci_b[0]
width_small = ci_b_small[1] - ci_b_small[0]
assert width_small > width_large * 1.5, \
    f"fewer data points should give a visibly wider CI: n=30 width={width_large:.4f}  n=10 width={width_small:.4f}"

print(f"n=30: bootstrap CI for b = ({ci_b[0]:.4f}, {ci_b[1]:.4f}), width={width_large:.4f}")
print(f"n=10: bootstrap CI for b = ({ci_b_small[0]:.4f}, {ci_b_small[1]:.4f}), width={width_small:.4f}")
print(f"=> fewer regime points (realistic for expensive scaling-law experiments) means much wider uncertainty")
```

**面试怎么问+追问链**(真实性验证轴):
- Q:"论文报告scaling exponent是0.34,你的复现实验拟合出0.31,这个差异说明复现失败了吗?"
- 追问1:"怎么判断0.34和0.31这个差异是否'有意义'?"(需要知道各自拟合的置信区间或标准误——如果两者的置信区间大幅重叠,0.03的差异完全可能是训练数据点数量有限导致的正常抽样波动,不代表复现有问题;只有当置信区间基本不重叠时,才有理由怀疑存在真实的方法论或实现差异)
- 深挖追问:"如果确实需要把scaling exponent的置信区间控制得更窄,应该怎么做?"(和14类知识点5"评测集大小与CI宽度"是同一类问题——需要更多的数据点(更多规模档位的训练实验)来缩小拟合不确定性,但scaling law实验的每个数据点(训练一个模型)成本极高,这个"想要更精确的指数估计,需要付出多大的实验成本"的权衡,是真实scaling law研究里一个重要但经常被低估的实践约束)

**常见坑:**
- 只报告拟合出的点估计,不给出任何形式的不确定性量化,让读者无法判断这个数字的精确程度。
- 数据点数量很少(比如只有3-4个)时依然直接做bootstrap,bootstrap方法本身在样本量极小时的可靠性也会下降,需要谨慎解读极小样本下bootstrap给出的区间。

---

## 4. 外推风险 —— 预测区间宽度随外推距离加速增长

**定义与记号:** 外推(extrapolation):用拟合出的scaling law公式,预测**训练数据规模范围之外**的x值对应的y值。外推风险:预测区间的宽度不是均匀的,而是随着外推距离(预测点离已观测数据范围的远近)增大而**加速**增大,在训练数据范围内插值相对可靠,范围外的外推距离越远,预测的不确定性放大得越快。

**一句话:** "模型在训练数据覆盖的规模范围内,scaling law拟合得很好(R²=0.99)"这句话,完全不能保证"用这个公式预测10倍大的模型的loss也会同样准确"——插值准和外推准是两个完全不同的问题,前者的高精度不能自动转移给后者。

**数学推导:** 用05类知识点5已建立的回归预测区间框架:对于线性回归Y=A+bX(log空间),给定新的X_new,预测区间宽度正比于 σ·√(1+1/n+(X_new-X̄)²/Sxx)——第二项(X_new-X̄)²/Sxx直接表明,预测点离训练数据均值越远,这一项就越大,且是**平方**关系,意味着这一项主导时,外推距离翻倍,对区间宽度的贡献会加速放大,不是线性增长。

**底层机制/为什么这样设计:** 为什么外推的不确定性会以加速的方式增长?因为线性回归(在log空间)拟合出的直线,是由训练数据点"钉住"的,在数据点密集的区域,很多种略有不同的直线斜率/截距组合都能同样好地拟合这些点,不确定性相互制约、比较小;但离数据越远,同样这些"略有不同的直线"之间的预测值差异会被距离放大——这是几何直觉(不同斜率的直线,越往外延伸,彼此之间分叉得越开)在统计公式里的精确体现。

**AI研究/工程场景:** Scaling law最有价值(也最有风险)的应用场景就是外推——用几个中小规模模型的训练结果,预测"如果训练一个10倍大的模型,loss会降到多少",从而在真正投入巨大成本训练超大模型之前先做出资源分配决策;历史上确实出现过大规模训练的实际表现和小规模外推预测出现偏离的案例(比如涌现能力/emergent abilities这类现象,在小规模数据点拟合出的平滑幂律曲线上完全无法预见),这是scaling law研究领域一个真实存在、被严肃对待的方法论局限。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

true_a, true_b = 100.0, -0.3
n_points = 30
x = np.exp(rng.uniform(np.log(1e5), np.log(1e6), n_points))  # 训练数据规模范围: 1e5 到 1e6
y = true_a * x ** true_b * rng.normal(1, 0.05, n_points)

log_x, log_y = np.log(x), np.log(y)
X_design = np.column_stack([np.ones(n_points), log_x])
beta = np.linalg.solve(X_design.T @ X_design, X_design.T @ log_y)
resid = log_y - X_design @ beta
sigma_hat = np.sqrt(np.sum(resid ** 2) / (n_points - 2))
log_x_bar = log_x.mean()
Sxx = np.sum((log_x - log_x_bar) ** 2)

def pred_interval_width(x_new, alpha=0.05):
    log_x_new = np.log(x_new)
    se_pred = sigma_hat * np.sqrt(1 + 1 / n_points + (log_x_new - log_x_bar) ** 2 / Sxx)
    t_crit = stats.t.ppf(1 - alpha / 2, n_points - 2)
    return 2 * t_crit * se_pred  # log空间的区间宽度

# 训练范围内(插值), 训练范围上限(1e6)的2倍/10倍/100倍(外推)
widths = [pred_interval_width(v) for v in [3e5, 2e6, 1e7, 1e8]]

# 核心断言1: 区间宽度应该随外推距离单调增大
assert all(widths[i] < widths[i + 1] for i in range(len(widths) - 1)), f"widths should increase monotonically: {widths}"

# 核心断言2: 增长应该是加速的(增量本身也在增大), 不只是简单的线性增长
increments = np.diff(widths)
assert all(increments[i] < increments[i + 1] for i in range(len(increments) - 1)), \
    f"the growth should accelerate (increasing increments), not just increase linearly: increments={increments}"

print(f"prediction interval width: interpolation(3e5)={widths[0]:.4f}")
print(f"  2x beyond training max(2e6)={widths[1]:.4f}  (+{increments[0]:.4f})")
print(f"  10x beyond training max(1e7)={widths[2]:.4f}  (+{increments[1]:.4f})")
print(f"  100x beyond training max(1e8)={widths[3]:.4f}  (+{increments[2]:.4f})")
print("=> the width doesn't just grow, it grows FASTER the further you extrapolate")
```

**面试怎么问+追问链**(规模递增轴):
- Q:"我们在1亿到10亿参数规模的模型上拟合出了一条scaling law曲线,R²=0.98,能不能用这条曲线预测1000亿参数模型的loss?"
- 追问1:"这个预测的可信度如何评估?"(需要计算在1000亿参数这个点上的预测区间宽度,而不是只看训练数据范围内的拟合优度——R²=0.98描述的是"在已训练规模范围内拟合得多好",不直接说明"外推到1000亿参数这个预测有多准",两者是不同的问题)
- 深挖追问:"如果历史上同类型模型的scaling law外推预测最终被证明基本准确,是不是说明这次的外推也可以放心相信?"(历史准确不能作为这次外推可靠的充分证据——需要具体分析这次外推的距离量级是否和历史上验证过准确的外推距离量级相当;此外还需要考虑外推范围内是否可能出现新的定性变化(比如涌现能力),纯粹的统计外推无法预见这类结构性变化)

**常见坑:**
- 只关注插值区间内的拟合优度(R²、残差大小),把这个"拟合得很好"的印象不加区分地当作外推预测同样可靠的证据。
- 完全不做任何形式的外推不确定性量化,直接把外推预测出的点估计当作确定的数字用于重大资源分配决策。

---

## 5. 模型选择直觉(简单AIC/BIC)—— 训练集拟合得好不等于泛化好

**定义与记号:** 面对同一份数据,幂律模型(2个参数a,b)和更高次多项式模型(比如9次多项式,10个参数)哪个更合适?多项式模型因为参数更多、更灵活,几乎总能在训练数据本身上取得更低的拟合误差,但这不代表泛化(样本外预测)能力更好。AIC/BIC是两种对"拟合优度"和"模型复杂度"做权衡的模型选择准则:AIC=n·ln(RSS/n)+2k,BIC=n·ln(RSS/n)+k·ln(n)(k是参数个数,n是样本量),复杂度惩罚项会对参数更多的模型做出"扣分"。

**一句话:** 用一个10参数的多项式去拟合本质上只需要2个参数就能描述的幂律数据,几乎总能在训练数据上拟合出更小的误差(参数越多,曲线越能扭曲穿过每一个训练数据点),但这种"更好的拟合"很大程度是在拟合噪声而不是真实规律,AIC/BIC通过显式的复杂度惩罚项,把"该不该为了这点拟合优度的提升,多花几个参数的复杂度代价"这个直觉判断变成一个可以直接计算比较的数字。

**数学推导:** 对于正态噪声的最小二乘拟合,-2ln(L̂)可以直接用残差平方和(RSS)表示:-2ln(L̂)=n·ln(RSS/n)+常数项。所以AIC和BIC的第一项都是"拟合优度"(RSS越小越好),第二项都是"复杂度惩罚"(参数k越多惩罚越大),BIC的惩罚系数ln(n)通常比AIC的惩罚系数2更大(n>7时ln(n)>2),BIC对复杂度的惩罚比AIC更严格。选择AIC/BIC更小的模型。**重要的实践限制**:当参数个数k接近样本量n时(残差自由度n-k很小),RSS可以被压得极低(接近完美插值),此时n·ln(RSS/n)这一项会主导,可能让AIC/BIC的复杂度惩罚都不足以纠正过拟合——这不是AIC/BIC"失效",而是它们的渐近推导本身要求n远大于k才可靠,样本量必须相对参数个数有足够富余,AIC/BIC才能正确发挥作用。

**底层机制/为什么这样设计:** 为什么"参数越多,训练误差越小"这个现象本身不能作为"模型更好"的证据?因为参数更多的模型有更强的"记忆训练数据里随机噪声"的能力,这种过拟合在训练数据本身上表现得像是"拟合得更好",但用来预测新数据时,这部分"记住的噪声"完全没有泛化能力,反而会让预测变差——AIC/BIC本质上是在用一个理论上推导出的惩罚项,近似模拟"如果拿一份独立的样本外数据来检验,复杂模型的表现会打多少折扣"这件事,不需要真的准备一份独立测试集就能做出接近的判断(虽然14类知识点6已经强调过,独立测试集永远是更直接、更可信的验证方式)。

**AI研究/工程场景:** 拟合scaling law时,如果观测到的(参数量,loss)数据点之间存在一些非单调的小波动,用一个足够高次的多项式几乎总能"完美"穿过所有观测点,但这样拟合出的曲线外推到未观测的规模区间时会给出完全不可靠、甚至荒谬的预测——AIC/BIC提供了一个防止研究者被"训练集拟合优度"这个表面数字诱导选择过于复杂模型的量化工具。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

true_a, true_b = 100.0, -0.3
n_points = 15  # 样本量适中: 足够让AIC/BIC的渐近近似有意义, 又不至于让高次多项式轻易插值
x_train = np.exp(rng.uniform(np.log(1e4), np.log(1e6), n_points))
y_train = true_a * x_train ** true_b * rng.normal(1, 0.05, n_points)
log_x_train = np.log(x_train)

# 模型1: 幂律(log-log线性, 2参数)
X1 = np.column_stack([np.ones(n_points), log_x_train])
beta1_log = np.linalg.solve(X1.T @ X1, X1.T @ np.log(y_train))
a_hat, b_hat = np.exp(beta1_log[0]), beta1_log[1]
pred1 = a_hat * x_train ** b_hat
rss1 = np.sum((y_train - pred1) ** 2)
k1 = 2

# 模型2: 9次多项式(10参数), 标准化log_x避免高次幂的数值病态
lx_mean, lx_std = log_x_train.mean(), log_x_train.std()
z_train = (log_x_train - lx_mean) / lx_std
degree = 9
k2 = degree + 1
X2 = np.column_stack([z_train ** p for p in range(k2)])
beta2 = np.linalg.solve(X2.T @ X2, X2.T @ y_train)
pred2 = X2 @ beta2
rss2 = np.sum((y_train - pred2) ** 2)

n = n_points
aic1, aic2 = n * np.log(rss1 / n) + 2 * k1, n * np.log(rss2 / n) + 2 * k2
bic1, bic2 = n * np.log(rss1 / n) + k1 * np.log(n), n * np.log(rss2 / n) + k2 * np.log(n)

# 核心断言1: 多项式在训练集上拟合误差确实更小(表面上"拟合得更好")
assert rss2 < rss1, f"the more flexible polynomial should fit training data at least as well: rss1={rss1:.4f} rss2={rss2:.4f}"

# 核心断言2: 但考虑复杂度惩罚后, AIC和BIC都应该判定幂律模型更优
assert aic1 < aic2, f"AIC should prefer the simpler power-law model, got aic1={aic1:.2f} aic2={aic2:.2f}"
assert bic1 < bic2, f"BIC should prefer the simpler power-law model, got bic1={bic1:.2f} bic2={bic2:.2f}"

# 核心断言3: 独立样本外测试集上, 多项式的预测误差应该远差于幂律模型(真正验证过拟合)
x_test = np.exp(rng.uniform(np.log(1e4), np.log(1e6), 300))
y_test = true_a * x_test ** true_b * rng.normal(1, 0.05, 300)
log_x_test = np.log(x_test)
z_test = (log_x_test - lx_mean) / lx_std
pred1_test = a_hat * x_test ** b_hat
pred2_test = np.column_stack([z_test ** p for p in range(k2)]) @ beta2
mse1_test = np.mean((y_test - pred1_test) ** 2)
mse2_test = np.mean((y_test - pred2_test) ** 2)
assert mse2_test > mse1_test * 5, \
    f"the polynomial's out-of-sample error should be dramatically worse: power={mse1_test:.4f} poly={mse2_test:.4f}"

print(f"training RSS: power-law(k=2)={rss1:.4f}  polynomial(k={k2})={rss2:.4f}  (polynomial 'fits better')")
print(f"AIC: power={aic1:.2f}  polynomial={aic2:.2f}  (AIC correctly prefers power-law)")
print(f"BIC: power={bic1:.2f}  polynomial={bic2:.2f}  (BIC correctly prefers power-law)")
print(f"out-of-sample MSE: power={mse1_test:.4f}  polynomial={mse2_test:.4f}  (ratio={mse2_test/mse1_test:.1f}x -- severe overfitting)")
```

**面试怎么问+追问链**(方案批判迭代轴,呼应14类模型评测统计的核心思想):
- Q:"我们试了幂律模型和一个更复杂的多项式模型拟合scaling law数据,多项式模型的训练集R²更高,是不是应该选多项式模型?"
- 追问1:"训练集R²更高能说明什么、不能说明什么?"(只能说明"在这些具体的训练数据点上,多项式模型的拟合误差更小",不能说明"多项式模型更准确地刻画了真实的scaling规律"——如果多项式的额外灵活性是在拟合训练数据里的随机噪声而不是真实信号,这个"更高的R²"是虚假的优势,这正是14类知识点6"刷榜的隐藏多重比较"同一原理在模型选择场景的另一个化身:模型越灵活、可调整的自由度越多,越容易在特定数据集上"刷"出好看的拟合指标)
- 深挖追问:"AIC和BIC经常给出不同的模型选择结论,应该信哪个?"(AIC的理论目标更接近最小化预测误差,BIC的理论目标是在贝叶斯框架下选出"真实生成数据的模型";如果目标是纯粹的预测性能,AIC的理论动机更贴近;如果有独立的测试集可用,14类已经强调过的"直接用样本外数据验证"永远比任何理论准则更直接可信,理论准则是在没有独立测试集时的次优替代方案)

**常见坑:**
- 单纯根据训练集拟合优度(R²、RSS)选择模型,不考虑模型复杂度,系统性地偏向选择过于复杂、容易过拟合的模型。
- 把AIC/BIC当作绝对客观、不需要再验证的最终结论,而不去做14类已经建立的更直接的独立测试集样本外验证,也不注意"参数个数接近样本量时AIC/BIC的渐近近似本身会失效"这个实践限制。

---

下一篇:[17-distribution-shift-and-monitoring.md](17-distribution-shift-and-monitoring.md) —— scaling law预测的是"训练时的性能会怎样";本文之后转向"部署之后,真实世界的数据分布本身在变化"这个同样重要的问题。
