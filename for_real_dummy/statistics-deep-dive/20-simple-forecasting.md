# 20 · 简单预测方法深挖(Simple Forecasting Methods)

> 总览见 [00-roadmap.md](00-roadmap.md)

板块V收官篇(19-20类,共11个知识点)。上一篇建立了理解时间序列的最小必要词汇——自相关、平稳性、ACF/PACF、白噪声、随机游走、差分。本文把这套直觉落地成几种真正能拿来做预测的具体方法:移动平均、指数平滑、最简单的AR(1)预测,以及贯穿全文最重要的一条纪律——评价一个预测方法好不好,不能只看"这个方法听起来是不是更高级",必须用MAE/RMSE/MAPE这些具体指标,和一个几乎不需要任何工作量的朴素基线(预测=昨天的值)真实地比出来,而且要如实接受"朴素基线有时候反而赢了"这个结果,不为了让"复杂方法更有用"这个期望中的故事成立而回避它。延续19类的原则,知识点3(简单AR(1))只用最小二乘直觉建立"用过去值的线性组合预测未来值"这一个具体做法,不深入完整ARIMA的形式化数学,不假设随机过程先修知识。

**环境声明:** 全部代码在仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1)下真实跑通,随机抽样固定种子(`np.random.default_rng(42)`)。移动平均/指数平滑/AR(1)最小二乘全部手写实现。

---

## 1. 移动平均 —— 平均掉噪声,但跟不上变化

**定义与记号:** 移动平均(Moving Average,MA):用最近k个观测值的算术平均作为对当前水平的估计,MA_t=(1/k)Σ_{i=0}^{k-1}X_{t-i}。k(窗口大小)是唯一的调节参数。

**一句话:** 移动平均用"平均掉噪声"换来"对真实变化的反应变慢"——窗口越大,平滑掉的随机波动越多,但对真实发生的水平变化的察觉也越滞后,这是一个没有免费午餐的权衡,不存在同时最优的k。

**数学推导:** 若X_t=μ+ε_t(真实水平μ恒定,ε_t是均值0、方差σ²、彼此独立的噪声——01类已建立的独立方差可加性质),则MA_t是k个独立同分布噪声的平均加上常数μ,Var(MA_t)=Var((1/k)Σε_{t-i})=(1/k²)·k·σ²=σ²/k——窗口越大,估计方差按1/k的速率下降,这是一个精确的代数结果,不是近似。

**底层机制/为什么这样设计:** 为什么窗口增大不是"免费"的?上面的推导有一个关键前提——真实水平μ必须在窗口跨越的k个时间点内保持不变。如果真实水平在这k个点内发生了变化(比如一次真实的业务水平跃升),MA_t还在用"旧水平和新水平混合"的k个点计算平均,直到新水平完全"冲刷掉"窗口内的旧数据(需要大约k步),MA才能完全跟上新水平——**窗口越大,方差降得越多,但对真实变化的滞后也越长**,这就是数学推导里"降噪"和可运行例子里"滞后"这两个效应此消彼长的根源。

**AI研究/工程场景:** 监控一个线上服务的延迟指标,用移动平均做平滑展示是最常见的做法之一——如果只是想让监控图表不那么"抖动",大窗口没问题;但如果这个移动平均值本身要触发告警(比如"超过阈值就报警"),窗口选得太大意味着真实的延迟劣化要等很多个周期之后告警才会触发,这个"检测滞后"在真实事故场景下可能就是几分钟到几十分钟的额外损失。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

n = 200
true_level = np.concatenate([np.zeros(100), np.ones(100) * 5.0])  # 第100步真实水平从0跳变到5
noise = rng.normal(0, 2.0, n)
X = true_level + noise

def moving_average(x, k):
    ma = np.full(len(x), np.nan)
    for t in range(k - 1, len(x)):
        ma[t] = np.mean(x[t - k + 1:t + 1])
    return ma

ma_short = moving_average(X, 5)
ma_long = moving_average(X, 20)

def residual_var(ma, k):
    idx = np.arange(len(X))
    valid = (idx >= k - 1) & ((idx < 90) | (idx >= 110))  # 排除跳变附近的过渡区间
    return np.var(ma[valid] - true_level[valid])

var_short = residual_var(ma_short, 5)
var_long = residual_var(ma_long, 20)

def detect_lag(ma):
    for t in range(100, len(ma)):
        if ma[t] > 2.5:  # 真实水平从0跳变到5, 用中点2.5作为"已经跟上跳变"的判据
            return t - 100
    return None

lag_short, lag_long = detect_lag(ma_short), detect_lag(ma_long)

# 核心断言1: 窗口越大, 平稳区间内估计方差越小(理论 sigma^2/k: 4/5=0.8 vs 4/20=0.2)
assert var_long < var_short, f"larger window should smooth more: var_short={var_short:.4f}, var_long={var_long:.4f}"

# 核心断言2: 窗口越大, 检测到跳变的滞后越长
assert lag_long > lag_short, f"larger window should lag more: lag_short={lag_short}, lag_long={lag_long}"

print(f"window=5:  residual variance={var_short:.4f}  lag to detect jump={lag_short} steps")
print(f"window=20: residual variance={var_long:.4f}  lag to detect jump={lag_long} steps")
print("=> larger window: less noise, more lag -- no free lunch")
```

**面试怎么问+追问链**(方案批判迭代轴):
- Q:"用移动平均做异常检测的告警阈值,窗口从5改成50,监控图表变得平滑很多,能不能直接这么上线?"
- 追问1:"这个改动有什么代价?"(窗口从5变成50,对真实水平变化的检测滞后会从个位数步数变成大约50步——如果检测周期是1分钟,意味着真实故障要等接近1小时才会触发告警,这在真实事故场景下可能是不可接受的)
- 深挖追问:"有没有办法在不牺牲这么多检测速度的前提下降噪?"(这正是知识点2指数平滑要解决的问题——用更聪明的加权方式,而不是简单粗暴地放大窗口)

**常见坑:**
- 只根据"图表好不好看"选窗口大小,不考虑这个移动平均值下游是否要用来做决策(触发告警、驱动业务决策),对滞后的容忍度可能完全不同。
- 误以为移动平均的降噪效果是"越大越好"没有代价——数学推导已经明确给出方差下降和滞后增长是同一个k变化产生的两个同步效应,不能只看其中一个。

---

## 2. 指数平滑 —— 让最近的数据天然更重要

**定义与记号:** 指数平滑(Exponential Smoothing):Ŝ_t=α·X_t+(1-α)·Ŝ_{t-1},Ŝ_0=X_0,α∈(0,1)是平滑系数,是移动平均之外另一种对当前水平做递推估计的方法。

**一句话:** 移动平均对窗口内的k个点给完全相等的权重,窗口外直接给0权重;指数平滑反过来,对**所有**历史数据都给非零权重,只不过权重随着"数据多旧"按几何级数衰减——离现在越近权重越大,这个"加权方式"上的本质差异,是本知识点要展开对比的核心。

**数学推导:** 把递推式反复展开(纯代数展开,不需要额外假设):Ŝ_t=α·X_t+(1-α)[α·X_{t-1}+(1-α)Ŝ_{t-2}]=...=Σ_{j=0}^{t-1}α(1-α)^j·X_{t-j}+(1-α)^t·X_0——当t足够大,(1-α)^t·X_0这一项衰减到可以忽略,Ŝ_t近似等于历史观测值按权重α(1-α)^j(j是"多少步之前")的几何加权和,权重之和Σ_{j=0}^{∞}α(1-α)^j=α·1/(1-(1-α))=1,精确等于1(等比数列求和公式),保证这是一个合法的加权平均。

**底层机制/为什么这样设计:** 为什么几何衰减权重比移动平均的"矩形"权重更合理?移动平均对窗口内最老的一个点和最新的一个点给同样的权重,这隐含了一个不太自然的假设——"k步之前的数据和1步之前的数据同等重要,但k+1步之前的数据完全不重要"(权重从1/k直接断崖式跌到0);指数平滑的几何衰减权重更符合直觉——越新的数据越重要,重要性平滑地递减,不存在这种人为的断崖,这也是为什么指数平滑只需要一个参数α就能连续调节"记忆长度",而不像移动平均那样需要一个离散的窗口大小k。

**AI研究/工程场景:** 深度学习训练里监控loss曲线时常用的"平滑"展示(TensorBoard的smoothing滑块),以及优化器动量项(momentum,Adam的一阶矩估计)的更新公式,本质上都是指数平滑的直接应用——`m_t=β·m_{t-1}+(1-β)·g_t`和指数平滑的递推式是同一个数学结构,只是符号约定不同(β对应1-α),这是指数平滑这个"简单预测方法"在AI工程里最常见、但很少被点破的真实身影。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

def exp_smooth(x, alpha):
    s = np.zeros(len(x))
    s[0] = x[0]
    for t in range(1, len(x)):
        s[t] = alpha * x[t] + (1 - alpha) * s[t - 1]
    return s

# 数学推导验证1: 递推定义 和 几何加权闭式解(含边界项) 精确相等, 不是近似
alpha = 0.3
x = rng.normal(0, 1, 60)
s = exp_smooth(x, alpha)

def closed_form(x, alpha, t):
    total = (1 - alpha) ** t * x[0]
    for j in range(t):
        total += alpha * (1 - alpha) ** j * x[t - j]
    return total

for t in [1, 5, 10, 30, 59]:
    assert abs(s[t] - closed_form(x, alpha, t)) < 1e-9, f"recursive and closed form should match exactly at t={t}"

# 数学推导验证2: 权重和趋于1(等比数列求和公式)
weights_50 = np.array([alpha * (1 - alpha) ** j for j in range(50)])
assert abs(np.sum(weights_50) - 1.0) < 1e-6, f"weights should sum to ~1, got {np.sum(weights_50):.8f}"

# 核心对比: 相近"记忆长度"下, 指数平滑对最近数据的加权明显更集中
k = 10
alpha2 = 0.2  # 常见经验换算: alpha ~ 2/(k+1)
ma_weight_each = 1.0 / k
ema_weights = np.array([alpha2 * (1 - alpha2) ** j for j in range(k)])

sum_ema_recent3 = np.sum(ema_weights[:3])
sum_ma_recent3 = ma_weight_each * 3
ema_weight_oldest = ema_weights[k - 1]  # 窗口内最老的一个点(第10个)

# 核心断言1: 指数平滑给最近3个点的权重之和明显高于移动平均
assert sum_ema_recent3 > sum_ma_recent3, \
    f"EMA should concentrate more weight on recent points: ema={sum_ema_recent3:.4f}, ma={sum_ma_recent3:.4f}"

# 核心断言2: 指数平滑给窗口内最老的点的权重明显低于移动平均(不存在断崖式的0)
assert ema_weight_oldest < ma_weight_each, \
    f"EMA should down-weight the oldest point vs MA's flat weight: ema={ema_weight_oldest:.4f}, ma={ma_weight_each:.4f}"

print(f"MA(k={k}) weight on each of the last {k} points: {ma_weight_each:.4f} (flat)")
print(f"EMA(alpha={alpha2}) weights (newest -> oldest): {np.round(ema_weights, 4)}")
print(f"sum of weight on most-recent-3: EMA={sum_ema_recent3:.4f} vs MA={sum_ma_recent3:.4f}")
```

**面试怎么问+追问链**(决策依据追问轴):
- Q:"同样是做平滑,什么情况下该选移动平均,什么情况下该选指数平滑?"
- 追问1:"两者的本质区别是什么,不是'哪个效果更好'这种笼统说法?"(移动平均对窗口内数据一视同仁、窗口外完全不看,只有一个离散的k可调;指数平滑对全部历史数据都有非零权重,只是越老权重越小,只有一个连续的α可调,更适合"希望最新数据的重要性天然更高,而且希望调节记忆长度更平滑"的场景)
- 深挖追问:"指数平滑的α怎么选,和移动平均的k怎么选,道理上是不是一回事?"(是同一件事的两种参数化——α越大(或k越小)对最新数据越敏感、越不稳定;α越小(或k越大)越平滑、越滞后,本质上是同一个"降噪vs滞后"权衡,只是参数的连续性不同)

**常见坑:**
- 认为指数平滑"更高级"所以总是比移动平均更好——两者只是加权方式不同,各自适合的场景不完全相同,不存在一种方法全面碾压另一种。
- 混淆α的方向——α越大表示越依赖最新观测值(越不平滑,反应越快),α越小才是越平滑越滞后,和移动平均的k方向正好相反(k越大越平滑),初次接触容易搞反直觉方向。

---

## 3. 简单AR(1)概念(初等形式) —— 用过去值的线性组合预测未来

**定义与记号:** AR(1)(一阶自回归,19类知识点1已引入其递推定义):X_t=φ·X_{t-1}+ε_t。这里聚焦一个具体、可操作的问题——**给定观测数据,怎么反过来估计φ**,并用估计出的φ̂做预测:X̂_{t+1}=φ̂·X_t。(**范围声明**:本知识点只讲这一个初等形式,不展开完整ARIMA的形式化数学,不需要随机过程先修知识——估计方法就是普通最小二乘,和05类回归推断用的是同一个工具。)

**一句话:** AR(1)预测的核心思想是"用昨天的值乘上一个系数,就是对今天的最佳线性预测",这个系数φ不是猜出来或者拍脑袋定出来的,而是像05类的线性回归系数一样,直接用最小二乘从历史数据里估计出来的。

**数学推导:** 把AR(1)递推式X_t=φX_{t-1}+ε_t看成一个"用X_{t-1}预测X_t"的回归问题(没有截距项,因为均值已经假设为0),最小二乘估计量是让Σ(X_t-φX_{t-1})²最小的φ。对φ求导令其为0:

-2Σ(X_t-φX_{t-1})X_{t-1} = 0

两边同时除以-2,再把括号乘开(φ是常数,可以从求和号里提出来):

Σ(X_t-φX_{t-1})X_{t-1} = ΣX_tX_{t-1} - φΣX_{t-1}² = 0

移项:ΣX_tX_{t-1}=φΣX_{t-1}²,两边除以Σ(X_{t-1}²),解得:

φ̂ = Σ(X_t·X_{t-1}) / Σ(X_{t-1}²)

和05类简单线性回归"斜率=Cov(X,Y)/Var(X)"完全同构的公式,只是这里的"X"和"Y"是同一个序列错开一位。

**底层机制/为什么这样设计:** 为什么可以直接套用05类的最小二乘公式,不需要专门为时间序列设计新的估计方法?因为最小二乘法本身只要求"用一组解释变量的线性组合预测一个目标变量,让残差平方和最小",并不要求解释变量和目标变量之间相互独立——AR(1)里的"解释变量"X_{t-1}和"目标变量"X_t虽然来自同一个序列,但在"给定X_{t-1}这一列数据,拟合X_t这一列数据"这个操作层面上,和一个普通的两变量回归没有本质区别,这是为什么时间序列里大量估计方法可以直接借用横截面回归工具的根本原因。

**AI研究/工程场景:** 用AR(1)这类最简单的自回归结构给一个业务指标的短期走势做基线预测,是任何更复杂预测系统(包括深度学习时序模型)几乎必须超越的最低门槛——如果一个复杂的神经网络预测模型的效果还不如一个几行代码就能估计出来的AR(1),说明这个复杂模型大概率哪里出了问题,这是评估任何新预测方法时最基础、最便宜的sanity check。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

n = 8000
phi_true = 0.7
sigma = 1.0
X = np.zeros(n)
for t in range(1, n):
    X[t] = phi_true * X[t - 1] + rng.normal(0, sigma)
X = X[500:]  # 丢弃烧入期

X_t, X_tm1 = X[1:], X[:-1]
phi_hat = np.sum(X_t * X_tm1) / np.sum(X_tm1 ** 2)

# 核心断言1: 最小二乘估计的phi应该非常接近真实值
assert abs(phi_hat - phi_true) < 0.02, f"phi_hat={phi_hat:.4f} should be close to phi_true={phi_true}"

# 用估计出的phi做一步预测, 和"naive baseline"(19类phi=1特例, 20类知识点5会展开)对比
forecast_ar1 = phi_hat * X_tm1
forecast_naive = X_tm1  # 等价于强行令phi=1
rmse_ar1 = np.sqrt(np.mean((X_t - forecast_ar1) ** 2))
rmse_naive = np.sqrt(np.mean((X_t - forecast_naive) ** 2))

# 核心断言2: 真实phi=0.7明显偏离1, 正确估计的AR(1)预测应该比"强行当作phi=1"更准
assert rmse_ar1 < rmse_naive, f"correctly-fit AR(1) should beat the phi=1 special case here: {rmse_ar1:.4f} vs {rmse_naive:.4f}"

print(f"phi_true={phi_true}  phi_hat={phi_hat:.4f}  (least squares, {len(X)} observations)")
print(f"one-step forecast RMSE: AR(1) fit={rmse_ar1:.4f}  vs naive(phi=1 special case)={rmse_naive:.4f}")
print("=> when the true process really mean-reverts (phi far from 1), estimating phi correctly pays off")
```

**面试怎么问+追问链**(规模递增轴+决策依据追问轴):
- Q:"AR(1)的φ怎么估计出来,这个估计量和线性回归有什么关系?"
- 追问1:"这个估计量在小样本下有什么已知问题?"(AR(1)的最小二乘估计在有限样本下存在轻微的向下偏差(bias),真实φ越接近1、样本量越小,这个偏差越明显——这是时间序列估计和标准回归的一个真实区别,标准回归的OLS在经典假设下是无偏的,但自回归模型里"解释变量"本身依赖于历史噪声,严格的无偏性不再成立,不过随样本量增大这个偏差会消失)
- 深挖追问:"如果拟合出来的φ̂非常接近1,应该怎么解读?"(提示序列可能接近19类知识点5讨论的随机游走,均值回归的力量很弱,这时候用AR(1)结构做长期预测要非常谨慎,预测区间会随时间跨度快速增长)

**常见坑:**
- 把AR(1)估计和一般线性回归的适用条件完全等同,忽视自回归模型里"解释变量本身来自同一个随时间演化的过程"这个特殊性(比如φ接近1时估计量的有限样本偏差)。
- 用短短几十个点的数据估计φ就直接下结论——最小二乘估计量的方差随样本量减小而增大,小样本下的φ̂可能离真实值很远,不能只看点估计不看它的不确定性。

---

## 4. 预测评估指标(MAE/RMSE/MAPE) —— 同一组误差,不同的放大镜

**定义与记号:** 给定n个预测误差e_i=X_i-X̂_i(真实值减预测值),三个最常用的汇总指标:MAE(平均绝对误差)=(1/n)Σ\|e_i\|;RMSE(均方根误差)=√((1/n)Σe_i²);MAPE(平均绝对百分比误差)=(1/n)Σ\|e_i/X_i\|×100%。

**一句话:** 这三个指标看的是同一组误差,但"放大镜"完全不同——MAE把每个误差一视同仁地取绝对值平均,RMSE先平方再开方所以对大误差格外敏感,MAPE把误差换算成相对于真实值的百分比所以天然对"真实值本身有多大"敏感,同一个预测结果在这三个指标下可能讲出三个完全不同的故事。

**数学推导:** RMSE≥MAE恒成立,这不是经验规律而是一个可以证明的不等式(幂平均不等式/QM-AM不等式的直接应用):对任意非负数列\|e_1\|,...,\|e_n\|,均方根(二次幂平均)≥算术平均(一次幂平均)恒成立,即√((1/n)Σe_i²)≥(1/n)Σ\|e_i\|,等号成立当且仅当所有\|e_i\|相等(误差大小完全一致,没有任何波动)。

证明只需要用到"平方和不可能是负数"这一个事实。把\|e_1\|,...,\|e_n\|本身看成一组数据,记它们的均值为ē=(1/n)Σ\|e_i\|(这正是MAE的定义),这组数据自己的"方差"必然≥0:

(1/n)Σ(\|e_i\|-ē)² ≥ 0

把左边的平方项展开(逐项按(x-ē)²=x²-2xē+ē²展开再对i求和):

(1/n)Σ(\|e_i\|-ē)² = (1/n)Σ\|e_i\|² - 2ē·(1/n)Σ\|e_i\| + ē²

注意\|e_i\|²=e_i²(平方去掉绝对值)、(1/n)Σ\|e_i\|=ē(均值的定义本身代入),化简右边:

= (1/n)Σe_i² - 2ē² + ē² = (1/n)Σe_i² - ē²

代回"必然≥0"这个事实:(1/n)Σe_i²-ē²≥0,也就是(1/n)Σe_i²≥ē²。两边开方(两边都非负,开方不改变不等号方向):√((1/n)Σe_i²)≥ē,也就是RMSE≥MAE。等号成立当且仅当那个"方差"精确为0,即所有\|e_i\|都等于同一个值ē,和前面陈述的等号条件完全一致。

这意味着RMSE和MAE的**差距大小**本身就携带信息——两者越接近,说明误差分布越均匀;RMSE远大于MAE,直接提示存在个别异常大的误差在拉高RMSE。

**底层机制/为什么这样设计:** 为什么会存在三个指标而不是一个"标准答案"?因为它们对误差分布的不同特征敏感,对应完全不同的业务需求:RMSE的平方项让一个特大误差贡献的惩罚远超其绝对值本身(误差翻倍,平方项的贡献变成4倍),适合"少数极端错误代价特别高"的场景(比如库存预测严重不足导致断货);MAE对每个误差的惩罚和误差大小成正比,不会被少数异常值主导,更适合"想要一个稳健、不被异常值左右的整体误差水平"的场景;MAPE把误差转换成相对百分比,适合需要跨不同量级的多个序列比较误差(不然一个预测销售额的误差100和一个预测点击数的误差100完全不能类比),但代价是真实值接近0时百分比会发散到无穷大,这是它天生的弱点,不是实现问题。

**AI研究/工程场景:** 评测一个预测系统"整体误差有多大"最容易被误用的地方,是不假思索地只报一个MAPE就下结论"误差百分之几,可以接受"——如果预测目标里有一部分真实值本来就很接近0(比如某些低频SKU的销量),这部分贡献的百分比误差会被极度放大,不成比例地拉高整体MAPE,让"实际预测得还不错"的系统看起来误差惊人。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

errors_uniform = rng.normal(0, 1.0, 1000)
errors_outlier = errors_uniform.copy()
errors_outlier[0] = 20.0  # 混入一个真实的异常大误差

def mae(e): return np.mean(np.abs(e))
def rmse(e): return np.sqrt(np.mean(e ** 2))

mae_a, rmse_a = mae(errors_uniform), rmse(errors_uniform)
mae_b, rmse_b = mae(errors_outlier), rmse(errors_outlier)

# 核心断言1: RMSE >= MAE 恒成立(QM-AM不等式)
assert rmse_a >= mae_a and rmse_b >= mae_b, "RMSE should never be smaller than MAE"

# 核心断言2: 混入异常值后, RMSE的相对涨幅应该明显超过MAE的相对涨幅
mae_ratio, rmse_ratio = mae_b / mae_a, rmse_b / rmse_a
assert rmse_ratio > mae_ratio, \
    f"a single large outlier should inflate RMSE more than MAE: rmse_ratio={rmse_ratio:.4f}, mae_ratio={mae_ratio:.4f}"

print(f"no outlier:   MAE={mae_a:.4f}  RMSE={rmse_a:.4f}  (RMSE/MAE={rmse_a/mae_a:.4f})")
print(f"with outlier: MAE={mae_b:.4f}  RMSE={rmse_b:.4f}  (RMSE/MAE={rmse_b/mae_b:.4f})")
print(f"relative inflation from the single outlier: MAE x{mae_ratio:.4f}  RMSE x{rmse_ratio:.4f}")

# MAPE在真实值接近0时发散的具体演示
actual = np.array([100.0, 50.0, 10.0, 1.0, 0.1])
predicted = actual + 1.0  # 每个点的绝对误差完全相同, 都是1
ape_terms = np.abs((actual - predicted) / actual) * 100
mape = np.mean(ape_terms)

# 核心断言3: 同样绝对误差1, 真实值越小, 百分比误差项越大, 最小真实值(0.1)贡献的单项百分比误差应该远超其余各项均值
assert ape_terms[-1] > 5 * np.mean(ape_terms[:-1]), \
    f"the near-zero actual value should dominate MAPE disproportionately: {ape_terms}"

print(f"\nsame absolute error (1.0) at each point, actual values {actual}:")
print(f"per-point APE: {np.round(ape_terms, 1)}%  ->  overall MAPE={mape:.1f}%")
print("=> the single near-zero actual value alone drags MAPE up to a number that misrepresents the other 4 points")
```

**面试怎么问+追问链**(诊断真实数据新题型):
- Q:"你们团队的预测系统上线后MAPE从15%降到8%,是不是说明模型明显变好了?"
- 追问1:"这个结论有什么需要警惕的地方?"(需要先确认这两次评测用的是不是同一批真实值——如果某个周期里低真实值(接近0)的样本占比变了,MAPE的变化可能主要是样本构成变化导致的,不完全是模型能力的提升;应该同时看MAE/RMSE在同一批数据上是否也同步改善,交叉验证结论)
- 深挖追问:"如果MAE和RMSE都在变差,只有MAPE在变好,说明了什么?"(几乎可以确定是评测集里真实值的分布变了(比如低值样本占比下降),而不是模型真的预测得更准了,这时候应该直接怀疑MAPE这个指标本身在这批数据上是否还适用,而不是庆祝模型进步)

**常见坑:**
- 只报一个汇总指标就下结论,不同时检查另外两个指标是否讲述同一个故事——三个指标对误差分布的不同特征敏感,只看一个容易被误导。
- 在真实值可能接近0或者跨越好几个数量级的场景下依然使用MAPE而不做任何调整,不了解它在这种数据分布下的固有缺陷。

---

## 5. Naive baseline陷阱 —— 复杂方法不一定赢

**定义与记号:** Naive baseline(朴素基线):预测明天的值直接等于今天的值,X̂_{t+1}=X_t,不需要拟合任何参数——本质上就是知识点3里φ=1的特例(19类知识点5随机游走的最优一步预测)。

**一句话:** 任何一个花了力气构建的"更复杂"预测方法,上线前都必须真实地和这个几乎不需要任何工作量的朴素基线比一比预测误差——本知识点用真实数值如实展示:在某些数据结构下,复杂方法确实明显更优;但在另一些数据结构下(尤其是高度持续性、接近随机游走的序列),复杂方法反而会**系统性地**输给朴素基线,这是一个真实存在、必须正视的陷阱,不是为了制造悬念而设计的反转。

**数学推导/说明:** 对随机游走X_t=X_{t-1}+ε_t(19类知识点5),给定X_t,下一步的条件期望E[X_{t+1}\|X_t]=E[X_t+ε_{t+1}\|X_t]=X_t——**朴素基线恰好就是均方误差意义下的最优一步预测**,不是凑巧表现还行,而是在这个数据生成过程下理论最优解;任何对历史多个点做平滑的方法(移动平均、指数平滑),本质上是在用"若干步之前的水平"去估计"当前水平",而对随机游走,"当前水平"和"若干步之前的水平"之间隔着若干步全新的、无法预测的随机增量,平滑操作引入的滞后在这里没有换来任何真实的降噪收益(因为根本没有一个稳定不变的"真实水平"可以被平均出来),只有纯粹的负收益。

**底层机制/为什么这样设计:** 为什么知识点1(移动平均)、知识点2(指数平滑)在关于X_t=μ+ε_t(固定水平+噪声)的推导和例子里都明确降低了误差,这里却反过来输给朴素基线?根本原因是这两个知识点的数学推导都建立在"存在一个短期内近似不变的真实水平μ"这个前提上——这个前提对"围绕固定水平波动、有均值回归倾向"的序列(知识点3例子里φ=0.7)成立,但对随机游走(φ=1,没有均值回归的力量)完全不成立,"要不要用平滑方法"本质上取决于数据本身的持续性/均值回归结构,不是"平滑方法天生更先进所以总是更好"这种一概而论的判断。

**AI研究/工程场景:** 时间序列预测的学术论文和产品评测里,一个长期存在、经常被更精细的复杂模型(包括不少深度学习方法)忽视的真实现象是:很多真实业务指标(尤其是短期、高频的指标)在统计性质上非常接近随机游走,这类序列上,复杂模型反复被朴素基线打平甚至打败并不罕见——这正是"用naive baseline做sanity check"在预测类项目里几乎是强制性第一步的根本原因,不做这一步,很容易把"复杂模型其实没有真实提升,只是恰好数据本身好预测"误判成"模型很成功"。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

def gen_ar1(n, phi, sigma=1.0):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, sigma)
    return x

def naive_forecast(x):
    return x[:-1]

def ma_forecast(x, k):
    f = np.full(len(x) - 1, np.nan)
    for t in range(1, len(x)):
        f[t - 1] = np.mean(x[max(0, t - k):t])
    return f

def exp_smooth_forecast(x, alpha):
    s = np.zeros(len(x))
    s[0] = x[0]
    for t in range(1, len(x)):
        s[t] = alpha * x[t] + (1 - alpha) * s[t - 1]
    return s[:-1]

def rmse(actual, forecast):
    return np.sqrt(np.mean((actual - forecast) ** 2))

n = 1200
# 场景A: 弱持续性, 强均值回归(phi=0.2, 更接近知识点1/2例子里"固定水平+噪声"的假设)
xA = gen_ar1(n, phi=0.2, sigma=1.0)[200:]
# 场景B: 随机游走(phi=1.0, 19类知识点5)
xB = gen_ar1(n, phi=1.0, sigma=1.0)[200:]

actualA, actualB = xA[1:], xB[1:]
methods = {"naive": naive_forecast, "ma5": lambda x: ma_forecast(x, 5), "ema0.3": lambda x: exp_smooth_forecast(x, 0.3)}

resA = {name: rmse(actualA, fn(xA)) for name, fn in methods.items()}
resB = {name: rmse(actualB, fn(xB)) for name, fn in methods.items()}

# 核心断言1: 场景A(弱持续性), 两种平滑方法都应该真实地比naive更准
assert resA["ma5"] < resA["naive"], f"scenario A: MA(5) should beat naive here: {resA}"
assert resA["ema0.3"] < resA["naive"], f"scenario A: EMA should beat naive here: {resA}"

# 核心断言2(如实展示陷阱): 场景B(随机游走), naive反而应该比两种平滑方法都更准, 不回避这个结果
assert resB["naive"] < resB["ma5"], f"scenario B: naive should beat MA(5) here: {resB}"
assert resB["naive"] < resB["ema0.3"], f"scenario B: naive should beat EMA here: {resB}"

print(f"scenario A (phi=0.2, mean-reverting):  naive={resA['naive']:.4f}  ma5={resA['ma5']:.4f}  ema0.3={resA['ema0.3']:.4f}")
print(f"  -> smoothing wins here: there really is a stable short-term level to average out noise around")
print(f"scenario B (phi=1.0, random walk):     naive={resB['naive']:.4f}  ma5={resB['ma5']:.4f}  ema0.3={resB['ema0.3']:.4f}")
print(f"  -> naive wins here: there is no stable level, smoothing only adds lag with no denoising payoff")
```

**面试怎么问+追问链**(真实性验证轴+方案批判迭代轴):
- Q:"你说你们的预测模型比简单方法'提升了20%',这个'简单方法'具体是什么,怎么比的?"
- 追问1:"如果对照组选的是'预测=昨天的值'这种最基础的baseline,20%提升说明什么,不说明什么?"(说明模型确实提取出了一些朴素基线没有捕捉到的可预测结构,但具体提升幅度是否"足够好"取决于这个业务指标本身的可预测性上限——同样是20%提升,在一个高度随机游走型的指标上做到可能已经接近理论上限,在一个强周期性、强规律性的指标上做到可能远远不够)
- 深挖追问:"如果模型在验证集上就没有稳定跑赢naive baseline,该怎么办,是不是说明模型本身有bug?"(不一定是bug——先要检查这个指标本身是否接近随机游走结构(可以用19类的ACF/知识点3的φ̂估计值判断,φ̂接近1就是明确信号),如果确实如此,"跑不赢naive"可能是数据本身的真实限制,而不是模型实现有问题,这时候盲目调参、换更复杂的模型架构去"死磕"这个baseline,往往是在浪费工程资源)

**常见坑:**
- 只在一种数据结构下验证过"复杂方法比naive baseline好",就把这个结论当作普遍真理应用到所有预测任务上——本知识点的例子已经明确展示,这个结论完全依赖数据本身的持续性结构,不是方法本身的固有属性。
- 看到复杂模型跑不赢naive baseline就本能地认为"一定是模型不够好/参数没调好",不优先检查"这个指标本身是不是就很接近随机游走,理论上限就在这附近"这个更根本的可能性。

---

板块V(时间序列基础,19-20类,共11个知识点)到此收官,统计学系列20个分类文件全部完成。下一篇:[21-mock-interview-capstone.md](21-mock-interview-capstone.md) —— 模拟终面capstone,把全系列(01-20类)知识点整合到一次连续追问场景里,用真实日志诊断收尾。
