# 15 · 排位系统深挖(Ranking Systems)

> 总览见 [00-roadmap.md](00-roadmap.md)

14类处理的是"给单个模型打一个分数"这个问题;本文处理"给一批模型排出相对高下"这个更贴近LLM Arena、游戏匹配、竞技体育排名的场景。核心工具是Bradley-Terry模型和它的在线学习版本Elo评分——这两者不是两套独立的理论,而是同一个优化目标的两种求解方式,`learning/llm-judge-arena/src/bradley_terry.py`已经用真实代码实现了两者的转换关系,本文在此基础上补充数学推导、K因子调优、以及贝叶斯化的TrueSkill思想。

**环境声明:** 全部代码在仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1)下真实跑通,随机抽样固定种子(`np.random.default_rng(42)`)。知识点4已实读`learning/llm-judge-arena/src/bradley_terry.py`源码(MM算法实现、`to_elo()`转换公式、`_self_test()`构造),引用的所有实现细节均已核实。

---

## 1. Bradley-Terry模型 —— 从一堆胜负记录到一个实力排序

**定义与记号:** Bradley-Terry模型:为N个待比较的对象各自估计一个"实力值"(log-strength参数化r_i),模型假设"i胜过j"的概率只取决于两者实力值之比:P(i beats j)=exp(r_i)/(exp(r_i)+exp(r_j))。给定一批两两对战的结果数据,通过最大似然估计(MLE)反推出每个对象的实力值。

**一句话:** Bradley-Terry模型把"一堆胜负记录"变成"每个选手一个实力分数"这件事的核心假设极其简洁——胜率完全由实力差决定,不存在"克制关系"(A打B必胜、B打C必胜,但C打A却能赢)这种非传递性结构,这个假设是否成立本身也是一个可以被检验的问题。

**数学推导:** 对数似然函数 ℓ(r)=Σ_{(i,j)} [wins_ij·log σ(r_i-r_j)](σ是sigmoid函数;这个形式和逻辑回归的对数似然完全同构,05类知识点8已经推导过Newton-Raphson求解逻辑回归的方法)。对r_i求偏导:∂ℓ/∂r_i=Σ_j [wins_ij - n_ij·σ(r_i-r_j)](n_ij是i、j的总对战次数),令偏导为0即MLE条件——i的胜场数应该恰好等于模型预测的期望胜场数,这是MLE"矩匹配"直觉在这个模型里的具体体现。

**底层机制/为什么这样设计:** 为什么Bradley-Terry模型只用一个标量r_i就能刻画"实力",而不需要每一对i,j单独估计一个胜率?因为"每一对单独估计"这种做法在对象数量增多时需要估计的参数数量是O(N²),数据需求会随对象数量平方增长;Bradley-Terry模型假设"实力"是一个可以被压缩到单一标量的可传递属性,把参数量降到O(N),用可传递性这个假设换取参数效率——这个假设在真实场景下不一定完全成立(比如游戏里确实存在"克制链"这种非传递结构),使用这个模型隐含地接受了这个简化。

**AI研究/工程场景:** `learning/llm-judge-arena/src/bradley_terry.py`(L04)是这个模型在Chatbot Arena场景下的真实工程实现——多个LLM两两对战(由用户或judge模型裁定胜负),用Bradley-Terry模型把海量的pairwise对战记录汇总成每个模型的一个实力分数,是LMSYS Chatbot Arena排行榜背后的真实统计机制。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

true_strengths = {"A": 2.0, "B": 1.0, "C": 0.0, "D": -1.5}
models = list(true_strengths.keys())

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 生成成对比较数据: 每一对模型互相对战n_battles_per_pair次
n_battles_per_pair = 40
battles = []
for a in range(len(models)):
    for b in range(a + 1, len(models)):
        i, j = models[a], models[b]
        p_i_wins = sigmoid(true_strengths[i] - true_strengths[j])
        wins_i = rng.binomial(n_battles_per_pair, p_i_wins)
        battles.append((i, j, wins_i, n_battles_per_pair))

# 手写梯度上升求MLE
r = {m: 0.0 for m in models}
lr = 0.02
for _ in range(3000):
    grad = {m: 0.0 for m in models}
    for i, j, wins_i, n_ij in battles:
        p_pred = sigmoid(r[i] - r[j])
        g = wins_i - n_ij * p_pred  # d(loglik)/dr_i
        grad[i] += g
        grad[j] -= g
    for m in models:
        r[m] += lr * grad[m]

mean_r = np.mean(list(r.values()))
r = {m: v - mean_r for m, v in r.items()}

# 核心断言1: 估计出的实力排序应该和真实排序完全一致
est_order = sorted(models, key=lambda m: -r[m])
true_order = sorted(models, key=lambda m: -true_strengths[m])
assert est_order == true_order, f"estimated order {est_order} should match true order {true_order}"

# 核心断言2: 估计值应该在合理误差范围内接近真实值(中心化后)
true_centered = {m: v - np.mean(list(true_strengths.values())) for m, v in true_strengths.items()}
max_err = max(abs(r[m] - true_centered[m]) for m in models)
assert max_err < 0.4, f"MLE estimates should be reasonably close to the true (centered) strengths, max error={max_err:.4f}"

print(f"true strengths (centered): { {m: round(v,3) for m,v in true_centered.items()} }")
print(f"MLE estimates:             { {m: round(v,3) for m,v in r.items()} }")
print(f"estimated order = {est_order}  (matches true order)")
```

上面这段代码真实跑出的6对两两胜负记录(win_i / n_battles_per_pair,同一次运行,数字完全对应):

```
A vs B: 27/40 (67.5%)   A vs C: 36/40 (90.0%)   A vs D: 38/40 (95.0%)
B vs C: 28/40 (70.0%)   B vs D: 39/40 (97.5%)   C vs D: 28/40 (70.0%)
```

这是Bradley-Terry要解决的核心可读性问题:6个两两独立的百分比,没法直接回答"A比B领先多少、这个领先幅度和B比C领先的幅度谁大谁小"——因为它们不在同一个可比较的刻度上。MLE拟合把这6个数字压缩成4个可以直接相减、直接比大小的实力值r(下面的柱状图长度∝这次真实运行里r与最弱者D的差距,不是编造数据):

```
D  -1.65 |                                (最弱, 作为基准)
C  -0.53 |###########                     (比D强 1.12)
B   0.70 |#######################         (比C再强 1.23)
A   1.48 |###############################  (比B再强 0.78)
```

这就是"从一堆胜负记录到一个实力排序"的直观样子:原始数据是N(N-1)/2=6条两两战绩,MLE输出的是N=4个点在同一条数轴上的位置,排序和差距现在可以直接读出来,不需要在6条战绩之间来回换算。

**面试怎么问+追问链**(决策依据追问轴):
- Q:"Bradley-Terry模型假设'胜率只取决于实力差',这个假设在什么场景下会失效?"
- 追问1:"能举一个'非传递性'(克制关系)的具体例子吗?"(石头剪刀布式的结构——A能稳定打赢B,B能稳定打赢C,但C反而能打赢A,这种循环克制关系直接违反了模型"存在一个可传递的实力排序"这个前提假设,拟合出来的实力值会是一种失真的近似)
- 深挖追问:"如果怀疑数据里存在非传递性结构,有什么办法可以检验?"(可以检查拟合出的BT模型对观测胜负的预测准确率是否明显偏低;也可以直接在数据里搜索"A胜B、B胜C、C胜A"这种循环三元组的出现频率是否显著高于纯粹随机噪声下的预期频率)

**常见坑:**
- 把Bradley-Terry估计出的实力值差异不加区分地当作"确定性事实",忽视了这些估计值本身也是从有限对战数据里估计出来的、带有抽样不确定性的统计量(呼应14类知识点1"评测分数也有抽样误差"的同一原则)。
- 忽视Bradley-Terry模型的可传递性假设可能不成立,在真实存在非传递结构的场景下依然直接套用模型、不做任何拟合优度检验。

---

## 2. Elo评分系统 —— Bradley-Terry的在线学习版本

**定义与记号:** Elo评分系统:每次对战后,根据"实际结果"和"模型预期结果"之差,增量式更新双方评分:R_i'=R_i+K·(S_i-E_i),S_i是实际得分(赢1、输0、平0.5),E_i是预期得分(1/(1+10^((R_j-R_i)/400))),K是控制更新步长的"K因子"。

**一句话:** Elo可以理解成Bradley-Terry模型的"在线学习"版本——不是一次性用全部历史对战数据做批量MLE,而是每来一场新对战就用一个简单的更新公式微调评分,天然适合"选手在不断新增对局、需要持续更新排名"这种流式场景。

**数学推导:** Elo的预期胜率公式 E_i=1/(1+10^((R_j-R_i)/400)) 和Bradley-Terry的胜率公式经过换底代换后完全等价——令R=1500+400/ln(10)·r(这正是`bradley_terry.py`里`to_elo()`函数用的转换公式:`base=1500, scale=400/math.log(10)`),可以验证10^((R_j-R_i)/400)=exp(r_j-r_i),代入后两个公式代数恒等。Elo更新公式本质上是对Bradley-Terry对数似然函数做**随机梯度上升**(每次只用一场对局的梯度更新一次参数,而不是用全部数据的梯度)。

**底层机制/为什么这样设计:** 为什么Elo这种"看一场对局更新一次"的在线算法,和Bradley-Terry这种"看完所有数据做一次批量优化"的方法,收敛后能给出等价的结果?这是机器学习里"批量梯度下降 vs 随机梯度下降"这对经典关系在评分系统场景下的具体体现——两者优化的是同一个目标函数,只是更新数据的粒度和频率不同,收敛点在温和的条件下应该趋于一致。

**AI研究/工程场景:** 国际象棋评分系统(Elo最初的应用场景)、`learning/llm-judge-arena/`的Chatbot Arena排行榜都用类似机制——Elo的在线特性使其特别适合"新模型不断加入、新对战不断产生"的持续更新场景,不需要每次都重新对全部历史数据做批量MLE,这是它相对纯批量Bradley-Terry方法在工程实现上的实际优势。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

true_strengths = {"A": 2.0, "B": 1.0, "C": 0.0, "D": -1.5}
models = list(true_strengths.keys())

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 生成随机顺序的对局序列(模拟真实比赛依次发生)
n_rounds = 500
match_log = []
for _ in range(n_rounds):
    i, j = rng.choice(models, 2, replace=False)
    p_i_wins = sigmoid(true_strengths[i] - true_strengths[j])
    winner = i if rng.random() < p_i_wins else j
    match_log.append((i, j, winner))

# Elo在线更新
elo = {m: 1500.0 for m in models}
K = 32
scale = 400 / np.log(10)
for i, j, winner in match_log:
    e_i = 1 / (1 + 10 ** ((elo[j] - elo[i]) / 400))
    s_i = 1.0 if winner == i else 0.0
    delta = K * (s_i - e_i)
    elo[i] += delta
    elo[j] -= delta

# Bradley-Terry批量MLE, 用同一份match_log数据
battles = {}
for i, j, winner in match_log:
    key = tuple(sorted([i, j]))
    battles.setdefault(key, {key[0]: 0, key[1]: 0})
    battles[key][winner] += 1

r = {m: 0.0 for m in models}
lr = 0.02
for _ in range(3000):
    grad = {m: 0.0 for m in models}
    for (i, j), wins in battles.items():
        n_ij = wins[i] + wins[j]
        p_pred = sigmoid(r[i] - r[j])
        g = wins[i] - n_ij * p_pred
        grad[i] += g
        grad[j] -= g
    for m in models:
        r[m] += lr * grad[m]
mean_r = np.mean(list(r.values()))
r = {m: v - mean_r for m, v in r.items()}
bt_elo = {m: 1500 + scale * v for m, v in r.items()}

# 核心断言1: 两种方法给出的排序应该完全一致
elo_order = sorted(models, key=lambda m: -elo[m])
bt_order = sorted(models, key=lambda m: -bt_elo[m])
assert elo_order == bt_order, f"online Elo order {elo_order} should match batch BT order {bt_order}"

# 核心断言2: 转换到同一刻度后, 两者数值应该高度接近(不要求完全相等, 在线vs批量本身有估计噪声)
max_diff = max(abs(elo[m] - bt_elo[m]) for m in models)
assert max_diff < 50, f"online Elo and batch BT-derived Elo should be numerically close, max diff={max_diff:.2f}"

print(f"Elo (online update):      { {m: round(v) for m,v in elo.items()} }")
print(f"BT-derived Elo (batch MLE): { {m: round(v) for m,v in bt_elo.items()} }")
print(f"orders match: {elo_order == bt_order}, max value diff = {max_diff:.2f}")
```

**面试怎么问+追问链**(工程约束递增轴):
- Q:"既然Elo和Bradley-Terry本质上是等价的,为什么实际系统两者都会涉及,不是只选一个?"
- 追问1:"Elo相对批量Bradley-Terry MLE,在工程实现上有什么具体优势?"(Elo是增量式更新,新增一场对战只需要O(1)的更新计算,不需要重新处理全部历史数据;批量MLE每次都需要基于全部历史对战数据重新迭代收敛,新增少量数据后要不要重新跑一次全量MLE是一个需要权衡的工程决策)
- 深挖追问:"Elo的'在线更新'这个特性,会不会带来批量方法没有的问题?"(会——Elo的评分结果依赖对战出现的**顺序**(K因子较大时尤其明显),而批量MLE在同样的收敛标准下通常和数据顺序无关,这是在线算法相对批量算法的一个常见代价,知识点3会更具体展开这个"顺序依赖"和K因子的关系)

**常见坑:**
- 认为Elo和Bradley-Terry是两种"不同的排名理论",而不理解两者数学上高度相关(Elo可以理解为BT对数似然的随机梯度上升)——这是面试时容易在"这两者关系是什么"这类追问上卡壳的地方。
- 忽视Elo评分的绝对数值本身没有绝对意义(1500这个基准值是人为设定的),只有Elo评分**之差**才直接对应胜率预测,单独看一个孤立的Elo数字意义有限。

---

## 3. Elo的K因子与收敛速度 —— 快速适应 vs 抗噪稳定,不可兼得

**定义与记号:** K因子:Elo更新公式里的步长参数,直接控制"单场对局的结果对评分调整幅度有多大"。K越大,评分对新信息反应越快,但也越容易被单场偶然的意外结果大幅拉动,评分不够稳定;K越小,评分变化越平缓、越抗噪声,但适应"选手真实实力发生变化"会更慢。

**一句话:** K因子是Elo系统里"信任新数据 vs 信任历史积累"这个经典权衡的具体化身——大K意味着更相信这一场比赛(方差大,但能快速跟上真实变化),小K意味着更相信过去积累的评分(方差小,但反应迟钝)。

**数学推导/说明:** 用数值模拟量化这个权衡:①收敛速度——从相同初始评分出发,测量评分达到接近真实差距所需要的对局数,K越大所需局数通常越少;②稳定性——评分收敛之后,继续输入更多对战数据,测量评分本身的方差如何随K变化,K越大波动幅度通常越大。这两个指标随K的变化方向相反,不存在一个"K越大越好"或"K越小越好"的单向结论。

**底层机制/为什么这样设计:** 为什么不存在一个"放之四海而皆准"的最优K值?因为"应该多快适应新信息"这个问题的答案,本质上取决于"真实实力本身变化的速度"和"单场比赛结果的噪声水平"这两个因素的相对大小——如果选手实力相对稳定,应该用较小的K;如果选手实力变化较快(比如LLM模型频繁迭代升级),需要用较大的K——这是信号(真实实力变化)和噪声(单场结果随机性)的相对强度决定最优响应速度的经典统计问题结构。

**AI研究/工程场景:** 国际象棋官方Elo系统对不同水平的选手实际会用不同的K因子(新手用更大的K,让评分快速反映真实水平;顶级选手用更小的K,减少评分因个别冷门比赛而大幅波动)——这个"分层设置K因子"的实践,正是本知识点"收敛速度vs稳定性权衡,没有单一最优值"这个结论在真实评分系统设计里的具体应用。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

true_r_A, true_r_B = 1.5, 0.0
p_A_wins = sigmoid(true_r_A - true_r_B)

def run_elo_once(K, n_matches, rng):
    elo_A, elo_B = 1500.0, 1500.0
    history = np.zeros(n_matches)
    for t in range(n_matches):
        e_A = 1 / (1 + 10 ** ((elo_B - elo_A) / 400))
        s_A = 1.0 if rng.random() < p_A_wins else 0.0
        delta = K * (s_A - e_A)
        elo_A += delta
        elo_B -= delta
        history[t] = elo_A - elo_B
    return history

n_repeats = 200
K_values = [8, 32, 64, 128]
avg_early_diff, avg_tail_std = {}, {}
for K in K_values:
    early_diffs, tail_stds = [], []
    for _ in range(n_repeats):
        hist = run_elo_once(K, 300, rng)
        early_diffs.append(hist[:20].mean())   # 前20局的平均差距: 越大说明适应越快
        tail_stds.append(hist[-100:].std())     # 收敛后(后100局)的标准差: 越大说明越不稳定
    avg_early_diff[K] = np.mean(early_diffs)
    avg_tail_std[K] = np.mean(tail_stds)

# 核心断言1: K越大, 早期评分差距发展得越快(适应更快)
early_vals = [avg_early_diff[K] for K in K_values]
assert all(early_vals[i] < early_vals[i + 1] for i in range(len(early_vals) - 1)), \
    f"larger K should adapt faster (larger early gap): {avg_early_diff}"

# 核心断言2: K越大, 收敛后的评分波动越大(越不稳定) -- 和断言1方向相反的两个指标
tail_vals = [avg_tail_std[K] for K in K_values]
assert all(tail_vals[i] < tail_vals[i + 1] for i in range(len(tail_vals) - 1)), \
    f"larger K should also be less stable after convergence: {avg_tail_std}"

print(f"avg early gap (first 20 matches) by K: { {k: round(v,1) for k,v in avg_early_diff.items()} }")
print(f"avg tail std (last 100 matches) by K:  { {k: round(v,1) for k,v in avg_tail_std.items()} }")
print("=> larger K adapts faster BUT is less stable -- a genuine tradeoff, not a free lunch")
```

**面试怎么问+追问链**(方案批判迭代轴):
- Q:"系统里模型的Elo评分波动很大,业务方投诉'这个模型排名一会儿高一会儿低,不稳定',你会怎么调整?"
- 追问1:"调小K因子能解决这个问题吗?"(能减少波动幅度,但会带来另一个代价——如果这个模型确实经历了真实的能力变化(比如更新了新版本),调小K之后评分系统会需要更长时间才能反映出这个真实变化)
- 深挖追问:"有没有办法同时兼顾'稳定'和'适应变化快'这两个目标,而不是死板地选一个固定K值?"(可以考虑动态K因子——对战局数较少的新选手用较大的K快速建立初步评分,随着对局数增加逐渐减小K;更复杂的做法(比如Glicko系统、知识点5的TrueSkill)进一步把"这个评分有多可信"也变成一个显式估计的量,而不是只靠调K因子隐式权衡)

**常见坑:**
- 把K因子当作一个"调一次就永久生效"的全局常数,不考虑针对不同场景(新选手/老选手、稳定项目/快速迭代项目)动态调整。
- 只关注K因子对"收敛速度"的影响,忽视它同时也在影响"收敛后的稳定性",单方面调大K因子追求"评分能快速反映最新表现",可能引入了原本可以避免的额外波动噪声。

---

## 4. 呼应llm-judge-arena —— MM算法与"完全可分"数据的数值退化

**定义与记号:** `learning/llm-judge-arena/src/bradley_terry.py`(已读取源码核实)用MM算法(Minorization-Maximization,Hunter 2004)求解Bradley-Terry MLE,更新公式:s_i ← W_i / Σ_j(n_ij/(s_i+s_j))(s_i=exp(r_i),W_i是i的总胜场数)——和知识点1采用的梯度上升是求解同一个优化目标的不同算法。

**一句话:** 同一个优化问题(Bradley-Terry MLE)可以用不同的迭代算法求解,MM算法的核心思想是"构造一个更容易优化、且在当前点和原目标函数相切的替代函数,反复优化这个替代函数来逼近原问题的最优解"——不需要计算梯度或Hessian矩阵,只需要一个简单的乘除法更新公式,也不需要设置学习率。

**数学推导:** MM算法的核心理论保证是**单调性**:每次更新后,似然函数的值不会减小——这是"优化替代函数"这个操作等价于"保证原函数单调不减"的直接后果,不需要额外的步长/学习率调参,这是它比梯度上升更容易稳定收敛的原因(梯度上升如果学习率设置不当可能发散或震荡,MM算法的更新公式天然不需要这个超参数)。

**底层机制/为什么这样设计:** 为什么这个仓库选择MM算法而不是本文知识点1教学时更常用的梯度法?从数值优化角度看,MM算法对"胜率类"MLE问题有天然适配性:更新公式不涉及任何需要调节的学习率超参数,每次迭代的计算量也很小,在选手/模型数量不太大的场景下(Chatbot Arena的几十到上百个模型)收敛速度足够快、实现足够简单——这体现了"同一个数学问题,不同求解算法各有各的工程权衡"这个更普遍的道理,不是"哪个算法绝对更好",而是"哪个算法更适配当前的规模和实现复杂度约束"。

**AI研究/工程场景:** `bradley_terry.py`的`_self_test()`构造了一个"A必胜B、B必胜C"的极端确定性场景(20次重复,没有任何随机噪声),README明确提示"mini_arena.py的Elo数值会很极端,因为toy数据接近完全可分——真实系统需要更多噪声、更多votes和置信区间"——这个坦诚的自我提示恰好呼应14类知识点1的核心教训(单个评分不该被当作没有不确定性的精确数字):即使算法实现正确,如果数据本身"完全可分",估计出的实力差距也会异常且不稳定。

**可运行例子:**
```python
import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mm_fit(models, W, n_iter=200):
    """与 bradley_terry.py 完全同构的MM算法实现"""
    log_s = {m: 0.0 for m in models}
    likelihoods = []
    for _ in range(n_iter):
        s = {m: math.exp(log_s[m]) for m in models}
        new_s = {}
        for i in models:
            W_i = sum(W.get((i, j), 0) for j in models if j != i)
            denom = 0.0
            for j in models:
                if i == j:
                    continue
                n_ij = W.get((i, j), 0) + W.get((j, i), 0)
                if n_ij > 0:
                    denom += n_ij / (s[i] + s[j])
            new_s[i] = (W_i / denom) if denom > 0 else s[i]
        log_s = {m: math.log(max(1e-9, new_s[m])) for m in models}
        mean = sum(log_s.values()) / len(log_s)
        log_s = {m: v - mean for m, v in log_s.items()}
        ll = 0.0
        for i in models:
            for j in models:
                if i >= j:
                    continue
                n_ij = W.get((i, j), 0) + W.get((j, i), 0)
                if n_ij == 0:
                    continue
                p_pred = sigmoid(log_s[i] - log_s[j])
                ll += W.get((i, j), 0) * np.log(max(p_pred, 1e-12)) + W.get((j, i), 0) * np.log(max(1 - p_pred, 1e-12))
        likelihoods.append(ll)
    return log_s, likelihoods

rng = np.random.default_rng(42)
scale = 400 / np.log(10)

# 场景A: 正常带噪声的数据
models = ["A", "B", "C"]
true_r = {"A": 1.5, "B": 0.5, "C": -1.0}
W_noisy = {}
for i, j in [("A", "B"), ("B", "C"), ("A", "C")]:
    p = sigmoid(true_r[i] - true_r[j])
    wi = rng.binomial(20, p)
    W_noisy[(i, j)] = wi
    W_noisy[(j, i)] = 20 - wi
r_noisy, likelihoods_noisy = mm_fit(models, W_noisy)
elo_noisy = {m: 1500 + scale * v for m, v in r_noisy.items()}
gap_noisy = elo_noisy["A"] - elo_noisy["C"]

# 核心断言1: MM算法的似然值应该单调不减(理论核心保证的直接数值验证)
diffs = np.diff(likelihoods_noisy)
assert np.all(diffs > -1e-6), f"MM algorithm's likelihood should be non-decreasing at every iteration, min diff={diffs.min():.2e}"

# 场景B: 完全可分数据(和bradley_terry.py的_self_test完全一致的构造: A全胜B, B全胜C)
W_separable = {("A", "B"): 20, ("B", "A"): 0, ("B", "C"): 20, ("C", "B"): 0, ("A", "C"): 0, ("C", "A"): 0}
r_separable, _ = mm_fit(models, W_separable)
elo_separable = {m: 1500 + scale * v for m, v in r_separable.items()}
gap_separable = elo_separable["A"] - elo_separable["C"]

# 核心断言2: 完全可分数据下, 实力差距应该异常巨大, 远超正常带噪声数据的差距(数值复现README的警告)
assert gap_separable > 2000, f"perfectly separable data should produce an extreme Elo gap, got {gap_separable:.1f}"
assert gap_noisy < 500, f"noisy realistic data should produce a much more modest gap, got {gap_noisy:.1f}"
assert gap_separable > gap_noisy * 5, \
    f"the separable-data gap should dwarf the noisy-data gap: separable={gap_separable:.1f} noisy={gap_noisy:.1f}"

print(f"noisy data: Elo gap(A-C) = {gap_noisy:.1f}  (realistic magnitude)")
print(f"perfectly separable data: Elo gap(A-C) = {gap_separable:.1f}  (extreme -- confirms the repo README's warning)")
print("=> 'toy data being perfectly separable' is a real numerical degeneracy, not just a theoretical concern")
```

**面试怎么问+追问链**(真实性验证轴):
- Q:"简历上写'实现了Bradley-Terry模型给模型排位',面试官追问'用的什么优化算法',你会怎么回答?"
- 追问1:"如果只会说'用梯度下降'算不算及格?"(及格但不够深入——更完整的回答应该能说清楚为什么选择这个算法,以及这个选择相对其他算法的具体权衡,而不是只知道"能跑起来"这一个事实)
- 深挖追问:"如果数据里出现了'完全可分'(某个选手对所有对手全胜)这种极端情况,你的实现会发生什么?"(实力值的MLE估计理论上会趋向无穷大,这个仓库的实现用`max(1e-9, ...)`做了数值下限保护、避免log(0)崩溃,但即使不崩溃,估计出的数值也会异常巨大且不稳定;严肃的生产系统需要额外的正则化(比如给实力值加一个先验,回到13类贝叶斯方法的思路)来避免这个退化)

**常见坑:**
- 只关注"排名结果对不对",不关心背后用的是哪种优化算法、这个算法有什么数值稳定性方面的已知局限(比如完全可分数据下的发散问题)。
- 把这个仓库的toy实现(明确标注为教学/demo性质,数据是mock的、没有连接真实judge API)误当作生产级别的实现直接照搬使用,忽视README里"真实系统需要更多噪声、更多votes和置信区间"这条明确的自我提醒。

---

## 5. TrueSkill简介 —— 用一个分布,而不是一个数字,刻画实力

**定义与记号:** TrueSkill(微软研究院提出,用于Xbox Live游戏匹配):Bradley-Terry/Elo的一个重要扩展——不再把每个选手的实力表示成一个单一点估计,而是表示成一个概率分布(通常是高斯N(μ,σ²)),μ是当前对实力的最佳估计(和Elo评分类似),σ是对这个估计的不确定性程度的刻画。本知识点只演示"用高斯分布刻画实力不确定性"这个核心思想的最简化版本,不要求完整实现TrueSkill的因子图/期望传播算法。

**一句话:** Elo只回答"这个选手现在的实力评分是多少";TrueSkill进一步回答"这个评分我们有多大把握"——一个打了1000场比赛、评分稳定在1800的老将,和一个只打了3场比赛、评分恰好也是1800的新秀,两者的"1800"含义完全不同,TrueSkill用显式的不确定性刻画把这个区别表达出来,这正是11-13类贝叶斯方法论"用完整分布而不是单一点估计刻画信念"这个思想在排位系统场景下的直接应用。

**数学推导/说明:** 用11类知识点2已建立的Normal-Normal贝叶斯更新机制做简化类比:把每个选手的实力θ看作一个待估计的正态分布参数,每场对局的表现值看作一次"观测",随着对局数增多,后验分布的方差σ²应该按照标准贝叶斯更新的规律收缩(11类知识点5已验证过的"数据越多、不确定性收缩越快"规律)。数据量少、先验又宽的新选手,同样场次的对局能让σ收缩得多;数据量已经很多、先验已经很窄的老选手,同样场次带来的σ收缩幅度小得多——这不是TrueSkill算法本身的数值(真实TrueSkill用的是因子图上的期望传播,不是简单的Normal-Normal更新),而是这个核心现象的简化演示。

**底层机制/为什么这样设计:** 为什么要显式建模"评分的不确定性",而不是像Elo那样只给一个点数?因为"不确定性"这个信息本身在很多实际决策场景里有直接价值——游戏匹配系统想要把"实力相近"的玩家配对在一起,如果只看Elo点数,两个"评分都是1800但一个打了1000局很确定、一个打了3局很不确定"的玩家会被系统认为"实力相近该配对",但实际上后者的真实实力可能在很宽的范围内波动;显式的不确定性刻画能让匹配系统更聪明地处理这种情形。

**AI研究/工程场景:** 这个"点估计+不确定性"的建模思路在评估新上线的AI模型/新版本judge时同样适用——一个刚上线、只有少量对战/评测数据的新模型,不应该仅凭几场对局的结果就被赋予一个和成熟模型同等"确定性"的排名位置,理想的排位系统应该同时展示"这个模型的排名"和"这个排名有多大把握",呼应14类知识点1"单个评测分数会骗人"以及知识点4"完全可分小数据导致数值退化"两个陷阱在这里的自然延伸。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

true_skill_new = 1500.0
true_skill_old = 1500.0  # 假设两人真实实力相同, 只是历史数据量不同, 专门对比不确定性收敛速度
sigma_perf = 200.0  # 单局表现值的噪声标准差

mu_new, var_new = 1500.0, 350.0 ** 2  # 新选手: 先验很宽(几乎不知道)
mu_old, var_old = 1500.0, 50.0 ** 2   # 老选手: 先验已经很窄(打过大量历史对局)

n_new_matches = 10
for _ in range(n_new_matches):
    obs = true_skill_new + rng.normal(0, sigma_perf)
    var_new_post = 1 / (1 / var_new + 1 / sigma_perf ** 2)
    mu_new_post = var_new_post * (mu_new / var_new + obs / sigma_perf ** 2)
    mu_new, var_new = mu_new_post, var_new_post

for _ in range(n_new_matches):
    obs = true_skill_old + rng.normal(0, sigma_perf)
    var_old_post = 1 / (1 / var_old + 1 / sigma_perf ** 2)
    mu_old_post = var_old_post * (mu_old / var_old + obs / sigma_perf ** 2)
    mu_old, var_old = mu_old_post, var_old_post

sigma_new_final, sigma_old_final = np.sqrt(var_new), np.sqrt(var_old)
new_reduction_ratio = sigma_new_final / 350.0
old_reduction_ratio = sigma_old_final / 50.0

# 核心断言1: 同样打了10场, 新选手的不确定性收缩幅度应该远大于老选手
assert new_reduction_ratio < 0.3, f"a new player's uncertainty should shrink dramatically after 10 matches, ratio={new_reduction_ratio:.4f}"
assert old_reduction_ratio > 0.6, f"a veteran's uncertainty (already tight) should shrink only modestly, ratio={old_reduction_ratio:.4f}"
assert new_reduction_ratio < old_reduction_ratio, \
    f"the new player's relative uncertainty reduction should be much larger: new={new_reduction_ratio:.4f} old={old_reduction_ratio:.4f}"

# 核心断言2: 两者的mu(点估计)应该数值上比较接近(真实实力相同), 但sigma差异巨大 -- 只看mu看不出这个区别
assert abs(mu_new - mu_old) < 150, "the point estimates alone don't reveal the confidence difference"
assert sigma_new_final > sigma_old_final * 1.3, \
    f"despite similar mu, sigma should differ substantially: new_sigma={sigma_new_final:.1f} old_sigma={sigma_old_final:.1f}"

print(f"new player (10 matches, wide prior): mu={mu_new:.1f}  sigma={sigma_new_final:.1f}  (shrunk to {new_reduction_ratio:.1%} of prior)")
print(f"old player (10 matches, tight prior): mu={mu_old:.1f}  sigma={sigma_old_final:.1f}  (shrunk to {old_reduction_ratio:.1%} of prior)")
print("=> same mu doesn't mean same confidence -- sigma carries information mu alone cannot express")
```

**面试怎么问+追问链**(规模递增轴):
- Q:"游戏匹配系统只用Elo分数配对玩家,有什么潜在问题?"
- 追问1:"具体会在什么场景下出问题?"(一个刚注册、只打了几局的新玩家,即使Elo分数恰好和一个身经百战的老玩家相同,两者的真实实力置信范围完全不同——如果匹配系统只看分数数字就认为两者"实力相近"该配对,可能导致对局体验严重失衡)
- 深挖追问:"TrueSkill这种显式建模不确定性的方法,相对单纯调大新玩家的K因子(知识点3的做法),有什么本质区别?"(调大K因子只是让评分点数"变化更快",但依然只有一个数字;TrueSkill显式维护一个不确定性参数,能够被下游决策(匹配算法、是否需要更多对局来确认评分)直接使用,这是"隐式地让点估计变化更敏感"和"显式建模并暴露不确定性这个量"两种不同思路的本质差异)

**常见坑:**
- 把TrueSkill简单理解成"更精确的Elo",而不理解它在建模范式上的本质区别——从"点估计+手工调节的更新步长"转变为"显式的概率分布+贝叶斯更新",这是11-13类贝叶斯方法论在排位系统这个具体场景下的自然延伸。
- 认为"不确定性大"就等同于"实力弱"——σ大只代表"对这个选手实力的把握程度低",和μ(实力点估计本身的高低)是两个完全独立的维度。

---

下一篇:[16-scaling-laws-and-extrapolation.md](16-scaling-laws-and-extrapolation.md) —— 从"模型间相对排位"转向"模型规模与性能的定量关系",scaling law拟合与外推的统计陷阱。
