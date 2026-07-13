# 18 · 标注一致性与分析方法论深挖(Annotator Agreement & Methodology)

> 总览见 [00-roadmap.md](00-roadmap.md)

板块IV收官篇。14-17类处理的都是"下游"问题——怎么公平比较模型、怎么给模型排位、怎么预测规模效应、怎么监控部署后的分布变化。这些分析全部建立在一个共同的、经常被忽视的"上游"前提之上:用来评测/排位/监控的**标注数据本身是否可靠**。本文引入标注一致性这个专门的统计工具家族(Cohen's kappa/Fleiss' kappa/Krippendorff's alpha),并用知识点5把板块IV(14-18类)已经产出的真实数字整合成一次"经得起连续追问"的方法论演练。

**环境声明:** 全部代码在仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1)下真实跑通,随机抽样固定种子(`np.random.default_rng(42)`)。一致性指标全部手写实现。

---

## 1. Cohen's kappa —— 扣除"纯靠运气"之后还剩多少一致

**定义与记号:** Cohen's kappa:衡量两个标注者在分类任务上的一致性,校正了"仅凭运气就能达到的一致率"这个基线之后的度量。κ=(p_o-p_e)/(1-p_e),p_o是观测一致率(两标注者判断相同的比例),p_e是期望一致率(假设两标注者判断完全独立、仅按各自边际分布随机打标签时,预期"碰巧"一致的比例)。

**一句话:** 两个标注者的原始一致率(比如90%)听起来很高,但如果这个任务本身有一个类别占了95%的样本,即使两人完全随机瞎猜,"碰巧"达到90%一致率也毫不意外——Cohen's kappa的核心价值就是把这种"纯粹靠运气/靠类别不均衡撑起来的虚假一致"从观测一致率里扣除,只留下真实超出运气水平的一致程度。

**数学推导:** p_o=(对角线格子样本数之和)/总样本数。p_e=Σ_k(类别k被标注者1标记的比例)×(类别k被标注者2标记的比例)(假设两标注者独立,每个类别"碰巧同时被两人标中"的概率是两个边际比例的乘积,对所有类别求和)。κ=(p_o-p_e)/(1-p_e)——分母是"除了运气之外理论上最多能达到的一致程度",分子是"实际超出运气水平的一致程度",κ=1完全一致,κ=0等同于纯粹随机瞎猜,κ<0比随机瞎猜还差。

**底层机制/为什么这样设计:** 为什么"一致率高但kappa低"这种情况会发生?因为观测一致率p_o完全没有对"类别分布本身是否不均衡"做任何调整——如果95%的样本都属于类别A,哪怕两个标注者的判断完全独立、各自按主观印象瞎猜,只要两人都倾向于"大部分时候标A"(因为A确实是多数类),两人的标注就会在大部分样本上"碰巧"一致,这个高p_o完全不能说明两人的判断标准真的一致,kappa通过减去这个"运气基线"p_e,把这种类别不均衡制造的虚假一致剥离出去。

**AI研究/工程场景:** LLM judge场景的标注一致性评估是Cohen's kappa最直接的现代应用——验证一个自动化的LLM judge和人类标注者的判断是否可靠一致,不能只看"人类和judge判断相同的比例",尤其在"大部分回答都是合格的、只有少数不合格"这种类别不均衡场景下(内容审核/质量评估任务里非常常见),必须用kappa这类经过运气校正的指标,否则很容易得出"judge和人类高度一致"这种误导性的虚高结论。

**可运行例子:**
```python
import numpy as np

def cohens_kappa(a, b):
    classes = sorted(set(a) | set(b))
    table = np.zeros((len(classes), len(classes)))
    for ai, bi in zip(a, b):
        table[classes.index(ai), classes.index(bi)] += 1
    n = table.sum()
    p_o = np.trace(table) / n
    row_marg, col_marg = table.sum(axis=1) / n, table.sum(axis=0) / n
    p_e = np.sum(row_marg * col_marg)
    kappa = (p_o - p_e) / (1 - p_e)
    return p_o, p_e, kappa

# 教科书标准手算例子: 2x2列联表(行=标注者A, 列=标注者B)
# Yes/Yes=20, Yes/No=5, No/Yes=10, No/No=15
table = np.array([[20, 5], [10, 15]])
n = table.sum()
p_o = np.trace(table) / n
row_marg, col_marg = table.sum(axis=1) / n, table.sum(axis=0) / n
p_e = np.sum(row_marg * col_marg)
kappa = (p_o - p_e) / (1 - p_e)

# 核心断言1: 和手算结果精确一致(p_o=0.7, p_e=0.5, kappa=0.4)
assert abs(p_o - 0.7) < 1e-9 and abs(p_e - 0.5) < 1e-9 and abs(kappa - 0.4) < 1e-9, \
    f"should match hand-calculated values exactly: p_o={p_o} p_e={p_e} kappa={kappa}"

# 类别不均衡反例: 构造两个"完全独立瞎猜"的标注者, 边际分布都强烈偏向多数类
rng = np.random.default_rng(42)
n_samples = 2000
rater_A = rng.choice([0, 1], size=n_samples, p=[0.95, 0.05])
rater_B = rng.choice([0, 1], size=n_samples, p=[0.94, 0.06])
p_o2, p_e2, kappa2 = cohens_kappa(rater_A, rater_B)

# 核心断言2: 观测一致率很高, 但kappa接近0(两人其实完全没有真实的判断一致性, 只是运气)
assert p_o2 > 0.85, f"observed agreement should look deceptively high, got {p_o2:.4f}"
assert kappa2 < 0.1, f"but kappa should reveal it's barely better than chance, got {kappa2:.4f}"

print(f"textbook example: p_o={p_o:.4f}  p_e={p_e:.4f}  kappa={kappa:.4f}")
print(f"imbalanced counterexample: p_o={p_o2:.4f} (looks great!)  kappa={kappa2:.4f} (barely better than chance)")
```

**面试怎么问+追问链**(真实性验证轴):
- Q:"标注报告里写'人类标注者和LLM judge的一致率是93%',这个数字能说明judge判断可靠吗?"
- 追问1:"还需要知道什么信息才能判断?"(需要知道任务本身的类别分布是否均衡——如果某个类别占了绝大多数样本,93%的一致率可能主要是"两边都倾向于标多数类"这种运气因素撑起来的,需要看Cohen's kappa这类经过运气校正的指标)
- 深挖追问:"如果kappa只有0.3,但原始一致率有93%,该怎么向不熟悉kappa的业务方解释这个矛盾?"(可以具体展示"如果两个标注者各自独立瞎猜,预期能达到的一致率是多少"这个反事实计算(也就是p_e),让业务方直观理解"93%里有相当一部分其实是任务本身类别不均衡'免费'贡献的",把抽象的kappa公式转化成一个具体可比较的反事实场景)

**常见坑:**
- 只报告原始一致率,不使用任何经过运气校正的指标,尤其是在类别不均衡的任务上,这个原始一致率可能严重高估标注者/judge的真实可靠性。
- 把kappa的经验分级当作放之四海而皆准的严格标准,不同任务的类别分布结构、标注难度本身差异很大,同一个kappa数值在不同任务背景下的实际含义可能不同。

---

## 2. Fleiss' kappa —— 推广到任意多个标注者,但n=2时的退化关系有一个精确条件

**定义与记号:** Fleiss' kappa:Cohen's kappa从"恰好两个标注者"推广到"两个或以上任意数量标注者"的版本。同样是κ=(p̄_o-p̄_e)/(1-p̄_e)的形式:p̄_o是所有样本上"标注者两两一致对数"占"所有可能两两配对数"的平均比例;p̄_e基于**所有标注者汇总起来的整体类别边际分布**计算(p̄_e=Σ_j p̄_j²,p̄_j是类别j在全部标注、全部样本汇总起来的整体比例)。

**一句话:** Fleiss' kappa回答的是"一群标注者整体上的判断一致程度,在扣除运气之后还剩多少",是Cohen's kappa"两人协议"这个概念在"多人协议"场景下的自然推广,但这个推广在数学上藏着一个容易被忽视的细节——知识点核心要澄清的正是这个细节。

**数学推导:** 一个常被想当然认为成立、但需要精确条件的说法是"n=2标注者时,Fleiss' kappa退化为Cohen's kappa"。真相是:Cohen's kappa的p_e=Σ_j p_A(j)·p_B(j)(两个标注者**各自独立**的边际分布相乘);Fleiss' kappa在n=2时的p̄_e=Σ_j[(p_A(j)+p_B(j))/2]²(两个标注者**汇总平均**的边际分布,平方后求和)。由均值不等式(a-b)²≥0推出(a+b)²/4≥ab,对每个类别j都成立,求和后得到:**Fleiss'的p̄_e ≥ Cohen's的p_e,等号成立当且仅当两个标注者的边际分布完全相同**。由于kappa是p_e的减函数(p_o固定时,p_e越大kappa越小),这意味着**n=2时Fleiss' kappa ≤ Cohen's kappa恒成立,只有两个标注者边际分布完全一致时才精确相等**——这是一个可以直接数值验证的精确关系,不是"大致相等"这种模糊说法。

**底层机制/为什么这样设计:** 为什么Fleiss' kappa选择用"汇总平均边际"而不是"保持每个标注者独立边际"?因为Fleiss' kappa设计的初衷就是处理"标注者数量任意、且样本可能由不同标注者子集标注"这种更松散的场景,不预设"标注者1"和"标注者2"这种固定身份对应关系(在真实多标注者场景里,谁标注了第i个样本、谁标注了第j个样本可能完全是不同的人),所以只能退而求其次,用所有标注(不区分具体是谁标的)汇总起来的整体分布作为运气基线的估计——这个设计选择在"标注者身份不重要,只关心整体一致程度"的场景下是合理的,但代价就是它不能精确复现"两个具名标注者、各自保留独立身份"的Cohen's kappa,除非这两人碰巧边际分布相同。

**AI研究/工程场景:** 大规模标注项目(构建RLHF人类偏好数据集、众包标注质量评估)往往涉及远多于2个标注者,而且经常采用"每个样本随机分配给标注者池里的几个人"这种不固定标注者组合的方式,Fleiss' kappa是评估这类大规模、标注者-样本对应关系不固定的标注项目整体质量的标准工具。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

def fleiss_kappa(ratings_matrix):
    n_samples, n_categories = ratings_matrix.shape
    n_raters = ratings_matrix.sum(axis=1)[0]
    p_i = (np.sum(ratings_matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar_o = p_i.mean()
    p_j = ratings_matrix.sum(axis=0) / (n_samples * n_raters)
    p_bar_e = np.sum(p_j ** 2)
    return (p_bar_o - p_bar_e) / (1 - p_bar_e), p_bar_o, p_bar_e

def cohens_kappa(a, b):
    classes = sorted(set(a) | set(b))
    table = np.zeros((len(classes), len(classes)))
    for ai, bi in zip(a, b):
        table[classes.index(ai), classes.index(bi)] += 1
    n = table.sum()
    p_o = np.trace(table) / n
    row_marg, col_marg = table.sum(axis=1) / n, table.sum(axis=0) / n
    p_e = np.sum(row_marg * col_marg)
    return (p_o - p_e) / (1 - p_e)

# 5标注者场景: 每个样本有一个"真实倾向", 标注者按这个倾向独立投票
n_samples, n_raters = 100, 5
true_prob = rng.uniform(0.1, 0.9, n_samples)
individual_ratings = np.array([rng.binomial(1, p, n_raters) for p in true_prob])
counts_5 = np.column_stack([n_raters - individual_ratings.sum(axis=1), individual_ratings.sum(axis=1)])
kappa_5, _, _ = fleiss_kappa(counts_5)

# 核心断言1: 5标注者的Fleiss kappa应该在合理范围([-1,1])内, 且反映出中等程度的一致性
assert -1 <= kappa_5 <= 1

# n=2退化关系的精确验证: 边际分布相同时精确相等, 边际分布不同时Fleiss <= Cohen
table_equal = np.array([[20, 5], [5, 20]])  # 行列边际都是[25,25]/50, 完全相同
table_unequal = np.array([[20, 5], [10, 15]])  # 行边际[25,25], 列边际[30,20], 不同

def table_to_cohen_and_fleiss(table):
    r1, r2 = [], []
    for i in range(2):
        for j in range(2):
            r1 += [i] * int(table[i, j])
            r2 += [j] * int(table[i, j])
    r1, r2 = np.array(r1), np.array(r2)
    cohen_k = cohens_kappa(r1, r2)
    counts = np.zeros((len(r1), 2), dtype=int)
    for idx, (a, b) in enumerate(zip(r1, r2)):
        counts[idx, a] += 1
        counts[idx, b] += 1
    fleiss_k, _, _ = fleiss_kappa(counts)
    return cohen_k, fleiss_k

cohen_eq, fleiss_eq = table_to_cohen_and_fleiss(table_equal)
cohen_uneq, fleiss_uneq = table_to_cohen_and_fleiss(table_unequal)

# 核心断言2: 边际分布相同时, 两者精确相等
assert abs(cohen_eq - fleiss_eq) < 1e-9, f"with equal marginals, Cohen and Fleiss should match exactly: {cohen_eq} vs {fleiss_eq}"
# 核心断言3: 边际分布不同时, Fleiss严格<=Cohen(不是简单的"大致相等")
assert fleiss_uneq < cohen_uneq, f"with unequal marginals, Fleiss kappa should be strictly less than Cohen's: {fleiss_uneq} vs {cohen_uneq}"

print(f"5-rater Fleiss kappa = {kappa_5:.4f}")
print(f"equal-marginal table:   Cohen={cohen_eq:.4f}  Fleiss={fleiss_eq:.4f}  (exactly equal)")
print(f"unequal-marginal table: Cohen={cohen_uneq:.4f}  Fleiss={fleiss_uneq:.4f}  (Fleiss < Cohen, not equal)")
```

**面试怎么问+追问链**(规模递增轴):
- Q:"标注项目从2个标注者扩展到5个标注者,一致性指标该怎么算?"
- 追问1:"能不能简单地对每一对标注者组合分别算Cohen's kappa,再取平均?"(这个做法直觉上说得通,而且Fleiss' kappa的p̄_o定义确实是基于两两配对计算的,但p̄_e(运气基线)的计算方式不完全等价于"简单平均每对Cohen's kappa各自的p_e"——Fleiss' kappa用整体汇总的类别边际分布计算p̄_e,是一个有单一公式支撑的规范做法,不是事后对多个两两指标做平均这种拼凑方案)
- 深挖追问:"n=2时Fleiss' kappa和Cohen's kappa是不是完全一样?"(不完全一样——只有当两个标注者的边际分布完全相同时才精确相等,边际分布不同时Fleiss' kappa严格小于Cohen's kappa,这是由p̄_e的均值不等式关系决定的精确数学结论,不是"应该差不多"这种模糊说法)

**常见坑:**
- 想当然地认为"n=2时Fleiss' kappa和Cohen's kappa总是精确相等",不理解这个退化关系需要"两标注者边际分布相同"这个精确条件——这是一个容易被面试官用来检验候选人是否只会套公式、还是真的理解公式背后数学结构的经典追问点。
- 在标注者-样本对应关系不固定、或者每个样本标注人数不同的场景下,直接套用假设"人数固定"的标准Fleiss' kappa公式,不做必要的调整。

---

## 3. Krippendorff's alpha简介 —— 能处理缺失数据的一致性指标

**定义与记号:** Krippendorff's alpha:标注一致性指标家族里更"全能"的成员,主要优势是能处理缺失数据(不是每个标注者都标注了每个样本)、能处理不同的数据类型。核心思想和kappa系列相同(扣除运气基线),但用"分歧"而不是"一致"作为基本度量单位:α=1-D_o/D_e,D_o是观测到的平均分歧度,D_e是期望的分歧度。

**一句话:** 如果说Cohen's kappa和Fleiss' kappa是一致性指标家族里最常用、最容易上手的两个成员,Krippendorff's alpha就是这个家族里更"全能"但也更复杂的成员——在名义变量、无缺失数据的最简单特例下,它和Fleiss' kappa数值高度接近,但它天生就能应对kappa系列公式结构上无法直接处理的缺失数据。

**数学推导/说明:** 在名义变量+0/1距离的最简单情形下,D_o是所有"实际存在的标注对"里判断不同的比例,D_e是把所有观测到的标注值汇总起来、两两配对后判断不同的期望比例——这个计算框架从"每一对实际存在的标注"出发,不要求来自固定的标注者身份、不要求每个样本标注人数一致,天然只统计"确实存在的标注对",缺失的标注根本不会被纳入计算。

**底层机制/为什么这样设计:** 为什么Krippendorff's alpha能处理缺失数据,而标准Fleiss' kappa公式做不到?因为Fleiss' kappa的p̄_o计算隐含假设了"每个样本有固定数量n个标注者的完整标注"(分母结构依赖这个n);Krippendorff's alpha的计算框架不依赖这个假设,不需要对缺失位置做任何填补或特殊处理——这是它在处理真实、不完美的众包标注数据时更受青睐的核心原因。

**AI研究/工程场景:** 大规模众包标注项目(比如RLHF偏好数据收集)中,由于标注者流失、任务分配不均等现实原因,"每个样本都被完全相同数量的标注者标注"这个理想情况经常不成立,这时候Krippendorff's alpha比Fleiss' kappa更适合直接处理这种"参差不齐"的真实标注数据。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

def krippendorff_alpha_nominal(ratings_matrix):
    """ratings_matrix: n_samples x n_raters, 值为类别标签, NaN表示缺失"""
    n_samples = ratings_matrix.shape[0]
    disagree_pairs, total_pairs = 0, 0
    all_values = []
    for i in range(n_samples):
        vals = ratings_matrix[i][~np.isnan(ratings_matrix[i])]
        all_values.extend(vals.tolist())
        m_i = len(vals)
        for a in range(m_i):
            for b in range(m_i):
                if a == b:
                    continue
                total_pairs += 1
                disagree_pairs += (vals[a] != vals[b])
    D_o = disagree_pairs / total_pairs

    all_values = np.array(all_values)
    N = len(all_values)
    disagree_expected, total_expected = 0, 0
    for a in range(N):
        for b in range(N):
            if a == b:
                continue
            total_expected += 1
            disagree_expected += (all_values[a] != all_values[b])
    D_e = disagree_expected / total_expected
    return 1 - D_o / D_e

def fleiss_kappa(ratings_matrix_counts):
    n_samples, n_categories = ratings_matrix_counts.shape
    n_raters = ratings_matrix_counts.sum(axis=1)[0]
    p_i = (np.sum(ratings_matrix_counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar_o = p_i.mean()
    p_j = ratings_matrix_counts.sum(axis=0) / (n_samples * n_raters)
    p_bar_e = np.sum(p_j ** 2)
    return (p_bar_o - p_bar_e) / (1 - p_bar_e)

n_samples, n_raters = 100, 5
true_prob = rng.uniform(0.1, 0.9, n_samples)
individual_ratings = np.array([rng.binomial(1, p, n_raters) for p in true_prob], dtype=float)

alpha_full = krippendorff_alpha_nominal(individual_ratings)
counts = np.column_stack([n_raters - individual_ratings.sum(axis=1), individual_ratings.sum(axis=1)]).astype(int)
kappa_full = fleiss_kappa(counts)

# 核心断言1: 名义变量+无缺失数据情形下, alpha和Fleiss kappa应该高度接近
assert abs(alpha_full - kappa_full) < 0.02, \
    f"in the simple no-missing nominal case, alpha and Fleiss kappa should be very close: {alpha_full:.4f} vs {kappa_full:.4f}"

# 引入20%随机缺失(每个样本的标注人数不再固定, Fleiss kappa的标准公式假设会被破坏)
individual_ratings_missing = individual_ratings.copy()
missing_mask = rng.random((n_samples, n_raters)) < 0.2
individual_ratings_missing[missing_mask] = np.nan
n_valid = (~np.isnan(individual_ratings_missing)).sum(axis=1)

# 核心断言2: 引入缺失后标注人数确实不再固定(标准Fleiss公式的前提被打破)
assert n_valid.min() < n_valid.max(), "after introducing missingness, raters-per-sample should now vary"

# 核心断言3: Krippendorff's alpha依然能对参差不齐的数据给出合理数值(不需要任何插补或特殊处理)
alpha_missing = krippendorff_alpha_nominal(individual_ratings_missing)
assert -1 <= alpha_missing <= 1, f"alpha should remain a valid, interpretable number even with missing data, got {alpha_missing:.4f}"

print(f"no missing data: Krippendorff's alpha={alpha_full:.4f}  Fleiss' kappa={kappa_full:.4f}  (close)")
print(f"with 20% missing: raters per sample ranges {n_valid.min()}-{n_valid.max()}, Krippendorff's alpha={alpha_missing:.4f}  (still computable)")
```

**面试怎么问+追问链**(工程约束递增轴):
- Q:"众包标注平台上,每个样本被标注的人数不完全一样,还有部分因为标注者中途放弃而缺失,这种情况下Fleiss' kappa还能用吗?"
- 追问1:"为什么标准的Fleiss' kappa公式在这种场景下会遇到困难?"(Fleiss' kappa公式的p̄_o计算依赖每个样本有固定标注者人数n这个假设,标注人数参差不齐会破坏这个假设,虽然存在一些推广变体能处理这种情况,但需要额外的数学调整)
- 深挖追问:"Krippendorff's alpha是怎么绕开这个'固定标注人数'假设的?"(它从"所有实际存在的标注对"这个更基础的单位出发构建计算框架,不预设每个样本必须有固定数量的标注,缺失的标注对自然地不参与计算)

**常见坑:**
- 遇到有缺失数据的标注场景,依然强行用Fleiss' kappa的标准公式(比如插补缺失值或简单丢弃有缺失的样本),而不是直接采用为处理缺失数据设计的Krippendorff's alpha。
- 把Krippendorff's alpha当作"永远比kappa更好"的替代品盲目使用,而不理解它需要为具体数据类型正确定义距离函数,在数据完整、类型简单的常规场景下,Cohen's/Fleiss' kappa足够胜任且更容易被同行理解。

---

## 4. 一致性不够时的处理策略 —— 先诊断根因,不要空谈"要沟通"

**定义与记号:** 当计算出的标注一致性指标低于可接受水平时,不应该止步于"一致性不够"这个结论,而应该系统性地排查具体原因——常见根因包括:**难样本型**(整体kappa低是少数天然模糊的样本拉低的,大部分样本一致性其实很高)、**问题标注者型**(某个特定标注者的判断标准系统性地偏离其他人)。本知识点用具体的合成案例展示如何用数据诊断出这几种不同的根因,而不是笼统地"重新培训所有人"。

**一句话:** "一致性不够,需要加强标注培训"这种笼统的应对方式,如果不先诊断出具体是哪种原因导致的低一致性,很可能是在盲目地"多做一点什么"而没有真正对症下药——排查根因这一步,和14-17类反复强调的"先诊断再分析"是完全同一种方法论纪律。

**数学推导/说明:** **难样本型**诊断:按样本的真实标注难度分层重新计算kappa,如果"简单样本"子集kappa很高、"难样本"子集kappa很低(甚至接近0),说明低一致性主要来自特定一部分本身就有争议的样本。**问题标注者型**诊断:计算每个标注者和"其他所有标注者多数意见"的一致率,如果某一个标注者的一致率显著低于其他人(其他标注者两两之间一致率都较高,唯独这一个人和大家都不一致),说明问题集中在这一个标注者身上。

**底层机制/为什么这样设计:** 为什么区分这些根因很重要,不能都用同一种方式解决?因为不同问题对应的最优解决方案完全不同——难样本型不一定需要"解决"(有些样本天然模糊,合理应对是标记为"低置信度"、引入仲裁流程,而不是苛求所有标注者对本质上有争议的样本达成一致);问题标注者型只需要单独和这一个人沟通、核实理解偏差,不需要对表现良好的其他标注者做不必要的重新培训——诊断错了根因,可能把大量精力花在不必要的"重新培训全员"上,而真正的问题却没有被针对性解决。

**AI研究/工程场景:** RLHF偏好数据标注、内容审核标注质量控制等真实项目里,标注质量团队的核心日常工作之一就是持续监控整体一致性指标,一旦发现下降或持续偏低,系统性地做这类分层/分标注者的诊断分析,而不是一发现问题就笼统地重新培训所有人。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

def fleiss_kappa(counts):
    n_samples, n_categories = counts.shape
    n_raters = counts.sum(axis=1)[0]
    p_i = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar_o = p_i.mean()
    p_j = counts.sum(axis=0) / (n_samples * n_raters)
    p_bar_e = np.sum(p_j ** 2)
    return (p_bar_o - p_bar_e) / (1 - p_bar_e)

def to_counts(ratings, n_raters):
    return np.column_stack([n_raters - ratings.sum(axis=1), ratings.sum(axis=1)]).astype(int)

# ---- 场景A: 难样本型 ----
n_raters = 6
n_easy, n_hard = 180, 20
easy_prob = rng.choice([0.05, 0.95], n_easy)  # 简单样本: 明确的0或1
hard_prob = rng.uniform(0.4, 0.6, n_hard)      # 难样本: 概率接近0.5, 天然容易分歧
easy_ratings = np.array([rng.binomial(1, p, n_raters) for p in easy_prob])
hard_ratings = np.array([rng.binomial(1, p, n_raters) for p in hard_prob])
all_ratings_a = np.vstack([easy_ratings, hard_ratings])

kappa_all_a = fleiss_kappa(to_counts(all_ratings_a, n_raters))
kappa_easy = fleiss_kappa(to_counts(easy_ratings, n_raters))
kappa_hard = fleiss_kappa(to_counts(hard_ratings, n_raters))

# 核心断言1: 分层后, 简单样本子集kappa明显更高, 难样本子集kappa接近0(拉低整体的元凶)
assert kappa_easy > kappa_hard + 0.3, f"easy-sample subset should show much higher agreement: easy={kappa_easy:.4f} hard={kappa_hard:.4f}"
assert kappa_hard < 0.3, f"hard-sample subset should show near-chance agreement, got {kappa_hard:.4f}"

# ---- 场景B: 问题标注者型 ----
n_samples_b = 150
true_prob_b = rng.choice([0.05, 0.95], n_samples_b)  # 全是清晰样本
good_ratings = np.array([rng.binomial(1, p, 5) for p in true_prob_b])  # 5个正常标注者
bad_rater = np.array([1 - int(rng.binomial(1, p)) if rng.random() < 0.3 else int(rng.binomial(1, p)) for p in true_prob_b])  # 第6个: 30%概率系统性反着标
all_ratings_b = np.column_stack([good_ratings, bad_rater])

def agreement_with_majority(ratings, rater_idx):
    others = np.delete(ratings, rater_idx, axis=1)
    majority = (others.sum(axis=1) > others.shape[1] / 2).astype(int)
    return (ratings[:, rater_idx] == majority).mean()

agreements = [agreement_with_majority(all_ratings_b, i) for i in range(6)]

# 核心断言2: 5个正常标注者和多数意见的一致率都很高, 第6个(问题标注者)明显偏低
assert min(agreements[:5]) > 0.85, f"the 5 good raters should all agree closely with the majority: {agreements[:5]}"
assert agreements[5] < min(agreements[:5]) - 0.15, \
    f"the problem rater should stand out clearly: rater6={agreements[5]:.4f} vs min_good={min(agreements[:5]):.4f}"

print(f"scenario A (hard samples): overall kappa={kappa_all_a:.4f}  easy subset={kappa_easy:.4f}  hard subset={kappa_hard:.4f}")
print(f"scenario B (problem annotator): agreement with majority per rater = {[round(a,3) for a in agreements]}")
print("  -> rater #6 is the clear outlier, not a general training problem")
```

**面试怎么问+追问链**(诊断真实数据新题型):
- Q:"整体标注一致性kappa只有0.35(偏低),你会怎么排查这个问题的根因?"
- 追问1:"有哪些具体的诊断步骤,而不是直接说'需要重新培训'?"(至少应该:①按样本难度或类别分层重新计算kappa,看是不是少数难样本在拉低整体数字;②计算每个标注者和其他人多数意见的一致率,看是不是某个特定标注者是异常值;③抽查分歧样本,人工review标注者们各自的判断依据)
- 深挖追问:"如果排查发现是'难样本型',这些样本该怎么处理,还要不要纳入最终的训练/评测数据?"(不应该简单丢弃这些样本,更合理的做法是标记为"标注者分歧较大/低置信度",下游使用时降低这些样本的权重、或引入更多轮标注/专家仲裁,而不是假装这些样本和其他清晰样本具有同等的标签可信度)

**常见坑:**
- 一旦发现整体一致性指标偏低,不做任何分层/分标注者的诊断分析,直接采取"重新培训全体标注者"这种一刀切、成本高但可能没有针对性的应对措施。
- 排查出是"难样本型"问题后,简单粗暴地把这些低一致性样本从数据集里删除,丢失了这部分样本本身携带的、关于任务边界模糊性的真实信息。

---

## 5. "经得起追问的具体数字"方法论收尾

**定义与记号:** 板块IV(14-18类)的收尾知识点,不引入新的统计方法,而是把已经验证过的真实数字整理成一次"如果被连续追问,是否都有具体数字支撑"的演练:14类知识点1(200题bootstrap CI宽度可达0.135)、14类知识点6(m=100次验证集选择膨胀>0.05,held-out重新评估无偏<0.01)、15类知识点4(完全可分数据Elo差距达5920分vs正常数据347分)、16类知识点5(9次多项式过拟合样本外MSE恶化153倍)、17类知识点4(概念偏移下模型MSE恶化39倍但特征分布监控完全无法检测)、18类知识点1(类别不均衡场景下一致率>90%但kappa接近0)——这些数字共同指向同一个方法论:任何"看起来合理"的结论,都应该能回答"具体数字是多少、怎么算出来的、和哪个基准比较"这三个追问。

**一句话:** 这个知识点不是重新验证一遍前面18个知识点,而是练习一种更高阶的能力——把分散在不同文件里的具体发现,组织成一个能够撑住"连续追问"的连贯论述,呼应dsa-deep-dive调研里"55分钟系统设计轮次,初版方案只占15分钟,其余40分钟全是追问"这个真实面试结构对"知识整合能力"的隐性要求。

**数学推导/说明:** 用一段整合性代码演示一个跨知识点的复合论证,而不是只在文字上宣称"两者有关系":构造两种标注质量场景(低kappa vs 高kappa,基于同一份真实底层标签),分别计算①标注一致性(本文knowledge point 2的Fleiss' kappa)、②基于多数投票标签的评测分数bootstrap置信区间(14类的方法)。核心发现:**bootstrap CI的宽度只反映"题目抽样"这一层不确定性,和标注本身是否可靠(kappa高低)几乎无关**——两种kappa截然不同的场景可能算出宽度几乎相同的bootstrap CI,但低kappa场景下多数投票标签本身的准确率可能只有60%出头,而高kappa场景下接近100%准确。这意味着"bootstrap CI很窄"绝不能被解读为"这个分数很可信",CI窄只保证了"如果重新抽一批同样规模的题目,分数不会有太大变化",完全不保证"用来打分的标签本身是对的"。

**底层机制/为什么这样设计:** 为什么要专门设计一个"整合场景"而不是简单罗列前面的数字?因为真实面试/工作场景里,不同的统计概念很少是孤立出现的——一个关于"模型评测是否可信"的完整论证,往往需要同时考虑"评测分数本身的抽样不确定性"(14类)和"评测所依赖的标注数据本身的质量"(18类)这两层完全不同来源的不确定性,只谈其中一层而忽视另一层,论证是不完整的。

**AI研究/工程场景:** 真实的模型评测报告如果要做到严谨、能扛住连续追问,理想情况下应该同时交代:评测集本身的规模和bootstrap置信区间(14类)、评测标签/标注的一致性水平(18类)、模型间比较是否用了配对方法(14类)、排位是否说明了不确定性(15类)、外推是否给出了距离和风险提示(16类)、监控体系是否覆盖了协变量偏移和概念偏移两类风险(17类)——不是要求每次汇报都堆砌所有内容,而是要求分析者对这些维度都心里有数,能在被问到时随时调出具体数字。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

def fleiss_kappa(counts):
    n_samples, n_categories = counts.shape
    n_raters = counts.sum(axis=1)[0]
    p_i = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar_o = p_i.mean()
    p_j = counts.sum(axis=0) / (n_samples * n_raters)
    p_bar_e = np.sum(p_j ** 2)
    return (p_bar_o - p_bar_e) / (1 - p_bar_e)

def bootstrap_ci(scores, n_boot=2000):
    boot_means = np.array([scores[rng.integers(0, len(scores), len(scores))].mean() for _ in range(n_boot)])
    return np.percentile(boot_means, [2.5, 97.5])

n_questions, n_raters = 200, 6
true_model_correct = rng.binomial(1, 0.6, n_questions)  # 同一份真实标签, 两种标注质量场景共用

def simulate_annotation(true_labels, agreement_strength, rng):
    votes = np.zeros((len(true_labels), n_raters), dtype=int)
    for i, t in enumerate(true_labels):
        p = np.clip(t * agreement_strength + (1 - t) * (1 - agreement_strength), 0.02, 0.98)
        votes[i] = rng.binomial(1, p, n_raters)
    return votes

votes_low = simulate_annotation(true_model_correct, 0.58, rng)   # 低一致性: 标注和真实标签关联很弱
votes_high = simulate_annotation(true_model_correct, 0.95, rng)  # 高一致性: 标注和真实标签强关联

def counts(votes):
    return np.column_stack([n_raters - votes.sum(axis=1), votes.sum(axis=1)]).astype(int)

kappa_low, kappa_high = fleiss_kappa(counts(votes_low)), fleiss_kappa(counts(votes_high))
majority_low = (votes_low.sum(axis=1) > n_raters / 2).astype(float)
majority_high = (votes_high.sum(axis=1) > n_raters / 2).astype(float)
ci_low, ci_high = bootstrap_ci(majority_low), bootstrap_ci(majority_high)
acc_low = (majority_low == true_model_correct).mean()
acc_high = (majority_high == true_model_correct).mean()

# 核心断言1: 两种场景的标注一致性(kappa)截然不同
assert kappa_low < 0.1, f"low-agreement scenario should show near-chance kappa, got {kappa_low:.4f}"
assert kappa_high > 0.6, f"high-agreement scenario should show strong kappa, got {kappa_high:.4f}"

# 核心断言2(核心发现): 尽管kappa天差地别, 两者的bootstrap CI宽度却几乎一样宽
width_low, width_high = ci_low[1] - ci_low[0], ci_high[1] - ci_high[0]
assert abs(width_low - width_high) < 0.03, \
    f"bootstrap CI width should look similarly 'precise' in both scenarios despite the huge kappa gap: low={width_low:.4f} high={width_high:.4f}"

# 核心断言3: 但多数投票标签的真实准确率天差地别 -- CI宽度完全没有暴露这个差距
assert acc_low < 0.7, f"low-kappa majority labels should be quite unreliable, got accuracy={acc_low:.4f}"
assert acc_high > 0.95, f"high-kappa majority labels should be highly reliable, got accuracy={acc_high:.4f}"
assert acc_high - acc_low > 0.3, "the reliability gap should be large -- exactly what bootstrap CI alone fails to reveal"

print(f"LOW annotation agreement:  kappa={kappa_low:.4f}  bootstrap CI width={width_low:.4f}  majority-label accuracy={acc_low:.4f}")
print(f"HIGH annotation agreement: kappa={kappa_high:.4f}  bootstrap CI width={width_high:.4f}  majority-label accuracy={acc_high:.4f}")
print("=> bootstrap CI width alone cannot tell you whether the underlying labels are trustworthy -- kappa is a separate, necessary check.")
```

**面试怎么问+追问链**(真实性验证轴+决策依据追问轴,汇总收束板块IV):
- Q:"你说这个模型评测结果'可信',具体是什么意思,能不能具体说说?"
- 追问1:"评测分数本身的不确定性有多大?"(应该能立刻给出类似14类知识点1的bootstrap置信区间数字,而不是只有一个点估计)
- 追问2:"这个评测依赖的标注/打分,标注者之间一致性如何?"(应该能给出本文知识点1-2的kappa数字,而不是假设标注天然就是"金标准"没有任何不确定性)
- 深挖追问:"如果标注质量本身有问题,会怎么影响你刚才给出的置信区间?"(应该能像本知识点的可运行例子那样,具体展示bootstrap CI宽度对标注质量问题完全不敏感——低kappa和高kappa场景下CI宽度几乎一样,但标签本身的准确率天差地别,这正说明"CI窄"不能替代"标注可靠"的独立验证)

**常见坑:**
- 把统计方法当作互相独立的工具箱,只在被明确要求时才使用某一个,而不去主动思考不同来源的不确定性(评测分数本身的抽样波动、评测依赖的标注质量)是否会叠加影响最终结论的可信度。
- 认为"bootstrap CI很窄"就等于"这个结果很可信"——CI窄只说明"重新抽一批同样规模的题目,分数不会有太大变化",不说明"用来打分的标签本身是准确的",这是两个完全独立的问题,必须分别验证。

---

板块IV(AI/ML场景专属统计,14-18类,共27个知识点)到此收官。下一篇:[19-time-series-foundations.md](19-time-series-foundations.md) —— 板块V时间序列基础,现场建立最小必要直觉,不假设随机过程先修知识。
