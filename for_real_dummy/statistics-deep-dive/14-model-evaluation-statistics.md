# 14 · 模型评测统计深挖(Model Evaluation Statistics)

> 总览见 [00-roadmap.md](00-roadmap.md)

板块IV(AI/ML场景专属统计)开篇。板块I-III建立的统计工具(t检验、配对设计、bootstrap、多重比较)在这里集中应用到一个所有AI从业者都会遇到的具体场景:模型评测。"benchmark分数"本质上和A/B测试指标、任何统计估计量是同一类对象——都是从有限样本算出的、带抽样不确定性的统计量,却经常被当作没有误差棒的"考试分数"直接拿来做模型选型决策。本文用05类多重比较、04类配对设计、bootstrap方法论,系统性地拆解模型评测里最常见的几个统计陷阱。

**环境声明:** 全部代码在仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1)下真实跑通,随机抽样固定种子(`np.random.default_rng(42)`)。`learning/llm-judge-arena/`模块(MT-Bench/Arena-Hard风格评测脚本)已核实其现状:`mt_bench_runner.py`、`arena_hard_runner.py`产出的是评测集上的点估计分数,没有计算bootstrap置信区间——这正是本文要补的分析层,不是虚构的关联。

---

## 1. 单个benchmark分数会骗人 —— 评测分数也是一个有抽样误差的统计量

**定义与记号:** 模型在benchmark上报告的"总分"是在一个固定、有限大小的评测题目集上计算出的经验平均值——这个分数本身是一个统计量,继承了03类"任何统计量都有抽样分布"这个基本事实:换一份同等规模、来自同一评测分布的题目子集重新打分,分数会有波动。忽视这个抽样波动,直接把两个模型分数的微小差异解读为"能力差异",是模型评测里最容易被忽视的统计陷阱。

**一句话:** "模型A得了85.2分,模型B得了84.9分,A更好"这句话如果不知道这0.3分的差异相对评测集本身的抽样波动有多大,信息量可能接近于0——评测集抽样波动本身完全可能超过0.3分。

**数学推导:** 用bootstrap方法(04类已建立)量化这个波动:对同一个评测题目池(n道题)重复抽取N次同等大小(n道题,有放回)的bootstrap重采样,每次重新计算平均分,得到一个分数的bootstrap分布——这个分布的标准差就是评测分数本身的抽样标准误,分布的宽度直接反映"如果换一批同样规模的评测题目,分数大概会在什么范围内波动"。

**底层机制/为什么这样设计:** 为什么benchmark分数需要像A/B测试指标一样认真对待抽样不确定性,而不是当作确定性的"考试分数"?因为benchmark本质上是从"这个模型在所有可能任务上的真实能力分布"里抽取的一个有限样本,和A/B测试从"用户总体行为分布"里抽取样本在统计结构上是同一件事——benchmark题目数量通常远小于A/B测试的用户样本量(可能只有几百道题),抽样波动往往比大规模A/B测试更显著,更需要被认真对待。

**AI研究/工程场景:** `learning/llm-judge-arena/`的`mt_bench_runner.py`(L02)、`arena_hard_runner.py`(L03)实现了MT-Bench、Arena-Hard风格的评测流程,已核实这些脚本输出的是评测集上的点估计胜率/分数,没有计算bootstrap置信区间——这正是模型评测在实际工程实践里经常被简化掉、但学术论文和严肃的模型对比报告里应该包含的分析层,本知识点补上这一层。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

n_questions = 200
# 每道题目有各自的"难度"(答对概率), 模拟真实benchmark里难度分布不均匀这个结构
question_difficulty = rng.beta(2, 2, n_questions)
official_scores = rng.binomial(1, question_difficulty)  # 模型在每道题上的作答(0/1)
official_score = official_scores.mean()

n_bootstrap = 2000
bootstrap_scores = np.zeros(n_bootstrap)
for b in range(n_bootstrap):
    idx = rng.integers(0, n_questions, n_questions)
    bootstrap_scores[b] = official_scores[idx].mean()

boot_std = bootstrap_scores.std()
ci_low, ci_high = np.percentile(bootstrap_scores, [2.5, 97.5])

# 核心断言: bootstrap标准差和CI宽度应该明显不可忽视(不是零误差的"精确分数")
assert boot_std > 0.02, f"bootstrap std should reflect meaningful sampling uncertainty, got {boot_std:.4f}"
assert (ci_high - ci_low) > 0.05, f"CI width should exceed the scale of typical 'meaningful' score differences, got {ci_high - ci_low:.4f}"

print(f"official score (point estimate) = {official_score:.4f}")
print(f"bootstrap std = {boot_std:.4f}")
print(f"95% CI = ({ci_low:.4f}, {ci_high:.4f})  width = {ci_high - ci_low:.4f}")
print("=> a reported score difference smaller than this CI width could easily be sampling noise, not a real capability gap")
```

**面试怎么问+追问链**(真实性验证轴):
- Q:"两个模型在同一个benchmark上的分数分别是85.2和84.9,能说明A模型更强吗?"
- 追问1:"这个0.3分的差异需要满足什么条件才有意义?"(需要知道这两个分数各自的抽样不确定性,如果两个模型置信区间大幅重叠,0.3分的差异可能完全在噪声范围内,不能可靠地说明谁更强)
- 深挖追问:"如果这0.3分差异确实超出了置信区间范围,是不是就能下结论了?"(还需要考虑04-05类讨论过的问题——如果同时对比了很多个模型/很多个benchmark维度,存在多重比较问题;此外还要确认两个模型是不是在完全相同的评测子集上测的)

**常见坑:**
- 把benchmark总分当作没有不确定性的"确定性事实"直接用于模型选型决策,不做任何形式的置信区间/显著性分析。
- 反过来走向另一个极端,认为"评测集太小了所以benchmark分数完全没有参考价值"——bootstrap分析恰恰能帮助量化"这个分数到底有多可信",而不是简单地全盘否定或全盘接受。

---

## 2. Paired bootstrap比较模型 —— 保留"同一批题目"这个配对结构

**定义与记号:** paired bootstrap(配对bootstrap):比较两个模型在**同一批**评测题目上的分数差异时,不是分别对两个模型的分数各自独立做bootstrap(那样会丢失"同一道题目上两个模型表现的相关性"这个信息),而是每次bootstrap重采样"题目"这个单位,对采样出的每一批题目子集,同时计算两个模型在这批题目上的分数,再算差值——这保持了配对结构(04类知识点2已讲过配对设计更有效的原理)。

**一句话:** 两个模型是在同一批题目上测的,题目本身的难度差异会同时影响两个模型的分数(难题两个模型都容易错,简单题两个模型都容易对)——如果分别独立bootstrap两个模型的分数再相减,会把这部分"题目难度共同影响"重新引入的噪声算重复了两次,配对bootstrap通过"每次重采样都对两个模型用同一批题目"避免了这个问题。

**数学推导:** 设第i道题目上模型A、B的得分分别是a_i、b_i,差值d_i=a_i-b_i。配对bootstrap本质上等价于直接对差值序列{d_i}做单样本bootstrap(04类bootstrap CI方法论),而不是对{a_i}和{b_i}分别独立bootstrap再相减。两者数学上不等价:Var(d_i)=Var(a_i)+Var(b_i)-2Cov(a_i,b_i)(配对方法,考虑了协方差项)通常明显小于Var(a_i)+Var(b_i)(错误的独立做法,隐含假设协方差为0),因为a_i和b_i(同一道题目上两个模型的表现)通常正相关(难题两个模型都倾向于错)。

**底层机制/为什么这样设计:** 为什么"错误的独立bootstrap"会系统性地高估比较的不确定性?因为同一题目上两个模型的正确/错误高度相关,这个正相关性意味着"哪些题目被抽到"这件事对两个模型的分数产生同向的影响,这个同向影响在计算**差值**时会被大量抵消(配对设计的核心优势);如果错误地独立重采样,相当于人为地打散了这种同向抵消的机会,重新引入了本可以被抵消掉的噪声,导致估计出的差值置信区间比真实情况更宽。

**AI研究/工程场景:** 学术论文和工程报告里比较两个模型在同一评测集上的表现时,如果要严谨地报告"模型A比模型B在这个benchmark上确实更好还是仅仅是噪声",配对bootstrap几乎是唯一正确的做法——`learning/llm-judge-arena/`的Arena-Hard(L03)、MT-Bench(L02)runner脚本天然产出的就是"同一批题目、两个模型各自的逐题结果"这种配对数据结构,是应用配对bootstrap的天然场景。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

n_questions = 150
question_difficulty = rng.beta(0.5, 0.5, n_questions)  # U型难度分布, 强化"共同难度因素"这个结构
true_delta = 0.05

noise_A = rng.normal(0, 0.05, n_questions)
noise_B = rng.normal(0, 0.05, n_questions)
prob_A = np.clip(question_difficulty + noise_A, 0.02, 0.98)
prob_B = np.clip(question_difficulty + true_delta + noise_B, 0.02, 0.98)
scores_A = rng.binomial(1, prob_A)
scores_B = rng.binomial(1, prob_B)
d = scores_B.astype(float) - scores_A.astype(float)
corr_AB = np.corrcoef(scores_A, scores_B)[0, 1]

n_boot = 3000
# 正确: 配对bootstrap(每次重采样题目索引, 两个模型用同一批题目)
paired_boot_deltas = np.zeros(n_boot)
for b in range(n_boot):
    idx = rng.integers(0, n_questions, n_questions)
    paired_boot_deltas[b] = d[idx].mean()
ci_paired = np.percentile(paired_boot_deltas, [2.5, 97.5])

# 错误: 独立bootstrap两组再相减(忽视配对结构)
indep_boot_deltas = np.zeros(n_boot)
for b in range(n_boot):
    idx_a = rng.integers(0, n_questions, n_questions)
    idx_b = rng.integers(0, n_questions, n_questions)
    indep_boot_deltas[b] = scores_B[idx_b].mean() - scores_A[idx_a].mean()
ci_indep = np.percentile(indep_boot_deltas, [2.5, 97.5])

w_paired = ci_paired[1] - ci_paired[0]
w_indep = ci_indep[1] - ci_indep[0]

# 核心断言: 同一道题目上两模型表现确实正相关, 配对bootstrap的区间应明显更窄
assert corr_AB > 0.2, f"same-question performance should be positively correlated, got corr={corr_AB:.4f}"
assert w_paired < w_indep * 0.9, \
    f"paired bootstrap CI should be visibly narrower than the (incorrect) independent bootstrap CI: paired={w_paired:.4f} indep={w_indep:.4f}"

print(f"corr(scores_A, scores_B) on the same questions = {corr_AB:.4f}")
print(f"paired bootstrap CI width = {w_paired:.4f}")
print(f"independent (wrong) bootstrap CI width = {w_indep:.4f}  (wider, wastes the pairing information)")
```

**面试怎么问+追问链**(决策依据追问轴):
- Q:"对比两个模型在同一个benchmark上的分数差异,你会怎么构造置信区间?"
- 追问1:"为什么不能分别对两个模型的分数独立做bootstrap再相减?"(同一道题目上两个模型的表现通常正相关,独立bootstrap会打散这种相关性,人为地让计算出的置信区间比真实情况更宽,低估了比较的精确度)
- 深挖追问:"如果两个模型不是在完全相同的题目集上测的,配对bootstrap还适用吗?"(不完全适用——配对方法要求"同一道题目"这个配对结构真实存在,如果测试集不同,就不能直接配对,退回到需要考虑两组独立样本的比较方式,此时通常需要更大的样本量才能达到同样的检验效力)

**常见坑:**
- 拿到两个模型的评测分数差异,直接对两组分数做独立样本的bootstrap或t检验,没有意识到同一测试集上的逐题数据天然是配对结构,浪费了配对设计本可以带来的精度提升。
- 认为"配对bootstrap"和"配对t检验"是完全等价、可以随意替换的两种方法——配对bootstrap不依赖正态性假设,在评测分数分布明显偏态时比配对t检验更稳健。

---

## 3. McNemar检验 —— 只看两个分类器"意见不一致"的那部分样本

**定义与记号:** McNemar检验:专门用于比较两个分类器在**同一测试集**上表现的假设检验方法,聚焦"不一致"的情形——把测试集按"分类器A对/错"×"分类器B对/错"分成四格列联表,只关注"A对B错"(n₁₀)和"A错B对"(n₀₁)这两个不一致的格子,检验统计量 χ²=(\|n₁₀-n₀₁\|-1)²/(n₁₀+n₀₁)(带连续性校正),在原假设(两分类器错误率相同)下近似服从自由度1的卡方分布。

**一句话:** 两个分类器"都对"或"都错"的样本对判断谁更好完全没有信息量(打平了),McNemar检验精明地只看两者"意见不一致"的那部分样本,数"A独赢多少次"vs"B独赢多少次",这个不对称程度才是真正带有信息量的部分。

**数学推导:** 在原假设H0(两分类器真实错误率相同)下,条件于n₁₀+n₀₁(总不一致样本数)这个量,n₁₀服从Binomial(n₁₀+n₀₁, 0.5)(每次"不一致"事件,谁对谁错各占50%机会,这是H0在配对不一致样本层面的精确数学含义)。McNemar统计量是对这个二项分布做正态近似(大样本下Binomial(n,0.5)近似N(n/2,n/4))转换成的卡方检验形式,连续性校正(减1)让离散二项分布用连续卡方分布近似时更准确。

**底层机制/为什么这样设计:** 为什么要"条件于n₁₀+n₀₁"这个操作?因为"两者都对"和"两者都错"的样本数,在H0和H1下都不能提供任何关于"谁更好"的直接信息,把分析条件在"不一致样本"这个子集上,恰好把无信息量的部分过滤掉——这是02类知识点5充分统计量思想在这里的具体应用:n₁₀+n₀₁和n₁₀这两个统计量联合起来对"两分类器错误率是否相同"这个问题是充分的,不需要完整的四格列联表。

**AI研究/工程场景:** 比较两个分类模型(内容审核模型新旧版本、或两个不同的LLM judge在同一批标注数据上的判断)在同一测试集上的准确率差异时,McNemar检验是标准工具——相比于把两组准确率当作独立比例做z检验(错误地忽视配对结构),McNemar检验正确地利用了"同一测试集"这个配对结构,是知识点2"paired bootstrap"精神的一个专门化、有解析解的经典版本。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

n = 300
true_label = rng.integers(0, 2, n)
difficulty = rng.beta(1.5, 3, n)  # 每道题的出错概率基线, 两个分类器共享同一个难度结构

err_prob_A = np.clip(difficulty + 0.05, 0.01, 0.95)
err_prob_B = np.clip(difficulty - 0.10, 0.01, 0.95)  # B整体更准
err_A = rng.random(n) < err_prob_A
err_B = rng.random(n) < err_prob_B
pred_A = np.where(err_A, 1 - true_label, true_label)
pred_B = np.where(err_B, 1 - true_label, true_label)

correct_A, correct_B = (pred_A == true_label), (pred_B == true_label)
n10 = np.sum(correct_A & ~correct_B)   # A对B错
n01 = np.sum(~correct_A & correct_B)   # A错B对

chi2_stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
binom_p = stats.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue  # 精确二项检验交叉验证

# 核心断言1: 不一致样本两格都非空(不是退化情形), 且方向符合"B更准"的设计
assert n10 > 0 and n01 > 0, "both discordant cells should be populated for a meaningful McNemar test"
assert n01 > n10, f"B being genuinely more accurate should show more 'A wrong, B right' cases: n10={n10} n01={n01}"

# 核心断言2: 卡方近似和精确二项检验的p值应该数值接近(交叉验证)
assert abs(p_value - binom_p) < 0.02, f"chi-square approximation should be close to the exact binomial test: {p_value:.4f} vs {binom_p:.4f}"

# 核心断言3: 真实存在差异, 检验应该正确拒绝"两分类器错误率相同"
assert p_value < 0.05, f"McNemar test should detect the genuine accuracy difference, got p={p_value:.4f}"

print(f"n10 (A right, B wrong) = {n10}, n01 (A wrong, B right) = {n01}")
print(f"McNemar chi2 = {chi2_stat:.4f}, p = {p_value:.6f}  (exact binomial p = {binom_p:.6f})")
```

**面试怎么问+追问链**(决策依据追问轴):
- Q:"比较两个分类模型在同一测试集上的准确率,为什么不能直接对两个准确率做两比例z检验?"
- 追问1:"两比例z检验的假设前提是什么,这里为什么不满足?"(两比例z检验假设两组比例来自两个独立样本,但这里两个分类器是在同一批样本上测的,"分类器A的对错"和"分类器B的对错"在同一个样本上是相关的,独立性假设被违反)
- 深挖追问:"如果两个分类器的'都对'和'都错'样本数占了绝大多数,只有很少几个样本'不一致',McNemar检验还可靠吗?"(不一致样本数太小时,卡方近似的可靠性会下降,应该直接用精确的二项检验而不是卡方近似版本)

**常见坑:**
- 把两个分类器在同一测试集上的准确率直接当作两个独立比例做z检验,导致检验的统计效力被低估或者p值不准确——和知识点2的错误做法是同一类问题的不同表现形式。
- 不一致样本数很小时依然使用卡方近似版本的McNemar检验,而不是切换到精确二项检验版本。

---

## 4. 配对vs非配对比较的选择 —— 同一份数据,用错方法会漏掉真实差异

**定义与记号:** 04类知识点2已经建立"配对检验比独立样本检验更有效"这个原理。本知识点用同一份评测数据,具体演示"错误地把配对数据当作独立数据处理"和"正确地识别并利用配对结构"这两种分析方式,在结论上可能产生的实际差异——不是重复04类的推导,而是提供一个"同一份数据,两种分析路径,结论不同"的直接对照案例。

**一句话:** 04类知识点2用抽象的模拟数据证明了配对检验在数学上更有效;本知识点更进一步,直接展示"记错了数据结构、用了错误的分析方法"这个操作本身能不能真的把一个原本能被检测出来的真实差异,分析成"不显著"的错误结论。

**数学推导:** 对比同一份逐题评测数据:①正确方法——配对t检验(对差值d_i做单样本t检验,标准误只需要考虑Var(d_i)/n);②错误方法——独立样本t检验(把两列分数当作独立数据,标准误考虑Var(a_i)/n+Var(b_i)/n,不减去协方差项)。当a_i、b_i正相关时(题目难度这个共同因素驱动),Var(d_i)=Var(a_i)+Var(b_i)-2Cov(a_i,b_i)明显小于Var(a_i)+Var(b_i),配对方法的标准误更小、检验功效更高,同样的真实效应量下更容易被正确识别为显著。

**底层机制/为什么这样设计:** 为什么"识别数据的配对结构"这个步骤在实践中经常被跳过?因为配对结构不总是像"同一批病人先测量后测量"这样直观地体现在数据格式里——评测数据可能以"模型A的所有分数一列、模型B的所有分数另一列"这种表面上看起来像两组独立数据的格式存储和呈现,分析者如果不主动核实"这两列数据是不是按同一批题目对齐的",很容易默认按独立样本处理,这是数据格式和数据真实生成结构脱节导致的、容易被忽视的分析陷阱。

**AI研究/工程场景:** 模型评测报告、A/B测试分析里,拿到两列看起来"平行"的数字,分析前的第一步应该是确认这两列数字之间是否存在配对关系,而不是想当然地套用某个默认的检验模板——这是一个数据理解层面的诊断技能,比记住"配对检验公式是什么"更重要,也更容易在实践中被忽视。

**可运行例子:**
```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

n = 60
difficulty = rng.normal(70, 15, n)  # 每道题的基线难度, 同时影响两个模型的分数(0-100分制)
true_delta = 3.0  # 真实存在但不大的差异

score_A = difficulty + rng.normal(0, 5, n)
score_B = difficulty + true_delta + rng.normal(0, 5, n)
corr_AB = np.corrcoef(score_A, score_B)[0, 1]

# 正确: 配对t检验
t_paired, p_paired = stats.ttest_rel(score_B, score_A)

# 错误: 独立样本t检验(忽视配对结构)
t_indep, p_indep = stats.ttest_ind(score_B, score_A)

# 核心断言: 强相关场景下, 配对方法正确检测出显著差异, 错误方法漏掉了这个真实差异
assert corr_AB > 0.5, f"scores should be strongly correlated via shared question difficulty, got corr={corr_AB:.4f}"
assert p_paired < 0.05, f"paired test should correctly detect the real difference, got p={p_paired:.4f}"
assert p_indep > 0.05, f"the (incorrect) independent-sample test should miss this difference, got p={p_indep:.4f}"

print(f"corr(score_A, score_B) = {corr_AB:.4f}")
print(f"paired t-test:      t={t_paired:.4f}  p={p_paired:.6f}  (correctly significant)")
print(f"independent t-test: t={t_indep:.4f}  p={p_indep:.6f}  (incorrectly NOT significant -- same data, wrong method)")
```

**面试怎么问+追问链**(诊断真实数据新题型):
- Q:"给你一份评测数据,'模型A分数'和'模型B分数'两列数字,你怎么判断该用配对检验还是独立样本检验?"
- 追问1:"光看这两列数字本身能判断吗?"(不能仅凭数字本身判断,必须搞清楚数据的生成过程——这两列数字是不是按同一批题目/同一批评测样本对齐排列的,这个信息通常不在数字本身里,需要向提供数据的人确认或查看数据采集流程的文档)
- 深挖追问:"如果确认是配对数据,但错误地用了独立样本检验,会系统性地导致检验结论偏保守还是偏激进?"(当两个测量正相关时(评测场景里几乎总是正相关,题目难度是共同驱动因素),错误地用独立样本方法会高估标准误,导致检验偏保守——更容易把真实存在的差异错误地判断为"不显著",而不是相反)

**常见坑:**
- 只根据"数据是不是用两个独立的列存储"这种表面数据格式来判断该用什么检验方法,而不去核实数据背后真实的生成/采集过程是否存在配对关系。
- 把"配对检验总是比独立样本检验更好"过度泛化——如果两个测量之间实际上没有正相关,配对方法相对独立方法的优势会消失甚至反转,配对方法的优势是有条件的(条件是正相关),不是配对形式本身自动带来优势。

---

## 5. 评测集大小与CI宽度 —— 精度翻倍需要4倍题量

**定义与记号:** 用知识点1的bootstrap方法论,系统性扫描评测集规模n对置信区间宽度的影响——03类点估计理论已经证明标准误的一般形式是σ/√n,因此置信区间宽度应该按1/√n的速率随评测集规模增大而收窄,这是可以直接用数值扫描验证的定量规律。

**一句话:** 评测集从100道题增加到400道题(4倍),置信区间宽度不会缩小到1/4,而是缩小到大约1/2(因为收窄速率是1/√n不是1/n)——这个具体的数量关系,决定了"想要把评测结果的置信区间缩小一半,需要把评测集扩大4倍"这个在评测集设计阶段就该纳入考虑的资源规划规律。

**数学推导:** 标准误SE=σ/√n(02-03类已建立)。置信区间宽度∝SE∝1/√n。扩大评测集规模到k倍,SE缩小到1/√k倍,不是1/k倍——想要把不确定性减半,需要4倍的评测题目,想要减小到1/10,需要100倍的评测题目,边际成本急剧上升,这是评测集设计时"该测多少题"这个问题背后真正的约束。

**底层机制/为什么这样设计:** 为什么"评测集大小"和"置信区间宽度"之间是平方根关系,而不是线性关系?这是中心极限定理/标准误公式本身的数学结构决定的,不是评测这个具体场景的特殊性质——这正是把"评测分数"当作和A/B测试指标本质相同的对象来对待的价值所在:一旦意识到这只是标准误公式的一个应用实例,06类样本量计算的整套方法论直接搬过来就能用,不需要为"评测集"这个具体场景重新发明一套精度分析理论。

**AI研究/工程场景:** 设计一个新的评测benchmark时,"应该收集多少道题目"这个问题,本质上和06类知识点1"A/B测试样本量计算"是同一类问题——如果知道大致想要达到的置信区间精度,可以直接反推出所需的题目数量下限,而不是凭感觉定一个"看起来够多"的数字。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

sizes = [50, 100, 200, 400, 800, 1600]
n_boot = 1000
widths = []

for n_q in sizes:
    difficulty = rng.beta(2, 2, n_q)
    scores = rng.binomial(1, difficulty)
    boot_means = np.array([scores[rng.integers(0, n_q, n_q)].mean() for _ in range(n_boot)])
    ci = np.percentile(boot_means, [2.5, 97.5])
    widths.append(ci[1] - ci[0])

# 核心断言1: 区间宽度应该随评测集规模单调递减
assert all(widths[i] > widths[i + 1] for i in range(len(widths) - 1)), f"CI width should shrink monotonically: {widths}"

# 核心断言2: log-log拟合斜率应该接近理论值-0.5
log_n, log_w = np.log(sizes), np.log(widths)
slope, _ = np.polyfit(log_n, log_w, 1)
assert -0.6 < slope < -0.35, f"log-log slope should be close to the theoretical -0.5, got {slope:.4f}"

# 核心断言3: 规模翻4倍, 宽度应该约减半
ratio_4x = widths[sizes.index(400)] / widths[sizes.index(100)]
assert 0.4 < ratio_4x < 0.65, f"quadrupling the eval set size should roughly halve the CI width, got ratio={ratio_4x:.4f}"

print(f"CI widths across sizes {sizes}: {[round(w, 4) for w in widths]}")
print(f"log-log fitted slope = {slope:.4f}  (theory: -0.5)")
print(f"width(400) / width(100) = {ratio_4x:.4f}  (theory: ~0.5)")
```

**面试怎么问+追问链**(规模递增轴):
- Q:"现有的100道题评测集,置信区间宽度是±2分,业务方希望缩小到±1分,应该怎么办?"
- 追问1:"需要把评测集扩大到多少道题?"(区间宽度要缩小到一半,根据1/√n的关系,需要把评测集规模扩大到4倍,也就是400道题,不是200道题——这是一个常见的直觉误区)
- 深挖追问:"如果继续要求把区间宽度从±1分进一步缩小到±0.1分(10倍精度),评测集需要扩大多少倍?"(需要100倍,也就是40000道题——这个数字级别的题目量在实践中往往已经超出合理的评测集设计范围,这时候更现实的思路是不再单纯依赖扩大评测集规模,而要考虑降低单题打分本身的噪声,或者采用方差削减技术(07类知识点3的CUPED思想在评测场景下的类比应用))

**常见坑:**
- 想要把置信区间宽度减半,直觉地认为把评测集规模也加倍就够了,忽视了1/√n这个平方根关系,实际上需要4倍的规模。
- 无限制地追求"更大的评测集"来缩小置信区间,而不考虑边际收益递减(06类知识点3功效曲线"边际收益递减"的同一原理在这里的复现)和实际评测成本。

---

## 6. "刷榜"的隐藏多重比较 —— 报告最大值本身就是一种选择偏差

**定义与记号:** "刷榜"(benchmark hacking):在验证集上尝试大量不同的超参数组合/模型变体/prompt设计,只挑选并报告表现最好的那一个配置——这本质上是05类知识点1"多重比较问题"和06类知识点4"窥探问题"的一个具体化身:每尝试一个新配置并在同一个验证集上打分,就相当于做了一次新的"检验",如果只报告"最好的那次"的分数,这个分数系统性地高估了这个配置的真实泛化性能。

**一句话:** 如果在验证集上试了100个超参数组合、只挑分数最高的那个汇报,这个"最高分"已经不再是对"这个配置真实性能"的无偏估计,而是"100次独立测量里的最大值"这个统计量,而最大值统计量系统性地高于任何单次测量的期望值,这个高估幅度随尝试次数增加而增大,是可以直接数值算出来的。

**数学推导:** 设每个候选配置的真实性能都相同(都是μ,试验之间没有真实差异,只是随机噪声导致观测分数波动,scores~N(μ,σ²)),尝试m个配置,汇报max(scores₁,...,scores_m)。max统计量的期望E[max]>μ,且随m增大而增大(极值统计的基本性质,增长速率大致是ln(m)量级,不是常数、也不是线性)。可以直接数值模拟验证:固定σ,扫描不同的m,测量E[max]-μ这个"虚假提升"如何随m变化。

**底层机制/为什么这样设计:** 为什么"报告最大值",即使每个候选配置的真实性能完全相同,也会系统性地产生一个看起来"更优"的假象?因为噪声本身是对称分布的,但"取最大值"这个操作只会挑中那些**恰好**因为噪声而偏高的观测,不会挑中偏低的观测——这是一种被动的、非故意的选择偏差(10类知识点2选择偏差的具体形式),不需要任何主观作弊意图,纯粹是"挑选评价最好的那个"这个操作本身的数学后果。

**AI研究/工程场景:** "我们尝试了A、B、C等多种方法,最终采用效果最好的方法X,在benchmark上达到SOTA"这类叙述,如果没有说明"尝试了多少种配置"这个信息、也没有在独立的held-out测试集上重新验证配置X的性能,读者没有办法判断这个SOTA数字里有多少是真实提升、有多少是"刷"出来的虚假提升——解决方案是标准的机器学习实践"验证集选模型、独立测试集报告最终性能",本质上就是05类"多重比较后用独立数据重新验证"这个原则的具体应用。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

true_mu = 0.70
sigma = 0.03

# 扫描不同尝试次数m下, "报告验证集最大值"这个操作系统性高估了多少
inflations = {}
for m in [1, 5, 20, 100]:
    max_scores = [rng.normal(true_mu, sigma, m).max() for _ in range(3000)]
    inflations[m] = np.mean(max_scores) - true_mu

# 核心断言1: 高估幅度应该随尝试次数单调增大
m_list = sorted(inflations)
assert all(inflations[m_list[i]] < inflations[m_list[i + 1]] for i in range(len(m_list) - 1)), \
    f"inflation should increase monotonically with more trials: {inflations}"
assert inflations[100] > 0.05, f"with 100 trials, the validation-max inflation should be substantial, got {inflations[100]:.4f}"
assert abs(inflations[1]) < 0.01, f"with only 1 trial (no selection), inflation should be near 0, got {inflations[1]:.4f}"

# 核心断言2: 独立held-out测试集重新评估"选出的配置", 应该基本无偏(重复实验框架, 不是单次抽样比较)
m = 100
n_repeats = 2000
val_inflations, holdout_inflations = [], []
for _ in range(n_repeats):
    val_scores = rng.normal(true_mu, sigma, m)
    val_best = val_scores.max()
    holdout = rng.normal(true_mu, sigma)  # 独立测试集: 对"选出的这一个配置"重新测一次
    val_inflations.append(val_best - true_mu)
    holdout_inflations.append(holdout - true_mu)

mean_val_inflation = np.mean(val_inflations)
mean_holdout_inflation = np.mean(holdout_inflations)
assert mean_val_inflation > 0.05, f"repeated validation-max inflation should be consistently substantial, got {mean_val_inflation:.4f}"
assert abs(mean_holdout_inflation) < 0.01, f"held-out re-evaluation should be essentially unbiased, got {mean_holdout_inflation:.4f}"

print(f"inflation E[max]-true_mu by number of trials m: {[(m, round(v, 4)) for m, v in inflations.items()]}")
print(f"mean validation-max inflation (m=100, over {n_repeats} repeats) = {mean_val_inflation:.4f}")
print(f"mean held-out re-evaluation inflation (same repeats) = {mean_holdout_inflation:.4f}  (essentially unbiased)")
```

**面试怎么问+追问链**(方案批判迭代轴,呼应05类/06类):
- Q:"团队尝试了50种不同的prompt设计,在验证集上选出表现最好的一种,报告的分数比baseline高了3个百分点,这个提升可信吗?"
- 追问1:"这个场景和05类的多重比较问题、06类的窥探问题有什么共同点?"(本质上是同一类问题——都是"反复尝试/检验、只报告最好的那次结果"这个操作模式导致的系统性偏差,05-06类分别在"多指标假设检验"和"实验时间维度反复查看"这两个具体场景下讨论过,这里是第三个具体化身:"超参数/配置搜索空间维度"上的反复尝试)
- 深挖追问:"怎么正确地报告这50种prompt设计里最好的那个的真实性能?"(标准做法:用一个和验证集完全独立、配置搜索过程完全没有接触过的held-out测试集,对"验证集选出的最佳配置"重新评估一次,这个独立测试集上的分数才是对真实性能相对无偏的估计;如果没有独立测试集,至少应该报告"尝试了50种配置"这个信息,让读者自己判断可能的虚假膨胀幅度)

**常见坑:**
- 论文/报告里只汇报"最好的配置得到了多少分",完全不提及"一共尝试了多少种配置才选出这一个",让读者无法评估潜在的选择偏差幅度。
- 用同一个验证集既做模型选择(挑出表现最好的配置)又做最终性能报告——这是刷榜问题的根源,标准的机器学习实践要求验证集(用于选择)和测试集(用于报告)严格分离,重复使用同一份数据做这两件事,不管子采样多少次都无法消除这个偏差。

---

下一篇:[15-ranking-systems.md](15-ranking-systems.md) —— 从"单模型评分"转向"多模型排位"——Elo、Bradley-Terry模型、以及`learning/llm-judge-arena/`模块Chatbot Arena排行榜背后的统计机制。
