# 21 · 模拟终面 Capstone

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第21个"知识点",而是把前20个分类文件里最容易在真实终面被连续追问的几个点——04类知识点1(t检验)、03类知识点4(p值真实含义)、06类知识点1(功效分析)、05类知识点1(多重比较)、08类知识点2(随机化如何消除混淆)、13类知识点1(贝叶斯A/B测试)、10类知识点5(诊断真实数据)——串成一场真实计时不到1小时、跨5个板块的连续追问。全篇每一处判断都在仓库根目录 `.venv` 里真实跑过,包括候选人最初给出的、后续被逐层拆穿的"不够严谨"的版本——终面很少在第一个答案就结束,追问链本身才是考察的核心。

---

## 题目(面试官,0:00)

> "你在一个frontier lab负责训练pipeline的优化。团队最近对默认优化器的一个超参数做了改动,想知道要不要合并进主训练分支。你被要求给出评估结论。说说你会怎么做,给出你的结论和依据。"

---

## 候选人的初版汇报(0:00 – 0:08)

**候选人:** 我们跑了5个随机种子的对比实验——每个种子分别用baseline配置和新配置各训练一次,记录最终验证集loss(paired设计,同一个种子的两次训练共享数据顺序/初始化随机性之外的其他随机因素,能提供比独立分组更强的功效,这是04类知识点2配对设计的直接应用)。5个种子里4个新配置更好、1个更差,平均改进有统计显著性,建议合并。

```python
import numpy as np
from scipy import stats

# 5个种子的最终验证loss(baseline vs 新优化器配置), 数值越低越好
baseline = np.array([2.51, 2.48, 2.53, 2.49, 2.52])
new_config = np.array([2.43, 2.41, 2.45, 2.41, 2.54])

diff = new_config - baseline  # 负数=变好(loss下降)
n_better = int(np.sum(diff < 0))
n_worse = int(np.sum(diff > 0))

t_stat, p_value = stats.ttest_rel(new_config, baseline)
mean_diff = diff.mean()
cohens_d = mean_diff / diff.std(ddof=1)

# 核心断言1: 5个种子里4个改进, 1个变差(候选人汇报的原始现象)
assert n_better == 4 and n_worse == 1

# 核心断言2: 配对t检验在alpha=0.05水平下确实"显著"
assert p_value < 0.05

print(f"diff per seed: {np.round(diff, 3)}")
print(f"n_better={n_better}  n_worse={n_worse}  mean_diff={mean_diff:.4f}")
print(f"paired t-test: t={t_stat:.4f}  p={p_value:.4f}  (n=5, df=4)")
print(f"observed Cohen's d={cohens_d:.4f}")
print("candidate's conclusion: p<0.05, statistically significant, recommend merging")
```

候选人说完这段,面试官没有立刻表态,而是开始逐层拆解——接下来6轮追问,每一轮都建立在前一轮的数字之上,不是孤立提问。

---

## 追问1(0:08 – 0:16):"显著"到底是什么意思?

**面试官:** "你说'统计显著',p<0.05具体是什么意思?能不能精确说一下,这是不是就代表'新方法一定有效'?"

**候选人:** 不是。p值的精确定义(03类知识点4已经展开过这个常见误用):**假设原假设为真(新配置和baseline真的没有差别),观测到当前这么极端、或者比当前更极端的结果的概率**。p=0.0416的意思是"如果新配置真的毫无效果,像这样5个种子里至少4个'看起来更好'、且均值差距这么大的模式,纯属巧合出现的概率大约是4.16%"——这是一个关于"数据在原假设下有多意外"的陈述,不是"原假设为真的概率"(那需要贝叶斯框架,后面会展开),更不是"效果有多大"或者"效果有多大概率是真的"的度量。

**候选人补充一个具体反例(数值验证"小概率≠不可能"):** 如果新配置真的毫无效果,把整个实验重复很多次,应该能看到"4/5个种子看起来更好、且p<0.05"这个模式偶尔也会纯属巧合地出现。

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

n = 5
n_sims = 50000
n_better = np.zeros(n_sims, dtype=int)
sig = np.zeros(n_sims, dtype=bool)
for i in range(n_sims):
    d = rng.normal(0, 0.05, n)  # 真实效应=0(纯噪声), 噪声标准差取和实际观测同量级
    n_better[i] = np.sum(d < 0)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    p = 2 * stats.t.sf(np.abs(t), df=n - 1)
    sig[i] = p < 0.05

p_sig_alone = sig.mean()
p_pattern = ((n_better >= 4) & sig).mean()

# 核心断言1: p<0.05的边际概率应该接近名义值0.05(sanity check, 呼应03类知识点4)
assert abs(p_sig_alone - 0.05) < 0.01

# 核心断言2: "4/5更好 且 p<0.05"这个候选人观测到的具体模式, 在真实效应=0时依然有实打实的非零概率出现
assert 0.01 < p_pattern < 0.05

print(f"P(p<0.05 | true effect=0) = {p_sig_alone:.4f}  (close to nominal 0.05, as expected)")
print(f"P(>=4/5 seeds look better AND p<0.05 | true effect=0) = {p_pattern:.4f}")
print("=> even with ZERO true effect, this exact pattern happens by pure chance about 1 time in 40")
print("=> p<0.05 shifts the odds, it does not prove the effect is real")
```

**面试官追问1:** "所以这个p=0.0416,应该怎么正确使用,不是直接说'显著就是有效'?"（候选人:应该说"在原假设为真的前提下,这个数据本身相当不寻常,这构成了怀疑原假设的证据",而不是"证明了新配置有效"——p值只回答"数据和原假设兼不兼容"这一个问题,不回答"效应有多大"或者"决策该怎么做",后两个问题需要接下来的功效分析和贝叶斯框架分别回答）

---

## 追问2(0:16 – 0:26):5个种子够不够?

**面试官:** "5个种子做出'合并还是不合并'这么大的决策,样本量够吗?"

**候选人:** 这需要功效分析(06类知识点1)。关键问题反过来问:如果真实效应量是一个"合理"的量级(比如Cohen's d=0.5,中等偏上的效应,对优化器超参数调整这类改动是一个不算离谱的假设),n=5的配对检验能有多大概率检测出来?

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

n = 5
true_d = 0.5  # 假设的真实效应量(中等偏上)
n_sims = 20000

sim_diffs = rng.normal(true_d, 1.0, (n_sims, n))
t_stats = sim_diffs.mean(axis=1) / (sim_diffs.std(axis=1, ddof=1) / np.sqrt(n))
p_values = 2 * stats.t.sf(np.abs(t_stats), df=n - 1)
rejected = p_values < 0.05
power = rejected.mean()

# winner's curse: 在"恰好显著"的那些模拟里, 观测到的效应量比真实值夸大多少
observed_d = sim_diffs.mean(axis=1) / sim_diffs.std(axis=1, ddof=1)
observed_d_given_sig = observed_d[rejected].mean()

# 核心断言1: n=5, d=0.5时功效很低(远低于常规80%目标)
assert power < 0.20, f"power at n=5 should be low, got {power:.4f}"

# 核心断言2(呼应候选人观测到的|d|=1.32): "恰好显著"的那部分模拟, 观测效应量应该比真实值明显夸大
assert observed_d_given_sig > 2 * true_d, f"significant-only observed d should be inflated well above true_d={true_d}"

print(f"power at n=5, assumed true Cohen's d={true_d}: {power:.4f}")
print(f"mean observed d among ALL simulations: {observed_d.mean():.4f}")
print(f"mean observed d among SIGNIFICANT-ONLY simulations: {observed_d_given_sig:.4f}  (true_d={true_d})")

for n_try in [5, 10, 20, 30, 50, 64]:
    sd = rng.normal(true_d, 1.0, (5000, n_try))
    ts = sd.mean(axis=1) / (sd.std(axis=1, ddof=1) / np.sqrt(n_try))
    ps = 2 * stats.t.sf(np.abs(ts), df=n_try - 1)
    pw = (ps < 0.05).mean()
    print(f"n={n_try}: power={pw:.3f}")
print("=> reaching a conventional 80% power target needs roughly 50+ seeds here, not 5")
```

**候选人的解读:** 两个问题。第一,n=5在d=0.5这个假设下,功效只有约14%——意味着即使新配置真的有中等效应,5个种子里有约86%的概率**根本检测不出显著性**,这次"侥幸"显著更多说明运气好或者真实效应比假设的更大,不能反过来说"5个种子就足够了"。第二个更值得警惕的现象是**赢家诅咒(winner's curse)**:模拟显示,在低功效场景下,"恰好达到显著"的那些实验,报告出来的效应量系统性地远大于真实效应量(这里夸大到3倍多)——候选人最初观测到的\|d\|=1.32这个"看起来很大"的效应量,很可能就是这个统计假象的一次具体体现,不是新配置真的有这么强的效果。

**面试官追问1:** "那应该建议怎么做,直接说样本不够、不能下结论就完了吗?"（候选人:更准确的建议是——现在的证据"值得进一步跑更多种子",而不是"已经证明有效"或者"已经证明无效";如果条件允许,应该补充实验把种子数扩大到功效分析给出的量级,而不是拿5个种子的结果直接下最终结论）

---

## 追问3(0:26 – 0:34):只挑了一个指标汇报?

**面试官:** "你刚才只提到了最终验证loss这一个指标。训练过程中,你们是不是同时也在跟踪其他指标,比如困惑度(perplexity)、下游任务准确率、训练稳定性这些?"

**候选人(如实承认):** 是的,同时跟踪了5个指标,只有验证loss这一个在5个种子上显著,其余几个没有显著差异,汇报时只提到了显著的这一个。

**面试官:** "这个做法本身有什么统计问题?"

**候选人:** 这正是05类知识点1的多重比较问题,只是这次"测多次"发生在**指标**这个维度上,不是重复看同一个指标多次。如果5个指标里新配置和baseline其实**完全没有任何真实差异**,只是每个指标各自的5%假阳性率赶巧碰上了,"5个指标里至少有1个显著"这件事发生的概率远高于5%。

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

n_seeds = 5
n_metrics = 5  # loss/perplexity/下游acc/训练稳定性/... 一共5个跟踪指标
n_trials = 20000

any_significant = np.zeros(n_trials, dtype=bool)
n_sig_per_trial = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    diffs = rng.normal(0, 1.0, (n_metrics, n_seeds))  # 完全零假设: 5个指标全部没有真实差异
    t_stats = diffs.mean(axis=1) / (diffs.std(axis=1, ddof=1) / np.sqrt(n_seeds))
    p_vals = 2 * stats.t.sf(np.abs(t_stats), df=n_seeds - 1)
    sig = p_vals < 0.05
    any_significant[i] = sig.any()
    n_sig_per_trial[i] = sig.sum()

empirical_fwer = any_significant.mean()
theory_fwer = 1 - (1 - 0.05) ** n_metrics

# 核心断言1: 实证族错误率(family-wise error rate)应该接近解析公式1-(1-alpha)^m
assert abs(empirical_fwer - theory_fwer) < 0.02

# 核心断言2: 这个族错误率应该明显超过单指标的名义alpha=0.05
assert empirical_fwer > 4 * 0.05

print(f"empirical P(at least 1 of {n_metrics} metrics significant | true null everywhere) = {empirical_fwer:.4f}")
print(f"theory 1-(1-0.05)^{n_metrics} = {theory_fwer:.4f}")
print(f"mean number of 'significant' metrics per trial: {n_sig_per_trial.mean():.4f}  (each metric alone: 0.05)")
print("=> tracking 5 metrics and reporting only the significant one inflates the effective false-positive rate ~4.5x")
```

**候选人的更正:** 单看loss这一个指标的p=0.0416,如果考虑到实际上做了5次检验、只挑了显著的那次汇报,真正的族错误率(至少一个指标"意外"显著的概率)在纯零假设下大约是22.6%,不是5%——这个结果本身的可信度应该打折扣,更严谨的做法是要么对5个指标做Bonferroni校正(05类知识点2)后重新判断,要么把"loss改进"当作需要额外独立验证的假设,而不是当作已经证实的结论直接汇报。

---

## 追问4(0:34 – 0:44):两组实验是同时跑的吗?

**面试官:** "5个baseline和5个新配置的训练,是在完全一样的软硬件环境下、同一批随机穿插跑的吗?"

**候选人(如实回忆):** ……不完全是。baseline的5次训练是上个月在旧集群上跑的,新配置的5次训练是这周在刚升级完CUDA版本的新集群上跑的。

**面试官:** "这个时间和环境上的差异,可能带来什么问题?"

**候选人:** 这正是08类知识点2随机化的核心价值——**真正的随机化,能保证除了被研究的处理变量之外,其他一切潜在的混淆因素在两组之间的分布是均衡的**。这里baseline和新配置不只是"优化器配置"不同,还完全绑定了"哪个集群/哪个CUDA版本/哪个时间段"这几个额外变量,一旦集群升级本身(比如新硬件的数值精度、kernel实现差异)就会让loss产生一个和优化器改动完全无关的系统性偏移,这个偏移会被误判成"新配置的效果"。

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

n = 5
true_optimizer_effect = 0.0       # 假设优化器改动本身真实效果=0(用来隔离检验"集群混淆"单独能造成多大假阳性)
cluster_confound_effect = -0.06   # 新集群本身(和优化器无关)让loss平均低0.06

# 场景A: 完全不随机化 -- baseline固定在旧集群, 新配置固定在新集群(候选人实际做法)
n_trials = 20000
naive_sig = np.zeros(n_trials, dtype=bool)
naive_diff = np.zeros(n_trials)
for i in range(n_trials):
    baseline_old_cluster = rng.normal(2.50, 0.05, n)
    new_on_new_cluster = rng.normal(2.50 + true_optimizer_effect + cluster_confound_effect, 0.05, n)
    d = new_on_new_cluster - baseline_old_cluster
    t_stat = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    p = 2 * stats.t.sf(np.abs(t_stat), df=n - 1)
    naive_sig[i] = p < 0.05
    naive_diff[i] = d.mean()

# 场景B: 真正随机化 -- 10次训练(5 baseline+5新配置), 随机决定每一次落在哪一代集群上
n_trials2 = 20000
rand_sig = np.zeros(n_trials2, dtype=bool)
for i in range(n_trials2):
    method = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])       # 0=baseline, 1=新配置, 各5次
    cluster_gen = rng.permutation(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))  # 集群代次独立随机分配, 与method无关
    losses = 2.50 + true_optimizer_effect * method + cluster_confound_effect * cluster_gen + rng.normal(0, 0.05, 10)
    t_stat, p = stats.ttest_ind(losses[method == 1], losses[method == 0])
    rand_sig[i] = p < 0.05

fpr_naive = naive_sig.mean()
fpr_randomized = rand_sig.mean()

# 核心断言1: 不随机化(集群和方法完全绑定)时, 即使优化器真实效果=0, 假阳性率也会被集群混淆严重推高
assert fpr_naive > 0.20, f"confounded false-positive rate should be well above nominal 0.05, got {fpr_naive:.4f}"

# 核心断言2: 真正随机化后, 假阳性率回到接近名义0.05
assert abs(fpr_randomized - 0.05) < 0.02, f"randomized false-positive rate should be near nominal 0.05, got {fpr_randomized:.4f}"

print(f"NOT randomized (baseline=old cluster, new_config=new cluster): P(false 'significant') = {fpr_naive:.4f}")
print(f"    mean apparent improvement (entirely driven by the cluster confound, true optimizer effect=0): {naive_diff.mean():.4f}")
print(f"randomized cluster assignment: P(false 'significant') = {fpr_randomized:.4f}  (back near nominal 0.05)")
print("=> without randomization, a silent infrastructure change alone produced a 'significant improvement' 30%+ of the time")
```

**候选人的结论:** 即使真实优化器效果是0,单单"新集群本身让loss平均低0.06"这一个混淆因素,就能让约30.8%的重复实验"看起来显著"——远超候选人最初汇报里那个让人安心的5%假阳性率印象。这不是"结果一定是假的",而是**当前的实验设计,原则上就无法把"优化器改动的效果"和"集群升级的效果"区分开**,必须补做一次真正随机化的对照(同一批集群、随机决定每次训练用哪个配置)才能让"p<0.05"这个数字重新变得可信。

---

## 追问5(0:44 – 0:52):能不能不要非黑即白?

**面试官:** "显著/不显著这种二元判断,业务方经常误解成'一定有效'或者'完全没用'。能不能换一种方式,直接说'这个改进有多大把握是真的'?"

**候选人:** 可以用13类知识点1的贝叶斯框架,直接对最初那5个种子的数据算一个后验概率:给定观测到的数据,新配置真的比baseline好的概率是多少。用Normal-Normal共轭:先验设为"效应量在0附近、典型优化器改动的量级大概在±0.08这个范围内、不预设方向"(体现"改动可能有用也可能没用,但不太可能是天翻地覆的巨大效应"这个合理的先验判断)。

```python
import numpy as np
from scipy import stats

diff = np.array([-0.08, -0.07, -0.08, -0.08, 0.02])  # 复用候选人最初汇报的5个种子差值
n = len(diff)
obs_mean = diff.mean()
obs_sd = diff.std(ddof=1)
se = obs_sd / np.sqrt(n)

def posterior_p_better(tau0, prior_mean=0.0):
    posterior_var = 1.0 / (1.0 / tau0 ** 2 + 1.0 / se ** 2)
    posterior_mean = posterior_var * (prior_mean / tau0 ** 2 + obs_mean / se ** 2)
    posterior_sd = np.sqrt(posterior_var)
    p_better = stats.norm.cdf(0, loc=posterior_mean, scale=posterior_sd)
    return posterior_mean, posterior_sd, p_better

pm_main, psd_main, pb_main = posterior_p_better(tau0=0.08)
ci_low, ci_high = stats.norm.ppf([0.025, 0.975], loc=pm_main, scale=psd_main)

# 核心断言1: 主先验下, 后验应该给出"新配置更好"的高概率陈述
assert pb_main > 0.99, f"posterior P(better) should be high under the main prior, got {pb_main:.4f}"

# 核心断言2: 95%可信区间不应该包含0(方向性结论明确), 但区间本身应该相当宽(反映n=5的真实不确定性)
assert ci_high < 0 and (ci_high - ci_low) > 0.05

print(f"observed: mean_diff={obs_mean:.4f}  se={se:.4f}")
print(f"main prior (tau0=0.08): posterior_mean={pm_main:.4f}  P(true diff<0, i.e. new config truly better)={pb_main:.4f}")
print(f"95% credible interval: [{ci_low:.4f}, {ci_high:.4f}]")

print("\nsensitivity to prior skepticism:")
for tau0 in [0.08, 0.03, 0.02, 0.01]:
    pm, psd, pb = posterior_p_better(tau0)
    print(f"  tau0={tau0}: posterior_mean={pm:.4f}  P(better)={pb:.4f}")
```

**候选人的解读:** 在"改动量级大概在±0.08"这个合理先验下,后验给出P(新配置真的更好)≈99.8%,95%可信区间是[-0.092, -0.017]——不包含0,方向性结论明确,但区间本身相当宽,如实反映了n=5这个小样本自带的不确定性。**候选人主动补充一个稳健性检查(面试官没问,但这是负责任汇报的一部分):** 换成一个更保守/更怀疑的先验(相信大多数超参数微调本身效果接近0),P(更好)会从99.8%降到91%左右,但方向性结论没有反转——说明这个结论对先验选择有一定的稳健性,不是"随便换个先验就能翻盘"的脆弱结果,但汇报"99.8%的把握"这个具体数字时应该说明它依赖于先验假设,不是一个不需要任何前提的客观数字。

**面试官追问1:** "这个99.8%和追问2里算出来的14%功效,是不是矛盾的说法?"（候选人:不矛盾,两者回答的是完全不同的问题——14%功效是一个**实验设计阶段**的问题("如果真实效应确实是d=0.5这个量级,这个实验设计有多大概率能捕捉到它"),99.8%是一个**拿到具体这批数据之后**的问题("给定这批具体观测值和这个先验,真实效应为负的后验概率是多少")。功效低说明"这个实验设计本身不太可靠,容易错过真实效应,也容易在恰好检测到时夸大效应量"(追问2的赢家诅咒),这个警告在拿到后验概率之后依然成立——后验概率高不代表实验设计问题被抹平了,只是从"这批具体数据"出发能得到的诚实结论）

---

## 最后一题(0:52 – 0:58):这个种子为什么长这样?

**面试官:** "给你这5个种子完整的训练loss曲线,其中第5个种子在训练中途有一段异常。这是真实的方差,还是训练脚本有bug?"

```
第5个种子(异常)在第100步附近的loss片段(其余4个种子在同一区间平稳下降, 未展示):
step  95: 0.4183
step  96: 0.4126
step  97: 0.4084
step  98: 0.4043
step  99: 0.4012
step 100: 0.7513   <- 单步暴涨
step 101: 0.7217
step 105: 0.6489
step 110: 0.5762
step 120: 0.4629
step 130: 0.3985
(checkpoint保存/恢复发生在每100步一次)
```

**候选人:** 先不下结论,列出能区分"真实方差"和"脚本bug"的几个具体、可检验的证据点,而不是凭印象猜。

```python
import numpy as np
rng = np.random.default_rng(42)

T = 200
checkpoint_step = 100  # 每100步存一次checkpoint

def clean_curve(seed_noise_scale=0.01):
    t = np.arange(T)
    trend = 1.0 * np.exp(-t / 60.0) + 0.3
    noise = rng.normal(0, seed_noise_scale, T)
    return trend + noise

def buggy_curve():
    t = np.arange(T)
    trend = 1.0 * np.exp(-t / 60.0) + 0.3
    noise = rng.normal(0, 0.01, T)
    curve = trend + noise
    # checkpoint重载后学习率调度状态未正确恢复(相当于warmup被意外重新触发):
    # loss短暂大幅抬升, 随后需要若干步重新收敛, 不是单步噪声
    bug_bump = np.zeros(T)
    bump_size, recovery_len = 0.35, 35
    for i in range(recovery_len):
        if checkpoint_step + i < T:
            bug_bump[checkpoint_step + i] = bump_size * np.exp(-i / 12.0)
    return curve + bug_bump

seeds_loss = [clean_curve() for _ in range(4)] + [buggy_curve()]

def max_single_step_jump(curve):
    jumps = np.diff(curve)
    idx = int(np.argmax(jumps))
    return jumps[idx], idx + 1

jump_info = [max_single_step_jump(c) for c in seeds_loss]
healthy_jumps = [mag for mag, step in jump_info[:4]]
healthy_jump_steps = [step for mag, step in jump_info[:4]]
buggy_jump_mag, buggy_jump_step = jump_info[4]

# 证据1: 跳变幅度 -- 异常种子的最大单步跳变应该远超其余4个种子的噪声水平
assert buggy_jump_mag > 5 * np.mean(healthy_jumps)

# 证据2: 跳变位置 -- 异常种子的跳变应该恰好落在checkpoint边界, 不是随机位置
assert abs(buggy_jump_step - checkpoint_step) <= 1

# 证据3: 健康种子的(小幅度)跳变位置应该分散, 不集中在checkpoint边界附近(对照组: 真实噪声不挑"日子")
assert not all(abs(s - checkpoint_step) <= 5 for s in healthy_jump_steps)

# 证据4: 恢复形态 -- 真实的单步噪声下一步就应该回到趋势线; bug应该表现为连续多步的逐渐衰减恢复
t = np.arange(T)
trend = 1.0 * np.exp(-t / 60.0) + 0.3
residual_after_jump = seeds_loss[4][buggy_jump_step:buggy_jump_step + 10] - trend[buggy_jump_step:buggy_jump_step + 10]
assert np.sum(residual_after_jump[:5] > 0.05) >= 4

print(f"healthy seeds' max jump (mean of 4): {np.mean(healthy_jumps):.4f}, at steps {healthy_jump_steps}")
print(f"seed 5's max jump: {buggy_jump_mag:.4f}, at step {buggy_jump_step}  (checkpoint boundary={checkpoint_step})")
print(f"residual (actual - trend) for 10 steps after the jump: {np.round(residual_after_jump, 4)}")
print("=> jump magnitude: ~12x healthy noise level")
print("=> jump location: exactly at the checkpoint boundary, while healthy seeds' jumps scatter elsewhere")
print("=> recovery shape: stays elevated for 5+ steps, decaying gradually -- not a memoryless 1-step noise spike")
print("diagnosis: checkpoint/LR-schedule state not properly restored, NOT genuine seed-to-seed variance")
```

**候选人的完整诊断链条:** 三个独立证据都指向同一个结论。第一,跳变幅度是其余种子噪声水平的10倍以上,远超"这只是运气差的一个种子"能解释的范围。第二,也是最关键的一条——跳变发生的**具体位置**恰好精确对齐checkpoint保存/恢复的边界,而其余4个种子各自的(小幅度)波动峰值分布在完全不同、看起来随机的位置,如果这真的是"纯粹的随机方差",没有理由所有异常都恰好挑在同一个系统性事件(checkpoint边界)发生。第三,跳变后的恢复是一个持续5步以上、逐渐衰减的过程,而不是"这一步偶然差、下一步就恢复正常"——这种"有记忆的多步恢复"形态,是学习率调度状态被意外重置(比如没有正确保存/恢复optimizer的调度器state_dict,导致重新加载checkpoint后warmup被意外触发)这类工程bug的典型特征,不是独立同分布噪声该有的样子。**结论:这是训练脚本在checkpoint恢复逻辑上的bug,不是第5个种子运气差**,不应该把这个种子的数据直接丢弃后当作"5个干净种子"重新汇报,而应该先定位并修复checkpoint恢复逻辑,再重新产生这个种子的数据。这正是10类知识点5"诊断真实数据"新题型的核心考察点——不是套用某个假设检验模板,而是从数据本身的具体异常特征反推系统行为。

---

## 复盘小结(0:58 – 1:00)

**候选人主动总结(不等面试官问):** 这场终面走过的完整链条是——最初的"4/5种子更好、p<0.05、建议合并"这个初版结论,在连续6轮追问里被逐层加固/修正:①p值被追问到精确定义,澄清"显著不等于一定有效"(03类知识点4);②样本量被追问到功效只有14%,揭示了赢家诅咒可能夸大了观测到的效应量(06类知识点1);③被追问出"只挑显著指标汇报"的隐藏多重比较,真实假阳性率被推高到22.6%(05类知识点1);④被追问出"集群升级"这个隐藏的非随机化混淆,真实假阳性率在最坏情况下高达30.8%(08类知识点2);⑤在澄清了以上问题之后,依然可以给出一个概率化、且对先验选择保持基本稳健的贝叶斯结论——P(真实更好)从99.8%(弱先验)到91%(怀疑先验)不等,但方向性判断没有反转(13类知识点1);⑥最后一题跳出假设检验框架,用具体的数据特征(幅度/位置/恢复形态三重证据)诊断出一个真实的系统bug,而不是把异常简单归因为"运气"(10类知识点5)。

这场模拟终面想具体展示的,是全系列反复出现的同一条纪律在"评估一个真实训练改动"这个场景下的完整应用:**每一层追问都不是在挑刺,而是在问"这个判断经不经得起更仔细的检验"**——初版结论没有任何一步是刻意犯错误来制造戏剧性,5个种子、4/5改进、p<0.05,这是任何人第一次做这类分析时都可能给出的诚实汇报;真正区分"熟练"和"生疏"的,不是第一版汇报的质量,而是被追问之后,能不能用数字(不是空话)具体回应每一层质疑,并且在质疑成立时如实修正结论,而不是为了"显得自己一开始就是对的"而回避。

---

*本篇全部代码块在仓库根目录 `.venv` 真实测试验证,固定随机种子`np.random.default_rng(42)`。综合运用了paired t检验、蒙特卡洛功效模拟、赢家诅咒的直接数值展示、多重比较族错误率的解析公式与蒙特卡洛交叉验证、隐藏混淆变量的对照模拟(不随机化vs随机化两种场景重新计算假阳性率)、Normal-Normal共轭贝叶斯后验(含先验敏感性分析)、以及基于幅度/位置/恢复形态三重证据的异常诊断。*
