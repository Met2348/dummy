# 10 · 真实陷阱案例集(Real-World Statistical Traps)

> 总览见 [00-roadmap.md](00-roadmap.md)

板块II(实验设计与因果推断)收官篇。本文汇总五个在真实数据分析工作里反复出现、后果严重、但常常被现有教学材料一笔带过的陷阱——Simpson悖论、选择偏差、幸存者偏差、SUTVA违反的另一种形式(处理版本不一致,和07类知识点5的网络效应是完全不同的违反方式)。知识点5是"诊断真实数据"这一新题型第一次以完整、独立、可运行案例的形式落地(dsa-deep-dive 20类调研发现的题型),直接建立在07类知识点4的新奇效应/学习效应模型之上——用一个精心设计的场景,考验读者能不能分清"这是一个真实的行为学效应"还是"这是一个数据管道故障",这个区分能力是capstone诊断环节的直接模板。

**环境声明:** 全部代码在仓库根目录 `.venv`(numpy 2.4.6、scipy 1.17.1)下真实跑通,随机抽样固定种子(`np.random.default_rng(42)`)。

---

## 1. Simpson悖论 —— 合并数据的加权方式本身能反转结论

**定义与记号:** Simpson悖论(Simpson's Paradox):在每个分组(子群体)内部都成立的统计趋势,在把所有分组合并到一起看时,趋势可能完全反转。形式化:即使对所有子群体g都有 rate_A,g > rate_B,g,合并后仍可能出现 rate_A,overall < rate_B,overall。

**一句话:** Simpson悖论不是统计方法出错,是"合并数据"这个操作本身——当两个待比较组在子群体的权重分布上不同时,会让"每个子群体都占优"的一方,在合并之后反而显得吃亏——差的不是运算,是不同的加权方式。

**数学推导:** 设子群体g的权重(样本占比)为π_A,g(A组落在子群体g的比例)、π_B,g(B组落在子群体g的比例)。合并比率:

rate_A,overall = Σ_g π_A,g·rate_A,g
rate_B,overall = Σ_g π_B,g·rate_B,g

即使rate_A,g>rate_B,g对所有g成立,只要π_A,g和π_B,g(两组各自在子群体间的分布)差异足够大——比如A组主要集中在"本来就更难"的子群体(该子群体两组比率都偏低)、B组主要集中在"本来就更容易"的子群体(该子群体两组比率都偏高)——合并后的加权平均就可能反转,因为合并比率本质上是两个**不同的权重向量**分别加权同一组子群体比率,不是同一权重加权,两个加权平均的大小关系可以和逐项比较的关系相反。

**底层机制/为什么这样设计:** 为什么这不是"算错了",而是一个真实的、逻辑自洽的现象?因为"A组整体比B组好"这句话如果不说明"整体"是怎么加权出来的,本身就是有歧义的——合并比率隐含了"按各自的样本构成加权"这个特定操作,当两组样本构成本身高度不平衡时(通常是因为子群体本身是一个混淆变量,同时影响"更可能被分到哪个组"和"结果本身",08类知识点3的混淆变量结构),合并这个操作就把一个真实的混淆结构伪装成了"A、B哪个更好"的简单问题。

**AI研究/工程场景:** 模型评测里的经典陷阱——"模型A在数据集的每个子类别上都优于模型B,但合并后总分模型B更高",往往是因为两个模型在测试集里被评测的子类别分布不完全相同(比如做过采样/欠采样调整,或者两个版本上线时间不同导致流量分布变了),此时应该按统一的子类别权重重新加权计算总分,而不是直接采信原始的、可能已经被子类别构成污染的合并总分。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

# 两种治疗方案A/B, 轻症/重症两个子群体
# 方案A在轻症和重症两个子群体里的治愈率都更高, 但A主要用于重症患者(更难治), B主要用于轻症患者(更容易治)
n_A_mild, n_A_severe = 100, 900   # A组: 90%都是重症
n_B_mild, n_B_severe = 900, 100   # B组: 90%都是轻症
p_A_mild, p_B_mild = 0.95, 0.90       # 轻症子群体: A > B
p_A_severe, p_B_severe = 0.75, 0.70   # 重症子群体: A > B

cure_A_mild = rng.binomial(1, p_A_mild, n_A_mild)
cure_A_severe = rng.binomial(1, p_A_severe, n_A_severe)
cure_B_mild = rng.binomial(1, p_B_mild, n_B_mild)
cure_B_severe = rng.binomial(1, p_B_severe, n_B_severe)

rate_A_mild, rate_B_mild = cure_A_mild.mean(), cure_B_mild.mean()
rate_A_severe, rate_B_severe = cure_A_severe.mean(), cure_B_severe.mean()

# 核心断言1: 子群体内部, A确实处处占优
assert rate_A_mild > rate_B_mild, f"A should beat B in the mild subgroup: {rate_A_mild:.4f} vs {rate_B_mild:.4f}"
assert rate_A_severe > rate_B_severe, f"A should beat B in the severe subgroup: {rate_A_severe:.4f} vs {rate_B_severe:.4f}"

rate_A_overall = np.concatenate([cure_A_mild, cure_A_severe]).mean()
rate_B_overall = np.concatenate([cure_B_mild, cure_B_severe]).mean()

# 核心断言2: 合并后趋势反转 -- Simpson悖论复现
assert rate_B_overall > rate_A_overall, \
    f"Simpson's paradox should reverse the ranking overall: A={rate_A_overall:.4f} B={rate_B_overall:.4f}"

# 核心断言3: 用统一权重(比如各50%轻症50%重症)重新加权, 悖论消解, A重新占优
rate_A_reweighted = 0.5 * rate_A_mild + 0.5 * rate_A_severe
rate_B_reweighted = 0.5 * rate_B_mild + 0.5 * rate_B_severe
assert rate_A_reweighted > rate_B_reweighted, \
    f"reweighting with a common mix should restore A's advantage: A={rate_A_reweighted:.4f} B={rate_B_reweighted:.4f}"

print(f"mild subgroup:   A={rate_A_mild:.4f}  B={rate_B_mild:.4f}  (A wins)")
print(f"severe subgroup: A={rate_A_severe:.4f}  B={rate_B_severe:.4f}  (A wins)")
print(f"raw overall:     A={rate_A_overall:.4f}  B={rate_B_overall:.4f}  (B wins -- paradox!)")
print(f"reweighted 50/50: A={rate_A_reweighted:.4f}  B={rate_B_reweighted:.4f}  (A wins again -- paradox resolved)")
```

**面试怎么问+追问链**(诊断真实数据新题型):
- Q:"数据显示'新算法在活跃用户和非活跃用户两个群体里点击率都更高,但整体点击率反而更低',这是怎么回事?"
- 追问1:"第一反应应该检查什么?"(检查新旧算法在两个群体里的流量分布是否一致——如果新算法的流量主要来自本来点击率就偏低的某个群体(可能是灰度发布策略导致的),即使在每个群体内部都更优,合并后也可能出现Simpson悖论)
- 深挖追问:"确认是Simpson悖论之后,应该以哪个数字作为最终结论?"(应该报告分群体的效果、以及在统一权重(比如按最终全量上线后预期的流量分布)下重新加权的合并效果,而不是原始的、被不均衡灰度分布污染的合并总分,同时要在报告里明确说明"合并总分为什么会反转"这个发现本身)

**常见坑:**
- 只报告合并后的总体数字,不做分群体的敏感性检查,让Simpson悖论完全隐藏在一个看似简单的结论后面。
- 一旦发现Simpson悖论,盲目认为"分群体的结论才是对的,合并数字是假的"——两者都是真实的数字,只是回答了不同的问题(在每个群体内部哪个更好 vs 在当前的群体构成下整体哪个更好),该用哪个取决于具体决策场景需要哪种口径,不是非此即彼。

---

## 2. 选择偏差 —— 分析用的数据本身已经被筛过一遍

**定义与记号:** 选择偏差(selection bias):分析所依据的样本,不是从目标总体中无偏抽取的,而是经过了某种和结果本身相关的筛选机制才进入样本——导致基于这个样本算出的统计量,不能代表真正想了解的目标总体。

**一句话:** 选择偏差的本质是"分析的这份数据,本身就是已经被筛过一遍的",而这个筛选标准恰好和关心的结果变量相关,所以数据看起来讲述的故事,其实主要是"筛选机制"的故事,不是"真实世界"的故事。

**数学推导:** 设目标总体的真实关系是Y=X+ε(斜率为1)。观测样本受选择机制S=1[Y>c]筛选(比如"只看通过了某个筛选门槛的用户")。在选择后的子样本里,X和Y的关系会发生系统性扭曲:在阈值附近,X偏低的个体需要ε偏大才能通过筛选,人为制造了X和"是否被保留"之间的关联,压低了子样本里估计出的斜率——这和08类知识点4的对撞偏差(collider bias)是同一个数学根源:"以某个依赖(X,Y)的变量为条件筛选样本"这个操作本身。

**底层机制/为什么这样设计:** 为什么选择偏差和08类的对撞偏差本质相同?因为"样本是否被观测到"这件事,本身可以看作一个隐藏的对撞节点——如果这个"是否被观测到"同时依赖X和Y,那么"只在被观测到的子样本里分析"就是在对一个对撞节点做条件化筛选,和08类知识点4的招聘漏斗例子是同一个机制在**数据收集阶段**(而非事后分析阶段)的体现——选择偏差往往在数据收集这一步就已经发生,比对撞偏差更隐蔽,因为分析者经常意识不到自己拿到的数据本身已经被筛选过。

**AI研究/工程场景:** 用户调研/问卷分析是选择偏差的重灾区——愿意花时间填问卷的用户本身就不是随机的(可能更活跃、更有强烈意见),基于问卷结果直接推断"全体用户怎么看"存在系统性偏差;推荐系统离线评测同理——只用"用户点击过的物品"的反馈数据训练/评测模型,而"用户点击"这个行为本身高度依赖模型之前的排序结果,构成一个自我强化的选择偏差闭环。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)
n = 10_000

X = rng.normal(0, 2, n)
Y = X + rng.normal(0, 1, n)  # 目标总体的真实关系: 斜率=1

slope_full = np.cov(X, Y, ddof=0)[0, 1] / np.var(X)
assert abs(slope_full - 1.0) < 0.1, f"full-population slope should be close to the true value 1.0, got {slope_full:.4f}"

# 只保留Y超过某个高阈值的子样本(模拟"只分析通过筛选的用户")
threshold = 2.0
mask = Y > threshold
X_sel, Y_sel = X[mask], Y[mask]

slope_selected = np.cov(X_sel, Y_sel, ddof=0)[0, 1] / np.var(X_sel)

# 核心断言: 选择偏差导致子样本里估计出的斜率明显低于真实值(截断回归的经典衰减效应)
assert slope_selected < slope_full * 0.7, \
    f"selection bias should visibly attenuate the estimated slope: full={slope_full:.4f} selected={slope_selected:.4f}"

print(f"full-population slope (unbiased) = {slope_full:.4f}  (true value = 1.0)")
print(f"selected-subsample slope (Y > {threshold}, n={mask.sum()}) = {slope_selected:.4f}  (badly attenuated)")
```

**面试怎么问+追问链**(真实性验证轴):
- Q:"用户调研显示'90%的受访用户对新功能满意',能不能说'90%的用户对新功能满意'?"
- 追问1:"这两句话有什么区别?"(前一句只是对"填了问卷的这部分人"的准确描述,后一句是对"全体用户"的推断——如果愿意填问卷的人本身不是全体用户的无偏代表,两句话可能完全不是一回事)
- 深挖追问:"怎么估计或校正这种选择偏差?"(可以对比"填问卷的用户"和"全体用户"在可观测特征上的分布差异,如果差异很大说明选择偏差风险高;更严谨的做法是采用加权回补或者改用行为数据交叉验证,但完全消除选择偏差通常很难,需要在结论里明确注明这个局限)

**常见坑:**
- 把"样本量很大"当作"样本没有选择偏差"的证据——选择偏差是系统性的(bias),不是随机噪声(variance),样本量再大也不会自动修正一个有偏的筛选机制。
- 意识不到数据本身在"进入分析管道之前"就已经经过了筛选(比如系统日志本身只记录了"成功"的请求,失败/超时的请求根本没有留下记录)——这种"沉默的筛选"往往比主动选择偏差更隐蔽,是知识点3幸存者偏差要专门展开的情形。

---

## 3. 幸存者偏差 —— 现存样本可能恰好讲述了相反的故事

**定义与记号:** 幸存者偏差(survivorship bias):选择偏差的一种特殊、但极其常见的形式——分析样本系统性地缺失了"失败/消亡/中途退出"的那部分个体,只能观测到"存活下来"的部分,导致基于现存样本得出的结论,可能严重误判了决定"存活"本身的那些因素的真实效应方向。

**一句话:** 如果只研究现在还活着的公司/还在使用的用户/还没崩溃的策略,天然看不到那些具备同样特征、但已经"死掉"的样本,这时候"现存样本的共同特点"很可能不是成功的原因,而恰恰是幸存的结果混杂了失败案例缺失导致的错觉。

**数学推导/说明:** 经典的二战战机装甲例子可以直接量化:如果只统计"返航战机"的弹孔分布,弹孔少的部位其实是"中弹了就再也回不来"的致命部位,而不是"不需要加装甲"的安全部位——因为真正被击中要害的战机根本没有出现在"返航"这个样本里。本知识点用一个更贴近数据分析场景的版本:两种策略(低风险/高风险),高风险策略的真实结果分布方差极大(赢得多、输得也可能很惨),如果亏损超过某个阈值的个体"出局"(从后续数据里彻底消失,比如公司倒闭、账户清零),现存样本里高风险策略的平均表现会被系统性地向上偏移——偏移方向和幅度可以直接算出,不是含糊的"可能有偏差"。

**底层机制/为什么这样设计:** 幸存者偏差为什么经常导致结论方向**完全反转**,而不只是数值上有点偏差?因为"存活"这个筛选机制经常和研究者关心的自变量存在强烈的、非线性的交互效应(高风险策略要么带来存活所需的高回报、要么直接导致出局,不是正态分布而更像双峰分布),这种情况下"现存样本"不是"总体的一个稍微有偏的子集",而是"总体里表现最差的那部分被系统性抹去"之后剩下的部分,偏差的方向往往和直觉相反,而不只是幅度上打折扣。

**AI研究/工程场景:** "分析现在还在维护的开源项目,发现某种架构选择和长期成功正相关"这类研究,天然只统计了活到现在的项目,已经废弃的项目(可能出于完全相同甚至更好的技术判断,只是运气不好或团队解散)完全不在样本里;模型训练领域的类似现象是发表偏差(publication bias)——"分析训练成功的大模型,发现某种超参数组合表现好",而大量因为同样的超参数组合训练崩溃、直接被放弃、从未发表的实验完全不会出现在这类分析的样本里。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)
n = 5000

strategy = rng.integers(0, 2, n)  # 0=低风险, 1=高风险
low_risk_return = rng.normal(5.0, 2.0, n)     # 低风险: 均值较高, 方差小
high_risk_return = rng.normal(3.0, 15.0, n)   # 高风险: 均值较低, 但方差极大(赢得多也可能输得很惨)
true_return = np.where(strategy == 1, high_risk_return, low_risk_return)

# 完整总体(包括后续会"出局"的个体): 低风险策略真实表现更好
full_mean_low = true_return[strategy == 0].mean()
full_mean_high = true_return[strategy == 1].mean()
assert full_mean_low > full_mean_high, \
    f"in the full population, low-risk should genuinely outperform high-risk: {full_mean_low:.4f} vs {full_mean_high:.4f}"

# 亏损超过10视为"出局", 从后续可分析的数据里彻底消失(账户清零/公司倒闭/项目下线)
survived = true_return > -10

survived_mean_low = true_return[(strategy == 0) & survived].mean()
survived_mean_high = true_return[(strategy == 1) & survived].mean()

# 核心断言: 只看幸存样本, 结论完全反转 -- 高风险策略看起来反而更好
assert survived_mean_high > survived_mean_low, \
    f"survivorship bias should reverse the ranking among survivors only: low={survived_mean_low:.4f} high={survived_mean_high:.4f}"
assert survived_mean_high > full_mean_high + 2.0, \
    f"the high-risk survivors' average should be substantially inflated above the true full-population mean: {survived_mean_high:.4f} vs {full_mean_high:.4f}"

print(f"full population (includes those that later 'went out'): low-risk={full_mean_low:.4f}  high-risk={full_mean_high:.4f}  (low-risk genuinely wins)")
print(f"survivors only:                                          low-risk={survived_mean_low:.4f}  high-risk={survived_mean_high:.4f}  (high-risk falsely appears to win)")
```

**面试怎么问+追问链**(诊断真实数据新题型):
- Q:"分析显示'公司里加班最多的那批员工,后续晋升率最高',能不能得出'多加班有助于晋升'的结论?"
- 追问1:"这个分析可能漏掉了哪些人?"(可能漏掉了"加班很多但最终因为过劳/不满离职、从未有机会被观测到'后续晋升'这个结果"的那部分员工——如果这部分人本来也占加班多这个群体的相当比例,现存分析样本已经被"留下来的人"这个筛选条件污染了)
- 深挖追问:"怎么设计更严谨的分析来验证这个猜想?"(需要包含"已离职员工"的完整队列数据,重新按入职时间对齐做队列分析,同时统计加班多组和加班少组各自的留存率、晋升率、离职原因,而不是只在"当前在职"这个已经被筛选过的截面上做分析)

**常见坑:**
- 只用"现存/成功"样本做归因分析,不去主动寻找和量化"失败/退出"样本的规模和特征——幸存者偏差最危险的地方在于,不知道自己的数据已经被过滤了这件事本身。
- 找到幸存者偏差的证据后,简单地把结论反过来说(比如"加班反而不利于晋升")——这同样是在缺乏完整数据的情况下做过度推断,正确的态度是承认现有数据不足以支撑任何方向的因果结论,需要补充缺失的部分才能重新评估。

---

## 4. SUTVA违反的其他形式 —— 名义上的"同一种处理",实际执行未必一致

**定义与记号:** 07类知识点5讲的SUTVA违反是"个体间存在溢出效应"(处理会泄漏到对照组)。本知识点聚焦SUTVA的另一半要求——"处理版本一致性"(no hidden variations of treatments):所有被标记为"接受处理Z=1"的个体,实际接受的处理内容/强度/实施方式并不完全相同,导致"处理效应τ"这个概念本身变得模糊——估计出来的其实是"多个不同版本处理的某种混合平均效应",不是单一、良好定义的处理效应。

**一句话:** 07类知识点5讲的是"A组和B组会互相传染"(处理会泄漏到对照组);本知识点讲的是另一种同样违反SUTVA、但完全不同的问题——"名义上都是A组,但A组内部实际执行的'处理'根本不是同一个东西"。

**数学推导:** potential outcomes框架(08类知识点1)隐含假设Y_i(1)是一个良好定义的量。如果处理Z=1实际对应多个不同的实施版本v∈{v_1,v_2,...},每个版本有各自的效应τ_v,那么估计出的"平均处理效应"实际上是 τ̂≈Σ_v π_v·τ_v(π_v是各版本在处理组里的实际占比),这个加权平均依赖于"各版本在当前实施中恰好的占比分布"——如果换一个时间/地点/团队,同样名义上的"处理Z=1"可能对应完全不同的版本分布,估计出的效应就不再适用,可复现性因此受损。这在数学结构上和知识点1的Simpson悖论加权机制同源(合并异质子群体的效应),只是这里"异质子群体"换成了"同一处理标签下的不同实施版本"。

**底层机制/为什么这样设计:** 为什么"处理版本不一致"会破坏SUTVA而不只是普通的测量噪声?因为SUTVA的完整表述要求"每个个体的处理只有一种可能的呈现形式",如果处理版本之间效应差异很大(不是围绕一个共同均值的小噪声,而是本质不同的干预),那么"平均处理效应"这个总结统计量本身的意义就变得可疑——它依赖于一个几乎肯定会变化的、当前实施恰好呈现出的版本混合比例,不是一个稳定的、可以脱离具体实施细节去理解和复现的因果参数。

**AI研究/工程场景:** "推荐算法改版"这类实验里,"处理组=用了新排序模型"这个标签背后,可能因为模型是持续迭代/AB切流逐步放量/不同人群命中不同的召回策略组合等原因,实际上在实验期间经历了好几个不完全相同的"新模型版本",如果不区分版本、把整个实验期笼统地当作"新模型 vs 旧模型"一次性分析,估计出的效应是几个不同版本效应的模糊平均,既无法准确复现,也难以指导"哪个具体改动带来了效果"这类更精细的决策。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)
n = 4000

baseline = rng.normal(50, 10, n)
version = rng.integers(0, 2, n)  # 0=版本A, 1=版本B, 各占一半, 都被笼统标记为"处理组"
tau_version_A = 8.0
tau_version_B = 1.0

Y_control = baseline + rng.normal(0, 3, n)
Y_treated = baseline + np.where(version == 0, tau_version_A, tau_version_B) + rng.normal(0, 3, n)

# 笼统地把"处理组"当作单一处理估计: 得到的是一个混合平均效应
mixed_estimate = Y_treated.mean() - Y_control.mean()
expected_mix = 0.5 * tau_version_A + 0.5 * tau_version_B
assert abs(mixed_estimate - expected_mix) < 0.5, \
    f"the lumped estimate should match the mixture average, got {mixed_estimate:.4f} vs expected {expected_mix:.4f}"

# 核心断言: 混合平均既不接近版本A的真实效应, 也不接近版本B的真实效应 -- 对任何具体决策都没有直接指导意义
assert abs(mixed_estimate - tau_version_A) > 2.0, "mixed estimate should not be close to version A's true effect"
assert abs(mixed_estimate - tau_version_B) > 2.0, "mixed estimate should not be close to version B's true effect"

# 按版本拆分重新估计: 应该能准确恢复各自的真实效应
est_A = Y_treated[version == 0].mean() - Y_control[version == 0].mean()
est_B = Y_treated[version == 1].mean() - Y_control[version == 1].mean()
assert abs(est_A - tau_version_A) < 0.5, f"version-A-only estimate should recover its true effect, got {est_A:.4f} vs {tau_version_A}"
assert abs(est_B - tau_version_B) < 0.5, f"version-B-only estimate should recover its true effect, got {est_B:.4f} vs {tau_version_B}"

print(f"lumped 'treatment' estimate = {mixed_estimate:.4f}  (a meaningless blend of {tau_version_A} and {tau_version_B})")
print(f"version A only = {est_A:.4f}  (true={tau_version_A})")
print(f"version B only = {est_B:.4f}  (true={tau_version_B})")
```

**面试怎么问+追问链**(方案批判迭代轴):
- Q:"实验显示'新排序策略平均提升了3%的点击率',这个数字能直接用来指导后续优化方向吗?"
- 追问1:"这个'新排序策略'在整个实验期间,实施细节是完全一致的吗?"(如果实验期间模型经历过几次迭代/参数调整/召回策略变化,3%这个数字其实是好几个不完全相同的"新策略版本"的混合平均效应,不对应任何一个具体、可复现的改动)
- 深挖追问:"如果确实存在多个版本混杂,应该怎么处理?"(需要在实验设计阶段就记录版本变更的时间点,分版本重新估计效应,而不是在分析阶段才发现问题;如果版本变化是渐进式的,可能需要用时间序列/分段的方式估计效应演变过程,而不是笼统报告一个平均数)

**常见坑:**
- 把"处理组内部效应存在差异"简单归因于个体异质性(08类知识点6的heterogeneous treatment effects),而不去检查这个差异是否其实来自"处理本身没有单一稳定版本"这个更根本的问题——两者的应对方式完全不同:个体异质性需要更精细的个体化建模,处理版本不一致首先需要在实验设计和数据记录阶段解决版本控制问题。
- 实验报告里完全不提及处理的具体实施细节是否在实验期间发生过变化,让读者误以为效应估计对应一个单一、稳定、可复现的干预。

---

## 5. 真实A/B测试事故复盘 —— "诊断真实数据"新题型的旗舰示范

**定义与记号:** 本知识点是"诊断真实数据"这一新题型(dsa-deep-dive 20类调研发现)第一次以完整、独立、可运行案例的形式落地——给定一段真实的埋点/日志数据,要求先诊断数据本身是否存在异常,而不是拿到数据就直接套用假设检验模板计算显著性。场景:某次A/B测试的"转化"事件埋点,在实验开始后第3天才真正生效上线(前3天由于埋点配置延迟,转化数据全部记录为0),如果不做诊断直接分析全部7天数据,会显著低估真实效应。

**一句话:** 这份数据的每日趋势表面上和07类知识点4的"学习效应"(前期效应弱、后期效应强)长得很像,但两者的生成机制截然不同——真实的行为学效应(不管新奇效应还是学习效应)是**平滑连续**变化的,数据管道故障造成的"假0"是**硬性断崖式**的,这个形状差异本身就是可以量化检验的诊断证据,不能仅凭"前几天效果弱"就想当然套用07类刚学的模型。

**数学推导/说明:** 构造合成"日志"数据:7天实验,两组用户,真实转化率处理组高于对照组(存在真实效应),但转化事件埋点在第1-3天存在配置缺陷,导致这3天记录的转化数**全部精确为0**,第4-7天恢复正常记录。区分"真实学习效应"和"埋点故障"的关键定量证据:如果这真的是07类知识点4式的平滑行为学效应,哪怕早期效应很弱(比如早期转化率仅p=0.02),3000次曝光里恰好一次转化都没有的概率是 (1-0.02)^3000≈4.8×10⁻²⁷——这是天文数字级别的不可能,而不是"有点巧合"。这个概率计算本身就是"排除掉学习效应假说"的严谨证据,不是凭直觉说"看起来像bug"。进一步,朴素(不做诊断)分析全部7天数据得到的效应估计,和只用第4-7天干净数据估计的效应之间存在一个**精确的代数关系**:因为前3天两组转化数都恒为0(只贡献分母曝光量、不贡献分子转化量),朴素合并比率 = (干净天数/总天数)×干净估计值,在本例中就是 naive_diff = (4/7)×clean_diff,这不是近似,是由构造方式决定的精确等式。

**底层机制/为什么这样设计:** 为什么"先诊断再分析"应该成为处理任何真实数据集的默认第一步?因为假设检验、置信区间等第一部分(01-06类)建立的所有推断工具,都隐含假设"输入的数据本身正确记录了它声称记录的东西"——这些工具没有能力、也不被设计用来"发现数据本身是否可信",一旦这个前提被违反,后续所有精心设计的统计推断都是在错误的地基上盖楼,计算出来的p值、置信区间在数学上依然"正确"(计算过程没有错),但对应的现实问题的答案是错的——这是"统计方法本身没错,但被喂了错误的输入"和"统计方法用错了"这两类完全不同错误的区分。

**AI研究/工程场景:** 埋点/日志/特征管道的部署时间和实验开始时间不完全同步,是数据基础设施里的常见真实故障模式(不限于实验分析,模型训练用的特征管道同样会出现"某个特征在某个时间点之前的历史数据是错的/缺失的"这类问题)。一个成熟的数据分析师/研究员应该把"检查数据的时间序列完整性"作为拿到任何新数据集后的常规操作,而不是等结果"看起来奇怪"了才回头检查。

**可运行例子:**
```python
import numpy as np

rng = np.random.default_rng(42)

n_days = 7
n_per_group_per_day = 3000
true_p_control = 0.10
true_p_treatment = 0.13  # 真实效应: +3个百分点
broken_days = 3          # 前3天埋点故障, 转化记录全部为0(不管真实是否转化, 曝光量正常记录)

daily_exposure_control, daily_conversion_control = [], []
daily_exposure_treatment, daily_conversion_treatment = [], []

for day in range(n_days):
    exp_c = exp_t = n_per_group_per_day
    true_conv_c = rng.binomial(exp_c, true_p_control)
    true_conv_t = rng.binomial(exp_t, true_p_treatment)
    if day < broken_days:
        recorded_conv_c, recorded_conv_t = 0, 0  # 埋点故障: 真实转化发生了, 但记录丢失
    else:
        recorded_conv_c, recorded_conv_t = true_conv_c, true_conv_t
    daily_exposure_control.append(exp_c); daily_conversion_control.append(recorded_conv_c)
    daily_exposure_treatment.append(exp_t); daily_conversion_treatment.append(recorded_conv_t)

daily_exposure_control = np.array(daily_exposure_control)
daily_conversion_control = np.array(daily_conversion_control)
daily_exposure_treatment = np.array(daily_exposure_treatment)
daily_conversion_treatment = np.array(daily_conversion_treatment)

# 诊断证据1: 如果这是07类知识点4式的平滑学习效应, 哪怕早期效应很弱, 3000次曝光里恰好0次转化的概率也荒谬地低
plausible_early_rate = 0.02
prob_exact_zero_if_real_effect = (1 - plausible_early_rate) ** n_per_group_per_day
assert prob_exact_zero_if_real_effect < 1e-20, \
    f"exact-zero conversions across {n_per_group_per_day} exposures should be astronomically unlikely under a real behavioral effect, got p={prob_exact_zero_if_real_effect:.2e}"

# 诊断证据2: 定位异常天数(两组转化数都恰好为0, 且曝光量本身正常 -- 排除"没有流量"这个更平凡的解释)
suspicious_days = [
    d for d in range(n_days)
    if daily_conversion_control[d] == 0 and daily_conversion_treatment[d] == 0
    and daily_exposure_control[d] > 0 and daily_exposure_treatment[d] > 0
]
assert suspicious_days == [0, 1, 2], f"diagnosis should pinpoint exactly the 3 broken days, got {suspicious_days}"

# 不做诊断, 直接用全部7天数据(朴素分析)
naive_p_control = daily_conversion_control.sum() / daily_exposure_control.sum()
naive_p_treatment = daily_conversion_treatment.sum() / daily_exposure_treatment.sum()
naive_diff = naive_p_treatment - naive_p_control
true_diff = true_p_treatment - true_p_control

# 排除掉确认有问题的天数后重新分析("诊断后"分析)
clean_mask = np.array([d not in suspicious_days for d in range(n_days)])
clean_p_control = daily_conversion_control[clean_mask].sum() / daily_exposure_control[clean_mask].sum()
clean_p_treatment = daily_conversion_treatment[clean_mask].sum() / daily_exposure_treatment[clean_mask].sum()
clean_diff = clean_p_treatment - clean_p_control

# 核心断言1: 朴素估计和干净估计之间存在精确的代数关系(不是近似) -- naive = (好天数/总天数) * clean
expected_dilution_ratio = (n_days - broken_days) / n_days
assert abs(naive_diff - clean_diff * expected_dilution_ratio) < 1e-9, \
    "naive_diff should exactly equal clean_diff scaled by the fraction of unbroken days"

# 核心断言2: 不做诊断会显著低估真实效应
assert naive_diff < true_diff * 0.85, \
    f"naive (undiagnosed) estimate should substantially understate the true effect: naive={naive_diff:.4f} true={true_diff:.4f}"

# 核心断言3: 诊断后排除故障天数, 估计量回到接近真实效应
assert abs(clean_diff - true_diff) < 0.015, \
    f"post-diagnosis clean estimate should be close to the true effect, got {clean_diff:.4f} vs true {true_diff}"

print(f"daily conversion rate (control):   {(daily_conversion_control / daily_exposure_control).round(3).tolist()}")
print(f"daily conversion rate (treatment): {(daily_conversion_treatment / daily_exposure_treatment).round(3).tolist()}")
print(f"P(exact 0 conversions | real weak effect p=0.02) = {prob_exact_zero_if_real_effect:.2e}  -- rules out a genuine behavioral effect")
print(f"diagnosed broken days = {suspicious_days}")
print(f"naive (undiagnosed, all 7 days) diff = {naive_diff:.4f}  (true effect = {true_diff})")
print(f"clean (post-diagnosis, days 4-7) diff = {clean_diff:.4f}")
```

**面试怎么问+追问链**(诊断真实数据新题型,本知识点自身即是这一题型的完整示范):
- Q:"这是一次为期7天的A/B测试的每日转化数据,前3天两组转化数都是0,第4-7天正常有转化,你看一眼,有什么想说的?"
- 追问1:"前3天两组转化数都是0,这说明了什么?"(两种可能:①真实效应确实要到第4天才开始显现,类似07类知识点4的学习效应;②更可能的解释是埋点/记录管道在前3天没有正常工作——区分方法是检查"曝光数"是否也是0:如果曝光数正常但转化数恰好为0,而且样本量足够大,可以直接算出"这是真实弱效应"这个假说下观测到精确0的概率有多离谱,用数字说话而不是凭直觉)
- 深挖追问:"确认是埋点故障之后,应该怎么处理这份数据,直接删掉前3天重新算吗?"(可以排除掉确认有问题的时间段重新分析,但必须在报告里如实说明"排除了前3天数据,原因是埋点配置延迟"这个数据处理决策,不能悄悄地删除数据而不留痕迹;同时应该推动修复数据管道问题本身,避免下次实验重复同样的故障)

**常见坑:**
- 拿到数据直接套用假设检验模板计算p值,从来不做"这份数据是否完整、是否有系统性异常模式"这一步基础检查,把数据质量问题误判成"效应不显著",或者被错误的数据严重稀释了真实效应却毫无察觉。
- 看到"效应随时间由弱变强"这个表面模式,不加区分地套用刚学过的07类知识点4学习效应模型,而不去检查这个模式的具体数值特征(比如是否精确为0、是否有硬性断崖)是否真的符合平滑行为学效应的数学特征——学过一个模型之后,最容易犯的错误就是看到相似的表面形状就直接套用,而不检验生成机制是否真的匹配。

---

板块II(实验设计与因果推断,06-10类,共28个知识点)到此收官。下一篇:[11-bayesian-inference-foundations.md](11-bayesian-inference-foundations.md) —— 板块III贝叶斯方法,从"频率派"视角切换到"贝叶斯派"视角重新审视不确定性。
