# 03 · Speculative Decoding 深挖(投机解码全谱)

> 总览见 [00-roadmap.md](00-roadmap.md)

01 号文件讲过 decode 阶段是 memory-bound——每步只算 1 个新 token,但要把全部历史 KV 从显存搬到计算单元,大部分时间花在"等数据"而不是"在算"。投机解码这整个家族要解决的问题是:既然搬数据的时间远超算 1 个 token 的时间,那能不能"顺手"多算几个候选 token,让本来要花的那份访存时间产出不止 1 个 token?本文是 `inference-serving-deep-dive` 系列第 3 篇,对应 `learning/speculative-decoding/`(Module 5《用大模型》第 3 专题,12 lectures + 11 个 src 源文件),从"draft+verify+rejection sampling为什么能做到零 bias"这个数学核心讲起,一路走过 Medusa→EAGLE→EAGLE-2→EAGLE-3 这条"draft 越来越聪明"的演化链,再看 Lookahead/Self-Spec 这两条"不训练额外模型"的旁支,最后落到 Tree Attention 这个让"验证一棵树"和"验证一条链"一样快的工程实现,和评测方法论+capstone 收尾。10 个知识点:1(直觉/速度公式)→2(Classic,严格证明+经验验证)→3(Medusa,多头)→4(EAGLE,feature 自回归)→5(EAGLE-2,动态树)→6(EAGLE-3,概念性,本仓库无对应源码)→7(Lookahead,n-gram 池)→8(Self-Spec,跳层)→9(Tree Attention,批量验证的工程底座)→10(评测方法论+Capstone)。

**和 00-roadmap.md 差异化声明的关系:** 本文严格遵守 roadmap 里"03 speculative-decoding"一节交代的特别纪律——`learning/speculative-decoding/src/` 下 11 个源文件里,只有 `capstone_eagle3.py` 和 `speculative_original_minimal.py` 两个有 `__main__`,直接 `python xxx.py` 有输出;其余 9 个(`common.py`/`classic_spec_decode.py`/`medusa_heads.py`/`eagle_minimal.py`/`eagle2.py`/`lookahead.py`/`self_spec.py`/`tree_attention.py`/`spec_eval.py`)是纯库模块,本文全部"可运行例子"对这 9 个模块统一走 `sys.path.insert` 后直接 import 类/函数、手写测试场景调用断言的路径,不依赖它们自带 `__main__`。

**一个重要的诚实标注(知识点 6):** L06(EAGLE-3)在源码目录里**没有对应的 `.py` 文件**——`learning/speculative-decoding/README.md` 自己的"总览"表格里 L06 那一行"代码"列写的就是"—"。本文知识点 6 不会假装存在一份 EAGLE-3 实现,而是如实说明这一点,用全系列共享的速度公式(`speculative_original_minimal.py::walltime_speedup`)代入 EAGLE-3 风格的参数做一次方向性的合理性检验(bigger target model → 相对更划算的 draft → 更高加速比这个定性趋势成立),不冒充复现了 EAGLE-3 的 multi-feature 具体机制。

**环境声明:** 本文全部代码在仓库根目录 `.venv`(Windows 11 原生,Python 3.13.9,torch 2.11.0+cu128)下用 `.venv/Scripts/python.exe` 实际跑通验证,文中数字是真实输出。10 个知识点里只有知识点 9(Tree Attention)用到 `torch`(纯 CPU 张量运算,不需要 GPU),其余全部是纯 Python 标准库(`random`/`math`/`heapq`/`dataclasses`),不依赖任何未安装的第三方包。

---

## 1. 投机解码全景与速度公式(`common.py`,L01)—— "小步快走(draft)+ 大刀验证(verify)",把 memory-bound 变 compute-bound

**是什么:**
```python
def softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def sample_from(probs: List[float], rng: random.Random) -> int:
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r < cum:
            return i
    return len(probs) - 1
```
(`common.py:10-24`)

```python
@dataclass
class SpecMetrics:
    n_iters: int = 0
    n_drafted: int = 0
    n_accepted: int = 0
    n_tokens_out: int = 0

    @property
    def accept_rate(self) -> float:
        return self.n_accepted / max(self.n_drafted, 1)

    @property
    def mau(self) -> float:
        """Mean accepted per iter (excluding bonus)."""
        return self.n_accepted / max(self.n_iters, 1)
```
(`common.py:34-48`)

**一句话:** `softmax`/`sample_from` 是全系列 8 个方法共享的两个基础算子(把 logits 变成合法概率分布、按概率分布采样一个 token id),`SpecMetrics` 是贯穿全系列的统一度量容器——`accept_rate`(每个被 draft 出来的 token 有多大概率被接受)和 `mau`(Mean Accepted per iter Understand,每次投机迭代平均净赚多少个 token)是后面 9 个知识点反复要用到的两把尺子。

**底层机制/为什么这样设计:** 从最笨的想法讲起——如果 LLM 推理只能"一次 forward 产 1 个 token",要生成 N 个 token 就必须做 N 次 forward,每次都要重新把全部历史 KV 从显存搬一遍(01 号文件知识点 1 已经建立过这个直觉)。投机解码的关键观察是:用一个**远比目标模型小、快**的"草稿模型"(draft model)先连续猜 k 个 token,再把这 k 个猜测**一次性**喂给目标模型做并行验证——验证 k+1 个位置的开销和验证 1 个位置几乎一样贵(01 号文件的语言来说:这一步同样是"访存搬 KV"占大头,多算 k 个位置的 logits 只增加 k 份很小的矩阵乘法),所以只要草稿模型猜对的概率不太低,单次"1 次小 forward × k + 1 次大 forward"就能换回不止 1 个 token,平均下来每个 token 摊到的目标模型 forward 次数就降下来了。L01 给出的速度公式是 `speedup ≈ (accept_k + 1) / (1 + k / N)`,其中 `accept_k` 是这一轮平均接受了几个草稿 token、`N` 是目标模型比草稿模型慢多少倍——这个公式后面知识点 3-8 的每一种具体方法,本质上都是在"怎么让 `accept_k` 更高"这一个变量上做文章,`k`(每轮猜几个)和 `N`(草稿相对多快)则是各家方法在工程侧的选择。这个公式本身不要求任何具体的"草稿"实现方式——它对独立小模型(Classic)、同模型多头(Medusa)、同模型自回归 feature(EAGLE 系列)统统成立,这也是为什么后面每一种变体都可以直接套用 L01 这一个统一的收益模型来理解"它相对前一代到底改进了什么"。

**AI 研究场景:** 这套"draft 猜、target 验"的框架不是纸面推导——vLLM 0.7+、SGLang、TensorRT-LLM 都内置了投机解码支持(02 号文件 SGLang 系列 L01 提到过"Agent 推理王"和"投机解码"是同一批 2024-2025 年生产级推理引擎竞相集成的核心特性),核心原因是它是少数"不改变输出分布、纯粹用空闲算力换延迟"的免费午餐——01 号文件讨论的 Continuous Batching/PagedAttention 是在"服务多个并发请求"这个维度上优化吞吐,投机解码是在"单个请求内部"这个维度上优化延迟,两者可以同时叠加使用,互不冲突。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/speculative-decoding/src")
from common import softmax, sample_from, SpecMetrics

probs = softmax([2.0, 1.0, 0.1])
assert abs(sum(probs) - 1.0) < 1e-9

# 大样本经验分布应当收敛到 softmax 给出的理论概率(不是别的分布)
import random
rng = random.Random(0)
counts = [0, 0, 0]
N = 20000
for _ in range(N):
    counts[sample_from(probs, rng)] += 1
empirical = [c / N for c in counts]
for i in range(3):
    assert abs(empirical[i] - probs[i]) < 0.02

m = SpecMetrics(n_iters=10, n_drafted=50, n_accepted=32, n_tokens_out=42)
assert m.accept_rate == 32 / 50
assert m.mau == 32 / 10

def speedup_formula(accept_k, k, N_ratio):
    return (accept_k + 1) / (1 + k / N_ratio)

su = speedup_formula(accept_k=3, k=5, N_ratio=20)
assert abs(su - 3.2) < 0.05   # 复核 L01 给出的例子:accept=3,k=5,N=20 → ≈3.2x
```

**实测(`.venv` 真跑):** `softmax([2.0,1.0,0.1])` 精确得 `[0.659, 0.2424, 0.0986]`,20000 次采样的经验频率 `[0.6584, 0.2441, 0.0974]`,和理论概率吻合(误差 < 0.02)。`SpecMetrics(n_iters=10, n_drafted=50, n_accepted=32, ...)` 的 `accept_rate=0.64`、`mau=3.2`,和手算一致。L01 速度公式代入 `accept=3, k=5, N=20` 精确得 `3.2`,和 lecture 给出的例子数字对上。

**面试怎么问 + 追问链:**
- **Q:** "投机解码为什么能在不改变模型输出分布的前提下加速?" —— 期望说出"draft 模型先猜 k 个 token,target 模型一次并行 forward 验证这 k+1 个位置(并行验证的开销接近验证 1 个位置),用 accept/reject 规则决定接受多少——只要 target 的一次并行 forward 比 target 做 k+1 次串行 forward 划算,就有净收益,而 accept/reject 规则本身(下一知识点详细展开)保证了最终输出分布严格等于 target 自己独立采样的分布"。
- **追问 1:** "如果 draft 模型和 target 模型输出的分布完全一样,收益是不是最大?" —— 期望辨析:分布完全一样时 `accept_k` 确实会接近 `k`(几乎全部接受),但这种情况现实中意义不大——如果真有一个和 target 输出分布完全一致的"小模型",那说明这个小模型本身已经学到了 target 的全部能力,不需要再靠"验证"这道工序,重点是分布要**足够接近但草稿本身要足够便宜**,两者要权衡。
- **追问 2(深挖速度公式):** "`speedup ≈ (accept_k+1)/(1+k/N)` 这个公式里,`k` 是不是选得越大越好?" —— 期望说出"不是,`k` 变大会让分子(平均接受数)和分母(草稿开销占比)同时变大,如果 draft 质量跟不上,`accept_k` 不会随 `k` 线性增长(后面 token 更容易被拒绝),但 `k/N` 这部分开销却是线性增长的,存在一个最优 `k`,这个也是后面知识点 2 里 `best_gamma()` 函数要解决的问题"。
- **追问 3:** "这套机制对 batch size 大的场景还有效吗?" —— 期望说出:投机解码的收益本质来自"单个请求内,访存瓶颈没被算力用满",batch 越大,target 模型的一次 forward 本身就越接近算力打满,能"顺便"多算的 draft token 验证的边际收益就越小——这也是为什么后面知识点 11(评测方法)会强调"投机解码在 batch=1 时最强,batch 越大收益越薄"。

**常见坑:** 把"投机解码"和"近似推理/量化"混为一谈,以为它是用精度换速度——恰恰相反,投机解码在 rejection sampling 保证下是**严格无损**的(下一知识点会给出数学证明和数值验证),它不改变任何一个 token 的最终采样概率,纯粹是"多算一点、少等一点"的工程优化。另一个坑是以为 `accept_k` 是一个只取决于 draft 模型能力的常数——实际上它同时取决于 target 分布本身的"尖锐程度"(01 号文件以来反复出现的主题:越确定性的任务如代码/数学,accept rate 越高;越有创造性的任务如故事续写,accept rate 越低),知识点 10 的评测结果会给出具体数字印证这一点。

---

## 2. Classic Speculative Decoding(`classic_spec_decode.py` + `speculative_original_minimal.py`,L02)—— rejection sampling 严格无 bias 的双重验证:解析证明 + 大样本经验复现

**是什么:**
```python
def rejection_sample(p: List[float], q: List[float], drafted: int, rng: random.Random) -> int:
    """Single-token accept/reject for a drafted token x = `drafted`."""
    r = rng.random()
    ratio = p[drafted] / max(q[drafted], 1e-12)
    if r < ratio:
        return drafted
    residual = [max(0.0, pi - qi) for pi, qi in zip(p, q)]
    s = sum(residual)
    if s <= 0:
        return drafted
    residual = [r / s for r in residual]
    return sample_from(residual, rng)
```
(`classic_spec_decode.py:17-32`)

```python
def exact_one_step_output_distribution(p: list[float], q: list[float]) -> list[float]:
    """Enumerate the speculative-sampling output distribution exactly."""
    check_distribution(p, q)
    accept_part = [min(pi, qi) for pi, qi in zip(p, q)]
    reject_mass = 1.0 - sum(accept_part)
    residual = residual_distribution(p, q)
    return [a + reject_mass * r for a, r in zip(accept_part, residual)]
```
(`speculative_original_minimal.py:51-57`)

**画出来看:每个被 draft 出来的 token,要经过这个判定才知道是被接受,还是被拒绝、改从残差分布重新采样:**

```
draft 已经用 q 采样出了 token x
              │
              ▼
   target 算出 p(x)(x 在 target 眼里的概率)
              │
              ▼
      抛一个 r ~ Uniform[0, 1)
              │
              ▼
      r < min(1, p(x)/q(x)) ?
       是 ↙            ↘ 否
        │                │
        ▼                ▼
    接受 x           从残差分布重新采样一个新 token:
  (直接输出,        residual(y) = norm(max(0, p(y) - q(y)))
   不用改)           (在 p 比 q "多给"的地方补偿采样)
```

`p(x)/q(x)` 越大(target 比 draft 更看好这个具体 token),接受概率越高,`min(1, ...)` 这个 clamp 保证接受概率永远落在合法的 `[0,1]` 区间(哪怕 `p(x)>q(x)` 让比值超过 1)。拒绝之后不是简单地"重新问 target 要一个新 token"就完事——必须从这个专门构造的残差分布里采样,只有接受和拒绝重采样这两部分的概率质量加起来,才能精确重建出 `p`(下方"一句话"和"底层机制"给出这个恒等式本身)。

**一句话:** `rejection_sample` 是投机解码算法的心脏——以 `min(1, p(x)/q(x))` 的概率接受草稿 token `x`,拒绝时改从"残差分布" `norm(max(0, p-q))` 里重新采样,`speculative_original_minimal.py` 用**解析枚举**(不采样,直接算出精确的输出概率分布)独立证明了这套规则的结果严格等于 `p`。

**底层机制/为什么这样设计:** 从最笨的想法讲起——为什么不能直接"draft 猜的对不对,对就用,不对就调用 target 重新生成"这么简单粗暴地做?因为这样会引入 bias:如果只在"猜错"时才调用 target,那些 draft 更容易猜对的 token(比如高概率 token)会被系统性地过度呈现,输出分布会偏向"draft 和 target 都容易生成的部分",不再是 target 的真实分布。正确的做法必须让每个 token 的"最终被采出的概率"精确等于 `p(x)`,而不只是"看起来合理"。数学上这样构造:`accept_part(x) = min(p(x), q(x))` 是"直接接受"贡献的概率质量,`reject_mass = 1 - sum(accept_part)` 是"被拒绝、需要重采样"的总概率,`residual_distribution(x) = norm(max(0, p(x)-q(x)))` 是"如果被拒绝,应该按这个分布重新采样"——把两部分加起来 `accept_part(x) + reject_mass * residual(x)`,精确等于 `p(x)`(这是一个可以纯代数验证的恒等式,不依赖任何采样过程)。`rejection_sample` 函数是这套解析结果的**采样版实现**:先做接受判定(`r < p/q`),不接受就从残差分布里采;`speculative_original_minimal.py::exact_one_step_output_distribution` 反过来是**不采样、直接解析算出"如果无限次重复这个过程,最终各 token 出现的概率是多少"**——两者从两个不同角度(蒙特卡洛 vs 解析枚举)验证同一个数学结论,这也是为什么本知识点特意把两个文件放在一起讲。

**AI 研究场景:** 这个"严格无损"的性质是投机解码能被生产系统放心采用的前提——如果它是一种近似(像量化那样有精度损失),production 系统在使用前要做大量 A/B 测试评估质量回退;正因为它数学上等价于直接从 target 采样,vLLM/SGLang 等引擎可以把它当成纯粹的性能开关打开,不需要重新评估模型质量。`overlap_mass`(01 号文件 L01 提到的 accept rate)在这里对应解析计算里的 `sum(min(p,q))`,04 号文件(quantization-deploy)会讨论的"量化引入的精度损失"和投机解码这种"零 bias 的加速"是完全不同性质的两类优化,读者不应该把两者混为一谈。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/speculative-decoding/src")
from common import sample_from
from classic_spec_decode import rejection_sample
from speculative_original_minimal import (
    overlap_mass, exact_one_step_output_distribution, best_gamma,
)

# 换一组和源文件 _self_test() 不同的 p,q(源文件用 p=[.5,.3,.2], q=[.35,.45,.2])
p = [0.10, 0.60, 0.30]
q = [0.40, 0.40, 0.20]

# --- Part A: 解析证明(不采样,直接算出理论输出分布必须严格等于 p) ---
out_exact = exact_one_step_output_distribution(p, q)
for a, b in zip(p, out_exact):
    assert abs(a - b) < 1e-9
alpha = overlap_mass(p, q)
assert abs(alpha - 0.7) < 1e-9   # sum(min(p,q)) = min(.1,.4)+min(.6,.4)+min(.3,.2) = .1+.4+.2 = .7

# --- Part B: 大样本经验复现(真的采样 3 万次,独立验证解析结论) ---
import random
rng = random.Random(42)
accept_hist = [0, 0, 0]
n_trials = 30000
for _ in range(n_trials):
    drafted = sample_from(q, rng)
    tok = rejection_sample(p, q, drafted, rng)
    accept_hist[tok] += 1
emp = [c / n_trials for c in accept_hist]
for a, b in zip(p, emp):
    assert abs(a - b) < 0.02   # 经验频率收敛到 target p,不是 draft q

# --- Part C: gamma(每轮猜几个)存在最优值,不是越大越好 ---
best = best_gamma(alpha=0.6, draft_cost_ratio=0.05, max_gamma=20)
assert 1 <= best[0] <= 20
```

**实测(`.venv` 真跑):** 换用 `p=[0.1,0.6,0.3], q=[0.4,0.4,0.2]`(和源文件自带 `_self_test()` 的 `p=[.5,.3,.2]` 不同)重新验证:`exact_one_step_output_distribution` 精确输出 `[0.1, 0.6, 0.3]`,和 `p` 逐位相等;`overlap_mass=0.7`。大样本经验测试(3 万次独立采样)得到最终 token 分布 `[0.1004, 0.6002, 0.2994]`,和 `p` 的误差都在 `0.001` 量级——这条经验路径和解析路径互相独立(一个靠代数恒等式、一个靠采样统计),结论一致。`best_gamma(alpha=0.6, draft_cost_ratio=0.05)` 返回 `(4, 1.921...)`,即在这组参数下每轮猜 4 个 token 时理论加速比最高(约 1.92 倍)。

**面试怎么问 + 追问链:**
- **Q:** "怎么证明投机解码的 rejection sampling 不会引入 bias?" —— 期望能写出或口述核心恒等式:`min(p,q) + (1-Σmin(p,q)) · norm(max(0,p-q)) = p`,并说明这是纯代数结论,不依赖任何具体的采样次数。
- **追问 1(考察是否真的验证过,不能只背公式):** "如果只用蒙特卡洛跑 100 次,能不能说明这个结论成立?" —— 期望说明:100 次的样本噪声可能掩盖真实的小偏差,必须用解析枚举(不依赖采样)或足够大的样本量(本知识点用了 3 万次,误差压到 0.001 量级)才有说服力,这也是为什么本文特意把"解析证明"和"经验复现"两条路径都摆出来,不是只信一条。
- **追问 2:** "`residual_distribution` 里如果 `sum(max(0,p-q))` 恰好等于 0 会怎样?" —— 期望说出:这种情况只发生在 `q` 在每个位置都不小于 `p`(即 draft 分布"处处覆盖"target),源码里显式检查 `if s <= 0: return drafted`(或 `return list(p)`)兜底,不会除零崩溃——这也是常见坑里要强调的一个边界条件。
- **追问 3(深挖工程约束):** "现实中 draft 模型和 target 模型的 tokenizer 不一致会有什么后果?" —— 期望说出:`p`/`q` 必须定义在同一个 support(同一个词表)上,`rejection_sample` 这套数学才成立;如果两个模型 tokenizer 不同,连"对齐同一个 token 位置比较概率"这个前提都不成立,实践中投机解码几乎总是选同族模型(比如 Qwen-0.5B 配 Qwen-7B)正是为了保证 tokenizer 天然一致。

**常见坑:** 把"accept rate 高"直接等同于"这个 draft 模型质量好"——`overlap_mass` 只衡量 draft 和 target 分布的重合程度,一个"处处均匀"的 draft 分布也可能和低熵的 target 分布凑出还不错的 overlap,但这种 draft 在真实生成任务里参考价值有限;真正有意义的是"draft 在语义上确实倾向于猜出 target 大概率会选的 token",不是单纯统计意义上的分布重合。另一个坑是以为 `best_gamma` 给出的 `k` 值放到任何模型对上都适用——它完全依赖 `alpha`(overlap)和 `draft_cost_ratio` 这两个输入,换一对模型这两个数字就会变,`k` 的最优值也要跟着重新算。

---

## 3. Medusa(`medusa_heads.py`,L03)—— 让 target 自己当 draft:多头并行猜测,以及一个真实的 `ZeroDivisionError` 边界坑

**是什么:**
```python
@dataclass
class MedusaHeads:
    n_heads: int = 4
    noise: float = 0.5    # how much each head's output differs from target

    def draft(self, target_dist: List[float], rng: random.Random) -> List[List[float]]:
        """Each head outputs a noisy version of `target_dist`."""
        out = []
        for h in range(self.n_heads):
            noisy = [max(p + rng.uniform(-self.noise, self.noise), 0.0) for p in target_dist]
            s = sum(noisy)
            out.append([p / s for p in noisy])
        return out
```
(`medusa_heads.py:17-29`)

```python
def medusa_step(
    target_fn: Callable[[List[int]], List[float]],
    heads: MedusaHeads,
    prefix: List[int],
    rng: random.Random,
) -> tuple[List[int], int]:
    """One Medusa iteration: draft `n_heads` then verify with target."""
    ...
```
(`medusa_heads.py:32-38`,签名+docstring)

**一句话:** Classic 投机解码需要单独训练/部署一个独立的小模型当 draft,Medusa 的解法是"不找外人,自己人干":在 target 模型最后一层接几个额外的 LM head,每个 head 各自预测未来第 `i` 步的 token——`MedusaHeads.draft()` 在这份教学实现里没有真训练权重,而是直接对 target 的真实分布加均匀噪声再归一化来**模拟**"一个训练得不错但仍不完美的 head"。

**底层机制/为什么这样设计:** 从最笨的想法讲起——Classic 方案的痛点是"要找一个能力和 target 相配、但又明显更快的独立小模型"这件事本身不容易(小了猜不准、不够小又不够快),Medusa 索性放弃"独立模型"这个思路:反正 target 模型的最后一层隐藏状态 `h` 已经包含了预测下一个 token 所需的绝大部分信息,那就在 `h` 上面并联挂几个轻量 MLP(head),让每个 head 各自学着预测"往后数第 i 个 token"——`head_0` 学正常的下一个 token(其实和 target 自己的 LM head 目标一样),`head_1` 学下下个 token,以此类推。因为只训练这几个小 head(backbone 冻结),训练成本远低于训练一个独立 draft 模型,而且推理时 backbone 只需要 forward 一次(所有 head 共享同一份 `h`),不需要像 Classic 那样额外跑一个独立小模型的完整 forward。本知识点撰写时发现一处**真实的边界情况 bug**:`draft()` 方法用 `max(p + rng.uniform(-noise, noise), 0.0)` 给每个分量加噪声再 clamp 到非负,如果 `noise` 相对于 `target_dist` 里最小的分量足够大,**所有分量都有非零概率同时被 clamp 到 0**,那么 `s = sum(noisy)` 会精确等于 `0.0`,下一行 `p / s` 直接 `ZeroDivisionError` 崩溃——这不是理论推演,是本知识点独立验证时真实复现出来的(见下方"可运行例子"),在 `noise=0.6` 这种远算不上极端的取值下,4000 次调用里就足以触发。这提示一个更普遍的道理:"用均匀噪声扰动一个概率分布再归一化"这种简化建模手法,如果不对 `noise` 相对分布本身的尺度做任何约束,天然存在数值不稳定的风险,真实系统里 Medusa head 的输出是训练出来的、自带 softmax 归一化的合法分布,不会有这个问题——这个坑是"教学 mock 简化带来的副作用",不是 Medusa 算法本身的缺陷。

**AI 研究场景:** Medusa(Cai et al., Princeton 2024)的实际收益(该 lecture 引用的数字)在 Llama-7B 上约 2.2-2.8 倍、Vicuna-13B 上约 2.3 倍——比同期 Classic 投机解码常见的 accept rate 更高(lecture 给出 Medusa accept rate 0.6-0.85 vs Classic 的 0.5-0.8),核心原因是同一个 backbone 出来的 hidden state 天然比"另找一个独立小模型"更贴近 target 自己的知识。它的局限也很直接:head 数量在训练时就固定了,不能推理时动态加减;而且几个 head 是"各自独立"地从同一个 `h` 出发预测,不像后面 EAGLE 那样让后一步的草稿依赖前一步草稿的结果——这正是知识点 4 EAGLE 要改进的地方。

**可运行例子:**
```python
import sys, random
sys.path.insert(0, "learning/speculative-decoding/src")
from common import softmax
from medusa_heads import MedusaHeads, medusa_step

target_fn = lambda prefix: softmax([1.0, 2.0, 0.5, 0.2, 0.1])

def run_medusa_trials(heads, n_trials=4000):
    total_acc, total_draft = 0, 0
    r = random.Random(7)
    for _ in range(n_trials):
        accepted, _ = medusa_step(target_fn, heads, [0], r)
        total_acc += len(accepted) - 1   # 减掉 bonus token
        total_draft += heads.n_heads
    return total_acc / total_draft

ar_low = run_medusa_trials(MedusaHeads(n_heads=4, noise=0.05))   # head 几乎不失真
ar_hi = run_medusa_trials(MedusaHeads(n_heads=4, noise=0.3))     # head 明显失真
assert ar_low > ar_hi   # head 越接近 target,accept rate 越高

# --- 真实边界 bug:noise 足够大时，全部分量同时被 clamp 到 0，除零崩溃 ---
heads_crash = MedusaHeads(n_heads=4, noise=0.9)
crashed = False
try:
    for trial_seed in range(200):
        medusa_step(target_fn, heads_crash, [0], random.Random(trial_seed))
except ZeroDivisionError:
    crashed = True
assert crashed, "expected to reproduce ZeroDivisionError within 200 seeded trials at noise=0.9"
```

**实测(`.venv` 真跑):** 同一个 target 分布下,`noise=0.05` 的 accept rate 精确为 `0.86`,`noise=0.3` 降到 `0.478`——噪声翻 6 倍,accept rate 打对折还多,验证了"head 越贴近 target,接受率越高"这条直觉。`noise=0.9` 场景下,200 个不同随机种子里稳定复现出 `ZeroDivisionError('float division by zero')`,证实这是一个真实、可复现的边界情况,不是猜测。

**面试怎么问 + 追问链:**
- **Q:** "Medusa 和 Classic 投机解码最核心的区别是什么?" —— 期望说出"Classic 用一个独立训练、独立部署的小模型当 draft;Medusa 在 target 模型自己的 backbone 上加几个轻量 LM head,复用同一次 forward 的隐藏状态,只训练这几个 head、backbone 冻结不动"。
- **追问 1:** "Medusa 的几个 head 是各自独立预测,还是互相依赖?" —— 期望说出"各自独立——`head_i` 直接从同一个共享的隐藏状态 `h` 出发预测第 `i` 步,不看其他 head 的输出",并能进一步指出这是 Medusa 相对 EAGLE 的局限(EAGLE 的草稿是链式自回归,后一步能利用前一步已经"确定"的 token)。
- **追问 2(考察是否真的跑过代码,不是背文档):** "如果给 head 加的噪声特别大,这份教学实现会发生什么?" —— 期望明确说"会真的抛 `ZeroDivisionError`——所有 5 个(或若干个)分量同时被扰动到负数、clamp 到 0 之后,分母 `sum(noisy)` 变成 0,后面的归一化除法直接崩溃;这不是理论上的担忧,是真实复现过的",能看出是真的动手测过而不是凭直觉猜。
- **追问 3:** "现实中训练好的 Medusa head 会不会也遇到类似的数值问题?" —— 期望辨析:不会,因为真实 head 内部自带 softmax 归一化(softmax 的分母恒正,不会出现"全部分量都是 0"的情况),这里的 `ZeroDivisionError` 是教学 mock 用"加噪声再手动归一化"这种简化建模方式引入的副作用,不是 Medusa 算法本身的问题——回答这道题的关键是能区分"教学简化引入的 bug"和"算法设计的固有缺陷"。

**常见坑:** 把"多头"理解成"模型并行"或"多卡推理"——Medusa 的多个 head 是在**同一次前向**里输出的(用同一个 `h`,只是接了几个不同的小 MLP),不涉及任何跨设备通信,和 05 号文件会讲的 Tensor/Pipeline Parallel 是完全不同层面的概念,不要混淆。另一个坑是被本知识点发现的 `ZeroDivisionError` 误导成"Medusa 这个算法本身不稳定"——上面的追问链已经辨析过,这是教学 mock 简化带来的边界问题,不代表真实训练出来的 Medusa head 有同样风险。

---

## 4. EAGLE(`eagle_minimal.py`,L04)—— feature 级自回归草稿,以及"同噪声下和 Medusa 打平"这个诚实的度量局限

**是什么:**
```python
@dataclass
class EagleDraft:
    """Single transformer-layer draft running auto-regressively over `k` steps."""
    k: int = 4
    noise: float = 0.30   # quality knob — EAGLE-style closer to target than Medusa

    def draft_sequence(
        self,
        target_dist_fn: Callable[[List[int]], List[float]],
        prefix: List[int],
        rng: random.Random,
    ) -> tuple[List[int], List[List[float]]]:
        """Auto-regressively draft `k` tokens; return (ids, per-step distributions)."""
        local = list(prefix)
        ids: List[int] = []
        qs: List[List[float]] = []
        for _ in range(self.k):
            true_p = target_dist_fn(local)
            noisy = [max(p + rng.uniform(-self.noise, self.noise), 0.0) for p in true_p]
            s = sum(noisy)
            q = [p / s for p in noisy]
            tid = sample_from(q, rng)
            ids.append(tid)
            qs.append(q)
            local.append(tid)
        return ids, qs
```
(`eagle_minimal.py:11-36`)

**一句话:** EAGLE 和 Medusa 的关键差异在于草稿怎么"接力"——Medusa 的 4 个 head 各自独立地从同一个 `h` 出发猜第 1/2/3/4 步;EAGLE 只有 1 个 draft 层,但让它**自回归地**跑 `k` 步,每一步都把上一步刚猜出的 token 接入 `local` 再喂给下一步,这样后面的猜测能利用前面猜测已经"确定"下来的上下文。

**底层机制/为什么这样设计:** 从最笨的想法讲起——Medusa 的每个 head 只看得到同一份"当前"隐藏状态,`head_3` 要预测第 3 步的 token 时,并不知道 `head_1`/`head_2` 实际猜了什么,这在信息论意义上是有损失的(预测第 3 个词,理应参考"如果第 1、2 个词是这样,第 3 个词更可能是什么",而不是三个头各自闭着眼睛独立猜)。EAGLE 的解法是只留 1 个 draft 层,但让它**自回归**:第一步用 `local=prefix` 预测出 token 1,第二步把 token 1 拼进 `local` 再预测 token 2,依此类推——`draft_sequence()` 里 `local.append(tid)` 这一行就是这个"接力"的具体实现,每一步的输入都包含了之前所有已经猜出的 token。lecture(L04)的原始动机是"EAGLE 让 draft 层预测的是下一层的 **feature**(隐藏状态),而不是 token id 本身",feature 是稠密向量、比离散 token id 携带的信息丰富得多,这是 EAGLE 论文里 accept rate 显著高于 Medusa 的根本原因。但这里必须诚实指出这份教学 mock 的一个局限:`draft_sequence()` 并没有真的建模"feature 级信息更丰富"这件事,它和 `MedusaHeads.draft()` 用的是**同一种**"对 target 真实分布加均匀噪声再归一化"的简化手法,唯一的区别只是"要不要把上一步的草稿 token 接入下一步的上下文"这个自回归结构,而不是"噪声来源的信息量"。本知识点用**相同**的 `noise=0.30` 独立对比了 `eagle_step` 和 `medusa_step`(见下方可运行例子),结果两者的 accept rate 几乎没有差异——这不是 bug,而是如实反映了这份 mock 的建模粒度:它精确复现了"EAGLE 自回归 vs Medusa 并联"这个**结构**上的区别(可以用来讲清楚两种机制怎么组织草稿),但没有编码"为什么 feature 级信息比 token 级信息更准"这个**信息量**上的区别——lecture 里 EAGLE 默认 `noise=0.30` 明显低于 Medusa 默认 `noise=0.5`,这组默认值的差异才是 mock 里"EAGLE 看起来更准"的直接原因,不是自回归结构本身带来的。

**AI 研究场景:** EAGLE(Li et al., THU 2024)lecture 引用的 Llama-7B 基准数字是 Medusa 2.5x 对 EAGLE 3.0x,现实中的差距确实来自"feature 比 token id 信息量大"这个论文核心论点,而不是"自回归 vs 并联"这个结构差异本身(结构差异带来的收益更多体现在"后面几步猜测质量能不能跟上"这一点上)。理解这个"mock 建模了结构、没建模信息量"的边界,对判断"这份教学代码到底能拿来说明什么、不能说明什么"很重要——可以拿它讲清楚 EAGLE 自回归链式草稿的工程结构,但不能拿它复现"EAGLE 论文报告的具体加速比数字"。

**可运行例子:**
```python
import sys, random
sys.path.insert(0, "learning/speculative-decoding/src")
from common import softmax
from medusa_heads import MedusaHeads, medusa_step
from eagle_minimal import EagleDraft, eagle_step

target_fn = lambda prefix: softmax([1.0, 2.0, 0.5, 0.2, 0.1])

def run_eagle_trials(draft, n_trials=4000):
    total_acc, total_draft = 0, 0
    r = random.Random(9)
    for _ in range(n_trials):
        accepted, _ = eagle_step(target_fn, draft, [0], r)
        total_acc += len(accepted) - 1
        total_draft += draft.k
    return total_acc / total_draft

def run_medusa_trials_same_seed(heads, n_trials=4000):
    total_acc, total_draft = 0, 0
    r = random.Random(9)   # 和上面用相同种子，公平对照
    for _ in range(n_trials):
        accepted, _ = medusa_step(target_fn, heads, [0], r)
        total_acc += len(accepted) - 1
        total_draft += heads.n_heads
    return total_acc / total_draft

ar_eagle = run_eagle_trials(EagleDraft(k=4, noise=0.30))
ar_medusa_same_noise = run_medusa_trials_same_seed(MedusaHeads(n_heads=4, noise=0.30))
# 关键诚实结论：同噪声水平下，这份 mock 里 EAGLE 并不显著优于 Medusa
assert abs(ar_eagle - ar_medusa_same_noise) < 0.05

# EAGLE 用自己"默认参数"(noise=0.30)对比 Medusa 用"默认参数"(noise=0.5) 才会看出差距
ar_medusa_default = run_medusa_trials_same_seed(MedusaHeads(n_heads=4, noise=0.5))
assert ar_eagle > ar_medusa_default
```

**实测(`.venv` 真跑):** 同一 `noise=0.30`、同一随机种子下,`eagle_step` 的 accept rate 是 `0.486`,`medusa_step` 是 `0.489`——两者相差不到 `0.003`,在统计噪声范围内基本打平,验证了"这份 mock 没有单独编码 EAGLE 的信息量优势"这条结论。换成 Medusa 的**默认**噪声 `0.5`(而不是强行对齐成 0.30)重新对比,`medusa` accept rate 明显低于 `eagle`——说明 lecture 展示的"EAGLE 更强"这个印象,来自两个方法**默认参数选择不同**,不是自回归结构本身在这份简化 mock 里带来的必然优势。

**面试怎么问 + 追问链:**
- **Q:** "EAGLE 相对 Medusa 的核心改进是什么?" —— 期望说出"EAGLE 让 draft 层预测下一层的 feature(隐藏状态)而不是 token id,feature 是稠密向量、信息量比离散 token id 大得多,而且 EAGLE 的草稿是单层自回归、能利用前面已经猜出的 token,不像 Medusa 几个 head 各自独立"。
- **追问 1(核心陷阱,考察是否真的动手对比过):** "这个仓库的教学代码里,如果给 EAGLE 和 Medusa 用完全相同的噪声参数,谁的 accept rate 更高?" —— 期望明确说"基本没有差异(本知识点实测两者相差不到 0.003),因为这份 mock 只是把'draft 质量'简化成一个噪声旋钮,没有真正建模'feature 信息量比 token id 大'这件事——EAGLE 默认参数选得比 Medusa 更接近 target,才是 lecture 展示的性能差距的直接原因",这道题专门筛"只会背结论、没有真的读代码验证过"的候选人。
- **追问 2:** "如果要让这份 mock 更真实地体现 EAGLE 的信息量优势,可以怎么改?" —— 期望能提出方向(不要求写出完整代码):比如让 EAGLE 的噪声模型依赖"已经确定的历史 token 数量"(历史越长、可用信息越多、噪声应该越小),而不是像现在这样每一步都用同一个固定 `noise` 常数独立加噪声——这样才能体现"自回归链式草稿能利用之前几步已经猜对的信息"这个结构性优势。
- **追问 3:** "EAGLE 的 draft 层需要重新训练吗?" —— 期望说出"需要,虽然只训练 1 个 transformer 层(远比训练一个完整的独立小模型便宜),backbone 依然冻结,LM head 复用 target 自己的——这是它和 Medusa 共享的'轻量微调、不动主干'这条设计哲学"。

**常见坑:** 只看 lecture 给出的加速比数字表格(Medusa 2.5x vs EAGLE 3.0x),就以为"EAGLE 算法结构本身天然比 Medusa 强",而不去看这份仓库自己的教学代码到底建模了什么——本知识点已经用实测证明这份 mock 里两者在同噪声下几乎没有差异,真实差距的来源(feature 信息量)在这份简化实现里被压缩成了"默认参数选得更保守"这一件事。这也是这个系列反复强调的一条纪律:教学 mock 能验证的是"结构/流程对不对",不代表能验证"某篇论文报告的具体数字",两者是不同粒度的问题。

---

## 5. EAGLE-2(`eagle2.py`,L05)—— 用置信度剪枝的动态草稿树,一次验证多条候选路径

**是什么:**
```python
@dataclass
class TreeNode:
    tokens: List[int] = field(default_factory=list)
    logprob: float = 0.0
    depth: int = 0


def build_dynamic_tree(
    target_fn: Callable[[List[int]], List[float]],
    prefix: List[int],
    K: int = 8,
    max_depth: int = 5,
    branch: int = 3,
    noise: float = 0.25,
    rng: random.Random = None,
) -> List[TreeNode]:
    """Beam-search expand a draft tree; keep K most promising leaf paths."""
    rng = rng or random.Random(0)
    leaves: List[Tuple[float, int, TreeNode]] = []
    counter = 0
    root = TreeNode()
    heappush(leaves, (0.0, counter, root))
    finals: List[TreeNode] = []
    while leaves and len(finals) < K:
        neg_lp, _, leaf = heappop(leaves)
        ...
```
(`eagle2.py:13-42`,节选)

**一句话:** EAGLE-1 每一步只沿着**一条**路径自回归猜到底,如果中途某个 token 被拒绝,后面全部猜测都白费;EAGLE-2 用一个**优先队列(最小堆)**驱动的 beam-search 式过程,同时维护多条候选路径,按累积 log 概率排序,只把"最有希望"的 `K` 条路径展开到底,提前砍掉低概率分支。

**底层机制/为什么这样设计:** 从最笨的想法讲起——EAGLE-1 单路径自回归的浪费很直观:假设第 2 步猜错了,那第 3、4 步即便"猜得再准"也没有意义(因为它们是建立在第 2 步这个错误猜测之上的),整条链从第 2 步往后全部作废。EAGLE-2 的解法是不要只押一条路径:`build_dynamic_tree` 用一个堆(Python `heapq`,存的是 `(-累积logprob, 计数器, 节点)` 三元组,取负是因为 `heapq` 是小顶堆、要模拟"取最大 logprob"的效果)反复弹出当前最有希望的叶子节点,对它展开 `branch` 个子节点(取 target 分布 top-`branch` 的候选 token),新的候选重新入堆——这本质上是 beam search 的标准写法,只是这里的"beam"不是固定宽度、而是靠堆自然维持"总是先展开最有希望的分支"这个性质。`counter` 变量的作用是给堆里的元素一个次要排序键,避免两个 `TreeNode` 因为不支持比较运算符(`<`)而在 `logprob` 恰好相等时让 `heapq` 尝试比较 `TreeNode` 对象本身报错——这是用 `heapq` 存自定义对象时一个容易被忽略的工程细节,不加这个 `counter`,一旦出现 `neg_lp` 相同的两个候选,`heapq` 内部比较退化到比较第二个元素,如果那也相同就会尝试比较第三个元素(即 `TreeNode`),而 `TreeNode` 没有定义 `__lt__`,会抛 `TypeError`。展开到 `max_depth` 的叶子直接收进 `finals`,循环直到攒够 `K` 条或者堆空了为止。

**AI 研究场景:** EAGLE-2(Li 2024.06)lecture 给出的收益数字是 Llama-7B/Vicuna-7B 上从 EAGLE-1 的 3.0x 提升到 4.0-4.5x,提升幅度和"树宽 `K` 能覆盖多少条有潜力的候选路径"直接相关。这套"动态展开、按置信度剪枝"的思路不是 EAGLE-2 独有——02 号文件 SGLang 的 Agent 场景里,`agent_patterns.py` 处理 ReAct/ToT 等多路径推理时也要面对类似的"要不要同时探索多条候选、怎么决定先展开哪条"的问题,本质上是同一类"用优先队列管理搜索前沿"的通用技巧在不同场景下的复用。

**可运行例子:**
```python
import sys, random
sys.path.insert(0, "learning/speculative-decoding/src")
from common import softmax
from eagle2 import build_dynamic_tree, topk, TreeNode

target_fn = lambda prefix: softmax([2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02])

tree = build_dynamic_tree(target_fn, [0], K=6, max_depth=4, branch=2, noise=0.15, rng=random.Random(3))
assert len(tree) == 6
assert all(isinstance(n, TreeNode) for n in tree)
assert all(n.depth <= 4 for n in tree)   # 没有节点超过 max_depth

tks = topk([0.1, 0.5, 0.05, 0.3, 0.05], 2)
assert tks == [1, 3]   # 概率最大的两个索引，按概率从大到小排序
```

**实测(`.venv` 真跑):** `K=6, max_depth=4, branch=2` 的配置下,精确得到 6 条叶子路径,深度分布是 `[4, 4, 4, 4, 4, 4]`——这组参数下(`branch=2` 每层扩两个子节点,4 层理论上限 `2^4=16` 条路径,`K=6` 远小于上限)全部 6 条最终路径都自然展开到了 `max_depth`,说明堆没有在中途因为"没有更多候选"而提前耗尽。`topk([0.1,0.5,0.05,0.3,0.05], 2)` 精确返回 `[1, 3]`(对应概率 0.5 和 0.3 的两个索引,按概率降序排列)。

**面试怎么问 + 追问链:**
- **Q:** "EAGLE-2 相对 EAGLE-1 解决了什么问题?" —— 期望说出"EAGLE-1 只沿单条路径自回归猜测,一旦中途被拒绝,后面的猜测全部浪费;EAGLE-2 用置信度驱动的动态树同时维护多条候选路径,只展开最有希望的分支,不在低概率路径上浪费 verify 预算"。
- **追问 1(工程细节,考察是否读过源码):** "`build_dynamic_tree` 里 `heappush` 的元组第二个元素 `counter` 是干什么用的?" —— 期望说出"防止两个节点的 `-logprob` 恰好相等时,`heapq` 退化去比较第三个元素(`TreeNode` 对象本身),而 `TreeNode` 没有定义比较运算符会抛错;`counter` 保证元组的前两项就能唯一确定排序,不会比较到不可比较的对象"。
- **追问 2:** "如果 `branch` 设得很大(比如 branch=vocab_size),这棵树会有什么问题?" —— 期望说出"树会指数级膨胀(每层节点数是 `branch^depth`),即便用堆按置信度排序、只保留 `K` 条最终路径,构建过程中入堆/出堆的开销依然会随 `branch` 增长,而且大部分候选的置信度会很低,verify 阶段没有意义——这是为什么 EAGLE-2 论文和这里的默认参数都把 `branch` 控制在 2-3 这种小数值"。
- **追问 3:** "EAGLE-2 的树最终怎么'验证'?" —— 期望回答能连到知识点 9(Tree Attention):把树展开成一个 flat token 序列,用一个专门构造的 attention mask(保证每个 token 只看到自己在树里的祖先链),一次 target forward 就能算出所有候选路径每个位置的真实概率,不需要对每条路径分别做一次 forward。

**常见坑:** 把"EAGLE-2 的树"和"Medusa 的候选组合"当成同一个概念——Medusa 的几个 head 各自独立输出,候选之间没有"父子"关系,谈不上"剪枝"这个操作;EAGLE-2 的树是显式的父子结构(`TreeNode.tokens` 记录从根到当前节点的完整路径),剪枝剪掉的是"整条子树",这是两种本质不同的搜索空间组织方式。另一个坑是假设 `K` 条最终路径互相之间没有重叠前缀——实际上多条路径可以共享同一段祖先(比如例子里深度为 4 的 6 条路径,很可能有多条共享前 1-2 步的祖先节点),这正是 Tree Attention(知识点 9)要处理"共享前缀只算一次"这件事的意义所在。

---

## 6. EAGLE-3(L06,概念性,本仓库无对应源码)—— multi-feature 输入,用共享速度公式做方向性合理性检验

**是什么:** 本知识点没有 `是什么` 意义上的源码摘录——`learning/speculative-decoding/README.md` 的总览表格里,L06 那一行"代码"列的取值就是 `—`,`src/` 目录下没有任何以 `eagle3` 命名的实现文件(`capstone_eagle3.py` 虽然文件名带 `eagle3`,但内容是知识点 10 讲的"4 method × 5 task 对照 capstone",并不是 EAGLE-3 算法本身的实现)。

**一句话:** EAGLE-3(Li 2025)相对 EAGLE-2 的改进点是让 draft 层同时吃进 backbone **多层**的隐藏状态(而不只是最后一层),lecture 给出的直觉是"最后一层隐藏状态已经'定型'到接近最终 token,而更早层的隐藏状态还保留更多语义信息,拼接多层能让 draft 掌握的信息更丰富"——这是一个纯概念性的知识点,本文不假装有代码可以运行验证这个具体机制。

**底层机制/为什么这样设计:** 从最笨的想法讲起——EAGLE-1/2 的 draft 层输入只有 backbone **最后一层**的隐藏状态,这一层的信息已经高度"压缩"向最终要预测的 token(因为它马上就要喂给 LM head 输出 logits 了),某种意义上"信息已经被消费得差不多了"。EAGLE-3 的思路是往前多要一点信息:把 backbone 中间几层(lecture 举例 `h_8, h_16, h_24, h_32`)的隐藏状态拼接起来一起喂给 draft 层——更早的层保留了更多"还没决定最终往哪个方向收敛"的语义信息,draft 层能用的信息量更大,自然有机会猜得更准。lecture 同时提到 EAGLE-3 引入了"训练时用 rollout 轨迹做 teacher-forcing"这个训练策略上的改动,以及支持 scale 到 32B+ 的 target 模型——这些都是训练方法论和工程扩展性层面的改进,和 multi-feature 这个架构改动是分开的两件事。因为这些改动全部发生在"draft 层具体怎么训练、喂什么输入"这个粒度,而这份教学仓库里从 EAGLE-1 到 EAGLE-2 的所有 mock 实现都没有真的训练/加载任何权重(全部是对 target 分布加噪声模拟),EAGLE-3 的 multi-feature 机制本身没有对应的可运行代码可以验证——诚实的做法是承认这一点,而不是编一份"看起来像 EAGLE-3"但实际上只是又一份加噪声 mock 的代码。本知识点退而求其次,用知识点 2 已经验证过的 `walltime_speedup()` 公式(这个公式对所有投机解码变体通用,不专属于任何一个具体算法)代入 EAGLE-3 lecture 给出的场景做一次方向性检验:用"目标模型相对草稿的成本比"来近似"目标模型越大,草稿相对越便宜"这个定性趋势,看数字是否符合"target 越大,加速比越高"这个 lecture 给出的定性结论——这只是一次合理性检验(sanity check),不是对 EAGLE-3 具体加速比数字(lecture 给出 70B 上 5.0-6.0x)的复现。

**AI 研究场景:** 这个知识点提醒一个重要的方法论:精读一个课程仓库时,如果发现某个主题**没有**对应源码,正确的做法是如实标注"这一部分是概念性的,本仓库没有实现",而不是勉强拿相邻主题的代码去"凑"一个看起来相关的运行例子——那样反而会让读者误以为某个具体机制已经被验证过。EAGLE-3 目前(lecture 原话)"全面胜过 Medusa-2,是业界 SOTA",据 lecture 已经集成进 HuggingFace(`spec_decode_method="eagle"`)、vLLM 0.7+、SGLang,如果要真正验证 EAGLE-3 的效果,需要去这些真实项目里跑,不是这份教学仓库能覆盖的范围。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/speculative-decoding/src")
from speculative_original_minimal import walltime_speedup

# L06 lecture 定性claim: "70B target + 1B draft -> 5x", "7B target + 200M draft -> 3x"
# 用 draft_cost_ratio ≈ draft 相对 target 的成本占比做一个粗糙的规模代理，
# 不是对 EAGLE-3 具体数字的复现，只检验"target 越大、加速比越高"这个方向趋势。
su_70b_proxy = walltime_speedup(alpha=0.75, gamma=5, draft_cost_ratio=1 / 70)
su_7b_proxy = walltime_speedup(alpha=0.75, gamma=5, draft_cost_ratio=1 / 35)
assert su_70b_proxy > su_7b_proxy   # 定性趋势成立：target 越大，相对加速比越高

# 明确标注：这不是 EAGLE-3 的实现，只是复用全系列共享的通用公式做合理性检验
assert not any("eagle3" in name.lower() for name in ["speculative_original_minimal"])
```

**实测(`.venv` 真跑):** 用同样的 `alpha=0.75, gamma=5`,只把 `draft_cost_ratio` 从 `1/35`(模拟"7B 级别 target")换成 `1/70`(模拟"70B 级别 target,草稿相对更便宜"),`walltime_speedup` 从 `2.88x` 升到 `3.07x`——方向和 lecture 给出的"target 越大、加速比越高"定性结论一致,但绝对数值(2.88x/3.07x)明显低于 lecture 引用的 EAGLE-3 具体数字(3x/5-6x),这是预期之内的:这里用的是一个粗糙的规模代理和通用公式,不是 EAGLE-3 论文的真实测量结果,两者不应该被当成同一件事去对比。

**面试怎么问 + 追问链:**
- **Q:** "EAGLE-3 相对 EAGLE-2 的核心改进是什么?" —— 期望说出"draft 层输入从只用 backbone 最后一层隐藏状态,改成拼接多层(比如第 8/16/24/32 层)的隐藏状态,更早的层保留更多未压缩的语义信息,让草稿能用的信息量更大"。
- **追问 1(诚实性检验,专门筛过度自信的回答):** "你能用这个仓库的代码演示一下 EAGLE-3 具体怎么工作吗?" —— 期望明确说"不能,这个仓库对 EAGLE-3 没有专门的实现文件,只有 lecture 概念性描述;我用的是全系列通用的速度公式做了一次方向性的合理性检验,不是 EAGLE-3 机制本身的复现",能诚实说出"做不到什么"和能吹嘘"做到了什么"同样重要,是候选人诚信度的一个考察点。
- **追问 2:** "为什么 EAGLE-3 要额外引入'rollout 轨迹 teacher-forcing'这个训练策略上的改动?" —— 期望能说出大致方向:draft 层如果只在"target 每一步都对"的干净轨迹上训练,容易在真实推理时(target 有时也会犯错、draft 需要跟着调整)遇到训练/推理分布不匹配的问题;用 rollout(真实自回归生成出来、可能包含误差的轨迹)做训练能缓解这种 exposure bias,这是候选人如果读过 EAGLE 系列论文才会知道的细节,如果不知道,合理的回答是坦率说"没有深入研究这一点"而不是编造。
- **追问 3:** "如果要在这个仓库里补全 EAGLE-3 的 mock 实现,大致需要做什么改动?" —— 期望能提出合理方向(不要求写代码):在 `eagle_minimal.py` 或新文件里,让"草稿质量"的建模除了当前的噪声大小之外,再引入一个"能看到几层历史信息"的参数,信息层数越多、有效噪声越小,以此近似"multi-feature 输入让草稿更准"这个定性效果——这本质上和知识点 4 讨论"如何让 mock 更真实体现 EAGLE 相对 Medusa 的优势"是同一类改造思路。

**常见坑:** 因为 EAGLE-3 是"热门 SOTA 方法",下意识假设仓库里一定有对应代码,没有先去确认 `src/` 目录和 lecture 总览表格——遇到"这个主题有没有实现"这类问题,第一步永远是先去看模块自己的 README/总览表格,而不是假设"应该有"。另一个坑是把知识点 6 给出的数字(2.88x/3.07x)错误引用成"EAGLE-3 的真实加速比"——这两个数字来自一个粗糙的规模代理和通用公式,不是 EAGLE-3 论文或任何真实系统的测量结果,如果被拿去做进一步论证会构成误导。

---

## 7. Lookahead Decoding(`lookahead.py`,L07)—— 不需要 draft 模型:n-gram 池 + Jacobi 式并行迭代,一次真实 forward 换四个 token

**是什么:**
```python
@dataclass
class NgramPool:
    """Maps prefix -> next-token id seen recently."""
    pool: Dict[Tuple[int, ...], int] = field(default_factory=dict)
    n: int = 3

    def add_sequence(self, seq: List[int]) -> None:
        for i in range(len(seq) - self.n):
            self.pool[tuple(seq[i : i + self.n])] = seq[i + self.n]

    def lookup(self, prefix: List[int]) -> int | None:
        if len(prefix) < self.n:
            return None
        return self.pool.get(tuple(prefix[-self.n :]))
```
(`lookahead.py:9-22`)

```python
def lookahead_step(
    target_fn: Callable[[List[int]], int],
    prefix: List[int],
    pool: NgramPool,
    lookahead_branches: int = 4,
) -> List[int]:
    """Generate next token AND lookahead branches; verify if any matches pool."""
    accepted = [target_fn(prefix)]
    local = prefix + accepted
    for _ in range(lookahead_branches):
        nxt = pool.lookup(local)
        if nxt is None:
            break
        accepted.append(nxt)
        local.append(nxt)
    pool.add_sequence(prefix + accepted)
    return accepted
```
(`lookahead.py:25-42`)

**一句话:** 前面 3-6 知识点全部需要一个额外的"草稿来源"(独立小模型/多头/自回归 feature 层),Lookahead 完全不需要——它维护一个"最近见过的 n-gram → 下一个 token"的哈希表(`NgramPool`),每次真实调用 target 生成 1 个 token 之后,如果当前上下文的最近 `n` 个 token 恰好命中池子里记录过的模式,就"白嫖"式地直接把池子里记的后续 token 接上去,不需要再调用 target。

**底层机制/为什么这样设计:** 从最笨的想法讲起——Classic/Medusa/EAGLE 都要解决"怎么造一个又快又准的草稿来源"这个问题,而 Lookahead 换了个思路:很多真实生成任务里(比如代码里重复的样板结构、数学解题里重复出现的表达式模式),同一个 n-gram 组合会在同一次生成过程里反复出现——如果之前已经"真的"通过 target 生成过 `"for i in range("` 后面接的是 `"len(...)"`,那下次再看到同样的 3-token 前缀,直接复用这个记录,不需要再问一次 target。`NgramPool.add_sequence()` 把一段序列里所有长度为 `n+1` 的滑动窗口都记下来(前 `n` 个 token 当 key,第 `n+1` 个 token 当 value);`lookahead_step()` 的流程是:先老老实实调用一次 `target_fn` 得到 1 个**真实**token(这一步不能省,保证至少有 1 个 token 是货真价实的模型输出),然后用这个新 token 更新出的最新窗口去查表,如果命中就直接接上池子里记的 token(不调用 target),再用**新的**窗口继续查,直到连续 `lookahead_branches` 次都没查到或者达到上限——最后把这一整段(1 个真实 + 若干个池子命中)都喂回 `add_sequence()`,让池子越用越丰富。这套机制严格来说**不是**投机解码原本"draft+verify"的框架(没有 rejection sampling、没有概率意义上的"零 bias"保证),它更接近一种缓存复用:如果历史上模型在类似上下文下真的生成过某个 token,直接复用这个既成事实,某种程度上默认"相同的 n-gram 前缀,下一个 token 大概率还是一样的"这个经验假设,在结构化程度高(重复 pattern 多)的任务上这个假设通常成立,在高熵、创造性任务上就不一定成立了。

**AI 研究场景:** Lookahead Decoding(UCSD 2024)lecture 给出的收益因任务而异:数学(重复 pattern 多)2-3x、代码(boilerplate 多)2.5x,通用 chat 只有 1.2-1.5x——这个"任务结构化程度决定收益"的规律和知识点 1 讨论的"accept rate 受 target 分布尖锐程度影响"是同一类现象在不同机制上的体现。它最大的工程吸引力是**不需要任何训练、不需要额外模型、不占用额外显存**(相比 Medusa/EAGLE 都要训练至少一小部分参数),对于"来不及训练专用 draft、但想立刻拿到一些加速"的部署场景是现成可用的选项。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/speculative-decoding/src")
from lookahead import NgramPool, lookahead_step

pool = NgramPool(n=3)
seq_history = [10, 20, 30, 40, 10, 20, 30, 41]
pool.add_sequence(seq_history)
# 滑窗记录: (10,20,30)->41(后写覆盖先写), (20,30,40)->10, (30,40,10)->20, (40,10,20)->30
assert pool.lookup([10, 20, 30]) == 41

# 构造一个会形成"链式命中"的场景：真实 target 第一次调用返回 40，
# 此后 (20,30,40)->10 ->(30,40,10)->20 ->(40,10,20)->30 ->(10,20,30)->41 依次命中，
# 全程只调用了 1 次 target_fn，却产出 4 个 token。
call_count = [0]
def tgt_fn(prefix):
    call_count[0] += 1
    return 40

out = lookahead_step(tgt_fn, [10, 20, 30], pool, lookahead_branches=3)
assert out[0] == 40          # 第一个 token 永远是真实 target 调用
assert out == [40, 10, 20, 30]
assert call_count[0] == 1    # 只用 1 次真实 forward 就产出了 4 个 token
```

**实测(`.venv` 真跑):** 用 `[10,20,30,40,10,20,30,41]` 播种池子后,`pool.lookup([10,20,30])` 精确返回 `41`(后写覆盖先写,证实"最近一次见过的延续"会覆盖更早的记录)。构造"第一个真实 token 恰好接上已知窗口"的场景后,`lookahead_step` 精确产出 `[40, 10, 20, 30]` 这 4 个 token,而 `target_fn` 的调用计数器全程只加了 `1`——验证了"1 次真实 forward + 3 次纯查表"就能产出 4 个 token 这个核心卖点。

**面试怎么问 + 追问链:**
- **Q:** "Lookahead Decoding 和前面 Classic/Medusa/EAGLE 这几种投机解码方法最本质的区别是什么?" —— 期望说出"前几种都需要一个额外的草稿来源(独立模型/多头/自回归层),Lookahead 完全不需要训练任何东西,靠维护一个 n-gram 到下一个 token 的查找表,命中历史模式就直接复用"。
- **追问 1(考察是否理解正确性边界):** "Lookahead 复用池子里的 token,还需要像 Classic 那样做 rejection sampling 验证吗?" —— 期望说出"不需要,也做不了——池子里存的是历史上模型自己真实生成过的 token,不是一个独立的'draft 分布' `q`,没有对应的 `p`/`q` 可以拿来算接受概率;Lookahead 本质是缓存复用,依赖的假设是'历史上相似上下文的延续,现在大概率还成立',不是投机解码原本'零 bias'那套数学保证"。
- **追问 2:** "什么样的任务最适合用 Lookahead,什么样的任务不适合?" —— 期望说出"重复 pattern 多的任务(代码 boilerplate、数学推导里重复出现的表达式)适合,因为同一个 n-gram 前缀真的会反复延续到相同的后续;高熵、强创造性的任务(故事续写、开放式聊天)不适合,因为相同的前几个词后面接什么变化很大,池子命中率低,收益有限"。
- **追问 3:** "`NgramPool` 会不会无限增长,内存有没有上限?" —— 期望能指出这份教学实现里 `pool: Dict` 确实没有做任何大小限制或过期淘汰机制(每次 `add_sequence` 只会往字典里加新 key、覆盖旧 key,不会主动删除),lecture 自己在"缺点"部分也提到"内存 ↑(n-gram pool)"是已知代价,真实生产系统需要额外的淘汰策略(比如 LRU),这里的教学版本没有实现。

**常见坑:** 把 Lookahead 的"命中"和投机解码的"接受"划等号,以为两者都有同样的正确性保证——如上面追问 1 辨析过,Lookahead 没有 `p/q` 意义上的严格验证,它是一种基于经验假设的缓存复用,理论上可能复用到一个在当前上下文下其实不对的历史延续(只是概率上通常问题不大)。另一个坑是以为 `lookahead_branches` 设得越大越好——`lookahead_branches` 只控制"最多尝试查多少次表",如果池子命中率本来就低,调大这个参数不会增加收益,只会增加每次查表的开销(虽然查表本身很便宜,但也不是完全免费)。

---

## 8. Self-Speculative Decoding(`self_spec.py`,L08)—— 跳层当草稿:零额外显存,但 accept rate 天然更低

**是什么:**
```python
@dataclass
class SelfSpec:
    skip_layers: int = 16
    base_noise: float = 0.5

    def draft(
        self,
        target_dist_fn: Callable[[List[int]], List[float]],
        prefix: List[int],
        rng: random.Random,
    ) -> tuple[int, List[float]]:
        true_p = target_dist_fn(prefix)
        # more skipped layers => higher noise
        noise = self.base_noise * self.skip_layers / 32
        noisy = [max(p + rng.uniform(-noise, noise), 0.0) for p in true_p]
        s = sum(noisy)
        q = [p / s for p in noisy]
        return sample_from(q, rng), q
```
(`self_spec.py:13-30`)

**一句话:** 前面所有方法(Classic/Medusa/EAGLE)都需要"另外的"参数(独立模型权重,或额外训练的 head/层),Self-Spec 连这个都不要——直接用**同一个** target 模型做草稿,只是 forward 时故意跳过一部分层(`skip_layers`),跳得越多、draft 越快但也越不准,`noise = base_noise * skip_layers / 32` 这一行公式就是这份 mock 对"跳层越多、和真实输出偏差越大"这条直觉的量化建模。

**底层机制/为什么这样设计:** 从最笨的想法讲起——Medusa/EAGLE 虽然比 Classic 省了"训练一个完全独立的小模型"这个成本,但依然需要额外训练一小部分新参数(head 或 draft 层),依然需要占用一点额外显存来存这些新参数。Self-Speculative Decoding 的思路更激进:target 模型的浅层(靠近输入的层)本身就已经能算出一个"不够精确但方向大致对"的中间表示,那能不能干脆用"跳过后面若干层、只用浅层算出来的近似结果"直接当草稿?这样完全不需要任何额外参数、任何额外训练——draft 和 target 是**同一份权重**,唯一的区别是要不要跑完全部层。`draft()` 方法里的 `noise = base_noise * skip_layers / 32` 是这份 mock 对"跳过的层数越多,输出和真实 target 分布的偏差越大"这个直觉的一个线性近似(注意分母固定是 32,这里隐含假设了 target 模型总共有 32 层,`skip_layers` 是相对这个总层数的比例);真实系统里"跳过后面 16 层是否能得到一个有意义的中间预测"其实高度依赖模型架构和训练方式(LayerSkip,Meta 2024.04,专门用"训练时随机 layer dropout"让模型具备这种"可以在任意深度提前退出"的能力,不是随便一个训练好的模型跳层就能用),这份线性公式只是一个教学层面的粗糙近似,不是真实跳层后 accept rate 的精确预测模型。

**AI 研究场景:** Self-Spec 最大的吸引力是"零额外显存",在显存极度紧张、或者不想为了投机解码专门训练/维护一份额外参数的场景下是现成可用的选项;代价是 lecture 给出的加速比只有 1.5-2x,明显低于 EAGLE 的 3-5x——这是一条清晰的权衡:省下的训练/显存成本,换来的是更低的 accept rate 上限。LayerSkip 这类"训练时就为跳层做准备"的方法能把加速比做到 1.86x(仍然低于 EAGLE 系列),说明"完全不训练任何东西"和"愿意为跳层单独训练一点东西"之间存在一个明显的性能差距,05 号文件会讨论的分布式推理场景里"哪些优化值得为它单独付出训练/工程成本"是同一类权衡思路的延伸。

**可运行例子:**
```python
import sys, random
sys.path.insert(0, "learning/speculative-decoding/src")
from common import softmax
from self_spec import SelfSpec

target_fn = lambda prefix: softmax([2.0, 1.0, 0.3])

ss_16 = SelfSpec(skip_layers=16, base_noise=0.5)   # 跳一半层
ss_32 = SelfSpec(skip_layers=32, base_noise=0.5)   # 跳满 32 层(全跳)
assert ss_16.base_noise * ss_16.skip_layers / 32 == 0.25
assert ss_32.base_noise * ss_32.skip_layers / 32 == 0.5

def measure_argmax_agreement(ss, n=6000):
    r = random.Random(5)
    agree = 0
    true_p = target_fn([0])
    for _ in range(n):
        tok, _ = ss.draft(target_fn, [0], r)
        agree += 1 if true_p[tok] == max(true_p) else 0
    return agree / n

acc16 = measure_argmax_agreement(ss_16)
acc32 = measure_argmax_agreement(ss_32)
assert acc16 >= acc32   # 跳的层数越少（skip_layers 越小），草稿越接近 target
```

**实测(`.venv` 真跑):** `skip_layers=16` 时噪声公式给出 `0.25`,`skip_layers=32`(跳满全部层)给出 `0.5`,和"跳层越多、noise 越大"的线性设计吻合。用"抽样 token 是否落在 target 分布众数上"这个粗糙指标衡量草稿质量,`skip_layers=16` 的命中率是 `0.643`,`skip_layers=32` 是 `0.606`——跳得少确实更准,但差距(`0.643` vs `0.606`)比知识点 3 里 Medusa 噪声从 `0.05` 到 `0.3` 造成的差距(`0.86` vs `0.478`)要小得多,提示这份线性 noise 公式对"跳层比例"这个变量本身不算敏感,和真实系统里"跳层数量对输出质量的影响是非线性的、和具体跳哪些层强相关"这个更复杂的现实情况相比是一个相当粗糙的简化。

**面试怎么问 + 追问链:**
- **Q:** "Self-Speculative Decoding 相比 Medusa/EAGLE,最大的优势和最大的代价分别是什么?" —— 期望说出"优势是完全不需要训练任何额外参数、不占用任何额外显存,draft 和 target 是同一份权重;代价是 accept rate 明显更低(lecture 给出 1.5-2x,远低于 EAGLE 的 3-5x),因为'跳过部分层'这个近似手段本身引入的误差比专门训练出来的草稿机制更大"。
- **追问 1:** "是不是任何一个训练好的模型,拿来跳几层当草稿都能用?" —— 期望说出"不一定——本知识点提到的 LayerSkip(Meta 2024.04)专门在训练阶段引入 layer dropout,让模型具备'可以在任意深度提前退出仍给出有意义预测'这个能力;一个完全按标准方式训练、从没见过'提前退出'这种情况的模型,直接跳层很可能给出质量差很多的预测,这份仓库里的线性 noise 公式只是教学近似,不代表真实跳层效果"。
- **追问 2(考察数值敏感度直觉):** "为什么这份 mock 的 noise 公式里,skip_layers 从 16 变到 32(翻倍),但衡量出来的草稿质量差距没有像知识点 3 Medusa 噪声翻 6 倍时那么悬殊?" —— 期望能说出"因为这里 noise 从 0.25 只涨到 0.5(只翻了一倍,而且起点、终点数值都不算特别极端),而 Medusa 那组对比是 0.05 到 0.3(翻了 6 倍,且起点极小),两组对比的噪声变化幅度本身不一样,不能直接类比谁的'跳层/加噪声敏感度'更高——这提示做这类对比实验时要注意控制变量,如果想公平比较两种机制对参数变化的敏感程度,应该让噪声变化的相对倍数一致"。
- **追问 3:** "Self-Spec 的 draft 和 verify 阶段,显存占用曲线大概是什么样?" —— 期望说出"因为 draft 和 target 是同一份权重,不需要像 Classic 那样同时装两个模型的权重进显存;唯一的额外开销是要维护'浅层输出的中间激活值'直到验证完成,这个开销远小于装载一个独立的小模型"。

**常见坑:** 把"跳层"简单理解成"用一个更小的模型"——本质区别是 Self-Spec 用的是**同一份**权重(只是不跑全部层),没有额外的模型文件、没有独立的参数量,这和知识点 3-4 里"用另一个专门训练的小模型/head/层"是完全不同的资源占用模式。另一个坑是想当然认为"跳的层数和 accept rate 下降幅度成正比"——真实系统里跳哪些层(浅层 vs 深层、连续跳 vs 间隔跳)对最终质量的影响是高度非线性、和具体模型架构强相关的,这份仓库的线性公式只是为了教学演示"跳层越多,草稿越不准"这个方向性直觉,不能当成量化预测真实系统行为的模型来使用。

---

## 9. Tree Attention(`tree_attention.py`,L09+L10 合并)—— 一次 forward 验证整棵候选树,靠一张 mask 让分叉互不干扰

**是什么:**
```python
def build_tree_mask(parent_idx: List[int]) -> torch.Tensor:
    """Return [N, N] bool mask: mask[i, j] = j is an ancestor of i (incl. i)."""
    n = len(parent_idx)
    mask = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        cur = i
        while cur != -1:
            mask[i, cur] = True
            cur = parent_idx[cur]
    return mask


def tree_attention_torch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, parent_idx: List[int],
) -> torch.Tensor:
    """Compute self-attn with a tree mask."""
    n, d = q.shape
    scale = 1.0 / math.sqrt(d)
    scores = q @ k.transpose(0, 1) * scale
    mask = build_tree_mask(parent_idx)
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v
```
(`tree_attention.py:10-35`)

**一句话:** 知识点 5(EAGLE-2)构造出的候选树如果对每条路径分别做一次 attention 计算,`K` 条路径就要做 `K` 次(还要重复算共享的公共前缀部分);`build_tree_mask` 把整棵树摊平成一个 `[N]` 的 token 序列,构造一个 `[N,N]` 的布尔 mask(`mask[i,j]=True` 当且仅当 `j` 是 `i` 在树里的祖先,包含自己),`tree_attention_torch` 用这张 mask 做**一次**标准的 self-attention,就能让每个 token 只关注到自己这条路径上的祖先,分叉的兄弟节点之间互不可见。

**底层机制/为什么这样设计:** 从最笨的想法讲起——如果 EAGLE-2 构造出的候选树有 `K=32` 条路径,最朴素的验证方式是对每条路径单独 batch 一次做 attention(把树"拍扁"成 32 个独立序列),这样计算量是 `32×` 单条路径的开销,而且路径之间共享的公共前缀(树的上半部分)会被重复计算 `32` 次。`build_tree_mask` 的做法是先把整棵树按某种顺序编号(比如 BFS 编号),`parent_idx[i]` 记录节点 `i` 的父节点编号(根节点是 `-1`);`build_tree_mask` 对每个节点 `i`,沿着 `parent_idx` 链一路往上走到根,把沿途经过的每个祖先节点在 `mask[i,:]` 这一行标成 `True`——这样 `mask[i,j]=True` 精确编码"`j` 是 `i` 的祖先(或就是 `i` 自己)"。`tree_attention_torch` 拿这张 mask 直接套进标准 attention 的 `scores.masked_fill(~mask, -inf)` 这一步:`i` 位置的 query 在算 softmax 归一化之前,凡是不在自己祖先链上的 `j` 位置的 score 都被设成负无穷,softmax 之后这些位置的权重精确为 `0`——效果就是"`i` 只能看到自己这条路径上出现过的 token",兄弟分支的信息完全被屏蔽,但公共祖先部分的 K/V 只需要在整个 `[N,d]` 的 `k`/`v` 张量里算 1 次,不会因为多条路径共享它而被重复计算。这就是"一次 forward 验证一整棵树"的数学基础:不是靠某种特殊的稀疏 kernel 才能实现,标准 attention 加一张合适构造的 mask 就足够表达"树状依赖关系"这个结构。

**AI 研究场景:** L09 lecture 给出的对比数字很直观:32 条路径如果各自独立 batch 验证,开销是 `32×`;用 tree attention,开销只有 `1×`(加上构造 mask 本身微小的开销)。lecture 提到 FlashAttention v2/v3 都原生支持 `tree_attention_mask`,vLLM 0.7+/SGLang 都把这个能力集成进了各自的 `spec_decode` 实现——这不是一个只停留在教学演示层面的技巧,是投机解码从"单路径草稿"升级到"多路径树状草稿"(Medusa 的候选组合、EAGLE-2/3 的动态树)之后,能够真正落地成生产系统的关键工程支撑。这套"共享前缀只算一次、用 mask 区分不同分支的可见性"的思路,和 01 号文件知识点 3 PagedAttention 里"多个请求共享物理 block、只在分叉点才复制"的 COW 机制,以及 02 号文件 RadixAttention"多请求共享 radix tree 前缀"是同一个底层直觉(避免为共享的计算/存储重复付出成本)在三个不同层面(显存管理、KV 复用、attention 计算本身)的三种具体实现。

**可运行例子:**
```python
import sys, math
sys.path.insert(0, "learning/speculative-decoding/src")
import torch
from tree_attention import build_tree_mask, tree_attention_torch

# 树结构(对应 L09 lecture 图示): root(0) -> A(1) -> Aa(2)
#                                  root(0) -> B(3) -> Ba(4)
parent_idx = [-1, 0, 1, 0, 3]
mask = build_tree_mask(parent_idx)
assert mask[2].tolist() == [True, True, True, False, False]   # Aa 只看到 root,A,Aa
assert mask[4].tolist() == [True, False, False, True, True]   # Ba 只看到 root,B,Ba

torch.manual_seed(11)
n, d = 5, 8
q, k, v = torch.randn(n, d), torch.randn(n, d), torch.randn(n, d)
out_tree = tree_attention_torch(q, k, v, parent_idx)

# 独立验证：手动只对 Aa 的祖先子集 [root,A,Aa] 做标准 attention，应与 tree_attention_torch 一致
scale = 1.0 / math.sqrt(d)
sub_idx = [0, 1, 2]
scores_manual = (q[2:3] @ k[sub_idx].T) * scale
attn_manual = torch.softmax(scores_manual, dim=-1)
out_manual = (attn_manual @ v[sub_idx]).squeeze(0)
assert torch.allclose(out_tree[2], out_manual, atol=1e-6)
```

**实测(`.venv` 真跑):** 5 节点树(`root→A→Aa`,`root→B→Ba`)构造出的 mask,`Aa`(索引 2)这一行精确是 `[True,True,True,False,False]`(看得到 root/A/自己,看不到 B 分支),`Ba`(索引 4)是 `[True,False,False,True,True]`——两条分支的 mask 互不重叠(除了共享的 root)。独立手写"只对 Aa 的祖先子集做标准 attention"重新算一遍,和 `tree_attention_torch` 给出的第 2 行输出逐元素对齐,最大误差 `1e-6` 量级(浮点误差范围内,视为精确一致)。

**面试怎么问 + 追问链:**
- **Q:** "为什么验证一整棵投机解码候选树,不需要对每条路径单独做一次 attention?" —— 期望说出"把树摊平成一个 flat token 序列,构造一张 [N,N] 的 mask,让每个位置只能 attend 到自己在树里的祖先链,一次标准 attention 就能算出所有路径的结果,公共祖先部分的 K/V 天然只被算一次,不会因为多条路径共享而重复计算"。
- **追问 1(考察是否理解 mask 的精确语义):** "`build_tree_mask` 里 `mask[i,j]=True` 具体是什么含义?对角线一定是 True 吗?" —— 期望说出"`j` 是 `i` 在树里的祖先节点(包括 `i` 自己)——每个节点走 `parent_idx` 链条一定会经过自己,所以对角线恒为 True;两个处于不同分支、没有祖先关系的节点之间,mask 双向都是 False"。
- **追问 2:** "如果两条候选路径共享前 3 个 token、只在第 4 个 token 分叉,tree attention 相比给这两条路径分别做 attention,具体省了多少计算?" —— 期望能定性回答:公共的前 3 个 token 部分,不管是 K/V 的存储还是 attention 的 QK 内积计算都只做 1 次(而不是 2 次),只有从分叉点往后的部分才是"各算各的";具体节省比例取决于共享前缀的长度占整条路径长度的比例,共享得越多,tree attention 相对独立验证的优势越大。
- **追问 3:** "这个知识点用的是纯 torch 实现,真实生产系统的 tree attention kernel 和这个有什么本质区别?" —— 期望说出"数学上表达的关系是一样的(用 mask 编码祖先可见性),区别在于工程实现——生产级 kernel(比如 FlashAttention v2/v3 的 tree mask 支持)会用 01 号文件讨论过的那些手段(分块处理、避免物化整个 `[N,N]` score 矩阵、SRAM 复用)把这个计算做得更省显存带宽,这里的 `torch.softmax`+`masked_fill` 实现是为了讲清楚数学关系,不追求性能"。

**常见坑:** 把 tree attention 的 mask 和 01 号文件知识点 3 讨论的"causal mask"(只能看到自己之前的 token)混为一谈——causal mask 是一个固定的下三角结构(每个位置能看到编号更小的全部位置),tree mask 是完全由 `parent_idx` 这个动态树结构决定的、因树而异的稀疏结构(祖先链之外的位置,即便编号更小,也可能被屏蔽,比如例子里 `Ba` 看不到 `A`/`Aa` 尽管它们的索引更小)。另一个坑是以为"验证树"和"投机解码的 rejection sampling"是两件独立的事——tree attention 只负责"一次算出树上每个节点位置的 target 概率分布",算出来之后依然要按知识点 2 的 accept/reject 规则逐条路径判断接受到哪一步,两者是流水线上的两个不同环节,不能互相替代。

---

## 10. 评测方法论 + Capstone(`spec_eval.py` + `capstone_eagle3.py`,L11+L12)—— accept rate/MAU/sim_speedup 三把尺子,4 方法 × 5 任务合成对照

**是什么:**
```python
def compute_mau(m: SpecMetrics) -> float:
    return m.mau

def compute_accept_rate(m: SpecMetrics) -> float:
    return m.accept_rate

def sim_speedup(m: SpecMetrics, draft_cost_ratio: float = 0.1) -> float:
    """Simulated speedup vs baseline of 1 token / iter."""
    if m.n_iters == 0:
        return 1.0
    k_per_iter = m.n_drafted / m.n_iters
    iter_cost = 1 + draft_cost_ratio * k_per_iter
    return m.n_tokens_out / (m.n_iters * iter_cost)
```
(`spec_eval.py:8-29`)

```python
TASKS = {
    "code":  dict(entropy=0.30),
    "math":  dict(entropy=0.20),
    "json":  dict(entropy=0.20),
    "qa":    dict(entropy=0.55),
    "story": dict(entropy=0.80),
}
```
(`capstone_eagle3.py:36-42`)

**一句话:** `spec_eval.py` 把知识点 1 引入的 `accept_rate`/`mau` 两个基础指标,加上一个"考虑了 draft 开销"的 `sim_speedup` 模拟加速比(不是真实 wall-clock 测量,是按"每轮迭代花费 = 1(大模型验证)+ draft_cost_ratio × 每轮草稿数"这个简化成本模型算出来的相对值);`capstone_eagle3.py::TASKS` 用 5 个不同"熵"(entropy,越高越不确定/越有创造性)的合成任务,对 classic/medusa/eagle1/eagle2 四种方法做统一对照。

**底层机制/为什么这样设计:** 从最笨的想法讲起——前面 9 个知识点每一种方法都各自证明了"自己能工作、accept rate 会随参数变化",但读者仍然缺一张"把它们放在同一把尺子下比较"的表。`sim_speedup` 的成本模型是:如果不用投机解码,产出 `n_tokens_out` 个 token 需要 `n_tokens_out` 次大模型 forward,每次成本记为 `1`;用了投机解码后,总共花了 `n_iters` 轮,每轮成本是 `1`(一次大模型验证 forward,不管验证几个位置都近似算 1 次开销,呼应知识点 1"验证 k+1 个位置和验证 1 个位置几乎一样贵"这个前提)加上 `draft_cost_ratio × k_per_iter`(草稿开销,和草稿了多少个 token 成正比),`sim_speedup = n_tokens_out / (n_iters × iter_cost)` 就是"投机解码产出的 token 数"除以"投机解码花费的等效大模型 forward 次数"。`capstone_eagle3.py::make_target()` 用 `entropy` 这一个参数控制目标分布的"平坦程度"(entropy 越高,`softmax` 之前的 logits 越接近纯正弦扰动、越平坦;entropy 越低,越接近纯高斯基底、越尖锐),`code`/`math`/`json` 三个任务给低 entropy(0.2-0.3,模拟"结构化、确定性高"的生成场景),`qa`/`story` 给高 entropy(0.55/0.8,模拟"开放式、创造性"的生成场景)——这是一种用一个标量参数近似"任务难度/确定性"这个多维概念的简化建模,不是真的对代码/数学/问答/故事这些任务做了语义区分,纯粹是"熵越低,accept rate 应该越高"这条规律的参数化实现。

**AI 研究场景:** 本知识点的 capstone 结果重新验证了这个模块自己 `README.md` 里已经明确标注过的一个诚实结论:**在这份合成数据下,classic 反而看起来比 eagle1/eagle2 更强**(README 原话:"Synthetic 数据下分布噪声小 ⇒ classic 看起来反胜;真模型场景需 Qwen-1.5B+0.5B 跑")。这不是这份代码的 bug,而是提醒一个重要的方法论边界:这些方法在真实模型上的排位(Medusa > Classic,EAGLE > Medusa,EAGLE-2 > EAGLE-1)依赖的是"同一个真实 target 模型,不同草稿机制谁更贴近它"这个关系,而这份 capstone 里 classic 的 draft 是用"目标 entropy+0.25"直接构造出来的一个**独立**分布(见 `eval_method` 里 `draft_fn = make_target(TASKS[task]["entropy"] + 0.25, ...)`),这个构造方式恰好让 classic 的 draft 和 target 天然比较接近,而 medusa/eagle 用的是"对 target 加噪声"这种建模方式,在这份简化 mock 里 draft 与 target 的接近程度反而不占优势——这再次印证了知识点 4 强调过的道理:合成 mock 能验证的是"每种方法自己的机制跑得通、参数变化方向正确",不能直接拿它的排位去对应真实模型上的排位。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/speculative-decoding/src")
from common import SpecMetrics
from spec_eval import compute_mau, compute_accept_rate, sim_speedup
from capstone_eagle3 import eval_method, to_md

m = SpecMetrics(n_iters=8, n_drafted=32, n_accepted=24, n_tokens_out=32)
assert compute_accept_rate(m) == 0.75
assert compute_mau(m) == 3.0
su = sim_speedup(m, draft_cost_ratio=0.1)
assert abs(su - 32 / (8 * (1 + 0.1 * 4))) < 1e-9

# 独立跑一遍 capstone 的一个子集（不同于源文件 __main__ 用的 n_tokens=40、seed=0 组合）
rows = [eval_method(method, task, n_tokens=40, seed=1)
        for method in ("classic", "medusa", "eagle1", "eagle2")
        for task in ("code", "story")]
table = to_md(rows)
assert "classic" in table and "eagle2" in table

by_key = {(r["method"], r["task"]): r for r in rows}
# 复现 README 已经如实标注过的结论：合成数据下 classic 的 accept rate 不低于 eagle 系列
assert by_key[("classic", "code")]["accept_rate"] >= by_key[("eagle2", "code")]["accept_rate"]
```

**实测(`.venv` 真跑):** `SpecMetrics(n_iters=8, n_drafted=32, n_accepted=24, n_tokens_out=32)` 精确得 `accept_rate=0.75`,`mau=3.0`,`sim_speedup(ratio=0.1)≈2.857`,和手算公式一致。换用 `n_tokens=40, seed=1`(不同于源文件自带 `__main__` 用的 `n_tokens=40, seed=0`)独立跑一遍 4 方法 × {code, story} 子集,结果:`classic/code` accept `0.889`、`classic/story` accept `1.0`,而 `medusa/code` 只有 `0.185`、`eagle1/code` 是 `0.238`、`eagle2/code` 是 `0.244`——classic 的接受率明显高于其余三种方法,和该模块 `README.md` 自己标注的"合成数据下 classic 看起来反胜"完全一致,证实这不是源文件自带 `__main__` 那一次运行的偶然结果,换种子后同样的定性结论依然成立。

**面试怎么问 + 追问链:**
- **Q:** "怎么系统评价一个投机解码方案的效果好不好?" —— 期望至少说出 accept rate(单 token 接受率)、MAU(每轮迭代平均净赚多少 token)、wall-clock speedup(真实端到端加速比)这三个维度,并能说明为什么只看 accept rate 不够(它没有把 draft 本身的开销算进去)。
- **追问 1(核心陷阱,考察诚实性):** "这个仓库的 capstone 显示 classic 投机解码比 EAGLE 系列表现更好,是不是说明 classic 是最优方案?" —— 期望明确说"不能这么下结论——这是合成数据下的一个已知现象(模块自己的 README 就标注了这一点),这份 mock 里 classic 的 draft 分布是用'target entropy+0.25'直接构造出来的、天然比较贴近 target,而 medusa/eagle 用的加噪声建模方式在这份简化实现下没有体现出真实模型上的优势;真实结论(EAGLE > Medusa > Classic)需要在真实模型上跑真实 benchmark 才能验证",这道题专门筛"看到一个数字就直接下结论、不去检查数字是怎么算出来的"的候选人。
- **追问 2:** "`sim_speedup` 和真实测出来的 wall-clock speedup 会有什么系统性差异?" —— 期望说出"`sim_speedup` 是一个简化的成本模型(每轮验证记 1、草稿按数量线性计费),没有考虑真实系统里的调度开销、批处理效应、显存带宽的实际利用率这些因素;知识点 11(评测方法论)lecture 提到的隐藏陷阱——短输出 warmup 占比高、batch=1 vs batch=16 表现差异巨大——这些都是 `sim_speedup` 这个简化公式没有建模、但真实测量会撞上的因素"。
- **追问 3:** "如果要把这份评测方法论用在真实模型上(不是合成数据),需要改哪些地方?" —— 期望说出:`make_target`/`draft_fn` 这类"用参数直接构造分布"的函数需要换成真实模型的 forward 调用,`target_fn`/`draft_fn` 的接口(输入 token 序列、输出下一个 token 的概率分布)已经足够通用、不需要改;更本质的变化是"真实模型的输出分布是训练学出来的,不是可以用一两个标量参数(entropy)完全刻画的",合成数据能验证流程正确性,不能替代真实 benchmark。

**常见坑:** 直接把这份仓库 capstone 输出的具体数字(accept rate、MAU、sim_speedup)当成"某个方法的真实性能指标"引用到别处(比如简历、论文)——这些数字全部来自合成、参数化的目标分布,不是任何真实模型的测量结果,只能用来说明"评测框架本身跑得通、能正确反映各方法在给定参数下的相对表现",不能替代在 Qwen/Llama 等真实模型上用 MT-bench/HumanEval/GSM8K(L11 lecture 提到的真实评测基准)跑出来的数字。另一个坑是把 5 个任务的 `entropy` 参数当成"这个仓库真的分析过 code/math/qa/story 这几类真实文本的统计特性"——`entropy` 只是一个人为设定的参数,用来模拟"确定性任务 vs 创造性任务"这个大方向的差异,不是从真实语料测出来的经验值。
