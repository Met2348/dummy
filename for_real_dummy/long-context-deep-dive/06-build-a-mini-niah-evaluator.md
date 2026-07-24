# 06 · 手把手实战:从零搭一个迷你 NIAH 评测器

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是知识点,不计入"17 个知识点"的统计——和 [05 类](05-advanced-interview-depth.md)一样,是"不计入统计的追加内容",但风格完全不同:05 号文件里,你是**旁观者**,跟着面试官和候选人的多级追问链把一条推理链条看一遍;这一篇里,你是**动手的人**——从一个空文件开始,一步步敲代码,每写一段就跑一次、看到真实效果,最后独立搭出一个真实能跑的迷你评测工具。这个"教程体"格式最早在 [dsa-deep-dive/21-build-a-mini-search-engine.md](../dsa-deep-dive/21-build-a-mini-search-engine.md) 试点验证,这是它在本系列的第一次落地。

**先把规矩立在最前面(全文会反复强调,这里先说一遍):** 这篇教程从头到尾不调用任何真实大模型、不联网,纯 CPU、秒级跑完。更重要的是——阶段 3、4 会尝试复现"Lost in the Middle"那条 U 型准确率曲线,但复现的方式是**在判分函数里手写一个和位置相关的衰减权重**,不是真的把一堆 needle 测试接到某个真实模型上跑出来的。[03 类知识点 3](03-long-context-eval.md) 已经立过同样的规矩("本仓库没有任何代码真正跑模型去验证这条 U 型曲线本身"),这篇教程如实延续这条规矩,不会让读者误以为下面看到的曲线是"又独立验证了一次"这个学术结论——具体在哪一步引入了这个人为设计,阶段 3 会现场拆开讲清楚,不会一笔带过。

---

## 为什么是"NIAH 评测器"

不是要发明新知识点,是把 [03 类](03-long-context-eval.md)已经讲过的三件事串成一个真正能跑起来的小工具:"怎么造测试用例"(NIAH 出题逻辑)、"怎么判分"(子串匹配)、"为什么要覆盖不同位置"(Lost in the Middle)。03 类的三个知识点是分开讲的、各自配一段"可运行例子";这一篇要做的是把它们攒成一条连贯的流程,自己从零手写一遍——**不导入**仓库里已经写好的 `learning/long-context/src/niah_eval.py`/`ruler_eval.py`。所以下面所有函数名都和那两个文件不一样(`make_document` 不是 `make_haystack`,`insert_needle` 不是 `make_niah_query`……):这是特意的,提醒自己这是重新手写的迷你版本,不是同一份代码,实现细节上可能有出入,不能拿这篇教程的行为反过去当成仓库源码的依据——真要确认 `niah_eval.py`/`ruler_eval.py` 自己的行为,请回 [03 类](03-long-context-eval.md)。

| 阶段 | 要让程序多会一件事 | 建立在哪个已有知识点之上 |
|------|------|------|
| 阶段 1 | 合成一批不同长度的"文档",长度精确可控 | [03 类知识点 1](03-long-context-eval.md) `make_haystack` 用重复填充文本控制长度的设计思路 |
| 阶段 2 | 在文档的任意相对位置插入一句独特可识别的 needle,并能读出它真实落在哪 | [03 类知识点 1](03-long-context-eval.md) `depth_pct` 百分比设计——为什么必须是相对位置,不能是绝对字符数 |
| 阶段 3 | 写一个判断"needle 有没有被找到"的函数——先看最朴素的版本为什么测不出位置差异,再让它对位置真的敏感起来 | [03 类知识点 1](03-long-context-eval.md) `check_answer` 子串匹配的设计思路 + [03 类知识点 3](03-long-context-eval.md) Lost in the Middle 的归因 |
| 阶段 4 | 批量跑一个"长度 × 位置"网格、统计命中率,组装成一个完整能跑的 `MiniNiahEvaluator` | [03 类知识点 1/2](03-long-context-eval.md) `niah_grid`/`ruler_grid` 批量出题的网格模式 |

每个阶段的代码都能独立运行(本文件用本目录下的 `_verify_md.py` 校验——这份脚本从 `dsa-deep-dive` 系列复制而来,校验逻辑和具体系列无关:把每个 ` ```python ` 代码块单独拎出来起一个新的 Python 子进程执行,块与块之间**不共享任何变量**,所以后面阶段用到前面阶段写过的函数时,会重新贴一遍定义,不是偷懒复制,是这套校验机制要求的)。

---

## 阶段 1:合成不同长度的文档——为什么不需要真实语料

NIAH 评测第一步要有"干草堆"。真实场景里研究者常常拿维基百科文章、长篇博客这类真实语料填充;但评测框架本身要验证的是"长度和位置怎么精确控制、判分逻辑对不对",这些和干草堆里具体写的是什么内容无关。用一句没有信息量的话反复拼接,长度反而更容易精确控制到字符级别——[03 类知识点 1](03-long-context-eval.md) 讲 `make_haystack` 时也是同一个思路:"造一段指定长度的填充文本,没有任何信息量,纯粹用来撑长度"。

```python
def make_document(target_length, filler=None):
    # 用一句没有信息量的填充句子反复拼接，长度够了再精确裁到 target_length。
    # 这就是"干草堆"本身——故意不含任何值得被找到的信息。
    if filler is None:
        filler = "the quick brown fox jumps over the lazy dog and keeps running through the field. "
    reps = target_length // len(filler) + 2   # 多拼几遍，保证裁剪前肯定够长
    text = (filler * reps)[:target_length]
    return text

for length in [50, 200, 1000, 5000]:
    doc = make_document(length)
    print(f"length={length:>5} len(doc)={len(doc):>5} doc[:50]={doc[:50]!r}")
    assert len(doc) == length   # 长度必须精确可控，不是"大概这么长"

# filler 换成别的句子，长度控制逻辑完全不变
doc_custom = make_document(300, filler="alpha beta gamma delta ")
assert len(doc_custom) == 300
print(f"custom filler ok, len={len(doc_custom)}")
print("stage1 ok")
```

**实测输出(`.venv` 里真跑的):**
```
length=   50 len(doc)=   50 doc[:50]='the quick brown fox jumps over the lazy dog and ke'
length=  200 len(doc)=  200 doc[:50]='the quick brown fox jumps over the lazy dog and ke'
length= 1000 len(doc)= 1000 doc[:50]='the quick brown fox jumps over the lazy dog and ke'
length= 5000 len(doc)= 5000 doc[:50]='the quick brown fox jumps over the lazy dog and ke'
custom filler ok, len=300
stage1 ok
```

`target_length // len(filler) + 2` 里的 `+2` 只是留出富余量——不需要精确算出"恰好拼够"要重复几次,反正最后会用切片 `[:target_length]` 硬裁到精确长度,多拼几遍完全无害,这比精确计算重复次数简单可靠得多。

---

## 阶段 2:在任意相对位置插入一根 needle

有了长度可控的干草堆,下一步是把一句独特、可识别的话("needle")插进文档的指定相对位置。**相对位置**这个词是关键——[03 类知识点 1](03-long-context-eval.md) 已经讲过为什么 `depth_pct` 必须是百分比而不是绝对字符数:要让"深度 50%"在任何长度的文档里都严格对应"正中间",不同长度的测试点才能落在同一个坐标系里比较。这里直接沿用这个设计,自己重新写一遍插入逻辑。

```python
def make_document(target_length, filler=None):
    if filler is None:
        filler = "the quick brown fox jumps over the lazy dog and keeps running through the field. "
    reps = target_length // len(filler) + 2
    text = (filler * reps)[:target_length]
    return text

def insert_needle(target_length, depth_pct, needle_code, filler=None):
    # 先造好干草堆，再按 depth_pct（0~100 的相对百分比，不是绝对字符数）
    # 换算出插入的字符偏移，把 needle 塞进去。
    haystack = make_document(target_length, filler=filler)
    needle = f" By the way, the secret code is {needle_code}. "
    insert_at = int(len(haystack) * depth_pct / 100)
    return haystack[:insert_at] + needle + haystack[insert_at:]

for depth in [0, 25, 50, 75, 100]:
    doc = insert_needle(2000, depth, needle_code="4242")
    pos = doc.find("4242")
    rel = pos / len(doc)
    print(f"depth_pct={depth:>3} -> real relative pos {rel:.3f}  doc_len={len(doc)}")

# 精确抽查三个边界/中心点，不能只靠打印肉眼看
doc0 = insert_needle(2000, 0, "1111")
doc50 = insert_needle(2000, 50, "2222")
doc100 = insert_needle(2000, 100, "3333")
assert doc0.find("1111") / len(doc0) < 0.05            # depth 0%   -> 应该贴近开头
assert 0.45 < doc50.find("2222") / len(doc50) < 0.55    # depth 50%  -> 应该贴近正中间
assert doc100.find("3333") / len(doc100) > 0.95         # depth 100% -> 应该贴近结尾
print("stage2 ok")
```

**实测输出:**
```
depth_pct=  0 -> real relative pos 0.016  doc_len=2038
depth_pct= 25 -> real relative pos 0.261  doc_len=2038
depth_pct= 50 -> real relative pos 0.506  doc_len=2038
depth_pct= 75 -> real relative pos 0.752  doc_len=2038
depth_pct=100 -> real relative pos 0.997  doc_len=2038
stage2 ok
```

和 [03 类知识点 1](03-long-context-eval.md) 实测过的现象一样:`depth_pct` 换算出的字符偏移和最终测出来的相对位置之间有几个百分点的量化误差(插入点按字符数取整,needle 本身又占用了一部分长度,把后半段的占比拉低了一点)——这不是 bug,是插入逻辑本身决定的效应,03 类已经如实记录过一次,这里自己重新实现依然复现了同一个效应,说明这不是那份源码独有的巧合。

---

## 阶段 3:模拟"能不能找到 needle"——先看最朴素的版本为什么不够

### 3a. 朴素判分:子串匹配为什么测不出位置差异

最直接的判分方式:看 needle 的编码字符串在不在整段文档里。[03 类知识点 1](03-long-context-eval.md) 的 `check_answer` 就是这个思路。

```python
def make_document(target_length, filler=None):
    if filler is None:
        filler = "the quick brown fox jumps over the lazy dog and keeps running through the field. "
    reps = target_length // len(filler) + 2
    return (filler * reps)[:target_length]

def insert_needle(target_length, depth_pct, needle_code, filler=None):
    haystack = make_document(target_length, filler=filler)
    needle = f" By the way, the secret code is {needle_code}. "
    insert_at = int(len(haystack) * depth_pct / 100)
    return haystack[:insert_at] + needle + haystack[insert_at:]

def naive_hit(document, needle_code):
    # 最朴素的判分方式：needle 的编码字符串在不在文档里，纯子串匹配。
    return needle_code in document

depths = [0, 25, 50, 75, 100]
results = {}
for depth in depths:
    doc = insert_needle(3000, depth, needle_code="8675")
    results[depth] = naive_hit(doc, "8675")

for depth, hit in results.items():
    print(f"depth_pct={depth:>3} naive_hit={hit}")

hit_rate = sum(results.values()) / len(results)
print(f"naive hit_rate across all depths = {hit_rate:.3f}")
# 因为 needle 是我们自己插进去的，子串匹配在任何位置都一定能找到——
# 命中率恒为 1.0，和插入位置完全无关。
assert hit_rate == 1.0
assert all(results.values())
print("stage3a ok: naive substring match cannot see position at all")
```

**实测输出:**
```
depth_pct=  0 naive_hit=True
depth_pct= 25 naive_hit=True
depth_pct= 50 naive_hit=True
depth_pct= 75 naive_hit=True
depth_pct=100 naive_hit=True
naive hit_rate across all depths = 1.000
stage3a ok: naive substring match cannot see position at all
```

这个结果比想象中更能说明问题:`hit_rate` 恒为 1.0,和 `depth_pct` 完全没关系——不是巧合,是必然。needle 是我们自己插进去的,Python 的 `in` 操作符看到的是整段文本(干草堆再长也无所谓),只要字符串真的在里面就一定能找到。真实模型会漏掉中间位置的信息,根源在于它的"注意力"是有限的、要在有限的计算量里给一大堆 token 分配权重;而这里的朴素判分函数根本没有任何"容量限制"的概念,等价于一个能同时看到全文任意角落、记忆力和算力都无限的"全知读者"——这种读者当然不会有 Lost in the Middle 问题。**要在一个不调用真实模型的模拟里看到位置效应,就必须显式地给判分函数加上某种"容量有限"的设定,不加就永远不会自己冒出来。**

### 3b. 引入人为设计的"注意力衰减"权重——诚实标注这是模拟出来的

下面这个检测器和 `naive_hit` 一样先用 `find()` 定位 needle,但找到之后不直接判"命中",而是看它落在文档的哪个相对位置,按一个人为设计的权重做一次随机抽样再决定命中与否。**这个权重函数是这篇教程自己手写死的,没有任何理论推导或真实测量支撑,只是照着 U 型曲线的形状(两端高、中间低)手选了几个数字**——这正是任务一开始就强调的:如果模拟检测器本身对位置不敏感,就故意在它里面加一个和位置相关的衰减权重来复现这个模式。

```python
import random

def make_document(target_length, filler=None):
    if filler is None:
        filler = "the quick brown fox jumps over the lazy dog and keeps running through the field. "
    reps = target_length // len(filler) + 2
    return (filler * reps)[:target_length]

def insert_needle(target_length, depth_pct, needle_code, filler=None):
    haystack = make_document(target_length, filler=filler)
    needle = f" By the way, the secret code is {needle_code}. "
    insert_at = int(len(haystack) * depth_pct / 100)
    return haystack[:insert_at] + needle + haystack[insert_at:]

def decay_weight(depth_pct, floor=0.35):
    # 人为设计的"注意力衰减"权重：越靠近文档两端权重越接近 1.0（几乎必中），
    # 越靠近正中间权重越接近 floor（明显更容易漏）。floor=0.35 这个数字
    # 没有任何理论依据，只是手选出来让 U 型足够明显，换成别的值同样成立。
    dist_from_edge = min(depth_pct, 100 - depth_pct) / 50.0  # 0=贴边, 1=正中间
    return 1.0 - (1.0 - floor) * dist_from_edge

def simulated_hit(document, needle_code, seed):
    # 和 naive_hit 一样先用 find() 定位 needle；区别在于找到之后不直接判"命中"，
    # 而是看它落在文档的哪个相对位置，按 decay_weight 算出的权重做一次随机抽样。
    pos = document.find(needle_code)
    if pos == -1:
        return False   # 真的没找到，这种情况权重再高也不可能命中
    real_depth_pct = 100.0 * pos / len(document)
    weight = decay_weight(real_depth_pct)
    rng = random.Random(seed)   # 用固定 seed 保证同样的输入永远得到同样的结果
    return rng.random() < weight

# 完全没插入 needle 的文档：不管权重多高，find() 找不到就一定是 miss
doc_no_needle = make_document(500)
assert simulated_hit(doc_no_needle, "0000", seed=1) is False

n_samples = 300
depths = [0, 25, 50, 75, 100]
rates = {}
for depth in depths:
    hits = 0
    for i in range(n_samples):
        doc = insert_needle(2500, depth, needle_code="6543")
        if simulated_hit(doc, "6543", seed=depth * 10000 + i):
            hits += 1
    rates[depth] = hits / n_samples
    print(f"depth_pct={depth:>3} simulated_hit_rate(n={n_samples})={rates[depth]:.3f}")

# 核心断言：中间位置的命中率明显低于两端——这正是我们在 decay_weight 里
# 手写死的规则的直接后果，不是任何"涌现"出来的现象。
assert rates[50] < rates[0]
assert rates[50] < rates[100]
assert rates[50] < rates[25]
assert rates[50] < rates[75]
print("stage3b ok: hit-rate now depends on where the needle actually landed")
```

**实测输出:**
```
depth_pct=  0 simulated_hit_rate(n=300)=0.997
depth_pct= 25 simulated_hit_rate(n=300)=0.663
depth_pct= 50 simulated_hit_rate(n=300)=0.390
depth_pct= 75 simulated_hit_rate(n=300)=0.643
depth_pct=100 simulated_hit_rate(n=300)=1.000
stage3b ok: hit-rate now depends on where the needle actually landed
```

`seed` 每次调用都固定传入,是为了让这个"模拟"本身可复现——同样的 `depth_pct`、同样的 `i`,任何时候重跑都应该得到完全一样的结果,这是"模拟"和"真的调用一个有随机性的模型"的本质区别之一:模型的输出不受你控制,这里的随机数完全受控。

---

## 阶段 4:组装成一个完整的 `MiniNiahEvaluator`,批量测出 U 型形状

把前三阶段拼进一个类,加上一个"长度 × 深度"批量网格(呼应 [03 类知识点 1/2](03-long-context-eval.md) `niah_grid`/`ruler_grid` 的批量出题模式),跑一次完整的端到端 demo,顺便统计出命中率。

```python
import random

class MiniNiahEvaluator:
    """从零手写的迷你 NIAH 评测器：只负责"出题 + 模拟判分"，
    从头到尾不调用任何真实大模型、不联网，全部逻辑都是确定性可复现的。"""

    def __init__(self, filler=None, floor=0.35, long_doc_threshold=4000, long_doc_penalty=0.85):
        self.filler = filler or "the quick brown fox jumps over the lazy dog and keeps running through the field. "
        self.floor = floor
        self.long_doc_threshold = long_doc_threshold
        self.long_doc_penalty = long_doc_penalty

    def make_document(self, target_length):
        reps = target_length // len(self.filler) + 2
        return (self.filler * reps)[:target_length]

    def insert_needle(self, target_length, depth_pct, needle_code):
        haystack = self.make_document(target_length)
        needle = f" By the way, the secret code is {needle_code}. "
        insert_at = int(len(haystack) * depth_pct / 100)
        return haystack[:insert_at] + needle + haystack[insert_at:]

    def _decay_weight(self, depth_pct, length):
        dist_from_edge = min(depth_pct, 100 - depth_pct) / 50.0
        weight = 1.0 - (1.0 - self.floor) * dist_from_edge
        if length > self.long_doc_threshold:
            # 额外惩罚：文档越长，草堆本身越大，模拟"更难搜"这件事。
            # 这同样是手写死的常数，不是从任何真实测量里拟合出来的。
            weight *= self.long_doc_penalty
        return weight

    def simulated_hit(self, document, needle_code, seed):
        pos = document.find(needle_code)
        if pos == -1:
            return False
        real_depth_pct = 100.0 * pos / len(document)
        weight = self._decay_weight(real_depth_pct, len(document))
        rng = random.Random(seed)
        return rng.random() < weight

    def run_grid(self, lengths, depths, n_samples=100):
        # 呼应 03 类知识点 1/2 的 niah_grid/ruler_grid：批量出题，
        # 每个 (长度, 深度) 组合跑 n_samples 次，统计命中率。
        grid = {}
        for length in lengths:
            for depth in depths:
                hits = 0
                for i in range(n_samples):
                    doc = self.insert_needle(length, depth, needle_code="NEEDLE")
                    hits += self.simulated_hit(doc, "NEEDLE", seed=length * 100000 + depth * 100 + i)
                grid[(length, depth)] = hits / n_samples
        return grid

    def render_report(self, grid, lengths, depths):
        lines = ["      " + "".join(f"{d:>7}%" for d in depths)]
        for length in lengths:
            row = f"len={length:<5}"
            for depth in depths:
                row += f"{grid[(length, depth)]:>7.2f} "
            lines.append(row)
        return "\n".join(lines)


evaluator = MiniNiahEvaluator()

# 端到端冒烟测试：needle 真的落在我们要求的位置
doc = evaluator.insert_needle(1500, depth_pct=50, needle_code="7777")
assert "7777" in doc
rel_pos = doc.find("7777") / len(doc)
assert 0.45 < rel_pos < 0.55

# 批量跑一个"长度 x 深度"网格，打印一张迷你热力图
lengths = [1000, 6000]
depths = [0, 25, 50, 75, 100]
grid = evaluator.run_grid(lengths, depths, n_samples=250)
print(evaluator.render_report(grid, lengths, depths))

# 复现 Lost in the Middle 的 U 型形状：同一长度下，中间明显比两端更差
for length in lengths:
    assert grid[(length, 50)] < grid[(length, 0)]
    assert grid[(length, 50)] < grid[(length, 100)]
# 更长的文档在同样是中间位置时更难——这也是 long_doc_penalty 写死的直接结果
assert grid[(6000, 50)] < grid[(1000, 50)]

print(f"mid vs edge at len=1000: {grid[(1000, 50)]:.3f} vs {grid[(1000, 0)]:.3f}")
print(f"mid at len=1000 vs len=6000: {grid[(1000, 50)]:.3f} vs {grid[(6000, 50)]:.3f}")
print("MiniNiahEvaluator end-to-end ok")
```

**实测输出:**
```
      0%     25%     50%     75%    100%
len=1000    0.96    0.60    0.36    0.69    0.99
len=6000    0.85    0.58    0.23    0.59    0.88
mid vs edge at len=1000: 0.360 vs 0.956
mid at len=1000 vs len=6000: 0.360 vs 0.232
MiniNiahEvaluator end-to-end ok
```

两端高、中间低——这张迷你热力图和 [03 类知识点 3](03-long-context-eval.md) 引用的 Lost in the Middle U 型曲线形状一致。**但这里必须再诚实说一遍:这条 U 型是 `_decay_weight` 里手写的衰减公式的直接产物,不是任何真实模型的行为。**

还有一个值得多看一眼的不完美细节,不回避地记录下来:`_decay_weight` 按设计是关于 50% 对称的(`dist_from_edge` 在深度 25% 和 75% 处应该相等),但实测 `len=1000` 这一行,25% 列是 0.60、75% 列是 0.69,并不完全对称。这不是 bug——一是阶段 2 已经如实记录过 `depth_pct` 换算成字符偏移本身存在量化误差,25% 和 75% 实际插入的真实相对位置不是精确镜像的两个点;二是 `n_samples=250` 属于有限采样,存在真实的统计噪声,换一批 `seed` 数字上会再变。这点不对称正是"这是一次真实运行的输出,不是编出来凑整齐的数字"的证据,留着比抹平更诚实。

---

## 可以怎么继续扩展(只指方向,不在本文实现)

- **接一个真实模型**:把 `simulated_hit` 换成真的调用一个模型的 `generate()` 拿回答、再用类似 `check_answer` 的逻辑判分——这才是真正验证 Lost in the Middle 是否在某个具体模型上成立,而不是停留在"设计一个会衰减的函数"。[03 类知识点 1](03-long-context-eval.md) 提到的 `eval_niah` 伪代码就是这个方向,本仓库同样没有实现、也没有跑过这一步。
- **滑窗相似度检测器**:现在阶段 3 的两个检测器要么整串精确匹配、要么在匹配到之后按位置加权,还没有试过"读一个固定大小的窗口、和 needle 算一个粗糙的相似度分数,分数超过阈值才算命中"这种更贴近"检索"直觉的实现方式——这个方向本文没有实现,只指出来。
- **RULER 风格的多 needle/多任务扩展**:仿照 [03 类知识点 2](03-long-context-eval.md) 的 `mk_niah`/`mv_niah`/`variable_tracking`,把这个评测器从"一根 needle 在不在"扩展成"同时插好几根、问其中指定的一根"或者"要求把所有 needle 的内容聚合出来"——现在的 `MiniNiahEvaluator` 只支持单 needle,不支持这些更接近真实长文档任务的形态。

这三个方向都不实现,是为了让这篇教程聚焦在"评测框架的结构怎么搭、模拟出来的效果怎么诚实呈现"这一件事上。

## 这篇教程展示的方法论

任何一条已完成的深挖系列,都可以用同样的模式产出"教程体"内容:挑几个关联的知识点 → 设计一个真实有用、读者一看就懂价值的小工具 → 分阶段增量实现,每一步都跑起来看到真实效果,而不是一次性甩出完整代码。这是这个格式第一次推广到 `dsa-deep-dive` 之外的系列,验证了它不依赖具体专题——[03 类](03-long-context-eval.md)讲的是评测方法论而不是数据结构,同样的"分阶段动手搭"节奏依然适用。

这篇教程还额外示范了一条更具体的纪律,值得单独写出来:**当手头没有真实模型、没有真实数据可用时,诚实标注"这是我设计的模拟,不是真实观测",比让读者误以为复现了一个学术结论重要得多。** [03 类知识点 3](03-long-context-eval.md) 已经在"可运行例子"里立过这个规矩(只演示 `depth_pct` 怎么控制位置,不暗示测过真实准确率);这篇教程更进一步——没有回避"复现 Lost in the Middle"这件事本身,还真的把它做成了一条 U 型曲线,但曲线的成因被完整摊开写清楚:阶段 3a 先证明朴素判分函数结构上不可能出现位置效应,阶段 3b 才引入人为设计的衰减权重,阶段 4 的最终数字里连"两侧本该对称却没有完全对称"这种不完美细节都保留了下来,不是挑好看的数字汇报。这和全系列"纯 CPU 模拟"的定位([00-roadmap.md](00-roadmap.md) 环境声明)是同一种纪律——没有真实模型可用的时候,把"模拟"这件事本身的边界讲清楚,比假装模拟等于真实更有价值。

---

*创建:2026-07-24*
