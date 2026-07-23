# 03 · 长上下文评测方法论深挖(Long-Context Evaluation Methodology)

> 总览见 [00-roadmap.md](00-roadmap.md)

这一批要解决的问题是:一个大模型宣传"支持 128k / 200k 上下文",这句话到底能不能信?面试官很喜欢在长上下文这个话题上追问"你怎么知道模型真的能用那么长",而不满足于"因为模型卡上写了"——这一批就是回答这个问题需要的方法论,也是本系列 01(RoPE 外推家族)、02(长上下文 attention 架构)那些"怎么让模型支持更长上下文"的技术,最终要拿去接受检验的那把尺子。

**本文定位:** `learning/long-context/src/niah_eval.py`、`learning/long-context/src/ruler_eval.py` 分别是知识点 1(NIAH)、知识点 2(RULER)的源码,`learning/long-context/lectures/10-needle-haystack.md` 是两者的配套 lecture;知识点 3(Lost in the Middle)完全来自 `learning/long-context/lectures/12-long-context-pitfalls.md`,**没有对应的可跑代码**——这一条会在小节内部反复标注边界,不会暗示本仓库跑出过这条曲线。和系列其它批次一样,每个知识点从"最笨的想法"讲起(比如先问"怎么证明模型真的读到了第 5 万个 token",再引出 NIAH 这个术语),不是直接甩黑话。

本文知识点 1、2 的所有可运行例子已在仓库根目录 `.venv`(Windows 原生,Python 3.13.9)下实际跑通验证。额外确认过一个容易被忽略的事实:`niah_eval.py`/`ruler_eval.py` 这两份源码零 ML 框架依赖——完整读过一遍,没有一行 `import torch`/`import transformers`,只用了标准库 `random`/`typing`,CPU 秒级跑完,不需要下载任何模型权重。知识点 3 没有可跑代码,严格标注为"lecture 引用的外部研究结论",不是本仓库的实验产出。

**本篇统一结构(与 00-roadmap.md 的模板一致):**
1. 签名/是什么
2. 一句话
3. **底层机制/为什么这样设计** —— 不停在"怎么用",讲到"为什么必须是这样"
4. AI 研究场景
5. 可运行例子 —— 带 `assert` 验证,真的在仓库 `.venv` 里跑过
6. **面试怎么问 + 追问链** —— 面试官大概率怎么问,追问会往哪个方向深挖
7. 常见坑

---

## 1. NIAH(Needle in a Haystack,`niah_eval.py`)—— 测试用例生成器 + 判分器,不是模型

**是什么:**
```python
def make_haystack(target_length: int, base: str = None) -> str: ...
def make_niah_query(target_length: int, depth_pct: float,
                    needle_code: str | None = None) -> tuple[str, str]: ...
def check_answer(answer: str, expected_code: str) -> bool: ...
def niah_grid(context_lengths: Iterable[int], depths: Iterable[float],
              n_samples: int = 3) -> dict: ...
```
`niah_eval.py` 一共就四个函数,翻成人话:
- `make_haystack`:造一段指定长度的"填充文本"(干草堆),没有任何信息量,纯粹用来撑长度。
- `make_niah_query`:造"一道题"——把一句藏着秘密编码的"针"塞进草堆里第 `depth_pct`% 的位置,拼成一段问答 prompt,连同"标准答案"一起返回。
- `check_answer`:判分——看模型的回答文本里有没有原样出现标准答案。
- `niah_grid`:批量出题——给一组长度 × 一组深度 × 每种组合出几份样本,产出整张测试网格。

**一句话:** NIAH 测的是"一根针(一句藏着特定信息的话)埋进一堆干草(很长的无关文本)里的不同深度,模型能不能精确地把针挑出来"——但 `niah_eval.py` 这份代码本身只管"埋针"和"判分",**不包含任何一行调用模型的代码**,它是评测流程里"出题 + 判卷"的那一半,不是"考试"本身。

**底层机制/为什么这样设计:**

先说一个容易被忽略的事实:这个文件从头到尾没有 `import torch`、没有 `import transformers`,结构上就不可能跑模型。这不是偷懒,是有意为之——把"出题"和"考试"拆成两个独立的关注点,出题逻辑可以脱离任何具体模型被单独测试(仓库里确实有 `learning/long-context/src/tests/test_niah_pass_rate.py` 专门测 `make_niah_query`/`niah_grid` 自己对不对,和"某个模型考得好不好"完全无关)。

再看 `depth_pct` 为什么是"百分比"而不是"第几个字符"。NIAH 的经典产出是一张热力图:横轴是 context length(4k/8k/32k/128k……),纵轴是 needle depth(0%~100%)。如果深度用绝对字符数表示,"第 500 个字符"在一个 1000 字符的草堆里是正中间,但在一个 10 万字符的草堆里几乎是开头——两次测试根本不可比。用百分比才能让"深度 50%"在任何长度下都严格对应"正中间",这也是下面知识点 3(Lost in the Middle)那条 U 型曲线能被画出来的坐标系前提。

示意图(单次测试里针插在 haystack 的什么位置,以及批量出题之后经典的"长度 × 深度"热力图长什么样):

```
单次测试:一根针插在整段 haystack 的第 depth_pct% 处
┌──────────────────────────────────────────────────────────────┐
│ ...无关填充文本... [针:"the secret code is 4242"] ...无关填充文本... │
└──────────────────────────────────────────────────────────────┘
0%                       depth_pct(针的相对位置)                       100%

depth_pct=0%   -> 针靠近开头:  [针]........................................
depth_pct=50%  -> 针在正中间:  .....................[针].....................
depth_pct=100% -> 针靠近结尾:  ........................................ [针]

批量出题(niah_grid)产出的经典热力图:每格是一次"长度×深度"组合的测试通过率
              depth=0%   25%    50%    75%   100%
length=4k       ✓        ✓      ✓      ✓      ✓
length=32k      ✓        ✓      ✗      ✓      ✓
length=128k     ✓        ✗      ✗      ✗      ✓
                            ↑
              横着看中间这一片偏"✗"的区域,就是知识点3"Lost in the Middle" U型曲线的由来
```

针的具体句子还会从 `NEEDLE_TEMPLATES` 三种模板里随机挑一个("The secret password is …"/"Janet's lucky number is …"/"The hidden code is …"),不是永远同一句话,这样能避免模型单纯靠记住某个固定句式的表面模式蒙对。

再看 `check_answer` 为什么用最朴素的子串匹配 `expected_code in answer`,而不是要求精确相等。真实场景里模型的回答是自由文本,比如"The secret code mentioned in the text is 4242, hope that helps!"——如果要求 `answer == expected_code` 精确相等,几乎所有正确回答都会被误判为错。子串匹配用"宽松"换取"鲁棒",代价是下面"常见坑"会讲到的假阳性风险。

**AI 研究场景:** NIAH(Kamradt 2024)是业界验证"这个模型宣传的 128k/200k 上下文窗口是不是营销数字"的标准手段——`learning/long-context/lectures/10-needle-haystack.md` 给了几个实测参考:Llama-3.1 8B 在 128k 上 NIAH 各测试点通过率 >95%,早期 Mistral 7B 在 32k 上只有 64%(典型的 lost-in-middle),Claude 3.5 在 200k 上接近 100%。`learning/long-context/lectures/12-long-context-pitfalls.md` 也直接点出"广告 128k window,实际 NIAH 90%+ 只有 32k,高质量 retrieval 只有 16k"这种落差。`learning/long-context/lectures/13-capstone.md` 里设想了一个更完整的闭环:
```python
def eval_niah(model, ctx_len, depth, n=5):
    correct = 0
    for _ in range(n):
        q, ans = make_niah_query(ctx_len, depth)
        pred = model.generate(q, max_new_tokens=20)
        if ans in pred:
            correct += 1
    return correct / n
```
但这只是 lecture 里的示意伪代码——本仓库真正能跑的 `capstone_yarn_llama32.py` 并不导入 `niah_eval`,默认 dry-run 也只做到 LoRA 挂载为止,不会真的把 NIAH 接到模型上跑一遍(这一点会在 04 数据工程与 Capstone 那一批严格标注边界)。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/long-context/src")
from niah_eval import make_haystack, make_niah_query, check_answer, niah_grid

# 1. haystack 只是填充材料,长度精确可控,没有信息量
h = make_haystack(120)
assert len(h) == 120

# 2. 单次出题:针 + 深度 + 标准答案
query, expected = make_niah_query(target_length=300, depth_pct=50, needle_code="4242")
assert expected == "4242"
assert "4242" in query
pos = query.find("4242")
print(f"needle 在第 {pos} 字符 / 总长 {len(query)},相对深度 {pos/len(query):.2f}")
print(query[max(0, pos - 30):pos + 30])

# 3. depth_pct 精确控制插入位置,不是模糊的"大概在中间"
for d in [0, 25, 50, 75, 100]:
    q, _ = make_niah_query(target_length=2000, depth_pct=d, needle_code="9999")
    rel = q.find("9999") / len(q)
    print(f"depth_pct={d:>3} -> 实际相对位置 {rel:.3f}")

# 4. 判分:纯子串匹配,不理解语义
assert check_answer("The secret password is 4242, that's it.", "4242") is True
assert check_answer("I'm not sure.", "4242") is False

# 5. 批量出题:只生成,不判分也不跑模型
grid = niah_grid([200, 500], [25, 50, 75], n_samples=2)
assert len(grid) == 2 * 3 * 2 == 12
print(grid[0])
```

**实测输出(节选,`.venv` 里真跑的):**
```
needle 在第 170 字符 / 总长 390,相对深度 0.44
uick brown The hidden code is 4242.  fox jumps over the lazy
depth_pct=  0 -> 实际相对位置 0.012
depth_pct= 25 -> 实际相对位置 0.251
depth_pct= 50 -> 实际相对位置 0.489
depth_pct= 75 -> 实际相对位置 0.728
depth_pct=100 -> 实际相对位置 0.967
{'length': 200, 'depth': 25, 'query_len': 294, 'expected': '6947'}
```
两个细节值得多看一眼:一是 `depth_pct` 和针在最终文本里的相对位置几乎一一对应,只有几个百分点的量化误差(插入点按字符数取整,针本身也占了一些长度,拉低了后半段草堆占比);二是草堆被从"quick"这个词中间硬生生切开了("uick brown"是被削掉首字母 q 剩下的部分)——`insert_at` 是纯字符偏移,不对齐单词边界,`depth_pct` 精确控制的是**字符位置**,不是 token 或单词位置。

**面试怎么问 + 追问链:**
- **Q:** "给你一个新模型,你会怎么验证它的长上下文能力,而不是直接相信模型卡上写的 context length?" —— 期望候选人主动提出"NIAH 式"的方法:构造不同长度、不同深度的测试用例,统计通过率,而不是抽查几个例子拍脑袋下结论。
- **追问 1:** "`check_answer` 用子串匹配来判断对不对,这个设计有什么风险?" —— 期望答出"假阳性"(标准答案恰好是模型胡乱回答里的一段无关数字,或者模型复述了整段草堆导致答案恰好包含在内)和"假阴性"(模型把数字拼写成英文单词、加了千位分隔符,子串就匹配不上)两个方向,而不是只说"还行吧"。
- **追问 2(区分度很高):** "为什么深度要用百分比,而不是固定 token 数控制针的位置?" —— 期望答出"要让不同 context_length 的测试点落在同一个坐标系里可比,固定 token 数在短文本里可能是结尾、在长文本里可能是开头",这条答对说明真正理解了 NIAH 热力图的坐标轴设计,能直接连到知识点 3。

**常见坑:**
- 把 `niah_eval.py` 当成"评测跑分脚本"——它只出题、只判分,真正"跑模型拿回答"这一步不在这个文件里,自己接入真实模型时容易漏掉这一环,以为跑一下这个文件就能测出模型好坏。
- 盲信类型注解:`niah_grid` 的签名写的是 `-> dict`,但实测它返回的是一个 `list`(元素才是 dict)——对照组 `ruler_eval.py` 里 `ruler_grid` 的注解写的是 `-> list` 才是对的。Python 的类型注解不会在运行时被强制检查,连仓库自己的教学代码都能写错,拿到一个陌生函数第一反应应该是 `type(x)` 现场确认,而不是读注解就当真。
- `check_answer` 的子串匹配在 `expected_code` 很短(比如两三位数)时假阳性概率会明显升高——如果草堆本身或者模型答案里的"废话"部分偶然出现同样的数字,会被误判为"找到了"。

---

## 2. RULER(`ruler_eval.py`)—— NIAH 的 4 子任务扩展版

**是什么:**
```python
def s_niah(target_length: int, depth_pct: float) -> dict: ...
def mk_niah(target_length: int, n_keys: int = 4) -> dict: ...
def mv_niah(target_length: int, n_values: int = 3) -> dict: ...
def variable_tracking(target_length: int, n_hops: int = 3) -> dict: ...
def check_answer(predicted: str, expected) -> bool: ...
def ruler_grid(context_lengths: Iterable[int], n_per_task: int = 10) -> list: ...
```
四个出题函数,每个都返回 `{"prompt": ..., "answer": ..., "task": ...}` 统一格式:
- `s_niah`:Single-NIAH,和知识点 1 的 NIAH 是同一件事(1 根针、1 个 key、1 个 value),只是重新实现了一份。
- `mk_niah`:Multi-Key NIAH,一次埋好几根针(不同 key,比如 alpha/beta/gamma),问其中指定的那一个。
- `mv_niah`:Multi-Value NIAH,一个 key 对应好几个 value,问"把所有 value 列出来"——答案是个 list。
- `variable_tracking`:变量追踪,`x=5, y=x, z=y, ...` 这样一条赋值链,问链尾变量最终的值。
- `check_answer`:判分,但因为 `mv_niah` 的答案是 list,这里要同时兼容"字符串子串匹配"和"list 里每一项都要出现"两种逻辑。
- `ruler_grid`:批量出题,按长度循环,每次把 4 个子任务各出 `n_per_task` 道题。

**一句话:** RULER(NVIDIA 2024)是 NIAH 的"能力扩展版"——NIAH 只测"能不能找到一个事实",RULER 在此之外又加了"多个候选里精确定位一个"(mk-niah)、"把散落的多个答案聚合齐全"(mv-niah)、"顺着引用链多跳推理"(variable-tracking)三种更接近真实长文档任务的能力。

**底层机制/为什么这样设计:**

先看 `check_answer` 为什么要判断 `isinstance(expected, list)`——四个子任务里只有 `mv_niah` 的答案天生是个 list(其余三个都是 str,实测见下面可运行例子),统一的判分函数必须同时处理"这一个字符串在不在"和"这一串字符串是不是全都在"两种逻辑,这一行分支本身就是"RULER 比 NIAH 多了聚合类任务"这句话的直接代码证据。

再看 `mk_niah` 为什么要放好几根**同一模板、不同 key** 的针(`needles = [f"The {k}-code is {v}." for k, v in codes.items()]`,除了 key 名字外句式完全一样),而不是像 NIAH 那样只放一根。如果只放一根针,模型可能靠"文本里有没有出现'The X-code is 数字'这种句式"就蒙对答案,根本不需要精确读懂问题问的是哪个 key。放 4-5 根长得几乎一样、只有 key 名不同的针,模型必须真正定位到"alpha-code"对应的那一句,才能排除掉 beta/gamma/delta 的干扰——这是在测**精确检索**,不是"文本里有没有类似的句子"。

`variable_tracking` 更进一步:它不是"一句话一个事实",而是一条依赖链。关键细节是这条链被打乱了插入顺序,不是按 `x=5 → y=x → z=y` 的顺序原样写进文本的——实测一次 `n_hops=3` 的调用,文本里出现的顺序是"Let y = x. Let w = z. Let x = 637. Let z = y.",完全不按依赖顺序排列。这意味着模型不能靠"顺着读下来记住最后一句"这种捷径,必须建立一张"变量名 → 值"的符号表,不管赋值语句出现的先后顺序,才能正确推出 `w` 最终等于 637——这测的是多跳状态追踪,和"找到一句话"是两种不同的能力。

这三个子任务(`mk_niah`/`mv_niah`/`variable_tracking`)在插入位置上还有一个用代码读出来、并且实测验证过的共同特点:它们都不像 `s_niah`(以及知识点 1 的 `make_niah_query`)那样把 haystack **切开**塞入针(`haystack[:pos] + needle + haystack[pos:]`),而是把整段 haystack 当成 `pieces` 列表里的**一个元素**,再用 `pieces.insert(random.randint(0, len(pieces)), needle)` 把针"插队"进这个只有寥寥几个元素的列表,最后 `" ".join(pieces)`。因为 haystack 这一整块占了文本绝大多数字符,列表一共只有几个位置可插,针要么被排在 haystack 这个大块**之前**、要么**之后**——实测 20 次 `mk_niah` 调用,针的相对位置全部落在 0.01~0.03(接近开头)或 0.95~0.98(接近结尾),没有一次落在中间附近。也就是说,`mk_niah`/`mv_niah`/`variable_tracking` 这三个子任务**不支持像 NIAH 那样精确控制"深度百分比"**,只能做到"大致在前面"或"大致在后面"。眼尖的话还会发现 `mk_niah` 里有一行 `pos = random.randint(0, len(hay))` 算出来之后压根没被用到——它长得很像是想复刻 `make_niah_query` 那种按字符偏移插入的写法,但实际起作用的是紧接着独立的那次 `random.randint(0, len(pieces))`,`pos` 是个不影响任何结果的死变量。

顺带一提:`ruler_eval.py` 并没有复用 `niah_eval.py` 的 `make_haystack`,而是自己重新实现了一份 `_make_haystack`(填充文本也不一样,多拼了一段 Lorem ipsum)——两个文件看起来在做同一件事,但没有共享代码,细节不能假设完全一致。

**AI 研究场景:** RULER 更贴近真实长文档/RAG 场景——一次问答经常需要"从多个检索片段里挑对那一个"(对应 mk-niah)、"把分散在文档各处的多条信息汇总"(对应 mv-niah)、"跟踪一个实体/状态在长对话里几经改动后的最终结果"(对应 variable-tracking),单纯"能不能找到一个事实"(NIAH)不足以暴露这些更细粒度的失败模式。`learning/long-context/README.md` 里的原话是"NIAH ≠ 真用——RULER 测试更接近真实长 ctx 需求"。也要如实说清楚:`learning/long-context/lectures/10-needle-haystack.md` 提到真实 RULER 论文一共有 13 个子任务(还包括 multi-query NIAH、词频统计聚合等),这里的 `ruler_eval.py` 只实现了其中最有代表性的 4 种作为教学子集,不是完整复刻——这一点和 roadmap 00 对这一节的定位("NIAH 的扩展:single/multi-key/multi-value NIAH + variable tracking 4 个子任务")完全一致。

**可运行例子:**
```python
import sys
from collections import Counter
sys.path.insert(0, "learning/long-context/src")
from ruler_eval import s_niah, mk_niah, mv_niah, variable_tracking, check_answer, ruler_grid

r1 = s_niah(300, depth_pct=50)
r2 = mk_niah(300, n_keys=4)
r3 = mv_niah(300, n_values=3)
r4 = variable_tracking(300, n_hops=3)

# 答案形态差异:只有 mv-niah 是 list,其余都是 str
assert isinstance(r1["answer"], str)
assert isinstance(r2["answer"], str)
assert isinstance(r3["answer"], list)
assert isinstance(r4["answer"], str)

# check_answer 要同时处理两种形态
assert check_answer("617, 265, 499", ["617", "265", "499"]) is True
assert check_answer("617, 265", ["617", "265", "499"]) is False   # 少一个就算错
assert check_answer("magic number is 81956 right?", "81956") is True

# 批量出题:4 个子任务 × n_per_task 均匀分布
grid = ruler_grid([500, 1000], n_per_task=2)
assert len(grid) == 2 * 2 * 4 == 16
print(Counter(q["task"] for q in grid))

# mk_niah 的插入位置:20 次全部落在两端,没有一次在中间
positions = []
for _ in range(20):
    r = mk_niah(2000, n_keys=4)
    positions.append(r["prompt"].find(r["answer"]) / len(r["prompt"]))
assert all(p < 0.06 or p > 0.94 for p in positions)
```

**实测输出(节选):**
```
s-niah:            task=s-niah            answer type=str  value=81956
mk-niah:           task=mk-niah           answer type=str  value=1361
mv-niah:           task=mv-niah           answer type=list value=['220', '455', '747']
variable-tracking: task=variable-tracking answer type=str  value=637
variable-tracking prompt 结尾: '... Let y = x. Let w = z. Let x = 637. Let z = y.\n\nQuestion: What is the value of w?'
distribution: {'s-niah': 4, 'mk-niah': 4, 'mv-niah': 4, 'variable-tracking': 4}
mk_niah 20 次相对位置样本: ['0.96','0.97','0.96','0.01','0.02','0.95','0.03','0.98','0.02','0.98','0.03','0.02','0.97','0.03','0.96','0.97','0.01','0.01','0.03','0.97']
```
`variable-tracking` 这条实测样例正好能验证上面"底层机制"里的说法:文本里出现的顺序是 `y=x → w=z → x=637 → z=y`,顺着读第一句完全不知道 `x` 是多少,必须读完全部四句、在脑子里搭一张符号表,才能推出 `x=637 → y=637 → z=637 → w=637`,答案是 637。

**面试怎么问 + 追问链:**
- **Q:** "NIAH 和 RULER 具体差在哪?如果要给一个新模型选一套更有说服力的长上下文测试集,你会怎么选?" —— 期望答出 RULER 覆盖多种任务形态(单点检索/多候选辨析/聚合/多跳追踪),能发现"背得出一个事实但做不了聚合或追踪"这类只用 NIAH 测不出来的弱点。
- **追问 1:** "如果一个模型在 s-niah 上接近 100% 通过,但 mv-niah 只有 60%,可能是什么原因?" —— 期望候选人推理出"聚合比单点检索难":模型可能找到第一个匹配就停止搜索,或者后续几个分散的 value 在长文本里被"注意力"顾不上,这是检索能力和聚合能力的差异,不是同一件事。
- **追问 2(区分度很高):** "`variable_tracking` 这道题,能不能靠'顺着文本读下来,记住看到的最后一个赋值句'就蒙对?" —— 期望候选人指出赋值语句在文本里的顺序是打乱的(不按依赖顺序排列),必须建立符号表做多跳解析,单纯顺序阅读拿不到正确答案,这条答上说明真的读过生成逻辑,不是望文生义。

**常见坑:**
- 以为 `ruler_grid` 像 `niah_grid` 一样对深度做网格扫描——实际上 `ruler_grid` 只对 `context_lengths` 循环,`s_niah` 内部深度写死在 `depth_pct=50`,`mk_niah`/`mv_niah`/`variable_tracking` 根本没有暴露 `depth_pct` 参数,想测别的深度需要自己改代码。
- 把 `mk_niah`/`mv_niah`/`variable_tracking` 的插入位置想象成"均匀分布在整段文本里"——上面已经实测过,受 `pieces.insert` 这种"列表内插队"机制影响,针实际上只会靠近开头或结尾两端,不会出现在中间附近,这一点和 `s_niah`/NIAH 精确的字符级 `depth_pct` 控制完全不同,不能一概而论。
- `check_answer` 对 list 类型答案只检查"expected 里的每一项是否都出现",不检查模型有没有多编造几个不存在的值——也就是只考核"召回",不考核"精确率",预测结果里夹带私货(多输出几个错误数字)照样能被判定为正确。

---

## 3. Lost in the Middle 现象(来自 lecture 12)—— 检索准确率的 U 型曲线

**是什么:** 这是一个来自外部研究的实证现象(`learning/long-context/lectures/12-long-context-pitfalls.md` 引用的 Liu 等 2023 年论文),**不是本仓库自己跑出来的实验结果**。lecture 12 第 1 页给出的原始描述是:
```
LLM accuracy 在 needle 位于
首部 90%, 中部 40%, 尾部 70%
                ↑ "middle U-curve"
```
也就是说,同一个模型、同一个"能不能找到关键信息"的任务,只是把关键信息挪到上下文的不同位置,正确率就能从 90% 掉到 40% 再回升到 70%——画成曲线,两端高、中间低,形状是个 U。

把这句话画成图(用 lecture 给出的三个具体数字 90%/40%/70%,不是另外新造的数字):

```
准确率
100% |
 90% | ●
 80% |
 70% |                                                    ●
 60% |
 50% |
 40% |                        ●
     +---------------------------------------------------------
        首部(depth_pct=0%)   中部(depth_pct=50%)   尾部(depth_pct=100%)
```

两端高、中间低——这就是"middle U-curve"这个名字的由来。也可以和知识点 1 新增的那张热力图对上号:固定住某一个 context length、横着切一刀,格子里"✓ 变 ✗ 再变 ✓"的规律,形状上大致就对应这条曲线。

**一句话:** 长上下文模型检索信息的准确率不是"在整个上下文窗口内均匀可靠",而是开头和结尾位置的信息容易被正确取用,中间位置的信息容易被忽略,哪怕模型名义上支持的上下文窗口很长。

**底层机制/为什么这样设计:** 严格说这一条不是"设计",是"病因"——lecture 12 第 2 页给出三点归因,这里逐条讲清楚为什么它们会导致中间掉分,不是简单照抄:
1. **RoPE 高频维度对近距离 attention 有偏置**——旋转位置编码里,高频维度旋转得快,相对距离一拉远,不同位置之间的点积相似度衰减得也快,这天然让模型的注意力更容易聚焦在"离当前位置近"的 token 上;上下文中部的信息离开头和结尾的 token 都不算近,天然吃亏(旋转位置编码本身的机制会在本系列 01 RoPE 外推家族里展开,这里只借用其结论)。
2. **causal mask 让前文 token 被"看"的次数系统性更多**——在自回归解码里,第 1 个 token 会被后面所有 token 的每一层 attention 反复关注,而最后一个 token 只有它自己能看到自己;越靠前的 token,在训练和推理过程中累积被关注、被编码进后续隐藏状态的机会本来就越多,这是"开头"位置占优的一部分原因。
3. **SFT 数据分布偏短**——微调数据里长对话/长文档样本本来就稀少,大部分样本是短对话,模型对"一段几万 token 的上下文正中间应该怎么被检索"这件事本身缺乏针对性训练,相当于"冷启动"。

lecture 同一页也给出缓解方向:RoPE scaling(YaRN 一类的方法,让长距离位置的编码更合理)、专门用 NIAH 风格的长数据做 SFT、用 middle-truncation 测试驱动开发——但特别标注"vanilla attention 没有改善",也就是说单纯换一种更快的 attention 实现(不改训练数据、不改位置编码)并不会让这条 U 型曲线变平。

这条现象和知识点 1、2 是同一件事的两面:正因为存在 Lost in the Middle,NIAH/RULER 才必须把 `depth_pct` 当成一个要扫描的独立变量,而不是只在文本末尾测一次——lecture 12 第 8 页那句"广告 128k window,实际 NIAH 90%+ 只有 32k,高质量 retrieval 只有 16k",本质上就是"模型宣传的窗口大小"和"U 型曲线中间那一段掉下去的准确率"共同作用的结果。

**AI 研究场景:** 这条结论直接影响两类工程决策。第一,评测设计上,如果只在 `depth_pct=100`(文本末尾)测试,几乎必然高估模型的真实长上下文能力,U 型曲线告诉你"中间"才是压力测试该覆盖的关键区间。第二,RAG/agent 的 prompt 拼接策略上,lecture 12 第 9 页给的建议是"长 ctx + RAG 组合"——不能因为模型宣传上下文很长,就假设把所有检索到的文档一股脑塞进去、丢给模型自己找,更稳妥的做法是把最关键的证据放在拼接后 prompt 的开头或结尾附近,而不是任由它落在中段。KV cache 显存开销(lecture 12 第 4 页)是另一个相关但独立的话题,留给 04 数据工程与 Capstone 那一批展开,这里不重复。

**可运行例子(如实说明边界):** 本仓库没有任何代码真正跑模型去验证这条 U 型曲线本身——`niah_eval.py`/`ruler_eval.py` 都不调用模型(见知识点 1、2),仓库里也没有别的地方把 needle 测试接到一个真实模型上统计准确率。下面这段代码复用知识点 1 的 `make_niah_query`,构造 `depth_pct=0/50/100` 三个测试用例,**只演示"深度参数怎么控制针插入文本的位置"**——这是 U 型曲线横轴(depth)的可操作定义,不是曲线纵轴(准确率)的复现:
```python
import sys
sys.path.insert(0, "learning/long-context/src")
from niah_eval import make_niah_query

# 只构造测试用例,不调用任何模型——U 型曲线的"准确率"这一半不在这段代码里
for d in [0, 50, 100]:
    q, code = make_niah_query(target_length=1000, depth_pct=d, needle_code="7777")
    rel = q.find("7777") / len(q)
    print(f"depth_pct={d:>3} -> 针实际插入在文本 {rel*100:5.1f}% 处")

q0, _ = make_niah_query(1000, 0, needle_code="1111")
q50, _ = make_niah_query(1000, 50, needle_code="2222")
q100, _ = make_niah_query(1000, 100, needle_code="3333")
assert q0.find("1111") / len(q0) < 0.05             # depth 0%   -> 靠近开头
assert 0.45 < q50.find("2222") / len(q50) < 0.55     # depth 50%  -> 中间
assert q100.find("3333") / len(q100) > 0.90          # depth 100% -> 靠近结尾
```
**实测输出:**
```
depth_pct=  0 -> 针实际插入在文本   2.2% 处
depth_pct= 50 -> 针实际插入在文本  47.9% 处
depth_pct=100 -> 针实际插入在文本  93.6% 处
```
如果要真正验证 Lost in the Middle 这条曲线本身,需要在这三个 `depth_pct` 各生成几十个测试用例,接一个真实模型的 `generate()`,再用 `check_answer` 统计每个深度的通过率——这正是知识点 1 提到的 `eval_niah` 那段 lecture 伪代码在做的事,但本仓库没有实现、也没有跑过这一步。

**面试怎么问 + 追问链:**
- **Q:** "一个模型宣称支持 128k 上下文窗口,你会怎么验证这个数字是不是'能打'?" —— 期望候选人主动提到不能只测末尾位置,要覆盖不同深度(尤其是中间),这正是 NIAH/RULER 方法论存在的理由。
- **追问 1:** "为什么信息放在上下文中间比放在开头/结尾更容易被模型漏掉?说说你知道的原因。" —— 期望至少答出 RoPE 的远距离衰减、causal mask 下前文被关注次数更多、SFT 数据长文档样本少这三条里的一到两条,而不是只会说"attention 有限"这种模糊表述。
- **追问 2(区分度很高):** "知道这个问题以后,你在设计一个需要塞很多检索结果进 prompt 的 RAG 系统时,会怎么调整拼接策略?" —— 开放题,期望候选人提出"把最关键的证据放在开头或结尾""不要无脑依赖超长上下文一次性放完所有材料,长上下文该和检索/重排流程配合使用"这类具体应对,而不是"再加大 context window 就行了"——加大窗口本身不解决 U 型曲线,lecture 明确写了"vanilla attention 没有改善"。

**常见坑:**
- 把"模型卡上写的 context length"直接当成"这个模型在这个长度下处处好用"——lecture 12 第 8 页的原话是"context limit ≠ usable limit",这条坑几乎是本知识点存在的全部意义。
- 把 Lost in the Middle 当成"只有老模型才有的过时问题"——lecture 也提到 Llama-3.1/DeepSeek-V3/GPT-4o 这类做过长上下文专门训练 + RoPE scaling 的现代模型"middle 不太差",但"不太差"不等于"曲线消失",而且这份改善来自额外的训练投入,不是"模型变新了就自动没事"。
- 最容易踩的坑是把本节的"构造测试用例"和"验证了 U 型曲线"混为一谈——上面的可运行例子只证明了 `depth_pct` 参数确实能精确控制针在文本里的插入位置,U 型准确率曲线本身是 lecture 引用的外部研究结论,本仓库既没有实现、也没有运行过任何"真正测量不同深度下模型准确率"的代码。
