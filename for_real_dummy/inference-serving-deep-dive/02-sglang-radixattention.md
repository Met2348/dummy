# 02 · SGLang RadixAttention 深挖(SGLang RadixAttention)

> 总览见 [00-roadmap.md](00-roadmap.md)

SGLang 和 vLLM 出自同一个团队(LMSys),但解决的是"推理服务"这同一个问题的不同侧面:vLLM 的 PagedAttention 把"单个大 batch 怎么把一张 GPU 的显存利用到极致"做到了极致;SGLang 的 RadixAttention 把"大量请求之间彼此重复的部分怎么彻底省掉"做到了极致——system prompt、few-shot 示例、agent 多轮历史、tree-of-thought 分支,这些场景里请求和请求之间共享着大段前缀,vLLM 的 block 级 prefix caching 只能在"凑巧对齐"时吃到这部分收益,SGLang 用一棵 token 级的 radix tree 把这个限制彻底打开。本文是 `inference-serving-deep-dive` 系列第 2 个文件,9 个知识点,源码全部在 `learning/sglang-radixattention/src/{radix_tree,constrained_sampler,grammar_fsm,jump_forward,frontend_lang,agent_patterns,agent_server,sglang_compare,sglang_original_minimal}.py`(9 个文件、929 行)+ 11 篇 lecture(565 行):知识点 1 是"全图",讲 SGLang 和 vLLM 的设计哲学差异到底体现在哪几层;知识点 2-3 讲 RadixAttention 本身(树结构怎么把 prefix caching 从"必须整块对齐"泛化成"任意长度前缀都能共享",以及这棵树的 split/insert/evict 具体怎么实现);知识点 4-6 讲受约束解码这条线(采样前加 mask → 把 mask 编译成 FSM 查表 → 利用"当前唯一合法字符"跳过没必要的前向);知识点 7-8 讲面向程序员的 Frontend DSL 和它撑起来的几种 agent 模式;知识点 9 是 Capstone——32 并发 agent server 的 radix 命中率实测,外加和 vLLM 的 5 场景横评。

**和 01 号文件的关系:** 01 号文件(`01-inference-engine-core.md`)知识点 7 讲的是 vLLM 风格的 Prefix Caching(`learning/inference-engine-core/src/prefix_cache.py::PrefixCache`)——按固定 `block_size`(默认 16 token)把 prompt 切块、对每块算 hash,hash 命中就复用那块的物理 KV,不命中就整块重新计算。这个方案有一个结构性死角:**共享前缀必须严格对齐到 `block_size` 的整数倍**,哪怕两个请求前 9 个 token 完全相同,只要它们所在的整个 16-token block 有任何一处不同,这 9 个 token 的复用收益就是 0(知识点 2 会用两个模块的真实代码并排跑出这个"0 vs 9"的具体数字,不是转述)。RadixAttention 要解决的正是这个死角:把"按固定大小的块查哈希表"换成"按 token 逐个走一棵前缀树(radix tree)",共享粒度从"block(16 token)"降到"1 token",任意长度的公共前缀都能被树自动识别、自动复用,不需要请求双方凑巧落在同一个 block 边界上。这也是为什么 00-roadmap.md 把知识点 2 定位成"承接 01 号文件知识点 7"——两者是同一个问题(避免重算重复前缀)在"够不够通用"这个维度上的两级实现,建议先理解 01 号文件的 block-hash 方案,再看本文怎么把它泛化成树;如果你还没读过 01 号文件也不影响阅读本文,知识点 2 会把 `PrefixCache` 需要知道的部分现场讲清楚。

**一个重要的诚实标注:** `environment/requirements.txt` 声明了 `sglang>=0.4`、`outlines>=0.0.45`、`xgrammar>=0.1`,但本文逐文件用 `ast` 模块解析核实过——全部 9 个 `src/*.py` 文件**零真实 import** 这三个库(也不 import torch/transformers/numpy),只有 stdlib(`dataclasses`/`typing`/`math`/`json`/`time`/`collections`)和模块间互相 import;`environment/verify_env.py` 会显式尝试 `import sglang`/`outlines`/`xgrammar` 并在失败时打印"NOT installed"而不是报错——这是模块自己的诚实设计,不是本文的猜测。`agent_server.py`(知识点 9 的 capstone)**不是真网络服务**,`run_server()` 是一次性跑完 32 个 agent 的 Python 循环,返回 `AgentMetrics` 后直接 `return`,没有端口、没有请求-响应循环。`sglang_compare.py`(同样在知识点 9 里)docstring 第一行就写"No real engine called"——它只对"prefill token 数量"建了一个简化的合成成本模型,不是真实 tok/s 吞吐测试。撰写过程中另外发现了几处源码本身"文档写的和代码实际做的不一致"的地方,如实记录、不代为掩饰:`radix_tree.py::RadixTree` 的 `root` 字段不参与自动生成的 `__eq__` 比较(知识点 3);`grammar_fsm.py` 模块 docstring 宣称支持的 `\w`/`.`/`+`/`*`/`a|b` 语法一个都没实现,连同 lecture 强调的"token-level 查表提升"函数本身也是全模块零调用点的死代码(知识点 5);`frontend_lang.py::function` 装饰器没用 `functools.wraps`,导致被装饰函数在 `__name__`/`__module__` 上"失忆"(知识点 7)。这些都是本文动手验证后的发现,不是源码作者自己标注的已知问题。

**环境声明:** 本文全部 9 个知识点的源码都是纯 Python 标准库实现,不需要 GPU、不需要装 torch/sglang/transformers,在仓库根目录 `.venv`(Windows 原生,Python 3.13.9)下秒级跑完,和系列里需要真实部署的 06 号文件(WSL2 + 真 vllm)性质完全不同。本文所有代码例子已用 `.venv/Scripts/python.exe` 实际跑通验证,文中数字是真实输出,不是估算或转述。

---

## 1. SGLang 全图与 vLLM 设计哲学差异(L01)—— 同一个团队,两种取舍

**是什么:**
```python
@dataclass
class RadixTree:
    cap: int = 100_000          # max tokens stored
    n_tokens: int = 0
    next_slot: int = 0
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        self.root = Node()
```
(`radix_tree.py:28-37`,创新一:KV cache 用 radix tree 而不是 block 哈希表)

```python
def apply(self, logits: List[float]) -> List[float]:
    mask = self.legal_token_mask()
    out = list(logits)
    for i, ok in enumerate(mask):
        if not ok:
            out[i] = -math.inf
    return out
```
(`constrained_sampler.py:30-36`,创新二:受约束解码在采样前直接把非法 token 摁成 `-inf`)

```python
@dataclass
class Gen:
    name: str
    max_tokens: int = 16
    stop: Optional[List[str]] = None


@dataclass
class Select:
    name: str
    choices: List[str]
```
(`frontend_lang.py:51-61`,创新三:`gen`/`select`/`fork` 这套 DSL 原语)

**一句话:** SGLang 和 vLLM 出自同一团队,但优化目标不同——vLLM 把"单个大 batch 尽可能塞满 GPU"做到极致(block 级 KV 管理 + PagedAttention),SGLang 把"大量请求之间的结构性重复"做到极致(token 级 radix tree 共享 + grammar 级约束解码 + DSL 级 fork/select 原语),两者不是互相替代关系,是"同一套显存管理思想"在两个不同优化方向上的分野。

**底层机制/为什么这样设计:** L01 给出的对比表格有四个维度(KV cache:block 哈希 vs radix 树;共享粒度:block(16 token) vs token(1);Constrained Decoding:outlines vs xgrammar;Frontend:OpenAI API vs DSL+API),这四个维度不是四个孤立的技术选型,而是一条一致的设计主线——SGLang 在每一层都选择了"更细粒度、但需要更多运行时基础设施"的方案:radix tree 比 block 哈希表更细粒度,但要维护树结构(split/evict,知识点 3);xgrammar 比 outlines 更快,是因为把 FSM 预编译成查表(知识点 5);DSL 比裸 OpenAI API 更灵活,是因为运行时能看到程序的完整结构(fork/select,知识点 7-8),从而知道在哪里可以复用 KV。反过来,vLLM 选择相对更简单粗放的方案(block 哈希、outlines、纯 API),换来的是更小的运行时复杂度和更成熟的生态——README 自己的结论是"vLLM OpenAI API 兼容更全 → 公网接 SDK 选 vLLM"。这不是"SGLang 全面更好":L01 自己的表格就承认"长 prompt 单请求"场景 vLLM 略胜——此时请求之间几乎没有共享,radix tree 的遍历/维护开销变成纯粹的额外成本,拿不到任何补偿(知识点 9 的 5 场景横评会给出这条结论对应的具体数字)。

**AI 研究场景:** 这四个维度的选型差异,最终都会落到同一类工作负载上——agent/多轮/结构化输出。README 用一张星级表总结:ToT 8 路、JSON 结构化、ReAct 5 步,SGLang 都是 ★★★★★,而"单 system prompt 大 batch""长 prompt 单请求"两个引擎打平或 vLLM 略胜。这不是巧合:前三个场景的共同点是"一次请求会产生多条 KV 路径(fork)或对输出格式有强约束",分别对应知识点 2-3(radix tree 捕捉分支间的共享前缀)和知识点 4-6(约束解码保证格式合法、jump-forward 省解码步数);后两个场景没有这些结构,SGLang 的额外机制自然也就没有用武之地。

**可运行例子:**
```python
import ast
import pathlib
import sys

REPO_ROOT = pathlib.Path("E:/Workspace/dummy")
SGLANG_SRC = REPO_ROOT / "learning/sglang-radixattention/src"
sys.path.insert(0, str(SGLANG_SRC))

# Claim 1: requirements.txt 声明了 3 个重型推理服务专用库。
req_text = (REPO_ROOT / "learning/sglang-radixattention/environment/requirements.txt").read_text()
declared = {line.split(">=")[0].split("==")[0].strip() for line in req_text.splitlines() if line.strip()}
assert {"sglang", "outlines", "xgrammar"} <= declared

# Claim 2: 9 个 src 文件没有一个真的 import 过任何重型运行时/ML 库。
heavy_libs = {"sglang", "outlines", "xgrammar", "torch", "transformers", "numpy", "vllm"}
py_files = sorted(SGLANG_SRC.glob("*.py"))
assert len(py_files) == 9
offenders = []
for f in py_files:
    tree = ast.parse(f.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders += [(f.name, a.name) for a in node.names if a.name.split(".")[0] in heavy_libs]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in heavy_libs:
                offenders.append((f.name, node.module))
assert offenders == []

# Claim 3: 尽管零重型依赖，"三大创新"仍然是真实、可 import、能跑的代码。
from radix_tree import RadixTree
from constrained_sampler import ConstrainedSampler
from frontend_lang import Stream, Gen, Select, function

# Claim 4（粒度对比，为知识点 2 打前站）：RadixTree 没有任何 block_size 概念，
# 共享粒度天然是 1 个 token，不是某个固定块大小的整数倍。
assert not hasattr(RadixTree, "block_size")
tree = RadixTree()
tree.insert([1, 2, 3])
_, matched = tree.insert([1, 2, 3, 4])   # 只比已有前缀多 1 个 token
assert matched == 3                       # 精确复用 3 个 token，不会被下取整成 0 或某个块的倍数
```

实测(`.venv` 真跑):`SGLANG_SRC.glob("*.py")` 精确列出 9 个文件;`offenders` 列表为空(`[]`),即 9 个文件对 `{sglang, outlines, xgrammar, torch, transformers, numpy, vllm}` 里任何一个库都零 `import`/`from...import`;`RadixTree` 没有 `block_size` 类属性;插入 `[1,2,3]` 再插入 `[1,2,3,4]` 时 `matched=3`——精确等于已有前缀长度,不像 01 号文件的 `PrefixCache` 那样会因为没凑够一个完整 block 而只能拿到 0(知识点 2 展开这一点)。

**面试怎么问 + 追问链:**
- **Q:** "SGLang 和 vLLM 都出自 LMSys,两者最核心的设计差异是什么?"—— 期望说出"vLLM 优化单 batch 的显存利用率(PagedAttention),SGLang 优化请求间的结构性重复(RadixAttention + 约束解码 + DSL),不是谁更强,是优化方向不同"。
- **追问 1:** "是不是任何场景 SGLang 都比 vLLM 好?"—— 期望明确说"不是",并举出"长 prompt 单请求"这个 vLLM 略胜的反例,解释原因是这类请求彼此之间没有共享结构,radix tree 的维护开销拿不到补偿。
- **追问 2(考察是否真读过依赖清单):** "这个模块的 `requirements.txt` 写了 `sglang`/`outlines`/`xgrammar`,是不是意味着跑这些例子需要先装好 SGLang?"—— 期望说出"不需要,这三个库在全部 9 个 src 文件里零真实 import,`requirements.txt` 反映的是'真实生产环境需要什么',不是'这份教学代码依赖什么',这个区分在阅读任何教学向仓库时都要留意"。
- **追问 3:** "共享粒度从 block(16 token)降到 token(1)，代价是什么?"—— 期望说出"radix tree 需要维护父子指针、在前缀分叉处做节点分裂,查找是 O(前缀长度) 的树遍历,不是 block 哈希表 O(1) 的字典查找",为知识点 2-3 的机制铺垫。

**常见坑:** 把"SGLang 三大创新更细粒度"直接等同于"SGLang 全面碾压 vLLM"——L01/L10/README 都明确写了"长 prompt 单请求"场景 vLLM 略胜,选型应该看具体工作负载而不是无脑选"更新的方案"。另一个坑是看到 `requirements.txt` 列出重型库就假设这些教学脚本背后真的在调用它们——本知识点已经用 AST 解析核实过零真实 import,这是判断"教学代码"与"生产代码"边界的一个通用方法(不要只看依赖清单,要看真实 import 语句)。

---

## 2. RadixAttention 原理(L02)—— prefix caching 的树结构泛化,承接 01 号文件

**是什么:**
```python
@dataclass
class Node:
    token_ids: List[int] = field(default_factory=list)
    children: Dict[int, "Node"] = field(default_factory=dict)
    parent: Optional["Node"] = None
    kv_slots: List[int] = field(default_factory=list)
    refcount: int = 0
    last_access: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return not self.children
```
(`radix_tree.py:14-25`)

```python
def match(self, prefix: List[int]) -> Tuple[Node, int]:
    """Walk as far as `prefix` matches the tree.

    Returns (deepest matched node, n_tokens_matched).  When the match ends
    midway through a node's `token_ids`, the returned node is that node
    and caller is responsible for splitting before insertion.
    """
    node = self.root
    i = 0
    while i < len(prefix):
        child = node.children.get(prefix[i])
        if child is None:
            return node, i
        # walk through child's segment
        seg = child.token_ids
        j = 0
        while j < len(seg) and i + j < len(prefix) and seg[j] == prefix[i + j]:
            j += 1
        ...
```
(`radix_tree.py:41-58`,省略号后(59-67 行)是分裂点判断,细节留给知识点 3)

对照组,01 号文件 `PrefixCache` 的核心逻辑:
```python
def mount(self, token_ids: List[int]) -> Tuple[List[int], int]:
    """Return (phys_block_ids, hit_token_count) for a prompt prefix."""
    ids: List[int] = []
    hit_tokens = 0
    for i in range(0, len(token_ids), self.block_size):
        blk = tuple(token_ids[i : i + self.block_size])
        if len(blk) < self.block_size:
            break       # partial trailing block never cached
        h = block_hash(blk)
        if h in self.table:
            ...
```
(`learning/inference-engine-core/src/prefix_cache.py:31-40`,省略号后是 phys id 分配与 LRU 记账,不是本文重点)

**一句话:** RadixAttention 是把 vLLM 那种"定长 block 查哈希表"的 prefix caching,替换成"变长 segment 组成的前缀树(radix tree)"——共享单位从"`block_size` 个 token"缩小到"1 个 token",任何两个请求只要前面有公共 token,不管这段公共前缀多长、在哪里结束,都能被树自动识别并复用,不需要凑巧落在 block 边界上。

**底层机制/为什么这样设计:** `PrefixCache.mount()` 把 prompt 切成不重叠的定长 block(默认 16 token),逐 block 算哈希、查表,一个 block 要么整块命中要么整块不命中——它的 `for` 循环步长就是 `self.block_size`,天然没有"半个 block 命中"这个概念。`RadixTree.match()` 完全是另一种思路:从根节点开始,沿着 `prefix` 逐 token 走,`child = node.children.get(prefix[i])` 先看第一个 token 有没有对应的子节点,找到后再用内层 `while` 逐个比较 `seg[j] == prefix[i+j]`,一旦某个位置不相等就停下——返回的 `matched` 是"实际重合的 token 数",可以是任意整数,不需要凑够一个 block。这个差异不是"哪个实现得更好",是数据结构选型决定的:哈希表用"整块相同才让 key 相同"的方式把比较开销降到 O(1),但因此丢掉了"部分相同"这个信息;树用"逐 token 走、走不下去就停"的方式保留了"部分相同"的信息,但查找开销从 O(1) 变成了 O(前缀长度)——用维护成本换取了粒度上的精确性。这也解释了为什么树上的节点还要维护 `refcount`(有多少个在途请求依赖这个节点)和 `last_access`(LRU 淘汰用):定长 block 因为切分方式固定,一个 block 淘汰了最多影响"卡在这个 block 边界之后"的请求;树里一个节点如果被误删,可能会切断某条正在被大量请求共享的深层分支,所以需要更精细的引用计数和淘汰策略,这些细节留给知识点 3。

**AI 研究场景:** `sglang_original_minimal.py::lm_program_prompts()` 构造了三类真实存在的共享模式:8 个"agent 调用"共享 `SYSTEM+TOOL_TEMPLATE`(不同的 query)、6 个"few-shot benchmark 调用"共享 `SYSTEM+FEW_SHOT`(不同的最终问题)、4 个"fork 分支"共享同一个 `branch_prefix`(不同的分支维度)——这正是知识点 1 提到的"agent/多轮/结构化输出"工作负载在 token 序列层面的具体样子。定长 block 哈希在这三类场景里都可能因为"共享部分长度不是 16 的整数倍"而白白损失一部分本该免费的复用,而树结构不会有这个损失,这正是 RadixAttention 论文选择"树"而不是"继续优化哈希表"的根本原因。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/sglang-radixattention/src")
sys.path.insert(0, "learning/inference-engine-core/src")
from radix_tree import RadixTree
from prefix_cache import PrefixCache
from sglang_original_minimal import summarize

# "Hello world" vs "Hello there" 的 token 化版本：前 9 个 token 相同，随后分叉，
# 总长度 20（超过 1 个 16-token block，但共享部分没有凑够一整块）。
shared = list(range(9))
prompt_a = shared + [900 + i for i in range(11)]
prompt_b = shared + [800 + i for i in range(11)]

# vLLM 式定长 block 哈希（block_size=16，生产默认值）：block 边界是全对齐或全不对齐。
pc = PrefixCache(block_size=16)
_, hit_a = pc.mount(prompt_a)
_, hit_b = pc.mount(prompt_b)
assert hit_a == 0                 # 第一次插入，本就该是 0
assert hit_b == 0                 # !! b 和 a 共享 9/20 个 token，block 哈希却完全看不见

# SGLang 式 radix tree：逐 token 走，精确找到这 9 个 token 的重合。
rt = RadixTree()
rt.insert(prompt_a)
_, matched_b = rt.insert(prompt_b)
assert matched_b == 9              # 精确捕捉 block 哈希漏掉的 9-token 重合

# 公平对照：共享部分本身就是 block_size 的整数倍时，block 哈希也能吃到全部收益，
# 不是 PrefixCache "坏了"，只是粒度更粗。
aligned_shared = list(range(16))
pc2 = PrefixCache(block_size=16)
pc2.mount(aligned_shared + [1])
_, hit_d = pc2.mount(aligned_shared + [2])
assert hit_d == 16

# sglang_original_minimal.py：论文形态最小例，LM-program 风格的多级共享 + jump-forward。
result = summarize()
assert result["saved_prefill_tokens"] > 0
assert result["hit_rate"] > 0.5
assert result["naive_prefill_tokens"] == 670          # 8*36 + 6*45 + 4*28，三类共享模式精确求和
```

实测(`.venv` 真跑):`hit_a=0`(首次插入,正常),`hit_b=0`——尽管 `prompt_b` 和已插入的 `prompt_a` 共享前 9 个 token,block 哈希(`block_size=16`)依然判定 0 命中,因为这 9 个 token 落在同一个 16-token block 内部、block 剩余部分不同,整块哈希就不相等;`matched_b=9`——radix tree 精确捕捉到这个哈希表看不见的重合。公平对照下 `hit_d=16`,证明 `PrefixCache` 本身没有 bug,只是粒度被锁定在 `block_size` 的整数倍。`sglang_original_minimal.py::summarize()` 的真实输出:`naive_prefill_tokens=670`(等于 `lm_program_prompts()` 三类共享模式的 token 数精确求和:8 个 agent 调用 × 36 + 6 个 few-shot 调用 × 45 + 4 个 fork 分支 × 28)、`radix_prefill_tokens=83`、`saved_prefill_tokens=587`、`hit_rate=0.876`。

**面试怎么问 + 追问链:**
- **Q:** "vLLM 的 prefix caching 和 SGLang 的 RadixAttention 本质区别是什么?"—— 期望说出"vLLM 按固定 block_size 切块算哈希,必须整块对齐才能共享;SGLang 用 radix tree 逐 token 匹配,任意长度的公共前缀都能共享"。
- **追问 1(要求给出可验证的证据):** "能不能构造一个具体例子,证明 block 哈希在某种情况下会'漏掉'本该可以共享的部分?"—— 期望想到"构造两个共享前缀长度不是 16 整数倍的 token 序列,分别喂给两种实现,对比命中数",这正是本知识点验证代码的思路。
- **追问 2:** "radix tree 是不是在任何情况下都比 block 哈希更好?"—— 期望说出"不是,当共享部分恰好是 block_size 的整数倍时,两者收益相同,但 block 哈希查找是 O(1) 字典查找,radix tree 是 O(前缀长度) 的树遍历,后者用维护成本换取了粒度精确性,不是免费的",避免把"更细粒度"简单等同于"更优"。
- **追问 3(深挖数据结构选型):** "既然要更细粒度,为什么不直接把 block_size 设成 1,而是换一整套树结构?"—— 期望说出"block_size=1 的哈希表退化成'逐 token 查字典',丢失了'公共前缀'这个结构信息,没法回答'这两个请求到底从哪个 token 开始分叉'这种问题;而树天然按前缀组织,分叉点、共享深度这些信息都直接体现在树形状里,这也是为什么淘汰(evict)、引用计数(refcount)这些管理逻辑要建在树节点上,而不是哈希表的 key 上(知识点 3 展开)"。

**常见坑:** 看到 `PrefixCache` 在这个例子里命中数是 0 就误以为它"有 bug"或者"设计得差"——它只是选择了粒度更粗、查找更便宜的方案,在共享部分恰好对齐的场景下(比如固定长度的 system prompt 后面紧跟固定长度的 few-shot 示例)两者效果相同。另一个坑是忽视 `RadixTree` 这里的 `kv_slots` 只是整数占位符,不是真实显存地址——一个真实系统仍然需要在物理层面管理 KV block(比如底层复用 PagedAttention 的分页机制),RadixAttention 解决的是"逻辑上怎么发现共享",不是"物理上怎么分配显存",这两层需要一起工作,不是二选一的关系。

---

## 3. Radix Tree 实现细节(`radix_tree.py`,L03)—— split/insert/evict 怎么维护一棵树

**是什么:**
```python
def _split(self, node: Node, at: int) -> Node:
    """Split `node.token_ids` at offset `at`; returns the head.

    `node` becomes the *tail*: its parent and head are updated, children
    remain on the tail (since the old leaf state lives there).
    """
    head_tokens = node.token_ids[:at]
    tail_tokens = node.token_ids[at:]
    head_slots = node.kv_slots[:at]
    tail_slots = node.kv_slots[at:]
    head = Node(
        token_ids=head_tokens,
        parent=node.parent,
        kv_slots=head_slots,
        last_access=node.last_access,
        refcount=node.refcount,
    )
    # Rewire parent to head.
    node.parent.children[head_tokens[0]] = head
    # mutate node into tail
    node.token_ids = tail_tokens
    node.kv_slots = tail_slots
    node.parent = head
    head.children = {tail_tokens[0]: node}
    return head
```
(`radix_tree.py:71-95`)

```python
def insert(self, prefix: List[int]) -> Tuple[Node, int]:
    """Insert/lookup prefix, return (leaf_node, n_tokens_matched_existing)."""
    node, matched = self.match(prefix)
    # may need to split if match landed inside a node
    if node is not self.root and matched > 0:
        # how far into `node`'s segment did we land?
        # We can compute by walking from root summing each ancestor's len.
        depth = self._depth(node)
        into = matched - (depth - len(node.token_ids))
        if 0 < into < len(node.token_ids):
            node = self._split(node, into)
    # add new node for remainder
    remainder = prefix[matched:]
    if remainder:
        ...
        node.children[remainder[0]] = new_node
    ...
    return node, matched
```
(`radix_tree.py:99-127`,`...` 处省略了新叶子节点的具体构造,不影响所示的分裂判断逻辑)

**一句话:** `_split()` 在前缀匹配到某个节点中间时,把这个节点从 offset 处切成 head(公共前段)和 tail(分叉后段)两个节点,用"原地修改原节点变成 tail、只新建一个节点当 head"的方式重新挂好父子关系;`insert()` 先 `match()` 找到匹配点,如果匹配点落在某个节点内部就调 `_split()`,最后把没匹配上的剩余 token 挂成新叶子。

**底层机制/为什么这样设计:** `_split()` 有一个容易被忽略但设计得很讲究的细节——它**没有**新建两个节点替换旧节点,而是保留原 `node` 对象、只修改它的 `token_ids`/`kv_slots`/`parent`(让它变成 tail),另外新建一个 `head` 对象接过原节点在父节点里的位置。这个"对象身份保留"的写法不是随手为之:如果某个请求在 split 发生**之前**已经通过 `acquire()` 拿到了这个节点的引用(比如一个正在等待 decode 的请求持有着它命中的 leaf),split 发生之后,这个引用对象本身没有变(还是同一个 Python 对象),只是它的内容变成了 tail 部分——调用方完全不需要知道发生过 split,沿着这个引用的 `.parent` 继续往上走,会自动、正确地穿过新生成的 `head` 节点到达 root。如果 `_split()` 换成"新建两个对象、丢弃旧对象"的写法,任何已经持有旧节点引用的调用方都会立刻读到过期数据。`insert()` 里 `into = matched - (depth - len(node.token_ids))` 这一行是在算"匹配点具体落在 `node.token_ids` 的第几个位置"——因为 `match()` 返回的 `matched` 是从 root 算起的绝对 token 数,而 `_split()` 要的 `at` 是相对于 `node` 自己这个 segment 的局部偏移,`_depth(node)` 先算出 `node` 底部对应的绝对深度,再倒推局部偏移,这是一处容易在"绝对位置"和"节点内相对位置"之间搞混的地方。

关于本知识点动手验证时发现的一处真实边界情况(严格来说不是"bug",但是一个容易踩的坑):`RadixTree.__post_init__` 里 `self.root = Node()` 是**在 `@dataclass` 生成的 `__init__` 跑完之后、以普通属性赋值的方式**设置的,`root` 从未出现在类体的字段标注里(对照 `cap`/`n_tokens`/`next_slot`/`hits`/`misses` 这五个才是真正的 dataclass 字段)。Python 的 `@dataclass` 自动生成 `__eq__` 时只比较**标注过的字段**,`root` 完全不在比较范围内——这意味着两棵树的实际前缀树内容(它们各自 `root.children` 里挂的是什么)对 `==` 比较完全不可见,只要两棵树的 `cap`/`n_tokens`/`next_slot`/`hits`/`misses` 五个计数器凑巧相同,哪怕内部存的 token 完全不相干,`==` 也会判定"相等"。本知识点用真实代码验证了这一点(见下方"可运行例子")。

**AI 研究场景:** `evict()` 对应真实多租户系统里一个必须回答的问题:KV cache 容量到了,该清理谁?代码里 `_leaves_zero_ref()` 只从**叶子且 `refcount==0`** 的节点里选淘汰对象(`n.is_leaf and n.refcount == 0`),这不是随意的限制——lecture L02 第 7 节点明了原因:"evict 内部节点会破坏树结构"。如果允许淘汰一个还有子节点的内部节点,它下面所有依赖这段前缀的子树都会失去挂载点;`refcount>0` 的节点则代表"当前有在途请求正依赖着这段前缀",淘汰它会导致这些请求的 KV 数据凭空消失。这套"只淘汰无引用叶子"的规则,和操作系统虚拟内存管理里"只能换出没有进程在用的页"是同一类约束。

**可运行例子:**
```python
import sys
import dataclasses
sys.path.insert(0, "learning/sglang-radixattention/src")
from radix_tree import RadixTree, Node

tree = RadixTree()
leaf1, m1 = tree.insert([1, 2, 3, 4, 5])
assert m1 == 0 and tree.n_tokens == 5
tree.acquire(leaf1)                        # 模拟一个在途请求持有这段前缀
assert leaf1.refcount == 1

leaf2, m2 = tree.insert([1, 2, 3, 8, 9])    # 触发对 [1,2,3,4,5] 节点在 offset=3 处的分裂
assert m2 == 3
assert tree.n_tokens == 7                   # 原有 5 + 新增 2（8,9）；[1,2,3] 被两支共享

# 对象身份保留：_split 原地把旧节点变成 tail([4,5])，leaf1 这个引用自动跟随。
assert leaf1.token_ids == [4, 5]
assert leaf1.refcount == 1                  # refcount 在 mutate 过程中保持不变
head = leaf1.parent
assert head.token_ids == [1, 2, 3]          # split 新建的 head 节点
assert head.refcount == 1                   # 从分裂前的节点复制而来，两段引用计数一致

# evict：只清理 refcount==0 的叶子；leaf1 分支被 acquire 钉住，不会被误删。
freed = tree.evict(want_tokens=2)
assert freed >= 2                           # 释放的是未被引用的 [8,9] 叶子
assert leaf1.token_ids == [4, 5]            # 被钉住的分支原封不动

# --- 一处真实发现：root 在 __post_init__ 里赋值，不是 dataclass 字段 ---
fields = {f.name for f in dataclasses.fields(RadixTree)}
assert "root" not in fields
assert fields == {"cap", "n_tokens", "next_slot", "hits", "misses"}

t1, t2 = RadixTree(), RadixTree()
t1.insert([1, 2, 3])      # 首次 miss，长度 3
t2.insert([7, 8, 9])      # 完全无关的 token，但同样是首次 miss、长度 3
assert (t1.cap, t1.n_tokens, t1.next_slot, t1.hits, t1.misses) == \
       (t2.cap, t2.n_tokens, t2.next_slot, t2.hits, t2.misses)
assert t1 == t2                             # !! dataclass 自动 __eq__ 判定"相等"
assert 1 in t1.root.children and 1 not in t2.root.children   # 但两棵树内容完全不相干
```

实测(`.venv` 真跑):`m1=0`、`tree.n_tokens=5`;`m2=3`、`tree.n_tokens=7`;split 后 `leaf1.token_ids` 变成 `[4, 5]`(同一个 Python 对象,原地变成了 tail),`leaf1.refcount` 仍是 `1`;新生成的 `head.token_ids=[1, 2, 3]`、`head.refcount=1`;`evict(want_tokens=2)` 释放了 `>=2` 个 token,且 `leaf1` 这条被 `acquire()` 钉住的分支完全没有受影响。`dataclasses.fields(RadixTree)` 精确返回 `{"cap", "n_tokens", "next_slot", "hits", "misses"}` 五个字段,`"root"` 确实不在其中;构造两棵插入了完全不同 token(`[1,2,3]` vs `[7,8,9]`,长度都是 3)的树后,`t1 == t2` 返回 `True`,但 `1 in t1.root.children` 为 `True`、`1 in t2.root.children` 为 `False`——两棵树的实际内容截然不同,`==` 却给出"相等"的错误印象。

**面试怎么问 + 追问链:**
- **Q:** "radix tree 在插入一个新前缀、发现匹配点落在某个已有节点中间时,具体怎么处理?"—— 期望说出"要把这个节点从匹配点处分裂成 head(公共部分)和 tail(分叉部分)两个节点,再把新的剩余 token 挂到分裂后的节点下面"。
- **追问 1(考察是否读过具体实现,不能只讲思路):** "分裂的时候,是新建两个节点替换旧节点,还是复用旧节点对象?这个选择重要吗?"—— 期望说出"是原地把旧节点 mutate 成 tail、只新建 head 节点;这样任何在分裂之前已经拿到旧节点引用的调用方(比如已经 acquire 过的请求)在分裂后依然能通过这个引用正确地走到 root,不会读到过期状态",能看出这是一个刻意设计而不是随手写的实现细节。
- **追问 2(核心陷阱,考察是否真的用代码验证过):** "两个 `RadixTree` 实例如果内容完全不同,`==` 比较结果一定是 `False` 吗?"—— 期望明确说"不一定,`root` 是在 `__post_init__` 里赋值的普通属性,不是 dataclass 标注字段,自动生成的 `__eq__` 根本不会比较它;只要两棵树的 `cap`/`n_tokens`/`next_slot`/`hits`/`misses` 这几个计数器凑巧相同,即使内部 token 完全不相干,`==` 也会判定相等",能说出这一点说明真的动手测过,而不是想当然地信任 dataclass 自动生成的方法。
- **追问 3:** "为什么 `evict()` 只从叶子节点里选淘汰对象,不能淘汰内部节点吗?"—— 期望说出"内部节点如果被淘汰,它下面所有子树都会失去挂载点,树结构会被破坏;只有叶子且 refcount=0(没有在途请求依赖)才是安全的淘汰对象,这和操作系统只能换出没有进程占用的内存页是同一类约束"。

**常见坑:** 信任 `RadixTree() == RadixTree()` 这类比较能反映树的真实内容——本知识点已经验证过这是一个真实的盲区,涉及需要精确比较两棵树内容是否相同的场景(比如写单元测试断言"两次运行结果一致"),应该显式比较 `root` 子树(或者 `total_nodes()`/遍历比较),不能依赖默认生成的 `==`。另一个坑是以为 `_split()` 里"谁是 head、谁是 tail"可以随意互换实现——实际上必须让**原节点对象**变成 tail 才能保证外部持有的引用不失效,如果反过来让新建的对象变成 tail、原对象变成 head,任何已经持有该节点引用、期待它是"分叉后半段"的调用方就会读错数据。

---

## 4. Constrained Decoding(`constrained_sampler.py`,L04)—— 采样前把非法 token 摁死

**是什么:**
```python
def legal_token_mask(self) -> List[bool]:
    """Return |V| bool list; True for tokens that keep us legal."""
    mask: List[bool] = []
    for tok in self.vocab:
        s = self.state
        ok = True
        for c in tok:
            s2 = self.fsm.advance(s, c)
            if s2 is None:
                ok = False
                break
            s = s2
        mask.append(ok)
    return mask
```
(`constrained_sampler.py:15-28`)

```python
def commit(self, token_id: int) -> bool:
    """Advance state with the picked token's chars; return False if invalid."""
    tok = self.vocab[token_id]
    s = self.state
    for c in tok:
        s2 = self.fsm.advance(s, c)
        if s2 is None:
            return False
        s = s2
    self.state = s
    return True
```
(`constrained_sampler.py:38-48`)

**一句话:** 在每一步采样之前,根据 grammar 当前状态算出一个"哪些 token 合法"的布尔掩码,把不合法位置的 logit 直接设成 `-inf`,让 softmax 永远采样不到非法 token——是"生成前预防",不是"生成后校验重试"。

**底层机制/为什么这样设计:** `legal_token_mask()` 对词表里的每个 token 都做一次"从当前状态出发、把这个 token 的每个字符依次喂给 FSM"的模拟,只要任意一个字符导致 `advance()` 返回 `None`(没有对应转移),这个 token 就被判定非法。这是一个 O(|V| × 平均 token 长度) 的操作,朴素地对整个词表重新走一遍——真实词表有 5 万+ token,这样每一步都要做几万次字符级模拟,这正是知识点 5 要引出的"预编译成查表"优化的动机(`compile_token_table` 把这个过程从"运行时现算"搬到"启动时算一次、运行时 O(1) 查")。`apply()` 用 `out = list(logits)` 先拷贝一份,不直接修改传入的 `logits`,是函数式风格——好处是调用方不用担心传进去的原始 logits 被意外污染,代价是每步都要拷贝一次可能几万维的向量,真实系统通常会换成原地写入的向量化操作(比如直接在 GPU 张量上 `masked_fill`)来省掉这次拷贝。`commit()` 是这个知识点里最值得注意的设计:它用一个**局部变量** `s` 做尝试性推进,只有 `for` 循环里的**所有**字符都成功转移(没有触发 `return False`),才会在最后一行把 `s` 写回 `self.state`——这是一次"要么完全提交、要么完全不生效"的事务式更新,如果传进来的 `token_id` 是非法 token,`commit()` 会在中途某个字符处返回 `False`,此时 `self.state` 一次都没有被修改过,调用方可以放心地认为"状态没有被破坏"。

**AI 研究场景:** 这套"采样前加 mask"的机制是今天几乎所有主流推理框架"结构化输出"(JSON mode、function calling、Anthropic 的 tool use)背后的通用原理——用语法约束从根本上保证输出合法,而不是寄希望于模型自己学会格式、生成完再用正则表达式校验/重试。对 agent/工具调用这类场景尤其关键:一次格式错误的 JSON 可能直接打断整条工具调用链路,重试不仅慢,还可能因为温度采样导致下一次仍然出错;而 mask 机制保证"不合法的路径从一开始就不存在",不依赖模型自身的格式遵循能力。

**可运行例子:**
```python
import sys
import math
sys.path.insert(0, "learning/sglang-radixattention/src")
from grammar_fsm import compile_literal, compile_digits_n, compile_concat
from constrained_sampler import ConstrainedSampler

fsm = compile_concat(compile_literal("v"), compile_digits_n(2))   # 语法：v\d{2}
vocab = ["a", "v", "1", "2", "12", "v12"]
sampler = ConstrainedSampler(fsm=fsm, vocab=vocab)

# 起始状态只有 "v" 和 "v12" 合法（都以强制字面量 'v' 开头）。
mask0 = sampler.legal_token_mask()
assert mask0 == [False, True, False, False, False, True]

# apply() 不能修改调用方传入的 logits（函数式风格）。
logits = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
original = list(logits)
masked = sampler.apply(logits)
assert logits == original
assert masked[vocab.index("a")] == -math.inf     # 非法 token -> -inf
assert masked[vocab.index("v")] == 2.0           # 合法 token -> logit 原样保留

# commit() 是事务式的：非法 token 不能部分推进 state。
state_before = sampler.state
assert sampler.commit(vocab.index("a")) is False   # "a" 在起始状态非法
assert sampler.state == state_before                # state 完全未被触碰

# 推进一个合法 token 后，mask 应随状态变化。
assert sampler.commit(vocab.index("v")) is True
mask1 = sampler.legal_token_mask()
assert mask1 != mask0
assert mask1[vocab.index("12")] is True             # 'v' 之后，两位数字合法
assert mask1[vocab.index("v")] is False             # 'v' 已经用过，不再合法

assert sampler.commit(vocab.index("12")) is True
assert sampler.at_accept() is True
```

实测(`.venv` 真跑):`mask0 = [False, True, False, False, False, True]`,精确对应"只有 `v` 和 `v12` 合法";`apply()` 前后 `logits` 原列表不变,`masked[vocab.index("a")]` 精确是 `-inf`,`masked[vocab.index("v")]` 保留原值 `2.0`;`commit(vocab.index("a"))` 返回 `False` 且 `sampler.state` 前后完全一致;推进合法的 `"v"` 后 `mask1 = [False, False, True, True, True, False]`,和 `mask0` 不同;`commit("12")` 后 `at_accept()` 返回 `True`。

**面试怎么问 + 追问链:**
- **Q:** "受约束解码为什么要在采样阶段加 mask,而不是生成完之后再校验格式?"—— 期望说出"生成后校验/重试没法保证一定能生成合法内容(尤其温度采样下),而且重试成本高;在采样前把非法 token 的 logit 设为 -inf,从根本上保证不会走到不合法的路径"。
- **追问 1:** "`legal_token_mask()` 这种实现方式,每一步的时间复杂度是多少?生产系统会怎么优化?"—— 期望说出"O(|V| × 平均 token 长度),对几万词表每步都要重新模拟一遍;生产系统(比如 xgrammar)会在启动时把 char-level FSM 预编译成 token-level 查表,运行时变成 O(1) 查表",为知识点 5 埋下伏笔。
- **追问 2(考察是否读过 commit 的具体实现):** "如果 `commit()` 传入一个非法 token,`self.state` 会不会被部分修改?"—— 期望说出"不会,`commit()` 用局部变量做尝试性推进,只有循环完整跑完所有字符都成功才会把结果写回 `self.state`,失败时是原子性的'完全不生效',不存在'推进了一半就失败'的中间状态"。
- **追问 3:** "`apply()` 每次都拷贝一份 logits 返回新列表,这在真实系统里是不是一个好的实现方式?"—— 期望说出"教学角度清晰、没有副作用,但真实系统 logits 往往是几万维的 GPU 张量,每步都拷贝一份代价不小,生产实现通常会做原地 masked_fill 而不是返回新对象",能看出候选人区分"教学清晰"和"生产高效"这两个不同目标。

**常见坑:** 以为 `apply()` 会原地修改传入的 `logits`——源码用 `out = list(logits)` 显式拷贝,不会;调用方如果误以为传入的原始列表也被修改了,可能在多处复用同一份 logits 时产生逻辑错误。另一个坑是以为 `commit()` 失败会导致内部状态"部分损坏"或需要额外回滚逻辑——源码的局部变量写法已经保证了原子性,不需要调用方自己做防御性拷贝或回滚。

---

## 5. Grammar FSM(`grammar_fsm.py`,L05)—— 把正则编译成状态转移表

**是什么:**
```python
DIGITS = set("0123456789")
WORD = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
ANY = None   # None means "any char"
```
(`grammar_fsm.py:13-15`)

```python
def compile_literal(literal: str) -> Fsm:
    """Compile a fixed literal string into a linear FSM."""
    n = len(literal)
    states = [State(i) for i in range(n + 1)]
    delta = {(i, literal[i]): i + 1 for i in range(n)}
    return Fsm(states=states, delta=delta, accept={n}, start=0)
```
(`grammar_fsm.py:54-59`)

```python
def compile_token_table(fsm: Fsm, vocab: List[str]) -> Dict[int, List[bool]]:
    """For each state, return a boolean mask of size |vocab|: True = accepted."""
    table: Dict[int, List[bool]] = {}
    for state in range(len(fsm.states)):
        mask = []
        for tok in vocab:
            s = state
            ok = True
            for c in tok:
                s2 = fsm.advance(s, c)
                if s2 is None:
                    ok = False
                    break
                s = s2
            mask.append(ok)
        table[state] = mask
    return table
```
(`grammar_fsm.py:99-115`)

**一句话:** 把"字面量/固定长度 `\d{n}`/两者拼接"这个很小的正则子集编译成一张"状态 × 字符 → 下一状态"的转移表(`delta`),再用 `compile_token_table` 把这张 char-level 的表"提升"成 token-level 的查表——但这第二步(也是 lecture L05 反复强调的核心性能优化)是一个孤立函数,全模块任何地方(包括它自己所在文件的 `__main__`)都没有调用过它。

**底层机制/为什么这样设计:** 模块 docstring 写的是"Supports a limited regex subset: literals, character classes (`\d`, `\w`, `.`), `+` and `*` quantifiers, alternation (`a|b`)",但本知识点用 `ast` 模块逐一核实过,实际只实现了 `compile_literal`(固定字面量)、`compile_digits_n`(**固定长度** `\d{n}`,不是无界的 `\d+`)和 `compile_concat`(拼接)三个构造函数——没有 `compile_word`/`compile_any`/`compile_star`/`compile_plus`/`compile_alt` 这类函数。`WORD`(对应 `\w`)和 `ANY`(对应 `.`,取值 `None`)这两个常量确实定义在模块顶部,但用 AST 检查"这两个名字有没有在任何地方以 `Load`(读取值)的方式被引用"后,答案是**零**——它们各自只在自己的赋值语句里出现过一次,此后再也没被任何函数用到过。也就是说,docstring 宣称的 5 种语法特性(`\w`/`.`/`+`/`*`/`a|b`)里,实际生效的是 0 种,只有"字面量"和"**固定**长度数字"这两个更基础的子集是真正实现的。这不是代码有 bug——`compile_literal`/`compile_digits_n`/`compile_concat` 三个函数本身运行完全正确(知识点 4/6 的例子都建立在它们之上),只是文档描述的能力边界比实际实现更宽,阅读源码时不能只看 docstring 就断定这个模块"支持"什么。`compile_token_table` 是另一层性质不同的发现:它不是"没实现",而是"实现了、语义正确、但从未被任何调用点用过"——本知识点手动调用它验证过语义(见下方例子),对给定 FSM 在每个 state 下算出 |vocab| 大小的合法性掩码,结果是对的;但它既没有被 `constrained_sampler.py` 用来加速 `legal_token_mask()`(那边是每步现算,知识点 4 已经指出这是 O(|V|) 的),也没有被 `jump_forward.py` 用到(知识点 6 会看到 jump-forward 走的是完全独立的 char-level 路径),甚至连它自己所在文件的 `__main__` demo 都没调用它。lecture L05 第 5 节强调的"预编译表让每步查询从慢变成 O(1)"这个核心卖点,在这份教学代码里只停留在"写出了这个函数"这一步,没有被接入任何实际使用它的调用链。

**AI 研究场景:** xgrammar 这类生产级约束解码引擎做的正是 `compile_token_table` 想表达的思路——启动时把 JSON schema 编译成 FSM、对每个 (state, token) 组合预算好是否合法,运行时直接查表,做到"每步 < 10 微秒"(lecture L05 第 5 节引用的数字)。L05 第 4 节给出的量级是 vocab=50k、200 state 时表大小约 1.25 MB/grammar——这是一次性的编译成本换取每一步推理时的常数时间,对于需要跑几十上百步 decode 的长输出,这个换算是明显划算的,这也是为什么真实系统里这一步几乎不会跳过,即便本教学仓库里对应的函数没有被接进主链路。

**可运行例子:**
```python
import ast
import sys
import inspect
sys.path.insert(0, "learning/sglang-radixattention/src")
import grammar_fsm as gm
from grammar_fsm import compile_literal, compile_digits_n, compile_concat, compile_token_table

# --- literal / \d{n} / concat：唯一真正实现的 3 个构造函数 ---
fsm = compile_concat(compile_literal('{"name":"'), compile_digits_n(4), compile_literal('"}'))
assert fsm.accepts('{"name":"1234"}')
assert not fsm.accepts('{"name":"abcd"}')

# --- 文档 vs 实现的差距 ---
doc = gm.__doc__
assert r"\w" in doc and "*" in doc and "a|b" in doc     # docstring 宣称支持这些

# WORD/ANY 常量存在（明显是为 \w / . 准备的）……
assert gm.WORD == set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
assert gm.ANY is None

# ……但 AST 级别核实：两者各自只被赋值过一次，从未在任何地方被当作值读取过。
src = inspect.getsource(gm)
tree = ast.parse(src)
loads = {"ANY": [], "WORD": []}
for node in ast.walk(tree):
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in loads:
        loads[node.id].append(node.lineno)
assert loads["ANY"] == [] and loads["WORD"] == []

# 也没有任何编译 \w / . / + / * / 交替(|) 的函数。
for missing in ("compile_word", "compile_any", "compile_star", "compile_plus", "compile_alt"):
    assert not hasattr(gm, missing)

# compile_digits_n(n) 是固定 n 位，docstring 暗示的无界 \d+ 并不存在。
assert compile_digits_n(4).accepts("1234")
assert not compile_digits_n(4).accepts("12345")   # \d+ 会接受，\d{4} 正确拒绝

# --- compile_token_table：存在、语义正确，但是零调用点的死代码 ---
call_sites = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name) and n.func.id == "compile_token_table"]
assert call_sites == []      # 连 grammar_fsm.py 自己的 __main__ 也没调用它

vocab = ['{"name":"', "12", "34", "ab", '"}']
table = compile_token_table(fsm, vocab)
assert len(table) == len(fsm.states)
assert table[0][vocab.index('{"name":"')] is True
assert table[0][vocab.index("ab")] is False
state_after_opening = len('{"name":"')
assert table[state_after_opening][vocab.index("12")] is True
assert table[state_after_opening][vocab.index("ab")] is False
```

实测(`.venv` 真跑):`fsm.accepts('{"name":"1234"}')` 为 `True`,`accepts('{"name":"abcd"}')` 为 `False`;`gm.__doc__` 确实包含 `\w`/`*`/`a|b` 三个关键词;AST 遍历确认 `loads["ANY"]` 和 `loads["WORD"]` 都是空列表(`[]`),即两个常量各自只在定义处出现一次,从未被读取;`hasattr` 逐一核实 `compile_word`/`compile_any`/`compile_star`/`compile_plus`/`compile_alt` 全部不存在;`compile_digits_n(4).accepts("12345")` 为 `False`(固定 4 位、多一位就拒绝,不是无界的 `+`);`call_sites` 精确是空列表,即整个 `src/` 目录没有任何一处 `compile_token_table(...)` 调用;手动调用后语义正确——`table[0]` 只对完整字面量开头 `'{"name":"'` 判 `True`,对 `"ab"` 判 `False`,`table[9]`(消费完 9 字符字面量之后)对两位数字 `"12"` 判 `True`、对字母 `"ab"` 判 `False`。

**面试怎么问 + 追问链:**
- **Q:** "`grammar_fsm.py` 把正则编译成什么样的数据结构?能表达哪些语法?"—— 期望说出"一张 `(state, char) -> next_state` 的转移表(`delta` 字典);但实际只实现了固定字面量和固定长度的 `\d{n}` 两种基础构造(加上拼接),不是完整的正则引擎"。
- **追问 1(考察是否只读了 docstring 就下结论):** "模块 docstring 说支持 `\w`、`.`、`+`、`*`、alternation,这些能力都能用吗?"—— 期望明确说"不能,这几项一个都没有对应的编译函数;`WORD`/`ANY` 这两个看起来是为 `\w`/`.` 准备的常量,经 AST 核实从未被任何代码读取过,属于文档超出实现范围的情况",能看出候选人有"读代码验证文档"的习惯而不是照单全收。
- **追问 2(深挖 lecture 强调的性能优化):** "lecture 里反复强调'预编译成 token-level 查表能把每步开销降到 O(1)',这份代码里这个优化真的生效了吗?"—— 期望说出"`compile_token_table` 这个函数确实实现了这个思路,语义也是对的,但在整个仓库里找不到任何调用它的地方——`constrained_sampler.py` 走的是运行时现算的路径(知识点 4),这个'理论上更快'的函数从未被接入实际使用链路",能区分"文档强调的优化点"和"代码里真正启用的优化点"。
- **追问 3:** "`compile_digits_n(4)` 和真正的正则 `\d{4}` 完全等价吗?"—— 期望说出"在'恰好 4 位数字'这一点上等价,但它不支持 `\d+`(不定长)、也不支持和其它字符类的组合;`compile_digits_n(n)` 的名字已经暗示了这是'恰好 n 位'的特化实现,不是通用的量词引擎"。

**常见坑:** 只读 docstring/README 就断定一个模块"支持"某个功能,不去核实具体调用是否真的存在对应实现——这是本知识点最核心的教训,`grammar_fsm.py` 的 docstring 和实际能力之间的落差如果不动手验证,很容易被当作"这个模块能处理任意正则"这种过度自信的结论。另一个坑是把"存在这个函数"等同于"这个函数在系统里发挥作用"——`compile_token_table` 是一个真实存在、语义正确的函数,但它是孤立的,不接入任何调用链,如果只看"这个模块里有没有实现某个优化"而不去追踪"这个优化有没有被用到",容易高估系统的实际性能特征。

---

## 6. Jump-Forward Decoding(`jump_forward.py`,L06)—— 唯一合法字符就不用 sample

**是什么:**
```python
def jump_forward(fsm: Fsm, state: int, max_lookahead: int = 64) -> Tuple[str, int]:
    """Greedily walk while exactly one character is legal.

    Returns (forced_string, new_state).  Empty string when no jump possible.
    """
    forced: List[str] = []
    cur = state
    for _ in range(max_lookahead):
        legal = fsm.legal_chars(cur)
        if len(legal) != 1:
            break
        ch = next(iter(legal))
        nxt = fsm.advance(cur, ch)
        if nxt is None:
            break
        forced.append(ch)
        cur = nxt
    return "".join(forced), cur
```
(`jump_forward.py:8-25`)

**一句话:** 当 grammar 在当前状态下只有唯一一个合法字符时,不需要 sample、不需要跑一次前向,直接把这个字符追加进输出,重复这个过程直到出现真正的分叉(合法字符数 > 1)才停下来,一次性跳过多个原本需要逐 token 前向的解码步骤。

**底层机制/为什么这样设计:** 循环体每一步先调用 `fsm.legal_chars(cur)` 拿到当前状态下所有合法字符的集合,`len(legal) != 1` 就 `break`——这是判断"是否还处于强制段"的核心条件:只要合法字符不止一个,就意味着接下来该由模型真正采样决定走哪条路,jump-forward 必须停手。`max_lookahead=64` 是一个安全阀,防止某个病态 grammar(比如一个几百字符的固定字面量)导致单次调用把 CPU/循环耗在一次性生成过长的强制串上——本知识点验证过,一个 100 字符的纯字面量在 `max_lookahead=64` 下会被正确截断在 64 字符处,而不是不受控制地跑完整个字符串。`legal_chars()`(定义在 `Fsm` 类里,不在这份文件中)的实现是对整个 `delta` 字典做一次线性扫描(`for (s, c), _ in self.delta.items(): if s == state: ...`),复杂度是 O(|delta|) 而不是 O(1)——这是教学版本的简化,和 lecture 里"xgrammar C++ 实现每步 < 10 微秒"的说法有明显差距,是有意为之的"讲清楚算法、不追求原始性能"的取舍。还有一处值得注意的架构事实:`jump_forward()` 完全通过 `Fsm.legal_chars`/`Fsm.advance` 这两个 **char 级别**的接口工作,不涉及知识点 5 提到的 `compile_token_table`(**token 级别**的预编译表)——这两条本该衔接起来的优化路径(char-level jump-forward 判断"接下来强制什么",token-level 查表判断"当前 token 合不合法")在这份代码里是完全独立、从未被接到一起的两套机制,吻合知识点 5 已经指出的"token-level 查表是孤立死代码"这一事实。

**AI 研究场景:** lecture L06 第 5 节给出的量级是:JSON schema 输出里 jump-forward 能省下约 40% 的解码步数(固定字段名 `"name":`、`"age":` 这类内容本身就是强制段),SQL 约 20%,自由 chat 几乎是 0%(几乎没有强制段);第 2 节引用的实测是"SGLang 在 JSON 任务上比 outlines 快 3-5 倍"。这条优化和知识点 2-3 的 radix tree 还有一层协同关系(L06 第 6 节):被跳过的强制字符序列本身也可以哈希后存进 radix 树,下次遇到同一个 JSON schema 直接命中,不需要重新走一遍 jump-forward 判断——lecture 把这个组合叫"JSON schema cache"。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/sglang-radixattention/src")
from grammar_fsm import compile_literal, compile_digits_n, compile_concat
from jump_forward import jump_forward

# 纯字面量：每个字符都是唯一合法项 -> 一次性跳过整段。
fsm = compile_literal('{"name":"')
forced, new_state = jump_forward(fsm, state=0)
assert forced == '{"name":"'
assert len(forced) == 9
assert new_state in fsm.accept

# 真正的分叉点：强制字面量之后是 \d{4}，10 个数字都合法 -> jump-forward 必须停手。
fsm2 = compile_concat(compile_literal('{"name":"'), compile_digits_n(4), compile_literal('"}'))
forced2, state2 = jump_forward(fsm2, state=0)
assert forced2 == '{"name":"'          # 强制跳过字面量开头
assert state2 == 9                     # 精确停在分叉点（与 runbook.yaml 文档记录一致）
assert len(fsm2.legal_chars(state2)) == 10   # 10 个数字都合法，真正需要 sample 的地方

# max_lookahead 安全阀：超长字面量不能被一次性无限制地全部强制推进。
long_literal = "x" * 100
fsm3 = compile_literal(long_literal)
forced3, state3 = jump_forward(fsm3, state=0, max_lookahead=64)
assert len(forced3) == 64              # 被截断在安全阀处，不是 100
assert state3 not in fsm3.accept       # 尚未到达 accept，正确地停在半途

# jump_forward 完全工作在 char 级别（Fsm.legal_chars/advance），和知识点 5 的
# token 级别 compile_token_table 是两条从未被接到一起的独立路径。
import inspect
src = inspect.getsource(jump_forward)
assert "compile_token_table" not in src
assert "legal_chars" in src and "advance" in src
```

实测(`.venv` 真跑):`forced == '{"name":"'`,长度精确 `9`,`new_state` 落在 `fsm.accept` 里;`fsm2` 上 `forced2` 同样是这 9 个字符,`state2=9`,此时 `legal_chars(9)` 返回的合法字符集合大小精确是 `10`(0-9 十个数字,真正需要模型采样的地方);100 字符字面量在 `max_lookahead=64` 下 `forced3` 长度精确截断在 `64`,`state3=64` 没有落进 `fsm3.accept`(该 FSM 唯一的 accept 状态是 100);源码检查确认 `jump_forward` 函数体里既不出现 `compile_token_table`,也确实用到了 `legal_chars`/`advance` 这两个 char 级别接口。

**面试怎么问 + 追问链:**
- **Q:** "Jump-Forward Decoding 具体是怎么省解码步数的?"—— 期望说出"每一步先看 grammar 当前状态下有几个合法字符,如果只有一个,就不需要模型采样,直接把这个字符追加进输出,重复直到出现真正的多选分叉才停,省掉的是那些'其实没有选择余地'的前向计算"。
- **追问 1(考察是否理解停止条件):** "什么情况下 jump-forward 会立刻停手,一步都不跳?"—— 期望说出"当前状态下合法字符数不是 1(可能是 0,代表已经出错;也可能是多个,代表真正需要模型决策),`legal_chars` 集合大小不等于 1 就直接 break",并能举出 `\d{4}` 这种"10 个数字都合法"的分叉点例子。
- **追问 2:** "`max_lookahead=64` 这个参数是干什么用的,如果没有它会怎样?"—— 期望说出"是安全阀,防止一个异常长的强制字面量段导致单次调用无限制地把整段都跳过去(比如影响批处理调度或者一次性占用过多计算);本知识点验证过 100 字符的纯字面量在这个参数下会被正确截断在 64 字符"。
- **追问 3(深挖架构完整性):** "jump-forward 用的合法性判断和知识点 5 提到的 token-level 预编译表是同一套机制吗?"—— 期望说出"不是,`jump_forward` 走的是 char 级别的 `Fsm.legal_chars`/`advance`,`compile_token_table` 是 token 级别的查表,这两个在这份教学代码里是完全独立、没有被接到一起的两条路径——知识点 5 已经验证过 `compile_token_table` 本身也是零调用点的死代码",能看出候选人在梳理这几个知识点之间关联时是系统性的,不是孤立记忆每个函数。

**常见坑:** 把 jump-forward 理解成"跳过若干个 token 不生成",实际上它跳过的是"若干个字符不需要经过一次模型前向去决定",这个字符段最终依然会被写进输出、依然需要被计入 KV cache(lecture L06 第 4 节提到"强制 token 没经过 forward,KV 没生成,需要批量补 forward 一次"),不是"凭空省略了这部分内容",而是"省略了为了决定这部分内容需要做的计算"。另一个坑是以为这个算法的合法字符检测是 O(1) 的——`Fsm.legal_chars()` 实际是对整个 `delta` 字典的线性扫描,真实系统(xgrammar)才是通过预编译表做到常数时间,这份教学代码在性能特征上和生产实现有明显差距,不能把两者的复杂度混为一谈。

---

## 7. Frontend DSL(`frontend_lang.py`,L07)—— `+=` 拼出来的 agent 程序

**是什么:**
```python
def __iadd__(self, other) -> "Stream":
    if isinstance(other, str):
        self.prompt += other
    elif isinstance(other, Gen):
        text = self.generator(self.prompt, other.name, other.max_tokens)
        self.vars[other.name] = text
        self.prompt += text
        self.n_forwards += 1
    elif isinstance(other, Select):
        best = max(other.choices, key=lambda c: len(c))   # mock judge
        self.vars[other.name] = best
        self.prompt += best
        self.n_forwards += 1
    else:
        raise TypeError(f"unsupported operand: {type(other)}")
    return self

def fork(self, k: int) -> List["Stream"]:
    return [
        Stream(prompt=self.prompt, vars=dict(self.vars), generator=self.generator, parent=self)
        for _ in range(k)
    ]
```
(`frontend_lang.py:27-48`)

```python
def function(fn):
    """Decorator analogous to @sgl.function; creates a Stream and runs fn."""
    def wrapper(*args, generator=None, **kwargs):
        s = Stream(generator=generator) if generator else Stream()
        fn(s, *args, **kwargs)
        return s
    return wrapper
```
(`frontend_lang.py:64-70`)

**一句话:** `Stream` 把 prompt 文本和一个 `vars` 字典绑在一起,重载 `+=` 运算符区分"追加字面文本"/"调用生成器"/"调用 mock judge 做选择"三种情况;`fork(k)` 用"prompt 字符串共享、`vars` 字典深拷贝"的方式模拟"从当前状态分出 k 条独立分支,不用重新 prefill 父前缀"。

**底层机制/为什么这样设计:** `__iadd__` 用 `isinstance` 判断右操作数类型来决定行为——这正是"像写普通 Python 代码一样拼 prompt"这个体验的来源:字符串字面量直接追加,`Gen` 实例触发一次生成器调用并记一次 `n_forwards`,`Select` 实例触发"选择"并同样记一次 forward,任何其它类型显式抛 `TypeError`(不是静默忽略)。`fork()` 的拷贝策略经过深思:`self.prompt` 是不可变的 Python 字符串,多个子分支直接共享同一个对象引用是安全的(不存在被某个分支意外修改、影响到其它分支的风险);但 `self.vars` 是可变字典,如果不显式 `dict(self.vars)` 拷贝一份,某个分支往自己的 `vars` 里写入新键值时会直接污染父 Stream 和所有兄弟分支共享的同一个字典对象——本知识点验证过,拷贝之后各分支的 `vars` 确实互不干扰。

这里有两处动手验证后才发现的、值得单独指出的事实。第一,`Select` 的"mock judge"极其朴素——只是 `max(choices, key=len)`,选最长的字符串,不做任何语义判断;更关键的是,如果用这个文件自带的 `default_generator`(纯粹是 `prompt`/`name`/长度 的函数),`fork(k)` 出来的 k 个分支因为起点(父 prompt)完全相同,分别调用同一个 `Gen` 时会生成**逐字符相同**的字符串——本知识点验证过 3 路 fork 全部产出 `'<gen:thought@18>'`,一模一样。这意味着 `Select` 的"挑选"逻辑在这个模块自带的默认演示里始终是在给一组完全相同的选项做无意义的平局判定,只有换成一个"输出真正依赖分支本身、不只依赖 prompt 长度"的生成器,选择才有实际意义(知识点 8 会现场证明这一点)。第二,`function` 装饰器**没有使用 `functools.wraps`**——`wrapper` 是在 `frontend_lang.py` 内部定义的普通闭包函数,任何用 `@function` 装饰的函数(比如知识点 8 的 `react_loop`、知识点 9 的 `react_agent`)在装饰之后,它的 `__name__` 会变成字面量 `"wrapper"`,`__module__` 会变成 `"frontend_lang"`——不是原来定义它的模块。这意味着任何依赖函数自省的工具(调试器、日志打印 `func.__name__`、自动生成的 API 文档)在这些被装饰函数身上都会显示出错误的名字和来源模块,这是全部使用 `@function` 的函数共有的副作用,不是某一处的孤立问题。此外,lecture L07 第 4 节写"fork 时……KV cache radix 节点 refcount++",但 `frontend_lang.py` 全文没有任何一处 import 或引用 `radix_tree`/`RadixTree`——`fork()` 纯粹是 Python 对象层面的拷贝,不涉及任何真实的 radix 树操作,这条"KV 层面的收益"只在知识点 9 的 `agent_server.py` 里以完全独立、post-hoc 的方式被间接体现,阅读 lecture 原文时不能理解成"这份 mock 代码内部真的在维护 radix 树引用计数"。

**AI 研究场景:** 这套 `+=`/`fork`/`gen`/`select` 的组合正是今天"LLM 编程"框架(真实 SGLang、Guidance、LMQL 等)共同的核心抽象——把多步/分支的 LLM 程序写成带状态变量的普通 Python 代码,让 runtime 在背后自动识别共享前缀并调度,而不需要开发者手工拼装每一次 API 调用的完整 message 数组。lecture L07 第 5 节点明了这类 DSL 和 LangChain/LlamaIndex 这类"编排框架"的区别:后者是在应用层做链式/图式编排,SGLang 是 runtime 级别原生支持,没有中间层的序列化开销。

**可运行例子:**
```python
import sys
import inspect
sys.path.insert(0, "learning/sglang-radixattention/src")
from frontend_lang import Stream, Gen, Select, function
import frontend_lang as fl

# += 类型分派：str 直接追加，Gen 调生成器并记一次 forward，Select 同理。
s = Stream()
s += "Q: 1+1?\n"
assert s.prompt == "Q: 1+1?\n" and s.n_forwards == 0
s += Gen("answer", max_tokens=5)
assert "answer" in s.vars and s.n_forwards == 1

# 不支持的操作数类型 -> 显式 TypeError，不是静默失败。
try:
    s += 42
    raise AssertionError("expected TypeError")
except TypeError:
    pass

# fork()：prompt（不可变 str）安全共享；vars（可变 dict）必须深拷贝，否则互相污染。
parent = Stream(prompt="shared ")
forks = parent.fork(3)
forks[0].vars["x"] = "A"
assert "x" not in forks[1].vars and "x" not in parent.vars
forks[0] += "branch0"
assert parent.prompt == "shared "          # 父 Stream 不受子分支修改影响
assert forks[1].prompt == "shared "        # 兄弟分支互不影响

# Select 的 "mock judge" 极其朴素：只挑最长的字符串。
sel = Stream()
sel += Select("choice", choices=["short", "much longer answer", "mid"])
assert sel.vars["choice"] == "much longer answer"

# 用默认生成器时，从同一个父 prompt fork 出来的 k 路分支会生成完全相同的文本
# （default_generator 只是 prompt 长度 + name 的纯函数）——Select 在这里挑的是平局。
s2 = Stream()
s2 += "Q: test?\nThought: "
demo_forks = s2.fork(3)
for f in demo_forks:
    f += Gen("thought", max_tokens=30)
thoughts = [f.vars["thought"] for f in demo_forks]
assert len(set(thoughts)) == 1

# fork() 纯粹是 Python 对象拷贝，完全不触碰 RadixTree/refcount。
src = inspect.getsource(fl)
assert "radix" not in src.lower() and "RadixTree" not in src

# function 装饰器没有用 functools.wraps —— 被装饰函数在自省层面"失忆"。
@function
def tiny(s, x):
    s += f"in={x}"
    s += Gen("out")
out = tiny("hi")
assert isinstance(out, Stream) and "in=hi" in out.prompt and "out" in out.vars
assert tiny.__name__ == "wrapper" and tiny.__module__ == "frontend_lang"
```

实测(`.venv` 真跑):`s.n_forwards` 从 `0` 变成 `1`;`s += 42` 精确抛出 `TypeError`;`fork(3)` 之后修改 `forks[0].vars` 不影响 `forks[1]`/`parent`,`forks[0] += "branch0"` 之后 `parent.prompt` 和 `forks[1].prompt` 都仍是 `"shared "`;`Select` 在三个候选里精确选中最长的 `"much longer answer"`;用默认生成器 fork 出的 3 路 `thought` 全部是完全相同的字符串 `'<gen:thought@18>'`,`len(set(thoughts))==1`;`inspect.getsource(fl)` 里既不出现 `"radix"`(不分大小写)也不出现 `"RadixTree"`;被 `@function` 装饰后,`tiny.__name__` 精确是 `"wrapper"`,`tiny.__module__` 精确是 `"frontend_lang"`——不是 `"tiny"`/调用者所在模块。

**面试怎么问 + 追问链:**
- **Q:** "`Stream.fork(k)` 具体做了什么,为什么这样设计能对应'不用重新 prefill 父前缀'这个说法?"—— 期望说出"浅拷贝 prompt(不可变字符串,天然安全共享)+ 深拷贝 vars(可变字典,必须显式拷贝避免分支间互相污染),这个操作本身在 Python 层面几乎零成本,用来模拟真实系统里'多个分支复用同一段父前缀的 KV,不需要重新计算'这件事"。
- **追问 1(核心陷阱,考察是否真的读过 lecture 原文和源码两边):** "lecture 里说 fork 时 'KV cache radix 节点 refcount++',这段逻辑在 `frontend_lang.py` 里能找到吗?"—— 期望明确说"找不到,这份文件完全没有 import 或引用 `RadixTree`,`fork()` 只是纯 Python 对象拷贝;lecture 描述的是'一个真实集成了 radix tree 的系统应该有的行为',这份 mock 代码本身没有做这层集成,真正把 Stream 输出接进 RadixTree 的是知识点 9 的 `agent_server.py`,而且是以完全独立的 post-hoc 方式"。
- **追问 2:** "如果换一个生成器(不再是默认的 `default_generator`),`Select` 的'挑最长字符串'这个 mock judge 有意义吗?"—— 期望说出"取决于生成器的输出是否真的依赖分支身份;用默认生成器时,所有从同一父 prompt fork 出来的分支会生成完全相同的字符串,'挑最长'只是在几个相同选项里做无意义的平局判定,只有生成器输出真正跟着分支变化,这个选择才有实际意义"。
- **追问 3:** "被 `@function` 装饰之后的函数,`__name__` 会变成什么?这在实际调试中会造成什么问题?"—— 期望说出"变成 `'wrapper'`,因为装饰器没有用 `functools.wraps` 保留原函数的元信息;如果生产代码依赖 `func.__name__` 做日志记录、性能分析打点、自动生成文档,所有被这个装饰器包过的函数都会在这些地方显示成同一个名字'wrapper',定位问题时会很困扰"。

**常见坑:** 望文生义地认为 lecture 描述的每一句机制(尤其"KV cache radix 节点 refcount++"这类具体到实现细节的描述)在对应的教学代码里都能找到逐行对应的实现——`frontend_lang.py` 就是一个反例,它只覆盖了 DSL 层面的行为,radix 层面的真正集成要去知识点 9 的 `agent_server.py` 里找。另一个坑是被 `@function` 装饰后用 `func.__name__`/`func.__module__` 做判断逻辑(比如某些框架会靠函数名做路由或缓存 key)——这份实现下所有被装饰函数都会撞到同一个 `"wrapper"`/`"frontend_lang"`,如果真的依赖这两个属性做区分,会直接出错。

---

## 8. Agent 模式(`agent_patterns.py`,L08)—— ReAct / ToT / Self-Consistency

**是什么:**
```python
@function
def react_loop(s: Stream, question: str, max_steps: int = 4) -> None:
    s += f"Question: {question}\n"
    for step in range(max_steps):
        s += f"Thought {step}: "
        s += Gen(f"t{step}", max_tokens=20)
        s += f"\nAction {step}: "
        s += Gen(f"a{step}", max_tokens=10)
        obs = mock_tool(s.vars[f"a{step}"])
        s += f"\nObservation: {obs}\n"
        if step >= 1 and "Final" in s.vars[f"t{step}"]:
            break
```
(`agent_patterns.py:17-28`)

```python
def tree_of_thought(question: str, k: int = 3) -> str:
    """Run k thought-branches in parallel; pick the longest as the answer."""
    s = Stream()
    s += f"Q: {question}\nThought: "
    forks = s.fork(k)
    for f in forks:
        f += Gen("thought", max_tokens=30)
    s += Select("choice", choices=[f.vars["thought"] for f in forks])
    s += "\nAnswer: "
    s += Gen("answer")
    return s.prompt
```
(`agent_patterns.py:31-41`)

**一句话:** 这个文件里"典型"agent 模式实际只有 3 种独立函数实现——ReAct 是单流线性累积(不 fork,每步 Thought/Action/Observation 直接拼进同一个 Stream)、ToT 是 fork 后选择(fork k 路独立想、从 k 个结果里选一个继续)、Self-Consistency 是 fork 后投票(fork k 路独立采样答案、多数表决)——但 lecture L08 标题写的是"5 种典型",实际"Multi-Step Tool Use"和"Conversational Agent"这两种在本文件里没有独立对应的函数。

**底层机制/为什么这样设计:** `react_loop` 的提前终止条件是 `step >= 1 and "Final" in s.vars[f"t{step}"]`——`step >= 1` 这个防御性条件确保不会在第 0 步就因为某个巧合的字符串触发终止,必须至少走完一轮完整的 Thought/Action/Observation 才允许提前结束。但本知识点验证过,这份文件自带的 `__main__` demo 用的是 `frontend_lang.default_generator`(纯粹是 `prompt` 长度和 `name` 的函数),这个生成器的输出格式是 `<gen:{name}@{len}>`,永远不可能产生子串 `"Final"`——也就是说,`react_loop` 自带 demo 跑起来时,这条提前终止分支从未被真正触发过,`max_steps=3` 时永远是 `2×3=6` 次完整的 Gen 调用,不会提前退出。要看到这条分支生效,必须换成一个真正会输出"Final"的生成器,这正是知识点 9 `agent_server.py::mock_generator` 做的事(它对 `idx>=2` 的 thought 显式返回 `"Final Answer: 42"`)。`tree_of_thought`/`self_consistency` 也有类似的"默认演示掩盖了真实机制"的情况:知识点 7 已经验证过,`default_generator` 在同一个父 prompt 下对所有 fork 分支产出完全相同的字符串,这意味着这个文件自带 demo 里 `Select`(挑最长)和 `Counter`(多数投票)两种"决策"逻辑,面对的都是一组彼此相同的候选项,是在对平局做选择,不是在真正比较有实质差异的候选——只有换上一个"输出真正随分支变化"的生成器,这两种模式的核心价值(多路探索、多数表决)才会真正体现出来。

**AI 研究场景:** 三种模式的共性(lecture L08 第 6 节点明)是"share prefix + branch"——父 prompt 在所有子任务里被重复使用(share),子任务各自独立 decode(branch),这正是 RadixAttention + fork 联合设计要服务的目标:ReAct 不 fork,靠 Stream 的连续累积让每一轮都直接在前一轮的 KV 基础上 append,不重新 prefill 历史(lecture 引用"比裸 OpenAI API 快 5-10 倍");ToT/Self-Consistency 用 fork 显式分支,让多条独立路径共享同一段父前缀。这也是为什么"Multi-Step Tool Use"和"Conversational Agent"这两种在 lecture 里被单独列出、却没有独立代码的模式,本质上分别是 ReAct 循环本身(工具调用已经内嵌在 `mock_tool(s.vars[f"a{step}"])` 里)和"更长的 ReAct 累积"(多轮历史全部拼进同一个 Stream)的变体,不需要额外的实现来演示。

**可运行例子:**
```python
import sys
import inspect
sys.path.insert(0, "learning/sglang-radixattention/src")
import agent_patterns as ap
from agent_patterns import react_loop, tree_of_thought, self_consistency, mock_tool
from frontend_lang import Stream, Gen

# L08 标题写"5 种典型"agent 模式，但这个文件只有 3 个独立函数实现。
assert hasattr(ap, "react_loop") and hasattr(ap, "tree_of_thought") and hasattr(ap, "self_consistency")
assert not hasattr(ap, "multi_step_tool_use")
assert not hasattr(ap, "conversational_agent")

# react_loop 的提前退出条件要求 step >= 1（防御性：不轻信第 0 步的巧合字符串）。
raw_src = inspect.getsource(ap)
assert "step >= 1" in raw_src and '"Final"' in raw_src

# ……但文件自带的 default_generator 永远不产生 "Final"，这条分支在自带 demo 里从未触发。
out = react_loop("hi", max_steps=3)
assert out.n_forwards == 2 * 3          # 每步都是 thought+action，从未提前退出
assert "Final" not in out.prompt

# tree_of_thought 在默认生成器下能正常跑完（但知识点 7 已证明各分支文本其实相同）。
text = tree_of_thought("Q?", k=4)
assert "Answer:" in text

# 换一个真正随分支变化的生成器，Self-Consistency 的多数表决才有实质意义。
def diverging_generator(prompt, name, max_tokens=16):
    diverging_generator.n += 1
    return f"ans-{diverging_generator.n % 3}"
diverging_generator.n = 0

s = Stream(generator=diverging_generator)
s += "Q: vote?\nA: "
samples = s.fork(6)
for sample in samples:
    sample += Gen("ans", max_tokens=10)
from collections import Counter
votes = Counter(sample.vars["ans"] for sample in samples)
assert len(votes) > 1                    # 分支真正产生了不同结果
majority = votes.most_common(1)[0][0]

# react_loop 的工具调用是真实接到一个可调用对象上的，不是占位符。
assert mock_tool("please search this") == "result: 42"
assert mock_tool("calc 6*7") == "result: 3.14"
assert mock_tool("unrelated") == "tool: ok"

# 另一处（承接知识点 7）：@function 装饰的 react_loop 同样丢失了自身的 __name__/__module__。
assert react_loop.__name__ == "wrapper"
assert react_loop.__module__ == "frontend_lang"
```

实测(`.venv` 真跑):`hasattr` 逐一核实只有 `react_loop`/`tree_of_thought`/`self_consistency` 三个模式函数,`multi_step_tool_use`/`conversational_agent` 均不存在;源码里确认 `"step >= 1"` 和 `'"Final"'` 同时出现;`react_loop("hi", max_steps=3)` 的 `n_forwards` 精确是 `6`(`2×3`,从未提前退出),`"Final"` 不出现在最终 `prompt` 里;`tree_of_thought("Q?", k=4)` 正常产出含 `"Answer:"` 的文本;换上 `diverging_generator` 后,6 路分支产生 `{'ans-1': 2, 'ans-2': 2, 'ans-0': 2}` 这样真正三路不同的投票分布(`len(votes)==3 > 1`),`most_common(1)` 给出确定性的多数结果;`mock_tool` 对三类关键词分别返回 `"result: 42"`/`"result: 3.14"`/`"tool: ok"`;`react_loop.__name__=="wrapper"`、`__module__=="frontend_lang"`,和知识点 7 发现的装饰器问题完全一致。

**面试怎么问 + 追问链:**
- **Q:** "ReAct、Tree-of-Thought、Self-Consistency 这三种 agent 模式,在 Stream 原语层面的核心区别是什么?"—— 期望说出"ReAct 不 fork,单流线性累积,每一轮都在同一个 Stream 上 append;ToT 是 fork k 路独立想、用 Select 选一路继续;Self-Consistency 是 fork k 路独立采样、用多数表决决定最终答案"。
- **追问 1(考察是否真的动手验证过,不是只读了 lecture 标题):** "lecture 标题说这是'5 种典型模式',这个文件里真的有 5 种独立实现吗?"—— 期望明确说"没有,只有 3 个函数(`react_loop`/`tree_of_thought`/`self_consistency`);'Multi-Step Tool Use'本质上就是 ReAct 循环里已经在做的工具调用,'Conversational Agent'在这个文件里没有独立代码,是概念性的扩展描述"。
- **追问 2(深挖默认演示的局限):** "用这个文件自带的默认生成器跑 Self-Consistency,多数投票逻辑真的在起作用吗?"—— 期望说出"不完全,默认生成器是 prompt 长度和 name 的纯函数,同一父 prompt fork 出来的所有分支会生成完全相同的字符串,'多数投票'面对的是一组相同选项,不是真正有分歧的候选;要看到投票逻辑真正生效,需要换一个输出会随分支变化的生成器",并能现场说出怎么验证(比如引入一个基于调用次数变化的生成器)。
- **追问 3:** "为什么 `react_loop` 的提前退出要求 `step >= 1`,而不是任何一步只要出现'Final'就退出?"—— 期望说出"防御性设计,避免第 0 步偶然包含'Final'这个子串就被误判为终止,至少要走完一轮完整交互才认可这个信号是真的终止意图"。

**常见坑:** 看到 lecture 标题写"5 种"就默认这个模块有 5 个对应的独立实现,答不上"另外两种在哪"这种追问——正确的做法是打开源码核实,这里只有 3 个函数,另外两种是概念性的模式描述,不是遗漏。另一个坑是跑一遍这个文件自带的 `__main__` demo,看到 ReAct/ToT 都正常产出了看起来合理的文本,就误以为"提前终止逻辑"和"多数投票/选择逻辑"都得到了充分验证——默认生成器的输出模式(纯粹依赖 prompt 长度)让这两类逻辑几乎不会被真正触发到有意义的分支,必须像本知识点这样换一个内容随分支变化的生成器,才能看到这些机制真正的效果。

---

## 9. Capstone:32 并发 Agent Server + vs vLLM 5 场景横评(`agent_server.py` + `sglang_compare.py`,L09+L10+L11)—— radix 命中率 91.7% 从哪来

**是什么:**
```python
SYSTEM_PROMPT = (
    "You are a helpful agent. Use tools when needed.\n"
    "Available tools: search, calc, weather, python.\n"
    "Always end with 'Final Answer: <answer>'.\n"
) * 25     # blow up to ~2000 token equivalent


def tokenize(text: str) -> List[int]:
    """Hash chars in 4-char groups as fake tokens (avoids real tokenizer dep)."""
    out: List[int] = []
    for i in range(0, len(text), 4):
        chunk = text[i : i + 4]
        out.append(hash(chunk) & 0xFFFFF)
    return out
```
(`agent_server.py:25-38`)

```python
def run_server(n_agents: int = 32, max_steps: int = 3) -> AgentMetrics:
    tree = RadixTree(cap=1_000_000)
    metrics = AgentMetrics(n_agents=n_agents)
    ...
    for q in queries:
        s = react_agent(q, max_steps, generator=mock_generator)
        # post-hoc: insert final prompt into radix to measure share
        tokens = tokenize(s.prompt)
        prev_hits = tree.hits
        leaf, matched = tree.insert(tokens)
        ...
        metrics.radix_hits += matched
        metrics.radix_misses += len(tokens) - matched
    ...
    return metrics
```
(`agent_server.py:92-114`,`...` 处省略了工具调用统计)

**一句话:** capstone 把 `radix_tree` + `frontend_lang` + `react_agent`(基于知识点 8 的 `react_loop` 改写)拼成"32 个并发 ReAct agent 在进程内跑一遍、每个 agent 生成完之后把完整最终 prompt 事后 tokenize 并插入同一棵 `RadixTree`、统计整体前缀命中率"的一次性模拟,量化"32 个 agent 共享同一段长 system prompt 时,radix tree 能省下多少 prefill"这件事;`sglang_compare.py`(L10)是这次横评的另一半——不测真实吞吐,只对"prefill token 数量"这一个指标建了一个诚实标注的合成成本模型。

**底层机制/为什么这样设计:**

*L09(Zero-Overhead Batch)没有对应源文件*——README 概览表里 L09 的代码列是"—",三招(CUDA Graph + bucketing、调度循环改写成 C++、prefill-decode 默认合批)和引用的 A100 实测表(vLLM 每 iter 12ms/17% 调度开销 vs SGLang 8ms/3%)都来自 lecture 文本,本仓库没有可运行代码复现这组数字,如实标注、不假装验证过,处理方式和 01 号文件对 CUDA Graphs 知识点的态度一致。

*`tokenize()` 是一个假分词器*:把文本按 4 字符一组切开,用 `hash(chunk) & 0xFFFFF` 当 token id,规避了引入真实 tokenizer 依赖。这里天然存在哈希碰撞的风险(不同的 4 字符片段哈希到同一个 20-bit 值),如果发生碰撞,`RadixTree` 会把两个本不相同的片段误判成相同 token,人为拉高命中率——本知识点专门检查过这次实际跑的 32-agent 场景:所有 agent 最终 prompt 里出现的 4 字符片段一共有 180 个互不相同的片段,精确映射到 180 个互不相同的 id,零碰撞,说明这次测出的命中率不是哈希碰撞的假象。另外,`SYSTEM_PROMPT` 用字符串 `* 25` 重复来"膨胀"到注释所说的"~2000 token equivalent"——但实际用这个文件自己的 `tokenize()` 一算,`SYSTEM_PROMPT` 只产生 863 个假 token(`len(SYSTEM_PROMPT)=3450` 字符,`3450 // 4 ≈ 863`),只有注释宣称的"~2000"的 43% 左右,是这次撰写过程中发现的又一处"注释和实际不完全对得上"的情况,不影响命中率结论,但如实记录。

*命中率统计是"post-hoc"(事后)的*:`run_server()` 对每个 agent 是先完整跑完 `react_agent(...)` 拿到最终 `Stream`(所有 Thought/Action/Observation 都已生成完毕),再把**整段最终 prompt** 一次性 `tokenize()`、一次性 `tree.insert()`。这和真实 SGLang 运行时"边生成边维护 radix 树、每一步 decode 时刻都能立刻复用刚才已经在树里的前缀"的在线做法不同——这里更接近"如果把这 32 段对话按顺序整体插入同一棵树,会有多少 token 重合"这样一个事后的统计工具,不直接展示"生成过程中随时间推移实时省下的 prefill"这件事。README/`runbook.yaml` 也用"进程内并发模拟""不是真网络服务"这类措辞明确了这一点。

*`forwards_per_agent=5.0` 是确定性结果,不是统计平均*:`mock_generator` 对名字形如 `t{idx}` 的 thought,在 `idx>=2` 时固定返回 `"Final Answer: 42"`,`idx<2` 时返回一句非终止文本。按 `react_agent` 的循环体,这意味着每个 agent 精确产生 thought0(非终止)→action0→thought1(非终止)→action1→thought2(终止,触发 break,不再有 action2)共 5 次 `Gen` 调用——32 个 agent 全部落在同一条确定性路径上,`forwards_per_agent` 精确是 `5.0`,不存在任何方差。

*`sglang_compare.py` 的"两套数字"不能混用*:lecture L10 引用的是"经验数字"表(A100/Qwen-7B 风格的 tok/s 吞吐,比如 `json_schema` 场景 SGLang +189%、`react_5step` +200%),这组数字来自 lecture 文本,本仓库没有 GPU 环境复现;`sglang_compare.py` 自己的合成成本模型只统计"prefill token 数量"这一项,对 `json_schema`/`react_5step`/`shared_system_chat`/`long_prompt_short_out` 四个场景精确算出 `+0.0%`(唯一非零的是 `tot_8way` 的 `+83.3%`)。这不是两套数字互相矛盾,而是它们在衡量完全不同的东西:成本模型完全没有对"xgrammar 约束解码在 decode 阶段更快""agent 多轮之间跨 step 复用 KV"这类优势建模,`other_advantage_not_modeled` 字段对每个场景都老老实实用文字标注了"这里还有一块没算进去",不是拿一个数字硬充当另一个数字的替代品——README 也记录了这次修复:此前版本的 `cost_vllm`/`cost_sglang` 两个函数公式实际上完全相同(两列 cost 恒相等),却另配了一个和这两列毫无关系的硬编码估计值显示"+83.3%",逻辑上自相矛盾,已被修正为"展示的百分比必须直接从两列 cost 反推得到,不能凭空硬编码"。

**AI 研究场景:** 32 个 agent 共享同一段基础 system prompt + 工具说明、只有 query 不同,这正是多租户 agent 托管平台的真实缩影——一个平台上服务的所有客户共用同一套基础设定,不同的只是各自的问题,并发数越高、这段共享前缀的复用收益占比通常越可观(本知识点是 32,真实生产环境可能是几千并发)。这个知识点也直接呼应 05 号文件(distributed-inference)"Prefix-Aware Routing"知识点提到的"单机树 vs 多副本路由"的分野——本文的 radix tree 是单机内的共享方案,当请求量大到需要多台机器时,如何在多个副本之间路由请求以尽量维持同一段前缀落在同一台机器上,是下一层需要解决的问题。

**可运行例子:**
```python
import sys
import inspect
sys.path.insert(0, "learning/sglang-radixattention/src")
from agent_server import run_server, metrics_json, SYSTEM_PROMPT, react_agent, mock_generator, tokenize
from sglang_compare import run_all

# --- Capstone：32 并发 ReAct agent，共享一段长 SYSTEM_PROMPT ---
m = run_server(n_agents=32, max_steps=3)
j = metrics_json(m)
assert j["n_agents"] == 32
assert j["forwards_per_agent"] == 5.0     # mock_generator 决定的确定性路径，非统计平均
assert j["radix_hit_rate"] >= 0.70        # 仓库自己的测试阈值（tests/test_agent_server.py）
assert j["tool_calls"] == {"search": 32, "calc": 32}

# run_server 是 post-hoc 统计：先跑完整段对话，再一次性 tokenize + insert。
src = inspect.getsource(run_server)
assert "tree.insert(tokens)" in src
assert src.index("react_agent(") < src.index("tree.insert(tokens)")

# tokenize() 是假分词器（hash & 0xFFFFF）——核实这次 32-agent 场景没有哈希碰撞，
# 91.7% 不是碰撞制造出来的假象。
chunk_to_id, seen_ids, collisions = {}, {}, 0
for i in range(32):
    text = react_agent(f"What is {i} * 7?", 3, generator=mock_generator).prompt
    for k in range(0, len(text), 4):
        chunk = text[k : k + 4]
        tid = hash(chunk) & 0xFFFFF
        if chunk in chunk_to_id:
            assert chunk_to_id[chunk] == tid
        chunk_to_id[chunk] = tid
        if tid in seen_ids and seen_ids[tid] != chunk:
            collisions += 1
        seen_ids[tid] = chunk
assert collisions == 0

# SYSTEM_PROMPT 注释宣称 "~2000 token equivalent"，用文件自己的 tokenize() 实测核对。
assert len(tokenize(SYSTEM_PROMPT)) < 1000     # 实际远低于注释宣称的 ~2000

# --- L10：sglang_compare.py 自己的诚实合成成本模型 ---
rows = {r["scenario"]: r for r in run_all()}
assert rows["tot_8way"]["vllm_prefill_tokens"] == 8400
assert rows["tot_8way"]["sglang_prefill_tokens"] == 1400
assert rows["tot_8way"]["sglang_prefill_gain"] == "+83.3%"
assert rows["shared_system_chat"]["sglang_prefill_gain"] == "+0.0%"
assert rows["json_schema"]["sglang_prefill_gain"] == "+0.0%"       # 见下方说明
assert rows["react_5step"]["sglang_prefill_gain"] == "+0.0%"       # 见下方说明

# 这两个 "+0.0%" 不等于 "SGLang 在这些场景没有优势"——只是这个成本模型没有对
# decode 阶段的优势建模，字段里老实写明了具体是哪块没算：
assert "xgrammar" in rows["json_schema"]["other_advantage_not_modeled"]
assert "KV" in rows["react_5step"]["other_advantage_not_modeled"]
```

实测(`.venv` 真跑):`j = {"n_agents": 32, "n_forwards": 160, "radix_hit_rate": 0.917, "radix_hits": 26815, "radix_total": 29248, "elapsed_s": ≈0.01, "tool_calls": {"search": 32, "calc": 32}, "forwards_per_agent": 5.0}`——`radix_hit_rate=0.917` 这个数字本文独立重跑了 5 次(含不同 `PYTHONHASHSEED`:0/1/42)全部精确复现,`tokenize()` 底层用到的 `hash()` 在不同哈希种子下返回的具体整数值不同,但 32 个 agent 之间"哪些 4 字符片段彼此相同"这个模式是由 `SYSTEM_PROMPT`/`mock_generator` 的文本内容决定的、和哈希种子无关,所以命中率对哈希随机化完全不敏感,不是需要在特定环境下才能复现的脆弱数字。哈希碰撞检查:32 个 agent 的最终 prompt 里一共出现 180 个互不相同的 4 字符片段,精确映射到 180 个互不相同的 id,`collisions=0`。`SYSTEM_PROMPT` 用自身 `tokenize()` 实测只产生 `863` 个假 token,明显低于源码注释宣称的"~2000"。`sglang_compare.py` 的 5 行结果:`tot_8way` 为 `vllm=8400 / sglang=1400 / +83.3%`,其余 4 个场景(`shared_system_chat`/`json_schema`/`react_5step`/`long_prompt_short_out`)全部是 `+0.0%`;`json_schema` 行的 `other_advantage_not_modeled` 字段包含"xgrammar"字样,`react_5step` 行包含"KV"字样,精确对应"这个成本模型没有建模的优势具体是什么"。

**面试怎么问 + 追问链:**
- **Q:** "这个 capstone 测出的 91.7% radix 命中率具体是怎么算出来的?"—— 期望说出"32 个 ReAct agent 共享同一段约 2000 字符量级的 SYSTEM_PROMPT,各自生成完整对话后,把最终 prompt 用一个基于 hash 的假分词器切成 token,插入同一棵 RadixTree,命中率是 `radix_hits / (radix_hits + radix_misses)`"。
- **追问 1(考察是否理解在线 vs 离线的区别):** "这个命中率统计,和真实 SGLang 运行时统计出来的命中率,在测量方式上有什么本质不同?"—— 期望说出"这里是 post-hoc 的——先完整生成整段对话,再一次性插树统计;真实系统是在线的——每一步 decode 都实时维护 radix 树、当场就能复用已经在树里的前缀。这个 capstone 本质上是一个'如果把这些对话整体插入同一棵树,会重合多少'的事后统计工具"。
- **追问 2(核心陷阱,考察是否会盲目相信一个'看起来精确'的数字):** "`tokenize()` 用 `hash()` 当假分词器,这样测出来的命中率可信吗?会不会是哈希碰撞把命中率灌高的?"—— 期望说出"存在这个风险(不同片段可能撞到同一个 20-bit 值,导致树误判为相同 token),但可以通过遍历这次实际跑出来的所有片段、检查是否存在两个不同片段映射到同一个 id 来验证;这次具体的 32-agent 场景检查下来是零碰撞",能看出候选人不会拿到一个数字就直接相信,而是会想办法验证测量本身有没有被污染。
- **追问 3:** "lecture L10 说 SGLang 在 JSON 结构化场景比 vLLM 快 189%,但 `sglang_compare.py` 自己的成本模型算出来是 +0.0%,这是不是说明代码有 bug?"—— 期望说出"不是 bug,是两个数字在衡量不同的东西:lecture 的 189% 是真实 GPU 上的 decode 吞吐(tok/s),包含了 xgrammar 约束解码本身的速度优势;`sglang_compare.py` 的成本模型只统计 prefill token 数量,这个场景里两个引擎需要 prefill 的 token 数确实相同,所以模型诚实地给出 0%,并且在 `other_advantage_not_modeled` 字段里明确写出'这里还有 xgrammar 的优势没算进去'——诚实地承认模型的局限,比用一个和公式本身不一致的数字掩盖局限更可取"。

**常见坑:** 把 `agent_server.py` 里的 `elapsed_s`(约 0.01 秒量级)误认为是"真实 GPU 推理延迟"或者拿它去论证"SGLang 有多快"——这个耗时是纯 Python 循环跑完 32 个 mock agent 的 wall time,和任何真实的模型前向计算都没有关系,这个 capstone 全程没有一次真实的矩阵运算。另一个常见坑是把 `sglang_compare.py` 里某个场景的 `+0.0%` 直接理解成"SGLang 在这个场景没有优势",而不去看 `other_advantage_not_modeled` 字段——这个成本模型的建模范围本来就只覆盖 prefill token 数量这一个维度,一个场景显示 `+0.0%` 只说明"在这一个维度上两个引擎打平",不代表"这个场景 SGLang 整体不占优",这是阅读任何简化成本模型时都要留意的边界。
