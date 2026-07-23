# 03 · Adapter 家族核心深挖(Adapter Family Core)

> 总览见 [00-roadmap.md](00-roadmap.md)

Adapter 是 PEFT 里比 LoRA 更早出现的一条技术路线(Houlsby et al. 2019 vs LoRA 2021),核心思路和 LoRA 完全不同:LoRA 是"改权重"——在冻结的线性层旁边并联一个低秩矩阵去逼近权重的增量;Adapter 是"加层"——直接在 Transformer block 内部插入一个新的小型子网络(down-project → 非线性 → up-project → 残差)。本文是 `peft-deep-dive` 系列第 3 个文件,6 个知识点,源码全部在 `learning/adapter-tuning-family/src/{adapter_original_minimal,houlsby_minimal,pfeiffer_minimal,adapterfusion_minimal,compacter_minimal,parallel_minimal,ia3_minimal,ia3_peft}.py`:知识点 1 从纯 Python(无 torch)实现讲清楚 bottleneck adapter 最原始的数学骨架;知识点 2 讲 Houlsby 和 Pfeiffer 这两种"每个 block 插几个 adapter"的经典选择;知识点 3(AdapterFusion)讲怎么用 attention 机制组合多个已经训好、冻结不动的 adapter;知识点 4(Compacter)讲怎么用 PHM/Kronecker 分解把 adapter 本身再压缩几倍;知识点 5(Parallel Adapter)讲把串联换成并联带来的结构性差异;知识点 6((IA)³)讲把矩阵整个换成对角缩放向量能省到什么程度,也是全系列唯一有 3 条独立实现路径互相交叉验证参数量的知识点。

**和 01/02 号文件的关系:** 01、02 号文件讲的是 LoRA 家族(LoRA 本身的数学与初始化变体、和量化的组合),03 号文件(本文)切换到 Adapter 家族——这是两条相对独立发展出来的 PEFT 技术路线,不是"LoRA 学完了、Adapter 是它的下一步"这种递进关系(时间线上 Adapter 2019 年的论文反而比 LoRA 2021 年更早)。两条路线在方法论上确实有交集——04 号文件知识点 5 的统一公式会证明 LoRA 本质上是"去掉非线性的并联 Adapter"——但阅读顺序上完全可以先读 03 再读 01/02,或者反过来,不存在谁是谁的前置知识,这是两条平行的技术路线,不是递进关系。

**和 04 号文件的关系:** 04 号文件(`04-adapter-advanced.md`,Adapter 进阶与统一视角)已经写完,里面 AdapterDrop、K-Adapter、MAD-X、AdaMix 四种方法全部直接复用本文知识点 2 的 `HoulsbyAdapter` 类做底层构件,04 号文件开篇也明确写了"03 号文件讲的是 adapter 这个构件本身的地基"——本文就是那个地基,04 号文件默认你已经读过本文。反过来本文不会涉及 04 号文件已经讲过的内容(AdapterDrop 的训练时随机丢/推理时永久丢、MAM 的 Prefix 近似实现、K-Adapter/MAD-X 的多 adapter 组合方式、AdaMix 的随机路由、Prompt+LoRA+Adapter 三线统一公式)。知识点 5(Parallel Adapter)会提到它和 LoRA 结构上的相似性(源文件自己的 docstring 就写了"这与 LoRA 的差异仅在于是否有非线性"),但把严格的代码级等价证明留给 04 号文件知识点 5 去做,这里不重复。

**一个重要的诚实标注:** 本文涉及的源文件都有配套 lecture(`learning/adapter-tuning-family/lectures/{01-houlsby-pfeiffer,02-adapterfusion,03-adapterdrop-compacter,04-parallel-adapter,05-ia3}.md`),写作时参考了它们对机制的讲解和部分论文实验数字,但**所有"可运行例子"的数字都是本文用不同于 lecture/源文件自带 `main()`/smoke test 的输入(不同随机种子、不同断言角度、部分直接构造 toy tensor)重新独立跑出来的**,凡是引用 lecture 里论文自己的 benchmark 数字(GLUE / ROUGE-L / T0 等),都明确标注"来自 lecture,本文未独立复现"。知识点 4(Compacter)额外发现了一处源码自身诊断打印语句的公式错误——`compacter_minimal.py::main()` 里估算 per-layer 参数量的打印语句用错了除数,把真实值低估了 4 倍,最后一行还留了个没算完的 `= ?`——这是本文自己动手对 `PHMLinear` 做 `.numel()` 精确计算后发现的,不是凭印象转述,也不影响模型真正的参数量(那部分走的是 `print_param_summary` 的真实 `.numel()` 求和,是对的,只是诊断打印语句本身有 bug)。知识点 6((IA)³)提到的 3 条实现路径里,`adapters` 库那一条本机确实无法 `import`,如实展示真实报错信息,不假装能跑通,也不借用 lecture 里(可能在别的环境下测出来的)具体数字冒充本机验证结果。

**环境声明:** 本文涉及的源文件全部要构造一个真实的 GPT-2(`gpt2`,124M 参数)模型做前向传播(知识点 3 额外做了一次真实的反向传播,验证梯度只流向 fusion 层、不流向冻结的 adapter),不需要 GPU,CPU 上秒级到几秒跑完,模型从本地 Hugging Face 缓存加载——运行时会看到"unauthenticated requests"的 Warning 提示,但权重加载速度是每秒数千次迭代,说明命中的是本地缓存,不是每次都重新联网下载。本文所有代码例子已在仓库根目录 `.venv`(Windows 原生,Python 3.13,torch 2.11.0+cu128,transformers 5.10.2,peft 0.19.1)下用 `.venv/Scripts/python.exe` 实际跑通验证,文中数字是真实输出,不是手算或转述;知识点 6 提到的 `adapters` 库 `ModuleNotFoundError`,同样是本机真实报错,不是编造。

---

## 1. 原始 bottleneck adapter 数学(`adapter_original_minimal.py`)—— 纯 Python 里的 down → 非线性 → up → 残差

**是什么:**
```python
def single_adapter_parameters(hidden_dim: int, adapter_dim: int) -> int:
    """Parameter count for down projection, up projection, and biases."""
    down = hidden_dim * adapter_dim + adapter_dim
    up = adapter_dim * hidden_dim + hidden_dim
    return down + up


def adapter_forward(
    x: list[float],
    down: list[list[float]],
    up: list[list[float]],
    down_bias: list[float],
    up_bias: list[float],
) -> list[float]:
    """A tiny bottleneck adapter: x + up(relu(down(x)))."""
    z = add(matvec(down, x), down_bias)
    z = relu(z)
    delta = add(matvec(up, z), up_bias)
    return add(x, delta)
```
(`adapter_original_minimal.py:29-33`、`:68-79`)

结尾函数(`adapter_original_minimal.py:128-137`,文件最后两行 `if __name__ == "__main__": _self_test()` 才是真正触发执行的地方):
```python
def _self_test() -> None:
    result = summarize()
    assert result["single_adapter_gpt2_r16"] == 25_360
    assert result["houlsby_gpt2_r16"] == 608_640
    assert result["adapter_9_tasks"] < result["full_finetune_9_tasks"]
    assert result["near_identity_input"] == result["near_identity_output"]

    for key, value in result.items():
        print(f"{key}: {value}")
    print("adapter_original_minimal self-test passed")
```

**一句话:** 这是整个 Adapter 家族最原始的数学骨架——高维输入先降到一个很窄的瓶颈维度、过一个非线性、再升回原维度、最后加回输入自身,这份实现完全不依赖 torch、不依赖任何 Transformer 结构,用纯 Python list 运算就能完整表达这套公式;它在命名上也和全系列其他文件不一样——用 `_self_test()` 收尾并带 4 条真 `assert`,而不是像其他文件那样用 `main()` 只做打印展示。

**底层机制/为什么这样设计:** 核心公式是 `Adapter(x) = x + up(relu(down(x)))`,`down` 把 `d` 维输入投影到 `r` 维瓶颈(`r ≪ d`),`up` 再投影回 `d` 维,最后残差加回原始输入。参数量公式 `single_adapter_parameters(d, r) = (d·r+r) + (r·d+d) = 2dr+d+r`——对 GPT-2 的 `d=768, r=16`:`2×768×16+768+16 = 24,576+784 = 25,360`,和 `houlsby_minimal.py` 里用真实 `nn.Linear` 跑出来的参数量精确一致(见知识点 2),说明这套纯 Python 公式和真实 torch 实现在参数量层面是同一件事,只是这里为了教学目的手写了矩阵乘法(`matvec`)和加法(`add`),没有借助任何框架。`near_identity_demo()` 把 `up` 矩阵显式设成全零(`zero_matrix`),这样不管 `down`/`relu` 算出什么中间结果,乘上全零的 `up` 之后 `delta` 恒为零向量,`Adapter(x) = x + 0 = x`——这和 LoRA 用零初始化 `B` 矩阵保证初始 `ΔW=0` 是完全相同的设计思想,而且 Adapter(2019)在时间线上比 LoRA(2021)更早使用这一技巧,这个"新增分支零初始化、保证训练起点等于原模型"的惯例可以说是从这里(或更早的同类工作)传下来的,不是 LoRA 发明的专利。`houlsby_adapter_parameters` 只是把 `single_adapter_parameters` 乘上 `layers × adapters_per_layer`,这里还不涉及"每个 block 插几个"的选择(`adapters_per_layer` 只是一个可配置数字,默认值 2 恰好对应知识点 2 要讲的 Houlsby 布局,但这个文件本身不关心插入拓扑,只关心"给定插入次数,参数量怎么算")。关于命名:全系列其他 `*_minimal.py` 文件清一色用 `if __name__ == "__main__": main()` 收尾,`main()` 里只做 `print`、不做 `assert`,真正的正确性验证放在 `tests/` 目录的独立 pytest 文件里;这个文件反过来,把 4 条 `assert` 直接写进了 `_self_test()`——但因为函数名以下划线开头且不匹配 `test_*`/`*_test` 命名模式,pytest 默认收集规则完全不会自动发现它,必须显式 `python adapter_original_minimal.py` 才会跑到。

**AI 研究场景:** `full_finetune_total_params`/`adapter_total_params` 这一对函数量化的是 2019 年 Houlsby 论文最核心的动机——多任务场景下,全参数微调要为每个任务存一份完整模型副本(存储量随任务数线性增长、系数是整个 backbone 的大小),而 Adapter 方案只需要一份冻结的 backbone 加上每个任务一份很小的 adapter 权重集合。这正是今天"一个基座模型 + 很多小插件"这类多租户模型托管平台(AdapterHub 就是这个思路的直接产物)的存储经济学基础,后来 LoRA/(IA)³ 等方法在存储效率上做得更极致,但"冻结主干、插拔小模块"这个哲学起点就是这里。

**可运行例子:**
```python
import inspect
import sys
sys.path.insert(0, "learning/adapter-tuning-family/src")
import adapter_original_minimal as aom

# 命名事实：只有 _self_test，没有 main
assert hasattr(aom, "_self_test")
assert not hasattr(aom, "main")
src = inspect.getsource(aom._self_test)
assert src.count("assert ") == 4          # 4 条真 assert，不是只打印

# 参数量公式核实（GPT-2 d=768, r=16）
assert aom.single_adapter_parameters(768, 16) == 25_360
cfg = aom.AdapterConfig(layers=12, hidden_dim=768, adapter_dim=16, adapters_per_layer=2)
assert aom.houlsby_adapter_parameters(cfg) == 608_640

# 零初始化 up -> 精确恒等
x, y = aom.near_identity_demo()
assert x == y

# 对照组：up 不是零矩阵时，恒等性质消失
down = [[0.10, -0.20, 0.05, 0.00], [0.00, 0.10, -0.10, 0.20]]
up_nonzero = [[0.5, -0.3], [0.1, 0.2], [-0.4, 0.6], [0.2, -0.1]]
y2 = aom.adapter_forward(x, down, up_nonzero, [0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
assert x != y2

# 9 个任务的存储经济学（沿用文件里 BERT-large 量级的配置）
bert_large_like = aom.AdapterConfig(layers=24, hidden_dim=1024, adapter_dim=64, adapters_per_layer=2)
full = aom.full_finetune_total_params(330_000_000, num_tasks=9)
adapted = aom.adapter_total_params(330_000_000, bert_large_like, num_tasks=9)
assert full == 2_970_000_000
assert adapted == 387_093_120
ratio = adapted / full
assert ratio < 0.14                        # adapter 方案只需要全量微调约 13% 的存储

# pytest 不会自动收集 _self_test（不匹配 test_* / *_test 命名）
assert not aom.__name__.startswith("test")
assert not aom._self_test.__name__.startswith("test")
```

实测(`.venv` 真跑):`_self_test` 源码里精确有 `4` 条 `assert`,模块没有 `main` 属性。零初始化下 `near_identity_demo()` 返回 `x == y == [1.0, -2.0, 0.5, 3.0]`;换成非零 `up` 后 `y2 = [1.1575, -1.8775, 0.5, 3.07]`,和 `x` 不再相等。存储经济学:9 个任务全参数微调 `full = 2,970,000,000`,Adapter 方案 `adapted = 387,093,120`,比值约 `0.1303`(即 Adapter 方案只需要约 13.03% 的存储,节省约 86.97%)。另外用 `.venv/Scripts/python.exe -m pytest learning/adapter-tuning-family/src/tests/test_adapter_original_minimal.py --collect-only -q` 实际核实过:pytest 只收集到该测试文件里 5 个 `test_*` 函数,`_self_test()` 本体完全没有出现在收集列表里——仓库真正接入 CI/pytest 的覆盖率来自 `tests/test_adapter_original_minimal.py` 里另外单独写的 5 个 `test_*` 函数(断言内容和 `_self_test()` 重复但并不是直接调用它),不是 `_self_test()` 自己。

**面试怎么问 + 追问链:**
- **Q:** "bottleneck adapter 的参数量怎么算,GPT-2(d=768, r=16)一个 adapter 多少参数?"—— 期望写出 `2dr+d+r` 的公式并算出 `25,360`。
- **追问 1:** "公式里为什么有一个 `+d+r`,这两项是什么?"—— 期望说出"是 down/up 两个线性层各自的 bias 项",能顺带指出如果两个线性层都不带 bias,公式会简化成 `2dr`。
- **追问 2(命名细节,考察是否真读过源码):** "这个文件用 `_self_test()` 而不是常见的 `main()`,这个区别只是风格问题吗?"—— 期望说出"不只是风格,`_self_test()` 里有真 `assert`,而且这个命名不会被 pytest 自动收集(不匹配 `test_*`/`*_test` 规则),仓库真正的 pytest 覆盖率来自 `tests/` 目录下另外重写的 5 个 `test_*` 函数",能看出这是"文件自查"和"CI 覆盖"两件独立的事。
- **追问 3(深挖存储经济学):** "9 个任务用 Adapter vs 全参数微调,存储差距大概是多少倍?"—— 期望能现场用公式估算量级(backbone 共享一份 vs backbone 存 9 份),并说出"任务数越多,Adapter 方案的相对优势越大,因为 backbone 那部分是一次性成本,不随任务数增长"。

**常见坑:** 把 `houlsby_adapter_parameters(cfg)` 误解成"这个函数决定了插入拓扑是 Houlsby 还是 Pfeiffer"——它只是"给定 `adapters_per_layer` 这个数字,算出总参数量"的纯算术函数,`adapters_per_layer` 默认值恰好是 2(对应 Houlsby),但这个文件完全没有"attn 后插一次、FFN 后插一次"这种拓扑逻辑,真正决定插入位置的是知识点 2 里 `HoulsbyGPT2`/`PfeifferGPT2` 的模型构造代码。另一个坑是想当然地认为 `_self_test()` 会被仓库的 pytest 套件自动跑到——必须显式执行 `python adapter_original_minimal.py` 才会触发,pytest 收集这个文件时完全跳过它,如果只看 CI 日志里"pytest 全绿"就认为这 4 条 assert 被验证过,是不准确的,它们只在手动运行这个文件时才会被执行。

---

## 2. Houlsby vs Pfeiffer(`houlsby_minimal.py` / `pfeiffer_minimal.py`)—— 插 2 个还是插 1 个

**是什么:**
```python
class HoulsbyAdapter(nn.Module):
    """单个 Houlsby Adapter: down → act → up → +residual"""

    def __init__(self, d: int, r: int = 16, act: str = "gelu"):
        super().__init__()
        self.down = nn.Linear(d, r)
        self.up = nn.Linear(r, d)
        self.act = nn.GELU() if act == "gelu" else nn.ReLU()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(x)
        h = self.act(h)
        h = self.up(h)
        return x + h
```
Houlsby(`houlsby_minimal.py:132-136`,在 `HoulsbyGPT2.__init__` 内部,每个 block 插 2 次):
```python
def _insert_houlsby(self, d, r):
    for block in self.lm.transformer.h:
        adapter_attn = HoulsbyAdapter(d, r)
        adapter_ffn = HoulsbyAdapter(d, r)
        block.attn = _AttnAdapterWrapper(block.attn, adapter_attn)
        block.mlp = _MlpAdapterWrapper(block.mlp, adapter_ffn)
```
Pfeiffer(`pfeiffer_minimal.py:44-47`,在 `PfeifferGPT2.__init__` 内部,每个 block 只插 1 次):
```python
def _insert_pfeiffer(self, d, r):
    for block in self.lm.transformer.h:
        adapter_ffn = HoulsbyAdapter(d, r)  # 复用 HoulsbyAdapter 结构
        block.mlp = _MlpAdapterWrapper(block.mlp, adapter_ffn)
```

**一句话:** 两者共用完全相同的单个 adapter 构件(`HoulsbyAdapter` 这个类),差异只在"每个 transformer block 挂几次"——Houlsby 在 attention 输出后和 FFN 输出后各挂一次(2 个),Pfeiffer 干脆只在 FFN 输出后挂一次(1 个),参数量精确减半。

**底层机制/为什么这样设计:** `HoulsbyAdapter` 本身的数学在两个文件里是同一个类,没有任何差异,这也是为什么源码注释里 Pfeiffer 的插入代码直接写"复用 HoulsbyAdapter 结构"。真正的区别在模型构造代码:`HoulsbyGPT2.__init__` 用 `_AttnAdapterWrapper` 包住 `block.attn`、再用 `_MlpAdapterWrapper` 包住 `block.mlp`,一个 block 内产生 2 个独立的 `HoulsbyAdapter` 实例;`PfeifferGPT2.__init__` 只做后一半,跳过 attention 那一次插入。Pfeiffer 论文(`AdapterFusion` 那篇)的实验发现,单独插在 FFN 后的 adapter 已经能拿到接近 Houlsby 双插入的效果——lecture(`lectures/01-houlsby-pfeiffer.md` Slide 9)引用的实验数字是 Houlsby GLUE 均分 84.0 vs Pfeiffer 84.2(Pfeiffer 反而略高,同时训练/推理时延还少 10%),这是论文的实验结论,本文没有条件独立复现下游任务指标,如实标注来源,不重新验证。因为两种布局共用同一个 `HoulsbyAdapter` 类,`up` 层零初始化这条规则对两者同时生效,两个模型在初始化那一刻都应该精确等于原始 GPT-2——这一点本知识点用真实前向传播验证过,不是纸面推导。这里还有一个容易被类名带偏的细节(04 号文件知识点 1 讨论 `AdapterDropGPT2` 时也专门提过):`HoulsbyAdapter` 这个类名描述的是"down→act→up→残差"这个构件本身,和"插入几次"的 Houlsby/Pfeiffer 之争是两个完全独立的维度——`PfeifferGPT2` 内部用的还是名叫 `HoulsbyAdapter` 的类,只是只实例化了一次,读代码时不能看到类名叫 Houlsby 就断定这个模型是 Houlsby 布局。

**AI 研究场景:** 2020 年后社区在生产环境里普遍默认选 Pfeiffer 而不是 Houlsby——同等效果下参数量减半、训练和推理都更快,没有理由多插一倍的 adapter。这个"先设计一个双插入的完整版本、再证明砍掉一半也不影响效果"的路径,在 PEFT 方法演化里反复出现(后面知识点 4 的 Compacter、知识点 6 的 (IA)³ 都是在"还能再省多少"这条主线上继续走),对工程选型的直接启示是:面对一个新论文提出的复杂结构,先问"这里面哪部分是真正必要的、哪部分只是没做消融实验前的保守设计",往往能找到更便宜的等价方案。

**可运行例子:**
```python
import sys
import torch
sys.path.insert(0, "learning/adapter-tuning-family/src")
from houlsby_minimal import HoulsbyGPT2
from pfeiffer_minimal import PfeifferGPT2
from transformers import GPT2LMHeadModel

torch.manual_seed(3)
houlsby = HoulsbyGPT2(r=16)
torch.manual_seed(3)
pfeiffer = PfeifferGPT2(r=16)

n_houlsby = sum(p.numel() for p in houlsby.parameters() if p.requires_grad)
n_pfeiffer = sum(p.numel() for p in pfeiffer.parameters() if p.requires_grad)
assert n_houlsby == 608_640
assert n_pfeiffer == 304_320
assert n_houlsby == 2 * n_pfeiffer                 # 插入次数减半 -> 参数量精确减半

# 两者初始化那一刻都必须精确等于原始 GPT-2（up 层零初始化对两个布局同时生效）
base = GPT2LMHeadModel.from_pretrained("gpt2").eval()
houlsby.eval()
pfeiffer.eval()
enc = houlsby.tokenizer("compare houlsby and pfeiffer insertion", return_tensors="pt", padding=True)
with torch.no_grad():
    out_h = houlsby(enc["input_ids"], enc["attention_mask"]).logits
    out_p = pfeiffer(enc["input_ids"], enc["attention_mask"]).logits
    out_b = base(enc["input_ids"], attention_mask=enc["attention_mask"]).logits
assert torch.equal(out_h, out_b)                    # Houlsby 初始 forward 与 base 逐 bit 相同
assert torch.equal(out_p, out_b)                    # Pfeiffer 同样逐 bit 相同

# 结构证据：Houlsby 每个 block attn 和 mlp 都被包了 adapter；Pfeiffer 只有 mlp
h_block0 = houlsby.lm.transformer.h[0]
p_block0 = pfeiffer.lm.transformer.h[0]
assert hasattr(h_block0.attn, "adapter") and hasattr(h_block0.mlp, "adapter")
assert not hasattr(p_block0.attn, "adapter") and hasattr(p_block0.mlp, "adapter")
```

实测(`.venv` 真跑):`n_houlsby = 608,640`,`n_pfeiffer = 304,320`,精确 2 倍关系;两个模型初始化后的 `logits` 都和裸 GPT-2 的 `logits` `torch.equal`(逐 bit 相同,不是近似);结构检查确认 Houlsby 的 `block.attn`/`block.mlp` 都挂了 `.adapter` 属性(来自 `_AttnAdapterWrapper`/`_MlpAdapterWrapper`),Pfeiffer 的 `block.attn` 是原始未包装的 `GPT2Attention`,没有 `.adapter` 属性。

**面试怎么问 + 追问链:**
- **Q:** "Houlsby 和 Pfeiffer 的核心区别是什么?"—— 期望说出"单个 adapter 构件完全一样,区别只在每个 block 插几次:Houlsby 插 2 次(attn 后+FFN 后),Pfeiffer 插 1 次(只 FFN 后)"。
- **追问 1:** "Pfeiffer 少插一半,效果会打折扣吗?"—— 期望说出"论文实验里 Pfeiffer 效果反而略优于 Houlsby(同时训练/推理更快),不是'牺牲效果换参数量'的权衡,而是发现 attn 后的那个 adapter 边际贡献很小",能主动标注这是论文引用数字、自己没有独立复现下游指标。
- **追问 2(容易被类名误导):** "`PfeifferGPT2` 内部用的类叫 `HoulsbyAdapter`,这是不是意味着它其实是 Houlsby 布局?"—— 期望明确说"不是,`HoulsbyAdapter` 只是单个'down→act→up→残差'构件的类名,和插入几次是两个维度的事;`PfeifferGPT2` 只是复用了这个类、只实例化一次"。
- **追问 3(深挖初始化):** "两个模型的 `up` 层都零初始化,能不能不用真跑前向、只看代码就确定两者初始化后 forward 完全相同?"—— 期望说出"能,因为 `HoulsbyAdapter.forward` 里 `up` 恒为零输出时 `Adapter(x)=x+0=x`,不管这个模块被插入到哪个位置、插了几次,只要每次插入后都紧跟着'加回原值',整个计算图在初始化那一刻就精确退化为原始 GPT-2",但也要认可"实际跑一遍前向"仍然是比纯推理更可靠的验证方式。

**常见坑:** 把"Houlsby 参数量是 Pfeiffer 的 2 倍"误记成"Houlsby 效果也应该更好"——多插一倍的可训练参数在直觉上容易让人以为效果必然更强,但知识点里引用的 lecture 数据显示 Pfeiffer 实验效果反而略优,这提醒"参数量更大"和"效果更好"不能简单划等号,尤其是在两个额外插入位置里其中一个边际贡献本来就很小的情况下。另一个坑是拿到一个 adapter 相关的类,看到类名里带 "Houlsby" 或者 "Pfeiffer" 字样就直接下结论说这个模型是哪种布局,必须实际看模型构造代码里"插了几次、插在哪"才能确定,04 号文件讨论 `AdapterDropGPT2` 时就踩过一次这个坑(类名叫 `HoulsbyAdapter`,但实际插入拓扑是 Pfeiffer 式的)。

---

## 3. AdapterFusion(`adapterfusion_minimal.py`)—— 用 attention 融合多个冻结 adapter

**是什么:**
```python
class FusionLayer(nn.Module):
    """单个 Fusion 层：用 attention 融合 N 个 frozen adapter 的输出。"""

    def __init__(self, d: int, n_adapters: int):
        super().__init__()
        self.d = d
        self.n_adapters = n_adapters
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.W_v.weight)               # value 投影恒等初始化
        nn.init.normal_(self.W_q.weight, std=0.02)
        nn.init.normal_(self.W_k.weight, std=0.02)

    def forward(self, x: torch.Tensor, adapter_outs: list[torch.Tensor]) -> torch.Tensor:
        Q = self.W_q(x)
        A = torch.stack(adapter_outs, dim=2)         # (b, s, N, d)
        K = self.W_k(A)
        V = self.W_v(A)
        scores = (Q.unsqueeze(2) * K).sum(dim=-1) / math.sqrt(self.d)
        attn = F.softmax(scores, dim=-1)
        return (attn.unsqueeze(-1) * V).sum(dim=2)
```
(`adapterfusion_minimal.py:36-70`)

**一句话:** 先各自单独训练好 N 个任务专属 adapter 并把它们冻结,再在每个 block 里加一个 attention 式的 fusion 层——用当前 hidden state 当 Query、N 个冻结 adapter 的输出分别当 Key 和 Value,让模型自己学"这一步该多信哪个 adapter"。

**底层机制/为什么这样设计:** `_FusionMlpWrapper.forward`(`adapterfusion_minimal.py:82-86`)先算出 base MLP 的输出 `h`,再让 `h` 分别过 N 个冻结 adapter 得到 `adapter_outs`,最后把 `h` 本身(作为 Query 来源)和这 N 个输出一起丢给 `FusionLayer`。这本质上是一个 N-key 的 cross-attention:`Q=W_q·h` 问的是"当前这个 hidden 需要什么样的知识",`K_i=W_k·a_i(h)` 是"第 i 个 adapter 能提供什么",`V_i=W_v·a_i(h)` 是"第 i 个 adapter 实际给出的内容",`softmax(QK^T/√d)` 算出的权重决定最终按什么比例混合 N 个 `V_i`。`W_v` 用 `nn.init.eye_` 恒等初始化,不是随机初始化——这样初始时 `V_i = a_i(h)`(没有经过任何线性变换的失真),配合 `W_q`/`W_k` 用很小的标准差(`std=0.02`)初始化(初始 attention 分数接近 0,`softmax` 后接近均匀分布),fusion 层第一步的行为接近"把 N 个 adapter 的输出做加权平均、权重接近均匀",不会一上来就引入剧烈失真——这和 `HoulsbyAdapter` 用 `up` 层零初始化保证"初始等于恒等"是同一类"最小化初始扰动"的设计思想,只是这里初始状态不是恒等映射,而是"温和地平均现有的 N 份知识"。整个 `AdapterFusionGPT2.__init__` 里,每个 block 新建的 N 个 `HoulsbyAdapter` 会立刻被显式冻结(源码注释直接写"冻 adapters(模拟'预训练 adapter')",实现上是对每个 adapter 各自的参数再跑一遍 `p.requires_grad=False`,不依赖 `freeze_base_model` 顺带冻住它们),只有 `fusion` 是可训练的——这意味着这个模型的训练目标非常单一:不改变任何已经学好的任务知识,只学"怎么组合"这一件事,论文管这个叫"non-destructive"(不破坏式)组合,是相对于"把 N 个任务的数据混在一起联合训练"(容易互相干扰、旧任务被新任务覆写)的一种更保守的替代方案。

**AI 研究场景:** 大型 SaaS 平台如果给每个客户/每个任务都训一个独立的小 adapter,面对一个新任务或新客户时,与其从头训练或者把所有历史数据混在一起联合训练(有灾难性遗忘风险),不如复用已经训好、已经在各自任务上验证过的那批 adapter,只额外学一个轻量的组合层。lecture(`lectures/02-adapterfusion.md` Slide 10)引用的论文实验数字是:单任务 Pfeiffer adapter 基线 GLUE 均分 84.2,把多个任务的数据混在一起联合训练反而降到 82.5(验证了"联合训练有干扰"的担忧),AdapterFusion 组合出来是 85.8(比单任务基线还高 1.6 分)——这是论文的实验结论,本文没有条件独立复现下游任务指标,如实标注来源。工程上的代价也很直观:本知识点验证过 fusion 层参数量是 `21,233,664`,而单个 Pfeiffer adapter 只有 `304,320`,前者是后者的约 69.8 倍——组合的能力不是免费的,`FusionLayer` 的 `Wq/Wk/Wv` 是三个满秩 `d×d` 矩阵,比它要组合的 adapter 本身重得多。

**可运行例子:**
```python
import sys
import torch
sys.path.insert(0, "learning/adapter-tuning-family/src")
from adapterfusion_minimal import AdapterFusionGPT2

torch.manual_seed(5)
model = AdapterFusionGPT2(n_adapters=3, r=16)
model.train()

# 结构证据：adapters 冻结，fusion 可训练
block0 = model.lm.transformer.h[0].mlp
assert all(not p.requires_grad for a in block0.adapters for p in a.parameters())
assert all(p.requires_grad for p in block0.fusion.parameters())

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert n_trainable == 12 * 3 * 768 * 768 == 21_233_664
n_single_adapter = 304_320  # 知识点 2 里验证过的 Pfeiffer 单 adapter 参数量
ratio = n_trainable / n_single_adapter
assert 69 < ratio < 70                              # fusion 比它要组合的 adapter 重约 70 倍

# 真实反传：确认梯度只流向 fusion，adapter 权重完全收不到梯度
enc = model.tokenizer("fusion gradient probe sentence", return_tensors="pt", padding=True)
enc["labels"] = enc["input_ids"].clone()
out = model(enc["input_ids"], enc["attention_mask"], labels=enc["labels"])
out.loss.backward()
assert all(p.grad is None for a in block0.adapters for p in a.parameters())
assert all(p.grad is not None for p in block0.fusion.parameters())

# W_v 恒等初始化的直接验证
import torch as _torch
assert _torch.allclose(block0.fusion.W_v.weight, _torch.eye(768))
```

实测(`.venv` 真跑):`n_trainable = 21,233,664`,精确等于 `12×3×768²`;`ratio ≈ 69.77`(fusion 层比单个 Pfeiffer adapter 重约 70 倍);反传后 3 个冻结 adapter 的全部参数 `.grad` 都是 `None`,fusion 层的全部参数 `.grad` 都不是 `None`;`W_v.weight` 在构造后确实精确等于 `768×768` 单位矩阵(`torch.allclose` 通过)。

**面试怎么问 + 追问链:**
- **Q:** "AdapterFusion 的 fusion 层具体在做什么运算?"—— 期望说出"是一个 attention:Query 来自当前 hidden state,Key/Value 来自 N 个冻结 adapter 各自的输出,softmax 加权求和决定怎么组合"。
- **追问 1:** "为什么 `W_v` 要用恒等初始化,`W_q`/`W_k` 却用随机初始化?"—— 期望说出"`W_v=I` 保证初始时 Value 就是 adapter 的原始输出、不引入额外失真;`W_q`/`W_k` 用小标准差随机初始化,让初始 attention 分数接近 0、softmax 后接近均匀分布,整体上是让 fusion 层刚开始训练时表现得像'温和平均',不是恒等映射(因为要平均的对象有 N 个,不是 1 个)"。
- **追问 2(深挖参数开销):** "Fusion 层的参数量和 adapter 数量 N 是什么关系?"—— 期望说出"`Wq/Wk/Wv` 是固定的 `d×d` 矩阵,和 N 无关,N 只影响运行时要 stack 几个 adapter 输出去做 attention;但 N 个 adapter 本身的参数量是随 N 线性增长的,只是这些参数是冻结的,不计入 fusion 阶段的可训练参数"。
- **追问 3(考察对"non-destructive"的理解):** "AdapterFusion 为什么被称为'non-destructive'(非破坏式)组合?"—— 期望说出"因为参与组合的 N 个任务 adapter 全程冻结,不会被新任务的梯度更新,新任务学到的东西只存在于额外加的 fusion 层里,不会覆写或污染已经训好的旧任务知识",可以对比"把多任务数据混在一起联合训练"这种会导致相互干扰/灾难性遗忘的替代方案。

**常见坑:** 把 fusion 层的 attention 和 Transformer 内部本来就有的 self-attention 混为一谈——这里的 Query/Key/Value 三者的"token"维度不是序列长度,而是"adapter 编号"(`N` 个),是在"该用哪个 adapter"这个离散选择集合上做 attention,不是在做序列内部的 token 间建模,这也是为什么 lecture 把它称为"task-level attention"而不是普通的 self-attention。另一个坑是忽视 fusion 层的参数开销——只看到"adapter 冻结了、只训一个新层"就以为这是一个轻量方案,实际上 fusion 层(21M)比它要组合的单个 adapter(304K)重了近 70 倍,量级上已经不能算"轻量微调",这也是 lecture 里明确提到的工程劣势,只在真正需要复用多个已训好 adapter 的多任务场景下才划算,单任务场景直接用 Pfeiffer/LoRA 更便宜。

---

## 4. Compacter(`compacter_minimal.py`)—— PHM + 跨层共享,本系列压缩率最高的方法之一

**是什么:**
```python
def kronecker(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """计算 Kronecker 积 A ⊗ B。A: (m, n), B: (p, q) -> (m*p, n*q)"""
    m, n = A.shape
    p, q = B.shape
    out = A.unsqueeze(1).unsqueeze(3) * B.unsqueeze(0).unsqueeze(2)
    return out.reshape(m * p, n * q)


class PHMLinear(nn.Module):
    """PHM-parameterized Linear layer：W = Σᵢ Aᵢ ⊗ Bᵢ，然后 y = Wx + bias"""

    def construct_weight(self) -> torch.Tensor:
        Ws = [kronecker(self.A[i], self.B[i]) for i in range(self.n)]
        return torch.stack(Ws, dim=0).sum(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.construct_weight()
        out = x @ W.T
        if self.bias is not None:
            out = out + self.bias
        return out
```
(`compacter_minimal.py:43-54`、`:57-114`)

**一句话:** Compacter 不直接学一个 `(d_out, d_in)` 的满秩 down/up 矩阵,而是把它分解成 `n` 个 Kronecker 积之和 `W=Σᵢ Aᵢ⊗Bᵢ`,其中"小而重"的 `Aᵢ`(`n×n`)在所有 12 层之间共享同一份,"大而轻"的 `Bᵢ` 才是每层独有的,双重压缩之下拿到本系列数一数二的压缩率。

**底层机制/为什么这样设计:** 先说清楚 `Aᵢ⊗Bᵢ` 这个 Kronecker 积本身在算什么,不然后面的参数量公式无从谈起(如果读过 [01-lora-core.md](01-lora-core.md) 知识点 6 的 LoKr,会发现是同一个操作,这里不要求先读过那一节,独立解释一遍):完整定义见 numpy-deep-dive/06-linear-algebra.md 知识点 16,最简版是——把 `Aᵢ` 的每个元素替换成"该元素 × 整个 `Bᵢ`"这一整块,拼成一个大矩阵;`Aᵢ.shape=(n,n)`、`Bᵢ.shape=(d_out/n, d_in/n)` 时,结果形状是 `(n·d_out/n, n·d_in/n)=(d_out,d_in)`,和普通权重矩阵的形状精确对齐。上面代码里 `kronecker(A, B)` 用 `unsqueeze`+广播乘法+`reshape` 实现的就是这条规则:`A.unsqueeze(1).unsqueeze(3)` 把 `A` 从 `(m,n)` 变成 `(m,1,n,1)`,`B.unsqueeze(0).unsqueeze(2)` 把 `B` 从 `(p,q)` 变成 `(1,p,1,q)`,两者相乘广播出 `(m,p,n,q)`,`reshape` 成 `(m·p, n·q)` 之后,位置 `[i·p+k, j·q+l]` 的值精确等于 `A[i,j]·B[k,l]`——这正是"每个 `A` 的元素乘上一整块 `B`"这条规则按逐元素展开后的样子,不需要逐步验证这几个 `unsqueeze` 在干什么,记住它算出来的结果符合这条规则即可。

弄清楚 Kronecker 积本身之后,再看 PHM 怎么用它压缩参数量:一个普通 `Linear(d_in, d_out)` 需要 `d_out×d_in` 个参数;PHM 把同样大小的权重矩阵拆成 `n` 个 Kronecker 积求和,每个 `Aᵢ∈ℝ^{n×n}`、每个 `Bᵢ∈ℝ^{(d_out/n)×(d_in/n)}`,`n` 个 `Aᵢ` 合计 `n³` 个参数,`n` 个 `Bᵢ` 合计 `d_out·d_in/n` 个参数,总参数量 `n³+d_out·d_in/n`——当 `d_out·d_in≫n⁴` 时近似有 `n` 倍压缩。`CompacterGPT2.__init__` 里额外做了第二层压缩:`shared_A_down`/`shared_A_up` 在模型级别只创建一次,12 个 `CompacterAdapter` 全部传入同一个 `nn.Parameter` 对象作为自己的 `A`——`PHMLinear.__init__` 在 `shared_A is not None` 时直接 `self.A = shared_A`,不新建参数。这里有一个 PyTorch 参数追踪的细节容易被忽视:因为 `nn.Module.__setattr__` 只要发现赋的值是 `nn.Parameter` 就会自动登记进这个模块自己的 `_parameters`,所以从单个 `PHMLinear` 实例的角度看,它的 `.parameters()` 依然会把 `shared_A` 算进去(不知道这份参数已经被别的模块"共享"了);真正的去重发生在从模型顶层(`CompacterGPT2`)调用 `.parameters()`/`.named_parameters()` 做递归遍历的时候——PyTorch 按参数对象的身份(不是按名字)去重,同一个 `shared_A` 不管被多少个子模块引用,在顶层遍历时只会被计入一次。`up` 层的 `B` 矩阵单独做了零初始化(`nn.init.zeros_(self.up.B)`,不是把整个 `up.A` 也清零),保证初始 `Adapter(x)=x`——这个零初始化只需要清空 `B`,因为 `construct_weight()` 是 `Aᵢ⊗Bᵢ` 求和,任何一个因子是零,对应那一项的 Kronecker 积就是零,不需要两个因子同时为零。

关于知识点 4 特有的诚实标注:`compacter_minimal.py::main()`(`:197-203`)里有一段诊断性的 `print` 语句,试图现场估算每层参数量,但公式写错了——`print(f"per layer B_down: 768*16/n^2 = {768*16//16}")` 用 `n²=16` 做除数,算出 `768`,但 `B_down` 真实的参数总量应该是 `n` 个 `(out/n)×(in/n)` 矩阵之和,即 `d_out·d_in/n = 768×16/4 = 3,072`——本文直接对 `PHMLinear(768,16,n=4)` 的 `.B` 张量取 `.numel()` 验证过,真实值确实是 `3,072`,`main()` 打印的 `768` 只是"单个 `Bᵢ` 子矩阵"的大小(`4×192=768`),被错误地当成了"4 个 `Bᵢ` 加总"的大小,相当于漏乘了 `n=4`;这段打印的最后一行 `print(f"12 layer 合计: 64*2 + 12 * (768 + 768 + 16 + 768) = ?")` 甚至直接留了一个没有代入计算的 `= ?`,像是写到一半忘了填。这处不准确只存在于 `main()` 的调试打印字符串里,不影响模型真实的参数量——真实总数走的是 `print_param_summary` 对 `.numel()` 的直接求和,本知识点验证过等于 `83,264`,这个数字是对的;反而是文件最顶部模块级 docstring(`:19-22`)给的单层示例 `PHM: 4³+768·16/4=64+3,072=3,136` 用的除数是对的(`n=4` 不是 `n²=16`),和 `main()` 内部打印的公式互相矛盾,读到这类文件内部两处表述不一致时,应该以真正参与计算的代码(`.numel()` 求和)为准,不能假设一个文件从头到尾所有注释/打印都互相自洽。

**AI 研究场景:** lecture(`lectures/03-adapterdrop-compacter.md` Slide 6)指出 Compacter 的动机是"对更大的模型,单个 adapter 的参数量本身也会变得可观"——如果 backbone 的隐藏维度从 GPT-2 的 768 涨到几千甚至上万,一个满秩 down/up adapter 的参数量会随 `d` 线性增长,而 PHM 分解让这部分增长速度变慢(尤其是跨层共享 `A` 之后,新增一层的边际成本只有 `B` 矩阵那部分)。lecture(Slide 16)引用的论文 GLUE 实验数字是 BERT-base 上 Pfeiffer 894K 参数、86.5 分,Compacter 64K 参数(约 14 倍压缩)、86.4 分——这是论文在不同模型/配置下的实验结论,和本文用 GPT-2、r=16、n=4 跑出来的 3.65 倍压缩比不是同一个设置,不能直接对比,如实标注来源。

**可运行例子:**
```python
import sys
import torch
sys.path.insert(0, "learning/adapter-tuning-family/src")
from compacter_minimal import CompacterGPT2, PHMLinear
from pfeiffer_minimal import PfeifferGPT2

torch.manual_seed(9)
compacter = CompacterGPT2(r=16, n=4)
n_compacter = sum(p.numel() for p in compacter.parameters() if p.requires_grad)
assert n_compacter == 83_264

torch.manual_seed(9)
pfeiffer = PfeifferGPT2(r=16)
n_pfeiffer = sum(p.numel() for p in pfeiffer.parameters() if p.requires_grad)
assert n_pfeiffer == 304_320

ratio = n_pfeiffer / n_compacter
assert 3.6 < ratio < 3.7                              # 整模型压缩比约 3.65x

# 单层 PHM vs 普通 Linear，核对文件顶部 docstring 给出的 3.9x 示例
plain = 768 * 16
down = PHMLinear(768, 16, n=4)
assert down.B.numel() == 3_072                        # 真实 B 参数量（main() 打印语句里错写成 768）
phm_no_bias = down.A.numel() + down.B.numel()
assert phm_no_bias == 64 + 3_072 == 3_136
assert abs(plain / phm_no_bias - 3.9184) < 1e-3

# 跨层共享 A 的去重机制：模型级别只计一次，单个子模块级别会各自"看到"它
shared_A = torch.nn.Parameter(torch.empty(4, 4, 4))
d1 = PHMLinear(768, 16, n=4, shared_A=shared_A)
d2 = PHMLinear(768, 16, n=4, shared_A=shared_A)
assert d1.A is d2.A is shared_A                        # 同一个对象，不是各自拷贝
combo = torch.nn.ModuleList([d1, d2])
n_combo = sum(p.numel() for p in combo.parameters())          # 顶层遍历：按对象身份去重
n_d1_alone = sum(p.numel() for p in d1.parameters())          # 单独看 d1：shared_A 被算了一次
assert n_combo == 2 * n_d1_alone - shared_A.numel()    # combo 里 shared_A 只多算了 0 次，不是 2 次
```

实测(`.venv` 真跑):`n_compacter = 83,264`,`n_pfeiffer = 304,320`,整模型压缩比约 `3.6549`;单层 `PHMLinear(768,16,n=4)` 的 `B` 张量 `.numel()` 精确是 `3,072`(不是 `main()` 打印语句里的 `768`),`A+B`(不含 bias)合计 `3,136`,和普通 `Linear` 的 `12,288` 相比压缩比约 `3.9184`,与文件顶部 docstring 给出的"3.9×"示例吻合;共享 `A` 去重实验里 `d1.A is d2.A is shared_A` 为 `True`,`combo`(两个 `PHMLinear` 放进 `ModuleList` 后从顶层统计)的参数量精确等于 `2×n_d1_alone - shared_A.numel()`,验证了"从顶层看只计一次共享参数,从单个子模块看会重复出现"这条 PyTorch 参数追踪规则。

**面试怎么问 + 追问链:**
- **Q:** "Compacter 用什么方法压缩 adapter 参数量,大概能压缩多少倍?"—— 期望说出"PHM,把满秩矩阵拆成 n 个 Kronecker 积之和,GPT-2 r=16 n=4 配置下整模型相对 Pfeiffer 压缩约 3.65 倍"。
- **追问 1:** "PHM 的压缩效果主要来自公式里的哪一项?"—— 期望说出"`d_out·d_in/n` 这一项随 `n` 线性变小是压缩的主要来源,`n³` 那一项(`A` 矩阵)在 `n` 较小时本身就很小,而且还能跨层共享进一步摊薄"。
- **追问 2(深挖跨层共享):** "跨层共享 `A` 矩阵在 PyTorch 里是怎么实现'只计一次参数'的,如果我在 12 个子模块里各自持有对同一个 `nn.Parameter` 的引用,分别调用每个子模块的 `.parameters()`,会不会重复计数?"—— 期望说出"会,单个子模块的 `.parameters()` 不知道这份参数被别处共享,只有从模型顶层递归遍历时,PyTorch 才会按参数对象的身份去重,只统计一次",这是一个很多人直觉上会答错的 PyTorch 细节。
- **追问 3(诚实标注类,考察是否真的动手算过):** "这个文件的 `main()` 函数打印的参数量分解和真实值对得上吗?"—— 期望说出"对不上,`main()` 内部一段打印语句把除数从 `n` 错写成 `n²`,导致打印出来的单层 `B` 参数量比真实值少了 4 倍,而且最后一行汇总公式没有算完,只留了个 `?`;但这只影响调试打印,不影响真实参数量(真实值走的是 `.numel()` 精确求和,没有问题)",能说出这一点说明是真的动手验证过,而不是读了打印就直接相信。

**常见坑:** 只看 `main()` 打印出来的"参数布局推算"就当成权威数字抄进笔记——本知识点已经验证过这段打印语句本身有除数错误,真实参数量必须用 `sum(p.numel() for p in model.parameters() if p.requires_grad)` 或者直接读 `PHMLinear` 的 `.A`/`.B` 张量形状重新算,不能相信任何未经验证的调试打印字符串,哪怕它就在源码文件里、看起来"官方"。另一个坑是把"跨层共享 `A`"想象成"12 层用同一套完整的 down/up 权重",实际上共享的只是 `A`(决定 Kronecker 积的"宏观结构"),真正携带大部分信息量的 `B` 矩阵(参数量远大于 `A`)依然是每层独立学习的,12 层的 Compacter adapter 并不是简单的权重共享(weight tying),只是共享了参数量占比很小的那一部分。

---

## 5. Parallel Adapter(`parallel_minimal.py`)—— 并联而不是串联

**是什么:**
```python
class ParallelAdapter(nn.Module):
    """单个 Parallel Adapter: out = base(x) + s * up(σ(down(x)))"""

    def __init__(self, base_module: nn.Module, r: int = 16, scaling: float = 1.0):
        super().__init__()
        for p in base_module.parameters():
            p.requires_grad = False
        self.base = base_module
        # ... 推断 d_in, d_out 后 ...
        self.down = nn.Linear(d_in, r)
        self.up = nn.Linear(r, d_out)
        self.act = nn.GELU()
        self.scaling = scaling
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        adapter_out = self.up(self.act(self.down(x))) * self.scaling
        return base_out + adapter_out
```
(`parallel_minimal.py:34-73`)

**一句话:** Houlsby/Pfeiffer/Compacter 全部是"串联"结构——adapter 的输入是 base 模块算完之后的输出(`adapter(base(x))`);Parallel Adapter 把这条链路拆开,adapter 分支和 base 分支各自独立拿原始输入 `x` 去算,算完了才相加(`base(x) + s·adapter(x)`),这个"加法式旁路"的设计后来是 LoRA 的重要前身/近亲。

**底层机制/为什么这样设计:** 对比 `forward` 代码就能看出关键差异:`_MlpAdapterWrapper.forward`(知识点 2/4 都在用)是 `h=self.base_mlp(x); h=self.adapter(h); return h`——adapter 的输入 `h` 已经是 base 算完的结果;`ParallelAdapter.forward` 是 `base_out=self.base(x); adapter_out=...(self.down(x))...; return base_out+adapter_out`——`self.down(x)` 用的是原始 `x`,和 `self.base(x)` 是两条独立的计算路径,只在最后一步相加,`base` 分支的具体数值完全不影响 `adapter` 分支的输出(两者共享的只是输入 `x`,不共享中间结果)。这带来一个可以直接用代码验证的结构性推论:如果把 `base` 模块的权重改成任意其他值,`ParallelAdapter` 的 adapter 分支输出应该完全不受影响(因为它压根不读 `base` 的输出),但换成串联结构(比如 `HoulsbyAdapter` 通过 `_MlpAdapterWrapper` 包裹 base 之后),adapter 的输出必然随着 `base` 的变化而变化,因为它的输入直接就是 `base(x)`。`up` 层依然零初始化,保证初始 `adapter_out=0`、`out=base(x)`,和串联结构享有同一条"新增分支零初始化"的惯例。并联结构还带来一个工程副产品:`base(x)` 和 `adapter(x)` 两条分支互不依赖对方的输出,理论上可以并行计算(GPU 上同时发射两条计算路径),而串联结构里 adapter 必须等 base 算完才能开始,这是并联相对串联在计算图层面的直接优势,和"效果好不好"是两个独立的维度。源文件自己的 docstring 提到"这与 LoRA 的差异仅在于是否有非线性 `σ`",这个观察是对的、也是后续 He et al. 2022"统一视角"论文的出发点,但把它转成严格的代码级等价证明(控制变量、对照实验)是 04 号文件知识点 5 的工作,这里不重复展开。

**AI 研究场景:** lecture(`lectures/04-parallel-adapter.md` Slide 9)引用的论文在 XSum 摘要任务上的实验数字是 Pfeiffer(串联)ROUGE-L 35.2、Parallel Adapter 35.6、LoRA 35.1——这是论文的实验结论,本文没有条件独立复现下游任务指标,如实标注来源。并联这个"结构改动"本身,是"Towards a Unified View of Parameter-Efficient Transfer Learning"这篇论文重新审视已有 PEFT 方法、寻找共同数学骨架的一块拼图——它提示了一个更一般的研究方法论:与其一直在"串联的具体实现细节"上打磨,不如退一步问"这个新增分支到底需要以什么方式和主干结合",往往能同时发现新方法(Parallel Adapter)和已有方法的隐藏联系(LoRA)。

**可运行例子:**
```python
import torch
import torch.nn as nn
import sys
sys.path.insert(0, "learning/adapter-tuning-family/src")
from parallel_minimal import ParallelAdapter
from houlsby_minimal import HoulsbyAdapter

torch.manual_seed(0)
d, r = 8, 4
base = nn.Linear(d, d)
for p in base.parameters():
    p.requires_grad = False

padapter = ParallelAdapter(base, r=r, scaling=1.0)
with torch.no_grad():
    padapter.up.weight.normal_(std=0.1)   # 打破零初始化，让 adapter 分支产生非零输出
x = torch.randn(2, 5, d)

adapter_branch_before = padapter.up(padapter.act(padapter.down(x)))
with torch.no_grad():
    base.weight.mul_(1000.0)              # 剧烈改变 base 的权重
adapter_branch_after = padapter.up(padapter.act(padapter.down(x)))
assert torch.equal(adapter_branch_before, adapter_branch_after)  # 并联：adapter 分支完全不受 base 变化影响

# 对照组：串联结构（Houlsby 风格），adapter 的输入就是 base 的输出，必然跟着变
hadapter = HoulsbyAdapter(d, r)
with torch.no_grad():
    hadapter.up.weight.normal_(std=0.1)
base2 = nn.Linear(d, d)
h_before = hadapter(base2(x))
with torch.no_grad():
    base2.weight.mul_(1000.0)
h_after = hadapter(base2(x))
diff = (h_before - h_after).abs().max().item()
assert diff > 1.0                          # 串联：base 一变，adapter 输出跟着剧烈变化

# ParallelAdapter 的 forward 就是 base(x) + adapter 分支
full_out = padapter(x)
manual = base(x) + adapter_branch_after
assert torch.allclose(full_out, manual, atol=1e-6)
```

实测(`.venv` 真跑):把 `base` 的权重放大 1000 倍后,`ParallelAdapter` 的 adapter 分支输出 `torch.equal` 前后完全一致(逐 bit 相同);同样的权重放大操作作用在串联结构(`HoulsbyAdapter` 包裹 `base2`)上,adapter 输出最大逐元素差异达到约 `1578.27`,量级悬殊——直接证明了"并联"和"串联"在计算图连接方式上的根本区别,不是效果层面的经验性差异,而是结构上可以严格验证的事实。`ParallelAdapter.forward` 的完整输出精确等于手动拼出的 `base(x)+adapter_branch`(容差 `1e-6` 内一致)。

**面试怎么问 + 追问链:**
- **Q:** "Parallel Adapter 和 Houlsby/Pfeiffer 那种串联结构,最本质的区别是什么?"—— 期望说出"串联里 adapter 的输入是 base 算完的输出;并联里 adapter 和 base 各自独立拿原始输入去算,只在最后相加",不能只停留在"一个叫串联一个叫并联"这种字面区分。
- **追问 1(要求给出可验证的证据,不能只讲直觉):** "能不能不看代码结构图,只用一个实验证明某个 adapter 是并联还是串联?"—— 期望想到"改变 base 的权重,看 adapter 分支的输出会不会跟着变;并联结构下完全不变,串联结构下必然改变",这正是本知识点验证代码的思路。
- **追问 2:** "并联结构在工程上比串联有什么直接好处?"—— 期望说出"两条分支互不依赖对方的输出,理论上可以并行计算,不需要像串联那样等 base 算完才能开始算 adapter",并能区分这是"计算效率"层面的优势,不是"最终效果"层面的优势(后者要看实验数据)。
- **追问 3(引向 04 号文件,但不要求当场证明):** "Parallel Adapter 和 LoRA 是什么关系?"—— 期望说出"源文件自己的注释就写了'差异仅在于是否有非线性',把 `σ` 换成恒等映射,`up(down(x))` 就是纯矩阵乘法,退化成 LoRA 的形式",但如果候选人进一步要求"证明"，可以说"完整的代码级对照实验在 04 号文件里做了,这里先建立直觉"，不需要当场重新推导。

**常见坑:** 把"并联"简单理解成"两个模块顺序颠倒"或者"adapter 先算、base 后算"——真正的区别不是计算顺序,而是"adapter 的输入到底是原始 `x` 还是 `base(x)`",这是数据依赖关系上的区别,不是执行顺序上的区别(实际实现里 `base_out=self.base(x)` 这行代码在 `forward` 里确实写在 `adapter_out` 前面,但这只是书写顺序,不影响两条分支互相独立这个事实)。另一个坑是看到"并联"和"LoRA 前身"这类说法,就直接假设两者在数值上已经等价,而不去追问"是在什么条件下等价"(需要去掉非线性 `σ`)——这个条件本身就是理解这条演化关系的关键,不能跳过。

---

## 6. (IA)³(`ia3_minimal.py` + `ia3_peft.py`)—— 3 个对角缩放向量,全系列唯一 3 条路径互相印证的方法

**是什么:**
```python
class _IA3AttnWrapper(nn.Module):
    def __init__(self, c_attn: nn.Module, d: int):
        super().__init__()
        for p in c_attn.parameters():
            p.requires_grad = False
        self.c_attn = c_attn
        self.d = d
        self.l_k = nn.Parameter(torch.ones(d))
        self.l_v = nn.Parameter(torch.ones(d))

    def forward(self, x):
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d, dim=-1)
        k = k * self.l_k
        v = v * self.l_v
        return torch.cat([q, k, v], dim=-1)
```
FFN 侧(`ia3_minimal.py:68-90`):
```python
class _IA3MlpWrapper(nn.Module):
    def __init__(self, base_mlp: nn.Module):
        super().__init__()
        for p in base_mlp.parameters():
            p.requires_grad = False
        self.base = base_mlp
        intermediate = base_mlp.c_fc.weight.shape[1]     # 3072
        self.l_ff = nn.Parameter(torch.ones(intermediate))

    def forward(self, x):
        h = self.base.c_fc(x)
        h = h * self.l_ff
        h = self.base.act(h)
        h = self.base.c_proj(h)
        return h
```
peft 库版本(`ia3_peft.py:15-26`):
```python
def build_ia3_model_peft(target_modules=None, feedforward_modules=None):
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    target_modules = target_modules or ["c_attn", "c_proj", "c_fc"]
    feedforward_modules = feedforward_modules or ["c_fc"]
    config = IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        feedforward_modules=feedforward_modules,
    )
    return get_peft_model(base, config)
```

**一句话:** (IA)³ 把 adapter 家族"学一个矩阵"的思路砍到极致——不学 down/up 投影矩阵,只学 3 个和 base 权重逐元素相乘的对角缩放向量(`l_k`/`l_v`/`l_ff`),全部用全 1 初始化(乘 1 等价恒等变换,不是零初始化);而且这是全系列唯一有 3 条独立实现路径的知识点——手写 `ia3_minimal.py`、`adapters` 库版本、peft 库 `ia3_peft.py`——其中 `adapters` 库那条路径在本机会真实报错,不能跳过不提。

**底层机制/为什么这样设计:** `_IA3AttnWrapper` 把 GPT-2 `c_attn`(一次性算出 Q、K、V 拼接的 `Conv1D`)的输出 `split` 成 Q/K/V 三段,只对 K、V 各乘一个可训练向量(`l_k`/`l_v`,形状都是 `(d,)=(768,)`),Q 保持不变;`_IA3MlpWrapper` 在 FFN 的中间激活(`c_fc` 输出、`GELU` 之前)乘一个 `l_ff`(形状 `(4d,)=(3072,)`)。三个向量全部 `torch.ones` 初始化,原因是这里的"扰动"是逐元素乘法,乘法的"零元"(不改变原值的那个数)是 1,不是加法语境下的 0——所以 (IA)³ 不是"零初始化"而是"恒等初始化",和 LoRA/Adapter 那套"新增分支零输出"的零初始化思想在数学操作类型上不同,但目的完全一样:保证训练起点等于原始 base 模型。参数量:`768(l_k)+768(l_v)+3072(l_ff)=4,608` 每层,`×12` 层 `=55,296`——本知识点实测确认这个数字和 GPT-2 上的 LoRA r=8(294,912)相比只有约 1/5.3,是这一路"压缩系列"(Houlsby 608K → Pfeiffer 304K → Compacter 83K → (IA)³ 55K)里最小的一个。

三条实现路径的对照是这个知识点的重点。**第一条**,`ia3_minimal.py` 手写,`l_k`/`l_v` 缩放 attention 的 K/V,`l_ff` 缩放 FFN 中间激活,和论文原始设计的位置完全对应。**第二条**,`adapters` 库版本(`ia3_adapters.py`),本机这个 `.venv` 里完全无法运行——直接 `import adapters` 会立刻抛出 `ModuleNotFoundError: No module named 'adapters'`,原因写在 `learning/adapter-tuning-family/environment/requirements.txt` 的注释里,原文是"重要:`adapters` 库强制 `transformers 4.x`,会从 5.9 降级"(这条注释提到的"5.9"应该是撰写时本仓库当时的 `transformers` 版本,现在环境声明里已经是 `5.10.2`,但这个版本号差异不影响结论——`adapters` 要求 `transformers<5.0`,不管本仓库具体停在 5.9 还是 5.10.2,只要是 5.x 就同样冲突),`runbook.yaml` 里 `ia3-adapters` 这一项的 `tier` 被明确标成 `skip`,注释写着"`adapters` 库本仓库未安装(它强制 `transformers 4.x`,与本仓库 `transformers 5.10.2` 冲突),故全部 `tier: skip`,脚本在 import 处即 fail-fast"。仓库自己的测试文件 `tests/test_ia3_three_way.py` 对此的处理方式也很诚实——用 `importlib.util.find_spec("adapters") is None` 判断,库不存在就打印 `[SKIP] adapters 库未安装,跳过 optional adapters path` 后直接 `return`,不假装这条路径跑通了。**第三条**,`ia3_peft.py` 用 peft 库的 `IA3Config`,本知识点实测参数量同样是 `55,296`,和手写版本精确一致——但"参数总量一致"不等于"内部实现策略一致"。本文直接读了 `peft` 库源码(`.venv/Lib/site-packages/peft/tuners/ia3/layer.py`)确认了这一点:`IA3Layer.update_layer` 里,如果某个目标模块被标记为 `is_feedforward`,新增参数的形状是 `(1, in_features)`(缩放**输入**,在调用 base 层之前乘上去);否则形状是 `(out_features, 1)`(缩放**输出**,在 base 层算完之后乘)。`ia3_peft.py` 默认配置 `target_modules=["c_attn","c_proj","c_fc"], feedforward_modules=["c_fc"]`,逐层展开是:`c_attn` 非 feedforward → 缩放整个 `c_attn` 输出(`2304` 维,也就是 Q、K、V 三段全部一起缩放,不是只缩放 K、V);`attn.c_proj` 非 feedforward → 缩放 attention 输出投影(`768` 维,原始论文设计里没有这一项);`mlp.c_fc` 被标记为 feedforward → 缩放的是它的**输入**(`768` 维,而不是手写版本里缩放的"中间激活输出",维度也从 `3072` 变成了 `768`);`mlp.c_proj` 非 feedforward → 缩放 FFN 输出投影(`768` 维,原始论文设计里也没有这一项)。加总 `2304+768+768+768=4,608`,和手写版本的 `768+768+3072=4,608` 总数精确相等,但四个加数完全不是同一组张量、同一个位置——lecture(`lectures/05-ia3.md` Slide 11)把这个总数相等称为"偶然":向量长度不同、缩放的具体张量不同,只是数字上刚好加总相等,不是同一套设计的两份代码实现。

**AI 研究场景:** lecture(Slide 12)引用的论文在 T0/T0++ few-shot benchmark 上的实验数字是 LoRA r=4 均分 65.8(参数占比 0.18%)、(IA)³ 67.1(参数占比 0.02%)、全参数微调 67.6(100%)——(IA)³ 用不到 LoRA 1/9 的参数拿到比 LoRA 更高的分数,逼近全参数微调,这是论文在 few-shot 场景下的实验结论,本文没有条件独立复现下游任务指标,如实标注来源。(IA)³ 因为不含非线性(纯逐元素缩放),推理时可以把缩放向量直接乘进 base 权重(`W_K'=W_K⊙l_k`)完全合并,推理时延和原始模型一样,这一点和 LoRA 的"可合并"性质相同、和 Houlsby/Pfeiffer/Compacter/Parallel Adapter(内部都有非线性,无法无损合并)不同,这也是它在"参数量已经很小、还要求推理零开销"的场景(比如同一个 GPU 上要同时服务大量不同任务、每个任务只留一份 55K 的缩放向量)里具有实际吸引力的原因。

**可运行例子:**
```python
import sys
import torch
sys.path.insert(0, "learning/adapter-tuning-family/src")
from ia3_minimal import IA3GPT2
from ia3_peft import build_ia3_model_peft
from transformers import GPT2LMHeadModel

torch.manual_seed(42)
m = IA3GPT2()
n_m = sum(p.numel() for p in m.parameters() if p.requires_grad)

torch.manual_seed(42)
p_model = build_ia3_model_peft()
n_p = sum(p.numel() for p in p_model.parameters() if p.requires_grad)

assert n_m == n_p == 55_296                            # 路径 1(手写) vs 路径 3(peft)：总参数量一致

# 结构证据：layer 0 上具体缩放了哪些张量，minimal 和 peft 完全不同
blk0 = m.lm.transformer.h[0]
minimal_layer0 = {
    "l_k": blk0.attn.c_attn.l_k.numel(),
    "l_v": blk0.attn.c_attn.l_v.numel(),
    "l_ff": blk0.mlp.l_ff.numel(),
}
assert minimal_layer0 == {"l_k": 768, "l_v": 768, "l_ff": 3072}

peft_layer0 = {n: p.numel() for n, p in p_model.named_parameters() if p.requires_grad and ".h.0." in n}
assert sum(peft_layer0.values()) == sum(minimal_layer0.values()) == 4_608
# peft 的四个张量和 minimal 的三个张量，名字、个数、每个的维度都不一样
assert len(peft_layer0) == 4 and len(minimal_layer0) == 3

# 两条能跑的路径，初始化都精确等于 base（1 初始化 = 恒等变换）
base = GPT2LMHeadModel.from_pretrained("gpt2").eval()
m.eval(); p_model.eval()
enc = m.tokenizer("hello ia3 three paths", return_tensors="pt", padding=True)
with torch.no_grad():
    out_m = m(enc["input_ids"], enc["attention_mask"]).logits
    out_p = p_model(enc["input_ids"], enc["attention_mask"]).logits
    out_b = base(enc["input_ids"], attention_mask=enc["attention_mask"]).logits
assert torch.equal(out_m, out_b)
assert torch.equal(out_p, out_b)

# 路径 2：adapters 库，本机真实不可用，如实展示报错而不是假装数字
try:
    import adapters  # noqa: F401
    raise AssertionError("adapters 库竟然装上了，需要更新本知识点的诚实标注")
except ModuleNotFoundError as e:
    assert "adapters" in str(e)
    print("adapters 库路径真实报错:", repr(e))
```

实测(`.venv` 真跑):`n_m = n_p = 55,296`,总量一致;`minimal_layer0 = {'l_k': 768, 'l_v': 768, 'l_ff': 3072}`,`peft_layer0` 是 4 个独立张量(`attn.c_attn.ia3_l` 形状 `(2304,1)`、`attn.c_proj.ia3_l` 形状 `(768,1)`、`mlp.c_fc.ia3_l` 形状 `(1,768)`、`mlp.c_proj.ia3_l` 形状 `(768,1)`),两边总和都精确是 `4,608`,但张量个数(3 vs 4)、每个张量的维度、缩放的具体位置全部不同。`out_m`/`out_p` 和裸 GPT-2 的 `out_b` 都 `torch.equal`(逐 bit 相同)。`adapters` 库路径:`import adapters` 精确抛出 `ModuleNotFoundError: No module named 'adapters'`,直接运行 `python learning/adapter-tuning-family/src/ia3_adapters.py` 同样在 `from adapters import AutoAdapterModel, AdapterConfig` 这一行 fail-fast,和 `runbook.yaml`/`requirements.txt` 里的文档化说明完全吻合,不是本文编造的报错。

**面试怎么问 + 追问链:**
- **Q:** "(IA)³ 学的是矩阵还是向量,为什么这样设计参数量能降到这么低?"—— 期望说出"只学 3 个对角缩放向量(`l_k`/`l_v`/`l_ff`),不学任何投影矩阵,GPT-2 上总共只有 55,296 个参数,是这个系列里最小的方法之一"。
- **追问 1:** "(IA)³ 用全 1 初始化而不是全 0,是不是和 LoRA/Adapter 的'零初始化保证初始恒等'这个原则矛盾?"—— 期望说出"不矛盾,是同一个原则在不同运算下的体现:LoRA/Adapter 的扰动是加法(新增分支的输出要初始为 0 才能保证恒等),(IA)³ 的扰动是逐元素乘法(缩放向量要初始为 1 才能保证恒等),两者的目标完全一致,只是加法的零元是 0、乘法的零元是 1"。
- **追问 2(核心陷阱,考察是否真的比较过 peft 源码):** "手写版本和 peft 版本的 (IA)³ 参数总量都是 55,296,这是不是说明两边的实现是等价的?"—— 期望明确说"总量相等不等于实现等价,peft 默认配置缩放的是整个 c_attn 输出(包含 Q,不只是 K/V)、还多缩放了两个 c_proj 投影(原始论文没有这两项),FFN 那部分缩放的是 c_fc 的输入而不是中间激活输出;总数相等只是 GPT-2 具体维度凑出来的巧合,不是两种设计在数学结构上等价"。
- **追问 3:** "(IA)³ 的 `adapters` 库版本为什么在这个仓库里跑不通,这种情况在实际工作中怎么处理?"—— 期望说出"`adapters` 库要求 `transformers<5.0`,和仓库统一环境的 `transformers 5.10.2` 冲突,库根本没装,`import` 直接 `ModuleNotFoundError`";工程上合理的处理方式是像仓库测试文件那样用 `importlib.util.find_spec` 做可选依赖检测、优雅跳过,而不是让整个测试套件因为一个可选依赖缺失而全部失败,或者反过来强行降级核心依赖(`transformers`)去迁就一个次要功能。

**常见坑:** 看到"三条路径参数量都是 55,296"就直接下结论"这三条实现是一回事,随便挑一条学就够了"——本知识点已经用 peft 源码验证过,至少 minimal 和 peft 这两条路径缩放的是不同的张量、不同的位置,总量相等是 GPT-2 具体维度搭配出来的巧合(lecture 原话是"偶然"),不是设计上的等价,面试被追问"具体缩放了哪些张量"答不出来,说明只停留在"数字对上了"这个表面。另一个坑是把 `adapters` 库路径的失败归咎于"代码写错了"——`ia3_adapters.py` 这份代码本身没有问题,失败的原因纯粹是环境依赖冲突(`adapters` 要求的 `transformers` 版本上限低于仓库统一使用的版本),这是一个环境配置问题,不是这个知识点讲的算法或实现有 bug,遇到类似情况不应该去改仓库统一的 `transformers` 版本迁就一个次要依赖,而应该像仓库自己的选择一样,把它标记为可选、优雅跳过。
