# 04 · Adapter 进阶与统一视角深挖(Advanced Adapters & Unification)

> 总览见 [00-roadmap.md](00-roadmap.md)

本文是 `peft-deep-dive` 系列的第 4 个、也是最后一个文件,5 个知识点,源码在 `learning/adapter-tuning-family/src/{adapterdrop_minimal,mam_minimal,k_adapter_minimal,madx_minimal,adamix_minimal}.py`,配套 lecture 在同目录 `lectures/{03-adapterdrop-compacter,06-mam-adapter,07-k-adapter-mad-x,08-adamix,09-three-line-unification}.md`。这 4 段代码合起来讲的是同一件事的四个不同侧面——"固定的一个 adapter 结构不够用之后,还能在哪些维度上继续折腾":AdapterDrop 折腾的是"要不要用"(训练时随机丢、推理时永久丢,换速度);MAM 折腾的是"往哪插"(attention 端用 Prefix 风格,FFN 端用 Parallel 风格,两条线拼在一起);K-Adapter 和 MAD-X 折腾的是"怎么组合多个 adapter"(一个用求和,一个用串联,服务的目标也不同——前者是知识注入,后者是跨语言迁移);AdaMix 折腾的是"随机选而不是精心设计"(N 个 adapter 训练时随机路由到 1 个,推理时再想办法把 N 个合并回 1 个的成本)。最后一个知识点跳出 Adapter 家族本身,把 Prompt、LoRA、Adapter 三条主线拉到同一个公式下,给整条 `peft-deep-dive` 系列收尾。

**和 03 号文件(`03-adapter-core.md`,Adapter 家族核心,按 00-roadmap.md 规划撰写)的关系:** 03 号文件讲的是 adapter 这个构件本身的"地基"——原始 bottleneck adapter 的 `down→非线性→up→残差` 数学、Houlsby(每个 block 插 2 次:attn 后 + FFN 后)和 Pfeiffer(每个 block 只插 1 次:FFN 后)怎么选插入位置、AdapterFusion 怎么用 attention 融合多个冻结 adapter、Compacter 怎么用 PHM(Parameterized Hypercomplex Multiplication)把 down/up 矩阵换成 Kronecker 积压缩参数、Parallel Adapter 怎么把串联换成并联。本文默认你已经理解这个最基本的构件——下面要看的四种方法里,AdapterDrop、K-Adapter、MAD-X、AdaMix 全部直接 `from houlsby_minimal import HoulsbyAdapter` 复用同一个类(`down: nn.Linear(d,r)` → `act`(默认 GELU)→ `up: nn.Linear(r,d)` 零初始化 → `+x` 残差),不重新定义 adapter 内部长什么样,它们的差异全部来自"怎么组织多个/多层这样的构件",不是构件本身数学的差异——这是贯穿本文前 4 个知识点的一条暗线,读到知识点 4 结束会重新点出来。唯一的例外是知识点 2(MAM)的 FFN 端复用的是 `ParallelAdapter`(并联结构),不是 `HoulsbyAdapter`,这个区别本身也是知识点 5 统一公式里"组合方式"这根轴要讲的内容。

**一个重要的诚实标注:** 知识点 1/2/4(AdapterDrop/MAM/AdaMix)有配套 lecture,写作时参考了它们对机制、实验数字(如 AdapterDrop 论文的"k=5 时推理提速 ~25%"、AdaMix 论文在 GLUE 上的 +0.3 over full-FT)的讲解,但**所有可运行例子的数字都是本文用不同于 lecture/源文件自带 `main()`/smoke test 的输入(不同随机种子、不同文本、不同层)重新独立跑出来的**,凡是引用 lecture 里论文自己跑出来的 benchmark 数字,都会明确标注"来自 lecture,本文未独立复现"。知识点 3(K-Adapter/MAD-X)按仓库调研阶段已经发现的事实,如实标注了两个文件里几个"定义了但没被这个文件自己的 `main()` 用上"的 toy 数据常量,并且额外用 `inspect.getsource()` 做了结构性核实,不是凭印象转述。知识点 2 在验证 MAM 的近似实现时,额外用 autograd 挖出了一个 lecture 里完全没提到的细节(`PrefixAttention` 的 `P_k` 参数实测反传梯度只有约 1e-9 量级,基本不参与训练)——这不是哪个 lecture 或 roadmap 里写好的结论,是本文自己读代码 + 推数学 + 跑梯度实测出来的,如实记录;这不代表源码有 bug(源码运行完全正常,能正常训练收敛),只是这个特定的简化选择带来了一个作者大概率没有专门检查过的副作用。

**环境声明:** 本文涉及的 5 个源文件全部要构造一个真实的 GPT-2(`gpt2`,124M 参数)模型做前向传播(部分知识点还会做真实的梯度反传),不需要 GPU,CPU 上秒级到几秒跑完,模型从本地 Hugging Face 缓存加载,不需要联网。本文所有代码例子已在仓库根目录 `.venv`(Windows 原生,Python 3.13,torch 2.11.0+cu128,transformers 5.10.2)下用 `.venv/Scripts/python.exe` 实际跑通验证,文中数字是真实输出,不是手算或转述。

---

## 1. AdapterDrop(`adapterdrop_minimal.py`)—— 训练时随机丢、推理时永久丢

**是什么:**
```python
class _DroppableMlpWrapper(nn.Module):
    """可丢弃的 Adapter wrapper。

    支持两种 drop 策略:
        - training: 用 self.drop_prob 概率丢
        - inference: 用 self.permanent_drop bool 控制
    """

    def __init__(self, base_mlp, adapter, layer_idx: int):
        super().__init__()
        self.base_mlp = base_mlp
        self.adapter = adapter
        self.layer_idx = layer_idx
        self.drop_prob = 0.0          # 训练时随机丢概率
        self.permanent_drop = False   # 推理时永久丢

    def forward(self, x):
        h = self.base_mlp(x)
        if self.permanent_drop:
            return h
        if self.training and self.drop_prob > 0 and torch.rand(1).item() < self.drop_prob:
            return h
        return self.adapter(h)
```

`AdapterDropGPT2.set_inference_drop`(`adapterdrop_minimal.py:80-83`):
```python
def set_inference_drop(self, k: int) -> None:
    """推理时永久丢前 k 层 adapter。"""
    for i, block in enumerate(self.lm.transformer.h):
        block.mlp.permanent_drop = (i < k)
```

**一句话:** 同一个"丢",训练时和推理时是两套完全不同粒度的机制——训练时是每一层各自独立掷骰子(用 `drop_prob` 决定这一次前向要不要跳过这一层的 adapter),推理时是确定性地把"前 k 层"整体永久关掉,用来换推理速度。

**底层机制/为什么这样设计:** 先看训练时的分支:`if self.training and self.drop_prob > 0 and torch.rand(1).item() < self.drop_prob: return h`——这个判断在**每一层自己的 `forward` 里各跑一次**,`_DroppableMlpWrapper` 是每个 transformer block 各自持有一个独立实例(`adapterdrop_minimal.py:74-78` 的循环里,每个 `block` 都 new 了一个 `wrapper`),`torch.rand(1)` 每次调用都从全局 RNG 里各自抽样,所以一次前向传播里,12 层里哪几层被跳过是相互独立的伯努利试验,不是"整体丢一段连续的层"——这与 lecture(`lectures/03-adapterdrop-compacter.md` Slide 4/5)给出的伪代码 `for layer in transformer: if random() < drop_prob: skip` 完全一致,论文管这个叫"Standard AdapterDrop",本质是把 Dropout 的思想从"神经元级别"搬到了"整个 adapter 模块级别"——训练时随机让模型适应"adapter 缺失"的情况,提高鲁棒性,这是它和经典 Dropout 的相似之处。但推理时的分支完全是另一套逻辑:`set_inference_drop(k)` 是确定性地把 `block.mlp.permanent_drop` 在 `i < k` 时设成 `True`,而且是"前 k 层"(浅层)优先丢,不是随机丢、也不是丢深层——这个选择来自论文的实验观察(浅层 adapter 对最终效果的边际贡献小于深层,详见 lecture Slide 3),丢掉贡献最小的部分换取最大的速度收益(lecture 给出的经验数字是 k=5 时推理提速约 25%,这是论文的实验结论,本文没有条件独立测 wall-clock 速度,如实标注来源,不重新验证)。两套机制唯一共享的是同一个 `forward` 里的 if 结构,粒度、随机性、触发条件完全不同,不能混为一谈。另外从"构件"角度看,`AdapterDropGPT2.__init__` 里每层用的是 `HoulsbyAdapter(d, r)` 这个类(`adapterdrop_minimal.py:74-78`),而且只插了一次(挂在 `block.mlp` 上,没有对 `block.attn` 做类似处理),对照 03 号文件会讲到的 Houlsby(插 2 次)/Pfeiffer(插 1 次)划分标准,这其实是 Pfeiffer 布局——`AdapterDropGPT2` 的 docstring 也确实自称"GPT-2 + Pfeiffer Adapter + AdapterDrop 机制"(`adapterdrop_minimal.py:53-54`)。`HoulsbyAdapter` 这个类名只是"标准 down-act-up-残差构件"的命名,和插入几次的 Houlsby/Pfeiffer 之争是两个维度的事,读代码时容易因为类名而先入为主。

**AI 研究场景:** 多租户在线服务场景下(同一个基座模型要同时服务好几个下游任务/客户,每个任务挂一套 adapter),adapter 带来的推理延迟会随并发的 adapter 数量累积,AdapterDrop 允许在明确知道"浅层贡献有限"的前提下,用几乎不需要重新训练的方式(只要训练时就用了随机丢做过鲁棒性预备)拿到一个可调节的速度/质量折中点——用同一份训练好的权重,通过调整推理时的 `k`,不需要重新训练就能在"更快但精度略降"和"更慢但精度更高"之间切换,这对边缘部署、成本敏感的服务场景是有实际意义的工程手段。

**可运行例子:**
```python
import sys
import torch
sys.path.insert(0, "learning/adapter-tuning-family/src")
from adapterdrop_minimal import AdapterDropGPT2

torch.manual_seed(7)
model = AdapterDropGPT2(r=8, train_drop_prob=0.0)
model.train()

target_layer = model.lm.transformer.h[0].mlp
target_layer.base_mlp.eval()  # 只关掉 base_mlp 内部 resid dropout 的随机性，不影响 wrapper 自己的 training 标志
d = model.lm.config.n_embd
x = torch.randn(2, 5, d)

# drop_prob=1.0：训练模式下这一层应该 100% 跳过 adapter，输出精确等于 base_mlp(x)
target_layer.drop_prob = 1.0
with torch.no_grad():
    out_dropped = target_layer(x)
    out_base = target_layer.base_mlp(x)
assert torch.equal(out_dropped, out_base)

# drop_prob=0.0：训练模式下这一层应该 100% 不跳过，输出精确等于 adapter(base_mlp(x))
target_layer.drop_prob = 0.0
with torch.no_grad():
    out_full = target_layer(x)
    out_adapter = target_layer.adapter(target_layer.base_mlp(x))
assert torch.equal(out_full, out_adapter)

# 推理时的确定性永久丢——用和仓库自带测试(k=5/k=11)不同的 k 值
model.eval()
assert model.get_active_layers() == 12
model.set_inference_drop(k=3)
assert model.get_active_layers() == 9
dropped_flags = [model.lm.transformer.h[i].mlp.permanent_drop for i in range(12)]
assert dropped_flags == [True, True, True] + [False] * 9   # 精确是"前 3 层"，不是随机挑的 3 层
model.set_inference_drop(k=9)
assert model.get_active_layers() == 3
model.set_inference_drop(k=0)
assert model.get_active_layers() == 12
```

实测(`.venv` 真跑):两组训练时 drop_prob 边界(1.0/0.0)的输出严格 `torch.equal`,没有任何数值容差;推理时 `k=3` 精确让前 3 层的 `permanent_drop` 变 `True`、其余 9 层保持 `False`,`get_active_layers()` 正确返回 `9`,`k=9`→`3`,`k=0`→`12`,和仓库自带 `tests/test_adapterdrop_compacter.py::test_adapterdrop_inference` 用的 `k=5`/`k=11` 是不同的取值,交叉验证的是同一条逻辑。

**面试怎么问 + 追问链:**
- **Q:** "AdapterDrop 训练时的『丢』和推理时的『丢』是同一个机制吗?"—— 期望明确说"不是":训练时是每层独立的随机伯努利决策(每次前向都重新抽样),推理时是确定性地永久关闭"前 k 层"这一个固定集合。
- **追问 1:** "为什么推理时丢的是前 k 层(浅层),而不是随机丢或者丢最后几层?"—— 期望说出"论文实验发现浅层 adapter 对效果的边际贡献相对更小,丢浅层是'花最小的精度代价换最大的速度收益'这个决策的结果",不是随便选的。
- **追问 2(容易被类名带偏):** "`AdapterDropGPT2` 内部用的类叫 `HoulsbyAdapter`,但代码注释说这是 Pfeiffer Adapter,这矛盾吗?"—— 期望说清楚"`HoulsbyAdapter` 只是『down→act→up→残差』这个基础构件的类名,和插入几个位置(Houlsby 插 2 次 vs Pfeiffer 插 1 次)是两回事;`AdapterDropGPT2` 只在 `block.mlp` 插了一次,插入拓扑是 Pfeiffer 式的,只是复用了这个字面上叫 Houlsby 的构件类"。
- **追问 3(对比 Dropout):** "AdapterDrop 训练时的随机丢,和标准 Dropout 有什么本质相似和不同?"—— 期望说出"相似点是训练时随机、推理时确定性;不同点是 Dropout 随机作用在神经元粒度、每次前向的 mask 都不同,AdapterDrop 随机作用在'整个 adapter 模块要不要执行'这个粗得多的粒度"。

**常见坑:** 把训练时随机丢和推理时永久丢当成同一套参数在起作用——`drop_prob` 只在 `self.training=True` 时生效,`permanent_drop` 在任何模式下都优先生效(`if self.permanent_drop: return h` 在方法最前面),两者可以同时设置但语义完全独立。另外一个是本文写验证代码时真实踩过的坑:直接调用两次 `base_mlp(x)` 想比较输出是否一致,却忘了 GPT-2 的 MLP 自带 `resid_pdrop` Dropout,训练模式下两次独立前向天然不会 bit-identical(和 adapter 有没有被丢完全无关),必须显式 `.eval()` 掉 `base_mlp` 那一层才能做严格的数值比较——这个坑本质上和 AdapterDrop 无关,但会让人误以为 drop 逻辑本身有 bug。

---

## 2. MAM(`mam_minimal.py`)—— Prefix 风格 attention 的一个不完整近似 + Parallel 风格 FFN

**是什么:**
```python
class PrefixAttention(nn.Module):
    """Prefix-style attention 注入 (MAM attn 端).

    在 K, V 前拼接 learnable prefix vectors:
        K' = [P_k; K], V' = [P_v; V]
    这等价于"在每个 query token 上加入一个 attention 偏置项"。
    """

    def __init__(self, c_attn: nn.Module, d: int, prefix_len: int = 30):
        super().__init__()
        for p in c_attn.parameters():
            p.requires_grad = False
        self.c_attn = c_attn
        self.d = d
        self.prefix_len = prefix_len
        self.P_k = nn.Parameter(torch.zeros(prefix_len, d))
        self.P_v = nn.Parameter(torch.zeros(prefix_len, d))
        nn.init.normal_(self.P_k, std=0.01)
        nn.init.normal_(self.P_v, std=0.01)

    def forward(self, x):
        """简化策略：把 P_k, P_v 加到 K, V 的 attention bias 上（等价但更简单）。
        实际 paper 是 prepend，需要修改 attention mask。这里用 attention bias 近似。
        """
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d, dim=-1)
        v_bias = self.P_v.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, d)
        v = v + v_bias
        k_bias = self.P_k.mean(dim=0, keepdim=True).unsqueeze(0)
        k = k + k_bias
        return torch.cat([q, k, v], dim=-1)
```
(`mam_minimal.py:36-74`,docstring 与实现均为原文引用)

**一句话:** 代码自己的注释写的是"简化策略……等价但更简单",但实测下来这不是"等价",是一个明显更粗糙的近似——它把 `prefix_len` 根 learnable 向量直接取均值坍缩成 1 个常数,再原样加到每个 token 的 K、V 上,完全没有真正拼接进注意力序列,也没有实现 lecture 里证明的那个依赖 query 的混合系数。

**底层机制/为什么这样设计:** 先看"近似"具体丢了什么。真正的 Prefix Tuning 是 `K'=[P_k;K], V'=[P_v;V]`——序列维度上真的多出 `prefix_len` 个可以被 attend 的位置,这些 prefix 位置要和真实 token 一起参与 softmax 竞争。`lectures/09-three-line-unification.md`(Slide 5)和 `lectures/06-mam-adapter.md`(Slide 4)给出的严格推导是:这样做的效果可以展开成 `attn(Q,K',V') = (1-λ(Q))·attn(Q,K,V) + λ(Q)·attn(Q,P_k,P_v)`,其中 `λ(Q) = Σexp(QP_k^T) / [Σexp(QP_k^T)+Σexp(QK^T)]` 是一个**依赖 query 内容**的混合系数——不同的 query 会以不同比例"参考"prefix。而这份 `mam_minimal.py` 的实现完全没有做这件事:`forward` 里 `qkv = self.c_attn(x)` 之后直接对 `k`、`v` 各加一个常数(`P_k.mean(dim=0)`/`P_v.mean(dim=0)`,形状 `(1,1,d)`,靠 broadcast 复制到每个 batch、每个 token 位置),序列长度没有变(`qkv.shape[1] == x.shape[1]`,不是 `x.shape[1] + prefix_len`),这个偏置也完全不依赖 Q——是全体 token 共享的同一个常量,和 lecture 证明的 `λ(Q)` 机制在数学结构上是两回事。更进一步——本文用 autograd 实测发现了一个连"近似"这个定性描述都不完全准确的细节:因为 softmax 对"给所有 logit 加同一个常数"是不敏感的(`softmax(z+c) = softmax(z)`),而 `k_bias` 对某个固定 query 位置来说,加到每一个 key 位置上的偏移量 `Q·k_bias` 是同一个值(不随 key 的位置 `i` 变化),所以 `k_bias` 这部分理论上对 attention 权重分布完全不产生影响,只有 `v_bias` 才会真正加到最终输出上。本文构造了一个独立于 GPT2Attention 内部实现的手写因果自注意力(不依赖 `mam_minimal.py` 之外任何 transformers 内部细节),让 `PrefixAttention` 的输出流过这个手写 attention 再算 loss、再反传,实测 `P_k` 的梯度绝对值最大值只有约 `7.5e-9`(`torch.allclose(P_k.grad, 0, atol=1e-6)` 为 `True`),而 `P_v` 的梯度绝对值最大值约 `0.985`,量级差了 8 个数量级;在跑完整 `MAMGPT2`(12 层)的真实前向 + 反传后同样成立(所有层 `P_k.grad` 绝对值最大约 `1.0e-9`,`P_v.grad` 约 `0.081`)。也就是说 `PrefixAttention` 里的 `P_k` 这组参数,在这份实现的数学结构下**基本不参与训练**——它确实被算进了可训练参数总量(`prefix_len=30` 时,12 层的 `P_k` 一共 `276,480` 个参数,占 MAM 总可训练参数 `857,280` 的 `32.25%`),却几乎收不到任何有意义的梯度信号。这不是本文瞎猜的结论,是数学推导(softmax 的平移不变性)加真实反传实验双重验证过的,源码的注释("等价但更简单")没有提到这个副作用,如实记录,不代表源码运行有问题(它确实能正常训练、正常收敛,只是 `P_k` 这部分实际上是陪跑)。

**AI 研究场景:** MAM 论文本身的核心论点是"组合优于单一方法"——lecture 引用的 XSum 实验里(`lectures/06-mam-adapter.md` Slide 9),Prefix attn + Parallel FFN 的组合(ROUGE-L 36.4)比单独用 Prefix(35.7)或单独用 Parallel(35.6)都更好,这类"往不同的注入位置分别塞不同风格的扰动"的设计思路在今天的多模态 PEFT(比如 LLaMA-Adapter 在 attention 端加 zero-init gating、在其他位置加 Adapter)里仍然常见——但本知识点同时也是一个工程提醒:论文的组合收益是在"忠实实现"prefix 机制的前提下测出来的,如果为了图省事把 prefix 机制简化成一个常数偏置(就像这份 minimal 代码做的那样),会实质性地削弱甚至废掉这一半设计的贡献,复现论文数字时如果偷懒近似了机制本身,拿不到论文声称的效果提升是完全可以预期的,不是"调参没调好"。

**可运行例子:**
```python
import torch
import torch.nn as nn
import sys
sys.path.insert(0, "learning/adapter-tuning-family/src")
from mam_minimal import PrefixAttention, MAMGPT2
from transformers import GPT2LMHeadModel

# 第一部分：单独验证 PrefixAttention 的"近似"具体做了什么
torch.manual_seed(3)
d, prefix_len = 16, 5
c_attn = nn.Linear(d, 3 * d)
pa = PrefixAttention(c_attn, d, prefix_len=prefix_len)

x = torch.randn(1, 4, d)
qkv_out = pa(x)
q_out, k_out, v_out = qkv_out.split(d, dim=-1)
q_raw, k_raw, v_raw = pa.c_attn(x).split(d, dim=-1)

assert torch.allclose(q_out, q_raw)                      # Q 完全没被 prefix 影响
k_bias = pa.P_k.mean(dim=0, keepdim=True).unsqueeze(0)
v_bias = pa.P_v.mean(dim=0, keepdim=True).unsqueeze(0)
assert torch.allclose(k_out, k_raw + k_bias)
assert torch.allclose(v_out, v_raw + v_bias)
assert k_bias.shape == (1, 1, d)                          # 关键结构证据：偏置在 seq 维度上是单例，天然与 token 位置无关
assert qkv_out.shape[1] == x.shape[1]                     # 序列长度没变——不是真的 prepend
assert qkv_out.shape[1] != x.shape[1] + prefix_len

# 第二部分：MAM 的"初始 forward ≈ base"到底有多不精确
torch.manual_seed(11)
model = MAMGPT2(prefix_len=30, r=16)
model.eval()
base = GPT2LMHeadModel.from_pretrained("gpt2").eval()
enc = model.tokenizer("deep learning is fascinating", return_tensors="pt", padding=True)
with torch.no_grad():
    out_mam = model(enc["input_ids"], enc["attention_mask"]).logits
    out_base = base(enc["input_ids"], attention_mask=enc["attention_mask"]).logits
diff_default = (out_mam - out_base).abs().max().item()
assert diff_default > 1e-3          # 不是精确恒等——因为 P_k/P_v 不是零初始化(std=0.01)

with torch.no_grad():
    for block in model.lm.transformer.h:
        block.attn.c_attn.P_k.zero_()
        block.attn.c_attn.P_v.zero_()
    out_mam_zeroed = model(enc["input_ids"], enc["attention_mask"]).logits
diff_zeroed = (out_mam_zeroed - out_base).abs().max().item()
assert diff_zeroed == 0.0           # 强制清零 P_k/P_v 后，才精确等于 base
assert diff_zeroed < diff_default / 100

# 第三部分：上面"底层机制"一段说 P_k 的梯度接近零，这是 softmax 平移不变性的直接推论——
# 这里现场手写一个独立于 GPT2Attention 内部实现的单头因果自注意力，把这句话变成可以自己重跑的代码，
# 不是只在正文里引用一个数字
import torch.nn.functional as F

def _causal_self_attention(q, k, v):
    """极简单头因果自注意力，只依赖 q/k/v 的形状 (batch, seq, d)，不借用 transformers 内部任何细节。"""
    dh = q.shape[-1]
    seq = q.shape[1]
    scores = (q @ k.transpose(-1, -2)) / (dh ** 0.5)
    causal_mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ v

torch.manual_seed(31)
pa3 = PrefixAttention(nn.Linear(d, 3 * d), d, prefix_len=prefix_len)
x3 = torch.randn(2, 6, d)
q3, k3, v3 = pa3(x3).split(d, dim=-1)
out3 = _causal_self_attention(q3, k3, v3)
loss3 = F.mse_loss(out3, torch.randn_like(out3))
loss3.backward()

pk_grad_max = pa3.P_k.grad.abs().max().item()
pv_grad_max = pa3.P_v.grad.abs().max().item()
assert pk_grad_max < 1e-6                                 # P_k 的梯度是浮点噪声量级，不是"很小但仍在学习"
assert pv_grad_max > 1e-3                                 # P_v 的梯度是正常量级
assert pv_grad_max / pk_grad_max > 1e5                     # 至少差 5 个数量级，不是同一量级的大小差异
```

实测(`.venv` 真跑):第一部分的全部结构性断言通过——`PrefixAttention` 给 K/V 加的偏置张量形状精确是 `(1,1,d)`,靠 broadcast 复制到每个 token,序列长度全程保持 `4`,没有变成 `4+5=9`。第二部分:默认初始化下 `diff_default = 1.920242e-01`(文本 `"deep learning is fascinating"`,种子 11);作为交叉验证,`mam_minimal.py` 自己的 `main()`(文本 `"hello world"`,种子 42)给出的是 `diff = 1.5141e-01`,同样是非平凡量级,不是本文这次跑的巧合。作为对照,`houlsby_minimal.py::main()` 里同样的"初始 forward vs base"检验给出的是精确的 `0.0000e+00`(因为 Houlsby/Parallel 用的是精确零初始化,MAM 的 Prefix 侧不是)。把 12 层的 `P_k`/`P_v` 手动清零后,`diff_zeroed` 精确为 `0.0`,验证了非零 diff 确实来自 Prefix 侧,不是 Parallel FFN 侧(它本来就是零初始化)。第三部分,`|P_k.grad|` 最大值 `5.005859e-10`(`torch.allclose(P_k.grad, 0, atol=1e-6)` 为 `True`),`|P_v.grad|` 最大值 `2.233610e-02`,两者比值约 `4.46e7`——差了超过 7 个数量级,`P_k` 的梯度是浮点舍入噪声的量级,不是"很小但仍在被训练"。这份手写因果自注意力的具体实现细节(有没有除以 `√d`、loss 用什么形式、GPT-2 自身的 attention/residual dropout 会不会额外消耗一点随机数状态)不需要和某一次特定验证的具体小数位完全对上——不同的合理实现给出的具体数值会有差异,但只要 `k_bias` 在 softmax 之前被加到每个 key 位置、且这个偏移量不随 key 的位置变化,`P_k` 的梯度就会被 softmax 的平移不变性结构性地压到浮点噪声量级,这是这份手写验证真正要钉死的结论,不是某一组具体小数。

**面试怎么问 + 追问链:**
- **Q:** "`mam_minimal.py` 里的 `PrefixAttention` 和真正的 Prefix Tuning 是一回事吗?"—— 期望说"不是同一回事,是一个近似实现",能具体说出"没有真的拼接进 K/V 序列,而是把 prefix 向量取均值变成一个常数偏置"。
- **追问 1(数学,呼应 lecture 严格证明):** "真正的 Prefix Tuning 数学上等价于 `(1-λ(Q))attn(Q,K,V)+λ(Q)attn(Q,P_k,P_v)`,这里的 `λ(Q)` 依赖 query。这份近似实现加的偏置依不依赖 query?"—— 期望说出"不依赖,偏置张量形状是 `(1,1,d)`,对所有 batch、所有 token 位置都一样,和 `λ(Q)` 完全不是一回事"。
- **追问 2(全场最深的追问,验证是否真读了代码):** "K 侧和 V 侧的偏置,对最终输出的影响是不是对称的?"—— 期望候选人能推出(或者被引导推出)"不对称:K 侧的常数偏置在 softmax 之前对每个 key 位置加了同一个值,而 softmax 对'所有 logit 加同一常数'不敏感(shift-invariant),所以 K 侧偏置理论上不影响 attention 权重;V 侧偏置是直接线性加到加权求和之后的输出上,货真价实地改变了结果"。有条件的话可以追问"能不能验证",引出 autograd 实测:`P_k.grad` 约 1e-9(噪声量级),`P_v.grad` 约 0.08-0.98(正常量级)。
- **追问 3:** "如果要修好这个近似,方向应该往哪改?"—— 无标准答案,考察工程直觉,合理方向包括"真的把 P_k/P_v 拼接进 K/V 序列,让它们参与 softmax 竞争"或者"至少让偏置的权重依赖 Q(比如学一个从 Q 到偏置强度的小映射)",单纯加一个常数偏置在数学上撑不起"prefix"这个名字要表达的机制。

**常见坑:** 望文生义,相信代码注释里"等价但更简单"这句话是字面意思的"数学等价"——实测初始 forward 与 base 的差异高达 `0.19`(对比 Houlsby/Parallel 精确为 `0.0` 的零初始化),差了好几个数量级,不是"约等于恒等"。更深一层的坑是即使发现了偏置的存在,也想当然地认为 `P_k`、`P_v` 两组参数"地位对称、都在正常训练"——这一点单看 `nn.Parameter` 声明和初始化代码完全看不出来,必须动手做梯度实测(或者推一遍 softmax 平移不变性的数学)才能发现 `P_k` 基本是陪跑的死参数,这是"读代码"和"读代码 + 推数学 + 跑实验"两个理解深度之间的真实差距。

---

## 3. K-Adapter / MAD-X(`k_adapter_minimal.py` / `madx_minimal.py`)—— 两种组合多个 adapter 的方式,以及几个没被用上的 toy 常量

**是什么:**
```python
# k_adapter_minimal.py:42-61（节选，完整各 10/5 条）
TOY_FACTUAL_TRIPLES = [
    "paris is the capital of france",
    "berlin is the capital of germany",
    # ... 共 10 条
]
TOY_LINGUISTIC_DATA = [
    "the cat sat on the mat",
    "dogs chase cats around the yard",
    # ... 共 5 条
]

class _KMlpWrapper(nn.Module):
    """挂载多个 K-Adapter，输出相加。"""
    def forward(self, x):
        h = self.base_mlp(x)
        delta = sum(a(h) - h for a in self.adapters)  # K-Adapter: 多类知识 adapter 的增量相加
        return h + delta
```

```python
# madx_minimal.py:38-42
TOY_MULTILINGUAL_DATA = {
    "en": ["hello world", "i love this", "absolutely beautiful day"],
    "de": ["hallo welt", "ich liebe das", "absolut schoner tag"],
    "fr": ["bonjour monde", "j aime ca", "journee absolument belle"],
}

class _MADXMlpWrapper(nn.Module):
    """MAD-X 的核心：每 block 嵌入 LA + TA。"""
    def forward(self, x):
        h = self.base_mlp(x)
        h = self.language_adapters[self.active_language](h)   # 先过 language adapter
        h = self.task_adapters[self.active_task](h)            # 再过 task adapter
        return h
```

**一句话:** K-Adapter 和 MAD-X 都是"用多个同样结构(`HoulsbyAdapter`)的 adapter 服务不同目的"这条思路的两个变体,区别在组合方式——K-Adapter 求和(多类知识各自贡献一份增量、加在一起),MAD-X 串联(先过语言 adapter 再过任务 adapter,顺序不可交换);而两个源文件顶部各自定义的 toy 数据常量,实际使用情况并不一致,必须逐个核实,不能一概而论。

**底层机制/为什么这样设计:** K-Adapter 的设计目标是知识注入——`factual`(训于 Wikidata 类三元组)、`linguistic`(训于依存句法)等不同类型的知识各自封装成一个独立的 `KAdapter`(内部就是一个 `HoulsbyAdapter`),`_KMlpWrapper.forward`(`k_adapter_minimal.py:133-137`)里 `delta = sum(a(h) - h for a in self.adapters)` 把每个 adapter 相对输入的"增量"加总,再统一加回 `h`——这样设计的好处是每类知识 adapter 可以独立训练、独立冻结(`MultiKnowledgeGPT2.freeze_adapter`,`k_adapter_minimal.py:116-122`,只冻结指定 `knowledge_type` 的那一个,不影响其他类别),知识之间是加法关系,理论上可以任意增减类别而不用重新设计结构。MAD-X 的设计目标是跨语言迁移——`_MADXMlpWrapper.forward`(`madx_minimal.py:141-146`)里是 `h = language_adapters[lang](h); h = task_adapters[task](h)`,顺序执行(先语言后任务),这是**串联**不是求和,而且顺序有意义:因为 `HoulsbyAdapter.forward` 内部有 GELU 非线性,两个非线性函数复合一般不满足交换律(`TA(LA(h)) ≠ LA(TA(h))`),这也是为什么 `set_active(language, task)` 可以自由换语言换任务、但换的是"选哪一个语言 adapter/任务 adapter",不是换执行顺序——LA→TA 这个顺序在 `_MADXMlpWrapper.forward` 里是硬编码的。这样设计的好处是语言和任务解耦:只训一次 `TA_ner`(配合 `LA_en`),推理时换成 `LA_de` 就能做零样本德语 NER,不需要为每个"语言×任务"组合单独训练一份。

现在讲如实标注的部分——**必须先用 `grep`/`inspect.getsource()` 核实,不能凭印象转述**:`k_adapter_minimal.py` 顶部的 `TOY_LINGUISTIC_DATA`(5 条句子,`:55-61`)在整个仓库范围内(`grep -rn "TOY_LINGUISTIC_DATA" learning/`)只在定义它的那一行出现过,没有任何地方(包括这个文件自己的 `main()`、任何测试、任何 notebook)引用过它,是彻底的死代码。`TOY_FACTUAL_TRIPLES`(10 条三元组,`:42-53`)情况不同:它没有被 `k_adapter_minimal.py` 自己的 `main()` 用到(`inspect.getsource(k_adapter_minimal.main)` 里搜不到这个名字),但确实被 `learning/adapter-tuning-family/src/tests/test_k_adapter_madx.py::test_mini_training` 拿去做了一次真实的 10 步训练(`enc = tok(TOY_FACTUAL_TRIPLES[:4], ...)`),notebook `notebooks/07-k-adapter-mad-x.ipynb` 里也用它的前 6 条做了 20 步训练并画了 loss 曲线——所以它不是"没用",只是没有被这个特定文件自己的演示入口用上。`madx_minimal.py` 的 `TOY_MULTILINGUAL_DATA`(en/de/fr 三种语言各 3 句,`:38-42`)是最彻底的"名不副实":它被 `madx_minimal.py` 自己的 `main()`、`tests/test_k_adapter_madx.py`、notebook 三个地方各 import 了一次,但三处都没有真正索引/读取过这个字典的内容——实际的语言切换演示用的是硬编码字符串 `"hello world"`/`"hallo welt"`,不是这个字典里的数据。

本文额外发现了一个和 toy 数据同一性质、但更值得注意的细节:MAD-X 论文的第三类 adapter——invertible adapter(处理 embedding 层的语言差异)——在 `MADXGPT2.__init__` 里确实被创建了(`self.invertible_adapters = nn.ModuleDict({lang: InvertibleAdapter(d) for lang in languages})`,`madx_minimal.py:103-105`),也确实被计入可训练参数总量(3 种语言 × 2 个参数(`mu`/`log_sigma`)× 768 维 = 4,608 个参数,精确对应 `main()` 打印的"Invertible adapters: 3 × 2 × 768 = 4,608"),但翻遍 `MADXGPT2.forward()`(`madx_minimal.py:117-123`,只调用了 `self.lm(...)`)和 `_MADXMlpWrapper.forward()`(只碰 `language_adapters`/`task_adapters`),没有任何一行代码真正调用过 `invertible_adapters` 的 `forward`/`inverse`——本文直接把这 3 个 invertible adapter 的参数改成标准差 100 的随机数再跑一遍前向,`logits` 输出和改之前逐 bit 相同(`torch.equal` 为 `True`),证明它们确实完全没有接入这份 minimal 实现的计算图。`InvertibleAdapter` 类本身的数学是对的(仓库 notebook 里单独把它拎出来做过 forward→inverse 往返验证,误差在 `1e-5` 以内),只是没有被接到 `MADXGPT2` 的前向链路上——这是"构件本身没问题,但没有被组装进最终系统"的一个具体例子。

**AI 研究场景:** K-Adapter 这种"多知识 adapter 求和"的组合方式,对应的是需要把多个独立训练的、语义上正交的能力叠加到同一个基座模型上的场景(医疗问答场景可能想同时叠加 factual 和 medical 两类知识 adapter);MAD-X 的"语言/任务解耦"思路,对应的是需要在有限标注资源下做跨语言迁移的场景(只有英语的 NER 标注数据,但需要支持德语、法语)——lecture(`lectures/07-k-adapter-mad-x.md` Slide 9)引用的论文 claim 是这种 zero-shot 迁移能达到监督训练效果的 80% 左右,这是论文的实验结论,本文没有条件独立复现下游任务指标,如实标注来源。这两种组合思路今天在"多 LoRA 切换"(不同任务/客户挂不同的 LoRA 权重,inference 时按需加载)这类工程实践里仍然能看到影子——本质上都是"冻结主干、按需插拔不同的小模块"这条 PEFT 最基本的设计哲学的具体应用。

**可运行例子:**
```python
import inspect
import sys
sys.path.insert(0, "learning/adapter-tuning-family/src")
import k_adapter_minimal
import madx_minimal
import torch

# 1. 两个 toy 常量确实存在，并核实条数
assert len(k_adapter_minimal.TOY_FACTUAL_TRIPLES) == 10
assert len(k_adapter_minimal.TOY_LINGUISTIC_DATA) == 5

# 2. k_adapter_minimal.py 自己的 main() 源码里，两个常量的名字都没出现过
main_src = inspect.getsource(k_adapter_minimal.main)
assert "TOY_FACTUAL_TRIPLES" not in main_src
assert "TOY_LINGUISTIC_DATA" not in main_src

# 3. MAD-X 的 toy 常量同样没被 madx_minimal.py 自己的 main() 用到
assert set(madx_minimal.TOY_MULTILINGUAL_DATA.keys()) == {"en", "de", "fr"}
madx_main_src = inspect.getsource(madx_minimal.main)
assert "TOY_MULTILINGUAL_DATA" not in madx_main_src

# 4. InvertibleAdapter 被创建、被计入参数量，但 forward 链路里完全没有调用它
forward_src = inspect.getsource(madx_minimal.MADXGPT2.forward)
wrapper_src = inspect.getsource(madx_minimal._MADXMlpWrapper.forward)
init_src = inspect.getsource(madx_minimal.MADXGPT2.__init__)
assert "invertible" not in forward_src.lower()
assert "invertible" not in wrapper_src.lower()
assert "self.invertible_adapters" in init_src   # 确实被创建（计入参数量）

# 5. 决定性验证：把 invertible adapter 参数改成天文数字，模型 logits 完全不变
torch.manual_seed(5)
m = madx_minimal.MADXGPT2(r=16, languages=("en", "de", "fr"), tasks=("ner",))
m.eval()
enc = m.tokenizer("paris is beautiful", return_tensors="pt")
with torch.no_grad():
    out1 = m(enc["input_ids"], enc["attention_mask"]).logits
with torch.no_grad():
    for ia in m.invertible_adapters.values():
        ia.mu.data.normal_(mean=0.0, std=100.0)
        ia.log_sigma.data.normal_(mean=0.0, std=100.0)
with torch.no_grad():
    out2 = m(enc["input_ids"], enc["attention_mask"]).logits
assert torch.equal(out1, out2)   # 参数改成天文数字，输出一个 bit 都不变

n_total = sum(p.numel() for p in m.parameters() if p.requires_grad)
n_ia = sum(p.numel() for ia in m.invertible_adapters.values() for p in ia.parameters())
assert n_ia == 3 * 2 * 768
assert n_total == 1_221_888   # 但这 4,608 个参数确实被计入可训练参数总量
```

实测(`.venv` 真跑):5 组断言全部通过。第 5 组是最直接的证据——把 `en`/`de`/`fr` 三个 invertible adapter 的 `mu`/`log_sigma` 全部重新采样成标准差 `100` 的随机数(几乎不可能和原来的小值凑巧抵消),`out1`/`out2` 两次 `logits` 依然 `torch.equal`;而这 3 个 invertible adapter 一共贡献 `4,608` 个参数,精确占 MAD-X 总可训练参数 `1,221,888` 的一部分,是被"计了数但没接线"的典型情况。K-Adapter 侧:实测冻结 `factual` 类别后,可训练参数从 `608,640` 精确减半到 `304,320`,和公式 `2 × 12 × 25,360 = 608,640` 完全吻合。

**面试怎么问 + 追问链:**
- **Q:** "K-Adapter 和 MAD-X 组合多个 adapter 的方式分别是什么?"—— 期望说出"K-Adapter 求和(多类知识各自贡献增量相加),MAD-X 串联(先语言后任务,顺序执行)"。
- **追问 1:** "MAD-X 的 LA→TA 顺序换成 TA→LA,数学上等价吗?"—— 期望说"不等价,因为两个 adapter 内部都有 GELU 这类非线性,非线性函数复合一般不可交换";这也是 lecture 思考题里直接问到的点,答不出说明只记住了结论没理解原因。
- **追问 2(如实标注,核心考点):** "这两个文件顶部定义的 toy 数据,是不是都完全没被用上?"—— 期望精确区分三种情况,不能一概而论:`TOY_LINGUISTIC_DATA` 全仓库任何地方都没引用过;`TOY_FACTUAL_TRIPLES` 没被这个文件自己的 `main()` 用,但在测试文件和 notebook 里做了真实训练;`TOY_MULTILINGUAL_DATA` 在三个地方都被 import 了,但没有一处真正读取过它的内容。答成"三个都完全没用"或者"三个都在什么地方用到了"都是不准确的。
- **追问 3(深挖,MAD-X 的第三类 adapter):** "MAD-X 论文讲了三类 adapter(LA/TA/IA),这份实现里三类都真正参与推理了吗?"—— 期望说"没有,invertible adapter(IA)被创建、计入参数量,但整个 `MADXGPT2` 的前向链路里没有任何地方调用它,可以用'把它的参数改成随机数、模型输出完全不变'来验证"。

**常见坑:** 把"文件顶部定义了就等于被使用"当成默认成立,或者反过来一刀切地说"这文件全是没用的摆设"——三个 toy 常量、以及 `InvertibleAdapter` 这个类,实际情况各不相同(彻底没用 / 没被这个文件自己用但别处真用了 / 被 import 了但从没被真正读取过 / 被创建计入参数量但没接入计算图),必须逐个用 `grep`/`inspect` 核实,如实说清楚每一个具体是哪种情况,不要为了图省事合并成一句笼统的结论,也不要强行解释成"这是留给读者的练习"这类没有依据的过度解读。

---

## 4. AdaMix(`adamix_minimal.py`)—— 训练时随机选 1 个 expert,推理时求平均或直接合并

**是什么:**
```python
class AdaMixLayer(nn.Module):
    """Mixture of N adapter experts。

    训练: 每个 forward 随机选一个 expert
    推理: 所有 expert 输出取平均
    """

    def __init__(self, d: int, r: int = 16, n_experts: int = 4):
        super().__init__()
        self.experts = nn.ModuleList([
            HoulsbyAdapter(d, r) for _ in range(n_experts)
        ])
        self.n_experts = n_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            idx = torch.randint(0, self.n_experts, (1,)).item()
            return self.experts[idx](x)
        else:
            outs = [e(x) for e in self.experts]
            return torch.stack(outs).mean(dim=0)
```

`merge_experts()`(`adamix_minimal.py:103-119`):
```python
def merge_experts(self) -> None:
    """推理优化：把所有 expert 平均成单个 expert（论文最终步骤）。"""
    for block in self.lm.transformer.h:
        adamix = block.mlp.adamix
        for attr in ("down", "up"):
            weights = torch.stack([
                getattr(e, attr).weight for e in adamix.experts
            ]).mean(dim=0)
            biases = torch.stack([
                getattr(e, attr).bias for e in adamix.experts
            ]).mean(dim=0)
            getattr(adamix.experts[0], attr).weight.data.copy_(weights)
            getattr(adamix.experts[0], attr).bias.data.copy_(biases)
        adamix.experts = nn.ModuleList([adamix.experts[0]])
        adamix.n_experts = 1
```

**一句话:** AdaMix 的"随机路由"比"Mixture"这个名字容易让人联想到的 MoE 更粗糙——`idx = torch.randint(...)` 每次 `forward` 调用只抽一次,整批输入(不管 batch 里有几条、每条有多少 token)全部走同一个 expert,不是逐 token 路由;而 `merge_experts()` 的权重平均和推理时的输出平均,因为 adapter 内部有 GELU 非线性,并不是精确等价的两种"平均"。

**底层机制/为什么这样设计:** 先看训练:`if self.training: idx = torch.randint(0, self.n_experts, (1,)).item(); return self.experts[idx](x)`——`torch.randint` 只调用一次,产生的 `idx` 是一个标量,直接用来索引 `self.experts[idx]`,而 `x` 是完整的 `(batch, seq, d)` 张量,一次性整个喂给这一个被选中的 expert。这意味着 AdaMix 的路由粒度是"每次 `forward` 调用"(近似等价于"每个训练 step"),不是像 Switch Transformer / Mixtral 那样"每个 token 各自决定去哪个 expert"——lecture(`lectures/08-adamix.md` Slide 3)自己也说"不需要 gating network、不需要 load balance loss",这正是因为路由粒度粗到不需要这些机制:反正每次只选 1 个 expert 处理全部输入,没有"某些 expert 被喂太多 token、某些几乎没被用到"这种负载不均衡的问题。这样设计的直接好处是训练速度和单 adapter 相同(每个 step 只算 1 个 expert 的前向 + 反传),不需要额外的路由网络参数。再看推理:`self.training=False` 分支是把 N 个 expert 的**输出**分别算出来再取平均(`torch.stack(outs).mean(dim=0)`),计算成本是 N 倍;`merge_experts()` 则是把 N 个 expert 的**权重**(`down`/`up` 的 weight 和 bias)先平均,再用这 1 个"平均出来的 expert"做前向,计算成本降回单 adapter 水平。这两者不是同一件事:`HoulsbyAdapter.forward` 是 `x + up(GELU(down(x)))`,GELU 是非线性函数,`mean_i(GELU(down_i(x)))` 一般不等于 `GELU(mean_i(down_i(x)))`——"先算完非线性再平均"和"先平均权重再算非线性"只有在整个函数是线性的情况下才严格相等,lecture(`lectures/08-adamix.md` Slide 9)也直接承认"严格上不等价……但实验上误差很小(<0.1% on GLUE)"。本文用故意构造得比真实训练结果更"离散"的随机专家权重做了一次数值验证:让 4 个 expert 的 `down`/`up` 权重独立随机初始化(标准差 0.3,比真实训练收敛后专家之间通常更接近的情况更极端),"输出平均"和"权重平均"两种做法在同一批输入上给出的最大逐元素差异达到约 `2.0`——差异的绝对大小很大程度上取决于专家之间差异有多大(真实训练出来的专家通常比这里故意构造的随机专家更接近,论文报告的 GLUE 差距在 0.1% 以内),但"两者在机制上不精确等价"这件事本身是数学结构决定的,和专家差异大小无关,是 `merge_experts()` 用可控的精度损失换取推理成本从 N× 降回 1× 的必然代价。另外一个容易被忽略的细节:`HoulsbyAdapter.__init__` 里 `up` 层永远是零初始化,所以刚构造出来的 `AdaMixLayer` 里,不管 `down` 层的随机初始化把 N 个 expert 分得多开,N 个 expert 的**输出**在初始化那一刻全部相同(都精确等于恒等映射 `x + 0 = x`)——这时候不管随机路由选中哪一个,结果都一样,"随机路由让训练产生差异"这件事要等训练把各个 expert 的 `up` 层训出实质差别之后才会真正体现出来,本文在下面的可运行例子里为了能观察到"确实选中了某一个 expert"这个现象,特意手动打破了默认的零初始化。

**AI 研究场景:** AdaMix 提供的是"不需要设计精细的 MoE 路由器,也能拿到一部分 mixture-of-experts 式收益"的低成本方案——lecture 引用的论文实验(`lectures/08-adamix.md` Slide 10)显示 AdaMix(N=4,merge 后参数量和单个 adapter 一样)在 GLUE 上甚至超过全参数微调,这是论文的实验结论,本文没有条件独立复现下游任务指标。它在工程上的取舍很清晰:训练阶段接受 N 倍参数量(但不接受 N 倍算力,因为每 step 只算 1 个 expert)换潜在的效果提升,推理阶段用 `merge_experts()` 的近似把参数量和算力都收回到单 adapter 水平——这种"训练时放大、推理时收拢"的思路,在其它需要"训练多个候选方向、推理前合并/精简掉大部分"的 PEFT 工程实践里也能看到类似的设计精神。

**可运行例子:**
```python
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "learning/adapter-tuning-family/src")
from adamix_minimal import AdaMixLayer

# 第一部分：路由粒度是"整次 forward"，不是逐 token
torch.manual_seed(9)
d, r, n = 32, 8, 4
layer = AdaMixLayer(d, r, n_experts=n)
with torch.no_grad():
    for e in layer.experts:
        e.up.weight.normal_(std=0.1)   # 打破零初始化，否则 4 个 expert 初始输出全相同，测不出"选中了谁"
layer.train()

x = torch.randn(2, 6, d)
torch.manual_seed(100)
out = layer(x)
with torch.no_grad():
    matches = [i for i in range(n) if torch.equal(out, layer.experts[i](x))]
assert len(matches) == 1   # 整个 (batch=2, seq=6) 张量精确命中同一个 expert，不是逐 token 各选各的

# 第二部分：merge_experts 的"权重平均" vs eval() 的"输出平均"，GELU 非线性下不精确等价
torch.manual_seed(21)
layer2 = AdaMixLayer(d, r, n_experts=n)
with torch.no_grad():
    for e in layer2.experts:
        e.down.weight.normal_(std=0.3)
        e.down.bias.normal_(std=0.1)
        e.up.weight.normal_(std=0.3)
        e.up.bias.normal_(std=0.1)
layer2.eval()
x2 = torch.randn(3, 5, d)
with torch.no_grad():
    out_avg_outputs = torch.stack([e(x2) for e in layer2.experts]).mean(dim=0)
    down_w = torch.stack([e.down.weight for e in layer2.experts]).mean(dim=0)
    down_b = torch.stack([e.down.bias for e in layer2.experts]).mean(dim=0)
    up_w = torch.stack([e.up.weight for e in layer2.experts]).mean(dim=0)
    up_b = torch.stack([e.up.bias for e in layer2.experts]).mean(dim=0)
    h = F.gelu(F.linear(x2, down_w, down_b))
    out_merged_weights = F.linear(h, up_w, up_b) + x2   # 手动复现 merge_experts() 的做法

diff = (out_avg_outputs - out_merged_weights).abs().max().item()
assert diff > 1e-3   # 两种"平均"不是精确相等
```

实测(`.venv` 真跑):第一部分,`(batch=2, seq=6, d=32)` 的整个输出张量精确命中 4 个 expert 里的第 `0` 号一个(`len(matches) == 1`),验证路由是"整批数据共用一个 expert",不是逐 token。第二部分,故意构造的、比真实训练结果更离散的随机 expert 权重下,"输出平均"和"权重平均"两种做法给出的最大逐元素差异是 `1.989861e+00`——差异的具体大小是本文这组人为放大的随机初始化的产物,不代表真实训练场景下的典型差距(lecture 引用论文的 GLUE 实验差距在 0.1% 量级),但这个数字本身足以证明"两种平均方式不是同一个函数",这一点不随专家差异大小而改变,是 GELU 非线性决定的结构性事实。

**面试怎么问 + 追问链:**
- **Q:** "AdaMix 的『随机路由』,粒度是每个 token 独立选 expert,还是整个 batch 共用一个 expert?"—— 期望说"整次 `forward` 调用共用同一个 expert",不是逐 token 路由,和 Switch Transformer/Mixtral 那种细粒度 MoE 不是一回事。
- **追问 1:** "这样粗粒度的路由,相比逐 token 路由,好处和代价分别是什么?"—— 期望说出"好处是实现简单、不需要 gating 网络、不需要 load balance loss、训练速度等于单 adapter;代价是没法让不同 token 用不同专长的 expert,牺牲了路由的灵活性换训练效率"。
- **追问 2(深挖 merge_experts 的精确性):** "`merge_experts()` 权重平均后的单专家,和不 merge、直接对 N 个专家的输出取平均,结果完全一样吗?"—— 期望说"不完全一样,因为 adapter 内部有 GELU 非线性,`mean(GELU(·))` 一般不等于 `GELU(mean(·))`,只有 f 是线性函数时两者才严格相等";如果候选人能主动提到"论文报告实验上差距很小(<0.1%)但机制上不是精确等价"体现出更完整的理解。
- **追问 3:** "如果不做 `merge_experts()`,直接推理时对 N 个 expert 的输出取平均,计算成本是多少?"—— 期望说"是单 adapter 的 N 倍(还是要跑 N 次 down/up 投影),`merge_experts()` 就是为了把这个 N 倍成本换回 1 倍,用可控的精度损失换推理效率"。

**常见坑:** 把 AdaMix 的"Mixture"直接类比成 Switch Transformer/Mixtral 那种逐 token gating 的 MoE——读代码会发现路由粒度粗得多(整次 forward 抽一次)。另一个坑是想当然地认为 `merge_experts()` 之后的结果和"老老实实对 N 个专家输出取平均"完全一致——这是把线性运算的性质(平均和函数可以交换顺序)错误套用到了含有 GELU 的非线性 adapter 上。还有一个容易被忽略的初始化细节:因为 `HoulsbyAdapter` 的 `up` 层永远零初始化,`AdaMixLayer` 刚构造出来时 N 个 expert 的输出全部相同(都等于恒等映射),此时随机路由选中哪个 expert 在数值上没有任何区别,必须等训练把 `up` 层训出差异后,"随机路由"才会真正影响到 loss——如果不注意这一点,拿一个刚初始化、完全没训练过的 `AdaMixLayer` 去验证"路由生效了没有",会得到"看起来路由没生效"的误导性结论,其实只是还没开始训练。

---

## 5. Prompt + LoRA + Adapter 三线统一公式(`lectures/09-three-line-unification.md`)—— 一个公式收编 28 种方法

**是什么:** 不是新代码,是 `learning/adapter-tuning-family/lectures/09-three-line-unification.md`(Slide 3)给出的一个统一公式,对 Transformer 里任意一个 functional unit(attention 或 FFN):

`h ← h + Δh`,`Δh = f(W_down · x) · W_up`

差异只在 4 个轴上(原文 Slide 3):
1. **`f`**:是 identity、非线性激活 `σ`、还是 attention 里那种 softmax
2. **`W_down, W_up`**:满秩矩阵、对角矩阵、跨层共享、还是 PHM(Compacter 那种 Kronecker 分解)
3. **位置**:插在 attention 的 K/V、attention 的输出、还是 FFN
4. **组合方式**:并联、串联、还是残差缩放

**一句话:** Prompt、LoRA、Adapter 三条主线在"设计一个新 PEFT 方法"这件事上,本质上都是在这 4 个轴上选一组取值——LoRA 是"`f`=identity 的并联 Adapter",`(IA)³` 是"`W_down`/`W_up` 退化成对角阵的 Adapter",Prefix Tuning 是"通过 attention 机制注入、数学上等价于并联 Adapter"的 Prompt 类方法——理解这一个公式,大致相当于理解了半个 PEFT 领域。

**底层机制/为什么这样设计:** 这个公式不是本文提出的,是 He et al. 2022(《Towards a Unified View of Parameter-Efficient Transfer Learning》,ICLR)的核心论点,`lectures/06-mam-adapter.md` 和 `lectures/09-three-line-unification.md` 都完整讲过推导,这里只讲落到今天四个知识点、以及呼应 LoRA 家族(`learning/lora-family/`)和 Prompt 家族(`learning/prompt-tuning-family/`)时分别是怎么对号入座的,并且用代码而不是公式重新验证其中最核心的一条等价关系。先看 LoRA ≡ 去非线性的 Parallel Adapter 这条(lecture Slide 6):`ParallelAdapter.forward`(`parallel_minimal.py:70-73`,03 号文件已详细讲过)是 `base(x) + s·up(σ(down(x)))`,`LoRALinear.forward`(`lora_minimal.py:67-73`)是 `base(x) + s·x@A^T@B^T`——把 `σ` 换成恒等函数、把 `A` 对应到 `down.weight`、`B` 对应到 `up.weight`、再去掉 `down`/`up` 各自的 bias 项(纯矩阵乘法本身没有偏置),两者的扰动项在代数上是同一个式子。本文没有停留在"看着像"就下结论,而是直接构造了一个 `ParallelAdapter` 实例(手动把 `act` 换成 `nn.Identity()`、清零 `down`/`up` 的 bias),和一个用完全相同 `down.weight`/`up.weight` 初始化的 `LoRALinear` 实例,在同一批随机输入上比较两者相对 `base(x)` 的扰动量——精确相等(逐元素误差 `0.0`)。再额外做了一个对照组:只把 `act` 换回默认的 `GELU()`,其他权重不变,扰动量立刻和 LoRA 产生明显差异(最大逐元素差 `0.36`)——反过来证明了"是否为 identity"确实是这条等价关系成立与否的开关,不是巧合。

Prompt 家族这条线,`(1-λ(Q))attn(Q,K,V)+λ(Q)attn(Q,P_k,P_v)` 这条严格证明(lecture 09 Slide 5)针对的是"真正"实现了 K/V 拼接的 Prefix Tuning——本仓库里这个"真正"版本就是 `learning/prompt-tuning-family/src/prefix_tuning_minimal.py::PrefixTuningGPT2`(类定义在 `:32`),它用 transformers 的 `DynamicCache` 真的把 prefix 向量塞进 `past_key_values`(`:26` 导入,`:85` 起的 `get_past_key_values` 方法),让 prefix 位置切切实实地参与 softmax 竞争——对比之下,知识点 2 里验证过的 `mam_minimal.py::PrefixAttention` 只是对这条数学关系的一次不完整的工程近似(常数偏置,没有拼接,没有 `λ(Q)` 的 query 依赖),这也是为什么知识点 2 用梯度实测发现 `P_k` 基本不参与训练——一旦把"真正拼接进序列参与 softmax"这个关键步骤简化掉,这条数学等价关系赖以成立的机制基础也跟着不成立了,产生副作用不算意外。`(IA)³` 这条(lecture Slide 7)则是把 `W_down` 设成对角矩阵 `diag(ℓ_k - 1)`、`W_up` 设成 `α·I`、`f`=identity——本质是"矩阵退化成对角阵"的 LoRA/Adapter,只学缩放,不学投影方向,这也是为什么 03 号文件会讲到它是这批方法里参数量最小的几个之一。

到这里,整条 `peft-deep-dive` 系列(4 个文件、约 24 个知识点)按 00-roadmap.md 的规划走完了两条技术路线:01/02 号文件是 LoRA 家族(LoRA 本身的数学与初始化变体、量化 + LoRA 的组合),03/04 号文件是 Adapter 家族(03 号文件的核心构件与经典变体,04 号文件——也就是本文——的进阶用法与统一视角)。这个统一公式回过头去看,恰好说明了为什么这两条路线可以放在同一个系列里深挖:它们不是两种互不相关的技术,LoRA 本来就是"去掉非线性的并联 Adapter"这一个特例,04 号文件今天讲的 MAM、AdaMix 这类"进阶"方法,也都是在这 4 个轴上做排列组合和工程取舍(MAM 是"attention 端用一种位置/组合方式,FFN 端用另一种"的混搭;AdaMix 是在"`W_down`/`W_up`"这根轴上从"1 组矩阵"变成"N 组矩阵 + 随机选择机制")。

**AI 研究场景:** 这套统一视角的实际价值不是"告诉你该用哪个方法"(工程选型仍然要回到显存、延迟、任务类型这些具体约束,`lectures/09-three-line-unification.md` Slide 10 的决策树本身也依赖大量论文经验性结论,本文不重复背诵),而是给"读新论文/设计新方法"提供一个检查清单——遇到一个自称全新的 PEFT 方法,先问它在 `f`/矩阵结构/位置/组合方式这 4 个轴上分别选了什么取值,往往能发现它其实是已有方法在某个轴上的重新排列组合(VeRA ≈ 共享投影 + 对角化的 `(IA)³`,DoRA ≈ LoRA + 幅度方向分解),而不是从零发明的新机制——这对快速评估一篇新论文"增量在哪"、面试被问到"你怎么看待层出不穷的 PEFT 新论文"这类开放问题时,是一个比"每个方法单独死记硬背"更省力、更能举一反三的框架。

**可运行例子:**
```python
import torch
import torch.nn as nn
import sys
sys.path.insert(0, "learning/adapter-tuning-family/src")
sys.path.insert(0, "learning/lora-family/src")
from parallel_minimal import ParallelAdapter
from lora_minimal import LoRALinear

torch.manual_seed(17)
d, r = 12, 4
base = nn.Linear(d, d)
for p in base.parameters():
    p.requires_grad = False

# 把 Parallel Adapter 的非线性去掉、清零 bias（LoRA 没有偏置项，这样比较才公平）
padapter = ParallelAdapter(base, r=r, scaling=1.0)
padapter.act = nn.Identity()
with torch.no_grad():
    padapter.down.bias.zero_()
    padapter.up.weight.normal_(std=0.1)   # LoRA 的 B 默认零初始化输出恒为 0，这里给个非零值才能比出"形状"是否一致
    padapter.up.bias.zero_()

# 用完全相同的 down/up 权重构造一个 LoRALinear
lora = LoRALinear(base, r=r, alpha=r)   # alpha=r -> scaling=1.0，与上面对齐
with torch.no_grad():
    lora.A.copy_(padapter.down.weight)
    lora.B.copy_(padapter.up.weight)

x = torch.randn(3, 7, d)
delta_parallel = padapter(x) - base(x)
delta_lora = lora(x) - base(x)
assert torch.allclose(delta_parallel, delta_lora, atol=1e-5)   # f=identity 时，两者的扰动量精确一致

# 对照组：恢复 GELU 非线性，同样的权重，扰动量立刻不再相等
padapter_gelu = ParallelAdapter(base, r=r, scaling=1.0)  # 默认 act=GELU()
with torch.no_grad():
    padapter_gelu.down.weight.copy_(padapter.down.weight)
    padapter_gelu.down.bias.zero_()
    padapter_gelu.up.weight.copy_(padapter.up.weight)
    padapter_gelu.up.bias.zero_()
delta_parallel_gelu = padapter_gelu(x) - base(x)
diff_gelu = (delta_parallel_gelu - delta_lora).abs().max().item()
assert diff_gelu > 1e-2   # 一旦恢复非线性，就不再等价于 LoRA
```

实测(`.venv` 真跑):`f=identity` 时 `delta_parallel` 与 `delta_lora` 的最大逐元素差异是 `0.000000e+00`,精确相等;把同一组 `down`/`up` 权重套进带 GELU 的 `ParallelAdapter` 后,和 `delta_lora` 的最大差异变成 `3.623471e-01`——不是"约等于",是数量级上明确不同,直接证明了"`f` 是否为 identity"是这条等价关系是否成立的唯一开关,其余权重、scaling、bias 处理方式全部保持不变。

**面试怎么问 + 追问链:**
- **Q:** "为什么说 LoRA 是『去掉非线性的 Parallel Adapter』?"—— 期望说出两者都是 `base(x)+s·up(f(down(x)))` 形式,LoRA 是 `f=identity` 的特例,并且知道这不只是"长得像",而是可以用代码直接验证扰动量精确相等。
- **追问 1:** "`(IA)³` 在这个统一公式里对应什么取值?"—— 期望说出"`W_down` 退化为对角矩阵,`f=identity`,`W_up` 也是恒等映射,本质是只学缩放向量、不学投影方向的极端压缩版本"。
- **追问 2(呼应 Prompt 家族与知识点 2):** "Prefix Tuning 属于统一公式里的哪条线,和本文知识点 2 讲的 MAM 的 `PrefixAttention` 是什么关系?"—— 期望说出"Prefix Tuning 数学上可以证明等价于在 attention 输出上做一次依赖 query 的加性扰动;MAM 的 `PrefixAttention` 想复刻这个效果,但简化成了一个不依赖 query 的常数偏置,是对这条数学关系的不完整近似,这也是知识点 2 里 `P_k` 梯度实测接近零的根本原因"——能不能把两个知识点串起来讲,是判断是否真正理解、而不是分别记忆两个孤立知识点的关键。
- **追问 3(工程判断力):** "这套统一公式对实际选型有帮助吗,还是纯理论?"—— 期望说出"直接帮助有限(选型还要看显存/延迟/任务这些经验约束),但对'评估一个新论文的增量在哪'非常有用——先检查它在 4 个轴上是不是已有方法的重新组合,这是一个可以系统化套用的检查清单,而不是每个方法孤立地死记硬背"。

**常见坑:** 把"数学上可以统一"误解成"效果上都一样,随便选一个就行"——统一视角说的是扰动的函数形式可以纳入同一个公式,不代表不同取值在具体任务上的实际效果、训练稳定性、参数量、显存开销是等价的(MAM 参数量比单独 LoRA/Adapter 都贵,本文知识点 2 也验证过它这种"混搭"在忠实实现和粗糙近似之间还有实打实的效果差距)。另一个坑是把"统一公式证明的严格等价关系"(比如 Prefix Tuning ≡ Parallel Adapter,针对的是"真正"实现了 K/V 拼接的版本)和"某个具体 minimal 实现对这个机制的近似"混为一谈——本文知识点 2 已经用真实梯度实验证明,`mam_minimal.py` 里的简化版并没有完整复刻理论上的等价关系,读到统一公式时如果直接假设"反正数学上证明过等价,代码肯定也是等价的",会忽略掉工程实现里可能存在的取巧和信息损失。
