# 04 · 数据工程与 Capstone 深挖(Data Engineering & Capstone)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一批是本系列的收官批次,性质和前三批不太一样:01(RoPE 外推家族)、02(长上下文 Attention 架构)、03(长上下文评测方法论)讲的都是"某一个孤立的技术组件怎么工作",这一批要做的是把这些组件串成一条更接近真实工程的链路——前两个知识点讲长文本预训练数据怎么组织(打包 + 课程学习),第三个知识点是一个把 01 的 YaRN、以及 LoRA 串起来的 capstone 脚本(**但要先破除一个误解:这不是一个能跑出真实训练结果的完整脚本**,本篇最重要的任务之一就是讲清楚它的边界在哪里),第四个知识点回到一个几乎每个长上下文工程师都会被问到的显存核算问题:KV cache 为什么会比模型权重本身还夸张。

**和 `learning/long-context/` 的关系(简短重复,完整版见 [00-roadmap.md](00-roadmap.md)):** 本文是 `learning/long-context/src/long_data_packing.py`、`capstone_yarn_llama32.py` 和 `lectures/12-long-context-pitfalls.md` 的精读伴读笔记,不是重新发明一套讲法。每个例子都要求能在 `.venv` 里独立跑通并用 `assert` 验证,不是转述源码或课件——尤其是"packing_efficiency 到底是多少""KV cache 到底算出来是多少 GB"这类具体数字,本篇全部现场跑一遍 Python 再写进文档,不采信任何"看起来眼熟"的预设数字。

**本篇统一结构(七步,与 00-roadmap.md / torch-deep-dive 完全一致):**
1. 签名/是什么
2. 一句话
3. **底层机制/为什么这样设计** —— 不停在"怎么用",讲到"为什么必须是这样"
4. AI 研究场景
5. 可运行例子(带 `assert`,真在仓库 `.venv` 里跑过)
6. **面试怎么问 + 追问链** —— 面试官大概率怎么问,追问会往哪个方向深挖
7. 常见坑

本文所有代码例子已在仓库根目录 `.venv`(torch 2.11.0+cu128、transformers 5.10.2、peft 0.19.1,纯 CPU、不下载任何模型权重)下实际跑通验证;凡是给出具体数字的地方(装箱效率、KV cache 显存 GB 数),都是现场跑出来的结果。

---

## 1. `pack_documents` / `concat_with_lens` / `make_doc_mask` —— 装箱打包与块对角 attention mask

**是什么:**
```python
# learning/long-context/src/long_data_packing.py
def pack_documents(docs: list, max_len: int = 8192) -> list: ...        # 11-34行
def concat_with_lens(batch: list) -> tuple: ...                          # 37-44行
def make_doc_mask(doc_lens: list, total_len: int = None) -> torch.Tensor: ...  # 47-59行
```
`pack_documents` 把一批长度参差不齐的"文档"(这里的 doc 就是一串 token id 的 `list`)重新组织成若干个"打包批次"(batch),每个批次拼起来的总长度不超过 `max_len`;`concat_with_lens` 把一个打包批次里的所有文档首尾拼接成一条 token 序列,同时记下每篇文档各自的长度(`doc_lens`);`make_doc_mask` 根据这些长度,生成一个形状 `(total_len, total_len)` 的布尔矩阵,值为 `True` 的位置表示"这两个 token 属于同一篇文档,允许互相 attend"。

**一句话:** `pack_documents` 先把超过 `max_len` 的单篇文档切成若干块,再用 first-fit-decreasing(FFD)装箱算法把这些块尽量塞满每个训练序列槽位,减少 padding 浪费;`make_doc_mask` 再补上一层"文档边界"限制,防止拼在同一条序列里的不同文档在 attention 时互相看见对方,产生不该有的虚假关联。

**底层机制/为什么这样设计:**

预训练语料里文档长度天然参差不齐(有的 100 token,有的几千 token)。如果每条训练序列严格对应一篇文档、不够 `max_len` 就 padding 补齐,短文档场景下 padding 占比可能高达 90% 以上,大量算力被浪费在无意义的 padding token 上。Packing 的思路是把多篇文档首尾相连拼进同一个 `max_len` 序列,让每一个 token 位置都被真实数据占满。

`pack_documents` 具体分两步:
1. **先切块:** 单篇文档如果本身就超过 `max_len`(比如一本书、一段很长的代码文件),不管怎么装箱都塞不进单个槽位,必须先按 `max_len` 切成若干段,最后一段是余数(比如 9500 长度、`max_len=4000`,会被切成 `[4000, 4000, 1500]` 三块,已在下面的例子里验证)。
2. **再装箱(FFD):** 把所有块(包括没被切过的正常文档)按长度**降序**排序,依次尝试放进已有的、放得下的第一个箱子,放不下就开一个新箱子。"先排序再装箱"不是随手加的一步——如果不排序、来什么装什么(退化成普通 first-fit),同样的数据装箱效率会明显更差(下面的例子会用同一份数据现场对比:FFD 是 98.75%/2 个箱子,不排序的 first-fit 是 65.83%/3 个箱子)。直觉上,先放"大件"能让后面陆续出现的"小件"更容易见缝插针地填补空隙。这是 bin packing(装箱问题)的经典近似算法——bin packing 本身是 NP-hard,FFD 不保证全局最优,但是一个足够简单、效果足够好的近似方案。

至于为什么必须要 `make_doc_mask`:普通的 causal mask 只保证"看不到未来",不保证"看不到别的文档"。如果不加文档边界限制,拼接后序列里后面文档的 token 会正常地"看到"前面不相关文档的所有 token(因为它们在物理序列里排在"过去"),模型会被迫学到"文档 A 的结尾和文档 B 的开头存在某种因果关系"这种纯属拼接产生的噪声关联。`make_doc_mask` 生成的是一个**块对角(block-diagonal)** 矩阵——在原有 causal mask 之上再叠加一层"必须同属一篇文档"的限制,真正跑训练时两层 mask 是 `&` 在一起用的(块对角 `&` 下三角)。

**AI 研究场景:** 预训练数据 pipeline 里这一步几乎是标配,通常被称为"sample packing"或"sequence packing"——千卡集群跑数月的训练里,吞吐量(tokens/sec)每提升几个百分点都是真金白银,padding 造成的浪费在这种规模下不可忽视。生产实现为了避免这里教学代码里显式物化一个 `(total_len, total_len)` 布尔矩阵带来的 `O(n²)` 显存开销,通常不会真的构造这么大的 mask,而是用类似 FlashAttention 2 的 `flash_attn_varlen_func` 那样,只传一个一维的"累积序列长度"数组(`cu_seqlens`,即 `doc_lens` 的前缀和)告诉 attention kernel 每篇文档的边界在哪,由 kernel 内部处理,不需要在显存里真正保存这个巨大的布尔矩阵。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/long-context/src")
from long_data_packing import (
    pack_documents, concat_with_lens, make_doc_mask, packing_efficiency,
)

# --- 用文件自带的示例数据跑一遍 pack_documents(不预设任何数字,跑出来什么就是什么) ---
docs = [list(range(n)) for n in [100, 500, 2000, 3000, 1500, 800]]
batches = pack_documents(docs, max_len=4000)

assert len(batches) == 2
assert [len(d) for d in batches[0]] == [3000, 800, 100]   # 实测:batch0 总长 3900
assert [len(d) for d in batches[1]] == [2000, 1500, 500]  # 实测:batch1 总长 4000
assert sum(len(d) for d in batches[0]) == 3900
assert sum(len(d) for d in batches[1]) == 4000

eff = packing_efficiency(batches, max_len=4000)
assert abs(eff - 0.9875) < 1e-9   # 实测 98.75%,不是预设数字

# --- 对比:如果不排序,退化成普通 first-fit,效率会明显更差(现场验证,不是猜的) ---
def first_fit_no_sort(docs, max_len):
    out = []
    for d in docs:
        for b in out:
            if sum(len(x) for x in b) + len(d) <= max_len:
                b.append(d); break
        else:
            out.append([d])
    return out

ff = first_fit_no_sort(docs, max_len=4000)
ff_eff = packing_efficiency(ff, max_len=4000)
assert len(ff) == 3                        # 3 个箱子,比 FFD 多开 1 个
assert abs(ff_eff - 0.6583333333) < 1e-6    # 65.83%,比 FFD 的 98.75% 明显更差

# --- 验证"超过 max_len 的单文档会被切块"这条规则(示例数据本身不触发,自己造一个) ---
long_doc = list(range(9500))
chunks = sorted(len(d) for b in pack_documents([long_doc], max_len=4000) for d in b)
assert chunks == [1500, 4000, 4000]   # ceil(9500/4000)=3 块,最后一块是余数,总 token 数不丢
assert sum(chunks) == 9500

# --- make_doc_mask 块对角验证:doc_lens=[3,4] ---
mask = make_doc_mask([3, 4])
assert mask.shape == (7, 7)
assert mask[0:3, 0:3].all()          # doc0(占[0,3))内部全 True
assert mask[3:7, 3:7].all()          # doc1(占[3,7))内部全 True
assert not mask[0:3, 3:7].any()      # doc0 看不到 doc1
assert not mask[3:7, 0:3].any()      # doc1 看不到 doc0
assert mask.sum().item() == 3*3 + 4*4   # 只有两个对角块是 True,其余全 False

# --- 和 concat_with_lens 串起来,在真实 batch 上再验证一次 ---
tokens, doc_lens = concat_with_lens(batches[0])
assert doc_lens == [3000, 800, 100] and tokens.shape[0] == 3900
mask_full = make_doc_mask(doc_lens)
assert mask_full[0:3000, 0:3000].all()
assert not mask_full[0, 3500].item()   # doc0 的 token 确实看不到 doc2 的 token

print("packing_efficiency:", eff, "| no-sort first-fit:", ff_eff)
```
本机实测:FFD 打包 98.75%(2 个批次),不排序的 first-fit 只有 65.83%(3 个批次)——这个差距完全是"先排序再装箱"这一步带来的,直接印证了上面"底层机制"里的解释。

**面试怎么问 + 追问链:**
- **Q:** "预训练数据里长度不一的文档,你会怎么组织进固定长度的训练序列?"—— 期望说出 packing(打包拼接)思路,而不是"每条 pad 到 max_len"。
- **追问 1:** "如果不加处理直接把多篇文档拼接进同一个序列,会有什么问题?"—— 期望说出"causal attention 会让后面的文档看到前面不相关文档的内容,产生虚假的跨文档关联",而不是只停留在"浪费 token"这个层面。
- **追问 2(区分度很高):** "block-diagonal mask 在实现上要注意什么?如果序列长度是 3900,这个 mask 占多少显存?生产系统真的会这样实现吗?"—— 期望能现场估算:`torch.bool` 每个元素占 1 字节,`3900×3900` 的 mask 占 `3900×3900×1 ≈ 15.2MB`(已实测,一个 batch 就要 15MB,batch size 一大、序列一长会迅速膨胀),更好的答案是提到生产实现会用 `cu_seqlens` 这类一维前缀和数组交给 attention kernel 处理,不会真的物化这个 `O(n²)` 矩阵。

**常见坑:** 装箱前忘记按长度降序排序,直接顺序(first-fit)装箱——效率会明显下降,上面例子里同一份数据从 98.75% 掉到 65.83%,箱子数从 2 个变成 3 个,不是理论上的小差异,是实打实的吞吐损失;只记得拼接 token,忘了同步生成并使用 `doc_mask`,这类 bug 不会在 loss 曲线上表现得多明显(训练照样能收敛),却会让模型学到"跨文档幻觉关联"这种很隐蔽的质量问题,不容易在训练阶段发现。还有一处更隐蔽的边界行为:`make_doc_mask(doc_lens, total_len=...)` 在 `total_len` 大于 `sum(doc_lens)`(比如打包完还留了 padding 位置)时,超出 `sum(doc_lens)` 的那部分行/列会**整行全 False**,连对角线(自己看自己)都是 `False`——已现场验证,`make_doc_mask([3, 4], total_len=10)` 之后 `mask[7:10, :].any()` 是 `False`,连 `mask[7, 7]` 也是 `False`。如果不做特殊处理直接把这种 mask 喂给 softmax,padding 位置会因为整行全 `False`(等价于全部加 `-inf`)导致 softmax 输出 `NaN`——用这个函数时如果 `total_len` 传了比实际 token 数更大的值,要自己额外处理 padding 区域的 mask,不能假设它"开箱即用"。

---

## 2. `curriculum_lengths` / `filter_by_curriculum` —— 4 阶段长度课程学习

**是什么:**
```python
# learning/long-context/src/long_data_packing.py
def curriculum_lengths(stage: int) -> tuple: ...           # 62-70行,stage → (min_len, max_len)
def filter_by_curriculum(docs: list, stage: int) -> list: ...  # 73-76行,按当前 stage 过滤文档长度
```

**一句话:** 训练早期只用短文档(序列短、显存省、吞吐高、训练更稳定),随着 `stage` 递增逐步放开允许的文档长度区间,让模型先在短序列上把基础能力打扎实,再逐步"毕业"到更长的序列,而不是一上来就直接扔最长的文档进去训练。

**底层机制/为什么这样设计:**

直接一上来就用目标长度(比如 128k)训练至少有三个问题:① 显存——activation 显存随序列长度增长(具体增长速度取决于是否用 FlashAttention 等优化,但方向上肯定是变长就更吃显存);② 吞吐——同样的 GPU-小时预算下,短序列能跑更多有效 step,模型见到的"梯度更新次数"更多;③ 训练稳定性——模型还没学会怎么利用长距离信息之前,长序列里大部分位置的信号可能是噪声,极端位置编码在模型没充分见过的区间还容易导致梯度不稳定(呼应 lecture 12 Slide 11 提到的"长 ctx fine-tune 训练崩溃"问题)。

再看 `curriculum_lengths` 实际给出的 4 个阶段:`(256,2048)` / `(2048,8192)` / `(8192,32768)` / `(32768,131072)`——每一段的上界正好等于下一段的下界,是一条无缝衔接、**每档放大 4 倍**的等比数列。选择指数增长而不是线性增长是有道理的:处理长度这件事本身的计算成本是随长度超线性增长的(序列长度翻倍,标准 attention 的计算量翻 4 倍),线性增长在长度已经很大时几乎没有意义(从 100000 到 101000,模型很难感知出差异;但从 8000 到 32000,难度和成本差异巨大)。

`filter_by_curriculum` 只是把 `curriculum_lengths` 返回的 `(lo, hi)` 区间套用到"筛选符合当前阶段的训练数据"这个动作上,逻辑很直白:只保留长度落在 `[lo, hi]` 之间的文档。但这也是一处**教学简化**——它是"硬切换"(某个 stage 只用这个区间的数据,其余数据完全不用),真实训练更常见的做法是 lecture 11 Slide 6 描述的"渐进式配比"(不同阶段短/中/长数据都保留一部分,只是比例逐渐向长序列倾斜,比如 base 阶段短文本占 80%、长上下文阶段降到 30%),而不是某个阶段完全没有短数据——完全切断短数据反而可能让模型在长序列阶段"忘掉"部分短序列上已经学好的能力。

**AI 研究场景:** 这个"先短后长"的思路在真实长上下文扩展工作里很常见——YaRN 论文和 lecture 11 Slide 10 描述的做法是"先用 PI 调到 32k、全量微调约 1B token,再扩到 128k、继续微调约 0.5B token",本质上就是一个两阶段的长度课程。下一个知识点(capstone 脚本)里的 `curriculum_max_len(step)` 是同一个思想在"训练 step"维度的具体应用——不是按数据集划分阶段,而是按训练进度直接决定"当前允许的最大序列长度",和这里按数据集切分阶段是同一个"渐进式增长"思路的两种不同实现方式。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/long-context/src")
from long_data_packing import curriculum_lengths, filter_by_curriculum

# --- 4 个阶段的真实数值,跑出来验证单调不减(每档的下界正好接上一档的上界) ---
stages = [curriculum_lengths(s) for s in range(1, 5)]
assert stages == [(256, 2048), (2048, 8192), (8192, 32768), (32768, 131072)]
for i in range(len(stages) - 1):
    assert stages[i][0] <= stages[i][1] <= stages[i + 1][0] <= stages[i + 1][1]

# --- 越界 stage 会静默 fallback 回 stage 1 的默认区间,不报错 ---
assert curriculum_lengths(5) == (256, 2048)
assert curriculum_lengths(0) == (256, 2048)

# --- filter_by_curriculum:按当前阶段的长度区间过滤文档 ---
test_docs = [list(range(n)) for n in [100, 1000, 5000, 20000, 50000]]
stage1_docs = filter_by_curriculum(test_docs, 1)   # 区间 (256, 2048)
assert [len(d) for d in stage1_docs] == [1000]      # 只有 1000 落在区间内(100 太短,5000 太长)

stage3_docs = filter_by_curriculum(test_docs, 3)   # 区间 (8192, 32768)
assert [len(d) for d in stage3_docs] == [20000]

print("stages:", stages)
```

**面试怎么问 + 追问链:**
- **Q:** "训练长上下文模型时,为什么不直接一开始就用目标长度(比如 128k)训练?"—— 期望"显存 / 吞吐 / 训练稳定性"三个角度至少能提到两个。
- **追问 1:** "如果要你设计课程的分档区间,你会选线性增长还是指数增长?为什么?"—— 期望结合"attention 计算量随长度超线性增长"来谈,并且能观察到源码里 `256→2048→8192→32768→131072` 正好是公比为 4 的等比数列,不是随手拍的数字。
- **追问 2(区分度很高):** "从短序列阶段切换到长序列阶段,除了'允许更长的文档'之外,还有哪些工程细节容易被忽略?"—— 期望提到:同样显存预算下序列变长意味着 batch size 要相应变小(或者靠梯度累积补);如果用了 RoPE scaling(比如 YaRN),缩放因子是否要跟着课程阶段调整,还是提前固定好——capstone 脚本里 `curriculum_max_len` 只管序列长度,YaRN 的 `scale` 是提前一次性注入好、不随课程变化的,这种"两个机制不同步演进"的设计取舍本身就是一个可以被追问出来的细节。

**常见坑:** 把 `filter_by_curriculum` 的"硬切换"当成生产级实现直接照搬——真实训练更常用渐进式数据配比(各长度段数据都保留一部分,只调整比例),完全切断某个长度区间的数据,理论上有引发灾难性遗忘的风险;`curriculum_lengths(stage)` 对越界 `stage`(比如 `0` 或 `5`)同样不会报错,而是静默 fallback 回 `stage=1` 的默认区间——如果调用方的 `stage` 计数逻辑写错了(比如以为从 0 开始计数结果传了 5),不会有任何提示,容易埋下"以为训练已经进入最长阶段,实际上一直卡在最短区间"这种很难在日志里发现的隐蔽 bug。

---

## 3. Capstone:Llama-3.2-1B + YaRN(scale=4) + LoRA → 32k(`capstone_yarn_llama32.py`)

**是什么:**
```python
# learning/long-context/src/capstone_yarn_llama32.py
def yarn_inv_freq(dim: int, base: float = 10000.0, scale: float = 4.0): ...   # 13-17行
def attn_temperature(scale: float = 4.0) -> float: ...                         # 20-21行
def inject_yarn(model, scale: float = 4.0, new_max_pos: int = 32768): ...      # 24-42行
def setup_lora(model): ...                                                      # 45-54行
def curriculum_max_len(step: int) -> int: ...                                   # 57-62行
```
这是一个带真实 `argparse` 命令行接口的脚本,把本系列 01 讲过的 YaRN RoPE scaling、和 LoRA 参数高效微调串成一个"如果真的要做上下文扩展,大致长这样"的端到端骨架。

**一句话:** 这个脚本演示的是把 YaRN 注入一个**已经加载好**的 HF 模型对象、再套上 LoRA 的完整"配置阶段",但**必须先破除一个误解**:默认运行是纯 dry-run(不加载任何模型权重,只打印参数),即便手动加上 `--train`,代码本身也明确在 LoRA setup 之后就停了——脚本注释原文写着"省略 data loader + Trainer 实例化",**这不是一个能跑出"Llama-3.2-1B 真的被扩到 32k 上下文"这种真实训练结果的完整脚本**。

**底层机制/为什么这样设计:**

- **`yarn_inv_freq`:** 和 01 系列 `rope_yarn.py::yarn_cos_sin` 用的是同一族 NTK-by-parts 数学,这里是专门给一个真实 HF 模型的 rotary embedding 准备"替换用"的 `inv_freq` 张量。`new_base = base * scale ** (dim / (dim - 2))` 是 NTK-aware 的 base 换算公式——注意这里的 `dim` 指的是 **head_dim**(单个 attention head 的维度),不是模型的 `hidden_size`,这是一个容易搞混的细节。

- **`attn_temperature`:** `sqrt(0.1 * ln(scale) + 1)`——这是脚本使用的简化版本,和 01 系列知识点 5 交叉核对过的结论完全一致:真实 `transformers==5.10.2` 的 `modeling_rope_utils.py::get_mscale` 是没有这个 `sqrt` 的(直接 `0.1*ln(s)+1`)。这里不重复那次验证过程,只提醒一句:这个脚本里算出来的温度值,数值上和生产库不完全一致,是教学简化,不要直接照搬进生产代码。

- **`inject_yarn` 具体做了什么:** 不是训练出一个新模型,而是"手术式"地改写一个**已经实例化**的 HF 模型对象的两处状态——① `model.config.rope_scaling` 和 `max_position_embeddings`,纯粹是元数据,方便后续 `save_pretrained()` 这类操作把"这个 checkpoint 用了 YaRN"记录下来;② 遍历 `model.model.layers`,把每一层 `self_attn.rotary_emb.inv_freq` 这个 buffer **原地替换**成新算出来的 YaRN `inv_freq`。这是"运行时打补丁"(monkey-patch)式的做法——不改 `transformers` 库代码,直接在模型加载完之后,在 Python 层面篡改已经实例化的对象属性。这类做法有一个真实的风险:如果库内部某处对 `inv_freq` 做了额外缓存,或者 `forward()` 走的是另一条路径重新计算 cos/sin 而不是每次都读 `self.inv_freq` 这个属性,这种运行时替换就会静默失效——这是"直接改一个封装好的库的内部实现细节"这类做法的通病,库一升级,你的补丁完全可能悄悄不起作用而不报任何错误。

- **`setup_lora`:** 标准 `peft.LoraConfig(r=16, lora_alpha=32, target_modules=[q_proj,k_proj,v_proj,o_proj], bias="none", task_type="CAUSAL_LM")`。`lora_alpha/r = 2` 是社区里很常见的一个默认缩放比例;只对 attention 的 4 个投影矩阵注入 LoRA、不动 MLP 层(`gate_proj`/`up_proj`/`down_proj`),是"用更少可训练参数换取可以接受的精度损失"的常见取舍。

- **`curriculum_max_len`:** 上一个知识点"课程学习"思想在"按训练 step 决定当前最大序列长度"这个维度的具体应用,和 lecture 13 Slide 8 给出的"step 0-100 用 8k、100-300 用 16k、300-500 用 32k"完全对应。

**AI 研究场景:** "YaRN 注入 + LoRA 微调"这个组合,是社区实际扩展开源模型上下文窗口时最常见的低成本方案之一——相比全量微调,LoRA 让这件事在消费级/单卡显卡上就能做,只是效果通常不如全量微调彻底(lecture 13 Slide 11 给出的预期数字里,`YaRN-only` 在 32k 上的 NIAH 准确率是 60%,`YaRN+LoRA` 是 82%,这是 lecture 给出的**预期数字**,不是本仓库的实测结果,引用时要说清楚这层区别)。

`dry-run` 模式本身也是一个值得学习的工程习惯:训练脚本先把所有超参数、课程表、模型/LoRA 配置完整打印一遍确认无误,再真正加载几 GB 到几十 GB 的模型权重开始训练——避免"模型加载了 10 分钟,才发现某个超参数从一开始就传错了"这种浪费时间的情况。

**可运行例子:**

第一步,跑默认(无参数)的 dry-run,不加 `--train`:
```
python learning/long-context/src/capstone_yarn_llama32.py
```
本机实测输出(完整,没有任何删减):
```
[capstone] dry-run=True
[capstone] model=meta-llama/Llama-3.2-1B-Instruct
[capstone] scale=4.0, target_ctx=32768
[capstone] steps=500, lr=5e-05
[capstone] curriculum:
  step    0: max_len = 8192
  step   50: max_len = 8192
  step  150: max_len = 16384
  step  350: max_len = 32768

[capstone] dry-run, no training.
[capstone] to actually train: --train --steps 500
```
可以看到:不传任何参数,脚本只是把 argparse 的默认值(`model=meta-llama/Llama-3.2-1B-Instruct`、`scale=4.0`、`target_ctx=32768`、`steps=500`、`lr=5e-5`)和课程表打印出来,**没有 import `transformers`,没有联网,没有加载任何权重**。

第二步,几个纯数学/纯配置的辅助函数不需要加载模型也能独立验证:
```python
import sys, math
sys.path.insert(0, "learning/long-context/src")
from capstone_yarn_llama32 import yarn_inv_freq, attn_temperature, curriculum_max_len

# --- yarn_inv_freq:纯数学函数,不依赖任何具体模型 ---
head_dim = 64  # 随便挑一个偶数验证公式本身
inv_freq = yarn_inv_freq(head_dim, base=10000.0, scale=4.0)
assert inv_freq.shape == (32,)                       # head_dim/2 个频率
assert abs(inv_freq[0].item() - 1.0) < 1e-6           # 第0维频率的指数项为0,恒等于1,不受 base 缩放影响

import torch
vanilla = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
assert torch.allclose(yarn_inv_freq(head_dim, scale=1.0), vanilla)  # scale=1 时应退化回普通 RoPE

# --- attn_temperature ---
assert abs(attn_temperature(4.0) - math.sqrt(0.1 * math.log(4.0) + 1.0)) < 1e-12
assert abs(attn_temperature(1.0) - 1.0) < 1e-12       # scale=1(不缩放)时温度退化成1,不改变 attention

# --- curriculum_max_len:和上面 dry-run 打印的课程表完全对应 ---
vals = [curriculum_max_len(s) for s in [0, 50, 99, 100, 150, 299, 300, 350, 10000]]
assert vals == [8192, 8192, 8192, 16384, 16384, 16384, 32768, 32768, 32768]

# --- LoraConfig 本身可以脱离模型独立构造和检查,不需要 get_peft_model ---
from peft import LoraConfig
cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"],
                  bias="none", task_type="CAUSAL_LM")
assert cfg.r == 16 and cfg.lora_alpha == 32
assert set(cfg.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}
assert cfg.lora_alpha / cfg.r == 2.0
```

第三步,`inject_yarn` 真正改的是一个**已加载模型对象**的属性,不需要下载真实权重也能验证它的行为——手搭一个字段结构相同的"假模型"(鸭子类型,只是几个普通 Python 对象):
```python
import sys, torch
sys.path.insert(0, "learning/long-context/src")
from capstone_yarn_llama32 import inject_yarn, yarn_inv_freq

class FakeConfig:
    def __init__(self):
        self.hidden_size, self.num_attention_heads = 2048, 32   # head_dim = 2048/32 = 64
        self.max_position_embeddings, self.rope_scaling = 8192, None

class FakeRotaryEmb:
    def __init__(self, head_dim):
        v = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.inv_freq = torch.nn.Parameter(v, requires_grad=False)

class FakeModel:
    def __init__(self, n_layers=4, head_dim=64):
        self.config = FakeConfig()
        self.model = type("Inner", (), {})()
        self.model.layers = [type("Layer", (), {"self_attn": type("Attn", (), {
            "rotary_emb": FakeRotaryEmb(head_dim)})()})() for _ in range(n_layers)]

fake = FakeModel()
old_inv_freq = fake.model.layers[0].self_attn.rotary_emb.inv_freq.clone()
result = inject_yarn(fake, scale=4.0, new_max_pos=32768)

assert result.config.max_position_embeddings == 32768
assert result.config.rope_scaling == {"type": "yarn", "factor": 4.0}
expected = yarn_inv_freq(64, scale=4.0)
for layer in result.model.layers:
    new_inv_freq = layer.self_attn.rotary_emb.inv_freq.data
    assert torch.allclose(new_inv_freq, expected)          # 四层全部换成同一份 YaRN inv_freq
    assert not torch.allclose(new_inv_freq, old_inv_freq)   # 确实和原始 vanilla inv_freq 不同了
```
本机实测:`inject_yarn` 打印 `[yarn] scale=4.0, new_max_pos=32768, attn_temp=1.0671`,四层假模型的 `inv_freq` 全部被正确替换,`config.rope_scaling` 从 `None` 变成 `{"type": "yarn", "factor": 4.0}`。这一步**没有下载任何真实权重**,是刻意的取舍——加载一个几 GB 的 gated 模型只为验证"两处属性被改写"这个逻辑,成本完全不成比例。

**面试怎么问 + 追问链:**
- **Q:** "给一个已经训练好的模型做上下文扩展(context extension),你的大致思路是什么?"—— 期望说出"RoPE scaling(PI/NTK/YaRN 选一种)+ 在目标长度附近的数据上继续训练(全量或 LoRA)+ 用 NIAH/RULER 之类的方法验证效果",而不是只说"把 `max_position_embeddings` 改大就行"。
- **追问 1:** "如果完全不做任何微调,只注入 YaRN 缩放,模型能不能直接在 32k 上下文上正常工作?"—— 期望知道"能'跑'(不会直接报错崩溃),但效果通常会打折扣"——lecture 13 Slide 11 给出的预期数字(`YaRN-only` @32k NIAH 60% vs `YaRN+LoRA` 82%)能佐证"只改 RoPE、不训练"和"改完还训练"之间有明显差距,回答时要说明这是 lecture 给出的预期数字,不是本仓库的实测结果。
- **追问 2(区分度很高,容易被问倒):** "`inject_yarn` 这种运行时直接修改已加载模型内部属性(monkey-patch)的方式,相比在 config 里声明 `rope_scaling` 再重新 `from_pretrained` 加载,有什么风险?"—— 期望提到:库内部实现可能不会真的在每次 `forward()` 都重新读这个被改写的属性(可能有缓存、可能走另一条计算路径),运行时补丁可能悄悄失效而不报错;声明式的 config 修改(如果库原生支持)更可靠,因为是库自己在初始化时就按这份配置构建对应组件,不依赖"猜对了内部实现细节"。
- **追问 3(工程判断力):** "这个脚本的 `--train` 模式为什么只做到 LoRA setup 就停了,没有真正的训练循环?你觉得这样设计的考虑是什么?"—— 没有标准答案,考察对"教学代码 vs 生产代码"边界的判断力,合理的回答包括"避免读者误以为跑一下就能训出真实结果""真正的 Trainer 配置(数据集、评估回调、保存策略、分布式配置)因项目而异,写死在教学脚本里价值有限,反而可能造成误导"。

**常见坑:** 误以为跑一下 `--train` 就能得到一个真的扩展到 32k 上下文的 Llama-3.2-1B——脚本注释原文写着"省略 data loader + Trainer 实例化",这是一个**故意留白**的教学骨架,不是"功能不全的训练脚本",而是"根本没打算让你端到端跑通"的配置阶段演示,这是本知识点最容易被误解、也最需要反复强调的一点。把 `attn_temperature` 这个教学实现的数值当成"YaRN 论文/`transformers` 库的标准实现"直接抄进生产代码,是另一个常见误用——已在 01 系列验证过两者相差一个 `sqrt`,这里是同一个函数的再次出现,如果跳过 01 直接看这里,容易被误导成"这就是标准公式"。另外,`meta-llama/Llama-3.2-1B-Instruct` 是 gated 模型(本机现场验证过:未授权访问,连 `config.json` 都拿不到,会直接抛 `GatedRepoError: 401 Client Error`),即使想跑 `--train` 模式,不先在 HuggingFace 网站申请权限、`huggingface-cli login` 也下载不了任何东西——dry-run 模式完全不需要这一步,这也是为什么本知识点的"可运行例子"止步于 dry-run 和纯函数/mock 验证。

---

## 4. KV-cache 显存膨胀(来自 `lectures/12-long-context-pitfalls.md` Slide 4)

**是什么:**

这一条不是某个具体函数,而是长上下文**推理**场景里一个必须会算的显存核算问题。lecture 12 Slide 4 给出的算例(公式和数字摘录自源文件,不是凭记忆转述):
```
1B model + 128k ctx:
  KV cache ≈ 32 layer × 8 KV head × 128 head_dim × 2 (K+V) × 128k × 2 byte
            ≈ 17 GB
vs 模型权重 1B × 2 = 2 GB → KV 是模型 8 倍。
```
写成通用公式就是:
```
KV cache 字节数 = n_layers × n_kv_heads × head_dim × 2(K+V) × seq_len × bytes_per_element
```

**一句话:** 模型权重的显存占用在训练/加载完之后是一个**常数**(参数量固定了,占用就固定了),但 KV cache 的显存占用是一个随"当前上下文长度"线性增长的**变量**——上下文越长,这个变量就越有可能反超权重本身,这正是长上下文推理里"权重不是显存瓶颈,KV cache 才是"这个反直觉现象的根源。

**底层机制/为什么这样设计:**

拆开公式里的每一项,每一项都对应一个具体的、不能省略的原因:
- **`n_layers`(层数):** 每一层 attention 都要维护自己独立的一份 KV cache,层与层之间不共享——层数直接线性放大总量。
- **`n_kv_heads`(KV 头数,不是 query 头数!):** 这是最容易被问倒的一点。在 GQA(grouped-query attention)/MQA 架构下,K/V 的头数可以远少于 Q 的头数(比如 Q 有 32 个头,K/V 只有 8 个头,多个 query 头共享同一组 K/V)。KV cache 存的是 **K 和 V 本身**,不是 attention score,所以它的大小天然只取决于"K/V 有多少个头",和 query 头数量无关——这也是 GQA/MQA 相比标准 MHA(K/V 头数=Q 头数)能大幅省显存的根本原因,lecture 12 Slide 5 把 GQA 列为第一个解决方案不是偶然的。
- **`head_dim`:** 每个头存的 K、V 向量各自的维度。
- **`2`(K+V):** 每个 token 要同时存一份 K 向量和一份 V 向量,缺一不可。
- **`seq_len`(当前上下文长度):** 这是和"模型权重"最本质的区别——**每多看到/生成一个 token,就要多存一份该 token 在每一层、每个 KV 头上的 K 和 V**,所以 KV cache 是运行时随输入变化的变量,不是训练完就固定下来的常数。上下文从 4k 涨到 128k,这一项直接放大 32 倍,而模型权重完全不变。
- **`bytes_per_element`(每个数占几个字节):** 精度决定的常数,bf16/fp16 是 2 字节;这一项越小,KV cache 就线性变小——这正是 lecture 12 Slide 5 里"KV quant int4"这条优化路径能生效的原因:int4 每个数只占 0.5 字节,相比 bf16 的 2 字节直接省 4 倍显存,是所有优化手段里改动成本最低、见效最直接的一种(代价是精度损失)。

**下面现场把 lecture 给出的这个算例重新算一遍**(不是照抄"≈17GB"这个结论,自己在 Python 里跑出来再核对):

```python
n_layers, n_kv_heads, head_dim = 32, 8, 128
kv_factor, seq_len, bytes_per_elem = 2, 128 * 1024, 2   # 128k = 128*1024 = 131072

kv_cache_bytes = n_layers * n_kv_heads * head_dim * kv_factor * seq_len * bytes_per_elem
assert kv_cache_bytes == 17_179_869_184

kv_cache_GB = kv_cache_bytes / 1e9        # 十进制 GB(1GB=10^9 字节)
kv_cache_GiB = kv_cache_bytes / 1024**3    # 二进制 GiB(1GiB=2^30 字节)
assert round(kv_cache_GB, 2) == 17.18      # lecture 说"≈17 GB",十进制换算下精确对上
assert round(kv_cache_GiB, 2) == 16.0      # 二进制换算下是整整 16 GiB(2^34 字节这个巧合)

model_weight_GB = 1_000_000_000 * 2 / 1e9   # 1B 参数,bf16(2字节),十进制 GB
assert model_weight_GB == 2.0               # 和 lecture 的"1B × 2 = 2GB"完全对上

ratio = kv_cache_GB / model_weight_GB
assert round(ratio, 2) == 8.59   # lecture 说"KV 是模型 8 倍",精确算出来是 8.59 倍,取整后一致
```
实测结论:`kv_cache_bytes = 17,179,869,184` 字节,换算成十进制 GB 是 **17.18 GB**,和 lecture 的"≈17 GB"精确对上;换算成二进制 GiB 恰好是 **16.0 GiB**(`kv_cache_bytes` 精确等于 `2^34`,是这组参数刚好凑出来的巧合——本身是 2 的整数次幂,不是普遍规律)。KV cache 相对模型权重的比例是 **8.59 倍**,和 lecture 的"8 倍"吻合(取整方向不同但数量级完全一致)。这次没有出现"和 lecture 数字对不上"的情况,但下面这组额外验证展示了这个"8 倍"结论**不能被当成一个到处适用的常数**:

```python
# 固定 lecture 给的这组注意力架构参数(32层/8个KV头/128 head_dim/128k上下文)不变,
# 只改变"模型总参数量"这一个变量,看 KV/权重比例怎么变化
for n_params, label in [(1_000_000_000, "1B"), (8_000_000_000, "8B"), (70_000_000_000, "70B")]:
    weight_GB = n_params * 2 / 1e9
    r = kv_cache_GB / weight_GB
    print(f"{label}: 权重={weight_GB:.1f}GB, KV/权重={r:.3f}x")
# 实测输出:
#   1B:  权重=2.0GB,  KV/权重=8.590x
#   8B:  权重=16.0GB, KV/权重=1.074x
#   70B: 权重=140.0GB, KV/权重=0.123x
assert round(kv_cache_GB / 2.0, 2) == 8.59
assert round(kv_cache_GB / 16.0, 2) == 1.07
assert round(kv_cache_GB / 140.0, 3) == 0.123
```
同样的 KV cache 绝对大小(17.18GB,因为层数/KV头数/head_dim/上下文长度都没变),换成 8B 模型,比例从"8 倍"骤降到"1 倍出头";换成 70B 模型,KV cache 反而只占权重的 12.3%。"KV cache 是模型的 8 倍"这个结论,只在"1B 这个参数规模 + 这组具体的层数/头数配置 + 128k 上下文"这个特定场景下成立,模型一变大,结论就完全不一样——这是本知识点除了公式本身,更值得记住的一层认知。

**AI 研究场景:** 这个核算直接决定了 lecture 12 Slide 5 列出的一整套工程对策该怎么选:GQA/MQA 直接减少 `n_kv_heads`;MLA(DeepSeek-V3)用低秩投影压缩 KV 表示本身;KV 量化(int8/int4)降低 `bytes_per_element`;H2O/SnapKV 这类方法减少有效 `seq_len`(剪掉不重要的 token,有丢信息的风险);PagedAttention 不减少 KV cache 总量,而是解决"显存碎片化"这个独立问题(类似操作系统虚拟内存分页)。这也和本系列 02(长上下文 Attention 架构)形成呼应——Infini-Attention 那种"用压缩记忆替代完整 KV cache"的思路,本质上也是在回应这里算出来的显存压力。部署长上下文服务时,"这个模型宣称支持 128k 上下文"和"128k 上下文并发情况下我的显卡撑不撑得住"是两个完全独立的问题,后者必须靠这里的公式现算,不能靠感觉。

**面试怎么问 + 追问链:**
- **Q:** "为什么长上下文推理时,显存瓶颈经常是 KV cache 而不是模型权重本身?"—— 期望说出"模型权重是常数,KV cache 随 batch × 序列长度线性增长,长上下文/大 batch 场景下很容易反超权重本身"。
- **追问 1:** "给定层数/KV头数/head_dim/精度,你能现场估算一个具体场景下 KV cache 有多大吗?"—— 期望能现场把公式列出来手算一遍(和本知识点的可运行例子完全对应),而不是只会背"KV cache 很大"这句结论。
- **追问 2(区分度很高):** "如果模型用的是 GQA(比如 8 个 KV head、32 个 query head),KV cache 公式里的头数应该用 8 还是 32?为什么?"—— 期望准确答出"用 KV head 数(8),因为 KV cache 存的是 K/V 本身,不是 attention score;GQA 的设计初衷就是让多个 query head 共享同一组 K/V,KV cache 大小从架构设计上就和 query head 数量无关",能顺带说出这正是 GQA 比标准 MHA 省显存的根本原因,说明理解到了架构设计的动机而不只是记住公式。
- **追问 3(工程场景):** "除了缩小 KV head 数量(GQA/MQA),还有哪些方法能缓解 KV cache 膨胀?各自的代价是什么?"—— 期望至少举出 2-3 种并说出代价:KV 量化(省显存但有精度损失风险)、H2O/SnapKV 类剪枝(省显存但可能丢掉后面才发现重要的 token)、PagedAttention(解决碎片化,不减少总量本身)、MLA(压缩率更高但要改动模型架构、不是即插即用)。

**常见坑:** 把"模型卡上宣称支持 128k 上下文"和"128k 场景下我的显卡真的扛得住"混为一谈——上下文长度上限是架构/位置编码层面的能力,实际能不能部署要单独核算 KV cache 显存,尤其是多用户并发场景下每个请求都要独立维护一份 KV cache,128k 这种长度很容易在没有明显征兆的情况下把显存打满(这也呼应 lecture 12 Slide 8"context limit ≠ usable limit"的另一层含义,不仅是准确率会掉,显存也可能扛不住)。混淆十进制 GB(`10^9` 字节)和二进制 GiB(`2^30` 字节)也是常见错误——这个例子里两种换算结果"17.18 GB"和"16.0 GiB"看起来差得不算多(不到 10%),但换算错误在更大规模(几百 GB 级别的显存预算规划)上累积起来会造成不可忽视的偏差,做容量规划时要先确认清楚自己用的工具/团队约定的是哪一种。最后也是最容易被忽视的一点:把"KV cache 是模型权重的 8 倍"当成一个可以到处套用的死结论——上面已经现场验证过,这个比例只在"1B 模型规模 + lecture 给的这组层数/头数配置 + 128k 上下文"这个具体场景下成立,换一个模型规模(8B/70B)结论完全不同,任何具体场景都应该重新代入公式现算,而不是记一个数字到处用。

---
