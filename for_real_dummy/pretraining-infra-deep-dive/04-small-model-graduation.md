# 04 · Module 3 毕业:五部曲(Small Model Graduation)

> 总览见 [00-roadmap.md](00-roadmap.md)。本文是叙事体 capstone,不采用七步知识点模板。

## 开场:这不是一道知识点,是一场答辩

前三篇文章分别讲了数据处理(01)、训练规模化(02)、预训练配方(03),每一篇都在讲某一个独立的子系统。`learning/small-model-graduation/`(Module 3《造大模型》第 8 专题,毕业站)要回答的问题不一样:**这些子系统真的组合起来,能不能训出一个一步步变强的模型?**

答辩的形式是五个 checkpoint(A→B→C→D→E),每一步只改一个变量:

| ckpt | 相对上一步改了什么 | 模型 | 数据 |
|---|---|---|---|
| **A** | (基线) | Vanilla GPT-2 124M | TinyStories + WebText |
| **B** | 改数据 | 同 A | Cosmopedia + 高质过滤 web |
| **C** | 改架构 | **Phi-tiny 270M**(GQA+RoPE+SwiGLU+RMSNorm) | 同 B |
| **D** | 长上下文扩展 | C + YaRN scale=4 | 长文档 |
| **E** | 课程学习综合 | C + curriculum | 全套配方 |

这五步分别对应本仓库已完成的三条系列:B 靠的是 01 号文件的数据处理管线,C 靠的是 03 号文件的 Phi-tiny 架构(`phi_tiny_model.py` 直接复用,不是重新实现),D 靠的是 [long-context-deep-dive](../long-context-deep-dive/01-rope-scaling-family.md) 的 YaRN(本系列不重复展开 YaRN 原理,只讲它在这个五部曲叙事里的位置)。**诚实声明**:五部曲里真正有本机 GPU 真实训练验证的只有 ckpt A(下方"幕一"），其余四个 ckpt 的训练/评测数字全部来自源码 `bench_matrix.py::EXPECTED` 这份**预先写好的参考数据**(源码 stdout 自己在跑 dry-run 时会打印 "using EXPECTED for dry-run"，不是伪装的真实训练结果)——这是诚实的报告骨架，用于演示五部曲叙事的完整流程，不是声称五个模型都真的训完了。

---

## 幕一:ckpt A —— Vanilla GPT-2 基线,以及一次意外的真实发现

**是什么**(`vanilla_gpt2.py`):标准 GPT-2 架构(Post-LayerNorm、`nn.MultiheadAttention`、学习的绝对位置编码、tied embedding),真实建模:

```python
import sys, torch
sys.path.insert(0, "learning/small-model-graduation/src")
from vanilla_gpt2 import GPT2Config, VanillaGPT2

c = GPT2Config()   # vocab=50257, hidden=768, n_head=12, n_layer=12
m = VanillaGPT2(c)
n = sum(p.numel() for p in m.parameters())
n_no_embed = n - m.tok_embed.weight.numel()
assert 124e6 < n < 125e6
x = torch.randint(0, c.vocab_size, (2, 64))
assert m(x).shape == (2, 64, c.vocab_size)
print(f"total={n/1e6:.1f}M  excl_embed={n_no_embed/1e6:.1f}M  fwd_shape={tuple(m(x).shape)}")
```

**实测:** 总参数 **124.4M**("GPT-2 124M"名字的由来),排除共享 embedding 后 85.8M。真实 forward `(2, 64) → (2, 64, 50257)`,shape 正确。

**真实 GPU 训练验证**(本系列第二处真实反向传播,紧接 03 号文件知识点 11 的 Phi-tiny 训练之后):

```python
import subprocess, sys, re
result = subprocess.run(
    [sys.executable, "learning/small-model-graduation/src/train_variant.py",
     "--variant", "A", "--max_step", "2", "--seq_len", "64",
     "--micro_batch", "2", "--grad_accum", "2", "--train"],
    capture_output=True, text=True, timeout=60, encoding="utf-8",
)
assert result.returncode == 0, result.stderr
assert "device=cuda" in result.stdout
loss = float(re.search(r"step\s+0 loss ([\d.]+)", result.stdout).group(1))
assert loss < 3.0     # 见下方"意外发现" —— 这个loss远低于随机基线，不是训练学到了什么
print(result.stdout)
print(f"step 0 loss = {loss}")
```

真实运行(本机 RTX 3080 Ti):`device=cuda`,`step 0 loss 0.9979`。

**意外发现(本次验证独立挖出,不是源码文档提到的内容):** 这个 loss 数字(≈1.0)第一眼看很反常——50257 词表下随机初始化模型的交叉熵理论基线是 `ln(50257)≈10.82`,03 号文件里 PhiTiny 用几乎相同的 mock 训练设置(`mock_loader` 同样是 `y = x.clone()`,标签直接等于输入、不做 next-token 位移)测出的 step 0 loss 是 **9.5-9.6**,和理论基线接近;但 VanillaGPT2 在同样的 mock 设置下,loss 却只有 **≈1.0**,比理论基线低了整整一个数量级。独立复现(3 个不同随机种子 0/7/123,GPT2 和 PhiTiny 用完全相同的种子和输入分别测试):

```python
import sys, torch, torch.nn.functional as F
import numpy as np
sys.path.insert(0, "learning/pretraining-recipe/src")
sys.path.insert(0, "learning/small-model-graduation/src")
from phi_tiny_model import PhiTinyConfig, PhiTiny
from vanilla_gpt2 import VanillaGPT2, GPT2Config

results = []
for seed in [0, 7, 123]:
    torch.manual_seed(seed)
    gpt2 = VanillaGPT2(GPT2Config()).cuda().bfloat16()
    torch.manual_seed(seed)
    phi = PhiTiny(PhiTinyConfig()).cuda().bfloat16()
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.integers(0, 50257, (2, 64)).astype(np.int64)).cuda()
    y = x.clone()   # 两个模型的 mock_loader 都用这个约定(标签=输入,不shift)
    loss_gpt2 = F.cross_entropy(gpt2(x).flatten(0, 1).float(), y.flatten()).item()
    loss_phi = F.cross_entropy(phi(x).flatten(0, 1).float(), y.flatten()).item()
    results.append((seed, loss_gpt2, loss_phi))
    assert loss_gpt2 < 2.0      # GPT2 稳定远低于随机基线
    assert loss_phi > 9.0        # PhiTiny 稳定接近随机基线 ln(50257)≈10.82
for seed, lg, lp in results:
    print(f"seed={seed}: VanillaGPT2={lg:.3f}  PhiTiny={lp:.3f}")
```

3 个种子全部确认:**VanillaGPT2 稳定在 0.99-1.03,PhiTiny 稳定在 9.55-9.65**,差距不是噪声。进一步做对照实验(同样两个模型,把标签换成真实 next-token 位移 `y=roll(x,-1)` 而不是 `y=x.clone()`):VanillaGPT2 的 loss 立刻从 ≈1.0 跳回 **11.39**(比理论基线还高,符合"随机模型+随机数据"预期),PhiTiny 从 9.6 变为 11.05。这说明**只有在"标签=输入本身、不做位移"这个特定 mock 设定下,VanillaGPT2 才会出现异常低的 loss**,原因指向架构差异,但这个差异具体怎么起作用值得拆开验证,不能只停留在"架构不同"这一句猜测上。

先解释两个新概念:**残差流(residual stream)** 指 transformer 内部贯穿所有层的那条加法直通通道——每一层不是把输入"替换"成自己的输出,而是"加"回去(`x = x + 子层输出(x)`,VanillaGPT2 和 PhiTiny 的每个 block 都是这个模式),所以浅层写入的信息有机会原样传到很深的层,不会被中途整个抹掉。**tied embedding** 指"输出层和输入 embedding 共用同一份权重"(`lm_head.weight = tok_embed.weight`)——这一点**两个模型都有**(03 号文件知识点 5 已经提到 PhiTiny 的 `lm_head.weight = embed.weight` 省了 51.5M 参数,VanillaGPT2 的写法完全一样),所以"是不是 tied"不是两者的差异点。tied 意味着最终 logit 的计算是 `hidden @ E^T`(`E` 就是 embedding 矩阵本身)——如果 `hidden` 恰好和某个 token 的 embedding 向量方向接近,这个 token 的 logit 就会格外突出,"当前位置输入 token 自己"也不例外。真正的差异点是:两个模型的残差流,谁的最终 `hidden` 更接近"当前位置输入 token 自己的 embedding"?这一点不需要靠猜测,可以直接测出来:

```python
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "learning/pretraining-recipe/src")
sys.path.insert(0, "learning/small-model-graduation/src")
from phi_tiny_model import PhiTinyConfig, PhiTiny
from vanilla_gpt2 import VanillaGPT2, GPT2Config

torch.manual_seed(0)
gpt2 = VanillaGPT2(GPT2Config())
torch.manual_seed(0)
phi = PhiTiny(PhiTinyConfig())
x = torch.randint(0, 50257, (2, 64))

def gpt2_final_hidden(m, x):
    B, T = x.shape
    pos = torch.arange(T, device=x.device).unsqueeze(0)
    h = m.tok_embed(x) + m.pos_embed(pos)
    for b in m.blocks:
        h = b(h)
    return m.ln_f(h), m.tok_embed(x)      # 送进lm_head之前的最终隐状态, 自己的输入embedding

def phi_final_hidden(m, x):
    h = m.embed(x)
    for b in m.blocks:
        h = b(h, m.rope_cos, m.rope_sin)
    return m.final_ln(h), m.embed(x)

h_gpt2, e_gpt2 = gpt2_final_hidden(gpt2, x)
h_phi, e_phi = phi_final_hidden(phi, x)

cos_gpt2 = F.cosine_similarity(h_gpt2, e_gpt2, dim=-1).mean().item()
cos_phi = F.cosine_similarity(h_phi, e_phi, dim=-1).mean().item()
assert cos_gpt2 > 0.5 and cos_phi < 0.15   # GPT2的最终隐状态和自己的输入embedding高度同向, Phi几乎无关

top1_gpt2 = gpt2.lm_head(h_gpt2).argmax(-1)
top1_phi = phi.lm_head(h_phi).argmax(-1)
frac_self_gpt2 = (top1_gpt2 == x).float().mean().item()   # top-1预测精确等于输入token自己的位置占比
frac_self_phi = (top1_phi == x).float().mean().item()
assert frac_self_gpt2 > 0.99
assert frac_self_phi < 0.05
```

**实测(`.venv` 真跑):** `cos(最终隐状态, 自己的输入 embedding)` 均值,VanillaGPT2 是 **0.682**,PhiTiny 只有 **0.066**(接近正交、基本无关联)——两个模型的残差流对"自己输入 embedding"的保留程度差了一个数量级。接到 tied 的 `lm_head` 上看最终后果:全部 128 个测试位置(batch=2×seq_len=64)里,VanillaGPT2 的 top-1 预测 **100%** 精确等于输入 token 自己;PhiTiny 只有 **1.56%**(仍然比 1/50257≈0.002% 的随机基线高出约784倍、接近三个数量级,说明这条"捷径"对 PhiTiny 也有微弱作用,只是远没有到"每次命中"的程度)。这条独立验证把"原因大概率是"这句猜测,换成了两个模型真实前向结果量出来的因果链条:**残差流保留输入 embedding 的程度 → 决定 tied lm_head 会不会把这个 embedding 自己顶到 top-1 → 决定 `y=x.clone()` 这种标签方案下 loss 是否异常偏低**。VanillaGPT2 的 Post-LayerNorm + 学习的绝对位置编码这套组合,残差流恰好更容易保留这个分量;PhiTiny 的 Pre-RMSNorm + RoPE(不在输入端加位置编码,而是在 attention 内部旋转 Q/K)这套组合则没有这个效应——这和两者 cosine 相似度的实测差异方向完全一致,但"Post-LN 具体通过什么机制让残差流更容易保留输入信息"本身是一个更深的架构理论问题,本文只验证到"确实存在这个差异、且方向和后果都对得上",不展开证明背后的数学必然性。

**这条发现的价值不在于"哪个数字更好看"**,而在于:如果不做这个交叉模型对照实验,单看 VanillaGPT2 的 loss 从 10+ 掉到 1 附近,很容易误判成"模型训练得很好、收敛很快"——但这只是两个架构在同一个有缺陷的 mock 标签方案下表现出的不同初始化偏置,和"模型学到了语言能力"毫无关系(数据本身是随机整数,不携带任何语言结构)。这是"5-10 步真实反向传播 smoke test 该看什么、不该看什么"的一个具体反面教材:loss 数字本身在跨架构比较时可能存在系统性偏置,必须先搞清楚 mock 数据/标签方案和被测架构之间是否存在这类交互效应,才能正确解读。

---

## 幕二:ckpt B —— 改数据(链接 01 号文件)

ckpt B 相对 A 唯一的变量是数据:从 TinyStories+WebText 换成 Cosmopedia+高质过滤 web。模型架构和训练超参完全不变。这一步的"改进机制"不需要重新讲——已经在 [01-data-curation.md](01-data-curation.md) 讲透:知识点 5(质量过滤,C4/Gopher 启发式+FineWeb-Edu classifier)就是"高质过滤"具体怎么做的答案,知识点 3/4(MinHash/SemDeDup 去重)保证 Cosmopedia 这类合成语料不会因为模板化内容被过度污染。`bench_matrix.py::EXPECTED` 给出的参考数字(dry-run 骨架,非真实训练结果):A 的 `val_loss=3.50`→B 的 `val_loss=3.20`,HellaSwag `0.35→0.40`(+5pp)。这条"仅换数据、模型不变"的对照,是整个五部曲里唯一一步"改进来源单一、容易归因"的一步——后面几步(C/D/E)都同时改了不止一个变量,给因果归因带来了实际的复杂度(下方"复盘"详细讨论)。

---

## 幕三:ckpt C —— 改架构(链接 03 号文件的 Phi-tiny)

ckpt C 把模型从 VanillaGPT2(124M)换成 Phi-tiny(270M,GQA+RoPE+SwiGLU+RMSNorm),数据延续 B。Phi-tiny 的完整架构解剖已经在 [03-pretraining-recipe.md 知识点5](03-pretraining-recipe.md#5-phi-tiny-270m-架构解剖phi_tiny_modelpy--pre-rmsnorm--gqa--rope--swiglu--权重共享五件套)讲过(GQA 让 KV cache 省 4 倍、tied embedding 省 51.5M 参数),这里不重复,只强调一点:**C 这一步同时改了"架构"和"模型规模"两个变量**(124M→270M),`EXPECTED` 表格把这一整步的收益都记在"arch (B→C)"名下(HellaSwag +8pp),但严格来说这个 +8pp 里有多少归功于架构设计本身(GQA/RoPE/SwiGLU 相比标准 MHA/绝对位置编码/GELU-MLP 的质变)、有多少纯粹是参数量翻倍带来的量变,`bench_matrix.py` 的对照设计本身**无法**把这两者分开(没有一个"270M VanillaGPT2"或"124M Phi-tiny"的中间对照组)。这不是本文档的疏漏,是源材料这份五部曲实验设计本身的一个真实局限——如实指出比假装归因清晰更重要。

同一次验证里,幕一发现的"VanillaGPT2 vs PhiTiny 在 mock 训练下 loss 基线不同"这条结果,恰好是"改架构"这一步在训练最初期就能观察到的真实、可测量的行为差异,即使数据仍是随机整数、不具备真实评测意义,也说明两个架构在训练动力学起点上确实不是同一回事。

---

## 幕四:ckpt D —— 长上下文扩展(链接 long-context-deep-dive)

ckpt D 从 C 的 checkpoint resume,应用 YaRN scale=4(把有效上下文窗口从训练时的长度外推到 4 倍)+ LoRA r=16(只用低秩适配器做这次短暂的 100 step 扩展训练,不动主干权重;`r` 是适配器内部"降维再升维"那一步先降到的维度——数值越大,适配器能表达的变换越丰富,但新增参数也越多,完整原理见 [peft-deep-dive/01-lora-core.md](../peft-deep-dive/01-lora-core.md),本系列不重复展开)。YaRN 的完整数学原理(为什么要分频段插值、为什么纯 PI 会破坏高频信息)已经在 [long-context-deep-dive/01-rope-scaling-family.md](../long-context-deep-dive/01-rope-scaling-family.md) 讲透,本系列不重复。`EXPECTED` 表格显示 D 相对 C 的变化极具指向性:6 个 benchmark 里 5 个(val_loss/hellaswag/piqa/tinymmlu/gsm8k)**完全不变**,只有 `niah_8k` 从 0.05 跳到 0.80(+75pp)——这份参考数据本身就在演示 YaRN 的一个核心特性:**长上下文扩展是一种"追加能力",不是"整体能力提升"**,不应该期待它连带改善其他所有 benchmark 的分数,如果扩展后某个短上下文 benchmark 分数下降了,反而说明扩展过程带来了不该有的能力损伤(这是 YaRN/PI 等方法真实评测时需要专门检查的回归项)。

---

## 幕五:ckpt E —— 课程学习综合,以及一处容易被误读的标签

ckpt E 是"全部合一"(C 的架构 + curriculum 数据课程 + 长上下文能力),`common.py::variant_desc("E")` 的描述是"全部(= Topic 7 final)"。"课程学习"(curriculum learning,训练不同阶段喂给模型不同难度/质量的数据,而不是全程用同一个固定配比)已经在 [03-pretraining-recipe.md](03-pretraining-recipe.md) 讲过具体机制:知识点 2(`data_mixture.py`)展示了 Phi 系列训练最后 20% 阶段切换到更高质量数据子集这个具体做法,这里不重复,只看它在五部曲里的定位和一处容易被误读的归因。`EXPECTED` 表格把 E 相对 C 的变化标注为 `ablation_breakdown` 函数里的 `"curriculum (C->E)"` 这个 key,但真实运行这个函数后独立核对发现一处**容易被字面误读的地方**:

```python
import sys
sys.path.insert(0, "learning/small-model-graduation/src")
from bench_matrix import EXPECTED, ablation_breakdown

diff = ablation_breakdown(EXPECTED)
curriculum_diff = diff["curriculum (C->E)"]
assert abs(curriculum_diff["niah_8k"] - 0.75) < 1e-9          # E的niah_8k继承了D的长上下文能力，不是curriculum本身的贡献
assert abs(curriculum_diff["hellaswag"] - 0.02) < 1e-9          # 但hellaswag只涨了2pp，远小于"data(A->B)"的5pp和"arch(B->C)"的8pp
print(f"'curriculum (C->E)' 标签下的niah_8k差值: {curriculum_diff['niah_8k']:+.2f}")
print("(这个+0.75不是curriculum学习法本身带来的，是因为E的EXPECTED数据假设E继承了D的YaRN长上下文能力)")
```

`"curriculum (C->E)"` 这个 key 名字暗示"这一栏的差异都是课程学习带来的",但 `niah_8k` 从 C 的 0.05 涨到 E 的 0.80,这个 +75pp 和幕四 D 相对 C 的涨幅**完全相同**——因为 E 的定位是"全部技术合一"(源码 `variant_desc` 明确写"全部"),`EXPECTED["E"]` 这份参考数据本身就假设 E 继承了 D 的 YaRN 长上下文能力,不是"单独把课程学习这一项应用在 C 上"的纯净对照。`ablation_breakdown` 函数把这一整栏差异贴上"curriculum"标签,容易让读者误以为"课程学习贡献了 75pp 的 NIAH 提升"——真实归因应该是"E 相对 C 的总差异中,niah_8k 那部分主要来自 D 的 YaRN(继承而来),hellaswag/piqa/tinymmlu/gsm8k 那小几个百分点的额外提升才是课程学习本身的边际贡献"。这是"ablation 分解函数的 key 名字"和"其真实衡量的因果关系"之间存在偏差的一个具体案例,读任何自动生成的 ablation 报告前都需要先搞清楚每一栏对照组的真实构成,不能只看列名。

---

## 综合报告:graduation_capstone.py 真实产出的 4 份文件

`graduation_capstone.py`(dry-run 模式,不带 `--train`)把上面五幕串成一次完整流程,真实运行产出 4 份报告文件:

```python
import subprocess, sys, tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    result = subprocess.run(
        [sys.executable, "learning/small-model-graduation/src/graduation_capstone.py",
         "--report_dir", tmp],
        capture_output=True, text=True, timeout=60, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert "using EXPECTED for dry-run" in result.stdout   # 诚实自我声明,不伪装成真实训练结果
    produced = sorted(p.name for p in Path(tmp).iterdir())
    assert produced == ["ablation.json", "benchmarks.csv", "generations.md", "report.md"]
    report_text = (Path(tmp) / "report.md").read_text(encoding="utf-8")
    assert "数据质量贡献" in report_text
    print(f"产出文件: {produced}")
```

**实测(`.venv` 真跑):** dry-run 完整走完 5 个步骤(训练→benchmark 矩阵→ablation 分解→可视化→markdown 报告),真实在临时目录写出 `benchmarks.csv`/`ablation.json`/`generations.md`/`report.md` 四份文件,stdout 明确打印"using EXPECTED for dry-run"(源码自己的诚实标注,不是本文档事后补充的说明)。文本版 spider chart 和 loss curve(`visualize.py`)也真实按 `EXPECTED` 数据渲染,五行数字和知识点表格完全对应。

**关于 `generations.md`(`generations_compare.py::MOCK_OUTPUTS`)的诚实说明:** 这份文件对照 5 个 prompt 在 5 个 ckpt 下的"生成结果",但 `MOCK_OUTPUTS` 是**手写的字面量字典**(源码里能直接看到 `"the pan and the the the and the cooked it"` 这类硬编码字符串),不是任何模型的真实推理输出——A 的输出故意写得重复破碎("the the the")体现"baseline 质量差",E 的输出故意写得流畅详尽体现"综合最强",这是一份**叙事道具**,用来演示"报告应该长什么样",不是可以引用的真实模型能力证据。`generation_template()` 函数倒是给出了一份真实可用的 top-p 采样生成代码模板(如果接上真实训练出的 ckpt,可以直接复用这段代码做真实生成)。

---

## 复盘:五部曲能回答什么问题,不能回答什么问题

**能回答的:** 一个"数据→架构→长上下文→课程"的渐进式改进故事在工程上怎么组织(每一步只引入必要的新变量、复用前序 checkpoint、有统一的 benchmark 矩阵横向对比)——这个**流程骨架**是真实的、可复用的,`graduation_capstone.py` 的五步流水线(训练→评测→ablation→可视化→报告)是任何团队做类似"渐进式模型改进项目"时都值得参考的组织方式。幕一的真实 GPU smoke test 也证明了"改架构"这一步(VanillaGPT2→PhiTiny)背后的代码路径确实能在本机真实硬件上跑通,不是纸上谈兵。

**不能回答的:** 这五个 checkpoint 之间"HellaSwag 涨了几个百分点"这类具体数字全部来自 `EXPECTED` 这份预先写好的参考数据,不是五次真实训练+真实评测的产物——**没有证据表明这些具体数字反映任何真实训练结果**,它们的作用是让读者看到"如果五次训练都成功,报告应该长这个样子",不能被引用为"Phi-tiny 架构相比 VanillaGPT2 在 HellaSwag 上确实能提升 8 个百分点"这类具体的经验结论。真正有第一手真实证据支撑的,只有:(a)两个模型的真实参数量(124.4M/315.7M)和真实 forward shape;(b)幕一独立发现并三种子验证过的"VanillaGPT2 vs PhiTiny 初始化阶段 mock-loss 差异"这一条具体的、可复现的实证结果。

**给未来读者的建议:** 如果要把这份五部曲叙事变成真正有说服力的实验报告,最小改动路径是——先把 `mock_loader`/`mock_data_loader` 换成任何一个真实小语料(哪怕只是重复过几遍的公开文本文件),五个变体各真训至少几百步,观察 loss 是否呈现有意义的下降趋势且五者之间出现可解释的相对排序,再决定要不要投入完整的 3000-4000 step 真实训练——这正是 03 号文件知识点 11"常见坑"部分给出的建议路径,在这里同样适用。

**面试怎么问 + 追问链:**
- **Q:** "如果一份实验报告的评测数字标注为'dry-run 参考值',你会怎么评估这份报告的价值?"—— 期望:这类报告的价值在于验证"实验设计和报告流程本身是完整、自洽的"(有没有遗漏关键对照组、指标选取是否合理、可视化和归因逻辑站不站得住脚),不能作为任何具体经验结论的证据来源;本文"curriculum (C→E)"标签的错位归因就是一个具体案例,说明即使承认是参考数据,报告的*叙事结构*本身也可能存在需要独立核查的问题。
- **追问1:** "本文'幕一'的 loss 异常发现,对设计新的 mock/smoke test 有什么启示?"—— 期望:如果 smoke test 的目的是"验证训练循环机制"而不是"观察真实收敛",选择的 mock 标签方案应该避免任何可能被架构 trivially 利用的捷径(比如 `y=x.clone()` 在部分架构下会退化成一个"复制自己输入"的简单任务而非真正的预测任务)——更稳妥的 mock 方案是用完全独立于输入 `x` 的随机标签 `y`(和 `x` 无关联),这样任何架构在这种标签下 step 0 的 loss 都应该稳定落在 `ln(vocab_size)` 附近,不会因为架构差异出现这种数量级的意外偏差,smoke test 的"基线是否正常"这条判据才更可靠。

---
*上一篇:[03-pretraining-recipe.md](03-pretraining-recipe.md) | 下一篇:[05-cuda-essentials.md](05-cuda-essentials.md) —— 从"造模型"转向"造 infra":CUDA 执行模型与内存优化基础。*
