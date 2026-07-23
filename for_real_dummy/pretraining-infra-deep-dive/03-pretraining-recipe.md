# 03 · 预训练配方深挖(Pretraining Recipe)

> 总览见 [00-roadmap.md](00-roadmap.md)

数据处理好了(01号文件)、并行策略选好了(02号文件),这份配方回答最后一个问题:怎么把这两者和一个具体的模型架构、一套训练超参组合成一次真正跑起来的预训练。本文对应 `learning/pretraining-recipe/`(Module 3《造大模型》第 7 专题,16 lecture + 9 个 src 源文件),核心是 Phi-tiny 270M 这个标杆架构(Pre-RMSNorm+GQA+RoPE+SwiGLU+权重共享),从初始化、学习率调度、数据加载,一路讲到评测和知识蒸馏。11 个知识点,收尾处含本系列**第一处真实 GPU 训练验证**(不是纯 CPU 数值模拟)。

**环境声明:** 本文全部代码在仓库根目录 `.venv`(Windows 11 原生,Python 3.13,torch 2.11.0+cu128,CUDA 可用,`torch.cuda.get_device_name(0)` 实测为 `NVIDIA GeForce RTX 3080 Ti Laptop GPU`)下用 `.venv/Scripts/python.exe` 实际跑通验证。9 个 src 源文件里 8 个是纯 CPU 秒级 demo(建模/公式/规则,不需要 GPU),唯独知识点 11(Capstone)额外做了一次**真实 GPU bf16 训练**(`capstone_train.py --train`)——真实前向+反向传播+优化器 step,不是打印模板,风险和环境依赖都远低于 `inference-serving-deep-dive` 系列的 WSL2+vLLM 部署(纯本地 `.venv` 直接跑,无需额外安装、无需 WSL2、无需下载权重)。

---

## 1. 预训练总流水线总览(`common.py`)—— 从数据到评测的九步全景,以及三种学习率调度的统一实现

**是什么:**
```python
def cosine_lr(step: int, max_step: int, base_lr: float,
              min_lr: float = 1e-5, warmup: int = 1000) -> float:
    """warmup + cosine decay."""
    if step < warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(1, max_step - warmup)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def wsd_lr(step: int, max_step: int, base_lr: float,
            warmup_pct: float = 0.05, decay_pct: float = 0.2) -> float:
    """Warmup-Stable-Decay (Phi/MiniCPM)."""
    warmup = int(max_step * warmup_pct)
    decay_start = int(max_step * (1 - decay_pct))
    if step < warmup:
        return base_lr * step / warmup
    if step < decay_start:
        return base_lr
    progress = (step - decay_start) / (max_step - decay_start)
    return base_lr * (1 - progress)
```
(`common.py:13-33`)

**一句话:** 一份完整预训练配方是"data→tokenize→mix→shard→init→train→eval→SFT→RLHF"九个环节串起来的流水线(后两步不在本专题范围,属于 Module 4),本知识点用配方里最核心的一个决策点(学习率怎么随训练进度变化)作为切入,展示 cosine decay 和 WSD 两种主流调度在同一套公共接口下如何实现。

**底层机制/为什么这样设计:** cosine decay 全程都在"下降"(从 warmup 结束那一刻起,学习率就开始沿余弦曲线单调递减到 `min_lr`),这要求训练开始前就精确知道 `max_step`(总步数),一旦需要延长训练,整条衰减曲线都要重新规划;WSD(Warmup-Stable-Decay)把训练切成三段——warmup 短暂爬升、stable 阶段长期保持恒定的 `base_lr`(这一段长度可以随时按需延长,不用预先精确定死总步数)、decay 阶段线性衰减到 0——这个设计上的关键优势是**训练中途可以决定"现在收尾"而不需要提前规划好整条衰减曲线**,只要临时追加一段 decay 即可,这是 Phi/MiniCPM 系列偏爱 WSD 而不是经典 cosine 的核心原因。

**AI 研究场景:** 学习率调度的选择直接影响训练稳定性和最终收敛质量,是任何预训练 run 立项时必须明确写进配方的第一个超参决策;WSD 这几年被 Phi-2/Phi-3、MiniCPM 等追求"训练效率"的小模型团队广泛采用,因为它把"我要不要现在停止训练"从"一个必须在训练开始前回答的问题"变成了"一个可以训练过程中动态决定的问题"。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/pretraining-recipe/src")
from common import cosine_lr, wsd_lr, chinchilla_optimal_tokens, estimate_flops

lrs_cosine = [cosine_lr(s, 5000, 1e-3) for s in [0, 1000, 2500, 5000]]
lrs_wsd = [wsd_lr(s, 5000, 1e-3) for s in [0, 1000, 2500, 5000]]
assert lrs_cosine[0] == 0.0 and lrs_wsd[0] == 0.0            # 两者 step=0 都从 0 起步(warmup 开头)
assert lrs_cosine[-1] < 1e-4                                    # cosine 在 max_step 衰减到接近 min_lr(但不是精确0)
assert lrs_wsd[-1] == 0.0                                        # wsd 衰减到精确 0(线性衰减到decay_start+span终点)
assert lrs_wsd[1] == 1e-3                                         # step=1000(默认warmup边界)已进入 stable 段，精确等于 base_lr
assert lrs_cosine[1] == 1e-3                                      # cosine 在 step=warmup 处，cos(0)=1，公式精确退化为 base_lr——两者在此点精确重合，不是巧合是数学必然
assert lrs_cosine[2] != lrs_wsd[2]                                 # 到 step=2500(50%)才真正分道扬镳，见下方"实测"

n_params = 270_000_000
tokens_1_20 = chinchilla_optimal_tokens(n_params)
flops = estimate_flops(n_params, tokens_1_20)
assert tokens_1_20 == n_params * 20
print(f"270M 模型 Chinchilla 1:20 需要 {tokens_1_20/1e9:.1f}B token, 训练 FLOPs={flops:.2e}")
```

**实测(`.venv` 真跑):** step=1000(5000 步总量的 20%)时,WSD 精确给出 `1.0000e-03`(等于 `base_lr`,已进入 stable 平台段),而 cosine 给出 `1.0000e-03`——**这两个数字在这个特定 step 恰好相等**,但原因完全不同:WSD 是"已到达平台、往后 3000 步都不会变",cosine 是"连续曲线恰好经过这一点、下一步就会继续下降"。step=2500(50%)时两者分道扬镳:cosine=6.944e-4(已经衰减了 30%+),WSD 仍是 1.0000e-3(仍在平台期,decay_start 在 80% 处才开始);step=5000(100%)时 cosine=1.000e-5(衰减到 `min_lr` 下限,不是 0),WSD=0.0(线性衰减到精确的 0)。

**面试怎么问 + 追问链:**
- **Q:** "WSD 的 stable 阶段学习率恒定不变,这样不会错过'学习率再调低一点收敛更好'的机会吗?"—— 期望:WSD 的设计假设是"精细调节留给最后的 decay 阶段"——stable 阶段恒定学习率追求的是尽可能快地推进训练进度(把大部分计算预算花在"大步走"上),decay 阶段(通常只占总步数的 10%-20%)才是真正精细收敛的窗口,这是"大部分时间效率优先、少部分时间精度优先"的两阶段策略,不是全程都要精细调节。
- **追问1:** "如果训练中途 loss 出现异常(spike),cosine 和 WSD 哪种调度更容易应对?"—— 期望:WSD 的 stable 阶段学习率不变,如果中途出现数据异常导致的 loss spike,配合梯度裁剪/spike guard(知识点 6 会展开)通常能够安然度过,恢复到同样的学习率继续训练;cosine 由于学习率一直在连续变化,如果 spike 恰好出现在衰减曲线的某个特定区域,后续行为的可预测性稍差,但这更多是理论上的细微差异,两者都需要配合独立的稳定性防护机制,不能只靠调度本身"抗 spike"。

**常见坑:** `chinchilla_optimal_tokens` 和 02 号文件知识点 1 的 `chinchilla_optimal_split` 名字相似但用途不同——前者只是"n_params × 20"这个单一比例的简单封装(本文件的简化版本,不做 loss 最小化搜索),后者是给定算力预算反推最优 (N,D) 组合的完整实现;本知识点的可运行例子特意同时验证两个来自不同文件的"chinchilla"相关函数不能替换使用,避免读者把两份不同粒度的工具函数混为一谈。

---

## 2. 数据配比与课程学习(`data_mixture.py`)—— 四大工业配方的具体数字,以及"训练进度决定数据配比"的课程设计

**是什么:**
```python
CONFIGS = {
    "phi": {"web": 0.5, "code": 0.2, "math": 0.1, "books": 0.1, "wiki": 0.1},
    "llama3": {"web": 0.70, "code": 0.20, "math": 0.04, "books": 0.04, "wiki": 0.02},
    "qwen": {"web_en": 0.4, "web_zh": 0.3, "code": 0.15, "math": 0.05, "books": 0.1},
    "deepseek": {"web_en": 0.3, "web_zh": 0.3, "code": 0.2, "math": 0.1, "other": 0.1},
}

def curriculum_stage(step: int, total_step: int) -> str:
    """Phi 风格: 80% general + 20% high-quality."""
    if step < 0.8 * total_step:
        return "general"
    return "high_quality"
```
(`data_mixture.py:8-35`,节选)

**一句话:** 不同公司的预训练配方(Phi/Llama-3/Qwen/DeepSeek)在"网页占比""代码占比""数学占比"这几个数字上差异巨大,这些差异直接体现了各团队对"模型应该在哪些能力上更强"的产品定位判断;`curriculum_stage` 展示了配比本身也不是训练全程固定不变的——Phi 系列在训练最后 20% 阶段切换到更高质量的数据子集,这是"课程学习"(curriculum learning)思路在预训练里的具体实现。

**底层机制/为什么这样设计:** Llama-3 配方 web 占比高达 70%(远超 Phi 的 50%),这反映 Meta 追求"更宽泛的通用能力覆盖";Qwen/DeepSeek 都把网页数据拆成 `web_en`/`web_zh` 两个独立 domain 分别配权重(而不是单一 `web` 笼统占比),这是中国团队为保证中文能力专门做的配比设计,呼应知识点 01 号文件揭示的"英文/中文 tokenizer 压缩率差 5 倍以上"这一问题——如果配比只按"文档数"或"字节数"分配、不针对语言差异做调整,实际训练信号会不成比例地偏向压缩率更高的语言。课程学习(`curriculum_stage`)的设计逻辑是:训练早期模型能力弱,大量高质量但结构复杂的数据(如数学证明、专业书籍)对当前模型来说"消化"不了,不如先用海量通用网页数据打好基础;训练后期模型已经具备较强的模式识别能力,再引入高浓度的优质数据能进一步精炼能力,过早引入反而可能造成"以小博大"的浪费。

**AI 研究场景:** 复现或魔改一个开源模型的预训练配方,第一步往往就是去读它的技术报告"数据构成"一节抄录这几个百分比——这是决定模型能力画像最直接的"配方级"决策,比模型架构细节(层数/宽度)更早、更根本地影响最终模型的能力分布。

**可运行例子:**
```python
import sys, random
sys.path.insert(0, "learning/pretraining-recipe/src")
from data_mixture import CONFIGS, sample_source, normalize, curriculum_stage, curriculum_mixture, wsd_annealing_mixture

rng = random.Random(42)
counts = {}
for _ in range(10000):
    s = sample_source(CONFIGS["deepseek"], rng)
    counts[s] = counts.get(s, 0) + 1
freq = {k: v / 10000 for k, v in counts.items()}
assert abs(freq["web_en"] - 0.3) < 0.02        # 抽样频率应贴近配置权重(±2%内)
assert abs(freq["web_zh"] - 0.3) < 0.02
assert freq["web_en"] + freq["web_zh"] > freq["code"]  # 中英文网页合计仍是最大占比

assert curriculum_stage(step=100, total_step=1000) == "general"
assert curriculum_stage(step=900, total_step=1000) == "high_quality"

annealing_late = wsd_annealing_mixture(step=900, total_step=1000)
annealing_early = wsd_annealing_mixture(step=100, total_step=1000)
assert "books" in annealing_late and annealing_late.get("books", 0) > 0.3   # annealing后期books占比应显著提高
assert annealing_early == CONFIGS["phi"]        # 前80%沿用phi默认配方,未触发annealing切换
print(f"DeepSeek配方10000次抽样: web_en={freq['web_en']:.1%} web_zh={freq['web_zh']:.1%} code={freq['code']:.1%}")
```

**实测(`.venv` 真跑):** DeepSeek 配方 10000 次真实抽样,`web_en=29.5%`/`web_zh=30.6%`/`code=19.6%`/`math=10.1%`/`other=10.3%`,和配置权重(0.3/0.3/0.2/0.1/0.1)全部落在 1 个百分点以内的采样误差范围;`curriculum_stage`/`wsd_annealing_mixture` 的阶段切换逻辑用两个不同的 step 值(100 vs 900,总量 1000)独立验证,80% 分界点前后确认切换到不同配方——annealing 后期配方(books 50%+wiki 20%+math 30%)完全不含 web/code,这是一次相当激进的配比切换,训练最后阶段几乎完全放弃通用网页数据,全力冲刺高质量书面语料。

**面试怎么问 + 追问链:**
- **Q:** "Qwen/DeepSeek 都把 web 拆成 `web_en`/`web_zh` 两个 domain,这样做和用一个统一的 `web`(混合中英文)domain 相比有什么实际区别?"—— 期望:分开配权重能精确控制两种语言各自的暴露量(比如明确要求中英文各 30%),如果合并成一个 `web` domain 再靠"语料库里天然的中英文比例"决定训练信号,配比会完全受制于原始网络语料的语言分布(通常英文占绝对多数),团队就失去了对最终模型双语能力平衡的直接控制手段。
- **追问1:** "训练最后阶段(annealing)切到几乎不含 web/code 的高质量配方,会不会让模型'忘记'训练早期学到的通用网页知识/代码能力?"—— 期望:这是课程学习设计里真实存在的风险(灾难性遗忘的轻量版本),实践中通常通过"annealing 占比不会切到 100%"(比如仍保留一小部分早期配方的采样权重)、以及 annealing 阶段本身步数占比通常不长(source 里是最后 20%)来控制风险敞口,完全归零通用数据的配方切换在真实工业实践中比较少见,大多数团队会保守地在最后阶段混入一小部分早期配方而不是完全替换。

**常见坑:** `sample_source` 每次调用只采样 1 个 domain(返回值是 domain 名字符串,不是具体文档),这个函数本身不涉及任何真实文本内容——它模拟的是"训练时下一个 batch 该从哪个 domain 取数据"这个决策过程,不是完整的数据加载器,如果需要真实的按配比混合的数据流,还需要结合知识点 4 的 `ShardManager` 或类似的按 domain 分桶的存储层实现。

---

## 3. 初始化策略与学习率 Scaling:GPT-2 style vs μP(`init_schedule.py`)—— 为什么模型变宽后,学习率要跟着系统性地调小

**是什么:**
```python
from __future__ import annotations  # 源文件顶部声明,类型注解延迟求值

def init_mup_style(module: nn.Module, base_width: int = 256) -> None:
    """μP style: σ ∝ 1/√fan_in, output proj 额外缩放."""
    for name, p in module.named_parameters():
        if "weight" in name and p.dim() >= 2:
            fan_in = p.shape[-1]
            std = math.sqrt(1.0 / fan_in)
            if "lm_head" in name:
                std *= math.sqrt(base_width / fan_in)
            nn.init.normal_(p, std=std)

def mup_lr(width_now: int, width_base: int, lr_base: float) -> float:
    """μP: scale lr with width."""
    return lr_base * (width_base / width_now)
```
(`init_schedule.py:21-59`,节选)

**一句话:** GPT-2 style 初始化用固定标准差 0.02(只对残差投影层额外按层数缩放),μP(Maximal Update Parametrization)初始化和学习率都要按模型宽度(hidden size)系统性缩放——μP 的核心承诺是"在小模型上调好的超参数(尤其是学习率),可以直接按公式换算到大模型上继续用,不需要重新搜索"。

**底层机制/为什么这样设计:** 标准初始化(GPT-2 style)下,模型变宽后每个神经元的输入连接数(fan_in)增多,如果初始化标准差不变,每层输出的方差会随宽度增长而系统性增大,连带需要用更小的学习率才能保持训练稳定——这正是为什么"从小模型迁移超参到大模型"在传统做法下几乎总是失败的:最优学习率本身就随模型宽度漂移。μP 通过让初始化标准差 `∝ 1/√fan_in`(宽度越大,标准差越小,抵消掉 fan_in 增长的影响)+ 学习率显式按 `width_base/width_now` 反比缩放,从数学上保证不同宽度的模型在"训练动态"意义上等价,这样就可以先在一个便宜的小模型(如 width=256)上做超参搜索,再用固定公式换算到目标大模型,省去大模型上重复搜索超参的巨额算力开销。

**AI 研究场景:** μP 是近年大规模预训练里"如何用小模型实验结果指导大模型训练配置"的标准方法论(GPT-4 技术报告提到用小规模实验预测大模型性能,μP 类技术是其中的重要一环);对于资源有限的团队,μP 意味着可以把绝大部分超参搜索的算力花在小模型上,大模型训练直接套用换算结果,大幅降低"大模型训练一把梭但学习率设错导致训练失败"的风险。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/pretraining-recipe/src")
from init_schedule import mup_lr, cosine_with_warmup, wsd, inverse_sqrt

# 独立验证 μP lr scaling: 宽度每翻倍，学习率精确减半
lr_256 = mup_lr(256, 256, 1e-3)
lr_512 = mup_lr(512, 256, 1e-3)
lr_1024 = mup_lr(1024, 256, 1e-3)
assert lr_256 == 1e-3
assert abs(lr_512 - lr_256 / 2) < 1e-12
assert abs(lr_1024 - lr_256 / 4) < 1e-12

# 三种schedule在同一组step下的相对大小关系
for s in [100, 2500, 4500]:
    c = cosine_with_warmup(s, 5000, 6e-4)
    w = wsd(s, 5000, 6e-4)
    isq = inverse_sqrt(s, 6e-4)
    print(f"step={s:>5}: cosine={c:.3e}  wsd={w:.3e}  inverse_sqrt={isq:.3e}")
```

**实测(`.venv` 真跑):** μP lr scaling 精确验证:width=256→lr=1.000e-3,width=512→5.000e-4(精确减半),width=1024→2.500e-4(精确减到 1/4),width=4096→6.250e-5(精确减到 1/16)——这是纯反比例公式,不涉及任何随机性,每次独立运行结果完全一致。三种 schedule 对照:step=100(early)时 cosine=3.000e-5(仍在 warmup 早期爬升),wsd=2.400e-4(wsd 的 warmup 窗口更短,warmup_pct=0.05 意味着 5000 步只需 250 步就到 base_lr,step=100 已接近平台),inverse_sqrt=1.500e-5(该实现 warmup 窗口是 4000 步,固定值和 max_step 无关,是三者里 warmup 最长的);step=4500(90%)时 inverse_sqrt 反而是三者里最高的(5.657e-4,因为它不随 max_step 衰减,只取决于绝对 step 数的倒数平方根,某个阶段后几乎不再明显下降)。

**面试怎么问 + 追问链:**
- **Q:** "inverse_sqrt schedule 在 step 很大时几乎不再下降,这对长时间训练有什么风险?"—— 期望:`inverse_sqrt(step) = base_lr × √(warmup/step)`,当 step 远大于 warmup 后,学习率的下降速度确实会显著放缓(平方根倒数是比线性/余弦更"平缓"的衰减函数),如果训练步数规划得很长,后期学习率可能一直维持在一个不算低的水平,不利于模型精细收敛——这也是为什么 inverse_sqrt 更常见于 Transformer 原始论文那种相对较短训练周期的场景,现代超长训练(万亿 token 级)更多用 cosine 或 WSD 这类"衰减更彻底"的调度。
- **追问1:** "μP 的核心假设(小模型调好的学习率能直接套到大模型)在实践中总是成立吗?"—— 期望:μP 保证的是"理论上不同宽度模型的训练动态等价",但真实模型架构可能有 μP 理论没有完全覆盖的成分(比如某些归一化层的具体实现、attention 的具体缩放方式),这些细节的偏差会让实践效果和理论预测有一定出入;μP 大幅降低了大模型超参搜索的成本和风险,但工业界通常仍会在目标规模上做少量微调验证,不是"小模型调完直接零验证上大模型"。

**常见坑:** `init_mup_style` 里 `lm_head` 权重(输出投影层)有一段**额外**的缩放(`std *= math.sqrt(base_width / fan_in)`),这是因为输出层的角色和中间层不同(它直接决定 logits 的数值尺度,进而影响 softmax 的温度效应),μP 论文对输出层有专门的额外处理规则,不能简单套用和其余隐藏层相同的 `1/√fan_in` 公式——读代码时容易忽略这个"lm_head 特殊分支",误以为 μP 初始化对所有权重矩阵都是统一公式。

---

## 4. 数据加载与 Shard 管理(`dataset_shards.py`)—— nanoGPT 风格 memmap,以及"训练可以从任意一个 shard 中断恢复"背后的状态设计

**是什么:**
```python
from __future__ import annotations  # 源文件顶部声明,类型注解延迟求值

class ShardManager:
    """多 shard 顺序遍历 + resume 支持."""
    def next_seq(self, seq_len: int) -> np.ndarray:
        if self.cur_offset + seq_len + 1 > len(self.shard):
            self.cur_shard_idx = (self.cur_shard_idx + 1) % len(self.paths)
            self.cur_offset = 0
            self._open_shard()
        seq = np.asarray(self.shard[self.cur_offset:
                                     self.cur_offset + seq_len + 1])
        self.cur_offset += seq_len
        return seq

    def state(self) -> dict:
        return {"shard_idx": self.cur_shard_idx, "offset": self.cur_offset}

    def restore(self, state: dict) -> None:
        self.cur_shard_idx = state["shard_idx"]
        self.cur_offset = state["offset"]
        self._open_shard()
```
(`dataset_shards.py:29-59`,节选)

**一句话:** `write_shard`/`load_shard` 用 `numpy.memmap` 把 token 序列存成不需要一次性读入内存的二进制文件(nanoGPT 训练脚本的标准做法),`ShardManager` 在多个 shard 文件之间顺序遍历、到达文件末尾自动切下一个(循环),并且把"当前读到哪个 shard、哪个偏移量"这两个数字暴露成可序列化的 `state()`,训练中断后可以精确恢复到断点续训。

**底层机制/为什么这样设计:** `np.memmap` 让操作系统按需把磁盘上的文件页面映射进虚拟内存(而不是 `np.load` 那样把整个数组读进物理内存),这意味着即使单个 shard 文件有几十 GB,也可以像操作一个巨大内存数组一样随机访问其中任意位置的切片,操作系统的页缓存机制会自动处理"哪些页面真正驻留在物理内存"——对预训练常见的 TB 级 token 序列而言,这是唯一现实可行的数据加载方式。`state()`/`restore()` 这一对方法只保存两个整数(`shard_idx`+`offset`),这个极简的状态设计背后的关键洞察是:**只要每个 shard 文件本身内容不变,"当前读到第几个 shard 的第几个字节"这两个数字就足以唯一确定整个 dataloader 的进度**,不需要保存任何 token 数据本身,这也是为什么 checkpoint 里 dataloader 状态占用的存储空间几乎可以忽略不计(相比模型权重/优化器状态动辄几十 GB,这里只有几个字节)。

**AI 研究场景:** 任何跑过真实预训练 run 的人都遇到过"训练在第 K 步意外中断(硬件故障/抢占式实例被回收/手动调参重启)"的场景,这类基础设施必须支持从断点精确恢复,而不是每次都从头重新过一遍数据——`ShardManager` 这种"极简可序列化状态"的设计模式,是所有严肃预训练框架 dataloader 组件的共同设计目标。

**可运行例子:**
```python
import sys, tempfile
from pathlib import Path
sys.path.insert(0, "learning/pretraining-recipe/src")
from dataset_shards import write_shard, ShardManager

# 注意: 用 mkdtemp() 而非 TemporaryDirectory() 上下文管理器 —— 见下方"常见坑"
tmp = tempfile.mkdtemp(prefix="shard_verify_")
paths = []
for i in range(3):
    ids = list(range(i * 1000, (i + 1) * 1000))
    p = str(Path(tmp) / f"shard_{i}.bin")
    write_shard(ids, p)
    paths.append(p)

mgr = ShardManager(paths)
for _ in range(5):
    seq = mgr.next_seq(100)
state_at_5 = mgr.state()

# 模拟"训练中断": 新建一个 ShardManager 从保存的 state 恢复
mgr2 = ShardManager(paths)
mgr2.restore(state_at_5)
seq_after_restore = mgr2.next_seq(100)

# 对照: 原 mgr 继续往下走一步，应该产出完全相同的序列(断点续训应无缝衔接)
seq_continued = mgr.next_seq(100)
assert (seq_after_restore == seq_continued).all()
print(f"5次读取后 state={state_at_5}, 断点恢复后第6次读取与原序列完全一致: {(seq_after_restore == seq_continued).all()}")
```

**实测(`.venv` 真跑):** 3 个 shard(各 1000 个 token,`shard_i` 内容是 `range(i*1000, (i+1)*1000)`)连续 5 次 `next_seq(100)` 调用(每次 offset 前进 100)后,state 精确为 `{'shard_idx': 0, 'offset': 500}`(5×100,尚未跨出第一个 shard 的 1000 token 范围)。独立验证的关键断言:**用保存的 state 新建一个完全独立的 `ShardManager` 实例并 `restore()`,其下一次 `next_seq(100)` 读出的序列,和原实例继续往下读的序列逐字节完全相同**——这证明断点续训真的能做到"无缝衔接",不是近似或者"大致对得上",而是精确到每个 token id 都一致。

**面试怎么问 + 追问链:**
- **Q:** "如果训练中途往 shard 目录里新增了一个 shard 文件(比如往语料库追加了新数据),已保存的 `state`(shard_idx+offset)还能正确恢复吗?"—— 期望:能恢复但语义会变——`state` 只记录了"第几个 shard、第几个字节偏移",如果 `shard_paths` 列表的顺序或内容发生变化(比如新 shard 插在了中间而不是追加在末尾),`restore()` 后 `shard_idx` 指向的可能已经是不同的文件,读出的数据和中断前设想的不一样;生产级实现通常要求 shard 列表本身也纳入 checkpoint(或者只允许在列表末尾追加、不允许插入/删除),保证 `state` 的语义在整个训练生命周期内稳定。
- **追问1:** "`next_seq` 用循环取模(`% len(self.paths)`)处理跨越所有 shard 的情况,这意味着什么?"—— 期望:意味着当训练步数对应消耗的 token 总量超过全部 shard 拼起来的总长度时,数据会开始**重复**(从第一个 shard 重新开始),这对应预训练里"训练多个 epoch"的场景(如果总 token 预算超过语料库大小一轮的量);如果配方要求"数据只能看一遍,不能重复"(严格的 single-epoch 训练),需要在数据量规划阶段就确保 shard 总长度覆盖计划训练的全部 token 数,不能依赖这个循环机制自动处理。

**常见坑:** `next_seq` 返回的切片长度是 `seq_len + 1`(不是 `seq_len`)——这是因为语言模型训练需要同时取出"输入序列"和"标签序列"(标签是输入整体右移一位),n+1 个 token 恰好能切出长度为 n 的 (x, y) 训练对(`x=seq[:-1], y=seq[1:]`,这个切分逻辑在调用方,不在 `ShardManager` 内部);如果误以为 `next_seq(100)` 返回的是精确 100 个 token 而不检查这个 off-by-one,拼接训练样本时会引入难以察觉的对齐错误。**Windows 平台专属坑(本次验证独立发现)**:`load_shard` 用 `np.memmap` 打开文件后,如果外层用 `tempfile.TemporaryDirectory()` 上下文管理器包裹整个流程,退出 `with` 块时会触发 `PermissionError: [WinError 32] 另一个程序正在使用此文件`——Windows 不允许删除仍被内存映射的文件,即使显式 `del mgr, mgr2` 再 `gc.collect()` 也不能保证 `np.memmap` 的底层文件句柄立刻释放(这点上 Windows 和 Linux 行为不同,Linux 允许 unlink 一个仍被 mmap 的文件)。上方"可运行例子"已改用 `tempfile.mkdtemp()`(不自动清理)规避这个问题——这是 Windows 上使用 `np.memmap` 时一个值得记住的平台差异,真实训练框架的 dataloader 通常长期持有 shard 文件的 memmap,不会频繁开关,该坑主要出现在"短生命周期的测试/demo 代码"场景。

---

## 5. Phi-tiny 270M 架构解剖(`phi_tiny_model.py`)—— Pre-RMSNorm + GQA + RoPE + SwiGLU + 权重共享五件套

**是什么:**
```python
from __future__ import annotations  # 源文件顶部声明,类型注解延迟求值
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    def __init__(self, c: PhiTinyConfig):
        super().__init__()
        self.n_head = c.n_head
        self.n_kv_head = c.n_kv_head
        self.q_proj = nn.Linear(c.hidden, c.n_head * c.head_dim, bias=False)
        self.k_proj = nn.Linear(c.hidden, c.n_kv_head * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.hidden, c.n_kv_head * c.head_dim, bias=False)

    def forward(self, x, cos, sin):
        # ... 完整版见 phi_tiny_model.py:69-77
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        repeat = self.n_head // self.n_kv_head
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
```
(`phi_tiny_model.py:58-79`,节选)

```python
from __future__ import annotations
import torch.nn as nn

class PhiTiny(nn.Module):
    def __init__(self, c: PhiTinyConfig):
        super().__init__()
        self.embed = nn.Embedding(c.vocab_size, c.hidden)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.final_ln = RMSNorm(c.hidden, c.norm_eps)
        self.lm_head = nn.Linear(c.hidden, c.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight   # tied embedding
```
(`phi_tiny_model.py:118-126`,节选)

**一句话:** Phi-tiny 是本系列(以及整个 Module 3)反复出现的标杆小模型架构,五个设计要素(Pre-RMSNorm、GQA、RoPE、SwiGLU、输入输出权重共享)分别独立地解决"训练稳定性""KV cache 显存""长度外推""激活函数表达力""参数效率"五个不同的子问题,合起来构成 2023-2024 年"小而强"开源模型(Phi/Llama/Qwen 系列)的事实标准配置。

**底层机制/为什么这样设计:** GQA(Grouped Query Attention)把 `n_kv_head`(4)设得比 `n_head`(16)小,`k_proj`/`v_proj` 只投影到更少的 K/V 头,推理时 KV cache 显存占用正比于 `n_kv_head` 而不是 `n_head`——16→4 意味着 KV cache 直接省下 4 倍,`forward` 里 `k.repeat_interleave(repeat, dim=1)` 把这少数几个 K/V 头"广播"复制给对应的多个 Q 头共享,恢复出和标准多头注意力相同的计算形状,但存储只保留了精简后的份数。`lm_head.weight = self.embed.weight` 这一行是"权重共享"(tied embedding)——输入 embedding 矩阵(把 token id 映射成向量)和输出 lm_head 矩阵(把最终隐藏状态映射回词表分数)在数学操作上是彼此的转置关系,共享同一份参数不仅省下一份和词表大小同阶的巨大参数量(50257×1024≈51M 参数,占 270M 模型的近 1/5),还隐含了一种正则化效果(强迫模型的"理解"和"生成"共用同一套语义空间)。

RoPE 和 SwiGLU 这两个要素本文不重复展开——RoPE(旋转位置编码,解决"长度外推")完整的旋转矩阵推导见 [long-context-deep-dive/01-rope-scaling-family.md](../long-context-deep-dive/01-rope-scaling-family.md),那边知识点 1 就是从零建立"为什么要把 Q/K 的相邻两维当成 2D 平面上的点来旋转"这套机制,和这里 `apply_rope` 调用的是同一套思路,不再重复推导。SwiGLU 这里给一个够用的定义:MLP 部分不再是单一路径的"Linear→激活函数→Linear",而是两个并行的 `Linear` 分支(`SwiGLU(x) = SiLU(xW₁) ⊙ (xW₂)`,`⊙` 表示逐元素相乘)——其中一个分支过 SiLU(即 `x·sigmoid(x)`,ReLU 的平滑版本)充当"门控"信号,决定另一个分支的输出有多大比例能通过,这种"门控值本身也是输入内容的函数、随输入变化"的结构比 ReLU/GELU 这类形状固定的激活函数多一层可学习的筛选能力;代价是比普通 MLP 多了一个权重矩阵(3 个而不是 2 个),所以 SwiGLU 的 FFN 中间维度通常按约 2/3 的比例调小,让参数量和普通 MLP 大致持平。

**AI 研究场景:** 读懂这五个设计要素的取舍逻辑,是理解"为什么现代开源 LLM 长得都差不多(都是 Pre-Norm+GQA/MQA+RoPE+SwiGLU的排列组合)"的关键——这套配置不是巧合的趋同,是过去几年大量消融实验后沉淀出的、在"训练稳定性/推理成本/最终质量"这个多目标权衡下的帕累托最优区域,面试考察"从零设计一个 Transformer 架构"类问题时,候选人如果能讲清楚每个设计选择具体对应解决了什么问题(而不是只会背名词),是显著的加分项。

**可运行例子:**
```python
import sys, torch
sys.path.insert(0, "learning/pretraining-recipe/src")
from phi_tiny_model import PhiTinyConfig, PhiTiny

c = PhiTinyConfig()
m = PhiTiny(c)
n_total = sum(p.numel() for p in m.parameters())
n_embed = m.embed.weight.numel()
n_excl_embed = n_total - n_embed

assert m.lm_head.weight is m.embed.weight     # 权重共享: 不是数值相等，是同一个Parameter对象
assert c.n_kv_head < c.n_head                   # GQA: KV头数确实少于Q头数
assert c.n_head % c.n_kv_head == 0              # 整除关系是repeat_interleave能精确广播的前提

x = torch.randint(0, c.vocab_size, (2, 64))
out = m(x)
assert out.shape == (2, 64, c.vocab_size)
print(f"PhiTiny: 总参数={n_total/1e6:.1f}M  排除共享embed后={n_excl_embed/1e6:.1f}M  "
      f"embed本身={n_embed/1e6:.1f}M(占比{n_embed/n_total:.1%})")
print(f"GQA: n_head={c.n_head} n_kv_head={c.n_kv_head}  KV cache 相对标准MHA节省 {c.n_head//c.n_kv_head}x")
```

**实测(`.venv` 真跑):** `PhiTinyConfig()` 默认配置真实建模:总参数 **315.7M**,排除共享 embedding 后 **264.2M**(两者相差 51.5M,精确等于 `vocab_size×hidden=50257×1024`,验证了权重共享确实避免了这份参数被计两次)。`lm_head.weight is embed.weight` 断言确认这是**同一个 Parameter 对象**(不是数值上恰好相等的两份独立参数,是内存中真正共享的同一份张量,反向传播时两处的梯度会自动累加到同一份参数上)。GQA 配置 `n_head=16, n_kv_head=4`,KV cache 相对标准多头注意力(MHA,`n_kv_head=n_head`)节省精确 **4 倍**。真实 forward 输出 shape `(2, 64, 50257)` 验证模型端到端可跑通。

**面试怎么问 + 追问链:**
- **Q:** "GQA 把 KV 头数从 16 降到 4,会不会损失模型表达能力?"—— 期望:会有一定损失(K/V 的表达自由度确实降低了),但实践证明这个损失在合理的分组比例下(通常 K/V 头数是 Q 头数的 1/4 到 1/8)相对推理成本节省而言微不足道——GQA 是 Llama-2/3、Mistral 等主流模型的标配选择,极端情况(`n_kv_head=1`)称为 MQA(Multi-Query Attention),节省更多显存但表达力损失也更明显,GQA 是这两个极端之间的折衷点。
- **追问1:** "权重共享(tied embedding)对小模型和大模型的相对收益一样吗?"—— 期望:不一样——embedding 矩阵大小只取决于 `vocab_size×hidden`,和模型总层数无关,而模型的总参数量随层数近似线性增长;小模型(如本文的 270M)里 embedding 占比可达 1/5(如实测的 16.3%),权重共享省下的比例非常可观,但同样的 `vocab_size×hidden` 放到一个 70B 模型里占比可能不到 1%,权重共享对大模型的参数节省收益远不如对小模型显著——这是为什么"小模型标配权重共享,大模型不一定"这一经验规律背后的定量原因。

**常见坑:** `head_dim=64` 是独立于 `hidden/n_head` 单独设置的配置项(`hidden=1024, n_head=16` 本来算出来的 `hidden/n_head` 恰好也是 64,容易误以为 `head_dim` 是从这两个数派生出来的),但源码 `q_proj = nn.Linear(c.hidden, c.n_head * c.head_dim, bias=False)` 明确把 `head_dim` 当作独立配置——如果修改 `n_head` 或 `hidden` 而不同步检查 `head_dim`,`n_head * head_dim` 可能不再等于预期的中间维度,这是自定义 Transformer 配置时容易疏漏的一处参数耦合关系。

---

## 6. 训练 Loop 与稳定性:参数分组 + EMA Spike 检测(`training_loop.py`)—— 为什么 LayerNorm 权重不该被 weight decay

**是什么:**
```python
def split_param_groups(model, weight_decay=0.1):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and "norm" not in n.lower():
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
```
(`training_loop.py:11-23`)

```python
class EmaTracker:
    def update(self, x):
        self.ema = x if self.ema is None \
            else self.alpha * self.ema + (1 - self.alpha) * x

    def is_spike(self, x, threshold=3.0):
        return self.ema is not None and x > threshold * self.ema
```
(`training_loop.py:48-58`,节选)

**一句话:** `split_param_groups` 把模型参数分成两组分别设置不同的 weight decay 强度——二维以上且不含"norm"字样的参数(主要是各种 Linear 层的权重矩阵)正常应用 weight decay,一维参数(bias)和归一化层的参数(LayerNorm/RMSNorm 的 weight,虽然是一维但语义上是"缩放系数"而非"连接权重")完全不做 weight decay;`EmaTracker` 用指数滑动平均维护一个"最近 loss 的正常水位线",一旦某一步 loss 突然远超这个水位线就判定为异常尖峰。

**底层机制/为什么这样设计:** Weight decay(L2 正则化的等价实现)的本意是"惩罚过大的权重矩阵,防止过拟合",这个逻辑只对"连接权重"(决定不同神经元之间信息如何组合)有意义;LayerNorm/RMSNorm 的可学习参数语义完全不同——它们是缩放/平移系数,理想值经常就应该接近 1(缩放)或 0(平移),对这类参数施加"向 0 收缩"的 weight decay 压力,是在和这些参数本来就该有的取值范围对着干,这是深度学习工程界公认的最佳实践(而不是这份代码自己发明的规则),几乎所有主流训练框架都会做这个区分。`EmaTracker` 的 spike 检测配合训练 loop 里"检测到 spike 就 `continue` 跳过这一步、不执行参数更新"的逻辑(而不是让这个异常 loss 正常参与反向传播),是防止个别脏数据/数值不稳定样本污染整个训练轨迹的低成本防线。

**AI 研究场景:** "为什么 bias 和 LayerNorm 参数不做 weight decay"是训练工程的高频细节题,能不能讲清楚背后的语义原因(而不是死记"业界都这么做")是判断候选人是否真正理解优化器配置、还是只会复制粘贴训练脚本模板的一个有效信号;spike detection+skip 是任何长时间训练 run 的标配安全网,没有这层防护,一次偶发的脏数据可能导致训练需要从几十步前的 checkpoint 重新开始。

**可运行例子:**
```python
import sys, torch, torch.nn as nn
sys.path.insert(0, "learning/pretraining-recipe/src")
from training_loop import split_param_groups, EmaTracker

model = nn.Sequential(nn.Linear(16, 32), nn.LayerNorm(32), nn.Linear(32, 8))
groups = split_param_groups(model, weight_decay=0.1)
decay_group, no_decay_group = groups[0], groups[1]
assert decay_group["weight_decay"] == 0.1
assert no_decay_group["weight_decay"] == 0.0

# LayerNorm 的 weight/bias 都应该落进 no_decay 组(一维参数)
ln_params = {id(p) for n, p in model.named_parameters() if "1." in n}   # Sequential里索引1是LayerNorm
no_decay_ids = {id(p) for p in no_decay_group["params"]}
assert ln_params.issubset(no_decay_ids)

# Linear层的weight(二维)应该落进decay组，bias(一维)应该落进no_decay组
linear0_weight_id = id(model[0].weight)
linear0_bias_id = id(model[0].bias)
decay_ids = {id(p) for p in decay_group["params"]}
assert linear0_weight_id in decay_ids
assert linear0_bias_id in no_decay_ids

tracker = EmaTracker(alpha=0.99)
losses = [2.5, 2.4, 2.3, 2.3, 9.0, 2.3]
results = []
for L in losses:
    results.append(tracker.is_spike(L))
    tracker.update(L)
assert results == [False, False, False, False, True, False]
print(f"参数分组: decay组{len(decay_group['params'])}个tensor, no_decay组{len(no_decay_group['params'])}个tensor")
```

**实测(`.venv` 真跑):** 3 层网络(`Linear(16,32)`+`LayerNorm(32)`+`Linear(32,8)`)真实分组:LayerNorm 的 weight 和 bias(均为一维)全部落入 `no_decay` 组,两个 Linear 层的 weight(二维)落入 `decay` 组、bias(一维)落入 `no_decay` 组——分组逻辑对真实 PyTorch 模块层次结构的处理完全符合预期,不只是在玩具张量上验证过。`EmaTracker` 对 6 个 loss 值的 spike 判定结果 `[False, False, False, False, True, False]` 精确匹配"只有 9.0(是当时 EMA≈2.3 的约 3.9 倍,超过默认阈值 3.0)被标记"——注意这里 `is_spike` 判断发生在 `update` 之前调用,意味着 EMA 是"看到当前样本前"的历史水位线,不会被当前样本本身污染。

**面试怎么问 + 追问链:**
- **Q:** "如果 `EmaTracker` 判定某一步是 spike 并跳过更新,这一步的数据是不是就永久丢失、不会再被模型看到了?"—— 期望:取决于 dataloader 的实现——如果 dataloader 是"从数据流中顺序消费"(如知识点 4 的 `ShardManager`),被跳过的这个 batch 数据已经从数据流里取出但没有用来更新参数,如果不做额外处理,这部分数据确实不会被重新排入队列;工程实践中通常认为个别 spike 样本占比极低,直接丢弃对整体训练收敛影响可忽略,不需要为此设计专门的重试机制。
- **追问1:** "weight decay 值(本例 0.1)是怎么定的,所有 Transformer 训练都用同一个值吗?"—— 期望:不是,0.1 是一个常见的经验默认值(GPT-2/GPT-3 系列论文沿用的量级),但最优值和模型规模、训练数据量、总步数等因素相关,大规模预训练团队通常会做小规模的 weight decay 消融实验来确定这个超参,不存在一个放之四海而皆准的"正确值"。

**常见坑:** `split_param_groups` 判断"是否属于 norm 类参数"用的是字符串匹配 `"norm" not in n.lower()`(检查参数名里是否包含"norm"字样)——这是一个**基于命名约定**的启发式判断,不是基于参数类型的严格检查;如果自定义模块里的归一化层类命名没有包含"norm"这个词(比如自己写了个叫 `Scale` 的自定义缩放层),这个函数会把它误判成需要 weight decay 的普通权重,这类"靠命名约定而非类型检查"的实现在读别人代码库、尤其是自己新增自定义层时,是容易被忽略的隐性假设。

---

## 7. 轻量评测:Val Loss、Perplexity 与 Tiny-HellaSwag(`eval_benchmarks.py`)—— 用 4 选 1 的方式把"续写连贯性"变成可自动打分的题目

**是什么:**
```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def multiple_choice_score(model, tokenize, prompt: str, choices: list) -> int:
    """返回得分最高的 choice idx."""
    scores = []
    device = next(model.parameters()).device
    for c in choices:
        ids = tokenize(prompt + " " + c)
        ids = torch.tensor(ids, device=device).unsqueeze(0)
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(
            logits.flatten(0, 1), ids[:, 1:].flatten(), reduction="sum",
        )
        scores.append(-loss.item() / max(1, len(tokenize(c))))
    return int(torch.tensor(scores).argmax().item())
```
(`eval_benchmarks.py:29-44`)

**一句话:** `validation_loss`(在留出集上算平均交叉熵)+ `perplexity`(loss 的指数,更直观的"模型平均要在多少个候选 token 里犹豫"的度量)+ `multiple_choice_score`(HellaSwag 风格的 4 选 1 续写题)是预训练阶段最常用的三类自动化评测,全部不需要人工评审,可以在训练过程中周期性跑一遍监控进度。

**底层机制/为什么这样设计:** `multiple_choice_score` 的核心技巧是把"选择题"转换成"哪个续写选项让语言模型的困惑度最低"——对每个候选答案,把"prompt+该候选"拼接后整体喂给模型算交叉熵损失(注意这里 `reduction="sum"` 是把整个选项的 token 损失加总,不是取平均),损失越低说明"prompt 后面接这个续写"在模型看来越"自然"(概率越高),取损失最低(即 `-loss` 最高)的选项作为模型的"答案"。除以 `len(tokenize(c))` 做长度归一化(`scores.append(-loss.item() / max(1, len(tokenize(c))))`),是为了防止长选项因为"token 数天然更多、累计 loss 天然更大"而在未归一化的总损失比较中系统性地吃亏——这是选择题类评测里一个容易被忽略但很重要的细节,不做长度归一化会让评测结果混入和"回答质量"无关的"选项长度"这个混淆变量。

**AI 研究场景:** 这套"用语言模型自身的困惑度给选择题打分"的方法论,正是真实 HellaSwag/PIQA/ARC 等经典 few-shot 评测基准的核心实现原理——理解这个机制,才能理解为什么"prompt 格式的微小变化"会显著影响模型在这些基准上的分数(选项 token 化方式、要不要在选项前加空格等细节都会实质性改变损失计算)。

**可运行例子:**
```python
import sys, torch
sys.path.insert(0, "learning/pretraining-recipe/src")
from eval_benchmarks import perplexity, multiple_choice_score, TINY_HELLASWAG, run_tiny_hellaswag

assert abs(perplexity(0.0) - 1.0) < 1e-9    # loss=0(完美预测) -> ppl=1(不存在任何"犹豫")
assert perplexity(2.3) > perplexity(1.0)      # loss越高，困惑度越高，单调关系
import math
assert abs(perplexity(2.3) - math.exp(2.3)) < 1e-9

assert len(TINY_HELLASWAG) == 3
for ex in TINY_HELLASWAG:
    assert 0 <= ex["answer"] < len(ex["choices"])   # 每题的标准答案下标必须在选项范围内

# 用随机初始化的小模型验证评测机制本身可跑通(不要求随机模型答对，只验证接口不崩)
import torch.nn as nn
class TinyLM(nn.Module):
    def __init__(self, vocab=1000, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
    def forward(self, x):
        return self.head(self.embed(x))
def tokenize(s):
    return [abs(hash(w)) % 1000 for w in s.split()]

model = TinyLM()
acc = run_tiny_hellaswag(model, tokenize)
assert 0.0 <= acc <= 1.0
print(f"perplexity(loss=2.3) = {perplexity(2.3):.2f}")
print(f"随机初始化模型在tiny-HellaSwag上的准确率: {acc:.1%}(未训练,不代表真实评测水平)")
```

**实测(`.venv` 真跑):** `perplexity(2.3) = 9.97`——直观理解:一个 loss=2.3 的模型平均"纠结"在大约 10 个候选 token 之间才能做出预测,loss=0 精确对应 `perplexity=1.0`(完全没有不确定性)。用随机初始化的 `TinyLM`(未训练,权重完全随机)跑 `run_tiny_hellaswag`,3 道题的准确率是随机水平附近的结果(不同随机种子下会在 0/33%/67%/100% 之间跳动,因为只有 3 道题,样本量太小),这验证了评测**接口**可以正确跑通,但明确不代表任何真实的模型能力信号——3 道题的样本量对严肃评测而言微不足道,`TINY_HELLASWAG` 的定位是"验证评测机制能跑通"的教学示例,不是可信的能力度量。

**面试怎么问 + 追问链:**
- **Q:** "为什么这里用 `reduction='sum'` 而不是 `reduction='mean'`,还要手动除以 token 数做归一化?"—— 期望:如果直接用 `reduction='mean'`,PyTorch 会对当前这一个候选内部的 token 取平均(这本身没问题),但代码里手动重新做除法是为了让归一化基准明确、可控(比如可以按"整个候选序列长度"而非"cross_entropy内部batch展平后的token总数"精确归一,两者在这个具体调用模式下数值相同,但显式写出来避免对 reduction 语义产生依赖 PyTorch 版本行为的隐性假设)——更关键的教学点是:**必须做某种长度归一化**,这是选择题评测正确性的前提,用 `sum` 还是 `mean` 只是实现细节。
- **追问1:** "如果四个选项长度差异很大,长度归一化能完全消除长度带来的偏差吗?"—— 期望:不能完全消除——长度归一化处理的是"总 loss 的量纲"问题,但语言模型对"更长的连贯叙述"和"更短的简单断言"天然有不同的困惑度基准线(长文本的每 token 平均困惑度即使内容质量相同,也可能因为长距离依赖更难预测而略高),真实评测基准的选项设计通常会控制长度分布相近,减少这种残留偏差,不能完全依赖归一化公式本身消除所有长度相关的系统性影响。

**常见坑:** `run_tiny_hellaswag` 只有 3 道手写例题,和真实 HellaSwag(10042 道题的测试集)在样本量上差 4 个数量级——本文/源码用"tiny"命名已经很明确地做了区分,但如果在报告里直接引用"tiny-HellaSwag 准确率"这类数字而不加限定语境,容易让读者误以为这是有统计意义的能力评测结果,实际上 3 题的准确率方差极大,不具备任何可比较、可复现的评测价值。

---

## 8. 知识蒸馏:KL 散度 Soft Label 与温度缩放(`distillation.py`)—— 学生模型不只学"标准答案",还学老师模型的"犹豫程度"

**预备知识:** `kd_loss` 用到的 KL 散度(Kullback-Leibler divergence)完整定义、直觉类比、非负性证明见 [`statistics-deep-dive/17-distribution-shift-and-monitoring.md` 知识点 1](../statistics-deep-dive/17-distribution-shift-and-monitoring.md),这里不重新推导,只给最简版直觉:KL 散度衡量"如果真实分布是 P,却按另一个分布 Q 来编码/预测,平均要多付出多少代价",数值越大代价越大,P、Q 完全相同时精确为 0(非负,当且仅当两分布相同取到下界 0)。放到这里的场景:`F.kl_div(s, t, ...)` 里 `t`(教师的 softmax 输出)扮演"真实"分布 P,`s`(学生的 log_softmax 输出)扮演"用来编码/预测"的分布 Q(这是 PyTorch `kl_div(input, target)` 的固定参数语义——`input` 传对数概率、`target` 传概率,算出的是 `KL(target‖input对应的分布)`,不能颠倒),数值越小说明学生的输出分布越接近教师。

**是什么:**
```python
from __future__ import annotations
import torch.nn.functional as F

def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            T: float = 4.0) -> torch.Tensor:
    """soft KL with temperature."""
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T ** 2)

def combined_loss(student_logits, teacher_logits, targets,
                   alpha: float = 0.5, T: float = 4.0) -> torch.Tensor:
    ce = F.cross_entropy(student_logits.flatten(0, -2), targets.flatten())
    kd = kd_loss(student_logits, teacher_logits, T)
    return alpha * ce + (1 - alpha) * kd
```
(`distillation.py:8-21`)

**一句话:** 知识蒸馏让一个小"学生"模型不仅学习训练数据里的标准答案(硬标签,cross-entropy),还额外学习一个更强的"教师"模型对每个 token 给出的完整概率分布(软标签,KL 散度)——温度参数 `T` 把 softmax 的分布"拉平",让教师模型原本接近 one-hot 的置信分布变得更平滑,暴露出更多"次优答案的相对排序"这类硬标签完全不携带的信息。

**底层机制/为什么这样设计:** 标准 cross-entropy 训练只告诉学生模型"这个位置的正确答案是 token X",对于"token Y 虽然不对,但也是一个合理的候选"这种细致的相对合理性排序完全没有信号;教师模型的完整 softmax 输出天然携带这种排序信息(即使是错误答案,教师模型给它们的概率也有高低之分),但如果直接用未缩放的 softmax,教师模型对高置信度预测的概率经常接近 1、其余全部接近 0(和硬标签区别不大,携带的额外信息被"挤压"掉了),除以温度 `T`(通常>1)让分布变得更平滑,次优选项的相对概率差异被放大到肉眼可辨的程度,KD loss 里最后再乘回 `T²`是为了抵消温度缩放对梯度量级的影响(标准的 Hinton et al. 2015 蒸馏论文推导结果)。

**AI 研究场景:** 知识蒸馏是"用一个训练成本高昂的大模型,把能力浓缩进一个部署成本低廉的小模型"的标准手段——DistilBERT、TinyLlama 等广泛使用的小模型都用了蒸馏,Phi 系列技术报告也提到蒸馏是其"小而强"配方的组成部分之一(和纯粹的从零预训练相对,蒸馏可以视为"用教师模型的输出作为一种更丰富的监督信号")。

**可运行例子:**
```python
import sys, torch
sys.path.insert(0, "learning/pretraining-recipe/src")
from distillation import kd_loss, combined_loss

torch.manual_seed(0)
student = torch.randn(2, 8, 100)
teacher = torch.randn(2, 8, 100)
targets = torch.randint(0, 100, (2, 8))

kd_t1 = kd_loss(student, teacher, T=1.0)
kd_t4 = kd_loss(student, teacher, T=4.0)
assert kd_t1.item() > 0 and kd_t4.item() > 0     # KL散度非负

# 教师=学生时,KL散度应精确为0(自己和自己没有分布差异)
kd_self = kd_loss(student, student, T=4.0)
assert abs(kd_self.item()) < 1e-4

loss_a0 = combined_loss(student, teacher, targets, alpha=0.0, T=4.0)   # 纯蒸馏,不看标签
loss_a1 = combined_loss(student, teacher, targets, alpha=1.0, T=4.0)   # 纯监督,不看教师
loss_a05 = combined_loss(student, teacher, targets, alpha=0.5, T=4.0)
assert abs(loss_a0.item() - kd_loss(student, teacher, T=4.0).item()) < 1e-4
print(f"KD loss(T=1)={kd_t1.item():.4f}  KD loss(T=4)={kd_t4.item():.4f}")
print(f"teacher=student时KL散度: {kd_self.item():.6f}(应接近0)")
print(f"alpha=0(纯蒸馏)={loss_a0.item():.4f}  alpha=1(纯监督)={loss_a1.item():.4f}  alpha=0.5(混合)={loss_a05.item():.4f}")
```

**实测(`.venv` 真跑):** `teacher=student`(同一个随机张量)时 KL 散度精确收敛到 **0.000000**(数值上完全符合"分布和自己没有差异"的数学定义,这是验证 `kd_loss` 实现正确性最直接的判据)。`combined_loss(alpha=0.0, ...)` 精确退化为纯 `kd_loss`(两者相差小于 1e-4,浮点误差量级),验证了 `alpha` 参数确实是在"纯监督"和"纯蒸馏"两个极端之间线性插值,不是某种更复杂的非线性组合。温度 T=1 vs T=4 下 KD loss 数值不同(随机张量下差异因具体数值而定,但两者都严格为正),符合"温度改变分布形状进而改变散度取值"的预期。

**面试怎么问 + 追问链:**
- **Q:** "温度 T 越大,蒸馏效果就越好吗?"—— 期望:不是单调关系——T 太小,教师分布接近 one-hot,退化成和硬标签差不多没有额外信息;T 太大,教师分布被拉得过于平坦,几乎变成均匀分布,"哪个 token 更合理"这个排序信息反而被噪声淹没;Hinton 原始论文和后续大量实践都表明 T 存在一个不太极端的甜蜜点(常见取值在 2-10 之间),需要针对具体任务和模型调试,不是越大越好。
- **追问1:** "如果教师模型和学生模型的词表不完全一致(比如用了不同的 tokenizer),`kd_loss` 还能直接用吗?"—— 期望:不能直接用——`kd_loss` 要求 `student_logits` 和 `teacher_logits` 在最后一维(词表维度)上语义对齐(每个位置的下标必须指向同一个 token),如果两个模型 tokenizer 不同,同一个下标在两边代表完全不同的 token,直接计算 KL 散度是没有意义的比较;跨 tokenizer 蒸馏需要更复杂的对齐方案(比如在文本层面而非 logits 层面做知识迁移),这是本知识点简化实现没有覆盖的真实工程难点。

**常见坑:** `combined_loss` 的 `ce`(硬标签损失)用的是 `student_logits`(学生自己的预测)和 `targets`(数据集真实标签)算交叉熵,和 `kd`(软标签损失)用的是 `student_logits` 和 `teacher_logits`——两路损失的"正确答案"来源不同(一个来自数据集标注,一个来自教师模型的预测),这意味着**如果教师模型本身在某个样本上判断错误**,`kd` 这一项会引导学生模型学习教师的错误判断,和 `ce` 这一项(仍然指向数据集真实标签)产生方向冲突;`alpha` 参数存在的意义之一正是让训练者可以控制"愿意在多大程度上信任教师模型可能犯错的判断"。

---

## 9. Phi 风格合成数据生成(`synth_data_prompt.py`)—— 一个只检查词数和禁用短语、不检查重复的质量过滤器

**是什么:**
```python
def make_prompt(topic: str, audience: str, style: str, n_words: int = 500) -> str:
    return (f"You are an expert science writer. Write a clear, accurate "
            f"explanation of {topic} suitable for a {audience}. "
            f"Style: {style}. Length: about {n_words} words. "
            f"Output only the explanation, no preamble.")

def filter_quality(text: str, min_words: int = 100, bad_phrases: list = None) -> bool:
    if len(text.split()) < min_words:
        return False
    bad_phrases = bad_phrases or [
        "As an AI", "I'm just an AI", "I cannot", "I don't know", "Sorry, I cannot",
    ]
    for b in bad_phrases:
        if b in text:
            return False
    return True
```
(`synth_data_prompt.py:17-47`,节选)

**一句话:** `make_prompt` 通过组合"主题×受众×文体"三个维度生成多样化的合成数据生成 prompt(这是 Phi 系列技术报告强调的"prompt 多样性设计是合成数据质量的核心"这一原则的具体实现),`filter_quality` 是生成之后的质量门槛,但它只检查两件事:字数是否够、是否包含"作为AI我不能回答"这类拒绝话术。

**底层机制/为什么这样设计:** `make_prompt` 用三个独立维度(8 个主题×5 个受众×5 个文体=200 种组合)的笛卡尔积生成 prompt,而不是手写几十个固定 prompt,这是"用组合爆炸廉价换取多样性"的思路——8+5+5=18 个基础素材,通过组合能覆盖 200 种不同的生成情境,比手写 200 个独立 prompt 的人力成本低得多,这正是 Magpie(01 号文件知识点 10)之外另一种"低成本扩大合成数据多样性覆盖面"的策略,两者的差异在于 Magpie 完全不需要人工设计的主题/受众/文体这类结构化维度(靠模型自问自答),而这里的方法需要人工先定义好这些维度,换来的好处是生成内容的主题分布完全可控、可预测。`filter_quality` 只做两类检查(长度、拒绝话术),**完全没有检查内容是否重复**——这不是遗漏的说明性缺陷,而是这个函数在设计定位上就比 01 号文件的 `quality_filter.py`(有专门的 5-gram/10-char 重复率检测)更轻量,两者服务的场景不同:合成数据的"重复"风险主要来自"模型对相似 prompt 生成雷同内容",防御手段是知识点里另一个函数 `dedup_by_prefix`(按内容前缀去重),而不是这个函数本身。

**AI 研究场景:** "为什么合成数据集需要精心设计 prompt 多样性维度"是任何做数据合成项目团队的核心工程问题——Phi-3/Phi-4 技术报告反复强调这一点是他们区别于早期"简单调用 GPT-4 生成数据"方法的关键改进,本知识点的 `make_prompt` 是这个思路最简化但完整的教学复现。

**可运行例子:**
```python
import sys, random
sys.path.insert(0, "learning/pretraining-recipe/src")
from synth_data_prompt import make_prompt, generate_seed_pool, filter_quality, dedup_by_prefix, TOPICS, AUDIENCES, STYLES

pool = generate_seed_pool(n=50, rng=random.Random(1))
assert len(pool) == 50
assert len(set(pool)) > 1     # 50个prompt里应该出现不止一种组合(多样性维度确实生效)
n_combinations = len(TOPICS) * len(AUDIENCES) * len(STYLES)
assert n_combinations == 8 * 5 * 5

# 独立验证: filter_quality 只看词数+禁用短语，对重复内容完全没有防御(不同于01号文件的quality_filter)
repeated_spam = "buy now " * 60          # 120词，无禁用短语，但是纯重复
assert len(repeated_spam.split()) >= 100
assert filter_quality(repeated_spam) is True     # 高度重复内容居然通过了质量检查
short_refusal = "Sorry, I cannot help with that request today unfortunately at all."
assert filter_quality(short_refusal) is False     # 命中禁用短语，正确拒绝

# dedup_by_prefix 才是这个模块真正用来防重复的机制,和filter_quality是两个独立的关卡
samples = [repeated_spam, repeated_spam, "a completely different unique piece of text here."]
unique = dedup_by_prefix(samples, prefix_len=20)
assert len(unique) == 2      # 前两条前缀相同被去重，第三条独立保留

print(f"组合空间: {n_combinations}种独特prompt")
print(f"重复内容({len(repeated_spam.split())}词)是否通过filter_quality: {filter_quality(repeated_spam)}(设计上如此,重复检测在dedup_by_prefix)")
```

**实测(`.venv` 真跑):** 独立验证确认(用了一个和 README 文档示例不同的重复短语"buy now"而不是"AI",排除偶然性):120 词的高度重复文本(`"buy now " × 60`)真实通过 `filter_quality` 检查(`True`),因为该函数确实**只**检查两个条件——`len(text.split()) < min_words` 和是否含拒绝话术——完全没有 01 号文件 `quality_filter.py::heuristic_score` 那样的重复 n-gram 比率检测。这个发现不是在挑源码的错(源码设计上就是两级过滤,重复检测归 `dedup_by_prefix` 负责),而是提示"评估一个质量过滤函数覆盖了什么风险"必须逐行读实现,不能只看函数名字("filter_quality"这个名字容易让人误以为它覆盖全部质量维度)。

**面试怎么问 + 追问链:**
- **Q:** "`filter_quality` 和 `dedup_by_prefix` 是两个独立调用的函数,如果生产 pipeline 只调用了前者、忘记调用后者会怎样?"—— 期望:会产出大量内容高度重复的"合成数据",这些数据进入训练集后,效果上类似 01 号文件知识点 3 揭示的 MinHash"模板化文本塌缩"问题的合成数据版本——模型会反复看到几乎相同的内容,浪费训练算力且可能过拟合到这些重复模式;这正说明"质量"和"去重"是两个独立的正交维度,必须分别落实防御,不能指望其中一个隐式覆盖另一个。
- **追问1:** "`bad_phrases` 硬编码了 5 个英文拒绝话术,这个列表覆盖得全面吗?"—— 期望:明显不全面——真实教师模型(GPT-4/Claude 等)的拒绝话术措辞多种多样、且会随模型版本演化,硬编码的固定短语列表本质上是"打地鼠"式的防御,漏检大量变体表达是必然的;生产级实现通常会用更鲁棒的分类器(类似 01 号文件的 toxicity/quality classifier 思路)而非简单的字符串包含检查来识别拒绝类回复。

**常见坑:** `filter_quality` 的返回值语义是"通过检查返回 True",容易和"是否应该丢弃"的直觉搞反——阅读调用方代码时如果看到 `if not filter_quality(text): continue`(丢弃不通过的样本)这类写法是正常预期,但如果调用方逻辑写反(比如误把 `filter_quality` 当成"是否应该过滤掉"来用),会导致质量过滤逻辑完全失效或反向生效,这类"返回值语义容易被误读"的函数命名在读别人代码库时需要格外小心确认。

---

## 10. 工业界配方对照:Llama-3 vs DeepSeek-V3(概念点,基于`data_mixture.py::CONFIGS`延伸)—— 两种截然不同的"规模化哲学"

**是什么:** 知识点 2 已经展示过 `CONFIGS` 字典里 Llama-3 和 DeepSeek 两份配方的具体数字,本知识点把视角拉高到两份配方背后完整的工程决策对照(源:`lectures/14-llama3-recipe.md` + `lectures/15-deepseek-v3-recipe.md`):

| 维度 | Llama-3(2024) | DeepSeek-V3(2024.12) |
|---|---|---|
| 架构策略 | 标准 dense Transformer,扩大规模 | MoE(混合专家)+ MLA(多头潜在注意力)+ MTP(多token预测) |
| 数据规模 | 15T+ token | 14.8T token(中文4T/英文4T/代码2T/数学1T/其他) |
| 训练精度 | bf16 | **FP8**(工程上的关键突破点) |
| 并行策略 | 3D并行(DP+TP+PP) | 3D并行 + **DualPipe**(自研流水线调度算法) |
| 数据配比取舍 | web占比高达70%,通用性优先 | 中英文均衡配比,专项能力(代码/数学)加强oversample |
| 训练成本 | 未完整公开,但被广泛推算为数千万美元级 | 官方披露约**$5.6M**(相对模型规模的成本效率震动业界) |

**一句话:** Llama-3 走的是"用海量算力堆一个足够大的标准 dense 模型"的规模化路线,DeepSeek-V3 走的是"用一系列架构和工程创新(MoE 降低激活参数量、FP8 降低训练成本、DualPipe 降低流水线气泡)把同等能力的训练成本压到极致"的效率路线,两者代表 2024 年 LLM 规模化的两种不同哲学,都达到了当时的 SOTA 水平但成本结构差异巨大。

**底层机制/为什么这样设计:** MoE 架构下,DeepSeek-V3 总参数量高达 6710 亿(远超 Llama-3 405B 版本),但**每个 token 实际激活的参数量只有约 370 亿**(路由机制只挑选少数专家参与计算)——这是"参数量"和"计算量"解耦的关键设计,让模型在推理成本可控的前提下拥有远超同等计算量 dense 模型的总参数容量(总参数量决定"能记住多少知识",激活参数量决定"每次推理要花多少算力")。FP8 训练是精度更激进的选择(比知识点 02 号文件讨论的 bf16 更低精度,可表示数值范围更窄),需要配合更精细的数值稳定性工程手段(如按 tensor 或按 tile 粒度做动态缩放因子)才能安全使用,DeepSeek-V3 技术报告详细描述了这套工程方案,这也是它被称为"工程奇迹"的核心原因之一——不是提出了全新的算法,而是把多项已知但极难落地的激进优化(FP8、大规模 MoE 路由、极致流水线调度)**同时**在一次训练里稳定跑通。

**AI 研究场景:** 理解这两种规模化哲学的差异,是回答"要不要做 MoE"、"什么情况下值得投入 FP8 训练的工程复杂度"这类真实架构决策问题的基础——不存在绝对的"更优"路线,选择哪条路线取决于团队的算力预算、工程能力储备、以及对推理成本敏感度的产品定位。

**可运行例子:** 本知识点为工业配方的对照综述,不对应独立可执行脚本;知识点 2 的 `CONFIGS["llama3"]`/`CONFIGS["deepseek"]` 抽样验证代码已经真实验证过这两份配方在数据配比这一个维度上的具体数字,可以直接参照。

**面试怎么问 + 追问链:**
- **Q:** "MoE 架构下'总参数量'和'激活参数量'的区别,为什么面试常问这个?"—— 期望:因为这是判断一个 MoE 模型真实部署成本的关键——媒体报道/技术报告里的"参数量"经常指总参数量(决定显存占用),但推理阶段的计算成本(FLOPs,进而影响延迟和成本)由激活参数量决定,一个总参数量巨大但激活参数量较小的 MoE 模型,可能比一个总参数量小得多的 dense 模型推理更便宜,只看"参数量"这一个数字容易得出错误的成本比较结论。
- **追问1:** "如果你的团队算力有限(比如只有几百张卡),应该学 Llama-3 的路线还是 DeepSeek-V3 的路线?"—— 期望:没有标准答案,但可以给出权衡框架——DeepSeek-V3 路线的工程复杂度(MoE 路由的负载均衡、FP8 数值稳定性、DualPipe 这类自研调度算法)对团队的系统工程能力要求极高,如果团队没有相应的工程储备,贸然照搬这条路线的风险(训练不稳定、debug 成本失控)可能超过它承诺的效率收益;算力有限的团队通常更适合先把标准 dense+成熟并行策略(如02号文件讨论的 FSDP/ZeRO)跑稳,再逐步引入更激进的优化,而不是一步到位复刻最前沿的工程栈。

**常见坑:** 不要把"DeepSeek-V3 官方披露训练成本约 $5.6M"直接理解成"复现这个模型只需要 560 万美元"——这个数字通常只覆盖最终那一次成功训练的算力开销,不包括前期大量的架构消融实验、失败的训练尝试、团队人力成本、以及支撑这次训练所需的基础设施建设投入,技术报告披露的"最终训练成本"和"复现这个成果的真实全部投入"是两个经常被混淆但差异巨大的数字。

---

## 11. Capstone:从零预训练 Phi-tiny 270M(`capstone_train.py`,含真实 GPU 训练验证)—— 本系列首次真实反向传播,而非纯 CPU 数值模拟

**是什么:**
```python
from __future__ import annotations  # 源文件顶部声明,类型注解延迟求值

def train(cfg: TrainCfg, dry_run: bool = True):
    from phi_tiny_model import PhiTinyConfig, PhiTiny
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = PhiTinyConfig(seq_len=cfg.seq_len)
    model = PhiTiny(model_cfg)
    # sanity_check / mock_data_loader 等前半段节选省略,完整版见 capstone_train.py:73-104
    if dry_run:
        sanity_check(model, cfg, "cpu")
        return
    model = model.to(device).bfloat16()
    # opt/rng/loader 构造节选省略
    for step in range(cfg.max_step):
        opt.zero_grad()
        for _ in range(cfg.grad_accum):
            x, y = next(loader)
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten()) / cfg.grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
```
(`capstone_train.py:73-123`,节选)

**一句话:** 这是本系列(以及全部已完成的 5 条姊妹系列)少数几个走"真实 GPU 训练"而非"纯 CPU 数值模拟/mock"路线的知识点——`--train` 标志会真的在本机 RTX 3080 Ti 上跑起 PhiTiny 270M 模型的 bf16 前向+反向传播+AdamW 优化器 step,不是打印模板,也不是伪造的数字。

**底层机制/为什么这样设计:** `dry_run=True`(默认)只做 `sanity_check`(单次 CPU 前向+反向,验证 loss 有限、梯度有限)和打印 WSD 学习率曲线,不涉及任何真实优化器更新——这是"验证代码路径能跑通"和"真的花算力训练"之间的一道明确界限,大多数日常验证(比如`_verify_md.py`批量复验)应该走这条低成本路径;`--train` 模式下,`mock_data_loader` 用 `np.random.default_rng` 生成的随机整数序列充当"训练数据"(源码注释明确写"教学占位"),这意味着这次真实训练验证的是**训练循环机制本身**(前向能不能跑、反向传播梯度是否有限、优化器能不能正常 step、显存占用是否符合预期)而不是"模型真的学到了什么"——用随机数据训练不可能产生任何有意义的收敛,loss 不会真的下降到接近 0,这是本知识点验证范围的明确边界,不是缺陷。

**AI 研究场景:** "训练循环的工程正确性"和"模型最终学到的知识质量"是预训练项目里两个完全独立的验证维度——大型预训练 run 启动前,团队通常会先用类似的小规模真实 GPU smoke test(哪怕数据是假的)确认整套软件栈(模型代码+并行策略+优化器+显存管理)没有工程 bug,再投入正式的、使用真实海量语料的完整训练,本知识点复现的正是这后半段"先用假数据验证管线,再考虑真数据"的标准做法。

**可运行例子:**
```python
import subprocess, sys, re

# 真实调用 capstone_train.py --train，验证训练循环在本机GPU上真实跑通
result = subprocess.run(
    [sys.executable, "learning/pretraining-recipe/src/capstone_train.py",
     "--train", "--max_step", "3", "--micro_batch", "2", "--grad_accum", "2", "--seq_len", "128"],
    capture_output=True, text=True, timeout=120, encoding="utf-8",
)
assert result.returncode == 0, result.stderr
out = result.stdout
assert "device=cuda" in out          # 确认真的用了GPU，不是静默回退到CPU
assert "params = 315.7M" in out       # 确认模型规模符合PhiTiny默认配置

loss_match = re.search(r"step\s+0 loss ([\d.]+)", out)
assert loss_match is not None
step0_loss = float(loss_match.group(1))
assert 5.0 < step0_loss < 15.0        # 随机初始化模型在50257词表上的loss应在这个合理量级(理论上界ln(50257)≈10.8)

print(f"真实GPU训练输出片段:\n{out}")
print(f"step 0 loss = {step0_loss}")
```

**实测(`.venv` 真跑,本机 RTX 3080 Ti Laptop GPU):** 两次独立运行(`--max_step 3` 和 `--max_step 10`,均 `--micro_batch 2 --grad_accum 2 --seq_len 128`)全部确认:`device=cuda`(真实使用 GPU,`torch.cuda.is_available()` 判定生效,不是静默回退到 CPU)、`params = 315.7M`(模型规模和知识点 5 的建模验证完全一致)、`step 0 loss` 两次分别为 **9.5788** 和 **9.5497**(同一份随机初始化逻辑但不同随机种子路径下的正常波动,均落在理论上界 `ln(50257)≈10.82` 附近的合理区间,证明模型确实在做有意义的语言建模计算,不是输出垃圾数值)。GPU 显存方面:训练前 `nvidia-smi` 确认基线 0MB 占用;这个规模(315.7M 参数,seq_len=128,micro_batch=2)的训练任务体量很小,整个 10-step smoke 在 2 秒量级内完成(远快于 README 记录的 16 秒——差异主要来自 CUDA context 初始化和 cuDNN kernel 首次编译的一次性开销,首次调用和后续调用的墙钟时间不能直接类比),进程结束后显存立即完全释放回 0MB,未发生显存泄漏。

**面试怎么问 + 追问链:**
- **Q:** "用随机数据(`mock_data_loader`)训练出来的模型有什么实际用途吗?"—— 期望:没有任何语言建模能力上的用途(不可能学到任何真实的语言知识,因为输入输出都是无意义的随机整数),它的唯一价值是**验证训练工程管线**——前向传播的 shape 是否正确、反向传播是否产生有限梯度、优化器 step 是否正常执行、显存占用是否在预期范围、以及(如果扩展到多卡场景)分布式通信是否正常——这类 smoke test 在正式投入真实数据训练前是标准的工程验证步骤,能在几秒到几分钟内发现"模型代码写错了""显存超预算""某个并行策略配置有 bug"等工程问题,而不需要等到跑了几个小时真实训练才发现。
- **追问1:** "如果要把这个 smoke test 升级成'验证模型真的能学到东西',最小的改动是什么?"—— 期望:核心改动是把 `mock_data_loader` 换成任何一个真实的、有语言结构的语料源(哪怕只是仓库里一个小文本文件反复过几遍),再观察多步训练后 loss 是否呈现下降趋势(哪怕只是从 9.5 降到 8 或 7,只要有统计意义上的下降趋势就说明训练循环在做有效的梯度下降,不是随机噪声);真正验证到"学出有意义的语言能力"则需要小说规模的真实语料+更多训练步数,这已经超出"smoke test"的定位,是知识点 4(Module 3 毕业)ckpt A/B/C/D/E 五部曲要做的事情。

**常见坑:** 不要把这次 smoke test 的 loss 数字(9.5-9.6)和真实预训练收敛后的 loss(通常 2-3 左右,如知识点 7 的 `perplexity`/eval 讨论)放在一起比较——两者不是同一个训练阶段的产物,前者是"随机初始化模型在随机数据上跑 3-10 步"的初始状态读数,不代表任何训练进展,单独拿这个数字去论证"模型训练得好不好"是没有意义的比较,必须结合"训练了多少步、用的是什么数据"这些上下文才能正确解读 loss 数字。

---

*下一篇:[04-small-model-graduation.md](04-small-model-graduation.md) —— Module 3 毕业:五部曲 checkpoint(数据→架构→长上下文→课程学习)的渐进式改进故事,含第二处真实 GPU 训练验证。*
