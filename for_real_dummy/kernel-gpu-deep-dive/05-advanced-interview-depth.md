# 05 · 进阶深度追加:4 个真实二面级别的多级追问链案例

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是知识点,不计入统计。

## 为什么需要这篇追加内容

`01-04` 全部完成并独立验证通过之后,用户转达了一位有经验从业者的反馈:现有材料没有达到 **2026 年大厂技术二面** 的深度。`dsa-deep-dive/20-advanced-interview-depth.md` 已经基于一次真实调研(检索中国大厂面经、西方大厂面经、面试官视角的元讨论)落地验证过一套追加格式——真实的追问不是"正确性→复杂度→能不能优化"这一条线性链,而是至少沿着 **5 条独立轴线** 展开:

1. **规模递增轴**——数据/负载规模一级一级往上跳,原方案会失效。
2. **工程约束递增轴(并发/分布式)**——单机正确→并发安全→分布式扩展。
3. **方案批判迭代轴**——面试官连续指出具体的工程缺陷,逼你换方案,不是"不够好"这种空话。
4. **决策依据追问轴**——不纠错,只逼问"你是怎么考虑选这个不选那个的"。
5. **真实性验证轴**——把"做了优化""自动选择了最优配置"这类抽象表述,追问压向具体数字、具体验证过程。

那次调研还发现一个现有材料完全没覆盖的题型:**给定一段真实日志/trace,诊断系统实际发生了什么**。这个题型天然适配"分布式系统的调用日志"这种载体——`dsa-deep-dive` 用缓存重试风暴的日志片段落地了这个题型。但 `kernel-gpu-deep-dive` 01-04 全部建立在纯 Python 数值/机制模拟之上(GPU 算力表、访存字节数公式、autotune 打分公式),不产生真实的系统调用日志,这篇追加不强行套这个题型去凑数——下面案例 4 会借用 `occupancy()` 返回的字典(字段名和 Nsight Compute 真实 profiler 报告里的 `achieved_occupancy`/`limiting_factor` 几乎一一对应,01 类知识点 4 已经点出这层关系)做一次"读一段结构化诊断输出、倒推工程决策"的练习,算是在这条系列的素材边界内,对这个题型精神的一次借用,而不是伪造一段"系统日志"。

组织原则和范围声明,和 `dsa-deep-dive/20-advanced-interview-depth.md` 一致:下面 4 个案例,每个都明确标注建立在 01-04 哪个已有知识点之上,包含完整还原的多级追问链(带参考答案)和真实验证过的可运行例子——而且专门换了不同于 01-04 正文的具体参数(不同矩阵尺寸、不同精度、不同 GPU 对、不同 tile 切法)重新验证一遍,不是照抄已有数字。**这不是要把 01-04 的 19 个知识点全部重写**——读者应该能把同样的思路自己套用到任何一个已有知识点上练习追问。

---

## 案例 1:Triton autotune 打分平局——"自动"选中的是最优解,还是候选列表的书写顺序?(真实性验证轴)

建立在 [03 类](03-kernel-design-triton-cutlass.md) 知识点 2 的发现之上:03 类已经验证过 `autotune(4096,4096,4096)` 选中 128×256 tile,并证明 128×256 和 256×128 两个候选的打分**位级完全相等**,选中 128×256 纯粹是 Python `max()` 在平局时"谁先出现选谁"的实现细节,不是算法判断谁更优(交换 `configs` 参数顺序会让结果翻转到 256×128)。这个案例不满足于"复现了一次平局",继续往下逼问:这个平局是这个打分公式的**通用性质**,还是只在特定条件下才出现的**特例**?

**追问链条完整还原:**

- **Q(基础,03 类已覆盖):** "4096³ 的大 GEMM,autotune 选中了 128×256 的 tile,为什么?"——期望答出"`reuse × num_stages` 这一项在所有可行候选里分数最高"。
- **追问 1(真实性验证,把"autotune"这个名字的字面承诺摆到台面上):** "'autotune' 这个名字暗示它会自动找到'最优'配置。你说 128×256 和 256×128 是位级相等的平局,选中 128×256 完全是列表顺序决定的——这句话你怎么当场证明给我看,而不是读文档相信作者写的注释?"——期望候选人现场手动重算两个候选的 `score`(不能只信任 `autotune()` 的返回值本身),再交换 `configs` 列表顺序验证输出会翻转,两步都要跑代码,不能停留在"我记得文档这么写"。
- **追问 2(深挖,逼问这是不是巧合):** "如果换一个矩阵形状,这个平局还会出现吗?平局是这个打分公式对所有输入都成立的性质,还是只在特定条件下出现?"——期望候选人能推导出 `reuse = block_m*block_n/(block_m+block_n)` 这一项的公式本身根本不含 `M`、`N`,对 `(128,256)`/`(256,128)` 这对输入天然对称(乘法和加法都满足交换律),所以真正能打破平局的只有 `parallelism_penalty` 这一项——而这一项只在 `n_tiles < 132` 时才非零。对于两个 tile 配置都让 `n_tiles` 落在 132 以上的"大而方"的 GEMM,惩罚项恒为 `0 == 0`,永远打平;但对"瘦" GEMM(比如批量很小的 decode 场景),`n_tiles` 会跌破 132、并且两个候选的 `n_tiles` 不再相等,平局就会真实打破。
- **深挖追问(现场验证一个具体的"瘦"场景):** "给你 M=128,N=4096,K=4096 这个'瘦'GEMM(对应 02 类 roofline capstone 里的 `gemm-128-4k-4k` 形状),128×256 和 256×128 还会打平吗?"——期望现场跑代码验证:不再打平,256×128 拿到决定性更高的分数,而且交换列表顺序验证选择结果不变(证明这次是真实赢,不是顺序伪影)。
- **决策依据追问(收尾,逼问工程后续,不是"发现了问题就结束"):** "知道了这个打分公式在特定条件下会打平局,如果这是一个真实生产系统的 autotuner,你会怎么改,让选择结果不再依赖候选列表的书写顺序这种偶然因素?"——期望候选人提出至少两条路:①打平时增加一个确定性的第二关键字兜底(而不是隐式依赖列表顺序);②更根本的做法(03 类"AI 研究/工程场景"段已经点出)是像真实 Triton 的 `@triton.autotune` 一样,最终跑真实 benchmark 的 wall-clock 时间决胜负,而不是完全依赖一个封闭形式的打分公式估算;还可以提到"至少要让这类平局在 CI 里被检测出来并报警,而不是被静默隐藏"。

**可运行例子(1/2):换两个比 4096³ 更大的方阵尺寸重新验证平局——如实记录,不是硬凑结论:**

```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\kernel-engineering\src")
from triton_style import TritonConfig, autotune

def score(c, M, N):
    reuse = c.block_m * c.block_n / max(1, c.block_m + c.block_n)
    n_tiles = ((M + c.block_m - 1) // c.block_m) * ((N + c.block_n - 1) // c.block_n)
    penalty = 0 if n_tiles >= 132 else (132 - n_tiles) * 0.1
    return reuse * c.num_stages - penalty

c_wide = TritonConfig(128, 256, 32, num_warps=8, num_stages=4)
c_tall = TritonConfig(256, 128, 32, num_warps=8, num_stages=4)

# 换两个和 03 类正文(4096^3)不同的方阵尺寸:6144^3 和 12288^3
for size in (6144, 12288):
    s_wide, s_tall = score(c_wide, size, size), score(c_tall, size, size)
    chosen = autotune(size, size, size)
    assert s_wide == s_tall, f"size={size}: 期望仍然打平,实际 {s_wide} vs {s_tall}"
    assert (chosen.block_m, chosen.block_n) == (128, 256), "列表里 128x256 排在前面,平局时应继续选中它"
    print(f"OK: M=N={size}, score_wide=score_tall={s_wide:.4f}(位级相等), autotune 选中 "
          f"{(chosen.block_m, chosen.block_n)}")

# reuse 这一项本身,对任意 (block_m, block_n) 互换都恒等对称,和 M/N 完全无关
# 这是平局在"大而方"场景下必然复现的根本原因,不是 4096 这个具体数字的巧合
import random
rng = random.Random(3)
for _ in range(5):
    bm, bn = rng.randint(32, 512), rng.randint(32, 512)
    reuse_1 = bm * bn / max(1, bm + bn)
    reuse_2 = bn * bm / max(1, bn + bm)
    assert reuse_1 == reuse_2, f"reuse 公式对 (block_m,block_n) 互换应该位级对称: {bm},{bn}"
print("OK: 随机抽样 5 组 (block_m,block_n),reuse 公式互换后位级相等——这一项从设计上就和 M/N 无关。")
```

**可运行例子(2/2):"瘦" GEMM 形状下平局被打破,且不再依赖列表顺序——这次是真实赢,不是伪影:**

```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\kernel-engineering\src")
from triton_style import TritonConfig, autotune

def score(c, M, N):
    reuse = c.block_m * c.block_n / max(1, c.block_m + c.block_n)
    n_tiles = ((M + c.block_m - 1) // c.block_m) * ((N + c.block_n - 1) // c.block_n)
    penalty = 0 if n_tiles >= 132 else (132 - n_tiles) * 0.1
    return reuse * c.num_stages - penalty

c_wide = TritonConfig(128, 256, 32, num_warps=8, num_stages=4)   # 128x256
c_tall = TritonConfig(256, 128, 32, num_warps=8, num_stages=4)   # 256x128

# 呼应 02 类 roofline capstone 的 "瘦" GEMM 形状:M=128(小 batch/decode 场景),N=4096
M, N = 128, 4096
s_wide, s_tall = score(c_wide, M, N), score(c_tall, M, N)

assert s_wide != s_tall, "瘦 GEMM 场景下,两个候选的分数应该真实分出高下,不再是平局"
assert s_tall > s_wide, "256x128(高瘦 tile)应该在 M 很小时得分更高"
margin = s_tall - s_wide
print(f"OK: M={M},N={N} 时 score(128x256)={s_wide:.4f}, score(256x128)={s_tall:.4f}, "
      f"差距={margin:.4f}(不是位级相等)")

# 关键验证:交换候选列表顺序,结果应该完全不变——证明这次是真实赢,不是列表顺序在打平局
chosen_a = autotune(M, N, 4096, configs=[c_wide, c_tall])
chosen_b = autotune(M, N, 4096, configs=[c_tall, c_wide])
assert (chosen_a.block_m, chosen_a.block_n) == (chosen_b.block_m, chosen_b.block_n) == (256, 128)
print(f"OK: 不论 configs=[wide,tall] 还是 [tall,wide],autotune 都选中 (256,128)——"
      f"顺序不再能左右结果,这是打分公式给出的真实判断,不是 03 类那个平局案例的顺序伪影。")

# 顺手验证这个差异的根源:两个候选的 n_tiles 在这个形状下不再相等
n_tiles_wide = ((M + c_wide.block_m - 1)//c_wide.block_m) * ((N + c_wide.block_n - 1)//c_wide.block_n)
n_tiles_tall = ((M + c_tall.block_m - 1)//c_tall.block_m) * ((N + c_tall.block_n - 1)//c_tall.block_n)
assert n_tiles_wide != n_tiles_tall
print(f"OK: n_tiles(128x256)={n_tiles_wide}, n_tiles(256x128)={n_tiles_tall}——"
      f"M=128 太小,两个候选切出的 tile 网格形状不再对称,parallelism_penalty 不再相等,平局的前提条件被打破。")
```

**常见坑:** 把"128×256 和 256×128 打平"当成这个打分公式**任意输入**下都成立的结论,一路套用到所有场景——实际上平局只在"两个候选的 `n_tiles` 都 ≥132"时成立(`reuse` 项恒对称是必要条件,但 `parallelism_penalty` 恒为 0 才是平局真正成立的充分条件);面试官换一个"瘦" GEMM 形状追问时,如果继续背诵"反正是平局、看列表顺序",会当场露馅——这次是真实分出高下的决策,不是顺序伪影,混淆这两种情况说明只记住了结论没理解成因。另一个坑是把"发现了一个平局"当成终点,答不出"生产系统该怎么修"这个决策依据追问,一个封闭形式的启发式打分公式在设计阶段几乎不可能穷尽所有对称性陷阱,真正的 autotuner 靠真实 benchmark 兜底,不是靠公式本身足够精巧。

---

## 案例 2:FlashAttention 的 HBM 节省曲线——流量节省曲线之外,显存容量这堵墙什么时候拦住你?(规模递增轴 + 工程约束递增轴)

建立在 [04 类](04-flashattention-and-fusion.md) 知识点 2/5 的 capstone 曲线之上:04 类已经验证过从 512 到 131072(128k)token,flash attention 相对 naive attention 的 HBM **流量**节省从 5.0x 一路涨到 1025.0x,公式是 `speedup = 1 + N/d`,04 类"常见坑"也已经点出"不要把访存流量被优化和显存占用被优化混为一谈"。这个案例把"规模继续往上跳"这条轴线走到底:先验证公式在 128k 之外还成不成立,再把"KV cache 装不装得下"这句话从空话变成一次真实的字节数计算——即使 flash attention 把访存流量压到极致,序列长到一定程度后,单卡显存**容量**这堵墙依然会把你逼向分布式。

**追问链条完整还原:**

- **Q(基础,04 类已覆盖):** "128k 上下文,flash attention 把 HBM 流量节省到 1025 倍,这是不是意味着 128k 上下文的显存问题已经被解决了?"——04 类已经给出"不是,这条曲线只覆盖访存流量,KV cache 存储仍是 O(N) 容量瓶颈"这个方向性答案,这里继续往下追问到具体数字。
- **追问 1(规模递增,先验证公式往上还能不能外推):** "如果上下文不是 128k,而是 100 万 token,这条曲线还成立吗?具体数字是多少?"——期望候选人先用公式 `1+N/d` 现场估算,再用真实函数验证。这一步的意义不是数字本身,而是确认"节省倍数会随 N 无限线性增长"这件事——正因为这个数字没有上限,才更容易让人产生"只要上了 FlashAttention,多长的上下文都不是问题"这种错觉。
- **追问 2(捅破"流量节省"和"装得下"的区别,给出具体计算):** "你说的这几千倍,衡量的是这次 attention 计算过程中往返 HBM 的字节总量,用完就释放。但推理时真正长期占着显存不放的是 KV cache 本身——这是'容量'问题,不是'流量'问题。给你一个具体的 8B 参数模型(32 层,8 个 KV head,每头 128 维,fp16 KV cache),100 万 token 的上下文,KV cache 本身要占多少字节?装得进一张 H100(80GB)吗?"(这里"8 个 KV head"用的是 GQA/Grouped Query Attention 配置——多个 query head 分组共享同一份更少数量的 K/V head,目的正是压缩 KV cache 体积,不是说这个模型总共只有 8 个 attention head,下面案例 4 的 `n_kv_heads` 也是同一个字段)——期望现场推出"每 token 的 KV cache 字节数 = 2(K 和 V)× 层数 × KV head 数 × head_dim × dtype_bytes"这个公式,代入算出 128KB/token,乘以 100 万 token 得到约 137GB,超过 H100 刨除模型权重后剩下的显存预算。
- **深挖追问(工程约束递增,逼向分布式,但先问清楚选项而不是直接给答案):** "既然一张 H100 装不下,你有哪些选项?"——期望能提出"换更大显存的卡"(单卡路线)和"多卡切分 KV cache"(分布式路线)两类;深入分布式路线时,期望能现场算出"换 2 张 H100 做 tensor parallel(权重和 KV head 预算都对半分),够不够撑住 100 万 token"这个具体阈值。
- **决策依据追问(收尾,逼问选择依据而不是背答案):** "单卡换 H200/B200,和多卡 TP 切 H100,都能不同程度撑住 100 万上下文,你会怎么选?"——没有标准答案,期望候选人能提出至少两个决策维度(比如"单卡方案没有跨卡通信开销和拓扑复杂度,但受限于当前买得到的最大单卡显存;多卡方案更灵活,但要先确认算法层面站不站得住脚")。如果追问更深一层,期望能答出"04 类知识点 1 的 online softmax 递推(`m`/`l`/`o` 三个 running state)在数学上和 block 具体从哪来无关,Ring Attention 就是把'下一个 block' 换成'环形通信收到的下一份 K/V 分片',是同一套递推——所以分布式不需要重新证明数学正确性";还可以提到这个 8B 模型的 `n_kv_heads=8` 天然给 tensor parallel 切 KV cache 定了一个上限(超过 8 卡,KV head 就要开始被复制而不是真正被分摊,继续往上加卡帮 KV cache 分摊,要换成按序列切分的 sequence/context parallel,也就是 Ring Attention 这一路)。

**可运行例子(1/2):把 04 类的曲线外推到 100 万 / 400 万 token,验证公式没有失效:**

```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\kernel-engineering\src")
from capstone_attn_speedup import hbm_naive_attn, hbm_flash_attn

d = 128  # 和 04 类 capstone 一致的 head_dim
for N in (1_048_576, 4_194_304):        # 04 类曲线止步于 131072,这里继续往上跳两级
    naive_bytes = hbm_naive_attn(N, d)
    flash_bytes = hbm_flash_attn(N, d)
    speedup = naive_bytes / flash_bytes
    formula = 1 + N / d
    assert abs(speedup - formula) < 1e-6, "speedup 应该精确等于 1+N/d,不是近似"
    print(f"OK: N={N:>9,}  naive={naive_bytes/1e9:9.2f}GB  flash={flash_bytes/1e6:8.2f}MB  "
          f"speedup={speedup:,.1f}x(公式预测 {formula:,.1f}x)")

# 100万 token 时,单看这条"节省倍数"曲线,数字大到 8193x —— 但这只是访存流量的比值,
# 完全没有回答"KV cache 这个持续占用的存储,装不装得进一张卡"这个问题,下一段现场算这个
```

**可运行例子(2/2):KV cache 容量算式——装不装得下是流量节省曲线回答不了的问题:**

```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\gpu-architecture\src")
from common import GPUS

# 这是本节新引入的场景/公式,不是 learning/ 源码里已有的函数——
# 只复用 common.py::GPUS 里真实的显存容量字段(hbm_gb),KV cache 容量公式是本节现场推导的
n_layers, n_kv_heads, head_dim, dtype_bytes = 32, 8, 128, 2     # 8B 量级模型的典型 GQA 配置
kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
assert kv_bytes_per_token == 131072                              # 128KB/token
weight_bytes = 8e9 * 2                                            # 8B 参数,bf16 权重
assert weight_bytes == 16e9                                       # 16GB 权重,单卡 H100 装得下权重本身

N_target = 1_048_576                                               # 100 万 token 上下文
kv_needed = N_target * kv_bytes_per_token
print(f"100万 token 的 KV cache 需要 {kv_needed/1e9:.2f} GB(不含模型权重)")

results = {}
for name in ("H100", "H200", "B200"):
    gpu = GPUS[name]
    budget = gpu.hbm_gb * 1e9 - weight_bytes           # 显存刨掉权重之后,剩下能装 KV cache 的预算
    fits = kv_needed <= budget
    results[name] = fits
    print(f"{name}: 显存={gpu.hbm_gb}GB, 权重占用后剩余预算={budget/1e9:.1f}GB, "
          f"能装下100万token的KV cache吗? {fits}")

assert results["H100"] is False and results["H200"] is False   # 单卡 H100/H200 都装不下
assert results["B200"] is True                                  # 只有单卡 B200(192GB)能独自装下

# 装不下怎么办:2 卡 H100 做 tensor parallel,权重和 KV cache 预算都按卡数均分
h100 = GPUS["H100"]
for G in (1, 2):
    per_gpu_budget = h100.hbm_gb * 1e9 - weight_bytes / G
    total_capacity_tokens = (per_gpu_budget * G) / kv_bytes_per_token
    fits_1m = total_capacity_tokens >= N_target
    print(f"H100 x{G}(tensor parallel): 每卡权重={weight_bytes/G/1e9:.1f}GB, "
          f"总KV容量={total_capacity_tokens:,.0f} token, 能撑住100万token吗? {fits_1m}")
    if G == 1:
        assert not fits_1m
    else:
        assert fits_1m   # 2 卡 H100 就能撑住,不需要等到买 B200
```

**常见坑:** 把"访存流量节省了 N 倍"和"显存占用降低了 N 倍"混为一谈——`speedup_curve()` 算的是 attention 计算这一步往返 HBM 的字节总量,用完即释放;KV cache 是跨越整个生成过程持续占用的存储,两者是完全不同维度的资源,flash attention 对后者的字节数没有任何影响(它从不减少 KV cache 本身需要保存的 K、V 张量大小)。另一个坑是看到"分布式"就想复杂,忘了 04 类已经证明 online softmax 递推和 block 来源无关——多卡场景不需要重新证明数学正确性,真正新增的工程成本是通信,不是算法本身要改。还有一个容易漏掉的点:GQA 的 `n_kv_heads` 给纯 tensor parallel 切分 KV cache 定了一个天然上限,超过这个卡数就不能再指望简单地"多买几张卡把 KV head 摊得更薄"。

---

## 案例 3:Ridge Point 非单调之外的第二份证据——B200 到 GB200,参数表更强背后的效率打折扣(真实性验证轴)

建立在 [02 类](02-roofline-model.md)知识点 3 的发现之上:02 类已经验证过 H100→H200 这次升级里,BF16 算力(`bf16_tflops`)一个数字都没变,但 HBM 带宽涨了 43.3%,导致 ridge point 反而从 295.2 降到 206.0——"更新的硬件不代表所有维度都更优"。这个案例不满足于"这只是 H100/H200 这一对的巧合",从两个此前没人验证过的角度重新独立复核同一个结论:①这个 30.2% 的降幅是不是 BF16 独有的巧合,换成 FP8 精度会不会不一样;②除了 ridge point,能不能在一个此前 01-04 全系列都没碰过的字段(`tdp_w`,功耗)上,找到另一对完全独立的 GPU 也复现"参数表更强 ≠ 每个维度都更优"这个结论。

**追问链条完整还原:**

- **Q(基础,02 类已覆盖):** "H100 升级到 H200,ridge point 反而从 295.2 降到 206.0,这是不是意味着 H200 不如 H100?"——02 类已有答案:不是,只是带宽涨了、算力没变,ridge point 是比值下降,不代表卡变差。
- **追问 1(真实性验证,逼问"这只是 BF16 这一个精度下的巧合吗"):** "这 30.2% 的降幅,只在 BF16 这个精度下成立吗?换成 FP8 会不会不一样?"——期望候选人现场推导:H100 和 H200 的 `fp8_tflops` 字段(和 `bf16_tflops` 一样)也完全相等,所以 FP8 的 ridge point 理应有和 BF16 完全相同的下降百分比——因为降幅纯粹是带宽比值决定的,分子(算力)在哪个精度上都没变,这个比例关系不可能因为换了精度就变。
- **追问 2(深挖,逼问"这是不是只有 H100/H200 这一对显卡才有的特例"):** "除了 H100/H200 这一对,你能不能找到另一个完全独立的例子,也能验证'参数表上更强的新卡,不代表每个指标都更好'这个结论?"——引导到 `GPUSpec` 里从没被 01-04 系列用过的 `tdp_w`(功耗)字段:现场算出 B200 的 TFLOPS/Watt(2.25)比 GB200(2.0833)反而更高——即"参数表上算力更强(2500>2250 TFLOPS)、显存带宽打平(都是 8.0TB/s)"的 GB200,单位功耗算力反而比 B200 更差。
- **深挖追问(逼问这背后是不是同一类数学结构):** "H100→H200 的 ridge point 下降,和 B200→GB200 的功耗效率下降,这两件事背后是不是同一类原因?"——期望候选人能抽象出共同结构:两者都是"一个比值,分子涨得比分母慢(甚至分子没涨、分母却涨了)"——ridge point 是算力/带宽,H200 带宽涨、算力没涨;功耗效率是算力/功耗,GB200 的 TDP 涨幅(1.2 倍)超过了算力涨幅(1.111 倍)。厂商在参数表上突出的那个"更强"的绝对数字,可能是靠拉高另一个成本维度换来的,只看单一绝对数字容易被误导,要看比值。
- **决策依据追问(收尾,落到一个具体的采购/部署场景):** "如果你的机房散热/功耗预算是固定的(比如每机柜功率上限固定),你会选多卡 B200 还是少卡 GB200?"——没有标准答案,期望候选人能提出"同样的功率预算下,B200 能塞下更多张卡、单位功耗算力更高,总算力未必更低;但 GB200 单卡绝对性能更强,而且 01 类 NVLink 知识点已经建过模的 NVL72 域原生支持 72 卡全互联,如果工作负载对跨卡通信延迟/带宽极度敏感,拓扑优势可能盖过单位功耗效率的劣势",考察的是能不能同时权衡多个互相牵扯的维度,而不是死记"效率更高=更好"这种单一标准。

**可运行例子(1/2):FP8 精度下 ridge point 非单调性的独立复现——不是 BF16 独有的巧合:**

```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\gpu-architecture\src")
from common import GPUS

h100, h200 = GPUS["H100"], GPUS["H200"]

# H100/H200 的 FP8 峰值算力也完全相等(和 02 类已经验证过的 bf16 一样)
assert h100.fp8_tflops == h200.fp8_tflops == 1979.0

def ridge_point(spec, tflops_field):
    peak = getattr(spec, tflops_field)
    return (peak * 1e12) / (spec.hbm_tb_s * 1e12)

rp_h100_bf16 = ridge_point(h100, "bf16_tflops")
rp_h200_bf16 = ridge_point(h200, "bf16_tflops")
rp_h100_fp8 = ridge_point(h100, "fp8_tflops")
rp_h200_fp8 = ridge_point(h200, "fp8_tflops")

assert round(rp_h100_bf16, 1) == 295.2 and round(rp_h200_bf16, 1) == 206.0   # 02 类已验证的 bf16 数字
assert round(rp_h100_fp8, 1) == 590.7 and round(rp_h200_fp8, 1) == 412.3     # fp8 通道下的新验证

drop_bf16 = 100 * (1 - rp_h200_bf16 / rp_h100_bf16)
drop_fp8 = 100 * (1 - rp_h200_fp8 / rp_h100_fp8)
assert abs(drop_bf16 - drop_fp8) < 1e-9   # 两个精度下的降幅位级相等,不是"差不多"

print(f"OK: bf16 ridge point H100={rp_h100_bf16:.1f} -> H200={rp_h200_bf16:.1f}, 降幅={drop_bf16:.4f}%")
print(f"OK: fp8  ridge point H100={rp_h100_fp8:.1f} -> H200={rp_h200_fp8:.1f}, 降幅={drop_fp8:.4f}%")
print(f"两个精度下降幅完全相等,证明这不是 bf16 这一个精度下凑出来的巧合——"
      f"根源是带宽涨了43.3%这个硬件事实,不随你选哪个精度通道去看而改变。")
```

**可运行例子(2/2):B200 到 GB200,功耗效率反直觉下降——全新指标,全新 GPU 对,同一个结论:**

```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\gpu-architecture\src")
from common import GPUS

b200, gb200 = GPUS["B200"], GPUS["GB200"]

tpw_b200 = b200.bf16_tflops / b200.tdp_w
tpw_gb200 = gb200.bf16_tflops / gb200.tdp_w
assert round(tpw_b200, 4) == 2.25
assert round(tpw_gb200, 4) == 2.0833
assert tpw_b200 > tpw_gb200   # 参数表更强的 GB200,单位功耗算力反而更低

compute_ratio = gb200.bf16_tflops / b200.bf16_tflops     # 算力涨了多少倍
tdp_ratio = gb200.tdp_w / b200.tdp_w                       # 功耗涨了多少倍
assert round(compute_ratio, 3) == 1.111
assert round(tdp_ratio, 3) == 1.2
assert tdp_ratio > compute_ratio   # 功耗涨得比算力快,单位功耗效率必然下降——纯除法关系,不是巧合

print(f"OK: B200 TFLOPS/W={tpw_b200:.4f}, GB200 TFLOPS/W={tpw_gb200:.4f}")
print(f"OK: 算力涨幅={compute_ratio:.4f}x, 功耗涨幅={tdp_ratio:.4f}x(功耗涨得更快)")

# 这个结论在 bf16/fp8/fp4 三个精度通道下同步成立,不是只在某一档精度下偶然出现
for field in ("bf16_tflops", "fp8_tflops", "fp4_tflops"):
    eff_b200 = getattr(b200, field) / b200.tdp_w
    eff_gb200 = getattr(gb200, field) / gb200.tdp_w
    assert eff_b200 > eff_gb200, f"{field} 通道下 B200 应该仍然更省电"
print("OK: bf16/fp8/fp4 三个精度通道下,B200 的单位功耗算力全部高于 GB200——"
      "这不是某一档精度的孤例,是 compute_ratio < tdp_ratio 这个纯算术关系在每个精度通道上的必然结果。")
```

**常见坑:** 只验证了 H100/H200 这一对就断言"新一代 GPU 都这样",没有意识到需要换一个独立的字段/独立的 GPU 对才能排除"这只是这一对显卡凑巧"的可能性——真实性验证轴的核心纪律是"换一组输入再测一次",不是"再读一遍同一个结论加深印象"。另一个坑是看到"B200 更省电"就直接推出"B200 全面比 GB200 好"——GB200 原生支持的 NVL72 全互联域(01 类已建模)是 B200 单卡不具备的能力,效率指标只是决策的一个维度,不是唯一维度,这和 02 类"常见坑"里"ridge point 更低不等于卡更差"是同一类需要多维度权衡、不能单一指标一票定优劣的陷阱。

---

## 案例 4:给 Fused MLP 中间激活值选存储层——从"字节数装得下"到"占用率被挤到只剩 1 个 block"(决策依据追问轴 + 方案批判迭代轴)

建立在 [01 类](01-gpu-hardware-and-memory.md)知识点 2(`recommend_tier`)和知识点 4(`occupancy` 的五重瓶颈模型)之上,并联系 [04 类](04-flashattention-and-fusion.md)知识点 3 的 fused MLP 场景(`H=16384` 的 FFN 隐藏维)。这个案例演示"方案批判迭代"这种追问模式:候选人先给出 `recommend_tier` 这个简化模型的天真建议,面试官不满足于"这个建议字面上说得通",连续用两层具体的、可以现场算出数字的工程约束把这个建议逼到需要重新设计——面试官甩出来的不是一段场景描述,而是直接读 `occupancy()` 返回的字典(字段名和 Nsight Compute 真实 profiler 报告里的 `achieved_occupancy`/`limiting_factor` 几乎一一对应),让你现场诊断。

**追问链条完整还原(方案批判迭代,不是深挖同一方案):**

- **面试官给场景:** "04 类的 fused MLP 里,`H=16384`(FFN 隐藏维),fp16 精度,每一行算出来的中间结果 `h_row` 会被同一行后续 `D=4096` 次矩阵乘法累加复用。你会把 `h_row` 放哪一级存储?"
- **候选人方案 1(直接套用 01 类的 `recommend_tier`):** "`h_row` 是 16384 个 fp16 元素,32768 字节=32KB,重用次数(D=4096)远超所有阈值,`recommend_tier(32768, 4096)` 返回 `registers`——放寄存器。"
- **面试官指出具体缺陷(不是"这样不够快",是"寄存器语义上装不下这种数据"):** "寄存器是每个线程私有的存储,不是一个 block 内所有线程能共享寻址的一块内存。`h_row` 这 16384 个元素要被同一个 block 内很多线程按下标去取用(矩阵乘法的规约维),如果真放进'寄存器',要么每个线程各自私藏一份完整的 16384 元素——这个代价有多大,你能现场算一下吗?"
- **候选人现场计算并意识到问题:** 假设一个 block 有 256 个线程,如果每个线程都私有一份完整的 `h_row` 副本,总共需要 `32KB × 256 = 8MB`,而整个 SM 的寄存器堆总共只有 256KB——超出预算 32 倍,这条路径行不通。`recommend_tier` 只检查了"字节数是否 ≤256KB"和"重用次数够不够",完全没检查这块数据的**访问模式**是不是"寄存器"这个存储层次的语义装得下的东西。
- **候选人方案 2(换成 SMEM 暂存,更符合"block 内线程共享寻址"的语义):** "那应该用共享内存,每个 block 先把它负责的几行的 `h_row` 搬进 SMEM,线程再从 SMEM 里按下标取值。"
- **面试官指出新方案的代价(具体量化,不是"占用率可能会低"这种空话):** "如果为了减少 kernel launch 次数、让每个 block 一次多处理几行以提升数据复用,把每个 block 处理的行数从 1 行提到 4 行,SMEM 占用从 32KB 涨到 128KB,这时候占用率会发生什么?"——期望候选人现场用 `occupancy()` 算出:1 行/block 时占用率 0.5(瓶颈其实是 `regs`,不是 `smem`);4 行/block 时 SMEM 占用 128KB,只够放 1 个 block/SM,占用率暴跌到 0.125,跌了 4 倍。
- **深挖追问(继续加码,逼近 03 类 SMEM 预算过滤的"无解"边界):** "如果一次性处理 8 行,占用率会怎样?"——期望现场算出 8 行=256KB,超过 H100 每 SM 的 228KB SMEM 总预算,`occupancy()` 算出 `blocks_per_sm=0`——一个 block 都放不进一个 SM,这已经不是"占用率低",是这个配置根本没法用来启动 kernel(呼应 03 类 `autotune()` 超预算直接 `raise ValueError` 的防御性设计逻辑)。
- **决策依据追问(收尾,逼问真实取舍而不是"选最小的那个"这种偷懒答案):** "那是不是应该无脑选 1 行/block,反正占用率最高?"——期望候选人不要掉进"occupancy 越高越好"这个 01 类知识点 4 追问 2 里已经出现过的陷阱,而是能提出:1 行/block 虽然占用率最高,但每个 block 处理的行数越少,同一份从 HBM 搬来的权重矩阵能在一次 kernel launch 内摊薄的访存开销就越少(更细碎的调度意味着更多次重复搬运权重),真正的最优 tile size 需要综合权衡"occupancy"和"每次 launch 能摊薄多少访存"两个目标,不能拿 occupancy 一个指标当唯一标准。

**可运行例子(1/2):`recommend_tier` 给出的建议,以及"寄存器"在语义上为什么装不下这块数据:**

```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\gpu-architecture\src")
from memory_hierarchy import recommend_tier
from sm_occupancy import H100_SM

# 04 类 fused_mlp.py 的 FFN 隐藏维 H=16384,fp16(2字节),同一行内被后续 D=4096 次乘加复用
H, dtype_bytes, D_reuse = 16384, 2, 4096
h_row_bytes = H * dtype_bytes
assert h_row_bytes == 32768   # 32KB

tier = recommend_tier(h_row_bytes, D_reuse)
assert tier.name == "registers"   # recommend_tier 的天真建议:字节数小、重用次数够多 -> 放寄存器
print(f"OK: recommend_tier({h_row_bytes}, {D_reuse}) -> {tier.name}(这是本节要批判的第一个方案)")

# 现场算清楚"放寄存器"这句话如果按字面意思(每个线程私有一份完整副本)要付出多大代价
total_register_file_bytes = H100_SM.max_regs_per_sm * 4   # 每个寄存器 4 字节(32-bit)
assert total_register_file_bytes == 262144                 # 256KB,和 01 类 H100_HIERARCHY 里 registers 一档一致

threads_per_block = 256
naive_replicate_bytes = h_row_bytes * threads_per_block     # 每个线程都私藏一份完整 h_row
overshoot_ratio = naive_replicate_bytes / total_register_file_bytes

assert naive_replicate_bytes == 8_388_608                   # 8MB
assert overshoot_ratio == 32.0                                # 恰好超出整个 SM 寄存器堆 32 倍

print(f"OK: 一个 block {threads_per_block} 个线程,如果每个线程私藏一份完整 h_row({h_row_bytes} 字节),"
      f"总共需要 {naive_replicate_bytes/1024/1024:.1f}MB, 而整个 SM 的寄存器堆只有 "
      f"{total_register_file_bytes/1024:.0f}KB —— 超出 {overshoot_ratio:.0f} 倍,这条路径行不通。"
      f"recommend_tier 检查的只是'字节数装不装得下',没检查'这种访问模式和寄存器的语义匹不匹配'。")
```

**可运行例子(2/2):SMEM tile size 从 1 行加到 8 行,占用率被具体数字逼到 0——方案批判的每一步都能现场算出来:**

```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\gpu-architecture\src")
from sm_occupancy import occupancy, H100_SM

H, dtype_bytes = 16384, 2
h_row_bytes = H * dtype_bytes
threads_per_block, regs_per_thread = 256, 64

records = {}
for rows_per_block in (1, 2, 4, 8):
    smem_kb = (h_row_bytes / 1024) * rows_per_block
    r = occupancy(threads_per_block, regs_per_thread, smem_kb, H100_SM)
    records[rows_per_block] = r
    print(f"rows_per_block={rows_per_block}: smem/block={smem_kb:6.1f}KB -> "
          f"blocks_per_sm={r['blocks_per_sm']}, occupancy={r['occupancy']}, bottleneck={r['bottleneck']}")

# 1 行/block:瓶颈其实是寄存器,不是 SMEM(这一步容易被想当然地归咎于 SMEM)
assert records[1]["bottleneck"] == "regs"
assert records[1]["occupancy"] == 0.5

# 2 行/block 起,SMEM 变成新瓶颈,占用率开始下滑
assert records[2]["bottleneck"] == "smem"
assert records[2]["occupancy"] < records[1]["occupancy"]

# 4 行/block:占用率比 1 行/block 暴跌到 1/4
assert records[4]["occupancy"] == records[1]["occupancy"] / 4
assert records[4]["blocks_per_sm"] == 1

# 8 行/block:彻底超出 228KB 的 SMEM 总预算,一个 block 都放不进去
assert records[8]["blocks_per_sm"] == 0
assert records[8]["occupancy"] == 0.0

print("OK: rows_per_block 从 1 加到 8,occupancy 从 0.5 一路跌到 0.0(彻底放不下)——"
      "'多处理几行以减少调度开销'这个直觉,每往前一步付出的代价都能现场用 occupancy() 算出来,"
      "不是靠'感觉占用率会变低'这种模糊表述。")
```

**常见坑:** 看到 `recommend_tier` 返回 `registers` 就直接采纳,不去检查这个建议在寄存器的"每线程私有"语义下到底是否成立——01 类知识点 2 已经强调过 `recommend_tier` 是一串简化的 if 阶梯,这里进一步说明"简化"具体简化掉了什么:它只做了字节数和重用次数的算术检查,没有做"数据结构的访问模式是否匹配这一层存储的编程模型"这层检查,这是比"L2 分支不检查 reuse_count"更根本的一类局限。另一个坑是把 SMEM tile size 的选择简化成"占用率最高的那个就是最优解"——01 类知识点 4 追问 2 已经指出高占用率只是"有更多 warp 可以在等内存时切换、掩盖延迟",这里进一步用具体数字证明"1 行/block 占用率最高,但不代表它就是整体吞吐最优的选择",脱离"每次 kernel launch 能摊薄多少访存"这个维度单独谈占用率,是不完整的分析。

---

## 小结:4 个案例对应调研发现的哪些轴线

| 案例 | 规模递增轴 | 工程约束递增轴(并发/分布式) | 方案批判迭代轴 | 决策依据追问轴 | 真实性验证轴 |
|---|---|---|---|---|---|
| 1. Triton autotune 打分平局 | | | | ✅(生产系统怎么改) | ✅ 核心 |
| 2. FlashAttention HBM 节省曲线 | ✅ 核心 | ✅(KV cache 分布式分摊) | | ✅(单卡/多卡怎么选) | |
| 3. Ridge Point 非单调第二证据 | | | | ✅(功耗预算怎么选) | ✅ 核心 |
| 4. Fused MLP 存储层选型 | | | ✅ 核心 | ✅(tile size 怎么选) | |

这 4 个案例不是要把 01-04 的 19 个知识点全部重写——它们演示的是**方法论本身**:拿到任何一个已经验证过的知识点,都可以自己追问"换一组从没试过的具体参数(矩阵形状、精度通道、GPU 型号、tile 切法),结论还成不成立""如果规模再跳一级,原来的解法会在哪个具体的数字上失效""面试官不满意我的第一个方案时,下一步该怎么用具体数字说服他,而不是含糊地说'再优化一下'"。真正的二面深度,是能不能对着一个自己已经很熟悉、觉得"应该没什么好深挖"的知识点,现场把这几条轴线重新走一遍,而不是满足于第一次验证时得到的那个结论。
