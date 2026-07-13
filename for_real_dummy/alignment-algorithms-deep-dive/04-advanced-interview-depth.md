# 04 · 进阶深度追加:4 个真实二面级别的多级追问链案例

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是知识点,不计入统计——它和 [dsa-deep-dive/20-advanced-interview-depth.md](../dsa-deep-dive/20-advanced-interview-depth.md) 是同一挂:方法论 + 案例,不是知识点列表。

## 为什么需要这篇追加内容

`01-03` 全部完成并自查通过之后,用户转达了一位有经验从业者的反馈:现有材料没有达到 **2026 年大厂技术二面** 的深度。`dsa-deep-dive/20-advanced-interview-depth.md` 已经基于一次真实调研(检索大厂面经、面试官视角的元讨论)落地验证过一套格式,核心发现是:真实的追问不是"正确性 → 复杂度 → 能不能优化"这一条线性链,而是沿着至少 5 条独立轴线展开——**规模递增轴**、**工程约束递增轴(并发/分布式)**、**方案批判迭代轴**、**决策依据追问轴**、**真实性验证轴**。

这条 `alignment-algorithms-deep-dive` 系列的知识点结构和 `dsa-deep-dive` 不同(没有数据结构/算法题那种天然的"单机→并发→分布式"或"暴力→优化"的台阶),但同样存在能被这几条轴线深挖的地方——而且系列里已经有一处**验证过程本身就是"真实性验证轴"的完美案例**:[03 类](03-rainbowpo-and-capstone.md)撰写时没有预设 `rainbowpo.py::unified_po_loss` 的"一个函数统一 7 个变体"这个宣称是真的,而是逐个变体重新算了一遍,发现只有 `dpo` 精确匹配。这不是刻意设计出来的教学案例,是撰写系列时真实撞见、又真实验证过的发现,本篇把它作为核心案例展开成完整的追问链。

本篇选了 4 个案例,分别对应 5 条轴线里和这个系列知识点关联最紧的 4 条(**真实性验证轴**、**方案批判迭代轴**、**决策依据追问轴**、**规模递增轴**);**工程约束递增轴(并发/分布式)**在这个系列里没有天然对应的知识点(偏好优化算法本身的 loss 计算不涉及并发数据结构,分布式训练细节属于另一个专题),这里不生搬硬套。每个案例都明确标注建立在哪个已有知识点之上,包含完整还原的多级追问链和至少一段真实验证过的可运行例子,且验证输入(种子、batch size、具体数字)全部和 01-03 类已经用过的不同——**这是方法论范例,不是把 15 个知识点全部重写**,读者应该能把同样的思路自己套用到任何一个已有知识点上练习追问。

---

## 案例 1:RainbowPO"一个函数统一 7 个变体"的真实性验证——从"配置表对得上"到"数字对不对"(核心案例,真实性验证轴)

建立在 [03 类](03-rainbowpo-and-capstone.md)知识点 1、2 之上——那两个知识点已经发现"数值统一只对 `dpo` 一个配置精确成立",本案例用**全新的随机种子、全新的 batch size、全新的手工反例数字**(和 03 类已经验证过的输入完全不同)独立重新跑一遍这个结论,证明它不是"凑巧对上了那一组特定数字",同时把这个验证过程本身包装成一条完整的追问链——这正是调研发现的"把抽象表述压向具体数字"这个模式,在代码审查场景下的具体应用:"这个库/代码声称统一了 N 个变体"和"简历上写'做了性能优化'"是同一类需要被追问戳穿的抽象宣称。

**追问链条完整还原:**

- **面试官:** "这个仓库的 `learning/dpo-family/README.md` 和 `rainbowpo.py` 自己的 `__main__` 输出都说,`unified_po_loss` 这一个函数 + `POConfig` 这一个 dataclass,统一了 6-7 个偏好优化变体。你怎么验证这句话是不是真的?"
- **候选人(第一反应,容易偷懒的回答):** "看 `POConfig` 的字段和每个变体对应的超参是不是对得上——如果 `VARIANTS` 字典里 `orpo` 那一行的 `use_ref=False`、`add_sft=True` 这些和独立实现的设定一致,那这个'统一'应该就成立。"
- **面试官追问(把"配置表对得上"和"数值对得上"这两件事拆开):** "配置表对得上,只能说明这个函数的'输入接口'设计得像那么回事。你有没有拿同一组输入,分别喂给 `unified_po_loss` 和各个变体自己独立的 `*_loss` 函数,把输出的数字摆在一起比过?"
- 候选人如实说没有,现场去验证,依次得到下面几层发现:
- **追问 1(先看最基础的):** "`dpo` 配置,数值对得上吗?" —— 期望:对得上,而且是全仓库唯一一处有 pytest 断言(`test_rainbowpo_dpo_matches`)锁定的配置。
- **追问 2(全系列最有戏剧性的变体):** "`dpop` 呢?这是专门修'chosen 概率下降'这个反例的方法。" —— 期望候选人现场发现:`VARIANTS["dpop"]` 的 4 个字段和 `VARIANTS["dpo"]` 逐字段相同(仅 `name` 不同),所以喂进 `unified_po_loss` 算出来的数字和 `dpo` 一模一样;而真实 `dpop_loss()` 因为 hinge 项会被大幅拉高。
- **深挖追问(逼问根因,不满足于"数值不一致"这个表面结论):** "这是不是意味着这份代码有 bug?" —— 期望候选人精确定位到:不是哪一行算错了,是 `unified_po_loss` 的 4 个开关(`use_ref`/`length_norm`/`loss_type`/`add_sft`)本身的表达能力覆盖不到"hinge 惩罚项"这个机制——`_pref_loss` 里虽然定义了 `"hinge"` 分支,但 `VARIANTS` 字典里没有任何一个配置真正选用它,是死代码。这是"抽象本身设计能力不够",不是"实现细节写错了"。
- **追问 3(比 dpop 更危险的一种):** "`kto` 呢?" —— 期望进一步指出:`kto` 比 `dpop` 更危险——`dpop` 至少还在用真实的、语义正确的 `log_p_c`/`log_p_r` 配对输入(只是没触发 hinge 机制,数值上"看起来正常");`kto` 的 4 个字段虽然也和 `dpo` 一样,但真实 `kto_loss` 要求的输入是单条 `log_p_actor`/`log_p_ref` + `is_desirable` 标签,不是配对的 chosen/rejected——把配对数据硬套进 `unified_po_loss` 的 `kto` 配置,不报错、能算出一个数字,但这个数字不对应任何有意义的计算。
- **追问 4(换一种偏差模式):** "`ipo` 呢,是不是也是同样的'配置没接对'问题?" —— 期望候选人发现这是**不同性质**的偏差:数值和真实实现之间存在一个精确的代数关系(相差 `β²` 倍),不是"跑错了算法",是"骨架接对了,但一个具体的代换细节不一致"——这一类偏差能通过代数推导定位,比 `dpop`/`kto` 那种"看起来没错但压根没跑该跑的逻辑"更容易发现,也更容易修。
- **追问 5(第三种偏差模式):** "`orpo` 和 `cpo` 呢?" —— 期望发现这是更严重的结构性偏差:`orpo` 缺了 `log_odds` 非线性变换(直接对原始 log-prob 做 sigmoid);`cpo` 的加权对象搞反了(`unified_po_loss` 固定是"偏好项权重 1、SFT 项权重 `λ_sft`",真实 `cpo_loss` 是"SFT 项权重 1、对比项权重 `λ_c`")。量级都对不上,不能靠一个简单的缩放系数对齐。
- **追问 6(最接近的一个):** "`simpo` 呢?" —— 期望发现这是 5 个非 `dpo` 配置里偏差最小的:只差一个具名的目标间隔常数 `γ`(`POConfig` 根本没有 `γ` 字段),length-normalization 的核心机制本身完全正确。
- **深挖追问(收束,考察有没有分类能力而不是罗列现象):** "把刚才发现的几种情况分个类,按'危险程度'排个序,而不是逐个念一遍。" —— 期望候选人总结出一个框架:第一类(`dpop`/`kto`,最危险)不报错、数值不是 `nan`/`inf`,但实际在执行完全不同的计算或接收了语义不兼容的输入,只能靠独立重算对比才能发现;第二类(`ipo`)数值有偏差但是精确的代数关系,少量推导就能定位和修复;第三类(`orpo`/`cpo`)结构性偏差,不能用缩放系数对齐,需要逐项重新推导;第四类(`simpo`)几乎正确,只漏一个具名参数。
- **追问(设计题,方法论落地):** "如果你是维护者,怎么防止这类问题?" —— 期望说出"给 `VARIANTS` 里每一个具名配置都补一条 `test_rainbowpo_X_matches`,而不是只对'看起来最重要'的那个配置写测试"。

**可运行例子(1/2):全新种子(123)+ 全新 batch size(6)+ 全新 log-prob 偏移量,重新验证一遍全部 7 个配置——证明"只有 dpo 精确成立"不是凑巧对上了 03 类那一组特定数字:**

```python
import sys
sys.path.insert(0, "learning/dpo-family/src")
import torch
from rainbowpo import VARIANTS, unified_po_loss
from dpo_minimal import dpo_loss
from ipo_minimal import ipo_loss
from orpo_minimal import orpo_loss
from simpo_minimal import simpo_loss
from cpo_minimal import cpo_loss

torch.manual_seed(123)
B = 6
c_a = torch.randn(B) * 0.15 - 0.8
c_r = torch.randn(B) * 0.15 - 1.3
r_a = torch.randn(B) * 0.15 - 1.6
r_r = torch.randn(B) * 0.15 - 0.9
mask6 = torch.ones(B, 10)
sft = torch.tensor(1.8)

results = {name: unified_po_loss(c_a, c_r, r_a, r_r, mask6, mask6, sft, VARIANTS[name])["total"].item()
           for name in ["dpo", "ipo", "orpo", "simpo", "cpo", "kto", "dpop"]}
real_dpo = dpo_loss(c_a, c_r, r_a, r_r, beta=0.1).item()

assert abs(results["dpo"] - real_dpo) < 1e-6
assert results["dpop"] == results["dpo"]
assert results["kto"] == results["dpo"]

real_ipo = ipo_loss(c_a, c_r, r_a, r_r, beta=0.1).item()
assert abs(results["ipo"] - (0.1 ** 2) * real_ipo) < 1e-4

real_cpo = cpo_loss(c_a, r_a, sft, beta=0.1, lambda_c=0.5).item()
assert abs(results["cpo"] - 2 * real_cpo) < 1e-4
real_orpo = orpo_loss(c_a, c_r, sft, lambda_or=0.1).item()
assert results["orpo"] / real_orpo > 5

T = 10
log_p_c_tok = (c_a / T).unsqueeze(1).expand(B, T).contiguous()
log_p_r_tok = (r_a / T).unsqueeze(1).expand(B, T).contiguous()
real_simpo_g0 = simpo_loss(log_p_c_tok, mask6, log_p_r_tok, mask6, beta=2.5, gamma=0.0).item()
assert results["simpo"] == real_simpo_g0

print(f"dpo={results['dpo']:.6f} (real dpo_loss={real_dpo:.6f}, exact match)")
print(f"dpop={results['dpop']:.6f} (bit-identical to dpo, silently degenerates)")
print(f"kto={results['kto']:.6f} (bit-identical to dpo, semantically incompatible but no error)")
print(f"ipo={results['ipo']:.6f} (= beta^2 * real ipo_loss = {(0.1**2)*real_ipo:.6f})")
print(f"cpo={results['cpo']:.6f} (= 2 * real cpo_loss = {2*real_cpo:.6f}, exact 2x)")
print(f"orpo={results['orpo']:.6f} (real orpo_loss={real_orpo:.6f}, off by {results['orpo']/real_orpo:.2f}x, wrong order of magnitude)")
print(f"simpo={results['simpo']:.6f} (= gamma=0 real simpo_loss = {real_simpo_g0:.6f}, exact match)")
```

实测(`.venv` 真跑,种子 123,和 03 类种子 0 完全不同):`dpo=0.637296`(精确匹配)、`dpop=0.637296`(和 dpo 位级相同)、`kto=0.637296`(和 dpo 位级相同)、`ipo=0.148342`(= `β²×` 真实 `ipo_loss`)、`cpo=4.257438`(= 2× 真实 `cpo_loss`,精确 2 倍)、`orpo=18.395504`(真实值 `1.838567`,相差 10.01 倍)、`simpo=0.606414`(= `γ=0` 时的真实 `simpo_loss`,精确匹配)——和 03 类用种子 0 得到的偏差模式(哪个配置精确匹配、哪个静默退化、哪个差一个精确倍数、哪个差一个数量级)完全一致,证明这些不是某一组特定输入下的巧合。

**可运行例子(2/2):`cpo` 的"精确 2 倍"用第二组完全无关的输入重新证明一次(代数恒等式,不依赖具体数字);`dpop` 静默退化用一组全新的手工反例数字(chosen 0.60→0.40,rejected 0.55→0.12,和 03 类的 0.5→0.3/0.4→0.1 不同)再验证一次:**

```python
import sys
sys.path.insert(0, "learning/dpo-family/src")
import torch
from rainbowpo import VARIANTS, unified_po_loss
from cpo_minimal import cpo_loss
from dpop_minimal import dpop_loss
from dpo_minimal import dpo_loss

torch.manual_seed(999)
c_a2 = torch.randn(3) - 2.0
r_a2 = torch.randn(3) - 3.0
c_r2 = torch.randn(3) - 2.5
r_r2 = torch.randn(3) - 1.0
mask3 = torch.ones(3, 5)
sft2 = torch.tensor(5.3)
out_cpo2 = unified_po_loss(c_a2, c_r2, r_a2, r_r2, mask3, mask3, sft2, VARIANTS["cpo"])["total"]
real_cpo2 = cpo_loss(c_a2, r_a2, sft2, beta=0.1, lambda_c=0.5)
assert abs(out_cpo2.item() - 2 * real_cpo2.item()) < 1e-4

log_p_c_actor = torch.log(torch.tensor([0.40]))
log_p_c_ref   = torch.log(torch.tensor([0.60]))
log_p_r_actor = torch.log(torch.tensor([0.12]))
log_p_r_ref   = torch.log(torch.tensor([0.55]))
mask1 = torch.ones(1, 6)
sft1 = torch.tensor(0.9)

out_dpop = unified_po_loss(log_p_c_actor, log_p_c_ref, log_p_r_actor, log_p_r_ref, mask1, mask1, sft1, VARIANTS["dpop"])["total"]
out_dpo1 = unified_po_loss(log_p_c_actor, log_p_c_ref, log_p_r_actor, log_p_r_ref, mask1, mask1, sft1, VARIANTS["dpo"])["total"]
L_dpop_real = dpop_loss(log_p_c_actor, log_p_c_ref, log_p_r_actor, log_p_r_ref, beta=0.1, lambda_p=50.0)
L_dpo_real = dpo_loss(log_p_c_actor, log_p_c_ref, log_p_r_actor, log_p_r_ref, beta=0.1)
log2 = torch.log(torch.tensor(2.0)).item()

assert out_dpop.item() == out_dpo1.item()
assert L_dpo_real.item() < log2
assert L_dpop_real.item() > 20 * L_dpo_real.item()

print(f"cpo second check: unified={out_cpo2.item():.6f}, 2*real_cpo={2*real_cpo2.item():.6f} (exact 2x identity holds again)")
print(f"dpop new counterexample: unified_dpop={out_dpop.item():.6f} == unified_dpo={out_dpo1.item():.6f}")
print(f"real dpop_loss={L_dpop_real.item():.6f} vs real dpo_loss={L_dpo_real.item():.6f} (log2={log2:.6f})")
```

实测:`cpo` 用种子 999 的另一组输入,`unified=11.262208`,`2×real_cpo=11.262208`,精确 2 倍恒等式再次成立(这不是巧合——代数上 `unified_cpo = L_contrast + 2·sft`,`real_cpo = sft + 0.5·L_contrast`,`2×real_cpo = 2·sft + L_contrast`,和 `unified_cpo` 是同一个表达式,对任意输入恒成立)。`dpop` 新反例:`unified_dpop=0.638858` 精确等于 `unified_dpo=0.638858`;真实 `dpop_loss=20.912113` 远大于真实 `dpo_loss=0.638858`(后者 `<log2=0.693147`,说明 DPO 独自会把这个"chosen 概率下降"的场景误判成进步)。

**常见坑:** 把"接口统一"(配置字段和超参表对得上)和"数值统一"(算出来的 loss 数字和独立实现一致)当成同一件事——这是本案例要戳穿的核心误区。另一个坑是发现"能跑通、没有报错、数值也不是 `nan`"就默认实现是对的——`kto` 配置正是反例:能跑,数值有限且"看起来正常",但语义完全是错的,这种错误只能靠对照独立实现重新算一遍才能发现。

---

## 案例 2:DPO→IPO→KTO→ORPO/SimPO/CPO→DPOP 方案批判迭代链——从"会不会训练过头"到"同一个骨架上打补丁"(方案批判迭代轴)

建立在 [02 类](02-po-variant-family.md)全部 7 个知识点之上。这条链和案例 1 的追问模式本质不同:面试官不是在一个方案上不断深挖,而是**针对每个方案指出一个具体的、可验证的缺陷,逼你换下一个方案**——直到最后一步,会发现"换方案"这条路径其实在某一步分岔成了两条独立的问题("要不要 ref"和"chosen 概率会不会绝对下降"),DPOP 不是第 6 个候选方案,而是回到 DPO 骨架上打的一个正交补丁,这本身也是一个值得追问的转折点。

**追问链条完整还原(方案批判迭代,不是深挖同一方案):**

- **面试官给约束:** "用 DPO 在一批 chosen/rejected 偏好数据上训练,训练后期你担心过拟合——margin 涨得非常快,但你不确定这是不是真的在学到东西,还是在无止境地放大某几个样本的优势。"
- **候选人方案 1(现状,DPO):** "DPO 本身应该没问题,loss 会随训练下降。"
- **面试官指出具体缺陷(不是"不够好"这种空话):** "DPO 的 loss 对 margin `h` 求导,`dL/dh = -β·sigmoid(-β·h)`——这个导数对任意有限的 `h` 恒为负,只是绝对值随 `h` 增大按指数衰减、趋近 0 但永远不等于 0。这意味着不管 margin 已经涨到多大,优化器收到的信号永远是'还要更大'。你怎么防止这件事一直发生?"
- **候选人方案 2(换成 IPO):** "换成 IPO——loss 是 `(h - 1/(2β))²`,在 `h = target` 处梯度精确为 0,过了 target 梯度变号,还会被推回来。"
- **面试官指出 IPO 没解决的问题(换维度追问,不是继续深挖同一个"训练稳不稳"的问题):** "IPO 解决了'会不会训练过头'的问题。但现在换个场景:我的数据是从产品日志里爬的,每条只有'用户对这条回复点了赞/点了踩'的单边标签,没有'同一个 prompt 下哪个回复更好'这种成对比较,IPO 能用吗?"
- **候选人方案 3(换成 KTO):** "IPO 和 DPO 一样都要求配对数据,结构上用不了。换成 KTO——它只要求 `(response, 好/坏标签)` 独立标注,不需要配对。"
- **面试官指出 KTO 没解决的问题(继续换维度,不是重复"配对"这个已经解决的问题):** "KTO 帮你砍掉了配对要求。但如果我现在训练资源紧张,显存只塞得下一份模型权重,KTO 能帮我吗?"
- **候选人方案 4(诚实排除 KTO,换成 ORPO/SimPO/CPO 之一):** "不能——`kto_loss` 的签名里明明白白有 `log_p_ref` 参数,KTO 省的是'配对',不是 ref 模型,这是两件独立的事。真正不需要 ref 的是 ORPO、SimPO、CPO 三个,但它们各自砍 ref 的方式不同:ORPO 用 odds ratio 替代相对偏移量、留了 SFT 项锚定;SimPO 直接用长度归一化后的 log-prob 做 reward;CPO 最朴素,直接用 actor 自己的 log-prob 差,同样靠 SFT 项防止两边概率一起被压低的退化解。"
- **面试官指出这三者都没解决、且和前面几步正交的问题:** "这三个都不需要 ref 了。但你在方案 1 里提过 DPO 有一种'相对 margin 掩盖 chosen 绝对概率下降'的现象——去掉 ref 之后,这个问题是被顺便解决了,还是完全是另一回事?"
- **候选人现场推理:** "是另一回事。'要不要 ref'改变的是 reward 怎么算(相对 ref 偏移量,还是 odds ratio,还是直接用 actor 自己的 log-prob);'chosen 绝对概率会不会下降'的根源是 loss 只看两条回答的**相对**margin,不直接约束任何一条回答的**绝对**概率方向——这个结构性问题在有没有 ref 的情况下都可能出现,是正交的两个维度。"
- **深挖追问(逼问'为什么不是换第 5 个方案,而是回到 DPO 打补丁'):** "那 DPOP 为什么不是在 ORPO/SimPO/CPO 的基础上继续改,而是回到 DPO 骨架上加一个 hinge 项?" —— 期望候选人指出:DPOP 的 hinge 项 `λ_p·relu(log_p_c_ref - log_p_c_actor)` 需要一个"ref 给 chosen 的绝对概率"作为基准——这个基准只有在"有 ref"的骨架(DPO/IPO/DPOP 自己)上才存在;ORPO/SimPO/CPO 已经主动把 ref 去掉了,没有这个基准可用,同一个 hinge 公式没法直接照搬过去。
- **深挖追问 2(检验候选人是否真的理解"正交"这个词,而不是只会背):** "这个 hinge 补丁能叠加在 IPO 上吗?" —— 期望候选人推出:能,因为 IPO 的签名里同样有 `log_p_c_ref`/`log_p_c_actor`,hinge 公式需要的基准都在;这也说明 DPOP 的修复思路本质上是"对任何保留了 ref 的骨架都通用的一个附加约束",不是 DPO 专属。

**可运行例子(1/2):不是复述"DPO 梯度不为 0"这句话,是真的做一个 800 步梯度下降模拟,让 margin 自由演化,观察 DPO 的 margin 会不会停、IPO 的 margin 会不会真的稳定在 target:**

```python
import sys
sys.path.insert(0, "learning/dpo-family/src")
import torch
from dpo_minimal import dpo_loss
from ipo_minimal import ipo_loss

beta = 0.1
target = 1.0 / (2 * beta)
assert target == 5.0

def simulate(loss_fn, steps, lr):
    h = torch.tensor([0.0], requires_grad=True)
    zero = torch.tensor([0.0])
    history = [h.item()]
    for _ in range(steps):
        L = loss_fn(h, zero, zero, zero, beta)
        L.backward()
        with torch.no_grad():
            h -= lr * h.grad
        h.grad.zero_()
        history.append(h.item())
    return history

# 同一个学习率(0.05,和03类capstone脚本同一量级),分别对DPO/IPO的margin做800步梯度下降优化
hist_dpo = simulate(dpo_loss, steps=800, lr=0.05)
hist_ipo = simulate(ipo_loss, steps=800, lr=0.05)

checkpoints = [10, 50, 100, 300, 500, 800]
dpo_track = [round(hist_dpo[i], 4) for i in checkpoints]
ipo_track = [round(hist_ipo[i], 4) for i in checkpoints]

# DPO: h应该持续增长,直到第800步依然在涨(最后200步的位移明显不为0)
assert all(dpo_track[i] < dpo_track[i + 1] for i in range(len(dpo_track) - 1))
dpo_last_200_move = hist_dpo[800] - hist_dpo[600]
assert dpo_last_200_move > 0.1

# IPO: h应该收敛到target=5.0并且停止移动(最后200步位移精确为0)
ipo_last_200_move = hist_ipo[800] - hist_ipo[600]
assert abs(hist_ipo[800] - target) < 1e-4
assert ipo_last_200_move == 0.0

print("checkpoints:", checkpoints)
print("DPO h:", dpo_track, " (still climbing, last-200-step move =", round(dpo_last_200_move, 4), ")")
print("IPO h:", ipo_track, " (converged to target, last-200-step move =", ipo_last_200_move, ")")
```

实测:`checkpoints=[10, 50, 100, 300, 500, 800]`,DPO 的 `h` 轨迹 `[0.025, 0.1246, 0.2485, 0.7362, 1.2118, 1.9035]`——单调递增,第 600→800 步依然移动了 `0.4582`,没有任何停下来的迹象;IPO 的 `h` 轨迹 `[3.2566, 4.9742, 4.9999, 5.0, 5.0, 5.0]`——第 100 步左右已经基本到达 `target=5.0`,第 600→800 步位移精确为 `0.0`,是真正意义上的"停止更新",不是"更新幅度变小到打印精度以下看不出来"。

**可运行例子(2/2):验证 DPOP 的 hinge 补丁确实能原样叠加在 IPO 上,而结构上没法叠加在 CPO 上(不是靠嘴说,靠函数签名验证):**

```python
import sys
sys.path.insert(0, "learning/dpo-family/src")
import torch
import torch.nn.functional as F
import inspect
from ipo_minimal import ipo_loss
from cpo_minimal import cpo_loss

def ipo_loss_with_dpop_patch(log_p_c_actor, log_p_c_ref, log_p_r_actor, log_p_r_ref, beta=0.1, lambda_p=50.0):
    """把dpop_minimal.py同款hinge直接叠加在ipo_loss上——两者都有log_p_c_ref这个基准,能对得上。"""
    L_ipo = ipo_loss(log_p_c_actor, log_p_c_ref, log_p_r_actor, log_p_r_ref, beta=beta)
    hinge = F.relu(log_p_c_ref - log_p_c_actor).mean()
    return L_ipo + lambda_p * hinge

# 换一组新的chosen概率下降场景(和案例1、和dpop_minimal.py自带反例都不同的数字)
log_p_c_actor = torch.log(torch.tensor([0.35]))
log_p_c_ref = torch.log(torch.tensor([0.50]))
log_p_r_actor = torch.log(torch.tensor([0.08]))
log_p_r_ref = torch.log(torch.tensor([0.30]))

L_ipo_alone = ipo_loss(log_p_c_actor, log_p_c_ref, log_p_r_actor, log_p_r_ref, beta=0.1)
L_ipo_patched = ipo_loss_with_dpop_patch(log_p_c_actor, log_p_c_ref, log_p_r_actor, log_p_r_ref, beta=0.1, lambda_p=50.0)
assert L_ipo_patched.item() > L_ipo_alone.item()

# cpo_loss结构上没有ref参数,同一个hinge公式在cpo上没有"基准"可用——签名里验证清楚
params = list(inspect.signature(cpo_loss).parameters.keys())
assert not any("ref" in p for p in params)

print(f"L_ipo_alone={L_ipo_alone.item():.6f}")
print(f"L_ipo_patched(+dpop hinge)={L_ipo_patched.item():.6f}")
print(f"cpo_loss params={params} -> no *_ref field, hinge has nothing to compare against")
```

实测:`L_ipo_alone=16.280569`,叠加 hinge 补丁之后 `L_ipo_patched=34.114315`(hinge 项非负,叠加之后 loss 只会更大,这里因为构造的场景里 chosen 概率确实下降,hinge 严格为正,实测两者相差超过 1 倍);`cpo_loss` 的参数列表 `['log_p_chosen_actor', 'log_p_rejected_actor', 'sft_loss_chosen', 'beta', 'lambda_c']` 里没有任何 `*_ref` 字段,验证了"同一个 hinge 公式在 CPO 上无法直接使用"不是猜测,是结构性事实。

**常见坑:** 把这条链理解成"后面的方法总是比前面的更好"——不对,每一步换方案都是**针对一个具体缺陷**做的权衡,不是单调的"升级"。比如 IPO 换掉 DPO 解决了"训练过头"的问题,但不解决"能不能用单边数据"的问题;ORPO/SimPO/CPO 解决了"要不要 ref"的问题,但不天然解决"chosen 绝对概率会不会下降"的问题——面试官追问的正是这些"没解决的维度",而不是让你反复论证同一个方案"不够好"。另一个坑是把"配对"和"要不要 ref"这两个独立的设计维度混为一谈(呼应 02 类知识点 7 的核心结论),在这条批判链里,如果分不清"KTO 解决的是配对问题、不是显存问题",整条链的追问 3→4 会直接答错方向。

---

## 案例 3:给定约束选算法——决策依据追问(决策依据追问轴)

建立在 [02 类知识点 7](02-po-variant-family.md) 的 8 算法横向对比表之上。这条追问轴不纠错,只逼问"给定一个具体约束,你会选哪个,为什么不选别的"——和案例 2 的区别在于:案例 2 是"方案被否定,换下一个方案"的连续过程;本案例是**同一张已知事实表格,在不同约束下反复查询**,包括诚实地推出"这几种方法都不满足"这种否定性结论的能力。

**追问链条完整还原:**

- **Q:** "显存只够放一份模型权重(没法同时常驻 actor + ref),你会从这 7 种(不含 DPO 本身共 8 种)算法里怎么选?" —— 期望:先排除 DPO/IPO/KTO/DPOP(都需要 ref,来自 02 类知识点 7 的表格),剩 ORPO/SimPO/CPO 三个候选。
- **追问 1(不满足于"三选一都行"这个答案):** "这三个都满足'不需要 ref'这条约束,你怎么进一步区分选哪个?" —— 期望候选人说清楚这张表本身只能回答"能不能训练"这条及格线,进一步区分需要引入表外的信息:数据里 chosen 是否天然比 rejected 长很多(如果是,SimPO 的长度归一化更对症,能直接从损失函数层面消掉这个偏置);现有 SFT checkpoint 质量是否已经过关(ORPO/CPO 都靠 SFT 项锚定,如果 SFT 本身没训好,两者都会继承这个问题);实现和调试复杂度预算(CPO 最简单——没有 odds 变换也没有 mask 归一化;ORPO 要处理 `log_odds` 的数值稳定性;SimPO 要处理长度 mask)。
- **追问 2(换一个约束维度):** "现在换个场景:数据是从客服系统爬来的,每条记录只有'这条回复被投诉了/没被投诉'的独立标签,没有成对比较,你选哪个?" —— 期望:KTO,即使 KTO 仍然需要 ref(追问约束的核心不是"越少约束越好",是"哪条约束在这个场景里是硬性的、必须满足")。
- **追问 3(两个约束同时出现,考验诚实推理而不是硬凑答案):** "如果同时有'数据不成对'和'显存只够一份模型'这两个约束呢?" —— 期望候选人现场推理出:这 7 种方法里,不存在一个"既不需要配对、又不需要 ref"的——ref-free 三兄弟(ORPO/SimPO/CPO)全部需要配对,唯一不需要配对的 KTO 需要 ref;表格本身的"配对"列和"ref"列的对勾组合里,不存在"两者都不需要"这一行。候选人应该能诚实地说"这个约束组合在这几种方法里无解",而不是被追问逼着硬凑一个不成立的答案。合理的下一步方向(开放问题,没有标准答案):跳出这个算法家族本身,比如用一个更强的模型给单边数据自动生成一个"更差"的对照回答、人工构造出 pair;或者接受用 KTO,但想办法降低 ref 的显存成本(比如 LoRA,只让极少数 adapter 参数在 actor/ref 之间不同,大部分权重共享)。
- **深挖追问(要求把推理转成可执行的验证,而不是停留在口头表格):** "你怎么用代码而不是背题的方式验证'不存在同时满足这两个约束的方法'这句话?" —— 期望现场写一个小校验:把"ref-free 集合"和"不需要配对集合"分别求出来,验证两个集合的交集是空集。

**可运行例子(1/2):把上面的推理过程写成一个程序化的决策函数,不是手填一个表格——包括"无解"这个否定性结论也是程序验证出来的,不是拍脑袋断言的:**

```python
import sys
sys.path.insert(0, "learning/dpo-family/src")
import inspect
from dpo_minimal import dpo_loss
from ipo_minimal import ipo_loss
from kto_minimal import kto_loss
from orpo_minimal import orpo_loss
from simpo_minimal import simpo_loss
from cpo_minimal import cpo_loss
from dpop_minimal import dpop_loss

fns = {"dpo": dpo_loss, "ipo": ipo_loss, "kto": kto_loss, "orpo": orpo_loss,
       "simpo": simpo_loss, "cpo": cpo_loss, "dpop": dpop_loss}

# 需不需要ref: 现场重新用inspect.signature验证一遍(不是抄02类的结论,是同一条规则在当前代码上重新跑)
needs_ref = {name: any("ref" in p.lower() for p in inspect.signature(fn).parameters)
             for name, fn in fns.items()}
ref_free = {name for name, v in needs_ref.items() if not v}
assert ref_free == {"orpo", "simpo", "cpo"}

# 需不需要配对数据: 02类已经人工核对过一次,这里直接复用同一份事实
needs_paired = {"dpo": True, "ipo": True, "kto": False, "orpo": True,
                 "simpo": True, "cpo": True, "dpop": True}
unpaired_ok = {name for name, v in needs_paired.items() if not v}

def recommend(single_model_copy: bool, unpaired_data: bool) -> set:
    """根据两条独立约束,过滤出结构上可行的算法集合。"""
    candidates = set(fns.keys())
    if single_model_copy:
        candidates &= ref_free
    if unpaired_data:
        candidates &= unpaired_ok
    return candidates

scenario1 = recommend(single_model_copy=True, unpaired_data=False)
scenario2 = recommend(single_model_copy=False, unpaired_data=True)
scenario3 = recommend(single_model_copy=True, unpaired_data=True)

assert scenario1 == {"orpo", "simpo", "cpo"}
assert scenario2 == {"kto"}
assert scenario3 == set()          # 两个约束同时成立时,程序验证"无解"这个结论,不是拍脑袋说的

print("只有显存约束(单份模型权重):", scenario1)
print("只有数据约束(单边标签,无法配对):", scenario2)
print("两个约束同时成立:", scenario3, "(结构上无解,需要跳出这8种方法本身找别的办法)")
```

实测:只有显存约束时可行集合 `{'orpo', 'simpo', 'cpo'}`;只有数据约束时 `{'kto'}`;两个约束同时成立时是空集 `set()`——和追问 3 里期望候选人推出的结论完全一致。

**可运行例子(2/2):把"配对数据"和"结构不兼容不代表会报错"这两个点合在一起——把 KTO 风格的单边标签硬塞进 ORPO(需要配对)会发生什么:**

```python
import sys
sys.path.insert(0, "learning/dpo-family/src")
import torch
from orpo_minimal import orpo_loss, log_odds

# 把KTO风格的单边数据(is_desirable 0/1标签)硬塞进orpo_loss要求的"rejected"位置——
# orpo_loss(log_p_chosen, log_p_rejected, sft_loss, lambda_or) 3个位置参数,数量对得上,不会报TypeError
torch.manual_seed(5)
log_p_actor = torch.randn(6) - 1.0
is_desirable = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
sft = torch.tensor(1.0)

L = orpo_loss(log_p_actor, is_desirable, sft, lambda_or=0.1)   # 不报错

# 追查为什么"不报错也不是nan": log_odds对>=-1e-6的输入一律clamp到同一个边界值,
# 0.0和1.0都被clamp到-1e-6, 意味着"is_desirable到底是0还是1"这个标签信息被clamp吃掉了,
# 对最终loss没有任何区分度——这就是"没报错但语义错误"的具体机制,不是笼统地说"结构不兼容"
o0 = log_odds(torch.tensor(0.0))
o1 = log_odds(torch.tensor(1.0))
assert o0.item() == o1.item()

print(f"orpo_loss(误用KTO风格输入) = {L.item():.6f}  (没有抛异常)")
print(f"log_odds(0.0) = {o0.item():.6f}, log_odds(1.0) = {o1.item():.6f}  (被clamp到同一个值,标签信息丢失)")
```

实测:`orpo_loss` 用 KTO 风格的 0/1 标签当 rejected 侧输入,不报错,算出 `L=2.225729`;深挖发现根因——`log_odds` 对所有 `≥-1e-6` 的输入统一 clamp 到 `-1e-6` 这个边界,`log_odds(0.0)=13.802318` 和 `log_odds(1.0)=13.802318` 精确相等,意味着这次误用里"标签是 0 还是 1"这个信息在计算过程中被 clamp 悄悄抹掉了,不会体现在最终 loss 里。

**常见坑:** 把决策依据追问答成"哪个方法更好"这种没有约束条件的价值判断——这类问题正确的回答方式永远是先复述清楚约束是什么,再说明约束怎么筛掉了哪些选项,筛不掉的部分才需要经验性判断(且要诚实标注这部分没法只靠读代码得出结论)。另一个坑是遇到"两个约束都要满足但无解"这种情况,不敢说"无解",努力从 7 个选项里勉强凑一个"矬子里拔大个"的答案——诚实推出"当前这个方法家族给不出解",和面试官讨论下一步怎么办,比硬凑一个不成立的选择更能体现决策能力。

---

## 案例 4(可选):真实训练 vs 纯数值 demo——规模递增会依次踩中哪些新问题(规模递增轴)

建立在 [01 类知识点 4](01-dpo-foundations.md)之上——`dpo_minimal.py` 是全系列 9 个源文件里唯一一个真实加载 GPT-2、真实跑 forward+backward 的脚本,其余 8 个都是不加载模型的纯张量数值 demo。这条轴线在这个系列里的特殊之处是:仓库本身只验证到"两条样本、CPU、GPT-2-small"这个规模,再往上跳(更大 batch、更大模型、多卡分布式)不是这个仓库能真实验证的,但知识点 4 已经指出的两处工程简化(**没有真正的 batching**、**response mask 是全 1**)刚好是"规模一旦增大就会从'能忍'变成'必须修'"的两个具体切入点——本案例把这两点分别量化到可验证的程度,再讨论再往上跳会遇到什么。

**追问链条完整还原:**

- **Q:** "`dpo_minimal.py` 里 `for ex in ds:` 这个训练循环,每次只处理一条样本,这在 200 条数据、GPT-2-small 上问题大吗?" —— 期望:不大,200 条数据本来就是"分钟级 smoke test"的规模,单条循环的额外开销相对训练本身的时间可以接受。
- **追问 1(把'不大'这个判断推向具体规模):** "如果这批数据不是 200 条,是 200 万条,或者模型不是 GPT-2-small(124M 参数)是一个 7B 模型,'单条 Python 循环'这件事还是无关紧要的细节吗?" —— 期望候选人推出:数据量增大,循环本身的 Python 解释器开销(不是模型计算本身)会线性累积,而且完全没有利用现代 GPU 在处理批量矩阵乘法时的并行优势——单条前向传播没有把 GPU 计算单元用满;模型变大到 7B,单份权重可能就要占满一张卡的大部分显存,batch size=1 这种"一条数据一次前向"的模式会让每一步的显存带宽利用率变得很差,训练吞吐量远低于同样硬件在合理 batch size 下能达到的水平。
- **追问 2(具体量化,不满足于"应该会变慢"这种定性描述):** "你能给出一个具体的数字,证明'循环 vs 批处理'在真实模型上确实有差距吗?" —— 期望候选人现场测量(见下面可运行例子 1/2),而不是停留在"理论上应该更快"。
- **追问 3(转向另一处工程简化):** "知识点 4 提到 `resp_mask_c` 是全 1,把整段 prompt+response 文本都算进了 log-prob 求和。这个简化在 200 条数据、单轮对话上问题大吗?" —— 期望候选人先意识到需要具体分析,而不是直接下结论:对 `margin = log_ratio_chosen - log_ratio_rejected` 这种做减法的 loss(DPO/IPO/DPOP),如果 chosen 和 rejected 共享完全相同的 prompt 前缀,prompt 部分对 actor 和 ref 的 log-prob 贡献在减法里理论上会互相抵消一部分(需要 actor 的 forward 在两次调用里对同一段 prompt 给出完全相同的输出,这本身也依赖 dropout 等随机因素是否被关闭,这里不展开验证);但对 DPOP 的 hinge 项(只看 chosen 一侧、不做 chosen-rejected 减法)以及所有对"某一条回答的绝对质量"敏感的场景,prompt 部分不会抵消,会实打实地稀释 response 部分的信号。
- **深挖追问(把'稀释多少'量化,并连接到规模递增):** "多轮对话(prompt 更长)会让这个稀释问题变得更明显还是更不明显?" —— 期望候选人推出:轮次越多,最后一句真正需要评估的回复占全部文本 token 数的比例越小,'全 1 mask 把大段无关的历史对话也算进 loss'这个问题只会被放大,不会自动缓解;这正是"规模"(这里的规模指对话轮数/上下文长度而不是数据条数)增加时,一个在小规模上可以忽略的简化,会变成必须优先修复的问题的具体例子。
- **深挖追问(收束到"这个仓库验证不了什么"这个诚实边界):** "如果真要把这个训练脚本推到 7B 模型、多卡场景,还会遇到什么在当前 CPU/单卡/200 条数据规模下完全看不到的新问题?" —— 没有标准答案,期望候选人能提到:单卡放不下 actor+ref 两份 7B 权重,需要考虑用 DeepSpeed ZeRO/FSDP 做参数分片,或者用 LoRA 只让 actor 的少量 adapter 参数可训练、ref 直接复用基座权重不用真的复制一份;多卡训练下 ref 模型的前向是纯推理,是否需要放到和 actor 不同的设备上做流水线重叠;数据量到百万级别之后,`load_dataset(...); ds = list(ds)`(`dpo_minimal.py` 现在的写法,一次性把整个数据集读进内存)会不会本身就成为瓶颈,需要换成流式/分片读取——**这些具体问题这个仓库的规模验证不到,诚实说"验证不到"比编一个没测过的具体数字更重要。**

**可运行例子(1/2):真实加载 GPT-2(和 `dpo_minimal.py` 用的同一个 `--model gpt2` 默认值),真实测量"逐条 forward"vs"一次 batched forward"的耗时差,不是理论推测:**

```python
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2", local_files_only=True)
model.eval()

N = 24
texts = [f"Human: question number {i} about something. Assistant: a short plausible answer that varies a little in length {i * 3}."
         for i in range(N)]

def loop_forward():
    """dpo_minimal.py 的真实模式: for ex in ds: tokenize_one(ex) -> 单条forward"""
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt")
            _ = model(**enc)

def batched_forward():
    """把N条样本一次性pad到同一个batch里,只做一次forward"""
    enc = tok(texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        _ = model(**enc)

def best_of(fn, trials=3):
    best = None
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best

loop_t = best_of(loop_forward, trials=3)
batch_t = best_of(batched_forward, trials=3)

assert loop_t > batch_t * 1.5   # 真实GPT-2上,逐条循环应该明显慢于一次batched forward

print(f"N={N}条样本, 真实gpt2 (CPU), best-of-3:")
print(f"  dpo_minimal.py现在的模式(逐条forward): {loop_t:.4f}s")
print(f"  一次batched forward:                  {batch_t:.4f}s")
print(f"  speedup = {loop_t / batch_t:.2f}x")
```

实测(仓库根目录 `.venv`,CPU,真实 `gpt2` 124M 参数权重,和 `dpo_minimal.py --model gpt2` 加载的是同一份本地缓存):`N=24` 条样本,`best-of-3` 取最优,逐条 forward 耗时 `2.6967s`,一次 batched forward 耗时 `1.1554s`,`speedup=2.33x`。这只是 CPU、24 条、GPT-2-small 上的量级,不能直接外推到"7B 模型、GPU、百万级数据"下具体是多少倍——但方向上的结论(不批处理浪费掉了可以被并行利用的计算资源)是这次真实测量能确定支持的。

**可运行例子(2/2):用真实 `Anthropic/hh-rlhf` 数据 + 真实 GPT-2 tokenizer,量化"全 1 mask 问题"随对话轮数增加而恶化(只统计聚合的 token 计数,不摘录数据集原文——hh-rlhf 是红队测试数据集,内容包含大量攻击性/敏感文本,不适合直接引用进教学文档):**

```python
from transformers import AutoTokenizer
from datasets import load_dataset
import statistics

tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
ds = load_dataset("Anthropic/hh-rlhf", split="train[:200]")

def response_fraction(text):
    """dpo_minimal.py 的 resp_mask_c 是全1, 把prompt+response全部算进log-prob;
    这里量化"真正应该被mask出来的response"占全部token的比例, 只统计token计数, 不引用对话原文。"""
    last_marker = text.rfind("\n\nAssistant:")
    if last_marker == -1:
        return None
    prompt_part = text[:last_marker + len("\n\nAssistant:")]
    n_total = len(tok(text)["input_ids"])
    n_prompt = len(tok(prompt_part)["input_ids"])
    if n_total == 0:
        return None
    n_turns = text.count("\n\nHuman:")
    return n_turns, (n_total - n_prompt) / n_total

single_turn_fracs, multi_turn_fracs = [], []
for ex in ds:
    stats = response_fraction(ex["chosen"])
    if stats is None:
        continue
    n_turns, frac = stats
    (single_turn_fracs if n_turns <= 1 else multi_turn_fracs).append(frac)

mean_single = statistics.mean(single_turn_fracs)
mean_multi = statistics.mean(multi_turn_fracs)

assert mean_single > mean_multi   # 轮数越多, 最后一句真实回复占全文token的比例应该越小
assert mean_single > 2 * mean_multi  # 真实数据上这个差距应该是数量级相关的明显差距, 不是噪声

print(f"单轮对话样本 n={len(single_turn_fracs)}, response token占比均值 = {mean_single:.4f}")
print(f"多轮对话样本 n={len(multi_turn_fracs)}, response token占比均值 = {mean_multi:.4f}")
print("轮数越多, 全1 mask把越大比例的历史对话log-prob当成'response'算进loss, 问题被放大, 不是缩小")
```

实测:`train[:200]` 里单轮对话样本 53 条,response token 占比均值 `0.5220`;多轮对话样本 147 条,均值只有 `0.2126`——多轮场景下平均只有约 21% 的 token 是真正需要评估的最后一句回复,其余近 79% 是被全 1 mask 一起算进 log-prob 求和的历史对话。这个测量依赖网络/HF Hub 访问 `Anthropic/hh-rlhf`(和 01 类知识点 4 依赖同一个数据源),如果本地无法访问,可以把这段当成"已经验证过一次的既有事实"来读,不强制要求重新跑一次网络请求。

**常见坑:** 把"这个仓库只验证到 CPU/单卡/百级数据"当成"这些工程问题不重要"的理由——知识点 4 已经如实标注这些是"minimal 教学脚本"和"生产训练脚本"之间的差距,本案例进一步说明这些差距不是恒定的,会随着规模(数据量、上下文长度、模型大小)增长而变得更严重,不能假设"现在能忍就永远能忍"。另一个坑是被问到"7B、多卡会遇到什么问题"时,编一个听起来专业但没有依据的具体数字(比如"大概会慢 10 倍")——诚实的做法是说清楚方向性的结论(不批处理浪费并行资源、mask 问题随规模恶化)和具体数字的适用范围(这里只测到 CPU/24 条/GPT-2-small),不把没验证过的外推说成是测过的。

---

## 小结:4 个案例对应调研发现的哪些轴线

| 案例 | 规模递增轴 | 方案批判迭代轴 | 决策依据追问轴 | 真实性验证轴 |
|---|---|---|---|---|
| 1. RainbowPO 统一声称验证 | | | | ✅ 核心 |
| 2. DPO→…→DPOP 批判迭代链 | | ✅ 核心 | | |
| 3. 给定约束选算法 | | | ✅ 核心 | |
| 4. 真实训练 vs 数值 demo | ✅ 核心 | | | |

工程约束递增轴(并发/分布式)在这个系列里没有出现——不是遗漏,是这个系列的知识点本身(偏好优化 loss 的数学/代码实现)不天然涉及并发数据结构或分布式一致性问题,案例 4 讨论到"7B、多卡"时点到了分布式训练的边缘(ZeRO/FSDP、多卡流水线),但如实标注为"这个仓库验证不到"的开放讨论,不强行包装成一条完整验证过的追问链。

这 4 个案例不是要把 15 个知识点全部重写——它们演示的是**方法论本身**:拿到任何一个已经掌握的知识点,都可以自己追问"这个'统一/简化/优化'的宣称,我有没有真的拿数字核对过""如果换一个约束条件,现在的最优解还是最优解吗""这个方案被淘汰,是因为它本身不行,还是因为它解决的是另一个维度的问题""如果规模再往上跳一个数量级,现在看不见的问题会不会变得看得见"。真正的二面深度,是能不能对着一个自己没准备过的知识点,现场把这几条轴线走一遍——而且,像案例 1 那样,愿意先怀疑一个听起来很有说服力的宣称("一个函数统一了 N 个变体"),再动手验证。
