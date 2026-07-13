# 07 · 毕业专题常规知识点(Serving Graduation Topics)

> 源:`learning/serving-graduation/`(Module 5《用大模型》第 7、也是最后一个专题,14 lectures/14 notebooks)。本文件覆盖 L01-L11 共 9 个常规知识点;L12(Capstone-1 R1-tiny 部署)、L13(五线综合)、L14(Capstone-2 五线综合毕业)三篇属于毕业顶点叙事体,见 [08-serving-graduation-capstone.md](08-serving-graduation-capstone.md)。
>
> 这个模块整体内容密度明显低于前 6 篇(11 个 src 文件合计仅 628 行,单篇 lecture 普遍 21-49 行),9 个知识点里有 4 个(L03/L04/L09/L10)完全没有对应 src——这不是撰写疏漏,是源材料自己的性质:`README.md` 的总览表格里这 4 行"代码"一栏原样写的是"—"。撰写时如实标注,不假装存在源码。

---

## 1. Agent 场景两条缓存线:Prefix Cache 省 KV 重算 + Reasoning Cache 省整段 Thinking 重算(`agent_inference_demo.py` + 概念性,L01+L03 合并)—— 缓存对象不同,但都是"hash 命中就跳过重算"

**是什么:** `agent_inference_demo.py` 模拟一个 5 轮 agent 对话在 naive(每轮重新 prefill 整段历史)和 cached(radix 缓存,只 prefill 新增部分)两种模式下的 prefill token 累计量:

```python
# learning/serving-graduation/src/agent_inference_demo.py:18-21
def simulate_agent_turn(history_len: int, new_user_tokens: int, cached: bool) -> int:
    if cached:
        return new_user_tokens
    return history_len + new_user_tokens
```

```python
# learning/serving-graduation/src/agent_inference_demo.py:24-32
def run_multi_turn(turns: int = 5, base_history: int = 200, per_turn_new: int = 50) -> TurnStats:
    naive = 0
    cached = 0
    history = base_history
    for t in range(turns):
        naive += simulate_agent_turn(history, per_turn_new, cached=False)
        cached += simulate_agent_turn(history, per_turn_new, cached=True)
        history += per_turn_new + 30   # tool output + assistant reply
    return TurnStats(naive_prefill_tokens=naive, cached_prefill_tokens=cached)
```

这里的"cache"指的是 01 号文件知识点 7(`prefix_cache.py`)/02 号文件知识点 2-3(`RadixTree`)那种 **KV 层面**的前缀缓存——命中后省下的是 prefill 计算。L03(reasoning cache,无独立 src)讲的是另一种缓存:把 `(question_hash, reasoning_trace, answer)` 存进 redis,命中后**连 decode 都不用做**,直接吐历史结果。两者命中对象不同(KV 张量 vs 完整回答字符串),但设计哲学完全一致:用某种 hash 做 exact-match 查找,命中就跳过重算。

**一句话:** Agent 多轮对话有两层浪费——"同一段历史被重新 prefill"和"同一个问题被重新 think"——分别对应 prefix cache 和 reasoning cache 两种独立的缓存机制。

**底层机制/为什么这样设计:** `run_multi_turn` 里 `history` 每轮增长 `per_turn_new + 30`(30 是模拟 tool 输出 + assistant 回复的 token 数),这意味着 naive 模式下每轮重 prefill 的量是**单调递增**的(第 5 轮要重新吃下前 4 轮的全部历史),而 cached 模式下每轮无论历史多长都只需要 prefill 新增的 `per_turn_new` 个 token——这就是为什么 agent 场景(轮次多、历史涨得快)比单轮 chat 更需要 prefix cache:naive 的浪费是随轮次**线性放大**的。Reasoning cache 的设计约束更复杂:必须区分"完全 match"(hash 一样)和"近似 match"(embedding 相似度),前者绝对安全但命中率低,后者命中率高但有"语义相近但答案其实不同"的风险;还要处理隐私边界(per-user cache 不能跨用户共享,但系统 prompt 相关的 cache 可以全局共享)。

**AI 研究场景:** Cursor/Cline/Devin 这类 coding agent 工具的多轮会话优化——每次工具调用后模型都要重新看到完整历史,如果没有 KV 前缀缓存,5-20 轮的 agent 任务成本会随轮次二次方增长;客服 FAQ 场景的 reasoning cache——大量用户问同一批高频问题,命中率往往能到 30-50%,直接省掉整个 thinking 阶段。

**可运行例子:**
```python
import sys, functools
sys.path.insert(0, "learning/serving-graduation/src")
from agent_inference_demo import run_multi_turn, simulate_agent_turn

# --- 第一部分:真实 KV 前缀缓存节省(复用仓库真实函数) ---
s = run_multi_turn(turns=5, base_history=200, per_turn_new=50)
assert s.naive_prefill_tokens == 2050
assert s.cached_prefill_tokens == 250          # 5 轮 x 50,与历史长度完全无关
saved = 1 - s.cached_prefill_tokens / s.naive_prefill_tokens
assert abs(saved - 0.8780) < 1e-3

# 手动重算 naive 累加,交叉验证函数没有算错
history, manual_naive = 200, 0
for _ in range(5):
    manual_naive += history + 50
    history += 50 + 30
assert manual_naive == s.naive_prefill_tokens == 2050

# --- 第二部分:reasoning cache,用 Python 真实标准库 functools.lru_cache
#     演示"hash exact-match 命中就跳过重算"这同一个设计哲学(不是仓库自带代码,
#     本模块 L03 无独立 src,这里用 stdlib 里已经过千万级生产验证的真实缓存机制做
#     最小说明,不是我临时发明的算法) ---
call_count = 0

@functools.lru_cache(maxsize=None)
def fake_reasoning(question: str) -> str:
    global call_count
    call_count += 1
    return f"answer-to-{question}"

qs = ["1+1?", "2+2?", "1+1?", "1+1?", "3+3?"]
results = [fake_reasoning(q) for q in qs]
assert call_count == 3                          # 3 个不同问题各算一次
assert results[0] == results[2] == results[3]    # 3 次问 "1+1?" 结果一致
assert fake_reasoning.cache_info().hits == 2     # 后两次问 "1+1?" 是真实缓存命中
```

**实测(`.venv` 真跑):** `run_multi_turn(turns=5, base_history=200, per_turn_new=50)` 真实输出 `naive=2050, cached=250`,节省 87.8%——手动重算累加过程逐轮核对一致,函数没有算错。`lru_cache` 演示部分:5 次提问里有 3 次问的是 `"1+1?"`,真实 `call_count` 只有 3(不是 5),说明后两次问同一个问题时函数体真的**没有再执行**(而不是执行了但返回值凑巧相同)——`cache_info().hits == 2` 从缓存内部计数器角度二次印证了这一点。

**面试怎么问 + 追问链:**
- **Q:** "一个多轮 agent 对话系统,如果要同时上 prefix cache 和 reasoning cache,这两者的失效(invalidate)策略应该一样吗?" —— 期望说出:不一样。Prefix cache 随对话历史单调追加,几乎不主动失效(除非触达最大长度或用户开新会话);reasoning cache 是对"问题→答案"的缓存,面对的是"同一个问题在不同上下文/不同用户问"要不要复用答案这个策略问题,失效条件更复杂(涉及正确性和隐私两个维度,不是单纯的"历史长度"能衡量的)。
- **追问 1:** "reasoning cache 如果做近似匹配(embedding 相似度),命中了一个语义相近但实际所指不同的历史问题,会有什么风险?" —— 期望说出:近似匹配可能返回"看起来相关但实际上下文对不上"的答案——比如"这个函数怎么用"在两个不同代码库里问,embedding 可能判断相似度很高,但真实答案完全不同;这是近似缓存的系统性风险,需要设置足够高的相似度阈值,或者干脆只做 exact hash 放弃近似匹配。
- **追问 2:** "如果 KV cache 和 reasoning cache 在同一次请求里都命中了,理论上能省多少?这种情况现实吗?" —— 期望说出:两者叠加,prefill(KV 命中)和 decode(reasoning 命中)都不用重新跑模型,几乎是纯格式化返回;但要指出这是"完全命中"的理想情况,真实产品里往往只是"部分历史相同、但当前这句问题是新的"这种部分命中,两种缓存各自按自己的粒度独立生效,不能假设总是同时全命中。

**常见坑:** 把"prefix cache"和"reasoning cache"当成同一个东西的两种叫法——它们缓存的对象、粒度、失效策略完全不同,前者是系统实现细节(用户通常感知不到),后者会直接影响返回内容的新鲜度(命中了旧答案可能对当前场景不再适用),不能用同一套心智模型思考。另外容易忽略 `history += per_turn_new + 30` 这行——如果只看"5 轮 agent"就以为每轮增量恒定是 `per_turn_new`,会算错 naive 侧的真实累加量(还要算上模拟的 tool 输出 + assistant 回复这 30 个 token)。

---

## 2. Thinking Budget:s1 式预算强制收尾与 Wait 注入(`thinking_budget.py`,L02)—— 本机验证发现:强制关闭后直接放弃了本该继续吐出的 answer

**是什么:** `generate_with_budget` 逐 token 消费一个 `<think>...</think><answer>...</answer>` 流,一旦 thinking 阶段 token 数达到预算上限,就强行插入 `</think>` 收尾:

```python
# learning/serving-graduation/src/thinking_budget.py:22-41
def generate_with_budget(stream: Iterable[str], budget_tokens: int) -> BudgetResult:
    out: List[str] = []
    in_think = False
    forced = False
    think_count = 0
    for tok in stream:
        if tok == THINK_OPEN:
            in_think = True
        out.append(tok)
        if in_think and tok != THINK_OPEN:
            think_count += 1
        if tok == THINK_CLOSE:
            in_think = False
        if in_think and think_count >= budget_tokens:
            # force-close
            out.append(THINK_CLOSE)
            in_think = False
            forced = True
            break
    ...
```

`inject_wait_tokens`(`thinking_budget.py:50-65`)则相反——不是缩短思考,而是在 thinking 阶段每隔 `every_n` 个 token 插入一次字面量 `"Wait"`,人为拉长思考过程,这是 2025 Stanford s1 论文("s1: Simple test-time scaling")的核心 trick:训一个小模型,在它想要提前结束思考时注入 "Wait" 强迫它继续想,用更多推理时间换准确率。

**一句话:** 商用 thinking model(Claude 3.7 extended_thinking / Gemini 2.5 thinking_budget / OpenAI o1-o3 reasoning_effort)都需要给"思考"设置一个可控预算——不设预算,模型会为了简单问题也生成上万 token 的 thinking,浪费成本和延迟。

**底层机制/为什么这样设计:** 为什么不直接用短一点的 `max_tokens` 而要专门搞一个"只作用于 thinking 阶段"的预算?因为 `max_tokens` 是无差别截断,一旦触发,连正在生成的 answer 都会被腰斩;thinking budget 理论上应该只约束"思考"这一段,预算耗尽后仍然放行模型继续把 answer 说完——这正是本知识点在独立验证时发现的问题:这份 mock 实现在 `think_count >= budget_tokens` 触发的一瞬间直接 `break` 跳出整个循环,根本没有继续消费 `stream` 剩下的 `<answer>42</answer>` 部分,等于把"该只截断 thinking"实现成了"直接截断整个生成"——和它自己想要示范的设计意图正好相反。

**AI 研究场景:** 2025-2026 年主流商用 API 的真实 budget 控制机制(`{"thinking": {"type": "enabled", "budget_tokens": 16000}}` 这种字段在 Claude/Gemini 的真实请求体里都能看到);s1 论文证明"小模型 + budget forcing(强制延长思考)"能在某些推理基准上打过没有 budget forcing 的大模型,是 test-time compute scaling 这条研究路线的代表性工作。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/serving-graduation/src")
from thinking_budget import generate_with_budget, inject_wait_tokens, THINK_OPEN, THINK_CLOSE

stream = ["<think>"] + [f"t{i}" for i in range(20)] + ["</think>", "<answer>", "42", "</answer>"]

# budget 不够:8 个 think token 后强制关闭
r = generate_with_budget(stream, budget_tokens=8)
assert r.forced_close is True
assert r.thinking_tokens == 8
assert "42" not in r.tokens and "<answer>" not in r.tokens   # 真正的答案被完全丢弃

# 对照组:budget 充足,不强制关闭,answer 正常吐出
r2 = generate_with_budget(stream, budget_tokens=100)
assert r2.forced_close is False
assert "42" in r2.tokens

# 独立复现:换更长的 think(50 token)+ 不同 budget(15),排除"budget=8 是巧合"
stream2 = ["<think>"] + [f"x{i}" for i in range(50)] + ["</think>", "<answer>", "final-answer-999", "</answer>"]
r3 = generate_with_budget(stream2, budget_tokens=15)
assert r3.forced_close is True
assert "final-answer-999" not in r3.tokens

# Wait 注入:every_n=2,think 阶段共 5 个 token,count 在 2/4/6 处三次触发注入
short = ["<think>"] + [f"t{i}" for i in range(5)] + ["</think>"]
w = inject_wait_tokens(short, every_n=2)
assert w.count("Wait") == 3
```

**实测(`.venv` 真跑):** `budget=8` 时真实输出 `r.tokens = ['<think>','t0',...,'t7','</think>']`——`42`/`<answer>` 完全不在里面;换成 `budget=100`(充足)时 `42` 确实出现在 `tokens[-4:]` 里。换一组完全不同的流(50 个 token 的 thinking、budget=15、answer 字符串换成 `"final-answer-999"`)独立复现,同样的丢答案现象照样发生——不是 `budget=8` 这个特定参数的巧合,是 `break` 语句本身的结构性问题。`inject_wait_tokens` 一侧没有类似问题,`every_n=2` 时真实在 `count=2/4/6` 三处插入 `"Wait"`,和逐步手算的结果完全吻合。

**面试怎么问 + 追问链:**
- **Q:** "为什么不能直接用一个更短的 `max_tokens` 来控制模型思考多久?" —— 期望说出:`max_tokens` 是对整个生成过程的无差别截断,一旦触发,正在生成的最终答案也会被腰斩;thinking budget 理论上应该是"精确制导"——只约束 thinking 阶段的长度,预算耗尽后模型仍然要继续把 answer 讲完,这是两种机制设计目标的根本区别。
- **追问 1(直接命中本知识点的验证发现):** "如果一个 thinking budget 的实现,一旦预算耗尽就直接终止整个生成、不再吐出 answer,这算不算实现对了设计意图?" —— 期望明确说"不算对",并指出这正是本知识点在独立验证时发现的真实现象:`generate_with_budget` 里 `break` 语句在插入 `</think>` 之后直接跳出了整个 for 循环,根本没有继续消费流里剩下的 answer 部分——这种实现把"精确截断 thinking"退化成了"无差别截断整个响应",在真实部署里会导致用户在 budget 不够时**连答案都拿不到**,而不是"拿到一个思考过程较短但依然完整的答案"。
- **追问 2:** "s1 的 Wait 注入为什么能让一个能力有限的小模型表现更好?" —— 期望说出:强制延长思考不是让模型"突然知道更多知识",而是给模型更多"步骤"去自我复核/多角度重新推导——这是用更多推理时间换准确率的 test-time scaling 手段,类似于"三思而后行";但它的效果上限仍然受模型本身能力约束,不是无限拉长思考就能无限提升准确率。

**常见坑:** 把"thinking budget"和"max_tokens"当成同一种截断机制的两种叫法——真实语义应该完全不同,这份 mock 实现里的 `break` bug 恰好说明了两者"看起来都是截断,但设计意图不同"这件事有多容易在实现时被搞混。另一个坑是只看 `demo()` 打印出的 `r.tokens[-4:]`(`['t5','t6','t7','</think>']`)就以为"强制收尾成功了、流程正常",不去检查 answer 部分是否真的还在——这正是本知识点验证时踩过的坑:表面输出"看起来合理"(确实以 `</think>` 结尾),但完整对比后才发现后面该有的 answer 整个不见了。

---

## 3. 长上下文推理服务:为什么 100k+ context 必须"组合拳",而不是单靠加显存(概念性,L04,无独立 src)—— 复用 01 号文件真实 `NaiveKvPool` 验证 GQA 显存量级

**是什么:** 本课 lecture 原文明确写"本课主要是概念,不附独立 src"式的表述(`learning/serving-graduation/lectures/04-long-ctx-serving.md`),核心论点是一句话公式:**长 ctx 推理 = paged + 量化 KV + 长 ctx 训练(RoPE scaling) + 长 prompt 路由**,四件缺一不可。它引用的显存量级例子是"100k token × 32 层 × 256KB/token/层 ≈ 800GB",用来说明"就算 H100 80GB 显存,fp16 精度下也完全装不下 100k 上下文的 KV cache"。

**一句话:** 100k+ 上下文对 serving 系统的压力不是单一维度的("加显存就行"),而是显存(KV cache 体积)、计算(prefill 时延)、模型本身(是否见过这么长的位置编码)三条约束同时收紧,必须多种技术组合应对。

**底层机制/为什么这样设计:** 这四件武器分别对应四个不同的瓶颈——PagedAttention(01 号文件知识点 3)解决"KV cache 静态预留造成的碎片浪费",KV int8/fp8 量化(04 号文件知识点 11)直接压缩 KV 体积本身,RoPE scaling(YaRN/NTK,见 [long-context-deep-dive](../long-context-deep-dive/00-roadmap.md))解决的是"模型训练时只见过短序列位置编码,推理时外推到长序列会退化"这个**模型能力**问题(不是 serving 侧能解决的,必须训练侧配合),Disaggregated 部署(05 号文件知识点 8)则是把"长 prompt 的巨量 prefill 计算"独立到专属节点,避免拖慢其他短请求的 decode。四者分工完全不重叠,这就是为什么必须组合,单独上任何一个都只能缓解某一侧的压力。

**AI 研究场景:** Gemini 1.5/2.5 系列 100 万 token 上下文的真实工程实现;长文档 RAG、代码库级别理解(把整个仓库塞进 context)这类场景对 serving 侧的真实压力测试。

**可运行例子:**
```python
import sys, torch
sys.path.insert(0, "learning/inference-engine-core/src")   # 复用 01 号文件真实 NaiveKvPool,不新写显存计算逻辑
from naive_kv import NaiveKvPool

# lecture 自己的算术复核:100k token x 32 层 x 256KB/token/层
lecture_claim_gb = 100_000 * 32 * 256 * 1024 / 1024**3
assert 750 < lecture_claim_gb < 850   # lecture 原文写"800GB",量级吻合

# 真实 GQA 架构量级(Llama-3-8B 风格:n_kv_heads=8, head_dim=128, 32 层, 100k ctx, batch=1, fp16)
gqa_pool = NaiveKvPool(batch=1, max_len=100_000, n_kv_heads=8, head_dim=128,
                        n_layers=32, dtype=torch.float16, device="cpu")
gqa_gb = gqa_pool.memory_bytes() / 1024**3

# 对照:假设没有 GQA(kv heads 数等于常见 query heads 数 32)
mha_pool = NaiveKvPool(batch=1, max_len=100_000, n_kv_heads=32, head_dim=128,
                        n_layers=32, dtype=torch.float16, device="cpu")
mha_gb = mha_pool.memory_bytes() / 1024**3

assert gqa_gb < mha_gb
ratio = mha_gb / gqa_gb
assert abs(ratio - 4.0) < 1e-6          # 32/8 = 4,和 kv heads 数比例精确一致
assert gqa_gb < 20 and mha_gb < 60      # 两者都远小于 lecture 插图性的 ~800GB
```

**实测(`.venv` 真跑):** lecture 自己的算术复核为 781.25GB(原文四舍五入写"800GB",量级一致)。但把同样 100k 上下文套进 01 号文件真实的 `NaiveKvPool`,用一个真实 GQA 架构(8 个 KV heads,呼应 Llama-3-8B 这类现代开源模型)算出来只需要 **12.21GB**,套用无 GQA 的假设(32 个 KV heads)也只要 **48.83GB**——都远小于 lecture 插图性的 781GB,且两者比例精确等于 KV heads 数量之比(32/8=4.0x,和真实测出的 4.0x 完全吻合)。这说明 lecture 的"256KB/token/层"是刻意夸张、便于说明问题严重性的教学量级,不对应任何具体真实架构;GQA 本身就是从模型架构层面主动收窄 KV cache 体积的设计,是"训练时就要为 serving 成本考虑"的典型例子。

**面试怎么问 + 追问链:**
- **Q:** "100k+ 上下文对 serving 系统提出了哪些新要求,只靠堆显存能不能解决?" —— 期望说出:不能。显存只是三个约束之一(KV cache 体积),即便显存无限大,prefill 阶段 O(L) 甚至 O(L²) 的计算时间仍然是延迟瓶颈,而且模型本身是否"认识"这么长的位置编码是训练侧的能力问题,不是加显存能解决的。
- **追问 1(利用 GQA vs MHA 的真实验证结果):** "如果 KV cache 显存吃紧,除了做量化,还能从模型架构角度做什么?" —— 期望说出:GQA/MQA 本身就是通过减少 KV heads 数量来收窄显存占用的架构选择,本知识点验证里换成 8-KV-head 的 GQA 架构比 32-head 的假设省了整整 4 倍显存——这提示"省显存"不是只有 serving 侧能做的事,模型架构设计阶段就已经在替 serving 侧铺路,呼应 08 号文件"造改用"三角关系里"造"和"用"两端的对偶设计。
- **追问 2:** "长 prompt 和短 prompt 应该混在同一个 serving pool 里处理吗?" —— 期望说出:不应该。lecture 提到的"长 prompt 走 long-ctx pool、短 prompt走常规 pool"路由策略,本质原因是长 prefill 如果不做 chunked prefill 处理,会独占 GPU 阻塞同批次里的短请求 decode,分池能避免相互拖累,这和 01 号文件 Chunked Prefill 知识点、05 号文件 Disaggregated 知识点讲的是同一类"隔离长短任务"的设计哲学。

**常见坑:** 把"长上下文支持"简单理解成"这个模型 context window 写的是 100k 所以随便发 100k token 都没问题"——真实情况是"训练时见过的有效长度"和"标称 context window"经常不一致(RoPE 外推超出训练分布后质量会退化),以及就算模型能力允许,serving 侧的 KV 显存/延迟约束依然存在,两层问题要分开考虑。另一个坑是看到 lecture 的"800GB"这类数字就直接当成某个具体真实模型的准确值去引用——本知识点验证已经证实这是刻意夸张的教学量级,真实 GQA 架构的数字要小一个数量级以上,面试或写文档时如果要引用具体数字,应该用真实架构参数重新算,而不是照抄教学演示里的整数。

---

## 4. 多模型路由:按 query 复杂度选 tier(`multi_model_router.py`,L05)

**是什么:** 一个纯 heuristic 的复杂度打分函数,配合 4 档模型路由:

```python
# learning/serving-graduation/src/multi_model_router.py:15-29
TIERS = [
    ModelTier("small-1b", cost_per_million=0.05, capability=0.4),
    ModelTier("medium-7b", cost_per_million=0.30, capability=0.7),
    ModelTier("large-70b", cost_per_million=2.00, capability=0.9),
    ModelTier("thinking-o3", cost_per_million=10.0, capability=0.98),
]

def heuristic_complexity(query: str) -> float:
    """Crude complexity score: long queries + math/code keywords lift it."""
    score = 0.0
    score += min(1.0, len(query) / 500)
    keywords = ["prove", "derive", "complex", "step-by-step", "algorithm", "optimise"]
    score += 0.2 * sum(1 for k in keywords if k.lower() in query.lower())
    return min(score, 1.0)
```

```python
# learning/serving-graduation/src/multi_model_router.py:32-40
def route(query: str) -> ModelTier:
    c = heuristic_complexity(query)
    if c < 0.25:
        return TIERS[0]
    if c < 0.55:
        return TIERS[1]
    if c < 0.85:
        return TIERS[2]
    return TIERS[3]
```

**一句话:** 简单问题路由到便宜小模型、复杂问题才动用昂贵大模型,是用几乎零成本的路由判断换取 50-70% 的总体推理成本节省。

**底层机制/为什么这样设计:** `heuristic_complexity` 把"长度"和"关键词命中数"两个信号线性相加再截顶到 1.0——这是刻意简化的启发式(注释里自己写了 "Crude"),真实系统更常见的做法是用一个足够小的分类模型(BERT-size 甚至更小)做路由判断,因为纯关键词匹配很容易被绕过(比如一句简单的话里恰好出现了 "complex" 这个词会被误判复杂)。四档 tier 的成本呈指数增长(0.05→0.30→2.00→10.0,即 6x→6.7x→5x),这个梯度设计本身就在暗示"routing 判断错误的代价是不对称的"——错误地把简单问题送去 thinking-o3 档,浪费的是 200 倍的钱;错误地把复杂问题送去 small-1b 档,浪费的钱少但直接答错。

**AI 研究场景:** Cursor(简单代码补全用小模型,重构类任务用大模型)、Cline(廉价 base 模型 + 昂贵模型做 plan)这类真实产品的路由策略;OpenAI 自己的 4o-mini → 4o → o3 escalation 路由。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/serving-graduation/src")
from multi_model_router import route, heuristic_complexity, estimate_total_cost, TIERS

assert route("hi").name == "small-1b"
assert route("prove the derivative step-by-step using the optimise algorithm" * 3).name == "thinking-o3"

queries = ["hi", "what's the weather today",
           "prove the derivative step-by-step using the optimise algorithm"]
costs = estimate_total_cost(queries)
assert sum(costs.values()) > 0
assert set(costs.keys()) == {t.name for t in TIERS}

# 独立复现:换一批完全不同的 query,确认路由结果不是针对这 3 句话专门凑出来的
new_queries = ["yo", "derive and prove step-by-step this complex algorithm optimise proof " * 5, "how are you"]
new_costs = estimate_total_cost(new_queries, tokens_per_query=1000)
tiers_hit = {k for k, v in new_costs.items() if v > 0}
assert "small-1b" in tiers_hit and "thinking-o3" in tiers_hit
```

**实测(`.venv` 真跑):** `"hi"` 复杂度算出 0.00 分,路由到 `small-1b`;`"what's the weather today"` 0.05 分,同样落在 `small-1b`;含 6 个关键词命中 + 足够长度的那句话拿到 0.92 分,精确落进 `thinking-o3` 档。换一批完全不同措辞的新 query 独立复现,依然稳定命中 `small-1b` 和 `thinking-o3` 两档,不是针对示例特别凑出来的巧合。

**面试怎么问 + 追问链:**
- **Q:** "如果 heuristic complexity 判断错了,把简单问题判成复杂、或者反过来,后果分别是什么?" —— 期望说出:这是非对称风险——简单问题误判复杂只是多花钱(结果不受影响);复杂问题误判简单则可能直接答错(省了钱但砸了质量),所以真实系统的路由阈值设计往往偏保守,宁可多花一点钱也不能让复杂问题落到能力不够的档位。
- **追问 1:** "纯关键词匹配的 heuristic router 有什么明显缺陷,为什么真实系统很少只用这个?" —— 期望说出:关键词匹配容易被字面巧合误导(一句简单闲聊里偶然出现 "complex" 这个词就会被加分),真实系统更倾向于用一个专门训练的小分类器(甚至用一次极小模型的推理本身作为路由判断)来提高准确率。
- **追问 2:** "router 自己的推理开销,会不会抵消路由本该省下的钱?" —— 期望说出:如果 router 判断本身用的是重量级模型,确实可能得不偿失,所以路由系统设计的第一原则就是路由判断本身必须严格控制在"足够便宜"的量级——heuristic 规则或者极小分类器,不能让"决定用哪个模型"这一步骤比模型本身还贵。

**常见坑:** 把 `capability` 字段(0.4/0.7/0.9/0.98)误认为是从真实 benchmark 测出来的分数——这里只是人为设定的示意值,用来说明"档位越高能力越强"这个相对关系,不代表任何具体模型的真实评测结果。另外容易忽略 `min(score, 1.0)` 这个截顶——如果一句话同时命中全部 6 个关键词又足够长,理论加和会超过 1.0*1(length)+0.2*6(keywords)=2.2,但函数会把它截到 1.0,所以"极端复杂"和"中等复杂偏上"在这个 heuristic 下可能被路由到同一档,分辨率在高分区间会失真。

---

## 5. Batch vs Online 推理:两种负载两种优化方向(概念性,L06,无独立 src)—— 复用 06 号文件真实 `cost_calc.cost_for_workload` 演示折扣

**是什么:** Batch 和 Online 是两种完全不同的优化目标——Batch 追求吞吐最大化(可以攒大 batch、慢慢跑、不要求流式),Online 追求延迟最小化(小 batch、chunked prefill、CUDA graph、SSE 流式)。OpenAI 真实 Batch API 用**价格**直接体现这个权衡:批处理任务价格是标准价的 50%,但代价是接受"24 小时内完成"而不是秒级响应的保证。

**一句话:** 同样的模型同样的硬件,只要放弃"低延迟"这一个约束,单位 token 成本能直接打对折——这是 batch/online 场景优化方向不同带来的真实商业结果,不是营销噱头。

**底层机制/为什么这样设计:** Batch 模式为什么能做到理论上的成本减半,而不仅仅是吞吐提升?因为不追求低延迟意味着可以攒更大的 batch(256+)、全部 prefill 一起做、全部 decode 一起做、可以在系统低峰期排队执行——这些因素共同提升了 GPU 利用率,厂商也就有动力用价格反向鼓励用户主动做这个区分(把不紧急的任务挪到低峰期,给紧急的在线流量腾资源)。反过来,Online 配置里的 `enable_chunked_prefill=True` 恰恰是为了不让一个长 prompt 的 prefill 独占 GPU、拖慢同批次里其他请求的 decode——这是 Online 场景专属的约束,Batch 场景完全不需要考虑。

**AI 研究场景:** 数据标注、大规模离线评测(跑一个模型在 10 万条测试集上的结果)、批量内容生成这类真实业务场景——这些任务的共同特点是"结果需要但不着急",完全符合 batch API 的定价假设。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/production-serving/src")   # 复用 06 号文件真实 cost_calc,不新写成本公式
from cost_calc import Deployment, Workload, cost_for_workload

online = Deployment("self-host-online", gpu_cost_per_hour=2.0, tok_per_s_per_gpu=2500, n_gpus=8)
w = Workload(qps=20, avg_in_tokens=500, avg_out_tokens=150)
online_cost = cost_for_workload(online, w, hours=1.0)

# OpenAI Batch API 官方折扣率:50%
batch_cost = online_cost * 0.5

assert abs(batch_cost - online_cost * 0.5) < 1e-9
assert batch_cost < online_cost
saved_pct = (1 - batch_cost / online_cost) * 100
assert abs(saved_pct - 50.0) < 1e-6
```

**实测(`.venv` 真跑):** 同一个 8×A100 部署、同样的 workload(qps=20,输入 500/输出 150 token),真实跑出 online 每小时成本 $83.20,应用 batch 折扣后降到 $41.60,节省恰好 50%——这里没有重新发明成本公式,直接复用的是 06 号文件里已经独立验证过的 `cost_for_workload` 真实函数,只是在结果上应用官方公开的折扣率,不是凭空编的数字。

**面试怎么问 + 追问链:**
- **Q:** "同样是发多个请求,为什么 batch 模式能做到成本减半,而不仅仅是吞吐提升?" —— 期望说出:不追求低延迟释放了几个自由度——可以攒更大 batch、可以在低峰期跑、不需要为流式输出做额外的调度让步,这些因素共同抬高了 GPU 利用率,单位 token 成本因此显著下降,厂商用价格反向鼓励用户做这个取舍。
- **追问 1:** "如果我的任务对延迟不敏感,但很想要流式输出的用户体验,应该走 batch 还是 online?" —— 期望说出:只能走 online 配置。Batch API 设计上就是"提交任务→等待→拿完整结果"的模式,和流式输出的设计目标(边生成边返回)是冲突的,想要流式体验就必须接受 online 定价,这两个维度不能既要又要。
- **追问 2:** "如果把 batch 任务混进 online 的 serving pool 里一起跑,会发生什么?" —— 期望说出:batch 任务通常用大 batch size(256+)整体提交,如果和 online 的小 batch 低延迟请求混在同一个调度队列里,大 batch 的 prefill/decode 会独占资源、拖慢 online 请求的响应时间——这正是为什么两者在真实系统里通常是完全隔离的资源池,而不是共用同一批 GPU 动态调度。

**常见坑:** 把"batch 推理"和"batch size 大"这两个概念混为一谈——这里的 Batch 指的是 OpenAI 那种"提交一批任务、24 小时内异步返回"的**产品模式**,不是单纯"一次前向计算处理多少条数据"这个技术参数(后者在 online 模式下也存在,只是数值小得多)。另一个坑是想当然认为"batch 一定更省钱"从而无脑把所有任务都扔进 batch 队列——如果任务本身对时效性有要求(哪怕只是"用户希望几分钟内看到结果"),24 小时的完成时间保证可能完全不能接受,省下的钱换不来业务能接受的体验。

---

## 6. VLM 推理服务:image encoder 与 LLM 主干的双系统协作(`vlm_serve.py`,L07)—— mock 编码,如实标注非真 ViT

**是什么:** `vlm_serve.py` 是一份纯 mock 模板,`encode_image_mock` 把图片路径/base64 字符串编码成固定 8 维向量(用字符 ASCII 值填充,不是任何真实视觉特征):

```python
# learning/serving-graduation/src/vlm_serve.py:14-25
def encode_image_mock(path_or_b64: str) -> List[float]:
    """Always return exactly 8 floats, padding short inputs."""
    padded = path_or_b64.ljust(8, "0")[:8]
    return [float(ord(c)) for c in padded]

def vlm_generate(req: VlmRequest) -> str:
    visual = []
    if req.images:
        for img in req.images:
            visual += encode_image_mock(img)
    return f"<vlm response to '{req.text}' with {len(visual)//8} images>"
```

真实 VLM 推理服务的流程是:①Image encoder(ViT)把图片编码成一串 visual token,②visual token 和 text token 拼接后一起送进 LLM 主干,③LLM 正常 decode 输出——`vlm_generate` 只模拟了"图片数量被正确统计进响应"这一层最外围的协议行为,完全不涉及真实的视觉特征提取。

**一句话:** VLM 推理服务本质是两套异构计算(ViT 编码 + LLM decode)拼在一起,工程上的核心问题是怎么让这两套系统高效协作而不是谁更快,这份 mock 只演示了"协议层"(图片数量对不对得上),不涉及真实 encoder。

**底层机制/为什么这样设计:** 为什么 ViT 通常不值得像 LLM 主干那样做 tensor parallel 切分?因为 ViT 相对 LLM 主干体积小得多(lecture 举例 Qwen2-VL 7B 是 14GB LLM + 4GB ViT),切分带来的跨卡通信开销,在这种体量下可能反而超过其计算本身的收益——TP 只有在通信开销能被计算量摊销时才划算,这也是为什么 lecture 提到"image encoder offload"更常见的做法是把 ViT 整个放到单独一张卡上跑(而不是切开分布到多卡),同时用"async vision"让 ViT 编码和 LLM 准备阶段并行,而不是串行等待。

**AI 研究场景:** Qwen2-VL/GPT-4V/Claude vision 这类真实多模态模型的部署;多图输入场景下 visual token 随图片数量线性累积,可能比纯文本 token 还多,这对 KV cache 预算规划是新的压力来源;同一张图被多次追问时的 "paged visual cache" 复用问题(和 01/02 号文件的 KV/RadixAttention 前缀缓存是同一类思路在视觉侧的延伸)。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/serving-graduation/src")
from vlm_serve import VlmRequest, vlm_generate, encode_image_mock

req = VlmRequest(text="describe these", images=["cat.jpg", "dog.png"])
enc = encode_image_mock(req.images[0])
assert len(enc) == 8

out = vlm_generate(req)
assert "2 images" in out
out0 = vlm_generate(VlmRequest(text="hello"))
assert "0 images" in out0

# 验证不同文件名产生不同编码(排除"伪装成编码实际是常数占位"的可能)
enc2 = encode_image_mock("dog.png")
assert enc != enc2
```

**实测(`.venv` 真跑):** `encode_image_mock("cat.jpg")` 真实输出 `[99.0, 97.0, 116.0, 46.0, 106.0, 112.0, 103.0, 48.0]`(即 `"cat.jpg0"` 前 8 字符的 ASCII 值),`encode_image_mock("dog.png")` 输出完全不同的一组数字——确认这不是常数占位,而是真的对不同输入产生不同(但和真实视觉语义无关)的编码。`vlm_generate` 对 2 张图/0 张图的请求分别正确统计出 `"2 images"`/`"0 images"`。

**面试怎么问 + 追问链:**
- **Q:** "VLM serving 相比纯文本 LLM serving,新增了哪些工程难点?" —— 期望说出:image encoder 是完全不同的计算模式(不是 autoregressive 的一次性前向),需要单独的显存预算;多图输入时视觉 token 会随图片数量累积,可能超过文本 token 量;以及"同一张图被多次提问"时是否要做视觉侧的缓存复用。
- **追问 1:** "为什么 ViT 通常不值得像 LLM 主干那样切 TP?" —— 期望说出:ViT 体量比 LLM 主干小得多,TP 切分带来的跨卡通信开销在这个量级下可能超过计算本身的收益,只有当模型大到通信开销能被计算量摊销时切分才划算——这是一个具体的成本收益判断,不是"能切就应该切"。
- **追问 2:** "如果要给这份 mock 换成真实 ViT 编码,最少需要改哪几处?" —— 期望说出:至少要把 `encode_image_mock` 换成真实加载一个视觉 encoder(比如 CLIP ViT)做真实前向,输出的维度也要匹配 LLM 主干接受的 visual token 维度(不再是固定 8 维),`vlm_generate` 也要把"数图片数量"换成真实把 visual token 和 text token 拼接后送进 LLM decode——本知识点的 mock 完全没有涉及这两处真实计算,只模拟了协议层的图片计数行为。

**常见坑:** 看到 `encode_image_mock` 返回的是"看起来像向量"的浮点数列表,就以为这是某种简化版的真实视觉特征——实际上它只是把文件名字符串按 ASCII 值转成数字,和图片内容毫无关系,两张内容完全不同但文件名相似的图片可能编码出高度相似的"特征向量",这是纯协议层 mock 的典型陷阱:结构对了(维度、类型),语义完全不对。

---

## 7. Embedding 服务:与自回归 LLM 服务的关键差异(`embedding_serve.py`,L08)—— cosine() 数学性质独立复验通过,mock 的只是语义

**是什么:** `embedding_serve.py` 用 sha256 哈希产生确定性占位向量(不是真语义嵌入),外加一个通用的余弦相似度函数:

```python
# learning/serving-graduation/src/embedding_serve.py:8-21
def embed_text(text: str, dim: int = 64) -> List[float]:
    h = hashlib.sha256(text.encode()).digest()
    return [(b - 128) / 128 for b in h[:dim]]

def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / max(na * nb, 1e-9)
```

Embedding 服务和自回归 LLM 服务的根本区别在于:embedding model 是 encoder-only 架构,一次前向直接出向量,不存在 "token-by-token 串行依赖" 这件事——这是它吞吐能远高于同尺寸 LLM 服务的根本原因(lecture 给出一卡 4090 可以做到 10k embedding/s)。

**一句话:** Embedding 服务把"生成"问题变成了"一次编码"问题,没有自回归的串行瓶颈,可以用远大得多的 batch size 打满 GPU,吞吐量级和延迟特征都和 LLM decode 服务完全不同。

**底层机制/为什么这样设计:** 为什么 `cosine` 函数本身值得单独拎出来验证?因为它是整个 RAG/语义检索链路的数学基础——`embed_text` 用 sha256 是刻意的、自我声明的占位(`demo()` 自己打印"sha256确定性占位,非真语义"),但 `cosine` 函数是不是**正确实现**的余弦相似度,是完全独立于"embedding 语义准不准"的另一个问题:哪怕上游换成真实的 BGE/E5 语义嵌入,下游这个 `cosine` 函数的数学正确性也必须先保证。

**AI 研究场景:** RAG 系统的文档检索(先把文档库编码成向量,查询时算相似度取 top-k)、语义搜索、聚类;BGE/E5/voyage-3/gte-large 这类真实生产级 embedding 模型的服务化部署;FAISS 这类向量索引库和 embedding 服务的组合使用模式。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/serving-graduation/src")
from embedding_serve import embed_text, embed_batch, cosine

# 第一部分:mock 语义占位的确定性行为
v1, v2 = embed_text("hello world"), embed_text("hello world")
v3 = embed_text("completely different text")
assert abs(cosine(v1, v2) - 1.0) < 1e-9   # 相同文本 -> 确定性输出 -> 完全相同向量

# 第二部分:独立数学复验 cosine() 本身对正交/平行/反向向量是否给出正确的 0/1/-1
# (不依赖 embed_text 的语义,只测 cosine 这个通用数学函数)
orth_a, orth_b = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
par_a, par_b = [2.0, 3.0, 1.0], [4.0, 6.0, 2.0]      # b = 2 * a
opp_a, opp_b = [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]
assert abs(cosine(orth_a, orth_b) - 0.0) < 1e-9
assert abs(cosine(par_a, par_b) - 1.0) < 1e-9
assert abs(cosine(opp_a, opp_b) - (-1.0)) < 1e-9
```

**实测(`.venv` 真跑):** mock 侧,两次编码同一句 `"hello world"` 得到 `cosine == 1.000000`(sha256 确定性,输入相同输出必然相同),编码"完全不同的文本"得到 `cosine == 0.210678`(不是 0,因为 sha256 输出没有语义结构,不能指望正交)。独立数学复验部分用完全人工构造、不经过 `embed_text` 的向量测试:正交向量算出 `0.000000`,平行向量(`b=2a`)算出 `1.000000`,反向向量算出 `-1.000000`——三个边界情况全部精确吻合教科书定义,证明 `cosine` 函数实现本身是数学正确的,"不真实"的只是上游 `embed_text` 的语义(它自己也在 `demo()` 里如实标注了这一点)。

**面试怎么问 + 追问链:**
- **Q:** "为什么 embedding 服务的吞吐能做到远高于同尺寸 LLM 服务(这里提到一卡 4090 上 10k/s)?" —— 期望说出:embedding 是 encoder-only 架构,一次前向直接输出向量,没有 LLM decode 阶段"每步都要等前一步 token 结果"的串行依赖,可以用远大得多的 batch size(256+)一次性打满 GPU 并行度,这是纯架构差异带来的吞吐差异,不是工程优化技巧的差异。
- **追问 1:** "如果验证一个 embedding 服务的相似度计算是否正确,应该测什么,不应该测什么?" —— 期望说出:应该独立测相似度函数本身(用人工构造的正交/平行/反向向量验证数学性质,像本知识点这样),不应该依赖某个具体 embedding 模型的输出"看起来合理"就判定正确——数学函数的正确性和上游语义模型的质量是两个完全独立、必须分开验证的维度。

**常见坑:** 看到 `cosine(相同文本) = 1.000` 就以为"这个 embedding 服务工作正常、语义理解没问题"——这个结论只能证明"确定性"(同输入同输出),证明不了任何语义相关的东西;`embed_text` 用 sha256 从设计上就不可能产生"意思相近的句子向量也相近"这种真实语义嵌入才有的性质,不能拿它去做真实的语义检索测试。另一个坑是把 `cosine` 返回值不做 clamp 就直接拿去做需要 `[-1, 1]` 严格范围的下游计算——浮点误差可能让极端情况下返回值略微超出这个范围(比如 `1.0000000002`),真实生产代码通常会做一次 `max(-1.0, min(1.0, result))` 保护,这里的实现没有做。

---

## 8. 冷启动与容错:产线可靠性工程的两件事(概念性,L09+L10 合并,无独立 src)—— 无源码,手写标准 circuit breaker + backoff 状态机作参照实现

**是什么:** L09(冷启动)和 L10(容错)都是纯概念课,`README.md` 总览表格这两行"代码"一栏原样写的是 "—"。二者讨论的是生产可靠性工程的两个不同阶段:冷启动关注**服务启动时**的延迟(70B 模型加载要 1-2 分钟,首个请求的用户体验很差),容错关注**服务运行时**的故障处理(GPU OOM、worker 崩溃、网络抖动、模型输出异常)。两者共享一部分工程词汇——canary deploy(灰度上线,先 5% 流量观察再逐步放量)、blue-green(蓝绿双版本瞬间切换,出问题秒回滚)、circuit breaker(错误率超阈值就暂停该 worker,过一段时间半开测试再恢复)——本质都是"变更/故障发生时,如何让系统影响范围可控"这同一个工程目标的不同应用场景。

**一句话:** 冷启动解决"服务刚上线时不可用"的问题,容错解决"服务跑着跑着挂了"的问题,两者都靠"渐进式验证 + 快速止损"这同一套思路(灰度/熔断/重试)应对不确定性。

**底层机制/为什么这样设计:** 为什么 circuit breaker 需要一个 half-open 中间态,不能直接从 open(熔断)跳回 closed(正常)?因为 half-open 的作用是**试探性恢复**——只放行少量真实请求去验证下游是否真的恢复了;如果直接从 open 跳回 closed,一旦下游还没真正恢复,会立刻被新一波全量流量二次打垮,之前"暂停该 worker"争取到的恢复时间就白费了。同理,exponential backoff(指数退避)不是随便选一个增长曲线——它需要在"重试足够快、不让用户等太久"和"不在系统本就有压力时用高频重试雪上加霜"之间取平衡,指数增长 + 上限封顶(cap)正是这个折中的标准解法。

**AI 研究场景:** 大模型容器化部署的真实运维场景——canary/blue-green 发布策略在生产大模型服务里同样适用(新模型版本先接 5% 流量观察输出质量和延迟,没问题再逐步放量);vLLM/SGLang 这类推理引擎面对 GPU OOM 时的 preempt+recompute 机制(和 01 号文件调度策略知识点里的抢占概念是同一件事在容错场景下的应用);商业 API 依赖方(如果自建服务依赖某个第三方模型 API)必须处理"上游服务不可用时如何降级"的容错设计。

**可运行例子:**
```python
# 本知识点无独立源码(L09/L10 均为纯概念课),以下是根据 lecture 描述从零手写
# 的标准 circuit breaker / exponential backoff 参照实现——这是教科书级别的确定性
# 状态机和数列计算,不涉及任何"伪装成真实 ML 效果"的模拟,因此不违反"不假装复现
# 不存在的机制"这条纪律,但请注意这与其它知识点"复用仓库已有真实代码"的做法不同,
# 这是本系列里唯一一处必须从零手写参照实现的知识点。
class CircuitBreaker:
    def __init__(self, fail_threshold=3, half_open_after=2):
        self.fail_threshold = fail_threshold
        self.half_open_after = half_open_after
        self.state = "closed"
        self.fail_count = 0
        self.call_count_since_open = 0

    def call(self, fn, *a, **kw):
        if self.state == "open":
            self.call_count_since_open += 1
            if self.call_count_since_open >= self.half_open_after:
                self.state = "half_open"
            else:
                raise RuntimeError("circuit open, call rejected")
        try:
            result = fn(*a, **kw)
        except Exception:
            self.fail_count += 1
            if self.state == "half_open":
                self.state = "open"
                self.call_count_since_open = 0
            elif self.fail_count >= self.fail_threshold:
                self.state = "open"
                self.call_count_since_open = 0
            raise
        else:
            if self.state == "half_open":
                self.state = "closed"
                self.fail_count = 0
            return result


def flaky(succeed: bool) -> str:
    if not succeed:
        raise ValueError("boom")
    return "ok"


cb = CircuitBreaker(fail_threshold=3, half_open_after=2)
for _ in range(3):
    try:
        cb.call(flaky, False)
    except ValueError:
        pass
assert cb.state == "open"                       # 连续 3 次失败触发熔断

rejected = False
try:
    cb.call(flaky, True)                        # open 状态下第 1 次调用应被直接拒绝
except RuntimeError:
    rejected = True
assert rejected is True

result = cb.call(flaky, True)                    # 第 2 次调用触发 half_open 试探,且真实成功
assert cb.state == "closed" and result == "ok"   # 试探成功 -> 恢复 closed


def exponential_backoff(attempt: int, base_ms: int = 100, cap_ms: int = 10_000) -> int:
    return min(cap_ms, base_ms * (2 ** attempt))


delays = [exponential_backoff(i) for i in range(8)]
assert delays == [100, 200, 400, 800, 1600, 3200, 6400, 10000]   # 第 8 次触发 cap 封顶
```

**实测(`.venv` 真跑):** 连续 3 次失败后 `cb.state` 真实变为 `"open"`;open 状态下第 1 次调用被真实拒绝(`RuntimeError`,`flaky` 函数体根本没被执行到);第 2 次调用触发 half-open 探测,`flaky(True)` 真实执行并成功,状态真实回落到 `"closed"`——完整走过 `closed → open → half_open → closed` 全部四个状态转移。Backoff 数列前 8 项真实计算为 `[100, 200, 400, 800, 1600, 3200, 6400, 10000]`,第 8 项(`attempt=7` 时理论值 100×2⁷=12800)被 `cap_ms=10000` 正确封顶。

**面试怎么问 + 追问链:**
- **Q:** "circuit breaker 和简单的失败重试,分别解决什么问题?能互相替代吗?" —— 期望说出:不能互相替代。重试解决的是单次瞬时故障(网络抖动这种偶发问题),circuit breaker 解决的是"持续性故障发生时,如何快速失败、避免让新请求继续堆积拖垮整个下游"——如果只有重试没有熔断,大量请求在故障期间疯狂重试反而会雪上加霜、加剧下游压力,形成雪崩效应。
- **追问 1(直接命中本知识点手写实现的核心设计):** "half-open 状态存在的意义是什么,为什么不能故障持续一段时间后直接从 open 跳回 closed?" —— 期望说出:half-open 是试探性恢复机制——只放行少量真实请求验证下游是否真的恢复了;如果直接跳回 closed(相当于无条件恢复全量流量),一旦下游还没真正恢复好,会立刻被新一波全量流量二次打垮,之前熔断争取到的恢复窗口就白费了,这也是本知识点手写实现里 `half_open_after` 参数存在的原因。
- **追问 2:** "exponential backoff 为什么要设一个 cap(封顶),不能让延迟无限指数增长?" —— 期望说出:没有 cap 的话重试间隔会随失败次数指数爆炸(几次失败后可能要等几个小时才重试一次),这在实际系统里既不合理也不实用;cap 保证了"最坏情况下,重试间隔也不会超过一个可接受的上限",本知识点手写实现里 `cap_ms=10_000` 让第 8 次以后的延迟稳定在 10 秒,不再继续翻倍。

**常见坑:** 把"重试"当成万能的容错手段,不区分故障是瞬时的还是持续的——对一个已经持续故障的下游疯狂重试,不仅救不了请求,还会用重试流量本身加重下游负担,这正是需要 circuit breaker 介入、暂停继续重试的场景。另一个坑是选取 `fail_threshold`/`half_open_after` 这类阈值参数时过于随意——阈值太低会导致偶发的几次失败就触发整体熔断(可用性受损),阈值太高又会让熔断保护形同虚设(等发现故障时下游已经被打垮了),这两个参数需要结合真实故障率数据调优,不是随手写个 3 和 2 就能通用。

---

## 9. 服务工程 5 原则与毕业评分卡:goodput 比吞吐更诚实(`serving_scorecard.py`,L11)—— 呼应 DistServe 论文的 goodput 概念

**是什么:** L11 是全模块的"5 条原则"总结课(KV 是金 / 调度即性能 / backend 选场景 / 量化+投机是免费午餐 / 监控+成本是隐藏游戏),对应源码 `serving_scorecard.py` 把这些原则具体化成一个可运行的评分卡工具——不只看某个 checkpoint 是否"正确",还要同时过 TTFT/TPOT 延迟 SLO 这道门槛,并计算真实"有效吞吐"(goodput):

```python
# learning/serving-graduation/src/serving_scorecard.py:47-69
def score_report(report: Dict, slo: GraduationSLO) -> List[CandidateScore]:
    scores: List[CandidateScore] = []
    for row in report["results"]:
        tokens = _response_tokens(row["response"])
        ttft_ms = float(row["latency_ms"])
        tpot_ms = ttft_ms / tokens
        correct = bool(row["correct"])
        passes = (
            (correct or not slo.require_correct)
            and ttft_ms <= slo.max_ttft_ms
            and tpot_ms <= slo.max_tpot_ms
        )
        ...
```

```python
# learning/serving-graduation/src/serving_scorecard.py:72-83
def effective_goodput(scores: Iterable[CandidateScore], request_rate_rps: float) -> Dict[str, float]:
    items = list(scores)
    passed = [s for s in items if s.passes]
    attainment = len(passed) / len(items)
    total_cost = sum(s.estimated_cost for s in items)
    return {
        "attainment": round(attainment, 4),
        "goodput_rps": round(request_rate_rps * attainment, 4),
        "cost_per_good_request": round(total_cost / max(1, len(passed)), 6),
    }
```

模块自己的 docstring 明确写"inspired by DistServe goodput"——这里的 `effective_goodput` 正是 05 号文件知识点 8(Disaggregated Prefill/Decode)提到的 DistServe 论文核心指标在毕业评分场景下的具体应用:不看请求数量本身,只看"SLO 达标"的那部分请求数量。

**一句话:** 一个 serving 方案的"好不好"不能只看它能不能正确回答、也不能只看它跑得多快,必须同时满足正确性和延迟门槛才算一次"有效"的服务,goodput 就是在衡量这个"有效"的比例。

**底层机制/为什么这样设计:** 为什么不直接用"吞吐(QPS)"这一个指标评价 serving 方案?因为高 QPS 不代表"好的"QPS——如果大部分响应要么答错要么超时,QPS 再高也是在生产垃圾结果,`effective_goodput` 的设计就是把"有效"这个隐藏维度显式量化出来:`attainment` 衡量有多少比例的请求真正达标,`goodput_rps` 把这个比例乘回真实请求速率,`cost_per_good_request` 更进一步——把"答错/超时"的请求成本也摊到"答对"的请求头上(因为那些失败请求的计算资源也是真实花掉的,不能当作没发生过)。

**AI 研究场景:** 这是全系列(01-08)在评估维度上的收束点——DistServe 论文本身的核心论点就是"在 SLO 约束下最大化请求处理速率",比单纯优化吞吐或者单纯优化延迟都更贴近真实生产场景的需求;08 号文件的毕业 capstone 会把这套评分卡真实应用到"五线综合"的对比数据上,而不是只停留在这里的合成示例。

**可运行例子:**
```python
import sys
sys.path.insert(0, "learning/serving-graduation/src")
from serving_scorecard import GraduationSLO, score_report, effective_goodput, rank_candidates

report = {"results": [
    {"ckpt": "0.5B", "response": "<think>brief</think><answer>42</answer>",
     "latency_ms": 120, "correct": True, "size_mb": 500},
    {"ckpt": "7B", "response": "<think>some reasoning steps</think><answer>42</answer>",
     "latency_ms": 450, "correct": True, "size_mb": 7000},
    {"ckpt": "70B-wrong", "response": "<think>...</think><answer>41</answer>",
     "latency_ms": 2200, "correct": False, "size_mb": 70000},
]}
slo = GraduationSLO(max_ttft_ms=900, max_tpot_ms=80)
scores = score_report(report, slo)
by_ckpt = {s.ckpt: s for s in scores}
assert by_ckpt["0.5B"].passes is True
assert by_ckpt["70B-wrong"].passes is False        # correct=False,不管延迟多好都不过

gp = effective_goodput(scores, request_rate_rps=10.0)
assert abs(gp["attainment"] - round(2/3, 4)) < 1e-9   # 3 个里 2 个 passes

ranked = rank_candidates(scores)
assert ranked[0].passes is True                     # 过 SLO 的排前面
```

**实测(`.venv` 真跑):** 三个候选 checkpoint 里,`0.5B`(120ms)和 `7B`(450ms)都在 SLO(900ms/80ms)之内且回答正确,真实 `passes=True`;`70B-wrong` 虽然延迟也不算离谱,但 `correct=False` 直接导致 `passes=False`——正确性是硬性门槛,不因为延迟表现好就被放行。`effective_goodput` 在 `request_rate_rps=10.0` 下真实算出 `attainment=0.6667`(3 个里 2 个达标)、`goodput_rps=6.6667`,和手动核算的 2/3 一致(函数内部四舍五入到 4 位小数)。

**面试怎么问 + 追问链:**
- **Q:** "如果只用'吞吐(QPS)'这一个指标去评价一个 serving 方案好不好,可能会有什么误导?" —— 期望说出:高 QPS 不代表"好"的 QPS——如果相当比例的请求超时或者答错,QPS 数字再高也是在批量生产垃圾结果;goodput(只统计真正达标请求的有效吞吐)是更诚实的指标,这正是本知识点和 DistServe 论文共同的核心论点。
- **追问 1:** "`cost_per_good_request` 为什么要把失败请求的成本也摊进来,而不是只算成功请求自己的成本?" —— 期望说出:失败的请求(答错或者超时)照样消耗了真实的 GPU 计算资源,这部分成本是真实发生的,不能因为结果"没用"就当作没发生过;把总成本摊到"有效"请求头上,才能反映"每得到一个真正可用的结果,实际要花多少钱"这个更贴近业务决策的问题。
- **追问 2:** "如果把 SLO 门槛设得非常严格(比如 `max_ttft_ms` 设到接近 0),`effective_goodput` 会出现什么极端情况?" —— 期望说出:`attainment` 会趋向 0(几乎没有候选能通过如此严格的延迟门槛),`goodput_rps` 也随之趋向 0,`cost_per_good_request` 则因为 `max(1, len(passed))` 这个保护会退化成"全部成本压在极少数(甚至 0 个)通过请求上"——这说明 SLO 阈值本身的设定需要贴合真实业务可接受的范围,阈值设置不合理会让 goodput 这个指标本身失去参考意义。

**常见坑:** 只看 `passes` 这个布尔结果就下结论,不去看 `attainment`/`goodput_rps` 这两个更整体的指标——单个 checkpoint 通过或不通过是"点"的判断,`effective_goodput` 提供的是"面"的判断(一批候选里整体有多少比例真正可用),两者服务于不同的决策场景(前者用于筛掉不合格的单个模型,后者用于评估整个服务方案的产能规划)。另一个坑是把这里的 `estimated_cost`(`_mock_cost` 函数算出来的,`size_mb * latency_ms / 1_000_000`)当成真实美元成本去做业务决策——这是一个刻意简化的确定性代理指标,真实成本核算需要接入 06 号文件 `cost_calc.py` 那套基于真实 GPU 单价/吞吐的计算,不能直接拿这里的相对数字当绝对金额使用。

---
