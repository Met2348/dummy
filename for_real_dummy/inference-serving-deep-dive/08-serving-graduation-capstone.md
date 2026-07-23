# 08 · Module 5 毕业答辩 Capstone

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 8 个"知识点",是把 `learning/serving-graduation/` 自己的收官三段——L12(Capstone-1:R1-tiny mock 部署)、L13(五线综合 lecture)、L14(Capstone-2:五线综合毕业作品)——串成一场"Module 5《用大模型》毕业答辩"。和 [dsa-deep-dive](../dsa-deep-dive/19-mock-interview-capstone.md)/[statistics-deep-dive](../statistics-deep-dive/21-mock-interview-capstone.md) 的"模拟终面"体裁不同(那两篇源材料本身就是一道题层层追问),这里源材料自己是三段独立但递进的既有产出(一个部署 demo + 一篇综述 + 一份对比报告),所以采用更贴合"答辩"这个真实场景的体裁:候选人依次展示三段成果,答辩委员会针对每一段追问,候选人当场验证、也当场承认验证中发现的真实缺陷——这和面试的区别只在"题目从哪来",不变的是仓库一贯的纪律:**每一句"我觉得"都配一段真实跑过的代码**。

---

## 开场:答辩场景(0:00)

> **答辩委员会:** "你已经学完了 Module 1(PEFT)、Module 3(造大模型)、Module 4(改大模型)、Module 5(用大模型)一共 25 个 topic。今天用三段作品证明你真的吃透了'用'这一部分,以及你知道这三段作品各自的真实边界在哪里——不是只会说'能跑',是清楚知道'跑出来的东西在多大程度上可信'。"

候选人依次展示 R1-tiny 部署 demo(幕一)、五线综合回顾(幕二)、五 checkpoint 端到端对比(幕三)。

---

## 幕一:Capstone-1 —— R1-tiny mock 部署(0:00 – 0:18)

**候选人展示:** 这是 `r1_tiny_deploy/serve.py` 的部署管线——把 Module 4 reasoning-r1 capstone 训出的 GPT-2-M R1-Zero checkpoint,经过 Module 5 Topic 4 的 AWQ 4bit 量化,再用 Module 5 Topic 6 的 vLLM 流式 API 包装,对 5 道数学题跑一遍,验收条件是"p50 延迟 < 1s 且 5 道题至少 3 道正确"。

```python
import sys
sys.path.insert(0, "learning/serving-graduation/src")
from r1_tiny_deploy.serve import MockR1Model, run_demo, QUESTIONS

model = MockR1Model()
out = run_demo()
for r in out:
    print(f"  Q: {r['question'][:50]!r:54} thinking_present={r['thinking_present']}  answer={r['answer']!r}")

assert len(out) == 5
assert all(r["thinking_present"] for r in out)
assert all(r["answer"] for r in out)
print("表面验收: 5道题全部跑出 thinking_present=True 且 answer 非空 —— 乍看全部达标")
```

**答辩委员会追问:** "你刚才说'5 道题全部有 thinking、全部有 answer',这就等于'5 道题的推理是对的'吗?这 5 道题的**真实答案**你核对过吗?"

**候选人(当场核对,发现真实问题):** 没有——`run_demo()` 本身只检查 `thinking_present` 和 `answer` 是否非空,从没有拿真实答案去比对过。现在补上这一步:

```python
import sys
sys.path.insert(0, "learning/serving-graduation/src")
from r1_tiny_deploy.serve import MockR1Model, QUESTIONS

model = MockR1Model()
# 对 5 道完全不同的题目分别调用 model.stream(),看 answer 是否真的因题而异
answers = []
for q in QUESTIONS:
    full = "".join(model.stream(q))
    ans = full.split("<answer>")[1].split("</answer>")[0] if "<answer>" in full else None
    answers.append(ans)

print("5 道题各自的 mock 输出 answer:", answers)
assert answers == ["18", "18", "18", "18", "18"]      # 5 个完全不同的问题,答案完全一样
print("确认: model.stream() 根本没有读 prompt 参数,对任何输入恒定复读同一段 <think>...18")

# 5 道题的真实数学答案分别是多少
true_answers = [18, 7, 120, 56, 55]   # Janet鸡蛋题=18 / x+5=12→x=7 / 60mph*2h=120 / 7*8=56 / 1..10求和=55
real_correct = sum(a == str(t) for a, t in zip(answers, true_answers))
assert real_correct == 1               # 只有第1题(Janet鸡蛋题)真实答案恰好是18,纯属巧合
print(f"若按真实答案核对: {real_correct}/5 正确 —— 远低于验收条件要求的 >=3/5")
```

**候选人的结论(当场承认,不回避):** `MockR1Model.stream()` 的函数签名虽然接收 `prompt: str` 参数,但函数体从头到尾没有用过这个参数一次——不管问哪道题,它都原样吐出同一段写死的 `"16-3-4=9,and9*2=18"` 推理轨迹和答案 `"18"`。`run_demo()` 自己的验收逻辑(`thinking_present`/`answer` 是否非空)根本没有拿真实答案去比对,所以表面上"5 道题全部有输出"看起来是过关的,但如果按 lecture 自己写的验收条件"5 道数学题 ≥ 3 道正确"去真实核对,结果是 **1/5**,连及格线的三分之一都不到——这道验收条件在代码层面从未被真正检验过,只是"看起来像是通过了"。

**答辩委员会:** "如果这是一次真实的产线验收,你会怎么处理这个发现?" —— **候选人:** 不会因为"这只是 mock demo"就放过去。会在验收报告里明确写"当前 `run_demo()` 的验收逻辑存在盲区,`thinking_present`/`answer` 非空不能代表内容正确,需要补一个基于真实标准答案的比对断言,否则这个验收条件本质上从未被真正测试过"——这正是全系列反复强调的"看到有输出不等于看到对的输出"这条纪律在毕业作品自己身上的验证。

---

## 幕二:五线综合回顾(0:18 – 0:34)

**候选人展示:** L13 lecture 用一句公式总结了任意 LLM 服务的构成——`Serve(model, quant, schedule, kv_mgmt, spec_decode) → API`,并给出 5 个杠杆各自的加速量级:

| lever | speedup | 对应本系列文件 |
|-------|---------|---------------|
| model size ↓ | 5-10x | (Module 3,本仓库暂无对应系列) |
| quant(FP16→AWQ-4) | 1.6-2x | [04-quantization-deploy.md](04-quantization-deploy.md) |
| schedule(naive→continuous+chunked) | 5-10x | [01-inference-engine-core.md](01-inference-engine-core.md) |
| KV mgmt(naive→paged) | 2-3x | [01](01-inference-engine-core.md)/[02](02-sglang-radixattention.md) |
| spec decode(EAGLE-2) | 2-4x | [03-speculative-decoding.md](03-speculative-decoding.md) |

lecture 自己算出理论叠加加速比 500-2000x(相对 naive 2022 GPT-3 部署),并用 DeepSeek-V3(100k tok/s)和 Claude Sonnet 定价拆解两个真实案例佐证这套框架。

**答辩委员会追问:** "这 5 个杠杆的加速比,是这门课自己测出来的,还是行业公开数据?你能不能验证至少一个数字?"

**候选人:** 这是 lecture 给出的行业经验数字,不是本系列自己实测的——但"KV mgmt(naive→paged)2-3x"这一条,我们在 01 号文件里确实做过独立测量,可以直接核对量级是否吻合:

```python
# 01 号文件知识点 9 已经独立测过 CUDA Graph 的加速比(不是这里 lecture 说的 paged attention,
# 是"调度"这个维度下的另一个具体手段),这里检索的是"数字量级是否落在合理范围",不是重新
# 逐一实测 5 个杠杆——那已经分别是 01-05 号文件各自的独立工作
import sys
sys.path.insert(0, "learning/inference-engine-core/src")

try:
    from paged_kv import demo_paged_attention_savings   # 若函数名不存在,在此处 ImportError,走 except 分支
    result = demo_paged_attention_savings()
    print("paged_kv 模块自带的对比函数:", result)
except (ImportError, AttributeError) as e:
    print(f"paged_kv.py 未提供同名对比函数({e});改用 01 号文件知识点 3 已经验证过的",
          "PagedAttention vs NaiveKV 碎片对比结论做交叉核对")
    from naive_kv import demo_fragmentation
    frag = demo_fragmentation(B=8, max_len=2048, avg_len=512)
    print("naive_kv 静态预留碎片浪费:", frag)
    assert frag["utilization"] < 0.5   # 01号文件已验证: 碎片浪费导致真实利用率显著低于100%
    print(f"确认: naive 静态预留下真实利用率仅 {frag['utilization']*100:.1f}%,")
    print(f"paged 方案理论上能把这部分浪费收回来,量级上支持 lecture 给出的 2-3x 这个范围")
```

**实测(`.venv` 真跑):** `paged_kv.py` 没有现成的、和 `naive_kv.py` 直接对比加速比的函数,改用 01 号文件知识点 2 已经验证过的 `naive_kv.demo_fragmentation()`——真实测出静态预留 KV cache 在 `B=8, max_len=2048, avg_len=512` 这组配置下利用率显著低于 100%,这个真实浪费量级和 lecture "paged 方案带来 2-3x" 的说法方向一致、量级不冲突,但没有一个单独实验能精确复现"2-3x"这个具体数字——如实标注:lecture 的 5 个杠杆加速比是行业经验区间,本系列 01-05 号文件各自独立验证过其中每个杠杆"确实有正向收益、且原理成立",但没有哪个文件专门测过"精确是几倍"这个数字本身,不应该把行业经验区间当成本系列自己的实测结论去引用。

**候选人主动补充边界声明:** L13 的"25 个 topic"框架里,Module 3(造大模型)的 Phi-tiny 270M 和 Module 4(改大模型)的 reasoning-r1/R1-Zero,本仓库目前**没有**对应的 `for_real_dummy` 精读系列——这两条线在幕三的五线对比里只会作为"消费方"出现(直接用预置的 mock checkpoint 数据),不深入它们各自的训练机制。已经完整覆盖的是 Module 1(PEFT,见 [peft-deep-dive](../peft-deep-dive/00-roadmap.md))和 Module 4 里的 DPO 对齐部分(见 [alignment-algorithms-deep-dive](../alignment-algorithms-deep-dive/00-roadmap.md))。

---

## 幕三:Capstone-2 —— 五线综合毕业对比(0:34 – 0:58)

**候选人展示:** `graduation_e2e/` 用同一道 Janet 鸡蛋题,对比 5 个 checkpoint(vanilla GPT-2 / LoRA-tuned / DPO-aligned / R1-Zero / Phi-tiny)的表现:

```python
import sys, tempfile, os
sys.path.insert(0, "learning/serving-graduation/src")
from graduation_e2e.ckpts import load_all, CKPTS, QUESTION
from graduation_e2e.compare import run_compare, to_md
from graduation_e2e.report import write_report

ck = load_all()
assert set(ck.keys()) == {"vanilla", "lora", "dpo", "r1_zero", "phi_tiny"}

report = run_compare()
assert report["question"] == QUESTION
assert len(report["results"]) == 5

with tempfile.TemporaryDirectory() as d:
    paths = write_report(d)
    for name in paths:
        p = os.path.join(d, name)
        assert os.path.exists(p) and os.path.getsize(p) > 0
    print(f"write_report 真实写盘: {list(paths.keys())}, 均非空文件")

for row in report["results"]:
    print(f"  {row['ckpt']:>9}: correct={row['correct']!s:5} latency={row['latency_ms']:>4}ms reasoning={row['reasoning']}")
```

**答辩委员会追问:** "这 5 个 checkpoint 的 response 和 latency,是真的分别拿 5 个模型跑出来的,还是写死的?"

**候选人(如实核对源码,不回避):** 是写死的——`graduation_e2e/ckpts.py` 里 `CKPTS` 这个字典的 5 条记录(`response`/`latency_ms`/`correct`/`size_mb`)全部是字面量常量,不涉及任何真实模型加载或推理:

```python
import sys
sys.path.insert(0, "learning/serving-graduation/src")
from graduation_e2e.ckpts import CKPTS

# 连续调用两次 load_all(),如果背后是真推理,两次的 latency 应该有真实抖动;
# 如果是预烤数据,两次结果应该逐字节相同
from graduation_e2e.ckpts import load_all
first = {k: v.response for k, v in load_all().items()}
second = {k: v.response for k, v in load_all().items()}
assert first == second   # 完全相同,连一个字符的抖动都没有
print("确认: 两次调用 load_all() 结果逐字符相同,是纯字面量字典查表,不是每次重新推理")

for name, c in CKPTS.items():
    print(f"  {name}: latency_ms={c.latency_ms} (常量), response={c.response[:40]!r}...")
```

`compare.py`/`report.py` 这两层是真实代码(真实遍历、真实拼 markdown、真实写盘),但它们消费的输入数据本身是预置的——这是"流程真实,数据是 fixture"的诚实边界,和幕一 `MockR1Model` 那种"连流程输出都和输入无关"的情况不一样,不能混为一谈:`compare.py`/`report.py` 如果换成真实 5 个模型的推理结果输入,这两层代码不需要改一行就能正常工作。

**答辩委员会追问(收官的追问,把 L11 和 L14 接起来):** "你在幕二说 Module 5 学的是'用大模型',07 号文件知识点 9 你做了一个服务工程评分卡(`serving_scorecard.py`)。这份五线对比数据,能不能用你自己做的评分卡真实评一遍,而不是只看 `correct`/`reasoning` 这种定性描述?"

**候选人(现场把 L11 的工具应用到 L14 的真实数据上):**

```python
import sys
sys.path.insert(0, "learning/serving-graduation/src")
from graduation_e2e.compare import run_compare
from serving_scorecard import GraduationSLO, score_report, effective_goodput, rank_candidates

report = run_compare()
strict_slo = GraduationSLO(max_ttft_ms=70, max_tpot_ms=8)
scores = score_report(report, strict_slo)
by_ckpt = {s.ckpt: s for s in scores}

for s in scores:
    print(f"  {s.ckpt:>10}: correct={s.correct!s:5} ttft={s.ttft_ms:>5.1f}ms tpot={s.tpot_ms:>4.2f}ms passes={s.passes}")

assert by_ckpt["vanilla"].passes is False    # 本来就答错,理所当然不过
assert by_ckpt["r1_zero"].passes is False    # 答对了、reasoning质量全场最高,但 ttft=80ms > 70ms 门槛
assert by_ckpt["lora"].passes is True
assert by_ckpt["dpo"].passes is True
assert by_ckpt["phi_tiny"].passes is True

gp = effective_goodput(scores, request_rate_rps=2.0)
assert gp["attainment"] == 0.6
ranked = rank_candidates(scores)
assert ranked[0].ckpt == "lora"              # 排名第一的不是"最强"的 r1_zero,是"够用且最快"的 lora
print(f"goodput@2rps = {gp}")
print(f"通过 SLO 优先排名 = {[s.ckpt for s in ranked]}")
```

**实测(`.venv` 真跑):** 在严格 SLO(`max_ttft_ms=70, max_tpot_ms=8`)下真实评分,`r1_zero`——五个 checkpoint 里唯一 `reasoning_quality="strong"`、推理过程最完整的一个——因为 `ttft=80ms` 超过 70ms 门槛,`passes=False`;反而是 `reasoning_quality` 只有 `"brief"` 的 `lora`(35ms)排到了 `rank_candidates` 的第一位。`effective_goodput` 算出 `attainment=0.6`(5 个里 3 个真正达标)。

**候选人的结论:** 这不是评分卡代码有 bug,是这份评分卡**如实揭示了一个真实的工程张力**——"推理链条最完整、最像人类思考"和"能在生产 SLO 下稳定交付"是两个不同的评价维度,前者是这门课 Module 4(改大模型)想要的效果,后者是 Module 5(用大模型)必须面对的约束。如果只看 L14 自己的对比表格(`correct`/`reasoning` 两列定性描述),`r1_zero` 看起来是"最好的模型";但套上一个真实的延迟 SLO 门槛重新评一遍,排名第一的位置会让给一个"没那么会想、但足够快"的方案——这正是 07 号文件知识点 9 想要说明的"不能只看吞吐/不能只看单一维度"这条原则,在毕业作品自己的数据上得到了真实印证,不是自问自答式的假设,是把两份本系列自己写的真实代码接在一起、跑出来的真实排名反转。

---

## 复盘小结(0:58 – 1:00)

**候选人主动总结(不等答辩委员会问):** 三段作品分别暴露出三种不同性质的"真实边界",按严重程度递增排列:

1. **幕三 `graduation_e2e/`**——流程代码真实,输入数据是刻意声明的预烤 fixture,`compare.py`/`report.py` 换上真推理结果也能直接工作。这是最"干净"的一种 mock:边界清楚,不会被误用。
2. **幕二 L13 的 5 杠杆加速比**——是行业经验数字,本系列 01-05 号文件各自验证过"每个杠杆方向都对",但没有单独验证过"精确是几倍",这是"结论范围正确、精确数值未逐一核实"的中间状态。
3. **幕一 `MockR1Model.stream()`**——最严重的一种:函数签名上有 `prompt` 参数,却完全不读它,导致这个 capstone 自己写的验收条件("5 道题 ≥ 3 道正确")从未被真正检验过,肉眼看输出"像是对的"(有 thinking、有 answer),实际核对后只有 1/5 真实正确。

这三种边界连起来,恰好是整个 `for_real_dummy/inference-serving-deep-dive/` 系列(01-08,含 06 号文件知识点 12 已经补写完成的真实部署 bonus——WSL2 + 真实 `vllm serve` + 真实 OpenAI 兼容 API 请求,过程中真实踩中并解决了 FFmpeg 缺失在内的五类环境阻塞)想要反复练习的同一件事:**"能跑出输出"和"输出是对的"之间,永远要留一步真实核对,不能用"看起来合理"替代"验证过"**。06 号文件独立发现的 `openai_api_server.py` 422 bug、07 号文件独立发现的 `generate_with_budget()` 丢答案 bug,以及这里幕一发现的 `MockR1Model` 恒定输出 bug,是同一条纪律在三个完全不同代码位置上抓到的三个真实缺陷——不是因为这个仓库代码质量差,是因为**每一处都真的去核对了**,而不是假设"能跑就是对的"。

Module 5《用大模型》到这里闭环:造(Module 3)→ 改(Module 4)→ 用(Module 5),量化 + 调度 + serve + 五线综合。下一程如 L13 自己列出的候选方向——Module 6(评测/安全)、Module 7(Agent 应用层)——留给以后单独的 massive 专题,不在本系列范围内。

---
