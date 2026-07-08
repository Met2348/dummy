# 04 · 字符串与现代语法惯用法(Strings & Modern Syntax)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一篇覆盖 5 个知识点:f-string 高级用法、`str.join`(以及为什么不要在循环里用 `+` 拼接字符串)、`str.split`/`strip`/`partition`、`pathlib.Path` 面向对象路径操作、`match`-`case` 结构化模式匹配(3.10+)。是 [python-advanced/](../python-advanced/00-roadmap.md) 系列"语言特性"(装饰器/生成器/OOP)之外的另一半——这里讲的是"同一件事有很多种写法时,哪一种更接近地道 Python 的表达习惯"。

本文所有代码例子已在仓库 `.venv`(Python 3.13.9)下实际跑通验证,不是凭空写的。"AI 研究代码里的真实例子"部分优先引用/复现 `learning/` 目录下博士学长自己写的真实代码(标注了具体文件路径和行号,不算 vendor 进来的第三方仓库 `official/repos/`),大部分点是直接 `sys.path.insert` 后真实 `import` 并调用,不是凭印象复现的假代码。第 1 点(f-string)最初想用的真实例子依赖 torch,但本次环境里 `import torch` 异常缓慢(实测 90 秒未完成,判断是本机瞬时问题,和这个知识点本身无关),于是换成了同样真实存在、不依赖 torch 的另一份源码。第 5 点(`match`-`case`)在仓库自己写的代码里暂未挖到真实使用,已如实标注为示例性质,没有编造假的仓库引用。

---

## 1. f-string 高级用法

**是什么:** `f"..."` 里的 `{}` 不只是"填进一个变量"——冒号后面能跟"格式说明符"控制精度/宽度/对齐/千分位等显示细节(比如 `f"{x:.2f}"`、`f"{x:>10}"`);3.8+ 支持在表达式末尾加一个 `=`,直接打印"表达式文本=值"(比如 `f"{x=}"`);`{}` 里能放的也不只是变量名,任意表达式、方法调用、推导式都可以塞进去。

**为什么课堂不教但很重要:** 课堂上的字符串格式化大多停留在 `print("x =", x)`,或者顶多 `.format()`,格式说明符这种"显示细节控制"通常被跳过。但训练/评测脚本几乎每一行日志都在跟这些细节打交道:loss 要固定小数位数不然刷屏刷得眼花,参数量/吞吐量要千分位分隔和宽度对齐才能一眼扫出表格,调试时想插一行"打印这个变量"又不想手打变量名(改名字时最容易忘记同步)。这些不是锦上添花,是"每天写日志、读日志"的真实痛点。

**从最笨的写法讲起:**

先看最基础的插值,三种写法功能等价:
```python
loss = 2.345678

old_percent = "loss=%.2f" % loss              # 最老的写法: % 格式化,格式代码要单独记一套符号
old_format = "loss={:.2f}".format(loss)         # .format() 写法: 变量和占位符分开写,长表达式时要来回对照
new_fstring = f"loss={loss:.2f}"                # f-string: 变量直接写在花括号里,不用来回跳着看

assert old_percent == old_format == new_fstring == "loss=2.35"
print(old_percent, "|", old_format, "|", new_fstring)
```
三种写法结果一样,但 f-string 把"用什么值"和"怎么格式化"写在同一个位置,`%` 和 `.format()` 都得让读者在字符串模板和参数列表之间来回对照才能配对上。

再看"调试打印一个变量"这个更具体的场景——笨办法要把变量名手打两遍,变量改名时很容易漏改一处:
```python
batch_size = 32

# 笨办法: "batch_size" 这个名字被打了两遍——一遍是字符串字面量,一遍是真正的变量名。
# 如果哪天把变量改名成 bsz,字符串里这一份很容易忘记跟着改,读起来就会文不对题
dumb_debug = f"batch_size={batch_size}"

# 正式写法(3.8+): 表达式后面加一个 `=`,Python 自动把"表达式的源码文本"和"算出来的值"都打印出来,
# 只有一份 "batch_size",不会出现两份文本不同步的问题
smart_debug = f"{batch_size=}"

assert dumb_debug == smart_debug == "batch_size=32"
print(dumb_debug, "|", smart_debug)
```

**AI 研究代码里的真实例子:** `learning/data-curation/src/vocab_compare.py` 第 36-47 行,横向比较 gpt2/cl100k_base/o200k_base 三种 tokenizer 在 5 种语言样本下的压缩率,打印一张对齐的表格,同时用到了左对齐 `:<10`、右对齐 `:>5`/`:>20`、小数精度 `:.2f`:
```python
import sys
import io
from contextlib import redirect_stdout

sys.path.insert(0, r"E:\Workspace\dummy\learning\data-curation\src")
import vocab_compare as vc

# 真实写法(vocab_compare.py:36-37):
#     print(f"\n{'sample':<10} {'chars':>5} | " + " | ".join(f"{m:>20}" for m in models))
# 真实写法(vocab_compare.py:41,47):
#     row = [f"{name:<10}", f"{n_chars:>5}"]; row.append(f"{n_tok:>5} tok ({ratio:.2f} c/t)")
buf = io.StringIO()
with redirect_stdout(buf):
    vc.main()          # 真实调用,内部真的跑了一遍 tiktoken 编码,不是伪造的输出
out = buf.getvalue()
print(out)

assert "gpt2" in out and "cl100k_base" in out        # 表头(:>20 右对齐)
assert "english" in out and "chinese" in out           # SAMPLES 里的语言样本
for line in out.splitlines():
    if line.startswith("chinese"):
        assert "tok (" in line and "c/t)" in line       # `{ratio:.2f} c/t)` 确实生效了
        break
print("[OK] 真的调用了 vocab_compare.main(),不是复现的假数据")
```
实测结果很有意思:中文样本在 `gpt2` 词表下压缩率只有 0.48 chars/token(中文字符大多要拆成多个 byte-level token),换成 `cl100k_base` 提升到 1.00,`o200k_base` 进一步到 1.54——这也是"为什么新模型要换词表"的一个直观数据支撑,而这整张对比表能对齐,全靠 `:<10`/`:>5`/`:>20` 这些宽度说明符。

**可运行例子:**
```python
# 格式说明符: 精度 / 宽度对齐 / 千分位 / 百分号
x = 3.14159265
assert f"{x:.2f}" == "3.14"

n = 42
assert f"{n:>10}" == "        42"     # 右对齐,宽度10
assert f"{n:<10}" == "42        "     # 左对齐,宽度10
assert f"{n:^10}" == "    42    "     # 居中,宽度10

big = 1234567
assert f"{big:,}" == "1,234,567"          # 千分位分隔符
assert f"{big:>12,}" == "   1,234,567"    # 宽度 + 千分位组合使用

ratio = 0.4567
assert f"{ratio:.2%}" == "45.67%"     # 百分号格式(自动乘100再加%)

# `=` 调试写法可以搭配格式说明符,也可以用在任意表达式上,不局限于单个变量名
lr = 0.001234
assert f"{lr=:.4f}" == "lr=0.0012"

values = [1, 2, 3, 4]
assert f"{sum(values)=}" == "sum(values)=10"

# {} 里能塞方法调用/推导式/三元表达式,不局限于"一个变量"
name = "gpt-mini"
assert f"{name.upper()}" == "GPT-MINI"
assert f"{[v * 2 for v in values]}" == "[2, 4, 6, 8]"
assert f"{'yes' if len(values) > 2 else 'no'}" == "yes"

print("[OK] f-string 高级用法全部验证通过")
```

**常见坑:**

**1. 单个 `{`/`}` 是表达式定界符,字面大括号必须写成 `{{`/`}}`,写错会直接 `SyntaxError`(编译期就能发现,不是运行时才炸):**
```python
assert f"{{literal brace}}" == "{literal brace}"

try:
    eval('f"{not closed"')
    raise AssertionError("应该 SyntaxError")
except SyntaxError as e:
    print("缺右括号直接 SyntaxError:", e)
```

**2. 格式说明符和数据类型强绑定,类型不对会直接报错,不是"忽略格式静默输出":**
```python
try:
    _ = f"{'abc':,}"        # 千分位分隔符是数字专属的,字符串用不了
    raise AssertionError("应该 ValueError")
except ValueError as e:
    print("给字符串用 `,` 千分位触发 ValueError:", e)   # Cannot specify ',' with 's'.
```

**3. f-string 是立即求值,不是惰性的——写日志时特别容易踩这个坑。** 对比 `logging` 模块的 `%s` 风格:`logger.debug("x=%s", expensive())` 只有在 debug 级别真的启用时才会调用 `expensive()`;而 `logger.debug(f"x={expensive()}")` 不管日志级别开没开,`expensive()` 都会先被执行一遍才能拼出这个字符串,白白浪费性能:
```python
calls = []

def expensive():
    calls.append(1)
    return "computed"

message = f"value={expensive()}"     # 即使这行最终没被用到/没被打印,expensive() 也已经跑完了
assert len(calls) == 1
print("f-string 参数总是立即求值,不像 logger.debug('%s', fn) 那样惰性")
```

**4.(版本相关)Python 3.12 之前,f-string 表达式内部不能复用和外层相同的引号字符**(比如 `f"{"abc"}"` 会报语法错误),必须换一种引号或者提前把值存到变量里;3.12(PEP 701)才放开了这个限制。我们环境是 3.13.9,不受影响,但如果代码要兼容更老的 Python,这一条还是要注意。

---

## 2. `str.join`(以及为什么不要在循环里用 `+` 拼接字符串)

**是什么:** `sep.join(iterable)` ——用 `sep` 这个字符串当"胶水",把一堆字符串粘成一个。注意调用顺序:**分隔符是调用者,要拼的列表是参数**,不是反过来。

**为什么课堂不教但很重要:** 课堂教字符串拼接基本只讲 `+`,顶多提一句"字符串是不可变的"。但训练/评测脚本里经常要把一长串 token、参数名、markdown 表格行拼成一份日志或报告——这时候在循环里 `result += piece` 不仅有性能隐患(下面用 `timeit` 实测),还容易在"要不要加分隔符"这件事上出 bug(最后一个元素后面多一个逗号,或者第一个元素前面少个空格,都是这么攒出来的)。`sep.join(list)` 把"攒内容"和"拼接"这两件事分开:先把每一段准备好放进一个 list,最后一次性拼完,不用操心"最后一个元素后面要不要加分隔符"这种边界情况。

**从最笨的写法讲起:**
```python
words = ["gpt-mini", "loss", "2.35"]

# 笨办法: 循环里用 += 累加,还得手动把最后多出来的分隔符去掉
concat_dumb = ""
for w in words:
    concat_dumb += w + "-"
concat_dumb = concat_dumb.rstrip("-")

# 正式写法: 先明确"拼接列表" + "分隔符",join 一次性拼完,没有"多余分隔符"这个善后步骤
concat_join = "-".join(words)

assert concat_dumb == concat_join == "gpt-mini-loss-2.35"
print(concat_dumb, "|", concat_join)
```
笨办法这里还只是"结果一样、多了一步善后",真正的问题是性能——下面用 `timeit` 实测。

**AI 研究代码里的真实例子:** `learning/agent-foundations/src/common.py` 第 51-58 行,`Trace.to_md()` 把一次 ReAct 轨迹渲染成 markdown 表格:先把表头、每一步的表格行、结尾的 Final 逐个 `.append()` 进 `lines` 列表,最后 `"\n".join(lines)` 一次拼完:
```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\agent-foundations\src")
from common import Trace, Step, Tool
from react_loop import build_initial_prompt

trace = Trace(question="2+3 等于几?")
trace.add(Step(step_num=1, thought="需要计算 2+3", action="calculator(2+3)", observation="5"))
trace.final = "5"

md = trace.to_md()    # 真实调用: 内部就是 "\n".join(lines),不是复现的假代码
assert md.startswith("# Trace\n\nQ: 2+3 等于几?\n")
assert "| 1 | 需要计算 2+3 | calculator(2+3) | 5 |" in md
assert md.endswith("\nFinal: 5")
print(md)

# 同一份源码里,react_loop.py:24 也是同款用法: names = ", ".join(tools.keys())
tools = {"calculator": Tool(name="calculator", description="do math", schema={}, func=lambda a: a)}
prompt = build_initial_prompt("What is 2+3?", tools)
assert "Available tools: calculator" in prompt
print("[OK] 真实 import 自 agent-foundations/src,不是复现")
```

**可运行例子:** 严格只测字符串拼接本身,不引入任何无关计算(这是本仓库 torch-deep-dive 08 篇踩过的真实教训——一次把无关的重计算也算进了计时循环,导致真实的性能差异信号被噪声淹没,得出了误导性的结论),用 `timeit.repeat` 多次取最小值抗抖动,拼接内容保持极简(固定长度短字符串):
```python
import timeit

def concat_with_plus(words: list[str]) -> str:
    result = ""
    for w in words:
        result += w
    return result

def concat_with_join(words: list[str]) -> str:
    return "".join(words)

words = [f"tok{i}" for i in range(5_000)]     # 5000 个短字符串,内容极简
assert concat_with_plus(words) == concat_with_join(words)   # 先确认结果一致,再谈性能

t_plus = min(timeit.repeat(lambda: concat_with_plus(words), number=50, repeat=5))
t_join = min(timeit.repeat(lambda: concat_with_join(words), number=50, repeat=5))

assert t_join < t_plus     # 只断言"谁更快"这个方向,不断言具体倍数(倍数会因机器/负载而异)
print(f"+= 循环: {t_plus:.5f}s   join: {t_join:.5f}s   本机实测 join 快 {t_plus / t_join:.1f} 倍")
```
本机(仓库 `.venv`,Python 3.13.9)多次独立实测,`n=500`/`5000`/`50000` 三个规模下 `join` 都稳定比 `+=` 快 20~30 倍左右,而且比值不会随规模爆炸式增长——这是因为 CPython 对"纯粹累加到同一个变量"这种模式做了就地扩容优化(不是教科书里常说的朴素 O(n²)),但 `join` 提前算出总长度、一次性分配好内存,依然稳定更快。更重要的是,这个"就地优化"是 CPython 的实现细节,不是 Python 语言规范保证的行为——`join` 才是不挑具体解释器实现的地道写法,这也是为什么前面 `agent-foundations` 的真实代码里全是 `.append()` + `join`,没有一处在循环里 `+=` 拼接。

**常见坑:**

**1. `join` 的参数必须是"字符串的可迭代对象",混入非字符串元素(比如 `int`)会直接 `TypeError`,不会自动转换:**
```python
try:
    "-".join(["a", 1, "b"])
    raise AssertionError("应该 TypeError")
except TypeError as e:
    print("join 列表里混入非字符串触发 TypeError:", e)
    # sequence item 1: expected str instance, int found
```

**2. 分隔符是调用者,不是参数——新手容易写反成 `list.join(sep)`,但 list 根本没有 `join` 方法:**
```python
try:
    ["a", "b"].join("-")
    raise AssertionError("应该 AttributeError")
except AttributeError as e:
    print("list 没有 join 方法,正确写法是 '-'.join(list):", e)
```

**3. 生成器只能消费一次,被 `join` 用过一次之后再 `join` 同一个生成器,得到的是空字符串而不是报错——这个坑很容易被忽视,因为它不报错:**
```python
gen = (str(i) for i in range(3))
first = "-".join(gen)
assert first == "0-1-2"
second = "-".join(gen)      # 生成器已经耗尽
assert second == ""          # 静默给出空字符串,不是异常
print("生成器耗尽后再次 join:", repr(second))
```

---

## 3. `str.split` / `strip` / `partition` 系列

**是什么:** 三个字符串"拆分"方法,场景各不相同:`split(sep=None, maxsplit=-1)` 按分隔符切成一个 list(不传 `sep` 时按任意空白切);`strip(chars=None)` 去掉首尾两端属于 `chars` 这个**字符集合**的字符(不传时去空白);`partition(sep)` 按分隔符切成固定 3 元素的 `(before, sep, after)` tuple。

**为什么课堂不教但很重要:** 解析 LLM 输出、命令行参数、日志行的时候,经常要"从一行文本里挖出结构化信息"——课堂通常只教过 `split(",")` 处理规整的 CSV。但真实场景里,"1. do something"这种编号列表、`"key1=1, key2=hello"` 这种参数串,都需要 `maxsplit`、无参数 `split()`、`partition` 搭配使用;不知道 `split()` 不传参和传固定分隔符的区别,或者不知道"分隔符不存在时 `split` 的返回值长度会变短",很容易在边界情况(空字符串、分隔符缺失)直接崩掉。

**从最笨的写法讲起:**

先看 `split()` 不传参数 vs 传固定分隔符的区别——两者不是"要不要写参数"这么随意,行为完全不同:
```python
messy = "  the   quick\tbrown\nfox  "

# 不传参数: 按"任意空白"分割(空格/制表符/换行都算),连续空白当一个分隔符,首尾空白产生的空字符串会被自动丢弃
words = messy.split()
assert words == ["the", "quick", "brown", "fox"]

# 传固定分隔符: 严格按这一个分隔符切,连续分隔符之间、首尾多余的分隔符,都会产生空字符串,不会自动清理
csv_row = "a,,b,c,"
parts_fixed = csv_row.split(",")
assert parts_fixed == ["a", "", "b", "c", ""]     # 中间和末尾的空字符串都原样保留了

# 如果拿 messy 这种"多个空格"的字符串传固定的单个空格分隔符,行为和不传参数完全不同
assert messy.split(" ") != messy.split()
assert "" in messy.split(" ")          # 会夹杂很多空字符串
```

再看 `partition` 相比 `split(sep, 1)` 的优势——"只想切成前后两半"这个场景下,`partition` 不管分隔符存不存在,返回值长度永远是 3,可以放心解包;`split(sep, 1)` 在分隔符不存在时返回值长度会退化成 1,解包直接报错:
```python
kv = "learning_rate=0.001"
no_sep = "just_a_flag"

# split(sep, 1): 分隔符存在时长度是2,不存在时长度退化成1——这个"长度不稳定"就是坑
assert len(kv.split("=", 1)) == 2
assert len(no_sep.split("=", 1)) == 1     # 不是 2!直接 a, b = no_sep.split("=", 1) 会 ValueError

# partition: 不管分隔符存不存在,永远返回 3 元素 tuple,可以放心解包
k, sep, v = kv.partition("=")
assert (k, sep, v) == ("learning_rate", "=", "0.001")

k2, sep2, v2 = no_sep.partition("=")       # 分隔符不存在: sep 和 after 都是空字符串,但长度依然是3,不会报错
assert (k2, sep2, v2) == ("just_a_flag", "", "")
print("partition 无论分隔符是否存在,都能安全解包成 3 个变量")
```

**AI 研究代码里的真实例子:** `learning/agent-foundations/src/plan_execute.py` 第 18-27 行的 `parse_plan()`,把 planner LLM 输出的编号列表文本解析成一个 step 列表:
```python
import sys
sys.path.insert(0, r"E:\Workspace\dummy\learning\agent-foundations\src")
from plan_execute import parse_plan
from common import parse_action

# 真实写法(plan_execute.py:21-26):
#     for line in plan_text.split("\n"):
#         line = line.strip()
#         if not line: continue
#         if line[0].isdigit() and "." in line[:3]:
#             steps.append(line.split(".", 1)[1].strip())
planner_output = """1. Search for the current CEO of OpenAI
2. Look up their age.
3. Calculate age in days (approx.)"""

steps = parse_plan(planner_output)    # 真实调用
assert steps == [
    "Search for the current CEO of OpenAI",
    "Look up their age.",             # maxsplit=1: 句子本身的句号完好保留,只切开了编号后面那个"."
    "Calculate age in days (approx.)",
]
print("parse_plan 真实结果:", steps)

# common.py:77 的 k, v = part.split("=", 1) —— parse_action 解析 "tool(key=value, ...)" 里的参数
name, args = parse_action('Action 1: search(query="openai ceo", top_k=3)')
assert name == "search"
assert args == {"query": "openai ceo", "top_k": 3}
print("[OK] 真实 import 自 agent-foundations/src,不是复现")
```

**可运行例子:**
```python
# maxsplit 参数: 只分割前 N 次,常用来"只想拿到前面一小段,后面原样保留"
line = "1. call tool_a(x=1)"
number, rest = line.split(".", 1)
assert number == "1" and rest.strip() == "call tool_a(x=1)"

# strip() 也可以传一个"字符集合"去掉首尾指定字符,不是只能去空白
assert "  hello  ".strip() == "hello"
assert "xxhelloxx".strip("x") == "hello"

# split("\n") 和 splitlines() 对 \r\n 的处理不一样(下面"常见坑"细讲)
text = "a\r\nb\nc"
assert text.split("\n") == ["a\r", "b", "c"]
assert text.splitlines() == ["a", "b", "c"]

print("[OK] split/strip/partition 全部验证通过")
```

**常见坑:**

**1. `strip(chars)` 的参数是"字符集合",不是"要去掉的子串"——这是最容易被望文生义的坑:**
```python
result = "banana".strip("na")
assert result == "b"          # 反复从两端剥掉属于 {n, a} 的字符,不是只去掉"na"这个子串
assert result != "ba"          # 很多人以为结果是去掉开头/结尾的"na"这两个字符,实际不是
print('"banana".strip("na") =', repr(result), "不是很多人以为的\"去掉子串\"")
```

**2. `split("\n")` 和 `splitlines()` 对不同换行符的处理不一样,混用容易在 Windows(`\r\n`)文本上出隐蔽 bug:**
```python
text_crlf = "a\r\nb\nc"
assert text_crlf.split("\n") == ["a\r", "b", "c"]     # \r 被留在了上一段末尾,容易忘记再 strip 一次
assert text_crlf.splitlines() == ["a", "b", "c"]        # splitlines 认识 \r\n,自动处理干净
print("split('\\n') 会留下 \\r,splitlines() 不会")
```

---

## 4. `pathlib.Path` 面向对象路径操作

**是什么:** `pathlib.Path` 把"文件路径"包成一个对象,而不是一个普通字符串——拼路径用 `/` 运算符(`Path(a) / b`),常用信息直接是属性/方法:`.exists()`/`.is_file()`/`.suffix`/`.stem`/`.parent`/`.name`,不用再靠 `os.path.xxx()` 一堆外挂函数操作字符串。

**为什么课堂不教但很重要:** 课堂教 `open("data.txt")` 这种写死的相对路径就够用了,但研究代码经常要"脚本自己找到我所在目录的上一级的 `src/`",或者"用 arXiv id 拼一个下载目标文件名"——用 `os.path.join`/字符串拼接不仅啰嗦,Windows 和 Linux 的路径分隔符还不一样(`\\` vs `/`),字符串拼接很容易埋下"只在自己电脑上能跑"的隐患。`pathlib` 把这些平台细节封装掉了,而且路径变成对象之后,"这是不是一个文件"、"后缀是什么"这些操作直接是方法/属性,不用记一堆 `os.path.isfile()`/`os.path.splitext()` 这样的独立函数名。

**从最笨的写法讲起:**
```python
import os
from pathlib import Path

# 笨办法: os.path.join 返回的是普通字符串,后续想知道"这是不是文件"还得再调用别的函数
joined_dumb = os.path.join("learning", "python-idioms", "04-strings-and-modern-syntax.md")

# 正式写法: Path 用 / 运算符拼接,返回的是 Path 对象,不是字符串
joined_path = Path("learning") / "python-idioms" / "04-strings-and-modern-syntax.md"

assert joined_dumb == str(joined_path)      # 转成字符串后,同一个平台上内容相同
assert isinstance(joined_path, Path)
assert not isinstance(joined_dumb, Path)     # os.path.join 的返回值终究只是个字符串
print("os.path.join:", joined_dumb)
print("Path 对象:   ", joined_path)
```
两种写法在字符串内容上是等价的,区别在于:`Path` 对象后面能直接接 `.exists()`/`.suffix` 这些方法,`os.path.join` 的返回值只是个普通字符串,要做同样的事得再调用 `os.path.isfile(joined_dumb)`、`os.path.splitext(joined_dumb)` 这类独立函数——拼路径和查询路径信息是两套完全不同的 API。

**AI 研究代码里的真实例子:** `learning/agent-harness-frontier/papers/download_papers.py` 第 120、165-166 行,批量下载论文 PDF 时用 `Path` 拼目标文件名,并用 `.exists()` + `.stat().st_size` 做"已经下载过就跳过"的幂等判断:
```python
import sys
from pathlib import Path

sys.path.insert(0, r"E:\Workspace\dummy\learning\agent-harness-frontier\papers")
import download_papers as dp    # 真实 import,导入时不会触发任何下载(main() 只在 __main__ 守卫里跑)

# 真实写法(download_papers.py:120): HERE = Path(__file__).resolve().parent
assert isinstance(dp.HERE, Path)
assert dp.HERE.exists() and dp.HERE.is_dir()
assert dp.HERE.name == "papers"

# 真实写法(download_papers.py:165-166):
#     dest = HERE / f"{aid}-{slug}.pdf"
#     if dest.exists() and dest.stat().st_size > 10_000: 判定"已下载,跳过"
real_pdf = dp.HERE / "2210.03629-react-reasoning-and-acting.pdf"   # 仓库里真实已经下载好的一篇论文
assert real_pdf.exists() and real_pdf.is_file()
assert real_pdf.suffix == ".pdf"
assert real_pdf.stem == "2210.03629-react-reasoning-and-acting"
assert real_pdf.stat().st_size > 10_000
print(f"{real_pdf.name}: {real_pdf.stat().st_size:,} bytes,幂等判断结果 =",
      real_pdf.exists() and real_pdf.stat().st_size > 10_000)
```

**可运行例子:**
```python
import os
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    this_file = Path(tmp) / "demo_script.py"
    this_file.write_text("# demo\n", encoding="utf-8")

    assert this_file.is_file()
    assert this_file.suffix == ".py"            # 后缀(带点)
    assert this_file.stem == "demo_script"       # 不带后缀的文件名主干
    assert this_file.parent.is_dir()              # 父目录

# 不存在的路径, exists() 返回 False 而不是抛异常(和 open() 打开不存在的文件直接报错不同)
assert Path("this_file_does_not_exist_xyz123.txt").exists() is False

# 跨平台路径分隔符: str(Path) 用当前系统的分隔符,as_posix() 总是正斜杠(适合写进 URL/manifest)
p = Path("a") / "b" / "c.txt"
assert p.as_posix() == "a/b/c.txt"
if os.name == "nt":
    assert str(p) == "a\\b\\c.txt"

# mkdir(parents=True, exist_ok=True): 递归建目录,已存在也不报错
with tempfile.TemporaryDirectory() as tmp2:
    target = Path(tmp2) / "a" / "b" / "c"
    target.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)   # 再建一次也不报错
    assert target.is_dir()

print("[OK] pathlib 全部验证通过")
```

**常见坑:**

**1. `/` 运算符至少要有一边是 `Path` 对象,两边都是普通字符串会直接 `TypeError`(`str` 没有定义 `/`):**
```python
try:
    eval('"a" / "b"')
    raise AssertionError("应该 TypeError")
except TypeError as e:
    print('"a" / "b"(两边都是字符串)触发 TypeError:', e)

# 只要有一边是 Path,顺序不影响——Path 同时实现了 __truediv__ 和 __rtruediv__
from pathlib import Path
assert (Path("a") / "b") == Path("a/b") == ("a" / Path("b"))
```

**2. 用 `/` 拼接一个"绝对路径"字面量时,前面的部分会被直接丢弃——这一点和 `os.path.join` 的行为是一致的,不是 `pathlib` 更"智能",很容易被误以为两者不一样:**
```python
import os
from pathlib import Path

joined_os = os.path.join("a", "/b")
joined_path = Path("a") / "/b"
print("os.path.join('a','/b') =", joined_os)     # 结果是 "/b",前面的 "a" 被丢弃了
print("Path('a')/'/b'         =", joined_path)     # 同样丢弃了 "a",这一点两套 API 表现一致
```

---

## 5. `match`-`case` 结构化模式匹配(Python 3.10+)

**是什么:** `match subject: case pattern: ...` 是 Python 3.10 引入的结构化模式匹配语句,表面上像其他语言的 `switch`,但能力更强——不仅能匹配值,还能同时把数据"拆开"并把子部分绑定成变量(解构),外加 `if` 守卫条件、`|` 多选一等能力。

**为什么课堂不教但很重要:** 一是这个语法本身较新(2021 年才随 3.10 发布),不少教材/课堂内容还停留在 3.10 之前;二是很多人习惯了"用 `if/elif` 判断一个 tuple/list/dict 的形状,再手动用下标 `[0]`/`[1]` 取值",`match`-`case` 能把"这份数据长什么形状"和"取出里面的字段"合并成一步来写,在解析 agent 的动作指令、API 响应这类"先看形状再取值"的场景里可读性明显更好。**仓库里(`learning/` 目录下博士学长自己写的代码,不算 vendor 进来的 `official/repos/`)目前还没有真实使用 `match`-`case` 的代码**——这符合预期,`match`-`case` 是通用现代 Python 语法,不是 AI/ML 研究代码的专属写法,这里如实标注为示例性质,不编造仓库引用。

**从最笨的写法讲起:**

先看最基础的"单值匹配",和 `if/elif` 链完全等价:
```python
def http_status_text_dumb(code: int) -> str:
    if code == 200:
        return "OK"
    elif code == 404:
        return "Not Found"
    elif code == 500:
        return "Server Error"
    else:
        return "Unknown"

def http_status_text(code: int) -> str:
    match code:
        case 200:
            return "OK"
        case 404:
            return "Not Found"
        case 500:
            return "Server Error"
        case _:                     # 通配符,相当于 else
            return "Unknown"

for c in (200, 404, 500, 999):
    assert http_status_text_dumb(c) == http_status_text(c)
print("[OK] 单值匹配和 if/elif 完全等价")
```
单值匹配这个层面,`match`-`case` 只是换了个写法,没有解决新问题。真正的差别在解构——这是 `if/elif` 做不到、只能靠手动查 `len()` 和下标模拟的能力:
```python
# 笨办法: 用 if/elif 判断一个"命令" tuple,得手动 len() 判断形状,再用下标一个个取
def handle_command_dumb(cmd: tuple) -> str:
    if len(cmd) == 2 and cmd[0] == "move":
        return f"move {cmd[1]}"
    elif len(cmd) == 3 and cmd[0] == "goto":
        return f"goto ({cmd[1]}, {cmd[2]})"
    elif len(cmd) == 1 and cmd[0] == "stop":
        return "stop"
    else:
        return "unknown command"

# match-case: 模式本身就描述了"这份数据长什么形状",匹配成功的同时自动把子元素绑定成变量,
# 不需要另外写 len() 判断,也不需要用下标 cmd[1]/cmd[2] 去取值
def handle_command(cmd: tuple) -> str:
    match cmd:
        case ("move", direction):     # 长度为2,且第一个元素是 "move" -> direction 自动绑定第二个元素
            return f"move {direction}"
        case ("goto", x, y):           # 长度为3 -> x, y 自动解构绑定
            return f"goto ({x}, {y})"
        case ("stop",):                 # 长度为1的单元素 tuple
            return "stop"
        case _:
            return "unknown command"

for cmd in [("move", "north"), ("goto", 3, 4), ("stop",), ("jump",)]:
    assert handle_command_dumb(cmd) == handle_command(cmd)
print("[OK] 解构匹配的结果和手写 if/elif + len() 判断一致,但不用手动查长度、查下标")
```

**AI 研究代码里的真实例子:** 仓库内(`learning/` 目录,不算 `official/repos/`)暂未挖到 `match`-`case` 的真实使用——这是通用现代 Python 语法的示例性质讲解,不是编造的仓库引用。下面用一个贴近 agent 场景的例子演示它的典型用法:解析"工具调用结果"这种"先看类型标签,再按标签取不同字段"的数据:
```python
def render_observation(result: dict) -> str:
    match result:
        case {"status": "ok", "value": value}:
            return f"OK: {value}"
        case {"status": "error", "code": code, "message": msg}:
            return f"ERROR[{code}]: {msg}"
        case {"status": status}:
            return f"UNKNOWN STATUS: {status}"
        case _:
            return "MALFORMED RESULT"

assert render_observation({"status": "ok", "value": 42}) == "OK: 42"
assert render_observation({"status": "error", "code": 404, "message": "not found"}) == "ERROR[404]: not found"
assert render_observation({"status": "pending"}) == "UNKNOWN STATUS: pending"
assert render_observation({}) == "MALFORMED RESULT"
print("[OK] dict 模式匹配:按 status 字段分派,同时把 value/code/message 解构出来")
```

**可运行例子:** 守卫条件(`case pattern if condition`)——模式匹配上了,还可以再加一层普通的布尔判断:
```python
def classify(point: tuple[int, int]) -> str:
    match point:
        case (0, 0):
            return "origin"
        case (x, y) if x == y:          # 守卫条件: 结构匹配上了,还要满足 x == y 才算命中这条分支
            return "on diagonal"
        case (x, 0):
            return f"on x-axis at {x}"
        case (0, y):
            return f"on y-axis at {y}"
        case (x, y):
            return f"point({x},{y})"

assert classify((0, 0)) == "origin"
assert classify((3, 3)) == "on diagonal"
assert classify((5, 0)) == "on x-axis at 5"
assert classify((0, 7)) == "on y-axis at 7"
assert classify((2, 9)) == "point(2,9)"
print("[OK] 守卫条件 case ... if ... 工作正常")
```

**常见坑:**

**1. `case` 后面孤零零的一个小写标识符,是"捕获模式"(永远匹配,并把值绑定给这个名字),不是"和外层同名变量比较值"——这是最容易踩的坑。** 如果整个 `case` 就是一个裸标识符,Python 编译器甚至会在编译期直接报错(因为能证明它必然吞掉后面所有分支),但如果这个裸标识符是嵌在更大的结构模式(比如 tuple)里面,编译器就管不到了,陷阱会真实存在:
```python
# 孤零零的裸标识符: 编译期直接报错,这是 Python 的善意提醒
try:
    compile(
        "expected = 1\nmatch 2:\n    case expected:\n        pass\n    case _:\n        pass\n",
        "<test>", "exec",
    )
    raise AssertionError("应该 SyntaxError")
except SyntaxError as e:
    print("孤立裸标识符 case 触发编译期 SyntaxError:", e)

# 嵌在 tuple 模式里的裸标识符: 编译器管不到,陷阱真实存在
expected_dir = "north"          # 外层已有一个同名变量,值是 "north"

def handle_wrong(cmd: tuple) -> str:
    match cmd:
        case ("move", expected_dir):    # 陷阱: expected_dir 在这里是"捕获位置",不要求等于外层的 "north"
            return f"bound expected_dir={expected_dir!r}"
        case _:
            return "no match"

assert handle_wrong(("move", "north")) == "bound expected_dir='north'"
assert handle_wrong(("move", "south")) == "bound expected_dir='south'"   # "south" 也照样匹配上了!
print("嵌在结构模式里的裸标识符依然是捕获,不会要求等于外层同名变量的值")

# 正确写法: 用 guard 显式比较,才会被当成"值比较"而不是"捕获"
def handle_right(cmd: tuple) -> str:
    match cmd:
        case ("move", direction) if direction == expected_dir:
            return f"move {direction} (matches expected)"
        case _:
            return "no match"

assert handle_right(("move", "north")) == "move north (matches expected)"
assert handle_right(("move", "south")) == "no match"
print("[OK] 加 guard 之后,只有等于外层 expected_dir 的值才会匹配")
```

**2. `match`-`case` 没有其他语言 `switch` 常见的"贯穿"(fallthrough)行为,每个 `case` 匹配成功后自动结束,不会继续往下执行下一个 `case`——这一点其实是"坑更少"而不是"更容易踩坑",但从其他语言转过来的人容易习惯性地找 `break`,Python 里不需要写,写了也是语法错误(`case` 块里没有 `break` 这个关键字用法)。**

---

## 小结:这一批 5 个知识点解决的问题

| 知识点 | 解决的问题 |
|---|---|
| f-string 高级用法 | 格式说明符统一了精度/宽度/千分位的显示控制;`=` 调试写法避免"变量名手打两遍"不同步 |
| `str.join` | 先攒列表、最后一次性拼接,比循环里 `+=` 更快,也不用手动处理"多余分隔符"的边界情况 |
| `str.split`/`strip`/`partition` | 无参数 `split()`/固定分隔符 `split(sep)`/`partition()` 分别对应不同的解析需求,选错会在边界情况上出 bug |
| `pathlib.Path` | 路径是对象不是字符串,`/` 拼接 + `.exists()`/`.suffix` 等属性方法,天然跨平台 |
| `match`-`case` | 把"数据长什么形状"和"取出字段"合并成一步,`if/elif` + 手动下标做不到解构 |

---

*更新:2026-07-08*
