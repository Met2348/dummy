# 06 · 手把手实战:从零搭一个迷你日志分析 CLI

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 27 个"知识点",不计入"26 个知识点"的统计——和 [05 类](05-advanced-interview-depth.md)一样是正式知识点之外的追加内容,但风格完全不一样:05 号文件里,你是**旁观者**,跟着 5 条多级追问链案例把一条推理链条看一遍;这一篇里,你是**动手的人**——从一个空文件开始,一步步敲代码,每写一段就跑一次、看到真实效果,最后独立搭出一个完整能用的小工具。格式借鉴自 [dsa-deep-dive/21-build-a-mini-search-engine.md](../dsa-deep-dive/21-build-a-mini-search-engine.md)(仓库里"教程体"格式的试点文件,已验证过一轮)。
>
> 标题里的"CLI"指的是这个分析器的**核心逻辑**——真实命令行工具还需要读文件、解析 `sys.argv`,这些留到最后"可以怎么继续扩展"一节里只指方向,不在本文实现(遵守教程体一贯的约束:玩具规模、几秒内跑完、不读写真实文件、不访问网络——日志数据全部现场用 Python 字符串列表合成)。

## 为什么是"日志分析器"

不是要发明新知识点,是把三类你已经学过的知识点串成一个真实有用的东西——"读一批日志、算出统计、给出简报",是后端服务和训练脚本里天天出现的一小段代码,而且现实里的日志**从来不会**行行格式规整,总有几行是空的、截断的、字段错位的:

| 阶段 | 要让程序多会一件事 | 建立在哪个已有知识点之上 |
|------|------|------|
| 阶段 1 | 一行行"脏日志"喂进来,判断这一行到底能不能解析,解析不了就跳过而不是崩溃 | [03 类](03-containers-and-stdlib.md) 第 6 节 EAFP vs LBYL 哲学 |
| 阶段 2 | 把一整批原始行一次性冲洗成"干净记录",顺带知道扔了多少条 | [01 类](01-comprehensions-and-functional.md) 第 1 节列表推导式 + [python-advanced/02 类](../python-advanced/02-iterators-and-generators.md) 第 3 节生成器表达式惰性求值 |
| 阶段 3 | 把干净记录按日志级别分组统计 | [02 类](02-unpacking-and-iteration.md) 第 5.3 节提到但没展开的 `itertools.groupby` + [03 类](03-containers-and-stdlib.md) 第 1/2 节 `Counter`/`defaultdict` |
| 阶段 4 | 把前三步拼成一个能持续喂数据、随时吐简报的分析器 | 阶段 1-3 全部组装 |

每个阶段的代码都能独立运行(本文件用 [`_verify_md.py`](_verify_md.py) 校验——从 [dsa-deep-dive/_verify_md.py](../dsa-deep-dive/_verify_md.py) 原样拷贝过来,校验逻辑和系列本身无关——校验方式是把每个 ` ```python ` 代码块单独拎出来起一个新的 Python 子进程执行,块与块之间**不共享任何变量**,所以后面阶段用到前面阶段的代码时,会重新贴一遍,不是偷懒复制,是这套校验机制要求的)。

---

## 阶段 1:EAFP 单行解析——脏数据不"先体检",直接尝试再说

日志行的约定格式是 `"2026-07-24 10:00:01 INFO service started"`——日期、时间、级别、消息,用空格分开。**LBYL 思路**会先写一堆 `if` 检查这一行"看起来对不对"(字段数够不够、级别是不是合法值、日期是不是真实存在),检查全部通过才敢真正解析。**EAFP 思路**([03 类第 6 节](03-containers-and-stdlib.md)详细讲过这对哲学)反过来:直接按格式尝试解析,失败了就说明这行有问题,用 `try/except` 接住,不用事先猜"可能会哪里错"。

```python
from datetime import datetime

LOG_LEVELS = {"INFO", "WARNING", "ERROR"}

def parse_line(line):
    # EAFP: 不做任何"体检",直接按约定格式尝试拆分/解析;哪一步失败了,哪一步自然抛异常
    date_str, time_str, level, message = line.split(" ", 3)
    if level not in LOG_LEVELS:
        raise ValueError(f"unrecognized level: {level!r}")
    ts = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    return {"ts": ts, "level": level, "message": message}

good = parse_line("2026-07-24 10:00:01 INFO service started")
assert good == {"ts": datetime(2026, 7, 24, 10, 0, 1), "level": "INFO", "message": "service started"}
print("parsed ok:", good)

# 故意混入 5 种不同方式格式不规范的脏行
bad_lines = [
    "",                                                              # 空行
    "corrupted line without enough fields",                          # 字段数/格式完全对不上
    "2026-07-24 10:00:20 DEBUG this level does not exist",           # 级别不在 INFO/WARNING/ERROR 里
    "2026-99-99 10:00:21 ERROR invalid date",                        # 日期本身不合法(没有99月)
    "2026-07-24,10:00:22,INFO,comma separated not space separated",  # 分隔符用了逗号,不是空格
]
for bl in bad_lines:
    try:
        parse_line(bl)
        raise AssertionError(f"expected ValueError for line: {bl!r}")
    except ValueError as e:
        print(f"rejected ({e})")
```

真实跑出来的 5 条拒绝原因很值得读一遍——**不是每一条都按"看起来应该错在哪"报错**:

```text
rejected (not enough values to unpack (expected 4, got 1))
rejected (unrecognized level: 'without')
rejected (unrecognized level: 'DEBUG')
rejected (time data '2026-99-99 10:00:21' does not match format '%Y-%m-%d %H:%M:%S')
rejected (unrecognized level: 'not')
```

第 2 行和第 5 行是意料之外的:`"corrupted line without enough fields"` 这句话本身有 5 个单词,`split(" ", 3)` 能顺利切出 4 段(`"corrupted"`、`"line"`、`"without"`、`"enough fields"`),根本没触发"字段数不够";真正让它出局的是 `"without"` 这个词恰好落在了"级别"这个位置、又不是合法级别。逗号分隔的那一行同理——整行没有一个空格能让 `split(" ")` 切开前半段,于是 `"comma"`/`"separated"`/`"not"` 分别落进了 date/time/level 三个槽位,`"not"` 恰好也不是合法级别。**EAFP 的好处正体现在这里**:不管一行日志是以哪种具体方式"长歪的",只要最终解析步骤里任何一步失败,统一被同一个 `except ValueError` 接住——不需要事先枚举"脏数据可能长成什么样子"。

**这不是纯粹的风格选择,是真的能少做重复工作。** 如果坚持用 LBYL 风格,校验函数为了确认"日期真的合法",内部几乎没有干净写法可以避开 `try/except`(手写闰年判断、每月天数表,代价远大于直接尝试解析),而如果校验函数内部还是用了 `try/except`,那对每一条合法日志,日期就被解析了两遍——一遍在校验里,一遍在真正解析里。实测一下这个重复代价:

```python
from datetime import datetime

LOG_LEVELS = {"INFO", "WARNING", "ERROR"}
strptime_calls = 0

def counted_strptime(s):
    global strptime_calls
    strptime_calls += 1
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

# LBYL: 校验函数为了确认"日期合法",自己也不得不 try/except 一次
def is_valid_line_lbyl(line):
    parts = line.split(" ", 3)
    if len(parts) != 4:
        return False
    date_str, time_str, level, message = parts
    if level not in LOG_LEVELS:
        return False
    try:
        counted_strptime(f"{date_str} {time_str}")
    except ValueError:
        return False
    return True

def parse_line_lbyl(line):
    if not is_valid_line_lbyl(line):
        raise ValueError("malformed line")
    date_str, time_str, level, message = line.split(" ", 3)
    ts = counted_strptime(f"{date_str} {time_str}")   # 解析成功后,又解析了一遍日期
    return {"ts": ts, "level": level, "message": message}

def parse_line_eafp(line):
    date_str, time_str, level, message = line.split(" ", 3)
    if level not in LOG_LEVELS:
        raise ValueError(f"unrecognized level: {level!r}")
    ts = counted_strptime(f"{date_str} {time_str}")
    return {"ts": ts, "level": level, "message": message}

sample = "2026-07-24 10:00:01 INFO service started"

strptime_calls = 0
r_lbyl = parse_line_lbyl(sample)
calls_lbyl = strptime_calls

strptime_calls = 0
r_eafp = parse_line_eafp(sample)
calls_eafp = strptime_calls

print(f"LBYL 版本日期解析次数: {calls_lbyl}, EAFP 版本日期解析次数: {calls_eafp}")
assert r_lbyl == r_eafp                # 两者结果完全一样
assert calls_lbyl == 2                 # 校验一次 + 真正解析一次
assert calls_eafp == 1                 # 只尝试一次
```

这和 [03 类第 6 节](03-containers-and-stdlib.md) `lookup_lbyl`/`lookup_eafp` 那组对比是同一个道理,只是"重复的动作"从"字典查找两次"换成了"日期解析两次"——LBYL 版本的校验函数本质上已经在用 EAFP(内部的 `try/except`),只是把"尝试"这个动作从主逻辑挪到了校验函数里,反而白多算一次。

最后把"解析成功返回记录、解析失败返回 `None`"包一层,方便下一阶段批量处理——这一步让程序学会:**面对一行不知道好坏的日志,能给出一个"要么是干净记录、要么是 `None`"的确定性答案,而不是让异常直接炸穿调用方**:

```python
from datetime import datetime

LOG_LEVELS = {"INFO", "WARNING", "ERROR"}

def parse_line(line):
    date_str, time_str, level, message = line.split(" ", 3)
    if level not in LOG_LEVELS:
        raise ValueError(f"unrecognized level: {level!r}")
    ts = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    return {"ts": ts, "level": level, "message": message}

def try_parse_line(line):
    try:
        return parse_line(line)
    except ValueError:
        return None

assert try_parse_line("2026-07-24 10:00:01 INFO service started") == {
    "ts": datetime(2026, 7, 24, 10, 0, 1), "level": "INFO", "message": "service started"
}
assert try_parse_line("") is None
assert try_parse_line("corrupted line without enough fields") is None
assert try_parse_line("2026-99-99 10:00:21 ERROR invalid date") is None
print("try_parse_line: 好行给记录, 坏行给 None, 都验证过了")
```

---

## 阶段 2:推导式 + 生成器表达式——把一整批脏行冲洗成干净记录

有了单行解析器,接下来处理真正的一批日志。这批数据里特意混了 5 行阶段 1 验证过的脏数据,分布在中间各处,不是都堆在开头或结尾——更贴近真实日志文件里错误行随机出现的样子:

```python
from datetime import datetime

LOG_LEVELS = {"INFO", "WARNING", "ERROR"}

def parse_line(line):
    date_str, time_str, level, message = line.split(" ", 3)
    if level not in LOG_LEVELS:
        raise ValueError(f"unrecognized level: {level!r}")
    ts = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    return {"ts": ts, "level": level, "message": message}

def try_parse_line(line):
    try:
        return parse_line(line)
    except ValueError:
        return None

RAW_LOG_LINES = [
    "2026-07-24 10:00:00 INFO service started",
    "2026-07-24 10:00:01 INFO listening on port 8080",
    "2026-07-24 10:00:03 WARNING cache miss for key=user:42",
    "",
    "2026-07-24 10:00:05 ERROR connection refused",
    "2026-07-24 10:00:06 INFO retrying connection",
    "corrupted line without enough fields",
    "2026-07-24 10:00:07 INFO connection established",
    "2026-07-24 10:00:09 ERROR connection refused",
    "2026-07-24 10:00:10 WARNING high memory usage 82 percent",
    "2026-99-99 10:00:11 ERROR invalid date",
    "2026-07-24 10:00:12 INFO request handled in 12ms",
    "2026-07-24 10:00:13 ERROR connection refused",
    "2026-07-24 10:00:14 INFO request handled in 9ms",
    "2026-07-24 10:00:16 WARNING disk usage above threshold",
    "2026-07-24 10:00:20 DEBUG this level does not exist",
    "2026-07-24 10:00:17 ERROR unhandled exception in worker",
    "2026-07-24 10:00:18 INFO worker restarted",
    "2026-07-24,10:00:22,INFO,comma separated not space separated",
    "2026-07-24 10:00:19 INFO shutting down",
]

# 生成器表达式 (try_parse_line(line) for line in RAW_LOG_LINES) 负责"逐行尝试解析",不提前算完;
# 外层列表推导式负责"过滤掉 None、把剩下的收集成一份干净列表" —— 两层职责分开但写在一起,
# 语法上和 01 类第 1 节讲的嵌套推导式是同一套写法习惯
parsed = [r for r in (try_parse_line(line) for line in RAW_LOG_LINES) if r is not None]

print(f"原始行数: {len(RAW_LOG_LINES)}, 解析成功: {len(parsed)}, 丢弃: {len(RAW_LOG_LINES) - len(parsed)}")
assert len(RAW_LOG_LINES) == 20
assert len(parsed) == 15          # 20 行里有 5 行是阶段1验证过的脏数据,应该被冲洗掉
assert len(RAW_LOG_LINES) - len(parsed) == 5
assert all(r["level"] in LOG_LEVELS for r in parsed)   # 冲洗完的记录,级别一定是合法的三者之一
```

如果只是想知道"这批日志里有几条 ERROR",不需要真的把全部记录物化成一个列表——直接用生成器表达式喂给 `sum()`,一条记录处理完就丢,内存里从来不存在一个完整的中间列表。这是 [python-advanced/02 类第 3 节](../python-advanced/02-iterators-and-generators.md)"生成器表达式 vs 列表推导:惰性求值"在这个场景下的真实应用,不是重复讲一遍那一节,而是用一个新场景验证同一个结论:

```python
from datetime import datetime

LOG_LEVELS = {"INFO", "WARNING", "ERROR"}

def parse_line(line):
    date_str, time_str, level, message = line.split(" ", 3)
    if level not in LOG_LEVELS:
        raise ValueError(f"unrecognized level: {level!r}")
    ts = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    return {"ts": ts, "level": level, "message": message}

def try_parse_line(line):
    try:
        return parse_line(line)
    except ValueError:
        return None

RAW_LOG_LINES = [
    "2026-07-24 10:00:00 INFO service started",
    "2026-07-24 10:00:03 WARNING cache miss for key=user:42",
    "",
    "2026-07-24 10:00:05 ERROR connection refused",
    "corrupted line without enough fields",
    "2026-07-24 10:00:09 ERROR connection refused",
]

# 只关心计数,不需要一份完整的中间列表 —— 用海象运算符 (01 类第 6 节) 顺便避免同一行解析两遍
error_count_lazy = sum(
    1 for line in RAW_LOG_LINES
    if (r := try_parse_line(line)) is not None and r["level"] == "ERROR"
)
print("只用生成器表达式数出的 ERROR 条数:", error_count_lazy)
assert error_count_lazy == 2

# 交叉验证: 先物化成完整列表, 再数一遍, 结果必须一致
parsed_all = [r for r in (try_parse_line(l) for l in RAW_LOG_LINES) if r is not None]
error_count_materialized = sum(1 for r in parsed_all if r["level"] == "ERROR")
assert error_count_materialized == error_count_lazy
print("两种数法结果一致:", error_count_lazy)
```

---

## 阶段 3:按级别分组统计——见识 `itertools.groupby` 的真实坑,再用 `Counter`/`defaultdict` 正确地做

[02 类第 5.3 节](02-unpacking-and-iteration.md)提到过 `itertools.groupby`"按连续相同的 key 分组",但当时只是提了一句"仓库内暂时没有直接调用的真实例子,这里只是提一下它们的存在,不展开讲"。现在真的用一次,会撞上一个新手几乎必踩的坑:`groupby` 的"分组"**只认连续出现**的相同 key,不是"把所有相同 key 的元素收集到一起"——如果同一个级别在日志里分散出现在好几个不相邻的位置(现实里几乎总是这样,INFO/WARNING/ERROR 按时间交替出现),直接对着按时间顺序排列的记录跑 `groupby`,会得到一堆破碎的小组,而不是三个干净的分组:

```python
import itertools
from datetime import datetime

LOG_LEVELS = {"INFO", "WARNING", "ERROR"}

def parse_line(line):
    date_str, time_str, level, message = line.split(" ", 3)
    if level not in LOG_LEVELS:
        raise ValueError(f"unrecognized level: {level!r}")
    ts = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    return {"ts": ts, "level": level, "message": message}

def try_parse_line(line):
    try:
        return parse_line(line)
    except ValueError:
        return None

RAW_LOG_LINES = [
    "2026-07-24 10:00:00 INFO service started",
    "2026-07-24 10:00:01 INFO listening on port 8080",
    "2026-07-24 10:00:03 WARNING cache miss for key=user:42",
    "",
    "2026-07-24 10:00:05 ERROR connection refused",
    "2026-07-24 10:00:06 INFO retrying connection",
    "corrupted line without enough fields",
    "2026-07-24 10:00:07 INFO connection established",
    "2026-07-24 10:00:09 ERROR connection refused",
    "2026-07-24 10:00:10 WARNING high memory usage 82 percent",
    "2026-99-99 10:00:11 ERROR invalid date",
    "2026-07-24 10:00:12 INFO request handled in 12ms",
    "2026-07-24 10:00:13 ERROR connection refused",
    "2026-07-24 10:00:14 INFO request handled in 9ms",
    "2026-07-24 10:00:16 WARNING disk usage above threshold",
    "2026-07-24 10:00:20 DEBUG this level does not exist",
    "2026-07-24 10:00:17 ERROR unhandled exception in worker",
    "2026-07-24 10:00:18 INFO worker restarted",
    "2026-07-24,10:00:22,INFO,comma separated not space separated",
    "2026-07-24 10:00:19 INFO shutting down",
]

parsed = [r for r in (try_parse_line(line) for line in RAW_LOG_LINES) if r is not None]

# 天真的写法: 直接对着"按时间顺序排列"的记录跑 groupby
groups_naive = [(level, list(g)) for level, g in itertools.groupby(parsed, key=lambda r: r["level"])]
group_sizes = [len(items) for _, items in groups_naive]

print("naive groupby 分出的组数:", len(groups_naive))
print("每组大小:", group_sizes)

assert sum(group_sizes) == 15                    # 15 条记录一条没少, 只是被拆散了
assert len(groups_naive) == 12                   # 本机实测: 15 条记录被拆成了 12 个小组
assert len(groups_naive) > len(LOG_LEVELS)        # 这才是真正违反直觉的地方: 组数比"级别种类数(3)"还多得多
```

本机真实跑出来是 **12 组**,不是期待中的 3 组——`INFO` 这个级别在结果里前前后后出现了 6 次独立的小组(`INFO x2`、`INFO x2`、`INFO x1`、`INFO x1`、`INFO x1`、`INFO x2` 分散在不同位置),因为日志本来就是按时间顺序、级别交替出现的,`groupby` 只要一遇到"这一条和上一条 key 不一样"就会立刻切开一个新组,哪怕这个 key 之前已经出现过。**这不是 bug,是 `itertools.groupby` 文档里明确写清楚的行为**,但不看清楚这一点、指望它像 `Counter`/`defaultdict` 那样"自动把所有同类收集到一起",就会拿到一份看起来毫无意义的破碎统计。

修复方式有两条路,分别对应"想继续用 `groupby`"和"换一个更适合这个场景的工具":

```python
import itertools
from collections import Counter, defaultdict
from datetime import datetime

LOG_LEVELS = {"INFO", "WARNING", "ERROR"}

def parse_line(line):
    date_str, time_str, level, message = line.split(" ", 3)
    if level not in LOG_LEVELS:
        raise ValueError(f"unrecognized level: {level!r}")
    ts = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    return {"ts": ts, "level": level, "message": message}

def try_parse_line(line):
    try:
        return parse_line(line)
    except ValueError:
        return None

RAW_LOG_LINES = [
    "2026-07-24 10:00:00 INFO service started",
    "2026-07-24 10:00:01 INFO listening on port 8080",
    "2026-07-24 10:00:03 WARNING cache miss for key=user:42",
    "",
    "2026-07-24 10:00:05 ERROR connection refused",
    "2026-07-24 10:00:06 INFO retrying connection",
    "corrupted line without enough fields",
    "2026-07-24 10:00:07 INFO connection established",
    "2026-07-24 10:00:09 ERROR connection refused",
    "2026-07-24 10:00:10 WARNING high memory usage 82 percent",
    "2026-99-99 10:00:11 ERROR invalid date",
    "2026-07-24 10:00:12 INFO request handled in 12ms",
    "2026-07-24 10:00:13 ERROR connection refused",
    "2026-07-24 10:00:14 INFO request handled in 9ms",
    "2026-07-24 10:00:16 WARNING disk usage above threshold",
    "2026-07-24 10:00:20 DEBUG this level does not exist",
    "2026-07-24 10:00:17 ERROR unhandled exception in worker",
    "2026-07-24 10:00:18 INFO worker restarted",
    "2026-07-24,10:00:22,INFO,comma separated not space separated",
    "2026-07-24 10:00:19 INFO shutting down",
]

parsed = [r for r in (try_parse_line(line) for line in RAW_LOG_LINES) if r is not None]

# 修法一: 想继续用 groupby, 先按 key 排序, 让相同级别的记录真的挨在一起
sorted_by_level = sorted(parsed, key=lambda r: r["level"])
groups_sorted = [(level, list(g)) for level, g in itertools.groupby(sorted_by_level, key=lambda r: r["level"])]
assert len(groups_sorted) == 3
assert [level for level, _ in groups_sorted] == ["ERROR", "INFO", "WARNING"]   # sorted() 按字母序: E < I < W
assert [len(items) for _, items in groups_sorted] == [4, 8, 3]

# 修法二: 换 Counter 只要计数, 不需要排序, 一次遍历搞定, 不关心记录原本的顺序
level_counts = Counter(r["level"] for r in parsed)
assert level_counts == Counter({"INFO": 8, "WARNING": 3, "ERROR": 4})

# 修法三: 换 defaultdict(list), 不仅要计数, 还要实际收集每个级别下的消息内容
by_level = defaultdict(list)
for r in parsed:
    by_level[r["level"]].append(r["message"])
assert {level: len(msgs) for level, msgs in by_level.items()} == {"INFO": 8, "WARNING": 3, "ERROR": 4}

# 三条完全独立的路径 (排序后groupby / Counter / defaultdict) 应该互相印证, 得到同一组数字
sizes_from_sorted_groupby = {level: len(items) for level, items in groups_sorted}
sizes_from_defaultdict = {level: len(msgs) for level, msgs in by_level.items()}
assert sizes_from_sorted_groupby == dict(level_counts) == sizes_from_defaultdict
print("三种方法互相印证:", dict(level_counts))
```

**这个坑对应的判断标准很直接:如果分组前不能保证同一个 key 的元素本来就挨在一起(或者你愿意先花一次 `sorted()` 的代价把它们排到一起),就不要用 `groupby`——`Counter`/`defaultdict` 天生不关心顺序,是"按 key 分组统计"这个需求更安全的默认选项;`groupby` 真正适合的场景是数据本来就已经有序(比如按时间排序后按小时聚合),这时"只认连续"不是缺点,反而是它故意设计成 O(n) 单遍扫描、不需要额外哈希表的原因。**

---

## 阶段 4:组装成 `MiniLogAnalyzer`——一个能持续喂数据、随时吐简报的小工具

真实的日志分析场景很少是"一次性给你全部日志",更常见的是日志分批到达(不同批次的文件、滚动的日志流)。把前三阶段拼进一个类,`ingest()` 方法可以被调用任意多次、每次喂一批新行,内部状态(`Counter`/`defaultdict`)持续累积;`report()` 方法随时可以在任意时刻被调用,吐出截至目前的统计简报:

```python
from collections import Counter, defaultdict
from datetime import datetime

LOG_LEVELS = {"INFO", "WARNING", "ERROR"}

def parse_line(line):
    date_str, time_str, level, message = line.split(" ", 3)
    if level not in LOG_LEVELS:
        raise ValueError(f"unrecognized level: {level!r}")
    ts = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    return {"ts": ts, "level": level, "message": message}

def try_parse_line(line):
    try:
        return parse_line(line)
    except ValueError:
        return None

class MiniLogAnalyzer:
    def __init__(self):
        self.total_lines = 0
        self.dropped = 0
        self.level_counts = Counter()             # 阶段3 修法二
        self.messages_by_level = defaultdict(list)  # 阶段3 修法三
        self.timestamps = []

    def ingest(self, raw_lines):
        # 阶段1 的 EAFP 单行解析 + 阶段2 "干净/丢弃"两条路径的分流逻辑
        for line in raw_lines:
            self.total_lines += 1
            record = try_parse_line(line)
            if record is None:
                self.dropped += 1
                continue
            self.level_counts[record["level"]] += 1
            self.messages_by_level[record["level"]].append(record["message"])
            self.timestamps.append(record["ts"])

    def report(self):
        lines = [
            f"total lines: {self.total_lines}",
            f"parsed ok: {self.total_lines - self.dropped}",
            f"dropped (malformed): {self.dropped}",
        ]
        if self.timestamps:
            lines.append(f"time range: {min(self.timestamps)} to {max(self.timestamps)}")
        lines.append("by level:")
        for level in ("INFO", "WARNING", "ERROR"):
            lines.append(f"  {level}: {self.level_counts[level]}")
        # most_common(1): 03 类第 1 节讲过的 Counter 方法, 这里用来挖"出现次数最多的那条 ERROR 消息"
        error_msg_counts = Counter(self.messages_by_level["ERROR"])
        top = error_msg_counts.most_common(1)
        if top:
            msg, cnt = top[0]
            lines.append(f'top recurring ERROR: "{msg}" x{cnt}')
        return "\n".join(lines)

# 模拟日志分两批到达 —— 和阶段2/3用的是同一份20行数据, 只是拆成两次 ingest()
BATCH_1 = [
    "2026-07-24 10:00:00 INFO service started",
    "2026-07-24 10:00:01 INFO listening on port 8080",
    "2026-07-24 10:00:03 WARNING cache miss for key=user:42",
    "",
    "2026-07-24 10:00:05 ERROR connection refused",
    "2026-07-24 10:00:06 INFO retrying connection",
    "corrupted line without enough fields",
    "2026-07-24 10:00:07 INFO connection established",
    "2026-07-24 10:00:09 ERROR connection refused",
    "2026-07-24 10:00:10 WARNING high memory usage 82 percent",
]
BATCH_2 = [
    "2026-99-99 10:00:11 ERROR invalid date",
    "2026-07-24 10:00:12 INFO request handled in 12ms",
    "2026-07-24 10:00:13 ERROR connection refused",
    "2026-07-24 10:00:14 INFO request handled in 9ms",
    "2026-07-24 10:00:16 WARNING disk usage above threshold",
    "2026-07-24 10:00:20 DEBUG this level does not exist",
    "2026-07-24 10:00:17 ERROR unhandled exception in worker",
    "2026-07-24 10:00:18 INFO worker restarted",
    "2026-07-24,10:00:22,INFO,comma separated not space separated",
    "2026-07-24 10:00:19 INFO shutting down",
]

analyzer = MiniLogAnalyzer()
analyzer.ingest(BATCH_1)
analyzer.ingest(BATCH_2)     # 第二次调用: 状态在上一次的基础上继续累积, 不是重新开始

report_text = analyzer.report()
print(report_text)

assert analyzer.total_lines == 20
assert analyzer.dropped == 5
assert dict(analyzer.level_counts) == {"INFO": 8, "WARNING": 3, "ERROR": 4}
assert "total lines: 20" in report_text
assert "parsed ok: 15" in report_text
assert "dropped (malformed): 5" in report_text
assert "time range: 2026-07-24 10:00:00 to 2026-07-24 10:00:19" in report_text
assert "  INFO: 8" in report_text
assert "  WARNING: 3" in report_text
assert "  ERROR: 4" in report_text
assert 'top recurring ERROR: "connection refused" x3' in report_text
```

到这里,`MiniLogAnalyzer` 已经是一个真实能用的小工具:喂给它任意批次的原始日志行(不管是一次性喂完还是分批持续喂),它能自动过滤掉格式不规范的脏数据、按级别统计数量、挖出出现次数最多的错误消息,吐出一份人能直接读的简报——三个能力分别对应 EAFP、推导式/生成器表达式、`Counter`/`defaultdict`(以及 `groupby` 的一次真实踩坑),拼起来是很多真实日志监控工具(不管是自己写的小脚本,还是 ELK/Datadog 这类成熟系统)最核心的那一小块逻辑。

## 可以怎么继续扩展(只指方向,不在本文实现)

- **真实数据来源**:现在的日志是内存里现成的字符串列表,真实场景要从文件或标准输入**逐行**读——用 `pathlib.Path.open()` 或 `sys.stdin` 迭代,配合阶段 2 讲过的生成器表达式思路,可以处理比内存大得多的日志文件而不用一次性全部读进来,这正是 [python-advanced/02 类](../python-advanced/02-iterators-and-generators.md)"惰性求值"的真实用武之地。
- **更灵活的日志格式**:现在只认死板的"日期 时间 级别 消息"四段格式,真实系统的日志格式五花八门(字段顺序不同、分隔符不同、偶尔多几个空格)——更稳健的做法是用 `re` 正则表达式做匹配,能容忍的变体比 `split(" ", 3)` 多得多。
- **按时间窗口聚合**:比如"每分钟的错误数"这种时间序列统计,思路是先按 `ts` 排序、再按"取整到分钟"这个 key 做分组——这时候 `itertools.groupby`"只认连续"的特性反而不是坑,是排过序之后的正确用法(呼应阶段 3 最后给出的判断标准)。
- **多行日志/堆栈跟踪合并**:真实系统里一次异常经常横跨好几行(Python 的 `Traceback` 就是典型例子),本文"一行对应一条记录"的模型处理不了这种情况,需要引入"看到不匹配约定格式的行,就并入上一条记录"这类归并逻辑,不再是纯粹的逐行独立解析。

## 这篇教程展示的方法论

任何一条已完成的深挖系列,都可以用同样的模式产出"教程体"内容:挑几个关联的知识点 → 设计一个真实有用、读者一看就懂价值的小工具 → 分阶段增量实现,每一步都跑起来看到真实效果。这一篇额外展示了一件事:**教程体不是只用来演示"这几个知识点搭在一起很好用",也可以老老实实展示"一个知识点(`itertools.groupby`)用不对场景会产出违反直觉的错误结果",然后现场纠正**——这和 [dsa-deep-dive/21 类](../dsa-deep-dive/21-build-a-mini-search-engine.md)对元组打平排序结果的处理是同一种纪律:不回避不完美的真实断言输出,把"为什么会这样、怎么修"讲清楚,比假装一切都完美更有教学价值。

---

*创建:2026-07-24*
