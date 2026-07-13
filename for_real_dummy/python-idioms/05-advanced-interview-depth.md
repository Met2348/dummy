# 05 · 进阶深度追加:5 个真实二面级别的多级追问链案例

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是知识点,不计入统计——它和 [dsa-deep-dive/20-advanced-interview-depth.md](../dsa-deep-dive/20-advanced-interview-depth.md)、[peft-deep-dive/05-advanced-interview-depth.md](../peft-deep-dive/05-advanced-interview-depth.md) 是同一挂:方法论 + 案例,不是知识点列表。

## 为什么需要这篇追加内容

`01-04` 全部完成并自查通过之后,用户转达了一位有经验从业者的反馈:现有材料没有达到 **2026 年大厂技术二面** 的深度。`dsa-deep-dive/20-advanced-interview-depth.md`、`peft-deep-dive/05-advanced-interview-depth.md` 等系列已经基于一次真实调研(三路 WebSearch:检索中国大厂面经、西方大厂面经、面试官视角的元讨论)落地验证过一套格式并沉淀进项目 memory,核心发现是:真实的追问不是"正确性 → 复杂度 → 能不能优化"这一条线性链,而是至少沿着 **5 条独立轴线** 展开,并且经常在同一道题里综合出现:

1. **规模递增轴**——数据/场景规模从小到大跳跃式增长,每次跳跃都让原方案失效,逼出下一层方案。
2. **工程约束递增轴(并发/分布式)**——单机/单线程正确,不等于并发场景下还正确;这条轴线在中国/西方大厂调研里独立收敛出几乎一致的结构,是三角验证最强的发现。
3. **方案批判迭代轴**——面试官不深挖单一方案的复杂度,而是连续指出该方案的具体工程缺陷逼你换方案,一题换 3-4 个技术方案是真实案例。
4. **决策依据追问轴("为什么选这个不选那个")**——不纠错,只逼问选择依据,而且要能进一步区分"这是当前配置凑不出解"还是"这类方法在任何配置下都凑不出解"。
5. **真实性验证轴**——把候选人简历/回答里抽象的表述("做了优化""提升了 N 倍")追问压向具体数字,鉴别是真做过还是只读过资料。

面试官视角的元规律:一个 55 分钟轮次里,候选人给出初版方案通常只占 15 分钟,剩下 40 分钟全是追问——这提示"追问链"内容的权重应该远高于普通知识点讲解。

`python-idioms` 系列(`01-04`)覆盖推导式与函数式基础、解包与迭代、容器与标准库、字符串与现代语法共 26 个知识点,风格是"表达习惯"而不是算法/数据结构本身——这意味着这批案例不能照搬 dsa-deep-dive 的算法批判范式,而是要从"写法选择"本身里挖出真正的工程深度:同一个 EAFP/LBYL 的取舍,原来只在理论上讨论过 TOCTOU 竞态,从没有真的用两个线程把它做出来过;同一个"推导式更清爽"的结论,没有验证过它在数据规模变大之后的内存代价;同一个"namedtuple 比 dataclass 省内存"的结论,没有追问过这个结论成立的前提条件。下面 5 个案例,每个都明确标注建立在哪个已有知识点之上,尝试把这批"写法选择"往下再压一层,压到"能不能用真实的并发/内存/性能数字回答'为什么'"这个深度。

本篇选了 **5 个案例**,组织原则是每个案例明确挂一条主轴线(部分案例会自然带出第二条轴线,在文末表格里如实标注,不强行只挂一条):

- **案例 1**(工程约束递增轴):EAFP vs LBYL 到 TOCTOU 竞态的真实复现,建立在 [03 类知识点 6](03-containers-and-stdlib.md) 之上。
- **案例 2**(规模递增轴 + 决策依据追问轴):推导式/生成器表达式在数据规模变大后的内存代价与复用取舍,建立在 [01 类知识点 7](01-comprehensions-and-functional.md) 之上。
- **案例 3**(决策依据追问轴):namedtuple vs dataclass,"更省内存"成立的前提、可哈希性、可变性三条约束的取舍,建立在 [03 类知识点 3](03-containers-and-stdlib.md) 之上。
- **案例 4**(真实性验证轴):"改成 pythonic 写法后快了很多倍"这句话拆开验证,哪些环节是数量级差异、哪些环节只是好看,建立在 [01 类知识点 1](01-comprehensions-and-functional.md)、[03 类知识点 2/4](03-containers-and-stdlib.md)、[04 类知识点 2](04-strings-and-modern-syntax.md) 之上。
- **案例 5**(方案批判迭代轴 + 决策依据追问轴):`sorted`/`key=` 的多级排序技巧到哪里失效,从两趟排序顺序搞反的真实 bug 到 `functools.cmp_to_key` 的必要性与代价,建立在 [03 类知识点 7](03-containers-and-stdlib.md) 之上。

**范围声明:** 这是方法论范例,不是把 26 个知识点全部重写一遍。每个案例都要求读者能看到"同样的追问方式,怎么套到任何一个已经掌握的知识点上"——读完之后,应该能自己对着 01-04 里任何一个没在这里出现的知识点(比如 `zip`、`itertools.pairwise`、`pathlib.Path`、`match`-`case`),现场把这几条轴线走一遍练习,而不是指望这篇文档穷举所有可能的追问。

---

## 案例 1:EAFP vs LBYL——从"理论上有竞态窗口"到用两个线程真的把它复现出来(工程约束递增轴)

建立在 [03-containers-and-stdlib.md 第 6 节](03-containers-and-stdlib.md) EAFP vs LBYL 之上——那一节已经引用了 Python 官方 glossary 原文和仓库 `tool-use-mcp` 讲义里的 TOCTOU 讨论,也用"文件被 `os.remove` 删除后再读"复现过 LBYL 会失败、EAFP 能优雅处理这件事,但那个复现是**顺序执行**的(先删除,再读),不是两个线程真的在"检查"和"使用"之间的窗口期里赛跑。面试官会当场指出这个差距,逼你把"理论上有竞态窗口"变成"真的用两个线程把它复现出来"。

**追问链条完整还原:**

- **Q(基础,03 类已覆盖):** "EAFP 和 LBYL,你会选哪个,为什么?" —— 期望候选人答出:EAFP 把检查和使用合并成一次操作,LBYL 是两步,中间有一个时间窗口;性能上 EAFP 通常省一次查找;安全上 EAFP 天然不怕 TOCTOU 竞态。
- **追问 1(不满足于背出"竞态"这个词):** "'中间有一个时间窗口'——这个窗口具体有多长?窗口期里如果真的有另一个线程把文件删了,LBYL 的代码会发生什么,你能现场写出来吗?" —— 期望候选人意识到"理论上存在"和"真实复现"是两回事,03 类已有的例子是顺序执行的 `os.remove` 后再读,不是真正的并发竞态。
- **追问 2(逼出确定性复现,不是"跑几次看运气"):** "用两个线程,不要用 `time.sleep` 去猜时机,怎么保证检查和删除**一定**按你想要的顺序交错?" —— 期望候选人想到用 `threading.Event` 在"检查通过"和"真正使用"之间插入一个精确的暂停点,让另一个线程在这个暂停期间执行删除,顺序由 `Event` 保证,不依赖操作系统调度的运气。
- **深挖追问(把"文件"泛化成"任何共享可变状态"):** "如果不是文件,是一个共享的 `dict`(比如一份内存里的 session 缓存),`if key in cache: return cache[key]` 这种写法,是不是也有同样的问题?这和文件场景是同一个 bug 还是两个不同的 bug?" —— 期望候选人指出这是**同一类** TOCTOU 问题在不同数据结构上的重演:`os.path.exists` + `open` 和 `key in mapping` + `mapping[key]` 都是"检查"和"使用"分成两步、中间留了窗口,03 类引用的 Python 官方 glossary 原文举的正是 `dict` 这个例子("can fail if another thread removes key from mapping after the test, but before the lookup")——这里要求候选人不能只会在"文件"这一个场景下复述答案,得能把同一个机制迁移到一个新的数据结构上,现场验证。

**可运行例子(1/2):文件 TOCTOU——用 `threading.Event` 精确卡在"检查通过、还没真正 open"这一点**

```python
import os
import tempfile
import threading

def eafp_read(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<missing>"

def race_lbyl(path):
    """把 LBYL 的逻辑手动展开: 检查(os.path.exists) 和使用(open) 之间插入一个精确的
    Event 同步点, 在暂停期间由主线程删除文件, 制造真实的 TOCTOU 窗口——不是凭运气
    等一个线程调度窗口, 顺序由 Event 严格保证。"""
    checked_event = threading.Event()
    deleted_event = threading.Event()
    outcome = {}

    def reader():
        if os.path.exists(path):                  # LBYL 的检查, 这一刻文件确实存在
            checked_event.set()
            deleted_event.wait()                    # 精确卡在"检查通过、还没真正 open"这个点
            try:
                with open(path, "r", encoding="utf-8") as f:
                    outcome["crashed"], outcome["result"] = False, f.read()
            except FileNotFoundError:
                outcome["crashed"], outcome["result"] = True, None
        else:
            outcome["crashed"], outcome["result"] = False, None

    t = threading.Thread(target=reader)
    t.start()
    checked_event.wait()        # 等 reader 线程完成"检查"这一步, 确认文件当时确实存在
    os.remove(path)              # 检查通过之后、真正 open 之前, 主线程把文件删了
    deleted_event.set()
    t.join()
    return outcome

tmp_dir = tempfile.mkdtemp()
path = os.path.join(tmp_dir, "config.txt")
with open(path, "w", encoding="utf-8") as f:
    f.write("real content")

outcome = race_lbyl(path)
os.rmdir(tmp_dir)

assert outcome["crashed"] is True, "LBYL 的检查通过之后, 文件被并发删除, 应该在 open 处真实崩溃"

# EAFP 对照: 文件在 eafp_read 被调用前就已经不存在了——但因为 EAFP 的"检查"和"使用"是
# 同一步(open 本身), 从来不存在一个"已确认存在但其实已经失效"的中间态可以被竞态利用
tmp_dir2 = tempfile.mkdtemp()
path2 = os.path.join(tmp_dir2, "config2.txt")
with open(path2, "w", encoding="utf-8") as f:
    f.write("real content")
os.remove(path2)
eafp_result = eafp_read(path2)
os.rmdir(tmp_dir2)

assert eafp_result == "<missing>", "EAFP 应该优雅处理, 不会抛出未捕获异常"

print(f"OK: LBYL 在'检查通过'之后的窗口期被并发删除文件, 真实崩溃(crashed={outcome['crashed']}); "
      f"EAFP 没有'检查通过但其实已失效'这个中间态可以被竞态利用, 始终优雅返回 {eafp_result!r}。")
```

**可运行例子(2/2):同一个机制迁移到 `dict`——官方 glossary 原文举的例子, 现场做成真实可复现的竞态**

```python
import threading

class SessionCache:
    """LBYL 风格: 先用 `in` 检查 key 是否存在, 再取值——这正是 Python 官方 glossary
    举的 TOCTOU 例子原文: 'can fail if another thread removes key from mapping
    after the test, but before the lookup'。"""
    def __init__(self):
        self.store = {}

    def get_lbyl(self, key, checked_event=None, deleted_event=None):
        if key in self.store:
            if checked_event is not None:
                checked_event.set()
                deleted_event.wait()
            return self.store[key]        # 如果 key 在这个窗口期被删了, 这里会真实 KeyError
        return None

    def get_eafp(self, key):
        try:
            return self.store[key]
        except KeyError:
            return None

cache = SessionCache()
cache.store["session_token"] = "abc123"

checked_event, deleted_event = threading.Event(), threading.Event()
outcome = {}

def reader():
    try:
        outcome["result"] = cache.get_lbyl("session_token", checked_event, deleted_event)
        outcome["crashed"] = False
    except KeyError:
        outcome["crashed"], outcome["result"] = True, None

t = threading.Thread(target=reader)
t.start()
checked_event.wait()
del cache.store["session_token"]     # 检查通过之后、取值之前, 另一个线程把 key 删了(比如 session 过期被清理)
deleted_event.set()
t.join()

assert outcome["crashed"] is True, "`in` 检查通过后 key 被并发删除, get_lbyl 应该真实 KeyError"

cache2 = SessionCache()
cache2.store["session_token"] = "abc123"
del cache2.store["session_token"]
eafp_outcome = cache2.get_eafp("session_token")
assert eafp_outcome is None

print(f"OK: dict 版 TOCTOU——LBYL 的 `in` 检查和取值之间被并发删除 key, 真实 KeyError(crashed={outcome['crashed']}); "
      f"EAFP 的 try/except 把'检查'和'取值'合并成一次原子操作, 优雅返回 {eafp_outcome!r}。"
      f"和例子(1/2)的文件场景是同一类 bug, 只是换了一个数据结构。")
```

**常见坑:** 把"先 `os.remove` 再读一次"这种顺序执行的模拟当成"复现了竞态"——这只验证了 EAFP/LBYL 在文件已经不存在时各自的行为,没有验证"检查通过之后、使用之前"这个真正危险的窗口;只会在"文件"这一个场景下复述 TOCTOU,换成 dict/共享变量之类的其它可变状态就想不到是同一个问题;加了 `try/except` 就以为已经"修好"了竞态,却忘了检查 `except` 范围是不是精确捕获了预期的异常——如果对文件的 `except` 写成裸 `except:`,除了 `FileNotFoundError`,连"权限不足"这类完全不同的问题也会被一起吞掉,这是 03 类"常见坑"已经强调过的另一条纪律。

---

## 案例 2:推导式还是生成器表达式——文件小的时候看不出差别,大了呢(规模递增轴 + 决策依据追问轴)

建立在 [01-comprehensions-and-functional.md 第 7 节](01-comprehensions-and-functional.md) one-liner 取舍之上——那一节的判断标准第 3 条提到"一行长到要横向滚动就该拆",但完全没有涉及"选列表推导式还是生成器表达式"这个同样常见、后果却完全不同的取舍;[python-advanced/02-iterators-and-generators.md](../python-advanced/02-iterators-and-generators.md) 已经用 `sys.getsizeof` 讲过两者的内存差异,但那是对着一个已经在内存里的小列表做的对比,没有放到"文件大小不可控"这个真实场景里推演过。

**追问链条完整还原:**

- **Q(基础,01 类已覆盖):** "写一个函数,从日志文件里挑出所有包含 `ERROR` 的行,返回给调用方。" —— 期望候选人写出 `[line for line in open(path) if "ERROR" in line]` 这样的一行推导式,理由是"一次过滤、一次转换,信息密度刚好"(01 类判断标准第 4 条)。
- **追问 1(规模递增,先纠正一个常见的错误归因):** "如果这份日志有 100GB,你这行代码会不会把整个文件读进内存?" —— 期望候选人先纠正问题本身的预设:文件对象自己的迭代协议本来就是逐行读的,`for line in open(path)` 这一步不会一次性加载整个文件;**真正的内存风险不在"读文件"这一步,在"把所有匹配行都装进同一个 list"这一步**——如果匹配率高、文件又大,这个 list 本身就可能是几十万甚至上百万个字符串对象。
- **追问 2(逼出真实数字,不能只停留在"应该会更省内存"这句话):** "你说生成器表达式更省内存,具体省多少?能说出一个大概的数量级吗?" —— 期望候选人现场用 `tracemalloc` 实测,而不是空口说"理论上更省"。
- **追问 3(决策依据追问轴,考察是不是"生成器无脑更好"的机械结论):** "那是不是所有场景都应该把推导式换成生成器表达式?" —— 期望候选人指出:如果后续需要对这批结果**再遍历第二次**(比如先统计一共匹配了多少行,再打印前几行内容),生成器表达式消费一次就耗尽,第二次拿到的是空的——这不是报错,是安静地给出错误结果;这时候就应该老老实实用 list,多花的内存是"复用"这个需求本身带来的必要代价,不是"忘记优化"。
- **深挖追问(把决策标准说清楚):** "所以你怎么判断一个场景该用哪个?" —— 期望候选人给出一个可操作的标准:只需要**从头到尾流过一遍**(比如求和、计数、写到另一个文件)用生成器表达式;需要 `len()`、随机访问、或者**不止一次遍历**,就必须用 list——这不是"生成器更高级"和"list 更笨"的问题,是两种不同消费模式各自唯一正确的选择。

**可运行例子(1/2):`tracemalloc` 真实测量峰值内存——不是断言"理论上更省",是量出具体字节数**

```python
import os
import tempfile
import tracemalloc

def make_log_file(path, n_lines):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_lines):
            if i % 2 == 0:
                f.write(f"ERROR line {i}: something went wrong in module {i % 50}, retry {i % 7}\n")
            else:
                f.write(f"INFO line {i}: heartbeat ok, uptime {i % 1000}s\n")

def sum_len_via_list(path, keyword):
    # 一行推导式看起来干净利落, 但整份匹配结果会被完整攒进一个 list
    with open(path, encoding="utf-8") as f:
        matches = [line for line in f if keyword in line]
    return sum(len(line) for line in matches)

def sum_len_via_generator(path, keyword):
    # 生成器表达式: 同一份文件, 同一个过滤条件, 但从来不会把所有匹配行同时放进内存
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in (ln for ln in f if keyword in ln):
            total += len(line)
    return total

tmp_dir = tempfile.mkdtemp()
path = os.path.join(tmp_dir, "big.log")
N = 250_000
make_log_file(path, N)

# 正确性优先: 两种写法必须算出同一个结果, 才谈得上比较内存
assert sum_len_via_list(path, "ERROR") == sum_len_via_generator(path, "ERROR")

tracemalloc.start()
sum_len_via_list(path, "ERROR")
_, peak_list = tracemalloc.get_traced_memory()
tracemalloc.stop()

tracemalloc.start()
sum_len_via_generator(path, "ERROR")
_, peak_gen = tracemalloc.get_traced_memory()
tracemalloc.stop()

os.remove(path)
os.rmdir(tmp_dir)

assert peak_list > peak_gen * 3, f"list 版峰值内存应该明显更高: list={peak_list}, gen={peak_gen}"
print(f"OK: {N:,} 行日志(一半匹配 ERROR)。list 推导式峰值内存={peak_list:,} 字节, "
      f"生成器表达式峰值内存={peak_gen:,} 字节, 比值={peak_list / peak_gen:.1f}x——"
      f"不是'读文件'本身爆内存(文件本来就是逐行迭代的), 是'把所有匹配行都攒进一个 list'这一步爆内存。")
```

实测(`.venv` 真跑):250,000 行日志(一半匹配 `ERROR`)。list 推导式峰值内存 `13,829,319` 字节,生成器表达式峰值内存只有 `34,686` 字节,比值约 `398.7x`——这不是一个"理论上更省"的模糊结论,匹配行越多、文件越大,这个比值只会继续往上涨。

**可运行例子(2/2):生成器只能消费一次——决策依据是"要不要复用",不是"哪个更高级"**

```python
import os
import tempfile

def make_log_file(path, n_lines):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_lines):
            tag = "ERROR" if i % 2 == 0 else "INFO"
            f.write(f"{tag} line {i}\n")

tmp_dir = tempfile.mkdtemp()
path = os.path.join(tmp_dir, "small.log")
N = 1000
make_log_file(path, N)

# 场景: 先数一共有多少条匹配, 再看前 3 条内容——两次都要用到"匹配结果"
f = open(path, encoding="utf-8")
matches_gen = (line for line in f if "ERROR" in line)

count = sum(1 for _ in matches_gen)      # 第一遍消费: 生成器被彻底耗尽
first_three = list(matches_gen)[:3]       # 第二遍: 生成器已经空了, 不报错, 但结果是空列表
f.close()

assert count == N // 2
assert first_three == [], f"生成器耗尽后第二次消费应该是空列表, 实际拿到 {first_three}"

# 需要复用两次(或者需要 len()/随机访问)时, 就是"该用 list"的信号——不能对着一个已经
# 决定"要流式"的生成器又提出"多消费一次"的要求; 要么改成 list 一次性拿到全部结果,
# 要么把两件事合并进同一次遍历里做完
with open(path, encoding="utf-8") as f2:
    matches_list = [line for line in f2 if "ERROR" in line]
count2 = len(matches_list)
first_three_2 = matches_list[:3]

assert count2 == N // 2
assert len(first_three_2) == 3

os.remove(path)
os.rmdir(tmp_dir)

print(f"OK: 生成器只能消费一次——先数完({count}条)后再取前 3 条, 第二遍拿到空列表 {first_three!r}; "
      f"改成 list 之后可以随便复用, count={count2}, 前 3 条真的有内容: {[s.strip() for s in first_three_2]}。"
      f"决策依据是需不需要复用: 只流过一次就够用选生成器省内存, 需要复用/len()/随机访问就必须选 list, "
      f"多花的内存是必要代价, 不是没做优化。")
```

**常见坑:** 把"生成器表达式更省内存"泛化成"生成器表达式总是更好",却答不出"如果需要复用结果怎么办"这个反问;把内存爆炸的原因错误地归结为"读文件这一步",而不是"攒 list 这一步"——这个归因错误会导致候选人给出错误的修复方向(比如去纠结"怎么更快地读文件",而不是"要不要把结果放进 list");生成器耗尽之后二次消费拿到空结果,却当成"这个函数本来就没匹配到东西",没有意识到是消费方式用错了,不是数据本身的问题。

---

## 案例 3:namedtuple 还是 dataclass——"更省内存"这个结论成立的前提是什么(决策依据追问轴)

建立在 [03-containers-and-stdlib.md 第 3 节](03-containers-and-stdlib.md) `collections.namedtuple` 之上——那一节"常见坑 2"用 `sys.getsizeof` 测过**单个孤立实例**:dataclass 实例本身 + 它的 `__dict__` 一共 144 字节左右,namedtuple 只要 64 字节左右,单实例比例大约 2.25 倍,并在"常见坑 1"给出了取舍原则("字段固定、创建后不会再变、想要更省内存选 namedtuple;字段需要默认值语法糖、需要可变、需要写方法选 dataclass"),也用过 `EvalRecord = collections.namedtuple("EvalRecord", ["task", "accuracy", "cost_usd"])` 这个具体场景。但单实例对比藏着一个没有被检验过的假设:**把这个单实例的内存差距乘以 20 万,是不是就等于 20 万条记录真实的聚合内存差距?** 本案例现场验证这个假设,会发现答案是"差得远",并且能现场找到原因。

**追问链条完整还原:**

- **Q(基础,03 类已覆盖):** "评测流水线要产出几十万条评测记录,每条记录有 task/accuracy/cost 三个字段,你会选裸 tuple、namedtuple 还是 dataclass?" —— 期望候选人复述 03 类已有的取舍原则:量大、字段固定、不需要改,选 namedtuple 更省内存;需要方法/默认值/可变,选 dataclass。
- **追问 1(逼问"乘出来的数字"是不是真的):** "03 类常见坑 2 测过一个实例:dataclass 144 字节左右,namedtuple 64 字节左右,大概 2.25 倍。如果我现在真的构建 20 万条记录,聚合内存差距是不是也接近 2.25 倍?" —— 期望候选人现场测量,而不是直接按比例心算乘出一个数字;实测会发现聚合差距只有约 1.14 倍,远小于单实例朴素外推的预期,候选人要能诚实报告"和预期不符",而不是硬凑一个数字或者含糊带过。
- **追问 2(逼问差距被抹平的原因):** "为什么聚合测出来的差距,比单实例乘出来的预期小这么多?" —— 期望候选人推出(或者在提示下想到)CPython 的 key-sharing dict 优化(PEP 412):同一个类的大量实例,如果 `__init__` 按相同顺序设置相同的属性名,会共享一份"属性名 → 槽位"的映射表,每个实例的 `__dict__` 只需要存自己的一份 values 数组,不需要每个实例各自完整存一份哈希表结构——03 类的单实例测量测的是"一个 `__dict__` 自己声称占多大",测不出"这份属性名表是不是被很多实例共享"这件事。
- **追问 3(不满足于"名词解释",要求现场验证机制真实存在):** "能不能证明确实是这个机制在起作用,不是你编的一个听起来合理的解释?" —— 期望候选人现场构造一个"故意打破 key-sharing"的对照组:给每个实例都挂一个名字互不相同的额外属性(破坏"所有实例属性名完全一致"这个前提),重新测量,如果内存显著反弹,就说明"共享属性名表"这件事真实存在且量级相符。
- **深挖追问(收尾到 `slots=True`):** "那 `dataclass(slots=True)` 呢,它是不是应该比这两者都更省?" —— 期望候选人现场测量出:`slots=True` 的 dataclass 实例连 `__dict__` 都没有(属性直接是固定偏移量的槽位),实测甚至比 namedtuple 更省一点——因为 namedtuple 本质仍然是一个 tuple 对象,带着 tuple 类型本身的开销,`slots` 对象没有这层开销。

**可运行例子(1/2):200,000 条记录的真实聚合内存测量——单实例外推的预期(约 2.25 倍)被现场证伪,并现场验证真正的原因**

```python
import tracemalloc
from dataclasses import dataclass
from collections import namedtuple

EvalRecordNT = namedtuple("EvalRecordNT", ["task", "accuracy", "cost_usd"])

@dataclass
class EvalRecordDC:
    task: str
    accuracy: float
    cost_usd: float

@dataclass(slots=True)
class EvalRecordDCSlots:
    task: str
    accuracy: float
    cost_usd: float

N = 200_000

def build_namedtuples():
    return [EvalRecordNT(f"task{i % 50}", (i % 100) / 100, 0.01 * (i % 20)) for i in range(N)]

def build_dataclasses():
    return [EvalRecordDC(f"task{i % 50}", (i % 100) / 100, 0.01 * (i % 20)) for i in range(N)]

def build_dataclasses_slots():
    return [EvalRecordDCSlots(f"task{i % 50}", (i % 100) / 100, 0.01 * (i % 20)) for i in range(N)]

def build_dataclasses_broken_sharing():
    # 每个实例额外挂一个"名字各不相同"的属性, 故意打破"同一个类的所有实例属性名完全一致"
    # 这个前提, 让 CPython 没办法在这些实例之间共享同一份属性名表
    out = []
    for i in range(N):
        r = EvalRecordDC(f"task{i % 50}", (i % 100) / 100, 0.01 * (i % 20))
        setattr(r, f"extra_{i}", i)
        out.append(r)
    return out

tracemalloc.start()
nt_list = build_namedtuples()
_, peak_nt = tracemalloc.get_traced_memory()
tracemalloc.stop()

tracemalloc.start()
dc_list = build_dataclasses()
_, peak_dc = tracemalloc.get_traced_memory()
tracemalloc.stop()

tracemalloc.start()
dcs_list = build_dataclasses_slots()
_, peak_dcs = tracemalloc.get_traced_memory()
tracemalloc.stop()

tracemalloc.start()
broken_list = build_dataclasses_broken_sharing()
_, peak_broken = tracemalloc.get_traced_memory()
tracemalloc.stop()

ratio_dc_nt = peak_dc / peak_nt
ratio_dcs_nt = peak_dcs / peak_nt
ratio_broken_dc = peak_broken / peak_dc

# 真实聚合差距远小于单实例乘出来的 2.25 倍预期——不是预期错了, 是"直接乘单实例数字"这个外推方式本身站不住脚
assert ratio_dc_nt < 1.3, f"聚合内存差距应该远小于单实例朴素外推的 2.25 倍, 实测 {ratio_dc_nt:.3f}"
assert ratio_dcs_nt <= 1.05, "dataclass(slots=True) 应该追平甚至略微超过 namedtuple 的内存效率"
# 打破 key-sharing 之后, 内存应该明显反弹, 证明"共享属性名表"这件事真实存在, 不是编的解释
assert ratio_broken_dc > 2.0, f"打破 key-sharing 后应该明显反弹, 实测只有 {ratio_broken_dc:.3f} 倍"

print(f"OK: {N:,} 条记录。namedtuple 峰值内存={peak_nt:,} 字节; 普通 dataclass 峰值内存={peak_dc:,} 字节"
      f"(比 namedtuple 多 {(ratio_dc_nt - 1) * 100:.1f}%, 远小于单实例朴素外推的 125%); "
      f"dataclass(slots=True) 峰值内存={peak_dcs:,} 字节(比 namedtuple 少 {(1 - ratio_dcs_nt) * 100:.1f}%); "
      f"故意打破 key-sharing(每个实例挂一个名字互不相同的额外属性)后峰值内存={peak_broken:,} 字节"
      f"(是正常情况下的 {ratio_broken_dc:.2f} 倍)——这证明了聚合差距变小不是随便发生的, 是 CPython 的"
      f"key-sharing dict(PEP 412)优化真实在起作用: 同一个类的大量实例只要按相同顺序设置相同的属性名, "
      f"就能共享一份属性名表, 单实例 sys.getsizeof 对比测不出这份'跨实例共享'的节省。")
```

实测(`.venv` 真跑):200,000 条记录。namedtuple 峰值内存 `34,983,944` 字节;普通 dataclass 峰值内存 `39,787,208` 字节,只比 namedtuple 多 `13.7%`(远小于单实例朴素外推的 `125%`);`dataclass(slots=True)` 峰值内存 `31,784,184` 字节,反而比 namedtuple 少 `9.1%`;故意打破 key-sharing 后峰值内存暴涨到 `118,754,066` 字节,是正常情况下的 `2.98` 倍——三个数字放在一起,才是"聚合内存差距为什么远小于单实例外推预期"这个问题的完整证据链,不是随便挑一个数字就能说明问题。

**可运行例子(2/2):可哈希性决定能不能去重, 不可变性决定"修改"必须生成新实例**

```python
from dataclasses import dataclass, replace
from collections import namedtuple

EvalRecordNT = namedtuple("EvalRecordNT", ["task", "accuracy", "cost_usd"])

@dataclass
class EvalRecordDC:
    task: str
    accuracy: float
    cost_usd: float

@dataclass(frozen=True)
class EvalRecordDCFrozen:
    task: str
    accuracy: float
    cost_usd: float

# 去重场景: 整条记录如果完全相同(所有字段都一样), 只保留一份——namedtuple 天然可以放进 set
records_nt = [EvalRecordNT("mmlu", 0.8, 0.01), EvalRecordNT("mmlu", 0.8, 0.01), EvalRecordNT("gsm8k", 0.9, 0.02)]
unique_nt = set(records_nt)
assert len(unique_nt) == 2

# 普通(可变) dataclass 默认不可哈希——@dataclass 自动生成了 __eq__, 但 eq=True/frozen=False
# 这个默认组合会让 Python 把 __hash__ 设成 None
try:
    hash(EvalRecordDC("mmlu", 0.8, 0.01))
    raised = False
except TypeError:
    raised = True
assert raised, "可变 dataclass 默认应该不可哈希"

# frozen=True 才能重新获得可哈希性质, 效果和 namedtuple 一致
records_frozen = [EvalRecordDCFrozen("mmlu", 0.8, 0.01), EvalRecordDCFrozen("mmlu", 0.8, 0.01)]
unique_frozen = set(records_frozen)
assert len(unique_frozen) == 1

# 不可变性带来的另一面: "修改"必须生成新实例, 不能原地改——namedtuple 用 _replace(),
# frozen dataclass 用 dataclasses.replace(), 只有可变 dataclass 才能真的原地赋值
r_nt = EvalRecordNT("mmlu", 0.8, 0.01)
r_nt_updated = r_nt._replace(cost_usd=0.02)
assert r_nt_updated.cost_usd == 0.02 and r_nt.cost_usd == 0.01   # 原实例不受影响

r_frozen = EvalRecordDCFrozen("mmlu", 0.8, 0.01)
r_frozen_updated = replace(r_frozen, cost_usd=0.02)
assert r_frozen_updated.cost_usd == 0.02 and r_frozen.cost_usd == 0.01

r_mutable = EvalRecordDC("mmlu", 0.8, 0.01)
r_mutable.cost_usd = 0.02       # 可变 dataclass 才能这样原地改, namedtuple/frozen dataclass 都不行
assert r_mutable.cost_usd == 0.02

print(f"OK: 三种类型的决策依据——去重需要放进 set: namedtuple/frozen dataclass 可以"
      f"(各自去重到 {len(unique_nt)}/{len(unique_frozen)} 条), 可变 dataclass 默认 TypeError 不可哈希; "
      f"需要频繁原地修改: 只有放弃不可变性的可变 dataclass 能做到, namedtuple/frozen dataclass "
      f"都得用 _replace()/replace() 生成新实例——没有全面更优的选项, 只有给定约束排除掉谁。")
```

**常见坑:** 把单实例的 `sys.getsizeof` 差距直接乘以实例数,当成"聚合内存差距"的可靠估计,不知道 CPython 的 key-sharing dict 优化会让同一个类的大量实例远比这个朴素外推更省——这是"真实性验证轴"里一个很容易被忽视的陷阱:不是数据造假,是**外推方式本身不成立**,朴素乘法在这里就是会给出错误答案;把"namedtuple 更省内存"当成 namedtuple 本身的魔法,不知道 dataclass 开 `slots=True` 能追平甚至反超;需要去重时机械地写 `if record not in seen_list: ...`(线性扫描,O(n) 每次判断),没想到用 `set`,也就没有触发"可变 dataclass 不可哈希"这个坑;遇到"不可哈希"报错就到处加 `frozen=True`,却没意识到这会连带地让"原地修改"变得不可能——决策依据类追问要求候选人能看到一个选择背后**同时**打开和关闭的能力,不是只看到打开的那一半。

---

## 案例 4:"改成 pythonic 写法后快了很多倍"——哪几个环节是真的,哪几个只是好看(真实性验证轴)

建立在多个知识点之上,不专属于某一节——[01 类知识点 1](01-comprehensions-and-functional.md)(推导式)、[03 类知识点 2](03-containers-and-stdlib.md)(`defaultdict`)、[03 类知识点 4](03-containers-and-stdlib.md)(`deque`)、[04 类知识点 2](04-strings-and-modern-syntax.md)(`str.join`)——这几节各自都验证过"pythonic 写法更好"这个结论,但从来没有把它们**放在同一个"性能提升了很多倍"的说法里横向比较过**:这四处替换里,只有两处是真正的算法复杂度差异(`deque.popleft()` 对 `list.pop(0)`,`join` 对循环 `+=`),另外两处(推导式对手写循环、`defaultdict` 对手写 `if/else`)带来的主要是可读性,不是可测量的数量级性能提升——这正是"真实性验证轴"要考察的能力:候选人能不能诚实地区分"这次改动带来了什么"和"这次改动没带来什么"。

**追问链条完整还原(面试官/候选人对话,呼应 dsa-deep-dive 案例 5 的风格):**

- **面试官:** "你说把一段处理日志的旧代码全面'pythonic 化'之后,性能提升了很多,展开说说。"
- **候选人(含糊的回答,会被继续拆穿):** "用了推导式、`defaultdict`、`deque` 这些写法,整体应该都变快了。"——这个回答听起来方向没错,但把四个完全不同性质的改动混成了一句"整体变快",没有拆分。
- **追问 1:** "这四个改动,你觉得都是为了性能,还是有的其实只是为了好看?" —— 期望候选人现场想清楚:推导式 vs 手写循环 + `.append()`,两者都还是 Python 层面逐元素跑一遍,复杂度一样,顶多因为省掉了重复的属性查找有个小的常数因子差异;`defaultdict` vs `if key not in d: d[key] = []`,两者都是每次一次哈希查找,复杂度也一样。真正有算法复杂度差异的是另外两处:`list.pop(0)` 每次都要把剩下的元素整体前移是 O(n),`deque.popleft()` 是 O(1);循环里 `+=` 拼接字符串在很多解释器实现下有重复拷贝的风险,`join` 提前算好总长度一次分配。
- **追问 2(逼出具体数字,不能停在"应该差不多"这个直觉):** "光说'差不多'和'有数量级差异'不够,能现场测一下,分别是多少倍吗?" —— 期望候选人用 `timeit` 分别测量四组对照,给出具体比值,而不是停留在定性描述。
- **深挖追问(收束到"整体倍数"这个问题本身取决于什么):** "所以'整体代码到底快了多少倍'这个问题,答案取决于什么?" —— 期望候选人指出:取决于原代码的瓶颈到底落在哪几个环节。如果原代码里根本没有 `deque`/`join` 能顶替的那种 O(n²) 操作,只是把几个手写循环换成了推导式,那所谓的"pythonic 化性能提升"很可能趋近于零——可读性的提升不等于性能的提升,这是这轮追问最终要落到的结论。

**可运行例子(1/2):数量级差异组——`deque.popleft()` 对 `list.pop(0)`,`join` 对循环 `+=`**

```python
import timeit
from collections import deque

def fifo_with_list(n):
    q = list(range(n))
    out = []
    while q:
        out.append(q.pop(0))
    return out

def fifo_with_deque(n):
    q = deque(range(n))
    out = []
    while q:
        out.append(q.popleft())
    return out

def concat_with_plus(words):
    result = ""
    for w in words:
        result += w
    return result

def concat_with_join(words):
    return "".join(words)

N_QUEUE = 6000
assert fifo_with_list(N_QUEUE) == fifo_with_deque(N_QUEUE)

t_list_pop0 = min(timeit.repeat(lambda: fifo_with_list(N_QUEUE), number=3, repeat=3))
t_deque_popleft = min(timeit.repeat(lambda: fifo_with_deque(N_QUEUE), number=3, repeat=3))

words = [f"tok{i}" for i in range(20_000)]
assert concat_with_plus(words) == concat_with_join(words)

t_plus = min(timeit.repeat(lambda: concat_with_plus(words), number=20, repeat=3))
t_join = min(timeit.repeat(lambda: concat_with_join(words), number=20, repeat=3))

ratio_queue = t_list_pop0 / t_deque_popleft
ratio_join = t_plus / t_join

assert ratio_queue > 5, f"deque 应该数量级快于 list.pop(0), 实测只快 {ratio_queue:.1f} 倍"
assert ratio_join > 5, f"join 应该数量级快于 +=, 实测只快 {ratio_join:.1f} 倍"

print(f"OK(数量级差异组): list.pop(0) 比 deque.popleft() 慢 {ratio_queue:.1f} 倍(N={N_QUEUE}); "
      f"+= 拼接比 join() 慢 {ratio_join:.1f} 倍({len(words)} 个短字符串)——这两处替换是真实的算法复杂度差异"
      f"(O(n^2) 对 O(n)), '性能提升了很多倍'这句话在这两处站得住脚。")
```

实测(`.venv` 真跑):`N=6000` 的队列场景,`list.pop(0)` 比 `deque.popleft()` 慢 `12.7` 倍;`20,000` 个短字符串拼接,循环 `+=` 比 `join()` 慢 `23.7` 倍——两处都是清清楚楚的数量级差异,不是几十个百分点的量级。

**可运行例子(2/2):外观差异组——推导式对手写循环,`defaultdict` 对手写 `if/else`,几乎不变**

```python
import timeit
from collections import defaultdict

def group_with_ifelse(pairs):
    result = {}
    for tag, name in pairs:
        if tag not in result:
            result[tag] = []
        result[tag].append(name)
    return result

def group_with_defaultdict(pairs):
    result = defaultdict(list)
    for tag, name in pairs:
        result[tag].append(name)
    return dict(result)

def squares_with_loop(nums):
    out = []
    for x in nums:
        out.append(x * x)
    return out

def squares_with_comprehension(nums):
    return [x * x for x in nums]

pairs = [(f"tag{i % 200}", f"item{i}") for i in range(30_000)]
assert group_with_ifelse(pairs) == group_with_defaultdict(pairs)

t_ifelse = min(timeit.repeat(lambda: group_with_ifelse(pairs), number=5, repeat=3))
t_defaultdict = min(timeit.repeat(lambda: group_with_defaultdict(pairs), number=5, repeat=3))

nums = list(range(200_000))
assert squares_with_loop(nums) == squares_with_comprehension(nums)

t_loop = min(timeit.repeat(lambda: squares_with_loop(nums), number=5, repeat=3))
t_comp = min(timeit.repeat(lambda: squares_with_comprehension(nums), number=5, repeat=3))

ratio_dict = t_ifelse / t_defaultdict
ratio_comp = t_loop / t_comp

assert 0.4 < ratio_dict < 3.0, f"defaultdict 对手写 if-else 不应该有数量级差异, 实测比值 {ratio_dict:.2f}"
assert 0.4 < ratio_comp < 3.0, f"推导式对手写循环不应该有数量级差异, 实测比值 {ratio_comp:.2f}"

print(f"OK(外观差异组, 几乎不变): if-else 对 defaultdict 比值={ratio_dict:.2f}x; "
      f"手写循环对推导式比值={ratio_comp:.2f}x——都在同一个数量级以内, "
      f"这两处替换换来的主要是可读性(不用手写'key 是否第一次出现'的判断、不用手写空 list + append), "
      f"不是可测量的数量级性能提升; 如果面试官追问'具体哪个环节快了多少倍', "
      f"诚实的答案是'这两处几乎没变, 真正快的是另外两处'。")
```

实测(`.venv` 真跑):`30,000` 条待分组记录,手写 `if/else` 比 `defaultdict` 只慢 `1.33` 倍;`200,000` 个数字求平方,手写循环比推导式只慢 `1.11` 倍——都在同一个数量级以内,和例子(1/2)里 `12.7`/`23.7` 倍的数量级差异形成鲜明对照。

**常见坑:** 把四个不同性质的写法改动打包成一句"整体快了很多倍",不去分别测量、也不去想清楚哪些是复杂度差异、哪些只是常数因子或者纯粹的可读性;反过来走向另一个极端,觉得"pythonic 写法都只是为了好看,不影响性能"——这同样是不诚实的,`deque`/`join` 这两处是真实的、可测量的数量级差异,一竿子打死同样是在回避具体数字;测量时不先验证两种写法的输出完全一致就直接比较耗时,如果两个版本其实算出了不同的结果,测出来的"性能提升"毫无意义。

---

## 案例 5:sorted 的多级排序技巧到哪里失效——从两趟排序踩坑到 functools.cmp_to_key(方案批判迭代轴 + 决策依据追问轴)

建立在 [03-containers-and-stdlib.md 第 7 节](03-containers-and-stdlib.md) `sorted`/`.sort()` 的 `key=` 参数之上——那一节"常见坑 1"已经验证过"先按次要键排、再按主要键排"这两趟稳定排序等价于一次 `sorted(pairs, key=lambda p: (p[1], p[0]))`,但没有验证过"顺序搞反会发生什么"这个更容易在真实代码里犯的错误,也没有讨论过 `key=` 这个机制本身在什么场景下会彻底失效。这个案例现场引入 `functools.cmp_to_key`(不在 01-04 任何一节的知识点列表里,类似 dsa-deep-dive 案例 3 现场引入限流算法的做法),作为"key= 失效之后换的下一个方案"。

**追问链条完整还原(面试官/候选人对话,方案连续被指出具体缺陷):**

- **面试官给约束:** "给一批评测记录排序:按 `cost` 升序;`cost` 相同的话,再按 `accuracy` 降序。"
- **候选人方案 1(两趟排序,依赖稳定性,但顺序搞反了):** "我先按 `cost` 排一趟,再按 `accuracy` 倒序排一趟。"
- **面试官指出具体缺陷(现场构造出具体错误结果,不是"顺序好像不太对"这种空话):** "拿 4 条具体记录跑一遍,你选的这个顺序算出来的结果,和'先按 cost 升序、cost 相同再按 accuracy 降序'这个要求对得上吗?" —— 期望候选人现场推演或者写代码验证,发现:先排主键(cost)、最后排次键(accuracy)的顺序是错的——最后一趟按 accuracy 排序的稳定性,只保证"accuracy 相同的元素维持之前的相对顺序",对"cost 分组"这件事一无所知,会把第一趟已经排好的 cost 分组重新打散。
- **候选人方案 2(纠正顺序,或者干脆换成单次 tuple key):** "那应该反过来,先排 accuracy,再排 cost;或者更省心,直接写 `sorted(records, key=lambda r: (r.cost, -r.accuracy))`,一次搞定,不用记'先排哪个'这条规则。"——面试官认可这个方案,继续加约束。
- **面试官指出新约束(不是缺陷,是需求变化,逼问 key= 这个机制本身的边界):** "现在需求变了:`cost`/`accuracy` 之外,还要求胜场数(`win_count`)相同的两个 agent,用它们**头对头(head-to-head)对战的胜者**决定谁排前面,不是看某个各自独立的字段。你这个 tuple key 方案还能表达这条规则吗?"
- **候选人(容易想岔的方向):** "我可以把头对头战绩也变成一个数字塞进 key 里?" —— 面试官追问:"这个数字要怎么算?'A 和 B 头对头谁赢了'是 A、B 两条记录之间的关系,不是 A 自己身上能读出来的一个属性,你打算怎么让 `key=` 这个'单个元素 → 单个可排序值'的函数,读到'另一个具体元素'的信息?" —— 期望候选人意识到:`key=` 函数的签名从根本上决定了它只能看到"当前这一个元素自己的字段",算不出"这个元素和另一个特定元素之间的关系"这种真正的成对比较,这不是"再想一个聪明的映射"能绕过去的,是这个机制的结构性边界。
- **候选人方案 3(换成 `functools.cmp_to_key`):** "那就不能用 `key=` 了,得用一个真正的两两比较函数,`functools.cmp_to_key` 包一下传给 `sorted`。"
- **深挖追问(逼问新方案的代价):** "`cmp_to_key` 包出来的比较函数,和直接传 `key=` 相比,复杂度上有什么区别?你能说出具体代价在哪吗?" —— 期望候选人指出:`key=` 只需要对每个元素算一次 key(整个排序过程只算 n 次),后续所有比较都是拿预先算好的值直接比;`cmp_to_key` 每次比较都要重新调用一次这个比较函数(不能预先算出来缓存),对于 Timsort 这类基于比较的排序算法,比较次数量级还是 O(n log n),但每一次比较的常数开销明显更高——这是"能用 key= 就不用 cmp_to_key"的真实代价来源,只有像头对头战绩这种真正依赖"两个元素之间关系"、没法压缩成单个元素自身可比较值的场景,才值得付这个代价。

**可运行例子(1/2):两趟排序顺序搞反的真实 bug——4 条记录, 具体错误结果**

```python
from dataclasses import dataclass

@dataclass
class RankedAgent:
    name: str
    cost: float
    accuracy: float

records = [
    RankedAgent("B", cost=1, accuracy=0.3),
    RankedAgent("D", cost=2, accuracy=0.5),
    RankedAgent("A", cost=1, accuracy=0.9),
    RankedAgent("C", cost=2, accuracy=0.7),
]

# 目标: 按 cost 升序; cost 相同的话再按 accuracy 降序
expected_order = ["A", "B", "C", "D"]

# 错误顺序: 先按"主键"cost 排, 最后才按"次键"accuracy 排——第二趟排序的稳定性只认
# accuracy, 对 cost 分组一无所知, 会把第一趟已经排好的 cost 分组重新打散
wrong_step1 = sorted(records, key=lambda r: r.cost)
wrong_final = sorted(wrong_step1, key=lambda r: -r.accuracy)
wrong_names = [r.name for r in wrong_final]

# 正确顺序: 先按"次键"accuracy 排, 最后按"主键"cost 排——稳定排序保证最后一趟(cost)
# 排序不会打乱同 cost 内部已经按 accuracy 排好的相对顺序
right_step1 = sorted(records, key=lambda r: -r.accuracy)
right_final = sorted(right_step1, key=lambda r: r.cost)
right_names = [r.name for r in right_final]

# 单次 tuple key: 等价于"正确顺序"的两趟排序, 而且不需要记住"先排哪个"这条规则
oneshot_names = [r.name for r in sorted(records, key=lambda r: (r.cost, -r.accuracy))]

assert wrong_names != expected_order, f"顺序搞反应该产出错误结果, 实际是 {wrong_names}"
assert right_names == expected_order == oneshot_names

print(f"OK: 两趟排序顺序搞反(先排主键 cost, 再排次键 accuracy) -> {wrong_names}"
      f"(错误, 不等于期望的 {expected_order}); "
      f"顺序对了(先排次键, 再排主键) -> {right_names}(正确); "
      f"单次 tuple key 排序 -> {oneshot_names}(正确, 且不用记'先排哪个'这条规则)。"
      f"03 号文件常见坑 1 已经点过'两趟法等价于一次多级排序'这个关系, 这里额外验证的是"
      f"'顺序搞反会得到什么具体错误结果'。")
```

**可运行例子(2/2):头对头胜负——`key=` 表达不了的成对关系, 以及 `cmp_to_key` 的真实性能代价**

```python
import random
import timeit
from functools import cmp_to_key

agents = [
    {"name": "beta", "win_count": 3},
    {"name": "alpha", "win_count": 5},
    {"name": "delta", "win_count": 1},
    {"name": "gamma", "win_count": 3},
]
head_to_head = {frozenset({"beta", "gamma"}): "gamma"}   # beta 和 gamma 头对头交手, gamma 赢了

def compare(a, b):
    if a["win_count"] != b["win_count"]:
        return b["win_count"] - a["win_count"]          # 胜场多的排前面(降序)
    winner = head_to_head.get(frozenset({a["name"], b["name"]}))
    if winner == a["name"]:
        return -1
    if winner == b["name"]:
        return 1
    return 0                                              # 没有头对头数据, 视为并列

standings_names = [a["name"] for a in sorted(agents, key=cmp_to_key(compare))]

# 只用 key=(胜场数降序), 胜场相同时只能维持输入顺序, 表达不出"gamma 头对头赢了 beta"这条规则
key_only_names = [a["name"] for a in sorted(agents, key=lambda a: -a["win_count"])]

assert key_only_names == ["alpha", "beta", "gamma", "delta"]     # 胜场相同时只是维持了输入顺序
assert standings_names == ["alpha", "gamma", "beta", "delta"]     # cmp_to_key 正确地把 gamma 排到了 beta 前面
assert key_only_names != standings_names

# cmp_to_key 的真实代价: 用同一批数据、语义等价的比较逻辑, 测直接 key= 和 cmp_to_key 的耗时差
rng = random.Random(42)
data = [rng.random() for _ in range(20_000)]

def numeric_cmp(a, b):
    return (a > b) - (a < b)

assert sorted(data) == sorted(data, key=cmp_to_key(numeric_cmp))

t_key = min(timeit.repeat(lambda: sorted(data, key=lambda x: x), number=5, repeat=3))
t_cmp = min(timeit.repeat(lambda: sorted(data, key=cmp_to_key(numeric_cmp)), number=5, repeat=3))
ratio = t_cmp / t_key

assert ratio > 3, f"cmp_to_key 应该明显慢于直接 key=, 实测只慢 {ratio:.1f} 倍"

print(f"OK: 头对头战绩场景, key= 只能给出 {key_only_names}(丢了头对头信息), "
      f"cmp_to_key 给出 {standings_names}(gamma 正确排到 beta 前面)。"
      f"性能代价: 同样排 {len(data):,} 个数字, 直接 key= 用时 {t_key:.4f}s, "
      f"cmp_to_key 包装的比较函数用时 {t_cmp:.4f}s, 慢 {ratio:.1f} 倍——"
      f"这就是'能用 key= 就不用 cmp_to_key'的具体代价来源。")
```

实测(`.venv` 真跑):头对头战绩场景,`key=` 只能给出 `['alpha', 'beta', 'gamma', 'delta']`(丢了头对头信息),`cmp_to_key` 给出正确的 `['alpha', 'gamma', 'beta', 'delta']`;性能代价上,同样排 `20,000` 个数字,直接 `key=` 用时 `0.0253s`,`cmp_to_key` 用时 `0.1854s`,慢 `7.3` 倍。

**常见坑:** 两趟排序时凭直觉先排"看起来更重要"的主键,而不是记住"稳定排序的两趟法必须次键在前、主键在后"这条具体规则;遇到 `key=` 表达不了的需求,第一反应是硬凑一个更复杂的映射函数,而不是识别出"这是 key= 这个机制本身的结构性边界"(它的签名只接受单个元素,读不到"另一个元素"的信息),白白花时间在一个注定绕不过去的方向上;用了 `cmp_to_key` 之后,不知道它有真实可测量的性能代价,以为"反正能排出正确结果就行",面试官追问代价时答不上来。

---

## 小结:5 个案例对应调研发现的哪些轴线

| 案例 | 规模递增轴 | 工程约束递增轴(并发) | 方案批判迭代轴 | 决策依据追问轴 | 真实性验证轴 |
|---|---|---|---|---|---|
| 1. EAFP/LBYL → TOCTOU 竞态 | | ✅ 核心 | | | |
| 2. 推导式 vs 生成器表达式 | ✅ 核心 | | | ✅(复用/内存取舍) | |
| 3. namedtuple vs dataclass | | | | ✅ 核心 | |
| 4. "pythonic 化"性能声称 | | | | | ✅ 核心 |
| 5. sorted key= → cmp_to_key | | | ✅ 核心 | ✅(key= 的结构性边界) | |

这 5 个案例不是要覆盖 26 个知识点里的每一个——它们演示的是**方法论本身**:拿到任何一个已经掌握的知识点,都可以自己追问"数据规模再大 10000 倍会怎样""换成多线程会怎样""如果面试官连续否定我的方案,下一个更合理的备选是什么""我怎么用具体数字而不是形容词说服别人"。真正的二面深度,是能不能对着一个自己没准备过的知识点,现场把这几条轴线走一遍。
