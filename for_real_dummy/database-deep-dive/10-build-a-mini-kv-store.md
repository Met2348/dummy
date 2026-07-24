# 10 · 手把手实战:从零搭一个迷你 KV 存储引擎

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 10 个"知识点",不计入"65 个知识点"的统计——和 [09 类](09-mock-interview-capstone.md)是同一挂,但风格不一样:09 号文件里,你是**旁观者**,跟着两条真实数据库连接的并发时序把一条推理链条看一遍;这一篇里,你是**动手的人**——从一个空文件开始,一步步敲代码,每写一段就跑一次、看到真实效果,最后独立搭出一个完整能用的小工具。这个格式最初在 [dsa-deep-dive/21-build-a-mini-search-engine.md](../dsa-deep-dive/21-build-a-mini-search-engine.md) 试点验证,那篇教程结尾"可以怎么继续扩展"提到"持久化存储属于 database-deep-dive 存储引擎内部机制的范畴"——这一篇就是把那个方向真正做出来。

## 为什么是"KV 存储引擎"

不是要发明新知识点,是把 [06 类](06-storage-engine-internals.md)(存储引擎内部机制)和 [04 类](04-transactions-and-isolation-levels.md)(事务与隔离级别)里已经讲过的内容串成一个真实能跑的小工具。真实数据库(PostgreSQL/MySQL 的 InnoDB/RocksDB 等)内部都有一层这样的机制,只是复杂得多——这篇教程只做最小可行版本,但每一步用到的原理和它们是同一套。

| 阶段 | 要让程序多会一件事 | 建立在哪个已有知识点之上 |
|------|------|------|
| 阶段 1 | 有一个能 get/set/delete 的存储接口(基线,还没有持久化) | 朴素 dict 封装,不涉及任何已完成知识点,是后面几个阶段要解决的问题的起点 |
| 阶段 2 | 每次写入先把变更记录追加写入磁盘日志、日志落盘之后才允许内存状态真正改变 | [06 类](06-storage-engine-internals.md) 知识点1:WAL"先写日志,日志比数据先落盘"的核心规则 |
| 阶段 3 | 进程重启后,从磁盘上的日志重放出崩溃前的完整状态;同时看清这个迷你引擎的"原子性"止步于哪里 | [06 类](06-storage-engine-internals.md) 知识点2(redo/undo分工)、知识点4(ARIES三阶段恢复思想)、[04 类](04-transactions-and-isolation-levels.md) 知识点1(事务原子性) |
| 阶段 4(可选加深) | 崩溃恰好发生在写到一半时,恢复逻辑能识别并跳过这条损坏记录,不整体崩溃 | [06 类](06-storage-engine-internals.md) 知识点1 真实捕获的 PostgreSQL 崩溃日志"invalid record length ... got 0"这一现象的迷你复现 |

每个阶段的代码都能独立运行(本文件用 database-deep-dive 目录下的 `_verify_md.py` 校验,校验方式和 dsa-deep-dive 系列一样:把每个 ` ```python ` 代码块单独拎出来起一个新的 Python 子进程执行——块与块之间**不共享任何变量**,所以后面阶段用到前面阶段的类时,会重新贴一遍完整定义,不是偷懒复制,是这套校验机制要求的)。日志记录格式定死成两种简单的文本行:`SET key value\n` 和 `DEL key\n`——为了让解析逻辑保持简单,这篇教程里的 key 不包含空格,value 可以包含空格(因为解析时用 `split(" ", 2)`,只切前两个空格,其余部分整体作为 value)。

---

## 阶段 1:纯内存版——先有一个能用的接口,但一退出就清空

最简单的存储引擎:一个 dict,外面包一层 `get`/`set`/`delete`。能正常工作,但完全不碰磁盘——进程一退出,数据就彻底没了,下一次启动是一张白纸。这是后面所有阶段要解决的问题的起点。

```python
class MiniKV:
    """Simplest possible KV store: a dict wrapped behind a get/set/delete
    interface. Nothing ever touches disk - once the process exits, every
    key is gone."""

    def __init__(self):
        self._data = {}

    def set(self, key, value):
        self._data[key] = value

    def get(self, key):
        return self._data.get(key)

    def delete(self, key):
        self._data.pop(key, None)


kv = MiniKV()
kv.set("alice", "100")
kv.set("bob", "50")
assert kv.get("alice") == "100"
assert kv.get("bob") == "50"
assert kv.get("nobody") is None

kv.delete("alice")
assert kv.get("alice") is None
assert kv.get("bob") == "50"   # deleting alice must not touch bob

# "process restart" in this version just means: build a brand new instance.
# There is no file involved anywhere, so a fresh instance starts empty no
# matter what the previous instance held.
restarted_kv = MiniKV()
assert restarted_kv.get("bob") is None, "a fresh process has zero memory of the previous instance's data"

print("stage1 ok: in-memory KV works while running, but a fresh instance remembers nothing")
```

`get`/`set`/`delete` 三个方法本身没有任何值得深挖的地方——真正的问题从下一阶段才开始:怎么让数据"扛得住进程重启"。

## 阶段 2:加一份预写日志(WAL)——写日志再改内存,而不是反过来

[06 类知识点1](06-storage-engine-internals.md) 给出的定义是:"任何数据修改必须先把'变更记录'写入WAL并落盘,才允许对应的数据页修改真正生效"。这一步就是把这条规则字面意义地实现出来:每次 `set`/`delete`,先把一行文本记录追加写入磁盘上的日志文件并调用 `os.fsync` 强制落盘,**这一步完全成功之后**,才允许修改内存里的 dict。顺序不能反——如果先改内存、日志随后才写,那么"内存已经改了但日志还没落盘"这个窗口期一旦崩溃,这次修改就会不明不白地丢失,WAL 的整套持久性保证就是靠"日志永远先于数据变更生效"这一条规则撑住的。

```python
import os
import tempfile


class MiniKVStore:
    """A KV store backed by a Write-Ahead Log: every write is appended to a
    plain-text log file on disk (and fsync'd) BEFORE the in-memory dict is
    updated. This is the "Write-Ahead" rule from the storage-engine WAL
    knowledge point - the log record must exist on disk before the change
    it describes is allowed to take effect."""

    def __init__(self, wal_path):
        self.wal_path = wal_path
        self._data = {}
        self._wal = open(wal_path, "ab")  # binary append: bytes written == bytes on disk, no OS newline translation

    def _append(self, line_str):
        self._wal.write(line_str.encode("utf-8"))
        self._wal.flush()
        os.fsync(self._wal.fileno())  # force the OS to actually put these bytes on disk, not just in a buffer

    def set(self, key, value):
        self._append(f"SET {key} {value}\n")
        self._data[key] = value

    def delete(self, key):
        self._append(f"DEL {key}\n")
        self._data.pop(key, None)

    def get(self, key):
        return self._data.get(key)

    def close(self):
        self._wal.close()


def replay(wal_path):
    """Independently reconstruct state by reading the WAL file from
    scratch. Used here only to CHECK that the log matches memory while the
    process is alive - real recovery-on-restart comes in stage 3."""
    state = {}
    with open(wal_path, "rb") as f:
        text = f.read().decode("utf-8")
    for line in text.split("\n"):
        if not line:
            continue
        parts = line.split(" ", 2)
        if parts[0] == "SET":
            state[parts[1]] = parts[2]
        elif parts[0] == "DEL":
            state.pop(parts[1], None)
    return state


tmpdir = tempfile.mkdtemp()
wal_path = os.path.join(tmpdir, "store.wal")
try:
    store = MiniKVStore(wal_path)
    store.set("alice", "100")
    store.set("bob", "50")
    store.delete("alice")
    store.set("carol", "75")

    # While the process is alive: independently replaying the on-disk WAL
    # must reproduce exactly the same state the in-memory dict holds.
    assert replay(wal_path) == store._data == {"bob": "50", "carol": "75"}

    # The WAL is real plain text sitting on disk - inspect it directly.
    with open(wal_path, "rb") as f:
        raw_lines = f.read().decode("utf-8").splitlines()
    assert raw_lines == ["SET alice 100", "SET bob 50", "DEL alice", "SET carol 75"]

    store.close()
    print("stage2 ok: on-disk WAL content and in-memory state stay consistent while the process is alive")
finally:
    if os.path.exists(wal_path):
        os.remove(wal_path)
    os.rmdir(tmpdir)
```

两个实现细节值得注意:

- **用 `"ab"`(二进制追加)而不是文本模式追加。** Windows 平台上,文本模式写入默认会把程序写的 `"\n"` 翻译成 `"\r\n"`,后面几个阶段要按字节精确定位"日志文件是不是恰好以完整的一行结尾"(阶段4会用到),这类判断必须建立在"写下去的字节和读回来的字节完全一致"这个前提上,二进制模式把这个隐患直接消除,不依赖"这台机器的换行符习惯恰好没有添乱"。
- **`os.fsync` 不是可有可无的装饰。** [06 类知识点1"常见坑"](06-storage-engine-internals.md) 提到过:"WAL 本身依赖的物理落盘保证(`fsync`)如果被...缓存策略绕过,WAL 机制本身提供的保证也会被削弱"——只调用 `write()`/`flush()` 只是把数据交给操作系统的页缓存,不代表真的落盘了,`os.fsync(fileno)` 才是真正等待磁盘确认写入完成的那一步。

## 阶段 3:模拟进程重启,重放 WAL 恢复状态——顺便看清这个引擎的原子性边界在哪里

给 `MiniKVStore` 加一个新能力:打开时,先检查磁盘上有没有已经存在的日志文件,如果有,在接受任何新写入之前,把它从头到尾重放一遍,重建出内存状态。"进程重启"在这里就模拟成:丢弃旧实例(不调用任何优雅关闭逻辑),再用同一个 WAL 路径构造一个全新实例。

```python
import os
import tempfile


class MiniKVStore:
    """Same WAL-backed store as stage 2, plus one new capability: on
    startup, before accepting any new writes, replay whatever WAL history
    already exists on disk to rebuild the in-memory state."""

    def __init__(self, wal_path):
        self.wal_path = wal_path
        self._data = {}
        self._recover()
        self._wal = open(wal_path, "ab")

    def _recover(self):
        if not os.path.exists(self.wal_path):
            return
        with open(self.wal_path, "rb") as f:
            text = f.read().decode("utf-8")
        for line in text.split("\n"):
            if not line:
                continue
            parts = line.split(" ", 2)
            if parts[0] == "SET":
                self._data[parts[1]] = parts[2]
            elif parts[0] == "DEL":
                self._data.pop(parts[1], None)

    def _append(self, line_str):
        self._wal.write(line_str.encode("utf-8"))
        self._wal.flush()
        os.fsync(self._wal.fileno())

    def set(self, key, value):
        self._append(f"SET {key} {value}\n")
        self._data[key] = value

    def delete(self, key):
        self._append(f"DEL {key}\n")
        self._data.pop(key, None)

    def get(self, key):
        return self._data.get(key)

    def close(self):
        self._wal.close()


tmpdir = tempfile.mkdtemp()
wal_path = os.path.join(tmpdir, "store.wal")
try:
    store = MiniKVStore(wal_path)
    store.set("alice", "100")
    store.set("bob", "50")
    store.delete("alice")
    store.set("carol", "75")
    state_before_crash = dict(store._data)

    # Simulate a crash: no graceful shutdown hook runs, we just drop the
    # file handle and the in-memory object. (Calling close() here only
    # releases the OS file lock so the next open() on Windows succeeds -
    # this toy engine has no separate "clean shutdown" code path anyway,
    # so a real process crash would leave the file in the exact same state.)
    store._wal.close()
    del store

    # Simulate a process restart: a brand new, empty in-memory state, and
    # re-opening the SAME WAL file - the constructor replays it automatically.
    recovered = MiniKVStore(wal_path)
    assert dict(recovered._data) == state_before_crash == {"bob": "50", "carol": "75"}, \
        f"replay must reproduce the exact pre-crash state, got {dict(recovered._data)}"
    recovered.close()
    print("stage3 ok: replaying the WAL from a fresh process reproduces the exact pre-crash state")
finally:
    if os.path.exists(wal_path):
        os.remove(wal_path)
    os.rmdir(tmpdir)
```

**为什么 `_recover()` 只做了"重放",完全没有出现 [06 类知识点2](06-storage-engine-internals.md) 讲的 undo?** 因为这个迷你引擎里根本不存在"未提交事务"这个状态——每一次 `set`/`delete` 调用,从它把这一行完整写入 WAL 并 `fsync` 成功的那一刻起,就已经是"生效"的,不存在"先执行、后面才决定要不要提交"这种中间状态。[06 类知识点4](06-storage-engine-internals.md) 的 ARIES 三阶段思想里,redo 负责"把日志里记录的全部修改重新应用一遍",undo 负责"撤销当时还没提交的部分"——这个迷你引擎因为没有"未提交"这个概念,天然只需要 redo 这一半,undo 那一半无从谈起。这不是偷懒省掉了什么,而是这个更简单的系统本来就不需要它。

但这也恰好暴露了这个迷你引擎的能力边界。[04 类知识点1](04-transactions-and-isolation-levels.md) 讲的"事务级原子性"针对的是**多条语句**要一起生效或一起不生效的场景(经典例子就是转账:A 账户扣款、B 账户加款必须同时成功或同时失败)。这个迷你引擎的每一次 `set`/`delete` 都是独立的一行 WAL 记录,彼此之间没有任何分组机制——用它来实现"转账"会是什么后果,现场验证一下:

```python
import os
import tempfile


class MiniKVStore:
    def __init__(self, wal_path):
        self.wal_path = wal_path
        self._data = {}
        self._recover()
        self._wal = open(wal_path, "ab")

    def _recover(self):
        if not os.path.exists(self.wal_path):
            return
        with open(self.wal_path, "rb") as f:
            text = f.read().decode("utf-8")
        for line in text.split("\n"):
            if not line:
                continue
            parts = line.split(" ", 2)
            if parts[0] == "SET":
                self._data[parts[1]] = parts[2]
            elif parts[0] == "DEL":
                self._data.pop(parts[1], None)

    def _append(self, line_str):
        self._wal.write(line_str.encode("utf-8"))
        self._wal.flush()
        os.fsync(self._wal.fileno())

    def set(self, key, value):
        self._append(f"SET {key} {value}\n")
        self._data[key] = value

    def delete(self, key):
        self._append(f"DEL {key}\n")
        self._data.pop(key, None)

    def get(self, key):
        return self._data.get(key)

    def close(self):
        self._wal.close()


tmpdir = tempfile.mkdtemp()
wal_path = os.path.join(tmpdir, "transfer.wal")
try:
    store = MiniKVStore(wal_path)
    store.set("account_a", "1000")
    store.set("account_b", "500")

    # Simulate a "transfer": move 100 from A to B. In a real transactional
    # engine these two writes would be wrapped in one transaction so they
    # either both take effect or neither does (04-class knowledge point 1,
    # transaction-level atomicity). This mini engine has no such grouping -
    # set() calls are independent, single-line WAL records.
    store.set("account_a", "900")   # step 1 is already durably written
    store._wal.close()              # crash happens right here
    del store                       # step 2 (crediting account_b) never ran

    recovered = MiniKVStore(wal_path)
    # The recovered state is a genuine half-done transfer: A was debited,
    # B was never credited. 100 has effectively vanished. This engine's
    # WAL only guarantees atomicity for a SINGLE set()/delete() call (one
    # line either fully lands or is discarded, see stage 4) - it does NOT
    # guarantee atomicity ACROSS multiple calls the way a real transaction does.
    assert recovered.get("account_a") == "900"
    assert recovered.get("account_b") == "500"
    recovered.close()
    print("atomicity gap reproduced: account_a debited but account_b never credited after the crash")
finally:
    if os.path.exists(wal_path):
        os.remove(wal_path)
    os.rmdir(tmpdir)
```

这个结果不是 bug,是这个迷你引擎"诚实的能力边界":它只保证"单条 `set`/`delete` 记录要么完整生效、要么完全没发生"(阶段4会验证这一点),但不保证"多条操作合起来是一个不可分割的整体"。真实数据库要解决这个问题,需要在 WAL 里引入事务边界——比如给一批操作加上 `BEGIN`/`COMMIT` 标记行,恢复时只有看到匹配的 `COMMIT` 才应用这批操作,否则整批丢弃(这正是 [04 类知识点1](04-transactions-and-isolation-levels.md) 底层机制里提到的"原子性依赖 undo 日志...在失败时回滚"落到 WAL 格式设计上的直接体现)。这个扩展本文不实现,留在最后"可以怎么继续扩展"里指出方向。

## 阶段 4(可选加深):模拟"写到一半崩溃"——恢复逻辑要能跳过损坏的最后一条记录

[06 类知识点1](06-storage-engine-internals.md) 真实用 `kill -9` 崩溃过一次 PostgreSQL,日志里捕获到这一行:`invalid record length at 0/C6DAEF8: expected at least 24, got 0`——这说明真实崩溃发生的那一刻,WAL 文件末尾完全可能停留着一条没写完的"半条"记录(操作系统缓冲区里的字节还没来得及完整落盘,进程就没了),恢复逻辑必须能识别这种情况并安全跳过,而不是把这段残缺数据当成有效记录去解析、或者干脆崩掉整个恢复过程。

这一版给 `_recover()` 加一个判断:健康的 WAL 文件里,每一条完整记录都以 `\n` 结尾,所以按 `\n` 切分之后,最后一段应该总是空字符串;如果不是空字符串,说明文件没有以完整的换行收尾——末尾这一截是写到一半的残留,应该被丢弃并从物理文件里截掉(截掉是为了不让后续新写入的记录紧贴着这段垃圾字节、被意外拼接成另一条无法解析的坏记录)。

```python
import os
import tempfile


class MiniKVStore:
    """Final version: recovery now also detects and repairs a torn last
    record, the way a real crash (process dies mid-write(), only part of
    the last line reaches disk) would leave the file."""

    def __init__(self, wal_path):
        self.wal_path = wal_path
        self._data = {}
        self.skipped_tail_bytes = 0
        self._recover()
        self._wal = open(wal_path, "ab")

    def _recover(self):
        if not os.path.exists(self.wal_path):
            return
        with open(self.wal_path, "rb") as f:
            text = f.read().decode("utf-8")
        lines = text.split("\n")
        tail = lines.pop()  # a healthy WAL always ends with "\n", so this is normally ""
        if tail:
            # The file does NOT end with "\n" - the last record was still
            # being written when the crash happened (compare: real PostgreSQL
            # logs "invalid record length ... expected at least 24, got 0"
            # for exactly this situation - see 06-storage-engine-internals.md
            # knowledge point 1's real captured crash log).
            self.skipped_tail_bytes = len(tail.encode("utf-8"))
            # Truncate the file down to the last complete record so future
            # writes append cleanly instead of gluing onto the garbage tail.
            good_bytes = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
            with open(self.wal_path, "wb") as f:
                f.write(good_bytes)
        for line in lines:
            if not line:
                continue
            parts = line.split(" ", 2)
            if parts[0] == "SET" and len(parts) == 3:
                self._data[parts[1]] = parts[2]
            elif parts[0] == "DEL" and len(parts) == 2:
                self._data.pop(parts[1], None)

    def _append(self, line_str):
        self._wal.write(line_str.encode("utf-8"))
        self._wal.flush()
        os.fsync(self._wal.fileno())

    def set(self, key, value):
        self._append(f"SET {key} {value}\n")
        self._data[key] = value

    def delete(self, key):
        self._append(f"DEL {key}\n")
        self._data.pop(key, None)

    def get(self, key):
        return self._data.get(key)

    def close(self):
        self._wal.close()


tmpdir = tempfile.mkdtemp()
wal_path = os.path.join(tmpdir, "crash_mid_write.wal")
try:
    store = MiniKVStore(wal_path)
    store.set("x", "1")
    store.set("y", "2")
    store.close()

    # Hand-craft a "crashed mid-write" tail: half a record, no value, no
    # trailing newline at all - exactly what's left on disk if the process
    # dies after the OS only flushed part of the write() call's bytes.
    incomplete_record = b"SET incomplete_key_with_no_va"
    with open(wal_path, "ab") as f:
        f.write(incomplete_record)

    recovered = MiniKVStore(wal_path)
    # The two complete records must recover normally.
    assert recovered.get("x") == "1"
    assert recovered.get("y") == "2"
    # The torn record must be skipped - never applied as if it were real data.
    assert recovered.get("incomplete_key_with_no_va") is None
    assert recovered.skipped_tail_bytes == len(incomplete_record), \
        f"expected {len(incomplete_record)} skipped tail bytes, got {recovered.skipped_tail_bytes}"

    # The on-disk file must now be repaired (garbage tail truncated away) -
    # confirm by reading raw bytes directly, not through the object.
    with open(wal_path, "rb") as f:
        repaired = f.read()
    assert repaired == b"SET x 1\nSET y 2\n", f"expected the corrupted tail to be truncated away, got {repaired!r}"

    # And the store keeps working normally after recovering from corruption -
    # crucially, the new write must land AFTER a clean truncation point, not
    # get glued onto the discarded garbage bytes (which would corrupt this
    # new record too if the file hadn't been repaired first).
    recovered.set("z", "3")
    assert recovered.get("z") == "3"
    with open(wal_path, "rb") as f:
        after_new_write = f.read()
    assert after_new_write == b"SET x 1\nSET y 2\nSET z 3\n", \
        f"new writes after repair must append cleanly, got {after_new_write!r}"
    recovered.close()

    print(f"stage4 ok: recovered x=1 y=2, skipped {recovered.skipped_tail_bytes} corrupted tail bytes without crashing, store still writable afterwards")
finally:
    if os.path.exists(wal_path):
        os.remove(wal_path)
    os.rmdir(tmpdir)
```

**这里的损坏检测比真实数据库粗糙,老实说清楚粗糙在哪:** [06 类知识点1"常见坑"](06-storage-engine-internals.md) 提到真实 WAL 记录"本身也有完整性校验[比如校验和/长度字段]"——真实引擎判断一条记录是否完整,靠的是记录头部自带的长度字段和校验和,不管这条记录是不是文件的最后一条,只要内容被篡改/损坏,都能被发现。这篇教程的判断逻辑简化成了"文件是不是以完整的 `\n` 结尾",这只能测出"最后一条记录写到一半"这一种情况,测不出"中间某条记录被静默损坏但恰好还是完整一行"这种情况(比如磁盘位翻转)。这个简化对本文范围是成立的——任务要求验证的正是"写到一半崩溃"这一种具体场景,不是通用的数据完整性校验——但如果照搬这个思路去实现一个要处理任意损坏的真实系统,会遗漏真实 WAL 靠校验和才能发现的那类问题,这一点在这里明确说清楚,不含糊带过。

---

## 可以怎么继续扩展(只指方向,不实现)

- **多操作事务边界**:阶段3已经现场验证过这个迷你引擎不保证"转账"这类多步操作的整体原子性。真实做法是给一组操作加 `BEGIN`/`COMMIT` 标记行,恢复时只有看到匹配的 `COMMIT` 才整体应用,这和 [04 类](04-transactions-and-isolation-levels.md) 的事务原子性、[06 类知识点2](06-storage-engine-internals.md) 的 undo 日志是同一件事的两种落地方式。
- **更鲁棒的损坏检测**:阶段4的"是否以 `\n` 结尾"判断只能测出末尾写到一半的情况。真实做法是给每条记录加长度字段和校验和(见 [06 类知识点1"常见坑"](06-storage-engine-internals.md)),不管损坏发生在文件的哪个位置都能被发现。
- **checkpoint,避免每次重启都要重放全部历史**:这个迷你引擎每次"重启"都要把 WAL 从第一行读到最后一行——记录一多,恢复会越来越慢。[06 类知识点3](06-storage-engine-internals.md) 的 checkpoint 机制解决的正是这个问题:定期把内存状态整体快照落盘,并记下"这个时间点之前的修改都已经落盘",下次恢复只需要从最近一次快照开始重放,不用回到最初。
- **并发写入**:这个迷你引擎假设只有一个进程在写。真实存储引擎要处理多个客户端并发写入同一份数据的场景,这属于 [05 类](05-mvcc-and-locking.md) 锁机制和 MVCC 的范畴,不是 WAL 本身要解决的问题。

这几个方向都不实现,是为了让这篇教程聚焦在"WAL 的核心规则本身"这一件事上——真要继续做下去,每一个方向单独展开都够写一整篇。

## 这篇教程展示的方法论

任何一条已完成的深挖系列,都可以用同样的模式产出"教程体"内容:挑几个关联的知识点 → 设计一个真实有用、读者一看就懂价值的小工具 → 分阶段增量实现,每一步都跑起来看到真实效果,而不是一次性甩出完整代码。这是这个格式在 dsa-deep-dive 之外的第二次落地,这次额外验证了一件事:教程体不是只能串同一条系列内部的知识点——这篇同时用到了 database-deep-dive 自己的两个板块(06 类存储引擎、04 类事务),说明只要两个知识点之间有真实的机制关联(这里是"WAL 保证持久性"和"事务保证原子性"共同构成数据库对崩溃场景的完整承诺),跨知识点组装出的小工具依然能站得住,不是生硬拼接。其余系列要不要配套同类文件,是后续单独决定的问题,这里不展开。

---

*创建:2026-07-24*
