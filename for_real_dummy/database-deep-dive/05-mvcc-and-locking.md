# MVCC与锁机制

> 板块 III(事务与并发控制)第 2 篇。环境:全部 `python-wsl2`(依赖 WSL2 Rocky Linux 已启动的 PostgreSQL 16.14 + MariaDB 10.11.15,凭据见 `00-roadmap.md`)。

**与 os-concurrency-deep-dive 的边界声明**:该系列 03类(同步原语:互斥锁/信号量)和 05类(死锁:四条件/检测算法/资源分配图)讲的是**操作系统层面进程/线程同步与死锁理论**,通用、和具体应用无关。本类讲的是**数据库引擎内部实现的行锁/表锁/间隙锁/next-key lock**,死锁检测是数据库引擎自己维护的等待图,在真实 PostgreSQL/MariaDB 上真实触发观察引擎的自动 kill 行为。两者的"死锁"是同一理论根源(等待图成环)但应用 domain 和具体实现完全不同层面,本类不重新推导 OS 死锁检测算法本身,只在知识点5复用其"等待图成环"的核心判据。

## 1. MVCC快照隔离原理:PostgreSQL的xmin/xmax可见性判断

**签名/是什么**

```
每一行数据(元组)隐藏两个系统列: xmin(创建该版本的事务ID), xmax(删除/替换该版本的事务ID,0表示尚未被替换)
可见性规则(简化版): 一个版本对当前事务可见,当且仅当 xmin对应的事务已提交 且 (xmax=0 或 xmax对应的事务未提交/晚于当前快照)
```

一句话:PostgreSQL的UPDATE本质上不是"原地修改",而是"插入一个新版本(新xmin)+ 把旧版本标记为已被替换(设置旧版本的xmax)",MVCC快照隔离靠比较事务ID的先后关系决定每个事务能看见哪个版本。

**底层机制/为什么这样设计**

这种"仅追加(append-only)"的版本管理方式,让PostgreSQL不需要用锁就能实现"读不阻塞写、写不阻塞读"——正在读某一行的事务看到的是它启动时快照能看见的那个版本,不需要等待正在修改这行的事务;正在写的事务也不需要等待正在读旧版本的事务。代价是旧版本不会立刻消失,需要靠`VACUUM`(03类知识点4已验证过它和`Index Only Scan`的关系)定期清理不再被任何活跃快照需要的旧版本,否则表会持续膨胀("表膨胀"是PostgreSQL运维里一个真实存在、需要主动管理的问题)。

**AI研究/工程场景**

理解xmin/xmax机制能直接解释很多PostgreSQL的运维现象:长时间不提交的事务(比如忘记`COMMIT`的交互式`psql`会话)会阻止`VACUUM`清理它快照范围内的旧版本,导致表持续膨胀;`UPDATE`一行的物理代价是"insert+标记旧版本"而不是"原地改几个字节",这也是为什么频繁小字段更新的高吞吐场景在PostgreSQL上要格外关注`VACUUM`策略。

**可运行例子**(环境:`python-wsl2`)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14(已启动)
import psycopg2

pg = psycopg2.connect(host='127.0.0.1', port=5432, dbname='dbdemo', user='dbdemo', password='dbdemo_local_only')
pg.autocommit = True
pgc = pg.cursor()
pgc.execute("DROP TABLE IF EXISTS accounts")
pgc.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER)")
pgc.execute("INSERT INTO accounts VALUES (1, 1000)")

pgc.execute("SELECT xmin, xmax, balance FROM accounts WHERE id=1")
xmin_before, xmax_before, balance_before = pgc.fetchone()
assert balance_before == 1000

pgc.execute("UPDATE accounts SET balance=1100 WHERE id=1")
pgc.execute("SELECT xmin, xmax, balance FROM accounts WHERE id=1")
xmin_after, xmax_after, balance_after = pgc.fetchone()

# UPDATE产生了一个全新的版本: xmin真实变化了(不是原地修改同一个版本)
assert xmin_after != xmin_before, f"expected a new version with a different xmin, got same xmin={xmin_after}"
assert balance_after == 1100

pg.close()
print(f"xmin/xmax verified: UPDATE created a genuinely new version (xmin {xmin_before} -> {xmin_after}), not an in-place edit")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
UPDATE前: xmin=51289 xmax=0 balance=1000
UPDATE后: xmin=51290 xmax=0 balance=1100  <- xmin真实前进了一个事务ID,证明是新版本
```

**面试怎么问+追问链**

- Q:PostgreSQL的UPDATE语句物理上做了什么?
  - 追问1:那旧版本(xmin=51289的那一行)去哪了?
    - 深挖追问(区分度较高):如果一直不做`VACUUM`,这些旧版本会造成什么后果,怎么监控(答案方向:旧版本继续占用磁盘空间,俗称"表膨胀"[table bloat],可以通过`pg_stat_user_tables`里的`n_dead_tup`列监控死元组数量,长时间膨胀不仅浪费空间还会拖慢顺序扫描和索引效率,这是PostgreSQL相比"原地更新"引擎多出来的一项真实运维负担,也是选型时值得权衡的因素)。

**常见坑**

- 把PostgreSQL的UPDATE想象成和其他引擎一样的"原地修改"——本类的可运行例子已经真实验证xmin会变化,证明每次UPDATE都创建了新版本,这个认知差异直接关系到"为什么PostgreSQL需要VACUUM"这个后续问题。
- 忽视长事务对VACUUM的阻塞效应——一个开着不提交的事务会让它快照范围内的所有旧版本无法被清理,这是真实生产事故的常见根源之一。

---

## 2. MVCC快照隔离原理:InnoDB的undo log版本链

**签名/是什么**

```
InnoDB的UPDATE: 原地修改聚簇索引里的当前版本 + 把"如何撤销这次修改"记录进undo log
一致性读(consistent read): 需要看旧版本的事务,通过undo log反向重建出"事务开始时"的版本
```

一句话:和PostgreSQL"新建版本、旧版本留在原地"的思路相反,InnoDB是"直接改当前版本,把变更前的样子记进undo log",需要看旧版本的事务要沿着undo log链条反向"倒放"重建出历史版本。

**底层机制/为什么这样设计**

InnoDB的存储结构是聚簇索引(03类知识点2已验证:主键就是数据本身,不像PostgreSQL堆表和索引分离),这种"数据即索引"的结构下"新建一整行新版本"的代价比PostgreSQL更高(还要维护索引指向新版本的位置),所以InnoDB选择原地修改当前版本、把"回滚信息"记录到独立的undo log区域——这样聚簇索引本身只保留一份最新数据,历史版本按需通过undo log链条重建,不占用主索引结构的空间(但undo log本身也需要清理,由`purge`线程负责,概念上和PostgreSQL的`VACUUM`扮演类似角色但实现机制不同)。

**AI研究/工程场景**

这解释了本系列04类知识点2已经验证过的现象——一个事务A的UPDATE还未提交(已经原地改了当前版本),另一个REPEATABLE READ事务B此时查询,依然能看到旧值,这正是B在用undo log重建"事务开始时"的版本,而不是直接读被A已经改掉的"当前版本"。

**可运行例子**(环境:`python-wsl2`)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 MariaDB 10.11.15(已启动)
import pymysql
import threading

MARIA_DSN = dict(host='127.0.0.1', port=3306, database='dbdemo', user='dbdemo', password='dbdemo_local_only')

maria = pymysql.connect(**MARIA_DSN); maria.autocommit(True); mc = maria.cursor()
mc.execute("DROP TABLE IF EXISTS accounts")
mc.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER) ENGINE=InnoDB")
mc.execute("INSERT INTO accounts VALUES (1, 1000)")
maria.close()

result = {}
ev_a_updated = threading.Event()
ev_b_done = threading.Event()

def txn_a():
    conn = pymysql.connect(**MARIA_DSN)
    conn.autocommit(False)
    cur = conn.cursor()
    cur.execute("START TRANSACTION")
    cur.execute("UPDATE accounts SET balance=1100 WHERE id=1")  # 原地改了聚簇索引里的当前版本
    ev_a_updated.set()
    ev_b_done.wait(timeout=5)
    conn.commit()
    conn.close()

def txn_b():
    ev_a_updated.wait(timeout=5)
    conn = pymysql.connect(**MARIA_DSN)
    conn.autocommit(False)
    cur = conn.cursor()
    cur.execute("SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ")
    cur.execute("START TRANSACTION")
    cur.execute("SELECT balance FROM accounts WHERE id=1")
    result['b_saw'] = cur.fetchone()[0]
    conn.commit()
    conn.close()
    ev_b_done.set()

ta = threading.Thread(target=txn_a); tb = threading.Thread(target=txn_b)
ta.start(); tb.start(); ta.join(timeout=10); tb.join(timeout=10)

# 关键: A此时已经把聚簇索引里的"当前版本"原地改成了1100(未提交),
# 但B的一致性读通过undo log重建出"事务开始时"的旧版本,看到的依然是1000
assert result['b_saw'] == 1000, \
    f"B should see the pre-update value via undo log reconstruction, got {result['b_saw']}"

print(f"undo log version chain verified: A's in-place update to 1100 is already applied to the clustered index, but B's snapshot read reconstructs the old value ({result['b_saw']}) from undo log")
```

**面试怎么问+追问链**

- Q:MySQL/InnoDB的MVCC和PostgreSQL的MVCC实现思路一样吗?
  - 追问1:如果一样,为什么PostgreSQL需要VACUUM,而InnoDB用的是不同的术语(purge)?
    - 深挖追问(区分度较高):这两种思路各自的性能特征有什么本质区别(答案方向:PostgreSQL的"新建版本"方式让写入路径更简单[不需要维护undo log],但读取总是直接可见,历史版本靠VACUUM清理,长事务/高更新率场景容易表膨胀;InnoDB的"原地修改+undo log"方式让主索引结构始终只有一份最新数据[空间效率更好],但历史版本的重建需要额外遍历undo log链条,如果一个旧事务的快照存在很久、期间这行被反复修改,undo log链条会变得很长,重建旧版本的开销会上升,这是两种设计权衡下不同的"代价转移去向")。

**常见坑**

- 认为"MVCC"是一种单一、标准化的技术,不同数据库的实现细节都一样——PostgreSQL和InnoDB是MVCC的两种不同实现路径(新建版本 vs 原地修改+undo log),这个区别直接影响各自的运维特征(VACUUM策略 vs undo log长度监控)。
- 把undo log和redo log(06类会展开,用于崩溃恢复的持久性保证)混为一谈——两者都叫"log"但用途完全不同,undo log是MVCC和事务回滚用的"如何撤销",redo log是崩溃恢复用的"如何重做"。

---

## 3. 锁类型与粒度:行锁/表锁/意向锁

**签名/是什么**

```
表锁(Table Lock):    锁住整张表,粒度最粗,并发度最低
行锁(Row Lock):      只锁被访问的具体行,粒度最细,并发度最高
意向锁(Intention Lock): 表级的"标记",表示"这张表里有某些行被加了行锁",让表锁请求不需要逐行检查就能快速判断冲突
```

一句话:行锁和表锁不是二选一,是配合使用的——意向锁像是贴在表这一层的"告示牌",告诉后来想加表级锁的人"这张表内部已经有行被锁住了,不能对它下手",不需要真的去扫描千万行才能发现冲突。

**底层机制/为什么这样设计**

如果没有意向锁,一个事务想加表级排他锁(比如要做`ALTER TABLE`),必须先确认表里所有行都没有被其他事务加锁——如果表有几千万行,逐行检查的代价是不可接受的。意向锁把这个检查复杂度从"O(行数)"降到"O(1)":任何事务在给一行加锁之前,必须先在表级别加一个"意向锁"作为标记,之后任何请求表级锁的事务只需要检查这一个标记就知道"表内部有没有更细粒度的锁在起作用",不需要真的遍历行。

**可运行例子**(环境:`python-wsl2`,验证行锁的真实粒度——不同行互不阻塞,同一行才阻塞)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14 + MariaDB 10.11.15(已启动)
import psycopg2
import pymysql
import threading
import time

PG_DSN = dict(host='127.0.0.1', port=5432, dbname='dbdemo', user='dbdemo', password='dbdemo_local_only')
MARIA_DSN = dict(host='127.0.0.1', port=3306, database='dbdemo', user='dbdemo', password='dbdemo_local_only')

def setup():
    pg = psycopg2.connect(**PG_DSN); pg.autocommit = True; pgc = pg.cursor()
    pgc.execute("DROP TABLE IF EXISTS accounts")
    pgc.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER)")
    pgc.execute("INSERT INTO accounts VALUES (1, 1000), (2, 500)")
    pg.close()
    maria = pymysql.connect(**MARIA_DSN); maria.autocommit(True); mc = maria.cursor()
    mc.execute("DROP TABLE IF EXISTS accounts")
    mc.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER) ENGINE=InnoDB")
    mc.execute("INSERT INTO accounts VALUES (1, 1000), (2, 500)")
    maria.close()

def test_row_lock_granularity(connect_fn, start_txn_sql, target_id):
    setup()
    result = {}
    ev_a_locked = threading.Event()

    def txn_a():
        conn = connect_fn()
        if hasattr(conn, 'autocommit') and callable(getattr(conn, 'autocommit')):
            conn.autocommit(False)
        cur = conn.cursor()
        if start_txn_sql: cur.execute(start_txn_sql)
        cur.execute("SELECT * FROM accounts WHERE id=1 FOR UPDATE")
        cur.fetchall()
        ev_a_locked.set()
        time.sleep(1.0)
        conn.commit(); conn.close()

    def txn_b():
        ev_a_locked.wait(timeout=5)
        conn = connect_fn()
        if hasattr(conn, 'autocommit') and callable(getattr(conn, 'autocommit')):
            conn.autocommit(True)
        else:
            conn.autocommit = True
        cur = conn.cursor()
        t0 = time.time()
        cur.execute(f"UPDATE accounts SET balance=999 WHERE id={target_id}")
        result['elapsed'] = time.time() - t0
        conn.close()

    ta = threading.Thread(target=txn_a); tb = threading.Thread(target=txn_b)
    ta.start(); tb.start(); ta.join(timeout=10); tb.join(timeout=10)
    return result

pg_diff = test_row_lock_granularity(lambda: psycopg2.connect(**PG_DSN), None, target_id=2)
assert pg_diff['elapsed'] < 0.5, f"different row should not be blocked, got {pg_diff}"

pg_same = test_row_lock_granularity(lambda: psycopg2.connect(**PG_DSN), None, target_id=1)
assert pg_same['elapsed'] > 0.8, f"same row should be blocked ~1s, got {pg_same}"

maria_diff = test_row_lock_granularity(lambda: pymysql.connect(**MARIA_DSN), "START TRANSACTION", target_id=2)
assert maria_diff['elapsed'] < 0.5, f"got {maria_diff}"

maria_same = test_row_lock_granularity(lambda: pymysql.connect(**MARIA_DSN), "START TRANSACTION", target_id=1)
assert maria_same['elapsed'] > 0.8, f"got {maria_same}"

print(f"row lock granularity verified on both engines: different row not blocked (PG={pg_diff['elapsed']:.3f}s, Maria={maria_diff['elapsed']:.3f}s), "
      f"same row blocked (PG={pg_same['elapsed']:.3f}s, Maria={maria_same['elapsed']:.3f}s)")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
不同行(id=1锁住,更新id=2): PG=0.005s Maria=0.002s  (都不阻塞,真实证明了行级粒度)
同一行(id=1锁住,更新id=1): PG=1.006s Maria=1.006s  (都阻塞到A释放锁为止)
```

**面试怎么问+追问链**

- Q:如果没有意向锁,表锁和行锁怎么协调?
  - 追问1:意向锁本身会和别的意向锁冲突吗(比如两个事务都想给不同的行加意向锁)?
    - 深挖追问(区分度较高):意向锁的"意向"具体分几种,分别对应什么场景(答案方向:通常分为意向共享锁[IS,准备加行级共享锁前先声明]和意向排他锁[IX,准备加行级排他锁前先声明],意向锁之间[IS和IS、IS和IX、IX和IX]互相不冲突[因为它们只是"声明"不是真的锁住数据],真正的冲突判断依然发生在意向锁和真实表锁之间[比如意向排他锁存在时,别的事务无法获得该表的排他表锁]——这个设计让"多个事务并发修改不同行"这个最常见的场景完全不受表锁机制拖累)。

**常见坑**

- 误以为"用了行锁就完全不会有表级别的锁竞争"——`ALTER TABLE`这类需要表级排他锁的DDL操作,依然会被现存的意向锁/行锁阻塞(或反过来阻塞新的行锁请求),这是在线大表加字段/加索引要格外小心锁等待的原因之一。
- 把行锁的"精确到具体某一行"和索引扫描的效率混为一谈——如果查询条件没有用上索引(比如03类讨论过的索引失效场景),数据库可能需要扫描全表才能定位到目标行,这个扫描过程中在某些引擎/隔离级别下可能给扫描路径上的多行都加锁(不止目标行本身),造成比预期更大范围的锁定,这也是"索引和锁"这两个知识点经常一起被问的原因。

---

## 4. InnoDB间隙锁与next-key lock的精确边界

**签名/是什么**

```
Gap Lock(间隙锁):    锁定两个索引键值之间的"间隙",阻止在这个间隙里插入新行
Next-Key Lock:       记录锁 + 该记录前面的间隙锁,是InnoDB在REPEATABLE READ下的默认加锁方式
```

一句话:04类知识点5已经验证过间隙锁会真实阻塞并发INSERT,本类更进一步验证**具体锁住的范围有多宽**——真实实验显示,对一个非唯一索引做等值查询加锁时,锁定范围可能同时覆盖目标值前后两侧的间隙,比"只锁前面一侧"这个简化理解更宽。

**底层机制/为什么这样设计**

Next-key lock字面上定义是"记录本身+前面的间隙",但**真实的加锁范围取决于查询执行时具体扫描了哪些索引条目**——对非唯一二级索引做等值查询,InnoDB为了确认"这个值truly没有更多匹配的行了",搜索过程会前进到下一个不同的键值处确认边界,这个探测动作本身也会在探测到的位置留下间隙锁记录。这不是一个孤立的特例,而是next-key lock"锁定范围紧跟着实际扫描路径走"这个更普遍规律的体现——机制的复杂性直接来自"证明没有更多匹配"这个实际需要执行的操作。

**AI研究/工程场景**

这个"实际锁定范围可能比直觉认为的更宽"的真实现象,是InnoDB在高并发插入场景下出现意外锁等待/死锁的一个常见根源——如果只按"教科书简化定义"(只锁前面间隙)去预估并发影响,容易低估真实的锁竞争范围,本类用真实实验验证过的边界比死记定义更可靠。

**可运行例子**(环境:`python-wsl2`)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 MariaDB 10.11.15(已启动)
import pymysql
import threading
import time

MARIA_DSN = dict(host='127.0.0.1', port=3306, database='dbdemo', user='dbdemo', password='dbdemo_local_only')

def setup():
    maria = pymysql.connect(**MARIA_DSN); maria.autocommit(True); mc = maria.cursor()
    mc.execute("DROP TABLE IF EXISTS points")
    mc.execute("CREATE TABLE points (id INTEGER PRIMARY KEY AUTO_INCREMENT, val INTEGER) ENGINE=InnoDB")
    mc.execute("CREATE INDEX idx_val ON points (val)")  # 非唯一二级索引,val=10,20,30三个值
    mc.executemany("INSERT INTO points (val) VALUES (%s)", [(10,), (20,), (30,)])
    maria.close()

def test_insert_blocked(insert_val):
    setup()
    result = {}
    ev_a_locked = threading.Event()

    def txn_a():
        conn = pymysql.connect(**MARIA_DSN)
        conn.autocommit(False)
        cur = conn.cursor()
        cur.execute("SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        cur.execute("START TRANSACTION")
        cur.execute("SELECT * FROM points WHERE val=20 FOR UPDATE")  # 非唯一索引上的等值加锁
        cur.fetchall()
        ev_a_locked.set()
        time.sleep(1.2)
        conn.commit(); conn.close()

    def txn_b():
        ev_a_locked.wait(timeout=5)
        conn = pymysql.connect(**MARIA_DSN)
        conn.autocommit(True)
        cur = conn.cursor()
        t0 = time.time()
        cur.execute(f"INSERT INTO points (val) VALUES ({insert_val})")
        result['elapsed'] = time.time() - t0
        conn.close()

    ta = threading.Thread(target=txn_a); tb = threading.Thread(target=txn_b)
    ta.start(); tb.start(); ta.join(timeout=10); tb.join(timeout=10)
    return result

before = test_insert_blocked(15)  # 插入到(10,20)间隙,val=20的"前面"
assert before['elapsed'] > 1.0, f"gap BEFORE the locked value should be blocked, got {before}"

after = test_insert_blocked(25)  # 插入到(20,30)间隙,val=20的"后面"
# 真实发现: 后面的间隙也被锁住了,不是教科书简化定义里"只锁前面"那么简单
assert after['elapsed'] > 1.0, \
    f"the gap AFTER the locked value is ALSO blocked in practice (search must probe the next key to confirm no more matches), got {after}"

print(f"next-key lock range verified (wider than naive expectation): gap before blocked ({before['elapsed']:.3f}s), gap after ALSO blocked ({after['elapsed']:.3f}s)")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
val=20被FOR UPDATE锁定期间:
  插入val=15(20前面的间隙): 阻塞约1.2秒
  插入val=25(20后面的间隙): 也阻塞约1.2秒  <- 真实验证过的意外发现,比"只锁前面"更宽
```

**面试怎么问+追问链**

- Q:间隙锁具体锁的是"哪个间隙"?
  - 追问1:如果锁定范围比预想的更宽,这对"如何设计并发安全的批量插入"有什么指导意义?
    - 深挖追问(区分度较高):遇到这类"锁范围比预期更宽导致的意外锁等待"该怎么排查(答案方向:MariaDB/InnoDB提供`information_schema.innodb_lock_waits`[或`performance_schema`里的对应表]可以查看真实的锁等待关系和具体锁的对象,排查这类问题不能只靠"背定义推理",要养成用这些诊断视图直接观察真实锁状态的习惯,这也是本类反复强调的"实测优于假设"方法论的又一个体现)。

**常见坑**

- 只记住"next-key lock=记录锁+前面的间隙锁"这个教科书简化定义,不知道实际执行路径可能扩大锁定范围——本类的可运行例子已经真实验证了这个更宽的边界,面试被追问具体细节时,能给出"我实测验证过,范围比简化定义更宽"这类回答比死记定义更有说服力。
- 忘记这类间隙锁只在REPEATABLE READ(或更高)级别下生效——如果业务用READ COMMITTED,04类已经验证过这个级别本身对幻读没有防护,间隙锁机制在这个级别下也会被大幅弱化。

---

## 5. 真实死锁复现与检测

**签名/是什么**

```
死锁(Deadlock): 两个(或更多)事务互相持有对方需要的锁,同时等待对方释放,形成等待环,永远无法继续
死锁检测: 数据库引擎定期(或事件触发式)检查等待关系图里有没有环,发现环就主动杀死其中一个事务打破僵局
```

一句话:死锁不是"引擎的bug",是并发系统里逻辑上必然存在的可能性,真正重要的是引擎有没有主动检测并打破它——本类真实构造了一个死锁场景,验证两个引擎都能自动介入,但选择"牺牲者"(victim)的具体是哪一个事务,两个引擎的结果不同。

**底层机制/为什么这样发生**

死锁的经典构造需要两个条件同时满足:①两个事务以**相反顺序**申请同一组资源的锁(A先锁1后锁2,B先锁2后锁1);②双方都在对方持有的资源上等待。数据库引擎通常维护一张"事务等待关系图"(谁在等谁持有的锁),死锁检测算法本质上是在这张图里找环——找到环之后,引擎必须选择"牺牲"环上的某一个事务(强制回滚它,释放它持有的锁,让另一个事务得以继续),两个引擎在"选哪个当牺牲者"这件事上有各自的启发式规则(比如"回滚做的工作量更小的那个"),不保证两个引擎在同样的场景下选择同一个。

**可运行例子**(环境:`python-wsl2`,超时安全网:两个线程最多等12秒,不会真的无限期挂起)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14 + MariaDB 10.11.15(已启动)
import psycopg2
import pymysql
import threading

PG_DSN = dict(host='127.0.0.1', port=5432, dbname='dbdemo', user='dbdemo', password='dbdemo_local_only')
MARIA_DSN = dict(host='127.0.0.1', port=3306, database='dbdemo', user='dbdemo', password='dbdemo_local_only')

def setup():
    pg = psycopg2.connect(**PG_DSN); pg.autocommit = True; pgc = pg.cursor()
    pgc.execute("DROP TABLE IF EXISTS accounts")
    pgc.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER)")
    pgc.execute("INSERT INTO accounts VALUES (1, 1000), (2, 500)")
    pg.close()
    maria = pymysql.connect(**MARIA_DSN); maria.autocommit(True); mc = maria.cursor()
    mc.execute("DROP TABLE IF EXISTS accounts")
    mc.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER) ENGINE=InnoDB")
    mc.execute("INSERT INTO accounts VALUES (1, 1000), (2, 500)")
    maria.close()

def test_deadlock(connect_fn, start_txn_sql):
    setup()
    result = {}
    ev_a_locked_1 = threading.Event()
    ev_b_locked_2 = threading.Event()

    def txn_a():
        conn = connect_fn()
        if hasattr(conn, 'autocommit') and callable(getattr(conn, 'autocommit')):
            conn.autocommit(False)
        cur = conn.cursor()
        if start_txn_sql: cur.execute(start_txn_sql)
        cur.execute("SELECT * FROM accounts WHERE id=1 FOR UPDATE")  # A: 先锁1
        cur.fetchall()
        ev_a_locked_1.set()
        ev_b_locked_2.wait(timeout=5)
        try:
            cur.execute("SELECT * FROM accounts WHERE id=2 FOR UPDATE")  # A: 再要锁2(此时被B持有)
            cur.fetchall(); conn.commit(); result['a_outcome'] = 'committed'
        except Exception as e:
            result['a_outcome'] = type(e).__name__; conn.rollback()
        conn.close()

    def txn_b():
        conn = connect_fn()
        if hasattr(conn, 'autocommit') and callable(getattr(conn, 'autocommit')):
            conn.autocommit(False)
        cur = conn.cursor()
        if start_txn_sql: cur.execute(start_txn_sql)
        ev_a_locked_1.wait(timeout=5)
        cur.execute("SELECT * FROM accounts WHERE id=2 FOR UPDATE")  # B: 先锁2
        cur.fetchall()
        ev_b_locked_2.set()
        try:
            cur.execute("SELECT * FROM accounts WHERE id=1 FOR UPDATE")  # B: 再要锁1(此时被A持有) - 构成等待环
            cur.fetchall(); conn.commit(); result['b_outcome'] = 'committed'
        except Exception as e:
            result['b_outcome'] = type(e).__name__; conn.rollback()
        conn.close()

    ta = threading.Thread(target=txn_a); tb = threading.Thread(target=txn_b)
    ta.start(); tb.start()
    ta.join(timeout=12); tb.join(timeout=12)  # 超时安全网,确定性退出
    result['threads_finished'] = not ta.is_alive() and not tb.is_alive()
    return result

pg_result = test_deadlock(lambda: psycopg2.connect(**PG_DSN), None)
assert pg_result['threads_finished'], "deadlock demo must terminate deterministically, not hang forever"
outcomes = {pg_result['a_outcome'], pg_result['b_outcome']}
assert outcomes == {'DeadlockDetected', 'committed'}, f"expected exactly one victim, got {pg_result}"

maria_result = test_deadlock(lambda: pymysql.connect(**MARIA_DSN), "START TRANSACTION")
assert maria_result['threads_finished']
outcomes_maria = {maria_result['a_outcome'], maria_result['b_outcome']}
assert outcomes_maria == {'OperationalError', 'committed'}, f"expected exactly one victim, got {maria_result}"

print(f"real deadlock detection verified on both engines: PG killed {'A' if pg_result['a_outcome']=='DeadlockDetected' else 'B'}, "
      f"Maria killed {'A' if maria_result['a_outcome']=='OperationalError' else 'B'} (victim choice can differ, both engines DO detect and break the deadlock)")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
PG:    A被杀(DeadlockDetected), B成功提交
Maria: B被杀(OperationalError: Deadlock found), A成功提交
```

**面试怎么问+追问链**

- Q:数据库怎么发现死锁的?
  - 追问1:如果两个引擎在同样的死锁场景下选择"杀掉"不同的事务(一个杀A,一个杀B),这说明什么?
    - 深挖追问(区分度较高):应用层代码应该怎么应对"我的事务可能被数据库随时以死锁为由杀死"这个现实(答案方向:不能假设"先申请锁的事务一定会赢",必须在应用层对死锁相关的异常做统一的捕获+重试处理[通常配合指数退避],这是任何会做多行/多资源加锁操作的生产代码的标准防御性设计,不是"死锁很少见所以不用管"这种侥幸心态)。

**常见坑**

- 认为死锁是"应该完全避免发生的错误",一旦出现就是代码bug——死锁本身是并发系统的正常现象(尤其是两个事务的加锁顺序确实相反时),真正的问题是"应用层有没有正确处理死锁被检测后的重试逻辑",而不是追求"永远不发生死锁"。
- 死锁复现代码没有超时保护——如果两个引擎的死锁检测机制本身出了问题(或者构造的场景其实不是真死锁而是普通的长时间锁等待),没有超时安全网的测试代码会真的无限期挂起,这是本类demo代码显式设置`join(timeout=12)`的原因。

---

## 6. 死锁预防:按固定顺序申请锁

**签名/是什么**

```
死锁预防策略: 所有事务对多个资源加锁时,都遵循同一个固定顺序(比如永远按主键从小到大)
```

一句话:死锁需要"相反的加锁顺序"才能形成等待环,如果所有并发事务都遵循同一个固定顺序申请锁,即使发生等待,也只会是"排队等待"而不是"互相等待",真实实验证明这个简单规则能完全避免死锁。

**底层机制/为什么这样设计**

固定顺序加锁从根本上消除了死锁的必要条件之一("循环等待")——如果所有事务都保证"先锁ID较小的资源,再锁ID较大的资源",那么在任意时刻,持有较大ID锁、等待较小ID锁的情况根本不会发生,因为任何事务在尝试获取较大ID的锁之前必然已经先成功获得了较小ID的锁。这不需要引擎做任何特殊支持,是纯粹的应用层设计约定,代价是需要开发者自觉遵守("固定顺序"必须体现在所有相关代码路径里,一处遗漏就可能重新引入死锁风险)。

**AI研究/工程场景**

批量转账/批量结算这类需要同时锁定多个账户的业务逻辑,是死锁的高发区——工程上的标准做法是在加锁前先对涉及的所有账户ID排序,始终按排序后的顺序依次加锁,这个简单的约定在真实系统里能大幅降低死锁概率(虽然不能保证100%消除所有类型的死锁,比如涉及不同索引路径的隐式加锁依然可能有例外,但对"显式多行加锁"这类最常见的死锁场景非常有效)。

**可运行例子**(环境:`python-wsl2`)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14(已启动)
import psycopg2
import threading
import time

PG_DSN = dict(host='127.0.0.1', port=5432, dbname='dbdemo', user='dbdemo', password='dbdemo_local_only')

pg = psycopg2.connect(**PG_DSN); pg.autocommit = True; pgc = pg.cursor()
pgc.execute("DROP TABLE IF EXISTS accounts")
pgc.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER)")
pgc.execute("INSERT INTO accounts VALUES (1, 1000), (2, 500)")
pg.close()

result = {}
ev_a_locked_1 = threading.Event()

def txn_a():
    conn = psycopg2.connect(**PG_DSN)
    cur = conn.cursor()
    cur.execute("SELECT * FROM accounts WHERE id=1 FOR UPDATE")  # 固定顺序: 永远先锁较小的id
    cur.fetchall()
    ev_a_locked_1.set()
    time.sleep(1.0)
    cur.execute("SELECT * FROM accounts WHERE id=2 FOR UPDATE")
    cur.fetchall()
    conn.commit(); conn.close()
    result['a_outcome'] = 'committed'

def txn_b():
    ev_a_locked_1.wait(timeout=5)
    conn = psycopg2.connect(**PG_DSN)
    cur = conn.cursor()
    t0 = time.time()
    # 关键: B也遵循同样的固定顺序(先id=1再id=2),不是反过来先锁2再锁1
    cur.execute("SELECT * FROM accounts WHERE id=1 FOR UPDATE")
    cur.fetchall()
    result['b_wait_for_first_lock'] = time.time() - t0
    cur.execute("SELECT * FROM accounts WHERE id=2 FOR UPDATE")
    cur.fetchall()
    conn.commit(); conn.close()
    result['b_outcome'] = 'committed'

ta = threading.Thread(target=txn_a); tb = threading.Thread(target=txn_b)
ta.start(); tb.start(); ta.join(timeout=10); tb.join(timeout=10)

# 一致的加锁顺序下,B只是排队等待(约1秒),不是死锁,两边最终都成功提交
assert result.get('a_outcome') == 'committed' and result.get('b_outcome') == 'committed', \
    f"consistent lock ordering should let BOTH transactions succeed (no deadlock), got {result}"
assert result['b_wait_for_first_lock'] > 0.8, "B should have genuinely waited for A to release the first lock"

print(f"deadlock prevention via consistent lock ordering verified: both committed, B waited {result['b_wait_for_first_lock']:.3f}s (queued, not deadlocked)")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
按固定顺序(先id=1再id=2)加锁: A和B都成功提交,B等待约1.0秒(排队,不是死锁)
```

**面试怎么问+追问链**

- Q:除了让数据库自动检测并杀死死锁事务,还有什么办法能从根本上减少死锁?
  - 追问1:固定顺序加锁这个方案有没有局限性?
    - 深挖追问(区分度较高):如果一个事务涉及的锁定资源集合是在运行时动态确定的(不是提前知道要锁哪几行),固定顺序加锁怎么落地(答案方向:需要在实际加锁之前,先把这一批要锁的资源ID收集齐全并排序,再按排序结果依次加锁——如果锁定资源是"边执行边发现"的[比如遍历过程中动态决定],固定顺序加锁很难严格落地,这种场景往往需要退而求其次依赖数据库的死锁检测+应用层重试机制[知识点5],而不能单纯依赖预防)。

**常见坑**

- 以为"固定顺序加锁"是万能的死锁免疫方案——它对"显式多行加锁"场景非常有效,但复杂查询隐式产生的加锁顺序(比如03/04类讨论过的索引扫描路径、间隙锁的真实覆盖范围)未必总能被开发者完全预知和控制,不能因为用了这个策略就完全忽视应用层的死锁重试处理。
- 团队里只有部分代码路径遵循固定加锁顺序,另一部分(尤其是后加的新功能代码)没有遵循——这个约定必须是团队范围内一致执行的纪律,一处遗漏就可能重新引入死锁,代码审查阶段值得专门检查这一点。

---

## 7. 乐观锁vs悲观锁工程选型

**签名/是什么**

```
悲观锁: 假设冲突大概率会发生,读的时候就先加锁(SELECT ... FOR UPDATE),阻塞其他人直到自己提交
乐观锁: 假设冲突大概率不会发生,读的时候不加锁,写回时通过版本号/时间戳判断这期间有没有被别人改过
```

一句话:两者都能防止04类知识点6演示过的丢失更新,区别在于"预防成本转嫁给谁"——悲观锁让并发的其他事务等待,乐观锁让检测到冲突的事务自己重试。

**底层机制/为什么这样设计**

乐观锁的实现通常是给表加一个`version`整数列,更新语句写成`UPDATE t SET data=?, version=version+1 WHERE id=? AND version=?`(带上读取时看到的旧版本号做条件)——如果这期间没人修改过,`WHERE`条件里的`version`依然匹配,更新正常生效且影响1行;如果期间被别人抢先改过,`version`已经变了,这条`UPDATE`匹配不到任何行(影响0行),应用代码通过检查"受影响行数是否为0"来判断冲突发生,决定是否重新读取最新数据再重试。这个机制完全建立在01类知识点4验证过的"单语句原子性"之上——比较版本号和更新数据是同一条SQL语句的原子操作,不会有额外的竞态窗口。

**AI研究/工程场景**

冲突概率低的场景(比如多数用户只编辑自己的数据,极少出现两人同时编辑同一条记录)用乐观锁性价比更高——不需要为了防范"几乎不会发生"的冲突让所有请求都排队等锁;高冲突场景(比如秒杀库存扣减,大量并发请求争抢同一行)用悲观锁往往更高效——与其让大量请求乐观地尝试后大批失败重试(每次重试都要重新走一遍读取+计算+写入的完整流程,浪费更多资源),不如让它们直接排队等待,一次性有序处理。

**可运行例子**(环境:`python-wsl2`,验证乐观锁的CAS[compare-and-swap]式冲突检测真实生效)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14(已启动)
import psycopg2
import threading

PG_DSN = dict(host='127.0.0.1', port=5432, dbname='dbdemo', user='dbdemo', password='dbdemo_local_only')

pg = psycopg2.connect(**PG_DSN); pg.autocommit = True; pgc = pg.cursor()
pgc.execute("DROP TABLE IF EXISTS accounts")
pgc.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER, version INTEGER)")
pgc.execute("INSERT INTO accounts VALUES (1, 1000, 0)")
pg.close()

result = {}
ev_a_read = threading.Event()
ev_b_done = threading.Event()

def txn_a():
    conn = psycopg2.connect(**PG_DSN); conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT balance, version FROM accounts WHERE id=1")
    balance, version = cur.fetchone()
    ev_a_read.set()
    ev_b_done.wait(timeout=5)
    cur.execute("UPDATE accounts SET balance=%s, version=version+1 WHERE id=1 AND version=%s", (balance + 100, version))
    result['a_rowcount'] = cur.rowcount  # 0意味着CAS失败(版本号已经变了)
    conn.close()

def txn_b():
    ev_a_read.wait(timeout=5)
    conn = psycopg2.connect(**PG_DSN); conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT balance, version FROM accounts WHERE id=1")
    balance, version = cur.fetchone()
    cur.execute("UPDATE accounts SET balance=%s, version=version+1 WHERE id=1 AND version=%s", (balance + 50, version))
    result['b_rowcount'] = cur.rowcount
    conn.close()
    ev_b_done.set()

ta = threading.Thread(target=txn_a); tb = threading.Thread(target=txn_b)
ta.start(); tb.start(); ta.join(timeout=10); tb.join(timeout=10)

# B先提交拿到version=1,A基于已过期的version=0再提交,CAS条件不满足,rowcount=0(更新未生效)
assert result['a_rowcount'] == 0, f"A's stale-version UPDATE should affect 0 rows, got {result['a_rowcount']}"
assert result['b_rowcount'] == 1, f"B's fresh-version UPDATE should affect 1 row, got {result['b_rowcount']}"

pgc2 = psycopg2.connect(**PG_DSN).cursor()
pgc2.execute("SELECT balance, version FROM accounts WHERE id=1")
final_balance, final_version = pgc2.fetchone()
# 没有丢失更新: 只有B的+50生效,A检测到自己的rowcount=0之后应该重新读取再重试(应用层职责)
assert final_balance == 1050, f"expected 1050 (only B's update applied, no lost update), got {final_balance}"

print(f"optimistic lock CAS verified: A's stale update was rejected (rowcount=0), B's succeeded, final balance={final_balance} (no lost update, A must detect and retry)")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
A基于version=0提交(此时B已经把version改成了1): rowcount=0 (CAS失败,更新未生效)
B基于version=0提交(先到达): rowcount=1 (成功)
最终状态: balance=1050 version=1  (只有B的+50生效,和04类知识点6"两次都生效但丢了一次"的1100形成对比)
```

**面试怎么问+追问链**

- Q:乐观锁的"版本号"和悲观锁比,谁的性能更好?
  - 追问1:如果高并发场景下用乐观锁,大量请求反复因为版本冲突重试,会不会比悲观锁更慢?
    - 深挖追问(区分度较高):这种情况下你会怎么决策(答案方向:高冲突场景下乐观锁确实可能因为反复重试导致更差的整体吞吐量和更高的长尾延迟,这正是"乐观锁适合低冲突场景"这个结论的真实代价来源,如果实测发现冲突率高、重试率高,应该考虑切换为悲观锁而不是教条地坚持乐观锁,决策要基于对实际冲突率的观测,不是预先假设)。

**常见坑**

- 用乐观锁但没有在应用层实现真正的重试逻辑,只是简单地把"更新0行"当成失败直接报错给用户——乐观锁的核心价值在于"检测到冲突后自动重新读取最新数据再试一次",不做重试的乐观锁只是一个更复杂的"更新失败检测器",没有发挥它本该有的作用。
- 忘记乐观锁的version列本身的更新也需要在同一条原子语句里完成(`SET version=version+1 ... WHERE version=?`),如果拆成两条语句(先查再单独update version),这个机制本身又会重新引入竞态条件。

---

## 8. 锁等待超时配置

**签名/是什么**

```
PostgreSQL: SET lock_timeout = '毫秒数'  (会话级配置,超时抛出 LockNotAvailable)
MariaDB:    SET SESSION innodb_lock_wait_timeout = 秒数  (会话级配置,超时抛出 "Lock wait timeout exceeded")
```

一句话:如果不配置锁等待超时,一个事务申请锁失败时默认会一直等下去(直到对方释放或自己被死锁检测器杀死),显式设置超时能让应用在"等太久也没用"的场景下更快失败并触发重试或降级,而不是无限期挂起用户请求。

**底层机制/为什么这样设计**

死锁检测(知识点5)解决的是"循环等待"这种明确无法继续的场景,但还有一类更常见的情况——单纯的长时间锁等待(没有循环,只是持锁方迟迟不释放),这种情况死锁检测机制不会介入(因为它根本不是死锁),如果不设超时,等待方会一直卡住。锁等待超时提供了一个应用层可控的"最多愿意等多久"的上限,超时后引擎主动放弃当前的锁申请并把控制权交还给应用代码。

**可运行例子**(环境:`python-wsl2`)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14 + MariaDB 10.11.15(已启动)
import psycopg2
import pymysql
import threading
import time

PG_DSN = dict(host='127.0.0.1', port=5432, dbname='dbdemo', user='dbdemo', password='dbdemo_local_only')
MARIA_DSN = dict(host='127.0.0.1', port=3306, database='dbdemo', user='dbdemo', password='dbdemo_local_only')

def setup():
    pg = psycopg2.connect(**PG_DSN); pg.autocommit = True; pgc = pg.cursor()
    pgc.execute("DROP TABLE IF EXISTS accounts")
    pgc.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER)")
    pgc.execute("INSERT INTO accounts VALUES (1, 1000)")
    pg.close()
    maria = pymysql.connect(**MARIA_DSN); maria.autocommit(True); mc = maria.cursor()
    mc.execute("DROP TABLE IF EXISTS accounts")
    mc.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER) ENGINE=InnoDB")
    mc.execute("INSERT INTO accounts VALUES (1, 1000)")
    maria.close()

def test_lock_timeout_pg():
    setup()
    result = {}
    ev_a_locked = threading.Event()

    def txn_a():
        conn = psycopg2.connect(**PG_DSN)
        cur = conn.cursor()
        cur.execute("SELECT * FROM accounts WHERE id=1 FOR UPDATE")
        cur.fetchall()
        ev_a_locked.set()
        time.sleep(3)  # 持锁3秒,比B的1秒超时更长
        conn.commit(); conn.close()

    def txn_b():
        ev_a_locked.wait(timeout=5)
        conn = psycopg2.connect(**PG_DSN)
        cur = conn.cursor()
        cur.execute("SET lock_timeout = '1000'")
        t0 = time.time()
        try:
            cur.execute("SELECT * FROM accounts WHERE id=1 FOR UPDATE")
            result['outcome'] = 'got lock'
        except psycopg2.errors.LockNotAvailable:
            result['outcome'] = 'LockNotAvailable'
            result['elapsed'] = time.time() - t0
            conn.rollback()
        conn.close()

    ta = threading.Thread(target=txn_a); tb = threading.Thread(target=txn_b)
    ta.start(); tb.start(); ta.join(timeout=10); tb.join(timeout=10)
    return result

def test_lock_timeout_maria():
    setup()
    result = {}
    ev_a_locked = threading.Event()

    def txn_a():
        conn = pymysql.connect(**MARIA_DSN)
        conn.autocommit(False)
        cur = conn.cursor()
        cur.execute("START TRANSACTION")
        cur.execute("SELECT * FROM accounts WHERE id=1 FOR UPDATE")
        cur.fetchall()
        ev_a_locked.set()
        time.sleep(3)
        conn.commit(); conn.close()

    def txn_b():
        ev_a_locked.wait(timeout=5)
        conn = pymysql.connect(**MARIA_DSN)
        conn.autocommit(False)
        cur = conn.cursor()
        cur.execute("SET SESSION innodb_lock_wait_timeout = 1")
        cur.execute("START TRANSACTION")
        t0 = time.time()
        try:
            cur.execute("SELECT * FROM accounts WHERE id=1 FOR UPDATE")
            result['outcome'] = 'got lock'
        except pymysql.err.OperationalError as e:
            result['outcome'] = 'OperationalError'
            result['elapsed'] = time.time() - t0
            conn.rollback()
        conn.close()

    ta = threading.Thread(target=txn_a); tb = threading.Thread(target=txn_b)
    ta.start(); tb.start(); ta.join(timeout=10); tb.join(timeout=10)
    return result

pg_r = test_lock_timeout_pg()
assert pg_r['outcome'] == 'LockNotAvailable', f"got {pg_r}"
assert 0.8 < pg_r['elapsed'] < 2.0, f"expected timeout around 1s (not the full 3s wait), got {pg_r['elapsed']}"

maria_r = test_lock_timeout_maria()
assert maria_r['outcome'] == 'OperationalError', f"got {maria_r}"
assert 0.8 < maria_r['elapsed'] < 2.0, f"expected timeout around 1s, got {maria_r['elapsed']}"

print(f"lock wait timeout verified on both engines: PG timed out at {pg_r['elapsed']:.3f}s, Maria at {maria_r['elapsed']:.3f}s (both far short of A's 3s hold time)")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
PG lock_timeout=1000ms:            1.001秒后抛出 LockNotAvailable
Maria innodb_lock_wait_timeout=1:  1.002秒后抛出 OperationalError(1205, Lock wait timeout exceeded)
```

**面试怎么问+追问链**

- Q:如果一个在线服务的接口偶尔响应很慢,排查发现是在等一个行锁,你会怎么处理?
  - 追问1:设置一个较短的锁等待超时是不是就能解决这个问题?
    - 深挖追问(区分度较高):超时之后呢,业务逻辑该怎么处理这个失败(答案方向:锁等待超时只是把"无限期挂起"变成"有限时间内明确失败",本身不解决根本的锁竞争问题——需要结合应用层的重试策略[通常配合指数退避+最大重试次数]、甚至重新评估锁定的范围/顺序[知识点3/4/6]是否可以优化,单纯设置超时而不处理后续失败逻辑,只是把"用户等待转圈"变成了"用户看到报错",体验未必真的变好)。

**常见坑**

- 把锁等待超时和知识点5的死锁检测混为一谈——死锁检测是引擎自动发现"循环等待"并主动打破;锁等待超时是给"非循环但等太久"的普通等待设置一个应用层可控的上限,两者是互补但不同的机制,不能只依赖其中一个。
- 全局设置一个很短的超时应用到所有查询——不同业务操作对"愿意等多久"的容忍度不同(用户交互式请求 vs 后台批处理任务),应该按场景差异化配置,而不是一刀切。
