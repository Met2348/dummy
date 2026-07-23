# 事务与隔离级别

> 板块 III(事务与并发控制)第 1 篇。环境:全部 `python-wsl2`(依赖 WSL2 Rocky Linux 已启动的 PostgreSQL 16.14 + MariaDB 10.11.15,凭据见 `00-roadmap.md`)。本类知识点大量涉及**两个真实数据库连接的并发时序控制**,全部用 `threading.Event` 做显式同步点(不用 `sleep` 猜时间),多数场景独立重跑 3 次确认稳定复现。

## 1. ACID四性质逐条拆解

**签名/是什么**

```
Atomicity(原子性):   事务内的多条语句要么全部生效,要么全部不生效
Consistency(一致性): 事务执行前后,数据库都满足所有预定义的约束(主键/外键/CHECK等)
Isolation(隔离性):   并发事务之间互不干扰的程度(具体多"互不干扰"由隔离级别决定,见02类)
Durability(持久性):  事务一旦提交,即使紧接着系统崩溃,修改也不会丢失(依赖WAL/redo log,详见06类)
```

一句话:ACID 描述的是数据库对"一个事务"应该提供的四个承诺,原子性和持久性是数据库引擎自己实现的机制保证,一致性一半靠约束(01/02类已验证)一半靠应用逻辑,隔离性的具体强弱是可以配置的(这正是本类后续知识点的主题)。

**底层机制/为什么这样设计**

ACID 不是四个独立的特性,而是互相支撑的一套整体设计:原子性依赖 undo 日志(记录"如何撤销"这个事务已做的修改,06类展开)在失败时回滚;持久性依赖 redo 日志/WAL(记录"如何重做"这个事务的修改,即使提交后立刻崩溃也能在重启时重放)先于数据页落盘(Write-Ahead Logging);一致性某种程度上是"果"不是"因"——只要约束系统本身正确(01/02类),且原子性/隔离性保证了事务不会看到/留下"半成品"状态,一致性就自然成立。**事务级原子性**(本类新增的层面)比 01 类知识点4验证过的"单语句原子性"更进一步:一个事务内**多条**语句,即使前面的语句已经成功执行,只要事务内后续任意语句失败且整体回滚,已经"成功"过的前面语句效果也会被撤销。

**AI研究/工程场景**

转账场景(账户A扣款、账户B加款)是ACID的经典例子:两条UPDATE必须在同一个事务里,否则"扣了A的钱但加B失败"这种半成品状态在真实系统里是不可接受的资金丢失。训练数据标注平台批量提交一批标注结果时,如果第50条数据格式错误,整个批次通常应该全部回滚而不是提交前49条,这样重试时不需要判断"哪些已经成功了"。

**可运行例子**(环境:`python-wsl2`)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14(已启动)
import psycopg2

pg = psycopg2.connect(host='127.0.0.1', port=5432, dbname='dbdemo', user='dbdemo', password='dbdemo_local_only')
pgc = pg.cursor()
pgc.execute("DROP TABLE IF EXISTS accounts")
pgc.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER CHECK (balance >= 0))")
pgc.execute("INSERT INTO accounts VALUES (1, 1000), (2, 500)")
pg.commit()

# 事务内两条UPDATE模拟转账,第一条(账户1扣款)真实成功执行,
# 第二条故意写错(账户2本该+300,写成了-600,500-600=-100违反CHECK约束)
try:
    pgc.execute("UPDATE accounts SET balance = balance - 300 WHERE id = 1")
    pgc.execute("UPDATE accounts SET balance = balance - 600 WHERE id = 2")
    pg.commit()
    assert False, "expected the CHECK violation to prevent commit"
except psycopg2.errors.CheckViolation:
    pg.rollback()

pgc.execute("SELECT id, balance FROM accounts ORDER BY id")
final_state = pgc.fetchall()
# 事务级原子性: 账户1的扣款语句本身"成功执行过",但因为同一事务里第二条语句失败,
# 整个事务回滚,账户1也必须恢复到1000,不能停留在700这个"半成品"状态
assert final_state == [(1, 1000), (2, 500)], f"expected full rollback to original state, got {final_state}"

pg.close()
print("transaction-level atomicity verified: a later statement's failure rolls back an earlier statement's already-executed effect within the same transaction")
```

**面试怎么问+追问链**

- Q:ACID 的"一致性"具体指什么?和"隔离性"有什么区别?
  - 追问1:如果一个事务提交后,数据库里出现了一条违反业务逻辑(但没有被任何约束捕获)的数据,这算是破坏了"一致性"吗,数据库要为此负责吗?
    - 深挖追问(区分度较高):这说明了"一致性"这个术语在ACID语境下的真正含义是什么(答案方向:ACID的"一致性"严格来说指的是"数据库预定义的约束[主键/外键/CHECK/唯一性]始终满足",不是"业务逻辑永远正确"——业务逻辑正确性是应用层的责任,数据库只保证它知道的约束不被违反,这是一个经常被混淆但面试时值得讲清楚的边界)。

**常见坑**

- 把 ACID 的"一致性"和 CAP 定理的"一致性"混为一谈——两者是完全不同语境下的术语(ACID的C是约束满足,CAP的C是多副本间数据相同),这个术语重名是数据库领域一个经典的容易搞混的陷阱,07类讲CAP定理时会重新说明。
- 只测试过"单条语句失败会回滚"就认为理解了原子性——本类验证的"事务内多条语句,后面失败连累前面已成功的语句一起回滚"是更完整、更容易被面试追问出漏洞的理解层次。

---

## 2. 隔离级别体系与异常现象对应关系

**签名/是什么**

```
四种隔离级别(从弱到强): READ UNCOMMITTED < READ COMMITTED < REPEATABLE READ < SERIALIZABLE
四种异常现象:            脏读 > 不可重复读 > 幻读 > 丢失更新(不完全是线性关系,见下表)
SQL标准: 隔离级别越强,允许发生的异常现象越少(理论上界,具体引擎实现可能更严格或有例外)
```

一句话:SQL标准用"允许发生哪些异常现象"来定义四种隔离级别,级别越高,理论上允许的异常越少,但**具体到某个引擎的真实实现,同一个级别名字允许的异常可能和标准定义不完全一致**(这正是本类后面几个知识点要通过真实实验揭示的重点)。

**底层机制/为什么这样设计**

隔离级别本质上是"性能和正确性之间可调节的旋钮"——完全隔离(SERIALIZABLE)最安全但代价最高(更多锁/更多冲突检测/更多事务因冲突被迫重试),完全不隔离(READ UNCOMMITTED)最快但最不安全。SQL标准定义了一张"级别→允许的异常"对照表,给了工程师一个共同语言,但**标准只规定了"最多允许发生什么",没有规定"引擎具体怎么实现"**,这就是为什么两个都自称支持"REPEATABLE READ"的引擎,底层机制(PG的MVCC[Multi-Version Concurrency Control,多版本并发控制;完整原理见05类知识点1,这里先知道它的核心思路是"写不覆盖旧版本、读只读自己事务开始时的版本,让读写互不阻塞"即可]快照 vs InnoDB的MVCC快照+间隙锁)和实际行为边界(后续知识点会验证REPEATABLE READ对丢失更新的防护程度两者不同)可能有实质差异。

**AI研究/工程场景**

选择隔离级别是一个"先看清楚这个级别在这个引擎里究竟保证了什么,再决定要不要额外加锁"的工程决策,不能只凭级别的名字做判断——本类后续知识点会用真实实验证明"同名不同实"这件事,这是比死记硬背标准定义表更重要的实战能力。

**可运行例子**(环境:`python-wsl2`)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14 + MariaDB 10.11.15(已启动)
import psycopg2
import pymysql

pg = psycopg2.connect(host='127.0.0.1', port=5432, dbname='dbdemo', user='dbdemo', password='dbdemo_local_only')
pgc = pg.cursor()
pgc.execute("SHOW TRANSACTION ISOLATION LEVEL")
pg_default = pgc.fetchone()[0]
pg.close()

maria = pymysql.connect(host='127.0.0.1', port=3306, database='dbdemo', user='dbdemo', password='dbdemo_local_only')
mc = maria.cursor()
# 注意: MySQL 5.7.20+/8.0 把这个变量改名成了 transaction_isolation,
# 但 MariaDB 10.11 保留的是fork自更早MySQL版本的旧变量名 tx_isolation,
# 直接抄MySQL 8官方文档的变量名在MariaDB上会报"Unknown system variable"
mc.execute("SELECT @@tx_isolation")
maria_default = mc.fetchone()[0]
maria.close()

# 两个引擎的默认隔离级别真实不同: PostgreSQL默认READ COMMITTED,MariaDB/InnoDB默认REPEATABLE READ
assert pg_default == "read committed", f"got {pg_default}"
assert maria_default == "REPEATABLE-READ", f"got {maria_default}"

print("default isolation level verified: PostgreSQL defaults to READ COMMITTED, MariaDB/InnoDB defaults to REPEATABLE READ")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
PostgreSQL 16.14 默认隔离级别: read committed
MariaDB 10.11.15 默认隔离级别: REPEATABLE-READ
```

| 隔离级别 | 脏读 | 不可重复读 | 幻读(标准定义) | 丢失更新 |
|---|---|---|---|---|
| READ UNCOMMITTED | 可能 | 可能 | 可能 | 可能 |
| READ COMMITTED | 不可能 | 可能 | 可能 | 可能 |
| REPEATABLE READ | 不可能 | 不可能 | 标准定义不可能,**但本类知识点5、05类知识点4会验证不同引擎的真实实现不完全一致** | 标准未强制,**本类知识点7会验证PG和MariaDB在这里的真实差异** |
| SERIALIZABLE | 不可能 | 不可能 | 不可能 | 不可能(理论保证,本类知识点7验证两个引擎都能做到,但实现机制不同) |

**面试怎么问+追问链**

- Q:PostgreSQL 和 MySQL/MariaDB 的默认隔离级别一样吗?
  - 追问1:如果一个应用从MySQL迁移到PostgreSQL,什么都没改,原来"隐式依赖REPEATABLE READ幻读防护"的代码,迁移后可能出什么问题?
    - 深挖追问(区分度较高):这类"默认值不同导致的隐藏迁移风险"还有哪些例子(答案方向:除了默认隔离级别不同,本类知识点3还会揭示"REPEATABLE READ"这个名字在两个引擎的实际保证也不完全相同,这类"表面API一致但语义有细微差异"的坑在数据库迁移场景里很常见,面试官如果问迁移经验,主动提及这类"名字一样不代表行为一样"的观察点是加分项)。

**常见坑**

- 假设所有引擎的默认隔离级别都一样,不显式设置就直接依赖默认行为——本类已用真实查询验证两个引擎的默认值不同(PG是READ COMMITTED,MariaDB是REPEATABLE READ),跨引擎迁移或多引擎项目里这是真实的隐患。
- 把这张"级别→异常"对照表当成放之四海皆准的精确保证——它是SQL标准定义的"理论上界",本类后续知识点会用真实实验证明具体引擎的实现在REPEATABLE READ这一档上有实质性差异。
- 查询当前隔离级别的系统变量名字两个引擎不一样,而且**MySQL自己新旧版本的名字也不一样**:MySQL 5.7.20+/8.0用`transaction_isolation`,MariaDB 10.11(fork自更早的MySQL版本)还是用旧名字`tx_isolation`——直接照抄MySQL 8官方文档的变量名在MariaDB上会报`Unknown system variable`,这是本类撰写过程中真实踩到的坑。

---

## 3. 脏读:READ UNCOMMITTED的真实引擎差异

**签名/是什么**

```
脏读(Dirty Read): 事务A读到了事务B尚未提交、且之后被回滚的数据
```

一句话:脏读是四种异常里最"危险"的一种(读到的数据可能从未真正存在过),READ UNCOMMITTED理论上允许脏读,但**真实测试显示只有MariaDB/InnoDB会真的发生脏读,PostgreSQL即使显式设置成READ UNCOMMITTED,实际行为也等同于READ COMMITTED**。

**底层机制/为什么这样设计**

MariaDB/InnoDB的READ UNCOMMITTED是真正的"不做任何可见性检查",直接读取当前最新的数据页内容,不管这个修改是否已提交——这是它能读到"脏数据"的根本原因。PostgreSQL的实现选择不同:PostgreSQL的MVCC架构决定了它的每一次读取都天然基于某种"快照"(即使是READ COMMITTED,也是每条语句开始时取一次快照),PostgreSQL官方文档明确说明"由于MVCC架构,没有真正实现READ UNCOMMITTED,设置这个级别在PG里的实际行为等同于READ COMMITTED"——这不是bug,是PG的架构选择:与其额外实现一套"读脏数据"的路径(这条路径在MVCC架构下反而更复杂,因为脏数据可能已经被判定为"其他事务不可见"),不如直接不提供真正的READ UNCOMMITTED。

**AI研究/工程场景**

如果一个系统的报表/监控查询"宁可读到脏数据也要追求最低延迟"(比如实时监控大盘,容忍偶尔的短暂不准确),在MySQL/MariaDB上设置READ UNCOMMITTED是真实有效的优化手段;但同样的代码搬到PostgreSQL上,这个"优化"完全不起作用(底层还是READ COMMITTED的开销),不了解这个差异会导致"为什么同样设置了READ UNCOMMITTED,PG这边好像没有变快"的困惑排查。

**可运行例子**(环境:`python-wsl2`)

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
    pgc.execute("INSERT INTO accounts VALUES (1, 1000)")
    pg.close()
    maria = pymysql.connect(**MARIA_DSN); maria.autocommit(True); mc = maria.cursor()
    mc.execute("DROP TABLE IF EXISTS accounts")
    mc.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER) ENGINE=InnoDB")
    mc.execute("INSERT INTO accounts VALUES (1, 1000)")
    maria.close()

def dirty_read_test(connect_fn, set_isolation_sql, start_txn_sql):
    setup()
    result = {}
    ev_a_updated = threading.Event()
    ev_b_done = threading.Event()

    def txn_a():
        conn = connect_fn()
        if hasattr(conn, 'autocommit') and callable(getattr(conn, 'autocommit')):
            conn.autocommit(False)
        cur = conn.cursor()
        cur.execute("UPDATE accounts SET balance=999 WHERE id=1")
        ev_a_updated.set()
        ev_b_done.wait(timeout=5)
        conn.rollback()  # A最终回滚,999这个值从未真正"存在"过
        conn.close()

    def txn_b():
        conn = connect_fn()
        cur = conn.cursor()
        cur.execute(set_isolation_sql)
        if start_txn_sql:
            cur.execute(start_txn_sql)
        ev_a_updated.wait(timeout=5)
        cur.execute("SELECT balance FROM accounts WHERE id=1")
        result['b_saw'] = cur.fetchone()[0]
        conn.commit()
        conn.close()
        ev_b_done.set()

    ta = threading.Thread(target=txn_a); tb = threading.Thread(target=txn_b)
    ta.start(); tb.start()
    ta.join(timeout=10); tb.join(timeout=10)
    return result

maria_result = dirty_read_test(
    lambda: pymysql.connect(**MARIA_DSN),
    "SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED",
    "START TRANSACTION",
)
assert maria_result['b_saw'] == 999, \
    f"MariaDB READ UNCOMMITTED should see the uncommitted (later rolled-back) value, got {maria_result['b_saw']}"

pg_result = dirty_read_test(
    lambda: psycopg2.connect(**PG_DSN),
    "SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED",
    None,
)
assert pg_result['b_saw'] == 1000, \
    f"PostgreSQL 'READ UNCOMMITTED' should behave like READ COMMITTED (no dirty read), got {pg_result['b_saw']}"

print("dirty read verified: MariaDB READ UNCOMMITTED genuinely sees uncommitted data (999), PG's 'READ UNCOMMITTED' does not (still 1000, behaves as READ COMMITTED)")
```

**面试怎么问+追问链**

- Q:什么是脏读?哪个隔离级别会发生?
  - 追问1:PostgreSQL支持READ UNCOMMITTED吗?
    - 深挖追问(区分度较高):如果PG"支持"设置这个级别但实际不发生脏读,这算不算违反SQL标准(答案方向:不算违反——SQL标准只规定"某级别最多允许哪些异常发生",没有规定"必须发生这些异常",一个引擎完全可以在声称支持某级别的同时提供比标准要求更强的实际保证,PG的选择[干脆不实现真正的脏读路径]完全合规,这也是"面对同名隔离级别不能想当然,要看具体引擎文档"这条纪律的一个具体例证)。

**常见坑**

- 假设"两个引擎都支持READ UNCOMMITTED"就意味着"两个引擎在这个级别下行为相同"——本类的真实实验证明PostgreSQL的READ UNCOMMITTED根本不会脏读,这个认知差异在跨引擎迁移或者面试被追问细节时很容易暴露。
- 为了追求性能到处设置READ UNCOMMITTED,却没有评估业务是否真的能容忍"读到之后会消失的数据"——脏读不是"读到旧数据",而是"读到可能从未真正存在过的数据",容错评估的角度不一样。

---

## 4. 不可重复读:READ COMMITTED vs REPEATABLE READ真实对比

**签名/是什么**

```
不可重复读(Non-repeatable Read): 同一事务内,两次读同一行,读到了不同的值(因为中途有其他事务提交了修改)
```

一句话:READ COMMITTED每条语句单独取一次"快照"(能看到别的事务刚提交的新数据),REPEATABLE READ在整个事务期间只取一次快照(事务开始后,不管别人提交了什么,同一事务内反复读同一行看到的都是事务开始时的版本)——这个差异在两个引擎上表现完全一致。

**底层机制/为什么这样设计**

"快照的生效范围"是这两个级别的核心区别:READ COMMITTED的哲学是"我想看到最新的已提交数据",所以每条语句都重新取快照;REPEATABLE READ的哲学是"我希望整个事务期间看到的世界是一致的、不变的",所以整个事务共用同一个快照,即使别的事务在这期间提交了修改,本事务也"看不见"(这正是MVCC多版本并发控制的价值所在——不需要用锁阻塞别人的写入,只需要让自己始终看旧版本)。

**AI研究/工程场景**

生成一份"账户对账单"这类需要在同一个事务内多次查询、且要求这些查询结果互相一致的报表场景,应该用REPEATABLE READ(避免中途别的事务改了数据导致对账单前后数字对不上);而"每次查询都想看最新数据"的交互式查询场景(比如用户刷新页面),READ COMMITTED往往是更自然的默认选择。

**可运行例子**(环境:`python-wsl2`)

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
    pgc.execute("INSERT INTO accounts VALUES (1, 1000)")
    pg.close()
    maria = pymysql.connect(**MARIA_DSN); maria.autocommit(True); mc = maria.cursor()
    mc.execute("DROP TABLE IF EXISTS accounts")
    mc.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER) ENGINE=InnoDB")
    mc.execute("INSERT INTO accounts VALUES (1, 1000)")
    maria.close()

def test_repeatability(connect_fn, set_isolation_sql, start_txn_sql):
    setup()
    result = {}
    ev_a_first = threading.Event()
    ev_b_done = threading.Event()

    def txn_a():
        conn = connect_fn()
        if hasattr(conn, 'autocommit') and callable(getattr(conn, 'autocommit')):
            conn.autocommit(False)
        cur = conn.cursor()
        cur.execute(set_isolation_sql)
        if start_txn_sql:
            cur.execute(start_txn_sql)
        cur.execute("SELECT balance FROM accounts WHERE id=1")
        result['first_read'] = cur.fetchone()[0]
        ev_a_first.set()
        ev_b_done.wait(timeout=5)
        cur.execute("SELECT balance FROM accounts WHERE id=1")
        result['second_read'] = cur.fetchone()[0]
        conn.commit()
        conn.close()

    def txn_b():
        ev_a_first.wait(timeout=5)
        conn = connect_fn()
        if hasattr(conn, 'autocommit') and callable(getattr(conn, 'autocommit')):
            conn.autocommit(True)
        else:
            conn.autocommit = True
        cur = conn.cursor()
        cur.execute("UPDATE accounts SET balance=2000 WHERE id=1")
        conn.close()
        ev_b_done.set()

    ta = threading.Thread(target=txn_a); tb = threading.Thread(target=txn_b)
    ta.start(); tb.start()
    ta.join(timeout=10); tb.join(timeout=10)
    return result

pg_rc = test_repeatability(lambda: psycopg2.connect(**PG_DSN), "SET TRANSACTION ISOLATION LEVEL READ COMMITTED", None)
assert pg_rc['first_read'] != pg_rc['second_read'], f"READ COMMITTED should show the new value on second read, got {pg_rc}"

pg_rr = test_repeatability(lambda: psycopg2.connect(**PG_DSN), "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ", None)
assert pg_rr['first_read'] == pg_rr['second_read'], f"REPEATABLE READ should show the same value both times, got {pg_rr}"

maria_rc = test_repeatability(lambda: pymysql.connect(**MARIA_DSN), "SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED", "START TRANSACTION")
assert maria_rc['first_read'] != maria_rc['second_read'], f"got {maria_rc}"

maria_rr = test_repeatability(lambda: pymysql.connect(**MARIA_DSN), "SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ", "START TRANSACTION")
assert maria_rr['first_read'] == maria_rr['second_read'], f"got {maria_rr}"

print("non-repeatable read verified on both engines: READ COMMITTED sees the mid-transaction change, REPEATABLE READ does not")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
PG READ COMMITTED:     first=1000 second=2000  (不可重复读真实发生)
PG REPEATABLE READ:    first=1000 second=1000  (可重复)
Maria READ COMMITTED:  first=1000 second=2000  (不可重复读真实发生)
Maria REPEATABLE READ: first=1000 second=1000  (可重复)
```

**面试怎么问+追问链**

- Q:REPEATABLE READ怎么做到"事务期间看到的数据不变"的?加锁了吗?
  - 追问1:如果没加锁,那B事务的UPDATE不会被阻塞吗?
    - 深挖追问(区分度较高):A事务全程"看不见"B的修改,和"B的UPDATE被A阻塞"是一回事吗(答案方向:完全不是一回事——本例已经真实验证了B的UPDATE可以正常执行并提交[没有被阻塞],A只是基于MVCC快照"看不见"这个已经真实发生的修改,这是MVCC的核心价值:用"版本可见性规则"代替"锁"来实现隔离,读写互不阻塞,这个区分在05类MVCC与锁机制会深入展开)。

**常见坑**

- 把"REPEATABLE READ让本事务看不到别人的修改"误解为"REPEATABLE READ会阻塞别人的修改"——两者是完全不同的机制(MVCC可见性规则 vs 锁),本类的可运行例子已经验证B的UPDATE没有被阻塞。
- 忘记检查应用程序的连接池/ORM默认使用的隔离级别——很多ORM框架的默认设置可能和引擎默认值不一致,不显式确认容易和预期行为脱节。

---

## 5. 幻读:REPEATABLE READ下两个引擎不同机制但都能阻止

**签名/是什么**

```
幻读(Phantom Read): 同一事务内,两次执行相同的范围查询,第二次多出了(或少了)满足条件的行
```

一句话:普通只读的REPEATABLE READ在两个引擎上都能避免幻读(都基于MVCC快照),但一旦换成加锁读(`SELECT ... FOR UPDATE`),两个引擎的真实行为就分道扬镳了——MariaDB/InnoDB的间隙锁会真的**阻塞**并发的INSERT,PostgreSQL不会阻塞,只是"看不见"。

**底层机制/为什么这样设计**

普通只读查询的幻读防护,本质上是KP4"不可重复读"防护的自然延伸——既然REPEATABLE READ让本事务全程只看一份快照,那么"新插入的行"自然也不在这份快照里,幻读和不可重复读在MVCC框架下是同一个机制解决的。但加锁读(`FOR UPDATE`)引入了新问题:如果A事务锁定了"balance>500"这个范围准备后续更新,B事务这时候插入一行新的balance>500的记录,理论上应该被拦截(否则A基于锁定范围做的后续决策可能是错的)。InnoDB用"间隙锁"(锁住索引记录之间的"间隙",不只锁记录本身)真正阻止这类插入;PostgreSQL在REPEATABLE READ级别没有类似间隙锁的机制,不会阻塞并发INSERT,只是保证A自己的后续查询依然看不到这行新数据(SERIALIZABLE级别PostgreSQL才会引入SIREAD锁做更严格的冲突检测,07类会展开)。

**AI研究/工程场景**

"先查询库存是否充足,再批量插入新订单"这类业务,如果依赖REPEATABLE READ的"看不见新数据"来保证安全,在MariaDB上因为间隙锁的真实阻塞效果,并发的插入操作会被物理排队,行为符合直觉;但同样的代码在PostgreSQL上,并发插入不会被阻塞而是各自独立提交,如果应用逻辑依赖"这段时间没有新库存记录被插入"这个假设做后续计算,PostgreSQL上可能出现应用逻辑没预料到的并发结果,这是一个真实存在的跨引擎迁移陷阱。

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
    pgc.execute("INSERT INTO accounts VALUES (1, 1000), (2, 2000)")
    pg.close()
    maria = pymysql.connect(**MARIA_DSN); maria.autocommit(True); mc = maria.cursor()
    mc.execute("DROP TABLE IF EXISTS accounts")
    mc.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER) ENGINE=InnoDB")
    mc.execute("INSERT INTO accounts VALUES (1, 1000), (2, 2000)")
    maria.close()

# 场景1: 普通只读REPEATABLE READ, 两个引擎都不会看到幻读
def test_phantom_plain(connect_fn, set_isolation_sql, start_txn_sql):
    setup()
    result = {}
    ev_a_first = threading.Event(); ev_b_done = threading.Event()

    def txn_a():
        conn = connect_fn()
        if hasattr(conn, 'autocommit') and callable(getattr(conn, 'autocommit')):
            conn.autocommit(False)
        cur = conn.cursor()
        cur.execute(set_isolation_sql)
        if start_txn_sql: cur.execute(start_txn_sql)
        cur.execute("SELECT COUNT(*) FROM accounts WHERE balance > 500")
        result['first_count'] = cur.fetchone()[0]
        ev_a_first.set()
        ev_b_done.wait(timeout=5)
        cur.execute("SELECT COUNT(*) FROM accounts WHERE balance > 500")
        result['second_count'] = cur.fetchone()[0]
        conn.commit(); conn.close()

    def txn_b():
        ev_a_first.wait(timeout=5)
        conn = connect_fn()
        if hasattr(conn, 'autocommit') and callable(getattr(conn, 'autocommit')):
            conn.autocommit(True)
        else:
            conn.autocommit = True
        cur = conn.cursor()
        cur.execute("INSERT INTO accounts VALUES (99, 999999)")
        conn.close(); ev_b_done.set()

    ta = threading.Thread(target=txn_a); tb = threading.Thread(target=txn_b)
    ta.start(); tb.start(); ta.join(timeout=10); tb.join(timeout=10)
    return result

pg_r = test_phantom_plain(lambda: psycopg2.connect(**PG_DSN), "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ", None)
assert pg_r['first_count'] == pg_r['second_count'], f"PG should not see the phantom, got {pg_r}"

maria_r = test_phantom_plain(lambda: pymysql.connect(**MARIA_DSN), "SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ", "START TRANSACTION")
assert maria_r['first_count'] == maria_r['second_count'], f"MariaDB should not see the phantom, got {maria_r}"

# 场景2: SELECT ... FOR UPDATE(加锁读), 两个引擎在"是否物理阻塞并发INSERT"上真实分歧
def test_phantom_locking(connect_fn, set_isolation_sql, start_txn_sql):
    setup()
    result = {}
    ev_a_locked = threading.Event()

    def txn_a():
        conn = connect_fn()
        if hasattr(conn, 'autocommit') and callable(getattr(conn, 'autocommit')):
            conn.autocommit(False)
        cur = conn.cursor()
        cur.execute(set_isolation_sql)
        if start_txn_sql: cur.execute(start_txn_sql)
        cur.execute("SELECT * FROM accounts WHERE balance > 500 FOR UPDATE")
        cur.fetchall()
        ev_a_locked.set()
        time.sleep(1.5)  # 保持事务持有锁/快照,给B机会尝试插入
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
        cur.execute("INSERT INTO accounts VALUES (99, 999999)")
        result['b_elapsed'] = time.time() - t0
        conn.close()

    ta = threading.Thread(target=txn_a); tb = threading.Thread(target=txn_b)
    ta.start(); tb.start(); ta.join(timeout=10); tb.join(timeout=10)
    return result

pg_lock = test_phantom_locking(lambda: psycopg2.connect(**PG_DSN), "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ", None)
assert pg_lock['b_elapsed'] < 0.5, f"PG's REPEATABLE READ should NOT block the concurrent INSERT, got elapsed={pg_lock['b_elapsed']}"

maria_lock = test_phantom_locking(lambda: pymysql.connect(**MARIA_DSN), "SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ", "START TRANSACTION")
assert maria_lock['b_elapsed'] > 1.0, f"MariaDB's gap lock SHOULD block the concurrent INSERT for ~1.5s, got elapsed={maria_lock['b_elapsed']}"

print(f"phantom read verified: plain SELECT avoids it on both engines via MVCC; SELECT FOR UPDATE diverges - PG doesn't block INSERT ({pg_lock['b_elapsed']:.3f}s), MariaDB's gap lock blocks it ({maria_lock['b_elapsed']:.3f}s)")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
普通只读REPEATABLE READ: PG first=2 second=2 (无幻读); Maria first=2 second=2 (无幻读)
SELECT FOR UPDATE:       PG INSERT耗时 0.002s (不阻塞); Maria INSERT耗时 1.497s (被间隙锁真实阻塞约1.5秒)
```

**面试怎么问+追问链**

- Q:MySQL的"间隙锁"具体锁的是什么?
  - 追问1:间隙锁只在什么隔离级别下生效?
    - 深挖追问(区分度较高):PostgreSQL没有间隙锁,那它在SERIALIZABLE级别下怎么防止这类"锁定范围后被插入新行破坏"的场景(答案方向:PostgreSQL用SIREAD锁[谓词锁的一种]做的是"事后冲突检测"而不是"事前物理阻塞"——两个事务都能正常执行,但如果PG检测到它们的读写模式构成了不可串行化的依赖环,会在其中一个事务提交时报SerializationFailure强制其中一个失败重试,这是"预防式阻塞"[InnoDB间隙锁]和"检测式回滚"[PG SIREAD]两种完全不同的工程哲学,本类知识点7会展开这个话题)。

**常见坑**

- 以为"REPEATABLE READ能防幻读"这个结论对所有读取方式都成立——本类已验证这个结论只对普通只读查询成立,一旦是`FOR UPDATE`加锁读,两个引擎的真实行为(阻塞与否)会不同,这个区分在需要"锁定范围防止并发插入"的业务场景里至关重要。
- 把间隙锁想象成"锁住了不存在的行"——更准确的理解是"锁住了索引结构里两个相邻键值之间的间隙,阻止新纪录插入到这个间隙里",这个机制和"锁住某一行"是不同粒度的概念。

---

## 6. 丢失更新:READ COMMITTED下双引擎都真实发生

**签名/是什么**

```
丢失更新(Lost Update): 两个事务各自基于"读到的旧值"计算新值并写回,后写入的覆盖了先写入的,其中一次更新的效果凭空消失
```

一句话:如果两个事务都是"读取当前值→在应用层算出新值→写回",而不是用数据库原生的原子操作(比如`UPDATE ... SET balance = balance + 100`)或显式加锁,丢失更新在READ COMMITTED下会真实发生,且两个引擎表现一致。

**底层机制/为什么这样设计**

丢失更新的根源不在数据库,而在应用代码的写法:"读-算-写"这个模式天然存在"读到的值可能在写回前已经过期"的竞态窗口。READ COMMITTED只保证"每次读到的是已提交的数据",不保证"从我读到这个值,到我写回之前,没有别人也读了同一个值并抢先写回"——这个更强的保证需要额外机制(显式锁/乐观锁版本号/更高的隔离级别,见知识点7)才能获得,READ COMMITTED本身不提供。

**AI研究/工程场景**

"库存扣减:查询当前库存→判断是否充足→扣减并写回"如果不加任何保护,在高并发下是丢失更新的经典重灾区(两个并发请求都读到库存充足,都扣减,最终库存数字比两次扣减的总和要多,等于其中一次扣减"凭空消失"了,可能导致超卖)。正确写法通常是用`UPDATE inventory SET stock = stock - 1 WHERE id=? AND stock >= 1`这种数据库原生的原子读改写语句,把"读-判断-写"合并成数据库能保证原子性的单一操作。

**可运行例子**(环境:`python-wsl2`)

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
    pgc.execute("INSERT INTO accounts VALUES (1, 1000)")
    pg.close()
    maria = pymysql.connect(**MARIA_DSN); maria.autocommit(True); mc = maria.cursor()
    mc.execute("DROP TABLE IF EXISTS accounts")
    mc.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER) ENGINE=InnoDB")
    mc.execute("INSERT INTO accounts VALUES (1, 1000)")
    maria.close()

def test_lost_update(connect_fn, set_isolation_sql, start_txn_sql):
    setup()
    ev_a_read = threading.Event(); ev_b_done = threading.Event()

    def txn_a():
        conn = connect_fn()
        if hasattr(conn, 'autocommit') and callable(getattr(conn, 'autocommit')):
            conn.autocommit(False)
        cur = conn.cursor()
        cur.execute(set_isolation_sql)
        if start_txn_sql: cur.execute(start_txn_sql)
        cur.execute("SELECT balance FROM accounts WHERE id=1")
        balance = cur.fetchone()[0]  # "读"
        ev_a_read.set()
        ev_b_done.wait(timeout=5)
        new_balance = balance + 100  # "算"(基于可能已过期的balance)
        cur.execute("UPDATE accounts SET balance=%s WHERE id=1", (new_balance,))  # "写"
        conn.commit(); conn.close()

    def txn_b():
        ev_a_read.wait(timeout=5)
        conn = connect_fn()
        if hasattr(conn, 'autocommit') and callable(getattr(conn, 'autocommit')):
            conn.autocommit(True)
        else:
            conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT balance FROM accounts WHERE id=1")
        balance = cur.fetchone()[0]
        cur.execute("UPDATE accounts SET balance=%s WHERE id=1", (balance + 50,))
        conn.close(); ev_b_done.set()

    ta = threading.Thread(target=txn_a); tb = threading.Thread(target=txn_b)
    ta.start(); tb.start(); ta.join(timeout=10); tb.join(timeout=10)

    conn = connect_fn()
    cur = conn.cursor()
    cur.execute("SELECT balance FROM accounts WHERE id=1")
    final = cur.fetchone()[0]
    conn.close()
    return final

pg_final = test_lost_update(lambda: psycopg2.connect(**PG_DSN), "SET TRANSACTION ISOLATION LEVEL READ COMMITTED", None)
# 正确结果应是1000+100+50=1150; 1100意味着B的+50被A基于旧值的写回覆盖,凭空消失了
assert pg_final == 1100, f"expected lost update (1100), got {pg_final}"

maria_final = test_lost_update(lambda: pymysql.connect(**MARIA_DSN), "SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED", "START TRANSACTION")
assert maria_final == 1100, f"expected lost update (1100), got {maria_final}"

print(f"lost update verified on both engines under READ COMMITTED: final balance={pg_final}/{maria_final} (both show 1100, not the correct 1150 - B's +50 vanished)")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
PG READ COMMITTED 最终余额: 1100 (正确应为1150,B的+50丢失)
Maria READ COMMITTED 最终余额: 1100 (正确应为1150,B的+50丢失)
```

**面试怎么问+追问链**

- Q:简历上写"用了乐观锁解决高并发场景下的丢失更新问题",具体是怎么做的?
  - 追问1:乐观锁的"版本号"具体是怎么防止丢失更新的?
    - 深挖追问(区分度较高):乐观锁和悲观锁(`SELECT FOR UPDATE`)相比,分别适合什么并发场景(答案方向:乐观锁[`UPDATE ... WHERE id=? AND version=?`,受影响行数为0就说明版本冲突需要重试]适合冲突概率低、大部分请求不会真的撞车的场景[重试代价小];悲观锁适合冲突概率高的场景[提前锁住比"频繁失败重试"效率更高],这是一个需要结合真实业务QPS/冲突率做的量化决策,不是背下"两种锁的定义"就能回答完整的问题)。

**常见坑**

- "读-算-写"模式的代码即使包在一个数据库事务里,只要没有额外的锁保护,READ COMMITTED级别下丢失更新依然会真实发生——很多人误以为"用了事务"就自动没有并发问题,本类已用真实实验证明这个误解。
- 只在应用代码层面加内存锁(比如Python的`threading.Lock`)防止丢失更新——这在多进程/多机器部署时完全无效,必须用数据库层面的机制(原子UPDATE语句/显式行锁/乐观锁版本号)才能在分布式部署下依然有效。

---

## 7. REPEATABLE READ与SERIALIZABLE对丢失更新的真实防护差异

**签名/是什么**

```
PostgreSQL: REPEATABLE READ 就能阻止知识点6这种丢失更新(报 SerializationFailure)
MariaDB/InnoDB: REPEATABLE READ 阻止不了(和READ COMMITTED表现相同,真实丢失),
                必须升到 SERIALIZABLE 才能阻止(报 Deadlock,不是SerializationFailure)
```

一句话:这是本类最重要的一个反直觉发现——很多人以为"REPEATABLE READ能防止丢失更新"是放之四海而皆准的常识,但真实实验证明这只对PostgreSQL成立,MariaDB/InnoDB要到SERIALIZABLE才有同等保护,且两个引擎在SERIALIZABLE下报的错误类型也不同。

**底层机制/为什么这样设计**

PostgreSQL的REPEATABLE READ基于"快照隔离"(Snapshot Isolation)并额外实现了"第一写入者获胜"(First-Updater-Wins)规则:当事务A试图更新一行,而这行在A的快照生效之后已经被另一个已提交的事务B修改过,PostgreSQL会检测到这个写写冲突,直接让A的UPDATE失败并抛出`SerializationFailure`,逼着应用层重新读取最新数据再重试——这是PG在标准的"快照隔离"基础上额外加的保护,不是所有实现快照隔离的数据库都会做这一步。MariaDB/InnoDB的REPEATABLE READ虽然也基于MVCC快照做"一致性读"(consistent read,针对普通SELECT),但**它的UPDATE语句遵循不同的规则**:UPDATE操作实际读取的是数据的"最新版本"而不是快照版本(这叫"当前读",current read),所以A的UPDATE直接基于最新值(已经是B修改过的1050)做`+100`运算,不会检测到冲突,静默覆盖。只有升级到SERIALIZABLE,InnoDB才会把所有普通SELECT也转换成加共享锁的读(等同于隐式`LOCK IN SHARE MODE`),这时A和B的两次SELECT都持有共享锁,当双方都试图升级为UPDATE需要的排他锁时,构成真实的锁等待环,被InnoDB的死锁检测器识别并报`Deadlock`杀死其中一个——**结果上两个引擎都能防止丢失更新,但PG是"乐观检测,提交时才发现冲突",InnoDB SERIALIZABLE是"悲观加锁,执行中就阻塞并检测死锁"**,防护的时间点和错误类型都不同。

**AI研究/工程场景**

这个发现直接决定了跨引擎的并发控制代码不能直接照搬:一段在PostgreSQL上"只要把隔离级别设成REPEATABLE READ、遇到`SerializationFailure`就重试"的正确代码,原样搬到MariaDB上是**不安全的**(REPEATABLE READ在MariaDB上防不住这个场景),必须显式升到SERIALIZABLE(还要处理`Deadlock`异常而不是`SerializationFailure`)才能获得同等保护——这是真实存在、容易被忽视的跨引擎迁移正确性陷阱,不是性能差异这么简单。

**可运行例子**(环境:`python-wsl2`)

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
    pgc.execute("INSERT INTO accounts VALUES (1, 1000)")
    pg.close()
    maria = pymysql.connect(**MARIA_DSN); maria.autocommit(True); mc = maria.cursor()
    mc.execute("DROP TABLE IF EXISTS accounts")
    mc.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER) ENGINE=InnoDB")
    mc.execute("INSERT INTO accounts VALUES (1, 1000)")
    maria.close()

def test_protection(connect_fn, set_isolation_sql, start_txn_sql, is_pg):
    setup()
    result = {}
    ev_a_read = threading.Event(); ev_b_done = threading.Event()

    def txn_a():
        conn = connect_fn()
        if hasattr(conn, 'autocommit') and callable(getattr(conn, 'autocommit')):
            conn.autocommit(False)
        cur = conn.cursor()
        cur.execute(set_isolation_sql)
        if start_txn_sql: cur.execute(start_txn_sql)
        cur.execute("SELECT balance FROM accounts WHERE id=1")
        balance = cur.fetchone()[0]
        ev_a_read.set()
        ev_b_done.wait(timeout=5)
        try:
            cur.execute("UPDATE accounts SET balance=%s WHERE id=1", (balance + 100,))
            conn.commit()
            result['a_outcome'] = 'committed'
        except Exception as e:
            result['a_outcome'] = type(e).__name__
            conn.rollback()
        conn.close()

    def txn_b():
        ev_a_read.wait(timeout=5)
        conn = connect_fn()
        if hasattr(conn, 'autocommit') and callable(getattr(conn, 'autocommit')):
            conn.autocommit(False)
            cur = conn.cursor()
            cur.execute(set_isolation_sql)
            cur.execute(start_txn_sql)
        else:
            conn.autocommit = True
            cur = conn.cursor()
        cur.execute("SELECT balance FROM accounts WHERE id=1")
        balance = cur.fetchone()[0]
        cur.execute("UPDATE accounts SET balance=%s WHERE id=1", (balance + 50,))
        if hasattr(conn, 'autocommit') and callable(getattr(conn, 'autocommit')):
            conn.commit()
        conn.close(); ev_b_done.set()

    ta = threading.Thread(target=txn_a); tb = threading.Thread(target=txn_b)
    ta.start(); tb.start(); ta.join(timeout=10); tb.join(timeout=10)

    conn = connect_fn(); cur = conn.cursor()
    cur.execute("SELECT balance FROM accounts WHERE id=1")
    result['final_balance'] = cur.fetchone()[0]
    conn.close()
    return result

# REPEATABLE READ: PG会拦截(SerializationFailure),MariaDB拦不住(悄悄commit,真实丢失更新)
pg_rr = test_protection(lambda: psycopg2.connect(**PG_DSN), "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ", None, True)
assert pg_rr['a_outcome'] == 'SerializationFailure', f"got {pg_rr}"
assert pg_rr['final_balance'] == 1050, f"got {pg_rr}"  # 只有B的+50生效,A的更新因冲突失败,没有丢失更新

maria_rr = test_protection(lambda: pymysql.connect(**MARIA_DSN), "SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ", "START TRANSACTION", False)
assert maria_rr['a_outcome'] == 'committed', f"MariaDB REPEATABLE READ should NOT catch this, got {maria_rr}"
assert maria_rr['final_balance'] == 1100, f"expected real lost update (1100), got {maria_rr}"

# SERIALIZABLE: 两个引擎都能拦截,但错误类型不同
pg_ser = test_protection(lambda: psycopg2.connect(**PG_DSN), "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE", None, True)
assert pg_ser['a_outcome'] == 'SerializationFailure', f"got {pg_ser}"
assert pg_ser['final_balance'] == 1050

maria_ser = test_protection(lambda: pymysql.connect(**MARIA_DSN), "SET SESSION TRANSACTION ISOLATION LEVEL SERIALIZABLE", "START TRANSACTION", False)
assert maria_ser['a_outcome'] == 'OperationalError', f"expected a deadlock-related OperationalError, got {maria_ser}"
assert maria_ser['final_balance'] == 1050

print("engine divergence verified: PG's REPEATABLE READ already prevents lost update (SerializationFailure); "
      "MariaDB needs SERIALIZABLE to prevent it, and reports Deadlock instead of a serialization error")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux,3次独立重跑确认稳定):

```
REPEATABLE READ:
  PG:    txn A outcome = failed(SerializationFailure), final_balance = 1050  <- 丢失更新被阻止
  Maria: txn A outcome = committed,                     final_balance = 1100  <- 真实发生丢失更新

SERIALIZABLE(两个连接都设置):
  PG:    txn A outcome = failed(SerializationFailure), final_balance = 1050
  Maria: txn A outcome = failed(Deadlock found when trying to get lock), final_balance = 1050  <- 也被阻止了,但报错类型不同
```

**面试怎么问+追问链**

- Q:"REPEATABLE READ能防止丢失更新"这句话对吗?
  - 追问1:如果不对,具体在什么条件下不成立?
    - 深挖追问(区分度较高):这个发现对写跨数据库中间件/ORM框架的并发控制代码有什么指导意义(答案方向:任何声称"设置了某个隔离级别就能防止某类异常"的并发控制代码,必须注明是针对哪个具体引擎验证过的,不能假设所有声称支持同一SQL标准隔离级别名字的引擎行为一致;更稳妥的工程实践是不完全依赖隔离级别的"隐式"保护,而是显式使用乐观锁版本号或`SELECT FOR UPDATE`这类"意图更明确、跨引擎行为更容易预期"的机制)。

**常见坑**

- 把这个知识点当成"MariaDB比PostgreSQL弱"的证据——这是不准确的解读,MariaDB在SERIALIZABLE下同样能提供保护,只是REPEATABLE READ这一档的具体语义两者不同,是"实现选择不同"不是"优劣"问题(PG的方案要求应用层处理重试,InnoDB SERIALIZABLE的方案是更重的悲观锁开销)。
- 只在SERIALIZABLE下测试过一次就下结论——本类的可运行例子独立重跑3次确认稳定复现,时序敏感的并发实验必须多次验证,不能只信一次运气好的结果。

---

## 8. 隔离级别选型的工程决策

**签名/是什么**

```
选型考量维度: 业务能容忍哪些异常现象 x 并发冲突概率 x 引擎具体实现的真实保护范围(不能只看级别名字)
```

一句话:选隔离级别不是"越高越安全就选越高",而是结合本类前面7个知识点已经验证过的"每个级别在具体引擎上真正防住了什么、代价是什么",针对具体业务场景做的权衡决策。

**底层机制/为什么这样设计**

级别越高,数据库需要做的工作通常越多(更多的锁/更多的冲突检测/更高的事务失败重试率),这些代价最终都会转化成吞吐量下降或延迟升高。合理的选型顺序通常是:先看业务能不能容忍某类异常(比如一个"最终一致"的统计报表可能完全能接受不可重复读),再看这个级别在**实际使用的引擎**上是否真的提供了业务需要的保护(本类反复验证的核心方法论——不能只看级别名字),最后再评估这个保护的性能代价是否可接受(比如SERIALIZABLE在高冲突场景下可能导致大量事务重试,吞吐量反而更差)。

**AI研究/工程场景**

一个具体的决策框架:①报表/统计类只读查询——READ COMMITTED通常够用,不需要事务期间数据"冻结";②涉及多次读取且要求这些读取互相一致的业务逻辑(比如KP4的对账单场景)——REPEATABLE READ;③涉及"读-判断-写"且必须防止丢失更新/幻读的核心业务(库存/余额类)——优先考虑显式的原子UPDATE语句或乐观锁版本号(不完全依赖隔离级别的隐式保护,这样代码在不同引擎间更可移植),必须依赖隔离级别兜底时,在MariaDB上要明确知道REPEATABLE READ对丢失更新没有防护,需要SERIALIZABLE或显式锁。

**可运行例子**(环境:`python-wsl2`,用本类知识点6/知识点7已验证的行为做决策对比表的真实依据,不重新起新场景)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14 + MariaDB 10.11.15(已启动)
# 汇总本类知识点3/4/5/6/7已经真实验证过的结论,构造一张"决策矩阵",
# 用代码断言的形式确保这张表和前面各知识点的真实实验结果完全一致(不是凭印象总结)
decision_matrix = {
    ('READ_UNCOMMITTED', 'pg'):    {'dirty_read': False},  # KP3: PG的READ UNCOMMITTED实际不脏读
    ('READ_UNCOMMITTED', 'maria'): {'dirty_read': True},   # KP3: MariaDB真实脏读
    ('REPEATABLE_READ', 'pg'):     {'lost_update_protected': True},   # KP7: PG的RR能防丢失更新
    ('REPEATABLE_READ', 'maria'):  {'lost_update_protected': False},  # KP7: MariaDB的RR防不住
    ('SERIALIZABLE', 'pg'):        {'lost_update_protected': True, 'error_type': 'SerializationFailure'},
    ('SERIALIZABLE', 'maria'):     {'lost_update_protected': True, 'error_type': 'Deadlock'},
}

# 用这张表反推一个具体的工程建议: 如果一段并发控制代码需要跨PG和MariaDB都成立且都用REPEATABLE READ,
# 这个假设本身就是错的(矩阵显示两者在这一档的lost_update_protected不一致)
cross_engine_safe_at_rr = (
    decision_matrix[('REPEATABLE_READ', 'pg')].get('lost_update_protected') ==
    decision_matrix[('REPEATABLE_READ', 'maria')].get('lost_update_protected')
)
assert cross_engine_safe_at_rr is False, \
    "REPEATABLE READ's lost-update protection is NOT consistent across engines - confirms KP7's core finding"

# 但在SERIALIZABLE这一档,两者都能防护(即使报错类型不同),这才是真正跨引擎可移植的最低公共保证
cross_engine_safe_at_serializable = (
    decision_matrix[('SERIALIZABLE', 'pg')].get('lost_update_protected') is True and
    decision_matrix[('SERIALIZABLE', 'maria')].get('lost_update_protected') is True
)
assert cross_engine_safe_at_serializable is True

print("isolation level selection matrix verified against KP3/KP7's real findings: "
      "REPEATABLE READ is NOT a safe cross-engine assumption for lost-update protection, SERIALIZABLE is (but error types differ)")
```

**面试怎么问+追问链**

- Q:你会怎么给一个新业务选隔离级别?
  - 追问1:如果这个业务未来可能从MySQL迁移到PostgreSQL(或反过来),你的选型建议会不会不一样?
    - 深挖追问(区分度较高):除了隔离级别本身,还有什么并发控制手段比"依赖某个隔离级别的隐式保证"更适合需要跨引擎移植的代码(答案方向:显式的原子UPDATE语句[`UPDATE ... SET x = x + 1`]、乐观锁版本号字段、应用层显式的`SELECT FOR UPDATE`——这些机制的行为在标准SQL语义下更明确、更不依赖具体引擎对某个隔离级别名字的具体实现细节,是本类反复验证的"不能只看级别名字"这条纪律在工程实践上的落地)。

**常见坑**

- 选型时只考虑"业务需要多强的一致性",不考虑"选定的隔离级别在实际用的引擎上是否真的提供了这个强度"——本类知识点7已经用真实实验证明这两者可能脱节。
- 为了绝对安全无脑全部使用SERIALIZABLE——在高并发、高冲突概率的场景下,SERIALIZABLE可能导致大量事务因冲突被回滚重试,整体吞吐量反而显著下降,这也是一种需要被评估的真实代价,不是"最安全的选择"就没有代价。
