# 关系模型与基础SQL

> 板块 I(关系模型与SQL基础)第 1 篇。环境:全部 `.venv`(stdlib `sqlite3`,SQLite 3.50.4,原生支持 `RIGHT JOIN`/`FULL JOIN`,验证时以此版本为准)。

## 1. 关系模型三要素:关系/元组/属性/域

**签名/是什么**

```
关系(Relation)  = 一张表(比如 students)
元组(Tuple)      = 表中的一行
属性(Attribute)  = 表的一列(id / name / grade)
域(Domain)       = 某个属性的合法取值集合(比如 grade 的域是 1~12 的整数)
```

一句话:关系模型把数据组织成"表 - 行 - 列 - 合法取值范围"这四层结构,SQL 的一切操作都建立在这四个概念上。

**底层机制/为什么这样设计**

关系模型(E.F. Codd, 1970)的核心洞察是:把数据和"如何存取数据"彻底分离。程序员只需要声明"关系"长什么样(有哪些属性、每个属性的域是什么),不需要关心数据在磁盘上具体怎么排列——这是 SQL 作为**声明式**语言的根源("SELECT 我要什么"而不是"怎么去拿")。域(domain)是经常被忽视但很重要的一环:它不只是"这一列的数据类型",而是"这一列在业务语义上合法的取值集合"。数据类型是域的**必要不充分**近似——`INTEGER` 类型能装下 13,但如果这一列是"年级(1~12)",13 就超出了域。`CHECK` 约束就是数据库引擎用来强制"数据类型之外的域规则"的机制。

**AI研究/工程场景**

设计一张新表本质上就是在定义关系模型的这四要素:先确定这张表代表什么业务实体(关系),再确定它有哪些字段(属性)和每个字段的合法范围(域),最后每条业务数据就是一个元组。工程上常见的"脏数据"问题,很大比例源于只声明了数据类型没有声明域(比如 `age INTEGER` 但没有 `CHECK(age >= 0)`,导致业务代码的 bug 能悄悄写入负数年龄)。

**可运行例子**(环境:`.venv`)

```python
import sqlite3

conn = sqlite3.connect(":memory:")
# ":memory:" 是SQLite的特殊文件名写法,意思是"在内存里建一个临时数据库,不落地成磁盘文件"
# ——conn.close()之后这个数据库就彻底消失,不会在硬盘上留下任何文件。本系列所有.venv例子
# 都用它,是因为每个知识点只关心"这次运行的逻辑对不对",不需要例子之间共享或持久化数据;
# 换成一个真实文件路径(比如sqlite3.connect("test.db"))效果和真实生产用法完全一样。
cur = conn.cursor()
# 游标(cursor):可以理解成"在这个数据库连接(conn)上打开的一个操作窗口"——
# 通过它发送SQL命令(cur.execute(...)),再通过它取回结果(cur.fetchone()/cur.fetchall());
# 一个连接上可以开多个互不干扰的游标(比如同时进行多个独立的查询)。这是Python DB-API
# 标准(sqlite3/psycopg2/pymysql等库都遵循同一套接口规范)的通用用法,后续所有知识点的
# 可运行例子都会用到这个"先拿连接、再拿游标"的模式,不再重复解释。
cur.execute("""
CREATE TABLE students (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    grade INTEGER CHECK(grade >= 1 AND grade <= 12)
)
""")
# 关系(relation)  = students 这张表
# 元组(tuple)     = 一行数据
# 属性(attribute) = id / name / grade 三列
# 域(domain)      = grade 的合法取值集合 {1, 2, ..., 12}
cur.execute("INSERT INTO students VALUES (1, 'Alice', 7)")
conn.commit()

cur.execute("SELECT * FROM students")
row = cur.fetchone()
assert row == (1, "Alice", 7)

# grade=13 超出域范围,即便13本身是合法的INTEGER,也应该被域约束拒绝
try:
    cur.execute("INSERT INTO students VALUES (2, 'Bob', 13)")
    assert False, "expected domain violation to be rejected"
except sqlite3.IntegrityError as e:
    assert "CHECK constraint failed" in str(e)

conn.close()
print("relation/tuple/attribute/domain verified: type-valid but domain-invalid value rejected")
```

**面试怎么问+追问链**

- Q:关系模型里"关系"和我们平时说的"表"是一回事吗?
  - 追问1:那"域"和"数据类型"是一回事吗?举个数据类型相同但域不同导致的真实 bug。
    - 深挖追问(区分度较高):如果一列的合法取值集合会随时间变化(比如"当前有效的国家代码列表"),这种"动态域"该怎么在关系模型里表达?(答案方向:用外键关联一张"国家代码"维表,而不是写死在 CHECK 里——这也是"为什么要拆表"的一个真实动机)

**常见坑**

- 只声明数据类型不声明域约束,是"类型正确但业务错误"的脏数据的主要来源之一。
- 把域约束全部写在应用代码里而不是数据库层(`CHECK`/`FOREIGN KEY`),一旦有第二个写入方(脚本/后台任务/别的服务直连数据库),约束就被绕过了。

---

## 2. 主键/外键/唯一约束/检查约束

**签名/是什么**

```
PRIMARY KEY  唯一标识一行,隐含 NOT NULL + UNIQUE
FOREIGN KEY  引用另一张表的主键,保证引用完整性
UNIQUE       该列(或列组合)在全表内不能重复,但允许多个NULL
CHECK        自定义布尔表达式,插入/更新时必须为真(或NULL)
```

一句话:四种约束分别管"这行是谁"(主键)、"这行指向谁"(外键)、"这列不能撞车"(唯一)、"这行数据是否合理"(检查),数据库引擎在每次写入时自动校验,不需要应用代码手动检查。

**底层机制/为什么这样设计**

约束下推到数据库层的核心价值是**单点强制**:无论数据是从 Web 后端写入、从批处理脚本写入、还是从另一个微服务写入,只要都连的是同一个数据库,约束都会生效,不会因为某个调用方"忘了检查"而破坏数据完整性。外键约束的实现通常依赖被引用列上必须存在索引(否则每次插入/删除都要全表扫描去验证引用关系,代价太高)——这也是"外键列要不要建索引"这个面试常见问题的根源。**一个容易被忽视的平台差异**:SQLite 默认**关闭**外键约束检查(`PRAGMA foreign_keys` 默认是 `0`),必须显式 `PRAGMA foreign_keys = ON` 才会真正校验——这是历史遗留的向后兼容选择,PostgreSQL/MySQL/MariaDB 没有这个开关,外键约束默认就是强制的。

**AI研究/工程场景**

微服务架构下,如果多个服务共享同一个数据库(反模式,但现实中常见),数据库层的约束是防止"某个服务的 bug 写脏另一个服务依赖的数据"的最后一道防线。训练数据 pipeline 场景也类似:如果多个数据采集脚本并发写入同一张标注表,唯一约束能防止重复标注记录静默产生。

**可运行例子**(环境:`.venv`)

```python
import sqlite3

conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.execute("CREATE TABLE departments (dept_id INTEGER PRIMARY KEY, dept_name TEXT UNIQUE NOT NULL)")
cur.execute("""CREATE TABLE employees (
    emp_id INTEGER PRIMARY KEY, name TEXT NOT NULL, dept_id INTEGER,
    salary INTEGER CHECK(salary > 0),
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
)""")
cur.execute("INSERT INTO departments VALUES (1, 'Engineering')")
conn.commit()

# 坑:SQLite 默认关闭外键检查(PRAGMA foreign_keys 默认是 0)
cur.execute("PRAGMA foreign_keys")
assert cur.fetchone()[0] == 0, "SQLite defaults to foreign_keys OFF"
cur.execute("INSERT INTO employees VALUES (100, 'Ghost', 999, 5000)")  # dept_id=999 不存在
conn.commit()
cur.execute("SELECT COUNT(*) FROM employees WHERE emp_id=100")
assert cur.fetchone()[0] == 1, "without PRAGMA, bad FK reference is silently accepted"
cur.execute("DELETE FROM employees WHERE emp_id=100")
conn.commit()

# 打开外键检查后,四种约束逐一验证
conn.execute("PRAGMA foreign_keys = ON")
cur.execute("INSERT INTO employees VALUES (1, 'Alice', 1, 5000)")
conn.commit()

try:
    cur.execute("INSERT INTO employees VALUES (1, 'Bob', 1, 6000)")  # 主键重复
    assert False
except sqlite3.IntegrityError as e:
    assert "UNIQUE constraint failed" in str(e)

try:
    cur.execute("INSERT INTO employees VALUES (2, 'Carol', 999, 5000)")  # 外键指向不存在的部门
    assert False
except sqlite3.IntegrityError as e:
    assert "FOREIGN KEY constraint failed" in str(e)

try:
    cur.execute("INSERT INTO departments VALUES (2, 'Engineering')")  # UNIQUE(dept_name)重复
    assert False
except sqlite3.IntegrityError as e:
    assert "UNIQUE constraint failed" in str(e)

try:
    cur.execute("INSERT INTO employees VALUES (3, 'Dave', 1, -100)")  # CHECK(salary > 0)失败
    assert False
except sqlite3.IntegrityError as e:
    assert "CHECK constraint failed" in str(e)

conn.close()
print("all four constraint types verified; SQLite foreign_keys-off-by-default gotcha confirmed")
```

**面试怎么问+追问链**

- Q:主键和唯一约束(UNIQUE)有什么区别?
  - 追问1:一张表可以有几个主键?可以有几个唯一约束?为什么主键只能有一个但唯一约束可以有多个?
    - 深挖追问(区分度较高):外键列要不要建索引?不建会有什么后果(具体到:删除父表一行时,数据库要去检查子表有没有引用它,没索引这一步是全表扫描;高并发下这类检查还可能放大锁的范围)。

**常见坑**

- SQLite 默认关闭外键检查,忘记 `PRAGMA foreign_keys = ON` 会让外键约束形同虚设——这是本例验证过的真实行为,不是理论上的警告。
- 以为 `UNIQUE` 和 `NOT NULL` 组合等于主键:实际上 `UNIQUE` 列允许多个 `NULL`(NULL 不参与唯一性比较),而主键隐含 `NOT NULL`,语义不完全等价。

---

## 3. DDL基础:CREATE/ALTER/DROP的语义与代价

**签名/是什么**

```
CREATE TABLE ...   创建新的关系(表结构)
ALTER TABLE  ...   修改已有关系的结构(加列/改列/加约束等)
DROP TABLE   ...   删除整个关系(结构+数据一起消失)
```

一句话:DDL(Data Definition Language)操作的是表结构本身而不是表里的数据行,三种操作的**代价**(要不要重写全表数据)是这一类知识点里最容易被忽视但面试常问的部分。

**底层机制/为什么这样设计**

`ALTER TABLE ADD COLUMN` 在 SQLite 和 PostgreSQL(11版本起)/大多数现代引擎里都是**纯元数据操作**:新列的默认值记录在表的元数据里,已有的行不会被物理重写,读取旧行时新列的值是"惰性"计算出来的默认值——这就是为什么给一张几亿行的表加一列默认值几乎是瞬间完成,而不是要花几个小时重写全表(这是 PostgreSQL 11 之前的行为,现代版本已经优化)。但 `ALTER TABLE ... ALTER COLUMN` 改变列的数据类型、或者加一个需要检查所有已有行是否满足的约束(比如给已有数据加 `NOT NULL`),通常还是要全表扫描验证,代价和数据量成正比。`DROP TABLE` 是最"便宜"也最危险的操作——只是删除元数据指针和释放数据页,没有任何"是否有行被误删"的确认机制。

**AI研究/工程场景**

给生产环境一张大表加字段,是否需要停机/是否会长时间锁表,直接取决于这个"代价"问题——这也是为什么"加字段容易,加约束/改类型要谨慎评估"是数据库运维里的常识。

**可运行例子**(环境:`.venv`)

```python
import sqlite3

conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
cur.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")
conn.commit()

# ALTER TABLE ADD COLUMN: 元数据操作,已有行的新列惰性取默认值,不重写数据
cur.execute("ALTER TABLE t ADD COLUMN age INTEGER DEFAULT 0")
cur.execute("SELECT * FROM t")
rows = cur.fetchall()
assert rows == [(1, "a", 0), (2, "b", 0)]

# DROP TABLE: 结构+数据一起消失,没有任何"确认"步骤,不可逆
cur.execute("DROP TABLE t")
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='t'")
assert cur.fetchone() is None, "table should be completely gone after DROP"

conn.close()
print("DDL semantics verified: ADD COLUMN is metadata-only, DROP is irreversible with zero confirmation")
```

**面试怎么问+追问链**

- Q:给一张有 10 亿行的生产表加一列,要注意什么?
  - 追问1:如果这一列还要求 `NOT NULL` 呢?和只加一个可为 NULL 的列比,代价有什么本质区别?
    - 深挖追问(区分度较高):除了执行代价,`ALTER TABLE` 通常需要对表加排他锁(exclusive lock,即锁住这张表期间不允许其他事务同时读写它,至少是短暂的;锁机制的完整讲解见05类"MVCC与锁机制",这里只需要知道"排他"就是"独占,别人进不来"),这在高QPS的在线服务上意味着什么(答案方向:锁等待队列可能瞬间堵住大量业务查询,即使实际锁的时间很短,也可能造成请求超时的连锁反应)。

**常见坑**

- 把 `DROP TABLE` 和 `DELETE FROM table`(清空数据但保留表结构)搞混,前者连表结构定义都没了。
- 想当然认为所有 `ALTER TABLE` 操作代价都一样低——只有"加一个有默认值的可空列"这类操作是真正的元数据操作,加约束、改类型往往需要扫描验证全表。

---

## 4. DML基础:INSERT/UPDATE/DELETE的原子性

**签名/是什么**

```
INSERT  插入新行
UPDATE  修改已有行
DELETE  删除已有行
```

一句话:单条 DML 语句本身是原子的——如果语句处理到一半失败(比如插入多行时某一行违反约束),数据库会把这条语句已经做的部分修改全部撤销,不会留下"部分生效"的中间状态。

**底层机制/为什么这样设计**

这个"语句级原子性"是数据库引擎在写入路径上维护的一个不变量,独立于"事务"这个更高层的概念存在(事务是多条语句合在一起的原子性,04 类会展开讲)。它的价值在于:应用代码不需要为"一条 INSERT 语句执行到第 3 行失败,前 2 行要不要手动删掉"这种情况写补偿逻辑——数据库保证要么这条语句全部生效,要么完全不生效。

**AI研究/工程场景**

批量导入训练样本、批量写入标注结果这类场景经常一次性 `INSERT` 上千行,如果不了解语句级原子性,遇到某一行数据格式错误导致插入失败时,容易误判"是不是前面已经成功的部分也要清理",实际上数据库已经自动保证了不会有部分残留。

**可运行例子**(环境:`.venv`)

```python
import sqlite3

conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER CHECK(val > 0))")
conn.commit()

# 单条INSERT语句插入4行,第3行(val=-5)违反CHECK约束
try:
    cur.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, -5), (4, 40)")
    conn.commit()
    assert False, "expected constraint violation"
except sqlite3.IntegrityError:
    conn.rollback()

cur.execute("SELECT COUNT(*) FROM t")
count = cur.fetchone()[0]
# 语句级原子性:前两行(val=10,20)不会残留,整条语句的效果被完全撤销
assert count == 0, f"expected 0 rows due to statement-level atomicity, got {count}"

# 对照组:全部合法的4行,整条语句完全生效
cur.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30), (4, 40)")
conn.commit()
cur.execute("SELECT COUNT(*) FROM t")
assert cur.fetchone()[0] == 4

conn.close()
print("DML statement-level atomicity verified: partial failure leaves zero rows, full success leaves all rows")
```

**面试怎么问+追问链**

- Q:一条 INSERT 语句同时插入 1000 行,其中第 500 行数据有问题,最终库里会有多少行残留?
  - 追问1:如果这 1000 行是拆成 1000 条独立的 INSERT 语句(不在一个事务里)执行,第 500 条失败,结果会一样吗?
    - 深挖追问(区分度较高):这个差异说明了"语句级原子性"和"事务级原子性"不是一回事——没有显式开事务时,1000条独立语句之间没有整体原子性保证,这正是 04 类"事务与隔离级别"要解决的问题,这里先建立"单条语句本身是原子的"这个更基础的认知。

**常见坑**

- 把"语句级原子性"误当成"事务级原子性":多条独立语句(没有显式 `BEGIN`/`COMMIT` 包裹)之间没有整体回滚保证,只有单条语句内部有。
- 批量插入失败后手动写"清理已插入的部分"的补偿代码——多余,数据库已经保证不会有部分残留。

---

## 5. 四种JOIN类型与执行逻辑

**签名/是什么**

```
INNER JOIN   只保留两张表都能匹配上的行
LEFT JOIN    保留左表全部行,右表未匹配的列填NULL
RIGHT JOIN   保留右表全部行,左表未匹配的列填NULL
FULL JOIN    两张表所有行都保留,未匹配的一侧填NULL
```

一句话:四种 JOIN 的区别只在于"对未匹配的行怎么处理"——INNER 直接丢弃,LEFT/RIGHT/FULL 按不同方向保留并用 NULL 填充缺失部分。

**底层机制/为什么这样设计**

JOIN 的执行逻辑可以理解为"先做笛卡尔积,再按 ON 条件过滤,再决定要不要把未匹配的行用 NULL 补回来"(真实引擎不会真的算笛卡尔积,会用哈希连接/嵌套循环/归并连接等算法优化,但逻辑语义等价于这个模型)。选 LEFT 还是 RIGHT 本质上只是"以哪张表为准"的可读性选择——`A LEFT JOIN B` 和 `B RIGHT JOIN A` 逻辑等价,所以工程上 `RIGHT JOIN` 用得比 `LEFT JOIN` 少得多(把"基准表"习惯性写在左边可读性更好,`RIGHT JOIN` 常常是重构时把表顺序换了但没改JOIN类型的产物)。

**AI研究/工程场景**

"查询所有用户,包括还没有下过订单的"这类需求必须用 LEFT JOIN(以用户表为基准);"统计所有订单,包括那些因为脏数据关联不到用户的"则要用 RIGHT JOIN 或者反过来 LEFT JOIN 订单表——JOIN 方向选错是数据分析里"明明有 100 个用户,统计出来却只有 80 个"这类问题的常见根源。

**可运行例子**(环境:`.venv`,SQLite 3.50.4 原生支持全部四种JOIN)

```python
import sqlite3

conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")
cur.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount INTEGER)")
cur.executemany("INSERT INTO customers VALUES (?, ?)", [(1, "Alice"), (2, "Bob"), (3, "Carol")])
# order 4 的 customer_id=99 不存在于 customers 表(模拟脏数据/已删除用户的历史订单)
cur.executemany("INSERT INTO orders VALUES (?, ?, ?)", [(1, 1, 100), (2, 1, 200), (3, 2, 50), (4, 99, 999)])
conn.commit()

cur.execute("SELECT c.name, o.amount FROM customers c INNER JOIN orders o ON c.id = o.customer_id")
inner = cur.fetchall()
assert len(inner) == 3  # Carol(无订单)和order4(customer_id=99不存在)都不出现
assert ("Carol", None) not in inner

cur.execute("SELECT c.name, o.amount FROM customers c LEFT JOIN orders o ON c.id = o.customer_id ORDER BY c.name")
left = cur.fetchall()
assert len(left) == 4  # Carol保留,amount为NULL
assert ("Carol", None) in left

cur.execute("SELECT c.name, o.amount FROM customers c RIGHT JOIN orders o ON c.id = o.customer_id ORDER BY o.id")
right = cur.fetchall()
assert len(right) == 4  # order4保留,name为NULL
assert (None, 999) in right

cur.execute("SELECT c.name, o.amount FROM customers c FULL JOIN orders o ON c.id = o.customer_id")
full = cur.fetchall()
assert len(full) == 5  # Carol和order4都保留
assert ("Carol", None) in full and (None, 999) in full

conn.close()
print("all four JOIN types verified: INNER=3 LEFT=4 RIGHT=4 FULL=5 rows, matching set-theoretic expectations")
```

**面试怎么问+追问链**

- Q:LEFT JOIN 和 INNER JOIN 什么时候结果一样?
  - 追问1:如果左表有 100 行,LEFT JOIN 右表之后结果一定还是 100 行吗?
    - 深挖追问(区分度较高):如果右表对同一个 customer_id 有多条订单(一对多关系),LEFT JOIN 之后行数会怎么变化?这对做聚合统计(比如"每个用户的订单数")意味着什么陷阱(答案方向:行数会膨胀成"左表行数 × 匹配的右表行数",如果这时候不小心对左表的某个字段做 `COUNT`,会被重复计数,必须配合 `DISTINCT` 或先聚合子查询再 JOIN)。

**常见坑**

- 一对多 JOIN 导致行数膨胀后,直接对基准表的字段做聚合(比如 `SUM`/`COUNT`)会重复计数,是数据分析里最常见的 JOIN 相关 bug 之一。
- `RIGHT JOIN` 在旧版本 SQLite(3.39之前)不支持,可移植性不如把表顺序换过来写成 `LEFT JOIN`。

---

## 6. NULL的三值逻辑陷阱

**签名/是什么**

```
SQL 里的布尔表达式有三种结果: TRUE / FALSE / NULL(未知)
任何值和 NULL 比较(=、<>、<、>等)结果都是 NULL,不是TRUE也不是FALSE
```

一句话:SQL 不是两值逻辑(真/假),而是三值逻辑(真/假/未知),`WHERE` 子句只保留结果为 TRUE 的行——结果为 NULL 或 FALSE 的行都会被过滤掉,这是很多"为什么明明有数据却查不出来"问题的根源。

**底层机制/为什么这样设计**

NULL 在关系模型里表示"未知"而不是"某个具体的空值"。既然是"未知",那么 `未知 = 未知` 的结果按逻辑就应该也是"未知"而不是"真"——两个都不知道的东西怎么能确定它们相等?这是三值逻辑在设计上的一致性,不是引擎的怪癖。这个规则会像"毒药"一样污染包含它的复合表达式:`NULL AND FALSE` 结果是 `FALSE`(因为不管另一边是什么,FALSE AND 任何值都是FALSE),但 `NULL AND TRUE` 结果是 `NULL`(不确定)。`IN`/`NOT IN` 在遇到子查询结果集里含 NULL 时会被"整体污染"——这是本类里最容易在真实业务代码里踩的坑。

**AI研究/工程场景**

数据清洗/特征工程阶段,如果某个业务字段允许为 NULL(比如"用户的可选简介"),用 `WHERE bio != '某敏感词'` 这类否定条件做过滤时,所有 `bio` 为 NULL 的行会被**静默排除**在结果之外(而不是被当成"不含敏感词"保留)——这在做数据统计/训练集筛选时是一个很隐蔽的样本丢失来源。

**可运行例子**(环境:`.venv`)

```python
import sqlite3

conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.execute("CREATE TABLE t (id INTEGER, val INTEGER)")
cur.executemany("INSERT INTO t VALUES (?, ?)", [(1, 10), (2, None), (3, 20)])
conn.commit()

# 坑1: NULL = NULL 不是 TRUE,是 NULL(未知),WHERE 永远过滤掉它,一行都查不到
cur.execute("SELECT * FROM t WHERE val = NULL")
assert cur.fetchall() == [], "val = NULL should never match anything"

# 正确写法必须用 IS NULL
cur.execute("SELECT * FROM t WHERE val IS NULL")
assert cur.fetchall() == [(2, None)]

# 坑2: NOT IN 遇到子查询结果集含NULL会被整体污染
# 子查询 (SELECT val FROM t WHERE id IN (1,2)) 返回 (10, NULL)
cur.execute("SELECT id FROM t WHERE val NOT IN (SELECT val FROM t WHERE id = 2 OR id = 1)")
poisoned = cur.fetchall()
assert poisoned == [], f"NOT IN against a NULL-containing set should match nothing, got {poisoned}"

# 对照组:子查询结果集不含NULL时,NOT IN 正常工作
cur.execute("SELECT id FROM t WHERE val NOT IN (10)")
clean = cur.fetchall()
assert clean == [(3,)], f"expected only id=3 (val=20), got {clean}"

conn.close()
print("NULL three-valued logic verified: val=NULL matches nothing, NOT IN poisoned by NULL in the comparison set")
```

**面试怎么问+追问链**

- Q:为什么 `WHERE val != 10` 查不出 `val` 为 NULL 的那些行?这是bug吗?
  - 追问1:那怎么写一个条件,能同时覆盖"val不等于10"和"val是NULL"这两种情况?
    - 深挖追问(区分度较高):`COUNT(*)` 和 `COUNT(某列)` 在这个字段有NULL时行为一样吗(答案方向:`COUNT(*)` 数所有行,`COUNT(column)` 只数该列非NULL的行,这也是三值逻辑影响聚合函数的一个具体表现,新手很容易写出偏低的计数结果而不自知)。

**常见坑**

- `NOT IN (子查询)` 只要子查询结果集里混进一个 NULL,整个 `NOT IN` 条件对所有行都不成立——这是本类实测验证过的真实行为,排查这类"明明数据存在却查不到"的问题时要第一时间怀疑子查询里有没有 NULL。
- 用 `!=`/`<>` 做否定过滤时忘记 NULL 行会被自动排除,导致统计结果比预期少。

---

## 7. DISTINCT与集合运算:UNION/INTERSECT/EXCEPT

**签名/是什么**

```
DISTINCT    对结果集去重
UNION       两个结果集的并集,自动去重
UNION ALL   两个结果集的并集,不去重
INTERSECT   两个结果集的交集
EXCEPT      两个结果集的差集(左边有、右边没有的部分)
```

一句话:这四个集合运算符都要求参与运算的两个 `SELECT` 列数相同、对应列类型兼容,`UNION`/`INTERSECT`/`EXCEPT` 默认都会去重(因为语义上是集合运算,集合里不该有重复元素),只有 `UNION ALL` 保留重复。

**底层机制/为什么这样设计**

去重(无论是显式 `DISTINCT` 还是 `UNION` 隐含的去重)在实现上通常需要排序或哈希整个结果集来找出重复项——这是一个和结果集大小相关的额外开销,不是"免费"的。这正是为什么 `UNION ALL` 存在:当调用方已经确定两个结果集不会有重复(比如按不同的、互斥的条件从同一张表查两次),用 `UNION ALL` 可以省掉这次去重的计算成本。`INTERSECT`/`EXCEPT` 的执行通常基于对两个输入排序后做归并比较,或者对较小的一侧建哈希表再探测较大的一侧。

**AI研究/工程场景**

"找出同时出现在A组和B组的用户"(INTERSECT)、"找出在A组但不在B组的用户"(EXCEPT,常用于"流失用户"这类分析)、"合并两个数据源但不要重复记录"(UNION)是数据分析里的高频操作;明确知道自己的场景是否可能有重复、需不需要去重,能直接决定该用 `UNION` 还是 `UNION ALL`,避免为不需要的去重付出额外的计算代价。

**可运行例子**(环境:`.venv`)

```python
import sqlite3

conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.execute("CREATE TABLE a (x INTEGER)")
cur.execute("CREATE TABLE b (x INTEGER)")
cur.executemany("INSERT INTO a VALUES (?)", [(1,), (2,), (2,), (3,)])
cur.executemany("INSERT INTO b VALUES (?)", [(2,), (3,), (4,)])
conn.commit()

cur.execute("SELECT DISTINCT x FROM a ORDER BY x")
assert cur.fetchall() == [(1,), (2,), (3,)]  # a里的重复的2被去重

cur.execute("SELECT x FROM a UNION SELECT x FROM b ORDER BY x")
assert cur.fetchall() == [(1,), (2,), (3,), (4,)]  # 并集去重

cur.execute("SELECT x FROM a UNION ALL SELECT x FROM b ORDER BY x")
union_all = cur.fetchall()
assert len(union_all) == 7  # 4(a的行数,含重复) + 3(b的行数),不去重

cur.execute("SELECT x FROM a INTERSECT SELECT x FROM b ORDER BY x")
assert cur.fetchall() == [(2,), (3,)]  # 交集:两边都有的值

cur.execute("SELECT x FROM a EXCEPT SELECT x FROM b ORDER BY x")
assert cur.fetchall() == [(1,)]  # 差集:a有b没有的值(注意EXCEPT不对称,B EXCEPT A结果会不同)

conn.close()
print("DISTINCT and set operations verified: UNION dedups, UNION ALL keeps 7 rows, INTERSECT/EXCEPT match set theory")
```

**面试怎么问+追问链**

- Q:`UNION` 和 `UNION ALL` 该怎么选?
  - 追问1:如果明确知道两个结果集不可能有重复行,还有必要用 `UNION` 吗?
    - 深挖追问(区分度较高):`A EXCEPT B` 和 `B EXCEPT A` 是同一个东西吗?如果不是,分别对应什么业务问题(答案方向:不对称——`A EXCEPT B` 是"A有B没有"[比如"注册了但没下单的用户"],`B EXCEPT A` 是"B有A没有"[比如"下了单但不在当前有效用户列表里的异常订单"],两者业务含义完全不同,搞反是真实会发生的低级错误)。

**常见坑**

- 明知结果集不会重复还习惯性用 `UNION`,白白付出一次不必要的去重开销(数据量大时差异明显)。
- 混淆 `EXCEPT` 的方向性,`A EXCEPT B` 写反成 `B EXCEPT A` 会得到语义完全相反的结果而不容易第一时间发现。
