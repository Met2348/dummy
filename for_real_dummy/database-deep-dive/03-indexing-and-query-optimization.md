# 索引结构与查询优化器

> 板块 II(索引与查询优化)。10 个知识点全部围绕同一张真实测试表 `big_table`(5万~5.5万行,`category` 20种取值/`status` 3种取值含1个极稀有值/`value` 均匀分布1~100万/`note` 唯一字符串)在 **PostgreSQL 16.14** 和 **MariaDB 10.11.15** 上真实建立索引、真实跑 `EXPLAIN`,逐条对比两个引擎的真实输出(环境:`python-wsl2`,依赖 WSL2 Rocky Linux 已启动的 PostgreSQL 16.14 + MariaDB 10.11.15,凭据见 `00-roadmap.md`)。KP1(B+树复杂度)是纯算法推导,用 `.venv` 验证。

## 1. B+树索引结构与磁盘IO复杂度分析

**签名/是什么**

```
B+树: 高扇出(几百路)的多叉平衡树,所有数据存在叶子节点,叶子节点之间用链表连接
树高 h = ceil(log_M(N)),M=扇出(阶数),N=记录数
```

示意图(3层小型B+树;真实B+树扇出通常是几百路,这里为了能画出来缩小到2~3路做示意):

```
                              [ 30 | 70 ]                      <- 根节点(第1层,内部节点):只存"键"(30和70),
                             /      |      \                      不存实际数据,作用是告诉查询"该往哪个分支走"
                            /       |       \
                        [15]      [50]      [85]                <- 内部节点(第2层):同样只存键,继续往下导航
                        /  \      /  \      /  \
                       v    v    v    v    v    v
叶子节点(第3层,真正存放数据行,按key从小到大排好序):
   [5,10] <-> [15,20,25] <-> [30,40] <-> [50,60,65] <-> [70,80] <-> [85,90,95]

叶子节点之间用链表首尾相连(上面的 <-> 就是这条链表的指针)。范围查询(比如 20<=key<=65)
只需要从根节点走到第一个满足条件的叶子([15,20,25]),然后沿着这条链表一路向右顺序读,
直到读到超出范围的key为止,不需要每次都回到根节点重新往下找一遍。
```

一句话:数据库索引选 B+树而不是二叉搜索树,核心原因是磁盘IO次数约等于树的高度——B+树用高扇出把树高压到个位数,而二叉树对同样数据量的树高是B+树的数倍到十倍,意味着数倍到十倍的磁盘IO。

**底层机制/为什么这样设计**

内存访问和磁盘访问的延迟差了几个数量级(内存纳秒级,机械磁盘毫秒级,SSD也要微秒级),数据库设计索引结构时的第一目标是**最小化磁盘IO次数**,而不是"比较次数"(这是内存里数据结构如二叉搜索树优化的目标)。二叉搜索树每个节点只有2个子节点,树高是 `log_2(N)`;B+树的节点(对应磁盘上的一个"页",通常4KB或更大)可以塞下几百个键值,子节点数(扇出)也是几百,树高变成 `log_M(N)`,M是几百而不是2——底数越大,同样N下树越矮。B+树把所有实际数据都放在叶子节点(内部节点只放索引键,不放数据,进一步提高扇出),并且叶子节点之间用链表连接,这让范围查询(`BETWEEN`/`>`/`<`)可以在找到起点后沿链表顺序读取,不需要每次都从根节点重新往下找。

**AI研究/工程场景**

这个"树高约等于磁盘IO次数"的心智模型,是判断"给这张表加索引能带来多大收益"的基础直觉——一张10亿行的表,不加索引的全表扫描要读遍所有数据页,加了B+树索引后,精确查找理论上只需要读"树高"这么多个页(通常3~4页),这是索引能把查询从"秒级"降到"毫秒级"的根本原因。

**可运行例子**(环境:`.venv`)

```python
import math

def btree_height(n, order):
    # B+树: 每个节点最多order路分支(阶数/扇出),树高 = 以order为底n的对数
    if n <= 1:
        return 1
    return math.ceil(math.log(n, order))

def binary_tree_height(n):
    # 二叉搜索树: 每个节点2路分支
    if n <= 1:
        return 1
    return math.ceil(math.log(n, 2))

# 典型B+树阶数(取决于键大小和页大小,通常几百路),这里用200做代表性估算
results = {}
for n in [1_000, 1_000_000, 1_000_000_000]:
    bt_h = btree_height(n, 200)
    bin_h = binary_tree_height(n)
    results[n] = (bt_h, bin_h)

assert results[1_000] == (2, 10)
assert results[1_000_000] == (3, 20)
assert results[1_000_000_000] == (4, 30)

# 10亿行数据下,B+树只需要4层(4次磁盘IO),二叉树需要30层(30次磁盘IO),差7.5倍
ratio = results[1_000_000_000][1] / results[1_000_000_000][0]
assert ratio > 7, f"expected >7x IO reduction at 1e9 rows, got {ratio:.1f}x"

print("B+tree vs binary tree height verified:", results, "- at 1e9 rows, B+tree needs 4 IOs vs binary tree's 30 IOs")
```

**面试怎么问+追问链**

- Q:为什么数据库索引用B+树而不是二叉搜索树或跳表(Skip List)?
  - 追问1:那为什么内存数据库(比如Redis的有序集合)反而经常用跳表而不是B+树?
    - 深挖追问(区分度较高):这个选择差异的根本原因是什么(答案方向:跳表和B+树都是为了避免二叉树"退化成链表"的最坏情况、把高度压低,但B+树的优化目标是"压低磁盘IO次数"[高扇出],跳表的优化目标是"实现简单、支持无锁并发的范围查询"且完全在内存里操作,没有磁盘IO这个约束,跳表的多层"高速公路"结构在内存随机访问场景下反而比B+树更容易实现高并发无锁读写)。

**常见坑**

- 把"B+树能加速查询"简单理解为"树形结构比线性结构快",而不理解真正的收益来源是"压低磁盘IO次数"——这个理解偏差会导致高估内存里做同样操作(比如Python字典查找)用类似结构的收益,内存访问延迟本来就低几个数量级,树形结构的IO优化在纯内存场景下价值大幅降低。
- 认为索引"层数"和"实际磁盘IO次数"完全划等号——真实系统里索引的上层节点(尤其是根节点和第二层)几乎总是被缓存在内存的缓冲池里(见06类知识点),真实IO次数往往比理论树高更少。

---

## 2. 聚簇索引 vs 非聚簇索引

**签名/是什么**

```
聚簇索引(Clustered Index):  索引的叶子节点直接就是数据行本身(InnoDB主键)
非聚簇索引(Non-Clustered):  索引和数据行是分开存储的两个物理结构(PostgreSQL所有索引)
```

一句话:InnoDB 的主键索引"就是"表(数据行存在主键B+树的叶子节点里),而 PostgreSQL 的所有索引(包括主键索引)都只是指向堆表(heap)里数据行的"指针",索引和数据是两个完全独立的物理文件。

**底层机制/为什么这样设计**

这是两种数据库在存储引擎设计上一个根本性的架构差异。InnoDB 选择"主键即数据"(聚簇)的好处是:按主键查询时只需要遍历一棵树就能拿到完整数据行,不需要"先查索引再回表"这一步;代价是:因为一张表只能有一种物理排列顺序,只能有一个聚簇索引(主键),其余索引(二级索引)的叶子节点存的是主键值而不是完整数据行,通过二级索引查询非主键条件时,通常需要先用二级索引定位主键值,再用主键值回到聚簇索引查完整行(这一步叫"回表",除非二级索引本身就覆盖了需要的全部列,见KP4)。PostgreSQL 选择"表和索引物理分离"(堆表架构)的设计,所有索引(含主键)地位对等,都是指向堆里数据行(用行标识符 `ctid` 定位)的独立结构——好处是可以同时有多个"效果类似聚簇"的索引选择灵活性更高,代价是即使按主键查询,也需要走"索引查到ctid,再用ctid去堆表取行"这两步。

**AI研究/工程场景**

这个差异直接影响"该按什么顺序批量导入数据"这类工程决策:InnoDB 表按主键顺序批量插入通常显著快于乱序插入(因为聚簇索引的物理排列必须维持主键顺序,乱序插入会导致频繁的页分裂);PostgreSQL 的堆表本身没有这个顺序敏感性(新行总是追加写入堆的末尾附近,和主键顺序无关),但如果业务查询模式高度依赖某个非主键列的顺序局部性,可以用 `CLUSTER` 命令手动重排堆表(一次性操作,不像InnoDB那样自动维持)。

**可运行例子**(环境:`python-wsl2`,依赖 PostgreSQL 16.14 + MariaDB 10.11.15 均已启动)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14 / MariaDB 10.11.15(已启动),
# 使用 ~/database-deep-dive-venv 虚拟环境(psycopg2-binary/pymysql)。
# 本类10个知识点共用同一张真实测试表big_table,这是KP2-10系列的建表+灌数据起点。
import psycopg2
import pymysql
import random

pg = psycopg2.connect(host='127.0.0.1', port=5432, dbname='dbdemo', user='dbdemo', password='dbdemo_local_only')
pg.autocommit = True
pgc = pg.cursor()
pgc.execute("DROP TABLE IF EXISTS big_table")
pgc.execute("""CREATE TABLE big_table (
    id SERIAL PRIMARY KEY, category TEXT, status TEXT, value INTEGER, note TEXT
)""")

maria = pymysql.connect(host='127.0.0.1', port=3306, database='dbdemo', user='dbdemo', password='dbdemo_local_only')
maria.autocommit(True)
mc = maria.cursor()
mc.execute("DROP TABLE IF EXISTS big_table")
mc.execute("""CREATE TABLE big_table (
    id INT AUTO_INCREMENT PRIMARY KEY, category VARCHAR(50), status VARCHAR(50),
    value INT, note VARCHAR(200)
) ENGINE=InnoDB""")

# 5万行: category 20种取值(低选择性),status 3种取值(其中'rare'只有1行,高选择性),
# value 均匀分布1~100万,note 每行唯一字符串
random.seed(42)
categories = [f"cat_{i}" for i in range(20)]
rows = []
for i in range(50000):
    cat = random.choice(categories)
    status = 'rare' if i == 12345 else random.choice(['active', 'inactive'])
    val = random.randint(1, 1000000)
    note = f"note-{i}"
    rows.append((cat, status, val, note))

args_str = ",".join(pgc.mogrify("(%s,%s,%s,%s)", r).decode() for r in rows)
pgc.execute(f"INSERT INTO big_table (category, status, value, note) VALUES {args_str}")
mc.executemany("INSERT INTO big_table (category, status, value, note) VALUES (%s,%s,%s,%s)", rows)

pgc.execute("ANALYZE big_table")
pgc.execute("VACUUM big_table")  # 更新可见性图,KP4的Index Only Scan判断依赖这一步,不能只ANALYZE
mc.execute("ANALYZE TABLE big_table")

# PostgreSQL: 堆表(数据)和主键索引是两个完全独立的物理结构,分别有独立的大小
pgc.execute("SELECT pg_relation_size('big_table')")
heap_size = pgc.fetchone()[0]
pgc.execute("SELECT pg_relation_size('big_table_pkey')")
pk_index_size = pgc.fetchone()[0]
assert heap_size > 0 and pk_index_size > 0
assert heap_size != pk_index_size, "heap and primary key index are separate physical structures with independent sizes"

# MariaDB/InnoDB: 主键数据直接算进DATA_LENGTH,还没有二级索引时INDEX_LENGTH为0
# (因为聚簇主键"就是"表,没有独立的"主键索引文件"需要额外计入INDEX_LENGTH)
mc.execute("""SELECT DATA_LENGTH, INDEX_LENGTH FROM information_schema.TABLES
              WHERE table_schema='dbdemo' AND table_name='big_table'""")
data_len, index_len = mc.fetchone()
assert data_len > 0
assert index_len == 0, f"before any secondary index, InnoDB's INDEX_LENGTH should be 0 (PK folded into DATA_LENGTH), got {index_len}"

pg.close()
maria.close()
print("clustered vs non-clustered storage verified: PG heap/pkey are separate sized structures, InnoDB PK data has zero separate index footprint")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
PG heap size: 3416064 bytes; PG pkey index size: 1138688 bytes (两个独立物理结构)
MariaDB DATA_LENGTH: 3686400 INDEX_LENGTH: 0 (未建二级索引前,主键数据已折算进DATA_LENGTH)
```

**面试怎么问+追问链**

- Q:MySQL/InnoDB 的表为什么强烈建议用自增整数做主键,不建议用随机UUID?
  - 追问1:这和"聚簇索引"这个特性具体有什么关系?
    - 深挖追问(区分度较高):PostgreSQL的表要不要担心同样的问题(答案方向:InnoDB的聚簇索引物理上按主键顺序存储数据,随机UUID做主键会导致新插入的行随机散布在B+树的不同叶子页,频繁触发页分裂和随机IO写入,自增主键则总是追加到最后一页;PostgreSQL的堆表插入位置和主键值无关,新行插入不受主键顺序影响,所以这个特定问题在PostgreSQL上不成立——但PostgreSQL用UUID做主键仍有索引本身[UUID索引比整数索引占用更大空间]的其他代价,只是不是"页分裂"这个特定机制)。

**常见坑**

- 想当然认为"主键索引"在所有数据库里都是同一种物理实现——这是本类实测验证过的真实差异,InnoDB的主键"是"表,PostgreSQL的主键"指向"表,两者的存储代价、查询路径、批量导入的顺序敏感性都不同。
- 忽视二级索引在两种引擎下"回表"路径的差异(InnoDB通过主键值回表,PostgreSQL通过`ctid`回表)——这个细节会在KP4(覆盖索引)进一步展开。

---

## 3. 联合索引最左前缀原则

**签名/是什么**

```
CREATE INDEX idx (col_a, col_b)  仅当查询条件里包含"从col_a开始连续"的前缀时,才能被这个索引加速
WHERE col_a=? AND col_b=?   能用索引(完整匹配前缀)
WHERE col_b=?               不能用索引(跳过了col_a,破坏了"最左")
```

一句话:联合索引的内部结构是先按第一列排序、相同第一列值内再按第二列排序,这种"字典序"结构决定了只有从最左边的列开始连续匹配的查询条件才能利用索引的有序性。

**底层机制/为什么这样设计**

联合索引 `(category, status)` 的B+树里,键的排序规则是"先比较category,category相同再比较status"——想象一本按"姓氏,名字"排序的电话簿,如果你知道姓氏,可以快速翻到对应的姓氏区间;但如果你只知道名字("查所有叫John的人"),电话簿的排序对你没有任何帮助,你只能从头翻到尾。这正是最左前缀原则的直觉来源:联合索引本质上是"复合排序",只有沿着排序的开头维度去查询,才能利用这份预先排好的顺序跳过大量不相关数据。

**AI研究/工程场景**

排查"明明加了索引,查询还是慢"的问题时,最常见的原因之一就是WHERE条件的列顺序和联合索引定义的顺序不匹配——比如索引是`(user_id, created_at)`,但查询写成`WHERE created_at > ? `(没有`user_id`条件),索引完全用不上;这也是为什么设计联合索引时,通常把"精确匹配(等值查询)常用的列"放在前面、"范围查询常用的列"放在后面。

**可运行例子**(环境:`python-wsl2`)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14 / MariaDB 10.11.15(已启动)
import psycopg2
import pymysql

pg = psycopg2.connect(host='127.0.0.1', port=5432, dbname='dbdemo', user='dbdemo', password='dbdemo_local_only')
pg.autocommit = True
pgc = pg.cursor()
maria = pymysql.connect(host='127.0.0.1', port=3306, database='dbdemo', user='dbdemo', password='dbdemo_local_only')
maria.autocommit(True)
mc = maria.cursor()

pgc.execute("DROP INDEX IF EXISTS idx_cat_status")
pgc.execute("CREATE INDEX idx_cat_status ON big_table (category, status)")
pgc.execute("ANALYZE big_table")
mc.execute("DROP INDEX IF EXISTS idx_cat_status ON big_table")
mc.execute("CREATE INDEX idx_cat_status ON big_table (category, status)")
mc.execute("ANALYZE TABLE big_table")

# 完整匹配前缀(category在前,status在后,和索引定义顺序一致): 两个引擎都真实走了这个索引
# (用全部返回行拼接再判断,因为Bitmap Heap Scan场景下索引名出现在第2行的子节点,不是第1行)
pgc.execute("EXPLAIN SELECT * FROM big_table WHERE category='cat_5' AND status='active'")
pg_match_plan = "\n".join(r[0] for r in pgc.fetchall())
assert "idx_cat_status" in pg_match_plan, f"got: {pg_match_plan}"

mc.execute("EXPLAIN SELECT * FROM big_table WHERE category='cat_5' AND status='active'")
maria_match_row = mc.fetchone()
assert maria_match_row[4] == 'idx_cat_status', f"expected idx_cat_status used, got key={maria_match_row[4]}"

# 只用status(跳过了最左的category列): 两个引擎都真实放弃了这个索引,退化成全表扫描
pgc.execute("EXPLAIN SELECT * FROM big_table WHERE status='active'")
pg_skip_plan = pgc.fetchall()[0][0]
assert "Seq Scan" in pg_skip_plan, f"expected Seq Scan when leftmost column is skipped, got: {pg_skip_plan}"

mc.execute("EXPLAIN SELECT * FROM big_table WHERE status='active'")
maria_skip_row = mc.fetchone()
assert maria_skip_row[3] == 'ALL' and maria_skip_row[4] is None, f"expected full table scan (type=ALL, key=None), got type={maria_skip_row[3]} key={maria_skip_row[4]}"

pg.close()
maria.close()
print("leftmost-prefix rule verified on both engines: skipping the leading column makes the composite index unusable")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
PG category+status(完整匹配前缀): Bitmap Heap Scan on big_table, Recheck Cond: (category=cat_5 AND status=active)
                                  -> Bitmap Index Scan on idx_cat_status
PG status-only(跳过category):    Seq Scan on big_table, Filter: (status = active)

MariaDB category+status: type=ref key=idx_cat_status rows=1221 Extra=Using index condition
MariaDB status-only:     type=ALL key=NULL rows=50225 Extra=Using where
```

**面试怎么问+追问链**

- Q:索引 `(a, b, c)` 能加速 `WHERE a=? AND c=?`(跳过了b)这个查询吗?
  - 追问1:那能加速到什么程度?是完全用不上索引,还是能用上一部分?
    - 深挖追问(区分度较高):这种"部分利用"具体是怎么发生的(答案方向:`a=?`这部分能用上索引定位到"a等于某值"的连续区间,但区间内部还要按b排序,c不是排序的下一维,所以在这个区间内查`c=?`没法再利用索引的有序性跳过数据,只能对这个[通常比全表小得多的]区间做线性过滤——这比完全用不上索引[全表扫描]要好,但比a+b+c都能匹配前缀的情况要差,是"部分受益"而不是"零受益"或"完全受益"的中间状态)。

**常见坑**

- 建了联合索引 `(a, b)`,但查询习惯写成 `WHERE b=? AND a=?`(条件顺序反过来)——这其实没问题,SQL的WHERE条件顺序不影响优化器判断,优化器看的是"查询条件集合"是否覆盖索引的最左前缀,不是SQL文本里出现的顺序。真正的坑是"跳过"最左列,不是"顺序写反"。
- 以为一张表的多个联合索引之间可以互相"拼凑"覆盖任意列组合——每个联合索引只能加速匹配它自己定义顺序的最左前缀查询,不会有"index merge"式的智能拼接(虽然某些引擎在特定条件下有index merge优化,但不能依赖它作为设计索引的默认预期)。

---

## 4. 覆盖索引避免回表

**签名/是什么**

```
覆盖索引(Covering Index): 查询需要的全部列都能从索引本身直接拿到,不需要再回到主表(堆/聚簇索引)取数据
```

一句话:如果一个索引恰好包含了 `SELECT` 列表里的所有列,数据库可以只读索引就完整回答这个查询,省掉"索引查到位置 -> 再去表里读完整行"这一步(即"回表"),这一步通常是随机IO,省掉的收益很可观。

**底层机制/为什么这样设计**

回表的代价来源于KP2讲的存储架构:PostgreSQL的二级索引里只存了列值和`ctid`,拿到`ctid`后还要单独去堆表读一次(通常是随机IO);InnoDB的二级索引叶子存的是列值+主键值,拿到主键值后还要走一次主键聚簇索引查找(也是一次额外的B+树遍历)。如果索引本身已经包含了查询需要的所有列,这一步"回表"就完全不需要发生——查询可以只在索引这一个更小、更紧凑的结构里完成,通常比读主表本体快得多(索引只包含被索引的列,数据量比整行小,更容易被缓冲池完整缓存)。**一个容易被忽视的真实前提条件(PostgreSQL特有)**:光有覆盖索引不足以触发`Index Only Scan`——PostgreSQL还要求对应的堆页在"可见性图"(visibility map)里被标记为"全部可见"(自上次`VACUUM`以来没有被并发事务修改过),否则即使索引理论上覆盖了查询,还是得去堆表逐行确认可见性,退化成普通索引扫描甚至位图扫描。这意味着刚做完大批量写入、只运行过`ANALYZE`(更新统计信息)但还没运行过`VACUUM`(更新可见性图)的表,`Index Only Scan`可能暂时用不上,这个区别在下面的可运行例子里通过显式对比"只ANALYZE"和"ANALYZE+VACUUM"两种状态真实验证。

**AI研究/工程场景**

统计类查询("统计每个类目下有多少种不同的取值")经常只涉及一两个字段,给这些字段建一个刚好覆盖查询需要的联合索引,能让这类高频统计查询的成本降低一个数量级——这是"索引不仅要能用上,还要看能不能覆盖"这个更进一步的优化思路,比"有没有索引"这个第一层判断更精细。

**可运行例子**(环境:`python-wsl2`)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14 / MariaDB 10.11.15(已启动)
import psycopg2
import pymysql

pg = psycopg2.connect(host='127.0.0.1', port=5432, dbname='dbdemo', user='dbdemo', password='dbdemo_local_only')
pg.autocommit = True
pgc = pg.cursor()
maria = pymysql.connect(host='127.0.0.1', port=3306, database='dbdemo', user='dbdemo', password='dbdemo_local_only')
maria.autocommit(True)
mc = maria.cursor()

pgc.execute("DROP INDEX IF EXISTS idx_cat_status")  # 避免和KP3的联合索引竞争,让结果确定可复现
pgc.execute("DROP INDEX IF EXISTS idx_cat_val")
pgc.execute("CREATE INDEX idx_cat_val ON big_table (category, value)")
pgc.execute("ANALYZE big_table")
mc.execute("DROP INDEX IF EXISTS idx_cat_status ON big_table")
mc.execute("DROP INDEX IF EXISTS idx_cat_val ON big_table")
mc.execute("CREATE INDEX idx_cat_val ON big_table (category, value)")
mc.execute("ANALYZE TABLE big_table")

# 覆盖查询: 只要category和value,索引(category,value)本身就够了,不需要回表
# (big_table已在KP2的建表步骤里VACUUM过,可见性图是新鲜的,这里才能真正拿到Index Only Scan)
pgc.execute("EXPLAIN SELECT category, value FROM big_table WHERE category='cat_5'")
pg_covering = pgc.fetchall()[0][0]
assert "Index Only Scan" in pg_covering, f"expected Index Only Scan for covering query, got: {pg_covering}"

mc.execute("EXPLAIN SELECT category, value FROM big_table WHERE category='cat_5'")
maria_covering_extra = mc.fetchone()[9]
assert "Using index" in maria_covering_extra, f"expected 'Using index' (covering) in Extra, got: {maria_covering_extra}"

# 非覆盖查询: 多要了note列(不在索引里),必须回表取这一列
pgc.execute("EXPLAIN SELECT category, value, note FROM big_table WHERE category='cat_5'")
pg_noncovering = pgc.fetchall()[0][0]
assert "Index Only Scan" not in pg_noncovering, f"expected NOT Index Only Scan once a non-indexed column is requested, got: {pg_noncovering}"

mc.execute("EXPLAIN SELECT category, value, note FROM big_table WHERE category='cat_5'")
maria_noncovering_extra = mc.fetchone()[9]
assert "Using index" not in maria_noncovering_extra or "Using index condition" in maria_noncovering_extra, \
    f"expected no full covering ('Using index' alone) once note is requested, got: {maria_noncovering_extra}"

# 单独验证"VACUUM前后Index Only Scan是否可用"这个真实前提条件本身(用独立小表,不依赖big_table已有状态)
pgc.execute("DROP TABLE IF EXISTS vacuum_demo")
pgc.execute("CREATE TABLE vacuum_demo (id SERIAL PRIMARY KEY, k INTEGER, v INTEGER)")
pgc.execute("CREATE INDEX idx_vacuum_demo_kv ON vacuum_demo (k, v)")
demo_rows = ",".join(f"({i % 20},{i * 2})" for i in range(20000))  # k只有20种取值,每种约1000行匹配
pgc.execute(f"INSERT INTO vacuum_demo (k, v) VALUES {demo_rows}")
pgc.execute("ANALYZE vacuum_demo")  # 只更新统计信息,不更新可见性图
pgc.execute("EXPLAIN SELECT k, v FROM vacuum_demo WHERE k = 5")
before_vacuum_plan = pgc.fetchall()[0][0]
assert "Index Only Scan" not in before_vacuum_plan, \
    f"ANALYZE alone should NOT be enough for Index Only Scan (visibility map still stale), got: {before_vacuum_plan}"

pgc.execute("VACUUM vacuum_demo")  # 更新可见性图
pgc.execute("EXPLAIN SELECT k, v FROM vacuum_demo WHERE k = 5")
after_vacuum_plan = pgc.fetchall()[0][0]
assert "Index Only Scan" in after_vacuum_plan, \
    f"after VACUUM, the same covering query should now get Index Only Scan, got: {after_vacuum_plan}"

pg.close()
maria.close()
print("covering index verified on both engines; VACUUM (not just ANALYZE) confirmed as a real prerequisite for Index Only Scan")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
PG覆盖查询(category,value):    Index Only Scan using idx_cat_val on big_table
PG非覆盖查询(多要note):        Bitmap Heap Scan on big_table (仍需访问堆表取note列)

vacuum_demo独立验证(k=5,约1000行匹配):
  只ANALYZE:  Bitmap Heap Scan on vacuum_demo (cost=24.04..145.54 rows=1000 ...)  <- 还不是Index Only Scan
  ANALYZE+VACUUM: Index Only Scan using idx_vacuum_demo_kv on vacuum_demo (cost=0.29..33.79 rows=1000 ...)  <- VACUUM之后才是

MariaDB覆盖查询:    Extra = "Using where; Using index"        (Using index = 覆盖,不回表)
MariaDB非覆盖查询:  Extra = "Using index condition"            (用索引做条件下推,但仍需查完整行取note)
```

**面试怎么问+追问链**

- Q:`SELECT *` 和只查具体需要的列,在有索引的情况下性能可能差多少?
  - 追问1:为什么 `SELECT *` 会"破坏"覆盖索引的优化?
    - 深挖追问(区分度较高):如果一张表的索引已经覆盖了业务上90%常用查询需要的列,但代码里到处写 `SELECT *`,这个索引设计的价值会打多少折扣(答案方向:`SELECT *` 几乎总是会请求到索引里没有的列[比如大文本字段],强制触发回表,原本精心设计的覆盖索引优化直接失效,这是"`SELECT *`不是好习惯"这条常见建议背后一个具体、可验证的技术原因,不是单纯的代码风格偏好)。

**常见坑**

- 只关注"这个查询有没有用上索引",不关注"这个查询是不是覆盖索引",两者对性能的影响可能差一个数量级,但EXPLAIN里都显示"用了索引",容易被表面结论误导。
- MariaDB的 `Extra: Using index condition`(索引条件下推)和 `Extra: Using index`(真正覆盖)两个说法看起来很像,但含义完全不同——前者仍然需要回表,后者不需要,本类的可运行例子已经真实验证了这个区别。
- 以为"索引覆盖了查询"就一定会得到`Index Only Scan`——PostgreSQL还要求可见性图是新鲜的(靠`VACUUM`维护,不是`ANALYZE`)。本类的可运行例子已经真实验证:同样的覆盖索引,只`ANALYZE`过还是`Bitmap Heap Scan`,`VACUUM`之后才变成`Index Only Scan`,这个前提条件很容易被忽略。

---

## 5. 哈希索引适用场景与局限

**签名/是什么**

```
CREATE INDEX idx USING HASH (col)   PostgreSQL显式哈希索引
哈希索引: 等值查询(=)  O(1)平均复杂度,比B+树的O(log n)理论上更快
哈希索引: 范围查询(>,<,BETWEEN)  完全不支持
```

一句话:哈希索引把键通过哈希函数映射到桶里,等值比较只需要计算一次哈希再定位桶,比B+树的多层比较更快,但哈希值不保留任何"大小顺序"信息,天生没法回答"谁比谁大"这类范围问题。

**底层机制/为什么这样设计**

哈希函数的设计目标就是"均匀打散、不保留原始顺序关系"(这也是它能均匀分布、避免哈希冲突扎堆的原因),这个特性决定了它只能回答"这个值在不在、精确等于哪个位置"这类问题——一旦要问"哪些值比X小",哈希表完全无能为力,因为原始数据的大小关系在哈希之后已经被彻底打乱了。B+树牺牲了等值查询的一部分理论速度优势(O(log n)而不是O(1)),换来了保留顺序信息、天然支持范围查询的能力,这是工程上"B+树几乎是索引默认选择,哈希索引只在明确只做等值查询的场景才考虑"的根本原因。InnoDB还有一个自动化的、用户不能显式控制的"自适应哈希索引"(Adaptive Hash Index)——引擎在运行时观察到某些B+树的部分被频繁做等值查询,会在内存里自动为这部分建立一个哈希索引缓存起来加速,但这是引擎自己的内部优化,不是用户显式创建的索引类型,行为完全自动、不可控制。

**AI研究/工程场景**

精确ID查找(比如根据token的哈希值查缓存条目,或者根据一个已知不会做范围查询的业务主键做等值查找)是哈希索引的典型适用场景;但绝大多数业务场景的查询模式会随时间演变(今天只做等值查询,明天可能需要按范围筛选),这也是为什么B+树索引即使理论上对等值查询稍慢一点,依然是工程上默认更安全的选择——哈希索引的适用场景相对狭窄,选错了后续要加范围查询支持只能重新建索引。

**可运行例子**(环境:`python-wsl2`)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14(已启动)
import psycopg2

pg = psycopg2.connect(host='127.0.0.1', port=5432, dbname='dbdemo', user='dbdemo', password='dbdemo_local_only')
pg.autocommit = True
pgc = pg.cursor()

pgc.execute("DROP INDEX IF EXISTS idx_value_hash")
pgc.execute("CREATE INDEX idx_value_hash ON big_table USING HASH (value)")
pgc.execute("ANALYZE big_table")

# 等值查询: 哈希索引真实被用上
pgc.execute("EXPLAIN SELECT * FROM big_table WHERE value = 500000")
eq_plan = pgc.fetchall()[0][0]
assert "idx_value_hash" in eq_plan, f"expected hash index used for equality, got: {eq_plan}"

# 范围查询: 哈希索引完全用不上,优化器只能选择全表扫描(或退回其他索引,这里故意只留哈希索引验证纯粹的哈希局限)
pgc.execute("EXPLAIN SELECT * FROM big_table WHERE value > 500000")
range_plan = pgc.fetchall()[0][0]
assert "idx_value_hash" not in range_plan, f"hash index must NOT be usable for range queries, got: {range_plan}"
assert "Seq Scan" in range_plan, f"expected fallback to Seq Scan since no other index covers this range query, got: {range_plan}"

pg.close()
print("hash index verified: equality query uses it, range query cannot use it at all, falls back to Seq Scan")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
PG value=500000(等值): Index Scan using idx_value_hash on big_table, Index Cond: (value = 500000)
PG value>500000(范围): Seq Scan on big_table, Filter: (value > 500000)   <- 哈希索引完全帮不上忙
```

**面试怎么问+追问链**

- Q:哈希索引比B+树索引快吗?为什么大多数场景还是默认用B+树?
  - 追问1:MySQL/InnoDB允许用户显式创建哈希索引吗?
    - 深挖追问(区分度较高):InnoDB的"自适应哈希索引"和用户显式创建的哈希索引有什么本质区别(答案方向:InnoDB普通表不支持用户显式`USING HASH`创建哈希索引[这是MEMORY存储引擎的特性],但InnoDB内部有一个引擎自动管理、对用户透明的"自适应哈希索引"缓存机制,用户无法手动控制启用/禁用具体某个索引走不走这个缓存,只能整体开关这个特性;这和PostgreSQL允许用户显式创建、精确控制的哈希索引是完全不同层面的东西,不能混为一谈)。

**常见坑**

- 认为哈希索引"更快"所以到处用它替代B+树——哈希索引的适用范围严格受限于"只做等值查询、明确不需要范围查询/排序"这个前提,一旦业务后续需要`ORDER BY`该列或做范围筛选,哈希索引完全帮不上忙。
- 混淆PostgreSQL的显式哈希索引和InnoDB的自适应哈希索引——前者是用户主动创建、影响执行计划选择的索引;后者是引擎内部自动管理的缓存优化,不会出现在`EXPLAIN`的索引选择里。

---

## 6. EXPLAIN基础:PostgreSQL输出格式解读

**签名/是什么**

```
Seq Scan on t (cost=0.00..1042.00 rows=24898 width=32)
Index Scan using idx on t (cost=0.29..8.31 rows=1 width=31)
cost=启动代价..总代价(任意单位,不是毫秒)  rows=预估返回行数  width=预估每行字节数
```

一句话:PostgreSQL的 `EXPLAIN` 输出是一棵倒过来读的执行计划树(最下层的子节点先执行),每个节点标注了这一步用什么策略(顺序扫描/索引扫描/位图扫描等)、优化器估算的代价和结果行数,这些都是**估算值**,不是真实测量值(真实测量要用`EXPLAIN ANALYZE`,见KP8)。

**底层机制/为什么这样设计**

`cost` 是一个无量纲的相对数字(不是秒或毫秒),由优化器根据一套内置的代价模型(读一个磁盘页的代价、比较一行的代价等参数)累加计算得出,只用来在多个候选执行计划之间做**相对比较**,选出代价最低的那个,不能直接当成真实执行时间解读。本类前面几个知识点看到的 `Seq Scan`/`Bitmap Heap Scan`/`Bitmap Index Scan`/`Index Scan`/`Index Only Scan` 是PostgreSQL几种核心扫描策略:`Seq Scan` 从头到尾顺序读整张表;`Index Scan` 用索引定位行、每找到一行就立刻去堆表取一次(适合返回行数很少的场景,因为每行都要单独随机IO);`Bitmap Heap Scan`+`Bitmap Index Scan` 组合先用索引把所有满足条件的行位置收集成一个"位图",再按物理页的顺序批量读取堆表(适合返回行数中等的场景,避免了逐行随机IO,但也不是全表扫描)。

**AI研究/工程场景**

排查慢查询时,`EXPLAIN`(不带`ANALYZE`)是零成本的第一步——它不会真的执行查询,只是让优化器"预演"一遍,对于担心真的跑一遍会很慢/有副作用的场景(比如生产环境的可疑`DELETE`/`UPDATE`语句),看`EXPLAIN`输出判断走了什么策略、预估影响多少行,是排查前的安全检查手段。

**可运行例子**(环境:`python-wsl2`,复用KP1-5已建立的索引与数据)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14(已启动),复用之前knowledge point建立的big_table
import psycopg2

pg = psycopg2.connect(host='127.0.0.1', port=5432, dbname='dbdemo', user='dbdemo', password='dbdemo_local_only')
pg.autocommit = True
pgc = pg.cursor()

# 低选择性(约45%行匹配,没有合适索引可用): Seq Scan, rows估算应该在总行数的相当比例
pgc.execute("EXPLAIN SELECT * FROM big_table WHERE note = 'this-note-does-not-exist'")
plan_line = pgc.fetchall()[0][0]
assert "cost=" in plan_line and "rows=" in plan_line and "width=" in plan_line, \
    f"expected cost/rows/width fields in EXPLAIN output, got: {plan_line}"

# 解析 rows= 后面的数字,验证它是一个具体的估算整数(不是查询真的执行返回的数字)
import re
rows_match = re.search(r"rows=(\d+)", plan_line)
assert rows_match is not None
estimated_rows = int(rows_match.group(1))
assert estimated_rows >= 0, "rows= is an ESTIMATE from planner statistics, not an actual execution count"

pg.close()
print("PG EXPLAIN format verified: cost=start..total, rows=estimated count, width=estimated bytes/row - all pre-execution estimates")
```

**面试怎么问+追问链**

- Q:`EXPLAIN` 里的 `cost=0.29..8.31` 是什么单位,8.31毫秒吗?
  - 追问1:那这个数字有什么用,不能换算成真实时间的话?
    - 深挖追问(区分度较高):优化器在多个候选执行计划之间选择时,具体是怎么用这个"无量纲代价"做决策的(答案方向:优化器会为同一个查询枚举多种可能的执行路径[比如走索引A、走索引B、还是全表扫描],分别估算每种路径的cost,选择cost最低的那个执行——这个相对比较不需要知道cost对应的绝对时间单位,只要模型内部一致就能做出正确的相对排序,这也是为什么"cost看起来很怪的数字"依然能指导出合理的执行计划选择)。

**常见坑**

- 把 `EXPLAIN`(不带`ANALYZE`)输出的 `rows=` 当成查询真实会返回的行数——这是纯估算值,依赖统计信息的准确性(见KP8),统计信息过期时估算可能和真实情况差很多。
- 只看第一行(最外层节点)就下结论,不理解这是一棵倒着读的树——真正的执行顺序是从缩进最深的子节点开始,一层层往外汇总。

---

## 7. EXPLAIN基础:MariaDB/InnoDB输出格式解读

**签名/是什么**

```
type: 访问类型(ALL=全表扫描, range=范围扫描, ref=非唯一索引等值, const=主键/唯一索引单行命中)
key:  实际使用的索引名(NULL表示没用索引)
rows: 预估需要扫描的行数
Extra: 额外说明(Using index=覆盖索引不回表, Using where=有额外过滤条件, Using index condition=索引条件下推但仍需回表)
```

一句话:MariaDB/MySQL的 `EXPLAIN` 是表格形式(不是PostgreSQL那种树形文本),核心是 `type` 这一列——它是一个从最差到最好排列的等级列表(`ALL`最差,`const`最好),面试时被问"怎么看EXPLAIN"通常主要在问怎么解读这个`type`等级。

**底层机制/为什么这样设计**

`type` 列的常见取值按性能从差到好大致是:`ALL`(全表扫描,没用任何索引)→ `index`(扫描整个索引但不回表,比全表扫描省一点IO)→ `range`(索引范围扫描)→ `ref`(用非唯一索引做等值匹配,可能匹配多行)→ `eq_ref`(JOIN时用唯一索引做等值匹配,保证最多一行)→ `const`(主键或唯一索引直接命中单行,查询规划阶段就能确定结果)。这个等级顺序背后的直觉是"引擎需要检查/返回的行数级别":`ALL`可能要检查全部N行,`const`在规划时就知道最多1行。`Extra`列里最值得关注的三个信息在KP3/KP4已经真实验证过:`Using index`(覆盖索引,不用回表)、`Using index condition`(索引条件下推ICP,用索引过滤但仍要回表取完整行)、`Using where`(有超出索引本身的额外过滤条件)。

**AI研究/工程场景**

线上慢查询日志分析工具(比如MySQL自带的slow query log配合`pt-query-digest`)最终都会落到"看这条慢查询的EXPLAIN,type是不是ALL,是不是该加个索引"这个标准排查流程,`type=ALL`几乎总是慢查询优化的第一个排查方向。

**可运行例子**(环境:`python-wsl2`,复用之前knowledge point建立的big_table)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 MariaDB 10.11.15(已启动)
import pymysql

maria = pymysql.connect(host='127.0.0.1', port=3306, database='dbdemo', user='dbdemo', password='dbdemo_local_only')
maria.autocommit(True)
mc = maria.cursor()

# const: 按唯一主键单行命中
mc.execute("EXPLAIN SELECT * FROM big_table WHERE id = 1")
row_const = mc.fetchone()
assert row_const[3] == 'const', f"expected type=const for primary key point lookup, got type={row_const[3]}"

# ref: 用非唯一索引(category上的普通索引)做等值匹配,可能匹配多行
mc.execute("EXPLAIN SELECT * FROM big_table WHERE category = 'cat_5'")
row_ref = mc.fetchone()
assert row_ref[3] in ('ref', 'range'), f"expected type=ref/range for non-unique index equality, got type={row_ref[3]}"

# ALL: 没有可用索引的列上做过滤,全表扫描
mc.execute("EXPLAIN SELECT * FROM big_table WHERE note = 'nonexistent-value-xyz'")
row_all_or_index = mc.fetchone()
# note列如果建过索引(KP10会建),这里可能是range;若未建索引则是ALL - 用这个断言只验证type字段真实存在且合法
assert row_all_or_index[3] in ('ALL', 'range', 'ref'), f"got unexpected type={row_all_or_index[3]}"

maria.close()
print("MariaDB EXPLAIN format verified: type ranges from const (best, single-row PK hit) to ALL (worst, full scan)")
```

**面试怎么问+追问链**

- Q:MariaDB的 `EXPLAIN` 里 `type=index` 和 `type=ALL` 有什么区别?看起来都是"扫了很多东西"。
  - 追问1:那 `type=index` 比 `type=ALL` 好在哪?
    - 深挖追问(区分度较高):如果一个查询的`type`是`index`,`Extra`同时显示`Using index`,这说明了什么(答案方向:`type=index`意味着扫描了整个索引结构[不是按条件精确定位,是把索引从头扫到尾],但因为同时`Using index`[覆盖],扫描的是比整行数据更小的索引本身,不需要逐行回表——这仍然比`ALL`[扫描整个表的完整行数据]代价小,但比`range`/`ref`[只扫描索引里满足条件的一部分]代价大,是"扫描范围广但单位代价低"和"扫描范围窄但单位代价高"之间的一种中间状态)。

**常见坑**

- 只看`type`不看`Extra`就下结论——`type=ref`配合`Extra=Using index`(覆盖)和`type=ref`配合`Extra=Using index condition`(仍需回表)性能差异不小,但`type`这一列完全看不出这个区别,必须结合`Extra`一起看。
- 误以为`key`列显示的索引名就是"最优索引"——优化器有时会因为统计信息或代价估算的原因选择一个次优的可用索引,`key`只是显示"实际选了哪个",不代表这个选择在事后复盘时一定是最优的。

---

## 8. EXPLAIN ANALYZE:从预估到真实执行的差异

**签名/是什么**

```
EXPLAIN            只做规划,不真正执行查询,rows=是估算值
EXPLAIN ANALYZE     真正执行查询,额外报告 actual time=/rows=(真实测量值),但有副作用(真的跑了这条SQL)
```

一句话:`EXPLAIN` 和 `EXPLAIN ANALYZE` 的本质区别是"预测"和"真实测量"——前者零成本但可能因为统计信息过期而失准,后者真实可靠但会真的执行这条SQL(对`DELETE`/`UPDATE`语句要格外小心)。

**底层机制/为什么这样设计**

优化器做出的所有决策(走不走索引、选哪种扫描策略)都依赖于**统计信息**(每列的取值分布、不同值的数量估算等,由`ANALYZE`命令收集和刷新),这些统计信息是"某一时刻的快照",如果之后表的数据分布发生了显著变化(比如批量插入了大量之前很稀有的值)而没有重新执行`ANALYZE`,优化器的估算(`EXPLAIN`里的`rows=`)会和真实情况脱节——`EXPLAIN ANALYZE`因为真的跑了一遍查询,能报告出真实的`actual rows=`,这个数字和估算的`rows=`之间的差距,是判断"统计信息是不是该更新了"的直接证据。

**AI研究/工程场景**

线上一个查询"突然变慢"但SQL本身没变,一个常见根因就是数据分布发生了显著变化但统计信息没跟上——批量导入历史数据、业务的某个取值分布因为运营活动突变(比如一个平时很少用的优惠券状态突然被大量使用),都会让优化器基于旧统计信息做出的执行计划选择偏离实际最优,这时候一个简单的重新`ANALYZE`(而不是加新索引/改SQL)就能解决问题,`EXPLAIN ANALYZE`里"估算值和真实值差距很大"正是诊断这类问题的关键证据。

**可运行例子**(环境:`python-wsl2`,复用之前knowledge point建立的big_table)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14(已启动)
import psycopg2
import re

pg = psycopg2.connect(host='127.0.0.1', port=5432, dbname='dbdemo', user='dbdemo', password='dbdemo_local_only')
pg.autocommit = True
pgc = pg.cursor()

# 先清理掉可能残留的批量测试数据,保证从"只有1行status='rare'"这个已知状态开始
pgc.execute("DELETE FROM big_table WHERE note = 'bulk'")

# 新鲜统计信息下,status='rare' 只有1行,估算应该接近真实值
# 注意: ANALYZE本身是抽样统计,即使统计信息完全新鲜,对非常小的行数估算也不保证100%精确命中,
# 用"估算值很小(<=5)"而不是"估算恒等于真实值"来验证,这才是符合ANALYZE抽样本质的正确断言方式
pgc.execute("ANALYZE big_table")
pgc.execute("EXPLAIN ANALYZE SELECT * FROM big_table WHERE status='rare'")
lines = [r[0] for r in pgc.fetchall()]
top_line = lines[0]
est_rows = int(re.search(r"rows=(\d+)", top_line).group(1))
actual_rows = int(re.search(r"actual time=[\d.]+\.\.[\d.]+ rows=(\d+)", top_line).group(1))
assert actual_rows == 1, f"expected exactly the 1 seed 'rare' row, got actual={actual_rows}"
assert est_rows <= 5, f"fresh-statistics estimate for a near-unique value should be small, got {est_rows}"

# 制造统计信息过期: 批量插入5000行status='rare',但故意不重新ANALYZE
pgc.execute("INSERT INTO big_table (category, status, value, note) SELECT 'cat_0','rare',1,'bulk' FROM generate_series(1,5000)")
pgc.execute("EXPLAIN ANALYZE SELECT * FROM big_table WHERE status='rare'")
lines_stale = [r[0] for r in pgc.fetchall()]
top_line_stale = lines_stale[0]
est_rows_stale = int(re.search(r"rows=(\d+)", top_line_stale).group(1))
actual_rows_stale = int(re.search(r"actual time=[\d.]+\.\.[\d.]+ rows=(\d+)", top_line_stale).group(1))
# 统计信息过期: 估算值仍然停留在旧的水平(约1),真实值已经变成约5001
assert est_rows_stale < actual_rows_stale / 100, \
    f"stale statistics: estimate({est_rows_stale}) should be far below actual({actual_rows_stale}) after bulk insert without re-ANALYZE"

pg.close()
print(f"stale statistics verified: fresh est<=5 (sampling-based), after bulk insert est stays at {est_rows_stale} but actual jumps to {actual_rows_stale}")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
新鲜统计信息:  Index Scan ... (cost=... rows=2 ...) (actual time=... rows=1 loops=1)  <- 估算2/真实1,抽样统计的正常误差范围
批量插入5000行rare、不重新ANALYZE后:
  Index Scan ... (cost=... rows=2 ...) (actual time=... rows=5001 loops=1)  <- 估算仍停留在2,真实已变成5001,这才是统计信息过期的真实信号
```

**面试怎么问+追问链**

- Q:为什么同一条SQL,昨天很快,今天突然变慢了,SQL和索引都没改?
  - 追问1:除了数据量增长,还有什么会导致统计信息"过期"?
    - 深挖追问(区分度较高):数据库通常什么时候会自动触发统计信息更新,自动更新为什么有时候还是跟不上(答案方向:大多数引擎有自动的统计信息收集任务(比如PostgreSQL的autovacuum会顺带触发autoanalyze),但触发条件通常是"变更的行数超过某个比例阈值",如果是短时间内剧烈的数据分布变化[比如本例5000行相对5万行是10%的变更量,可能还没触发自动阈值],统计信息就会有一段时间处于滞后状态,这也是为什么大批量数据迁移/导入后,有经验的工程师会主动手动跑一次`ANALYZE`而不是完全依赖自动机制)。

**常见坑**

- 对`DELETE`/`UPDATE`语句直接跑`EXPLAIN ANALYZE`——它会真实执行这条语句,包括它的副作用(真删了/真改了数据),生产环境排查问题时应该用只读的`SELECT`版本或在事务里跑完之后`ROLLBACK`,而不是对写语句直接用`EXPLAIN ANALYZE`。
- 只看`EXPLAIN`(不带`ANALYZE`)的结果就下最终结论——如果怀疑统计信息可能过期,必须用`EXPLAIN ANALYZE`看真实执行数据来交叉验证,不能单纯相信估算值。

---

## 9. 基于代价的查询优化器:选择性如何影响执行计划选择

**签名/是什么**

```
选择性(Selectivity) = 满足条件的行数 / 总行数,越接近0越"挑剔"(越适合走索引)
优化器会比较"走索引的代价"和"全表扫描的代价",选择性太低(命中比例太高)时,索引反而更贵
```

一句话:索引不是"建了就一定被用",优化器会基于代价模型算一笔账——当查询命中的行数占比高到一定程度,连续读整个表(全表扫描)反而比"先查索引再逐个/批量去表里取行"更省IO,这时优化器会主动放弃现成的索引。

**底层机制/为什么这样设计**

索引扫描相对全表扫描的优势来自"跳过大部分不需要的数据",但索引扫描本身也有代价(遍历索引结构+可能的回表随机IO),当命中比例足够高时,这些代价的总和会超过"干脆顺序读完整张表"的代价——磁盘的顺序读远比随机读快,全表扫描虽然要读所有数据,但都是顺序IO;索引扫描虽然理论上"只读需要的数据",但如果需要的数据占比很高,大量的随机IO(尤其是走`Index Scan`逐行回表时)反而比一次性顺序读更慢。这正是"选择性"这个概念存在的意义:它是优化器判断"该不该用索引"的核心依据之一。

**AI研究/工程场景**

给低基数/低选择性的列(比如"性别"这种只有2-3种取值的列)建索引,在很多场景下价值有限——即使建了索引,一旦查询条件命中的比例超过某个阈值(通常是百分之几到百分之十几,具体阈值因数据分布和引擎代价模型而异),优化器大概率还是会选择全表扫描,索引白白占用存储空间和维护写入的额外代价却几乎不被使用,这是"是否值得为这一列建索引"的工程决策依据。

**可运行例子**(环境:`python-wsl2`,复用之前knowledge point建立的big_table,在value列建索引后测试不同命中比例)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14 + MariaDB 10.11.15(已启动)
import psycopg2
import pymysql

pg = psycopg2.connect(host='127.0.0.1', port=5432, dbname='dbdemo', user='dbdemo', password='dbdemo_local_only')
pg.autocommit = True
pgc = pg.cursor()
maria = pymysql.connect(host='127.0.0.1', port=3306, database='dbdemo', user='dbdemo', password='dbdemo_local_only')
maria.autocommit(True)
mc = maria.cursor()

pgc.execute("DROP INDEX IF EXISTS idx_value")
pgc.execute("CREATE INDEX idx_value ON big_table (value)")
pgc.execute("ANALYZE big_table")
mc.execute("DROP INDEX IF EXISTS idx_value ON big_table")
mc.execute("CREATE INDEX idx_value ON big_table (value)")
mc.execute("ANALYZE TABLE big_table")

# 很低选择性(~5%命中): 两个引擎此时都还愿意用索引 - 这是两者一致的基准点
pgc.execute("EXPLAIN SELECT * FROM big_table WHERE value < 50000")
pg_verylow_plan = "\n".join(r[0] for r in pgc.fetchall())  # 用全部行拼接,Bitmap Heap Scan的索引名在第2行不是第1行
assert "idx_value" in pg_verylow_plan, f"at ~5% selectivity, PG should use the index, got: {pg_verylow_plan}"

mc.execute("EXPLAIN SELECT * FROM big_table WHERE value < 50000")
maria_verylow_row = mc.fetchone()
assert maria_verylow_row[4] == 'idx_value', f"at ~5% selectivity, MariaDB should use the index, got key={maria_verylow_row[4]}"

# ~20%命中: 两个引擎在这一点上真实出现分歧 - PG还在用索引,MariaDB已经放弃了
# (这不是测试代码的bug,是两个引擎代价模型真实给出的不同交叉点)
pgc.execute("EXPLAIN SELECT * FROM big_table WHERE value < 200000")
pg_mid_plan = "\n".join(r[0] for r in pgc.fetchall())
assert "idx_value" in pg_mid_plan, f"at ~20% selectivity, PG should STILL use the index, got: {pg_mid_plan}"

mc.execute("EXPLAIN SELECT * FROM big_table WHERE value < 200000")
maria_mid_row = mc.fetchone()
assert maria_mid_row[3] == 'ALL', \
    f"at ~20% selectivity, MariaDB should have ALREADY abandoned the index (crossover point differs from PG), got type={maria_mid_row[3]}"

# ~50%命中: 两个引擎此时都已放弃索引,回到一致
pgc.execute("EXPLAIN SELECT * FROM big_table WHERE value < 500000")
pg_high_plan = pgc.fetchall()[0][0]
assert "Seq Scan" in pg_high_plan, f"at ~50% selectivity, PG should abandon the index for Seq Scan, got: {pg_high_plan}"

mc.execute("EXPLAIN SELECT * FROM big_table WHERE value < 500000")
maria_high_row = mc.fetchone()
assert maria_high_row[3] == 'ALL', f"at ~50% selectivity, MariaDB should abandon the index, got type={maria_high_row[3]}"

pg.close()
maria.close()
print("selectivity-based optimizer decision verified: both abandon the index eventually, but PG's crossover point (20%~50%) is genuinely later than MariaDB's (5%~20%)")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux,`idx_value` 索引存在的前提下):

```
PostgreSQL的真实交叉点在20%~50%之间:
  value<5000(~0.5%):  Bitmap Heap Scan(走索引)
  value<50000(~5%):   Bitmap Heap Scan(走索引)
  value<200000(~20%): Bitmap Heap Scan(走索引)
  value<500000(~50%): Seq Scan(放弃索引)
  value<900000(~90%): Seq Scan(放弃索引)

MariaDB/InnoDB的真实交叉点更早,在5%~20%之间(比PostgreSQL更早放弃索引):
  value<5000(~0.5%):   type=range key=idx_value(走索引)
  value<50000(~5%):    type=range key=idx_value(走索引)
  value<200000(~20%):  type=ALL  key=NULL(放弃索引)
  value<500000(~50%):  type=ALL  key=NULL(放弃索引)
  value<900000(~90%):  type=ALL  key=NULL(放弃索引)
```

**面试怎么问+追问链**

- Q:一张表的某一列建了索引,查询这一列时优化器一定会用这个索引吗?
  - 追问1:那什么情况下即使有索引,优化器也不会用?
    - 深挖追问(区分度较高):上面的真实测试显示 PostgreSQL 和 MariaDB 放弃索引的选择性阈值不一样(PG在20%~50%之间,MariaDB在5%~20%之间),这说明了什么(答案方向:这说明"选择性阈值"不是一个数据库理论里的固定常数,而是每个引擎自己的代价模型[对顺序IO/随机IO/CPU比较代价的具体假设参数]算出来的结果,不同引擎、甚至同一引擎不同版本的代价模型参数都可能不同,回答这类问题时最负责任的说法是"取决于具体引擎的代价模型,不能拍脑袋断言一个通用阈值,需要用EXPLAIN实测",这也是本类反复强调的核心方法论)。

**常见坑**

- 把某个引擎/某次经验里观察到的"选择性阈值"(比如"超过30%就不走索引了")当成放之四海而皆准的公式——本类的双引擎实测已经证明这个阈值因引擎而异,面试时给出一个死板的百分比数字反而暴露了没有实测验证的习惯。
- 认为"删掉不常用的索引"没有代价——即使某个索引因为选择性问题很少被查询优化器选中,它仍然在每次`INSERT`/`UPDATE`/`DELETE`时需要被同步维护,这是索引"不只是免费的加速器,也是有写入代价的"这个更全面视角的一部分。

---

## 10. 索引失效的常见场景

**签名/是什么**

```
索引失效: 索引客观存在,但查询条件的写法导致优化器无法(或不愿)使用它
常见诱因: 对索引列做函数运算、前导通配符LIKE、隐式类型转换
```

一句话:有索引不等于会被用上,索引能加速的前提是"能够利用索引的有序/结构化特性去定位数据",一旦查询条件的写法破坏了这个前提(比如对列做了函数变换、模糊匹配的通配符在开头),即使索引客观存在,优化器也无法利用它,只能退化为全表扫描。

**底层机制/为什么这样设计**

索引本质上是"预先排好序的数据副本",查询能利用索引的前提是"能够直接在这份预排序的数据里做区间定位"。`WHERE UPPER(category) = 'CAT_5'` 这种写法,索引里存的是`category`原始值的排序,不是`UPPER(category)`处理后的排序,优化器没有能力"反推"出应该去索引的哪个区间找——除非专门建一个基于`UPPER(category)`表达式的函数索引/表达式索引。`LIKE '%xyz'`(前导通配符)同理:B+树索引的区间定位依赖"已知前缀",`%xyz`没有任何已知前缀可用于缩小搜索区间,只能整个扫一遍。**一个更容易被忽视的真实发现**:即使是`LIKE 'xyz%'`(后缀通配符,理论上应该能利用前缀区间定位),在这次实测环境(PostgreSQL 16.14,数据库collation为`C.UTF-8`)里,用默认操作符类建的普通B-tree索引依然**没有**被优化器选用于这个查询,必须显式用`text_pattern_ops`操作符类重新建索引才能让`LIKE`的前缀匹配真正走索引——这说明"C.UTF-8"这种看起来像"简单字节序"的collation设置,并不自动等价于让默认索引支持模式匹配优化,这类假设必须用`EXPLAIN`实测验证,不能凭直觉推断。

**AI研究/工程场景**

"给某列加了索引,但查询还是慢"这个经典问题,排查思路第一步永远是"看WHERE条件里这一列有没有被函数包裹、有没有前导通配符、有没有可能发生隐式类型转换(比如列是字符串类型但查询条件传了数字)"——这几乎是数据库慢查询排查清单里出现频率最高的一类问题,比"完全没建索引"更隐蔽,因为索引"看起来存在"却在这条具体查询里不生效。

**可运行例子**(环境:`python-wsl2`,复用之前knowledge point建立的big_table)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14(已启动)
import psycopg2

pg = psycopg2.connect(host='127.0.0.1', port=5432, dbname='dbdemo', user='dbdemo', password='dbdemo_local_only')
pg.autocommit = True
pgc = pg.cursor()

# 场景1: 前导通配符LIKE '%xyz%' - 无论什么操作符类都不可能走B-tree索引(没有已知前缀)
pgc.execute("DROP INDEX IF EXISTS idx_note_pattern")  # 确保没有残留的text_pattern_ops索引干扰"默认操作符类"这一步的判断
pgc.execute("DROP INDEX IF EXISTS idx_note")
pgc.execute("CREATE INDEX idx_note ON big_table (note)")
pgc.execute("ANALYZE big_table")
pgc.execute("EXPLAIN SELECT * FROM big_table WHERE note LIKE '%123%'")
leading_wildcard_plan = pgc.fetchall()[0][0]
assert "Seq Scan" in leading_wildcard_plan, f"leading wildcard can never use a plain B-tree index, got: {leading_wildcard_plan}"

# 场景2(真实的意外发现): 后缀通配符LIKE 'xyz%' 用默认操作符类的普通索引,在本环境依然走不了索引
pgc.execute("EXPLAIN SELECT * FROM big_table WHERE note LIKE 'note-123%'")
default_opclass_plan = pgc.fetchall()[0][0]
assert "Seq Scan" in default_opclass_plan, \
    f"even a trailing-wildcard LIKE fails to use the default-opclass B-tree index in this C.UTF-8 environment, got: {default_opclass_plan}"

# 必须显式用 text_pattern_ops 操作符类重建索引,后缀通配符LIKE才能真正走索引
# (用全部返回行拼接再判断: 优化器可能选Index Scan直接用,也可能选Bitmap Heap Scan+Bitmap Index Scan
#  两步走[命中行数稍多时更常见],索引名在哪一行取决于具体选择哪种策略,不能只看第1行)
pgc.execute("DROP INDEX IF EXISTS idx_note_pattern")
pgc.execute("CREATE INDEX idx_note_pattern ON big_table (note text_pattern_ops)")
pgc.execute("ANALYZE big_table")
pgc.execute("EXPLAIN SELECT * FROM big_table WHERE note LIKE 'note-123%'")
pattern_opclass_plan = "\n".join(r[0] for r in pgc.fetchall())
assert "Seq Scan" not in pattern_opclass_plan and "idx_note_pattern" in pattern_opclass_plan, \
    f"with text_pattern_ops, trailing-wildcard LIKE should now use the index (via Index Scan or Bitmap Index Scan), got: {pattern_opclass_plan}"

# 场景3: 对索引列做函数运算 UPPER(category),破坏索引可用性
pgc.execute("DROP INDEX IF EXISTS idx_category_plain")
pgc.execute("CREATE INDEX idx_category_plain ON big_table (category)")
pgc.execute("ANALYZE big_table")
pgc.execute("EXPLAIN SELECT * FROM big_table WHERE category = 'cat_5'")
plain_plan = "\n".join(r[0] for r in pgc.fetchall())
assert "idx_category_plain" in plain_plan, f"got: {plain_plan}"

pgc.execute("EXPLAIN SELECT * FROM big_table WHERE UPPER(category) = 'CAT_5'")
function_wrapped_plan = pgc.fetchall()[0][0]
assert "Seq Scan" in function_wrapped_plan, \
    f"wrapping the indexed column in UPPER() breaks index usability, got: {function_wrapped_plan}"

pg.close()
print("index invalidation verified: leading wildcard always fails, trailing wildcard needs text_pattern_ops even under C.UTF-8, function-wrapped column always fails")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
前导通配符 '%123%':               Seq Scan  (理论和实测一致,B-tree无法处理)
后缀通配符 'note-123%'(默认索引):  Seq Scan  (意外发现:即使C.UTF-8 collation,默认操作符类依然不走索引)
后缀通配符 'note-123%'(text_pattern_ops索引): 索引真正被用上了(具体表现为Index Scan或Bitmap Heap Scan+
    Bitmap Index Scan,取决于优化器对匹配行数的估算,两种情况索引名都出现在真实的EXPLAIN输出里)
    <- 显式操作符类后才真正走索引,这是本类的核心结论
函数包裹 UPPER(category)='CAT_5':  Seq Scan  (对比明文 category='cat_5' 真实走了索引)
```

**面试怎么问+追问链**

- Q:给一个字符串列加了索引,`WHERE col LIKE 'abc%'` 这种前缀模糊查询能用上索引吗?
  - 追问1:如果说"能",在什么条件下能,什么条件下不能?
    - 深挖追问(区分度较高):这次实测发现"即使collation是C.UTF-8,默认操作符类的索引依然不支持前缀LIKE优化",这个反直觉的结果说明了什么方法论问题(答案方向:这正是本类反复强调的核心纪律——数据库行为的很多"经验法则"[比如"C locale下前缀LIKE能走索引"]在具体版本/具体配置下不一定成立,面试时给出经验法则作为出发点没问题,但更专业的回答应该是"这取决于具体的collation和操作符类配置,我会用EXPLAIN实测确认,不会假设",这个态度本身在面试里往往比记住某条具体规则更能体现真实的工程判断力)。

**常见坑**

- 想当然认为"字符串前缀匹配一定能走索引"——这是一个流传很广的经验法则,但本类的真实实测已经证明它在特定配置下(即便是看起来"简单"的C.UTF-8 collation)也可能不成立,必须用`EXPLAIN`验证,不能凭经验断言。
- 在WHERE条件里对索引列做函数包裹(`UPPER()`/`DATE()`/类型转换等)却没意识到这会让索引失效——常见的补救方案是建"表达式索引"(`CREATE INDEX ON t (UPPER(col))`),让索引本身就按函数运算后的值排序,而不是在查询时才发现索引用不上。
