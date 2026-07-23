# 范式与进阶SQL

> 板块 I(关系模型与SQL基础)第 2 篇。环境:全部 `.venv`(stdlib `sqlite3` 3.50.4)。范式相关知识点全部用"反例驱动"讲法——先构造违反范式的表,真实触发更新异常,再展示分解后异常如何被结构性消除,不是背诵范式定义。

## 1. 第一范式(1NF):原子性与消除重复组

**签名/是什么**

```
1NF: 每个属性(列)必须是原子值,不能是列表/集合/嵌套结构
```

一句话:1NF 是"能不能叫关系模型"的门槛——一列里塞一个逗号分隔列表看起来省事,但会让这一列失去被数据库索引/精确匹配/参与JOIN的能力。

**底层机制/为什么这样设计**

关系模型的数学基础是集合论,一个属性的值必须来自一个确定的域(见01类知识点1),"逗号分隔的字符串"这种"一列多值"的存法,本质上是把多个独立的事实硬塞进一个原子值的容器里,数据库引擎没有办法知道这个字符串内部还有结构,只能把它当成一个不透明的大字符串整体处理——查询/索引/约束这些机制全部失效到只能退化为字符串模式匹配。

**AI研究/工程场景**

训练数据的标注字段("这张图片的标签是猫,动物,可爱")、用户画像的兴趣标签,新手很容易图省事直接存成逗号分隔字符串,后续想按单个标签精确检索或做标签维度的统计时,才发现子串匹配既慢又不准确(容易有前缀/子串误判)。

**可运行例子**(环境:`.venv`)

```python
import sqlite3

conn = sqlite3.connect(":memory:")
cur = conn.cursor()

# 违反1NF: items列塞了一个逗号分隔的"重复组"
cur.execute("CREATE TABLE bad_orders (order_id INTEGER, items TEXT)")
cur.execute("INSERT INTO bad_orders VALUES (1, 'apple,banana,cherry')")
cur.execute("INSERT INTO bad_orders VALUES (2, 'pineapple,mango')")
conn.commit()

# 用LIKE子串匹配查"含apple的订单",订单2根本没有"apple"这个item,
# 但"pineapple"这个词里含有子串"apple",被错误地匹配上
cur.execute("SELECT order_id FROM bad_orders WHERE items LIKE '%apple%'")
false_positive_rows = cur.fetchall()
assert (2,) in false_positive_rows, "order 2 (pineapple,mango) should be a false positive for 'apple' substring search"

# 1NF: 拆成两张表,每个属性只存一个原子值(一行一个item)
cur.execute("CREATE TABLE good_order_items (order_id INTEGER, item TEXT)")
cur.executemany("INSERT INTO good_order_items VALUES (?, ?)",
                 [(1,'apple'),(1,'banana'),(1,'cherry'),(2,'pineapple'),(2,'mango')])
conn.commit()
cur.execute("SELECT order_id FROM good_order_items WHERE item = 'apple'")
exact = cur.fetchall()
assert exact == [(1,)], "atomic decomposition gives exact match, no false positive from 'pineapple'"

conn.close()
print("1NF verified: comma-packed column causes real false-positive substring match, atomic decomposition fixes it")
```

**面试怎么问+追问链**

- Q:为什么不能在一列里存一个 JSON 数组或逗号分隔的多个值?
  - 追问1:现代数据库(PostgreSQL的 `jsonb`/`array` 类型)明明支持存数组/JSON,这是不是说1NF已经过时了?
    - 深挖追问(区分度较高):PostgreSQL 的 `jsonb`/`array` 列**可以**被索引(比如 `GIN` 索引)和精确查询单个元素,这和"一列存一个逗号分隔字符串"本质区别在哪?(答案方向:1NF反对的不是"这一列的类型是复合类型"本身,而是反对"数据库引擎没有办法看懂这个复合结构内部、只能整体当不透明字符串处理"——`jsonb`/`array` 是数据库原生理解并能索引的结构化类型,和逗号字符串是两回事,这也是很多人对1NF的误解)。

**常见坑**

- 把"数据库支持数组/JSON类型"和"1NF已经不适用了"混为一谈——原生结构化类型和不透明字符串拼接完全是两回事。
- 用 `LIKE '%x%'` 在拼接字符串里搜索,很容易漏掉这个查询会命中"子串"而不是"精确元素"这个陷阱。

---

## 2. 第二范式(2NF):消除部分依赖

**签名/是什么**

```
2NF: 满足1NF,且每个非主属性完全依赖于整个候选键(不能只依赖复合主键的一部分)
```

一句话:如果一张表的主键是"订单ID+商品ID"这种复合键,但商品名称只依赖"商品ID"这一部分,不依赖订单ID,这就是"部分依赖",违反2NF。

**底层机制/为什么这样设计**

部分依赖的本质问题是"同一个事实(商品100的名字叫Widget)被重复存储在每一行订单明细里"——这不是空间浪费的问题,而是**没有单一数据来源(single source of truth)**:同一个事实的多份拷贝之间没有任何机制保证它们始终一致,任何一次只更新了其中一份拷贝的写操作,都会让数据库进入"同一个商品有两个不同名字"这种逻辑上不该存在的状态。

**AI研究/工程场景**

标注任务表(task_id, annotator_id, annotator_name, label)里 `annotator_name` 只依赖 `annotator_id`,如果标注员改了名字只更新了她最近几条任务记录,历史记录里她的名字就会和最新记录不一致——这类"看似能跑但埋着不一致地雷"的表结构,在标注平台/众包系统里很常见。

**可运行例子**(环境:`.venv`)

```python
import sqlite3

conn = sqlite3.connect(":memory:")
cur = conn.cursor()

# 违反2NF: 复合主键(order_id, product_id), product_name只依赖product_id(部分依赖),不依赖完整主键
cur.execute("""CREATE TABLE order_detail (
    order_id INTEGER, product_id INTEGER, product_name TEXT, quantity INTEGER,
    PRIMARY KEY(order_id, product_id)
)""")
cur.executemany("INSERT INTO order_detail VALUES (?,?,?,?)",
                 [(1, 100, 'Widget', 2), (2, 100, 'Widget', 5), (3, 100, 'Widget', 1)])
conn.commit()

# 更新异常: 只改了订单1里的商品名字,订单2/3里的同一个商品名字没变,数据不一致了
cur.execute("UPDATE order_detail SET product_name='Gadget' WHERE order_id=1 AND product_id=100")
conn.commit()
cur.execute("SELECT DISTINCT product_name FROM order_detail WHERE product_id=100")
names = [r[0] for r in cur.fetchall()]
assert len(names) == 2, f"partial dependency causes real inconsistency: product_id=100 now has names {names}"

# 2NF分解: product_name放进只依赖product_id的独立表,商品明细只留外键引用
cur.execute("CREATE TABLE products (product_id INTEGER PRIMARY KEY, product_name TEXT)")
cur.execute("CREATE TABLE order_detail_2nf (order_id INTEGER, product_id INTEGER, quantity INTEGER, PRIMARY KEY(order_id, product_id))")
cur.execute("INSERT INTO products VALUES (100, 'Widget')")
cur.executemany("INSERT INTO order_detail_2nf VALUES (?,?,?)", [(1,100,2),(2,100,5),(3,100,1)])
conn.commit()

# 改名字现在只需要改products表的一行,结构上就不可能出现不一致
cur.execute("UPDATE products SET product_name='Gadget' WHERE product_id=100")
conn.commit()
cur.execute("SELECT DISTINCT p.product_name FROM order_detail_2nf od JOIN products p ON od.product_id=p.product_id")
names_2nf = [r[0] for r in cur.fetchall()]
assert names_2nf == ['Gadget'], "2NF decomposition makes the inconsistency structurally impossible, not just less likely"

conn.close()
print("2NF verified: partial dependency causes real multi-value inconsistency, decomposition eliminates it structurally")
```

**面试怎么问+追问链**

- Q:什么情况下一张表天然不可能违反2NF?
  - 追问1:如果主键只是单个自增列(不是复合键),这张表还需要检查2NF吗?
    - 深挖追问(区分度较高):为什么(答案方向:2NF的"部分依赖"这个概念只在主键是复合键时才有意义——单列主键不存在"依赖主键的一部分"这回事,所以单主键表天然满足2NF,这也是为什么现代很多设计直接用自增ID做主键而不是业务复合键,能顺带绕开2NF这一层的考量,但这不代表可以不管3NF)。

**常见坑**

- 只在"表结构设计阶段"想范式,不理解"部分依赖"是一个和复合主键强相关的概念,遇到单主键表却在纠结"是不是符合2NF"。
- 把2NF违反的后果误认为只是"存储冗余",忽视了它真正可怕的地方是**结构性的数据不一致风险**。

---

## 3. 第三范式(3NF)与BCNF:传递依赖与候选键场景

**签名/是什么**

```
3NF:  满足2NF,且非主属性不能传递依赖于候选键(A->B->C,C不能只通过B间接依赖A)
BCNF: 更严格版的3NF——每一个决定其他属性的"决定项"本身必须是候选键(不能有例外)
```

一句话:3NF 处理的是"非主属性之间也存在依赖链"的问题(比如"员工->部门->部门经理"),BCNF 处理的是一种3NF会放过的特殊场景——某个非候选键的属性反过来能决定别的候选键属性。

**底层机制/为什么这样设计**

3NF的传递依赖和2NF的部分依赖本质是同一类问题(单一数据来源被破坏),只是依赖链条更长——A决定B,B决定C,那么C事实上被存了多份(每个A的取值都带着一份C的拷贝),即使表面上A和C之间没有直接的函数依赖关系。BCNF存在的原因是:3NF的"非主属性"这个限定词留了个口子——如果一个决定其他属性的列(比如某个教授只教一门课,professor决定subject)本身**恰好**是某个候选键的一部分(subject是主键{student,subject}里的一部分,3NF豁免了"prime attribute"),3NF就不认为这违规,但这仍然会导致真实的数据异常。

**AI研究/工程场景**

排课系统"学生-课程-教授"关系(每个教授只教一门固定的课,但一门课可以有多个教授教)是BCNF问题的经典真实场景——如果只按3NF设计,某个教授调去教另一门课时,所有涉及该教授的历史选课记录需要被逐一手动同步,极易遗漏。

**可运行例子**(环境:`.venv`)

```python
import sqlite3

conn = sqlite3.connect(":memory:")
cur = conn.cursor()

# 表：(student, subject, professor),复合主键(student, subject)
# 业务规则: 每个教授只教一门课(professor -> subject 是一个真实的函数依赖)
# 但 professor 不是候选键的全部(候选键是{student,subject}),这是3NF会放过、但BCNF不会放过的场景
cur.execute("""CREATE TABLE student_subject_prof (
    student TEXT, subject TEXT, professor TEXT, PRIMARY KEY(student, subject)
)""")
cur.executemany("INSERT INTO student_subject_prof VALUES (?,?,?)", [
    ('Alice', 'Math', 'Smith'), ('Bob', 'Math', 'Smith'), ('Carol', 'Physics', 'Jones'),
])
conn.commit()

# Smith 改教 Physics 了,但只手动同步了 Alice 这一行的记录(Bob的忘记同步)
cur.execute("UPDATE student_subject_prof SET subject='Physics' WHERE student='Alice' AND professor='Smith'")
conn.commit()
cur.execute("SELECT DISTINCT subject FROM student_subject_prof WHERE professor='Smith'")
subjects = [r[0] for r in cur.fetchall()]
assert len(subjects) == 2, f"BCNF violation causes real anomaly: professor Smith now appears to teach {subjects}"

# BCNF分解: 把 professor -> subject 这个函数依赖单独拆出一张表(此时professor是这张新表的候选键)
cur.execute("CREATE TABLE professor_subject (professor TEXT PRIMARY KEY, subject TEXT)")
cur.execute("CREATE TABLE enrollment (student TEXT, professor TEXT, PRIMARY KEY(student, professor))")
cur.execute("INSERT INTO professor_subject VALUES ('Smith','Math'), ('Jones','Physics')")
cur.executemany("INSERT INTO enrollment VALUES (?,?)", [('Alice','Smith'),('Bob','Smith'),('Carol','Jones')])
conn.commit()

# 现在Smith改教Physics只需要改professor_subject表的一行,enrollment表完全不用动,不可能出现不一致
cur.execute("UPDATE professor_subject SET subject='Physics' WHERE professor='Smith'")
conn.commit()
cur.execute("""SELECT DISTINCT ps.subject FROM enrollment e
               JOIN professor_subject ps ON e.professor = ps.professor
               WHERE e.professor='Smith'""")
subjects_bcnf = [r[0] for r in cur.fetchall()]
assert subjects_bcnf == ['Physics'], "BCNF decomposition makes the professor-teaches-two-subjects anomaly structurally impossible"

conn.close()
print("BCNF verified: a non-superkey determinant causes real anomaly even though the table is already in 3NF")
```

**面试怎么问+追问链**

- Q:3NF 和 BCNF 具体差在哪?能不能举一个满足3NF但不满足BCNF的例子?
  - 追问1:上面这个例子里,为什么这张表已经是3NF了(3NF为什么"放过"了它)?
    - 深挖追问(区分度较高):BCNF 分解有时会导致"无损连接但不保持函数依赖"(分解后原来的某个业务约束无法再单靠一张表的主键约束表达,需要额外的应用层检查或触发器)——这是工程上很多团队"知道BCNF更严格但止步于3NF"的真实原因,不是不懂,而是权衡过后的选择。

**常见坑**

- 以为只要满足3NF就不会有更新异常——BCNF描述的场景(决定项不是候选键但本身是某个候选键的一部分,因而被3NF豁免)是真实存在的反例,上面的例子就是可复现的真实异常。
- 教条式追求BCNF而不评估分解后是否会丢失能被数据库结构本身表达的业务约束。

---

## 4. 反范式权衡

**签名/是什么**

```
反范式(Denormalization): 有意违反范式,冗余存储数据以减少查询时的JOIN
```

一句话:范式化换来的是"改一处、处处生效"的一致性保证,代价是读取时经常需要JOIN;反范式化用可控的冗余换取读路径的简化,代价是主动放弃了范式化提供的那份一致性保证,必须自己承担维护冗余数据同步的责任。

**底层机制/为什么这样设计**

反范式不是"不懂范式"的产物,而是在"读多写少"场景下的工程权衡:如果一张订单表 99% 的查询都要连着显示客户姓名,而客户改名这种写操作极其罕见,把 `customer_name` 冗余存一份进订单表可以省掉几乎每次查询都要付出的JOIN代价。这本质上是拿 02/03 类演示过的那种"更新异常风险"换取"读路径更简单"——这个风险不是消失了,而是从"数据库结构上不可能发生"变成"需要应用代码/触发器主动维护,一旦漏改就会真实发生"。

**AI研究/工程场景**

数据仓库/报表宽表(大量冗余、故意反范式化以加速分析查询)和在线事务处理(OLTP,通常更范式化以保证写入一致性)是同一个权衡在两种场景下的不同取舍——这也是为什么很多系统会用"范式化的事务库 + ETL同步到反范式化的宽表做分析"这种分层架构,而不是在同一张表里两头都要。

**可运行例子**(环境:`.venv`)

```python
import sqlite3

conn = sqlite3.connect(":memory:")
cur = conn.cursor()

# 范式化: orders只存customer_id,查询名字必须JOIN
cur.execute("CREATE TABLE customers_n (id INTEGER PRIMARY KEY, name TEXT)")
cur.execute("CREATE TABLE orders_n (id INTEGER PRIMARY KEY, customer_id INTEGER, amount INTEGER)")
cur.execute("INSERT INTO customers_n VALUES (1, 'Alice')")
cur.execute("INSERT INTO orders_n VALUES (1, 1, 100)")
conn.commit()
cur.execute("SELECT o.id, c.name, o.amount FROM orders_n o JOIN customers_n c ON o.customer_id=c.id")
assert cur.fetchall() == [(1, "Alice", 100)]

# 反范式化: orders直接冗余存customer_name,读不需要JOIN
cur.execute("CREATE TABLE orders_denorm (id INTEGER PRIMARY KEY, customer_id INTEGER, customer_name TEXT, amount INTEGER)")
cur.executemany("INSERT INTO orders_denorm VALUES (?,?,?,?)", [(1,1,'Alice',100),(2,1,'Alice',200)])
conn.commit()
cur.execute("SELECT id, customer_name, amount FROM orders_denorm")
assert len(cur.fetchall()) == 2  # 不需要JOIN就拿到了名字

# 反范式化的真实代价: Alice改名,如果漏改了一行,同一个customer_id出现两个不同名字
# (和本类知识点2/3演示的是完全同一种异常,只是这次是"主动选择承担"而不是"设计失误")
cur.execute("UPDATE orders_denorm SET customer_name='Alicia' WHERE id=1")
conn.commit()
cur.execute("SELECT DISTINCT customer_name FROM orders_denorm WHERE customer_id=1")
names = [r[0] for r in cur.fetchall()]
assert len(names) == 2, f"denormalization reintroduces the exact update anomaly normalization eliminates: {names}"

conn.close()
print("denormalization tradeoff verified: avoids JOIN on read, but reintroduces the same update anomaly as an accepted risk")
```

**面试怎么问+追问链**

- Q:什么场景下你会选择反范式化设计?
  - 追问1:反范式化之后,谁来保证冗余数据的一致性?
    - 深挖追问(区分度较高):如果选择"应用代码在改名时同步更新所有冗余副本"这个方案,遇到应用代码在同步一半时崩溃/网络分区,会出现什么问题?这和04类"事务与隔离级别"要解决的问题是不是同一类(答案方向:是同一类问题——反范式化的一致性维护本质上就是"多个写操作需要要么全部生效要么全部不生效",没有事务包裹的话,反范式化的冗余同步逻辑本身就有可能被中断在不一致的中间状态,这是为什么"反范式化"和"靠谱的事务边界设计"经常需要放在一起讨论)。

**常见坑**

- 把反范式化当成"性能优化的免费午餐",不设计任何保证冗余一致性的机制(触发器/事务/定期校对任务),异常迟早会真实发生。
- 在写多读少的场景(比如高频更新的库存表)也无脑反范式化,反而让每次写入都要多付出同步冗余副本的代价,得不偿失。

---

## 5. GROUP BY/HAVING与聚合函数求值顺序

**签名/是什么**

```
逻辑求值顺序: FROM -> WHERE -> GROUP BY -> HAVING -> SELECT -> ORDER BY
WHERE  过滤原始行(聚合发生之前,不能用聚合函数)
HAVING 过滤聚合之后的分组结果(可以用聚合函数)
```

一句话:`WHERE` 和 `HAVING` 都是"过滤条件",但作用的对象完全不同——`WHERE` 在分组前逐行过滤,`HAVING` 在分组聚合后对每个分组的聚合结果过滤,顺序搞反会得到错误结果而不是报错。

**底层机制/为什么这样设计**

SQL 语句的书写顺序(`SELECT...FROM...WHERE...GROUP BY...HAVING...ORDER BY`)和真正的逻辑求值顺序不一致,这是初学者最容易困惑的地方之一——`SELECT` 虽然写在最前面,但逻辑上是在 `GROUP BY`/`HAVING` 之后才求值的(所以 `SELECT` 里能用聚合函数,`WHERE` 不能)。`WHERE` 在 `GROUP BY` 之前执行,这时候还没有"分组"这个概念,自然无法引用某个分组的聚合值;`HAVING` 在分组之后执行,此时逐行的原始数据已经不可见(除非也出现在 `GROUP BY` 列里),只能操作聚合后的结果。

**AI研究/工程场景**

"统计每个实验配置下,平均准确率超过90%的那些配置"——这个"平均准确率超过90%"是对聚合结果的过滤,必须写在 `HAVING` 里;如果错写成 `WHERE accuracy > 0.9`,含义就完全变了(变成"先只保留单次准确率超过90%的实验记录,再统计这些记录的平均值",两者结果可能天差地别)。

**可运行例子**(环境:`.venv`)

```python
import sqlite3

conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.execute("CREATE TABLE sales (region TEXT, amount INTEGER)")
cur.executemany("INSERT INTO sales VALUES (?,?)",
                 [('East', 100), ('East', 200), ('West', 50), ('West', 30), ('North', 500)])
conn.commit()

# WHERE amount > 40 在分组前逐行过滤(West的30这一行被排除)
# GROUP BY region 之后, HAVING SUM(amount) > 100 只保留聚合和超过100的分组
cur.execute("""SELECT region, SUM(amount) FROM sales
               WHERE amount > 40
               GROUP BY region
               HAVING SUM(amount) > 100
               ORDER BY region""")
result = cur.fetchall()
# East: 100+200=300(都通过WHERE) -> 300>100通过HAVING
# West: 只剩50(30被WHERE排除) -> 50<=100不通过HAVING,整个West分组消失
# North: 500 -> 通过HAVING
assert result == [("East", 300), ("North", 500)], f"got {result}"

# 反例:如果把WHERE的过滤条件误写进HAVING(逻辑完全不同)
cur.execute("""SELECT region, SUM(amount) FROM sales
               GROUP BY region
               HAVING SUM(amount) > 40 AND SUM(amount) > 100
               ORDER BY region""")
wrong_semantics_result = cur.fetchall()
# West现在是50+30=80(WHERE没有排除30了),80<=100依然不通过,但East/North的SUM也变了(因为WHERE没有先过滤)
assert wrong_semantics_result == [("East", 300), ("North", 500)]
# 这个例子里巧合地结果一样,但过滤的中间语义完全不同,数据变化时很容易产生真实分歧,不能依赖"结果碰巧一样"

conn.close()
print("GROUP BY/HAVING evaluation order verified: WHERE filters pre-aggregation rows, HAVING filters post-aggregation groups")
```

**面试怎么问+追问链**

- Q:能不能把 `WHERE amount > 40` 直接写成 `HAVING amount > 40`?
  - 追问1:如果表里没有 `GROUP BY`(整张表当一个分组),`HAVING` 里能不能引用没有被聚合的列?
    - 深挖追问(区分度较高):`WHERE` 里不能用聚合函数,那如果我确实想基于"某个聚合结果"过滤原始行(比如"只看销售额超过区域平均值的订单"),该怎么写?(答案方向:要么用窗口函数配合外层再过滤一次,要么用子查询先算出聚合值再和原表JOIN/比较,这也是"进阶SQL的多种等价写法"这个主题下一个自然的延伸)。

**常见坑**

- 把本该在 `WHERE` 里做的逐行过滤写进 `HAVING`,虽然多数简单场景结果凑巧一样,但性能上 `HAVING` 的过滤发生在聚合计算**之后**,`WHERE` 发生在聚合**之前**——用 `WHERE` 能提前减少参与聚合计算的行数,写反了会让数据库多做无谓的聚合工作。
- 在 `WHERE` 子句里直接写聚合函数(比如 `WHERE SUM(amount) > 100`),这在标准SQL里是语法错误,因为 `WHERE` 求值时聚合结果还不存在。

---

## 6. 窗口函数:ROW_NUMBER/RANK/LAG-LEAD

**签名/是什么**

```
函数() OVER (PARTITION BY 分组列 ORDER BY 排序列)
ROW_NUMBER()  组内行号,即使值相同也严格递增,不并列
RANK()        组内排名,值相同则并列,并列后下一名跳过相应名次
LAG(col)      取组内按排序顺序的"上一行"该列的值
LEAD(col)     取组内按排序顺序的"下一行"该列的值
```

一句话:窗口函数在**不改变结果集行数**的前提下,让每一行都能"看到"它所在分组内其他行的信息(排名/前后邻居的值),这是它和 `GROUP BY`(会把多行压缩成一行)最根本的区别。

**底层机制/为什么这样设计**

`GROUP BY` 聚合之后,原始的逐行明细就丢失了(只剩每组一行的汇总结果);而"求每个学科成绩第一名是谁""每个学生比上一次考试进步了多少"这类问题,恰恰需要"既保留每一行的明细,又能引用同组其他行的聚合/排序信息"。窗口函数通过 `OVER (PARTITION BY ... ORDER BY ...)` 定义了一个"对当前行可见的窗口"(默认是同一分组内、按排序顺序的所有行),函数在这个窗口内计算,但计算结果附加在原来的每一行上,行数不变——这是它区别于 `GROUP BY` 的核心设计。`RANK()` 和 `ROW_NUMBER()` 的关键区别只在"如何处理并列":`ROW_NUMBER` 强制打散并列(哪怕值相同也给不同的序号),`RANK` 尊重并列但会在并列后产生名次空缺(两个并列第1,下一个直接是第3,不是第2)。

**AI研究/工程场景**

"每个用户最近一次登录时间"(`ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_time DESC)` 取第1行)、"计算每次实验相比上一次实验的指标变化"(`LAG` 取上一行的指标值再相减)是窗口函数最典型的两类真实工程用法,比自己写自连接(self-join)或子查询实现同样效果要清晰得多。

**可运行例子**(环境:`.venv`)

```python
import sqlite3

conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.execute("CREATE TABLE scores (student TEXT, subject TEXT, score INTEGER)")
cur.executemany("INSERT INTO scores VALUES (?,?,?)", [
    ('Alice','Math',90), ('Bob','Math',85), ('Carol','Math',90),
    ('Alice','Physics',70), ('Bob','Physics',95),
])
conn.commit()

cur.execute("""SELECT student, subject, score,
    ROW_NUMBER() OVER (PARTITION BY subject ORDER BY score DESC) as rn,
    RANK() OVER (PARTITION BY subject ORDER BY score DESC) as rk
    FROM scores ORDER BY subject, rn""")
rows = cur.fetchall()
math_rows = [r for r in rows if r[1] == 'Math']
# Math: Alice90, Carol90(并列第一), Bob85(第三)
row_numbers = [r[3] for r in math_rows]
ranks = [r[4] for r in math_rows]
assert row_numbers == [1, 2, 3], "ROW_NUMBER always strictly increasing, never ties"
assert ranks == [1, 1, 3], "RANK gives tied rows the same rank and skips the next rank number"

cur.execute("""SELECT student, subject, score,
    LAG(score) OVER (PARTITION BY subject ORDER BY score DESC) as prev_score,
    LEAD(score) OVER (PARTITION BY subject ORDER BY score DESC) as next_score
    FROM scores WHERE subject='Physics' ORDER BY score DESC""")
phys = cur.fetchall()
# Physics 按分数降序: Bob(95), Alice(70)
assert phys[0][3] is None and phys[0][4] == 70, "first row has no prev, next is Alice's 70"
assert phys[1][3] == 95 and phys[1][4] is None, "last row's prev is Bob's 95, no next"

conn.close()
print("window functions verified: ROW_NUMBER never ties, RANK ties-and-skips, LAG/LEAD access neighbor rows correctly")
```

**面试怎么问+追问链**

- Q:`RANK()` 和 `DENSE_RANK()`(没在上面例子里但常被一起问)有什么区别?
  - 追问1:两个人并列第一,`RANK()` 给下一个人第几名?`DENSE_RANK()` 呢?
    - 深挖追问(区分度较高):什么业务场景下你会选 `RANK()` 而不是 `DENSE_RANK()`(答案方向:比如奥运奖牌榜——两人并列金牌,下一个确实应该是"季军"[对应`RANK`的3]而不是"银牌"[对应`DENSE_RANK`的2],因为金牌被两人占用后银牌这个名次事实上空缺了;而"考试成绩分等级"这种场景可能更想用`DENSE_RANK`保证等级连续不跳空)。

**常见坑**

- 把窗口函数和 `GROUP BY` 混着理解,忘记窗口函数不会压缩行数——`SELECT` 列表里同时出现窗口函数和普通聚合函数(不带`OVER`)时容易在语义上搞混。
- 忘记 `PARTITION BY` 是可选的——不写就是把整个结果集当一个大分组处理,`ORDER BY` 也是可选的——不写 `RANK`/`ROW_NUMBER` 这类依赖顺序的函数结果是不确定的。

---

## 7. 子查询/JOIN/CTE(含递归)的等价改写与优化器行为差异

**签名/是什么**

```
子查询(Subquery): SELECT ... WHERE col IN (SELECT ...)
JOIN:             SELECT ... FROM a JOIN b ON ...
CTE(WITH子句):     WITH name AS (SELECT ...) SELECT ... FROM name
递归CTE:           WITH RECURSIVE name AS (基础case UNION ALL 递归case) SELECT ...
```

一句话:同一个查询需求经常可以用子查询、JOIN、CTE 三种不同写法表达,它们在**逻辑结果**上可以完全等价,但数据库优化器**不保证**会把它们变成同一个执行计划,递归CTE则是处理"层级/图结构数据"(比如组织架构树)时,子查询和JOIN都无法优雅表达的场景。

**底层机制/为什么这样设计**

优化器面对"WHERE col IN (子查询)"和"JOIN"这两种写法,理论上可以做等价改写(把 IN 子查询转成一个半连接 semi-join),但实际是否这样做、选择什么具体算法,取决于每个引擎自己的优化器实现和当时的统计信息——**不能假设两种写法性能一定相同,必须用 `EXPLAIN` 实测**。CTE(`WITH` 子句)本质上是给一个子查询起了个名字方便复用/提高可读性,大多数现代引擎会把非递归CTE当成子查询或临时视图处理(具体是"内联展开"还是"物化成临时结果"取决于引擎)。递归CTE的特殊之处在于它定义了一个真正的循环结构:先算出"基础case"(种子行),再反复把"递归case"应用在上一轮新产生的行上,直到某一轮不再产生新行为止(自然终止),这是表达"层级展开"(比如"某个员工的所有下属,一层一层往下")在纯SQL里几乎唯一优雅的方式。

**AI研究/工程场景**

CTE 在写复杂的多步骤数据处理查询时(比如"先筛选出活跃用户,再计算这些用户的特征聚合,再和另一张表关联")能把原本要嵌套好几层的子查询拆成一串命名清晰、可读性好的步骤,等价于给SQL查询本身做"变量命名"式重构。递归CTE是查询组织架构、类目树、社交网络"N度好友"这类层级/图结构数据的标准手段。

**可运行例子**(环境:`.venv`)

```python
import sqlite3

conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.execute("CREATE TABLE depts (id INTEGER PRIMARY KEY, name TEXT)")
cur.execute("CREATE TABLE emps (id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER, salary INTEGER)")
cur.execute("INSERT INTO depts VALUES (1,'Eng'),(2,'Sales')")
cur.executemany("INSERT INTO emps VALUES (?,?,?,?)", [(1,'A',1,100),(2,'B',1,200),(3,'C',2,50)])
conn.commit()

# 子查询 vs JOIN: 逻辑结果完全等价
cur.execute("SELECT name FROM emps WHERE dept_id IN (SELECT id FROM depts WHERE name='Eng') ORDER BY name")
sub_result = cur.fetchall()
cur.execute("SELECT e.name FROM emps e JOIN depts d ON e.dept_id=d.id WHERE d.name='Eng' ORDER BY e.name")
join_result = cur.fetchall()
assert sub_result == join_result == [("A",), ("B",)], "subquery and JOIN forms are logically equivalent here"

# 但EXPLAIN QUERY PLAN显示SQLite实际选择了不同的执行策略,不能假设两种写法性能一致
cur.execute("EXPLAIN QUERY PLAN SELECT name FROM emps WHERE dept_id IN (SELECT id FROM depts WHERE name='Eng')")
sub_plan = [row[3] for row in cur.fetchall()]
cur.execute("EXPLAIN QUERY PLAN SELECT e.name FROM emps e JOIN depts d ON e.dept_id=d.id WHERE d.name='Eng'")
join_plan = [row[3] for row in cur.fetchall()]
# 如实记录观察到的真实差异:子查询走了"LIST SUBQUERY"+布隆过滤器,JOIN走了直接的主键索引查找
assert any("SUBQUERY" in step for step in sub_plan), f"expected subquery-specific strategy, got {sub_plan}"
assert any("SEARCH" in step for step in join_plan), f"expected index search strategy, got {join_plan}"
assert sub_plan != join_plan, "SQLite's optimizer genuinely chose different execution strategies for logically equivalent queries"

# CTE: 给子查询起名字,可读性更好,逻辑结果不变
cur.execute("WITH eng_emps AS (SELECT * FROM emps WHERE dept_id=1) SELECT name FROM eng_emps ORDER BY name")
assert cur.fetchall() == [("A",), ("B",)]

# 递归CTE: 生成1到5的序列,验证递归能正确终止(WHERE n < 5这个条件是终止条件)
cur.execute("""
WITH RECURSIVE seq(n) AS (
    SELECT 1
    UNION ALL
    SELECT n+1 FROM seq WHERE n < 5
)
SELECT n FROM seq
""")
assert cur.fetchall() == [(1,), (2,), (3,), (4,), (5,)], "recursive CTE terminates correctly at the WHERE condition"

conn.close()
print("subquery/JOIN/CTE equivalence and optimizer divergence verified; recursive CTE terminates correctly")
```

**面试怎么问+追问链**

- Q:子查询和JOIN,哪个性能更好?
  - 追问1:上面的例子里,SQLite对子查询和JOIN到底给出了一样的执行计划吗?
    - 深挖追问(区分度较高):既然结果证明了执行计划**不一样**,那"子查询和JOIN性能一样,只是写法不同"这种说法为什么还这么常见(答案方向:很多简单场景下,优化器确实能把两者转成等价的执行策略,差异小到可以忽略,导致这个说法在"大多数情况下"成立,但不是恒成立的定理——本例用EXPLAIN QUERY PLAN实测出的真实差异[LIST SUBQUERY+布隆过滤器 vs 主键索引查找]就是一个反例,面试时给出"取决于具体查询和引擎,需要实测"是比"性能一样"更准确的回答)。

**常见坑**

- 想当然认为"子查询会被优化器自动改写成JOIN,所以性能一样"——这是一个流传很广但不总是成立的说法,本类的可运行例子里 SQLite 就选择了两种真实不同的执行策略(不是理论推测,是`EXPLAIN QUERY PLAN`的真实输出)。
- 写递归CTE忘记终止条件(比如把 `WHERE n < 5` 漏掉),会导致无限递归,大部分引擎有默认的递归深度上限会报错终止,但这个上限本身也是需要知道的运维细节。
