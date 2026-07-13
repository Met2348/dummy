# 数据库原理与实战深挖 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task(内联执行模式,不用subagent-driven-development——用户已明确"立即开始",延续本仓库标准授权模式,任务间不暂停等待确认)。Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `for_real_dummy/database-deep-dive/` 下产出一套约 65 个知识点、6 大板块 8 个分类文件 + 1 篇模拟终面 capstone 的数据库原理与实战深挖系列,深度对标本仓库已完成的 os-concurrency-deep-dive(79点)/computer-networking-deep-dive(80点),达到技术终面级别深度广度,关系数据库加重(分布式/NoSQL点到面试常考程度)。

**Architecture:** 每个分类文件独立成篇,遵循统一的七步知识点模板(签名/是什么→一句话→底层机制/为什么这样设计→AI研究/工程场景→可运行例子→面试怎么问+追问链→常见坑),五轴追问链方法论从第一天融入"面试怎么问"步骤。验证采用双环境策略:关系模型/SQL基础/纯算法模拟用仓库 `.venv`(stdlib `sqlite3`);隔离级别/MVCC/锁/复制这类深水区机制,用 WSL2 Rocky Linux 内**真实安装的 PostgreSQL 16.14 + MariaDB 10.11.15 双引擎**给出对比验证证据,外加 Valkey 8.0.9(Rocky Linux 10 因 Redis 2024 年许可证变更,dnf 仓库提供的是 Linux Foundation 维护的协议兼容分支 Valkey,不是字面意义的"redis"包——这是本次环境搭建阶段发现的真实情况,如实记录并在 08 类知识点里向读者说明)覆盖 NoSQL 板块。

**Tech Stack:** Python 3(仓库根目录 `.venv`,标准库为主:`sqlite3`,不新增第三方依赖)+ WSL2 Rocky Linux 10(`/usr/bin/python3` 3.12.13 用于系统级验证;**新增专用虚拟环境** `~/database-deep-dive-venv`,内装 `psycopg2-binary 2.9.12` / `PyMySQL 1.2.0` / `redis 8.0.1` 三个第三方驱动——这是相对前两部"标准库优先、不新增依赖"纪律的必要例外,因为连接真实 PostgreSQL/MariaDB/Redis 协议标准库没有对应实现)+ PostgreSQL 16.14 + MariaDB 10.11.15 + Valkey 8.0.9(均通过 dnf 直接安装,systemd 管理服务)。

## Global Constraints

- 环境:默认仓库根目录 `.venv`(Windows 原生,`sqlite3` 覆盖关系模型/SQL基础/范式/纯算法模拟);真实双引擎深水区机制验证用 WSL2 Rocky Linux(`wsl.exe -d RockyLinux`,Git Bash 调用前必须 `export MSYS_NO_PATHCONV=1`)。
- **WSL2 内允许为 psycopg2-binary/PyMySQL/redis 三个包新增第三方依赖**(标准库没有对应协议实现,这是数据库系列相对前两部"不新增依赖"纪律的必要修订),但必须装在专用虚拟环境 `~/database-deep-dive-venv`(已创建,已验证三个驱动均可用),不装进系统 python3(Rocky Linux 10 的系统 Python 是 PEP 668 externally-managed,且系统 python3 本身连 `pip` 模块都未预装)。Windows 端 `.venv` 继续保持纯标准库,不新增任何依赖。
- **真实数据库连接凭据(已创建,固定复用,不要重新生成)**:
  - PostgreSQL:host=`127.0.0.1` port=`5432` dbname=`dbdemo` user=`dbdemo` password=`dbdemo_local_only`(`pg_hba.conf` 已把 `127.0.0.1/32` 规则从默认 `ident` 改为 `scram-sha-256`,允许密码认证;Unix socket 仍是默认 `peer`,`su - postgres -c psql` 可继续用于管理员操作)。
  - MariaDB:host=`127.0.0.1` port=`3306` database=`dbdemo` user=`dbdemo` password=`dbdemo_local_only`(已建 `dbdemo`@`127.0.0.1` 和 `dbdemo`@`localhost` 两条,root 走 Unix socket 免密)。
  - Valkey(Redis协议兼容):host=`127.0.0.1` port=`6379`,无密码(本地回环+trusted zone,纯学习环境不设 `requirepass`,如实标注这是有意选择不是疏漏)。
- **firewalld 环境坑(已修复,写入 Task 1 供复现)**:WSL2 镜像网络模式下会出现一个叫 `loopback0` 的合成网卡(NetworkManager 连接名 `Wired connection 5`,不是内核真正的 `lo`),firewalld 默认把它划进限制性的 `public` zone,导致所有到 `127.0.0.1` 的 TCP 连接报 `[Errno 113] No route to host`——这正是上一部 computer-networking-deep-dive 系列里"未解决"的同名问题,这次真正定位到根因并修复(`nmcli connection modify 'Wired connection 5' connection.zone trusted`),不是重新踩坑。`public` zone 对 `eth1/eth2/eth3` 的限制未受影响,不破坏 rhcsa-bash-deep-dive 08类 firewalld 教学内容的有效性。
- 知识点模板固定七步(签名/是什么→一句话→底层机制/为什么这样设计→AI研究/工程场景→可运行例子→面试怎么问+追问链→常见坑),不单独拆"数学推导"步骤——B+树IO复杂度/ARIES恢复算法/MVCC可见性规则/CAP推导直接写进"底层机制"步骤。
- 五轴追问链方法论(规模递增/工程约束递增/方案批判迭代/决策依据追问/真实性验证 + "诊断真实数据"新题型)从第一天融入"面试怎么问+追问链"步骤,每点挑 1~2 条最自然的轴线,不强行凑满 5 轴。
- 涉及真实数据库连接的例子一律用 `python-wsl2` 围栏(不在 Windows `.venv` 执行),脚本开头必须显式标注"以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14 / MariaDB 10.11.15 / Valkey 8.0.9(已启动),使用 `~/database-deep-dive-venv` 虚拟环境"。sqlite3 相关例子用普通 `python` 围栏在 `.venv` 执行。
- 涉及具体 EXPLAIN 输出格式/系统表结构的断言,注明引擎版本(PostgreSQL 16.14 / MariaDB 10.11.15),避免未来版本升级后断言失效被误判为代码错误。
- 隔离级别异常复现涉及两个真实数据库连接的时序控制,必须用显式同步点(`threading.Event`/队列,不能用 `sleep` 猜时间)控制两个连接的操作顺序,并至少重复 5~10 次确认现象稳定复现。
- 不连接外部/生产数据库,所有实例监听本地(Unix socket 或 127.0.0.1),测试数据全部是脚本内合成的最小数据集。
- 需要真实构造死锁/长事务阻塞的 demo 必须带确定性退出的超时安全网(延续 os-concurrency-deep-dive 死锁 demo 纪律)。
- print() 语句必须纯 ASCII(历史教训:Windows GBK locale 下 `_verify_md.py` 子进程 reader 线程遇到非 ASCII 字符会 UnicodeDecodeError)。
- 每个板块验证通过后独立 git commit;`git add` 必须显式列出文件路径,不用 `git add -A`/`.`。
- 跨文件引用其他分类知识点用既定的"NN类知识点M"纯文本格式,不用 markdown 链接锚点。
- 涉及 LRU缓存淘汰策略(06类,交叉引用dsa-deep-dive 05类)、一致性哈希(07类,交叉引用dsa-deep-dive 20类)、连接池通用机制(08类,交叉引用computer-networking-deep-dive 11类)、OS级互斥锁与死锁理论(05类,和os-concurrency-deep-dive 03/05类划清边界不重复推导)时,交叉引用已有内容的真实结论,不重新写一遍代码或重新推导。

---

### Task 1: 脚手架 + WSL2 双引擎与 Valkey 环境搭建(已实测验证,本任务记录可复现步骤)

**Files:**
- Create: `for_real_dummy/database-deep-dive/00-roadmap.md`
- Create: `for_real_dummy/database-deep-dive/_verify_md.py`(从 `for_real_dummy/computer-networking-deep-dive/_verify_md.py` 原样拷贝,含 `python-wsl2` 围栏标记支持)

**Interfaces:**
- Produces:`00-roadmap.md` 含 6 板块 8 文件 + capstone 的进度表(初始全部标"⏳ 未开始"),供 Task 2-9 逐行更新为 "✅ 已完成"。`_verify_md.py` 供 Task 2-9 每篇验证时调用:`python _verify_md.py <path/to/file.md>`。真实数据库凭据见 Global Constraints,供 Task 4-9 直接复用不重新创建。

- [ ] **Step 1: 拷贝验证脚本**

```bash
cp for_real_dummy/computer-networking-deep-dive/_verify_md.py for_real_dummy/database-deep-dive/_verify_md.py
```

- [ ] **Step 2: 安装 PostgreSQL 16.14 并配置本地密码认证**

```bash
export MSYS_NO_PATHCONV=1
wsl.exe -d RockyLinux -- dnf install -y postgresql-server postgresql-contrib
wsl.exe -d RockyLinux -- postgresql-setup --initdb
wsl.exe -d RockyLinux -- systemctl enable --now postgresql
wsl.exe -d RockyLinux -- systemctl is-active postgresql
```
Expected: `active`(已验证)。

```bash
wsl.exe -d RockyLinux -- sed -i 's/^host    all             all             127.0.0.1\/32            ident/host    all             all             127.0.0.1\/32            scram-sha-256/' /var/lib/pgsql/data/pg_hba.conf
wsl.exe -d RockyLinux -- systemctl restart postgresql
wsl.exe -d RockyLinux -- su - postgres -c "psql -c \"CREATE USER dbdemo WITH PASSWORD 'dbdemo_local_only';\""
wsl.exe -d RockyLinux -- su - postgres -c "psql -c \"CREATE DATABASE dbdemo OWNER dbdemo;\""
```
Expected: `CREATE ROLE` / `CREATE DATABASE`(已验证)。

- [ ] **Step 3: 安装 MariaDB 10.11.15 并配置本地密码认证**

```bash
wsl.exe -d RockyLinux -- dnf install -y mariadb-server
wsl.exe -d RockyLinux -- systemctl enable --now mariadb
wsl.exe -d RockyLinux -- mysql --version
```
Expected: `mysql  Ver 15.1 Distrib 10.11.15-MariaDB...`(已验证)。

```bash
wsl.exe -d RockyLinux -- mysql -u root -e "CREATE USER 'dbdemo'@'127.0.0.1' IDENTIFIED BY 'dbdemo_local_only'; CREATE USER 'dbdemo'@'localhost' IDENTIFIED BY 'dbdemo_local_only'; CREATE DATABASE dbdemo; GRANT ALL PRIVILEGES ON dbdemo.* TO 'dbdemo'@'127.0.0.1'; GRANT ALL PRIVILEGES ON dbdemo.* TO 'dbdemo'@'localhost'; FLUSH PRIVILEGES;"
```
Expected: 无输出即成功(已验证,`SELECT User, Host FROM mysql.user WHERE User='dbdemo'` 返回两行)。

- [ ] **Step 4: 安装 epel-release + Valkey(Redis协议兼容分支)**

```bash
wsl.exe -d RockyLinux -- dnf install -y epel-release
wsl.exe -d RockyLinux -- dnf install -y valkey
wsl.exe -d RockyLinux -- systemctl enable --now valkey
wsl.exe -d RockyLinux -- systemctl is-active valkey
```
Expected: `active`(已验证)。**如实记录**:Rocky Linux 10 的 dnf 仓库没有 `redis` 包——2024 年 Redis Inc. 把 Redis 许可证改为非 OSI 认证的 SSPL/RSALv2 后,RHEL 系发行版转向 Linux Foundation 主导的开源分支 Valkey(协议完全兼容,`redis`-py 驱动可直接连接)。08 类知识点撰写时需要向读者说明这个 2024-2026 年真实发生的生态变化,不能含糊带过当成"就是装了redis"。

- [ ] **Step 5: 修复 firewalld 阻断 127.0.0.1 的问题(WSL2镜像网络已知坑,必须做,否则 Step 6 起的 Python 驱动连接会全部失败)**

先复现问题确认(可跳过,已知会失败):
```bash
wsl.exe -d RockyLinux -- python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', 6379))"
```
Expected(未修复前): `OSError: [Errno 113] No route to host`。

根因:WSL2 镜像网络模式下的合成网卡 `loopback0`(NetworkManager 连接名 `Wired connection 5`)被 firewalld 划进限制性的 `public` zone。修复:

```bash
wsl.exe -d RockyLinux -- nmcli connection modify 'Wired connection 5' connection.zone trusted
wsl.exe -d RockyLinux -- firewall-cmd --reload
wsl.exe -d RockyLinux -- firewall-cmd --get-zone-of-interface=loopback0
```
Expected: `trusted`(已验证)。**不要**用 `firewall-cmd --zone=trusted --change-interface=loopback0 --permanent` 单独修——这条命令表面执行成功但会被 NetworkManager 的 zone 绑定覆盖,重载后打回 `public`,必须通过 `nmcli connection modify` 修改 NM 侧的 `connection.zone` 属性才会真正生效。修复后验证 `public` zone 仍只管 `eth1 eth2 eth3`(不影响 rhcsa-bash-deep-dive 08类 firewalld 教学内容):

```bash
wsl.exe -d RockyLinux -- firewall-cmd --zone=public --list-interfaces
```
Expected: `eth1 eth2 eth3`(已验证)。

- [ ] **Step 6: 创建 WSL2 专用 Python 虚拟环境并安装三个数据库驱动**

```bash
wsl.exe -d RockyLinux -- bash -lc "python3 -m venv ~/database-deep-dive-venv && ~/database-deep-dive-venv/bin/pip install --quiet --upgrade pip && ~/database-deep-dive-venv/bin/pip install --quiet psycopg2-binary pymysql redis"
wsl.exe -d RockyLinux -- bash -lc "~/database-deep-dive-venv/bin/pip list | grep -iE 'psycopg2|pymysql|redis'"
```
Expected: `psycopg2-binary 2.9.12` / `PyMySQL 1.2.0` / `redis 8.0.1`(已验证,版本号可能因后续`pip`索引更新而略有不同,以实际输出为准)。

- [ ] **Step 7: 端到端验证三个驱动都能真实连接(Task 4 起的知识点撰写前必须确认这一步通过)**

```bash
wsl.exe -d RockyLinux -- bash -lc "~/database-deep-dive-venv/bin/python3 -c \"
import psycopg2, pymysql, redis
pg = psycopg2.connect(host='127.0.0.1', port=5432, dbname='dbdemo', user='dbdemo', password='dbdemo_local_only')
cur = pg.cursor(); cur.execute('SELECT version();'); assert 'PostgreSQL 16' in cur.fetchone()[0]; cur.close(); pg.close()
maria = pymysql.connect(host='127.0.0.1', port=3306, database='dbdemo', user='dbdemo', password='dbdemo_local_only')
cur = maria.cursor(); cur.execute('SELECT VERSION();'); assert 'MariaDB' in cur.fetchone()[0]; cur.close(); maria.close()
r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
r.set('k','v'); assert r.get('k')=='v'; r.delete('k')
print('ALL_DRIVERS_VERIFIED')
\""
```
Expected: `ALL_DRIVERS_VERIFIED`(已验证)。

- [ ] **Step 8: 撰写 `00-roadmap.md`**

内容包含:目标声明(约65个知识点,6板块8文件+capstone,对标os-concurrency-deep-dive/computer-networking-deep-dive深度,关系数据库加重)、与 rhcsa-bash-deep-dive(核实过完全无重叠)+ dsa-deep-dive(08类无B+树/05类LRU缓存/20类一致性哈希交叉引用)+ os-concurrency-deep-dive(03/05类OS锁与死锁 vs 本系列数据库引擎锁的边界)+ computer-networking-deep-dive(11类连接池交叉引用)的边界声明、七步知识点模板说明、五轴追问链方法论表格(照抄 spec §4)、双环境验证声明(含真实凭据表格,复制 Global Constraints 里的连接信息)、firewalld loopback0 坑的修复记录(供未来复现)、进度表(下表,初始状态全部"⏳ 未开始"):

| # | 板块 | 分类 | 文件 | 知识点数(约) | 环境 | 状态 |
|---|------|------|------|------------|------|------|
| 01 | I 关系模型与SQL基础 | 关系模型与基础SQL | 01-relational-model-and-sql-basics.md | 7 | .venv | ⏳ |
| 02 | I | 范式与进阶SQL | 02-normalization-and-advanced-sql.md | 7 | .venv | ⏳ |
| 03 | II 索引与查询优化 | 索引结构与查询优化器 | 03-indexing-and-query-optimization.md | 10 | .venv+WSL2 | ⏳ |
| 04 | III 事务与并发控制 | 事务与隔离级别 | 04-transactions-and-isolation-levels.md | 8 | WSL2双引擎 | ⏳ |
| 05 | III | MVCC与锁机制 | 05-mvcc-and-locking.md | 8 | WSL2双引擎 | ⏳ |
| 06 | IV 存储引擎内部机制 | 存储引擎内部机制 | 06-storage-engine-internals.md | 9 | .venv+WSL2 | ⏳ |
| 07 | V 分布式数据库与复制 | 复制与分布式基础 | 07-replication-and-distributed-basics.md | 7 | WSL2双引擎 | ⏳ |
| 08 | VI NoSQL与现代工程场景 | NoSQL与缓存模式 | 08-nosql-and-caching-patterns.md | 9 | WSL2 Valkey | ⏳ |
| 09 | 收尾 | 模拟终面capstone | 09-mock-interview-capstone.md | —(不计入合计) | 混合 | ⏳ |

- [ ] **Step 9: Commit**

```bash
git add for_real_dummy/database-deep-dive/00-roadmap.md for_real_dummy/database-deep-dive/_verify_md.py
git commit -m "docs(database-deep-dive): 脚手架 - roadmap + 验证脚本 + WSL2双引擎环境搭建"
```

---

### Task 2: 01-relational-model-and-sql-basics.md(关系模型与基础SQL)

**Files:**
- Create: `for_real_dummy/database-deep-dive/01-relational-model-and-sql-basics.md`
- Modify: `for_real_dummy/database-deep-dive/00-roadmap.md`(第01行)

知识点范围(7个):关系模型三要素(关系/元组/属性/域)、主键/外键/唯一约束/检查约束、DDL基础(CREATE/ALTER/DROP的语义与代价)、DML基础(INSERT/UPDATE/DELETE的原子性)、四种JOIN类型(INNER/LEFT/RIGHT/FULL)与执行逻辑、NULL的三值逻辑陷阱、DISTINCT与集合运算(UNION/INTERSECT/EXCEPT)。

- [ ] **Step 1: 设计知识点并验证核心结论**

全部用 `.venv` stdlib `sqlite3` 验证:约束验证(插入违反外键/唯一约束的行,验证 `sqlite3.IntegrityError` 真实抛出,需要 `PRAGMA foreign_keys = ON`,SQLite 默认关闭外键检查这个坑本身也是一个知识点)。四种JOIN:构造两张有部分匹配、部分不匹配行的表,验证 INNER/LEFT/RIGHT(SQLite不支持RIGHT JOIN需要用LEFT JOIN对调表顺序模拟,如实标注)/FULL(SQLite 3.39+原生支持,用`.venv`实际sqlite3版本验证,不支持则用UNION模拟并标注)四种JOIN返回的行数和NULL填充位置符合预期。NULL三值逻辑:构造 `NULL = NULL`(返回NULL不是TRUE)、`NULL AND FALSE`(返回FALSE)、`NOT IN` 遇到NULL整体失效这三个真实反例,`assert` 验证查询结果和"以为的"结果不同。DISTINCT/UNION/INTERSECT/EXCEPT:构造有重复行的数据,验证三种集合运算的真实返回行数符合集合论定义。

- [ ] **Step 2: 撰写完整 markdown**

按七步模板撰写全部 7 个知识点,"面试怎么问+追问链"融入五轴方法论。

- [ ] **Step 3: 运行验证脚本**

```bash
cd for_real_dummy/database-deep-dive && python _verify_md.py 01-relational-model-and-sql-basics.md
```
Expected: 全部代码块 PASS。

- [ ] **Step 4: 检查 print() 语句纯 ASCII**

```bash
grep -P '[^\x00-\x7F]' for_real_dummy/database-deep-dive/01-relational-model-and-sql-basics.md | grep 'print('
```
Expected: 无匹配。

- [ ] **Step 5: 更新 roadmap 第01行状态为 ✅ 已完成**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/database-deep-dive/01-relational-model-and-sql-basics.md for_real_dummy/database-deep-dive/00-roadmap.md
git commit -m "docs(database-deep-dive): 01类 关系模型与基础SQL(7点)"
```

---

### Task 3: 02-normalization-and-advanced-sql.md(范式与进阶SQL)

**Files:**
- Create: `for_real_dummy/database-deep-dive/02-normalization-and-advanced-sql.md`
- Modify: `00-roadmap.md`(第02行)

知识点范围(7个):1NF/2NF/3NF/BCNF递进式反例驱动讲解、业务场景下的反范式权衡、GROUP BY/HAVING与聚合函数求值顺序、窗口函数(ROW_NUMBER/RANK/LAG-LEAD)、子查询vs JOIN的等价改写与优化器行为差异、CTE(WITH子句)与递归查询。

- [ ] **Step 1: 设计知识点并验证核心结论**

范式:`.venv` sqlite3 构造一张违反 2NF(部分依赖)的表,演示更新异常真实发生(改一行导致数据不一致,`assert` 验证异常状态确实出现),再拆分成符合 2NF 的两张表验证异常消失。GROUP BY/HAVING 求值顺序:构造一个 `WHERE` 和 `HAVING` 都能过滤但结果不同的查询,验证执行顺序(FROM→WHERE→GROUP BY→HAVING→SELECT→ORDER BY)真实影响结果集。窗口函数:SQLite 3.25+ 原生支持,验证 `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)` 分组内排名正确,`LAG`/`LEAD` 取到相邻行的值正确。子查询vs JOIN:构造语义等价的两种写法,验证返回结果集相同(`assert` 两种写法结果一致),用 `EXPLAIN QUERY PLAN` 观察 SQLite 优化器是否将子查询改写为 JOIN(如实记录实际观察到的执行计划,不预设结论)。递归CTE:经典的"组织架构树"或数字序列生成,验证递归终止条件正确、生成的行数符合预期。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本并确认 PASS**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第02行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/database-deep-dive/02-normalization-and-advanced-sql.md for_real_dummy/database-deep-dive/00-roadmap.md
git commit -m "docs(database-deep-dive): 02类 范式与进阶SQL(7点) - 板块I完成"
```

---

### Task 4: 03-indexing-and-query-optimization.md(索引结构与查询优化器)

**Files:**
- Create: `for_real_dummy/database-deep-dive/03-indexing-and-query-optimization.md`
- Modify: `00-roadmap.md`(第03行)

知识点范围(10个):B+树索引结构与磁盘IO复杂度分析(为什么是B+树不是二叉树/跳表)、聚簇索引vs非聚簇索引(InnoDB主键即聚簇索引 vs PostgreSQL堆表+索引)、联合索引最左前缀原则、覆盖索引避免回表、哈希索引适用场景与局限、EXPLAIN/EXPLAIN ANALYZE真实解读(双引擎输出格式对比)、基于代价的查询优化器基础(统计信息/选择率估算)。

- [ ] **Step 1: 设计知识点并验证核心结论**

B+树磁盘IO复杂度:`.venv` 纯算法模拟一棵B+树(阶数M,N条记录),计算树高 `ceil(log_M(N))`,和二叉树高度 `ceil(log_2(N))` 对比,`assert` 验证百万级记录时B+树高度(约3-4层)远小于二叉树(约20层),体现"为什么用高扇出的多叉树而不是二叉树"这个磁盘IO次数=树高的核心论据。聚簇 vs 非聚簇索引、EXPLAIN真实解读、最左前缀、覆盖索引、优化器基础:全部用 WSL2 双引擎真实验证(`python-wsl2`围栏)——建同样结构的表(带联合索引),分别在 PostgreSQL 和 MariaDB 上跑 `EXPLAIN`/`EXPLAIN ANALYZE`,真实捕获两个引擎的输出格式差异(PG 输出 `Index Scan using ...` + `cost=`,MariaDB/InnoDB 输出 `type: ref` + `key:` + `rows:`),`assert` 验证走联合索引的查询和不走索引的全表扫描在 `EXPLAIN` 输出里能被程序化区分(比如断言 PG 输出包含 `Index Scan` 而不是 `Seq Scan`,MariaDB 输出的 `type` 不是 `ALL`)。最左前缀:建 `(a,b,c)` 联合索引,验证 `WHERE a=? AND b=?` 走索引、`WHERE b=? AND c=?`(缺a)不走索引,双引擎对比。覆盖索引:验证 `EXPLAIN` 输出出现 `Index Only Scan`(PG)/`Extra: Using index`(MariaDB)时确实不需要回表。哈希索引:PostgreSQL 支持显式 `CREATE INDEX ... USING HASH`,验证等值查询可用、范围查询(`>`/`<`)不可用(优化器不选用哈希索引,退化为全表扫描)。

- [ ] **Step 2: 撰写完整 markdown**

每个知识点"可运行例子"步骤开头标注环境(纯算法部分`.venv`,EXPLAIN相关部分`python-wsl2`,依赖 Task 1 已验证的 PostgreSQL 16.14 + MariaDB 10.11.15)。

- [ ] **Step 3: 运行验证脚本(`.venv`部分用`_verify_md.py`,`python-wsl2`部分单独在WSL2跑并记录真实输出)**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第03行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/database-deep-dive/03-indexing-and-query-optimization.md for_real_dummy/database-deep-dive/00-roadmap.md
git commit -m "docs(database-deep-dive): 03类 索引结构与查询优化器(10点) - 板块II完成"
```

---

### Task 5: 04-transactions-and-isolation-levels.md(事务与隔离级别)

**Files:**
- Create: `for_real_dummy/database-deep-dive/04-transactions-and-isolation-levels.md`
- Modify: `00-roadmap.md`(第04行)

知识点范围(8个):ACID四性质逐条拆解、四种隔离级别(READ UNCOMMITTED/READ COMMITTED/REPEATABLE READ/SERIALIZABLE)、四种异常现象(脏读/不可重复读/幻读/丢失更新)在PostgreSQL和MariaDB上的真实触发对比。

- [ ] **Step 1: 设计知识点并验证核心结论**

全部 `python-wsl2` 围栏,用两个真实数据库连接(通过 `threading.Event` 做显式同步点,不用sleep)构造经典异常场景:①脏读——连接A `UPDATE`未提交,连接B在`READ UNCOMMITTED`下能读到未提交值(MariaDB支持该级别可真实复现;PostgreSQL把READ UNCOMMITTED实现等同于READ COMMITTED,如实标注这个双引擎差异,不强行凑出PG的脏读)。②不可重复读——连接A两次读同一行,中间连接B提交了修改,`READ COMMITTED`下A两次读到不同值,`REPEATABLE READ`下两次读到相同值(双引擎都验证)。③幻读——连接A两次范围查询,中间连接B插入了新行,验证`REPEATABLE READ`下:PostgreSQL(快照隔离,天然不出现幻读)vs MariaDB/InnoDB(靠间隙锁阻止幻读,同样不出现但机制不同——这个对比是本类最有教学价值的一点)。④丢失更新——两个连接同时读同一行的值、各自计算后写回,后写的覆盖先写的,验证在没有加锁/乐观锁版本号保护时丢失更新真实发生。每个场景至少重复5次确认稳定复现(时序用`threading.Event`严格控制,不依赖运气)。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本,时序敏感断言额外重复5次确认稳定**

```bash
cd for_real_dummy/database-deep-dive
for i in $(seq 1 5); do python _verify_md.py 04-transactions-and-isolation-levels.md; done
```

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第04行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/database-deep-dive/04-transactions-and-isolation-levels.md for_real_dummy/database-deep-dive/00-roadmap.md
git commit -m "docs(database-deep-dive): 04类 事务与隔离级别(8点) - 板块III启动"
```

---

### Task 6: 05-mvcc-and-locking.md(MVCC与锁机制)

**Files:**
- Create: `for_real_dummy/database-deep-dive/05-mvcc-and-locking.md`
- Modify: `00-roadmap.md`(第05行)

知识点范围(8个):MVCC快照隔离原理(PostgreSQL的xmin/xmax可见性判断 vs InnoDB的undo log版本链)、锁类型与粒度(行锁/表锁/意向锁)、InnoDB间隙锁与next-key lock、真实死锁复现与检测、乐观锁vs悲观锁工程选型。

**文件头部声明边界**:本类讲数据库引擎内部实现的行锁/表锁/间隙锁/next-key lock,死锁检测是数据库引擎自己维护的等待图。os-concurrency-deep-dive 03类(互斥锁/信号量)和05类(死锁四条件/检测算法/资源分配图)讲的是操作系统层面进程/线程同步与死锁理论,通用、和具体应用无关——两者的"死锁"是同一理论根源(等待图成环)但应用domain完全不同,不重新推导OS死锁检测算法本身,只引用其"等待图成环"的核心判据。

- [ ] **Step 1: 设计知识点并验证核心结论**

MVCC:`python-wsl2`,PostgreSQL 用 `SELECT xmin, xmax, * FROM t` 真实观察行版本号在UPDATE前后的变化(UPDATE产生新版本,旧版本xmax被标记);MariaDB/InnoDB 用 `SELECT trx_id FROM information_schema.innodb_trx` 配合未提交事务观察版本链存在。两者对比着写,不各写各的互不关联。锁粒度:PostgreSQL `SELECT ... FOR UPDATE` 真实验证行锁只锁住命中的行(另一个连接改别的行不阻塞,改同一行会阻塞,用`threading.Event`+超时验证阻塞真实发生)。间隙锁/next-key lock:MariaDB/InnoDB 专属机制(PostgreSQL没有间隙锁,用谓词锁实现类似效果,如实标注差异),构造 `REPEATABLE READ` 下 `SELECT ... FOR UPDATE` 一个范围,验证另一个连接往这个范围**内**插入新行被阻塞(间隙锁生效),验证范围**外**插入不受影响。真实死锁复现:两个连接故意以相反顺序锁两行(A锁1再等锁2,B锁2再等锁1),验证数据库引擎的死锁检测器真实介入,自动kill掉一个事务并返回死锁错误(PostgreSQL `deadlock detected` / MariaDB `Deadlock found`),demo必须带超时安全网(比如主线程等待两个连接线程最多10秒,超时也要能确定性退出并报告,不能真的死等)。乐观锁:用版本号列实现 `UPDATE ... WHERE version=?`,验证并发更新时后到的更新因版本号不匹配而影响0行,能被程序检测到并重试。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本,死锁/锁竞争类断言额外重复5次确认稳定**

```bash
cd for_real_dummy/database-deep-dive
for i in $(seq 1 5); do python _verify_md.py 05-mvcc-and-locking.md; done
```

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第05行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/database-deep-dive/05-mvcc-and-locking.md for_real_dummy/database-deep-dive/00-roadmap.md
git commit -m "docs(database-deep-dive): 05类 MVCC与锁机制(8点) - 板块III完成"
```

---

### Task 7: 06-storage-engine-internals.md(存储引擎内部机制)

**Files:**
- Create: `for_real_dummy/database-deep-dive/06-storage-engine-internals.md`
- Modify: `00-roadmap.md`(第06行)

知识点范围(9个):WAL(Write-Ahead Log)为什么能保证持久性、redo log与undo log分工、checkpoint机制与恢复时间权衡、崩溃恢复流程(ARIES三阶段思想简化版)、缓冲池与页面置换(交叉引用dsa-deep-dive 05类LRU缓存)、LSM-tree结构与compaction(对比B+树)。

- [ ] **Step 1: 设计知识点并验证核心结论**

WAL持久性:`python-wsl2`,PostgreSQL 用 `pg_current_wal_lsn()` 在事务提交前后观察 WAL 位置(LSN)真实推进,证明提交确实先写了WAL日志;检查 `/var/lib/pgsql/data/pg_wal/` 目录下真实存在WAL段文件。checkpoint:触发 `CHECKPOINT` 命令,观察 `pg_stat_bgwriter` 里 checkpoint 计数增加。缓冲池:交叉引用dsa-deep-dive 05类LRU缓存的链表+哈希表实现,不重新推导LRU本身,聚焦数据库特有的"脏页"维度——`.venv` 模拟一个"脏页感知"的缓冲池(区分干净页可以直接淘汰、脏页淘汰前必须先落盘),验证淘汰脏页时确实触发了"写回"这一步,和普通LRU相比多了这个约束。ARIES崩溃恢复:`.venv` 纯算法模拟简化版三阶段(分析阶段确定哪些事务在崩溃时未提交、重做阶段replay所有WAL记录、回滚阶段撤销未提交事务的修改),构造一个"崩溃"场景(模拟WAL日志+一份"崩溃时"的数据状态),验证恢复算法执行后数据状态等价于"所有已提交事务生效、未提交事务全部撤销"的正确终态。LSM-tree vs B+树:`.venv` 模拟LSM-tree的分层结构(内存memtable+多层SSTable)和简化compaction(合并多个SSTable消除重复key,保留最新版本),对比同样写入负载下B+树(原地更新,随机IO)和LSM-tree(顺序写memtable,IO模式完全不同)的写入路径差异,`assert` 验证compaction后重复key只保留一份最新值。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本(`.venv`部分 + WSL2真实WAL/checkpoint观察部分单独验证并记录)**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第06行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/database-deep-dive/06-storage-engine-internals.md for_real_dummy/database-deep-dive/00-roadmap.md
git commit -m "docs(database-deep-dive): 06类 存储引擎内部机制(9点) - 板块IV完成"
```

---

### Task 8: 07-replication-and-distributed-basics.md(复制与分布式基础)

**Files:**
- Create: `for_real_dummy/database-deep-dive/07-replication-and-distributed-basics.md`
- Modify: `00-roadmap.md`(第07行)

知识点范围(7个):主从复制原理与binlog/WAL传输、同步复制vs异步复制vs半同步复制的一致性权衡、CAP定理(用真实网络分区场景推导)、分库分表策略(垂直vs水平,水平分片交叉引用dsa-deep-dive 20类一致性哈希)、两阶段提交2PC点到为止。

- [ ] **Step 1: 设计知识点并验证核心结论**

主从复制:`python-wsl2`,MariaDB 相对容易在单机WSL2内真实搭建一主一从(两个不同 `datadir`+不同端口的 `mariadbd` 实例,或用 PostgreSQL 的 `pg_basebackup` + `standby.signal` 搭建物理复制),二选一实测(优先选相对更简单、文档更成熟的路径,实际搭建时确认可行性),真实观察主库写入后从库通过复制延迟(用 `SHOW SLAVE STATUS`/`pg_stat_replication` 里的延迟指标或轮询对比数据出现的时间差)最终一致地同步过去。异步复制的延迟窗口:主库写入后立刻查从库,验证短时间内可能读不到最新数据(真实复现复制延迟导致的读不一致,而不是断言"理论上会这样")。CAP定理:`.venv` 用两个模拟"节点"(各自维护本地状态字典)+一个可以人为切断的"网络"标志位,构造网络分区场景,验证分区期间选择"继续服务但可能不一致"(AP)vs"拒绝服务保证一致"(CP)两种策略的真实行为差异,不是背公式而是真实跑一遍状态机。分库分表:垂直拆分讲清楚"为什么"(不同业务表读写模式差异大,拆分后可以独立扩展),水平分片提及一致性哈希时写"详见 dsa-deep-dive/20-advanced-interview-depth.md 案例1,此处引用其结论:引入虚拟节点后节点负载标准差显著降低",不重新实现。2PC:概念性讲解协调者+参与者两阶段的消息流程,可以用`.venv`模拟一个简化的2PC状态机(prepare阶段全部participant确认后才commit,任一participant拒绝则全部abort)验证正确性,但明确标注"工程上很少直接用2PC(阻塞问题+协调者单点),深水区(Saga/TCC等替代方案)留给系统设计系列"。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本(WSL2复制部分单独验证并记录,包含真实复制延迟数据)**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第07行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/database-deep-dive/07-replication-and-distributed-basics.md for_real_dummy/database-deep-dive/00-roadmap.md
git commit -m "docs(database-deep-dive): 07类 复制与分布式基础(7点) - 板块V完成"
```

---

### Task 9: 08-nosql-and-caching-patterns.md(NoSQL与缓存模式)

**Files:**
- Create: `for_real_dummy/database-deep-dive/08-nosql-and-caching-patterns.md`
- Modify: `00-roadmap.md`(第08行)

知识点范围(9个):Redis/Valkey五种核心数据结构(string/hash/list/set/zset)与底层编码优化、持久化机制RDB vs AOF权衡、缓存模式(cache-aside/read-through/write-through/write-behind)真实对比、缓存一致性问题(双写不一致场景真实复现)、数据库连接池在数据库场景下的特殊性(交叉引用computer-networking-deep-dive 11类)。

**开篇如实说明**:本类使用 Valkey 8.0.9(Rocky Linux 10 因 Redis 2024 年许可证变更转向的 Linux Foundation 开源分支,协议完全兼容,`redis`-py 驱动可直接连接),行文统一用"Redis/Valkey"或直接用"Redis"指代这套数据结构/协议(业界目前仍普遍用"Redis"称呼这类KV存储,读者面试时对方大概率也说"Redis"),但环境声明和可运行例子里如实标注实际连接的是Valkey。

- [ ] **Step 1: 设计知识点并验证核心结论**

`python-wsl2`,连接 Task 1 已启动的 Valkey 实例。五种数据结构:分别用 string(`SET`/`GET`/`INCR`原子性验证)、hash(`HSET`/`HGETALL`)、list(`LPUSH`/`RPOP`验证队列语义)、set(`SADD`+集合运算`SINTER`/`SUNION`)、zset(`ZADD`+`ZRANGE`按分数排序验证)各写一个真实操作+assert。持久化:RDB(`BGSAVE`后验证rdb文件真实生成且mtime更新)vs AOF(开启`appendonly yes`后验证aof文件真实追加写入,行数随操作增加),对比两者"丢失最后N秒数据的风险"这一核心权衡(不需要真实模拟崩溃,通过对比两种机制的写入时机[RDB定期快照 vs AOF每条命令]讲清楚风险来源)。缓存模式:构造一个简化的"数据库"(dict模拟)+"缓存"(Valkey),分别实现cache-aside(应用先查缓存未命中再查库再写缓存)、write-through(写操作同时写库和缓存)、write-behind(写操作先写缓存,异步批量落库,真实验证有一个时间窗口内数据只在缓存里),四种模式的真实读写路径和命中率统计。缓存一致性:构造"先更新数据库再删缓存"vs"先删缓存再更新数据库"两种顺序,配合并发读请求,验证后者存在真实的不一致窗口(一个并发读请求在删缓存后、更新数据库前查询,把旧值重新写回缓存,造成脏缓存持续存在直到下次更新)。连接池:交叉引用computer-networking-deep-dive 11类连接池的阻塞/复用通用机制结论,本类只聚焦数据库连接池的特殊约束——写一个demo验证"归还连接前必须重置事务状态"(比如一个连接在归还前有未提交事务或改过的`SESSION`变量,不重置直接复用会污染下一个使用者,真实验证重置前后的行为差异)。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第08行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/database-deep-dive/08-nosql-and-caching-patterns.md for_real_dummy/database-deep-dive/00-roadmap.md
git commit -m "docs(database-deep-dive): 08类 NoSQL与缓存模式(9点) - 板块VI完成,全部8个分类文件完成"
```

---

### Task 10: 09-mock-interview-capstone.md(模拟终面capstone)

**Files:**
- Create: `for_real_dummy/database-deep-dive/09-mock-interview-capstone.md`
- Modify: `00-roadmap.md`(第09行)

**Interfaces:**
- Consumes: 01-08 类全部知识点编号(用于追问链里的 cross-reference,格式"NN类知识点M")

场景设定:"一次索引失效引发的慢查询雪崩,叠加隔离级别误用导致的幻读业务bug"双线索排查。结构参照 os-concurrency-deep-dive 12类capstone(候选人初版汇报→面试官多轮追问→最终诊断)。覆盖至少3条五轴追问链轴线,cross-reference至少5处不同类别知识点(比如03类索引/04类隔离级别/05类MVCC/06类缓冲池/08类缓存一致性)。

- [ ] **Step 1: 设计capstone叙事结构与验证场景**

`python-wsl2` 真实构造:①索引失效线索——一张表原本有联合索引,业务加了一个新的查询条件顺序不符合最左前缀,真实跑 `EXPLAIN` 验证从 `Index Scan` 退化成 `Seq Scan`,测量真实执行耗时差异(小数据量下差异可能不明显,如实标注重点是执行计划的定性变化不是绝对耗时)。②幻读业务bug线索——构造一个"库存扣减"场景,在`REPEATABLE READ`下两个并发事务基于范围查询结果做决策,真实复现由于快照隔离/间隙锁理解错误导致的超卖或漏判(双引擎场景可以选一个更容易复现的来讲,如实标注)。两条线索的真实数据(执行计划输出、并发时序日志)都要是脚本真实生成的,不是编的故事。

- [ ] **Step 2: 撰写完整 markdown(候选人初版汇报→多轮追问→最终诊断)**

- [ ] **Step 3: 运行验证脚本,并独立重跑至少3次确认稳定**

```bash
cd for_real_dummy/database-deep-dive
python _verify_md.py 09-mock-interview-capstone.md
python _verify_md.py 09-mock-interview-capstone.md
python _verify_md.py 09-mock-interview-capstone.md
```

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第09行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/database-deep-dive/09-mock-interview-capstone.md for_real_dummy/database-deep-dive/00-roadmap.md
git commit -m "docs(database-deep-dive): 09类 模拟终面capstone"
```

---

### Task 11: 全库自查回归 + README + memory 更新 + 最终提交

**Files:**
- Modify: `for_real_dummy/database-deep-dive/00-roadmap.md`(合计行)
- Modify: `for_real_dummy/README.md`
- Create: `C:\Users\ericp\.claude\projects\e--Workspace-dummy\memory\database-deep-dive-complete.md`
- Modify: `C:\Users\ericp\.claude\projects\e--Workspace-dummy\memory\MEMORY.md`
- Modify: `C:\Users\ericp\.claude\projects\e--Workspace-dummy\memory\feedback-function-by-function-teaching.md`

- [ ] **Step 1: 逐文件独立子进程重跑验证**

```bash
cd for_real_dummy/database-deep-dive
for f in 01-relational-model-and-sql-basics.md 02-normalization-and-advanced-sql.md 03-indexing-and-query-optimization.md 04-transactions-and-isolation-levels.md 05-mvcc-and-locking.md 06-storage-engine-internals.md 07-replication-and-distributed-basics.md 08-nosql-and-caching-patterns.md 09-mock-interview-capstone.md; do
  echo "=== $f ==="
  python _verify_md.py "$f"
done
```
Expected: 全部 PASS。

- [ ] **Step 2: 结构标记计数核对**

```bash
grep -c "^## [0-9]" for_real_dummy/database-deep-dive/*.md
```
逐文件核对实际知识点数和 roadmap 表格声明的数字是否一致,不一致则修正 roadmap 表格为精确值(参照此前系列"先估后核"先例)。

- [ ] **Step 3: 双引擎/时序敏感类断言额外复测稳定性**

```bash
for i in $(seq 1 5); do python _verify_md.py 04-transactions-and-isolation-levels.md && python _verify_md.py 05-mvcc-and-locking.md && python _verify_md.py 07-replication-and-distributed-basics.md; done
```

- [ ] **Step 4: 更新 `00-roadmap.md` 合计行为精确数字**

- [ ] **Step 5: 更新 `for_real_dummy/README.md`**

在"独立技能系列"表格新增一行(参照 computer-networking-deep-dive 行的格式),补一段说明本系列是"职业发展与需求四部曲"第3部、和 rhcsa-bash-deep-dive(无重叠)/dsa-deep-dive/os-concurrency-deep-dive/computer-networking-deep-dive 的边界、双真实引擎验证策略、firewalld loopback0 坑的发现与修复(真正定位到上一部遗留的"未解决"问题根因),目录树新增 `database-deep-dive/` 条目。

- [ ] **Step 6: 创建 memory 文件 `database-deep-dive-complete.md`**

frontmatter `type: project`,内容参照 `computer-networking-deep-dive-complete.md` 的结构:精确知识点数/板块文件数、七步模板(延续不变)、双真实引擎验证策略(为什么是这次的核心新决策)、firewalld loopback0 问题的完整根因与修复记录(这是对上一部遗留问题的真正解决,应重点记录方便未来WSL2相关工作参考)、Redis→Valkey生态变化的真实发现、和已有系列的边界、四部曲进度更新(操作系统与并发✅/计算机网络✅/数据库原理与实战✅,系统设计排队中)。

- [ ] **Step 7: 更新 `MEMORY.md` 索引**

新增一行指向 `database-deep-dive-complete.md`;同时检查 `computer-networking-deep-dive-complete.md` 的index行是否需要同步移除"后续排队数据库"字样(现在数据库也完成了)。

- [ ] **Step 8: 更新 `feedback-function-by-function-teaching.md`**

追加一段简短说明:四部曲第3部完成,后续排队项明确(仅剩系统设计),并记录firewalld根因定位这个技术亮点(体现"如实排查而不是绕过问题"的一贯纪律又一次真实兑现)。

- [ ] **Step 9: 最终提交(确认 `git status`/`git diff` 只涉及本系列相关文件)**

```bash
git status --short for_real_dummy/database-deep-dive/ for_real_dummy/README.md
git add for_real_dummy/database-deep-dive/00-roadmap.md for_real_dummy/README.md
git commit -m "docs(database-deep-dive): 全库自查回归 + README集成 + 收尾提交"
```

---

*创建:2026-07-14*
