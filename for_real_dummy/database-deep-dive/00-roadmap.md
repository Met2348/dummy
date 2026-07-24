# 数据库原理与实战深挖 —— 路线图与进度表

> 目标:约 65 个数据库知识点,由浅入深,关系数据库加重(分布式数据库与NoSQL点到面试常考程度),深度对标 [os-concurrency-deep-dive/](../os-concurrency-deep-dive/00-roadmap.md)(79点)/[computer-networking-deep-dive/](../computer-networking-deep-dive/00-roadmap.md)(80点)(面试二三面深度,不是"这个概念是什么"的教科书搬运)。
> 定位:仓库"职业发展与需求"四部曲第 3 部——操作系统与并发(已完成,79点)→ 计算机网络(已完成,80点)→ 数据库原理与实战 → 系统设计。
> 设计文档:[docs/superpowers/specs/2026-07-14-database-deep-dive-design.md](../../docs/superpowers/specs/2026-07-14-database-deep-dive-design.md)、实施计划:[docs/superpowers/plans/2026-07-14-database-deep-dive.md](../../docs/superpowers/plans/2026-07-14-database-deep-dive.md)。

---

## 与仓库已有内容的边界(已核实,不重复)

- **vs rhcsa-bash-deep-dive**:已用 Grep 核实全部 10 个文件,完全没有数据库安装配置/运维内容——和前两部不同,本系列在这个方向上是全新领域,不需要专门划边界。
- **vs dsa-deep-dive**:08类(树)完全没有B+树(磁盘导向的多叉平衡树),本系列03类放心从零讲。05类已有LRU缓存knowledge point,本系列06类存储引擎缓冲池章节交叉引用其淘汰策略,只聚焦"脏页/WAL先行"这类数据库特有约束。20类进阶深度追加案例1已有真实一致性哈希负载均衡实测代码,本系列07类分库分表提及水平分片时交叉引用。
- **vs os-concurrency-deep-dive**:03类同步原语(互斥锁/信号量)和05类死锁讲的是操作系统层面进程/线程同步与死锁理论,通用、和具体应用无关。本系列05类"MVCC与锁机制"讲的是数据库引擎内部实现的行锁/表锁/间隙锁/next-key lock,两者的"死锁"是同一理论根源(等待图成环)但应用domain完全不同,不重新推导OS死锁检测算法本身。
- **vs computer-networking-deep-dive**:11类已有连接池的通用机制知识点。本系列08类提到数据库连接池时交叉引用,只聚焦"数据库连接是有状态事务边界"这个数据库特有约束。

## 知识点结构模板(七步,延续前两部模板不变)

1. **签名/是什么** 2. **一句话** 3. **底层机制/为什么这样设计**(B+树磁盘IO复杂度推导、ARIES恢复算法思想、MVCC可见性判断规则、CAP定理推导思路——直接写在这一步,不单独拆"数学推导"步骤) 4. **AI研究/工程场景** 5. **可运行例子**(assert验证,**必须显式标注验证环境与引擎**:`.venv`(sqlite3)或 `WSL2 Rocky Linux`(PostgreSQL 16.14 / MariaDB 10.11.15 / Valkey 8.0.9)) 6. **面试怎么问+追问链**(五轴方法论,见下表) 7. **常见坑**

## 五轴追问链方法论(从第一天融入,不事后补)

| 轴线 | 含义 | 数据库学科例子 |
|------|------|------|
| 规模递增轴 | 小规模→大规模→极限行为 | 单表百万行索引优化→千万行分表→亿级数据分布式分片 |
| 工程约束递增轴 | 单机→跨机→全球分发 | 单机事务→主从复制读写分离→跨地域多活(为系统设计系列埋伏笔) |
| 方案批判迭代轴 | 面试官连续指出具体缺陷逼换方案 | 全表扫描去重→加唯一索引仍慢→分区表→分区键选择错误导致热点→复合分区键 |
| 决策依据追问轴 | 不纠错,只逼问选择依据 | "为什么这里用REPEATABLE READ不用SERIALIZABLE","为什么选乐观锁不选悲观锁" |
| 真实性验证轴 | 简历"优化了慢查询"被追问到具体数字 | "具体加了哪个索引,EXPLAIN执行计划改动前后分别是什么,扫描行数从多少降到多少" |
| 诊断真实数据(新题型) | 给真实EXPLAIN输出/慢查询日志,要求诊断而非套公式 | 给一段真实EXPLAIN ANALYZE输出,要求判断走没走索引、瓶颈在排序还是回表 |

每个知识点挑 1~2 条最自然的轴线走 2~3 层深,不强行凑满 5 轴。09 类模拟终面 capstone 是唯一要求同时用 3 条以上轴线的文件。

## 环境声明:双真实引擎验证策略(本系列核心新决策,不同于前两部)

- **默认环境**:仓库根目录 `.venv`(Windows 原生 Python),stdlib `sqlite3` 覆盖:关系模型/DDL/DML/JOIN/范式反例/聚合与窗口函数/CTE、B+树磁盘IO次数纯算法模拟、LSM-tree compaction纯算法模拟、ARIES崩溃恢复纯算法模拟。
- **WSL2 Rocky Linux 10**(`Python 3.12.13`,复用 rhcsa-bash-deep-dive/os-concurrency-deep-dive/computer-networking-deep-dive 已装好的系统,2026-07-14 新装并验证):
  - **PostgreSQL 16.14**:`dnf install postgresql-server postgresql-contrib` + `postgresql-setup --initdb` + `systemctl enable --now postgresql`。`pg_hba.conf` 已把 `127.0.0.1/32` 规则从默认 `ident` 改为 `scram-sha-256` 允许密码认证。连接凭据:`host=127.0.0.1 port=5432 dbname=dbdemo user=dbdemo password=dbdemo_local_only`。
  - **MariaDB 10.11.15**:`dnf install mariadb-server` + `systemctl enable --now mariadb`。连接凭据:`host=127.0.0.1 port=3306 database=dbdemo user=dbdemo password=dbdemo_local_only`。
  - **Valkey 8.0.9**(Redis协议兼容分支):`dnf install epel-release` + `dnf install valkey` + `systemctl enable --now valkey`。**如实记录**:Rocky Linux 10 的 dnf 仓库没有 `redis` 包——2024年 Redis Inc. 把许可证改为非OSI认证的SSPL/RSALv2后,RHEL系发行版转向 Linux Foundation 主导的开源分支 Valkey(协议完全兼容,`redis`-py 驱动可直接连接)。连接:`host=127.0.0.1 port=6379`,无密码(本地回环+trusted zone,纯学习环境有意不设`requirepass`)。
  - **Python驱动**:专用虚拟环境 `~/database-deep-dive-venv`(不装进系统python3——Rocky Linux 10 系统Python是PEP668 externally-managed,且默认连pip模块都没装),内装 `psycopg2-binary 2.9.12` / `PyMySQL 1.2.0` / `redis 8.0.1`,已端到端验证三个驱动都能真实连接对应引擎并完成真实读写操作。**这是本系列相对前两部"标准库优先不新增依赖"纪律的必要修订**——连接真实数据库协议标准库没有对应实现。
  - **firewalld 环境坑(已修复,且发现会反复复发,已加固)**:WSL2 镜像网络模式下会出现一个叫 `loopback0` 的合成网卡(NetworkManager 连接名 `Wired connection 5`,不是内核真正的`lo`),firewalld 默认把它划进限制性的 `public` zone,导致所有到 `127.0.0.1` 的 TCP 连接报 `[Errno 113] No route to host`——这正是 computer-networking-deep-dive 系列遗留的"未解决"同名问题,这次真正定位到根因并修复:`nmcli connection modify 'Wired connection 5' connection.zone trusted` + `firewall-cmd --reload`。**真实撰写07类过程中这个问题反复复发了4次**(推测和 NetworkManager 在网络接口状态变化时[比如启动新的mariadbd进程監聽新端口]重新协商/reset连接属性有关,具体触发条件未完全定位),每次都要重新执行一遍上述命令很低效,最终改用更稳固的**基于源地址而不是接口**的规则:`firewall-cmd --permanent --zone=trusted --add-source=127.0.0.0/8` + `firewall-cmd --reload`——这条规则信任"任何来自127.0.0.0/8的流量",不依赖`loopback0`这个接口具体被分到哪个zone,加固后在07类剩余的测试里没有再复发。`public` zone 对 `eth1/eth2/eth3` 的限制未受影响,不破坏 rhcsa-bash-deep-dive 08类 firewalld 教学内容的有效性。**如果后续任务里再次出现`No route to host`,先确认这条`--add-source=127.0.0.0/8`规则是否还在(`firewall-cmd --zone=trusted --list-sources`),不在的话重新加一遍。**
- 每个知识点的"可运行例子"步骤开头**必须**显式标注验证环境与引擎,涉及真实数据库连接的例子一律用 `python-wsl2` 围栏。

## 数据库类特有验证纪律

- 隔离级别异常复现涉及两个真实数据库连接的时序控制,必须用显式同步点(`threading.Event`,不能用`sleep`猜时间)控制两个连接的操作顺序,并至少重复5~10次确认现象稳定复现。
- **版本锁定**:断言涉及具体EXPLAIN输出格式/系统表结构时注明引擎版本(PostgreSQL 16.14 / MariaDB 10.11.15),避免未来版本升级后断言失效被误判为代码错误。
- **不连接外部/生产数据库**:所有实例监听本地(Unix socket或127.0.0.1),测试数据全部是脚本内合成的最小数据集,不使用真实业务数据。
- **需要真实构造死锁/长事务阻塞的 demo 必须带确定性退出的超时安全网**,延续 os-concurrency-deep-dive 死锁 demo 纪律。

## 进度表

| # | 板块 | 分类 | 文件 | 知识点数(约) | 环境 | 状态 |
|---|------|------|------|------------|------|------|
| 01 | I 关系模型与SQL基础 | 关系模型与基础SQL | [01-relational-model-and-sql-basics.md](01-relational-model-and-sql-basics.md) | 7 | .venv | ✅ 已完成(7个`.venv`代码块全部通过) |
| 02 | I | 范式与进阶SQL | [02-normalization-and-advanced-sql.md](02-normalization-and-advanced-sql.md) | 7 | .venv | ✅ 已完成(7个`.venv`代码块全部通过,板块I完成) |
| 03 | II 索引与查询优化 | 索引结构与查询优化器 | [03-indexing-and-query-optimization.md](03-indexing-and-query-optimization.md) | 10 | .venv+WSL2双引擎 | ✅ 已完成(1个`.venv`代码块+9个`python-wsl2`代码块,双引擎真实EXPLAIN对比,板块II完成。真实发现:①`Index Only Scan`需要`VACUUM`更新可见性图,只`ANALYZE`不够;②C.UTF-8 collation下默认操作符类依然不支持前缀LIKE索引,需显式`text_pattern_ops`;③PG和MariaDB放弃索引改用全表扫描的选择性交叉点不同[PG约20~50%,MariaDB约5~20%]) |
| 04 | III 事务与并发控制 | 事务与隔离级别 | [04-transactions-and-isolation-levels.md](04-transactions-and-isolation-levels.md) | 8 | WSL2双引擎 | ✅ 已完成(8个`python-wsl2`代码块,双引擎并发时序真实验证,板块III启动。核心发现:①PG的`READ UNCOMMITTED`实际不脏读,MariaDB会;②PG的`REPEATABLE READ`能阻止丢失更新[SerializationFailure],MariaDB要到`SERIALIZABLE`才能[且报Deadlock不是序列化失败];③`SELECT FOR UPDATE`下MariaDB间隙锁真实阻塞并发INSERT约1.5秒,PG不阻塞;④MariaDB系统变量名是`tx_isolation`不是MySQL8的`transaction_isolation`) |
| 05 | III | MVCC与锁机制 | [05-mvcc-and-locking.md](05-mvcc-and-locking.md) | 8 | WSL2双引擎 | ✅ 已完成(8个`python-wsl2`代码块,双引擎真实验证,板块III完成。核心发现:①PG的xmin真实随UPDATE前进,InnoDB原地改+undo log重建旧版本;②行锁粒度真实验证[不同行不阻塞,同一行阻塞~1s];③next-key lock真实锁定范围比教科书定义更宽[目标值前后两侧间隙都被锁]；④真实死锁复现,双引擎victim选择不固定;⑤固定顺序加锁真实避免死锁;⑥乐观锁CAS真实防丢失更新;⑦锁等待超时双引擎真实触发) |
| 06 | IV 存储引擎内部机制 | 存储引擎内部机制 | [06-storage-engine-internals.md](06-storage-engine-internals.md) | 9 | .venv+WSL2 | ✅ 已完成(7个`.venv`代码块+2个`python-wsl2`代码块,板块IV完成。亮点:真实`kill -9`崩溃PostgreSQL主进程再重启,真实捕获"database system was not properly shut down"+"redo starts/redo done"恢复日志,验证已提交数据完整存活;ARIES三阶段/LSM-tree结构与compaction/脏页LRU用`.venv`模拟;SQLite WAL模式与PG WAL理念对比) |
| 07 | V 分布式数据库与复制 | 复制与分布式基础 | [07-replication-and-distributed-basics.md](07-replication-and-distributed-basics.md) | 7 | WSL2 MariaDB主从(二选一,理由见文件头) | ✅ 已完成(4个`python-wsl2`代码块+3个`.venv`代码块,板块V完成。亮点:WSL2内真实搭建MariaDB一主一从[独立datadir/端口3307/GTID复制],真实测量复制延迟[亚毫秒级],真实模拟异步复制滞后窗口[STOP/START SLAVE SQL_THREAD],真实模拟网络分区演示CAP的AP/CP权衡[STOP/START SLAVE IO_THREAD];垂直分片/水平分片痛点[朴素取模82.9%迁移率]/2PC状态机用`.venv`) |
| 08 | VI NoSQL与现代工程场景 | NoSQL与缓存模式 | [08-nosql-and-caching-patterns.md](08-nosql-and-caching-patterns.md) | 9 | WSL2 Valkey | ✅ 已完成(9个`python-wsl2`代码块,板块VI完成,8个分类文件全部完成。亮点:真实RDB/AOF持久化文件验证[含现代多部分AOF格式发现];用`threading.Event`精确构造真实的缓存双写不一致竞态;缓存击穿[20→1]/雪崩[1→30个过期时间点]/穿透[10→1]三种场景+真实缓解手段;数据库连接池会话状态泄漏真实复现) |
| 09 | 收尾 | 模拟终面capstone | [09-mock-interview-capstone.md](09-mock-interview-capstone.md) | —(不计入合计) | WSL2 MariaDB | ✅ 已完成(1个综合`python-wsl2`代码块,双线索——索引最左前缀违反导致的慢查询雪崩[03类知识点3] + REPEATABLE READ误用导致的优惠券超发[04类知识点7]——同一份代码里独立验证,覆盖5轴追问链中4条轴线,cross-reference 5个不同分类文件。**验证阶段真实踩坑**:超发demo最初5个线程各自独立`connect→SELECT→UPDATE`没有显式同步点,连接建立耗时不均导致偶发`got 3`而非预期`5`[后面的线程慢到看见前面线程已提交的新值]——本质是用"起5个线程"冒充"5个人同时查询",时序噪声掩盖了真实bug。用`threading.Barrier(5)`强制全部5个连接都完成SELECT建立好REPEATABLE READ快照后才放行到UPDATE阶段,修复后连续5次重跑稳定复现5/5超发、修复后1/5正确。) |

| 10 | 收尾 | 手把手实战:从零搭一个迷你KV存储引擎 | [10-build-a-mini-kv-store.md](10-build-a-mini-kv-store.md) | 4阶段(不计入合计) | ✅ 已完成(5个`.venv`代码块独立通过;串联06类知识点1 WAL"先写日志"规则、06类知识点2/4 redo分工与ARIES三阶段恢复思想、04类知识点1事务原子性,四阶段从朴素dict封装到WAL持久化到崩溃重放到损坏记录识别,只用标准库文件I/O,不连真实PostgreSQL/MariaDB引擎。呼应dsa-deep-dive/21结尾"持久化存储属于database-deep-dive范畴"这句话,把那个方向真正做出来) |

**最终合计:65 个知识点(7+7+10+8+8+9+7+9),8 个分类文件 + 1 篇模拟终面 capstone + 1 篇教程体(4 阶段,不计入合计)。** 精确数字以全库自查阶段的逐文件核对为准(见 Task 11)。

**关于 10 类的方法论说明:** "教程体"格式最早在 [dsa-deep-dive/21](../dsa-deep-dive/21-build-a-mini-search-engine.md) 试点,这是该格式在 dsa-deep-dive 之外的第二次落地——这次额外验证了一件事:教程体不必局限于串同一个知识点文件内部的内容,只要两个知识点之间有真实的机制关联(这里是"WAL 保证持久性"和"事务保证原子性"共同构成数据库对崩溃场景的完整承诺),跨文件组装出的小工具依然站得住,不是生硬拼接。是否继续推广到本系列其余知识点或其他系列,留给后续单独决定。

---

## 验证纪律

- 验证脚本 `_verify_md.py`(regex 提取 ` ```python ` 代码块,每块独立 subprocess 执行;另支持 ` ```python-wsl2 ` 标记只统计不执行)直接拷贝自 `computer-networking-deep-dive/_verify_md.py`,不重新设计。
- 数学/机制结论必须用 `assert` 数值验证,不能只摆公式描述。
- print() 语句必须纯 ASCII(历史教训:Windows GBK locale 下 `_verify_md.py` 子进程 reader 线程遇到非 ASCII 字符会 UnicodeDecodeError);正文 markdown 数学/协议符号不受此限制。
- Windows端 `.venv` 不新增任何第三方依赖(sqlite3已足够);WSL2端为连接真实数据库协议新增 `psycopg2-binary`/`PyMySQL`/`redis` 三个依赖,装在专用虚拟环境 `~/database-deep-dive-venv`,不装进系统python3。
- 设计文档与实施计划见上方链接。

---

*创建:2026-07-14*
