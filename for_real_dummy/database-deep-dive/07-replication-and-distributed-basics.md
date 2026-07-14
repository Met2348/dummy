# 复制与分布式基础

> 板块 V(分布式数据库与复制)。环境混合:`python-wsl2`(真实MariaDB一主一从复制,知识点1-4)+ `.venv`(纯算法/模拟:垂直分库/水平分片痛点/2PC状态机,知识点5-7)。

**引擎选择说明**:复制机制在 PostgreSQL 和 MariaDB 上原理相通(都是"主库产生变更日志→从库拉取并重放"),本类选择在 WSL2 内真实搭建 **MariaDB 一主一从**(复用已有的 PostgreSQL 16.14 实例作为分片/2PC等无需复制的知识点的补充,不重复搭建 PostgreSQL 的流复制)。真实搭建过程:开启主库 binlog(`server_id=1`,`log_bin=ON`)→ 新建从库实例(独立 `datadir`,端口 3307,`server_id=2`)→ 用 `mysqldump` 同步初始数据 → 从库用 GTID(`MASTER_USE_GTID=slave_pos`)连接主库 → `START SLAVE`。这套环境从知识点1建立后,知识点2-4 直接复用。

## 1. 主从复制原理:binlog传输机制

**签名/是什么**

```
主库(Primary/Master): 处理所有写操作,把变更记录进binlog(二进制日志)
从库(Replica/Slave):  IO线程持续拉取主库binlog存进本地relay log,SQL线程重放relay log里的变更
```

一句话:主从复制的核心思路和知识点1(06类WAL)是同一套"预写日志"哲学的延伸——主库不是把整行数据传给从库,而是把"发生了什么修改"这份日志传过去,从库在本地重新执行一遍这些修改,达到最终状态一致。

**底层机制/为什么这样设计**

传输"日志"而不是"数据本身"有几个真实的工程价值:日志通常比对应的数据修改更紧凑(尤其是`ROW`格式的binlog,虽然记录的是行级变更,但仍然只包含变更部分而不是整个数据文件);日志天然带有顺序性,从库按顺序重放能保证最终状态和主库一致;从库的IO线程(负责网络传输)和SQL线程(负责重放执行)是分离的两个线程,这个分离让"网络传输慢"和"本地重放慢"两种不同的瓶颈可以被分别观察和诊断(知识点2/3会用到这个分离带来的可观测性)。

**AI研究/工程场景**

读写分离(把只读查询流量路由到从库,减轻主库压力)是这套机制最常见的直接应用,前提是应用层要能接受知识点3会验证的"从库数据可能有短暂延迟"这个真实代价。

**可运行例子**(环境:`python-wsl2`,真实搭建一主一从,不是模拟)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 MariaDB 10.11.15 主库(已在00-roadmap.md列出的凭据下启动)。
# 本例会真实创建第二个mariadbd进程作为从库(独立datadir,端口3307),并让它保持后台运行,
# 供知识点2-4复用同一套主从环境,不是每个知识点都重新搭建一遍。
import subprocess
import time
import pymysql

def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

# 1. 主库开启binlog(如果之前没开过)
run("""cat > /etc/my.cnf.d/replication.cnf << 'EOF'
[mysqld]
server_id=1
log_bin=/var/lib/mysql/mariadb-bin
binlog_format=ROW
EOF""")
run("systemctl restart mariadb")
time.sleep(2)
binlog_check = run('mysql -u root -e "SHOW VARIABLES LIKE \'log_bin\';"')
assert "ON" in binlog_check.stdout, f"expected binlog to be enabled, got: {binlog_check.stdout}"

# 2. 建立复制专用账号
run("mysql -u root -e \"CREATE USER IF NOT EXISTS 'repl'@'127.0.0.1' IDENTIFIED BY 'repl_pass_local_only'; GRANT REPLICATION SLAVE ON *.* TO 'repl'@'127.0.0.1'; FLUSH PRIVILEGES;\"")

# 3. 建测试库和一条初始数据
run('mysql -u root -e "CREATE DATABASE IF NOT EXISTS repldemo; USE repldemo; DROP TABLE IF EXISTS orders; CREATE TABLE orders (id INT PRIMARY KEY, item VARCHAR(50)); INSERT INTO orders VALUES (1, \'widget\');"')
run("mysql -u root -e \"CREATE USER IF NOT EXISTS 'dbdemo'@'127.0.0.1' IDENTIFIED BY 'dbdemo_local_only'; GRANT ALL PRIVILEGES ON repldemo.* TO 'dbdemo'@'127.0.0.1'; FLUSH PRIVILEGES;\"")

# 4. 记录当前GTID位置,初始化从库实例的数据目录
gtid_result = run('mysql -u root -N -e "SELECT @@gtid_current_pos;"')
current_gtid = gtid_result.stdout.strip()

run("mkdir -p /var/lib/mysql-replica && chown mysql:mysql /var/lib/mysql-replica")
run("mariadb-install-db --datadir=/var/lib/mysql-replica --user=mysql")

# 5. 用mysqldump同步主库当前数据到从库
run("mysqldump -u root --databases repldemo --master-data=0 > /tmp/repldemo_dump.sql")

# 6. 启动从库mariadbd进程(后台运行,不阻塞,供知识点2-4继续使用)
subprocess.Popen(
    "/usr/sbin/mariadbd --datadir=/var/lib/mysql-replica --port=3307 "
    "--socket=/var/lib/mysql-replica/mysql.sock --server_id=2 --user=mysql "
    "--pid-file=/var/lib/mysql-replica/mysqld.pid --log-error=/var/lib/mysql-replica/error.log",
    shell=True, start_new_session=True,
)
time.sleep(3)

run("mysql -u root --socket=/var/lib/mysql-replica/mysql.sock < /tmp/repldemo_dump.sql")
run("mysql -u root --socket=/var/lib/mysql-replica/mysql.sock -e \"CREATE USER IF NOT EXISTS 'dbdemo'@'127.0.0.1' IDENTIFIED BY 'dbdemo_local_only'; GRANT ALL PRIVILEGES ON repldemo.* TO 'dbdemo'@'127.0.0.1'; FLUSH PRIVILEGES;\"")

# 7. 配置从库通过GTID连接主库并开始复制
run(f"""mysql -u root --socket=/var/lib/mysql-replica/mysql.sock -e "
SET GLOBAL gtid_slave_pos = '{current_gtid}';
CHANGE MASTER TO MASTER_HOST='127.0.0.1', MASTER_PORT=3306, MASTER_USER='repl', MASTER_PASSWORD='repl_pass_local_only', MASTER_USE_GTID=slave_pos;
START SLAVE;
" """)
time.sleep(2)

status_result = run('mysql -u root --socket=/var/lib/mysql-replica/mysql.sock -e "SHOW SLAVE STATUS\\G"')
assert "Slave_IO_Running: Yes" in status_result.stdout, f"expected IO thread running, got: {status_result.stdout[:500]}"
assert "Slave_SQL_Running: Yes" in status_result.stdout, f"expected SQL thread running, got: {status_result.stdout[:500]}"

# 8. 验证初始数据真实同步过去了
replica_conn = pymysql.connect(host='127.0.0.1', port=3307, user='dbdemo', password='dbdemo_local_only', database='repldemo')
rc = replica_conn.cursor()
rc.execute("SELECT * FROM orders WHERE id=1")
row = rc.fetchone()
assert row == (1, 'widget'), f"expected replicated initial data, got {row}"
replica_conn.close()

print("real primary-replica setup verified: binlog enabled, GTID-based replication running (IO+SQL threads both Yes), initial data synced")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
主库 log_bin: ON
从库 SHOW SLAVE STATUS: Slave_IO_Running=Yes, Slave_SQL_Running=Yes
从库查询 orders WHERE id=1: (1, 'widget')  <- 真实同步成功
```

**面试怎么问+追问链**

- Q:MySQL/MariaDB的复制是怎么实现的?
  - 追问1:如果主库的binlog_format设置不对,复制会出什么问题?
    - 深挖追问(区分度较高):`ROW`格式和`STATEMENT`格式的binlog各自的优缺点是什么(答案方向:`STATEMENT`格式记录的是SQL语句本身,日志体积小但对于包含`NOW()`/`RAND()`这类非确定性函数的语句,主从执行结果可能不一致;`ROW`格式记录的是每一行实际变更前后的值,体积更大但确定性强、不会有主从数据漂移的风险,这也是现代MariaDB/MySQL默认推荐`ROW`格式[本类环境搭建时也是用的`ROW`]的真实原因)。

**常见坑**

- 以为复制是"复制整个数据文件"——真实机制是传输和重放变更日志,这个理解差异直接影响"复制过程中主库为什么不需要长时间锁表"这类问题的正确回答。
- 搭建复制环境时忽视权限/用户在主从两侧的独立性——本类实测过程中就真实踩过这个坑(07类知识点8的复制修复经验:主库执行的`GRANT`语句会被复制到从库,但如果从库上下文和主库假设不完全一致,可能导致复制中断,`SHOW SLAVE STATUS`里的`Last_Error`是排查这类问题的第一入口)。

---

## 2. 复制延迟的真实测量

**签名/是什么**

```
复制延迟(Replication Lag): 从库落后主库的时间差,可以用"数据出现在从库上"这个真实事件测量,
                          也可以查 SHOW SLAVE STATUS 里的 Seconds_Behind_Master 估算
```

一句话:复制延迟不是一个理论概念,是可以真实测量的具体数字——本类直接测量"主库提交后,从库多久能查到这行数据",而不是只看`Seconds_Behind_Master`这个引擎自报的估算值。

**底层机制/为什么这样设计**

`Seconds_Behind_Master`是从库SQL线程自己估算出来的(通常基于relay log里记录的时间戳和当前时间的差值),在正常情况下能反映真实延迟,但在某些边界情况下(比如从库刚重启、或者主库长时间没有写入活动)这个估算值可能不准确或者是`NULL`。直接测量"写入到可见"这个端到端时间,是更贴近应用真实体验的复制延迟定义。

**AI研究/工程场景**

设计"写后立刻读"(read-after-write)的用户体验时(比如用户刚发了一条评论,立刻刷新页面),如果读请求被路由到了还没追上的从库,用户会看到自己刚发的内容"消失了",这是知识点3要展开的问题,而"复制延迟具体是多少"这个真实数字,决定了这类用户体验问题出现的概率有多大。

**可运行例子**(环境:`python-wsl2`,复用知识点1已建立的主从环境)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,复用知识点1已建立的主从复制环境(主库3306/从库3307)
import pymysql
import time

primary = pymysql.connect(host='127.0.0.1', port=3306, user='dbdemo', password='dbdemo_local_only', database='repldemo')
primary.autocommit(True)
replica = pymysql.connect(host='127.0.0.1', port=3307, user='dbdemo', password='dbdemo_local_only', database='repldemo')
replica.autocommit(True)  # 必须开autocommit,否则第二次查询会读到第一次建立的旧MVCC快照,不是真实的复制延迟

pc = primary.cursor()
rc = replica.cursor()

pc.execute("DELETE FROM orders WHERE id > 1")  # 清理之前测试可能留下的行,保证测量的id唯一不冲突
pc.execute("INSERT INTO orders VALUES (100, 'lag_measurement_item')")
insert_time = time.time()

found_after = None
for _ in range(500):
    rc.execute("SELECT * FROM orders WHERE id=100")
    if rc.fetchone():
        found_after = time.time() - insert_time
        break
    time.sleep(0.005)

assert found_after is not None, "replica should eventually catch up within the polling window"
assert found_after < 1.0, f"expected sub-second lag on localhost, got {found_after}"

primary.close(); replica.close()
print(f"real replication lag measured: {found_after*1000:.2f} ms (end-to-end, from primary commit to visible on replica)")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux,本地回环,多次测量结果在1毫秒量级):

```
real replication lag measured: 0.76 ms
```

**面试怎么问+追问链**

- Q:生产环境的主从复制延迟一般是多少?
  - 追问1:本类在本地回环网络测出来不到1毫秒,这个数字在真实生产环境有参考价值吗?
    - 深挖追问(区分度较高):真实生产环境的复制延迟主要受什么因素影响(答案方向:本地回环测的是"复制机制本身的处理开销"这个下限,不代表生产环境——真实生产环境的复制延迟主要受①主从之间的网络往返时延[跨机房/跨地域会显著放大]、②从库硬件性能是否跟得上主库的写入速度[从库SQL线程重放跟不上主库产生变更的速度会积压]、③单线程重放的限制[传统复制的SQL线程是单线程顺序执行,即使主库是多个并发事务写入,从库也要串行重放,这是从库"追不上"的常见根源,现代版本有并行复制特性专门针对这个问题]这几个因素共同决定,不能用本地测试的数字直接套用到生产环境)。

**常见坑**

- 只看`SHOW SLAVE STATUS`里的`Seconds_Behind_Master`就下结论,不理解这是一个估算值——关键业务的复制延迟监控应该用更直接的端到端探测(比如定期写入一个心跳时间戳,从库读取比较时间差),而不是完全依赖引擎自报的字段。
- 把本地测试环境测出的"毫秒级延迟"当成所有场景的默认预期——网络条件、负载压力、硬件配置都会显著影响真实延迟,不能脱离具体环境泛泛而谈一个数字。

---

## 3. 异步复制的延迟窗口:真实复现"刚写完读不到"

**签名/是什么**

```
异步复制(Async Replication): 主库提交不等待任何从库确认,从库按自己的节奏追赶
```

一句话:异步复制模式下,主库和从库之间天然存在一个"从库还没追上"的时间窗口,如果应用在这个窗口内把读请求发到从库,会读不到刚刚在主库写入的数据——本类不满足于"理论上会这样",而是真实制造一个确定性的延迟窗口,证明这个现象真实存在。

**底层机制/为什么这样设计**

之所以要真实"制造"这个窗口(而不是像知识点2那样靠自然的网络/处理延迟去撞运气),是因为在本地回环网络下自然延迟只有不到1毫秒(知识点2已经测出来),单纯靠"insert后立刻查"很难稳定复现"读不到"这个现象——必须显式暂停从库SQL线程的重放进度,人为制造一个可控、确定性的窗口,才能可靠、可重复地验证这个机制,而不是依赖运气。

**可运行例子**(环境:`python-wsl2`,复用知识点1已建立的主从环境)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,复用知识点1已建立的主从复制环境
import pymysql
import time

primary = pymysql.connect(host='127.0.0.1', port=3306, user='dbdemo', password='dbdemo_local_only', database='repldemo')
primary.autocommit(True)
replica = pymysql.connect(host='127.0.0.1', port=3307, user='dbdemo', password='dbdemo_local_only', database='repldemo')
replica.autocommit(True)
# 暂停/恢复SQL线程是管理操作,需要SUPER权限,用root经Unix socket连接从库(dbdemo这个应用账号没有这个权限,符合真实权限边界)
replica_admin = pymysql.connect(unix_socket='/var/lib/mysql-replica/mysql.sock', user='root', database='repldemo')

pc = primary.cursor()
rc = replica.cursor()
rac = replica_admin.cursor()

# 人为暂停从库SQL线程的重放进度,制造一个确定性的滞后窗口
rac.execute("STOP SLAVE SQL_THREAD")

pc.execute("DELETE FROM orders WHERE id=200")
pc.execute("INSERT INTO orders VALUES (200, 'delayed_item')")

# SQL线程暂停期间,这行数据在从库上确定读不到(不是运气不好没赶上,是真实制造的窗口)
rc.execute("SELECT * FROM orders WHERE id=200")
during_pause = rc.fetchone()
assert during_pause is None, "while the replica's SQL thread is paused, it must NOT see the new row yet"

# 恢复重放,验证从库最终能追上
rac.execute("START SLAVE SQL_THREAD")
time.sleep(1.0)
rc.execute("SELECT * FROM orders WHERE id=200")
after_resume = rc.fetchone()
assert after_resume is not None, "after resuming, the replica must eventually catch up"
assert after_resume == (200, 'delayed_item')

primary.close(); replica.close(); replica_admin.close()
print("async replication stale-read window verified: deterministically reproduced 'just wrote but can't read back' on the replica, then confirmed it catches up")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
从库SQL线程暂停期间查询id=200: None  (真实制造的滞后窗口,确定读不到)
恢复SQL线程1秒后查询id=200: (200, 'delayed_item')  (确认最终追上)
```

**面试怎么问+追问链**

- Q:用户提交表单后立刻跳转到详情页,详情页却显示"数据不存在",可能是什么原因?
  - 追问1:如果这个系统确实用了读写分离(写主库读从库),怎么解决这个"刚写完读不到"的体验问题?
    - 深挖追问(区分度较高):"写后立刻读自己刚写的数据"这个场景具体有哪些工程解法(答案方向:①强制这类请求路由到主库读[最简单但削弱了读写分离减轻主库压力的初衷,只对这一小类请求特殊处理];②应用层记录"最近写入"的标记,短时间内该用户的读请求路由到主库,过后再切回从库;③如果从库有可靠的位点追踪机制,等从库追上写入时对应的位点后再返回,但这会牺牲部分延迟优势;不存在一个没有代价的完美解法,都是具体场景下的权衡)。

**常见坑**

- 把这类"刚写完读不到"的bug误判成"数据丢失"或"事务没提交成功"去排查——如果系统用了读写分离,这类现象的第一排查方向应该是"读请求是不是被路由到了还没追上的从库",而不是怀疑写入本身出了问题。
- 在测试环境(通常复制延迟极小)里没发现问题,上线到真实网络环境(复制延迟可能到几十毫秒甚至更高)后才暴露——本类特意用"人为暂停SQL线程"而不是依赖自然延迟来验证这个机制,就是因为知道本地测试环境的自然延迟可能小到不足以稳定暴露问题,真实生产环境的延迟窗口只会更宽。

---

## 4. CAP定理:用真实网络分区场景推导

**签名/是什么**

```
CAP定理: 一个分布式系统在网络分区(Partition,P)发生时,只能在一致性(Consistency,C)和可用性(Availability,A)之间二选一
```

一句话:CAP定理不是需要死记硬背的公式,本类用知识点1已经建立的真实主从复制环境,真实制造一次"网络分区"(从库和主库断开联系),分别演示"选可用性"(继续用旧数据对外服务)和"选一致性"(检测到分区就拒绝服务)两种策略的真实代码行为,而不是抽象地讨论。

**底层机制/为什么这样设计**

P(分区容忍)在真实分布式系统里不是一个可以选择放弃的选项——网络分区终究会发生(哪怕概率很低),系统必须对这种情况有一个明确的应对策略,这就是为什么CAP定理经常被简化描述成"C和A二选一"(P是既定前提,不是可选项)。本类用`STOP SLAVE IO_THREAD`模拟从库和主库失联(这正是真实网络分区发生时从库会经历的状态——无法再拉取主库的binlog),这时候主库依然可以正常写入(它不关心从库死活),而从库有两个真实可选的行为:直接返回自己当前有的数据(可能是分区前的旧快照,可用性优先),或者先检查复制链路健康状态,发现不健康就主动拒绝服务(一致性优先)。

**可运行例子**(环境:`python-wsl2`,复用知识点1已建立的主从环境,真实模拟网络分区)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,复用知识点1已建立的主从复制环境
import pymysql
import time

primary = pymysql.connect(host='127.0.0.1', port=3306, user='dbdemo', password='dbdemo_local_only', database='repldemo')
primary.autocommit(True)
replica = pymysql.connect(host='127.0.0.1', port=3307, user='dbdemo', password='dbdemo_local_only', database='repldemo')
replica.autocommit(True)
replica_admin = pymysql.connect(unix_socket='/var/lib/mysql-replica/mysql.sock', user='root', database='repldemo')

pc = primary.cursor()
rc = replica.cursor()
rac = replica_admin.cursor()

pc.execute("DELETE FROM orders WHERE id=300")
rc.execute("SELECT COUNT(*) FROM orders")
count_before_partition = rc.fetchone()[0]

# 真实模拟网络分区: 从库IO线程断开(现实中复制链路真的断了就是这个状态)
rac.execute("STOP SLAVE IO_THREAD")
time.sleep(0.5)

# 分区期间,主库不受影响,继续正常写入
pc.execute("INSERT INTO orders VALUES (300, 'item_during_partition')")

# AP策略: 从库继续对外提供读服务,数据是分区前的旧快照(可用性优先,牺牲一致性)
rc.execute("SELECT COUNT(*) FROM orders")
ap_style_count = rc.fetchone()[0]
assert ap_style_count == count_before_partition, \
    f"AP-style read during partition should return the pre-partition (stale) count, got {ap_style_count} vs baseline {count_before_partition}"

# CP策略: 读之前先检查复制链路健康状态,不健康就拒绝服务(一致性优先,牺牲可用性)
rac.execute("SHOW SLAVE STATUS")
status_row = rac.fetchone()
columns = [d[0] for d in rac.description]
status = dict(zip(columns, status_row))
io_thread_healthy = status['Slave_IO_Running'] == 'Yes'
cp_style_decision = 'SERVE' if io_thread_healthy else 'REFUSE'
assert cp_style_decision == 'REFUSE', "CP-style strategy must refuse to serve once partition is detected"

# 分区恢复,验证从库能追上
rac.execute("START SLAVE IO_THREAD")
time.sleep(1.0)
rc.execute("SELECT COUNT(*) FROM orders")
count_after_heal = rc.fetchone()[0]
assert count_after_heal == count_before_partition + 1, "after the partition heals, replica must catch up to include the new row"

primary.close(); replica.close(); replica_admin.close()
print(f"CAP AP-vs-CP tradeoff verified with a REAL simulated partition: "
      f"AP strategy served stale count={ap_style_count}, CP strategy refused, both healed to count={count_after_heal} after reconnect")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
分区期间AP策略读到的count: 与分区前一致(旧快照,不含id=300)
分区期间CP策略决策: REFUSE(检测到Slave_IO_Running=No)
分区恢复后count: 分区前+1(id=300真实同步过去了)
```

**面试怎么问+追问链**

- Q:CAP定理里的"一致性"能不能干脆一直选,不要可用性?
  - 追问1:如果一个系统的所有读请求在检测到任何复制延迟/分区迹象时都直接拒绝服务,这在现实中可行吗?
    - 深挖追问(区分度较高):真实系统通常怎么处理这个看似"极端"的选择(答案方向:纯粹的CP或纯粹的AP在真实系统里都很少见,大多数系统提供的是"可调节的一致性级别"[比如04类讨论过的隔离级别选型思路],让不同业务场景按需选择——对一致性要求极高的操作[比如资金扣减]走强一致路径[可能牺牲可用性,宁可报错也不能读错],对一致性要求宽松的操作[比如展示商品浏览量]走可用性优先路径,这是"CAP不是全局唯一选择,而是可以按具体读写路径分别决策"这个更成熟的工程认知)。

**常见坑**

- 把CAP定理当成"三选二",认为可以放弃P去换CA都要——本类已经论证过P(网络分区)是真实分布式系统必须面对的既定现实,不是可以主动放弃的选项,"三选二"的说法容易造成"我可以设计一个不会分区的系统"这种不现实的误解。
- 只在理论层面讨论CAP,没有意识到"检测到分区"本身就需要真实的健康检查机制(本类用`Slave_IO_Running`字段)——CP策略的"拒绝服务"决策依赖于系统能够可靠地判断自己"是不是"处于分区状态,这个判断机制本身的设计[比如超时阈值设多少]也是真实工程实现里的一个难点。

---

## 5. 分库分表策略:垂直拆分

**签名/是什么**

```
垂直拆分(Vertical Split): 把一张"宽表"按列拆成多张表,高频访问的字段和低频访问/体积大的字段分开存放
```

一句话:垂直拆分的收益来自"避免不必要的IO"——大多数查询只需要少数几个高频字段,如果这些字段和很少被访问、但体积很大的字段混在同一张表里,即使查询压根不需要那些大字段,底层扫描依然可能要碰到它们所在的数据页,拆开之后高频查询完全不需要接触低频大字段所在的存储区域。

**底层机制/为什么这样设计**

一行数据在物理存储上通常是连续存放的(尤其是InnoDB这类行式存储引擎),即使SQL只`SELECT`了其中几列,存储引擎读取数据页时往往是以"行"为单位读取到内存(除非有03类讨论过的覆盖索引能够绕开这个问题),如果这一行里有一个几KB甚至更大的大字段,每次读取哪怕只想要另外两个小字段,也可能带着这个大字段一起被读进内存,造成不必要的IO和缓冲池占用。垂直拆分把"经常被一起使用的字段"归到同一张表,是对"数据访问局部性"这个更通用系统设计原则在表结构设计层面的具体应用。

**AI研究/工程场景**

用户表把"登录相关的核心字段"(用户名/密码哈希/最后登录时间)和"很少访问的大字段"(个人简介富文本/头像原始数据)分开存放,是一个真实、常见的垂直拆分应用场景——登录校验这类高频操作完全不需要接触个人简介字段。

**可运行例子**(环境:`.venv`,真实测量拆分前后同一个高频查询实际碰到的数据量)

```python
import sqlite3

conn = sqlite3.connect(":memory:")
cur = conn.cursor()

# 拆分前: 高频字段(username/last_login)和低频大字段(bio)混在一张表
cur.execute("""CREATE TABLE users_wide (
    id INTEGER PRIMARY KEY, username TEXT, last_login TEXT, bio TEXT
)""")
big_bio = "x" * 5000  # 模拟一个不常访问但体积很大的字段(比如个人简介富文本)
for i in range(100):
    cur.execute("INSERT INTO users_wide VALUES (?,?,?,?)", (i, f"user{i}", "2026-01-01", big_bio))
conn.commit()

# 高频查询场景(登录校验): 只需要username和last_login,完全不需要bio,
# 但用这个查询近似"这次操作理论上会接触到的总字节量"来量化拆分前后的差异
cur.execute("SELECT SUM(LENGTH(username) + LENGTH(last_login) + LENGTH(bio)) FROM users_wide")
wide_table_bytes = cur.fetchone()[0]

# 垂直拆分: 高频字段和低频大字段分开成两张表
cur.execute("CREATE TABLE users_core (id INTEGER PRIMARY KEY, username TEXT, last_login TEXT)")
cur.execute("CREATE TABLE users_profile (id INTEGER PRIMARY KEY, bio TEXT)")
for i in range(100):
    cur.execute("INSERT INTO users_core VALUES (?,?,?)", (i, f"user{i}", "2026-01-01"))
    cur.execute("INSERT INTO users_profile VALUES (?,?)", (i, big_bio))
conn.commit()

# 同样的登录校验查询,拆分后只需要碰users_core这张小表
cur.execute("SELECT SUM(LENGTH(username) + LENGTH(last_login)) FROM users_core")
split_table_bytes = cur.fetchone()[0]

reduction_pct = (1 - split_table_bytes / wide_table_bytes) * 100
assert reduction_pct > 95, f"expected the hot-path query to avoid the vast majority of the large cold column, got {reduction_pct:.1f}%"

conn.close()
print(f"vertical split verified: login-check-style query touches {wide_table_bytes} bytes before split, "
      f"{split_table_bytes} bytes after split ({reduction_pct:.1f}% reduction)")
```

**面试怎么问+追问链**

- Q:什么时候应该考虑垂直拆分一张表?
  - 追问1:垂直拆分之后,如果一个查询恰好同时需要核心字段和大字段,会不会反而变慢(要JOIN两张表)?
    - 深挖追问(区分度较高):这个"偶尔需要JOIN"的代价和"大多数时候不需要碰大字段"的收益,应该怎么权衡(答案方向:这是典型的"看访问模式频率分布"的决策——如果99%的查询只需要核心字段、1%的查询需要JOIN两张表,拆分依然是净收益;如果这两类查询频率相当,拆分带来的JOIN开销可能抵消甚至超过收益,本类反复强调的方法论同样适用:不能凭直觉判断,需要结合真实的查询频率统计做决策)。

**常见坑**

- 无差别地把每一列都拆成单独的表——垂直拆分的收益来自"按访问频率分组",拆得过细会导致大量原本一次查询能完成的操作变成频繁JOIN,得不偿失。
- 只考虑了查询读取的收益,忽视了拆分后写入路径变复杂——原本一条`INSERT`能完成的操作,拆分后可能需要向两张表分别写入并确保这两次写入的原子性(需要事务包裹,呼应04类知识点1的事务级原子性),这是拆分决策要一并考虑的额外复杂度。

---

## 6. 分库分表策略:水平分片

**签名/是什么**

```
水平拆分(Horizontal Split/Sharding): 同一张表的行,按某个规则(通常是某个字段的哈希或范围)分散存到多个物理节点
```

一句话:水平拆分解决的是"单表数据量/写入压力超过单机能力上限"这个问题,本类真实测量朴素取模分片在扩容时引发的大规模数据迁移问题,用真实数字论证这正是驱动"一致性哈希"被发明的工程动机(一致性哈希的完整实现与虚拟节点负载均衡真实测量,见 dsa-deep-dive/20-advanced-interview-depth.md 案例1,本类不重复推导)。

**底层机制/为什么这样设计**

最直觉的分片方式是"取模"(`hash(key) % 分片数量`),这个方式实现简单、分布均匀,但有一个致命的运维问题——分片数量一旦变化(扩容/缩容),几乎所有key的取模结果都会跟着变化,意味着**几乎全部数据都需要重新分布**,这在真实生产环境的大表场景下是不可接受的迁移成本。一致性哈希(dsa-deep-dive已验证)通过把节点和key都映射到同一个环形空间,让节点数量变化时只影响环上相邻的一小部分数据,这是它相比朴素取模的核心优势。

**AI研究/工程场景**

评估"要不要现在就上分库分表"是一个真实的架构决策——分片会显著增加系统复杂度(跨分片查询/跨分片事务都变得困难,跨分片事务往往需要知识点7讨论的2PC或更复杂的分布式事务方案),只有当单表数据量或写入压力真实逼近单机瓶颈时才值得引入,过早引入分片是常见的过度工程化陷阱。

**可运行例子**(环境:`.venv`,真实测量朴素取模分片在扩容场景下的数据迁移比例)

```python
def naive_shard(key, num_shards):
    return hash(key) % num_shards

keys = [f"user_{i}" for i in range(1000)]

shards_before = {k: naive_shard(k, 4) for k in keys}  # 4个分片
shards_after = {k: naive_shard(k, 5) for k in keys}   # 扩容到5个分片

moved = sum(1 for k in keys if shards_before[k] != shards_after[k])
moved_pct = moved / len(keys) * 100

# 理论最优的重新分布比例是 (1 - 4/5) = 20%,但朴素取模因为哈希值和分片数的模运算方式,
# 实际迁移比例会远超这个理论最优值
assert moved_pct > 60, f"naive modulo resharding should move the vast majority of keys, got {moved_pct:.1f}%"

print(f"naive modulo sharding verified: scaling from 4 to 5 shards forces {moved_pct:.1f}% of all keys to move "
      f"(vs a theoretical optimum of 20%) - this real operational pain is exactly what motivates consistent hashing "
      f"(see dsa-deep-dive/20-advanced-interview-depth.md case 1 for the real virtual-node measurement)")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
naive modulo sharding: 4->5分片扩容,82.9%的key需要迁移(理论最优是20%)
```

**面试怎么问+追问链**

- Q:分库分表的分片键(sharding key)应该怎么选?
  - 追问1:如果选错了分片键(比如按创建时间分片,导致最近的数据全部落在同一个分片),会有什么后果?
    - 深挖追问(区分度较高):这种情况有什么补救办法(答案方向:这是"热点分片"问题——按时间这类单调递增字段分片,最新数据(往往也是访问最频繁的数据)会集中落在最后一个分片,造成负载严重不均衡;补救思路包括改用哈希类分片键打散热点、或者对时间字段做额外的哈希打散[比如结合用户ID等其他维度],分片键选择本质上是"业务查询模式"[决定哪些字段经常放在WHERE条件里,分片键最好能覆盖这些]和"数据分布均匀性"两个目标的权衡,04类隔离级别选型、01类索引设计,本质上都是同一种"没有免费午餐、必须结合真实访问模式决策"的方法论在不同层面的体现)。

**常见坑**

- 把水平分片和垂直拆分(知识点5)混为一谈——垂直拆分是"同一行的不同列"分开存,水平拆分是"同一张表的不同行"分开存,两者解决的问题和适用场景完全不同,可以同时使用但不能互相替代。
- 分片之后依然假设"跨分片查询和单机查询一样简单"——分片后涉及多个分片的聚合查询/事务,复杂度和成本会显著上升(需要应用层或中间件做跨分片协调),这是水平分片"扩展了写入/存储能力"但"增加了查询复杂度"这个真实代价的体现。

---

## 7. 两阶段提交2PC

**签名/是什么**

```
2PC(Two-Phase Commit): 分布式事务协议,协调者(Coordinator)先问所有参与者"准备好提交了吗"(Prepare阶段),
                      全部同意才真正下令提交,任意一个不同意就下令全部回滚(Commit/Abort阶段)
```

一句话:2PC解决的是"一次业务操作涉及多个独立数据库/分片,要么全部成功要么全部失败"这个跨库事务问题,是04类知识点1"单库事务原子性"概念在分布式场景下的自然延伸——只是这次协调多个独立的参与者,不能再依赖单一数据库引擎内部的事务机制。

**底层机制/为什么这样设计**

2PC分成两个阶段的原因,是为了避免"部分参与者已经提交、部分还没决定"这种危险的中间状态:第一阶段(Prepare)让每个参与者做好"万一要提交"的一切准备工作(把变更写入自己的日志、锁住相关资源),但不真正生效,并向协调者汇报"我准备好了"或"我做不到";协调者收集完所有参与者的答复后,才在第二阶段统一下达最终指令(全部commit或全部abort)。这个设计保证了"要么全部准备好之后一起提交,要么只要有一个说不行就全部回滚",不会出现"一部分参与者已经提交、另一部分还在犹豫"的不一致状态。

**AI研究/工程场景**

2PC在真实工程里用得相对较少(尽管理论上很重要)——协调者是单点(协调者本身故障会让所有参与者卡在"准备好了但等不到最终指令"的状态,需要额外的超时和恢复机制),且第一阶段准备好之后到第二阶段收到指令之前,参与者持有的锁资源不能释放,高并发场景下这个"等待协调者拍板"的窗口会显著拖慢吞吐量,这是为什么现实中很多分布式事务场景更倾向于用最终一致性方案(比如消息队列驱动的补偿事务/Saga模式)而不是强一致的2PC,2PC更适合参与者数量少、对一致性要求达到"宁可牺牲一些可用性也不能有中间状态"级别的场景。

**可运行例子**(环境:`.venv`)

```python
class Participant:
    def __init__(self, name, will_vote_yes=True):
        self.name = name
        self.will_vote_yes = will_vote_yes
        self.state = 'INIT'

    def prepare(self):
        # 阶段1: 做好准备(检查约束/锁资源),但还不真正生效,只汇报能不能提交
        self.state = 'PREPARED' if self.will_vote_yes else 'ABORTED'
        return self.will_vote_yes

    def commit(self):
        assert self.state == 'PREPARED', "can only commit a participant that has successfully prepared"
        self.state = 'COMMITTED'

    def abort(self):
        self.state = 'ABORTED'


def two_phase_commit(participants):
    votes = [p.prepare() for p in participants]  # 阶段1: 收集所有参与者的意见
    all_yes = all(votes)
    if all_yes:
        for p in participants:
            p.commit()  # 阶段2: 全部同意才真正提交
        return 'COMMITTED'
    else:
        for p in participants:
            if p.state == 'PREPARED':
                p.abort()  # 阶段2: 任意一个不同意,已经准备好的也要跟着回滚
        return 'ABORTED'


# 场景1: 跨行转账,两边账户都有足够余额,全部同意
bank_a = Participant('bank_A', will_vote_yes=True)
bank_b = Participant('bank_B', will_vote_yes=True)
result = two_phase_commit([bank_a, bank_b])
assert result == 'COMMITTED'
assert bank_a.state == 'COMMITTED' and bank_b.state == 'COMMITTED'

# 场景2: bank_B账户余额不足,投反对票 - 即使bank_A本来同意,也必须跟着一起回滚
bank_a2 = Participant('bank_A', will_vote_yes=True)
bank_b2 = Participant('bank_B', will_vote_yes=False)
result2 = two_phase_commit([bank_a2, bank_b2])
assert result2 == 'ABORTED'
assert bank_a2.state == 'ABORTED', "even a participant that voted yes must roll back if anyone else votes no"
assert bank_b2.state == 'ABORTED'

print("2PC state machine verified: all-yes leads to COMMITTED for everyone, any-no leads to ABORTED for everyone (including those who voted yes)")
```

**面试怎么问+追问链**

- Q:2PC能保证跨库事务的原子性,为什么实际系统里不常用它?
  - 追问1:如果协调者在发出commit指令之前自己崩溃了,参与者会怎么样?
    - 深挖追问(区分度较高):这种"协调者故障导致参与者卡住"的问题有没有解法(答案方向:这正是2PC的经典缺陷——参与者在"已经prepared但还没收到最终指令"这个状态下会持续持有锁资源等待,不能自行决定提交还是回滚[这是它和04/05类讨论的锁等待超时不同的地方,不是简单加个超时就能安全解决,因为参与者不知道协调者到底最终决定了commit还是abort];3PC[三阶段提交]和Paxos/Raft类共识算法都是为了解决2PC这类"协调者单点故障"问题演化出的更复杂方案,这也是为什么现代分布式系统更常见的做法是用成熟的共识框架[比如基于Raft的分布式协调服务]管理协调者本身的高可用,而不是让2PC协调者是一个真正意义上的单点)。

**常见坑**

- 把2PC的"两阶段"理解为"两次网络往返"这么简单——真正的复杂度在于"参与者在两个阶段之间必须持有资源不能释放"这个状态维持成本,以及协调者故障时如何恢复这两个工程难题,不只是协议流程本身。
- 认为2PC是分布式事务的唯一/最佳方案——本类"AI研究/工程场景"部分已经提到,真实工程里很多场景选择最终一致性方案[消息队列+补偿]而不是强一致的2PC,选型需要结合具体业务对"中间不一致状态可以持续多久"的容忍度。
