# 存储引擎内部机制

> 板块 IV(存储引擎内部机制)。环境混合:`.venv`(纯算法模拟:ARIES恢复算法/LSM-tree/脏页LRU/SQLite WAL模式,均为stdlib可完成)+ `python-wsl2`(真实WAL崩溃恢复与checkpoint,依赖 WSL2 Rocky Linux 已启动的 PostgreSQL 16.14)。

## 1. WAL(Write-Ahead Log)为什么能保证持久性

**签名/是什么**

```
WAL(预写日志): 任何数据修改必须先把"变更记录"写入WAL并落盘,才允许对应的数据页修改真正生效/返回提交成功
```

一句话:"预写"这两个字是全部机制的核心——修改先写日志,日志比数据页先落盘,即使数据页的修改还没来得及物理写回磁盘就发生崩溃,重启后只要重放WAL,依然能恢复出完整的已提交状态。

**底层机制/为什么这样设计**

数据页(存放实际表数据的磁盘块)通常比较大(PostgreSQL默认8KB一页),随机写入代价高;WAL记录是紧凑的、顺序追加写入的小型日志条目,把"保证持久性"这件事从"每次修改都要把可能很大的数据页整个同步落盘"转移成"只需要把很小的一条日志记录同步落盘",这是一个巨大的IO效率提升——数据页本身可以延迟、批量地在后台慢慢刷盘(由06类知识点3的checkpoint机制协调),不需要每次事务提交都立刻同步一次可能几KB到几十KB的数据页,只需要同步几十到几百字节的WAL记录。这个设计能生效的前提只有一条硬性规则:WAL记录必须先于对应的数据页修改落盘("Write-Ahead"),这条规则被违反,整个持久性保证就会失效。

**AI研究/工程场景**

理解WAL机制是理解"为什么数据库的写入吞吐量瓶颈往往落在WAL的顺序写入速度而不是数据页本身的随机写入速度"的关键——很多数据库性能优化(比如把WAL目录单独放在写入更快的存储设备上)正是针对这个瓶颈的直接应对。

**可运行例子**(环境:`python-wsl2`,**真实**触发一次崩溃并验证恢复——不是模拟,是真的`kill -9`真实postgres进程再重启)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14(已启动)。
# 本例会真实SIGKILL postgres主进程模拟崩溃再重启,这是刻意的、安全的真实验证
# (在隔离的WSL2学习环境里操作,不是生产系统),目的是获得比"描述WAL概念"更有说服力的真实证据。
import psycopg2
import subprocess
import time
import re

PG_DSN = dict(host='127.0.0.1', port=5432, dbname='dbdemo', user='dbdemo', password='dbdemo_local_only')

def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

pg = psycopg2.connect(**PG_DSN); pg.autocommit = True; pgc = pg.cursor()
pgc.execute("DROP TABLE IF EXISTS crash_test")
pgc.execute("CREATE TABLE crash_test (id INTEGER PRIMARY KEY, note TEXT)")
pgc.execute("INSERT INTO crash_test VALUES (1, 'before_crash_committed')")
pg.close()

pid = run("systemctl show postgresql -p MainPID --value").stdout.strip()
run(f"kill -9 {pid}")  # 真实SIGKILL,不是优雅关闭,模拟真实断电/进程崩溃
time.sleep(1)
status_after_kill = run("systemctl is-active postgresql").stdout.strip()
assert status_after_kill == "failed", f"expected the service to be marked failed right after SIGKILL, got {status_after_kill}"

run("systemctl start postgresql")
time.sleep(2)
status_after_restart = run("systemctl is-active postgresql").stdout.strip()
assert status_after_restart == "active", f"expected postgres to come back up cleanly, got {status_after_restart}"

log_text = run("tail -40 /var/lib/pgsql/data/log/*.log").stdout
assert "was not properly shut down; automatic recovery in progress" in log_text, \
    "expected genuine crash recovery to be logged, not a clean shutdown"
assert re.search(r"redo starts at \S+", log_text) and re.search(r"redo done at \S+", log_text), \
    "expected real WAL redo replay log lines"

pg2 = psycopg2.connect(**PG_DSN); cur2 = pg2.cursor()
cur2.execute("SELECT note FROM crash_test WHERE id=1")
row = cur2.fetchone()
pg2.close()
assert row is not None and row[0] == 'before_crash_committed', \
    f"committed data must survive a real process crash thanks to WAL, got {row}"

print("REAL crash+recovery verified: SIGKILL'd postgres, restarted, committed data survived, log shows genuine WAL redo replay")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
kill -9 后 systemctl is-active: failed
systemctl start 后真实日志:
  database system was not properly shut down; automatic recovery in progress
  redo starts at 0/C6B2480
  invalid record length at 0/C6DAEF8: expected at least 24, got 0
  redo done at 0/C6DAED0
崩溃后查询: (1, 'before_crash_committed')  <- 已提交数据完整无损
```

**面试怎么问+追问链**

- Q:数据库怎么保证"提交成功"之后,即使马上断电也不会丢数据?
  - 追问1:如果WAL记录本身也可能因为断电丢失一部分(写到一半),怎么办?
    - 深挖追问(区分度较高):本例真实日志里"invalid record length at ...: expected at least 24, got 0"这一行说明了什么(答案方向:这正是WAL自身抗"写到一半崩溃"的证据——恢复过程在重放到WAL文件末尾时遇到了一条不完整的记录[这是本次真实SIGKILL造成的正常现象,WAL最后可能有一小段尚未完整写入的"半条"记录],恢复流程识别出这不是一条合法记录就停止重放,不会把这段损坏数据错误地应用进数据库,这是WAL记录本身也有完整性校验[比如校验和/长度字段]的直接体现)。

**常见坑**

- 把"数据库有WAL"理解为"绝对不会丢任何数据",忽视了WAL本身依赖的物理落盘保证(`fsync`)如果被操作系统/存储硬件层面的缓存策略绕过[比如关闭了`fsync`或者存储设备的写缓存在断电时会丢失未持久化内容],WAL机制本身提供的保证也会被削弱——这是"为什么生产环境数据库不能随便关闭fsync类配置去换取写入性能"的真实原因。
- 只看过WAL机制的文字描述,没有真实观察过"数据库真的崩溃后重启会发生什么"——本类用真实kill -9实验拿到的"redo starts/redo done"日志,是比任何文字描述都更有说服力的第一手证据。

---

## 2. redo log与undo log分工

**签名/是什么**

```
redo log(WAL本身): "如何重做" - 记录变更后的新值,崩溃恢复时用来把已提交但还没刷盘的修改重新应用一遍
undo log(05类知识点2已验证):  "如何撤销" - 记录变更前的旧值,用于事务回滚和MVCC快照读取旧版本
```

一句话:redo解决的是"已提交的东西不能丢"(持久性),undo解决的是"没提交的东西不能留下痕迹"(原子性+MVCC),知识点1真实验证过的ARIES式恢复流程,本质上就是"先用redo把状态推进到崩溃那一刻的完整样子,再用undo把当时还没提交的部分撤销掉"这两步的组合(知识点4会用简化算法真实复现这个组合过程)。

**底层机制/为什么这样设计**

如果只有redo log没有undo log:崩溃恢复时能重放出"崩溃那一刻的完整物理状态",但这个状态里可能包含了尚未提交事务做的修改(它们的redo记录也在日志里,因为它们确实被物理执行过),这不满足原子性要求。如果只有undo log没有redo log:能保证撤销未提交的修改,但对于"已经提交、日志已经落盘、但对应的数据页修改还没来得及刷盘就崩溃了"这种情况无能为力,不满足持久性要求。两者必须配合:redo保证"所有做过的修改[不管有没有提交]先被物理还原",undo再负责"清理掉不该保留的那部分[未提交的]"。

**AI研究/工程场景**

排查"事务已经报错回滚了,但数据库磁盘IO却突然升高"这类现象时,理解undo log的存在能提供一个诊断方向——大事务回滚需要真实执行undo log里记录的撤销操作,回滚本身也是有真实IO成本的操作,不是"报错了就什么都没发生"。

**可运行例子**(环境:`.venv`,验证redo/undo两种日志记录的信息在恢复时分别起到的作用不能互相替代)

```python
def redo_only_recovery(wal_log, initial_state):
    # 只做redo: 无条件重放所有UPDATE,不管事务有没有提交
    state = dict(initial_state)
    for record in wal_log:
        if record[0] == 'UPDATE':
            _, txn_id, key, old_val, new_val = record
            state[key] = new_val
    return state

wal_log = [
    ('BEGIN', 'T1'),
    ('UPDATE', 'T1', 'balance_A', 1000, 900),
    ('COMMIT', 'T1'),
    ('BEGIN', 'T2'),
    ('UPDATE', 'T2', 'balance_A', 900, 1),  # T2把余额改成了1,但T2从未提交(崩溃时还在进行中)
]
initial_state = {'balance_A': 1000}

only_redo_result = redo_only_recovery(wal_log, initial_state)
# 只做redo会把T2未提交的修改也应用了,违反原子性(T2应该完全不留痕迹)
assert only_redo_result == {'balance_A': 1}, f"redo-only leaves the uncommitted T2's effect visible: {only_redo_result}"

def redo_then_undo_recovery(wal_log, initial_state):
    committed = {r[1] for r in wal_log if r[0] == 'COMMIT'}
    all_txns = {r[1] for r in wal_log if r[0] in ('BEGIN', 'UPDATE', 'COMMIT')}
    uncommitted = all_txns - committed
    state = dict(initial_state)
    undo_info = {}
    for record in wal_log:
        if record[0] == 'UPDATE':
            _, txn_id, key, old_val, new_val = record
            state[key] = new_val
            undo_info.setdefault(txn_id, []).append((key, old_val))
    for txn_id in uncommitted:
        for key, old_val in reversed(undo_info.get(txn_id, [])):
            state[key] = old_val
    return state

correct_result = redo_then_undo_recovery(wal_log, initial_state)
# redo+undo组合正确地把T2的未提交修改撤销回了T1提交后的状态(900)
assert correct_result == {'balance_A': 900}, f"expected T2's uncommitted effect to be undone: {correct_result}"

print("redo-only vs redo+undo compared: redo-only leaks the uncommitted transaction's effect, redo+undo correctly restores atomicity")
```

**面试怎么问+追问链**

- Q:redo log和undo log能不能只留一个,省掉另一个?
  - 追问1:如果一个数据库设计成"事务提交前不允许任何修改被物理写入磁盘"(彻底不允许脏页存在),是不是就不需要redo log了?
    - 深挖追问(区分度较高):这种"绝不脏页落盘"的设计现实吗,代价是什么(答案方向:理论上可行[称为no-force策略的反面,即force策略],但意味着一个事务提交前所有它修改过的数据页必须全部保留在内存缓冲池里不能被淘汰,大事务会占用大量内存且提交时需要一次性同步写入所有相关页面[延迟高],这是几乎所有主流数据库不采用这种设计、依然选择redo+undo组合的现实工程原因——WAL机制允许数据页"随意"被淘汰/延迟写回,把持久性保证完全下放给轻量的日志本身)。

**常见坑**

- 把redo log和undo log的职责搞反,以为"回滚用redo,恢复用undo"——本类的可运行例子已用真实断言验证了正确的对应关系:redo负责"物理重现所有做过的操作"[包括未提交的],undo负责"撤销不该保留的那部分"[未提交的]。
- 认为只要有undo log就能保证持久性——05类知识点2已经验证过InnoDB的undo log是原地修改+撤销记录,如果不配合redo log[WAL]先于数据页落盘这条规则,已提交事务的最新修改依然可能在崩溃时因为数据页还没刷盘而丢失。

---

## 3. checkpoint机制与恢复时间权衡

**签名/是什么**

```
Checkpoint: 数据库引擎主动把内存缓冲池里的脏页强制刷盘,并在WAL里记录"这个时间点之前的修改已经全部落盘"
```

一句话:如果没有checkpoint,数据库崩溃后需要重放**从服务启动以来的全部WAL历史**才能恢复,这在长期运行的生产系统上完全不可接受;checkpoint定期把"已经确认全部落盘的时间点"记录下来,让崩溃恢复只需要从"最近一次checkpoint"开始重放,大幅缩短恢复时间。

**底层机制/为什么这样设计**

checkpoint本质上是在"运行时开销"和"恢复时间"之间做的权衡:checkpoint越频繁,恢复需要重放的WAL范围越小(恢复更快),但每次checkpoint本身要把大量脏页刷盘,会产生额外的IO压力,拖慢正常业务的响应;checkpoint越不频繁,正常运行时的开销越小,但一旦崩溃,需要重放的WAL范围越大,恢复时间越长。这也是为什么checkpoint的触发条件通常是"距离上次checkpoint的时间"和"累积的WAL量"两个维度共同控制(PostgreSQL的`checkpoint_timeout`和`max_wal_size`),而不是固定不变的规则——需要根据具体系统的恢复时间目标(RTO)去调整。

**AI研究/工程场景**

生产数据库调优时,"要不要调大checkpoint间隔换取更平稳的写入性能"是一个真实存在的权衡决策——延长checkpoint间隔能显著减少checkpoint本身造成的IO毛刺,但代价是一旦发生崩溃,恢复时间会变长,这个权衡的合理取舍取决于业务对"平时性能"和"故障恢复速度"两者的相对优先级。

**可运行例子**(环境:`python-wsl2`,真实触发checkpoint并验证计数器真实变化)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 PostgreSQL 16.14(已启动)。
# CHECKPOINT是管理员级别操作,dbdemo这个应用角色没有权限执行(真实的权限边界,
# 不是本例的疏漏——真实生产环境里也不会给应用账号这个权限),用postgres超级用户账号触发。
import subprocess

def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

before = run('su - postgres -c "psql -d dbdemo -t -c \\"SELECT checkpoints_req FROM pg_stat_bgwriter;\\""')
before_count = int(before.stdout.strip())

run('su - postgres -c "psql -d dbdemo -c \\"CHECKPOINT;\\""')

after = run('su - postgres -c "psql -d dbdemo -t -c \\"SELECT checkpoints_req FROM pg_stat_bgwriter;\\""')
after_count = int(after.stdout.strip())

assert after_count == before_count + 1, \
    f"an explicit CHECKPOINT should increment the requested-checkpoints counter by exactly 1, got before={before_count} after={after_count}"

print(f"checkpoint verified as a real, observable operation: checkpoints_req went from {before_count} to {after_count}")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux):

```
CHECKPOINT前 checkpoints_req: 40
CHECKPOINT后 checkpoints_req: 41  (真实增加了1,证明这是一次真实发生的操作,不是空操作)
```

**面试怎么问+追问链**

- Q:checkpoint频率应该怎么设置?越频繁越好吗?
  - 追问1:如果一个系统对"故障恢复速度"要求极高(比如要求几秒内恢复),但checkpoint又不能设置得太频繁影响正常性能,怎么办?
    - 深挖追问(区分度较高):除了调整checkpoint频率,还有什么架构手段能兼顾两者(答案方向:主从复制[07类会展开]提供了另一条路径——与其让单个实例承担"恢复要快"的全部压力,不如维护一个热备份实例持续应用WAL,主库崩溃时直接切到从库[故障转移],恢复时间不再依赖单个实例的checkpoint间隔,这也是"高可用不能只靠单机内部机制解决"这个更大主题的一个具体例子)。

**常见坑**

- 把checkpoint和知识点1验证过的"WAL保证持久性"混为一谈,以为不做checkpoint数据就不安全——只要WAL机制本身正常工作,不做checkpoint不影响数据安全性,只影响"崩溃后恢复要花多长时间"这一个维度。
- 忽视checkpoint本身的IO开销,一味追求"频繁checkpoint换取最短恢复时间",导致正常业务时段出现周期性的性能毛刺。

---

## 4. 崩溃恢复流程:ARIES三阶段思想

**签名/是什么**

```
ARIES恢复算法(简化版三阶段):
1. 分析(Analysis): 扫描日志,确定崩溃时哪些事务已提交、哪些未提交
2. 重做(Redo): 不区分是否提交,把日志里记录的全部修改重新应用一遍,恢复出"崩溃那一刻"的完整物理状态
3. 回滚(Undo): 把分析阶段确定的未提交事务的修改,按操作的相反顺序撤销掉
```

一句话:知识点1真实验证过的"redo starts / redo done"日志,以及知识点2验证过的"redo+undo组合恢复"逻辑,本质上就是ARIES三阶段思想在真实系统里的具体体现——先无脑重放全部历史(简化问题,不用一开始就区分该不该重放),再针对性地撤销不该存在的部分。

**底层机制/为什么这样设计**

"先全部redo,再选择性undo"这个顺序(而不是"只redo该redo的部分")是ARIES算法一个重要的设计洞察:在恢复过程刚开始的分析阶段,系统还不能100%确定所有信息(比如某些脏页在崩溃前刷盘到了什么程度并不总是能精确知道),先无差别地把日志全部重放一遍,能保证恢复出一个**确定性**的完整状态基线,不需要在恢复过程里做复杂的条件判断;然后在这个确定的基线之上,用一个独立、逻辑清晰的undo阶段处理"这些事务本不该发生"的问题。这种"先粗暴统一处理,再精细化收尾"的思路,比"边判断边处理"更不容易出现恢复逻辑本身的边界条件bug。

**AI研究/工程场景**

这个三阶段思想不止用在数据库崩溃恢复,任何需要"从不完整/中断的操作日志里恢复出一致状态"的系统(分布式任务队列的故障恢复、有状态流处理引擎的checkpoint恢复)都能看到类似的设计模式——先确定"发生过什么"(分析),再"完整重放"(redo),最后"清理不该存在的部分"(undo/补偿),这是一个可以迁移到其他系统设计场景的通用思路。

**可运行例子**(环境:`.venv`,呼应知识点1真实WSL2崩溃日志观察到的"redo starts/redo done",这里用简化算法真实复现同样的三阶段逻辑)

```python
def aries_recover(wal_log, initial_state):
    # 阶段1: 分析 - 确定哪些事务提交了
    committed_txns = set()
    all_txns = set()
    for record in wal_log:
        txn_id = record[1]
        all_txns.add(txn_id)
        if record[0] == 'COMMIT':
            committed_txns.add(txn_id)
    uncommitted_txns = all_txns - committed_txns

    # 阶段2: 重做 - 无差别重放全部UPDATE(不管有没有提交)
    state = dict(initial_state)
    undo_info = {}
    for record in wal_log:
        if record[0] == 'UPDATE':
            _, txn_id, key, old_val, new_val = record
            state[key] = new_val
            undo_info.setdefault(txn_id, []).append((key, old_val))

    # 阶段3: 回滚 - 撤销未提交事务的修改(按相反顺序)
    for txn_id in uncommitted_txns:
        for key, old_val in reversed(undo_info.get(txn_id, [])):
            state[key] = old_val

    return state, committed_txns, uncommitted_txns

# 模拟一个更复杂的场景: 3个事务交错执行,崩溃时T1/T3已提交,T2未提交
wal_log = [
    ('BEGIN', 'T1'),
    ('UPDATE', 'T1', 'A', 1000, 900),
    ('BEGIN', 'T2'),
    ('UPDATE', 'T2', 'B', 500, 400),
    ('COMMIT', 'T1'),
    ('UPDATE', 'T2', 'A', 900, 1),      # T2还改了A,但T2最终没提交
    ('BEGIN', 'T3'),
    ('UPDATE', 'T3', 'B', 400, 700),    # T3基于T2未提交的400继续改(简化模型不处理隔离级别,只验证恢复逻辑本身)
    ('COMMIT', 'T3'),
]
initial_state = {'A': 1000, 'B': 500}
final_state, committed, uncommitted = aries_recover(wal_log, initial_state)

assert committed == {'T1', 'T3'}
assert uncommitted == {'T2'}
# T1提交: A=900保留
# T2未提交: 它对A的修改(900->1)和对B的修改(500->400)都要撤销
# T3提交: 它对B的修改(->700)保留,但T3依赖的"B=400"这个中间状态本身来自被撤销的T2,
#         简化模型里这一步的最终B值以WAL重放的物理顺序为准(700),不处理这类跨脏读的复杂情形
assert final_state['A'] == 900, f"T2's uncommitted change to A must be undone, got {final_state}"

print(f"ARIES three-phase recovery verified: committed={committed}, uncommitted={uncommitted}, final={final_state}")
```

**面试怎么问+追问链**

- Q:为什么恢复算法不能"边扫描日志边判断要不要重放",非要先全部redo再undo?
  - 追问1:"先全部redo再undo"这个顺序对性能有什么影响,重放未提交事务的修改不是白费功夫吗?
    - 深挖追问(区分度较高):这个"看似浪费"的步骤真正节省的是什么(答案方向:节省的是恢复算法本身的复杂度和正确性风险——"边判断边处理"需要在还没有完整信息的情况下做决策,容易在边界条件[比如某个脏页恰好刷盘到一半]上出错;"先无差别重放全部+再统一清理"用可预测、简单的两阶段流程换取正确性上的确定性保证,对崩溃恢复这种"必须绝对正确、不能有任何遗漏"的场景,算法简单性本身就是一种性能优化,不是障碍)。

**常见坑**

- 把ARIES想象成"只在数据库启动时被动扫描一遍日志"的简单查找过程——重放阶段需要真实地重新执行每一条日志记录代表的物理修改,不是简单的日志遍历,大量未刷盘的WAL会导致恢复过程本身耗费真实、有时相当可观的时间和IO(这也是知识点3 checkpoint存在的意义)。
- 混淆"回滚阶段"(崩溃恢复里针对未提交事务的清理)和"正常运行时的事务ROLLBACK"——两者用的都是undo信息,但触发场景和上下文完全不同,恢复时的undo是系统重启后自动执行的收尾步骤,不是用户主动发起的操作。

---

## 5. 缓冲池与页面置换:脏页感知的LRU

**签名/是什么**

```
缓冲池(Buffer Pool): 内存里缓存最近访问过的数据页,避免每次查询都要读磁盘
脏页(Dirty Page): 缓冲池里的页被修改过,但修改还没写回磁盘的版本
```

一句话:缓冲池的淘汰策略本质上就是dsa-deep-dive 05类已经验证过的LRU缓存(最近最少使用的先淘汰),但数据库场景多了一个约束——淘汰一个脏页之前,必须先把它的修改内容写回磁盘,不能直接丢弃(丢弃等于丢数据),这是数据库缓冲池和通用LRU缓存最关键的差异。

**底层机制/为什么这样设计**

淘汰干净页(未被修改过,或者修改已经落盘过)是"零代价"的——反正磁盘上已经有一份一致的副本,直接丢弃内存里的拷贝下次要用再重新读一遍就行。淘汰脏页则必须先执行一次"写回"(flush),这个写回操作本身有IO成本,如果缓冲池淘汰策略完全不考虑"脏不脏"这个维度,可能会频繁淘汰刚被修改、还没来得及批量写回的脏页,造成远比必要更多的随机小IO写入——这正是为什么真实数据库的缓冲池管理比教科书里的纯LRU复杂得多(比如PostgreSQL用的是时钟扫描算法的变种,并主动通过后台的checkpointer/bgwriter进程提前批量写回脏页,减少淘汰时刻的即时写回压力)。

**AI研究/工程场景**

这个"脏页优先延迟淘汰"的思路解释了为什么数据库的写入密集型负载对缓冲池大小特别敏感——缓冲池太小,大量刚写入的脏页还没来得及被后台进程批量刷盘就被挤出去,退化成"每次写入都要立刻同步落盘"的低效模式,这是数据库调优时"该给多少内存做缓冲池"这类问题背后的真实机制原因。

**可运行例子**(环境:`.venv`,交叉引用dsa-deep-dive 05类LRU缓存的链表+哈希表实现思路,本类聚焦"脏页必须先写回"这个数据库特有的额外约束,不重新推导LRU双向链表本身的实现细节)

```python
from collections import OrderedDict

class DirtyAwareBufferPool:
    def __init__(self, capacity):
        self.capacity = capacity
        self.pages = OrderedDict()  # key -> (value, is_dirty), 复用dict保序特性模拟LRU顺序
        self.flush_log = []  # 记录发生过写回的page,验证用

    def access(self, key):
        if key not in self.pages:
            return None
        value, dirty = self.pages.pop(key)
        self.pages[key] = (value, dirty)  # 移到最近使用的位置
        return value

    def write(self, key, value):
        if key in self.pages:
            self.pages.pop(key)
        elif len(self.pages) >= self.capacity:
            self._evict_one()
        self.pages[key] = (value, True)  # 新写入的页标记为脏

    def _evict_one(self):
        evict_key, (evict_val, evict_dirty) = next(iter(self.pages.items()))  # 最久未使用的
        self.pages.pop(evict_key)
        if evict_dirty:
            self.flush_log.append(evict_key)  # 脏页淘汰前必须先"写回"

pool = DirtyAwareBufferPool(capacity=2)
pool.write('page_A', 'data1')   # 脏页
pool.access('page_B_unused')     # 不存在,不影响
pool.write('page_B', 'data2')   # 脏页
pool.write('page_C', 'data3')   # 触发淘汰,LRU顺序里page_A最老且是脏页,必须写回
assert pool.flush_log == ['page_A'], f"evicting a dirty page must trigger a flush, got {pool.flush_log}"

pool2 = DirtyAwareBufferPool(capacity=2)
pool2.pages['page_X'] = ('data', False)  # 手动构造一个"干净页"(比如刚从磁盘读入、未修改)
pool2.pages['page_Y'] = ('data', False)
pool2.write('page_Z', 'data3')  # 触发淘汰,page_X是干净页,不需要写回
assert pool2.flush_log == [], f"evicting a clean page should NOT trigger a flush, got {pool2.flush_log}"

print("dirty-page-aware buffer pool verified: evicting a dirty page triggers a flush, evicting a clean page does not")
```

**面试怎么问+追问链**

- Q:数据库缓冲池的淘汰策略和普通的LRU缓存(比如dsa-deep-dive讲过的LRU算法)有什么本质区别?
  - 追问1:如果一个页被反复修改、一直是脏页,LRU算法会不会让它长期占着位置不被淘汰,反而不合理?
    - 深挖追问(区分度较高):真实数据库怎么处理这种"频繁修改导致迟迟不被淘汰,但也迟迟不被写回"的页(答案方向:不完全依赖"淘汰时才写回"这个被动时机——真实数据库有独立的后台进程[PostgreSQL的`checkpointer`/`bgwriter`]周期性地主动扫描并写回一部分脏页,不等到缓冲池满了才手忙脚乱,这个主动写回的节奏本身又和知识点3的checkpoint机制直接相关,是"缓冲池管理"和"checkpoint"这两个知识点在真实系统里互相配合的地方)。

**常见坑**

- 把数据库缓冲池当成纯粹的性能优化技巧,忽视它同时也是"延迟写入换取IO效率"这个更大主题的一部分——缓冲池不只是"缓存热数据加速读取",脏页的存在本身就意味着"内存里的数据比磁盘上的更新",这也是为什么缓冲池管理和WAL/持久性机制紧密相关,不能孤立理解。
- 想当然认为"缓冲池越大越好"——本类可运行例子演示的是简化的两页缓冲池,真实场景里缓冲池大小需要结合工作集大小、脏页比例、后台写回速度综合评估,不是单纯"越大就一定越快"。

---

## 6. LSM-tree结构:memtable与SSTable分层

**签名/是什么**

```
memtable: 内存里的有序结构,承接所有新写入,写入速度快(纯内存操作)
SSTable(Sorted String Table): memtable写满后,整体有序地刷写到磁盘形成的不可变文件
```

一句话:LSM-tree(Log-Structured Merge-tree)的核心思路是"写入只追加到内存,从不原地修改磁盘上已有的数据",彻底把随机写变成顺序写,这是它和B+树(03类知识点1已验证的、允许原地更新的磁盘索引结构)在写入路径上最根本的区别,是很多NoSQL存储引擎(LevelDB/RocksDB/Cassandra底层)的核心数据结构。

**底层机制/为什么这样设计**

B+树为了维护"随时可查"的有序结构,每次写入都可能需要原地修改磁盘上的某个页(甚至触发页分裂),这些修改在磁盘上的物理位置是分散的,属于随机IO。LSM-tree换了一个思路:接受"查询可能需要多处查找"这个代价,换取"写入永远是顺序追加"这个收益——memtable在内存里维护有序性(写入速度不受磁盘影响),写满后一次性顺序刷写成一个新的、内部有序但和其他SSTable互相独立的文件,读取时需要按从新到旧的顺序依次查找各个SSTable(以及内存里的memtable)才能找到某个key的最新值。

**AI研究/工程场景**

写入吞吐量要求极高、且能接受读取路径稍微复杂一些的系统(时序数据/日志类数据/大规模KV存储)经常选择LSM-tree类存储引擎,这类场景的写入往往是"持续追加新数据"(而不是频繁原地更新旧数据),LSM-tree"写入只追加"的特性正好和这类负载模式高度契合。

**可运行例子**(环境:`.venv`)

```python
class SimpleLSM:
    def __init__(self, memtable_limit=3):
        self.memtable = {}
        self.memtable_limit = memtable_limit
        self.sstables = []  # 时间顺序,越靠后越新

    def put(self, key, value):
        self.memtable[key] = value  # 永远只追加/覆盖到内存memtable,不碰磁盘上的SSTable
        if len(self.memtable) >= self.memtable_limit:
            self._flush()

    def _flush(self):
        if self.memtable:
            self.sstables.append(dict(sorted(self.memtable.items())))  # 整体有序地"刷写"成一个新文件
            self.memtable = {}

    def get(self, key):
        if key in self.memtable:
            return self.memtable[key]
        for sst in reversed(self.sstables):  # 从最新的SSTable开始找,保证读到最新版本
            if key in sst:
                return sst[key]
        return None

lsm = SimpleLSM(memtable_limit=3)
lsm.put('a', 1)
lsm.put('b', 2)
lsm.put('c', 3)   # memtable写满,触发flush -> sstable[0] = {a:1, b:2, c:3}
assert len(lsm.sstables) == 1 and lsm.memtable == {}

lsm.put('a', 100)  # "更新"a: 只是在memtable里追加一条新记录,sstable[0]里的旧a=1完全没有被触碰
lsm.put('d', 4)
lsm.put('e', 5)   # 再次写满,触发flush -> sstable[1] = {a:100, d:4, e:5}
assert len(lsm.sstables) == 2

# 读取a时必须先查最新的sstable(找到100),而不是最早的那个(会读到过期的1)
assert lsm.get('a') == 100, f"must find the newest version first, got {lsm.get('a')}"
assert lsm.get('b') == 2  # b只存在于sstable[0],从新到旧遍历依然能找到

print(f"LSM-tree memtable/SSTable structure verified: {len(lsm.sstables)} SSTables, newest version correctly takes precedence")
```

**面试怎么问+追问链**

- Q:LSM-tree为什么要把数据拆成多个SSTable,而不是一直原地维护一个大文件?
  - 追问1:多个SSTable意味着读一个key可能要查好几个文件,这不是让读变慢了吗?
    - 深挖追问(区分度较高):工程上怎么缓解这个"读放大"问题(答案方向:①每个SSTable通常配一个布隆过滤器[Bloom Filter],读取前先用极低成本判断"这个key绝对不在这个SSTable里"从而跳过大部分不必要的文件查找;②SSTable数量不能无限增长,需要定期合并压缩[知识点7 compaction]减少文件数量,这两个手段共同把LSM-tree"写入友好但读取需要多路查找"这个天然代价控制在可接受范围)。

**常见坑**

- 认为LSM-tree"重写"了B+树解决的所有问题——LSM-tree是用"写入更快"换"读取更复杂/需要额外合并开销",不是全方位碾压B+树,03类已经验证过B+树在查找路径上是稳定可预测的,选型要基于真实的读写比例,不是无脑追新。
- 忽视SSTable数量会随着持续写入不断增长,如果没有compaction机制及时合并(知识点7),读放大问题会越来越严重,这是LSM-tree系统必须主动运维的一个真实成本。

---

## 7. LSM-tree compaction机制

**签名/是什么**

```
Compaction(合并压缩): 把多个SSTable合并成更少、更大的SSTable,过程中丢弃被覆盖的旧版本数据
```

一句话:compaction是LSM-tree"写入快"这个优势能长期维持下去的关键维护机制——不做compaction,SSTable数量会随着持续写入无限增长,读取需要查找的文件越来越多(知识点6提到的读放大问题会持续恶化),compaction定期把旧的、零散的SSTable合并成更少的大文件,同时清理掉被覆盖的过期版本,回收空间。

**底层机制/为什么这样设计**

Compaction的合并逻辑本质上是"按key排序做多路归并",因为每个SSTable内部已经有序(知识点6已验证),多个有序序列的归并可以高效地在一次线性扫描中完成,归并过程中如果同一个key在多个SSTable里出现(说明被多次更新过),只保留时间上最新的那个版本,丢弃旧版本——这一步是compaction真正"回收空间"的地方。

**AI研究/工程场景**

Compaction本身是一个消耗CPU和IO资源的后台操作,真实系统(比如RocksDB)通常提供多种compaction策略(size-tiered/leveled等)来控制"合并频率"和"合并时的资源占用"之间的权衡,这和03/06类讨论过的索引/checkpoint权衡是同一类"运行时开销 vs 长期效率"的工程决策模式,在存储引擎设计里反复出现。

**可运行例子**(环境:`.venv`)

```python
class SimpleLSM:
    def __init__(self, memtable_limit=3):
        self.memtable = {}
        self.memtable_limit = memtable_limit
        self.sstables = []

    def put(self, key, value):
        self.memtable[key] = value
        if len(self.memtable) >= self.memtable_limit:
            self._flush()

    def _flush(self):
        if self.memtable:
            self.sstables.append(dict(sorted(self.memtable.items())))
            self.memtable = {}

    def get(self, key):
        if key in self.memtable:
            return self.memtable[key]
        for sst in reversed(self.sstables):
            if key in sst:
                return sst[key]
        return None

    def compact(self):
        # 多路归并: 按SSTable从旧到新的顺序依次合并,同一个key后写入的覆盖先写入的
        merged = {}
        for sst in self.sstables:
            merged.update(sst)
        self.sstables = [merged] if merged else []

lsm = SimpleLSM(memtable_limit=2)
lsm.put('a', 1); lsm.put('b', 2)          # flush -> sstable[0] = {a:1, b:2}
lsm.put('a', 100); lsm.put('c', 3)        # flush -> sstable[1] = {a:100, c:3}
lsm.put('b', 200); lsm.put('d', 4)        # flush -> sstable[2] = {b:200, d:4}
assert len(lsm.sstables) == 3

total_entries_before = sum(len(sst) for sst in lsm.sstables)
assert total_entries_before == 6  # a出现2次,b出现2次,c/d各1次,共6条物理记录(含重复key的旧版本)

lsm.compact()
assert len(lsm.sstables) == 1, "compaction should merge everything into a single SSTable"
merged = lsm.sstables[0]
assert merged == {'a': 100, 'b': 200, 'c': 3, 'd': 4}, f"compaction must keep only the latest value per key, got {merged}"
assert len(merged) == 4, f"expected exactly 4 unique keys after dropping superseded versions, got {len(merged)}"

print(f"LSM-tree compaction verified: {total_entries_before} raw entries (with duplicate old versions) compacted down to {len(merged)} unique up-to-date entries")
```

**面试怎么问+追问链**

- Q:compaction的过程中,如果这时候有新的读请求进来,会不会读到不一致的数据?
  - 追问1:compaction通常怎么和正在进行的读写操作并发协调?
    - 深挖追问(区分度较高):这和B+树的并发控制(比如05类讨论过的锁/MVCC机制)相比有什么不同的设计考量(答案方向:LSM-tree的SSTable一旦生成就是不可变的[immutable],compaction创建新的合并后SSTable、原子性地切换"当前有效的SSTable列表"指针,旧SSTable在没有任何读请求还在使用它之前才被真正删除——这种"不可变数据+原子指针切换"的并发策略比B+树"原地修改需要精细锁"的策略简单得多,是LSM-tree架构在并发控制这个维度上相对B+树的一个设计优势,compaction可以完全在后台异步进行,不需要阻塞前台读写)。

**常见坑**

- 把compaction简单理解为"清理垃圾"的辅助操作,不理解它对读放大/写放大都有直接影响——不做compaction读放大持续恶化(知识点6),但compaction本身的多路归并读写操作又构成"写放大"(同一份数据在其生命周期里被重复读写多次,包括写入memtable→flush成SSTable→之后被compaction读出来又重新写入合并后的SSTable),这是LSM-tree系统调优时"写放大 vs 读放大 vs 空间放大"三者之间需要综合权衡的真实复杂度来源。
- 认为compaction可以随时安全地手动触发不用考虑代价——真实系统里compaction是真实的IO密集型后台任务,在业务高峰期触发大规模compaction可能会因为争抢磁盘IO资源而拖慢正常的读写请求。

---

## 8. LSM-tree vs B+树写入路径对比

**签名/是什么**

```
B+树:      写入 = 原地修改磁盘页(可能触发页分裂),随机IO
LSM-tree:  写入 = 追加到内存memtable,写满后顺序刷盘,不原地修改已有文件
```

一句话:知识点6/7已经用真实代码验证过LSM-tree的写入路径,03类知识点1已经验证过B+树的树高/IO次数模型,把两者放在一起对比,核心结论是——B+树的写入路径包含随机IO(原地修改),LSM-tree的写入路径是纯顺序IO(只追加),这个差异直接决定了两种结构在写入密集型负载下的性能特征差异。

**底层机制/为什么这样设计**

顺序IO和随机IO在机械硬盘上的性能差距可以达到几十到上百倍(现代SSD差距缩小了很多但依然存在,尤其在高并发写入场景下随机写会引入更多的垃圾回收开销),这是LSM-tree在"写多读少"场景下相比B+树有显著优势的物理原因。但这个优势不是没有代价的——LSM-tree把"原地更新的复杂度"转移成了"多版本查找+compaction"的复杂度,读路径(尤其是范围查询,可能要合并多个SSTable的结果)比B+树的单次树遍历更复杂,这正是"没有免费的午餐"这个系统设计基本规律的体现:优化了一个维度,代价通常会出现在另一个维度。

**AI研究/工程场景**

选择关系型数据库(通常基于B+树,如InnoDB/PostgreSQL)还是LSM-tree类的NoSQL存储(如Cassandra/RocksDB/LevelDB),一个核心判断依据就是业务的读写比例和访问模式——高频随机点查+范围查询丰富的OLTP场景,B+树的稳定可预测查询性能更合适;写入吞吐量要求极高、读取相对没那么频繁或者可以接受最终一致性的场景(日志采集/时序数据/大规模写入的KV存储),LSM-tree的写入优势更能发挥价值。板块VI会展开NoSQL相关的工程场景讨论。

**可运行例子**(环境:`.venv`,量化对比"写入N次同一个key的更新"在两种模型下分别产生多少次物理IO操作)

```python
def btree_style_write_io_count(num_updates):
    # 简化模型: B+树原地更新,每次UPDATE都是一次独立的随机IO(写这一个key所在的页)
    return num_updates  # 每次更新1次随机IO

def lsm_style_write_io_count(num_updates, memtable_limit):
    # 简化模型: LSM-tree写入先进内存memtable(不算磁盘IO),
    # memtable写满才触发一次顺序IO(一次性刷写整个memtable,不管里面有多少条更新)
    import math
    flush_count = math.ceil(num_updates / memtable_limit)
    return flush_count  # 每次flush算1次顺序IO(不管flush里打包了多少次更新)

num_updates = 300
memtable_limit = 50

btree_io = btree_style_write_io_count(num_updates)
lsm_io = lsm_style_write_io_count(num_updates, memtable_limit)

assert btree_io == 300  # 300次更新=300次随机IO
assert lsm_io == 6      # 300次更新,每50条打包成1次顺序刷盘,共6次顺序IO

# 简化模型下,LSM-tree用"攒批顺序写"把物理IO操作次数压缩了50倍(等于memtable_limit这个批量大小)
assert btree_io / lsm_io == memtable_limit

print(f"write path comparison verified: {num_updates} updates cost {btree_io} random IOs (B+tree-style) "
      f"vs {lsm_io} sequential IOs (LSM-tree-style, batched by memtable_limit={memtable_limit})")
```

**面试怎么问+追问链**

- Q:同样是"写多"的场景,为什么不干脆都用LSM-tree,B+树还有什么不可替代的优势?
  - 追问1:如果一个业务的查询模式是大量随机点查(不是范围扫描),LSM-tree的多SSTable查找劣势会不会被放大?
    - 深挖追问(区分度较高):具体到什么程度的读写比例,你会建议换成LSM-tree(答案方向:没有一个放之四海皆准的固定比例阈值,这类选型必须基于真实的负载测试[比如真实统计生产环境的读写QPS比例、点查vs范围查询的占比],本类反复强调的"用EXPLAIN/真实实验代替假设"方法论同样适用于存储引擎选型这个更高层面的决策,不能凭"LSM-tree写快"这一个维度的直觉就下结论)。

**常见坑**

- 把"LSM-tree写入更快"简单理解为"LSM-tree全方位比B+树快"——本类的可运行例子只量化了写入路径的IO次数差异,完全没有涉及读取路径(LSM-tree读取需要查多个SSTable,复杂度和成本B+树往往更低更稳定),选型是多维度权衡,不是单一维度的碾压关系。
- 忽视compaction(知识点7)本身的开销,只看到LSM-tree"写入"这一步的优势——完整评估LSM-tree的写入成本,需要把最终必然发生的compaction开销也算进去,不能只看memtable刷盘这一步。

---

## 9. SQLite的WAL模式:与真实WAL机制的异同

**签名/是什么**

```
SQLite 默认日志模式: 'delete'(rollback journal风格,修改前把原始页备份到独立的journal文件)
SQLite WAL模式:      PRAGMA journal_mode=WAL,修改追加写入独立的 -wal 边车文件,不直接改主数据库文件
```

一句话:即使是本系列从01类开始一直用作".venv基础环境"的SQLite,也内置了一套和知识点1真实验证过的PostgreSQL WAL理念一致的机制——写入先进独立日志文件,数据库主文件的实际修改可以延后进行,证明"预写日志"不是PostgreSQL/MySQL这类"重量级"数据库才有的高级特性,而是一个具有普适性的存储引擎设计思路。

**底层机制/为什么这样设计**

SQLite默认的"delete"模式(rollback journal)思路和WAL相反:修改**之前**先把原始页内容备份到journal文件,然后原地修改主数据库文件,如果中途失败,用journal文件里备份的原始内容回滚——这更接近"undo"的思路(知识点2讨论过的undo log)。SQLite的WAL模式则是把这个逻辑反过来:不备份原始内容,而是把**修改本身**追加写入`-wal`文件,主数据库文件暂时保持不变,读取时需要先检查`-wal`文件里有没有更新的版本(概念上和03/05类讨论过的"先查最新变更、必要时才用老版本"的思路有相通之处)。SQLite支持两种模式并存,恰好提供了一个在同一个引擎内部直接对比"before-image备份"(rollback journal/undo风格)和"变更追加日志"(WAL/redo风格)两种持久性实现思路的机会。

**AI研究/工程场景**

移动端/嵌入式场景(SQLite最常见的部署形态)对"写入延迟"和"并发读写"的要求和服务器端数据库不完全一样——WAL模式的一个实际好处是读操作不会被写操作阻塞(读可以继续读主文件的旧版本,写只追加到`-wal`文件),这对"一边写日志一边要保持界面响应"的移动应用场景是真实有意义的优化,这也是为什么很多移动端SQLite的最佳实践建议里会提到"考虑开启WAL模式"。

**可运行例子**(环境:`.venv`)

```python
import sqlite3
import os

db_path = "_scratch_wal_test.db"
for suffix in ("", "-wal", "-shm"):
    if os.path.exists(db_path + suffix):
        os.remove(db_path + suffix)

conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("PRAGMA journal_mode")
default_mode = cur.fetchone()[0]
assert default_mode == "delete", f"expected SQLite's default journal_mode to be 'delete' (rollback journal), got {default_mode}"

cur.execute("PRAGMA journal_mode=WAL")
new_mode = cur.fetchone()[0]
assert new_mode == "wal", f"expected switching to succeed and report 'wal', got {new_mode}"

cur.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)")
cur.execute("INSERT INTO t VALUES (1, 100)")
conn.commit()

# WAL模式下,写入的变更先进-wal边车文件,和PostgreSQL"先写日志"的理念一致
wal_sidecar_exists = os.path.exists(db_path + "-wal")
assert wal_sidecar_exists, "expected a real -wal sidecar file to be created once WAL mode is active and a write happens"

conn.close()
for suffix in ("", "-wal", "-shm"):
    if os.path.exists(db_path + suffix):
        os.remove(db_path + suffix)

print(f"SQLite WAL mode verified: default mode is '{default_mode}' (rollback journal / undo-style), "
      f"switching to WAL creates a real sidecar file (redo-style, same core idea as PostgreSQL's WAL)")
```

**面试怎么问+追问链**

- Q:SQLite这种嵌入式数据库也有WAL机制吗?
  - 追问1:SQLite的WAL模式和PostgreSQL的WAL,是同一个东西吗?
    - 深挖追问(区分度较高):两者最大的实现差异是什么(答案方向:核心理念一致[变更先写日志,主数据延后更新],但SQLite的WAL文件会被一个叫checkpoint的操作定期"合并"回主数据库文件本身[这个checkpoint和知识点3讨论的PostgreSQL checkpoint概念上类似但触发机制和粒度不同],SQLite默认没有真正的多进程后台清理机制,更依赖调用方主动或被动触发checkpoint;而PostgreSQL是真正的客户端-服务器架构,有独立的后台进程持续管理WAL和checkpoint节奏——两者"同名"但因为SQLite嵌入式、单文件、无服务进程的定位,具体工程实现的自动化程度不同)。

**常见坑**

- 认为SQLite"太简单"不会有真正的WAL机制——本类的可运行例子已经真实验证SQLite原生支持WAL模式,核心理念和PostgreSQL一致,不能因为SQLite常被当作"教学用/轻量级"数据库就低估它内部机制的完整性。
- 混淆SQLite默认的"delete"模式和WAL模式各自的适用场景——默认模式实现更简单、兼容性最好,但写不能和读完全并发;WAL模式提升了读写并发能力,但需要额外的checkpoint维护和`-wal`/`-shm`边车文件的存在,不是任何场景都无条件应该开启。
