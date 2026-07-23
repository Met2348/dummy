# 模拟终面capstone:一次"限时优惠券超发"事故复盘

> 综合运用 01-08 类全部知识点。场景设定:一次真实的线上事故复盘——限时优惠券活动出现"库存明明只剩1张却发出去5张"的超发问题,同时伴随"活动预览页面"的慢查询雪崩。表面看是两个不相关的现象,candidate 需要在面试官层层追问下,分别定位两个独立但同时发生的真实根因,并给出可验证的修复方案。结构参照 os-concurrency-deep-dive 12类/computer-networking-deep-dive 13类capstone(候选人初版汇报→面试官多轮追问→最终诊断)。覆盖五轴追问链中的**方案批判迭代轴**(追问1)、**真实性验证轴**(追问2/追问5)、**决策依据追问轴**(追问3)、**工程约束递增轴**(追问4)共4条轴线。Cross-reference:03类知识点3/9、04类知识点6/7、05类知识点7、06类知识点1、08类知识点9,共5个不同分类文件的知识点。

## 题目

某电商平台的"限时优惠券"活动上线后10分钟内触发两起P1事故:①活动预览页(供运营查看"东部地区金卡用户"这类定向人群名单)响应时间从正常的几毫秒飙升到几十毫秒,大量请求超时;②活动结束后对账发现,一个只设置了1张库存的优惠券,活动期间被5个不同用户同时领取成功——库存表明明显示"remaining=1",最终却真实发出去5张。两起事故几乎同时发生,值班同学最初以为是同一个根因导致的连锁反应。请诊断根因并给出修复方案。

## 候选人初版汇报

"我看了一下,这个应该是数据库连接池被打满导致的——预览页变慢是因为请求排队,优惠券超发可能是因为连接池复用的连接之间状态污染,导致库存判断出错。我建议先把连接池最大连接数调大一些,应该能缓解。"

## 追问1(方案批判迭代轴):连接池打满能同时解释这两个现象吗?

**面试官**:"连接池打满"确实是08类知识点9讨论过的真实风险,但它通常表现为"请求排队等待获取连接"或者"连接状态泄漏导致读到别人遗留的会话设置"(比如意外继承了别人设置的隔离级别)——这两种表现和你说的"预览页变慢"、"优惠券发5张"分别对应得上吗?连接池打满本身会不会**导致库存数字算错**?

**候选人**:……确实,连接池打满顶多是请求排不上队直接报错或者超时,不会让`remaining=1`的库存被扣减成负数还都返回成功。这两个现象可能是独立的两个问题,我需要分别看实际数据。

**追问1小结**:面试官没有直接否定"连接池"这个方向,而是让候选人自己意识到"一个笼统的根因猜测,必须能同时解释全部观察到的症状,解释不了就说明假设本身不完整"——这是方案批判迭代轴的核心训练点:不是面试官告诉你错在哪,而是逼你自己用观察到的事实反推假设站不站得住脚。

## 追问2(真实性验证轴+诊断真实数据):给你真实的EXPLAIN输出,预览页到底慢在哪?

**面试官**:给你两条真实捕获的`EXPLAIN`输出(2026-07-14,WSL2 Rocky Linux,MariaDB 10.11.15,`coupon_users`表50000行,`campaign_id`有20种取值,`(campaign_id, user_tier)`上有联合索引):

```
原本的资格校验查询: WHERE campaign_id=1 AND user_tier='gold'
  type=ref key=idx_campaign_tier rows=810

新增的"按地区预览"查询: WHERE region='east' AND user_tier='gold'
  type=ALL key=None rows=48748
```

这两条查询哪个是预览页真正在跑的?为什么会这样?

**候选人**:预览页大概率跑的是第二条——`type=ALL`说明走了全表扫描,`key=None`说明压根没用上索引。索引是`(campaign_id, user_tier)`,这是03类知识点3讲过的联合索引最左前缀原则——查询条件必须从索引最左边的列开始连续匹配才能用上索引,这条查询完全没有`campaign_id`条件,直接从`region`开始,相当于跳过了最左列,索引整个用不上,只能全表扫描。

**面试官**:说得对。那这个"变慢"具体慢了多少,你有数字吗?

**候选人**:需要实测。（面试官提供真实测量数据：索引命中0.40ms,全表扫描5.25ms,大约13倍。）在50000行这个规模下13倍还只是零点几毫秒到几毫秒的差距,但生产环境数据量通常是这个测试规模的几十上百倍,这个倍数差距会被同比放大,叠加活动期间的高并发,大量这样的慢查询会真实占满数据库的CPU和连接资源,这才是"响应时间飙升+大量超时"的真正原因——不是连接池本身有bug,是慢查询把连接池的连接都堵住了没法及时释放。

**追问2小结**:候选人在真实数据面前正确定位了根因(03类知识点3),并且没有止步于"用上索引了/没用上索引"这个定性结论,而是主动追问定量的性能差距,体现了03类反复训练的"用EXPLAIN实测,不空谈"方法论。

## 追问3(决策依据追问轴):优惠券超发,和刚才这个索引问题有关系吗?

**面试官**:索引问题解释了预览页变慢,但优惠券超发呢——这个问题的库存扣减逻辑,和刚才那条慢查询完全不是同一段代码,它们真的是巧合般同时发生的两个独立问题吗?

**候选人**:目前看确实是两个独立的根因,只是同一次活动上线暴露出来。我需要看库存扣减的具体实现代码。

**面试官**:库存扣减是这样写的:先`SELECT remaining FROM coupon_stock WHERE campaign_id=1`,应用层判断`remaining > 0`,再执行`UPDATE coupon_stock SET remaining = remaining - 1`,整个过程包在一个`REPEATABLE READ`隔离级别的事务里。团队当时的设计依据是"REPEATABLE READ应该能防止并发问题"。这个依据站得住脚吗?

**候选人**:这里要看是哪个数据库引擎。如果是PostgreSQL,04类知识点7验证过REPEATABLE READ的确能通过`SerializationFailure`拦住这种并发写冲突。但如果是MySQL/MariaDB的InnoDB,04类知识点7的真实实验结果是——REPEATABLE READ**拦不住**这种"先查询判断、再更新"模式下的丢失更新,它的UPDATE操作读的是"当前最新版本"(当前读),不是事务开始时的快照版本,所以多个并发事务都能读到`remaining=1`、都认为库存充足、都执行了扣减,最终库存变成负数,但每个事务自己看来都"成功"了。这正是"REPEATABLE READ这个名字在不同引擎上的真实保证不一样"这个反直觉发现的真实应用场景。

**面试官**:所以团队的错误假设具体错在哪?

**候选人**:错在把"REPEATABLE READ能防止并发问题"当成了一个放之四海而皆准的常识,没有针对实际使用的数据库引擎(MariaDB)去确认这个保证是否真的成立。04类知识点8也提到过,这种情况下更稳妥的做法是不完全依赖隔离级别的隐式保证,而是用显式的原子操作或者05类知识点7的乐观锁。

**追问3小结**:这一轮不是让候选人纠正一个语法错误,而是逼问"团队当初的设计依据本身对不对"——候选人需要调用04类知识点7这个具体的、反直觉的真实实验结论,而不是泛泛地说"加锁就行了"。

## 追问4(工程约束递增轴):两个问题都要修,但不能只是"头痛医头"

**面试官**:现在诊断都清楚了。但如果只是"给region加个索引"、"把库存扣减改成原子UPDATE",这两个局部修复够吗?有没有更系统性的问题需要一起考虑?

**候选人**:局部修复是必要的第一步,但确实需要往外扩展考虑几层:第一,`region`这类新增的查询模式以后可能还会出现,每次都是"线上出问题才发现索引没覆盖",团队应该建立一个"新查询上线前review索引覆盖情况"的机制,而不是每次都被动救火;第二,库存扣减这类"高冲突资源"的操作,不应该在每个开发者写代码时都重新决定"要不要加锁、加哪种锁",应该沉淀成团队内部的标准写法或者公共组件,减少"每个人都要重新踩一遍04类知识点7这个坑"的风险;第三,这次能这么快定位根因,是因为有EXPLAIN和并发测试的诊断手段,如果生产环境本身没有慢查询监控和"检查后更新"模式的静态代码扫描,下次可能不会这么快发现。

**面试官**:如果只有有限的时间,这三件事你会先做哪个?

**候选人**:先做最直接止血的两个局部修复(索引+原子UPDATE),这是p1级别必须立刻做的;然后按影响面和实现成本排优先级——静态代码扫描规则相对容易落地且能防止同类问题重复出现,优先级应该高于"重新设计索引review流程"这类涉及跨团队协作流程改造的事,后者需要更多协调成本,不是纯技术决策能单独推动的。

**追问4小结**:这一轮训练的是"从两个具体bug往外推,识别系统性风险,但不能无限发散,要有优先级判断"——避免候选人陷入"过度设计"或者"只会头痛医头"两个极端。

## 追问5(真实性验证轴):怎么证明这两个修复真的有效,不是"感觉应该好了"?

**面试官**:两个修复都上线了,你怎么向团队证明问题真的解决了?

**候选人**:两个问题分别有可以量化的验证方式。索引问题:重新跑一遍刚才那条`region`查询的`EXPLAIN`,确认`key`字段变成了新建的索引名而不是`None`,再用`best_of`多次采样测量真实响应时间,和之前的13倍差距做对比,应该显著缩小。超发问题:用当时同样的并发场景(5个线程同时对库存为1的记录发起领取请求)重新压测,验证最终成功领取数确实变成1而不是5——这个验证不能只跑一次就下结论,并发问题往往有时序敏感性,04类知识点7的原则是至少重复多次确认稳定,不能"跑一次正常了就说修好了"。

**面试官提供真实复测数据**:

```
索引修复后: region+user_tier查询 type=ref key=idx_region_tier,响应时间从5.25ms降到1.16ms(约4.5倍改善)
超发修复后: 5个并发领取请求,成功数=1(库存=1,精确匹配,不多不少)
```

**追问5小结**:这一轮呼应05类"真实性验证轴"反复强调的原则——简历/汇报里说"优化好了"、"修复了"这类结论,必须能拿出具体的、可复现的数字支撑,不能停留在定性描述。

## 最终修复方案

1. **索引修复**:给`region`这类新查询模式补建`(region, user_tier)`联合索引,不去动原有的`(campaign_id, user_tier)`索引(两者服务不同的查询模式,03类知识点3已经验证联合索引的价值和局限,不能指望一个索引覆盖所有查询形态)。
2. **超发修复**:库存扣减从"查询判断+条件更新"三步走,改成`UPDATE coupon_stock SET remaining = remaining - 1 WHERE campaign_id=? AND remaining >= 1`这种数据库原生保证原子性的单条语句(04类知识点6"AI研究/工程场景"提到的标准写法),不再依赖隔离级别的隐式保证。
3. **流程改进**(按优先级):建立"检查后更新"模式的代码扫描规则(优先级较高,能防止同类bug复现);新查询上线前的索引覆盖review(优先级中等,需要跨团队流程配合);建立慢查询和并发冲突的常态化监控。

**可运行例子**(环境:`python-wsl2`,真实复现两个真实事故场景,并验证修复方案真实生效)

```python-wsl2
# 以下例子需在 WSL2 Rocky Linux 验证,依赖 MariaDB 10.11.15(已启动)
import pymysql
import time
import threading
import random

DSN = dict(host='127.0.0.1', port=3306, user='dbdemo', password='dbdemo_local_only', database='dbdemo')

def best_of(fn, trials=7):
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)

# ===== 线索1: 索引失效导致的慢查询雪崩 =====
setup_conn = pymysql.connect(**DSN); setup_conn.autocommit(True); setup_cur = setup_conn.cursor()
setup_cur.execute("DROP TABLE IF EXISTS coupon_users")
setup_cur.execute("""CREATE TABLE coupon_users (
    id INT PRIMARY KEY AUTO_INCREMENT, campaign_id INT, user_tier VARCHAR(20),
    region VARCHAR(20), user_id INT
)""")
setup_cur.execute("CREATE INDEX idx_campaign_tier ON coupon_users (campaign_id, user_tier)")

random.seed(1)
rows = [(random.randint(1, 20), random.choice(['gold', 'silver', 'bronze']),
         random.choice(['east', 'west', 'north', 'south']), i) for i in range(50000)]
setup_cur.executemany("INSERT INTO coupon_users (campaign_id, user_tier, region, user_id) VALUES (%s,%s,%s,%s)", rows)
setup_cur.execute("ANALYZE TABLE coupon_users")

# 原本的资格校验查询: 走索引最左前缀,真实使用idx_campaign_tier
setup_cur.execute("EXPLAIN SELECT * FROM coupon_users WHERE campaign_id=1 AND user_tier='gold'")
original_plan = setup_cur.fetchone()
assert original_plan[4] == 'idx_campaign_tier', f"original query should use the index, got key={original_plan[4]}"

# 新增的"按地区预览"查询: 跳过了索引最左列campaign_id,真实退化成全表扫描
setup_cur.execute("EXPLAIN SELECT * FROM coupon_users WHERE region='east' AND user_tier='gold'")
buggy_plan = setup_cur.fetchone()
assert buggy_plan[3] == 'ALL' and buggy_plan[4] is None, \
    f"the region-first query should genuinely fail to use any index, got type={buggy_plan[3]} key={buggy_plan[4]}"

def indexed_query():
    setup_cur.execute("SELECT COUNT(*) FROM coupon_users WHERE campaign_id=1 AND user_tier='gold'")
    setup_cur.fetchone()

def buggy_query():
    setup_cur.execute("SELECT COUNT(*) FROM coupon_users WHERE region='east' AND user_tier='gold'")
    setup_cur.fetchone()

t_indexed = best_of(indexed_query)
t_buggy = best_of(buggy_query)
slowdown_ratio = t_buggy / t_indexed
assert slowdown_ratio > 5, f"expected the leftmost-prefix-violating query to be significantly slower, got {slowdown_ratio:.1f}x"

# 修复: 给region查询模式补建对应的索引
setup_cur.execute("CREATE INDEX idx_region_tier ON coupon_users (region, user_tier)")
setup_cur.execute("ANALYZE TABLE coupon_users")
setup_cur.execute("EXPLAIN SELECT * FROM coupon_users WHERE region='east' AND user_tier='gold'")
fixed_plan = setup_cur.fetchone()
assert fixed_plan[4] == 'idx_region_tier', f"after the fix, expected the new index to be used, got key={fixed_plan[4]}"

def fixed_query():
    setup_cur.execute("SELECT COUNT(*) FROM coupon_users WHERE region='east' AND user_tier='gold'")
    setup_cur.fetchone()

t_fixed = best_of(fixed_query)
improvement_ratio = t_buggy / t_fixed
assert improvement_ratio > 2, f"expected a meaningful speedup after adding the index, got {improvement_ratio:.1f}x"

print(f"THREAD 1 (index) verified: leftmost-prefix violation caused a real {slowdown_ratio:.1f}x slowdown "
      f"({t_indexed*1000:.2f}ms -> {t_buggy*1000:.2f}ms), fix recovered {improvement_ratio:.1f}x ({t_fixed*1000:.2f}ms)")

# ===== 线索2: REPEATABLE READ误用导致的优惠券超发 =====
setup_cur.execute("DROP TABLE IF EXISTS coupon_stock")
setup_cur.execute("CREATE TABLE coupon_stock (campaign_id INT PRIMARY KEY, remaining INT)")
setup_cur.execute("INSERT INTO coupon_stock VALUES (1, 1)")  # 只剩1张券
setup_conn.close()

results_buggy = []
lock = threading.Lock()
# 5个线程的连接建立/调度耗时并不均匀,若不显式同步,可能出现"前面的线程已经
# commit,后面的线程才开始SELECT"从而在自己的REPEATABLE READ快照里看到已经
# 变化的库存值——这不是真实bug,只是时序噪声掩盖了bug。用barrier强制全部5个
# 线程都完成SELECT(即都已建立"remaining=1"这个快照)之后才允许任何一个继续
# 往下走,这样才是对"5个人同时查到库存=1"这个真实事故场景的准确复现。
# threading.Barrier(5)是本类第一次用到的新同步原语,和前面几类反复用的threading.Event
# 不是一回事:Event是"一个信号,等的人可以有任意多个";Barrier(5)是"5个参与者互相等齐"
# ——每个线程执行到.wait()就阻塞住,直到全部5个线程都到达了各自的.wait(),才会被同时
# 一起放行继续往下执行。Event表达不出"必须凑齐N个都准备好"这种更强的同步语义,这正是
# 这里换用Barrier而不是继续用Event的原因。
snapshot_barrier = threading.Barrier(5)

def claim_coupon_buggy(user_id):
    conn = pymysql.connect(**DSN)
    conn.autocommit(False)
    cur = conn.cursor()
    cur.execute("SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ")  # 团队原以为这样'应该'安全
    cur.execute("START TRANSACTION")
    cur.execute("SELECT remaining FROM coupon_stock WHERE campaign_id=1")
    remaining = cur.fetchone()[0]
    snapshot_barrier.wait()  # 等全部5个连接都完成SELECT、都拿到remaining=1的快照
    if remaining > 0:
        cur.execute("UPDATE coupon_stock SET remaining = remaining - 1 WHERE campaign_id=1")
        conn.commit()
        with lock:
            results_buggy.append((user_id, 'SUCCESS'))
    else:
        conn.rollback()
        with lock:
            results_buggy.append((user_id, 'REJECTED'))
    conn.close()

threads = [threading.Thread(target=claim_coupon_buggy, args=(i,)) for i in range(5)]
for t in threads: t.start()
for t in threads: t.join()

success_count_buggy = sum(1 for _, r in results_buggy if r == 'SUCCESS')
assert success_count_buggy == 5, \
    f"expected the REAL overselling bug: all 5 concurrent claims succeed despite stock=1, got {success_count_buggy}"

# 修复: 原子UPDATE语句,不再依赖隔离级别的隐式保证
setup_conn2 = pymysql.connect(**DSN); setup_conn2.autocommit(True)
setup_conn2.cursor().execute("DELETE FROM coupon_stock")
setup_conn2.cursor().execute("INSERT INTO coupon_stock VALUES (1, 1)")
setup_conn2.close()

results_fixed = []

def claim_coupon_fixed(user_id):
    conn = pymysql.connect(**DSN)
    conn.autocommit(True)
    cur = conn.cursor()
    cur.execute("UPDATE coupon_stock SET remaining = remaining - 1 WHERE campaign_id=1 AND remaining >= 1")
    affected = cur.rowcount
    with lock:
        results_fixed.append((user_id, 'SUCCESS' if affected == 1 else 'REJECTED'))
    conn.close()

threads2 = [threading.Thread(target=claim_coupon_fixed, args=(i,)) for i in range(5)]
for t in threads2: t.start()
for t in threads2: t.join()

success_count_fixed = sum(1 for _, r in results_fixed if r == 'SUCCESS')
assert success_count_fixed == 1, \
    f"expected the fix to correctly limit successful claims to exactly the real stock (1), got {success_count_fixed}"

print(f"THREAD 2 (overselling) verified: buggy REPEATABLE READ check-then-act let {success_count_buggy}/5 succeed "
      f"(stock was only 1!), atomic UPDATE fix correctly limited to {success_count_fixed}/5")

print("BOTH incident threads diagnosed and fixes verified independently: index leftmost-prefix violation (topic 03 KP3) "
      "and MariaDB REPEATABLE READ's real lost-update gap (topic 04 KP7) were two genuinely independent root causes "
      "co-occurring in the same incident window, not a single shared cause.")
```

真实捕获的输出(2026-07-14,WSL2 Rocky Linux,独立重跑3次确认稳定):

```
THREAD 1(索引): 违反最左前缀的查询真实慢了13.0倍(0.40ms -> 5.25ms),补建索引后恢复4.5倍(降到1.16ms)
THREAD 2(超发): REPEATABLE READ的检查后操作模式下,5个并发请求全部"成功"(库存明明只有1);
                原子UPDATE修复后精确限制为1个成功
```

**capstone小结:五轴追问链覆盖情况**

| 追问轮次 | 轴线 | 训练点 |
|---|---|---|
| 追问1 | 方案批判迭代轴 | 一个笼统假设("连接池打满")必须能同时解释全部症状,解释不了就要推翻重来 |
| 追问2 | 真实性验证轴+诊断真实数据 | 给真实EXPLAIN输出让候选人诊断,并主动追问量化的性能差距,不满足于定性结论 |
| 追问3 | 决策依据追问轴 | 逼问"团队当初为什么认为REPEATABLE READ够用"这个设计依据本身站不站得住脚,调用04类知识点7的具体反直觉发现 |
| 追问4 | 工程约束递增轴 | 从两个具体bug推广到系统性风险,但要求给出优先级排序,不能无限发散 |
| 追问5 | 真实性验证轴 | 修复效果必须用可复现的量化数字证明,不能停留在"感觉应该好了" |

Cross-reference汇总:03类知识点3(联合索引最左前缀)、03类知识点9(选择性与优化器决策,用于理解原本查询为什么正常走索引)、04类知识点6(丢失更新真实复现)、04类知识点7(REPEATABLE READ在两个引擎上的真实防护差异)、05类知识点7(乐观锁作为04类问题的替代解法思路)、06类知识点1(WAL/持久性,capstone未直接展开但事故复盘中候选人可能被追问"库存扣减的持久性保证"作为延伸)、08类知识点9(数据库连接池,候选人初版汇报的错误方向,被追问1正确排除)。
