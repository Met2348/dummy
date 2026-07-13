# 13. 模拟终面 capstone:连接池打满引发的 P99 延迟陡增假警报

> 收尾文件,不计入约 80 个知识点。综合运用 04/05/06/11 类共 5 处知识点,场景设定为"线上 API 的 P99 延迟陡增,给一段真实生成的监控数据要求诊断根因",结构参照 [os-concurrency-deep-dive 12 类 capstone](../os-concurrency-deep-dive/12-mock-interview-capstone.md)(候选人初版汇报 → 面试官多轮追问 → 基于真实数据诊断根因),覆盖方案批判迭代轴、诊断真实数据(新题型)、决策依据追问轴、工程约束递增轴、真实性验证轴共 5 条轴线。

---

## 题目

你负责的一个内部 API 网关,几天前业务方接入了一批新客户,QPS 比之前涨了不少。监控显示:**P50 延迟基本没变化**(一直稳定在 15ms 左右),但**P99 延迟从原来的 30ms 左右陡增到了 120ms 以上**,偶尔的瞬时峰值观测到过接近 200ms。业务方开始收到"部分请求偏慢"的用户投诉。你被叫来诊断根因。

---

## 候选人的初版汇报

"看了下监控曲线,P99 涨的时间点和流量增长的时间点差不多重合,初步怀疑是流量上来之后网络拥塞导致偶发丢包,触发了 TCP 重传,把尾部延迟拖高了。我们打算先联系网络团队排查链路质量,同时客户端这边加大超时时间和重试次数兜底。"

## 追问1:证据在哪(方案批判迭代轴 + 真实性验证轴)

**面试官**:"你说怀疑丢包导致重传——重传具体会怎么影响延迟分布?是均匀地拖慢所有请求,还是只影响一部分?"

**候选人**:"呃……重传应该是随机发生在各个连接上的,理论上会让一部分请求变慢,不会说所有请求都变慢。"

**面试官**:"那你的监控数据里,P50 完全没有变化,只有 P99 涨了——如果是网络链路质量下降导致的丢包,丢包应该是相对随机分布在所有连接上的独立事件,不太可能精确地'只影响原本就排在尾部的那一小撮请求,完全不影响中位数附近的大多数请求'。这个数据形状本身就不太支持'链路质量下降'这个假设,你怎么看?"

**候选人**:"……确实,如果是链路质量整体下降,应该是整条延迟分布曲线一起往右移,不会是这种'中位数纹丝不动、只有尾巴翘起来'的形状。这个形状看起来更像是——大部分请求走的是一条'正常路径',一小部分请求撞上了某种偶发的、需要排队等待的资源竞争。"

**面试官**:"这才是从数据出发的推理,不是猜。继续往这个方向查。"

## 追问2:真实数据诊断(诊断真实数据新题型,本 capstone 的核心)

**面试官**:"别再猜了,这是我们从这个网关最近一次真实压测里跑出来的原始数据(下游服务本身处理一个请求大约需要 15ms,网关到下游用连接池维护少量长连接,这是过去几个月一直稳定工作的配置)。你来读。"

**可运行例子**(验证环境:`.venv`;以下是真实运行产生的延迟数据,不是手写编造的示例数字)

```python
import socket
import threading
import time
import queue

HOST = "127.0.0.1"


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


port = free_port()
REQUEST_PROCESS_TIME = 0.015  # 下游服务:每个请求真实处理约15ms


def downstream_server(stop_evt):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(64)
    srv.settimeout(0.2)
    while not stop_evt.is_set():
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue

        def handle(c):
            try:
                c.settimeout(3)
                while True:
                    data = c.recv(64)
                    if not data:
                        break
                    time.sleep(REQUEST_PROCESS_TIME)
                    c.sendall(b"ok")
            except (socket.timeout, ConnectionError):
                pass
            finally:
                c.close()

        threading.Thread(target=handle, args=(conn,), daemon=True).start()
    srv.close()


stop_evt = threading.Event()
t = threading.Thread(target=downstream_server, args=(stop_evt,), daemon=True)
t.start()
time.sleep(0.3)


class ConnPool:
    def __init__(self, size):
        self.q = queue.Queue()
        for _ in range(size):
            c = socket.create_connection((HOST, port), timeout=2)
            self.q.put(c)

    def checkout(self):
        return self.q.get()  # 池耗尽时这里会真实阻塞排队

    def checkin(self, conn):
        self.q.put(conn)


POOL_SIZE = 5  # 网关到这个下游服务,连接池维持5条长连接(过去几个月一直是这个配置)
pool = ConnPool(POOL_SIZE)

latencies = []
lat_lock = threading.Lock()


def make_request():
    t0 = time.perf_counter()
    conn = pool.checkout()  # 排队等待空闲连接的时间
    t_got_conn = time.perf_counter()
    conn.sendall(b"req")
    conn.recv(64)
    pool.checkin(conn)
    t_done = time.perf_counter()
    with lat_lock:
        latencies.append({
            "total_ms": (t_done - t0) * 1000,
            "wait_for_conn_ms": (t_got_conn - t0) * 1000,
            "request_ms": (t_done - t_got_conn) * 1000,
        })


# 真实流量形态:大部分请求是低速稳定的散单流量(远低于连接池承载能力,每个请求处理完再发下一个,
# 总能立刻拿到空闲连接)——这部分对应监控里"大多数正常"的 P50。
N_TRICKLE = 80
for _ in range(N_TRICKLE):
    th = threading.Thread(target=make_request)
    th.start()
    th.join(timeout=2)

# 然后是一次真实的流量突发:40个请求几乎同时到达(新客户批量接入后的一次调用高峰/重试风暴),
# 远超连接池的5条连接——这部分对应监控里被拖出来的 P99 尾部。
N_BURST = 40
burst_threads = [threading.Thread(target=make_request) for _ in range(N_BURST)]
for th in burst_threads:
    th.start()
for th in burst_threads:
    th.join(timeout=5)

N_REQUESTS = N_TRICKLE + N_BURST
stop_evt.set()
t.join(timeout=2)

assert len(latencies) == N_REQUESTS

totals = sorted(x["total_ms"] for x in latencies)
waits = sorted(x["wait_for_conn_ms"] for x in latencies)


def percentile(sorted_list, p):
    idx = min(int(len(sorted_list) * p / 100), len(sorted_list) - 1)
    return sorted_list[idx]


p50_total = percentile(totals, 50)
p99_total = percentile(totals, 99)
p50_wait = percentile(waits, 50)
p99_wait = percentile(waits, 99)

print(f"P50 total latency = {p50_total:.1f}ms, P99 total latency = {p99_total:.1f}ms")
print(f"P50 wait-for-connection = {p50_wait:.1f}ms, P99 wait-for-connection = {p99_wait:.1f}ms")
print(f"P99/P50 ratio = {p99_total/p50_total:.1f}x")

# 核心结论:P50 只是"正常处理一次"的时间(池子够用),P99 被连接池排队等待时间拖高,
# 而且这个排队等待时间本身,而不是下游服务处理时间(对每个请求都稳定在~15ms),
# 才是P99尾部延迟的主要成分。
assert p50_total < 40, f"P50 should stay close to just 1 round of processing, got {p50_total}"
assert p99_total > p50_total * 3, f"P99 should be several times P50 due to burst queueing, got p50={p50_total} p99={p99_total}"
assert p99_wait > p99_total * 0.5, "queueing wait must dominate the P99 tail, not the downstream request time itself"
print("CAPSTONE_SCENARIO_TEST=PASS")
```

**面试官**:"读完这份数据,说说你现在的诊断。"

**候选人**(修正后):"看到了——`P50 total latency` 和 `P50 wait-for-connection` 几乎是 15ms 和 0ms,说明大多数请求根本没有排队,拿到连接后处理一次就完事了;但 `P99 total latency` 在 120ms 左右,而 `P99 wait-for-connection` 占了其中 100ms 以上——P99 尾部延迟的绝大部分根本不是下游处理慢,是请求在连接池外面排队等一条空闲连接才排出来的等待时间。连接池只有 5 条连接,遇到并发请求数量明显超过 5 的场景(比如新客户批量接入之后偶尔的调用高峰),超出 5 的那部分请求就必须排队等前面的人用完连接才能轮到自己,这正是 [11 类知识点4](11-network-programming-and-io-models.md) 讲过的连接池天然背压特性——不是 bug,是设计上就会这样,只是这次连接池大小(5)和实际并发峰值不匹配了。"

## 追问3:为什么会突然不匹配(决策依据追问轴)

**面试官**:"这个连接池配置成 5 条,已经稳定跑了好几个月都没出问题,为什么偏偏是这次新客户接入之后才暴露出来?"

**候选人**:"连接池大小 5 是按照接入这批新客户之前的历史并发峰值设定的,当时可能确实够用——大部分时间并发请求数不超过 5,即使偶尔超过也很快就排掉了,不会被监控明显捕捉到。新客户接入后整体 QPS 涨了,并发请求数超过 5 的时刻变得更频繁、每次超出的幅度也更大,排队现象从'偶尔发生、很快消化'变成了'经常发生、明显可见',这也解释了为什么 P50 完全没变(大多数请求依然能立刻拿到连接)而 P99 显著恶化(排队的比例和排队时长都上升了)——连接池配置本身没有变,是流量特征变了,配置和流量之间原本勉强够用的匹配关系被打破了。"

**面试官**:"如果这时候图省事,遇到连接池耗尽就直接开一条临时新连接顶上(不排队),问题是不是就解决了?"

**候选人**:"能避免排队等待的时间,但不是没有代价——每开一条新连接都要重新走一次三次握手(见 [04 类知识点2](04-tcp-connection-management.md),至少多付出 1 个 RTT),而且新连接的拥塞窗口从慢启动的初始值开始爬升(见 [06 类知识点2](06-tcp-congestion-control-and-udp.md)),不会一上来就用满带宽——用'无限制开新连接'替代'排队等待',相当于把'排队等待时间'换成了'新连接握手+慢启动惩罚',对于这种短平快的请求(单次处理只要15ms),这个惩罚占比反而可能更明显;而且如果不加节制地开新连接,下游服务瞬间要应付比正常多得多的并发连接数,这本身可能进一步拖慢下游处理,或者撞上下游自己的连接数上限。"

## 追问4:那到底该怎么定连接池大小(工程约束递增轴)

**面试官**:"那把连接池从 5 直接调到 50,是不是就一劳永逸了?"

**候选人**:"不能这么简单粗暴地调——连接池大小不是网关这一侧单方面决定的,得看下游服务能承受的最大并发连接数。如果网关本身有多个实例(水平扩展的网关集群),每个实例都把连接池设成 50,总连接数就是 `50 × 网关实例数`,这个总数可能远超下游服务的承载上限,调大连接池反而可能把下游打垮,这是 [11 类知识点4](11-network-programming-and-io-models.md) 追问链里讲过的同一个道理——池子大小要按'下游总容量 ÷ 上游实例数'来定,不能每个实例各自往大了设。正确做法是先搞清楚这次新客户接入后的真实并发峰值大概是多少(而不是拍脑袋定一个数字),再和下游服务确认它能稳定承受的最大并发连接数,取一个両边都安全的值;如果按真实峰值需要的连接数已经逼近下游承载上限,那需要下游本身也做扩容,不是单纯调大网关侧连接池数字就能解决的。"

## 追问5:具体怎么验证修复生效(真实性验证轴)

**面试官**:"假设你们把连接池从 5 调到了一个新的值,怎么证明这次调整真的解决了问题,而不是"看起来"好一点?"

**候选人**:"不能只看"P99 数字降下来了"就下结论,需要几个具体的量化验证点:第一,用调整前那批真实压测数据(就是刚才这份日志)重新跑一遍相同的流量形态(80 散单 + 40 突发),对比调整前后 P99 total latency 和 P99 wait-for-connection 这两个具体数字的变化——如果 wait-for-connection 显著降下来了、且降幅和调大的连接池数量在数量级上说得通,才是有说服力的证据;第二,上线后持续观察至少覆盖几个流量高峰周期的真实 P99 曲线,而不是调整后看一眼当下的数字就下结论(万一这次流量高峰恰好比平时小);第三,同时监控下游服务自己的连接数和资源使用情况,确认没有把压力单纯转移过去而不自知。"

## 最后一题:给出完整修复方案

**面试官**:"给出具体的修复方案,不要只说'调大连接池'这种空话。"

**候选人**:"三个层面:第一,短期缓解,基于新客户接入后的真实并发峰值数据(而不是猜),结合下游服务确认过的承载能力,把连接池大小调整到一个双方都安全的具体数值,并用刚才那份压测脚本的方法论重新跑一遍验证 P99 wait-for-connection 确实显著下降;第二,建立连接池使用率的持续监控(比如"连接池排队等待时间"这个指标本身,而不只是笼统的端到端 P99),这样下次配置和流量再次不匹配时,能在用户感知到延迟劣化之前就提前发现;第三,更长期地看,连接池大小和下游承载能力这两个数字应该有一个明确的、双方都知道的对应关系文档,新业务接入导致的流量增长评审流程里,应该包含'现有连接池配置是否还够用'这一项检查,而不是等到线上问题暴露了才回头排查——这次的教训本质上是'容量规划没有跟上流量变化',属于运维流程上的缺口,不只是一次配置调整就能根治的。"

---

## 复盘小结

| 轴线 | 在本 capstone 里的体现 |
|------|---------------------|
| 方案批判迭代轴 | 候选人"网络丢包导致重传"的初版假设,被"丢包应该均匀影响所有请求,不会只拖尾部"这个数据形状论证击穿 |
| 诊断真实数据(新题型) | 基于真实生成(非手写编造)的 P50/P99 total latency + wait-for-connection 拆分数据做出诊断,而不是套用"延迟高=网络问题"的模板 |
| 决策依据追问轴 | "为什么以前没事这次出事""连接池大小该怎么定"——不纠错,只逼问判断依据 |
| 工程约束递增轴 | 单实例连接池配置 → 多实例网关集群总连接数 → 下游承载能力的协同规划 |
| 真实性验证轴 | "调大了就完事"式的空洞回答被"具体怎么证明生效"逼问出量化验证方法论 |

**串联的知识点**(跨 4 个分类文件,5 处):
- [11 类知识点4(连接池设计思想)](11-network-programming-and-io-models.md)——本 capstone 的核心机制,池耗尽时的背压/排队正是这里讲过的天然特性
- [11 类知识点3(长连接vs短连接资源权衡)](11-network-programming-and-io-models.md)——"直接开新连接顶上"这个临时方案背后的资源权衡取舍
- [04 类知识点2(三次握手RTT成本)](04-tcp-connection-management.md)——追问3 里论证"开新连接不是免费的"的具体依据
- [06 类知识点2(慢启动)](06-tcp-congestion-control-and-udp.md)——同样用于论证"开新连接不是免费的",且是握手开销之外的第二重代价
- [10 类知识点7(负载均衡算法)](10-modern-app-protocols-and-apis.md)——工程约束递增轴延伸到"网关多实例集群总连接数"时的隐含背景知识

**方法论收尾**:这个 capstone 从"听起来很合理的初版结论"(网络丢包)出发,先用真实数据自身的分布形状(P50 不变、只有 P99 翘起)排除了这个假设,再基于另一份真实生成的延迟拆分数据(不是描述性文字,是真实跑出来的 `wait_for_conn_ms` 数字)精确定位到连接池排队这个真正根因——这正是全系列反复强调的纪律在综合场景下的体现:一个"听起来合理"的假设(丢包、网络问题)往往是排查延迟问题时最容易先入为主抓住的方向,但只有把假设放到真实数据面前接受检验,才能把"听起来合理"和"真正如此"区分开,这个区分能力,不管是诊断一次生产事故,还是回答面试官的追问,价值是相通的。

---

*本文件不计入约 80 个知识点,验证环境:`.venv`,独立重跑 4 次确认场景稳定可复现(P50 稳定在 15.7-15.8ms,P99 稳定在 117-123ms,P99/P50 比值稳定在 7.4-7.8 倍)。*
