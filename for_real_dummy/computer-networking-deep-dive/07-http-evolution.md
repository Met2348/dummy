# 07. HTTP 演进(板块 IV:应用层协议)

> 板块 IV 开篇。前三个板块讲的是"数据怎么送到对方机器",这个板块讲"应用程序之间怎么用送到的数据说话"——起点就是全世界用得最多的应用层协议 HTTP。本类沿着 HTTP/0.9 → 1.0 → 1.1 → 2 → 3 的演进主线,每一次版本跃迁都是"上一版本暴露了什么具体缺陷,下一版本用什么机制填坑",这也是面试里"为什么会有 HTTP/2"这类问题的标准答题框架。

**先画出整条演进时间线(每一步都是"填上一版本挖的坑",后面 KP1~KP6 逐段展开):**

```
HTTP/0.9 (1991)
  只有 GET,无版本号/请求头/状态码,发完数据直接关连接
  │
  │ 坑:图片/样式表等非 HTML 内容出现后,客户端不知道"这坨字节该怎么解析",
  │     也无法区分"成功"还是"资源不存在"
  ▼
HTTP/1.0 (1996,RFC 1945)
  +方法/请求头/状态码/Content-Type,但默认仍是"一个请求一个 TCP 连接"(见 KP1)
  │
  │ 坑:网页嵌入多张图片后,"每张图都要重新握手+慢启动"的开销迅速成为瓶颈
  ▼
HTTP/1.1 (1997)
  +默认持久连接(复用同一条TCP连接)、+管道化(不等响应就发下一个请求)(见 KP2)
  │
  │ 坑:管道化要求响应必须按请求顺序返回——一个慢请求会卡住排在它后面的
  │     所有请求(应用层队头阻塞),浏览器因此从未真正启用管道化
  ▼
HTTP/2 (2015)
  +二进制分帧+多路复用(一条连接内多个请求的帧交错发送,不再要求整体
  按序)、+HPACK首部压缩(见 KP5)
  │
  │ 坑:多路复用只解决了"应用层"HOL——底层仍是同一条TCP连接,TCP的
  │     字节流保序语义意味着一旦这条连接丢包,所有stream一起卡住等重传
  ▼
HTTP/3(2022年成为RFC 9114正式标准)
  彻底放弃TCP,改用自建可靠传输的QUIC(基于UDP,见 KP6)——每个
  stream独立做丢包检测和重传,一个stream丢包不再拖累其他stream
```

---

## KP1. HTTP/0.9 到 1.0:从"只有 GET"到"有头有body的真协议"

**签名/是什么:**

```
HTTP/0.9(1991):请求只有一行 "GET /path",没有版本号、没有请求头、没有状态码;
                响应只有裸的 HTML 字节流,发完就关连接。
HTTP/1.0(1996,RFC 1945):请求增加版本号("GET /path HTTP/1.0")、请求头(Host/User-Agent/...);
                响应增加状态行("HTTP/1.0 200 OK")、响应头(Content-Type/Content-Length/...)、
                状态码体系;但默认仍然是"一个请求一个 TCP 连接",除非显式带 Connection: keep-alive。
```

**一句话:** HTTP/0.9 只能取 HTML,HTTP/1.0 把它变成了一个真正有语义协商能力(方法/头部/状态码/内容类型)的协议,但连接管理仍然是"一锤子买卖"。

**底层机制/为什么这样设计:** 0.9 诞生于 Web 刚发明时,唯一场景是"浏览器问服务器要一个 HTML 文件",连方法字段都不需要——反正只有 GET 一种操作。但很快出现了图片、样式表等非 HTML 内容,以及"这个资源到底存不存在"这类需要表达失败语义的场景,于是 1.0 引入了 `Content-Type`(让客户端知道怎么解析响应体,而不是死认 HTML)和状态码(用数字分类结果:2xx 成功、4xx 客户端错、5xx 服务端错,不用解析自然语言错误信息)。但 1.0 每个请求仍然新开一个 TCP 连接——这在只取 1 个 HTML 文件时没问题,可一旦网页里嵌入多张图片(很快就发生了),"每张图都握手三次+慢启动从头来一遍"的开销迅速成为瓶颈,直接催生了 1.1 的持久连接(见 KP2)。

**AI 研究/工程场景:** 现代内部微服务和一些嵌入式/IoT 场景仍会手写极简的类 0.9/1.0 风格协议(没有连接复用需求、单请求单连接、省掉 HTTP 头部开销),因为在受控环境里连接建立成本可以忽略,而 HTTP 头部的字节开销在高频小包场景反而是负担——理解 0.9/1.0 的"极简设计"有助于判断什么时候可以裁剪协议到这个程度。

**可运行例子(验证环境:`.venv`):**

```python
import socket
import threading
import time
import http.client

HOST = "127.0.0.1"


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


port = free_port()


def raw_09_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(1)
    conn, _ = srv.accept()
    req = conn.recv(1024)
    assert req.startswith(b"GET"), req
    # HTTP/0.9: 没有状态行、没有响应头,直接回裸字节流,发完关连接。
    conn.sendall(b"<html>hello-09</html>")
    conn.close()
    srv.close()


t = threading.Thread(target=raw_09_server, daemon=True)
t.start()
time.sleep(0.2)
c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
c.connect((HOST, port))
c.sendall(b"GET /\r\n")  # 0.9 风格:没有版本号,没有请求头
resp = c.recv(4096)
c.close()
t.join(timeout=2)
assert resp == b"<html>hello-09</html>"
assert not resp.startswith(b"HTTP/"), "0.9 响应不应该有状态行"
print("0.9 raw response:", resp)

# HTTP/1.0:用标准库 http.client 发一个带版本号/请求头的请求,响应有状态行+响应头。
port2 = free_port()


def http10_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port2))
    srv.listen(1)
    conn, _ = srv.accept()
    req = conn.recv(4096)
    assert b"HTTP/1." in req, req  # 1.0 请求行带版本号
    body = b"hello-10"
    resp = (
        b"HTTP/1.0 200 OK\r\n"
        b"Content-Type: text/plain\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"\r\n" + body
    )
    conn.sendall(resp)
    conn.close()
    srv.close()


t2 = threading.Thread(target=http10_server, daemon=True)
t2.start()
time.sleep(0.2)
conn = http.client.HTTPConnection(HOST, port2, timeout=2)
conn.request("GET", "/", headers={"Connection": "close"})
r = conn.getresponse()
assert r.status == 200
assert r.read() == b"hello-10"
conn.close()
t2.join(timeout=2)
print("1.0 status:", r.status, "(has status-line+headers, 0.9 does not -> structural diff verified)")
```

**面试怎么问+追问链:**
- Q:HTTP/0.9 和 1.0 的本质区别是什么?
  - 追问1:1.0 为什么要引入状态码而不是让客户端自己解析响应体判断成功失败?(答案要落到"数字化、可编程判断 vs 需要理解自然语言/HTML 内容才能判断错误"这个区分度)
  - 深挖追问(方案批判迭代轴):"1.0 每个请求开一个新连接,为什么当时不直接设计成默认持久连接?"——需要答出历史语境:早期网页内容量小、服务器并发连接数是稀缺资源,直到网页内嵌资源数量暴涨后,持久连接的收益才明显超过其对服务器连接数的压力,这是渐进暴露问题、渐进填坑的过程,不是设计失误。

**常见坑:**
- 误以为 HTTP/1.0 默认就是持久连接——1.0 的持久连接是通过非标准的 `Connection: keep-alive` 头"事后追加"的扩展,不是规范默认行为,标准默认行为仍是一请求一连接;真正把持久连接定为默认行为的是 1.1(见 KP2)。

---

## KP2. HTTP/1.1 持久连接与管道化局限(队头阻塞)

**签名/是什么:**

```
持久连接(Persistent Connection):HTTP/1.1 默认复用同一条 TCP 连接发送多个请求/响应,
                                  除非显式带 Connection: close。
管道化(Pipelining):允许客户端不等第一个响应返回,就把第二个请求也发出去;
                    但服务器必须按请求到达顺序依次返回响应 —— 这是 HOL(Head-of-Line)阻塞的根源。
```

**一句话:** 持久连接解决了"每个请求都重新握手"的开销问题,但管道化为了保证响应顺序正确,把"一个慢请求"变成了"后面所有请求的拦路虎"。

**底层机制/为什么这样设计:** HTTP/1.1 规定管道化下响应必须按请求发出的顺序依次返回(不能先回后发的请求的响应),这是协议正确性的硬要求——否则客户端收到一堆无序响应体,完全没法知道哪个响应对应哪个请求(HTTP/1.1 报文本身不带请求 ID)。这个"必须顺序返回"的约束就是队头阻塞的直接原因:如果排在队首的请求处理慢(比如查询一个大表),即使排在它后面的请求早就处理完了,响应也必须攒在服务器手里等前面的先发完。这也是为什么大多数浏览器实际上默认关闭管道化、转而用"每个域名开 6~8 条并发连接"这种"并发多连接"而不是"单连接内管道化"的方式规避 HOL 阻塞——用连接数换并发度,治标不治本(连接数本身也是资源),真正的治本方案要等到 HTTP/2 在同一条连接内部做多路复用(见 KP5)。

**AI 研究/工程场景:** 这个"顺序保证 vs 并发效率"的矛盾是分布式系统里反复出现的母题——消息队列的严格顺序消费保证(如 Kafka 单分区)同样会遇到"一条消息处理慢阻塞后面所有消息"的问题,解决思路(增加并行度/连接数、放松顺序约束到某个更小的粒度)和这里高度同构,理解 HTTP/1.1 管道化 HOL 阻塞的本质,能直接迁移到理解消息队列积压问题。

**可运行例子(验证环境:`.venv`):**

```python
import socket
import threading
import time

HOST = "127.0.0.1"


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


port = free_port()
order_seen = []


def handle(conn):
    # 服务器严格按请求到达顺序处理并回复 —— 这就是真实 HTTP/1.1 管道化的响应顺序要求。
    buf = b""
    requests = []
    conn.settimeout(3)
    try:
        while len(requests) < 2:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\r\n\r\n" in buf:
                head, buf = buf.split(b"\r\n\r\n", 1)
                requests.append(head)
    except socket.timeout:
        pass
    for head in requests:
        path = head.split(b"\r\n")[0].split(b" ")[1]
        if path == b"/slow":
            time.sleep(0.5)  # 模拟慢请求(比如复杂查询)
            body = b"slow-body"
        else:
            body = b"fast-body"
        order_seen.append(path.decode())
        resp = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        conn.sendall(resp)
    conn.close()


def server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(1)
    conn, _ = srv.accept()
    handle(conn)
    srv.close()


t = threading.Thread(target=server, daemon=True)
t.start()
time.sleep(0.2)

c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
c.connect((HOST, port))
# 管道化:两个请求背靠背发出,不等第一个响应回来就发第二个。
c.sendall(b"GET /slow HTTP/1.1\r\nHost: x\r\n\r\n")
c.sendall(b"GET /fast HTTP/1.1\r\nHost: x\r\n\r\n")
t0 = time.perf_counter()
data = b""
c.settimeout(3)
while data.count(b"HTTP/1.1") < 2:
    chunk = c.recv(4096)
    if not chunk:
        break
    data += chunk
elapsed = time.perf_counter() - t0
c.close()
t.join(timeout=3)

slow_pos = data.find(b"slow-body")
fast_pos = data.find(b"fast-body")
assert slow_pos != -1 and fast_pos != -1
assert slow_pos < fast_pos, "尽管 fast 请求处理成本低,它的响应字节必须排在 slow 后面"
assert order_seen == ["/slow", "/fast"]
assert elapsed >= 0.5, f"客户端拿到完整数据的总耗时被 slow 请求拖累到 >= {elapsed:.3f}s"
print(f"order={order_seen}, elapsed={elapsed:.3f}s -> fast blocked behind slow (HOL blocking)")
```

**面试怎么问+追问链:**
- Q:什么是 HTTP 队头阻塞?
  - 追问1:管道化的队头阻塞和 TCP 层面的队头阻塞(一个丢包卡住整条连接后续所有字节,见 [05 类 KP1](05-tcp-reliability-and-flow-control.md) 累计确认机制)是同一个问题吗?
  - 深挖追问(区分度很高):不是同一层——管道化 HOL 是应用层协议规则造成的(响应必须按请求顺序返回),即使 TCP 传输完全没丢包也会发生;TCP 层 HOL 是传输层字节流保序语义造成的(一个报文段丢失,后续已经到达的报文段也不能交给应用层,必须等重传)。HTTP/2 的多路复用(KP5)解决了应用层 HOL,但复用的是同一条 TCP 连接,所以 TCP 层 HOL 依然存在——这正是 HTTP/3 弃用 TCP、改用 QUIC/UDP(KP6)的直接动机。这条追问链能不能答清楚"两层 HOL 是不同问题、HTTP/2 只解决了一层"是检验候选人是否真正理解演进逻辑还是背话术的分水岭。

**常见坑:**
- 把"HTTP/1.1 默认持久连接"和"HTTP/1.1 默认开启管道化"搞混——持久连接是默认行为,但管道化因为 HOL 阻塞和一些代理/服务器实现的兼容性问题,主流浏览器从未默认启用,实际生产环境几乎不会遇到真正的管道化流量。

---

## KP3. HTTP 方法/状态码语义(安全性与幂等性)

**签名/是什么:**

```
方法安全性(Safe):不会对服务端资源产生副作用的方法 —— GET / HEAD / OPTIONS。
方法幂等性(Idempotent):执行 1 次和执行 N 次,服务端最终状态相同的方法
                        —— GET / HEAD / PUT / DELETE / OPTIONS(POST 和 PATCH 通常不是)。
状态码类别:1xx 信息性、2xx 成功、3xx 重定向、4xx 客户端错误、5xx 服务端错误。
```

**一句话:** 安全性问"这个方法会不会改数据",幂等性问"重复调用会不会导致状态继续变化"——两者是正交的两个维度,GET 两者都满足,POST 两者都不满足,PUT 不安全但幂等。

**底层机制/为什么这样设计:** 这套语义存在的根本原因是让"中间设备"(浏览器、代理、CDN——内容分发网络,10类KP6会详细展开、重试框架)能够在不理解业务含义的前提下做出安全的自动化决策。比如浏览器的"前进/刷新"操作默认可以对 GET 请求静默重放,因为规范保证 GET 是安全的;但浏览器对 POST 请求重放前必须弹出"确认重新提交表单"的警告,因为 POST 没有幂等保证,重放可能导致重复下单。同理,网络库的自动重试机制通常只对幂等方法自动重试失败请求(比如连接超时后重试 GET/PUT 是安全的,但自动重试 POST 可能导致同一个操作被服务端执行两次)——这也是设计幂等 API(比如给创建类操作加 `Idempotency-Key` 请求头去重)在支付、下单等场景成为工程标配的原因:把本质上不幂等的 POST 操作,通过应用层的去重键改造成"事实上幂等"。

**安全性 × 幂等性画成一张 2×2 表(4 种组合里只有 3 种真实存在):**

|  | 幂等(重复调用,状态不再继续变化) | 不幂等(重复调用,状态持续变化) |
|---|---|---|
| **安全(不改数据)** | GET / HEAD / OPTIONS | 逻辑上不存在 |
| **不安全(会改数据)** | PUT / DELETE | POST / PATCH(通常) |

右上角这一格不是"暂时没找到例子",而是逻辑上就不可能存在:"安全"的定义就是完全不改变服务端状态,不改状态的操作不管重复调用多少次,结果显然还是"没有状态改变"——"安全"必然蕴含"幂等"。这也是为什么"安全性"和"幂等性"虽然是两个独立定义出来的概念,却不是完全对称、四格均可自由组合的两个维度。

**AI 研究/工程场景:** 设计 AI Agent 调用外部工具/API 的自动重试策略时,同样必须先判断该工具调用是否幂等——如果 Agent 因为网络抖动对一个"发送邮件"或"转账"的非幂等 POST 类工具自动重试,会导致真实的重复副作用;成熟的 Agent 框架会要求或建议为写类工具调用附带幂等键,这正是这套 HTTP 语义在 Agent 系统设计里的直接映射。

**可运行例子(验证环境:`.venv`):**

```python
import socket
import threading
import time

HOST = "127.0.0.1"


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


port = free_port()
counter = {"posts": 0, "resource": "v0"}


def server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(5)
    for _ in range(7):
        conn, _ = srv.accept()
        req = conn.recv(4096)
        method = req.split(b" ")[0].decode()
        if method == "POST":
            counter["posts"] += 1  # 非幂等:每次调用都累加副作用
        elif method == "PUT":
            counter["resource"] = "v1"  # 幂等:重复调用收敛到同一个最终状态
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        conn.close()
    srv.close()


t = threading.Thread(target=server, daemon=True)
t.start()
time.sleep(0.2)


def do(method):
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.connect((HOST, port))
    c.sendall(f"{method} /res HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n".encode())
    c.recv(4096)
    c.close()


# GET 安全:重复调用不触碰 counter。
for _ in range(2):
    do("GET")
assert counter["posts"] == 0 and counter["resource"] == "v0"

# POST 非幂等:3 次调用 -> 3 次累加副作用。
for _ in range(3):
    do("POST")
assert counter["posts"] == 3, counter

# PUT 幂等:重复调用最终状态保持一致。
do("PUT")
state_1 = counter["resource"]
do("PUT")
state_2 = counter["resource"]
assert state_1 == state_2 == "v1"

t.join(timeout=3)
print("GET safe(no side effect) / POST non-idempotent(3 calls accumulate to 3) / PUT idempotent(stable v1) -> all verified")
```

**面试怎么问+追问链:**
- Q:PUT 和 POST 都能用来"创建/更新资源",怎么选?
  - 追问1:如果我用 POST 实现一个"创建订单"接口,前端网络超时后自动重试了一次,会发生什么?怎么防?
  - 深挖追问(真实性验证轴):候选人如果回答"加个 Idempotency-Key",追问"这个 key 应该存在哪、去重窗口设多久、并发场景下两个携带相同 key 的请求同时到达怎么保证只有一个真正执行"——这是简历上"设计了幂等下单接口"这句话到底有没有真做过的试金石,答不出并发场景处理细节(通常需要数据库唯一索引或分布式锁配合)基本可以判断是概念性了解而非实操过。

**常见坑:**
- 认为"幂等 = 安全"——PUT 是幂等但不安全的典型反例(它会修改服务端状态,只是修改结果不随调用次数累积);混淆这两个概念会导致在设计缓存策略(只应该缓存安全方法)或自动重试策略(应该允许重试幂等方法)时犯错。

---

## KP4. HTTP 缓存机制(Cache-Control / ETag / Last-Modified)

**签名/是什么:**

```
Cache-Control: max-age=N       # 强缓存:N 秒内客户端直接用本地副本,不发请求
ETag: "<内容指纹>"              # 协商缓存标识:内容变化则指纹变化(通常是内容哈希)
If-None-Match: "<上次拿到的ETag>"  # 客户端带着上次的 ETag 问"内容变了吗"
304 Not Modified                # 服务端确认没变:响应头照常返回,但不重传响应体
Last-Modified / If-Modified-Since  # ETag 的时间戳版平替,精度只到秒,内容指纹不如 ETag 精确
```

**一句话:** 强缓存(`Cache-Control`)让客户端在有效期内完全不发请求,协商缓存(`ETag`)让客户端仍然发请求但服务端可以用 304 省掉重传响应体的带宽。

**底层机制/为什么这样设计:** 两级缓存分别针对两种不同的"确定性"设计:强缓存适合那种"生成时就知道多久不会变"的资源(比如带版本号哈希的静态 JS 文件 `app.a1b2c3.js`——内容一旦变了文件名本身就会变,所以旧文件名对应的内容永远不会变,可以让浏览器缓存到天荒地老);协商缓存适合"不确定多久会变,但每次都能快速判断变没变"的资源(比如 API 返回的数据),ETag 通常是内容的哈希值,只要服务端重新计算出的哈希和客户端带来的一致,就能 100% 确定内容没变,不需要传输真正的内容来比较,把"传输一整个响应体"的成本降级成"传输一个几十字节的哈希"。为什么还需要 `Last-Modified` 这个精度更低的备选方案?因为计算内容哈希本身有 CPU 成本,对于修改频率不高、用时间戳判断"大概率没变"就足够的资源,用文件系统自带的 mtime 比自己算哈希更省事,只是这个方案有精度到秒导致的"1 秒内多次修改会被误判为没变"的已知缺陷。

**AI 研究/工程场景:** LLM 推理服务对静态资源(模型权重文件、tokenizer 词表文件)的 CDN 分发普遍采用强缓存+内容哈希文件名的模式(类似 `model.a1b2c3.safetensors`),因为模型文件一旦发布不会再变、体积巨大,让边缘节点长期缓存能显著降低源站带宽压力;而模型服务的配置类接口(比如"当前可用模型列表")更适合 ETag 协商缓存,因为配置变化频率不确定但需要保证客户端总能拿到最新值。

**可运行例子(验证环境:`.venv`):**

```python
import socket
import threading
import time
import hashlib

HOST = "127.0.0.1"


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


port = free_port()
content = b"cacheable-body-xyz"
etag = hashlib.md5(content).hexdigest()


def server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(2)
    for _ in range(2):
        conn, _ = srv.accept()
        req = conn.recv(4096).decode(errors="replace")
        if f'If-None-Match: "{etag}"' in req:
            conn.sendall(b'HTTP/1.1 304 Not Modified\r\nETag: "' + etag.encode() + b'"\r\n\r\n')
        else:
            resp = (
                b'HTTP/1.1 200 OK\r\nETag: "' + etag.encode() + b'"\r\n'
                b"Content-Length: " + str(len(content)).encode() + b"\r\n\r\n" + content
            )
            conn.sendall(resp)
        conn.close()
    srv.close()


t = threading.Thread(target=server, daemon=True)
t.start()
time.sleep(0.2)

# 第一次请求:没有缓存,期望 200 + 完整响应体 + 服务端下发的 ETag。
c1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
c1.connect((HOST, port))
c1.sendall(b"GET /r HTTP/1.1\r\nHost: x\r\n\r\n")
r1 = c1.recv(4096)
c1.close()
assert b"200 OK" in r1 and content in r1

# 第二次请求:带上第一次拿到的 ETag 作为 If-None-Match,期望 304 + 空响应体。
c2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
c2.connect((HOST, port))
c2.sendall(f'GET /r HTTP/1.1\r\nHost: x\r\nIf-None-Match: "{etag}"\r\n\r\n'.encode())
r2 = c2.recv(4096)
c2.close()
t.join(timeout=3)
assert b"304" in r2, r2
assert content not in r2, "304 不应该重传响应体"
print(f"1st=200(+body {len(content)}B), 2nd=304(no body), etag prefix={etag[:8]}")
```

**面试怎么问+追问链:**
- Q:强缓存和协商缓存的区别,分别用在什么场景?
  - 追问1(决策依据追问轴):一个前端静态资源部署方案,为什么 `index.html` 本身通常设置成不缓存(`Cache-Control: no-cache`)或极短缓存,而它引用的 `app.[hash].js` 却可以设置成一年强缓存?
  - 深挖追问:因为 `index.html` 是整个"版本发现"链条的入口——如果连 `index.html` 都被强缓存,用户永远拿不到指向新版本 JS 文件哈希名的新 `index.html`,新版本就永远发布不出去;而 JS 文件名本身自带哈希,内容变化必然导致文件名变化,所以对它可以放心强缓存到最大程度。这个追问考察的是候选人是否理解"缓存策略要按整条更新链路设计,不是每个资源独立决定"。

**常见坑:**
- 认为 304 响应"节省了这次请求的所有开销"——实际上 304 仍然需要完整走一次 TCP(如果连接未复用)+ HTTP 请求/响应往返,只是省下了响应体的传输字节,请求本身的往返延迟(RTT)开销并没有省掉;真正连请求都不发的是强缓存命中,这是两者最容易被面试追问出的性能收益差异。

---

## KP5. HTTP/2 二进制分帧多路复用与 HPACK 首部压缩

**签名/是什么:**

```
二进制分帧(Binary Framing):HTTP/2 把每个请求/响应拆成若干"帧"(Frame),
                            每帧带 stream_id,同一条 TCP 连接上不同 stream 的帧可以交错发送。
多路复用(Multiplexing):多个请求/响应的帧在一条连接上并发交织传输,
                        不要求任何一个 stream 的帧连续,解决了应用层队头阻塞(对比 KP2)。
HPACK:HTTP/2 的首部压缩算法 —— 用"静态表+动态表"给重复出现的请求头做索引引用,
      重复头部只需传 1 个索引号,不用每次都传完整文本。
```

**一句话:** HTTP/2 把"一条连接一次只能顺序处理一个请求"的模型,改造成"一条连接内部按帧交织并发处理多个请求",从协议根子上解决了 KP2 那种应用层 HOL 阻塞。

**底层机制/为什么这样设计:** HTTP/1.1 的 HOL 阻塞根源是"响应必须整体、按顺序返回"这个粒度太粗的约束;HTTP/2 把约束粒度从"整个响应"细化到"一个帧",不同 stream 的帧可以自由交错,只要求同一个 stream 内部的帧要按序重组——这样一个慢请求只会拖慢自己的 stream,不会挡住其他 stream 的帧穿插传输。HPACK 首部压缩则针对另一个 1.1 时代被忽视的开销:现代网页一个页面可能发几十个请求,每个请求都带一整套几乎相同的请求头(Cookie、User-Agent、Accept-* 系列),1.1 下这些重复文本每次都要重新完整传输,HPACK 用双方维护的一张"索引表"把"再次出现的相同头部"压缩成一个字节的索引引用,首次出现才需要传完整文本——这个设计直接对应"重复数据用字典编码"这个通用压缩思路,只是应用在了 HTTP 头部这个具体场景。需要注意:HTTP/2 的多路复用解决的只是应用层 HOL,底层仍然是一条 TCP 连接,如果这条 TCP 连接本身发生丢包,TCP 的字节流保序语义(见 [05 类](05-tcp-reliability-and-flow-control.md))依然会让所有 stream 一起卡住等重传——这正是 HTTP/3 要彻底换掉 TCP 的原因(见 KP6)。

**AI 研究/工程场景:** gRPC(见 [10 类](10-modern-app-protocols-and-apis.md))直接构建在 HTTP/2 之上,大规模微服务集群/模型 serving 场景大量使用 gRPC 正是看中 HTTP/2 的多路复用能力——一个到某个模型服务实例的长连接可以并发承载大量并行推理请求,不需要像 HTTP/1.1 那样为了并发开一堆 TCP 连接,显著降低了连接数爆炸和相应的服务端资源开销。

**可运行例子(验证环境:`.venv`,教学性模拟——标准库无 HTTP/2 支持,用简化二进制帧格式演示多路复用与 HPACK 的核心结构,非真实 HTTP/2 实现;`struct.pack`格式字符串语法见01号知识点3,这里不重复解释):**

```python
import struct


def make_frame(stream_id, ftype, payload):
    # 简化帧格式:[stream_id:1字节][type:1字节][len:2字节][payload]
    return struct.pack("!BBH", stream_id, ftype, len(payload)) + payload


def parse_frames(buf):
    frames = []
    i = 0
    while i < len(buf):
        sid, ftype, length = struct.unpack("!BBH", buf[i:i + 4])
        payload = buf[i + 4:i + 4 + length]
        frames.append((sid, ftype, payload))
        i += 4 + length
    return frames


# 多路复用:stream 1(慢/大)和 stream 2(快/小)的帧交织在同一条"连接"上。
wire = b""
wire += make_frame(1, 0, b"stream1-part-A")
wire += make_frame(2, 0, b"stream2-COMPLETE")  # stream 2 全部帧先发完
wire += make_frame(1, 0, b"stream1-part-B-final")
frames = parse_frames(wire)
assert frames[1][0] == 2, "后开始的 stream 2 的帧可以插在 stream 1 完成之前"

# 按 stream_id 重组(这就是真实 HTTP/2 在帧层做的事)。
streams = {}
for sid, ftype, payload in frames:
    streams.setdefault(sid, b"")
    streams[sid] += payload
assert streams[2] == b"stream2-COMPLETE"
assert streams[1] == b"stream1-part-Astream1-part-B-final"
print("frame order (by stream_id):", [f[0] for f in frames], "-> stream2 completes without waiting on stream1")

# HPACK 风格模拟:重复请求头用索引号引用,而不是每次都传完整文本。
header_value = b":method: GET / :scheme: https / host: api.example.com"
raw_repeats = 5
raw_bytes = len(header_value) * raw_repeats  # HTTP/1.1 风格:每次都传完整文本

table = {}
encoded_bytes = 0
for _ in range(raw_repeats):
    if header_value in table:
        encoded_bytes += 1  # HPACK 命中:只传 1 字节索引
    else:
        table[header_value] = len(table)
        encoded_bytes += len(header_value)  # 首次出现:传完整文本

assert encoded_bytes < raw_bytes
saving_pct = 100 * (1 - encoded_bytes / raw_bytes)
assert saving_pct > 70, saving_pct
print(f"raw repeated={raw_bytes}B, hpack-sim={encoded_bytes}B, saving={saving_pct:.1f}%")
```

**面试怎么问+追问链:**
- Q:HTTP/2 怎么解决 HTTP/1.1 的队头阻塞?
  - 追问1:那 HTTP/2 是不是就彻底没有队头阻塞了?
  - 深挖追问(方案批判迭代轴,区分度很高):没有——HTTP/2 只解决了应用层 HOL,多个 stream 仍然共享同一条 TCP 连接,TCP 保证字节流顺序交付,一旦发生丢包,TCP 协议栈会让这条连接上所有 stream 的后续数据都在内核缓冲区里等重传,应用层完全不知情也无法绕过。这个"TCP 层 HOL"依然存在,是业界公认 HTTP/2 相对 1.1 的收益在弱网/高丢包环境下会打折扣的真实原因,也是 HTTP/3 直接放弃 TCP 改用 QUIC/UDP 的第一动机(见 KP6)。

**常见坑:**
- 把 HTTP/2 的多路复用和 HTTP/1.1 时代浏览器"对同一域名开 6~8 条并发连接"的并发模型混为一谈——后者是"多条连接各自独立处理",前者是"一条连接内部帧级别交织",连接数暴涨带来的握手开销、拥塞窗口重新增长(见 [06 类 KP2](06-tcp-congestion-control-and-udp.md) 慢启动)等问题只有 HTTP/2 的单连接多路复用能真正避免。

---

## KP6. HTTP/3 与 QUIC:为什么放弃 TCP

**签名/是什么:**

```
QUIC:构建在 UDP 之上的传输协议,自己实现可靠传输、拥塞控制、多路复用、加密(内置类似 TLS 1.3 的握手)。
HTTP/3:运行在 QUIC 之上的 HTTP 语义层,取代 HTTP/2 直接跑在 TCP 上的做法。
连接迁移(Connection Migration):QUIC 连接用一个独立的 Connection ID 标识,不绑定四元组,
                                  客户端网络切换(WiFi -> 4G)时连接不中断。
```

**一句话:** HTTP/3 把"传输层多路复用"从 TCP 手里彻底移交给自己掌控的 QUIC/UDP,一次性解决了 KP5 遗留的 TCP 层队头阻塞和握手延迟问题。

**底层机制/为什么这样设计:** HTTP/2 的根本局限在于它无法改变 TCP 的行为——TCP 的可靠字节流语义是操作系统内核实现的,应用层拿不到"跳过丢失的字节、先处理后面已到达的数据"这种控制权。QUIC 的解法是完全绕开内核 TCP 栈,在 UDP(内核几乎不介入的"傻瓜"传输层)之上,由应用层/用户态自己重新实现一套"多个独立 stream 各自可靠传输"的机制——每个 QUIC stream 有自己独立的丢包检测和重传,一个 stream 丢包只影响它自己,不会像 TCP 那样波及同一连接上的其他 stream,这才是真正彻底解决队头阻塞。另一个关键收益是握手延迟:TCP+TLS 分两次握手(先 TCP 三次握手,再 TLS 握手,见 [08 类](08-https-and-tls.md)),QUIC 把传输层握手和加密握手合并成一次(1-RTT,首次连接;0-RTT,曾经连接过的场景下客户端可以在第一个包里就带上加密的请求数据),减少了建立连接到发出第一个真实请求之间的往返次数。连接迁移能力则解决了 TCP"连接=四元组"带来的脆弱性——手机从 WiFi 切换到 4G 会导致 IP 地址变化,TCP 连接必然中断需要重新握手,QUIC 用独立于四元组的 Connection ID 标识连接,IP 变了连接可以无缝延续。

**AI 研究/工程场景:** 移动端 AI 应用(比如手机端语音助手持续和云端模型服务保持流式连接)天然是"网络环境频繁切换"的场景,QUIC 的连接迁移能力可以避免用户在 WiFi/蜂窝网络切换瞬间感受到语音流中断重连,这也是大厂移动端 API 网关近年积极推动 HTTP/3 落地的直接工程动机之一。

**可运行例子:** 本知识点为纯概念性讲解,不提供可运行代码。原因:QUIC 需要在 UDP 之上自行实现完整的可靠传输+拥塞控制+加密握手状态机,标准库没有内置支持,忠实模拟这套机制的工作量和价值不成正比(核心思想——"给每个 stream 独立的丢包检测与重传"——已经在 KP5 的分帧多路复用模拟中体现了实质,QUIC 只是把这个思想从应用层下沉到了自建的传输层);真实 QUIC 握手可以在 [08 类](08-https-and-tls.md)用 WSL2 `openssl` 观察 TLS 1.3 握手时对照理解(QUIC 内置的加密握手在流程结构上高度借鉴了 TLS 1.3 的 1-RTT 设计)。

**面试怎么问+追问链:**
- Q:为什么 HTTP/3 要放弃 TCP 改用 UDP?UDP 不是不可靠吗?
  - 追问1(决策依据追问轴):那 QUIC 在 UDP 之上重新发明了一套可靠传输,这和直接用 TCP 有什么本质区别,不是多此一举吗?
  - 深挖追问:本质区别在于"谁掌握可靠性实现的控制权"——TCP 的可靠性由操作系统内核实现,应用层只能整体使用或整体放弃,拿不到"这个 stream 丢包不要影响那个 stream"这种细粒度控制;QUIC 把可靠传输搬到用户态自己实现,粒度可以做到每个 stream 独立,这是 TCP 架构性做不到的能力,不是简单的重复造轮子。这条追问链的价值在于检验候选人是否理解"UDP 不可靠"和"QUIC 基于 UDP 自建可靠性"之间不矛盾,以及为什么这个自建的可靠性能比 TCP 更细粒度。
- 规模递增轴追问:单个请求场景下,QUIC 相对 TCP+TLS 的握手 RTT 节省可能只有 1 次往返,但如果是一个需要频繁短连接的场景(比如移动端 App 频繁唤醒后台轮询),这 1 次 RTT 的节省乘以连接次数,累积效应在弱网环境下会非常显著——这类问题考察候选人是否能把"协议层面的小优化"和"具体业务场景的累积收益"联系起来估算量级。

**常见坑:**
- 认为"QUIC 基于 UDP,所以 HTTP/3 不可靠"——UDP 本身不提供可靠性保证,但 QUIC 在其上层自己实现了完整的确认重传机制,可靠性并不比 TCP 差,某些场景(多 stream 独立丢包恢复)反而更优;"基于 UDP"只是说明它不使用内核 TCP 栈的可靠性实现,不代表协议整体不可靠。

---

## KP7. Keep-Alive 连接复用的工程含义

**签名/是什么:**

```
Connection: keep-alive     # HTTP/1.1 中为默认行为,显式声明可省略
Connection: close          # 显式要求本次响应后关闭连接
```

**一句话:** Keep-Alive 让同一个客户端到同一个服务器的多次请求复用同一条已建立的 TCP 连接,省掉的是重复的三次握手+慢启动开销,不是省请求本身的处理时间。

**底层机制/为什么这样设计:** 每新建一条 TCP 连接都要走三次握手(至少 1 个 RTT 延迟,见 [04 类 KP2](04-tcp-connection-management.md)),而且新连接的拥塞窗口从慢启动的初始值开始增长(见 [06 类 KP2](06-tcp-congestion-control-and-udp.md)),不会一上来就用满带宽——这意味着短连接不仅有握手延迟,还享受不到"连接已经跑热"之后的传输效率。对于同一个客户端短时间内向同一服务器发起多次请求的场景(比如一个页面加载引用了同域名下的多个 API),复用连接可以把 N 次握手开销压缩成 1 次,把 N 次慢启动压缩成 1 次(后续请求直接享用已经增长起来的拥塞窗口)。这也是为什么现代 HTTP 客户端库(包括 Python 标准库 `http.client.HTTPConnection`)默认支持连接复用,以及为什么服务端反向代理(Nginx 等)都会针对性做"连接池"优化,尽量减少到后端服务的新建连接次数。

**AI 研究/工程场景:** LLM 推理服务的客户端 SDK 内部普遍使用连接池(维护一批到模型服务的复用连接),而不是每次推理请求都新建连接——大模型单次推理本身耗时可能是几百毫秒到几秒,如果每次都重新握手,握手延迟占比虽然不算大头,但在高并发批量调用场景下,新建连接数暴涨会给服务端带来显著的连接管理开销(每条 TCP 连接都要占用内核的 socket 资源和内存),这是连接池成为高性能客户端标配的直接原因。

**可运行例子(验证环境:`.venv`,用服务端 `accept()` 调用次数直接证明连接是否被复用):**

```python
import socket
import threading
import time
import http.client

HOST = "127.0.0.1"


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


port = free_port()
accept_count = {"n": 0}


def server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(5)
    srv.settimeout(5)
    deadline = time.time() + 4
    while time.time() < deadline:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            break
        accept_count["n"] += 1  # 每次真正新建 TCP 连接,accept() 才会返回一次
        conn.settimeout(2)
        try:
            while True:
                req = conn.recv(4096)
                if not req:
                    break
                body = b"pong"
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: keep-alive\r\n\r\n" + body
                )
                if b"Connection: close" in req:
                    break
        except socket.timeout:
            pass
        conn.close()
    srv.close()


t = threading.Thread(target=server, daemon=True)
t.start()
time.sleep(0.2)

# 复用连接:5 次请求共用 1 个 http.client.HTTPConnection 对象 -> 期望只有 1 次 accept()。
conn = http.client.HTTPConnection(HOST, port, timeout=2)
for _ in range(5):
    conn.request("GET", "/ping")
    r = conn.getresponse()
    assert r.read() == b"pong"
conn.close()
time.sleep(0.3)
reused_accepts = accept_count["n"]
assert reused_accepts == 1, reused_accepts

# 不复用:3 次请求,每次都新建一个 HTTPConnection(=每次都重新三次握手)。
for _ in range(3):
    c = http.client.HTTPConnection(HOST, port, timeout=2)
    c.request("GET", "/ping", headers={"Connection": "close"})
    r = c.getresponse()
    assert r.read() == b"pong"
    c.close()
    time.sleep(0.05)
time.sleep(0.3)
fresh_accepts = accept_count["n"] - reused_accepts
assert fresh_accepts == 3, fresh_accepts
t.join(timeout=6)
print(f"reused-conn: 5 reqs triggered {reused_accepts} accept(); fresh-conn: 3 reqs triggered {fresh_accepts} accept()")
```

**面试怎么问+追问链:**
- Q:Keep-Alive 能带来多大的性能收益,具体省在哪?
  - 追问1(真实性验证轴):"具体省了多少,你量过吗?"
  - 深挖追问:经得起追问的回答需要拆成两块可测量的数字——① 省掉的握手 RTT(用 `tcpdump`/`ss` 或应用层计时,量出一次握手在当前网络环境下的真实耗时,见 [04 类 KP2](04-tcp-connection-management.md) 的真实抓包方法论);② 省掉的慢启动重新爬升时间(拥塞窗口从初始值重新增长到稳定值需要多少个 RTT,见 [06 类 KP2](06-tcp-congestion-control-and-udp.md)),这个收益在传输大文件时比小文件更明显。只会说"Keep-Alive 能提升性能"而给不出具体量化路径,通常说明只是背了结论没有真正测过。
- 规模递增轴追问:如果客户端和服务端之间有 N 个请求要发,持久连接把握手次数从 N 次降到 1 次,但如果 N 大到需要拆分成多条并发连接才能利用多核/避免 HOL 阻塞(回顾 KP2),Keep-Alive 的收益要怎么和"连接数"这个新变量一起权衡?——这是把本知识点和 KP2、[11 类连接池](11-network-programming-and-io-models.md) 串起来的自然追问方向。

**常见坑:**
- 以为 Keep-Alive 连接可以永久保持——服务端和客户端都会设置空闲超时(比如 Nginx 默认 `keepalive_timeout` 75 秒),超过这个时间没有新请求,连接会被服务端主动关闭;连接池设计必须处理"复用的连接可能已经被对端悄悄关闭"这种情况(发送时才发现连接已失效,需要捕获异常后重新建连接重试),这也是连接池实现里最容易被面试追问到的边界条件。

---

*本篇完成:2026-07-14,7 个知识点。验证环境:全部 6 个可运行代码块均为 `.venv`(HTTP/0.9 与 1.0 结构对比、管道化 HOL 阻塞真实复现、方法安全性/幂等性真实复现、ETag 协商缓存真实复现、二进制分帧多路复用+HPACK 教学性模拟、Keep-Alive 连接复用 accept() 计数证明);KP6(HTTP/3 与 QUIC)为纯概念性讲解,不提供代码(原因见该知识点"可运行例子"小节说明)。板块 IV(应用层协议)第 1 篇完成。*
