# 14 · 手把手实战:从零搭一个迷你 HTTP 服务器

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 14 个"知识点",不计入"约 80 个知识点"的统计——和 [13 类](13-mock-interview-capstone.md)模拟终面 capstone 是同一挂(收尾板块的加餐),但风格不一样:13 类里,你是**旁观者**,跟着面试官和候选人的对话把"连接池打满引发 P99 延迟陡增"这条推理链条看一遍;这一篇里,你是**动手的人**——从一个空文件开始,一步步敲代码,每写一段就跑一次、看到真实字节在真实 socket 里收发,最后独立搭出一个能被标准库 `http.client` 真实连上的迷你 HTTP 服务器。

## 为什么是"HTTP 服务器"

不是要发明新知识点,是把几个你已经学过的知识点串成一个真实有用的东西——而且刻意不用 `http.server`/Flask 这类现成框架,因为框架恰恰把"怎么从 socket 原始字节里解析出一个 HTTP 请求、又怎么拼出一个合法响应"这一步封装藏了起来,这正是本篇要亲手拆开看的部分:

| 阶段 | 要让程序多会一件事 | 建立在哪个已有知识点之上 |
|------|------|------|
| 阶段 1 | 接一个连接,看懂请求行(method/path/version),回一个写死的合法 200 响应 | [04 类知识点 7](04-tcp-connection-management.md) socket 编程基础(bind/listen/accept/connect 阻塞语义) |
| 阶段 2 | 同一个服务器,不同 path 返回不同内容(路由)+ 解析 headers | [07 类 KP1](07-http-evolution.md) HTTP 请求行/headers/状态行/状态码的协议格式 |
| 阶段 3 | 一条连接上连续处理多个请求,不再靠"关闭连接"当结束信号,而是让 Content-Length 说了算 | [07 类 KP2/KP7](07-http-evolution.md) 持久连接与 Keep-Alive、[05 类知识点 7](05-tcp-reliability-and-flow-control.md) TCP 字节流粘包根源 |
| 阶段 4 | 组装成一个类,用标准库真实客户端从"另一端"验证全链路真实跑通 | 阶段 1-3 全部组装 + [11 类 KP1](11-network-programming-and-io-models.md) socket 编程实操 |

每个阶段的代码都能独立运行(本文件用该系列自己的 `_verify_md.py` 校验,校验方式是把每个 ` ```python ` 代码块单独拎出来起一个新的 Python 子进程执行——块与块之间**不共享任何变量**,所以后面阶段用到前面阶段写过的函数时,会重新贴一遍,不是偷懒复制,是这套校验机制要求的)。全部代码只连 `127.0.0.1` 本地回环、用端口 0 让操作系统分配一个当前空闲的端口(再用 `getsockname()` 读出真实分配到的端口号),不依赖任何写死的端口号,也不访问任何外部网络。

---

## 阶段 1:接一个连接,看懂请求行,回一个写死的响应

[04 类知识点 7](04-tcp-connection-management.md) 已经讲过 socket 编程的标准顺序:服务端 `create → bind → listen → accept`,客户端 `create → connect`,`accept()`/`connect()` 默认都是阻塞调用。这一步把这个顺序真正连起来跑一次,并且第一次亲手看到 HTTP 请求"长什么样"——不是文档里的示意图,是 `recv()` 真实收到的字节。

HTTP 请求的第一行(请求行)格式是 `METHOD PATH VERSION`,用空格分隔三个字段,行尾是 `\r\n`——[07 类 KP1](07-http-evolution.md) 已经展示过 HTTP/1.0 请求带这一行版本号。这一步先只解析这一行,响应则完全写死(状态行 + 一个 `Content-Length` + 一段固定 body),重点是先把"连接 → 收字节 → 解析出三个字段 → 拼字节发回去"这条最短路径跑通,不要一上来就想着把所有情况都处理了。

**这一步响应发完就直接关闭连接**——还没有引入"一条连接处理多个请求"的复杂度,客户端用"一直 `recv()` 到收到空字节(对方关闭连接)为止"这个最简单的读法就够用,这正是 [07 类 KP1](07-http-evolution.md) 里 HTTP/0.9 那种"发完就关连接"的极简模型,不是巧合选择,是刻意先站在最简单的起点,把复杂度留到阶段 3 再引入。

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


def parse_request_line(raw):
    first_line = raw.split(b"\r\n", 1)[0]
    method, path, version = first_line.decode("ascii").split(" ")
    return method, path, version


port = free_port()
parsed = {}


def serve_one():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(1)
    srv.settimeout(3)
    conn, addr = srv.accept()
    conn.settimeout(3)
    raw = conn.recv(4096)
    parsed["raw"] = raw
    parsed["method"], parsed["path"], parsed["version"] = parse_request_line(raw)

    body = b"hello from stage1"
    response = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
        b"\r\n" + body
    )
    conn.sendall(response)
    conn.close()
    srv.close()


t = threading.Thread(target=serve_one, daemon=True)
t.start()
time.sleep(0.2)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.settimeout(3)
client.connect((HOST, port))
client.sendall(b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")

chunks = []
while True:
    chunk = client.recv(4096)
    if not chunk:
        break
    chunks.append(chunk)
response = b"".join(chunks)
client.close()
t.join(timeout=3)

assert parsed["method"] == "GET"
assert parsed["path"] == "/"
assert parsed["version"] == "HTTP/1.1"
assert response.startswith(b"HTTP/1.1 200 OK\r\n")
assert response.endswith(b"hello from stage1")
print(f"stage1 ok: parsed method={parsed['method']} path={parsed['path']} version={parsed['version']}")
print(f"stage1 raw request bytes: {parsed['raw']!r}")
print(f"stage1 raw response bytes: {response!r}")
```

真实输出证实了两件事:请求行被正确拆成了 `method=GET path=/ version=HTTP/1.1` 三个字段;响应字节 `HTTP/1.1 200 OK\r\nContent-Length: 17\r\n\r\nhello from stage1` 是我们自己一段一段拼出来的原始字节流,不是某个库背着我们生成的——这也是全篇故意不用 `http.server`/Flask 的原因:框架会替你做完这一步,但看不到"HTTP 报文本质上就是一段按 `\r\n` 分行的纯文本"这个事实。

---

## 阶段 2:根据 path 返回不同内容——路由,顺带解析 headers

真实的服务器不会对所有请求都回同一句话。这一步把阶段 1"只看请求行"扩展成解析完整的请求头部:先按 `\r\n\r\n` 把"头部"和"body"切开(头部结束、body 开始的分界线,HTTP 协议里固定是这四个字节),头部再按 `\r\n` 切成一行行,每一行按第一个 `:` 切成 header 名字和值——[07 类 KP1](07-http-evolution.md) 提到 HTTP/1.0 引入了 `Host`/`User-Agent` 这类请求头,这里第一次亲手把它们解析进一个字典里。**HTTP header 名字是大小写不敏感的**(`Host` 和 `host` 该被当成同一个 header),所以字典的 key 统一转成小写存,这是解析 headers 时容易漏掉的一个协议细节,不是随手加的代码风格。

路由本身很直接:一个 `{path: body}` 字典,查得到就 200,查不到就 404——[07 类 KP3](07-http-evolution.md) 讲过状态码分类,2xx 和 4xx 分别对应这两种情况的语义。

这一步依然沿用阶段 1"响应完就关连接"的简化模型(下一阶段才会打破这个简化),所以还不需要操心 Content-Length 算得准不准——客户端还是靠"读到连接关闭"来判断一条响应结束,这个安全网这一步还在,坑要等到阶段 3 才会真正暴露出来。

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


def parse_request(raw):
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    method, path, version = lines[0].decode("ascii").split(" ")
    headers = {}
    for line in lines[1:]:
        if not line:
            continue
        name, _, value = line.decode("ascii").partition(":")
        headers[name.strip().lower()] = value.strip()
    return method, path, version, headers, body


def build_response(status_code, status_text, body):
    return (
        f"HTTP/1.1 {status_code} {status_text}\r\n".encode("ascii")
        + b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
        + b"\r\n" + body
    )


ROUTES = {
    "/": b"welcome to the mini server",
    "/about": b"this is the about page, a bit longer than home",
}


def handle_one(conn):
    raw = conn.recv(4096)
    method, path, version, headers, body = parse_request(raw)
    if path in ROUTES:
        resp = build_response(200, "OK", ROUTES[path])
    else:
        resp = build_response(404, "Not Found", b"no such route: " + path.encode("ascii"))
    conn.sendall(resp)
    return method, path, headers


port = free_port()
seen = []


def server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(3)
    srv.settimeout(3)
    for _ in range(3):
        conn, _ = srv.accept()
        conn.settimeout(3)
        method, path, headers = handle_one(conn)
        seen.append((method, path, headers.get("host")))
        conn.close()
    srv.close()


t = threading.Thread(target=server, daemon=True)
t.start()
time.sleep(0.2)


def request(path):
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.settimeout(3)
    c.connect((HOST, port))
    c.sendall(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode("ascii"))
    chunks = []
    while True:
        chunk = c.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
    c.close()
    return b"".join(chunks)


r_home = request("/")
r_about = request("/about")
r_missing = request("/nope")
t.join(timeout=3)

assert r_home.startswith(b"HTTP/1.1 200 OK\r\n")
assert r_home.endswith(b"welcome to the mini server")
assert r_about.startswith(b"HTTP/1.1 200 OK\r\n")
assert r_about.endswith(b"about page, a bit longer than home")
assert r_missing.startswith(b"HTTP/1.1 404 Not Found\r\n")
assert b"no such route: /nope" in r_missing

assert seen == [("GET", "/", "127.0.0.1"), ("GET", "/about", "127.0.0.1"), ("GET", "/nope", "127.0.0.1")]
print("stage2 ok: routing dispatched 3 different paths to 3 different responses")
print(f"stage2 headers parsed correctly, e.g. Host header seen for each request: {[h for _, _, h in seen]}")
```

三次独立的请求分别命中 `/`、`/about`、`/nope` 三种情况,响应体和状态码都对得上,而且 headers 字典真的解析出了每次请求携带的 `Host: 127.0.0.1`——证明这不是凑巧蒙对的,是真的把结构化数据从原始字节里抠出来了。

---

## 阶段 3:Content-Length 必须算对——一个真实复现的坑

阶段 1/2 能蒙混过关,是因为"响应完就关连接"这个简化模型让 Content-Length 从来没有被真正依赖过——客户端靠"连接关闭"就知道响应结束了,即使 Content-Length 写错也发现不了。但真实世界的服务器不会每个请求都开一条新连接([07 类 KP2](07-http-evolution.md) 讲过持久连接为什么是 HTTP/1.1 的默认行为,[07 类 KP7](07-http-evolution.md) 用 `accept()` 调用次数证明了连接复用是真实发生的,不是纸面上的规定)——一旦一条连接要连续服务多个请求,"关闭连接"这个信号就不能再用来判断"一个响应到哪里结束",必须靠 Content-Length 明确告诉对方"body 有多少字节"。这正好呼应 [05 类知识点 7](05-tcp-reliability-and-flow-control.md) 讨论 TCP 粘包问题时留下的追问——"HTTP 是怎么解决消息边界问题的?"——答案就是 Content-Length,这一步现场验证这个答案,包括算错了会有什么真实的、字节级别的后果。

**先复现一次真实的坑。** 服务器在同一条连接上依次发送两个响应:第一个响应 body 是 10 个字节的 `A`,Content-Length 写 10,正确;第二个响应 body 应该是 30 个字节的 `B`,但代码复制粘贴第一个响应的写法时忘了同步改 Content-Length,依然写死成 10。客户端这边实现一个诚实的 `read_one_response`:严格按 Content-Length 声明的字节数截取 body,多出来的字节留在缓冲区里,交给下一次调用当作"下一条消息可能的开头"——这正是一条持久连接上如果没有正确实现消息边界,数据会错位到哪里去的真实机制。

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


def read_one_response(sock, leftover=b""):
    buf = leftover
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("peer closed before headers were complete")
        buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    status_line = lines[0].decode("ascii")
    headers = {}
    for line in lines[1:]:
        name, _, value = line.decode("ascii").partition(":")
        headers[name.strip().lower()] = value.strip()
    content_length = int(headers.get("content-length", "0"))
    while len(rest) < content_length:
        chunk = sock.recv(4096)
        if not chunk:
            break
        rest += chunk
    body = rest[:content_length]
    leftover_after = rest[content_length:]
    return status_line, headers, body, leftover_after


port = free_port()


def buggy_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(1)
    srv.settimeout(3)
    conn, _ = srv.accept()
    conn.settimeout(3)

    conn.recv(4096)  # request A, content not inspected for this demo
    body_a = b"A" * 10
    resp_a = b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n" + body_a
    conn.sendall(resp_a)

    conn.recv(4096)  # request B, on the SAME kept-alive connection
    body_b = b"B" * 30
    # BUG: Content-Length copy-pasted from response A instead of computed from body_b's real length.
    resp_b = b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n" + body_b
    conn.sendall(resp_b)
    conn.close()
    srv.close()


t = threading.Thread(target=buggy_server, daemon=True)
t.start()
time.sleep(0.2)

c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
c.settimeout(3)
c.connect((HOST, port))

c.sendall(b"GET /a HTTP/1.1\r\nHost: x\r\n\r\n")
status_a, headers_a, body_a_seen, leftover = read_one_response(c)
assert status_a == "HTTP/1.1 200 OK"
assert body_a_seen == b"A" * 10

c.sendall(b"GET /b HTTP/1.1\r\nHost: x\r\n\r\n")
status_b, headers_b, body_b_seen, leftover = read_one_response(c, leftover)

assert body_b_seen == b"B" * 10, body_b_seen
assert leftover == b"B" * 20, leftover
assert not leftover.startswith(b"HTTP/1.1"), "leftover should be raw leftover body bytes, not a real status line"

c.close()
t.join(timeout=3)
print(f"BUG reproduced: response B body truncated to {body_b_seen!r} (real body was 30 B's)")
print(f"BUG reproduced: {len(leftover)} real body bytes misfiled as 'leftover stream data': {leftover!r}")
```

真实输出:客户端以为第二个响应的 body 只有 10 个字节(`b'BBBBBBBBBB'`),剩下 20 个字节的 `B`(`b'BBBBBBBBBBBBBBBBBBBB'`)被错误地留在了缓冲区里,如果这条连接上还有第三个真实响应,这 20 个字节会和第三个响应的真实数据混在一起、彻底解析错乱——这就是 [05 类知识点 7](05-tcp-reliability-and-flow-control.md) 讲的粘包问题在 HTTP 这一层的真实表现形式,不是抽象的理论描述,是可以现场复现的字节错位。

**修:Content-Length 永远从真实 body 的字节长度现算,不能写死、更不能复制粘贴上一次的数字。**

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


def read_one_response(sock, leftover=b""):
    buf = leftover
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("peer closed before headers were complete")
        buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    status_line = lines[0].decode("ascii")
    headers = {}
    for line in lines[1:]:
        name, _, value = line.decode("ascii").partition(":")
        headers[name.strip().lower()] = value.strip()
    content_length = int(headers.get("content-length", "0"))
    while len(rest) < content_length:
        chunk = sock.recv(4096)
        if not chunk:
            break
        rest += chunk
    body = rest[:content_length]
    leftover_after = rest[content_length:]
    return status_line, headers, body, leftover_after


def build_response(body):
    # FIX: Content-Length is always computed from the real body bytes, never hardcoded.
    return (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
        b"Connection: keep-alive\r\n"
        b"\r\n" + body
    )


port = free_port()


def fixed_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(1)
    srv.settimeout(3)
    conn, _ = srv.accept()
    conn.settimeout(3)

    conn.recv(4096)  # request A
    conn.sendall(build_response(b"A" * 10))

    conn.recv(4096)  # request B, same connection
    conn.sendall(build_response(b"B" * 30))
    conn.close()
    srv.close()


t = threading.Thread(target=fixed_server, daemon=True)
t.start()
time.sleep(0.2)

c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
c.settimeout(3)
c.connect((HOST, port))

c.sendall(b"GET /a HTTP/1.1\r\nHost: x\r\n\r\n")
status_a, headers_a, body_a, leftover = read_one_response(c)
assert status_a == "HTTP/1.1 200 OK"
assert body_a == b"A" * 10
assert leftover == b"", "no leftover expected: response A framing must be exact"

c.sendall(b"GET /b HTTP/1.1\r\nHost: x\r\n\r\n")
status_b, headers_b, body_b, leftover = read_one_response(c, leftover)
assert status_b == "HTTP/1.1 200 OK"
assert body_b == b"B" * 30, body_b
assert leftover == b"", "no leftover expected: response B framing must be exact"

c.close()
t.join(timeout=3)
print(f"FIXED: response A body correct ({len(body_a)} bytes), response B body correct ({len(body_b)} bytes)")
print("FIXED: zero leftover bytes after each response -> stream stays correctly framed across 2 requests on 1 connection")
```

修好之后,同一条连接连续处理两个 body 长度不同的请求,每次读完都恰好零字节剩余——这正是"一条连接可以承载多个请求"这个能力真正被解锁的时刻:不是把阶段 1/2 的代码原样搬过来跑两次就行,必须先有正确的 Content-Length,持久连接才谈得上正确工作。

---

## 阶段 4:组装成一个 `MiniHTTPServer` 类,用标准库真实客户端验证全链路

前三阶段的"客户端"都是我们自己手写的裸 socket——这足以验证服务端和自己写的客户端能对上话,但还没有证明这个服务器讲的是"标准 HTTP",还是"只有我们自己的测试代码才懂的方言"。这一步换成 Python 标准库 `http.client.HTTPConnection`(完全独立于我们的服务端实现,严格按 HTTP 规范收发字节)当客户端——如果它能正确对话,才是"这个服务器真的说 HTTP"的过硬证据,呼应任务本身的要求:不是只在同一个进程里手工调用解析函数,而是真实起一个监听 socket、真实连接、真实收发字节。

把前三阶段的能力装进一个类:`_read_request` 合并了阶段 2 的头部解析和阶段 3 学到的"body 要按 Content-Length 精确截取"(这次用在请求方向,和阶段 3 的响应方向是同一个原理镜像用了一次——请求也可能带 body,比如 POST);`_build_response` 是阶段 3 修好的动态 Content-Length 写法;`_handle_connection` 在同一条连接上循环处理请求直到客户端关闭连接——这正是阶段 3 解锁的能力。新增的路由额外支持 `POST`,用来验证请求 body(不只是响应 body)也能被正确读到。

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


class MiniHTTPServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.routes = {}
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(5)
        self._sock.settimeout(0.5)
        self._stop = threading.Event()
        self._thread = None

    def add_route(self, method, path, handler):
        self.routes[(method, path)] = handler

    def start(self):
        self._thread = threading.Thread(target=self._serve_forever, daemon=True)
        self._thread.start()

    def stop(self, timeout=3):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._sock.close()

    def _serve_forever(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            conn.settimeout(2)
            try:
                self._handle_connection(conn)
            except (ConnectionError, OSError):
                pass
            finally:
                conn.close()

    def _handle_connection(self, conn):
        leftover = b""
        while True:
            try:
                method, path, version, headers, body, leftover = self._read_request(conn, leftover)
            except ConnectionError:
                return
            handler = self.routes.get((method, path))
            if handler is None:
                status_code, status_text, resp_body = 404, "Not Found", b"no such route"
            else:
                status_code, status_text, resp_body = handler(headers, body)
            conn.sendall(self._build_response(status_code, status_text, resp_body))
            if headers.get("connection", "").lower() == "close":
                return

    def _read_request(self, conn, leftover):
        buf = leftover
        while b"\r\n\r\n" not in buf:
            chunk = conn.recv(4096)
            if not chunk:
                raise ConnectionError("client closed before headers were complete")
            buf += chunk
        head, _, rest = buf.partition(b"\r\n\r\n")
        lines = head.split(b"\r\n")
        method, path, version = lines[0].decode("ascii").split(" ")
        headers = {}
        for line in lines[1:]:
            name, _, value = line.decode("ascii").partition(":")
            headers[name.strip().lower()] = value.strip()
        content_length = int(headers.get("content-length", "0"))
        while len(rest) < content_length:
            chunk = conn.recv(4096)
            if not chunk:
                break
            rest += chunk
        body = rest[:content_length]
        leftover_after = rest[content_length:]
        return method, path, version, headers, body, leftover_after

    def _build_response(self, status_code, status_text, body):
        return (
            f"HTTP/1.1 {status_code} {status_text}\r\n".encode("ascii")
            + b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
            + b"Connection: keep-alive\r\n"
            + b"\r\n" + body
        )


def handle_home(headers, body):
    return 200, "OK", b"welcome to the mini server"


def handle_hello(headers, body):
    return 200, "OK", b"hello there"


def handle_echo(headers, body):
    return 200, "OK", b"echo:" + body


port = free_port()
server = MiniHTTPServer(HOST, port)
server.add_route("GET", "/", handle_home)
server.add_route("GET", "/hello", handle_hello)
server.add_route("POST", "/echo", handle_echo)
server.start()
time.sleep(0.2)

conn = http.client.HTTPConnection(HOST, port, timeout=3)

conn.request("GET", "/")
r1 = conn.getresponse()
assert r1.status == 200
assert r1.read() == b"welcome to the mini server"

conn.request("GET", "/hello")
r2 = conn.getresponse()
assert r2.status == 200
assert r2.read() == b"hello there"

conn.request("POST", "/echo", body=b"ping-payload-123")
r3 = conn.getresponse()
assert r3.status == 200
assert r3.read() == b"echo:ping-payload-123"

conn.request("GET", "/does-not-exist")
r4 = conn.getresponse()
assert r4.status == 404
conn.close()

# A second, completely independent connection -- proves routing/parsing works fresh each time,
# not by accident relying on state left over from the first connection.
conn2 = http.client.HTTPConnection(HOST, port, timeout=3)
conn2.request("GET", "/hello")
r5 = conn2.getresponse()
assert r5.status == 200
assert r5.read() == b"hello there"
conn2.close()

server.stop()
print("stage4 ok: GET /, GET /hello, POST /echo, GET /missing(404) all verified over real http.client sockets")
print("stage4 ok: 4 requests reused 1 connection, 1 request used a fresh 2nd connection -- both worked correctly")
```

真实输出证明了两件事:同一个 `http.client.HTTPConnection` 对象连续发出 4 个请求(`GET /`、`GET /hello`、`POST /echo`、`GET /does-not-exist`)全部得到正确响应,靠的是服务端同一条连接上的 `_handle_connection` 循环(呼应 [07 类 KP7](07-http-evolution.md) "一条连接复用、`accept()` 只需调用 1 次"的验证方法论,虽然这里没有再重复计数 `accept()`,机制是同一个);另开一个全新连接单独发第 5 个请求同样正确,证明路由和解析不依赖任何"残留状态"。

**一个容易被混淆的细节**:上面的 accept 循环用的是 `self._sock.settimeout(0.5)`,让阻塞的 `accept()` 最多等 0.5 秒就抛 `socket.timeout`,这样服务器线程能定期检查"要不要停下来"(`self._stop.is_set()`)。这依然是**阻塞模式**——数据没来时线程真的在等待,只是等待有个上限——不是 [11 类 KP1](11-network-programming-and-io-models.md) 讲的 `setblocking(False)` 真非阻塞模式(那种模式下没数据会立刻抛 `BlockingIOError`,完全不等待)。两者都能让程序"不永远卡住",但机制不同,面试被追问细节时不能混为一谈,这也是 11 类 KP1"常见坑"提醒过的"非阻塞 IO 和异步 IO 不能混为一谈"在这里的一个近亲版本:"带超时的阻塞"和"真非阻塞"同样是两回事。

---

## 可以怎么继续扩展(只指方向,不实现)

- **并发处理多个客户端**:现在的 `MiniHTTPServer` 一次只能完整处理完一个连接的所有请求,才会去 accept 下一个——真实服务器要么每个连接开一个线程,要么用 [11 类 KP2](11-network-programming-and-io-models.md) 的 `select`/`epoll` 单线程事件循环同时监控多个连接,这是本篇故意留白、下一步最自然的扩展方向。
- **chunked 传输编码**:body 长度如果在发送时还不知道(比如边生成边发的流式响应),Content-Length 这条路走不通,需要 `Transfer-Encoding: chunked`——每个 chunk 前面带一个十六进制长度前缀,用长度为 0 的 chunk 标记结束,是 Content-Length 之外解决"消息边界"问题的另一种方案。
- **更完整的请求解析**:查询字符串(`?key=value`)、URL 百分号编码、重复出现的同名 header、用 `Transfer-Encoding` 而不是 `Content-Length` 描述的请求体,这些都是真实 HTTP 解析要处理、但本篇为了聚焦核心机制而跳过的边界情况。
- **HTTPS**:给这个 socket 包一层 TLS([08 类](08-https-and-tls.md)已经讲过握手细节),Python 标准库 `ssl.SSLContext.wrap_socket` 可以直接包在现有 `accept()` 返回的连接对象外面,业务解析逻辑基本不用改。
- **静态文件服务**:结合 [11 类 KP5](11-network-programming-and-io-models.md) 的 `sendfile()` 零拷贝,给大文件路由一条更高效的响应路径,不用先把整个文件读进内存再 `sendall()`。

这几个方向都不实现,是为了让这篇教程聚焦在"几个已学知识点怎么拼成一个真实能被标准客户端连上的服务器"这一件事上——每一个方向单独展开都够写一整套内容。

## 这篇教程展示的方法论

和 [dsa-deep-dive 21 类](../dsa-deep-dive/21-build-a-mini-search-engine.md)是同一个模式:挑几个关联的知识点 → 设计一个真实有用、读者一看就懂价值的小工具 → 分阶段增量实现,每一步都跑起来看到真实字节,而不是一次性甩出完整代码。这一篇额外的收获是一次"真实性验证"的现场示范:阶段 3 如果没有真的构造两个 body 长度不同的响应去复现粘包错位,"Content-Length 写死会出错"就只是一句正确但空洞的话;阶段 4 如果不换成标准库客户端,"这是一个真正的 HTTP 服务器,不是自娱自乐的方言"这句话也拿不出过硬的证据。

---

*创建:2026-07-24。4 个阶段(不计入约 80 个知识点统计),5 个可运行代码块(阶段 1/2/4 各 1 个,阶段 3 拆成 bug 复现 + 修复共 2 个),全部在 `.venv` 独立验证通过,只连 `127.0.0.1` 本地回环。*
