# 10. 现代应用协议与 API(板块 IV 收官)

> 板块 IV 最后一篇,从"协议本身"转向"协议之上怎么设计一个可用的 API/实时通信系统"。KP7 负载均衡算法里的一致性哈希部分交叉引用 [dsa-deep-dive 20 类](../dsa-deep-dive/20-advanced-interview-depth.md)已有的真实一致性哈希+虚拟节点负载均衡实测代码,不重复推导——本类只讲它在负载均衡算法家族里的定位。

---

## KP1. WebSocket 协议(握手升级机制,vs HTTP 长轮询)

**签名/是什么:**

```
握手升级:客户端发一个特殊的 HTTP 请求(带 Upgrade: websocket 头),
         服务器同意后返回 101 Switching Protocols,同一条 TCP 连接从这一刻起
         不再跑 HTTP 语义,改为跑 WebSocket 帧协议 —— 全双工、双向都能随时主动发消息。
Sec-WebSocket-Key/Accept:客户端发一个随机 key,服务器必须返回
         base64(SHA1(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11")),
         用来证明服务器真的理解 WebSocket 协议(而不是一个不认识 Upgrade 头、误把请求转发回声的中间代理)。
```

**一句话:** WebSocket 借用 HTTP 的握手过程"借壳上市"(这样能顺利穿过大多数只认识 HTTP 的防火墙/代理),握手一旦成功,连接就变成一条服务器可以随时主动推送数据的全双工通道,不再是 HTTP 那种"客户端问一句、服务器答一句"的请求-响应模型。

**底层机制/为什么这样设计:** 在 WebSocket 出现之前,想做到"服务器主动推送消息给浏览器"(比如聊天应用、实时通知)只能用长轮询(Long Polling)这种变通手法:客户端发一个 HTTP 请求,服务器"憋着"不立刻回复,直到有新消息才返回,客户端收到后立刻发起下一个长轮询请求——本质上还是"请求-响应"模型,只是把响应刻意延迟,每一条消息依然要重新走一次完整的 HTTP 请求(带上全部请求头,包括 Cookie 等)。WebSocket 的握手设计成"看起来像一次普通的 HTTP GET 请求",是为了兼容性——中间的代理、防火墙、负载均衡器很多只认识标准 HTTP,如果 WebSocket 一开始就用一种全新的、它们不认识的协议格式,大概率会被这些中间设备拒绝或搞坏;先用 HTTP 语义完成握手,取得"你情我愿升级协议"的共识后,再切换到轻量的二进制帧格式,是一种向后兼容的实用主义设计。Sec-WebSocket-Key/Accept 这套看起来像"加密"但其实只是简单哈希校验的机制,真实目的不是安全,而是防止一个不理解 WebSocket、只会把 HTTP 请求原样转发/缓存的旧代理,错把客户端的握手请求当作普通请求处理并返回一个假的"成功"响应——服务器返回的 Accept 值必须是对客户端 Key 做特定哈希运算的结果,只有真正理解协议的服务器才答得对。

**AI 研究/工程场景:** 流式生成场景(比如聊天机器人一个字一个字往前端吐字的效果)是 WebSocket 或它的近亲 SSE(Server-Sent Events,单向流式推送)的经典应用场景——模型逐 token 生成的结果需要尽快推给前端展示,而不是等生成完整个回答才一次性返回,这种"服务器主动、持续推送"的模式正是长轮询天然别扭、WebSocket/SSE 天然适合的场景。

**可运行例子(验证环境:`.venv`,真实完成 RFC 6455 握手计算 + 真实收发一帧 WebSocket 数据):**

```python
import socket
import threading
import time
import hashlib
import base64

WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def compute_accept(key):
    return base64.b64encode(hashlib.sha1((key + WS_MAGIC).encode()).digest()).decode()


def make_ws_frame(payload: bytes, opcode=0x1):
    # 最简单的服务器->客户端未掩码文本帧:FIN=1,opcode,小长度直接编码(payload<126字节场景)。
    b1 = 0x80 | opcode
    length = len(payload)
    assert length < 126
    return bytes([b1, length]) + payload


def parse_ws_frame_from_client(data: bytes):
    # 客户端->服务器的帧按 RFC 6455 规定必须打掩码,服务器要先用 mask 异或还原出真实 payload。
    b1, b2 = data[0], data[1]
    masked = bool(b2 & 0x80)
    length = b2 & 0x7F
    assert masked, "client frames must be masked"
    mask = data[2:6]
    payload = data[6:6 + length]
    return bytes(payload[i] ^ mask[i % 4] for i in range(length))


HOST = "127.0.0.1"


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


port = free_port()


def server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(1)
    conn, _ = srv.accept()
    req = conn.recv(4096).decode()
    key = next(line.split(":", 1)[1].strip() for line in req.split("\r\n") if line.lower().startswith("sec-websocket-key:"))
    accept = compute_accept(key)
    resp = (
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
        f"Connection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
    )
    conn.sendall(resp.encode())
    data = conn.recv(4096)
    payload = parse_ws_frame_from_client(data)
    conn.sendall(make_ws_frame(b"echo:" + payload))
    conn.close()
    srv.close()


t = threading.Thread(target=server, daemon=True)
t.start()
time.sleep(0.2)

client_key = base64.b64encode(b"0123456789012345").decode()
req = (
    f"GET /chat HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
    f"Sec-WebSocket-Key: {client_key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
)
c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
c.connect((HOST, port))
c.sendall(req.encode())
resp = c.recv(4096).decode()
assert "101 Switching Protocols" in resp
assert compute_accept(client_key) in resp  # 服务器的 Accept 值必须严格等于 RFC6455 公式算出的结果

# 发一帧真实的(打了掩码的)客户端 WebSocket 帧。
payload = b"hi"
mask = bytes([0x12, 0x34, 0x56, 0x78])
masked_payload = bytes(payload[i] ^ mask[i % 4] for i in range(len(payload)))
frame = bytes([0x81, 0x80 | len(payload)]) + mask + masked_payload
c.sendall(frame)
server_frame = c.recv(4096)
srv_len = server_frame[1] & 0x7F
srv_payload = server_frame[2:2 + srv_len]
assert srv_payload == b"echo:hi", srv_payload
c.close()
t.join(timeout=3)
print(f"handshake Accept matches RFC6455 formula, real frame round trip got {srv_payload}")

# 对比:N 条小消息用长轮询 vs WebSocket 各自的字节开销数量级。
N = 5
http_request_headers = b"GET /poll HTTP/1.1\r\nHost: example.com\r\nCookie: session=abcdef123456\r\n\r\n"
http_response_headers = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 20\r\n\r\n"
longpoll_total = N * (len(http_request_headers) + len(http_response_headers) + 20)
ws_handshake_bytes = 300
ws_total = ws_handshake_bytes + N * (2 + 20)
assert ws_total < longpoll_total
print(f"{N} messages -> long-polling~{longpoll_total}B vs websocket~{ws_total}B "
      f"(websocket avoids re-sending full HTTP headers every single message)")
```

**面试怎么问+追问链:**
- Q:为什么 WebSocket 握手要伪装成一个 HTTP 请求,而不是直接设计一个全新的协议握手过程?
  - 追问1:如果客户端和服务器之间有一个不理解 WebSocket 的旧代理,会发生什么?
  - 深挖追问(诊断真实数据轴):这类代理通常会把 `Upgrade` 头当成不认识的普通头忽略掉,按普通 HTTP 请求处理并可能返回 200 而不是 101——一个健壮的 WebSocket 客户端库必须显式检查响应状态码是否精确等于 101、`Sec-WebSocket-Accept` 是否匹配,握手失败要能被业务代码感知并降级(比如退回长轮询),而不是假设握手一定成功;这是真实生产环境里 WebSocket 连接失败率排查的第一检查点。

**常见坑:**
- 以为 WebSocket 握手成功后就完全脱离了 HTTP,后续可以用 HTTP 相关的工具(比如某些只解析 HTTP 协议的抓包/调试工具)继续观察——握手完成瞬间协议语义整体切换为 WebSocket 帧格式,通用 HTTP 分析工具无法再解析后续帧内容,需要专门支持 WebSocket 协议的调试工具。

---

## KP2. gRPC(基于 HTTP/2 + Protocol Buffers)

**签名/是什么:**

```
gRPC = HTTP/2(传输层,复用 07类KP5 的多路复用/二进制分帧) + Protocol Buffers(序列化格式)。
Protocol Buffers:预先用 .proto 文件定义消息的字段名和类型,编译成代码;
                 序列化时不传输字段名文本,只传"字段编号+类型+值"的二进制编码 —— 比 JSON 紧凑得多。
```

**一句话:** gRPC 的性能优势来自两层叠加——传输层用 HTTP/2 的多路复用/二进制分帧(见 [07 类 KP5](07-http-evolution.md)),序列化层用 Protocol Buffers 把"字段名文本"从每条消息里去掉,只留纯二进制的字段编号和值。

**底层机制/为什么这样设计:** JSON 这类文本格式每次序列化都要把字段名(如 `"user_id"`)原样写进消息体,如果一条消息有十几个字段、每秒钟被序列化传输几十万次,这些重复的字段名文本本身就是纯粹的带宽浪费——接收方和发送方其实都"知道"字段名和类型(因为双方共享同一份 `.proto` schema 定义),没有必要每次都在报文里重新声明一遍。Protocol Buffers 的解法是把"字段叫什么名字"这件事从运行时的报文里挪到编译时的 schema 定义里:序列化时只写"字段编号(一个小整数)+ 变长编码的值",解析方拿着同一份 schema 就知道"编号 1 对应哪个字段名、什么类型",不需要报文里出现任何文本字段名。这个设计的代价是牺牲了 JSON"人类直接可读、不需要 schema 就能理解"的优点,换来更小的报文体积和更快的序列化/反序列化速度——这是一个典型的"强类型契约换性能"的工程权衡,在双方是同一个团队维护的内部服务之间(不需要给外部第三方阅读原始报文),这个权衡通常是划算的。

**AI 研究/工程场景:** 模型 serving 内部的微服务调用(比如网关服务调用具体的推理 worker)以及需要传输张量/embedding 这类高频大数据量的场景,普遍选用 gRPC 而不是 REST/JSON——一方面推理请求可能每秒几千次,字段名重复传输的开销会被放大;另一方面 HTTP/2 的多路复用天然适合"一个长连接上并发跑大量并行推理请求"这种模式,不需要为每个并发请求单独开一条 TCP 连接。

**可运行例子(验证环境:`.venv`,教学性模拟——标准库没有 protobuf 支持,用简化的 tag-varint 编码器演示"省掉字段名文本"这个核心机制,并与等价 JSON 做真实字节数对比):**

```python
import json


def encode_varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def protobuf_style_encode(user_id, score):
    # 字段1(user_id)tag=1,字段2(score)tag=2 —— schema(字段名对应关系)在编译期双方就已知晓,
    # 不需要在报文里传输 "user_id" "score" 这些字符串本身。
    return bytes([0x08]) + encode_varint(user_id) + bytes([0x10]) + encode_varint(score)


user_id, score = 123456, 9001
binary_msg = protobuf_style_encode(user_id, score)
json_msg = json.dumps({"user_id": user_id, "score": score}).encode()

assert len(binary_msg) < len(json_msg)
saving_pct = 100 * (1 - len(binary_msg) / len(json_msg))
assert saving_pct > 20
print(f"json={len(json_msg)}B ({json_msg}) vs protobuf-style={len(binary_msg)}B, saving={saving_pct:.0f}%")
```

**面试怎么问+追问链:**
- Q:为什么 AI 模型 serving 场景经常选 gRPC 而不是 RESTful JSON API?
  - 追问1(决策依据追问轴):如果这个推理服务需要直接被浏览器前端调用(不是服务间调用),还应该继续用 gRPC 吗?
  - 深挖追问:通常不会——浏览器原生不支持直接发起 gRPC 调用(需要 gRPC-Web 这类额外的转译层,而且失去了 JSON 那种直接用浏览器开发者工具肉眼调试的便利性),这种场景下 REST/JSON 或者干脆 WebSocket/SSE 做流式推送反而更合适;gRPC 的性能优势在"服务间高频调用、双方都是受控的后端代码"场景下才能充分兑现,面向不受控的外部客户端(尤其是浏览器)时,兼容性和调试便利性往往比极致性能更重要。这条追问检验候选人是否理解技术选型要看具体的调用双方是谁,而不是简单地认为"gRPC 更快所以总是更好"。

**常见坑:**
- 认为 gRPC 的性能优势全部来自"用了 HTTP/2"——如果只是把 JSON 报文原封不动地跑在 HTTP/2 上(不改用 Protocol Buffers),依然能获得多路复用的收益,但拿不到序列化体积/速度的收益;gRPC 的完整性能优势是"HTTP/2 传输 + Protobuf 序列化"两层叠加的结果,只占了其中一层就断言"和 gRPC 性能一样"是不准确的。

---

## KP3. RESTful API 设计原则

**签名/是什么:**

```
资源导向(Resource-Oriented):URL 代表"名词"(资源),不代表"动词"(操作)——
  /users/5          好:资源是"编号为5的用户"
  /getUserById?id=5 差:URL 里出现了动词"get",把 RPC 风格的调用硬套在 HTTP 上
用 HTTP 方法表达操作意图(见 07类KP3 安全性/幂等性),用状态码表达结果语义,不用自定义字段重复表达。
```

**一句话:** RESTful 设计的核心是"让 HTTP 协议本身已有的语义(方法、状态码、URL 结构)去表达业务操作的意图",而不是把 HTTP 仅仅当作一个传输管道、所有语义都自己在请求体里重新发明一遍。

**底层机制/为什么这样设计:** "RPC 风格 over HTTP"(比如所有接口都是 `POST /api/doAction`,具体做什么操作、操作哪个资源全部塞在请求体的 JSON 字段里)虽然一样能实现功能,但放弃了 HTTP 协议自带的一整套语义能力:浏览器/CDN/代理无法从 URL 判断这是不是一个安全的只读操作(因为所有请求都是 POST),缓存层无法基于 URL 做智能缓存决策(见 [07 类 KP4](07-http-evolution.md)),负载均衡器无法从状态码判断请求是否成功需要重试。RESTful 设计把"资源是什么"编码进 URL 路径(`/users/5` 而不是参数里的 `id=5`),把"对资源做什么操作"编码进 HTTP 方法(GET 查询、POST 创建、PUT/PATCH 更新、DELETE 删除),把"操作结果如何"编码进状态码(2xx 成功、4xx 客户端错误、5xx 服务端错误)——这样中间的各种通用 HTTP 基础设施(缓存、代理、监控、重试框架)都能不理解具体业务逻辑的情况下,仅凭 HTTP 协议本身的语义就能做出正确决策,这正是"资源导向"设计相比"RPC over HTTP"的核心优势所在。

**AI 研究/工程场景:** 模型管理类的控制面 API(比如"创建一个新的模型部署"、"查询某次训练任务的状态")通常会被设计成 RESTful 风格(`POST /deployments` 创建、`GET /jobs/{id}` 查状态),因为这类 API 调用频率不高但需要良好的可读性、可缓存性(查询状态类接口很适合走 HTTP 缓存)、以及被各种通用运维工具(监控、审计日志分析)直接理解——这和 KP2 讨论的"高频服务间调用用 gRPC"是两种不同的调用场景,选型依据也不同。

**可运行例子(验证环境:`.venv`,真实构造资源导向的 URL 结构,验证方法语义+状态码语义正确落地):**

```python
import socket
import threading
import time
import json

HOST = "127.0.0.1"


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


port = free_port()
users = {"1": {"name": "alice"}}


def server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(5)
    for _ in range(3):
        conn, _ = srv.accept()
        req = conn.recv(4096).decode()
        method, path, _ = req.split(" ", 2)
        if method == "GET" and path == "/users/1":
            body = json.dumps(users["1"]).encode()
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        elif method == "GET" and path == "/users/999":
            conn.sendall(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
        elif method == "POST" and path == "/users":
            body = json.dumps({"id": "3", "name": "carol"}).encode()
            conn.sendall(
                b"HTTP/1.1 201 Created\r\nLocation: /users/3\r\nContent-Length: "
                + str(len(body)).encode() + b"\r\n\r\n" + body
            )
        conn.close()
    srv.close()


t = threading.Thread(target=server, daemon=True)
t.start()
time.sleep(0.2)


def do(method, path):
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.connect((HOST, port))
    c.sendall(f"{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n".encode())
    resp = c.recv(4096)
    c.close()
    return resp


# 资源存在:GET /users/{id} -> 200 + 资源内容(URL 里是"名词",没有动词)。
r1 = do("GET", "/users/1")
assert b"200 OK" in r1 and b"alice" in r1

# 资源不存在:状态码本身就表达了结果,不需要在响应体里额外发明一套"success: false"字段。
r2 = do("GET", "/users/999")
assert b"404" in r2

# 创建资源:POST 到集合 URL(/users,不是/users/newUser这种动词式路径),
# 201 Created + Location 头直接告诉客户端新资源在哪,不需要额外查询。
r3 = do("POST", "/users")
assert b"201 Created" in r3 and b"Location: /users/3" in r3

t.join(timeout=3)
print("resource URL /users/{id} -> 200+body, missing resource -> 404, "
      "POST /users creates -> 201 Created + Location header (all semantics live in URL/method/status, not custom fields)")
```

**面试怎么问+追问链:**
- Q:`/getUserById?id=5` 和 `GET /users/5` 从功能上看起来做的是同一件事,区别到底在哪?
  - 追问1:如果一个系统内部全是 `/getXxx`、`/updateXxx` 这种风格的接口,业务能跑起来吗?
  - 深挖追问(方案批判迭代轴):业务功能上完全能跑起来——这正是这个追问的陷阱所在,候选人如果只会说"不规范""不优雅"这类空泛评价,说明没有讲到点子上;应该讲清楚具体损失了什么能力:无法被 HTTP 缓存正确识别为可缓存的只读操作、无法让负载均衡器安全地对超时请求做自动重试(因为所有请求都是 POST,没法确定是否幂等)、监控系统无法仅从 URL+方法就区分读写流量占比。这些能力的损失在系统规模小的时候不明显,规模变大后会变成实实在在的运维成本。

**常见坑:**
- 过度追求"纯粹 RESTful"到不切实际的程度——比如坚持所有操作都必须精确映射到 CRUD 四个动词,遇到"批量导入"、"发送重置密码邮件"这类天然不是简单增删改查的操作时生搬硬套,结果 URL 变得比 RPC 风格还别扭;实践中主流做法是"资源导向为主,遇到真正不是资源操作的场景,允许在 URL 里出现一个动词化的子路径"(如 `POST /users/5/reset-password`),不是教条主义地追求理论纯粹性。

---

## KP4. GraphQL(概念性对比)

**签名/是什么:**

```
GraphQL:客户端在一次请求里,用一份查询语句精确声明"我要哪些字段、哪些关联资源",
        服务器一次性返回恰好匹配这份声明的数据 —— 不多不少。
对比 REST 的两个典型问题:
  - 过度获取(Over-fetching):REST 端点返回固定字段集合,客户端只需要其中2个字段,却要接收全部10个。
  - 请求不足(Under-fetching):获取关联数据(如"用户+他的文章列表")往往需要对 REST 端点发起多次请求。
```

**一句话:** GraphQL 把"要哪些数据"的决定权从服务器端(REST 端点的固定响应结构)转移到客户端(每次查询自己声明字段),用一次请求解决 REST 里常见的"字段拿多了"或者"关联数据要跑好几趟"的问题。

**底层机制/为什么这样设计:** REST 的每个端点通常返回一个预先设计好的固定字段集合,这在客户端的实际需求和端点设计者的预期完全吻合时没有问题,但真实产品迭代中客户端的需求(比如某个页面只想展示用户名和头像,不需要其他 8 个字段)和端点的固定返回结构经常对不齐——要么端点返回了用不上的多余字段(过度获取,浪费带宽),要么端点没提供某个关联资源、客户端必须再发一次请求单独去拿(请求不足,增加往返延迟)。GraphQL 用一个统一的查询入口(通常只有一个 URL,如 `/graphql`)替代"每种资源一个端点"的模式,客户端在请求体里用 GraphQL 查询语言精确描述想要的字段结构(包括跨资源的关联字段,比如"要这个用户的 name,以及他的 posts 里每篇的 title"),服务器端的解析器(resolver)负责把这份声明式的字段需求翻译成内部对多个数据源的实际查询、拼装成客户端要求的形状返回。这个设计把"数据整形"的责任从多次 REST 往返、客户端自己拼装,转移到服务器一次性完成,代价是服务器端 resolver 层的实现复杂度显著上升(需要处理任意字段组合的查询,还要小心设计不当的深度嵌套查询导致的"N+1 查询"性能问题)。

**AI 研究/工程场景:** 复杂的模型管理控制台前端(比如需要同时展示"这次训练任务的状态、使用的数据集信息、关联的评估指标"这类横跨多个后端资源的聚合视图)是 GraphQL 的典型适用场景——如果用 REST,前端可能要连续发起 3~4 次请求分别取任务、数据集、指标信息再自己拼装;GraphQL 允许前端一次查询声明"我要这几块关联数据",后端一次性组装返回,减少前端网络往返和拼装逻辑复杂度。

**可运行例子(验证环境:`.venv`,在同一份数据上真实对比 REST 需要的请求次数与 GraphQL 风格单次查询需要的请求次数):**

```python
users_db = {"1": {"name": "alice"}}
posts_db = {"1": [{"title": "hello"}, {"title": "world"}]}

# REST 风格:获取"用户+他的文章列表"需要两次独立的往返 —— 先查用户,再单独查文章。
rest_round_trips = 0


def rest_get_user(uid):
    global rest_round_trips
    rest_round_trips += 1
    return users_db[uid]


def rest_get_posts(uid):
    global rest_round_trips
    rest_round_trips += 1
    return posts_db[uid]


user = rest_get_user("1")
posts = rest_get_posts("1")
combined_rest = {"name": user["name"], "posts": posts}
assert rest_round_trips == 2

# GraphQL 风格:一次查询,声明要 name + posts 两个字段,resolver 内部一次性把关联数据拼好返回。
graphql_round_trips = 0


def graphql_query(uid, fields):
    global graphql_round_trips
    graphql_round_trips += 1
    result = {}
    if "name" in fields:
        result["name"] = users_db[uid]["name"]
    if "posts" in fields:
        result["posts"] = posts_db[uid]
    return result


combined_gql = graphql_query("1", fields=["name", "posts"])
assert graphql_round_trips == 1
assert combined_gql == combined_rest, "两种方式最终拿到的数据必须完全一致,只是请求次数不同"
assert graphql_round_trips < rest_round_trips

print(f"REST needed {rest_round_trips} round trips (user, then posts separately); "
      f"GraphQL-style needed {graphql_round_trips} round trip for the exact same combined data")
```

**面试怎么问+追问链:**
- Q:GraphQL 解决了 REST 的过度获取/请求不足问题,那是不是所有 REST API 都应该迁移到 GraphQL?
  - 追问1(方案批判迭代轴):GraphQL 有没有引入 REST 原本没有的新问题?
  - 深挖追问:有——最典型的是"N+1 查询问题"从客户端的多次网络请求,转移成了服务器 resolver 内部的多次数据库查询(比如查询 100 个用户各自的 posts,如果 resolver 天真地为每个用户单独查一次数据库,内部就是 101 次数据库查询,只是这个开销从"客户端可见的网络往返"转移到了"客户端不可见的服务器内部延迟",需要 DataLoader 这类批量合并查询的技术专门优化);另外 GraphQL 因为查询灵活,天然不利于 HTTP 层缓存(REST 的 GET 请求可以被 CDN/浏览器按 URL 缓存,GraphQL 的查询通常是 POST 请求体里的自由文本,传统 HTTP 缓存机制难以直接套用)。这条追问检验候选人是否把 GraphQL 当作银弹,还是理解它是用一类问题交换另一类问题,需要结合具体场景判断值不值。

**常见坑:**
- 认为 GraphQL 天生比 REST"性能更好"——GraphQL 减少的是网络往返次数,不代表服务器内部处理开销更小(甚至可能因为要支持任意字段组合查询而增加复杂度和 N+1 问题风险);性能对比必须端到端衡量(网络延迟 + 服务器处理时间),而不是只看"请求次数变少了"这一个维度。

---

## KP5. 长连接保活心跳设计

**签名/是什么:**

```
心跳(Heartbeat):长连接的一方定期发送一个轻量探测消息(如 WebSocket 的 Ping 帧),
                 对方必须在约定时间内回应(如 Pong 帧),连续若干次没有回应即判定连接已死,主动关闭重连。
```

**一句话:** 心跳机制解决的问题是"TCP 连接在网络异常(比如中间路由器悄悄丢弃、NAT 映射过期)情况下可能已经名存实亡,但操作系统的 TCP 状态机毫不知情,应用层必须自己主动探测才能及时发现连接真的死了"。

**底层机制/为什么这样设计:** TCP 连接理论上是可靠的字节流,但这个可靠性建立在"双方都还在正常工作、网络路径畅通"的前提上——真实网络环境里存在大量会让连接"悄悄死掉但双方都不知道"的场景:客户端网络切换(WiFi 断开切 4G)导致原有连接彻底不可达却没有任何一方发送 FIN/RST、中间的 NAT 网关因为长时间没有流量经过而清除了地址映射表项、对端进程崩溃但操作系统来不及发送 FIN。如果没有心跳机制,应用层唯一能发现连接已死的方式是"尝试发送数据后收到错误"或者"永远等不到期望的响应,但不知道要等多久才能确认对方真的挂了",这在长连接场景(可能几小时甚至几天没有真实业务数据往来)下完全不可接受。心跳的解法是主动、周期性地发一个极小的探测包,并设定一个明确的超时窗口——如果连续 N 次心跳都没有收到回应,不管 TCP 连接的操作系统状态显示什么,应用层直接判定这条连接已经不可用,主动关闭并触发重连逻辑,而不是被动等待可能永远不会到来的错误通知。

**AI 研究/工程场景:** 流式推理场景(客户端保持一条长连接持续接收模型逐 token 输出)如果没有心跳机制,一旦网络中途异常但连接没有被任何一方显式关闭,客户端可能会无限期"卡住"等待更多 token、永远不知道连接已经名存实亡,导致用户界面一直显示"生成中"却再也不会更新——这是实时流式 AI 应用里必须处理的一类真实故障场景,心跳超时检测是让客户端及时发现异常并给用户明确反馈("连接已断开,请重试")的基础机制。

**可运行例子(验证环境:`.venv`,真实验证存活的对端能持续应答心跳、已死的对端能被超时机制真实检测出来):**

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


def alive_server(stop_evt):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(1)
    srv.settimeout(3)
    conn, _ = srv.accept()
    conn.settimeout(0.5)
    while not stop_evt.is_set():
        try:
            data = conn.recv(64)
            if not data:
                break
            if data == b"PING":
                conn.sendall(b"PONG")
        except socket.timeout:
            continue
    conn.close()
    srv.close()


stop_evt = threading.Event()
t = threading.Thread(target=alive_server, args=(stop_evt,), daemon=True)
t.start()
time.sleep(0.2)

c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
c.connect((HOST, port))
c.settimeout(1)

# 存活的对端:每次心跳都能在超时窗口内收到真实的 PONG。
missed = 0
for _ in range(3):
    c.sendall(b"PING")
    try:
        resp = c.recv(64)
        if resp != b"PONG":
            missed += 1
    except socket.timeout:
        missed += 1
    time.sleep(0.1)
assert missed == 0, f"alive peer should answer every heartbeat, missed={missed}"

stop_evt.set()
t.join(timeout=2)
c.close()

# 已死的对端:服务器已经不在监听这个端口了 -> 心跳发送/连接必须真实地失败或超时。
dead_detected = False
try:
    c2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c2.settimeout(1)
    c2.connect((HOST, port))
    c2.sendall(b"PING")
    resp = c2.recv(64)
    if not resp:
        dead_detected = True
except (ConnectionRefusedError, socket.timeout, OSError):
    dead_detected = True
assert dead_detected
print(f"alive peer answered {3-missed}/3 heartbeats; dead peer correctly detected via connection failure/timeout")
```

**面试怎么问+追问链:**
- Q:心跳间隔设多长合适?
  - 追问1(工程约束递增轴):如果这是一个有几百万并发长连接的服务端(比如即时通讯网关),心跳间隔的选择要考虑什么额外因素?
  - 深挖追问:心跳间隔太短(比如每秒一次),单机维持的连接数越多,心跳本身产生的流量和 CPU 唤醒开销就越可观——几百万连接每秒都要收发一次心跳包,这本身就是不小的负载;心跳间隔太长,连接异常后要等更久才能被发现,用户体验受损。真实系统通常会采用比客户端标准更长的心跳间隔(比如 30 秒到几分钟),并且很多网关会给不同连接错开心跳时间点(避免所有连接的心跳请求在同一时刻突刺),这是候选人需要意识到"心跳频率"本身也是一个需要结合并发规模权衡的系统设计参数,不是一个孤立选择。

**常见坑:**
- 把"心跳"和"TCP 自带的 Keepalive 选项"(操作系统内核层面的 `SO_KEEPALIVE`)混为一谈——内核 TCP Keepalive 默认间隔通常长达 2 小时,且不同操作系统/网络环境的行为不完全一致,不受应用层直接控制,大多数需要及时发现连接异常的场景都会在应用层自己实现心跳逻辑,而不是依赖内核默认的 Keepalive 机制。

---

## KP6. CDN 工作原理(边缘节点 / 回源 / 缓存策略)

**签名/是什么:**

```
边缘节点(Edge Node):CDN 服务商在全球各地部署的缓存服务器,离终端用户的网络距离比源站近得多。
回源(Origin Fetch):边缘节点本地没有缓存命中的内容时,代替用户去源站拉取一次,
                    拉到后既返回给用户,也在本地缓存一份供后续同一份内容的其他用户直接复用。
```

**一句话:** CDN 的核心价值是"让大多数请求命中离用户最近的边缘缓存,只有少数真正的缓存未命中才需要长途跋涉打到源站",边缘节点本质上是一层分布在全球的、面向用户的共享缓存。

**底层机制/为什么这样设计:** 如果所有用户都直接访问源站,一是每个用户请求都要承受源站物理位置带来的网络延迟(跨国访问可能是几百毫秒),二是源站服务器要承受全球全部用户的请求压力。CDN 的解法是在用户和源站之间插入一层地理上分布式的缓存网络:用户被(通常通过 [09 类 KP5](09-dns-resolution.md) 的 GeoDNS 或者 Anycast IP 路由)引导到离自己最近的边缘节点,如果这个节点本地已经缓存了用户想要的内容(常见于静态资源:图片、JS/CSS、视频分片),直接从边缘节点返回,用户完全感受不到源站的存在,延迟只取决于到最近边缘节点的距离。如果边缘节点没有缓存命中(内容从没被请求过、或者缓存已过期),这个节点才会代替用户向源站发起一次"回源"请求,把结果既返回给用户,也顺手缓存下来——后续其他用户请求同一份内容,即使是第一次访问这个边缘节点,也能直接命中刚刚缓存下来的副本。这个设计本质上是 KP3/文件 07 KP4 讨论的"缓存"思路应用在"地理分布式反向代理"这个更大的尺度上:用共享缓存吸收掉绝大多数重复请求,只让真正的"首次请求"或"缓存失效后的请求"触碰到源站。

**AI 研究/工程场景:** 大模型权重文件、tokenizer 词表这类体积巨大但发布后基本不变的静态资源,是 CDN 强缓存(见 [07 类 KP4](07-http-evolution.md))的典型受益场景——把模型文件放在 CDN 后面分发,全球各地的推理服务实例都能从最近的边缘节点快速拉取,而不必全部直接打到承载原始文件的源存储,这对模型分发这种"一次发布、全球海量拉取"的场景是显著的带宽和延迟优化。

**可运行例子(验证环境:`.venv`,真实验证边缘缓存命中避免回源、未命中时真实触发回源并把结果缓存下来供后续复用):**

```python
origin_fetch_count = {"n": 0}
origin_store = {"/logo.png": b"PNG-BYTES-ORIGIN-CONTENT"}
edge_cache = {}


def origin_fetch(path):
    origin_fetch_count["n"] += 1  # 每调用一次,代表真的打了一次源站(回源)
    return origin_store[path]


def edge_get(path):
    if path in edge_cache:
        return edge_cache[path], "HIT"
    content = origin_fetch(path)
    edge_cache[path] = content
    return content, "MISS(origin fetched)"


# 这个边缘节点第一次收到这份内容的请求:缓存未命中,必须真实回源一次。
c1, status1 = edge_get("/logo.png")
assert status1.startswith("MISS") and origin_fetch_count["n"] == 1

# 后续对同一个边缘节点的重复请求(哪怕来自不同用户):全部命中缓存,源站完全不会被再次打扰。
for _ in range(5):
    c2, status2 = edge_get("/logo.png")
    assert status2 == "HIT" and c2 == c1
assert origin_fetch_count["n"] == 1, "5次重复命中不应该触发5次额外回源"

print(f"1st request={status1}(origin fetches={origin_fetch_count['n']}), "
      f"next 5 requests all HIT, origin fetch count stayed at {origin_fetch_count['n']}")
```

**面试怎么问+追问链:**
- Q:CDN 适合缓存所有类型的内容吗?
  - 追问1(规模递增轴):如果源站的内容是"每个用户看到的都不一样的个性化首页",CDN 边缘缓存还有意义吗?
  - 深挖追问(方案批判迭代轴):对于纯个性化、每个用户响应都不同的内容,传统的"整页缓存"意义不大(缓存命中率会趋近于零,因为几乎不会有两个用户请求出完全相同的响应),但 CDN 仍然可以发挥作用的方式包括:①缓存页面里的公共静态部分(图片、CSS、JS,这些跨用户是相同的),动态个性化部分单独走一个不缓存的 API 请求;②边缘计算(在边缘节点直接执行部分个性化逻辑,而不是所有个性化计算都要回源)。这条追问检验候选人是否理解 CDN 缓存收益和内容的"跨用户复用程度"直接相关,不是无脑"上了 CDN 就一定变快"。

**常见坑:**
- 以为 CDN 只能缓存静态内容——现代 CDN 服务普遍支持给动态内容设置短 TTL 缓存(比如新闻网站首页缓存 30 秒,足以吸收掉短时间内的重复请求,同时保证内容不会过时太久),以及边缘计算能力(在边缘节点执行简单的个性化逻辑),"CDN=只服务静态文件"是过时的认知。

---

## KP7. 负载均衡算法(轮询 / 加权 / 最少连接 / 一致性哈希)

**签名/是什么:**

```
轮询(Round Robin):按顺序依次把请求分给每个后端,均匀轮转。
加权轮询(Weighted Round Robin):按预设权重比例分配 —— 权重3的后端接收的请求数约为权重1后端的3倍。
最少连接(Least Connections):把新请求分给"当前活跃连接数最少"的后端,实时反映负载状态,不是静态轮转。
一致性哈希(Consistent Hashing):按 key 的哈希值分配到环形空间上最近的节点 ——
   交叉引用:真实虚拟节点负载均衡实测代码见 dsa-deep-dive 20类,不在本类重复推导。
```

**一句话:** 轮询和加权轮询是"不看后端实时状态、按预先设定的规则静态分配"的策略,最少连接是"实时感知后端当前负载、动态决策"的策略,一致性哈希解决的是完全不同的问题——"同一个 key 要稳定路由到同一个后端"(见 dsa-deep-dive 20 类的分布式缓存分片场景)。

**底层机制/为什么这样设计:** 轮询假设所有后端处理能力相同、每个请求的处理成本也大致相同,这个假设在异构后端(有的机器配置更好)或者请求成本差异很大(有的请求是轻量查询,有的是重量级计算)的场景下会失真,加权轮询通过给每个后端配置一个权重系数缓解"处理能力不同"这一半的问题,但仍然不感知"某个后端这一刻实际负载已经很高"这种动态状态。最少连接策略直接观测每个后端当前的活跃连接数(近似代表当前负载),把新请求导向负载最轻的那一个,能更好地适应"部分请求处理时间远超预期,导致某个后端突然堆积"这种轮询类算法完全无法感知的动态场景——代价是负载均衡器需要维护并实时更新每个后端的连接计数状态,比无状态的轮询实现更复杂。一致性哈希则完全是另一个维度的问题:它要解决的不是"怎么让每台机器负载均匀",而是"给定同一个 key,能不能稳定地路由到同一台机器"(比如分布式缓存场景,同一个用户的会话数据应该稳定落在同一个缓存节点,不然缓存命中率会因为路由抖动而大幅下降)——这也是为什么一致性哈希经常用在有状态的分片场景(缓存分片、数据库分片),而轮询/最少连接更多用在无状态的请求分发场景(比如 Web 服务器集群)。

**AI 研究/工程场景:** 模型 serving 网关面对不同复杂度的推理请求(短文本 vs 长文本、不同批大小)时,单纯的轮询容易把好几个耗时长的大请求恰好分配到同一个 worker,造成局部过载,而这个 worker 完全没有机会"提前声明自己很忙"——这正是最少连接(或者更进一步,基于实时 GPU 利用率/队列深度的自定义负载感知策略)在推理网关场景比简单轮询更受青睐的原因;一致性哈希则用在需要"同一个用户/会话稳定路由到同一个 worker"的场景(比如该 worker 缓存了这个会话的 KV cache,路由抖动会导致缓存失效,见 dsa-deep-dive 20 类讨论的缓存分片一致性权衡)。

**可运行例子(验证环境:`.venv`,真实验证轮询绝对均匀、加权轮询严格按比例分配、最少连接能对负载变化做出实时反应):**

```python
# 轮询:严格均匀轮转。
backends = ["A", "B", "C"]
rr_i = {"n": 0}


def round_robin():
    b = backends[rr_i["n"] % len(backends)]
    rr_i["n"] += 1
    return b


counts_rr = {"A": 0, "B": 0, "C": 0}
for _ in range(30):
    counts_rr[round_robin()] += 1
assert counts_rr == {"A": 10, "B": 10, "C": 10}, counts_rr

# 加权轮询:权重3的后端应该拿到约3倍于权重1后端的请求量。
weighted_backends = [("A", 3), ("B", 1)]
expanded = []
for name, w in weighted_backends:
    expanded.extend([name] * w)
wrr_i = {"n": 0}


def weighted_round_robin():
    b = expanded[wrr_i["n"] % len(expanded)]
    wrr_i["n"] += 1
    return b


counts_wrr = {"A": 0, "B": 0}
for _ in range(40):
    counts_wrr[weighted_round_robin()] += 1
assert counts_wrr["A"] == 30 and counts_wrr["B"] == 10, counts_wrr
assert counts_wrr["A"] / counts_wrr["B"] == 3

# 最少连接:必须对"实时"活跃连接数变化做出反应,不是死板轮转。
active_conns = {"A": 0, "B": 0, "C": 0}


def least_connections_pick():
    return min(active_conns, key=lambda k: active_conns[k])


def start_conn(name):
    active_conns[name] += 1


def end_conn(name):
    active_conns[name] -= 1


picks = []
for _ in range(3):
    p = least_connections_pick()
    start_conn(p)
    picks.append(p)
assert set(picks) == {"A", "B", "C"}, "全部空闲时,前3次挑选必须分散到全部3个后端"

# A、B 此刻各有1个连接在忙,C 刚释放变回0个连接 -> 下一次必须优先选 C。
end_conn("C")
p4 = least_connections_pick()
assert p4 == "C", f"必须挑选当前活跃连接数最少的后端,得到{p4}"

print(f"round-robin={counts_rr}(perfectly even), "
      f"weighted(3:1)={counts_wrr}(ratio={counts_wrr['A']/counts_wrr['B']:.0f}), "
      f"least-connections correctly picked freed-up backend '{p4}' over the still-busy ones")
```

**面试怎么问+追问链:**
- Q:什么场景下你会选最少连接而不是轮询?
  - 追问1:一致性哈希算不算负载均衡算法的一种?
  - 深挖追问(决策依据追问轴):这个问题本身有陷阱——一致性哈希的首要设计目标是"路由稳定性"(同一个 key 稳定落到同一节点),负载均衡只是它的一个附带考量(通过虚拟节点缓解不均衡,具体真实测量见 [dsa-deep-dive 20 类](../dsa-deep-dive/20-advanced-interview-depth.md)的一致性哈希负载分布实测代码);轮询/加权轮询/最少连接的首要设计目标就是"负载均衡"本身,不需要考虑"同一个请求要不要稳定路由到同一节点"这件事。候选人如果能明确指出两者"设计目标的优先级不同",而不是笼统地把四种算法并列成"一类东西的四个选项",说明理解到位;笼统并列是最常见的、区分度不高的回答。

**常见坑:**
- 认为最少连接总是比轮询"更好"、应该无条件优先选择——最少连接需要维护额外的状态(每个后端的实时连接计数)并在每次分配决策时读取,在后端处理成本高度同质、请求量极大的场景下,这点额外开销可能得不偿失,轮询的无状态、O(1) 决策优势在这类场景反而更合适;算法选择要匹配请求成本的同质/异质程度,不是无脑追求"更智能"的算法。

---

*本篇完成:2026-07-14,7 个知识点。验证环境:全部 7 个可运行代码块均为 `.venv`(WebSocket真实RFC6455握手+真实帧收发、protobuf风格编码与JSON真实字节对比、RESTful真实资源URL+状态码语义、REST与GraphQL风格真实往返次数对比、心跳存活/死亡真实检测、CDN边缘缓存命中/回源真实验证、轮询/加权轮询/最少连接算法真实分布验证);KP7一致性哈希部分交叉引用 [dsa-deep-dive 20类](../dsa-deep-dive/20-advanced-interview-depth.md)已有真实实测代码,不重复推导。板块 IV(应用层协议)全部完成(07-10,4/4)。*
