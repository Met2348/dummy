# 01 · 分层模型与链路层基础(Layering & Link Layer Basics)

> 总览见 [00-roadmap.md](00-roadmap.md)。本篇覆盖 6 个知识点,是计算机网络系列的地基板块——不是重点,但后续所有板块(IP/TCP/HTTP)都建立在"分层"这个设计哲学之上,值得先讲清楚。

---

## 1. OSI七层模型与TCP/IP四层/五层模型对比

**签名/是什么:**
```
OSI七层(理论模型): 物理层 -> 数据链路层 -> 网络层 -> 传输层 -> 会话层 -> 表示层 -> 应用层
TCP/IP四层(工程实现): 网络接口层 -> 网际层 -> 传输层 -> 应用层
```

**一句话:** OSI 是 ISO 组织在 1984 年设计的理论参考模型,追求学术上的概念纯粹性(每层职责严格单一);TCP/IP 是 Internet 工程实践中先有协议、后总结出的模型,更贴合真实实现——这也是为什么今天工程师说"四层模型"或"五层模型"(物理/链路/网络/传输/应用)时,指的都是 TCP/IP 体系,而不是 OSI。

**底层机制/为什么这样设计:** OSI 的会话层(管理会话:建立/维护/终止)、表示层(数据格式转换:加密/压缩/编码)在真实工程实践中很少被实现成独立的协议层——这两层的职责在 TCP/IP 世界里被"下放"给了应用层协议自己处理(比如 HTTPS 的 TLS 握手承担了"表示层"的加密职责,HTTP Cookie/Session 承担了"会话层"的会话管理职责)。这不是 TCP/IP "偷工减料",而是工程实践发现:把这些职责精确切分成独立层,不如让应用协议按自己的需要灵活组合更实用。这也解释了一个常见的面试追问——"为什么很多人说网络是五层不是四层":五层模型把 TCP/IP 的"网络接口层"进一步拆成"数据链路层"+"物理层",是教学中常见的四层/五层混合叫法,和 OSI 的七层不是一回事。

**AI研究/工程场景:** 排查模型 serving 服务的网络延迟问题时,分层思维能帮你快速定位问题在哪一层——是物理链路本身丢包(链路层)、路由绕远(网络层)、TCP 重传(传输层),还是 gRPC/HTTP 应用层序列化耗时,每一层需要用完全不同的工具排查(链路层看 `ip link`/交换机日志,传输层看 `ss`/`tcpdump`,应用层看服务本身的 trace)。

**可运行例子(环境:`.venv`):**
```python
osi_layers = ["physical", "data_link", "network", "transport", "session", "presentation", "application"]
tcpip_layers = {
    "network_interface": ["physical", "data_link"],
    "internet": ["network"],
    "transport": ["transport"],
    "application": ["session", "presentation", "application"],
}

covered = set()
for osi_subset in tcpip_layers.values():
    covered.update(osi_subset)
assert covered == set(osi_layers), "TCP/IP四层必须完整覆盖OSI七层,不能有遗漏"
assert len(tcpip_layers) == 4
assert len(osi_layers) == 7
assert len(tcpip_layers["application"]) == 3, "TCP/IP应用层合并了OSI的会话/表示/应用三层"
print("OK: TCP/IP four layers fully cover OSI seven layers")
print("OK: TCP/IP application layer merges OSI session+presentation+application")
```

**面试怎么问+追问链:**
- Q: "OSI 和 TCP/IP 模型有什么区别?" → 追问1: "为什么现在业界基本只说 TCP/IP 模型,OSI 很少被直接用来设计协议?"(**决策依据追问轴**:考察是否理解 OSI 是"先设计模型再套协议"、TCP/IP 是"先有协议再总结模型"这一历史顺序差异) → 深挖追问: "HTTPS 里的 TLS 握手,你觉得它在 OSI 模型里算哪一层?"(没有标准答案,考察能否论证:TLS 工作在传输层之上、应用层协议之下,常被非正式称为"表示层"或独立的"安全层",重点是候选人能不能讲清楚为什么,而不是背答案)

**常见坑:** 混淆"四层"和"五层"的说法就以为二者是不同的模型——本质上是同一个 TCP/IP 体系,只是"网络接口层"要不要进一步拆分成"数据链路层+物理层"这一个教学习惯问题,不是两套不同的协议栈。

---

## 2. 封装与解封装(PDU命名:段/包/帧/比特)

**签名/是什么:**
```
应用层数据 -> [传输层加TCP/UDP首部] Segment/Datagram
           -> [网络层加IP首部] Packet
           -> [链路层加以太网首部+尾部] Frame
           -> [物理层] Bit流
```

**一句话:** 数据从应用层往下传递时,每一层协议都会在上一层给的数据前面(有时后面)包一层自己的首部(header),这个逐层打包的过程叫封装(encapsulation);到达接收方后逐层拆包、剥掉对应层的首部,叫解封装(decapsulation)——每一层只关心自己加的那层包装,不关心里面到底装的是什么,这正是"分层"能够独立演化、互不干扰的根本原因。

**底层机制/为什么这样设计:** 这是一种"俄罗斯套娃"式设计,好处是层与层之间完全解耦——网络层(IP)完全不需要知道上面跑的是 TCP 还是 UDP,只需要把收到的 payload 原样往下传;链路层也完全不关心里面是不是 IP 包。这种设计使得"在 IP 之上新增一种传输层协议"(比如 QUIC 选择基于 UDP 而不是 TCP)完全不需要改动网络层和链路层的任何实现,新协议只要按照"上一层交给我什么,我原样封装转发"的约定来做即可。

**AI研究/工程场景:** 理解封装开销对理解"为什么小请求的网络开销占比反而更高"很关键——一次 gRPC 调用如果只传几十字节的推理结果,以太网首部(14字节)+IP首部(20字节)+TCP首部(20字节)+TLS记录层开销 这些固定开销可能比真正的业务数据还大,这也是为什么高频小请求场景(比如逐 token 流式返回)要考虑批量化或者长连接复用,而不是每次都走完整的连接建立+封装开销。

**可运行例子(环境:`.venv`):**
```python
def encapsulate(payload: bytes) -> bytes:
    transport_segment = b"TCPHDR20B__" + payload
    network_packet = b"IPHDR20B____" + transport_segment
    link_frame = b"ETHHDR14B_" + network_packet + b"FCS4B"
    return link_frame

def decapsulate(frame: bytes) -> bytes:
    network_packet = frame[len(b"ETHHDR14B_"):-len(b"FCS4B")]
    transport_segment = network_packet[len(b"IPHDR20B____"):]
    payload = transport_segment[len(b"TCPHDR20B__"):]
    return payload

original = b"HELLO_APPLICATION_DATA"
frame = encapsulate(original)
assert frame != original
assert len(frame) > len(original), "封装后必然比原始payload更大(逐层加首部)"
recovered = decapsulate(frame)
assert recovered == original, f"解封装后应恢复原始payload, got {recovered}"
print(f"OK: encapsulated {len(original)} bytes into {len(frame)} bytes, recovered exactly on decapsulation")
```

**面试怎么问+追问链:**
- Q: "描述一下一个 HTTP 请求从应用层到网卡发出去,经历了哪些封装?" → 追问1: "如果这个请求要经过一个路由器转发,路由器会拆开到哪一层再重新封装?"(**规模递增轴**变体,考察对"路由器工作在网络层"的理解——路由器会拆到网络层看 IP 首部决定转发路径,但不会拆开 TCP 首部,拆到链路层重新加一个新的链路层首部再发出去,因为下一跳的链路层地址变了但网络层信息不变) → 深挖追问: "为什么路由器要换链路层首部,却不用换网络层首部?"(考察对 MAC 地址是"逐跳"有效、IP 地址是"端到端"有效这一核心区别的理解)

**常见坑:** 以为"封装"只是简单的"拼接字符串"——实际每一层的首部格式都是严格的二进制协议(字段有固定的偏移和长度,比如 IP 首部的版本号占4比特、总长度占16比特),不是任意格式的文本包装,解析时必须严格按照协议定义的字节布局来解包,错一个字节offset整个报文就解析错乱。

---

## 3. 以太网帧结构与MAC地址

**签名/是什么:**
```
import struct
struct.pack("!6s6sH", dst_mac_bytes, src_mac_bytes, ethertype)  # 14字节以太网首部(纯示意,非可执行)
```

**Python `struct` 模块语法(全系列第一次出现 `struct.pack`,这里集中说明一次):** `struct` 是 Python 标准库里专门处理"C 语言风格二进制数据"的模块——网络协议的报文格式本质上是"从第几个字节到第几个字节是什么类型的字段"这样严格的定长二进制布局,不是普通的 Python 对象。`struct.pack(fmt, *values)` 按 `fmt` 描述的格式,把若干个 Python 值(整数、字节串等)打包成一段连续的字节串;`struct.unpack(fmt, data)` 是反过来,把字节串按同样的格式解析回一组 Python 值。`fmt` 格式字符串的写法:第一个字符是字节序前缀,`!` 代表"网络字节序"(大端,big-endian,是网络协议的标准约定);后面每个字符描述一个字段的类型+长度,本系列最常用到的几个是 `B`=1 字节无符号整数、`H`=2 字节无符号短整数、`L`=4 字节无符号长整数、`6s`=6 字节定长字节串(数字+`s` 表示"定长字节串",数字就是字节数)。所以这里的 `"!6s6sH"` 读作:网络字节序 + 一个 6 字节字段 + 另一个 6 字节字段 + 一个 2 字节无符号整数字段,正好对应"目的 MAC(6 字节) + 源 MAC(6 字节) + EtherType(2 字节)"这三个字段拼成的 14 字节以太网首部——把这些格式字符拼起来,就是在精确描述一段二进制数据每一段各是什么类型、占几个字节。后续知识点和其他文件里再遇到 `struct.pack`/`struct.unpack` 的格式字符串,不再重复解释这套语法,可以回来查这里。

**一句话:** 以太网帧首部由 6 字节目的 MAC 地址 + 6 字节源 MAC 地址 + 2 字节 EtherType(标识上层协议类型,比如 `0x0800` 代表 IPv4、`0x0806` 代表 ARP)组成,MAC 地址是烧录在网卡硬件里的 48 位全球唯一标识符,工作在链路层,只在"同一个二层网络内"有意义。

**底层机制/为什么这样设计:** MAC 地址和 IP 地址的本质区别是"作用域"不同——MAC 地址是"逐跳"(hop-by-hop)有效的:一个帧每经过一个路由器,链路层首部(包括目的 MAC)都会被完全替换成"到下一跳"的 MAC 地址;而 IP 地址是"端到端"(end-to-end)有效的:从源主机到目的主机,IP 首部里的源/目的 IP 地址全程不变(NAT 场景除外)。这个设计让链路层可以自由采用不同的物理介质技术(以太网/WiFi/光纤),路由器负责在不同链路技术之间"翻译"链路层地址,而网络层完全不需要感知底层链路的具体实现。

**AI研究/工程场景:** 数据中心内部 GPU 服务器之间的高速互联(比如通过 RoCEv2,RDMA over Converged Ethernet)仍然基于标准以太网帧格式封装 RDMA 报文——理解以太网帧结构是理解这类"融合以太网"技术如何在标准网络设备上跑高性能通信协议的基础(RoCE/InfiniBand 的物理层选型细节属于 [learning/cluster-networking](../../learning/cluster-networking/README.md) 的范围,这里只讲通用以太网帧格式本身)。

**可运行例子(环境:`.venv`):**
```python
import struct

def mac_to_bytes(mac_str: str) -> bytes:
    return bytes(int(x, 16) for x in mac_str.split(":"))

def build_eth_header(dst_mac: str, src_mac: str, ethertype: int) -> bytes:
    return struct.pack("!6s6sH", mac_to_bytes(dst_mac), mac_to_bytes(src_mac), ethertype)

hdr = build_eth_header("AA:BB:CC:DD:EE:FF", "11:22:33:44:55:66", 0x0800)
assert len(hdr) == 14, f"Ethernet header must be 14 bytes (6+6+2), got {len(hdr)}"

dst, src, etype = struct.unpack("!6s6sH", hdr)
assert dst == mac_to_bytes("AA:BB:CC:DD:EE:FF")
assert src == mac_to_bytes("11:22:33:44:55:66")
assert etype == 0x0800, "0x0800 is EtherType for IPv4"
print(f"OK: Ethernet header is 14 bytes, parsed EtherType=0x{etype:04X} (IPv4)")
```

**面试怎么问+追问链:**
- Q: "MAC 地址和 IP 地址分别在什么场景下起作用?" → 追问1: "一个数据包从你的笔记本发往一个跨国服务器,中间经过 10 个路由器,MAC 地址会变化几次?IP 地址呢?"(**规模递增轴**:考察是否理解 MAC 逐跳更换、IP 端到端不变——正确答案是 MAC 地址理论上每跳都变,IP 地址全程不变) → 深挖追问: "如果其中一跳是通过 WiFi 而不是有线以太网转发的,这个'逐跳更换链路层首部'的机制还成立吗?"(考察是否理解链路层协议可以是异构的,路由器负责在不同链路技术间转换,机制本质不变)

**常见坑:** 误以为 MAC 地址"全球唯一"意味着可以用来做设备的可靠身份认证——MAC 地址在软件层面是可以被修改的(称为 MAC 欺骗/MAC spoofing),烧录在硬件里的出厂 MAC 只是默认值而非不可篡改的信任根,安全敏感场景不能仅凭 MAC 地址做身份验证。

---

## 4. ARP协议(地址解析协议)

**签名/是什么:**
```
ARP请求(广播): "谁是 192.168.1.1? 请把你的MAC地址告诉 11:22:33:44:55:66"
ARP响应(单播): "192.168.1.1 的MAC地址是 AA:BB:CC:DD:EE:01"
```

**一句话:** ARP(Address Resolution Protocol)负责把网络层的 IP 地址解析成链路层的 MAC 地址——因为以太网帧的发送必须知道目的 MAC 地址,但应用程序通常只知道目的 IP 地址,ARP 就是这中间"翻译"的一步,解析结果会被缓存在本机的 ARP 表里避免每次都重新广播查询。

**底层机制/为什么这样设计:** ARP 请求用广播(目的 MAC 全 `FF:FF:FF:FF:FF:FF`)发送,因为发起方压根不知道目标机器在哪个端口/哪台交换机后面,只能让同一广播域内所有设备都收到这个请求,只有 IP 匹配的那台机器会以单播方式响应——这个"广播问、单播答"的模式和后面会讲的 DNS 递归查询(09类)在设计思路上有相似之处:发起方缺少定位信息时,先用更"粗暴"的方式触达潜在目标,拿到精确信息后转为高效的点对点通信。

**AI研究/工程场景:** 分布式训练集群节点间通信建立前也依赖 ARP 解析对端 MAC 地址——如果某台机器的 ARP 缓存出现异常(比如网络设备更换后旧的 IP-MAC 映射没有及时过期),会表现为"看起来像是训练卡住/网络分区",但实际排查下来是链路层地址解析失败,这类问题定位往往需要排除到链路层才能发现,是一类容易被误诊为"应用层/框架bug"的基础设施问题。

**可运行例子(环境:`.venv`):**
```python
import struct
import socket as sk

def mac_to_bytes(mac_str: str) -> bytes:
    return bytes(int(x, 16) for x in mac_str.split(":"))

def build_arp_packet(sender_mac, sender_ip, target_mac, target_ip, opcode):
    # HTYPE=1(Ethernet) PTYPE=0x0800(IPv4) HLEN=6 PLEN=4 OPER=opcode
    return struct.pack(
        "!HHBBH6s4s6s4s",
        1, 0x0800, 6, 4, opcode,
        mac_to_bytes(sender_mac), sk.inet_aton(sender_ip),
        mac_to_bytes(target_mac), sk.inet_aton(target_ip),
    )

# 主机 11:22:33:44:55:66 广播查询 192.168.1.1 的MAC
request = build_arp_packet("11:22:33:44:55:66", "192.168.1.10",
                            "00:00:00:00:00:00", "192.168.1.1", opcode=1)
assert len(request) == 28, f"ARP packet must be 28 bytes, got {len(request)}"
_, _, _, _, oper, _, _, _, _ = struct.unpack("!HHBBH6s4s6s4s", request)
assert oper == 1, "opcode=1 means ARP request"

# 网关 AA:BB:CC:DD:EE:01 响应,填入自己真实MAC
reply = build_arp_packet("AA:BB:CC:DD:EE:01", "192.168.1.1",
                          "11:22:33:44:55:66", "192.168.1.10", opcode=2)
_, _, _, _, oper2, gw_mac, gw_ip, my_mac, my_ip = struct.unpack("!HHBBH6s4s6s4s", reply)
assert oper2 == 2, "opcode=2 means ARP reply"

resolved_gateway_mac = ":".join(f"{b:02X}" for b in gw_mac)
assert resolved_gateway_mac == "AA:BB:CC:DD:EE:01"
print(f"OK: ARP request (28 bytes) -> gateway reply, resolved gateway MAC={resolved_gateway_mac}")
```

**面试怎么问+追问链:**
- Q: "ARP 请求为什么用广播而不是直接问网关?" → 追问1: "如果目标主机和发起方不在同一个子网,ARP 广播还能解析到目标 MAC 吗?"(**决策依据追问轴**:考察是否理解 ARP 广播的作用域被子网/广播域限制——跨子网时,发送方实际上是解析"默认网关"的 MAC 地址,由网关负责把包转发到目标子网,而不是直接解析远端主机的 MAC) → 深挖追问: "ARP 欺骗攻击(ARP spoofing)的原理是什么,为什么中间人攻击可以利用它?"(考察对 ARP 协议本身缺乏身份验证机制这一安全弱点的理解——任何设备都可以伪造 ARP 响应声称自己是某个 IP 对应的 MAC,被欺骗的主机会把流量发给攻击者而非真正的目标)

**常见坑:** 把 ARP 表项当作永久有效——真实系统里 ARP 缓存是有过期时间的(典型几分钟到几十分钟不等),过期后需要重新广播解析,如果调试网络问题时看到的是一条很久以前缓存的、已经过期但尚未清除的 ARP 记录,可能会得到误导性的排查结论。

---

## 5. 交换机工作原理(MAC地址表学习/转发,冲突域vs广播域)

**签名/是什么:**
```
交换机行为 = 学习(记录 源MAC -> 到达端口) + 转发决策(查表决定单播还是洪泛)
```

**一句话:** 二层交换机通过"偷看"每个经过的帧的源 MAC 地址来动态构建一张"MAC 地址 -> 端口"的转发表(这个过程叫 MAC 学习),之后如果收到的帧的目的 MAC 已经在表里,就直接单播转发到对应端口;如果目的 MAC 未知(表里没有),就洪泛(flood)到除入端口外的所有端口——这也是为什么交换机"用得越久,行为看起来越智能",本质是转发表被逐渐学满的过程。

**底层机制/为什么这样设计:** 早期以太网用集线器(Hub)简单地把信号广播到所有端口,任意两台设备同时发送都会造成信号冲突(需要 CSMA/CD 冲突检测机制),集线器连接的所有设备属于同一个"冲突域"。交换机通过为每个端口维护独立的转发决策,把冲突域缩小到"每个端口自己",消除了同一交换机下不同端口间的信号冲突问题——但交换机不会阻止广播帧(目的 MAC 全 F)扩散到所有端口,所以交换机连接的所有设备仍然共享同一个"广播域"。这正是为什么"冲突域"和"广播域"是两个不同粒度的概念:交换机能划分冲突域(缩小到端口级别),但不能划分广播域(需要靠 VLAN,见下一个知识点)。

**AI研究/工程场景:** 大规模 GPU 训练集群的网络拓扑设计(比如 fat-tree,详见 [learning/cluster-networking](../../learning/cluster-networking/README.md) 的拓扑专题)本质上是在"交换机的基础转发行为"之上,通过多层交换机的互联结构去解决"如何用有限端口数的交换机构建出支持海量节点、高对分带宽(bisection bandwidth)的大网络"这个问题——理解单台交换机最基础的学习/转发行为,是理解这类大规模拓扑设计动机的起点。

**可运行例子(环境:`.venv`):**
```python
class SimpleSwitch:
    def __init__(self, ports):
        self.ports = set(ports)
        self.mac_table = {}
        self.flood_count = 0
        self.unicast_count = 0

    def receive_frame(self, src_mac, dst_mac, arrival_port):
        self.mac_table[src_mac] = arrival_port
        if dst_mac in self.mac_table:
            self.unicast_count += 1
            return [self.mac_table[dst_mac]]
        else:
            self.flood_count += 1
            return [p for p in self.ports if p != arrival_port]

sw = SimpleSwitch(ports=[1, 2, 3])

out1 = sw.receive_frame("MAC_A", "MAC_B", arrival_port=1)
assert set(out1) == {2, 3}, "first time dst MAC unknown, must flood to all ports except arrival"

out2 = sw.receive_frame("MAC_B", "MAC_A", arrival_port=2)
assert out2 == [1], "switch already learned MAC_A is on port 1, should unicast not flood"

out3 = sw.receive_frame("MAC_A", "MAC_B", arrival_port=1)
assert out3 == [2], "after learning, should unicast to port 2, no more flooding"

assert sw.flood_count == 1 and sw.unicast_count == 2
print(f"OK: switch learning reduced flooding ({sw.flood_count}) in favor of unicast ({sw.unicast_count})")
```

**面试怎么问+追问链:**
- Q: "交换机怎么知道该往哪个端口转发帧?" → 追问1: "如果两台交换机之间接了两根网线互为冗余,会发生什么问题?"(**方案批判迭代轴**:考察是否知道"广播风暴/环路"问题——没有环路防护机制的话,广播帧会在环路里无限复制扩散,最终耗尽带宽;追问候选人应该提到生成树协议 STP 作为标准解法,这是刻意留给系统设计系列的伏笔,这里点到为止不展开 STP 算法细节) → 深挖追问: "生成树协议大致是怎么解决这个问题的?"(概念性追问,期待候选人能说出"选出一棵无环的逻辑拓扑树,阻塞掉多余链路"这个思路,不要求现场推导算法细节)

**常见坑:** 把"交换机隔离冲突域"和"交换机隔离广播域"混为一谈——这是两个不同粒度的概念:冲突域被交换机缩小到端口级别(几乎不用再担心冲突),但广播域默认覆盖整个交换网络(除非配置VLAN),"交换机能防止广播风暴"是一个常见的错误认知。

---

## 6. VLAN基础概念(虚拟局域网)

**签名/是什么:**
```
VLAN = 给交换机端口打上"逻辑分组"标签,广播/未知目的的帧只在同一个VLAN标签内洪泛
```

**一句话:** VLAN(Virtual LAN)通过在交换机端口(或者以太网帧本身,802.1Q tag)上标记"逻辑网络编号",把原本同一物理交换机下的所有设备,人为切分成多个相互隔离的广播域——即使它们插在同一台物理交换机上,不同 VLAN 之间的广播流量也不会互相可见,如果要跨 VLAN 通信,必须经过路由器或三层交换机(这一步等价于跨越了不同的网络层子网)。

**底层机制/为什么这样设计:** 沿用上一个知识点的结论——单纯的交换机无法隔离广播域,大型网络里如果所有设备共享一个广播域,一次广播风暴或者一次错误的 DHCP 广播请求可能影响全网所有设备,而且不同部门/不同安全等级的设备被迫暴露在同一个二层网络里,存在安全和故障隔离的双重问题。VLAN 的本质是用软件配置的方式,在物理拓扑不变的前提下,重新定义"哪些端口属于同一个逻辑网络",相当于用一台物理交换机模拟出多台相互独立的逻辑交换机。

**AI研究/工程场景:** 多租户的云 GPU 训练平台通过 VLAN(或者更现代的 overlay 网络技术,比如 VXLAN)把不同租户的网络流量隔离开——即使物理上多个租户的训练任务跑在共享的同一批物理服务器和交换机上,不同租户之间不应该能在链路层互相"看见"对方的广播流量,这是多租户隔离的基础安全要求之一。

**可运行例子(环境:`.venv`,基于知识点5的SimpleSwitch扩展):**
```python
class VlanSwitch:
    def __init__(self, port_vlan_map):
        self.port_vlan = port_vlan_map
        self.mac_table = {}

    def receive_frame(self, src_mac, dst_mac, arrival_port):
        self.mac_table[src_mac] = arrival_port
        my_vlan = self.port_vlan[arrival_port]
        same_vlan_ports = [p for p, v in self.port_vlan.items() if v == my_vlan and p != arrival_port]

        if dst_mac in self.mac_table and self.port_vlan[self.mac_table[dst_mac]] == my_vlan:
            return [self.mac_table[dst_mac]]
        return same_vlan_ports  # 广播/未知目的只在同VLAN内洪泛,不会跨VLAN扩散

vsw = VlanSwitch(port_vlan_map={1: "VLAN10", 2: "VLAN10", 3: "VLAN20", 4: "VLAN20"})

out_broadcast_vlan10 = vsw.receive_frame("MAC_X", "FF:FF:FF:FF:FF:FF", arrival_port=1)
assert out_broadcast_vlan10 == [2], \
    f"VLAN10 broadcast must only reach port 2 (same VLAN), not port 3/4 (VLAN20), got {out_broadcast_vlan10}"

out_broadcast_vlan20 = vsw.receive_frame("MAC_Y", "FF:FF:FF:FF:FF:FF", arrival_port=3)
assert out_broadcast_vlan20 == [4], \
    f"VLAN20 broadcast must only reach port 4, got {out_broadcast_vlan20}"

print("OK: VLAN10 broadcast never reaches VLAN20 ports, broadcast domain isolated by VLAN tag")
```

**面试怎么问+追问链:**
- Q: "VLAN 解决了什么问题?" → 追问1: "两个不同 VLAN 里的设备要通信,数据包要怎么走?"(**工程约束递增轴**:考察是否理解跨 VLAN 通信必须经过路由/三层交换,本质上是把"逻辑上不同的网络"当成了不同网段来处理,呼应知识点1"网络层地址是端到端有效"的结论) → 深挖追问: "如果你是新公司的网络架构师,给 100 台服务器规划 VLAN,你会按什么维度划分?"(开放性场景题,期待候选人提出"按业务/环境(生产vs测试)/安全等级"等实际考量维度,不是唯一标准答案,重点看是否有工程判断力而非死记概念)

**常见坑:** 误以为 VLAN 隔离等同于"物理隔离"级别的安全性——VLAN 本质上是通过交换机软件配置实现的逻辑隔离,如果交换机配置存在漏洞(比如 VLAN hopping 攻击利用 802.1Q 双重标签欺骗),VLAN 隔离是可能被绕过的,对安全性要求极高的场景(比如金融核心系统)通常还需要额外的物理隔离或更严格的边界防护,不能完全依赖 VLAN 作为唯一的安全边界。

---

*本篇完成:2026-07-14,6 个知识点。验证环境:`.venv`(全部6点)。板块 I(分层模型与链路层基础)完成。*
