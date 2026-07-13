# 计算机网络深挖 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task(内联执行模式,不用subagent-driven-development——用户已明确"直接执行",延续本仓库标准授权模式,任务间不暂停等待确认)。Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `for_real_dummy/computer-networking-deep-dive/` 下产出一套约 80 个知识点、5 大板块 12 个分类文件 + 1 篇模拟终面 capstone 的计算机网络深挖系列,深度对标本仓库已完成的 os-concurrency-deep-dive(79点)/dsa-deep-dive(140点)/statistics-deep-dive(116点),达到技术终面级别深度广度,全栈覆盖(链路层→网络层→传输层→应用层)且应用层加重。

**Architecture:** 每个分类文件独立成篇,遵循统一的七步知识点模板(签名/是什么→一句话→底层机制/为什么这样设计→AI研究/工程场景→可运行例子→面试怎么问+追问链→常见坑),五轴追问链方法论从第一天融入"面试怎么问"步骤。验证采用双环境策略:socket编程/协议报文构造/算法模拟用仓库 `.venv`(Windows 原生 Python),真实 Linux 网络工具强相关的知识点(tcpdump/tc/network namespace+veth/raw socket/真实TCP状态观察)用已确认可复用的 WSL2 Rocky Linux。新增"不引入外部网络依赖"纪律:所有可运行例子只连本地回环或 WSL2 内部构造的虚拟网络拓扑,不对真实互联网发起请求。

**Tech Stack:** Python 3(仓库根目录 `.venv`,标准库为主:`socket`/`ssl`/`ipaddress`/`select`/`selectors`/`http.server`/`hashlib`/`base64`/`struct`,不新增第三方依赖,即不引入 `grpcio`/`protobuf`/`requests`/`scapy` 等)+ WSL2 Rocky Linux 10.2(`/usr/bin/python3`,3.12.13,通过 `wsl.exe -d RockyLinux` 调用,复用 rhcsa-bash-deep-dive/os-concurrency-deep-dive 已装好的环境)。

## Global Constraints

- 环境:默认仓库根目录 `.venv`(Windows 原生);真实 Linux 网络语义(`tcpdump`/`tc`/network namespace+veth/raw socket/`ss` 真实TCP状态/`openssl s_client`)用 WSL2 Rocky Linux(已确认可用,复用 rhcsa-bash-deep-dive/os-concurrency-deep-dive 环境,不新建)。Git Bash 调 `wsl.exe` 前必须 `export MSYS_NO_PATHCONV=1`。
- 每个知识点的"可运行例子"步骤开头必须显式标注验证环境(`.venv` 或 `WSL2 Rocky Linux`)。
- 不新增任何第三方 Python 依赖,标准库优先——`socket`/`ssl`/`ipaddress`/`select`/`selectors`/`http.server`/`hashlib`/`base64`/`struct` 等已覆盖本系列绝大多数验证需求;涉及 gRPC/Protobuf 等标准库没有的协议,以概念性讲解为主并如实标注"不新增第三方依赖,不提供可运行demo"。
- 知识点模板固定七步(签名/是什么→一句话→底层机制/为什么这样设计→AI研究/工程场景→可运行例子→面试怎么问+追问链→常见坑),不单独拆"数学推导"或"协议细节"步骤——协议细节直接写进"底层机制"步骤。
- 五轴追问链方法论(规模递增/工程约束递增/方案批判迭代/决策依据追问/真实性验证 + "诊断真实数据"新题型)从第一天融入"面试怎么问+追问链"步骤,每点挑 1~2 条最自然的轴线,不强行凑满 5 轴。
- 数学/机制结论必须用 `assert` 数值验证,不能只摆公式描述。
- print() 语句必须纯 ASCII(历史教训:Windows GBK locale 下 `_verify_md.py` 子进程 reader 线程遇到非 ASCII 字符会 UnicodeDecodeError)。正文 markdown 里的数学/协议符号不受此限制,只有 CODE 块内的 print() 输出受限。
- **不引入外部网络依赖**:所有可运行例子只连接本地回环地址(`127.0.0.1`)或 WSL2 内部构造的虚拟网络拓扑(network namespace + veth pair),不对真实互联网发起请求——DNS 类知识点用手工构造报文或 `/etc/hosts`/Windows hosts 本地条目验证,不查询真实公网 DNS 服务器。
- 依赖真实网络往返/抓包统计的断言,要求至少重复运行 5~10 次确认现象稳定复现,不满足于运气好复现一次。
- 需要真实构造超时/重传/连接失败的 demo 必须带确定性退出的超时安全网(延续 os-concurrency-deep-dive 死锁 demo 纪律)。
- 计时对比断言用 `best_of(fn, *args, trials=N)` 取多次采样最小值,规模不能小到被系统调度噪声/网络抖动淹没。
- 每个板块(不是每个文件)验证通过后可以独立 git commit,但每个文件至少一次 commit;`git add` 必须显式列出文件路径,不用 `git add -A`/`.`。
- 涉及跨文件引用其他分类知识点,用既定的"NN类知识点M"纯文本格式,不用 markdown 链接锚点。
- 涉及 learning/cluster-networking 关联前(12类),必须先实际读取该模块 README/lectures 确认真实技术关联,不能凭听起来相关就编造联系(同 G8 纪律)。
- 涉及一致性哈希(10类)、限流算法(12类)时,交叉引用 dsa-deep-dive 20-advanced-interview-depth.md 已有的真实代码验证,不重新写一遍。
- 11类"网络编程与IO模型"落到网络服务器实操层面,复用 os-concurrency-deep-dive 08类"IO模型演进"关于 select/poll/epoll 机制本身的结论(不重新讲机制内部原理),聚焦"在真实网络场景怎么用"。

---

### Task 1: 脚手架

**Files:**
- Create: `for_real_dummy/computer-networking-deep-dive/00-roadmap.md`
- Create: `for_real_dummy/computer-networking-deep-dive/_verify_md.py`(从 `for_real_dummy/os-concurrency-deep-dive/_verify_md.py` 原样拷贝,含 `python-wsl2` 围栏标记支持)

**Interfaces:**
- Produces:`00-roadmap.md` 含 5 板块 12 文件 + capstone 的进度表(初始全部标"⏳ 未开始"),供 Task 2-14 逐行更新为 "✅ 已完成"。`_verify_md.py` 供 Task 2-14 每篇验证时调用:`python _verify_md.py <path/to/file.md>`。

- [ ] **Step 1: 拷贝验证脚本**

```bash
cp for_real_dummy/os-concurrency-deep-dive/_verify_md.py for_real_dummy/computer-networking-deep-dive/_verify_md.py
```

- [ ] **Step 2: 确认 WSL2 Rocky Linux 环境可用**

```bash
export MSYS_NO_PATHCONV=1 && wsl.exe -d RockyLinux -- python3 --version
```
Expected: `Python 3.12.13`。

- [ ] **Step 3: 确认 WSL2 内网络诊断工具可用(tcpdump/tc/ip netns)**

```bash
export MSYS_NO_PATHCONV=1 && wsl.exe -d RockyLinux -- bash -c "command -v tcpdump; command -v tc; command -v ip; openssl version"
```
Expected: 四个命令均输出可执行路径/版本号。若 `tcpdump` 缺失,记录 `dnf install tcpdump` 安装,如实标注在 roadmap 环境声明里(参照 rhcsa-bash-deep-dive"如实标注工具默认是否预装"纪律)。

- [ ] **Step 4: 撰写 `00-roadmap.md`**

内容包含:目标声明(约80个知识点,5板块12文件+capstone,对标os-concurrency-deep-dive/dsa-deep-dive/statistics-deep-dive深度,全栈覆盖应用层加重)、与 rhcsa-bash-deep-dive(07/08类操作层)+ learning/cluster-networking(AI infra集群网络专精层)+ dsa-deep-dive 20类(一致性哈希/限流交叉引用)+ os-concurrency-deep-dive 08类(IO模型机制角度)的边界声明、七步知识点模板说明、五轴追问链方法论表格(照抄 spec §4)、双环境验证声明、"不引入外部网络依赖"纪律说明、进度表(下表,初始状态全部"⏳ 未开始"):

| # | 板块 | 分类 | 文件 | 知识点数(约) | 状态 |
|---|------|------|------|------------|------|
| 01 | I 分层模型与链路层基础 | 分层模型与链路层 | 01-layering-and-link-basics.md | 6 | ⏳ |
| 02 | II 网络层:IP与路由 | IP编址与子网划分 | 02-ip-addressing-and-subnetting.md | 7 | ⏳ |
| 03 | II | 路由与网络层机制 | 03-routing-and-network-layer-mechanisms.md | 7 | ⏳ |
| 04 | III 传输层:TCP/UDP | TCP连接管理 | 04-tcp-connection-management.md | 7 | ⏳ |
| 05 | III | TCP可靠传输与流量控制 | 05-tcp-reliability-and-flow-control.md | 7 | ⏳ |
| 06 | III | TCP拥塞控制与UDP | 06-tcp-congestion-control-and-udp.md | 7 | ⏳ |
| 07 | IV 应用层协议 | HTTP演进 | 07-http-evolution.md | 7 | ⏳ |
| 08 | IV | HTTPS与TLS | 08-https-and-tls.md | 6 | ⏳ |
| 09 | IV | DNS域名解析 | 09-dns-resolution.md | 6 | ⏳ |
| 10 | IV | 现代应用协议与API | 10-modern-app-protocols-and-apis.md | 7 | ⏳ |
| 11 | V 现代网络与工程场景 | 网络编程与IO模型 | 11-network-programming-and-io-models.md | 7 | ⏳ |
| 12 | V | 现代网络专题 | 12-modern-networking-topics.md | 6 | ⏳ |
| 13 | 收尾 | 模拟终面capstone | 13-mock-interview-capstone.md | —(不计入合计) | ⏳ |

- [ ] **Step 5: Commit**

```bash
git add for_real_dummy/computer-networking-deep-dive/00-roadmap.md for_real_dummy/computer-networking-deep-dive/_verify_md.py
git commit -m "docs(computer-networking): 脚手架 - roadmap + 验证脚本"
```

---

### Task 2: 01-layering-and-link-basics.md(分层模型与链路层基础)

**Files:**
- Create: `for_real_dummy/computer-networking-deep-dive/01-layering-and-link-basics.md`
- Modify: `for_real_dummy/computer-networking-deep-dive/00-roadmap.md`(第01行状态改✅)

知识点范围(6个):OSI七层与TCP/IP四层/五层模型对比(分层设计哲学)、封装与解封装(PDU命名:段/包/帧/比特)、以太网帧结构与MAC地址、ARP协议、交换机工作原理(MAC地址表学习/转发,冲突域vs广播域)、VLAN基础概念。

- [ ] **Step 1: 设计知识点并验证核心结论**

用 `.venv` 手工用 `struct` 构造以太网帧首部字节布局(目的MAC 6字节+源MAC 6字节+EtherType 2字节),验证字段偏移和长度符合协议定义。ARP 请求/响应报文同样用 `struct` 手工构造并解析,验证往返后能从"IP找到MAC"。交换机 MAC 学习表用 Python dict 模拟(收到帧记录 源MAC→端口 映射,转发时查表决定单播转发还是洪泛),构造一个小拓扑验证学习后不再洪泛已知目的地。VLAN 因缺乏真实交换机硬件,标注为概念性讲解为主(为什么要隔离广播域,802.1Q tag 格式简要介绍),不强求可运行demo。

- [ ] **Step 2: 撰写完整 markdown**

按七步模板撰写全部 6 个知识点,"面试怎么问+追问链"融入五轴方法论。

- [ ] **Step 3: 运行验证脚本**

```bash
cd for_real_dummy/computer-networking-deep-dive && python _verify_md.py 01-layering-and-link-basics.md
```
Expected: 全部代码块 PASS。

- [ ] **Step 4: 检查 print() 语句纯 ASCII**

```bash
grep -P '[^\x00-\x7F]' for_real_dummy/computer-networking-deep-dive/01-layering-and-link-basics.md | grep 'print('
```
Expected: 无匹配。

- [ ] **Step 5: 更新 roadmap 第01行状态为 ✅ 已完成**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/computer-networking-deep-dive/01-layering-and-link-basics.md for_real_dummy/computer-networking-deep-dive/00-roadmap.md
git commit -m "docs(computer-networking): 01类 分层模型与链路层基础(6点) - 板块I完成"
```

---

### Task 3: 02-ip-addressing-and-subnetting.md(IP编址与子网划分)

**Files:**
- Create: `for_real_dummy/computer-networking-deep-dive/02-ip-addressing-and-subnetting.md`
- Modify: `00-roadmap.md`(第02行)

知识点范围(7个):IPv4地址结构与分类(A/B/C类历史与CIDR取代)、子网掩码与子网划分计算、CIDR记法与路由聚合(超网化)、公网IP与私网IP(RFC1918)、IPv4地址枯竭与IPv6设计(128位地址、简化包头)、IPv6过渡技术概念性(双栈/隧道/NAT64)、特殊地址(环回/广播/组播范围)。

- [ ] **Step 1: 设计知识点并验证核心结论**

全部用 `.venv` 标准库 `ipaddress` 模块验证(`ipaddress.ip_network`/`ip_address`/`.subnets()`/`.supernet()`):子网划分计算(一个 /24 划分成 4 个 /26,验证每个子网的网络地址/广播地址/可用主机数正确)、CIDR 路由聚合(4 个连续 /24 聚合成 1 个 /22,验证聚合后网段确实覆盖全部原网段)、RFC1918 私网范围判定(`ip.is_private`)、IPv6 地址压缩展开(`::1` 展开为完整形式再压缩回去验证往返一致)、特殊地址判定(`is_loopback`/`is_multicast`)。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本并确认 PASS**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第02行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/computer-networking-deep-dive/02-ip-addressing-and-subnetting.md for_real_dummy/computer-networking-deep-dive/00-roadmap.md
git commit -m "docs(computer-networking): 02类 IP编址与子网划分(7点)"
```

---

### Task 4: 03-routing-and-network-layer-mechanisms.md(路由与网络层机制)

**Files:**
- Create: `for_real_dummy/computer-networking-deep-dive/03-routing-and-network-layer-mechanisms.md`
- Modify: `00-roadmap.md`(第03行)

知识点范围(7个):路由表结构与最长前缀匹配、静态路由vs动态路由、距离矢量vs链路状态路由协议概念性对比(RIP vs OSPF)、域内vs域间路由(IGP vs BGP概念性)、NAT机制与工作原理(SNAT/DNAT/端口转换表)、ICMP协议与traceroute工作原理(TTL递减)、分片与MTU(Path MTU Discovery)。

- [ ] **Step 1: 设计知识点并验证核心结论**

最长前缀匹配:`.venv` 手写路由表查找算法(路由表为 `[(网络, 掩码, 下一跳)]` 列表),构造多条重叠路由(如同时有 `0.0.0.0/0` 默认路由和 `10.0.0.0/8` 更具体路由),验证查找算法选中最长前缀匹配的条目而非默认路由。距离矢量算法:用 Bellman-Ford 风格实现,构造一个小型网络图(5-6个节点),验证多轮迭代后收敛到全局最短路径,和直接跑 Dijkstra 的结果比对一致。NAT:`.venv` 用 dict 模拟 NAT 转换表(`(内网IP,内网端口) <-> 外网端口`),验证双向转换正确且端口复用时不冲突;如果 WSL2 环境允许,额外用 network namespace + veth pair + `iptables` 做一次真实 SNAT 验证作为加强(非必需,`.venv` 模拟已能验证机制本身)。ICMP/traceroute:WSL2 Rocky Linux 用 raw socket 或调用真实 `traceroute`/`ping -t` 观察 TTL 从 1 递增时每一跳返回 "TTL exceeded",验证 TTL 递减机制;如果 WSL2 环境权限/工具受限,退回 `.venv` 纯拓扑模拟(多个"路由器"对象,每转发一次 TTL-1,到 0 时返回超时事件)并如实标注选择的验证方式。

- [ ] **Step 2: 撰写完整 markdown**

RIP vs OSPF、IGP vs BGP 部分明确标注"概念性对比,不提供可运行demo"(协议实现复杂度超出教学示例范围,聚焦为什么设计、优劣权衡、面试怎么问)。

- [ ] **Step 3: 运行验证脚本并确认 PASS(WSL2 部分单独验证并记录)**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第03行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/computer-networking-deep-dive/03-routing-and-network-layer-mechanisms.md for_real_dummy/computer-networking-deep-dive/00-roadmap.md
git commit -m "docs(computer-networking): 03类 路由与网络层机制(7点) - 板块II完成"
```

---

### Task 5: 04-tcp-connection-management.md(TCP连接管理)

**Files:**
- Create: `for_real_dummy/computer-networking-deep-dive/04-tcp-connection-management.md`
- Modify: `00-roadmap.md`(第04行)

知识点范围(7个):TCP报文段结构、三次握手(为什么不是两次/四次,SYN flood原理)、四次挥手与TIME_WAIT状态(为什么需要2MSL)、TCP状态机完整图(11个状态)、端口与五元组、半连接队列与全连接队列backlog、socket编程基础(bind/listen/accept/connect阻塞语义)。

- [ ] **Step 1: 设计知识点并验证核心结论**

`.venv` 用真实 `socket.socket(AF_INET, SOCK_STREAM)` 在本地回环建立 client-server 连接,握手/挥手是内核自动完成的,用 WSL2 Rocky Linux 的 `ss -tan` 在连接建立/关闭的不同阶段抓取真实状态(`ESTABLISHED`/`TIME_WAIT`/`CLOSE_WAIT` 等),验证真实状态机转换和理论一致。TIME_WAIT 真实触发:client 主动 `close()` 后立即 `ss -tan` 观察到 TIME_WAIT 残留连接,验证其确实存在且会在 2MSL 后消失(不必真实等待完整 2MSL,只需验证其存在)。backlog:`listen(backlog=1)`,用多个并发线程同时发起 `connect()`,验证超出 backlog 的连接排队/延迟接受行为。SYN flood 原理:概念性讲解(伪造大量 SYN 占满半连接队列),不真实发起攻击性质流量,如实标注这是防御性安全知识不提供攻击代码。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本(`.venv` 部分用 `_verify_md.py`,WSL2 `ss` 观察部分单独验证并在文中记录)**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第04行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/computer-networking-deep-dive/04-tcp-connection-management.md for_real_dummy/computer-networking-deep-dive/00-roadmap.md
git commit -m "docs(computer-networking): 04类 TCP连接管理(7点)"
```

---

### Task 6: 05-tcp-reliability-and-flow-control.md(TCP可靠传输与流量控制)

**Files:**
- Create: `for_real_dummy/computer-networking-deep-dive/05-tcp-reliability-and-flow-control.md`
- Modify: `00-roadmap.md`(第05行)

知识点范围(7个):序列号与确认应答机制、超时重传与RTT估算(RTO计算)、快速重传(3个重复ACK)、滑动窗口协议与流量控制、Nagle算法与延迟确认、零窗口与窗口探测、TCP字节流特性(粘包/拆包根源)。

- [ ] **Step 1: 设计知识点并验证核心结论**

**方法论声明(写入文中如实标注)**:序列号/重传/滑动窗口这些机制发生在内核 TCP 协议栈内部,用户态 Python 无法直接控制或观测其内部状态转换,因此这几个知识点采用**教学性模拟**——在 UDP(用户态完全可控、不可靠)之上手写一个简化可靠传输协议,显式实现序列号、ACK、超时重传、滑动窗口,用来验证机制本身而不是观测真实内核行为。`.venv` 实现:构造一个会丢包的模拟信道(按概率丢弃 UDP 包),验证加了序列号+重传的自制协议最终仍能让接收方收到完整有序数据,不加重传机制的版本则会丢数据——用对比实验凸显机制的必要性。粘包/拆包**可以真实复现**(不需要模拟):TCP socket 连续两次小 `send()`,接收端一次 `recv(4096)` 收到两条消息粘在一起,证明字节流没有消息边界,应用层需要自定义分隔符或长度前缀。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本并确认 PASS**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第05行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/computer-networking-deep-dive/05-tcp-reliability-and-flow-control.md for_real_dummy/computer-networking-deep-dive/00-roadmap.md
git commit -m "docs(computer-networking): 05类 TCP可靠传输与流量控制(7点)"
```

---

### Task 7: 06-tcp-congestion-control-and-udp.md(TCP拥塞控制与UDP)

**Files:**
- Create: `for_real_dummy/computer-networking-deep-dive/06-tcp-congestion-control-and-udp.md`
- Modify: `00-roadmap.md`(第06行)

知识点范围(7个):拥塞控制与流量控制的区别、慢启动与拥塞避免、拥塞检测与快速恢复、Reno/Cubic/BBR算法对比、UDP协议特性与首部结构、TCP vs UDP选型决策(含QUIC为什么选UDP)、端口多路复用/多路分解机制。

- [ ] **Step 1: 设计知识点并验证核心结论**

拥塞窗口演变是纯数学状态机,`.venv` 数值模拟:实现 `cwnd` 随轮次变化(慢启动阶段指数翻倍直到 `ssthresh`,之后拥塞避免线性增长,遇到丢包乘性减半),验证曲线形状符合理论(指数段用 `assert cwnd[i+1] == cwnd[i]*2`,线性段用 `assert cwnd[i+1] == cwnd[i]+1`,丢包后 `assert cwnd_after == cwnd_before // 2`)。Reno/Cubic/BBR 用简化模型对比核心差异(Reno 丢包才降窗 vs BBR 基于带宽时延积主动探测,不依赖丢包信号),概念性对比为主,不实现完整算法。UDP:`.venv` 真实 `socket.SOCK_DGRAM`,验证无连接语义(`sendto()` 到一个没有进程监听的端口不报错,不像 TCP `connect()` 会被拒绝)。TCP vs UDP 选型:WSL2 用 `tc qdisc` 注入一定丢包率,对比同样数据量下 TCP(自动重传,最终完整到达但更慢)vs UDP(不重传,更快但确认丢了部分数据)的真实行为差异。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本(`.venv` 部分 + WSL2 `tc` 部分单独验证并记录)**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第06行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/computer-networking-deep-dive/06-tcp-congestion-control-and-udp.md for_real_dummy/computer-networking-deep-dive/00-roadmap.md
git commit -m "docs(computer-networking): 06类 TCP拥塞控制与UDP(7点) - 板块III完成"
```

---

### Task 8: 07-http-evolution.md(HTTP演进)

**Files:**
- Create: `for_real_dummy/computer-networking-deep-dive/07-http-evolution.md`
- Modify: `00-roadmap.md`(第07行)

知识点范围(7个):HTTP/0.9到1.0、HTTP/1.1持久连接与管道化局限(队头阻塞)、HTTP方法/状态码语义、HTTP缓存机制(Cache-Control/ETag/Last-Modified)、HTTP/2二进制分帧多路复用与HPACK首部压缩、HTTP/3与QUIC(为什么放弃TCP)、Keep-Alive连接复用工程含义。

- [ ] **Step 1: 设计知识点并验证核心结论**

`.venv` 用标准库 `http.server` 起一个真实本地 HTTP 服务器,手工用 `socket` 构造 HTTP/1.0 风格请求(每次新建 TCP 连接发一个请求)vs HTTP/1.1 风格请求(`Connection: keep-alive`,同一个连接连续发多个请求),实测并对比建立的 TCP 连接数量差异,验证持久连接确实减少了握手开销。缓存机制:server 返回 `ETag`,client 带 `If-None-Match` 重新请求,验证收到 `304 Not Modified` 而非完整 body。HTTP/2 二进制分帧:标准库无 HTTP/2 实现,用 `struct` 手工构造一个 HTTP/2 帧头(长度+类型+标志+流ID,9字节格式),验证字段解析正确,配合概念性讲解多路复用/HPACK 原理(不实现完整协议栈,如实标注)。HTTP/3/QUIC 为纯概念性讲解(标准库和系统均无 QUIC 支持,不提供可运行demo)。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本并确认 PASS**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第07行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/computer-networking-deep-dive/07-http-evolution.md for_real_dummy/computer-networking-deep-dive/00-roadmap.md
git commit -m "docs(computer-networking): 07类 HTTP演进(7点)"
```

---

### Task 9: 08-https-and-tls.md(HTTPS与TLS)

**Files:**
- Create: `for_real_dummy/computer-networking-deep-dive/08-https-and-tls.md`
- Modify: `00-roadmap.md`(第08行)

知识点范围(6个):对称加密vs非对称加密在TLS里的分工、TLS握手完整流程(1.2 vs 1.3对比)、数字证书与CA信任链、中间人攻击与证书验证机制、HTTPS性能优化(会话复用/OCSP装订)、前向保密概念。

- [ ] **Step 1: 设计知识点并验证核心结论**

`.venv` 用标准库 `ssl` + 手工生成的自签名证书(用 `ssl` 模块或提前用 WSL2 `openssl req` 生成测试证书文件,仓库内不提交私钥仅用于验证阶段临时文件)起一个 HTTPS server,真实 client 用 `ssl.create_default_context()` 连接:①指定信任该自签名证书时握手成功;②不信任时真实触发 `ssl.SSLCertVerificationError`,验证证书验证机制确实会阻止未受信任的连接(教学化复现中间人攻击场景为什么会被 TLS 挡住)。TLS1.2 vs 1.3:连接后读取 `ssl_socket.version()` 确认协商版本,对比两者握手往返次数的理论差异(1.3 是 1-RTT,1.2 全握手是 2-RTT),如实标注:本地回环环境往返耗时差异可能不明显,重点验证协商结果正确而非计时。WSL2 用 `openssl s_client -connect 127.0.0.1:PORT` 连接本地起的 server,观察真实握手详情输出。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本并确认 PASS**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第08行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/computer-networking-deep-dive/08-https-and-tls.md for_real_dummy/computer-networking-deep-dive/00-roadmap.md
git commit -m "docs(computer-networking): 08类 HTTPS与TLS(6点)"
```

---

### Task 10: 09-dns-resolution.md(DNS域名解析)

**Files:**
- Create: `for_real_dummy/computer-networking-deep-dive/09-dns-resolution.md`
- Modify: `00-roadmap.md`(第09行)

知识点范围(6个):DNS域名层级结构、递归查询vs迭代查询完整链路、DNS缓存与TTL机制(各层缓存)、DNS记录类型(A/AAAA/CNAME/MX/TXT)、DNS负载均衡与GeoDNS概念性、DNS劫持与DNSSEC概念性。

- [ ] **Step 1: 设计知识点并验证核心结论**

**不引入外部网络依赖**:不发起真实公网 DNS 查询。用 `.venv` 手工用 `struct` 构造 DNS 查询报文字节格式(header 12字节+question section),再手工构造一个对应的 DNS 响应报文字节并解析,验证报文格式(ID/flags/QDCOUNT/ANCOUNT 等字段)理解正确。递归查询 vs 迭代查询链路:自己写几个"根域名服务器"/"顶级域服务器"/"权威服务器"角色的 Python 对象(每个只知道下一跳该问谁),模拟一次完整递归查询链路(本地DNS代替client逐层问)和一次迭代查询链路(client自己逐层问),验证两种模式下"谁负责递归"的职责分配不同这一核心差异。DNS 缓存:用一个简单的 dict + 过期时间戳模拟 TTL 缓存,验证 TTL 内直接命中缓存、TTL 过期后重新查询。真实解析本地条目:`socket.getaddrinfo` 查询 Windows/WSL2 本地 hosts 文件里手工添加的一条测试记录,验证走的是本地解析而非发起网络请求。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本并确认 PASS**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第09行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/computer-networking-deep-dive/09-dns-resolution.md for_real_dummy/computer-networking-deep-dive/00-roadmap.md
git commit -m "docs(computer-networking): 09类 DNS域名解析(6点)"
```

---

### Task 11: 10-modern-app-protocols-and-apis.md(现代应用协议与API)

**Files:**
- Create: `for_real_dummy/computer-networking-deep-dive/10-modern-app-protocols-and-apis.md`
- Modify: `00-roadmap.md`(第10行)

知识点范围(7个):WebSocket协议(握手升级机制,vs HTTP长轮询)、gRPC(基于HTTP/2+Protocol Buffers,为什么AI模型serving/微服务常用)、RESTful API设计原则、GraphQL概念性对比、长连接保活心跳设计、CDN工作原理(边缘节点/回源/缓存策略)、负载均衡算法(轮询/加权/最少连接/一致性哈希——一致性哈希部分交叉引用dsa-deep-dive 20类,不重复推导)。

- [ ] **Step 1: 设计知识点并验证核心结论**

WebSocket 握手**真实实现**:`.venv` 手工构造 HTTP Upgrade 请求(`Upgrade: websocket` + `Sec-WebSocket-Key`),server 端按 RFC6455 算法(`Sec-WebSocket-Key` + magic GUID 做 SHA1 再 Base64,标准库 `hashlib`+`base64` 可完成)计算 `Sec-WebSocket-Accept` 返回,client 验证返回值和自己独立计算的期望值一致,证明握手升级机制真实可行。gRPC:如实标注"标准库无 `grpcio`/`protobuf`,不新增第三方依赖,本节以概念讲解为主(基于HTTP/2的语义:强类型接口定义、二进制序列化、流式RPC),不提供可运行gRPC demo"。负载均衡算法:轮询/加权轮询/最少连接用 `.venv` 纯 Python 实现并验证分配结果符合算法预期(比如加权轮询验证权重比例);一致性哈希**不重新实现**,在文中写"一致性哈希的负载均衡实测详见 dsa-deep-dive/20-advanced-interview-depth.md 案例1,此处引用其结论:引入虚拟节点后节点负载标准差显著降低"。CDN:概念性讲解边缘节点/回源机制为主,可以用一个简化的"本地缓存代理"模拟(client先查本地缓存字典,未命中才转发到"源站",命中则直接返回并统计命中率)体现"就近获取、减少回源"这一核心思想。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本并确认 PASS**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第10行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/computer-networking-deep-dive/10-modern-app-protocols-and-apis.md for_real_dummy/computer-networking-deep-dive/00-roadmap.md
git commit -m "docs(computer-networking): 10类 现代应用协议与API(7点) - 板块IV完成"
```

---

### Task 12: 11-network-programming-and-io-models.md(网络编程与IO模型)

**Files:**
- Create: `for_real_dummy/computer-networking-deep-dive/11-network-programming-and-io-models.md`
- Modify: `00-roadmap.md`(第11行)

知识点范围(7个):socket阻塞/非阻塞模式的网络编程实操(呼应os-concurrency-deep-dive 08类机制角度,这里落到真实网络场景)、IO多路复用在真实网络服务器的应用(单线程Redis为什么能扛高并发连接)、长连接vs短连接资源权衡、连接池设计思想、零拷贝在网络传输的应用(sendfile)、网络编程背压机制、真实抓包分析实践(tcpdump基础)。

- [ ] **Step 1: 设计知识点并验证核心结论**

阻塞/非阻塞:`.venv` 真实验证 `socket.setblocking(False)` 后,无数据到达时 `recv()` 立即抛 `BlockingIOError` 而不是像阻塞模式那样挂起等待。IO多路复用真实网络服务器:用标准库 `selectors` 写一个真实的单线程 echo server,同时用多个 client 线程并发连接发消息,验证单线程 server 确实能正确处理多个并发连接而不互相阻塞(这是复用 os-concurrency-deep-dive 08类 select/poll/epoll 结论的"实际应用"落地,不重新讲机制内部原理)。连接池:模拟一个简化连接池类(固定大小,`acquire()`/`release()`),对比"每次新建连接"vs"复用连接池"在多次操作下的总连接建立次数差异。零拷贝:`socket.socket.sendfile()` 标准库支持,对比 `sendfile()` vs 手动 `read()+send()` 循环发送同一个文件,如实标注:本地小文件场景耗时差异可能不明显,重点讲清楚零拷贝节省的是"用户态内核态数据拷贝次数"这个机制,不是单纯比谁更快。tcpdump 实践:WSL2 Rocky Linux 起一个本地 HTTP 请求,同时 `tcpdump` 抓包(限定本地回环接口+端口,避免抓到无关流量),解析输出验证能看到真实的 SYN/SYN-ACK/ACK 包序列。

- [ ] **Step 2: 撰写完整 markdown**

- [ ] **Step 3: 运行验证脚本(`.venv` 部分 + WSL2 tcpdump 部分单独验证并记录)**

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第11行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/computer-networking-deep-dive/11-network-programming-and-io-models.md for_real_dummy/computer-networking-deep-dive/00-roadmap.md
git commit -m "docs(computer-networking): 11类 网络编程与IO模型(7点)"
```

---

### Task 13: 12-modern-networking-topics.md(现代网络专题)

**Files:**
- Create: `for_real_dummy/computer-networking-deep-dive/12-modern-networking-topics.md`
- Modify: `00-roadmap.md`(第12行)

知识点范围(6个):微服务架构下的服务发现与网络拓扑、API网关的网络层职责(限流交叉引用dsa-deep-dive 20类案例3,不重复)、Service Mesh与边车代理概念性、分布式训练网络需求简介(轻量连接learning/cluster-networking,不重复其内容)、网络安全基础概念(DDoS原理与防御思路概念性)、云原生网络(容器网络/CNI概念性,呼应os-concurrency-deep-dive 11类namespaces基础)。

- [ ] **Step 1: 核实 learning/cluster-networking 真实关联(前置步骤,写作前必须完成)**

```bash
grep -rn "socket\|TCP\|RDMA需要\|以太网\|通用网络" learning/cluster-networking/README.md learning/cluster-networking/lectures/*.md
```
读取匹配结果,确认"为什么分布式训练网络需要RDMA/NCCL而不是普通socket/TCP"这个连接点的真实技术依据(比如:普通TCP协议栈的用户态-内核态拷贝+中断开销在GPU间超高带宽场景下成为瓶颈,RDMA绕过内核直接网卡到网卡搬数据),只写核实过的真实关联,不夸大成"本系列后续会深入讲解集群网络"(那是cluster-networking的范围)。

- [ ] **Step 2: 设计知识点并验证核心结论**

服务发现:`.venv` 用一个简单的 dict 模拟服务注册表(`register(name, addr)`/`discover(name)`/`heartbeat`超时剔除),验证注册的服务能被正确发现、心跳超时的服务被正确剔除。API网关限流:概念性讲解网关在网络请求链路里的位置和职责(鉴权/路由/限流/日志),限流算法本身写"详见 dsa-deep-dive/20-advanced-interview-depth.md 案例3,此处不重复推导",聚焦"为什么限流放在网关层而不是每个后端服务自己做"这个架构决策。Service Mesh/DDoS/CNI 三点概念性讲解为主(标注:缺乏真实多容器/K8s集群环境,以机制原理讲解为主,不提供可运行demo)。

- [ ] **Step 3: 撰写完整 markdown**

- [ ] **Step 4: 运行验证脚本并确认 PASS**

- [ ] **Step 5: 检查 print() 纯 ASCII**

- [ ] **Step 6: 更新 roadmap 第12行状态为 ✅**

- [ ] **Step 7: Commit**

```bash
git add for_real_dummy/computer-networking-deep-dive/12-modern-networking-topics.md for_real_dummy/computer-networking-deep-dive/00-roadmap.md
git commit -m "docs(computer-networking): 12类 现代网络专题(6点) - 板块V完成,全部12个分类文件完成"
```

---

### Task 14: 13-mock-interview-capstone.md(模拟终面capstone)

**Files:**
- Create: `for_real_dummy/computer-networking-deep-dive/13-mock-interview-capstone.md`
- Modify: `00-roadmap.md`(第13行)

**Interfaces:**
- Consumes: 01-12 类全部知识点编号(用于追问链里的 cross-reference,格式"NN类知识点M")

场景设定:线上 API 服务 P99 延迟陡增,给一段真实模拟的抓包/监控数据(重传率、连接数、RTT分布等,由 Python 脚本真实生成而非手写编造)要求诊断根因。结构参照 os-concurrency-deep-dive 12类capstone(候选人初版汇报→面试官多轮追问→最终诊断真实异常数据),覆盖至少 3 条五轴追问链轴线,cross-reference 至少 5 处不同类别的知识点(比如 04类TCP状态/06类拥塞控制/07类HTTP/11类连接池/09类DNS)。

- [ ] **Step 1: 设计capstone叙事结构与验证场景**

用 `.venv` 真实构造一个"看起来是服务端处理慢、实际是连接池耗尽导致每次请求都要重新三次握手+TLS握手"的场景:模拟一批并发请求,一部分复用连接池(耗时短),一部分因池耗尽新建连接(耗时长,握手开销可测量),生成真实的模拟监控数据(P50/P99延迟分布、新建连接数 vs 复用连接数比例),验证数据里确实能看到"新建连接的请求延迟显著更高"这一可诊断的具体异常特征(不是编的故事)。

- [ ] **Step 2: 撰写完整 markdown(候选人初版汇报→多轮追问→最终诊断)**

- [ ] **Step 3: 运行验证脚本,并独立重跑至少 3 次确认稳定(随机性固定种子)**

```bash
cd for_real_dummy/computer-networking-deep-dive
python _verify_md.py 13-mock-interview-capstone.md
python _verify_md.py 13-mock-interview-capstone.md
python _verify_md.py 13-mock-interview-capstone.md
```

- [ ] **Step 4: 检查 print() 纯 ASCII**

- [ ] **Step 5: 更新 roadmap 第13行状态为 ✅**

- [ ] **Step 6: Commit**

```bash
git add for_real_dummy/computer-networking-deep-dive/13-mock-interview-capstone.md for_real_dummy/computer-networking-deep-dive/00-roadmap.md
git commit -m "docs(computer-networking): 13类 模拟终面capstone"
```

---

### Task 15: 全库自查回归 + README + memory 更新 + 最终提交

**Files:**
- Modify: `for_real_dummy/computer-networking-deep-dive/00-roadmap.md`(合计行)
- Modify: `for_real_dummy/README.md`
- Create: `C:\Users\ericp\.claude\projects\e--Workspace-dummy\memory\computer-networking-deep-dive-complete.md`
- Modify: `C:\Users\ericp\.claude\projects\e--Workspace-dummy\memory\MEMORY.md`
- Modify: `C:\Users\ericp\.claude\projects\e--Workspace-dummy\memory\feedback-function-by-function-teaching.md`

- [ ] **Step 1: 逐文件独立子进程重跑验证**

```bash
cd for_real_dummy/computer-networking-deep-dive
for f in 01-layering-and-link-basics.md 02-ip-addressing-and-subnetting.md 03-routing-and-network-layer-mechanisms.md 04-tcp-connection-management.md 05-tcp-reliability-and-flow-control.md 06-tcp-congestion-control-and-udp.md 07-http-evolution.md 08-https-and-tls.md 09-dns-resolution.md 10-modern-app-protocols-and-apis.md 11-network-programming-and-io-models.md 12-modern-networking-topics.md 13-mock-interview-capstone.md; do
  echo "=== $f ==="
  python _verify_md.py "$f"
done
```
Expected: 全部 PASS。

- [ ] **Step 2: 结构标记计数核对**

```bash
grep -c "^## [0-9]" for_real_dummy/computer-networking-deep-dive/*.md
```
逐文件核对实际知识点数和 roadmap 表格声明的数字是否一致,不一致则修正 roadmap 表格为精确值(参照此前系列"先估后核"先例,允许最终精确数字和最初"约80"有合理浮动)。

- [ ] **Step 3: 网络类断言额外复测稳定性(重传/RTT/backlog等依赖真实网络往返的断言)**

```bash
for i in $(seq 1 5); do python _verify_md.py 04-tcp-connection-management.md && python _verify_md.py 05-tcp-reliability-and-flow-control.md && python _verify_md.py 06-tcp-congestion-control-and-udp.md; done
```

- [ ] **Step 4: 更新 `00-roadmap.md` 合计行为精确数字**

- [ ] **Step 5: 更新 `for_real_dummy/README.md`**

在"独立技能系列"表格新增一行(参照 os-concurrency-deep-dive 行的格式),补一段说明本系列是"职业发展与需求四部曲"第2部、和 rhcsa-bash-deep-dive/learning-cluster-networking/dsa-deep-dive 20类的边界、双环境验证策略+"不引入外部网络依赖"纪律,目录树新增 `computer-networking-deep-dive/` 条目。

- [ ] **Step 6: 创建 memory 文件 `computer-networking-deep-dive-complete.md`**

frontmatter `type: project`,内容参照 `os-concurrency-deep-dive-complete.md` 的结构:精确知识点数/板块文件数、七步模板(延续不变)、双环境验证策略+"不引入外部网络依赖"新纪律、和 rhcsa-bash-deep-dive/cluster-networking/dsa-deep-dive 的边界、四部曲进度更新(操作系统与并发✅/计算机网络✅完成,数据库/系统设计排队中)。

- [ ] **Step 7: 更新 `MEMORY.md` 索引**

新增一行指向 `computer-networking-deep-dive-complete.md`。

- [ ] **Step 8: 更新 `feedback-function-by-function-teaching.md`**

追加一段简短说明:四部曲第2部完成,后续排队项明确(数据库/系统设计)。

- [ ] **Step 9: 最终提交(确认 `git status`/`git diff` 只涉及本系列相关文件)**

```bash
git status --short for_real_dummy/computer-networking-deep-dive/ for_real_dummy/README.md
git add for_real_dummy/computer-networking-deep-dive/00-roadmap.md for_real_dummy/README.md
git commit -m "docs(computer-networking): 全库自查回归 + README集成 + 收尾提交"
```

---

*创建:2026-07-14*
