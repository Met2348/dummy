# 计算机网络深挖 —— 设计文档

> 定位:`for_real_dummy/` 第 14 条独立技能系列,是"职业发展与需求"四部曲(操作系统与并发→计算机网络→数据库原理与实战→系统设计)的第 2 部。第 1 部 os-concurrency-deep-dive 已完成(79 点,commit 5971baa...70cb26b)。四部曲总体顺序已在系列 1 立项时和用户确认,本次不重新讨论顺序,但范围/板块/模板细节按本仓库纪律重新走一轮独立 brainstorming(2026-07-14,用户批准"全栈覆盖,应用层加重"作为范围决策)。

## 1. 目标与规模

约 **80 个知识点**(±10~15% 浮动,和此前系列同样的先估后核流程,最终以全库自查的精确计数为准),5 大板块,12 个分类文件 + 1 篇模拟终面 capstone。深度对标本仓库已完成系列(dsa-deep-dive 140 点、statistics-deep-dive 116 点、os-concurrency-deep-dive 79 点)的"技术终面 1 小时持续面试级别的深度广度"。

**范围决策(用户 2026-07-14 已确认)**:全栈覆盖(链路层→网络层→传输层→应用层),应用层加重——因为"通用软件工程方向"的日常面试与工作里,HTTP/HTTPS/DNS/WebSocket/gRPC 这类应用层协议出现频率不低于甚至高于 TCP 底层细节,值得比传统教科书更重的篇幅。

## 2. 板块与文件规划

| # | 板块 | 分类 | 文件 | 知识点(约) |
|---|------|------|------|-----------|
| 01 | I 分层模型与链路层基础 | 分层模型与链路层 | 01-layering-and-link-basics.md | 6 |
| 02 | II 网络层:IP与路由 | IP编址与子网划分 | 02-ip-addressing-and-subnetting.md | 7 |
| 03 | II | 路由与网络层机制 | 03-routing-and-network-layer-mechanisms.md | 7 |
| 04 | III 传输层:TCP/UDP | TCP连接管理 | 04-tcp-connection-management.md | 7 |
| 05 | III | TCP可靠传输与流量控制 | 05-tcp-reliability-and-flow-control.md | 7 |
| 06 | III | TCP拥塞控制与UDP | 06-tcp-congestion-control-and-udp.md | 7 |
| 07 | IV 应用层协议 | HTTP演进 | 07-http-evolution.md | 7 |
| 08 | IV | HTTPS与TLS | 08-https-and-tls.md | 6 |
| 09 | IV | DNS域名解析 | 09-dns-resolution.md | 6 |
| 10 | IV | 现代应用协议与API | 10-modern-app-protocols-and-apis.md | 7 |
| 11 | V 现代网络与工程场景 | 网络编程与IO模型 | 11-network-programming-and-io-models.md | 7 |
| 12 | V | 现代网络专题 | 12-modern-networking-topics.md | 6 |
| 13 | 收尾 | 模拟终面capstone | 13-mock-interview-capstone.md | —(不计入合计) |

**每个文件明细(撰写时按此展开,允许在保持知识点独立完整的前提下微调个别点的合并/拆分)**:

- **01 分层模型与链路层基础**:OSI七层与TCP/IP四层/五层模型对比(分层设计哲学)、封装与解封装(PDU命名:段/包/帧/比特)、以太网帧结构与MAC地址、ARP协议、交换机工作原理(MAC地址表学习/转发,冲突域vs广播域)、VLAN基础概念。
- **02 IP编址与子网划分**:IPv4地址结构与分类(A/B/C类历史与CIDR取代)、子网掩码与子网划分计算、CIDR记法与路由聚合、公网IP与私网IP(RFC1918)、IPv4地址枯竭与IPv6设计、IPv6过渡技术概念性(双栈/隧道/NAT64)、特殊地址(环回/广播/组播)。
- **03 路由与网络层机制**:路由表结构与最长前缀匹配、静态路由vs动态路由、距离矢量vs链路状态路由协议概念性对比(RIP vs OSPF)、域内vs域间路由(IGP vs BGP概念性)、NAT机制与工作原理(SNAT/DNAT/端口转换表)、ICMP协议与traceroute工作原理(TTL递减)、分片与MTU(Path MTU Discovery)。
- **04 TCP连接管理**:TCP报文段结构、三次握手(为什么不是两次/四次,SYN flood原理)、四次挥手与TIME_WAIT(为什么需要2MSL)、TCP状态机完整图、端口与五元组、半连接队列与全连接队列backlog、socket编程基础(bind/listen/accept/connect阻塞语义)。
- **05 TCP可靠传输与流量控制**:序列号与确认应答机制、超时重传与RTT估算(RTO计算)、快速重传(3个重复ACK)、滑动窗口协议与流量控制、Nagle算法与延迟确认、零窗口与窗口探测、TCP字节流特性(粘包/拆包根源)。
- **06 TCP拥塞控制与UDP**:拥塞控制与流量控制的区别、慢启动与拥塞避免、拥塞检测与快速恢复、Reno/Cubic/BBR算法对比、UDP协议特性与首部结构、TCP vs UDP选型决策(含QUIC为什么选UDP)、端口多路复用/多路分解机制。
- **07 HTTP演进**:HTTP/0.9到1.0、HTTP/1.1持久连接与管道化局限(队头阻塞)、HTTP方法/状态码语义、HTTP缓存机制(Cache-Control/ETag/Last-Modified)、HTTP/2二进制分帧多路复用与HPACK首部压缩、HTTP/3与QUIC(为什么放弃TCP)、Keep-Alive连接复用工程含义。
- **08 HTTPS与TLS**:对称加密vs非对称加密在TLS里的分工、TLS握手完整流程(1.2 vs 1.3对比)、数字证书与CA信任链、中间人攻击与证书验证机制、HTTPS性能优化(会话复用/OCSP装订)、前向保密概念。
- **09 DNS域名解析**:DNS域名层级结构、递归查询vs迭代查询完整链路、DNS缓存与TTL机制(各层缓存)、DNS记录类型(A/AAAA/CNAME/MX/TXT)、DNS负载均衡与GeoDNS概念性、DNS劫持与DNSSEC概念性。
- **10 现代应用协议与API**:WebSocket协议(握手升级机制,vs HTTP长轮询)、gRPC(基于HTTP/2+Protocol Buffers,为什么AI模型serving/微服务常用)、RESTful API设计原则、GraphQL概念性对比、长连接保活心跳设计、CDN工作原理(边缘节点/回源/缓存策略)、负载均衡算法(轮询/加权/最少连接/一致性哈希——一致性哈希部分交叉引用dsa-deep-dive 20类LRU分布式案例,不重复推导)。
- **11 网络编程与IO模型**:socket阻塞/非阻塞模式的网络编程实操(呼应os-concurrency-deep-dive 08类机制角度,这里落到真实网络场景)、IO多路复用在真实网络服务器的应用(单线程Redis为什么能扛高并发连接)、长连接vs短连接资源权衡、连接池设计思想、零拷贝在网络传输的应用(sendfile)、网络编程背压机制、真实抓包分析实践(tcpdump基础)。
- **12 现代网络专题**:微服务架构下的服务发现与网络拓扑、API网关的网络层职责(限流交叉引用dsa-deep-dive 20类案例3,不重复)、Service Mesh与边车代理概念性、分布式训练网络需求简介(轻量连接learning/cluster-networking,不重复其内容)、网络安全基础概念(DDoS原理与防御思路概念性)、云原生网络(容器网络/CNI概念性,呼应os-concurrency-deep-dive 11类namespaces基础)。
- **13 模拟终面capstone**:综合运用前12类知识点,场景设定为"线上API延迟P99陡增,给一段真实模拟抓包/监控数据要求诊断根因",结构参照 os-concurrency-deep-dive 12类capstone(候选人初版汇报→面试官多轮追问→最终诊断真实异常数据)。

## 3. 知识点结构模板(七步,延续 os-concurrency-deep-dive 模板不变)

1. **签名/是什么** 2. **一句话** 3. **底层机制/为什么这样设计**(网络学科的协议细节——TCP拥塞控制算法族、DNS递归解析层级、TLS握手步骤——本身就属于这一步要讲的核心内容,不单独拆"数学推导"或"协议细节"步骤) 4. **AI研究/工程场景** 5. **可运行例子**(assert验证,注明验证环境) 6. **面试怎么问+追问链**(五轴方法论,见下节) 7. **常见坑**

不新增独立步骤——已核实网络学科虽然协议细节比OS更"系统性"(比如TCP拥塞控制有明确的算法演进谱系),但这些内容天然归属"底层机制/为什么这样设计"这一步的写作范围,额外拆分步骤只会制造和第3步重复的内容,不产生新的信息增量。

## 4. 五轴追问链方法论落地(从第一天融入,不事后补)

| 轴线 | 含义 | 网络学科例子 |
|------|------|------|
| 规模递增轴 | 小规模→大规模→极限行为 | 单连接→高并发短连接(C10K)→海量长连接(WebSocket集群) |
| 工程约束递增轴 | 单机→跨机→全球分发 | 单机HTTP服务→跨机负载均衡→全球CDN分发(为系统设计系列埋伏笔) |
| 方案批判迭代轴 | 面试官连续指出具体缺陷逼换方案 | DNS轮询负载均衡→无法感知后端健康状态→四层LB→无法感知HTTP语义(如按路径路由)→七层LB,每次因具体缺陷被打回 |
| 决策依据追问轴 | 不纠错,只逼问选择依据 | "为什么这里用gRPC不用REST","为什么选UDP不选TCP" |
| 真实性验证轴 | 简历"优化了接口延迟"被追问到具体数字 | "具体减少了几次往返RTT,用了什么手段,改动前后P99数字是多少" |
| 诊断真实数据(新题型) | 给抓包/监控数据,要求诊断而非套公式 | 给一段真实tcpdump重传统计或延迟分布,要求定位是拥塞、丢包还是服务端处理慢 |

每个知识点挑 1~2 条最自然的轴线走 2~3 层深,不强行凑满 5 轴。13 类capstone 是唯一要求同时用 3 条以上轴线的文件。

## 5. 验证环境与纪律

**双环境,逐点标注,延续 os-concurrency-deep-dive 策略,具体分工按网络学科调整**:

- **默认环境**:仓库根目录 `.venv`(Windows 原生 Python)。Python `socket` 模块的基础 TCP/UDP client-server 编程跨平台可行,用于:socket编程基础、HTTP请求/响应手工构造解析、WebSocket握手手工实现、拥塞控制算法数值模拟、路由算法(距离矢量/链路状态)图算法模拟、DNS查询报文手工构造与解析。
- **WSL2 Rocky Linux**(复用 rhcsa-bash-deep-dive/os-concurrency-deep-dive 已装好并验证过的环境,`Python 3.12.13`,不新建):用于真实 Linux 语义强相关的知识点——`tcpdump` 真实抓包、`tc` 流量控制(真实注入延迟/丢包观察TCP行为变化)、network namespace + veth pair(模拟双主机网络,比单纯namespace更进一步)、raw socket(需root权限,Linux专属行为)、`ss -tan` 观察真实TCP状态转换(TIME_WAIT/SYN_SENT等)、`openssl s_client` 真实TLS握手过程检查。
- 每个知识点的"可运行例子"步骤开头必须显式标注验证环境,不笼统一句"已验证"带过。

**并发/计时/网络类特有验证纪律(继承 os-concurrency-deep-dive 已验证有效的纪律,按网络学科调整)**:

- 计时对比断言用 `best_of(fn, *args, trials=N)` 取多次采样最小值,规模不能小到被系统调度噪声/网络抖动淹没。
- **依赖真实网络往返(RTT)、抓包统计的断言,要求多次运行确认稳定复现**——本地回环(127.0.0.1)网络抖动通常小于真实公网,但仍需至少重复5~10次确认现象稳定,不满足于运气好复现一次;涉及真实公网请求的例子一律避免(不引入外部网络依赖/不确定性,所有client-server demo用本地回环或WSL2内部veth pair)。
- **需要真实构造超时/重传/连接失败的 demo 必须带确定性退出的超时安全网**,延续 os-concurrency-deep-dive 死锁 demo 的纪律——网络异常场景(连接超时、重传耗尽)是真实的教学价值所在,但验证脚本本身必须能确定性退出。
- **不引入外部网络依赖**:所有可运行例子只连接本地回环地址或WSL2内部构造的虚拟网络拓扑,不对真实互联网发起请求(避免网络不可用/被墙/限流导致验证失败,这是本系列相对os-concurrency-deep-dive新增的一条纪律,OS系列没有"外部依赖不可控"这类风险)。

## 6. 与仓库已有内容的边界(已核实,不重复)

- **vs rhcsa-bash-deep-dive 07类(网络配置)+ 08类(firewalld防火墙)**:已实读两个文件全部知识点列表核实——07类是 `nmcli`/`nmtui`/`hostnamectl`/`ip`命令族/静态路由怎么操作,08类是 `firewall-cmd`/`firewalld` zone/rich rule怎么配置,均为纯操作层面,面向RHCSA上机考试,明确不采用"面试怎么问"框架。本系列讲协议机制是什么、为什么这样设计、技术面试怎么问,操作层面vs机制/理论层面,不重复。
- **vs learning/cluster-networking(AI infra集群网络研究模块)**:已实读该模块README核实——专注GPU集群互联(NVLink/InfiniBand物理层选型、ring/tree/halving-doubling/SHARP等all-reduce算法时延公式、NCCL协议内部机制),是建立在"已掌握通用网络基础"之上的更高更窄专精层,不讲TCP握手/HTTP/DNS这类通用网络基础。本系列12类"现代网络专题"提及分布式训练网络需求时只做轻量连接(为什么这里需要RDMA/NCCL而不是普通socket/TCP),不重新推导ring/SHARP算法本身,那是cluster-networking的范围。
- **vs dsa-deep-dive 20类进阶深度追加**:该文件案例1(LRU分布式扩展)已有真实一致性哈希负载均衡实测代码,案例3(限流算法)已有固定窗口/滑动窗口/令牌桶/漏桶四种算法的真实边界突刺断言验证。本系列10类"负载均衡算法"和12类"API网关限流"涉及这两个主题时交叉引用,不重新写一遍代码。
- **vs os-concurrency-deep-dive 08类(IO模型演进)**:该类的 `select`/`poll`/`epoll` 只讲"操作系统怎么实现IO事件通知"这个机制角度。本系列11类"网络编程与IO模型"落到网络编程实操层面(比如单线程Redis用epoll处理海量连接的具体应用),不重新讲select/poll/epoll的操作系统内部机制本身,而是引用该系列结论并聚焦"在网络服务器里怎么用"。
- **vs learning/ 其余模块**:未发现其他OSI/TCP-IP协议栈基础理论专题模块,不构成重复。

## 7. 实施阶段划分(供 writing-plans 阶段细化为具体任务)

- Phase 0:脚手架(`for_real_dummy/computer-networking-deep-dive/00-roadmap.md` + 拷贝 `_verify_md.py`,含 `python-wsl2` 围栏标记支持)
- Phase 1:板块 I(01 文件)
- Phase 2:板块 II(02-03 文件)
- Phase 3:板块 III(04-06 文件)
- Phase 4:板块 IV(07-10 文件)
- Phase 5:板块 V(11-12 文件)
- Phase 6:13 类capstone
- Phase 7:全库自查回归(逐文件独立子进程重跑 `_verify_md.py`、结构标记计数核对、网络类断言额外重复验证稳定性)+ `for_real_dummy/README.md` 更新 + memory 更新 + 最终提交

每个板块验证通过后独立提交,不是最后一次性提交,延续此前系列的既有节奏。

## 8. 后续排队项(不在本次范围内,如实记录)

系列 3(数据库原理与实战)、系列 4(系统设计,整合前三者的capstone性质系列)各自需要独立的 brainstorm→spec→plan→execute 循环,本 spec 只覆盖系列 2(计算机网络)。四部曲的总体顺序已经用户确认:操作系统与并发(已完成)→ 计算机网络(本次)→ 数据库原理与实战 → 系统设计。

---

*创建:2026-07-14*
