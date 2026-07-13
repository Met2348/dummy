# 计算机网络深挖 —— 路线图与进度表

> 目标:约 80 个计算机网络知识点,由浅入深,全栈覆盖(链路层→网络层→传输层→应用层),应用层加重,深度对标 [os-concurrency-deep-dive/](../os-concurrency-deep-dive/00-roadmap.md)/[dsa-deep-dive/](../dsa-deep-dive/00-roadmap.md)/[statistics-deep-dive/](../statistics-deep-dive/00-roadmap.md)(面试二三面深度,不是"这个协议是什么"的教科书搬运)。
> 定位:仓库"职业发展与需求"四部曲第 2 部——操作系统与并发(已完成,79点)→ 计算机网络 → 数据库原理与实战 → 系统设计。
> 设计文档:[docs/superpowers/specs/2026-07-14-computer-networking-deep-dive-design.md](../../docs/superpowers/specs/2026-07-14-computer-networking-deep-dive-design.md)、实施计划:[docs/superpowers/plans/2026-07-14-computer-networking-deep-dive.md](../../docs/superpowers/plans/2026-07-14-computer-networking-deep-dive.md)。

---

## 与仓库已有内容的边界(已核实,不重复)

- **vs rhcsa-bash-deep-dive 07类(网络配置)+ 08类(firewalld防火墙)**:已实读两个文件全部知识点列表核实——07类是 `nmcli`/`nmtui`/`hostnamectl`/`ip`命令族/静态路由怎么操作,08类是 `firewall-cmd`/`firewalld` zone/rich rule怎么配置,均为纯操作层面,面向RHCSA上机考试,明确不采用"面试怎么问"框架。本系列讲协议机制是什么、为什么这样设计、技术面试怎么问,不重复。
- **vs learning/cluster-networking(AI infra集群网络研究模块)**:已实读该模块README核实——专注GPU集群互联(NVLink/InfiniBand物理层、ring/tree/halving-doubling/SHARP等all-reduce算法时延公式、NCCL协议内部机制),是建立在"已掌握通用网络基础"之上的更高更窄专精层。本系列12类提及分布式训练网络需求时只做轻量连接,不重新推导ring/SHARP算法本身。
- **vs dsa-deep-dive 20类进阶深度追加**:该文件案例1(LRU分布式扩展)已有真实一致性哈希负载均衡实测代码,案例3(限流算法)已有固定窗口/滑动窗口/令牌桶/漏桶四种算法的真实边界突刺断言验证。本系列10类"负载均衡算法"和12类"API网关限流"涉及这两个主题时交叉引用,不重新写一遍代码。
- **vs os-concurrency-deep-dive 08类(IO模型演进)**:该类的 `select`/`poll`/`epoll` 只讲操作系统怎么实现IO事件通知这个机制角度。本系列11类"网络编程与IO模型"落到网络编程实操层面,引用该系列结论并聚焦"在网络服务器里怎么用"。

## 知识点结构模板(七步,延续 os-concurrency-deep-dive 模板不变)

1. **签名/是什么** 2. **一句话** 3. **底层机制/为什么这样设计**(协议细节——TCP拥塞控制算法族、DNS递归解析层级、TLS握手步骤——直接写在这一步,不单独拆步骤) 4. **AI研究/工程场景** 5. **可运行例子**(assert验证,**必须显式标注验证环境**:`.venv` 或 `WSL2 Rocky Linux`) 6. **面试怎么问+追问链**(五轴方法论,见下表) 7. **常见坑**

## 五轴追问链方法论(从第一天融入,不事后补)

| 轴线 | 含义 | 网络学科例子 |
|------|------|------|
| 规模递增轴 | 小规模→大规模→极限行为 | 单连接→高并发短连接(C10K)→海量长连接(WebSocket集群) |
| 工程约束递增轴 | 单机→跨机→全球分发 | 单机HTTP服务→跨机负载均衡→全球CDN分发(为系统设计系列埋伏笔) |
| 方案批判迭代轴 | 面试官连续指出具体缺陷逼换方案 | DNS轮询负载均衡→无法感知后端健康→四层LB→无法感知HTTP语义→七层LB |
| 决策依据追问轴 | 不纠错,只逼问选择依据 | "为什么这里用gRPC不用REST","为什么选UDP不选TCP" |
| 真实性验证轴 | 简历"优化了接口延迟"被追问到具体数字 | "具体减少了几次往返RTT,用了什么手段,改动前后P99数字是多少" |
| 诊断真实数据(新题型) | 给抓包/监控数据,要求诊断而非套公式 | 给一段真实tcpdump重传统计或延迟分布,要求定位是拥塞、丢包还是服务端处理慢 |

每个知识点挑 1~2 条最自然的轴线走 2~3 层深,不强行凑满 5 轴。13 类模拟终面 capstone 是唯一要求同时用 3 条以上轴线的文件。

## 环境声明:双环境验证策略(延续 os-concurrency-deep-dive,分工按网络学科调整)

- **默认环境**:仓库根目录 `.venv`(Windows 原生 Python),用于:socket编程基础(TCP/UDP client-server跨平台可行)、HTTP请求/响应手工构造解析、WebSocket握手手工实现、拥塞控制算法数值模拟、路由算法(距离矢量/最长前缀匹配)图算法模拟、DNS查询报文手工构造与解析、`ipaddress`标准库子网计算。
- **WSL2 Rocky Linux**(`Rocky Linux 10.2`,`Python 3.12.13`,复用 rhcsa-bash-deep-dive/os-concurrency-deep-dive 已装好的环境,2026-07-14 已确认可用;本次额外安装了 `tcpdump`/`iproute-tc`(提供`tc`)/`openssl` 三个此前系列未装过的工具,`ip`/`ss` 此前已装好):用于真实 Linux 网络语义强相关的知识点——`tcpdump` 真实抓包、`tc` 流量控制(真实注入延迟/丢包观察TCP行为变化)、network namespace + veth pair、raw socket(需root权限)、`ss -tan` 观察真实TCP状态转换、`openssl s_client` 真实TLS握手过程检查。
- 每个知识点的"可运行例子"步骤开头**必须**显式标注验证环境。

## 网络类特有验证纪律

- 计时对比断言用 `best_of(fn, *args, trials=N)` 取多次采样最小值,规模不能小到被系统调度噪声/网络抖动淹没。
- **依赖真实网络往返(RTT)、抓包统计的断言,要求多次运行确认稳定复现**(至少5~10次),不满足于运气好复现一次。
- **需要真实构造超时/重传/连接失败的 demo 必须带确定性退出的超时安全网**,延续 os-concurrency-deep-dive 死锁 demo 纪律。
- **不引入外部网络依赖**:所有可运行例子只连本地回环地址或 WSL2 内部构造的虚拟网络拓扑,不对真实互联网发起请求——DNS 类知识点用手工构造报文或本地 hosts 条目验证,不查询真实公网 DNS 服务器。这是本系列相对 os-concurrency-deep-dive 新增的一条纪律(OS 系列没有"外部依赖不可控"这类风险)。

## 进度表

| # | 板块 | 分类 | 文件 | 知识点数(约) | 状态 |
|---|------|------|------|------------|------|
| 01 | I 分层模型与链路层基础 | 分层模型与链路层 | [01-layering-and-link-basics.md](01-layering-and-link-basics.md) | 6 | ✅ 已完成(6个`.venv`代码块全部通过) |
| 02 | II 网络层:IP与路由 | IP编址与子网划分 | [02-ip-addressing-and-subnetting.md](02-ip-addressing-and-subnetting.md) | 7 | ✅ 已完成(6个`.venv`代码块通过,1点[KP6 IPv6过渡技术]概念性讲解) |
| 03 | II | 路由与网络层机制 | [03-routing-and-network-layer-mechanisms.md](03-routing-and-network-layer-mechanisms.md) | 7 | ✅ 已完成(6个`.venv`代码块通过,1点[KP4 IGP vs BGP]概念性讲解) |
| 04 | III 传输层:TCP/UDP | TCP连接管理 | [04-tcp-connection-management.md](04-tcp-connection-management.md) | 7 | ✅ 已完成(6个`.venv`代码块通过,3点[KP2/3/4]额外用WSL2真实tcpdump+ss验证,1点[KP6]真实跨平台backlog差异对比) |
| 05 | III | TCP可靠传输与流量控制 | [05-tcp-reliability-and-flow-control.md](05-tcp-reliability-and-flow-control.md) | 7 | ✅ 已完成(7个`.venv`代码块通过,6点教学性模拟+1点[KP7]真实TCP socket复现粘包) |
| 06 | III | TCP拥塞控制与UDP | [06-tcp-congestion-control-and-udp.md](06-tcp-congestion-control-and-udp.md) | 7 | ⏳ 未开始 |
| 07 | IV 应用层协议 | HTTP演进 | [07-http-evolution.md](07-http-evolution.md) | 7 | ⏳ 未开始 |
| 08 | IV | HTTPS与TLS | [08-https-and-tls.md](08-https-and-tls.md) | 6 | ⏳ 未开始 |
| 09 | IV | DNS域名解析 | [09-dns-resolution.md](09-dns-resolution.md) | 6 | ⏳ 未开始 |
| 10 | IV | 现代应用协议与API | [10-modern-app-protocols-and-apis.md](10-modern-app-protocols-and-apis.md) | 7 | ⏳ 未开始 |
| 11 | V 现代网络与工程场景 | 网络编程与IO模型 | [11-network-programming-and-io-models.md](11-network-programming-and-io-models.md) | 7 | ⏳ 未开始 |
| 12 | V | 现代网络专题 | [12-modern-networking-topics.md](12-modern-networking-topics.md) | 6 | ⏳ 未开始 |
| 13 | 收尾 | 模拟终面capstone | [13-mock-interview-capstone.md](13-mock-interview-capstone.md) | —(不计入合计) | ⏳ 未开始 |

**目标合计:约 80 个知识点,12 个分类文件 + 1 篇模拟终面 capstone。** 精确数字以全库自查阶段的逐文件核对为准。

---

## 验证纪律

- 验证脚本 `_verify_md.py`(regex 提取 ` ```python ` 代码块,每块独立 subprocess 执行;另支持 ` ```python-wsl2 ` 标记只统计不执行)直接拷贝自 `os-concurrency-deep-dive/_verify_md.py`,不重新设计。
- 数学/机制结论必须用 `assert` 数值验证,不能只摆公式描述。
- print() 语句必须纯 ASCII(历史教训:Windows GBK locale 下 `_verify_md.py` 子进程 reader 线程遇到非 ASCII 字符会 UnicodeDecodeError);正文 markdown 数学/协议符号不受此限制。
- 不新增任何第三方 Python 依赖,标准库优先。
- 设计文档与实施计划见上方链接。

---

*创建:2026-07-14*
