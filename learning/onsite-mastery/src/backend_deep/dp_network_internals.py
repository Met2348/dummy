"""计算机网络深水区（终面深水区，约 12 个 DeepPoint）。

覆盖：TCP拥塞控制(Cubic/BBR)行为差异、HTTPS握手优化(会话复用/TLS1.3 0-RTT)、DNS解析失败排查、
三次握手/四次挥手真实网络异常边界(TIME_WAIT过多)、长连接心跳设计权衡，以及SYN flood、HTTP/2-3
队头阻塞、抓包排查、Nagle+延迟确认交互、PMTUD黑洞等相邻真实生产topic。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_common import DeepPoint, categories, drill, grade_chain  # noqa: E402

CAT = "计算机网络深水"

BANK: list[DeepPoint] = [
    DeepPoint(
        id="dp-be-net-01", cat=CAT,
        trigger="三次握手你肯定会背，但如果握手过程中丢包了，实际会发生什么？",
        chain=(
            ("正常的TCP三次握手/四次挥手流程是什么？",
             "三次握手：客户端发SYN(seq=x)进入SYN_SENT；服务端回SYN+ACK(seq=y,ack=x+1)进入SYN_RCVD；"
             "客户端回ACK(ack=y+1)，双方进入ESTABLISHED。四次挥手：主动关闭方发FIN，被动方回ACK"
             "(此时还能继续发数据，进入CLOSE_WAIT)，数据发完后再发FIN，主动方回ACK并进入TIME_WAIT。",
             ("SYN", "SYN_RCVD", "FIN", "CLOSE_WAIT", "TIME_WAIT")),
            ("为什么是四次挥手而不是像握手一样三次？能不能把中间的ACK和FIN合并？",
             "因为TCP是全双工的，收到对方FIN只表示对方不再发送数据了，本端可能还有数据没发完，不能"
             "立刻把ACK和FIN合并回复；如果被动关闭方此时已经没有数据要发送，协议栈层面可以延迟发送"
             "ACK，等待和自己的FIN一起捎带发出(delayed ACK优化)合并成三次，但这不是协议保证的，取决于"
             "实现。",
             ("全双工", "捎带", "delayed ACK", "实现")),
            ("如果客户端发的第三次握手ACK在网络上丢了，服务端和客户端分别是什么状态？会发生什么？",
             "服务端一直停留在SYN_RCVD状态，SYN+ACK会按tcp_synack_retries控制的次数超时重传；客户端"
             "因为自己已经发出ACK，会直接进入ESTABLISHED状态并可能开始发数据；如果长时间收不到ACK，"
             "服务端的半连接会因为超过重传次数而被丢弃，客户端后续发的数据会收到RST。",
             ("SYN_RCVD", "超时重传", "tcp_synack_retries", "RST")),
            ("线上你怎么用抓包判断某次连接建立是不是卡在了握手阶段丢包重传？以及主动关闭方大量停留在"
             "FIN_WAIT_2会是什么原因？",
             "用tcpdump抓包看SYN包时间戳，如果同一五元组短时间内出现多个SYN(序号相同)说明发生了SYN"
             "重传；用ss -tan统计各状态连接数，如果大量连接停留在FIN_WAIT_2，通常是对端迟迟不发最后"
             "的FIN——可能对端应用层资源泄漏没有正确调用close，可以用tcp_fin_timeout给FIN_WAIT_2设置"
             "超时强制清理。",
             ("tcpdump", "ss -tan", "FIN_WAIT_2", "tcp_fin_timeout")),
        ),
        pitfall="很多人背得出四次挥手的流程图，但答不出握手/挥手中间某一步丢包之后双方状态机具体停在"
                "哪、怎么恢复；第4层常常答不出FIN_WAIT_2堆积是什么信号",
        real_world_link=""),

    DeepPoint(
        id="dp-be-net-02", cat=CAT,
        trigger="你提到线上TIME_WAIT连接数太多导致端口耗尽，具体怎么处理的？",
        chain=(
            ("TIME_WAIT状态存在的意义是什么？",
             "主动关闭方发出最后的ACK后进入TIME_WAIT，停留2MSL，目的有两个：防止最后这个ACK丢失、"
             "被动方重传FIN时能被正确响应；防止这一对四元组关闭后立刻被新连接复用，导致网络中滞留的"
             "旧报文被新连接误收(串门)。",
             ("2MSL", "ACK丢失", "旧报文被新连接误收")),
            ("如果作为高并发的客户端(比如反向代理向后端发大量短连接)，TIME_WAIT暴涨会导致什么具体"
             "问题？",
             "每个TIME_WAIT连接占用一个本地(源IP,源端口)组合，本地端口范围有限(ip_local_port_range)，"
             "当同时存在的TIME_WAIT连接数逼近可用端口数时，新建连接找不到空闲源端口，会报错"
             "'Cannot assign requested address'，本质是客户端出口的四元组资源耗尽。",
             ("资源耗尽", "ip_local_port_range", "Cannot assign requested address")),
            ("既然TIME_WAIT是为了'安全关闭'，为什么不能简单粗暴地把2MSL等待时间调短来解决端口耗尽？",
             "调短2MSL会让网络里滞留的旧连接延迟报文有更大概率在新连接建立后才到达，被误收导致数据"
             "错乱；真正安全的做法是用tcp_tw_reuse(只在满足tcp_timestamps等条件、确认安全的前提下"
             "允许把TIME_WAIT状态的四元组用于新的出向连接)，这是内核在合适条件下做的安全复用，而不是"
             "简单缩短等待时间。",
             ("调短2MSL", "tcp_tw_reuse", "tcp_timestamps")),
            ("生产环境里具体怎么系统性解决这个问题，而不是只调内核参数？",
             "根本思路是减少'频繁建立-关闭短连接'这个行为本身——反向代理到后端改用长连接/连接池复用，"
             "从源头减少TIME_WAIT产生速率；内核参数层面开启net.ipv4.tcp_tw_reuse=1，扩大"
             "ip_local_port_range；要注意tcp_tw_recycle在NAT环境下已知有问题，高版本内核已经彻底"
             "移除；排查用ss -tan state time-wait量化实际数量。",
             ("连接池复用", "tcp_tw_reuse", "tcp_tw_recycle", "ss -tan state time-wait")),
        ),
        pitfall="很多人只会说'调tcp_tw_reuse'，说不清这和缩短2MSL在安全性上的区别，也不知道"
                "tcp_tw_recycle在NAT环境下的坑（这是真实的历史故障案例）",
        real_world_link=""),

    DeepPoint(
        id="dp-be-net-03", cat=CAT,
        trigger="你提到TCP用Cubic做拥塞控制，具体的窗口调整行为是什么样的？",
        chain=(
            ("Cubic算法拥塞窗口的调整规则是什么？",
             "Cubic用一个三次函数(cubic function)描述拥塞窗口随时间的增长曲线：发生丢包后先把窗口"
             "乘性减小记为Wmax，之后窗口按三次函数增长——先快后慢再快，在接近Wmax附近增长明显放缓"
             "('先平后陡'的S形曲线)，一旦窗口超过Wmax附近后又重新加速探测更大的带宽上限。",
             ("三次函数", "Wmax", "乘性减小", "先平后陡")),
            ("Cubic为什么被认为对高带宽时延积(BDP)网络更友好，相比传统的Reno/AIMD有什么优势？",
             "Reno类算法窗口增长是线性的，每个RTT只加1个MSS，在高BDP网络下需要非常多个RTT才能恢复到"
             "丢包前的窗口，恢复慢导致带宽利用率长期偏低；Cubic的窗口增长和时间(而不是RTT次数)相关，"
             "恢复速度基本独立于RTT，对RTT差异很大的混合网络环境更公平。",
             ("BDP", "线性的", "恢复速度", "更公平")),
            ("Cubic这种基于丢包来判断拥塞的方式，在什么网络环境下会表现不好？这引出了BBR想解决"
             "什么问题？",
             "Cubic是loss-based拥塞控制，在有缓冲区较大的网络设备(bufferbloat)场景下，窗口会一直"
             "增长把中间路由器的缓冲区填满，排queuing时延大幅上升却还没丢包，Cubic会持续加剧排队而"
             "不自知；在有少量随机丢包但并非真正拥塞的无线网络，Cubic会把随机丢包误判为拥塞信号从而"
             "不必要地大幅降低窗口。",
             ("loss-based", "bufferbloat", "排queuing时延", "随机丢包误判")),
            ("BBR具体是怎么绕开丢包信号、用另一套逻辑判断拥塞的？生产环境什么场景该考虑切换到BBR？",
             "BBR不依赖丢包，而是持续测量瓶颈链路的带宽BtlBw(观察一段时间内的最大交付速率)和最小RTT"
             "RTprop(排除排队时延的纯传播时延)，用BDP=BtlBw×RTprop算出理论最优发送速率主动控制发送；"
             "跨地域/跨国长距离链路、视频直播等对延迟敏感又需要高吞吐的场景BBR通常表现优于Cubic，但"
             "BBR和Cubic流共享瓶颈链路时BBR可能更激进地抢占带宽，需要结合具体网络环境评估。",
             ("BtlBw", "RTprop", "BDP", "抢占带宽")),
        ),
        pitfall="很多人能说出'Cubic是三次函数曲线'但说不清楚为什么这样设计对高BDP网络友好；第4层常"
                "答不出BBR和Cubic混跑时的公平性问题这个真实的生产权衡",
        real_world_link=""),

    DeepPoint(
        id="dp-be-net-04", cat=CAT,
        trigger="你提到通过会话复用优化了HTTPS握手性能，具体怎么做的？",
        chain=(
            ("一次完整的TLS握手(以TLS1.2为例)大致经过哪些步骤，为什么说它'慢'？",
             "完整TLS1.2握手：客户端发ClientHello；服务端回ServerHello+证书+ServerHelloDone；客户端"
             "验证证书后用非对称加密交换预主密钥并发送，双方各自推导出会话密钥，再发送Finished确认；"
             "整个过程通常需要2个RTT，且涉及非对称加密运算和证书链验证，对短连接、高并发场景延迟"
             "影响明显。",
             ("ClientHello", "预主密钥", "2个RTT", "非对称加密")),
            ("会话复用(session resumption)具体是怎么把握手简化的，Session ID和Session Ticket两种"
             "方式有什么区别？",
             "会话复用让后续重连时跳过完整密钥协商，只需1个RTT甚至0-RTT恢复通信；Session ID方式服务端"
             "把主密钥等会话状态缓存起来，客户端下次带ID来查询恢复，多机部署需要做会话共享，运维成本"
             "较高；Session Ticket方式服务端把会话状态加密后交给客户端自己保存，服务端不需要存任何"
             "东西，是无状态的，更适合多机/CDN场景。",
             ("Session ID", "Session Ticket", "无状态", "会话共享")),
            ("Session Ticket方式下，如果加密Ticket用的密钥长期不轮换，会有什么安全隐患？",
             "Session Ticket的加密密钥(STEK)如果长期不更换，一旦泄露，攻击者可以解密出历史上所有用"
             "这个密钥加密过的ticket，从而恢复出对应会话的密钥，破坏了前向保密——即使每次会话本身用了"
             "支持前向保密的ECDHE，ticket密钥不轮换等于给所有历史会话开了后门，因此STEK需要定期轮换"
             "并安全销毁旧密钥。",
             ("STEK", "前向保密", "历史会话开了后门", "定期轮换")),
            ("生产环境里怎么验证TLS会话复用真的生效了，以及多机部署下具体怎么落地Session Ticket的"
             "密钥同步？",
             "用openssl s_client -connect host:443 -reconnect模拟同一client多次连接，观察后续握手是"
             "否显示Reused字样确认复用生效；也可以抓包看第二次ClientHello是否带Session Ticket扩展、"
             "握手消息数量是否减少。多机部署要保证所有边缘节点使用同一份STEK并按统一节奏轮换，通常由"
             "中心化的KMS定期生成新密钥分发到所有边缘节点，同时保留上一版本密钥一小段时间过渡。",
             ("openssl s_client", "Reused", "抓包看", "KMS")),
        ),
        pitfall="很多人只知道'session resumption能加速'，说不清Session ID和Session Ticket服务端"
                "有状态/无状态这个关键区别，更答不出STEK不轮换会破坏前向保密这个安全权衡",
        real_world_link=""),

    DeepPoint(
        id="dp-be-net-05", cat=CAT,
        trigger="你提到TLS1.3的0-RTT能进一步优化握手延迟，具体是怎么做到'0-RTT'的，有什么代价？",
        chain=(
            ("TLS1.3相比1.2在握手轮数上做了什么优化，完整握手需要几个RTT？",
             "TLS1.3把完整握手从1.2的2-RTT优化到1-RTT——客户端在ClientHello里直接带上它认为服务端"
             "会选用的key_share(基于双方都支持的椭圆曲线)一起发出，服务端如果同意可以在ServerHello里"
             "直接带上自己的key_share并计算共享密钥，同时把证书、Finished都在第一次回复里带上，全程"
             "只需要1个RTT。",
             ("1-RTT", "key_share", "ServerHello", "证书")),
            ("0-RTT具体是在1-RTT基础上又省掉了哪一次往返，靠什么机制实现？",
             "0-RTT建立在会话恢复机制之上——客户端如果之前建立过连接，会保存一个PSK(通常通过"
             "NewSessionTicket扩展获得)，重连时客户端可以在第一个数据包里就把ClientHello和用PSK派生"
             "密钥加密的Early Data一起发出去，不需要等服务端回应就能抢跑发送业务数据。",
             ("PSK", "Early Data", "NewSessionTicket", "抢跑发送业务数据")),
            ("0-RTT这种'抢跑'带来了什么安全风险，为什么说它不能天然防重放？",
             "因为0-RTT的Early Data在服务端还没有完成对客户端实时'新鲜性'确认之前就被处理了——攻击者"
             "截获这个0-RTT请求包可以原样重放给服务端，服务端无法区分是新请求还是重放的旧请求，会"
             "重复执行；这对幂等操作问题不大，但对非幂等操作如果被重放会造成严重业务问题，这是协议"
             "设计上的权衡，不是bug。",
             ("原样重放", "新鲜性", "非幂等操作", "设计上的权衡")),
            ("生产环境如果要开启0-RTT，具体怎么在业务层规避重放风险？",
             "常见做法是服务端只对明确标记为幂等的请求类型允许走0-RTT通道，非幂等的写操作即使客户端"
             "尝试发送也要求走完整握手；一些实现支持配置early data的反重放窗口，拒绝重复的early data"
             "(但无法防止跨多实例的重放，除非有集中式重放缓存)；很多团队的实际选择是只在CDN边缘对"
             "只读接口开启0-RTT，写接口一律强制走完整握手，用读写分离规避重放风险。",
             ("标记为幂等", "反重放窗口", "集中式重放缓存", "读写分离")),
        ),
        pitfall="很多人只知道'0-RTT更快'，说不清楚它靠PSK+Early Data具体怎么省掉一次RTT，更答不出"
                "重放攻击是协议设计上的固有权衡而非bug，也不知道生产环境怎么在业务层规避",
        real_world_link=""),

    DeepPoint(
        id="dp-be-net-06", cat=CAT,
        trigger="你说线上有个服务偶发'connection failed'，最后查出来是DNS解析问题，具体怎么一步步"
                "排查的？",
        chain=(
            ("遇到DNS解析问题，最基本的排查命令和顺序是什么？",
             "先用dig(或nslookup)域名，观察能否解析出IP、返回的TTL、以及查询耗时；如果直接查询公共"
             "DNS能解析但用本机默认DNS服务器解析不出来，问题范围缩小到本机/内部DNS服务器配置；同时"
             "检查resolv.conf里配置的nameserver是否可达、hosts文件是否有异常的静态解析记录覆盖。",
             ("dig", "resolv.conf", "记录覆盖")),
            ("如果dig能查到正确结果但应用日志里报解析超时/失败，可能是哪个环节的问题？",
             "常见原因包括应用容器内的DNS配置(比如Kubernetes里的CoreDNS配置)和宿主机不一致；本地DNS"
             "缓存(nscd/systemd-resolved)缓存了过期记录没有及时刷新；也可能是并发量大时本地UDP DNS"
             "查询的conntrack表被打满导致部分查询包被丢弃。",
             ("容器内的DNS配置", "本地DNS缓存", "conntrack表被打满")),
            ("如果确认是conntrack表满导致UDP DNS查询丢包，为什么这个问题往往只在高并发/短时间内"
             "爆发，平时测试却复现不了？",
             "conntrack表记录的是内核为UDP这种无连接协议维护的'伪连接'跟踪项，正常情况下记录很快因为"
             "收到响应或超时而回收；但当QPS突然飙升，短时间内创建的跟踪记录来不及回收，逼近"
             "nf_conntrack_max上限，新查询包会被内核直接丢弃(dmesg里能看到table full dropping "
             "packet)，因此只有高并发瞬时冲击时才会暴露。",
             ("跟踪记录", "nf_conntrack_max", "table full dropping packet")),
            ("确认是conntrack打满导致DNS丢包之后，具体怎么修复和预防？",
             "应急上可以临时调大nf_conntrack_max(同时调大nf_conntrack_buckets)，或缩短"
             "nf_conntrack_udp_timeout让记录更快回收；根本预防措施是从应用层减少重复的DNS查询——"
             "引入本地DNS结果缓存/连接池复用，把'每次请求都查一次DNS'的模式改成定期刷新缓存的解析"
             "结果，从源头降低DNS查询的QPS。",
             ("nf_conntrack_max", "nf_conntrack_udp_timeout", "本地DNS结果缓存")),
        ),
        pitfall="很多人只会说'用dig排查'，遇到高并发下偶发DNS失败时答不出conntrack表满这个隐蔽原因，"
                "更想不到这是'瞬时冲击'特有的现象而非稳定复现的问题",
        real_world_link=""),

    DeepPoint(
        id="dp-be-net-07", cat=CAT,
        trigger="你提到给长连接加了心跳机制，具体的心跳间隔怎么定的，为什么不直接依赖TCP自带的"
                "keepalive？",
        chain=(
            ("TCP自带的keepalive机制是怎么工作的，为什么很多应用还要自己在应用层实现心跳？",
             "TCP keepalive是内核层面的探测机制，连接空闲超过tcp_keepalive_time(默认常见2小时)后开始"
             "发送探测包，连续tcp_keepalive_probes次都没响应才判定连接已死；这个默认周期对需要快速"
             "感知连接异常的场景太长，而且只能感知网络层面连接是否还通，无法感知应用层是否还活着"
             "(进程假死但TCP连接本身正常)，所以应用层通常自己实现更短周期、能验证业务健康的心跳。",
             ("tcp_keepalive_time", "网络层面", "进程假死", "业务健康")),
            ("应用层心跳间隔设置成怎样是合理的，设太短/太长分别有什么代价？",
             "心跳间隔设太短，在连接数很大的场景下会造成巨大的心跳流量和CPU开销；设太长则客户端断线"
             "后服务端要等很久才能感知到连接已死，期间可能继续推送消息造成丢失或积压，也可能长期占用"
             "服务端连接资源不释放；实践中根据业务对断线感知延迟的容忍度反推心跳周期，同时允许一定的"
             "重试次数避免网络抖动造成误判。",
             ("心跳流量", "连接资源不释放", "断线感知延迟", "重试次数")),
            ("如果是移动端弱网环境，固定心跳间隔的方案会遇到什么问题，业界一般怎么优化？",
             "移动网络下NAT网关/运营商网络设备通常会在连接空闲一段时间后主动清理NAT映射表项，如果"
             "应用层心跳间隔比这个NAT超时阈值长，连接会被中间设备静默断开；同时移动网络下频繁心跳"
             "非常耗电和流量，业界通常采用自适应心跳策略——先用较短间隔探测出网络能容忍的最大空闲"
             "间隔，再回退到略小于这个值的间隔，不同网络类型动态调整。",
             ("清理NAT映射表项", "非常耗电", "自适应心跳", "动态调整")),
            ("生产环境里怎么验证心跳机制真的在断线时能被正确、及时地感知到，怎么排查'心跳正常但业务"
             "其实已经死了'的假活问题？",
             "验证手段包括主动模拟断线后掐表观察对端多久感知到断连并清理资源；对于假活问题，需要在"
             "心跳里加入业务层面的自检——比如要求对端返回真实的处理队列积压情况而不只是回一个空的"
             "pong；此外可以结合服务端主动监控每个连接最近一次成功业务响应的时间戳，超过阈值即使心跳"
             "正常也主动断开重连，作为兜底。",
             ("掐表观察", "层面的自检", "假活问题", "响应的时间戳")),
        ),
        pitfall="很多人只会说'心跳间隔设几秒'，说不清为什么TCP keepalive不够用，更答不出移动网络"
                "NAT超时和'假活'这两个更深的坑",
        real_world_link=""),

    DeepPoint(
        id="dp-be-net-08", cat=CAT,
        trigger="你提到线上遭遇过SYN flood攻击，具体是怎么识别和处理的？",
        chain=(
            ("半连接队列和全连接队列分别是什么，各自对应TCP握手的哪个阶段？",
             "服务端收到SYN后，在半连接队列(SYN queue)里创建记录(状态SYN_RCVD)并回复SYN+ACK，等待"
             "客户端最终ACK；收到ACK后这个连接会从半连接队列移到全连接队列(accept queue)，等待应用"
             "调用accept()取走；半连接队列大小由tcp_max_syn_backlog控制，全连接队列大小由listen()"
             "的backlog参数(同时受somaxconn限制)控制。",
             ("半连接队列", "全连接队列", "tcp_max_syn_backlog", "somaxconn")),
            ("SYN flood攻击具体是利用了这个机制的什么弱点？",
             "攻击者伪造大量不同的源IP发送SYN包但从不回复最后的ACK，服务端为每个SYN都要在半连接队列"
             "里维护一条记录并等待重传超时才能回收，攻击者只需维持发送速率超过半连接队列的回收速率，"
             "就能把半连接队列迅速填满，导致合法用户的正常SYN请求被丢弃，是典型的资源耗尽型DoS攻击。",
             ("源IP", "迅速填满", "请求被丢弃", "资源耗尽型DoS")),
            ("Linux内核默认是怎么防御SYN flood的？SYN Cookies机制具体怎么做到'不用维护半连接队列"
             "状态'的？",
             "开启tcp_syncookies后，当半连接队列即将溢出时，内核不再在队列里为每个SYN分配存储空间，"
             "而是把连接的关键参数编码进SYN+ACK返回的初始序列号(通过加密哈希函数把源地址、端口、"
             "服务端密钥、时间窗口等编码进序列号)，服务端不需要保存任何状态；客户端的最终ACK到来时，"
             "服务端从确认号反推出之前编码的序列号，重新计算哈希验证是否合法。",
             ("tcp_syncookies", "加密哈希", "分配存储空间", "确认号反推")),
            ("既然SYN Cookies能防SYN flood，为什么它不是默认一直无条件开启，而是'队列快满了才"
             "触发'？以及生产环境具体怎么监控/确认正在遭受SYN flood？",
             "SYN Cookies有代价——因为不保存真实的TCP选项状态，一些高级特性(如窗口缩放因子)在走"
             "SYN Cookies路径时会被迫降级，影响正常连接性能，所以只在队列快满、判断可能正遭受攻击"
             "时才启用；生产环境可以用netstat -s看SYNs to LISTEN sockets dropped等计数器是否快速"
             "增长，也可以监控ss -s统计SYN_RECV状态连接数是否异常暴涨，确认后除内核缓解外还应该在"
             "边界负载均衡器/云WAF上做流量清洗。",
             ("被迫降级", "netstat -s", "异常暴涨", "边界负载均衡器/云WAF")),
        ),
        pitfall="很多人知道'开syncookies能防'，但说不清楚它具体怎么在'不存储状态'的前提下还能验证"
                "ACK合法性，也答不出为什么不能无条件一直开启（TCP选项降级代价）",
        real_world_link=""),

    DeepPoint(
        id="dp-be-net-09", cat=CAT,
        trigger="你提到HTTP/2解决了队头阻塞，但HTTP/3又是为了解决什么进一步的问题？",
        chain=(
            ("HTTP/1.1的队头阻塞(head-of-line blocking)是怎么产生的，HTTP/2怎么解决的？",
             "HTTP/1.1在一条TCP连接上同一时刻只能有一个请求在等待响应，多个响应必须按发送顺序依次"
             "返回，这就是应用层的队头阻塞；HTTP/2引入了'流(stream)'，在同一条连接上把不同请求的数据"
             "切成帧(frame)交织发送，每个帧带stream id，接收端按stream id重新组装，多个请求可以真正"
             "并发传输。",
             ("依次返回", "stream", "交织发送", "应用层的队头阻塞")),
            ("HTTP/2既然已经解决了应用层队头阻塞，为什么实际使用中还是会遇到类似队头阻塞的卡顿"
             "现象？",
             "因为HTTP/2的多路复用发生在应用层，但底层还是承载在一条TCP连接上；TCP要求字节流严格"
             "按序交付给上层应用——如果这条TCP连接上发生了丢包，在收到重传的丢失数据包之前，即使后面"
             "的字节已经先到达，TCP协议栈也不会把它们提前交给上层，一个packet丢失会阻塞该连接上所有"
             "stream的数据，这是发生在传输层的队头阻塞。",
             ("一条TCP连接", "按序交付", "所有stream的数据", "传输层的队头阻塞")),
            ("HTTP/3底层改用QUIC协议，QUIC具体是怎么从协议设计上避免这种传输层队头阻塞的？",
             "QUIC基于UDP实现，自己重新实现了可靠传输，关键区别是QUIC把stream的概念下沉到了传输层"
             "本身——每个stream有自己独立的可靠传输和排序保证，如果stream A的某个包丢了，只有stream A"
             "需要等待重传，stream B/C的数据只要自己没丢就能立刻被处理；QUIC还把连接标识从四元组改成"
             "了独立的Connection ID，使连接在客户端网络切换时也能不中断地迁移延续。",
             ("基于UDP实现", "排序保证", "Connection ID", "迁移延续")),
            ("既然QUIC这么好，为什么它落地推广的过程中还会遇到一些实际部署上的阻力/问题？",
             "一是很多传统网络中间设备对UDP流量的处理没有TCP成熟，有些环境会限速甚至直接封禁大流量"
             "UDP，导致QUIC连接反而不如TCP稳定，需要有失败自动回退到TCP+HTTP/2的降级策略；二是QUIC"
             "的加密和拥塞控制都在用户态实现，纯软件处理在高吞吐场景下的CPU开销比内核态TCP更高，需要"
             "依赖UDP GSO/GRO这类批量收发优化或硬件网卡offload来降低成本。",
             ("封禁大流量UDP", "自动回退", "内核态TCP更高", "GSO/GRO")),
        ),
        pitfall="很多人只会说'HTTP/2解决了队头阻塞'，说不清楚它其实只解决了应用层的、底层TCP层面的"
                "队头阻塞仍然存在；也答不出QUIC具体怎么让stream互相独立、以及现实部署中UDP被限流的"
                "阻力",
        real_world_link=""),

    DeepPoint(
        id="dp-be-net-10", cat=CAT,
        trigger="你提到用tcpdump抓包定位了一次线上的网络性能问题，具体的排查过程是怎样的？",
        chain=(
            ("怀疑网络层面有问题时，tcpdump抓包的基本使用姿势是什么？",
             "通常先用tcpdump -w capture.pcap落盘(生产环境优先落盘用Wireshark离线分析，避免抓包本身"
             "影响现场排查节奏)，加-s 0保证抓到完整包不被截断；导入Wireshark后用Conversations视图先"
             "整体看这条连接的时间线，再用过滤器tcp.analysis.retransmission筛出所有被自动识别为重传"
             "的包。",
             ("tcpdump -w", "Wireshark", "Conversations视图", "tcp.analysis.retransmission")),
            ("Wireshark标记的tcp.analysis.retransmission和tcp.analysis.duplicate_ack分别对应什么"
             "场景，怎么用它们区分'丢包'和'乱序'？",
             "retransmission表示发送方重发了之前发过的数据；duplicate_ack表示接收方收到了乱序到达的"
             "数据段，回复了和之前一样的ACK号；如果看到少量duplicate_ack后紧跟着一次触发快速重传机制"
             "的retransmission然后恢复正常，通常是网络抖动造成的乱序；如果是大量连续的duplicate_ack、"
             "之后是RTO超时触发的重传，说明是真的丢包而不是简单乱序。",
             ("duplicate_ack", "快速重传", "RTO超时触发", "真的丢包")),
            ("如果抓包发现请求发出去很久之后才有响应，但既没有明显的重传也没有丢包迹象，这时候问题"
             "可能出在哪，怎么进一步定位是客户端、网络还是服务端的问题？",
             "在Wireshark里用tcp.time_delta字段看请求包和响应包之间的时间差；如果请求发出后很久服务端"
             "才回复第一个响应包(且这中间没有任何重传)，通常说明延迟发生在服务端内部处理，而不是网络"
             "传输问题——如果是纯网络问题一般会伴随可观察到的重传/乱序特征；这时需要结合服务端自己的"
             "APM/trace进一步定位是哪个内部环节慢。",
             ("tcp.time_delta", "网络传输问题", "内部环节慢", "APM/trace")),
            ("生产环境不可能对所有流量长期全量抓包，出现偶发性、难复现的网络问题时，你怎么设计一个"
             "可持续的抓包排查方案？",
             "常见做法是条件触发式抓包——监控关键指标(如P99延迟、重传率)超过阈值自动触发一段时间的"
             "tcpdump抓包并保存；也可以用tcpdump的-C和-W参数做滚动抓包，只保留最近一段时间的流量；对"
             "特别关键的服务，用ss -i查看每条连接的tcp_info统计信息(重传次数、RTT等)做低开销的持续"
             "监控，只有怀疑范围明确锁定后才针对性定向抓包。",
             ("条件触发式抓包", "滚动抓包", "ss -i", "定向抓包")),
        ),
        pitfall="很多人只会说'抓包看一下'，说不清楚duplicate_ack和retransmission的具体区别及怎么用"
                "它们区分乱序和真实丢包；第4层常答不出生产环境不能长期全量抓包时该怎么设计可持续方案",
        real_world_link=""),

    DeepPoint(
        id="dp-be-net-11", cat=CAT,
        trigger="你提到一个RPC接口偶发40ms左右的延迟，最后定位到是Nagle算法和延迟确认的interaction，"
                "具体是怎么回事？",
        chain=(
            ("Nagle算法是做什么的，延迟确认(delayed ACK)又是做什么的？",
             "Nagle算法是发送方的优化——如果之前发出去的小数据包没有被确认，新产生的小数据不会立刻"
             "发送，而是攒着等ACK到达或攒够一个MSS才一起发出；延迟确认是接收方的优化——收到数据后不"
             "立刻回ACK，而是等约40ms看是否有数据要一起捎带发送。",
             ("Nagle算法", "攒", "延迟确认", "40ms")),
            ("这两个各自都是好的优化，为什么放在一起会造成明显的延迟叠加？",
             "典型场景是请求-响应模式的应用协议分两次write发送一个逻辑请求，第一次write发出的小包"
             "因为Nagle算法要等ACK才能发第二个包；而接收方由于延迟确认，收到这个小包后不着急回ACK，"
             "而是等着看有没有更多数据一起确认——发送方在等ACK才敢发第二部分数据，接收方在等更多数据"
             "才回ACK，直到接收方的延迟确认定时器超时被迫独立发出这个ACK，发送方才能继续发第二个包。",
             ("两次write", "才敢发第二部分数据", "由于延迟确认", "定时器超时")),
            ("如果只关闭Nagle算法(设置TCP_NODELAY)，这个问题一定能解决吗？为什么很多RPC框架的最佳"
             "实践里既要开TCP_NODELAY又要注意write的调用方式？",
             "设置TCP_NODELAY关闭Nagle后发送方不再攒包，能避免'发送方等ACK'这一侧的延迟，但如果应用层"
             "仍然习惯性地拆成多次小的write调用，即使关闭了Nagle，也会产生多个独立的小TCP段；最佳实践"
             "通常是在关闭Nagle的同时，尽量在应用层把一个逻辑消息拼装成一个缓冲区一次性write出去(或用"
             "writev聚合写)，不只依赖关闭Nagle来掩盖拆包习惯带来的问题。",
             ("TCP_NODELAY", "习惯性地拆", "writev聚合", "不只依赖关闭Nagle")),
            ("生产环境里怎么定位一个延迟问题具体就是Nagle+延迟确认导致的，而不是别的原因造成的几十"
             "毫秒延迟？",
             "最直接的方式是抓包观察请求的两个分片包之间的时间差——如果看到发送方两次write对应的两个"
             "TCP段之间恰好间隔30-40ms左右且中间夹着一个纯ACK包，这是非常典型的Nagle+延迟确认交互"
             "特征；也可以在应用层加耗时打点确认延迟卡在两次write之间；确认后给socket设置TCP_NODELAY"
             "重新测试，如果那额外的30-40ms延迟消失，就基本实锤了原因。",
             ("间隔30-40ms左右", "夹着一个纯ACK包", "设置TCP_NODELAY", "耗时打点")),
        ),
        pitfall="很多人知道Nagle和延迟确认各自是什么，但说不清楚两者交互会产生几十毫秒延迟的具体"
                "机制，也容易以为'关掉Nagle就万事大吉'而忽略了应用层拆包本身也要优化",
        real_world_link=""),

    DeepPoint(
        id="dp-be-net-12", cat=CAT,
        trigger="你提到线上出现了'大包发不出去、小包正常'的诡异现象，这是什么原因？",
        chain=(
            ("什么是MTU，为什么大于MTU的IP包需要分片？",
             "MTU(Maximum Transmission Unit)是链路层单次能传输的最大帧大小(以太网通常1500字节)，"
             "如果IP层要发送的数据包大于出口链路的MTU，就需要在IP层做分片(fragmentation)，拆成多个"
             "小于MTU的分片分别发送，接收端再重新组装。",
             ("MTU", "1500字节", "fragmentation", "重新组装")),
            ("TCP是怎么尽量避免IP分片的？PMTUD(路径MTU发现)具体是怎么工作的？",
             "TCP连接建立时通过MSS选项协商双方能接受的最大段大小，从源头避免分片；但一条路径上可能"
             "有多段不同MTU的链路，PMTUD通过发送时设置IP头的DF(Don't Fragment)位，中间路由器发现包"
             "超过自己的MTU又不能分片时会丢弃该包并回一个ICMP'Fragmentation Needed'报文告知发送方"
             "路径实际的MTU，发送方据此调低后续包大小。",
             ("MSS", "Don't Fragment", "Fragmentation Needed", "调低后续包大小")),
            ("如果中间某个防火墙把这个ICMP报文全部过滤掉了，会发生什么现象，这就是所谓的PMTUD"
             "黑洞吗？",
             "是的——如果路径中间确实存在MTU更小的环节，但探测失败的ICMP报文被安全设备过滤丢弃，"
             "发送方永远收不到'需要调小MTU'的反馈，会持续用过大的包发送、不断被中间设备丢弃，表现为"
             "小的请求/响应能正常收发，但传输大数据量时发不出去或卡住超时，这就是PMTUD黑洞，非常隐蔽"
             "因为常规的连通性测试(ping小包)完全正常。",
             ("PMTUD黑洞", "过滤丢弃", "卡住超时", "连通性测试")),
            ("生产环境里具体怎么排查确认是PMTUD黑洞导致的问题，以及有什么绕过方案？",
             "用tcpdump抓包观察，如果看到发送方反复重传同一个较大的包但从未收到ACK，且包设置了DF"
             "标志位，可以怀疑是MTU黑洞；用ping -M do -s <size>逐步增大size探测出实际可通过的最大包"
             "大小，或用tracepath自动探测路径MTU；绕过方案包括放行ICMP Type3 Code4(治本)，或在服务端"
             "用iptables的clamp-mss-to-pmtu规则主动调低MSS，从源头避免发出过大的包。",
             ("tcpdump抓包观察", "ping -M do", "tracepath", "clamp-mss-to-pmtu")),
        ),
        pitfall="很多人根本没听说过PMTUD黑洞这个现象，遇到'小包正常大包卡死'时会先怀疑是应用层或"
                "带宽问题，想不到是ICMP被过滤导致MTU协商失败",
        real_world_link=""),
]


def _self_test() -> None:
    assert 11 <= len(BANK) <= 13, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [dp.id for dp in BANK]
    assert len(ids) == len(set(ids)), "存在重复 id"
    assert all(i.startswith("dp-be-net-") for i in ids), "id 前缀不一致"
    for dp in BANK:
        assert len(dp.chain) >= 3, f"{dp.id} chain层数不足"
        assert dp.trigger and dp.pitfall, f"{dp.id} 缺少trigger/pitfall"
        for q, ref, keys in dp.chain:
            assert q and ref and keys, f"{dp.id} 存在字段缺失的层"
        answers = [ref for _, ref, _ in dp.chain]
        scores = grade_chain(dp, answers)
        assert all(s == 1.0 for s in scores), f"{dp.id} 自洽性检查失败: {scores}"
    assert drill(BANK, cat=CAT, n=3) == BANK[:3]
    print(f"[PASS] dp_network_internals: {len(BANK)} 个 DeepPoint 通过自检")


if __name__ == "__main__":
    _self_test()
