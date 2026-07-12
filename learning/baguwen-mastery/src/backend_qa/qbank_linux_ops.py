"""Linux 与运维基础 八股问答库（约 13 题）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_common import QA, categories, grade, quiz  # noqa: E402

CAT = "Linux与运维基础"

BANK: list[QA] = [
    QA(
        id="be-linux-01", cat=CAT,
        q="ps 命令的作用是什么？和 top 有什么区别？",
        a="ps（process status）用来查看当前系统进程的一次性快照，比如进程号 PID、状态、占用的 CPU/内存等，"
          "常见用法如 'ps aux' 或 'ps -ef' 列出全部进程；它只输出一次结果，不会自动刷新。"
          "top 则是动态、持续刷新的进程监控工具，默认每隔几秒刷新一次，实时展示系统整体负载和 CPU/内存使用率，"
          "并按 CPU 或内存占用排序展示进程列表，适合持续观察资源占用的变化；"
          "ps 更适合在脚本里一次性抓取、配合 grep/awk 做过滤处理。",
        keys=("ps", "快照", "top", "实时"),
        follow_ups=("怎么用 ps 配合 grep 找到某个进程的 PID？", "top 里 load average 三个数字分别代表什么？"),
    ),
    QA(
        id="be-linux-02", cat=CAT,
        q="top 命令能看到哪些关键信息？怎么用它初步判断系统负载来源？",
        a="top 打开后顶部展示系统整体信息：uptime、load average（1/5/15 分钟平均负载）、任务总数与各状态任务数、"
          "CPU 各类型占用（user/system/idle/iowait 等）、内存和交换分区使用情况；下方按默认排序"
          "（通常是 CPU 占用）列出进程列表，包含 PID、用户、CPU%、内存%、进程状态、命令名等信息。"
          "初步判断负载来源时，可以看 load average 是否明显高于 CPU 核数，再看 CPU 里是 user 高"
          "（应用计算密集）还是 iowait 高（在等磁盘/网络 IO），最后结合进程列表里 CPU% 或内存% 最高的几个进程定位到具体进程。",
        keys=("load average", "iowait", "核数", "进程列表"),
        follow_ups=("iowait 占比很高说明什么问题？", "load average 超过 CPU 核数一定代表系统过载吗？"),
    ),
    QA(
        id="be-linux-03", cat=CAT,
        q="grep 命令的核心作用是什么？",
        a="grep（global regular expression print）用来在文本（文件或标准输入）里按行搜索匹配某个模式"
          "（可以是普通字符串，也可以是正则表达式）的内容，并把匹配到的行打印出来；常配合管道使用，"
          "比如从日志里过滤出包含某个关键字（如 'ERROR'）的行，是命令行排查问题时最常用的文本过滤工具之一。",
        keys=("grep", "正则表达式", "过滤", "管道"),
        follow_ups=("grep 想同时看到匹配行的上下文怎么办？", "grep、egrep、fgrep 有什么区别？"),
    ),
    QA(
        id="be-linux-04", cat=CAT,
        q="awk 的核心定位是什么？和 grep 相比它多做了什么？",
        a="awk 是一个以'行'为处理单位、以空白符（或指定分隔符）自动切分'字段'（列，$1、$2……）的文本处理语言，"
          "不仅能像 grep 一样按模式过滤行，还能对切分出来的字段做提取、拼接、算术运算、条件判断、累加统计等，"
          "比如统计某一列数值的总和或平均值，或者只打印某几列。grep 只做'过滤'，"
          "awk 在过滤的基础上还能做'按字段加工和统计'，因此常用来处理结构化的日志或表格文本，"
          "比如提取某一列 IP、统计某个字段出现的次数。",
        keys=("awk", "字段", "过滤", "统计"),
        follow_ups=("awk 里 $0、$1、NF、NR 分别代表什么？", "怎么用 awk 统计一个日志文件里某个字段的出现次数？"),
    ),
    QA(
        id="be-linux-05", cat=CAT,
        q="sed 的核心作用是什么？",
        a="sed（stream editor，流编辑器）用来对文本做非交互式的批量编辑，比如按行替换、删除、插入内容，"
          "逐行读取输入、对每一行按指定规则做变换后输出，默认不会修改原文件（除非显式加上 -i 参数原地编辑）。"
          "最常见的用法是批量替换文本里的某个字符串，或者按行号/模式删除特定行，"
          "适合在脚本里自动化处理配置文件、日志文件等文本内容。",
        keys=("sed", "流编辑器", "替换", "批量"),
        follow_ups=("sed 怎么做到只替换每行第一次出现的匹配，而不是全部替换？", "sed -i 直接原地修改文件有什么风险？"),
    ),
    QA(
        id="be-linux-06", cat=CAT,
        q="kill 命令发送的信号里，SIGTERM、SIGKILL、SIGHUP 分别是什么含义？",
        a="kill 命令本质是给进程发送信号，而不是直接杀死进程，进程如何响应取决于信号类型以及进程自己有没有处理逻辑。"
          "SIGTERM（信号 15，kill 不加参数时的默认信号）是'礼貌地'请求进程终止，进程可以捕获这个信号，"
          "先做清理工作（关闭文件、释放资源、保存状态）再退出，也可以选择忽略它；"
          "SIGKILL（信号 9，即 kill -9）是强制终止信号，进程无法捕获也无法忽略，内核直接终止进程，"
          "不会给进程任何清理机会，可能导致资源没释放干净或者数据没保存完整；"
          "SIGHUP（信号 1）最初含义是终端断开连接时通知前台进程，很多常驻后台的守护进程把它复用为"
          "'重新加载配置文件但不重启进程'的信号，比如 nginx 收到 SIGHUP 会重新加载配置。",
        keys=("SIGTERM", "SIGKILL", "SIGHUP", "信号"),
        follow_ups=("正常应该先用哪个信号尝试终止进程，什么情况下才用 -9？", "nohup 命令和 SIGHUP 信号是什么关系？"),
    ),
    QA(
        id="be-linux-07", cat=CAT,
        q="nohup 命令的作用是什么？",
        a="nohup（no hang up）让一个命令在启动它的终端会话关闭后，忽略本该收到的 SIGHUP 信号从而继续在后台运行，"
          "不会因为退出 SSH 连接或关闭终端而被终止；常见用法是 'nohup ./run.sh &'，"
          "默认会把标准输出和标准错误重定向到当前目录的 nohup.out 文件里。"
          "它只解决'终端断开导致进程被终止'这一个问题，如果需要更完善的后台任务管理"
          "（比如统一日志、自动重启、开机自启），通常会用 systemd，或者 screen/tmux 会话来管理。",
        keys=("nohup", "SIGHUP", "后台运行", "systemd"),
        follow_ups=("nohup 和 disown 有什么区别？", "为什么用 systemd 管理服务比 nohup 更规范？"),
    ),
    QA(
        id="be-linux-08", cat=CAT,
        q="systemd 是什么？它相比传统的 init/SysV 脚本有什么优势？",
        a="systemd 是现代 Linux 发行版（如 CentOS 7+、Ubuntu 16.04+）默认使用的系统和服务管理器，"
          "作为系统启动的 1 号进程（PID 1），负责初始化系统、管理各个服务的启停。"
          "每个服务由一个 unit 文件（如 xxx.service）描述，声明启动命令、依赖关系、重启策略等，"
          "通过 systemctl 命令（start/stop/restart/status/enable）来管理。相比传统 SysV init 脚本按顺序串行启动服务，"
          "systemd 支持并行启动、按需启动以加快开机速度，还内置了服务异常退出自动重启、"
          "日志统一由 journald 收集（journalctl 查看）等能力，管理更规范、可观测性更好。",
        keys=("systemd", "unit", "systemctl", "PID 1"),
        follow_ups=("怎么用 systemctl 让一个服务开机自启？", "journalctl 相比普通日志文件有什么优势？"),
    ),
    QA(
        id="be-linux-09", cat=CAT,
        q="线上机器 CPU 占用突然飙高，排查思路是什么？",
        a="先用 top（或 htop）看整体 CPU 使用率构成（user/system/iowait 哪个高）和按 CPU% 排序的进程列表，"
          "定位到占用最高的进程 PID；如果是 Java 进程，进一步用 'top -Hp <PID>' 看这个进程内部哪些线程占用 CPU 最高，"
          "拿到线程 TID（十进制），转换成十六进制后去 jstack 打印出来的线程栈里搜索对应的 nid，"
          "定位到具体是哪段代码（比如死循环、频繁 GC）在吃 CPU；如果不是 Java 进程，可以用 perf top 或 "
          "strace -p <PID> 观察进程在执行什么系统调用或热点函数。定位到具体代码后再判断是算法效率问题"
          "（比如死循环、不合理的重试）、并发问题（比如锁竞争导致自旋）还是流量突增带来的正常高负载。",
        keys=("top", "jstack", "线程", "定位"),
        follow_ups=("为什么要把线程 TID 转成十六进制去 jstack 里找？", "CPU 占用高和 load average 高是一回事吗？"),
    ),
    QA(
        id="be-linux-10", cat=CAT,
        q="怀疑程序有内存泄漏，怎么定位？",
        a="先确认内存占用是不是持续、不可回收地增长，而不是正常的高水位使用——比如 Java 进程可以用 jstat -gcutil "
          "观察老年代占用是否在 Full GC 之后依然持续升高、降不下来，如果 Full GC 之后老年代仍然回收不下来，"
          "大概率存在泄漏；然后用 jmap 导出堆转储（heap dump）文件，用 MAT（Eclipse Memory Analyzer）等工具分析，"
          "找到占用内存最多的对象类型以及它们的 GC Roots 引用链，确认是哪些对象本该被回收却因为被意外持有引用"
          "（比如静态集合不断 add 却没有清理、监听器没有反注册、连接池没有归还连接）而无法回收。"
          "非 JVM 场景（比如 C/C++）常用 valgrind、pmap 观察进程虚拟内存映射的变化来辅助定位。",
        keys=("堆转储", "GC Roots", "老年代", "泄漏"),
        follow_ups=("常见的 Java 内存泄漏场景有哪些？", "Full GC 之后内存降不下来一定是内存泄漏吗？"),
    ),
    QA(
        id="be-linux-11", cat=CAT,
        q="磁盘 IO 占用高，排查思路是什么？",
        a="先用 iostat -x 看各个磁盘设备的 %util（设备繁忙程度）、await（IO 平均等待时间）、r/s、w/s（读写次数）等指标，"
          "确认磁盘本身是不是瓶颈；如果确认磁盘繁忙，再用 iotop 按进程维度看是哪个进程在大量读写磁盘；"
          "定位到进程后结合业务逻辑判断是不是不合理的使用方式，比如频繁的小文件随机写、没有做批量写入合并、"
          "日志级别开太低导致打印过多、数据库缺索引导致大量全表扫描落到磁盘 IO 上，"
          "或者内存不够导致频繁 swap（这种情况用 free 命令能看到 swap 使用量在持续上涨）。",
        keys=("iostat", "iotop", "await", "swap"),
        follow_ups=("iostat 里 %util 100% 就一定是瓶颈吗？", "swap 使用量高对系统性能有什么影响？"),
    ),
    QA(
        id="be-linux-12", cat=CAT,
        q="网络连接出问题时，netstat、telnet、curl 各自能帮你确认什么？",
        a="netstat（或更现代的 ss）用来查看本机的网络连接状态和监听端口，比如确认目标服务的端口是不是真的在本机处于 "
          "LISTEN 状态、有没有大量连接堆积在 CLOSE_WAIT 或 TIME_WAIT 状态；telnet <host> <port> 用来测试到目标主机"
          "某个端口的 TCP 连通性，能连上说明网络链路和端口都通，连不上能帮你区分是网络不通还是端口没监听；"
          "curl 则是应用层的测试工具，在 TCP 连通的基础上进一步验证 HTTP(S) 请求是否被正确处理，"
          "比如返回的状态码、响应体内容、各阶段耗时，排查的是应用层逻辑问题而不只是网络连通性。"
          "三者组合起来通常是先用 netstat 确认本机监听状态，再用 telnet 确认端口连通性，最后用 curl 确认应用层响应是否正常。",
        keys=("netstat", "telnet", "curl", "连通性"),
        follow_ups=("CLOSE_WAIT 大量堆积通常说明什么问题？", "curl 怎么单独看 DNS 解析、建连、首字节各花了多少时间？"),
    ),
    QA(
        id="be-linux-13", cat=CAT,
        q="线上出问题时，日志排查的基本思路是什么？",
        a="先明确时间范围（问题大概发生在什么时间点）和关键错误信号词（比如异常类名、错误码、traceID），"
          "用 grep 在对应时间段的日志文件里过滤出相关行；如果是分布式系统，需要靠请求链路里传递的 traceID "
          "把跨多个服务、多台机器的日志串联起来还原完整调用链，而不是只看单台机器的日志；"
          "找到第一处（最早）出现异常的位置往往比找到报错最多的地方更关键，因为后面的报错很可能只是上游异常引发的连锁反应；"
          "同时要结合日志的上下文（前后几行）以及该时间点的系统指标（CPU、内存、网络）交叉验证，"
          "避免只凭一条孤立的错误日志下结论。",
        keys=("traceID", "grep", "时间范围", "调用链"),
        follow_ups=("分布式链路追踪（如 Jaeger/Zipkin）解决了日志排查的什么痛点？", "为什么定位问题要优先看最早出现的异常而不是报错最多的？"),
    ),
]


def _self_test() -> None:
    assert 11 <= len(BANK) <= 15, len(BANK)
    assert categories(BANK) == [CAT]
    ids = [qa.id for qa in BANK]
    assert len(ids) == len(set(ids)), "存在重复 id"
    assert all(i.startswith("be-linux-") for i in ids), "id 前缀不一致"
    assert all(qa.follow_ups for qa in BANK), "存在缺失追问链的题"
    scores = [grade(qa.a, qa) for qa in BANK]
    avg = sum(scores) / len(scores)
    assert avg >= 0.6, f"自洽性过低: {avg:.2f}"
    assert quiz(BANK, cat=CAT) == BANK
    print(f"[PASS] qbank_linux_ops: {len(BANK)}题 + 自洽性 {avg:.0%}")


if __name__ == "__main__":
    _self_test()
