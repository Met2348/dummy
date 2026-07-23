# 07 · 网络配置(Networking)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一篇覆盖 10 个知识点:nmcli/nmtui、主机名管理、`/etc/hosts`、DNS 客户端配置、连通性排查工具、`ip` 命令族、网络服务管理、静态路由。**本文所有代码例子已在 Rocky Linux 10.2(WSL2)下实际跑通验证**。
>
> **本篇特别的安全纪律**:凡是涉及**修改** IP 地址/路由的知识点(第 2、10 节),一律在一个独立创建的 **dummy 虚拟网卡**上操作,不触碰真实的 `eth0` 连接——`eth0` 是这个 WSL 环境真正连外网、也是我撰写和验证本系列全部内容所依赖的连接,误操作断网会中断后续所有验证工作。每个知识点验证完都已现场确认 `eth0` 配置和外网连通性完全没有受到影响。
>
> **和前四条系列的差异声明**(详见 [00-roadmap.md](00-roadmap.md) 开头):本仓库没有 Linux 系统管理场景可挖,"真实场景例子"部分统一标注为典型运维/RHCSA 考试场景,不冒充仓库代码里挖出来的。

---

## 开始之前:IP 地址、子网、网关、端口、DNS 到底是什么(下面 10 节反复会用到,但没有一节专门讲,这里先补上)

这一篇是本系列第一次真正碰网络类知识点——前六类(工具/进程/存储/文件权限/软件部署/用户组)完全没有涉及"IP 地址""网关""端口"这些概念。下面把后面十节反复要用到的几个地基概念一次性讲清楚,免得看到 `192.168.1.100/24` 却不知道 `/24` 是什么、看到"网关"却不知道它具体解决什么问题。

**IP 地址是什么:** 每台联网设备的每个网络接口(网卡),要有一个 IP 地址才能在网络上被找到——IPv4 地址是一个 32 位数字,习惯写成 4 段十进制数、每段 0-255、用点分隔(比如 `192.168.1.100`),这就是 `nmcli`/`ip addr` 命令里反复出现的那串数字。

**子网 / CIDR 记法(`/24`)是什么:** 一个 IP 地址单独存在没有意义,必须搭配"这个地址所在的网络范围有多大"一起看——`/24` 这种写法(CIDR,Classless Inter-Domain Routing)表示"这个 IP 地址的前 24 个比特是**网络部分**,剩下 8 个比特是**主机部分**"。`/24` 换算成传统的子网掩码写法就是 `255.255.255.0`,这个网络里一共有 2^8=256 个地址(实际能分给主机用的是 254 个,一个是网络地址、一个是广播地址,两者都不能分给某台具体设备用)。这个数字为什么重要,直接决定了数据包接下来怎么走:

| 场景 | 目的地址和本机是否在同一个 `/24` 网络里 | 数据包怎么走 |
|---|---|---|
| 访问同一局域网内的另一台机器 | 是(比如本机 `192.168.1.100/24` 访问 `192.168.1.50`) | 直接用 ARP 协议问到对方的物理(MAC)地址,直接发送,不经过网关 |
| 访问外网服务器(比如访问一个网站) | 否(目的 IP 和本机不在同一个 `/24` 范围) | 发给下面说的**网关**,由网关负责继续转发,本机自己不知道、也没办法知道该往哪继续送 |

**网关是什么:** 网关(gateway)是本机发送"目的地不在本地网络范围内"的数据包时,必须先送到的那台路由设备(通常是一台路由器)——本机不知道怎么直接找到那个远方目标,但知道"网关知道",于是把包丢给网关,网关再继续往下转发。日常说的"能上内网但上不了外网",排查到最后经常就是"网关配错了/网关不通"。第 10 节讲的"路由表"/"静态路由",本质上就是"针对更多种目的地址范围,分别配置该送去哪个网关/哪个接口"这套规则的集合,不是一条孤立的知识点。

**端口是什么(本篇第 7 节 `ss`、[08 类](08-security-selinux-firewall.md) firewalld 都会用到):** 一台机器可能同时对外提供好几种网络服务(比如同时跑着 SSH 和一个网站),光有 IP 地址分不清"这个数据包是找 SSH 的还是找网站的"——**端口**是一个额外的 16 位数字(0-65535),挂在 IP 地址后面,精确指定"这个连接是找这台机器上的哪个具体服务"(SSH 默认用 22 号端口,网站的 HTTP 默认用 80 号端口)。端口号前面通常要说清楚是 **TCP** 还是 **UDP**——这是两种最常见的传输层协议:TCP 先"打电话"建立连接、保证数据不丢不乱序,更可靠但开销更大;UDP 不建立连接,发出去就不管了,更快但不保证送达。`ss -t`只看 TCP、`ss -u`只看 UDP,firewalld 的 service/port 规则同样要分清楚协议,不是随便哪个都能通用。

**DNS 是什么(第 5-6 节会用到):** 人记 `www.example.com` 这样的名字比记一串 IP 地址容易得多,但网络传输实际用的还是 IP 地址——**DNS(Domain Name System,域名系统)**就是负责把这种人类好记的名字,查询翻译成真正 IP 地址的系统,查询过程通常需要问一台专门的 **DNS 服务器**(可能在本地局域网内,也可能是公网上的,比如 `8.8.8.8`)。第 5 节的 `/etc/hosts` 是一种更原始、完全离线、只能查有限几条记录的"文件版 DNS";第 6 节的 `/etc/resolv.conf` 才是配置"该问哪台 DNS 服务器"的地方。

---

## 1. `nmcli` 查看网络状态(device/connection 概念)

**命令/配置:**
```bash
nmcli device status        # 列出所有网络设备及其当前状态
nmcli connection show        # 列出所有配置好的连接(connection profile)
nmcli device show DEVICE       # 查看某个设备的完整详情
nmcli connection show NAME       # 查看某个连接配置的完整详情
```

**一句话是什么:** NetworkManager 的核心抽象里,**device**(设备)是物理或虚拟的网卡本身(`eth0` 这块"网卡"),**connection**(连接)是一套预先配置好的连接参数(IP/网关/DNS 等);一个 device 可以关联多个 connection profile,但同一时刻只能有一个被真正激活——这个"设备 vs 配置"的分离设计,让"临时切换到另一套网络参数"变得只是切换 connection、不需要重新输入所有参数。

**为什么 RHCSA 真考 / 生产会用到:** `nmcli` 是 RHCSA 明确要求掌握的网络配置工具,是现代 RHEL 系统命令行管理网络的标准入口;排障"网络不通"的第一步几乎总是先用 `nmcli device status`/`nmcli connection show` 摸清楚当前状态,再决定下一步动作。

**从最容易犯错的做法讲起:** 混淆 device 和 connection 这两个概念,想要"切换网络配置"时去找"另一个设备",而不是"给同一个设备切换到另一个 connection profile"——同一块物理网卡上,完全可以配置多套 connection(比如"办公室静态 IP"和"家里 DHCP"两套),需要切换的是 connection,不是设备本身。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 排查服务器网络故障,第一步 `nmcli device status` 确认网卡本身是否被识别、状态是 connected 还是 disconnected;确认设备正常后 `nmcli connection show` 看当前激活的是哪个 profile、`nmcli connection show <name>` 深入看这个 profile 具体配置的 IP/网关/DNS 是否符合预期。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

device_output=$(nmcli device status 2>&1)
echo "$device_output" | grep -q "eth0" && echo "OK: nmcli device status 能看到eth0设备"

connection_output=$(nmcli connection show 2>&1)
echo "$connection_output" | grep -q "eth0" && echo "OK: nmcli connection show 能看到eth0连接"

# 查询eth0这个device当前关联的是哪个connection
active_conn=$(nmcli -t -f GENERAL.CONNECTION device show eth0 | cut -d: -f2)
assert_eq "$active_conn" "eth0"
```
本机实测:全部检查点输出 `OK`。

**常见坑:** `nmcli` 的输出在**终端交互**和**脚本管道**两种场景下格式不同——直接跑 `nmcli device status` 是给人看的对齐表格,写脚本解析输出时要加 `-t`(terse,简洁模式,冒号分隔的机器可读格式)和 `-f`(指定只要哪些字段),否则脚本很容易被表格的空格对齐问题坑到,解析出乱七八糟的结果。

---

## 2. `nmcli` 配置静态 IP

**命令/配置:**
```bash
nmcli connection add type ethernet ifname eth0 con-name my-static \
    ipv4.method manual ipv4.addresses 192.168.1.100/24 ipv4.gateway 192.168.1.1 ipv4.dns 8.8.8.8
nmcli connection up my-static     # 激活这个新配置
```

**一句话是什么:** `nmcli connection add` 创建一个新的连接配置(不会立即生效),`ipv4.method manual` 是"静态 IP"模式的关键参数(对应的 `auto` 是 DHCP 动态获取),配好后必须额外 `connection up` 才会真正应用到设备上——"创建配置"和"激活配置"是两个独立步骤。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"配置 IPv4/IPv6 网络设置";服务器场景几乎总是用静态 IP(而不是 DHCP 动态分配,避免 IP 变化导致服务不可达),这是最基础的服务器网络配置技能。

**插一句,dummy 网卡是什么(下面"从最容易犯错的做法讲起"和第 10 节都会用到,这里先解释清楚,不然会不知道突然冒出的这个叫"dummy"的网卡是什么来路):** `ip link add NAME type dummy` 创建的是内核提供的一种纯虚拟网络设备——它不对应任何真实网卡硬件。现场用 `ip link show` 确认过,新建出来的 dummy 接口带着 `NOARP` 标志,意味着它连"用 ARP 协议去问同网段其他地址的物理地址是谁"这件事都不会做(对照上面"网关"部分的表格——它连最基本的同网段直连都做不到,是一个彻底孤立的沙盒,不是"功能弱化版真网卡"),所以可以在它身上配置任意 IP、任意路由,这些配置只影响这一个孤立的虚拟设备本身,不会碰到真实网络,也不会和真实网卡 `eth0` 产生任何冲突。本篇后面凡是涉及"修改 IP/路由"这类有真实风险的操作(第 2、10 节),统一在这样一块 dummy 网卡上做实验——命令语法和在真实网卡上完全一样,只是背后这块"网卡"是假的,操作出错也只需要删掉重建,不会影响真实连接。

**从最容易犯错的做法讲起:** **本机验证过程中现场踩到、也是本节最值得记录的坑**:在一个用 `ip link add ... type dummy` 命令行手动创建的虚拟网卡上尝试配置静态 IP,直接套用 `nmcli connection add type ethernet ...` 会报错 `No suitable device found`——根因有两层:① NetworkManager 默认把这类"不是自己创建"的设备标记为 `unmanaged`(外部设备),必须先 `nmcli device set DEVICE managed yes` 显式接管;② 连接类型要和设备类型匹配,`type ethernet` 的 profile 找不到匹配的以太网设备时,会尝试往默认的 `eth0` 身上凑而报"接口名不匹配",dummy 类型设备必须用 `nmcli connection add type dummy ...`,类型不对,报错信息会让人摸不着头脑。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 新服务器初始化,`nmcli connection add type ethernet ifname eth0 con-name static-eth0 ipv4.method manual ipv4.addresses 10.0.0.50/24 ipv4.gateway 10.0.0.1` 配好静态 IP,`connection up` 激活;RHCSA 考试要求"配置指定的 IP 地址",这套命令是标准解法,不需要手改配置文件。

**可运行例子(在独立 dummy 虚拟网卡上操作,不影响真实 eth0):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

ip link add rhcsa07dummy type dummy
ip link set rhcsa07dummy up
nmcli device set rhcsa07dummy managed yes    # 先显式接管这个"外部创建"的设备
sleep 1

nmcli connection add type dummy ifname rhcsa07dummy con-name rhcsa07-static \
    ipv4.method manual ipv4.addresses 192.168.99.10/24 ipv4.gateway 192.168.99.1 >/dev/null 2>&1
assert_eq "$?" "0"

nmcli connection up rhcsa07-static >/dev/null 2>&1
assert_eq "$?" "0"
sleep 1
assert_eq "$(nmcli -t -f IP4.ADDRESS device show rhcsa07dummy | cut -d: -f2)" "192.168.99.10/24"
assert_eq "$(nmcli -t -f IP4.GATEWAY device show rhcsa07dummy | cut -d: -f2)" "192.168.99.1"

nmcli connection down rhcsa07-static >/dev/null 2>&1
nmcli connection delete rhcsa07-static >/dev/null 2>&1
ip link delete rhcsa07dummy 2>&1
```
本机实测:全部断言输出 `OK`;验证完毕后现场确认 `eth0` 的真实 IP(`172.30.207.100/20`)和外网连通性(`ping 8.8.8.8` 正常)完全没有受到任何影响。

**常见坑:** 见上方"从最容易犯错的做法"——`managed yes` 和"connection type 要匹配设备类型"这两点,是在真实 RHCSA 考试环境(标准的 `eth0`/`ens*` 网卡,不是手动创建的虚拟设备)下通常不会遇到的坑,只在"自己动手创建虚拟网卡做实验"这种场景下才会碰到,但排查这类报错信息本身,是理解 NetworkManager 设备管理机制的一次很好的实战练习。

---

## 3. `nmtui` 交互式网络配置

**命令/配置:**
```bash
nmtui              # 打开主菜单(编辑连接/激活连接/修改主机名)
nmtui edit CON_NAME    # 直接进入编辑指定连接的界面
nmtui connect            # 直接进入激活连接选择界面
```

**一句话是什么:** `nmtui` 是 `nmcli` 的文本图形界面(TUI, Text User Interface)版本,提供菜单式导航,不需要记住任何具体的命令参数,适合在 SSH 终端里手动配置网络、或者不熟悉 `nmcli` 精确命令语法时使用。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 考试环境允许使用 `nmtui`,对不熟练记忆 `nmcli` 参数的考生是一个可靠的备选方案;生产环境临时登录到一台不熟悉的服务器做网络调整,`nmtui` 的菜单式操作比现查 `nmcli` 参数文档更快。

**从最容易犯错的做法讲起:** **本节有一个必须诚实说明的限制**:`nmtui` 是纯交互式的全屏 TUI 程序,没有提供非交互/脚本化模式,无法像其他知识点那样写一段可以自动跑通、断言验证结果的 bash 脚本——这不是本篇"偷懒少写"的省略,而是这个工具的设计性质决定的,本节的可运行例子**只能验证工具本身存在、可以被启动**,真正的菜单操作需要读者自己在终端里手动尝试。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 通过 SSH 远程连接到一台新服务器,发现网络需要调整成静态 IP,但一时想不起 `nmcli` 精确参数,直接 `nmtui` 打开菜单,选"Edit a connection"、选中对应网卡、按提示填 IP/网关/DNS,保存退出,比现查文档更直接。

**可运行例子(如实说明:仅验证存在性,交互操作无法自动化验证):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

nmtui_present=0; command -v nmtui >/dev/null 2>&1 && nmtui_present=1
assert_eq "$nmtui_present" "1"
echo "本机确认:nmtui 属于独立的 NetworkManager-tui 包(不在 NetworkManager 主包里),精简镜像默认未装,需额外 dnf install"
```
本机实测:断言输出 `OK`(工具本身在额外 `dnf install NetworkManager-tui` 之后确认存在)。

**常见坑:** `nmtui` 属于**独立的软件包** `NetworkManager-tui`,不是 `NetworkManager` 主包自带的——本机实测确认这个精简 Rocky Linux 镜像默认没有预装它,遇到"明明装了 NetworkManager 却找不到 nmtui 命令"这类情况,先检查这个专属包是否装了,而不是怀疑 NetworkManager 本身有问题。

---

## 4. 主机名管理(`hostnamectl`)

**命令/配置:**
```bash
hostnamectl                       # 查看当前主机名及系统信息
hostnamectl --static                # 只输出静态主机名(适合脚本解析)
hostnamectl set-hostname NAME         # 设置新的主机名(立即生效且持久化)
hostname                                # 传统命令,查看当前会话的主机名
```

**一句话是什么:** `hostnamectl` 是 systemd 提供的现代主机名管理工具,能同时设置"静态"(持久化到 `/etc/hostname`)、"瞬态"(运行时,可能被 DHCP 等机制动态覆盖)、"美观"(供人阅读的展示名,可以含空格/特殊字符)三种主机名——日常最常用的是 `--static`,`set-hostname` 不加额外选项时会同时设置全部三种。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"配置主机名解析";主机名是系统身份标识的一部分,很多服务(证书 CN、集群节点识别)依赖准确的主机名配置,`hostnamectl` 是设置它的标准现代工具。

**从最容易犯错的做法讲起:** 直接手改 `/etc/hostname` 文件,改完却发现当前会话里 `hostname` 命令显示的还是旧值——手改文件只更新了"静态主机名"的持久化存储,不会让内核里当前生效的"瞬态主机名"跟着同步变化;用 `hostnamectl set-hostname` 命令而不是直接改文件,能保证持久化配置和运行时状态同时正确更新,不会出现"改了文件但没生效"的割裂状态。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 云主机初始化脚本按照约定的命名规范(比如 `web-prod-01`)设置主机名:`hostnamectl set-hostname web-prod-01`,后续所有依赖 `hostname`/`hostnamectl --static` 查询主机名的服务和脚本都能拿到一致、正确的值。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

original_hostname=$(hostnamectl --static)

hostnamectl set-hostname rhcsa07-test-host
assert_eq "$(hostnamectl --static)" "rhcsa07-test-host"
assert_eq "$(hostname)" "rhcsa07-test-host"    # 传统hostname命令也同步反映了新值,不是割裂状态

hostnamectl set-hostname "$original_hostname"    # 改回原值,不留副作用
assert_eq "$(hostnamectl --static)" "$original_hostname"
```
本机实测:全部断言输出 `OK`,验证了改名和精确恢复原主机名的完整闭环。

**常见坑:** 主机名规范上不建议包含下划线 `_`(虽然 Linux 本身不会阻止你这么设置)——这是因为主机名最终经常要嵌入到 DNS 记录、URL 里,而 DNS 标准(RFC 952/1123)不允许下划线,只能用连字符 `-`,提前用连字符命名能避免以后接入 DNS/证书系统时才发现命名不合规、需要返工重命名的麻烦。

---

## 5. `/etc/hosts` 本地解析

**命令/配置:**
```
# /etc/hosts 格式: IP地址 主机名 [别名...]
127.0.0.1   localhost
192.168.1.10  myserver.local  myserver
```
```bash
getent hosts HOSTNAME    # 查询解析结果(会综合/etc/hosts和DNS,是比直接grep文件更准确的查询方式)
```

**一句话是什么:** `/etc/hosts` 是最古老、最简单粗暴的主机名解析方式——一个纯文本文件,直接把"主机名"和"IP 地址"写死映射在一起,查询时**优先于** DNS 被检查(这个优先顺序由 `/etc/nsswitch.conf` 控制),适合少量、固定、不希望依赖外部 DNS 服务器的场景。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 考试环境经常预置一些通过 `/etc/hosts` 解析的内部主机名(没有真实 DNS 服务器);排障"能 ping 通 IP 但 ping 不通主机名"这类问题,`/etc/hosts` 是要检查的第一个地方。

**从最容易犯错的做法讲起:** 需要验证"主机名解析结果对不对"时,直接 `grep hostname /etc/hosts` 只看这一个文件——但实际生效的解析结果还受 DNS、`/etc/nsswitch.conf` 里配置的查询顺序等因素影响,只看 `/etc/hosts` 文件内容不代表这就是系统实际会用到的解析结果(比如如果 DNS 查询顺序在 hosts 之前,或者 hosts 里的条目被格式错误注释掉了);`getent hosts` 才是查询"系统实际会解析出什么"的准确方式,不是直接翻文件。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 测试环境没有内部 DNS 服务器,几台服务器之间需要通过主机名互相访问,在每台机器的 `/etc/hosts` 里手工维护一份"主机名-IP"映射表;RHCSA 考试如果要求"配置本地主机名解析",直接编辑这个文件加一行是标准做法。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

cp /etc/hosts /tmp/rhcsa07_hosts_backup
echo "192.168.100.50 rhcsa07-test-alias.local" >> /etc/hosts

resolved_ip=$(getent hosts rhcsa07-test-alias.local | awk '{print $1}')
assert_eq "$resolved_ip" "192.168.100.50"    # getent确认这个新加的映射真实生效了

cp /tmp/rhcsa07_hosts_backup /etc/hosts    # 恢复原状,不留痕迹
rm -f /tmp/rhcsa07_hosts_backup
```
本机实测:断言输出 `OK`。

**常见坑:** `/etc/hosts` 里同一个 IP 出现多次或者格式写错(比如漏掉主机名和 IP 之间的空格)不会有任何报错提示,系统只是安静地解析失败或者解析出意料之外的结果——修改这个文件后,养成用 `getent hosts` 现场验证一下的习惯,不要假设"文件保存了就一定生效了"。

---

## 6. DNS 客户端配置(`/etc/resolv.conf`,NetworkManager 接管)

**命令/配置:**
```bash
cat /etc/resolv.conf         # 查看当前DNS服务器配置
nmcli connection modify CON_NAME ipv4.dns "8.8.8.8 1.1.1.1"    # 通过NetworkManager配置DNS(推荐方式)
```

**一句话是什么:** `/etc/resolv.conf` 是传统上配置 DNS 服务器的文件,但**在 NetworkManager 接管网络的现代 RHEL 系统上,这个文件通常是自动生成的**——直接手改它,下次网络配置刷新(比如 NetworkManager 重启、连接重新激活)时改动会被覆盖,想要持久化的 DNS 配置,应该通过 `nmcli connection modify ... ipv4.dns` 配置到 connection profile 里,让 NetworkManager 在生成 `resolv.conf` 时自动带上。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"配置主机名解析"(涵盖 DNS 客户端配置);理解"这个文件是自动生成的,不要手改"是避免"配置了 DNS,重启网络后又变回默认值"这类困惑的关键。

**从最容易犯错的做法讲起:** 直接 `vi /etc/resolv.conf` 手改 DNS 服务器地址,过一段时间(或者网络服务重启后)发现配置"离奇消失"——**本机实测确认**这个文件头部通常带有"这是自动生成的文件,请勿手动编辑"这类提示注释,手改的内容随时可能被 NetworkManager 重新生成的内容覆盖;正确做法是把 DNS 配置写进对应的 connection profile(`nmcli connection modify`),让它成为"权威配置来源"的一部分,而不是直接改这个"派生产物"。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 需要长期使用企业内部 DNS 服务器:`nmcli connection modify eth0 ipv4.dns "10.0.0.53 10.0.0.54"` 把 DNS 配置写进 connection profile,`nmcli connection up eth0` 让它生效并重新生成 `resolv.conf`,这样即便网络服务重启,DNS 配置依然正确,不会丢失。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

resolv_content=$(cat /etc/resolv.conf 2>&1)
echo "$resolv_content" | grep -q "nameserver" && echo "OK: /etc/resolv.conf 确实有nameserver配置"

# 确认这是NetworkManager自动生成的文件(头部有对应提示),而不是需要手改的静态配置
resolv_header=$(head -3 /etc/resolv.conf)
echo "$resolv_header" | grep -qi "generated\|managed\|NetworkManager" && echo "OK: resolv.conf文件头部确实有'自动生成'的提示注释"
```
本机实测:两个检查点均输出 `OK`。

**常见坑:** 极少数场景(比如完全不用 NetworkManager、改用手动网络配置的特殊环境)`/etc/resolv.conf` 确实是需要手改的静态文件——排查"为什么改了 resolv.conf 没生效"之前,先确认这台机器到底是不是由 NetworkManager 管理网络(`systemctl is-active NetworkManager`,见第 9 节),管理方式不同,DNS 配置的"权威来源"完全不同,不能一概而论。

---

## 7. 网络连通性排查工具(`ping`/`ss`/`ip`)

**命令/配置:**
```bash
ping -c N HOST          # 发送N次ICMP请求测试连通性(-c避免无限发送)
ss -tlnp                  # 显示所有正在监听的TCP端口及对应进程(t=tcp l=listen n=数字端口 p=进程)
ss -tunp                    # 显示所有已建立的TCP/UDP连接(u=udp,其余字母含义同上一行)
ip addr / ip route            # 见第8节
```

**一句话是什么:** `ping` 测试"能不能到达"这个最基本的连通性问题——它靠发送 **ICMP**(Internet Control Message Protocol,一种专门用于网络诊断/状态通知的协议,不像 TCP/UDP 那样传输应用数据本身)echo request 报文、等对方回一个 echo reply 来判断对方是否可达,和"某个具体端口通不通"是完全独立的两件事(见下方"从最容易犯错的做法讲起");`ss`(socket statistics)是查看本机网络连接/监听状态的现代工具(传统的 `netstat` 在很多精简安装里已经不预装,`ss` 是官方推荐的替代品),排查"服务到底有没有监听在预期端口上"是它最高频的用途。

**为什么 RHCSA 真考 / 生产会用到:** 排障能力是 RHCSA 隐含贯穿全程的要求,"服务配置了但访问不了"这类问题的标准排查三部曲通常是:服务本身是否 active(`systemctl`)、端口有没有正确监听(`ss`)、防火墙有没有放行(08 类会讲)——`ping`/`ss` 是这条排查链路里网络层面的基本功。

**从最容易犯错的做法讲起:** 服务访问不通,只会用 `ping` 测试,却忽略了 `ping` 测的是 ICMP 协议这一层的连通性,和"某个具体 TCP 端口(比如 80/443)是否可达"是两回事——很多生产防火墙策略允许 ICMP(能 ping 通)却屏蔽了特定端口,`ping` 通不代表对应服务端口就一定可达,反过来某些环境禁 ping(ICMP 被过滤)但服务端口完全正常,`ping` 不通也不代表服务真的有问题,不能把 `ping` 当成万能的连通性判断依据。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 部署了一个 Web 服务但外部访问不了,先 `ss -tlnp | grep :80` 确认这个服务是不是真的在监听 80 端口(如果没监听,说明是服务本身没启动/配置错误,根本不是网络问题);确认监听正常后再排查防火墙(08 类)。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

assert_ok command -v ping
assert_ok command -v ss

ping_result=$(ping -c 2 -W 2 127.0.0.1 2>&1)
echo "$ping_result" | grep -q "0% packet loss" && echo "OK: ping本地回环地址成功,0丢包"

ss_output=$(ss -tlnp 2>&1)
echo "$ss_output" | grep -q "LISTEN" && echo "OK: ss -tlnp 能看到正在监听的TCP端口(比如sshd的22端口)"
```
本机实测:全部检查点输出 `OK`。

**常见坑:** `ss`(和更老的 `netstat`)显示的**进程信息**(`-p` 选项)通常需要 root 权限才能看全——非 root 用户执行同样的命令,能看到端口和连接状态,但"哪个进程在用这个端口"这一列可能是空的或者不完整,排障时如果发现进程信息缺失,先检查是不是权限不够,而不是怀疑工具本身出了问题。

---

## 8. `ip` 命令族(`ip addr`/`ip route` 替代 `ifconfig`/`route`)

**命令/配置:**
```bash
ip addr show [DEVICE]        # 查看IP地址配置(替代老旧的ifconfig)
ip link show                    # 查看网络接口本身的状态(up/down、MAC地址等,不含IP)
ip route show                     # 查看路由表(替代老旧的route命令)
ip neigh show                       # 查看ARP/邻居表(替代arp命令)
```

**一句话是什么:** `ip` 命令族(来自 `iproute2` 工具集)是现代 Linux 网络管理的统一入口,取代了一批各自独立的老命令(`ifconfig`/`route`/`arp`)——`ip` 按"对象.子命令"(`ip addr show`、`ip route add` 这种结构)组织,功能覆盖更全面,是当前及未来的标准。`ip neigh show` 里的 **ARP**(Address Resolution Protocol,地址解析协议)就是本篇开头"网关"部分提到的、同一个 `/24` 网段内两台机器互相找到对方物理(MAC)地址所用的协议——"邻居表"缓存的正是"这个 IP 对应的 MAC 地址是什么"这批查询结果,不用每次通信都重新问一遍。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求掌握 `ip` 命令族;`ifconfig`/`route` 这些老命令在很多现代最小化安装的 RHEL 系统上已经**默认不预装**(属于逐渐被淘汰的 `net-tools` 包),排障时找不到 `ifconfig` 命令是很常见的真实场景,必须熟悉 `ip` 这套新工具。

**从最容易犯错的做法讲起:** 凭旧知识/旧教程习惯性敲 `ifconfig` 查网卡信息,发现命令不存在就以为是系统坏了或者网络工具没装全——**本机实测确认**:这个精简镜像默认连 `ip` 命令本身所在的 `iproute` 包都没有预装(需要额外 `dnf install iproute` 才能用),更不用说更老旧的 `ifconfig`(`net-tools` 包);现代 RHEL 环境下,应该养成直接用 `ip` 命令族的习惯,不需要先找老命令碰壁再切换。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 排查网络问题的标准起手式:`ip addr show` 确认接口有没有 IP、`ip link show` 确认接口本身是不是 up 状态(有 IP 但接口 down 了也访问不了)、`ip route show` 确认默认网关配置对不对——三条命令搭配起来,能覆盖网络层面绝大多数基础排障需求。

**可运行例子(诚实说明:本机确认 `iproute` 包默认未安装,`ip` 命令本身需要额外安装):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

assert_ok command -v ip

ip_addr_output=$(ip addr show eth0 2>&1)
echo "$ip_addr_output" | grep -q "inet " && echo "OK: ip addr show 能看到接口的IP地址"

ip_route_output=$(ip route show 2>&1)
default_route_count=$(echo "$ip_route_output" | grep -c "^default")
assert_ok test "$default_route_count" -ge 1    # 确认有默认路由(网关)配置
```
本机实测:全部检查点输出 `OK`。

**常见坑:** `ip addr show DEVICE` 和 `ip -br addr show DEVICE`(加 `-br`,brief 简洁模式)输出格式差异很大——前者是详细多行格式,后者是一行一个接口的紧凑摘要,写脚本解析优先考虑 `-br` 模式或者配合 `-j`(JSON 输出,较新版本支持)减少手工解析文本格式出错的可能性,不要死磕正则表达式去解析详细模式的多行输出。

---

## 9. 网络服务的开机管理(NetworkManager vs `network.service`)

**命令/配置:**
```bash
systemctl status NetworkManager    # 现代RHEL的标准网络管理服务
systemctl status network             # 传统的、遗留的网络服务(是否存在因系统而异)
```

**一句话是什么:** 早期 RHEL(6/7 时代及之前)用传统的 `network.service` 脚本式地管理网络接口配置(读取 `/etc/sysconfig/network-scripts/ifcfg-*` 这类文件);现代 RHEL 已经**完全统一到 NetworkManager**,`nmcli`/`nmtui`/图形界面背后都是同一个 NetworkManager 服务在做实际工作,这是理解"为什么现在配网络都用 nmcli 而不是直接改 ifcfg 文件"这一历史沿革的关键背景。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 现行考纲完全基于 NetworkManager 体系出题,不会要求手工维护 `ifcfg-*` 这类传统配置文件的语法;理解这段历史沿革,能避免被网上大量还停留在老版本 RHEL 知识的过时教程带偏方向。

**从最容易犯错的做法讲起:** 在网上搜到的老教程里看到"编辑 `/etc/sysconfig/network-scripts/ifcfg-eth0` 文件配置静态 IP"这类内容,照搬到现代 RHEL 系统上操作——**本机实测确认**:现代 Rocky Linux 10 上传统的 `network.service` 单元根本不存在(`systemctl list-unit-files` 查不到这个单元),网络配置的唯一权威来源是 NetworkManager 管理的 connection profile,不是这些历史遗留的 ifcfg 文件;凡是涉及网络配置的操作,统一用 `nmcli`(第 1-2 节),不要被过时教程带回手改文件的老路子。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 排查"网络配置为什么没有按预期生效",第一步确认到底是哪个服务在管理网络(`systemctl is-active NetworkManager`),如果是 NetworkManager 管理,所有配置都应该通过 `nmcli` 进行,手改任何底层文件都可能被 NetworkManager 之后的行为覆盖或者产生冲突。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

assert_ok systemctl is-active NetworkManager
assert_ok systemctl is-enabled NetworkManager

# 如实探测:传统network.service在现代RHEL上是否还存在
legacy_network_present=0
systemctl list-unit-files 2>/dev/null | grep -q "^network.service" && legacy_network_present=1
assert_eq "$legacy_network_present" "0"    # 本机确认:这个传统单元已经不存在了

# NetworkManager管理的connection profile存放位置
assert_ok test -d /etc/NetworkManager/system-connections
```
本机实测:全部检查点输出 `OK`,现场确认了 `network.service` 这个历史遗留单元在 RHEL 10 上确实已经不存在。

**常见坑:** 千万不要凭旧版本 RHEL 的记忆去找/etc/sysconfig/network-scripts/ifcfg-\* 这类文件——这再次印证本系列反复强调的原则:环境会随版本演进,"我记得应该是这样"的旧知识需要现场核实,不能直接套用到当前版本上;RHCSA 既然已经切换到 RHEL 10 基准,网络配置的知识体系也要跟着更新,不能停留在更老版本的心智模型里。

---

## 10. 主机路由表查看与静态路由添加

**命令/配置:**
```bash
ip route show                                 # 查看当前路由表
ip route add NETWORK/PREFIX via GATEWAY dev DEVICE    # 添加一条静态路由
ip route del NETWORK/PREFIX via GATEWAY dev DEVICE      # 删除一条静态路由
nmcli connection modify CON_NAME +ipv4.routes "NETWORK/PREFIX GATEWAY"   # 持久化到connection profile
```

**一句话是什么:** 路由表决定"发往某个目标网段的数据包,应该从哪个接口、经过哪个网关出去"——大多数主机只需要一条"默认路由"(所有不匹配其他规则的流量都走这条,即默认网关)就够用,但当主机需要同时和多个不同网段直接通信、且不能都走默认网关时,就需要手工添加针对特定网段的静态路由。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 大纲涉及基础网络配置里理解路由的概念;多网卡服务器(比如一张网卡对内网、一张对外网)、需要访问特定内部网段(通过专用网关而非默认网关)的场景,都要用到静态路由配置。

**从最容易犯错的做法讲起:** 用 `ip route add` 添加的路由是**临时的**,不会持久化——重启后这条手工加的路由会消失,这和本篇/其他章节反复出现的"临时 vs 永久"设计模式(手动 `mount` vs `/etc/fstab`、`sysctl -w` vs 配置文件)完全一致;需要长期生效的路由,要通过 `nmcli connection modify CON_NAME +ipv4.routes` 写进 connection profile,让 NetworkManager 在每次激活这个连接时自动应用。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 服务器需要直接访问一个特定的内部管理网段(`10.99.0.0/24`),这个网段的流量应该走专用的管理网关而不是默认的业务网关,`ip route add 10.99.0.0/24 via 10.0.0.254 dev eth1` 添加这条专用路由,确认工作正常后用 `nmcli connection modify` 持久化,避免重启丢失。

**可运行例子(在独立 dummy 虚拟网卡上操作,不影响真实路由表):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

ip link add rhcsa07dummy type dummy
ip link set rhcsa07dummy up
nmcli device set rhcsa07dummy managed yes
sleep 1
nmcli connection add type dummy ifname rhcsa07dummy con-name rhcsa07-static \
    ipv4.method manual ipv4.addresses 192.168.99.10/24 ipv4.gateway 192.168.99.1 >/dev/null 2>&1
nmcli connection up rhcsa07-static >/dev/null 2>&1
sleep 1

before_route_count=$(ip route show | wc -l)
ip route add 10.99.0.0/24 via 192.168.99.1 dev rhcsa07dummy 2>&1
after_route_count=$(ip route show | wc -l)
assert_ok test "$after_route_count" -gt "$before_route_count"
assert_eq "$(ip route show | grep -c '10.99.0.0/24')" "1"

ip route del 10.99.0.0/24 via 192.168.99.1 dev rhcsa07dummy 2>&1
assert_eq "$(ip route show | grep -c '10.99.0.0/24')" "0"    # 删除确实生效

nmcli connection down rhcsa07-static >/dev/null 2>&1
nmcli connection delete rhcsa07-static >/dev/null 2>&1
ip link delete rhcsa07dummy 2>&1
```
本机实测:全部断言输出 `OK`,操作全程在隔离的 dummy 网卡上进行,验证完毕后现场确认真实路由表(含默认路由)完全没有受到影响。

**常见坑:** 添加静态路由时如果指定的网关地址本身**不在**这条路由所绑定网卡的直连网段内,`ip route add` 会直接报错拒绝(网关必须是"下一跳",必须是当前网卡能直接二层可达的地址,不能是随便一个 IP)——这个报错本身是一层合理性校验保护,遇到"添加路由被拒绝"先检查网关地址和网卡所在网段是否匹配,而不是怀疑命令语法写错了。

---

*本篇完成:2026-07-11,10 个知识点。验证环境:Rocky Linux 10.2(WSL2)。全部代码块真实跑通验证,含多处现场发现的真实细节:`iproute`/`NetworkManager-tui` 默认未预装、NetworkManager 默认不接管"外部创建"的设备(需要 `nmcli device set managed yes`)、connection type 需要和设备类型精确匹配、传统 `network.service` 在 RHEL 10 上已不存在。涉及修改 IP/路由的知识点全部在独立 dummy 虚拟网卡上操作,验证过程全程确认未影响真实 `eth0` 网络连接。*
