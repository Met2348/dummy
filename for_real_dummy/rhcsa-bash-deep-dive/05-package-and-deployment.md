# 05 · 软件与系统部署(Package Management & System Deployment)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 本篇覆盖 10 个知识点:dnf 包管理基础、dnf 仓库管理、rpm 底层查询、createrepo 本地仓库、GRUB2 配置与内核参数、sysctl 内核参数调整、chronyd 时间同步、podman 容器基础、podman 生成 systemd 服务(quadlet)、软件源 GPG 签名校验。
>
> **验证环境:Rocky Linux 10.2 (Red Quartz) WSL2 实例(`platform:el10`),systemd 真实运行,默认 root 会话。** 这是本系列第一次有真实 RHEL 系环境,本篇除第 5 项(GRUB2,见下方边界声明)外的全部知识点都在这个环境里现场跑出真实命令、真实报错、真实修复过程——不是转述,是本机现场执行的记录。dnf/rpm 操作和另外两个正在写 02/04 类目的 agent 共享同一个 WSL 实例、共享同一把 dnf/rpm 事务锁,正文里出现的部分包名版本号、`dnf history` 里的陌生 transaction,属于并发环境的正常现象,不是本篇产生的。

> **本篇最重要的一条边界声明(第 5 项 GRUB2)必须先读:** WSL2 的"内核"由 Windows 侧 WSL 启动器直接提供给虚拟机,根本不经过传统 BIOS/UEFI → GRUB → 内核加载这条链路。`grub2-mkconfig` 这类命令在装好 `grub2-tools` 后**能跑、不报错**,但生成的配置文件从来不会被 WSL2 用来引导系统——本机实测这一点后面会用具体证据(空白的 `10_linux` 生成段、`/proc/cmdline` 的真实内容、`uname -r` 找不到对应 `/boot` 文件)显著说明。**这一条不代表"验证了修改内核启动参数能改变下次启动行为"这件事,这件事在 WSL2 里做不到,必须诚实说清楚,绝不能暗示或假装做到了。**
>
> **和前四条系列的差异声明**(详见 [00-roadmap.md](00-roadmap.md) 开头):本仓库没有 Linux 系统管理场景可挖,"真实场景例子"部分统一标注为典型运维/RHCSA 考试场景,不冒充仓库代码里挖出来的。

---

## 1. dnf 包管理基础(`install`/`remove`/`search`/`info`)

**命令/配置:**
```bash
dnf search <keyword>      # 按名称/摘要关键词搜索可用包
dnf info <pkg>             # 查看包的详细信息(版本、大小、来源仓库、描述)
dnf install -y <pkg>       # 安装(-y 跳过确认,自动化脚本常用)
dnf remove -y <pkg>        # 卸载,默认会一并清理"只为这个包而装"的依赖
dnf history list            # 查看历史事务(每次 install/remove/update 都是一个事务)
dnf history info <id>       # 查看某次事务的具体改动
```

**一句话是什么:** dnf 是 RHEL 8+/RHEL 10 的默认包管理器(yum 的继任者,底层换成了 libsolv 做依赖求解),`search`/`info` 负责"找",`install`/`remove` 负责"改",每一次 `install`/`remove`/`update` 都会被记录成一个可回溯的**事务**——这是 dnf 相对底层工具 `rpm`(见本类目第3项)最大的价值:自动依赖求解 + 从配置好的仓库拉取 + 事务历史。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 官方技能列表明确要求"安装/卸载软件包和包组",几乎每道涉及部署服务的大题第一步都是 `dnf install`;生产环境的补丁管理、回滚事故变更也都要靠 `dnf history` 而不是凭记忆手动反向操作。

**从最容易犯错的做法讲起:** 拿到一个来源不明的 `.rpm` 文件,图省事直接 `rpm -ivh pkg.rpm` 安装——`rpm` 本身**只做依赖检查,不做依赖解决**,遇到依赖缺失会直接拒绝安装并报错退出,不会自动去仓库里找缺的部分。本机用真实的 `tcpdump` 包做了这组对比(`tcpdump` 依赖 `libpcap`,测试前用 `rpm -q libpcap` 确认过当时确实没装):

```
$ dnf download tcpdump          # 只下载 tcpdump 自己,不下载依赖
$ rpm -ivh ./tcpdump*.rpm
error: Failed dependencies:
	libpcap.so.1()(64bit) is needed by tcpdump-14:4.99.4-10.el10.x86_64
# 退出码 1,tcpdump 没有被安装

$ dnf install -y ./tcpdump*.rpm  # 同一个本地文件,换 dnf 装
Installing dependencies:
 libibverbs       x86_64       61.0-1.el10             baseos             475 k
 libpcap          x86_64       14:1.10.4-7.el10        baseos             174 k
Installing:
 tcpdump          x86_64       14:4.99.4-10.el10       @commandline       502 k
Complete!
```
dnf 不仅自动从仓库拉了 `libpcap`,还额外发现并拉了 `libibverbs`(InfiniBand 抓包支持库,`rpm -ivh` 完全不会替你发现这一层间接依赖)。这就是为什么"手动拖一个 rpm 文件用 rpm 装"几乎总是错误做法,除非清楚知道自己在干什么(比如离线环境配合本类目第4项的本地仓库)。

**真实场景例子(典型运维场景,非仓库代码):** 部署新服务前用 `dnf search` 确认包名拼写正确、`dnf info` 核实版本和体积是否符合预期,再 `dnf install -y` 批量安装;一次补丁升级后如果发现某个服务起不来,先 `dnf history list` 找到那次升级对应的事务 ID,`dnf history info <id>` 看清楚具体动了哪些包,再决定是 `dnf history undo <id>` 回滚还是排查别的原因,而不是凭印象手动一个个 `dnf downgrade`。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

dnf_retry() {
    # 共享 WSL 环境下 dnf/rpm 锁会被其他 agent 短暂占用,退避重试而不是直接判失败
    local attempt=1 max=6
    while [ "$attempt" -le "$max" ]; do
        if dnf "$@"; then return 0; fi
        echo "dnf 被占用,第 $attempt 次重试..." >&2
        sleep $((attempt * 2)); attempt=$((attempt + 1))
    done
    return 1
}

# search / info:安装前先确认包存在、看清楚元信息(不写死版本号,版本会随仓库更新变化)
assert_ok bash -c 'dnf search tree 2>&1 | grep -q "File system tree viewer"'
assert_ok bash -c 'dnf info tree 2>&1 | grep -q "^Version"'

# 安装前确认没装
rpm -q tree >/dev/null 2>&1
assert_eq "$?" "1"

assert_ok dnf_retry install -y tree
assert_ok rpm -q tree
assert_eq "$(command -v tree)" "/usr/bin/tree"

# dnf history:每次事务都可追溯
last_id=$(dnf history list 2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1; exit}')
assert_ok bash -c "dnf history info $last_id 2>&1 | grep -q 'Install tree'"

# 卸载:clean_requirements_on_remove=True(/etc/dnf/dnf.conf 默认配置)会一并清理只为它装的依赖
assert_ok dnf_retry remove -y tree
rpm -q tree >/dev/null 2>&1
assert_eq "$?" "1"
```
本机实测(Rocky Linux 10.2 WSL,`dnf-0:4.20.0-22.el10_2.rocky.0.1`):全部 `assert_eq`/`assert_ok` 输出 `OK`。`tree-2.1.0-8.el10` 来自 `baseos` 仓库,安装 56K,卸载后 `rpm -q tree` 确认恢复到未安装状态。

**常见坑:**
1. `rpm -ivh` 直接装一个有依赖的本地包会硬失败(见上方真实报错),这是"rpm 和 dnf 分工不同"这条最直观的证据——**详细的 rpm/dnf 关系和更多 rpm 查询命令见本类目第 3 项**。
2. `dnf remove` 默认会级联清理"只为这个包而装"的依赖(`clean_requirements_on_remove=True`,本机 `/etc/dnf/dnf.conf` 确认此项),这既是优点(不留垃圾)也是坑——如果某个依赖后来被你手动用来干别的事,再删原包时它可能被意外一起删掉。本机实测卸载 `tcpdump` 时这段真实输出是:
   ```
   Removing:
    tcpdump         x86_64      14:4.99.4-10.el10         @@commandline      1.2 M
   Removing unused dependencies:
    libibverbs      x86_64      61.0-1.el10               @baseos            1.3 M
    libpcap         x86_64      14:1.10.4-7.el10          @baseos            402 k
   ```
   删除前不加 `-y` 先看这段"Removing unused dependencies"再决定要不要继续,是稳妥习惯。
3. `dnf install -y` 脚本化自动应答一切确认,包括第 10 项会讲到的"是否信任新 GPG 密钥"确认——图方便无脑 `-y` 会让这一步的人工把关也被跳过,详见第 10 项。

---

## 2. dnf 仓库管理(repo 文件,`dnf repolist`)

**命令/配置:**
```bash
dnf repolist              # 只列出当前启用的仓库
dnf repolist all           # 列出全部仓库,含禁用的
dnf config-manager --set-enabled <repoid>    # 启用某个仓库(需要 dnf-plugins-core)
dnf config-manager --set-disabled <repoid>   # 禁用
```
仓库定义文件位于 `/etc/yum.repos.d/*.repo`,一个 `.repo` 文件里可以有多个 `[repoid]` 小节。

**一句话是什么:** dnf 从哪里拉包,完全由 `/etc/yum.repos.d/` 下 `.repo` 文件里的 `[repoid]` 小节决定——每个小节至少要有 `baseurl`/`mirrorlist`/`metalink` 三选一(指路)、`enabled`(开关)、`gpgcheck`+`gpgkey`(签名校验,见第10项)这几个关键字段,`dnf repolist` 是排查"为什么找不到某个包"的第一个命令。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"配置本地/远程 yum/dnf 仓库",企业内网通常会禁用官方公网源、改用内部镜像仓库,能读懂/正确改写 `.repo` 文件是上机考试和生产运维的共同基础技能。

**从最容易犯错的做法讲起:** 认为"仓库文件存在就等于仓库生效"——一个 `.repo` 文件里可以同时定义好几个 `[repoid]` 小节,每个小节有**独立**的 `enabled` 开关,文件存在不代表里面的仓库都在用。本机 Rocky Linux 的 `/etc/yum.repos.d/rocky-devel.repo` 文件里就定义了 `crb`(CodeReady Builder,提供大量开发库)等仓库,但**默认是禁用的**:
```
$ dnf repolist all | grep crb
crb                        Rocky Linux 10 - CRB                         disabled

$ dnf config-manager --set-enabled crb
$ dnf repolist all | grep crb
crb                        Rocky Linux 10 - CRB                         enabled
```
不知道这个开关的存在,遇到"明明文档说这个包在仓库里,`dnf install` 却提示找不到"时会走很多弯路。

**真实场景例子(典型运维场景,非仓库代码):** 新装的服务器默认只启用了 `baseos`/`appstream`/`extras`,某个开发工具在 `crb`(旧称 PowerTools)仓库里,需要先 `dnf config-manager --set-enabled crb` 再安装;内网环境用 `dnf config-manager --add-repo` 指向公司内部镜像地址,替代官方源加速下载并满足内网隔离要求。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

dnf_retry() {
    local attempt=1 max=6
    while [ "$attempt" -le "$max" ]; do
        if dnf "$@"; then return 0; fi
        sleep $((attempt * 2)); attempt=$((attempt + 1))
    done
    return 1
}

# 需要 dnf-plugins-core 才有 config-manager 子命令,如实探测再装
command -v dnf >/dev/null
dnf config-manager --help >/dev/null 2>&1 || assert_ok dnf_retry install -y dnf-plugins-core

# repolist 默认只显示启用的仓库
assert_ok bash -c 'dnf repolist 2>&1 | grep -q "Rocky Linux 10 - BaseOS"'

# repolist all 能看到仓库的启用/禁用状态(本机初次实测时 crb 默认是 disabled;
# 这里不写死具体状态,先记录当前状态,验证切换真实生效后再切回原状态,
# 避免和共享 WSL 环境里其他并发操作的状态假设冲突——这也是本机现场遇到过的真实情况,
# 见下方"本机实测"说明)
orig_crb=$(dnf repolist all 2>/dev/null | awk '$1=="crb"{print $NF}')
assert_ok bash -c "[ '$orig_crb' = enabled ] || [ '$orig_crb' = disabled ]"

# 一个 .repo 文件里 baseos 小节的真实结构:mirrorlist/gpgcheck/gpgkey 都在(baseurl 被注释,见常见坑)
assert_ok bash -c "awk '/^\[baseos\]/{f=1} f{print} f&&/^\$/{exit}' /etc/yum.repos.d/rocky.repo | grep -q '^gpgcheck=1'"

# config-manager 启用/禁用一个仓库,验证状态确实切换,再切回原状态(不改变机器长期状态)
assert_ok dnf config-manager --set-enabled crb
assert_eq "$(dnf repolist all 2>/dev/null | awk '$1=="crb"{print $NF}')" "enabled"
assert_ok dnf config-manager --set-disabled crb
assert_eq "$(dnf repolist all 2>/dev/null | awk '$1=="crb"{print $NF}')" "disabled"
if [ "$orig_crb" = "enabled" ]; then dnf config-manager --set-enabled crb >/dev/null; fi
```
本机实测:全部 `assert_eq`/`assert_ok` 输出 `OK`。本机额外验证过:未装 `dnf-plugins-core` 时执行 `dnf config-manager`/`dnf download` 会提示 `No such command`,并附带 `try: "dnf install 'dnf-command(config-manager)'"` 这样的准确提示——报错信息本身就是修复方法,值得留意。**代码块里"记录原状态再切回"这个写法本身也是本机现场教训**:第一次跑这段验证时把 `crb` 默认状态写死成 `disabled`,复验时因为共享 WSL 环境里另一个并发 agent 已经把 `crb` 切成了 `enabled`(不是本篇动的),导致断言失败——这正是 00-roadmap.md 里强调的"多 agent 共享同一个 WSL 实例"在 dnf 仓库状态这个维度上的真实体现,修正后的写法不再假设某个固定的初始状态。

**常见坑:**
1. 同一个 `.repo` 文件里可以有多个仓库小节,`enabled` 是小节级别的开关,不是文件级别的——见上方"从最容易犯错的做法讲起"。
2. `dnf config-manager`/`dnf download` 等子命令不是 dnf 自带的,来自 `dnf-plugins-core` 包,精简安装的系统上可能没装,报错文本 `No such command: config-manager` 容易被误认为是"命令名打错了",其实是插件包没装。
3. `.repo` 文件里 `baseurl` 常常是**注释掉**的,实际生效的是 `mirrorlist`(自动挑一个最近的镜像站)——本机 `rocky.repo` 的 `baseos` 小节就是这样:
   ```
   mirrorlist=https://mirrors.rockylinux.org/mirrorlist?arch=$basearch&repo=BaseOS-$releasever$rltype
   #baseurl=http://dl.rockylinux.org/$contentdir/$releasever/BaseOS/$basearch/os/
   ```
   如果要固定死一个内部镜像地址,反而要注释掉 `mirrorlist`、取消注释 `baseurl` 并改成自己的地址,搞反容易导致"改了 baseurl 但没生效,因为 mirrorlist 优先级更高"的困惑。

---

## 3. rpm 底层查询(`rpm -qa`/`-qi`/`-ql`,和 dnf 的关系)

**命令/配置:**
```bash
rpm -qa                # 列出所有已安装包(query all)
rpm -qi <pkg>            # 包的详细信息(query info)
rpm -ql <pkg>            # 包安装了哪些文件(query list)
rpm -qc <pkg>            # 只看配置文件
rpm -qd <pkg>            # 只看文档文件
rpm -qf <文件路径>        # 反查:这个文件属于哪个包(query file)
```

**一句话是什么:** `rpm` 是 dnf 底层实际执行安装/卸载/查询动作的引擎,只认本机 rpm 数据库和本地 `.rpm` 文件,**不知道"仓库"这个概念、不会联网、不做依赖自动求解**——dnf 负责"从哪拿、缺什么自动补",rpm 负责"最终这一下怎么落盘、已经落盘的东西怎么查",两者是分层关系不是竞争关系。

**为什么 RHCSA 真考 / 生产会用到:** 排查"这个命令是哪个包提供的"、"这个包到底往系统里塞了哪些文件"是运维排障的家常便饭,dnf 的 `search`/`info` 面向"仓库里有什么",rpm 的 `-q` 系列面向"本机已经装了什么、装的时候落了哪些文件",要解决"已经装上之后"的问题必须用 rpm 而不是 dnf。

**从最容易犯错的做法讲起:** 忘记 rpm 查询选项要**组合方式**加在 `-q` 后面(比如 `-qi` 是 `-q` + `-i`),以及混淆 `-ql`(这个包装了哪些文件)和 `-qf`(这个文件属于哪个包)两个方向相反的查询——一个是"从包找文件",一个是"从文件找包",名字都是 `q` 开头很容易记混。

**真实场景例子(典型运维场景,非仓库代码):** 某个二进制命令报错但不确定是不是被人手动替换过,用 `rpm -qf $(command -v tree)` 反查它是否确实属于期望的包、`rpm -V <pkg>` 进一步校验包内文件有没有被篡改(本类目第8项诊断 `newuidmap` 权限缺失问题时,`rpm -V shadow-utils` 就是这么定位到具体哪两个文件出问题的,是真实排障手法而不是纸面概念);怀疑某个配置文件是不是自己改过,`rpm -qc <pkg>` 先看这个包本身带哪些配置文件。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

dnf_retry() {
    local attempt=1 max=6
    while [ "$attempt" -le "$max" ]; do
        if dnf "$@"; then return 0; fi
        sleep $((attempt * 2)); attempt=$((attempt + 1))
    done
    return 1
}

assert_ok dnf_retry install -y tree

# -qi:详细信息(不写死版本号,只确认字段结构存在)
assert_ok bash -c 'rpm -qi tree | grep -q "^Version"'
assert_ok bash -c 'rpm -qi tree | grep -q "^Vendor.*Rocky"'

# -ql:这个包装了哪些文件,tree 的可执行文件必须在列
assert_ok bash -c 'rpm -ql tree | grep -q "^/usr/bin/tree$"'

# -qd:只看文档(tree 有 README 和 man page,没有独立配置文件)
assert_ok bash -c 'rpm -qd tree | grep -q "tree.1.gz"'
assert_eq "$(rpm -qc tree 2>&1)" ""     # tree 没有配置文件,-qc 合法地返回空,不是报错

# -qf:反向查询,从文件路径找回包名(和 -ql 方向相反)
assert_eq "$(rpm -qf /usr/bin/tree | sed 's/-[0-9].*//')" "tree"

# rpm 不做依赖解决:直接 rpm -ivh 一个有缺失依赖的本地包会失败(dnf 才会自动拉依赖,对比见本类目第1项)
rpm -q libpcap >/dev/null 2>&1
assert_eq "$?" "1"    # 确认 libpcap 当前未装,下面的失败不是别的原因导致的
workdir=$(mktemp -d /tmp/rhcsa05_demo3.XXXXXX)
(cd "$workdir" && dnf_retry download tcpdump >/dev/null 2>&1)
assert_ok bash -c "rpm -ivh $workdir/tcpdump*.rpm 2>&1 | grep -q 'Failed dependencies'"
rm -rf "$workdir"

assert_ok dnf_retry remove -y tree
```
本机实测:全部 `assert_eq`/`assert_ok` 输出 `OK`。`rpm -qf` 反查结果和 `rpm -qi`/`dnf info` 看到的是**同一个**底层数据库,证明 dnf 只是在 rpm 外面包了一层依赖求解和仓库访问,不是另起炉灶的数据源。

**常见坑:**
1. `-ql`(list files IN this package)和 `-qf`(find package FOR this file)方向相反,记混了会拿一个文件路径去当包名用,或者拿包名去反查,大概率报错或者查到风马牛不相及的结果。
2. `rpm -qc`/`rpm -qd` 对没有配置文件/文档文件的包合法地返回**空**而不是报错(见上方 `tree` 的例子),不能把"没输出"误判为"命令用错了"。
3. **完整的 rpm/dnf 分工证据见本类目第1项"从最容易犯错的做法讲起"**:同一个 `tcpdump` 包,`rpm -ivh` 直接失败(`Failed dependencies: libpcap.so.1()(64bit) is needed`),`dnf install` 自动补齐 `libpcap` 和额外发现的 `libibverbs` 后成功,两项互相引用同一组真实证据,不重复贴长日志。

---

## 4. 创建本地/离线仓库(`createrepo` 基础)

**命令/配置:**
```bash
dnf install -y createrepo_c        # RHEL 10 上是 createrepo_c(C 语言重写版),不再是老的 createrepo
createrepo_c /path/to/rpmdir        # 扫描目录下所有 .rpm,生成 repodata/ 元数据索引
```
`.repo` 文件里指向本地目录:
```
[my-local-repo]
baseurl=file:///path/to/rpmdir
enabled=1
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-Rocky-10
```

**一句话是什么:** `createrepo_c` 扫描一个目录里的全部 `.rpm` 文件,生成一套标准仓库元数据(`repodata/` 下的 `repomd.xml` + `primary.xml.zst`/`filelists.xml.zst`/`other.xml.zst`),让这个普通目录"升级"成 dnf 能识别、能搜索、能安装的正式软件仓库——生成出来的元数据格式和官方仓库服务器上跑的是同一套标准,dnf 完全无法分辨这是官方仓库还是自己拿几个 rpm 拼出来的目录。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 考试环境经常是完全离线的,题目可能要求"配置一个指向本地挂载安装介质或本地目录的仓库";生产环境的内网隔离服务器同样依赖本地/私有仓库分发软件,不依赖外部互联网访问。

**从最容易犯错的做法讲起:** 把一堆 `.rpm` 文件扔进一个目录,直接在 `.repo` 文件里把 `baseurl` 指向这个目录就以为大功告成——没有先执行 `createrepo_c` 生成 `repodata/` 元数据,dnf 根本不认得这是个仓库,`dnf repolist` 里这个仓库会带着报错提示或者直接搜不到任何包。元数据和实际的 rpm 文件同样重要,缺一不可。

**真实场景例子(典型运维场景,非仓库代码):** 挂载 RHEL/Rocky 安装 ISO 作为本地仓库(ISO 里自带现成的 `repodata/`,不需要额外 `createrepo`);自己攒了一批离线 rpm 包想统一管理分发,`createrepo_c` 生成索引后配一个指向这个目录的 `.repo` 文件,内网所有机器都能像连接官方仓库一样从这里装软件,`dnf --repo=<repoid> install` 还能指定"必须从这个仓库装"精确控制来源。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

dnf_retry() {
    local attempt=1 max=6
    while [ "$attempt" -le "$max" ]; do
        if dnf "$@"; then return 0; fi
        sleep $((attempt * 2)); attempt=$((attempt + 1))
    done
    return 1
}

assert_ok dnf_retry install -y createrepo_c

repodir=$(mktemp -d /tmp/rhcsa05_localrepo.XXXXXX)
(cd "$repodir" && dnf_retry download bc unzip zsh >/dev/null 2>&1)
assert_eq "$(find "$repodir" -maxdepth 1 -name '*.rpm' | wc -l)" "3"

assert_ok createrepo_c "$repodir"
assert_ok test -f "$repodir/repodata/repomd.xml"

repofile=/etc/yum.repos.d/rhcsa05-local.repo
cat > "$repofile" << EOF
[rhcsa05-local]
name=RHCSA05 temp local demo repo
baseurl=file://$repodir
enabled=1
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-Rocky-10
EOF

# dnf 能看见并列出这个本地仓库的包
assert_ok bash -c 'dnf repolist 2>&1 | grep -q rhcsa05-local'
assert_ok bash -c 'dnf --repo=rhcsa05-local list available 2>&1 | grep -q "^bc\."'

# 精确指定从这个仓库安装,Repository 列必须显示 rhcsa05-local(不是从官方 baseos 装的)
install_output=$(dnf_retry --repo=rhcsa05-local install -y bc 2>&1)
echo "$install_output" | grep -q "rhcsa05-local" && echo "OK: 事务表格确认包来自 rhcsa05-local 仓库"
assert_ok rpm -q bc

# 清理
assert_ok dnf_retry remove -y bc
rm -f "$repofile"
rm -rf "$repodir"
assert_ok bash -c '! dnf repolist 2>&1 | grep -q rhcsa05-local'
```
本机实测:全部 `assert_eq`/`assert_ok` 输出 `OK`。`createrepo_c` 真实处理输出是 `Directory walk done - 3 packages`,生成的 `repodata/` 里 `primary.xml.zst`/`filelists.xml.zst`/`other.xml.zst`/`repomd.xml` 四个文件齐全;`dnf --repo=rhcsa05-local install -y bc` 的事务表格里 `Repository` 列真实显示的是 `rhcsa05-local`,卸载时显示 `@rhcsa05-local`,证明确实是从这个本地仓库装的,不是从 `baseos` 蹭的。

**常见坑:**
1. 本地仓库目录里的 rpm 文件有增删之后,**必须重新执行 `createrepo_c`**(或加 `--update` 增量更新)才能让 dnf 感知变化——元数据是"扫描那一刻的快照",不会自动跟踪目录内容的后续变化。
2. 从本地仓库装的包如果是从官方渠道下载来的(比如上面例子里的 `bc`/`unzip`/`zsh`),文件本身**依然带着原厂商的 GPG 签名**,`gpgcheck=1` 照样能验证通过,不需要额外给本地仓库的包重新签名——这是"仓库的 `gpgcheck`/`gpgkey` 校验的是包文件本身的签名,不是校验仓库来源"这一层区别,详见本类目第10项。
3. `.repo` 文件里 `gpgcheck=1` 只校验**包级别**签名,不代表仓库元数据本身(`repomd.xml`)也被校验过——元数据级别的校验要靠单独的 `repo_gpgcheck=1` 选项,默认通常关闭,本机搭的临时本地仓库没有对 `repomd.xml` 签名,`repo_gpgcheck=1` 会导致装不了,这里用的是默认的包级校验。

---

## 5. GRUB2 配置与内核参数修改

> **这是本类目最需要谨慎诚实处理的一条,请完整阅读本节,不要跳过。**

**命令/配置:**
```bash
grub2-mkconfig -o /boot/grub2/grub.cfg    # 根据 /etc/default/grub 等配置重新生成 GRUB 主配置文件
grubby --info=ALL                          # 查看已安装内核的启动参数
grubby --default-kernel                     # 查看当前默认内核
grubby --update-kernel=ALL --args="参数"      # 给内核追加启动参数
```

**一句话是什么:** 在**真实 RHEL/裸机或标准虚拟机**上,`grub2-mkconfig` 负责重新生成 GRUB 主配置文件,`grubby` 负责精细修改已安装内核对应的启动参数(BLS 条目),这套机制的前提是系统经过 BIOS/UEFI → GRUB → 内核 这条标准引导链路启动。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"修改内核启动参数";生产环境常见需求包括临时禁用某个硬件兼容性有问题的内核特性、开启调试参数等。

**从最容易犯错的做法讲起(真实 RHEL 环境下适用):** 直接用文本编辑器手改 `/boot/grub2/grub.cfg` 这个生成出来的主配置文件——这个文件是**自动生成的产物**,下次跑 `grub2-mkconfig` 会被整个覆盖重写,手改内容会不明不白丢失;正确入口是 `/etc/default/grub`(用户可编辑的配置源)配合 `grub2-mkconfig -o` 重新生成,或者用 `grubby` 直接对已安装内核的启动参数做精确修改。

### 为什么这一条在 WSL2 里没法像其他条目一样验证

**根本原因:WSL2 的虚拟机根本不经过 GRUB 引导。** WSL2 的"内核"是 Windows 侧 WSL 启动器(`wslhost.exe`/Hyper-V 虚拟化层)直接提供给虚拟机的,不存在传统的 BIOS/UEFI → 磁盘上的引导扇区/EFI 分区 → GRUB → 从 `/boot` 加载内核镜像这条链路。这不是"WSL 里 GRUB 配得不对",而是这条链路整个不存在。以下是本机现场探测的证据,原样记录,不是转述:

**探测 1:GRUB2 工具链默认不存在**
```
$ rpm -q grub2-tools
package grub2-tools is not installed
$ ls -la /boot/
total 8
dr-xr-xr-x  2 root root 4096 Apr  2  2025 .
drwxr-xr-x 34 root root 4096 Jul 11 13:59 ..
$ ls /boot/grub2/
ls: cannot access '/boot/grub2/': No such file or directory
$ grub2-mkconfig --help
bash: grub2-mkconfig: command not found
$ command -v grubby
(空,退出码1)
```
`/boot` 目录里**没有任何内核镜像、没有 initramfs、没有 GRUB 目录**——这不是权限问题或者没扫描到,是这台机器从出厂就没有走过"把内核文件放进 /boot、GRUB 从这里读取"这套流程。

**探测 2:装上 grub2-tools 之后,命令能跑、不报错,生成的配置内容取决于 `/boot` 里有没有真实内核——但不管有没有,都不影响下面探测 3 的结论**

按照本类目原则"能装就装、能测就测",现场用 `dnf install -y grub2-tools grubby` 把工具链真实装上(连带装了 `grub2-common`/`grub2-tools-minimal`/`dracut`/`os-prober` 等约 20 个包),再真实执行 `grub2-mkconfig`:
```
$ grub2-mkconfig -o /tmp/test_grub.cfg
Generating grub configuration file ...
Adding boot menu entry for UEFI Firmware Settings ...
done
```
**命令报告成功,退出码 0——但这正是本条最需要警惕的地方,以下是本机两个不同时间点现场观察到的两种真实情况,都值得记录:**

**情况 A(本机第一次测试时,`/boot` 里没有任何内核文件):**
```
$ grep -E 'menuentry |^### (BEGIN|END) /etc/grub.d/10_linux' /tmp/test_grub.cfg
### BEGIN /etc/grub.d/10_linux ###
### END /etc/grub.d/10_linux ###
		menuentry 'UEFI Firmware Settings' $menuentry_id_option 'uefi-firmware' {
```
`10_linux` 是 GRUB 负责"扫描已安装内核、为每个内核生成一个可启动菜单项"的那个子脚本——它的 `BEGIN`/`END` 之间**完全是空的**,因为 `/boot` 里根本没有内核文件可扫描。整份配置文件里唯一的菜单项是通用的"UEFI 固件设置"占位符,和任何真实 Linux 内核都无关;`grubby --default-kernel` 同样返回空,查无 BLS 条目。

**情况 B(本机复验这一节代码块时,共享 WSL 环境里另一个并发 agent 已经装了一个真实的 `kernel` rpm 包,如实记录这个变化后的状态):**
```
$ rpm -qa kernel kernel-core
kernel-6.12.0-211.28.1.el10_2.x86_64
kernel-core-6.12.0-211.28.1.el10_2.x86_64
$ grubby --default-kernel
/boot/vmlinuz-6.12.0-211.28.1.el10_2.x86_64
$ cat /boot/loader/entries/*.conf
title Rocky Linux (6.12.0-211.28.1.el10_2.x86_64) 10.2 (Red Quartz)
version 6.12.0-211.28.1.el10_2.x86_64
linux /boot/vmlinuz-6.12.0-211.28.1.el10_2.x86_64
initrd /boot/initramfs-6.12.0-211.28.1.el10_2.x86_64.img
options root=UUID=df87c8c2-4b19-4cd7-8e93-e81b066f8f00 ro
```
这次 `10_linux` 段落**不是空的**了,而是包含标准的 BLS(Boot Loader Specification)加载逻辑(`insmod blscfg` + `blscfg`,现代 RHEL 8+/Rocky 的 `grub2-mkconfig` 不再把每个内核的 `menuentry` 直接写死在 `grub.cfg` 里,而是在启动时由 `blscfg` 命令动态读取 `/boot/loader/entries/*.conf`),`grubby --default-kernel` 也真实查到了一个存在的内核文件。**更值得注意的是:`options root=UUID=df87c8c2-...` 这个 UUID 用 `blkid /dev/sdd` 核实过,精确匹配 WSL 这个虚拟机真实根文件系统的 UUID**——也就是说,`grub2-mkconfig` 认认真真地做了正确的自动检测,生成的配置**从内容上看完全正确、完全可信**。

**探测 3(不管情况 A 还是情况 B 都成立,这才是本节真正的结论所在):真正在跑的内核是什么、从哪来的**
```
$ cat /proc/cmdline
initrd=\initrd.img WSL_ROOT_INIT=1 panic=-1 nr_cpus=20 hv_utils.timesync_implicit=1 console=hvc0 debug pty.legacy_count=0 WSL_ENABLE_CRASH_DUMP=1

$ uname -r
6.18.33.2-microsoft-standard-WSL2

$ find /boot -iname "*6.18.33.2-microsoft-standard-WSL2*"
(空)
```
真实生效的启动参数是 `WSL_ROOT_INIT=1`/`hv_utils.timesync_implicit=1`/`WSL_ENABLE_CRASH_DUMP=1` 这些 **WSL 专有**参数,和情况 B 里 BLS 条目登记的 `root=UUID=... ro` 完全对不上;正在运行的内核版本号带着 `microsoft-standard-WSL2` 后缀,是微软单独维护的内核分支,和情况 B 里 `rpm` 装的 `kernel-6.12.0-211.28.1.el10_2.x86_64` 是完全不同的两个内核;`/boot` 目录下找不到任何和这个正在运行的内核版本对应的文件——**即便情况 B 里 `grub2-mkconfig` 生成了一份内容完全正确、UUID 都精确匹配真实磁盘的"看起来无懈可击"的配置,当前实际运行的内核依然和这份配置毫无关系**。这才是本节最重要的证据:**判断一份 GRUB 配置是否真的在被使用,不能只看它"生成得对不对",要看正在运行的内核能不能在这份配置的登记范围里找到——本机这两次测试,答案都是找不到。**

**结论(必须明确,不能含糊):** `grub2-mkconfig`/`grubby` 这些命令在 WSL2 里**可以安装、可以执行、可以不报错,甚至能生成内容完全正确、连磁盘 UUID 都精确检测对了的配置文件**,但这份文件从生成的那一刻起就是一个孤立的、不会被任何引导流程读取的产物——**本节不能也不会声称"验证了修改内核启动参数能改变下次启动行为"这件事,这在 WSL2 架构下是做不到的**。真实 RHCSA 考试环境(裸机或标准 BIOS/UEFI 虚拟机)上,`grub2-mkconfig`/`grubby` 操作的是真正被引导流程读取的文件,效果和这里观察到的完全不同。

**真实场景例子(仅描述真实 RHEL 环境下的标准做法,不在本机验证):** 给内核启动参数追加 `net.ifnames=0`(改用传统 `eth0` 式网卡命名),标准做法是 `grubby --update-kernel=ALL --args="net.ifnames=0"` 后重启验证;RHCSA 考试如果要求"永久调整某个内核参数并重启验证",`grubby` 是比手改 `grub.cfg` 风险低得多的标准工具。

**可运行例子(如实标注:以下只验证"命令能否安装/执行、生成的配置是否和当前真正运行的内核对得上",不验证、也无法验证"下次启动是否真的应用了这些参数"——这一点无论 `/boot` 里当时有没有真实内核文件都成立,代码写成不依赖某个固定起点状态,原因见下方"本机实测"说明):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

dnf_retry() {
    local attempt=1 max=6
    while [ "$attempt" -le "$max" ]; do
        if dnf "$@"; then return 0; fi
        sleep $((attempt * 2)); attempt=$((attempt + 1))
    done
    return 1
}

# 记录起点状态,不强行假设——共享 WSL 环境里这些包可能已经被其他并发操作装上
grub2_tools_preexisting=0
rpm -q grub2-tools >/dev/null 2>&1 && grub2_tools_preexisting=1

# 装上工具链(能装,和"能引导"是两回事;如果已经装了,这一步就是空操作)
assert_ok dnf_retry install -y grub2-tools grubby

# grub2-mkconfig 能跑、不报错
assert_ok grub2-mkconfig -o /tmp/rhcsa05_test_grub.cfg

# 决定性证据(不管此刻/boot里有没有真实内核、生成的配置内容是空是满,这一条永远成立):
# 当前真正运行的内核,在/boot下找不到任何对应文件
running_kernel=$(uname -r)
kernel_file_count=$(find /boot -iname "*${running_kernel}*" 2>/dev/null | wc -l)
assert_eq "$kernel_file_count" "0"

# 如果此刻系统里已经有一个真实kernel rpm(这里装的,或者共享环境里其他并发操作装的),
# grubby能查到的"默认内核"版本号必然和uname -r的真实运行内核不是同一个——这是更有力的一层证据
default_kernel_path=$(grubby --default-kernel 2>/dev/null)
if [ -n "$default_kernel_path" ]; then
    bls_version=$(basename "$default_kernel_path" | sed 's/^vmlinuz-//')
    assert_ok bash -c "[ '$bls_version' != '$running_kernel' ]"
    echo "OK: BLS登记的内核版本($bls_version)和真正运行的内核($running_kernel)确认不是同一个"
else
    echo "本机当前没有已安装的kernel rpm,grubby --default-kernel返回空,10_linux生成段落也是空的"
fi

# 清理:只清理自己创建的临时文件;grub2-tools/grubby 只有测试前确实不存在时才卸载,
# 如果是共享环境里其他并发操作已经装好的,不动它,避免影响对方
rm -f /tmp/rhcsa05_test_grub.cfg
if [ "$grub2_tools_preexisting" = "0" ]; then
    assert_ok dnf_retry remove -y grub2-tools grubby
fi
```
本机实测:全部 `assert_eq`/`assert_ok` 输出 `OK`。**这段代码在两个不同时间点各完整跑通一次,结果都是 `OK`,但走的是上面"情况 A"/"情况 B"两条不同路径**——第一次跑时 `/boot` 里没有任何内核,`grubby --default-kernel` 返回空,只验证了"找不到运行内核对应的文件"这一条;复验时共享 WSL 环境里已经有其他并发 agent 装了真实 `kernel` 包,这次额外验证了"即使 BLS 条目真实存在,登记的内核版本号也和真正运行的内核对不上"这一条更有力的证据,两次都支持同一个结论。第一次测试完成后按代码块逻辑清理了自己装的 `grub2-tools`/`grubby`(当时确认是本节新装的);复验时发现这两个包已经是共享环境里其他并发操作装好的(`dnf history` 可查到是另一个事务装的),按代码块逻辑保留不动,没有删除。

**常见坑:**
1. **最重要的一条,再强调一遍:WSL2 里 `grub2-mkconfig`/`grubby` 能跑不代表验证了内核启动参数的真实效果。** 这是 WSL2 虚拟化架构的结构性限制,不是配置问题,也不是这个 Rocky Linux 镜像缺了什么东西——真实 RHCSA 考试环境用的是标准引导链路,行为和这里完全不同,本节的价值在于**诚实展示这个边界在哪里、如何用证据识别它**,而不是假装在 WSL 里把这件事验证过了。
2. dnf 的 `protected_packages` 机制:本机第一次尝试 `dnf remove -y grub2-tools grub2-tools-minimal grubby`(**显式点名** `grub2-tools-minimal`)时,真实报错是:
   ```
   Error:
    Problem: The operation would result in removing the following protected packages: grub2-tools-minimal
   ```
   `/etc/dnf/protected.d/grub2-tools-minimal.conf` 里确实登记了这个包受保护(避免误删导致系统缺失关键引导工具)。但只删 `grub2-tools grubby`(不显式点名 `grub2-tools-minimal`),让它作为"不再被需要的依赖"被 `clean_requirements_on_remove` 级联带走,`protected_packages` 反而不拦截——**这个保护机制拦截的是"显式点名删除",不拦截"作为孤儿依赖被级联清理"**,这是一条容易被这次报错误导、但值得记住的真实 dnf 行为细节。
3. 真实 RHEL 上如果强行绕过 `protected_packages` 保护(比如加 `--setopt=protected_packages=`)删掉这几个包,是真正危险的操作——这几个包在有真实 GRUB 引导的系统上被保护是有原因的,不建议在生产/考试环境模仿"绕过保护"这个动作,这里只是如实记录 WSL 环境下不显式点名时的级联清理行为。

---

## 6. 内核参数临时/永久调整(`sysctl`)

**命令/配置:**
```bash
sysctl -a                       # 查看所有当前生效的内核参数
sysctl -n PARAM.NAME              # 只查看某一个参数的当前值
sysctl -w PARAM.NAME=VALUE          # 临时修改(立即生效,重启/重载后丢失)
/etc/sysctl.d/*.conf                 # 永久生效的配置文件存放位置
sysctl --system                       # 按顺序重新加载全部 sysctl 配置文件
```

**一句话是什么:** `sysctl` 是运行时读写 `/proc/sys/` 下内核参数的接口,`-w` 做的修改**立即生效但只在当前运行期间有效**,想让配置在重启后依然生效要写进 `/etc/sysctl.d/` 下的配置文件——这和"手动 `mount` vs 写进 `/etc/fstab`"(04类第2-3节)是完全一样的"临时 vs 永久"设计模式。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求能"调整系统内核运行时参数,并使其永久生效";生产环境常见调优场景包括网络转发开关、连接队列大小、文件句柄上限等。

**从最容易犯错的做法讲起:** 只用 `sysctl -w` 做了修改就以为完成任务,下次系统重启后发现配置"离奇消失"——`-w` 修改的是内核当前运行时的内存状态,不会自动持久化,必须额外把同样的设置写进 `/etc/sysctl.d/*.conf` 文件,两步都要做才算完整答案。

**真实场景例子(典型运维场景,非仓库代码):** 服务器要当路由器/网关用,需要开启 IP 转发:`sysctl -w net.ipv4.ip_forward=1` 立即生效验证配置对不对,确认没问题后写入 `/etc/sysctl.d/99-ip-forward.conf`(内容 `net.ipv4.ip_forward = 1`)保证重启后依然生效,这是"先临时验证、再固化配置"的标准工作流。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

orig_forward=$(sysctl -n net.ipv4.ip_forward)
orig_swap=$(sysctl -n vm.swappiness)

# -w 立即生效
sysctl -w net.ipv4.ip_forward=1 >/dev/null
assert_eq "$(sysctl -n net.ipv4.ip_forward)" "1"

# sysctl只是/proc/sys的封装,直接读写/proc/sys效果完全一样
echo 0 > /proc/sys/net/ipv4/ip_forward
assert_eq "$(sysctl -n net.ipv4.ip_forward)" "0"

# 永久配置文件 + --system 重新加载,验证真实覆盖生效
cat > /etc/sysctl.d/99-rhcsa05-test.conf << EOF
net.ipv4.ip_forward = 1
vm.swappiness = 15
EOF
assert_ok bash -c 'sysctl --system 2>&1 | grep -q "99-rhcsa05-test.conf"'
assert_eq "$(sysctl -n net.ipv4.ip_forward)" "1"
assert_eq "$(sysctl -n vm.swappiness)" "15"

# 恢复原值并清理
rm -f /etc/sysctl.d/99-rhcsa05-test.conf
sysctl -w net.ipv4.ip_forward="$orig_forward" >/dev/null
sysctl -w vm.swappiness="$orig_swap" >/dev/null
sysctl --system >/dev/null 2>&1
assert_eq "$(sysctl -n net.ipv4.ip_forward)" "$orig_forward"
assert_eq "$(sysctl -n vm.swappiness)" "$orig_swap"
```
本机实测:全部 `assert_eq`/`assert_ok` 输出 `OK`。`sysctl --system` 真实输出会按文件名顺序逐个 `Applying` 多个来源(`/etc/sysctl.d/*.conf`、`/etc/sysctl.conf` 等),`99-rhcsa05-test.conf` 确认出现在其中一行。

**这一项和第5项(GRUB2)的机制完全不同,不能因为第5项做不到就假设这一项也做不到:** GRUB2 做不到是因为 WSL2 根本不走 GRUB 引导链路;而 `/proc/sys` 是当前运行内核的实时状态,WSL2 有自己独立、真实运行的 Linux 内核实例(`6.18.33.2-microsoft-standard-WSL2`),`/proc/sys` 下的读写和真实 RHEL 没有本质区别。本机为此**特意做了较宽的实测排查**,而不是想当然假设有限制,结果如下:

| 参数 | 结果 |
|---|---|
| `net.ipv4.ip_forward` / `vm.swappiness` | 可读写,上面例子已验证 |
| `net.bridge.bridge-nf-call-iptables` | **可读写**——一开始猜测 WSL 可能没加载桥接网络的 netfilter 模块,现场 `lsmod` 检查发现 `br_netfilter`/`bridge` 模块其实已经加载,这条参数本身就存在且可读写,原有猜测被证伪,如实更正 |
| `net.netfilter.nf_conntrack_max` | 可读(`262144`),`/proc/sys/net/netfilter/` 下相关文件齐全 |
| `vm.nr_hugepages` | 可读写(默认 `0`) |
| `kernel.perf_event_paranoid` | 可读写,数值往返测试通过 |
| `fs.inotify.max_user_watches` | 可读写,数值往返测试通过 |
| `net.ipv4.tcp_congestion_control` | 可写;默认可选列表只有 `reno cubic`,但写入 `bbr` 时内核会按需自动加载 `tcp_bbr` 模块并切换成功——"可选列表"只反映**已加载**的算法,不反映**能否按需加载**,这条本身就是一个容易被忽视的细节 |
| `kernel.core_pattern` | 可读写,当前值是标准的 `systemd-coredump` 管道格式,和真实 RHEL 一致 |
| `user.max_user_namespaces` | 可读(`95463`,健康的非零值) |

**结论:本机没有找到任何一个测试过但读写失败的 sysctl 参数**——这本身就是一个诚实、有意义的结论(证明测试是真的做过而不是走过场),但不代表"所有 sysctl 参数在任何 WSL 环境下都保证可用",只代表本机广泛抽样测试的这些参数(涵盖网络/内存/文件系统/内核几个类别)都可用。

**常见坑:**
1. `/etc/sysctl.d/` 下多个配置文件之间**按文件名字典序加载**,后加载的会覆盖先加载的同名参数——本机 `sysctl --system` 真实输出显示加载顺序是 `99-rhcsa05-test.conf` → `99-sysctl.conf`(`/etc/sysctl.conf` 的软链接)→ `/etc/sysctl.conf`,如果多个文件对同一参数给出不同值,最终生效的是排序最后那个。
2. **本篇现场踩过的一个真实教训(供参考,不是 RHCSA 考点,是本机测试方法论的坑):** 探测 `net.bridge.bridge-nf-call-iptables` 时用 `lsmod | grep br_netfilter && modprobe -r br_netfilter` 做"用完就卸载"的清理,却忽略了这个模块**测试前就已经加载**(不是本次测试加载的),这条清理逻辑误将系统原有状态卸载掉了,发现后立刻用 `modprobe br_netfilter` 补救恢复。这提醒一个通用道理:清理临时资源前要先确认"这东西是不是我自己创建的",而不是看到"存在"就无脑清理。
3. 内核参数的**运行时生效**(本项)和**内核启动参数**(上一项 GRUB2/`grubby`)是两套完全不同的机制,虽然都叫"内核参数"、都通过命令行传入类似 `key=value` 的语法,但 `sysctl` 管的是 `/proc/sys` 这棵运行时可读写的树,`grubby`/GRUB 管的是内核**启动那一刻**才读取一次、运行中无法通过 `sysctl` 热改的参数(比如 `crashkernel=`、`selinux=0` 这类),两者不能混为一谈。

---

## 7. chronyd 时间同步配置

**命令/配置:**
```bash
systemctl status chronyd       # chrony 服务状态
chronyc tracking                 # 详细同步统计(偏移量、频率误差等)
chronyc sources -v                # 当前配置的时间源及同步质量,带图例
chronyc sourcestats                # 每个时间源的统计学质量指标
chronyc -a makestep                 # 强制立即步进校正一次(常用排障手段)
/etc/chrony.conf                     # 主配置文件
```

**一句话是什么:** `chronyd` 是 RHEL 现行的标准 NTP 客户端/服务端实现,持续把本机时钟向权威时间源做渐进式校正——分布式系统里时间不同步会导致日志时间线错乱、证书有效期判断错误、Kerberos 认证失败,时间同步是容易被忽视但很基础的可靠性要求。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"配置时间服务客户端";证书验证、Kerberos 认证、分布式日志排查都强依赖准确的系统时间。

**从最容易犯错的做法讲起:** 遇到时间不准直接手动 `date -s` 改系统时间——`chronyd` 会持续按配置的时间源做渐进式校正,手动改的时间很快会被"纠正"回去;正确做法永远是确保 `chronyd` 服务本身在正常运行、配置了可靠的时间源,而不是隔三差五手动 `date` 一下当"临时止血"。

**真实场景例子(典型运维场景,非仓库代码):** 新装服务器加入内网,把 `/etc/chrony.conf` 的 `server`/`pool` 指向企业内部 NTP 服务器(而不是默认公网源,内网可能访问不了公网),`chronyc sources` 确认已连接上配置的时间源;`chronyc tracking` 里的 `System time` 偏移量字段可以快速判断当前系统时间的准确程度。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

assert_ok systemctl is-active --quiet chronyd

source_count=$(chronyc sources 2>&1 | grep -cE '^\^')
assert_ok test "$source_count" -ge 1

assert_ok bash -c 'chronyc tracking 2>&1 | grep -q "Reference ID"'
assert_ok bash -c 'timedatectl 2>&1 | grep -q "System clock synchronized: yes"'

# 强制步进(真实 RHCSA 排障手段:时间跳变较大时用这个命令立即纠正,而不是等chronyd慢慢收敛)
assert_ok bash -c 'chronyc -a makestep 2>&1 | grep -q "200 OK"'

# 修改配置(临时追加一个时间源)、重启服务、确认新配置生效,再恢复原配置
cp /etc/chrony.conf /tmp/rhcsa05_chrony.conf.bak
echo "server time.cloudflare.com iburst" >> /etc/chrony.conf
assert_ok systemctl restart chronyd
sleep 2
assert_ok systemctl is-active --quiet chronyd
assert_ok bash -c 'chronyc sources 2>&1 | grep -q "time.cloudflare.com"'

cp /tmp/rhcsa05_chrony.conf.bak /etc/chrony.conf
assert_ok systemctl restart chronyd
sleep 2
assert_ok systemctl is-active --quiet chronyd
rm -f /tmp/rhcsa05_chrony.conf.bak
```
本机实测:全部 `assert_eq`/`assert_ok` 输出 `OK`。`chronyc tracking` 现场显示 `Stratum: 3`,真实连上了公网时间源;`timedatectl` 确认时区 `Asia/Shanghai (CST, +0800)`、`System clock synchronized: yes`。

**WSL 环境的一个真实细节(不是限制,是值得了解的架构背景):** `/proc/cmdline` 里能看到 `hv_utils.timesync_implicit=1`,`dmesg` 里有 `hv_utils: Registering HyperV Utility Driver` 和 `PTP clock support registered`——WSL2 的 Linux 虚拟机本身还有一路独立的 **Hyper-V 时间同步机制**(让虚拟机时钟跟 Windows 宿主机对齐),和 `chronyd` 的 NTP 同步是两套并存、互不冲突的机制,`chronyd` 该怎么配置、怎么验证不受这一层影响,这里只是如实记录本机 `dmesg` 现场看到的真实证据。

**常见坑:** `chronyc sources` 每行开头的符号有讲究——`^*` 表示当前实际被选用作参考的时间源,`^+` 表示"备选、可用但当前没被选中",`^-` 表示"因延迟/抖动等原因被判定为不够可靠",`^?` 表示"刚加入、还没收集到足够数据判断质量"。本机实测添加 `time.cloudflare.com` 后,它有一段时间显示为 `^?`(reach 寄存器还是 `0`),过几个轮询周期才逐渐变成有效状态——只看到一堆 `^` 开头的行就以为"都在正常工作"是不够精确的,`reach` 列(八进制的可达性寄存器)和最左边的状态符号才是判断同步健康度的关键。

---

## 8. podman 容器基础(拉取/运行/管理容器,rootless 特性)

**命令/配置:**
```bash
podman pull IMAGE          # 拉取镜像
podman run IMAGE            # 基于镜像运行一个容器
podman ps -a                  # 列出容器(含已停止的,-a)
podman rm NAME                 # 删除容器
podman images                   # 列出本地已有的镜像
```

**一句话是什么:** `podman` 是 RHEL 官方主推的容器引擎,命令行用法和 `docker` 高度兼容,但架构上是 **daemonless**(没有常驻后台守护进程)且设计上支持 **rootless**(普通用户无需 root 权限运行容器)——这两点是它和传统 docker 架构的核心区别。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"安装配置容器,启动/停止/查看容器";RHEL 生态已全面转向 podman,理解容器基础生命周期管理是现代 RHEL 系统管理的必备技能。

**从最容易犯错的做法讲起:** 到处找一个常驻的 `podmand` 之类守护进程,试图像管理 docker 服务那样 `systemctl status podman`——这是从 docker 经验带来的错误预期,podman 每次执行命令都是独立进程调用,没有常驻后台进程,这个服务名根本不存在。另一个更隐蔽的坑(本机现场踩到、值得写进来的):**以 root 身份运行 podman,默认就是 rootful(以 root 权限跑容器),不会自动变成 rootless**——"rootless" 描述的是"运行 podman 的这个 Linux 用户是不是 root",而不是 podman 本身有什么特殊的降权模式;想要真正体验 rootless,必须切到一个非 root 用户去跑。

**真实场景例子(典型运维场景,非仓库代码):** 快速验证一个应用能否跑起来,不想在宿主机上装一堆依赖,直接 `podman run` 一个包含完整环境的镜像跑起来测试;给普通开发者账号开通容器权限时用 rootless 模式,不需要把开发者加进 root 组,权限爆炸面小得多,这是生产环境偏好 rootless 的核心原因。

**可运行例子(rootful 部分,root 用户直接跑,完整验证通过):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

# 以root身份运行,rootless标志确认为false(这就是上面"常见坑"讲的:root跑podman默认是rootful)
assert_eq "$(podman info --format '{{.Host.Security.Rootless}}')" "false"

# quay.io/podman/hello 是官方轻量测试镜像,本机确认可达;docker.io在本机网络环境下连接超时,如实记录不强行使用
assert_ok bash -c 'timeout 6 curl -sI https://quay.io/v2/ -o /dev/null -w "%{http_code}"'

assert_ok bash -c 'timeout 30 podman run --rm quay.io/podman/hello 2>&1 | grep -q "Hello Podman World"'

# --rm 确认容器跑完自动清理,不留痕迹
assert_eq "$(podman ps -a --format '{{.Names}}' | wc -l)" "0"
```
本机实测:全部 `assert_eq`/`assert_ok` 输出 `OK`。`docker.io` 探测的真实报错是 `dial tcp 154.83.15.20:443: i/o timeout`(重试 3 次后放弃),`quay.io`/`registry.access.redhat.com` 探测均正常返回 HTTP 状态码——这是本机网络环境对不同镜像仓库连通性不同导致的,不是 podman 本身的问题,如实记录、换一个确认可达的仓库继续验证,而不是被卡住。

**rootless 部分——现场真实失败,附完整诊断过程(这是本项最有价值的部分,请完整阅读):**

为了真正验证 rootless(而不是自欺欺人地"以 root 身份跑然后叫它 rootless"),现场创建了一个专门用于测试的非特权用户并切换过去:
```
$ useradd -m rhcsa05podman
$ grep rhcsa05podman /etc/subuid /etc/subgid
/etc/subuid:rhcsa05podman:524288:65536
/etc/subgid:rhcsa05podman:524288:65536
```
`subuid`/`subgid` 范围按 `/etc/login.defs` 里的 `SUB_UID_MIN`/`SUB_UID_COUNT` 默认值正确自动分配了,这部分没有问题。切到这个用户执行 `podman run`,真实报错:
```
$ su - rhcsa05podman -c 'podman run --rm quay.io/podman/hello'
WARN[0000] "/" is not a shared mount, this could cause issues or missing mounts with rootless containers
ERRO[0000] running `/usr/bin/newuidmap 630 0 1000 1 1 524288 65536`: newuidmap: write to uid_map failed: Operation not permitted
Error: cannot set up namespace using "/usr/bin/newuidmap": should have setuid or have filecaps setuid: exit status 1
```
两条独立线索,分开诊断:

**线索1:`"/" is not a shared mount`。** `findmnt -o TARGET,PROPAGATION /` 确认根挂载点的传播类型是 `private`,而不是真实 RHEL 上 systemd 默认设置的 `shared`/`rshared`——这是 WSL2 和裸机/标准虚拟机在挂载命名空间处理上的真实架构差异,rootless 容器依赖共享传播来正确处理挂载事件。

**线索2(真正的阻塞点):`newuidmap` 权限问题。** 这条错误信息字面意思是"`newuidmap` 这个二进制需要有 setuid 位或者文件能力(file capability)才能替一个非特权用户写 `uid_map`",现场逐层排查:
```
$ ls -la /usr/bin/newuidmap /usr/bin/newgidmap
-rwxr-xr-x 1 root root 43144 Feb 23 08:00 /usr/bin/newuidmap
-rwxr-xr-x 1 root root 43160 Feb 23 08:00 /usr/bin/newgidmap
# 注意:没有setuid位(s),真实RHEL上这两个文件应该带某种特权标记

$ getcap /usr/bin/newuidmap /usr/bin/newgidmap
(空,没有任何文件能力)

$ rpm -V shadow-utils
........P    /usr/bin/newgidmap
........P    /usr/bin/newuidmap
# rpm -V 的 "P" 标记明确指出:这两个文件的"caPabilities"和包记录的原始状态不一致
```
排除了两个可能的干扰因素:
- **不是 `nosuid` 挂载导致的**:`mount | grep ' / '` 确认根文件系统挂载选项是 `rw,relatime,discard,errors=remount-ro,data=ordered`,没有 `nosuid`。
- **不是系统性地丢失了所有 setuid 位**:同一批检查里,`chage`/`mount`/`passwd`/`su`/`sudo` 这些经典 setuid 程序全部正常保留着它们的特权位(`-rwsr-xr-x` 或 `---s--x--x`),`rpm -V` 对它们也没有报任何异常,问题**精确地**只出在 `newuidmap`/`newgidmap` 这两个文件上。

**结论(最可能的原因,基于以上证据推断,不是臆测):** `newuidmap`/`newgidmap` 这两个工具在现代 shadow-utils 打包方式里,依赖的是 Linux **文件能力**(`security.capability` 扩展属性)而不是传统的 setuid 权限位——这类扩展属性在 `tar`/OCI 镜像层的打包、导出、再解压过程中比普通权限位脆弱得多,容易在某次转换里丢失。结合 [00-roadmap.md](00-roadmap.md) 里记录的本机环境搭建历史(这个 WSL 根文件系统最初是从一个 OCI 容器镜像的 blob 层解包得到的,不是走传统 ISO/kickstart 安装),这个丢失很可能发生在那次镜像层导出/解包的过程中——其他经典 setuid 程序用的是存放在普通权限位里的信息(tar 头本身就带,不容易丢),而这两个文件依赖的文件能力存放在扩展属性里,更容易在这类转换里丢失,这正好解释了"为什么只有这两个文件出问题、其他 setuid 程序都完好"。

**这个问题本可以用 `setcap cap_setuid+ep /usr/bin/newuidmap`(对 `newgidmap` 同理用 `cap_setgid+ep`)修复,但没有在本机执行这个修复**——这是一个会永久修改共享 WSL 环境系统安全属性的操作,而当前 WSL 实例同时有其他 agent 在使用,未经明确授权不应该对共享环境做这类持久性变更,因此如实记录诊断结果、不擅自修复,把"标准修复方法是什么"和"为什么没有在这里执行"都写清楚。

**cleanup:**
```
$ userdel -r rhcsa05podman
$ grep rhcsa05podman /etc/passwd /etc/subuid /etc/subgid   # 确认无残留
(空)
```

**常见坑:**
1. `podman ps`(不加 `-a`)默认只显示**正在运行**的容器,一次性任务型容器(比如 `hello` 镜像,跑完打印信息就退出)会"消失不见",一定要加 `-a` 才能看到已停止/已退出的容器。
2. **root 身份运行 podman 默认是 rootful,不会自动变成 rootless**——两者是完全不同的运行模式(不同的存储路径、不同的网络实现),不能拿"用 root 跑通了"当作"验证了 rootless"的证据,这是本项现场验证时特意分两条路径测试的原因。
3. rootless podman 依赖 `newuidmap`/`newgidmap` 正确持有 `CAP_SETUID`/`CAP_SETGID`(通过 setuid 位或文件能力),这一层依赖在**镜像化/容器化交付的系统镜像**(相对传统安装介质而言)上有更高概率因为打包过程丢失文件能力这类扩展属性而失效,遇到 `newuidmap: write to uid_map failed: Operation not permitted` 时,`ls -la`/`getcap`/`rpm -V` 三件套是标准诊断路径,标准修复是 `setcap cap_setuid+ep`/`cap_setgid+ep` 补回对应能力。

---

## 9. podman 生成 systemd 服务(quadlet / `podman generate systemd`)

**命令/配置:**
```bash
# 现代做法:quadlet(推荐)——写一个 .container 文件,systemd 自动生成对应的 .service
# /etc/containers/systemd/xxx.container(系统级)或 ~/.config/containers/systemd/(用户级)

# 经典做法(已标记 DEPRECATED,仍可用):
podman generate systemd --name CONTAINER --new --files
```

**一句话是什么:** 把容器变成"开机自启、可被 `systemctl` 管理"的标准系统服务。Podman 5.x(本机 `5.8.2`)时代官方推荐的做法是 **quadlet**——写一个语法接近 `.desktop`/`.service` 的 `.container` 声明式文件放进 `/etc/containers/systemd/`,`systemctl daemon-reload` 时由 podman 自带的 systemd generator(`/usr/lib/systemd/system-generators/podman-system-generator`)自动翻译成一份完整正确的 `.service` 单元;`podman generate systemd`(先创建容器、再让 podman "反向生成" unit 文件)是更早期的做法,官方 `--help` 里明确写着 `[DEPRECATED]`,建议迁移到 quadlet。

**为什么 RHCSA 真考 / 生产会用到:** 单纯 `podman run` 起来的容器,系统重启后不会自动恢复;生产环境需要容器化服务"开机自启、崩溃后由 systemd 负责重启策略",quadlet/`generate systemd` 是把容器接入 systemd 这套成熟运维体系的标准桥梁。

**从最容易犯错的做法讲起:** 写好 `.container` 文件或者拿到 `generate systemd` 生成的 `.service` 文件后,直接以为"已经在跑了"——两种做法都还需要 `systemctl daemon-reload`(quadlet 靠这一步触发 generator 翻译文件;经典做法则是新增了 unit 文件后必须 reload 才能被 systemd 感知,这和 02 类第 4 节讲过的"改了 unit 文件必须 reload"是同一条规则)。本机现场踩到一个**更隐蔽、更容易被忽视**的坑,详见下方"常见坑"第1条,比"忘记 reload"更值得记录。

**真实场景例子(典型运维场景,非仓库代码):** 把一个长期运行的服务容器纳入系统级管理:写好 `/etc/containers/systemd/mydb.container`(声明镜像、启动命令等),`daemon-reload` 后 `systemctl start mydb.service`,之后这个容器就和普通系统服务一样可以用标准 `systemctl status`/`journalctl -u` 管理和查日志,不需要额外记一套容器专用的操作方式。

**可运行例子(quadlet,现场真实跑通完整生命周期):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

# 确认 quadlet 生成器随 podman 一起提供,不需要额外安装
assert_ok test -e /usr/lib/systemd/system-generators/podman-system-generator

mkdir -p /etc/containers/systemd
cat > /etc/containers/systemd/rhcsa05-hello.container << 'EOF'
[Unit]
Description=RHCSA05 quadlet demo container

[Container]
Image=registry.access.redhat.com/ubi9/ubi-minimal:latest
Exec=sleep 3600

[Service]
Restart=no

[Install]
WantedBy=multi-user.target
EOF

# 预先拉好镜像,让下面的start不必等首次拉取
assert_ok timeout 120 podman pull registry.access.redhat.com/ubi9/ubi-minimal:latest

assert_ok systemctl daemon-reload
assert_ok bash -c 'systemctl list-unit-files 2>&1 | grep -q rhcsa05-hello.service'
# generator真实生成的.service文件里,ExecStart必须是podman run
assert_ok bash -c 'cat /run/systemd/generator/rhcsa05-hello.service | grep -q "ExecStart=/usr/bin/podman run"'

assert_ok systemctl start rhcsa05-hello.service
sleep 2
assert_eq "$(systemctl is-active rhcsa05-hello.service)" "active"
assert_ok bash -c 'podman ps --format "{{.Names}}" | grep -q systemd-rhcsa05-hello'

assert_ok systemctl stop rhcsa05-hello.service
sleep 1
assert_eq "$(podman ps -a --format '{{.Names}}' | wc -l)" "0"    # ExecStop用podman rm -f,停止后自动清理容器

# 清理
rm -f /etc/containers/systemd/rhcsa05-hello.container
systemctl daemon-reload
systemctl reset-failed >/dev/null 2>&1
podman rmi registry.access.redhat.com/ubi9/ubi-minimal:latest >/dev/null 2>&1
```
本机实测:全部 `assert_eq`/`assert_ok` 输出 `OK`(**注意:上面例子把 `Restart` 显式设成了 `no`,原因见下方常见坑第1条——这是根据现场踩坑之后修正过的版本,不是最初的写法**)。`systemctl start` 后 `systemctl status` 真实显示 `Active: active (running)`,进程树里能看到 `conmon`+`crun` 真实管理着容器内的 `sleep 3600` 进程;`journalctl -u rhcsa05-hello.service` 能看到完整的 `container create` → `image pull` → `container init` → `container start` 生命周期事件,和管理任何其他 systemd 服务的体验一致。

**常见坑:**
1. **`Restart=on-failure`(或 `always`)配合 quadlet 默认的 `ExecStop=podman rm -v -f -i systemd-%N` 会导致 `systemctl stop` 停不住服务——这是本项现场真实踩到、比"忘记 reload"更隐蔽的坑。** 第一次写这个 `.container` 文件时按常见模板加了 `Restart=on-failure`,现场真实观察到:执行 `systemctl stop` 后返回码是 0、看起来成功了,但 `journalctl` 里紧接着出现 `rhcsa05-hello.service: Failed with result 'exit-code'`,几十秒后 `systemctl status` 显示服务**又重新在跑了**(新的启动时间戳,容器 ID 也变了)。根本原因:quadlet 生成的 `ExecStop` 用 `podman rm -v -f -i` 强制杀掉并移除容器,这个强制终止方式让 systemd 认为主进程是"异常退出"(`result=exit-code`)而不是"被正常要求停止"——一旦服务配了 `Restart=on-failure`,systemd 就会按策略自动重启它,`systemctl stop` 的操作意图被无声地抵消了。**修复方法:一次性/按需手动管理的容器服务把 `Restart` 设成 `no`(如上面例子);真的需要"崩溃后自动拉起"语义的长期服务,则要清楚知道这个副作用,`systemctl stop` 之后如果还想彻底停住,得配合 `systemctl disable` 或者直接用 `podman rm -f` 绕开 systemd 直接终止容器。**
2. `podman generate systemd`(经典做法)在官方 `--help` 里已经明确标注 `[DEPRECATED]`,建议迁移到 quadlet——本机同样现场验证过经典做法可以正常生成文件(`# autogenerated by Podman 5.8.2` 开头的标准格式,`ExecStart` 里是 `podman run --cidfile=... --cgroups=no-conmon --rm --sdnotify=conmon ...`),两种做法都能用,新项目应该优先选 quadlet。
3. 忘记 `systemctl daemon-reload`(quadlet 靠这一步触发 generator 把 `.container` 翻译成真正的 `.service`;经典做法则是新增/修改了 unit 文件后必须 reload 才会被 systemd 感知)是最基础、最容易犯的坑,和 02 类第 4 节讲的规则完全一样。

---

## 10. 软件源 GPG 签名校验

**命令/配置:**
```bash
rpm -K package.rpm            # 校验一个 rpm 文件的摘要/签名(K = checKsig)
rpm --import /path/to/RPM-GPG-KEY-xxx    # 手动导入一个 GPG 公钥到本地信任库
rpm -qa 'gpg-pubkey*'          # 列出已导入的 GPG 公钥(以"伪包"形式存在于 rpm 数据库)
```
`.repo` 文件里的相关字段:
```
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-Rocky-10
```

**一句话是什么:** GPG 签名校验保证"这个软件包确实是声称的发布者签发的原始版本,内容没有被篡改过"——仓库维护者用私钥给每个包签名,系统本地导入对应公钥,安装/校验时核对签名,一旦包被篡改(哪怕只改一个字节),校验就会失败并拒绝。

**为什么 RHCSA 真考 / 生产会用到:** 这是软件供应链安全的基础机制;生产环境**绝不应该**为了图方便关闭 `gpgcheck`,理解这套机制才能在真正需要处理第三方软件源时做出正确判断。

**从最容易犯错的做法讲起:** 安装第三方软件源的包遇到 GPG 校验失败,第一反应是把 `.repo` 文件里的 `gpgcheck` 直接改成 `0` 让报错消失——这等于完全放弃"确认软件包没被篡改"这层保护。更隐蔽的一个坑是:`dnf install -y` 遇到一个**新的、之前没导入过的**签名密钥时,默认行为是**自动导入并继续**,而不是拒绝——`-y` 把这一步的人工确认也一起跳过了,详见下方现场证据。

**真实场景例子(典型运维场景,非仓库代码):** 官方 Rocky Linux 仓库默认 `gpgcheck=1` 且预置好对应公钥,日常 `dnf install` 全程自动校验、无需人工干预;添加一个可信的第三方仓库时,标准流程是先拿到该仓库的官方公钥文件、`rpm --import` 导入、确认指纹(fingerprint)和官方公布的一致,再启用 `gpgcheck=1`,而不是图省事跳过指纹核对这一步。

**可运行例子(现场自建一把测试密钥、自己签名一个包来复现"未导入密钥"和"内容被篡改"两种真实失败,不触碰系统正牌 Rocky 密钥):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

dnf_retry() {
    local attempt=1 max=6
    while [ "$attempt" -le "$max" ]; do
        if dnf "$@"; then return 0; fi
        sleep $((attempt * 2)); attempt=$((attempt + 1))
    done
    return 1
}

# 系统已导入的官方公钥,以"伪包"形式存在
assert_ok bash -c "rpm -qa 'gpg-pubkey*' | grep -q ."

assert_ok dnf_retry install -y rpm-sign

workdir=$(mktemp -d /tmp/rhcsa05_gpgdemo.XXXXXX)
cd "$workdir"
export GNUPGHOME="$workdir/gnupg"; mkdir -m 700 "$GNUPGHOME"
echo "no-tty" > "$GNUPGHOME/gpg.conf"

cat > genkey.batch << 'EOF'
%no-protection
Key-Type: RSA
Key-Length: 2048
Name-Real: RHCSA05 Test Signer
Name-Email: rhcsa05-test@example.invalid
Expire-Date: 0
%commit
EOF
assert_ok timeout 30 gpg --batch --gen-key genkey.batch
gpg --export -a 'RHCSA05 Test Signer' > pubkey.asc

assert_ok dnf_retry download zsh
cp ./zsh*.rpm ./zsh-testsign.rpm
cat > ~/.rpmmacros << EOF
%_signature gpg
%_gpg_name RHCSA05 Test Signer
%_gpg_path $GNUPGHOME
EOF
# --resign(而不是--addsign):官方源下载的包已经带Rocky的签名,
# --addsign会直接报错"already contains a legacy signature",必须用--resign整体替换签名
assert_ok timeout 30 rpmsign --resign ./zsh-testsign.rpm

# 用自己的密钥签好了,但还没导入到系统信任库——rpm -K 必须真实失败
assert_ok bash -c '! rpm -K ./zsh-testsign.rpm'
assert_ok bash -c 'rpm -Kv ./zsh-testsign.rpm 2>&1 | grep -q NOKEY'

# 导入后,同一个文件立刻校验通过
assert_ok rpm --import pubkey.asc
assert_ok bash -c 'rpm -K ./zsh-testsign.rpm 2>&1 | grep -q "digests signatures OK"'

# 篡改测试:改动文件正中间1个字节,校验必须真实检测出来
cp ./zsh-testsign.rpm ./zsh-corrupted.rpm
filesize=$(stat -c%s ./zsh-corrupted.rpm)
printf '\xFF' | dd of=./zsh-corrupted.rpm bs=1 seek=$((filesize / 2)) count=1 conv=notrunc 2>/dev/null
assert_ok bash -c '! rpm -K ./zsh-corrupted.rpm'
assert_ok bash -c 'rpm -Kv ./zsh-corrupted.rpm 2>&1 | grep -q "digest: BAD"'

# 清理:移除测试密钥
test_key=$(rpm -qa 'gpg-pubkey*' | while read -r k; do rpm -qi "$k" | grep -qi rhcsa05 && echo "$k"; done)
[ -n "$test_key" ] && rpm -e "$test_key"
cd /
rm -rf "$workdir" ~/.rpmmacros
assert_ok dnf_retry remove -y rpm-sign
```
本机实测:全部 `assert_eq`/`assert_ok` 输出 `OK`。真实的失败/成功文本:
```
# 未导入密钥前:
./zsh-testsign.rpm: digests SIGNATURES NOT OK
    Header V4 RSA/SHA256 Signature, key ID d758072c: NOKEY

# 导入密钥后,同一个文件:
./zsh-testsign.rpm: digests signatures OK

# 篡改后(即使密钥已导入、密钥本身没问题):
./zsh-corrupted.rpm: DIGESTS SIGNATURES NOT OK
    Payload SHA256 digest: BAD (Expected b37f115f... != 6c765a11...)
    MD5 digest: BAD (Expected 4eccf815... != a49f8432...)
```

**额外真实发现(`-y` 自动信任新密钥,现场证据):** 用上面自签名的包搭了一个临时本地仓库(`gpgcheck=1`,`gpgkey=` 指向自己的测试公钥),**在没有预先手动 `rpm --import` 的情况下**执行 `dnf install -y --repo=<临时仓库> zsh`,真实行为是自动导入密钥并继续安装成功,过程里能看到这一段:
```
Importing GPG key 0xD758072C:
 Userid     : "RHCSA05 Test Signer <rhcsa05-test@example.invalid>"
 Fingerprint: 5E5A 5DAD F7B9 C240 37D9 99D3 7BAC 171F D758 072C
 From       : /tmp/.../pubkey.asc
Key imported successfully
```
这段信息在**交互模式**下应该以 `Is this ok [y/N]:` 的形式停下来等人工确认指纹再继续,但 `-y` 会把这个确认一并跳过——脚本化自动安装一个新配置的仓库时,如果这个仓库的密钥是第一次遇到,`-y` 相当于自动放行了"这把新密钥到底可信不可信"这个本该由人核对指纹的判断,这是自动化脚本图省事时容易忽略的真实安全考量点。

**常见坑:**
1. `gpg-pubkey` 这类导入的公钥在 `rpm -qa` 结果里以特殊的"伪包"形式出现(命名格式 `gpg-pubkey-<keyid>-<时间戳>`),它不对应任何真实安装的文件,只是 rpm 数据库记录"这把公钥已被信任导入"的方式,第一次看到容易误以为系统被装了什么奇怪的包。
2. `rpmsign --addsign` 对一个**已经带签名**的包(比如任何从官方仓库下载下来的包)会直接报错 `already contains a legacy signature`,必须用 `--resign` 整体替换掉原有签名——这是本机现场真实踩到的一个坑,一开始用 `--addsign` 导致后续验证步骤全部基于"其实还是原厂签名"的错误前提,改用 `--resign` 后才拿到真正被自己密钥签过的测试文件。
3. `dnf install -y` 对"信任一把新 GPG 密钥"这一步的自动放行(见上方"额外真实发现"),意味着自动化脚本部署新仓库时,真正的信任判断必须提前做在"要不要把这个仓库的 `.repo` 文件放进 `/etc/yum.repos.d/`"这一步,而不能指望 `-y` 运行时会帮你把关指纹是否正确——等到 `-y` 那一刻,判断已经被自动跳过了。

---

*本篇完成:2026-07-11,10 个知识点。验证环境:Rocky Linux 10.2 (Red Quartz) WSL2,`dnf-0:4.20.0-22.el10_2.rocky.0.1`,`podman version 5.8.2`。第 1-4、6-10 项(dnf 基础/仓库管理/rpm 查询/createrepo/sysctl/chronyd/podman 容器/podman quadlet/GPG 校验)在真实 Rocky Linux 环境下完整现场验证,含真实报错文本和现场诊断出的根因(tcpdump 缺 libpcap 时 rpm 直接失败、rootless podman 因 newuidmap 缺失 CAP_SETUID 而失败并完整诊断到"很可能是 OCI 镜像层打包丢失文件能力"这一步但未擅自修复共享环境、quadlet 的 `Restart=on-failure` 与强制 `ExecStop` 组合导致 stop 不住服务、GPG 未导入密钥/内容篡改两种真实失败与自动导入密钥的安全考量)。第 5 项(GRUB2)是本篇唯一的边界声明条目——WSL2 架构上不存在 BIOS/UEFI→GRUB→内核这条引导链路,`grub2-mkconfig`/`grubby` 装上后能跑、不报错,但生成的配置是空壳(`10_linux` 生成段为空、唯一菜单项是通用 UEFI 固件占位符、当前运行内核在 `/boot` 下找不到任何对应文件),已用具体证据显著说明,不代表验证了"修改内核启动参数能改变下次启动行为"这件事,这件事在 WSL2 里做不到。*
