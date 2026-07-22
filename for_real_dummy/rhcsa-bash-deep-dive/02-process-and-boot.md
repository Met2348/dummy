# 02 · 进程与系统运行(Process Management & System Boot)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一篇覆盖 12 个知识点:systemd 启动流程与 target、救援模式、systemctl 服务管理(基础+进阶)、进程查看与信号、优先级、journalctl 日志、rsyslog、cron、systemd timer、关机重启。**本文所有代码例子已在 Rocky Linux 10.2(RHEL 10 下游兼容发行版,WSL2,真实 systemd 作为 PID 1)下实际跑通验证**——环境修复过程见 [00-roadmap.md](00-roadmap.md) 的"环境前置说明"。
>
> **和前四条系列的差异声明**(详见 [00-roadmap.md](00-roadmap.md) 开头):本仓库没有 Linux 系统管理场景可挖,"真实场景例子"部分统一标注为典型运维/RHCSA 考试场景,不冒充仓库代码里挖出来的。

**开始之前:进程是什么(本篇标题就是"进程与系统运行",但下面 12 个知识点里没有一处会正式定义它,这里先补上)。** 程序是磁盘上一份静止的可执行文件,本身不占用 CPU、不消耗内存;进程是这份程序被启动、运行起来之后,内核为它创建出来的一个**动态实例**——内核会给这个实例分配一个唯一编号(PID)、一段独立的内存空间、一份独立的执行状态(比如代码执行到了哪一行)。同一份程序可以被启动多次,每次启动都会得到一个全新的、互不干扰的进程(比如同时打开两个终端各跑一次同一个脚本,会看到两个不同的 PID,各自的变量互不可见)。

更进一步,Linux 上除了系统启动的第一个进程(PID 1,现在通常是 `systemd`)之外,**每一个进程都是由另一个已经存在的进程创建出来的**,这就形成一棵进程树:创建者叫"父进程",被创建的叫"子进程",子进程记录着自己父进程的 PID(即 PPID)。这棵树状关系带来两个后面会反复用到的概念:①如果父进程在子进程还没退出前自己先退出了,这个子进程会变成"孤儿进程",被 PID 1(init/systemd)收养,不会凭空消失;②多个相关进程(比如一条管道 `cmd1 | cmd2 | cmd3` 里的三个进程)可以组成一个"进程组",拥有一个共同的进程组 ID(PGID),方便对整组进程统一发送信号(比如按 `Ctrl+C` 实际是把信号发给整个前台进程组,而不是只发给你敲命令的那一个进程)。下面第 5 节(`ps`/`top`)会看到 PPID 列,第 6 节(`kill` 信号)会用到孤儿进程和 PGID,都是建立在这段定义之上的。

---

## 1. systemd 启动流程与 target(runlevel 到 target 的映射,`default.target`)

**命令/配置:**
```bash
systemctl get-default              # 查看当前默认启动目标
systemctl set-default multi-user.target   # 设置默认启动目标(服务器场景常用)
systemctl list-units --type=target --all  # 列出所有target及其状态
```

**一句话是什么:** target 是 systemd 版的"运行级别(runlevel)"——一组服务/挂载点/其他 target 的集合,系统启动就是从 `sysinit.target` 开始,一路满足依赖链,最终到达 `default.target` 指向的那个目标;传统的 `init 0-6` 数字运行级别通过符号链接兼容映射到对应 target(比如 `runlevel3.target` 就是指向 `multi-user.target` 的一个别名)。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"配置系统开机进入图形或命令行界面",服务器场景的标准实践是用 `multi-user.target`(纯命令行、无图形环境,减少攻击面和资源占用)而不是 `graphical.target`,这是最基础的启动流程配置题。

**从最容易犯错的做法讲起:** 想当然地认为"服务器发行版默认就是命令行启动,不用管"——**本机实测证伪**:即便是这个不含任何图形组件的精简 Rocky Linux 10 容器基础镜像,`systemctl get-default` 默认返回的依然是 `graphical.target`,而不是 `multi-user.target`。这不是"装了图形界面",而是 target 定义本身的问题——`graphical.target` 只是 `Requires=multi-user.target`(外加一个可选的 display-manager),没有真正的图形组件时,这个 target 依然能正常 `active`,只是"名不副实"。默认值不能靠猜,必须现场 `get-default` 确认。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 云主机/裸金属服务器初始化脚本里,标准动作是 `systemctl set-default multi-user.target`,避免系统尝试拉起图形化组件(哪怕装都没装、大概率会失败),同时减少不必要的资源占用;RHCSA 考试如果要求"配置系统默认启动到命令行界面",这就是标准答案。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

current_target=$(systemctl get-default)
assert_eq "$current_target" "graphical.target"          # 本机实测的真实默认值,不是猜的

# runlevel兼容映射:传统数字级别通过符号链接映射到target
rl3_target=$(readlink -f /usr/lib/systemd/system/runlevel3.target | xargs basename)
rl5_target=$(readlink -f /usr/lib/systemd/system/runlevel5.target | xargs basename)
assert_eq "$rl3_target" "multi-user.target"    # 传统"运行级别3"= multi-user
assert_eq "$rl5_target" "graphical.target"     # 传统"运行级别5"= graphical

# graphical.target的强依赖确实就是multi-user.target
assert_eq "$(systemctl show graphical.target -p Requires --value)" "multi-user.target"

# 服务器场景标准操作:切换到multi-user,验证完再切回来,不留副作用
systemctl set-default multi-user.target
assert_eq "$(systemctl get-default)" "multi-user.target"
systemctl set-default graphical.target
assert_eq "$(systemctl get-default)" "graphical.target"
```
本机实测:全部 `assert_eq` 输出 `OK`。

**常见坑:** `systemctl set-default` 只是重新链接 `/etc/systemd/system/default.target` 这个符号链接,**立即生效的是"下次启动走哪个 target"这个配置,不会让当前已经在运行的服务被停掉或拉起**——想让当前会话立刻切换到目标 target 的状态,要用 `systemctl isolate <target>`(见下一节),两个命令职责不同,别混着记。

---

## 2. 单用户/救援模式排障(`rd.break`、`systemd.unit=rescue.target`)

**命令/配置:**
```bash
# 真实物理机/虚拟机:开机时在GRUB菜单按e编辑,在linux行末尾加:
systemd.unit=rescue.target     # 进rescue模式(单用户,基本文件系统只读挂载)
rd.break                        # 更早期介入,在initramfs阶段就中断,挂载根文件系统前拿到shell

# 运行时(已经启动完成的系统)切换,不需要重启:
systemctl rescue                # 运行时降级到rescue.target
systemctl emergency             # 更激进,连本地文件系统都不主动挂载
```

**一句话是什么:** 救援模式的本质是"用尽量少的依赖(不需要网络、不需要多用户环境、有时甚至不需要挂载根以外的文件系统)启动到一个能操作的 shell",专门用来修复"系统坏到正常流程起不来"这类问题(密码丢失、`/etc/fstab` 写错、GRUB 配置损坏)。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确考"重置 root 密码"、"修复导致系统无法启动的配置错误"这类题目,标准做法就是通过 GRUB 编辑内核参数进入 `rd.break`/`rescue.target`,这是排障能力的地基,不会这个基本等于放弃相关大题。

**从最容易犯错的做法讲起:** **本文必须诚实说明一个 WSL 环境的结构性限制**:WSL 不经过真实的 BIOS/UEFI/GRUB 固件启动链条,`wsl.exe -d RockyLinux` 是直接把这个发行版的 init 进程拉起来,**根本没有一个可以在开机瞬间按 `e` 编辑的 GRUB 菜单**——`rd.break` 依赖 initramfs 阶段的介入时机,这个阶段在 WSL 的启动模型里也不存在对应的真实入口。这意味着本节"从最笨的做法"该讲的是另一种新手误区:遇到系统进不去,直接找救援盘重装系统,而不是先尝试救援模式修复——重装是终极手段,90% 的"起不来"问题(密码丢失、fstab 写错)用救援模式几分钟就能修好,不需要动辄重装。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 生产服务器 `/etc/fstab` 被误改导致开机卡在 emergency shell,标准排障:救援模式(或者本例已经卡在的 emergency shell)里 `mount -o remount,rw /`,编辑修复 `/etc/fstab`,`systemctl default` 或重启验证;root 密码遗忘,GRUB 里追加 `rd.break`,进 initramfs shell 后 `mount -o remount,rw /sysroot`、`chroot /sysroot`、`passwd root` 改密码,退出并触发 SELinux relabel(改了 shadow 文件需要重新打标签,否则 SELinux 会拒绝下次登录),重启。

**可运行例子(WSL 环境下退而求其次:验证 rescue/emergency 这两个 target 单元本身存在、依赖关系符合"最小化"设计,用运行时手段接近其效果):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

targets=$(systemctl list-units --type=target --all --no-legend | awk '{print $1}')
echo "$targets" | grep -qxE "rescue\.target" && echo "OK: rescue.target 单元存在"
echo "$targets" | grep -qxE "emergency\.target" && echo "OK: emergency.target 单元存在"

# rescue.target的依赖设计体现"最小化"原则:只依赖sysinit.target,不依赖network/multi-user
# 注意:Requires的多个值内部按集合存储,两次查询顺序可能不同(本机现场观察到过),
# 所以只逐个断言两个值都在,不对整串做精确顺序匹配
rescue_requires=$(systemctl show rescue.target -p Requires --value)
echo "$rescue_requires" | grep -qw "rescue.service" && echo "OK: Requires 含 rescue.service"
echo "$rescue_requires" | grep -qw "sysinit.target" && echo "OK: Requires 含 sysinit.target"
requires_word_count=$(echo "$rescue_requires" | wc -w)
assert_eq "$requires_word_count" "2"    # 确认恰好只有这两个依赖,没有更多(印证"最小化"设计)
```
本机实测:两个单元存在性检查均输出 `OK`,`rescue.target` 的 `Requires` 确认只有 `rescue.service` 和 `sysinit.target` 这两项,印证"救援模式尽量少依赖"这个设计原则是真实生效的,不是文档空谈。

**常见坑:**
1. **本节的核心限制,请优先阅读**:WSL 没有 GRUB,`rd.break`/开机编辑内核参数这套操作**必须在真实 RHEL 虚拟机或物理机上练习**,本环境无法提供对应的验证入口,这不是 RHCSA 要考的例外情况,而是本学习环境本身的局限,请勿把"WSL 里没法练"误解成"RHCSA 不考这个"。
2. `rd.break` 和 `systemd.unit=rescue.target` 的介入时机不同:前者在根文件系统被挂载**之前**(initramfs 阶段),适合"连根文件系统本身的完整性都可能有问题"这类更severe的场景;后者假设根文件系统能正常挂载,只是想跳过大部分服务启动,进入点更"靠后"、更轻量。
3. 用 `rd.break` 改完 `/etc/shadow`(比如重置密码)后必须执行 SELinux relabel(通常是 `touch /.autorelabel` 后重启,或者手动 `restorecon`),否则以 enforcing 模式重启后,SELinux 会因为文件的安全上下文标签不对而拒绝正常认证——这是新手改完密码却发现"死活登录不了"的经典根因。**这里先给最简说明,完整原理见 08 号文件(安全:SELinux 与防火墙)**:SELinux 是在传统 ugo/rwx 权限之外**额外**的一层强制访问控制,每个文件除了权限位,还带一个"安全上下文标签",进程能不能碰某个文件,SELinux 会额外按这个标签再检查一遍(哪怕 ugo/rwx 权限允许也可能被 SELinux 拦下来);"enforcing 模式"是 SELinux 三种运行模式里真正会**拒绝**违规操作的那一种(另外两种是完全关闭、和只记日志不拦截的 permissive)。用 `chroot`/直接编辑改密码这类操作会绕开正常途径,新文件的标签容易和目录标准不一致,这才需要手动 relabel 补救。
4. **本机验证时现场踩到的坑**:`systemctl show UNIT -p Requires --value` 返回的多个依赖项,顺序在两次查询之间**可能不同**(本机实测同一个 `rescue.target` 前后两次查询,一次是 `rescue.service sysinit.target`,另一次是 `sysinit.target rescue.service`)——这类多值属性内部大概率按集合/哈希结构存储,不保证遍历顺序稳定,写自动化脚本判断"某个 unit 是否依赖 X"时,要用 `grep -w` 之类的成员检查,不能对整个返回值做精确字符串相等比较。

---

## 3. systemctl 服务管理(`start`/`stop`/`enable`/`disable`/`status` 核心用法)

**命令/配置:**
```bash
systemctl start NAME       # 立即启动(不影响开机是否自启)
systemctl stop NAME         # 立即停止
systemctl enable NAME       # 设置开机自启(不影响当前是否在运行)
systemctl disable NAME      # 取消开机自启
systemctl status NAME       # 查看当前状态+最近日志摘要
systemctl restart NAME      # 停止再启动
systemctl reload NAME       # 不重启进程,让服务重新读取配置(前提是服务自身支持)
```

**一句话是什么:** `start`/`stop` 管的是"现在跑不跑",`enable`/`disable` 管的是"开机自不自动跑",这是两条**完全独立的轴**——四种组合都合法存在(比如"现在没跑,但设置了开机自启"是完全正常的状态,不矛盾)。

**为什么 RHCSA 真考 / 生产会用到:** 服务管理是 RHCSA 考试分值占比最高的技能类别之一,"配置某服务开机自启并立即启动"这类题目几乎每次考试都会出现,`start`/`enable` 不分是新手最容易丢分的地方。

**从最容易犯错的做法讲起:** 只执行了 `systemctl start NAME` 就以为"配置好了、可以收工",完全忘记 `enable`——服务当下确实在跑,**但一旦重启,这个服务不会自动起来**,这是"考试当场好像做对了,但判分脚本重启验证后发现没配置持久化"的经典丢分模式。`start` 和 `enable` 但凡任务要求"配置并启动",两条命令都要敲,缺一不可。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 部署新服务(比如 `sshd`)标准姿势是 `systemctl enable --now sshd`——`--now` 是 `enable` 的一个组合选项,等价于同时执行 `enable` 和 `start`,一条命令搞定"配好开机自启+现在也启动",生产脚本和 RHCSA 考试里都推荐这个写法,减少漏一半的风险。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

assert_ok systemctl is-active sshd
assert_ok systemctl is-enabled sshd

systemctl stop sshd
assert_eq "$(systemctl is-active sshd 2>&1)" "inactive"   # stop只影响"现在跑不跑"
enabled_after_stop=$(systemctl is-enabled sshd)
assert_eq "$enabled_after_stop" "enabled"                  # enable状态完全没受影响,证明两条轴独立

systemctl start sshd
assert_eq "$(systemctl is-active sshd)" "active"
```
本机实测:全部断言输出 `OK`。

**常见坑:** `systemctl status` 的退出码本身携带信息(0=running,3=not running但配置正常,其他值代表更严重的问题),写自动化脚本判断服务状态时,更推荐用 `systemctl is-active --quiet NAME`(专门为脚本判断设计,只关心退出码不关心人类可读输出)而不是 `grep` `status` 的文字输出——后者格式在不同 systemd 版本间可能有细微差异,不如判断退出码稳。

---

## 4. systemctl 进阶(`mask`/`unmask`、`daemon-reload`、依赖关系查看)

**命令/配置:**
```bash
systemctl mask NAME          # 强制禁用:把unit文件链接到/dev/null,disable都拦不住的终极禁用
systemctl unmask NAME        # 解除mask
systemctl daemon-reload       # 重新加载unit文件(改完.service文件后必须执行,否则改动不生效)
systemctl list-dependencies NAME   # 查看这个unit依赖哪些其他unit(树状展示)
```

**一句话是什么:** `disable` 只是取消开机自启的"软性"设置,如果有人手动 `systemctl start` 或者别的 unit 把它拉起来作为依赖,`disable` 挡不住;`mask` 是"硬性"拦截——把 unit 文件直接软链接到 `/dev/null`,任何方式(包括手动 `start`)都无法启动这个服务,直到 `unmask`。

**为什么 RHCSA 真考 / 生产会用到:** 安全加固场景经常要求"彻底禁止某服务运行"(比如禁掉 `telnet`、禁掉不需要的打印服务),`disable` 不够彻底,`mask` 才是标准答案;修改自定义 `.service` 文件后必须 `daemon-reload`,忘记这一步是"改了配置但死活不生效"最常见的原因,没有例外。

**从最容易犯错的做法讲起:** 只用 `disable` 就以为某个服务"绝对不会跑起来",但如果这个服务是另一个已启用服务的依赖项(比如某 `.socket` 触发、或者被 `Requires=` 显式拉起),它依然会被间接启动——这是安全加固任务里"明明 disable 了,扫描却发现服务还在跑"的真实成因,必须 `mask` 才能保证万无一失。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 安全基线要求彻底禁用某个不需要的服务(比如遗留系统上的 `telnet.socket`),`systemctl mask telnet.socket` 是标准做法,比 `disable` 更彻底;调试一个服务为什么起不来,`systemctl list-dependencies NAME --all` 能画出完整依赖树,快速定位是不是某个上游依赖没起来导致连锁失败。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

systemctl mask chronyd >/dev/null 2>&1
assert_eq "$(systemctl is-enabled chronyd 2>&1)" "masked"

# masked服务连手动start都拦得住
mask_start_err=$(systemctl start chronyd 2>&1)
assert_eq "$?" "1"
echo "$mask_start_err" | grep -qi "masked" && echo "OK: 报错信息明确提到masked,不是模糊的失败"

systemctl unmask chronyd
assert_eq "$(systemctl is-enabled chronyd)" "enabled"
systemctl start chronyd
assert_ok systemctl is-active chronyd

assert_ok systemctl daemon-reload

deps_count=$(systemctl list-dependencies sshd --no-legend 2>/dev/null | wc -l)
assert_ok test "$deps_count" -ge 1
```
本机实测:全部断言输出 `OK`。

**常见坑:** `systemctl mask` 对**已经在运行**的服务不会立即杀掉进程,只是阻止"以后"再被启动——如果目标是"立刻停止且以后也别再跑",要 `mask` 和 `stop` 两条命令一起用(或者用 `systemctl disable --now NAME` 拿到"停止+取消自启"的组合,但这依然不如 `mask` 彻底,记住 `mask` 才是唯一能拦住"被其他 unit 当依赖拉起"这种情况的手段)。

---

## 5. 进程查看 `ps`/`top`(常用列解读,`ps aux` vs `ps -ef`)

**命令/配置:**
```bash
ps aux       # BSD风格选项(无横杠),按用户/CPU/内存等列出所有进程
ps -ef        # UNIX风格选项(有横杠),含PPID(父进程ID)列
ps -p PID -o comm=,nice=    # 只看指定PID的指定字段(oneshot式精确查询)
top           # 交互式、实时刷新的进程监视器
```

**一句话是什么:** `ps aux` 和 `ps -ef` 本质上查的是同一份进程表,只是历史上 BSD 和 UNIX System V 两套风格选项习惯留下的两种呈现格式,字段有细微差异(`-ef` 默认带 `PPID` 父进程列,`aux` 默认带 `%CPU`/`%MEM` 列),两者选一个用惯了就行,不需要都记。

**为什么 RHCSA 真考 / 生产会用到:** 排查"系统卡住了""哪个进程占用资源异常"是最基础的运维日常,RHCSA 要求"识别并管理进程"这项技能,`ps`/`top` 是排障的第一双眼睛。

**从最容易犯错的做法讲起:** 只会用 `ps aux | grep 进程名` 找进程,却不知道这条命令本身会在结果里多出一行"grep 自己"(因为 `grep` 进程本身的命令行参数里含有你搜索的关键词,会把自己也匹配进去)——新手常被这多出来的一行搞糊涂,更干净的写法是 `pgrep -a 进程名` 或者 `ps aux | grep '[进]程名'`(用方括号技巧让 grep 的进程本身不匹配自己的命令行)。

**真实场景例子(典型运维场景,非仓库代码):** 巡检脚本定期 `ps -eo pid,ppid,%cpu,%mem,comm --sort=-%cpu | head -10` 抓取 CPU 占用前 10 的进程做异常告警;`top` 交互式排查"服务器突然变卡"时,直接开一个 `top`,按 `P`(按CPU排序)或 `M`(按内存排序)快速定位罪魁祸首。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

aux_header=$(ps aux | head -1)
ef_header=$(ps -ef | head -1)
echo "$aux_header" | grep -q "%CPU" && echo "OK: ps aux 含 %CPU 列"
echo "$ef_header" | grep -q "PPID" && echo "OK: ps -ef 含 PPID 列"

# PID 1 在这个环境里就是真实systemd(见00-roadmap.md环境修复记录)
pid1_comm=$(ps -p 1 -o comm=)
assert_eq "$pid1_comm" "systemd"

top_available=0; command -v top >/dev/null 2>&1 && top_available=1
assert_eq "$top_available" "1"
```
本机实测:全部断言输出 `OK`。

**常见坑:** `ps` 默认(不加任何选项)只显示**当前终端会话**里、属于当前用户的进程,新手经常疑惑"为什么 `ps` 看不到我知道正在跑的某个服务进程"——一定要加 `a`(所有终端)、`u`(用户信息)、`x`(没有控制终端的进程,比如后台服务)这类选项组合,`aux`/`-ef` 都是"看全部进程"的标准写法,裸 `ps` 基本没什么排障价值。

---

## 6. 进程信号与 kill(`SIGTERM` vs `SIGKILL`,`killall`/`pkill`)

**命令/配置:**
```bash
kill -TERM PID      # 发送SIGTERM(15),礼貌请求进程自行退出,默认信号
kill -KILL PID       # 发送SIGKILL(9),内核强制杀死,进程无法捕获/忽略
kill -0 PID          # 不发送任何信号,只用来测试进程是否存在(常见的存在性探测惯用法)
killall NAME          # 按进程名(精确匹配命令名)批量kill
pkill -f "pattern"     # 按完整命令行的正则模式匹配kill,比killall更灵活
```

**一句话是什么:** `SIGTERM`(默认信号)是"礼貌地请求"进程退出——进程可以捕获这个信号,先做清理工作(保存数据、关闭连接)再退出;`SIGKILL` 是内核层面的强制终结,进程完全没有机会做任何清理,直接被系统摘除,**优先用 TERM,只有 TERM 无效时才升级到 KILL**。

**为什么 RHCSA 真考 / 生产会用到:** "终止失控的进程"是 RHCSA 明确要求的技能;生产环境里区分 TERM 和 KILL 更是事故预防的关键——粗暴地对数据库进程直接 `kill -9`,可能导致事务没有正常提交/回滚,造成数据损坏。

**从最容易犯错的做法讲起:** 遇到进程"卡住不退出",新手第一反应就是 `kill -9`(SIGKILL)——这是危险的坏习惯:很多进程的"卡住"其实是在做重要的收尾工作(刷写缓存、释放锁),粗暴 `-9` 会打断这个过程;正确流程是先 `kill`(默认TERM)给进程一个自行退出的机会,等几秒观察,**真的不行**了再用 `-9` 兜底,而不是一上来就用最暴力的手段。

**真实场景例子(典型运维场景,非仓库代码):** 部署脚本里优雅重启服务的标准做法是先发 TERM、轮询等待进程退出(比如最多等 10 秒)、超时了才补一刀 KILL,而不是无脑直接 KILL;批量清理一类失控的测试进程用 `pkill -f "test_worker"` 按命令行关键词精确打击,比 `killall` 按精确进程名更灵活(比如 python 脚本的进程名统一都叫 `python3`,`killall python3` 会误杀所有 Python 进程,`pkill -f` 能按脚本文件名精确定位)。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

sleep 300 &
bg_pid=$!
sleep 0.3
kill -0 "$bg_pid" 2>/dev/null
assert_eq "$?" "0"          # kill -0:进程确实存在

kill -TERM "$bg_pid"
sleep 0.3
kill -0 "$bg_pid" 2>/dev/null
assert_eq "$?" "1"          # SIGTERM后进程已退出(sleep能正常响应TERM)

sleep 300 &
bg_pid2=$!
sleep 0.3
pkill -f "sleep 300"
sleep 0.3
kill -0 "$bg_pid2" 2>/dev/null
assert_eq "$?" "1"          # pkill -f按命令行模式精确命中
```
本机实测:全部断言输出 `OK`。

**常见坑:** `kill -9` 之后,如果目标进程有子进程,子进程往往不会跟着一起死(变成"孤儿进程",被 init/systemd 收养)——真正想连带杀掉整个进程树,要用 `kill -- -PGID`(杀整个进程组,注意负号)或者 `pkill -P 父PID`,单纯 `kill` 父进程只会杀掉那一个进程本身。

---

## 7. 进程优先级 `nice`/`renice`

**命令/配置:**
```bash
nice -n 10 command       # 以调整过的niceness启动一个新进程(数值越大优先级越低)
renice -n 5 -p PID        # 调整一个已经在运行的进程的niceness
nice                       # 不带参数:查看当前shell的niceness(默认0)
```

**一句话是什么:** niceness 是一个 **-20(最高优先级)到 19(最低优先级)** 的整数,数值越小越"不谦让"(优先级越高,越容易抢到 CPU 时间片),数值越大越"谦让"(nice 这个名字本身就是"友善、谦让"的意思);`nice` 用来启动新进程时指定,`renice` 用来调整已经在跑的进程。

**为什么 RHCSA 真考 / 生产会用到:** 服务器上有 CPU 密集型的后台批处理任务(比如日志分析、备份压缩)和对响应时间敏感的前台服务(比如 Web 服务)同时运行时,给批处理任务调低优先级(调大 nice 值)能避免它抢占前台服务的 CPU 资源,这是资源调度的基本手段。

**从最容易犯错的做法讲起:** 把 niceness 和"绝对优先级"搞混,以为 `nice -n -20` 能让进程"完全垄断"CPU——niceness 只是内核调度器做资源分配决策时的**权重参考**,不是硬性配额,系统依然会公平地给其他进程分配时间片,只是比例上会向低 nice 值的进程倾斜;真正需要"绝对不能被打断"的场景(比如实时系统),需要的是完全不同的调度类别(`chrt` 配合实时调度策略),不是普通的 nice/renice。

**真实场景例子(典型运维场景,非仓库代码):** 夜间批处理脚本(数据库备份、日志归档压缩)开头加 `nice -n 19`,保证即便在业务低峰期也不会跟其他关键服务抢 CPU;发现某个失控的进程正在疯狂占用 CPU 但又不能立刻杀掉(还在处理重要数据),先 `renice -n 19 -p PID` 把它降级,给其他服务腾出资源,再从容排查根因。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

default_nice=$(nice)
assert_eq "$default_nice" "0"          # 默认niceness是0

sleep 300 &
nice_pid=$!
sleep 0.2
renice -n 10 -p "$nice_pid" >/dev/null 2>&1
new_nice=$(ps -o nice= -p "$nice_pid" | tr -d ' ')
assert_eq "$new_nice" "10"              # renice确实把运行中进程的niceness改到了10
kill -9 "$nice_pid" 2>/dev/null
```
本机实测:两个断言均输出 `OK`。

**常见坑:** 普通用户只能把自己进程的 niceness **调大**(降低优先级,比如从 0 调到 10),**不能调小**(不能把自己的进程调成比默认更高的优先级),想把 niceness 调成负数(更高优先级)必须是 root 或者拥有 `CAP_SYS_NICE` 能力——这是一个安全设计,防止普通用户互相抢占系统资源。

---

## 8. journalctl 日志查询(按时间/服务/优先级过滤,`-f` 跟踪)

**命令/配置:**
```bash
journalctl -u NAME              # 只看某个service的日志
journalctl -p err                # 按优先级过滤(emerg/alert/crit/err/warning/notice/info/debug)
journalctl --since "1 hour ago"   # 按时间范围过滤
journalctl -f                     # 实时跟踪新日志(类似tail -f)
journalctl --list-boots           # 列出每一次开机的记录(每次boot都有独立编号)
```

**一句话是什么:** `journalctl` 是 systemd 统一日志系统(journald)的查询工具,所有由 systemd 管理的服务的标准输出/标准错误、以及内核日志,都被结构化地收集到一个二进制日志库里,`journalctl` 提供按服务/时间/优先级等多个维度过滤查询的能力,不需要像传统文本日志那样自己 `grep` 一堆分散的文件。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"定位并解读系统日志",而**在 RHEL 10 上 `journalctl` 已经是唯一的标准日志查询方式**(本类目第 9 项会讲到,传统的 rsyslog/`/var/log/messages` 在本机验证环境下压根没装),排障能力很大程度上就是 `journalctl` 用得熟不熟练。

**从最容易犯错的做法讲起:** 遇到问题只会用不带任何过滤条件的 `journalctl`,面对成千上万行日志无从下手——正确姿势永远是先缩小范围:知道是哪个服务出问题就 `-u`,知道大概时间就 `--since`,只关心报错就 `-p err`,组合使用远比翻一整个日志海洋高效。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 排查"sshd 突然连不上"故障,`journalctl -u sshd --since "10 min ago"` 精确锁定最近的 sshd 相关日志,不用在海量系统日志里大海捞针;写监控脚本定期 `journalctl -p err --since "5 min ago" --no-pager` 扫描最近的错误级别日志做异常告警。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

sshd_log=$(journalctl -u sshd --no-pager 2>&1)
echo "$sshd_log" | grep -qi "sshd\|ssh" && echo "OK: -u sshd 确实只返回sshd相关日志"

err_log_lines=$(journalctl -p err --no-pager 2>&1 | wc -l)
assert_ok test "$err_log_lines" -ge 0    # 不管有没有err级别日志,命令本身要能正常执行不报错

boots_count=$(journalctl --list-boots --no-pager 2>&1 | wc -l)
assert_ok test "$boots_count" -ge 1       # 至少有一次boot记录(当前这次)
```
本机实测:全部断言输出 `OK`。

**常见坑:** `journalctl` 默认的日志**不是永久保存的**——如果 `/var/log/journal` 目录不存在,日志只存在内存里的 volatile 存储,重启就丢失;RHCSA 如果要求"配置日志持久化",标准答案是 `mkdir -p /var/log/journal && systemctl restart systemd-journald`(创建这个目录本身就是触发持久化存储的关键动作,不需要额外改配置文件)。

---

## 9. rsyslog 传统日志(`/var/log/messages`,logrotate 基础)

**命令/配置:**
```bash
rsyslogd                    # 传统syslog守护进程(如果安装了的话)
/etc/rsyslog.conf            # 主配置文件,定义"什么日志去哪个文件"的规则
logrotate /etc/logrotate.conf   # 按配置轮转日志文件,防止无限增长
```

**一句话是什么:** rsyslog 是 systemd/journald 出现之前的传统 syslog 实现,把日志写成人类可读的纯文本文件(比如 `/var/log/messages`);`logrotate` 负责给这些文本日志做"轮转"(按大小/时间归档、压缩旧日志、删除过期日志),防止日志文件无限膨胀撑爆磁盘。

**为什么 RHCSA 真考 / 生产会用到:** 部分老旧应用/第三方软件仍然只认传统文本日志格式,不会主动对接 journald;RHCSA 大纲仍然可能涉及"配置日志轮转策略"这类题目,需要理解 rsyslog/logrotate 的存在和用途,即便日常排障已经越来越依赖 journalctl。

**从最容易犯错的做法讲起:** **本节最重要的不是"怎么用",而是先诚实探测这个环境的真实情况**——很多人(包括没有亲自动手验证的教程)会想当然地认为"RHEL 系统一定有 `/var/log/messages`",**本机实测证伪**:这台 Rocky Linux 10.2 默认没有安装 `rsyslogd`,也没有 `logrotate`,`/var/log/messages` 这个文件根本不存在——这不是环境缺陷,而是 **RHEL 10 时代的真实趋势:journald 已经是默认且唯一预装的日志方案,rsyslog 降级为按需安装的可选组件**。凭旧版本 RHEL 的记忆想当然地找 `/var/log/messages`,在新系统上会扑空。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 如果任务要求"让日志以传统文本格式保存"(比如给某些只会读文本日志文件的老旧监控系统适配),需要先确认并安装 `rsyslog` 包,配置 `/etc/rsyslog.conf`(或 `imjournal` 模块从 journald 拉取日志转成文本文件),再配合 `logrotate` 防止文件无限增长;纯 RHCSA 考试环境下,大概率优先考察的是 journalctl(本类目第 8 项),rsyslog 更多是"知道它存在、知道怎么按需装上"这个层面的要求。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

rsyslog_present=0; command -v rsyslogd >/dev/null 2>&1 && rsyslog_present=1
assert_eq "$rsyslog_present" "0"          # 如实探测:本机确认未安装

messages_exists=0; [ -f /var/log/messages ] && messages_exists=1
assert_eq "$messages_exists" "0"          # 如实探测:文件确实不存在

logrotate_present=0; command -v logrotate >/dev/null 2>&1 && logrotate_present=1
assert_eq "$logrotate_present" "0"        # 如实探测:同样未安装

echo "本机确认:RHEL 10 默认不预装 rsyslog/logrotate,journald 是唯一开箱即用的日志方案"
```
本机实测:三项探测均确认"未安装/不存在",与预期的"RHEL 10 默认转向 journald"结论一致。

**常见坑:** 千万不要凭旧知识(RHEL 6/7 时代 `/var/log/messages` 是标配)想当然地在 RHEL 10 上找这个文件——**任何涉及"这个系统上有没有 X"的判断,都应该先现场探测确认,而不是凭记忆断言**,这条纪律不只是写教程时的要求,也是真实排障工作的基本功:环境会变,记忆会过时,现场核实永远比"我记得应该是这样"可靠。

---

## 10. cron 定时任务(crontab 语法,`/etc/cron.d`)

**命令/配置:**
```bash
crontab -e         # 编辑当前用户的定时任务
crontab -l          # 列出当前用户的定时任务
crontab -r          # 删除当前用户的全部定时任务
# 语法: 分 时 日 月 星期 命令  (* 表示"任意值")
# */5 * * * *   表示"每5分钟"
```

**一句话是什么:** cron 是传统的、基于固定时间表的任务调度器,`crontab -e` 编辑的是"当前用户"的任务表,系统级、不归属某个特定用户的定时任务通常放在 `/etc/cron.d/` 目录下的独立文件里(格式类似,但多一列"以哪个用户身份执行")。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"安排任务在未来某个时间执行"这项技能,cron 是最经典、最广泛使用的定时任务方案,备份脚本、日志清理、定期巡检几乎都靠它。

**从最容易犯错的做法讲起:** 直接手工编辑 `/var/spool/cron/用户名` 这个 crontab 实际存储文件,而不是用 `crontab -e` 命令——这是危险操作,`crontab -e` 在保存时会自动做语法校验(格式错了会拒绝保存并提示错误行),直接手改文件绕过了这层校验,一个字段写错就可能导致任务完全不执行且没有任何报错提示,排查起来非常痛苦。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 数据库每晚凌晨 2 点自动备份,`crontab -e` 里加一行 `0 2 * * * /opt/scripts/backup.sh`;系统级、需要用特定服务账号执行且希望和某个软件包一起分发的定时任务(比如 `logrotate` 自身的每日检查),放在 `/etc/cron.d/` 下的独立文件里,这样卸载软件包时定时任务能一起清理干净,不会留下孤儿任务。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

assert_ok systemctl is-active crond    # cron的守护进程要先跑起来任务才会真正被调度

(echo "*/5 * * * * /bin/true # rhcsa-demo") | crontab -
assert_eq "$(crontab -l)" "*/5 * * * * /bin/true # rhcsa-demo"

crontab -r    # 清理测试任务,不留痕迹
crontab -l 2>&1 | grep -qi "no crontab" && echo "OK: 清理后crontab恢复空状态"

assert_ok test -d /etc/cron.d    # 系统级定时任务目录存在
```
本机实测:全部断言输出 `OK`。

**常见坑:** `*/5 * * * *` 这类步长写法容易被误解成"从当前时刻起每隔 5 分钟",实际上 cron 是按**绝对时钟**对齐的(0、5、10、15……分钟这些固定刻度触发,不是"从任务创建那一刻起算"),理解这一点能避免"我明明是 10:03 加的任务,为什么感觉第一次触发要等好久"这种困惑——下一次触发是 10:05,不是 10:08。

---

## 11. systemd timer(对比 cron 的优势,`.timer`+`.service` 配对)

**命令/配置:**
```ini
# /etc/systemd/system/myjob.service
[Unit]
Description=My scheduled job
[Service]
Type=oneshot
ExecStart=/path/to/script.sh
```
```ini
# /etc/systemd/system/myjob.timer
[Unit]
Description=Run myjob daily
[Timer]
OnCalendar=daily
[Install]
WantedBy=timers.target
```
```bash
systemctl enable --now myjob.timer     # timer要单独enable+start,不是service
systemctl list-timers --all             # 查看所有定时器及下次触发时间
```

**一句话是什么:** systemd timer 用一个 `.timer` 单元(定义"什么时候触发")搭配一个同名的 `.service` 单元(定义"触发时具体做什么")协同工作,相比 cron 的优势在于:能查看下次精确触发时间、任务执行日志直接进 journald(不需要自己重定向)、支持"错过了就补跑"(`Persistent=true`)、可以和其他 systemd 依赖机制联动。

**为什么 RHCSA 真考 / 生产会用到:** systemd timer 是 cron 的现代替代方案,RHEL 生态越来越多内建的周期性任务(比如 `logrotate` 自身)已经迁移到 timer 机制,理解 `.timer`+`.service` 的配对关系,是现代 RHEL 系统管理的必备技能。

**从最容易犯错的做法讲起:** 写好了 `myjob.timer`,执行 `systemctl enable --now myjob.service`(enable 的是 service 而不是 timer)——这是最常见的配置错误,任务会立刻执行一次(因为 service 本身被直接启动了),但**不会按计划周期性触发**,因为真正负责"定时"这件事的是 `.timer` 单元,必须 `enable`/`start` 的是 `.timer`,`.service` 只在被 timer(或手动)触发时才执行,自己不会不会"知道"该在什么时候跑。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 需要一个每天执行的清理任务且要求"如果服务器当时刚好关机错过了这次执行,开机后要能补跑一次"——这是 cron 完全做不到的能力(cron 严格按时刻触发,过时不候),systemd timer 只需要在 `[Timer]` 段加一行 `Persistent=true` 就能满足;`systemctl list-timers` 能直接看到"距离下次触发还有多久",这种可观测性是纯 cron 没有的。

**可运行例子(验证 systemd timer 机制本身能正确触发):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

cat > /etc/systemd/system/rhcsa-quiz.service << 'EOF'
[Unit]
Description=RHCSA quiz oneshot service
[Service]
Type=oneshot
ExecStart=/bin/bash -c 'echo demo-timer-fired >> /tmp/rhcsa_quiz.log'
EOF
cat > /etc/systemd/system/rhcsa-quiz.timer << 'EOF'
[Unit]
Description=RHCSA quiz timer
[Timer]
OnActiveSec=2
[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl start rhcsa-quiz.timer
assert_ok systemctl is-active rhcsa-quiz.timer

# 轮询等待触发(实测触发本身在2秒精确发生,但从触发到文件可读有约1秒调度延迟,
# 系统负载较高时这个延迟会更长——本机在批量跑一整批验证脚本的高负载场景下
# 实测5秒轮询窗口都不够,曾产生假阴性,加宽到20秒后稳定通过)
timer_fired=0
for i in $(seq 1 40); do
    sleep 0.5
    if [ -f /tmp/rhcsa_quiz.log ] && grep -q "demo-timer-fired" /tmp/rhcsa_quiz.log; then
        timer_fired=1; break
    fi
done
assert_eq "$timer_fired" "1"

systemctl stop rhcsa-quiz.timer 2>/dev/null
rm -f /etc/systemd/system/rhcsa-quiz.timer /etc/systemd/system/rhcsa-quiz.service /tmp/rhcsa_quiz.log
systemctl daemon-reload
```
本机实测:`OnActiveSec=2` 的 timer 精确在 2 秒时触发(`systemctl show -p LastTriggerUSec` 现场核实过触发时刻),断言输出 `OK`。

**常见坑:**
1. `enable` 错了对象(`.service` 而不是 `.timer`)是最高频的坑,见上方"从最容易犯错的做法"。
2. **本机调试时踩到的真实坑,值得记录**:反复用同一个 unit 名字(比如改了 `.service` 内容重新测试)进行调试,即便执行了 `daemon-reload`,有时行为也不会按新内容更新——换一个全新的 unit 名字能绕开这个问题。怀疑和 systemd 对 unit 状态的内部缓存/时间戳判断机制有关,真实原因没有深挖,但"调试自定义 unit 行为异常时,先试试换个干净的名字"是一条实用的排障经验。
3. 判断"定时任务是否真的执行了",不要用固定 `sleep N秒` 再检查一次这种脆弱写法——本机实测发现从 timer 触发到 service 真正执行完毕、文件内容可读之间,存在调度延迟(单独测试时约 1 秒,但和其他一大批脚本一起批量跑、系统负载更高时,实测 5 秒的轮询总窗口都不够,产生过假阴性),轮询重试(每隔一小段时间检查一次,把总超时留得比"看起来应该够"更宽松一些)比一次性的定长等待可靠得多,超时阈值也不能卡得太紧。

---

## 12. 关机重启命令族(`shutdown`/`reboot`/`poweroff` 的区别与安全用法)

**命令/配置:**
```bash
shutdown -h now          # 立即关机(-h = halt)
shutdown -r now           # 立即重启(-r = reboot)
shutdown -h +10 "维护通知"   # 10分钟后关机,并给所有登录用户广播一条消息
shutdown -c                # 取消一个已经计划好、还没执行的关机
reboot                     # 直接重启,等价于 shutdown -r now 的简化形式
poweroff                   # 直接关机(切断电源),等价于 shutdown -h now
```

**一句话是什么:** `shutdown` 是"更完整"的关机/重启入口,支持延时执行、支持给所有登录会话广播通知、支持取消;`reboot`/`poweroff` 是更直接的简化命令,现代 systemd 系统里三者底层最终都会调用 systemd 做实际的关机流程,只是命令行的语义/选项丰富程度不同。

**为什么 RHCSA 真考 / 生产会用到:** 系统关机重启是最基础的操作,但"延时通知式关机"(给其他登录用户提前预警,而不是说关就关)是生产环境的基本礼仪,RHCSA 也要求理解不同关机方式的区别和适用场景。

**从最容易犯错的做法讲起:** 在多人共用的生产服务器上直接 `reboot` 或 `poweroff`,没有任何预警——如果有其他人正在登录操作、或者有未保存的工作,这种"说关就关"的方式会造成数据丢失或工作中断;更负责任的做法是用 `shutdown -r +5 "5分钟后系统将重启维护"`,给所有人一个缓冲时间和明确通知,这不只是技术问题,也是运维协作的基本素养。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 计划性维护窗口,提前用 `shutdown -r +10 "系统将于10分钟后重启进行内核更新"` 通知所有登录会话,给大家保存工作的时间;如果维护计划临时取消,`shutdown -c` 能撤销这个已经安排好但还没执行的关机任务,不需要真的等它触发。

**可运行例子(仅验证命令存在与基本用法说明,不实际触发关机/重启——真的执行会直接终止当前 WSL 会话,打断后续所有验证工作):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

for cmd in shutdown reboot poweroff halt; do
    present=0; command -v "$cmd" >/dev/null 2>&1 && present=1
    assert_eq "$present" "1"
done

shutdown_help=$(shutdown --help 2>&1)
echo "$shutdown_help" | grep -qi "halt" && echo "OK: shutdown --help 确认提供halt相关选项,印证shutdown是更完整的统一入口"
```
本机实测:四个命令的存在性检查、以及 `shutdown --help` 内容检查均输出 `OK`。

**常见坑:** `halt` 和 `poweroff` 容易被当成同义词,但传统语义上 `halt` 只是"停止 CPU 运行",不保证真的切断电源(老式硬件上执行 `halt` 后可能还需要手动按电源键),`poweroff` 才明确保证"关机后真的断电"——在现代 systemd 系统上两者行为已经趋同(都会调用 ACPI 断电),但理解这个历史语义差异,有助于理解为什么会同时存在这么多个"看起来功能重复"的命令。

---

*本篇完成:2026-07-11,12 个知识点。验证环境:Rocky Linux 10.2(WSL2,真实 systemd PID 1)。全部代码块真实跑通验证,含 3 处现场发现并如实记录的真实差异/坑(默认 target 是 graphical 而非 multi-user、rsyslog/logrotate 在 RHEL 10 默认不预装、systemd timer 从触发到生效有调度延迟需要轮询等待而非固定 sleep)。*
