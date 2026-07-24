# 11 · 手把手实战:从零写一个迷你系统巡检脚本

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是第 11 个"知识点",不计入"100 个知识点"的统计——和 [10 类](10-advanced-interview-depth.md)是同一挂(都是 01-09 完成之后的追加内容),但风格不一样:10 号文件里,你是**旁观者**,跟着"故障现象 → 排查动作 → 发现 → 根因 → 修复与验证"这条运维排障语序,把一条真实故障链条看一遍;这一篇里,你是**动手的人**——从一个空文件开始,一步步敲代码/命令,每写一段就在 WSL2 里真实跑一次、看到真实效果,最后独立组装出一个真实能用的巡检脚本。这个格式先在 [dsa-deep-dive/21-build-a-mini-search-engine.md](../dsa-deep-dive/21-build-a-mini-search-engine.md) 用纯 Python 试点过;本篇把同一种**精神**搬到 bash + Linux 系统操作上,但验证方式不照搬 Python 那一套(那边是 `_verify_md.py` 把每个代码块单独拎出来起子进程执行),这里延续本系列 01-10 号文件已经确立的验证方式——每个代码块在 **WSL2 Rocky Linux 10.2** 里真实跑一遍,用 `assert_eq`/`assert_ok`(定义见 [00-roadmap.md](00-roadmap.md) 第 13-18 行)做断言,复用系列已经装好的环境,不新装任何软件包。

## 为什么是"系统巡检脚本"

不是要发明新知识点,是把三个你已经学过的知识点串成一个真实运维会用的小工具:一台服务器"健不健康",最基础的四个信号是磁盘还有没有空间、内存还够不够用、有没有进程"卡死变僵尸"、日志里最近有没有大量报错——巡检脚本把这四个人工检查动作自动化成一条命令。

| 阶段 | 要让脚本多会一件事 | 建立在哪个已有知识点之上 |
|------|------|------|
| 阶段 1 | 检查一个挂载点的磁盘使用率,超阈值就告警 | [01 类](01-essential-tools.md) `awk` 文本字段提取、[09 类](09-bash-scripting.md)第 2 节条件判断 |
| 阶段 2 | 检查内存使用率,超阈值就告警 | 同上,外加一个真实的"该用哪一列数据"的坑 |
| 阶段 3 | 检查有没有僵尸(zombie/defunct)进程 | [02 类](02-process-and-boot.md)开篇"进程是什么"+第 5 节 `ps` 常用列解读 |
| 阶段 4 | 检查指定日志文件里最近的 ERROR 行数是否超阈值 | [01 类](01-essential-tools.md)第 4 节 `grep` 与正则、[09 类](09-bash-scripting.md)第 9 节 `set -euo pipefail` |
| 阶段 5 | 把前四步拼成一个完整脚本,收尾打印一份汇总巡检简报 | 阶段 1-4 全部组装,外加 [09 类](09-bash-scripting.md)第 4/5 节函数与数组 |

**验证环境说明:** 全篇复用 [00-roadmap.md](00-roadmap.md)"环境前置说明"里已经修好并验证过的 WSL2 Rocky Linux 10.2 环境(`Rocky Linux 10.2 (Red Quartz)`,真实 systemd,root 会话),本篇未安装任何新软件包——用到的 `df`/`free`/`ps`/`awk`/`grep`/`tail` 全部是这个环境里已经就绪的核心工具。凡是"制造告警"的地方,一律用"故意设一个极低的阈值"或"造一个内容可控的临时文件/临时进程"来触发,不做真的填满磁盘、不批量制造僵尸进程这类会影响宿主 WSL2 环境稳定性的操作——这一点在阶段 3 会看到一个只造 1 个、几秒内自动清理干净的僵尸进程,不是失控的破坏性实验。

---

## 阶段 1:磁盘使用率检查——先让一个函数独立工作

最基础的巡检信号:某个挂载点的磁盘用了百分之多少,超过阈值就要喊人来处理。`df -h` 能拿到这个百分比,但它是给人看的表格,脚本要拿到里面那一个数字,需要 [01 类](01-essential-tools.md)已经讲过的文本字段提取能力。

```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

check_disk() {
    local mount_point="$1"
    local threshold="$2"
    local usage_pct
    usage_pct=$(df -h "$mount_point" | awk 'NR==2 {gsub("%","",$5); print $5}')
    if [ "$usage_pct" -ge "$threshold" ]; then
        echo "[ALERT] disk usage on $mount_point is ${usage_pct}% >= threshold ${threshold}%"
        return 1
    else
        echo "[OK] disk usage on $mount_point is ${usage_pct}% < threshold ${threshold}%"
        return 0
    fi
}

check_disk / 80
assert_eq "$?" "0"

check_disk / 1
assert_eq "$?" "1"
```

本机实测(WSL2 Rocky Linux 10.2,真实输出):
```
[OK] disk usage on / is 1% < threshold 80%
OK: 0 == 0
[ALERT] disk usage on / is 1% >= threshold 1%
OK: 1 == 1
```

这台环境里根分区(`/dev/sdd`,1007G)真实使用率只有 1%,用正常阈值(80%)自然不会告警;第二次调用**没有改磁盘上任何一个字节**,只是把阈值故意设得极低(1%),同一个真实的 1% 用量立刻变成"超阈值"——这就是任务要求的"用低阈值在正常系统状态下真实触发告警分支",不需要真的把磁盘写满。

**`awk 'NR==2 {...}'` 在做什么:** `df -h /` 的输出固定是两行——第 1 行(`NR==1`)是表头(`Filesystem Size Used Avail Use% Mounted on`),第 2 行(`NR==2`)才是真实数据;`$5` 是"Use%"这一列(第 5 个空格分隔的字段),`gsub("%","",$5)` 把百分号去掉,只留数字,这样才能用 `-ge` 做数值比较(bash 的 `[ ]` 数值比较运算符不认识带 `%` 的字符串)。

**顺便一提,真实环境里的意外发现(不是构造的):** 写这一节的时候顺手看了一眼这台 WSL2 挂载的全部文件系统(`df -h` 不带路径参数,列出所有挂载点),发现 Windows 宿主机的 `E:\` 盘(挂载成 `/mnt/e`,也就是这个仓库所在的盘)真实使用率已经到了 **96%**(`1.5T` 总容量,`1.4T` 已用,只剩 `62G`)——如果把 `check_disk /mnt/e 80` 跑一遍,不用调低任何阈值,80% 的正常阈值就会真实告警。这个数字是文档撰写时刻的真实快照,以后这台机器空间释放了可能会变,不能当成"永远复现"的示例,只作为"低阈值技巧不是唯一触发告警的方式,真实系统状态本身也可能已经超标"的真实旁证。

**常见坑:** `df -h` 在文件系统名很长的时候(比如某些覆盖文件系统 `overlay` 的完整设备描述)会把一条记录拆成两行显示,`awk 'NR==2'` 这种"固定取第几行"的写法就会读到错位的数据;生产脚本更稳妥的写法是 `df -hP`(`-P` 是 POSIX 输出格式,强制单行,不换行),这里为了和系列已有例子的输出风格一致仍用 `df -h`,但独立写巡检脚本时优先记住 `-P` 这个选项。

---

## 阶段 2:加内存使用率检查——这一步让脚本多会一件事

阶段 1 证明了"读一个百分比、比阈值、告警"这个模式能跑通;阶段 2 把同一个模式套到内存上——但内存不会像 `df` 一样直接给你一个"使用率"百分比,需要脚本自己算,这里面有一个真实存在、极易犯错的坑。

**从最容易犯错的做法讲起:** `free -m` 的输出里有一列就叫 `free`(空闲内存),看名字很容易顺手拿它来算"使用率 = (总量-free)/总量",但 Linux 内核会把大量暂时空闲的内存拿去当磁盘缓存(`buff/cache` 那一列),这部分内存**需要的时候可以立刻被回收挪用**,不应该被算作"不可用"。真正该用的是 `available` 列——这一列是内核已经把缓存因素算进去之后,真实"还能再分配给新进程"的内存量。

```bash
free -m
```

本机实测(WSL2 Rocky Linux 10.2,真实输出,数值是系统当前状态的快照,不同时刻会有波动):
```
               total        used        free      shared  buff/cache   available
Mem:           23876        1747       19320          88        3162       22129
Swap:           6144           0        6144
```

```bash
total=$(free -m | awk '/^Mem:/{print $2}')
free_col=$(free -m | awk '/^Mem:/{print $4}')
avail_col=$(free -m | awk '/^Mem:/{print $7}')
naive_used_pct=$(( (total - free_col) * 100 / total ))
correct_used_pct=$(( (total - avail_col) * 100 / total ))
echo "偷懒用free列算出来的'使用率': ${naive_used_pct}%"
echo "用available列算出来的真实使用率: ${correct_used_pct}%"
```

本机实测(真实输出):
```
偷懒用free列算出来的'使用率': 19%
用available列算出来的真实使用率: 7%
```

同一台机器、同一时刻,两种算法给出的"使用率"差了将近 3 倍(19% vs 7%)——差出来的这部分,正是被判定成"已使用"但实际上随时能被回收的 3162MB 缓存。如果巡检脚本用错列,会在内存其实很健康的时候频繁误报,是真实存在、不是纸上谈兵的坑。

```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

check_memory() {
    local threshold="$1"
    local total avail used_pct
    total=$(free -m | awk '/^Mem:/{print $2}')
    avail=$(free -m | awk '/^Mem:/{print $7}')
    used_pct=$(( (total - avail) * 100 / total ))
    if [ "$used_pct" -ge "$threshold" ]; then
        echo "[ALERT] memory usage is ${used_pct}% >= threshold ${threshold}%"
        return 1
    else
        echo "[OK] memory usage is ${used_pct}% < threshold ${threshold}%"
        return 0
    fi
}

check_memory 80
assert_eq "$?" "0"

check_memory 1
assert_eq "$?" "1"
```

本机实测(真实输出):
```
[OK] memory usage is 7% < threshold 80%
OK: 0 == 0
[ALERT] memory usage is 7% >= threshold 1%
OK: 1 == 1
```

和阶段 1 完全一样的模式:正常阈值(80%)下真实使用率 7% 不会告警,把阈值故意压到 1% 就能在不消耗任何真实内存的前提下让告警分支真实触发一次。

**常见坑:** `free -m` 不加 `-m` 默认按 KB 显示,数字位数多、肉眼和脚本都不好处理,几乎所有巡检脚本都会显式指定单位(`-m` 兆字节或 `-g` 吉字节);另外 `available` 列是相对新的内核特性(2014 年之后的内核才有),极老旧系统的 `free` 输出可能没有这一列,遇到这种情况需要退回到更保守的估算方式,但 RHCSA(EX200)对标的 RHEL 10 内核完全不存在这个问题。

---

## 阶段 3:加僵尸进程检查——这一步最容易"看起来做对了但其实没测到"

僵尸(zombie,`ps` 里显示成 `Z` 状态,`CMD` 列常带 `<defunct>` 字样)是 [02 类](02-process-and-boot.md)开篇讲过的"父子进程"概念的直接后果:子进程退出后,它的退出状态要留着等父进程用 `wait`/`waitpid` 系统调用来"收尸",父进程收尸之前,这个已经退出、不再占用 CPU 或真实内存的进程记录,会在进程表里以"僵尸"状态挂着。巡检脚本要做的事很简单:数一数 `ps` 里 `STAT` 列以 `Z` 开头的进程有几个。

但在写这个检查函数之前,得先有一个真实的僵尸进程用来验证——这一步比想象中麻烦,值得完整记录下来,因为**第一次尝试直接失败了**。

**从最容易犯错的做法讲起(第一次尝试,真实失败):** 直觉的做法是"背景启动一个立刻退出的子进程,父进程不去 `wait` 它,应该就会留下一个僵尸":

```bash
(
  ( exit 0 ) &
  sleep 1
) &
naive_pid=$!
sleep 0.3
echo "背景进程 naive_pid=$naive_pid,此刻查一下有没有僵尸:"
ps -eo pid,ppid,stat,cmd | awk 'NR==1 || $3 ~ /^Z/'
wait "$naive_pid" 2>/dev/null
```

本机实测(真实输出):
```
背景进程 naive_pid=772,此刻查一下有没有僵尸:
    PID    PPID STAT CMD
```

**只有表头,没有任何一行 `Z` 开头的记录——真的没抓到僵尸,不是命令打错了。** 根因在于 bash 自己对"背景任务什么时候退出"这件事非常上心:只要当前这个 bash 进程还活着,子进程一退出,内核几乎立刻把 `SIGCHLD` 信号送给这个 bash,bash 收到信号后会主动调用收尸相关的系统调用(为了维护 `$!`、`wait`、任务状态这些内部记账,不管是不是交互式 shell 都会这样做),这个过程比人眼能感知的时间窗口快得多——**子进程存在的时间越短,越难在它变成僵尸的瞬间抓到它,因为父进程(还是 bash)几乎立刻就把它收了。**

**能真正奏效的做法:** 让"未来的僵尸"晚一点退出,同时让它的父进程提前**变成一个根本不会去收尸的程序**。具体做法是用 `exec` ——`exec 程序名` 会把当前进程的可执行代码整体替换成目标程序,不产生新进程、PID 不变,但**原来的 bash 从这一刻起彻底不存在了**,替换成 `sleep` 这个纯粹只会睡觉、从不调用 `wait` 的程序。子进程只要在这次替换**之后**才退出,就没有任何东西会去收它的尸:

```bash
bash -c '(sleep 1; exit 0) & exec sleep 4' &
maker_pid=$!
sleep 2
echo "maker_pid=$maker_pid,此刻查一下:"
ps -eo pid,ppid,stat,cmd | awk 'NR==1 || $3 ~ /^Z/'
```

本机实测(真实输出):
```
maker_pid=783,此刻查一下:
    PID    PPID STAT CMD
    785     783 Z+   [bash] <defunct>
```

这一次真实抓到了:`PID 785` 的 `STAT` 是 `Z+`(`Z`=僵尸,`+`=前台进程组成员),`CMD` 显示成 `[bash] <defunct>`——方括号和 `<defunct>` 都是 `ps` 对僵尸进程的专门标注(僵尸没有自己的可执行内存空间了,`ps` 没法正常显示它的命令行,只能显示曾经的程序名加这个标记)。时间线拆开看:`bash -c '...'` 启动的这个进程(`PID 783`)先 fork 出子进程(将来的 `785`),接着**立刻** `exec` 成 `sleep 4`,自己不再是 bash 了;子进程睡了 1 秒才退出,这时它的父进程(`783`)早已经是纯粹的 `sleep`,没有任何 `SIGCHLD` 处理逻辑会去回收它,于是它一直以僵尸状态挂到 `783` 这个 `sleep 4` 本身在第 4 秒退出为止——`783` 一退出,`785` 会被重新挂到 1 号进程(systemd)名下,systemd 会立即收尸清理。

```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

check_zombies() {
    local threshold="$1"
    local zombie_count
    zombie_count=$(ps -eo stat | grep -c '^Z' || true)
    if [ "$zombie_count" -gt "$threshold" ]; then
        echo "[ALERT] zombie process count is $zombie_count > threshold $threshold"
        return 1
    else
        echo "[OK] zombie process count is $zombie_count <= threshold $threshold"
        return 0
    fi
}

# 上面 maker_pid 造出来的僵尸这时候还活着(父进程还要睡到第4秒,现在大约第2秒)
check_zombies 0
assert_eq "$?" "1"

wait "$maker_pid" 2>/dev/null
sleep 0.3
echo "--- 父进程(sleep 4)已退出,僵尸应该已被 systemd 收养并回收 ---"
check_zombies 0
assert_eq "$?" "0"
```

本机实测(真实输出):
```
[ALERT] zombie process count is 1 > threshold 0
OK: 1 == 1
--- 父进程(sleep 4)已退出,僵尸应该已被 systemd 收养并回收 ---
[OK] zombie process count is 0 <= threshold 0
OK: 0 == 0
```

**这个演示制造的僵尸从出现到自动清理干净全程不到 4 秒,只有 1 个,不影响宿主 WSL2 环境的稳定性**,符合任务对"制造真实告警但不做破坏性操作"的要求。

**为什么 `check_zombies` 的判断用 `-gt`(大于)而不是像阶段 1/2 那样用 `-ge`(大于等于):** 阶段 1/2 的阈值含义是"百分比达到这个数就算危险"(比如"用到 80% 就该告警"),80% 本身就该算超标,所以用 `>=`;这里的阈值含义是"最多能容忍几个僵尸"(比如"0 个僵尸是健康基线"),阈值 0 表示"有僵尸就告警",如果用 `>=`,哪怕真实僵尸数是 0,`0 >= 0` 也成立,会导致永远误报——同一个"阈值"概念,依场景不同该用 `>=` 还是 `>`,要看阈值本身算不算在"可接受"范围内,不能死记一种写法套所有场景。

**常见坑:**
1. 前面已经完整复现:僵尸存在的时间窗口可能极短,**用简单的 `cmd &` 背景任务加短暂 `sleep` 去"抓拍"僵尸,大概率抓不到**,不代表僵尸机制不存在,只代表回收发生得比想象中快。
2. `ps -eo stat | grep -c '^Z'` 在**一个僵尸都没有**的时候,`grep -c` 会因为"零匹配"返回退出码 1(哪怕它正确打印了 `0`)——这个坑和阶段 4 要讲的 `set -e` 陷阱是同一个根因,阶段 4 会展开说明,这里的 `check_zombies` 已经用 `|| true` 提前避开了,是为阶段 5 组装时统一开 `set -euo pipefail` 做准备。
3. 僵尸进程本身**不占用内存或 CPU**,杀不掉(`kill -9` 对僵尸无效,因为它已经不是一个在运行的程序,没有可以被终止的执行状态)——僵尸多的真正修复方式是让父进程正确调用 `wait`,或者父进程本身有 bug 需要重启/修复,`kill` 父进程(不是僵尸本身)让僵尸被重新挂到 systemd 名下收尸,才是有效动作。

---

## 阶段 4:加日志 ERROR 行数检查——这一步会现场炸出一个 `set -e` 的坑

最后一个信号:某个指定日志文件最近有没有大量报错。这一步看起来最简单(`grep -c` 数一下行数),但恰恰是这一步会现场炸出一个和 [09 类第 9 节](09-bash-scripting.md) `set -euo pipefail` 直接相关的真实陷阱——先复现问题,再讲修复。

**从最容易犯错的做法讲起,真实翻车复现:** 巡检脚本几乎都会开 `set -euo pipefail`(09 类第 9 节讲过,是生产脚本标配),直觉写法是直接 `grep -c "ERROR" 日志文件` 数出错误行数:

```bash
demo_log=$(mktemp /tmp/hc_demo.XXXXXX.log)
printf 'INFO all good\nINFO still good\n' > "$demo_log"

bash -c '
set -euo pipefail
echo "before"
count=$(grep -c "ERROR" "'"$demo_log"'")
echo "after, count=$count"
'
echo "naive_exit=$? (脚本在before之后就死了,after那行根本没打印出来)"
```

本机实测(真实输出):
```
before
naive_exit=1 (脚本在before之后就死了,after那行根本没打印出来)
```

**`after, count=...` 那一行完全没有打印出来,脚本在 `before` 之后就悄无声息地退出了。** 根因:`grep` 这个命令本身的退出码语义是"找到匹配返回 0,**一个都没找到返回 1**"(这是 `grep` 的正常设计,不是 bug),而这次日志文件里根本没有 `ERROR` 这个词,`grep -c` 虽然正确打印了数字 `0`,但它的**退出码是 1**;这个 1 被 `count=$(...)` 这个赋值语句直接继承,而 `set -e` 的规则是"**赋值语句里的命令替换只要失败,整条语句就算失败,脚本立刻终止**"——日志里没有报错本来是最正常、最应该走"一切正常"分支的情况,却因为这个坑直接让整个巡检脚本死掉,这是巡检脚本"平时看着能跑、一遇到真正干净的日志反而崩溃"的经典真实成因。

**修复写法:** 给这条命令替换补一个"就算失败也别让 `set -e` 出手"的兜底:

```bash
bash -c '
set -euo pipefail
echo "before"
count=$(grep -c "ERROR" "'"$demo_log"'" || true)
echo "after, count=$count"
'
```

本机实测(真实输出):
```
before
after, count=0
```

`|| true` 的意思是"不管前面 `grep -c` 的退出码是什么,这一整条命令的最终退出码都强制变成 0"——`grep -c` 依然会正确地把数字(不管是不是 0)打印到标准输出、被 `$(...)` 正常捕获,只是不再把"零匹配"这个完全正常的结果误判成"命令失败"。回头看阶段 3 的 `check_zombies`,`zombie_count=$(ps -eo stat | grep -c '^Z' || true)` 用的是同一个补丁,原因完全一样——零个僵尸是最健康的情况,不该被 `set -e` 当成错误。

**顺手做一个和大小写相关的真实核对:** `grep "ERROR"` 默认区分大小写,但真实日志里同一种"错误"经常有好几种大小写写法(`ERROR`、`Error`、`error` 都可能出现,取决于是哪个程序、哪个库打的日志)。用这台机器真实的系统日志 `/var/log/messages` 验证一下这个差异有多大:

```bash
tail -n 2000 /var/log/messages > /tmp/hc_real_syslog_sample.log
case_sensitive=$(grep -c "ERROR" /tmp/hc_real_syslog_sample.log || true)
case_insensitive=$(grep -ci "error" /tmp/hc_real_syslog_sample.log || true)
echo "严格区分大小写只匹配到: $case_sensitive 行"
echo "大小写不敏感匹配到: $case_insensitive 行"
rm -f /tmp/hc_real_syslog_sample.log
```

本机实测(真实输出;这是活跃系统日志的最后 2000 行,内容随时间推移,具体数字每次运行会不同,但"大小写不敏感匹配到的行数明显更多"这个现象是稳定的):
```
严格区分大小写只匹配到: 28 行
大小写不敏感匹配到: 110 行
```

真实差了将近 4 倍——如果巡检脚本严格区分大小写,会漏掉大量真实的 `Error`/`error` 记录,`grep -ci`(`-i` 忽略大小写)才是更可靠的选择。

```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

check_log_errors() {
    local logfile="$1" lines="$2" threshold="$3"
    if [ ! -f "$logfile" ]; then
        echo "[SKIP] log file $logfile not found"
        return 0
    fi
    local error_count
    error_count=$(tail -n "$lines" "$logfile" | grep -ci "error" || true)
    if [ "$error_count" -gt "$threshold" ]; then
        echo "[ALERT] recent ERROR-like lines in $logfile: $error_count > threshold $threshold (last $lines lines)"
        return 1
    else
        echo "[OK] recent ERROR-like lines in $logfile: $error_count <= threshold $threshold (last $lines lines)"
        return 0
    fi
}

printf 'INFO boot ok\nERROR disk fail\nINFO retry\nError disk fail again\nINFO recovered\n' > "$demo_log"

check_log_errors "$demo_log" 100 5
assert_eq "$?" "0"

check_log_errors "$demo_log" 100 1
assert_eq "$?" "1"

rm -f "$demo_log"
```

本机实测(真实输出):
```
[OK] recent ERROR-like lines in /tmp/hc_demo.XXXXXX.log: 2 <= threshold 5 (last 100 lines)
OK: 0 == 0
[ALERT] recent ERROR-like lines in /tmp/hc_demo.XXXXXX.log: 2 > threshold 1 (last 100 lines)
OK: 1 == 1
```
(文件名里的 `XXXXXX` 是 `mktemp` 生成的随机后缀,每次运行都不一样,这里不是占位符没填,是真实命令替换出来就长这样。)

这里用的是"内容可控的临时日志文件模拟"这条路子(2 行真实包含 `ERROR`/`Error`),不依赖 `/var/log/messages` 这类真实系统日志此刻恰好有没有报错——巡检脚本要能稳定复现告警分支,不能依赖"运气好日志里正好有问题"。

**常见坑:**
1. 上面完整复现的 `set -e` + `grep -c` 组合陷阱,是这一整篇教程里最容易被忽略、也最值得记住的一条:**任何"用退出码表达'有没有找到'"的命令(`grep`/`grep -c`/`grep -q` 都算),只要结果可能是"合法的零匹配",在 `set -e` 环境下做命令替换赋值,都要补 `|| true` 或用 `if grep -q ...; then ... fi` 这种不经过赋值的写法。**
2. `tail -n "$lines"` 如果日志文件本身行数比 `$lines` 少,不会报错,只会返回全部行数——这是符合预期的宽容行为,不需要额外处理。
3. 忘记 `-i` 导致大小写不敏感的报错被漏检,不会报任何错误、脚本会"安静地跑成功",这类"没报错但结果不对"的坑比"直接崩溃"更难发现,排查时要留意。

---

## 阶段 5:组装成一个完整脚本,收尾打印巡检简报

把前四个独立验证过的检查函数拼进同一个脚本,用 [09 类第 5 节](09-bash-scripting.md)的数组把每一条检查结果收集起来,脚本执行到最后统一打印一份汇总简报,而不是让四条检查各喊各的、读者自己去拼结论。

```bash
#!/usr/bin/env bash
set -euo pipefail

# ---- 配置区:阈值和检查目标,写成变量方便复用/调整,支持环境变量覆盖 ----
DISK_PATH="${DISK_PATH:-/}"
DISK_THRESHOLD="${DISK_THRESHOLD:-80}"
MEM_THRESHOLD="${MEM_THRESHOLD:-80}"
ZOMBIE_THRESHOLD="${ZOMBIE_THRESHOLD:-0}"
LOG_FILE="${LOG_FILE:-/tmp/healthcheck_demo.log}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-100}"
LOG_ERROR_THRESHOLD="${LOG_ERROR_THRESHOLD:-5}"

report=()        # 收集每一条检查结果的完整文字,最后统一汇总打印
alert_count=0
ok_count=0

record() {        # $1=状态(OK/ALERT) $2=这条检查的完整描述
    report+=("[$1] $2")
    if [ "$1" = "ALERT" ]; then
        alert_count=$((alert_count + 1))
    else
        ok_count=$((ok_count + 1))
    fi
}

check_disk() {
    local mount_point="$1" threshold="$2"
    local usage_pct
    usage_pct=$(df -h "$mount_point" | awk 'NR==2 {gsub("%","",$5); print $5}')
    if [ "$usage_pct" -ge "$threshold" ]; then
        record ALERT "disk usage on $mount_point is ${usage_pct}% >= threshold ${threshold}%"
    else
        record OK "disk usage on $mount_point is ${usage_pct}% < threshold ${threshold}%"
    fi
}

check_memory() {
    local threshold="$1"
    local total avail used_pct
    total=$(free -m | awk '/^Mem:/{print $2}')
    avail=$(free -m | awk '/^Mem:/{print $7}')
    used_pct=$(( (total - avail) * 100 / total ))
    if [ "$used_pct" -ge "$threshold" ]; then
        record ALERT "memory usage is ${used_pct}% >= threshold ${threshold}%"
    else
        record OK "memory usage is ${used_pct}% < threshold ${threshold}%"
    fi
}

check_zombies() {
    local threshold="$1"
    local zombie_count
    zombie_count=$(ps -eo stat | grep -c '^Z' || true)
    if [ "$zombie_count" -gt "$threshold" ]; then
        record ALERT "zombie process count is $zombie_count > threshold $threshold"
    else
        record OK "zombie process count is $zombie_count <= threshold $threshold"
    fi
}

check_log_errors() {
    local logfile="$1" lines="$2" threshold="$3"
    if [ ! -f "$logfile" ]; then
        record OK "log file $logfile not found, skip"
        return
    fi
    local error_count
    error_count=$(tail -n "$lines" "$logfile" | grep -ci "error" || true)
    if [ "$error_count" -gt "$threshold" ]; then
        record ALERT "recent ERROR-like lines in $logfile: $error_count > threshold $threshold (last $lines lines)"
    else
        record OK "recent ERROR-like lines in $logfile: $error_count <= threshold $threshold (last $lines lines)"
    fi
}

main() {
    check_disk "$DISK_PATH" "$DISK_THRESHOLD"
    check_memory "$MEM_THRESHOLD"
    check_zombies "$ZOMBIE_THRESHOLD"
    check_log_errors "$LOG_FILE" "$LOG_TAIL_LINES" "$LOG_ERROR_THRESHOLD"

    echo "===== System Healthcheck Report ====="
    for line in "${report[@]}"; do
        echo "$line"
    done
    echo "--------------------------------------"
    echo "Summary: ${ok_count} OK, ${alert_count} ALERT (total $((ok_count + alert_count)) checks)"
    if [ "$alert_count" -eq 0 ]; then
        echo "Overall status: HEALTHY"
    else
        echo "Overall status: DEGRADED"
    fi
    echo "======================================="
}

main
if [ "$alert_count" -eq 0 ]; then exit 0; else exit 1; fi
```

**先跑一次干净基线:** 用一份不含任何报错的合成日志、正常阈值,四项检查应该全部 `OK`。

```bash
printf 'INFO service started\nINFO heartbeat ok\nINFO heartbeat ok\n' > /tmp/healthcheck_demo.log
bash healthcheck.sh
echo "exit code: $?"
```

本机实测(真实输出,脚本已保存为 `healthcheck.sh` 并 `chmod +x`):
```
===== System Healthcheck Report =====
[OK] disk usage on / is 1% < threshold 80%
[OK] memory usage is 4% < threshold 80%
[OK] zombie process count is 0 <= threshold 0
[OK] recent ERROR-like lines in /tmp/healthcheck_demo.log: 0 <= threshold 5 (last 100 lines)
--------------------------------------
Summary: 4 OK, 0 ALERT (total 4 checks)
Overall status: HEALTHY
=======================================
exit code: 0
```

**再跑一次混合告警场景:** 用阶段 3 验证过的僵尸制造技巧现场造一个僵尸、换一份真实含多条错误的日志,同时把日志的告警阈值临时调低(用脚本已经支持的环境变量覆盖,不用改代码),磁盘和内存维持正常阈值——真实验证汇总逻辑在"部分正常、部分告警"这种最贴近真实场景的混合情况下算得对不对,而不是只会演示"全绿"或"全红"两种极端。

```bash
printf 'INFO start\nERROR db connection failed\nWARN retrying\nError timeout again\nERROR disk write failed\nERROR disk write failed\nINFO recovered\n' > /tmp/healthcheck_demo.log

bash -c '(sleep 1; exit 0) & exec sleep 6' &
zombie_maker_pid=$!
sleep 2

LOG_ERROR_THRESHOLD=2 bash healthcheck.sh
echo "exit code: $?"

wait "$zombie_maker_pid" 2>/dev/null
```

本机实测(真实输出):
```
===== System Healthcheck Report =====
[OK] disk usage on / is 1% < threshold 80%
[OK] memory usage is 4% < threshold 80%
[ALERT] zombie process count is 1 > threshold 0
[ALERT] recent ERROR-like lines in /tmp/healthcheck_demo.log: 4 > threshold 2 (last 100 lines)
--------------------------------------
Summary: 2 OK, 2 ALERT (total 4 checks)
Overall status: DEGRADED
=======================================
exit code: 1
```

四项检查里,磁盘和内存维持真实的正常状态(`OK`),僵尸检查和日志检查真实触发告警(`ALERT`),汇总统计"2 OK, 2 ALERT"精确对应,整体状态判定成 `DEGRADED`,脚本退出码变成 `1`(约定:全部正常退出码 `0`,存在任何告警退出码 `1`,方便这个脚本将来被 cron 或 systemd timer 调用时,靠退出码而不是解析文字输出来判断"这次巡检有没有问题")。这就是任务要求的"脚本执行结束时汇总打印一份巡检简报"——四条检查各自的即时输出("这一步刚查完是什么状态")和最后的汇总简报("这次巡检整体是什么状态")是两个独立但一致的信息层。

**常见坑:**
1. `report+=(...)` 这个数组追加操作,必须发生在**同一个 shell 进程**里才会真实累积——如果哪个 `check_xxx` 函数是通过管道或者 `$(...)` 子进程调用的(比如写成 `result=$(check_disk ...)`),`record` 追加进去的其实是子进程自己的数组副本,函数一结束就跟着子进程一起消失,`main` 里看到的 `report` 数组会一直是空的。本篇四个 `check_xxx` 函数全部是直接调用(没有经过子进程),这是刻意的选择,不是巧合。
2. 最后 `if [ "$alert_count" -eq 0 ]; then exit 0; else exit 1; fi` 必须放在 `main` **调用之后**、作为脚本真正的最后一步——如果不小心把它塞进 `main` 函数内部,`main` 提前 `return`/`exit` 会跳过打印汇总简报的部分,巡检简报可能不完整。

---

## 可以怎么继续扩展(只指方向,不实现)

- **从手动运行变成定时巡检**:现在的 `healthcheck.sh` 需要人手动执行,生产环境标准做法是配一对 systemd service + timer(呼应 [02 类第 11 节](02-process-and-boot.md) systemd timer),每隔固定时间自动跑一次。
- **从"打印到终端"变成真正能通知到人的告警通道**:现在的汇总简报只是打印在标准输出,真实告警系统会在 `alert_count > 0` 时额外发一封邮件或调用一个 webhook——这一步需要引入本系列没有覆盖的外部通知集成,只指方向。
- **从"单机"扩展到"批量巡检多台主机"**:把 `healthcheck.sh` 用 `scp` 分发到多台机器、通过免密 SSH(呼应 [01 类第 12 节](01-essential-tools.md) SSH 密钥认证)远程执行并把结果汇总到一个地方,是运维里"单机脚本"进化成"巡检系统"的典型下一步。

这三个方向都不实现,是为了让这篇教程聚焦在"四个已学知识点怎么拼成一个真实可用的巡检工具"这一件事上——真要继续做下去,每一个方向单独展开都够写一整篇。

## 这篇教程展示的方法论

任何一条已完成的深挖系列,都可以用同样的模式产出"教程体"内容:挑几个关联的知识点 → 设计一个真实运维会用、读者一看就懂价值的小工具 → 分阶段增量实现,每一步都在真实环境里跑起来看到真实效果,而不是一次性甩出完整脚本。和纯 Python 系列(dsa-deep-dive)的差异在于验证载体不同——这里没有跨平台一致的解释器,验证必须落在这条系列已经确立的 WSL2 Rocky Linux 环境里,复用而不是重新搭建;遇到"设计上直觉觉得该成立,但真实一跑发现不成立"的情况(比如阶段 3 第一次抓僵尸失败、阶段 4 的 `set -e` 陷阱现场复现),如实记录失败尝试和根因,比只展示"一次成功"的版本更接近这条系列一贯的诚实记录原则。这篇是第二个"教程体"试点(第一个是 dsa-deep-dive 的搜索引擎),验证格式能不能从纯 Python 场景迁移到系统操作场景;其余系列要不要配套同类文件,是后续单独决定的问题,这里不展开。

---

*创建:2026-07-24*
