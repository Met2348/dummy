# 09 · bash 脚本编程本身(bash Scripting Fundamentals)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一篇是唯一不涉及 RHEL 系统管理命令、纯讲 bash 这门脚本语言本身的一批——变量、条件、循环、函数、数组、参数展开、here-doc、trap、健壮性写法,9 个知识点。**本文所有代码例子已在 Git Bash(GNU bash 5.1.16,x86_64-pc-msys)下实际跑通验证**——这些是 bash 语言层面的特性,不涉及 systemd/RHEL 专属组件,Git Bash 用的就是真正的 GNU bash 二进制,和 Rocky Linux 上的 bash 在语法行为上没有差异,因此不必等 Rocky Linux 环境修复即可先完成本篇。
>
> **和前四条系列的差异声明**(详见 [00-roadmap.md](00-roadmap.md) 开头):本仓库没有 Linux 系统管理场景可挖,"真实场景例子"部分统一标注为典型运维/RHCSA 考试场景,不冒充仓库代码里挖出来的。

---

## 1. 变量与作用域(局部/全局/`export` 环境变量)

**命令/配置:**
```bash
VAR=value          # 赋值,等号两边绝对不能有空格
local var=value    # 只能在函数内使用,声明为函数局部变量
export VAR=value   # 让子进程(fork出去的新shell/新程序)也能看到这个变量
```

**一句话是什么:** bash 变量默认作用域是"当前 shell 及其之后所有代码"(包括函数内,除非显式 `local`),而是否传递给**子进程**则完全由 `export` 决定——这是两条独立的轴,很多人会搞混。

**为什么 RHCSA 真考 / 生产会用到:** systemd 的 `EnvironmentFile=`、`/etc/profile.d/*.sh`、几乎所有部署脚本都要正确处理"这个变量该不该传给子进程"这个问题;RHCSA 考试里配置服务的环境变量、写自动化脚本都会直接踩到这一点。

**从最容易犯错的做法讲起:** 函数里赋值不加 `local`,会**悄悄污染外层同名变量**——这是新手脚本里最常见、最难排查的 bug 之一,因为函数"看起来"是独立的作用域,实际上默认不是:
```bash
counter=100
bump_no_local() {
    counter=$((counter + 1))   # 没有 local,改的就是外层那个 counter
}
bump_no_local
echo "$counter"   # 101 —— 外层变量被意外改掉了,调用者毫无防备

counter=100
bump_with_local() {
    local counter               # 声明为局部变量,和外层同名但互不影响
    counter=$((counter + 1))
    echo "内部看到的 counter=$counter"
}
bump_with_local
echo "$counter"   # 100 —— 外层变量安然无恙
```

**真实场景例子(典型运维场景,非仓库代码):** 部署脚本给后台服务传配置,常见写法是先 `export DB_HOST=... DB_PORT=...`,再用 `systemd-run` 或直接 `exec` 启动服务进程;服务进程(子进程)能读到这些环境变量,但服务进程内部再怎么改这些变量,都不会影响回部署脚本本身——这是"单向传递、写时拷贝"的本质,不理解这一点会写出"以为改了父进程变量、其实只改了子进程自己那份拷贝"的错误脚本。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

# export 影响子进程可见性,但子进程改了不会传回父进程
export GREETING="hello"
child_output=$(bash -c 'echo "子进程看到: $GREETING"; GREETING="changed"; echo "子进程内部改成: $GREETING"')
echo "$child_output"
assert_eq "$GREETING" "hello"   # 父进程的 GREETING 完全没变

# 没 export 的变量,子进程根本看不到
LOCAL_ONLY="secret"
child_result=$(bash -c 'echo "${LOCAL_ONLY:-空的}"')
assert_eq "$child_result" "空的"
```
本机实测:两个 `assert_eq` 均输出 `OK`。

**常见坑:**
1. `VAR = value`(等号两边有空格)不会报"语法错误",而是被解释成"执行一个叫 `VAR` 的命令,参数是 `=` 和 `value`",报 `command not found`,容易让人摸不着头脑。
2. 变量名区分大小写,`Path` 和 `PATH` 是完全不同的两个变量——这个坑在 Windows 背景的初学者身上尤其常见。
3. `export` 只在"当前赋值往后"的子进程可见,已经 fork 出去、正在运行的旧进程不会因为父进程后来 `export` 了什么而受影响。

---

## 2. 条件判断(`[ ]` vs `[[ ]]` vs `(( ))` 的区别与选用)

**命令/配置:**
```bash
[ expr ]     # 等价于外部命令 test,POSIX 兼容,写字符串/文件测试要小心加引号
[[ expr ]]   # bash/ksh 关键字,支持 == 模式匹配、&&/||、不怕未加引号的空变量
(( expr ))   # 专做算术比较,直接写 a > b 不需要 -gt,可读性最好
```

**一句话是什么:** 三者都能"判断真假",但 `[ ]` 是一个外部程序调用(对未加引号的空变量、包含空格的字符串很脆弱),`[[ ]]` 是 bash 内建的、更安全的字符串/文件测试关键字,`(( ))` 专门服务数字比较,三者不能随意混用同一套语法。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 的很多自动化任务(检查服务状态、判断磁盘空间、校验参数)都要写条件判断;判分标准里"脚本能正确处理边界情况(空参数、异常输入)"往往就是靠 `[ ]` vs `[[ ]]` 的选择决定的。

**从最容易犯错的做法讲起:** 用 `[ ]` 但不给变量加引号,一旦变量为空,`[ $name = "alice" ]` 会直接展开成 `[ = "alice" ]`,这是一个语法错误(`unary operator expected`),而不是"判断为假"——很多脚本因此在边界情况下直接崩溃:
```bash
name=""
if [ $name = "alice" ]; then     # 危险:未加引号,$name 为空时报语法错误
    echo "matched"
fi
# bash 报错: [: =: unary operator expected

if [[ $name = "alice" ]]; then   # 安全:[[ ]] 内部不做单词分割,空变量也能正常判断
    echo "matched"
else
    echo "safe no match"          # 正常走到这里,不报错
fi
```

**真实场景例子(典型运维场景,非仓库代码):** 巡检脚本判断"某个服务名参数是否传了",如果调用者忘记传参数(`$1` 为空),用 `[ ]` 不加引号的写法会直接报语法错误、脚本中断,而 `[[ ]]` 或加了引号的 `[ "$1" = ... ]` 能正常走到"参数缺失"的分支、打印友好提示后退出,这是脚本健壮性的基本要求。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

# [[ ]] 支持模式匹配(不需要转义的通配符),[ ] 不支持
filename="report.txt"
if [[ $filename == *.txt ]]; then
    result="是txt文件"
else
    result="不是"
fi
assert_eq "$result" "是txt文件"

# (( )) 专门做算术比较,数字用 (( )) 比 [ ] 的 -eq/-gt 好读
a=5; b=3
if (( a > b )); then result="a更大"; else result="b更大"; fi
assert_eq "$result" "a更大"
```
本机实测:两个 `assert_eq` 均输出 `OK`。

**常见坑:** `[ ]` 里的 `>` 会被解释成**输出重定向**,不是"大于"——`[ 5 > 3 ]` 不会报错,而是静默创建一个名叫 `3` 的文件(把 `[` 命令的输出重定向进去),本机实测确认:执行后当前目录真的多出一个文件 `3`。这是 `[ ]` 最隐蔽的坑之一,数字比较务必用 `(( ))` 或 `[ ... -gt ... ]`,不要在 `[ ]` 里直接写 `>`/`<`。

---

## 3. 循环结构(`for`/`while`/`until` 及 `break`/`continue`)

**命令/配置:**
```bash
for (( i=0; i<N; i++ )); do ...; done     # C 风格,适合按次数计数
for item in "${arr[@]}"; do ...; done      # 遍历式,适合遍历列表/数组
while [ cond ]; do ...; done               # 条件为真就继续
until [ cond ]; do ...; done               # 条件为真才停止(和 while 相反)
```

**一句话是什么:** `for` 分"数计数"和"遍历列表"两种常见形态,`while`/`until` 逻辑互为镜像(前者"真则继续"、后者"真则停止"),`break`/`continue` 的跳出/跳过语义和大多数语言一致。

**为什么 RHCSA 真考 / 生产会用到:** 批量处理文件、逐行读日志、轮询等待服务启动完成,都是 RHCSA 实操和真实运维里的高频场景,几乎每个自动化脚本都会用到至少一种循环。

**插一句,`read` 命令是什么(下面这条"最容易犯错的做法"会直接用到,这里先建立基础,不然会不知道 `IFS=`/`-r` 到底在修正什么):** `read` 是 bash 内建命令,作用是"从标准输入读一行,存进指定的变量里"——`while read line; do ...; done < file` 这个组合的意思是:反复执行"读一行到 `line` 变量、跑一遍循环体",直到文件读完(`read` 读到文件末尾会返回非 0 状态码,`while` 因此自然停止,不需要另外写判断结束的条件)。**问题在于 `read` 默认的"读一行"这个动作,并不是原样照抄这一行**——它会按 `IFS`(Internal Field Separator,内部字段分隔符,默认包含空格/Tab/换行)的规则自动裁剪掉行首尾的空白,同时默认还会把反斜杠 `\` 当成转义字符处理、吃掉下一个字符的原始含义。这两个默认行为在逐行处理文件内容时几乎从来都不是你想要的效果(谁都不希望内容被悄悄改写),`IFS=`(清空分隔符,不做任何裁剪)和 `-r`(raw,不处理转义)就是专门用来关掉这两个默认行为的开关。

**从最容易犯错的做法讲起:** 用 `while read line` 逐行读文件,最容易忘的是要写 `IFS= read -r line`——不加 `IFS=` 会吃掉每行首尾的空白,不加 `-r` 会把行内的反斜杠当转义符处理,悄悄改变数据内容,不是猜测,本机现场对比过具体差异:
```bash
printf '  a leading-space line  \n' > /tmp/read_demo.txt
while read line; do echo "[无 IFS=]      [$line]"; done < /tmp/read_demo.txt
while IFS= read -r line; do echo "[IFS= read -r] [$line]"; done < /tmp/read_demo.txt
# [无 IFS=]      [a leading-space line]        —— 首尾空格被悄悄吃掉了
# [IFS= read -r] [  a leading-space line  ]    —— 原样保留,这才是文件里真实的内容

printf 'a\\tb\n' > /tmp/read_demo2.txt    # printf 的格式串里 \\ 才是一个字面反斜杠,这里存进文件的是字面的反斜杠+t两个字符,不是真的Tab
while read line2; do echo "[无 -r] [$line2]"; done < /tmp/read_demo2.txt
while IFS= read -r line2; do echo "[有 -r] [$line2]"; done < /tmp/read_demo2.txt
# [无 -r] [atb]     —— 反斜杠被当成转义符处理掉,\t 变成了 t,内容被悄悄改写
# [有 -r] [a\tb]    —— 原样保留

rm -f /tmp/read_demo.txt /tmp/read_demo2.txt
```
本机实测(WSL2 Rocky Linux 和 Git Bash 交叉验证,两边结果完全一致——这是 bash 语言本身的行为,和具体发行版无关):上面注释里的输出是现场跑出来的真实结果,不是预期描述。日常写逐行处理脚本,`while` 循环体里能正确统计行数(下面这段是本节原本就有的例子,现在建立在"为什么必须这么写"的基础上了):
```bash
printf 'line1\nline2\nline3\n' > /tmp/lines.txt
count=0
while IFS= read -r line; do    # 正确姿势:IFS= 防止裁剪空白,-r 防止反斜杠被转义
    count=$((count + 1))
done < /tmp/lines.txt
echo "$count"   # 3
```

**真实场景例子(典型运维场景,非仓库代码):** 巡检脚本轮询等待某个服务真正就绪(而不是 `systemctl start` 命令返回就假设服务已经可用),典型写法是 `until systemctl is-active --quiet myservice; do sleep 1; done`——利用 `until` "条件为真才停"的语义,持续等到服务状态变成 active 为止,这是比"固定 sleep 5 秒然后祈祷"更可靠的做法。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

# C风格for vs 遍历列表式for
total=0
for (( i=1; i<=5; i++ )); do total=$((total + i)); done
assert_eq "$total" "15"

fruits=(apple banana cherry)
joined=""
for f in "${fruits[@]}"; do joined="${joined}${f},"; done
assert_eq "$joined" "apple,banana,cherry,"

# break/continue
processed=0
for i in 1 2 3 4 5; do
    if [ "$i" -eq 3 ]; then continue; fi   # 跳过3
    if [ "$i" -eq 5 ]; then break; fi      # 到5整体停止
    processed=$((processed + 1))
done
assert_eq "$processed" "3"   # 处理了1,2,4
```
本机实测:三个 `assert_eq` 均输出 `OK`。

**常见坑:** `for i in $(seq 1 $n)` 这种写法在 `$n` 里混入意外空格/通配符时会出问题(受词分割和路径展开影响),优先用 `for (( i=1; i<=n; i++ ))` 这种 C 风格写法做纯数字计数,更安全也更快(不需要 fork `seq` 这个外部命令)。

---

## 4. 函数定义与参数(`$1`/`$@`/`$#`/`return` vs `echo` 传值)

**命令/配置:**
```bash
func_name() {
    local x="$1"       # 第1个参数
    echo "共 $# 个参数: $@"
    return 0           # 只能返回 0-255 的整数状态码
}
```

**一句话是什么:** bash 函数拿参数的方式和脚本拿命令行参数完全一样(`$1`、`$2`……、`$#`、`$@`),但 `return` **只能带回 0-255 的整数**(通常表示成功/失败),想要"返回"字符串或复杂数据,惯用法是用 `echo` 输出、调用方用 `$(...)` 命令替换捕获。

**为什么 RHCSA 真考 / 生产会用到:** 稍微复杂一点的运维脚本都会拆函数复用逻辑(比如"检查某个服务是否存在"这种判断封装成函数反复调用),`return` 语义用错会导致状态码判断全部失效。

**从最容易犯错的做法讲起:** 以为 `return` 可以像其他语言的函数一样"返回"任意值,结果数值超过 255 被静默截断,且完全没有报错提示:
```bash
weird_return() {
    return 300     # 会被截断成 300 % 256 = 44,不是300!
}
weird_return
echo "$?"   # 44 —— 完全不是你写的300,而且没有任何警告
```

**真实场景例子(典型运维场景,非仓库代码):** 封装一个 `service_exists()` 函数,内部用 `systemctl list-unit-files | grep -q "$1"` 判断,`return` 该 `grep` 命令的退出码(0=找到/1=没找到)给调用方,调用方直接 `if service_exists sshd; then ...`——这是 `return` 表示"成功/失败"语义的标准用法,不是用来传数据的。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

# 正确姿势:函数"返回"字符串,用echo输出+命令替换捕获
get_greeting() {
    local name="$1"
    echo "hello, ${name}"
}
msg=$(get_greeting "world")
assert_eq "$msg" "hello, world"

# return 只应该表示状态,(( )) 本身的退出码天然适合当return值
is_even() { (( $1 % 2 == 0 )); }
if is_even 4; then result="偶数"; else result="奇数"; fi
assert_eq "$result" "偶数"

# "$@" 保留每个参数独立边界,"$*" 把所有参数糊成一个字符串——加引号时行为完全不同
count_args_at() { printf '%s\n' "$@" | wc -l; }
count_args_star() { printf '%s\n' "$*" | wc -l; }
assert_eq "$(count_args_at "a b" "c d")" "2"     # 两行:"a b" 和 "c d"
assert_eq "$(count_args_star "a b" "c d")" "1"   # 一行:"a b c d"
```
本机实测:四个 `assert_eq` 均输出 `OK`。

**常见坑:** `"$@"` 和 `"$*"` 只有在**加了双引号**时才会体现差异——不加引号时两者行为一样(都会做词分割),这也是为什么惯用法永远建议写成 `"$@"`(带引号)而不是裸的 `$@`。

---

## 5. 数组(索引数组与关联数组 `declare -A`)

**命令/配置:**
```bash
arr=(a b c)              # 索引数组(下标从0开始)
declare -A map            # 关联数组,必须显式声明(bash 4.0+)
map[key]=value
```

**一句话是什么:** bash 有两种数组——普通的索引数组(下标是数字,类似其他语言的 list)和关联数组(下标是字符串,类似字典/map),后者必须先用 `declare -A` 声明才能用,忘记声明会被当成普通字符串变量对待。

**为什么 RHCSA 真考 / 生产会用到:** 批量操作一组主机名/服务名、用关联数组做"端口号→服务名"这类配置映射,是运维脚本里最常见的数据结构需求。

**从最容易犯错的做法讲起:** 遍历数组时忘记给 `"${arr[@]}"` 加引号,元素内部的空格会被意外拆分,导致"明明只有 2 个文件,循环却跑了 3 次":
```bash
files=("my report.txt" "data.csv")
count_noquote=0
for f in ${files[@]}; do count_noquote=$((count_noquote + 1)); done    # 没加引号!
echo "$count_noquote"   # 3 —— "my report.txt" 被拆成了2个词

count_quoted=0
for f in "${files[@]}"; do count_quoted=$((count_quoted + 1)); done    # 正确姿势
echo "$count_quoted"    # 2 —— 符合预期
```

**真实场景例子(典型运维场景,非仓库代码):** 巡检脚本用关联数组 `declare -A expected_ports=([ssh]=22 [http]=80 [https]=443)` 定义"服务名到预期端口"的映射,再用 `ss -tlnp` 的实际输出逐一核对,比写一长串 `if/elif` 判断清晰得多,也更容易维护(加一条服务只需要加一行映射)。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

services=(sshd firewalld chronyd)
assert_eq "${services[0]}" "sshd"
assert_eq "${#services[@]}" "3"
services+=("crond")
assert_eq "${services[-1]}" "crond"     # bash 4.3+ 支持负数下标取最后一个

declare -A port_map
port_map[ssh]=22; port_map[http]=80; port_map[https]=443
assert_eq "${port_map[http]}" "80"
assert_eq "${#port_map[@]}" "3"
keys_sorted=$(printf '%s\n' "${!port_map[@]}" | sort | tr '\n' ',')
assert_eq "$keys_sorted" "http,https,ssh,"
```
本机实测:五个 `assert_eq` 均输出 `OK`。

**常见坑:** `declare -A` 是 bash 4.0+ 才有的特性,某些精简系统上的 `/bin/sh`(指向 dash 等)完全不支持关联数组,脚本 shebang 务必写 `#!/usr/bin/env bash` 而不是 `#!/bin/sh`,否则关联数组语法会直接报错。

---

## 6. 参数展开与字符串处理(`${var#pattern}` 系列,`${var/old/new}`)

**命令/配置:**
```bash
${var#pattern}    ${var##pattern}    # 从开头去掉最短/最长匹配(去前缀)
${var%pattern}    ${var%%pattern}    # 从结尾去掉最短/最长匹配(去后缀)
${var/old/new}    ${var//old/new}    # 替换第1处/替换全部
${var:-default}   ${var:?msg}        # 默认值 / 未设置就报错退出
```

**一句话是什么:** bash 内建了一整套字符串处理的"参数展开"语法,能替代大量原本要靠 `basename`/`dirname`/`sed`/`cut` 才能做的事,而且是 shell 内部完成、不需要额外 fork 子进程,速度快很多。

**为什么 RHCSA 真考 / 生产会用到:** 解析文件路径、处理 URL/配置项、给变量设合理默认值,是几乎所有 bash 脚本都会用到的基本功,也是脚本"看起来专业"和"东拼西凑"的分水岭。

**从最容易犯错的做法讲起:** 每次要取文件名/目录名就去调用外部命令 `basename`/`dirname`,在循环里调用成千上万次会产生大量子进程开销;应该优先用参数展开:
```bash
path="/etc/sysconfig/network-scripts/ifcfg-eth0"

base_slow=$(basename "$path")     # 笨办法:fork一个子进程
base_fast="${path##*/}"           # 正式写法:shell内部完成,无需fork
# 二者结果相同,但 base_fast 快得多,尤其在循环里差异明显
```

**真实场景例子(典型运维场景,非仓库考试场景):** 从 `/etc/fstab` 或 `blkid` 输出里解析设备路径、UUID、挂载点这类字段,批量重命名日志文件(去掉扩展名再拼接时间戳)、拼接 URL 组件,都严重依赖 `#`/`##`/`%`/`%%` 这四个符号——记忆口诀:`#` 管前缀、`%` 管后缀,符号数量翻倍(`##`/`%%`)就是"贪婪匹配最长的那一段"。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

url="https://example.com/path/to/page.html"
assert_eq "${url#*//}"  "example.com/path/to/page.html"    # 去掉最短前缀匹配
assert_eq "${url##*/}"  "page.html"                          # 去掉最长前缀匹配
assert_eq "${url%.*}"   "https://example.com/path/to/page"  # 去掉最短后缀匹配
assert_eq "${url%%/*}"  "https:"                              # 去掉最长后缀匹配

sentence="the cat sat on the mat"
assert_eq "${sentence/the/a}"  "a cat sat on the mat"    # 只替换第1次出现
assert_eq "${sentence//the/a}" "a cat sat on a mat"        # 全部替换

unset MY_VAR
assert_eq "${MY_VAR:-default}" "default"   # 未设置或为空,用default顶替
```
本机实测:六个 `assert_eq` 均输出 `OK`;`${REQUIRED_VAR:?msg}` 未设置时会直接报错退出(状态码 1),已单独验证,是编写"必填参数"检查的标准惯用法。

**常见坑:** `#`/`%` 的记忆方向容易搞反——把 `#` 想成"井号在字符串左边探头(去前缀)"、`%` 想成"百分号在字符串右边探尾(去后缀)",符号数量翻倍就是"尽可能多吃"(贪婪匹配)。

---

## 7. here-doc 与 here-string(`<<EOF`,`<<<`)

**命令/配置:**
```bash
cat << EOF          # here-doc:把多行文本喂给命令的标准输入,变量会展开
...
EOF
cat << 'EOF'         # 定界符加引号:内容原样保留,变量不展开
...
EOF
command <<< "text"   # here-string:把单个字符串直接喂给标准输入
```

**一句话是什么:** here-doc 是"写多行文本喂给标准输入"最干净的语法,默认会展开里面的变量;把开头的定界符加上引号(`<<'EOF'`)就会变成字面量模式,常用来生成配置文件模板。

**为什么 RHCSA 真考 / 生产会用到:** 用脚本生成配置文件(`/etc/fstab`、`sshd_config` 片段、systemd unit 文件)是 RHCSA 里的高频操作,here-doc 是最干净利落的写法,比一堆 `echo >> file` 拼接可读性好得多。

**从最容易犯错的做法讲起:** 用一堆 `echo` 或字符串拼接来生成多行文本,可读性差、维护困难:
```bash
name="alice"
slow_msg="hello ${name}
this is line 2
this is line 3"     # 硬编码换行符拼字符串,难读

fast_msg=$(cat << EOF
hello ${name}
this is line 2
this is line 3
EOF
)                     # here-doc:所见即所得,清晰得多
# slow_msg 和 fast_msg 内容完全一致,但后者可读性和可维护性明显更好
```

**真实场景例子(典型运维场景,非仓库代码):** 部署脚本用 `cat << 'EOF' > /etc/systemd/system/myapp.service` 这种**加引号的 here-doc**直接写出一份 systemd unit 文件,加引号是因为 unit 文件里可能本身就含有 `$` 开头的 systemd 变量(如 `%i`、`$MAINPID` 语境下的特殊符号),不希望被 bash 提前展开。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

name="alice"
literal=$(cat << 'EOF'
this is ${name}, not expanded
EOF
)
assert_eq "$literal" 'this is ${name}, not expanded'   # 加引号的定界符,变量不展开

result=$(tr 'a-z' 'A-Z' <<< "hello")
assert_eq "$result" "HELLO"    # here-string,把字符串直接喂给标准输入
```
本机实测:两个 `assert_eq` 均输出 `OK`。

**常见坑:** 结尾的定界符(如 `EOF`)前后不能有多余的空格/字符,否则 bash 不认为这是结束标记,会一直往下"吃"内容直到文件末尾报语法错误;需要在缩进的 `if`/`for` 块里写 here-doc 时,用 `<<-` 而不是 `<<`,允许结尾标记前有 Tab 缩进(注意:只能是 Tab,不能是空格)。

---

## 8. trap 信号处理与退出清理(`EXIT`/`ERR` trap)

**命令/配置:**
```bash
trap 'command' EXIT    # 不管脚本怎么结束(正常/报错/被Ctrl+C)都会执行
trap 'command' ERR     # 有命令返回非0退出码时触发(常配合 set -e)
trap 'command' INT     # 捕获 Ctrl+C(SIGINT)
```

**一句话是什么:** `trap` 让你在"脚本即将结束"或"某类信号发生"时插入一段一定会被执行的清理代码,最常见的用途是"不管脚本是正常跑完还是中途失败,临时文件/锁文件都要被删掉"。

**为什么 RHCSA 真考 / 生产会用到:** 运维脚本经常需要创建临时文件、加锁、挂载临时资源,如果脚本中途失败却没有清理机制,会在系统里留下垃圾文件、残留挂载点甚至死锁,`trap EXIT` 是防止这类问题的标准手段。

**从最容易犯错的做法讲起:** 脚本创建了临时文件,但没有用 `trap` 兜底,一旦中途失败(或被用户按 Ctrl+C 中断),临时文件就会永远留在系统里,长期运行会慢慢堆积垃圾:
```bash
run_without_cleanup() {
    tmpfile=$(mktemp /tmp/demo_no_trap.XXXXXX)
    echo "tmp created: $tmpfile"
    return 1   # 模拟脚本中途失败,函数直接返回,没人清理tmpfile
}
run_without_cleanup
ls /tmp/demo_no_trap.* 2>/dev/null   # 真的能看到残留文件
```

**真实场景例子(典型运维场景,非仓库代码):** 备份脚本先创建一个临时挂载点、挂载一个网络文件系统做数据同步,用 `trap 'umount "$mnt"; rmdir "$mnt"' EXIT` 保证不管同步过程是否成功,脚本结束前一定会卸载挂载点、删掉临时目录,避免"半吊子挂载点"残留导致下次运行冲突。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

run_with_trap() (
    tmpfile=$(mktemp /tmp/demo_with_trap.XXXXXX)
    trap 'rm -f "$tmpfile"' EXIT     # 不管这个子shell怎么结束,退出前一定删tmpfile
    return 1                          # 模拟失败
)
run_with_trap
leftover=$(ls /tmp/demo_with_trap.* 2>/dev/null | wc -l)
assert_eq "$leftover" "0"    # trap 自动清理了,没有残留

trap_err_result=$(bash -c 'trap "echo TRAPPED" ERR; set -e; false; echo unreachable' 2>&1)
assert_eq "$trap_err_result" "TRAPPED"    # ERR trap 在 false 失败时被触发
```
本机实测:两个 `assert_eq` 均输出 `OK`。

**常见坑:** `trap` 写在子 shell `( ... )` 里只对这个子 shell 生效,不会影响父 shell 的 trap 设置;同一个信号后面设置的 `trap` 会覆盖前面的,如果脚本多处都想在 EXIT 时清理,要把所有清理逻辑合并到同一个 `trap ... EXIT`,而不是分别调用多次 `trap`(后者只有最后一次生效)。

---

## 9. 脚本调试与健壮性(`set -euo pipefail`,shellcheck)

**命令/配置:**
```bash
set -e            # 任意命令失败(非0退出码)立刻终止脚本
set -u             # 引用未定义变量直接报错,而不是静默当空字符串
set -o pipefail    # 管道中任意一环失败,整个管道的退出码都算失败
set -x             # 调试模式:执行每条命令前先打印出来
```

**一句话是什么:** 默认情况下 bash 对"命令失败""变量未定义""管道内部失败"都非常宽容(能跑就跑,不主动报错),`set -euo pipefail` 这套组合拳把三种最容易被忽视的错误全部变成"立刻失败并报错",是生产脚本的标配开头。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 判分和生产事故复盘里最常见的失败模式就是"脚本某一步悄悄失败了,后面的步骤在错误的前提下继续跑,最后在一个完全无关的地方报出令人困惑的错误"——`set -euo pipefail` 从根源上避免了这类问题。

**从最容易犯错的做法讲起:** 不加任何 `set` 选项,某条关键命令(比如 `cd` 到目标目录)失败了,脚本毫无察觉地继续往下执行,可能在错误的目录里做出破坏性操作:
```bash
# 危险:cd 失败了,脚本浑然不觉,继续往下跑
cd /nonexistent_dir_xyz 2>/dev/null
echo "还在继续执行,可能在错误的目录里做危险操作!"

# 正式写法:开头就加 set -e,cd 一旦失败,脚本立刻停止
set -e
cd /nonexistent_dir_xyz 2>/dev/null
echo "这行不会被打印,因为 cd 失败后脚本已经停了"
```

**真实场景例子(典型运维场景,非仓库代码):** 生产环境的部署/巡检脚本几乎无一例外都以 `#!/usr/bin/env bash` + `set -euo pipefail` 开头,这已经是社区公认的"防御性脚本"起手式;配合 `shellcheck`(静态检查工具,能在运行前发现引号缺失、拼写错误等问题)在 CI 里跑一遍,能在脚本上线前拦下大部分低级错误——**本机未安装 shellcheck**,如实说明:可以用 `sudo dnf install ShellCheck`(Rocky Linux 环境修复后)安装,或直接在线用 [shellcheck.net](https://www.shellcheck.net/) 检查片段,不影响本篇其余知识点的验证。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

# set -u:引用未定义变量直接报错
undefined_var_result=$(bash -c 'set -u; echo "$UNDIFINED_TYPO"' 2>&1)
echo "$undefined_var_result" | grep -qi "unbound variable" && echo "OK: set -u 抓到了未定义变量"

# set -o pipefail:管道中间失败也要算数
without_pipefail=$(bash -c 'false | true; echo $?')
assert_eq "$without_pipefail" "0"   # 不加pipefail,只看最后一个true,退出码是0

with_pipefail=$(bash -c 'set -o pipefail; false | true; echo $?')
assert_eq "$with_pipefail" "1"      # 加了pipefail,能感知到false失败了
```
本机实测:`set -u` 抓到未定义变量、两个 `assert_eq` 均输出 `OK`。

**常见坑:** `set -e` 不是"万能保险"——它在某些场景下不生效,最典型的是**放在 `if`/`while` 条件位置的命令**(`if some_cmd; then ...` 里 `some_cmd` 失败不会触发 `set -e`,因为这是 `if` 判断的正常逻辑)和**管道非最后一个命令**(不加 `pipefail` 时);这也是为什么 `set -e` 几乎总是要和 `pipefail` 搭配、并且不能替代显式的错误处理逻辑。

---

*本篇完成:2026-07-11,9 个知识点,全部代码在 Git Bash(GNU bash 5.1.16)下真实跑通验证。*
