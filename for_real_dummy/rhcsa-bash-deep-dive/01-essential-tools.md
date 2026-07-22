# 01 · 必备工具与文本处理(Essential Tools & Text Processing)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 本篇覆盖 14 个知识点:文件操作、通配符、I/O 重定向、grep/sed/find 三大文本处理工具、tar 归档、硬软链接、权限模型、man/info、vim、SSH 密钥、scp/rsync、文件哈希。
>
> **验证方式和 09 类目不同,必须先读懂这一段:** 00-roadmap.md 原计划把本类目推迟到 Rocky Linux WSL 环境修复后统一完成,但截至撰写时该环境仍处于 `getpwuid(0) failed` 半初始化故障状态。本篇改为在 **Windows Git Bash**(GNU bash 5.1.16,x86_64-pc-msys;GNU grep 3.0 / GNU sed 4.8 / GNU findutils 4.9.0 / GNU tar 1.34 / GNU coreutils 8.32 / OpenSSH_9.0p1 / vim 8.2)下完成验证,14 个知识点按验证结论分两类,**不能一概而论**:
> - **A 类(shell 语法/GNU 工具本身的行为,和操作系统内核无关)**:第 1-7、14 项(文件操作、通配符、I/O 重定向、grep、sed、find、tar、哈希)。这些是 bash 内建语法或者真正的 GNU 二进制在处理,和 Rocky Linux 上的行为**没有差异**,已完整验证。
> - **B 类(依赖 Windows 内核/文件系统真实语义,和真实 RHEL 有实打实差异)**:第 8-13 项(硬软链接、chmod 权限模型、man/info、vim 的 `:wq!`、SSH、scp/rsync)。这几项**只验证了 Git Bash/NTFS 下的实际行为**,和真实 RHCSA 考试环境(RHEL 的 ext4/xfs + 完整 OpenSSH 生态)的差异会在每一条里如实指出,**不代表"已完整验证 RHCSA 环境下的真实效果"**。差异证据(真实报错文本、`mount`/`icacls` 输出)保留在正文里供复核,不是转述,是本机现场跑出来的。
>
> **和前四条系列的差异声明**(详见 [00-roadmap.md](00-roadmap.md) 开头):本仓库没有 Linux 系统管理场景可挖,"真实场景例子"部分统一标注为典型运维/RHCSA 考试场景,不冒充仓库代码里挖出来的。

---

## 1. 文件与目录基本操作回顾(`cp`/`mv`/`rm`/`mkdir` 的易错选项:`-r`/`-i`/`-v`/`-p`)

**命令/配置:**
```bash
cp -r src/ dst/     # 拷贝目录必须加 -r(recursive),拷贝单个文件不需要
cp -p src dst       # 保留源文件的时间戳/权限等元数据,默认不保留
cp -v src dst       # 打印"做了什么"(verbose),批量操作时确认用
rm -i file          # 删除前逐个确认(interactive)
mkdir -p a/b/c      # 一次创建多级目录,且目标已存在也不报错
mv src dst          # 改名(同目录)和移动(跨目录)是同一个命令
```

**一句话是什么:** 这几个命令看似是入门第一课,但选项组合是长期使用中最容易漏、最容易在自动化脚本里踩坑的地方——`-p` 在 `cp` 和 `mkdir` 里含义完全不同(`cp -p` 是"保留元数据",`mkdir -p` 是"允许多级、允许已存在"),千万不能望文生义地当成同一个意思记忆。

**为什么 RHCSA 真考 / 生产会用到:** 批量文件操作、防止脚本误删数据、保留原始时间戳做审计追溯,是运维日常里出现频率最高的基础操作;RHCSA 虽然不会单独出一道"考 cp 怎么用"的题,但几乎每道大题(备份、部署、恢复)的中间步骤都依赖这几个命令用对。

**从最容易犯错的做法讲起:** 拷贝目录忘记加 `-r`,`cp` 不会报"语法错误",而是明确拒绝执行并给出提示;批量删除脚本图省事直接 `rm -rf` 不做任何确认,一旦路径变量算错,后果不可逆——`-i` 虽然在全自动化脚本里通常会被 `-f` 覆盖掉,但手动操作、尤其是清理生产数据前,养成先 `-i` 确认的习惯比事后追悔更划算。

**真实场景例子(典型运维场景,非仓库代码):** 部署脚本在覆盖生产配置前,先用 `cp -p current.conf current.conf.bak` 保留原始文件的修改时间做审计追溯(而不是让备份文件的时间戳变成"刚刚",丢失"这份配置是什么时候生效的"这条线索);批量清空过期日志目录前,先用不加 `-f` 的 `rm -i *.log` 在少量文件上人工确认一遍,确认路径变量没写错,再切换成非交互的 `-f` 跑全量。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

start_dir=$(pwd)
demo_dir=$(mktemp -d /tmp/rhcsa01_demo1.XXXXXX)
cd "$demo_dir"

# mkdir -p:一次创建多级目录,重复执行也不报错(幂等)
mkdir -p project/src/nested
assert_ok test -d project/src/nested
assert_ok mkdir -p project/src/nested

# cp -p:保留时间戳(先把源文件时间强制改成一个过去的时间,再验证拷贝后是否一致)
echo "original" > original.txt
touch -d "2020-01-01 00:00:00" original.txt
cp -p original.txt preserved.txt
orig_mtime=$(stat -c '%Y' original.txt)
preserved_mtime=$(stat -c '%Y' preserved.txt)
assert_eq "$preserved_mtime" "$orig_mtime"

# cp 拷贝目录不加 -r:明确拒绝,不是模糊报错
mkdir srcdir && echo "nested file" > srcdir/inner.txt
cp srcdir nodir_copy 2>cp_err.txt
cp_exit=$?
assert_eq "$cp_exit" "1"
assert_eq "$(grep -c 'omitting directory' cp_err.txt)" "1"
cp -r srcdir dstdir
assert_eq "$(cat dstdir/inner.txt)" "nested file"

# rm -i:用 yes 命令模拟交互输入,验证 n(取消)和 y(确认)两种走向
touch will_be_asked.txt
yes n | rm -i will_be_asked.txt >/dev/null 2>&1
assert_ok test -f will_be_asked.txt      # 回答n,文件还在
yes y | rm -i will_be_asked.txt >/dev/null 2>&1
assert_ok test ! -e will_be_asked.txt    # 回答y,文件被删

# mv:改名和移动是同一个命令,取决于目标是"新文件名"还是"已存在的目录"
mv original.txt renamed.txt
assert_ok test -f renamed.txt
assert_ok test ! -e original.txt
mkdir archive
mv renamed.txt archive/
assert_ok test -f archive/renamed.txt

cd "$start_dir"
rm -rf "$demo_dir"
```
本机实测:全部 `assert_eq`/`assert_ok` 输出 `OK`。

**常见坑:**
1. `cp` 拷贝目录不加 `-r`,本机实测报错文本是 `cp: -r not specified; omitting directory 'srcdir'`,退出码 1——不是静默失败,而是明确拒绝,养成看返回值/错误信息的习惯就不会被这个坑绊倒。
2. `-p` 在 `cp`(preserve 元数据)和 `mkdir`(parents,允许多级/已存在)里含义完全不同,同一个字母不代表同一个意思,不能跨命令类推记忆。
3. `rm -i` 在没有人交互输入的自动化脚本环境里(比如 cron 任务、CI 流水线)会一直卡在等待确认——这也是为什么生产自动化脚本几乎都用 `-f` 跳过确认,而把"确认"这一步挪到脚本外层(比如需要人工审批才能触发脚本执行)。

---

## 2. 通配符与路径展开(glob:`*`、`?`、`[]`、`{}`)

**命令/配置:**
```bash
*        # 匹配任意长度(含0)的任意字符,但只匹配"确实存在"的文件名
?        # 匹配恰好一个字符
[abc]    # 匹配方括号内列举的任意一个字符
{a,b,c}  # 花括号展开:纯文本操作,不检查文件是否存在,发生在glob之前
```

**一句话是什么:** `*`/`?`/`[]` 是真正的**文件名通配符**——shell 会先检查磁盘上有没有对应的文件,只把"确实存在"的文件名传给命令;而 `{}` 是**花括号展开**,是纯文本层面的字符串组合游戏,跟文件系统、文件是否存在毫无关系,很多人把两者混为一谈。

**为什么 RHCSA 真考 / 生产会用到:** 批量重命名、按扩展名筛选一批文件、用 `mkdir -p project/{src,bin,docs}` 一次性创建标准项目骨架,是运维脚本和考试实操里的高频动作;搞混"glob"和"花括号展开"的区别,会在"为什么没匹配到文件却不报错"这类问题上卡很久。

**从最容易犯错的做法讲起:** 以为通配符没匹配到任何文件时,shell 会"聪明地"把它当成空、什么都不传;实际上 bash 默认行为是**原样保留字面量**,把 `*.xyz` 这几个字符本身当成一个文件名传给命令,导致下游命令收到一个看似"合法"但实际不存在的文件名,报出让人摸不着头脑的错误:
```bash
echo nomatch*.xyz        # 没有匹配文件,默认原样打印: nomatch*.xyz
ls nomatch*.xyz           # ls: cannot access 'nomatch*.xyz': No such file or directory
```
想要"没匹配到就当空值处理",要显式开启 `shopt -s nullglob`。

**真实场景例子(典型运维场景,非仓库代码):** 初始化一个新项目时用 `mkdir -p project/{src,bin,docs,tests}` 一次性把标准目录骨架建好,比写四条 `mkdir` 或者一条 `mkdir -p project/src project/bin project/docs project/tests` 都简洁;批量清理某一批命名规律固定的临时文件时用 `rm -f /tmp/session_[0-9][0-9].tmp` 精确匹配两位数字编号的文件,避免用 `*` 误删不该删的文件。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

start_dir=$(pwd)
demo_dir=$(mktemp -d /tmp/rhcsa01_demo2.XXXXXX)
cd "$demo_dir"

touch file1.txt file2.txt file10.txt notes.md

# * 匹配任意长度,按字典序排列(注意:file10.txt排在file2.txt前面,不是数值排序)
star_result=$(echo *.txt | tr ' ' ',')
assert_eq "$star_result" "file1.txt,file10.txt,file2.txt"

# ? 只匹配恰好一个字符,file10.txt有2个字符所以不匹配
question_result=$(echo file?.txt | tr ' ' ',')
assert_eq "$question_result" "file1.txt,file2.txt"

# [] 匹配方括号内列举的任意一个字符
bracket_result=$(echo file[12].txt | tr ' ' ',')
assert_eq "$bracket_result" "file1.txt,file2.txt"

# {} 花括号展开是纯文本操作,a.txt/b.txt/c.txt根本不存在,但照样原样展开成三个词
brace_result=$(echo {a,b,c}.txt | tr ' ' ',')
assert_eq "$brace_result" "a.txt,b.txt,c.txt"

# 默认(未开nullglob):没有匹配项时原样保留字面量,不会消失
no_match_result=$(echo nomatch*.xyz)
assert_eq "$no_match_result" "nomatch*.xyz"

# 开启nullglob后:没有匹配项的通配符展开成空字符串
shopt -s nullglob
no_match_nullglob=$(echo nomatch*.xyz)
assert_eq "$no_match_nullglob" ""
shopt -u nullglob

cd "$start_dir"
rm -rf "$demo_dir"
```
本机实测:六个 `assert_eq` 均输出 `OK`。

**常见坑:** 写"判断有没有匹配到文件"的脚本逻辑时,如果不开 `nullglob` 直接 `for f in *.xyz; do ...; done`,当没有任何 `.xyz` 文件时,循环体依然会执行一次,`$f` 的值是字面量字符串 `*.xyz`——这是新手写批处理脚本时"明明没有文件却处理了一次"的经典 bug 来源,健壮的写法要么开 `shopt -s nullglob`,要么循环体内部用 `[ -e "$f" ]` 再判断一次。

---

## 3. I/O 重定向进阶(`>`、`>>`、`2>`、`2>&1`、`&>`、管道 `|`)

**命令/配置:**
```bash
cmd > file        # 标准输出覆盖写入文件
cmd >> file        # 标准输出追加写入文件
cmd 2> file        # 标准错误写入文件
cmd > file 2>&1    # 标准输出写入文件,标准错误也重定向到"标准输出当前指向的位置"
cmd &> file        # bash扩展写法,等价于 > file 2>&1
cmd1 | cmd2        # 管道:cmd1的标准输出接到cmd2的标准输入(默认不连接stderr)
```

**一句话是什么:** 每个进程默认有 fd0(标准输入)/fd1(标准输出)/fd2(标准错误)三个通道,重定向的本质是"改变某个 fd 指向哪里",而 `2>&1` 这类写法是"把 fd2 指向 fd1**此刻**指向的位置"——是一次性的赋值,不是建立持续的绑定关系,这就是为什么重定向的**顺序**会直接影响结果。

**为什么 RHCSA 真考 / 生产会用到:** 巡检脚本、日志收集脚本几乎都要分离或合并标准输出和标准错误,方便事后分别归档"正常结果"和"报错信息";RHCSA 判分经常检查"脚本产生的错误信息有没有被正确处理",而不是放任报错信息刷屏到终端。

**从最容易犯错的做法讲起:** `command 2>&1 > file` 和 `command > file 2>&1` 顺序反过来,结果完全不同——很多人凭直觉以为"反正都写了这两个符号,顺序无所谓",这是最容易踩的坑:
```bash
# 错误顺序:2>&1 执行时,fd1还指向原来的地方(比如终端),之后 > file 只重定向了fd1
# 结果:标准错误还是跑去了原来的地方,没有进文件
ls /no/such/path 2>&1 > wrong_order.txt

# 正确顺序:先把fd1指向文件,再让fd2指向"fd1现在指向的地方"(也就是文件)
# 结果:标准错误正确进了文件
ls /no/such/path > right_order.txt 2>&1
```

**真实场景例子(典型运维场景,非仓库代码):** 系统巡检脚本用 `some_check.sh > /var/log/check.out 2> /var/log/check.err` 把"正常输出"和"错误信息"分别归档到两个文件,方便后续分别做成功率统计和报错分析;需要把两者合并成一份完整日志时用更简洁的 `some_check.sh &> /var/log/check.all`(bash 专有写法,`sh` 脚本里要写成 portable 的 `> file 2>&1`)。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

start_dir=$(pwd)
demo_dir=$(mktemp -d /tmp/rhcsa01_demo3.XXXXXX)
cd "$demo_dir"

# > 覆盖写, >> 追加写
echo "first" > out.txt
echo "second" >> out.txt
assert_eq "$(sed -n '1p' out.txt)" "first"
assert_eq "$(sed -n '2p' out.txt)" "second"
assert_eq "$(wc -l < out.txt | tr -d ' ')" "2"

# 2> 只捕获标准错误
ls nonexistent_xyz 2> err_only.txt
assert_eq "$(wc -l < err_only.txt | tr -d ' ')" "1"

# 顺序敏感:2>&1写在重定向之前,stderr会"泄漏"到外层原本的stdout,不会进目标文件
wrong_stdout_capture=$(mktemp)
bash -c "ls nonexistent_xyz 2>&1 > '$wrong_stdout_capture'" > leaked_to_outer_stdout.txt 2>leaked_to_outer_stderr.txt
assert_eq "$(cat "$wrong_stdout_capture")" ""                                  # 目标文件是空的(ls本身没有标准输出)
assert_eq "$(grep -c 'cannot access' leaked_to_outer_stdout.txt)" "1"          # 错误信息泄漏到了外层stdout
assert_eq "$(wc -c < leaked_to_outer_stderr.txt | tr -d ' ')" "0"              # 外层stderr是空的
rm -f "$wrong_stdout_capture"

# 顺序正确:> file 2>&1,标准错误被正确写进文件
ls nonexistent_xyz > right.txt 2>&1
assert_eq "$(grep -c 'cannot access' right.txt)" "1"

# &> 一步到位同时重定向stdout和stderr
{ echo "to stdout"; echo "to stderr" >&2; } &> combined.txt
assert_eq "$(wc -l < combined.txt | tr -d ' ')" "2"

# | 管道:连接标准输出到标准输入
pipe_result=$(printf 'b\na\nc\n' | sort | tr '\n' ',')
assert_eq "$pipe_result" "a,b,c,"

cd "$start_dir"
rm -rf "$demo_dir"
```
本机实测:全部 `assert_eq` 输出 `OK`。

**常见坑:** `&>` 是 bash 的专有扩展语法,POSIX `sh`(比如某些精简系统上 `/bin/sh` 指向 dash)不认识这个写法,写需要跨 shell 移植的脚本要老老实实写 `> file 2>&1`;管道 `|` 默认只连接标准输出,某条命令的报错信息不会被下一级命令看到,想连 stderr 一起接力要用 `|&`(bash 4.0+ 的简写,等价于 `2>&1 |`)。

---

## 4. grep 与正则表达式(BRE vs ERE,`-E`/`-P`,常用元字符)

**命令/配置:**
```bash
grep 'pattern' file      # 默认BRE(基础正则):+ ? | ( ) 是普通字符,要加反斜杠才有特殊含义
grep -E 'pattern' file    # ERE(扩展正则):+ ? | ( ) 原生生效,不需要转义
grep -P 'pattern' file    # PCRE(Perl正则):功能最强,但不一定所有环境都支持
```

**一句话是什么:** grep 默认用 **BRE**(Basic Regular Expression),`?`/`+`/`|`/`(`/`)` 这些在其他语言里司空见惯的正则元字符,在 BRE 里全都是**普通字符**,必须加反斜杠(`\?`/`\+`/`\|`)才会被当成元字符解释;`-E` 切换到 **ERE**(Extended),这些符号不转义就原生生效,和大多数编程语言的正则习惯一致;`-P` 是 **PCRE**,功能最全,但依赖具体的 grep 构建版本和运行环境。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 有大量"从日志/配置文件里过滤出符合条件的行"的任务,BRE/ERE 搞混会导致规则完全不生效(不报错,只是安静地匹配不到东西,排查起来比直接报错更麻烦)。

**从最容易犯错的做法讲起:** 在默认 BRE 模式下直接写 `grep 'cat|dog'` 期望匹配"cat 或 dog",实际上 `|` 在 BRE 里没有特殊含义,这行代码是在字面匹配"包含竖线字符 `|` 的行"——本机实测这个查询在测试数据里匹配了 **0 行**,而写成 `grep 'cat\|dog'`(转义)或者 `grep -E 'cat|dog'`(切换到ERE)才能正确匹配到 "cat or dog" 这一行。

**真实场景例子(典型运维场景,非仓库代码):** 从 `/var/log/messages` 或应用日志里过滤 `"ERROR|WARN|CRITICAL"` 多个级别的行做告警,日志分析脚本几乎必用 `-E` 做多模式匹配;用 `grep -E '^[0-9]{1,3}(\.[0-9]{1,3}){3}$'` 这类 ERE 写法校验一行文本是否是合法的 IPv4 地址格式(不追求完全严谨,但比手写一堆 `-o`/`-e` 拼接的 BRE 简洁得多)。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

start_dir=$(pwd)
demo_dir=$(mktemp -d /tmp/rhcsa01_demo4.XXXXXX)
cd "$demo_dir"

printf 'apple\nApple\nBANANA123\ncat or dog\ncolor\ncolour\naaa\n' > words.txt

# BRE(默认):| ? + 要转义才生效
bre_alt=$(grep 'cat\|dog' words.txt)
assert_eq "$bre_alt" "cat or dog"

bre_optional=$(grep 'colou\?r' words.txt | tr '\n' ',')
assert_eq "$bre_optional" "color,colour,"

bre_plus=$(grep 'a\+' words.txt | wc -l | tr -d ' ')
assert_eq "$bre_plus" "3"    # apple/cat or dog/aaa含小写a(grep默认区分大小写,Apple/BANANA123不算)

# BRE下不转义的|只是字面竖线字符,几乎匹配不到任何行
bre_literal_pipe=$(grep 'cat|dog' words.txt | wc -l | tr -d ' ')
assert_eq "$bre_literal_pipe" "0"

# ERE(-E):| ? + 原生生效,不需要转义
ere_alt=$(grep -E 'cat|dog' words.txt)
assert_eq "$ere_alt" "cat or dog"
ere_optional=$(grep -E 'colou?r' words.txt | tr '\n' ',')
assert_eq "$ere_optional" "color,colour,"

# 锚点^ $ 和字符类[]是BRE/ERE共有的基本元字符
anchor_result=$(grep '^[A-Z]' words.txt | tr '\n' ',')
assert_eq "$anchor_result" "Apple,BANANA123,"
digit_result=$(grep -E '[0-9]+$' words.txt)
assert_eq "$digit_result" "BANANA123"

# -P 的环境依赖:本机在LANG/LC_ALL为空的干净shell里,-P会直接报错退出码2
pcre_bare_exit=$(env -u LANG -u LC_ALL bash -c 'echo "abc123" | grep -P "\d+"' >/dev/null 2>&1; echo $?)
assert_eq "$pcre_bare_exit" "2"
# 显式指定UTF-8 locale后,-P恢复正常
pcre_fixed_result=$(env -u LANG LC_ALL=C.UTF-8 bash -c 'echo "abc123" | grep -P "\d+"')
assert_eq "$pcre_fixed_result" "abc123"

cd "$start_dir"
rm -rf "$demo_dir"
```
本机实测:全部 `assert_eq` 输出 `OK`。

**常见坑:**
1. BRE 下裸写 `|`/`?`/`+` 不会报语法错误,而是安静地退化成字面字符匹配,这是最隐蔽的坑——**本机实测**:`grep 'cat|dog' words.txt` 在含有 `cat or dog` 这一行的测试数据上返回 0 行结果,不报任何警告,新手很容易误以为是"数据有问题"而不是"正则语法用错了"。
2. `-P` 在本机 Git Bash 会话里(`LANG`/`LC_ALL` 环境变量为空)直接报错 `grep: -P supports only unibyte and UTF-8 locales`,退出码 2——即使 `locale` 命令显示系统默认值是 `C.UTF-8`,grep 依然要求 `LANG`/`LC_ALL` 这两个环境变量本身被**显式设置**,这是本机现场踩到的真实坑,加一句 `export LC_ALL=C.UTF-8` 就能修复,值得记录下来避免以后重复排查。

---

## 5. sed 流编辑器基础(替换/删除/行选择)

**命令/配置:**
```bash
sed 's/old/new/' file      # 替换:每行只替换第1次出现
sed 's/old/new/g' file      # 替换:g后缀表示替换全部出现
sed '2d' file                # 删除第2行
sed -n '2p' file             # -n关闭默认输出,配合p只打印匹配的行
sed -i.bak 's/x/y/' file    # 原地修改,.bak后缀自动生成修改前的备份
```

**一句话是什么:** sed 是"流编辑器"——逐行读入、按脚本规则处理、逐行输出到标准输出,**默认不修改原文件**(除非显式加 `-i`),`s`(substitute替换)/`d`(delete删除)/`p`(print打印,通常配合 `-n` 用)是最常用的三个子命令。

**为什么 RHCSA 真考 / 生产会用到:** 批量修改配置文件(比如把某一行的默认值改掉)、从命令输出里用行号/模式提取特定内容,是 RHCSA 和运维脚本的高频需求;sed 不用打开编辑器就能完成批量修改,特别适合写进自动化脚本里。

**从最容易犯错的做法讲起:** 直接用 `sed -i 's/old/new/' file` 原地修改配置文件,一旦替换规则写错(比如正则匹配范围过大),原文件直接被覆盖且没有备份,没法恢复;正确的防御性写法是 `sed -i.bak 's/old/new/' file`,让 sed 在原地修改前自动生成一份 `file.bak` 备份,改错了随时能找回原始内容。

**真实场景例子(典型运维场景,非仓库代码):** 批量把一批 `sshd_config` 文件里的 `PermitRootLogin yes` 改成 `PermitRootLogin no` 做安全加固:`sed -i.bak 's/^PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config`;从 `ip addr show` 的输出里用 sed 配合正则提取纯 IP 地址部分,是脚本化网络巡检的常见写法。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

start_dir=$(pwd)
demo_dir=$(mktemp -d /tmp/rhcsa01_demo5.XXXXXX)
cd "$demo_dir"

printf 'apple banana apple\nbanana\ncherry\n' > fruits.txt

# s/// 默认只替换每行第一次出现
sub_first=$(sed 's/apple/APPLE/' fruits.txt | sed -n '1p')
assert_eq "$sub_first" "APPLE banana apple"

# s///g 替换全部出现
sub_all=$(sed 's/apple/APPLE/g' fruits.txt | sed -n '1p')
assert_eq "$sub_all" "APPLE banana APPLE"

# d 删除指定行(只影响输出,不修改原文件)
after_delete=$(sed '2d' fruits.txt | wc -l | tr -d ' ')
assert_eq "$after_delete" "2"
assert_eq "$(sed '2d' fruits.txt | tail -1)" "cherry"

# -n 配合 p 只打印匹配的行
only_line2=$(sed -n '2p' fruits.txt)
assert_eq "$only_line2" "banana"
range_result=$(sed -n '1,2p' fruits.txt | wc -l | tr -d ' ')
assert_eq "$range_result" "2"

# -i.bak 原地修改并自动生成备份
cp fruits.txt fruits_original.txt
sed -i.bak 's/cherry/CHERRY/' fruits.txt
assert_eq "$(tail -1 fruits.txt)" "CHERRY"
assert_ok test -f fruits.txt.bak
diff_result=$(diff fruits.txt.bak fruits_original.txt > /dev/null; echo $?)
assert_eq "$diff_result" "0"     # 备份文件和原始内容完全一致

cd "$start_dir"
rm -rf "$demo_dir"
```
本机实测:全部 `assert_eq`/`assert_ok` 输出 `OK`。

**常见坑:**
1. `s/old/new/` 默认只替换每行**第一次**出现,忘记加 `g` 是最常见的"改了一部分,还有一部分没改"的困惑来源。
2. `sed 'Nd'` 这类基于行号的操作,行号是"处理时的行号",如果一次性删除多行(比如循环删除),前一次删除会导致后面的行号整体往前移,批量按固定行号删除多行时容易被这个偏移坑到——更安全的做法是按**模式**匹配删除(`sed '/pattern/d'`),而不是硬编码行号。

---

## 6. find 查找文件(按名称/类型/时间/大小/权限,`-exec`)

**命令/配置:**
```bash
find path -name "*.txt"      # 按文件名(必须加引号!)
find path -type f            # 按类型:f文件 d目录
find path -size +5k          # 按大小:+大于 -小于
find path -newer ref_file     # 比参照文件更新的文件
find path -exec cmd {} \;    # 对每个匹配结果单独调用一次命令
find path -exec cmd {} +     # 尽量批量调用,类似xargs,更快
```

**一句话是什么:** find 从起始路径开始逐个检查每个文件/目录是否满足一串"测试条件"(name/type/size/……),条件之间默认是 AND 关系,`-exec` 能让每一个匹配到的结果都被送去执行任意命令——find/grep/sed 是 RHCSA 文本处理三大工具,find 负责"定位文件",grep/sed(见本类目第4、5项)负责"处理内容"。

**为什么 RHCSA 真考 / 生产会用到:** "找出所有超过某大小的文件"、"找出N天前的临时文件并清理"是 RHCSA 实操和日常运维的高频操作,几乎每个磁盘清理、备份筛选脚本都离不开 find。

**从最容易犯错的做法讲起:** `find . -name *.txt` 不加引号,如果当前目录里**存在两个或以上**匹配的 `.txt` 文件,`*.txt` 会先被 shell 自身展开成多个具体文件名(比如 `alpha.txt beta.txt`),而不是原样传给 find——本机实测,find 遇到这种情况会直接报错并给出诊断提示:
```
$ find . -name *.txt
find: paths must precede expression: `beta.txt'
find: possible unquoted pattern after predicate `-name'?
```
GNU find 足够智能,能识别出"这很可能是没加引号导致的"并给出提示,但依然是一次执行失败(退出码1)。正确写法必须给通配符加引号 `-name "*.txt"`,让 find 自己解释通配符,而不是让 shell 提前展开。

**真实场景例子(典型运维场景,非仓库代码):** 清理 7 天前的临时文件:`find /tmp -mtime +7 -type f -exec rm {} \;`;巡检占用空间过大的文件:`find / -xdev -size +100M -type f -exec ls -lh {} \;`(`-xdev` 限制在同一文件系统内搜索,避免跨越到挂载的网络存储)。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

start_dir=$(pwd)
demo_dir=$(mktemp -d /tmp/rhcsa01_demo6.XXXXXX)
cd "$demo_dir"

mkdir -p subdir
touch a.txt b.log c.txt subdir/d.txt
dd if=/dev/zero of=bigfile.bin bs=1024 count=10 2>/dev/null

# -name(加引号!)按文件名匹配
name_result=$(find . -name "*.txt" | sort | tr '\n' ',')
assert_eq "$name_result" "./a.txt,./c.txt,./subdir/d.txt,"

# -type 按类型
type_f_count=$(find . -type f | wc -l | tr -d ' ')
assert_eq "$type_f_count" "5"      # a.txt b.log c.txt subdir/d.txt bigfile.bin
type_d_count=$(find . -type d | wc -l | tr -d ' ')
assert_eq "$type_d_count" "2"      # . 和 subdir

# -size 按大小(bigfile.bin是10KB)
size_result=$(find . -size +5k)
assert_eq "$size_result" "./bigfile.bin"

# -newer 找出比参照文件更新的文件
touch reference.marker
sleep 1
echo "fresh" > freshfile.txt
newer_result=$(find . -newer reference.marker -type f)
assert_eq "$newer_result" "./freshfile.txt"

# -exec ... \; 逐个调用 vs -exec ... + 批量调用
exec_semicolon_calls=$(find . -name "*.txt" -exec echo "called" \; | wc -l | tr -d ' ')
assert_eq "$exec_semicolon_calls" "4"    # 4个txt文件,每个单独触发一次echo
exec_plus_calls=$(find . -name "*.txt" -exec echo "called" {} + | wc -l | tr -d ' ')
assert_eq "$exec_plus_calls" "1"          # 4个文件打包进1次echo调用,只有1行输出

# 不加引号的通配符被shell提前展开,匹配到2个以上文件时find直接报错
find . -name *.txt >unquoted_out.txt 2>&1
unquoted_exit=$?
assert_eq "$unquoted_exit" "1"
assert_eq "$(grep -c 'paths must precede expression' unquoted_out.txt)" "1"

cd "$start_dir"
rm -rf "$demo_dir"
```
本机实测:全部 `assert_eq` 输出 `OK`。

**常见坑:**
1. `-name` 不加引号被 shell 提前展开的坑(现场复现,见上方"从最容易犯错的做法讲起"和可运行例子最后一段)。
2. `-exec cmd {} \;` 会对每一个匹配结果单独 fork 一次命令(文件多的时候很慢),`-exec cmd {} +` 会尽量把多个文件打包成一次命令调用(类似 `xargs` 的行为,显著更快),两者最终效果一样但性能差异在大量文件场景下很明显。
3. **诚实说明**:`-perm` 精确权限匹配这一项在本机 Windows/NTFS 环境下**没有参考价值**——因为 `chmod` 本身在这个环境下不能真正设置精确的权限位(见本类目第9项的详细说明),`find . -perm 600` 在明明存在一个"看起来已经chmod 600"的文件时会返回空结果,这不是 find 的问题,是上游 chmod 就没有真正生效,`-perm` 的真实语义必须在真实 RHEL 环境下才能验证。

---

## 7. tar 归档与压缩(与 gzip/bzip2/xz 的组合参数)

**命令/配置:**
```bash
tar czf archive.tar.gz dir/     # c创建 z用gzip压缩 f指定归档文件名
tar cjf archive.tar.bz2 dir/    # j用bzip2压缩
tar cJf archive.tar.xz dir/     # J用xz压缩
tar tzf archive.tar.gz          # t列出内容(不解压)
tar xzf archive.tar.gz -C dst/   # x解压,-C指定目标目录
```

**一句话是什么:** tar 本身只做"打包"(把多个文件合并成一个 `.tar` 流,不压缩),压缩是交给 gzip/bzip2/xz 三种外部算法之一来做的——`c`(create)/`x`(extract)/`t`(list)/`f`(指定文件名)是最常用的四个模式选项,`z`/`j`/`J` 分别指定用哪种压缩算法联动。

**为什么 RHCSA 真考 / 生产会用到:** 备份与恢复、软件分发、日志归档,RHCSA 官方明确要求"能用 tar 创建和还原归档",是最基础的必考操作之一。

**从最容易犯错的做法讲起:** 拿到一个来源不明的 tar 包,不先看内容就直接 `tar x` 解压——如果包内路径写的是绝对路径或者带 `../` 向上跳出,解压可能会覆盖系统文件或者跳出预期目录(这是真实存在的安全隐患,不是危言耸听)。正确习惯是先 `tar tf archive.tar.gz` 预览内容确认没问题,再决定要不要解压,尤其是处理外部来源、不受信任的归档文件时。

**真实场景例子(典型运维场景,非仓库代码):** 备份脚本按数据特点选择压缩算法——日志这类文本重复率高、要求速度快的用 `gzip`(压缩率一般但最快);冷备份、长期归档、非常在意体积的用 `xz`(压缩率最高但最慢,本机实测同样内容 xz 压缩耗时明显长于 gzip);介于两者之间需要折中时用 `bzip2`。三种格式互不兼容,解压时必须用对应的参数(或者用不指定压缩算法的 `tar xf`,GNU tar 能自动识别归档使用的压缩格式)。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

start_dir=$(pwd)
demo_dir=$(mktemp -d /tmp/rhcsa01_demo7.XXXXXX)
cd "$demo_dir"

mkdir -p src/nested
echo "hello" > src/file1.txt
echo "world" > src/nested/file2.txt

# 三种压缩算法都能正常创建归档
tar czf archive.tar.gz src
tar cjf archive.tar.bz2 src
tar cJf archive.tar.xz src
assert_ok test -f archive.tar.gz
assert_ok test -f archive.tar.bz2
assert_ok test -f archive.tar.xz

# t 列出内容而不解压,养成先看后解的习惯
listed=$(tar tzf archive.tar.gz | sort | tr '\n' ',')
assert_eq "$listed" "src/,src/file1.txt,src/nested/,src/nested/file2.txt,"

# x 解压后内容必须和原始文件完全一致
mkdir extracted
tar xzf archive.tar.gz -C extracted
diff_result=$(diff -r src extracted/src > /dev/null; echo $?)
assert_eq "$diff_result" "0"
assert_eq "$(cat extracted/src/nested/file2.txt)" "world"

# f后面必须紧跟归档文件名,但f在选项字符串里的书写位置(cfz还是czf)不影响解析结果
tar cfz archive_altorder.tar.gz src
same_listing=$(tar tzf archive_altorder.tar.gz | sort | tr '\n' ',')
assert_eq "$same_listing" "$listed"

cd "$start_dir"
rm -rf "$demo_dir"
```
本机实测:全部 `assert_eq`/`assert_ok` 输出 `OK`。

**常见坑:**
1. `tar czf archive.tar.gz dir` 和 `tar cfz archive.tar.gz dir`(`f` 和 `z` 顺序互换)效果完全相同——很多人以为 `f` 必须写在选项串最后,其实规则是"`f` 后面紧跟的**第一个位置参数**就是归档文件名",和 `f` 在字母组合里的书写顺序无关,但**参数本身的顺序**(文件名必须紧跟在这串选项之后)不能变,写反了会把目录名当成归档文件名去创建,把预期打包的目录当成了目标文件名。
2. 解压未知来源的归档前不预览内容是真实的安全隐患,恶意构造的 tar 包可能包含 `../../etc/cron.d/evil` 这样的路径,解压时覆盖到预期目录之外。
3. **诚实说明**:tar 的"保留权限"这个卖点(`tar` 默认会记录并在解压时还原文件的权限位)在本机 Windows/NTFS 环境下**验证不出真实语义**——因为源文件本身的权限位就是本类目第9项里说明的"假权限"(NTFS 上 chmod 不能精确设置 ugo/rwx),tar 只能忠实地把这份"假权限"打包进归档、解压时再原样应用回来,garbage-in-garbage-out,这不是 tar 的 bug,而是上游权限模型本身在这个环境下就不真实,必须在真实 RHEL 上才能验证 tar 权限保留的真实效果。

---

## 8. 硬链接与软链接(inode 本质区别,`ln -s` 常见坑)

**命令/配置:**
```bash
ln target hardlink       # 硬链接:同一个inode的另一个目录项
ln -s target symlink     # 软链接:独立的文件,内容是指向目标的路径字符串
```

**inode 是什么(本节标题就叫"inode 本质区别",但这个词本身从未被正式定义过,这里先补上):** 一个文件在磁盘上其实被拆成两部分存:一个叫 **inode** 的结构体,存这个文件的全部元信息(权限、大小、属主、修改时间、指向真正数据块的指针……),但**唯独不存文件名**;文件名只存在于**目录**里,是目录这个特殊文件内部的一条"条目(entry)",内容是"某个名字 → 某个 inode 编号"这样一条映射。也就是说,你平时说的"文件名",其实只是目录指向 inode 的一个指针的名字,不是文件本身的一部分——这也是为什么"改文件名"(`mv`)几乎是瞬间完成的:改的只是目录里那条映射,inode 和它指向的真实数据完全没有被移动。

**一句话是什么:** 有了上面这层认识,硬链接和软链接的区别就很直接了——硬链接是"同一份数据在目录树里的另一个名字"(本质是在某个目录里新增一条指向**同一个 inode 编号**的条目,两个名字完全对等,内核用**引用计数**追踪有多少条目指向同一个 inode,删掉其中一个只是删掉了这条目录条目,只要引用计数还大于 0、还有至少一个名字存在,inode 和数据就不会被真正释放);软链接是一个**独立的、有自己 inode 的**特殊文件,内容是一串指向目标路径的字符串,目标被删除或改名后软链接会变成"悬空链接"(broken symlink)——因为它存的只是一个路径字符串,不是 inode 编号,自然不会跟着目标的 inode 走。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 官方技能列表明确写着"create and configure hard and soft links",是必考点;理解 inode 引用计数的本质,也是排查"磁盘明明满了但 `find` 找不到大文件在哪"这类问题的关键——被删除但仍被某个进程打开的文件,数据依然占着磁盘空间,本质也是"还有一个引用(这次是进程的文件描述符而不是目录项)指向这个 inode"。

**从最容易犯错的做法讲起:** 混淆软硬链接的适用边界——硬链接**不能**跨文件系统/跨分区(下面会用本机真实报错验证)、**不能**对目录使用(防止目录树出现环,破坏树状结构);软链接**可以**跨文件系统、**可以**指向目录,但目标被删除/改名后会失效变成悬空链接,`ls -l --color` 通常会用醒目颜色标出悬空链接提醒你。

**真实场景例子(典型运维场景,非仓库代码):** `/etc/alternatives` 机制、`/usr/bin/python` 这类软链接指向具体版本的可执行文件(比如 `/usr/bin/python3.11`),切换默认版本时只需要重新指向,不用真的挪动/复制任何文件,是软链接在生产系统上最经典的用法。

**可运行例子(以下内容需要重点阅读——这是全篇 Windows/NTFS 和真实 RHEL 差异最大的一条,如实记录了本机现场探测到的真实行为):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

start_dir=$(pwd)
demo_dir=$(mktemp -d /tmp/rhcsa01_demo8.XXXXXX)
cd "$demo_dir"

# ---- 硬链接:NTFS原生支持,在Git Bash下是真实、完整可验证的 ----
echo "hard link content" > target.txt
ln target.txt hardlink.txt
target_inode=$(stat -c '%i' target.txt)
hardlink_inode=$(stat -c '%i' hardlink.txt)
assert_eq "$hardlink_inode" "$target_inode"     # 同一个inode,证明是真硬链接,不是拷贝
link_count=$(stat -c '%h' target.txt)
assert_eq "$link_count" "2"                       # 链接计数变成2
echo "appended via hardlink" >> hardlink.txt
assert_eq "$(wc -l < target.txt | tr -d ' ')" "2"    # 通过hardlink.txt写入,target.txt也能看到:同一份数据

# ---- 跨盘符(跨设备)硬链接:真实报错(本机用D盘测试;没有第二个盘符会自动跳过) ----
if [ -d /d ] && (echo probe > /d/rhcsa01_cross_probe_$$.txt) 2>/dev/null; then
    cross_err=$(ln /d/rhcsa01_cross_probe_$$.txt ./cross_hardlink.txt 2>&1)
    cross_exit=$?
    assert_eq "$cross_exit" "1"
    assert_eq "$(echo "$cross_err" | grep -ic 'cross-device')" "1"
    rm -f /d/rhcsa01_cross_probe_$$.txt
else
    echo "SKIP:本机没有可写的第二个盘符,跳过跨设备测试(真实报错文本见下方常见坑)"
fi

# ---- 软链接:命令报告成功(exit 0),但Windows默认权限下不是真正的符号链接 ----
echo "original target content" > symtarget.txt
ln -s symtarget.txt mysymlink.txt
assert_eq "$?" "0"                                # ln -s 报告执行成功……

readlink mysymlink.txt >/dev/null 2>&1
assert_eq "$?" "1"                                # ……但readlink证明它根本不是符号链接(真符号链接readlink会成功)

symlink_inode=$(stat -c '%i' mysymlink.txt)
symtarget_inode=$(stat -c '%i' symtarget.txt)
[ "$symlink_inode" != "$symtarget_inode" ] && echo "OK: mysymlink.txt是独立inode,不是target的另一个名字(也不是硬链接)"

# 决定性证据:修改target后,"symlink"不会跟着变,证明它只是创建那一刻的一份静态拷贝
echo "MODIFIED - real symlink would show this" > symtarget.txt
stale_content=$(cat mysymlink.txt)
assert_eq "$stale_content" "original target content"

cd "$start_dir"
rm -rf "$demo_dir"
```
本机实测:全部断言输出 `OK`(跨设备部分本机因为存在可写的 D 盘,实际执行并通过,没有走 SKIP 分支)。

**常见坑(本篇分量最重的一节,请完整阅读):**

1. **`ln -s` "成功"了,但根本不是符号链接,而是一份会悄悄过期的静态拷贝——这是本文档最重要的发现,比"直接报错"更危险。** 创建真正的 NTFS 符号链接需要 `SeCreateSymbolicLinkPrivilege` 权限(通常要开启 Windows"开发者模式"或以管理员身份运行),本机既没有开发者模式也不是管理员权限。按常理预期,`ln -s` 应该会报 `Permission denied` 之类的错误——但本机实测完全不是这样:`ln -s` **报告执行成功(退出码0)**,`ls -la` 显示的类型是普通文件(`-rw-r--r--`)而不是符号链接该有的 `lrwxrwxrwx`,`readlink` 对它调用直接失败(证明它不是符号链接),`stat` 显示它有自己独立的 inode 且链接计数是1(证明它也不是硬链接)——**它是 `ln` 在拿不到符号链接权限时,静默把目标文件的内容整个复制了一份**。用 PowerShell 的 `Get-Item` 交叉验证:这个"symlink"文件的 `Attributes` 只有普通的 `Archive`,`LinkType` 属性是空的,不是 `SymbolicLink`。最要命的是随后修改了 `target.txt` 的内容后,这个"symlink"文件的内容完全没有跟着变——这才是决定性证据:它是创建那一刻的一份孤立拷贝,不是一个活的引用。如果只看退出码判断"命令成功了就是符号链接建好了",会在数据悄悄过期很久之后才发现问题,排查起来会走很多弯路。
2. 用 `MSYS=winsymlinks:nativestrict ln -s target.txt strict_symlink.txt` 可以强制关闭这种静默降级、要求必须创建真正的原生符号链接,本机实测这样跑会得到期望中的真实报错:`ln: failed to create symbolic link 'strict_symlink.txt': Operation not permitted`,退出码1,且文件确实没有被创建——这印证了默认行为是"用静默拷贝掩盖了本该出现的权限错误"。如果要在 Windows 上写依赖符号链接语义的脚本,强烈建议设置这个环境变量,让失败清清楚楚地失败,而不是悄悄退化成一个看起来正常、实际上是定时炸弹的拷贝。
3. 对**目录**执行 `ln -s dirtarget dirlink` 同样"成功"且同样不是真链接——本机实测这是一次**递归拷贝**(创建后往 `dirtarget` 里新增文件,`dirlink` 不会同步出现新文件),行为和文件的情况完全一致,不是"文件退化成拷贝、目录用某种真实机制(比如 junction)顶替"这种更温和的降级。
4. 硬链接不能跨文件系统/跨分区,本机用 C 盘(`/tmp` 所在盘)和 D 盘做跨盘符测试,真实报错文本是 `ln: failed to create hard link ... Invalid cross-device link`——这一点和真实 Linux 的行为、错误提示文字都是一致的(因为这是内核级限制,不是 NTFS 特有问题,ext4 之间挂载在不同分区一样会报同样的错)。
5. **上述第1-3条只反映 Git Bash + 当前 Windows 权限配置下的行为,不代表所有 Windows 环境都这样**——如果目标机器开启了开发者模式或者以管理员权限运行,`ln -s` 大概率会创建真正的符号链接;这也不代表真实 RHCSA 考试环境(RHEL)的行为,RHEL 上 `ln -s` 创建的就是标准、无条件生效的符号链接,不存在这里描述的任何降级问题——这几条常见坑是"Windows 上用 Git Bash 学 Linux 命令"这个学习路径本身特有的陷阱,不是 RHCSA 要考的内容,但了解它能避免被自己的验证环境误导出错误的心智模型。

---

## 9. 标准权限模型 ugo/rwx(`chmod` 符号法 vs 八进制法)

**命令/配置:**
```bash
chmod u+x file      # 符号法:u/g/o/a(身份) + +/-/=(操作) + rwx(权限)
chmod g-w file
chmod a=r file
chmod 750 file       # 八进制法:r=4 w=2 x=1,三个数字分别对应 u/g/o 的权限总和
```

**一句话是什么:** Linux 权限模型是"三类身份(owner/group/other)× 三种权限(read/write/execute)"的 3×3 矩阵,符号法(`u`/`g`/`o`/`a` + `+`/`-`/`=` + `rwx`)描述的是**相对于当前状态的改变**,八进制法(比如 `750`)描述的是**目标状态的绝对值**,两者语法不同但改的是同一组权限位。

**把"3×3 矩阵"这句话真正画出来(以 `750` 为例):**
```
              r(4)   w(2)   x(1)
            ┌──────┬──────┬──────┐
  owner(7)  │  ✓   │  ✓   │  ✓   │   7 = 4+2+1(rwx 全开)
            ├──────┼──────┼──────┤
  group(5)  │  ✓   │      │  ✓   │   5 = 4+1(r-x,能读能执行,不能改)
            ├──────┼──────┼──────┤
  other(0)  │      │      │      │   0(什么都不行)
            └──────┴──────┴──────┘
```
每一行是一类身份,每一列是一种权限,一个 3×3 共 9 个格子里"点亮"了哪些,决定了这个八进制数字/符号法字符串具体是什么。

**r/w/x 对"文件"和对"目录"分别意味着什么(这一点极其容易被忽略,04 类第 9 节排查权限问题时才会真正用到,这里先建立完整认识,不留到后面才提):**
| 权限 | 对**文件**意味着 | 对**目录**意味着 |
|---|---|---|
| `r` | 能否读取文件内容(`cat` 之类) | 能否列出目录里有哪些文件名(`ls`) |
| `w` | 能否修改文件内容 | 能否在目录里新建/删除文件(**注意**:删除一个文件看的是目录的 `w` 权限,不是文件自己的权限——这是新手最容易搞反的一点) |
| `x` | 能否把这个文件当程序**执行** | 能否"进入"这个目录、访问目录里文件的元信息——这个权限有个专门的名字叫**"搜索权限"**,只有 `x` 没有 `r` 的目录,你没法 `ls` 看到里面有什么文件名,但如果你已经知道某个文件的确切名字,依然可以直接访问它(比如 `cat 目录/已知文件名.txt` 能成功,但 `ls 目录` 会被拒绝)——04 类第 9 节会用 `namei` 命令现场复现这个反直觉的行为 |

**为什么 RHCSA 真考 / 生产会用到:** `chmod` 是 RHCSA 最基础的必考操作之一,权限位算错会导致后续依赖这些权限的题目(共享目录、脚本可执行、服务读取配置)连带出错——而且系统管理场景有一个 python-idioms 系列完全没有的评分维度:**过度授权本身就是错误答案**,不是"能达到效果就算对"。

**从最容易犯错的做法讲起:** 遇到"权限不够"的报错,图省事直接 `chmod 777` 让报错消失——这是安全大忌,RHCSA 会因为"权限过度开放"直接扣分甚至判定这道题不通过,即便功能表现上"看起来是对的"。正确做法是先搞清楚问题到底出在 owner/group/other 哪个身份、缺哪个权限,用最小权限原则精确补上,而不是无脑放开全部权限掩盖问题。

**真实场景例子(典型运维场景,非仓库代码):** 给团队共享的部署脚本设置 `750`(owner 自己能读写执行、同组的人能读能执行但不能改、其他人完全不能碰),比 `777` 安全得多,也是 RHCSA 判分会关注的细节。

**可运行例子(以下内容同样需要重点阅读——这是本篇第二个必须诚实处理的 Windows/NTFS 差异点):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

start_dir=$(pwd)
demo_dir=$(mktemp -d /tmp/rhcsa01_demo9.XXXXXX)
cd "$demo_dir"

# 语法层面:符号法和八进制法都能正常执行,退出码都是0(命令语法本身没问题)
touch syntax.txt
chmod u+x syntax.txt;  assert_eq "$?" "0"
chmod g-w syntax.txt;  assert_eq "$?" "0"
chmod a=r syntax.txt;  assert_eq "$?" "0"
chmod 750 syntax.txt;  assert_eq "$?" "0"

# 八进制算法本身(r=4 w=2 x=1)是纯数学计算,和操作系统无关,可以独立验证
calc_owner=$(( 4 + 2 + 1 )); assert_eq "$calc_owner" "7"    # rwx
calc_group=$(( 4 + 1 ));      assert_eq "$calc_group" "5"    # r-x

# ---- 诚实边界:以下验证的是"NTFS到底强制了什么",不是chmod的完整ugo/rwx语义 ----
# mount显示Git Bash的NTFS是noacl挂载,不使用Windows ACL模拟9个独立权限位
mount_noacl_lines=$(mount | grep -c 'noacl')
assert_ok test "$mount_noacl_lines" -ge 1

for mode in 644 600 400 200 700 755 000 666; do
    touch "perm_$mode.txt"
    chmod "$mode" "perm_$mode.txt"
done
# 请求"owner写位=1"的模式,不论具体数值是什么,NTFS上一律显示成644
assert_eq "$(ls -l perm_644.txt | awk '{print $1}')" "-rw-r--r--"
assert_eq "$(ls -l perm_600.txt | awk '{print $1}')" "-rw-r--r--"    # 请求600,实际显示644,不是600!
assert_eq "$(ls -l perm_700.txt | awk '{print $1}')" "-rw-r--r--"    # 请求700,实际显示644
assert_eq "$(ls -l perm_666.txt | awk '{print $1}')" "-rw-r--r--"
# 不请求"owner写位"的模式,统一折叠成444
assert_eq "$(ls -l perm_400.txt | awk '{print $1}')" "-r--r--r--"    # 请求400,实际显示444,不是400!
assert_eq "$(ls -l perm_200.txt | awk '{print $1}')" "-rw-r--r--"    # 200含义是"只写不读",owner写位=1,同样折叠成644,读写语义完全丢失
assert_eq "$(ls -l perm_000.txt | awk '{print $1}')" "-r--r--r--"

# 更关键的是:即便ls -l显示成r--r--r--(看起来"只读"),owner自己的cat依然能读到内容——
# 这个环境唯一真实存在的维度是"整个文件能不能被写",压根没有"不同身份读权限不同"这回事
echo "secret" > noperm.txt
chmod 000 noperm.txt
cat_result=$(cat noperm.txt 2>&1)
cat_exit=$?
assert_eq "$cat_exit" "0"
assert_eq "$cat_result" "secret"    # chmod 000在真实RHEL上会让cat直接Permission denied,这里完全没挡住

cd "$start_dir"
rm -rf "$demo_dir"
```
本机实测:全部断言输出 `OK`。

**常见坑:**
1. **误以为 Windows 上"ls -l 显示的权限位在变化"就等于"chmod 真的生效了"。** 本机实测(`mount` 命令确认 Git Bash 的 NTFS 是以 `noacl` 方式挂载的):无论 `chmod` 请求的八进制值是 `644`/`600`/`700`/`666` 中的哪一个,只要请求里 owner 的写位是 1,`ls -l` 一律显示成 `rw-r--r--`(644);只要 owner 写位是 0(比如 `400`/`000`),一律折叠显示成 `r--r--r--`(444)。也就是说 NTFS 通过 Git Bash 实际只追踪了**一个二元状态**——"整个文件能不能被写"(对应 Windows 原生的"只读"文件属性),group/other 各自独立的读写执行位、以及 owner 的执行位,全部是**显示层面的模拟**,不是真正生效的内核强制。用 `icacls` 查看同一个文件的 Windows 原生 ACL 可以看到,`chmod` 命令执行前后,真正的 Windows ACL 权限条目(继承自父目录的 `(I)` 标记权限)完全没有变化——`chmod` 改的从始至终只是那一个 DOS 只读属性位。
2. 即使 `ls -l` 显示成看起来"最严格"的 `r--r--r--`(对应请求的 `chmod 000`),owner 自己的 `cat` 依然能正常读到文件内容(本机实测退出码0,内容原样输出)——这个环境里根本不存在"某个身份被禁止读取"这回事,`r--r--r--` 只是"不能写"的另一种显示形式,不是真的把读权限精确分配给了三类身份。
3. **本文在 Git Bash/NTFS 下只验证了 chmod 命令语法本身能正确执行(符号法/八进制法怎么写、八进制怎么算)、以及"owner写位"这一个维度确实能真实控制文件能不能被覆盖,但 NTFS 不会像 ext4/xfs 那样把 ugo 三类身份 × rwx 三种权限的完整 9 个位都强制生效——这和真实 RHCSA 考试环境(真实 RHEL 文件系统)的实际强制效果不同,标准权限模型的完整语义(尤其是"其他用户/同组用户被真正拒绝访问")必须在真实 Linux 环境下才能验证,本文不能也不会声称这一条"已完整验证"。**

---

## 10. man/info 与系统自带文档(`--help`、`man -k`、`/usr/share/doc`)

**命令/配置:**
```bash
man command        # 打开命令的手册页(按章节分类:1普通命令 5配置文件格式 8系统管理命令)
man -k keyword     # 关键词搜索所有手册页标题(等价于 apropos)
man 5 passwd        # 显式指定查第5章(文件格式),而不是默认的第1章(命令本身)
command --help      # 程序自带的精简用法提示,不依赖man数据库
ls /usr/share/doc    # RHEL上每个rpm包安装时留下的详细文档/示例配置目录
```

**一句话是什么:** `man` 是 Linux 最标准的命令手册系统,按数字分章节(1 是普通命令,5 是配置文件格式,8 是系统管理命令,同一个名字在不同章节可能是完全不同的内容),`--help` 是程序自带的精简提示,`/usr/share/doc` 下有各软件包附带的详细文档和示例配置。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 考试环境官方明确允许查阅 `man` 手册页(这是一场"上机操作考"而不是"闭卷背诵考"),熟练用 `man -k`/`man 5 xxx` 快速定位陌生命令的精确参数,是真实生产环境和考试共同需要的技能,比死记硬背所有参数更现实也更可靠。

**从最容易犯错的做法讲起:** 不知道 man 分章节,比如直接 `man passwd` 默认只会打开第1章(`passwd` 这个**命令**本身的用法),而 `/etc/passwd` 这个**文件的格式说明**其实在第5章,要显式 `man 5 passwd` 才能看到——这是新手"明明查了 man 却没找到我要的内容"困惑的常见来源,一个名字在不同章节可能对应完全不同的主题。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 考试中途忘记 `firewall-cmd` 某个参数的精确写法,考场允许开一个终端 `man firewall-cmd` 现查用法示例,比死记硬背所有子命令的拼写更可靠;不确定某个配置文件该怎么写,先 `man -k` 搜索一下有没有对应章节的手册页,再用 `man 5 xxx` 精确定位。

**可运行例子(诚实说明:本机 Git Bash 未安装 man/info,以下只能验证探测结果本身和 `--help` 部分,man 的具体交互行为未能现场验证):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

# 第一步:如实探测,不假设
man_present=0; command -v man >/dev/null 2>&1 && man_present=1
info_present=0; command -v info >/dev/null 2>&1 && info_present=1
assert_eq "$man_present" "0"     # 本机确认未安装man
assert_eq "$info_present" "0"    # 本机确认未安装info

doc_dir_exists=0; [ -d /usr/share/doc ] && doc_dir_exists=1
assert_eq "$doc_dir_exists" "0"   # /usr/share/doc 在本机也不存在(不是RHEL的rpm数据库体系)

# 能验证的部分:--help不依赖man数据库,是程序自己打印的,可以现场验证
help_output=$(ls --help 2>&1)
assert_eq "$(echo "$help_output" | head -1)" "Usage: ls [OPTION]... [FILE]..."
assert_eq "$(grep --help | grep -c -- '--extended-regexp')" "1"
assert_eq "$(grep --help | grep -c -- '--perl-regexp')" "1"

echo "本机确认:man/info/\`/usr/share/doc\` 均不存在,--help可正常使用"
```
本机实测:三个 `assert_eq` 均输出 `OK`(探测结果本身是可验证的事实)。

**常见坑:**
1. `man passwd` 默认打开第1章、`/etc/passwd` 文件格式在第5章,不显式指定章节号会找错内容,这是本条最容易踩的坑。
2. **本文没有、也无法验证真实 `man` 交互界面下的翻页/搜索操作(`/pattern` 页内搜索、`q` 退出等)**,因为本机 Git Bash 没有安装 `man`;这部分需要在真实 RHEL(或者 Rocky Linux WSL 环境修复后)现场验证,在此如实标注,不冒充已经验证过。

---

## 11. vim 基础操作(三种模式切换、常用命令、`:wq!` 的坑)

**命令/配置:**
```bash
vim file       # 打开文件,默认进入"普通模式"
i / a / o      # 普通模式下进入"插入模式"(在光标前/后插入、新开一行插入)
Esc            # 任意时候回到普通模式
:              # 普通模式下进入"命令行模式"
:wq  :q!  :wq!  # 保存退出 / 强制不保存退出 / 强制保存退出
```

**一句话是什么:** vim 是模式编辑器,核心是"普通模式(移动/删除/复制)"、"插入模式(打字)"、"命令行模式(`:` 开头,保存/退出/替换/查找)"三种模式之间的切换,几乎所有新手的痛苦都来自"不知道自己当前在哪个模式"。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 考试环境不保证有 nano/gedit 这类图形化或更简易的编辑器,官方明确假设考生会用 `vi`/`vim` 修改配置文件,这是能不能完成考试的最基础前提,不会用 vim 基本上等于放弃了大半张卷子。

**从最容易犯错的做法讲起:** 在插入模式下直接敲 `:wq`,会发现这几个字符被原样打进了正文内容里,而不是被当成命令执行——因为根本没有先按 `Esc` 退出插入模式,vim 还在"一切输入都是要打进文档里的文字"这个状态。这是新手最典型的"vim 怎么退不出去"窘境的根源,正确姿势永远是:先按 `Esc` 确认回到普通模式,再敲 `:` 开始输入命令。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 编辑 `/etc/fstab` 或 `sshd_config` 忘记先 `sudo`,vim 打开的是没有写权限的文件,正常 `:wq` 保存会失败——这时候要分清楚两种情况:如果只是 vim **自己**因为某种原因认为这是只读 buffer(比如用 `vim -R` 或 `view` 打开的),`:wq!` 能强制覆盖 vim 自己的保护、正常写入;但如果是操作系统层面本来就没给你写权限(没有 sudo 编辑 root 拥有的文件),`:wq!` 同样会失败报错——`!` 只能覆盖 vim 自己的内部状态,不能让操作系统内核网开一面。

**可运行例子(用 `vim -Es` 非交互脚本化模式验证,不需要真实终端交互):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

start_dir=$(pwd)
demo_dir=$(mktemp -d /tmp/rhcsa01_demo11.XXXXXX)
cd "$demo_dir"

# vim -Es(Ex模式+silent)可以在没有真实终端的情况下脚本化验证编辑效果
# 务必加 < /dev/null:不给vim任何标准输入来源,避免它在异常分支里等待交互输入导致脚本挂起
printf 'line one\nline two\nline three\n' > doc.txt
vim -Es -c '2s/two/TWO/' -c 'wq' doc.txt < /dev/null
assert_eq "$(sed -n '2p' doc.txt)" "line TWO"

# :wq(不带!)在vim自己认为"只读"的buffer(-R参数模拟)上会失败——这是vim内部状态,不是操作系统权限
printf 'abc\n' > ro.txt
vim -Es -R -c '%s/abc/xyz/' -c 'wq' ro.txt < /dev/null >/dev/null 2>&1
wq_noforce_exit=$?
assert_eq "$wq_noforce_exit" "1"
assert_eq "$(cat ro.txt)" "abc"     # 没有写入成功,内容还是原样

# :wq!(带感叹号)强制覆盖vim自己的只读保护,写入成功
vim -Es -R -c '%s/abc/xyz/' -c 'wq!' ro.txt < /dev/null >/dev/null 2>&1
wq_force_exit=$?
assert_eq "$wq_force_exit" "0"
assert_eq "$(cat ro.txt)" "xyz"

cd "$start_dir"
rm -rf "$demo_dir"
```
本机实测:全部 `assert_eq` 输出 `OK`(vim 8.2)。

**常见坑:**
1. 插入模式下直接敲 `:wq` 把命令打进了正文——见上方"从最容易犯错的做法讲起"。
2. `:wq!` 常被误解成"万能强制保存",实际上它只能覆盖 **vim 自己**内部的保护状态(比如用 `-R`/`view` 打开、或者"内容改了但还没保存"这类 vim 层面的警告);如果文件在**操作系统**层面本来就没有写权限(比如没有 sudo 去改一个 root 拥有的文件),`:wq!` 同样会失败,报 `E212: Can't open file for writing` 之类的错误——这是本条例子特意用 `-R`(vim 自己的只读标记)而不是用 `chmod` 去构造"只读"场景的原因:后者在本机 Windows/NTFS 环境下的权限语义本身就不可靠(见本类目第9项),用 vim 自己的 `-R` 参数能干净地演示"vim 内部保护 vs 操作系统权限"这两种不同层次的只读,不会被环境问题干扰。
3. 用 `vim -Es`(Ex 脚本模式)做非交互自动化编辑时,一定要显式重定向标准输入(`< /dev/null`)——本机现场踩过这个坑:不加这个重定向,当某个 `-c` 命令执行失败(比如上面例子里在只读 buffer 上执行不带 `!` 的 `wq`)时,vim 会尝试等待进一步的交互输入,脚本会直接卡死不返回,而不是像预期那样打印错误后立刻退出。

---

## 12. SSH 密钥认证登录(`ssh-keygen`、`ssh-copy-id`、`~/.ssh/config`)

**命令/配置:**
```bash
ssh-keygen -t ed25519 -f keyfile -N "" -C "comment"    # 本地生成密钥对
ssh-copy-id -i keyfile.pub user@host                    # 把公钥追加到远程authorized_keys
```
`~/.ssh/config`:
```
Host myserver
    HostName 192.0.2.10
    User deploy
    Port 2222
    IdentityFile ~/.ssh/id_ed25519
```

**一句话是什么:** SSH 密钥认证基于非对称加密的"挑战应答"机制,本地留私钥(绝不外传)、公钥追加到远程主机的 `~/.ssh/authorized_keys`;`ssh-copy-id` 就是自动化"把公钥内容追加到远程"这一步的工具;`~/.ssh/config` 让你给常用连接起别名、固化参数,不用每次都打一长串命令行选项。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"配置基于密钥的 SSH 认证",禁用密码登录改用密钥认证是生产安全基线的第一步,也是自动化部署(脚本免交互登录远程主机)的前提。

**从最容易犯错的做法讲起:** 私钥文件权限设置得过于开放(比如变成组内/其他人可读),真实 Linux 上 OpenSSH 客户端/服务端会**直接拒绝使用这把私钥**,报出 `UNPROTECTED PRIVATE KEY FILE!` 这类醒目警告并中止认证——这是 RHCSA 考试和生产环境里极其常见的"密钥对本身没问题,就是登录不上"排障点,新手往往会去反复检查密钥内容对不对,而忽略了检查私钥文件本身的权限位。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 给自动化部署账号配置免密登录,批量给多台机器分发同一把公钥,让 CI/CD 流水线能够无人值守地通过 SSH 连接目标服务器执行部署命令。

**可运行例子(诚实说明:本地生成密钥对可以完整验证;`ssh-copy-id` 和真实登录需要可连接的目标主机,本环境没有,不冒充已验证,用 `ssh -G` 做不需要网络连接的配置文件语法验证作为替代):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

start_dir=$(pwd)
demo_dir=$(mktemp -d /tmp/rhcsa01_demo12.XXXXXX)
cd "$demo_dir"

# 本地生成密钥对:完全可验证,不需要任何远程主机(注意:-f指定的是本demo目录下的临时文件,
# 不会碰用户真实的 ~/.ssh 目录)
ssh-keygen -t ed25519 -f id_demo -N "" -C "rhcsa-demo" >/dev/null 2>&1
assert_ok test -f id_demo
assert_ok test -f id_demo.pub

# 私钥文件格式:OpenSSH私钥固定以这一行开头
assert_eq "$(head -1 id_demo)" "-----BEGIN OPENSSH PRIVATE KEY-----"

# 公钥文件格式:算法类型 + base64编码内容 + 注释,固定三段式,空格分隔
pubkey_algo=$(awk '{print $1}' id_demo.pub)
pubkey_comment=$(awk '{print $3}' id_demo.pub)
assert_eq "$pubkey_algo" "ssh-ed25519"
assert_eq "$pubkey_comment" "rhcsa-demo"

# ~/.ssh/config 语法验证:ssh -G 只打印"解析后生效的配置",不实际发起网络连接
cat > test_ssh_config << 'EOF'
Host myserver
    HostName 192.0.2.10
    User deploy
    Port 2222
    IdentityFile ~/.ssh/id_demo
EOF
resolved=$(ssh -F test_ssh_config -G myserver 2>/dev/null)
resolved_hostname=$(echo "$resolved" | awk '/^hostname /{print $2}')
resolved_user=$(echo "$resolved" | awk '/^user /{print $2}')
resolved_port=$(echo "$resolved" | awk '/^port /{print $2}')
assert_eq "$resolved_hostname" "192.0.2.10"
assert_eq "$resolved_user" "deploy"
assert_eq "$resolved_port" "2222"

cd "$start_dir"
rm -rf "$demo_dir"
```
本机实测:全部 `assert_eq`/`assert_ok` 输出 `OK`。

**常见坑:**
1. 私钥权限过开在真实 Linux 上会被 OpenSSH 直接拒绝使用——**本文无法验证这一点在本机是否会触发**:本机用 `stat` 查看 `ssh-keygen` 生成的私钥文件权限,显示的是 `644`(而不是 OpenSSH 期望的 `600`),这和本类目第9项讲的"NTFS 上 chmod 无法精确设置权限位"是同一个根因;本机实测尝试用这把 644 权限的私钥连接一个不存在的主机,ssh 在**域名解析**阶段就失败了(`Could not resolve hostname`),根本没有走到"检查私钥权限"这一步,所以连"本机会不会触发这个检查"都无法确认——如实标注:这一条必须在真实 RHEL 环境下、用可达的目标主机才能验证。
2. `ssh-copy-id` 本质只是把公钥内容通过 SSH 追加写入远程 `~/.ssh/authorized_keys` 文件末尾;理解了这个本质,即使没有 `ssh-copy-id` 这个工具(比如精简系统没装),手动 `cat id_demo.pub | ssh user@host "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"` 也能达到完全一样的效果——**本文没有可连接的真实目标主机,`ssh-copy-id` 本身的执行效果和上述手动等价写法均未能现场验证**,只描述其原理和标准用法。

---

## 13. scp/rsync 文件传输(增量同步、保留权限)

**命令/配置:**
```bash
scp -r src/ user@host:/dst/       # 全量复制,-r递归复制目录
rsync -avz src/ user@host:/dst/    # 增量同步:a归档模式(保留属性+递归) v详细输出 z压缩传输
```

**一句话是什么:** `scp` 是"每次全量复制"(哪怕文件只改了1个字节也要完整重传整个文件),`rsync` 用差量算法只传输"变化的部分";两者都能通过 SSH 做加密传输,但 `rsync` 功能明显更强大(支持排除规则、断点续传、镜像删除同步等),`scp` 更简单、历史上几乎所有系统都自带。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"用 `scp` 在系统间安全传输文件";`rsync` 虽不一定在 RHCSA 必考范围内,但是生产环境事实标准的同步工具,一并掌握性价比很高。

**从最容易犯错的做法讲起:** 用 `scp` 复制目录忘记加 `-r`——本机实测,Windows 上 Git 自带的这个 `scp`(基于 Win32-OpenSSH)在缺少 `-r` 复制目录时报的错误文本和 `cp` 一模一样:`cp: -r not specified; omitting directory 'srcdir'`,退出码1,说明这个 `scp` 实现在处理本地路径参数时直接复用了 `cp` 的底层例程;`rsync` 最经典的坑是**源路径结尾带不带斜杠 `/` 结果天差地别**——`rsync -av src/ dst/` 是把 `src/` **目录里的内容**同步进 `dst/`,而 `rsync -av src dst/`(源路径不带斜杠)是把 `src` **这个目录本身**同步进 `dst/` 下面、变成 `dst/src/`,这是 `rsync` 使用者几乎人人都踩过的坑。

**真实场景例子(典型运维场景,非仓库代码):** 每晚增量同步 Web 服务器日志到备份服务器,`rsync -avz --delete /var/log/nginx/ backup-host:/backup/nginx-logs/`,只传输当天新增/变化的部分,大幅节省带宽和时间,`--delete` 让备份端和源端保持镜像一致(源端删除的文件备份端也同步删除,使用前要想清楚是不是真的需要这个行为)。

**可运行例子(诚实说明:`rsync` 本机未安装,`command -v` 探测确认后不做执行,只讲解语法;`scp` 本机存在,但没有可连接的远程主机,以下只验证了它在"无主机部分"时退化成本地拷贝的边界行为和 `-r` 参数的报错场景,**不代表已经验证了它的核心远程传输能力**):
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

start_dir=$(pwd)
demo_dir=$(mktemp -d /tmp/rhcsa01_demo13.XXXXXX)
cd "$demo_dir"

# rsync在本机Git Bash里未安装,如实探测确认,不假装能跑
rsync_present=0; command -v rsync >/dev/null 2>&1 && rsync_present=1
assert_eq "$rsync_present" "0"

# scp:验证"没有主机部分(两个参数都不含冒号)时退化成本地拷贝"这个边界行为
# 这只证明scp二进制能正确复制字节,不涉及它的核心功能(通过SSH做远程加密传输)
echo "local content" > src.txt
scp src.txt dst.txt >/dev/null 2>&1
assert_eq "$?" "0"
assert_eq "$(cat dst.txt)" "local content"

# scp -r 忘记加:对目录直接报错(不需要远程主机就能验证)
mkdir srcdir && echo "x" > srcdir/f.txt
scp srcdir destdir 2>scp_err.txt
scp_dir_exit=$?
assert_eq "$scp_dir_exit" "1"
assert_eq "$(grep -c 'omitting directory' scp_err.txt)" "1"
scp -r srcdir destdir >/dev/null 2>&1
assert_eq "$(cat destdir/f.txt)" "x"

cd "$start_dir"
rm -rf "$demo_dir"
```
本机实测:全部 `assert_eq` 输出 `OK`。

**常见坑:**
1. `rsync` 源路径结尾带不带 `/` 结果天差地别(见上方"从最容易犯错的做法讲起")——这是网上求职面试和实际运维都会被问到的经典点,即便本机没装 `rsync` 无法现场跑,原理必须记清楚:**结尾带 `/` 指的是"目录里的内容",不带 `/` 指的是"目录本身"**。
2. `scp` 复制目录忘记 `-r` 的报错本机可以现场复现(见可运行例子),但**核心的远程传输能力(加密信道建立、密码/密钥认证、断点相关行为)完全没有验证**,因为没有可连接的第二台主机——如实标注,不冒充已验证。
3. `rsync` 在本机 Git Bash 环境下确认未安装(`command -v rsync` 无输出),如果需要用,要么用 Rocky Linux/真实 Linux 环境,要么装 MSYS2 的 rsync 包,本文不做进一步安装尝试(避免为了"演示能跑"而改变本机环境配置)。

---

## 14. 文件哈希校验(`sha256sum`/`md5sum` 及一致性校验用法)

**命令/配置:**
```bash
sha256sum file              # 计算SHA-256哈希
sha256sum file > sums.txt    # 保存校验文件
sha256sum -c sums.txt        # 核对文件内容是否和记录时一致
md5sum file                  # 计算MD5哈希(不适合安全场景,见下方常见坑)
```

**一句话是什么:** 哈希算法把任意长度的文件内容压缩成一个固定长度的"指纹",只要文件内容有哪怕 1 bit 的变化,指纹就会几乎完全不同(雪崩效应),常用来校验文件传输/下载过程有没有损坏或被篡改。

**为什么 RHCSA 真考 / 生产会用到:** 校验 ISO 镜像完整性、核实备份文件没有在传输过程中损坏,是 RHCSA 和生产运维的常规动作;理解"哈希校验的是内容、不是权限/元数据"也是本类目和第9项(权限模型)的一个对比记忆点——两者是完全独立的维度。

**从最容易犯错的做法讲起:** 把 `md5sum` 用在需要抵御"故意伪造"的安全场景(比如验证软件签名、密码存储)——MD5 早已被证明存在**碰撞攻击**(能人为构造出内容不同但 MD5 相同的两个文件),只适合"检测意外损坏"这种非对抗性场景;生产安全相关的场景一律应该用 `sha256` 或更高强度的算法,这不是"哪个更时髦"的选择,是真实的安全边界问题。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 下载 RHEL/Rocky Linux 官方 ISO 后,官网会同时发布一个 `CHECKSUM` 文件,下载完成后先 `sha256sum -c CHECKSUM` 确认镜像文件没有下载损坏、也没有被中间人篡改,确认无误后再刻录安装介质——这是安装操作系统前的标准前置动作,跳过这一步在生产环境是不专业的做法。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

start_dir=$(pwd)
demo_dir=$(mktemp -d /tmp/rhcsa01_demo14.XXXXXX)
cd "$demo_dir"

echo "hello world" > data.txt
sha256_val=$(sha256sum data.txt | awk '{print $1}')
assert_eq "$sha256_val" "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"
assert_eq "${#sha256_val}" "64"     # sha256固定64个十六进制字符(256bit)

md5_val=$(md5sum data.txt | awk '{print $1}')
assert_eq "$md5_val" "6f5902ac237024bdd0c176cb93063dc4"
assert_eq "${#md5_val}" "32"        # md5固定32个十六进制字符(128bit)

# -c 一致性校验:先生成校验文件,再核对
sha256sum data.txt > data.sha256
check_ok=$(sha256sum -c data.sha256 2>&1)
assert_eq "$(echo "$check_ok" | grep -c ': OK')" "1"

# 人为篡改文件内容后,同一份校验文件必须能检测出FAILED
echo "tampered" >> data.txt
check_fail_output=$(sha256sum -c data.sha256 2>&1)
check_fail_exit=$?
assert_eq "$check_fail_exit" "1"
assert_eq "$(echo "$check_fail_output" | grep -c ': FAILED')" "1"

# 哈希校验的是内容,文件大小相同不代表内容相同(常见的误判来源)
echo "AAAAAAAAAA" > same_size_1.txt
echo "BBBBBBBBBB" > same_size_2.txt
size1=$(stat -c '%s' same_size_1.txt)
size2=$(stat -c '%s' same_size_2.txt)
assert_eq "$size1" "$size2"    # 大小相同
hash1=$(sha256sum same_size_1.txt | awk '{print $1}')
hash2=$(sha256sum same_size_2.txt | awk '{print $1}')
[ "$hash1" != "$hash2" ] && echo "OK: 大小相同但内容不同,哈希值完全不同"

cd "$start_dir"
rm -rf "$demo_dir"
```
本机实测:全部 `assert_eq` 输出 `OK`,大小相同内容不同的两个文件哈希值确认不同。

**常见坑:**
1. `sha256sum -c` 要求校验文件里记录的文件名/相对路径,和当前核对时的相对位置**完全匹配**——校验文件是在哪个目录下生成的,就要在同一个相对位置执行核对,路径对不上会报 `No such file or directory` 而不是"校验不通过",容易被误判成"文件丢失"而不是"校验姿势不对"。
2. 判断两个文件内容是否相同,比较哈希值远比"肉眼比较内容"或者"比较文件大小"可靠——本机实测两个内容不同但字节数完全相同的文件(各11字节),哈希值完全不同,证明"大小相同"不能作为"内容相同"的任何证据。
3. MD5 不适合安全场景(见"从最容易犯错的做法讲起"),但作为"快速检测文件是否意外损坏"依然完全够用且计算更快,不要把"MD5不安全"和"MD5没用"划等号,场景不同结论不同。

---

*本篇完成:2026-07-11,14 个知识点。验证环境:Windows Git Bash(GNU bash 5.1.16)。第1-7、14 项(A类,shell/GNU工具本身语法)完整验证,和 Rocky Linux 上的行为一致。第8-13 项(B类,依赖 Windows 内核/文件系统真实语义)按各条内文如实标注的范围验证:硬链接、chmod 语法、vim、ssh-keygen 本地部分完整验证;软链接的"实际行为"完整验证(确认是静默拷贝而非真符号链接,这本身就是一条真实的差异记录);chmod 的完整 ugo/rwx 强制效果、man/info 交互、ssh-copy-id、scp/rsync 的远程传输能力受限于本机环境(无 man/info、无 rsync、无可达的第二台主机),未能验证,已在对应小节如实注明,需要在 Rocky Linux 环境修复后补充验证。*
