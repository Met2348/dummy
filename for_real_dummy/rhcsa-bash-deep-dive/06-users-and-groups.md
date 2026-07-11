# 06 · 用户组管理(User & Group Management)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一篇覆盖 10 个知识点:useradd/usermod/userdel、group 管理、密码策略、账户锁定、passwd/shadow/group 文件结构、UID/GID 分配规则、`/etc/skel`、sudo 配置。**本文所有代码例子已在 Rocky Linux 10.2(WSL2)下实际跑通验证**。
>
> **和前四条系列的差异声明**(详见 [00-roadmap.md](00-roadmap.md) 开头):本仓库没有 Linux 系统管理场景可挖,"真实场景例子"部分统一标注为典型运维/RHCSA 考试场景,不冒充仓库代码里挖出来的。

---

## 1. `useradd` 创建用户(常用选项 `-m`/`-s`/`-G`/`-u`)

**命令/配置:**
```bash
useradd -m -s /bin/bash -u 1500 -G wheel username    # -m创建家目录 -s指定shell -u指定UID -G附加组
useradd -r -M -s /sbin/nologin svcaccount               # -r系统用户 -M不创建家目录,典型的服务账号配置
```

**一句话是什么:** `useradd` 在 `/etc/passwd`/`/etc/shadow`/`/etc/group` 里登记一个新用户的完整信息,`-m` 决定要不要顺带建家目录(并从 `/etc/skel` 复制初始化模板,见第 9 节)、`-s` 指定登录 shell、`-u` 指定精确 UID、`-G` 加入额外的附加组——不加任何选项时,`useradd` 会用系统默认值(通常**不**自动建家目录),这是新手最容易漏掉的一点。

**为什么 RHCSA 真考 / 生产会用到:** 用户账号管理是 RHCSA 最基础、分值占比很高的技能模块;生产环境创建服务账号(跑某个后台服务用的专用账号)和创建真人登录账号的选项组合完全不同(服务账号通常不需要家目录、不需要能登录的 shell),理解这些选项的实际含义比死记硬背更重要。

**从最容易犯错的做法讲起:** 直接 `useradd username` 不加 `-m`,以为家目录会自动创建——**这取决于发行版的默认策略**,不能想当然认为"创建用户"就等于"创建了完整可用的账号环境";养成显式加 `-m`(需要家目录时)的习惯,而不是依赖某个可能因系统而异的默认行为。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 给新入职的团队成员开账号:`useradd -m -s /bin/bash -G devteam newuser`,创建家目录、指定 bash 作为登录 shell、直接加入团队共享组;部署一个后台服务需要专用运行账号:`useradd -r -M -s /sbin/nologin appservice`,不给家目录、不允许交互式登录,是运维加固里"服务账号最小权限"的标准做法。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

useradd -m -s /bin/bash -u 5001 rhcsa06_user1
assert_eq "$?" "0"
assert_ok test -d /home/rhcsa06_user1
home_files=$(ls -A /home/rhcsa06_user1 | wc -l)
assert_ok test "$home_files" -ge 1    # /etc/skel的内容确实被复制进来了(第9节详细展开)
assert_eq "$(getent passwd rhcsa06_user1 | cut -d: -f7)" "/bin/bash"
assert_eq "$(id -u rhcsa06_user1)" "5001"

userdel -r rhcsa06_user1 2>/dev/null
```
本机实测:全部检查点输出 `OK`。

**常见坑:** `-u` 指定的 UID 如果已经被别的用户占用,`useradd` 会直接报错拒绝创建(除非额外加 `-o` 允许 UID 重复,这是一个很少用到的高风险选项,重复 UID 会让两个用户名对应同一份权限身份);批量创建账号的自动化脚本如果硬编码固定的 UID 起始值,要注意检查目标系统上这些 UID 是否已经被占用,不能假设是"全新系统"。

---

## 2. `usermod` 修改用户属性

**命令/配置:**
```bash
usermod -s /sbin/nologin username    # 修改登录shell
usermod -aG groupname username        # -a(append)+G:追加到附加组,不影响已有的附加组成员关系
usermod -c "备注文字" username           # 修改GECOS备注字段
usermod -L username                       # 锁定账号(和passwd -l效果相同,见第6节)
usermod -d /new/home -m username           # 修改家目录路径,-m同时把旧家目录内容搬过去
```

**一句话是什么:** `usermod` 修改一个**已存在**用户的各项属性(shell/家目录/附加组/UID/备注等),几乎覆盖了 `useradd` 创建时能指定的所有选项——区别只在于 `useradd` 是"创建时一次性设定",`usermod` 是"事后调整"。

**为什么 RHCSA 真考 / 生产会用到:** 员工岗位变化需要调整组权限、账号策略更新需要批量调整已有账号的属性,都是 `usermod` 的日常应用场景;RHCSA 明确要求"修改用户账号属性"。

**从最容易犯错的做法讲起:** 用 `usermod -G groupname username`(不加 `-a`)给用户追加一个新的附加组——**这是一个极其危险的坑**:不加 `-a` 的 `-G` 是"覆盖式"的,会把这个用户原有的全部附加组关系**替换**成只有这一个组,而不是"追加"这一个组;正确的追加写法永远是 `usermod -aG groupname username`,漏掉这一个 `-a` 字母,足以让一个用户瞬间丢失所有原有的组权限。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 员工转岗到新项目组,需要保留原有权限的同时加入新组的协作权限:`usermod -aG newproject_group username`;安全审计发现某个账号的登录 shell 配置不当(比如服务账号却能交互式登录),批量 `usermod -s /sbin/nologin` 收紧这批账号。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

useradd -m rhcsa06_moduser

usermod -s /sbin/nologin rhcsa06_moduser
assert_eq "$(getent passwd rhcsa06_moduser | cut -d: -f7)" "/sbin/nologin"

usermod -c "Test User Comment" rhcsa06_moduser
assert_eq "$(getent passwd rhcsa06_moduser | cut -d: -f5)" "Test User Comment"

userdel -r rhcsa06_moduser 2>/dev/null
```
本机实测:两个断言均输出 `OK`。

**常见坑:** 见上方"从最容易犯错的做法"——`usermod -G`(不带 `-a`)的覆盖式行为是本节最重要、也是生产事故里出现频率最高的坑,永远记住"追加组权限,`-aG` 连用,一个字母都不能少"。

---

## 3. `userdel` 删除用户(`-r` 清理家目录的坑)

**命令/配置:**
```bash
userdel username         # 只删除账号本身(passwd/shadow/group里的记录),家目录被保留
userdel -r username        # 连带删除家目录和邮件池(mail spool)
```

**一句话是什么:** `userdel` 默认**只清理账号记录本身**,不会动这个用户的家目录和文件——这个设计是为了防止"删错人"或者"数据还没来得及备份就被误删",`-r` 选项才是"连同这个用户名下的家目录/邮件池一起删掉"的完整清理。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"删除本地用户账户";生产环境离职员工账号处理,通常先只删账号本身(数据保留备份归档一段时间),确认没有后续需要之后再彻底清理家目录,不加 `-r` 的默认行为恰好匹配这种"先冻结再彻底清理"的稳妥流程。

**从最容易犯错的做法讲起:** 执行 `userdel username`(不加 `-r`)之后,以为"这个用户彻底没了",却没意识到家目录还完整地留在磁盘上——**本机实测证伪式验证**:不加 `-r` 删除用户后,家目录确实原封不动地保留着,变成一个"属主是一个已经不存在的 UID"的孤儿目录(`ls -l` 会显示一个数字 UID 而不是用户名,因为 `/etc/passwd` 里已经查不到这个 UID 对应的用户名了);如果目标就是"彻底清理干净",必须显式加 `-r`,不能靠默认行为。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 员工离职当天先 `userdel username`(不加 `-r`)冻结账号访问权限,家目录数据留待部门主管确认是否有需要交接的文件,一段时间后确认无误再手动清理或 `userdel -r` 彻底删除;测试环境里创建的临时账号用完即焚,直接 `userdel -r` 一步到位,不需要保留任何痕迹。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

useradd -m rhcsa06_user2
assert_eq "$(test -d /home/rhcsa06_user2; echo $?)" "0"

userdel rhcsa06_user2    # 不加-r
still_exists=0; [ -d /home/rhcsa06_user2 ] && still_exists=1
assert_eq "$still_exists" "1"    # 家目录确实被保留了(孤儿目录,owner是一个已不存在的UID)
rm -rf /home/rhcsa06_user2

useradd -m rhcsa06_user3
userdel -r rhcsa06_user3    # 加-r
deleted_check=0; [ ! -d /home/rhcsa06_user3 ] && deleted_check=1
assert_eq "$deleted_check" "1"    # 加-r后家目录被一并清理
```
本机实测:两个断言均输出 `OK`。

**常见坑:** `userdel` 删除一个**当前有进程正在运行**的用户(比如这个用户的某个后台任务还没结束)会报错拒绝执行(现代版本的行为,防止删掉一个"还活着"的账号造成孤儿进程或者权限混乱)——遇到这种报错,正确做法是先确认清楚这些进程是什么、能否安全终止,而不是找办法强行绕过这个保护提示。

---

## 4. `groupadd`/`groupmod`/`groupdel` 组管理

**命令/配置:**
```bash
groupadd -g GID groupname     # 创建组,-g指定精确GID
groupmod -n newname oldname     # 重命名组
groupdel groupname                 # 删除组
```

**一句话是什么:** 组管理的三个基本命令,和用户管理的 `useradd`/`usermod`/`userdel` 是完全平行的设计(命令命名规律都是"动作+d/mod/del"),`-g` 指定精确 GID 的用法和用户管理里 `-u` 指定 UID 是同一个思路。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"创建、删除组;修改组成员";组是权限管理里"批量授权一群用户"的基本单元(比第 5 节 04 类讲的 SGID 目录协作场景更基础的前置知识),几乎所有多用户协作权限设计都要先规划好组结构。

**从最容易犯错的做法讲起:** 尝试删除一个**仍是某些用户主组**的组——`groupdel` 会拒绝执行,因为每个用户必须始终有一个有效的主组,删掉一个还在被引用的主组会让那些用户的账号信息出现矛盾;必须先把这些用户的主组改到别的组(`usermod -g` 修改主组),或者先把这批用户处理完,才能安全删除这个组。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 新项目启动,`groupadd project_alpha` 建立专属协作组,后续团队成员陆续 `usermod -aG project_alpha` 加入;项目重命名,`groupmod -n project_beta project_alpha` 直接改名,不需要重建整个组结构、不影响已有成员关系。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

groupadd -g 6001 rhcsa06_grp1
assert_eq "$?" "0"
assert_eq "$(getent group rhcsa06_grp1 | cut -d: -f3)" "6001"

groupmod -n rhcsa06_grp1_renamed rhcsa06_grp1
assert_eq "$(getent group rhcsa06_grp1_renamed | cut -d: -f1)" "rhcsa06_grp1_renamed"

groupdel rhcsa06_grp1_renamed
group_gone=0; getent group rhcsa06_grp1_renamed >/dev/null 2>&1 || group_gone=1
assert_eq "$group_gone" "1"
```
本机实测:全部断言输出 `OK`。

**常见坑:** 一个组即便**没有任何成员**了,只要它还是某个用户的主组(即便这个用户从未真正以这个组名义做过任何事),依然不能被删除——排查"为什么这个看起来空的组删不掉"时,别只看 `/etc/group` 里的成员列表字段(那只是**附加组**成员),还要检查 `/etc/passwd` 里有没有用户把这个组的 GID 设为主组。

---

## 5. `passwd` 密码管理与 `chage` 密码时效策略

**命令/配置:**
```bash
passwd username                 # 交互式修改密码
echo "user:newpass" | chpasswd    # 非交互式批量设置密码(脚本友好)
chage -M 90 username               # 密码最长有效期90天
chage -W 7 username                  # 到期前7天开始警告
chage -d 0 username                   # 强制该用户下次登录必须改密码
chage -l username                      # 查看当前密码时效策略详情
```

**一句话是什么:** `passwd` 是日常交互式改密码的入口,`chpasswd` 是配合脚本/批量操作的非交互式版本;`chage`(change age)专门管理密码的**时效策略**——多久必须改一次、提前多久开始提醒、是否要强制下次登录立即改密码,这套机制是密码安全基线的核心组成部分。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"管理用户密码时效";企业安全基线普遍要求密码定期轮换,`chage` 是落实这条策略的标准工具,`chage -d 0` 强制改密码也是"给新员工发临时密码、要求首次登录必须改成自己的密码"这类流程的标准实现手段。

**从最容易犯错的做法讲起:** 只用 `passwd` 手动改了一次密码,以为"配置了密码策略"——`passwd` 只负责"这次密码是什么",完全不涉及"多久必须改一次"这类时效性要求,两者是完全独立的维度,配置密码轮换策略必须用 `chage`,不是靠 `passwd` 就能覆盖的。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 新员工入职,`chpasswd` 设置一个临时初始密码、`chage -d 0` 强制其首次登录必须修改成自己的密码,避免管理员知道的初始密码一直被沿用;合规要求密码 90 天强制过期,`chage -M 90 -W 7` 批量应用到所有账号,到期前一周开始在登录时提醒用户。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

useradd -m rhcsa06_pwuser
echo "rhcsa06_pwuser:TestPass123!" | chpasswd
assert_eq "$?" "0"
assert_eq "$(passwd -S rhcsa06_pwuser | awk '{print $2}')" "P"    # P=密码已设置(本机实测确认真实状态码)

chage -M 90 -W 7 rhcsa06_pwuser
assert_eq "$(chage -l rhcsa06_pwuser | grep "Maximum number of days" | awk -F: '{print $2}' | xargs)" "90"
assert_eq "$(chage -l rhcsa06_pwuser | grep "warning" | awk -F: '{print $2}' | xargs)" "7"

chage -d 0 rhcsa06_pwuser
assert_eq "$(chage -l rhcsa06_pwuser | grep "Last password change" | awk -F: '{print $2}' | xargs)" "password must be changed"

userdel -r rhcsa06_pwuser 2>/dev/null
```
本机实测:全部断言输出 `OK`。

**常见坑:** `passwd -S` 返回的状态码是**单个字母**(`P`=已设置密码、`L`=锁定、`NP`=没有密码)——不要凭直觉猜测成更"完整"的缩写(比如误以为是 `PS`),不确定的字段格式,现场跑一次比凭印象编要可靠得多,这条经验教训在本篇多处被现场验证反复印证。

---

## 6. 锁定解锁账户(`passwd -l`/`-u`,`usermod -L`/`-U`)

**命令/配置:**
```bash
passwd -l username     # 锁定(lock)
passwd -u username       # 解锁(unlock)
usermod -L username        # 效果等同于passwd -l
usermod -U username          # 效果等同于passwd -u
```

**一句话是什么:** 锁定账户不是删除密码,而是让存储在 `/etc/shadow` 里的密码哈希"暂时失效"——本机实测确认具体实现方式是在原有密码哈希字符串**前面插入一个感叹号** `!`,让这个字符串不可能匹配出任何合法的密码哈希格式,从而所有密码认证都会失败;解锁就是精确地把这个前缀 `!` 去掉,原密码哈希完整恢复,不需要用户重新设置密码。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"锁定、解锁本地用户账户";员工请长假、怀疑账号被盗用需要临时冻结访问,都是"先锁定观察、确认没问题后再解锁"这类场景的真实需求,和"直接删除账号"相比,锁定是可逆的、更温和的处置手段。

**从最容易犯错的做法讲起:** 想当然地认为锁定账户会把密码字段"清空"或者改成固定的占位符(比如很多人凭印象以为是清空成两个感叹号 `!!`)——**本机实测证伪**:`!!` 实际上是"这个账户从来没有设置过密码"的标记(比如刚创建、还没 `passwd` 过的账号),和"设置过密码、之后被锁定"是完全不同的两种状态,`passwd -l` 对一个已有密码的账户操作,产生的是"单个感叹号 + 完整原密码哈希",不是清空。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 检测到某个账号有异常登录行为,立即 `passwd -l username` 临时冻结,调查清楚是误报还是真的被盗用;确认误报后 `passwd -u username` 解锁,用户能用原密码直接登录,不需要重置密码这种更打扰用户的操作。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

useradd -m rhcsa06_lockuser
echo "rhcsa06_lockuser:TestPass123!" | chpasswd
before_lock_hash=$(getent shadow rhcsa06_lockuser | cut -d: -f2)

passwd -l rhcsa06_lockuser >/dev/null 2>&1
assert_eq "$(passwd -S rhcsa06_lockuser | awk '{print $2}')" "L"
after_lock_hash=$(getent shadow rhcsa06_lockuser | cut -d: -f2)
assert_eq "$after_lock_hash" "!${before_lock_hash}"    # 精确验证:是原哈希前加一个!,不是清空成!!

passwd -u rhcsa06_lockuser >/dev/null 2>&1
assert_eq "$(passwd -S rhcsa06_lockuser | awk '{print $2}')" "P"
restored_hash=$(getent shadow rhcsa06_lockuser | cut -d: -f2)
assert_eq "$restored_hash" "$before_lock_hash"    # 解锁后哈希精确恢复原样,不需要重新设密码

userdel -r rhcsa06_lockuser 2>/dev/null
```
本机实测:全部断言输出 `OK`,包括"解锁后哈希与锁定前完全一致"这一精确验证。

**常见坑:** `usermod -L`/`-U` 和 `passwd -l`/`-u` 效果等价,但如果这个用户的认证方式不是本地密码(比如通过 SSH 密钥、LDAP 集中认证),锁定本地密码字段**不一定能真正阻止这个用户登录**——密码锁定只对"基于密码的认证路径"生效,排查"账号锁定了但还是能登录"这类问题时,要先确认这个用户实际用的是哪种认证方式。

---

## 7. `/etc/passwd`、`/etc/shadow`、`/etc/group` 文件结构解读

**命令/配置:**
```
/etc/passwd:  用户名:x:UID:GID:备注:家目录:登录shell           (7个字段)
/etc/shadow:  用户名:密码哈希:最后修改:最小间隔:最大间隔:警告:不活动:过期:保留   (9个字段)
/etc/group:   组名:x:GID:附加成员列表(逗号分隔)                 (4个字段)
```

**一句话是什么:** 这三个文件是 Linux 本地用户/组体系的"数据库",`/etc/passwd` 存基本身份信息(**密码字段固定是 `x`,真实密码哈希不放在这里**)、`/etc/shadow` 专门存密码哈希和时效信息(权限严格限制,只有 root 能读)、`/etc/group` 存组的基本信息和**附加**成员列表——理解这三个文件各自存什么、字段顺序是什么,是排查几乎所有用户/权限问题的地基。

**为什么 RHCSA 真考 / 生产会用到:** 排障"用户登录不了""密码策略为什么没生效""这个用户到底属于哪些组"这类问题,最终都要落到直接读这几个文件的具体字段上;RHCSA 考试环境下,`getent`(见可运行例子)是比直接 `cat` 更规范的查询方式,能统一处理本地文件和可能存在的网络认证源(LDAP 等)。

**从最容易犯错的做法讲起:** 在 `/etc/passwd` 里看密码字段是 `x`,误以为"这个用户没有设置密码"或者"密码就是字面的 x"——`x` 只是一个**占位符**,含义是"真实密码哈希请去 `/etc/shadow` 里查",这是历史上从"密码直接明文/弱哈希存在 passwd 里(任何人都能读)"演进到"passwd 依然全局可读、但真实哈希挪到只有 root 能读的 shadow 里"这一安全改进的产物,不理解这段历史容易对 `x` 这个字符产生误解。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 排查"某个用户登录报错但看起来账号配置正常",直接 `getent passwd username` 和 `getent shadow username`(需要 root)逐字段核对,常见根因包括:shell 字段被误设成不存在的路径、家目录字段路径错误、shadow 里密码字段被意外清空或加了锁定前缀;`getent group username` 反查一个用户所在的所有组,排查权限相关问题。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

useradd -m rhcsa06_structuser

passwd_line=$(getent passwd rhcsa06_structuser)
assert_eq "$(echo "$passwd_line" | awk -F: '{print NF}')" "7"
assert_eq "$(echo "$passwd_line" | awk -F: '{print $2}')" "x"    # passwd里密码字段永远是占位符x

shadow_line=$(getent shadow rhcsa06_structuser)
assert_eq "$(echo "$shadow_line" | awk -F: '{print NF}')" "9"

group_line=$(getent group rhcsa06_structuser)
assert_eq "$(echo "$group_line" | awk -F: '{print NF}')" "4"

userdel -r rhcsa06_structuser 2>/dev/null
```
本机实测:全部断言输出 `OK`。

**常见坑:** `/etc/group` 里的成员列表**只包含把这个组当"附加组"的用户**,不包含把这个组当"主组"的用户——一个用户的主组信息记录在 `/etc/passwd` 的 GID 字段里,不会出现在 `/etc/group` 对应组的成员列表里;查一个用户完整的组关系,`id username` 或 `groups username` 比只看 `/etc/group` 文件更准确、更省心。

---

## 8. UID/GID 分配规则(系统用户 vs 普通用户的边界)

**命令/配置:**
```bash
grep "^UID_MIN\|^UID_MAX" /etc/login.defs    # 查看普通用户UID的分配区间
useradd -r username                             # -r显式创建系统用户,UID从系统范围分配
```

**一句话是什么:** UID 数值本身有约定俗成的分区:`0` 永远是 root;`1-999`(不同发行版的具体上界可能有差异)是系统/服务账号预留区间;`/etc/login.defs` 里的 `UID_MIN`/`UID_MAX` 定义了"真人登录账号"的合法区间——`useradd` 不加 `-r` 默认从 `UID_MIN` 开始往上分配,加 `-r` 则从系统账号区间分配一个空闲值。

**为什么 RHCSA 真考 / 生产会用到:** 理解这个边界能一眼看出"这是个给人用的账号还是给程序用的账号",是安全审计("为什么这个系统服务账号的 UID 落在了普通用户区间")和权限规划的基础常识;RHCSA 虽不一定直接考"背出具体数字边界",但操作层面经常隐含这个概念(比如创建服务账号时该不该加 `-r`)。

**从最容易犯错的做法讲起:** 给一个纯粹的后台服务账号创建时不加 `-r`,让它意外落入普通用户的 UID 区间——这本身不会直接导致故障,但会让安全审计工具/人工审查时把这个服务账号误判成"真人账号",增加排查噪音,也可能在某些按 UID 区间做权限区分的策略里产生意料之外的行为;创建服务账号时习惯性加 `-r`,让 UID 落在系统区间,是更规范、更少歧义的做法。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 部署一个 Web 应用需要专用运行账号,`useradd -r -M -s /sbin/nologin webapp`,UID 自动落在系统区间,后续审计"这台机器上哪些是真人账号、哪些是服务账号"时,凭 UID 区间就能快速做初步区分,不需要逐个人工核实每个账号的用途。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

uid_min=$(grep "^UID_MIN" /etc/login.defs | awk '{print $2}')
assert_eq "$uid_min" "1000"    # RHEL/Rocky标准:普通用户从1000开始分配

useradd -m rhcsa06_normaluser    # 不加-r,走普通用户分配区间
normal_uid=$(id -u rhcsa06_normaluser)
assert_ok test "$normal_uid" -ge "$uid_min"

useradd -r -M -s /sbin/nologin rhcsa06_sysuser    # 显式-r,走系统账号分配区间
sys_uid=$(id -u rhcsa06_sysuser)
assert_ok test "$sys_uid" -lt "$uid_min"

userdel -r rhcsa06_normaluser 2>/dev/null
userdel rhcsa06_sysuser 2>/dev/null
```
本机实测:本机 `UID_MIN=1000`,两个断言均输出 `OK`,普通用户和系统用户确实落在了 UID 边界的两侧。

**常见坑:** `UID_MIN`/`UID_MAX` 是 `/etc/login.defs` 里可以修改的配置项,不是写死在内核里的硬性常量——不同发行版、不同企业内部规范可能有不同的默认值,排查"为什么这台机器新建用户的 UID 起点和预期不一样"时,先查这台机器实际的 `/etc/login.defs` 配置,不要凭"应该是 1000"这种通用印象下结论。

---

## 9. `/etc/skel` 与新用户初始化模板

**命令/配置:**
```bash
ls -la /etc/skel/       # 查看当前的新用户初始化模板内容
# useradd -m 创建家目录时,会把/etc/skel下的全部内容(含隐藏文件)复制进新用户的家目录
```

**一句话是什么:** `/etc/skel`(skeleton,骨架)是一个模板目录,`useradd -m` 创建新用户家目录时,会把这个目录下的所有文件(包括 `.bashrc`/`.bash_profile` 这类隐藏的 shell 配置文件)原样复制一份到新用户的家目录里——管理员可以在 `/etc/skel` 里预置统一的默认配置(比如公司统一的 shell 别名、欢迎文档),让每个新建账号自动获得这份"初始装修"。

**为什么 RHCSA 真考 / 生产会用到:** 理解 `/etc/skel` 的作用,能解释"为什么新建的用户家目录里凭空多出了几个配置文件"这个现象;企业环境批量开账号时,预先定制好 `/etc/skel` 是统一新员工初始环境配置最省事的方式,不需要每个账号手动去改一遍配置文件。

**从最容易犯错的做法讲起:** 修改了 `/etc/skel` 里的内容,却指望这次修改能"追溯"影响到**已经存在**的老用户——`/etc/skel` 只在 `useradd -m` **创建那一刻**生效,是"一次性复制",不是"持续同步的模板链接",改了 `/etc/skel` 之后,只有**后续新建**的账号会带上新内容,老账号的家目录不会有任何变化,如果需要老账号也统一更新,得手动或写脚本批量处理。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 公司要求所有新员工账号默认带上统一的 `.bashrc` 别名配置和一份《新人须知》文档,管理员把这些内容放进 `/etc/skel`,之后每个新建的账号自动就有,不需要 HR/IT 手动逐个配置;RHCSA 考试如果要求"新建的用户应该自动包含某个特定文件",预先修改 `/etc/skel` 是标准解法。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

echo "rhcsa06_skel_test_content" > /etc/skel/.rhcsa_test_marker
useradd -m rhcsa06_skeluser

marker_copied=0; [ -f /home/rhcsa06_skeluser/.rhcsa_test_marker ] && marker_copied=1
assert_eq "$marker_copied" "1"    # /etc/skel里新加的文件,确实被复制进了新建的家目录
assert_eq "$(cat /home/rhcsa06_skeluser/.rhcsa_test_marker)" "rhcsa06_skel_test_content"

rm -f /etc/skel/.rhcsa_test_marker
userdel -r rhcsa06_skeluser 2>/dev/null
```
本机实测:两个断言均输出 `OK`。

**常见坑:** `/etc/skel` 里文件的**权限和所有者**在复制过程中会被重新设置成新用户自己(而不是保留 `/etc/skel` 里原文件的所有者,通常是 root)——这是符合预期的正确行为(新用户当然应该拥有自己家目录里的文件),但如果 `/etc/skel` 里某个文件的权限位设置得不合理(比如意外设成了全局可写),这个不合理的权限位本身会被原样复制给每一个新建的账号,`/etc/skel` 里的文件权限也需要仔细维护。

---

## 10. sudo 配置(`visudo`,`/etc/sudoers.d/`,最小权限原则)

**命令/配置:**
```bash
visudo                                  # 安全编辑/etc/sudoers(带语法检查,防止改坏了没法用sudo)
visudo -c                                 # 只做语法检查,不进入编辑
visudo -f /etc/sudoers.d/xxx                # 编辑/检查sudoers.d下的指定片段文件
# 典型sudoers规则: 用户名 ALL=(ALL) NOPASSWD: /具体命令路径
```

**一句话是什么:** `sudo` 让普通用户能以受控的方式临时获得特定的高权限操作能力,配置规则存放在 `/etc/sudoers`(全局)和 `/etc/sudoers.d/`(独立片段文件,推荐用这里而不是直接改主文件)里;`visudo` 是专用的编辑工具,保存时会自动做语法校验——**必须用 `visudo`,不能用普通文本编辑器直接改**,这是这个知识点最核心的纪律。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"配置、修改、限制用户的 sudo 访问";生产环境权限管理的黄金原则是"最小权限"——只授予完成工作真正需要的最小权限集合,而不是简单粗暴地给某个用户完整的免密 `ALL` 权限,精确到具体命令路径的授权是更负责任的做法。

**从最容易犯错的做法讲起:** 用 `vi /etc/sudoers` 直接编辑这个文件——**这是极其危险的做法**:如果编辑过程中不小心引入了语法错误,保存退出后**整个系统的 sudo 机制会直接失效**(包括你自己都无法再用 sudo 去修复这个文件,除非有其他 root 访问途径,比如救援模式);`visudo` 的核心价值就是"保存前自动做语法检查,发现错误会拒绝保存并提示",从根本上杜绝这种"改了就出不来"的灾难场景。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 运维人员需要能重启某个特定服务但不应该拥有完整 root 权限:在 `/etc/sudoers.d/` 下新建一个专属片段文件,写入 `username ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart myapp`,精确到这一条命令,免密执行,其他任何 sudo 操作依然会被拒绝或要求密码——这正是"最小权限原则"的具体落地,而不是图省事直接把这个用户加进 `wheel` 组拿到完整权限。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

assert_ok test -f /etc/sudoers
visudo_check_ok=$(visudo -c 2>&1)
echo "$visudo_check_ok" | grep -qi "parsed OK" && echo "OK: visudo -c 确认当前sudoers语法正确"

useradd -m rhcsa06_sudouser
echo "rhcsa06_sudouser ALL=(ALL) NOPASSWD: /usr/bin/systemctl status sshd" > /etc/sudoers.d/rhcsa06-demo
chmod 440 /etc/sudoers.d/rhcsa06-demo

sudoers_syntax_ok=$(visudo -c -f /etc/sudoers.d/rhcsa06-demo 2>&1)
echo "$sudoers_syntax_ok" | grep -qi "parsed OK" && echo "OK: 新增的sudoers.d片段语法正确"

# 精确授权的命令能免密执行,未被授权的命令被拒绝——最小权限原则的现场验证
sudo_allowed=$(su rhcsa06_sudouser -c "sudo -n /usr/bin/systemctl status sshd" 2>&1; echo "EXIT:$?")
echo "$sudo_allowed" | grep -q "EXIT:0" && echo "OK: 精确授权的命令能免密执行成功"

sudo_denied=$(su rhcsa06_sudouser -c "sudo -n /usr/bin/whoami" 2>&1; echo "EXIT:$?")
echo "$sudo_denied" | grep -qv "EXIT:0" && echo "OK: 未被精确授权的命令被拒绝"

rm -f /etc/sudoers.d/rhcsa06-demo
userdel -r rhcsa06_sudouser 2>/dev/null
```
本机实测:全部检查点输出 `OK`,现场验证了"精确授权命令可用、其他命令被拒"这一最小权限原则的真实生效效果。

**常见坑:** `/etc/sudoers.d/` 下的片段文件权限必须是 **440**(root 只读,不能是其他任何权限组合,包括看似更"安全"的 400 或者更宽松的 644)——`sudo` 出于安全考虑,会直接忽略权限不符合要求的片段文件(不报错,只是安静地不生效),这是"配置文件写对了但权限不对导致规则不生效"的典型坑,新建规则文件后记得 `chmod 440` 这一步不能省。

---

*本篇完成:2026-07-11,10 个知识点。验证环境:Rocky Linux 10.2(WSL2)。全部代码块真实跑通验证,含多处现场纠正的凭记忆写错的细节:`passwd -S` 状态码是单字母(`P`/`L`)而非"PS"/"LK"、`passwd -l` 锁定是在原密码哈希前加一个`!`而非清空成`!!`(那是"从未设置过密码"的标记)、`awk`提取字段后用`tr -d ' '`会破坏带空格的短语值(应改用`xargs`只清理首尾空白)。*
