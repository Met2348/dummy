# 08 · 安全:SELinux 与防火墙(Security: SELinux & Firewall)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一篇覆盖 10 个知识点:SELinux 三种模式、上下文概念、`ls -Z`/`ps -Z`、`restorecon`/`chcon`、`semanage`、SELinux 布尔值、故障排查工具、firewalld 基本概念、`firewall-cmd` 常用操作、富规则。**本文所有代码例子已在 Rocky Linux 10.2(WSL2)下实际跑通验证**。
>
> **本篇必须优先阅读的重大诚实声明**:防火墙部分(第 8-10 节)在本环境下功能完整、验证扎实,和前面几篇质量一致。**SELinux 部分(第 1-7 节)则受到三层递进的真实 WSL2 内核限制**,详见下方专门的说明段落——这不是偷懒,而是撰写过程中层层排查后得到的确凿结论,每一层限制都现场取证、如实记录,没有一处凭空断言。
>
> **和前四条系列的差异声明**(详见 [00-roadmap.md](00-roadmap.md) 开头):本仓库没有 Linux 系统管理场景可挖,"真实场景例子"部分统一标注为典型运维/RHCSA 考试场景,不冒充仓库代码里挖出来的。

---

## SELinux 在本环境下的三层限制(撰写前必读)

按发现顺序如实记录,每一层都是现场排查、有真实证据支撑的结论,不是主观猜测:

1. **配置文件说的和实际生效的不是一回事**:`/etc/selinux/config` 里写的是 `SELINUX=enforcing`,但 `getenforce` 返回 `Disabled`。根因是现代 Fedora/RHEL 体系里,"完全禁用 SELinux"已经不再由这个配置文件决定,而是由**内核启动参数** `selinux=0` 决定(`/etc/selinux/config` 文件自己的注释里就写明了这一点)。本机确认 `/proc/cmdline` 里的参数是纯 WSL2 自己的启动参数(`WSL_ROOT_INIT=1 panic=-1 ...`),完全不经过 GRUB/`grubby` 管理的 BLS 配置——这与 02 类第 2 节"WSL 没有真实 GRUB"是同一个根因的不同表现。
2. **尝试移除内核参数、重启、确认无效**:用 `grubby --update-kernel ALL --remove-args selinux` 移除了这个参数、`wsl --terminate` 重启这个发行版,`getenforce` 依然是 `Disabled`——因为 WSL2 根本不读取 BLS 配置里的内核参数,这条路径在 WSL 环境下彻底不通。
3. **深入排查发现内核层面比预想的更复杂**:`dmesg` 显示 `SELinux: Initializing.`,`/sys/fs/selinux` 目录确实存在(和最初以为"内核完全不支持"不同),但这个虚拟文件系统的实现是**残缺的**——连最基础的 `/sys/fs/selinux/policyvers` 文件都不存在,`load_policy` 命令尝试加载真实策略文件(`policy.35`,文件本身完好,3.3MB,合法的策略文件)时报错 `No such file or directory`。这是比 VDO(04 类第 13 节,`dm-vdo` 内核模块完全不存在)更微妙的一种限制:**内核声称初始化了 SELinux LSM,但配套的用户空间接口不完整,导致策略无法真正被加载激活**。

**这如何影响本篇内容**:第 1-4、6-7 节涉及"SELinux 真正 enforcing 时的强制访问控制效果"的部分,一律如实标注"未能验证,需要真实 RHEL 环境"。但**第 5 节(`semanage`)是重要的例外**——本机验证发现,`semanage` 管理的是策略源文件/数据库本身(编译进策略、写入自定义规则),这个操作层面**不依赖内核是否真正 enforcing**,因此可以扎实验证;这也顺带解释了为什么 `getsebool`/`setsebool`(操作"运行时活跃状态")在本环境失败,而 `semanage boolean -l`(查询"策略数据库定义")却能正常工作——两者操作的不是同一层东西。

---

## 1. SELinux 三种模式(enforcing/permissive/disabled)

**命令/配置:**
```bash
getenforce                       # 查看当前运行时模式
setenforce 0                       # 运行时临时切换到permissive(0)
setenforce 1                         # 运行时临时切换到enforcing(1),没有disabled这个选项(见下)
/etc/selinux/config                    # 持久化配置(重启后生效),SELINUX=enforcing|permissive|disabled
```

**一句话是什么:** `enforcing` 是完全强制模式(违反策略的操作直接被拒绝);`permissive` 是"只记录不拦截"模式(违反策略的操作依然被允许执行,但会记录一条日志,常用于调试新策略、排查"如果启用 enforcing 会拦截哪些东西"而不影响业务);`disabled` 是完全不加载策略,SELinux 形同虚设。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"配置 SELinux 的强制/许可/禁用模式";生产环境部署新服务时,先用 `permissive` 观察一段时间收集 AVC 日志(见第 7 节),确认没有遗漏的合法访问被记录后再切到 `enforcing`,是比"直接 enforcing 然后手忙脚乱排查"更稳妥的上线流程。

**从最容易犯错的做法讲起:** 用 `setenforce 0`/`setenforce 1` 在 `permissive` 和 `enforcing` 之间切换后,以为这个改动"应该"是持久的——**`setenforce` 只影响当前运行时**,重启后系统会按 `/etc/selinux/config` 里的持久化配置重新决定模式;而且 `setenforce` 根本没有"切换到 disabled"这个选项(`disabled` 只能通过修改配置文件 + 重启才能进入或退出,这本身就是为了防止"一条命令不小心就把 SELinux 整个关掉"这种误操作)。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 部署一个新的自定义应用,不确定 SELinux 策略是否会拦截它的正常操作,先 `setenforce 0` 切到 permissive 跑一段时间观察 `ausearch -m avc` 有没有记录、确认都是预期内的访问模式后,写好对应的策略调整(第 5-6 节),再 `setenforce 1` 切回 enforcing 验证确实不再产生新的拒绝记录。

**可运行例子(如实验证本环境的真实状态和根因,不假装模式切换能生效):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

current_mode=$(getenforce)
assert_eq "$current_mode" "Disabled"

config_file_setting=$(grep "^SELINUX=" /etc/selinux/config | cut -d= -f2)
assert_eq "$config_file_setting" "enforcing"    # 配置文件写的和实际生效的不是一回事(见上方"三层限制"说明)

echo "本机确认:配置文件写的是enforcing,getenforce却返回Disabled——真正禁用靠内核启动参数selinux=0,不是这个文件"
```
本机实测:两个断言均输出 `OK`,现场确认了配置文件内容和实际生效状态之间的真实落差。

**常见坑:** 见上方"从最容易犯错的做法"——`disabled` 无法通过 `setenforce` 运行时进入或退出,这是刻意的安全设计;另外,从 `disabled` 切换到 `enforcing`/`permissive`(哪怕是通过配置文件+重启)通常需要一次完整的文件系统 relabel(给所有文件重新打上正确的安全上下文标签,这个过程可能耗时较长),直接冒然切换可能导致系统里大量文件因为没有正确标签而在 enforcing 模式下访问异常,RHEL 通常会在这种切换时自动触发 `/.autorelabel` 机制处理这个问题。

---

## 2. SELinux 上下文基本概念(`user:role:type:level`)

**命令/配置:**
```
system_u:object_r:httpd_sys_content_t:s0
   │        │            │              │
 SELinux   角色         类型          敏感度级别
  用户    (role)       (type)         (level/MLS)
```

**一句话是什么:** SELinux 上下文是一个四段式标签,贴在**每一个**进程和文件上——访问控制的核心判断依据几乎全部落在第三段 **type**(类型)上("目标进程能不能访问目标文件",本质上是"进程的 type 能不能对文件的 type 做这个操作"这条策略规则查表),这也是为什么排障和配置时几乎所有精力都花在"这个文件/进程该打什么 type"上,而不太需要关心 user/role 这两段。

**为什么 RHCSA 真考 / 生产会用到:** 理解"访问控制看的是 type 不是文件所有者"这个核心机制,是排查"标准 ugo 权限明明对、SELinux 却拦截"这类问题的基础(这类问题极其常见,新手第一反应往往是反复检查 chmod/chown,却没意识到还有 SELinux 这一层完全独立的访问控制)。

**从最容易犯错的做法讲起:** 把 SELinux 上下文和标准 Unix 权限混为一谈,以为"这个文件的 owner 是 apache 用户,所以 httpd 进程理所当然能访问"——SELinux 是完全独立于标准 ugo 权限模型的**第二层**访问控制,两层都要通过才能真正访问成功,标准权限对了不代表 SELinux 那层也一定放行,反之亦然(SELinux 那层放行了,标准 ugo 权限拒绝了,同样访问不了)。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 排查"Web 服务器访问不了某个目录",标准权限检查(owner/group/mode)全部正常,但访问依然被拒绝——这是 SELinux 上下文类型不对的经典征兆(比如把网站内容放在了一个不是 `httpd_sys_content_t` 类型的自定义目录里),需要用第 4-5 节的 `chcon`/`semanage` 把这个目录标记成正确的类型才能解决。

**可运行例子(策略数据库里的真实规则定义,不依赖enforcing真正生效):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

sample_context=$(semanage fcontext -l 2>&1 | grep "^/  " | awk '{print $NF}')
assert_ok test -n "$sample_context"
echo "根目录/在策略里定义的标准上下文: $sample_context"

field_count=$(echo "$sample_context" | awk -F: '{print NF}')
assert_eq "$field_count" "4"    # user:role:type:level 标准4段式,即便SELinux未激活,策略数据库里的定义依然是真实、完整的
```
本机实测:两个检查点均输出 `OK`。

**常见坑:** 类型命名有一套非正式但被广泛遵循的约定——以 `_t` 结尾(比如 `httpd_sys_content_t`),看到这个后缀基本能确认这是一个 SELinux type;初学者常把整串上下文当成一个不可分割的黑盒去记,实际上四段各自独立、含义不同,拆开理解比整串死记容易得多。

---

## 3. `ls -Z`/`ps -Z` 查看 SELinux 上下文

**命令/配置:**
```bash
ls -Z file_or_dir      # 在标准ls输出基础上附加显示SELinux上下文
ps -Z                    # 在标准ps输出基础上附加显示进程的SELinux上下文(LABEL列)
id -Z                      # 查看当前登录用户自己的SELinux上下文
```

**一句话是什么:** 这几个熟悉命令的 `-Z` 选项统一约定用来"顺带显示 SELinux 上下文",不需要额外学新命令——这是 SELinux 生态和传统 Unix 工具集成的设计,想查看安全上下文,先想想能不能在你已经熟悉的命令后面加个 `-Z`。

**为什么 RHCSA 真考 / 生产会用到:** 排障第一步永远是"先看看现状",`ls -Z`/`ps -Z` 是查看文件/进程当前 SELinux 标签状态最快的方式,是诊断链路的起点。

**从最容易犯错的做法讲起:** 在 SELinux 处于 `disabled` 状态的系统上执行 `ls -Z`,看到输出全是问号 `?`,以为这是命令本身出错或者语法用错了——**这其实是符合预期的正常行为**:`disabled` 状态下压根没有任何上下文标签数据可显示,`?` 就是"没有数据"的诚实表示,不是报错;类似地 `ps -Z` 在 disabled 状态下会显示一个统一的 `kernel` 占位符,而不是真实的进程上下文。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 在一台真实 enforcing 状态的 RHEL 服务器上排查权限问题,`ls -Z /var/www/html/` 快速确认网站目录下每个文件的类型是不是都是 `httpd_sys_content_t`,如果发现某个文件显示成了别的类型(比如误上传时保留了错误的类型),这就是访问被拒绝的直接线索。

**可运行例子(如实展示 disabled 状态下的真实输出,不假装看到了真实标签):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

touch /tmp/rhcsa08_lstest.txt
ls_z_output=$(ls -Z /tmp/rhcsa08_lstest.txt)
echo "$ls_z_output" | grep -q "?" && echo "OK: disabled状态下ls -Z显示?占位符,不是报错,是诚实的'无数据'表示"

ps_z_output=$(ps -Z --pid $$ 2>&1)
echo "$ps_z_output" | grep -qi "kernel" && echo "OK: disabled状态下ps -Z的LABEL列显示kernel占位符"

rm -f /tmp/rhcsa08_lstest.txt
```
本机实测:两个检查点均输出 `OK`,现场确认了 disabled 状态下这两个命令的真实(占位符式)输出格式。

**常见坑:** `ls -Z` 默认不递归——只看到当前这一级列出的文件/目录本身的上下文,子目录内部文件的上下文看不到,需要配合 `-R` 递归查看,或者用 `find dir -exec ls -Zd {} \;` 这类组合才能批量检查整个目录树,一次性的 `ls -Z` 容易让人误以为"看过了、都没问题",其实只看了最外层一层。

---

## 4. `restorecon`/`chcon` 恢复修改上下文

**命令/配置:**
```bash
restorecon -v file           # 把文件的上下文恢复成策略里定义的"应该是什么"(-v显示改动了什么)
restorecon -Rv dir             # 递归恢复整个目录树
chcon -t type_t file             # 临时修改文件的type(不查策略、直接指定,重启/relabel后会被覆盖回策略定义值)
```

**一句话是什么:** `restorecon` 是"把标签改回策略说它应该是什么样"(查策略数据库、按规则重新打标签,这是**推荐**的标准做法);`chcon` 是"直接把标签改成我说的这个值"(不管策略怎么定义,强行指定,效果是临时的、下次 relabel 会被覆盖)——日常运维绝大多数场景应该用 `restorecon`,`chcon` 更多用在临时调试或者策略本身还没来得及更新的过渡场景。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"列出、识别、恢复文件 SELinux 上下文";最常见的真实场景是"文件被移动/复制到了新位置,上下文没有跟着自动更新",`restorecon` 是修复这类问题的标准命令。

**从最容易犯错的做法讲起:** 文件从一个位置 `mv` 到另一个位置后,上下文标签**可能不会自动更新**(取决于是同一文件系统内移动还是跨文件系统/用 `cp` 复制,不同操作对上下文的处理规则不同)——很多人排查"文件放对地方了但服务还是访问不了"时忽略了这一点,标准应对是操作完文件之后养成习惯性执行一次 `restorecon -Rv` 的习惯,而不是假设"文件在正确位置=标签也对了"。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 从备份恢复了一批网站文件到 `/var/www/html/`,文件权限(ugo)都对,但网站访问不了——`restorecon -Rv /var/www/html/` 把这批文件的 SELinux 上下文重新按策略规则打一遍标(备份恢复的文件常常带着原来的、可能不正确的上下文),这是这类问题的标准解法,几乎不需要先诊断就可以放心执行(它只会把标签改成策略认为"正确"的值,不会有副作用)。

**可运行例子(如实验证disabled状态下这两个命令的真实局限):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

touch /tmp/rhcsa08_chcontest.txt

chcon_output=$(chcon -t tmp_t /tmp/rhcsa08_chcontest.txt 2>&1)
assert_eq "$?" "1"
echo "$chcon_output" | grep -qi "unlabeled" && echo "OK: chcon在完全无初始标签的文件系统上拒绝执行,报错明确提示unlabeled"

restorecon_output=$(restorecon -v /tmp/rhcsa08_chcontest.txt 2>&1)
assert_eq "$?" "0"
no_change=0; [ -z "$restorecon_output" ] && no_change=1
assert_eq "$no_change" "1"
echo "OK: restorecon命令本身执行成功(退出码0),但disabled状态下没有任何标签体系可relabel,-v也没有任何变更输出"

rm -f /tmp/rhcsa08_chcontest.txt
```
本机实测:全部断言输出 `OK`。这条"chcon 在完全无标签的文件系统上拒绝执行"本身就是一条真实、有价值的发现——即便 SELinux 没有真正 enforcing,`chcon` 依然会检查文件是否有一个基础的、可以被"修改"的现有标签,disabled 状态下从未打过标的文件连这个基础条件都不满足。

**常见坑:** `chcon` 修改的效果是**临时的**——下一次系统触发文件系统 relabel(不管是手动 `restorecon` 还是 `/.autorelabel` 触发的整体重新打标)时,`chcon` 设置的自定义值会被策略里定义的"正确"值覆盖回去;如果需要一个"持久化、不会被 relabel 覆盖"的自定义上下文规则,应该用第 5 节的 `semanage fcontext -a` 把这条规则写进策略本身,而不是靠 `chcon` 打临时补丁。

---

## 5. `semanage` 管理 SELinux 策略持久化

**命令/配置:**
```bash
semanage fcontext -a -t TYPE "PATH_PATTERN"    # 添加一条持久化的文件上下文规则
semanage fcontext -l                              # 列出所有fcontext规则(含系统内置+自定义)
semanage fcontext -d "PATH_PATTERN"                 # 删除一条自定义规则
semanage boolean -l                                   # 列出所有布尔值及其当前/默认状态
semanage port -l                                        # 列出端口和SELinux类型的映射关系
```

**一句话是什么:** `semanage` 是管理 SELinux **策略配置本身**(而不是某一个文件的临时标签)的工具——`fcontext -a` 添加的规则会被写入持久化存储,之后不管跑多少次 `restorecon`、不管系统怎么 relabel,这条自定义规则都会被遵守,这是让"自定义目录应该用什么类型"这类决策长期生效的正确方式。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求掌握 `semanage`;把网站内容放在非标准路径(比如不是 `/var/www/html` 而是自定义的 `/srv/mywebapp`)是很常见的部署需求,必须用 `semanage fcontext -a` 告诉 SELinux"这个新路径也应该被当作 web 内容对待",否则每次 relabel 后这个目录都会变回默认类型、重新导致访问失败。

**从最容易犯错的做法讲起:** 只用 `chcon` 给自定义目录打了正确的类型,以为"已经解决了、可以收工"——`chcon` 的效果不持久,下次系统重新 relabel(比如从其它渠道触发的整体 relabel,或者只是巧合地对这个目录跑了一次 `restorecon`)会让类型被打回默认值,过一段时间"配置又失效了"却想不起来当初是怎么解决的;正确姿势是从一开始就用 `semanage fcontext -a` 写好持久化规则,`chcon`/`restorecon` 只是让这条规则立刻生效的手段,规则本身要靠 `semanage` 才能长期存在。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 部署应用到自定义路径 `/srv/mywebapp`,先 `semanage fcontext -a -t httpd_sys_content_t "/srv/mywebapp(/.*)?"`(正则表达式匹配这个目录及其下所有内容),再 `restorecon -Rv /srv/mywebapp` 让新规则立刻应用到已存在的文件上——这两步组合起来,才是"让自定义路径永久性地被当作合法 web 内容"的完整标准流程。

**可运行例子(不依赖enforcing真正生效,这是本篇SELinux部分里验证最扎实的一节):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

mkdir -p /tmp/rhcsa08_webroot
semanage fcontext -a -t httpd_sys_content_t "/tmp/rhcsa08_webroot(/.*)?" >/dev/null 2>&1
assert_eq "$?" "0"

added_rule=$(semanage fcontext -l 2>&1 | grep "rhcsa08_webroot")
assert_ok test -n "$added_rule"
echo "$added_rule" | grep -q "httpd_sys_content_t" && echo "OK: 自定义规则真实写入了策略数据库,能被完整查询到"

semanage fcontext -d "/tmp/rhcsa08_webroot(/.*)?" >/dev/null 2>&1
assert_eq "$?" "0"
assert_eq "$(semanage fcontext -l 2>&1 | grep -c 'rhcsa08_webroot')" "0"    # 删除确实生效

rmdir /tmp/rhcsa08_webroot
```
本机实测:全部断言输出 `OK`——这是本篇里少数几个能不打折扣、完整验证真实效果的 SELinux 知识点,因为 `semanage` 操作的是策略数据库这一层,不依赖内核是否真正处于 enforcing 状态。

**常见坑:** `semanage fcontext -a` 添加的规则只是"注册"了一条策略,**不会自动应用**到已经存在的文件上——新建的文件会自动按新规则打标,但已有文件必须额外手动 `restorecon` 一次才能生效,这是新手常漏掉的第二步,加了规则却发现"怎么还是不对"往往就是忘了这一步。

---

## 6. SELinux 布尔值(`getsebool`/`setsebool`)

**命令/配置:**
```bash
getsebool BOOLEAN_NAME              # 查询某个布尔值当前是否开启(需要SELinux真正运行)
getsebool -a                          # 列出全部布尔值当前状态(同样需要SELinux真正运行)
setsebool BOOLEAN_NAME on|off           # 运行时临时修改(重启丢失)
setsebool -P BOOLEAN_NAME on|off          # -P持久化,写入策略,重启后依然生效
```

**一句话是什么:** SELinux 布尔值是策略里预先定义好的"开关"——不需要写完整的自定义策略规则,通过打开/关闭一个预置的布尔值就能调整一整类行为(比如 `httpd_can_network_connect` 控制"Web 服务是否允许主动发起网络连接",这类需求足够常见,策略作者预先做成了开关,不用每次都手写规则)。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"使用布尔值改变 SELinux 策略设置";布尔值是日常调整 SELinux 行为最常用、最不容易出错的手段(相比手写自定义策略模块),遇到"这个常见需求 SELinux 拦住了"的场景,第一反应应该是先查有没有现成的布尔值能解决,而不是直接跳去写自定义规则。

**从最容易犯错的做法讲起:** 用 `setsebool` 修改了布尔值,却忘了加 `-P`——不加 `-P` 的修改和 `setenforce` 一样是**临时**的,重启后会恢复成策略里定义的默认值;这个坑和"手动 mount vs 写 fstab"是同一类"临时 vs 永久"设计模式,在 RHCSA 系统管理的方方面面反复出现,理解一次原理能推广到所有类似场景。

**真实场景例子(典型运维场景/RHCSA 考试场景):** Web 服务需要连接外部数据库(网络连接),但默认策略不允许 httpd 主动发起网络连接,`setsebool -P httpd_can_network_connect on` 一条命令解决,不需要为这个常见需求手写策略模块;排查"为什么加了 -P 还是没生效",往往是漏看了布尔值名字打错了(布尔值名字很多,容易记混或拼错)。

**可运行例子(展示两种查询方式的本质区别——离线策略数据库 vs 运行时活跃状态):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

# semanage boolean -l:查询"策略数据库里定义了什么",不需要内核真正enforcing
bool_list=$(semanage boolean -l 2>&1)
echo "$bool_list" | grep -q "httpd_can_network_connect" && echo "OK: semanage boolean -l 能查到策略里定义的布尔值(离线数据库层面)"

# getsebool/setsebool:查询/修改"运行时活跃状态",本环境SELinux未真正运行,如实验证这个局限
getsebool_output=$(getsebool httpd_can_network_connect 2>&1)
assert_eq "$?" "1"
echo "$getsebool_output" | grep -qi "disabled" && echo "OK: getsebool明确报错SELinux is disabled,不是含糊的失败"

setsebool_output=$(setsebool httpd_can_network_connect on 2>&1)
assert_ok test "$?" -ne 0
echo "$setsebool_output" | grep -qi "Invalid boolean\|disabled" && echo "OK: setsebool同样无法在disabled状态下修改运行时布尔值"
```
本机实测:全部检查点输出 `OK`,清晰验证了"离线策略数据库查询"和"运行时活跃状态查询/修改"这两类操作在本环境下的不同表现,不是笼统地说"SELinux 不能用"。

**常见坑:** 布尔值的名字**没有统一的命名规范可以完全猜对**,虽然大部分是"服务名\_动作"这种模式(比如 `httpd_can_network_connect`),但具体某个功能是否有对应布尔值、叫什么名字,还是要靠 `getsebool -a`(真实 enforcing 环境下)或 `semanage boolean -l`(离线也能查)配合 `grep` 关键词去搜索确认,不要凭猜测直接敲一个"看起来应该是这个名字"的布尔值名。

---

## 7. SELinux 故障排查(`ausearch`/`sealert`)

**命令/配置:**
```bash
ausearch -m avc              # 查询audit日志里的AVC(Access Vector Cache)拒绝记录,即SELinux拦截事件
ausearch -m avc -ts recent      # 只看最近的拒绝记录
sealert -a /var/log/audit/audit.log    # 把AVC拒绝记录翻译成人类可读的分析报告+具体修复建议
```

**一句话是什么:** SELinux 每次拦截一个操作,都会在 audit 日志里留下一条 AVC 记录(拒绝了谁、访问什么、需要什么类型权限);`ausearch` 是从海量审计日志里精确筛选出这类记录的工具,`sealert` 更进一步,把这些晦涩的技术记录翻译成人类可读的说明,甚至直接给出"运行这条命令就能修复"的具体建议——这是 SELinux 排障从"完全摸不着头脑"变得可操作的关键工具链。

**为什么 RHCSA 真考 / 生产会用到:** "诊断和解决 SELinux 相关问题"隐含贯穿在 RHCSA 涉及 SELinux 的全部题目里;真实生产环境排查"服务突然访问不了、其他方面都正常"这类诡异问题,AVC 拒绝记录往往是最终揭示真相的关键线索,不看 audit 日志基本没法确诊是不是 SELinux 导致的。

**从最容易犯错的做法讲起:** 遇到疑似 SELinux 导致的访问问题,直接把 SELinux 切到 `permissive` 甚至 `disabled` 让报错"消失",而不是用 `ausearch`/`sealert` 真正诊断问题根源——这是关闭安全防护而不是解决问题,治标不治本,而且一旦养成"遇到 SELinux 报错就先关掉"的习惯,会逐渐丧失真正排查和修复权限问题的能力;正确流程永远是先诊断(本节)、理解根因、用第 4-6 节的工具精确修复,SELinux 本身极少是应该被绕过而不是被正确配置的对象。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 服务部署后访问被拒绝,`ausearch -m avc -ts recent` 找到对应的拒绝记录,看到 "denied" 后面跟着的具体权限类型和涉及的 type 名字,对照第 5-6 节判断应该用 `semanage fcontext`(路径类型不对)还是 `setsebool`(缺少某个开关)来解决;`sealert -a /var/log/audit/audit.log` 更进一步,直接给出格式化的分析和"you can run 类似 restorecon 命令"这种可执行的修复建议。

**可运行例子(如实探测auditd服务在本环境的真实状态,不假装能产生真实拒绝事件供排查):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

assert_ok command -v ausearch
assert_ok command -v sealert

ausearch_help=$(ausearch --help 2>&1)
echo "$ausearch_help" | grep -qi "message type\|--message" && echo "OK: ausearch --help 显示了按事件类型过滤AVC拒绝记录的用法"

sealert_help=$(sealert --help 2>&1)
echo "$sealert_help" | grep -qi "alert\|analyze" && echo "OK: sealert --help 显示了分析AVC拒绝告警的用法"

# 如实探测:auditd这个后台服务本身在本环境下的真实状态
auditd_status=$(systemctl is-active auditd 2>&1)
echo "本机auditd服务状态: $auditd_status"
auditd_fail_reason=$(journalctl -u auditd --no-pager 2>&1 | grep -c "Operation not permitted")
assert_ok test "$auditd_fail_reason" -ge 1
echo "OK: 现场确认auditd启动失败的根因是audit netlink操作被拒绝(Operation not permitted),是WSL2内核对audit子系统的限制,不是配置问题"
```
本机实测:全部检查点输出 `OK`。**必须诚实说明**:`auditd` 服务本身在本环境下无法启动(`journalctl` 现场确认根因是 `Error sending status request (Operation not permitted)`,WSL2 内核限制了 audit netlink 相关操作),这意味着本节只能验证 `ausearch`/`sealert` 这两个工具本身存在、命令语法正确,**无法产生真实的 AVC 拒绝记录供排查演示**——这是继 VDO(`dm-vdo` 模块缺失)、SELinux 策略加载(`selinuxfs` 残缺)之后,本系列撰写过程中发现的第三个 WSL2 内核结构性限制,真实的 `ausearch`/`sealert` 排障流程演示需要在真实 RHEL 环境下进行。

**常见坑:** `sealert` 给出的"建议修复命令"是基于对这条具体拒绝记录的分析生成的,**不能不假思索直接照抄执行**——它是很好的排查起点和参考,但每个环境的实际情况可能有细微差异,执行前应该理解这条建议命令具体在做什么(比如是不是在提议一个和第 6 节常见坑提到的、有安全隐患的过度放开权限的操作),而不是把它当成"点一下就能自动修好"的黑盒按钮。

---

## 8. firewalld 基本概念(zone/service/port)

**命令/配置:**
```bash
firewall-cmd --get-default-zone      # 查看默认zone
firewall-cmd --get-active-zones        # 查看当前哪些zone正在被使用(哪个网卡属于哪个zone)
firewall-cmd --get-zones                 # 列出所有可用的zone
firewall-cmd --list-all                    # 查看默认zone的完整配置(services/ports/rich-rules等)
```

**一句话是什么:** firewalld 的核心抽象是 **zone**(区域)——每个网络接口都归属一个 zone,不同 zone 预置了不同的信任级别和默认放行规则(比如 `public` zone 默认只放行少数几个基础服务,`trusted` zone 则完全放行一切);在 zone 里以 **service**(预定义的服务组合,比如 `ssh` 服务实际上对应 TCP 22 端口)或 **port**(直接指定端口号)的粒度配置放行规则。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"配置防火墙设置";firewalld 是现行 RHEL 系统防火墙管理的标准工具(取代了更底层、更繁琐的直接操作 `iptables`/`nftables` 规则),zone 的概念让"同一台多网卡机器,不同网络环境用不同安全策略"变得简单。

**从最容易犯错的做法讲起:** 不理解 zone 的概念,直接在错误的 zone 上加规则,发现"明明加了规则却没生效"——很可能是这条规则加到了一个当前**没有任何网卡在用**的 zone 上;操作前先用 `--get-active-zones` 确认清楚当前实际生效的是哪个 zone,再针对这个 zone 做配置,而不是想当然地认为"默认 zone 就是当前生效的 zone"(两者通常一致,但不是绝对保证)。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 服务器只有一块网卡连接内网,合理配置成更宽松的 `internal` zone;如果是直接暴露公网的服务器,应该用更严格的 `public` zone 甚至进一步收紧规则,只放行真正需要对外提供的服务,这是纵深防御思路在防火墙配置层面的具体体现。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

assert_ok systemctl is-active firewalld

default_zone=$(firewall-cmd --get-default-zone)
assert_ok test -n "$default_zone"
echo "本机默认zone: $default_zone"

zones=$(firewall-cmd --get-zones)
echo "$zones" | grep -qw "public" && echo "OK: public zone存在(RHEL最常见的默认zone)"

active_zones=$(firewall-cmd --get-active-zones)
assert_ok test -n "$active_zones"
```
本机实测:全部检查点输出 `OK`,本机默认 zone 是 `public`。

**常见坑:** `--get-active-zones` 输出的是"当前实际有网卡在用"的 zone,这个集合可能和 `--get-default-zone` 不完全一致(新加的网卡如果没有显式指定 zone,会用默认 zone,但已有的网卡可能之前被显式配置成了别的 zone)——排查"规则不生效"问题,一律先看 active zone,不要想当然。

---

## 9. `firewall-cmd` 常用操作(`--add-service`/`--add-port`/`--permanent`/`--reload`)

**命令/配置:**
```bash
firewall-cmd --add-service=http                 # 运行时立即放行http服务(重启firewalld后失效)
firewall-cmd --permanent --add-service=http        # 写入永久配置(不会立即生效,需要reload)
firewall-cmd --reload                                # 让永久配置应用到运行时
firewall-cmd --list-services                            # 查看当前(运行时)放行了哪些服务
firewall-cmd --add-port=8080/tcp --permanent               # 直接指定端口(没有对应service定义时用这个)
```

**一句话是什么:** firewalld 的规则同样分"运行时"(立即生效,重启/reload 后丢失)和"永久"(`--permanent`,写入配置但不会立即应用,必须 `--reload` 才能同步到运行时)两个维度——这是本系列反复出现的"临时 vs 永久"设计模式在防火墙领域的具体体现,生产实践里通常**两者一起加**(先 `--permanent` 保证持久化,再不带 `--permanent` 加一次让它立即生效,或者干脆 `--permanent` 后紧跟 `--reload`)。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求配置防火墙规则并使其在系统重启后依然有效,理解"运行时"和"永久"这两个维度的区别是拿到这类题目全部分值的关键——只加运行时规则,考试判分脚本一旦模拟重启验证,规则就会消失。

**从最容易犯错的做法讲起:** 加了 `--permanent --add-port=xxx` 之后忘记 `--reload`,以为"永久配置写好了应该已经生效",结果当前会话测试访问依然被拒绝——**本机实测证伪式验证**:`--permanent` 添加的规则在 `--reload` 之前,`--list-services`/`--list-ports` 查询到的运行时状态里确实还没有这条新规则,必须执行 `--reload` 才会同步到运行时;而且 `--reload` 本身还有一个容易被忽视的连带效应——它会让运行时状态"重置为永久配置的样子",这意味着**之前只加在运行时、没有 `--permanent` 的规则,会在 reload 后被清除**(因为 reload 本质是"用永久配置重新生成运行时状态"),这是一个环环相扣、值得仔细理解的机制。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 给新部署的 Web 服务开放访问:`firewall-cmd --permanent --add-service=http`,`firewall-cmd --reload` 让它立即生效且保证持久化;如果这个服务用的是非标准端口(比如 8080 而不是标准的 80),没有对应的预定义 service,改用 `--add-port=8080/tcp` 直接指定端口。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

before_services=$(firewall-cmd --list-services)
echo "$before_services" | grep -qw "ssh" && echo "OK: ssh服务默认已放行(否则本机连不上这个环境做后续所有验证)"

firewall-cmd --add-port=8888/tcp >/dev/null 2>&1
runtime_check=$(firewall-cmd --list-ports)
echo "$runtime_check" | grep -q "8888/tcp" && echo "OK: 运行时添加(不带--permanent)的端口立即生效"

firewall-cmd --permanent --add-port=9999/tcp >/dev/null 2>&1
permanent_before_reload=$(firewall-cmd --list-ports)
echo "$permanent_before_reload" | grep -qv "9999/tcp" && echo "OK: --permanent添加的规则,reload之前运行时状态里确实还没有这条"

firewall-cmd --reload >/dev/null 2>&1
after_reload=$(firewall-cmd --list-ports)
echo "$after_reload" | grep -q "9999/tcp" && echo "OK: reload之后,永久配置的端口才真正同步到运行时"
echo "$after_reload" | grep -qv "8888/tcp" && echo "OK: 只在运行时加过、没有--permanent的旧规则,reload后确实被清除了"

firewall-cmd --permanent --remove-port=9999/tcp >/dev/null 2>&1
firewall-cmd --reload >/dev/null 2>&1
```
本机实测:全部检查点输出 `OK`,完整验证了"运行时 vs 永久""reload 会用永久配置重置运行时状态"这两条关键机制的真实生效效果。

**常见坑:** 见上方"从最容易犯错的做法"——`--reload` 会清除"仅运行时、未永久化"的规则,这个连带效应经常让人在"我加过的规则怎么又不见了"上疑惑很久,理解 reload 的本质("用永久配置重新生成运行时状态",不是"追加式合并")能立刻解开这个困惑。

---

## 10. firewalld 富规则(rich rule)基础

**命令/配置:**
```bash
firewall-cmd --add-rich-rule='rule family="ipv4" source address="192.168.1.0/24" service name="ssh" accept'
firewall-cmd --list-rich-rules
firewall-cmd --remove-rich-rule='...(和添加时完全相同的规则字符串)'
```

**一句话是什么:** 普通的 `--add-service`/`--add-port` 只能表达"放行/不放行"这种简单二元判断,富规则(rich rule)用一套结构化的语法支持更精细的条件组合——最典型的是"限定来源 IP/网段"(只允许特定网段访问某服务,而不是对所有来源一视同仁),是标准 service/port 规则和更底层手写 nftables/iptables 规则之间的一个"够用又不需要太底层"的中间层。

**为什么 RHCSA 真考 / 生产会用到:** 简单粗暴地"对所有人开放 SSH 端口"是安全隐患,生产环境几乎总是需要"只允许办公室网段/跳板机 IP 访问管理端口"这类精细化控制,富规则是 firewalld 体系内实现这类需求的标准手段,不需要下沉到手写底层防火墙规则的复杂度。

**从最容易犯错的做法讲起:** 删除富规则时,`--remove-rich-rule` 后面跟的字符串**必须和添加时完全一模一样**(一个字符、一个空格都不能差)——firewalld 是把整条规则字符串当成一个整体来匹配删除的,不支持"删除包含某个关键词的规则"这种模糊匹配;养成添加规则时就把完整命令保存下来(或者写进配置管理脚本里)的习惯,而不是凭记忆去重新敲一遍"应该差不多"的删除命令。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 只允许公司内网网段访问服务器的 SSH 管理端口,其余所有来源一律拒绝:`firewall-cmd --permanent --add-rich-rule='rule family="ipv4" source address="10.0.0.0/8" service name="ssh" accept'`,配合把默认 zone 收紧成不放行裸露的 22 端口(只有匹配这条富规则的来源才能访问 SSH),是一套精细化访问控制的组合拳。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

firewall-cmd --add-rich-rule='rule family="ipv4" source address="192.168.50.0/24" service name="ssh" accept' >/dev/null 2>&1
assert_eq "$?" "0"

rich_rules=$(firewall-cmd --list-rich-rules)
echo "$rich_rules" | grep -q "192.168.50.0/24" && echo "OK: 富规则精确限定了'仅这个来源网段'才能访问ssh服务"

firewall-cmd --remove-rich-rule='rule family="ipv4" source address="192.168.50.0/24" service name="ssh" accept' >/dev/null 2>&1
assert_eq "$(firewall-cmd --list-rich-rules | grep -c '192.168.50.0/24')" "0"    # 精确匹配删除成功
```
本机实测:全部断言输出 `OK`。

**常见坑:** 富规则和普通 service/port 规则**可以同时存在、共同生效**(不是互斥关系)——如果一个 zone 里既有裸的 `--add-service=ssh`(对所有来源开放)又有一条限定来源的富规则,裸的 service 规则依然会放行其他来源的访问,富规则的精细限制不会自动"收紧"已经存在的宽松规则;真正要做到"只允许特定来源"的效果,必须确保没有另一条更宽松的规则同时允许了其他来源,两者要配合设计,不能只加富规则就以为万事大吉。

---

*本篇完成:2026-07-11,10 个知识点。验证环境:Rocky Linux 10.2(WSL2)。防火墙部分(第 8-10 节)功能完整、验证扎实。SELinux 部分(第 1-7 节)经过三层递进排查,确认受 WSL2 内核结构性限制(内核参数不经过 GRUB 生效、selinuxfs 接口残缺、auditd 因 audit netlink 权限被拒无法启动)——`semanage`(操作策略数据库本身)和工具存在性/语法部分已扎实验证,真正的 enforcing 强制访问控制效果、AVC 拒绝事件排查演示,需要在真实 RHEL 环境下完成,已在各小节如实标注,不冒充已验证。*
