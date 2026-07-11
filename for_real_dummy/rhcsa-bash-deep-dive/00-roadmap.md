# RHCSA 全景 · Linux 系统管理与 bash 编程深挖 —— 路线图与进度表

> 目标:100 个 RHCSA(EX200) 级别的 Linux 系统管理 / bash 编程知识点,由浅入深,9 个分类,分批次完成。
> 背景:前四条系列(numpy/torch/python-advanced/python-idioms)补的是"读懂这个 AI/ML 仓库"需要的 Python 能力;这一条补的是完全独立的新领域——**Linux 系统管理 + bash 脚本编程**,对标 Red Hat 官方 RHCSA(EX200) 认证的技能范围。
> 范围决策(2026-07-11 与用户确认):聚焦 **RHCSA(EX200)**,不含 RHCE(EX294)——因为 RHCE 现已 100% 转向 Ansible playbook 自动化,不再考 bash/命令行,和"linux bash 编程"这个原始需求关系不大。RHCSA 官方基准 2026-05 起已从 RHEL 9 切换到 RHEL 10,具体命令细节撰写时需对照官方 RHEL 10 文档核实,不能凭旧版本知识直接照搬。

---

## 本系列和前四条系列的关键差异(必须诚实面对,不能假装一样)

1. **"真实例子"来源不同**:numpy/torch/python-idioms 都能从仓库里博士学长的真实代码挖到例子。这个仓库是 AI/ML 研究仓库,**没有 Linux 系统管理场景可挖**(已用 `find` 确认仓库内所有 `.sh` 文件全部是 `.venv/` 或 `official/repos/` 下 vendor 进来的第三方依赖脚本,不算"自己的代码")。本系列第 5 步统一标注为"典型生产运维场景 / RHCSA 真实考试场景",不冒充仓库代码里挖出来的。
2. **验证环境不同**:前四条系列纯 Python,跨平台一致,在仓库 `.venv` 里跑。本系列的系统管理类命令(`systemctl`/LVM/`firewalld`/SELinux 等)是 systemd/RHEL 生态特有的,Windows 的 Git Bash 装不出这种环境,统一在 **WSL2 的 Rocky Linux**(RHEL 下游二进制兼容发行版)里真实验证。纯 bash 脚本语法部分(第 09 类)在 Git Bash 下也能验证,但为了和系统管理部分保持一致的验证环境,同样以 Rocky Linux 为准。
3. **bash 没有原生 `assert`**:统一用下面两个函数做等价断言,效果对应 Python 系列的 `assert`:
   ```bash
   assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
   assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }
   ```
   `assert_eq` 比较两个字符串(命令输出 vs 期望值);`assert_ok` 直接执行一条命令并检查退出码是否为 0。每篇可运行例子默认这两个函数已在当前 shell 会话定义好。

---

## 每个知识点的固定讲解结构(七步,在 python-idioms 六步基础上改造)

1. **命令/配置**——签名,人话翻译
2. **一句话是什么**
3. **为什么 RHCSA 真考 / 生产会用到**
4. **从最容易犯错/最不安全的做法讲起**——危险操作 vs 正确操作对比(对应 python-idioms 的"笨办法"环节,系统管理场景里价值更高——很多 RHCSA 扣分点就是"能达到效果但方式错误/不持久/不安全")
5. **真实场景例子**——如实标注为典型运维/考试场景,不冒充仓库代码
6. **可运行例子**——Rocky Linux WSL 环境真实跑通,用 `assert_eq`/`assert_ok` 做断言
7. **常见坑**——含 RHCSA 判分点提示

不采用 torch 系列的"面试怎么问"环节——RHCSA 是纯上机操作考试,没有面试环节,该框架不适用。

---

## 环境前置说明

- 验证环境:WSL2 → Rocky Linux 发行版(`wsl.exe -d RockyLinux`)。
- 连通性检查:`wsl.exe -d RockyLinux -- whoami`——**截至 2026-07-11,该发行版仍报 `getpwuid(0) failed` 错误,处于半初始化状态,尚未可用**,需要用户在 Windows 侧修复(默认用户配置问题)后才能开始需要真实环境验证的知识点。
- root/sudo:待环境修复后确认该发行版默认是 root 会话还是需要 sudo 提权,不预设。
- LVM/存储类实验一律用 `dd if=/dev/zero of=/tmp/xxx.img bs=1M count=N` + `losetup` 构造 loop device,不动真实磁盘,验证完在同一代码块内清理干净。

---

## 进度表(由浅入深;撰写顺序见下方"撰写顺序"说明,和分类编号顺序不完全一致)

| # | 分类 | 文件 | 知识点数 | 状态 |
|---|------|------|---------|------|
| 01 | 必备工具与文本处理 | [01-essential-tools.md](01-essential-tools.md) | 14 | ✅ 已完成(14个代码块 Git Bash 下实际执行全部通过;第1-7/14、14项为shell/GNU工具语法本身,和真实RHEL行为一致,完整验证;第8-13项依赖Windows内核/文件系统语义,已如实记录与RHEL的真实差异——其中硬链接、chmod语法、vim、ssh-keygen本地部分、软链接的"实际降级行为"均完整验证,但chmod的完整ugo/rwx强制效果、man/info交互、ssh-copy-id、scp/rsync远程传输能力受限于本机环境(无man/info/rsync/无可达远程主机)未能验证,需 Rocky Linux 环境修复后补充,详见文内各条"常见坑") |
| 02 | 进程与系统运行 | [02-process-and-boot.md](02-process-and-boot.md) | 12 | ⏳ 待开始 |
| 03 | 本地存储与 LVM | [03-storage-and-lvm.md](03-storage-and-lvm.md) | 12 | ⏳ 待开始 |
| 04 | 文件系统与权限 | [04-filesystem-and-permissions.md](04-filesystem-and-permissions.md) | 13 | ⏳ 待开始 |
| 05 | 软件与系统部署 | [05-package-and-deployment.md](05-package-and-deployment.md) | 10 | ⏳ 待开始 |
| 06 | 用户组管理 | [06-users-and-groups.md](06-users-and-groups.md) | 10 | ⏳ 待开始 |
| 07 | 网络配置 | [07-networking.md](07-networking.md) | 10 | ⏳ 待开始 |
| 08 | 安全:SELinux 与防火墙 | [08-security-selinux-firewall.md](08-security-selinux-firewall.md) | 10 | ⏳ 待开始 |
| 09 | bash 脚本编程本身 | [09-bash-scripting.md](09-bash-scripting.md) | 9 | ✅ 已完成(已验证,27个代码块 Git Bash 下语法检查+实际执行全部通过) |

**合计:100 个知识点,9 篇,23/100 完成(2/9 篇)。**(状态如实反映——没有验证过就不标"已完成",参照 qa/03 记录里"不能一度提前标全部完成"的教训)

**撰写顺序(按 root 依赖/风险递增,不是文件编号顺序;2026-07-11 实测修正)**:
1. 09 bash 脚本本身——**已完成**,不需要 root,纯语言特性,Git Bash 验证足够真实。
2. 01 工具类**原计划预期"不需要 root",但实测证伪**:在当前 Windows/NTFS 环境下 `chmod` 是空操作(644 改成 600 再查还是 644,权限位没有真正生效)、`ln -s` 在当前权限下退化成普通文件而非真符号链接(`ls -la` 显示 `-rw-r--r--` 而非 `lrwxrwxrwx`)、`man`/`rsync` 命令不存在。14 个知识点里约 6 个(硬链接与软链接、标准权限模型、man/info 文档、SSH 完整登录流程、scp/rsync 传输)必须依赖 Rocky Linux 才能获得真实验证,不能提前用 Git Bash 做出"看起来通过但实际没验证到真实语义"的假结果。~~因此 01 类和 02-08 类一样,整体推迟到 Rocky Linux 环境修复后统一完成~~,保持同一文件内部的验证质量一致,不做"部分知识点验证过、部分没有"的参差状态。
   **2026-07-11 更新,上述"整体推迟"被更明确的指示取代:** 不再要求"一篇文档内部验证颗粒度完全整齐"这个前提,改为在同一篇文档内按知识点**分别标注验证颗粒度**——shell/GNU 工具语法本身(和 RHEL 行为一致)完整验证,依赖 Windows 内核/文件系统真实语义的条目逐条如实记录"验证到什么程度、和真实 RHEL 差异是什么"(而不是笼统标"已验证"或干脆整篇搁置)。01-essential-tools.md 已按此标准完成撰写并现场跑通全部 14 个代码块,不再等待 Rocky Linux 环境修复;其中"软链接在无权限的 Windows 上会静默降级成普通拷贝"本身作为一条真实差异被完整记录(而不是被跳过),chmod 的完整 ugo/rwx 强制效果、man/info 交互、ssh-copy-id、scp/rsync 远程传输能力仍受限于本机环境未能验证,已在文内逐条标注,留待 Rocky Linux 环境修复后补充。
3. 02 进程系统运行、04 文件系统权限、05 软件部署——部分操作需要 root,风险和复杂度较低。
4. 03 存储 LVM、06 用户组、07 网络、08 安全——需要 root + 构造临时资源或有"配置错了可能锁死自己/断连 WSL"的风险,放最后、单独小心处理。

---

## 每一批具体覆盖哪些知识点(明细)

### 01 必备工具与文本处理(14)
1. 文件与目录基本操作回顾(`cp`/`mv`/`rm`/`mkdir` 的易错选项:`-r`/`-i`/`-v`/`-p`)
2. 通配符与路径展开(glob:`*`、`?`、`[]`、`{}`)
3. I/O 重定向进阶(`>`、`>>`、`2>`、`2>&1`、`&>`、管道 `|`)
4. grep 与正则表达式(BRE vs ERE,`-E`/`-P`,常用元字符)
5. sed 流编辑器基础(替换/删除/行选择)
6. find 查找文件(按名称/类型/时间/大小/权限,`-exec`)
7. tar 归档与压缩(与 gzip/bzip2/xz 的组合参数)
8. 硬链接与软链接(inode 本质区别,`ln -s` 常见坑)
9. 标准权限模型 ugo/rwx(`chmod` 符号法 vs 八进制法)
10. man/info 与系统自带文档(`--help`、`man -k`、`/usr/share/doc`)
11. vim 基础操作(三种模式切换、常用命令、`:wq!` 的坑)
12. SSH 密钥认证登录(`ssh-keygen`、`ssh-copy-id`、`~/.ssh/config`)
13. scp/rsync 文件传输(增量同步、保留权限)
14. 文件哈希校验(`sha256sum`/`md5sum` 及一致性校验用法)

### 02 进程与系统运行(12)
1. systemd 启动流程与 target(runlevel 到 target 的映射,`default.target`)
2. 单用户/救援模式排障(`rd.break`、`systemd.unit=rescue.target`)
3. systemctl 服务管理(`start`/`stop`/`enable`/`disable`/`status` 核心用法)
4. systemctl 进阶(`mask`/`unmask`、`daemon-reload`、依赖关系查看)
5. 进程查看 `ps`/`top`(常用列解读,`ps aux` vs `ps -ef`)
6. 进程信号与 kill(`SIGTERM` vs `SIGKILL`,`killall`/`pkill`)
7. 进程优先级 `nice`/`renice`
8. journalctl 日志查询(按时间/服务/优先级过滤,`-f` 跟踪)
9. rsyslog 传统日志(`/var/log/messages`,logrotate 基础)
10. cron 定时任务(crontab 语法,`/etc/cron.d`)
11. systemd timer(对比 cron 的优势,`.timer`+`.service` 配对)
12. 关机重启命令族(`shutdown`/`reboot`/`poweroff` 的区别与安全用法)

### 03 本地存储与 LVM(12)
1. 磁盘分区基础(MBR vs GPT,`fdisk` vs `parted` vs `gdisk`)
2. parted 交互式分区操作
3. `lsblk`/`blkid` 查看磁盘与 UUID
4. LVM 概念与三层模型(PV/VG/LV)
5. 创建物理卷 `pvcreate`/`pvdisplay`
6. 创建卷组 `vgcreate`/`vgextend`
7. 创建逻辑卷 `lvcreate`(线性卷)
8. 扩展逻辑卷 `lvextend` + 文件系统同步扩容(`resize2fs`/`xfs_growfs`)
9. 缩减/删除 LV/VG/PV 的正确顺序
10. swap 分区/文件的创建与启用(`mkswap`、`swapon`、`/etc/fstab` 条目)
11. 磁盘配额基础(quota 工具链)
12. 用 loop device 模拟磁盘做实验(`dd`+`losetup`,本系列验证专用技巧,和"环境前置说明"呼应)

### 04 文件系统与权限(13)
1. `mkfs` 创建文件系统(ext4 vs xfs 的选择与差异)
2. `/etc/fstab` 语法与 UUID 挂载(手改导致开机失败的经典坑)
3. `mount`/`umount` 手动挂载与临时挂载
4. 特殊权限位 SUID/SGID/Sticky Bit
5. SGID 目录用于团队协作共享
6. umask 默认权限计算
7. ACL 访问控制列表(`setfacl`/`getfacl`,mask 的坑)
8. 默认 ACL(目录继承)
9. 排查权限问题的系统方法(`namei -l`,从根逐级检查)
10. LUKS 磁盘加密(`cryptsetup luksFormat`/`luksOpen`)
11. NFS 客户端挂载(`showmount`,`/etc/fstab` 里的 NFS 条目)
12. autofs 自动挂载配置
13. VDO 虚拟数据优化基础概念(RHEL 存储去重压缩技术)

### 05 软件与系统部署(10)
1. dnf 包管理基础(`install`/`remove`/`search`/`info`)
2. dnf 仓库管理(repo 文件,`dnf repolist`)
3. rpm 底层查询(`rpm -qa`/`-qi`/`-ql`,和 dnf 的关系)
4. 创建本地/离线仓库(`createrepo` 基础)
5. GRUB2 配置与内核参数修改
6. 内核参数临时/永久调整(`sysctl`)
7. chronyd 时间同步配置
8. podman 容器基础(拉取/运行/管理容器,rootless 特性)
9. podman 生成 systemd 服务(quadlet / `podman generate systemd` 概念)
10. 软件源 GPG 签名校验

### 06 用户组管理(10)
1. `useradd` 创建用户(常用选项 `-m`/`-s`/`-G`/`-u`)
2. `usermod` 修改用户属性
3. `userdel` 删除用户(`-r` 清理家目录的坑)
4. `groupadd`/`groupmod`/`groupdel` 组管理
5. `passwd` 密码管理与 `chage` 密码时效策略
6. 锁定解锁账户(`passwd -l`/`-u`,`usermod -L`/`-U`)
7. `/etc/passwd`、`/etc/shadow`、`/etc/group` 文件结构解读
8. UID/GID 分配规则(系统用户 vs 普通用户的边界)
9. `/etc/skel` 与新用户初始化模板
10. sudo 配置(`visudo`,`/etc/sudoers.d/`,最小权限原则)

### 07 网络配置(10)
1. `nmcli` 查看网络状态(device/connection 概念)
2. `nmcli` 配置静态 IP
3. `nmtui` 交互式网络配置
4. 主机名管理(`hostnamectl`)
5. `/etc/hosts` 本地解析
6. DNS 客户端配置(`/etc/resolv.conf`,NetworkManager 接管)
7. 网络连通性排查工具(`ping`/`ss`/`ip`)
8. `ip` 命令族(`ip addr`/`ip route` 替代 `ifconfig`/`route`)
9. 网络服务的开机管理(NetworkManager vs `network.service`)
10. 主机路由表查看与静态路由添加

### 08 安全:SELinux 与防火墙(10)
1. SELinux 三种模式(enforcing/permissive/disabled)
2. SELinux 上下文基本概念(`user:role:type:level`)
3. `ls -Z`/`ps -Z` 查看 SELinux 上下文
4. `restorecon`/`chcon` 恢复修改上下文
5. `semanage` 管理 SELinux 策略持久化
6. SELinux 布尔值(`getsebool`/`setsebool`)
7. SELinux 故障排查(`ausearch`/`sealert`,常见"明明权限对了但访问被拒")
8. firewalld 基本概念(zone/service/port)
9. `firewall-cmd` 常用操作(`--add-service`/`--add-port`/`--permanent`/`--reload`)
10. firewalld 富规则 rich rule 基础

### 09 bash 脚本编程本身(9)
1. 变量与作用域(局部/全局/`export` 环境变量)
2. 条件判断(`[ ]` vs `[[ ]]` vs `(( ))` 的区别与选用)
3. 循环结构(`for`/`while`/`until` 及 `break`/`continue`)
4. 函数定义与参数(`$1`/`$@`/`$#`/`return` vs `echo` 传值)
5. 数组(索引数组与关联数组 `declare -A`)
6. 参数展开与字符串处理(`${var#pattern}` 系列,`${var/old/new}`)
7. here-doc 与 here-string(`<<EOF`,`<<<`)
8. trap 信号处理与退出清理(`EXIT`/`ERR` trap)
9. 脚本调试与健壮性(`set -euo pipefail`,shellcheck)

---

*创建:2026-07-11*
