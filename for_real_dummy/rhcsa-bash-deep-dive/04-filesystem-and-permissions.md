# 04 · 文件系统与权限(File Systems & Permissions)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一篇覆盖 13 个知识点:创建挂载文件系统、`/etc/fstab`、特殊权限位、ACL、权限排查方法、LUKS 加密、NFS、autofs、VDO。**本文所有代码例子已在 Rocky Linux 10.2(WSL2,真实 systemd)下实际跑通验证**,LVM/存储类实验按 [00-roadmap.md](00-roadmap.md) 既定纪律用 `dd`/`truncate` + `losetup` 构造 loop device,不动真实磁盘,验证完在代码块内清理干净。
>
> **和前四条系列的差异声明**(详见 [00-roadmap.md](00-roadmap.md) 开头):本仓库没有 Linux 系统管理场景可挖,"真实场景例子"部分统一标注为典型运维/RHCSA 考试场景,不冒充仓库代码里挖出来的。

---

## 1. `mkfs` 创建文件系统(ext4 vs xfs 的选择与差异)

**命令/配置:**
```bash
mkfs.ext4 /dev/sdX      # 创建ext4文件系统
mkfs.xfs /dev/sdX        # 创建xfs文件系统
mkfs.xfs -f /dev/sdX     # -f强制覆盖已存在的文件系统签名
blkid /dev/sdX            # 查看设备上文件系统的类型/UUID等信息
```

**一句话是什么:** `mkfs`(make filesystem)在一个块设备上写入文件系统的元数据结构,让它从"一堆裸字节"变成操作系统能识别、能存文件的文件系统;RHEL 生态里 `ext4` 和 `xfs` 是最常见的两种选择,RHEL 7 之后 `xfs` 是默认文件系统,但 `ext4` 依然广泛使用,两者互有取舍。

**为什么 RHCSA 真考 / 生产会用到:** "创建并配置文件系统"是 RHCSA 明确列出的技能;生产环境选文件系统不是随便选的——`xfs` 对大文件、高并发场景性能更好且原生支持在线扩容,`ext4` 生态更成熟、工具链更丰富、小分区场景开销更低,需要理解两者差异才能做出合理选择而不是盲选。

**从最容易犯错的做法讲起:** 想当然地认为"文件系统之间只是名字不同,创建门槛都一样"——**本机实测证伪**:`ext4` 可以在几十 MB 的小分区上正常创建,但 `xfs` 有**硬性的最小 300MB 要求**,本机用 100MB 的 loop device 尝试 `mkfs.xfs` 直接报错拒绝(`Filesystem must be larger than 300MB.`),换成 400MB 才成功——这是两种文件系统设计取向的直接体现:`xfs` 从设计上就面向"大容量、高性能"场景,不为极小分区做优化。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 给数据库这类需要频繁大文件读写、未来可能要在线扩容的存储卷,`xfs` 是更主流的选择(`xfs_growfs` 支持在线扩容,`ext4` 的 `resize2fs` 也支持增长但某些历史版本对在线收缩支持更弱);RHCSA 考试如果只是要求"创建一个文件系统并挂载",两者都是正确答案,除非题目明确指定类型。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

dd if=/dev/zero of=/tmp/rhcsa04_disk1.img bs=1M count=400 2>/dev/null   # xfs最小300MB,留余量用400MB
loop1=$(losetup -f)
losetup "$loop1" /tmp/rhcsa04_disk1.img

mkfs.ext4 -q "$loop1"
assert_eq "$(blkid -o value -s TYPE "$loop1")" "ext4"

mkfs.xfs -f -q "$loop1"
assert_eq "$(blkid -p -o value -s TYPE "$loop1")" "xfs"   # -p强制重新探测,不吃blkid缓存

losetup -d "$loop1"
rm -f /tmp/rhcsa04_disk1.img
```
本机实测:两个 `assert_eq` 均输出 `OK`。

**常见坑:**
1. `xfs` 最小 300MB 的门槛,见上方"从最容易犯错的做法"——测试/实验环境规划磁盘大小时容易踩到。
2. `blkid` 默认可能返回缓存结果,同一个设备重新格式化成别的文件系统类型后,不加 `-p`(强制重新探测)直接查询,有极小概率拿到过期信息;`-p` 更保险。
3. `xfs` 文件系统一旦创建**不支持缩小**(只能增长),`ext4` 在卸载状态下能用 `resize2fs` 缩小——如果预期未来可能需要缩容,这是选型时要考虑的硬约束,不是"哪个更快"这种可以事后补救的问题。

---

## 2. `/etc/fstab` 语法与 UUID 挂载(手改导致开机失败的经典坑)

**命令/配置:**
```
UUID=xxxx-xxxx  /mnt/data  ext4  defaults  0 0
```
```bash
mount -a          # 挂载fstab里所有还没挂载的条目
mount -fav        # 只做语法校验,不真正挂载(安全的"预演"模式)
```

**一句话是什么:** `/etc/fstab` 定义"开机时自动挂载哪些文件系统",六个字段依次是:设备标识、挂载点、文件系统类型、挂载选项、`dump` 备份标记、`fsck` 检查顺序;设备标识**推荐用 UUID 而不是 `/dev/sdX` 这种设备名**,因为设备名可能因为硬件插拔顺序变化,UUID 是文件系统创建时生成的、不会变的唯一标识。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"配置系统开机自动挂载文件系统",这是必考操作;`/etc/fstab` 写错格式是导致 RHEL 系统"开机进 emergency shell"最常见的原因之一,排障能力和正确配置能力同等重要。

**从最容易犯错的做法讲起:** 直接手工 `vi /etc/fstab` 改完就重启验证,如果哪个字段写错了(比如挂载点路径打错、UUID 抄错一个字符),系统重启后会因为 `mount -a` 失败而卡在 emergency shell——正确的防御性流程是改完之后先执行 `mount -fav`(只校验语法/可行性,不真正挂载,不会因为出错而搞坏当前运行中的系统),确认没问题再重启验证,而不是"改完直接赌一把重启"。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 新加一块数据盘,`mkfs` 完成后用 `blkid` 拿到 UUID,写入 `/etc/fstab` 一行 `UUID=xxx /data ext4 defaults 0 0`,`mount -a` 让它立即生效(不用重启就能验证配置对不对),这是标准的"加盘"流程闭环。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

# (接上一节创建好的loop device,这里重新构造一个独立场景)
dd if=/dev/zero of=/tmp/rhcsa04_disk1.img bs=1M count=400 2>/dev/null
loop1=$(losetup -f); losetup "$loop1" /tmp/rhcsa04_disk1.img
mkfs.ext4 -q "$loop1"
disk_uuid=$(blkid -o value -s UUID "$loop1")

mkdir -p /mnt/rhcsa04_test
echo "UUID=$disk_uuid /mnt/rhcsa04_test ext4 defaults 0 0" >> /etc/fstab
mount -a
assert_eq "$(mount | grep -c "/mnt/rhcsa04_test")" "1"

# mount -fav:考试改完fstab后的标准验证步骤,只校验不实际挂载
fstab_verify=$(mount -fav 2>&1)
echo "$fstab_verify" | grep -q "rhcsa04_test" && echo "OK: mount -fav 能验证fstab条目语法正确性"

umount /mnt/rhcsa04_test
sed -i "\|rhcsa04_test|d" /etc/fstab    # 清理测试条目,不留痕迹
losetup -d "$loop1"; rm -f /tmp/rhcsa04_disk1.img; rmdir /mnt/rhcsa04_test
```
本机实测:两个断言均输出 `OK`。

**常见坑:** 最后一个字段(`fsck` 顺序)填错也会埋雷——根文件系统必须是 `1`,其他文件系统习惯填 `2`(表示晚于根文件系统检查),填 `0` 表示"从不自动 fsck",对需要定期完整性检查的重要数据盘来说不是好选择;这个字段容易被无脑抄成 `0 0` 敷衍了事,但理解它的含义能避免长期积累的文件系统一致性问题不被发现。

---

## 3. `mount`/`umount` 手动挂载与临时挂载

**命令/配置:**
```bash
mount /dev/sdX /mnt/point       # 手动挂载(不写入fstab,重启后失效)
mount -t ext4 /dev/sdX /mnt/point   # 显式指定文件系统类型(通常可以省略,自动探测)
umount /mnt/point                 # 卸载
umount -l /mnt/point               # 懒卸载:立即从命名空间摘除,但等所有引用释放后才真正解除
```

**一句话是什么:** `mount`/`umount` 是"立即生效、当次会话有效"的挂载/卸载操作,不会持久化——重启后如果没有对应的 `/etc/fstab` 条目,挂载不会自动恢复,这和上一节"写进 fstab 让它开机自动挂载"是两个独立的能力,经常配合使用但职责不同。

**为什么 RHCSA 真考 / 生产会用到:** 临时挂载一块新盘、临时挂载一个 ISO 镜像做安装源、排障时临时挂载检查文件系统内容,都是不需要"每次开机都自动挂载"的场景,`mount`/`umount` 是这类临时操作的标准工具。

**从最容易犯错的做法讲起:** 卸载一个"正忙"(有进程打开了里面的文件,或者当前 shell 的工作目录就在挂载点内)的文件系统,直接 `umount` 会报 `target is busy` 拒绝执行——新手常见的错误应对是不分青红皂白直接上 `umount -f`(强制)或者 `umount -l`(懒卸载),但更负责任的做法是先用 `lsof +D /mnt/point` 或 `fuser -v /mnt/point` 找出到底是谁在占用,视情况决定是等它自己结束、手动关闭,还是确实需要强制卸载。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 临时挂载一个 ISO 文件做安装源:`mount -o loop rhel10.iso /mnt/iso`(`-o loop` 让 mount 自动关联一个 loop 设备,不需要手动 `losetup`);排障时怀疑某块盘的数据没写全,临时挂载到一个独立目录里人工检查内容,查完就 `umount`,不需要为这种一次性操作污染 `/etc/fstab`。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

dd if=/dev/zero of=/tmp/rhcsa04_disk1.img bs=1M count=400 2>/dev/null
loop1=$(losetup -f); losetup "$loop1" /tmp/rhcsa04_disk1.img
mkfs.ext4 -q "$loop1"
mkdir -p /mnt/rhcsa04_test

mount "$loop1" /mnt/rhcsa04_test
assert_eq "$(mount | grep -c "/mnt/rhcsa04_test")" "1"
echo "test content" > /mnt/rhcsa04_test/file.txt
umount /mnt/rhcsa04_test
assert_eq "$(mount | grep -c "/mnt/rhcsa04_test")" "0"

# 卸载不等于删除数据:重新挂载,内容还在
mount "$loop1" /mnt/rhcsa04_test
assert_eq "$(cat /mnt/rhcsa04_test/file.txt)" "test content"
umount /mnt/rhcsa04_test

losetup -d "$loop1"; rm -f /tmp/rhcsa04_disk1.img; rmdir /mnt/rhcsa04_test
```
本机实测:三个断言均输出 `OK`。

**常见坑:** `mount` 手动挂载的文件系统,重启后**不会自动恢复**,这是新手最容易掉的坑——手动挂载测试没问题之后,如果这个挂载点是要长期使用的,一定要记得同步写入 `/etc/fstab`(见上一节),否则下次重启这个目录就变回空的了,容易被误以为"数据丢了",实际上数据还在磁盘上,只是没挂载而已。

---

## 4. 特殊权限位 SUID/SGID/Sticky Bit

**命令/配置:**
```bash
chmod u+s file      # SUID:执行时以文件owner身份运行,而不是执行者身份
chmod g+s file       # SGID(对文件):执行时以文件所属组身份运行
chmod g+s dir         # SGID(对目录):目录里新建文件自动继承目录的组(见下一节)
chmod +t dir          # Sticky Bit:目录里的文件只有owner自己能删,即便其他人对目录有写权限
chmod 4755 file       # 数字法:第一位4=SUID,2=SGID,1=Sticky,可以相加组合
```

**一句话是什么:** 这三个是在标准 ugo/rwx 九个权限位之外、额外的三个特殊标志位,分别解决三个不同问题:SUID 让程序临时"借用"owner 的身份权限执行(经典例子是 `passwd` 命令本身需要 root 权限改 `/etc/shadow`,但普通用户也能执行它改自己的密码);SGID 在目录上用于"组内协作共享";Sticky Bit 用于"公共可写目录防止互相删文件"(`/tmp` 就是典型例子)。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确列出"列出、设置标准 ugo/rwx 权限"和理解特殊权限位;安全加固场景需要审计系统里"意外带 SUID 的可执行文件"(这是权限提升攻击的常见入口),理解 SUID 是排查安全隐患的基础。

**从最容易犯错的做法讲起:** 看到八进制权限第一位是 `4`/`2`/`1` 这类数字,不理解这是独立于后面三位 ugo 权限的"第四位",误以为 `chmod 4755` 里的 `4` 和后面的 `755` 是不相关的两件事分别设置——实际上 `4755` 完整地表示"SUID + rwxr-xr-x",第一位数字和后三位数字要作为一个整体记忆,`stat -c '%a'` 查询时也会完整显示这四位数字。

**真实场景例子(典型运维场景,非仓库代码):** `/usr/bin/passwd` 本身就带 SUID(owner 是 root),这样普通用户执行它时能以 root 权限写 `/etc/shadow`(否则普通用户连自己的密码都没法改,因为 `/etc/shadow` 本身对普通用户是不可写的);`/tmp` 目录权限是 `1777`(Sticky Bit + 所有人可读写执行),任何用户都能在里面创建文件,但只有文件的 owner 自己能删除/改名它,防止用户之间互相破坏彼此的临时文件。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

WORKDIR=$(mktemp -d /tmp/rhcsa04_demo.XXXXXX); cd "$WORKDIR"

touch suid_test
chmod 755 suid_test    # 先设成明确的基础权限,不依赖外部二进制的不可预测原始权限
assert_eq "$(stat -c '%A' suid_test)" "-rwxr-xr-x"
chmod u+s suid_test
assert_eq "$(stat -c '%A' suid_test)" "-rwsr-xr-x"     # s出现在owner的x位置
assert_eq "$(stat -c '%a' suid_test)" "4755"            # 八进制第一位4代表SUID
chmod g+s suid_test
assert_eq "$(stat -c '%A' suid_test)" "-rwsr-sr-x"     # SUID+SGID同时生效

mkdir sticky_test; chmod +t sticky_test
stat -c '%A' sticky_test | grep -q "t$" && echo "OK: sticky bit 生效,权限位末尾显示t"

cd /; rm -rf "$WORKDIR"
```
本机实测:全部断言输出 `OK`。

**常见坑:** 如果 SUID/SGID 设置在一个**没有对应执行权限**的文件上,`ls -l` 会显示大写的 `S`(而不是小写 `s`)——这是一个视觉上的警告信号,提示"这个特殊权限位设置了,但因为缺少对应的 x 执行位,实际上不会生效",很多人只看到字母 `S`/`s` 存在就以为配置成功了,没注意到大小写传达的完全不同的信息。

---

## 5. SGID 目录用于团队协作共享

**命令/配置:**
```bash
chmod g+s /shared/dir      # 或者用数字法: chmod 2775 dir
chgrp teamgroup /shared/dir
chmod 2775 /shared/dir      # 常见组合:SGID + rwxrwxr-x
```

**一句话是什么:** 普通情况下,新建文件的所属组是创建者的**主组**;给目录设置 SGID 后,这条规则被改写——目录里新建的所有文件/子目录,所属组自动变成**目录本身的组**,而不是创建者的主组,这是让多个用户在共享目录里协作、同时保证文件组权限一致的标准手段。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"创建配置带 set-GID 目录用于团队协作共享";生产环境里项目组共享目录几乎都要配置这个,否则不同成员创建的文件属于不同的主组,组内其他人反而看不到彼此的文件(如果标准权限模型的组权限设置得比较严格的话)。

**从最容易犯错的做法讲起:** 只设置了目录权限为 `775`(组内可读写),却忘记加 SGID——这样虽然组内成员对目录本身有写权限,但每个人在里面新建的文件依然归属**各自的主组**,如果团队成员的主组本来就不统一(比如每个用户默认主组是和用户名同名的私有组,这是 RHEL 的默认策略),那"组内共享"这个目标根本没达到,新建的文件对同事来说可能仍然不可写甚至不可读。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 创建一个 `/srv/project` 共享目录给开发团队,`groupadd devteam`、把所有成员加入这个组、`chgrp devteam /srv/project`、`chmod 2775 /srv/project`——之后不管哪个成员在这个目录下建文件,自动都属于 `devteam` 组,配合默认的组读写权限,团队协作不会因为文件归组不一致而互相看不到/改不了对方的文件。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

mkdir /tmp/rhcsa04_shared_dir
chmod 2775 /tmp/rhcsa04_shared_dir
stat -c '%A' /tmp/rhcsa04_shared_dir | grep -qx "drwxrwsr-x" && echo "OK: SGID目录权限位正确显示(s在group的x位置)"

groupadd rhcsa_shared_grp 2>/dev/null
chgrp rhcsa_shared_grp /tmp/rhcsa04_shared_dir
dir_group=$(stat -c '%G' /tmp/rhcsa04_shared_dir)

touch /tmp/rhcsa04_shared_dir/newfile.txt
newfile_group=$(stat -c '%G' /tmp/rhcsa04_shared_dir/newfile.txt)
assert_eq "$newfile_group" "$dir_group"    # 新文件的组=目录的组,不是创建者root的主组(root主组通常也叫root,这里能验证出确实继承了目录组而非默认主组)

rm -rf /tmp/rhcsa04_shared_dir
groupdel rhcsa_shared_grp 2>/dev/null
```
本机实测:两个断言均输出 `OK`。

**常见坑:** SGID 目录的"组继承"只对**新建**的文件生效,已经存在、之前用其他组创建的旧文件不会被自动改组——迁移一个目录到"团队共享"模式时,记得对已有内容额外执行一次 `chgrp -R teamgroup /shared/dir` 补齐历史文件的组归属,不能只设了 SGID 就以为万事大吉。

---

## 6. umask 默认权限计算

**命令/配置:**
```bash
umask           # 查看当前umask值
umask 022        # 设置umask(当前shell会话有效)
# 文件默认起点666(rw-rw-rw-,不给可执行位),目录默认起点777(rwxrwxrwx)
# 最终权限 = 起点 与 umask按位"清除"(不是简单相减,但对022这类常见值效果等同于减法)
```

**一句话是什么:** 新建文件/目录的默认权限不是随便定的,而是一个"起点"(文件 666,目录 777)减去 `umask` 里设置的"要屏蔽掉的位"得到的结果——`umask` 定义的是"默认**不**给哪些权限",不是"默认给哪些权限",这个反着来的设计是新手理解 umask 最容易卡壳的地方。

**为什么 RHCSA 真考 / 生产会用到:** 理解 umask 能解释"为什么我新建的文件权限长这样",是排查权限相关问题的基础知识;RHCSA 环境配置类题目可能要求"配置默认权限策略",umask 是标准手段(通常在 `/etc/profile` 或 `/etc/bashrc` 里全局配置)。

**从最容易犯错的做法讲起:** 把 umask 值和"最终权限值"搞反,以为 `umask 022` 意味着"给文件 022 这个权限"——umask **越大代表限制越多、默认权限越少**,这和 `chmod` 数字法"数字越大权限越开放"的直觉正好相反,容易搞混:`umask 022` 下文件默认是 644(相对宽松),`umask 077` 下文件默认是 600(严格得多,组和其他人完全没有任何权限)。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 多用户共享服务器上,给需要更严格默认权限的账号(比如存放敏感数据的服务账号)在其 `~/.bashrc` 里设置 `umask 077`,确保这个账号新建的任何文件默认只有自己能访问,不用每次手动 `chmod`;协作性质更强的团队账号可能反而用更宽松的 `umask 002`(让组内成员新建文件默认组内可写)。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

WORKDIR=$(mktemp -d /tmp/rhcsa04_umask.XXXXXX); cd "$WORKDIR"
old_umask=$(umask)

umask 022
touch umask_test_file.txt
mkdir umask_test_dir
assert_eq "$(stat -c '%a' umask_test_file.txt)" "644"    # 666 - 022 = 644
assert_eq "$(stat -c '%a' umask_test_dir)" "755"          # 777 - 022 = 755

umask 077
touch umask_strict_file.txt
assert_eq "$(stat -c '%a' umask_strict_file.txt)" "600"   # 666 - 077 = 600,组和其他人完全没权限

umask "$old_umask"
cd /; rm -rf "$WORKDIR"
```
本机实测:三个断言均输出 `OK`。

**常见坑:** umask 对文件的"屏蔽"永远不会赋予执行位——即便 `umask 000`(完全不屏蔽任何权限),新建的**文件**默认也只会是 666(rw-rw-rw-),不会是 777,因为文件的默认起点本身就没有 x 位;这是刻意的安全设计,防止刚创建的普通文本文件意外变成"可执行"状态,可执行权限必须显式 `chmod +x` 才能加上。

---

## 7. ACL 访问控制列表(`setfacl`/`getfacl`,mask 的坑)

**命令/配置:**
```bash
setfacl -m u:username:rwx file    # 给特定用户设置ACL权限
setfacl -m g:groupname:rx file     # 给特定组设置ACL权限
getfacl file                        # 查看完整ACL列表
setfacl -x u:username file          # 删除某个用户的ACL条目
setfacl -b file                      # 清空所有ACL,回到纯标准ugo权限
```

**一句话是什么:** 标准 ugo/rwx 权限模型只能表达"owner"、"所属组"、"其他所有人"这三类身份,ACL(Access Control List)在此基础上扩展出"精确针对某个特定用户/特定组"单独授权的能力,不需要为了给一个人开权限就去改动整个文件的 owner 或 group。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"创建配置 ACL";现实中经常遇到"这个文件本来是 A 的,但需要临时让 B 也能读"这类需求,标准权限模型处理不了(除非把 B 加入某个共享的组,这个改动范围更大、更粗糙),ACL 是精确处理这类需求的标准工具。

**从最容易犯错的做法讲起:** 设置了 ACL 之后,权限却"看起来没生效"——最常见的根因是 **ACL mask** 的限制:mask 定义了"ACL 条目实际能生效的权限上限",即便某个用户的 ACL 条目写的是 `rwx`,如果 mask 只有 `r--`,这个用户实际拿到的权限也会被 mask 收窄到 `r--`;新手往往只看自己设置的那条 ACL 权限,没意识到 mask 这层"隐藏的天花板"。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 一份合同文件本来只属于财务部门(标准权限只对 owner 和财务组开放),现在需要让审计部门的某个特定审计员临时也能只读查看,不应该把这个审计员加进财务组(权限范围过大),而是 `setfacl -m u:auditor_username:r-- contract.pdf` 精确只给这一个人开只读权限,不影响其他任何人的权限设置。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

WORKDIR=$(mktemp -d /tmp/rhcsa04_acl.XXXXXX); cd "$WORKDIR"
useradd -M -s /sbin/nologin rhcsa_acl_user 2>/dev/null

touch acl_test.txt
chmod 640 acl_test.txt    # 标准权限下,other完全没有读权限

setfacl -m u:rhcsa_acl_user:r acl_test.txt
getfacl --omit-header acl_test.txt 2>/dev/null | grep "user:rhcsa_acl_user" | grep -q "r--" && echo "OK: ACL精确授权给特定用户读权限,不改标准ugo"

# mask的坑:mask会限制ACL条目实际生效的上限
setfacl -m mask::r-- acl_test.txt
getfacl --omit-header acl_test.txt 2>/dev/null | grep "^mask" | grep -q "r--" && echo "OK: mask显式设置为r--,是ACL条目生效的天花板"

cd /; rm -rf "$WORKDIR"
userdel rhcsa_acl_user 2>/dev/null
```
本机实测:两个断言均输出 `OK`。

**常见坑:** `ls -l` 只会在权限位末尾显示一个不起眼的 `+` 号来提示"这个文件还有 ACL"(比如 `-rw-r-----+`),不会展开具体内容——排查权限问题时如果只看 `ls -l` 的常规 ugo 那几位、没注意到这个 `+` 号,很容易误判"权限应该是这样但实际表现不一致",一定要养成看到 `+` 号就追加 `getfacl` 确认完整情况的习惯。

---

## 8. 默认 ACL(目录继承)

**命令/配置:**
```bash
setfacl -d -m u:username:rx /shared/dir    # 设置目录的"默认ACL"
getfacl /shared/dir                          # 会同时显示"访问ACL"和"默认ACL"两组
```

**一句话是什么:** 普通 ACL(第 7 节讲的)只对设置时已经存在的文件生效,"默认 ACL"是专门设置在**目录**上的一种特殊 ACL,效果是"这个目录里以后新建的所有文件/子目录,自动继承这份 ACL 设置",解决"每次新建文件都要重新手动 `setfacl` 一遍"的麻烦。

**为什么 RHCSA 真考 / 生产会用到:** 团队共享目录场景下,新成员/新文件不断产生,不可能每次都手动补 ACL,默认 ACL 让权限策略能自动、持续地应用到未来新增的内容上,是可持续的团队协作权限方案,和第 5 节的 SGID(继承组)是互补关系(一个管组归属,一个管精确的用户/组权限)。

**从最容易犯错的做法讲起:** 混淆"默认 ACL"和"访问 ACL"——只对目录本身设置了普通 ACL(`setfacl -m`,没有 `-d`),以为目录下新文件也会自动带上这个权限,结果发现每次都要重新设置;`-d` 这个选项是"默认 ACL"和"访问 ACL"的唯一区别,决定了这份权限设置是"仅对当前这个目录本身生效"还是"作为模板持续应用到未来新增内容"。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 项目共享目录 `setfacl -d -m g:devteam:rwx /srv/project`,之后不管谁在这个目录下新建任何文件/子目录,`devteam` 组都自动拥有读写执行权限,不需要每次手动补;新建的子目录本身也会继续携带这份"默认 ACL"设置,继续对更深层新建的内容生效(默认 ACL 沿着目录树递归传递)。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

WORKDIR=$(mktemp -d /tmp/rhcsa04_dacl.XXXXXX); cd "$WORKDIR"
useradd -M -s /sbin/nologin rhcsa_acl_user 2>/dev/null

mkdir default_acl_dir
setfacl -d -m u:rhcsa_acl_user:rx default_acl_dir
touch default_acl_dir/inherited_file.txt

inherited_acl=$(getfacl --omit-header default_acl_dir/inherited_file.txt 2>/dev/null | grep "user:rhcsa_acl_user")
assert_ok test -n "$inherited_acl"    # 新建文件自动继承了目录的默认ACL,不需要手动重新setfacl

cd /; rm -rf "$WORKDIR"
userdel rhcsa_acl_user 2>/dev/null
```
本机实测:断言输出 `OK`。

**常见坑:** 默认 ACL 只对**新建**的内容生效(和 SGID 的组继承逻辑一样),给一个已经运营了一段时间、里面已有大量历史文件的目录补设默认 ACL,不会追溯影响已存在的文件——如果需要历史文件也统一应用新的 ACL 策略,要额外用 `setfacl -R -m ...` 递归地对已有内容显式补一遍。

---

## 9. 排查权限问题的系统方法(`namei -l`,从根逐级检查)

**命令/配置:**
```bash
namei -l /path/to/file    # 逐级列出路径上每一层的权限,从根目录开始
```

**一句话是什么:** 想访问一个文件,不仅这个文件本身的权限要够,**路径上每一级目录**都必须有对应的"通过权限"(目录的 x 位,即"搜索权限")——`namei -l` 一次性把整条路径每一层的权限都列出来,不需要一层层手动 `ls -ld` 查,是排查"明明文件权限看着没问题却访问不了"这类问题最高效的工具。

**为什么 RHCSA 真考 / 生产会用到:** "诊断修复权限问题"是 RHCSA 明确列出的技能;真实排障中,权限问题的根因经常不在文件本身,而在路径中间某一层目录缺少 x 权限,只看目标文件的权限会陷入"我看权限明明是对的"的困惑,系统性地检查整条路径才是正确方法论。

**从最容易犯错的做法讲起:** 遇到"权限被拒绝"(Permission denied),第一反应是反复检查目标文件本身的权限,却忽略了路径上的父目录——一个文件权限是 `777`(所有人任意读写执行)也完全没用,只要它所在的目录对某个用户没有 x(搜索)权限,这个用户依然完全无法访问,因为连"进入这个目录去找到这个文件"这一步都过不去。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 用户反馈"我明明对这个文件有权限,但就是打不开",用 `namei -l /home/otheruser/private/data.txt` 一次性看到整条路径每一级的权限,可能发现问题出在 `/home/otheruser` 这一级目录权限是 `700`(只有 owner 自己能进),而不是 `data.txt` 本身的权限有问题——这种"中间路径挡路"的情况,只查目标文件权限永远查不出来。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

WORKDIR=$(mktemp -d /tmp/rhcsa04_namei.XXXXXX); cd "$WORKDIR"
mkdir -p namei_test/sub1/sub2
touch namei_test/sub1/sub2/target.txt

namei_output=$(namei -l namei_test/sub1/sub2/target.txt 2>&1)
line_count=$(echo "$namei_output" | wc -l)
assert_ok test "$line_count" -ge 4    # 至少展示了路径上每一级(namei_test/sub1/sub2/target.txt)的权限行

cd /; rm -rf "$WORKDIR"
```
本机实测:断言输出 `OK`。

**常见坑:** 目录的 `x` 权限和 `r` 权限是两件不同的事,容易被混为一谈——目录有 `x`(搜索权限)没 `r`(列出内容权限),意味着"如果你已经知道文件的确切名字,可以直接访问它,但不能 `ls` 这个目录看到里面有什么";反过来有 `r` 没 `x`,能看到文件名列表但没法真正进入访问任何一个——这个组合经常被用来实现"目录列表可见但内容受限"或者相反的精细控制场景。

---

## 10. LUKS 磁盘加密(`cryptsetup luksFormat`/`luksOpen`)

**命令/配置:**
```bash
cryptsetup luksFormat /dev/sdX       # 初始化LUKS加密,会覆盖设备上原有数据,需要设置密码
cryptsetup luksOpen /dev/sdX name      # 用密码解锁,解锁后在 /dev/mapper/name 出现一个可用的明文映射设备
cryptsetup luksClose name               # 关闭映射,设备重新变回加密状态,不可直接使用
cryptsetup isLuks /dev/sdX               # 检测某个设备是否是LUKS加密卷
```

**一句话是什么:** LUKS(Linux Unified Key Setup)是 Linux 标准的块设备级全盘加密方案——加密和解密发生在块设备这一层,上层的文件系统(ext4/xfs 等)对加密完全无感知,你在加密卷上创建文件系统、挂载、读写文件,操作方式和普通盘完全一样,只是这块盘在没有正确密码解锁前,里面的数据对任何人(包括拿到物理硬盘的人)都是不可读的乱码。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"创建和配置 LUKS 加密卷";丢失的笔记本电脑、被物理窃取的硬盘,如果没有加密,数据裸奔;LUKS 加密是数据防泄露的基础防线,尤其对移动设备和可能被物理接触的服务器更是刚需。

**从最容易犯错的做法讲起:** 忘记 `luksFormat` 是**破坏性操作**——这条命令会覆盖目标设备上原有的所有数据(包括原来的分区表/文件系统),在真实生产环境的已有数据盘上误操作执行这条命令,数据基本无法恢复;操作前务必反复确认目标设备路径是对的,不是"大概是这个盘"这种模糊确认。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 给存放客户敏感数据的备份盘配置加密:`cryptsetup luksFormat` 初始化、`luksOpen` 解锁得到 `/dev/mapper/backup_crypt`、在这个映射设备上正常 `mkfs`+`mount`+ 存数据;需要长期自动挂载(不想每次开机手动输密码)的场景,搭配密钥文件(而不是交互式密码)+ `/etc/crypttab` 配置开机自动解锁,这是比手动 `luksOpen` 更进阶的生产用法。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

dd if=/dev/zero of=/tmp/rhcsa04_disk2.img bs=1M count=100 2>/dev/null
loop2=$(losetup -f); losetup "$loop2" /tmp/rhcsa04_disk2.img

echo -n "TestPassphrase123" | cryptsetup luksFormat "$loop2" -q --batch-mode
assert_eq "$(cryptsetup isLuks "$loop2"; echo $?)" "0"

echo -n "TestPassphrase123" | cryptsetup luksOpen "$loop2" rhcsa04_luks_test -
assert_ok test -e /dev/mapper/rhcsa04_luks_test

mkfs.ext4 -q /dev/mapper/rhcsa04_luks_test
mkdir -p /mnt/rhcsa04_luks
mount /dev/mapper/rhcsa04_luks_test /mnt/rhcsa04_luks
echo "encrypted data" > /mnt/rhcsa04_luks/secret.txt
assert_eq "$(cat /mnt/rhcsa04_luks/secret.txt)" "encrypted data"

umount /mnt/rhcsa04_luks
cryptsetup luksClose rhcsa04_luks_test

# 错误密码必须被拒绝(注意:退出码是2,不是常见的1,别凭直觉假设)
wrong_pw_exit=$(echo -n "WrongPassword" | cryptsetup luksOpen "$loop2" rhcsa04_luks_test2 - 2>/dev/null; echo $?)
assert_eq "$wrong_pw_exit" "2"

losetup -d "$loop2"; rm -f /tmp/rhcsa04_disk2.img; rmdir /mnt/rhcsa04_luks
```
本机实测:全部断言输出 `OK`。

**常见坑:** `cryptsetup luksOpen` 密码错误时的退出码是 **2**,不是大多数命令行工具"失败就是 1"的常见惯例——本机现场实测确认;写自动化脚本判断"密码是否正确"时,不能凭经验想当然地只判断"非 0 即失败"就够了,涉及区分具体失败原因时,退出码的具体数值需要现查文档或实测,不能靠通用惯例猜。

---

## 11. NFS 客户端挂载(`showmount`,`/etc/fstab` 里的 NFS 条目)

**命令/配置:**
```bash
showmount -e server_host        # 查看远程NFS服务器导出了哪些共享
mount -t nfs server:/export/path /mnt/point    # 挂载远程NFS共享
# /etc/fstab里的NFS条目:
# server:/export/path  /mnt/point  nfs  defaults  0 0
```

**一句话是什么:** NFS(Network File System)让你像访问本地文件一样访问远程服务器上的目录;客户端要做的事情很简单——先用 `showmount -e` 看服务器开放了哪些共享(不需要提前知道),再用 `mount -t nfs` 挂载,挂载之后的读写体验和本地文件系统几乎没有区别(只是多了网络这一层)。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"挂载和卸载网络文件系统";多台服务器共享同一份数据(配置文件、共享存储、用户主目录集中管理)是企业环境的常见架构,NFS 是最经典、最广泛支持的方案之一。

**从最容易犯错的做法讲起:** 直接盲猜或者去问别人"共享路径是什么"就尝试挂载,挂载失败了却搞不清是路径错了还是权限问题——正确的第一步永远是先用 `showmount -e 服务器地址` 确认服务器到底导出了哪些共享、路径怎么写,这是免费的、不需要权限的信息查询步骤,能排除大部分"到底该挂哪个路径"的低级错误。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 多台 Web 服务器需要共享同一份用户上传的媒体文件目录,存储服务器上用 `/etc/exports` 导出一个共享目录,各 Web 服务器分别 `mount -t nfs storage-server:/srv/media /var/www/media`,任何一台服务器写入的新文件,其他服务器立刻能看到,不需要额外的文件同步机制。

**可运行例子(本机同时充当 server 和 client,用 localhost 演示完整挂载流程):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

mkdir -p /srv/rhcsa_nfs_export
echo "shared via nfs" > /srv/rhcsa_nfs_export/shared.txt
echo "/srv/rhcsa_nfs_export *(rw,sync,no_subtree_check,no_root_squash)" > /etc/exports
systemctl enable --now nfs-server >/dev/null 2>&1
sleep 1
exportfs -ra

showmount_result=$(showmount -e localhost 2>&1)
echo "$showmount_result" | grep -q "rhcsa_nfs_export" && echo "OK: showmount -e 能看到导出的共享,不需要提前知道路径"

mkdir -p /mnt/rhcsa_nfs_client
mount -t nfs localhost:/srv/rhcsa_nfs_export /mnt/rhcsa_nfs_client
assert_eq "$(mount | grep -c "rhcsa_nfs_client")" "1"
assert_eq "$(cat /mnt/rhcsa_nfs_client/shared.txt)" "shared via nfs"

umount /mnt/rhcsa_nfs_client; rmdir /mnt/rhcsa_nfs_client
rm -f /etc/exports; rm -rf /srv/rhcsa_nfs_export
systemctl stop nfs-server
```
本机实测:两个断言均输出 `OK`(本机同时是 NFS server 和 client,验证的是完整挂载链路的真实工作情况,不是伪造的假设)。

**常见坑:** `no_root_squash` 这个导出选项要谨慎使用——默认情况下(`root_squash`,这是安全默认值)客户端的 root 用户在挂载的 NFS 共享上会被降级成一个无特权账号(`nobody`),防止任何能拿到 root 权限的客户端都能对服务器共享目录为所欲为;`no_root_squash` 关掉这层保护,只应该在你完全信任所有能挂载这个共享的客户端时才使用,生产环境滥用这个选项是真实的安全隐患。

---

## 12. autofs 自动挂载配置

**命令/配置:**
```
# /etc/auto.master.d/xxx.autofs
/mount/base/point   /etc/auto.xxx_map
```
```
# /etc/auto.xxx_map (map文件)
subdir  -fstype=nfs  server:/export/path
```
```bash
systemctl enable --now autofs
```

**一句话是什么:** autofs 解决"NFS/其他网络文件系统不应该一直占着挂载状态"这个问题——传统 `/etc/fstab` 挂载是"开机就挂上,一直占用",autofs 则是"平时完全不挂载,只有真正有进程访问那个路径的那一刻,才自动触发挂载;闲置一段时间没人访问,又自动卸载",按需使用,减少不必要的资源占用和网络连接。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"配置 autofs 自动挂载";大型环境里可能有几十上百个 NFS 共享路径,如果全部用 `/etc/fstab` 一直挂着,开机时间和资源浪费都是问题,autofs 的按需挂载是标准的规模化解决方案。

**从最容易犯错的做法讲起:** 配置完 autofs,直接 `ls` 挂载点的父目录,发现"里面是空的",就以为配置失败了——这其实是 autofs 设计上的正常行为(平时确实没有真正挂载,"空"是预期状态),真正验证配置是否成功,要**访问具体的子路径**(比如 `cat /mount/base/point/subdir/somefile`),访问动作本身才会触发挂载;只看父目录是否"看起来有内容"是错误的验证方法。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 每个用户的主目录实际存放在中央存储服务器上,通过 autofs 配置成"用户登录、真正访问自己主目录的那一刻才挂载",不需要给每个用户账号在每台机器上都提前手动挂好,新增用户也不需要额外配置每台机器,是集中式用户主目录管理的标准方案。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

mkdir -p /srv/rhcsa_autofs_target
echo "autofs works" > /srv/rhcsa_autofs_target/marker.txt
echo "/mnt/rhcsa_auto /etc/auto.rhcsa_demo" > /etc/auto.master.d/rhcsa-demo.autofs
echo "mnt -fstype=none,bind :/srv/rhcsa_autofs_target" > /etc/auto.rhcsa_demo
systemctl restart autofs
sleep 1

# 关键验证点:访问之前,确实没有挂载(这是autofs的正常状态,不是bug)
before_access=$(mount | grep -c "rhcsa_auto/mnt")
assert_eq "$before_access" "0"

# 一访问子路径,autofs立刻自动触发挂载
content=$(cat /mnt/rhcsa_auto/mnt/marker.txt 2>&1)
assert_eq "$content" "autofs works"
sleep 1
after_access=$(mount | grep -c "rhcsa_auto/mnt")
assert_eq "$after_access" "1"    # 访问后确实触发了自动挂载

systemctl stop autofs
rm -f /etc/auto.master.d/rhcsa-demo.autofs /etc/auto.rhcsa_demo
rm -rf /srv/rhcsa_autofs_target
```
本机实测:三个断言均输出 `OK`,现场验证了"访问前无挂载、访问后自动触发"这个 autofs 核心行为。

**常见坑:** `/etc/auto.master.d/` 下的映射配置文件必须以 `.autofs` 结尾,systemd 的 autofs 服务只会扫描加载这个特定后缀的文件——文件名/后缀写错,配置不会报任何错误,只是安安静静地不生效,排查起来容易摸不着头脑,这也是本节最容易被忽视的一个格式细节。

---

## 13. VDO 虚拟数据优化基础概念(RHEL 存储去重压缩技术)

**命令/配置:**
```bash
lvcreate --type vdo -n LV_NAME -L 物理容量 -V 逻辑容量 VG_NAME    # RHEL 9+已整合进LVM,不再是独立vdo命令
vdostats --human-readable         # 查看VDO卷的空间/去重统计
vdocalculatesize --human-readable --physical-size=MB --logical-size=MB   # 规划阶段估算VDO的空间/内存开销
```

**一句话是什么:** VDO(Virtual Data Optimizer)是 RHEL 的块设备级数据精简技术,通过**去重**(发现并合并重复数据块)和**压缩**,让"逻辑容量"能远大于"物理容量"(比如物理 4GB 的盘,逻辑上呈现出 8GB 甚至更多可用空间),特别适合虚拟机镜像、备份这类天然存在大量重复数据的场景;**RHEL 9 开始 VDO 的管理方式已经整合进 LVM**,不再是独立的 `vdo create` 命令,而是 `lvcreate --type vdo`。

**为什么 RHCSA 真考 / 生产会用到:** VDO 是 RHEL 存储技术栈里相对进阶的一项,理解它的存在和基本工作方式,是应对"如何在有限物理存储上支撑更大逻辑容量"这类存储规划问题的知识储备,虚拟化/容器化平台的底层存储优化经常会用到。

**从最容易犯错的做法讲起:** **凭旧版本 RHEL(7/8 时代)的记忆,想当然地找独立的 `vdo` 命令行工具去创建卷**——本机实测证伪:RHEL 10 上 `vdo` 这个传统管理命令已经不存在,只剩下几个辅助工具(`vdostats`/`vdoformat`/`vdocalculatesize`/`vdoforcerebuild`),核心的创建操作已经完全并入 `lvcreate --type vdo`,和创建普通逻辑卷用的是同一套 LVM 命令体系,只是多了 `--type vdo` 这个类型声明。这提醒一个更普遍的教训:**RHEL 各版本之间同一个技术的管理接口会演进,不能凭旧知识直接套用,必须现场核实当前版本的实际情况**。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 虚拟化平台存放大量结构相似的虚拟机镜像(操作系统文件在不同虚拟机之间高度重复),用 VDO 卷承载这些镜像存储,实际能节省大量物理空间;备份服务器存放多个时间点的增量备份,重复数据块占比通常很高,同样是 VDO 的典型适用场景。

**可运行例子(诚实说明:VDO 的核心功能依赖 `dm-vdo` 这个内核模块,本机 WSL2 使用的是微软定制内核,没有编译进这个模块——这是内核层面的硬性限制,不是软件包缺失,也不是命令用法问题;以下验证软件包/工具确实齐全、`vdocalculatesize` 规划工具能真实运行,并如实现场诊断出内核限制本身作为教学内容):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

vdo_pkg_installed=0; rpm -q vdo >/dev/null 2>&1 && vdo_pkg_installed=1
assert_eq "$vdo_pkg_installed" "1"    # 软件包确实装了

# vdocalculatesize不需要真正的VDO设备,是纯粹的规划期计算工具,可以真实运行出结果
calc_output=$(vdocalculatesize --human-readable --physical-size=5120 --logical-size=8192 2>&1)
echo "$calc_output" | grep -q "UDS index size" && echo "OK: 真实计算出UDS索引大小(这也是VDO需要相对大空间的原因——索引本身就占好几百MB到几GB)"

# 如实诊断:确认核心内核模块缺失,而不是凭空断言"这里不能用"
dm_vdo_check=$(modprobe dm-vdo 2>&1; echo "EXIT:$?")
echo "$dm_vdo_check" | grep -q "EXIT:1" && echo "OK: 现场确认dm-vdo模块加载失败"
echo "$dm_vdo_check" | grep -qi "not found" && echo "OK: 报错明确是Module dm-vdo not found,即WSL2内核未编译此模块"

# lvcreate --type vdo 因此必然失败,但这验证的是"失败原因确实是内核限制",不是命令语法错误
truncate -s 5G /tmp/rhcsa04_vdo_probe.img
loop_vdo=$(losetup -f); losetup "$loop_vdo" /tmp/rhcsa04_vdo_probe.img
pvcreate -y "$loop_vdo" >/dev/null 2>&1
vgcreate rhcsa_vdo_probe_vg "$loop_vdo" >/dev/null 2>&1
vdo_create_output=$(lvcreate --type vdo -n probelv -L 4G -V 8G rhcsa_vdo_probe_vg 2>&1)
assert_eq "$?" "3"
echo "$vdo_create_output" | grep -qi "device-mapper target" && echo "OK: 报错确认是device-mapper目标缺失,不是命令语法/参数错误"

vgremove -y rhcsa_vdo_probe_vg >/dev/null 2>&1
pvremove -y "$loop_vdo" >/dev/null 2>&1
losetup -d "$loop_vdo"; rm -f /tmp/rhcsa04_vdo_probe.img
```
本机实测:全部检查点均输出 `OK`——软件包/工具链完整、`vdocalculatesize` 真实计算出结果(仅 5GB 物理空间就要占约 2.59G 的 UDS 索引,直观说明了为什么 VDO 卷通常需要相对较大的规划空间)、`dm-vdo` 内核模块缺失被现场确认,`lvcreate --type vdo` 因此报错但报错原因和命令本身语法无关。**真正创建可用的 VDO 逻辑卷这个核心操作,必须在真实 RHEL 内核环境(物理机/KVM 虚拟机)下验证,本环境受结构性限制无法完成这一步。**

**常见坑:**
1. 最大的坑就是本节反复强调的:**不要凭旧版本 RHEL 记忆去找独立的 `vdo` 命令**,RHEL 9+ 统一走 `lvcreate --type vdo`。
2. VDO 的"逻辑容量可以远大于物理容量"不是变魔术——如果实际写入的数据重复率/可压缩率不够高,逻辑空间用满之前物理空间可能先耗尽,导致写入失败,规划 VDO 容量必须对实际数据的重复/压缩特征有基本预估,不能盲目认为"设置了大的逻辑容量就一定能装下"。
3. VDO 需要相当的物理空间来存放去重索引(UDS index)本身,见上方 `vdocalculatesize` 的真实计算结果——小规模测试/实验环境很容易忘记预留这部分开销,导致"怎么算都不够用"的困惑。

---

*本篇完成:2026-07-11,13 个知识点。验证环境:Rocky Linux 10.2(WSL2)。全部代码块真实跑通验证,含多处现场发现并如实记录的真实差异:xfs 最小 300MB 门槛、cryptsetup 密码错误退出码是 2 而非 1、VDO 因 WSL2 内核缺 dm-vdo 模块无法创建真实卷(工具链/规划计算/失败诊断均已验证,核心创建功能受限已如实标注)。*
