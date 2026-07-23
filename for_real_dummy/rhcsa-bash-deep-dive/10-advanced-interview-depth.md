# 10 · 进阶深度追加:5 个真实故障排查链案例

> 总览见 [00-roadmap.md](00-roadmap.md)。这不是知识点,不计入统计。

## 为什么需要这篇追加内容

`01-09` 全部完成并自查通过之后,用户转达了一位有经验从业者的反馈:现有材料没有达到 **2026 年大厂技术二面** 的深度。这条反馈原本来自一次针对面试场景的真实调研(WebSearch 检索中国大厂真实面经、西方大厂真实面经、面试官视角的元讨论),核心发现是:真实的技术追问不是"正确性 → 复杂度 → 能不能优化"这一条线性链,而是至少沿着 **5 条独立轴线** 展开,并且经常在同一个问题里综合出现——规模递增、工程约束递增(并发/分布式)、方案批判迭代、决策依据追问、真实性验证。这套格式已经在 [dsa-deep-dive/20-advanced-interview-depth.md](../dsa-deep-dive/20-advanced-interview-depth.md) 落地验证过。

**但 [00-roadmap.md](00-roadmap.md) 第 22-32 行早就定过一条规矩:本系列不采用 torch/dsa 系列的"面试怎么问"环节——RHCSA 是纯上机操作考试,没有面试环节,"面试官问/候选人答"这套对话体框架不适用。** 这篇追加内容尊重这条决定,不引入任何面试对话体,而是把 5 条轴线的**精神**——同一个问题沿着规模、约束、方案缺陷、决策依据、真实性这几条线不断深挖、逼近更真实的复杂度——翻译成这条系列自己的语言:**多步骤真实故障排查链**。具体做法:

- 不写"面试官给约束/候选人方案 1/面试官指出缺陷",而是写"故障现象 → 排查动作 1 → 发现新线索 → 排查动作 2(升级)→ … → 根因定位 → 修复与验证"这种运维排障的自然语序。
- 延续 01-09 已经在用的"危险操作 vs 正确操作对比 + 判分点提示"这个核心结构(见 [00-roadmap.md](00-roadmap.md)"每个知识点的固定讲解结构"第 4 步),把它作为这篇追加内容里"逼近更深层复杂度"的主要手段——比如"这一步看起来解决了,但没做 A,下次同样的场景会打回原形""这个操作能达到效果,但不是 RHCSA 判分标准认可的方式"。
- 5 条轴线在这篇里对应的运维语言:
  - **规模递增轴** → "这个操作在 1 个用户/1 块盘下没事,更大规模、共享资源、批量场景下会先撞上什么"
  - **工程约束递增轴** → 这条系列没有集群/分布式素材,如实翻译成这个系列真正有的约束:**持久性**("重启后/换一次启动顺序还在不在生效")
  - **方案批判迭代轴** → 连续指出同一个"修复方案"的新缺陷,逼着换成更彻底的方案
  - **决策依据追问轴** → "为什么用这个工具/这个参数,不用另一个"
  - **真实性验证轴** → "你觉得这样配置就生效了?现场验证给我看,不要假设"

**组织原则**:下面 5 个案例,每个都明确标注建立在 01-09 哪个已有知识点之上,包含完整还原的故障排查链条(含根因定位和修复验证)和 1-2 段真实在 WSL2 Rocky Linux 里跑过的可运行例子。这是方法论范例,不是把 100 个知识点全部重写——读者应该能把同样的思路自己套用到任何一个已有知识点上练习深挖。

**范围声明**:5 个案例全部在本机 WSL2 Rocky Linux 10.2 环境里真实验证。验证过程中如实发现并记录了两处和已发布内容不完全一致的地方(quota 硬限制在本机内核上没有真正拦截超额写入、sudoers.d 片段权限的真实拒绝阈值是"是否 world-writable"而不是"必须精确等于 440")——这些差异会在对应案例的"常见坑"里如实说明,不假装没看到,也不代表最初的推荐做法是错的。凡是"真实重启后是否持久"这类需要真实重启才能验证的问题,一律如实标注这一层验证不到,不冒充已经验证过。

---

## 案例 1:LVM 紧急扩容连环告警——从"扩了但没生效"到"扩容正在吃别人的份额"(方案批判迭代轴 + 规模递增轴)

建立在 [03 类](03-storage-and-lvm.md) 第 7-9 节(`lvcreate`/`lvextend`+`resize2fs`/删除顺序)和第 11 节(磁盘配额)之上。

**故障排查链条完整还原:**

- **故障现象:** 服务器上有一个共享的卷组(VG),两个业务服务各自用一个逻辑卷(LV)存数据——服务 A 用 `data_lv`,服务 B 用 `other_lv`。监控告警:`data_lv` 的挂载点使用率到了 100%,服务 A 写入开始报错。
- **排查动作 1:** 值班的人第一反应是"紧急扩容",执行 `lvextend -L +50M /dev/vg/data_lv`,命令返回成功,回复"已处理"。
- **发现:** 几分钟后告警**没有解除**——`df` 看到的使用率纹丝不动。回头看 `lvextend` 的输出,LVM 层面确实说"逻辑卷已经变大",但文件系统层面(`df`/应用实际能写的空间)完全没有感知到这次扩容。
- **排查动作 2(升级):** 想起 [03 类第 8 节](03-storage-and-lvm.md)讲过 `lvextend` 和 `resize2fs`/`xfs_growfs` 必须成对执行,补上 `resize2fs`。这次 `df` 的使用率真的降下来了,告警解除。
- **发现(方案批判,继续深挖):** 正准备收工,多问一句:这个 VG 是两个服务共享的池子,`data_lv` 这次扩容占的是公共份额——如果服务 B(`other_lv`)紧接着也告警,这个池子还扩得动吗?查一下 `vgs` 的剩余空间,只剩几十 MB 了。
- **根因:** 这不是孤立事件——`data_lv` 上有一个持续增长、没人管的日志文件,这类"扩容 → 缓解 → 再告警"的循环过去已经发生过几次,每次都在吃同一个共享池子的份额。继续把"临时扩容"当成"解决问题",相当于每次都往同一个共享账户里超支,迟早会在某个服务身上把余额刷穿——这不是猜测,下面可运行例子(1/2)会现场验证这次扩容确实吃掉了 `other_lv` 下次能用的余量。
- **修复与验证:** 真正的修复不是"再扩一次",而是回到日志持续增长本身——给这个日志文件的写入方设置磁盘配额上限(呼应 [03 类第 11 节](03-storage-and-lvm.md)),让同样的失控增长下次撞上一个**当场可发现**的限额,而不是继续悄悄蚕食共享 VG 的公共份额(可运行例子 2/2,含本机验证到什么程度的如实说明)。

**可运行例子(1/2):共享 VG 连环扩容——"扩了但没生效"和"抢了别人的份额"都是真实复现,不是描述**

```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

dd if=/dev/zero of=/tmp/rhcsa10_pv1.img bs=1M count=300 2>/dev/null
dd if=/dev/zero of=/tmp/rhcsa10_pv2.img bs=1M count=300 2>/dev/null
loopA=$(losetup -f); losetup "$loopA" /tmp/rhcsa10_pv1.img
loopB=$(losetup -f); losetup "$loopB" /tmp/rhcsa10_pv2.img
pvcreate -y "$loopA" "$loopB" >/dev/null 2>&1
vgcreate rhcsa10_vg "$loopA" "$loopB" >/dev/null 2>&1

# 同一个VG池子里,服务A(data_lv)和服务B(other_lv)各自的LV——典型的"共享存储池"生产形态
lvcreate -y -n rhcsa10_data -L 250M rhcsa10_vg >/dev/null 2>&1
lvcreate -y -n rhcsa10_other -L 250M rhcsa10_vg >/dev/null 2>&1
mkfs.ext4 -q /dev/rhcsa10_vg/rhcsa10_data
mkfs.ext4 -q /dev/rhcsa10_vg/rhcsa10_other
mkdir -p /mnt/rhcsa10_data /mnt/rhcsa10_other
mount /dev/rhcsa10_vg/rhcsa10_data /mnt/rhcsa10_data
mount /dev/rhcsa10_vg/rhcsa10_other /mnt/rhcsa10_other
assert_eq "$(mount | grep -c rhcsa10_data)" "1"
assert_eq "$(mount | grep -c rhcsa10_other)" "1"

# 故障现象: data_lv(服务A)被一个持续增长的日志文件写满,告警触发
dd if=/dev/zero of=/mnt/rhcsa10_data/growing.log bs=1M count=215 2>/dev/null
data_usage_pct=$(df --output=pcent /mnt/rhcsa10_data | tail -1 | tr -d ' %')
assert_ok test "$data_usage_pct" -ge 99

# 排查动作1: 现场紧急扩容,只做了lvextend
lvextend -L +50M /dev/rhcsa10_vg/rhcsa10_data >/dev/null 2>&1
after_lvextend_only=$(df --output=pcent /mnt/rhcsa10_data | tail -1 | tr -d ' %')
assert_eq "$after_lvextend_only" "$data_usage_pct"    # LVM层面变大了,df看到的使用率纹丝不动——"扩了但没生效"

# 排查动作2(升级): 补上resize2fs
resize2fs /dev/rhcsa10_vg/rhcsa10_data >/dev/null 2>&1
after_resize2fs=$(df --output=pcent /mnt/rhcsa10_data | tail -1 | tr -d ' %')
assert_ok test "$after_resize2fs" -lt "$after_lvextend_only"    # 这一步之后才真正生效

# 深挖: 这个VG是两个服务共享的池子,这次扩容吃掉的是公共份额——服务B紧接着告警,还扩得动吗?
vg_free_after=$(vgs rhcsa10_vg --noheadings -o vg_free --units m 2>/dev/null | tr -d ' ')
lvextend -L +50M /dev/rhcsa10_vg/rhcsa10_other >/dev/null 2>&1
other_lvextend_exit=$?
assert_eq "$other_lvextend_exit" "5"    # 真实失败,不是"应该会不够"的猜测

echo "OK: VG扩容后剩余仅 $vg_free_after,服务B的+50M扩容请求被LVM拒绝(退出码5)——服务A这次扩容确实吃光了服务B下次能用的余量,不是理论推演。"

umount /mnt/rhcsa10_data; umount /mnt/rhcsa10_other
lvremove -y rhcsa10_vg/rhcsa10_data rhcsa10_vg/rhcsa10_other >/dev/null 2>&1
vgremove -y rhcsa10_vg >/dev/null 2>&1
pvremove -y "$loopA" "$loopB" >/dev/null 2>&1
losetup -d "$loopA"; losetup -d "$loopB"
rm -f /tmp/rhcsa10_pv1.img /tmp/rhcsa10_pv2.img
rmdir /mnt/rhcsa10_data /mnt/rhcsa10_other
```

本机实测:全部断言输出 `OK`。VG 扩容后剩余空间仅 36.00m,服务 B 的 `+50M` 扩容请求被 LVM 拒绝,退出码 5(和 [03 类第 9 节](03-storage-and-lvm.md)"顺序错误的 `pvremove`"是同一个退出码,LVM 对"操作被拒绝"这类状态似乎有统一的退出码惯例,但不是通用规范,遇到别的工具不能凭这次经验直接套用)。

**可运行例子(2/2):从"再扩一次"切换到"给增长设上限"——如实验证配置和记账,如实标注硬限制强制拦截这一层本机验证不到**

```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

dd if=/dev/zero of=/tmp/rhcsa10_qpv.img bs=1M count=300 2>/dev/null
loopq=$(losetup -f); losetup "$loopq" /tmp/rhcsa10_qpv.img
pvcreate -y "$loopq" >/dev/null 2>&1
vgcreate rhcsa10_qvg "$loopq" >/dev/null 2>&1
lvcreate -y -n rhcsa10_qlv -L 250M rhcsa10_qvg >/dev/null 2>&1
mkfs.ext4 -q /dev/rhcsa10_qvg/rhcsa10_qlv
tune2fs -O quota /dev/rhcsa10_qvg/rhcsa10_qlv >/dev/null 2>&1
mkdir -p /mnt/rhcsa10_qlv
mount /dev/rhcsa10_qvg/rhcsa10_qlv /mnt/rhcsa10_qlv
assert_eq "$(mount | grep -c rhcsa10_qlv)" "1"

# 根因修复的第一步: 不再"每次告警就临时扩容",而是给这个日志写入者配置一个硬上限
useradd -M -s /sbin/nologin rhcsa10_logwriter 2>/dev/null
mkdir -p /mnt/rhcsa10_qlv/logs
chown rhcsa10_logwriter /mnt/rhcsa10_qlv/logs
setquota -u rhcsa10_logwriter 20480 20480 0 0 /mnt/rhcsa10_qlv
quota_configured=$(quota -u rhcsa10_logwriter | grep -c "20480")
assert_ok test "$quota_configured" -ge 1    # 配额真实写入并可查询到,不是敲了命令就假设生效

# 真实性验证: 配置好了不等于生效了——复现同样的"持续增长的日志"行为,故意写超限,现场看结果
su rhcsa10_logwriter -s /bin/bash -c "dd if=/dev/zero of=/mnt/rhcsa10_qlv/logs/growing.log bs=1M count=50 conv=fsync" >/dev/null 2>&1

# 现场发现(如实记录,不假装): repquota的记账是真实、精确的——它确实"知道"这次写入超限了
over_flag_line=$(repquota /mnt/rhcsa10_qlv 2>/dev/null | grep rhcsa10_logwriter)
echo "$over_flag_line" | awk '{print $2}' | grep -q '+' && echo "OK: repquota记账精确识别出超限(标志位含+),accounting这一层完全真实可信"

# 如实标注: 本机WSL2内核(6.18.x-microsoft-standard-WSL2)上,记账正确但硬上限并没有真正拦下这次写入——
# 这是继08类SELinux enforcing效果之后,本系列第二处现场确认的"配置和记账层完全真实,
# 但最终强制效果在WSL2内核上未能复现"的情况
actual_written_mb=$(du -m /mnt/rhcsa10_qlv/logs/growing.log | cut -f1)
assert_eq "$actual_written_mb" "51"    # 如实验证: 50MB(fsync后约51MB)全部写入成功,20MB硬上限没有真正拦住这次写入
echo "本机如实记录: setquota配置和repquota/quota -u记账这两层都验证扎实真实生效,但'硬上限阻止写入'这个最终强制效果,在本机WSL2内核上未能复现——这一层需要真实RHEL内核环境验证,不能假装这里已经验证过。即便如此,'配置真实生效+超限被记账系统精确捕获'本身已经把'悄悄吃光共享VG'变成了'可监控、可告警'的状态,这才是本案例真正能在这台机器上兑现的改善。"

umount /mnt/rhcsa10_qlv
lvremove -y rhcsa10_qvg/rhcsa10_qlv >/dev/null 2>&1
vgremove -y rhcsa10_qvg >/dev/null 2>&1
pvremove -y "$loopq" >/dev/null 2>&1
losetup -d "$loopq"
rm -f /tmp/rhcsa10_qpv.img
rmdir /mnt/rhcsa10_qlv
for _ in 1 2 3 4 5; do userdel rhcsa10_logwriter 2>/dev/null && break; sleep 1; done
```

本机实测:全部断言输出 `OK`。`repquota` 精确显示 `rhcsa10_logwriter` 超限(标志位 `+`,已用 51202 块 vs 硬限 20480 块),但故意写入的 50MB 数据依然完整落盘,硬限制没有真正拦截写入——这一发现已经在案例内如实标注,不代表 `setquota`/`repquota` 这套工具本身不可靠。

**常见坑(含判分点提示):**
- `lvextend` 和 `resize2fs`/`xfs_growfs` 不成对执行,是 RHCSA 存储类大题最容易丢分的地方之一——判分脚本通常按 `df` 实际测出来的可用空间判定,不会因为执行过 `lvextend` 命令就给分。
- 只盯着眼前告警"扩容 → 解除",不去看这个 VG 是不是共享池子、剩余空间还够不够撑下一次,是"头痛医头"最典型的表现——`vgs` 剩余空间检查应该成为每次紧急扩容后的例行动作,不是可选项。
- **如实记录一处和直觉不符的真实发现**:本机(WSL2 内核 6.18.33.2-microsoft-standard-WSL2)上 `setquota`/`quota -u`/`repquota` 三层配置和记账全部真实生效、可查询,但故意写超过硬限制的数据依然完整写入成功——这是继 [08 类](08-security-selinux-firewall.md) SELinux enforcing 效果之后,本系列第二处确认的"配置和记账层完全真实,但最终强制效果在 WSL2 内核上未能复现"的情况,不能假装这里已经验证过硬限制会拦截写入,真实 RHEL 内核环境下的强制拦截行为需要另外验证。

---

## 案例 2:fstab "挂载成功"却挂错了盘——mount 不报错不等于配置对了(真实性验证轴)

建立在 [04 类第 2 节](04-filesystem-and-permissions.md)(`/etc/fstab` 语法与 UUID 挂载)之上。

**故障排查链条完整还原:**

- **故障现象:** 运维人员要把新接入的数据盘(`diskA`,里面是数据库文件)永久挂载到 `/mnt/rhcsa10_data2`,并在群里报告"配置好了,`/etc/fstab` 已经加好这条 UUID 记录"。
- **排查动作 1(naive verification):** 复核的人只做了最基本的检查——`mount -a`,没有报错;`mount | grep` 挂载点,确实出现在列表里。两项都通过,复核回复"确认无误"。
- **发现:** 但没人去看**挂载上来的内容对不对**。当时 `blkid` 屏幕上同时列着两块新盘(`diskA` 和无关的 `diskB`)的信息,复制 UUID 时手滑复制成了 `diskB` 那一行——`diskB` 恰好也是一块格式化好的、合法的 ext4 分区,所以从 `mount` 的角度看,这是一次完全"正常"的挂载,没有任何理由报错。
- **排查动作 2(真实性验证,不信任"没报错"):** 直接读挂载点里的文件内容,而不是只看 `mount` 命令的返回状态——内容对不上:读出来的是 `diskB` 的内容,不是数据库该有的数据。
- **发现(逼问"标准防线"够不够):** [04 类第 2 节](04-filesystem-and-permissions.md)已经确立"改完 fstab 先 `mount -fav` 预检"是标准动作,追问一句:这次的错误,`mount -fav` 能不能提前拦住?现场验证——**拦不住**,`mount -fav` 只校验"这条 UUID 存在、文件系统类型对、挂载点存在"这类语法/可行性层面的东西,不知道、也没办法知道"你脑子里以为挂的是哪块盘"。这类语义错误,任何自动化预检工具都测不出来,只有真的去读内容才行。
- **根因:** UUID 本身没有任何语义标签,它只是一串十六进制数字,和"这块盘该装什么数据"这件事没有任何联系——`blkid`/`mount`/`mount -fav` 全部只能验证"这个 UUID 存在且能正常挂载",没有任何一层工具能替你确认"这是不是你想要的那块盘"。
- **修复与验证:** 改成正确的 UUID(`diskA` 的),重新 `mount -a`,这次不再只满足于"挂载点出现了",而是重新读一遍内容确认真的是预期数据,才算真正完成。

**可运行例子:两块合法的盘,一次"完全不报错"的挂错——现场验证内容,不是只看 mount 输出**

```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

dd if=/dev/zero of=/tmp/rhcsa10_diskA.img bs=1M count=100 2>/dev/null
dd if=/dev/zero of=/tmp/rhcsa10_diskB.img bs=1M count=100 2>/dev/null
loopA=$(losetup -f); losetup "$loopA" /tmp/rhcsa10_diskA.img
loopB=$(losetup -f); losetup "$loopB" /tmp/rhcsa10_diskB.img
mkfs.ext4 -q "$loopA"
mkfs.ext4 -q "$loopB"

mkdir -p /mnt/rhcsa10_probeA /mnt/rhcsa10_probeB /mnt/rhcsa10_data2
mount "$loopA" /mnt/rhcsa10_probeA
mount "$loopB" /mnt/rhcsa10_probeB
echo "diskA: production database volume" > /mnt/rhcsa10_probeA/marker.txt
echo "diskB: unrelated scratch volume" > /mnt/rhcsa10_probeB/marker.txt
umount /mnt/rhcsa10_probeA
umount /mnt/rhcsa10_probeB

uuidA=$(blkid -o value -s UUID "$loopA")
uuidB=$(blkid -o value -s UUID "$loopB")

# 故障现象: 要把diskA(数据库卷)写进/etc/fstab永久挂载到/mnt/rhcsa10_data2,
# 但当时屏幕上同时有两块新盘的blkid信息,复制UUID时手滑复制成了diskB那一行
echo "UUID=$uuidB /mnt/rhcsa10_data2 ext4 defaults 0 0" >> /etc/fstab

# 排查动作1(naive verification): 只看mount -a报不报错、挂载点出不出现
mount_a_exit=0
mount -a || mount_a_exit=$?
assert_eq "$mount_a_exit" "0"
assert_eq "$(mount | grep -c /mnt/rhcsa10_data2)" "1"

# 真实性验证轴: "mount不报错"就等于"挂对了盘"?现场读内容
actual_content=$(cat /mnt/rhcsa10_data2/marker.txt)
assert_eq "$actual_content" "diskB: unrelated scratch volume"

# mount -fav(04类已确立的标准预检手段) 能不能提前拦住这个错误?
umount /mnt/rhcsa10_data2
fav_output=$(mount -fav 2>&1)
echo "$fav_output" | grep -q "rhcsa10_data2" && echo "OK: mount -fav 对这条fstab语法/可行性校验通过——但通过不代表挂的是对的盘"

# 修复: 改成正确的UUID,重新mount -a,现场验证内容这次真的对了
sed -i "\|rhcsa10_data2|d" /etc/fstab
echo "UUID=$uuidA /mnt/rhcsa10_data2 ext4 defaults 0 0" >> /etc/fstab
mount -a
fixed_content=$(cat /mnt/rhcsa10_data2/marker.txt)
assert_eq "$fixed_content" "diskA: production database volume"

umount /mnt/rhcsa10_data2
sed -i "\|rhcsa10_data2|d" /etc/fstab
losetup -d "$loopA"; losetup -d "$loopB"
rm -f /tmp/rhcsa10_diskA.img /tmp/rhcsa10_diskB.img
rmdir /mnt/rhcsa10_data2 /mnt/rhcsa10_probeA /mnt/rhcsa10_probeB
```

本机实测:全部断言输出 `OK`。挂载"成功"且读到的确实是 `diskB` 的内容(错误的那块盘),`mount -fav` 确认对这条错误条目校验通过(证明预检拦不住语义错误),改成正确 UUID 后重新验证内容,这次才是真的对了。

**常见坑(含判分点提示):**
- RHCSA 判分脚本大概率也是按"挂载点下的内容/属性对不对"来判分,不是只看 `/etc/fstab` 格式对不对或者 `mount -a` 报不报错——这条案例演示的正是"语法对、命令不报错"和"结果对"之间可能存在的落差。
- 这个案例没有、也没必要真正触发一次内核级的开机失败来验证"改坏 fstab 会掉进 emergency shell"——[02 类第 2 节](02-process-and-boot.md)已经如实说明 WSL 环境没有真实 GRUB,无法安全复现"编辑内核参数进 rescue 模式"这条路径;本案例刻意选择完全不依赖真实重启的验证方式(`mount -a`/`mount -fav` 模拟"开机时会发生什么"),这不是环境限制下的将就,而是 RHCSA 真实考试和生产环境里本来就该有的纪律——真实重启验证代价高、速度慢,`mount -fav` 这类本地预检永远应该走在"赌一把重启"前面,不管有没有能力做真实重启测试。
- 复制粘贴 UUID 是极容易发生的真实失误,尤其是同一屏 `blkid`/`lsblk -f` 输出里同时列着好几块新盘的时候——写自动化脚本生成 fstab 条目时,把"设备到 UUID 的映射"作为变量传递,而不是靠肉眼在一屏文字里数第几行,能从根本上消除这类错误的发生概率。

---

## 案例 3:sudo 规则"看着对"却完全不生效——最小权限的两次收紧(决策依据追问轴)

建立在 [06 类第 10 节](06-users-and-groups.md)(sudo 配置,`visudo`,`/etc/sudoers.d/`)之上。

**故障排查链条完整还原:**

- **故障现象:** 需要给一个初级运维账号开一条精确权限——只允许他免密执行 `systemctl status sshd` 这一条命令,不多不少。运维人员在 `/etc/sudoers.d/` 下新建了一个片段文件,内容写得完全正确;后来因为要临时找同事一起核对这条规则,顺手把这个文件 `chmod 666` 方便协作编辑,改完忘了改回去。
- **排查动作 1(naive verification):** 复核的人只检查了两件事:`visudo -c` 语法检查通过;直接打开文件看内容,规则写得一字不差。两项都合格,判断"配置没问题"。
- **发现(真实性验证,不信任语法检查):** 不看文件、不看语法,直接切到那个初级运维账号本人,现场跑一遍 `sudo -n systemctl status sshd`——被拒绝,要求输密码。规则内容完全正确,但对这个账号完全不生效。
- **排查动作 2(升级,定位真正原因):** 回头检查文件权限,发现是 `666`——sudo 出于安全考虑,拒绝加载任何"其他用户可写"的 sudoers 片段(这类文件一旦被 owner 之外的人写入,相当于任何人都能给自己加无限权限,是明摆着的提权漏洞,sudo 直接整份忽略,连警告都不会出现在这条 `sudo -n` 命令自己的报错里,只有回头查权限位才能定位)。收紧到 `440` 之后,同一条规则、同一个命令,立刻生效。
- **深挖(方案批判 + 决策依据):** 权限问题解决之后,追问一句决策依据:当初为什么要精确到 `systemctl status sshd` 这一条命令,而不是图省事写成 `ALL=(ALL) NOPASSWD: ALL`?现场把这条"图省事"的方案跑一遍:一旦这么写,这个"只该重启/查看 sshd 状态"的账号,立刻能用 `sudo cat /etc/shadow` 读到全系统的密码哈希——一个具体、可复现的权限扩大风险,不是"理论上不安全"这种空话。
- **根因:** 两层问题分别独立存在——权限位错误导致规则完全不生效(表面症状是"配置了却没用"),范围过宽则是另一类完全不同的风险(配置生效了,但生效的范围超出了本来的意图)。两者都需要各自的收紧动作,解决了第一层不代表第二层也自动没事。
- **修复与验证:** 精确路径 + 正确权限(`440`)两者都到位,以该账号本人身份现场验证两件事:该做的事(`systemctl status sshd`)能做,不该做的事(读 `/etc/shadow`)做不了。

**可运行例子:同一条规则,两次收紧——权限位从 666 到 440,授权范围从 ALL 到精确路径**

```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

useradd -m rhcsa10_sudouser 2>/dev/null

# 排查动作0: 精确到单条命令的sudoers.d片段,内容完全正确
echo "rhcsa10_sudouser ALL=(ALL) NOPASSWD: /usr/bin/systemctl status sshd" > /etc/sudoers.d/rhcsa10-demo
# 后来为了让同事临时也能协作编辑这份配置,顺手chmod 666,改完忘了改回去
chmod 666 /etc/sudoers.d/rhcsa10-demo

visudo_ok=$(visudo -c 2>&1)
echo "$visudo_ok" | grep -qi "parsed OK" && echo "OK: visudo -c 语法检查通过——规则内容完全正确,语法这一层看不出任何问题"

# 真实性验证轴: 不看文件内容/语法,直接以这个用户的身份现场测试
su rhcsa10_sudouser -c "sudo -n /usr/bin/systemctl status sshd" >/tmp/rhcsa10_sudo1.log 2>&1
denied_exit=$?
assert_ok test "$denied_exit" -ne 0    # 现场测试:规则完全不生效
grep -qi "sorry\|password" /tmp/rhcsa10_sudo1.log && echo "OK: 现场以用户身份测试才发现规则没生效——sudo拒绝了,但原因不写在这条报错里,只有回头查权限位才能定位"

# 排查动作1(升级): 收紧到440——这也顺带修复了"666本身就是更严重的问题"(任何人都能改这份授权规则本身)
chmod 440 /etc/sudoers.d/rhcsa10-demo
su rhcsa10_sudouser -c "sudo -n /usr/bin/systemctl status sshd" >/tmp/rhcsa10_sudo2.log 2>&1
assert_eq "$?" "0"

# 深挖(方案批判+决策依据): 图省事写成 ALL=(ALL) NOPASSWD: ALL 会怎样?现场验证真实风险,不是空谈
echo "rhcsa10_sudouser ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/rhcsa10-demo-broad
chmod 440 /etc/sudoers.d/rhcsa10-demo-broad
su rhcsa10_sudouser -c "sudo -n cat /etc/shadow" >/tmp/rhcsa10_sudo3.log 2>&1
broad_exit=$?
assert_eq "$broad_exit" "0"    # 真实验证: 一个"只该重启sshd"的账号,现在能读/etc/shadow——具体、可复现的风险,不是"理论上不安全"
rm -f /etc/sudoers.d/rhcsa10-demo-broad

# 最终状态: 精确路径+正确权限——该做的事能做,不该做的事做不了
su rhcsa10_sudouser -c "sudo -n /usr/bin/systemctl status sshd" >/tmp/rhcsa10_sudo4.log 2>&1
assert_eq "$?" "0"
su rhcsa10_sudouser -c "sudo -n cat /etc/shadow" >/tmp/rhcsa10_sudo5.log 2>&1
final_denied=$?
assert_ok test "$final_denied" -ne 0

rm -f /etc/sudoers.d/rhcsa10-demo /tmp/rhcsa10_sudo*.log
for _ in 1 2 3 4 5; do userdel -r rhcsa10_sudouser 2>/dev/null && break; sleep 1; done
```

本机实测:全部断言输出 `OK`。`666` 权限下规则完全不生效(sudo 静默拒绝),收紧到 `440` 后立即生效;`ALL=(ALL) NOPASSWD: ALL` 这条对照组现场验证了"能读 `/etc/shadow`"这一具体风险,精确路径版本则正确拒绝了这个操作。

**常见坑(含判分点提示):**
- RHCSA 判分"sudo 权限配置"类题目,大概率是**以目标账号身份实际执行一遍命令**来判分,不是检查 `/etc/sudoers.d/` 里的文本内容——文件内容对、权限位错,在判分脚本眼里就是"没做对",这条案例演示的正是这种落差。
- **如实记录一处和 [06 类](06-users-and-groups.md)已发布内容不完全一致的地方**:本机(`sudo 1.9.17p2`)现场对多组权限位做了扫描测试,发现真正触发"整份忽略"的阈值是**其他用户是否具备写权限**——`644`/`640`/`400` 在本机实测里都能正常加载生效,只有 `666`/`777`(other 有写权限)会被拒绝。这和"必须精确等于 440,644 也不行"这种更严格的表述不完全一致;如果这里的结论和别处看到的说法有出入,以本案例现场实测的数值为准——这不代表 `440` 不再是推荐做法(它依然是最小、最清晰、无歧义的正确选择),只是"拒绝阈值具体是什么"这一点需要现场核实,不能凭刻板印象断言一个过于精确的数字。
- 精确到单条命令路径,还要留意命令参数是否也被锁定——`ALL=(ALL) NOPASSWD: /usr/bin/systemctl status sshd` 只放行这一个精确的参数组合,`systemctl status nginx`、`systemctl restart sshd` 都不在授权范围内,这是"精确路径"这个说法容易被误解的地方:锁的不只是命令本身,是命令加参数的完整字符串。

---

## 案例 4:改错静态 IP 导致远程会话失联——批量上线前的"退路"设计(规模递增轴)

建立在 [07 类第 2 节](07-networking.md)(`nmcli` 配置静态 IP)之上,修复手段借用 [02 类第 11 节](02-process-and-boot.md)(systemd timer)。

**故障排查链条完整还原:**

- **故障现象:** 一批服务器要做子网迁移,运维人员计划把网卡的静态 IP 从旧子网改到新子网。计划是:SSH 登录到服务器,直接用 `nmcli` 把网卡改成新子网的地址。
- **排查动作 1(critique 的直接对象):** 第一版方案就是"改完直接看行不行"——`nmcli connection up` 切到新配置,连不上了再想办法。
- **发现(真实性验证,逼问"退路"在哪里):** 追问一句:如果切换之后连不上了,你打算怎么办?"再改回来"——可是你已经断线了,改回来的命令要在哪里执行?答不上来。现场把这个场景真实复现一遍(在隔离的虚拟网卡对上,不影响真实网络):网卡从"管理员当前所在的子网"切到一个完全不同的新子网后,原来的地址已经不在这块网卡上了,管理员这一刻真的连不上了——不是猜测,是真实的 `ping` 失败(可运行例子 1/2)。
- **排查动作 2(升级,决策依据——为什么不是简单地用 `&` 丢后台一个定时回滚脚本):** 有人提议写个脚本,改之前先在后台起一个"N 秒后自动切回旧配置"的定时任务,没确认就自动回滚。但如果这个后台任务是当前 SSH 会话里用 `&` 丢到后台的子进程,它的生死跟这次会话绑在一起——如果这次改动恰好就是导致会话断开的原因,这个"安全网"会跟着会话一起死掉,起不到任何兜底作用。正确做法是用一个**独立于当前会话的系统级机制**——`systemd-run --on-active=N` 会让 systemd 当场生成一个 **transient unit**(瞬态单元——不是提前写在磁盘上的 `.service`/`.timer` 文件,而是运行时临时创建、直接交给 systemd 管理的单元,行为上和普通 unit 完全一样,同样受 systemd 管这一整套生命周期,只是没有对应的配置文件),这个单元由 PID 1(systemd 自己)管理,和发起它的 shell/会话生死无关(呼应 [02 类第 10-11 节](02-process-and-boot.md)的 cron/systemd timer)。
- **根因:** 风险的本质不是"IP 配错了"本身,而是**这次操作没有退路**——一旦断线,唯一的恢复手段是物理/带外控制台,对云主机可能意味着几个小时的工单排队。
- **深挖(规模递增):** 追问一步——如果这不是 1 台机器,是自动化脚本对 100 台机器做同样的批量迁移,这个"定时自动回滚"方案还够用吗?100 台机器同时因为同一个配置错误集体失联,是比 1 台机器失联严重得多的事故,而且"等所有机器各自的回滚定时器分别到期"本身就不是一个可控的恢复节奏。规模上去之后,正确的做法是先在 1-2 台机器上试点、确认没问题再批量铺开,而不是对着 100 台机器一次性提交同一个未经验证的变更。
- **修复与验证:** 武装好独立于会话的自动回滚定时器再提交变更;如果变更导致失联,定时器到期后自动恢复连通性;如果变更本身没问题,在定时器到期前主动确认取消,新配置正常保留、不会被误回滚——两种结局都要现场跑通(可运行例子 2/2)。

**插一句,veth 对是什么(和 [07 类](07-networking.md)的 dummy 网卡不是一回事,这里先说清楚区别,不然会疑惑这个案例为什么不直接复用 dummy):** `ip link add A type veth peer name B` 一次性创建出**一对**互相连着的虚拟网卡 A 和 B——可以把它想象成一根虚拟网线,线的一头插在 A 上、另一头插在 B 上,从 A 发出去的包会从 B 那头冒出来,反之亦然。这和 07 类"dummy 网卡是什么"里讲的 dummy 接口**不是同一种东西**:dummy 是孤零零一个、带 `NOARP` 标志、连 ARP 都不做的隔离沙盒,连不了任何"对端";veth 天生成对出现,现场用 `ip link show` 确认过,veth 接口**没有** `NOARP` 标志,支持完整的 ARP/二层通信,两端可以真的互相 `ping` 通。这正是本案例选 veth 而不是 dummy 的原因——这里要模拟"服务器网卡"和"管理员客户端"两台机器之间真实的可达性变化(连得上/连不上),dummy 那种连 ARP 都不具备的孤立设备根本模拟不出"两端互通"这件事,必须用 veth 这种成对、可双向通信的接口;两者的共同点仅仅是"都不对应真实硬件、都不会影响真实 `eth0`",安全隔离这一点是一致的,但内部机制完全不同。

**可运行例子(1/2):没有退路的直接切换——真实复现失联,不是描述"理论上会断"**

```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

# veth对模拟"服务器网卡(veth0)<->管理员客户端(veth1)"这条真实链路,可以做真正的可达性测试,
# 不影响真实eth0(veth是什么、和07类dummy网卡的区别见上方说明)
ip link add rhcsa10veth0 type veth peer name rhcsa10veth1
ip link set rhcsa10veth0 up
ip link set rhcsa10veth1 up
ip addr add 192.168.77.2/24 dev rhcsa10veth1    # 管理员客户端,固定不变

nmcli device set rhcsa10veth0 managed yes >/dev/null 2>&1
sleep 1
nmcli connection add type ethernet ifname rhcsa10veth0 con-name rhcsa10-safe \
    ipv4.method manual ipv4.addresses 192.168.77.1/24 >/dev/null 2>&1
nmcli connection up rhcsa10-safe >/dev/null 2>&1
sleep 1

# 前置: 管理员当前能连通服务器(模拟现在这条远程管理会话是通的)
assert_ok ping -c 2 -W 2 192.168.77.1

# 排查动作1(critique对象): 子网迁移,直接把网卡改到新子网,没有任何退路设计
nmcli connection add type ethernet ifname rhcsa10veth0 con-name rhcsa10-risky \
    ipv4.method manual ipv4.addresses 10.50.0.5/24 >/dev/null 2>&1
nmcli connection up rhcsa10-risky >/dev/null 2>&1
sleep 1

# 真实性验证: 旧地址是否还在,现场ping,不是"应该会断"的猜测
locked_out=1
ping -c 2 -W 2 192.168.77.1 >/dev/null 2>&1 || locked_out=0
assert_eq "$locked_out" "0"

# 此刻没有任何自动兜底机制——如果这是一次真实的远程SSH会话,管理员这一步已经把自己锁在外面了,
# 唯一的恢复手段是物理/带外控制台访问,新子网10.50.0.5/24管理员的客户端根本不在这个网段里,
# 连尝试连到新地址都做不到(不在同一个二层网络,不知道网关在哪)

nmcli connection down rhcsa10-risky >/dev/null 2>&1
nmcli connection delete rhcsa10-risky >/dev/null 2>&1
nmcli connection down rhcsa10-safe >/dev/null 2>&1
nmcli connection delete rhcsa10-safe >/dev/null 2>&1
ip link delete rhcsa10veth0 2>/dev/null
```

本机实测:切换前 `ping` 0% 丢包,切换后 `locked_out` 断言正确捕获到失联(`ping` 全部超时)。验证完毕后现场确认真实 `eth0`(`172.30.207.100/20`)和外网连通性(`ping 8.8.8.8`)完全没有受到任何影响。

**可运行例子(2/2):武装"确认才提交"的自动回滚安全网——两种结局都现场验证**

```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

ip link add rhcsa10veth0 type veth peer name rhcsa10veth1
ip link set rhcsa10veth0 up
ip link set rhcsa10veth1 up
ip addr add 192.168.77.2/24 dev rhcsa10veth1

nmcli device set rhcsa10veth0 managed yes >/dev/null 2>&1
sleep 1
nmcli connection add type ethernet ifname rhcsa10veth0 con-name rhcsa10-safe \
    ipv4.method manual ipv4.addresses 192.168.77.1/24 >/dev/null 2>&1
nmcli connection up rhcsa10-safe >/dev/null 2>&1
sleep 1
nmcli connection add type ethernet ifname rhcsa10veth0 con-name rhcsa10-risky \
    ipv4.method manual ipv4.addresses 10.50.0.5/24 >/dev/null 2>&1

# 排查动作2(升级修复,第一遍: 武装好"无确认则自动回滚"的安全网,再提交这次危险切换)
# 用systemd-run --on-active而不是简单地"&"丢后台: 这是一个独立于当前shell/会话的系统级transient unit,
# 不会随着"这次切换正好把当前SSH会话切断了"而一起消失——这正是它必须具备的"死人开关"属性
systemd-run --on-active=5s --unit=rhcsa10-revert bash -c "nmcli connection up rhcsa10-safe" >/dev/null 2>&1
revert_is_system_unit=$(systemctl show rhcsa10-revert.timer -p Unit --value 2>/dev/null)
assert_eq "$revert_is_system_unit" "rhcsa10-revert.service"

nmcli connection up rhcsa10-risky >/dev/null 2>&1
sleep 1
locked_out=1
ping -c 2 -W 2 192.168.77.1 >/dev/null 2>&1 || locked_out=0
assert_eq "$locked_out" "0"

# 真实性验证: 不做任何确认动作(模拟管理员被锁在门外,没法登录进来手动确认),等安全网自动兜底
recovered=0
for i in $(seq 1 40); do
    sleep 1
    ping -c 1 -W 1 192.168.77.1 >/dev/null 2>&1 && { recovered=1; break; }
done
assert_eq "$recovered" "1"
assert_ok ping -c 2 -W 2 192.168.77.1

# 排查动作2(升级修复,第二遍: 这次真的确认变更没问题,应该保留新配置,不应该被回滚)
systemctl reset-failed rhcsa10-revert.service 2>/dev/null
nmcli connection up rhcsa10-risky >/dev/null 2>&1
sleep 1
systemd-run --on-active=5s --unit=rhcsa10-revert2 bash -c "nmcli connection up rhcsa10-safe" >/dev/null 2>&1
sleep 1
# 管理员这次确认变更成功(模拟: 带外通道确认新配置符合预期),主动取消这次定时回滚
systemctl stop rhcsa10-revert2.timer >/dev/null 2>&1
cancelled_exit=$?
assert_eq "$cancelled_exit" "0"
sleep 6    # 等过原定的触发时刻,确认真的没有被回滚
still_on_new_subnet=$(nmcli -t -f IP4.ADDRESS device show rhcsa10veth0 | cut -d: -f2)
assert_eq "$still_on_new_subnet" "10.50.0.5/24"

nmcli connection down rhcsa10-risky >/dev/null 2>&1
nmcli connection delete rhcsa10-risky >/dev/null 2>&1
nmcli connection down rhcsa10-safe >/dev/null 2>&1
nmcli connection delete rhcsa10-safe >/dev/null 2>&1
ip link delete rhcsa10veth0 2>/dev/null
systemctl stop rhcsa10-revert.timer rhcsa10-revert2.timer 2>/dev/null
systemctl reset-failed rhcsa10-revert.service rhcsa10-revert2.service 2>/dev/null
```

本机实测:全部断言输出 `OK`。第一遍(不确认)在定时器到期后自动恢复连通性;第二遍(主动确认取消)新配置正确保留、没有被误回滚。验证完毕后现场确认真实 `eth0` 和外网连通性完全没有受到任何影响。

**常见坑(含判分点提示):**
- RHCSA 考试环境通常是单一控制台/虚拟机管理界面接入,断网后一般还有考试平台自带的控制台可以救场,这也是为什么很多考生没有养成"改网络前先想退路"的习惯——但这条纪律在真实生产环境(尤其是纯远程管理的云主机)上是刚需,不能因为考试环境宽容就忽略。
- `nmcli connection up` 切换到新 profile 的瞬间,旧 profile 会被**顶替下线**,不是"两个配置同时生效、可以随时切回"这种并存状态——这是本案例"没有退路"这个说法的真实技术根源,呼应 [07 类第 1 节](07-networking.md)"device 同一时刻只能激活一个 connection"这条机制。
- 用 `systemd-run --on-active=` 做自动回滚,记得同时想清楚"什么算确认成功"——本案例用手动 `systemctl stop` 定时器模拟"确认",真实生产环境里这一步往往是自动化的(比如脚本自己探测到新 IP 可达、外部监控上报健康检查通过才触发取消),不能让"确认"这一步本身也依赖这次可能已经失联的连接。

---

## 案例 5:systemctl enable 了就万事大吉?——启动顺序依赖的隐藏坑(工程约束递增轴)

建立在 [02 类第 3-4 节](02-process-and-boot.md)(systemctl 服务管理、`daemon-reload`、依赖关系查看)之上,修复手段用到 [04 类第 2 节](04-filesystem-and-permissions.md)的 fstab UUID 挂载。

**故障排查链条完整还原:**

- **故障现象:** 部署一个新的自定义 oneshot 服务,执行 `systemctl enable --now`,命令没有报错,回复"部署完成,已配置开机自启"。
- **排查动作 1(真实性验证,不信任"命令没报错"):** 不看命令返回值,直接查这次操作到底有没有换来"开机自启"这个结果——`systemctl is-enabled` 返回的是 `static`,不是 `enabled`。回头看这份 `.service` 文件,确实漏了 `[Install]` 段(`WantedBy=multi-user.target`)——没有这一段,`enable` 命令本身其实打印过一行警告("unit files have no installation config…not meant to be enabled"),但这行警告很容易被当成无关紧要的噪音划过去,命令的退出码依然是 0。
- **排查动作 2(升级,第一层修复):** 补上 `[Install]` 段,重新 `enable`,这次 `is-enabled` 正确显示 `enabled`,现场确认 `/etc/systemd/system/multi-user.target.wants/` 下真的多出了一个指向这份 `.service` 的符号链接——这才是"开机自启"背后真正发生的事情,不是一句抽象的配置(可运行例子 1/2)。
- **发现(更深一层,持久性/工程约束轴——"现在能跑"不等于"任何启动顺序下都能跑"):** 服务现在确实能正常启动、也确实配置了开机自启,但追问一句:这个服务读的配置文件在一块独立挂载的数据盘上,`enable` 只保证了"系统认为该启动它",完全没有保证"启动它的那一刻,它依赖的数据盘已经挂载好了"。真实开机过程中,各个服务和挂载点的启动顺序,如果没有显式声明依赖关系,systemd 并不保证谁先谁后。
- **排查动作 3(现场复现这个隐藏的竞态,不满足于"应该会有问题"这种猜测):** 由于本环境无法安全触发一次真实重启(见下方"常见坑"的诚实说明),用一个结构等价的方式复现"服务恰好先于挂载点跑起来"这个场景:手动卸载数据盘,重新启动这个服务——服务**真实地失败了**(`systemctl is-failed` 返回 `failed`),因为它试图读取一个此刻并不存在的文件。同一份 `enable` 配置,只因为这一次赶上挂载点还没就绪,结果就从"正常"变成"失败"——这正是没有声明依赖关系的服务在真实开机时可能撞上的竞态,不是虚构的场景。
- **根因:** `enable` 只回答了"要不要在开机时启动这个服务"这一个问题,完全不回答"启动它的那一刻,它需要的前置条件是否已经就绪"这第二个问题——这是两件独立的事情,而第二件事情不声明,systemd 不会替你自动推断。
- **修复与验证(决策依据——为什么用 `RequiresMountsFor=` 而不是手写 `After=挂载点单元名`):** 给 `.service` 文件加一行 `RequiresMountsFor=/mnt/挂载点路径`,`daemon-reload` 之后,复现完全相同的"挂载点没就绪,直接启动服务"场景——这次 systemd 自动先把依赖的挂载点补上,再启动服务,数据也是正确的那一份。选 `RequiresMountsFor=`(按路径声明)而不是手写 `After=挂载点对应的 .mount 单元名`,是因为前者同时隐含了强依赖(没挂上就不启动)和顺序(先挂再启动)两层语义,后者手写 `After=` 只解决顺序、不解决"没挂上也照样启动"这层更根本的问题,而且路径写法不需要记住 systemd 把挂载点路径转换成单元名的转义规则(可运行例子 2/2)。

**可运行例子(1/2):enable 了却不是真的 enabled——补上 [Install] 段前后对比**

```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

# 排查动作0(naive first attempt): 一个看起来完全正常的oneshot服务单元
cat > /etc/systemd/system/rhcsa10-app.service << 'EOF'
[Unit]
Description=RHCSA demo app
[Service]
Type=oneshot
ExecStart=/bin/true
EOF
systemctl daemon-reload
systemctl enable --now rhcsa10-app.service >/tmp/rhcsa10_enable.log 2>&1

# 真实性验证: 不看"命令有没有报错",直接查这次操作到底有没有换来"开机自启"
enabled_state=$(systemctl is-enabled rhcsa10-app.service 2>&1)
assert_eq "$enabled_state" "static"    # 不是"enabled"——这个单元没有[Install]段,enable命令本身几乎什么都没做
grep -qi "not meant to be enabled" /tmp/rhcsa10_enable.log && echo "OK: enable命令自己其实打印了警告,只是很容易被当成无关紧要的噪音划过去"

# 排查动作1(升级修复): 补上[Install]段,这才是"开机自启"真正需要声明的地方
cat > /etc/systemd/system/rhcsa10-app.service << 'EOF'
[Unit]
Description=RHCSA demo app
[Service]
Type=oneshot
ExecStart=/bin/true
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable rhcsa10-app.service >/dev/null 2>&1
fixed_state=$(systemctl is-enabled rhcsa10-app.service)
assert_eq "$fixed_state" "enabled"    # 这次才是真的"开机自启"

symlink_exists=0
[ -L /etc/systemd/system/multi-user.target.wants/rhcsa10-app.service ] && symlink_exists=1
assert_eq "$symlink_exists" "1"    # 现场验证真实机制: enable的本质就是在multi-user.target.wants/下创建一个符号链接

systemctl disable rhcsa10-app.service >/dev/null 2>&1
rm -f /etc/systemd/system/rhcsa10-app.service /tmp/rhcsa10_enable.log
systemctl daemon-reload
```

本机实测:全部断言输出 `OK`。缺少 `[Install]` 段时 `is-enabled` 返回 `static`(不是 `enabled`),`enable` 命令本身打印过安装提示但退出码依然是 0;补齐后 `is-enabled` 正确显示 `enabled`,且 `multi-user.target.wants/` 下真实出现了对应符号链接。

**可运行例子(2/2):同一份正确 enable 的配置,挂载点没就绪时真实失败;声明 RequiresMountsFor 之后,systemd 自动补上依赖**

```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

dd if=/dev/zero of=/tmp/rhcsa10_appdata.img bs=1M count=100 2>/dev/null
loopd=$(losetup -f); losetup "$loopd" /tmp/rhcsa10_appdata.img
mkfs.ext4 -q "$loopd"
uuid=$(blkid -o value -s UUID "$loopd")
mkdir -p /mnt/rhcsa10_appdata
mount "$loopd" /mnt/rhcsa10_appdata
echo "real_config_value=42" > /mnt/rhcsa10_appdata/config.txt
umount /mnt/rhcsa10_appdata
echo "UUID=$uuid /mnt/rhcsa10_appdata ext4 defaults 0 0" >> /etc/fstab
systemctl daemon-reload

# 排查动作0: 服务已经正确enable(呼应例1的教训,这次[Install]段是对的),看起来"配置完成"
cat > /etc/systemd/system/rhcsa10-app.service << 'EOF'
[Unit]
Description=RHCSA demo app reading data from mount
[Service]
Type=oneshot
ExecStart=/bin/bash -c 'cat /mnt/rhcsa10_appdata/config.txt > /tmp/rhcsa10_app_result.txt'
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable rhcsa10-app.service >/dev/null 2>&1
mount -a
systemctl start rhcsa10-app.service >/dev/null 2>&1
assert_eq "$(systemctl is-enabled rhcsa10-app.service)" "enabled"
assert_eq "$(cat /tmp/rhcsa10_app_result.txt)" "real_config_value=42"

# 真实性验证: "现在启动正常"不等于"任何启动顺序下都正常"——模拟"这次它恰好先于挂载点跑起来"
umount /mnt/rhcsa10_appdata
rm -f /tmp/rhcsa10_app_result.txt
systemctl stop mnt-rhcsa10_appdata.mount 2>/dev/null
systemctl reset-failed rhcsa10-app.service mnt-rhcsa10_appdata.mount 2>/dev/null
systemctl start rhcsa10-app.service >/dev/null 2>&1
race_failed=$(systemctl is-failed rhcsa10-app.service)
assert_eq "$race_failed" "failed"    # 同一个单元、同一份enable配置,只因为这次挂载点还没就绪,就真实地失败了——
                                       # 这说明"enable了+现在能跑"并不能证明"任何一次真实启动顺序下都能跑"

# 排查动作1(升级修复,决策依据: 为什么用RequiresMountsFor而不是手写After=mnt-rhcsa10_appdata.mount):
# RequiresMountsFor按"路径"表达依赖,系统自动解析成对应的.mount单元,即使将来改了挂载点命名也不用同步改这里;
# 而且它同时隐含了Requires(强依赖,不只是顺序)+After,手写After=只解决顺序,不解决"没挂上就不该跑"这层强制
cat > /etc/systemd/system/rhcsa10-app.service << 'EOF'
[Unit]
Description=RHCSA demo app reading data from mount
RequiresMountsFor=/mnt/rhcsa10_appdata
[Service]
Type=oneshot
ExecStart=/bin/bash -c 'cat /mnt/rhcsa10_appdata/config.txt > /tmp/rhcsa10_app_result.txt'
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl reset-failed rhcsa10-app.service mnt-rhcsa10_appdata.mount 2>/dev/null

# 复现完全相同的场景: 挂载点没就绪,直接启动服务
mount_before=$(mount | grep -c rhcsa10_appdata)
assert_eq "$mount_before" "0"
systemctl start rhcsa10-app.service >/dev/null 2>&1
fixed_result=$(systemctl is-failed rhcsa10-app.service)
assert_eq "$fixed_result" "inactive"    # 这次没有失败
mount_after=$(mount | grep -c rhcsa10_appdata)
assert_eq "$mount_after" "1"    # systemd自己先把依赖的挂载点补上了,不是手动挂的
assert_eq "$(cat /tmp/rhcsa10_app_result.txt)" "real_config_value=42"    # 数据也是真实、正确的那一份

systemctl disable rhcsa10-app.service >/dev/null 2>&1
rm -f /etc/systemd/system/rhcsa10-app.service /tmp/rhcsa10_app_result.txt
systemctl daemon-reload
umount /mnt/rhcsa10_appdata 2>/dev/null
sed -i "\|rhcsa10_appdata|d" /etc/fstab
systemctl daemon-reload
losetup -d "$loopd"
rm -f /tmp/rhcsa10_appdata.img
rmdir /mnt/rhcsa10_appdata
```

本机实测:全部断言输出 `OK`。没有 `RequiresMountsFor` 时,挂载点未就绪导致服务真实失败(`is-failed` = `failed`);加上之后,同样"挂载点未就绪"的起点,systemd 自动先挂载依赖、再启动服务,`is-failed` 变回 `inactive`(未失败),读到的数据也正确。

**常见坑(含判分点提示):**
- RHCSA 判分"配置服务开机自启"类题目,标准做法是重启考试虚拟机后重新检查服务状态——这正是本案例"`is-enabled` 显示 `static` 而不是 `enabled`"这类问题会被真实抓到的地方,`enable` 命令本身"没报错"不代表判分会通过。
- **如实说明本环境的验证边界**:本案例没有、也不会在 WSL2 里做一次真实的完整重启来验证"挂载顺序竞态在真实开机时会不会发生"——[02 类第 2 节](02-process-and-boot.md)已经如实说明 WSL 没有真实 GRUB/完整开机流程,本案例用"手动卸载数据盘 + 重新启动服务"复现的是这个竞态**背后的机制**(缺少依赖声明时,服务不会等待/要求它需要的挂载点),这个机制经过真实验证,但"真实开机顺序恰好会不会踩中这个坑"这个更具体的概率性问题,需要真实 RHEL 环境里的多次重启测试才能给出结论,这里不冒充已经验证过。
- 忘记 `daemon-reload` 是这条系列反复出现的坑([02 类第 4 节](02-process-and-boot.md)已经点出过)——改完 `.service` 文件不 `daemon-reload`,新内容不会生效,本案例两次修改单元文件都不能漏这一步。

---

## 小结:5 个案例对应的运维排障轴线

| 案例 | 规模递增轴 | 持久性(工程约束递增轴) | 方案批判迭代轴 | 决策依据追问轴 | 真实性验证轴 |
|---|---|---|---|---|---|
| 1. LVM 连环扩容 | ✅ 核心 | | ✅ 核心 | ✅ | |
| 2. fstab 挂错盘 | | | | | ✅ 核心 |
| 3. sudo 两次收紧 | | | ✅ | ✅ 核心 | ✅ |
| 4. 静态 IP 失联 | ✅ 核心 | | ✅ | ✅ | ✅ |
| 5. enable 隐藏坑 | | ✅ 核心 | | ✅ | ✅ |

这 5 个案例不是要把 100 个知识点全部重写——它们演示的是**方法论本身**:拿到任何一个已经掌握的知识点,都可以自己追问"这个操作在 10 倍/100 倍规模、共享资源、批量场景下会先撞上什么""这个配置重启之后/换一次执行顺序还在不在生效""如果连续被指出这个方案的新缺陷,下一个更彻底的方案是什么""我凭什么认定这样配置就选对了参数,不是拍脑袋""我怎么用现场验证而不是'应该没问题'说服别人"。真实的运维排障深度,是能不能对着一个自己没准备过的知识点,现场把这几条轴线走一遍——而且永远记住这条贯穿全部 5 个案例的底线:**命令不报错,不等于配置对了;配置对了,不等于任何时候都对了。**
