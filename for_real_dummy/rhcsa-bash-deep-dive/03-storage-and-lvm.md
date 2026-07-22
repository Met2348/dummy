# 03 · 本地存储与 LVM(Local Storage & LVM)

> 总览见 [00-roadmap.md](00-roadmap.md)
> 这一篇覆盖 12 个知识点:分区基础、parted、LVM 三层模型完整生命周期(PV→VG→LV→扩展→正确删除顺序)、swap、磁盘配额、loop device 实验技巧本身。**本文所有代码例子已在 Rocky Linux 10.2(WSL2)下实际跑通验证**,按 [00-roadmap.md](00-roadmap.md) 既定纪律用 `dd`/`truncate` + `losetup` 构造 loop device 模拟磁盘,不动真实磁盘,验证完在代码块内清理干净。
>
> **和前四条系列的差异声明**(详见 [00-roadmap.md](00-roadmap.md) 开头):本仓库没有 Linux 系统管理场景可挖,"真实场景例子"部分统一标注为典型运维/RHCSA 考试场景,不冒充仓库代码里挖出来的。

---

## 1. 磁盘分区基础(MBR vs GPT,`fdisk` vs `parted` vs `gdisk`)

**命令/配置:**
```bash
fdisk /dev/sdX       # 传统交互式分区工具,MBR/GPT都支持(较新版本)
parted /dev/sdX        # 更现代的分区工具,脚本化(-s)友好,MBR/GPT都原生支持
gdisk /dev/sdX          # 专门针对GPT设计的分区工具(fdisk的GPT专用版本)
```

**一句话是什么:** MBR(Master Boot Record)是传统的分区表格式,受限于设计年代,最多支持 4 个主分区、单分区最大 2TB;GPT(GUID Partition Table)是现代标准,支持远超 2TB 的大容量磁盘、理论上近乎无限的分区数量,是当前新装系统的默认选择;`fdisk`/`parted`/`gdisk` 是三种分区管理工具,`parted` 因为脚本化能力强,是自动化场景的首选。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"列出、创建、删除分区";超过 2TB 的磁盘在 MBR 下根本无法完整使用,现代服务器/云主机存储动辄数百 GB 到 TB 级,理解 MBR 的容量限制、知道该用 GPT,是磁盘规划的基本常识。

**从最容易犯错的做法讲起:** 想当然地认为 RHCSA 考纲提到的三个工具在任意 RHEL/Rocky 系统上都预装好了——**本机实测证伪**:`gdisk` 默认没有安装(不在基础/AppStream 仓库里),需要额外配置仓库或者干脆用 `parted` 代替(`parted` 对 MBR/GPT 都原生支持,功能上完全能覆盖 `gdisk` 的场景),这提醒一个更普遍的原则:操作前先确认工具是否存在,不能凭考纲"应该有"就假设。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 新加一块 4TB 的数据盘,必须用 GPT(MBR 处理不了超过 2TB 的容量);老旧的、遗留的小容量系统盘可能还在用 MBR(历史遗留,不代表推荐做法),排障时要先确认现有分区表类型,而不是想当然地用新知识套旧系统。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

gdisk_present=0; command -v gdisk >/dev/null 2>&1 && gdisk_present=1
assert_eq "$gdisk_present" "0"    # 如实探测:默认仓库没有gdisk,用parted代替演示

dd if=/dev/zero of=/tmp/rhcsa03_disk1.img bs=1M count=200 2>/dev/null
loop1=$(losetup -f); losetup "$loop1" /tmp/rhcsa03_disk1.img

parted -s "$loop1" mklabel msdos    # msdos就是MBR在parted里的叫法
assert_eq "$(parted -s "$loop1" print 2>&1 | grep "Partition Table" | awk '{print $NF}')" "msdos"

parted -s "$loop1" mklabel gpt
assert_eq "$(parted -s "$loop1" print 2>&1 | grep "Partition Table" | awk '{print $NF}')" "gpt"

losetup -d "$loop1"; rm -f /tmp/rhcsa03_disk1.img
```
本机实测:两个断言均输出 `OK`。

**常见坑:** 一块磁盘只能有一种分区表(MBR 或 GPT 二选一),`mklabel` 切换分区表类型是**破坏性操作**,会清空这块磁盘原有的所有分区信息——生产环境上误执行 `mklabel` 相当于瞬间抹掉整个磁盘的分区结构,数据虽然物理上可能还在,但找回来的难度和成本极高,操作前务必反复确认目标设备。

---

## 2. parted 交互式分区操作

**命令/配置:**
```bash
parted /dev/sdX                          # 进入交互模式
parted -s /dev/sdX mkpart primary ext4 1MiB 100MiB    # -s脚本模式,一条命令直接完成不进交互
parted -s /dev/sdX print                    # 查看当前分区布局
parted -s /dev/sdX rm 1                      # 删除1号分区
```

**一句话是什么:** `parted` 既能像传统工具一样进入交互式菜单一步步操作,也能用 `-s`(script)配合完整参数一条命令直接完成分区动作,后者是自动化脚本首选的用法——起止位置支持 `MiB`/`GiB` 这类明确单位,也支持 `100%` 这种表示"用满剩余全部可用空间"的特殊写法。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 考试环境下,脚本化、可重复的分区操作比手动交互式点选更可靠、更适合在时间压力下减少失误;`parted -s` 是自动化部署脚本创建分区的标准写法。

**从最容易犯错的做法讲起:** 手动计算分区终点时,把终点直接写成磁盘的绝对总容量(比如磁盘是 200MiB 就写终点 `200MiB`)——**本机实测证伪**:这样做会报错 `Error: The location 200MiB is outside of the device`,因为 GPT 分区表会在磁盘**末尾**保留一小段空间存储备份分区表头,实际可用空间比磁盘总大小略小;正确做法是用 `100%` 表示"用满剩余全部可用空间",让 `parted` 自己计算出考虑了 GPT 尾部保留区域之后的准确终点,不需要手动精确计算。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 自动化部署脚本给新盘分区,`parted -s /dev/sdb mklabel gpt mkpart primary ext4 1MiB 100%`,一条命令完成"建 GPT 表 + 用满整块盘建一个分区",不需要人工计算任何具体数值,脚本换到不同容量的盘上依然通用。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

dd if=/dev/zero of=/tmp/rhcsa03_disk2.img bs=1M count=200 2>/dev/null
loop2=$(losetup -f); losetup "$loop2" /tmp/rhcsa03_disk2.img

parted -s "$loop2" mklabel gpt
parted -s "$loop2" mkpart primary ext4 1MiB 100MiB
parted -s "$loop2" mkpart primary ext4 100MiB 100%    # 100%避免手动计算GPT尾部保留区域出错

part_count=$(parted -s "$loop2" print 2>&1 | grep -c "^ [0-9]")
assert_eq "$part_count" "2"

losetup -d "$loop2"; rm -f /tmp/rhcsa03_disk2.img
```
本机实测:断言输出 `OK`。

**常见坑:** `parted` 的修改是**立即生效**的,不像某些工具有"先编辑、最后统一写入确认"的缓冲步骤——每一条 `mkpart`/`rm` 命令敲下去就已经真正改动了分区表,没有"最后再确认一次"的后悔机会,这和 `fdisk` 交互模式下需要显式敲 `w` 才真正写入磁盘的设计理念不同,操作 `parted` 时要格外小心。

---

## 3. `lsblk`/`blkid` 查看磁盘与 UUID

**命令/配置:**
```bash
lsblk                    # 树状展示所有块设备及其分区/挂载关系
lsblk -f                   # 额外显示文件系统类型和UUID
blkid                       # 列出所有设备的文件系统类型/UUID/LABEL等元数据
blkid /dev/sdX1               # 只查看指定设备
```

**一句话是什么:** `lsblk`(list block devices)以直观的树状结构展示"磁盘→分区→(可能的)LVM/挂载点"这层层嵌套关系,适合快速建立"整个系统的存储长什么样"的全局视图;`blkid` 专注在"某个具体设备的文件系统元数据"(类型、UUID、标签),是获取 UUID(用于 04 类第 2 节 `/etc/fstab` 配置)最直接的方式。

**为什么 RHCSA 真考 / 生产会用到:** 几乎每个存储相关的 RHCSA 大题第一步都是先用 `lsblk` 摸清楚当前磁盘/分区布局,再决定下一步操作;`blkid` 获取 UUID 是配置持久化挂载(`/etc/fstab`)的前置必经步骤。

**从最容易犯错的做法讲起:** 排障或者写自动化脚本时用**设备名**(比如 `/dev/sdb1`)而不是 UUID 来标识一块盘——设备名会随着硬件插拔顺序、甚至系统重启后的枚举顺序发生变化,今天的 `/dev/sdb` 可能因为加了一块新盘变成明天的 `/dev/sdc`;UUID 是文件系统创建时生成的、和硬件插入顺序无关的唯一标识,永远不会因为这类原因发生变化,这是为什么 04 类反复强调 `/etc/fstab` 要用 UUID 而不是设备名的根本原因。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 新加一块盘,先 `lsblk` 确认它出现在系统里、设备名是什么,分区/格式化之后用 `blkid` 拿到 UUID 写进 `/etc/fstab`;排障"某个挂载点消失了",先 `lsblk` 看看对应的分区还在不在、状态如何,再决定是重新挂载还是要进一步排查硬件问题。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

dd if=/dev/zero of=/tmp/rhcsa03_disk3.img bs=1M count=200 2>/dev/null
loop3=$(losetup -f); losetup "$loop3" /tmp/rhcsa03_disk3.img
mkfs.ext4 -q "$loop3"

lsblk "$loop3" 2>&1 | grep -q "loop" && echo "OK: lsblk能看到这个loop设备"

uuid=$(blkid -o value -s UUID "$loop3")
assert_ok test -n "$uuid"
# UUID符合标准RFC4122格式:8-4-4-4-12个十六进制字符,用连字符分隔
uuid_format_check=$(echo "$uuid" | grep -cE '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
assert_eq "$uuid_format_check" "1"

losetup -d "$loop3"; rm -f /tmp/rhcsa03_disk3.img
```
本机实测:两个检查点均输出 `OK`。

**常见坑:** `blkid` 对没有文件系统的"裸"分区/设备不会返回任何 TYPE 信息(因为压根没有可识别的文件系统特征)——刚 `parted` 分完区、还没 `mkfs` 的分区,`blkid` 查询会是空的,这是正常现象不是 bug,顺序上必须先 `mkfs` 之后 `blkid` 才有意义。

---

## 4. LVM 概念与三层模型(PV/VG/LV)

**命令/配置:**
```
PV (Physical Volume,物理卷)  = 一块磁盘/分区,LVM能识别管理的最小物理单元
VG (Volume Group,卷组)        = 若干个PV组成的一个"存储池子",容量是池子里所有PV容量之和
LV (Logical Volume,逻辑卷)     = 从VG这个池子里按需切出来的一块空间,是最终用来mkfs/挂载的对象
```

**一句话是什么:** LVM(Logical Volume Manager)在"物理磁盘"和"能挂载使用的分区"之间插入了两层抽象——PV 把物理磁盘/分区标准化成 LVM 能管理的单元,VG 把多个 PV 池化成一个不受单块物理磁盘容量限制的大池子,LV 再从这个池子里灵活切出任意大小的逻辑空间;这套三层模型解决了传统分区"大小固定、扩容困难、受限于单块物理磁盘容量"的根本问题。

**把"多对一再一对多"这个拓扑关系画出来:**
```
  物理磁盘/分区(PV候选)
  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ /dev/sda │  │ /dev/sdb │  │ /dev/sdc │    三块独立物理盘,大小可以完全不同
  │  100 GB  │  │  200 GB  │  │   50 GB  │
  └────┬─────┘  └────┬─────┘  └────┬─────┘
       │ pvcreate     │ pvcreate    │ pvcreate    ← 第一步:每块盘各自标准化成一个PV
       ▼              ▼             ▼
   [ PV(sda) ]    [ PV(sdb) ]    [ PV(sdc) ]
       │              │             │
       └──────────────┼─────────────┘  ← 第二步:vgcreate,多个PV"多对一"池化成一个VG
                       ▼
            ┌───────────────────┐
            │  VG(总容量=350GB)   │   一个"存储池子",往上不再关心数据具体落在哪块物理盘
            └──────────┬─────────┘
                        │  ← 第三步:lvcreate,一个VG"一对多"切出多个LV,按需分配大小
           ┌────────────┼────────────┐
           ▼            ▼            ▼
      [LV data]     [LV logs]    [LV backup]   ← 这里才是最终 mkfs/挂载使用的对象
       200 GB         50 GB        100 GB
```
这也是为什么"VG 容量能突破单块物理磁盘限制"——LV(比如上面的 `data`)拿到的 200GB,底层数据可能横跨 sda 和 sdb 两块物理盘,使用者完全不需要关心这一点。

**为什么 RHCSA 真考 / 生产会用到:** LVM 相关操作(创建/扩展/管理)是 RHCSA 考试分值占比很高的核心技能块;几乎所有生产环境的 RHEL 系统盘都用 LVM 管理(而不是直接用裸分区),因为业务增长后"这块盘装不下了要扩容"是必然会遇到的需求,LVM 让这件事变得简单(不需要停机、不需要数据搬迁)。

**从最容易犯错的做法讲起:** 混淆"分区容量"和"VG 容量能突破单块磁盘限制"这个 LVM 的核心价值——传统分区一旦划定大小,想扩大极其麻烦(往往需要备份数据、重新分区、恢复数据这类高风险操作);LVM 下,只要给 VG `vgextend` 加入新的物理盘(哪怕是全新买来的一块盘),LV 就能立刻 `lvextend` 扩容,VG 的总容量不受"必须是同一块物理磁盘"这个限制。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 数据库存储空间快用完了,不需要停机迁移数据,买一块新盘插上,`pvcreate` 识别它、`vgextend` 加入现有 VG、`lvextend` 给数据库对应的 LV 扩容、`resize2fs`/`xfs_growfs` 让文件系统感知到新空间——整个过程业务不中断,这是 LVM 相对传统分区最大的实际价值。

**可运行例子:**
```bash
echo "PV(物理卷) → VG(卷组,池化多个PV) → LV(逻辑卷,从VG按需切出) 三层递进关系"
echo "本节是概念铺垫,完整的PV/VG/LV创建流程见接下来第5-9节的连续可运行例子"
```

**常见坑:** 新手容易以为"删除 LV 就等于清空了对应的物理磁盘数据"——实际上物理磁盘(PV)、卷组(VG)、逻辑卷(LV)是三个独立的管理层级,删除 LV 只是从 VG 这个池子里释放出对应的空间(变回"未分配"状态),PV 和 VG 本身依然存在,底层物理磁盘也毫发无损,后面第 9 节会具体展开"删除的正确顺序"。

---

## 5. 创建物理卷 `pvcreate`/`pvdisplay`

**命令/配置:**
```bash
pvcreate /dev/sdX          # 把一块磁盘/分区初始化成LVM能识别的物理卷
pvdisplay [/dev/sdX]         # 查看物理卷详情
pvs                            # 简洁的表格式列出所有PV(pv summary的常用简写形式)
```

**一句话是什么:** `pvcreate` 在目标设备开头写入 LVM 的元数据标签,把一块普通的磁盘/分区"注册"成 LVM 体系能识别管理的物理卷——这是使用 LVM 的第一步,没有先 `pvcreate`,这块盘对 LVM 来说就是"看不见"的普通存储,不能被纳入任何 VG。

**为什么 RHCSA 真考 / 生产会用到:** 这是 LVM 完整操作流程(PV→VG→LV)的第一环,RHCSA 明确要求掌握;`pvdisplay`/`pvs` 是排查"这块盘到底有没有被 LVM 正确识别"的第一道检查点。

**从最容易犯错的做法讲起:** 直接对一块**已经有其他文件系统/数据**的设备执行 `pvcreate`,以为这只是"注册"不会影响原有数据——实际上 `pvcreate` 会在设备开头写入 LVM 元数据,这个操作本身可能覆盖掉原有分区表/文件系统的关键元数据区域(尤其是设备开头部分),等同于一次数据破坏性操作,对已有数据的设备操作前必须先确认清楚这块盘是不是"干净的"或者数据是否已经备份。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 新买的空白磁盘接入服务器,`pvcreate /dev/sdb` 把它注册进 LVM 体系,为后续加入 VG 做准备;RHCSA 考试题目要求"给系统扩容"时,第一步几乎总是先确认目标设备是空白的、再 `pvcreate`。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

dd if=/dev/zero of=/tmp/rhcsa03_pv_demo.img bs=1M count=300 2>/dev/null
loop5=$(losetup -f); losetup "$loop5" /tmp/rhcsa03_pv_demo.img

pvcreate -y "$loop5" >/dev/null 2>&1
assert_eq "$?" "0"
assert_eq "$(pvs "$loop5" --noheadings 2>/dev/null | wc -l)" "1"
pvdisplay "$loop5" 2>&1 | grep -q "PV Name" && echo "OK: pvdisplay能查看PV详情"

pvremove -y "$loop5" >/dev/null 2>&1
losetup -d "$loop5"; rm -f /tmp/rhcsa03_pv_demo.img
```
本机实测:全部检查点输出 `OK`。

**常见坑:** `pvcreate` 对目标设备大小有隐性的**最低要求**(需要留出空间存放 LVM 元数据本身),极小的设备(比如几 MB)可能会创建失败或者可用空间被元数据占掉大半——实验/测试用的 loop device 不要设置得太小,留出合理的余量。

---

## 6. 创建卷组 `vgcreate`/`vgextend`

**命令/配置:**
```bash
vgcreate VG_NAME /dev/sdX1 [/dev/sdX2 ...]    # 用一个或多个PV创建卷组
vgextend VG_NAME /dev/sdY                       # 把新的PV加入已存在的卷组,扩大池子容量
vgs                                                # 简洁列出所有VG
vgdisplay VG_NAME                                    # 查看VG详情
```

**一句话是什么:** `vgcreate` 把一个或多个已经 `pvcreate` 过的物理卷组织成一个卷组——这个"池子"的总容量是所有成员 PV 容量之和;`vgextend` 是让这个池子后续继续变大的手段,不需要在最初创建时就规划好全部容量,这正是 LVM 相比传统分区最大的灵活性来源。

**为什么 RHCSA 真考 / 生产会用到:** VG 是连接"物理存储资源"和"逻辑使用空间"的枢纽层,RHCSA 明确要求掌握 VG 的创建和扩展操作;生产环境"加盘扩容"的标准流程就是 `pvcreate` 新盘 + `vgextend` 加入现有 VG。

**从最容易犯错的做法讲起:** 创建 VG 时想着"一次性把未来可能用到的所有盘都加进去",但实际业务扩容需求是渐进式的——更合理的做法是先用当前实际需要的 PV `vgcreate`,未来真的需要更多空间时再 `vgextend`,不需要一开始就过度规划,LVM 的设计本身就支持"随用随扩"这种渐进模式。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 服务器初始只有一块盘,`vgcreate data_vg /dev/sdb1` 建立初始卷组;半年后业务增长,新增一块盘,`pvcreate /dev/sdc1` 注册、`vgextend data_vg /dev/sdc1` 把它并入同一个卷组,VG 总容量立刻增加,后续可以直接给现有 LV 扩容,不需要重新规划整个存储架构。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

dd if=/dev/zero of=/tmp/rhcsa03_vg_demo1.img bs=1M count=300 2>/dev/null
dd if=/dev/zero of=/tmp/rhcsa03_vg_demo2.img bs=1M count=300 2>/dev/null
loop6a=$(losetup -f); losetup "$loop6a" /tmp/rhcsa03_vg_demo1.img
loop6b=$(losetup -f); losetup "$loop6b" /tmp/rhcsa03_vg_demo2.img

pvcreate -y "$loop6a" >/dev/null 2>&1
vgcreate rhcsa03_vg_demo "$loop6a" >/dev/null 2>&1
assert_eq "$?" "0"

pvcreate -y "$loop6b" >/dev/null 2>&1
vgextend rhcsa03_vg_demo "$loop6b" >/dev/null 2>&1
assert_eq "$?" "0"
pv_count_in_vg=$(vgs rhcsa03_vg_demo --noheadings -o pv_count 2>/dev/null | tr -d ' ')
assert_eq "$pv_count_in_vg" "2"    # vgextend确实把第2个PV并入了同一个VG,池子容量翻倍

vgremove -y rhcsa03_vg_demo >/dev/null 2>&1
pvremove -y "$loop6a" "$loop6b" >/dev/null 2>&1
losetup -d "$loop6a"; losetup -d "$loop6b"
rm -f /tmp/rhcsa03_vg_demo1.img /tmp/rhcsa03_vg_demo2.img
```
本机实测:两个断言均输出 `OK`,VG 从 1 个 PV 扩展到 2 个 PV,总容量随之增加。

**常见坑:** `vgextend` 加入的新 PV,即便它之前**已经**装过其他文件系统或者被别的 LVM 结构使用过,`vgcreate`/`vgextend` 依然可能"成功"执行——LVM 层面的操作不会自动检查底层是否有残留的其他类型数据,重要生产环境操作前务必先确认目标设备真的是"干净"或者已经过妥善清理的,不能只看命令是否报错来判断安全性。

---

## 7. 创建逻辑卷 `lvcreate`(线性卷)

**命令/配置:**
```bash
lvcreate -n LV_NAME -L SIZE VG_NAME    # 按绝对大小创建LV(比如-L 10G)
lvcreate -n LV_NAME -l 100%FREE VG_NAME   # 按VG剩余空间百分比创建LV(用满剩余全部空间)
lvs                                          # 简洁列出所有LV
```

**一句话是什么:** `lvcreate` 从 VG 这个池子里"切"出一块指定大小的逻辑空间,创建出来的 LV 会在 `/dev/VG_NAME/LV_NAME`(或 `/dev/mapper/VG_NAME-LV_NAME`)出现一个设备节点,后续对它 `mkfs`/挂载的操作方式和对待一个普通分区完全一样——LV 是 LVM 体系里真正拿来"用"的最终产物。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"创建、删除逻辑卷";理解 `-L`(绝对大小)和 `-l`(百分比,常配合 `%FREE`/`%VG` 使用)这两种指定容量的方式,能应对不同场景下"精确控制大小"还是"用满可用空间"的不同需求。

**从最容易犯错的做法讲起:** 创建 LV 时把 VG 的全部容量一次性切光(比如用 `-l 100%FREE`),之后发现还需要给另一个用途也创建 LV,却发现 VG 已经没有剩余空间——除非明确知道这个 VG 未来不会再需要切分给其他用途,否则更稳妥的做法是按需分配一个合理大小(`-L` 指定具体数值),给未来的灵活性留出余地,而不是一次性用尽。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 给数据库预留 20GB 存储:`lvcreate -n db_lv -L 20G data_vg`,后续如果空间不够用 `lvextend` 追加(下一节讲),而不是一开始就把整个 VG 容量全部切给这一个用途;需要快速创建一个"用满剩余全部可用空间"的临时测试卷,`lvcreate -n test_lv -l 100%FREE data_vg` 一步到位。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

dd if=/dev/zero of=/tmp/rhcsa03_lv_demo1.img bs=1M count=300 2>/dev/null
dd if=/dev/zero of=/tmp/rhcsa03_lv_demo2.img bs=1M count=300 2>/dev/null
loop7a=$(losetup -f); losetup "$loop7a" /tmp/rhcsa03_lv_demo1.img
loop7b=$(losetup -f); losetup "$loop7b" /tmp/rhcsa03_lv_demo2.img
pvcreate -y "$loop7a" "$loop7b" >/dev/null 2>&1
vgcreate rhcsa03_lv_vg "$loop7a" "$loop7b" >/dev/null 2>&1

lvcreate -y -n rhcsa03_lv_demo -L 400M rhcsa03_lv_vg >/dev/null 2>&1
assert_eq "$?" "0"
assert_ok test -e /dev/rhcsa03_lv_vg/rhcsa03_lv_demo    # 设备节点真实出现,可以像普通分区一样mkfs/挂载

lvremove -y rhcsa03_lv_vg/rhcsa03_lv_demo >/dev/null 2>&1
vgremove -y rhcsa03_lv_vg >/dev/null 2>&1
pvremove -y "$loop7a" "$loop7b" >/dev/null 2>&1
losetup -d "$loop7a"; losetup -d "$loop7b"
rm -f /tmp/rhcsa03_lv_demo1.img /tmp/rhcsa03_lv_demo2.img
```
本机实测:LV 创建成功,设备节点 `/dev/rhcsa03_lv_vg/rhcsa03_lv_demo` 真实存在,大小约 400MB(跨越了单个 PV 约 296MB 的容量上限,证明确实是从两个 PV 组成的 VG 池子里切出来的,这正是 LVM 相对单块物理磁盘限制的价值所在)。

**常见坑:** `lvcreate -L` 请求的容量,如果 VG 剩余空间不够,会直接报错拒绝创建(不会创建一个"打折"的更小 LV)——这是安全的失败模式,但也提醒创建前最好先 `vgs` 确认 VG 剩余容量是否真的够用,不要凭印象猜测。

---

## 8. 扩展逻辑卷 `lvextend` + 文件系统同步扩容(`resize2fs`/`xfs_growfs`)

**命令/配置:**
```bash
lvextend -L +SIZE /dev/VG_NAME/LV_NAME       # LV容量增加指定大小(注意+号,表示"追加"而不是"设为")
resize2fs /dev/VG_NAME/LV_NAME                  # ext4文件系统同步感知新的LV大小
xfs_growfs /mount/point                           # xfs文件系统同步扩容(注意:传的是挂载点,不是设备路径!)
```

**一句话是什么:** `lvextend` 只是让 LV 这个"块设备"在 LVM 层面变大了,但**文件系统本身并不会自动感知**这个变化(文件系统有自己独立记录的大小信息)——必须额外执行 `resize2fs`(ext4)或 `xfs_growfs`(xfs)这一步,文件系统才会"看到"新增的空间并真正可用,这是两个独立的操作,缺一不可。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"扩展现有逻辑卷";"只扩容了 LVM 层面,文件系统层面没同步扩容,导致 `df` 看到的可用空间依然是旧值"是最常见的丢分/生产事故场景,两步都做才算完整完成任务。

**从最容易犯错的做法讲起:** 执行完 `lvextend` 就以为扩容大功告成,`df -h` 一看发现可用空间完全没变化——这正是"忘记同步扩容文件系统"这个坑,`lvextend` 和 `resize2fs`/`xfs_growfs` 必须**成对**执行,`lvextend` 之后不跟着做文件系统扩容,这次操作在用户可感知的层面上等于什么都没做。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 应用日志把磁盘写满了,紧急扩容:`lvextend -L +5G /dev/data_vg/log_lv` 先让 VG 里的空闲空间并入这个 LV,紧接着 `resize2fs /dev/data_vg/log_lv`(如果是 ext4)让文件系统立刻能使用这新增的 5GB,应用无需重启,写入操作可以继续进行;`xfs_growfs` 特别要注意传参是**挂载点路径**而不是设备路径,这和 `resize2fs` 的参数习惯不同,容易搞混。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

dd if=/dev/zero of=/tmp/rhcsa03_ext_demo1.img bs=1M count=300 2>/dev/null
dd if=/dev/zero of=/tmp/rhcsa03_ext_demo2.img bs=1M count=300 2>/dev/null
loop8a=$(losetup -f); losetup "$loop8a" /tmp/rhcsa03_ext_demo1.img
loop8b=$(losetup -f); losetup "$loop8b" /tmp/rhcsa03_ext_demo2.img
pvcreate -y "$loop8a" "$loop8b" >/dev/null 2>&1
vgcreate rhcsa03_ext_vg "$loop8a" "$loop8b" >/dev/null 2>&1
lvcreate -y -n rhcsa03_ext_lv -L 400M rhcsa03_ext_vg >/dev/null 2>&1

mkfs.ext4 -q /dev/rhcsa03_ext_vg/rhcsa03_ext_lv
mkdir -p /mnt/rhcsa03_ext_test
mount /dev/rhcsa03_ext_vg/rhcsa03_ext_lv /mnt/rhcsa03_ext_test
size_before=$(df -m /mnt/rhcsa03_ext_test | tail -1 | awk '{print $2}')

lvextend -L +100M /dev/rhcsa03_ext_vg/rhcsa03_ext_lv >/dev/null 2>&1
resize2fs /dev/rhcsa03_ext_vg/rhcsa03_ext_lv >/dev/null 2>&1
size_after=$(df -m /mnt/rhcsa03_ext_test | tail -1 | awk '{print $2}')
assert_ok test "$size_after" -gt "$size_before"    # 文件系统层面(df看到的)确实感知到了扩容,不只是LVM层面变大

umount /mnt/rhcsa03_ext_test
lvremove -y rhcsa03_ext_vg/rhcsa03_ext_lv >/dev/null 2>&1
vgremove -y rhcsa03_ext_vg >/dev/null 2>&1
pvremove -y "$loop8a" "$loop8b" >/dev/null 2>&1
losetup -d "$loop8a"; losetup -d "$loop8b"
rm -f /tmp/rhcsa03_ext_demo1.img /tmp/rhcsa03_ext_demo2.img
rmdir /mnt/rhcsa03_ext_test
```
本机实测:`df` 看到的可用空间从约 365MB 增长到约 459MB,确认扩容在文件系统层面真实生效,断言输出 `OK`。

**常见坑:** `resize2fs` 直接对**设备路径**操作(`/dev/VG/LV`),`xfs_growfs` 却要求传**挂载点路径**——这个参数习惯上的不一致是新手极易搞混的地方,传错参数类型通常会直接报错(不会误操作到别的设备),但排查起来容易浪费时间,记不清楚时随手 `--help` 确认一下比凭印象瞎试更快。

---

## 9. 缩减/删除 LV/VG/PV 的正确顺序

**命令/配置:**
```bash
lvremove VG_NAME/LV_NAME    # 第一步:先删LV
vgremove VG_NAME               # 第二步:LV都删完了,再删VG
pvremove /dev/sdX                # 第三步:VG都删完了,再把PV标签清除
```

**一句话是什么:** LVM 的三层结构(PV→VG→LV)决定了删除必须**反着来**——先删最上层的 LV,再删中间层的 VG,最后才轮到清除最底层 PV 的 LVM 标签,任何一步顺序颠倒都会被 LVM 拒绝执行,这是一层保护机制,防止误删还在被上层依赖的底层资源。

**为什么 RHCSA 真考 / 生产会用到:** "缩减删除存储"和"扩容存储"同样是 RHCSA 明确列出的技能;真实运维中"下线一块不再需要的存储"是常见操作,搞错顺序不仅会操作失败,排查报错信息本身也是一项排障能力。

**从最容易犯错的做法讲起:** 直接尝试 `pvremove` 一块还在被 VG 使用(VG 还没删)的物理卷——**本机实测证伪式验证**:这个操作会被明确拒绝,报错信息包含"belongs to"或"in use"这类提示,退出码是 5(不是常见的 1),这是 LVM 特意设计的保护机制,防止你在不知情的情况下破坏一个还在使用中的存储结构;必须严格按 LV→VG→PV 的顺序,一层层地"退役"。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 下线一台测试服务器上不再需要的存储:先确认 LV 上的数据已经不需要或已备份,`lvremove` 删除 LV,`vgremove` 删除已经空了的 VG,`pvremove` 最后清除 PV 标签让这块盘"退出"LVM 体系、变回一块可以另作他用的裸盘。

**可运行例子(完整走一遍拆卸流程,含"顺序错误会被拒绝"的现场验证):**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

dd if=/dev/zero of=/tmp/rhcsa03_rm_demo.img bs=1M count=300 2>/dev/null
loop9=$(losetup -f); losetup "$loop9" /tmp/rhcsa03_rm_demo.img
pvcreate -y "$loop9" >/dev/null 2>&1
vgcreate rhcsa03_rm_vg "$loop9" >/dev/null 2>&1
lvcreate -y -n rhcsa03_rm_lv -L 200M rhcsa03_rm_vg >/dev/null 2>&1

# 正确的第一步:删LV
lvremove -y rhcsa03_rm_vg/rhcsa03_rm_lv >/dev/null 2>&1
assert_eq "$?" "0"
assert_eq "$(lvs rhcsa03_rm_vg --noheadings 2>/dev/null | wc -l)" "0"

# 故意示范顺序错误:VG还在时直接pvremove,验证LVM会主动拒绝
wrong_order_output=$(pvremove -y "$loop9" 2>&1)
assert_eq "$?" "5"
echo "$wrong_order_output" | grep -qi "belongs to\|in use" && echo "OK: 顺序错误被正确拒绝,报错明确提示PV仍归属某个VG"

# 正确的第二步:VG里的LV都清空了,现在可以删VG
vgremove -y rhcsa03_rm_vg >/dev/null 2>&1
assert_eq "$?" "0"

# 正确的第三步:VG已经不存在了,现在pvremove能成功
pvremove -y "$loop9" >/dev/null 2>&1
assert_eq "$?" "0"

losetup -d "$loop9"; rm -f /tmp/rhcsa03_rm_demo.img
```
本机实测:全部断言输出 `OK`,包括"顺序错误确实被拒绝"这一现场验证——退出码 5、报错信息明确提示 PV 仍归属某个 VG。

**常见坑:** 顺序错误时 LVM 的报错信息**已经足够明确**(会直接说"这个 PV 还属于某个 VG"),但很多人遇到报错的第一反应是加 `-f`(force)强行绕过,而不是先读懂报错在说什么——正确的应对永远是先理解报错原因、按正确顺序重新操作,`-f` 强制选项存在潜在数据丢失风险,不应该作为"看不懂报错就先试试"的默认手段。

---

## 10. swap 分区/文件的创建与启用(`mkswap`,`swapon`,`/etc/fstab` 条目)

**命令/配置:**
```bash
mkswap /path/to/swapfile      # 把一个文件或分区初始化成swap格式
swapon /path/to/swapfile        # 立即启用(临时,重启后失效)
swapoff /path/to/swapfile        # 关闭
swapon --show                      # 查看当前生效的所有swap
# /etc/fstab里的永久条目: /swapfile none swap defaults 0 0
```

**一句话是什么:** swap 是当物理内存(RAM)不够用时,内核把暂时不活跃的内存页面换出到磁盘上的"备用内存空间"——可以是独立的分区,也可以是普通文件系统上的一个文件(现代 Linux 都支持后者,更灵活,不需要提前规划专用分区)。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"创建配置 swap 分区";内存吃紧时没有 swap 会导致 OOM Killer(内存溢出杀手)直接强制终止进程,配置合理的 swap 是系统稳定性的基本保障之一。

**从最容易犯错的做法讲起:** 创建 swap 文件时忘记设置严格的文件权限——swap 文件本质上会临时存放内存里的数据(可能包含敏感信息),如果权限过于宽松(比如默认的 644,其他用户可读),理论上存在信息泄露风险;规范做法是 `mkswap` 之前先 `chmod 600` 把权限收紧到只有 owner(通常是 root)可读写。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 云主机套餐内存较小但磁盘空间充裕,`dd` 创建一个 swap 文件、`chmod 600`、`mkswap`、`swapon` 立即生效缓解内存压力,再写进 `/etc/fstab` 保证重启后依然生效——这是没有独立 swap 分区(比如云盘按需分配、没有传统分区概念)时的标准补救方案。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

before_swap=$(free -m | grep Swap | awk '{print $2}')

dd if=/dev/zero of=/tmp/rhcsa03_swap.img bs=1M count=128 2>/dev/null
chmod 600 /tmp/rhcsa03_swap.img    # 收紧权限,swap文件可能临时存放内存中的敏感数据
mkswap /tmp/rhcsa03_swap.img >/dev/null 2>&1
assert_eq "$?" "0"

swapon /tmp/rhcsa03_swap.img
assert_ok test "$(free -m | grep Swap | awk '{print $2}')" -gt "$before_swap"
swapon --show 2>&1 | grep -q "rhcsa03_swap" && echo "OK: swapon --show 能看到新增的swap文件"

swapoff /tmp/rhcsa03_swap.img
assert_eq "$(free -m | grep Swap | awk '{print $2}')" "$before_swap"    # 关闭后swap总量恢复原状
rm -f /tmp/rhcsa03_swap.img
```
本机实测:全部检查点输出 `OK`。

**常见坑:** `swapon`/`swapoff` 是**临时**操作(等价于手动 `mount`/`umount`),不会持久化——只做了这两步,系统下次重启后 swap 配置会"消失",要长期生效必须额外写进 `/etc/fstab`,这和 04 类第 2-3 节讲过的"临时挂载 vs 永久挂载"是完全相同的设计模式,在 RHEL 系统管理里反复出现,理解一次能举一反三。

---

## 11. 磁盘配额基础(quota 工具链)

**命令/配置:**
```bash
tune2fs -O quota /dev/sdX1     # 现代方式:让ext4文件系统原生支持quota特性
mount -o usrquota,grpquota ...    # 挂载时启用配额(传统外部quota文件方式需要这个,现代原生方式不需要)
setquota -u username soft_block hard_block soft_inode hard_inode /mount/point   # 设置某用户的配额
quota -u username -v                # 查询某个用户的配额使用情况
repquota /mount/point                 # 报告整个文件系统上所有用户的配额使用情况
```

**一句话是什么:** 磁盘配额限制特定用户/组在某个文件系统上最多能使用多少存储空间(按块数)或最多能创建多少个文件(按 inode 数),防止单个用户/服务占用过多共享存储资源影响其他人;"soft limit"(软限制,超过后有宽限期警告)和"hard limit"(硬限制,绝对不能超过)是配额体系的两级阈值设计。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 明确要求"配置磁盘配额";多用户共享服务器(比如大学机房、共享开发服务器)上,防止某个用户的数据无限增长挤占其他人的可用空间,是最基本的资源治理需求。

**从最容易犯错的做法讲起:** **本机实测发现一个值得记录的现代化变化**:传统教程/旧版本 RHEL 上配置 quota 通常要求挂载时加 `usrquota,grpquota` 选项、依赖"外部 quota 文件"(`aquota.user`/`aquota.group`)——本机实测这套传统方式在 RHEL 10 上依然能工作,但 `quotacheck`/`quotaon` 会明确报出 deprecation 警告,提示"内核已经原生支持 ext4 quota 特性,建议用 `tune2fs -O quota` 代替外部 quota 文件方式"。用旧知识不假思索地照搬传统方式会得到能工作但已经过时的配置。

**真实场景例子(典型运维场景/RHCSA 考试场景):** 给一批学生账号统一设置配额上限,防止某个学生的作业数据占满整个共享磁盘:`tune2fs -O quota` 启用文件系统原生 quota 支持,逐个或批量 `setquota` 给账号设置合理的软硬限制,定期 `repquota` 检查整体使用情况,发现异常增长及时介入。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }
assert_ok() { "$@" || { echo "FAIL: command failed: $*" >&2; exit 1; }; echo "OK: $*"; }

assert_ok command -v quotacheck
assert_ok command -v edquota
assert_ok command -v repquota

dd if=/dev/zero of=/tmp/rhcsa03_quota.img bs=1M count=200 2>/dev/null
loopq=$(losetup -f); losetup "$loopq" /tmp/rhcsa03_quota.img
mkfs.ext4 -q "$loopq"
tune2fs -O quota "$loopq" >/dev/null 2>&1    # 现代方式:文件系统原生支持,不用外部quota文件
assert_eq "$?" "0"

mkdir -p /mnt/rhcsa03_quota
mount "$loopq" /mnt/rhcsa03_quota
useradd -M -s /sbin/nologin rhcsa03_quota_user 2>/dev/null
setquota -u rhcsa03_quota_user 10240 12288 0 0 /mnt/rhcsa03_quota

# quota -u 直接查询,不依赖用户是否已经在文件系统上创建过文件
quota_check=$(quota -u rhcsa03_quota_user -v 2>&1)
echo "$quota_check" | grep -q "10240" && echo "OK: quota -u 确认限额设置真实生效"

umount /mnt/rhcsa03_quota
losetup -d "$loopq"; rm -f /tmp/rhcsa03_quota.img
userdel rhcsa03_quota_user 2>/dev/null
rmdir /mnt/rhcsa03_quota
```
本机实测:全部检查点输出 `OK`。

**常见坑:** `repquota` 的默认报告**只显示在这个文件系统上已经有实际文件/目录记录的用户**——本机实测确认:刚 `setquota` 设置好限额、但这个用户还从未在这个文件系统上创建过任何文件时,`repquota` 的输出列表里根本不会出现这个用户,容易被误判成"配置没生效";用 `quota -u username` 直接查询指定用户,不受这个限制,是更可靠的验证方式,该用户一旦真正创建了文件,才会出现在 `repquota` 的汇总报告里。

---

## 12. 用 loop device 模拟磁盘做实验(`dd`+`losetup`,本系列验证专用技巧)

**命令/配置:**
```bash
dd if=/dev/zero of=disk.img bs=1M count=N     # 创建一个N MB的空白镜像文件(也可用truncate -s更快)
losetup -f                                       # 查找一个空闲的loop设备名(不实际关联)
losetup /dev/loopX disk.img                        # 把镜像文件关联到指定loop设备,此后当成真实磁盘操作
losetup -a                                           # 查看当前所有已关联的loop设备
losetup -j disk.img                                    # 反查:这个镜像文件关联的是哪个loop设备
losetup -d /dev/loopX                                    # 解除关联,不再需要这块"虚拟磁盘"时清理
```

**一句话是什么:** loop device(回环设备)让一个普通文件在内核层面"伪装"成一块真实的块设备,可以像对待真实磁盘一样对它做分区、`pvcreate`、`mkfs` 等任何存储操作,是本篇乃至整个 04 类反复使用的核心实验技巧——不需要真实的多余物理磁盘,就能安全地练习几乎所有存储管理操作,操作全部发生在一个可以随时删除的普通文件里,不会影响宿主系统的真实磁盘。

**为什么 RHCSA 真考 / 生产会用到:** RHCSA 考试本身在真实(通常是虚拟机)环境里操作真实的虚拟磁盘,不需要 loop device 这个技巧;但对**学习和实验**存储管理操作来说,loop device 是最安全、最容易获取、最方便重复试错的练习环境,不需要额外挂载真实硬盘就能反复练习分区/LVM 全流程,这也是本系列贯穿 03/04 两类反复采用这个技巧的原因。

**从最容易犯错的做法讲起:** 做完实验之后忘记 `losetup -d` 解除关联就直接删除了背后的镜像文件——这会留下一个"关联着已经不存在的文件"的僵尸 loop 设备(`losetup -a` 会显示类似 `(deleted)` 的标记),占用了一个 loop 设备编号却没有实际用途,长期不清理会导致后续实验 `losetup -f` 分配到的设备编号越来越靠后,排查环境状态时容易困惑;正确顺序永远是先 `losetup -d` 解除关联,再删除背后的镜像文件。

**真实场景例子(这里就是元层面的场景——这个技巧本身贯穿本篇 12 个知识点的验证过程):** 本篇第 1-11 节的每一个可运行例子,都是先用 `dd`/`truncate` 创建一个几百 MB 的镜像文件、`losetup` 关联成虚拟磁盘,做完分区/LVM/swap/quota 各种实验后,再依次 `losetup -d` 解除关联、删除镜像文件,全程不触碰宿主系统真实的磁盘设备,这正是能够放心大胆做破坏性存储实验(格式化、删分区、mklabel 清空分区表)而不用担心搞坏真实系统的原因。

**可运行例子:**
```bash
assert_eq() { [ "$1" = "$2" ] || { echo "FAIL: expected [$2] got [$1]" >&2; exit 1; }; echo "OK: $1 == $2"; }

dd if=/dev/zero of=/tmp/rhcsa03_meta.img bs=1M count=50 2>/dev/null
assert_eq "$(stat -c '%s' /tmp/rhcsa03_meta.img)" "52428800"    # 50*1024*1024,dd精确创建了预期大小

loopm=$(losetup -f)
losetup "$loopm" /tmp/rhcsa03_meta.img
losetup -a | grep -q "$loopm" && echo "OK: losetup -a 能看到刚建立的关联"
losetup -j /tmp/rhcsa03_meta.img | grep -q "$loopm" && echo "OK: losetup -j 能反查这个镜像文件对应哪个loop设备"

losetup -d "$loopm"
# 轮询重试确认已解除关联,不用固定sleep一次就检查——本机实测发现losetup -d返回后,
# 内核层面完全清理关联可能存在极短暂的时间差,立即查一次偶尔会撞上这个竞态窗口
detached=0
for i in $(seq 1 10); do
    [ "$(losetup -a | grep -c "$loopm")" = "0" ] && { detached=1; break; }
    sleep 0.2
done
assert_eq "$detached" "1"    # 正确顺序:先解除关联,后面才删文件
rm -f /tmp/rhcsa03_meta.img
```
本机实测:全部检查点输出 `OK`。

**常见坑:**
1. **本机验证时现场发现的时序问题**:`losetup -d` 命令返回(退出码 0)之后,内核层面完全清理这个设备关联可能存在极短暂的时间差——本机实测过連续执行"`losetup -d` 后立即查 `losetup -a`"这个检查,少数情况下会得到"设备依然存在"的结果(多次重复执行,失败概率不算低),换成轮询重试(检查几次、每次间隔一小段时间)后完全稳定。这提醒一个更普遍的道理:凡是涉及内核/系统资源状态变化的检查,"命令返回成功"不严格等价于"状态已经对下一条查询命令可见",写自动化脚本核实这类状态变化时,轮询重试比"改完立即查一次"更可靠——这和 [02 类第 11 节](02-process-and-boot.md) `systemd timer` 的调度延迟,是同一类问题的不同表现形式。
2. `dd if=/dev/zero of=... bs=1M count=N` 创建镜像文件时会**真实写入** N MB 的零字节(即便内容毫无意义,也要实打实地花时间/占用磁盘空间写完);如果只是需要一个"看起来有这么大、但暂时不需要真实占用磁盘空间"的稀疏文件(比如本篇 04 类第 13 节演示 VDO 时用到的 5GB 镜像),用 `truncate -s SIZE file` 创建**稀疏文件**要快得多、也不会立刻消耗对应的真实磁盘空间——两个工具的选择取决于是否需要文件内容"确实是全零"这个前提,大多数分区/LVM/文件系统实验并不关心初始内容是不是真的全零,`truncate` 通常是更高效的选择。

---

*本篇完成:2026-07-11,12 个知识点。验证环境:Rocky Linux 10.2(WSL2)。全部代码块真实跑通验证,含多处现场发现的真实细节:gdisk默认未装、parted分区终点不能设成磁盘绝对容量(GPT尾部保留区域)需要用100%写法、pvremove顺序错误时退出码是5、quota在RHEL10已演进为ext4原生特性(tune2fs -O quota)而非传统外部文件方式、repquota默认只显示已有文件记录的用户。*