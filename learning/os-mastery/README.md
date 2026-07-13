# OS Mastery —— 操作系统专题(浅→深→社招级别)

> 定位:老手要求的五个 CS 基础专题(软件工程/设计模式/数据库/网络/OS)队列的第五个,也是最后一个,组织方式与前四个专题一致——`DeepPoint`/`ScenarioPoint` 都带 `explain` 系统性讲解字段,因为用户完全没系统学过操作系统。

## 组织轴:难度分层

| 层 | 目录 | 定位 | 点数 | 类型 |
|---|---|---|:--:|:--:|
| 浅(Tier 1) | `src/tier1_shallow/` | 进程与线程基础、内存管理基础 | 35 | DeepPoint |
| 深(Tier 2) | `src/tier2_deep/` | 调度与同步深水、文件系统与IO深水、虚拟化与容器深水 | 45 | DeepPoint |
| 社招级别(Tier 3) | `src/tier3_social_hire/` | 生产性能问题定位判断、系统设计中的OS层判断 | 28 | ScenarioPoint |

**合计108点(80个DeepPoint + 28个ScenarioPoint)**。

## 内容总览

| 文件 | 覆盖内容 | 点数 |
|---|---|:--:|
| `tier1_shallow/dp_os_process_thread_basics.py` | 进程状态转换、PCB、上下文切换开销、并发vs并行、用户级/内核级线程、协程、IPC四种方式、孤儿僵尸进程、fork与COW | 18 |
| `tier1_shallow/dp_os_memory_management_basics.py` | 虚拟内存核心思想、分页分段、多级页表、TLB、缺页中断、页面置换算法、进程内存布局、内存泄漏vs溢出 | 17 |
| `tier2_deep/dp_os_scheduling_synchronization_deep.py` | CPU调度算法对比、Linux CFS、锁/信号量/条件变量、死锁四条件、死锁预防-避免-检测、饥饿与优先级反转 | 15 |
| `tier2_deep/dp_os_filesystem_io_deep.py` | inode机制、硬链接vs符号链接、select-poll-epoll演进、零拷贝、Page Cache、日志文件系统、顺序IO vs随机IO | 15 |
| `tier2_deep/dp_os_virtualization_container_deep.py` | VM vs容器本质区别、Hypervisor类型、Linux namespace、cgroups、容器镜像分层、容器网络模型、内核态用户态切换 | 15 |
| `tier3_social_hire/sc_os_performance_troubleshooting_judgment.py` | CPU飙升定位、内存泄漏排查、IO瓶颈判断、容器OOMKilled/CPU throttling判断、swap使用判断 | 14 |
| `tier3_social_hire/sc_os_system_design_os_layer_judgment.py` | 高并发IO模型选型、cgroups多租户隔离、内核参数调优边界、大页内存、CPU亲和性、系统调用开销优化 | 14 |

## 新增:本科课件式完整教材

老手进一步反馈:即便加了 `explain` 字段,108个知识点的追问链结构本质上仍然是"直接上面试",对完全没系统学过的人不够。因此新增 `lectures/01-textbook.md`——一份约20万字符、七章的完整教材式讲义,仿照《操作系统概念》(恐龙书)写法,按进程与线程基础→内存管理基础→调度与同步深水→文件系统与IO深水→虚拟化与容器深水→生产性能问题定位判断→系统设计中的OS层判断的顺序,用连贯的教科书叙述系统教会每个主题,不是知识点罗列或追问链。**建议学习顺序:先读完 `01-textbook.md` 建立系统认知,再回到下面的 DeepPoint/ScenarioPoint 用追问链和场景判断检验"是否真的能被追问住"。**

## 数据结构

```python
import sys
sys.path.insert(0, "learning/os-mastery/src")
from os_mastery import ALL_DP, ALL_SP, TIERS
from deep_common import drill, grade_chain, grade_scenario
```

## 环境与测试

```bash
python learning/os-mastery/src/tests/test_all.py
```

9个模块(deep_common + 7个内容文件 + 总聚合)全部通过 `_self_test()`。

## 诚实说明

内容全部基于经典操作系统教材共识(《操作系统概念》(Silberschatz)关于进程/内存管理/调度的标准定义、Linux内核官方文档关于CFS调度器/epoll/namespace/cgroups的实现机制、火星探路者任务优先级反转这一真实历史案例)。`real_world_link` 字段全部留空——不编造本地文件路径或不确定的公司案例。

## 队列状态:五个CS基础专题至此全部完成

- Software Engineering: 108点,tag `software-engineering-mastery-v1`
- Design Patterns: 108点,tag `design-patterns-mastery-v1`
- Database: 108点,tag `database-mastery-v1`
- Networking: 108点,tag `networking-mastery-v1`
- OS: 108点,tag `os-mastery-v1`

五个专题合计540点,加上此前已完成的LLM/面试专题体系(1115点),全仓库面试题库grand total 1655点,十三个包全局id/trigger程序化验证零冲突。老手2026-07-13提出的"软件工程/设计模式/数据库/网络/OS各做100+知识点系统化学习且讲仔细"要求至此完整交付。
