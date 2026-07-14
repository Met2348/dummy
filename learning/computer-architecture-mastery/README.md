# Computer Architecture Mastery —— 计算机体系结构专题(浅→深→社招级别)

> 定位:老师在CS基础五件套完成后要求"仔细思考和科班CS本科相比还差哪些能力",梳理出六个理论必修课空白,这是第四个。这里的"计算机体系结构"是**通用CPU体系结构**——指令集/流水线/乱序执行/缓存一致性,和仓库里已有的 `gpu-architecture`/`cuda-essentials`/`kernel-engineering`(GPU专用架构)是完全不同的细分领域,执行前已确认无内容重叠。

## 和其他专题的关系

存储层次一章的cache基础是os-mastery专题里虚拟内存/分页机制的硬件基础;多核缓存一致性一章的false sharing/内存一致性模型和software-engineering-mastery专题里并发编程的正确性直接相关;性能瓶颈判断一章呼应os-mastery专题的"生产性能问题定位判断"但视角更偏硬件底层(cache/分支预测/流水线,而不是进程调度/IO)。

## 组织轴:难度分层

| 层 | 目录 | 定位 | 点数 | 类型 |
|---|---|---|:--:|:--:|
| 浅(Tier 1) | `src/tier1_shallow/` | 指令集与数据通路(RISC vs CISC/寻址模式/单周期多周期/IF-ID-EX-MEM-WB)、存储层次基础(局部性原理/cache基础/写策略) | 35 | DeepPoint |
| 深(Tier 2) | `src/tier2_deep/` | 流水线与冒险深水(结构数据控制冒险/forwarding/分支预测)、乱序执行与高级微架构深水(Tomasulo/寄存器重命名/ROB/推测执行)、多核与缓存一致性深水(MESI/一致性协议/内存一致性模型/false sharing) | 45 | DeepPoint |
| 社招级别(Tier 3) | `src/tier3_social_hire/` | 性能瓶颈定位判断(CPU-bound/memory-bound/cache miss)、硬件感知的软件优化判断(数据局部性/分支预测友好/false sharing规避) | 28 | ScenarioPoint |

**合计108点(80个DeepPoint + 28个ScenarioPoint)**。

## 内容总览

| 文件 | 覆盖内容 | 点数 |
|---|---|:--:|
| `tier1_shallow/dp_arch_isa_datapath_basics.py` | ISA/RISC vs CISC、寻址模式、数据通路组成、单周期多周期、IF-ID-EX-MEM-WB | 18 |
| `tier1_shallow/dp_arch_memory_hierarchy_basics.py` | 存储层次、局部性原理、cache基础(直接映射/组相联)、写策略、替换策略 | 17 |
| `tier2_deep/dp_arch_pipelining_hazards_deep.py` | 流水线原理、结构/数据/控制冒险、forwarding、分支预测、BTB | 15 |
| `tier2_deep/dp_arch_out_of_order_execution_deep.py` | 超标量、乱序执行、寄存器重命名、Tomasulo算法、ROB、推测执行 | 15 |
| `tier2_deep/dp_arch_multicore_cache_coherence_deep.py` | MESI协议、目录协议、内存一致性模型、false sharing | 15 |
| `tier3_social_hire/sc_arch_performance_bottleneck_judgment.py` | 性能瓶颈定位判断(CPU-bound/memory-bound/cache miss/分支预测) | 14 |
| `tier3_social_hire/sc_arch_hardware_aware_optimization_judgment.py` | 硬件感知优化判断(数据局部性/false sharing/SIMD/可移植性) | 14 |

## 新增:本科课件式完整教材

沿用CS基础五件套的做法,新增 `lectures/01-textbook.md`——七章连贯教科书叙述,按指令集数据通路→存储层次→流水线冒险→乱序执行→多核缓存一致性→性能瓶颈判断→硬件感知优化判断的顺序系统教学。**建议先读完 `01-textbook.md` 再做下面的追问链/场景判断自测。**

## 数据结构:DeepPoint + ScenarioPoint(含 explain 字段)

```python
import sys
sys.path.insert(0, "learning/computer-architecture-mastery/src")
from computer_architecture_mastery import ALL_DP, ALL_SP, TIERS
from deep_common import drill, grade_chain, grade_scenario
```

## 环境与测试

```bash
python learning/computer-architecture-mastery/src/tests/test_all.py
```

9个模块(deep_common + 7个内容文件 + 总聚合)全部通过 `_self_test()`。

## 诚实说明

内容全部是经典计算机体系结构教材共识(Patterson-Hennessy《计算机组成与设计》/Hennessy-Patterson《计算机体系结构:量化研究方法》体系里的标准表述),不需要追赶前沿论文,但仍要求核实具体术语和机制描述的准确性。`real_world_link` 字段全部留空。

至此,老师要求的六个CS科班理论专题第四个完成,后续两个(编译原理/安全密码学基础)陆续推进。
