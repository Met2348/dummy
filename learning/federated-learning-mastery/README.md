# Federated Learning Mastery —— 联邦学习专题(浅→深→社招级别)

> 定位：老手要求 Foundation Model、Federated Learning、Diffusion 三个"enormous"级别专题各做100+点(浅到深到社招级别)，这是队列第二个。联邦学习此前在整个题库(898点)里完全没有覆盖，是一个全新领域，不存在需要规避的重复内容。

## 组织轴：难度分层(与 `foundation-model-mastery` 一致)

| 层 | 目录 | 定位 | 点数 | 类型 |
|---|---|---|:--:|:--:|
| 浅(Tier 1) | `src/tier1_shallow/` | FL核心概念与FedAvg机制、系统角色与部署形态 | 35 | DeepPoint |
| 深(Tier 2) | `src/tier2_deep/` | Non-IID挑战与个性化、通信效率与压缩、隐私与安全深水 | 46 | DeepPoint |
| 社招级别(Tier 3) | `src/tier3_social_hire/` | 生产部署与激励机制判断、FL for LLM时代前沿判断 | 28 | ScenarioPoint |

**合计109点(81个DeepPoint + 28个ScenarioPoint)**，练法建议按 Tier 1→2→3 顺序推进。

## 内容总览

| 文件 | 覆盖内容 | 点数 |
|---|---|:--:|
| `tier1_shallow/dp_fl_basics_fedavg.py` | FL存在的根本原因(隐私法规/带宽成本/数据所有权)、FedAvg算法机制、通信轮定义、cross-device vs cross-silo、client-server架构与信任假设、IID假设 | 18 |
| `tier1_shallow/dp_fl_system_deployment.py` | Client selection机制、同步vs异步FL、straggler应对、设备可用性窗口、hierarchical FL、真实生产案例 | 17 |
| `tier2_deep/dp_fl_noniid_personalization.py` | Non-IID的几种类型、client drift机制、FedProx/SCAFFOLD/FedBN、个性化联邦学习 | 16 |
| `tier2_deep/dp_fl_communication_efficiency.py` | 通信瓶颈本质、量化/稀疏化压缩、结构化更新、通信轮次与收敛稳定性权衡、联邦蒸馏 | 15 |
| `tier2_deep/dp_fl_privacy_security.py` | 梯度反演攻击(DLG)、差分隐私((ε,δ)/user-level vs sample-level)、安全聚合协议、投毒/后门攻击、Byzantine-robust聚合(Krum/trimmed mean) | 15 |
| `tier3_social_hire/sc_fl_production_incentive_judgment.py` | Client selection策略判断、激励机制设计、straggler容忍窗口权衡、多地区法规合规、模型质量下降归因、FL是否值得做的判断 | 14 |
| `tier3_social_hire/sc_fl_llm_frontier_judgment.py` | 联邦微调LLM可行性、FedLoRA聚合设计、联邦RLHF/对齐、FL不适合LLM预训练阶段的判断、边缘设备部署约束、2026年这个交叉领域的真实成熟度 | 14 |

与已有七子包(全局id/trigger程序化验证零冲突)合计1007点。

## 数据结构：复用已验证的 DeepPoint + ScenarioPoint

`src/deep_common.py` 是 `onsite-mastery`/`foundation-model-mastery` 同名文件的独立副本(各 mastery track 自成一体，不跨track相互import)。

```python
import sys
sys.path.insert(0, "learning/federated-learning-mastery/src")
from federated_learning_mastery import ALL_DP, ALL_SP, TIERS
from deep_common import drill, grade_chain, grade_scenario
```

## 环境与测试

```bash
python learning/federated-learning-mastery/src/tests/test_all.py
```

9个模块(deep_common + 7个内容文件 + 总聚合)全部通过 `_self_test()`。

## 诚实说明

7个文件里 `tier3_social_hire/sc_fl_production_incentive_judgment.py` 因派发agent两次遇到429/503服务端错误、且未在磁盘留下文件，由我直接手写完成(同一模板、同一自检严格度)。其余6个文件均由agent自主WebSearch验证后完成，`real_world_link`字段逐一核实过——非空的引用均为可验证的论文/arXiv编号，不存在编造的本地文件路径。
