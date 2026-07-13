# Networking Mastery —— 计算机网络专题(浅→深→社招级别)

> 定位:老手要求的五个 CS 基础专题(软件工程/设计模式/数据库/网络/OS)队列的第四个,组织方式与前三个专题一致——`DeepPoint`/`ScenarioPoint` 都带 `explain` 系统性讲解字段,因为用户完全没系统学过计算机网络。

## 与仓库已有 `cluster-networking/` 的边界

仓库里已有 `cluster-networking/`(GPU集群NCCL/allreduce/fat-tree拓扑,面向分布式训练的网络工程),这是完全不同的细分领域——那是"大规模GPU集群互联"的专门知识,本track是**通用计算机网络基础**(OSI/TCP-IP/HTTP/负载均衡),两者没有内容重叠。

## 组织轴:难度分层

| 层 | 目录 | 定位 | 点数 | 类型 |
|---|---|---|:--:|:--:|
| 浅(Tier 1) | `src/tier1_shallow/` | 分层模型与寻址基础、TCP-UDP与HTTP基础 | 35 | DeepPoint |
| 深(Tier 2) | `src/tier2_deep/` | TCP拥塞控制深水、HTTPS与Web安全深水、负载均衡与CDN深水 | 45 | DeepPoint |
| 社招级别(Tier 3) | `src/tier3_social_hire/` | 网络架构选型判断、生产网络故障定位判断 | 28 | ScenarioPoint |

**合计108点(80个DeepPoint + 28个ScenarioPoint)**。

## 内容总览

| 文件 | 覆盖内容 | 点数 |
|---|---|:--:|
| `tier1_shallow/dp_net_osi_tcpip_addressing_basics.py` | OSI七层与TCP-IP四层、IP地址与子网掩码、NAT、MAC vs IP、ARP、路由、DNS解析流程 | 18 |
| `tier1_shallow/dp_net_tcp_udp_http_basics.py` | TCP三次握手四次挥手、TIME_WAIT、滑动窗口、TCP vs UDP、HTTP/1.1到HTTP/2演进 | 17 |
| `tier2_deep/dp_net_tcp_congestion_control_deep.py` | cwnd/rwnd、慢启动、拥塞避免、快重传快恢复、CUBIC/BBR、Nagle算法 | 15 |
| `tier2_deep/dp_net_https_security_deep.py` | TLS握手、证书链、TLS1.3改进、对称非对称加密、HTTP/2多路复用、HTTP/3-QUIC、XSS/CSRF | 15 |
| `tier2_deep/dp_net_load_balancing_cdn_deep.py` | 四层七层负载均衡、负载均衡算法、一致性哈希、CDN缓存策略、DNS负载均衡、服务发现、反向代理 | 15 |
| `tier3_social_hire/sc_net_architecture_selection_judgment.py` | Service Mesh vs API Gateway、跨机房网络设计、CDN选型判断 | 14 |
| `tier3_social_hire/sc_net_production_troubleshooting_judgment.py` | 延迟排查分层思路、抓包分析、连接耗尽、DNS异常排查、脑裂判断 | 14 |

## 数据结构

```python
import sys
sys.path.insert(0, "learning/networking-mastery/src")
from networking_mastery import ALL_DP, ALL_SP, TIERS
from deep_common import drill, grade_chain, grade_scenario
```

## 环境与测试

```bash
python learning/networking-mastery/src/tests/test_all.py
```

9个模块(deep_common + 7个内容文件 + 总聚合)全部通过 `_self_test()`。

## 诚实说明

内容全部基于经典网络教材共识(《计算机网络:自顶向下方法》关于TCP拥塞控制/HTTP演进的标准阐述、RFC文档关于TCP三次握手四次挥手的规范定义、TLS1.3 RFC 8446关于握手流程的改进)。`real_world_link` 字段全部留空——不编造本地文件路径或不确定的公司案例。执行过程中修正过一处格式不一致(T1-2文件CAT常量全角冒号统一为半角),`dp_net_load_balancing_cdn_deep.py` 因两次agent派发均未能成功写出文件(在写文件前中途结束但未报错),最终由我直接手写完成并通过自检。

至此,五个CS基础专题队列(软件工程✓/设计模式✓/数据库✓/网络✓/OS)完成四个,最后一个(操作系统)陆续推进。
