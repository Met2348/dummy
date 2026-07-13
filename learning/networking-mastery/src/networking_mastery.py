"""Networking Mastery 总聚合：浅→深→社招级别三层，合计7个文件108点。

老手要求的五个 CS 基础专题(软件工程/设计模式/数据库/网络/OS)队列的第四个，组织方式沿用前三个
专题已验证的"讲解+追问链/场景判断"混合格式：

- tier1(浅)：分层模型与寻址基础(OSI/TCP-IP/IP地址/NAT/ARP/DNS解析)、TCP-UDP与HTTP基础(三次
  握手四次挥手/滑动窗口/HTTP1.1到HTTP2的演进)
- tier2(深)：TCP拥塞控制深水(慢启动/拥塞避免/快重传快恢复/Nagle算法)、HTTPS与Web安全深水(TLS
  握手/证书链/HTTP2多路复用/XSS-CSRF)、负载均衡与CDN深水(四层七层负载均衡/一致性哈希/CDN缓存
  策略/服务发现)
- tier3(社招级别)：网络架构选型判断(Service Mesh vs API Gateway/跨机房设计)、生产网络故障定位
  判断(延迟排查/丢包定位/连接耗尽)——无标准答案的资深判断

与仓库已有的 `cluster-networking/`(GPU集群NCCL/allreduce/fat-tree，分布式训练网络)是完全不同的
细分领域，本track是通用计算机网络基础，没有内容重叠。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deep_common import categories, grade_chain, grade_scenario  # noqa: E402

from tier1_shallow.dp_net_osi_tcpip_addressing_basics import BANK as _t1_osi  # noqa: E402
from tier1_shallow.dp_net_tcp_udp_http_basics import BANK as _t1_http  # noqa: E402
from tier2_deep.dp_net_tcp_congestion_control_deep import BANK as _t2_cc  # noqa: E402
from tier2_deep.dp_net_https_security_deep import BANK as _t2_sec  # noqa: E402
from tier2_deep.dp_net_load_balancing_cdn_deep import BANK as _t2_lb  # noqa: E402
from tier3_social_hire.sc_net_architecture_selection_judgment import BANK as _t3_arch  # noqa: E402
from tier3_social_hire.sc_net_production_troubleshooting_judgment import BANK as _t3_ops  # noqa: E402

ALL_DP = list(_t1_osi) + list(_t1_http) + list(_t2_cc) + list(_t2_sec) + list(_t2_lb)
ALL_SP = list(_t3_arch) + list(_t3_ops)

TIERS: tuple[dict, ...] = (
    {"n": 1, "name": "shallow", "label": "浅:分层模型/寻址与TCP-UDP-HTTP基础", "kind": "dp",
     "modules": ({"name": "net_osi_tcpip_addressing_basics", "cat": "网络基础一:分层模型与寻址基础", "bank": _t1_osi},
                 {"name": "net_tcp_udp_http_basics", "cat": "网络基础二:TCP-UDP与HTTP基础", "bank": _t1_http})},
    {"n": 2, "name": "deep", "label": "深:TCP拥塞控制/HTTPS安全/负载均衡CDN深水", "kind": "dp",
     "modules": ({"name": "net_tcp_congestion_control_deep", "cat": "网络深水一:TCP拥塞控制深水", "bank": _t2_cc},
                 {"name": "net_https_security_deep", "cat": "网络深水二:HTTPS与Web安全深水", "bank": _t2_sec},
                 {"name": "net_load_balancing_cdn_deep", "cat": "网络深水三:负载均衡与CDN深水", "bank": _t2_lb})},
    {"n": 3, "name": "social_hire", "label": "社招级别:架构选型与生产故障判断", "kind": "sp",
     "modules": ({"name": "net_architecture_selection_judgment", "cat": "网络社招级别一:网络架构选型判断", "bank": _t3_arch},
                 {"name": "net_production_troubleshooting_judgment", "cat": "网络社招级别二:生产网络故障定位判断", "bank": _t3_ops})},
)


def _self_test() -> None:
    assert len(ALL_DP) >= 70, len(ALL_DP)
    assert len(ALL_SP) >= 25, len(ALL_SP)
    assert len(ALL_DP) + len(ALL_SP) >= 100, len(ALL_DP) + len(ALL_SP)
    assert len(categories(ALL_DP)) == 5, len(categories(ALL_DP))
    assert len(categories(ALL_SP)) == 2, len(categories(ALL_SP))

    dp_ids = [dp.id for dp in ALL_DP]
    sp_ids = [sp.id for sp in ALL_SP]
    assert len(dp_ids) == len(set(dp_ids)), "networking-mastery ALL_DP 内存在重复 id"
    assert len(sp_ids) == len(set(sp_ids)), "networking-mastery ALL_SP 内存在重复 id"
    assert all(i.startswith("dp-net-") for i in dp_ids), "存在不以dp-net-开头的DeepPoint id"
    assert all(i.startswith("sc-net-") for i in sp_ids), "存在不以sc-net-开头的ScenarioPoint id"

    dp_triggers = [dp.trigger for dp in ALL_DP]
    sp_triggers = [sp.trigger for sp in ALL_SP]
    assert len(dp_triggers) == len(set(dp_triggers)), "networking-mastery ALL_DP 内存在重复 trigger"
    assert len(sp_triggers) == len(set(sp_triggers)), "networking-mastery ALL_SP 内存在重复 trigger"

    assert all(len(dp.explain) >= 100 for dp in ALL_DP), "存在explain过短的DeepPoint"
    assert all(len(sp.explain) >= 100 for sp in ALL_SP), "存在explain过短的ScenarioPoint"

    assert len(TIERS) == 3, len(TIERS)
    assert [t["n"] for t in TIERS] == [1, 2, 3], "TIERS 顺序不连续"
    tier_bank_total = sum(len(m["bank"]) for t in TIERS for m in t["modules"])
    assert tier_bank_total == len(ALL_DP) + len(ALL_SP), "TIERS 里的bank总数与ALL_DP+ALL_SP不一致"

    print(
        f"[PASS] networking_mastery: 3层(浅/深/社招级别)7个模块 汇总完整,"
        f"合计{len(ALL_DP)}个DeepPoint + {len(ALL_SP)}个ScenarioPoint = {len(ALL_DP) + len(ALL_SP)}点"
    )


if __name__ == "__main__":
    _self_test()
