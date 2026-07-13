"""跑全部 Networking Mastery 模块的 _self_test():
deep_common 自检 + 7 个 tier 模块自检 + 总聚合校验。
"""
from __future__ import annotations

import importlib
import os
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

MODULES = [
    "deep_common",
    # tier1 浅:分层模型/寻址基础与TCP-UDP-HTTP基础
    "tier1_shallow.dp_net_osi_tcpip_addressing_basics",
    "tier1_shallow.dp_net_tcp_udp_http_basics",
    # tier2 深:TCP拥塞控制/HTTPS安全/负载均衡CDN深水
    "tier2_deep.dp_net_tcp_congestion_control_deep",
    "tier2_deep.dp_net_https_security_deep",
    "tier2_deep.dp_net_load_balancing_cdn_deep",
    # tier3 社招级别:架构选型与生产故障判断
    "tier3_social_hire.sc_net_architecture_selection_judgment",
    "tier3_social_hire.sc_net_production_troubleshooting_judgment",
    # 总聚合校验
    "networking_mastery",
]


def main() -> int:
    passed = 0
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            mod._self_test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"=== {passed}/{len(MODULES)} modules passed ===")
    return 0 if passed == len(MODULES) else 1


if __name__ == "__main__":
    sys.exit(main())
