"""跑全部 Security & Crypto Mastery 模块的 _self_test():
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
    # tier1 浅:密码学基础与认证访问控制基础
    "tier1_shallow.dp_sec_cryptography_basics",
    "tier1_shallow.dp_sec_auth_access_control_basics",
    # tier2 深:常见漏洞/网络安全协议/应用安全工程深水
    "tier2_deep.dp_sec_common_vulnerabilities_deep",
    "tier2_deep.dp_sec_network_security_protocols_deep",
    "tier2_deep.dp_sec_application_security_engineering_deep",
    # tier3 社招级别:安全事件应急与安全架构选型判断
    "tier3_social_hire.sc_sec_incident_response_judgment",
    "tier3_social_hire.sc_sec_security_architecture_selection_judgment",
    # 总聚合校验
    "security_crypto_mastery",
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
