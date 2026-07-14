"""Security & Crypto Mastery 总聚合：浅→深→社招级别三层，合计7个文件108点。

老师"仔细思考和科班CS本科相比还差哪些能力"要求补齐的六个新专题(离散数学/计算理论/算法理论
证明向/计算机体系结构/编译原理/安全密码学基础)第六个也是最后一个。这是**通用信息安全和
密码学基础**,和仓库里已有的 `safety-defense`/`red-team-jailbreak`(LLM安全,面向大模型
越狱/对齐攻防)是完全不同的细分领域。`explain` 字段承载"讲仔细"的系统性教学讲解，组织方式
沿用"难度分层"轴：

- tier1(浅)：密码学基础(对称非对称加密/哈希与数字签名/密钥交换/前向保密)、认证与访问控制
  基础(认证三要素/OAuth2.0/Kerberos/RBAC-ABAC/会话管理)
- tier2(深)：常见漏洞深水(缓冲区溢出/SQL注入/XSS-CSRF/反序列化/竞态条件TOCTOU)、网络安全
  协议深水(TLS 1.3/PKI证书链/VPN/防火墙IDS-IPS)、应用安全工程深水(Secure SDLC/STRIDE
  威胁建模/secrets管理/供应链安全SBOM)
- tier3(社招级别)：安全事件应急判断(数据泄露响应/漏洞应急/日志取证)、安全架构选型判断
  (零信任vs边界防御/加密方案选择)——无标准答案的资深判断
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deep_common import categories, grade_chain, grade_scenario  # noqa: E402

from tier1_shallow.dp_sec_cryptography_basics import BANK as _t1_crypto  # noqa: E402
from tier1_shallow.dp_sec_auth_access_control_basics import BANK as _t1_auth  # noqa: E402
from tier2_deep.dp_sec_common_vulnerabilities_deep import BANK as _t2_vuln  # noqa: E402
from tier2_deep.dp_sec_network_security_protocols_deep import BANK as _t2_net  # noqa: E402
from tier2_deep.dp_sec_application_security_engineering_deep import BANK as _t2_eng  # noqa: E402
from tier3_social_hire.sc_sec_incident_response_judgment import BANK as _t3_ir  # noqa: E402
from tier3_social_hire.sc_sec_security_architecture_selection_judgment import BANK as _t3_arch  # noqa: E402

ALL_DP = list(_t1_crypto) + list(_t1_auth) + list(_t2_vuln) + list(_t2_net) + list(_t2_eng)
ALL_SP = list(_t3_ir) + list(_t3_arch)

TIERS: tuple[dict, ...] = (
    {"n": 1, "name": "shallow", "label": "浅:密码学基础与认证访问控制基础", "kind": "dp",
     "modules": ({"name": "sec_cryptography_basics", "cat": "安全与密码学基础一:密码学基础", "bank": _t1_crypto},
                 {"name": "sec_auth_access_control_basics", "cat": "安全与密码学基础二:认证与访问控制基础", "bank": _t1_auth})},
    {"n": 2, "name": "deep", "label": "深:常见漏洞/网络安全协议/应用安全工程深水", "kind": "dp",
     "modules": ({"name": "sec_common_vulnerabilities_deep", "cat": "安全与密码学深水一:常见漏洞深水", "bank": _t2_vuln},
                 {"name": "sec_network_security_protocols_deep", "cat": "安全与密码学深水二:网络安全协议深水", "bank": _t2_net},
                 {"name": "sec_application_security_engineering_deep", "cat": "安全与密码学深水三:应用安全工程深水", "bank": _t2_eng})},
    {"n": 3, "name": "social_hire", "label": "社招级别:安全事件应急与安全架构选型判断", "kind": "sp",
     "modules": ({"name": "sec_incident_response_judgment", "cat": "安全与密码学社招级别一:安全事件应急判断", "bank": _t3_ir},
                 {"name": "sec_security_architecture_selection_judgment", "cat": "安全与密码学社招级别二:安全架构选型判断", "bank": _t3_arch})},
)


def _self_test() -> None:
    assert len(ALL_DP) >= 70, len(ALL_DP)
    assert len(ALL_SP) >= 25, len(ALL_SP)
    assert len(ALL_DP) + len(ALL_SP) >= 100, len(ALL_DP) + len(ALL_SP)
    assert len(categories(ALL_DP)) == 5, len(categories(ALL_DP))
    assert len(categories(ALL_SP)) == 2, len(categories(ALL_SP))

    dp_ids = [dp.id for dp in ALL_DP]
    sp_ids = [sp.id for sp in ALL_SP]
    assert len(dp_ids) == len(set(dp_ids)), "security-crypto-mastery ALL_DP 内存在重复 id"
    assert len(sp_ids) == len(set(sp_ids)), "security-crypto-mastery ALL_SP 内存在重复 id"
    assert all(i.startswith("dp-sec-") for i in dp_ids), "存在不以dp-sec-开头的DeepPoint id"
    assert all(i.startswith("sc-sec-") for i in sp_ids), "存在不以sc-sec-开头的ScenarioPoint id"

    dp_triggers = [dp.trigger for dp in ALL_DP]
    sp_triggers = [sp.trigger for sp in ALL_SP]
    assert len(dp_triggers) == len(set(dp_triggers)), "security-crypto-mastery ALL_DP 内存在重复 trigger"
    assert len(sp_triggers) == len(set(sp_triggers)), "security-crypto-mastery ALL_SP 内存在重复 trigger"

    assert all(len(dp.explain) >= 100 for dp in ALL_DP), "存在explain过短的DeepPoint"
    assert all(len(sp.explain) >= 100 for sp in ALL_SP), "存在explain过短的ScenarioPoint"

    assert len(TIERS) == 3, len(TIERS)
    assert [t["n"] for t in TIERS] == [1, 2, 3], "TIERS 顺序不连续"
    tier_bank_total = sum(len(m["bank"]) for t in TIERS for m in t["modules"])
    assert tier_bank_total == len(ALL_DP) + len(ALL_SP), "TIERS 里的bank总数与ALL_DP+ALL_SP不一致"

    print(
        f"[PASS] security_crypto_mastery: 3层(浅/深/社招级别)7个模块 汇总完整,"
        f"合计{len(ALL_DP)}个DeepPoint + {len(ALL_SP)}个ScenarioPoint = {len(ALL_DP) + len(ALL_SP)}点"
    )


if __name__ == "__main__":
    _self_test()
