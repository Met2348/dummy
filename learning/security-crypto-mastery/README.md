# Security & Crypto Mastery —— 安全与密码学基础专题(浅→深→社招级别)

> 定位:老师在CS基础五件套完成后要求"仔细思考和科班CS本科相比还差哪些能力",梳理出六个理论必修课空白,这是第六个也是最后一个。这是**通用信息安全和密码学基础**,和仓库里已有的 `safety-defense`/`red-team-jailbreak`(LLM安全,面向大模型越狱/对齐攻防)是完全不同的细分领域,执行前已确认无内容重叠。

## 和其他专题的关系

网络安全协议一章的TLS/PKI直接建立在networking-mastery专题HTTPS一章的基础上,但讲得更深(TLS 1.3握手细节/证书链验证/OCSP);常见漏洞深水一章的竞态条件(TOCTOU)呼应os-mastery专题的并发章节;应用安全工程深水一章的"左移"(shift left)理念直接呼应software-engineering-mastery专题的软件工程一般原理。这个专题也是CS科班理论六件套的收官——和前五个偏"纯理论"的专题不同,安全是一个天生"应用导向"的领域,tier3判断层的场景判断更容易找到真实的生产事故类比。

## 组织轴:难度分层

| 层 | 目录 | 定位 | 点数 | 类型 |
|---|---|---|:--:|:--:|
| 浅(Tier 1) | `src/tier1_shallow/` | 密码学基础(对称非对称加密/哈希与数字签名/密钥交换/前向保密)、认证与访问控制基础(认证三要素/OAuth2.0/Kerberos/RBAC-ABAC/会话管理) | 35 | DeepPoint |
| 深(Tier 2) | `src/tier2_deep/` | 常见漏洞深水(缓冲区溢出/SQL注入/XSS-CSRF/反序列化/竞态条件TOCTOU)、网络安全协议深水(TLS 1.3/PKI证书链/VPN/防火墙IDS-IPS)、应用安全工程深水(Secure SDLC/STRIDE威胁建模/secrets管理/供应链安全SBOM) | 45 | DeepPoint |
| 社招级别(Tier 3) | `src/tier3_social_hire/` | 安全事件应急判断(数据泄露响应/漏洞应急/日志取证)、安全架构选型判断(零信任vs边界防御/加密方案选择) | 28 | ScenarioPoint |

**合计108点(80个DeepPoint + 28个ScenarioPoint)**。

## 内容总览

| 文件 | 覆盖内容 | 点数 |
|---|---|:--:|
| `tier1_shallow/dp_sec_cryptography_basics.py` | 对称非对称加密、AES/RSA/ECC、哈希与数字签名、Diffie-Hellman密钥交换、前向保密 | 18 |
| `tier1_shallow/dp_sec_auth_access_control_basics.py` | 认证三要素、OAuth2.0/SAML/Kerberos、RBAC/ABAC、会话管理、零信任雏形 | 17 |
| `tier2_deep/dp_sec_common_vulnerabilities_deep.py` | 缓冲区溢出、SQL/命令注入、XSS/CSRF、反序列化漏洞、TOCTOU竞态条件 | 15 |
| `tier2_deep/dp_sec_network_security_protocols_deep.py` | TLS 1.3握手、PKI证书链、VPN协议对比、防火墙/IDS-IPS | 15 |
| `tier2_deep/dp_sec_application_security_engineering_deep.py` | Secure SDLC、STRIDE威胁建模、secrets管理、供应链安全/SBOM | 15 |
| `tier3_social_hire/sc_sec_incident_response_judgment.py` | 数据泄露响应、漏洞应急处置顺序、日志取证判断 | 14 |
| `tier3_social_hire/sc_sec_security_architecture_selection_judgment.py` | 零信任vs边界防御、加密方案选择、安全与可用性权衡 | 14 |

## 新增:本科课件式完整教材

沿用CS基础五件套的做法,新增 `lectures/01-textbook.md`——七章连贯教科书叙述,按密码学基础→认证访问控制基础→常见漏洞深水→网络安全协议深水→应用安全工程深水→安全事件应急判断→安全架构选型判断的顺序系统教学。**建议先读完 `01-textbook.md` 再做下面的追问链/场景判断自测。**

## 数据结构:DeepPoint + ScenarioPoint(含 explain 字段)

```python
import sys
sys.path.insert(0, "learning/security-crypto-mastery/src")
from security_crypto_mastery import ALL_DP, ALL_SP, TIERS
from deep_common import drill, grade_chain, grade_scenario
```

## 环境与测试

```bash
python learning/security-crypto-mastery/src/tests/test_all.py
```

9个模块(deep_common + 7个内容文件 + 总聚合)全部通过 `_self_test()`。

## 诚实说明

内容全部是经典信息安全/密码学教材共识(OWASP Top 10/RFC 8446 TLS 1.3/RFC 6749 OAuth2.0/RFC 4120 Kerberos体系),不需要追赶前沿论文,但仍要求核实具体标准和机制描述的准确性。这是**防御性安全教育内容**——帮助工程师理解漏洞原理以便写出安全代码,不提供可直接用于攻击生产系统的完整exploit代码。`real_world_link` 字段全部留空。

至此,老师要求的六个CS科班理论专题(离散数学/计算理论/算法理论证明向/计算机体系结构/编译原理/安全密码学基础)**全部完成**。
