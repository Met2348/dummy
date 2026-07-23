# 08. HTTPS 与 TLS

> 本类知识点全部围绕一个自签名证书展开——用 WSL2 `openssl req` 生成后,证书/私钥的 PEM 内容被直接内嵌进每个 `.venv` 可运行例子里(10 年有效期,保证长期可复现,不依赖读者本地有 openssl),做到"签名/是什么"之外的每个可运行例子都是对真实 TLS 握手的真实观测,而不是描述性讲解。KP2 额外嵌入 WSL2 `openssl s_client` 真实抓取的握手摘要作为佐证。

---

## KP1. 对称加密与非对称加密在 TLS 里的分工

**签名/是什么:**

```
非对称加密(Asymmetric):一对公钥/私钥,加密解密用不同的密钥 —— 用于握手阶段的身份认证(证书签名验证)
                        和密钥交换(ECDHE),计算成本高,不适合加密大量数据。
对称加密(Symmetric):加密解密用同一个密钥 —— 握手结束后协商出共享密钥,
                     用它(如 AES-256-GCM)加密所有实际应用数据,计算成本远低于非对称算法。
```

**一句话:** TLS 只在握手阶段"少量、关键"的操作上使用非对称加密(证明身份、交换密钥),握手一结束就切换到对称加密处理所有实际流量,因为对称算法的计算成本比非对称算法低几个数量级。

**底层机制/为什么这样设计:** 如果整个连接都用非对称加密(比如直接用服务器的 RSA 公钥加密每一个应用层字节),会立刻遇到两个问题:一是 RSA 这类非对称算法的单次运算成本比 AES 这类对称算法高得多(通常是几百倍到上千倍的量级),把它用在大流量数据上会让吞吐量断崖式下跌;二是非对称加密没有天然的"流"处理模式,不适合像视频、大文件下载这种连续字节流场景。TLS 的解法是"用非对称加密解决对称加密天生的痛点——密钥怎么安全地在双方之间协商",而不是用非对称加密替代对称加密:握手阶段用非对称技术(证书签名验证身份 + ECDHE 密钥交换)在不安全的网络上安全地协商出一个只有双方知道的共享密钥,握手一结束就退回性能友好的对称加密处理真正的数据。这个"用昂贵的操作只做一次性的信任建立,后续大批量操作换成便宜的操作"的思路,是很多安全协议(不只是 TLS)的通用设计范式。ECDHE(Elliptic Curve Diffie-Hellman Ephemeral)这个缩写拆开看:Diffie-Hellman 是一种"双方各自只发一个公开值,就能各自算出同一个共享密钥,而窃听者光看这两个公开值算不出这个共享密钥"的经典密钥协商算法(数学原理本身依赖离散对数问题的困难性,不在本篇展开);EC(椭圆曲线)是这个算法具体的数学实现方式;末尾的 Ephemeral(临时的)是最关键的一个字——它意味着每次握手都会现场生成一对全新的密钥、用完立刻丢弃、绝不复用,这个"临时"特性正是 KP6"前向保密"的核心机制来源,记住这一点就足够理解本篇后续内容。

**AI 研究/工程场景:** 模型 serving 网关(比如对外暴露的推理 API endpoint)每天可能要处理数百万次 TLS 连接,如果每次连接都要做一次完整的非对称握手(证书验证+密钥交换),CPU 开销会显著挤占本该用于实际业务逻辑的算力;这正是 KP5(会话复用)存在的动因——尽量减少"贵"的非对称握手次数,把 CPU 预算留给真正需要非对称安全性的场景。

**可运行例子(验证环境:`.venv`,强制 TLS 1.2 让密码套件名称显式拼出非对称算法名):**

```python
import ssl
import socket
import threading
import time
import os
import tempfile

CERT_PEM = """-----BEGIN CERTIFICATE-----
MIIDCTCCAfGgAwIBAgIUPsrZokZ5vMOSDyXEK8a/31URpYowDQYJKoZIhvcNAQEL
BQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MB4XDTI2MDcxMzE3NTg1MFoXDTM2MDcx
MDE3NTg1MFowFDESMBAGA1UEAwwJbG9jYWxob3N0MIIBIjANBgkqhkiG9w0BAQEF
AAOCAQ8AMIIBCgKCAQEAuq9qXdm91mqwQfZO6ihyYU75Ve60+M7KvYjFAf5W3FJm
qDoxtVoePFTSgAeSMbKtGJJA1Wczg0F97Q8XZSuT/GAMMwj2HbYv0A6p91HLgMCa
9NckKcgsVi8esslB0BL0prSNNYJKM0+R3umjW+LIgnZIvSGAh1FVhDKILJzb2uwq
kHSLqEdXx+vYbFe6RhbWnlyygCcO9GXPIfRDXG1Gd9QlUxMK6RetkFuPzZ9Tx30M
qgKPDEGi23mJ/SvhnGaXiafaK5hu6fnkdL1T3aq0HLdxNqkkXU1xCYWZyMIk3iSq
hNkyYpTPyRJZX8HGLTcyH2FOOwDihwxZ/epgZKK6mQIDAQABo1MwUTAdBgNVHQ4E
FgQUtfH95mB0GI7NLZl2qDD2Z0F0XTAwHwYDVR0jBBgwFoAUtfH95mB0GI7NLZl2
qDD2Z0F0XTAwDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQsFAAOCAQEAO+VI
Yhh8PRJadkTEHSsGr7YFi6606FCz2Ht1AZByDwIAzCfRQu278ZrQHfx9fgLqRsO7
c3Z3tNFU+hphR/AlLdGdxzXpNASWsmaRHlANoEZEXyAylFxFeDfkiXVsOHbJgy8X
9c2acqJAxv1zPWmicsGnJHNQbeOzciZFnxm/py0wg3AzVRB5PMRz/PVzGEDzT0ap
8PJUFTE/aKo5vP+A1AlksXJH6DGvUfaYAtgDaRC/x5NvaJzEDtDP+ujP++ZxWs1C
mBNR8GVyAlRCzAD09eaGJhL3u4/YbrwqbDjnbVHWZgjQobUN4uUxkahakRAWsKMI
mDpVYxyh7SN2ucVV1g==
-----END CERTIFICATE-----
"""
KEY_PEM = """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC6r2pd2b3WarBB
9k7qKHJhTvlV7rT4zsq9iMUB/lbcUmaoOjG1Wh48VNKAB5Ixsq0YkkDVZzODQX3t
DxdlK5P8YAwzCPYdti/QDqn3UcuAwJr01yQpyCxWLx6yyUHQEvSmtI01gkozT5He
6aNb4siCdki9IYCHUVWEMogsnNva7CqQdIuoR1fH69hsV7pGFtaeXLKAJw70Zc8h
9ENcbUZ31CVTEwrpF62QW4/Nn1PHfQyqAo8MQaLbeYn9K+GcZpeJp9ormG7p+eR0
vVPdqrQct3E2qSRdTXEJhZnIwiTeJKqE2TJilM/JEllfwcYtNzIfYU47AOKHDFn9
6mBkorqZAgMBAAECggEASYlirVBhoq4I+xrCkCNZlvIsbkim2eFfZFSVQglFfuko
jfjKbuTuRxakipEw6cm3vJeT2VwbIwdvyqsorI4Db6UH+Sx9AkwrPlogYo9CSdfU
0kBS+vBiYpqSDZwQhB2LaIVzoFHP17FuxO+kjUPidt2QHowtXGBR/s6G7IfnZK73
tg2Fw280qS8niBKWgkS1CKDf9iqWLA0xAzEvgfLOP/D8WIGkJj2erCSXykbEdk1f
phAFbZU7bVvfBUw3ds5M6AqBSiI89AiXnww3JBJdNV8TJa+MzhOTkw5j3CNdeJEC
QRnHqMktbNrj5kZm/WH/m1yyLp4YpwSAofj/zl4lxwKBgQD8qRkdybNF/COP5SDu
FyU/i0NAUL/MyBQOl9e9rsbKgNn3lAh3ZSlQlyd54FGHLEmy2bGwVF2ATgHlzsP1
1CMG3Pgkbqzi6/yj/Tp+eKUHjGcFwDlm0Yt4ZRmRsanN5FAAGeY+4xRPJTfOPNyg
Nvax3q4XI3KoOeZRmBJNqK34HwKBgQC9JxVXTMQiSxKbAvYOiAMD8yJnm9mXt3Wd
VQBjFAfF9Zs/+5oFq6s9NU4MOtsCFx/BmwwwMQKNxiwJ+QNVy238ZPTBYOtWdcaE
GyZzCil3CXYkXRFsqRoT/FtsGRscPvgLb1tJCEmCIdXEI8ogyubpBvK8LHElFXX2
pH2IVefWRwKBgQDB8/kg4cYp8j1GZ+jYfJIObpRomdQymmCzNyLZLILTzwgDwvKg
3NpTUEVwjJ9pObk1f2Gk7457QOa6B/hsDLX6vcQLC57R78AYDvtJPMnKuqAXTRGD
eVYsTMfNDOpB8ILtIPSbz+u2OebV/eiLYMYNkthnUMHim9fPSQK79MHflwKBgQCl
nOO1lRQhRoPazyPrIzEosyeLecxxZwMGpxb4qOAJdnrg8YUws9bxd4uHb7yzvtVi
KUPpqe+nfDyw2qMN6li5ZRdbfWVwRRx0LGVk0h/uBo23Vvlbn0+i8hCFvBGdkJCT
SoOtlSDvXHqTChb+0UiN/TRoh6zlID4xnmH/DeA4HwKBgENBUt6S43xeOEDa8K/y
1r+4Ffleqv7p1yYcfahp1mkWjI+S9DxEaroWd0qrpllL1L01z0mGh2IOw0bzuugP
dmQuwXJD/RYppUxfYj2t1PJl45KnjRpQGm36L39DycRGfdojyMTmGUw60+B3+u57
QqmL2Vdq/ZvqNVIsy1VQCEii
-----END PRIVATE KEY-----
"""


def write_temp(content):
    fd, path = tempfile.mkstemp(suffix=".pem")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


CERTFILE = write_temp(CERT_PEM)
KEYFILE = write_temp(KEY_PEM)
HOST = "127.0.0.1"


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


port = free_port()
BULK = os.urandom(4 * 1024 * 1024)  # 4 MiB bulk payload


def server():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=CERTFILE, keyfile=KEYFILE)
    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    raw.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    raw.bind((HOST, port))
    raw.listen(1)
    conn, _ = raw.accept()
    tls_conn = ctx.wrap_socket(conn, server_side=True)
    tls_conn.sendall(BULK)
    tls_conn.close()
    raw.close()


t = threading.Thread(target=server, daemon=True)
t.start()
time.sleep(0.2)

ctx = ssl.create_default_context(cafile=CERTFILE)
ctx.minimum_version = ssl.TLSVersion.TLSv1_2
ctx.maximum_version = ssl.TLSVersion.TLSv1_2
t0 = time.perf_counter()
with socket.create_connection((HOST, port), timeout=3) as raw:
    with ctx.wrap_socket(raw, server_hostname="localhost") as s:
        handshake_done = time.perf_counter()
        cipher_name, proto, secret_bits = s.cipher()
        received = b""
        while len(received) < len(BULK):
            chunk = s.recv(65536)
            if not chunk:
                break
            received += chunk
transfer_done = time.perf_counter()
t.join(timeout=3)
os.remove(CERTFILE)
os.remove(KEYFILE)

assert received == BULK
print(f"cipher={cipher_name} proto={proto} handshake={  (handshake_done-t0)*1000:.1f}ms transfer(4MiB)={(transfer_done-handshake_done)*1000:.1f}ms")

# 密码套件名称本身就拆解了分工:ECDHE(非对称密钥交换)+ RSA(非对称身份认证) 只用于握手,
# AES256-GCM(对称密码) 用于上面传输的全部 4 MiB 数据——这个词本身还能再拆一层:
# AES256 = 用 256 位密钥的 AES 加密算法,负责"加密";
# GCM = Galois/Counter Mode,一种分组密码工作模式,负责在加密的同时提供"完整性认证"(能检测数据是否被篡改);
# 二者合起来是"用 AES256 加密 + 用 GCM 模式做认证",AES256-GCM 不是一个不可拆分的整体名词。
parts = cipher_name.split("-")
assert "ECDHE" in parts and "RSA" in parts
assert "AES256" in cipher_name and "GCM" in cipher_name
print("cipher name decomposition: key-exchange=ECDHE(asymmetric) auth=RSA(asymmetric) bulk-cipher=AES256-GCM(symmetric)")
```

**面试怎么问+追问链:**
- Q:为什么 TLS 不直接全程用非对称加密,反而要多此一举协商一个对称密钥?
  - 追问1:能不能举一个具体的数量级感受,非对称加密到底比对称加密慢多少?
  - 深挖追问(真实性验证轴):候选人如果只会说"慢很多",追问要求给出量级——RSA/ECDSA 这类非对称算法的单次运算通常比 AES-NI 硬件加速下的 AES-GCM 慢 2~3 个数量级,这也是为什么现代 CPU 会专门为 AES 提供硬件指令集加速(AES-NI),却不会给 RSA 这样做——因为 RSA 只在握手阶段执行一次,批量吞吐场景根本轮不到它。

**常见坑:**
- 以为"用了非对称加密"和"用了对称加密"是互斥的两种 HTTPS 配置——真实 TLS 连接总是两者都用,只是分工不同阶段;误以为某个网站"用了 RSA 加密"就意味着传输数据全程 RSA 加密,是最常见的概念混淆。

---

## KP2. TLS 握手完整流程(1.2 vs 1.3 对比)

**签名/是什么:**

```
TLS 1.2 完整握手(2-RTT):
  ClientHello ->
  <- ServerHello, Certificate, ServerKeyExchange, ServerHelloDone
  ClientKeyExchange, ChangeCipherSpec, Finished ->
  <- ChangeCipherSpec, Finished
  (至此才能发应用数据 —— 需要 2 个网络往返)

TLS 1.3 完整握手(1-RTT):
  ClientHello + KeyShare ->
  <- ServerHello + KeyShare, Certificate, CertificateVerify, Finished
  Finished -> (可以和第一个应用数据包一起发出)
  (只需要 1 个网络往返)
```

**一句话:** TLS 1.3 把 1.2 需要两轮协商(先谈判用什么算法,再交换密钥)压缩成一轮——客户端直接在第一条消息里"猜"服务器支持的密钥交换算法并带上 KeyShare,大多数情况一次往返就能建立好加密通道。

**底层机制/为什么这样设计:** TLS 1.2 的握手之所以要两轮,是因为它保留了"先协商用哪种密码套件,双方都同意了,再做真正的密钥交换"这种保守但啰嗦的顺序:第一轮 ClientHello/ServerHello 只是协商版本和密码套件,第二轮才做真正的密钥交换和身份验证。TLS 1.3 观察到:现实中客户端和服务器几乎总是会用双方都支持的、最优先的那组算法(比如 ECDHE + X25519 几乎已成事实标准),于是设计成"客户端乐观地猜测服务器会用哪种密钥交换算法,提前把自己的 KeyShare(临时公钥)一起发过去",服务器如果同意这个猜测,可以直接在第一个响应里就把密钥交换、证书、签名验证、Finished 消息全部带上——相当于把 1.2 的两轮"先谈判、后交换"合并成一轮"边谈边换",猜错了才退化到需要额外一轮重试(HelloRetryRequest,少数情况)。这不是简单的消息合并,而是协议设计思路的转变:用"大概率蒙对"换取"绝大多数场景下减少一次网络往返"。

**AI 研究/工程场景:** 移动端和跨国网络场景下,一次网络往返(RTT)可能是几十到几百毫秒,对于需要频繁建立新连接的场景(比如移动 App 唤醒后台后重新连接云端服务),TLS 1.3 少掉的这一轮 RTT 累积起来是实打实的用户可感知延迟改善,这也是各大云服务商和 CDN 近几年积极推动全站升级到 TLS 1.3 的直接工程动机。

**可运行例子(验证环境:`.venv`,用 `ssl.MemoryBIO` 手动驱动握手状态机,真实统计完成握手所需的 socket 往返次数——不是理论推算,是真实测量):**

```python
import ssl
import socket
import threading
import time
import os
import tempfile

CERT_PEM = """-----BEGIN CERTIFICATE-----
MIIDCTCCAfGgAwIBAgIUPsrZokZ5vMOSDyXEK8a/31URpYowDQYJKoZIhvcNAQEL
BQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MB4XDTI2MDcxMzE3NTg1MFoXDTM2MDcx
MDE3NTg1MFowFDESMBAGA1UEAwwJbG9jYWxob3N0MIIBIjANBgkqhkiG9w0BAQEF
AAOCAQ8AMIIBCgKCAQEAuq9qXdm91mqwQfZO6ihyYU75Ve60+M7KvYjFAf5W3FJm
qDoxtVoePFTSgAeSMbKtGJJA1Wczg0F97Q8XZSuT/GAMMwj2HbYv0A6p91HLgMCa
9NckKcgsVi8esslB0BL0prSNNYJKM0+R3umjW+LIgnZIvSGAh1FVhDKILJzb2uwq
kHSLqEdXx+vYbFe6RhbWnlyygCcO9GXPIfRDXG1Gd9QlUxMK6RetkFuPzZ9Tx30M
qgKPDEGi23mJ/SvhnGaXiafaK5hu6fnkdL1T3aq0HLdxNqkkXU1xCYWZyMIk3iSq
hNkyYpTPyRJZX8HGLTcyH2FOOwDihwxZ/epgZKK6mQIDAQABo1MwUTAdBgNVHQ4E
FgQUtfH95mB0GI7NLZl2qDD2Z0F0XTAwHwYDVR0jBBgwFoAUtfH95mB0GI7NLZl2
qDD2Z0F0XTAwDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQsFAAOCAQEAO+VI
Yhh8PRJadkTEHSsGr7YFi6606FCz2Ht1AZByDwIAzCfRQu278ZrQHfx9fgLqRsO7
c3Z3tNFU+hphR/AlLdGdxzXpNASWsmaRHlANoEZEXyAylFxFeDfkiXVsOHbJgy8X
9c2acqJAxv1zPWmicsGnJHNQbeOzciZFnxm/py0wg3AzVRB5PMRz/PVzGEDzT0ap
8PJUFTE/aKo5vP+A1AlksXJH6DGvUfaYAtgDaRC/x5NvaJzEDtDP+ujP++ZxWs1C
mBNR8GVyAlRCzAD09eaGJhL3u4/YbrwqbDjnbVHWZgjQobUN4uUxkahakRAWsKMI
mDpVYxyh7SN2ucVV1g==
-----END CERTIFICATE-----
"""
KEY_PEM = """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC6r2pd2b3WarBB
9k7qKHJhTvlV7rT4zsq9iMUB/lbcUmaoOjG1Wh48VNKAB5Ixsq0YkkDVZzODQX3t
DxdlK5P8YAwzCPYdti/QDqn3UcuAwJr01yQpyCxWLx6yyUHQEvSmtI01gkozT5He
6aNb4siCdki9IYCHUVWEMogsnNva7CqQdIuoR1fH69hsV7pGFtaeXLKAJw70Zc8h
9ENcbUZ31CVTEwrpF62QW4/Nn1PHfQyqAo8MQaLbeYn9K+GcZpeJp9ormG7p+eR0
vVPdqrQct3E2qSRdTXEJhZnIwiTeJKqE2TJilM/JEllfwcYtNzIfYU47AOKHDFn9
6mBkorqZAgMBAAECggEASYlirVBhoq4I+xrCkCNZlvIsbkim2eFfZFSVQglFfuko
jfjKbuTuRxakipEw6cm3vJeT2VwbIwdvyqsorI4Db6UH+Sx9AkwrPlogYo9CSdfU
0kBS+vBiYpqSDZwQhB2LaIVzoFHP17FuxO+kjUPidt2QHowtXGBR/s6G7IfnZK73
tg2Fw280qS8niBKWgkS1CKDf9iqWLA0xAzEvgfLOP/D8WIGkJj2erCSXykbEdk1f
phAFbZU7bVvfBUw3ds5M6AqBSiI89AiXnww3JBJdNV8TJa+MzhOTkw5j3CNdeJEC
QRnHqMktbNrj5kZm/WH/m1yyLp4YpwSAofj/zl4lxwKBgQD8qRkdybNF/COP5SDu
FyU/i0NAUL/MyBQOl9e9rsbKgNn3lAh3ZSlQlyd54FGHLEmy2bGwVF2ATgHlzsP1
1CMG3Pgkbqzi6/yj/Tp+eKUHjGcFwDlm0Yt4ZRmRsanN5FAAGeY+4xRPJTfOPNyg
Nvax3q4XI3KoOeZRmBJNqK34HwKBgQC9JxVXTMQiSxKbAvYOiAMD8yJnm9mXt3Wd
VQBjFAfF9Zs/+5oFq6s9NU4MOtsCFx/BmwwwMQKNxiwJ+QNVy238ZPTBYOtWdcaE
GyZzCil3CXYkXRFsqRoT/FtsGRscPvgLb1tJCEmCIdXEI8ogyubpBvK8LHElFXX2
pH2IVefWRwKBgQDB8/kg4cYp8j1GZ+jYfJIObpRomdQymmCzNyLZLILTzwgDwvKg
3NpTUEVwjJ9pObk1f2Gk7457QOa6B/hsDLX6vcQLC57R78AYDvtJPMnKuqAXTRGD
eVYsTMfNDOpB8ILtIPSbz+u2OebV/eiLYMYNkthnUMHim9fPSQK79MHflwKBgQCl
nOO1lRQhRoPazyPrIzEosyeLecxxZwMGpxb4qOAJdnrg8YUws9bxd4uHb7yzvtVi
KUPpqe+nfDyw2qMN6li5ZRdbfWVwRRx0LGVk0h/uBo23Vvlbn0+i8hCFvBGdkJCT
SoOtlSDvXHqTChb+0UiN/TRoh6zlID4xnmH/DeA4HwKBgENBUt6S43xeOEDa8K/y
1r+4Ffleqv7p1yYcfahp1mkWjI+S9DxEaroWd0qrpllL1L01z0mGh2IOw0bzuugP
dmQuwXJD/RYppUxfYj2t1PJl45KnjRpQGm36L39DycRGfdojyMTmGUw60+B3+u57
QqmL2Vdq/ZvqNVIsy1VQCEii
-----END PRIVATE KEY-----
"""


def write_temp(content):
    fd, path = tempfile.mkstemp(suffix=".pem")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


HOST = "127.0.0.1"


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


def server_thread(port, certfile, keyfile, errors):
    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=certfile, keyfile=keyfile)
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        raw.bind((HOST, port))
        raw.listen(1)
        raw.settimeout(3)
        conn, _ = raw.accept()
        tls_conn = ctx.wrap_socket(conn, server_side=True)
        data = tls_conn.recv(10)
        tls_conn.sendall(b"echo:" + data)
        tls_conn.close()
        raw.close()
    except Exception as e:
        errors.append(repr(e))


def client_round_trips(port, tls_version, certfile):
    incoming = ssl.MemoryBIO()
    outgoing = ssl.MemoryBIO()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    if tls_version == "1.2":
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.maximum_version = ssl.TLSVersion.TLSv1_2
    else:
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        ctx.maximum_version = ssl.TLSVersion.TLSv1_3
    obj = ctx.wrap_bio(incoming, outgoing, server_hostname="localhost")

    sock = socket.create_connection((HOST, port), timeout=3)
    sock.settimeout(3)

    round_trips = 0
    while True:
        try:
            obj.do_handshake()
            pending = outgoing.read()
            if pending:
                sock.sendall(pending)
            break
        except ssl.SSLWantReadError:
            pending = outgoing.read()
            if pending:
                sock.sendall(pending)
            sock_data = sock.recv(65536)
            incoming.write(sock_data)
            round_trips += 1  # 每多一次“发给对方 -> 等对方回信”就是一次真实网络往返

    ver = obj.version()
    obj.write(b"hi")
    sock.sendall(outgoing.read())
    resp = b""
    while not resp:
        incoming.write(sock.recv(65536))
        try:
            resp = obj.read(1024)
        except ssl.SSLWantReadError:
            continue
    assert resp == b"echo:hi"
    try:
        obj.unwrap()
        sock.sendall(outgoing.read())
    except ssl.SSLWantReadError:
        pass
    sock.close()
    return ver, round_trips


certfile = write_temp(CERT_PEM)
keyfile = write_temp(KEY_PEM)
results = {}
for v in ("1.2", "1.3"):
    port = free_port()
    errors = []
    t = threading.Thread(target=server_thread, args=(port, certfile, keyfile, errors), daemon=True)
    t.start()
    time.sleep(0.2)
    ver, rt = client_round_trips(port, v, certfile)
    t.join(timeout=3)
    assert not errors, errors
    results[v] = rt
    print(f"requested={v} negotiated={ver} real_socket_round_trips={rt}")
os.remove(certfile)
os.remove(keyfile)

assert results["1.2"] == 2, results
assert results["1.3"] == 1, results
print("assert ok: TLS1.2 needs 2 real round trips before app data, TLS1.3 needs only 1 -- matches the protocol spec exactly")
```

**WSL2 `openssl s_client` 真实握手摘要(佐证上面的 round_trips 测量,协议/密码套件层面的独立验证):**

```
=== TLS 1.2 handshake summary ===
New, TLSv1.2, Cipher is ECDHE-RSA-AES256-GCM-SHA384
Protocol: TLSv1.2

=== TLS 1.3 handshake summary ===
New, TLSv1.3, Cipher is TLS_AES_256_GCM_SHA384
Protocol: TLSv1.3
```
(验证环境:`WSL2 Rocky Linux`,`openssl s_server` 起一个真实 TLS 服务器,`openssl s_client -tls1_2`/`-tls1_3` 分别连接并抓取协商结果;注意 TLS 1.3 的密码套件名称 `TLS_AES_256_GCM_SHA384` 里已经不含密钥交换/认证算法——因为 1.3 里 ECDHE 密钥交换是强制的,不再是可协商项,这一点直接呼应 KP6 的"前向保密成为强制默认"。)

**面试怎么问+追问链:**
- Q:TLS 1.3 相比 1.2 具体在协议层面做了什么改动,使得握手能从 2-RTT 降到 1-RTT?
  - 追问1:如果客户端"猜错"了服务器支持的密钥交换算法怎么办?
  - 深挖追问(方案批判迭代轴):退化成 HelloRetryRequest——服务器发现客户端 KeyShare 里没有自己支持的算法,会要求客户端用正确的算法重新发一次 ClientHello,这种情况下 1.3 的握手反而要 2-RTT,退化到和 1.2 一样(甚至更啰嗦一轮),但实践中因为 X25519/P-256 这类算法几乎已成为事实标准,这个"猜错"分支发生概率很低,1-RTT 是绝大多数场景下的真实表现,不是保证。追问这一步是检验候选人是否只知道"1.3 更快"这个结论,还是理解这个结论背后有一个"通常情况"的前提条件。

**常见坑:**
- 把"1.3 更快"简单归因于"报文更少",而没意识到真正的收益来源是"减少了网络往返次数"(RTT)——在高带宽低延迟的局域网环境,报文数量差异带来的收益微乎其微,1.3 的收益主要体现在广域网/移动网络这种单次往返延迟本身就很大的场景,脱离场景空谈"更快"是回答不到点子上的。

---

## KP3. 数字证书与 CA 信任链

**签名/是什么:**

```
数字证书:把"公钥"和"这个公钥属于谁(域名)"绑定在一起的文件,由证书颁发机构(CA)用自己的私钥签名担保。
CA 信任链:叶子证书(网站的证书) -> 中间 CA 证书 -> 根 CA 证书,
          根 CA 证书由操作系统/浏览器预置的"信任列表"直接信任,链上每一级都由上一级签名担保。
自签名证书:证书的签发者(issuer)就是自己(subject == issuer),没有任何第三方 CA 为它背书,
           默认不被任何标准信任链信任。
```

**一句话:** 证书验证在回答一个问题——"这份公钥真的属于它声称的那个域名吗",答案的可信度来自"有没有一条从这份证书一路追溯到操作系统内置信任的根 CA 的签名链条",自签名证书因为压根没有这条链条,默认必然被拒绝。

**底层机制/为什么这样设计:** 如果没有 CA 体系,客户端拿到一份证书时无法判断"这份公钥真的是目标网站的,还是攻击者伪造的"——单靠证书自己声称"我是 example.com"毫无意义,任何人都能生成一份自称是 example.com 的证书。CA 体系的解法是引入一个客户端提前就信任的第三方(操作系统/浏览器出厂就预置了一批根 CA 证书),网站向 CA 证明自己确实控制着某个域名后,CA 用自己的私钥对"这个公钥属于这个域名"这个事实签名担保;客户端验证证书时,沿着签名链条一路往上验证签名(叶子证书的签名能用中间 CA 的公钥验证通过、中间 CA 证书的签名能用根 CA 的公钥验证通过),直到追溯到一个自己本来就信任的根,整条链条的信任就成立了。自签名证书没有这条可追溯链条,客户端没有任何独立第三方可以依赖来确认"这份证书真的代表它声称的身份",所以默认验证必然失败——这不是 bug,是这套信任模型的必然结果,也是为什么开发环境用自签名证书测试时,必须显式把这个证书加入客户端的信任列表(或者使用 `mkcert` 这类工具生成本地信任的证书)才能让验证通过。

**AI 研究/工程场景:** 内部微服务之间的 mTLS(双向 TLS)场景通常会搭建一套内部私有 CA,而不是用公网 CA 给每个内部服务签发证书——因为这些服务从不对外暴露,没必要让全世界都能验证它们的身份,只需要集群内部的服务互相信任这一个私有 CA 根证书即可,这是信任链模型"信任根可以按需自定义边界"这个特性的直接应用。

**可运行例子(验证环境:`.venv`,用同一份自签名证书对比"默认信任链"与"显式添加为信任锚点"两种验证结果):**

```python
import ssl
import socket
import threading
import time
import os
import tempfile

CERT_PEM = """-----BEGIN CERTIFICATE-----
MIIDCTCCAfGgAwIBAgIUPsrZokZ5vMOSDyXEK8a/31URpYowDQYJKoZIhvcNAQEL
BQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MB4XDTI2MDcxMzE3NTg1MFoXDTM2MDcx
MDE3NTg1MFowFDESMBAGA1UEAwwJbG9jYWxob3N0MIIBIjANBgkqhkiG9w0BAQEF
AAOCAQ8AMIIBCgKCAQEAuq9qXdm91mqwQfZO6ihyYU75Ve60+M7KvYjFAf5W3FJm
qDoxtVoePFTSgAeSMbKtGJJA1Wczg0F97Q8XZSuT/GAMMwj2HbYv0A6p91HLgMCa
9NckKcgsVi8esslB0BL0prSNNYJKM0+R3umjW+LIgnZIvSGAh1FVhDKILJzb2uwq
kHSLqEdXx+vYbFe6RhbWnlyygCcO9GXPIfRDXG1Gd9QlUxMK6RetkFuPzZ9Tx30M
qgKPDEGi23mJ/SvhnGaXiafaK5hu6fnkdL1T3aq0HLdxNqkkXU1xCYWZyMIk3iSq
hNkyYpTPyRJZX8HGLTcyH2FOOwDihwxZ/epgZKK6mQIDAQABo1MwUTAdBgNVHQ4E
FgQUtfH95mB0GI7NLZl2qDD2Z0F0XTAwHwYDVR0jBBgwFoAUtfH95mB0GI7NLZl2
qDD2Z0F0XTAwDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQsFAAOCAQEAO+VI
Yhh8PRJadkTEHSsGr7YFi6606FCz2Ht1AZByDwIAzCfRQu278ZrQHfx9fgLqRsO7
c3Z3tNFU+hphR/AlLdGdxzXpNASWsmaRHlANoEZEXyAylFxFeDfkiXVsOHbJgy8X
9c2acqJAxv1zPWmicsGnJHNQbeOzciZFnxm/py0wg3AzVRB5PMRz/PVzGEDzT0ap
8PJUFTE/aKo5vP+A1AlksXJH6DGvUfaYAtgDaRC/x5NvaJzEDtDP+ujP++ZxWs1C
mBNR8GVyAlRCzAD09eaGJhL3u4/YbrwqbDjnbVHWZgjQobUN4uUxkahakRAWsKMI
mDpVYxyh7SN2ucVV1g==
-----END CERTIFICATE-----
"""
KEY_PEM = """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC6r2pd2b3WarBB
9k7qKHJhTvlV7rT4zsq9iMUB/lbcUmaoOjG1Wh48VNKAB5Ixsq0YkkDVZzODQX3t
DxdlK5P8YAwzCPYdti/QDqn3UcuAwJr01yQpyCxWLx6yyUHQEvSmtI01gkozT5He
6aNb4siCdki9IYCHUVWEMogsnNva7CqQdIuoR1fH69hsV7pGFtaeXLKAJw70Zc8h
9ENcbUZ31CVTEwrpF62QW4/Nn1PHfQyqAo8MQaLbeYn9K+GcZpeJp9ormG7p+eR0
vVPdqrQct3E2qSRdTXEJhZnIwiTeJKqE2TJilM/JEllfwcYtNzIfYU47AOKHDFn9
6mBkorqZAgMBAAECggEASYlirVBhoq4I+xrCkCNZlvIsbkim2eFfZFSVQglFfuko
jfjKbuTuRxakipEw6cm3vJeT2VwbIwdvyqsorI4Db6UH+Sx9AkwrPlogYo9CSdfU
0kBS+vBiYpqSDZwQhB2LaIVzoFHP17FuxO+kjUPidt2QHowtXGBR/s6G7IfnZK73
tg2Fw280qS8niBKWgkS1CKDf9iqWLA0xAzEvgfLOP/D8WIGkJj2erCSXykbEdk1f
phAFbZU7bVvfBUw3ds5M6AqBSiI89AiXnww3JBJdNV8TJa+MzhOTkw5j3CNdeJEC
QRnHqMktbNrj5kZm/WH/m1yyLp4YpwSAofj/zl4lxwKBgQD8qRkdybNF/COP5SDu
FyU/i0NAUL/MyBQOl9e9rsbKgNn3lAh3ZSlQlyd54FGHLEmy2bGwVF2ATgHlzsP1
1CMG3Pgkbqzi6/yj/Tp+eKUHjGcFwDlm0Yt4ZRmRsanN5FAAGeY+4xRPJTfOPNyg
Nvax3q4XI3KoOeZRmBJNqK34HwKBgQC9JxVXTMQiSxKbAvYOiAMD8yJnm9mXt3Wd
VQBjFAfF9Zs/+5oFq6s9NU4MOtsCFx/BmwwwMQKNxiwJ+QNVy238ZPTBYOtWdcaE
GyZzCil3CXYkXRFsqRoT/FtsGRscPvgLb1tJCEmCIdXEI8ogyubpBvK8LHElFXX2
pH2IVefWRwKBgQDB8/kg4cYp8j1GZ+jYfJIObpRomdQymmCzNyLZLILTzwgDwvKg
3NpTUEVwjJ9pObk1f2Gk7457QOa6B/hsDLX6vcQLC57R78AYDvtJPMnKuqAXTRGD
eVYsTMfNDOpB8ILtIPSbz+u2OebV/eiLYMYNkthnUMHim9fPSQK79MHflwKBgQCl
nOO1lRQhRoPazyPrIzEosyeLecxxZwMGpxb4qOAJdnrg8YUws9bxd4uHb7yzvtVi
KUPpqe+nfDyw2qMN6li5ZRdbfWVwRRx0LGVk0h/uBo23Vvlbn0+i8hCFvBGdkJCT
SoOtlSDvXHqTChb+0UiN/TRoh6zlID4xnmH/DeA4HwKBgENBUt6S43xeOEDa8K/y
1r+4Ffleqv7p1yYcfahp1mkWjI+S9DxEaroWd0qrpllL1L01z0mGh2IOw0bzuugP
dmQuwXJD/RYppUxfYj2t1PJl45KnjRpQGm36L39DycRGfdojyMTmGUw60+B3+u57
QqmL2Vdq/ZvqNVIsy1VQCEii
-----END PRIVATE KEY-----
"""


def write_temp(content):
    fd, path = tempfile.mkstemp(suffix=".pem")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


CERTFILE = write_temp(CERT_PEM)
KEYFILE = write_temp(KEY_PEM)
HOST = "127.0.0.1"


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_tls_echo_server(port, ready_evt, stop_evt):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=CERTFILE, keyfile=KEYFILE)
    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    raw.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    raw.bind((HOST, port))
    raw.listen(5)
    raw.settimeout(0.5)
    ready_evt.set()
    while not stop_evt.is_set():
        try:
            conn, _ = raw.accept()
        except socket.timeout:
            continue
        try:
            tls_conn = ctx.wrap_socket(conn, server_side=True)
            data = tls_conn.recv(1024)
            tls_conn.sendall(b"echo:" + data)
            tls_conn.close()
        except Exception:
            pass
    raw.close()


port = free_port()
ready, stop = threading.Event(), threading.Event()
t = threading.Thread(target=start_tls_echo_server, args=(port, ready, stop), daemon=True)
t.start()
ready.wait(2)
time.sleep(0.2)

# 场景1:用系统默认信任链验证 —— 这份自签名证书不在任何根 CA 的信任链上,必须失败。
default_ctx = ssl.create_default_context()
failed_as_expected = False
err_reason = None
try:
    with socket.create_connection((HOST, port), timeout=2) as raw:
        with default_ctx.wrap_socket(raw, server_hostname="localhost") as s:
            s.recv(10)
except ssl.SSLCertVerificationError as e:
    failed_as_expected = True
    err_reason = e.reason
assert failed_as_expected
print("scenario1 ok: default trust chain REJECTS self-signed cert, reason:", err_reason)

# 场景2:显式把这份证书本身加入信任锚点(相当于告诉客户端"我信任这一个特定的自签名证书") -> 必须成功。
trusting_ctx = ssl.create_default_context(cafile=CERTFILE)
with socket.create_connection((HOST, port), timeout=2) as raw:
    with trusting_ctx.wrap_socket(raw, server_hostname="localhost") as s:
        s.sendall(b"hi")
        resp = s.recv(1024)
assert resp == b"echo:hi"
print("scenario2 ok: explicitly-trusted cert as anchor SUCCEEDS, got:", resp)

stop.set()
t.join(timeout=2)
os.remove(CERTFILE)
os.remove(KEYFILE)
```

**面试怎么问+追问链:**
- Q:浏览器怎么判断一个 HTTPS 网站的证书是不是可信的?
  - 追问1:如果某个中间 CA 的私钥泄露了,已经签发出去的所有证书怎么办?
  - 深挖追问(诊断真实数据轴):这正是证书吊销机制存在的原因——CRL(证书吊销列表)和 OCSP(在线证书状态协议,见 KP5)让客户端能查询"这份证书是否已被 CA 主动吊销",而不需要等到证书自然过期;这条追问能检验候选人是否理解信任链不是"验证通过就永远安全",而是一个需要持续维护(吊销机制)的动态信任体系。

**常见坑:**
- 混淆"证书过期"和"证书不受信任"两种失败原因——过期是时间维度的检查(`NotBefore`/`NotAfter`),不受信任是信任链维度的检查(能否追溯到根 CA),两者是独立的检查项,一份证书可以"链条完全可信但已过期"或者"没过期但压根不在任何信任链上",debug 证书问题时把两者分开排查是基本功。

---

## KP4. 中间人攻击与证书验证机制

**签名/是什么:**

```
中间人攻击(MITM):攻击者拦截客户端与服务器之间的通信,伪装成对方,分别与两边建立独立连接,
                   从而窃听或篡改双方本以为是"直接对话"的流量。
证书验证的两个独立检查:① 信任链检查(见 KP3,这份证书是否由客户端信任的 CA 签发)
                       ② 主机名匹配检查(证书的 CN/SAN 字段是否和目标域名一致)
                       —— 两者必须同时通过,少一个都会让 MITM 有机可乘。
```

**一句话:** 证书验证机制之所以能防住中间人攻击,核心在于"攻击者可以截获流量,但没有目标网站私钥就无法伪造出一份能通过信任链+主机名双重检查的合法证书",证书验证本质上就是这道防线。

**底层机制/为什么这样设计:** 假设没有证书验证,一个能够劫持网络流量的攻击者(比如恶意 WiFi 热点、被攻陷的路由器)可以简单地在客户端和真实服务器之间插入自己,分别和两边建立 TLS 连接,一边解密客户端发来的数据、一边用另一条连接转发给真实服务器,反之亦然——客户端和服务器都以为在直接通信,实际上所有明文数据都流经了攻击者手中。证书验证堵住这条路径的方式是:要求客户端验证收到的证书①有可信 CA 签名(攻击者没有任何 CA 的私钥,无法自己签发一份能通过信任链检查的证书)、②证书上的域名和客户端实际要访问的域名一致(即使攻击者偷到了另一个网站的合法证书,也过不了主机名匹配这一关)。两个检查环环相扣:只做信任链检查、不做主机名检查,等于允许攻击者拿"任意一个自己合法拥有的域名的证书"来冒充别的网站;只做主机名检查、不做信任链检查,等于允许攻击者自己签发一份声称是目标域名的证书。这也是为什么 KP4 的可运行例子会分别展示"域名不匹配即使证书受信也被拒绝"和"关掉验证后连接会在域名不匹配的情况下依然被接受"——这正是配置错误如何在真实系统里打开 MITM 窗口的最小可复现例子。

**画出来看(MITM的核心结构:表面上1条连接,实际上是攻击者分别维持的2条独立TLS连接):**

| 视角 | 以为在和谁直接通信 | 实际存在的TLS连接 |
|------|-----------------|------------------|
| 客户端 | 以为直接连到了服务器 | 连接①:客户端 <-> 攻击者(攻击者用伪造/窃取的证书应答客户端的握手) |
| 服务器 | 以为直接连到了客户端 | 连接②:攻击者 <-> 服务器(攻击者伪装成客户端发起这条连接) |
| 攻击者 | 清楚知道自己在两条连接的中间 | 同时握着连接①和连接②:从连接①解密读出客户端发的明文,再用连接②加密转发给服务器;服务器的回应反向操作一遍 |

关键在于:客户端和服务器各自都只看到"一条看起来正常的TLS连接",谁都不知道对面其实是攻击者而不是真正的对端——这也是为什么"信任链检查"和"主机名匹配检查"必须卡在连接①这一端:只要客户端能在建立连接①时就识破"这份证书通不过验证",攻击者伪装成服务器这一步就失败了,整个MITM结构也就搭不起来。

**AI 研究/工程场景:** 一些 AI 应用开发者在本地调试阶段为了图方便,会把 HTTP 客户端库的证书验证整体关掉(比如 `requests` 的 `verify=False`、`curl` 的 `-k`)去绕开自签名证书报错,如果这个配置不小心被带进生产环境,整个客户端就会对任何伪造证书失去防御能力——这是真实发生过的生产事故模式,也是安全代码审查里专门会检查"是否存在被禁用的证书验证"这一项的原因。

**可运行例子(验证环境:`.venv`,复用 KP3 的自签名证书服务器,展示"域名不匹配"如何被正确拒绝、以及关闭验证后这个防线如何消失):**

```python
import ssl
import socket
import threading
import time
import os
import tempfile

CERT_PEM = """-----BEGIN CERTIFICATE-----
MIIDCTCCAfGgAwIBAgIUPsrZokZ5vMOSDyXEK8a/31URpYowDQYJKoZIhvcNAQEL
BQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MB4XDTI2MDcxMzE3NTg1MFoXDTM2MDcx
MDE3NTg1MFowFDESMBAGA1UEAwwJbG9jYWxob3N0MIIBIjANBgkqhkiG9w0BAQEF
AAOCAQ8AMIIBCgKCAQEAuq9qXdm91mqwQfZO6ihyYU75Ve60+M7KvYjFAf5W3FJm
qDoxtVoePFTSgAeSMbKtGJJA1Wczg0F97Q8XZSuT/GAMMwj2HbYv0A6p91HLgMCa
9NckKcgsVi8esslB0BL0prSNNYJKM0+R3umjW+LIgnZIvSGAh1FVhDKILJzb2uwq
kHSLqEdXx+vYbFe6RhbWnlyygCcO9GXPIfRDXG1Gd9QlUxMK6RetkFuPzZ9Tx30M
qgKPDEGi23mJ/SvhnGaXiafaK5hu6fnkdL1T3aq0HLdxNqkkXU1xCYWZyMIk3iSq
hNkyYpTPyRJZX8HGLTcyH2FOOwDihwxZ/epgZKK6mQIDAQABo1MwUTAdBgNVHQ4E
FgQUtfH95mB0GI7NLZl2qDD2Z0F0XTAwHwYDVR0jBBgwFoAUtfH95mB0GI7NLZl2
qDD2Z0F0XTAwDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQsFAAOCAQEAO+VI
Yhh8PRJadkTEHSsGr7YFi6606FCz2Ht1AZByDwIAzCfRQu278ZrQHfx9fgLqRsO7
c3Z3tNFU+hphR/AlLdGdxzXpNASWsmaRHlANoEZEXyAylFxFeDfkiXVsOHbJgy8X
9c2acqJAxv1zPWmicsGnJHNQbeOzciZFnxm/py0wg3AzVRB5PMRz/PVzGEDzT0ap
8PJUFTE/aKo5vP+A1AlksXJH6DGvUfaYAtgDaRC/x5NvaJzEDtDP+ujP++ZxWs1C
mBNR8GVyAlRCzAD09eaGJhL3u4/YbrwqbDjnbVHWZgjQobUN4uUxkahakRAWsKMI
mDpVYxyh7SN2ucVV1g==
-----END CERTIFICATE-----
"""
KEY_PEM = """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC6r2pd2b3WarBB
9k7qKHJhTvlV7rT4zsq9iMUB/lbcUmaoOjG1Wh48VNKAB5Ixsq0YkkDVZzODQX3t
DxdlK5P8YAwzCPYdti/QDqn3UcuAwJr01yQpyCxWLx6yyUHQEvSmtI01gkozT5He
6aNb4siCdki9IYCHUVWEMogsnNva7CqQdIuoR1fH69hsV7pGFtaeXLKAJw70Zc8h
9ENcbUZ31CVTEwrpF62QW4/Nn1PHfQyqAo8MQaLbeYn9K+GcZpeJp9ormG7p+eR0
vVPdqrQct3E2qSRdTXEJhZnIwiTeJKqE2TJilM/JEllfwcYtNzIfYU47AOKHDFn9
6mBkorqZAgMBAAECggEASYlirVBhoq4I+xrCkCNZlvIsbkim2eFfZFSVQglFfuko
jfjKbuTuRxakipEw6cm3vJeT2VwbIwdvyqsorI4Db6UH+Sx9AkwrPlogYo9CSdfU
0kBS+vBiYpqSDZwQhB2LaIVzoFHP17FuxO+kjUPidt2QHowtXGBR/s6G7IfnZK73
tg2Fw280qS8niBKWgkS1CKDf9iqWLA0xAzEvgfLOP/D8WIGkJj2erCSXykbEdk1f
phAFbZU7bVvfBUw3ds5M6AqBSiI89AiXnww3JBJdNV8TJa+MzhOTkw5j3CNdeJEC
QRnHqMktbNrj5kZm/WH/m1yyLp4YpwSAofj/zl4lxwKBgQD8qRkdybNF/COP5SDu
FyU/i0NAUL/MyBQOl9e9rsbKgNn3lAh3ZSlQlyd54FGHLEmy2bGwVF2ATgHlzsP1
1CMG3Pgkbqzi6/yj/Tp+eKUHjGcFwDlm0Yt4ZRmRsanN5FAAGeY+4xRPJTfOPNyg
Nvax3q4XI3KoOeZRmBJNqK34HwKBgQC9JxVXTMQiSxKbAvYOiAMD8yJnm9mXt3Wd
VQBjFAfF9Zs/+5oFq6s9NU4MOtsCFx/BmwwwMQKNxiwJ+QNVy238ZPTBYOtWdcaE
GyZzCil3CXYkXRFsqRoT/FtsGRscPvgLb1tJCEmCIdXEI8ogyubpBvK8LHElFXX2
pH2IVefWRwKBgQDB8/kg4cYp8j1GZ+jYfJIObpRomdQymmCzNyLZLILTzwgDwvKg
3NpTUEVwjJ9pObk1f2Gk7457QOa6B/hsDLX6vcQLC57R78AYDvtJPMnKuqAXTRGD
eVYsTMfNDOpB8ILtIPSbz+u2OebV/eiLYMYNkthnUMHim9fPSQK79MHflwKBgQCl
nOO1lRQhRoPazyPrIzEosyeLecxxZwMGpxb4qOAJdnrg8YUws9bxd4uHb7yzvtVi
KUPpqe+nfDyw2qMN6li5ZRdbfWVwRRx0LGVk0h/uBo23Vvlbn0+i8hCFvBGdkJCT
SoOtlSDvXHqTChb+0UiN/TRoh6zlID4xnmH/DeA4HwKBgENBUt6S43xeOEDa8K/y
1r+4Ffleqv7p1yYcfahp1mkWjI+S9DxEaroWd0qrpllL1L01z0mGh2IOw0bzuugP
dmQuwXJD/RYppUxfYj2t1PJl45KnjRpQGm36L39DycRGfdojyMTmGUw60+B3+u57
QqmL2Vdq/ZvqNVIsy1VQCEii
-----END PRIVATE KEY-----
"""


def write_temp(content):
    fd, path = tempfile.mkstemp(suffix=".pem")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


CERTFILE = write_temp(CERT_PEM)
KEYFILE = write_temp(KEY_PEM)
HOST = "127.0.0.1"


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_tls_echo_server(port, ready_evt, stop_evt):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=CERTFILE, keyfile=KEYFILE)
    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    raw.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    raw.bind((HOST, port))
    raw.listen(5)
    raw.settimeout(0.5)
    ready_evt.set()
    while not stop_evt.is_set():
        try:
            conn, _ = raw.accept()
        except socket.timeout:
            continue
        try:
            tls_conn = ctx.wrap_socket(conn, server_side=True)
            data = tls_conn.recv(1024)
            tls_conn.sendall(b"echo:" + data)
            tls_conn.close()
        except Exception:
            pass
    raw.close()


port = free_port()
ready, stop = threading.Event(), threading.Event()
t = threading.Thread(target=start_tls_echo_server, args=(port, ready, stop), daemon=True)
t.start()
ready.wait(2)
time.sleep(0.2)

# 证书的 CN 是 "localhost",这里故意用一个不同的主机名去连接(模拟 MITM 场景下证书与目标域名不符)。
trusting_ctx = ssl.create_default_context(cafile=CERTFILE)
hostname_mismatch_caught = False
try:
    with socket.create_connection((HOST, port), timeout=2) as raw:
        with trusting_ctx.wrap_socket(raw, server_hostname="evil.example") as s:
            s.recv(10)
except ssl.SSLCertVerificationError:
    hostname_mismatch_caught = True
assert hostname_mismatch_caught, "hostname mismatch must be rejected by a verifying client"
print("scenario1 ok: hostname-mismatch REJECTED -- this is exactly what catches a MITM-substituted cert")

# 危险配置:整体关闭验证(常见于图省事绕过自签名证书报错) -> 防线消失,即使域名不符也照常连接。
unsafe_ctx = ssl._create_unverified_context()
with socket.create_connection((HOST, port), timeout=2) as raw:
    with unsafe_ctx.wrap_socket(raw, server_hostname="evil.example") as s:
        s.sendall(b"hi")
        resp = s.recv(1024)
assert resp == b"echo:hi", "unsafe context connects despite hostname mismatch -- this IS the MITM exposure"
print("scenario2 ok: unverified context connects DESPITE hostname mismatch -- demonstrates exactly why disabling verification opens the door to MITM")

stop.set()
t.join(timeout=2)
os.remove(CERTFILE)
os.remove(KEYFILE)
```

**面试怎么问+追问链:**
- Q:公共 WiFi 环境下用 HTTPS 访问网站,还会被中间人攻击窃听内容吗?
  - 追问1:那证书验证是不是就能完全杜绝 MITM?
  - 深挖追问(方案批判迭代轴):不完全——证书验证能杜绝"攻击者伪造一份能通过验证的证书"这条路径,但杜绝不了①用户在证书验证失败警告面前手动选择"继续访问"(社会工程学层面的失守,技术机制形同虚设)、②客户端自身的信任列表被攻击者篡改(比如恶意软件在系统信任列表里塞入自己的根证书,之后签发的任何证书都能通过验证,这是部分企业级"SSL 检查"代理和某些恶意软件采用的相同技术路径)。这条追问检验候选人是否理解"证书验证是防线,不是绝对保证",技术机制的信任根基最终还是依赖信任列表本身没有被污染。

**常见坑:**
- 认为"用了 HTTPS 就不可能被中间人攻击"——这是把"协议提供了防御机制"和"用户/系统一定正确使用了这个机制"混为一谈,真实世界里证书验证被误关闭、信任列表被污染、用户忽略警告点击继续访问,都是让 HTTPS 防线失效的真实攻击面,不是协议设计缺陷,而是使用/配置层面的问题。

---

## KP5. HTTPS 性能优化(会话复用 / OCSP 装订)

**签名/是什么:**

```
会话复用(Session Resumption):第二次连接复用第一次握手已经协商出的密钥材料,
                              跳过完整的证书验证+密钥交换,只需一次简化握手甚至零额外往返。
  - Session ID(TLS 1.2 经典机制):服务器记住 session_id -> 密钥材料 的映射
  - Session Ticket(TLS 1.2/1.3 都支持):服务器把密钥材料加密后交给客户端保管,自己不用存状态
OCSP 装订(OCSP Stapling):服务器主动向 CA 的 OCSP 服务器查询自己证书的吊销状态,
                          把查询结果"装订"在自己的 TLS 握手响应里一起发给客户端,
                          客户端不用自己再单独往返一次去问 CA。
```

**一句话:** 会话复用省掉的是"重新做一次完整非对称握手"的开销,OCSP 装订省掉的是"客户端自己去问 CA 证书有没有被吊销"这一次额外的网络往返,两者都是"把原本需要额外往返/计算的步骤,挪到不在关键路径上或者干脆省掉"的优化思路。

**底层机制/为什么这样设计:** 完整 TLS 握手的开销主要来自两处:非对称密钥交换/签名验证的计算成本(见 KP1),以及至少一次额外的网络往返(见 KP2)。会话复用的核心洞察是:如果客户端和服务器之前已经协商过一次共享密钥,没有必要每次重新走一遍完整流程从零建立信任——只要双方都还记得(或者客户端能把加密后的记忆结果交还给服务器)之前协商出的密钥材料,就可以直接派生出新的会话密钥,跳过代价最高的非对称运算部分。OCSP 装订解决的是另一个独立的性能问题:证书吊销检查(KP3 提到的"证书是否已被 CA 主动作废")如果由客户端每次都主动去问 CA 的 OCSP 服务器,等于每次 HTTPS 握手都多绑定了一次对第三方服务器的网络往返(而且这个第三方服务器如果变慢或不可达,还会拖慢或搞砸原本正常的握手)——OCSP 装订把这个查询责任转移给服务器自己提前完成并缓存,握手时直接把结果"捎带"给客户端,客户端只需要验证这个装订结果的签名是否有效,不需要再自己发起额外请求。这两个优化分别针对握手开销的两个不同来源(自身的密钥协商成本、依赖第三方的吊销检查成本),经常被一起提及但要能区分开。

**AI 研究/工程场景:** 高并发的模型推理网关如果面对大量"短连接、高频次"的调用模式(比如很多轻量级客户端各自发起短暂请求后断开),会话复用能显著降低服务端处理新建连接的 CPU 开销;这也是为什么生产环境的负载均衡器/反向代理(Nginx、Envoy)都会默认开启 TLS session ticket 或类似机制,并将其作为高并发 HTTPS 服务的标配调优项。

**可运行例子(验证环境:`.venv`,用 `ssl` 模块的 `session`/`session_reused` 接口真实验证会话复用生效与否;OCSP 装订本身依赖真实 CA 基础设施,不在自签名证书场景下可复现,仅做概念性说明):**

```python
import ssl
import socket
import threading
import time
import os
import tempfile

CERT_PEM = """-----BEGIN CERTIFICATE-----
MIIDCTCCAfGgAwIBAgIUPsrZokZ5vMOSDyXEK8a/31URpYowDQYJKoZIhvcNAQEL
BQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MB4XDTI2MDcxMzE3NTg1MFoXDTM2MDcx
MDE3NTg1MFowFDESMBAGA1UEAwwJbG9jYWxob3N0MIIBIjANBgkqhkiG9w0BAQEF
AAOCAQ8AMIIBCgKCAQEAuq9qXdm91mqwQfZO6ihyYU75Ve60+M7KvYjFAf5W3FJm
qDoxtVoePFTSgAeSMbKtGJJA1Wczg0F97Q8XZSuT/GAMMwj2HbYv0A6p91HLgMCa
9NckKcgsVi8esslB0BL0prSNNYJKM0+R3umjW+LIgnZIvSGAh1FVhDKILJzb2uwq
kHSLqEdXx+vYbFe6RhbWnlyygCcO9GXPIfRDXG1Gd9QlUxMK6RetkFuPzZ9Tx30M
qgKPDEGi23mJ/SvhnGaXiafaK5hu6fnkdL1T3aq0HLdxNqkkXU1xCYWZyMIk3iSq
hNkyYpTPyRJZX8HGLTcyH2FOOwDihwxZ/epgZKK6mQIDAQABo1MwUTAdBgNVHQ4E
FgQUtfH95mB0GI7NLZl2qDD2Z0F0XTAwHwYDVR0jBBgwFoAUtfH95mB0GI7NLZl2
qDD2Z0F0XTAwDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQsFAAOCAQEAO+VI
Yhh8PRJadkTEHSsGr7YFi6606FCz2Ht1AZByDwIAzCfRQu278ZrQHfx9fgLqRsO7
c3Z3tNFU+hphR/AlLdGdxzXpNASWsmaRHlANoEZEXyAylFxFeDfkiXVsOHbJgy8X
9c2acqJAxv1zPWmicsGnJHNQbeOzciZFnxm/py0wg3AzVRB5PMRz/PVzGEDzT0ap
8PJUFTE/aKo5vP+A1AlksXJH6DGvUfaYAtgDaRC/x5NvaJzEDtDP+ujP++ZxWs1C
mBNR8GVyAlRCzAD09eaGJhL3u4/YbrwqbDjnbVHWZgjQobUN4uUxkahakRAWsKMI
mDpVYxyh7SN2ucVV1g==
-----END CERTIFICATE-----
"""
KEY_PEM = """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC6r2pd2b3WarBB
9k7qKHJhTvlV7rT4zsq9iMUB/lbcUmaoOjG1Wh48VNKAB5Ixsq0YkkDVZzODQX3t
DxdlK5P8YAwzCPYdti/QDqn3UcuAwJr01yQpyCxWLx6yyUHQEvSmtI01gkozT5He
6aNb4siCdki9IYCHUVWEMogsnNva7CqQdIuoR1fH69hsV7pGFtaeXLKAJw70Zc8h
9ENcbUZ31CVTEwrpF62QW4/Nn1PHfQyqAo8MQaLbeYn9K+GcZpeJp9ormG7p+eR0
vVPdqrQct3E2qSRdTXEJhZnIwiTeJKqE2TJilM/JEllfwcYtNzIfYU47AOKHDFn9
6mBkorqZAgMBAAECggEASYlirVBhoq4I+xrCkCNZlvIsbkim2eFfZFSVQglFfuko
jfjKbuTuRxakipEw6cm3vJeT2VwbIwdvyqsorI4Db6UH+Sx9AkwrPlogYo9CSdfU
0kBS+vBiYpqSDZwQhB2LaIVzoFHP17FuxO+kjUPidt2QHowtXGBR/s6G7IfnZK73
tg2Fw280qS8niBKWgkS1CKDf9iqWLA0xAzEvgfLOP/D8WIGkJj2erCSXykbEdk1f
phAFbZU7bVvfBUw3ds5M6AqBSiI89AiXnww3JBJdNV8TJa+MzhOTkw5j3CNdeJEC
QRnHqMktbNrj5kZm/WH/m1yyLp4YpwSAofj/zl4lxwKBgQD8qRkdybNF/COP5SDu
FyU/i0NAUL/MyBQOl9e9rsbKgNn3lAh3ZSlQlyd54FGHLEmy2bGwVF2ATgHlzsP1
1CMG3Pgkbqzi6/yj/Tp+eKUHjGcFwDlm0Yt4ZRmRsanN5FAAGeY+4xRPJTfOPNyg
Nvax3q4XI3KoOeZRmBJNqK34HwKBgQC9JxVXTMQiSxKbAvYOiAMD8yJnm9mXt3Wd
VQBjFAfF9Zs/+5oFq6s9NU4MOtsCFx/BmwwwMQKNxiwJ+QNVy238ZPTBYOtWdcaE
GyZzCil3CXYkXRFsqRoT/FtsGRscPvgLb1tJCEmCIdXEI8ogyubpBvK8LHElFXX2
pH2IVefWRwKBgQDB8/kg4cYp8j1GZ+jYfJIObpRomdQymmCzNyLZLILTzwgDwvKg
3NpTUEVwjJ9pObk1f2Gk7457QOa6B/hsDLX6vcQLC57R78AYDvtJPMnKuqAXTRGD
eVYsTMfNDOpB8ILtIPSbz+u2OebV/eiLYMYNkthnUMHim9fPSQK79MHflwKBgQCl
nOO1lRQhRoPazyPrIzEosyeLecxxZwMGpxb4qOAJdnrg8YUws9bxd4uHb7yzvtVi
KUPpqe+nfDyw2qMN6li5ZRdbfWVwRRx0LGVk0h/uBo23Vvlbn0+i8hCFvBGdkJCT
SoOtlSDvXHqTChb+0UiN/TRoh6zlID4xnmH/DeA4HwKBgENBUt6S43xeOEDa8K/y
1r+4Ffleqv7p1yYcfahp1mkWjI+S9DxEaroWd0qrpllL1L01z0mGh2IOw0bzuugP
dmQuwXJD/RYppUxfYj2t1PJl45KnjRpQGm36L39DycRGfdojyMTmGUw60+B3+u57
QqmL2Vdq/ZvqNVIsy1VQCEii
-----END PRIVATE KEY-----
"""


def write_temp(content):
    fd, path = tempfile.mkstemp(suffix=".pem")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


CERTFILE = write_temp(CERT_PEM)
KEYFILE = write_temp(KEY_PEM)
HOST = "127.0.0.1"


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_tls_echo_server(port, ready_evt, stop_evt):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=CERTFILE, keyfile=KEYFILE)
    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    raw.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    raw.bind((HOST, port))
    raw.listen(5)
    raw.settimeout(0.5)
    ready_evt.set()
    while not stop_evt.is_set():
        try:
            conn, _ = raw.accept()
        except socket.timeout:
            continue
        try:
            tls_conn = ctx.wrap_socket(conn, server_side=True)
            data = tls_conn.recv(1024)
            tls_conn.sendall(b"echo:" + data)
            tls_conn.close()
        except Exception:
            pass
    raw.close()


port = free_port()
ready, stop = threading.Event(), threading.Event()
t = threading.Thread(target=start_tls_echo_server, args=(port, ready, stop), daemon=True)
t.start()
ready.wait(2)
time.sleep(0.2)

ctx = ssl.create_default_context(cafile=CERTFILE)

# 第一次连接:完整握手,拿到 session 对象保存下来。
with socket.create_connection((HOST, port), timeout=2) as raw:
    with ctx.wrap_socket(raw, server_hostname="localhost") as s:
        s.sendall(b"first")
        s.recv(1024)
        session = s.session
assert session is not None

# 第二次连接:把保存的 session 带回去 -> 期望 session_reused == True(简化握手)。
with socket.create_connection((HOST, port), timeout=2) as raw:
    with ctx.wrap_socket(raw, server_hostname="localhost", session=session) as s:
        s.sendall(b"second")
        s.recv(1024)
        reused = s.session_reused

# 第三次连接:全新 context/session -> 期望 session_reused == False(重新走完整握手)。
ctx2 = ssl.create_default_context(cafile=CERTFILE)
with socket.create_connection((HOST, port), timeout=2) as raw:
    with ctx2.wrap_socket(raw, server_hostname="localhost") as s:
        s.sendall(b"third")
        s.recv(1024)
        not_reused = s.session_reused

assert reused is True, "second connection with the saved session must be an abbreviated handshake"
assert not_reused is False, "a brand-new context/session must do a full handshake"
print(f"with saved session -> session_reused={reused}; fresh session -> session_reused={not_reused}")

stop.set()
t.join(timeout=2)
os.remove(CERTFILE)
os.remove(KEYFILE)
```

**面试怎么问+追问链:**
- Q:会话复用有没有安全代价?
  - 追问1(决策依据追问轴):既然复用能避免重新协商密钥,是不是复用的次数、有效期越长越好?
  - 深挖追问:不是——会话复用本质上是在用"旧的密钥材料"派生新的会话密钥,如果这份旧密钥材料被攻击者拿到(比如服务器的 session ticket 加密密钥泄露),所有基于它复用出来的会话都会受影响,这正是 KP6"前向保密"要单独讨论的问题;因此生产环境通常会限制 session ticket 的有效期(比如几小时到一天)并定期轮换加密它的密钥,在"性能收益"和"密钥材料暴露窗口"之间做权衡,不是无限期复用。

**常见坑:**
- 把"会话复用"和"HTTP Keep-Alive"([07 类 KP7](07-http-evolution.md))混为一谈——两者都叫"复用"但作用在不同层:Keep-Alive 复用的是已经建立好的 TCP+TLS 连接本身(同一条连接上跑多个 HTTP 请求),会话复用针对的是"建立一条新连接时,能不能免去完整密钥协商"(即使是两条完全不同的 TCP 连接,只要 TLS 会话状态可以复用,依然能加速);两者可以同时发生,也可以独立发生,概念层面不能划等号。

---

## KP6. 前向保密(Forward Secrecy)

**签名/是什么:**

```
前向保密:即使服务器的长期私钥(证书对应的私钥)未来某一天泄露,攻击者也无法用它解密"过去已经录下的"加密流量。
实现方式:握手阶段用临时(ephemeral)密钥对做 ECDHE 密钥交换 —— 每次握手都生成一对新的临时密钥,
         用完即弃,不落盘、不复用;长期私钥只用于对这次临时密钥交换的结果做签名认证(证明"这确实是我"),
         不直接参与加密数据的密钥推导。
```

**一句话:** 前向保密的关键在于"加密数据用的密钥是临时生成、用完即弃的,和服务器的长期私钥是两回事",所以就算长期私钥未来泄露,过去每一次握手临时生成的密钥早已从内存中消失,没有任何东西可以帮攻击者逆向出当年的会话密钥。

**底层机制/为什么这样设计:** 如果密钥交换直接用服务器的长期私钥做(比如老式的静态 RSA 密钥交换:客户端用服务器的公钥加密一个随机数,只有服务器能用私钥解出来),那么这个长期私钥就成了解密"所有历史流量"的万能钥匙——只要攻击者攻陷服务器拿到这份私钥(哪怕是很久之后才拿到),之前录下的所有历史加密流量都能被回溯破解(这类攻击模式称为"先囤积密文,等将来钥匙到手再解密")。ECDHE 密钥交换换了一种设计:双方各自生成一对临时的椭圆曲线密钥对,只在这一次握手中用来推导共享密钥,握手一结束这对临时私钥就从内存里丢弃;服务器的长期私钥全程只被用来对"这次临时公钥确实是我发的"这个声明做数字签名(身份认证),不参与实际的密钥推导计算。这样即使长期私钥未来泄露,攻击者能伪造的只是"未来新的连接",而不能用它解密任何"过去已经完成、临时密钥已经销毁"的会话——这正是 KP2 提到的"TLS 1.3 移除了静态 RSA 密钥交换选项,ECDHE 变成强制项"的根本原因:协议设计者认为前向保密的安全收益足够重要,值得把它从"可选项"提升为"强制默认"。

**AI 研究/工程场景:** 大规模用户数据(包括发给云端大模型 API 的用户输入)在传输过程中如果被具备"先录流量、等未来某天破解"能力的对手长期监听(比如国家级对手或者长期潜伏的内部威胁),前向保密直接决定了"就算今天的密钥体系未来某天被攻破,历史流量是否还安全"——这也是为什么"前向保密是否强制启用"是安全合规审查(尤其是处理敏感数据的 AI 服务)里的一项具体检查点,而不只是理论讨论。

**真实证据(验证环境:`WSL2 Rocky Linux`,`openssl s_client` 抓取的密钥交换详情,证明真实握手确实在用临时/混合密钥):**

```
TLS 1.2 (ECDHE-RSA-AES256-GCM-SHA384):
  Peer Temp Key: X25519, 253 bits
  # "Temp Key" 就是 OpenSSL 对"临时密钥"的直接称呼 —— 每次握手都会重新生成一对。

TLS 1.3 (TLS_AES_256_GCM_SHA384):
  Negotiated TLS1.3 group: X25519MLKEM768
  # 这份 2026 年的 OpenSSL 3.5.5 build 默认协商出的是一个"混合"密钥交换组:
  # X25519(经典椭圆曲线,前向保密)+ ML-KEM-768(后量子密钥封装算法)组合使用 ——
  # 这是为了同时防御"传统计算机今天破解"和"未来量子计算机破解今天录下的密文"两种风险,
  # 属于前向保密概念的延伸(抗量子前向保密),不是本知识点要求掌握的范围,列在这里作为真实环境的原始记录。
```

**可运行例子(验证环境:`.venv`,证明默认信任上下文的密码套件集合里根本不包含非前向保密的静态密钥交换选项——现代 TLS 客户端"默认就前向保密",不需要用户手动选择):**

```python
import ssl

# 显式请求一个"非前向保密"的经典密码套件(静态 RSA 密钥交换,没有 ECDHE)—— 这类套件依然存在于
# OpenSSL 的算法库里(向后兼容老系统),但下面会验证它并不在"默认信任上下文"实际会用到的套件集合里。
legacy_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
legacy_ctx.minimum_version = ssl.TLSVersion.TLSv1_2
legacy_ctx.maximum_version = ssl.TLSVersion.TLSv1_2
legacy_ctx.set_ciphers("AES256-GCM-SHA384")  # 静态 RSA 密钥交换,没有 ECDHE,不具备前向保密
legacy_ciphers = [c["name"] for c in legacy_ctx.get_ciphers()]
assert "AES256-GCM-SHA384" in legacy_ciphers
print("explicitly requested non-FS cipher IS available in OpenSSL's algorithm library:", legacy_ciphers)

# 但 create_default_context() 是绝大多数真实应用实际使用的方式 —— 检查它的默认套件集合。
default_ctx = ssl.create_default_context()
default_ciphers = [c["name"] for c in default_ctx.get_ciphers()]
non_fs_in_default = [
    name for name in default_ciphers
    if "ECDHE" not in name and "DHE" not in name and not name.startswith("TLS_")
]
assert non_fs_in_default == [], non_fs_in_default
print(f"default context offers {len(default_ciphers)} ciphers, ZERO of them are non-forward-secret")
print("assert ok: every cipher a default-configured client would actually negotiate provides forward secrecy")
```

**面试怎么问+追问链:**
- Q:前向保密具体防住的是哪一种威胁场景?
  - 追问1:那前向保密能防住"实时的"中间人攻击吗?
  - 深挖追问(决策依据追问轴):不能——前向保密解决的是"过去录下的密文,未来钥匙泄露后还能不能被解密"这个时间维度的问题,和 KP4 讨论的"实时会话身份是否被冒充"是完全不同的两个安全属性,一个密钥交换算法可以同时具备前向保密(ECDHE)和证书认证(RSA/ECDSA 签名),两者分别防御不同的威胁模型,不能用其中一个代替另一个。这条追问检验候选人是否把"前向保密"和"防中间人攻击"这两个常被同时提到、但实际正交的属性混为一谈。

**常见坑:**
- 认为"前向保密"意味着"服务器的私钥不再重要"——长期私钥依然是整个信任链的根基(用它做签名认证服务器身份,见 KP1/KP3),前向保密只是确保这份私钥"不再是解密历史流量的钥匙",私钥泄露依然会让攻击者能够在未来伪造该服务器的身份发起新的 MITM 攻击,只是无法要挟"过去"。

---

*本篇完成:2026-07-14,6 个知识点。验证环境:5 个可运行代码块为 `.venv`(KP1 对称/非对称分工+真实4MiB吞吐、KP2 用 `ssl.MemoryBIO` 手动驱动握手真实测出 TLS1.2=2次往返/TLS1.3=1次往返、KP3 信任链验证真实拒绝/接受对比、KP4 主机名不匹配+关闭验证的MITM暴露真实复现、KP5 会话复用 `session_reused` 真实验证)+ 1 个 `.venv` 代码块(KP6 默认密码套件集合不含非前向保密选项);KP2、KP6 各附一段 `WSL2 Rocky Linux` `openssl s_client` 真实握手证据作为独立佐证。板块 IV(应用层协议)进度 2/4。*
