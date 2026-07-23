# 09. DNS 域名解析

> 纪律声明:本类全部可运行例子使用**手工构造的模拟 DNS 层级/记录数据**(虚构域名 `example.com` 及其下属主机名,虚构 IP),不对真实公网 DNS 服务器发起任何查询——这是本系列"不引入外部网络依赖"纪律在 DNS 这个具体学科上的落地(真实公网查询结果不可控、不确定、且会让 `_verify_md.py` 的可复现性依赖网络状态)。所有断言验证的是 DNS 协议本身的**机制逻辑**(层级委派怎么走、缓存怎么过期、签名怎么校验),这些逻辑在模拟数据和真实数据上是完全等价的。

---

## KP1. DNS 域名层级结构

**签名/是什么:**

```
DNS 命名空间是一棵倒过来读的树:
  .(根)  ->  com(顶级域 TLD)  ->  example.com(二级域)  ->  www.example.com(主机名/子域)
每一级都有自己的"权威服务器"负责回答"我这一级下面该去哪问",只有最后一级才给出真正的 IP 地址。
```

**一句话:** 域名从右往左读才是"从粗到细"的层级顺序,每一级服务器只知道"下一级该去问谁",没有任何一台服务器天生知道所有域名对应的 IP,答案是沿着树逐级问出来的。

**底层机制/为什么这样设计:** 如果要求存在一台服务器保存全世界所有域名到 IP 的映射,这台服务器会立刻成为单点瓶颈(全球查询量)、单点故障(它挂了整个互联网域名解析瘫痪)、且更新极其低效(任何一个网站改 IP 都要更新这台全局服务器)。DNS 的解法是把命名空间按层级切分成一个个"区域(zone)",每个区域只需要知道两件事:自己直接管辖的记录,以及"我下面每个子域该去问哪台服务器"(这叫委派/delegation)。根服务器只需要知道 13 组根服务器地址是公开写死的、以及"`.com` 归哪些 TLD 服务器管"这类极少量的顶层委派信息;`.com` 的 TLD 服务器只需要知道每个二级域(如 `example.com`)归哪个权威服务器管,不需要知道 `www.example.com` 具体解析成什么 IP。这种"每一级只管好自己这一层、把剩下的问题委派给下一层"的设计,是大规模分布式系统里"分层自治、避免单点"思路的一个经典范例,和 [03 类](03-routing-and-network-layer-mechanisms.md) 路由表用最长前缀匹配做层级化转发是同一种设计哲学在不同学科的复用。

**画成一棵树看(注意:域名书写顺序是"从叶子到根",从左往右读和这棵树的层级顺序刚好相反):**

```
                    .  (根,root)
                    │
                    │  委派:"com 这个顶级域归这些TLD服务器管"(只给线索,不给答案)
                    ▼
                   com  (顶级域,TLD)
                    │
                    │  委派:"example.com 归这台权威服务器管"
                    ▼
              example.com  (二级域,由权威服务器直接管辖)
                    │
                    │  这一级权威服务器直接持有真正的记录(A/AAAA/...)
                    ▼
          www.example.com  (主机名,查到这一步才拿到真正的IP)
```

对应下面代码里 `qname = "www.example.com"` 的解析路径:`root_lookup` 返回的是"去问 `tld-server-com`"这条线索,不是答案;`tld_lookup` 返回的是"去问 `auth-server-example`"这条线索,同样不是答案;只有最后 `auth_lookup` 才返回真正的 IP `93.184.216.34`——沿着树往下走,前面几级给的都是"下一步该敲哪扇门",只有走到底才拿到真正想要的东西。

**AI 研究/工程场景:** 大规模模型 serving 集群内部的服务发现(见 [12 类](12-modern-networking-topics.md))经常借用同样的分层委派思路——顶层配置中心只知道"哪个子集群/区域负责哪类模型",不需要知道每个具体模型实例的地址,具体地址解析委派给区域内部的服务注册表处理,这样顶层配置改动频率低、压力小,和 DNS 根服务器的角色定位高度类似。

**可运行例子(验证环境:`.venv`,构造三级模拟 DNS 层级,真实走一遍逐级委派解析路径):**

```python
# 模拟三级 DNS 层级:root -> TLD(.com) -> 权威服务器(example.com)
ROOT_ZONE = {"com": "tld-server-com"}
TLD_ZONE = {"tld-server-com": {"example.com": "auth-server-example"}}
AUTH_ZONE = {"auth-server-example": {"www.example.com": "93.184.216.34"}}


def root_lookup(qname):
    tld = qname.split(".")[-1]
    return ROOT_ZONE.get(tld)  # 根返回的是委派:"去问这台 TLD 服务器"


def tld_lookup(server, qname):
    domain = ".".join(qname.split(".")[-2:])
    zone = TLD_ZONE.get(server, {})
    return zone.get(domain)  # TLD 返回的也是委派:"去问这台权威服务器"


def auth_lookup(server, qname):
    zone = AUTH_ZONE.get(server, {})
    return zone.get(qname)  # 权威服务器返回真正的答案


qname = "www.example.com"
tld_server = root_lookup(qname)
assert tld_server == "tld-server-com"
auth_server = tld_lookup(tld_server, qname)
assert auth_server == "auth-server-example"
ip = auth_lookup(auth_server, qname)
assert ip == "93.184.216.34"
print(f"root->'{tld_server}' -> tld->'{auth_server}' -> auth->'{ip}' (3-level delegation walked)")
```

**面试怎么问+追问链:**
- Q:为什么域名从右往左读才是层级顺序,`www.example.com` 里谁是"父"谁是"子"?
  - 追问1:根服务器一共有多少台,它们保存了整个互联网的域名数据吗?
  - 深挖追问(真实性验证轴):全球只有 13 组根服务器"地址"(用 Anycast 技术——多个物理地点的服务器对外宣告同一个 IP 地址,路由器按正常的路由选路规则,自动把请求送到网络意义上"最近"的那一台,请求者完全无感知——在多个物理地点部署,不是真的只有 13 台机器),它们只保存"顶级域归哪些 TLD 服务器管"这一层极少量的委派信息,完全不保存任何二级域名及以下的具体记录——候选人如果回答"根服务器保存了所有网站的 IP",说明对层级委派的本质理解有误。

**常见坑:**
- 把"DNS 层级"和"域名注册的所有权层级"混淆——技术上的委派关系(谁的服务器负责回答)和业务上的所有权关系(谁在 ICANN/注册商那里注册了这个域名)是两回事,一个域名的所有者可以把解析权限委派给和自己完全无关的第三方服务器(比如很多网站用 Cloudflare/DNSPod 的服务器做权威解析,而不是自己运营的服务器)。

---

## KP2. 递归查询 vs 迭代查询完整链路

**签名/是什么:**

```
迭代查询(Iterative):查询方自己拿着上一级返回的"委派线索"逐级往下问,自己承担全部查询次数。
递归查询(Recursive):查询方只问一次"递归解析器",由递归解析器代劳走完整个迭代过程,
                     最后把最终答案一次性返回给查询方。
```

**一句话:** 客户端到本地/公共递归解析器(如 8.8.8.8)之间通常是递归查询(客户端只问一次,解析器负责跑腿到底),递归解析器再到根/TLD/权威服务器之间才是真正的迭代查询(解析器自己一级一级问下去)。

**底层机制/为什么这样设计:** 如果要求每个终端设备(手机、笔记本)自己实现完整的迭代查询逻辑——先问根服务器要 TLD 委派、再问 TLD 要权威服务器委派、最后再问权威服务器要答案——每个应用每次域名解析都要发起多次网络请求,对终端设备的网络资源和电量都是负担,而且大量终端设备各自反复查询相同的热门域名(如 `google.com`),完全没有复用查询结果的机制。递归解析器的设计把这个"跑腿"的工作集中到少数几台专门的服务器上:客户端只需要问递归解析器一次,解析器内部去完成迭代查询的全部工作,并且因为解析器服务大量客户端,天然具备做"共享缓存"(见 KP3)的条件——一旦解析器帮某个客户端查过 `google.com`,后续所有其他客户端查询同一个域名都能直接命中缓存,不需要重新走一遍迭代查询。这是"把重复劳动集中到一个能够复用结果的中间层"的设计范式,和 CDN 边缘节点缓存热门内容([10 类](10-modern-app-protocols-and-apis.md))本质上是同一个思路。

**AI 研究/工程场景:** 企业内部网络通常会部署自己的内部递归 DNS 解析器(而不是让每台内部机器直接查询公网 DNS),除了性能收益(内部查询走内网、命中内部缓存更快),更重要的是安全与可观测性——所有 DNS 查询集中经过一个出口,便于统一做访问审计、恶意域名黑名单拦截,这也是很多企业级安全产品(比如 DNS 层面的威胁防护)的落地位置。

**可运行例子(验证环境:`.venv`,复用同一套模拟层级,分别统计"客户端自己做迭代查询"和"客户端只问递归解析器"两种方式下,客户端自身实际发出的查询次数):**

```python
ROOT_ZONE = {"com": "tld-server-com"}
TLD_ZONE = {"tld-server-com": {"example.com": "auth-server-example"}}
AUTH_ZONE = {"auth-server-example": {"www.example.com": "93.184.216.34"}}


def root_lookup(qname):
    return ROOT_ZONE.get(qname.split(".")[-1])


def tld_lookup(server, qname):
    return TLD_ZONE.get(server, {}).get(".".join(qname.split(".")[-2:]))


def auth_lookup(server, qname):
    return AUTH_ZONE.get(server, {}).get(qname)


qname = "www.example.com"

# 迭代查询:客户端自己拿着委派线索逐级去问,每一级都算客户端发出的一次查询。
client_query_count = 0


def client_iterative_resolve(qname):
    global client_query_count
    client_query_count += 1
    tld_server = root_lookup(qname)  # 查询1:问根
    client_query_count += 1
    auth_server = tld_lookup(tld_server, qname)  # 查询2:问TLD
    client_query_count += 1
    ip = auth_lookup(auth_server, qname)  # 查询3:问权威服务器
    return ip


ip1 = client_iterative_resolve(qname)
assert ip1 == "93.184.216.34"
assert client_query_count == 3, client_query_count

# 递归查询:客户端只问递归解析器一次,解析器内部自己去做完全一样的三级迭代。
resolver_internal_query_count = 0


def recursive_resolver_resolve(qname):
    global resolver_internal_query_count
    resolver_internal_query_count += 1
    tld_server = root_lookup(qname)
    resolver_internal_query_count += 1
    auth_server = tld_lookup(tld_server, qname)
    resolver_internal_query_count += 1
    ip = auth_lookup(auth_server, qname)
    return ip


client_query_count_2 = 1  # 客户端只发了这一次查询(问递归解析器)
ip2 = recursive_resolver_resolve(qname)
assert ip2 == "93.184.216.34"
assert resolver_internal_query_count == 3  # 解析器内部依然做了3次跳转
assert client_query_count_2 == 1

print(f"iterative-client made {client_query_count} queries itself; "
      f"recursive-client made {client_query_count_2} query (resolver did {resolver_internal_query_count} internal hops)")
```

**面试怎么问+追问链:**
- Q:平时上网用的 DNS 是递归查询还是迭代查询?
  - 追问1:如果递归解析器本身也没有缓存、每次都要重新走一遍完整的迭代查询,相比让客户端自己迭代查询,到底省了什么?
  - 深挖追问(方案批判迭代轴):即使假设不考虑缓存收益,递归解析器仍然把"迭代查询逻辑的实现和维护复杂度"从数以亿计的终端设备转移到少数专业运维的服务器上——终端设备的 DNS 客户端库可以做得极其简单(发一个 UDP 包等一个回复),复杂的重试、超时、多级查询逻辑全部集中在解析器一侧维护和优化,这是关注点分离在网络协议设计里的具体体现,不是单纯的"省查询次数"。

**常见坑:**
- 认为"递归查询"和"迭代查询"是同一次域名解析里非此即彼的选择——真实链路里两者通常同时存在,只是发生在链路的不同段(客户端到解析器是递归段,解析器到根/TLD/权威服务器之间是迭代段),把两者理解成互斥的"配置选项"是常见误区。

---

## KP3. DNS 缓存与 TTL 机制(各层缓存)

**签名/是什么:**

```
TTL(Time To Live):权威服务器给每条记录标注的"建议缓存时长"(单位:秒),
                   写在 DNS 响应报文里,告诉所有中间缓存"这个答案在多久之内可以放心复用"。
多层缓存:操作系统 stub resolver 缓存 -> 本地/ISP 递归解析器缓存 -> (部分)应用层自己的缓存,
         同一条记录可能同时存在于好几层缓存里,各自独立倒计时过期。
```

**一句话:** TTL 到期前,任何一层缓存都可以直接把上次的答案交回去而不用重新走一遍解析链路,TTL 一到期,下一次查询就必须重新触发真实解析,缓存值和"权威答案"之间永远存在最长不超过 TTL 的滞后窗口。

**底层机制/为什么这样设计:** 如果没有缓存,每一次域名访问都要重新走一遍 KP1/KP2 描述的完整层级查询链路,对权威服务器和根/TLD 服务器的压力会随全球查询量线性增长到不可承受——事实上,热门域名(如搜索引擎首页)每秒的真实访问量是巨大的,但对应的 IP 记录可能几天甚至几周都不变一次,绝大多数查询其实是在向权威服务器问一个"根本没变过"的答案。TTL 机制让权威服务器自己声明"这条记录大概率多久不会变",把这个时间窗口内的重复查询压力转移给缓存层去承担,权威服务器只需要在真正发生变更时被查询一次(缓存过期后的第一次查询)。这里有一个必须理解的权衡:TTL 设得越长,缓存命中率越高、权威服务器压力越小,但如果记录真的发生变更(比如紧急切换服务器 IP),所有还没过期的缓存在 TTL 剩余时间内都会继续返回旧答案——这也是为什么计划内的 IP 迁移通常会提前把 TTL 调低(比如从默认几小时调到几分钟),等迁移窗口过后再调回正常值,用短暂的缓存效率损失换取变更生效的及时性。

**AI 研究/工程场景:** 蓝绿部署/金丝雀发布如果依赖 DNS 切换流量(把域名从旧集群 IP 切到新集群 IP),必须提前规划好 TTL 策略——如果切换前 TTL 是默认的几小时甚至一天,即使权威记录已经更新,大量客户端本地缓存仍会在几小时内继续访问旧集群,导致"发布已完成但用户还在用旧版本"的诡异现象,这是生产环境值班工程师需要提前排查的真实坑,不是理论假设。

**可运行例子(验证环境:`.venv`,真实用 `time.sleep` 让缓存条目跨越 TTL 边界,验证过期前命中缓存、过期后触发真实重新解析):**

```python
import time

cache = {}
resolve_call_count = {"n": 0}


def authoritative_resolve(qname):
    resolve_call_count["n"] += 1  # 每调用一次,代表真的走了一趟权威服务器解析
    return "93.184.216.34"


def cached_resolve(qname, ttl):
    now = time.monotonic()
    if qname in cache:
        ip, expire_at = cache[qname]
        if now < expire_at:
            return ip, "HIT"
    ip = authoritative_resolve(qname)
    cache[qname] = (ip, now + ttl)
    return ip, "MISS"


# 第一次查询:缓存是空的 -> MISS,触发真实解析。
ip1, status1 = cached_resolve("www.example.com", ttl=0.3)
assert status1 == "MISS" and resolve_call_count["n"] == 1

# 紧接着第二次查询:还在 TTL 有效期内 -> HIT,不触发新的解析调用。
ip2, status2 = cached_resolve("www.example.com", ttl=0.3)
assert status2 == "HIT" and resolve_call_count["n"] == 1
assert ip1 == ip2

# 真实等待超过 TTL 时长 -> 缓存条目确实过期 -> 第三次查询必须 MISS,触发新的真实解析。
time.sleep(0.4)
ip3, status3 = cached_resolve("www.example.com", ttl=0.3)
assert status3 == "MISS" and resolve_call_count["n"] == 2

print(f"1st={status1}(resolve calls=1), 2nd={status2}(within TTL, calls stay 1), "
      f"after real TTL expiry 3rd={status3}(calls={resolve_call_count['n']})")
```

**面试怎么问+追问链:**
- Q:如果权威服务器的记录改了,但客户端还在用缓存的旧值,怎么办?
  - 追问1(规模递增轴):单机应用的本地缓存问题不大,但如果是一个有几百万日活用户、缓存分散在全球各地 ISP 递归解析器里的场景,如何尽量缩短"记录已更新但仍有用户读到旧值"的时间窗口?
  - 深挖追问:核心手段就是提前调低 TTL——但要清楚这个操作本身也需要提前规划(旧的、TTL 更长的记录已经被缓存在外部无法控制的 ISP 解析器里,调低 TTL 只对"调整后新发出的查询"生效,对已经缓存的旧记录不起作用),所以实践中通常提前数小时到一天调低 TTL,等待旧缓存自然过期,再执行真正的记录变更,最后再把 TTL 调回正常值——这是候选人是否理解"TTL 变更本身也有滞后性"的分水岭。

**常见坑:**
- 以为"清空自己电脑的 DNS 缓存(如 `ipconfig /flushdns`)"就能让所有人立刻看到最新记录——本地缓存只是众多缓存层级中的一层,ISP 递归解析器、其他用户设备上的缓存完全不受你本地操作的影响,清本地缓存只解决"你自己这台机器"的问题。

---

## KP4. DNS 记录类型(A / AAAA / CNAME / MX / TXT)

**签名/是什么:**

```
A     :域名 -> IPv4 地址
AAAA  :域名 -> IPv6 地址
CNAME :域名 -> 另一个域名(别名,最终必须递归解析到一个 A/AAAA 记录才算真正解析完成)
MX    :域名 -> 负责接收该域名邮件的邮件服务器域名(带优先级数字,数字越小优先级越高)
TXT   :域名 -> 任意文本(常用于域名归属验证、SPF/DKIM 反垃圾邮件策略声明等)
```

**一句话:** A/AAAA 是"终点"记录(直接给出 IP),CNAME 是"指路"记录(告诉你去问另一个名字,可能要连续问好几次才能到终点),MX/TXT 是服务发现/元数据类记录,不直接参与"这个域名对应哪个 IP"这件事。

**底层机制/为什么这样设计:** CNAME 存在的核心价值是让"域名指向哪里"和"底层实际部署在哪里"解耦——比如网站用了 CDN 服务,`www.example.com` 可以设置成 CNAME 指向 CDN 服务商分配的域名(如 `edge.cdn.example.com`),CDN 服务商随时可以调整这个域名底层对应的真实 IP,网站运营方完全不需要关心底层 IP 变化,也不需要每次 CDN 侧调整时手动去改自己的 DNS 记录。但 CNAME 有一条容易被忽视的协议限制:一个域名如果设置了 CNAME 记录,就不能在同一个名字上再设置其他任何类型的记录(包括 MX)——这也是为什么根域名(如 `example.com` 本身,不带 `www`)几乎不会直接设成 CNAME,而是用 A 记录直接指向 IP,或者用云服务商提供的 "ALIAS/ANAME" 这类非标准扩展变通实现同样的效果。MX 记录不能指向一个 CNAME(必须直接指向一个能被 A/AAAA 解析的名字),这条规则同理是为了避免"发邮件时既要查 MX、又要顺着 CNAME 链条追、还要担心链条中间掺入奇怪记录类型"这种不必要的复杂度。

**AI 研究/工程场景:** 模型 serving API 的域名规划通常会用 CNAME 把面向用户的域名(如 `api.example.com`)指向云负载均衡器/API 网关分配的域名,云服务商随时可以在后台扩缩容、更换底层 IP 而不需要通知用户改任何配置——这正是 CNAME"解耦域名和底层实现"这个设计价值在真实工程里的直接应用。

**可运行例子(验证环境:`.venv`,构造带 CNAME 链的模拟区域数据,真实走一遍链式解析到最终 A 记录,并验证 MX/TXT 记录读取):**

```python
zone = {
    ("www.example.com", "CNAME"): "webserver.example.com",
    ("webserver.example.com", "CNAME"): "edge.cdn.example.com",
    ("edge.cdn.example.com", "A"): "93.184.216.34",
    ("edge.cdn.example.com", "AAAA"): "2606:2800:220:1:248:1893:25c8:1946",
    ("example.com", "MX"): "10 mail.example.com",
    ("example.com", "TXT"): "v=spf1 include:_spf.example.com ~all",
}


def resolve_with_cname_chain(qname, rtype, max_hops=10):
    current = qname
    chain = []
    for _ in range(max_hops):
        if (current, rtype) in zone:
            return zone[(current, rtype)], chain
        if (current, "CNAME") in zone:
            chain.append(current)
            current = zone[(current, "CNAME")]
            continue
        return None, chain
    raise RuntimeError("CNAME loop / too many hops")


# A 记录查询要顺着两层 CNAME 链才能到达真正的 IP。
ip, chain = resolve_with_cname_chain("www.example.com", "A")
assert ip == "93.184.216.34", ip
assert chain == ["www.example.com", "webserver.example.com"], chain

# 同一条 CNAME 链对 AAAA 类型查询同样适用(链条本身与记录类型无关)。
ipv6, _ = resolve_with_cname_chain("www.example.com", "AAAA")
assert ipv6 == "2606:2800:220:1:248:1893:25c8:1946"

mx = zone[("example.com", "MX")]
assert mx.split()[1] == "mail.example.com"

txt = zone[("example.com", "TXT")]
assert "spf1" in txt

print(f"A-via-CNAME-chain={ip} (chain={chain}), AAAA={ipv6[:12]}..., MX={mx}, TXT contains SPF marker")
```

**面试怎么问+追问链:**
- Q:CNAME 和 A 记录能不能设置在同一个域名上?
  - 追问1:那 `example.com` 这种根域名如果也想接入 CDN(通常要求用 CNAME),但同时又要收邮件(需要 MX 记录),怎么办?
  - 深挖追问(方案批判迭代轴):这正是根域名 CNAME 限制在真实工程里制造的具体麻烦——标准做法是用云服务商的 "ALIAS"/"ANAME"(非标准 RFC 扩展,行为像 CNAME 但只在权威服务器内部生效、返回给客户端时已经展开成 A 记录,所以不违反"不能和其他记录共存"的限制),或者干脆只让子域名(`www.example.com`)接 CDN,根域名用 301 重定向跳转到 `www` 子域名。这条追问检验候选人是否只知道"CNAME 有限制"这个孤立事实,还是理解这个限制在真实场景里怎么被工程化地绕过。

**常见坑:**
- 混淆 CNAME 链条的"客户端可见层"和"底层实现层"——很多云 DNS 服务商的控制台允许用户在根域名上配置看起来像 CNAME 的"智能解析",但协议层面实际返回给客户端的是展开后的 A 记录(即 ALIAS/ANAME 机制),这和标准 CNAME 是两种不同的实现,面试被追问底层协议细节时要能区分"控制台展示的抽象"和"协议报文里实际的记录类型"。

---

## KP5. DNS 负载均衡与 GeoDNS(概念性)

**签名/是什么:**

```
DNS 轮询(Round Robin):一个域名配置多条 A 记录,每次查询返回的记录排列顺序不同,
                       客户端通常取第一条使用,从而把流量粗略地分摊到多台后端服务器。
GeoDNS:权威服务器根据发起查询的客户端(或其递归解析器)的地理/网络位置,
        对不同位置的查询返回不同的 IP —— 通常是离该位置"最近"的数据中心地址。
```

**一句话:** DNS 轮询是"给不同的查询,随机换一个顺序回答"的粗粒度负载均衡,GeoDNS 是"给不同位置的查询,回答不同的、离得更近的服务器"的位置感知式负载均衡,两者都发生在 DNS 解析阶段,而不是流量真正打到服务器之后。

**底层机制/为什么这样设计:** DNS 轮询的粗糙之处在于它对后端服务器的真实健康状态和负载一无所知——即使某台服务器已经宕机或严重过载,权威服务器依然会按顺序把它的 IP 排在某些查询的最前面,而且更麻烦的是,一旦某个答案被缓存(见 KP3),在 TTL 到期之前客户端会持续访问缓存里那个可能已经不可用的 IP,DNS 层面完全没有办法主动纠正。这也是为什么 DNS 轮询通常只用作"简单、粗粒度的流量打散",真正对后端健康敏感的负载均衡决策会放到 [10 类](10-modern-app-protocols-and-apis.md) 讨论的四层/七层负载均衡器去做(它们能实时探测后端健康状态,DNS 记录改一次要等 TTL 过期才生效,负载均衡器则是每个请求都能实时决策)。GeoDNS 解决的是另一个维度的问题:同一个服务在全球部署了多个数据中心,理想情况下每个用户应该被导向离自己最近(网络延迟最低)的那一个,但 DNS 协议本身没有"客户端在哪里"这个信息——GeoDNS 的实现依赖权威服务器识别发起查询的递归解析器的 IP(近似代表查询者的大致地理位置,不是精确定位),查表得出"这个位置该导向哪个数据中心",给不同位置的查询返回不同的答案。这个机制的精度受限于"递归解析器的位置不一定等于终端用户的真实位置"(比如用户手动配置了千里之外的公共 DNS),这也是为什么后来出现了 EDNS Client Subnet 这类扩展,让递归解析器把客户端的网段信息也带给权威服务器,提高定位精度。

**AI 研究/工程场景:** 全球化部署的模型推理服务如果只有单一区域的入口,海外用户会因为跨国网络延迟(可能几百毫秒)显著拖慢首字返回时间;GeoDNS(或者更精细的 [10 类](10-modern-app-protocols-and-apis.md) 提到的 CDN/Anycast 方案)把用户导向地理上最近的推理集群入口,是跨国 AI 服务优化用户体验延迟的第一道、也是最简单直接的一道优化手段。

**可运行例子(验证环境:`.venv`,真实验证轮询确实轮转到全部后端、GeoDNS 表确实按客户端位置返回不同数据中心):**

```python
# DNS 轮询:多条 A 记录,每次查询把不同的记录排在最前面。
pool = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
rr_state = {"i": 0}


def round_robin_resolve():
    n = len(pool)
    rotated = pool[rr_state["i"] % n:] + pool[:rr_state["i"] % n]
    rr_state["i"] += 1
    return rotated[0]  # 客户端通常取第一条使用


seen_first = [round_robin_resolve() for _ in range(6)]
assert seen_first == ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.1", "10.0.0.2", "10.0.0.3"]
assert len(set(seen_first)) == 3, "6次查询里必须真的轮转到全部3个后端,而不是每次都返回同一个"

# 玩具版 GeoDNS:把(虚构的)客户端网段前缀映射到最近的数据中心。
geo_table = {
    "203.0.113.": "us-east-datacenter-ip",
    "198.51.100.": "eu-west-datacenter-ip",
    "192.0.2.": "ap-southeast-datacenter-ip",
}


def geo_resolve(client_ip):
    for prefix, dc_ip in geo_table.items():
        if client_ip.startswith(prefix):
            return dc_ip
    return "default-datacenter-ip"


us_client = geo_resolve("203.0.113.55")
eu_client = geo_resolve("198.51.100.20")
assert us_client == "us-east-datacenter-ip"
assert eu_client == "eu-west-datacenter-ip"
assert us_client != eu_client, "不同地理位置的客户端必须解析到不同的最近数据中心"

print(f"round-robin sequence={seen_first}; geoDNS us_client->{us_client}, eu_client->{eu_client}")
```

**面试怎么问+追问链:**
- Q:DNS 轮询能实现精确的负载均衡吗(比如让流量按 7:3 精确分配到两台服务器)?
  - 追问1:那为什么不干脆都用四层/七层负载均衡器,DNS 轮询还有什么存在的价值?
  - 深挖追问(决策依据追问轴):DNS 轮询无法精确控制比例(客户端行为、中间缓存都会干扰实际分配比例,达不到"精确"二字),但它的价值在于负载均衡器本身覆盖不到的场景——比如把流量分配到"完全独立、跨地域甚至跨云厂商的多个数据中心入口",这种场景下不可能有一台集中的负载均衡器同时坐在所有数据中心前面(那样反而制造了新的单点和额外的一跳延迟),DNS 轮询/GeoDNS 天然适合做这种"顶层、粗粒度、跨数据中心"的流量入口分配,细粒度的负载均衡决策再交给每个数据中心内部的负载均衡器去做——两者是分层协作,不是互相替代的关系。

**常见坑:**
- 把 GeoDNS 的"就近接入"和 CDN 的"内容就近缓存"([10 类](10-modern-app-protocols-and-apis.md))当成同一回事——GeoDNS 只是让你连接到离你近的服务器入口,连接之后这台服务器有没有你要的内容、内容新不新鲜,是 CDN 缓存策略要解决的完全独立的问题,一个偏"路由入口选择",一个偏"内容分发缓存"。

---

## KP6. DNS 劫持与 DNSSEC(概念性)

**签名/是什么:**

```
DNS 劫持(DNS Hijacking):攻击者通过篡改客户端的 DNS 设置、污染中间网络的查询响应、
                         或攻陷递归解析器本身,让受害者对某个域名的查询得到错误(通常是恶意)的 IP 地址。
DNSSEC(DNS Security Extensions):给 DNS 记录加上数字签名(RRSIG 记录),
                                 客户端/解析器可以验证收到的记录确实来自持有对应私钥的权威服务器、
                                 且传输过程中未被篡改,验证失败则拒绝该答案。
```

**一句话:** 普通 DNS 协议本身不提供任何"这个答案真的是权威服务器给的、没被中途篡改"的验证机制,DNS 劫持正是利用了这个信任真空,DNSSEC 用数字签名给每条记录补上这道验证防线。

**底层机制/为什么这样设计:** 原始 DNS 协议设计时(1980 年代)完全没有考虑对抗性网络环境下的安全性——UDP 报文可以被任意伪造,只要伪造的响应能在真实权威服务器的响应之前(或者干脆没有真实响应竞争)到达客户端,客户端没有任何机制识别"这个响应是不是真的来自我问的那台权威服务器"。这个信任真空可以在链路上的任意一环被利用:客户端本地 DNS 设置被恶意软件篡改指向攻击者的服务器、公共 WiFi 网络的路由器在转发 DNS 查询时偷偷替换响应内容、甚至递归解析器本身被攻陷后对特定域名撒谎——受害者在这些情况下访问的域名字符串完全正确,但拿到的 IP 是攻击者控制的服务器,常见于钓鱼攻击(伪装成银行/邮箱登录页盗取凭证)。DNSSEC 的解法是引入公钥签名体系(和 [08 类 KP3](08-https-and-tls.md) 的证书信任链是同一类思路,但应用在 DNS 记录本身而不是 TLS 证书上):权威服务器用自己的私钥对区域内的记录签名,签名结果作为 RRSIG 记录一起发布,验证方用对应公钥验证签名,任何篡改都会导致签名验证失败——这样即使中间人能拦截并替换响应内容,只要它没有权威服务器的私钥,就无法伪造出一份能通过验证的签名,客户端可以明确拒绝这份被篡改的答案而不是被动接受。

**AI 研究/工程场景:** 对外提供 API 服务的公司如果域名遭遇 DNS 劫持,后果可能是用户的 API 请求(包含调用密钥、敏感输入数据)被悄悄导向攻击者控制的服务器截获——这也是为什么面向企业客户的关键业务域名往往会开启 DNSSEC 并配合证书透明度日志监控(检测是否有未经授权的证书被签发),作为域名劫持类供应链攻击的纵深防御手段之一。

**可运行例子(验证环境:`.venv`;DNSSEC 部分用 HMAC 对称签名模拟真实的非对称签名机制,以适配纯标准库约束——核心思想完全一致:被签名的记录能通过验证,被篡改的记录无法通过验证):**

```python
import hashlib
import hmac

# ---- 场景A:DNS 劫持暴露 —— 一个不做任何验证的客户端无法分辨真实答案和被劫持的答案。
legit_zone = {"www.example.com": "93.184.216.34"}
rogue_zone = {"www.example.com": "6.6.6.6"}  # 攻击者控制的答案,比如指向钓鱼页面


def naive_client_resolve(zone, qname):
    return zone.get(qname)  # 无条件相信返回结果,不做任何验证


legit_answer = naive_client_resolve(legit_zone, "www.example.com")
hijacked_answer = naive_client_resolve(rogue_zone, "www.example.com")
assert legit_answer != hijacked_answer
print(f"legit resolver -> {legit_answer}, rogue/hijacked resolver -> {hijacked_answer} "
      f"(a naive client cannot tell these apart -- this IS the hijack exposure)")

# ---- 场景B:DNSSEC 式签名验证(简化为 HMAC 对称签名,真实 DNSSEC 用非对称公钥签名/RRSIG,
# 核心思想相同:签名和记录内容绑定,篡改记录内容会导致签名验证失败)。
zone_signing_key = b"toy-zone-signing-key-not-real-dnssec"


def sign_record(qname, ip):
    msg = f"{qname}:{ip}".encode()
    return hmac.new(zone_signing_key, msg, hashlib.sha256).hexdigest()


def verify_record(qname, ip, signature):
    expected = sign_record(qname, ip)
    return hmac.compare_digest(expected, signature)


real_sig = sign_record("www.example.com", "93.184.216.34")
assert verify_record("www.example.com", "93.184.216.34", real_sig) is True

# 攻击者篡改了 IP,但没有 zone_signing_key,无法伪造出一份匹配的新签名 —— 用旧签名去验证新数据必须失败。
tampered_ip = "6.6.6.6"
assert verify_record("www.example.com", tampered_ip, real_sig) is False
print("signature verifies for the ORIGINAL record, fails for a TAMPERED record with the same signature "
      "-- this is DNSSEC's core defense")
```

**面试怎么问+追问链:**
- Q:开启了 DNSSEC 是不是就能彻底杜绝 DNS 劫持?
  - 追问1:DNSSEC 只解决"记录内容有没有被篡改"这一个问题,那客户端本地 DNS 设置被恶意软件直接改成攻击者的恶意解析器地址,DNSSEC 还能防住吗?
  - 深挖追问(方案批判迭代轴):防不住——如果客户端一开始问的就是攻击者控制的"解析器"(而不是被中间人在链路中篡改了一个合法解析器的正常响应),攻击者完全可以对不支持 DNSSEC 校验的查询直接伪造整套记录(包括伪造"这个域名没有部署 DNSSEC"的响应,诱使客户端跳过验证——这类降级攻击是 DNSSEC 部署中一个真实存在的薄弱环节);DNSSEC 防的是"链路中间的篡改",防不住"客户端一开始问的对象就是恶意的"这类更前置的攻陷。这条追问和 [08 类 KP4](08-https-and-tls.md) 的"HTTPS 能否杜绝 MITM"追问链是完全同构的方法论——任何一层安全机制都只能防御自己所在的那个特定信任假设被破坏的场景,不能替代其他层。

**常见坑:**
- 认为 DNSSEC 会对 DNS 查询内容本身加密——DNSSEC 解决的是"完整性和来源真实性"(记录有没有被篡改、是不是真的来自权威服务器),完全不提供机密性(任何人依然能看到你在查询什么域名,报文本身没有被加密);如果需要查询内容本身的机密性,需要的是 DNS over HTTPS(DoH)或 DNS over TLS(DoT)这类独立的加密传输机制,两者解决的是正交的两个问题,不能互相替代。

---

*本篇完成:2026-07-14,6 个知识点。验证环境:全部 6 个可运行代码块均为 `.venv`(模拟三级 DNS 层级委派、递归vs迭代真实查询次数对比、缓存TTL真实过期复现、CNAME链式解析、DNS轮询与GeoDNS真实验证、DNS劫持暴露与DNSSEC签名校验模拟)——全部使用虚构域名/IP构造的本地模拟数据,不对真实公网DNS发起任何查询,遵守本系列"不引入外部网络依赖"纪律。板块 IV(应用层协议)进度 3/4。*
