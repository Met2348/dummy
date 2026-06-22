# L08 · 安全与控制：disposes 的全部

> Part II · 40-min lecture · 目标: 本构要素 ④ control 的正面战场——权限门、destructive-action hooks、预算护栏、防失控、企业级 RBAC/审批。把 agent 从「危险黑箱」变成「可托付系统」。

---

## 0. 没有控制层，agent 就是个会执行任意命令的黑箱

回忆 L02 的核心架构: **Model proposes, harness disposes.** 模型提议动作 (可能是 `rm -rf /`、可能是把密钥发到外网), harness 必须在动作落到真实世界**之前**做处置。没有这一层, agent 会:
- 漏 token (无意义地烧钱)
- 无限循环 (失控)
- 执行危险命令 (灾难)

控制层就是把这些堵住。

---

## 1. 五道控制闸（从轻到重）

```
模型提议一个动作
   │
   ├─ ① schema 校验      参数合法吗? (类型/必填/注入)          ← L07 已讲
   ├─ ② 权限门           这个动作这个角色现在能做吗?
   ├─ ③ destructive hook 这是不可逆/危险动作吗? (删/写/外发)
   ├─ ④ 预算护栏         还在 token/步数/时间/花费预算内吗?
   └─ ⑤ 审批工作流       高危动作 → 暂停, 等人/上级 agent 批准
   │
   ▼ 全过 → 执行 → 记录 (L09 trace)
```

### ② 权限门（permission gate）
你 Module 7 `permissions.py` 的 `auto / readonly / ask` 三档就是它的最小版。生产级再加**范围限定** (这个 agent 只能动这个目录 / 这些 API)。

### ③ destructive-action hooks
> **destructive-action hooks 坐在权限门后面。** 对「不可逆」动作 (删除、覆盖、对外发送、不可撤销的 API 调用) 单独拦一道——即使权限允许, 也要额外确认。

Hashimoto 的精神在这里落地: agent 犯过一次危险操作? **改环境** (加一个 hook 让这类操作必须确认), 而不是 prompt 它「请小心」。

### ④ 预算护栏（防失控）
你 Module 7 `errors.py` 的 `LoopGuard` 就是。生产级护栏覆盖多维: **token 上限 / 步数上限 / 墙钟时间 / 美元花费**。任一超限 → 停。长任务 (L05) 尤其需要——跨窗口跑几小时, 没有预算护栏会烧到失控。

### ⑤ 审批工作流（企业级）
高危/高影响动作 → harness **暂停**, 推给人 (或更高权限的 agent) 审批, 批了再继续。这是企业部署的硬需求。

---

## 2. 企业级：RBAC、审计、身份

把 agent 放进企业, 控制层要再加一圈「**operational requirements (运营要求)**」——这正是「玩具 harness」和「企业 runtime」的分界:

```
玩具 harness            企业级 runtime (AWS AgentCore / Microsoft Agent Framework ...)
─────────────          ────────────────────────────────────────────────────
auto/ask 权限      →    RBAC (基于角色的访问控制)
本地 trace        →    审计日志 (谁、何时、做了什么, 可追溯)
无身份            →    身份 (agent/用户的认证, identity)
直连              →    VPC 网络隔离 + 内部系统接入授权
随手跑            →    审批工作流 (approval workflows)
```

> 记住这张表: 当有人问「你的 agent 能上企业生产吗」, 他们真正在问的是这一列**运营要求**满足了没——而不是模型多强。AWS AgentCore (带身份/VPC/可观测)、Microsoft Agent Framework (Azure/.NET) 是两个把这些打包好的托管例子。

---

## 3. 安全与「65% 失败」的关系

回忆 L01: 65% 的 harness 失败源于 context drift / schema 错配 / state 退化。控制层直接对治其中两类:
- **schema 错配** ← ① schema 校验
- **state 退化** ← ②③④ 防止危险/失控动作把 state 搞坏 + L05 的文件系统真相
- (context drift 主要靠 L04 compaction 的「该 pin 的别丢」)

所以**控制层不是「锦上添花的安全」, 它直接决定 agent 能不能活到生产。**

---

## 4. 本讲小结 + 通往 Part III

- 控制层 = disposes 的全部: schema 校验 → 权限门 → destructive hook → 预算护栏 → 审批。
- 精神: 出过危险? 改环境加 hook, 别 prompt 求它小心 (Hashimoto)。
- 企业级再加一圈运营要求: RBAC / 审计 / 身份 / VPC / 审批——这是玩具与生产 runtime 的真正分界。
- 控制层直接对治 65% 失败里的 schema 错配与 state 退化。

> **Part III 开始 (L09-L12)**: 你的 harness 现在能接真模型、管上下文、跑长任务、受控。但你**看得见它在干什么吗? 测得出它好不好吗? 它可移植吗?** 这是成熟度三件套。L09 先讲生产可观测性——OpenTelemetry 式的 trace, 配套 `src/otel_trace.py`。
