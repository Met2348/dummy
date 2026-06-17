# L14 · Hooks 与可扩展性

> 概念 + 在 mini-harness 里的扩展点说明。

## 为什么 harness 要可扩展

一个好的 harness 不该把所有行为写死。它应提供**扩展点**,让用户在不改核心 loop 的前提下:注册新工具、在关键时刻插入自定义逻辑、接入外部能力(MCP)。

## 三类扩展机制

### 1) 工具注册(已实现)
[tools.py](../src/harness/tools.py) 的 `ToolRegistry.add(name, desc, fn)` 就是最基本的扩展点——加能力 = 注册工具,核心 loop 不动:

```python
reg.add("my_new_tool", "does X", my_fn, read_only=True)
```

### 2) Hooks(生命周期回调)
在 loop 的关键时刻触发用户回调,改变/观察行为而不改 loop。挂载点对应 [loop.py](../src/harness/loop.py) 的各阶段:

| Hook 点 | 时机 | 典型用途 |
|---------|------|---------|
| `pre_model` | 调模型前 | 注入额外 context、改 prompt |
| `pre_tool` | 执行工具前 | 额外校验、改写参数、审计 |
| `post_tool` | 工具返回后 | 裁剪结果、记录、触发副作用 |
| `on_finish` | 收尾时 | 持久化、通知 |

> 权限检查(L09)本质就是一个内建的 `pre_tool` hook;工具结果裁剪(L05)是内建的 `post_tool` hook。把它们抽象成可注册的 hook,就得到通用扩展机制。

### 3) MCP / 外部工具接入
Model Context Protocol(见 `tool-use-mcp` 专题)让 harness 以标准协议**动态发现**外部工具服务器,把远端工具注册进本地 registry。harness 侧只需:连接 MCP server → 拉取工具 schema → `registry.add` 适配器。

## 设计要点

1. **核心 loop 要瘦**:能力靠注册/hook 加,而不是往 loop 里堆 if-else。
2. **hook 要在安全点**:和 steering 一样,在回合/工具边界触发。
3. **扩展不破坏可观测**:hook 行为也该进 trace,别成为黑箱。
4. **MCP 工具同样过权限**:外部来的工具更要走 permission 闸。

## 与本仓库的关系(诚实说明)

mini-harness 已实现"工具注册"这一扩展点;hooks 和 MCP 接入是**架构留好的位置**(权限/裁剪已是内建 hook 的雏形)。把 `pre_tool`/`post_tool` 抽成可注册列表,即得到完整 hook 系统——这是读者可自行扩展的练习。

## 退出条件
- [ ] 说出三类扩展机制
- [ ] 指出 hooks 在 loop 的挂载点
- [ ] 理解"核心 loop 瘦、能力靠扩展"的原则
