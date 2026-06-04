# L02 · RadixAttention 原理

## 1 · 痛点
vLLM 的 prefix caching：
- 按 BLOCK_SIZE=16 切块 hash
- 公共前缀必须**完全对齐到 16 整数倍**
- "Hello world" vs "Hello there" 前 9 token 相同 → 共享 0（block 不齐）

## 2 · 解：radix tree
经典 trie 数据结构：每个节点存 token 序列前缀，子节点分叉。

```
root
├── [Hello world]→[my friend]→...
├── [Hello there]→...
└── [Bonjour]→...
```

token "Hello world my friend" 进来 → 沿树走 → 命中 [Hello world] + 缺 [my friend] 部分。

## 3 · KV cache 挂在节点上
每个 tree node 存：
- `token_ids`: List[int]
- `kv`: 对应 K/V (per layer)
- `children`: Dict[token, Node]
- `last_access_time`: 时间戳
- `refcount`: 当前引用此前缀的请求数

## 4 · radix vs trie 的差异
- 普通 trie：每节点 1 token → 树极深
- radix tree：节点可存连续 token 段 → 共享分叉点压缩
- 命中后**继续走**有差异时才分裂节点

## 5 · 节点分裂例
- 已存：节点 N = "Hello world my friend"
- 来：req "Hello world my dog"
- 命中前 3 token，第 4 token 起分叉 → 把 N 分裂：
  - N1: "Hello world my"  (公共)
  - N1.children: "friend" (老) + "dog" (新)

## 6 · 内存：每个节点 KV 占用
若节点存 k token，KV = `2 * n_kv_heads * d_h * dtype * k * n_layers`。

例：7B fp16, 8 KV heads, d_h=128, k=20 token, layers=32 → 2 MB / node。

## 7 · 淘汰策略 (LRU)
- 节点带 `last_access_time`
- 满时挑 `refcount=0 && oldest` 节点 evict
- evict 内部节点会破坏树结构 → 只 evict 叶子（**叶子先 evict** 是必须的）

## 8 · 实现：[radix_tree.py](../src/radix_tree.py)
- `Node` / `RadixTree.match()` / `.insert()` / `.evict()`
- 不带真实 KV，存 token 序列 + 模拟内存
