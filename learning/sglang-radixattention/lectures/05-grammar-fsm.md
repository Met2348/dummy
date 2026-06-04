# L05 · Grammar FSM — char-level 状态机

## 1 · FSM 抽象
- `states: Set[S]`
- `start: S`
- `accept: Set[S]`
- `delta: (S, char) -> S`

## 2 · regex → NFA → DFA → minimized DFA
经典三步：
1. Thompson 构造 NFA
2. 子集构造 DFA
3. Hopcroft 最小化

输出后每个状态都有清晰的"下一个合法字符集"。

## 3 · JSON schema → FSM
JSON 有递归结构，严格说需要 PDA (push-down)：
- 但实践中 schema 深度有限 → 展开成 DFA
- xgrammar 用 "early termination" DFA：到 `}` 必停

例 schema `{"name":"<string>","age":<integer>}`：
```
start --'{'--> s1 --'"'--> s2 ... --'}'--> accept
```

## 4 · token-level lift
char-level FSM → token-level mask 表：
```python
TOKEN_NEXT_STATE: Dict[(state, token), state | INVALID]
```
预编译时：对每个 (state, token)，模拟把 token 的 chars 喂给 FSM；若任一 char 失败 → INVALID。

存储：`state × |V|` 大小，bit-packed。例 V=50k, 200 state → 1.25 MB / grammar。

## 5 · 性能
- 朴素遍历所有 token：每 step 50k 查询，慢
- 预编译表：每 step O(1)
- xgrammar C++ 实现，每 step < 10 µs

## 6 · 边界：unicode / BPE merge
- BPE token 包含多个 unicode 字节
- 字节级 BPE 时 token 可能不对齐 char → 需要按 byte 比较
- xgrammar 的 "compressed" 设计：每 byte 一节点

## 7 · 实现：[grammar_fsm.py](../src/grammar_fsm.py)
- `RegexFsm` 简化版（只支持 `\d+`, `\w+`, 字面量）
- `compile_token_table(fsm, vocab)`
- 教学版用 char level，不做 BPE merge
