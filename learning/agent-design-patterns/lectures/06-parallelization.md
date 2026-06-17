# L06 · Workflow 模式 3 — Parallelization

## 两种子形态

并行化指多个**互相独立**的 LLM 调用同时跑(本仓库是顺序执行,但逻辑独立),再聚合。两个用法:

### A) Sectioning(分段)— 把任务切成独立子任务
```
task → ┬→ [评价:价格]  ┐
       ├→ [评价:质量]  ├→ aggregate → output
       └→ [评价:支持]  ┘
```
适合:任务有**互不依赖的多个侧面**,想并行提速 / 关注点分离。

### B) Voting(投票)— 同一任务多次,取共识
```
text → ┬→ [判定 variant0] ┐
       ├→ [判定 variant1] ├→ 多数票 → output
       ├→ [判定 variant2] ┤
       ├→ [判定 variant3] ┤
       └→ [判定 variant4] ┘
```
适合:**单个难判断**(代码有无漏洞、内容是否违规),多次采样投票提可靠性。

## src 走读

[parallelization.py](../src/patterns/parallelization.py) 两个函数:

```python
def run_sectioning(subtasks, worker, aggregate, tracker=None): ...
def run_voting(text, worker, n, aggregate, tracker=None):
    votes = [worker(text, variant=i) for i in range(n)]
    final = aggregate(votes)
    ok = votes.count(final) > n // 2     # 真有多数同意才算可靠
```

- sectioning demo:对产品的 price/quality/support 三面独立打分再拼。
- voting demo:5 个 framing 判情感,3 票 positive 压过 1 neutral + 1 negative。

## 设计要点

1. **sectioning 看"独立性"**:子任务之间不能有依赖,否则该用 chaining。
2. **voting 看"方差"**:确定性 mock 投票无意义,所以 demo 用 `variant` 让每次输出不同——真实里靠采样温度/不同 prompt 制造多样性。
3. **聚合策略**:sectioning 用拼接/综合;voting 用多数票/取并集(召回优先)/取交集(精确优先)。

## 成本视角

并行不省 token(所有分支都跑),省的是**墙钟时间**(wall-clock)。voting 更是 N 倍 token 换可靠性——只对高价值判断值得。

## 退出条件
- [ ] 能区分 sectioning(切任务)和 voting(切采样)
- [ ] 理解并行省时间不省 token
- [ ] 知道聚合策略(多数/并集/交集)各自的取向
