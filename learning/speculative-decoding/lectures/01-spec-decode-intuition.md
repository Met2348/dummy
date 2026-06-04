# L01 · 投机解码直觉（猜测 + 验证）

## 1 · decode 瓶颈
- LLM decode：每步 1 token，memory-bound
- GPU FLOPs 利用率 < 5%
- 主存读：每步 7B fp16 = 14 GB 必须从 HBM 读

## 2 · 关键观察
**Verify k tokens 用并行 forward 几乎和 verify 1 token 一样快**：
- HBM 读时间为主
- 多算 k 个位置的 logits 只增加 k × tiny compute

## 3 · 投机解码范式
1. **Draft**: 用小快模型猜 k 个 token (`q(y|x)`)
2. **Verify**: 大模型一次性 forward 这 k+1 个位置，得 k+1 个 `p(y|x)`
3. **Accept/Reject**: 按 `p/q` 概率接受，第一个被拒绝的位置起重新 sample

## 4 · 速度公式
- 朴素：每 token 1 大 forward
- 投机：每 (accept_k+1) token = 1 小 forward × k + 1 大 forward
- 大模型 forward 比小快 N× → 加速比 ≈ `(accept_k + 1) / (1 + k/N)`

例：accept=3, k=5, N=20 → 加速 ≈ 4 / 1.25 ≈ **3.2x**

## 5 · 关键 trade-off
- k 太小 → 加速少
- k 太大 → reject 多浪费
- draft 太弱 → accept 率低
- draft 太强 → draft 自己慢

最佳点取决于具体 task。

## 6 · 谱系
| 方法 | draft 来源 |
|------|----------|
| Classic SpecDec | 独立小模型 |
| Medusa | 同模型多 head |
| EAGLE-1/2/3 | 同模型 feature 投机 |
| Lookahead | n-gram + Jacobi |
| Self-SpecDec | skip layer |

## 7 · 一句话
> 投机解码 = **小步快走（draft）+ 大刀验证（verify）**，本质是把 memory-bound 变 compute-bound。
