# L02 · Classic Speculative Decoding（Leviathan 2023 / Chen 2023）

## 1 · 论文
- Leviathan *et al.*, "Fast Inference from Transformers via Speculative Decoding" (2023.02 DeepMind)
- Chen *et al.*, "Accelerating Large Language Model Decoding with Speculative Sampling" (2023.02 OpenAI)
- 几乎同时 — 标准范式确立

## 2 · 算法
设 target `p`，draft `q`。

```
1. for i = 1..k: x_i ~ q(.|x_{<i})            # draft k token
2. parallel forward: p(.|x_{<i}) for i=1..k+1
3. for i = 1..k:
     r ~ Uniform(0,1)
     if r < p(x_i)/q(x_i):
         accept x_i
     else:
         reject; sample x_i' ~ norm(max(0, p-q));  break
4. if all k accepted: bonus x_{k+1} ~ p(.|x_{<k+1})
```

## 3 · 正确性证明
> 修正分布 `p_accept(x) = q(x) · min(1, p(x)/q(x)) + (rejected → resample)` 严格等于 `p(x)`。

数学：
- `q(x) min(1, p/q) = min(q, p)`
- `q(x) (1 - min(1, p/q)) · max(0, p-q)/(1-min(q,p)) = (q-min(q,p)) · (p-q)/(1-min(q,p))`
- 两项相加 = `min(p,q) + max(0,p-q) = p`

→ 投机解码**严格等价**于从 target sample，无 bias！

## 4 · accept rate 与 KL
- accept rate α = E[min(1, p/q)] = 1 - 0.5·TV(p,q)
- draft 越接近 target → α 越大
- typical α = 0.5-0.8 for Llama-7B as draft of Llama-70B

## 5 · 失败模式
- draft 与 target 分布差太大 → 浪费多
- temperature 1.0 时 KL 大 → α 低
- T = 0（greedy）时 → 退化为"draft 是否预测同 token"

## 6 · 工程要点
- draft 和 target tokenizer 必须**一致**
- 一致才能复用 prompt KV cache（不一致需双份 prefill）
- 实践常用同族 (Qwen-0.5B + Qwen-7B / Llama-1B + Llama-70B)

## 7 · 实现：[classic_spec_decode.py](../src/classic_spec_decode.py)
- `rejection_sample` 严格算法
- `speculative_decode_loop`
- accept rate 度量

## 8 · 一句话
> "猜得快+验得准+rejection sampling = **零 bias 加速**"。这个公式 2023 后被所有变体继承。
