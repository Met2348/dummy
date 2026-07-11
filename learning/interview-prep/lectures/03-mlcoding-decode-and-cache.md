# L03 · mlcoding 打法:解码采样 + KV cache 不变量

对应代码:`src/mlcoding/{sampling,kv_cache,lora,training_loop,bpe}.py`

## 采样:top-k vs top-p 的一句话区分

- **top-k**:固定候选**个数** k。分布尖锐时够用,但平坦时会砍掉合理选项。
- **top-p(nucleus)**:固定累计概率**质量** p,候选数自适应——尖锐时少、平坦时多。更常用。
- **temperature**:`logits/T`。T<1 更尖锐(确定)、T>1 更平坦(多样)。`sampling.py` 验升温使熵增。

> 实现坑:top-p 要保留"累计首次 ≥ p 的那一个"。self_test 里 `remove` 右移一位就是干这个——漏了会少留一个候选。

## beam search:为什么它能破"贪心陷阱"

贪心每步取局部最优,可能咬死一条次优路径。`sampling.py` 构造了一个陷阱:第一步 token1 略高,但选 token0 后第二步有大奖励。贪心选 [1,…],beam(宽度2)找到全局最优 [0,2]。**面试点**:beam 是"宽度受限的 BFS",按**累计 logprob** 排序保留 top-w。

## KV cache:面试杀手锏

面试官必问:"你怎么证明缓存版和全量版一致?" `kv_cache.py` 的 self_test 就是答案——**逐 token 走缓存路径,与一次喂全序列的输出逐元素对拍**(err 1e-7)。

**为什么能缓存**:因果注意力下,token t 的表示只依赖 ≤t 的 K/V,历史 K/V 不随新 token 改变 → 存下来复用,每步从 O(T²) 降到 O(T)。**推论**:只有 decoder(因果)能用,encoder(双向)不能——这是常见追问。

## LoRA:三个必答点

1. `ΔW = B·A`,A 随机、**B 初始化为 0** → 起始等价原层(不破坏预训练)。
2. 只训 A,B;**基座 `requires_grad=False`**。`lora.py` 验基座无梯度。
3. 缩放 `alpha/r`。参数量从 `in×out` 降到 `r×(in+out)`。

## training_loop:四步不能漏

`zero_grad → backward → clip_grad_norm → step`。漏 `zero_grad` → 梯度累积;漏 clip → Transformer 易爆。再加 **warmup+cosine** LR(纯函数单独测边界 step=0/warmup/total,这是最容易写错的地方)。

## bpe:tokenizer 从哪来

BPE 贪心反复合并语料中**最高频相邻符号对**;encode 按学到的 merge 顺序应用,decode 只是拼接。`bpe.py` 验往返还原 + 高频对被合并。面试问"tokenizer 怎么训"时答这个。
