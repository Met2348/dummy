# L02 · mlcoding 打法:attention 家族(得分点在细节)

对应代码:`src/mlcoding/{attention,norm,rope,transformer_block}.py`

## 手写 attention 的真实评分

公式 `softmax(QKᵀ/√d)V` 谁都会背。面试官区分候选人靠**三个"活人才知道"的细节**:

1. **为什么 /√d**:Q·K 是 d 个乘积之和,方差随 d **线性增长**;值太大 → softmax 落入饱和区 → 梯度趋零。除 √d 把打分方差归一。
   > 追问准备:"不除会怎样?"→ 训练初期就梯度消失、注意力退化成近似 one-hot。

2. **causal mask 用 `-inf` 加在 softmax 之前**,不是乘 0 在之后。乘法掩码会让被掩位置仍分到 softmax 概率质量、**泄露未来**。`attention.py` 的 self_test 专门验:改未来的 V 不影响当前输出。

3. **多头靠 reshape**:`(B,T,D) → (B,H,T,D/H)`,不是开 H 个独立矩阵。一次 QKV 投影再 split,`transpose(1,2)` 把头维提前。

## 对拍官方 = 回答"你怎么知道你写对了"

`attention.py` 里手写 SDPA 与 `F.scaled_dot_product_attention` 数值对齐(atol 1e-5)。**面试时主动说这句**:"我会拿 torch 官方 kernel 对拍验证正确性"——展示工程素养。

## norm:BN vs LN 的必答点

- BN 沿 **batch 维**、依赖 batch 统计、推理用移动平均;LN 沿 **特征维**、逐样本、与 batch 无关。
- **Transformer 为何用 LN**:序列变长 + 小 batch,BN 统计不稳;LN 逐样本逐位置,不依赖 batch。
- RMSNorm:省去去均值、只按均方根缩放(LLaMA 用),`norm.py` 验它输出单位 RMS。

## rope:一句话打动面试官

"RoPE 把位置编码成**旋转**,于是 `<RoPE(q,m), RoPE(k,n)>` 只依赖相对位置 `m−n`——这既给了相对位置感、又有更好的长度外推。" `rope.py` 的 self_test 用固定位移验证了这个不变量,并证明旋转保范数。

## transformer_block:pre-norm 的理由

`x + attn(LN(x))` 的残差主干让梯度直通深层、常可免 warmup。`transformer_block.py` 验:把权重清零时 `out == in`(残差直通),说明主干没断。

## 练法

闭卷默写 `attention.py` → 跑 self_test 对拍 → 出错处正是你的盲点。目标:**30 分钟内从空文件写出 MHA + 因果掩码并自证正确**。
