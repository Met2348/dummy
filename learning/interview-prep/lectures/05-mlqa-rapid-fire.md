# L05 · rapid-fire:30 秒说清的刻意练习

对应代码:`src/mlqa/qbank.py`(45 题 / 8 类)

## 失分点不是"不知道",是"说不清 30 秒版"

基础轮节奏快、一题一两分钟。你可能全懂,但**组织不出简洁答案**就丢分。解法:每题背一个**采分点集合**,答案里必须蹦出那几个词。

## 关键词自评:把开放问答变刻意练习

`qbank.py` 每题存 `keys`(采分关键词),`grade(你的答案, qa)` 按命中率打分:

```python
from mlqa.qbank import BANK, grade, quiz
qa = quiz("归一化")[0]          # 取一题
print(qa.q)                      # 大声答出来
print(grade("我的口头答案……", qa))   # 命中率 0..1
print(qa.a)                      # 再对标准答案
```

## 8 类覆盖

优化 / 正则化 / 归一化 / 指标 / Transformer / RL·RLHF / 可解释性 / 泛化·统计。

**几个高频必背(采分点)**:
- **BN vs LN**:batch 维 / 特征维 / 推理移动平均 / 逐样本。
- **为何 /√d**:点积方差随 d 增长 / softmax 饱和 / 梯度。
- **PPO clip**:概率比 / 裁剪 / 防一步走太远。
- **RLHF 三段**:SFT / 奖励模型 / PPO+KL。
- **reward hacking**:钻奖励空子 / 高分不达真目标 —— 你 PhD 方向。
- **bias-variance**:偏差² + 方差 + 噪声 / 欠拟合 vs 过拟合。

## 和你 PhD 方向的连接

`可解释性` 那一类(probing / logit lens / activation patching / SAE / superposition / linear representation)不只是面试题——它们是你 judge-internals 研究的日常词汇。**面试里能把这些讲成研究品味,是研究岗的强信号**。

## 练法(每天 5 题)

用 `quiz()` 抽 5 题 → 大声(录音)答 30 秒 → `grade()` 自评 → 命中率 <0.6 的进 `tracker` 重刷。目标:任意一题都能**不打磕、结构化、30 秒**。
