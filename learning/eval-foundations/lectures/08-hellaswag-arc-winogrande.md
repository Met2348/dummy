# L08 · 经典三件套 — HellaSwag / ARC / Winogrande

> 2019-2023 的"老黄历"，全饱和了，但要懂

## 1. HellaSwag (Zellers 2019)

**任务**：句子续写最合理选项。

```
"A man is shoveling snow off his driveway. He..."
A. puts the shovel into space
B. continues to clear the path  ✓
C. drinks the snow
D. flies away
```

**数据**：~70k 训练 + 10k 测试 + 4 选项
**陷阱**：错选项也"看起来合理"（adversarial filtering）
**饱和度**：GPT-4 95%+，几乎不用了

## 2. ARC (AI2 Reasoning Challenge, Clark 2018)

**任务**：3-9 年级科学考题
**分支**：
- ARC-Easy（IR baseline 也能做）
- **ARC-Challenge**（IR fail，需推理）

```
Q: Which is a renewable energy source?
A. Coal  B. Solar  C. Natural gas  D. Petroleum
```

**饱和度**：GPT-4 ARC-Challenge 96%

## 3. Winogrande (Sakaguchi 2019)

**任务**：代词消解（Winograd schema 升级版）

```
"The trophy didn't fit in the suitcase because IT was too big."
IT = trophy or suitcase?
```

```
"The trophy didn't fit in the suitcase because IT was too small."
IT = trophy or suitcase?
```

**陷阱**：只改一个词（big ↔ small），答案翻转。测的是真理解。
**饱和度**：GPT-4 87%（最接近未饱和的一个）

## 为什么"老黄历"还要学

1. **OpenLLM Leaderboard v1 用 4 年**：~2021-2024 排行靠它
2. **数据规模大**：训练 task vector / probing 经常用
3. **理解 saturation**：体会 bench 寿命

## 饱和不等于无用

```
"分数被刷高"≠"任务已解决"
```

例如 Winogrande 在反向数据集（性别翻转）上掉到 60%。
→ 仍可作为 **robustness** scenario。

## 实操

src/commonsense_runner.py 每个 2 题（共 6 题）展示格式。

## 一句话

> 经典三件套是"小学考试"，模型小学毕业了，但小学教学法依然有效。
