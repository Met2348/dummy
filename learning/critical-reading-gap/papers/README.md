# papers/ — 案例论文与方法参考

本专题练「批判式读论文 + 找问题」, 用到两类材料:

## 一、方法参考 (怎么做研究本身)

| 文献 | 作用 | 获取 |
|---|---|---|
| S. Keshav, *How to Read a Paper* (2007) | L1 三遍读法的原始出处, 4 页, 必读 | 见下方下载 / 搜标题, Stanford/UWaterloo 有公开 PDF |
| W. G. Griswold / S. Keshav, *How to Read a Research Paper* 系列讲义 | 三遍读法的扩展 | 公开网页 |
| (可选) J. Widom, *Tips for Writing Technical Papers* | 衔接 9.7 写作 | Stanford 公开 |

> 若本机能联网: 运行 `python papers/fetch_refs.py` 会尝试下载 Keshav 的 PDF 到本目录;
> 失败 (无网/被墙) 不影响专题 —— 三遍读法的全部要点已写进 `lectures/L1`。

## 二、案例原论文 (用来练手的真论文)

刻意选你**已经复现过**的, 这样你能把"读论文"和"我亲手做过、最懂其弱点"对接起来:

| 论文 | 对应你的专题 | 在本专题哪里用 |
|---|---|---|
| DPO (Rafailov et al., 2023) | `learning/dpo-family` | N1 解剖 + 引用邻域 |
| DeepSeek-R1 / R1-Zero (2025) | `learning/reasoning-r1` | N2 找 gap 的主材料 |

这些原论文请用你已有的 `learning/<topic>/papers/` 里下载好的版本 (避免重复下载),
或从 arXiv 按标题获取。本专题**不重复囤 PDF**, 只引用你已有的。

## 为什么 papers/ 这么"轻"

技术专题的 papers/ 要囤很多原论文, 因为知识在论文里。
研究技能专题不同: **技能在你的动作里, 不在 PDF 里。** 这里只放"怎么做研究"的少量方法参考 +
指向你已有复现的指针。真正的"练习材料"是你自己的 48 个专题。
