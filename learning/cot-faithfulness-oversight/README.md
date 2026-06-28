# 12.6 cot-faithfulness-oversight — CoT 忠实性 + 对齐前沿

> **Module 12「机制可解释性」· 第 6 专题 (interp × 推理 × 安全)**
> CoT 是真的「内心独白」吗? 测**忠实性** (CoT 反映真实计算吗) + **scalable oversight** (模型>人怎么监督) + **欺骗检测**。这是你 (interp×reasoning) 最可能转 PhD 题的交汇点, 接你 reasoning-r1。

---

## 这个专题要解决的真问题
- **CoT 忠实吗?** → 不一定 (可能事后合理化); 测**偏置敏感性** (答案被未陈述因素带偏)。
- **CoT 监控?** → 愿景诱人但脆弱 (不忠实 + 可学会规避 + 优化压力反作用)。
- **模型>人怎么监督?** → scalable oversight; **weak-to-strong** (弱监督引出强能力)。
- **欺骗/装弱?** → 行为监督脆弱; **interp 看内部** 是关键希望 (M12.2-12.5)。

> **小而真**: N1 真 TinyLlama 偏置敏感性 (答案被无关提示大幅带偏); N2 weak-to-strong 玩具。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-reasoning-model-interp.md` | CoT 是真内心独白吗 (忠实 vs 事后合理化) |
| L2 | `lectures/L2-cot-faithfulness-monitoring.md` | 忠实性测法 (干预/偏置) + CoT 监控脆弱 |
| L3 | `lectures/L3-scalable-oversight.md` | 模型>人怎么监督; weak-to-strong |
| L4 | `lectures/L4-deception-sandbagging.md` | 欺骗/装弱检测; interp 是关键希望 |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-cot-faithfulness.ipynb` | 真 TinyLlama 偏置敏感性 (无关提示带偏答案=忠实性缺口) |
| `notebooks/N2-weak-to-strong.ipynb` | weak-to-strong 玩具 (弱监督引出强学生超过它) |

## 工具 (`src/`)
- `cot_probe.py` — CoT 偏置敏感性 (真 TinyLlama) + weak-to-strong 玩具 (numpy/torch)。

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```

## 完成本专题后你应该能
- [ ] 解释 CoT 忠实性 (反映真实计算吗) 及为什么关乎安全
- [ ] 用偏置敏感性测 CoT 不忠实 (答案被未陈述因素带偏)
- [ ] 说清 CoT 监控的前景与脆弱 (优化压力反作用)
- [ ] 解释 scalable oversight + weak-to-strong, 及 interp 在欺骗检测的角色

---
## 在 Module 12 中的位置
```
  12.1→...→12.5 → 12.6 CoT忠实性 ◄你在这 → 12.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module12-interpretability-design.md`
> 接你 reasoning-r1 (推理) + safety/red-team (对齐安全)。
