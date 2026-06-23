# 10.7 multimodal-graduation — Module 10 收官 (Capstone)

> **Module 10「多模态/VLM 基础」· 第 7 专题 (收官 Capstone)**
> 把 10.1-10.6 装配成一条完整 mini-VLM 流水线, 并用 9.3 gap 雷达把 M10 变成你的 **PhD 研究入口**。

---

## 这个专题做什么

```
   装配: 10.1 视觉塔 → 10.2 连接器 → 10.3 训练 → 10.4 生成 → 10.5 时序 → 10.6 评测
        = 一条端到端、机制完整的 mini-VLM 流水线 (CPU 跑通)
   找 gap: 用 9.3 的 6 类 gap 雷达 + 优先级公式扫多模态前沿
        = 收敛出对你最友好的 VLM 研究题目 (起 idea 卡)
```

> M10 不是「学了一堆技术」, 而是「会造 VLM (工程) + 能找 VLM 研究 gap (研究)」。这正是 harness-engineering 示范的「同一热点既当工程练、又当研究挖」模式, 在多模态的复刻。

## 学习路径 (2 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-assemble-the-pipeline.md` | 把 10.1-10.6 串成完整 mini-VLM 流水线; tiny↔真实一一对应 |
| L2 | `lectures/L2-vlm-research-gaps.md` | 用 9.3 gap 雷达扫多模态; 3 个最友好起手题; 迁移优势 |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-end-to-end-vlm.ipynb` | 用 `mm_capstone` 检查装配, 真跑端到端 mini-VLM (复用 10.1-10.4 src): 图→理解→画图→评测 |
| `notebooks/N2-multimodal-gap-card.ipynb` | 用 `mm_capstone` 扫 gap, 挑优先级最高的, 用 9.3/9.4 卡模板起一张完整多模态研究 idea 卡 |

## 工具 (`src/`)
- `mm_capstone.py` — M10 流水线装配检查 + 多模态研究 gap 雷达 (6 类 + 优先级, 接 9.3)

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```
Python 3.13 / torch / numpy / matplotlib。N1 跨专题复用 10.1-10.4 的 src。

## 完成本专题后你应该能
- [ ] 把 M10 全线装配成一条端到端 mini-VLM 流水线
- [ ] 说清 tiny 组件与真实 VLM 的一一对应 (只换规模/预训练)
- [ ] 用 9.3 gap 雷达扫出多模态研究题目并排优先级
- [ ] 用迁移优势 (dpo/long-context/Module 9) 把 gap 做成 idea 卡
- [ ] 产出一张能动手的 VLM 研究 idea 卡 (M10 真正产出)

---
## 在 Module 10 中的位置 (收官)
```
M10 多模态/VLM (7/7 完成):
  10.1 视觉塔 ✅ → 10.2 融合 ✅ → 10.3 训练 ✅ → 10.4 生成 ✅
  → 10.5 视频音频 ✅ → 10.6 评测 ✅ → 10.7 收官 ◄你在这
```
> M10 给 48 工程专题装上眼睛和嘴 = 第 9 大画像「会做多模态的人」。
> 设计文档: `docs/superpowers/specs/2026-06-24-module10-multimodal-vlm-design.md`
