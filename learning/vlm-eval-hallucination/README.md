# 10.6 vlm-eval-hallucination — VLM 评测与视觉幻觉

> **Module 10「多模态/VLM 基础」· 第 6 专题 (会造也要会测)**
> M10.1-10.5 造了 VLM (读/画/听/说)。本专题测它: **它说得对吗? 真看图吗? 会幻觉吗?** 接你的评测安全 (M6) + Module 9 诚实工具链。

---

## 这个专题要解决的真问题

VLM 评测继承 M6 那套, 但多了独特难题:
- **真在看图吗?** 答对 ≠ 看图 (接模态坍缩诊断)。
- **视觉幻觉**: 图里没有的东西却说有 (VLM 头号障碍)。
- **多维能力**: 感知/推理/OCR/定位/诚实, 不能单一打分。

> 核心工具 **POPE**: 问「图里有 X 吗」(正负各半 + 诱饵), 看 **yes-bias** 量化幻觉。会造也要会测 —— 评测是 VLM 工程的良心。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-why-vlm-eval-special.md` | VLM 评测的独特难题 (看图吗/幻觉/多维) |
| L2 | `lectures/L2-vlm-benchmarks.md` | MMMU/MME/POPE 各测什么 |
| L3 | `lectures/L3-visual-hallucination.md` | 幻觉机理 (语言先验过强) + 探测 + 缓解 (DPO 抗幻觉) |
| L4 | `lectures/L4-eval-pitfalls-honesty.md` | 评测陷阱 + 诚实报告 (9.3+9.4+9.6 叠加) |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-pope-hallucination.ipynb` | 用 `src/vlm_eval.py` 对不同幻觉率的模拟 VLM 跑 POPE, 看 yes-rate 偏离 0.5 量化幻觉, 加空白图诊断 |
| `notebooks/N2-vlm-eval-figure.ipynb` | 用 9.6 `plotstyle` 把 POPE 结果画成出版级图 (幻觉率 vs 准确率/yes-rate, 带误差棒) |

## 工具 (`src/`)
- `vlm_eval.py` — POPE 式幻觉探测 + 指标 (准确率/yes率/F1) + 可调幻觉率模拟 VLM (纯 numpy)

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```
Python 3.13 / numpy / matplotlib。模拟 VLM 离线确定性 (真用时换真 VLM 的 yes/no 输出)。

## 完成本专题后你应该能
- [ ] 说清 VLM 评测比文本评测多了哪些难题
- [ ] 用 POPE + yes-bias 量化视觉幻觉
- [ ] 解释幻觉机理 (语言先验过强) 和缓解 (含 DPO 抗幻觉)
- [ ] 列 VLM 评测陷阱并用 9.3/9.4/9.6 逐条避
- [ ] 诚实报告 VLM 结果 (多维剖面 + 抗幻觉 + 方差)

---
## 在 Module 10 中的位置
```
  10.1-10.5 (造 VLM) → 10.6 评测 ◄你在这 → 10.7 capstone (M10 收官)
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module10-multimodal-vlm-design.md`
