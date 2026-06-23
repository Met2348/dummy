<!-- VLM 评测诚实报告卡 (M10.6-L4) · 报 VLM 结果前过一遍. 9.3+9.4+9.6 叠加. -->
# VLM 评测报告: ____________

## 多维剖面 (不报单一总分, 9.6 一图一信息)
| 维度 | benchmark | 分数 ± std |
|---|---|---|
| 感知 | MME | |
| 推理 | MMMU | |
| OCR | OCRBench | |
| 定位 | RefCOCO | |
| **抗幻觉** | POPE (yes-rate/F1) | |

## 抗幻觉 (必报, L2/L3)
- POPE yes-rate: ____ (≈0.5 理想; >>0.5 = 幻觉)
- 空白图诊断: 给空白图准确率是否掉到随机? ____

## 公平对照 checklist (9.4-L3)
- [ ] 同分辨率 · [ ] 同 prompt 协议 · [ ] 同评测集
- [ ] baseline 已尽力调

## 方差 + prompt 鲁棒性 (9.4-L5)
- [ ] 多 prompt/多次评测报 error bar
- [ ] 换问法分数稳吗 (prompt 敏感性)

## 诚实声明 (9.3+9.7)
- [ ] benchmark 污染风险声明 (训练是否见过)
- [ ] 定性样例非 cherry-picked, 含失败案例
- [ ] 评测 prompt 已公开
