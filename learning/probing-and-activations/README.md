# 12.2 probing-and-activations — 线性探针 + logit lens

> **Module 12「机制可解释性」· 第 2 专题 (读取工具)**
> mech interp 第一件工具: 从激活**读出**内部表示。线性探针 (概念是否被线性编码) + logit lens (每层在想什么)。但记住: 读取只证**相关**, 引出 12.3 干预。

---

## 这个专题要解决的真问题
- **residual stream?** → 信息主干 (黑板); 组件从它读写; 线性可分解 (L1)。
- **线性探针?** → 线性分类器从激活读概念; 高准确率=被**线性编码** (一个方向)。
- **logit lens?** → 中间 residual 过最终 unembed, 零训练看预测逐层成形。
- **陷阱?** → 探针读出 **≠** 模型在用 (相关非因果) → 必须走向干预 (M12.3)。

> **小而真**: 探针在真 gpt2 上读「是否数字」(1.00); logit lens 看 gpt2 的「Paris」逐层浮现。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-residual-stream.md` | residual stream=黑板; 组件读写; 线性可分解 |
| L2 | `lectures/L2-linear-probes.md` | 线性探针读概念 (必须线性); feature=方向 |
| L3 | `lectures/L3-logit-lens.md` | logit lens 看预测成形 (零训练) |
| L4 | `lectures/L4-probing-pitfalls.md` | 探针陷阱: 相关≠因果 → 引出干预 |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-linear-probes.ipynb` | 探针读概念: 玩具读「当前值」+ 真 gpt2 读「是否数字」 |
| `notebooks/N2-logit-lens.ipynb` | logit lens: 玩具+1在哪层成形 + 真 gpt2「Paris」逐层浮现 |

## 工具 (`src/`)
- `probing.py` — 通用线性探针 + logit lens; 复用 12.1 tiny_transformer, 也用于真实 gpt2 激活。

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```

## 完成本专题后你应该能
- [ ] 解释 residual stream 视角 (组件读写黑板, 线性可分解)
- [ ] 用线性探针测概念是否被线性编码 (及为什么探针必须线性)
- [ ] 用 logit lens 看预测在哪层成形
- [ ] 说清探针的根本陷阱 (相关≠因果) 及为什么需要干预

---
## 在 Module 12 中的位置
```
  12.1 → 12.2 probing ◄你在这 → 12.3 patching → 12.4 SAE → 12.5 circuits → 12.6 CoT → 12.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module12-interpretability-design.md`
