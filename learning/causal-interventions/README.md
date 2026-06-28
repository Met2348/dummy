# 12.3 causal-interventions — activation patching (mech interp 核心)

> **Module 12「机制可解释性」· 第 3 专题 (因果工具, 最核心)**
> 探针只证相关 (M12.2)。要证**因果**, 必须**干预**: activation patching = 把 clean 激活贴进 corrupt 运行看恢复。这是 mech interp 区别于「看图讲故事」的命门。

---

## 这个专题要解决的真问题
- **为什么必须干预?** → 读取只证相关 (旁路/冗余/混淆); 因果黄金标准=干预 (改A看B)。
- **activation patching?** → clean 激活贴进 corrupt, 看行为恢复=该处因果携带信息 (充分)。
- **ablation?** → 删激活看损害=必要组件; 配 patching = 充要因果。
- **怎么严谨?** → 单变量/对照/度量/分布内 (mean非zero)/充要双验证 (接 M9.4)。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-correlation-vs-causation.md` | 相关三失败模式; 因果靠干预; 网络是因果实验天堂 |
| L2 | `lectures/L2-activation-patching.md` | patching: clean→corrupt 恢复 = 因果定位 (核心) |
| L3 | `lectures/L3-ablation-causal-paths.md` | ablation 找必要; 因果路径; 充要双验证 |
| L4 | `lectures/L4-intervention-rigor.md` | 干预严谨性 (=消融极致, 接 M9.4); 分布外陷阱 |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-activation-patching.ipynb` | patching 热图: 因果定位「最后位置携带答案」(恢复率1.0) |
| `notebooks/N2-ablation-scan.ipynb` | ablation 找必要组件; patching+ablation=充要因果 |

## 工具 (`src/`)
- `patching.py` — activation patching + ablation (forward hook 实现); 复用 12.1 tiny_transformer。**M12 最核心工具。**

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```

## 完成本专题后你应该能
- [ ] 解释相关≠因果, 及读取工具的三个失败模式
- [ ] 做 activation patching (clean/corrupt 对照 + 恢复率热图) 因果定位
- [ ] 用 ablation 找必要组件, 配 patching 做充要论证
- [ ] 用 7 问 checklist 做严谨干预 (分布内/单变量/充要)

---
## 在 Module 12 中的位置
```
  12.1 → 12.2 → 12.3 patching ◄你在这 → 12.4 SAE → 12.5 circuits → 12.6 CoT → 12.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module12-interpretability-design.md`
> `patching.py` 是 mech interp 最核心工具; 12.5 circuits 复用。
