# 12.1 interp-foundations — 机制可解释性地基

> **Module 12「机制可解释性 / 对齐前沿」· 第 1 专题 (地基)**
> 你会造模型了, 但懂它内部吗? **机制可解释性 (mech interp)** = 把训练好的网络当程序, 逆向工程出内部算法 (feature + circuit)。这是 2026 前沿热点, 对你 (NLP + EE 数学) 特别友好。

---

## 这个专题要解决的真问题

- **mech interp 是什么?** → 逆向工程网络内部算法 (feature/circuit), 不是事后归因 (相关)。
- **核心概念?** → **feature** (概念=激活方向) / **circuit** (协作组件=算法) / **superposition** (多义神经元)。
- **头号难题?** → **superposition**: 网络叠加压缩 feature → 神经元多义 → 不能逐神经元读。
- **方法论?** → 逆向工程循环: 观察→假设→**干预 (因果)**→提炼电路; 从窄行为入手。
- **陷阱?** → 相关当因果 / 讲故事 / 玩具≠真实 (批判式读, 接 M9.3)。

> **小而真**: N1 直接在**真实 gpt2** 上看多义神经元 (superposition); N2 用受控玩具学方法。

## 学习路径 (4 讲)
| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-why-open-the-black-box.md` | mech interp vs 事后归因; 为什么打开黑箱 |
| L2 | `lectures/L2-features-circuits-superposition.md` | feature/circuit/superposition 三核心概念 |
| L3 | `lectures/L3-reverse-engineering-program.md` | 逆向工程纲领 (网络=程序); residual stream |
| L4 | `lectures/L4-methodology-and-limits.md` | 方法论陷阱与局限 (批判式读, 接 M9.3) |

## 动手 (2 个 notebook)
| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-polysemantic-neurons.ipynb` | **真实 gpt2**: 看多义神经元 (一个神经元对多个不相关概念激活) |
| `notebooks/N2-toy-model-dissection.ipynb` | 训受控玩具 transformer, 读中间激活 (解剖基座) |

## 工具 (`src/`)
- `tiny_transformer.py` — 可 hook 中间激活的最小 transformer (run_with_cache); 玩具任务 increment-mod-V。**M12 受控解剖基座, 12.2-12.5 复用。**
- 真实模型演示用 `learning/_shared/realmodels.py` (gpt2)。

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 全部通过 ✅
```
Python 3.13 / torch (tiny CPU) / transformers (真实 gpt2) / numpy / matplotlib。

## 完成本专题后你应该能
- [ ] 区分机制可解释性 vs 事后归因 (因果 vs 相关)
- [ ] 解释 feature / circuit / superposition, 及为什么 superposition 是核心难题
- [ ] 描述逆向工程循环 (观察→假设→干预→电路) 和 residual stream 视角
- [ ] 用 M9.3 的 5 问 checklist 批判一篇 interp 论文
- [ ] 在真 gpt2 上看到多义神经元, 在玩具上读取激活

---
## 在 Module 12 中的位置
```
  12.1 地基 ◄你在这 → 12.2 probing → 12.3 patching → 12.4 SAE → 12.5 circuits → 12.6 CoT忠实性 → 12.7 capstone
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module12-interpretability-design.md`
> `tiny_transformer.py` 是 M12 全模块的共享解剖基座。
