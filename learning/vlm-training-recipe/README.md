# 10.3 vlm-training-recipe — 训出一个能问答的 VLM

> **Module 10「多模态/VLM 基础」· 第 3 专题 (10.1+10.2 的收获)**
> 视觉塔 (10.1) + 连接器 (10.2) 组装好了, 但是随机权重。本专题用**两阶段配方**把它真正训成能回答图像问题的 VLM。

---

## 这个专题要解决的真问题

```
   随机 mini-VLM → ??? → 能回答图像问题的 VLM
```

VLM 训练不是一锅端, 而是**两阶段配方**:
- **阶段 1 对齐预训练**: 只训连接器 (视觉塔+LLM 冻), 教 LLM 读视觉, 用图文对。
- **阶段 2 指令微调**: 训连接器+LLM, 教按指令答, 用视觉指令数据。

> 关键细节决定成败: 数据 (图文对/交错/指令) + 冻结策略 (能冻就冻、逐步解冻) + 避坑 (模态坍缩/灾难遗忘/数据失衡)。你的 LLM 指令微调 (M1/M4) 知识直接迁移。

## 学习路径 (4 讲)

| 讲 | 文件 | 一句话 |
|---|---|---|
| L1 | `lectures/L1-two-stage-recipe.md` | 两阶段: 对齐预训练 + 指令微调, 为什么分两阶段 |
| L2 | `lectures/L2-data-and-freezing.md` | 三类数据 + 冻结策略 (该用 9.4 消融) |
| L3 | `lectures/L3-llava-recipe.md` | LLaVA 配方逐步拆 (博0 复现起点) |
| L4 | `lectures/L4-training-pitfalls.md` | 三个坑: 模态坍缩/灾难遗忘/数据配比 |

## 动手 (2 个 notebook)

| notebook | 你会真的做什么 |
|---|---|
| `notebooks/N1-train-mini-vlm.ipynb` | 用 `src/mini_vlm.py` 端到端训一个 mini-VLM (视觉塔+连接器+tiny LLM), 看 loss↓ acc 0.25→~1.0; 再做**模态坍缩诊断** (空白图准确率掉不掉) |
| `notebooks/N2-freezing-ablation.ipynb` | 用 `set_freeze` 对比「冻 LLM 只训连接器」vs「全解冻」, 看可训练参数量 vs 准确率 (接 9.4 消融) |

## 工具 (`src/`)
- `mini_vlm.py` — 可训练 mini-VLM (组装 10.1 视觉塔 + 10.2 连接器 + tiny LLM) + 合成 VQA 任务 + 冻结开关 + 训练循环

## 环境
```bash
pip install -r environment/requirements.txt
python environment/verify_env.py     # 应输出: 全部通过 ✅
```
Python 3.13 / torch (tiny CPU 秒级训练) / numpy / matplotlib。合成 VQA 数据离线确定性。

## 完成本专题后你应该能
- [ ] 说清两阶段配方 + 为什么 (课程式/保护预训练/数据效率)
- [ ] 给 VLM 定冻结策略并解释 (连接器永训/视觉塔默认冻/LLM 阶段2解冻)
- [ ] 照 LLaVA 配方写出复现 checklist
- [ ] 识别三个训练坑并给诊断/避法 (尤其「空白图诊断」查模态坍缩)
- [ ] 端到端训一个能从图判类别的 mini-VLM

---
## 在 Module 10 中的位置
```
  10.1 视觉塔 ✅ → 10.2 融合 ✅ → 10.3 训练 ◄你在这 → 10.4 生成 → 10.5-10.7
```
> 设计文档: `docs/superpowers/specs/2026-06-24-module10-multimodal-vlm-design.md`
