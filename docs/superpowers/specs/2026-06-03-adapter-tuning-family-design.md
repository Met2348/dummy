# Adapter Tuning 家族学习专题 — 设计文档

> **承接**: prompt-tuning-family (输入端 PEFT) + lora-family (权重端 PEFT)
> **本专题**: 结构端 PEFT — Adapter 系列
> **战略地位**: PEFT 三大主线收官之作

---

## 1. 专题定位

Adapter Tuning 是 **最早的 PEFT 方法**（Houlsby et al., 2019），通过在 Transformer 每个 block 中插入小型可训练模块 (Adapter) 实现高效微调。本专题：
- **完整覆盖** 11 种 Adapter 家族方法（含 K-Adapter / MAD-X / AdaMix 应用方法）
- **桥接前两专题**：L9 三线综合 + L10 PEFT 下一步（含 Adapter 多模态复活）
- **三轨代码**：手写 minimal + AdapterHub `adapters` 库 + peft（仅 (IA)³）

### 1.1 为什么 Adapter 方法 2022 后"断更"
1. **LoRA 范式吞噬**：LoRA = 去掉非线性的 Adapter，但可合并 → 推理零延迟
2. **结构性 limitation**：串联非线性必然增加推理时延，LLM 时代致命
3. **(IA)³ 已是参数效率天花板**：再压只剩 BitFit
4. **行业重心迁移**：2023+ 主线变为量化 / 长上下文 / 对齐
5. **工业默认 LoRA**：peft 主推 LoRA 系，AdapterHub 维护放缓

### 1.2 Adapter 思想在多模态时代的"变形复活"（L10 内容）
- LLaMA-Adapter v1/v2 (2023): 指令微调 + 多模态
- Q-Former (BLIP-2, 2023): 查询式跨模态 adapter
- LLaVA projector (2023): 单层 MLP "adapter"
- AdapterSoup (2023): 多 adapter 权重平均

---

## 2. 方法清单（11 种）

| # | 方法 | 年份 | 论文 | 核心 idea |
|---|------|------|------|-----------|
| 1 | **Houlsby Adapter** | 2019 | Houlsby et al., ICML | 串联双 adapter（attn 后 + FFN 后）|
| 2 | **Pfeiffer Adapter** | 2020 | Pfeiffer et al., EMNLP | 简化版（仅 FFN 后）|
| 3 | **AdapterFusion** | 2021 | Pfeiffer et al., EACL | 多任务 attention 融合 |
| 4 | **AdapterDrop** | 2020 | Rücklé et al., EMNLP | 动态丢浅层 adapter，加速推理 |
| 5 | **Compacter** | 2021 | Karimi Mahabadi, NeurIPS | PHM 超复数乘 + 跨层共享 |
| 6 | **Parallel Adapter** | 2021 | He et al., ICLR | 并联结构 (vs 串联) |
| 7 | **(IA)³** | 2022 | Liu et al., NeurIPS | 3 个对角缩放向量 |
| 8 | **MAM Adapter** | 2022 | He et al., ICLR | Prefix + LoRA + Parallel 统一 |
| 9 | **K-Adapter** | 2020 | Wang et al., ACL | 注入外部知识 |
| 10 | **MAD-X** | 2020 | Pfeiffer et al., EMNLP | 跨语言（lang + task adapter）|
| 11 | **AdaMix** | 2022 | Wang et al., EMNLP | Mixture of Adapters + 随机路由 |

---

## 3. Lecture 结构（10 篇 = 8 主线 + 2 capstone）

| Lecture | 主方法 | 附录 | 核心 idea | 时长 |
|---------|--------|------|----------|------|
| **L1** Houlsby+Pfeiffer | Houlsby | Pfeiffer | 串联 Adapter 基础 | 40 min |
| **L2** AdapterFusion | AdapterFusion | — | 多任务组合 | 40 min |
| **L3** AdapterDrop+Compacter | AdapterDrop, Compacter | — | 推理加速 + PHM 压缩 | 60 min (含 PHM 数学) |
| **L4** Parallel Adapter | Parallel | — | 串/并联对比 | 40 min |
| **L5** (IA)³ | (IA)³ | — | 极致压缩 | 40 min |
| **L6** MAM Adapter | MAM | — | 统一视角（本系列理论高点）| 60 min |
| **L7** K-Adapter+MAD-X | K-Adapter, MAD-X | — | 知识 + 跨语言 | 50 min |
| **L8** AdaMix | AdaMix | — | MoE 路由 | 40 min |
| **L9** 三线综合 | — | UniPELT | Prompt+LoRA+Adapter 统一公式 + 28 方法决策树 | 60 min |
| **L10** PEFT 下一步 | — | LLaMA-Adapter, Q-Former, AdapterSoup | Adapter 多模态复活 + 后续专题路线图 | 50 min |

**总学时**: 8 hours (lecture) + 5 hours (notebook 实验) ≈ 13 hours

---

## 4. Lecture 模板（PPT-style，每篇 22-28 slides）

```markdown
# Lecture N: {方法名}

> 论文：xxx
> 配套代码：xxx
> 配套 notebook：xxx

## Slide 1: 上节回顾 + 本节路线
## Slide 2: 动机 (痛点)
## Slide 3: 核心公式 (1-2 个)
## Slide 4-6: 结构图解 (ASCII)
## Slide 7-10: 数学推导 / 反向梯度
## Slide 11-13: 代码逐行解读 (minimal + 库对照)
## Slide 14-16: 实验结果 / 论文 claim 复现
## Slide 17-19: 与前序方法对比 (含 LoRA/Prompt 跨专题)
## Slide 20-22: 思考题 (公式题/设计题/对比题)
## Slide 23-25: 工程选型 + 局限 + 下节预告
```

---

## 5. 代码三轨策略

| 方法 | minimal | adapters 库 | peft | 备注 |
|------|---------|-------------|------|------|
| Houlsby, Pfeiffer | ✅ 手写 | ✅ | — | adapter_config="houlsby"/"pfeiffer" |
| AdapterFusion | ✅ 手写 | ✅ | — | Fusion 阶段独立训 |
| AdapterDrop | ✅ 手写 | ✅ | — | adapters 1.3 仍支持 |
| Compacter | ✅ 手写 | ✅ | — | PHM 数学详写 |
| Parallel Adapter | ✅ 手写 | ✅ | — | adapters 称为 "scaled_parallel" |
| (IA)³ | ✅ 手写 | ✅ | ✅ | **三轨对照** (peft.IA3Config) |
| MAM Adapter | ✅ 手写 | — | — | 无库支持，自实现 |
| K-Adapter | ✅ 手写 | ✅ | — | toy knowledge triples |
| MAD-X | ✅ 手写 | ✅ | — | toy 3 语言子集 |
| AdaMix | ✅ 手写 | — | — | 无库支持（peft 不支持） |

**目录约定**:
- `{method}_minimal.py` — 手写最小实现
- `{method}_adapters.py` — adapters 库对照
- `{method}_peft.py` — peft 库对照（仅 ia3）

---

## 6. 一致性测试

```python
# 模板：每个方法至少 3 个测试
def test_param_count():       # 参数量正确
def test_initial_forward():   # 初始 forward = base (因为 zero init)
def test_mini_training():     # 几步 SGD loss 下降
```

**强一致**（minimal vs adapters 库的 forward 等价）:
- Pfeiffer: 强一致（结构最简单）
- Compacter: 弱一致（PHM 实现细节差异）
- (IA)³: 强一致 vs peft

**弱一致**:
- AdapterFusion: 实现差异大
- MAM, AdaMix: 仅 minimal 自洽

---

## 7. Notebook 结构（10 个）

每个 lecture 一个 ipynb：
1. import & 模型加载
2. 核心结构展示（参数布局）
3. minimal vs 库对照（强/弱一致性验证）
4. mini training（GPT-2 + toy 数据）
5. 论文 claim 复现（如 Compacter 的 PHM 压缩比）
6. 横向对比图表（matplotlib）
7. 思考题 + 下节预告

---

## 8. 环境配置

```
# requirements.txt
torch>=2.5
transformers>=4.55,<5.0   # adapters 1.3 要求 transformers 4.x
adapters>=1.3
peft>=0.13
bitsandbytes>=0.43
accelerate
datasets
scipy
matplotlib
numpy
```

**重要**：adapters 库强制依赖 transformers 4.x，会从 5.9 降级到 4.57。**这只影响本专题的 venv，不影响其它专题代码**（LoRA/Prompt 专题代码不依赖 5.x 新特性）。

**verify_env.py**:
- Part A: 基础（torch, transformers, peft, adapters）版本对齐
- Part B: GPU + sm_120 (Blackwell)
- Part C: adapters 库 GPT-2 smoke test

---

## 9. Git 里程碑

| Tag | 内容 | 预计 commits |
|-----|------|------|
| `adapter-base` | L1: Houlsby + Pfeiffer | 4 |
| `adapter-multitask` | L2-L3: Fusion + Drop + Compacter | 5 |
| `adapter-structure` | L4-L5: Parallel + (IA)³ | 4 |
| `adapter-unified` | L6: MAM Adapter | 2 |
| `adapter-app` | L7-L8: K-Adapter + MAD-X + AdaMix | 4 |
| `adapter-family-complete` | L9 + L10 + README | 3 |

---

## 10. 跨专题衔接（README + L9 内容）

### 28 方法横向表（README 新增）

| 主线 | 方法数 | 切入点 | 典型方法 | 推理时延 |
|------|--------|-------|---------|---------|
| Prompt-based | 5 | 输入端（soft token）| Prompt, Prefix, P-Tuning v1/v2, P-Tuning | 有（占 context）|
| Weight-based | 12 | 权重端（低秩/量化）| LoRA, QLoRA, DoRA... | 0（可合并）|
| **Adapter-based** | **11** | 结构端（加层）| Houlsby, (IA)³, MAM... | 有（无法合并）|

### L9 三线统一公式（基于 MAM Adapter 论文）

He et al. 证明：
- **Prefix Tuning** ≈ Parallel Adapter（可参数化转换）
- **LoRA** ≈ Adapter（无非线性、低秩约束、可合并）
- 三者本质是 transformer 注入向量的不同形式

---

## 11. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| adapters 库版本冲突 | 已发生 | 中 | 接受 transformers 降级到 4.57，与 LoRA 专题环境隔离 |
| PHM 数学（Compacter）难度高 | 高 | 中 | L3 配详细 ASCII 推导 + 玩具例子（n=2 时手算）|
| K-Adapter / MAD-X 数据需求 | 中 | 低 | toy 数据：10 条 factual triples + 3 语言玩具 |
| AdaMix MoE 路由不稳定 | 低 | 低 | mini training 用 5 个 expert，固定 seed |
| MAM Adapter 实现复杂 | 高 | 中 | 拆成 Prefix-like + LoRA-like + Parallel 三个独立 module |

---

## 12. 论文 PDF 占位策略

papers/ 下 11 篇论文先创建占位 README，后续手动补 PDF（与 LoRA/Prompt 专题一致做法）：

```
papers/
├── 01-houlsby-2019.md          # 论文元信息占位
├── 02-pfeiffer-2020.md
├── 03-adapterfusion-2021.md
├── 04-adapterdrop-2020.md
├── 05-compacter-2021.md
├── 06-parallel-2021.md         # He et al., "Towards a Unified View"
├── 07-ia3-2022.md
├── 08-mam-2022.md              # 同上论文
├── 09-k-adapter-2020.md
├── 10-mad-x-2020.md
├── 11-adamix-2022.md
└── README.md                   # 总 index
```

---

## 13. 执行授权

用户已明确授权 "一口气从计划到全流程实施，中间无需反复确认"，spec → plan → 11 phase 实施一气呵成。
