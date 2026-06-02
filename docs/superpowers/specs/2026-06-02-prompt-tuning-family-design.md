# 设计文档：`prompt-tuning-family` 专题学习包

- **日期**：2026-06-02
- **作者**：用户 + Claude（协作设计）
- **背景书目**：《大模型算法：强化学习、微调与对齐》ISBN 9787121500725，第 2.1.4 节"基于 Prompt 的微调"
- **状态**：待用户确认

## 1. 目的与范围

### 1.1 学习目标

帮助用户深入理解"基于 Prompt 的参数高效微调"这一系列的四篇经典论文：

1. **Prefix-Tuning**（Li & Liang, 2021, arXiv:2101.00190）
2. **Prompt Tuning**（Lester, Al-Rfou, Constant, 2021, arXiv:2104.08691）
3. **P-Tuning v1**（Liu, Zheng, Du, et al., 2021, arXiv:2103.10385）
4. **P-Tuning v2**（Liu, Ji, Fu, et al., 2022, arXiv:2110.07602）

学习产出对标"研究生级课程课件"标准：

- 能讲清楚论文动机、方法、实验、局限
- 能在 PyTorch 中复现核心算法的最小可运行版本
- 能向他人讲解每一行公式中每个符号的含义

### 1.2 范围

**包含**：
- 4 篇原版 PDF 论文
- 4 份中文 PPT 风格教学 Markdown（每份 25-35 张幻灯片，对应约 30-40 分钟讲解）
- 4 份 PyTorch 最小实现 `.py` 文件（不依赖 peft，从零实现）
- 4 份 peft 调包版 `.py` 文件（对照验证：参数量、输出 logits 是否与手写版一致）
- 4 份 Jupyter Notebook 交互式教学（手写版与 peft 版并列演示）
- 1 份专题 README，统一记号、给出阅读顺序、横向对比四种方法
- 1 份环境配置说明 + `requirements.txt`
- 仓库通过 `git init` 初始化，所有产出按里程碑提交

**不包含**：
- 大规模训练（仅做小模型示例，避免显存/时间约束）
- 与 LoRA / Adapter 等其他参数高效微调方法的深入对比（仅在每篇 lecture 末尾的"延伸阅读"中提到）
- SOTA 复现（不追求论文数字一致，只追求算法正确）

### 1.3 成功标准

- 文件结构按设计完整生成
- 4 篇 lecture 的内容深度满足"重读不卡壳"——用户即使忘记前文定义，也能在当前页找到所有符号的解释
- 4 个 notebook 在用户本机环境下能跑通至少前 5 个 cell（涉及模型推理的部分用 `gpt2` 级别的小模型）
- README 中的"四种方法对比表"能让用户秒回忆起每种方法的关键差异

## 2. 目录结构

```
learning/prompt-tuning-family/
├── README.md
├── papers/
│   ├── 01-prefix-tuning-2021.pdf
│   ├── 02-prompt-tuning-2021.pdf
│   ├── 03-p-tuning-2021.pdf
│   └── 04-p-tuning-v2-2022.pdf
├── lectures/
│   ├── 01-prefix-tuning.md
│   ├── 02-prompt-tuning.md
│   ├── 03-p-tuning.md
│   └── 04-p-tuning-v2.md
├── code/
│   ├── prefix_tuning_minimal.py
│   ├── prompt_tuning_minimal.py
│   ├── p_tuning_minimal.py
│   └── p_tuning_v2_minimal.py
├── notebooks/
│   ├── 01-prefix-tuning.ipynb
│   ├── 02-prompt-tuning.ipynb
│   ├── 03-p-tuning.ipynb
│   └── 04-p-tuning-v2.ipynb
└── environment/
    ├── requirements.txt
    └── README.md
```

## 3. lecture 文档规范

### 3.1 PPT 风格的实现

- 用 `---`（水平线）分隔幻灯片
- 每张幻灯片有清晰的标题（`## 第 X 张幻灯片：标题`）
- 每张幻灯片内容不超过一屏（约 200-400 字 + 1 个示意图或 1 个公式块）
- 全篇约 25-35 张幻灯片（按每张 1 分钟讲解，对应约 30-40 分钟课时）
- 不使用花哨表情、颜色，保持研究生课件的严肃质感

### 3.2 标准章节结构

每篇 lecture 严格遵循以下章节顺序：

| 章节 | 幻灯片数 | 内容要点 |
|------|----------|----------|
| 封面与导读 | 1 | 论文标题、作者、机构、发表会议、核心 claim 一句话 |
| **符号速查表** | 1 | 所有符号集中定义：含义、维度、首次出现的公式编号 |
| 1. 历史背景与研究动机 | 2-3 | 前作的局限是什么？这篇论文要解决什么问题？ |
| 2. 核心思想（直觉理解） | 2 | 用一句话总结方法 + 一张类比图 |
| 3. 数学符号约定（再次重申） | 1 | 把速查表中的符号在公式上下文中再说一遍 |
| 4. 方法详解 | 8-12 | 公式逐项重述、每个公式都画图、关键变量维度标注 |
| 5. 架构示意图 | 2-3 | Mermaid 流程图 + ASCII 框图，从输入到输出完整画一遍 |
| 6. 实验设计与结果 | 3-4 | 实验设置（任务、模型、指标）+ 关键表格的中文解读 |
| 7. 优缺点与适用边界 | 2 | 哪些场景好用？哪些场景失效？工程视角 |
| 8. 与同期方法横向对比 | 2 | 与四种方法中的其他三种对比（差异、共性） |
| 9. PyTorch 关键代码片段 | 1-2 | 算法核心 5-20 行代码，配中文注释 |
| 10. 思考题与延伸阅读 | 1 | 3-5 个思考题、关联书本章节、相关论文链接 |

### 3.3 数学公式呈现规范

**核心原则**：永远不让读者回滚查定义。

具体做法：

1. **符号速查表置顶**：每篇 lecture 第 2 张幻灯片为符号速查表，格式如下：

   | 符号 | 含义 | 维度 | 首次出现 |
   |------|------|------|----------|
   | $h_i$ | 第 $i$ 层 Transformer 的隐状态 | $\mathbb{R}^d$ | 公式 (1) |
   | $P_\theta$ | 可训练的前缀参数矩阵 | $\mathbb{R}^{p \times d}$ | 公式 (2) |

2. **公式当场重述**：每个公式紧跟一段"逐项解读"，例如：

   $$h_i = \mathrm{Transformer}(h_{i-1}; \theta_{\mathrm{LM}}) \quad (1)$$

   其中：
   - $h_i \in \mathbb{R}^d$：第 $i$ 层的隐状态向量，$d$ 是模型隐层维度（如 GPT-2 是 768）
   - $h_{i-1} \in \mathbb{R}^d$：上一层的隐状态
   - $\theta_{\mathrm{LM}}$：预训练语言模型的所有参数（在本方法中**冻结不动**）

3. **关键定义用 callout**：

   > **定义 1**：**Prefix（前缀）**是一段可训练的连续向量序列，长度为 $p$，每个向量维度为 $d$，与原输入 token embedding 拼接后送入 Transformer。

4. **跨页引用提醒**：当某符号在后续幻灯片再次出现，写"（$\theta$ 含义参见第 3 张幻灯片的符号速查表）"

### 3.4 示意图规范

- **流程图/数据流**：用 Mermaid `flowchart` 或 `sequenceDiagram`
- **网络结构/张量形状**：用 ASCII art（等宽对齐）
- 示例 ASCII：
  ```
  Input tokens:    [x1, x2, x3]            shape: (B, L)
                        │
                        ▼
  Embedding:       [e1, e2, e3]            shape: (B, L, d)
                        │
              ┌─── Prefix prepend ───┐
              ▼                       ▼
  Final input: [p1, p2, e1, e2, e3]   shape: (B, p+L, d)
  ```
- 每张图配一两句中文说明

### 3.5 PyTorch 代码片段规范

- 仅展示算法核心 5-20 行（如：前缀拼接、reparameterization MLP、提示位置 mask 构造）
- 完整可运行版本放在 `code/*.py` 中并被 lecture 引用
- 代码注释用中文，符号与公式中的记号对应

## 4. 代码（`code/`）规范

### 4.1 设计原则

- **可独立运行**：每个 `.py` 文件包含 `if __name__ == "__main__":`，能直接 `python prefix_tuning_minimal.py` 跑通
- **手写版依赖最小**：`*_minimal.py` 仅依赖 `torch`、`transformers`，不依赖 `peft`（让用户看到底层实现）
- **peft 对照版**：每个方法另写一份 `*_peft.py`，使用 `peft` 库实现同一方法
- **一致性验证**：每对 `(_minimal, _peft)` 文件必须证明：
  1. 可训练参数量在同一量级（允许差异，因为 peft 内部可能多 wrapper 层）
  2. 给定相同输入和相同初始化（手动复制权重），输出 logits 在数值上一致（差异 < 1e-5）
- **小模型示范**：使用 `gpt2`（约 117M 参数）或 `gpt2-medium`，避免显存压力
- **教学优先**：宁可慢、宁可不优化，只要算法清晰

### 4.1.1 文件配对

| 方法 | 手写版 | peft 对照版 | 对照所用 peft 类 |
|------|--------|-------------|------------------|
| Prefix Tuning | `prefix_tuning_minimal.py` | `prefix_tuning_peft.py` | `PrefixTuningConfig` |
| Prompt Tuning | `prompt_tuning_minimal.py` | `prompt_tuning_peft.py` | `PromptTuningConfig` |
| P-Tuning v1 | `p_tuning_minimal.py` | `p_tuning_peft.py` | `PromptEncoderConfig` |
| P-Tuning v2 | `p_tuning_v2_minimal.py` | `p_tuning_v2_peft.py` | `PrefixTuningConfig`（peft 中 v2 与 v1 prefix 同接口） |

### 4.2 通用结构

```python
"""
<方法名> 最小实现
对应 lecture: lectures/0X-<method>.md
对应论文:    papers/0X-<method>-YYYY.pdf
"""
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. 加载预训练模型（冻结所有参数）
# 2. 定义可训练的 prompt/prefix 模块
# 3. 定义前向传播（如何把可训练参数注入到 LM 中）
# 4. 一个 toy 训练 loop（用一个简单的玩具任务展示训练过程）
# 5. 推理示例

if __name__ == "__main__":
    main()
```

### 4.3 玩具任务

每个 `.py` 用同一个简单任务（如"句子情感二分类"或"给定 prompt 续写带特定情感的句子"），让用户能横向对比四种方法的代码差异。

## 5. Notebook 规范

每个 notebook 内容是对应 lecture 的"动手版"：

- Cell 1：环境检查（import + 版本打印）
- Cell 2-3：加载小模型（`gpt2`），打印参数量
- Cell 4-N：分步实现 `.py` 中的逻辑，每步配 Markdown 说明
- 最后 cell：与原始 fine-tuning 的参数量对比表

**注意**：涉及训练的 cell，把 `max_steps` 设为 10-50（仅演示流程，不追求收敛）。

## 6. README 规范（`learning/prompt-tuning-family/README.md`）

包含以下章节：

1. **专题简介**：一段话讲清楚"为什么这四篇要放在一起学"
2. **四篇论文关系图**（Mermaid 时间线 + 演进关系）
3. **统一记号表**：在整个专题中保持一致的符号（与每篇 lecture 的速查表对齐）
4. **推荐学习顺序**：
   - 路径 A（按发表时间）：Prefix → P-Tuning v1 → Prompt Tuning → P-Tuning v2
   - 路径 B（按概念由易到难，推荐）：Prompt Tuning → Prefix Tuning → P-Tuning v1 → P-Tuning v2
5. **四种方法横向对比表**：

   | 方法 | 可训练参数位置 | 可训练参数量级 | reparameterization | 适用模型规模 |
   |------|----------------|---------------|---------------------|--------------|
   | Prefix Tuning | 每一层 KV 前缀 | $L \times p \times 2d$ | MLP | 任意 |
   | Prompt Tuning | 仅输入层 embedding 前缀 | $p \times d$ | 无 | 仅大模型有效 |
   | P-Tuning v1 | 输入层 embedding 任意位置 | $p \times d$ | LSTM/MLP | NLU 任务 |
   | P-Tuning v2 | 每一层 KV 前缀 | $L \times p \times 2d$ | 可选 | 任意规模、任意任务 |

6. **如何使用本专题**：建议每篇 lecture 用 40 分钟阅读 + 30 分钟跑 notebook

## 7. 环境配置

### 7.1 检查步骤

按以下顺序检查并补齐：

1. Python 版本（要求 ≥ 3.10）
2. `pip` / `uv` 可用性
3. `torch`（CPU 或 CUDA 版均可，CUDA 优先）
4. `transformers`
5. `jupyter` + `ipykernel`
6. `matplotlib`（用于 notebook 中的可视化）
7. （可选）`peft` —— 仅用于在最后做对照（让用户看到"原来 peft 是这么封装的"）

### 7.2 `requirements.txt` 内容（拟定）

```
torch>=2.0
transformers>=4.40
jupyterlab>=4.0
ipykernel>=6.0
matplotlib>=3.7
peft>=0.10  # 可选
```

### 7.3 验证脚本

`environment/README.md` 中提供一个"环境自检"小脚本：导入所有依赖，打印 torch CUDA 可用性，加载 `gpt2` 模型并跑一次推理，确保整套环境可用。

## 8. 工作流（执行阶段）

下一步将由 `writing-plans` 技能转写为详细任务清单，但顶层流程是：

0. **Git 初始化**：在仓库根目录 `git init`，写 `.gitignore`（忽略 `__pycache__`、`.ipynb_checkpoints`、模型 cache 等），首次提交包含设计文档
1. **环境检查与补齐**（用户本机 Windows 11）
2. **论文下载**（从 arXiv 直链）
3. **写 README**（先有专题骨架）
4. **逐篇推进**（lecture → minimal.py → peft.py → 一致性验证脚本 → notebook → 端到端验证）：
   - Prompt Tuning（最简单，先做）
   - Prefix Tuning
   - P-Tuning v1
   - P-Tuning v2（综合前面三种思想）
5. **最终验证**：
   - 跑通所有 notebook 前 5 cell
   - 跑通所有 `(_minimal, _peft)` 对照的一致性验证脚本
   - 交叉检查 README 对比表与 lecture 内容一致
6. **里程碑提交**：每完成一篇论文的全部产出（lecture + 两版 py + notebook），形成一个 git commit

## 9. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 数学符号在四篇论文中不统一，混淆用户 | README 中给出"统一记号表"，每篇 lecture 在头部明确"本篇采用的记号" |
| 模型下载慢 / 网络问题 | 用 `gpt2`（117M）作为示范，国内可用 `HF_ENDPOINT=https://hf-mirror.com` |
| 用户本机 GPU 显存不够 | notebook 全程支持 CPU 跑，仅训练步骤减小 batch_size |
| 工作量过大、单次会话写不完 | 由 `writing-plans` 拆分为可独立验证的子任务，分多轮推进 |
| arXiv PDF 直链偶尔失败 | 写好 fallback URL（arxiv-vanity / OpenReview） |

## 10. 验收清单

完成后用户应能：

- [ ] 在 `learning/prompt-tuning-family/` 下看到完整的 5 个子目录
- [ ] 打开任意一篇 lecture，从头读到尾不需要回滚查符号
- [ ] 在 notebooks 中跑通至少前 5 个 cell
- [ ] 看 README 的对比表 30 秒回忆起四种方法差异
- [ ] 直接拿 `.py` 文件做后续扩展实验
- [ ] 看 `*_minimal.py` 和 `*_peft.py` 的对照，理解 peft 内部如何封装这些算法
- [ ] 仓库已 `git init` 并包含按里程碑组织的 commit 历史

---

**下一步**：用户确认本设计后，调用 `superpowers:writing-plans` 把它转写为可执行的实施计划。
