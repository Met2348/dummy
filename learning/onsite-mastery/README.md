# Onsite Mastery —— 终面深水区（知识深度 + 项目深挖 + 代码默写）

> 定位：`baguwen-mastery` 练"知道"，`interview-prep/mlcoding` 练"真 torch 手写"，本 track 补终面这层——**被连续追问三四层还不垮 + 自己的项目被当众拆解 + 闭卷手撕代码**。这不是知识差，是"扛住逼问"和"讲得出自己做的东西"之间的表现差。

## 先读这篇

[`lectures/00-how-to-defend.md`](lectures/00-how-to-defend.md) —— 为什么初面题库不够用、追问链/项目深挖/默写练习三套练法、与另外两个 track 的边界。

## 三块内容

| 模块 | 内容 | 练的是什么 |
|---|---|---|
| `src/ai_deep/` + `src/backend_deep/` | 深水知识点（`DeepPoint` 追问链，每点 3-4 层，每层带参考答案+采分词） | 被连续追问不垮 |
| `lectures/project-defense/` | 5 篇项目深挖文档（PhD 研究方向 + 4 个真实 capstone） | 自己的项目被拆解还能扛住 |
| `src/dictation/` | 闭卷手撕代码默写（17 个目标，spec/solution/check 三件套） | 白板/pair coding 闭卷手速 |

## 深水知识点总览

**合计 208 个 DeepPoint / 17 类**（2026 前沿补强前的初版），全部通过 `_self_test()`（链长 >= 3 层、id 全局唯一、trigger 不重复，见 `src/deep_common.py` 的 `grade_chain()`）。

| # | AI 深水类别（`src/ai_deep/`） | 点数 | # | 后端深水类别（`src/backend_deep/`） | 点数 |
|---|---|:--:|---|---|:--:|
| 01 | Transformer与注意力深水 | 12 | 01 | 操作系统深水 | 12 |
| 02 | RLHF与对齐深水 | 12 | 02 | 计算机网络深水 | 12 |
| 03 | MoE架构深水 | 12 | 03 | 数据库原理深水 | 12 |
| 04 | 分布式训练深水 | 12 | 04 | 分布式系统深水 | 14 |
| 05 | 推理部署与服务化深水 | 12 | 05 | 缓存与高并发深水 | 14 |
| 06 | 可解释性深水 | 12 | | **后端小计** | **64** |
| 07 | RAG与Agent深水 | 12 | | | |
| 08 | 预训练与数据深水 | 12 | | | |
| 09 | 评测与安全深水 | 12 | | | |
| 10 | 经典ML与系统设计深水 | 12 | | | |
| 11 | Scaling Law与训练动力学深水 | 12 | | | |
| 12 | Agent系统与Harness工程深水 | 12 | | | |
| | **AI 小计** | **144** | | | |

用法：`from ai_deep import ALL_DP as AI_DP; from backend_deep import ALL_DP as BE_DP`，配合 `deep_common.drill(bank, cat=None, n=None)` 抽题、`deep_common.grade_chain(dp, your_answers)` 逐层自评。

优先级沿用 `baguwen-mastery` 的既有说明：`ai_deep/` 是你的主战场（研究岗终面几乎全靠它+项目深挖），`backend_deep/` 仅在你也准备国内大厂后端/算法岗终面时才需要投入同等精力。

## 2026 前沿补强（`src/frontier_deep/`）

老手拿初版 208 点二次审阅后的反馈：**深度够了，但不够"新"**——不足以应付 2026 年 top 级别公司的深度面试。这一批不是重复内容，而是三个方向的差异化补强：

1. **内容新鲜度**：每一类都锚定 2025-2026 年真实的论文/技术报告/面经细节（如 circuit tracing/cross-layer transcoder、reward hacking 2026 形式化论文、Shumailov 2024 Nature 的 model collapse 数学结果、DeepSeek 的 MLA-RoPE 适配、Mooncake 的 KVCache 分离式架构），而不是重复初版已覆盖的经典知识。
2. **一个全新的内容维度**：`dp_multimodal_vla.py`——初版 12 个 AI 类目完全没有多模态/VLA 相关内容，这次补上 VLM vs VLA 建模目标差异、MoT 架构、深度/几何信息注入等前沿细节。
3. **一个全新的数据结构**：`ScenarioPoint`（`sc_engineering_judgment.py`）——"线上效果变差怎么定位""如何证明新 Prompt 更好"这类问题**没有唯一正确答案**，考的是工程判断力而非知识记忆，不适合用 `DeepPoint` 的"每层带参考答案"格式，`grade_scenario()` 按 rubric 要点覆盖率自查而非对错匹配。

**合计新增 207 点（185 个 DeepPoint / 9 类 + 22 个 ScenarioPoint / 1 类）**，与初版 208 点全局 id/trigger 均无冲突（`ai_deep`+`backend_deep`+`frontier_deep` 三者合计 415 点，程序化验证过）。

| # | 类别（`src/frontier_deep/`） | 点数 | 类型 |
|---|---|:--:|:--:|
| 01 | 推理模型与Test-Time-Compute深水 | 21 | DeepPoint |
| 02 | Agent生产工程2026深水 | 20 | DeepPoint |
| 03 | 对齐与可扩展监督2026深水 | 21 | DeepPoint |
| 04 | 可解释性2026前沿深水 | 19 | DeepPoint |
| 05 | 多模态与VLA深水（全新类目） | 22 | DeepPoint |
| 06 | 开源前沿模型技术细节深水（DeepSeek/MiniMax） | 22 | DeepPoint |
| 07 | LLM推理系统2026深水 | 20 | DeepPoint |
| 08 | 数据与Scaling2026深水 | 19 | DeepPoint |
| 09 | RAG与工具调用工程2026深水 | 21 | DeepPoint |
| 10 | 工程判断力场景题(无标准答案) | 22 | **ScenarioPoint** |

用法：`from frontier_deep import ALL_DP as FR_DP, ALL_SP as FR_SP`，`ALL_DP` 复用 `deep_common.drill()`/`grade_chain()`；`ALL_SP` 配合新增的 `deep_common.grade_scenario(sp, your_answer)` 做要点覆盖率自查（不是对错判断）。

## 项目深挖文档

| # | 文档 | 素材来源 |
|---|---|---|
| 00 | [PhD方向答辩稿](lectures/project-defense/00-phd-direction.md) | `harness-phd-direction-deck.md` |
| 01 | [interp capstone 深挖](lectures/project-defense/01-interp-capstone.md) | `learning/interp-graduation/src/interp_capstone.py` |
| 02 | [R1-Zero capstone 深挖](lectures/project-defense/02-reasoning-r1-capstone.md) | `learning/reasoning-r1/` |
| 03 | [DPO capstone 深挖](lectures/project-defense/03-dpo-alignment-capstone.md) | `learning/dpo-family/src/capstone_dpo_comparison.py` |
| 04 | [harness-engineering capstone 深挖](lectures/project-defense/04-harness-engineering-capstone.md) | `learning/harness-engineering/` |

## 默写目标（17 个）

`python learning/onsite-mastery/src/dictation/harness.py <name>` 逐个练，见 [`00-how-to-defend.md`](lectures/00-how-to-defend.md) §4。

- mlcoding 系（8）：attention / rope / rmsnorm / lora / kv_cache / sampling / training_step / transformer_block
- 对齐/RL 系（4）：ppo_clip / gae / dpo_loss / grpo_advantage
- MoE（1）：moe_router
- 系统类（4）：consistent_hashing / lru_cache / rate_limiter / bm25

## 环境与测试

```powershell
$env:PYTHONIOENCODING="utf-8"
pip install torch numpy
python learning/onsite-mastery/environment/verify_env.py
python learning/onsite-mastery/src/tests/test_all.py
```

## 诚实说明

`lectures/project-defense/` 里的文档是**脚手架**，不是台词稿——它们帮你把项目细节想全，但终面真正考的是你**自己**能不能讲清楚、扛住反问。背下这几篇 md 没用，对着草稿纸/口头练一遍、让别人（或者对着 `deep_common.drill()`）随机抽问才有用。

## 一句话

> 初面拼知道，终面拼扛得住。这个 track 不是又一批要背的知识点，是把你已经会的东西，磨出"被连续追问也不垮"的耐操度。
