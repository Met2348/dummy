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

**合计 208 个 DeepPoint / 17 类**，全部通过 `_self_test()`（链长 >= 3 层、id 全局唯一、trigger 不重复，见 `src/deep_common.py` 的 `grade_chain()`）。

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
