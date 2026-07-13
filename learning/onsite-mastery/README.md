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

## 层层通关:Staff Gauntlet 资深关卡训练(`src/staff_gauntlet/`)

老手拿前两版(初版208点+2026前沿补强207点,合计415点)三审后的反馈:"Frontier Lab招人那叫一个难",要求专门查清楚2026年这些实验室到底考什么,做一份"那种层次的面试层层通关教程"。用12轮WebSearch核实后发现,这次的差异化不是"更多同主题知识点",而是一个**全新的组织轴**:

1. **面试loop本身就是分阶段关卡制**:OpenAI/Anthropic/DeepMind的真实面试确实按顺序过关——screen(动机匹配)→research coding(论文转代码)→paper critique(论文批判)→系统设计判断→values round(价值观关)→跨团队协作。前两版都按**知识主题**分类,这次改为按**面试关卡阶段**组织,练法也不同:要按顺序过关,不是随机抽题。
2. **Staff/Senior的核心分水岭是"定方向"而非"知识深度"**:资深叙事关(gate7)专门训练"我主导了方向"而非"我执行了任务"这种更高阶的叙事框架。
3. **values round是完全空白的内容维度**:Anthropic式的价值观关,45分钟非技术、奖励"真实怀疑"而非"表演式认同"、奖励"讲自己出错的经历"而非"永远正确的故事"。
4. **"面试官临场改约束"是系统设计判断的新范式**:约束变了是打补丁还是推倒重来,是资深与否的行为分水岭。
5. **HPC/大规模集群基础设施是老手中途追加的专项要求**:大集群训模型的网络拓扑/故障容错/checkpoint工程是真正稀缺、此前"点到为止"的领域,单独成一整关(gate11),锚定Meta Llama 3 405B真实故障数据(419次中断/54天/16384块H100)等具体数字。
6. **定级/谈判判断是另一个完全空白的维度**:面试官质疑你的level时怎么用证据argue、多offer怎么比较含金量而不是只看title。

同时,考虑到候选人项目里大量代码可能是AI coding agent代写而非自己深度理解("研究玩具级别代码"风险),多个关卡(尤其gate2/gate7/gate9)专门设计了"AI代写代码穿帮测试"——面试官让你现场修改/调试/解释一段"你说是你写的"代码,如果答不上来暴露的正是没有真正吃透的部分。

**合计219点(6个DeepPoint关120点 + 5个ScenarioPoint关99点)**,与前两版415点全局id/trigger均无冲突(四个子包合计634点,程序化验证过)。

| 关 | 类别(`src/staff_gauntlet/`) | 点数 | 类型 | 对应真实面试阶段 |
|---|---|:--:|:--:|---|
| 01 | 动机筛选与方向匹配 | 19 | DeepPoint | recruiter/HM screen |
| 02 | 论文转代码研究手写 | 20 | DeepPoint | research coding / take-home |
| 03 | 论文批判与研究流利度 | 19 | DeepPoint | paper discussion round |
| 04 | ML基础设施系统设计判断 | 20 | ScenarioPoint | 分布式训练/评测基建设计+临场改约束 |
| 05 | Agent生产系统设计判断 | 20 | ScenarioPoint | orchestrator-vs-LLM边界判断 |
| 06 | 价值观与安全立场关(Values Round) | 21 | ScenarioPoint | Anthropic式45分钟非技术关 |
| 07 | 资深叙事与研究方向主导权 | 20 | DeepPoint | Staff vs Senior差异化叙事 |
| 08 | 跨团队协作与模糊决策判断 | 20 | ScenarioPoint | 组织张力/资源冲突判断 |
| 09 | 国内大厂资深社招视角 | 20 | DeepPoint | P8级决策权/风险取舍/带教 |
| 10 | 定级与谈判判断 | 18 | ScenarioPoint | leveling/offer谈判 |
| 11 | 大规模集群HPC基础设施深水 | 22 | DeepPoint | 网络拓扑/故障容错/checkpoint工程 |

用法:`from staff_gauntlet import ALL_DP as SG_DP, ALL_SP as SG_SP, GATES`,`GATES`是按真实面试loop顺序排列的关卡元数据(每项含`n`/`name`/`cat`/`kind`/`bank`),配合 `deep_common.drill()`/`grade_chain()`/`grade_scenario()` 使用。**练法和前三个子包不同**:应该按`GATES`里`n`从1到11的顺序过关,而不是随机抽题——详见 [`00-how-to-defend.md`](lectures/00-how-to-defend.md) 新增的第7节。

## 层层通关(下篇):Social Hire Gauntlet 国内大厂资深社招全流程通关(`src/social_hire_gauntlet/`)

老手四审后的反馈:社招和校招的核心差异不在知识点本身,而在于面试全流程里那些`staff_gauntlet`(对标Frontier Lab)没有覆盖的**国内大厂特有关卡**——项目拷打的量化拷问方式、"AI作弊检测时代"下手撕代码这一关本身的性质已经变了、隐藏bug调试的真实代码级深水、P7/P8业务委员会评审、离职话术与谈薪战术、背景调查时代的诚实一致性、反问环节的战略性。这次同样按**面试关卡阶段**组织,但对标国内(阿里/字节/腾讯等)资深社招真实流程,而非frontier lab的research loop:

1. **"手撕代码"这一关的性质在2026年已经变了**:gate2专门锚定Fabric对19,368场AI辅助面试的分析(2025年7月到2026年1月,AI作弊信号触发率从9%涨到45%,整体38.5%)、"为什么用这个变量名"这类现场追问技术、双设备监控"四维防控体系"——这不是传统的"手撕算法题"复习,而是"你的临场表现本身正在被用新方式验证"这个新现实。
2. **隐藏bug调试深水(gate3)是完全区别于`dictation/`的新练法**:`dictation/`练"闭卷默写已学过的标准实现",gate3是给一段**看起来正常、有具体触发条件的真实bug代码**(可变默认参数、闭包延迟绑定、TOCTOU竞态、numpy溢出、异步单线程假设错误等21类真实bug),练的是"读代码预测运行时行为→定位根因→系统性预防",而非默写记忆。
3. **P7/P8交叉面/委员会面(gate5)是国内大厂特有的评审机制**:三个P9一致同意才能通过的晋升委员会机制、业务战略定位追问、字节2-1/2-2↔阿里P6/P7/P8的title映射(明确按"有争议"如实呈现,不是编一个假共识)。
4. **离职话术与谈薪谈判(gate6)是完全空白的实战维度**:10个离职原因场景(与主管关系差/纯粹嫌钱少/业务转型被边缘化/裸辞空窗期怎么解释)+10个谈薪战术场景(不主动报价/多offer杠杆/薪资单验证要求),都不是`staff_gauntlet`gate10练的frontier lab title/equity比较。
5. **背景调查时代的诚实一致性(gate7)**:锚定国内约38%简历美化/12%核心事实造假的数据、offer后置背调的真实时序压力、BEI/STAR面试技巧交叉验证候选人陈述一致性。
6. **反问环节与快速融入(gate8)**:反问环节本身在评估什么信号(而非只是走过场)、社招心态从"公司给我时间熟悉"转变到"我要主动驱动融入"。

与`staff_gauntlet`的差异化边界(避免重复,已在`__init__.py`模块文档里显式说明):gate4(系统设计主导权)vs staff_gauntlet gate4(约束临场改判断)——这里练"主动驱动设计对话"本身；gate5(交叉面)vs staff_gauntlet gate9(国内资深社招视角)——这里专练P7/P8委员会评审机制，不重复分布式训练等已覆盖知识点。

**合计158点(5个DeepPoint关100点 + 3个ScenarioPoint关58点)**,与前四个子包(415+219=634点)全局id/trigger均无冲突(五个子包合计792点,程序化验证过)。

| 关 | 类别(`src/social_hire_gauntlet/`) | 点数 | 类型 | 对应真实面试阶段 |
|---|---|:--:|:--:|---|
| 01 | 项目拷打与个人贡献量化 | 20 | DeepPoint | 简历深挖/项目defend |
| 02 | 手撕代码与AI作弊检测时代 | 20 | DeepPoint | 手撕算法/AI辅助面试监测 |
| 03 | 隐藏Bug调试深水 | 21 | DeepPoint | debug round/代码阅读能力 |
| 04 | 系统设计主导权判断 | 20 | ScenarioPoint | 系统设计面(主动驱动力) |
| 05 | 交叉面与委员会面(P7/P8业务视角) | 20 | DeepPoint | 委员会评审/交叉面 |
| 06 | 离职原因与谈薪谈判 | 20 | ScenarioPoint | HR面/谈薪谈判 |
| 07 | 背景调查时代的诚实一致性 | 18 | ScenarioPoint | offer后背调阶段 |
| 08 | 反问环节与快速融入新业务 | 19 | DeepPoint | 反问环节/入职前评估 |

用法:`from social_hire_gauntlet import ALL_DP as SH_DP, ALL_SP as SH_SP, GATES`,同样应按`GATES`里`n`从1到8的顺序过关,而非随机抽题。

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
