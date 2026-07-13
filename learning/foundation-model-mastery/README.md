# Foundation Model Mastery —— 基座模型专题(浅→深→社招级别)

> 定位：老手在 `onsite-mastery` 三轮通关(初版208+前沿207+Staff Gauntlet 219+Social Hire Gauntlet 158=792点)基础上追加的要求——Foundation Model、Federated Learning、Diffusion 三个"enormous"级别的独立大专题，每个都要求 100+ 点、难度"浅→深→社招级别"全跨度。本 track 是这个队列里的第一个。

## 组织轴：难度分层，不是知识主题

`onsite-mastery` 的 `ai_deep`/`backend_deep`/`frontier_deep` 按知识主题组织，`staff_gauntlet`/`social_hire_gauntlet` 按面试关卡阶段组织；本 track 换了第三种组织轴——**按难度分层**：

| 层 | 目录 | 定位 | 点数 | 类型 |
|---|---|---|:--:|:--:|
| 浅(Tier 1) | `src/tier1_shallow/` | 对基座模型领域相对陌生候选人的入门认知框架 | 35 | DeepPoint |
| 深(Tier 2) | `src/tier2_deep/` | 已建立基础框架、需要接受连续追问的机制深水 | 43 | DeepPoint |
| 社招级别(Tier 3) | `src/tier3_social_hire/` | 已有实际工作经验、需要在真金白银预算/组织张力下拍板的资深战略判断 | 28 | ScenarioPoint |

**合计106点(78个DeepPoint + 28个ScenarioPoint)**，练法建议按 Tier 1→2→3 顺序推进——Tier 3 的资深判断题预设你已经具备 Tier 1/2 的知识框架，不建议跳着从 Tier 3 练起。

## 为什么不是简单重复 `ai_deep`/`frontier_deep` 已有的内容

`ai_deep`/`backend_deep`/`frontier_deep`/`staff_gauntlet`/`social_hire_gauntlet` 五个子包合计792点，标准Transformer内部机制(attention/位置编码/归一化)、RLHF/DPO/GRPO对齐、MoE路由、scaling law本身的数学规律、预训练数据配比这些角度已经被覆盖得很深，本track写每个文件前都先读了对应的已有文件确认差异化角度，只做真正的知识空白：

| 文件 | 覆盖的知识空白 | 点数 |
|---|---|:--:|
| `tier1_shallow/dp_fm_taxonomy_landscape.py` | 基座模型宏观分类框架(架构/开放程度/训练范式三维度)、模型家族谱系生态位、许可证与开放策略争议 | 18 |
| `tier1_shallow/dp_tokenizer_design.py` | Tokenizer设计(此前任何题库都只是顺带提及，从未独立成篇)：BPE/Unigram算法机制、vocab size权衡、多语言公平性、数字/代码tokenization耦合 | 17 |
| `tier2_deep/dp_non_transformer_architectures.py` | 非Transformer与混合架构(此前完全空白的知识维度)：SSM/Mamba/Jamba/RWKV/RetNet | 15 |
| `tier2_deep/dp_model_family_derivation.py` | 模型家族衍生工程：continual pretraining/蒸馏/模型合并(TIES/SLERP)/长上下文扩展 | 14 |
| `tier2_deep/dp_fm_evaluation_methodology.py` | 发布前评测方法论(区别于`dp_eval_safety`的RLHF安全评测角度)：污染检测/benchmark饱和/评测方差/LLM-judge偏见 | 14 |
| `tier3_social_hire/sc_fm_release_governance.py` | 发布治理判断(完全空白维度)：model card披露/分阶段发布/许可证选择/safety case论证 | 14 |
| `tier3_social_hire/sc_fm_training_economics_judgment.py` | 训练经济学与算力战略判断(完全空白维度)：scaling law确认实验、chip allocation、止损重启、买卡vs租卡 | 14 |

与已有五子包(792点)全局id/trigger程序化验证零冲突，六子包合计898点。

## 数据结构：复用已验证的 DeepPoint + ScenarioPoint

`src/deep_common.py` 是 `onsite-mastery/src/deep_common.py` 的独立副本(各 mastery track 自成一体，不跨track相互import)，`DeepPoint`(trigger+chain+pitfall)用于Tier1/2的追问链，`ScenarioPoint`(trigger+rubric+trap)用于Tier3的无标准答案场景判断。

用法：
```python
import sys
sys.path.insert(0, "learning/foundation-model-mastery/src")
from foundation_model_mastery import ALL_DP, ALL_SP, TIERS
from deep_common import drill, grade_chain, grade_scenario
```

`TIERS` 是按难度顺序排列的分层元数据(每项含 `n`/`name`/`label`/`kind`/`modules`)，供后续讲义/README引用。

## 环境与测试

```bash
python learning/foundation-model-mastery/src/tests/test_all.py
```

9个模块(deep_common + 7个内容文件 + 总聚合)全部通过 `_self_test()`。

## 诚实说明

内容全部来自各 agent 自主 WebSearch 验证的真实信息(具体模型版本号/许可证条款/论文编号/实测数据)，写作时发现并修正过一处编造的本地文件引用(`real_world_link` 字段)——凡是无法验证真实存在的引用一律留空，不留虚假路径。
