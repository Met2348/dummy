# Foundation Model Mastery 使用说明

## 为什么这个track的组织轴是"难度分层"

`onsite-mastery` 已有的五个子包分别按"知识主题"(`ai_deep`/`backend_deep`/`frontier_deep`)和"面试关卡阶段"(`staff_gauntlet`/`social_hire_gauntlet`)组织。老手这次的要求是"Foundation Model 单独做一个100+的大专题，难度深度级别必须浅到深最后到社招级别"——这是第三种组织轴：**同一个主题(基座模型)内部，按难度台阶从入门到资深纵向切分**，而不是横向切成多个知识主题或多个面试阶段。

## 三层怎么练

1. **Tier 1(浅，`tier1_shallow/`）先练**：这一层是认知框架层，练法和 `deep_common.grade_chain` 用法一致——看到 `trigger` 先自己口头/写下答案，再用 `chain` 里的参考答案+采分词自查。这一层答不上来说明基座模型这个领域的地图你还没画清楚，不要跳过直接去练 Tier 2。
2. **Tier 2(深，`tier2_deep/`）**：机制深水层，追问链通常有3-4层，最后一层经常是"这个技术为什么会失败/局限在哪"。这一层特别要注意 `dp_non_transformer_architectures.py`——这是此前所有题库(792点)都完全没覆盖的全新维度(SSM/Mamba/RWKV这类非Transformer架构)，不要因为陌生就跳过，这恰恰是老手强调"要有深度"最容易补的知识盲区。
3. **Tier 3(社招级别，`tier3_social_hire/`）最后练**：这一层是 `ScenarioPoint`，无标准答案，练法是对着 `rubric` 自查覆盖面而不是找"正确答案"。这一层的两个文件(发布治理判断、训练经济学与算力战略判断)考的是"没有唯一正确答案、需要在巨大不确定性和真金白银成本下拍板"的资深判断力，如果发现自己只会说"看情况"而给不出具体判断依据，对着 `trap` 字段检查自己具体漏掉了哪个维度。

## 和已有五个子包的关系

不是重复练习——如果你在 `ai_deep`/`frontier_deep` 里已经练过 Transformer attention/RLHF/MoE/scaling law 这些内容，这里不会再考一遍，本track专门补的是这些子包完全没覆盖的角度(见 README.md 表格)。如果你还没练过 `ai_deep`，建议先完成那边的基础内容，再来练这个track的Tier 2/3，否则会缺乏必要的背景知识去理解追问链里的机制细节。
