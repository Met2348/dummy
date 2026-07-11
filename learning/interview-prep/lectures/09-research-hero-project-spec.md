# L09 · 研究 Hero Project Spec(对接 PhD 仓,不在本 track 硬建)

> 这是 spec 不是实现。你已有 `research/` 仓(70-paper repo + 24-src audit + judge-internals 方向 + 15pt ideation),hero project 应**并入那里**,不另起炉灶。本页给面试导向的成型标准。

## 目标(一个,不要多)

在**真模型**上,针对**"LLM-as-judge 内部是否编码了可被 reward-hack 利用的信号"**,跑出一个**真数字、可复现、可讲**的最小结果。既是面试弹药,又是 PhD 第一年的种子。

## 为什么是它

- 直接命中你 PhD 方向(judge-internals probing / reviewer reward-hack detection via mech interp)。
- 用得上你已建的**概念专题**(`probing-and-activations` / `causal-interventions` / `circuits-attention` / `llm-judge-arena`)——把概念**变成真模型上的数字**。
- 面试三用:behavioral 素材 + 研究品味 + 博士开局。

## 最小可交付(MVP,能讲就行)

1. **模型**:开源可跑的 judge(如 Pythia / GPT-2 规模的 reward/judge 模型,或小型指令模型当 judge)。CPU/单卡即可起步。
2. **探针(probing)**:在中间层激活上训线性探针,读"该回答会被判高分"这一信号,报**逐层准确率曲线**。→ 信号在哪层线性可读?
3. **因果验证(activation patching)**:把"谄媚/长度"等已知 reward-hack 特征对应的激活 patch 进去,看 judge 打分是否被因果推动。→ 不只相关,是因果。
4. **一张图 + 一段结论**:probing 曲线 + patching 效应量。结论一句话:"judge 在第 k 层线性编码了 X,且干预它能使评分偏移 Δ。"

## 成型标准(面试可讲)

- [ ] 公开 repo:README + 一键复现脚本 + 真数字(不是 mock)。
- [ ] 能 2 分钟讲清:问题 → 方法 → 结果 → 局限 → 下一步。
- [ ] 至少一个**被数据推翻的假设**(展示信念更新)。
- [ ] 明确 threat model / 局限:小模型不代表前沿、probing≠因果的边界。

## 安全边界(沿用全仓约束)

- 只做**检测/可解释性**方向,不产出可用于攻击生产模型的有效 jailbreak。
- reward-hack 演示限于**受控、开源、小模型**;不针对未授权的生产系统。

## 与本 track 的关系

本 track(interview-prep)只放**这份 spec**;真实现走你的 `research/` 仓。原因:研究是活的、要和你导师/方向迭代,塞进教学 track 会僵化,也会和你已跑的 audit/ideation 重复。
