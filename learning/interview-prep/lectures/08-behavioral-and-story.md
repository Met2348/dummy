# L08 · behavioral:把 85 专题讲成 impact

## 问题:自学难讲"impact"

behavioral / 项目轮里,面试官挖的是"**你**做了什么、**多少**提升、什么 tradeoff、什么翻过车"。85 个 mock 专题若讲成"我学了很多",是弱信号。要重新包装成**有决策、有结果、有反思**的故事。

## STAR + 量化

每个故事套 **Situation → Task → Action → Result**,Result 尽量带数字:

- 弱:"我实现了几种 agent 设计模式。"
- 强:"我把同一个任务用 6 种 agent 设计各实现一遍,量化出 routing 比 autonomous agent **省 2.3× 调用**——所以我主张'模型固定时,架构就是成本杠杆',而不是默认上最重的 agent。"

后者展示了:动手、量化、**有观点的取舍**。

## 从你的 portfolio 里提炼 3 个招牌故事

1. **"架构即成本杠杆"**(agent-design-patterns / harness):同任务多设计对照 + 成本数字。
2. **"model × harness"**(agent-harness-design):同模型换权限模式,readonly 拦截并 surface,agent 优雅收尾——展示你懂**系统性 failure mode**。
3. **hero project**(可解释性真复现):真模型、真数字、真发现——这是最强的一个,也接你 PhD。

## 必备的"反思"素材

面试官爱问"你犯过的错 / 最难的 debug"。从真经历里挑:
- 一个**假设被数据推翻**的时刻(展示更新信念)。
- 一个**tradeoff 你主动选了次优但更合适**的决定。
- 一次**验证救了你**的经历(如 KV cache 对拍全量发现 off-by-one)。

## 练法

写下 3 个招牌故事的 STAR 脚本,每个能 **2 分钟讲完 + 承接追问**。录音,删掉"我们"改成"我"(讲清你的个人贡献),给每个 Result 补一个数字。
