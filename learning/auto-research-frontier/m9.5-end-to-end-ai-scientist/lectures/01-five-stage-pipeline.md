# L01 · AI Scientist 的五阶段，以及我们怎么诚实地缩小它

## 1. The AI Scientist 把"做研究"拆成五阶段

来自 [The AI Scientist v1](https://arxiv.org/abs/2408.06292)（Sakana AI, 2024）与 [v2](https://arxiv.org/abs/2504.08066)（2025，树搜索版）：

1. **Ideation（创意）**：生成研究 idea / 假设。
2. **Experiment（实验）**：把 idea 变成可执行实验、写代码、跑。
3. **Analysis（分析）**：收集结果、画图、判断假设是否成立。
4. **Writeup（写作）**：自动写出完整论文。
5. **Review（评审）**：用 LLM 给自己的论文打分。

v1 靠**人写的模板**起步；v2 去掉模板，改用**agentic 树搜索 + 实验经理 agent + VLM 看图反馈**，并诚实承认：更自主的 v2 **成功率反而更低**。

## 2. 我们的缩小版：把"假"换成"真"

业界 demo 的通病（见 [Hidden Pitfalls](https://arxiv.org/abs/2509.08713)）是**实验那一环常常是假的**——硬编码分数、幻觉结果。本模块反其道而行：

| 阶段 | 我们的实现 | 关键：真在哪 |
|------|-----------|-------------|
| Ideation | `ideation.py` 模板 idea 库（可插真 LLM） | idea 自带"事前自评 novelty"，留作打脸 |
| Experiment | `experiment.py` **真训** torch MLP on make_moons | 指标来自真实训练，确定性可复现 |
| Analysis | `analysis.py` 跨种子 mean±std + `verdict()` | 判定带误差棒，不靠单跑一次 |
| Writeup | `writeup.py` 1 页报告 | 数字==实验真值（test 锁死） |
| Review | `review.py` mock 评审 | **故意可被刷**，教你别信 |

## 3. 为什么"任务种子 vs 模型种子分离"很重要

`experiment.py` 里 `data_seed`（任务）和 `seed`（模型/训练）是分开的：

- 对照 baseline vs treatment 时，**数据集必须一模一样**，只变模型/训练——否则你比的是两件事，结论无意义。
- 跨多个**模型种子**重复，得到 mean±std，量化"这点提升是不是只是初始化运气"。

这是最朴素却最常被 demo 忽略的实验纪律。AI Scientist 被批"每个 idea 只跑一两次实验、结论不可靠"（见各系统 Limitations），根子就在这。

## 4. 动手

```powershell
python ../src/run.py --idea all --device cpu      # 看五阶段光谱
```

读 `pipeline.py` 的 `run_pipeline`：它就是把上面五个文件按 1→5 串起来。**下一讲**我们盯住第 2、3 阶段，看"真实验"怎么逼出 ideation-execution gap。
