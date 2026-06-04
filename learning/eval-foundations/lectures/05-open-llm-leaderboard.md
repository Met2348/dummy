# L05 · Open LLM Leaderboard v2 (HF 2024)

## 背景

HuggingFace Open LLM Leaderboard 是开源社区的"高考排行"：
- v1 (2023.06)：ARC + HellaSwag + MMLU + TruthfulQA + Winogrande + GSM8K
- **v2 (2024.06)**：全替换，因为 v1 全饱和、污染严重

## v2 的 6 个 bench

| Bench | 测什么 | 题目数 |
|-------|------|--------|
| **MMLU-Pro** | 升级 MMLU（L03）| 12k |
| **GPQA** | google-proof Q&A，研究生级别 | 448 diamond |
| **MuSR** | multi-step reasoning（mystery/team allocation）| ~750 |
| **MATH Lvl 5** | MATH 中难度 5（最难）部分 | 1324 |
| **IFEval** | instruction following 严格 follow | 541 |
| **BBH** | Big-Bench Hard（L06）| 6512 |

## 为什么换

v1 问题：
1. **HellaSwag** 已饱和 (>95%)
2. **MMLU** 污染严重 + 标错
3. **TruthfulQA** 数据泄露（很多 paper 拿它训）
4. **GSM8K** 太简单

v2 设计原则：
- **难**：v1 顶模型 60-70%；v2 拉回 30-50% 区间
- **干净**：发布后立即冻结测试集，新 bench 抽样自 2024
- **多样**：6 个角度（知识/推理/数学/指令/逻辑/常识）

## 当下 (2025) 排行示例

| 模型 | Avg | MMLU-Pro | GPQA | MuSR | MATH 5 | IFEval | BBH |
|------|-----|----------|------|------|--------|--------|-----|
| Qwen 2.5 72B-Inst | 39.4 | 51.4 | 16.9 | 17.5 | 35.8 | 86.4 | 44.3 |
| Llama 3.1 70B-Inst | 33.6 | 47.8 | 14.2 | 16.5 | 21.6 | 86.4 | 50.2 |
| Mistral Large 2 | 36.1 | 52.7 | 13.7 | 16.2 | 35.5 | 84.7 | 48.7 |

可看出：MATH/MuSR/GPQA 仍卷不动 → 推理仍是开源 gap。

## 用 lm-evaluation-harness 跑

```bash
pip install lm-eval[vllm]
lm_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-7B-Instruct \
        --tasks leaderboard --batch_size auto
```

5090 24GB 单卡跑 7B model 全套 ~12h。

## 看 leaderboard 的姿势

1. **Avg ≠ 一切**：MATH 强不代表 IFEval 强（推理 ≠ 指令遵守）
2. **大小看 Pareto**：参数 3B 但 BBH 40 比 70B 25 更值钱
3. **看微调侧**：Base / Instruct 差距告诉你 fine-tune 收益
4. **看许可证**：商用 vs 研究用

## 一句话

> v1 是开源社区的小学考试，v2 是初中考试，o1/R1 在写高中题。
