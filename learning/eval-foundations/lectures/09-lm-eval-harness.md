# L09 · lm-evaluation-harness — EleutherAI 标准工具

GitHub: `EleutherAI/lm-evaluation-harness`

## 为什么这是事实标准

1. **600+ task** 全收录（MMLU/BBH/GSM8K/HumanEval/...）
2. **统一 backend**：HF transformers / vllm / OpenAI API / Anthropic API / TGI
3. **Open LLM Leaderboard 官方**用它
4. **持续更新**：新 bench 几天就 merge

## 五分钟上手

```bash
pip install lm-eval

# HF model
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-0.5B \
        --tasks mmlu --num_fewshot 5 \
        --batch_size 8 \
        --output_path results/

# vLLM 加速
lm_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-0.5B \
        --tasks leaderboard --batch_size auto
```

## 关键概念

| 概念 | 含义 |
|------|------|
| **task** | bench 单元（如 `mmlu_high_school_math`）|
| **group** | task 集合（如 `leaderboard` = 6 bench）|
| **filter** | 后处理（answer extract、regex）|
| **metric** | acc / acc_norm / exact_match / perplexity / ... |
| **num_fewshot** | k-shot 数 |
| **chat_template** | apply tokenizer 的对话模板 |

## acc vs acc_norm

```
acc        = exact letter match
acc_norm   = log-prob 比较 (P(option) / option_length)
```

对短答案 acc，对长答案常用 acc_norm。

## task config 长这样

```yaml
# tasks/mmlu/_default_template_yaml
task: mmlu
dataset_path: hails/mmlu_no_train
test_split: test
doc_to_text: "Question: {{question}}\nA. {{choices[0]}}\n..."
doc_to_choice: ['A', 'B', 'C', 'D']
doc_to_target: "{{['A','B','C','D'][answer]}}"
metric_list:
  - metric: acc
num_fewshot: 5
```

## 自定义 task

3 步：
1. 写 yaml 配置
2. 实现 `doc_to_text` / `doc_to_target`
3. 注册到 `tasks/`

我们 src/lm_eval_adapter.py 提供 mock 版 evaluate_tasks API，可作为本地替身。

## 对比其他工具

| 工具 | 强项 |
|------|------|
| **lm-eval-harness** | 任务最多、社区最大 |
| HELM | 全息（多 metric）|
| OpenCompass | 中文 bench 全 |
| BIG-bench | 原版基准 |

## 实际跑全套有多贵

- Qwen 7B 跑 leaderboard 全套，5090 24GB：≈ 12h
- API 模型（GPT-4）跑全套：≈ $300-500

## 一句话

> lm-eval-harness 是评测界的 PyTorch — 谁绕开它谁吃亏。
