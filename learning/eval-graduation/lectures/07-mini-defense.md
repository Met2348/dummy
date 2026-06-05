# L07 · mini-defense 加固

## 防御 pipeline

```
input_classifier (mock Llama Guard)
    ↓ unsafe → BLOCK
ckpt 生成
    ↓
output_classifier (mock leak detector)
    ↓ unsafe → BLOCK
返回
```

## 效果（vanilla 例）

```
no defense:
- direct        ASR 100%
- persona_wrap  ASR 100%
- multi_turn    ASR 100%
mean: 100%

with defense:
- direct        ASR 0% (input blocked)
- persona_wrap  ASR 0% (input blocked)
- multi_turn    ASR 0% (input blocked)
mean: 0%
```

→ 即使 vanilla 这种"全漏"模型，加 input classifier 也能 0% ASR。

## 5 ckpt 对照

```
| ckpt | no_def avg | with_def avg | reduction |
|---|---:|---:|---:|
| vanilla   | 100% | 0% | -100% |
| lora      | 0%   | 0% | -0%   |
| dpo       | 0%   | 0% | -0%   |
| r1_tiny   | 0%   | 0% | -0%   |
| phi_tiny  | 0%   | 0% | -0%   |
```

注：本身已安全的 ckpt 加 classifier 也没用（已 0%）。

## 真世界意义

```
weak open-source LLM + Llama Guard 3 = OK 安全
strong closed LLM + Constitutional Classifiers = 极强
```

→ classifier 是**"快速救活弱模型"**的方法。

## 一句话

> classifier 是"安全大补丸" — 即使 base 模型很弱也能救回来。
