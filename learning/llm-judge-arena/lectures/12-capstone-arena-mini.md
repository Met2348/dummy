# L12 · Capstone — mini-Arena (5 mock ckpt)

## 目标

5 个 mock ckpt 互打（round-robin），fit Bradley-Terry，输出 Elo 排行。
对应 Module 4 RL 系列的 5 个里程碑 ckpt：

- **vanilla** (GPT-2 base)
- **lora** (PEFT 后)
- **dpo** (对齐后)
- **r1_tiny** (R1-Zero 推理强化)
- **phi_tiny** (Phi 风格 data 优化)

## 跑

```python
from mini_arena import (
    make_arena_models, run_round_robin, make_leaderboard, to_md
)
from common import make_length_judge

models = make_arena_models()
judge = make_length_judge(prefer_longer=True)  # toy judge
battles = run_round_robin(models, judge)
lb = make_leaderboard(battles)
print(to_md(lb))
```

预期输出：
```
# mini-Arena leaderboard
| rank | model | Elo |
|---|---|---:|
| 1 | r1_tiny  | 1700+ |
| 2 | dpo      | 1550+ |
| 3 | phi_tiny | 1500 |
| 4 | lora     | 1400 |
| 5 | vanilla  | 1300 |
```

注：toy length judge 偏好长答案，所以 r1_tiny（最 verbose）赢。
真 judge 需用 prometheus 2 mock 才有意义。

## 设计要点

1. **5 ckpt 命名对应 module 系列**（与 Module 5 graduation capstone 一致）
2. **round-robin + 双向 swap**：5*4*8*2 = 320 battles
3. **MM algorithm fit BT** (n_iter=200)
4. **Elo base=1500，scale=400/ln10**

## 真模型推广

```python
# 用真 ckpt
real_models = {
    "vanilla": load_gpt2_base,
    "lora": load_lora_ckpt,
    "dpo": load_dpo_ckpt,
    "r1_tiny": load_r1_tiny_ckpt,
    "phi_tiny": load_phi_tiny_ckpt,
}
real_judge = load_prometheus2()  # 7B 本地

battles = run_round_robin(real_models, real_judge)
print(to_md(make_leaderboard(battles)))
```

## 退出条件

- 5 个 model 都有 Elo
- BT 数值合理（差距 100-500）
- mini_arena._self_test PASS

## 一句话

> Mini-Arena = 5 ckpt 自家"Chatbot Arena"，体验 BT-Elo 排行机制。
