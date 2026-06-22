# RL Foundations — 强化学习基础（PPO 为核心）

> 7 专题 RL 系列的第 1 站。配套书籍《大模型算法：强化学习、微调与对齐》Ch3 + Schulman 的 PPO + GAE 论文。
>
> 设计文档：[2026-06-03-rl-foundations-design.md](../../docs/superpowers/specs/2026-06-03-rl-foundations-design.md)
> 实施计划：[2026-06-03-rl-foundations.md](../../docs/superpowers/plans/2026-06-03-rl-foundations.md)

---

## 专题概览

| # | Lecture | 主方法 | 核心 idea | Notebook |
|---|---------|-------|----------|----------|
| 01 | mdp-policy | REINFORCE | PG 定理 / log-derivative trick | [.ipynb](notebooks/01-mdp-policy.ipynb) |
| 02 | actor-critic | A2C | TD advantage / 共享 backbone | [.ipynb](notebooks/02-actor-critic.ipynb) |
| 03 | trpo | TRPO | trust region / Fisher / natural gradient | [.ipynb](notebooks/03-trpo.ipynb) |
| 04 | ppo-core | **PPO** ⭐⭐⭐⭐⭐ | clip 一行替代 trust region | [.ipynb](notebooks/04-ppo-core.ipynb) |
| 05 | gae | GAE | λ-return / 反向递推 5 行 | [.ipynb](notebooks/05-gae.ipynb) |
| 06 | ppo-tricks | 7 件套 | Engstrom 2020 工程细节 | [.ipynb](notebooks/06-ppo-tricks.ipynb) |
| 07 | cartpole-lab | 5 算法横向 | gymnasium / TensorBoard | [.ipynb](notebooks/07-cartpole-lab.ipynb) |
| 08 | ppo-for-llm | LLM-PPO | 4 model / token-level PPO | [.ipynb](notebooks/08-ppo-for-llm.ipynb) |
| 09 | toy-rl-llm | sentiment 玩具 | BERT-SST2 当 RM | [.ipynb](notebooks/09-toy-rl-llm.ipynb) |
| 10 | rl-pitfalls | 陷阱合集 | Reward Hacking / Over-optimization | [.ipynb](notebooks/10-rl-pitfalls.ipynb) |
| 11 | **Capstone** | IMDb PPO | GPT-2 + BERT-SST2 + trl | [.ipynb](notebooks/11-capstone-ppo-llm.ipynb) |
| 12 | summary | 引出 RLHF | 桥接专题 2 | [.ipynb](notebooks/12-summary.ipynb) |

**估计学时**：14 h（lecture + 实操）

---

## 学习路径

```
        L01 MDP & REINFORCE
                |
        L02 Actor-Critic / A2C
                |
        L03 TRPO   ← 数学推导
                |
        L04 PPO Core ⭐ 必学
        L05 GAE
        L06 PPO Tricks
                |
        L07 CartPole Lab（实战）
                |
   ↓ 切换到 LLM ↓
        L08 PPO for LLM
        L09 Toy RL-LLM
        L10 RL Pitfalls
                |
        L11 Capstone (IMDb PPO)
        L12 Summary → 专题 2 RLHF
```

---

## 目录结构

```
rl-foundations/
├── README.md
├── environment/
│   ├── requirements.txt
│   └── verify_env.py
├── papers/
│   └── README.md           # 12 篇论文索引
├── lectures/
│   └── 01..12-*.md         # 12 lecture markdown
├── src/
│   ├── common.py           # GAE / log_prob / KL / RolloutBuffer
│   ├── reinforce_minimal.py
│   ├── a2c_minimal.py + a2c_sb3.py
│   ├── trpo_minimal.py
│   ├── ppo_minimal.py + ppo_sb3.py
│   ├── gae.py + ppo_tricks_ablation.py
│   ├── cartpole_full.py    # 5 算法切换
│   ├── ppo_gpt2_minimal.py + ppo_gpt2_trl.py
│   ├── sentiment_reward.py
│   ├── capstone_imdb_ppo.py
│   └── tests/
│       ├── test_reinforce_cartpole.py
│       ├── test_a2c_cartpole.py
│       ├── test_ppo_consistency.py
│       └── test_gpt2_ppo.py
└── notebooks/
    └── 01..12-*.ipynb
```

---

## 环境配置

```powershell
# 复用 PEFT 系列已装的 torch cu130 nightly 环境
pip install -r learning/rl-foundations/environment/requirements.txt

# 三段式验证
python learning/rl-foundations/environment/verify_env.py
# 预期：Part A/B/C 全 PASS
```

关键依赖：
- `gymnasium>=0.29` — RL 环境
- `stable-baselines3>=2.3` — 算法对照
- `trl>=0.13` — LLM-PPO
- `peft>=0.19` — capstone LoRA

---

## 横向对比

| 算法 | critic | KL 约束 | 实现难度 | LLM 适用 | CartPole 表现 |
|------|--------|--------|---------|---------|--------------|
| REINFORCE | ✗ | ✗ | 极易 | 仅玩具 | ~200 |
| A2C | ✓ TD | ✗ | 易 | ✓ | ~480 |
| TRPO | ✓ | KL ≤ δ | 难 (F+CG) | ✗ | ~490 |
| **PPO** ⭐ | ✓ | **clip** | 中 | ✓ 默认 | **~500** |

**算法选型决策树**：
- 教学 / 调试 → REINFORCE / A2C
- 论文复现严格 → TRPO（但 PPO 几乎同效）
- 工业 / LLM-RL → **PPO**（无脑选）

---

## 关键公式（脑内 cheatsheet）

```
g_PG = ∇log π · A                            ← PG 定理
δ_t = r_t + γV(s_{t+1}) - V(s_t)             ← 1-step TD
A^GAE_t = δ_t + γλ A^GAE_{t+1}               ← GAE 反向递推
r(θ) = π_θ / π_old                           ← IS ratio
L_clip = -min(r·A, clip(r, 1±ε)·A)           ← PPO clip
KL_approx = E[r - 1 - log r]                 ← Schulman cheap KL
```

---

## 自测 12 题

1. 推导 PG 定理：从 `∇_θ E[G(τ)]` 出发，到 `Σ ∇log π(a_t|s_t) · G_t`。
2. 比较 1-step TD / Monte-Carlo / GAE 三者在 bias-variance 上的差异。
3. PPO 的 `min(r·A, clip(r, 1±ε)·A)` 中为何取 `min`？
4. 当 `A > 0` 且 `r > 1+ε` 时，PPO 的梯度方向是什么？
5. TRPO 的 Fisher 信息矩阵 `F` 是什么，为何不直接存？
6. GAE λ=0 与 λ=1 各等价于什么经典方法？
7. LLM-PPO 的 4 模型分别负责什么？
8. KL ref penalty 加在 reward 内还是 loss 外？两种实现的差异？
9. Reward Hacking 5 类各举一个例子。
10. PPO 7 件套，影响最大的是？
11. Spurious Rewards 给我们什么教训？
12. 为什么 DPO 之后是 R1（PEFT vs RLHF vs R1 各改什么）？

---

## Git 里程碑

| Tag | 内容 |
|-----|------|
| `rl-foundations` | 本专题完结 |
| `rl-base-pg` | L1+L2 REINFORCE + A2C |
| `rl-base-ppo` | L3-L6 TRPO + PPO + GAE + tricks |
| `rl-base-cartpole` | L7 实战 |
| `rl-base-llm` | L8-L10 LLM-PPO |
| `rl-foundations` | L11+L12 Capstone + 完结 |

---

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V0+V1 验证通过。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules rl-foundations
> ```

**CartPole 5 算法横向**（每个约几分钟；smoke 用 `--total-steps 1500`）：

```powershell
foreach ($algo in 'reinforce','a2c','trpo','ppo','sb3_ppo') {
    python learning/rl-foundations/src/cartpole_full.py --algo $algo --total-steps 30000
}
```

**Capstone：GPT-2 + IMDb + SST-2 RM + PPO**：

```powershell
# 真实跑（mean_R(pos-prob) 上升）
python learning/rl-foundations/src/capstone_imdb_ppo.py --total-iters 30 --batch-size 4
# 快速 smoke（验证可跑通）
python learning/rl-foundations/src/capstone_imdb_ppo.py --total-iters 2 --batch-size 4 --n-prompts 32
```

> ⚠️ **trl 版本坑**：本仓库环境装的是 trl 1.5.x，已移除经典 `PPOConfig`/`PPOTrainer` 情感微调 API。
> capstone 会**自动回退**到手写 token-level PPO（复用 `ppo_gpt2_minimal.py`）+ 真实 SST-2 RM，照样在 3080 Ti 上跑通。
> 纯 trl 对照 demo `ppo_gpt2_trl.py` 需 trl<0.12，否则会显式报错指路手写版（不会静默假成功）。
> IMDb 用命名空间 id `stanfordnlp/imdb`；离线时 capstone 回退内置 prompts。

**测试（V2）**：

```powershell
python -m pytest learning/rl-foundations/src/tests/ -v -m "not slow"
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules rl-foundations --tests
```

---

## 跨专题衔接

| 下一专题 | 衔接点 |
|---------|-------|
| `rlhf-classic` | InstructGPT 三段管线，本专题 capstone 用 BERT-SST2 代 RM，下一专题训自己的 RM |
| `dpo-family` | 用 DPO 替代 RLHF，本专题 PPO+KL 公式直接演化成 DPO loss |
| `process-reward` | PRM 工具箱，准备进 R1 时代 |
| `reasoning-r1` ⭐⭐⭐ | GRPO = PPO 去 critic + group baseline + rule reward |
| `rl-sota-2026` | DAPO 4 件套 = PPO + 4 trick |
| `multimodal-agent` | VLM-R1 / Agent RL + 五线综合毕业 |

---

## 完成验收（自查）

- [ ] 12 lecture 全过
- [ ] 12 notebook 全跑（CartPole 类可短跑）
- [ ] 5 算法 src 全部看过（不需要逐行精读）
- [ ] CartPole capstone：sb3 PPO 200k step 达 500
- [ ] LLM capstone：GPT-2 + IMDb sentiment 提升 ≥ 30%
- [ ] verify_env.py Part A/B/C 全 PASS
- [ ] tag `rl-foundations`

---

🎓 **专题 1 完成 → 进入专题 2 RLHF Classic**
