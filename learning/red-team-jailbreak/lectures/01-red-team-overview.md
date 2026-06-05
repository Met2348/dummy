# L01 · Red-team 历史 + Anthropic 框架

## 安全声明（本 Topic 必读）

本 Topic 所有代码都是**教学 mock**：
- 不针对真实生产模型
- 不输出有效 jailbreak prompt
- 目的：理解攻击 → 才能更好防御（Topic 6）

**严禁**用所学攻击未授权的他人模型。

## 红队 = 安全研究的攻击侧

```
红队 (red team) = 主动找漏洞
蓝队 (blue team) = 加固防御
紫队 (purple team) = 攻防协同
```

LLM 红队 ≠ 网络安全红队，但范式一致。

## 历史

| 时间 | 事件 |
|------|------|
| 2022.04 | Anthropic 第一篇 manual red-team paper (32 人手动) |
| 2022.06 | OpenAI ChatGPT alpha (内部红队) |
| 2023.04 | LLaMA-2 leak → 民间 jailbreak 爆发 |
| 2023.07 | **GCG** (Zou) gradient-based 攻击 |
| 2023.10 | DAN/AutoDAN/PAIR 自动化兴起 |
| 2024.04 | **Many-shot Jailbreak** (Anthropic) |
| 2024.05 | **Crescendo** (Microsoft) multi-turn |
| 2025.02 | **Constitutional Classifiers** (Anthropic) — 防御王者 |

## 自动化阶梯

```
manual prompt → 几十 trial
   ↓
templates (DAN, AIM) → 1 模板试万 query
   ↓
LLM-attacks-LLM (PAIR) → 主动迭代
   ↓
gradient (GCG) → 白盒优化
   ↓
multi-turn (Crescendo) → 渐进策略
   ↓
agentic red-team → LLM 自动管理整轮
```

## Anthropic 红队框架（2022）

3 角色：
- **Attacker**：人或 LLM
- **Target**：被攻 LLM
- **Judge**：另一个 LLM 评 "complied vs refused"

3 评估维度：
- **Attack Success Rate (ASR)**：complied / total
- **Diversity**：攻击多样性
- **Cost**：攻击代价（trials / tokens）

## 真实 ASR 数字

| 模型 | GCG ASR | PAIR ASR | AutoDAN ASR |
|------|---------|----------|-------------|
| Vicuna-7B | 99% | 100% | 100% |
| GPT-3.5 | 86% | 60% | 25% |
| GPT-4 | 47% | 40% | 30% |
| Claude 2 | 4% | 5% | 4% |
| Claude 3 Opus | 2% | 9% | 1% |
| **Claude 3.7 + Constit. Classifiers** | **0.4%** | **2.0%** | **0.6%** |

→ 现代 frontier model 攻击难度上去了。

## 本 Topic 覆盖

L02: jailbreak 4 大类
L03-L09: 7 大攻击方法
L10: 多模态
L11: HarmBench
L12: Capstone red-team matrix

## 一句话

> 红队 = 找洞，蓝队 = 补洞 — 两侧都得懂才能造安全 LLM。
