# L09 · 39-Topic Portfolio v2 设计

## Portfolio v2 vs v1

| 维度 | v1 (Module 6 末) | v2 (本 Module 7 末) |
|------|-----------------|--------------------|
| 专题数 | 32 | **39** (+ Module 7 7 个) |
| 大画像 | 5 | **6** (+ 造 agent 产品) |
| Capstone | 3 (HELM/Arena/Red) | **6** (+ DRA / τ-bench / Portfolio v2) |
| 决策树 | 评测/造模型 | + agent framework 选型 |

## Portfolio v2 结构（约 300 行）

```markdown
# 39-topic LLM Learning Portfolio (v2)

> Generated 2026-06-05 · Module 7 收官 · 整个学习马拉松完成

## Section 1: Timeline (按 Module)
- Module 1 PEFT (3 专题)
- Module 3 造大模型 (8)
- Module 4 改大模型 (7)
- Module 5 用大模型 (7)
- Module 6 评测安全 (7)
- Module 7 Agent 应用层 (7)

## Section 2: 6-ckpt zoo (Module 6 v1 5 个 + 新 agent)
- vanilla / lora / dpo / r1_tiny / phi_tiny + dra_v1

## Section 3: All Capstones
- Module 3 small-model graduation
- Module 4 五线综合
- Module 5 serving graduation
- Module 6 (mini-HELM / mini-Arena / 红队 / 防御 / Portfolio v1)
- Module 7 (DRA / τ-bench / Portfolio v2) ⭐ ⭐ ⭐

## Section 4: Selection trees
- Bench 选型 (Module 6)
- Framework 选型 (Module 7)
- Inference engine 选型 (Module 5)
- RL 算法选型 (Module 4)

## Section 5: 6 大画像
- 造模型 (Module 3)
- 改模型 (Modules 1+4)
- 用模型 (Module 5)
- 评模型 (Module 6)
- 守模型 (Module 6)
- 造 agent (Module 7) ⭐ NEW

## Section 6: What I can do (cover letter)
- 简历用 paragraph
- LinkedIn post
- GitHub repo README
- 面试讲解 1 分钟

## Section 7: Career paths
- LLM infra engineer
- AI application engineer
- ML research engineer
- AI safety engineer
- Product manager (AI)
```

## Generator 实现

```python
def write_portfolio_v2(path: str) -> str:
    md = []
    md.append(HEADER)
    md.append(TIMELINE_39_TOPICS)
    md.append(CKPT_ZOO_TABLE)
    md.append(CAPSTONES_SUMMARY)
    md.append(SELECTION_TREES)
    md.append(SIX_PROFILES)
    md.append(WHAT_I_CAN_DO)
    md.append(CAREER_PATHS)

    with open(path, "w") as f:
        f.write("\n".join(md))
    return path
```

## 用法

| 用法 | 形式 |
|------|------|
| Resume attachment | 直接 PDF |
| GitHub README | repo 主页 |
| LinkedIn post | 摘要 + 链接 |
| Cover letter | "What I can do" 段 |
| 面试讲解 | 1 分钟 per section |

## 退出条件

- 能列 7 section
- 知道 v1 vs v2 差异
- 能讲 6 大画像

## 一句话

> Portfolio v2 = 39 专题 × 7 section × 6 画像 — 出门作品集 + LinkedIn + 面试讲解 三合一。
