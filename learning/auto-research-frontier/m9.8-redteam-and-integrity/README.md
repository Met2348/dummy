# 9.8 · 批判·安全·科研诚信（红队 + 守卫，毕业 Capstone）⭐⭐

> M9 系列的收口。拿一个 mini-AI-Scientist 当靶子，**红队它**——找出它会在哪刷分/幻觉/换数据，
> 再给它加**诚信守卫**。毕业标准：你的 AI-Scientist **既跑得通，又扛得住自己的红队**。

## 一句话结论（本模块跑出来的毕业判定）

```
天真评审被骗的攻击数：4/4 —— 自报指标分不出真假。
诚实报告通过全部守卫：True；每种攻击都被对应守卫抓住：True。
>> 通过：这个 mini-scientist 既跑得通，又扛得住自己的红队。
```

一个只信报告自评分的**天真评审，被全部 4 种造假骗过**；而 4 个**诚信守卫**逐一戳穿它们，
同时放行诚实报告。

## 四种攻击 ↔ 四个守卫（每个守卫都是前面某一课）

| 攻击 | 怎么造假 | 守卫 | 对应的课 |
|------|---------|------|---------|
| 幻觉消融表 | 加一行从没跑过的"神奇配置 0.99" | **provenance**：每个数字必须可追溯到真实运行 | 9.2 接地 |
| 偷换数据集 | 在 easy 上跑却声称 hard | **dataset**：sha256 指纹必须与声称的数据集相符 | 9.4 忠实 |
| 硬编码指标 | 预测没动，acc 直接写 0.99 | **metric**：从保存的预测**独立复算** | 9.6 held-out / 9.7 别信代理 |
| 刷自评 | 质量没变，自评分拉满 | **independent_review**：无视自评，独立打分 | 9.3 自偏好 / 9.5 grading-own-homework |

> **整个 M9 在这里收束成一句话**：所有自动科研的可信度，最终都压在"独立验证"这一环——
> 别信系统自己说"我做出了什么"，去**独立复算**它。

## 跑起来（防御教育用途）

```powershell
python src/run.py                          # 诚实基线 + 4 种攻击 + 毕业判定
python src/run.py --attack hardcode-metric # 单看一种攻击的守卫细节

python scripts/eric_3080ti_env_audit.py --runbook --tests `
  --modules auto-research-frontier/m9.8-redteam-and-integrity `
  --json-out $env:TEMP/m9.json --md-out $env:TEMP/m9.md
```

## 目录

```
m9.8-redteam-and-integrity/
├── runbook.yaml
├── lectures/
│   ├── 01-the-attack-surface.md       它会怎么骗你（Hidden Pitfalls 全景）
│   ├── 02-four-attacks.md             四种攻击 + 天真评审被骗
│   └── 03-guards-and-graduation.md    四个守卫 + 可复现性 + 系列收口
└── src/
    ├── run.py
    ├── integrity/
    │   ├── data.py             两数据集变体 + sha256 指纹 + 真训练
    │   ├── naive_scientist.py  天真 scientist：真跑 + 可注入 4 种造假
    │   └── guards.py           四个诚信守卫 + audit
    └── tests/test_redteam.py   8 测试：诚实过全守卫、每种攻击各被抓、毕业判定
```

## Hands-on / 毕业答辩（轮到你）

1. **造第五种攻击**：想一个本模块还没覆盖的造假（如"只报最好的 5 个种子、丢掉差的"——
   cherry-picking）。它能骗过现有 4 个守卫吗？要加第 5 个守卫来抓它吗？
2. **把守卫接到 9.5**：真正的毕业作业——给 9.5 的 `mini_ai_scientist` 挂上本模块的守卫
   （独立复算它报告里的数字、给它的数据集加指纹），让那个"裁判=选手"的 `review` 不再说了算。
3. **写一份诚实报告**：用一段话回答老师的问题——"research 被 Agent 接管到什么程度？"
   用你这 7 个模块跑出来的证据（自主性反相关可信度、ideation-execution gap、reward hacking……）支撑。

## 这就是系列毕业 Capstone 的诚信半边

> 配合 9.5（端到端跑通）+ 9.6（复现评测）+ 本模块（红队守卫），
> 你就有了一个**可信的 mini AI-Scientist**：既能跑、又有独立验证、还扛得住自己的造假倾向。
> 这是你对"2026，research 被 Agent 接管到什么程度"的**亲手答案**。

## 桥接

- **red-team-jailbreak · safety-defense · eval-graduation**（M6 尾三件套在这里收口）。
- 合流：9.2 接地 · 9.3 自偏好 · 9.4 忠实 · 9.6 独立验证 · 9.7 reward hacking —— 全部汇入这四个守卫。
