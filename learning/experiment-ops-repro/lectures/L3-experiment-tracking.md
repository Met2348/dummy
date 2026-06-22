# L3 · 实验追踪: 让几十个 run 可查、可对比、可复现

> 30-min lecture · 目标: 学会用实验追踪 (wandb / mlflow 的思路 + 本专题的本地实现) 自动记录每个 run 的 config + metrics + 环境指纹, 把 9.4 的消融矩阵从「一堆散落的输出」变成「可查询的数据库」。

---

## 0. 病根: 实验记在脑子里和截图里

新手记实验的方式: 跑完看一眼终端输出, 记在脑子里 / 截个图 / 抄进一个 Excel。跑到第 30 个 run 时:

- 「`beta=0.1 noise=0.4` 那次的 win-rate 是多少来着?」—— 翻不到了。
- 「这两张图哪张是旧版代码跑的?」—— 不知道。
- 「上周最好的那组配置是什么?」—— 忘了。

> **实验追踪 = 给每个 run 自动建一条结构化档案**, 包含三件套: **config (跑了什么) + metrics (跑出什么) + 指纹 (在什么代码/环境跑的)**。有了它, 上面每个问题都是一次查询, 而不是一次考古。

---

## 1. 一条合格的实验记录: 三件套

```
   一个 run 的完整档案
   ┌─────────────────────────────────────────────┐
   │ ① config   method/lr/beta/noise/seed/dataset │ ← L2 (跑了什么)
   │ ② metrics  win_rate / loss / ...             │ ← 结果 (跑出什么)
   │ ③ 指纹      git_sha / python+包版本 / 时间戳   │ ← L1+L4 (在什么环境跑的)
   └─────────────────────────────────────────────┘
```

第 ③ 件「指纹」是新手最常漏、却对复现最关键的:
- **git_sha**: 这个结果是**哪一版代码**跑的。没有它, 半年后代码改了 20 次, 你无法知道当时跑的是哪版逻辑。`exp_tracker` 还会标 `-dirty` (有未提交改动 → 复现性打折)。
- **包版本 / python 版本**: 换了 torch 版本结果会漂 (L1)。记下来才知道环境是否一致。
- **时间戳**: 排序、对应实验日志。

> 本专题 `src/exp_tracker.py` 是一个 80 行的本地追踪器, 接口和 wandb 几乎一样 (`init / log / finish`), 但把这三件套**摊开给你看**。理解了「一条记录该有什么」, 你换 wandb 只是换 API。

---

## 2. 工具选型: wandb / mlflow / tensorboard / 本地

| 工具 | 形态 | 优点 | 适用 |
|---|---|---|---|
| **Weights & Biases (wandb)** | 云 dashboard | 强大、协作好、扫参可视化、社区标准 | NLP/LLM 组主流, 推荐学 |
| **MLflow** | 自托管 | 开源、可私有部署、含模型注册 | 看重数据私有/自托管 |
| **TensorBoard** | 本地 | 轻、看曲线方便 | 单机看 loss 曲线 |
| **本地 jsonl (本专题)** | 一行一 run | 零依赖、可 grep、可进 git、十年后打得开 | 个人小规模、理解原理 |

> 推荐路线: **理解原理用本专题的本地版** → **日常用 wandb** (业界事实标准, 找实习/合作都用它)。本地 jsonl 也别小看: 对个人研究, 一个能进 git diff、十年后还打得开的纯文本记录, 常比云 dashboard 更耐用。两者不冲突, wandb 也能导出。

---

## 3. 把追踪嵌进实验循环

追踪的正确姿势是**嵌进你的实验循环**, 让每个 run 自动留痕, 而不是事后手动记:

```python
import exp_tracker as et

for method in ["DPO", "Robust-DPO"]:
    for noise in [0.0, 0.2, 0.4]:
        for seed in range(8):
            config = {"method": method, "noise": noise, "seed": seed,
                      "dataset": "hh-rlhf@v2"}
            run = et.init(project="dpo-noise", config=config)   # 自动盖 git/env 指纹
            win = run_experiment(method, noise, seed)           # ← 9.4 的实验
            run.log({"win_rate": win})
            run.finish()                                         # 落盘成一行 jsonl
```

跑完, 整个 9.4 消融矩阵 (48 个 run) 就变成一个 48 行的 jsonl, 每行自带 config+metrics+指纹。然后:

```python
import pandas as pd
runs = et.load_runs("dpo-noise")
df = pd.json_normalize(runs)          # 直接变 DataFrame
df.groupby(["config.method","config.noise"])["metrics.win_rate"].agg(["mean","std"])
```

> **追踪和 9.4 的 aggregate 在这里合流**: 9.4 教你怎么聚合分析 (mean/std/显著性), 9.5 教你怎么把原始 run **可靠地记下来**给那个分析用。设计 (9.4) + 留痕 (9.5) = 可信的结果。

---

## 4. run 的组织: 别让 30 个 run 变成一锅粥

记下来还不够, 要能**组织和检索**:
- **project**: 一个研究问题一个 project (如 `dpo-noise-robustness`)。
- **命名/分组**: 用 config 的关键字段当 run 名 (如 `Robust-DPO_n0.4_s3`), 一眼可辨。
- **tags**: 给 run 打标签 (`mve` / `final` / `debug`), 方便筛掉调试 run。
- **筛掉脏 run**: `-dirty` 的 run (代码未提交) 标出来, 写论文时只采用 clean run。

> 一个好习惯: **跑正式结果前先 commit 代码** (让 git_sha 是 clean 的)。`-dirty` 的 run 适合探索, 但进论文的数应该来自一个**确定的、已提交的代码版本** —— 否则你无法回答审稿人「这个数是哪版代码跑的」。

---

## 5. 本讲小结 + 通往 L4

- 实验追踪 = 给每个 run 自动建结构化档案: **config + metrics + 指纹 (git_sha/env/时间)**。
- 指纹是新手最常漏、对复现最关键的一件; `git_sha` 的 `-dirty` 标记尤其要注意。
- 工具: wandb (业界标准, 推荐) / mlflow / tensorboard / 本地 jsonl (理解原理 + 耐用)。
- 追踪要**嵌进实验循环**自动留痕; 跑完一个 `load_runs` + pandas 就能接 9.4 的聚合分析。
- 组织: 按 project 分、用 config 命名、打 tag、正式结果先 commit 让 git 干净。

> **下一讲 L4「repo 卫生 + 复现 checklist」**: 单个 run 留痕了, 但整个项目仓库要让别人 (和半年后的你) 能一键复现, 还需要: 清晰的目录结构、锁定的依赖、数据版本、一键复现脚本。L4 给你一张可勾选的复现 checklist。

**动手**: 去 `N1-config-and-tracking.ipynb`, 用 `exp_tracker` 把 9.4 的消融矩阵完整跑一遍并留痕, 然后用 pandas 查询「高噪声下哪个方法最好」—— 体会「可查询的实验数据库」和「一锅粥」的区别。
