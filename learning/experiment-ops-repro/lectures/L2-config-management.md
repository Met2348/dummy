# L2 · 配置管理: 一个 run = 一份 config + 一个 seed

> 30-min lecture · 目标: 学会 **config as code** —— 把所有超参从代码里抽出来集中管理, 让每次实验都由一份可保存、可 diff、可复现的配置定义。这是 Reproducible 的地基。

---

## 0. 病根: 硬编码超参

新手代码长这样:

```python
model = train(lr=1e-4, batch=32, beta=0.1, epochs=3)   # 数字散落各处
# ...另一个文件里...
eval_on(testset, threshold=0.5)
```

为什么这是灾难:
- **半年后你不知道当时跑的是什么。** 那个 `beta=0.1` 是哪次实验留下的? 你改过几次?
- **没法系统扫参。** 想试 `beta ∈ {0.05, 0.1, 0.5}` 得手动改代码跑三次, 还容易忘了改回来。
- **没法复现。** 你报告的结果对应哪组超参? 代码现在的值还是当时的值? 无人知晓。

> **核心原则: 超参是数据, 不是代码。** 把它们从代码里全部抽出来, 集中到一份**配置 (config)** 里。代码只读 config, 不写死任何超参。这样「跑了什么」就等于「哪份 config + 哪个 seed」, 一句话说清、可保存、可复现。

---

## 1. config as code: 一份配置定义一次实验

把上面的代码改造成:

```python
# config (一份 yaml / 一个 dataclass)
config = {
    "method": "Robust-DPO",
    "lr": 1e-4, "batch": 32, "beta": 0.1, "epochs": 3,
    "noise": 0.4,
    "dataset": "hh-rlhf@v2",     # ← 数据也有版本 (L4)
    "seed": 3,
}
# 代码只读 config, 不写死任何超参
model = train(config)
```

好处立刻显现:
- **一个 run ≡ 一份 config**: 保存这份 config (进 git / 进实验记录), 就完整定义了这次实验。
- **扫参 = 改 config**: 想扫 `beta`, 程序里循环替换 config 的一个字段, 代码一行不动。
- **复现 = 加载旧 config**: 把当时那份 config 喂回 `train`, 就重建了当时的设置。

> 9.4 的 `ablation_grid` 本质就在做这件事: 它遍历 `(method, noise, seed)` 的组合, 每个组合是一份 config, 喂给 `run_experiment`。**消融矩阵 = 一组 config 的笛卡尔积。** 9.4 和 9.5 在这里无缝接上。

---

## 2. config 该放哪: 三种载体

| 载体 | 形态 | 优点 | 适用 |
|---|---|---|---|
| **YAML / JSON 文件** | 一份 `config.yaml` | 人读友好、可进 git、可 diff | 大多数情况, 推荐起步 |
| **dataclass (Python)** | `@dataclass class Config` | 有类型检查、IDE 补全、有默认值 | 代码内强类型管理 |
| **Hydra / OmegaConf** | 分层 yaml + 命令行覆盖 | 组合配置、命令行扫参、强大 | 实验规模大时, 工业标准 |

> 给博0 的路线: **先用 dataclass 或一份 yaml**, 把超参集中起来 (这一步就解决 80% 的问题)。等实验多到需要「组合配置 + 命令行扫参」时, 再上 **Hydra** (NLP/LLM 组的事实标准, 值得学)。别一开始就上重武器, 也别一直手动改代码。

一个 dataclass config 的样子 (类型 + 默认值 + 自带 seed):

```python
from dataclasses import dataclass, asdict

@dataclass
class DPOConfig:
    method: str = "DPO"
    lr: float = 1e-4
    beta: float = 0.1
    noise: float = 0.0
    dataset: str = "hh-rlhf@v2"
    seed: int = 0

    def to_dict(self): return asdict(self)   # 方便存进实验记录 (L3)
```

---

## 3. config 的纪律: 三条铁律

1. **没有「魔法数字」漏在代码里。** 任何会影响结果的数, 都进 config。自检: 「改这个数会改变结果吗?」会 → 进 config。
2. **config 必须含 seed 和 data 引用。** seed 是确定性的钥匙 (L1); data 引用 (名称+版本+哈希) 是复现的另一半 (L4)。少了任何一个, config 都不完整。
3. **跑过的 config 要存档, 不可事后覆盖。** 每次 run 把当时的 config **快照**进实验记录 (L3 的 tracker 自动做)。改 config 重跑 = 新 run, 不是覆盖旧 run。

> 第 3 条最容易违反: 你改了 config 重跑, 旧 config 就被覆盖了, 旧结果对应的设置永远丢失。**正确做法是「config 不可变, 每跑一次存一份快照」** —— 这正是 L3 实验追踪要解决的。

---

## 4. 本讲小结 + 通往 L3

- **超参是数据不是代码**: 用 **config as code** 把超参全部抽出集中管理。
- 一个 run ≡ 一份 config + 一个 seed; **扫参 = 改 config, 复现 = 加载旧 config**。
- 载体三选一: yaml (起步) / dataclass (强类型) / Hydra (规模大时的工业标准)。
- 三铁律: 无魔法数字 / config 含 seed+data / config 不可变且每次存快照。

> **下一讲 L3「实验追踪」**: config 定义了「跑什么」, 但你还需要把「跑出什么 (metrics)」+「在什么环境跑的 (git/包版本)」一起记下来, 而且要能查询、对比几十个 run。L3 教你用追踪器 (wandb 思路 + 本专题的本地版) 把 9.4 的整个消融矩阵自动留痕。

**动手**: 把你某个复现专题里散落的超参, 抽进一个 dataclass config。然后去 N1, 用这个 config 思路驱动 9.4 的消融矩阵, 并把每个 run 的 config 自动记进追踪器。
