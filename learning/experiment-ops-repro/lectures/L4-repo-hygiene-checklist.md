# L4 · repo 卫生 + 复现 checklist: 让别人 (和半年后的你) 一键复现

> 30-min lecture · 目标: 学会把一个研究项目的仓库整理到「别人 clone 下来能跑通、能复现」的程度; 掌握一张可勾选的复现 checklist, 作为投稿前的硬关卡。

---

## 0. 从「单个 run 留痕」到「整个项目可复现」

L3 让每个 run 留了痕。但当你投稿、开源代码时, 考验升级为: **一个陌生人 `git clone` 你的 repo, 能不能在自己机器上跑出你论文里的数?** 这要求整个**仓库**是可复现的, 不只是单个 run。

> reproducible 不是一个开关, 是一组工程纪律的总和: 清晰的结构 + 锁定的依赖 + 数据的可获取性 + 固定的随机 + 一键复现的入口 + 完整的记录。少任何一环, 复现就断在那。

---

## 1. repo 结构: 一眼能找到东西

一个研究项目的健康目录 (你的 `learning/` 专题其实已经接近):

```
   my-research-project/
   ├── README.md            # 项目是什么 + 如何复现 (最重要的一页)
   ├── requirements.txt     # 或 environment.yml / pyproject + lockfile
   ├── configs/             # 所有实验配置 (L2), 一份 config 定义一次实验
   ├── src/                 # 可复用代码 (模型/训练/评测)
   ├── scripts/             # 一键复现脚本 (reproduce_table2.sh ...)
   ├── data/                # 数据 (或数据获取脚本 + 版本/哈希说明)
   ├── runs/                # 实验记录 (L3 的 jsonl / wandb 链接)
   ├── results/             # figures + 表格 (由 scripts 生成, 可重建)
   └── .gitignore           # 排除大文件/缓存/密钥
```

几条卫生原则:
- **代码与产物分离**: `src/` 是输入, `results/` 是输出; 产物应能**由代码重建**, 而不是手工攒的。
- **配置与代码分离**: 超参在 `configs/`, 不在代码里 (L2)。
- **大文件不进 git**: 模型权重、数据集用 `.gitignore` 排除; 必要时用 git-LFS / DVC / 外部存储 + 下载脚本。
- **永远不提交密钥**: API key / token 进 `.gitignore`, 用环境变量。(这条出事是真出事 —— 泄露的 key 会被人扫到盗用。)

---

## 2. 锁依赖: 别让「在我机器上能跑」毁掉复现

「在我机器上能跑」是复现的头号杀手。原因: 你的 `requirements.txt` 写 `torch`, 别人装到的是新版本, 行为变了, 结果对不上 (L1 的环境漂移)。

依赖管理的三个层次 (越往下越严):
1. **requirements.txt (松)**: 列出包名。最低限度, 但不锁版本 → 不够。
2. **锁版本 (紧)**: `torch==2.3.1` 写死每个直接依赖的版本。够大多数情况。
3. **lockfile (最紧)**: `pip freeze > requirements.lock` / `uv.lock` / `conda env export` —— 锁住**整棵依赖树** (包括间接依赖)。投稿/开源应做到这层。

> 实操: 平时开发用松的 `requirements.txt`; **投稿前生成一份 lockfile** 附在 repo 里, 并在 README 写明「精确复现请用 lockfile」。`exp_tracker` 记的 env 指纹是这个的轻量替身 —— 它至少让你**知道**当时用的什么版本, 即使没锁。

---

## 3. 数据版本: 结论的另一半地基

代码固定了, 数据变了, 结论一样崩。数据可复现的要素:
- **数据来源 + 版本**: 不是「hh-rlhf」, 而是「hh-rlhf, HuggingFace, revision abc123」。
- **数据哈希**: 对关键文件算个哈希 (md5/sha256), 验证别人拿到的和你一样。
- **预处理的确定性**: 清洗/划分/过滤的代码也要进 repo, 且用固定 seed (划分 train/test 别用随机不固定的)。
- **不可公开的数据**: 给出获取说明 + 一个小的**可公开样例**让别人跑通流程。

> config 里那个 `dataset: "hh-rlhf@v2"` 字段 (L2) 就是数据版本的最小记录。`repro_check.py` 会专门检查它在不在。

---

## 4. 复现 checklist: 投稿前的硬关卡

ML 社区 (NeurIPS/ICML 等) 有正式的 reproducibility checklist。把它内化成你**投稿前必过**的一关。本专题 `repro_check.py` 把核心 6 项代码化:

```
   可复现性体检 (repro_check.py 的 6 项)
   ☐ 1. seed 已固定          (确定性的钥匙, L1)
   ☐ 2. git_sha 已记 (非dirty) (哪版代码, L3)
   ☐ 3. config 完整          (跑了什么, L2)
   ☐ 4. env 指纹/lockfile     (什么环境, L1/L4)
   ☐ 5. 数据版本/哈希已记      (什么数据, L4)
   ☐ 6. metrics 结构化记录     (结果留痕, L3)
```

扩展到项目级, 再加几项 (checklist 模板 `templates/repro-checklist.md` 全列):
- [ ] README 有明确的「如何复现」段落
- [ ] 有一键复现脚本 (`scripts/reproduce_*.sh`), 不是「跟着论文手动跑」
- [ ] 关键结果有多种子 + error bar (接 9.4-L5)
- [ ] 报告了算力 (GPU 型号/小时数), 让别人估成本
- [ ] limitations / 已知不可复现点诚实写出 (如「GPU 上有 ±0.3 波动」)

> 最后一条体现成熟度: **诚实地说出「这部分我也无法完全复现 (如 GPU 非确定性导致 ±0.3)」, 比假装一切完美更可信。** 这也呼应 9.3 的批判式阅读 —— 你希望别人怎么诚实地对你, 就怎么对读者。

---

## 5. 给博0 的现实建议

1. **复现纪律是「平时」养的, 不是投稿前补的。** 投稿前一周才想起来整理 repo, 会发现三个月前的某个结果已经无法复现 (代码/数据/环境都变了)。从第一个实验就用 config + tracker。
2. **你的 `learning/` 专题已经是好榜样。** 它们有 README + environment + src + 可跑 notebook —— 这套结构直接就是研究项目的复现骨架。把研究项目也按这个套路组织。
3. **`repro_check` 当 git pre-commit / 投稿 gate。** 把它接进你的工作流, 每个进论文的结果都先过体检。

---

## 6. 本讲小结 (9.5 收口) + 通往 9.6

- 从「单 run 留痕」升到「整个 repo 可复现」: 结构清晰 / 锁依赖 / 数据版本 / seed / 一键脚本 / 完整记录。
- **锁依赖**: 投稿前出 lockfile, 治「在我机器上能跑」。
- **数据版本**: 来源+版本+哈希+确定性预处理; 数据变结论崩。
- **复现 checklist** 是投稿硬关卡: seed/git/config/env/data/metrics 六项 + 项目级扩展; 诚实写出不可复现点。
- 复现纪律平时养, 不是投稿前补。

> **下一专题 9.6「research-figures」**: 实验设计对了 (9.4)、可靠跑出并留痕了 (9.5), 接下来要把结果**讲给人看**。9.6 教你做出版级科研图 —— 把你 9.4 那张交互效应图、9.5 的对比, 升级到能进顶会论文的水准。

**动手**: 去 `N2-reproducibility-audit.ipynb`, 用 `repro_check` 给两条实验记录 (一条规范、一条潦草) 做体检, 再亲手验证一次「同 config+seed 两次跑结果一致 / 不固定 seed 则不一致」的确定性演示。然后拿 checklist 审视你自己某个复现专题, 看缺哪几项。
